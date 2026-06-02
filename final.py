#!/usr/bin/env python3
"""CureQ Microscopy Analysis Suite — final_5.py — final_4.py
DL aggregate detection (10-fold U-Net ensemble)
Per-cell aggregate quantification across 4 sub-cellular regions"""
import os, sys, json, traceback, warnings, datetime, csv, time
warnings.filterwarnings("ignore")
import numpy as np
from scipy.ndimage import gaussian_filter, label as scipy_label, binary_dilation
from scipy.stats import pearsonr, mannwhitneyu
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MplPath
import tifffile
from PIL import Image as PILImage, ImageEnhance
try:
    from readlif.reader import LifFile as _LifFile
    HAS_READLIF = True
except ImportError:
    HAS_READLIF = False
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes, ball, binary_opening
from skimage.draw import polygon as sk_polygon
from skimage.measure import label as sk_label, regionprops
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu as sk_otsu
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
try:
    from cellpose import models as cp_models, io as cp_io
    cp_io.logger_setup()
    CELLPOSE_AVAILABLE = True
except Exception:
    CELLPOSE_AVAILABLE = False
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QTabWidget, QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QGroupBox, QScrollArea, QFileDialog, QMessageBox, QProgressBar, QStatusBar, QToolBar, QAction, QLineEdit, QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QFrame, QDialog, QDialogButtonBox, QButtonGroup, QRadioButton, QColorDialog, QToolTip)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QIntValidator, QPixmap, QIcon, QPainter, QPen, QCursor


# ── Simplified theme config ────────────────────────────────────────────────
# accent      = tab highlight + active elements
# background  = main window + widget bg
# toolbar     = top toolbar bg
# viewer_tb   = matplotlib navigation toolbar bg
APP_THEME = {
    "accent":     "#e0357a",
    "background": "#1c1c32",
    "toolbar":    "#1a1d2e",
    "viewer_tb":  "#2a2f45",
}

def _derive(theme):
    """Derive full color set from the 4-key theme."""
    acc = theme["accent"]
    bg  = theme["background"]
    tb  = theme["toolbar"]
    # Parse accent for lighter variant
    try:
        r = int(acc[1:3],16); g = int(acc[3:5],16); b = int(acc[5:7],16)
        acc_lit = "#{:02x}{:02x}{:02x}".format(min(255,r+40),min(255,g+40),min(255,b+40))
    except Exception:
        acc_lit = acc
    # bg variants derived from background
    try:
        r = int(bg[1:3],16); g = int(bg[3:5],16); b = int(bg[5:7],16)
        bg_mid  = "#{:02x}{:02x}{:02x}".format(min(255,r+10),min(255,g+10),min(255,b+16))
        bg_low  = "#{:02x}{:02x}{:02x}".format(max(0,r-4), max(0,g-4), max(0,b-2))
        bg_dark = "#{:02x}{:02x}{:02x}".format(max(0,r-12),max(0,g-12),max(0,b-8))
    except Exception:
        bg_mid = bg_low = bg_dark = bg
    return {
        "accent": acc, "accent_lit": acc_lit,
        "bg_deep": bg, "bg_mid": bg_mid, "bg_low": bg_low, "bg_dark": bg_dark,
        "toolbar": tb,
        "fg": "#dcdcf0", "fg_dim": "#8888b0", "border": "#2a2a50",
    }


def build_qss(theme_raw=None):
    """Build full QSS. Accepts either a raw 4-key theme dict or a pre-derived dict."""
    if theme_raw is None: theme_raw = APP_THEME
    # If it's the 4-key format, derive full palette; otherwise use as-is
    t = _derive(theme_raw) if "background" in theme_raw else theme_raw
    tb_bg = theme_raw.get("toolbar", t["bg_dark"])
    return f"""
QMainWindow, QDialog {{ background: {t["bg_deep"]}; }}
QWidget {{ background: {t["bg_mid"]}; color: {t["fg"]}; font-size: 12px; font-family: 'Segoe UI', 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif; }}
QTabWidget::pane {{ border: 1px solid {t["border"]}; background: {t["bg_low"]}; }}
QTabBar {{ background: {t["bg_low"]}; }}
QTabBar::tab {{ background: {t["bg_dark"]}; color: {t["fg_dim"]}; padding: 8px 16px; min-width: 80px; border: 1px solid {t["border"]}; border-bottom: none; margin-right: 2px; border-radius: 4px 4px 0 0; font-size: 12px; font-weight: 500; }}
QTabBar::tab:selected {{ background: {t["accent"]}; color: #ffffff; border: 1px solid {t["accent_lit"]}; border-bottom: none; font-weight: 700; }}
QTabBar::tab:hover:!selected {{ background: #28284a; color: #d0d0f0; }}
QScrollBar:vertical {{ background: {t["bg_low"]}; width: 8px; border-radius: 4px; }}
QScrollBar::handle:vertical {{ background: #3a3a70; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: {t["bg_low"]}; height: 8px; border-radius: 4px; }}
QScrollBar::handle:horizontal {{ background: #3a3a70; border-radius: 4px; min-width: 24px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QProgressBar {{ background: {t["bg_low"]}; border: 1px solid #3a3a60; border-radius: 5px; color: #e0e0ff; text-align: center; height: 16px; }}
QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #4040a8,stop:1 #7070e8); border-radius: 4px; }}
QPushButton {{ background: #1e1e3c; color: #c8c8e8; border: 1px solid #3c3c68; border-radius: 5px; padding: 6px 16px; min-height: 26px; font-size: 12px; }}
QPushButton:hover {{ background: #2a2a58; border-color: #5050a0; color: #ffffff; }}
QPushButton:pressed {{ background: #141424; }}
QPushButton:disabled {{ color: #404060; background: #141428; border-color: #202040; }}
QComboBox {{ background: {t["bg_low"]}; color: #d8d8f0; border: 1px solid #3a3a65; border-radius: 4px; padding: 3px 8px; min-height: 24px; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{ background: {t["bg_low"]}; color: {t["fg"]}; selection-background-color: {t["accent"]}; selection-color: #fff; border: 1px solid {t["border"]}; }}
QSpinBox, QDoubleSpinBox {{ background: {t["bg_low"]}; color: #d8d8f0; border: 1px solid #3a3a65; border-radius: 4px; padding: 2px 6px; min-height: 22px; }}
QLineEdit {{ background: {t["bg_low"]}; color: #d8d8f0; border: 1px solid #3a3a65; border-radius: 4px; padding: 3px 6px; min-height: 22px; }}
QGroupBox {{ border: 1px solid #3a3a60; border-radius: 6px; margin-top: 8px; padding-top: 6px; font-weight: 600; color: #a0a0d0; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; padding: 0 6px; color: #b0b0e0; }}
QCheckBox {{ color: #c8c8e8; spacing: 6px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid #4a4a80; border-radius: 3px; background: {t["bg_low"]}; }}
QCheckBox::indicator:checked {{ background: {t["accent"]}; border-color: {t["accent_lit"]}; }}
QLabel {{ color: {t["fg"]}; background: transparent; }}
QTableWidget {{ background: {t["bg_low"]}; color: {t["fg"]}; gridline-color: #2a2a50; border: 1px solid {t["border"]}; }}
QTableWidget QHeaderView::section {{ background: #1a1a30; color: #a0a0d0; border: 1px solid #2a2a50; padding: 4px; font-weight: 600; }}
QToolBar {{ background: {tb_bg}; border: none; spacing: 4px; padding: 2px 6px; min-height: 32px; }}
QToolBar QToolButton {{ background: transparent; color: #ffffff; border: none; border-radius: 4px; padding: 4px 8px; font-size: 16px; font-weight: bold; min-width: 32px; min-height: 32px; }}
QToolBar QToolButton:hover {{ background: rgba(255,255,255,0.20); }}
QToolBar QToolButton:pressed {{ background: rgba(0,0,0,0.20); }}
QMenuBar {{ background: {t["bg_deep"]}; color: {t["fg"]}; }}
QMenuBar::item:selected {{ background: {t["accent"]}; color: #fff; }}
QMenu {{ background: {t["bg_low"]}; color: {t["fg"]}; border: 1px solid {t["border"]}; }}
QMenu::item:selected {{ background: {t["accent"]}; color: #fff; }}
QStatusBar {{ background: {t["bg_deep"]}; color: #8888b0; font-size: 11px; }}
QSplitter::handle {{ background: {t["border"]}; }}
QTextEdit {{ background: {t["bg_low"]}; color: {t["fg"]}; border: 1px solid {t["border"]}; border-radius: 4px; }}
"""

def apply_theme(window, new_theme):
    """Update APP_THEME with new_theme dict and re-apply stylesheet to whole app."""
    APP_THEME.update(new_theme)
    qss = build_qss(APP_THEME)
    window.setStyleSheet(qss)
    QApplication.instance().setStyleSheet(qss)
    # Re-apply viewer toolbars (matplotlib NavigationToolbar instances need explicit update)
    vtb_bg = APP_THEME.get("viewer_tb", "#2a2f45")
    vtb_qss = (f"QToolBar{{background:{vtb_bg};border:none;}}"
               f"QToolButton{{color:#ffffff;background:transparent;font-size:14px;}}"
               f"QToolButton:hover{{background:rgba(255,255,255,0.15);}}")
    for w in QApplication.instance().allWidgets():
        if isinstance(w, NavigationToolbar):
            w.setStyleSheet(vtb_qss)

BTN_RUN    = "QPushButton{background:#1a3a1a;color:#80ff80;border:1px solid #3a7a3a;border-radius:5px;padding:6px 16px;min-height:28px;font-weight:600;}QPushButton:hover{background:#234a23;border-color:#50a050;}QPushButton:disabled{color:#406040;background:#111a11;border-color:#2a4a2a;}"
BTN_LOAD   = "QPushButton{background:#1a2a3a;color:#80c0ff;border:1px solid #2a5a7a;border-radius:5px;padding:6px 16px;min-height:28px;font-weight:600;}QPushButton:hover{background:#233a50;}QPushButton:disabled{color:#406080;}"
BTN_WARN   = "QPushButton{background:#3a1a1a;color:#ff8080;border:1px solid #7a3a3a;border-radius:5px;padding:6px 16px;min-height:28px;font-weight:600;}QPushButton:hover{background:#4a2020;}"
BTN_PURPLE = "QPushButton{background:#2a1a3a;color:#c080ff;border:1px solid #5a3a7a;border-radius:5px;padding:6px 16px;min-height:28px;font-weight:600;}QPushButton:hover{background:#3a2a50;}"
BTN_HELP   = "QPushButton{background:#1a1a2e;color:#7070b0;border:1px solid #3a3a60;border-radius:4px;padding:2px 8px;font-size:11px;}QPushButton:hover{background:#222240;color:#9090d0;}"

BIOLOGICAL_LABELS = ["—", "Nucleus", "HA (mHTT)", "CCT1", "A11", "Other"]

LABEL_DEFAULT_COLORS = {
    "Nucleus":   "blue",
    "HA (mHTT)": "red",
    "CCT1":      "green",
    "A11":       "magenta",
    "Other":     "gray",
    "—":         "gray",
}


def _viewer_tb_qss():
    """Returns stylesheet for matplotlib NavigationToolbar — themed, high-contrast."""
    bg = APP_THEME.get("viewer_tb", "#2a2f45")
    return (f"QToolBar{{background:{bg};border:none;padding:2px;}}"
            "QToolButton{color:#ffffff;background:transparent;font-size:13px;"
            "min-width:24px;min-height:24px;border-radius:3px;}"
            "QToolButton:hover{background:rgba(255,255,255,0.18);}"
            "QToolButton:pressed{background:rgba(0,0,0,0.20);}")


def _require_preprocessed(state, parent):
    if not state.has_preprocessed():
        QMessageBox.warning(parent, "No image", "Load an image in Image Viewer, then optionally run Preprocessing.")
        return False
    return True

def _populate_channel_combo(combo, state, preferred_label="", active_only=False):
    combo.blockSignals(True)
    combo.clear()
    pool = ([m for m in state.channel_mappings if m.get("enabled",True)] if active_only else state.channel_mappings)
    if not pool:
        combo.addItem("(no channels)")
        combo.blockSignals(False)
        return
    best_idx = 0
    for i, m in enumerate(pool):
        bio  = m.get("biological_label","—") or "—"
        name = m.get("channel_name","")
        col  = m.get("color","gray")
        cidx = m["data_channel_index"]
        off  = "  [off]" if not m.get("enabled",True) else ""
        lbl  = bio if bio not in ("—","") else f"Ch {cidx}"
        combo.addItem(f"{lbl}  ({col}) — {name or f'Ch {cidx}'}{off}")
        if preferred_label and bio == preferred_label:
            best_idx = i
    combo.setCurrentIndex(best_idx)
    combo.blockSignals(False)

def _channel_index_from_combo(combo, state, active_only=False):
    pool = ([m for m in state.channel_mappings if m.get("enabled",True)] if active_only else state.channel_mappings)
    i = combo.currentIndex()
    return pool[i]["data_channel_index"] if 0 <= i < len(pool) else None

def _label_from_combo(combo, state, active_only=False):
    pool = ([m for m in state.channel_mappings if m.get("enabled",True)] if active_only else state.channel_mappings)
    i = combo.currentIndex()
    return pool[i].get("biological_label","—") if 0 <= i < len(pool) else "—"


def _populate_channel_combo_full(combo, state, preferred_label=""):
    """Populate combo with all raw channels PLUS all derived channels.
    Raw channels show their biological label; derived channels are prefixed with ★.
    Stores (kind, value) as item data: kind='raw' → channel index, kind='derived' → key.
    """
    combo.blockSignals(True)
    combo.clear()
    best_idx = 0
    i = 0
    for m in state.channel_mappings:
        bio  = m.get("biological_label", "—") or "—"
        name = m.get("channel_name", "")
        col  = m.get("color", "gray")
        cidx = m["data_channel_index"]
        lbl  = bio if bio not in ("—", "") else f"Ch {cidx}"
        combo.addItem(f"{lbl}  ({col}) — {name or f'Ch {cidx}'}")
        combo.setItemData(i, ("raw", cidx))
        if preferred_label and bio == preferred_label:
            best_idx = i
        i += 1
    for key in state.channels.get("derived", {}):
        combo.addItem(f"★ {key}  [derived]")
        combo.setItemData(i, ("derived", key))
        if preferred_label and key == preferred_label:
            best_idx = i
        i += 1
    if combo.count() == 0:
        combo.addItem("(no channels)")
    combo.setCurrentIndex(best_idx)
    combo.blockSignals(False)


def _get_combo_image(combo, state):
    """Return a 2-D float32 (H, W) max-projection for the channel selected in combo.
    Works for both raw channels and derived channels.
    """
    data = combo.currentData()
    if data is None:
        # Fallback: treat as raw channel index 0
        vol = state.preprocessed_image if state.has_preprocessed() else state.raw_image
        if vol is None:
            return None
        return vol[:, 0].max(axis=0).astype("float32")
    kind, val = data
    if kind == "raw":
        vol = state.preprocessed_image if state.has_preprocessed() else state.raw_image
        if vol is None:
            return None
        ch = min(int(val), vol.shape[1] - 1)
        return vol[:, ch].max(axis=0).astype("float32")
    else:  # derived
        d = state.get_derived_channel(val)
        if d is None:
            return None
        return (d.max(axis=0) if d.ndim == 3 else d).astype("float32")


def _cb_build_composite_for_segmentation(state):
    img = state.preprocessed_image if state.has_preprocessed() else state.raw_image
    if img is None: return None
    nz, nc, ny, nx = img.shape
    _CMAP = {"red":(1,0,0),"green":(0,1,0),"blue":(0,0,1),"cyan":(0,1,1),"magenta":(1,0,1),"yellow":(1,1,0),"white":(1,1,1),"gray":(1,1,1),"orange":(1,.5,0)}
    composite = np.zeros((ny,nx,3), dtype=np.float32)
    any_ch = False
    for m in state.channel_mappings:
        if not m.get("enabled",True): continue
        cidx = m.get("data_channel_index")
        if cidx is None or cidx >= nc: continue
        rgb   = np.array(_CMAP.get(m.get("color","gray").lower(),(1,1,1)), dtype=np.float32)
        plane = np.max(img[:,cidx], axis=0).astype(np.float32)
        lo, hi = float(np.percentile(plane,2)), float(np.percentile(plane,98))
        pn = np.clip((plane-lo)/(hi-lo+1e-9),0,1)
        composite += pn[:,:,None]*rgb[None,None,:]
        any_ch = True
    if not any_ch: return np.zeros((ny,nx), dtype=np.uint8)
    from PIL import Image as _PIL
    return np.array(_PIL.fromarray((np.clip(composite,0,1)*255).astype(np.uint8),"RGB").convert("L"), dtype=np.uint8)

def _cb_postprocess(masks, min_size=0):
    if masks is None: return None
    if min_size <= 0: return masks.astype(np.int32)
    result = masks.astype(np.int32).copy()
    for prop in regionprops(masks):
        if prop.area < min_size: result[masks==prop.label] = 0
    return result

def _detect_channels(arr):
    N = arr.shape[0]
    if N < 4: return 1
    means = [float(arr[i].mean()) for i in range(N)]
    best_k, best_score = 1, 0.0
    for k in range(2, min(N//2+1, 33)):
        if N % k: continue
        ch = [means[c::k] for c in range(k)]
        intra = sum(float(np.std(m)) for m in ch) / k
        inter = float(np.std([float(np.mean(m)) for m in ch]))
        s = inter/(intra+1e-6)
        if s > best_score: best_score, best_k = s, k
    if best_k > 1 and best_score > 5.0:
        print(f"[load] interleaving K={best_k} score={best_score:.1f}")
        return best_k
    return 1

def _to_zcyx(arr):
    arr = arr.astype("float32")
    if arr.ndim == 2: return arr[None, None]
    if arr.ndim == 5: arr = arr[0]
    if arr.ndim == 4:
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            arr = arr.transpose(1,0,2,3)
        return arr
    if arr.ndim == 3:
        k = _detect_channels(arr)
        if k > 1:
            N,Y,X = arr.shape
            return arr.reshape(N//k, k, Y, X)
        return arr[:,None]
    raise ValueError(f"Unexpected ndim={arr.ndim}")

# Central state object — all tabs share one instance
class AppState:
    def __init__(self):
        self.raw_image = None
        self.preprocessed_image = None
        self.current_channel = 0
        self.current_z_slice = 0
        self.results_dict = {}
        self.file_path = ""
        self.voxel_size_xyz = (1.0, 1.0, 1.0)
        self.aics_image = None
        self._observers = []
        self.channel_mappings = []
        self.channel_names = []
        self.coloc_results = None   # stores last colocalization analysis results in memory
        # ----
        # Central derived-channel store.  Two built-in slots: "aggregates" and "chaperones".
        # Each entry: {"mask": np.ndarray (Z,Y,X) float32,
        #              "source": str,
        #              "visible": bool}
        self.channels = {
            "raw":     {},   # mirrors channel_mappings – populated on image load
            "derived": {},   # filled by Aggregate Detection and Chaperone Segmentation
        }
        # Active slice system: None = full 3D, int = specific Z slice
        self.active_z_slice: "int | None" = None
        # Active mask IDs: deselected cells NEVER used in downstream tools
        self.active_cell_ids: dict = {
            "cell_body": set(),
            "nucleus":   set(),
        }

    def subscribe(self, cb):
        if cb not in self._observers: self._observers.append(cb)

    def notify(self, what=""):
        for cb in self._observers:
            try: cb(what)
            except Exception: pass

    def set_result(self, tool, data):
        self.results_dict[tool] = data
        self.notify("results")

    def has_image(self): return self.raw_image is not None
    def has_preprocessed(self): return self.preprocessed_image is not None

    def get_active_channels(self):
        return [m["data_channel_index"] for m in self.channel_mappings if m.get("enabled",True)]

    def get_channel_index_for_label(self, label):
        for m in self.channel_mappings:
            if m.get("enabled",True) and m.get("biological_label","—") == label:
                return m["data_channel_index"]
        return None

    def get_label_combo_items(self):
        items = []
        for m in self.channel_mappings:
            bio  = m.get("biological_label","—") or "—"
            name = m.get("channel_name","")
            col  = m.get("color","gray")
            cidx = m["data_channel_index"]
            off  = "  [off]" if not m.get("enabled",True) else ""
            lbl  = bio if bio not in ("—","") else f"Ch {cidx}"
            items.append(f"{lbl}  ({col}) — {name or f'Ch {cidx}'}{off}")
        return items

    def get_mapping_for_combo_index(self, combo_idx, active_only=False):
        pool = ([m for m in self.channel_mappings if m.get("enabled",True)] if active_only else self.channel_mappings)
        return pool[combo_idx] if 0 <= combo_idx < len(pool) else None

    # ----
    def set_derived_channel(self, key: str, mask: np.ndarray, source: str) -> None:
        """Register a segmentation result as a reusable derived channel.
        Parameters"""
        self.channels["derived"][key] = {
            "mask":    mask.astype(np.float32),
            "source":  source,
            "visible": True,
        }
        self.notify("derived_channels")

    # ----
    def get_derived_channel(self, key: str) -> np.ndarray | None:
        """Return the (Z, Y, X) mask for a derived channel, or None if absent."""
        entry = self.channels["derived"].get(key)
        return entry["mask"] if entry is not None else None

    # ----
    def has_derived_channel(self, key: str) -> bool:
        return key in self.channels["derived"]

    # ----
    def get_channel_volume_by_label(self, label: str) -> np.ndarray | None:
        """Return a (Z, Y, X) float32 volume for *any* channel identified by label.
        Resolution order:"""
        # 1. Check derived channels first
        derived_key = label.lower().replace(" ", "_")
        if self.has_derived_channel(derived_key):
            return self.get_derived_channel(derived_key)
        # Also try exact key match
        if self.has_derived_channel(label):
            return self.get_derived_channel(label)

        # 2. Fall back to raw/preprocessed image
        img = self.preprocessed_image if self.has_preprocessed() else self.raw_image
        if img is None:
            return None
        ch_idx = self.get_channel_index_for_label(label)
        if ch_idx is None:
            return None
        return img[:, ch_idx, :, :].astype(np.float32)

    def set_active_z_slice(self, z: "int | None") -> None:
        """Set the global working slice. None = full 3D mode."""
        self.active_z_slice = z
        self.notify("active_z_slice")

    def filter_mask_by_active_ids(self, mask: np.ndarray,
                                   kind: str = "cell_body") -> np.ndarray:
        """Zero-out labels NOT in active_cell_ids[kind].
        Parameters"""
        ids = self.active_cell_ids.get(kind, set())
        if not ids:
            return mask   # nothing selected yet → return unchanged
        out = mask.copy()
        all_in_mask = set(np.unique(mask)) - {0}
        for lbl in all_in_mask:
            if lbl not in ids:
                out[mask == lbl] = 0
        return out

    def apply_active_z_to_mask(self, mask: np.ndarray) -> np.ndarray:
        """Slice a 3-D mask (Z, Y, X) to active_z_slice if set.

        Returns a 2-D array when a slice is selected, or the full 3-D mask.
        """
        if self.active_z_slice is None or mask.ndim < 3:
            return mask
        z = min(int(self.active_z_slice), mask.shape[0] - 1)
        return mask[z]

    def get_working_volume(self) -> "np.ndarray | None":
        """Return the image volume to use for analysis, respecting active_z_slice.
        When a slice is active the result has shape (1, C, Y, X) so all"""
        img = self.preprocessed_image if self.has_preprocessed() else self.raw_image
        if img is None:
            return None
        if self.active_z_slice is None:
            return img
        z = min(int(self.active_z_slice), img.shape[0] - 1)
        return img[z : z + 1]   # shape (1, C, Y, X)

    # ----
    def get_all_channel_labels(self) -> list[tuple[str, str]]:
        """Return list of (label, kind) for all selectable channels.
        kind is "raw" for image channels or "derived" for segmentation masks."""
        items = []
        for m in self.channel_mappings:
            bio  = m.get("biological_label", "—") or "—"
            name = m.get("channel_name", "") or f"Ch {m['data_channel_index']}"
            display = bio if bio not in ("—", "") else name
            items.append((display, "raw"))
        for key in self.channels["derived"]:
            display = key.replace("_", " ").title()
            items.append((display, "derived"))
        return items

    def get_display_slice(self, channel=None, z=None, projection="max"):
        img = self.preprocessed_image if self.has_preprocessed() else self.raw_image
        if img is None: return None
        c = min(channel if channel is not None else self.current_channel, img.shape[1]-1)
        vol = img[:,c]
        if projection == "max":    plane = np.max(vol, axis=0)
        elif projection == "mean": plane = np.mean(vol, axis=0)
        else:
            zi = min(z if z is not None else self.current_z_slice, vol.shape[0]-1)
            plane = vol[zi]
        plane = plane.astype(np.float32)
        rng = np.ptp(plane)
        return (plane-plane.min())/rng if rng > 0 else plane

    @staticmethod
    def normalise_shape(arr):
        arr = arr.astype("float32")
        if arr.ndim == 2: return arr[None,None]
        if arr.ndim == 3: return arr[:,None]
        if arr.ndim == 4: return arr
        if arr.ndim == 5: return arr[0]
        raise ValueError(f"Unsupported shape {arr.shape}")


class ImageCanvas(FigureCanvas):
    def __init__(self, figsize=(6, 5), parent=None):
        self._fig = Figure(figsize=figsize, tight_layout=True)
        self._ax  = self._fig.add_subplot(111)
        super().__init__(self._fig)
        self.setParent(parent)
        self._style_dark()

    def _style_dark(self):
        self._fig.patch.set_facecolor("#0a0a14")
        self._ax.set_facecolor("#0a0a14")
        self._ax.tick_params(colors="#6060a0")
        for sp in self._ax.spines.values():
            sp.set_color("#2a2a50")

    @property
    def ax(self):  return self._ax
    @property
    def fig(self): return self._fig

    def show_image(self, img: np.ndarray, cmap="gray", title="",
                   vmin=None, vmax=None):
        self._ax.clear()
        self._style_dark()
        if img.ndim == 2:
            self._ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                            interpolation="nearest", origin="upper")
        else:
            self._ax.imshow(np.clip(img, 0, 1),
                            interpolation="nearest", origin="upper")
        self._ax.set_title(title, color="#a0a0d0", fontsize=9, pad=4)
        self._ax.axis("off")
        self.draw_idle()

    def clear_canvas(self):
        self._ax.clear()
        self._style_dark()
        self.draw_idle()


class SliderRow(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, label, lo, hi, val, decimals=2, parent=None):
        super().__init__(parent)
        self._dec   = decimals
        self._scale = 10 ** decimals
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(f"{label}:")
        lbl.setMinimumWidth(170)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lbl.setStyleSheet("color:#9090c0;")
        self._sl = QSlider(Qt.Horizontal)
        self._sl.setMinimum(int(lo * self._scale))
        self._sl.setMaximum(int(hi * self._scale))
        self._sl.setValue(int(val * self._scale))
        self._sl.setSingleStep(max(1, int(0.01 * self._scale)))
        self._vl = QLabel(f"{val:.{decimals}f}")
        self._vl.setMinimumWidth(48)
        self._vl.setStyleSheet("font-family:monospace;color:#c0c0ff;")
        self._sl.valueChanged.connect(self._on_change)
        lay.addWidget(lbl)
        lay.addWidget(self._sl, stretch=1)
        lay.addWidget(self._vl)

    def _on_change(self, raw):
        f = raw / self._scale
        self._vl.setText(f"{f:.{self._dec}f}")
        self.valueChanged.emit(f)

    def value(self):
        return self._sl.value() / self._scale

    def setValue(self, v):
        self._sl.blockSignals(True)
        self._sl.setValue(int(v * self._scale))
        self._vl.setText(f"{v:.{self._dec}f}")
        self._sl.blockSignals(False)

    def setEnabled(self, e):
        super().setEnabled(e)
        self._sl.setEnabled(e)


class StatusLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setStyleSheet("color:#707090;font-size:11px;")

    def info(self, msg):
        self.setStyleSheet("color:#80c0ff;font-size:11px;")
        self.setText(f"ℹ  {msg}")

    def ok(self, msg):
        self.setStyleSheet("color:#80ff80;font-size:11px;")
        self.setText(f"✓  {msg}")

    def warn(self, msg):
        self.setStyleSheet("color:#ffb060;font-size:11px;")
        self.setText(f"⚠  {msg}")

    def err(self, msg):
        self.setStyleSheet("color:#ff6060;font-size:11px;")
        self.setText(f"✗  {msg}")


TAB_HELP_TEXT: dict = {
    "Image Viewer": (
        "<h3>Image Viewer</h3>"
        "<p>Load <b>TIFF</b> or <b>LIF</b> multi-channel microscopy files. "
        "The image is normalised to <code>(Z, C, Y, X)</code> float32.</p>"
        "<h4>Channel Mapping</h4>"
        "<p>Assign a biological role (e.g. <i>Nucleus</i>, <i>CCT1</i>) to each channel. "
        "All tool tabs use these labels to auto-select the correct channel.</p>"
        "<p>Select <b>Other</b> from the dropdown to enter any custom label "
        "(e.g. <i>LC3</i>, <i>p62</i>, <i>Rab7</i>). "
        "The custom name propagates globally to overlays, exports, colocalization, "
        "and derived channels.</p>"
        "<h4>Projections</h4>"
        "<p><b>Max projection</b> — brightest pixel per column across Z slices.<br>"
        "<b>Mean projection</b> — average intensity.<br>"
        "<b>Single slice</b> — one Z plane, chosen with the Z slider.<br>"
        "<b>Composite</b> — colour blend of all enabled channels.</p>"
    ),
    "Preprocessing": (
        "<h3>Preprocessing</h3>"
        "<p>Cleans up the raw image before any segmentation runs. "
        "The pipeline executes in this fixed order:</p>"
        "<ol>"
        "<li><b>Background subtraction</b> — Gaussian (smooth σ) or rolling-ball. "
        "Removes slow uneven illumination. Larger σ → flatter background.</li>"
        "<li><b>Top-Hat filter</b> — enhances small bright objects (aggregates, puncta) "
        "by subtracting a morphological opening. Disk radius ≈ punctum diameter.</li>"
        "<li><b>Noise suppression</b> — Gaussian (fast, blurs edges) or bilateral "
        "(edge-preserving). Use σ ≈ 1.0–1.5 for typical confocal stacks.</li>"
        "<li><b>Percentile normalisation</b> — clips intensities to "
        "[p_low, p_high]. Defaults 1 % / 99 % handle most fluorescent images.</li>"
        "<li><b>CLAHE</b> — local contrast equalisation. Useful when bright cells "
        "drown out dim ones; skip for uniformly bright fields.</li>"
        "</ol>"
        "<h4>★ Recommended presets</h4>"
        "<ul>"
        "<li><b>Aggregate</b> — BG Gaussian σ = 50, Denoise Gaussian σ = 1.0, "
        "all other steps OFF. Matches the DL aggregate-detection paper.</li>"
        "<li><b>Cell Body (CCT1)</b> — strong BG + percentile norm to flatten "
        "cytoplasmic stains for Cellpose.</li>"
        "<li><b>Nucleus (fluorescent)</b> — minimal cleanup; keeps DAPI/Hoechst "
        "intensity structure intact for nuclear segmentation.</li>"
        "</ul>"
        "<h4>Preview</h4>"
        "<p>The right canvas shows Original vs Processed for the currently "
        "selected channel and Z-slice. Use the <b>Colormap</b> dropdown "
        "(default <i>hot</i>) to inspect dim structure.</p>"
        "<h4>Apply &amp; Store</h4>"
        "<p>Click <b>Apply Preprocessing</b> to commit; every downstream tab "
        "(Cell Body / Nuclei, Aggregate Detection, Colocalization, Cell Region "
        "Analysis) automatically picks up the preprocessed image. "
        "Use <b>Store as Derived</b> to keep both raw and processed versions "
        "selectable in channel dropdowns.</p>"
    ),
    "Cell Body Detection": (
        "<h3>Cell Body Detection</h3>"
        "<p>Uses <b>Cellpose</b> to segment cell bodies from the selected channel.</p>"
        "<p>Set the <i>diameter</i> to the approximate cell diameter in pixels. "
        "Use <i>0</i> for automatic estimation.</p>"
    ),
    "Cell Body Nuclei Detection": (
        "<h3>Cell Body &amp; Nuclei Detection</h3>"
        "<p>Two independent Cellpose runs in one tab: one for cell bodies, one for "
        "nuclei. Each uses its own channel, parameters, and Run button.</p>"
        "<h4>Channel assignment</h4>"
        "<p>Pick the channel that <i>visually shows</i> the structure: "
        "for cells the cytoplasmic stain (e.g. CCT1), for nuclei the DAPI/Hoechst "
        "channel. Preprocessed and derived channels are also selectable.</p>"
        "<h4>Segmentation modes</h4>"
        "<ul>"
        "<li><b>Max / Mean Projection</b> — collapse the Z-stack first. Fast; "
        "loses 3-D information.</li>"
        "<li><b>Slice (Z)</b> — segment one chosen plane.</li>"
        "<li><b>All slices (Z)</b> — run Cellpose on every Z slice and stack "
        "the labels (3-D label volume).</li>"
        "</ul>"
        "<h4>Active Z-layer</h4>"
        "<p>After <b>All slices (Z)</b>, click <b>Select Active Cell/Nucleus Layer</b> "
        "to pin one Z plane as the downstream truth (used by Cell Region Analysis, "
        "statistics, exports). Choose <i>Use projection mode</i> in the dialog to "
        "revert to the current projection mode.</p>"
        "<h4>Key parameters</h4>"
        "<ul>"
        "<li><b>Diameter (px)</b> — most important. Measure a typical object "
        "edge-to-edge. Use 0 for auto-estimate (slower, dataset-adaptive).</li>"
        "<li><b>Flow threshold</b> (0.0–1.5, default 0.4) — higher = more detections. "
        "Lower (0.2–0.3) to stop touching objects merging; raise (0.5–0.7) to "
        "rescue elongated or dim objects.</li>"
        "<li><b>Cell prob</b> (–6…+6, default 0) — lower expands masks (catches "
        "dim edges); raise to tighten boundaries.</li>"
        "<li><b>Min area (px)</b> — discards objects below this size (noise / "
        "debris). Typical: 100–800 px depending on magnification.</li>"
        "</ul>"
        "<h4>Manual editing</h4>"
        "<p>After a run, click cells on the canvas to toggle inclusion. "
        "Use <b>Draw Polygon</b> to add missing objects manually. Undo/Redo "
        "are available per channel.</p>"
        "<h4>Downstream</h4>"
        "<p>The masks are stored as derived channels <code>cell_body_mask</code> "
        "and <code>nucleus_mask</code> and feed Cell Region Analysis, statistics, "
        "and exports.</p>"
    ),
    "Aggregate Detection": (
        "<h3>Aggregate Detection</h3>"
        "<p>Deep Learning ensemble segmentation (10-fold U-Net) for protein aggregates. "
        "Uses TTA (8× augmentation) and Hann-windowed tiling for robust results.</p>"
        "<p>After running: use <b>Correction</b> tab to review detections against ground truth, "
        "and <b>Validation</b> tab to compute precision/recall metrics.</p>"
        "<p>The derived channel <code>aggregates</code> is also registered for use in "
        "Colocalization and Cell Region Analysis.</p>"
    ),
    "Colocalization": (
        "<h3>Colocalization</h3>"
        "<p>Measures spatial overlap between two channels.</p>"
        "<p>Reports <b>Pearson correlation</b> and <b>Manders coefficients</b>.</p>"
        "<p>Channels are selected by <b>biological label</b> — raw channels and "
        "derived channels (Aggregates) are available. "
        "Custom labels (set via <i>Other</i> in Image Viewer) appear here too.</p>"
    ),
    "Chaperone Segmentation": (
        "<h3>Chaperone Segmentation</h3>"
        "<p><i>This tab has been removed.</i> "
        "Use the Colocalization tab to compare any two derived channels.</p>"
    ),
    "Results": (
        "<h3>Results</h3>"
        "<p>View and export results from any tool tab.</p>"
        "<p>Use <b>Export CSV</b> or <b>Export JSON</b> to save data.</p>"
    ),
    "Nucleus_Aggregate_Segmentation": (
        "<h3>Nucleus + Aggregate Segmentation</h3>"
        "<p>Jointly segments nuclei and detects aggregates within nuclear regions.</p>"
    ),
    "Cell_Nuclei_Segmentation": (
        "<h3>Cell + Nuclei Segmentation</h3>"
        "<p>Segments cell bodies and nuclei simultaneously for paired analysis.</p>"
    ),

    "Pre Processing": None,            # alias filled in below
    "Cell Body / Nuclei Tool": None,   # alias filled in below
    "Cell Region Analysis": (
        "<h3>Cell Region Analysis</h3>"
        "<p>Classifies every pixel into four biologically meaningful regions, "
        "<b>fully computed per cell</b> — no regions ever merge across cell boundaries:</p>"
        "<ol>"
        "<li><b>Nucleus</b> — nucleus pixels belonging to this cell. "
        "Stored with the originating <code>cell_id</code> as the pixel value, "
        "preserving ownership for statistics and exports.</li>"
        "<li><b>Perinuclear</b> — <i>adaptive shape-aware</i> shell around the nucleus. "
        "Thickness = 0.5 × local nucleus radius per direction "
        "Elongated and irregular nuclei receive correctly shaped shells; "
        "no fixed-width uniform ring is used.</li>"
        "<li><b>Cytoplasm</b> — everything inside this cell that is not nucleus, "
        "perinuclear or periphery. "
        "Computed exclusively within the cell boundary — "
        "touching cells <em>never</em> share cytoplasm pixels.</li>"
        "<li><b>Cell periphery</b> — outer cortical band within N pixels of the cell edge.</li>"
        "</ol>"
        "<h4>Per-cell isolation &amp; black separation</h4>"
        "<p>Cell boundary pixels are detected by comparing each pixel's label with its "
        "4-connected neighbours. A pixel is a border if any neighbour belongs to a "
        "different cell, correctly identifying the line between every pair of touching cells. "
        "Border pixels are subtracted from region masks before rendering and "
        "burned as black lines on top of the overlay.</p>"
        "<h4>Active Z-layer</h4>"
        "<p>Use <b>Select Active Cell/Nucleus Layer</b> to pin a Z plane. "
        "Select <i>Use projection mode</i> to revert to max/mean projection.</p>"
        "<h4>Signal per Region</h4>"
        "<ul>"
        "<li><b>Pixels</b> — total region area in pixels</li>"
        "<li><b>% signal</b> — fraction of channel intensity in this region</li>"
        "<li><b>Objects</b> — distinct connected structures (e.g. aggregates) "
        "detected via Otsu + connected-component labelling</li>"
        "</ul>"
        "<h4>Workflow</h4>"
        "<ol>"
        "<li>Run cell body + nucleus segmentation; optionally select active Z-layer.</li>"
        "<li>Click <b>Run Cell Region Analysis</b>.</li>"
        "<li>Select a signal channel and click <b>Compute Region Signal</b>.</li>"
        "</ol>"
    ),
}

# Display-name aliases so HelpButton("Pre Processing") and
# HelpButton("Cell Body / Nuclei Tool") resolve to the canonical entries.
TAB_HELP_TEXT["Pre Processing"]          = TAB_HELP_TEXT["Preprocessing"]
TAB_HELP_TEXT["Cell Body / Nuclei Tool"] = TAB_HELP_TEXT["Cell Body Nuclei Detection"]
# Toolbar tab display string uses "&"; alias it so the global Help button resolves.
TAB_HELP_TEXT["Cell Body & Nuclei Detection"] = TAB_HELP_TEXT["Cell Body Nuclei Detection"]


class HelpButton(QPushButton):
    def __init__(self, tab_key: str, parent=None):
        super().__init__("?", parent)
        self._tab_key = tab_key
        self.setStyleSheet(BTN_HELP)
        self.setToolTip(f"Help for: {tab_key}")
        self.clicked.connect(self._show_help)
        self.setFixedSize(24, 24)

    def _show_help(self):
        html = TAB_HELP_TEXT.get(self._tab_key,
               f"<p>No help available for <b>{self._tab_key}</b>.</p>")
        dlg = QMessageBox(self)
        dlg.setWindowTitle(f"Help – {self._tab_key}")
        dlg.setTextFormat(Qt.RichText)
        dlg.setText(html)
        dlg.setStandardButtons(QMessageBox.Ok)
        dlg.setStyleSheet(
            "QMessageBox{background:#1c1c32;color:#dcdcf0;}"
            "QLabel{color:#dcdcf0;font-size:12px;}"
            "QPushButton{background:#2a2a50;color:#c0c0ff;border:1px solid #3a3a70;"
            "border-radius:4px;padding:5px 16px;}")
        dlg.exec_()

def make_scroll_widget() -> tuple:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    inner  = QWidget()
    layout = QVBoxLayout(inner)
    layout.setSpacing(6)
    layout.setContentsMargins(4, 4, 4, 4)
    scroll.setWidget(inner)
    return scroll, inner, layout

def section(title: str) -> QGroupBox:
    g = QGroupBox(title)
    return g

_SPINBOX_BTN_QSS = (
    "QPushButton{"
    "background:#2a2a55;color:#d0d0ff;"
    "border:1px solid #4a4a80;"
    "border-radius:5px;"
    "font-size:16px;font-weight:bold;"
    "padding:0;min-width:28px;min-height:28px;"
    "}"
    "QPushButton:hover{background:#3c3c70;border-color:#6060b0;color:#ffffff;}"
    "QPushButton:pressed{background:#1a1a38;}"
    "QPushButton:disabled{color:#404060;background:#141428;border-color:#202040;}"
)

def _make_spinbox_row(label: str, spinbox,
                      tooltip: str = "") -> QHBoxLayout:
    """Return a QHBoxLayout: label | [−] spinbox [+]
    Cross-platform (macOS + Windows): large buttons, no native arrows.
    """
    from PyQt5.QtWidgets import QAbstractSpinBox as _ASB
    # Disable native Qt arrows — required for macOS consistency
    spinbox.setButtonSymbols(_ASB.NoButtons)
    spinbox.setMinimumHeight(28)
    spinbox.setMinimumWidth(72)

    row = QHBoxLayout()
    row.setContentsMargins(0, 2, 0, 2)
    row.setSpacing(5)

    lbl = QLabel(label)
    lbl.setMinimumWidth(130)
    lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    lbl.setStyleSheet("color:#a0a0cc;")

    if tooltip:
        spinbox.setToolTip(tooltip)

    minus_btn = QPushButton("−")
    minus_btn.setFixedSize(28, 28)
    minus_btn.setStyleSheet(_SPINBOX_BTN_QSS)
    minus_btn.setToolTip("Decrease")

    plus_btn = QPushButton("+")
    plus_btn.setFixedSize(28, 28)
    plus_btn.setStyleSheet(_SPINBOX_BTN_QSS)
    plus_btn.setToolTip("Increase")

    minus_btn.clicked.connect(spinbox.stepDown)
    plus_btn.clicked.connect(spinbox.stepUp)

    row.addWidget(lbl)
    row.addWidget(minus_btn)
    row.addWidget(spinbox)
    row.addWidget(plus_btn)
    row.addStretch()
    return row


class WorkerSignals(QObject):
    started          = pyqtSignal()
    progress         = pyqtSignal(str)
    result           = pyqtSignal(object)
    error            = pyqtSignal(str)
    finished         = pyqtSignal()
    progress_percent = pyqtSignal(int)   # 0-100 real progress percentage


# Base class for all background tasks (wraps QThread)
class BaseWorker(QThread):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = WorkerSignals()

    def run(self):
        self.signals.started.emit()
        try:
            result = self.run_task()
            self.signals.result.emit(result)
        except Exception as e:
            self.signals.error.emit(f"{e}\n\n{traceback.format_exc()}")
        finally:
            self.signals.finished.emit()

    def run_task(self):
        raise NotImplementedError


class ImageViewerTab(QWidget):

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self._state = state
        self._worker = None
        self._ch_mapping_widgets: list[dict] = []
        self._lif_series: list = []   # list of (name, np.ndarray (Z,C,Y,X))
        self._build_ui()
        state.subscribe(self._on_state_change)

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        ls, _, ll = make_scroll_widget()
        ls.setFixedWidth(290)

        fg = section("Load Image")
        fl = QVBoxLayout(fg)
        self._load_btn = QPushButton("📂  Load TIFF / LIF")
        self._load_btn.setStyleSheet(BTN_LOAD)
        self._file_lbl = StatusLabel()
        self._file_lbl.setText("No file loaded")
        fl.addWidget(self._load_btn)
        fl.addWidget(self._file_lbl)
        self._series_lbl = QLabel("Series:")
        self._series_lbl.setStyleSheet("color:#9090c0;font-size:11px;")
        self._series_combo = QComboBox()
        self._series_combo.setToolTip(
            "Select which image series to display (LIF files only).\n"
            "Each series is a separate acquisition in the same file.")
        self._series_lbl.setVisible(False)
        self._series_combo.setVisible(False)
        fl.addWidget(self._series_lbl)
        fl.addWidget(self._series_combo)
        ll.addWidget(fg)

        cg = section("Channel & Projection")
        cg_hdr = QHBoxLayout()
        cg_title = QLabel("Channel & Projection")
        cg_title.setStyleSheet("color:#9090c8;font-weight:600;font-size:11px;")
        _help_btn_viewer = HelpButton("Image Viewer")
        cg_hdr.addWidget(cg_title)
        cg_hdr.addStretch()
        cg_hdr.addWidget(_help_btn_viewer)
        cf = QFormLayout(cg)
        cf.addRow(cg_hdr)
        self._ch_spin = QSpinBox()
        self._ch_spin.setRange(0, 0)
        self._ch_spin.setButtonSymbols(QSpinBox.NoButtons)
        self._ch_spin.setMinimumHeight(28)
        self._proj_combo = QComboBox()
        self._proj_combo.addItems([
            "max projection",
            "mean projection",
            "single slice",
            "composite (all channels)",
        ])
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(["gray","hot","inferno","magma","viridis","plasma","bone","cividis"])
        cf.addRow("Channel:", self._ch_spin)
        cf.addRow("Projection:", self._proj_combo)
        cf.addRow("Colormap:", self._cmap_combo)
        self._cmap_note = QLabel("(colormap ignored in composite mode)")
        self._cmap_note.setStyleSheet("color:#505070;font-size:10px;")
        self._cmap_note.setVisible(False)
        cf.addRow("", self._cmap_note)
        ll.addWidget(cg)

        mg = section("Channel Mapping  ★ Source of Truth ★")
        mg_note_lbl = QLabel(
            "Assign biological roles here. All tool tabs use these labels.")
        mg_note_lbl.setWordWrap(True)
        mg_note_lbl.setStyleSheet(
            "color:#5050a0;font-size:10px;font-style:italic;")
        self._mapping_layout = QVBoxLayout(mg)
        self._mapping_layout.addWidget(mg_note_lbl)
        self._mapping_note = QLabel("Load an image to configure channel mappings.")
        self._mapping_note.setWordWrap(True)
        self._mapping_note.setStyleSheet("color:#606080;font-size:11px;")
        self._mapping_layout.addWidget(self._mapping_note)
        ll.addWidget(mg)

        zg = section("Z-Slice")
        zl = QVBoxLayout(zg)
        self._z_lbl = QLabel("Z: 0 / 0")
        self._z_lbl.setStyleSheet("color:#8080c0;")
        self._z_slider = QSlider(Qt.Horizontal)
        self._z_slider.setRange(0, 0)
        zl.addWidget(self._z_lbl)
        zl.addWidget(self._z_slider)
        ll.addWidget(zg)

        ig = section("Image Info")
        il = QVBoxLayout(ig)
        self._info_lbl = QLabel("—")
        self._info_lbl.setStyleSheet("color:#707090;font-size:11px;")
        self._info_lbl.setWordWrap(True)
        il.addWidget(self._info_lbl)
        ll.addWidget(ig)

        ll.addStretch()
        root.addWidget(ls)

        self._view_tabs = QTabWidget()
        self._view_tabs.setDocumentMode(True)

        cw_single = QWidget()
        cl_single = QVBoxLayout(cw_single)
        cl_single.setContentsMargins(0, 0, 0, 0)
        self._canvas = ImageCanvas(figsize=(8, 7))
        self._nav    = NavigationToolbar(self._canvas, self)
        self._nav.setStyleSheet(_viewer_tb_qss())
        cl_single.addWidget(self._nav)
        cl_single.addWidget(self._canvas, stretch=1)

        cw_multi = QWidget()
        cl_multi = QVBoxLayout(cw_multi)
        cl_multi.setContentsMargins(2, 2, 2, 2)
        cl_multi.setSpacing(4)

        mc_bar = QWidget()
        mc_bar_l = QHBoxLayout(mc_bar)
        mc_bar_l.setContentsMargins(0, 0, 0, 0)
        mc_bar_l.setSpacing(8)
        self._mc_show_all = QPushButton("Show All")
        self._mc_show_all.setStyleSheet(BTN_LOAD)
        self._mc_show_all.setFixedHeight(24)
        self._mc_hide_all = QPushButton("Hide All")
        self._mc_hide_all.setStyleSheet(BTN_WARN)
        self._mc_hide_all.setFixedHeight(24)
        mc_bar_l.addWidget(self._mc_show_all)
        mc_bar_l.addWidget(self._mc_hide_all)
        mc_bar_l.addWidget(QLabel("Projection:"))
        self._mc_proj = QComboBox()
        self._mc_proj.addItems(["Max", "Mean", "Slice"])
        self._mc_proj.setFixedWidth(70)
        mc_bar_l.addWidget(self._mc_proj)
        mc_bar_l.addWidget(QLabel("Z:"))
        self._mc_z_slider = QSlider(Qt.Horizontal)
        self._mc_z_slider.setRange(0, 0)
        self._mc_z_slider.setEnabled(False)
        self._mc_z_slider.setFixedWidth(100)
        self._mc_z_lbl = QLabel("0")
        self._mc_z_lbl.setStyleSheet("color:#8080c0;min-width:24px;")
        mc_bar_l.addWidget(self._mc_z_slider)
        mc_bar_l.addWidget(self._mc_z_lbl)
        mc_bar_l.addStretch()
        cl_multi.addWidget(mc_bar)

        mc_ch_scroll = QScrollArea()
        mc_ch_scroll.setWidgetResizable(True)
        mc_ch_scroll.setFixedHeight(46)
        mc_ch_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        mc_ch_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        mc_ch_scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        self._mc_ch_container = QWidget()
        self._mc_ch_layout = QHBoxLayout(self._mc_ch_container)
        self._mc_ch_layout.setContentsMargins(2, 0, 2, 0)
        self._mc_ch_layout.setSpacing(8)
        self._mc_ch_placeholder = QLabel("Load an image to see channels.")
        self._mc_ch_placeholder.setStyleSheet("color:#505070;font-size:11px;")
        self._mc_ch_layout.addWidget(self._mc_ch_placeholder)
        self._mc_ch_layout.addStretch()
        mc_ch_scroll.setWidget(self._mc_ch_container)
        cl_multi.addWidget(mc_ch_scroll)

        self._mc_canvas = ImageCanvas(figsize=(8, 7))
        self._mc_nav    = NavigationToolbar(self._mc_canvas, self)
        self._mc_nav.setStyleSheet(_viewer_tb_qss())
        cl_multi.addWidget(self._mc_nav)
        cl_multi.addWidget(self._mc_canvas, stretch=1)

        self._view_tabs.addTab(cw_single, "Single Channel")
        self._view_tabs.addTab(cw_multi,  "Multi-Channel 🎨")
        root.addWidget(self._view_tabs, stretch=1)

        self._mc_ch_rows: list = []

        self._load_btn.clicked.connect(self._load_file)
        self._series_combo.currentIndexChanged.connect(self._series_changed)
        self._ch_spin.valueChanged.connect(self._refresh)
        self._proj_combo.currentIndexChanged.connect(self._on_proj_change)
        self._z_slider.valueChanged.connect(self._on_z_change)
        self._cmap_combo.currentTextChanged.connect(self._refresh)

        self._mc_show_all.clicked.connect(self._mc_on_show_all)
        self._mc_hide_all.clicked.connect(self._mc_on_hide_all)
        self._mc_proj.currentIndexChanged.connect(self._mc_on_proj_change)
        self._mc_z_slider.valueChanged.connect(self._mc_on_z_change)

    def _rebuild_mc_rows(self) -> None:
        for row in self._mc_ch_rows:
            row.setParent(None)
        self._mc_ch_rows.clear()

        mappings = self._state.channel_mappings
        if not mappings:
            self._mc_ch_placeholder.setVisible(True)
            return
        self._mc_ch_placeholder.setVisible(False)

        _default_colors = {"CCT1": "Green", "HA (mHTT)": "Magenta",
                           "Nucleus": "Blue", "A11": "Red"}
        _cycle = ["Cyan", "Yellow", "Orange", "White", "Gray",
                  "Green", "Red", "Blue", "Magenta"]

        for i, m in enumerate(mappings):
            bio   = m.get("biological_label", "—") or "—"
            name  = m.get("channel_name", f"Ch {i}")
            label = bio if bio not in ("—", "") else name
            col_from_map = m.get("color", "").capitalize()
            if col_from_map in _NAMED_COLORS:
                default_col = col_from_map
            elif bio in _default_colors:
                default_col = _default_colors[bio]
            else:
                default_col = _cycle[i % len(_cycle)]
            row = ChannelRowWidget(m["data_channel_index"], label, default_col)
            row.changed.connect(self._mc_refresh)
            self._mc_ch_layout.insertWidget(i, row)
            self._mc_ch_rows.append(row)

        img = self._state.preprocessed_image if self._state.has_preprocessed() else self._state.raw_image
        if img is not None:
            nz = img.shape[0]
            self._mc_z_slider.blockSignals(True)
            self._mc_z_slider.setRange(0, max(0, nz - 1))
            self._mc_z_slider.setValue(0)
            self._mc_z_slider.blockSignals(False)
            self._mc_z_lbl.setText("0")

    def _mc_on_show_all(self):
        for row in self._mc_ch_rows:
            row.set_enabled(True)
        self._mc_refresh()

    def _mc_on_hide_all(self):
        for row in self._mc_ch_rows:
            row.set_enabled(False)
        self._mc_refresh()

    def _mc_on_proj_change(self, idx: int):
        self._mc_z_slider.setEnabled(idx == 2)
        self._mc_refresh()

    def _mc_on_z_change(self, v: int):
        self._mc_z_lbl.setText(str(v))
        if self._mc_proj.currentIndex() == 2:
            self._mc_refresh()

    def _mc_refresh(self, _=None):
        img = self._state.preprocessed_image if self._state.has_preprocessed() else self._state.raw_image
        if img is None or not self._mc_ch_rows:
            self._mc_canvas.clear_canvas()
            return
        proj_map = {0: "max", 1: "mean", 2: "slice"}
        proj   = proj_map.get(self._mc_proj.currentIndex(), "max")
        z_idx  = self._mc_z_slider.value()
        states = [row.state() for row in self._mc_ch_rows]
        enabled = [s for s in states if s["enabled"]]
        if not enabled:
            self._mc_canvas.clear_canvas()
            return
        composite = blend_channels(img, states, z_index=z_idx, projection=proj)
        names = []
        for s in enabled:
            for m in self._state.channel_mappings:
                if m["data_channel_index"] == s["channel_index"]:
                    bio = m.get("biological_label", "—")
                    names.append(bio if bio not in ("—", "") else m.get("channel_name", f"Ch{s['channel_index']}"))
                    break
        proj_lbl = {"max": "Max-Proj", "mean": "Mean-Proj"}.get(proj, f"Z={z_idx}")
        self._mc_canvas.show_image(composite, title=f"{proj_lbl}  |  {' + '.join(names)}")

    _MAPPING_COLORS = ["gray", "red", "green", "blue",
                       "cyan", "magenta", "yellow", "white"]

    _LABEL_KEYWORDS: list[tuple[str, str]] = [
        ("cct",      "CCT1"),
        ("htt",      "HA (mHTT)"),
        ("ha",       "HA (mHTT)"),
        ("mhtt",     "HA (mHTT)"),
        ("nucleus",  "Nucleus"),
        ("nucl",     "Nucleus"),
        ("dapi",     "Nucleus"),
        ("hoechst",  "Nucleus"),
        ("a11",      "A11"),
    ]

    def _guess_biological_label(self, raw_name: str) -> str:
        lower = raw_name.lower()
        for kw, label in self._LABEL_KEYWORDS:
            if kw in lower:
                return label
        return "—"

    def _build_channel_mappings(self, nc: int,
                                ch_names: list[str] | None = None) -> None:
        for row_widgets in self._ch_mapping_widgets:
            for w in row_widgets.values():
                if hasattr(w, "setParent"):
                    w.setParent(None)
        self._ch_mapping_widgets.clear()

        self._mapping_note.setVisible(False)

        if ch_names and len(ch_names) == nc:
            labels = [str(n).strip() or f"Channel {c+1}" for c, n in enumerate(ch_names)]
        else:
            labels = [f"Channel {c+1}" for c in range(nc)]

        self._state.channel_names = labels

        for c in range(nc):
            ch_label = labels[c]

            row_w = QWidget()
            row_l = QVBoxLayout(row_w)
            row_l.setContentsMargins(0, 2, 0, 4)
            row_l.setSpacing(2)

            top_row = QWidget()
            top_l   = QHBoxLayout(top_row)
            top_l.setContentsMargins(0, 0, 0, 0)
            top_l.setSpacing(4)

            cb = QCheckBox(ch_label)
            cb.setChecked(True)
            cb.setToolTip(f"Enable/disable channel: {ch_label}")
            cb.setStyleSheet("font-size:11px;color:#b0b0d8;")
            top_l.addWidget(cb)
            top_l.addStretch()

            row_l.addWidget(top_row)

            bot_row = QWidget()
            bot_l   = QHBoxLayout(bot_row)
            bot_l.setContentsMargins(12, 0, 0, 0)
            bot_l.setSpacing(4)

            bio_combo = QComboBox()
            bio_combo.addItems(BIOLOGICAL_LABELS)
            guessed = self._guess_biological_label(ch_label)
            bio_combo.setCurrentText(guessed)
            bio_combo.setFixedWidth(100)
            bio_combo.setToolTip(
                "Assign a biological role to this channel.\n"
                "Select 'Other' to enter a custom label (e.g. LC3, p62).\n"
                "All tool tabs use these labels to auto-select the correct channel.")


            _saved_mappings = self._state.channel_mappings
            _saved_custom = ""
            _saved_bio_combo_val = guessed
            if c < len(_saved_mappings):
                sm = _saved_mappings[c]
                _saved_bio_combo_val = sm.get("bio_combo_value", guessed)
                _saved_custom = sm.get("custom_label_text", "")
                if not _saved_bio_combo_val:
                    # Backward compat: if custom label was stored as biological_label
                    bl = sm.get("biological_label", "")
                    if bl and bl not in [lbl for lbl in BIOLOGICAL_LABELS if lbl != "Other"]:
                        _saved_bio_combo_val = "Other"
                        _saved_custom = bl
                    else:
                        _saved_bio_combo_val = bl or guessed
            bio_combo.setCurrentText(_saved_bio_combo_val)


            custom_label_edit = QLineEdit()
            custom_label_edit.setPlaceholderText("Custom label…")
            custom_label_edit.setFixedWidth(90)
            custom_label_edit.setMinimumHeight(26)
            custom_label_edit.setToolTip(
                "Enter a custom biological label (e.g. LC3, p62, Rab7).\n"
                "This label will be used everywhere 'Other' would appear.")
            custom_label_edit.setText(_saved_custom)
            custom_label_edit.setVisible(_saved_bio_combo_val == "Other")
            custom_label_edit.setStyleSheet(
                "QLineEdit{background:#1a1a38;color:#d0d0ff;"
                "border:1px solid #4a4a80;border-radius:4px;padding:2px 6px;}")

            color_combo = QComboBox()
            color_combo.addItems(self._MAPPING_COLORS)

            _color_key = _saved_bio_combo_val if _saved_bio_combo_val != "Other" else _saved_custom
            default_color = LABEL_DEFAULT_COLORS.get(_color_key,
                            LABEL_DEFAULT_COLORS.get(_saved_bio_combo_val,
                            self._MAPPING_COLORS[c % len(self._MAPPING_COLORS)]))
            color_combo.setCurrentText(default_color)
            color_combo.setFixedWidth(74)
            color_combo.setToolTip("Display colour (used in composite view and overlays)")

            op_spin = QDoubleSpinBox()
            op_spin.setRange(0.0, 1.0)
            op_spin.setSingleStep(0.1)
            op_spin.setValue(1.0)
            op_spin.setDecimals(1)
            op_spin.setFixedWidth(58)
            op_spin.setToolTip("Opacity α (used in composite blending)")
            from PyQt5.QtWidgets import QAbstractSpinBox as _ASBop
            op_spin.setButtonSymbols(_ASBop.NoButtons)
            op_spin.setMinimumHeight(28)

            bot_l.addWidget(bio_combo)
            bot_l.addWidget(custom_label_edit)
            bot_l.addWidget(color_combo)

            op_minus = QPushButton("−")
            op_minus.setFixedSize(28, 28)
            op_minus.setStyleSheet(_SPINBOX_BTN_QSS)
            op_minus.setToolTip("Decrease opacity")
            op_minus.clicked.connect(op_spin.stepDown)
            op_plus = QPushButton("+")
            op_plus.setFixedSize(28, 28)
            op_plus.setStyleSheet(_SPINBOX_BTN_QSS)
            op_plus.setToolTip("Increase opacity")
            op_plus.clicked.connect(op_spin.stepUp)

            bot_l.addWidget(QLabel("α"))
            bot_l.addWidget(op_minus)
            bot_l.addWidget(op_spin)
            bot_l.addWidget(op_plus)
            bot_l.addStretch()
            row_l.addWidget(bot_row)

            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setStyleSheet("color:#2a2a48;")
            row_l.addWidget(line)

            self._mapping_layout.addWidget(row_w)
            widgets = {
                "container":       row_w,
                "enabled":         cb,
                "bio_label":       bio_combo,
                "custom_label":    custom_label_edit,
                "color":           color_combo,
                "opacity":         op_spin,
                "channel_index":   c,
                "channel_name":    ch_label,
            }
            self._ch_mapping_widgets.append(widgets)

            def _on_bio_change(text, cc=color_combo, cle=custom_label_edit):

                cle.setVisible(text == "Other")
                suggested = LABEL_DEFAULT_COLORS.get(text, "")
                if suggested and suggested in [cc.itemText(i) for i in range(cc.count())]:
                    cc.setCurrentText(suggested)
                self._save_mappings()

            bio_combo.currentTextChanged.connect(_on_bio_change)

            custom_label_edit.textChanged.connect(self._save_mappings)
            cb.stateChanged.connect(self._save_mappings)
            color_combo.currentTextChanged.connect(self._save_mappings)
            op_spin.valueChanged.connect(self._save_mappings)

        self._save_mappings()

    def _save_mappings(self, _=None) -> None:
        mappings = []
        for w in self._ch_mapping_widgets:
            raw_label = w["bio_label"].currentText()

            if raw_label == "Other":
                custom = w["custom_label"].text().strip()
                effective_label = custom if custom else "Other"
            else:
                effective_label = raw_label
            mappings.append({
                "data_channel_index": w["channel_index"],
                "channel_name":       w["channel_name"],
                "biological_label":   effective_label,
                # store raw combo value so we can restore "Other" + custom on reload
                "bio_combo_value":    raw_label,
                "custom_label_text":  w["custom_label"].text().strip(),
                "color":              w["color"].currentText(),
                "enabled":            w["enabled"].isChecked(),
                "opacity":            round(w["opacity"].value(), 1),
            })
        self._state.channel_mappings = mappings
        self._state.notify("channel_mappings")

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open microscopy image", "",
            "Images (*.tif *.tiff *.lif *.png *.jpg);;All files (*)")
        if not path:
            return
        self._file_lbl.info(f"Loading {os.path.basename(path)}…")
        QApplication.processEvents()
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".lif":
                self._load_lif(path)
            else:
                arr = self._read_tiff_or_image(path)
                arr = AppState.normalise_shape(arr)
                self._lif_series = []
                self._series_lbl.setVisible(False)
                self._series_combo.setVisible(False)
                self._apply_array(arr, path)
        except Exception as e:
            self._file_lbl.err(f"Load failed: {e}")
            QMessageBox.critical(self, "Load Error", str(e))

    def _load_lif(self, path: str) -> None:
        if not HAS_READLIF:
            try:
                arr = self._read_tiff_or_image(path)
                arr = AppState.normalise_shape(arr)
                self._lif_series = []
                self._series_lbl.setVisible(False)
                self._series_combo.setVisible(False)
                self._apply_array(arr, path)
                return
            except Exception:
                raise ImportError(
                    "readlif is not installed and AICSImage fallback failed.\n"
                    "Install with:  pip install readlif")

        lif = _LifFile(path)
        self._lif_series = []

        for img in lif.get_iter_image():
            dims = img.dims
            nz   = dims.z if dims.z > 0 else 1
            nc   = img.channels if (hasattr(img, "channels") and img.channels > 0) else 1
            h, w = dims.y, dims.x
            data = np.zeros((nz, nc, h, w), dtype=np.float32)
            for ci in range(nc):
                for zi in range(nz):
                    frame = np.array(img.get_frame(z=zi, t=0, c=ci), dtype=np.float32)
                    data[zi, ci] = frame

            ch_names: list[str] = []
            try:
                raw_names = getattr(img, "channel_names", None)
                if raw_names and len(raw_names) == nc:
                    ch_names = [str(n).strip() or f"Channel {ci+1}"
                                for ci, n in enumerate(raw_names)]
                else:
                    ch_names = [f"Channel {ci+1}" for ci in range(nc)]
            except Exception:
                ch_names = [f"Channel {ci+1}" for ci in range(nc)]

            self._lif_series.append((img.name, data, ch_names))

        if not self._lif_series:
            raise ValueError("No image series found in the LIF file.")

        self._series_combo.blockSignals(True)
        self._series_combo.clear()
        for name, _, _ch in self._lif_series:
            self._series_combo.addItem(name)
        self._series_combo.blockSignals(False)

        multi = len(self._lif_series) > 1
        self._series_lbl.setVisible(multi)
        self._series_combo.setVisible(multi)

        self._series_combo.setCurrentIndex(0)
        name, arr, ch_names = self._lif_series[0]
        self._apply_array(arr, path, series_name=name, ch_names=ch_names)
        self._file_lbl.ok(
            os.path.basename(path)
            + (f"  ({len(self._lif_series)} series)" if multi else ""))

    def _series_changed(self, idx: int) -> None:
        if not self._lif_series or idx < 0 or idx >= len(self._lif_series):
            return
        name, arr, ch_names = self._lif_series[idx]
        self._apply_array(arr, self._state.file_path,
                          series_name=name, ch_names=ch_names)

    def _apply_array(self, arr: np.ndarray, path: str,
                     series_name: str = "",
                     ch_names: list[str] | None = None) -> None:
        if arr.ndim != 4: arr = AppState.normalise_shape(arr)
        nz, nc, ny, nx = arr.shape
        if nz >= 4:
            m = [float(arr[z,0].mean()) for z in range(min(16,nz))]
            d = [m[i+1]-m[i] for i in range(len(m)-1)]
            sg = [1 if x>0 else -1 for x in d]
            if len(sg)>2 and all(sg[i]!=sg[i+1] for i in range(len(sg)-1)):
                flat = arr.reshape(nz*nc, ny, nx)
                k = _detect_channels(flat)
                if k > 1:
                    arr = flat.reshape(nz*nc//k, k, ny, nx).astype("float32")
                    nz, nc, ny, nx = arr.shape
        print(f"[load] FINAL Z={nz} C={nc}")
        self._state.raw_image          = arr
        self._state.preprocessed_image = arr.copy()
        self._state.file_path          = path

        try:
            from aicsimageio import AICSImage
            aics = AICSImage(path)
            self._state.aics_image = aics
            try:
                ps = aics.physical_pixel_sizes
                vx = float(ps.X) if ps.X else 1.0
                vy = float(ps.Y) if ps.Y else 1.0
                vz = float(ps.Z) if ps.Z else 1.0
                self._state.voxel_size_xyz = (vx, vy, vz)
            except Exception:
                pass
        except Exception:
            pass

        self._ch_spin.blockSignals(True)
        self._ch_spin.setRange(0, max(0, nc - 1))
        self._ch_spin.setValue(0)
        self._ch_spin.blockSignals(False)

        self._z_slider.blockSignals(True)
        self._z_slider.setRange(0, max(0, nz - 1))
        self._z_slider.setValue(0)
        self._z_slider.blockSignals(False)
        self._state.current_z_slice = 0
        self._z_lbl.setText(f"Z: 0 / {nz - 1}")

        self._build_channel_mappings(nc, ch_names=ch_names)

        vx, vy, vz = self._state.voxel_size_xyz
        self._info_lbl.setText(
            f"Shape: {nz}Z × {nc}C × {ny}Y × {nx}X\n"
            f"Dtype: {arr.dtype}\n"
            f"Voxel: {vx:.3f} × {vy:.3f} × {vz:.3f} µm"
            + (f"\nSeries: {series_name}" if series_name else ""))
        if not series_name:
            self._file_lbl.ok(os.path.basename(path))
        self._refresh()
        self._state.notify("raw_image")

    @staticmethod
    def _read_tiff_or_image(path):
        try:
            from aicsimageio import AICSImage
            aics = AICSImage(path)
            nc = int(getattr(aics.dims,'C',1) or 1)
            stacks = []
            for c in range(nc):
                try: ch = aics.get_image_data("ZYX",C=c,T=0).astype("float32")
                except: ch = aics.get_image_data("ZYX",C=c,T=0,S=0).astype("float32")
                stacks.append(ch)
            r = np.stack(stacks, axis=1)
            m = [float(r[z,0].mean()) for z in range(min(8,r.shape[0]))]
            d = [m[i+1]-m[i] for i in range(len(m)-1)]
            sg = [1 if x>0 else -1 for x in d]
            osc = len(sg)>2 and all(sg[i]!=sg[i+1] for i in range(len(sg)-1))
            if not osc: print(f"[load] AICSImage OK {r.shape}"); return r
            print("[load] AICSImage oscillates")
        except Exception as e: print(f"[load] AICSImage failed: {e}")
        try:
            with tifffile.TiffFile(path) as tif:
                s = tif.series[0] if tif.series else None
                if s:
                    axes = s.axes.upper()
                    arr  = s.asarray().astype('float32')
                    if 'T' in axes:
                        arr = arr.take(0, axis=axes.index('T'))
                        axes = axes.replace('T','')
                    if arr.ndim == 4:
                        t = {'CZYX':(1,0,2,3),'ZYXC':(0,3,1,2)}.get(axes)
                        if t: arr = arr.transpose(t)
                    r = _to_zcyx(arr)
                    print(f"[load] tifffile {r.shape}"); return r
        except Exception as e: print(f"[load] tifffile failed: {e}")
        try:
            r = _to_zcyx(tifffile.imread(path).astype('float32'))
            print(f"[load] tifffile raw {r.shape}"); return r
        except Exception as e: print(f'[load] raw failed: {e}')
        return _to_zcyx(np.array(PILImage.open(path)).astype('float32'))

    def _render_composite(self, z_idx: int, proj: str) -> np.ndarray | None:
        img = self._state.preprocessed_image if self._state.has_preprocessed()               else self._state.raw_image
        if img is None:
            return None

        h, w = img.shape[2], img.shape[3]
        composite = np.zeros((h, w, 3), dtype=np.float32)

        for m in self._state.channel_mappings:
            if not m.get("enabled", True):
                continue
            c_idx   = m["data_channel_index"]
            color   = m.get("color",   "gray")
            opacity = m.get("opacity", 1.0)
            rgb     = _CMAP.get(color.lower(), (1.0, 1.0, 1.0))

            plane = self._state.get_display_slice(channel=c_idx,
                                                  z=z_idx, projection=proj)
            if plane is None:
                continue

            for ch_rgb, rgb_val in enumerate(rgb):
                composite[:, :, ch_rgb] += plane * rgb_val * opacity

        return np.clip(composite, 0.0, 1.0)

    def _refresh(self, _=None):
        if self._state.raw_image is None:
            return
        proj_idx = self._proj_combo.currentIndex()
        proj_map = {0: "max", 1: "mean", 2: "slice", 3: "composite"}
        proj = proj_map.get(proj_idx, "max")

        ch_idx = self._ch_spin.value()
        z_idx  = self._z_slider.value()

        is_composite = (proj == "composite")
        self._cmap_note.setVisible(is_composite)

        if is_composite:
            rgb = self._render_composite(z_idx, "max")
            if rgb is None:
                return
            self._canvas.show_image(rgb, title="Composite (all enabled channels)")
            return

        plane = self._state.get_display_slice(channel=ch_idx,
                                              z=z_idx, projection=proj)
        if plane is None:
            return

        cmap = self._cmap_combo.currentText()

        ch_names = getattr(self._state, "channel_names", None)
        if ch_names and ch_idx < len(ch_names):
            ch_label = ch_names[ch_idx]
        else:
            ch_label = f"Ch {ch_idx}"

        if proj == "slice":
            title = f"{ch_label}  |  Z = {z_idx}"
        else:
            title = f"{ch_label}  |  {self._proj_combo.currentText()}"

        self._canvas.show_image(plane, cmap=cmap, title=title)

    def _on_proj_change(self, idx):
        self._z_slider.setEnabled(idx == 2)
        self._refresh()

    def _on_z_change(self, v):
        nz = self._state.raw_image.shape[0] if self._state.raw_image is not None else 0
        self._z_lbl.setText(f"Z: {v} / {max(0,nz-1)}")
        self._state.current_z_slice = v
        self._refresh()

    def _on_state_change(self, what):
        if what in ("raw_image", "preprocessed", "channel_mappings"):
            self._rebuild_mc_rows()
            self._mc_refresh()
        if what in ("preprocessed", "raw_image") and self._state.has_image():
            self._refresh()


# Static image processing utilities used by the preprocessing pipeline
class Preprocessor:
    """Stateless preprocessing operations. Each method works on a 2-D float32 plane [0,1]."""

    @staticmethod
    def normalize(img: np.ndarray, pmin: float = 1.0, pmax: float = 99.9) -> np.ndarray:
        lo, hi = np.percentile(img, [pmin, pmax])
        if hi <= lo:
            return np.zeros_like(img, dtype=np.float32)
        return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)

    @staticmethod
    def subtract_bg_rolling_ball(img: np.ndarray, radius: int = 30) -> np.ndarray:
        """Morphological opening — estimates & removes background haze."""
        from skimage.morphology import opening as _opening, disk as _disk
        img_f = img.astype(np.float32)
        r = max(3, min(radius, min(img_f.shape) // 4, 40))
        bg = _opening(img_f, _disk(r))
        return np.clip(img_f - bg, 0.0, None).astype(np.float32)

    @staticmethod
    def subtract_bg_gaussian(img: np.ndarray, sigma: float = 30.0) -> np.ndarray:
        """Gaussian background estimate & subtraction."""
        img_f = img.astype(np.float32)
        bg = gaussian_filter(img_f, sigma=sigma)
        return np.clip(img_f - bg * 0.95, 0.0, None).astype(np.float32)

    @staticmethod
    def tophat(img: np.ndarray, radius: int = 6) -> np.ndarray:
        """White Top-Hat: keeps objects smaller than the structuring element."""
        from skimage.morphology import white_tophat, disk as _disk
        return white_tophat(img.astype(np.float32), _disk(radius)).astype(np.float32)

    @staticmethod
    def multiscale_tophat(img: np.ndarray, radius: int = 6) -> np.ndarray:
        """Multi-scale Top-Hat: fuses top-hat at r, r+3, r-2 for size robustness."""
        from skimage.morphology import white_tophat, disk as _disk
        img_f = img.astype(np.float32)
        radii = (max(2, radius - 2), radius, radius + 3)
        result = np.zeros_like(img_f)
        for r in radii:
            result = np.maximum(result, white_tophat(img_f, _disk(r)))
        return result

    @staticmethod
    def clahe(img: np.ndarray, clip_limit: float = 0.03) -> np.ndarray:
        from skimage.exposure import equalize_adapthist
        img_f = np.clip(img.astype(np.float32), 0.0, 1.0)
        return equalize_adapthist(img_f, clip_limit=clip_limit).astype(np.float32)

    @staticmethod
    def denoise_gaussian(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return gaussian_filter(img.astype(np.float32), sigma=sigma).astype(np.float32)

    @staticmethod
    def denoise_bilateral(img: np.ndarray, sigma_color: float = 0.12,
                          sigma_spatial: float = 2.0) -> np.ndarray:
        try:
            from skimage.restoration import denoise_bilateral as _db
            img_f = np.clip(img.astype(np.float32), 0.0, 1.0)
            return _db(img_f, sigma_color=sigma_color, sigma_spatial=sigma_spatial,
                       mode='reflect').astype(np.float32)
        except Exception:
            return gaussian_filter(img.astype(np.float32), sigma=sigma_spatial).astype(np.float32)


# Runs the preprocessing pipeline in a background thread
class PreprocessingWorker(BaseWorker):
    """Runs the preprocessing pipeline on one selected channel of the image."""

    def __init__(self, raw: np.ndarray, params: dict,
                 channel: int = 0, parent=None):
        super().__init__(parent)
        self._raw     = raw
        self._params  = params
        self._channel = channel

    def run_task(self):
        p   = self._params

        # processed channels are not overwritten
        arr = self._raw.astype(np.float32).copy()
        nz, nc = arr.shape[0], arr.shape[1]


        c = min(self._channel, nc - 1)

        total_steps  = nz
        current_step = 0
        self.signals.progress_percent.emit(0)
        self.signals.progress.emit(
            f"Processing channel {c} ({nz} Z-slices)…")

        for z in range(nz):
            plane = arr[z, c]

            # 1. Background subtraction
            if p.get("do_bg", False):
                self.signals.progress.emit(f"Background subtraction C={c} Z={z}…")
                method = p.get("bg_method", "gaussian")
                r      = p.get("bg_radius", 30)
                if method == "rolling_ball":
                    plane = Preprocessor.subtract_bg_rolling_ball(plane, r)
                else:
                    plane = Preprocessor.subtract_bg_gaussian(plane, float(r))

            # 2. Top-Hat — normalize to 0-1 first so the filter works correctly
            if p.get("do_tophat", False):
                self.signals.progress.emit(f"Top-Hat C={c} Z={z}…")
                plane = Preprocessor.normalize(plane)
                r = p.get("tophat_radius", 6)
                if p.get("tophat_multiscale", True):
                    plane = Preprocessor.multiscale_tophat(plane, r)
                else:
                    plane = Preprocessor.tophat(plane, r)

            # 3. Noise suppression
            if p.get("do_denoise", False):
                self.signals.progress.emit(f"Denoising C={c} Z={z}…")
                sig = p.get("denoise_sigma", 1.0)
                if p.get("denoise_bilateral", False):
                    plane = Preprocessor.normalize(plane)  # bilateral filter needs 0-1 input
                    plane = Preprocessor.denoise_bilateral(
                        plane, sigma_color=0.12, sigma_spatial=sig * 2)
                else:
                    plane = Preprocessor.denoise_gaussian(plane, sig)

            # 4. Percentile normalisation
            if p.get("do_norm", False):
                self.signals.progress.emit(f"Normalising C={c} Z={z}…")
                plane = Preprocessor.normalize(plane, p.get("pmin", 1.0), p.get("pmax", 99.9))

            # 5. CLAHE
            if p.get("do_clahe", False):
                self.signals.progress.emit(f"CLAHE C={c} Z={z}…")
                plane = Preprocessor.clahe(plane, p.get("clahe_clip", 0.03))

            arr[z, c] = plane

            current_step += 1
            percent = int((current_step / total_steps) * 100)
            self.signals.progress_percent.emit(percent)

        self.signals.progress_percent.emit(100)

        return {"array": arr, "channel": c}


# --- Preprocessing Tab ---
# Controls for BG subtraction, TopHat, denoising, normalisation, CLAHE
class PreprocessingTab(QWidget):
    """Preprocessing tab: left panel = pipeline controls, right = before/after preview."""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self._state  = state
        self._worker = None

        self._last_processed_array   = None
        self._last_processed_channel = 0
        self._build_ui()
        state.subscribe(self._on_state_change)

    # ── Build UI ──────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # ── LEFT PANEL ────────────────────────────────────────────────────
        ls, _, ll = make_scroll_widget()
        ls.setFixedWidth(300)

        hdr_row = QHBoxLayout()
        hdr_row.addStretch()
        hdr_row.addWidget(HelpButton("Preprocessing"))
        ll.addLayout(hdr_row)

        # Quick presets
        qg = section("⚡ Quick Presets")
        ql = QVBoxLayout(qg)
        btn_agg = QPushButton("★ Preset Aggregates (HA/mHTT)")
        btn_agg.setStyleSheet(BTN_RUN)
        btn_agg.setToolTip(
            "Recommended pipeline for protein-aggregate detection\n"
            "(matches engelse_versie 2.py):\n"
            "• Gaussian background subtraction (σ=50)\n"
            "• Gaussian denoise (σ=1.0)\n"
            "• Percentile normalisation OFF\n"
            "• Top-Hat OFF, CLAHE OFF\n\n"
            "Apply → store as 'preprocessed_aggregates'\n"
            "then select it in Aggregate Detection tab.")
        btn_agg.clicked.connect(self._set_recommended_aggregates)


        btn_cell = QPushButton("★ Preset Cell Body (CCT1)")
        btn_cell.setStyleSheet(BTN_PURPLE)
        btn_cell.setToolTip(
            "Optimal preprocessing before cell body segmentation:\n"
            "• Percentile normalisation (pmin=0.5, pmax=99.5)\n"
            "• Gaussian background subtraction (σ=40) — removes illumination gradient\n"
            "• Bilateral denoising (σ=0.8) — preserves cell borders\n"
            "• No Top-Hat (keeps cytoplasm texture intact)\n"
            "• No CLAHE\n\n"
            "After applying: use the derived channel 'preprocessed_cell_body'\n"
            "in the Cell Body & Nuclei Detection tab for best results.")
        btn_cell.clicked.connect(self._set_preset_cell_body)


        btn_nuc = QPushButton("★ Preset Nucleus")
        btn_nuc.setStyleSheet(BTN_LOAD)
        btn_nuc.setToolTip(
            "Recommended pipeline for nucleus segmentation\n"
            "(fluorescent nucleus stain — DAPI, Hoechst, etc.):\n"
            "• Percentile normalisation (pmin=1.0, pmax=99.0)\n"
            "• Gaussian background subtraction (σ=60)\n"
            "• Gaussian denoising (σ=0.5) — light smoothing only\n"
            "• Top-Hat OFF, CLAHE OFF (preserve nuclear texture)\n\n"
            "Apply → store as 'preprocessed_nucleus'\n"
            "then select in Cell Body & Nuclei Detection tab.")
        btn_nuc.clicked.connect(self._set_preset_nucleus)

        btn_reset = QPushButton("↺  Reset all")
        btn_reset.clicked.connect(self._reset_all)
        ql.addWidget(btn_agg)
        ql.addWidget(btn_cell)
        ql.addWidget(btn_nuc)
        ql.addWidget(btn_reset)
        ll.addWidget(qg)


        store_grp = section("💾 Store as Derived Channel")
        store_lyt = QVBoxLayout(store_grp)
        _store_info = QLabel(
            "After applying preprocessing, store the result as a named\n"
            "derived channel that the segmentation tabs can then use\n"
            "directly instead of the raw/preprocessed image.")
        _store_info.setWordWrap(True)
        _store_info.setStyleSheet("color:#7070a0;font-size:10px;")
        store_lyt.addWidget(_store_info)

        store_row = QHBoxLayout()
        self._store_name_edit = QLineEdit()
        self._store_name_edit.setPlaceholderText("e.g. preprocessed_cell_body")
        self._store_name_edit.setText("preprocessed_cell_body")
        self._store_name_edit.setToolTip(
            "Name for the derived channel.\n\n"
            "Use descriptive names like:\n"
            "  preprocessed_cell_body   — for CCT1 after cell body preset\n"
            "  preprocessed_nucleus     — for DAPI after nucleus preset\n\n"
            "This name will appear in the channel dropdowns of the\n"
            "Cell Body & Nuclei Detection tab.")
        store_row.addWidget(self._store_name_edit, stretch=1)

        self._store_ch_spin = QSpinBox()
        self._store_ch_spin.setRange(0, 0)
        self._store_ch_spin.setButtonSymbols(QSpinBox.NoButtons)
        self._store_ch_spin.setMinimumHeight(28)
        self._store_ch_spin.setToolTip("Which channel to store as a derived channel.")
        store_row.addWidget(QLabel("Ch:"))
        store_row.addWidget(self._store_ch_spin)
        store_lyt.addLayout(store_row)

        self._store_btn = QPushButton("📦  Store Preprocessed Channel")
        self._store_btn.setStyleSheet(BTN_PURPLE)
        self._store_btn.setEnabled(False)
        self._store_btn.setToolTip(
            "Saves the current preprocessed result for the selected channel\n"
            "as a derived channel under the given name.\n\n"
            "The stored channel will then appear in the channel selection\n"
            "dropdowns of the Cell Body & Nuclei Detection tab.")
        self._store_btn.clicked.connect(self._store_as_derived)
        store_lyt.addWidget(self._store_btn)
        ll.addWidget(store_grp)

        # Pipeline steps (scrollable)
        pg = section("🔧 Preprocessing Steps")
        pgl = QVBoxLayout(pg)
        inner_scroll = QScrollArea()
        inner_scroll.setWidgetResizable(True)
        inner_scroll.setFrameShape(QFrame.NoFrame)
        inner_w = QWidget()
        iv = QVBoxLayout(inner_w)
        iv.setSpacing(6)

        def _sep():
            f = QFrame(); f.setFrameShape(QFrame.HLine)
            f.setStyleSheet("color:#2a2a48;"); return f

        # ── Step 1: Percentile normalisation ──────────────────────────────
        self._chk_norm = QCheckBox("Percentile Normalisation")
        self._chk_norm.setChecked(True)
        iv.addWidget(self._chk_norm)
        self._spn_pmin = QDoubleSpinBox()
        self._spn_pmin.setRange(0, 49); self._spn_pmin.setValue(1.0)
        self._spn_pmin.setSingleStep(0.5)

        self._spn_pmin.setToolTip(
            "Lower percentile for intensity normalisation (0 – 49).\n\n"
            "Pixels at or below this percentile are mapped to 0.\n"
            "Increase to clip dim background; decrease to preserve faint signal.\n"
            "Typical value: 0.5 – 2.0.")
        iv.addLayout(_make_spinbox_row("pmin:", self._spn_pmin))
        self._spn_pmax = QDoubleSpinBox()
        self._spn_pmax.setRange(51, 100); self._spn_pmax.setValue(99.9)
        self._spn_pmax.setSingleStep(0.5)
        self._spn_pmax.setToolTip(
            "Upper percentile for intensity normalisation (51 – 100).\n\n"
            "Pixels at or above this percentile are mapped to 1.\n"
            "Lower to clip bright outliers (saturated pixels / hot spots).\n"
            "Typical value: 99.0 – 99.9.")
        iv.addLayout(_make_spinbox_row("pmax:", self._spn_pmax))
        iv.addWidget(_sep())

        # ── Step 2: Background subtraction ───────────────────────────────
        self._chk_bg = QCheckBox("Background Subtraction")
        iv.addWidget(self._chk_bg)
        bg_method_row = QHBoxLayout()
        bg_method_row.addWidget(QLabel("Method:"))
        self._cmb_bg_method = QComboBox()
        self._cmb_bg_method.addItems(["Gaussian", "Rolling Ball (morphological)"])
        bg_method_row.addWidget(self._cmb_bg_method)
        iv.addLayout(bg_method_row)
        self._spn_bg_radius = QSpinBox()
        self._spn_bg_radius.setRange(5, 500); self._spn_bg_radius.setValue(30)
        self._spn_bg_radius.setToolTip(
            "Radius (or Gaussian σ) for background subtraction in pixels.\n\n"
            "Should be larger than your cells/structures of interest.\n"
            "Increase to capture slower background variation.\n"
            "Decrease for tighter local background estimation.\n"
            "Typical range: 20 – 100 px (aim for ≥ 2× the largest cell diameter).")
        iv.addLayout(_make_spinbox_row("Radius/σ (px):", self._spn_bg_radius))
        iv.addWidget(_sep())

        # ── Step 3: Top-Hat filter ────────────────────────────────────────
        self._chk_tophat = QCheckBox("Top-Hat Filter")
        self._chk_tophat.setChecked(True)
        iv.addWidget(self._chk_tophat)
        th_mode_row = QHBoxLayout()
        th_mode_row.addWidget(QLabel("Mode:"))
        self._cmb_tophat_mode = QComboBox()
        self._cmb_tophat_mode.addItems(["Single", "Multi-scale (recommended)"])
        self._cmb_tophat_mode.setCurrentIndex(1)
        th_mode_row.addWidget(self._cmb_tophat_mode)
        iv.addLayout(th_mode_row)
        self._spn_tophat = QSpinBox()
        self._spn_tophat.setRange(1, 50); self._spn_tophat.setValue(6)
        self._spn_tophat.setToolTip(
            "Disk radius for white top-hat morphological filter in pixels.\n\n"
            "Enhances bright structures smaller than this radius.\n"
            "Set to approximately the radius of aggregates or puncta you want to detect.\n"
            "Increase for larger structures; decrease for fine puncta.\n"
            "Typical range: 3 – 15 px.")
        iv.addLayout(_make_spinbox_row("Radius (px):", self._spn_tophat))
        ms_lbl = QLabel("Multi-scale: uses r, r+3, r-2 simultaneously")
        ms_lbl.setStyleSheet("color:#606080;font-size:10px;")
        iv.addWidget(ms_lbl)
        iv.addWidget(_sep())

        # ── Step 4: Noise suppression ─────────────────────────────────────
        self._chk_denoise = QCheckBox("Noise Suppression")
        iv.addWidget(self._chk_denoise)
        dn_mode_row = QHBoxLayout()
        dn_mode_row.addWidget(QLabel("Method:"))
        self._cmb_denoise_mode = QComboBox()
        self._cmb_denoise_mode.addItems(["Gaussian", "Bilateral (preserves edges) ★"])
        self._cmb_denoise_mode.setCurrentIndex(1)
        dn_mode_row.addWidget(self._cmb_denoise_mode)
        iv.addLayout(dn_mode_row)
        self._spn_denoise_sigma = QDoubleSpinBox()
        self._spn_denoise_sigma.setRange(0.1, 10.0)
        self._spn_denoise_sigma.setValue(1.0)
        self._spn_denoise_sigma.setSingleStep(0.1)
        self._spn_denoise_sigma.setToolTip(
            "Smoothing strength (σ) for Gaussian / bilateral noise suppression.\n\n"
            "Higher σ → stronger smoothing → removes more noise but blurs edges.\n"
            "Lower σ  → lighter smoothing → preserves fine structure.\n"
            "Default: 1.0.  Typical range: 0.5 – 3.0.\n"
            "Use ~0.5–1.0 for fine puncta; ~1.5–3.0 for diffuse staining.")
        iv.addLayout(_make_spinbox_row("Sigma:", self._spn_denoise_sigma))
        iv.addWidget(_sep())

        # ── Step 5: CLAHE ─────────────────────────────────────────────────
        self._chk_clahe = QCheckBox("CLAHE Contrast Enhancement")
        iv.addWidget(self._chk_clahe)
        self._spn_clahe_clip = QDoubleSpinBox()
        self._spn_clahe_clip.setRange(0.001, 1.0)
        self._spn_clahe_clip.setValue(0.03)
        self._spn_clahe_clip.setSingleStep(0.005)
        self._spn_clahe_clip.setToolTip(
            "CLAHE clip limit — controls contrast enhancement strength.\n\n"
            "Lower values (e.g. 0.01) → stronger local contrast enhancement.\n"
            "Higher values (e.g. 0.10) → gentler enhancement, less noise amplification.\n"
            "Default: 0.03.  Typical range: 0.01 – 0.10.\n"
            "Reduce if the result looks grainy; increase if enhancement is too subtle.")
        iv.addLayout(_make_spinbox_row("Clip limit:", self._spn_clahe_clip))

        iv.addStretch()
        inner_w.setLayout(iv)
        inner_scroll.setWidget(inner_w)
        pgl.addWidget(inner_scroll)
        ll.addWidget(pg)


        chan_sel_grp = section("📺 Channel to Preview & Process")
        chan_sel_lyt = QVBoxLayout(chan_sel_grp)
        _chan_info = QLabel(
            "Select which channel to preview and preprocess.\n"
            "Each channel can be preprocessed independently\n"
            "with different settings and stored separately.")
        _chan_info.setWordWrap(True)
        _chan_info.setStyleSheet("color:#7070a0;font-size:10px;")
        chan_sel_lyt.addWidget(_chan_info)

        self._ch_combo_prep = QComboBox()
        self._ch_combo_prep.setToolTip(
            "Select which channel to preview and preprocess.\n\n"
            "Channel names come from Image Viewer → Channel Mapping.\n"
            "Tip: select CCT1 → apply Cell Body preset → store.\n"
            "Then select Nucleus → apply Nucleus preset → store.")
        self._ch_combo_prep.currentIndexChanged.connect(self._on_channel_changed)
        chan_sel_lyt.addWidget(self._ch_combo_prep)
        ll.addWidget(chan_sel_grp)

        # Keep old spinbox hidden for internal compatibility
        self._prev_ch_spin = QSpinBox()
        self._prev_ch_spin.setRange(0, 0)
        self._prev_ch_spin.setVisible(False)

        # Apply / Reset buttons
        self._run_btn   = QPushButton("▶  Apply Preprocessing (selected channel)")
        self._run_btn.setStyleSheet(BTN_RUN)
        self._run_btn.setToolTip(
            "Apply the preprocessing pipeline to the currently selected channel.\n\n"
            "Only the selected channel is modified — other channels are unchanged.\n"
            "After applying, click '📦 Store' to save as a derived channel.")
        self._reset_btn = QPushButton("↺  Reset selected channel to raw")
        ll.addWidget(self._run_btn)
        ll.addWidget(self._reset_btn)

        self._prog     = QProgressBar()
        self._prog.setRange(0, 0)
        self._prog.setVisible(False)
        self._stat_lbl = StatusLabel()
        ll.addWidget(self._prog)
        ll.addWidget(self._stat_lbl)
        ll.addStretch()
        root.addWidget(ls)

        # ── RIGHT PANEL — dual canvas ────────────────────────────────────
        right_w   = QWidget()
        right_lyt = QVBoxLayout(right_w)
        right_lyt.setContentsMargins(0, 0, 0, 0)
        right_lyt.setSpacing(4)

        # ── Colormap selector ─────────────────────────────────────────────────
        _cmap_w   = QWidget()
        _cmap_lay = QHBoxLayout(_cmap_w)
        _cmap_lay.setContentsMargins(4, 2, 4, 2)
        _cmap_lbl = QLabel("Colormap:")
        _cmap_lbl.setStyleSheet("color:#a0a0c0;font-size:11px;")
        _cmap_lay.addWidget(_cmap_lbl)
        self._cmb_cmap = QComboBox()
        self._cmb_cmap.addItems(["gray", "hot", "magma", "viridis", "inferno", "plasma"])
        self._cmb_cmap.setCurrentIndex(1)  # default: hot
        self._cmb_cmap.currentIndexChanged.connect(self._refresh_canvases)
        _cmap_lay.addWidget(self._cmb_cmap)
        _cmap_lay.addStretch()
        right_lyt.addWidget(_cmap_w)

        canvases_w   = QWidget()
        canvases_lyt = QHBoxLayout(canvases_w)
        canvases_lyt.setContentsMargins(0, 0, 0, 0)
        canvases_lyt.setSpacing(4)

        # Original
        orig_w = QWidget()
        orig_l = QVBoxLayout(orig_w)
        orig_l.setContentsMargins(0, 0, 0, 0)
        orig_title = QLabel("Original")
        orig_title.setAlignment(Qt.AlignCenter)
        orig_title.setStyleSheet("color:#8080c0;font-size:11px;font-weight:600;")
        self._canvas_orig = ImageCanvas(figsize=(6, 5))
        self._nav_orig    = NavigationToolbar(self._canvas_orig, self)
        self._nav_orig.setStyleSheet(_viewer_tb_qss())
        orig_l.addWidget(orig_title)
        orig_l.addWidget(self._nav_orig)
        orig_l.addWidget(self._canvas_orig, stretch=1)

        # Processed
        proc_w = QWidget()
        proc_l = QVBoxLayout(proc_w)
        proc_l.setContentsMargins(0, 0, 0, 0)
        proc_title = QLabel("After Preprocessing")
        proc_title.setAlignment(Qt.AlignCenter)
        proc_title.setStyleSheet("color:#80ff80;font-size:11px;font-weight:600;")
        self._canvas_proc = ImageCanvas(figsize=(6, 5))
        self._nav_proc    = NavigationToolbar(self._canvas_proc, self)
        self._nav_proc.setStyleSheet(_viewer_tb_qss())
        proc_l.addWidget(proc_title)
        proc_l.addWidget(self._nav_proc)
        proc_l.addWidget(self._canvas_proc, stretch=1)

        canvases_lyt.addWidget(orig_w,  stretch=1)
        canvases_lyt.addWidget(proc_w,  stretch=1)
        right_lyt.addWidget(canvases_w, stretch=1)
        root.addWidget(right_w, stretch=1)

        # ── Connections ───────────────────────────────────────────────────
        self._run_btn.clicked.connect(self._run_preprocessing)
        self._reset_btn.clicked.connect(self._reset)

    # ── Helpers ───────────────────────────────────────────────────────────
    def _get_params(self) -> dict:
        return {
            "do_norm":          self._chk_norm.isChecked(),
            "pmin":             self._spn_pmin.value(),
            "pmax":             self._spn_pmax.value(),
            "do_bg":            self._chk_bg.isChecked(),
            "bg_method":        ("rolling_ball"
                                 if self._cmb_bg_method.currentIndex() == 1
                                 else "gaussian"),
            "bg_radius":        self._spn_bg_radius.value(),
            "do_tophat":        self._chk_tophat.isChecked(),
            "tophat_radius":    self._spn_tophat.value(),
            "tophat_multiscale": self._cmb_tophat_mode.currentIndex() == 1,
            "do_denoise":       self._chk_denoise.isChecked(),
            "denoise_bilateral": self._cmb_denoise_mode.currentIndex() == 1,
            "denoise_sigma":    self._spn_denoise_sigma.value(),
            "do_clahe":         self._chk_clahe.isChecked(),
            "clahe_clip":       self._spn_clahe_clip.value(),
        }

    def _on_channel_changed(self, _=None):

        self._sync_ch_spinbox()
        self._show_original()
        self._show_processed()
        # Also update store name suggestion based on biological label
        idx = self._ch_combo_prep.currentIndex()
        if 0 <= idx < len(self._state.channel_mappings):
            m   = self._state.channel_mappings[idx]
            bio = m.get("biological_label", "—") or "—"
            if bio not in ("—", ""):
                safe = bio.lower().replace(" ", "_").replace("(", "").replace(")", "")
                self._store_name_edit.setText(f"preprocessed_{safe}")

    def _sync_ch_spinbox(self):

        idx = self._ch_combo_prep.currentIndex()
        if idx >= 0:
            self._prev_ch_spin.setValue(idx)
            self._store_ch_spin.setValue(idx)

    def _ch_index(self) -> int:

        idx = self._ch_combo_prep.currentIndex()
        return max(0, idx)

    def _show_original(self):
        img = self._state.raw_image
        if img is None: return
        c  = min(self._ch_index(), img.shape[1] - 1)
        pl = np.max(img[:, c], axis=0).astype(np.float32)
        lo, hi = float(np.percentile(pl, 1)), float(np.percentile(pl, 99))
        if hi > lo: pl = np.clip((pl - lo) / (hi - lo), 0, 1)
        ch_name = self._get_ch_display_name(c)
        self._canvas_orig.show_image(pl, cmap=self._cmb_cmap.currentText(),
            title=f"Original — {ch_name} (max proj)")

    def _show_processed(self):

        # This means the right canvas only shows what was just processed,
        # and all other tabs always see the clean raw image.
        buf = getattr(self, "_last_processed_array", None)
        c   = getattr(self, "_last_processed_channel", self._ch_index())

        # If no buffer yet (nothing processed in this session), show a placeholder
        if buf is None:
            self._canvas_proc.clear_canvas()
            ax = self._canvas_proc.ax
            ax.set_facecolor("#0a0a14")
            ax.text(0.5, 0.5,
                    "Apply preprocessing to see result here.\n"
                    "Other tabs always see the original image.",
                    color="#505070", ha="center", va="center",
                    fontsize=10, transform=ax.transAxes, wrap=True)
            self._canvas_proc.draw_idle()
            return

        c  = min(c, buf.shape[1] - 1)
        pl = np.max(buf[:, c], axis=0).astype(np.float32)
        lo, hi = float(np.percentile(pl, 1)), float(np.percentile(pl, 99))
        if hi > lo: pl = np.clip((pl - lo) / (hi - lo), 0, 1)
        ch_name = self._get_ch_display_name(c)
        self._canvas_proc.show_image(pl, cmap=self._cmb_cmap.currentText(),
            title=f"Preview result — {ch_name}  ← not yet stored")

    def _update_preview_only(self):

        # Right canvas stays as-is until user applies preprocessing again
        self._show_original()

    def _refresh_canvases(self):
        """Redraw both canvases with the currently selected colormap."""
        self._show_original()
        self._show_processed()

    def _set_recommended_aggregates(self):
        """Load recommended settings for protein-aggregate detection."""
        self._chk_norm.setChecked(False)                 # Norm OFF
        self._chk_bg.setChecked(True)
        self._cmb_bg_method.setCurrentIndex(0)           # Gaussian
        self._spn_bg_radius.setValue(50)
        self._chk_tophat.setChecked(False)
        self._chk_denoise.setChecked(True)
        self._cmb_denoise_mode.setCurrentIndex(0)        # Gaussian
        self._spn_denoise_sigma.setValue(1.0)
        self._chk_clahe.setChecked(False)
        self._store_name_edit.setText("preprocessed_aggregates")
        self._stat_lbl.info(
            "Aggregate preset loaded. Apply → store as 'preprocessed_aggregates'.")


    def _set_preset_cell_body(self):
        """Settings optimised for cell body segmentation input (CCT1 / cytoplasm stain).
        Goal: give Cellpose cyto2 a clean, background-free image where cell"""
        self._chk_norm.setChecked(True)
        self._spn_pmin.setValue(0.5)
        self._spn_pmax.setValue(99.5)
        self._chk_bg.setChecked(True)
        self._cmb_bg_method.setCurrentIndex(0)    # Gaussian
        self._spn_bg_radius.setValue(40)
        self._chk_tophat.setChecked(False)         # keep cytoplasm texture
        self._chk_denoise.setChecked(True)
        self._cmb_denoise_mode.setCurrentIndex(1)  # Bilateral — preserves borders
        self._spn_denoise_sigma.setValue(0.8)
        self._chk_clahe.setChecked(False)

        self._store_name_edit.setText("preprocessed_cell_body")
        self._stat_lbl.info(
            "Cell body preset loaded. Apply → then store as 'preprocessed_cell_body'.")


    def _set_preset_nucleus(self):
        """Settings optimised for nucleus segmentation (fluorescent nucleus stain).
        Goal: give Cellpose nuclei a uniform, clean nuclear signal without the"""
        self._chk_norm.setChecked(True)
        self._spn_pmin.setValue(1.0)
        self._spn_pmax.setValue(99.0)
        self._chk_bg.setChecked(True)
        self._cmb_bg_method.setCurrentIndex(0)    # Gaussian
        self._spn_bg_radius.setValue(60)
        self._chk_tophat.setChecked(False)         # top-hat hurts nuclear texture
        self._chk_denoise.setChecked(True)
        self._cmb_denoise_mode.setCurrentIndex(0)  # Gaussian — lightest possible
        self._spn_denoise_sigma.setValue(0.5)
        self._chk_clahe.setChecked(False)          # CLAHE damages nuclear texture

        self._store_name_edit.setText("preprocessed_nucleus")
        self._stat_lbl.info(
            "Nucleus preset loaded. Apply → then store as 'preprocessed_nucleus'.")

    def _reset_all(self):
        self._chk_norm.setChecked(False)
        self._chk_bg.setChecked(False)
        self._chk_tophat.setChecked(False)
        self._chk_denoise.setChecked(False)
        self._chk_clahe.setChecked(False)


    def _store_as_derived(self):
        """Save the preprocessing result as a named derived channel. Reads from local buffer (_last_processed_array),"""
        buf = getattr(self, "_last_processed_array", None)
        c   = getattr(self, "_last_processed_channel", self._ch_index())
        if buf is None:
            QMessageBox.warning(self, "No data",
                "Apply preprocessing first, then store the result.")
            return
        name = self._store_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "No name",
                "Enter a channel name before storing.")
            return
        c   = min(c, buf.shape[1] - 1)
        vol = buf[:, c, :, :].astype(np.float32)
        self._state.set_derived_channel(
            key    = name,
            mask   = vol,
            source = f"preprocessing_tab_ch{c}",
        )
        ch_name = self._get_ch_display_name(c)
        self._stat_lbl.ok(
            f"\u2713  '{ch_name}' stored as '{name}'."
            f"\nAvailable in Cell Body & Nuclei Detection \u2192 channel dropdown.")
        # Update right canvas title to confirm storage
        pl = np.max(vol, axis=0).astype(np.float32)
        lo, hi = float(np.percentile(pl, 1)), float(np.percentile(pl, 99))
        if hi > lo: pl = np.clip((pl - lo) / (hi - lo), 0, 1)
        self._canvas_proc.show_image(pl, cmap=self._cmb_cmap.currentText(),
            title=f"\u2713 Stored as '{name}'")

    def _run_preprocessing(self):
        if not self._state.has_image():
            QMessageBox.warning(self, "No image", "Load an image in the Image Viewer first.")
            return
        self._run_btn.setEnabled(False)
        self._prog.setVisible(True)
        self._prog.setRange(0, 100)
        self._prog.setValue(0)
        self._prog.setFormat("%p%")
        self._stat_lbl.info("Running preprocessing pipeline…")

        selected_ch = self._ch_index()
        self._worker = PreprocessingWorker(
            raw       = self._state.raw_image,
            params    = self._get_params(),
            channel   = selected_ch,
            parent    = self)
        self._worker.signals.progress.connect(self._stat_lbl.info)
        self._worker.signals.progress_percent.connect(self._prog.setValue)
        self._worker.signals.result.connect(self._on_done)
        self._worker.signals.error.connect(self._on_err)
        self._worker.signals.finished.connect(self._on_fin)
        self._worker.start()

    def _on_done(self, result):
        # We do NOT write into state.preprocessed_image here.
        # Writing preprocessed data into preprocessed_image would make ALL other
        # tabs (Image Viewer, Cell Body, Nucleus) see the modified channel, which
        # is wrong — the user only wants to preprocess for segmentation input.
        #
        # Instead we keep the result in a local buffer and show it in the right
        # canvas only.  The '📦 Store' button then saves it as a derived channel.
        if isinstance(result, dict):
            arr          = result["array"]
            processed_ch = result.get("channel", self._ch_index())
        else:
            arr          = result
            processed_ch = self._ch_index()

        # Store the processed plane in a local buffer for preview + store
        self._last_processed_array   = arr
        self._last_processed_channel = processed_ch

        ch_name = self._get_ch_display_name(processed_ch)

        # Show the result in the right canvas (reads from buffer, not state)
        self._show_processed()

        self._stat_lbl.ok(
            f"✓  '{ch_name}' preprocessed — preview shown right.\n"
            f"Click '📦 Store' to save as a derived channel for segmentation.\n"
            f"Other tabs are NOT affected until you Store.")
        self._store_btn.setEnabled(True)

    def _get_ch_display_name(self, ch_idx: int) -> str:

        mappings = self._state.channel_mappings
        if 0 <= ch_idx < len(mappings):
            m   = mappings[ch_idx]
            bio = m.get("biological_label", "—") or "—"
            if bio not in ("—", ""):
                return bio
            return m.get("channel_name", f"Ch {ch_idx}") or f"Ch {ch_idx}"
        return f"Ch {ch_idx}"

    def _on_err(self, msg):
        QMessageBox.critical(self, "Preprocessing Error", msg)
        self._stat_lbl.err("Preprocessing failed.")

    def _on_fin(self):
        self._run_btn.setEnabled(True)
        self._prog.setRange(0, 0)
        self._prog.setFormat("")
        self._prog.setVisible(False)

    def _reset(self):

        # preprocessed_image so other tabs are unaffected
        self._last_processed_array   = None
        self._last_processed_channel = self._ch_index()
        self._store_btn.setEnabled(False)
        self._show_processed()   # shows placeholder message
        ch_name = self._get_ch_display_name(self._ch_index())
        self._stat_lbl.ok(f"Preview cleared for '{ch_name}'.")

    def _on_state_change(self, what):
        if what in ("raw_image", "preprocessed", "channel_mappings"):
            if self._state.raw_image is not None:
                nz, nc = self._state.raw_image.shape[:2]
                self._prev_ch_spin.setRange(0, max(0, nc - 1))
                self._store_ch_spin.setRange(0, max(0, nc - 1))

                self._rebuild_ch_combo()
            self._show_original()
            if what == "preprocessed":
                self._show_processed()
                self._store_btn.setEnabled(self._state.has_preprocessed())

    def _rebuild_ch_combo(self):

        self._ch_combo_prep.blockSignals(True)
        prev_idx = self._ch_combo_prep.currentIndex()
        self._ch_combo_prep.clear()
        mappings = self._state.channel_mappings
        if not mappings:
            img = self._state.raw_image
            if img is not None:
                for c in range(img.shape[1]):
                    self._ch_combo_prep.addItem(f"Channel {c}")
        else:
            for m in mappings:
                bio  = m.get("biological_label", "—") or "—"
                name = m.get("channel_name", "") or f"Ch {m['data_channel_index']}"
                col  = m.get("color", "gray")
                cidx = m["data_channel_index"]
                lbl  = bio if bio not in ("—", "") else name
                self._ch_combo_prep.addItem(f"Ch {cidx}: {lbl}  [{col}]")
        # restore previous selection if possible
        nc = self._ch_combo_prep.count()
        if nc > 0:
            self._ch_combo_prep.setCurrentIndex(
                min(max(prev_idx, 0), nc - 1))
        self._ch_combo_prep.blockSignals(False)
        self._sync_ch_spinbox()


# --- Results Tab ---
# Collects and exports results from all analysis tabs
class ResultsTab(QWidget):

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self._state = state
        self._build_ui()
        state.subscribe(self._on_state_change)

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        ls, _, ll = make_scroll_widget()
        ls.setFixedWidth(230)

        sg = section("Tool Results")
        sg_hdr = QHBoxLayout(); sg_hdr.addStretch()
        sg_hdr.addWidget(HelpButton("Results"))
        sl = QVBoxLayout(sg)
        sl.addLayout(sg_hdr)
        self._tool_combo = QComboBox()
        self._tool_combo.addItem("(no results yet)")
        sl.addWidget(QLabel("Select tool:"))
        sl.addWidget(self._tool_combo)
        ll.addWidget(sg)

        eg = section("Export")
        el = QVBoxLayout(eg)
        self._csv_btn  = QPushButton("Export CSV")
        self._json_btn = QPushButton("Export JSON")
        for b in [self._csv_btn, self._json_btn]:
            b.setEnabled(False)
            el.addWidget(b)
        ll.addWidget(eg)

        self._summary_lbl = QTextEdit()
        self._summary_lbl.setReadOnly(True)
        self._summary_lbl.setPlaceholderText("Run a tool to see results here…")
        ll.addWidget(self._summary_lbl)
        ll.addStretch()
        root.addWidget(ls)

        self._table = QTableWidget()
        self._table.setStyleSheet(
            "QTableWidget{background:#0a0a14;color:#c0c0e0;"
            "gridline-color:#1a1a30;}"
            "QHeaderView::section{background:#14142a;color:#8080c0;}")
        root.addWidget(self._table, stretch=1)

        self._tool_combo.currentTextChanged.connect(self._show_tool_result)
        self._csv_btn.clicked.connect(self._export_csv)
        self._json_btn.clicked.connect(self._export_json)

    def _on_state_change(self, what):
        if what != "results":
            return
        current = self._tool_combo.currentText()
        self._tool_combo.blockSignals(True)
        self._tool_combo.clear()
        for k in self._state.results_dict:
            self._tool_combo.addItem(k)
        if current in self._state.results_dict:
            self._tool_combo.setCurrentText(current)
        self._tool_combo.blockSignals(False)
        self._tool_combo.currentTextChanged.emit(self._tool_combo.currentText())
        self._csv_btn.setEnabled(bool(self._state.results_dict))
        self._json_btn.setEnabled(bool(self._state.results_dict))

    def _show_tool_result(self, tool_name: str):
        if tool_name not in self._state.results_dict:
            return
        data = self._state.results_dict[tool_name]

        df = data.get("dataframe", None)
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            self._table.setRowCount(len(df))
            self._table.setColumnCount(len(df.columns))
            self._table.setHorizontalHeaderLabels(list(df.columns))
            self._table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeToContents)
            for r, row in df.iterrows():
                for c, col in enumerate(df.columns):
                    item = QTableWidgetItem(str(row[col]))
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self._table.setItem(r, c, item)
        else:
            self._table.setRowCount(0)
            self._table.setColumnCount(0)

        summary = data.get("summary", "")
        self._summary_lbl.setPlainText(summary)

    def _current_df(self) -> pd.DataFrame | None:
        tool = self._tool_combo.currentText()
        if tool not in self._state.results_dict:
            return None
        return self._state.results_dict[tool].get("dataframe", None)

    def _export_csv(self):
        df = self._current_df()
        if df is None:
            QMessageBox.warning(self, "No data", "No table data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV (*.csv)")
        if path:
            df.to_csv(path, index=False)
            QMessageBox.information(self, "Exported", f"Saved to:\n{path}")

    def _export_json(self):
        tool = self._tool_combo.currentText()
        if tool not in self._state.results_dict:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save JSON", "", "JSON (*.json)")
        if path:
            data = self._state.results_dict[tool]
            safe = {}
            for k, v in data.items():
                if isinstance(v, pd.DataFrame):
                    safe[k] = v.to_dict(orient="records")
                elif isinstance(v, (str, int, float, list, dict, bool)):
                    safe[k] = v
                elif isinstance(v, np.ndarray):
                    safe[k] = f"<ndarray shape={v.shape}>"
            with open(path, "w") as f:
                json.dump(safe, f, indent=2)
            QMessageBox.information(self, "Exported", f"Saved to:\n{path}")


def _populate_channel_combo(combo: QComboBox, state: AppState,
                             preferred_label: str = "",
                             active_only: bool = False) -> None:
    combo.blockSignals(True)
    combo.clear()
    pool = ([m for m in state.channel_mappings if m.get("enabled", True)]
            if active_only else state.channel_mappings)
    if not pool:
        combo.addItem("(no channels — load an image first)")
        combo.blockSignals(False)
        return
    best_idx = 0
    for i, m in enumerate(pool):
        bio   = m.get("biological_label", "—") or "—"
        name  = m.get("channel_name",     "")
        color = m.get("color",            "gray")
        cidx  = m["data_channel_index"]
        off   = "  [off]" if not m.get("enabled", True) else ""
        label_part = bio if bio not in ("—", "") else f"Ch {cidx}"
        combo.addItem(f"{label_part}  ({color}) — {name or f'Ch {cidx}'}{off}")
        if preferred_label and bio == preferred_label:
            best_idx = i
    combo.setCurrentIndex(best_idx)
    combo.blockSignals(False)

def _channel_index_from_combo(combo: QComboBox, state: AppState,
                               active_only: bool = False) -> int | None:
    pool = ([m for m in state.channel_mappings if m.get("enabled", True)]
            if active_only else state.channel_mappings)
    i = combo.currentIndex()
    if 0 <= i < len(pool):
        return pool[i]["data_channel_index"]
    return None

def _label_from_combo(combo: QComboBox, state: AppState,
                      active_only: bool = False) -> str:
    pool = ([m for m in state.channel_mappings if m.get("enabled", True)]
            if active_only else state.channel_mappings)
    i = combo.currentIndex()
    if 0 <= i < len(pool):
        return pool[i].get("biological_label", "—") or "—"
    return "—"

def _cb_build_composite_for_segmentation(state: "AppState") -> np.ndarray | None:
    img = state.preprocessed_image if state.has_preprocessed() else state.raw_image
    if img is None:
        return None
    nz, nc, ny, nx = img.shape
    _COLOR_MAP = {
        "red": (1.0,0.0,0.0), "green": (0.0,1.0,0.0), "blue": (0.0,0.0,1.0),
        "cyan": (0.0,1.0,1.0), "magenta": (1.0,0.0,1.0), "yellow": (1.0,1.0,0.0),
        "white": (1.0,1.0,1.0), "gray": (1.0,1.0,1.0), "orange": (1.0,0.5,0.0),
    }
    composite = np.zeros((ny, nx, 3), dtype=np.float32)
    processed_any = False
    for m in state.channel_mappings:
        if not m.get("enabled", True):
            continue
        cidx = m.get("data_channel_index", None)
        if cidx is None or cidx >= nc:
            continue
        color_name = m.get("color", "gray").lower()
        rgb = np.array(_COLOR_MAP.get(color_name, (1.0,1.0,1.0)), dtype=np.float32)
        vol = img[:, cidx, :, :].astype(np.float32)
        plane = np.max(vol, axis=0)
        lo, hi = float(np.percentile(plane, 2)), float(np.percentile(plane, 98))
        plane_n = np.clip((plane - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        composite += plane_n[:, :, np.newaxis] * rgb[np.newaxis, np.newaxis, :]
        processed_any = True
    if not processed_any:
        return np.zeros((ny, nx), dtype=np.uint8)
    composite = np.clip(composite, 0.0, 1.0)
    from PIL import Image as _PIL
    pil_gray = _PIL.fromarray((composite * 255).astype(np.uint8), mode="RGB").convert("L")
    return np.array(pil_gray, dtype=np.uint8)

def _normalize_for_cellpose(plane: np.ndarray) -> np.ndarray:
    # Pure minimal normalization so Cellpose receives float32 in [0,1].
    # NO background subtraction, NO denoising, NO CLAHE, NO adaptive logic.
    # All preprocessing must be done explicitly by the user in the
    # Preprocessing tab, which stores derived channels for segmentation.
    img = plane.astype(np.float32)
    lo  = float(img.min())
    hi  = float(img.max())
    if hi > lo:
        return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return np.zeros_like(img, dtype=np.float32)


def _seg_preprocess(plane: np.ndarray, model_type: str = "cyto2") -> np.ndarray:
    # Kept for backward-compat with 3D workers — now just normalises, no extras.
    return _normalize_for_cellpose(plane)


def _cb_preprocess(image_np: np.ndarray) -> np.ndarray:
    # kept for backward compatibility
    return image_np


# ── Cellpose model cache ───────────────────────────────────────────────────────
_CELLPOSE_MODEL_CACHE: dict = {}


def _cb_postprocess(masks: np.ndarray, min_size: int = 0) -> np.ndarray:
    if masks is None:
        return None
    if min_size <= 0:
        return masks.astype(np.int32)  # No filtering — matches original exactly

    result = masks.astype(np.int32).copy()
    for prop in regionprops(masks):
        if prop.area < min_size:
            result[masks == prop.label] = 0
    return result

def _cb_calculate_exact_boundaries(mask_array: np.ndarray) -> np.ndarray | None:
    from scipy import ndimage as ndi_inner
    if mask_array is None or mask_array.size == 0:
        return None
    if mask_array.ndim != 2:
        return None
    all_boundaries = np.zeros_like(mask_array, dtype=bool)
    struct = ndi_inner.generate_binary_structure(2, 1)  # 4-connectivity
    for cell_id in np.unique(mask_array):
        if cell_id == 0:
            continue
        single = mask_array == cell_id
        eroded = ndi_inner.binary_erosion(single, structure=struct, border_value=0)
        all_boundaries |= (single ^ eroded)
    return all_boundaries


def _require_image(state, parent):
    if not state.has_image():
        QMessageBox.warning(parent, "No image", "Load an image in the Image Viewer first.")
        return False
    return True

def _populate_channel_combo(combo: QComboBox, state: AppState,
                             preferred_label: str = "",
                             active_only: bool = False) -> None:
    combo.blockSignals(True)
    combo.clear()
    pool = ([m for m in state.channel_mappings if m.get("enabled", True)]
            if active_only else state.channel_mappings)
    if not pool:
        combo.addItem("(no channels — load an image first)")
        combo.blockSignals(False)
        return
    best_idx = 0
    for i, m in enumerate(pool):
        bio   = m.get("biological_label", "—") or "—"
        name  = m.get("channel_name",     "")
        color = m.get("color",            "gray")
        cidx  = m["data_channel_index"]
        off   = "  [off]" if not m.get("enabled", True) else ""
        label_part = bio if bio not in ("—", "") else f"Ch {cidx}"
        combo.addItem(f"{label_part}  ({color}) — {name or f'Ch {cidx}'}{off}")
        if preferred_label and bio == preferred_label:
            best_idx = i
    combo.setCurrentIndex(best_idx)
    combo.blockSignals(False)


def _populate_channel_combo_with_derived(
        combo, state,
        preferred_label="",
        derived_prefix="preprocessed"):
    """Populate a channel combo with raw channels AND derived preprocessed channels.
    Raw channels first, then derived channels with a star marker.
    """
    combo.blockSignals(True)
    combo.clear()
    best_idx = 0
    total    = 0

    for m in state.channel_mappings:
        bio   = m.get("biological_label", "—") or "—"
        name  = m.get("channel_name",     "")
        color = m.get("color",            "gray")
        cidx  = m["data_channel_index"]
        off   = "  [off]" if not m.get("enabled", True) else ""
        label_part = bio if bio not in ("—", "") else f"Ch {cidx}"
        combo.addItem(f"{label_part}  ({color}) — {name or f'Ch {cidx}'}{off}")
        if preferred_label and bio == preferred_label:
            best_idx = total
        total += 1


    for key in sorted(state.channels.get("derived", {}).keys()):
        if not key.startswith(derived_prefix):
            continue
        display = key.replace("_", " ").title()
        combo.addItem(f"★ {display}  [preprocessed]")
        if preferred_label.lower().replace(" ", "_") == key:
            best_idx = total
        total += 1

    if total == 0:
        combo.addItem("(no channels — load an image first)")
    else:
        combo.setCurrentIndex(best_idx)
    combo.blockSignals(False)


def _get_plane_from_combined_combo(combo, state, mode="max", z_idx=0,
                                    derived_prefix="preprocessed"):
    """Return the 2-D float32 plane from a combined raw+derived combo. Uses sorted key list — identical order to populate function.
    """
    idx   = combo.currentIndex()
    n_raw = len(state.channel_mappings)

    if idx < n_raw:
        ch_idx = state.channel_mappings[idx]["data_channel_index"]
        proj_map = {"slice": "slice", "max": "max", "mean": "mean",
                    "Max Projection": "max", "Mean Projection": "mean",
                    "Slice (Z)": "slice", "All slices (Z)": "max"}
        proj = proj_map.get(mode, "max")
        return state.get_display_slice(channel=ch_idx, z=z_idx, projection=proj)


    derived_keys = sorted(
        [k for k in state.channels.get("derived", {})
         if k.startswith(derived_prefix)]
    )
    di = idx - n_raw
    if di < 0 or di >= len(derived_keys):
        return None
    vol = state.get_derived_channel(derived_keys[di])
    if vol is None:
        return None
    if mode in ("slice", "Slice (Z)"):
        z = min(z_idx, vol.shape[0] - 1)
        plane = vol[z]
    elif mode in ("mean", "Mean Projection"):
        plane = np.mean(vol, axis=0)
    else:
        plane = np.max(vol, axis=0)
    lo, hi = float(plane.min()), float(plane.max())
    if hi > lo:
        return np.clip((plane - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return np.zeros_like(plane, dtype=np.float32)


def _channel_index_from_combo(combo: QComboBox, state: AppState,
                               active_only: bool = False) -> int | None:
    pool = ([m for m in state.channel_mappings if m.get("enabled", True)]
            if active_only else state.channel_mappings)
    i = combo.currentIndex()
    if 0 <= i < len(pool):
        return pool[i]["data_channel_index"]
    return None

def _label_from_combo(combo: QComboBox, state: AppState,
                      active_only: bool = False) -> str:
    pool = ([m for m in state.channel_mappings if m.get("enabled", True)]
            if active_only else state.channel_mappings)
    i = combo.currentIndex()
    if 0 <= i < len(pool):
        return pool[i].get("biological_label", "—") or "—"
    return "—"

def _cb_build_composite_for_segmentation(state: "AppState") -> np.ndarray | None:
    img = state.preprocessed_image if state.has_preprocessed() else state.raw_image
    if img is None:
        return None
    nz, nc, ny, nx = img.shape
    _COLOR_MAP = {
        "red": (1.0,0.0,0.0), "green": (0.0,1.0,0.0), "blue": (0.0,0.0,1.0),
        "cyan": (0.0,1.0,1.0), "magenta": (1.0,0.0,1.0), "yellow": (1.0,1.0,0.0),
        "white": (1.0,1.0,1.0), "gray": (1.0,1.0,1.0), "orange": (1.0,0.5,0.0),
    }
    composite = np.zeros((ny, nx, 3), dtype=np.float32)
    processed_any = False
    for m in state.channel_mappings:
        if not m.get("enabled", True):
            continue
        cidx = m.get("data_channel_index", None)
        if cidx is None or cidx >= nc:
            continue
        color_name = m.get("color", "gray").lower()
        rgb = np.array(_COLOR_MAP.get(color_name, (1.0,1.0,1.0)), dtype=np.float32)
        vol = img[:, cidx, :, :].astype(np.float32)
        plane = np.max(vol, axis=0)
        lo, hi = float(np.percentile(plane, 2)), float(np.percentile(plane, 98))
        plane_n = np.clip((plane - lo) / (hi - lo + 1e-9), 0.0, 1.0)
        composite += plane_n[:, :, np.newaxis] * rgb[np.newaxis, np.newaxis, :]
        processed_any = True
    if not processed_any:
        return np.zeros((ny, nx), dtype=np.uint8)
    composite = np.clip(composite, 0.0, 1.0)
    from PIL import Image as _PIL
    pil_gray = _PIL.fromarray((composite * 255).astype(np.uint8), mode="RGB").convert("L")
    return np.array(pil_gray, dtype=np.uint8)

def _cb_preprocess(image_np: np.ndarray) -> np.ndarray:

    return image_np

def _cb_postprocess(masks: np.ndarray, min_size: int = 0) -> np.ndarray:
    if masks is None:
        return None
    if min_size <= 0:
        return masks.astype(np.int32)  # No filtering — matches original exactly

    result = masks.astype(np.int32).copy()
    for prop in regionprops(masks):
        if prop.area < min_size:
            result[masks == prop.label] = 0
    return result

def _score_segmentation(masks: np.ndarray) -> float:
    """Score a 2-D segmentation mask for auto-parameter search.
    Higher is better. Returns -inf for degenerate results.
    """
    if masks is None or masks.max() == 0:
        return -np.inf
    n = int(masks.max())
    if n < 1 or n > 5000:
        return -np.inf
    props    = regionprops(masks)
    areas    = np.array([p.area for p in props], dtype=np.float32)
    if len(areas) == 0:
        return -np.inf
    med_area = float(np.median(areas))
    if med_area < 10:
        return -np.inf
    circs = []
    for p in props:
        perim = p.perimeter
        if perim > 0:
            circs.append(min(1.0, 4 * np.pi * p.area / (perim ** 2)))
    mean_circ = float(np.mean(circs)) if circs else 0.5
    score = (np.log1p(n) * 0.4
             + np.log1p(med_area) * 0.3
             + mean_circ * 0.3)
    return float(score)


def _run_cellpose_for_scoring(plane_f32: np.ndarray, diameter: float,
                               flow_threshold: float, cellprob_threshold: float,
                               model_type: str) -> np.ndarray:
    """Run Cellpose on a 2-D plane with explicit thresholds for scoring.
    Returns labeled int32 mask (zeros on failure/unavailability).
    """
    if not CELLPOSE_AVAILABLE:
        return np.zeros(plane_f32.shape, dtype=np.int32)
    try:
        img = plane_f32.astype(np.float32)
        lo, hi = np.percentile(img, [1, 99])
        if hi > lo:
            img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
        else:
            img = np.zeros_like(img)
        key = model_type
        if key not in _CELLPOSE_MODEL_CACHE:
            _CELLPOSE_MODEL_CACHE[key] = cp_models.CellposeModel(
                gpu=True, model_type=model_type)
        model = _CELLPOSE_MODEL_CACHE[key]
        masks_list, _, _ = model.eval(
            [img], diameter=diameter, channels=[0, 0],
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold)
        return masks_list[0].astype(np.int32)
    except Exception:
        return np.zeros(plane_f32.shape, dtype=np.int32)


class _AutoParamWorker(BaseWorker):
    """Random search for best Cellpose parameters. Returns {diameter, flow_threshold, cellprob_threshold}."""
    def __init__(self, test_slice: np.ndarray, model_type: str, parent=None):
        super().__init__(parent)
        self._slice      = test_slice
        self._model_type = model_type

    def run_task(self):
        # Fast auto-parameters: an analytical object-size estimate (no
        # Cellpose) followed by a single Cellpose validation run. Replaces
        # the old 40-trial random search — same return dict, much faster.
        self.signals.progress_percent.emit(0)
        self.signals.progress.emit("Estimating object size analytically…")

        plane  = self._slice.astype(np.float32)
        lo, hi = np.percentile(plane, [1, 99])
        norm   = (np.clip((plane - lo) / (hi - lo), 0.0, 1.0)
                  if hi > lo else np.zeros_like(plane))

        # ── Phase 1: analytical diameter estimate (no Cellpose) ──────────
        d_est = 30.0
        try:
            from skimage.filters import threshold_otsu as _ot, gaussian as _ga
            from skimage.measure import label as _lbl, regionprops as _rp
            from skimage.morphology import remove_small_objects as _rso
            sm  = _ga(norm, sigma=1.0)
            bw  = sm > _ot(sm)
            bw  = _rso(bw, min_size=20)
            eqd = [p.equivalent_diameter for p in _rp(_lbl(bw)) if p.area >= 20]
            if eqd:
                d_est = float(np.clip(np.median(eqd), 6.0, 120.0))
        except Exception as _e:
            print(f"[auto-params] analytical estimate failed: {_e}")
        self.signals.progress_percent.emit(40)
        self.signals.progress.emit(
            f"Estimated diameter ≈ {d_est:.0f} px — validating with Cellpose…")

        # ── Phase 2: single Cellpose validation (+ cheap fallbacks) ──────
        best       = (d_est, 0.4, 0.0)
        best_score = -np.inf
        for c in (0.0,):
            masks = _run_cellpose_for_scoring(norm, d_est, 0.4, c, self._model_type)
            s     = _score_segmentation(masks)
            if s > best_score:
                best_score, best = s, (d_est, 0.4, c)
        if best_score == -np.inf:
            for c in (-1.0, 1.0):
                masks = _run_cellpose_for_scoring(norm, d_est, 0.4, c, self._model_type)
                s     = _score_segmentation(masks)
                if s > best_score:
                    best_score, best = s, (d_est, 0.4, c)

        d, f, c = best
        self.signals.progress_percent.emit(100)
        self.signals.progress.emit(
            f"Auto-parameters done — diameter {int(round(d))}, "
            f"flow {round(f, 2)}, cellprob {round(c, 2)}")
        return {"diameter": int(round(d)), "flow": round(f, 2), "cellprob": round(c, 2)}


# Runs Cellpose cell body segmentation in the background
class CellBodyWorker(BaseWorker):
    def __init__(self, image_np: np.ndarray, diameter: float,
                 min_size: int, flow_threshold: float = 0.4,
                 cellprob_threshold: float = 0.0, parent=None):
        super().__init__(parent)
        self._img      = image_np
        self._diam     = diameter
        self._min_sz   = min_size
        self._flow     = flow_threshold
        self._cellprob = cellprob_threshold

    def run_task(self):

        self.signals.progress_percent.emit(5)

        if not CELLPOSE_AVAILABLE:
            self.signals.progress.emit("Cellpose unavailable – using Otsu fallback…")
            norm = self._img.astype(np.float32)
            rng  = float(norm.max()) - float(norm.min())
            if rng > 0:
                norm = (norm - norm.min()) / rng
            binary = norm > sk_otsu(norm)
            self.signals.progress_percent.emit(60)
            if self._min_sz > 0:
                binary = remove_small_objects(binary, min_size=self._min_sz)
            masks = sk_label(binary).astype(np.int32)
            self.signals.progress_percent.emit(100)
            return {"masks": masks}


        self.signals.progress.emit("Preparing image for Cellpose (cell bodies)…")
        img = _normalize_for_cellpose(self._img)
        self.signals.progress_percent.emit(10)

        self.signals.progress.emit("Loading Cellpose cyto2 model…")
        cache_key = "cyto2"
        if cache_key not in _CELLPOSE_MODEL_CACHE:
            _CELLPOSE_MODEL_CACHE[cache_key] = cp_models.CellposeModel(
                gpu=True, model_type="cyto2")
        model = _CELLPOSE_MODEL_CACHE[cache_key]
        self.signals.progress_percent.emit(15)

        self.signals.progress.emit("Running Cellpose cell body segmentation…")
        masks_out, flows, styles = model.eval(
            [img],
            diameter=self._diam,
            channels=[0, 0],
            flow_threshold=self._flow,
            cellprob_threshold=self._cellprob,
        )
        mask = masks_out[0]
        self.signals.progress_percent.emit(75)


        self.signals.progress.emit("Filling cell mask holes…")
        from scipy.ndimage import binary_fill_holes as _bfh
        mask_filled = np.zeros_like(mask, dtype=np.int32)
        n_obj = int(mask.max())
        for obj_id in range(1, n_obj + 1):
            mask_filled[_bfh(mask == obj_id)] = obj_id
            if n_obj > 0:
                pct = 75 + int((obj_id / n_obj) * 15)
                self.signals.progress_percent.emit(min(pct, 90))
        mask = mask_filled
        self.signals.progress_percent.emit(92)


        self.signals.progress.emit("Applying post-processing filter…")
        mask = _cb_postprocess(mask, min_size=self._min_sz)
        self.signals.progress_percent.emit(100)
        return {"masks": mask}


class CellBodyTab(QWidget):

    BOUNDARY_COLORS = {"Green":(0,255,0),"Red":(255,0,0),"Blue":(0,0,255),
                       "Yellow":(255,255,0),"Cyan":(0,255,255),"White":(255,255,255)}

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self._state          = state
        self._mask_array     = None
        self._included_cells = set()
        self._undo_stack     = []
        self._redo_stack     = []
        self._drawing_pts    = []
        self._is_drawing     = False
        self._worker         = None
        self._last_composite_img8 = None  # cached composite shown in canvas
        self._mask_3d        = None   # (Z,Y,X) int32 — result of 3D propagation
        self._is_3d_mode     = False  # True while displaying 3D result
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        ls, _, ll = make_scroll_widget()
        ls.setFixedWidth(295)

        cg = section("Input (from preprocessed image)")
        cg_hdr = QHBoxLayout()
        cg_hdr.addStretch()
        cg_hdr.addWidget(HelpButton("Cell Body Detection"))
        cf = QFormLayout(cg)
        cf.addRow(cg_hdr)
        self._ch_combo = QComboBox()   # populated from AppState.channel_mappings
        self._ch_combo.setToolTip(
            "Select the image channel used for cell body segmentation.\n\n"
            "Channel labels are assigned in Image Viewer → Channel Mapping.\n"
            "Typically use a cytoplasmic stain (e.g. CCT1, phalloidin, CellMask).\n"
            "Avoid using the nucleus channel here unless no cytoplasm stain exists.")
        self._zm_combo = QComboBox()

        self._zm_combo.addItems(["Max Projection", "Mean Projection", "Slice (Z)", "All slices (Z)"])
        self._zm_combo.setToolTip(
            "How to handle Z-stacks before segmentation:\n\n"
            "Max Projection  — brightest pixel per XY position across all Z (default).\n"
            "                  Best for sparse bright cells.\n\n"
            "Mean Projection — average intensity across Z.\n"
            "                  Better for evenly distributed staining.\n\n"
            "Slice (Z)       — segment a single Z plane chosen by the Z-slice control.\n"
            "                  Fastest; use when cells are in one focal plane.\n\n"
            "All slices (Z)  — run Cellpose independently on every Z plane and merge.\n"
            "                  Most thorough for 3-D stacks. Then use\n"
            "                  'Select Active Cell Layer' to choose which Z feeds\n"
            "                  downstream analysis.")
        self._z_spin = QSpinBox(); self._z_spin.setRange(0, 0)
        from PyQt5.QtWidgets import QAbstractSpinBox as _ASB2
        self._z_spin.setButtonSymbols(_ASB2.NoButtons)
        self._z_spin.setMinimumHeight(28)
        self._z_spin.setToolTip(
            "Z-plane index used when Projection is set to 'Slice (Z)'.\n\n"
            "0 = first (bottom) Z plane; max = top of stack.\n"
            "Only active when 'Slice (Z)' or 'All slices (Z)' is selected.")
        cf.addRow("Channel:", self._ch_combo)
        cf.addRow("Projection:", self._zm_combo)
        cf.addRow("Z-slice:", self._z_spin)
        ll.addWidget(cg)

        sg = section("Segmentation Parameters")
        sl = QVBoxLayout(sg)
        self._diameter_spin = QSpinBox()
        self._diameter_spin.setRange(1, 2000)
        self._diameter_spin.setValue(100)

        self._diameter_spin.setToolTip(
            "Approximate cell diameter in pixels.\n\n"
            "Increase for large or spread cells.\n"
            "Decrease for small, compact, or tightly packed cells.\n"
            "Set to 0 to let Cellpose estimate automatically (slower).\n\n"
            "Typical range: 30–200 px.\n"
            "Rule of thumb: measure a representative cell edge-to-edge in pixels.")
        sl.addLayout(_make_spinbox_row("Cellpose diameter:", self._diameter_spin))
        self._flow_threshold_spin = QDoubleSpinBox()
        self._flow_threshold_spin.setRange(0.0, 1.5)
        self._flow_threshold_spin.setSingleStep(0.05)
        self._flow_threshold_spin.setDecimals(2)
        self._flow_threshold_spin.setValue(0.4)
        self._flow_threshold_spin.setToolTip(
            "Cellpose flow error threshold (0.0 – 1.5).\n\n"
            "Higher values → more permissive → more (possibly noisy) detections.\n"
            "Lower values → stricter → fewer but cleaner detections.\n\n"
            "Default: 0.4. Recommended range: 0.2–0.8.\n"
            "Increase if cells are missed; decrease if too many false positives appear.")
        sl.addLayout(_make_spinbox_row("Flow threshold:", self._flow_threshold_spin))
        self._cellprob_threshold_spin = QDoubleSpinBox()
        self._cellprob_threshold_spin.setRange(-6.0, 6.0)
        self._cellprob_threshold_spin.setSingleStep(0.1)
        self._cellprob_threshold_spin.setDecimals(2)
        self._cellprob_threshold_spin.setValue(0.0)
        self._cellprob_threshold_spin.setToolTip(
            "Cellpose cell probability threshold (–6.0 to +6.0).\n\n"
            "Lower (negative) → include more low-confidence regions → larger masks.\n"
            "Higher (positive) → stricter → smaller, sparser masks.\n\n"
            "Default: 0.0.  Typical range: –2.0 to +2.0.\n"
            "Decrease for faint or weakly-stained cells; increase to trim oversized masks.")
        sl.addLayout(_make_spinbox_row("Cell prob:", self._cellprob_threshold_spin))
        self._min_sz_sp = QSpinBox()
        self._min_sz_sp.setRange(50, 100000)
        self._min_sz_sp.setValue(2000)
        self._min_sz_sp.setToolTip(
            "Minimum cell area in pixels (2D projected).\n\n"
            "Objects smaller than this are discarded as debris or noise.\n"
            "Increase to remove small artefacts; decrease to keep tiny cells.\n\n"
            "Typical range: 500–5 000 px depending on magnification.")
        sl.addLayout(_make_spinbox_row("Min cell area (px):", self._min_sz_sp))
        self._auto_param_btn = QPushButton("🔍  Auto Parameters (Cell)")
        self._auto_param_btn.setStyleSheet(BTN_PURPLE)
        self._auto_param_btn.setEnabled(False)
        sl.addWidget(self._auto_param_btn)
        self._seg_btn = QPushButton("▶  Run Segmentation")
        self._seg_btn.setStyleSheet(BTN_RUN)
        self._seg_btn.setEnabled(False)
        sl.addWidget(self._seg_btn)
        ll.addWidget(sg)

        mg = section("Manual Mask Drawing")
        ml = QVBoxLayout(mg)
        self._draw_btn     = QPushButton("✏  Start Drawing Polygon")
        self._finalize_btn = QPushButton("✓  Finalize Polygon Mask")
        self._finalize_btn.setEnabled(False)
        ml.addWidget(self._draw_btn)
        ml.addWidget(self._finalize_btn)
        ll.addWidget(mg)

        hg = section("History")
        hl = QHBoxLayout(hg)
        self._undo_btn = QPushButton("↩ Undo")
        self._redo_btn = QPushButton("↪ Redo")
        self._undo_btn.setEnabled(False)
        self._redo_btn.setEnabled(False)
        hl.addWidget(self._undo_btn)
        hl.addWidget(self._redo_btn)
        self._reset_btn = QPushButton("🗑 Reset")
        self._reset_btn.setToolTip("Clear all segmentation results and reset to a fresh state.")
        hl.addWidget(self._reset_btn)
        ll.addWidget(hg)

        og = section("Overlay Options")
        ol = QVBoxLayout(og)
        self._show_orig_cb  = QCheckBox("Show original image"); self._show_orig_cb.setChecked(True)
        self._show_mask_cb  = QCheckBox("Show cell masks");     self._show_mask_cb.setChecked(False)
        self._show_bnd_cb   = QCheckBox("Show cell outlines");  self._show_bnd_cb.setChecked(True)
        self._show_num_cb   = QCheckBox("Show cell IDs");       self._show_num_cb.setChecked(True)
        bc_row = QHBoxLayout()
        bc_row.addWidget(QLabel("Outline color:"))
        self._bc_combo = QComboBox()
        self._bc_combo.addItems(list(self.BOUNDARY_COLORS.keys()))
        bc_row.addWidget(self._bc_combo)
        for cb in [self._show_orig_cb, self._show_mask_cb,
                   self._show_bnd_cb,  self._show_num_cb]:
            ol.addWidget(cb)
        ol.addLayout(bc_row)
        ll.addWidget(og)

        self._prog     = QProgressBar(); self._prog.setRange(0,0); self._prog.setVisible(False)
        self._stat_lbl = StatusLabel()
        ll.addWidget(self._prog)
        ll.addWidget(self._stat_lbl)
        ll.addStretch()
        root.addWidget(ls)

        cw = QWidget(); cl = QVBoxLayout(cw); cl.setContentsMargins(0,0,0,0)
        self._canvas = ImageCanvas(figsize=(7,6))
        self._nav    = NavigationToolbar(self._canvas, self)
        self._nav.setStyleSheet(_viewer_tb_qss())
        cl.addWidget(self._nav)
        cl.addWidget(self._canvas, stretch=1)
        root.addWidget(cw, stretch=1)

        rw = QWidget(); rw.setFixedWidth(235); rl = QVBoxLayout(rw)
        sg2 = section("Statistics"); sl2 = QVBoxLayout(sg2)
        self._stats_lbl = QLabel("")
        self._stats_lbl.setWordWrap(True)
        self._stats_lbl.setStyleSheet("color:#80ff80;font-family:monospace;font-size:11px;")
        sl2.addWidget(self._stats_lbl)
        rl.addWidget(sg2)
        eg = section("Export"); el = QVBoxLayout(eg)
        self._exp_mask_btn = QPushButton("Export Mask (TIF)")
        self._exp_pdf_btn  = QPushButton("Export PDF Report")
        for b in [self._exp_mask_btn, self._exp_pdf_btn]:
            b.setEnabled(False); el.addWidget(b)
        rl.addWidget(eg)
        rl.addStretch()
        root.addWidget(rw)

        self._seg_btn.clicked.connect(self._run_seg)
        self._auto_param_btn.clicked.connect(self._run_auto_params)
        self._draw_btn.clicked.connect(self._start_draw)
        self._finalize_btn.clicked.connect(self._finalize_poly)
        self._undo_btn.clicked.connect(self._undo)
        self._redo_btn.clicked.connect(self._redo)
        self._reset_btn.clicked.connect(self._reset_state)
        for cb in [self._show_orig_cb, self._show_mask_cb,
                   self._show_bnd_cb, self._show_num_cb]:
            cb.stateChanged.connect(lambda _: self._refresh())
        self._bc_combo.currentTextChanged.connect(lambda _: self._refresh())
        self._zm_combo.currentIndexChanged.connect(self._on_cell_mode_change)
        self._z_spin.valueChanged.connect(self._refresh)
        self._ch_combo.currentIndexChanged.connect(self._refresh)
        self._exp_mask_btn.clicked.connect(self._export_mask)
        self._exp_pdf_btn.clicked.connect(self._export_pdf)
        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._state.subscribe(self._on_state_change)

    def _on_state_change(self, what):
        if what in ("raw_image", "preprocessed", "channel_mappings"):
            arr = self._state.preprocessed_image
            if arr is not None:
                nz = arr.shape[0]
                self._z_spin.setRange(0, max(0, nz-1))
                self._seg_btn.setEnabled(True)
                self._auto_param_btn.setEnabled(True)
            _populate_channel_combo(self._ch_combo, self._state,
                                    preferred_label="CCT1")
            self._last_composite_img8 = None
            if arr is not None:
                self._refresh()


    def _on_cell_mode_change(self, _=None):
        """Enable Z-spin for Slice (Z) and All slices (Z) modes."""
        txt = self._zm_combo.currentText()
        # "Slice (Z)" = index 2, "All slices (Z)" = index 3
        self._z_spin.setEnabled(txt in ("Slice (Z)", "All slices (Z)"))
        self._refresh()

    def _get_input_slice(self) -> np.ndarray | None:
        if not self._state.has_preprocessed():
            return None
        txt = self._zm_combo.currentText()
        proj_map = {"Max Projection": "max", "Mean Projection": "mean",
                    "Slice (Z)": "slice", "All slices (Z)": "max"}
        proj = proj_map.get(txt, "max")
        ch_idx = _channel_index_from_combo(self._ch_combo, self._state)
        if ch_idx is None:
            ch_idx = 0
        return self._state.get_display_slice(
            channel=ch_idx,
            z=self._z_spin.value(),
            projection=proj)

    def _run_seg(self):
        if not _require_preprocessed(self._state, self):
            return


        diam     = float(self._diameter_spin.value())
        flow     = self._flow_threshold_spin.value()
        cellprob = self._cellprob_threshold_spin.value()
        min_sz   = self._min_sz_sp.value()


        if self._zm_combo.currentText() == "All slices (Z)":
            # 3D slice propagation mode
            img = (self._state.preprocessed_image
                   if self._state.has_preprocessed()
                   else self._state.raw_image)
            if img is None:
                QMessageBox.warning(self, "No Image", "Load an image first.")
                return
            ch_idx = _channel_index_from_combo(self._ch_combo, self._state)
            if ch_idx is None:
                ch_idx = 0
            self._seg_btn.setEnabled(False)
            self._prog.setVisible(True)
            self._prog.setRange(0, 100); self._prog.setValue(0); self._prog.setFormat("%p%")
            self._stat_lbl.info("Running All slices (Z) segmentation…")
            self._worker = _CellBody3DWorker(img, ch_idx, diam, min_sz, parent=self)
            self._worker.signals.progress.connect(self._stat_lbl.info)
            self._worker.signals.progress_percent.connect(self._prog.setValue)
            self._worker.signals.result.connect(self._on_3d_done)
            self._worker.signals.error.connect(self._on_seg_err)
            self._worker.signals.finished.connect(self._on_seg_fin)
            self._worker.start()
            return

        # Existing 2D projection path
        img8 = _cb_build_composite_for_segmentation(self._state)

        if img8 is None or img8.size == 0:
            plane = self._get_input_slice()
            if plane is None:
                QMessageBox.warning(self, "No Image",
                                    "Could not build segmentation input. "
                                    "Load an image and configure channels first.")
                return
            img8 = (plane * 255).astype(np.uint8)

        self._seg_btn.setEnabled(False)
        self._prog.setVisible(True)
        self._prog.setRange(0, 0); self._prog.setFormat("")
        self._stat_lbl.info("Segmenting…")

        self._worker = CellBodyWorker(img8, diam, min_sz,
                                      flow_threshold=flow,
                                      cellprob_threshold=cellprob, parent=self)
        self._worker.signals.progress.connect(self._stat_lbl.info)
        self._worker.signals.result.connect(self._on_seg_done)
        self._worker.signals.error.connect(self._on_seg_err)
        self._worker.signals.finished.connect(self._on_seg_fin)
        self._worker.start()

    def _run_auto_params(self):
        """Launch auto-parameter search in background. Updates UI spinboxes when done."""
        if not _require_preprocessed(self._state, self):
            return
        # Use a representative test slice (max projection of selected channel)
        plane = self._get_input_slice()
        if plane is None:
            self._stat_lbl.warn("No image available for auto-parameter search.")
            return
        test_slice = (plane * 65535).astype(np.float32)
        self._auto_param_btn.setEnabled(False)
        self._seg_btn.setEnabled(False)
        self._prog.setVisible(True)
        self._prog.setRange(0, 100); self._prog.setValue(0); self._prog.setFormat("%p%")
        self._stat_lbl.info("Auto parameter search running…")
        self._auto_worker = _AutoParamWorker(test_slice, "cyto2", parent=self)
        self._auto_worker.signals.progress.connect(self._stat_lbl.info)
        self._auto_worker.signals.progress_percent.connect(self._prog.setValue)
        self._auto_worker.signals.result.connect(self._on_auto_param_done)
        self._auto_worker.signals.error.connect(
            lambda msg: (self._stat_lbl.err("Auto search failed."),
                         self._auto_param_btn.setEnabled(True),
                         self._seg_btn.setEnabled(True),
                         self._prog.setVisible(False)))
        self._auto_worker.signals.finished.connect(
            lambda: (self._auto_param_btn.setEnabled(True),
                     self._seg_btn.setEnabled(True),
                     self._prog.setRange(0, 0),
                     self._prog.setFormat(""),
                     self._prog.setVisible(False)))
        self._auto_worker.start()

    def _on_auto_param_done(self, res):
        d = res.get("diameter", 40)
        f = res.get("flow",     0.4)
        c = res.get("cellprob", 0.0)
        # Update UI spinboxes — user can still change them freely
        self._diameter_spin.setValue(int(d))
        self._flow_threshold_spin.setValue(float(f))
        self._cellprob_threshold_spin.setValue(float(c))
        self._stat_lbl.ok(
            f"Auto parameters applied → diameter={d}, flow={f}, prob={c} (editable)")

    def _on_seg_done(self, res):
        masks = res.get("masks")
        if masks is None:
            self._stat_lbl.warn("No masks returned."); return
        self._push_undo()
        self._mask_array     = masks
        self._included_cells = set(np.unique(masks)) - {0}
        self._last_composite_img8 = _cb_build_composite_for_segmentation(self._state)
        self._refresh()
        self._update_stats()
        self._exp_mask_btn.setEnabled(True)
        self._exp_pdf_btn.setEnabled(True)
        n = len(self._included_cells)
        self._stat_lbl.ok(f"Done. {n} cells detected.")
        self._push_results()

    def _on_seg_err(self, msg):
        QMessageBox.critical(self, "Segmentation Error", msg)
        self._stat_lbl.err("Failed.")

    def _on_seg_fin(self):
        self._seg_btn.setEnabled(True)
        self._prog.setRange(0, 0); self._prog.setFormat("")
        self._prog.setVisible(False)

    def _on_3d_done(self, res):
        masks_3d = res.get("masks_3d")
        if masks_3d is None or masks_3d.max() == 0:
            self._stat_lbl.warn("3D segmentation returned empty result."); return
        self._push_undo()
        self._mask_3d        = masks_3d                   # (Z, Y, X)
        z                    = self._z_spin.value()
        self._mask_array     = masks_3d[min(z, masks_3d.shape[0]-1)]
        self._is_3d_mode     = True
        self._included_cells = set(np.unique(self._mask_3d)) - {0}
        self._last_composite_img8 = _cb_build_composite_for_segmentation(self._state)
        self._refresh()
        self._update_stats()
        self._exp_mask_btn.setEnabled(True)
        self._exp_pdf_btn.setEnabled(True)
        n = len(self._included_cells)
        self._stat_lbl.ok(f"3D done. {n} unique labels across Z. Use Z-spin to scroll.")
        self._push_results()

    def _start_draw(self):
        self._is_drawing = True; self._drawing_pts = []
        self._draw_btn.setEnabled(False); self._finalize_btn.setEnabled(True)
        self._stat_lbl.info("Click canvas to add vertices. Press 'Finalize' when done.")

    def _finalize_poly(self):
        if len(self._drawing_pts) < 3:
            QMessageBox.warning(self, "Drawing", "Need ≥ 3 points."); return
        plane = self._get_input_slice()
        if plane is None: return
        h, w = plane.shape
        from PIL import ImageDraw as _IDraw
        pi = PILImage.new("L", (w, h), 0)
        _IDraw.Draw(pi).polygon(self._drawing_pts, fill=1)
        pn = np.array(pi, dtype=np.int32)
        nid = (int(self._mask_array.max()) + 1 if self._mask_array is not None else 1)
        nm  = (self._mask_array.copy() if self._mask_array is not None
               else np.zeros((h, w), dtype=np.int32))
        nm[pn == 1] = nid
        self._push_undo()
        self._mask_array = nm
        self._included_cells = set(np.unique(nm)) - {0}
        self._drawing_pts = []; self._is_drawing = False
        self._draw_btn.setEnabled(True); self._finalize_btn.setEnabled(False)
        self._refresh(); self._update_stats()
        self._stat_lbl.ok("Manual mask added.")

    def _on_click(self, event):
        if event.inaxes != self._canvas.ax: return
        x, y = int(event.xdata or 0), int(event.ydata or 0)
        if self._is_drawing:
            self._drawing_pts.append((x, y)); self._refresh(); return
        if self._mask_array is not None:
            h, w = self._mask_array.shape[:2]
            if 0 <= y < h and 0 <= x < w:
                cid = int(self._mask_array[y, x])
                if cid != 0:
                    if cid in self._included_cells: self._included_cells.discard(cid)
                    else: self._included_cells.add(cid)
                    self._refresh(); self._update_stats()

    def _push_undo(self):
        snap = {"mask": self._mask_array.copy() if self._mask_array is not None else None,
                "inc":  set(self._included_cells)}
        self._undo_stack.append(snap); self._redo_stack.clear()
        if len(self._undo_stack) > 30: self._undo_stack.pop(0)
        self._undo_btn.setEnabled(True); self._redo_btn.setEnabled(False)

    def _undo(self):
        if not self._undo_stack: return
        self._redo_stack.append({"mask": self._mask_array.copy() if self._mask_array is not None else None,
                                  "inc": set(self._included_cells)})
        s = self._undo_stack.pop()
        self._mask_array = s["mask"]; self._included_cells = s["inc"]
        self._refresh(); self._update_stats()
        self._undo_btn.setEnabled(bool(self._undo_stack))
        self._redo_btn.setEnabled(True)

    def _redo(self):
        if not self._redo_stack: return
        self._undo_stack.append({"mask": self._mask_array.copy() if self._mask_array is not None else None,
                                  "inc": set(self._included_cells)})
        s = self._redo_stack.pop()
        self._mask_array = s["mask"]; self._included_cells = s["inc"]
        self._refresh(); self._update_stats()
        self._redo_btn.setEnabled(bool(self._redo_stack))

    def _reset_state(self):
        """Reset Cell Body tool to a fresh state without reloading the image."""
        # Clear segmentation results
        self._mask_array = None
        self._included_cells = set()
        self._drawing_pts = []
        self._is_drawing = False
        self._last_composite_img8 = None
        self._mask_3d    = None
        self._is_3d_mode = False

        # Clear history stacks
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._undo_btn.setEnabled(False)
        self._redo_btn.setEnabled(False)

        # Clear export buttons
        self._exp_mask_btn.setEnabled(False)
        self._exp_pdf_btn.setEnabled(False)

        # Clear statistics display
        self._stats_lbl.setText("")

        # Clear canvas
        self._canvas.clear_canvas()

        # Remove stored result from AppState (this tab only)
        self._state.results_dict.pop("Cell Body Detection", None)

        # Notify other components
        self._state.notify("results")

        self._stat_lbl.ok("Reset complete.")

    def _refresh(self, _=None):
        ax = self._canvas.ax; ax.clear(); ax.set_facecolor("#0a0a14")

        # In 3D mode: update _mask_array to current Z slice so overlays are correct
        if self._is_3d_mode and self._mask_3d is not None:
            z = self._z_spin.value()
            z = min(z, self._mask_3d.shape[0] - 1)
            self._mask_array = self._mask_3d[z]

        if self._last_composite_img8 is not None:
            display_img  = self._last_composite_img8.astype(np.float32) / 255.0
            display_cmap = "gray"
        else:
            display_img  = self._get_input_slice()
            display_cmap = "gray"

        # In 3D mode: show the raw volume slice at the current Z for the background
        if self._is_3d_mode and self._state.has_preprocessed():
            ch_idx = _channel_index_from_combo(self._ch_combo, self._state)
            if ch_idx is None:
                ch_idx = 0
            z = min(self._z_spin.value(),
                    self._state.preprocessed_image.shape[0] - 1)
            display_img  = self._state.get_display_slice(
                channel=ch_idx, z=z, projection="slice")
            display_cmap = "gray"

        if display_img is None:
            self._canvas.draw_idle(); return

        if self._show_orig_cb.isChecked():
            ax.imshow(display_img, cmap=display_cmap, vmin=0, vmax=1,
                      interpolation="nearest", origin="upper")

        if self._mask_array is not None:
            h, w = self._mask_array.shape[:2]
            all_ids  = set(np.unique(self._mask_array)) - {0}
            ids_draw = self._included_cells

            if self._show_mask_cb.isChecked() and ids_draw:
                from skimage.color import label2rgb
                colors = [(0.1,0.7,0.1) if lid in self._included_cells
                          else (0.7,0.1,0.1) for lid in sorted(all_ids)]
                ov = label2rgb(self._mask_array, bg_label=0, bg_color=None, colors=colors)
                ax.imshow(ov, alpha=0.35, interpolation="nearest", origin="upper")

            if self._show_bnd_cb.isChecked():
                dm = np.zeros_like(self._mask_array)
                for lid in ids_draw: dm[self._mask_array == lid] = lid
                bnd = _cb_calculate_exact_boundaries(dm)
                if bnd is not None and bnd.any():
                    bc  = self.BOUNDARY_COLORS.get(self._bc_combo.currentText(),(0,255,0))
                    bcn = tuple(v/255.0 for v in bc)
                    rgba = np.zeros((h,w,4),dtype=np.float32)
                    rgba[bnd,:3] = bcn; rgba[bnd,3] = 1.0
                    ax.imshow(rgba, interpolation="nearest", origin="upper")

            if self._show_num_cb.isChecked():
                for prop in regionprops(self._mask_array):
                    if prop.label not in ids_draw: continue
                    cy, cx = prop.centroid
                    color = "#80ff80" if prop.label in self._included_cells else "#ff8080"
                    ax.text(cx, cy, str(prop.label), fontsize=6, color=color,
                            ha="center", va="center", fontweight="bold")

        if self._is_drawing and self._drawing_pts:
            xs = [p[0] for p in self._drawing_pts]+[self._drawing_pts[0][0]]
            ys = [p[1] for p in self._drawing_pts]+[self._drawing_pts[0][1]]
            ax.plot(xs, ys, "y-", lw=1.5)
            ax.scatter([p[0] for p in self._drawing_pts],
                       [p[1] for p in self._drawing_pts], c="yellow", s=12, zorder=5)

        n_sel = len(self._included_cells)
        n_tot = len(set(np.unique(self._mask_array))-{0}) if self._mask_array is not None else 0
        ax.set_title(f"Cell Body Detection  |  {n_sel}/{n_tot} selected",
                     color="#a0a0d0", fontsize=9, pad=4)
        ax.axis("off"); self._canvas.draw_idle()

    def _update_stats(self):
        if self._mask_array is None: self._stats_lbl.setText(""); return
        n_tot = len(set(np.unique(self._mask_array))-{0})
        n_sel = len(self._included_cells)
        areas = [p.area for p in regionprops(self._mask_array) if p.label in self._included_cells]
        lines = [f"Total cells : {n_tot}",
                 f"Selected    : {n_sel}",
                 f"Deselected  : {n_tot-n_sel}"]
        if areas: lines.append(f"Mean area   : {np.mean(areas):.0f} px²")
        self._stats_lbl.setText("\n".join(lines))

    def _push_results(self):
        if self._mask_array is None: return
        rows = []
        for prop in regionprops(self._mask_array):
            rows.append({"cell_id": prop.label,
                         "area_px": prop.area,
                         "centroid_y": round(prop.centroid[0],1),
                         "centroid_x": round(prop.centroid[1],1),
                         "selected": prop.label in self._included_cells})
        df = pd.DataFrame(rows)
        summary = (f"Total cells: {len(rows)}\n"
                   f"Selected: {sum(1 for r in rows if r['selected'])}\n"
                   f"Mean area: {df['area_px'].mean():.0f} px²" if rows else "")
        self._state.set_result("Cell Body Detection", {
            "dataframe": df,
            "summary": summary,
            "mask": self._mask_array,
        })

    def _export_mask(self):
        if self._mask_array is None: return
        path, _ = QFileDialog.getSaveFileName(self,"Export Mask","","TIFF (*.tif)")
        if path:
            tifffile.imwrite(path, self._mask_array.astype(np.uint16))
            QMessageBox.information(self,"Exported",f"Saved:\n{path}")

    def _export_pdf(self):
        path, _ = QFileDialog.getSaveFileName(self,"Export PDF Report","","PDF (*.pdf)")
        if not path: return
        plane = self._get_input_slice()
        try:
            with PdfPages(path) as pdf:
                fig, ax = plt.subplots(figsize=(10,8))
                if plane is not None: ax.imshow(plane, cmap="gray")
                if self._mask_array is not None:
                    bnd = _cb_calculate_exact_boundaries(self._mask_array)
                    if bnd is not None:
                        bc = tuple(v/255.0 for v in self.BOUNDARY_COLORS.get(
                            self._bc_combo.currentText(),(0,255,0)))
                        rgba = np.zeros((*bnd.shape,4),dtype=np.float32)
                        rgba[bnd,:3]=bc; rgba[bnd,3]=1.0; ax.imshow(rgba)
                ax.set_title("Cell Body Segmentation"); ax.axis("off")
                pdf.savefig(fig); plt.close(fig)
            QMessageBox.information(self,"Exported",f"Saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self,"Export Error",str(e))


# ── Segmentation mode constants ───────────────────────────────────────────────
SEG_MODE_SLICE      = "Slice (Z)"
SEG_MODE_MAX_PROJ   = "Max Projection"
SEG_MODE_MEAN_PROJ  = "Mean Projection"

# removed "Propagated Max Projection" (obsolete)
SEG_MODE_3D         = "All slices (Z)"
SEG_MODES           = [SEG_MODE_SLICE, SEG_MODE_MAX_PROJ, SEG_MODE_MEAN_PROJ,
                       SEG_MODE_3D]

def get_input_plane(img, channel_idx, mode, z_index=0):
    """Extract a 2D plane from (Z,C,Y,X) for segmentation.
    Modes:"""
    vol = img[:, channel_idx].astype("float32")   # (Z, Y, X)
    if mode == SEG_MODE_SLICE:
        z     = min(z_index, vol.shape[0] - 1)
        plane = vol[z]
        print(f"[seg] mode: slice  z={z}")
    elif mode == SEG_MODE_MAX_PROJ:
        plane = np.max(vol, axis=0).astype(np.float32)
        print(f"[seg] mode: max projection")
    else:
        plane = np.mean(vol, axis=0).astype(np.float32)
        print(f"[seg] mode: mean projection")
    return plane


def _segment_2d_plane(plane_f32: np.ndarray, diameter, model_type: str,
                      min_size: int) -> np.ndarray:
    """Run Cellpose on a single 2-D float32 plane. Pure Cellpose: only min/max normalise, no extra transforms.
    """
    if not CELLPOSE_AVAILABLE:
        norm = plane_f32.astype(np.float32)
        rng  = float(norm.max()) - float(norm.min())
        if rng > 0:
            norm = (norm - norm.min()) / rng
        binary = norm > sk_otsu(norm)
        if min_size > 0:
            binary = remove_small_objects(binary, min_size=min_size)
        return sk_label(binary).astype(np.int32)


    img = _normalize_for_cellpose(plane_f32)

    cache_key = model_type
    if cache_key not in _CELLPOSE_MODEL_CACHE:
        _CELLPOSE_MODEL_CACHE[cache_key] = cp_models.CellposeModel(
            gpu=True, model_type=model_type)
    model = _CELLPOSE_MODEL_CACHE[cache_key]

    masks_list, _, _ = model.eval(
        [img], diameter=diameter, channels=[0, 0],
        flow_threshold=0.4, cellprob_threshold=0.0)
    mask = masks_list[0]

    from scipy.ndimage import binary_fill_holes as _bfh
    mask_filled = np.zeros_like(mask, dtype=np.int32)
    for obj_id in range(1, int(mask.max()) + 1):
        mask_filled[_bfh(mask == obj_id)] = obj_id
    mask = mask_filled

    if min_size > 0:
        binary = mask > 0
        binary = remove_small_objects(binary, min_size=min_size)
        mask   = (mask * binary).astype(np.int32)

    return mask.astype(np.int32)


def _propagate_labels_across_z(masks_per_z: list) -> np.ndarray:
    """Propagate and unify labels across Z slices.
    For each slice, matches objects to the previous slice using IoU overlap"""
    if not masks_per_z:
        return np.zeros((0,), dtype=np.int32)

    Z = len(masks_per_z)
    H, W = masks_per_z[0].shape
    final_masks   = np.zeros((Z, H, W), dtype=np.int32)
    current_label = 1

    for z, current in enumerate(masks_per_z):
        if current is None or current.max() == 0:
            # Empty slice — try gap-filling from previous slice
            if z > 0:
                prev = final_masks[z - 1]
                if prev.max() > 0:
                    filled = np.zeros_like(prev, dtype=np.int32)
                    for lbl in np.unique(prev):
                        if lbl == 0:
                            continue
                        dilated = binary_dilation(prev == lbl, iterations=2)
                        filled[dilated] = lbl
                    final_masks[z] = filled
            continue

        if z == 0:
            # First slice — assign labels as-is, re-label sequentially
            new_slice = np.zeros_like(current, dtype=np.int32)
            for region in regionprops(current):
                new_slice[current == region.label] = current_label
                current_label += 1
            final_masks[0] = new_slice
            continue

        prev      = final_masks[z - 1]
        new_slice = np.zeros_like(current, dtype=np.int32)

        for region in regionprops(current):
            mask_px = current == region.label

            if prev.max() == 0:
                # No previous labels — fresh label
                new_slice[mask_px] = current_label
                current_label += 1
                continue

            overlap_vals = prev[mask_px]
            fg_vals      = overlap_vals[overlap_vals > 0]

            if len(fg_vals) > 0:
                counts    = np.bincount(fg_vals)
                best_lbl  = int(np.argmax(counts))
                if best_lbl == 0:
                    # argmax hit background — use next free label
                    new_slice[mask_px] = current_label
                    current_label += 1
                else:
                    new_slice[mask_px] = best_lbl
            else:
                # No overlap with previous slice → new object
                new_slice[mask_px] = current_label
                current_label += 1

        final_masks[z] = new_slice

    return final_masks


def propagated_max_projection(masks_3d: np.ndarray) -> np.ndarray:
    """Create a label-preserving max projection from a propagated 3-D mask.
    Unlike a naive np.max projection, this method prevents label merging:"""
    if masks_3d.ndim != 3:
        raise ValueError(f"Expected 3-D mask, got shape {masks_3d.shape}")
    Z, Y, X = masks_3d.shape
    # Count votes per label per pixel (Y, X)
    # For each pixel (y,x), whichever label appears most across Z wins.
    projection = np.zeros((Y, X), dtype=np.int32)
    unique_labels = np.unique(masks_3d)
    unique_labels = unique_labels[unique_labels != 0]
    # vote_count[y,x] tracks the best count so far
    best_count = np.zeros((Y, X), dtype=np.int32)
    for lbl in unique_labels:
        # Boolean cube where this label appears
        presence = (masks_3d == lbl)  # (Z, Y, X) bool
        count = presence.sum(axis=0).astype(np.int32)  # (Y, X) — slices present
        better = count > best_count
        projection[better] = lbl
        best_count[better] = count[better]
    return projection


def run_3d_segmentation(volume: np.ndarray, channel_idx: int,
                        diameter, model_type: str, min_size: int,
                        progress_cb=None) -> np.ndarray:
    """Segment every Z slice independently, then propagate labels.
    Parameters"""
    vol = volume[:, channel_idx].astype(np.float32)   # (Z, Y, X)
    Z   = vol.shape[0]

    masks_per_z = []
    for z in range(Z):
        plane = vol[z]
        try:
            m = _segment_2d_plane(plane, diameter, model_type, min_size)
        except Exception as exc:
            print(f"[3D seg] slice z={z} failed: {exc}")
            m = np.zeros(plane.shape, dtype=np.int32)
        masks_per_z.append(m)
        if progress_cb is not None:
            pct = int(((z + 1) / Z) * 90)   # 90% for segmentation, 10% for propagation
            progress_cb(pct)

    result = _propagate_labels_across_z(masks_per_z)
    if progress_cb is not None:
        progress_cb(100)
    return result


class _CellBody3DWorker(BaseWorker):
    """Worker for 3D slice-propagation segmentation of cell bodies."""
    def __init__(self, volume: np.ndarray, channel_idx: int,
                 diameter, min_size: int, parent=None):
        super().__init__(parent)
        self._vol     = volume
        self._ch_idx  = channel_idx
        self._diam    = diameter
        self._min_sz  = min_size

    def run_task(self):
        self.signals.progress.emit("Running 3D cell body segmentation…")
        self.signals.progress_percent.emit(0)
        masks_3d = run_3d_segmentation(
            self._vol, self._ch_idx, self._diam, "cyto2", self._min_sz,
            progress_cb=lambda p: self.signals.progress_percent.emit(p))
        self.signals.progress_percent.emit(100)
        return {"masks_3d": masks_3d}


class _Nucleus3DWorker(BaseWorker):
    """Worker for 3D slice-propagation segmentation of nuclei."""
    def __init__(self, volume: np.ndarray, channel_idx: int,
                 diameter, min_size: int, parent=None):
        super().__init__(parent)
        self._vol    = volume
        self._ch_idx = channel_idx
        self._diam   = diameter
        self._min_sz = min_size

    def run_task(self):
        self.signals.progress.emit("Running 3D nucleus segmentation…")
        self.signals.progress_percent.emit(0)
        masks_3d = run_3d_segmentation(
            self._vol, self._ch_idx, self._diam, "nuclei", self._min_sz,
            progress_cb=lambda p: self.signals.progress_percent.emit(p))
        self.signals.progress_percent.emit(100)
        return {"masks_3d": masks_3d}


def run_cellpose_segmentation(image, diameter, model_type, mode="max", z_slice=0):
    """Unified Cellpose segmentation for cell bodies and nuclei."""
    if not CELLPOSE_AVAILABLE:
        raise RuntimeError("Cellpose is not installed.")

    # ── 1. Ensure 2D ──────────────────────────────────────────────────────────
    img2d = image
    if img2d.ndim == 4:
        vol = img2d[:, 0]
        if mode == "slice":
            img2d = vol[min(z_slice, vol.shape[0] - 1)]
        elif mode == "mean":
            img2d = np.mean(vol, axis=0)
        else:
            img2d = np.max(vol, axis=0)
    elif img2d.ndim == 3:
        img2d = img2d.max(axis=0)


    img2d = _normalize_for_cellpose(img2d)

    # ── 2. Load (or reuse cached) Cellpose model ──────────────────────────────
    cache_key = model_type
    if cache_key not in _CELLPOSE_MODEL_CACHE:
        _CELLPOSE_MODEL_CACHE[cache_key] = cp_models.CellposeModel(
            gpu=True, model_type=model_type
        )
    model = _CELLPOSE_MODEL_CACHE[cache_key]

    # ── 3. Run model.eval ─────────────────────────────────────────────────────
    masks, flows, styles = model.eval(
        [img2d],
        diameter=diameter,
        channels=[0, 0],
        flow_threshold=0.4,
        cellprob_threshold=0.0,
    )
    mask = masks[0]

    # ── 4. Fill holes per object ──────────────────────────────────────────────
    from scipy.ndimage import binary_fill_holes as _bfh
    mask_filled = np.zeros_like(mask, dtype=np.int32)
    for obj_id in range(1, int(mask.max()) + 1):
        cell = mask == obj_id
        mask_filled[_bfh(cell)] = obj_id

    return mask_filled.astype(np.int32)


class NucleusWorker(BaseWorker):
    # --- versie27 nieuw ---
    """Cellpose nucleus segmentation
    Key changes vs versie26:"""

    def __init__(self, plane: np.ndarray, diameter, min_size: int,
                 flow_threshold: float = 0.4, cellprob_threshold: float = 0.0,
                 parent=None):
        super().__init__(parent)
        self._plane    = plane
        self._diam     = diameter
        self._min_sz   = min_size
        self._flow     = flow_threshold
        self._cellprob = cellprob_threshold

    def run_task(self):

        # The input plane should already be preprocessed by the user if needed.
        if not CELLPOSE_AVAILABLE:
            self.signals.progress.emit("Cellpose unavailable – using Otsu fallback…")
            self.signals.progress_percent.emit(10)
            norm = self._plane.astype(np.float32)
            rng  = float(norm.max()) - float(norm.min())
            if rng > 0:
                norm = (norm - norm.min()) / rng
            binary = norm > sk_otsu(norm)
            self.signals.progress_percent.emit(60)
            if self._min_sz > 0:
                binary = remove_small_objects(binary, min_size=self._min_sz)
            masks = sk_label(binary).astype(np.int32)
            self.signals.progress_percent.emit(100)
            return {"masks": masks}


        self.signals.progress.emit("Normalising nucleus image for Cellpose…")
        self.signals.progress_percent.emit(5)
        img = _normalize_for_cellpose(self._plane)


        self.signals.progress.emit("Loading Cellpose nuclei model…")
        self.signals.progress_percent.emit(10)
        cache_key = "nuclei"
        if cache_key not in _CELLPOSE_MODEL_CACHE:
            _CELLPOSE_MODEL_CACHE[cache_key] = cp_models.CellposeModel(
                gpu=True, model_type="nuclei"
            )
        model = _CELLPOSE_MODEL_CACHE[cache_key]

        self.signals.progress.emit("Running Cellpose nucleus segmentation…")
        self.signals.progress_percent.emit(15)

        masks_list, flows, styles = model.eval(
            [img],
            diameter=self._diam,
            channels=[0, 0],
            flow_threshold=self._flow,
            cellprob_threshold=self._cellprob,
        )
        mask = masks_list[0]
        self.signals.progress_percent.emit(70)

        if mask is None or mask.max() == 0:
            print("⚠️ No nuclei detected — try lowering Cell prob or reducing Diameter")


        self.signals.progress.emit("Filling nucleus holes…")
        self.signals.progress_percent.emit(75)
        from scipy.ndimage import binary_fill_holes as _bfh
        mask_filled = np.zeros_like(mask, dtype=np.int32)
        n_obj = int(mask.max())
        for obj_id in range(1, n_obj + 1):
            mask_filled[_bfh(mask == obj_id)] = obj_id

            if n_obj > 0:
                pct = 75 + int((obj_id / n_obj) * 10)
                self.signals.progress_percent.emit(min(pct, 85))
        mask = mask_filled


        self.signals.progress_percent.emit(90)


        self.signals.progress.emit("Post-processing nucleus masks…")
        # Re-label by connected component of EQUAL label value (not the
        # binary union). skimage.label groups neighbouring pixels only when
        # they share the same value, so touching nuclei that watershed gave
        # different labels stay separate, AND a single label spread across a
        # gap becomes two labels — non-touching nuclei no longer count as one.
        mask = sk_label(mask).astype(np.int32)
        if self._min_sz > 0:
            mask = remove_small_objects(mask, min_size=self._min_sz).astype(np.int32)

        self.signals.progress_percent.emit(100)
        return {"masks": mask.astype(np.int32)}


def export_mask(mask: np.ndarray,
                selected_ids: "set | None" = None,
                z_slice: "int | None" = None) -> np.ndarray:
    """Unified export function that respects selection and slice.
    Parameters"""
    out = mask.astype(np.int32).copy()

    # 1. Selection filtering
    if selected_ids is not None:
        all_lbls = set(np.unique(out)) - {0}
        for lbl in all_lbls:
            if lbl not in selected_ids:
                out[out == lbl] = 0

    # 2. Slice filtering
    if z_slice is not None and out.ndim == 3:
        z = min(int(z_slice), out.shape[0] - 1)
        out = out[z]

    return out


class ActiveSliceSelector(QWidget):
    """Compact widget exposing the global working-slice selector.
    Wraps AppState.active_z_slice and broadcasts changes via state.notify()."""
    sliceChanged = pyqtSignal(object)   # emits int or None

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self._state = state
        self._build()
        state.subscribe(self._on_state_change)

    def _build(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(QLabel("Working slice:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Full 3D", "Slice mode"])
        self._mode_combo.setToolTip(
            "Full 3D: all tools use the entire Z stack.\n"
            "Slice mode: all tools use only the selected Z slice.")
        self._z_spin = QSpinBox()
        self._z_spin.setRange(0, 0)
        self._z_spin.setButtonSymbols(QSpinBox.NoButtons)
        self._z_spin.setMinimumHeight(28)
        self._z_spin.setEnabled(False)
        lay.addWidget(self._mode_combo)
        lay.addWidget(self._z_spin)
        self._mode_combo.currentIndexChanged.connect(self._on_change)
        self._z_spin.valueChanged.connect(self._on_change)

    def _on_state_change(self, what: str):
        if what in ("raw_image", "preprocessed"):
            img = (self._state.preprocessed_image
                   if self._state.has_preprocessed()
                   else self._state.raw_image)
            if img is not None:
                self._z_spin.setMaximum(max(0, img.shape[0] - 1))

    def _on_change(self, _=None):
        if self._mode_combo.currentIndex() == 0:
            self._z_spin.setEnabled(False)
            self._state.active_z_slice = None
        else:
            self._z_spin.setEnabled(True)
            self._state.active_z_slice = self._z_spin.value()
        self._state.notify("active_z_slice")
        self.sliceChanged.emit(self._state.active_z_slice)

    def current_z(self) -> "int | None":
        return self._state.active_z_slice


class ExportMaskDialog(QDialog):
    """Dialog for choosing export scope: full 3D / current slice / propagated projection."""

    def __init__(self, state: AppState, mask_3d: np.ndarray,
                 selected_ids: "set | None", title: str = "Export Mask",
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedWidth(340)
        self._state    = state
        self._mask_3d  = mask_3d
        self._sel_ids  = selected_ids
        self._result   = None

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Export options:"))

        self._full_rb  = QRadioButton("Full 3D mask")
        self._slice_rb = QRadioButton("Current slice only")
        self._prop_rb  = QRadioButton("Max projection across Z (label-preserving)")

        self._full_rb.setChecked(True)

        for rb in [self._full_rb, self._slice_rb, self._prop_rb]:
            lay.addWidget(rb)

        has_3d = (mask_3d is not None and mask_3d.ndim == 3 and mask_3d.shape[0] > 1)
        self._prop_rb.setEnabled(has_3d)
        self._slice_rb.setEnabled(has_3d)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._on_ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _on_ok(self):
        mask = self._mask_3d
        if mask is None:
            self.reject(); return
        z = self._state.active_z_slice

        if self._slice_rb.isChecked():
            self._result = export_mask(mask, self._sel_ids, z_slice=z)
        elif self._prop_rb.isChecked() and mask.ndim == 3:
            # Filter by selection first, then project
            pre = export_mask(mask, self._sel_ids, z_slice=None)
            self._result = propagated_max_projection(pre)
        else:
            self._result = export_mask(mask, self._sel_ids, z_slice=None)
        self.accept()

    def get_result(self) -> "np.ndarray | None":
        return self._result


# --- Cell Body & Nuclei Detection Tab ---
# Cellpose segmentation for cell bodies and nuclei; interactive mask editing
class CellBodyNucleiTab(QWidget):

    CELL_BOUNDARY_COLORS = {
        "Green": (0, 255, 0), "Red": (255, 0, 0), "Blue": (0, 0, 255),
        "Yellow": (255, 255, 0), "Cyan": (0, 255, 255), "White": (255, 255, 255),
    }
    NUC_BOUNDARY_COLORS = {
        "Cyan": (0, 255, 255), "Magenta": (255, 0, 255), "Yellow": (255, 255, 0),
        "White": (255, 255, 255), "Green": (0, 255, 0), "Red": (255, 0, 0),
    }

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self._state = state

        self._cell_mask      = None
        self._cell_included  = set()
        self._cell_undo      = []
        self._cell_redo      = []
        self._cell_worker    = None

        self._nuc_mask       = None
        self._nuc_included   = set()
        self._nuc_undo       = []
        self._nuc_redo       = []
        self._nuc_worker     = None

        self._drawing_pts    = []
        self._is_drawing     = False
        self._draw_target    = "cell"   # "cell" or "nuc"
        self._cell_mask_3d   = None   # (Z,Y,X) int32 — 3D cell body result
        self._nuc_mask_3d    = None   # (Z,Y,X) int32 — 3D nucleus result
        self._cell_is_3d     = False
        self._nuc_is_3d      = False

        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        ls, _, ll = make_scroll_widget()
        ls.setFixedWidth(310)

        # Help button (top of left panel)
        _help_row = QHBoxLayout()
        _help_row.addStretch()
        _help_row.addWidget(HelpButton("Cell Body Nuclei Detection"))
        ll.addLayout(_help_row)

        chan_grp = section("Channel Assignment")
        chan_lyt = QFormLayout(chan_grp)
        self._cell_ch_combo = QComboBox()
        self._cell_ch_combo.setToolTip(
            "Channel for cell body segmentation.\n\n"
            "Raw channels: assigned in Image Viewer → Channel Mapping.\n"
            "★ Preprocessed channels: created in Preprocessing tab\n"
            "  using the 'Cell Body (CCT1)' preset → Store as Derived.\n\n"
            "Workflow for best results:\n"
            "1. Go to Preprocessing tab\n"
            "2. Click 'Preset: Cell Body (CCT1)'\n"
            "3. Click '▶ Apply Preprocessing'\n"
            "4. Click '📦 Store Preprocessed Channel' (name: preprocessed_cell_body)\n"
            "5. Return here and select '★ Preprocessed Cell Body [preprocessed]'\n"
            "6. Run Segmentation — pure Cellpose on your prepared image.")
        self._nuc_ch_combo = QComboBox()
        self._nuc_ch_combo.setToolTip(
            "Channel for nucleus segmentation.\n\n"
            "Raw channels: assigned in Image Viewer → Channel Mapping.\n"
            "★ Preprocessed channels: created in Preprocessing tab\n"
            "  using the 'Nucleus (fluorescent)' preset → Store as Derived.\n\n"
            "Workflow for best results:\n"
            "1. Go to Preprocessing tab\n"
            "2. Click 'Preset: Nucleus (fluorescent)'\n"
            "3. Click '▶ Apply Preprocessing'\n"
            "4. Click '📦 Store Preprocessed Channel' (name: preprocessed_nucleus)\n"
            "5. Return here and select '★ Preprocessed Nucleus [preprocessed]'\n"
            "6. Run Segmentation — pure Cellpose on your prepared image.")
        chan_lyt.addRow("Cell body channel:", self._cell_ch_combo)
        chan_lyt.addRow("Nucleus channel:",   self._nuc_ch_combo)
        ll.addWidget(chan_grp)

        # active_z_slice is now set exclusively via state.active_z_slice
        # which downstream tools (Region Analysis, exports) already respect.


        # ── Cell body segmentation mode ───────────────────────────────────────
        cell_mode_grp = section("Cell Body Segmentation Mode")
        cell_mode_lyt = QFormLayout(cell_mode_grp)
        self._cell_mode_combo = QComboBox()
        self._cell_mode_combo.addItems(SEG_MODES)
        self._cell_mode_combo.setCurrentIndex(1)  # default: max projection

        self._cell_mode_combo.setToolTip(
            "How to collapse the Z-stack before cell body segmentation:\n\n"
            "Slice (Z)       — segment a single Z plane (set Z-slice below).\n"
            "Max Projection  — brightest pixel across all Z planes (default).\n"
            "Mean Projection — average intensity across all Z planes.\n"
            "All slices (Z)  — run Cellpose on every Z plane, merge label stacks.\n"
            "                  Then use 'Select Active Cell Layer' to choose\n"
            "                  which Z layer feeds Region Analysis and exports.")
        self._cell_z_spin = QSpinBox()
        self._cell_z_spin.setRange(0, 0)
        self._cell_z_spin.setButtonSymbols(QSpinBox.NoButtons)
        self._cell_z_spin.setMinimumHeight(28)
        self._cell_z_spin.setEnabled(False)
        self._cell_z_spin.setToolTip(
            "Z-plane index for cell body segmentation.\n\n"
            "Active only when mode is 'Slice (Z)' or 'All slices (Z)'.\n"
            "0 = first (bottom) plane; maximum = top of Z-stack.")
        cell_mode_lyt.addRow("Mode:", self._cell_mode_combo)
        cell_mode_lyt.addRow("Z-slice (Slice mode):", self._cell_z_spin)
        ll.addWidget(cell_mode_grp)


        # ── Nucleus segmentation mode (identical to cell body) ───────────────────
        nuc_mode_grp = section("Nucleus Segmentation Mode")
        nuc_mode_lyt = QFormLayout(nuc_mode_grp)
        self._nuc_mode_combo = QComboBox()
        self._nuc_mode_combo.addItems(SEG_MODES)
        self._nuc_mode_combo.setCurrentIndex(1)  # default: max projection
        self._nuc_mode_combo.setToolTip(
            "How to collapse the Z-stack before nucleus segmentation:\n\n"
            "Slice (Z)       — segment a single Z plane (set Z-slice below).\n"
            "Max Projection  — brightest pixel across all Z planes (default).\n"
            "Mean Projection — average intensity across all Z planes.\n"
            "All slices (Z)  — run Cellpose on every Z plane, merge label stacks.\n"
            "                  Then use 'Select Active Nucleus Layer' to choose\n"
            "                  which Z layer feeds Region Analysis and exports.")
        self._nuc_z_spin = QSpinBox()
        self._nuc_z_spin.setRange(0, 0)
        self._nuc_z_spin.setButtonSymbols(QSpinBox.NoButtons)
        self._nuc_z_spin.setMinimumHeight(28)
        self._nuc_z_spin.setEnabled(False)
        self._nuc_z_spin.setToolTip(
            "Z-plane index for nucleus segmentation.\n\n"
            "Active only when mode is 'Slice (Z)' or 'All slices (Z)'.\n"
            "0 = first (bottom) plane; maximum = top of Z-stack.")
        nuc_mode_lyt.addRow("Mode:", self._nuc_mode_combo)
        nuc_mode_lyt.addRow("Z-slice (Slice mode):", self._nuc_z_spin)
        ll.addWidget(nuc_mode_grp)


        cb_grp = section("Cell Body Parameters")
        cb_lyt = QVBoxLayout(cb_grp)

        self._cell_diameter_spin = QSpinBox()
        self._cell_diameter_spin.setRange(1, 1000)
        self._cell_diameter_spin.setValue(100)

        self._cell_diameter_spin.setToolTip(
            "Approximate cell body diameter in pixels.\n\n"
            "This is the MOST IMPORTANT parameter — measure a representative\n"
            "cell edge-to-edge in the image before setting this.\n\n"
            "Typical values by magnification / cell type:\n"
            "  • 10× objective, large cells → 80–200 px\n"
            "  • 20× objective, medium cells → 40–120 px\n"
            "  • 40× objective, small cells  → 20–70 px\n\n"
            "Troubleshooting:\n"
            "  • Cells MERGE into one blob   → decrease diameter by 20–30%\n"
            "  • Each cell splits into PARTS → increase diameter by 20–30%\n"
            "  • Set to 0 for auto-estimate  → slower but dataset-adaptive\n\n"
            "Note: the adaptive preprocessor (u5) now normalises intensity\n"
            "automatically, so you can focus on setting diameter correctly.")
        cb_lyt.addLayout(_make_spinbox_row("Diameter (px):", self._cell_diameter_spin))
        self._cell_flow_spin = QDoubleSpinBox()
        self._cell_flow_spin.setRange(0.0, 1.5)
        self._cell_flow_spin.setSingleStep(0.05)
        self._cell_flow_spin.setDecimals(2)
        self._cell_flow_spin.setValue(0.4)

        self._cell_flow_spin.setToolTip(
            "Cellpose flow error threshold for cell bodies (0.0 – 1.5).\n\n"
            "Controls how strictly Cellpose enforces cell boundary flow vectors.\n"
            "Default: 0.4  (good for most datasets).\n\n"
            "Troubleshooting guide:\n"
            "  • Cells MERGE into one large mask  → lower to 0.2–0.3\n"
            "  • Masks LEAK beyond true borders   → lower to 0.2–0.3\n"
            "  • Real cells MISSING from result   → raise to 0.5–0.8\n"
            "  • Dense, touching cell populations → use 0.2–0.35\n"
            "  • Sparse, well-separated cells     → 0.4–0.6 works well\n"
            "  • Dim / weak fluorescence          → raise slightly (0.5–0.7)\n\n"
            "Adjust in steps of 0.05–0.10 and re-run to see the effect.")
        cb_lyt.addLayout(_make_spinbox_row("Flow threshold:", self._cell_flow_spin))
        self._cell_cellprob_spin = QDoubleSpinBox()
        self._cell_cellprob_spin.setRange(-6.0, 6.0)
        self._cell_cellprob_spin.setSingleStep(0.1)
        self._cell_cellprob_spin.setDecimals(2)
        self._cell_cellprob_spin.setValue(0.0)

        self._cell_cellprob_spin.setToolTip(
            "Cell probability threshold for cell bodies (–6.0 to +6.0).\n\n"
            "Controls which pixels Cellpose accepts as belonging to a cell.\n"
            "Default: 0.0.\n\n"
            "Troubleshooting guide:\n"
            "  • Dim cells not detected at all    → lower to –1.0 to –3.0\n"
            "  • Masks include too much background → raise to +0.5 to +2.0\n"
            "  • Touching cells merge             → raise slightly (+0.5)\n"
            "  • Cells appear undersized/too small → lower to –0.5 to –1.5\n\n"
            "Tip: adjust Cell prob FIRST before changing Flow threshold.\n"
            "The adaptive preprocessor (u5) normalises intensity automatically,\n"
            "so you rarely need values below –2.0.")
        cb_lyt.addLayout(_make_spinbox_row("Cell prob:", self._cell_cellprob_spin))
        self._cell_min_sz = QSpinBox()
        self._cell_min_sz.setRange(0, 100000)
        self._cell_min_sz.setValue(2000)

        self._cell_min_sz.setToolTip(
            "Minimum cell body area in pixels (2D projected).\n\n"
            "Objects smaller than this are discarded as debris / noise.\n\n"
            "Typical values by magnification:\n"
            "  • 10× objective → 500 – 2 000 px\n"
            "  • 20× objective → 2 000 – 8 000 px\n"
            "  • 40× objective → 8 000 – 30 000 px\n\n"
            "Troubleshooting:\n"
            "  • Too many small fragments → increase (×2 or ×3)\n"
            "  • Small real cells removed → decrease\n"
            "  • Rule of thumb: ~25% of expected cell area")
        cb_lyt.addLayout(_make_spinbox_row("Min area (px):", self._cell_min_sz))
        self._cell_auto_btn = QPushButton("🔍  Auto Parameters (Cell)")
        self._cell_auto_btn.setStyleSheet(BTN_PURPLE)
        self._cell_auto_btn.setEnabled(False)
        cb_lyt.addWidget(self._cell_auto_btn)
        self._cell_run_btn = QPushButton("▶  Run Cell Body Segmentation")
        self._cell_run_btn.setStyleSheet(BTN_RUN)
        self._cell_run_btn.setEnabled(False)
        cb_lyt.addWidget(self._cell_run_btn)
        ll.addWidget(cb_grp)

        nuc_grp = section("Nucleus Parameters")
        nuc_lyt = QVBoxLayout(nuc_grp)

        self._nuc_diameter_spin = QSpinBox()
        self._nuc_diameter_spin.setRange(1, 1000)
        self._nuc_diameter_spin.setValue(83)

        self._nuc_diameter_spin.setToolTip(
            "Approximate nucleus diameter in pixels.\n\n"
            "The MOST IMPORTANT parameter — measure a representative nucleus\n"
            "edge-to-edge before setting this.\n\n"
            "Typical values by cell type:\n"
            "  • Small cells (HeLa, HEK293)  → 30 – 60 px\n"
            "  • Medium cells (fibroblasts)  → 60 – 100 px\n"
            "  • Large cells (neurons, iPSC) → 80 – 150 px\n\n"
            "Troubleshooting:\n"
            "  • Nuclei appear MERGED         → decrease diameter by 20–30%\n"
            "  • Each nucleus splits into TWO → increase diameter by 20–30%\n"
            "  • Elongated nuclei fail        → the adaptive preprocessor\n"
            "    (u5) + augment=True now improves this automatically\n"
            "  • Set to 0 for auto-estimate   → slower but dataset-adaptive")
        nuc_lyt.addLayout(_make_spinbox_row("Diameter (px):", self._nuc_diameter_spin))
        self._nuc_flow_spin = QDoubleSpinBox()
        self._nuc_flow_spin.setRange(0.0, 1.5)
        self._nuc_flow_spin.setSingleStep(0.05)
        self._nuc_flow_spin.setDecimals(2)
        self._nuc_flow_spin.setValue(0.4)

        self._nuc_flow_spin.setToolTip(
            "Cellpose flow error threshold for nuclei (0.0 – 1.5).\n\n"
            "Default: 0.4  (good starting point for most nucleus types).\n\n"
            "Troubleshooting guide:\n"
            "  • Touching nuclei MERGE         → lower to 0.2–0.3\n"
            "  • Elongated nuclei fragment     → raise to 0.5–0.7\n"
            "  • Missing dim nuclei            → raise to 0.5–0.8\n"
            "  • Dense packed nuclei (DAPI)    → lower to 0.2–0.35\n"
            "  • Irregular / non-round nuclei  → raise to 0.5–0.6\n\n"
            "Note (u5): segmentation now uses test-time augmentation\n"
            "(augment=True) which significantly improves elongated nucleus\n"
            "detection — you may need less flow threshold adjustment.")
        nuc_lyt.addLayout(_make_spinbox_row("Flow threshold:", self._nuc_flow_spin))
        self._nuc_cellprob_spin = QDoubleSpinBox()
        self._nuc_cellprob_spin.setRange(-6.0, 6.0)
        self._nuc_cellprob_spin.setSingleStep(0.1)
        self._nuc_cellprob_spin.setDecimals(2)
        self._nuc_cellprob_spin.setValue(0.0)

        self._nuc_cellprob_spin.setToolTip(
            "Cell probability threshold for nuclei (–6.0 to +6.0).\n\n"
            "Controls which pixels are accepted as nuclear.\n"
            "Default: 0.0.\n\n"
            "Troubleshooting guide:\n"
            "  • Weak/dim nuclei not detected  → lower to –1.0 to –3.0\n"
            "  • Nuclei include background     → raise to +0.5 to +2.0\n"
            "  • Touching nuclei merged        → raise slightly (+0.5)\n"
            "  • Very bright nuclei oversized  → raise to +1.0 to +2.0\n\n"
            "Note (u5): the adaptive preprocessor normalises intensity\n"
            "automatically — you rarely need values below –2.0.")
        nuc_lyt.addLayout(_make_spinbox_row("Cell prob:", self._nuc_cellprob_spin))
        self._nuc_min_sz = QSpinBox()
        self._nuc_min_sz.setRange(0, 50000)
        self._nuc_min_sz.setValue(200)

        self._nuc_min_sz.setToolTip(
            "Minimum nucleus area in pixels (2D projected).\n\n"
            "Objects smaller than this are discarded as nuclear fragments.\n\n"
            "Typical values:\n"
            "  • Small nuclei (20× HeLa)    → 100 – 400 px\n"
            "  • Medium nuclei (20× fibro)  → 300 – 800 px\n"
            "  • Large nuclei (10× neurons) → 500 – 2 000 px\n\n"
            "Troubleshooting:\n"
            "  • Too many small fragments  → increase (×2 or ×3)\n"
            "  • Small nuclei removed      → decrease\n"
            "  • Oversized merged nuclei   → the u5 split algorithm\n"
            "    handles these automatically when area > 4× expected")
        nuc_lyt.addLayout(_make_spinbox_row("Min area (px):", self._nuc_min_sz))

        self._nuc_auto_btn = QPushButton("🔍  Auto Parameters (Nucleus)")
        self._nuc_auto_btn.setStyleSheet(BTN_PURPLE)
        self._nuc_auto_btn.setEnabled(False)
        nuc_lyt.addWidget(self._nuc_auto_btn)
        self._nuc_run_btn = QPushButton("▶  Run Nucleus Segmentation")
        self._nuc_run_btn.setStyleSheet(BTN_RUN)
        self._nuc_run_btn.setEnabled(False)
        nuc_lyt.addWidget(self._nuc_run_btn)
        ll.addWidget(nuc_grp)

        draw_grp = section("Manual Mask Drawing")
        draw_lyt = QVBoxLayout(draw_grp)
        draw_target_row = QHBoxLayout()
        draw_target_row.addWidget(QLabel("Draw for:"))
        self._draw_target_combo = QComboBox()
        self._draw_target_combo.addItems(["Cell Bodies", "Nuclei"])
        draw_target_row.addWidget(self._draw_target_combo)
        draw_lyt.addLayout(draw_target_row)
        self._draw_btn     = QPushButton("✏  Start Drawing Polygon")
        self._finalize_btn = QPushButton("✓  Finalize Polygon Mask")
        self._finalize_btn.setEnabled(False)
        draw_lyt.addWidget(self._draw_btn)
        draw_lyt.addWidget(self._finalize_btn)
        ll.addWidget(draw_grp)

        hist_grp = section("History")
        hist_lyt = QHBoxLayout(hist_grp)
        self._undo_btn = QPushButton("↩ Undo")
        self._redo_btn = QPushButton("↪ Redo")
        self._undo_btn.setEnabled(False)
        self._redo_btn.setEnabled(False)
        hist_lyt.addWidget(self._undo_btn)
        hist_lyt.addWidget(self._redo_btn)
        self._reset_btn = QPushButton("🗑 Reset")
        self._reset_btn.setToolTip("Clear all segmentation results and reset to a fresh state.")
        hist_lyt.addWidget(self._reset_btn)
        ll.addWidget(hist_grp)

        cell_ov_grp = section("Cell Body Overlay")
        cell_ov_lyt = QVBoxLayout(cell_ov_grp)
        self._cell_show_orig = QCheckBox("Show original image")
        self._cell_show_orig.setChecked(True)
        self._cell_show_mask = QCheckBox("Show cell masks")
        self._cell_show_mask.setChecked(False)
        self._cell_show_bnd  = QCheckBox("Show cell outlines")
        self._cell_show_bnd.setChecked(True)
        self._cell_show_num  = QCheckBox("Show cell IDs")
        self._cell_show_num.setChecked(True)
        cell_bc_row = QHBoxLayout()
        cell_bc_row.addWidget(QLabel("Outline color:"))
        self._cell_bc_combo = QComboBox()
        self._cell_bc_combo.addItems(list(self.CELL_BOUNDARY_COLORS.keys()))
        cell_bc_row.addWidget(self._cell_bc_combo)
        for w in [self._cell_show_orig, self._cell_show_mask,
                  self._cell_show_bnd, self._cell_show_num]:
            cell_ov_lyt.addWidget(w)
        cell_ov_lyt.addLayout(cell_bc_row)
        ll.addWidget(cell_ov_grp)

        nuc_ov_grp = section("Nucleus Overlay")
        nuc_ov_lyt = QVBoxLayout(nuc_ov_grp)
        self._nuc_show_orig = QCheckBox("Show original image")
        self._nuc_show_orig.setChecked(True)
        self._nuc_show_mask = QCheckBox("Show nucleus masks")
        self._nuc_show_mask.setChecked(False)
        self._nuc_show_bnd  = QCheckBox("Show nucleus outlines")
        self._nuc_show_bnd.setChecked(True)
        self._nuc_show_num  = QCheckBox("Show nucleus IDs")
        self._nuc_show_num.setChecked(True)
        nuc_bc_row = QHBoxLayout()
        nuc_bc_row.addWidget(QLabel("Outline color:"))
        self._nuc_bc_combo = QComboBox()
        self._nuc_bc_combo.addItems(list(self.NUC_BOUNDARY_COLORS.keys()))
        nuc_bc_row.addWidget(self._nuc_bc_combo)
        for w in [self._nuc_show_orig, self._nuc_show_mask,
                  self._nuc_show_bnd, self._nuc_show_num]:
            nuc_ov_lyt.addWidget(w)
        nuc_ov_lyt.addLayout(nuc_bc_row)
        ll.addWidget(nuc_ov_grp)


        comb_grp = section("Combined Overlay")
        comb_lyt = QVBoxLayout(comb_grp)

        # Cell body options in combined view
        _comb_cell_hdr = QLabel("Cell Bodies:")
        _comb_cell_hdr.setStyleSheet("color:#60c060;font-weight:bold;font-size:11px;")
        comb_lyt.addWidget(_comb_cell_hdr)
        self._comb_show_cell_orig  = QCheckBox("Show original image")
        self._comb_show_cell_orig.setChecked(True)
        self._comb_show_cell_orig.setToolTip(
            "Show the microscopy image as a grayscale background in the Combined view.\n"
            "Uncheck to see only the overlay masks on black.")
        self._comb_show_cell_mask  = QCheckBox("Show cell masks")
        self._comb_show_cell_mask.setChecked(False)
        self._comb_show_cell_mask.setToolTip(
            "Overlay semi-transparent filled cell body masks in the Combined view.\n"
            "Each cell is colour-coded by its label ID.")
        self._comb_show_cell_bnd   = QCheckBox("Show cell outlines")
        self._comb_show_cell_bnd.setChecked(True)
        self._comb_show_cell_bnd.setToolTip(
            "Draw 1-px boundary outlines around each segmented cell body.\n"
            "Colour is set by 'Cell outline color' below.")
        self._comb_show_cell_ids   = QCheckBox("Show cell IDs")
        self._comb_show_cell_ids.setChecked(False)
        self._comb_show_cell_ids.setToolTip(
            "Draw numeric label IDs at the centroid of each cell body.")
        comb_cell_bc_row = QHBoxLayout()
        comb_cell_bc_row.addWidget(QLabel("Cell outline color:"))
        self._comb_cell_bc_combo = QComboBox()
        self._comb_cell_bc_combo.addItems(list(self.CELL_BOUNDARY_COLORS.keys()))
        self._comb_cell_bc_combo.setToolTip("Outline colour for cell bodies in the Combined view.")
        comb_cell_bc_row.addWidget(self._comb_cell_bc_combo)
        for w in [self._comb_show_cell_orig, self._comb_show_cell_mask,
                  self._comb_show_cell_bnd, self._comb_show_cell_ids]:
            comb_lyt.addWidget(w)
        comb_lyt.addLayout(comb_cell_bc_row)

        # Nucleus options in combined view
        _comb_nuc_hdr = QLabel("Nuclei:")
        _comb_nuc_hdr.setStyleSheet("color:#60a0e0;font-weight:bold;font-size:11px;margin-top:4px;")
        comb_lyt.addWidget(_comb_nuc_hdr)
        self._comb_show_nuc_mask   = QCheckBox("Show nucleus masks")
        self._comb_show_nuc_mask.setChecked(False)
        self._comb_show_nuc_mask.setToolTip(
            "Overlay semi-transparent filled nucleus masks in the Combined view.")
        self._comb_show_nuc_bnd    = QCheckBox("Show nucleus outlines")
        self._comb_show_nuc_bnd.setChecked(True)
        self._comb_show_nuc_bnd.setToolTip(
            "Draw 1-px boundary outlines around each segmented nucleus.\n"
            "Colour is set by 'Nucleus outline color' below.")
        self._comb_show_nuc_ids    = QCheckBox("Show nucleus IDs")
        self._comb_show_nuc_ids.setChecked(False)
        self._comb_show_nuc_ids.setToolTip(
            "Draw numeric label IDs at the centroid of each nucleus.")
        comb_nuc_bc_row = QHBoxLayout()
        comb_nuc_bc_row.addWidget(QLabel("Nucleus outline color:"))
        self._comb_nuc_bc_combo = QComboBox()
        self._comb_nuc_bc_combo.addItems(list(self.NUC_BOUNDARY_COLORS.keys()))
        self._comb_nuc_bc_combo.setToolTip("Outline colour for nuclei in the Combined view.")
        comb_nuc_bc_row.addWidget(self._comb_nuc_bc_combo)
        for w in [self._comb_show_nuc_mask, self._comb_show_nuc_bnd, self._comb_show_nuc_ids]:
            comb_lyt.addWidget(w)
        comb_lyt.addLayout(comb_nuc_bc_row)

        self._combine_btn = QPushButton("🔀  Refresh Combined Overlay")
        self._combine_btn.setStyleSheet(BTN_PURPLE)
        self._combine_btn.setEnabled(False)
        self._combine_btn.setToolTip(
            "Render the Combined overlay using the settings above.\n\n"
            "Shows cell body outlines and nucleus outlines on a single canvas.\n"
            "Requires at least one of: cell body mask, nucleus mask.")
        comb_lyt.addWidget(self._combine_btn)
        ll.addWidget(comb_grp)

        self._prog     = QProgressBar()
        self._prog.setRange(0, 0)
        self._prog.setVisible(False)
        self._stat_lbl = StatusLabel()
        ll.addWidget(self._prog)
        ll.addWidget(self._stat_lbl)
        ll.addStretch()
        root.addWidget(ls)

        self._view_tabs = QTabWidget()
        self._view_tabs.setDocumentMode(True)

        cell_w = QWidget()
        cell_l = QVBoxLayout(cell_w)
        cell_l.setContentsMargins(0, 0, 0, 0)
        self._cell_canvas = ImageCanvas(figsize=(7, 6))
        self._cell_nav    = NavigationToolbar(self._cell_canvas, self)
        self._cell_nav.setStyleSheet(_viewer_tb_qss())
        cell_l.addWidget(self._cell_nav)
        cell_l.addWidget(self._cell_canvas, stretch=1)

        nuc_w = QWidget()
        nuc_l = QVBoxLayout(nuc_w)
        nuc_l.setContentsMargins(0, 0, 0, 0)
        self._nuc_canvas = ImageCanvas(figsize=(7, 6))
        self._nuc_nav    = NavigationToolbar(self._nuc_canvas, self)
        self._nuc_nav.setStyleSheet(_viewer_tb_qss())
        nuc_l.addWidget(self._nuc_nav)
        nuc_l.addWidget(self._nuc_canvas, stretch=1)

        comb_w = QWidget()
        comb_l = QVBoxLayout(comb_w)
        comb_l.setContentsMargins(0, 0, 0, 0)
        self._comb_canvas = ImageCanvas(figsize=(7, 6))
        self._comb_nav    = NavigationToolbar(self._comb_canvas, self)
        self._comb_nav.setStyleSheet(_viewer_tb_qss())
        comb_l.addWidget(self._comb_nav)
        comb_l.addWidget(self._comb_canvas, stretch=1)

        self._view_tabs.addTab(cell_w,  "Cell Bodies")
        self._view_tabs.addTab(nuc_w,   "Nuclei")
        self._view_tabs.addTab(comb_w,  "Combined 🔀")
        root.addWidget(self._view_tabs, stretch=1)

        rw = QWidget()
        rw.setFixedWidth(240)
        rl = QVBoxLayout(rw)

        cell_stat_grp = section("Cell Body Stats")
        self._cell_stats = QLabel("")
        self._cell_stats.setWordWrap(True)
        self._cell_stats.setStyleSheet("color:#80ff80;font-family:monospace;font-size:11px;")
        QVBoxLayout(cell_stat_grp).addWidget(self._cell_stats)
        rl.addWidget(cell_stat_grp)

        nuc_stat_grp = section("Nucleus Stats")
        self._nuc_stats = QLabel("")
        self._nuc_stats.setWordWrap(True)
        self._nuc_stats.setStyleSheet("color:#80c0ff;font-family:monospace;font-size:11px;")
        QVBoxLayout(nuc_stat_grp).addWidget(self._nuc_stats)
        rl.addWidget(nuc_stat_grp)


        sel_grp = section("Active Segmentation Layer")
        sel_lyt = QVBoxLayout(sel_grp)
        _sel_info = QLabel(
            "When 'All slices (Z)' is used, each Z plane is segmented separately.\n"
            "Use these buttons to pick which Z layer feeds into Region Analysis,\n"
            "exports, statistics and overlays.")
        _sel_info.setWordWrap(True)
        _sel_info.setStyleSheet("color:#7070a0;font-size:10px;")
        sel_lyt.addWidget(_sel_info)
        self._sel_cell_btn = QPushButton("🗂  Select Active Cell Layer…")
        self._sel_nuc_btn  = QPushButton("🗂  Select Active Nucleus Layer…")
        self._sel_cell_btn.setToolTip(
            "Choose which segmented Z-layer to use as the active cell body mask.\n\n"
            "Only relevant after 'All slices (Z)' segmentation where each Z plane\n"
            "has its own mask. The selected layer feeds into:\n"
            "  • Cell Region Analysis\n"
            "  • Statistics panel\n"
            "  • Exports (TIF / PDF)\n"
            "  • Overlay display\n\n"
            "For single-plane segmentation (Max/Mean/Slice), Z0 is the only layer.\n"
            "Tip: use the Z-slice spin in the mode section to preview any layer first.")
        self._sel_nuc_btn.setToolTip(
            "Choose which segmented Z-layer to use as the active nucleus mask.\n\n"
            "Only relevant after 'All slices (Z)' segmentation where each Z plane\n"
            "has its own mask. The selected layer feeds into:\n"
            "  • Cell Region Analysis\n"
            "  • Statistics panel\n"
            "  • Exports (TIF / PDF)\n"
            "  • Overlay display\n\n"
            "For single-plane segmentation (Max/Mean/Slice), Z0 is the only layer.")
        self._sel_cell_btn.setEnabled(False)
        self._sel_nuc_btn.setEnabled(False)
        for b in [self._sel_cell_btn, self._sel_nuc_btn]:
            sel_lyt.addWidget(b)
        rl.addWidget(sel_grp)

        exp_grp = section("Export")
        exp_lyt = QVBoxLayout(exp_grp)
        self._exp_cell_btn    = QPushButton("Export Cell Masks (TIF)")
        self._exp_nuc_btn     = QPushButton("Export Nucleus Masks (TIF)")
        self._exp_overlay_btn = QPushButton("Export Combined Overlay (TIF)")
        self._exp_pdf_btn     = QPushButton("Export PDF Report")
        for b in [self._exp_cell_btn, self._exp_nuc_btn,
                  self._exp_overlay_btn, self._exp_pdf_btn]:
            b.setEnabled(False)
            exp_lyt.addWidget(b)
        rl.addWidget(exp_grp)
        rl.addStretch()
        root.addWidget(rw)

        self._cell_run_btn.clicked.connect(self._run_cell_seg)
        self._nuc_run_btn.clicked.connect(self._run_nuc_seg)
        self._draw_btn.clicked.connect(self._start_draw)
        self._finalize_btn.clicked.connect(self._finalize_poly)
        self._undo_btn.clicked.connect(self._undo)
        self._redo_btn.clicked.connect(self._redo)
        self._reset_btn.clicked.connect(self._reset_state)
        self._combine_btn.clicked.connect(self._refresh_combined)
        self._cell_auto_btn.clicked.connect(self._run_cell_auto_params)
        self._nuc_auto_btn.clicked.connect(self._run_nuc_auto_params)

        for cb in [self._cell_show_orig, self._cell_show_mask,
                   self._cell_show_bnd, self._cell_show_num]:
            cb.stateChanged.connect(lambda _: self._refresh_cells())
        self._cell_bc_combo.currentTextChanged.connect(lambda _: self._refresh_cells())

        for cb in [self._nuc_show_orig, self._nuc_show_mask,
                   self._nuc_show_bnd, self._nuc_show_num]:
            cb.stateChanged.connect(lambda _: self._refresh_nuclei())
        self._nuc_bc_combo.currentTextChanged.connect(lambda _: self._refresh_nuclei())


        for cb in [self._comb_show_cell_orig, self._comb_show_cell_mask,
                   self._comb_show_cell_bnd, self._comb_show_cell_ids,
                   self._comb_show_nuc_mask, self._comb_show_nuc_bnd,
                   self._comb_show_nuc_ids]:
            cb.stateChanged.connect(lambda _: self._refresh_combined())
        self._comb_cell_bc_combo.currentTextChanged.connect(lambda _: self._refresh_combined())
        self._comb_nuc_bc_combo.currentTextChanged.connect(lambda _: self._refresh_combined())

        self._cell_canvas.mpl_connect("button_press_event", self._on_cell_click)
        self._nuc_canvas.mpl_connect("button_press_event",  self._on_nuc_click)
        self._cell_mode_combo.currentIndexChanged.connect(self._on_proj_change)
        self._nuc_mode_combo.currentIndexChanged.connect(self._on_proj_change)
        self._cell_z_spin.valueChanged.connect(lambda _: self._refresh_cells())
        self._nuc_z_spin.valueChanged.connect(lambda _: self._refresh_nuclei())
        self._exp_cell_btn.clicked.connect(self._export_cell_mask)
        self._exp_nuc_btn.clicked.connect(self._export_nuc_mask)
        self._exp_overlay_btn.clicked.connect(self._export_combined_overlay)
        self._exp_pdf_btn.clicked.connect(self._export_pdf)
        self._sel_cell_btn.clicked.connect(self._open_cell_selection_dialog)
        self._sel_nuc_btn.clicked.connect(self._open_nuc_selection_dialog)
        self._state.subscribe(self._on_state_change)

    def _on_state_change(self, what):
        if what in ("raw_image", "preprocessed", "channel_mappings",
                    "derived_channels"):
            arr = (self._state.preprocessed_image  # r3 - proper conditional, avoids numpy or-ambiguity
                   if self._state.has_preprocessed()
                   else self._state.raw_image)
            if arr is not None:
                nz = arr.shape[0]
                for sp in [self._nuc_z_spin, self._cell_z_spin]:
                    sp.setRange(0, max(0, nz-1))
                self._cell_run_btn.setEnabled(True)
                self._nuc_run_btn.setEnabled(True)
                self._cell_auto_btn.setEnabled(True)
                self._nuc_auto_btn.setEnabled(True)

            _populate_channel_combo_with_derived(
                self._cell_ch_combo, self._state, preferred_label="CCT1")
            _populate_channel_combo_with_derived(
                self._nuc_ch_combo,  self._state, preferred_label="Nucleus")
            if arr is not None:
                self._refresh_cells()
                self._refresh_nuclei()

        elif what == "active_z_slice":
            self._refresh_cells()
            self._refresh_nuclei()

    def _on_proj_change(self, _=None):
        # Enable Z-spin for both single-slice and 3D-propagation modes
        cell_txt = self._cell_mode_combo.currentText()
        nuc_txt  = self._nuc_mode_combo.currentText()
        self._cell_z_spin.setEnabled(cell_txt in (SEG_MODE_SLICE, SEG_MODE_3D))
        self._nuc_z_spin.setEnabled(nuc_txt  in (SEG_MODE_SLICE, SEG_MODE_3D))
        self._refresh_cells()
        self._refresh_nuclei()

    def _get_slice(self, channel_combo, mode_combo, z_spin):
        if not self._state.has_image(): return None  # r3 - raw image is enough; derived channels work without global preprocessing
        return _get_plane_from_combined_combo(
            channel_combo, self._state,
            mode=mode_combo.currentText(),
            z_idx=z_spin.value())

    def _run_cell_seg(self):
        if not _require_image(self._state, self):
            return
        diam     = float(self._cell_diameter_spin.value())
        flow     = self._cell_flow_spin.value()
        cellprob = self._cell_cellprob_spin.value()

        if self._cell_mode_combo.currentText() == SEG_MODE_3D:
            img = (self._state.preprocessed_image
                   if self._state.has_preprocessed()
                   else self._state.raw_image)
            if img is None:
                QMessageBox.warning(self, "No Image", "Load an image first."); return
            ch_idx = _channel_index_from_combo(self._cell_ch_combo, self._state)
            if ch_idx is None: ch_idx = 0
            self._cell_run_btn.setEnabled(False)
            self._prog.setVisible(True)
            self._prog.setRange(0, 100); self._prog.setValue(0); self._prog.setFormat("%p%")
            self._stat_lbl.info("Running 3D cell body segmentation…")
            self._cell_worker = _CellBody3DWorker(img, ch_idx, diam,
                                                   self._cell_min_sz.value(), parent=self)
            self._cell_worker.signals.progress.connect(self._stat_lbl.info)
            self._cell_worker.signals.progress_percent.connect(self._prog.setValue)
            self._cell_worker.signals.result.connect(self._on_cell_3d_done)
            self._cell_worker.signals.error.connect(self._on_cell_err)
            self._cell_worker.signals.finished.connect(lambda: (
                self._cell_run_btn.setEnabled(True),
                self._prog.setRange(0, 0), self._prog.setFormat(""),
                self._prog.setVisible(False)))
            self._cell_worker.start()
            return


        plane = self._get_slice(
            self._cell_ch_combo, self._cell_mode_combo, self._cell_z_spin)
        if plane is None:
            QMessageBox.warning(self, "No Image",
                "Could not build segmentation input. "
                "Load an image and configure channels first.")
            return
        img8 = (plane * 255).astype(np.uint8)

        self._cell_run_btn.setEnabled(False)
        self._prog.setVisible(True)

        self._prog.setRange(0, 100); self._prog.setValue(0); self._prog.setFormat("%p%")
        self._stat_lbl.info("Segmenting cell bodies (pure Cellpose)…")
        self._cell_worker = CellBodyWorker(img8, diam, self._cell_min_sz.value(),
                                            flow_threshold=flow,
                                            cellprob_threshold=cellprob, parent=self)
        self._cell_worker.signals.progress.connect(self._stat_lbl.info)

        self._cell_worker.signals.progress_percent.connect(self._prog.setValue)
        self._cell_worker.signals.result.connect(self._on_cell_done)
        self._cell_worker.signals.error.connect(self._on_cell_err)
        self._cell_worker.signals.finished.connect(lambda: (
            self._cell_run_btn.setEnabled(True),
            self._prog.setRange(0, 100), self._prog.setValue(100),
            self._prog.setFormat("Done"),
            QTimer.singleShot(800, lambda: self._prog.setVisible(False))))
        self._cell_worker.start()

    def _on_cell_done(self, res):
        masks = res.get("masks")
        if masks is None:
            self._stat_lbl.warn("No cell masks returned.")
            return
        self._push_cell_undo()
        self._cell_mask     = masks
        self._cell_included = set(np.unique(masks)) - {0}

        self._state.active_cell_ids["cell_body"] = set(self._cell_included)
        self._refresh_cells()
        self._update_cell_stats()
        self._exp_cell_btn.setEnabled(True)
        self._exp_pdf_btn.setEnabled(True)
        self._sel_cell_btn.setEnabled(True)
        if self._nuc_mask is not None:
            self._combine_btn.setEnabled(True)
            self._exp_overlay_btn.setEnabled(True)
        n = len(self._cell_included)
        self._stat_lbl.ok(f"Cell body done. {n} cells detected.")

        # CellRegionWorker can match each nucleus to its containing cell.
        # float32 with values = original label IDs preserves correspondence.
        cell_float = masks.astype(np.float32)
        if cell_float.ndim == 2:
            cell_float = cell_float[np.newaxis]
        self._state.set_derived_channel(
            key    = "cell_body_mask",
            mask   = cell_float,
            source = "cell_nuclei_tab",
        )
        self._push_cell_results()

    def _on_cell_err(self, msg):
        QMessageBox.critical(self, "Cell Segmentation Error", msg)
        self._stat_lbl.err("Cell body segmentation failed.")

    def _on_cell_3d_done(self, res):
        masks_3d = res.get("masks_3d")
        if masks_3d is None or masks_3d.max() == 0:
            self._stat_lbl.warn("3D cell segmentation returned empty result."); return
        self._push_cell_undo()
        self._cell_mask_3d  = masks_3d
        self._cell_is_3d    = True
        z = min(self._cell_z_spin.value(), masks_3d.shape[0] - 1)
        self._cell_mask     = masks_3d[z]
        self._cell_included = set(np.unique(masks_3d)) - {0}

        self._state.active_cell_ids["cell_body"] = set(self._cell_included)
        self._refresh_cells()
        self._update_cell_stats()
        self._exp_cell_btn.setEnabled(True)
        self._exp_pdf_btn.setEnabled(True)
        self._sel_cell_btn.setEnabled(True)
        if self._nuc_mask is not None:
            self._combine_btn.setEnabled(True)
            self._exp_overlay_btn.setEnabled(True)
        n = len(self._cell_included)
        self._stat_lbl.ok(f"3D cell done. {n} unique labels. Scroll Z to inspect.")
        self._push_cell_results()


    # subscribe to "active_z_slice" via _on_state_change directly.
    def _run_nuc_seg(self):
        if not _require_image(self._state, self): return
        img = self._state.preprocessed_image if self._state.has_preprocessed() else self._state.raw_image
        if img is None:
            QMessageBox.warning(self, "No image", "Load an image first."); return

        ch_idx = _channel_index_from_combo(self._nuc_ch_combo, self._state)
        if ch_idx is None: ch_idx = 0

        diam     = float(self._nuc_diameter_spin.value())
        flow     = self._nuc_flow_spin.value()
        cellprob = self._nuc_cellprob_spin.value()

        if self._nuc_mode_combo.currentText() == SEG_MODE_3D:
            self._nuc_run_btn.setEnabled(False)
            self._prog.setVisible(True)
            self._prog.setRange(0, 100); self._prog.setValue(0); self._prog.setFormat("%p%")
            self._stat_lbl.info("Running 3D nucleus segmentation…")
            self._nuc_worker = _Nucleus3DWorker(img, ch_idx, diam,
                                                self._nuc_min_sz.value(), parent=self)
            self._nuc_worker.signals.progress.connect(self._stat_lbl.info)
            self._nuc_worker.signals.progress_percent.connect(self._prog.setValue)
            self._nuc_worker.signals.result.connect(self._on_nuc_3d_done)
            self._nuc_worker.signals.error.connect(self._on_nuc_err)
            self._nuc_worker.signals.finished.connect(lambda: (
                self._nuc_run_btn.setEnabled(True),
                self._prog.setRange(0, 0), self._prog.setFormat(""),
                self._prog.setVisible(False)))
            self._nuc_worker.start()
            return


        plane = self._get_slice(
            self._nuc_ch_combo, self._nuc_mode_combo, self._nuc_z_spin)
        if plane is None:
            QMessageBox.warning(self, "No Image",
                "Could not get nucleus channel. Check channel assignment."); return

        self._nuc_run_btn.setEnabled(False)
        self._prog.setVisible(True)

        self._prog.setRange(0, 100); self._prog.setValue(0); self._prog.setFormat("%p%")
        self._stat_lbl.info("Segmenting nuclei (pure Cellpose)…")
        self._nuc_worker = NucleusWorker(plane, diam, self._nuc_min_sz.value(),
                                         flow_threshold=flow,
                                         cellprob_threshold=cellprob,
                                         parent=self)
        self._nuc_worker.signals.progress.connect(self._stat_lbl.info)

        self._nuc_worker.signals.progress_percent.connect(self._prog.setValue)
        self._nuc_worker.signals.result.connect(self._on_nuc_done)
        self._nuc_worker.signals.error.connect(self._on_nuc_err)
        self._nuc_worker.signals.finished.connect(lambda: (
            self._nuc_run_btn.setEnabled(True),
            self._prog.setRange(0, 100), self._prog.setValue(100),
            self._prog.setFormat("Done"),
            QTimer.singleShot(800, lambda: self._prog.setVisible(False))))
        self._nuc_worker.start()

    def _on_nuc_done(self, res):
        masks = res.get("masks")
        if masks is None:
            self._stat_lbl.warn("No nucleus masks returned.")
            return
        self._push_nuc_undo()
        self._nuc_mask     = masks
        self._nuc_included = set(np.unique(masks)) - {0}

        self._state.active_cell_ids["nucleus"] = set(self._nuc_included)
        self._refresh_nuclei()
        self._update_nuc_stats()
        self._exp_nuc_btn.setEnabled(True)
        self._exp_pdf_btn.setEnabled(True)
        self._sel_nuc_btn.setEnabled(True)
        if self._cell_mask is not None:
            self._combine_btn.setEnabled(True)
            self._exp_overlay_btn.setEnabled(True)
        n = len(self._nuc_included)
        self._stat_lbl.ok(f"Nucleus done. {n} nuclei detected.")

        # CellRegionWorker can match each nucleus to its containing cell.
        nuc_float = masks.astype(np.float32)
        if nuc_float.ndim == 2:
            nuc_float = nuc_float[np.newaxis]   # (1, Y, X)
        self._state.set_derived_channel(
            key    = "nucleus_mask",
            mask   = nuc_float,
            source = "cell_nuclei_tab",
        )
        self._push_nuc_results()

    def _on_nuc_err(self, msg):
        QMessageBox.critical(self, "Nucleus Segmentation Error", msg)
        self._stat_lbl.err("Nucleus segmentation failed.")

    def _run_cell_auto_params(self):
        """Auto-parameter search for cell body segmentation."""
        if not _require_image(self._state, self): return
        plane = self._get_slice(self._cell_ch_combo, self._cell_mode_combo, self._cell_z_spin)
        if plane is None:
            self._stat_lbl.warn("No image slice available for auto-parameter search."); return
        test_slice = (plane * 65535).astype(np.float32)
        self._cell_auto_btn.setEnabled(False)
        self._cell_run_btn.setEnabled(False)
        self._prog.setVisible(True)
        self._prog.setRange(0, 100); self._prog.setValue(0); self._prog.setFormat("%p%")
        self._stat_lbl.info("Cell body auto parameter search running…")
        self._cell_auto_worker = _AutoParamWorker(test_slice, "cyto2", parent=self)
        self._cell_auto_worker.signals.progress.connect(self._stat_lbl.info)
        self._cell_auto_worker.signals.progress_percent.connect(self._prog.setValue)
        self._cell_auto_worker.signals.result.connect(self._on_cell_auto_done)
        self._cell_auto_worker.signals.error.connect(
            lambda msg: (self._stat_lbl.err("Cell auto search failed."),
                         self._cell_auto_btn.setEnabled(True),
                         self._cell_run_btn.setEnabled(True),
                         self._prog.setVisible(False)))
        self._cell_auto_worker.signals.finished.connect(
            lambda: (self._cell_auto_btn.setEnabled(True),
                     self._cell_run_btn.setEnabled(True),
                     self._prog.setRange(0, 0), self._prog.setFormat(""),
                     self._prog.setVisible(False)))
        self._cell_auto_worker.start()

    def _on_cell_auto_done(self, res):
        d = res.get("diameter", 100)
        f = res.get("flow",     0.4)
        c = res.get("cellprob", 0.0)
        self._cell_diameter_spin.setValue(int(d))
        self._cell_flow_spin.setValue(float(f))
        self._cell_cellprob_spin.setValue(float(c))
        self._stat_lbl.ok(
            f"Cell auto params → diameter={d}, flow={f}, prob={c} (editable)")

    def _run_nuc_auto_params(self):
        """Auto-parameter search for nucleus segmentation."""
        if not _require_image(self._state, self): return
        plane = self._get_slice(self._nuc_ch_combo, self._nuc_mode_combo, self._nuc_z_spin)
        if plane is None:
            self._stat_lbl.warn("No image slice available for auto-parameter search."); return
        test_slice = (plane * 65535).astype(np.float32)
        self._nuc_auto_btn.setEnabled(False)
        self._nuc_run_btn.setEnabled(False)
        self._prog.setVisible(True)
        self._prog.setRange(0, 100); self._prog.setValue(0); self._prog.setFormat("%p%")
        self._stat_lbl.info("Nucleus auto parameter search running…")
        self._nuc_auto_worker = _AutoParamWorker(test_slice, "nuclei", parent=self)
        self._nuc_auto_worker.signals.progress.connect(self._stat_lbl.info)
        self._nuc_auto_worker.signals.progress_percent.connect(self._prog.setValue)
        self._nuc_auto_worker.signals.result.connect(self._on_nuc_auto_done)
        self._nuc_auto_worker.signals.error.connect(
            lambda msg: (self._stat_lbl.err("Nucleus auto search failed."),
                         self._nuc_auto_btn.setEnabled(True),
                         self._nuc_run_btn.setEnabled(True),
                         self._prog.setVisible(False)))
        self._nuc_auto_worker.signals.finished.connect(
            lambda: (self._nuc_auto_btn.setEnabled(True),
                     self._nuc_run_btn.setEnabled(True),
                     self._prog.setRange(0, 0), self._prog.setFormat(""),
                     self._prog.setVisible(False)))
        self._nuc_auto_worker.start()

    def _on_nuc_auto_done(self, res):
        d = res.get("diameter", 83)
        f = res.get("flow",     0.4)
        c = res.get("cellprob", 0.0)
        self._nuc_diameter_spin.setValue(int(d))
        self._nuc_flow_spin.setValue(float(f))
        self._nuc_cellprob_spin.setValue(float(c))
        self._stat_lbl.ok(
            f"Nucleus auto params → diameter={d}, flow={f}, prob={c} (editable)")

    def _on_nuc_3d_done(self, res):
        masks_3d = res.get("masks_3d")
        if masks_3d is None or masks_3d.max() == 0:
            self._stat_lbl.warn("3D nucleus segmentation returned empty result."); return
        self._push_nuc_undo()
        self._nuc_mask_3d   = masks_3d
        self._nuc_is_3d     = True
        z = min(self._nuc_z_spin.value(), masks_3d.shape[0] - 1)
        self._nuc_mask      = masks_3d[z]
        self._nuc_included  = set(np.unique(masks_3d)) - {0}
        self._refresh_nuclei()
        self._update_nuc_stats()
        self._exp_nuc_btn.setEnabled(True)
        self._exp_pdf_btn.setEnabled(True)
        if self._cell_mask is not None:
            self._combine_btn.setEnabled(True)
            self._exp_overlay_btn.setEnabled(True)
        n = len(self._nuc_included)
        self._stat_lbl.ok(f"3D nucleus done. {n} unique labels. Scroll Z to inspect.")
        self._push_nuc_results()

    def _start_draw(self):
        self._is_drawing   = True
        self._drawing_pts  = []
        self._draw_target  = ("cell" if self._draw_target_combo.currentIndex() == 0
                              else "nuc")
        self._draw_btn.setEnabled(False)
        self._finalize_btn.setEnabled(True)
        canvas = self._cell_canvas if self._draw_target == "cell" else self._nuc_canvas
        tab_idx = 0 if self._draw_target == "cell" else 1
        self._view_tabs.setCurrentIndex(tab_idx)
        self._stat_lbl.info(f"Click on the {self._draw_target} canvas to add vertices.")

    def _finalize_poly(self):
        if len(self._drawing_pts) < 3:
            QMessageBox.warning(self, "Drawing", "Need ≥ 3 points.")
            return
        plane = (self._get_slice(self._cell_ch_combo, self._cell_mode_combo, self._cell_z_spin)
                 if self._draw_target == "cell"
                 else self._get_slice(self._nuc_ch_combo, self._nuc_mode_combo, self._nuc_z_spin))
        if plane is None:
            return
        h, w = plane.shape
        from skimage.draw import polygon as sk_polygon
        pts_r = np.array([p[1] for p in self._drawing_pts], dtype=np.float64)
        pts_c = np.array([p[0] for p in self._drawing_pts], dtype=np.float64)
        rr, cc = sk_polygon(pts_r, pts_c, shape=(h, w))
        if self._draw_target == "cell":
            nid = (int(self._cell_mask.max()) + 1 if self._cell_mask is not None else 1)
            nm  = (self._cell_mask.copy() if self._cell_mask is not None
                   else np.zeros((h, w), dtype=np.int32))
            nm[rr, cc] = nid
            self._push_cell_undo()
            self._cell_mask     = nm
            self._cell_included = set(np.unique(nm)) - {0}
            self._refresh_cells()
            self._update_cell_stats()
        else:
            nid = (int(self._nuc_mask.max()) + 1 if self._nuc_mask is not None else 1)
            nm  = (self._nuc_mask.copy() if self._nuc_mask is not None
                   else np.zeros((h, w), dtype=np.int32))
            nm[rr, cc] = nid
            self._push_nuc_undo()
            self._nuc_mask     = nm
            self._nuc_included = set(np.unique(nm)) - {0}
            self._refresh_nuclei()
            self._update_nuc_stats()
        self._drawing_pts  = []
        self._is_drawing   = False
        self._draw_btn.setEnabled(True)
        self._finalize_btn.setEnabled(False)
        self._stat_lbl.ok("Manual mask added.")

    def _on_cell_click(self, event):
        if event.inaxes != self._cell_canvas.ax or event.xdata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if self._is_drawing and self._draw_target == "cell":
            self._drawing_pts.append((event.xdata, event.ydata))
            self._refresh_cells()
            return
        if self._cell_mask is not None:
            h, w = self._cell_mask.shape[:2]
            if 0 <= y < h and 0 <= x < w:
                cid = int(self._cell_mask[y, x])
                if cid != 0:
                    if cid in self._cell_included:
                        self._cell_included.discard(cid)
                    else:
                        self._cell_included.add(cid)

                    self._state.active_cell_ids["cell_body"] = set(self._cell_included)
                    self._refresh_cells()
                    self._update_cell_stats()

    def _on_nuc_click(self, event):
        if event.inaxes != self._nuc_canvas.ax or event.xdata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if self._is_drawing and self._draw_target == "nuc":
            self._drawing_pts.append((event.xdata, event.ydata))
            self._refresh_nuclei()
            return
        if self._nuc_mask is not None:
            h, w = self._nuc_mask.shape[:2]
            if 0 <= y < h and 0 <= x < w:
                nid = int(self._nuc_mask[y, x])
                if nid != 0:
                    if nid in self._nuc_included:
                        self._nuc_included.discard(nid)
                    else:
                        self._nuc_included.add(nid)

                    self._state.active_cell_ids["nucleus"] = set(self._nuc_included)
                    self._refresh_nuclei()
                    self._update_nuc_stats()

    def _push_cell_undo(self):
        snap = {"mask": self._cell_mask.copy() if self._cell_mask is not None else None,
                "inc":  set(self._cell_included)}
        self._cell_undo.append(snap); self._cell_redo.clear()
        if len(self._cell_undo) > 30: self._cell_undo.pop(0)
        self._undo_btn.setEnabled(True); self._redo_btn.setEnabled(False)

    def _push_nuc_undo(self):
        snap = {"mask": self._nuc_mask.copy() if self._nuc_mask is not None else None,
                "inc":  set(self._nuc_included)}
        self._nuc_undo.append(snap); self._nuc_redo.clear()
        if len(self._nuc_undo) > 30: self._nuc_undo.pop(0)
        self._undo_btn.setEnabled(True); self._redo_btn.setEnabled(False)

    def _undo(self):
        tab = self._view_tabs.currentIndex()
        if tab == 0 and self._cell_undo:
            self._cell_redo.append({
                "mask": self._cell_mask.copy() if self._cell_mask is not None else None,
                "inc":  set(self._cell_included)})
            s = self._cell_undo.pop()
            self._cell_mask = s["mask"]; self._cell_included = s["inc"]
            self._refresh_cells(); self._update_cell_stats()
        elif tab == 1 and self._nuc_undo:
            self._nuc_redo.append({
                "mask": self._nuc_mask.copy() if self._nuc_mask is not None else None,
                "inc":  set(self._nuc_included)})
            s = self._nuc_undo.pop()
            self._nuc_mask = s["mask"]; self._nuc_included = s["inc"]
            self._refresh_nuclei(); self._update_nuc_stats()
        self._undo_btn.setEnabled(bool(self._cell_undo) or bool(self._nuc_undo))
        self._redo_btn.setEnabled(True)

    def _redo(self):
        tab = self._view_tabs.currentIndex()
        if tab == 0 and self._cell_redo:
            self._cell_undo.append({
                "mask": self._cell_mask.copy() if self._cell_mask is not None else None,
                "inc":  set(self._cell_included)})
            s = self._cell_redo.pop()
            self._cell_mask = s["mask"]; self._cell_included = s["inc"]
            self._refresh_cells(); self._update_cell_stats()
        elif tab == 1 and self._nuc_redo:
            self._nuc_undo.append({
                "mask": self._nuc_mask.copy() if self._nuc_mask is not None else None,
                "inc":  set(self._nuc_included)})
            s = self._nuc_redo.pop()
            self._nuc_mask = s["mask"]; self._nuc_included = s["inc"]
            self._refresh_nuclei(); self._update_nuc_stats()
        self._redo_btn.setEnabled(bool(self._cell_redo) or bool(self._nuc_redo))

    def _reset_state(self):
        """Reset Cell Body & Nuclei tool to a fresh state without reloading the image."""
        # Clear cell segmentation results
        self._cell_mask = None
        self._cell_included = set()
        self._cell_undo.clear()
        self._cell_redo.clear()

        # Clear nuclei segmentation results
        self._nuc_mask = None
        self._nuc_included = set()
        self._nuc_undo.clear()
        self._nuc_redo.clear()

        self._cell_mask_3d = None
        self._nuc_mask_3d  = None
        self._cell_is_3d   = False
        self._nuc_is_3d    = False

        # Clear drawing state
        self._drawing_pts = []
        self._is_drawing = False

        # Reset history button states
        self._undo_btn.setEnabled(False)
        self._redo_btn.setEnabled(False)

        # Clear export button
        self._exp_pdf_btn.setEnabled(False)

        # Clear canvases
        self._cell_canvas.clear_canvas()
        self._nuc_canvas.clear_canvas()

        # Clear statistics displays
        if hasattr(self, "_cell_stats_lbl"):
            self._cell_stats_lbl.setText("")
        if hasattr(self, "_nuc_stats_lbl"):
            self._nuc_stats_lbl.setText("")

        # Remove stored results from AppState (this tab only)
        self._state.results_dict.pop("Cell Body Detection", None)
        self._state.results_dict.pop("Nucleus Detection", None)

        # Notify other components
        self._state.notify("results")

        self._stat_lbl.ok("Reset complete.")

    @staticmethod
    def _draw_mask_overlay(ax, mask, included, show_mask, show_bnd, show_num,
                            bc_map, bc_name, plane):
        if mask is None or plane is None:
            return
        h, w = mask.shape[:2]
        all_ids  = set(np.unique(mask)) - {0}
        ids_draw = included

        if show_mask and ids_draw:
            from skimage.color import label2rgb
            colors = [(0.1, 0.7, 0.1) if lid in included else (0.7, 0.1, 0.1)
                      for lid in sorted(all_ids)]
            ov = label2rgb(mask, bg_label=0, bg_color=None, colors=colors)
            ax.imshow(ov, alpha=0.35, interpolation="nearest", origin="upper")

        if show_bnd:
            dm = np.zeros_like(mask)
            for lid in ids_draw: dm[mask == lid] = lid
            bnd = _cb_calculate_exact_boundaries(dm)
            if bnd is not None and bnd.any():
                bc  = bc_map.get(bc_name, (0, 255, 0))
                bcn = tuple(v / 255.0 for v in bc)
                rgba = np.zeros((h, w, 4), dtype=np.float32)
                rgba[bnd, :3] = bcn; rgba[bnd, 3] = 1.0
                ax.imshow(rgba, interpolation="nearest", origin="upper")

        if show_num:
            for prop in regionprops(mask):
                if prop.label not in ids_draw: continue
                cy, cx = prop.centroid
                color = "#80ff80" if prop.label in included else "#ff8080"
                ax.text(cx, cy, str(prop.label), fontsize=6, color=color,
                        ha="center", va="center", fontweight="bold")

    def _refresh_cells(self, _=None):
        if self._cell_is_3d and self._cell_mask_3d is not None:
            z = min(self._cell_z_spin.value(), self._cell_mask_3d.shape[0] - 1)
            self._cell_mask = self._cell_mask_3d[z]
            ch_idx = _channel_index_from_combo(self._cell_ch_combo, self._state)
            if ch_idx is None: ch_idx = 0
            plane = self._state.get_display_slice(channel=ch_idx, z=z, projection="slice")
        else:

            plane = self._get_slice(
                self._cell_ch_combo, self._cell_mode_combo, self._cell_z_spin)
        ax = self._cell_canvas.ax
        ax.clear(); ax.set_facecolor("#0a0a14")
        if plane is None:
            self._cell_canvas.draw_idle(); return

        if self._cell_show_orig.isChecked():
            ax.imshow(plane, cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest", origin="upper")

        self._draw_mask_overlay(
            ax, self._cell_mask, self._cell_included,
            self._cell_show_mask.isChecked(), self._cell_show_bnd.isChecked(),
            self._cell_show_num.isChecked(),
            self.CELL_BOUNDARY_COLORS, self._cell_bc_combo.currentText(), plane)

        if self._is_drawing and self._draw_target == "cell" and self._drawing_pts:
            xs = [p[0] for p in self._drawing_pts]
            ys = [p[1] for p in self._drawing_pts]
            if len(xs) > 1:
                ax.plot(xs + [xs[0]], ys + [ys[0]], "y-", lw=1.5, zorder=4)
            ax.scatter(xs, ys, c="yellow", s=14, zorder=5)
            if xs: ax.scatter([xs[0]], [ys[0]], c="cyan", s=18, zorder=6)

        n_sel = len(self._cell_included)
        n_tot = (len(set(np.unique(self._cell_mask)) - {0})
                 if self._cell_mask is not None else 0)
        ax.set_title(f"Cell Bodies  |  {n_sel}/{n_tot} selected",
                     color="#a0a0d0", fontsize=9, pad=4)
        ax.axis("off"); self._cell_canvas.draw_idle()

    def _refresh_nuclei(self, _=None):
        if self._nuc_is_3d and self._nuc_mask_3d is not None:
            z = min(self._nuc_z_spin.value(), self._nuc_mask_3d.shape[0] - 1)
            self._nuc_mask = self._nuc_mask_3d[z]
            ch_idx = _channel_index_from_combo(self._nuc_ch_combo, self._state)
            if ch_idx is None: ch_idx = 0
            plane = self._state.get_display_slice(channel=ch_idx, z=z, projection="slice")
        else:

            plane = self._get_slice(
                self._nuc_ch_combo, self._nuc_mode_combo, self._nuc_z_spin)
        ax = self._nuc_canvas.ax
        ax.clear(); ax.set_facecolor("#0a0a14")
        if plane is None:
            self._nuc_canvas.draw_idle(); return

        if self._nuc_show_orig.isChecked():
            ax.imshow(plane, cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest", origin="upper")

        self._draw_mask_overlay(
            ax, self._nuc_mask, self._nuc_included,
            self._nuc_show_mask.isChecked(), self._nuc_show_bnd.isChecked(),
            self._nuc_show_num.isChecked(),
            self.NUC_BOUNDARY_COLORS, self._nuc_bc_combo.currentText(), plane)

        if self._is_drawing and self._draw_target == "nuc" and self._drawing_pts:
            xs = [p[0] for p in self._drawing_pts]
            ys = [p[1] for p in self._drawing_pts]
            if len(xs) > 1:
                ax.plot(xs + [xs[0]], ys + [ys[0]], "y-", lw=1.5, zorder=4)
            ax.scatter(xs, ys, c="yellow", s=14, zorder=5)
            if xs: ax.scatter([xs[0]], [ys[0]], c="cyan", s=18, zorder=6)

        n_sel = len(self._nuc_included)
        n_tot = (len(set(np.unique(self._nuc_mask)) - {0})
                 if self._nuc_mask is not None else 0)
        ax.set_title(f"Nuclei  |  {n_sel}/{n_tot} selected",
                     color="#a0a0d0", fontsize=9, pad=4)
        ax.axis("off"); self._nuc_canvas.draw_idle()

    def _refresh_combined(self, _=None):

        active_z = self._state.active_z_slice
        if active_z is not None and self._cell_mask_3d is not None and self._cell_is_3d:
            z = min(int(active_z), self._cell_mask_3d.shape[0] - 1)
            self._cell_mask = self._cell_mask_3d[z]
        if active_z is not None and self._nuc_mask_3d is not None and self._nuc_is_3d:
            z = min(int(active_z), self._nuc_mask_3d.shape[0] - 1)
            self._nuc_mask = self._nuc_mask_3d[z]
        plane = self._get_slice(self._cell_ch_combo, self._cell_mode_combo, self._cell_z_spin)
        ax = self._comb_canvas.ax
        ax.clear(); ax.set_facecolor("#0a0a14")
        if plane is None:
            self._comb_canvas.draw_idle(); return

        # Background image
        if self._comb_show_cell_orig.isChecked():
            ax.imshow(plane, cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest", origin="upper")

        # ── Cell body overlay ─────────────────────────────────────────────
        if self._cell_mask is not None:
            h, w = self._cell_mask.shape
            # Filled masks
            if self._comb_show_cell_mask.isChecked():
                from matplotlib.colors import hsv_to_rgb
                rgba_fill = np.zeros((h, w, 4), dtype=np.float32)
                ids = sorted(self._cell_included)
                for i, lid in enumerate(ids):
                    hue = (i * 0.618033) % 1.0
                    rgb = hsv_to_rgb([hue, 0.6, 0.8])
                    rgba_fill[self._cell_mask == lid, :3] = rgb
                    rgba_fill[self._cell_mask == lid, 3]  = 0.35
                ax.imshow(rgba_fill, interpolation="nearest", origin="upper")
            # Outlines
            if self._comb_show_cell_bnd.isChecked():
                dm_c = np.zeros_like(self._cell_mask)
                for lid in self._cell_included: dm_c[self._cell_mask == lid] = lid
                bnd_c = _cb_calculate_exact_boundaries(dm_c)
                if bnd_c is not None and bnd_c.any():
                    bc  = self.CELL_BOUNDARY_COLORS.get(self._comb_cell_bc_combo.currentText(), (0, 255, 0))
                    bcn = tuple(v / 255.0 for v in bc)
                    rgba_c = np.zeros((h, w, 4), dtype=np.float32)
                    rgba_c[bnd_c, :3] = bcn; rgba_c[bnd_c, 3] = 1.0
                    ax.imshow(rgba_c, interpolation="nearest", origin="upper")
            # Cell IDs
            if self._comb_show_cell_ids.isChecked():
                for prop in regionprops(self._cell_mask):
                    if prop.label in self._cell_included:
                        cy, cx = prop.centroid
                        ax.text(cx, cy, str(prop.label), color="white",
                                fontsize=6, ha="center", va="center",
                                fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.1",
                                          facecolor="black", alpha=0.4, edgecolor="none"))

        # ── Nucleus overlay ───────────────────────────────────────────────
        if self._nuc_mask is not None:
            h, w = self._nuc_mask.shape
            # Filled masks
            if self._comb_show_nuc_mask.isChecked():
                from matplotlib.colors import hsv_to_rgb as _h2r
                rgba_nf = np.zeros((h, w, 4), dtype=np.float32)
                for i, lid in enumerate(sorted(self._nuc_included)):
                    hue = (i * 0.618033 + 0.5) % 1.0
                    rgb = _h2r([hue, 0.5, 0.75])
                    rgba_nf[self._nuc_mask == lid, :3] = rgb
                    rgba_nf[self._nuc_mask == lid, 3]  = 0.30
                ax.imshow(rgba_nf, interpolation="nearest", origin="upper")
            # Outlines
            if self._comb_show_nuc_bnd.isChecked():
                dm_n = np.zeros_like(self._nuc_mask)
                for lid in self._nuc_included: dm_n[self._nuc_mask == lid] = lid
                bnd_n = _cb_calculate_exact_boundaries(dm_n)
                if bnd_n is not None and bnd_n.any():
                    bc  = self.NUC_BOUNDARY_COLORS.get(self._comb_nuc_bc_combo.currentText(), (0, 255, 255))
                    bcn = tuple(v / 255.0 for v in bc)
                    rgba_n = np.zeros((h, w, 4), dtype=np.float32)
                    rgba_n[bnd_n, :3] = bcn; rgba_n[bnd_n, 3] = 0.9
                    ax.imshow(rgba_n, interpolation="nearest", origin="upper")
            # Nucleus IDs
            if self._comb_show_nuc_ids.isChecked():
                for prop in regionprops(self._nuc_mask):
                    if prop.label in self._nuc_included:
                        cy, cx = prop.centroid
                        ax.text(cx, cy, str(prop.label), color="#80c0ff",
                                fontsize=6, ha="center", va="center",
                                fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.1",
                                          facecolor="black", alpha=0.4, edgecolor="none"))

        n_c = len(self._cell_included) if self._cell_mask is not None else 0
        n_n = len(self._nuc_included)  if self._nuc_mask  is not None else 0
        ax.set_title(f"Combined  |  {n_c} cells  +  {n_n} nuclei",
                     color="#a0a0d0", fontsize=9, pad=4)
        ax.axis("off"); self._comb_canvas.draw_idle()
        self._view_tabs.setCurrentIndex(2)

    def _update_cell_stats(self):
        if self._cell_mask is None:
            self._cell_stats.setText("")
            return
        n_tot = len(set(np.unique(self._cell_mask)) - {0})
        n_sel = len(self._cell_included)
        areas = [p.area for p in regionprops(self._cell_mask)
                 if p.label in self._cell_included]
        lines = [f"Total  : {n_tot}", f"Selected: {n_sel}",
                 f"Deselected: {n_tot - n_sel}"]
        if areas:
            lines.append(f"Mean area: {np.mean(areas):.0f} px²")
        self._cell_stats.setText("\n".join(lines))

    def _update_nuc_stats(self):
        if self._nuc_mask is None:
            self._nuc_stats.setText("")
            return
        n_tot = len(set(np.unique(self._nuc_mask)) - {0})
        n_sel = len(self._nuc_included)
        areas = [p.area for p in regionprops(self._nuc_mask)
                 if p.label in self._nuc_included]
        lines = [f"Total  : {n_tot}", f"Selected: {n_sel}",
                 f"Deselected: {n_tot - n_sel}"]
        if areas:
            lines.append(f"Mean area: {np.mean(areas):.0f} px²")
        self._nuc_stats.setText("\n".join(lines))

    def _push_cell_results(self):
        if self._cell_mask is None: return
        rows = []
        for prop in regionprops(self._cell_mask):
            rows.append({"cell_id": prop.label, "area_px": prop.area,
                         "centroid_y": round(prop.centroid[0], 1),
                         "centroid_x": round(prop.centroid[1], 1),
                         "selected": prop.label in self._cell_included})
        df = pd.DataFrame(rows)
        summary = (f"Total cells: {len(rows)}\n"
                   f"Selected: {sum(1 for r in rows if r['selected'])}\n"
                   f"Mean area: {df['area_px'].mean():.0f} px²" if rows else "")
        self._state.set_result("Cell Body Detection", {
            "dataframe": df, "summary": summary, "mask": self._cell_mask})


    def _export_combined_overlay(self):
        """Export the current Combined view as an RGB TIFF overlay."""
        plane = self._get_slice(self._cell_ch_combo, self._cell_mode_combo, self._cell_z_spin)
        if plane is None:
            QMessageBox.warning(self, "No image", "No image available to export."); return
        h, w = plane.shape

        # Build RGB base from grayscale
        base = (np.clip(plane, 0, 1) * 255).astype(np.uint8)
        overlay = np.stack([base, base, base], axis=-1)   # (H, W, 3) uint8

        # Draw cell body outlines in chosen colour (default green)
        if self._cell_mask is not None:
            dm_c = np.zeros_like(self._cell_mask)
            for lid in self._cell_included: dm_c[self._cell_mask == lid] = lid
            bnd_c = _cb_calculate_exact_boundaries(dm_c)
            if bnd_c is not None and bnd_c.any():
                bc = self.CELL_BOUNDARY_COLORS.get(self._cell_bc_combo.currentText(), (0, 255, 0))
                overlay[bnd_c] = bc

        # Draw nucleus outlines in chosen colour (default cyan)
        if self._nuc_mask is not None:
            dm_n = np.zeros_like(self._nuc_mask)
            for lid in self._nuc_included: dm_n[self._nuc_mask == lid] = lid
            bnd_n = _cb_calculate_exact_boundaries(dm_n)
            if bnd_n is not None and bnd_n.any():
                bc = self.NUC_BOUNDARY_COLORS.get(self._nuc_bc_combo.currentText(), (0, 255, 255))
                overlay[bnd_n] = bc

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Combined Overlay", "", "TIFF (*.tif)")
        if path:
            try:
                tifffile.imwrite(path, overlay.astype(np.uint8))
                QMessageBox.information(self, "Exported", f"Saved:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def _push_nuc_results(self):
        if self._nuc_mask is None: return
        rows = []
        for prop in regionprops(self._nuc_mask):
            rows.append({"nucleus_id": prop.label, "area_px": prop.area,
                         "centroid_y": round(prop.centroid[0], 1),
                         "centroid_x": round(prop.centroid[1], 1),
                         "selected": prop.label in self._nuc_included})
        df = pd.DataFrame(rows)
        summary = (f"Total nuclei: {len(rows)}\n"
                   f"Selected: {sum(1 for r in rows if r['selected'])}\n"
                   f"Mean area: {df['area_px'].mean():.0f} px²" if rows else "")
        self._state.set_result("Nucleus Detection", {
            "dataframe": df, "summary": summary, "mask": self._nuc_mask})


    def _open_cell_selection_dialog(self):
        """Let the user pick which segmented Z-layer is the active cell mask."""
        mask_3d = self._cell_mask_3d if self._cell_is_3d and self._cell_mask_3d is not None else (
            self._cell_mask[np.newaxis] if self._cell_mask is not None else None)
        if mask_3d is None:
            QMessageBox.information(self, "No Data", "Run cell body segmentation first.")
            return
        nz = mask_3d.shape[0]
        current_z = self._state.active_z_slice if self._state.active_z_slice is not None else 0
        self._show_z_layer_dialog(
            title="Select Active Cell Body Layer",
            nz=nz, current_z=current_z,
            mask_3d=mask_3d,
            on_apply=self._apply_cell_z_layer,
            hint="Select which Z-layer to use as the active cell body mask.\n"
                 "This layer feeds into Region Analysis, statistics, exports and overlays.")

    def _open_nuc_selection_dialog(self):
        """Let the user pick which segmented Z-layer is the active nucleus mask."""
        mask_3d = self._nuc_mask_3d if self._nuc_is_3d and self._nuc_mask_3d is not None else (
            self._nuc_mask[np.newaxis] if self._nuc_mask is not None else None)
        if mask_3d is None:
            QMessageBox.information(self, "No Data", "Run nucleus segmentation first.")
            return
        nz = mask_3d.shape[0]
        current_z = self._state.active_z_slice if self._state.active_z_slice is not None else 0
        self._show_z_layer_dialog(
            title="Select Active Nucleus Layer",
            nz=nz, current_z=current_z,
            mask_3d=mask_3d,
            on_apply=self._apply_nuc_z_layer,
            hint="Select which Z-layer to use as the active nucleus mask.\n"
                 "This layer feeds into Region Analysis, statistics, exports and overlays.")

    def _show_z_layer_dialog(self, title, nz, current_z, mask_3d, on_apply, hint=""):
        """Generic Z-layer picker dialog showing per-layer cell counts.
        Includes a 'Use projection mode' option to clear any active layer.
        """
        from skimage.measure import regionprops as _rp
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setMinimumWidth(390)
        dlg.setStyleSheet(
            "QDialog{background:#1c1c32;color:#dcdcf0;}"
            "QLabel{color:#dcdcf0;}"
            "QPushButton{background:#2a2a50;color:#c0c0ff;border:1px solid #3a3a70;"
            "border-radius:4px;padding:4px 12px;}"
            "QPushButton:hover{background:#3a3a70;}"
            "QRadioButton{color:#c0c0e0;spacing:6px;}"
            "QScrollArea{background:#141428;border:none;}")
        vl = QVBoxLayout(dlg)
        if hint:
            hl = QLabel(hint)
            hl.setStyleSheet("color:#8080b0;font-size:11px;")
            hl.setWordWrap(True)
            vl.addWidget(hl)


        vl.addWidget(QLabel("Active segmentation layer:"))
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); inner.setStyleSheet("background:#141428;")
        inner_l = QVBoxLayout(inner); inner_l.setSpacing(4)

        bg = QButtonGroup(dlg)
        PROJ_ID = -99   # sentinel: means "clear active layer / use projection"

        rb_proj = QRadioButton("Use projection mode  (default — no active layer)")
        rb_proj.setToolTip(
            "Clear any active Z-layer selection.\n\n"
            "Overlays, Region Analysis, exports and statistics will use\n"
            "the current projection mode (Max / Mean / Slice) instead of\n"
            "a pinned Z-layer.\n\n"
            "Select this to return to normal projection-based workflow.")
        rb_proj.setStyleSheet("color:#a0e0a0;font-style:italic;")
        bg.addButton(rb_proj, PROJ_ID)
        inner_l.addWidget(rb_proj)

        # Separator
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#2a2a50;")
        inner_l.addWidget(sep)
        inner_l.addWidget(QLabel("  — or pin a specific Z-layer: —"))

        radios = {}
        for z in range(nz):
            plane_mask = mask_3d[z]
            n_cells = len(set(np.unique(plane_mask)) - {0})
            rb = QRadioButton(f"Z{z}  —  {n_cells} object{'s' if n_cells != 1 else ''} detected")
            rb.setToolTip(f"Pin Z-slice {z} as the active segmentation layer.\n"
                          f"Contains {n_cells} segmented object(s).\n\n"
                          f"All downstream tools (Region Analysis, statistics, exports,\n"
                          f"overlays) will use only this Z plane.")
            bg.addButton(rb, z)
            if current_z is not None and z == current_z:
                rb.setChecked(True)
            inner_l.addWidget(rb)
            radios[z] = rb

        # If no layer is currently pinned, default to projection mode
        if current_z is None:
            rb_proj.setChecked(True)

        inner_l.addStretch()
        scroll.setWidget(inner)
        vl.addWidget(scroll)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        vl.addWidget(bb)

        if dlg.exec_() == QDialog.Accepted:
            chosen = bg.checkedId()
            if chosen == PROJ_ID:

                on_apply(None, mask_3d)
            elif chosen >= 0:
                on_apply(chosen, mask_3d)

    def _apply_cell_z_layer(self, z, mask_3d: np.ndarray):
        """Set Z-layer z as the active cell body mask, or clear if z is None."""

        if z is None:
            self._state.set_active_z_slice(None)
            # Revert cell mask to max-projection of 3D stack if available
            if self._cell_is_3d and self._cell_mask_3d is not None:
                proj = self._cell_mask_3d.max(axis=0)
                self._cell_mask = proj
                self._cell_included = set(np.unique(proj)) - {0}

            cell_float = self._cell_mask.astype(np.float32)[np.newaxis] if self._cell_mask is not None else None
            if cell_float is not None:
                self._state.set_derived_channel("cell_body_mask", cell_float, source="cell_nuclei_tab")
            self._refresh_cells()
            self._update_cell_stats()
            if self._nuc_mask is not None:
                self._refresh_combined()
            self._stat_lbl.ok("Active cell layer cleared — using projection mode.")
            return

        self._state.set_active_z_slice(z)
        # Also update the 2D working mask for this tab
        self._cell_mask = mask_3d[z]
        self._cell_included = set(np.unique(self._cell_mask)) - {0}
        self._state.active_cell_ids["cell_body"] = set(self._cell_included)
        # Republish cell_body_mask derived channel with the selected layer

        cell_float = self._cell_mask.astype(np.float32)[np.newaxis]
        self._state.set_derived_channel("cell_body_mask", cell_float, source="cell_nuclei_tab")
        self._refresh_cells()
        self._update_cell_stats()
        if self._nuc_mask is not None:
            self._refresh_combined()
        self._stat_lbl.ok(f"Active cell layer set to Z{z}.")

    def _apply_nuc_z_layer(self, z, mask_3d: np.ndarray):
        """Set Z-layer z as the active nucleus mask, or clear if z is None."""

        if z is None:
            self._state.set_active_z_slice(None)
            if self._nuc_is_3d and self._nuc_mask_3d is not None:
                proj = self._nuc_mask_3d.max(axis=0)
                self._nuc_mask = proj
                self._nuc_included = set(np.unique(proj)) - {0}

            nuc_float = self._nuc_mask.astype(np.float32)[np.newaxis] if self._nuc_mask is not None else None
            if nuc_float is not None:
                self._state.set_derived_channel("nucleus_mask", nuc_float, source="cell_nuclei_tab")
            self._refresh_nuclei()
            self._update_nuc_stats()
            if self._cell_mask is not None:
                self._refresh_combined()
            self._stat_lbl.ok("Active nucleus layer cleared — using projection mode.")
            return
        self._state.set_active_z_slice(z)
        self._nuc_mask = mask_3d[z]
        self._nuc_included = set(np.unique(self._nuc_mask)) - {0}
        self._state.active_cell_ids["nucleus"] = set(self._nuc_included)

        nuc_float = self._nuc_mask.astype(np.float32)[np.newaxis]
        self._state.set_derived_channel("nucleus_mask", nuc_float, source="cell_nuclei_tab")
        self._refresh_nuclei()
        self._update_nuc_stats()
        if self._cell_mask is not None:
            self._refresh_combined()
        self._stat_lbl.ok(f"Active nucleus layer set to Z{z}.")

    def _export_cell_mask(self):

        mask_3d = self._cell_mask_3d if self._cell_is_3d and self._cell_mask_3d is not None else (
            self._cell_mask[np.newaxis] if self._cell_mask is not None else None)
        if mask_3d is None:
            return
        dlg = ExportMaskDialog(
            self._state, mask_3d, self._cell_included,
            title="Export Cell Masks", parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return
        out = dlg.get_result()
        if out is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Cell Masks", "", "TIFF (*.tif)")
        if path:
            tifffile.imwrite(path, out.astype(np.uint16))
            QMessageBox.information(self, "Exported", f"Saved:\n{path}")

    def _export_nuc_mask(self):

        mask_3d = self._nuc_mask_3d if self._nuc_is_3d and self._nuc_mask_3d is not None else (
            self._nuc_mask[np.newaxis] if self._nuc_mask is not None else None)
        if mask_3d is None:
            return
        dlg = ExportMaskDialog(
            self._state, mask_3d, self._nuc_included,
            title="Export Nucleus Masks", parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return
        out = dlg.get_result()
        if out is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Nucleus Masks", "", "TIFF (*.tif)")
        if path:
            tifffile.imwrite(path, out.astype(np.uint16))
            QMessageBox.information(self, "Exported", f"Saved:\n{path}")

    def _export_pdf(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export PDF Report", "", "PDF (*.pdf)")
        if not path: return
        plane_c = self._get_slice(self._cell_ch_combo, self._cell_mode_combo, self._cell_z_spin)
        plane_n = self._get_slice(self._nuc_ch_combo, self._nuc_mode_combo, self._nuc_z_spin)
        try:
            with PdfPages(path) as pdf:
                fig, axes = plt.subplots(1, 2, figsize=(14, 7))
                if plane_c is not None:
                    axes[0].imshow(plane_c, cmap="gray")
                    if self._cell_mask is not None:
                        bnd = _cb_calculate_exact_boundaries(self._cell_mask)
                        if bnd is not None:
                            bc = tuple(v / 255.0 for v in self.CELL_BOUNDARY_COLORS.get(
                                self._cell_bc_combo.currentText(), (0, 255, 0)))
                            rgba = np.zeros((*bnd.shape, 4), dtype=np.float32)
                            rgba[bnd, :3] = bc; rgba[bnd, 3] = 1.0
                            axes[0].imshow(rgba)
                axes[0].set_title("Cell Bodies"); axes[0].axis("off")
                if plane_n is not None:
                    axes[1].imshow(plane_n, cmap="gray")
                    if self._nuc_mask is not None:
                        bnd = _cb_calculate_exact_boundaries(self._nuc_mask)
                        if bnd is not None:
                            bc = tuple(v / 255.0 for v in self.NUC_BOUNDARY_COLORS.get(
                                self._nuc_bc_combo.currentText(), (0, 255, 255)))
                            rgba = np.zeros((*bnd.shape, 4), dtype=np.float32)
                            rgba[bnd, :3] = bc; rgba[bnd, 3] = 1.0
                            axes[1].imshow(rgba)
                axes[1].set_title("Nuclei"); axes[1].axis("off")
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
            QMessageBox.information(self, "Exported", f"Saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))


# DL AGGREGATE DETECTION  
import csv as _csv, time as _time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field as dc_field
from pathlib import Path as _Path

try:
    import torch as _torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import segmentation_models_pytorch as _smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False


# ── SegmentationResult ────────────────────────────────────────────────────────
@dataclass
class SegmentationResult:
    method_name: str = ""
    params: dict = dc_field(default_factory=dict)
    label_image: Optional[np.ndarray] = None
    n_objects: int = 0
    properties: List[dict] = dc_field(default_factory=list)
    time_seconds: float = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────
def _filter_by_morphology(label_img, min_area, max_area):
    out = label_img.copy()
    for region in regionprops(label_img):
        area = region.area
        if not (min_area <= area <= max_area):
            out[label_img == region.label] = 0
            continue
        if region.eccentricity > 0.90:
            out[label_img == region.label] = 0
            continue
        if region.eccentricity > 0.82 and area > 80:
            out[label_img == region.label] = 0
            continue
        if region.solidity < 0.65:
            out[label_img == region.label] = 0
            continue
        if area > 500:
            minor = region.minor_axis_length
            major = region.major_axis_length
            if major > 0 and minor / max(major, 1) < 0.25:
                out[label_img == region.label] = 0
                continue
    return sk_label(out > 0)


def _label_to_props(label_img, img):
    props = []
    for region in regionprops(label_img, intensity_image=img.astype(np.float32)):
        props.append({
            "id":             region.label,
            "y":              region.centroid[0],
            "x":              region.centroid[1],
            "area_px2":       region.area,
            "mean_intensity": float(region.mean_intensity),
            "max_intensity":  float(region.max_intensity),
            "eccentricity":   float(region.eccentricity),
            "perimeter":      float(region.perimeter),
            "solidity":       float(region.solidity),
            "radius_px":      float(np.sqrt(region.area / np.pi)),
        })
    return props


def _dl_draw_contours(ax, result, circle_color="#00ffcc", text_color="#ffffff",
                      circle_lw=1.2, font_size=6.5, show_numbers=True):
    from skimage import measure
    if result.label_image is None or result.label_image.max() == 0:
        return
    for region in regionprops(result.label_image)[:500]:
        mask = (result.label_image == region.label)
        for contour in measure.find_contours(mask, 0.5):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=circle_lw,
                    color=circle_color, zorder=4)
        if show_numbers:
            cy, cx = region.centroid
            ax.text(cx, cy, str(region.label), color=text_color,
                    fontsize=font_size, fontweight="bold",
                    ha="center", va="center", zorder=5, clip_on=True)


def _load_gt_mask_from_file(path: str) -> np.ndarray:
    img = tifffile.imread(path)
    if img.ndim == 3:
        img = img.max(axis=0)
    if img.ndim == 4:
        img = img.max(axis=(0, 1))
    return (img > 0).astype(np.uint8)


def _compute_validation_metrics(pred_mask, gt_mask):
    p = pred_mask.astype(bool).ravel()
    g = gt_mask.astype(bool).ravel()
    tp = int(np.logical_and(p,  g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1, iou=iou)


def _result_to_binary_mask(result):
    if result is None:
        return None
    if result.label_image is not None:
        return (result.label_image > 0).astype(np.uint8)
    return None


# ── EnsemblePreprocessor ──────────────────────────────────────────────────────
# Builds 3-channel input for the DL aggregate model (norm + tophat + DoG)
class EnsemblePreprocessor:
    @staticmethod
    def pct_norm(img: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(img, [1.0, 99.9])
        if hi <= lo:
            return np.zeros_like(img, dtype=np.float32)
        return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)

    @staticmethod
    def build(img: np.ndarray) -> np.ndarray:
        from skimage.morphology import white_tophat, disk
        from skimage import filters
        raw = img[:, :, 0].astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
        norm   = EnsemblePreprocessor.pct_norm(raw)
        tophat = EnsemblePreprocessor.pct_norm(
            white_tophat(norm, disk(5)).astype(np.float32)
        )
        dog = EnsemblePreprocessor.pct_norm(
            np.clip(
                filters.gaussian(norm, 0.8) - filters.gaussian(norm, 2.5),
                0, None
            )
        )
        stacked = np.stack([norm, tophat, dog], axis=0)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        return (stacked - mean) / std


# ── EnsembleEngine ────────────────────────────────────────────────────────────
# Runs 10-fold U-Net ensemble; tiles the image and averages probability maps
class EnsembleEngine:
    @staticmethod
    def _predict_tta(model, processed, device, tile=256, overlap=32):
        import torch
        model.eval()
        _, H, W = processed.shape
        stride = max(1, tile - overlap)
        pred_sum  = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)
        hann   = np.hanning(tile + 2)[1:-1]
        window = np.outer(hann, hann).astype(np.float32)
        ys = list(range(0, max(1, H - tile + 1), stride))
        xs = list(range(0, max(1, W - tile + 1), stride))
        if not ys or ys[-1] + tile < H:
            ys.append(max(0, H - tile))
        if not xs or xs[-1] + tile < W:
            xs.append(max(0, W - tile))
        tta_ops = [(False, 0), (False, 1), (False, 2), (False, 3),
                   (True,  0), (True,  1), (True,  2), (True,  3)]
        with torch.no_grad():
            for y0 in ys:
                for x0 in xs:
                    y1 = min(y0 + tile, H)
                    x1 = min(x0 + tile, W)
                    patch = processed[:, y0:y1, x0:x1]
                    ph, pw = y1 - y0, x1 - x0
                    if ph < tile or pw < tile:
                        pad = np.zeros((3, tile, tile), dtype=np.float32)
                        pad[:, :ph, :pw] = patch
                        patch = pad
                    tta_preds = []
                    for flip, k in tta_ops:
                        aug = patch.copy()
                        if flip:
                            aug = np.flip(aug, axis=-1)
                        if k:
                            aug = np.rot90(aug, k, axes=(-2, -1))
                        t_ = torch.from_numpy(aug.copy()[np.newaxis]).float().to(device)
                        p_ = torch.sigmoid(model(t_))[0, 0].cpu().numpy()
                        if k:
                            p_ = np.rot90(p_, -k)
                        if flip:
                            p_ = np.flip(p_, axis=-1)
                        tta_preds.append(p_)
                    mp = np.mean(tta_preds, axis=0)
                    wp = window[:y1 - y0, :x1 - x0]
                    pred_sum[y0:y1, x0:x1]  += mp[:y1 - y0, :x1 - x0] * wp
                    count_map[y0:y1, x0:x1] += wp
        return np.where(count_map > 0, pred_sum / (count_map + 1e-8), 0)

    @staticmethod
    def run(img, model_dir, threshold_override, min_area, max_area,
            device_name="auto", use_tta=True):
        import torch
        from skimage.morphology import remove_small_objects, binary_closing, disk
        t0 = _time.time()
        if device_name == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_name)
        model_files = sorted(_Path(model_dir).glob("model_fold*.pth"))
        if not model_files:
            raise FileNotFoundError(f"No model_fold*.pth files found in:\n{model_dir}")
        processed = EnsemblePreprocessor.build(img)
        all_proba = []
        all_thr   = []
        skipped   = []
        for mf in model_files:
            try:
                ckpt = torch.load(str(mf), map_location=device, weights_only=False)
            except Exception as e:
                warnings.warn(f"Skipping corrupt model {mf.name}: {e}")
                skipped.append(mf.name)
                continue
            encoder        = ckpt.get("encoder", "resnet34")
            version        = ckpt.get("version", "v7")
            attention_type = "scse" if version in ("v8", "v9") else None
            if not HAS_SMP:
                raise ImportError("segmentation_models_pytorch is not installed.\n"
                                  "pip install segmentation-models-pytorch")
            model = _smp.Unet(
                encoder_name=encoder,
                encoder_weights=None,
                in_channels=3,
                classes=1,
                decoder_attention_type=attention_type,
            ).to(device)
            model.load_state_dict(ckpt["state_dict"])
            tile_sz = ckpt.get("tile", 256)
            opt_thr = ckpt.get("best_thr", 0.5)
            proba = EnsembleEngine._predict_tta(model, processed, device, tile=tile_sz)
            all_proba.append(proba)
            all_thr.append(opt_thr)
        if not all_proba:
            raise RuntimeError(
                f"No valid models loaded (skipped: {skipped}) — check .pth files in: {model_dir}"
            )
        ensemble_proba = np.mean(all_proba, axis=0)
        avg_thr = float(np.mean(all_thr))
        thr = threshold_override if threshold_override is not None else avg_thr
        binary = (ensemble_proba >= thr)
        binary = remove_small_objects(binary, min_size=max(1, min_area))
        binary = binary_closing(binary, disk(1))
        lbl    = sk_label(binary)
        lbl    = _filter_by_morphology(lbl, min_area, max_area)
        props  = _label_to_props(lbl, img.astype(np.float32))
        n_folds = len(all_proba)
        n_skip  = len(skipped)
        skip_note = f", {n_skip} skipped" if n_skip else ""
        return SegmentationResult(
            method_name=f"Ensemble DL ({n_folds} folds{skip_note}, thr={thr:.2f})",
            params=dict(model_dir=model_dir, n_models=n_folds,
                        threshold=thr, avg_thr=avg_thr),
            label_image=lbl,
            n_objects=int(lbl.max()),
            properties=props,
            time_seconds=_time.time() - t0,
        )


# ── EnsembleDLWorker  (BaseWorker adaptation) ─────────────────────────────────
class EnsembleDLWorker(BaseWorker):
    def __init__(self, img, model_dir, threshold_override, min_area, max_area,
                 device_name, use_tta, parent=None):
        super().__init__(parent)
        self.img                = img
        self.model_dir          = model_dir
        self.threshold_override = threshold_override
        self.min_area           = min_area
        self.max_area           = max_area
        self.device_name        = device_name
        self.use_tta            = use_tta

    def run_task(self):
        result = EnsembleEngine.run(
            self.img, self.model_dir,
            threshold_override=self.threshold_override,
            min_area=self.min_area,
            max_area=self.max_area,
            device_name=self.device_name,
            use_tta=self.use_tta,
        )
        return {"result": result}


# ── DLAggregateSubTab ─────────────────────────────────────────────────────────
# --- Aggregate Detection: Deep Learning sub-tab ---
# Load model folder, run ensemble inference, show results
class DLAggregateSubTab(QWidget):
    result_ready = pyqtSignal(object)   # emits SegmentationResult

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state                                        = state
        self.current_result: Optional[SegmentationResult] = None
        self._ensemble_worker: Optional[EnsembleDLWorker] = None
        self._model_dir                                    = ""
        self._build_ui()
        state.subscribe(self._on_state_change)

    # ── State ─────────────────────────────────────────────────────────────────
    def _on_state_change(self, what):
        if what in ("raw_image", "preprocessed", "channel_mappings", "derived_channels"):
            _populate_channel_combo_full(self._agg_ch_combo, self._state,
                                         preferred_label="HA (mHTT)")

    def _get_image(self):
        """Return 2-D float32 (H, W) for the selected channel (raw or derived)."""
        return _get_combo_image(self._agg_ch_combo, self._state)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setSpacing(6)
        main.setContentsMargins(4, 4, 4, 4)

        ls, _, lv = make_scroll_widget()
        ls.setFixedWidth(360)

        # Warnings
        if not HAS_TORCH:
            w = QLabel("⚠  PyTorch not installed.\npip install torch torchvision")
            w.setStyleSheet("color:#ff8800;font-weight:bold;padding:8px;")
            w.setWordWrap(True); lv.addWidget(w)
        if not HAS_SMP:
            w2 = QLabel("⚠  segmentation_models_pytorch not found.\n"
                        "pip install segmentation-models-pytorch")
            w2.setStyleSheet("color:#ff8800;font-size:10px;padding:6px;")
            w2.setWordWrap(True); lv.addWidget(w2)

        # Channel
        ch_grp = section("Channel Assignment")
        ch_lyt = QFormLayout(ch_grp)
        self._agg_ch_combo = QComboBox()
        self._agg_ch_combo.setToolTip(
            "Select the channel containing aggregates (e.g. HA / mHTT).\n"
            "Labels are assigned in Image Viewer → Channel Mapping.")
        ch_lyt.addRow("Aggregate channel:", self._agg_ch_combo)
        lv.addWidget(ch_grp)

        # Ensemble models
        ens_grp = section("Ensemble Models  (folder with model_fold*.pth)")
        ev = QVBoxLayout(ens_grp)
        ens_row = QHBoxLayout()
        self.lbl_ens_path = QLabel("No folder selected")
        self.lbl_ens_path.setStyleSheet("color:#8b949e;font-size:11px;")
        self.lbl_ens_path.setWordWrap(True)
        ens_row.addWidget(self.lbl_ens_path, stretch=1)
        btn_load_dir = QPushButton("📂  Load Model Folder")
        btn_load_dir.clicked.connect(self._load_model_dir)
        ens_row.addWidget(btn_load_dir)
        ev.addLayout(ens_row)
        self.lbl_ens_info = QLabel("")
        self.lbl_ens_info.setStyleSheet("color:#58a6ff;font-size:11px;")
        ev.addWidget(self.lbl_ens_info)
        thr_row = QHBoxLayout()
        self.chk_auto_thr = QCheckBox("Auto-threshold (averaged from models)")
        self.chk_auto_thr.setChecked(True)
        self.chk_auto_thr.toggled.connect(lambda c: self.spn_ens_thr.setEnabled(not c))
        thr_row.addWidget(self.chk_auto_thr)
        ev.addLayout(thr_row)
        self.spn_ens_thr = QDoubleSpinBox()
        self.spn_ens_thr.setRange(0.01, 0.99)
        self.spn_ens_thr.setValue(0.5)
        self.spn_ens_thr.setSingleStep(0.05)
        self.spn_ens_thr.setEnabled(False)
        ev.addLayout(_make_spinbox_row("Fixed threshold:", self.spn_ens_thr,
            "Manual threshold (0.01-0.99). Only active when auto-threshold is off.\n"
            "Lower = more/larger detections.  Higher = fewer/smaller."))
        self.chk_tta = QCheckBox("Test-Time Augmentation (TTA, 8×)")
        self.chk_tta.setChecked(True)
        self.chk_tta.setToolTip("Averages 8 augmentations (4 rotations + 2 flips). ~8× slower but more robust.")
        ev.addWidget(self.chk_tta)
        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Device:"))
        self.cmb_device = QComboBox()
        self.cmb_device.addItems(["auto", "cpu", "cuda", "mps"])
        self.cmb_device.setToolTip("auto = GPU if available, else CPU.\ncuda = NVIDIA GPU.\nmps = Apple Silicon.")
        dev_row.addWidget(self.cmb_device)
        ev.addLayout(dev_row)
        lv.addWidget(ens_grp)

        # Post-processing
        post_grp = section("Post-processing")
        pv = QVBoxLayout(post_grp)
        self.spn_min_area = QSpinBox()
        self.spn_min_area.setRange(1, 9999)
        self.spn_min_area.setValue(5)
        pv.addLayout(_make_spinbox_row("Min area (px²):", self.spn_min_area,
            "Minimum area (px²) of a detected object.\n"
            "Objects smaller than this are removed as noise.\n"
            "Typical value: 5–50 px² depending on resolution."))
        self.spn_max_area = QSpinBox()
        self.spn_max_area.setRange(1, 999999)
        self.spn_max_area.setValue(50000)
        self.spn_max_area.setSingleStep(1000)
        pv.addLayout(_make_spinbox_row("Max area (px²):", self.spn_max_area,
            "Maximum area (px²) of a detected object.\n"
            "Objects larger are removed (e.g. dead cells, debris)."))
        lv.addWidget(post_grp)

        # Overlay
        ov_grp = section("Overlay Options")
        ov_v = QVBoxLayout(ov_grp)
        self.chk_show_circles = QCheckBox("Draw contours")
        self.chk_show_circles.setChecked(True)
        self.chk_show_numbers = QCheckBox("Show numbers")
        self.chk_show_numbers.setChecked(True)
        self.chk_show_fill    = QCheckBox("Filled region")
        self.chk_show_fill.setChecked(True)
        ov_v.addWidget(self.chk_show_circles)
        ov_v.addWidget(self.chk_show_numbers)
        ov_v.addWidget(self.chk_show_fill)
        # Color combo stays as label+combo row
        _col_row = QHBoxLayout()
        _col_lbl = QLabel("Color:"); _col_lbl.setStyleSheet("color:#a0a0cc;")
        _col_lbl.setMinimumWidth(130); _col_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.cmb_circle_color = QComboBox()
        self.cmb_circle_color.addItems(["#00ffcc","#ff4466","#ffff00","#ffffff","#00aaff","#ff8800"])
        _col_row.addWidget(_col_lbl); _col_row.addWidget(self.cmb_circle_color)
        ov_v.addLayout(_col_row)
        self.spn_circle_lw = QDoubleSpinBox()
        self.spn_circle_lw.setRange(0.3, 5); self.spn_circle_lw.setValue(1.2)
        ov_v.addLayout(_make_spinbox_row("Line width:", self.spn_circle_lw))
        self.spn_font_size = QDoubleSpinBox()
        self.spn_font_size.setRange(3, 16); self.spn_font_size.setValue(6.5)
        ov_v.addLayout(_make_spinbox_row("Font size:", self.spn_font_size))
        # Background colormap combo
        _bg_row = QHBoxLayout()
        _bg_lbl = QLabel("Background:"); _bg_lbl.setStyleSheet("color:#a0a0cc;")
        _bg_lbl.setMinimumWidth(130); _bg_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(["hot","gray","inferno","magma","viridis","plasma"])
        _bg_row.addWidget(_bg_lbl); _bg_row.addWidget(self.cmb_cmap)
        ov_v.addLayout(_bg_row)
        lv.addWidget(ov_grp)

        # Run + status
        self.btn_run = QPushButton("▶  Run Ensemble Segmentation")
        self.btn_run.setStyleSheet(BTN_RUN)
        self.btn_run.setToolTip(
            "Runs ensemble DL segmentation on the preprocessed aggregate channel.\n\n"
            "1. Load all model_fold*.pth from selected folder\n"
            "2. Run tiled inference with TTA (8× augmentation)\n"
            "3. Average probability maps across all folds\n"
            "4. Threshold + morphological filtering\n\n"
            "Note: may take several minutes on CPU.")
        self.btn_run.clicked.connect(self._run_ensemble)
        lv.addWidget(self.btn_run)
        self._prog = QProgressBar()
        self._prog.setRange(0, 0)
        self._prog.setVisible(False)
        lv.addWidget(self._prog)
        self._stat_lbl = StatusLabel()
        lv.addWidget(self._stat_lbl)

        # Stats
        stats_grp = section("Statistics")
        sv = QVBoxLayout(stats_grp)
        self.txt_stats = QTextEdit()
        self.txt_stats.setReadOnly(True)
        self.txt_stats.setMinimumHeight(100)
        sv.addWidget(self.txt_stats)
        lv.addWidget(stats_grp)

        btn_csv = QPushButton("💾  Export CSV")
        btn_csv.setToolTip("Export detected object properties (id, area, intensity, etc.) to CSV.")
        btn_csv.clicked.connect(self._export_csv)
        btn_img = QPushButton("🖼  Export annotated image")
        btn_img.setToolTip("Save side-by-side image with input + annotations to PNG/TIFF.")
        btn_img.clicked.connect(self._export_image)
        lv.addWidget(btn_csv)
        lv.addWidget(btn_img)
        lv.addStretch()
        main.addWidget(ls)

        # Canvas area
        right_w = QWidget()
        rv = QVBoxLayout(right_w)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(4)
        self.canvas = ImageCanvas(figsize=(11, 9))
        nav = NavigationToolbar(self.canvas, self)
        nav.setStyleSheet(_viewer_tb_qss())
        rv.addWidget(nav)
        rv.addWidget(self.canvas, stretch=1)
        main.addWidget(right_w, stretch=1)

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _load_model_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with model_fold*.pth", "")
        if not folder:
            return
        model_files = sorted(_Path(folder).glob("model_fold*.pth"))
        if not model_files:
            QMessageBox.warning(self, "No models found",
                f"No model_fold*.pth found in:\n{folder}")
            return
        self._model_dir = folder
        self.lbl_ens_path.setText(f"✅  {_Path(folder).name}")
        self.lbl_ens_info.setText(f"{len(model_files)} models found")
        if HAS_TORCH:
            try:
                import torch
                thresholds = []
                for mf in model_files:
                    ckpt = torch.load(str(mf), map_location="cpu", weights_only=False)
                    if "best_thr" in ckpt:
                        thresholds.append(float(ckpt["best_thr"]))
                if thresholds:
                    avg = float(np.mean(thresholds))
                    self.spn_ens_thr.setValue(avg)
                    self.lbl_ens_info.setText(
                        f"{len(model_files)} models  |  avg. threshold: {avg:.2f}")
            except Exception:
                pass

    def _run_ensemble(self):
        if not self._model_dir:
            QMessageBox.warning(self, "No folder", "Load a model folder first.")
            return
        if not HAS_SMP:
            QMessageBox.critical(self, "Package missing",
                "Install segmentation_models_pytorch:\npip install segmentation-models-pytorch")
            return
        img = self._get_image()
        if img is None:
            QMessageBox.warning(self, "No image",
                "Load and preprocess an image first.")
            return
        thr_override = (None if self.chk_auto_thr.isChecked()
                        else self.spn_ens_thr.value())
        self.btn_run.setEnabled(False)
        self._prog.setVisible(True)
        self._stat_lbl.info("Running ensemble inference…")
        self._ensemble_worker = EnsembleDLWorker(
            img=img,
            model_dir=self._model_dir,
            threshold_override=thr_override,
            min_area=self.spn_min_area.value(),
            max_area=self.spn_max_area.value(),
            device_name=self.cmb_device.currentText(),
            use_tta=self.chk_tta.isChecked(),
            parent=self,
        )
        self._ensemble_worker.signals.result.connect(self._on_result)
        self._ensemble_worker.signals.error.connect(self._on_error)
        self._ensemble_worker.signals.finished.connect(lambda: (
            self.btn_run.setEnabled(True),
            self._prog.setVisible(False),
        ))
        self._ensemble_worker.start()

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self._prog.setVisible(False)
        self._stat_lbl.err("Inference failed.")
        QMessageBox.critical(self, "Deep Learning error", msg)

    def _on_result(self, r: dict):
        result = r.get("result")
        if result is None:
            return
        self.current_result = result
        self._stat_lbl.ok(
            f"Done: {result.n_objects} objects  |  {result.time_seconds:.2f}s")
        self._display_result(result)
        self._show_stats(result)
        # Register as derived channel so CellRegion + Colocalization can use it
        if result.label_image is not None:
            agg_mask = (result.label_image > 0).astype(np.float32)
            self._state.set_derived_channel(
                "aggregates", agg_mask[np.newaxis], source="aggregate_detection")
        self.result_ready.emit(result)

    def _display_result(self, result):
        img  = self._get_image()
        cmap = self.cmb_cmap.currentText()
        self.canvas.fig.clf()
        axes = self.canvas.fig.subplots(1, 2)
        blank = np.zeros((64, 64), dtype=np.float32)
        show_img = img if img is not None else blank
        axes[0].imshow(show_img, cmap=cmap, aspect="equal", interpolation="nearest")
        axes[0].set_title("Preprocessed input", color="#79c0ff", fontsize=10)
        axes[0].axis("off"); axes[0].set_facecolor("#0d1117")
        if (self.chk_show_fill.isChecked()
                and result.label_image is not None
                and result.label_image.max() > 0
                and img is not None):
            from skimage import color as _sc
            img_n = EnsemblePreprocessor.pct_norm(img)
            bg    = np.stack([img_n] * 3, axis=-1)
            ov    = _sc.label2rgb(result.label_image, image=bg,
                                  alpha=0.35, bg_label=0, bg_color=None)
            axes[1].imshow(np.clip(ov, 0, 1), aspect="equal", interpolation="nearest")
        else:
            axes[1].imshow(show_img, cmap=cmap, aspect="equal", interpolation="nearest")
        if self.chk_show_circles.isChecked():
            _dl_draw_contours(
                axes[1], result,
                circle_color=self.cmb_circle_color.currentText(),
                circle_lw=self.spn_circle_lw.value(),
                font_size=self.spn_font_size.value(),
                show_numbers=self.chk_show_numbers.isChecked(),
            )
        axes[1].set_title(
            f"{result.method_name}   N = {result.n_objects}   t = {result.time_seconds:.2f}s",
            color="#79c0ff", fontsize=10)
        axes[1].axis("off"); axes[1].set_facecolor("#0d1117")
        self.canvas.fig.tight_layout(pad=0.5)
        self.canvas.draw_idle()

    def _show_stats(self, result):
        if not result.properties:
            self.txt_stats.setPlainText(
                f"Objects: {result.n_objects}\nNo properties available.")
            return
        areas = [p.get("area_px2", 0) for p in result.properties]
        ints  = [p.get("mean_intensity", 0) for p in result.properties]
        rads  = [p.get("radius_px", 0) for p in result.properties]
        self.txt_stats.setPlainText(
            f"Method:       {result.method_name}\n"
            f"Objects:      {result.n_objects}\n"
            f"Time:         {result.time_seconds:.3f} s\n"
            f"\n── Area (px²) ──\n"
            f"  Mean:    {np.mean(areas):.1f}\n"
            f"  Median:  {np.median(areas):.1f}\n"
            f"  Min/Max: {np.min(areas):.0f} / {np.max(areas):.0f}\n"
            f"\n── Radius (px) ──\n"
            f"  Mean:    {np.mean(rads):.1f}\n"
            f"  Median:  {np.median(rads):.1f}\n"
            f"\n── Mean intensity ──\n"
            f"  Mean:    {np.mean(ints):.4f}\n"
            f"  Median:  {np.median(ints):.4f}\n"
        )

    def _export_csv(self):
        if not self.current_result or not self.current_result.properties:
            QMessageBox.warning(self, "No data", "Run segmentation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "dl_aggregates.csv", "CSV (*.csv)")
        if not path:
            return
        props = self.current_result.properties
        keys  = list(props[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            import csv as _c
            w = _c.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(props)
        QMessageBox.information(self, "Saved", f"{len(props)} objects saved:\n{path}")

    def _export_image(self):
        if self.current_result is None:
            QMessageBox.warning(self, "No result", "Run segmentation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save annotated image", "",
            "PNG (*.png);;TIFF (*.tif)")
        if not path:
            return
        img  = self._get_image()
        cmap = self.cmb_cmap.currentText()
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="#0d1117")
        blank = np.zeros((64, 64), dtype=np.float32)
        show_img = img if img is not None else blank
        axes[0].imshow(show_img, cmap=cmap, aspect="equal")
        axes[0].set_title("Input", color="#79c0ff")
        axes[0].axis("off")
        if (self.chk_show_fill.isChecked()
                and self.current_result.label_image is not None
                and img is not None):
            from skimage import color as _sc3
            img_n = EnsemblePreprocessor.pct_norm(img)
            bg    = np.stack([img_n] * 3, axis=-1)
            ov    = _sc3.label2rgb(self.current_result.label_image, image=bg,
                                   alpha=0.35, bg_label=0, bg_color=None)
            axes[1].imshow(np.clip(ov, 0, 1), aspect="equal")
        else:
            axes[1].imshow(show_img, cmap=cmap, aspect="equal")
        if self.chk_show_circles.isChecked():
            _dl_draw_contours(
                axes[1], self.current_result,
                circle_color=self.cmb_circle_color.currentText(),
                show_numbers=self.chk_show_numbers.isChecked(),
            )
        axes[1].set_title(
            f"{self.current_result.method_name}  N={self.current_result.n_objects}",
            color="#79c0ff")
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        QMessageBox.information(self, "Saved", f"Image saved:\n{path}")


# ── DoubleCanvas  (for ValidationSubTab) ─────────────────────────────────────
class _DoubleCanvas(FigureCanvas):
    OUTER_BG = "#161b22"
    INNER_BG = "#0d1117"

    def __init__(self, parent=None):
        from PyQt5.QtWidgets import QSizePolicy as _SP
        self.fig = Figure(figsize=(10, 5), dpi=100, facecolor=self.OUTER_BG)
        self.axes = self.fig.subplots(1, 2)
        for ax, t in zip(self.axes, ["Ground Truth", "Deep Learning (Ensemble)"]):
            ax.set_facecolor(self.INNER_BG)
            ax.set_title(t, color="#79c0ff", fontsize=10, fontweight="bold")
            ax.axis("off")
        self.fig.tight_layout(pad=1.5)
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, _SP.Expanding, _SP.Expanding)
        FigureCanvas.updateGeometry(self)

    def update_plots(self, gt_mask, dl_mask):
        self.fig.clf()
        axes = self.fig.subplots(1, 2)

        def _mask_rgb(pred, gt):
            if pred is None:
                return None
            p   = pred.astype(bool)
            g   = gt.astype(bool)
            rgb = np.zeros((*g.shape, 3), dtype=np.uint8)
            rgb[g & p]  = [0, 200, 80]
            rgb[p & ~g] = [220, 50, 50]
            rgb[g & ~p] = [50, 100, 220]
            return rgb

        gt_rgb = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        gt_rgb[gt_mask.astype(bool)] = [0, 200, 80]
        overlays  = [gt_rgb, _mask_rgb(dl_mask, gt_mask)]
        titles    = ["Ground Truth", "Deep Learning (Ensemble)"]
        subtitles = ["Ground Truth", "TP=green  FP=red  FN=blue"]

        for ax, overlay, title, sub in zip(axes, overlays, titles, subtitles):
            ax.set_facecolor(self.INNER_BG)
            ax.set_title(title, color="#79c0ff", fontsize=10, fontweight="bold")
            ax.axis("off")
            if overlay is not None:
                ax.imshow(overlay, aspect="equal", interpolation="nearest")
                ax.set_xlabel(sub, color="#8b949e", fontsize=8)
            else:
                ax.text(0.5, 0.5, "Not yet\nsegmented",
                        ha="center", va="center", color="#484f58", fontsize=11,
                        transform=ax.transAxes)
        self.fig.tight_layout(pad=1.5)
        self.draw_idle()


# ── CorrectionSubTab ──────────────────────────────────────────────────────────
# --- Aggregate Detection: Correction sub-tab ---
# Compare DL detections with a ground-truth mask; click to add/remove objects
class CorrectionSubTab(QWidget):
    """Compare DL detections with a manually annotated GT mask.
    GREEN  = DL region overlaps GT (auto-approved)"""
    corrected_mask_ready = pyqtSignal(object)
    _OVERLAP_THRESHOLD   = 0.3

    def __init__(self, dl_tab: DLAggregateSubTab, state, parent=None):
        super().__init__(parent)
        self.dl_tab  = dl_tab
        self._state  = state
        self.gt_mask = None
        self.gt_path = ""
        self._region_status:    Dict[int, bool]        = {}
        self._dl_regions:       List[dict]             = []
        self._gt_region_status: Dict[int, bool]        = {}
        self._gt_label_image:   Optional[np.ndarray]   = None
        self._gt_regions:       List[dict]             = []
        self._cid = None
        self._build_ui()
        dl_tab.result_ready.connect(self._on_dl_updated)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        # Actions
        grp_a = QGroupBox("Actions")
        av = QHBoxLayout(grp_a)
        av.setSpacing(8)
        self.btn_load_gt = QPushButton("📂  Load GT Mask")
        self.btn_load_gt.setToolTip(
            "Load a manually annotated ground-truth mask (TIFF/PNG).\n"
            "White (255) = aggregate present.")
        self.btn_load_gt.clicked.connect(self._load_gt_mask)
        av.addWidget(self.btn_load_gt)
        self.btn_refresh = QPushButton("🔄  Refresh / fetch DL result")
        self.btn_refresh.setToolTip("Fetch the latest DL result and overlay it on the GT mask.")
        self.btn_refresh.clicked.connect(self._auto_classify_and_draw)
        self.btn_refresh.setEnabled(False)
        av.addWidget(self.btn_refresh)
        self.btn_reset = QPushButton("↺  Reset corrections")
        self.btn_reset.clicked.connect(self._reset_corrections)
        self.btn_reset.setEnabled(False)
        av.addWidget(self.btn_reset)
        self.btn_send = QPushButton("✅  Send to Validation")
        self.btn_send.setStyleSheet(BTN_RUN)
        self.btn_send.setToolTip("Send corrected mask to the Validation tab.")
        self.btn_send.clicked.connect(self._send_to_validation)
        self.btn_send.setEnabled(False)
        av.addWidget(self.btn_send)
        self.btn_save = QPushButton("💾  Save mask")
        self.btn_save.clicked.connect(self._save_corrected_mask)
        self.btn_save.setEnabled(False)
        av.addWidget(self.btn_save)
        av.addStretch()
        root.addWidget(grp_a)

        self.lbl_status = QLabel(
            "Step 1: Load GT mask.  "
            "Step 2: Refresh to overlay DL detections.  "
            "Step 3: Click red/blue regions to correct.  "
            "Step 4: Send to Validation.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(
            "color:#8b949e;font-size:11px;padding:4px 8px;"
            "background:#161b22;border-radius:4px;border:1px solid #21262d;")
        root.addWidget(self.lbl_status)

        mid = QHBoxLayout()
        mid.setSpacing(8)

        left = QWidget(); left.setFixedWidth(220)
        lv = QVBoxLayout(left); lv.setSpacing(8)
        grp_leg = QGroupBox("Legend")
        lv2 = QVBoxLayout(grp_leg)
        for color, txt in [
            ("#00c850", "✔  Approved (overlaps GT)"),
            ("#e03030", "✖  Not approved (click to correct)"),
            ("#4488ff", "◌  GT region (click to remove)"),
        ]:
            row = QHBoxLayout()
            dot = QLabel("●"); dot.setStyleSheet(f"color:{color};font-size:18px;")
            lbl = QLabel(txt); lbl.setStyleSheet("color:#c9d1d9;font-size:11px;")
            lbl.setWordWrap(True)
            row.addWidget(dot); row.addWidget(lbl, stretch=1)
            lv2.addLayout(row)
        lv.addWidget(grp_leg)
        grp_cnt = QGroupBox("Counter")
        cv = QFormLayout(grp_cnt)
        self.lbl_n_green     = QLabel("—"); self.lbl_n_green.setStyleSheet("color:#3fb950;font-weight:bold;")
        self.lbl_n_red       = QLabel("—"); self.lbl_n_red.setStyleSheet("color:#f85149;font-weight:bold;")
        self.lbl_n_total     = QLabel("—")
        self.lbl_n_gtremoved = QLabel("—"); self.lbl_n_gtremoved.setStyleSheet("color:#79c0ff;font-weight:bold;")
        cv.addRow("Approved:",   self.lbl_n_green)
        cv.addRow("Rejected:",   self.lbl_n_red)
        cv.addRow("Total DL:",   self.lbl_n_total)
        cv.addRow("GT removed:", self.lbl_n_gtremoved)
        lv.addWidget(grp_cnt)
        lv.addStretch()
        mid.addWidget(left)

        right_w = QWidget(); rv = QVBoxLayout(right_w); rv.setSpacing(4)
        self.canvas_corr = ImageCanvas(figsize=(12, 9))
        nav = NavigationToolbar(self.canvas_corr, self)
        nav.setStyleSheet(_viewer_tb_qss())
        rv.addWidget(nav); rv.addWidget(self.canvas_corr, stretch=1)
        mid.addWidget(right_w, stretch=1)

        root.addLayout(mid, stretch=1)

    def _get_bg_image(self):
        img_vol = (self._state.preprocessed_image
                   if self._state.has_preprocessed()
                   else self._state.raw_image)
        if img_vol is None:
            return None
        plane = img_vol[:, 0, :, :].max(axis=0).astype(np.float32)
        mn, mx = float(plane.min()), float(plane.max())
        if mx > mn:
            return (plane - mn) / (mx - mn)
        return np.zeros_like(plane)

    def _load_gt_mask(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Ground-Truth Mask", "",
            "Mask files (*.tif *.tiff *.png);;All files (*)")
        if not path:
            return
        try:
            mask = _load_gt_mask_from_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot load mask:\n{e}")
            return
        if mask.sum() == 0:
            QMessageBox.warning(self, "Empty mask",
                "The loaded mask contains no foreground pixels.")
            return
        self.gt_mask = mask
        self.gt_path = path
        gt_labeled = sk_label(mask.astype(bool))
        self._gt_label_image  = gt_labeled
        self._gt_region_status = {}
        self._gt_regions       = []
        for region in regionprops(gt_labeled):
            self._gt_region_status[region.label] = True
            self._gt_regions.append({"label": region.label,
                                     "cy": region.centroid[0],
                                     "cx": region.centroid[1],
                                     "area": region.area})
        self.lbl_status.setText(
            f"GT loaded: {_Path(path).name}  |  Shape: {mask.shape}  |  "
            f"Click 'Refresh' to overlay DL detections.")
        self.lbl_status.setStyleSheet(
            "color:#3fb950;font-size:11px;padding:4px 8px;"
            "background:#0d2b0d;border-radius:4px;border:1px solid #238636;")
        self.btn_refresh.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self._auto_classify_and_draw()

    def _auto_classify_and_draw(self):
        if self.gt_mask is None:
            return
        result   = getattr(self.dl_tab, "current_result", None)
        dl_label = result.label_image if result is not None else None
        if dl_label is None:
            self._region_status = {}
            self._dl_regions    = []
            self._draw(None)
            self.lbl_status.setText(
                "No DL segmentation available. Run Deep Learning segmentation first.")
            self.lbl_status.setStyleSheet(
                "color:#d29922;font-size:11px;padding:4px 8px;"
                "background:#2b1d0e;border-radius:4px;border:1px solid #9e6a03;")
            self.btn_send.setEnabled(False)
            return
        gt = self.gt_mask
        if dl_label.shape != gt.shape:
            from skimage.transform import resize as sk_resize
            dl_label = sk_resize(dl_label.astype(float), gt.shape,
                                 order=0, anti_aliasing=False,
                                 preserve_range=True).astype(np.int32)
        gt_bool = gt.astype(bool)
        self._region_status = {}
        self._dl_regions    = []
        for region in regionprops(dl_label):
            lbl_id      = region.label
            region_mask = (dl_label == lbl_id)
            intersection = int(np.logical_and(region_mask, gt_bool).sum())
            overlap      = intersection / region.area if region.area > 0 else 0.0
            self._region_status[lbl_id] = (overlap >= self._OVERLAP_THRESHOLD)
            self._dl_regions.append({"label": lbl_id,
                                     "cy": region.centroid[0],
                                     "cx": region.centroid[1],
                                     "area": region.area})
        self._draw(dl_label)
        self._update_counters()
        self.btn_send.setEnabled(True)
        self.btn_save.setEnabled(True)
        n_green = sum(1 for v in self._region_status.values() if v)
        n_red   = len(self._region_status) - n_green
        self.lbl_status.setText(
            f"{len(self._region_status)} DL regions  |  "
            f"Green: {n_green}  |  Red: {n_red}  |  "
            "Click on a red region to approve it.")
        self.lbl_status.setStyleSheet(
            "color:#79c0ff;font-size:11px;padding:4px 8px;"
            "background:#0d1b2b;border-radius:4px;border:1px solid #1f6feb;")
        if self._cid is not None:
            self.canvas_corr.mpl_disconnect(self._cid)
        self._cid = self.canvas_corr.mpl_connect(
            "button_press_event", self._on_click)

    def _draw(self, dl_label: Optional[np.ndarray]):
        from skimage import measure as sk_measure
        ax = self.canvas_corr.ax
        ax.cla()
        ax.set_facecolor("#0d1117")
        ax.axis("off")
        if self.gt_mask is None:
            ax.text(0.5, 0.5, "No mask loaded",
                    ha="center", va="center", color="#484f58",
                    fontsize=13, transform=ax.transAxes)
            self.canvas_corr.draw_idle()
            return
        bg = self._get_bg_image()
        H, W = self.gt_mask.shape
        if bg is not None:
            ax.imshow(bg, cmap="gray", aspect="equal",
                      interpolation="nearest", zorder=1)
        else:
            dark = np.zeros((H, W, 3), dtype=np.uint8)
            dark[self.gt_mask.astype(bool)] = [50, 50, 70]
            ax.imshow(dark, aspect="equal", interpolation="nearest", zorder=1)
        # GT contours (blue dashed)
        if self._gt_label_image is not None:
            for gt_reg in self._gt_regions:
                if not self._gt_region_status.get(gt_reg["label"], True):
                    continue
                for c in sk_measure.find_contours(
                        (self._gt_label_image == gt_reg["label"]).astype(float), 0.5):
                    ax.plot(c[:, 1], c[:, 0], color="#4488ff",
                            linewidth=1.0, linestyle="--", alpha=0.85, zorder=2)
        elif self.gt_mask is not None:
            for c in sk_measure.find_contours(self.gt_mask.astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], color="#4488ff",
                        linewidth=1.0, linestyle="--", alpha=0.85, zorder=2)
        # DL overlay
        if dl_label is not None and self._region_status:
            green_mask = np.zeros((H, W), dtype=bool)
            red_mask   = np.zeros((H, W), dtype=bool)
            for lbl_id, approved in self._region_status.items():
                px = (dl_label == lbl_id)
                if approved:
                    green_mask |= px
                else:
                    red_mask   |= px
            rgba = np.zeros((H, W, 4), dtype=np.float32)
            rgba[green_mask] = [0.0, 0.78, 0.31, 0.35]
            rgba[red_mask]   = [0.87, 0.19, 0.19, 0.40]
            ax.imshow(rgba, aspect="equal", interpolation="nearest", zorder=3)
            for lbl_id, approved in self._region_status.items():
                color = "#00c850" if approved else "#e03030"
                for c in sk_measure.find_contours(
                        (dl_label == lbl_id).astype(float), 0.5):
                    ax.plot(c[:, 1], c[:, 0], color=color,
                            linewidth=1.4, zorder=5)
            for reg in self._dl_regions:
                lbl_id = reg["label"]
                color  = "#00ff66" if self._region_status.get(lbl_id, False) else "#ff6666"
                ax.text(reg["cx"], reg["cy"], str(lbl_id),
                        color=color, fontsize=6, fontweight="bold",
                        ha="center", va="center", zorder=6, clip_on=True)
        n_green = sum(1 for v in self._region_status.values() if v)
        n_red   = len(self._region_status) - n_green
        n_rem   = sum(1 for v in self._gt_region_status.values() if not v)
        ax.set_title(
            f"Correction view  |  GT (blue dashed) + DL  |  "
            f"Green: {n_green}  ·  Red: {n_red}  ·  GT removed: {n_rem}",
            color="#79c0ff", fontsize=9, pad=4)
        self.canvas_corr.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self.canvas_corr.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        cx_click, cy_click = event.xdata, event.ydata
        result   = getattr(self.dl_tab, "current_result", None)
        dl_label = result.label_image if result is not None else None
        if dl_label is not None and dl_label.shape != self.gt_mask.shape:
            from skimage.transform import resize as sk_resize
            dl_label = sk_resize(dl_label.astype(float), self.gt_mask.shape,
                                 order=0, anti_aliasing=False,
                                 preserve_range=True).astype(np.int32)
        best_dl_lbl, best_dl_dist = None, float("inf")
        for reg in self._dl_regions:
            dist   = np.sqrt((reg["cx"] - cx_click)**2 + (reg["cy"] - cy_click)**2)
            radius = np.sqrt(reg["area"] / np.pi)
            if dist < max(radius * 1.5, 10) and dist < best_dl_dist:
                best_dl_dist = dist; best_dl_lbl = reg["label"]
        best_gt_lbl, best_gt_dist = None, float("inf")
        for reg in self._gt_regions:
            dist   = np.sqrt((reg["cx"] - cx_click)**2 + (reg["cy"] - cy_click)**2)
            radius = np.sqrt(reg["area"] / np.pi)
            if dist < max(radius * 1.5, 10) and dist < best_gt_dist:
                best_gt_dist = dist; best_gt_lbl = reg["label"]
        clicked_dl = best_dl_lbl is not None
        clicked_gt = best_gt_lbl is not None
        if clicked_dl and clicked_gt:
            if best_gt_dist < best_dl_dist:
                clicked_dl = False
            else:
                clicked_gt = False
        if clicked_dl:
            self._region_status[best_dl_lbl] = not self._region_status[best_dl_lbl]
            self._draw(dl_label)
            self._update_counters()
        elif clicked_gt:
            self._gt_region_status[best_gt_lbl] = not self._gt_region_status.get(best_gt_lbl, True)
            self._draw(dl_label)
            self._update_counters()

    def _update_counters(self):
        n_green = sum(1 for v in self._region_status.values() if v)
        n_red   = len(self._region_status) - n_green
        n_rem   = sum(1 for v in self._gt_region_status.values() if not v)
        self.lbl_n_green.setText(str(n_green))
        self.lbl_n_red.setText(str(n_red))
        self.lbl_n_total.setText(str(len(self._region_status)))
        self.lbl_n_gtremoved.setText(
            f"{n_rem} / {len(self._gt_region_status)}"
            if self._gt_region_status else "—")

    def _reset_corrections(self):
        for lbl_id in self._gt_region_status:
            self._gt_region_status[lbl_id] = True
        self._auto_classify_and_draw()

    def _send_to_validation(self):
        if self.gt_mask is None:
            QMessageBox.warning(self, "No mask", "Load a GT mask first.")
            return
        result   = getattr(self.dl_tab, "current_result", None)
        dl_label = result.label_image if result is not None else None
        corrected = self.gt_mask.copy().astype(np.uint8)
        if self._gt_label_image is not None:
            for gt_id, active in self._gt_region_status.items():
                if not active:
                    corrected[self._gt_label_image == gt_id] = 0
        if dl_label is not None:
            if dl_label.shape != self.gt_mask.shape:
                from skimage.transform import resize as sk_resize
                dl_label = sk_resize(dl_label.astype(float), self.gt_mask.shape,
                                     order=0, anti_aliasing=False,
                                     preserve_range=True).astype(np.int32)
            for lbl_id, approved in self._region_status.items():
                if approved:
                    corrected[dl_label == lbl_id] = 1
        self.corrected_mask_ready.emit(corrected)
        QMessageBox.information(self, "Sent",
            f"Corrected mask sent to Validation.\n"
            f"Foreground pixels: {int(corrected.sum()):,}")

    def _save_corrected_mask(self):
        if self.gt_mask is None:
            QMessageBox.warning(self, "No mask", "Load a GT mask first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save corrected mask", "corrected_mask.tif",
            "TIFF (*.tif *.tiff);;PNG (*.png)")
        if not path:
            return
        result   = getattr(self.dl_tab, "current_result", None)
        dl_label = result.label_image if result is not None else None
        corrected = self.gt_mask.copy().astype(np.uint8)
        if self._gt_label_image is not None:
            for gt_id, active in self._gt_region_status.items():
                if not active:
                    corrected[self._gt_label_image == gt_id] = 0
        if dl_label is not None:
            if dl_label.shape != self.gt_mask.shape:
                from skimage.transform import resize as sk_resize
                dl_label = sk_resize(dl_label.astype(float), self.gt_mask.shape,
                                     order=0, anti_aliasing=False,
                                     preserve_range=True).astype(np.int32)
            for lbl_id, approved in self._region_status.items():
                if approved:
                    corrected[dl_label == lbl_id] = 1
        tifffile.imwrite(path, (corrected * 255).astype(np.uint8))
        QMessageBox.information(self, "Saved",
            f"Corrected mask saved:\n{path}\n"
            f"Foreground pixels: {int(corrected.sum()):,}")

    def _on_dl_updated(self, result):
        if self.gt_mask is not None:
            self._auto_classify_and_draw()


# ── ValidationSubTab ──────────────────────────────────────────────────────────
# --- Aggregate Detection: Validation sub-tab ---
# Compute precision / recall / F1 against a ground-truth annotation file
class ValidationSubTab(QWidget):
    """Compare DL result against a GT mask.  Receives corrected mask from CorrectionSubTab."""

    def __init__(self, dl_tab: DLAggregateSubTab, state, parent=None):
        super().__init__(parent)
        self.dl_tab            = dl_tab
        self._state            = state
        self.gt_mask           = None
        self.gt_path           = ""
        self._last_metrics     = None
        self._corrected_mask   = None
        self._build_ui()
        dl_tab.result_ready.connect(self._on_dl_updated)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(10, 10, 10, 10)

        grp_a = QGroupBox("Actions")
        av = QHBoxLayout(grp_a)
        av.setSpacing(8)
        self.btn_load_gt = QPushButton("📂  Load Ground-Truth Mask")
        self.btn_load_gt.setToolTip(
            "Load a manually annotated GT mask (TIFF/PNG).\n"
            "Or send via the Correction tab → auto-loaded.")
        self.btn_load_gt.clicked.connect(self._load_gt_mask)
        av.addWidget(self.btn_load_gt)
        self.btn_compare = QPushButton("📊  Compare Result")
        self.btn_compare.setEnabled(False)
        self.btn_compare.setToolTip(
            "Calculate validation metrics (F1, IoU, Precision, Recall)\n"
            "by comparing the DL result with the GT mask.")
        self.btn_compare.clicked.connect(self._run_comparison)
        av.addWidget(self.btn_compare)
        self.btn_export = QPushButton("💾  Export Report (CSV)")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_csv)
        av.addWidget(self.btn_export)
        av.addStretch()
        root.addWidget(grp_a)

        self.lbl_status = QLabel("Step 1: Load a ground-truth mask or receive from Correction tab.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(
            "color:#8b949e;font-size:11px;padding:4px 8px;"
            "background:#161b22;border-radius:4px;border:1px solid #21262d;")
        root.addWidget(self.lbl_status)

        grp_m = QGroupBox("Validation Metrics")
        mv = QVBoxLayout(grp_m)
        self.tbl_metrics = QTableWidget(6, 2)
        self.tbl_metrics.setHorizontalHeaderLabels(["Metric", "Deep Learning (Ensemble)"])
        self.tbl_metrics.verticalHeader().setVisible(False)
        self.tbl_metrics.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_metrics.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl_metrics.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_metrics.setMaximumHeight(230)
        self._populate_empty_table()
        mv.addWidget(self.tbl_metrics)
        root.addWidget(grp_m)

        grp_v = QGroupBox("Visual Comparison  —  Ground Truth | Deep Learning")
        vv = QVBoxLayout(grp_v)
        self.canvas = _DoubleCanvas()
        nav = NavigationToolbar(self.canvas, self)
        nav.setStyleSheet(_viewer_tb_qss())
        vv.addWidget(nav); vv.addWidget(self.canvas)
        legend_bar = QHBoxLayout()
        for color, lbl_txt in [("#00c850","TP — True Positive"),
                                ("#dc3232","FP — False Positive"),
                                ("#3264dc","FN — False Negative")]:
            dot = QLabel("●"); dot.setStyleSheet(f"color:{color};font-size:16px;")
            lbl = QLabel(lbl_txt); lbl.setStyleSheet("color:#8b949e;font-size:11px;")
            legend_bar.addWidget(dot); legend_bar.addWidget(lbl)
            legend_bar.addSpacing(16)
        legend_bar.addStretch()
        vv.addLayout(legend_bar)
        root.addWidget(grp_v, stretch=1)

    def _populate_empty_table(self):
        rows = [("F1-score (Dice)","—"),("IoU (Jaccard)","—"),
                ("Precision","—"),("Recall","—"),
                ("True Positives","—"),("False Positives","—")]
        self._fill_table(rows, highlight=False)

    def _fill_table(self, rows, highlight=True):
        SCORE_ROWS = {0, 1, 2, 3}
        for i, (metric, val) in enumerate(rows):
            item_m = QTableWidgetItem(metric)
            item_m.setFont(QFont("Segoe UI", 10, QFont.Bold))
            item_m.setForeground(QColor("#79c0ff"))
            self.tbl_metrics.setItem(i, 0, item_m)
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            if highlight and i in SCORE_ROWS and val not in ("—","N/A"):
                try:
                    score = float(val)
                    if score >= 0.75:
                        item.setBackground(QColor("#0d2b0d"))
                        item.setForeground(QColor("#3fb950"))
                    elif score >= 0.50:
                        item.setBackground(QColor("#2b1d0e"))
                        item.setForeground(QColor("#d29922"))
                    else:
                        item.setBackground(QColor("#2b0a0a"))
                        item.setForeground(QColor("#f85149"))
                except ValueError:
                    item.setForeground(QColor("#c9d1d9"))
            else:
                item.setForeground(QColor("#c9d1d9"))
            self.tbl_metrics.setItem(i, 1, item)
        self.tbl_metrics.resizeRowsToContents()

    def _load_gt_mask(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Ground-Truth Mask", "",
            "Mask files (*.tif *.tiff *.png);;All files (*)")
        if not path:
            return
        try:
            mask = _load_gt_mask_from_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot load mask:\n{e}")
            return
        if mask.sum() == 0:
            QMessageBox.warning(self, "Empty mask",
                "The loaded mask contains no foreground pixels.")
            return
        self.gt_mask = mask
        self.gt_path = path
        n_pos = int(mask.sum())
        self.lbl_status.setText(
            f"GT loaded: {_Path(path).name}  |  "
            f"Shape: {mask.shape}  |  Foreground: {n_pos:,} px")
        self.lbl_status.setStyleSheet(
            "color:#3fb950;font-size:11px;padding:4px 8px;"
            "background:#0d2b0d;border-radius:4px;border:1px solid #238636;")
        self.btn_compare.setEnabled(True)

    def receive_corrected_mask(self, corrected_mask: np.ndarray):
        """Slot: receives corrected mask from CorrectionSubTab."""
        self._corrected_mask = corrected_mask
        n_pos = int(corrected_mask.sum())
        self.lbl_status.setText(
            f"Corrected GT received from Correction tab  |  "
            f"Shape: {corrected_mask.shape}  |  Foreground: {n_pos:,} px  |  "
            "Click 'Compare Result' to validate.")
        self.lbl_status.setStyleSheet(
            "color:#3fb950;font-size:11px;padding:4px 8px;"
            "background:#0d2b0d;border-radius:4px;border:1px solid #238636;")
        self.gt_mask = corrected_mask
        self.gt_path = "(corrected via Correction tab)"
        self.btn_compare.setEnabled(True)

    def _run_comparison(self):
        if self.gt_mask is None:
            QMessageBox.warning(self, "No GT", "Load a ground-truth mask first.")
            return
        dl_mask = _result_to_binary_mask(getattr(self.dl_tab, "current_result", None))
        if dl_mask is None:
            QMessageBox.warning(self, "No segmentation",
                "Run Deep Learning segmentation first.")
            return
        gt = self.gt_mask
        def _resize_if_needed(m, target):
            if m is None or m.shape == target: return m
            from skimage.transform import resize as sk_resize
            r = sk_resize(m.astype(float), target, order=0,
                          anti_aliasing=False, preserve_range=True)
            return (r > 0.5).astype(np.uint8)
        dl_mask = _resize_if_needed(dl_mask, gt.shape)
        m_dl = _compute_validation_metrics(dl_mask, gt)
        def _v(m, key):
            v = m.get(key)
            if v is None: return "N/A"
            return f"{v:.4f}" if isinstance(v, float) else f"{v:,}"
        rows = [
            ("F1-score (Dice)",  _v(m_dl, "f1")),
            ("IoU (Jaccard)",    _v(m_dl, "iou")),
            ("Precision",        _v(m_dl, "precision")),
            ("Recall",           _v(m_dl, "recall")),
            ("True Positives",   _v(m_dl, "tp")),
            ("False Positives",  _v(m_dl, "fp")),
        ]
        self._fill_table(rows, highlight=True)
        self.canvas.update_plots(gt, dl_mask)
        self._last_metrics = {"dl": m_dl}
        self.btn_export.setEnabled(True)
        f1 = m_dl.get("f1")
        verdict = f"F1 = {f1:.4f}" if f1 is not None else "no metrics"
        self.lbl_status.setText(f"Comparison complete  |  {verdict}")
        self.lbl_status.setStyleSheet(
            "color:#79c0ff;font-size:11px;padding:4px 8px;"
            "background:#0d1b2b;border-radius:4px;border:1px solid #1f6feb;")

    def _export_csv(self):
        if self._last_metrics is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export report", "validation_report.csv", "CSV (*.csv)")
        if not path:
            return
        keys   = ["f1","iou","precision","recall","tp","fp","fn"]
        labels = ["F1-score (Dice)","IoU (Jaccard)","Precision","Recall",
                  "True Positives","False Positives","False Negatives"]
        m_dl = self._last_metrics["dl"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            import csv as _c
            w = _c.writer(f)
            w.writerow(["Metric","DeepLearning_Ensemble"])
            w.writerow(["Ground-truth mask", self.gt_path])
            w.writerow([])
            for key, lbl_txt in zip(keys, labels):
                dv = m_dl.get(key)
                w.writerow([
                    lbl_txt,
                    f"{dv:.6f}" if isinstance(dv, float) else (str(dv) if dv is not None else "N/A"),
                ])
        QMessageBox.information(self, "Exported", f"Report saved:\n{path}")

    def _on_dl_updated(self, result):
        self.lbl_status.setText(
            f"DL updated: {result.method_name} ({result.n_objects} objects). "
            "Click 'Compare Result' to validate.")


# ── Container AggregateDetectionTab ──────────────────────────────────────────
# Container tab: three sub-tabs (Deep Learning · Correction · Validation)
class AggregateDetectionTab(QWidget):
    """Container: 3 sub-tabs (Deep Learning / Correction / Validation)."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state    = state
        self._sub_tabs = QTabWidget()

        self._dl_tab   = DLAggregateSubTab(state, self)
        self._corr_tab = CorrectionSubTab(self._dl_tab, state, self)
        self._val_tab  = ValidationSubTab(self._dl_tab, state, self)

        # Wire signals
        self._corr_tab.corrected_mask_ready.connect(self._val_tab.receive_corrected_mask)

        self._sub_tabs.addTab(self._dl_tab,   "Deep Learning")
        self._sub_tabs.addTab(self._corr_tab, "Correction")
        self._sub_tabs.addTab(self._val_tab,  "Validation")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._sub_tabs)

    # Property forwarding so AutoOptimization can find these
    @property
    def _agg_ch_combo(self):
        return self._dl_tab._agg_ch_combo

    @property
    def spn_min_area(self):
        return self._dl_tab.spn_min_area

    @property
    def spn_max_area(self):
        return self._dl_tab.spn_max_area


# CELL REGION ANALYSIS  
# Computes nucleus / perinuclear / cytoplasm / periphery region masks per cell
class CellRegionWorker(BaseWorker):
    """Splits each cell into 4 regions: nucleus, perinuclear shell,
    cytoplasm, and cell periphery. Each region mask stores the cell_id value."""

    def __init__(self, nucleus_mask: np.ndarray,
                 cell_body_mask: np.ndarray,
                 periphery_px: int = 2,
                 peri_r_factor: float = 0.5,
                 parent=None):
        super().__init__(parent)
        self._nuc          = nucleus_mask
        self._cell         = cell_body_mask
        self._peri_px      = periphery_px
        self._peri_r_factor = peri_r_factor

    def run_task(self):
        from scipy.ndimage import (distance_transform_edt as _edt,
                                   binary_erosion        as _ber)

        self.signals.progress.emit("Preparing masks…")

        # ── Flatten to 2-D ──────────────────────────────────────────────────
        nuc_raw  = self._nuc
        cell_raw = self._cell
        if nuc_raw.ndim == 3:
            nuc_raw  = nuc_raw.max(axis=0)
        if cell_raw.ndim == 3:
            cell_raw = cell_raw.max(axis=0)

        # ── Ensure integer labels ────────────────────────────────────────────
        if nuc_raw.dtype == bool or nuc_raw.max() <= 1:
            nuc_labeled  = sk_label(nuc_raw > 0).astype(np.int32)
        else:
            nuc_labeled  = nuc_raw.astype(np.int32)

        if cell_raw.dtype == bool or cell_raw.max() <= 1:
            cell_labeled = sk_label(cell_raw > 0).astype(np.int32)
        else:
            cell_labeled = cell_raw.astype(np.int32)

        H, W = nuc_labeled.shape

        # ── Output accumulators: values = float(cell_id) for ownership ───────
        out_nuc   = np.zeros((H, W), dtype=np.float32)
        out_peri  = np.zeros((H, W), dtype=np.float32)
        out_cyto  = np.zeros((H, W), dtype=np.float32)
        out_perip = np.zeros((H, W), dtype=np.float32)

        cell_ids  = [c for c in np.unique(cell_labeled) if c != 0]
        n_cells   = len(cell_ids)
        radii_all = []
        struct3   = np.ones((3, 3), dtype=bool)

        # ── Majority-vote: assign each nucleus to the cell with most overlap ──
        nuc_ids = [n for n in np.unique(nuc_labeled) if n != 0]
        nuc_to_cell = {}  # r3 - maps nucleus_id -> dominant cell_id
        for nid in nuc_ids:
            nuc_px = nuc_labeled == nid
            best_cid, best_cnt = 0, 0
            for cid in cell_ids:
                overlap = int((nuc_px & (cell_labeled == cid)).sum())
                if overlap > best_cnt:
                    best_cnt, best_cid = overlap, cid
            nuc_to_cell[nid] = best_cid  # r3 - 0 means nucleus is outside all cells
        # Build a resolved nucleus map: nuc pixels carry their assigned cell_id
        nuc_resolved = np.zeros((H, W), dtype=np.int32)
        for nid, cid in nuc_to_cell.items():
            if cid != 0:
                nuc_resolved[nuc_labeled == nid] = cid

        self.signals.progress.emit(f"Processing {n_cells} cells individually…")

        for idx, cid in enumerate(cell_ids):
            if idx % max(1, n_cells // 10) == 0:
                self.signals.progress.emit(
                    f"Cell {idx+1}/{n_cells}  (id={cid})…")

            # ── Single cell body mask ────────────────────────────────────────
            cell_px = cell_labeled == cid

            # ── Nucleus pixels inside this cell ─────────────────────────────
            nuc_in_cell = (nuc_resolved == cid) & cell_px  # r3 - only nuclei majority-assigned to this cell
            if not nuc_in_cell.any():
                # No nucleus -> whole interior is cytoplasm
                out_cyto[cell_px] = float(cid)
                continue

            nuc_local = sk_label(nuc_in_cell).astype(np.int32)
            nuc_bool  = nuc_local > 0

            props_loc = regionprops(nuc_local)
            if props_loc:
                r_mean = float(np.mean([np.sqrt(p.area / np.pi) for p in props_loc]))
            else:
                r_mean = float(np.sqrt(nuc_bool.sum() / np.pi)) if nuc_bool.any() else 15.0
            radii_all.append(r_mean)

            # ── EDT from nucleus surface ─────────────────────────────────────
            dist_from_nuc = _edt(~nuc_bool).astype(np.float32)
            dist_from_nuc[~cell_px] = 0.0

            # ── Distance from cell edge inward ───────────────────────────────
            dist_from_edge = _edt(cell_px).astype(np.float32)

            # ── Shape-aware perinuclear shell  ─────────────
            # For each exterior pixel we find its nearest nucleus BOUNDARY pixel
            # (via EDT return_indices), then compute:
            #   local_r = dist(nucleus_centroid → that boundary pixel)
            #           = the local nucleus radius in that outward direction
            #
            # A pixel is perinuclear when:
            #   dist_from_nuc  ≤  peri_r_factor × local_r
            #
            # This gives a shell that is:
            #   • Thicker on the long axis of elongated nuclei (large local_r)
            #   • Thinner on short sides (small local_r)
            #   • Follows the actual nucleus contour — not a uniform ring
            #
            # peri_r_factor (default 0.5) is user-controllable via the UI.
            nuc_inner = _ber(nuc_bool, structure=struct3, border_value=0)
            nuc_bnd   = nuc_bool & ~nuc_inner
            if not nuc_bnd.any():
                nuc_bnd = nuc_bool  # degenerate: single-pixel nucleus

            # Nearest nucleus boundary pixel for every (y, x)
            _, bnd_idx = _edt(~nuc_bnd, return_indices=True)

            # Nucleus centroid (within this cell only)
            nuc_ys, nuc_xs = np.where(nuc_bool)
            cy_loc = float(nuc_ys.mean())
            cx_loc = float(nuc_xs.mean())

            # local_r[y,x] = dist from centroid to the boundary pixel
            # that is nearest to (y,x) — i.e. the local nucleus radius
            # in the direction of (y,x)
            local_r = np.hypot(
                bnd_idx[0].astype(np.float32) - cy_loc,
                bnd_idx[1].astype(np.float32) - cx_loc,
            )
            local_r = np.maximum(local_r, 1.0)   # avoid zero for tiny nuclei

            peri_ring = (
                cell_px     &
                (~nuc_bool) &
                (dist_from_nuc > 0) &
                (dist_from_nuc <= self._peri_r_factor * local_r)
            )
            periph_band = (
                cell_px &
                (~nuc_bool) &
                (~peri_ring) &
                (dist_from_edge <= self._peri_px)
            )

            # ── Cytoplasm: exclusive to THIS cell ────────────────────────────
            cyto = cell_px & (~nuc_bool) & (~peri_ring) & (~periph_band)

            # ── Store with cell_id value: preserves ownership ─────────────────
            out_nuc  [nuc_bool]    = float(cid)
            out_peri [peri_ring]   = float(cid)
            out_perip[periph_band] = float(cid)
            out_cyto [cyto]        = float(cid)

        r_est = float(np.mean(radii_all)) if radii_all else 30.0
        print(f"[region] per-cell done  n={n_cells}  mean_r={r_est:.1f}px  "
              f"peri_r_factor={self._peri_r_factor}  "
              f"mean_peri_width≈{self._peri_r_factor * r_est:.1f}px")
        self.signals.progress.emit("Done — packaging results…")

        def _to_f32(arr):
            return arr.astype(np.float32)[np.newaxis]

        return {
            "nucleus":        _to_f32(out_nuc),
            "perinuclear":    _to_f32(out_peri),
            "cytoplasm":      _to_f32(out_cyto),
            "periphery":      _to_f32(out_perip),
            "r_est":          r_est,
            "peri_r_factor":  self._peri_r_factor,
            "dist_from_nuc":  np.zeros((H, W), dtype=np.float32),

            "cell_labeled":   cell_labeled,
            "nuc_to_cell":    nuc_to_cell,   # r3 - nucleus-to-cell assignment dict
            "nuc_resolved":   nuc_resolved,  # r3 - resolved nucleus label map
        }


# --- Cell Region Analysis Tab ---
# Per-cell aggregate quantification across 4 sub-cellular regions
class CellRegionAnalysisTab(QWidget):
    """NEW TAB — Cell Region Analysis.
    Classifies every pixel into one of four biological regions:"""

    # Overlay colours for each region  (RGBA)
    REGION_COLORS = {
        "nucleus":     (0.2, 0.4, 1.0, 0.55),   # blue
        "perinuclear": (1.0, 0.6, 0.0, 0.55),   # orange
        "cytoplasm":   (0.2, 0.8, 0.2, 0.45),   # green
        "periphery":   (0.9, 0.2, 0.9, 0.45),   # magenta
    }
    REGION_LABELS = ["nucleus", "perinuclear", "cytoplasm", "periphery"]

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self._state  = state
        self._result = None
        self._worker = None
        # per-region overlay toggles
        self._region_checks = {}
        self._build_ui()
        state.subscribe(self._on_state_change)

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        ls, _, ll = make_scroll_widget()
        ls.setFixedWidth(310)

        # ── Source masks ──────────────────────────────────────────────────
        src_grp = section("Source Masks")
        src_lyt = QFormLayout(src_grp)

        self._nuc_src_combo = QComboBox()
        self._nuc_src_combo.addItems([
            "nucleus_mask  (from Cell Body tab)",
            "region_nucleus  (re-run)",
            "— manual: select channel —",
        ])

        self._nuc_src_combo.setToolTip(
            "Which nucleus mask to use for per-cell region computation.\n\n"
            "nucleus_mask (from Cell Body tab)  — use the labeled mask produced by\n"
            "  'Run Nucleus Segmentation' in Cell Body & Nuclei Detection.\n"
            "  This is the recommended option.\n\n"
            "region_nucleus (re-run)  — re-derive nucleus pixels from a previously\n"
            "  computed region channel (advanced).\n\n"
            "Manual  — pick any derived channel as the nucleus mask source.\n\n"
            "Prerequisite: run nucleus segmentation and select the active Z-layer\n"
            "before clicking 'Run Cell Region Analysis'.")
        src_lyt.addRow("Nucleus source:", self._nuc_src_combo)

        self._cell_src_combo = QComboBox()
        self._cell_src_combo.addItems([
            "cell_body_mask  (from Cell Body tab)",
            "— use nucleus as proxy —",
        ])
        self._cell_src_combo.setToolTip(
            "Which cell body mask to use for per-cell region computation.\n\n"
            "cell_body_mask (from Cell Body tab)  — use the labeled cell mask\n"
            "  produced by 'Run Cell Body Segmentation'.\n"
            "  Recommended: gives cytoplasm and periphery exclusive to each cell.\n\n"
            "Use nucleus as proxy  — dilates the nucleus mask to approximate\n"
            "  cell boundaries when no cell body stain is available.\n"
            "  Less accurate; periphery band will be estimated.\n\n"
            "Prerequisite: run cell body segmentation and select the active Z-layer\n"
            "before clicking 'Run Cell Region Analysis'.")
        src_lyt.addRow("Cell body source:", self._cell_src_combo)

        ll.addWidget(src_grp)

        # ── Parameters ───────────────────────────────────────────────────
        param_grp = section("Parameters")
        param_lyt = QVBoxLayout(param_grp)

        # Periphery width
        self._peri_px_sp = QSpinBox()
        self._peri_px_sp.setRange(1, 200)
        self._peri_px_sp.setValue(2)
        self._peri_px_sp.setToolTip(
            "Width of the cell periphery band in pixels.\n\n"
            "Pixels within this distance from the cell body edge (inward)\n"
            "are classified as 'cell periphery' (pericellular / cortical zone).\n\n"
            "Typical range: 5 – 30 px (at 20× magnification ≈ 0.5–3 µm).")
        param_lyt.addLayout(_make_spinbox_row("Periphery width (px):", self._peri_px_sp))


        self._peri_r_sp = QDoubleSpinBox()
        self._peri_r_sp.setRange(0.1, 2.0)
        self._peri_r_sp.setSingleStep(0.05)
        self._peri_r_sp.setDecimals(2)
        self._peri_r_sp.setValue(0.5)
        self._peri_r_sp.setToolTip(
            "Perinuclear shell thickness multiplier  (r-factor, default 0.5).\n\n"
            "The shell thickness at each point around the nucleus equals:\n"
            "  thickness = r-factor × local_nucleus_radius\n\n"
            "Where 'local_nucleus_radius' is the distance from the nucleus\n"
            "CENTROID to the nucleus BOUNDARY in that outward direction.\n"
            "This makes the shell shape-aware:\n"
            "  • Thicker on the long sides of elongated nuclei\n"
            "  • Thinner on the short sides\n"
            "  • Follows the actual nucleus contour — not a uniform ring\n\n"
            "Increase (e.g. 0.8) for a wider perinuclear zone.\n"
            "Decrease (e.g. 0.3) for a tighter zone closer to the nucleus.")
        param_lyt.addLayout(_make_spinbox_row("Perinuclear r-factor:", self._peri_r_sp))

        ll.addWidget(param_grp)

        self._run_btn = QPushButton("▶  Run Cell Region Analysis")
        self._run_btn.setStyleSheet(BTN_RUN)
        self._run_btn.setEnabled(False)
        ll.addWidget(self._run_btn)

        self._prog     = QProgressBar(); self._prog.setRange(0, 0); self._prog.setVisible(False)
        self._stat_lbl = StatusLabel()
        ll.addWidget(self._prog); ll.addWidget(self._stat_lbl)

        # ── Overlay toggles ───────────────────────────────────────────────
        ov_grp = section("Region Overlays")
        ov_lyt = QVBoxLayout(ov_grp)
        _color_names = {"nucleus": "Blue", "perinuclear": "Orange",
                        "cytoplasm": "Green", "periphery": "Magenta"}
        for key in self.REGION_LABELS:
            cb = QCheckBox(f"Show {key}  [{_color_names[key]}]")
            cb.setChecked(True)
            cb.stateChanged.connect(self._refresh_overlay)
            self._region_checks[key] = cb
            ov_lyt.addWidget(cb)
        ll.addWidget(ov_grp)

        # ── Signal metrics ────────────────────────────────────────────────
        sig_grp = section("Signal Metrics")
        sig_lyt = QVBoxLayout(sig_grp)
        self._metrics_combo = QComboBox()
        self._metrics_combo.addItem("— none (run analysis first) —")

        self._metrics_combo.setToolTip(
            "Select a derived channel to measure per-region signal and object counts.\n\n"
            "Available channels include:\n"
            "  • Aggregates      — segmented aggregate masks from Aggregate Detection\n"
            "  • nucleus_mask    — nucleus pixels\n"
            "  • cell_body_mask  — cell body pixels\n"
            "  • Any other registered derived channel\n\n"
            "Results show:\n"
            "  Pixels     — total region area in pixels\n"
            "  % signal   — fraction of channel intensity in this region\n"
            "  Objects    — distinct connected structures inside this region\n"
            "               (e.g. number of aggregates in the perinuclear zone)\n\n"
            "Object counting uses Otsu thresholding + connected-component labelling.")
        sig_lyt.addWidget(QLabel("Measure signal channel:"))
        sig_lyt.addWidget(self._metrics_combo)
        self._measure_btn = QPushButton("📊  Compute Region % + Inclusion Analysis")
        self._measure_btn.setEnabled(False)
        sig_lyt.addWidget(self._measure_btn)
        ll.addWidget(sig_grp)

        # ── Aggregate Summary (moved to left panel) ───────────────────────
        agg_sum_grp = section("Aggregate Summary")
        agg_sum_lyt = QVBoxLayout(agg_sum_grp)
        self._agg_sum_lbl = QLabel("Run region analysis with aggregates channel\nto see inclusion statistics.")
        self._agg_sum_lbl.setWordWrap(True)
        self._agg_sum_lbl.setStyleSheet("color:#80ff80;font-family:monospace;font-size:11px;")
        agg_sum_lyt.addWidget(self._agg_sum_lbl)
        self._export_csv_btn = QPushButton("💾  Export Per-Cell Data (CSV)")
        self._export_csv_btn.setStyleSheet(BTN_PURPLE)
        self._export_csv_btn.setEnabled(False)
        self._export_csv_btn.setToolTip("Export the per-cell aggregate table to CSV.")
        self._export_csv_btn.clicked.connect(self._export_cell_csv)
        agg_sum_lyt.addWidget(self._export_csv_btn)
        ll.addWidget(agg_sum_grp)

        ll.addStretch()
        root.addWidget(ls)

        # ── Canvas ────────────────────────────────────────────────────────
        cw = QWidget(); cl = QVBoxLayout(cw); cl.setContentsMargins(0, 0, 0, 0)
        self._canvas = ImageCanvas(figsize=(7, 6))
        self._nav    = NavigationToolbar(self._canvas, self)
        self._nav.setStyleSheet(_viewer_tb_qss())
        # Mouse-hover handler: shows the cell ID + per-cell stats under the cursor
        self._canvas.mpl_connect("motion_notify_event", self._on_hover)
        # Initialised by _refresh_overlay / _populate_cell_stats
        self._cell_labeled     = None
        self._cell_stats_rows  = {}
        cl.addWidget(self._nav); cl.addWidget(self._canvas, stretch=1)
        root.addWidget(cw, stretch=1)

        # ── Right panel: tabbed tables + aggregate summary at bottom ──────────
        rw = QWidget()
        rw.setMinimumWidth(360)
        rl = QVBoxLayout(rw)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)

        # Region summary line (compact, always visible)
        sg = section("Region Summary")
        sl = QVBoxLayout(sg)
        self._sum_lbl = QLabel("Run analysis to see region breakdown.")
        self._sum_lbl.setWordWrap(True)
        self._sum_lbl.setStyleSheet("color:#80ff80;font-family:monospace;font-size:10px;")
        sl.addWidget(self._sum_lbl)
        rl.addWidget(sg)

        # Display toggles (canvas overlay options)
        disp_grp = section("Display")
        disp_lyt = QVBoxLayout(disp_grp)

        # Per-channel display rows (checkbox + channel selector combo)
        _ch1_row_w = QWidget(); _ch1_row_l = QHBoxLayout(_ch1_row_w)
        _ch1_row_l.setContentsMargins(0, 0, 0, 0)
        self._display_ch1       = QCheckBox("Ch 1:")
        self._display_ch1.setChecked(True)
        self._display_ch1_combo = QComboBox()
        self._display_ch1_combo.setToolTip("Channel shown in display slot 1 (gray)")
        _ch1_row_l.addWidget(self._display_ch1)
        _ch1_row_l.addWidget(self._display_ch1_combo, stretch=1)
        disp_lyt.addWidget(_ch1_row_w)

        _ch2_row_w = QWidget(); _ch2_row_l = QHBoxLayout(_ch2_row_w)
        _ch2_row_l.setContentsMargins(0, 0, 0, 0)
        self._display_ch2       = QCheckBox("Ch 2:")
        self._display_ch2.setChecked(False)
        self._display_ch2_combo = QComboBox()
        self._display_ch2_combo.setToolTip("Channel shown in display slot 2 (gray, overlaid)")
        _ch2_row_l.addWidget(self._display_ch2)
        _ch2_row_l.addWidget(self._display_ch2_combo, stretch=1)
        disp_lyt.addWidget(_ch2_row_w)

        self._display_masks    = QCheckBox("Show region masks (colored fills)")
        self._display_masks.setChecked(True)
        self._display_outlines = QCheckBox("Show cell outlines")
        self._display_outlines.setChecked(True)
        self._display_ids      = QCheckBox("Show cell IDs")
        self._display_ids.setChecked(False)
        self._display_aggs     = QCheckBox("Show aggregates")
        self._display_aggs.setChecked(False)
        self._display_ch1.stateChanged.connect(self._refresh_overlay)
        self._display_ch2.stateChanged.connect(self._refresh_overlay)
        self._display_ch1_combo.currentIndexChanged.connect(self._refresh_overlay)
        self._display_ch2_combo.currentIndexChanged.connect(self._refresh_overlay)
        for _cb in (self._display_masks, self._display_outlines,
                    self._display_ids, self._display_aggs):
            _cb.stateChanged.connect(self._refresh_overlay)
            disp_lyt.addWidget(_cb)
        self._hover_lbl = QLabel("Hover a cell to see its stats.")
        self._hover_lbl.setWordWrap(True)
        self._hover_lbl.setStyleSheet(
            "color:#ffd080;font-family:monospace;font-size:11px;"
            "background:#14142a;padding:4px;border:1px solid #2a2a4a;")
        disp_lyt.addWidget(self._hover_lbl)
        rl.addWidget(disp_grp)

        # Three data tables in a QTabWidget so each gets full panel width
        _data_tabs = QTabWidget()
        _data_tabs.setStyleSheet(
            "QTabWidget::pane{border:1px solid #2a2a4a;background:#0a0a14;}"
            "QTabBar::tab{background:#14142a;color:#8080c0;padding:5px 12px;border:1px solid #2a2a4a;}"
            "QTabBar::tab:selected{background:#1e1e3a;color:#c0c0ff;border-bottom:2px solid #6060ff;}")

        # Tab 1 — Signal per Region
        _sig_w = QWidget(); _sig_l = QVBoxLayout(_sig_w)
        self._sig_table = QTableWidget()
        self._sig_table.setStyleSheet(
            "QTableWidget{background:#0a0a14;color:#c0c0e0;font-size:12px;}"
            "QHeaderView::section{background:#14142a;color:#8080c0;font-size:12px;padding:4px;}")
        self._sig_table.setColumnCount(4)
        self._sig_table.setHorizontalHeaderLabels(["Region", "Pixels", "% signal", "Objects"])
        self._sig_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._sig_table.setAlternatingRowColors(True)
        self._sig_table.setToolTip(
            "Signal in each region for the selected channel.\n"
            "Pixels = total area · % signal = intensity fraction · Objects = aggregate count")
        _sig_l.addWidget(self._sig_table)
        _data_tabs.addTab(_sig_w, "Signal per Region")

        # Tab 2 — Per-Cell Statistics
        _cell_w = QWidget(); _cell_l = QVBoxLayout(_cell_w)
        self._cell_stats_table = QTableWidget()
        self._cell_stats_table.setStyleSheet(
            "QTableWidget{background:#0a0a14;color:#c0c0e0;font-size:12px;}"
            "QHeaderView::section{background:#14142a;color:#8080c0;font-size:11px;padding:3px;}")
        self._cell_stats_table.setColumnCount(13)
        self._cell_stats_table.setHorizontalHeaderLabels([
            "Cell ID",
            "Nuc px", "Peri px", "Cyto px", "Periph px",
            "NucAgg", "PeriAgg", "CytoAgg", "PeriphAgg",
            "Total Agg", "Agg Area px²",
            "Nuc Incl", "Cyto Incl",
        ])
        self._cell_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._cell_stats_table.setSortingEnabled(True)
        self._cell_stats_table.setAlternatingRowColors(True)
        self._cell_stats_table.setToolTip(
            "Per-cell aggregate quantification.\n"
            "NucAgg/PeriAgg/CytoAgg/PeriphAgg = aggregate objects per region.\n"
            "Nuc Incl = YES if ≥1 aggregate in nucleus zone.\n"
            "Cyto Incl = YES if ≥1 aggregate in perinuclear or cytoplasm zone.")
        _cell_l.addWidget(self._cell_stats_table)
        _data_tabs.addTab(_cell_w, "Per-Cell Stats")


        rl.addWidget(_data_tabs, stretch=1)

        root.addWidget(rw)

        # ── Connections ───────────────────────────────────────────────────
        self._run_btn.clicked.connect(self._run)
        self._measure_btn.clicked.connect(self._measure_signal)

    def _on_state_change(self, what: str):
        if what in ("raw_image", "preprocessed", "channel_mappings", "derived_channels"):
            has_nuc  = self._state.has_derived_channel("nucleus_mask")
            has_cell = self._state.has_derived_channel("cell_body_mask")
            self._run_btn.setEnabled(self._state.has_image() and (has_nuc or has_cell))
            # Populate signal metrics combo
            self._metrics_combo.blockSignals(True)
            self._metrics_combo.clear()
            for key in self._state.channels.get("derived", {}):
                self._metrics_combo.addItem(key)
            if self._metrics_combo.count() == 0:
                self._metrics_combo.addItem("— none —")
            self._metrics_combo.blockSignals(False)
            has_regions = self._state.has_derived_channel("region_nucleus")
            self._measure_btn.setEnabled(has_regions)
            # Populate per-channel display combos (preserve current selection)
            _prev1 = self._display_ch1_combo.currentIndex()
            _prev2 = self._display_ch2_combo.currentIndex()
            _populate_channel_combo(self._display_ch1_combo, self._state)
            _populate_channel_combo(self._display_ch2_combo, self._state)
            if _prev1 > 0 and _prev1 < self._display_ch1_combo.count():
                self._display_ch1_combo.setCurrentIndex(_prev1)
            # Default Ch2 combo to second channel if available
            if _prev2 > 0 and _prev2 < self._display_ch2_combo.count():
                self._display_ch2_combo.setCurrentIndex(_prev2)
            elif self._display_ch2_combo.count() > 1:
                self._display_ch2_combo.setCurrentIndex(1)
            # When the aggregates channel is added or updated (e.g. after
            # Aggregate Detection runs) and we already have a region result,
            # automatically re-populate the per-cell inclusion statistics so
            # the user never sees stale zeros from an earlier run.
            if what == "derived_channels" and self._result is not None:
                self._populate_cell_stats(self._result)

        elif what == "active_z_slice" and self._result is not None:
            self._refresh_overlay()

    def _get_masks(self):
        """Resolve nucleus and cell body masks from state.
        active_z_slice slicing so all downstream analysis is consistent."""
        nuc_mask  = None
        cell_mask = None

        nuc_idx = self._nuc_src_combo.currentIndex()
        if nuc_idx == 0:
            nuc_mask = self._state.get_derived_channel("nucleus_mask")
        elif nuc_idx == 1:
            nuc_mask = self._state.get_derived_channel("region_nucleus")

        cell_idx = self._cell_src_combo.currentIndex()
        if cell_idx == 0:
            cell_mask = self._state.get_derived_channel("cell_body_mask")


        # Masks now store label IDs as float values — use them directly as int
        if nuc_mask is not None:
            nuc_ids = self._state.active_cell_ids.get("nucleus", set())
            if nuc_ids:
                nuc_int = nuc_mask.astype(np.int32)
                nuc_mask = self._state.filter_mask_by_active_ids(nuc_int, "nucleus").astype(np.float32)

        if cell_mask is not None:
            cell_ids_active = self._state.active_cell_ids.get("cell_body", set())
            if cell_ids_active:
                cell_int = cell_mask.astype(np.int32)
                cell_mask = self._state.filter_mask_by_active_ids(cell_int, "cell_body").astype(np.float32)


        active_z = self._state.active_z_slice
        if active_z is not None:
            if nuc_mask is not None and nuc_mask.ndim == 3:
                z = min(int(active_z), nuc_mask.shape[0] - 1)
                nuc_mask = nuc_mask[z : z + 1]
            if cell_mask is not None and cell_mask.ndim == 3:
                z = min(int(active_z), cell_mask.shape[0] - 1)
                cell_mask = cell_mask[z : z + 1]

        # Proxy: dilate nucleus if no cell mask
        if cell_mask is None and nuc_mask is not None:
            from scipy.ndimage import binary_dilation as _bd
            nuc_2d = (nuc_mask.max(axis=0) > 0) if nuc_mask.ndim == 3 else nuc_mask > 0
            cell_2d = _bd(nuc_2d, iterations=40)
            cell_mask = cell_2d.astype(np.float32)[np.newaxis]

        return nuc_mask, cell_mask

    def _run(self):
        nuc_mask, cell_mask = self._get_masks()
        if nuc_mask is None:
            QMessageBox.warning(self, "Missing mask",
                "No nucleus mask found.\n"
                "Run nucleus segmentation in 'Cell Body  Nuclei Detec...' tab first.")
            return
        if cell_mask is None:
            QMessageBox.warning(self, "Missing mask",
                "No cell body mask found.\n"
                "Run cell body segmentation first, or enable the nucleus proxy option.")
            return

        self._run_btn.setEnabled(False); self._prog.setVisible(True)
        self._stat_lbl.info("Running cell region analysis…")

        # Pass labeled integer masks so the worker can iterate per-cell.
        # Max-project 3-D stacks to get a single 2-D representative plane.
        def _as_labeled(mask):
            if mask is None:
                return None
            m = mask.max(axis=0) if mask.ndim == 3 else mask
            # If already labeled (max > 1) keep as-is; otherwise re-label
            m = m.astype(np.float32)
            if m.max() > 1:
                return m.astype(np.int32)
            return sk_label(m > 0).astype(np.int32)

        nuc_labeled  = _as_labeled(nuc_mask)
        cell_labeled = _as_labeled(cell_mask)

        self._worker = CellRegionWorker(
            nucleus_mask   = nuc_labeled,
            cell_body_mask = cell_labeled,
            periphery_px   = self._peri_px_sp.value(),
            peri_r_factor  = self._peri_r_sp.value(),
            parent         = self)
        self._worker.signals.progress.connect(self._stat_lbl.info)
        self._worker.signals.result.connect(self._on_result)
        self._worker.signals.error.connect(lambda m: (
            QMessageBox.critical(self, "Region Analysis Error", m),
            self._stat_lbl.err("Failed.")))
        self._worker.signals.finished.connect(lambda: (
            self._run_btn.setEnabled(True), self._prog.setVisible(False)))
        self._worker.start()

    def _on_result(self, r: dict):
        self._result = r

        # Store each region as a derived channel
        region_keys = {
            "region_nucleus":     r["nucleus"],
            "region_perinuclear": r["perinuclear"],
            "region_cytoplasm":   r["cytoplasm"],
            "region_periphery":   r["periphery"],
        }
        for key, mask in region_keys.items():
            self._state.set_derived_channel(key, mask, source="region_tool")

        r_est  = r["r_est"]
        r_factor = r.get("peri_r_factor", 0.5)

        counts = {k: int((r[k] > 0).sum()) for k in self.REGION_LABELS}
        total  = sum(counts.values()) or 1
        lines  = [f"Mean nucleus radius r ≈ {r_est:.1f} px",
                  f"Perinuclear: shape-aware shell  (r-factor = {r_factor})",
                  f"  thickness = {r_factor} × local nucleus radius",
                  f"  ≈ {r_factor * r_est:.1f} px mean  (thicker on long sides)",
                  ""]
        for k, cnt in counts.items():
            lines.append(f"{k:<15}: {cnt:>8,}  ({100*cnt/total:.1f}%)")
        self._sum_lbl.setText("\n".join(lines))

        # Enable signal measurement
        self._measure_btn.setEnabled(True)

        self._refresh_overlay()
        self._stat_lbl.ok(
            f"Done. 4 per-cell region masks  |  "
            f"r≈{r_est:.1f} px  peri_r={r_factor}  "
            f"(≈{r_factor*r_est:.1f} px ring)")

        self._state.set_result("Cell Region Analysis", {
            "dataframe": pd.DataFrame([
                {"region": k, "pixels": counts[k], "pct": 100*counts[k]/total}
                for k in self.REGION_LABELS
            ]),
            "summary": self._sum_lbl.text(),
            "r_est":   r_est,
        })

        # ── Populate per-cell statistics table ────────────────────────────────
        self._populate_cell_stats(r)

    def _refresh_overlay(self, _=None):

        if self._result is None:
            return
        from scipy.ndimage import binary_erosion  as _ber
        from scipy.ndimage import binary_dilation as _bdil

        ax = self._canvas.ax
        ax.clear()
        ax.set_facecolor("#0a0a14")

        # ── Background: per-channel display slots (Ch 1 / Ch 2) ─────────────
        _disp_img = (self._state.preprocessed_image
                     if self._state.has_preprocessed()
                     else self._state.raw_image)
        for _ch_cb, _ch_cbo in (
                (getattr(self, "_display_ch1", None),
                 getattr(self, "_display_ch1_combo", None)),
                (getattr(self, "_display_ch2", None),
                 getattr(self, "_display_ch2_combo", None))):
            if _ch_cb is None or not _ch_cb.isChecked():
                continue
            _ch_idx = (_channel_index_from_combo(_ch_cbo, self._state)
                       if _ch_cbo is not None else 0)
            if _ch_idx is None:
                _ch_idx = 0
            if (_disp_img is not None and _disp_img.ndim >= 4
                    and _ch_idx < _disp_img.shape[1]):
                _vol  = _disp_img[:, _ch_idx].astype(np.float32)
                _base = np.max(_vol, axis=0)
                _rng  = np.ptp(_base)
                if _rng > 0:
                    _base = (_base - _base.min()) / _rng
                ax.imshow(_base, cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest", origin="upper")

        # Display toggle defaults (in case _build_ui not yet finished)
        show_masks    = getattr(self, "_display_masks", None)
        show_outlines = getattr(self, "_display_outlines", None)
        show_ids      = getattr(self, "_display_ids", None)
        show_masks    = True if show_masks    is None else show_masks.isChecked()
        show_outlines = True if show_outlines is None else show_outlines.isChecked()
        show_ids      = False if show_ids    is None else show_ids.isChecked()

        # ── True inter-cell separation: pixel is border if any 4-connected
        # neighbour belongs to a DIFFERENT cell (or background) ──────────────
        struct3      = np.ones((3, 3), dtype=bool)
        cell_sep     = None
        cell_labeled = (self._result or {}).get("cell_labeled")
        # Cache for the hover handler
        self._cell_labeled = cell_labeled

        if cell_labeled is not None and cell_labeled.ndim == 2:
            H_c, W_c = cell_labeled.shape
            in_cell  = cell_labeled > 0
            _sep     = np.zeros((H_c, W_c), dtype=bool)
            # right/left neighbours
            _sep[:, :-1] |= in_cell[:, :-1] & (cell_labeled[:, 1:]  != cell_labeled[:, :-1])
            _sep[:, 1:]  |= in_cell[:, 1:]  & (cell_labeled[:, :-1] != cell_labeled[:, 1:])
            # down/up neighbours
            _sep[:-1, :] |= in_cell[:-1, :] & (cell_labeled[1:, :]  != cell_labeled[:-1, :])
            _sep[1:, :]  |= in_cell[1:, :]  & (cell_labeled[:-1, :] != cell_labeled[1:, :])
            cell_sep = _sep
        else:
            # Fallback when cell_labeled not available (e.g. proxy mode)
            cm3d = self._state.get_derived_channel("cell_body_mask")
            if cm3d is not None:
                cm2d     = cm3d[0] if cm3d.ndim == 3 else cm3d
                cm_bin   = cm2d > 0
                cm_inner = _ber(cm_bin, structure=struct3, border_value=0)
                cell_sep = cm_bin & ~cm_inner

        # ── Region rendering ────────────────────────────────────────────────
        REGION_FILL = {
            "nucleus":     (0.25, 0.50, 1.00, 0.32),
            "perinuclear": (1.00, 0.60, 0.00, 0.32),
            "cytoplasm":   (0.20, 0.90, 0.20, 0.20),   # low alpha for large area
            "periphery":   (0.90, 0.20, 0.90, 0.32),
        }
        REGION_EDGE = {
            "nucleus":     (0.40, 0.70, 1.00, 0.95),
            "perinuclear": (1.00, 0.70, 0.00, 0.95),
            "cytoplasm":   (0.25, 1.00, 0.25, 0.90),
            "periphery":   (0.95, 0.25, 0.95, 0.90),
        }

        for key in self.REGION_LABELS:
            if not show_masks:
                break
            if not self._region_checks[key].isChecked():
                continue
            mask_3d = self._state.get_derived_channel(f"region_{key}")
            if mask_3d is None:
                continue
            mask_2d  = mask_3d[0] if mask_3d.ndim == 3 else mask_3d

            bin_mask = mask_2d > 0
            if not bin_mask.any():
                continue

            # Stamp out border pixels BEFORE labelling so touching-cell regions
            # remain separate connected components
            if cell_sep is not None:
                bin_mask = bin_mask & ~cell_sep

            labeled = sk_label(bin_mask)
            n_obj   = int(labeled.max())
            if n_obj == 0:
                continue

            fill_rgba = np.zeros((*bin_mask.shape, 4), dtype=np.float32)
            edge_rgba = np.zeros((*bin_mask.shape, 4), dtype=np.float32)
            fc = REGION_FILL[key]
            ec = REGION_EDGE[key]

            for obj_id in range(1, n_obj + 1):
                obj   = labeled == obj_id
                inner = _ber(obj, structure=struct3, border_value=0)
                edge  = obj & ~inner
                fill_rgba[obj]  = fc
                edge_rgba[edge] = ec

            ax.imshow(fill_rgba, interpolation="nearest", origin="upper")
            ax.imshow(edge_rgba, interpolation="nearest", origin="upper")

        # ── Burn black separation lines on top (only if Outlines enabled) ───
        if show_outlines and cell_sep is not None and cell_sep.any():
            H2, W2   = cell_sep.shape
            black    = np.zeros((H2, W2, 4), dtype=np.float32)
            sep_wide = _bdil(cell_sep, structure=struct3)
            black[sep_wide] = (0.0, 0.0, 0.0, 0.85)
            ax.imshow(black, interpolation="nearest", origin="upper")

        # ── Cell IDs (only if enabled): label each cell at its centroid ─────
        if show_ids and cell_labeled is not None and cell_labeled.ndim == 2:
            try:
                from skimage.measure import regionprops as _rp_ids
                for prop in _rp_ids(cell_labeled):
                    cy, cx = prop.centroid
                    ax.text(cx, cy, str(prop.label),
                            color="yellow", fontsize=8, weight="bold",
                            ha="center", va="center",
                            path_effects=None)
            except Exception as _ide:
                print(f"[cell-IDs] skipped: {_ide}")

        # ── Aggregate overlay (only if enabled): magenta fill ───────────
        if getattr(self, "_display_aggs", None) is not None and self._display_aggs.isChecked():
            try:
                agg_bin, _ = self._make_agg_bin(self._state)
                if agg_bin is not None and agg_bin.any():
                    Ha, Wa = agg_bin.shape
                    agg_rgba = np.zeros((Ha, Wa, 4), dtype=np.float32)
                    agg_rgba[agg_bin.astype(bool)] = (1.0, 0.0, 0.9, 0.55)
                    ax.imshow(agg_rgba, interpolation="nearest", origin="upper")
            except Exception as _age:
                print(f"[aggregate-overlay] skipped: {_age}")

        ax.set_title("Cell Region Analysis",
                     color="#a0a0d0", fontsize=9, pad=4)
        ax.axis("off")
        self._canvas.draw_idle()

    def _on_hover(self, event):
        """Hover handler: shows per-cell stats for the cell under the cursor."""
        if event.inaxes is None or self._cell_labeled is None:
            self._hover_lbl.setText("Hover a cell to see its stats.")
            return
        if event.xdata is None or event.ydata is None:
            return
        H, W = self._cell_labeled.shape
        x = int(max(0, min(W - 1, event.xdata)))
        y = int(max(0, min(H - 1, event.ydata)))
        cid = int(self._cell_labeled[y, x])
        if cid == 0:
            self._hover_lbl.setText("(background)")
            return
        info = self._cell_stats_rows.get(cid)
        if info:
            self._hover_lbl.setText(info)
            try:
                QToolTip.showText(QCursor.pos(), info.replace("  ", "\n"))
            except Exception:
                pass
        else:
            self._hover_lbl.setText(f"Cell {cid}: (no stats yet)")

    def _measure_signal(self):
        """Compute % signal and object count per region for selected derived channel.
        Object count uses connected-component labelling of the signal channel"""
        if self._result is None:
            return
        key = self._metrics_combo.currentText()
        if not key or key.startswith("—"):
            return
        sig = self._state.get_derived_channel(key)
        if sig is None:
            self._stat_lbl.warn(f"Channel '{key}' not found.")
            return

        # Project to 2D
        sig_2d = sig.max(axis=0) if sig.ndim == 3 else sig

        # Binarise signal: handle binary / label-derived float32 channels
        # (Otsu on an all-1.0 mask returns ~1.0 → empty result; avoid that)
        _pos_vals = sig_2d[sig_2d > 0]
        if _pos_vals.size == 0:
            sig_bin = np.zeros(sig_2d.shape, dtype=bool)
        elif np.unique(_pos_vals).size <= 2 or float(sig_2d.max()) <= 1.0:
            sig_bin = sig_2d > 0
        else:
            try:
                from skimage.filters import threshold_otsu as _otsu
                sig_bin = sig_2d > _otsu(_pos_vals)
            except Exception:
                sig_bin = sig_2d > float(sig_2d.max()) * 0.5

        rows = []
        for region in self.REGION_LABELS:
            m3 = self._state.get_derived_channel(f"region_{region}")
            if m3 is None:
                continue
            m2d = m3[0] if m3.ndim == 3 else m3
            m2d = m2d > 0
            n_px   = int(m2d.sum())
            sig_in = float(sig_2d[m2d].sum()) if n_px > 0 else 0.0

            # Count distinct connected objects of the signal channel inside this region
            sig_in_region = sig_bin & m2d
            if sig_in_region.any():
                from skimage.measure import label as _lbl
                labeled_sig = _lbl(sig_in_region)
                n_objects = int(labeled_sig.max())
            else:
                n_objects = 0

            rows.append({"region": region, "pixels": n_px,
                         "signal_sum": sig_in, "objects": n_objects})

        total_sig = sum(r["signal_sum"] for r in rows) or 1.0
        self._sig_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            pct = 100.0 * row["signal_sum"] / total_sig
            self._sig_table.setItem(i, 0, QTableWidgetItem(row["region"]))
            self._sig_table.setItem(i, 1, QTableWidgetItem(f"{row['pixels']:,}"))
            self._sig_table.setItem(i, 2, QTableWidgetItem(f"{pct:.1f}%"))
            self._sig_table.setItem(i, 3, QTableWidgetItem(str(row["objects"])))
        for i in range(len(rows)):
            for j in range(4):
                item = self._sig_table.item(i, j)
                if item:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

        total_obj = sum(r["objects"] for r in rows)
        self._stat_lbl.ok(
            f"Signal '{key}' measured across {len(rows)} regions. "
            f"Total objects: {total_obj}.")
        # Re-populate per-cell inclusion stats: the user has now chosen a
        # signal channel, so update aggregate counts and NucIncl/CytoIncl.
        if self._result is not None:
            self._populate_cell_stats(self._result)

    # ── helper: build binary aggregate mask from derived channel ──────────
    @staticmethod
    def _make_agg_bin(state):
        """Return (agg_bin, agg_labeled) or (None, None) if no aggregates."""
        agg_vol = state.get_derived_channel("aggregates")
        if agg_vol is None:
            return None, None
        agg_2d = agg_vol.max(axis=0) if agg_vol.ndim == 3 else agg_vol
        # The DL pipeline already writes a binary/labeled mask (float32 of
        # {0,1} or integer labels). Otsu on an all-1.0 mask returns ~1.0 and
        # `> thr` yields an EMPTY mask, zeroing every downstream statistic —
        # so treat effectively-binary / label-derived data as foreground (>0).
        pos = agg_2d[agg_2d > 0]
        if pos.size == 0:
            agg_bin = np.zeros(agg_2d.shape, dtype=bool)
        elif np.unique(pos).size <= 2 or float(agg_2d.max()) <= 1.0:
            agg_bin = agg_2d > 0
        else:
            try:
                from skimage.filters import threshold_otsu as _ot
                agg_bin = agg_2d > _ot(pos)
            except Exception:
                agg_bin = agg_2d > float(agg_2d.max()) * 0.5
        from skimage.measure import label as _slbl
        agg_labeled = _slbl(agg_bin)
        return agg_bin, agg_labeled

    def _populate_cell_stats(self, r: dict):
        """Fill per-cell aggregate statistics table (13 columns)."""
        from skimage.measure import regionprops as _rp

        agg_bin, agg_labeled = self._make_agg_bin(self._state)
        has_agg = agg_bin is not None

        regions = ["nucleus", "perinuclear", "cytoplasm", "periphery"]
        masks_2d = {}
        for reg in regions:
            vol = self._state.get_derived_channel(f"region_{reg}")
            if vol is not None:
                masks_2d[reg] = (vol[0] if vol.ndim == 3 else vol).astype(np.int32)

        # Pull authoritative cell-label image from worker result
        cell_lbl_2d = r.get("cell_labeled") if r else None

        # Collect all cell IDs (from region masks; fallback to cell_labeled)
        all_cell_ids = set()
        for m in masks_2d.values():
            all_cell_ids |= set(np.unique(m)) - {0}
        if not all_cell_ids and cell_lbl_2d is not None:
            all_cell_ids = set(np.unique(cell_lbl_2d)) - {0}
        all_cell_ids = sorted(all_cell_ids)

        # Shape-guard: align agg_labeled to the cell label image shape
        if has_agg and cell_lbl_2d is not None and agg_labeled.shape != cell_lbl_2d.shape:
            try:
                from skimage.transform import resize as _sk_resize
                agg_labeled = _sk_resize(
                    agg_labeled.astype(float), cell_lbl_2d.shape,
                    order=0, preserve_range=True, anti_aliasing=False
                ).astype(np.int32)
                agg_bin = agg_labeled > 0
            except Exception as _re:
                print(f"[cell-stats] resize failed: {_re}")

        # Pre-compute per-aggregate properties once
        agg_props = []
        if has_agg:
            for prop in _rp(agg_labeled):
                agg_props.append({"label": prop.label, "area": prop.area,
                                   "coords": prop.coords})

        # Build a single region-id map: nucleus=1, perinuclear=2, cytoplasm=3, periphery=4
        # Stamp low→high priority so nucleus (1) wins at overlap pixels.
        _cell_ref = (cell_lbl_2d if cell_lbl_2d is not None
                     else (list(masks_2d.values())[0] if masks_2d else None))
        region_id_map = None
        if _cell_ref is not None:
            region_id_map = np.zeros(_cell_ref.shape, dtype=np.int32)
            for _sid, _reg in zip([4, 3, 2, 1],
                                   ["periphery", "cytoplasm", "perinuclear", "nucleus"]):
                _rm = masks_2d.get(_reg)
                if _rm is None:
                    continue
                if _rm.shape != region_id_map.shape:
                    try:
                        from skimage.transform import resize as _sk_resize2
                        _rm = _sk_resize2(_rm.astype(float), region_id_map.shape,
                                          order=0, preserve_range=True,
                                          anti_aliasing=False).astype(np.int32)
                    except Exception:
                        continue
                region_id_map[_rm > 0] = _sid

        own_map = cell_lbl_2d if cell_lbl_2d is not None else _cell_ref

        # Fast cell-first assignment: O(total aggregate pixels)
        # For every aggregate: majority-vote its owning cell from cell_labeled,
        # then sub-classify region via region_id_map; catch-all = cytoplasm.
        _reg_id_to_name = {1: "nucleus", 2: "perinuclear", 3: "cytoplasm", 4: "periphery"}
        per_cell = {int(_cid): {_rn: 0 for _rn in
                    regions + [f"_area_{_rn}" for _rn in regions]}
                    for _cid in all_cell_ids}
        _n_total_agg = len(agg_props)
        _n_assigned  = 0
        _n_dropped   = 0

        if has_agg and own_map is not None:
            for _prop in agg_props:
                _rows = _prop["coords"][:, 0]
                _cols = _prop["coords"][:, 1]
                _H, _W = own_map.shape
                _ok = (_rows >= 0) & (_rows < _H) & (_cols >= 0) & (_cols < _W)
                _rows, _cols = _rows[_ok], _cols[_ok]
                if _rows.size == 0:
                    _n_dropped += 1; continue
                _owners = own_map[_rows, _cols]
                _nz = _owners[_owners != 0]
                if _nz.size == 0:
                    _n_dropped += 1; continue
                _owner = int(np.bincount(_nz).argmax())
                if _owner == 0 or _owner not in per_cell:
                    _n_dropped += 1; continue
                if region_id_map is not None:
                    _Hr, _Wr = region_id_map.shape
                    _rv = (_rows < _Hr) & (_cols < _Wr)
                    _cpx = (_owners[_rv] == _owner)
                    _rvals = region_id_map[_rows[_rv], _cols[_rv]][_cpx]
                    _nzr = _rvals[_rvals != 0]
                    _reg_name = (_reg_id_to_name.get(
                                     int(np.bincount(_nzr).argmax()), "cytoplasm")
                                 if _nzr.size > 0 else "cytoplasm")
                else:
                    _reg_name = "cytoplasm"
                per_cell[_owner][_reg_name]              += 1
                per_cell[_owner][f"_area_{_reg_name}"]   += _prop["area"]
                _n_assigned += 1

        print(f"[cell-stats] aggregates={_n_total_agg}  assigned={_n_assigned}  "
              f"extracellular={_n_dropped}  cells={len(all_cell_ids)}")

        # Totals per cell (needed by summary block that follows)
        n_nuc_inc     = 0
        n_cyto_inc    = 0
        total_agg_counts = []
        total_agg_areas  = []
        _per_cell_rc  = {}
        _per_cell_ra  = {}
        for _cid in all_cell_ids:
            _cid = int(_cid)
            _rac = {_r: per_cell[_cid][_r]            for _r in regions}
            _raa = {_r: per_cell[_cid][f"_area_{_r}"] for _r in regions}
            _per_cell_rc[_cid] = _rac
            _per_cell_ra[_cid] = _raa
            _tot_c = sum(_rac.values())
            _tot_a = sum(_raa.values())
            total_agg_counts.append(_tot_c)
            total_agg_areas.append(_tot_a)
            if _rac["nucleus"] > 0:                         n_nuc_inc  += 1
            if _rac["perinuclear"] + _rac["cytoplasm"] > 0: n_cyto_inc += 1

        # Fill the 13-column table
        self._cell_stats_table.setSortingEnabled(False)
        self._cell_stats_table.setRowCount(len(all_cell_ids))

        for row_i, cid in enumerate(all_cell_ids):
            cid = int(cid)
            rac  = _per_cell_rc[cid]
            total_agg  = total_agg_counts[row_i]
            total_area = total_agg_areas[row_i]
            reg_px = {}
            for reg in regions:
                m = masks_2d.get(reg)
                reg_px[reg] = int((m == cid).sum()) if m is not None else 0
            nuc_inc  = rac["nucleus"] > 0
            cyto_inc = (rac["perinuclear"] + rac["cytoplasm"]) > 0
            row_vals = [
                str(cid),
                str(reg_px["nucleus"]), str(reg_px["perinuclear"]),
                str(reg_px["cytoplasm"]), str(reg_px["periphery"]),
                str(rac["nucleus"]),    str(rac["perinuclear"]),
                str(rac["cytoplasm"]),  str(rac["periphery"]),
                str(total_agg),
                str(total_area),
                "YES" if nuc_inc  else "no",
                "YES" if cyto_inc else "no",
            ]
            for col_i, val in enumerate(row_vals):
                item = QTableWidgetItem(val)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if col_i in (11, 12) and val == "YES":
                    item.setForeground(__import__("PyQt5.QtGui", fromlist=["QColor"]).QColor("#ffff00"))
                self._cell_stats_table.setItem(row_i, col_i, item)

        self._cell_stats_table.setSortingEnabled(True)
        self._export_csv_btn.setEnabled(len(all_cell_ids) > 0)

        # ── Update aggregate summary label (structured, actionable) ────────
        import statistics as _stats
        n_cells = len(all_cell_ids)
        # Build per-cell hover rows — always includes per-region agg counts
        # and inclusion status (NucIncl / CytoIncl) from the cached _per_cell_rc
        # dict so the info is visible even before Aggregate Detection runs.
        self._cell_stats_rows = {}
        for i, cid in enumerate(all_cell_ids):
            _rac     = _per_cell_rc.get(cid, {r: 0 for r in regions})
            _nuc_in  = _rac.get("nucleus", 0) > 0
            _cyto_in = (_rac.get("perinuclear", 0) + _rac.get("cytoplasm", 0)) > 0
            self._cell_stats_rows[cid] = (
                f"Cell {cid}"
                f"  |  Aggs  Nuc={_rac.get('nucleus',0)}"
                f"  Peri={_rac.get('perinuclear',0)}"
                f"  Cyto={_rac.get('cytoplasm',0)}"
                f"  Pe={_rac.get('periphery',0)}"
                f"  |  Total={total_agg_counts[i]} ({total_agg_areas[i]} px²)"
                f"  |  NucIncl={'YES' if _nuc_in else 'no'}"
                f"  CytoIncl={'YES' if _cyto_in else 'no'}"
            )
        if n_cells > 0 and has_agg:
            # Per-cell distribution
            n_any_inc  = sum(1 for c in total_agg_counts if c > 0)
            mean_count = _stats.mean(total_agg_counts)
            std_count  = _stats.pstdev(total_agg_counts) if len(total_agg_counts) > 1 else 0.0
            med_count  = _stats.median(total_agg_counts)
            max_count  = max(total_agg_counts)
            # Detection (size stats over all aggregates, not per cell)
            all_areas  = [p["area"] for p in agg_props]
            n_total    = len(all_areas)
            tot_area   = sum(all_areas)
            mean_size  = _stats.mean(all_areas) if all_areas else 0.0
            std_size   = _stats.pstdev(all_areas) if len(all_areas) > 1 else 0.0
            med_size   = _stats.median(all_areas) if all_areas else 0.0
            # Spatial distribution: sum aggregates assigned per region
            # Sum per-region aggregate counts from the cached _per_cell_rc dict
            spatial = {"nucleus": 0, "perinuclear": 0, "cytoplasm": 0, "periphery": 0}
            for _sc_cid in all_cell_ids:
                for _sr in spatial:
                    spatial[_sr] += _per_cell_rc.get(_sc_cid, {}).get(_sr, 0)
            spatial_total = sum(spatial.values()) or 1
            # Cell coverage
            cell_areas = []
            for i, cid in enumerate(all_cell_ids):
                tot = 0
                for col_idx in (1, 2, 3, 4):  # Nuc/Peri/Cyto/Periph px columns
                    itm = self._cell_stats_table.item(i, col_idx)
                    if itm is not None:
                        try: tot += int(itm.text())
                        except Exception: pass
                cell_areas.append(tot)
            mean_cell_area = _stats.mean(cell_areas) if cell_areas else 0.0
            mean_frac = (
                _stats.mean([(a / c * 100.0) for a, c in zip(total_agg_areas, cell_areas) if c > 0])
                if any(c > 0 for c in cell_areas) else 0.0
            )
            lines = [
                "=== Aggregate analysis summary ===",
                f"Channel source : aggregates  (derived)",
                f"Cells analysed : {n_cells}",
                "",
                "— Detection —",
                f"Total aggregate objects     : {n_total}",
                f"Total aggregate area        : {int(tot_area)}   px²",
                f"Mean aggregate size         : {mean_size:.1f} ± {std_size:.1f}  px²",
                f"Median aggregate size       : {med_size:.1f}        px²",
                "",
                "— Per-cell distribution —",
                f"Cells with ≥1 aggregate     : {n_any_inc}  ({100*n_any_inc/n_cells:.1f}%)",
                f"Cells with nuclear inclusion: {n_nuc_inc}  ({100*n_nuc_inc/n_cells:.1f}%)",
                f"Cells with cyto  inclusion  : {n_cyto_inc}  ({100*n_cyto_inc/n_cells:.1f}%)",
                f"Mean aggregates / cell      : {mean_count:.2f} ± {std_count:.2f}",
                f"Median aggregates / cell    : {int(med_count)}",
                f"Max aggregates in one cell  : {max_count}",
                "",
                "— Spatial distribution (% of all aggregates) —",
                f"  Nucleus      : {100*spatial['nucleus']/spatial_total:5.1f}%",
                f"  Perinuclear  : {100*spatial['perinuclear']/spatial_total:5.1f}%",
                f"  Cytoplasm    : {100*spatial['cytoplasm']/spatial_total:5.1f}%",
                f"  Periphery    : {100*spatial['periphery']/spatial_total:5.1f}%",
                "",
                "— Cell coverage —",
                f"Mean cell area              : {int(mean_cell_area)}  px²",
                f"Mean aggregate-area fraction: {mean_frac:.2f}%  (agg_area / cell_area)",
            ]
            self._agg_sum_lbl.setText("\n".join(lines))
            self._agg_sum_lbl.setStyleSheet(
                "color:#80ff80;font-family:monospace;font-size:11px;")
            self._agg_sum_lbl.setWordWrap(True)
        elif n_cells > 0:
            self._agg_sum_lbl.setText(
                f"Cells analysed: {n_cells}\n"
                "(run Aggregate Detection first for inclusion stats)")
        # Store data for CSV export
        self._cell_csv_data = {
            "cell_ids":    all_cell_ids,
            "table_rows":  self._cell_stats_table.rowCount(),
        }

    def _export_cell_csv(self):
        """Export per-cell aggregate statistics table to CSV."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Per-Cell Data", "cell_aggregate_stats.csv",
            "CSV files (*.csv)")
        if not path:
            return
        tbl = self._cell_stats_table
        headers = [tbl.horizontalHeaderItem(c).text()
                   for c in range(tbl.columnCount())]
        rows = []
        for r in range(tbl.rowCount()):
            row = []
            for c in range(tbl.columnCount()):
                item = tbl.item(r, c)
                row.append(item.text() if item else "")
            rows.append(row)
        import csv as _csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(headers)
            w.writerows(rows)
        QMessageBox.information(self, "Exported",
            f"Per-cell data saved to:\n{path}")


def _coloc_visualize(mhtt_vol, cct1_vol, z, ms, cs, sigma, min_size,
                     mc, cc, show_overlap, use_mip):

    def _safe_norm(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            return (arr - lo) / (hi - lo)
        return np.zeros_like(arr, dtype=np.float32)

    mn = _safe_norm(m_sl)
    cn = _safe_norm(c_sl)

    ms_ = gaussian(mn, sigma=sigma)
    cs_ = gaussian(cn, sigma=sigma)

    try:
        mt = min(1.0, float(threshold_otsu(ms_)) * ms)
    except Exception:
        mt = ms * 0.5

    try:
        ct = min(1.0, float(threshold_otsu(cs_)) * cs)
    except Exception:
        ct = cs * 0.5

    mm  = remove_small_objects(ms_ > mt, min_size=max(1, int(min_size)))
    cm_ = remove_small_objects(cs_ > ct, min_size=max(1, int(min_size)))
    ov  = mm & cm_
    _, n_ov = scipy_label(ov)

    composite = np.zeros((*mn.shape, 3), dtype=np.float32)
    if not show_overlap:
        for src, msk in [(mc, mm), (cc, cm_)]:
            ch   = color_map[src]
            arr_ = mn if src == mc else cn
            if isinstance(ch, int):
                composite[:, :, ch] = np.clip(arr_ * msk.astype(np.float32), 0, 1)
            else:
                for ch_ in ch:
                    composite[:, :, ch_] = np.clip(arr_ * msk.astype(np.float32), 0, 1)
    else:
        if mc in ('red', 'green', 'blue') and cc in ('red', 'green', 'blue'):
            composite[:, :, color_map[mc]] = np.clip(mn * ov, 0, 1)
            composite[:, :, color_map[cc]] = np.clip(cn * ov, 0, 1)

    overlay = np.zeros((*mn.shape, 3), dtype=np.float32)
    for src, arr_ in [(mc, mn), (cc, cn)]:
        ch = color_map[src]
        if isinstance(ch, int):
            overlay[:, :, ch] += arr_
        else:
            for ch_ in ch:
                overlay[:, :, ch_] += arr_
    overlay = np.clip(overlay, 0, 1)

    sum_mm  = max(1, int(np.sum(mm)))
    sum_cm  = max(1, int(np.sum(cm_)))
    sum_ov  = int(np.sum(ov))

    return dict(
        title=title, mn=mn, cn=cn, mt=mt, ct=ct,
        mm=mm, cm_=cm_, ov=ov,
        composite=composite, overlay=overlay,
        mhtt_area=sum_mm, cct1_area=sum_cm,
        overlap_area=sum_ov,
        n_ov=n_ov if show_overlap else 0,
        overlap_pct_m=100.0 * sum_ov / sum_mm,
        overlap_pct_c=100.0 * sum_ov / sum_cm,
    )


def _coloc_visualize(mhtt_vol, cct1_vol, z, ms, cs, sigma, min_size,
                     mc, cc, show_overlap, use_mip):
    color_map = {'red': 0, 'green': 1, 'blue': 2,
                 'magenta': [0, 2], 'yellow': [0, 1], 'cyan': [1, 2]}

    if use_mip:
        m_sl  = np.max(mhtt_vol, axis=0)
        c_sl  = np.max(cct1_vol, axis=0)
        title = "Maximum Projection"
    else:
        z_idx = max(0, min(int(z), mhtt_vol.shape[0] - 1))
        m_sl  = mhtt_vol[z_idx]
        c_sl  = cct1_vol[z_idx]
        title = f"Z-Slice {z_idx}"

    def _safe_norm(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            return (arr - lo) / (hi - lo)
        return np.zeros_like(arr, dtype=np.float32)

    mn = _safe_norm(m_sl)
    cn = _safe_norm(c_sl)

    ms_ = gaussian(mn, sigma=sigma)
    cs_ = gaussian(cn, sigma=sigma)

    try:
        mt = min(1.0, float(threshold_otsu(ms_)) * ms)
    except Exception:
        mt = ms * 0.5

    try:
        ct = min(1.0, float(threshold_otsu(cs_)) * cs)
    except Exception:
        ct = cs * 0.5

    mm  = remove_small_objects(ms_ > mt, min_size=max(1, int(min_size)))
    cm_ = remove_small_objects(cs_ > ct, min_size=max(1, int(min_size)))
    ov  = mm & cm_
    _, n_ov = scipy_label(ov)

    composite = np.zeros((*mn.shape, 3), dtype=np.float32)
    if not show_overlap:
        for src, msk in [(mc, mm), (cc, cm_)]:
            ch   = color_map[src]
            arr_ = mn if src == mc else cn
            if isinstance(ch, int):
                composite[:, :, ch] = np.clip(arr_ * msk.astype(np.float32), 0, 1)
            else:
                for ch_ in ch:
                    composite[:, :, ch_] = np.clip(arr_ * msk.astype(np.float32), 0, 1)
    else:
        if mc in ('red', 'green', 'blue') and cc in ('red', 'green', 'blue'):
            composite[:, :, color_map[mc]] = np.clip(mn * ov, 0, 1)
            composite[:, :, color_map[cc]] = np.clip(cn * ov, 0, 1)

    overlay = np.zeros((*mn.shape, 3), dtype=np.float32)
    for src, arr_ in [(mc, mn), (cc, cn)]:
        ch = color_map[src]
        if isinstance(ch, int):
            overlay[:, :, ch] += arr_
        else:
            for ch_ in ch:
                overlay[:, :, ch_] += arr_
    overlay = np.clip(overlay, 0, 1)

    sum_mm  = max(1, int(np.sum(mm)))
    sum_cm  = max(1, int(np.sum(cm_)))
    sum_ov  = int(np.sum(ov))

    return dict(
        title=title, mn=mn, cn=cn, mt=mt, ct=ct,
        mm=mm, cm_=cm_, ov=ov,
        composite=composite, overlay=overlay,
        mhtt_area=sum_mm, cct1_area=sum_cm,
        overlap_area=sum_ov,
        n_ov=n_ov if show_overlap else 0,
        overlap_pct_m=100.0 * sum_ov / sum_mm,
        overlap_pct_c=100.0 * sum_ov / sum_cm,
    )

class ColocWorker(BaseWorker):
    def __init__(self, md, cd, **kw):
        super().__init__()
        self._md=md; self._cd=cd; self._kw=kw
    def run_task(self):
        return _coloc_visualize(self._md, self._cd, **self._kw)

class ColocAnalysisWorker(BaseWorker):
    def __init__(self, vis, md, cd, ms, cs, sigma, min_size, mc, cc, mp, cp, parent=None):
        super().__init__(parent)
        self._vis=vis; self._md=md; self._cd=cd
        self._ms=ms; self._cs=cs; self._sigma=sigma; self._min=min_size
        self._mc=mc; self._cc=cc; self._mp=mp; self._cp=cp

    def run_task(self):
        mm=self._vis["mm"]; cm_=self._vis["cm_"]; ov=self._vis["ov"]
        mn=self._vis["mn"]; cn=self._vis["cn"]
        union = mm|cm_
        mv=mn[union]; cv=cn[union]
        pr,pp = pearsonr(mv,cv) if len(mv)>1 else (0,1)
        m1 = np.sum(mn*ov)/max(1e-10,np.sum(mn*mm))
        m2 = np.sum(cn*ov)/max(1e-10,np.sum(cn*cm_))
        dice = 2*np.sum(ov)/max(1,np.sum(mm)+np.sum(cm_))
        mi = mn[mm&~cm_].flatten(); mo = mn[ov].flatten()
        if len(mi)>0 and len(mo)>0:
            u_stat,p_val = mannwhitneyu(mi,mo,alternative='two-sided')
        else:
            u_stat,p_val = 0,1
        if len(mi)>0 and len(mo)>0:
            mean_in=np.mean(mo); mean_out=np.mean(mi)
            ni=len(mo); no=len(mi)
            pool = np.sqrt(((ni-1)*np.std(mo)**2+(no-1)*np.std(mi)**2)/max(1,ni+no-2))
            cohen_d = abs(mean_in-mean_out)/max(1e-10,pool)
        else:
            mean_in=mean_out=cohen_d=0
        iou = np.sum(ov)/max(1,np.sum(union))

        # Results are returned to memory only — no disk I/O here.
        # Call save_coloc_pdf() separately when the user explicitly requests a save.
        return dict(pr=pr,pp=pp,m1=m1,m2=m2,dice=dice,
                    u_stat=u_stat,p_val=p_val,cohen_d=cohen_d,
                    mean_in=mean_in,mean_out=mean_out,iou=iou,
                    vis=self._vis, ms=self._ms, cs=self._cs,
                    sigma=self._sigma, min_size=self._min,
                    mc=self._mc, cc=self._cc,
                    mp=self._mp, cp=self._cp)

def save_coloc_pdf(r: dict) -> str:
    """Write a colocalization PDF report to disk from in-memory results dict.
    This is the ONLY place that writes colocalization data to disk."""
    vis      = r["vis"]
    mn, cn   = vis["mn"], vis["cn"]
    mm, cm_  = vis["mm"], vis["cm_"]
    ov       = vis["ov"]
    mc, cc   = r["mc"], r["cc"]
    mp, cp   = r["mp"], r["cp"]

    report_dir = "colocalization_reports"
    os.makedirs(report_dir, exist_ok=True)
    ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_fn = os.path.join(report_dir, f"colocalization_report_{ts}.pdf")

    color_map = {'red':0,'green':1,'blue':2,'magenta':[0,2],'yellow':[0,1],'cyan':[1,2]}
    fig_r, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(mn, cmap='gray'); axs[0].contour(mm, colors='green', linewidths=0.5)
    axs[0].set_title(f"Ch-A {vis['title']}"); axs[0].axis('off')
    axs[1].imshow(cn, cmap='gray'); axs[1].contour(cm_, colors='red', linewidths=0.5)
    axs[1].set_title(f"Ch-B {vis['title']}"); axs[1].axis('off')
    oi = np.zeros((*mn.shape, 3))
    for src, arr_, msk in [(mc, mn, ov), (cc, cn, ov)]:
        ch = color_map[src]
        if isinstance(ch, int): oi[:, :, ch] = arr_ * msk
        else:
            for c_ in ch: oi[:, :, c_] = arr_ * msk
    axs[2].imshow(np.clip(oi, 0, 1)); axs[2].set_title("Colocalization Map"); axs[2].axis('off')
    plt.tight_layout()

    with PdfPages(pdf_fn) as pdf:
        ft = plt.figure(figsize=(8.5, 11))
        ft.suptitle("Colocalization Analysis Report", fontsize=16, y=0.95)
        plt.figtext(0.5, 0.85, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ha='center', fontsize=12)
        plt.figtext(0.5, 0.80, f"Channel A: {os.path.basename(mp)}", ha='center', fontsize=10)
        plt.figtext(0.5, 0.77, f"Channel B: {os.path.basename(cp)}", ha='center', fontsize=10)
        plt.figtext(0.5, 0.72, f"Mode: {vis['title']}", ha='center', fontsize=12, weight='bold')
        plt.figtext(0.5, 0.67, "Parameters:", ha='center', fontsize=12, weight='bold')
        plt.figtext(0.5, 0.63, f"mHTT factor={r['ms']}, CCT1 factor={r['cs']}", ha='center', fontsize=10)
        plt.figtext(0.5, 0.60, f"sigma={r['sigma']}, min_size={r['min_size']}", ha='center', fontsize=10)
        plt.axis('off'); pdf.savefig(ft); plt.close(ft)
        fs = plt.figure(figsize=(8.5, 11))
        fs.suptitle("Colocalization Metrics Summary", fontsize=16, y=0.95)
        plt.figtext(0.1, 0.90, "Basic Metrics:", fontsize=12, weight='bold')
        plt.figtext(0.1, 0.87, f"Ch-A ({mp}) area: {np.sum(mm)} pixels", fontsize=10)
        plt.figtext(0.1, 0.84, f"Ch-B ({cp}) area: {np.sum(cm_)} pixels", fontsize=10)
        plt.figtext(0.1, 0.81, f"Overlap area: {np.sum(ov)} pixels", fontsize=10)
        plt.figtext(0.1, 0.78, f"Overlap: {100*np.sum(ov)/max(1,np.sum(mm)):.2f}% of mHTT, {100*np.sum(ov)/max(1,np.sum(cm_)):.2f}% of CCT1", fontsize=10)
        plt.figtext(0.1, 0.73, "1. Pearson Correlation:", fontsize=12, weight='bold')
        plt.figtext(0.1, 0.70, f"   r = {r['pr']:.4f}  (p = {r['pp']:.2e})", fontsize=10)
        plt.figtext(0.1, 0.65, "2. Manders Coefficients:", fontsize=12, weight='bold')
        plt.figtext(0.1, 0.62, f"   M1 = {r['m1']:.4f}", fontsize=10)
        plt.figtext(0.1, 0.59, f"   M2 = {r['m2']:.4f}", fontsize=10)
        plt.figtext(0.1, 0.54, "3. Dice Similarity:", fontsize=12, weight='bold')
        plt.figtext(0.1, 0.51, f"   Dice = {r['dice']:.4f}", fontsize=10)
        plt.figtext(0.1, 0.46, "4. Mann-Whitney U:", fontsize=12, weight='bold')
        plt.figtext(0.1, 0.43, f"   U = {r['u_stat']:.2f}  p = {r['p_val']:.2e}", fontsize=10)
        plt.figtext(0.1, 0.35, "5. Cohen's d:", fontsize=12, weight='bold')
        plt.figtext(0.1, 0.32, f"   d = {r['cohen_d']:.4f}", fontsize=10)
        plt.figtext(0.1, 0.21, "6. IoU:", fontsize=12, weight='bold')
        plt.figtext(0.1, 0.18, f"   IoU = {r['iou']:.4f}", fontsize=10)
        plt.figtext(0.5, 0.05, f"Saved: {pdf_fn}", ha='center', fontsize=8, style='italic')
        plt.axis('off'); pdf.savefig(fs); plt.close(fs)
        pdf.savefig(fig_r); plt.close(fig_r)

    return pdf_fn


# --- Colocalization Tab ---
# Pearson and Manders coefficients for any two channels
class ColocalizationTab(QWidget):

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self._state      = state
        self._mhtt_data  = None   # np.ndarray (Z,Y,X)
        self._cct1_data  = None   # np.ndarray (Z,Y,X)
        self._mhtt_label = ""
        self._cct1_label = ""
        self._vis_result = None
        self._ana_result = None
        self._vis_worker = None
        self._ana_worker = None
        self._build_ui()
        state.subscribe(self._on_state_change)

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        ls, _, ll = make_scroll_widget()
        ls.setFixedWidth(315)

        # ----
        csel = section("Channel Selection  (by biological label / derived channel)")
        csel_hdr = QHBoxLayout()
        csel_hdr.addStretch()
        csel_hdr.addWidget(HelpButton("Colocalization"))
        csl  = QVBoxLayout(csel)
        csl.addLayout(csel_hdr)

        self._mapping_info = QLabel(
            "Channels are listed by biological label.\n"
            "Derived channels (Aggregates) appear after\n"
            "running the respective detection tabs.")
        self._mapping_info.setWordWrap(True)
        self._mapping_info.setStyleSheet("color:#707090;font-size:11px;")
        csl.addWidget(self._mapping_info)

        row_a = QHBoxLayout()
        row_a.addWidget(QLabel("Channel A:"))
        self._ch_a_combo = QComboBox()

        self._ch_a_combo.setToolTip(
            "First channel for colocalization analysis.\n\n"
            "Channels are identified by biological label (set in Image Viewer).\n"
            "Derived channels (e.g. Aggregates) are also available after detection.\n"
            "Custom labels (set via 'Other') appear here with their custom name.\n\n"
            "Typical use: select the protein of interest (e.g. HA-mHTT, A11).")
        row_a.addWidget(self._ch_a_combo)
        csl.addLayout(row_a)

        row_b = QHBoxLayout()
        row_b.addWidget(QLabel("Channel B:"))
        self._ch_b_combo = QComboBox()
        self._ch_b_combo.setToolTip(
            "Second channel for colocalization analysis.\n\n"
            "Should differ from Channel A.\n"
            "Typical use: select the chaperone or marker to compare against\n"
            "Channel A (e.g. CCT1, nucleus stain, or a derived Aggregates channel).")
        row_b.addWidget(self._ch_b_combo)
        csl.addLayout(row_b)

        self._sel_status = StatusLabel()
        self._sel_status.setText("Configure mappings in Image Viewer →")
        csl.addWidget(self._sel_status)

        self._refresh_ch_btn = QPushButton("↺  Refresh from Image Viewer")
        self._refresh_ch_btn.setToolTip(
            "Reload the channel list from Image Viewer channel mappings.\n\n"
            "Use this after adding or renaming channels, or after running\n"
            "detection tabs that add derived channels.")
        self._refresh_ch_btn.clicked.connect(self._populate_channel_combos)
        csl.addWidget(self._refresh_ch_btn)

        ll.addWidget(csel)

        pg = section("Analysis Parameters ")
        pl = QVBoxLayout(pg)
        # Z-Slice: integer (max set at runtime when an image is loaded)
        self._z_sl = QSpinBox()
        self._z_sl.setRange(0, 0)
        self._z_sl.setSingleStep(1)
        self._z_sl.setValue(0)
        self._z_sl.setToolTip(
            "Z-plane for single-slice colocalization visualisation.\n\n"
            "Ignored when 'Use maximum intensity projection' is checked.")

        # Ch-A intensity scaling factor
        self._ms_sl = QDoubleSpinBox()
        self._ms_sl.setRange(1.0, 10.0); self._ms_sl.setSingleStep(0.1)
        self._ms_sl.setDecimals(1);      self._ms_sl.setValue(4.0)
        self._ms_sl.setToolTip(
            "Intensity scaling factor for Channel A in the composite overlay.\n\n"
            "Increase to brighten Channel A; does not affect quantitative analysis.")

        # Ch-B intensity scaling factor
        self._cs_sl = QDoubleSpinBox()
        self._cs_sl.setRange(1.0, 10.0); self._cs_sl.setSingleStep(0.1)
        self._cs_sl.setDecimals(1);      self._cs_sl.setValue(4.5)
        self._cs_sl.setToolTip(
            "Intensity scaling factor for Channel B in the composite overlay.\n\n"
            "Increase to brighten Channel B; does not affect quantitative analysis.")

        # Gaussian sigma applied before analysis
        self._sig_sl = QDoubleSpinBox()
        self._sig_sl.setRange(0.1, 5.0); self._sig_sl.setSingleStep(0.1)
        self._sig_sl.setDecimals(1);     self._sig_sl.setValue(3.5)
        self._sig_sl.setToolTip(
            "Gaussian smoothing σ applied before colocalization analysis (pixels).\n\n"
            "Smoothing reduces noise before computing overlap.\n"
            "Increase for noisy images; decrease to preserve fine puncta.")

        # Minimum object size in px
        self._msz_sl = QSpinBox()
        self._msz_sl.setRange(1, 100); self._msz_sl.setSingleStep(1)
        self._msz_sl.setValue(20)
        self._msz_sl.setToolTip(
            "Minimum object size (pixels) for colocalization detection.\n\n"
            "Objects smaller than this are excluded from the colocalizing spot count.\n"
            "Increase to ignore small noise; decrease to capture tiny puncta.")

        cr = QWidget()
        crl = QHBoxLayout(cr)
        crl.setContentsMargins(0, 0, 0, 0)
        crl.addWidget(QLabel("Ch-A color:"))
        self._mc_combo = QComboBox()
        self._mc_combo.addItems(['red','green','blue','magenta','yellow','cyan'])
        self._mc_combo.setCurrentText('green')
        self._mc_combo.setToolTip("Display colour for Channel A in the colocalization overlay.")
        crl.addWidget(self._mc_combo)
        crl.addWidget(QLabel("Ch-B color:"))
        self._cc_combo = QComboBox()
        self._cc_combo.addItems(['red','green','blue','magenta','yellow','cyan'])
        self._cc_combo.setCurrentText('red')
        self._cc_combo.setToolTip("Display colour for Channel B in the colocalization overlay.")
        crl.addWidget(self._cc_combo)

        self._ov_cb  = QCheckBox("Show overlap only")
        self._ov_cb.setChecked(False)
        self._ov_cb.setToolTip(
            "If checked, only pixels where both channels exceed threshold are shown.\n"
            "Useful to highlight colocalizing regions exclusively.")
        self._mip_cb = QCheckBox("Use maximum intensity projection")
        self._mip_cb.setChecked(True)
        self._mip_cb.setToolTip(
            "Project the Z-stack using maximum intensity before analysis.\n\n"
            "Recommended for 3-D stacks to capture all focal planes.\n"
            "Uncheck to analyse a single Z-plane (set with Z-Slice slider above).")

        pl.addLayout(_make_spinbox_row("Z-Slice:",     self._z_sl))
        pl.addLayout(_make_spinbox_row("Ch-A Factor:", self._ms_sl))
        pl.addLayout(_make_spinbox_row("Ch-B Factor:", self._cs_sl))
        pl.addLayout(_make_spinbox_row("Gaussian σ:",  self._sig_sl))
        pl.addLayout(_make_spinbox_row("Min Size:",    self._msz_sl))
        for w in [cr, self._ov_cb, self._mip_cb]:
            pl.addWidget(w)
        ll.addWidget(pg)

        bg = section("Actions")
        bl = QVBoxLayout(bg)
        self._ana_btn = QPushButton("▶  Run Colocalization Analysis")
        self._ana_btn.setStyleSheet(BTN_RUN)
        self._ana_btn.setEnabled(False)
        bl.addWidget(self._ana_btn)
        self._save_btn = QPushButton("💾  Save Colocalization Report")
        self._save_btn.setStyleSheet(BTN_PURPLE)
        self._save_btn.setEnabled(False)
        self._save_btn.setToolTip("Save the current analysis results as a PDF report to disk.")
        bl.addWidget(self._save_btn)
        ll.addWidget(bg)

        self._prog     = QProgressBar()
        self._prog.setRange(0, 0)
        self._prog.setVisible(False)
        self._stat_lbl = StatusLabel()
        ll.addWidget(self._prog)
        ll.addWidget(self._stat_lbl)
        ll.addStretch()
        root.addWidget(ls)

        cw = QWidget()
        cl = QVBoxLayout(cw)
        cl.setContentsMargins(0, 0, 0, 0)
        self._canvas1 = ImageCanvas(figsize=(7, 3))   # composite
        self._canvas2 = ImageCanvas(figsize=(7, 3))   # raw overlay
        self._nav1 = NavigationToolbar(self._canvas1, self)
        self._nav2 = NavigationToolbar(self._canvas2, self)
        for nav in [self._nav1, self._nav2]:
            nav.setStyleSheet(_viewer_tb_qss())
        cl.addWidget(self._nav1)
        cl.addWidget(self._canvas1, stretch=1)
        cl.addWidget(self._nav2)
        cl.addWidget(self._canvas2, stretch=1)
        root.addWidget(cw, stretch=1)

        rw = QWidget()
        rw.setFixedWidth(305)
        rl = QVBoxLayout(rw)
        og = section("Output")
        ol = QVBoxLayout(og)
        self._out_txt = QTextEdit()
        self._out_txt.setReadOnly(True)
        ol.addWidget(self._out_txt)
        rl.addWidget(og, stretch=1)
        root.addWidget(rw)

        self._ana_btn.clicked.connect(self._run_full_analysis)
        self._save_btn.clicked.connect(self._save_coloc_report)
        for w in [self._ms_sl, self._cs_sl, self._sig_sl, self._msz_sl]:
            w.valueChanged.connect(self._on_param_change)
        self._z_sl.valueChanged.connect(self._on_param_change)
        for cb in [self._ov_cb, self._mip_cb]:
            cb.stateChanged.connect(self._on_param_change)
        for combo in [self._mc_combo, self._cc_combo]:
            combo.currentTextChanged.connect(self._on_param_change)
        self._mip_cb.stateChanged.connect(
            lambda s: self._z_sl.setEnabled(not bool(s)))
        self._ch_a_combo.currentIndexChanged.connect(self._extract_channel_data)
        self._ch_b_combo.currentIndexChanged.connect(self._extract_channel_data)

    def _populate_channel_combos(self) -> None:
        # ----
        # Build label list from both raw channels (biological_label) AND derived channels.
        # This allows selecting "Aggregates" vs "Chaperones" directly.
        all_labels = self._state.get_all_channel_labels()   # [(display, kind), …]
        # Require at least 2 selectable sources
        if len(all_labels) < 2:
            msg = (f"{len(all_labels)} selectable channel(s) — need ≥ 2.\n"
                   "Enable channels in Image Viewer, or run Aggregate / Chaperone detection.")
            self._mapping_info.setText(msg)
            self._sel_status.warn("Need ≥ 2 channels (raw or derived).")
            self._ana_btn.setEnabled(False)
            return

        # Preserve previous selections by display-name
        prev_a = self._ch_a_combo.currentText()
        prev_b = self._ch_b_combo.currentText()

        self._ch_a_combo.blockSignals(True)
        self._ch_b_combo.blockSignals(True)
        self._ch_a_combo.clear()
        self._ch_b_combo.clear()

        for display, kind in all_labels:
            tag = "  [derived]" if kind == "derived" else ""
            self._ch_a_combo.addItem(f"{display}{tag}")
            self._ch_b_combo.addItem(f"{display}{tag}")

        # Restore prior selection or pick sensible defaults
        def _find(target: str) -> int:
            for i in range(self._ch_a_combo.count()):
                if target and target in self._ch_a_combo.itemText(i):
                    return i
            return 0

        idx_a = _find(prev_a) if prev_a else _find("Aggregates")
        idx_b = _find(prev_b) if prev_b else _find("Chaperones")
        if idx_a == idx_b:
            idx_b = 1 if idx_a == 0 else 0

        n = self._ch_a_combo.count()
        self._ch_a_combo.setCurrentIndex(min(idx_a, n - 1))
        self._ch_b_combo.setCurrentIndex(min(idx_b, n - 1))
        self._ch_a_combo.blockSignals(False)
        self._ch_b_combo.blockSignals(False)

        n_raw     = len(self._state.channel_mappings)
        n_derived = len(self._state.channels["derived"])
        self._mapping_info.setText(
            f"{n_raw} raw channel(s)  +  {n_derived} derived channel(s) available.\n"
            f"Select Channel A and B to compare.")
        self._sel_status.ok(f"{n} total channels available.")

        self._extract_channel_data()

    def _extract_channel_data(self, _=None) -> None:
        # ----
        # Resolve channel volumes by biological label rather than raw index.
        # Derived channels (aggregates, chaperones) are supported here.
        all_labels = self._state.get_all_channel_labels()   # [(display, kind), …]

        if len(all_labels) < 2:
            self._ana_btn.setEnabled(False)
            return

        idx_a = self._ch_a_combo.currentIndex()
        idx_b = self._ch_b_combo.currentIndex()

        if idx_a < 0 or idx_b < 0 or idx_a >= len(all_labels) or idx_b >= len(all_labels):
            self._ana_btn.setEnabled(False)
            return

        if idx_a == idx_b:
            self._sel_status.warn("Channel A and B must be different.")
            self._ana_btn.setEnabled(False)
            return

        label_a, kind_a = all_labels[idx_a]
        label_b, kind_b = all_labels[idx_b]

        # Resolve the actual (Z, Y, X) volumes using the unified resolver
        vol_a = self._state.get_channel_volume_by_label(label_a)
        vol_b = self._state.get_channel_volume_by_label(label_b)

        if vol_a is None or vol_b is None:
            self._sel_status.warn(
                f"Could not resolve volume for "
                f"{'A' if vol_a is None else 'B'}.  "
                f"Run the relevant detection tab first.")
            self._ana_btn.setEnabled(False)
            return

        # Ensure both volumes have the same spatial shape by broadcasting if needed
        if vol_a.shape != vol_b.shape:
            # Trim to the smaller of the two on every axis
            nz = min(vol_a.shape[0], vol_b.shape[0])
            ny = min(vol_a.shape[1], vol_b.shape[1])
            nx = min(vol_a.shape[2], vol_b.shape[2])
            vol_a = vol_a[:nz, :ny, :nx]
            vol_b = vol_b[:nz, :ny, :nx]

        self._mhtt_data  = vol_a
        self._cct1_data  = vol_b
        self._mhtt_label = f"{label_a}  [{kind_a}]"
        self._cct1_label = f"{label_b}  [{kind_b}]"


        active_z = self._state.active_z_slice
        if active_z is not None:
            z_a = min(int(active_z), self._mhtt_data.shape[0] - 1)
            z_b = min(int(active_z), self._cct1_data.shape[0] - 1)
            self._mhtt_data = self._mhtt_data[z_a : z_a + 1]
            self._cct1_data = self._cct1_data[z_b : z_b + 1]

        nz = self._mhtt_data.shape[0]
        self._z_sl.blockSignals(True)
        self._z_sl.setMaximum(max(0, nz - 1))
        self._z_sl.blockSignals(False)

        self._sel_status.ok(
            f"A = {self._mhtt_label}   B = {self._cct1_label}   ({nz} Z-slices)")
        self._ana_btn.setEnabled(True)

        self._on_param_change()

    def _on_state_change(self, what: str) -> None:
        # ----
        # Also refresh when a new derived channel is registered (e.g. after
        # Aggregate Detection or Chaperone Segmentation completes).
        if what in ("channel_mappings", "preprocessed", "raw_image",
                    "derived_channels"):
            self._populate_channel_combos()

        elif what == "active_z_slice":
            self._extract_channel_data()

    def _params(self) -> dict:
        return dict(
            z          = int(self._z_sl.value()),
            ms         = self._ms_sl.value(),
            cs         = self._cs_sl.value(),
            sigma      = self._sig_sl.value(),
            min_size   = int(self._msz_sl.value()),
            mc         = self._mc_combo.currentText(),
            cc         = self._cc_combo.currentText(),
            show_overlap = self._ov_cb.isChecked(),
            use_mip    = self._mip_cb.isChecked(),
        )

    def _on_param_change(self, _=None) -> None:
        if self._mhtt_data is None or self._cct1_data is None:
            return

        if self._vis_worker and self._vis_worker.isRunning():
            self._vis_worker.signals.result.disconnect()
            self._vis_worker.quit()
            self._vis_worker.wait(200)

        p = self._params()
        try:
            r = _coloc_visualize(
                self._mhtt_data, self._cct1_data,
                z=p["z"], ms=p["ms"], cs=p["cs"],
                sigma=p["sigma"], min_size=p["min_size"],
                mc=p["mc"], cc=p["cc"],
                show_overlap=p["show_overlap"],
                use_mip=p["use_mip"])
            self._on_vis_done(r)
        except Exception as exc:
            self._sel_status.warn(f"Preview error: {exc}")

    def _run_synchronous_preview(self) -> bool:
        if self._mhtt_data is None or self._cct1_data is None:
            return False
        p = self._params()
        try:
            r = _coloc_visualize(
                self._mhtt_data, self._cct1_data,
                z=p["z"], ms=p["ms"], cs=p["cs"],
                sigma=p["sigma"], min_size=p["min_size"],
                mc=p["mc"], cc=p["cc"],
                show_overlap=p["show_overlap"],
                use_mip=p["use_mip"])
            self._on_vis_done(r)
            return True
        except Exception as exc:
            self._sel_status.err(f"Preview failed: {exc}")
            return False

    def _on_vis_done(self, r) -> None:
        self._vis_result = r
        ax1 = self._canvas1.ax
        ax1.clear()
        ax1.set_facecolor("#0a0a14")
        ax1.imshow(r["composite"])
        ch_a = getattr(self, "_mhtt_label", "Channel A")
        ch_b = getattr(self, "_cct1_label", "Channel B")
        ax1.set_title(
            f"{r['title']}  [{ch_a}] thr>{r['mt']:.2f}  [{ch_b}] thr>{r['ct']:.2f}",
            color="#a0a0d0", fontsize=9)
        ax1.axis("off")
        self._canvas1.draw_idle()

        ax2 = self._canvas2.ax
        ax2.clear()
        ax2.set_facecolor("#0a0a14")
        ax2.imshow(r["overlay"])
        ax2.set_title(f"{r['title']} – Raw Overlay",
                      color="#a0a0d0", fontsize=9)
        ax2.axis("off")
        self._canvas2.draw_idle()

        txt = self._out_txt
        txt.clear()
        txt.append(f"Channel A ({self._mhtt_label}) area: {r['mhtt_area']} pixels")
        txt.append(f"Channel B ({self._cct1_label}) area: {r['cct1_area']} pixels")
        txt.append(f"Overlap area: {r['overlap_area']} pixels")
        txt.append(f"Overlap: {r['overlap_pct_m']:.2f}% of A, "
                   f"{r['overlap_pct_c']:.2f}% of B")
        txt.append(f"Colocalized aggregates: {r['n_ov']}")

    def _run_full_analysis(self) -> None:
        if self._mhtt_data is None or self._cct1_data is None:
            QMessageBox.warning(self, "No data",
                                "Select two channels in the Image Viewer first.")
            return

        if self._vis_result is None:
            self._stat_lbl.info("Computing preview…")
            QApplication.processEvents()
            if not self._run_synchronous_preview():
                return  # error already shown

        p = self._params()
        self._ana_btn.setEnabled(False)
        self._prog.setVisible(True)
        self._stat_lbl.info("Running full analysis + PDF…")
        src_path = self._state.file_path or "preprocessed_image"
        self._ana_worker = ColocAnalysisWorker(
            self._vis_result,
            self._mhtt_data, self._cct1_data,
            p["ms"], p["cs"], p["sigma"], p["min_size"],
            p["mc"], p["cc"],
            f"{src_path}  [{self._mhtt_label}]",
            f"{src_path}  [{self._cct1_label}]",
            parent=self)
        self._ana_worker.signals.result.connect(self._on_ana_done)
        self._ana_worker.signals.error.connect(
            lambda m: (QMessageBox.critical(self, "Analysis Error", m),
                       self._stat_lbl.err("Failed.")))
        self._ana_worker.signals.finished.connect(lambda: (
            self._ana_btn.setEnabled(True),
            self._prog.setVisible(False),
            self._stat_lbl.ok("Analysis complete.")))
        self._ana_worker.start()

    def _on_ana_done(self, r) -> None:
        self._ana_result = r
        p   = self._params()
        vis = self._vis_result
        txt = self._out_txt
        txt.append(f"\n===== COLOCALIZATION ANALYSIS RESULTS =====")
        txt.append(f"Channel A: {self._mhtt_label}   Channel B: {self._cct1_label}")
        txt.append(f"Parameters: A-factor={p['ms']}, B-factor={p['cs']}, "
                   f"sigma={p['sigma']}, min_size={p['min_size']}")
        txt.append(f"Thresholds: A>{vis['mt']:.4f}  B>{vis['ct']:.4f}")
        txt.append(f"1. Pearson r = {r['pr']:.4f}  (p={r['pp']:.2e})")
        txt.append(f"2. Manders M1={r['m1']:.4f}  M2={r['m2']:.4f}")
        txt.append(f"3. Dice = {r['dice']:.4f}")
        txt.append(f"4. Mann-Whitney U={r['u_stat']:.2f}  p={r['p_val']:.2e}")
        txt.append(f"5. Cohen's d={r['cohen_d']:.4f}  "
                   f"mean_in={r['mean_in']:.2f}  mean_out={r['mean_out']:.2f}")
        txt.append(f"6. IoU = {r['iou']:.4f}")

        df = pd.DataFrame([
            {"metric": k, "value": round(float(v), 4) if isinstance(v, float) else v}
            for k, v in r.items() if isinstance(v, (int, float))
        ])
        self._state.set_result("Colocalization", {
            "dataframe":     df,
            "summary":       txt.toPlainText(),
            "channel_a":     self._mhtt_label,
            "channel_b":     self._cct1_label,
        })
        # Store full results in AppState memory for on-demand export
        self._state.coloc_results = r
        # Enable save button now that results exist
        self._save_btn.setEnabled(True)
        self._stat_lbl.ok("Analysis complete. Click 'Save Colocalization Report' to export.")

    def _save_coloc_report(self) -> None:
        """Called when user clicks 'Save Colocalization Report'.
        Writes the in-memory results to disk using the same PDF format"""
        r = self._state.coloc_results
        if r is None:
            QMessageBox.warning(self, "No Results",
                                "Run colocalization analysis first before saving.")
            return
        try:
            self._save_btn.setEnabled(False)
            self._stat_lbl.info("Saving report…")
            QApplication.processEvents()
            pdf_fn = save_coloc_pdf(r)
            self._save_btn.setEnabled(True)
            self._stat_lbl.ok(f"Report saved → {pdf_fn}")
            QMessageBox.information(self, "Report Saved",
                                    f"Colocalization report saved to:\n{pdf_fn}")
        except Exception as exc:
            self._save_btn.setEnabled(True)
            self._stat_lbl.err(f"Save failed: {exc}")
            QMessageBox.critical(self, "Save Error", str(exc))


class ThemeDialog(QDialog):
    """Compact dialog: one colour-wheel picker per theme slot."""

    _LABELS = [
        ("accent",     "Accent  (tabs, highlights)"),
        ("background", "Background"),
        ("toolbar",    "Top Toolbar"),
        ("viewer_tb",  "Viewer Toolbar (zoom bar)"),
    ]

    def __init__(self, current_theme, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Customise Theme")
        self.setFixedWidth(320)
        self._theme = dict(current_theme)
        self.result_theme = {}

        main_lyt = QVBoxLayout(self)
        main_lyt.setSpacing(8)

        form = QFormLayout()
        form.setSpacing(6)
        self._btns = {}

        for key, label in self._LABELS:
            btn = QPushButton()
            btn.setFixedSize(60, 24)
            self._update_btn(btn, self._theme.get(key, "#222222"))
            btn.clicked.connect(lambda _, k=key, b=btn: self._pick(k, b))
            form.addRow(label + ":", btn)
            self._btns[key] = btn

        main_lyt.addLayout(form)

        hint = QLabel("Click a colour swatch to change it.")
        hint.setStyleSheet("color:#8080a0;font-size:11px;")
        main_lyt.addWidget(hint)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._on_ok)
        bb.rejected.connect(self.reject)
        main_lyt.addWidget(bb)

    def _update_btn(self, btn, hex_col):
        btn.setStyleSheet(f"background:{hex_col};border:1px solid #555;border-radius:3px;")

    def _pick(self, key, btn):
        color = QColorDialog.getColor(QColor(self._theme.get(key, "#222222")), self,
                                      f"Choose colour — {key}")
        if color.isValid():
            self._theme[key] = color.name()
            self._update_btn(btn, color.name())

    def _on_ok(self):
        self.result_theme = dict(self._theme)
        self.accept()


# Main application window — assembles all tabs and the menu bar
class MainWindow(QMainWindow):
    APP_TITLE   = "CureQ Microscopy Analysis Application"
    APP_VERSION = "Final"

    def __init__(self):
        super().__init__()
        self.setMinimumSize(1400, 900)
        self.resize(1700, 1000)
        self.setWindowTitle(f"{self.APP_TITLE}  {self.APP_VERSION}")
        self.setWindowIcon(QIcon("Q_Logo.jpg"))
        self._state = AppState()
        self._build_menus()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()
        self.setStyleSheet(build_qss())
        self._apply_palette()
        self._sb.showMessage("CureQ Microscopy Analysis Application Final — ready.", 6000)

    def _apply_palette(self):
        pal = QPalette()
        pal.setColor(QPalette.Window,          QColor(28, 28, 50))
        pal.setColor(QPalette.WindowText,      QColor(220, 220, 240))
        pal.setColor(QPalette.Base,            QColor(18, 18, 30))
        pal.setColor(QPalette.AlternateBase,   QColor(26, 26, 46))
        pal.setColor(QPalette.Text,            QColor(220, 220, 240))
        pal.setColor(QPalette.Button,          QColor(30, 30, 60))
        pal.setColor(QPalette.ButtonText,      QColor(220, 220, 240))
        pal.setColor(QPalette.Highlight,       QColor(224, 53, 122))
        pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        pal.setColor(QPalette.ToolTipBase,     QColor(22, 22, 40))
        pal.setColor(QPalette.ToolTipText,     QColor(200, 200, 230))
        pal.setColor(QPalette.Link,            QColor(100, 140, 255))
        self.setPalette(pal)

    def _build_menus(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File")
        _TABS = ["Image Viewer","Pre Processing","Cell Body & Nuclei Detection",
                 "Aggregate Detection","Cell Region Analysis","Colocalization","Results"]
        for i, name in enumerate(_TABS):
            a = QAction(name, self)
            a.triggered.connect(lambda _, idx=i: self._tabs.setCurrentIndex(idx))
            fm.addAction(a)
        fm.addSeparator()
        qa = QAction("&Quit", self); qa.setShortcut("Ctrl+Q"); qa.triggered.connect(self.close)
        fm.addAction(qa)
        hm = mb.addMenu("&Help")
        ab = QAction("About", self); ab.triggered.connect(self._about); hm.addAction(ab)

    def _build_toolbar(self):
        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)

        _btn_style = (
            "QPushButton{background:transparent;color:#fff;"
            "border:1px solid rgba(255,255,255,0.28);border-radius:4px;"
            "padding:2px 10px;font-size:12px;min-height:22px;}"
            "QPushButton:hover{background:rgba(255,255,255,0.15);}"
            "QPushButton:pressed{background:rgba(0,0,0,0.18);}"
        )

        def _tb_btn(label, tip, slot):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.setStyleSheet(_btn_style)
            b.clicked.connect(slot)
            tb.addWidget(b)
            return b

        _tb_btn("📂  Open",        "Open TIFF / LIF file",            self._toolbar_open)
        _tb_btn("💾  Save",        "Save current result",             self._toolbar_save)
        _tb_btn("📊  Statistics",  "Show quick statistics",           self._toolbar_stats)
        _tb_btn("❓  Help",        "Show help for current tab",       self._toolbar_help)

        # separator
        sep_lbl = QLabel("  |  ")
        sep_lbl.setStyleSheet("color:rgba(255,255,255,0.30);")
        tb.addWidget(sep_lbl)

        _tb_btn("🎨  Theme",       "Customise app colours",           self._pick_theme)

    def _toolbar_open(self):
        """Delegate to ImageViewerTab load button."""
        self._tabs.setCurrentIndex(0)
        self._image_viewer_tab._load_file()

    def _toolbar_save(self):
        """Quick-save: export current results as CSV if available."""
        if not self._state.results_dict:
            QMessageBox.information(self, "Save", "No results to save yet.\nRun a tool first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "CSV files (*.csv);;JSON files (*.json)")
        if not path: return
        import csv as _csv, json as _json
        tool = next(iter(self._state.results_dict))
        data = self._state.results_dict[tool]
        df   = data.get("dataframe")
        if path.endswith(".json"):
            with open(path, "w") as f:
                _json.dump({k: str(v) for k, v in data.items() if k != "dataframe"}, f, indent=2)
        elif df is not None:
            df.to_csv(path, index=False)
        else:
            with open(path, "w") as f:
                _json.dump({k: str(v) for k, v in data.items() if k != "dataframe"}, f, indent=2)
        self._sb.showMessage(f"Saved to {path}", 4000)

    def _toolbar_stats(self):
        """Show a quick stats summary dialog."""
        if not self._state.results_dict:
            QMessageBox.information(self, "Statistics", "No results available yet.")
            return
        lines = []
        for tool, data in self._state.results_dict.items():
            lines.append(f"<b>{tool}</b>")
            df = data.get("dataframe")
            if df is not None and hasattr(df, "__len__"):
                lines.append(f"  Objects: {len(df)}")
            summary = data.get("summary", "")
            if summary:
                lines.append(f"  {summary[:200]}")
            lines.append("")
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Statistics")
        dlg.setTextFormat(Qt.RichText)
        dlg.setText("<br>".join(lines) or "No summary available.")
        dlg.setStyleSheet(
            "QMessageBox{background:#1c1c32;color:#dcdcf0;}"
            "QLabel{color:#dcdcf0;font-size:12px;}"
            "QPushButton{background:#2a2a50;color:#c0c0ff;border:1px solid #3a3a70;"
            "border-radius:4px;padding:5px 16px;}")
        dlg.exec_()

    def _toolbar_help(self):
        """Show help for the currently active tab."""
        tab_names = [
            "Image Viewer", "Pre Processing",
            "Cell Body & Nuclei Detection", "Aggregate Detection",
            "Cell Region Analysis", "Colocalization", "Results",
        ]
        idx  = self._tabs.currentIndex()
        key  = tab_names[idx] if idx < len(tab_names) else "Image Viewer"
        html = TAB_HELP_TEXT.get(key,
               f"<p>No help available for <b>{key}</b>.</p>")
        dlg  = QMessageBox(self)
        dlg.setWindowTitle(f"Help – {key}")
        dlg.setTextFormat(Qt.RichText)
        dlg.setText(html)
        dlg.setStyleSheet(
            "QMessageBox{background:#1c1c32;color:#dcdcf0;}"
            "QLabel{color:#dcdcf0;font-size:12px;}"
            "QPushButton{background:#2a2a50;color:#c0c0ff;border:1px solid #3a3a70;"
            "border-radius:4px;padding:5px 16px;}")
        dlg.exec_()

    def _pick_theme(self):
        """Open a simple theme dialog — one colour picker per slot."""
        dlg = ThemeDialog(APP_THEME.copy(), self)
        if dlg.exec_() == QDialog.Accepted:
            apply_theme(self, dlg.result_theme)
            pal = self.palette()
            pal.setColor(QPalette.Highlight, QColor(APP_THEME["accent"]))
            self.setPalette(pal)

    def _build_central(self):
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._image_viewer_tab  = ImageViewerTab(self._state)
        self._preprocessing_tab = PreprocessingTab(self._state)
        self._cell_nuclei_tab   = CellBodyNucleiTab(self._state)
        self._agg_tab           = AggregateDetectionTab(self._state)
        self._region_tab        = CellRegionAnalysisTab(self._state)
        self._coloc_tab         = ColocalizationTab(self._state)
        self._results_tab       = ResultsTab(self._state)
        self._tabs.addTab(self._image_viewer_tab,  "Image Viewer")
        self._tabs.addTab(self._preprocessing_tab, "Pre Processing")
        self._tabs.addTab(self._cell_nuclei_tab,   "Cell Body & Nuclei Detection")
        self._tabs.addTab(self._agg_tab,           "Aggregate Detection")
        self._tabs.addTab(self._region_tab,        "Cell Region Analysis")
        self._tabs.addTab(self._coloc_tab,         "Colocalization")
        self._tabs.addTab(self._results_tab,       "Results")
        self.setCentralWidget(self._tabs)

    def _build_statusbar(self):
        self._sb = QStatusBar()
        self.setStatusBar(self._sb)
        self._state.subscribe(self._on_state_for_status)

    def _on_state_for_status(self, what):
        if what == "raw_image" and self._state.raw_image is not None:
            arr = self._state.raw_image
            self._sb.showMessage(
                f"Loaded: {self._state.file_path}  |  "
                f"Shape: {arr.shape[0]}Z x {arr.shape[1]}C x {arr.shape[2]}Y x {arr.shape[3]}X", 0)
        elif what == "preprocessed":
            self._sb.showMessage("Preprocessing complete. All tools ready.", 5000)
        elif what == "results":
            self._sb.showMessage(f"Results: {', '.join(self._state.results_dict)}", 5000)

    def _about(self):
        QMessageBox.about(self, f"About {self.APP_TITLE}",
            f"<h2>{self.APP_TITLE}  {self.APP_VERSION}</h2>"
            "<p>Unified PyQt5 microscopy analysis pipeline.</p>"
            "<p>Data flow: Image Viewer → Preprocessing → Analysis Tools → Results</p>"
            "<p>Segmentation modes: Slice (Z) / Max Projection / Mean Projection</p>")


def _load_app_icon(icon_path: str) -> "QIcon | None":
    candidates = [icon_path]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in ("cureq_icon.png", "cureq_icon.jpg", "neuron_icon.png",
                  "app_icon.png", "icon.png"):
        candidates.append(os.path.join(base_dir, fname))
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            px = QPixmap(path)
            if not px.isNull():
                return QIcon(px)
            from PIL import Image as _PILImg
            pil = _PILImg.open(path).convert("RGBA")
            data = pil.tobytes("raw", "RGBA")
            from PyQt5.QtGui import QImage as _QImg
            qi = _QImg(data, pil.width, pil.height, _QImg.Format_RGBA8888)
            px2 = QPixmap.fromImage(qi)
            if not px2.isNull():
                return QIcon(px2)
        except Exception:
            continue
    return None

def _apply_icon_to_dialog(dlg: "QDialog", icon: "QIcon | None"):
    if icon is not None:
        dlg.setWindowIcon(icon)

APP_ICON_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cureq_icon.png"   # CureQ brain icon — place next to this script
)

_APP_ICON: "QIcon | None" = None

def get_app_icon() -> "QIcon | None":
    global _APP_ICON
    if _APP_ICON is None:
        _APP_ICON = _load_app_icon(APP_ICON_PATH)
    return _APP_ICON

_NAMED_COLORS: dict = {
    "Red":     (1.0, 0.0, 0.0),
    "Green":   (0.0, 1.0, 0.0),
    "Blue":    (0.0, 0.0, 1.0),
    "Cyan":    (0.0, 1.0, 1.0),
    "Magenta": (1.0, 0.0, 1.0),
    "Yellow":  (1.0, 1.0, 0.0),
    "White":   (1.0, 1.0, 1.0),
    "Orange":  (1.0, 0.5, 0.0),
    "Gray":    (0.7, 0.7, 0.7),
}

def _normalize_plane(plane: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(plane, 2))
    hi = float(np.percentile(plane, 98))
    if hi <= lo:
        return np.zeros_like(plane, dtype=np.float32)
    return np.clip((plane.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)

def blend_channels(
    image: np.ndarray,          # shape (Z, C, Y, X)
    channel_states: list,       # list of dicts: {enabled, color_name, channel_index}
    z_index: int = 0,
    projection: str = "max",    # "max" | "mean" | "slice"
) -> np.ndarray:                # returns (H, W, 3) float32 in [0, 1]
    if image is None or image.ndim != 4:
        return np.zeros((1, 1, 3), dtype=np.float32)

    nz, nc, ny, nx = image.shape
    composite = np.zeros((ny, nx, 3), dtype=np.float32)

    for state in channel_states:
        if not state.get("enabled", True):
            continue
        cidx = state["channel_index"]
        if cidx >= nc:
            continue

        vol = image[:, cidx, :, :].astype(np.float32)   # (Z, Y, X)
        if projection == "max":
            plane = np.max(vol, axis=0)
        elif projection == "mean":
            plane = np.mean(vol, axis=0)
        else:  # single slice
            z = min(z_index, nz - 1)
            plane = vol[z]

        plane_n = _normalize_plane(plane)

        color_name = state.get("color_name", "White")
        rgb = np.array(_NAMED_COLORS.get(color_name, (1.0, 1.0, 1.0)),
                       dtype=np.float32)
        colored = plane_n[:, :, np.newaxis] * rgb[np.newaxis, np.newaxis, :]

        composite += colored

    return np.clip(composite, 0.0, 1.0)


class ChannelRowWidget(QWidget):
    changed = pyqtSignal()

    _COLOUR_NAMES = list(_NAMED_COLORS.keys())

    def __init__(self, channel_index: int, label: str,
                 default_color: str = "White", parent=None):
        super().__init__(parent)
        self._channel_index = channel_index
        self._color_name    = default_color

        row = QHBoxLayout(self)
        row.setContentsMargins(2, 1, 2, 1)
        row.setSpacing(6)

        self._chk = QCheckBox(label)
        self._chk.setChecked(True)
        self._chk.setMinimumWidth(110)
        self._chk.stateChanged.connect(self.changed)
        row.addWidget(self._chk, stretch=1)

        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(28, 22)
        self._color_btn.setToolTip("Click to pick a colour for this channel")
        self._color_btn.clicked.connect(self._pick_color)
        self._update_color_button()
        row.addWidget(self._color_btn)

        self._color_lbl = QLabel(default_color)
        self._color_lbl.setFixedWidth(54)
        self._color_lbl.setStyleSheet("color:#9090b0;font-size:10px;")
        row.addWidget(self._color_lbl)

    def state(self) -> dict:
        return {
            "channel_index": self._channel_index,
            "enabled":       self._chk.isChecked(),
            "color_name":    self._color_name,
        }

    def set_enabled(self, en: bool):
        self._chk.setChecked(en)

    def _pick_color(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Pick channel colour")
        icon = get_app_icon()
        if icon:
            dlg.setWindowIcon(icon)
        dlg.setFixedWidth(220)
        lyt = QVBoxLayout(dlg)
        lyt.addWidget(QLabel("Choose display colour:"))
        combo = QComboBox()
        combo.addItems(self._COLOUR_NAMES)
        combo.setCurrentText(self._color_name)
        lyt.addWidget(combo)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lyt.addWidget(bb)
        dlg.setStyleSheet(
            "QDialog{background:#22223a;color:#dcdcf0;}"
            "QLabel{color:#b0b0d0;}"
            "QComboBox{background:#18182e;color:#d8d8f0;"
            "border:1px solid #3a3a65;border-radius:4px;padding:3px;}")
        if dlg.exec_() == QDialog.Accepted:
            self._color_name = combo.currentText()
            self._color_lbl.setText(self._color_name)
            self._update_color_button()
            self.changed.emit()

    def _update_color_button(self):
        rgb = _NAMED_COLORS.get(self._color_name, (0.8, 0.8, 0.8))
        r, g, b = [int(v * 255) for v in rgb]
        self._color_btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b});"
            f"border: 1px solid #3a3a60; border-radius: 3px;")


def _compute_quality_score(
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    n_objects: int,
    properties: list,
    target_n: int = 50,
) -> float:
    if n_objects == 0 or not properties:
        return 0.0

    n_score = float(np.exp(
        -0.5 * ((np.log(max(n_objects, 1)) - np.log(target_n)) / 1.5) ** 2
    ))

    solidities = [float(p.get("solidity", 0.5)) for p in properties
                  if p.get("solidity") is not None]
    compactness = float(np.mean(solidities)) if solidities else 0.5

    areas = [float(p.get("area_px2", p.get("area", 1))) for p in properties]
    if len(areas) > 1:
        cv = float(np.std(areas) / (np.mean(areas) + 1e-6))
        size_score = float(np.exp(-cv))
    else:
        size_score = 0.5   # single object — neutral

    if label_image is not None and intensity_image is not None:
        fg_vals = intensity_image[label_image > 0].astype(np.float32)
        bg_vals = intensity_image[label_image == 0].astype(np.float32)
        if len(fg_vals) > 0 and len(bg_vals) > 0:
            fg_mean = float(fg_vals.mean())
            bg_mean = float(bg_vals.mean())
            bg_std  = float(bg_vals.std()) + 1e-6
            contrast = min(1.0, (fg_mean - bg_mean) / (3.0 * bg_std + 1e-6))
            contrast = max(0.0, contrast)
        else:
            contrast = 0.5
    else:
        contrast = 0.5

    score = (
        0.35 * n_score +
        0.25 * compactness +
        0.20 * size_score +
        0.20 * contrast
    )
    return float(np.clip(score, 0.0, 1.0))

def _auto_preprocessing_params(image: np.ndarray) -> dict:
    from skimage.filters import gaussian as _gauss
    from skimage.morphology import white_tophat, disk

    img = image.astype(np.float32)

    pmin = float(np.percentile(img, 1))
    pmax = float(np.percentile(img, 99.5))

    smoothed  = _gauss(img, sigma=1.0, preserve_range=True)
    noise_std = float(np.std(img - smoothed))
    signal_range = pmax - pmin + 1e-6
    snr = signal_range / (noise_std + 1e-6)
    if snr > 20:
        sigma = 0.5
    elif snr > 8:
        sigma = 1.0
    elif snr > 4:
        sigma = 1.5
    else:
        sigma = 2.0

    h, w   = img.shape[-2], img.shape[-1]
    bg_rad = max(10, min(h, w) // 10)
    bg_rad = min(bg_rad, 100)  # cap for performance

    tophat_r = max(3, min(h, w) // 60)
    tophat_r = min(tophat_r, 20)

    clahe_clip = min(0.08, max(0.01, 1.0 / (snr + 1.0)))

    return {
        "pmin":        round(max(0.0, (pmin / (signal_range + pmax)) * 100), 1),
        "pmax":        round(99.5, 1),
        "sigma":       round(sigma, 1),
        "bg_radius":   int(bg_rad),
        "tophat_r":    int(tophat_r),
        "clahe_clip":  round(clahe_clip, 3),
        "_snr":        round(snr, 1),
        "_noise_std":  round(noise_std, 4),
    }

def _build_search_grid(method_name: str, n_samples: int = 40) -> list:
    rng = np.random.default_rng(42)  # reproducible

    def _uniform(lo, hi, n, integer=False):
        vals = rng.uniform(lo, hi, n)
        return [int(round(v)) for v in vals] if integer else [float(round(v, 3)) for v in vals]

    grids = {
        "blob_log": lambda: [
            dict(
                min_sigma=ms, max_sigma=mxs,
                threshold=t, overlap=ov,
                min_area=5, max_area=2000
            )
            for ms, mxs, t, ov in zip(
                _uniform(0.5, 4.0,   n_samples),
                _uniform(4.0, 12.0,  n_samples),
                _uniform(0.01, 0.15, n_samples),
                _uniform(0.3, 0.8,   n_samples),
            ) if ms < mxs
        ],
        "combined": lambda: [
            dict(
                min_sigma=ms, max_sigma=mxs,
                blob_threshold=bt,
                k=k,
                weight_blob=wb,
                fuse_threshold=ft,
                min_area=5, max_area=2000
            )
            for ms, mxs, bt, k, wb, ft in zip(
                _uniform(0.5, 4.0,   n_samples),
                _uniform(4.0, 12.0,  n_samples),
                _uniform(0.01, 0.12, n_samples),
                _uniform(5, 25,      n_samples, integer=True),
                _uniform(0.2, 0.8,   n_samples),
                _uniform(0.2, 0.6,   n_samples),
            ) if ms < mxs
        ],
        "knn": lambda: [
            dict(k=k, z_score_thresh=z, min_area=5, max_area=2000)
            for k, z in zip(
                _uniform(5, 30, n_samples, integer=True),
                _uniform(1.0, 4.0, n_samples),
            )
        ],
        "otsu_ws": lambda: [
            dict(tophat_radius=r, gaussian_sigma=s, min_area=5, max_area=2000)
            for r, s in zip(
                _uniform(2, 15, n_samples, integer=True),
                _uniform(0.5, 3.0, n_samples),
            )
        ],
        "percentile": lambda: [
            dict(percentile=p, tophat_radius=r, min_area=5, max_area=2000)
            for p, r in zip(
                _uniform(80.0, 99.0, n_samples),
                _uniform(2, 15, n_samples, integer=True),
            )
        ],
    }
    builder = grids.get(method_name, grids["blob_log"])
    grid = builder()
    if len(grid) > n_samples:
        idxs = rng.choice(len(grid), n_samples, replace=False)
        grid = [grid[i] for i in idxs]
    return grid


def _compute_quality_score(
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    n_objects: int,
    properties: list,
    target_n: int = 50,
) -> float:
    if n_objects == 0 or not properties:
        return 0.0

    n_score = float(np.exp(
        -0.5 * ((np.log(max(n_objects, 1)) - np.log(target_n)) / 1.5) ** 2
    ))

    solidities = [float(p.get("solidity", 0.5)) for p in properties
                  if p.get("solidity") is not None]
    compactness = float(np.mean(solidities)) if solidities else 0.5

    areas = [float(p.get("area_px2", p.get("area", 1))) for p in properties]
    if len(areas) > 1:
        cv = float(np.std(areas) / (np.mean(areas) + 1e-6))
        size_score = float(np.exp(-cv))
    else:
        size_score = 0.5   # single object — neutral

    if label_image is not None and intensity_image is not None:
        fg_vals = intensity_image[label_image > 0].astype(np.float32)
        bg_vals = intensity_image[label_image == 0].astype(np.float32)
        if len(fg_vals) > 0 and len(bg_vals) > 0:
            fg_mean = float(fg_vals.mean())
            bg_mean = float(bg_vals.mean())
            bg_std  = float(bg_vals.std()) + 1e-6
            contrast = min(1.0, (fg_mean - bg_mean) / (3.0 * bg_std + 1e-6))
            contrast = max(0.0, contrast)
        else:
            contrast = 0.5
    else:
        contrast = 0.5

    score = (
        0.35 * n_score +
        0.25 * compactness +
        0.20 * size_score +
        0.20 * contrast
    )
    return float(np.clip(score, 0.0, 1.0))

def _auto_preprocessing_params(image: np.ndarray) -> dict:
    from skimage.filters import gaussian as _gauss
    from skimage.morphology import white_tophat, disk

    img = image.astype(np.float32)

    pmin = float(np.percentile(img, 1))
    pmax = float(np.percentile(img, 99.5))

    smoothed  = _gauss(img, sigma=1.0, preserve_range=True)
    noise_std = float(np.std(img - smoothed))
    signal_range = pmax - pmin + 1e-6
    snr = signal_range / (noise_std + 1e-6)
    if snr > 20:
        sigma = 0.5
    elif snr > 8:
        sigma = 1.0
    elif snr > 4:
        sigma = 1.5
    else:
        sigma = 2.0

    h, w   = img.shape[-2], img.shape[-1]
    bg_rad = max(10, min(h, w) // 10)
    bg_rad = min(bg_rad, 100)  # cap for performance

    tophat_r = max(3, min(h, w) // 60)
    tophat_r = min(tophat_r, 20)

    clahe_clip = min(0.08, max(0.01, 1.0 / (snr + 1.0)))

    return {
        "pmin":        round(max(0.0, (pmin / (signal_range + pmax)) * 100), 1),
        "pmax":        round(99.5, 1),
        "sigma":       round(sigma, 1),
        "bg_radius":   int(bg_rad),
        "tophat_r":    int(tophat_r),
        "clahe_clip":  round(clahe_clip, 3),
        "_snr":        round(snr, 1),
        "_noise_std":  round(noise_std, 4),
    }

def _build_search_grid(method_name: str, n_samples: int = 40) -> list:
    rng = np.random.default_rng(42)  # reproducible

    def _uniform(lo, hi, n, integer=False):
        vals = rng.uniform(lo, hi, n)
        return [int(round(v)) for v in vals] if integer else [float(round(v, 3)) for v in vals]

    grids = {
        "blob_log": lambda: [
            dict(
                min_sigma=ms, max_sigma=mxs,
                threshold=t, overlap=ov,
                min_area=5, max_area=2000
            )
            for ms, mxs, t, ov in zip(
                _uniform(0.5, 4.0,   n_samples),
                _uniform(4.0, 12.0,  n_samples),
                _uniform(0.01, 0.15, n_samples),
                _uniform(0.3, 0.8,   n_samples),
            ) if ms < mxs
        ],
        "combined": lambda: [
            dict(
                min_sigma=ms, max_sigma=mxs,
                blob_threshold=bt,
                k=k,
                weight_blob=wb,
                fuse_threshold=ft,
                min_area=5, max_area=2000
            )
            for ms, mxs, bt, k, wb, ft in zip(
                _uniform(0.5, 4.0,   n_samples),
                _uniform(4.0, 12.0,  n_samples),
                _uniform(0.01, 0.12, n_samples),
                _uniform(5, 25,      n_samples, integer=True),
                _uniform(0.2, 0.8,   n_samples),
                _uniform(0.2, 0.6,   n_samples),
            ) if ms < mxs
        ],
        "knn": lambda: [
            dict(k=k, z_score_thresh=z, min_area=5, max_area=2000)
            for k, z in zip(
                _uniform(5, 30, n_samples, integer=True),
                _uniform(1.0, 4.0, n_samples),
            )
        ],
        "otsu_ws": lambda: [
            dict(tophat_radius=r, gaussian_sigma=s, min_area=5, max_area=2000)
            for r, s in zip(
                _uniform(2, 15, n_samples, integer=True),
                _uniform(0.5, 3.0, n_samples),
            )
        ],
        "percentile": lambda: [
            dict(percentile=p, tophat_radius=r, min_area=5, max_area=2000)
            for p, r in zip(
                _uniform(80.0, 99.0, n_samples),
                _uniform(2, 15, n_samples, integer=True),
            )
        ],
    }
    builder = grids.get(method_name, grids["blob_log"])
    grid = builder()
    if len(grid) > n_samples:
        idxs = rng.choice(len(grid), n_samples, replace=False)
        grid = [grid[i] for i in idxs]
    return grid


def main():
    """Start the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("CureQ Microscopy Analysis Suite")
    app.setApplicationVersion("16.0.0")
    app.setStyleSheet(build_qss())
    app.setWindowIcon(QIcon("Q_Logo.jpg"))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
