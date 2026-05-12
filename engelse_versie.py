#!/usr/bin/env python3
"""
Protein Aggregate Analyzer — Custom Version
=====================================================
Customised version for the end user:
  ★ Tabs 1, 2 and 3 (Viewer, Pre-processing, Cell Bodies) retained
  ★ Tab 4 (Traditional Segmentation) removed
  ★ Tab 5 (Deep Learning) now contains exclusively the Ensemble method
  ★ Tab 6 (Validation) adapted for Deep Learning comparison only

Requirements: PyQt5, numpy, scipy, scikit-image, matplotlib, tifffile
Optional:  readlif (for .lif files), torch, segmentation_models_pytorch, cellpose
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
import traceback
import csv
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any

import tifffile
from scipy import ndimage as ndi
from skimage import exposure, filters, feature, morphology, measure, segmentation, color
from skimage.filters import (threshold_otsu, threshold_multiotsu, gaussian,
                              threshold_local, rank)
from skimage.morphology import (white_tophat, disk, remove_small_objects,
                                 binary_closing, binary_opening, ball)
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.draw import disk as draw_disk
from skimage.restoration import denoise_bilateral
from scipy.ndimage import maximum_filter, minimum_filter

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QGroupBox, QScrollArea,
    QFileDialog, QMessageBox, QProgressBar, QStatusBar,
    QTextEdit, QToolBar, QAction, QSizePolicy, QFrame,
    QDialogButtonBox, QDialog, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QColor, QPalette, QFont

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.cm as cm
from skimage import color as skcolor

try:
    from readlif.reader import LifFile
    HAS_READLIF = True
except ImportError:
    HAS_READLIF = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from cellpose import models as cellpose_models
    from scipy.ndimage import binary_fill_holes as _bfh
    HAS_CELLPOSE = True
except ImportError:
    HAS_CELLPOSE = False

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False


# ═══════════════════════════════════════════════════════════════════════════════
#  COLOR THEME  —  Professional Dark UI
# ═══════════════════════════════════════════════════════════════════════════════
DARK_THEME = """
/* ── Base ─────────────────────────────────────────────────────────────────── */
QMainWindow, QDialog {
    background-color: #0d1117;
    color: #c9d1d9;
}
QWidget {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Segoe UI', 'Inter', 'SF Pro Display', Arial, sans-serif;
    font-size: 12px;
}

/* ── Tab bar ──────────────────────────────────────────────────────────────── */
QTabWidget::pane {
    border: 1px solid #21262d;
    background-color: #161b22;
    border-radius: 0px 6px 6px 6px;
}
QTabBar {
    background-color: transparent;
}
QTabBar::tab {
    background-color: #161b22;
    color: #8b949e;
    padding: 9px 22px;
    border: 1px solid #21262d;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    margin-right: 3px;
    font-weight: 600;
    font-size: 12px;
    min-width: 100px;
}
QTabBar::tab:selected {
    background-color: #1f6feb;
    color: #ffffff;
    border-color: #1f6feb;
}
QTabBar::tab:hover:!selected {
    background-color: #21262d;
    color: #c9d1d9;
}

/* ── Group boxes ──────────────────────────────────────────────────────────── */
QGroupBox {
    border: 1px solid #21262d;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 14px;
    padding-left: 6px;
    padding-right: 6px;
    padding-bottom: 6px;
    font-weight: 700;
    font-size: 11px;
    color: #58a6ff;
    background-color: #161b22;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #79c0ff;
    letter-spacing: 0.3px;
    font-size: 12px;
    font-weight: 700;
}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
QPushButton {
    background-color: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 16px;
    font-weight: 600;
    font-size: 12px;
}
QPushButton:hover {
    background-color: #30363d;
    color: #ffffff;
    border-color: #58a6ff;
}
QPushButton:pressed {
    background-color: #1f6feb;
    border-color: #1f6feb;
    color: #ffffff;
}
QPushButton:disabled {
    background-color: #161b22;
    color: #484f58;
    border-color: #21262d;
}
QPushButton#primary {
    background-color: #1f6feb;
    color: #ffffff;
    border-color: #1f6feb;
    font-size: 13px;
    padding: 8px 22px;
    border-radius: 6px;
}
QPushButton#primary:hover {
    background-color: #388bfd;
    border-color: #388bfd;
}
QPushButton#primary:pressed {
    background-color: #1158c7;
    border-color: #1158c7;
}

/* ── Sliders ──────────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 4px;
    background: #21262d;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #58a6ff;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    border: 2px solid #0d1117;
}
QSlider::handle:horizontal:hover {
    background: #79c0ff;
}
QSlider::sub-page:horizontal {
    background: #1f6feb;
    border-radius: 2px;
}

/* ── Spinboxes / Combos / Inputs ──────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #0d1117;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px 8px;
    selection-background-color: #1f6feb;
}
QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    border-color: #58a6ff;
}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background-color: #21262d;
    border: none;
    border-radius: 3px;
    width: 16px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #30363d;
}
QComboBox {
    background-color: #0d1117;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px 8px;
    selection-background-color: #1f6feb;
}
QComboBox:focus {
    border-color: #58a6ff;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #58a6ff;
    margin-right: 6px;
}
QComboBox QAbstractItemView {
    background-color: #161b22;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    selection-background-color: #1f6feb;
    outline: none;
    padding: 2px;
}

/* ── Checkboxes ───────────────────────────────────────────────────────────── */
QCheckBox {
    color: #c9d1d9;
    spacing: 8px;
    font-size: 12px;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border-radius: 4px;
    border: 1px solid #30363d;
    background-color: #0d1117;
}
QCheckBox::indicator:hover {
    border-color: #58a6ff;
}
QCheckBox::indicator:checked {
    background-color: #1f6feb;
    border-color: #1f6feb;
    image: none;
}

/* ── Labels ───────────────────────────────────────────────────────────────── */
QLabel {
    color: #c9d1d9;
}

/* ── Tables ───────────────────────────────────────────────────────────────── */
QTableWidget {
    background-color: #0d1117;
    alternate-background-color: #161b22;
    gridline-color: #21262d;
    color: #c9d1d9;
    border: 1px solid #21262d;
    border-radius: 6px;
    selection-background-color: #1f6feb;
}
QHeaderView::section {
    background-color: #161b22;
    color: #8b949e;
    padding: 6px 10px;
    border: none;
    border-right: 1px solid #21262d;
    border-bottom: 1px solid #21262d;
    font-weight: 700;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

/* ── Progress bar ─────────────────────────────────────────────────────────── */
QProgressBar {
    border: 1px solid #21262d;
    border-radius: 6px;
    background-color: #161b22;
    text-align: center;
    color: #c9d1d9;
    font-weight: 600;
    font-size: 11px;
    height: 14px;
}
QProgressBar::chunk {
    background-color: #1f6feb;
    border-radius: 5px;
}

/* ── Status bar ───────────────────────────────────────────────────────────── */
QStatusBar {
    background-color: #161b22;
    color: #8b949e;
    border-top: 1px solid #21262d;
    font-size: 11px;
    padding: 2px 8px;
}
QStatusBar::item {
    border: none;
}

/* ── Scroll bars ──────────────────────────────────────────────────────────── */
QScrollBar:vertical {
    background: #0d1117;
    width: 8px;
    border-radius: 4px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #30363d;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: #484f58;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal {
    background: #0d1117;
    height: 8px;
    border-radius: 4px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #30363d;
    border-radius: 4px;
    min-width: 24px;
}
QScrollBar::handle:horizontal:hover {
    background: #484f58;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }

/* ── Text edit (log / metadata) ───────────────────────────────────────────── */
QTextEdit {
    background-color: #0d1117;
    color: #79c0ff;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 6px;
    font-family: 'Cascadia Code', 'Fira Code', 'Courier New', monospace;
    font-size: 11px;
    line-height: 1.5;
    selection-background-color: #1f6feb;
}

/* ── Toolbar ──────────────────────────────────────────────────────────────── */
QToolBar {
    background-color: #161b22;
    border-bottom: 1px solid #21262d;
    spacing: 4px;
    padding: 4px 8px;
}
QToolBar QToolButton {
    background-color: transparent;
    color: #8b949e;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 5px 12px;
    font-weight: 600;
    font-size: 12px;
}
QToolBar QToolButton:hover {
    background-color: #21262d;
    color: #c9d1d9;
    border-color: #30363d;
}
QToolBar QToolButton:pressed {
    background-color: #1f6feb;
    color: #ffffff;
    border-color: #1f6feb;
}

/* ── Scroll area ──────────────────────────────────────────────────────────── */
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollArea > QWidget > QWidget {
    background-color: transparent;
}

/* ── Form layout labels ───────────────────────────────────────────────────── */
QFormLayout QLabel {
    color: #c9d1d9;
    font-size: 12px;
    font-weight: 500;
}

/* ── Message boxes ────────────────────────────────────────────────────────── */
QMessageBox {
    background-color: #161b22;
    color: #c9d1d9;
}
QMessageBox QLabel {
    color: #c9d1d9;
}
QMessageBox QPushButton {
    min-width: 80px;
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class ImageStack:
    name: str = ""
    data: Optional[np.ndarray] = None   # (Z, C, Y, X)
    z_count: int = 0
    channel_count: int = 0
    height: int = 0
    width: int = 0
    channel_names: List[str] = field(default_factory=list)
    pixel_size_xy: float = 1.0
    pixel_size_z: float = 1.0
    bit_depth: int = 16
    source_file: str = ""

    def max_projection(self, channel: int = 0) -> np.ndarray:
        if self.data is None:
            return np.zeros((1, 1), dtype=np.float32)
        return self.data[:, channel].max(axis=0).astype(np.float32)

    def get_slice(self, z: int, channel: int = 0) -> np.ndarray:
        if self.data is None:
            return np.zeros((1, 1), dtype=np.float32)
        return self.data[z, channel].astype(np.float32)


@dataclass
class SegmentationResult:
    method_name: str = ""
    params: Dict = field(default_factory=dict)
    label_image: Optional[np.ndarray] = None
    blob_list: Optional[np.ndarray] = None
    n_objects: int = 0
    properties: List[Dict] = field(default_factory=list)
    score: float = 0.0
    time_seconds: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  HOT COLORMAP DECODER
# ═══════════════════════════════════════════════════════════════════════════════
def decode_hot_colormap(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb.astype(np.float32) / (rgb.max() + 1e-8)
    if rgb.ndim == 3 and rgb.shape[2] == 1:
        return rgb[:, :, 0].astype(np.float32) / (rgb.max() + 1e-8)
    if rgb.ndim == 3 and rgb.shape[2] >= 3:
        r = rgb[:, :, 0].astype(np.float32) / 255.0
        g = rgb[:, :, 1].astype(np.float32) / 255.0
        b = rgb[:, :, 2].astype(np.float32) / 255.0
        intensity = np.where(
            r < 1.0 - 1e-4,
            r / 3.0,
            np.where(
                g < 1.0 - 1e-4,
                1.0 / 3.0 + g / 3.0,
                2.0 / 3.0 + b / 3.0
            )
        )
        return np.clip(intensity, 0.0, 1.0).astype(np.float32)
    return rgb.astype(np.float32).mean(axis=2) / 255.0


def is_hot_encoded(arr: np.ndarray) -> bool:
    if arr.ndim != 3 or arr.shape[2] < 3:
        return False
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    frac_blue_zero = (b < 10).mean()
    frac_red_high = (r > 50).mean()
    return bool(frac_blue_zero > 0.7 and frac_red_high > 0.1)


# ═══════════════════════════════════════════════════════════════════════════════
#  BESTANDEN LADEN
# ═══════════════════════════════════════════════════════════════════════════════
class ImageLoader:
    @staticmethod
    def load(filepath: str) -> List[ImageStack]:
        ext = Path(filepath).suffix.lower()
        if ext == ".lif":
            return ImageLoader._load_lif(filepath)
        elif ext in (".tif", ".tiff"):
            return [ImageLoader._load_tiff(filepath)]
        else:
            raise ValueError(f"Niet-ondersteund bestandsformaat: {ext}")

    @staticmethod
    def _load_lif(filepath: str) -> List[ImageStack]:
        if not HAS_READLIF:
            raise ImportError("readlif is not installed.\nInstall with: pip install readlif")
        lif = LifFile(filepath)
        stacks = []
        for img in lif.get_iter_image():
            dims = img.dims
            z = dims.z if dims.z > 0 else 1
            c = img.channels if hasattr(img, 'channels') and img.channels > 0 else 1
            h, w = dims.y, dims.x
            data = np.zeros((z, c, h, w), dtype=np.uint16)
            for ci in range(c):
                for zi in range(z):
                    frame = np.array(img.get_frame(z=zi, t=0, c=ci))
                    data[zi, ci] = frame
            stacks.append(ImageStack(
                name=img.name, data=data.astype(np.float32), z_count=z, channel_count=c,
                height=h, width=w, channel_names=[f"Ch{i}" for i in range(c)],
                pixel_size_xy=getattr(img, 'scale', [1.0]*3)[0],
                pixel_size_z=getattr(img, 'scale', [1.0]*3)[2],
                bit_depth=16, source_file=filepath
            ))
        return stacks

    @staticmethod
    def _load_tiff(filepath: str) -> ImageStack:
        with tifffile.TiffFile(filepath) as tif:
            arr = tif.asarray()
            metadata = tif.imagej_metadata or {}

        decoded = False
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            if is_hot_encoded(arr):
                gray = decode_hot_colormap(arr)
                arr = gray[np.newaxis, np.newaxis]
                decoded = True
        if arr.ndim == 3 and arr.shape[2] in (3, 4) and not decoded:
            gray = arr.max(axis=2).astype(np.float32)
            arr = gray[np.newaxis, np.newaxis]

        arr = ImageLoader._normalize_shape(arr)
        z, c, h, w = arr.shape
        return ImageStack(
            name=Path(filepath).stem, data=arr.astype(np.float32),
            z_count=z, channel_count=c, height=h, width=w,
            channel_names=[f"Ch{i}" for i in range(c)],
            pixel_size_xy=float(metadata.get("PixelSize", 1.0)),
            pixel_size_z=float(metadata.get("spacing", 1.0)),
            bit_depth=int(metadata.get("bit_depth", 16)),
            source_file=filepath
        )

    @staticmethod
    def _normalize_shape(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return arr[np.newaxis, np.newaxis]
        elif arr.ndim == 3:
            if arr.shape[2] <= 4:
                return arr.transpose(2, 0, 1)[np.newaxis]
            return arr[:, np.newaxis]
        elif arr.ndim == 4:
            if arr.shape[3] <= 4:
                return arr.transpose(0, 3, 1, 2)
            return arr
        elif arr.ndim == 5:
            return arr[0]
        return arr


# ═══════════════════════════════════════════════════════════════════════════════
#  PRE-PROCESSING  (extended)
# ═══════════════════════════════════════════════════════════════════════════════
class Preprocessor:

    @staticmethod
    def normalize(img: np.ndarray, pmin: float = 2.0, pmax: float = 99.8) -> np.ndarray:
        lo, hi = np.percentile(img, [pmin, pmax])
        if hi == lo:
            return np.zeros_like(img, dtype=np.float32)
        return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)

    @staticmethod
    def subtract_background_rolling_ball(img: np.ndarray, radius: int = 50) -> np.ndarray:
        img_f = img.astype(np.float32)
        r = max(3, min(radius, min(img_f.shape) // 4, 40))
        from skimage.morphology import opening as morph_opening
        bg = morph_opening(img_f, disk(r))
        return np.clip(img_f - bg, 0, None).astype(np.float32)

    @staticmethod
    def subtract_background_gaussian(img: np.ndarray, sigma: float = 30.0) -> np.ndarray:
        img_f = img.astype(np.float32)
        bg = gaussian(img_f, sigma=sigma)
        return np.clip(img_f - bg * 0.95, 0, None).astype(np.float32)

    @staticmethod
    def tophat(img: np.ndarray, radius: int = 5) -> np.ndarray:
        return white_tophat(img.astype(np.float32), disk(radius)).astype(np.float32)

    @staticmethod
    def multiscale_tophat(img: np.ndarray, radii: Tuple[int, ...] = (3, 6, 10)) -> np.ndarray:
        img_f = img.astype(np.float32)
        result = np.zeros_like(img_f)
        for r in radii:
            th = white_tophat(img_f, disk(r))
            result = np.maximum(result, th)
        return result

    @staticmethod
    def denoise_gaussian(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return gaussian(img.astype(np.float32), sigma=sigma).astype(np.float32)

    @staticmethod
    def denoise_bilateral(img: np.ndarray, sigma_color: float = 0.1,
                          sigma_spatial: float = 2.0) -> np.ndarray:
        img_f = np.clip(img.astype(np.float32), 0, 1)
        try:
            return denoise_bilateral(
                img_f, sigma_color=sigma_color, sigma_spatial=sigma_spatial,
                mode='reflect'
            ).astype(np.float32)
        except Exception:
            return gaussian(img_f, sigma=1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS FOR SEGMENTATION & DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

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
    return label(out > 0)

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

def overlay_labels_on_image(img, labels, alpha=0.45):
    img_n = Preprocessor.normalize(img)
    bg = np.stack([img_n] * 3, axis=-1)
    if labels.max() == 0:
        return (bg * 255).astype(np.uint8)
    overlay = skcolor.label2rgb(labels, image=bg, alpha=alpha, bg_label=0, bg_color=None)
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

def draw_actual_contours(ax, result, circle_color="#00ffcc", text_color="#ffffff",
                         circle_lw=1.2, font_size=6.5, show_numbers=True, max_labels=500):
    from skimage import measure
    if result.label_image is None or result.label_image.max() == 0:
        return
    for region in regionprops(result.label_image)[:max_labels]:
        mask = (result.label_image == region.label)
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=circle_lw,
                    color=circle_color, zorder=4)
        if show_numbers:
            cy, cx = region.centroid
            ax.text(cx, cy, str(region.label), color=text_color,
                    fontsize=font_size, fontweight="bold",
                    ha="center", va="center", zorder=5, clip_on=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB CANVAS
# ═══════════════════════════════════════════════════════════════════════════════
class MplCanvas(FigureCanvas):
    OUTER_BG = "#161b22"
    INNER_BG = "#0d1117"
    TICK_CLR = "#c9d1d9"
    GRID_CLR = "#30363d"

    def __init__(self, parent=None, width=6, height=5, dpi=100, bgcolor=None):
        bg = bgcolor or self.OUTER_BG
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=bg)
        self.axes = self.fig.add_subplot(111)
        self._style_axes(self.axes)
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def _style_axes(self, ax):
        ax.set_facecolor(self.INNER_BG)
        ax.tick_params(colors=self.TICK_CLR, labelsize=9)
        ax.xaxis.label.set_color(self.TICK_CLR)
        ax.yaxis.label.set_color(self.TICK_CLR)
        for spine in ax.spines.values():
            spine.set_edgecolor(self.GRID_CLR)

    def clear(self):
        self.fig.clf()
        self.axes = self.fig.add_subplot(111)
        self._style_axes(self.axes)
        self.draw_idle()


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — VIEWER
# ═══════════════════════════════════════════════════════════════════════════════
class ViewerTab(QWidget):
    stack_loaded = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.current_stack: Optional[ImageStack] = None
        self.all_stacks: List[ImageStack] = []
        self._build_ui()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(8)

        left = QWidget()
        left.setFixedWidth(260)
        lv = QVBoxLayout(left)
        lv.setSpacing(8)

        grp_file = QGroupBox("📂  Load File")
        fv = QVBoxLayout(grp_file)
        self.btn_open = QPushButton("Open .LIF / .TIF file")
        self.btn_open.setObjectName("primary")
        self.btn_open.setToolTip(
            "Open a microscopy file.\n"
            "Supported formats:\n"
            "  • .LIF  — Leica Image File (multiple series possible)\n"
            "  • .TIF / .TIFF — standard TIFF (also multi-channel/Z-stack)"
        )
        self.btn_open.clicked.connect(self._open_file)
        fv.addWidget(self.btn_open)
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet("color:#58a6ff; font-size:11px;")
        fv.addWidget(self.lbl_file)
        self.cmb_series = QComboBox()
        self.cmb_series.setToolTip(
            "Choose the series (experiment) to view within the loaded file.\n"
            "LIF files can contain multiple recordings (series);\n"
            "TIF files typically have only one series."
        )
        self.cmb_series.currentIndexChanged.connect(self._series_changed)
        fv.addWidget(QLabel("Series:"))
        fv.addWidget(self.cmb_series)
        lv.addWidget(grp_file)

        grp_view = QGroupBox("🎨  Display Settings")
        vv = QFormLayout(grp_view)
        self.cmb_channel = QComboBox()
        self.cmb_channel.setToolTip(
            "Choose which fluorescence channel to view.\n"
            "Each channel corresponds to a different dye or marker\n"
            "(e.g. DAPI for nuclei, GFP for proteins)."
        )
        self.cmb_channel.currentIndexChanged.connect(self._refresh_image)
        vv.addRow("Channel:", self.cmb_channel)
        self.cmb_display = QComboBox()
        self.cmb_display.addItems(["Max Projection", "Z-Slice"])
        self.cmb_display.setToolTip(
            "Max Projection: shows the maximum pixel value across all Z-layers.\n"
            "  → Provides a complete overview of all structures in the volume.\n\n"
            "Z-Slice: shows one specific Z-layer at a time.\n"
            "  → Use the slider to navigate through the layers."
        )
        self.cmb_display.currentIndexChanged.connect(self._toggle_view_mode)
        vv.addRow("Mode:", self.cmb_display)
        self.sld_z = QSlider(Qt.Horizontal)
        self.sld_z.setMinimum(0); self.sld_z.setMaximum(0)
        self.sld_z.setToolTip(
            "Slide to browse through the Z-layers (depth) of the stack.\n"
            "Only active in 'Z-Slice' mode."
        )
        self.sld_z.valueChanged.connect(self._refresh_image)
        self.sld_z.setEnabled(False)
        self.lbl_z = QLabel("Z: 0 / 0")
        vv.addRow(self.lbl_z, self.sld_z)
        lv.addWidget(grp_view)

        grp_cmap = QGroupBox("🌈  Color Map")
        cv = QFormLayout(grp_cmap)
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(["hot", "gray", "inferno", "viridis",
                                 "magma", "plasma", "cividis", "turbo"])
        self.cmb_cmap.setToolTip(
            "Choose the color map for displaying the image:\n"
            "  • hot      — black → red → yellow → white (good for aggregates)\n"
            "  • gray     — grayscale (standard microscopy)\n"
            "  • inferno  — black → purple → orange → white\n"
            "  • viridis  — dark blue → green → yellow (colorblind-safe)\n"
            "  • magma    — black → purple → pink → white\n"
            "  • plasma   — blue → purple → yellow\n"
            "  • cividis  — blue → green → yellow (colorblind-safe)\n"
            "  • turbo    — rainbow with better perception"
        )
        self.cmb_cmap.currentIndexChanged.connect(self._refresh_image)
        cv.addRow("Color map:", self.cmb_cmap)
        self.chk_autoscale = QCheckBox("Auto-scale intensity")
        self.chk_autoscale.setChecked(True)
        self.chk_autoscale.setToolTip(
            "If checked: automatically adjusts brightness to the\n"
            "minimum and maximum value of the visible image.\n\n"
            "If unchecked: uses a fixed scale from 0 to the\n"
            "maximum pixel value in the image."
        )
        self.chk_autoscale.stateChanged.connect(self._refresh_image)
        cv.addWidget(self.chk_autoscale)
        lv.addWidget(grp_cmap)

        grp_meta = QGroupBox("📋  Metadata")
        mv = QVBoxLayout(grp_meta)
        self.txt_meta = QTextEdit()
        self.txt_meta.setReadOnly(True)
        self.txt_meta.setMinimumHeight(120)
        mv.addWidget(self.txt_meta)
        lv.addWidget(grp_meta)
        lv.addStretch()
        main_layout.addWidget(left)

        right_layout = QVBoxLayout()
        self.canvas_view = MplCanvas(width=10, height=8)
        nav = NavigationToolbar(self.canvas_view, self)
        nav.setStyleSheet("background:#f0f3f6; border-bottom: 1px solid #d0d7de; border-radius: 6px 6px 0 0;")
        right_layout.addWidget(nav)
        right_layout.addWidget(self.canvas_view)
        right_w = QWidget()
        right_w.setLayout(right_layout)
        main_layout.addWidget(right_w)

    def _open_file(self):
        filt = "Microscopy files (*.lif *.tif *.tiff);;All files (*)"
        path, _ = QFileDialog.getOpenFileName(self, "Open file", "", filt)
        if not path:
            return
        try:
            stacks = ImageLoader.load(path)
            self.all_stacks = stacks
            self.cmb_series.blockSignals(True)
            self.cmb_series.clear()
            for s in stacks:
                self.cmb_series.addItem(s.name)
            self.cmb_series.blockSignals(False)
            self._load_stack(stacks[0])
            self.lbl_file.setText(Path(path).name)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def _series_changed(self, idx):
        if 0 <= idx < len(self.all_stacks):
            self._load_stack(self.all_stacks[idx])

    def _load_stack(self, stack):
        self.current_stack = stack
        self.cmb_channel.blockSignals(True)
        self.cmb_channel.clear()
        for name in stack.channel_names:
            self.cmb_channel.addItem(name)
        self.cmb_channel.blockSignals(False)
        self.sld_z.setMaximum(max(0, stack.z_count - 1))
        self.sld_z.setValue(0)
        self._update_metadata()
        self._refresh_image()
        self.stack_loaded.emit(stack)

    def _toggle_view_mode(self, idx):
        self.sld_z.setEnabled(idx == 1)
        self._refresh_image()

    def _refresh_image(self):
        if self.current_stack is None:
            return
        ch   = self.cmb_channel.currentIndex()
        cmap = self.cmb_cmap.currentText()
        mode = self.cmb_display.currentIndex()
        if mode == 0:
            img = self.current_stack.max_projection(ch)
            title = f"Max Projectie — {self.current_stack.channel_names[ch]}"
        else:
            z = self.sld_z.value()
            self.lbl_z.setText(f"Z: {z} / {self.current_stack.z_count - 1}")
            img = self.current_stack.get_slice(z, ch)
            title = f"Z={z} — {self.current_stack.channel_names[ch]}"
        ax = self.canvas_view.axes
        ax.cla()
        vmin, vmax = (None, None) if self.chk_autoscale.isChecked() else (0, img.max())
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
                  interpolation="nearest")
        ax.set_title(title, color="#79c0ff", fontsize=11)
        ax.axis("off")
        self.canvas_view.fig.tight_layout(pad=0.5)
        self.canvas_view.draw_idle()

    def _update_metadata(self):
        s = self.current_stack
        if s is None:
            return
        txt = (
            f"Naam:       {s.name}\n"
            f"Vorm:       Z={s.z_count}, C={s.channel_count}, "
            f"Y={s.height}, X={s.width}\n"
            f"Kanalen:    {', '.join(s.channel_names)}\n"
            f"Pixel XY:   {s.pixel_size_xy:.4f} µm\n"
            f"Pixel Z:    {s.pixel_size_z:.4f} µm\n"
            f"Bitdiepte:  {s.bit_depth}\n"
            f"Bron:       {Path(s.source_file).name}\n"
        )
        self.txt_meta.setPlainText(txt)

    def get_current_image(self) -> Optional[np.ndarray]:
        if self.current_stack is None:
            return None
        ch   = self.cmb_channel.currentIndex()
        mode = self.cmb_display.currentIndex()
        if mode == 0:
            return self.current_stack.max_projection(ch)
        return self.current_stack.get_slice(self.sld_z.value(), ch)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — PRE-PROCESSING (extended)
# ═══════════════════════════════════════════════════════════════════════════════
class PreprocessTab(QWidget):
    preprocessed = pyqtSignal(np.ndarray)

    def __init__(self, viewer_tab: ViewerTab):
        super().__init__()
        self.viewer = viewer_tab
        self.result_img: Optional[np.ndarray] = None
        self._build_ui()

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setSpacing(8)

        left = QWidget()
        left.setFixedWidth(290)
        lv = QVBoxLayout(left)

        grp_quick = QGroupBox("⚡  Quick Buttons")
        qv = QVBoxLayout(grp_quick)
        btn_recommended = QPushButton("★  Recommended pipeline (aggregates)")
        btn_recommended.setObjectName("primary")
        btn_recommended.setToolTip(
            "Loads the recommended settings:\n"
            "• Background subtraction ON — Gaussian (σ=50)\n"
            "• Noise reduction ON — Gaussian (σ=1.0)\n"
            "• All other steps OFF\n\n"
            "Optimal starting point for protein aggregate analysis in confocal microscopy."
        )
        btn_recommended.clicked.connect(self._set_recommended)
        qv.addWidget(btn_recommended)
        btn_reset_all = QPushButton("↺  Reset All")
        btn_reset_all.setToolTip(
            "Resets all pre-processing steps to factory defaults\n"
            "and shows the original, unprocessed image."
        )
        btn_reset_all.clicked.connect(self._reset)
        qv.addWidget(btn_reset_all)
        lv.addWidget(grp_quick)

        grp_steps = QGroupBox("🔧  Pre-processing Steps")
        sv = QVBoxLayout(grp_steps)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        iv = QVBoxLayout(inner)

        # Normalisatie
        self.chk_norm = QCheckBox("Percentile Normalization")
        self.chk_norm.setChecked(True)
        self.chk_norm.setToolTip(
            "Scales pixel intensities so that pmin% becomes the darkest\n"
            "and pmax% becomes the brightest value.\n\n"
            "Reduces the influence of extreme (outlier) pixels\n"
            "and makes images from different recordings comparable."
        )
        iv.addWidget(self.chk_norm)
        pn = QHBoxLayout()
        pn.addWidget(QLabel("pmin:"))
        self.spn_pmin = QDoubleSpinBox()
        self.spn_pmin.setRange(0, 49); self.spn_pmin.setValue(1.0); self.spn_pmin.setSingleStep(0.5)
        self.spn_pmin.setToolTip(
            "Lower percentile point (%) for normalization.\n"
            "Pixels below this threshold are set to 0.\n"
            "Default: 1.0% — filters noise in the dark background."
        )
        pn.addWidget(self.spn_pmin)
        pn.addWidget(QLabel("pmax:"))
        self.spn_pmax = QDoubleSpinBox()
        self.spn_pmax.setRange(51, 100); self.spn_pmax.setValue(99.9); self.spn_pmax.setSingleStep(0.5)
        self.spn_pmax.setToolTip(
            "Upper percentile point (%) for normalization.\n"
            "Pixels above this threshold are set to 1.\n"
            "Default: 99.9% — prevents a single bright pixel\n"
            "from ruining the scale."
        )
        pn.addWidget(self.spn_pmax)
        iv.addLayout(pn)
        self._add_sep(iv)

        # Achtergrondsubtractie
        self.chk_bg = QCheckBox("Background Subtraction")
        self.chk_bg.setToolTip(
            "Removes the diffuse background glow from the image.\n"
            "Important when the image has an uneven illumination background.\n\n"
            "Recommended for aggregate analysis to avoid false-positive detections."
        )
        iv.addWidget(self.chk_bg)
        bg_type_row = QHBoxLayout()
        bg_type_row.addWidget(QLabel("Method:"))
        self.cmb_bg_method = QComboBox()
        self.cmb_bg_method.addItems(["Rolling Ball (morphological)", "Gaussian"])
        self.cmb_bg_method.setToolTip(
            "Rolling Ball (morphological):\n"
            "  Estimates background via morphological opening (white tophat).\n"
            "  Good for local, uneven backgrounds.\n\n"
            "Gaussian:\n"
            "  Applies a large Gaussian blur as background estimate.\n"
            "  Fast and effective for gradually varying backgrounds.\n"
            "  Recommended for aggregate analysis (σ ≈ 50 px)."
        )
        bg_type_row.addWidget(self.cmb_bg_method)
        iv.addLayout(bg_type_row)
        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("Radius/σ (px):"))
        self.spn_bg_radius = QSpinBox()
        self.spn_bg_radius.setRange(5, 500); self.spn_bg_radius.setValue(30)
        self.spn_bg_radius.setToolTip(
            "Radius (in pixels) for background estimation.\n\n"
            "Rolling Ball: larger = coarser background estimate.\n"
            "Gaussian (σ): larger = more blur, more large structures seen as background.\n\n"
            "Guideline: at least 2–3× the maximum aggregate size.\n"
            "Recommended for Gaussian: σ = 50 px."
        )
        bg_row.addWidget(self.spn_bg_radius)
        iv.addLayout(bg_row)
        self._add_sep(iv)

        # Top-hat
        self.chk_tophat = QCheckBox("Top-Hat Filter")
        self.chk_tophat.setChecked(True)
        self.chk_tophat.setToolTip(
            "Enhances small, bright structures (aggregates) relative\n"
            "to the surrounding background.\n\n"
            "Works by subtracting the morphological opening from the image,\n"
            "leaving only structures smaller than the specified radius."
        )
        iv.addWidget(self.chk_tophat)
        th_mode_row = QHBoxLayout()
        th_mode_row.addWidget(QLabel("Mode:"))
        self.cmb_tophat_mode = QComboBox()
        self.cmb_tophat_mode.addItems(["Single", "Multi-scale (recommended)"])
        self.cmb_tophat_mode.setCurrentIndex(1)
        self.cmb_tophat_mode.setToolTip(
            "Single:\n"
            "  Uses one fixed radius for the top-hat filter.\n\n"
            "Multi-scale (recommended):\n"
            "  Combines three radii simultaneously: r-2, r and r+3.\n"
            "  Detects aggregates of different sizes in one step.\n"
            "  More robust for heterogeneous preparations."
        )
        th_mode_row.addWidget(self.cmb_tophat_mode)
        iv.addLayout(th_mode_row)
        th_row = QHBoxLayout()
        th_row.addWidget(QLabel("Radius (px):"))
        self.spn_tophat = QSpinBox()
        self.spn_tophat.setRange(1, 50); self.spn_tophat.setValue(6)
        self.spn_tophat.setToolTip(
            "Radius of the structuring element (disk) in pixels.\n\n"
            "Choose a value slightly larger than the typical aggregate radius.\n"
            "Too small: background is not well suppressed.\n"
            "Too large: small aggregates are filtered out.\n"
            "Typical value: 4–10 px depending on microscopy resolution."
        )
        th_row.addWidget(self.spn_tophat)
        iv.addLayout(th_row)
        info_th = QLabel("Multi-scale: uses r, r+3, r-2 simultaneously")
        info_th.setStyleSheet("color:#8b949e; font-size:11px;")
        iv.addWidget(info_th)
        self._add_sep(iv)

        # Denoise
        self.chk_denoise = QCheckBox("Noise Reduction")
        self.chk_denoise.setToolTip(
            "Reduces noise in the image before segmentation.\n"
            "Reduces false-positive detections from noise peaks.\n\n"
            "Note: too much suppression can blur small aggregates."
        )
        iv.addWidget(self.chk_denoise)
        dn_mode_row = QHBoxLayout()
        dn_mode_row.addWidget(QLabel("Method:"))
        self.cmb_denoise_mode = QComboBox()
        self.cmb_denoise_mode.addItems(["Gaussian", "Bilateral (preserves edges) ★"])
        self.cmb_denoise_mode.setCurrentIndex(1)
        self.cmb_denoise_mode.setToolTip(
            "Gaussian:\n"
            "  Fast, isotropic blur. Simple but also blurs edges.\n"
            "  Good for high-noise images where edge preservation is less important.\n\n"
            "Bilateral (★ recommended):\n"
            "  Suppresses noise while preserving sharp edges (aggregate boundaries).\n"
            "  Slower but qualitatively better for aggregate detection."
        )
        dn_mode_row.addWidget(self.cmb_denoise_mode)
        iv.addLayout(dn_mode_row)
        dn_row = QHBoxLayout()
        dn_row.addWidget(QLabel("Sigma:"))
        self.spn_sigma = QDoubleSpinBox()
        self.spn_sigma.setRange(0.1, 10.0); self.spn_sigma.setValue(0.8); self.spn_sigma.setSingleStep(0.1)
        self.spn_sigma.setToolTip(
            "Strength of noise reduction (standard deviation of the Gaussian).\n\n"
            "Gaussian: higher sigma = more blurring.\n"
            "Bilateral: higher sigma = larger spatial range of the filter.\n\n"
            "Typical values: 0.5–2.0.\n"
            "Start low (0.8) and increase only if a lot of noise is visible."
        )
        dn_row.addWidget(self.spn_sigma)
        iv.addLayout(dn_row)
        self._add_sep(iv)

        inner.setLayout(iv)
        scroll.setWidget(inner)
        sv.addWidget(scroll)
        lv.addWidget(grp_steps)

        btn_apply = QPushButton("▶  Apply Pre-processing")
        btn_apply.setObjectName("primary")
        btn_apply.setToolTip(
            "Applies all checked pre-processing steps to the current image\n"
            "in order: background subtraction → top-hat → noise reduction → normalization.\n\n"
            "The result is shown on the right and passed to the segmentation tabs."
        )
        btn_apply.clicked.connect(self._apply)
        lv.addWidget(btn_apply)
        lv.addStretch()
        main.addWidget(left)

        right_layout = QVBoxLayout()
        self.canvas_pre = MplCanvas(width=11, height=8)
        nav = NavigationToolbar(self.canvas_pre, self)
        nav.setStyleSheet("background:#f0f3f6; border-bottom: 1px solid #d0d7de; border-radius: 6px 6px 0 0;")
        right_layout.addWidget(nav)
        right_layout.addWidget(self.canvas_pre)
        right_w = QWidget()
        right_w.setLayout(right_layout)
        main.addWidget(right_w)

    def _add_sep(self, layout):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Plain)
        line.setStyleSheet("color: #0f3460;")
        layout.addWidget(line)

    def _set_recommended(self):
        self.chk_norm.setChecked(False)
        self.chk_bg.setChecked(True)
        self.cmb_bg_method.setCurrentIndex(1)
        self.spn_bg_radius.setValue(50)
        self.chk_tophat.setChecked(False)
        self.chk_denoise.setChecked(True)
        self.cmb_denoise_mode.setCurrentIndex(0)
        self.spn_sigma.setValue(1.0)
        self._apply()

    def _get_raw(self):
        return self.viewer.get_current_image()

    def _apply(self):
        raw = self._get_raw()
        if raw is None:
            QMessageBox.warning(self, "No image", "Load an image first.")
            return
        img = raw.copy().astype(np.float32)

        if self.chk_bg.isChecked():
            if self.cmb_bg_method.currentIndex() == 0:
                img = Preprocessor.subtract_background_rolling_ball(img, self.spn_bg_radius.value())
            else:
                img = Preprocessor.subtract_background_gaussian(img, float(self.spn_bg_radius.value()))
        if self.chk_tophat.isChecked():
            img_n = Preprocessor.normalize(img)
            r = self.spn_tophat.value()
            if self.cmb_tophat_mode.currentIndex() == 1:
                img = Preprocessor.multiscale_tophat(img_n, radii=(max(1, r-2), r, r+3))
            else:
                img = Preprocessor.tophat(img_n, r)
        if self.chk_denoise.isChecked():
            if self.cmb_denoise_mode.currentIndex() == 1:
                img = Preprocessor.denoise_bilateral(
                    Preprocessor.normalize(img),
                    sigma_color=0.12, sigma_spatial=self.spn_sigma.value() * 2
                )
            else:
                img = Preprocessor.denoise_gaussian(img, self.spn_sigma.value())
        if self.chk_norm.isChecked():
            img = Preprocessor.normalize(img, self.spn_pmin.value(), self.spn_pmax.value())

        self.result_img = img
        self._show_comparison(raw, img)
        self.preprocessed.emit(img)

    def _reset(self):
        raw = self._get_raw()
        if raw is None:
            return
        self.result_img = raw.copy().astype(np.float32)
        self._show_comparison(raw, raw)
        self.preprocessed.emit(self.result_img)

    def _show_comparison(self, before, after):
        self.canvas_pre.fig.clf()
        axes = self.canvas_pre.fig.subplots(1, 2)
        for ax, img, title in zip(axes, [before, after], ["Original", "After pre-processing"]):
            ax.imshow(img, cmap="hot", aspect="equal", interpolation="nearest")
            ax.set_title(title, color="#79c0ff")
            ax.axis("off")
            ax.set_facecolor("#0d1117")
        self.canvas_pre.fig.tight_layout()
        self.canvas_pre.draw_idle()

    def get_image(self) -> Optional[np.ndarray]:
        if self.result_img is not None:
            return self.result_img
        return self._get_raw()


# ═══════════════════════════════════════════════════════════════════════════════
#  STATISTICS DIALOG
# ═══════════════════════════════════════════════════════════════════════════════
class StatsDialog(QDialog):
    def __init__(self, result: SegmentationResult, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Statistieken — {result.method_name}")
        self.resize(900, 650)
        self.setStyleSheet(DARK_THEME)
        lyt = QVBoxLayout(self)

        if result.properties:
            keys = list(result.properties[0].keys())
            tbl = QTableWidget(len(result.properties), len(keys))
            tbl.setHorizontalHeaderLabels(keys)
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            tbl.setAlternatingRowColors(True)
            for row, prop in enumerate(result.properties):
                for col, k in enumerate(keys):
                    v = prop.get(k, "")
                    item = QTableWidgetItem(f"{v:.3f}" if isinstance(v, float) else str(v))
                    item.setTextAlignment(Qt.AlignCenter)
                    tbl.setItem(row, col, item)
            lyt.addWidget(tbl)

        fig = Figure(figsize=(8, 3), facecolor="#0d1117")
        canvas = FigureCanvas(fig)
        if result.properties:
            areas = [p.get("area_px2", 0) for p in result.properties]
            ax = fig.add_subplot(121)
            ax.hist(areas, bins=30, color="#e94560", edgecolor="#161b22")
            ax.set_title("Area (px²)", color="#79c0ff")
            ax.set_facecolor("#0d1117"); ax.tick_params(colors="#79c0ff")
            intns = [p.get("mean_intensity", 0) for p in result.properties]
            ax2 = fig.add_subplot(122)
            ax2.hist(intns, bins=30, color="#79c0ff", edgecolor="#161b22")
            ax2.set_title("Intensity distribution", color="#79c0ff")
            ax2.set_facecolor("#0d1117"); ax2.tick_params(colors="#79c0ff")
            fig.tight_layout()
        lyt.addWidget(canvas)

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lyt.addWidget(bb)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — CELLPOSE CELL BODY SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
class CellposeWorker(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, img: np.ndarray, kanaal: int,
                 model_type: str, diameter,
                 flow_threshold: float, cellprob_threshold: float,
                 z_count: int, parent=None):
        super().__init__(parent)
        self.img               = img
        self.kanaal            = kanaal
        self.model_type        = model_type
        self.diameter          = diameter
        self.flow_threshold    = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.z_count           = z_count

    def run(self):
        try:
            import torch
            from cellpose import models as cp_models
            from scipy.ndimage import binary_fill_holes
            import time

            use_gpu = True
            t0 = time.perf_counter()

            self.progress.emit("⏳  Loading Cellpose model…")
            model = cp_models.CellposeModel(gpu=use_gpu, model_type=self.model_type)
            
            img2d = self.img
            if img2d.ndim == 3:
                img2d = img2d.max(axis=0)

            self.progress.emit(
                f"⏳  Segmentation on channel {self.kanaal} "
                f"(shape {img2d.shape}, diameter={self.diameter})…"
            )

            masks, flows, styles = model.eval(
                [img2d],
                diameter=self.diameter,
                channels=[0, 0],
                flow_threshold=self.flow_threshold,
                cellprob_threshold=self.cellprob_threshold,
            )
            mask = masks[0]

            self.progress.emit("⏳  Filling cell nucleus holes…")
            mask_filled = np.zeros_like(mask)
            for cel_id in range(1, int(mask.max()) + 1):
                cel = mask == cel_id
                mask_filled[binary_fill_holes(cel)] = cel_id
            mask = mask_filled

            elapsed = time.perf_counter() - t0
            n_cells = int(mask.max())

            result = {
                "mask":     mask,
                "binary":   (mask > 0),
                "n_cells":  n_cells,
                "elapsed":  elapsed,
                "model":    self.model_type,
                "diameter": self.diameter,
            }
            self.finished.emit(result)

        except Exception:
            self.error.emit(traceback.format_exc())

class CellposeTab(QWidget):
    mask_ready = pyqtSignal(object)

    def __init__(self, preprocess_tab: PreprocessTab, viewer_tab=None, parent=None):
        super().__init__(parent)
        self.prep        = preprocess_tab
        self.viewer      = viewer_tab
        self.cell_mask: Optional[np.ndarray] = None
        self._worker: Optional[CellposeWorker] = None
        self._build_ui()

    def get_cell_mask(self) -> Optional[np.ndarray]:
        return self.cell_mask

    def _get_image(self) -> Optional[np.ndarray]:
        if self.viewer is not None:
            return self.viewer.get_current_image()
        return self.prep.get_image()

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setSpacing(6)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(340)
        left_inner = QWidget()
        lv = QVBoxLayout(left_inner)
        lv.setSpacing(10)

        if not HAS_CELLPOSE:
            warn = QLabel(
                "⚠  Cellpose is not installed.\n"
                "Install with:\n  pip install cellpose"
            )
            warn.setStyleSheet("color:#ff8800; font-weight:bold; padding:10px;")
            warn.setWordWrap(True)
            lv.addWidget(warn)

        grp_model = QGroupBox("🔬  Model & Channel")
        mf = QFormLayout(grp_model)

        self.spn_kanaal = QSpinBox()
        self.spn_kanaal.setRange(0, 15)
        self.spn_kanaal.setValue(2)
        mf.addRow("Segmentation channel:", self.spn_kanaal)

        self.cmb_model = QComboBox()
        self.cmb_model.addItems(["cyto2", "cyto", "nuclei", "cyto3"])
        self.cmb_model.setCurrentText("cyto2")
        self.cmb_model.setToolTip(
            "Choose the Cellpose model suited to your preparation:\n\n"
            "  • cyto2   — improved cytoplasm model (recommended for cells)\n"
            "  • cyto    — original cytoplasm model\n"
            "  • nuclei  — optimised for cell nuclei (DAPI/Hoechst)\n"
            "  • cyto3   — latest generation cytoplasm model\n\n"
            "Try cyto2 as a starting point for most cell lines."
        )
        mf.addRow("Model:", self.cmb_model)

        self.spn_diameter = QSpinBox()
        self.spn_diameter.setRange(0, 999)
        self.spn_diameter.setValue(80)
        self.spn_diameter.setSpecialValueText("Auto")
        self.spn_diameter.setToolTip(
            "Expected cell diameter in pixels.\n\n"
            "Set to 0 for automatic estimation by Cellpose.\n"
            "For manual input: measure a representative cell in the viewer\n"
            "and enter the diameter in pixels.\n\n"
            "Too small: cells are split.\n"
            "Too large: multiple cells are merged."
        )
        mf.addRow("Cell diameter (px, 0=auto):", self.spn_diameter)

        lv.addWidget(grp_model)

        grp_thr = QGroupBox("⚙  Segmentation Parameters")
        tf = QFormLayout(grp_thr)

        self.spn_flow = QDoubleSpinBox()
        self.spn_flow.setRange(0.1, 1.0)
        self.spn_flow.setSingleStep(0.05)
        self.spn_flow.setValue(0.8)
        self.spn_flow.setToolTip(
            "Flow threshold: maximum allowed error in optical flow fields.\n\n"
            "Lower (e.g. 0.4): accepts more imperfect segmentations → more cells\n"
            "Higher (e.g. 0.9): stricter, only well-formed masks → fewer cells\n\n"
            "Default: 0.8 — good starting point for most preparations.\n"
            "Lower if too few cells are found."
        )
        tf.addRow("Flow threshold:", self.spn_flow)

        self.spn_cellprob = QDoubleSpinBox()
        self.spn_cellprob.setRange(-8.0, 6.0)
        self.spn_cellprob.setSingleStep(0.5)
        self.spn_cellprob.setValue(-4.0)
        self.spn_cellprob.setToolTip(
            "Cell probability threshold: minimum predicted probability to be included as a cell.\n\n"
            "Lower (e.g. -6.0): more pixels seen as cell → larger masks\n"
            "Higher (e.g. 0.0): only the most certain areas → smaller/fewer masks\n\n"
            "Default: -4.0 — liberal setting that also includes weakly labelled cells.\n"
            "Increase if too much background is detected as cell."
        )
        tf.addRow("Cellprob threshold:", self.spn_cellprob)

        btn_reset = QPushButton("↺  Restore Defaults")
        btn_reset.clicked.connect(self._reset_params)
        tf.addRow(btn_reset)

        lv.addWidget(grp_thr)

        grp_info = QGroupBox("ℹ  How It Works")
        iv = QVBoxLayout(grp_info)
        lbl_info = QLabel(
            "Cellpose computes a <b>max-intensity projection</b> across all "
            "Z-slices on the chosen channel and segments the cell bodies on it.\n\n"
            "The resulting binary mask can be used in further steps "
            "to filter out the background."
        )
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet("color:#8b949e; font-size:11px; padding:4px;")
        iv.addWidget(lbl_info)
        lv.addWidget(grp_info)

        self.btn_run = QPushButton("▶  Segment Cell Bodies")
        self.btn_run.setObjectName("primary")
        self.btn_run.setEnabled(HAS_CELLPOSE)
        self.btn_run.setToolTip(
            "Starts Cellpose cell body segmentation on the current image.\n\n"
            "Cellpose computes a max projection over all Z-layers and segments\n"
            "cell bodies automatically based on the chosen model.\n\n"
            "The resulting binary mask is used to restrict aggregate detections\n"
            "to the cell body (background is masked).\n\n"
            "Required: pip install cellpose"
        )
        self.btn_run.clicked.connect(self._run)
        lv.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        lv.addWidget(self.progress)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color:#58a6ff; font-size:11px;")
        self.lbl_status.setWordWrap(True)
        lv.addWidget(self.lbl_status)

        grp_stats = QGroupBox("📊  Result")
        sv = QVBoxLayout(grp_stats)
        self.txt_stats = QTextEdit()
        self.txt_stats.setReadOnly(True)
        self.txt_stats.setMinimumHeight(80)
        self.txt_stats.setMaximumHeight(140)
        sv.addWidget(self.txt_stats)
        lv.addWidget(grp_stats)

        btn_export = QPushButton("💾  Export Cell Mask as TIFF")
        btn_export.setToolTip(
            "Saves the segmented cell mask as a TIFF file.\n\n"
            "The mask is binary: white (255) = cell body, black (0) = background.\n"
            "Can be reloaded later or used for batch processing."
        )
        btn_export.clicked.connect(self._export_mask)
        lv.addWidget(btn_export)

        lv.addStretch()
        left_inner.setLayout(lv)
        left_scroll.setWidget(left_inner)
        main.addWidget(left_scroll)

        right_w = QWidget()
        rv = QVBoxLayout(right_w)
        rv.setSpacing(4)
        self.canvas = MplCanvas(width=11, height=9)
        nav = NavigationToolbar(self.canvas, self)
        nav.setStyleSheet("background:#f0f3f6; border-bottom:1px solid #d0d7de; border-radius:6px 6px 0 0;")
        rv.addWidget(nav)
        rv.addWidget(self.canvas)
        main.addWidget(right_w)

    def _reset_params(self):
        self.spn_flow.setValue(0.8)
        self.spn_cellprob.setValue(-4.0)
        self.spn_diameter.setValue(80)

    def _run(self):
        img = self._get_image()
        if img is None:
            QMessageBox.warning(self, "No image", "Load an image first.")
            return

        diameter = self.spn_diameter.value() or None

        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.lbl_status.setText("Running…")
        self.txt_stats.clear()

        self._worker = CellposeWorker(
            img=img,
            kanaal=self.spn_kanaal.value(),
            model_type=self.cmb_model.currentText(),
            diameter=diameter,
            flow_threshold=self.spn_flow.value(),
            cellprob_threshold=self.spn_cellprob.value(),
            z_count=getattr(img, "shape", (1,))[0] if img.ndim == 3 else 1,
        )
        self._worker.finished.connect(self._on_result)
        self._worker.progress.connect(self.lbl_status.setText)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_result(self, result: dict):
        self.cell_mask = result["binary"]

        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)

        n  = result["n_cells"]
        t  = result["elapsed"]
        diam = result["diameter"] if result["diameter"] else "auto"

        self.lbl_status.setText(f"✅  Done — {n} cells detected in {t:.1f}s")
        self.txt_stats.setPlainText(
            f"Model          : {result['model']}\n"
            f"Diameter       : {diam} px\n"
            f"Flow threshold : {self.spn_flow.value():.2f}\n"
            f"Cellprob thr.  : {self.spn_cellprob.value():.1f}\n"
            f"Gedetect. cellen: {n}\n"
            f"Rekentijd      : {t:.2f} s\n"
            f"Mask covers    : {self.cell_mask.mean()*100:.1f}% of the image"
        )

        self._display_result(result)
        self.mask_ready.emit(self.cell_mask)

    def _on_error(self, err: str):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self.lbl_status.setText("❌  Error during segmentation")
        QMessageBox.critical(self, "Cellpose error", err)

    def _display_result(self, result: dict):
        img     = self._get_image()
        mask    = result["mask"]
        binary  = result["binary"]

        img2d = img.max(axis=0) if img.ndim == 3 else img
        lo, hi = np.percentile(img2d, [1, 99.9])
        img_norm = np.clip((img2d.astype(float) - lo) / max(hi - lo, 1e-6), 0, 1)

        img_masked = img_norm.copy()
        img_masked[~binary] = 0

        self.canvas.fig.clf()
        axes = self.canvas.fig.subplots(1, 3)

        TITLE  = dict(color="#79c0ff", fontsize=10, fontweight="bold", pad=6)
        BG     = "#0d1117"

        axes[0].imshow(img_norm, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        axes[0].set_title("Max projection (input)", **TITLE)

        axes[1].imshow(mask, cmap="nipy_spectral", interpolation="nearest")
        axes[1].set_title(f"Cel maskers — {result['n_cells']} cellen", **TITLE)

        axes[2].imshow(img_masked, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        axes[2].set_title("Cell bodies only", **TITLE)

        for ax in axes:
            ax.axis("off")
            ax.set_facecolor(BG)

        self.canvas.fig.set_facecolor(BG)
        self.canvas.fig.tight_layout(pad=0.5)
        self.canvas.draw_idle()

    def _export_mask(self):
        if self.cell_mask is None:
            QMessageBox.warning(self, "No mask", "Run cell body detection first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save cell mask", "cellpose_mask.tif",
            "TIFF (*.tif);;All files (*)"
        )
        if not path:
            return
        tifffile.imwrite(path, self.cell_mask.astype(np.uint8) * 255)
        QMessageBox.information(self, "Saved", f"Celmasker opgeslagen:\n{path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENSEMBLE PREPROCESSOR & INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class EnsemblePreprocessor:
    @staticmethod
    def pct_norm(img: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(img, [1.0, 99.9])
        if hi <= lo:
            return np.zeros_like(img, dtype=np.float32)
        return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)

    @staticmethod
    def build(img: np.ndarray) -> np.ndarray:
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


class EnsembleEngine:
    @staticmethod
    def _predict_tta(model, processed: np.ndarray, device,
                     tile: int = 256, overlap: int = 32) -> np.ndarray:
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
    def run(img: np.ndarray, model_dir: str,
            threshold_override: Optional[float],
            min_area: int, max_area: int,
            device_name: str = "auto",
            use_tta: bool = True) -> SegmentationResult:
        import torch
        t0 = time.time()

        if device_name == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_name)

        model_files = sorted(Path(model_dir).glob("model_fold*.pth"))
        if not model_files:
            raise FileNotFoundError(
                f"No model_fold*.pth files found in:\n{model_dir}"
            )

        processed = EnsemblePreprocessor.build(img)

        all_proba: List[np.ndarray] = []
        all_thr:   List[float]      = []

        for mf in model_files:
            ckpt = torch.load(str(mf), map_location=device, weights_only=False)
            encoder  = ckpt.get("encoder", "resnet34")
            version  = ckpt.get("version", "v7")
            attention_type = "scse" if version in ("v8", "v9") else None
            
            if HAS_SMP:
                model = smp.Unet(
                    encoder_name=encoder,
                    encoder_weights=None,
                    in_channels=3,
                    classes=1,
                    decoder_attention_type=attention_type
                ).to(device)
                model.load_state_dict(ckpt["state_dict"])
            else:
                raise ImportError(
                    "segmentation_models_pytorch is not installed.\n"
                    "Install with:  pip install segmentation-models-pytorch"
                )

            tile_sz  = ckpt.get("tile", 256)
            opt_thr  = ckpt.get("best_thr", 0.5)

            proba = EnsembleEngine._predict_tta(
                model, processed, device, tile=tile_sz
            )
            all_proba.append(proba)
            all_thr.append(opt_thr)

        ensemble_proba = np.mean(all_proba, axis=0)
        avg_thr = float(np.mean(all_thr))
        thr = threshold_override if threshold_override is not None else avg_thr

        binary = (ensemble_proba >= thr)
        binary = remove_small_objects(binary, min_size=max(1, min_area))
        binary = binary_closing(binary, disk(1))

        lbl   = label(binary)
        lbl   = _filter_by_morphology(lbl, min_area, max_area)
        props = _label_to_props(lbl, img.astype(np.float32))

        return SegmentationResult(
            method_name=(f"Ensemble DL ({len(model_files)} folds, thr={thr:.2f})"),
            params=dict(model_dir=model_dir, n_models=len(model_files),
                        threshold=thr, avg_thr=avg_thr),
            label_image=lbl,
            n_objects=int(lbl.max()),
            properties=props,
            time_seconds=time.time() - t0
        )


class EnsembleDLWorker(QThread):
    finished   = pyqtSignal(object)
    error      = pyqtSignal(str)
    progress   = pyqtSignal(str)

    def __init__(self, img, model_dir, threshold_override,
                 min_area, max_area, device_name, use_tta):
        super().__init__()
        self.img                = img
        self.model_dir          = model_dir
        self.threshold_override = threshold_override
        self.min_area           = min_area
        self.max_area           = max_area
        self.device_name        = device_name
        self.use_tta            = use_tta

    def run(self):
        try:
            result = EnsembleEngine.run(
                self.img, self.model_dir,
                threshold_override=self.threshold_override,
                min_area=self.min_area,
                max_area=self.max_area,
                device_name=self.device_name,
                use_tta=self.use_tta
            )
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — DEEP LEARNING SEGMENTATION (Ensemble Only)
# ═══════════════════════════════════════════════════════════════════════════════
class DeepLearningTab(QWidget):
    result_ready = pyqtSignal(object)

    def __init__(self, preprocess_tab: PreprocessTab, viewer_tab=None):
        super().__init__()
        self.prep                   = preprocess_tab
        self.viewer                 = viewer_tab
        self.current_result: Optional[SegmentationResult] = None
        self._ensemble_worker: Optional[EnsembleDLWorker] = None
        self._model_dir             = ""
        self._build_ui()

    def _get_image(self) -> Optional[np.ndarray]:
        if self.viewer is not None:
            return self.viewer.get_current_image()
        return self.prep.get_image()

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setSpacing(6)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(360)
        left_inner  = QWidget()
        lv = QVBoxLayout(left_inner)
        lv.setSpacing(8)

        if not HAS_TORCH:
            warn = QLabel(
                "⚠  PyTorch is niet geïnstalleerd.\n"
                "Install with:\n  pip install torch torchvision"
            )
            warn.setStyleSheet("color:#ff8800; font-weight:bold; padding:8px;")
            warn.setWordWrap(True)
            lv.addWidget(warn)

        if not HAS_SMP:
            warn_smp = QLabel(
                "⚠  segmentation_models_pytorch niet gevonden.\n"
                "Ensemble vereist:  pip install segmentation-models-pytorch"
            )
            warn_smp.setStyleSheet("color:#ff8800; font-size:10px; padding:6px;")
            warn_smp.setWordWrap(True)
            lv.addWidget(warn_smp)

        grp_ens = QGroupBox("🗂  Ensemble Models (folder with model_fold*.pth)")
        ev = QVBoxLayout(grp_ens)

        ens_path_row = QHBoxLayout()
        self.lbl_ensemble_path = QLabel("No folder selected")
        self.lbl_ensemble_path.setStyleSheet("color:#8b949e; font-size:11px;")
        self.lbl_ensemble_path.setWordWrap(True)
        ens_path_row.addWidget(self.lbl_ensemble_path, stretch=1)
        btn_load_dir = QPushButton("📂  Load Model Folder")
        btn_load_dir.clicked.connect(self._load_model_dir)
        ens_path_row.addWidget(btn_load_dir)
        ev.addLayout(ens_path_row)

        self.lbl_ensemble_info = QLabel("")
        self.lbl_ensemble_info.setStyleSheet("color:#58a6ff; font-size:11px;")
        self.lbl_ensemble_info.setWordWrap(True)
        ev.addWidget(self.lbl_ensemble_info)

        ens_thr_row = QHBoxLayout()
        self.chk_auto_thr = QCheckBox("Auto-threshold (averaged from models)")
        self.chk_auto_thr.setChecked(True)
        self.chk_auto_thr.setToolTip(
            "If checked: automatically uses the optimal threshold\n"
            "determined during training of each model.\n"
            "The final threshold is the average across all folds.\n\n"
            "Recommended for most situations."
        )
        self.chk_auto_thr.toggled.connect(self._on_auto_thr_toggled)
        ens_thr_row.addWidget(self.chk_auto_thr)
        ev.addLayout(ens_thr_row)

        ens_thr2_row = QHBoxLayout()
        ens_thr2_row.addWidget(QLabel("Fixed threshold:"))
        self.spn_ens_threshold = QDoubleSpinBox()
        self.spn_ens_threshold.setRange(0.01, 0.99)
        self.spn_ens_threshold.setValue(0.5)
        self.spn_ens_threshold.setSingleStep(0.05)
        self.spn_ens_threshold.setEnabled(False)
        self.spn_ens_threshold.setToolTip(
            "Manual threshold for the ensemble probability map (0.01–0.99).\n\n"
            "Pixels with a predicted probability ≥ threshold are labelled as aggregate.\n\n"
            "Lower (e.g. 0.3): more/larger detections, more false positives.\n"
            "Higher (e.g. 0.7): fewer/smaller detections, fewer false positives.\n\n"
            "Only active when 'Auto-threshold' is unchecked."
        )
        ens_thr2_row.addWidget(self.spn_ens_threshold)
        ev.addLayout(ens_thr2_row)

        self.chk_ens_tta = QCheckBox("Test-Time Augmentation (TTA, 8×)")
        self.chk_ens_tta.setChecked(True)
        self.chk_ens_tta.setToolTip(
            "Averages 8 augmentations (4 rotations + 2 flips).\n"
            "Improves quality but is ~8× slower."
        )
        ev.addWidget(self.chk_ens_tta)

        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Device:"))
        self.cmb_device = QComboBox()
        self.cmb_device.addItems(["auto", "cpu", "cuda", "mps"])
        self.cmb_device.setToolTip(
            "Choose the hardware on which inference will run:\n\n"
            "  • auto  — automatically selects GPU (cuda) if available, otherwise cpu\n"
            "  • cpu   — use the processor (slower, always available)\n"
            "  • cuda  — use an NVIDIA GPU (much faster, requires CUDA driver)\n"
            "  • mps   — use Apple Silicon GPU (M1/M2/M3 Mac)\n\n"
            "For CUDA errors: set to 'cpu' as a temporary fix."
        )
        device_row.addWidget(self.cmb_device)
        ev.addLayout(device_row)

        lv.addWidget(grp_ens)

        grp_post = QGroupBox("🔧  Post-processing")
        pf = QFormLayout(grp_post)

        self.spn_min_area = QSpinBox()
        self.spn_min_area.setRange(1, 9999)
        self.spn_min_area.setValue(5)
        self.spn_min_area.setToolTip(
            "Minimum area (in pixels²) of a detected object.\n\n"
            "Objects smaller than this value are removed as noise.\n\n"
            "Too low: noise peaks are included as false positives.\n"
            "Too high: small genuine aggregates are filtered out.\n\n"
            "Typical value: 5–50 px² depending on microscopy resolution."
        )
        pf.addRow("Min area (px²):", self.spn_min_area)

        self.spn_max_area = QSpinBox()
        self.spn_max_area.setRange(1, 999999)
        self.spn_max_area.setValue(50000)
        self.spn_max_area.setToolTip(
            "Maximum area (in pixels²) of a detected object.\n\n"
            "Objects larger than this value are removed.\n"
            "Prevents large artefacts (e.g. dead cells, debris) from being included.\n\n"
            "Set high if you also want to detect large aggregate clusters."
        )
        pf.addRow("Max area (px²):", self.spn_max_area)

        lv.addWidget(grp_post)

        grp_ov = QGroupBox("🖍  Overlay Options")
        ovf = QFormLayout(grp_ov)
        self.chk_show_circles = QCheckBox("Draw contours")
        self.chk_show_circles.setChecked(True)
        self.chk_show_circles.setToolTip(
            "Draws the outline of each detected object over the image.\n"
            "Makes the exact boundaries of the segmentation visible."
        )
        self.chk_show_numbers = QCheckBox("Show numbers")
        self.chk_show_numbers.setChecked(True)
        self.chk_show_numbers.setToolTip(
            "Shows the ID number of each object at the centre of the contour.\n"
            "Useful for locating specific objects in the CSV export."
        )
        self.chk_show_fill    = QCheckBox("Filled region")
        self.chk_show_fill.setChecked(True)
        self.chk_show_fill.setToolTip(
            "Fills the area of each detected object with a semi-transparent colour.\n"
            "Gives a better overview of the total segmentation than contours alone."
        )
        self.cmb_circle_color = QComboBox()
        self.cmb_circle_color.addItems(["#00ffcc","#ff4466","#ffff00","#ffffff","#00aaff","#ff8800"])
        self.cmb_circle_color.setToolTip(
            "Color of the drawn contours:\n"
            "  #00ffcc — cyan (default, clearly visible on dark background)\n"
            "  #ff4466 — red/pink\n"
            "  #ffff00 — yellow\n"
            "  #ffffff — white\n"
            "  #00aaff — blue\n"
            "  #ff8800 — orange"
        )
        self.spn_circle_lw  = QDoubleSpinBox()
        self.spn_circle_lw.setRange(0.3, 5); self.spn_circle_lw.setValue(1.2)
        self.spn_circle_lw.setToolTip(
            "Line width of the drawn contours in points.\n\n"
            "Thinner (0.3–1.0): less conspicuous, more detail visible.\n"
            "Thicker (2.0–5.0): better visible for small objects or exported images."
        )
        self.spn_font_size  = QDoubleSpinBox()
        self.spn_font_size.setRange(3, 16); self.spn_font_size.setValue(6.5)
        self.spn_font_size.setToolTip(
            "Font size of the object ID numbers in points.\n\n"
            "Adjust based on the size of objects in the image:\n"
            "small for small aggregates (4–6), larger for cells (8–12)."
        )
        self.cmb_cmap       = QComboBox()
        self.cmb_cmap.addItems(["hot","gray","inferno","magma","viridis","plasma"])
        self.cmb_cmap.setToolTip(
            "Color map for the background image in the results display.\n\n"
            "  • hot     — black → red → white (good for fluorescence microscopy)\n"
            "  • gray    — grayscale\n"
            "  • inferno — black → purple → orange → white\n"
            "  • magma   — black → purple → pink → white\n"
            "  • viridis — dark blue → green → yellow (colorblind-safe)\n"
            "  • plasma  — blue → purple → yellow"
        )
        ovf.addWidget(self.chk_show_circles)
        ovf.addWidget(self.chk_show_numbers)
        ovf.addWidget(self.chk_show_fill)
        ovf.addRow("Color:",         self.cmb_circle_color)
        ovf.addRow("Line width:",   self.spn_circle_lw)
        ovf.addRow("Font size:", self.spn_font_size)
        ovf.addRow("Background:",   self.cmb_cmap)
        lv.addWidget(grp_ov)

        self.btn_run_ensemble = QPushButton("▶  Run Ensemble Segmentation")
        self.btn_run_ensemble.setObjectName("primary")
        self.btn_run_ensemble.setToolTip(
            "Starts the ensemble deep learning segmentation on the current image.\n\n"
            "The algorithm:\n"
            "  1. Loads all model_fold*.pth files from the selected folder\n"
            "  2. Runs inference on each model (optionally with TTA)\n"
            "  3. Averages the probability maps of all models\n"
            "  4. Thresholds the averaged probability map\n"
            "  5. Applies morphological filtering (min/max area)\n\n"
            "Note: this may take several minutes on a CPU."
        )
        self.btn_run_ensemble.clicked.connect(self._run_ensemble)
        lv.addWidget(self.btn_run_ensemble)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        lv.addWidget(self.progress)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color:#58a6ff; font-size:11px;")
        self.lbl_status.setWordWrap(True)
        lv.addWidget(self.lbl_status)

        grp_stats = QGroupBox("📊  Statistics")
        sv = QVBoxLayout(grp_stats)
        self.txt_stats = QTextEdit()
        self.txt_stats.setReadOnly(True)
        self.txt_stats.setMinimumHeight(90)
        sv.addWidget(self.txt_stats)
        lv.addWidget(grp_stats)

        btn_csv = QPushButton("💾  Export CSV")
        btn_csv.setToolTip(
            "Exports the properties of all detected objects to a CSV file.\n\n"
            "Columns per object:\n"
            "  • id, x, y         — identification and position (centroid)\n"
            "  • area_px2         — area in pixels²\n"
            "  • mean/max_intensity — average and maximum pixel intensity\n"
            "  • eccentricity     — degree of ellipse shape (0=circle, 1=line segment)\n"
            "  • perimeter        — perimeter length in pixels\n"
            "  • solidity         — fill ratio (convex hull)\n"
            "  • radius_px        — equivalent radius"
        )
        btn_csv.clicked.connect(self._export_csv)
        lv.addWidget(btn_csv)

        btn_img = QPushButton("🖼  Export annotated image")
        btn_img.setToolTip(
            "Saves an image with the original image and the annotated\n"
            "segmentation result side by side (PNG or TIFF).\n\n"
            "The image shows contours and colouring as configured\n"
            "in the Overlay Options."
        )
        btn_img.clicked.connect(self._export_image)
        lv.addWidget(btn_img)

        lv.addStretch()
        left_inner.setLayout(lv)
        left_scroll.setWidget(left_inner)
        main.addWidget(left_scroll)

        right_w = QWidget()
        rv = QVBoxLayout(right_w)
        rv.setSpacing(4)
        self.canvas = MplCanvas(width=11, height=9)
        nav = NavigationToolbar(self.canvas, self)
        nav.setStyleSheet("background:#f0f3f6; border-bottom: 1px solid #d0d7de; border-radius: 6px 6px 0 0;")
        rv.addWidget(nav)
        rv.addWidget(self.canvas)
        main.addWidget(right_w)

    def _on_auto_thr_toggled(self, checked: bool):
        self.spn_ens_threshold.setEnabled(not checked)

    def _load_model_dir(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder with model_fold*.pth files", ""
        )
        if not folder:
            return
        model_files = sorted(Path(folder).glob("model_fold*.pth"))
        if not model_files:
            QMessageBox.warning(
                self, "No models found",
                f"No model_fold*.pth files found in:\n{folder}"
            )
            return
        self._model_dir = folder
        self.lbl_ensemble_path.setText(f"✅  {Path(folder).name}")
        self.lbl_ensemble_info.setText(f"{len(model_files)} models found")

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
                    self.spn_ens_threshold.setValue(avg)
                    self.lbl_ensemble_info.setText(
                        f"{len(model_files)} models  |  avg. threshold: {avg:.2f}"
                    )
            except Exception:
                pass

    def _run_ensemble(self):
        if not self._model_dir:
            QMessageBox.warning(self, "No folder", "Load a model folder first.")
            return
        if not HAS_SMP:
            QMessageBox.critical(
                self, "Package missing",
                "Install segmentation_models_pytorch:\n"
                "  pip install segmentation-models-pytorch"
            )
            return
        img = self._get_image()
        if img is None:
            QMessageBox.warning(self, "No image", "Load and process an image first.")
            return

        thr_override = None if self.chk_auto_thr.isChecked() \
                       else self.spn_ens_threshold.value()

        self.btn_run_ensemble.setEnabled(False)
        self.progress.setVisible(True)
        self.lbl_status.setText("Running ensemble inference…")

        self._ensemble_worker = EnsembleDLWorker(
            img=img,
            model_dir=self._model_dir,
            threshold_override=thr_override,
            min_area=self.spn_min_area.value(),
            max_area=self.spn_max_area.value(),
            device_name=self.cmb_device.currentText(),
            use_tta=self.chk_ens_tta.isChecked()
        )
        self._ensemble_worker.finished.connect(self._on_ensemble_result)
        self._ensemble_worker.error.connect(self._on_error)
        self._ensemble_worker.start()

    def _on_ensemble_result(self, result: SegmentationResult):
        self.current_result = result
        self.btn_run_ensemble.setEnabled(True)
        self.progress.setVisible(False)
        self.lbl_status.setText(
            f"Done: {result.n_objects} objects  |  {result.time_seconds:.2f}s"
        )
        self._display_result(result)
        self._show_stats(result)
        self.result_ready.emit(result)

    def _on_error(self, err: str):
        self.btn_run_ensemble.setEnabled(True)
        self.progress.setVisible(False)
        self.lbl_status.setText("❌  Error during inference")
        QMessageBox.critical(self, "Deep Learning error", err)

    def _display_result(self, result: SegmentationResult):
        img  = self._get_image()
        cmap = self.cmb_cmap.currentText()
        circle_color = self.cmb_circle_color.currentText()
        show_circles = self.chk_show_circles.isChecked()
        show_numbers = self.chk_show_numbers.isChecked()
        show_fill    = self.chk_show_fill.isChecked()
        lw    = self.spn_circle_lw.value()
        fsize = self.spn_font_size.value()

        self.canvas.fig.clf()
        axes = self.canvas.fig.subplots(1, 2)

        axes[0].imshow(img, cmap=cmap, aspect="equal", interpolation="nearest")
        axes[0].set_title("Pre-processed input", color="#79c0ff", fontsize=10)
        axes[0].axis("off"); axes[0].set_facecolor("#0d1117")

        if show_fill and result.label_image is not None and result.label_image.max() > 0:
            overlay = overlay_labels_on_image(img, result.label_image, alpha=0.35)
            axes[1].imshow(overlay, aspect="equal", interpolation="nearest")
        else:
            axes[1].imshow(img, cmap=cmap, aspect="equal", interpolation="nearest")

        if show_circles and result.properties:
            draw_actual_contours(
                axes[1], result, circle_color=circle_color,
                text_color="#ffffff", circle_lw=lw,
                font_size=fsize, show_numbers=show_numbers
            )

        axes[1].set_title(
            f"{result.method_name}   N = {result.n_objects}   t = {result.time_seconds:.2f}s",
            color="#79c0ff", fontsize=10
        )
        axes[1].axis("off"); axes[1].set_facecolor("#0d1117")
        self.canvas.fig.tight_layout(pad=0.5)
        self.canvas.draw_idle()

    def _show_stats(self, result: SegmentationResult):
        if not result.properties:
            self.txt_stats.setPlainText(f"Number of objects: {result.n_objects}\nNo properties.")
            return
        areas = [p.get("area_px2", 0) for p in result.properties]
        ints  = [p.get("mean_intensity", 0) for p in result.properties]
        rads  = [p.get("radius_px", 0) for p in result.properties]
        self.txt_stats.setPlainText(
            f"Method:         {result.method_name}\n"
            f"Aantal obj.:    {result.n_objects}\n"
            f"Tijd:           {result.time_seconds:.3f} s\n"
            f"\n── Area (px²) ──\n"
            f"  Gemiddeld: {np.mean(areas):.1f}\n"
            f"  Mediaan:   {np.median(areas):.1f}\n"
            f"  Min/Max:   {np.min(areas):.0f} / {np.max(areas):.0f}\n"
            f"\n── Radius (px) ──\n"
            f"  Gemiddeld: {np.mean(rads):.1f}\n"
            f"  Mediaan:   {np.median(rads):.1f}\n"
            f"\n── Mean intensity ──\n"
            f"  Gemiddeld: {np.mean(ints):.4f}\n"
            f"  Mediaan:   {np.median(ints):.4f}\n"
        )

    def _export_csv(self):
        if not self.current_result or not self.current_result.properties:
            QMessageBox.warning(self, "No data", "Run segmentation first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV (*.csv)")
        if not path:
            return
        props = self.current_result.properties
        keys  = list(props[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(props)
        QMessageBox.information(self, "Saved", f"{len(props)} objects → {path}")

    def _export_image(self):
        if self.current_result is None:
            QMessageBox.warning(self, "No result", "Run segmentation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save annotated image", "",
            "PNG (*.png);;TIFF (*.tif)"
        )
        if not path:
            return
        import matplotlib.pyplot as plt
        img  = self._get_image()
        cmap = self.cmb_cmap.currentText()
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="#0d1117")
        axes[0].imshow(img, cmap=cmap, aspect="equal")
        axes[0].set_title("Input", color="#79c0ff"); axes[0].axis("off")
        if self.chk_show_fill.isChecked() and self.current_result.label_image is not None:
            overlay = overlay_labels_on_image(img, self.current_result.label_image)
            axes[1].imshow(overlay, aspect="equal")
        else:
            axes[1].imshow(img, cmap=cmap, aspect="equal")
        if self.chk_show_circles.isChecked():
            draw_actual_contours(
                axes[1], self.current_result,
                circle_color=self.cmb_circle_color.currentText(),
                show_numbers=self.chk_show_numbers.isChecked()
            )
        axes[1].set_title(
            f"{self.current_result.method_name}  N={self.current_result.n_objects}",
            color="#79c0ff"
        )
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        QMessageBox.information(self, "Saved", f"Image saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — CORRECT (Correcting ground-truth annotations)
# ═══════════════════════════════════════════════════════════════════════════════

class CorrectionTab(QWidget):
    """
    Workflow:
      1. Load your manual ground-truth mask.
      2. DL detections are overlaid on it:
           GREEN  = DL region overlaps sufficiently with GT  → already approved
           RED    = DL region does NOT overlap GT           → click to approve anyway
      3. Click on a red region to make it green (approved).
         You can also click on a green region to set it back to red.
      4. Send the corrected mask to Validation.
    """
    corrected_mask_ready = pyqtSignal(object)

    _OVERLAP_THRESHOLD = 0.3   # min IoU-overlap om als TP te beschouwen

    def __init__(self, dl_tab, viewer_tab=None, parent=None):
        super().__init__(parent)
        self.dl_tab = dl_tab
        self.viewer_tab = viewer_tab

        self.gt_mask: Optional[np.ndarray] = None
        self.gt_path = ""

        # per DL label id: True = approved (green), False = rejected (red)
        self._region_status: Dict[int, bool] = {}
        # region-properties cache
        self._dl_regions: List[Dict] = []

        # per GT label id: True = active (blue), False = removed by user
        self._gt_region_status: Dict[int, bool] = {}
        # GT region-properties cache (labelled mask)
        self._gt_label_image: Optional[np.ndarray] = None
        self._gt_regions: List[Dict] = []

        self._cid = None   # matplotlib click connection
        self._build_ui()

    # ──────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Action bar ──────────────────────────────────────────────────────
        grp_actions = QGroupBox("⚙️  Actions")
        av = QHBoxLayout(grp_actions)
        av.setSpacing(8)

        self.btn_load_gt = QPushButton("📂  Load GT Mask")
        self.btn_load_gt.setObjectName("primary")
        self.btn_load_gt.setToolTip(
            "Load a manually annotated ground-truth mask (TIFF file).\n\n"
            "This mask contains the 'correct' segmentation against which\n"
            "DL detections are compared.\n\n"
            "White (255) = aggregate present, Black (0) = no aggregate."
        )
        self.btn_load_gt.clicked.connect(self._load_gt_mask)
        av.addWidget(self.btn_load_gt)

        self.btn_refresh = QPushButton("🔄  Refresh / fetch DL result")
        self.btn_refresh.setToolTip(
            "Fetches the latest segmentation result from the Deep Learning tab\n"
            "and automatically compares it with the loaded GT mask.\n\n"
            "Green contours = DL detection overlaps with GT (true positive).\n"
            "Red contours = DL detection does NOT overlap GT (possible false positive).\n"
            "Blue contours = GT region (manual annotation)."
        )
        self.btn_refresh.clicked.connect(self._auto_classify_and_draw)
        self.btn_refresh.setEnabled(False)
        av.addWidget(self.btn_refresh)

        self.btn_reset = QPushButton("↺  Reset All Corrections")
        self.btn_reset.setToolTip(
            "Resets all manual corrections to the automatic classification.\n"
            "All green/red statuses are recalculated based\n"
            "on the overlap with the GT mask."
        )
        self.btn_reset.clicked.connect(self._reset_corrections)
        self.btn_reset.setEnabled(False)
        av.addWidget(self.btn_reset)

        self.btn_send = QPushButton("✅  Send Corrected Mask to Validation")
        self.btn_send.setObjectName("primary")
        self.btn_send.setToolTip(
            "Sends the corrected mask (after your adjustments) to\n"
            "the Validation tab for quantitative evaluation.\n\n"
            "Approved regions + active GT regions are merged\n"
            "into the final corrected mask."
        )
        self.btn_send.clicked.connect(self._send_to_validation)
        self.btn_send.setEnabled(False)
        av.addWidget(self.btn_send)

        self.btn_save_mask = QPushButton("💾  Save Corrected Mask")
        self.btn_save_mask.setToolTip(
            "Saves the corrected mask as a TIFF file.\n"
            "This mask can be reloaded later as a GT mask\n"
            "or used for further analysis."
        )
        self.btn_save_mask.clicked.connect(self._save_corrected_mask)
        self.btn_save_mask.setEnabled(False)
        av.addWidget(self.btn_save_mask)

        av.addStretch()
        root.addWidget(grp_actions)

        # ── Status label ─────────────────────────────────────────────────────
        self.lbl_status = QLabel(
            "Step 1: Load a ground-truth mask.  "
            "Step 2: Click 'Refresh' to classify DL detections.  "
            "Step 3: Click on red regions (approve) or blue GT regions (remove).  "
            "Step 4: Send to Validation."
        )
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(
            "color:#8b949e; font-size:11px; padding:4px 8px;"
            "background:#161b22; border-radius:4px; border:1px solid #21262d;"
        )
        root.addWidget(self.lbl_status)

        # ── Main layout: left = legend/counter, right = canvas ───────────
        mid = QHBoxLayout()
        mid.setSpacing(8)

        # left: legend + counter
        left = QWidget()
        left.setFixedWidth(220)
        lv = QVBoxLayout(left)
        lv.setSpacing(8)

        grp_legend = QGroupBox("🎨  Legend")
        lv2 = QVBoxLayout(grp_legend)
        for color, txt in [
            ("#00c850", "✔  Approved (overlaps GT)"),
            ("#e03030", "✖  Not approved (click to correct)"),
            ("#4488ff", "◌  GT region (blue; click to remove)"),
        ]:
            row = QHBoxLayout()
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{color}; font-size:18px;")
            lbl = QLabel(txt)
            lbl.setStyleSheet("color:#c9d1d9; font-size:11px;")
            lbl.setWordWrap(True)
            row.addWidget(dot)
            row.addWidget(lbl, stretch=1)
            lv2.addLayout(row)
        lv.addWidget(grp_legend)

        grp_count = QGroupBox("📊  Counter")
        cv = QFormLayout(grp_count)
        self.lbl_n_green = QLabel("—")
        self.lbl_n_red   = QLabel("—")
        self.lbl_n_total = QLabel("—")
        self.lbl_n_gt_removed = QLabel("—")
        self.lbl_n_green.setStyleSheet("color:#3fb950; font-weight:bold;")
        self.lbl_n_red.setStyleSheet("color:#f85149; font-weight:bold;")
        self.lbl_n_gt_removed.setStyleSheet("color:#79c0ff; font-weight:bold;")
        cv.addRow("Approved:",   self.lbl_n_green)
        cv.addRow("Rejected:",     self.lbl_n_red)
        cv.addRow("Total DL:",     self.lbl_n_total)
        cv.addRow("GT removed:", self.lbl_n_gt_removed)
        lv.addWidget(grp_count)

        grp_tip = QGroupBox("💡  Tip")
        tv = QVBoxLayout(grp_tip)
        tip_lbl = QLabel(
            "Click on a <b style='color:#e03030'>red</b> region to make it "
            "<b style='color:#00c850'>green</b> (approved).<br><br>"
            "Click on a <b style='color:#00c850'>green</b> region to set it "
            "back to <b style='color:#e03030'>red</b>.<br><br>"
            "Click on a <b style='color:#4488ff'>blue GT region</b> to "
            "remove it from the mask."
        )
        tip_lbl.setWordWrap(True)
        tip_lbl.setStyleSheet("color:#8b949e; font-size:11px;")
        tv.addWidget(tip_lbl)
        lv.addWidget(grp_tip)

        lv.addStretch()
        mid.addWidget(left)

        # right: canvas
        right_w = QWidget()
        rv = QVBoxLayout(right_w)
        rv.setSpacing(4)
        self.canvas_corr = MplCanvas(width=12, height=9)
        nav = NavigationToolbar(self.canvas_corr, self)
        nav.setStyleSheet(
            "background:#f0f3f6; border-bottom:1px solid #d0d7de;"
            "border-radius:6px 6px 0 0;"
        )
        rv.addWidget(nav)
        rv.addWidget(self.canvas_corr)
        mid.addWidget(right_w, stretch=1)

        root.addLayout(mid, stretch=1)

    # ──────────────────────────────────────────────────────────────────────────
    # Load mask
    # ──────────────────────────────────────────────────────────────────────────
    def _load_gt_mask(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Ground-Truth Mask", "",
            "Mask files (*.tif *.tiff *.png);;All files (*)"
        )
        if not path:
            return
        try:
            mask = _load_gt_mask_from_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Error loading", f"Kan masker niet laden:\n{e}")
            return
        if mask.sum() == 0:
            QMessageBox.warning(self, "Empty mask",
                                "The loaded mask contains no foreground pixels.")
            return

        self.gt_mask = mask
        self.gt_path = path

        # Label GT regions for click detection and removal
        gt_labeled = label(mask.astype(bool))
        self._gt_label_image = gt_labeled
        self._gt_region_status = {}
        self._gt_regions = []
        for region in regionprops(gt_labeled):
            self._gt_region_status[region.label] = True   # default active (blue)
            self._gt_regions.append({
                "label": region.label,
                "cy": region.centroid[0],
                "cx": region.centroid[1],
                "area": region.area,
            })

        n_pos = int(mask.sum())
        pct   = 100.0 * n_pos / mask.size
        self.lbl_status.setText(
            f"✅  GT loaded: {Path(path).name}  |  Shape: {mask.shape}  |  "
            f"Foreground: {n_pos:,} px ({pct:.1f}%)  |  "
            "Click 'Refresh' to load DL detections."
        )
        self.lbl_status.setStyleSheet(
            "color:#3fb950; font-size:11px; padding:4px 8px;"
            "background:#0d2b0d; border-radius:4px; border:1px solid #238636;"
        )
        self.btn_refresh.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self._auto_classify_and_draw()

    # ──────────────────────────────────────────────────────────────────────────
    # Automatically classify: green if overlap >= threshold, otherwise red
    # ──────────────────────────────────────────────────────────────────────────
    def _auto_classify_and_draw(self):
        if self.gt_mask is None:
            return

        result = getattr(self.dl_tab, "current_result", None)
        dl_label = result.label_image if result is not None else None

        if dl_label is None:
            # Geen DL-resultaat: toon alleen GT
            self._region_status = {}
            self._dl_regions = []
            self._draw(dl_label)
            self.lbl_status.setText(
                "⚠  No DL segmentation available. "
                "Voer eerst Deep Learning segmentatie uit (Tab 4)."
            )
            self.lbl_status.setStyleSheet(
                "color:#d29922; font-size:11px; padding:4px 8px;"
                "background:#2b1d0e; border-radius:4px; border:1px solid #9e6a03;"
            )
            self.btn_send.setEnabled(False)
            return

        # Ensure mask and label dimensions match
        gt = self.gt_mask
        if dl_label.shape != gt.shape:
            from skimage.transform import resize as sk_resize
            dl_label = sk_resize(
                dl_label.astype(float), gt.shape,
                order=0, anti_aliasing=False, preserve_range=True
            ).astype(np.int32)

        gt_bool = gt.astype(bool)
        self._region_status = {}
        self._dl_regions = []

        for region in regionprops(dl_label):
            lbl_id = region.label
            region_mask = (dl_label == lbl_id)
            intersection = int(np.logical_and(region_mask, gt_bool).sum())
            union = int(np.logical_or(region_mask, gt_bool).sum())
            iou = intersection / union if union > 0 else 0.0
            # overlap = fraction of the DL region that falls within the GT
            overlap = intersection / region.area if region.area > 0 else 0.0
            approved = overlap >= self._OVERLAP_THRESHOLD
            self._region_status[lbl_id] = approved
            self._dl_regions.append({
                "label": lbl_id,
                "cy": region.centroid[0],
                "cx": region.centroid[1],
                "area": region.area,
            })

        self._draw(dl_label)
        self._update_counters()
        self.btn_send.setEnabled(True)
        self.btn_save_mask.setEnabled(True)

        n_green = sum(1 for v in self._region_status.values() if v)
        n_red   = len(self._region_status) - n_green
        self.lbl_status.setText(
            f"📊  {len(self._region_status)} DL regions loaded  |  "
            f"Green (approved): {n_green}  |  Red (click to correct): {n_red}  |  "
            "Click on a red region to approve it."
        )
        self.lbl_status.setStyleSheet(
            "color:#79c0ff; font-size:11px; padding:4px 8px;"
            "background:#0d1b2b; border-radius:4px; border:1px solid #1f6feb;"
        )

        # Connect click event
        if self._cid is not None:
            self.canvas_corr.mpl_disconnect(self._cid)
        self._cid = self.canvas_corr.mpl_connect("button_press_event", self._on_click)

    # ──────────────────────────────────────────────────────────────────────────
    # Draw canvas
    # ──────────────────────────────────────────────────────────────────────────
    def _draw(self, dl_label: Optional[np.ndarray]):
        ax = self.canvas_corr.axes
        ax.cla()
        ax.set_facecolor("#0d1117")
        ax.axis("off")

        if self.gt_mask is None:
            ax.text(0.5, 0.5, "No mask loaded",
                    ha="center", va="center", color="#484f58",
                    fontsize=13, transform=ax.transAxes)
            self.canvas_corr.draw_idle()
            return

        gt = self.gt_mask
        H, W = gt.shape

        # ── Background: actual microscopy image ────────────────────────
        micro_img = None
        if self.viewer_tab is not None:
            micro_img = self.viewer_tab.get_current_image()

        if micro_img is not None:
            # Normalise to [0,1] for display
            img_show = micro_img.astype(np.float32)
            lo, hi = np.percentile(img_show, [1, 99])
            if hi > lo:
                img_show = np.clip((img_show - lo) / (hi - lo), 0, 1)
            else:
                img_show = np.zeros_like(img_show)
            ax.imshow(img_show, cmap="gray", aspect="equal",
                      interpolation="nearest", zorder=1)
        else:
            # Fallback: dark background with GT pixels faintly highlighted
            bg = np.zeros((H, W, 3), dtype=np.uint8)
            bg[gt.astype(bool)] = [50, 50, 70]
            ax.imshow(bg, aspect="equal", interpolation="nearest", zorder=1)

        # ── GT contour (blue dashed) — active regions only ─────────────
        from skimage import measure as sk_measure
        if self._gt_label_image is not None:
            for gt_reg in self._gt_regions:
                gt_lbl_id = gt_reg["label"]
                if not self._gt_region_status.get(gt_lbl_id, True):
                    continue   # verwijderd door gebruiker; niet tekenen
                gt_contours = sk_measure.find_contours(
                    (self._gt_label_image == gt_lbl_id).astype(float), 0.5)
                for c in gt_contours:
                    ax.plot(c[:, 1], c[:, 0], color="#4488ff", linewidth=1.0,
                            linestyle="--", alpha=0.85, zorder=2)
        else:
            # Fallback: hele GT als één contour
            gt_contours = sk_measure.find_contours(gt.astype(float), 0.5)
            for c in gt_contours:
                ax.plot(c[:, 1], c[:, 0], color="#4488ff", linewidth=1.0,
                        linestyle="--", alpha=0.85, zorder=2)

        # ── DL regions as semi-transparent overlay ─────────────────────────
        if dl_label is not None and len(self._region_status) > 0:
            green_mask = np.zeros((H, W), dtype=bool)
            red_mask   = np.zeros((H, W), dtype=bool)

            for lbl_id, approved in self._region_status.items():
                region_px = (dl_label == lbl_id)
                if approved:
                    green_mask |= region_px
                else:
                    red_mask |= region_px

            # Filled areas (RGBA)
            rgba = np.zeros((H, W, 4), dtype=np.float32)
            rgba[green_mask] = [0.0,  0.78, 0.31, 0.35]   # groen
            rgba[red_mask]   = [0.87, 0.19, 0.19, 0.40]   # rood
            ax.imshow(rgba, aspect="equal", interpolation="nearest", zorder=3)

            # Contours per region
            for lbl_id, approved in self._region_status.items():
                color = "#00c850" if approved else "#e03030"
                contours = sk_measure.find_contours(
                    (dl_label == lbl_id).astype(float), 0.5)
                for c in contours:
                    ax.plot(c[:, 1], c[:, 0], color=color,
                            linewidth=1.4, zorder=5)

            # Numbers at centroid
            for reg in self._dl_regions:
                lbl_id = reg["label"]
                color  = "#00ff66" if self._region_status.get(lbl_id, False) else "#ff6666"
                ax.text(reg["cx"], reg["cy"], str(lbl_id),
                        color=color, fontsize=6, fontweight="bold",
                        ha="center", va="center", zorder=6, clip_on=True)

        n_green = sum(1 for v in self._region_status.values() if v)
        n_red   = len(self._region_status) - n_green
        n_gt_removed = sum(1 for v in self._gt_region_status.values() if not v)
        ax.set_title(
            f"Correction view  |  "
            f"GT contour (blue)  +  DL overlay  |  "
            f"Green: {n_green}  ·  Red: {n_red}  ·  GT removed: {n_gt_removed}",
            color="#79c0ff", fontsize=10
        )
        self.canvas_corr.fig.tight_layout(pad=0.5)
        self.canvas_corr.draw_idle()

    # ──────────────────────────────────────────────────────────────────────────
    # Click handler: toggle status of clicked region
    # ──────────────────────────────────────────────────────────────────────────
    def _on_click(self, event):
        if event.inaxes != self.canvas_corr.axes:
            return
        if event.xdata is None or event.ydata is None:
            return

        cx_click = event.xdata
        cy_click = event.ydata

        # ── Find nearest DL region centroid ──────────────────────────
        best_dl_lbl  = None
        best_dl_dist = float("inf")
        for reg in self._dl_regions:
            dist = np.sqrt((reg["cx"] - cx_click) ** 2 + (reg["cy"] - cy_click) ** 2)
            radius = np.sqrt(reg["area"] / np.pi)
            if dist < max(radius * 1.5, 10) and dist < best_dl_dist:
                best_dl_dist = dist
                best_dl_lbl  = reg["label"]

        # ── Find nearest GT region centroid ──────────────────────────
        best_gt_lbl  = None
        best_gt_dist = float("inf")
        for reg in self._gt_regions:
            dist = np.sqrt((reg["cx"] - cx_click) ** 2 + (reg["cy"] - cy_click) ** 2)
            radius = np.sqrt(reg["area"] / np.pi)
            if dist < max(radius * 1.5, 10) and dist < best_gt_dist:
                best_gt_dist = dist
                best_gt_lbl  = reg["label"]

        # ── Determine which region was clicked (DL or GT, smallest distance) ──
        result = getattr(self.dl_tab, "current_result", None)
        dl_label = result.label_image if result is not None else None
        if dl_label is not None and dl_label.shape != self.gt_mask.shape:
            from skimage.transform import resize as sk_resize
            dl_label = sk_resize(
                dl_label.astype(float), self.gt_mask.shape,
                order=0, anti_aliasing=False, preserve_range=True
            ).astype(np.int32)

        clicked_dl = best_dl_lbl is not None
        clicked_gt = best_gt_lbl is not None

        if clicked_dl and clicked_gt:
            # Both nearby: choose the closest
            if best_gt_dist < best_dl_dist:
                clicked_dl = False
            else:
                clicked_gt = False

        if clicked_dl:
            # Toggle DL region: red ↔ green
            self._region_status[best_dl_lbl] = not self._region_status[best_dl_lbl]
            self._draw(dl_label)
            self._update_counters()
            n_green = sum(1 for v in self._region_status.values() if v)
            n_red   = len(self._region_status) - n_green
            status  = "approved ✔" if self._region_status[best_dl_lbl] else "rejected ✖"
            self.lbl_status.setText(
                f"🖱  DL region {best_dl_lbl} → {status}  |  "
                f"Green: {n_green}  ·  Red: {n_red}"
            )

        elif clicked_gt:
            # Toggle GT region: active (blue) ↔ removed
            was_active = self._gt_region_status.get(best_gt_lbl, True)
            self._gt_region_status[best_gt_lbl] = not was_active
            self._draw(dl_label)
            self._update_counters()
            n_gt_removed = sum(1 for v in self._gt_region_status.values() if not v)
            gt_status = "removed ✖" if was_active else "restored ✔"
            self.lbl_status.setText(
                f"🖱  GT region {best_gt_lbl} → {gt_status}  |  "
                f"GT removed: {n_gt_removed} / {len(self._gt_region_status)}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Update counters
    # ──────────────────────────────────────────────────────────────────────────
    def _update_counters(self):
        n_green = sum(1 for v in self._region_status.values() if v)
        n_red   = len(self._region_status) - n_green
        n_gt_removed = sum(1 for v in self._gt_region_status.values() if not v)
        self.lbl_n_green.setText(str(n_green))
        self.lbl_n_red.setText(str(n_red))
        self.lbl_n_total.setText(str(len(self._region_status)))
        self.lbl_n_gt_removed.setText(
            f"{n_gt_removed} / {len(self._gt_region_status)}"
            if self._gt_region_status else "—"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Reset all corrections to auto-classification
    # ──────────────────────────────────────────────────────────────────────────
    def _reset_corrections(self):
        # Also restore removed GT regions
        for lbl_id in self._gt_region_status:
            self._gt_region_status[lbl_id] = True
        self._auto_classify_and_draw()

    # ──────────────────────────────────────────────────────────────────────────
    # Build corrected mask and send to ValidationTab
    # ──────────────────────────────────────────────────────────────────────────
    def _send_to_validation(self):
        if self.gt_mask is None:
            QMessageBox.warning(self, "No mask", "Load a GT mask first.")
            return

        result = getattr(self.dl_tab, "current_result", None)
        dl_label = result.label_image if result is not None else None

        corrected = self.gt_mask.copy().astype(np.uint8)

        # Remove GT regions that the user has disabled
        if self._gt_label_image is not None:
            for gt_lbl_id, active in self._gt_region_status.items():
                if not active:
                    corrected[self._gt_label_image == gt_lbl_id] = 0

        if dl_label is not None:
            if dl_label.shape != self.gt_mask.shape:
                from skimage.transform import resize as sk_resize
                dl_label = sk_resize(
                    dl_label.astype(float), self.gt_mask.shape,
                    order=0, anti_aliasing=False, preserve_range=True
                ).astype(np.int32)

            # Add approved DL regions to the corrected mask
            for lbl_id, approved in self._region_status.items():
                if approved:
                    corrected[dl_label == lbl_id] = 1

        self.corrected_mask_ready.emit(corrected)
        n_gt_removed = sum(1 for v in self._gt_region_status.values() if not v)
        n_added = int(corrected.sum()) - int(self.gt_mask.sum())
        QMessageBox.information(
            self, "Sent",
            f"Corrected mask sent to Validation.\n"
            f"Original GT: {int(self.gt_mask.sum()):,} px\n"
            f"GT regions removed: {n_gt_removed}\n"
            f"Corrected: {int(corrected.sum()):,} px\n"
            f"Net difference: {n_added:+,} px"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Save corrected mask as TIFF
    # ──────────────────────────────────────────────────────────────────────────
    def _save_corrected_mask(self):
        if self.gt_mask is None:
            QMessageBox.warning(self, "No mask", "Load a GT mask first.")
            return

        result = getattr(self.dl_tab, "current_result", None)
        dl_label = result.label_image if result is not None else None

        # Build corrected mask (same logic as _send_to_validation)
        corrected = self.gt_mask.copy().astype(np.uint8)

        # Remove GT regions that the user has disabled
        if self._gt_label_image is not None:
            for gt_lbl_id, active in self._gt_region_status.items():
                if not active:
                    corrected[self._gt_label_image == gt_lbl_id] = 0

        if dl_label is not None:
            if dl_label.shape != self.gt_mask.shape:
                from skimage.transform import resize as sk_resize
                dl_label = sk_resize(
                    dl_label.astype(float), self.gt_mask.shape,
                    order=0, anti_aliasing=False, preserve_range=True
                ).astype(np.int32)
            for lbl_id, approved in self._region_status.items():
                if approved:
                    corrected[dl_label == lbl_id] = 1

        path, _ = QFileDialog.getSaveFileName(
            self, "Save corrected mask",
            "corrected_mask.tif",
            "TIFF (*.tif *.tiff);;PNG (*.png)"
        )
        if not path:
            return

        # Save as binary mask (0/255) for universal readability
        tifffile.imwrite(path, (corrected * 255).astype(np.uint8))

        n_pos = int(corrected.sum())
        QMessageBox.information(
            self, "Saved",
            f"Corrected mask saved:\n{path}\n\n"
            f"Foreground pixels: {n_pos:,}\n"
            f"Dimensions: {corrected.shape[1]} × {corrected.shape[0]} px"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Slot: when DL tab has a new result
    # ──────────────────────────────────────────────────────────────────────────
    def _on_dl_updated(self, result):
        if self.gt_mask is not None:
            self._auto_classify_and_draw()
        self.lbl_status.setText(
            f"ℹ️  New DL result: {result.method_name} "
            f"({result.n_objects} objects). "
            "Display automatically updated."
        )


class DoubleCanvas(FigureCanvas):
    """Matplotlib canvas with 2 subplots: [Ground Truth | Tab 4]."""

    OUTER_BG = "#161b22"
    INNER_BG = "#0d1117"
    TICK_CLR = "#c9d1d9"
    GRID_CLR = "#30363d"
    TITLES   = ["🏁  Ground Truth", "🧠  Deep Learning (Ensemble)"]

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 5), dpi=100, facecolor=self.OUTER_BG)
        self.axes = self.fig.subplots(1, 2)
        for ax, title in zip(self.axes, self.TITLES):
            self._style_ax(ax, title)
        self.fig.tight_layout(pad=1.5)
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def _style_ax(self, ax, title=""):
        ax.set_facecolor(self.INNER_BG)
        ax.tick_params(colors=self.TICK_CLR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(self.GRID_CLR)
        ax.set_title(title, color="#79c0ff", fontsize=10, fontweight="bold", pad=6)
        ax.axis("off")

    def update_plots(self, gt_mask, dl_mask):
        self.fig.clf()
        axes = self.fig.subplots(1, 2)

        def _mask_rgb(pred, gt):
            if pred is None:
                return None
            p = pred.astype(bool)
            g = gt.astype(bool)
            rgb = np.zeros((*g.shape, 3), dtype=np.uint8)
            rgb[g & p]  = [0, 200, 80]    # TP
            rgb[p & ~g] = [220, 50, 50]   # FP
            rgb[g & ~p] = [50, 100, 220]  # FN
            return rgb

        gt_rgb = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        gt_rgb[gt_mask.astype(bool)] = [0, 200, 80]

        overlays   = [gt_rgb, _mask_rgb(dl_mask, gt_mask)]
        subtitles  = ["Ground Truth", "TP=green  FP=red  FN=blue"]

        for ax, overlay, title, sub in zip(axes, overlays, self.TITLES, subtitles):
            self._style_ax(ax, title)
            if overlay is not None:
                ax.imshow(overlay, aspect="equal", interpolation="nearest")
                ax.set_xlabel(sub, color="#8b949e", fontsize=8)
            else:
                ax.text(0.5, 0.5, "Not yet\nsegmented",
                        ha="center", va="center",
                        color="#484f58", fontsize=11,
                        transform=ax.transAxes)

        self.fig.tight_layout(pad=1.5)
        self.draw_idle()


def _compute_validation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    p = pred_mask.astype(bool).ravel()
    g = gt_mask.astype(bool).ravel()
    tp = int(np.logical_and(p,  g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1  = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn,
                precision=precision, recall=recall, f1=f1, iou=iou)


def _result_to_binary_mask(result) -> Optional[np.ndarray]:
    if result is None:
        return None
    if result.label_image is not None:
        return (result.label_image > 0).astype(np.uint8)
    return None


def _load_gt_mask_from_file(path: str) -> np.ndarray:
    img = tifffile.imread(path)
    if img.ndim == 3:
        img = img.max(axis=0)
    if img.ndim == 4:
        img = img.max(axis=(0, 1))
    return (img > 0).astype(np.uint8)


class ValidationTab(QWidget):
    def __init__(self, dl_tab, parent=None):
        super().__init__(parent)
        self.dl_tab   = dl_tab
        self.gt_mask  = None
        self.gt_path  = ""
        self._last_metrics = None
        self._corrected_mask: Optional[np.ndarray] = None  # van CorrectionTab
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(10, 10, 10, 10)

        grp_actions = QGroupBox("⚙️  Actions")
        av = QHBoxLayout(grp_actions)
        av.setSpacing(8)

        self.btn_load_gt = QPushButton("📂  Load Ground-Truth Mask")
        self.btn_load_gt.setObjectName("primary")
        self.btn_load_gt.setToolTip(
            "Load a manually annotated ground-truth mask (TIFF file).\n\n"
            "This mask is used as reference for validation.\n"
            "White (255) = aggregate present, Black (0) = no aggregate.\n\n"
            "Tip: the corrected mask from Tab 5 is automatically loaded\n"
            "when you send it via the Correction tab."
        )
        self.btn_load_gt.clicked.connect(self._load_gt_mask)
        av.addWidget(self.btn_load_gt)

        self.btn_compare = QPushButton("📊  Compare Result")
        self.btn_compare.setEnabled(False)
        self.btn_compare.setToolTip(
            "Calculates validation metrics by comparing the DL segmentation result\n"
            "with the loaded ground-truth mask.\n\n"
            "Calculated metrics:\n"
            "  • F1/Dice  — harmonic mean of precision and recall\n"
            "  • IoU      — intersection over union (Jaccard index)\n"
            "  • Precision — fraction of detections that are correct\n"
            "  • Recall   — fraction of true aggregates that were found\n"
            "  • TP/FP/FN — true positive / false positive / false negative (pixels)"
        )
        self.btn_compare.clicked.connect(self._run_comparison)
        av.addWidget(self.btn_compare)

        self.btn_export = QPushButton("💾  Export Report (CSV)")
        self.btn_export.setEnabled(False)
        self.btn_export.setToolTip(
            "Exports the validation metrics to a CSV file.\n"
            "Useful for tracking results across multiple images or runs."
        )
        self.btn_export.clicked.connect(self._export_csv)
        av.addWidget(self.btn_export)

        av.addStretch()
        root.addWidget(grp_actions)

        self.lbl_status = QLabel("Step 1: Load a ground-truth mask to get started.")
        self.lbl_status.setStyleSheet(
            "color: #8b949e; font-size: 11px; padding: 4px 8px;"
            "background: #161b22; border-radius: 4px; border: 1px solid #21262d;"
        )
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        grp_metrics = QGroupBox("📈  Validation Metrics")
        mv = QVBoxLayout(grp_metrics)

        self.tbl_metrics = QTableWidget(6, 2)
        self.tbl_metrics.setHorizontalHeaderLabels(["Metric", "Deep Learning (Ensemble)"])
        self.tbl_metrics.verticalHeader().setVisible(False)
        self.tbl_metrics.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_metrics.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl_metrics.setAlternatingRowColors(True)
        self.tbl_metrics.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_metrics.setMaximumHeight(230)
        self._populate_empty_table()
        mv.addWidget(self.tbl_metrics)
        root.addWidget(grp_metrics)

        grp_visual = QGroupBox("🖼️  Visual Comparison  —  Ground Truth | Deep Learning")
        vv = QVBoxLayout(grp_visual)

        self.canvas = DoubleCanvas()
        nav = NavigationToolbar(self.canvas, self)
        nav.setStyleSheet(
            "background:#f0f3f6; border-bottom: 1px solid #d0d7de;"
            "border-radius: 6px 6px 0 0;"
        )
        vv.addWidget(nav)
        vv.addWidget(self.canvas)

        legend_bar = QHBoxLayout()
        for color, label_txt in [
            ("#00c850", "TP — True Positive"),
            ("#dc3232", "FP — False Positive"),
            ("#3264dc", "FN — False Negative"),
        ]:
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 16px;")
            lbl = QLabel(label_txt)
            lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
            legend_bar.addWidget(dot)
            legend_bar.addWidget(lbl)
            legend_bar.addSpacing(16)
        legend_bar.addStretch()
        vv.addLayout(legend_bar)

        root.addWidget(grp_visual, stretch=1)

    def _populate_empty_table(self):
        rows = [
            ("F1-score (Dice)",  "—"),
            ("IoU (Jaccard)",    "—"),
            ("Precision",        "—"),
            ("Recall",           "—"),
            ("True Positives",   "—"),
            ("False Positives",  "—"),
        ]
        self._fill_table(rows, highlight=False)

    def _fill_table(self, rows, highlight=True):
        SCORE_ROWS = {0, 1, 2, 3}

        for i, (metric, val_dl) in enumerate(rows):
            item_metric = QTableWidgetItem(metric)
            item_metric.setFont(QFont("Segoe UI", 10, QFont.Bold))
            item_metric.setForeground(QColor("#79c0ff"))
            self.tbl_metrics.setItem(i, 0, item_metric)

            item = QTableWidgetItem(val_dl)
            item.setTextAlignment(Qt.AlignCenter)
            item.setFont(QFont("Cascadia Code", 10))

            if highlight and i in SCORE_ROWS and val_dl not in ("—", "N/A"):
                try:
                    score = float(val_dl)
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
            "Mask files (*.tif *.tiff *.png);;All files (*)"
        )
        if not path:
            return
        try:
            mask = _load_gt_mask_from_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Error loading", f"Kan masker niet laden:\n{e}")
            return

        if mask.sum() == 0:
            QMessageBox.warning(
                self, "Empty mask",
                "The loaded mask contains no foreground pixels."
            )
            return

        self.gt_mask = mask
        self.gt_path = path
        n_pos = int(mask.sum())
        pct   = 100.0 * n_pos / mask.size
        self.lbl_status.setText(
            f"✅  GT loaded: {Path(path).name}  |  Shape: {mask.shape}  |  "
            f"Foreground: {n_pos:,} px ({pct:.1f}%)  |  "
            "Click 'Refresh' to load DL detections."
        )
        self.lbl_status.setStyleSheet(
            "color:#3fb950; font-size:11px; padding:4px 8px;"
            "background:#0d2b0d; border-radius:4px; border:1px solid #238636;"
        )
        self.btn_refresh.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self._auto_classify_and_draw()

        self.lbl_status.setText(
            f"✅  Ground-truth loaded: {Path(path).name}  |  "
            f"Shape: {mask.shape}  |  Foreground: {n_pos:,} px ({pct:.1f}%)"
        )
        self.lbl_status.setStyleSheet(
            "color: #3fb950; font-size: 11px; padding: 4px 8px;"
            "background: #0d2b0d; border-radius: 4px; border: 1px solid #238636;"
        )
        self.btn_compare.setEnabled(True)

    def _run_comparison(self):
        if self.gt_mask is None:
            QMessageBox.warning(self, "No ground truth", "Load a ground-truth mask first.")
            return

        dl_mask  = _result_to_binary_mask(getattr(self.dl_tab, "current_result", None))

        if dl_mask is None:
            QMessageBox.warning(
                self, "No segmentation",
                "Run segmentation in Tab 4 (Deep Learning) first."
            )
            return

        gt = self.gt_mask

        def _resize_if_needed(mask, target):
            if mask is None or mask.shape == target:
                return mask
            from skimage.transform import resize as sk_resize
            r = sk_resize(mask.astype(float), target,
                          order=0, anti_aliasing=False, preserve_range=True)
            return (r > 0.5).astype(np.uint8)

        dl_mask  = _resize_if_needed(dl_mask,  gt.shape)

        def _metrics_or_na(mask):
            if mask is None:
                return {k: None for k in ("f1", "iou", "precision", "recall", "tp", "fp")}
            return _compute_validation_metrics(mask, gt)

        m_dl  = _metrics_or_na(dl_mask)

        def _v(m, key):
            v = m.get(key)
            if v is None:
                return "N/A"
            return f"{v:.4f}" if isinstance(v, float) else f"{v:,}"

        rows = [
            ("F1-score (Dice)",  _v(m_dl, "f1")),
            ("IoU (Jaccard)",    _v(m_dl, "iou")),
            ("Precision",         _v(m_dl, "precision")),
            ("Recall",           _v(m_dl, "recall")),
            ("True Positives",   _v(m_dl, "tp")),
            ("False Positives",  _v(m_dl, "fp")),
        ]
        self._fill_table(rows, highlight=True)
        self.canvas.update_plots(gt, dl_mask)

        f1_dl  = m_dl.get("f1")
        if f1_dl is not None:
            verdict = f"Comparison successful — F1 = {f1_dl:.4f}."
        else:
            verdict = "No valid metrics available."

        self.lbl_status.setText(f"📊  Comparison complete  |  {verdict}")
        self.lbl_status.setStyleSheet(
            "color: #79c0ff; font-size: 11px; padding: 4px 8px;"
            "background: #0d1b2b; border-radius: 4px; border: 1px solid #1f6feb;"
        )
        self._last_metrics = {"dl": m_dl}
        self.btn_export.setEnabled(True)

    def _export_csv(self):
        if self._last_metrics is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export validation report", "validation_report.csv",
            "CSV files (*.csv)"
        )
        if not path:
            return
        keys   = ["f1", "iou", "precision", "recall", "tp", "fp", "fn"]
        labels = ["F1-score (Dice)", "IoU (Jaccard)", "Precision", "Recall",
                  "True Positives", "False Positives", "False Negatives"]
        m_dl  = self._last_metrics["dl"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Metric", "DeepLearning_Ensemble"])
            w.writerow(["Ground-truth mask", self.gt_path])
            w.writerow([])
            for key, label_txt in zip(keys, labels):
                dv = m_dl.get(key)
                w.writerow([
                    label_txt,
                    f"{dv:.6f}" if isinstance(dv, float) else (str(dv) if dv is not None else "N/A"),
                ])
        QMessageBox.information(self, "Exported", f"Rapport opgeslagen:\n{path}")

    def _on_dl_updated(self, result):
        self.lbl_status.setText(
            f"ℹ️  Tab 4 updated: {result.method_name} ({result.n_objects} objects). "
            "Click 'Compare Result' to validate."
        )

    def receive_corrected_mask(self, corrected_mask: np.ndarray):
        """Receives the corrected mask from the Correction tab."""
        self._corrected_mask = corrected_mask
        n_pos = int(corrected_mask.sum())
        pct   = 100.0 * n_pos / corrected_mask.size
        self.lbl_status.setText(
            f"✅  Corrected GT mask received from Correction tab  |  "
            f"Shape: {corrected_mask.shape}  |  Foreground: {n_pos:,} px ({pct:.1f}%)  |  "
            "Click 'Compare Result' to validate."
        )
        self.lbl_status.setStyleSheet(
            "color:#3fb950; font-size:11px; padding:4px 8px;"
            "background:#0d2b0d; border-radius:4px; border:1px solid #238636;"
        )
        # Gebruik het gecorrigeerde masker als gt_mask voor vergelijking
        self.gt_mask = corrected_mask
        self.gt_path = "(corrected via Correction tab)"
        self.btn_compare.setEnabled(True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Protein Aggregate Analyzer — Deep Learning Edition")
        self.resize(1400, 900)
        self._apply_theme()
        self._build_ui()
        self._wire_signals()
        self._show_welcome()

    def _apply_theme(self):
        self.setStyleSheet(DARK_THEME)
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#0d1117"))
        palette.setColor(QPalette.WindowText, QColor("#c9d1d9"))
        self.setPalette(palette)

    def _build_ui(self):
        self.status = QStatusBar()
        self.status.showMessage("Ready — Open a .LIF or .TIF file to get started.")
        self.setStatusBar(self.status)

        tb = QToolBar("Main")
        tb.setIconSize(QSize(20, 20))
        tb.setMovable(False)
        tb.setStyleSheet("background:#161b22; spacing: 4px; padding: 4px 8px; border-bottom: 1px solid #21262d;")
        self.addToolBar(tb)

        for label, slot, tip in [
            ("📂  Open",        self._open_file,      "Open .lif of .tif bestand"),
            ("💾  Save",       self._save_result,    "Save current segmentation"),
            ("📊  Statistics", self._show_stats_dialog, "Show detailed statistics"),
            ("❓  Help",         self._show_help,      "Documentation & tips"),
        ]:
            act = QAction(label, self)
            act.setToolTip(tip)
            act.triggered.connect(slot)
            tb.addAction(act)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.setCentralWidget(self.tabs)

        self.viewer_tab   = ViewerTab()
        self.prep_tab     = PreprocessTab(self.viewer_tab)
        self.cellpose_tab = CellposeTab(self.prep_tab, viewer_tab=self.viewer_tab)
        self.dl_tab       = DeepLearningTab(self.prep_tab, viewer_tab=self.viewer_tab)
        self.corr_tab     = CorrectionTab(self.dl_tab, viewer_tab=self.viewer_tab)
        self.val_tab      = ValidationTab(self.dl_tab)

        self.tabs.addTab(self.viewer_tab,   "🔭  1. Viewer")
        self.tabs.addTab(self.prep_tab,     "🔧  2. Pre-processing")
        self.tabs.addTab(self.cellpose_tab, "🔬  3. Cell Bodies")
        self.tabs.addTab(self.dl_tab,       "🧠  4. Deep Learning")
        self.tabs.addTab(self.corr_tab,     "✏️  5. Correct")
        self.tabs.addTab(self.val_tab,      "📊  6. Validation")

    def _wire_signals(self):
        self.viewer_tab.stack_loaded.connect(
            lambda s: self.status.showMessage(
                f"Loaded: {s.name}  |  Z={s.z_count}, C={s.channel_count}, "
                f"{s.height}×{s.width} px"
            )
        )
        self.dl_tab.result_ready.connect(
            lambda r: self.status.showMessage(
                f"DL Segmentation done: {r.method_name}  |  "
                f"N={r.n_objects} objects  |  {r.time_seconds:.2f}s"
            )
        )
        self.dl_tab.result_ready.connect(self.val_tab._on_dl_updated)
        self.dl_tab.result_ready.connect(self.corr_tab._on_dl_updated)
        # Gecorrigeerd masker van Corrigeer-tab → Validatie-tab
        self.corr_tab.corrected_mask_ready.connect(self.val_tab.receive_corrected_mask)

    def _open_file(self):
        self.tabs.setCurrentIndex(0)
        self.viewer_tab._open_file()

    def _save_result(self):
        result = self.dl_tab.current_result
        if result is None or result.label_image is None:
            QMessageBox.warning(self, "No result", "Run segmentation in Tab 4 first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save label image", "", "TIFF (*.tif);;PNG (*.png)"
        )
        if path:
            tifffile.imwrite(path, result.label_image.astype(np.uint16))
            QMessageBox.information(self, "Saved", f"Labelbeeld opgeslagen:\n{path}")

    def _show_stats_dialog(self):
        result = self.dl_tab.current_result
        if result is None:
            QMessageBox.information(self, "No result", "Run segmentation in Tab 4 first.")
            return
        dlg = StatsDialog(result, self)
        dlg.exec_()

    def _show_help(self):
        txt = """
<body style='background-color:#0d1117; color:#c9d1d9;
             font-family:"Segoe UI",Arial,sans-serif; font-size:13px;
             margin:0; padding:0;'>

  <!-- ── HEADER ── -->
  <div style='background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
              stop:0 #0d1b2b, stop:1 #0d1117);
              border-bottom:2px solid #1f6feb;
              padding:22px 24px 18px 24px; margin-bottom:0;'>
    <p style='margin:0 0 4px 0; font-size:11px; color:#388bfd;
              letter-spacing:2px; text-transform:uppercase;'>
      ✦ User Guide
    </p>
    <h1 style='margin:0 0 6px 0; font-size:22px; font-weight:700;
               color:#ffffff; letter-spacing:0.3px;'>
      🧬 Protein Aggregate Analyzer
    </h1>
    <p style='margin:0; font-size:12px; color:#8b949e;'>
      Deep Learning Editie &nbsp;·&nbsp; Ensemble U-Net segmentatie
    </p>
  </div>

  <div style='padding:20px 24px;'>

    <!-- ── AANBEVOLEN WERKSTROOM ── -->
    <h2 style='color:#79c0ff; font-size:15px; font-weight:700;
               margin:0 0 14px 0; border-bottom:1px solid #21262d;
               padding-bottom:8px;'>
      ⚡ Recommended workflow
    </h2>

    <!-- Stap 1 -->
    <table width='100%' cellspacing='0' cellpadding='0'
           style='margin-bottom:10px;'>
      <tr>
        <td width='42' valign='top'>
          <div style='background:#1f6feb; color:#ffffff; font-weight:700;
                      font-size:13px; border-radius:50%; width:30px;
                      height:30px; text-align:center; line-height:30px;
                      margin-top:2px;'>1</div>
        </td>
        <td valign='top'>
          <div style='background:#161b22; border:1px solid #21262d;
                      border-left:3px solid #1f6feb; border-radius:0 8px 8px 0;
                      padding:10px 14px;'>
            <span style='color:#79c0ff; font-weight:700; font-size:13px;'>
              🔭 Viewer
            </span><br/>
            <span style='color:#c9d1d9; font-size:12px;'>
              Load a <b>.LIF</b> or <b>.TIF</b> file here and select
              the images you want to use for pre-processing.
            </span>
          </div>
        </td>
      </tr>
    </table>

    <!-- Stap 2 -->
    <table width='100%' cellspacing='0' cellpadding='0'
           style='margin-bottom:10px;'>
      <tr>
        <td width='42' valign='top'>
          <div style='background:#1f6feb; color:#ffffff; font-weight:700;
                      font-size:13px; border-radius:50%; width:30px;
                      height:30px; text-align:center; line-height:30px;
                      margin-top:2px;'>2</div>
        </td>
        <td valign='top'>
          <div style='background:#161b22; border:1px solid #21262d;
                      border-left:3px solid #388bfd; border-radius:0 8px 8px 0;
                      padding:10px 14px;'>
            <span style='color:#79c0ff; font-weight:700; font-size:13px;'>
              🔧 Pre-processing
            </span><br/>
            <span style='color:#c9d1d9; font-size:12px;'>
              Click the
              <span style='background:#21262d; color:#f0f6fc;
                           border-radius:4px; padding:1px 6px;
                           font-size:11px;'>★ Recommended pipeline</span>
              button here to apply the optimal image processing.
            </span>
          </div>
        </td>
      </tr>
    </table>

    <!-- Stap 3 -->
    <table width='100%' cellspacing='0' cellpadding='0'
           style='margin-bottom:10px;'>
      <tr>
        <td width='42' valign='top'>
          <div style='background:#1f6feb; color:#ffffff; font-weight:700;
                      font-size:13px; border-radius:50%; width:30px;
                      height:30px; text-align:center; line-height:30px;
                      margin-top:2px;'>3</div>
        </td>
        <td valign='top'>
          <div style='background:#161b22; border:1px solid #21262d;
                      border-left:3px solid #58a6ff; border-radius:0 8px 8px 0;
                      padding:10px 14px;'>
            <span style='color:#79c0ff; font-weight:700; font-size:13px;'>
              🔬 Cell Bodies
            </span><br/>
            <span style='color:#c9d1d9; font-size:12px;'>
              To remove the background you can have the cell bodies
              automatically selected here via <b>Cellpose</b>.
            </span>
          </div>
        </td>
      </tr>
    </table>

    <!-- Stap 4 -->
    <table width='100%' cellspacing='0' cellpadding='0'
           style='margin-bottom:10px;'>
      <tr>
        <td width='42' valign='top'>
          <div style='background:#1f6feb; color:#ffffff; font-weight:700;
                      font-size:13px; border-radius:50%; width:30px;
                      height:30px; text-align:center; line-height:30px;
                      margin-top:2px;'>4</div>
        </td>
        <td valign='top'>
          <div style='background:#161b22; border:1px solid #21262d;
                      border-left:3px solid #79c0ff; border-radius:0 8px 8px 0;
                      padding:10px 14px;'>
            <span style='color:#79c0ff; font-weight:700; font-size:13px;'>
              🧠 Deep Learning
            </span><br/>
            <span style='color:#c9d1d9; font-size:12px;'>
              Run aggregate segmentation here by loading the folder with
              <b>Deep Learning models</b>
              (<code style='background:#0d1117; padding:1px 5px;
                            border-radius:3px; font-size:11px;'>model_fold*.pth</code>)
              and starting the ensemble prediction.
            </span>
          </div>
        </td>
      </tr>
    </table>

    <!-- Stap 5 -->
    <table width='100%' cellspacing='0' cellpadding='0'
           style='margin-bottom:10px;'>
      <tr>
        <td width='42' valign='top'>
          <div style='background:#1f6feb; color:#ffffff; font-weight:700;
                      font-size:13px; border-radius:50%; width:30px;
                      height:30px; text-align:center; line-height:30px;
                      margin-top:2px;'>5</div>
        </td>
        <td valign='top'>
          <div style='background:#161b22; border:1px solid #21262d;
                      border-left:3px solid #56d364; border-radius:0 8px 8px 0;
                      padding:10px 14px;'>
            <span style='color:#79c0ff; font-weight:700; font-size:13px;'>
              ✏️ Correct
            </span><br/>
            <span style='color:#c9d1d9; font-size:12px;'>
              Load your manual <b>ground-truth mask</b> here. DL detections
              are automatically coloured <b style='color:#00c850'>green</b> (overlaps GT)
              or <b style='color:#e03030'>red</b> (does not overlap GT).
              Click on a red region to approve it.
              Then send the corrected mask to Validation.
            </span>
          </div>
        </td>
      </tr>
    </table>

    <!-- Stap 6 -->
    <table width='100%' cellspacing='0' cellpadding='0'
           style='margin-bottom:20px;'>
      <tr>
        <td width='42' valign='top'>
          <div style='background:#1f6feb; color:#ffffff; font-weight:700;
                      font-size:13px; border-radius:50%; width:30px;
                      height:30px; text-align:center; line-height:30px;
                      margin-top:2px;'>6</div>
        </td>
        <td valign='top'>
          <div style='background:#161b22; border:1px solid #21262d;
                      border-left:3px solid #a5d6ff; border-radius:0 8px 8px 0;
                      padding:10px 14px;'>
            <span style='color:#79c0ff; font-weight:700; font-size:13px;'>
              📊 Validation
            </span><br/>
            <span style='color:#c9d1d9; font-size:12px;'>
              Compare the DL result with the (corrected) ground-truth mask
              (F1/Dice, IoU, precision &amp; recall). The corrected mask
              from Tab 5 is automatically loaded here.
            </span>
          </div>
        </td>
      </tr>
    </table>

    <!-- ── TIPS ── -->
    <h2 style='color:#79c0ff; font-size:15px; font-weight:700;
               margin:0 0 12px 0; border-bottom:1px solid #21262d;
               padding-bottom:8px;'>
      💡 Useful tips
    </h2>

    <table width='100%' cellspacing='6' cellpadding='0'
           style='margin-bottom:20px;'>
      <tr>
        <td width='49%' valign='top'>
          <div style='background:#161b22; border:1px solid #21262d;
                      border-radius:8px; padding:12px 14px; height:100%;'>
            <p style='margin:0 0 5px 0; color:#f0c040; font-weight:700;
                      font-size:12px;'>⚠ Too many detections?</p>
            <p style='margin:0; color:#8b949e; font-size:11px; line-height:1.5;'>
              Increase the threshold or enlarge the minimum area
              (px²) in the post-processing settings.
            </p>
          </div>
        </td>
        <td width='2%'></td>
        <td width='49%' valign='top'>
          <div style='background:#161b22; border:1px solid #21262d;
                      border-radius:8px; padding:12px 14px; height:100%;'>
            <p style='margin:0 0 5px 0; color:#f0c040; font-weight:700;
                      font-size:12px;'>⚠ Slow segmentation?</p>
            <p style='margin:0; color:#8b949e; font-size:11px; line-height:1.5;'>
              Disable TTA for faster results, or set Device
              to <i>cuda</i> if you have an NVIDIA GPU.
            </p>
          </div>
        </td>
      </tr>
      <tr><td colspan='3' height='6'></td></tr>
      <tr>
        <td width='49%' valign='top'>
          <div style='background:#161b22; border:1px solid #21262d;
                      border-radius:8px; padding:12px 14px; height:100%;'>
            <p style='margin:0 0 5px 0; color:#f0c040; font-weight:700;
                      font-size:12px;'>⚠ CUDA error?</p>
            <p style='margin:0; color:#8b949e; font-size:11px; line-height:1.5;'>
              Set Device to <i>cpu</i> in Tab 4 to bypass the GPU.
            </p>
          </div>
        </td>
        <td width='2%'></td>
        <td width='49%' valign='top'>
          <div style='background:#161b22; border:1px solid #21262d;
                      border-radius:8px; padding:12px 14px; height:100%;'>
            <p style='margin:0 0 5px 0; color:#f0c040; font-weight:700;
                      font-size:12px;'>⚠ No models found?</p>
            <p style='margin:0; color:#8b949e; font-size:11px; line-height:1.5;'>
              Check that the file names follow the pattern
              <code style='background:#0d1117; padding:1px 4px;
                            border-radius:3px;'>model_fold0.pth</code>
              and that the correct folder is selected.
            </p>
          </div>
        </td>
      </tr>
    </table>

    <!-- ── PAKKETTEN ── -->
    <h2 style='color:#79c0ff; font-size:15px; font-weight:700;
               margin:0 0 10px 0; border-bottom:1px solid #21262d;
               padding-bottom:8px;'>
      ⚙️ Required packages
    </h2>
    <div style='background:#161b22; border:1px solid #21262d;
                border-radius:8px; padding:12px 16px; margin-bottom:8px;'>
      <p style='margin:0 0 4px 0; color:#58a6ff; font-size:11px;
                font-weight:700; text-transform:uppercase;
                letter-spacing:0.5px;'>Required</p>
      <p style='margin:0; color:#8b949e; font-size:11px;'>
        PyQt5 &nbsp;·&nbsp; numpy &nbsp;·&nbsp; scipy &nbsp;·&nbsp;
        scikit-image &nbsp;·&nbsp; matplotlib &nbsp;·&nbsp; tifffile
      </p>
    </div>
    <div style='background:#161b22; border:1px solid #21262d;
                border-radius:8px; padding:12px 16px; margin-bottom:8px;'>
      <p style='margin:0 0 4px 0; color:#58a6ff; font-size:11px;
                font-weight:700; text-transform:uppercase;
                letter-spacing:0.5px;'>Deep Learning (Tab 4)</p>
      <code style='color:#79c0ff; font-size:11px;'>
        pip install torch torchvision segmentation-models-pytorch
      </code>
    </div>
    <div style='background:#161b22; border:1px solid #21262d;
                border-radius:8px; padding:12px 16px; margin-bottom:8px;'>
      <p style='margin:0 0 4px 0; color:#58a6ff; font-size:11px;
                font-weight:700; text-transform:uppercase;
                letter-spacing:0.5px;'>Cellpose (Tab 3)</p>
      <code style='color:#79c0ff; font-size:11px;'>pip install cellpose</code>
    </div>
    <div style='background:#161b22; border:1px solid #21262d;
                border-radius:8px; padding:12px 16px; margin-bottom:6px;'>
      <p style='margin:0 0 4px 0; color:#58a6ff; font-size:11px;
                font-weight:700; text-transform:uppercase;
                letter-spacing:0.5px;'>.LIF files (Leica)</p>
      <code style='color:#79c0ff; font-size:11px;'>pip install readlif</code>
    </div>

  </div>
</body>
"""
        dlg = QDialog(self)
        dlg.setWindowTitle("Help — Protein Aggregate Analyzer")
        dlg.resize(820, 680)
        dlg.setStyleSheet(DARK_THEME)
        lyt = QVBoxLayout(dlg)
        lyt.setContentsMargins(0, 0, 0, 10)
        lyt.setSpacing(8)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setHtml(txt)
        te.setStyleSheet(
            "QTextEdit { background-color: #0d1117; color: #c9d1d9; "
            "border: none; font-family: 'Segoe UI', Arial, sans-serif; "
            "font-size: 13px; }"
        )
        lyt.addWidget(te)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        bb.setStyleSheet(
            "QDialogButtonBox { padding: 0 12px; }"
            "QPushButton { min-width: 100px; padding: 7px 20px; "
            "background:#1f6feb; color:#fff; border:none; border-radius:6px; "
            "font-weight:700; font-size:12px; }"
            "QPushButton:hover { background:#388bfd; }"
        )
        lyt.addWidget(bb)
        dlg.exec_()

    def _show_welcome(self):
        self.status.showMessage(
            "Welcome! — Open a .LIF or .TIF file using 📂 Open."
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Protein Aggregate Analyzer")
    app.setOrganizationName("ConfocalLab")
    app.setStyle("Fusion")
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()