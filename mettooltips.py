#!/usr/bin/env python3
"""
Protein Aggregate Analyzer — Custom Versie
=====================================================
Aangepaste versie voor de eindgebruiker:
  ★ Tab 1, 2 en 3 (Viewer, Pre-processing, Cellichamen) behouden
  ★ Tab 4 (Traditionele Segmentatie) verwijderd
  ★ Tab 5 (Deep Learning) bevat nu exclusief de Ensemble methode
  ★ Tab 6 (Validatie) aangepast voor uitsluitend Deep Learning vergelijking

Vereisten: PyQt5, numpy, scipy, scikit-image, matplotlib, tifffile
Optioneel:  readlif (voor .lif bestanden), torch, segmentation_models_pytorch, cellpose
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
#  KLEURTHEMA  —  Professional Dark UI
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
            raise ImportError("readlif is niet geïnstalleerd.\nInstalleer met: pip install readlif")
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
#  PRE-PROCESSING  (uitgebreid)
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
#  HULPFUNCTIES VOOR SEGMENTATIE & WEERGAVE
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

        grp_file = QGroupBox("📂  Bestand laden")
        fv = QVBoxLayout(grp_file)
        self.btn_open = QPushButton("Open .LIF / .TIF bestand")
        self.btn_open.setObjectName("primary")
        self.btn_open.setToolTip(
            "Open een microscopie-bestand.\n"
            "Ondersteunde formaten:\n"
            "  • .LIF  — Leica Image File (meerdere series mogelijk)\n"
            "  • .TIF / .TIFF — standaard TIFF (ook multi-channel/Z-stack)"
        )
        self.btn_open.clicked.connect(self._open_file)
        fv.addWidget(self.btn_open)
        self.lbl_file = QLabel("Geen bestand geladen")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet("color:#58a6ff; font-size:11px;")
        fv.addWidget(self.lbl_file)
        self.cmb_series = QComboBox()
        self.cmb_series.setToolTip(
            "Kies de te bekijken serie (experiment) binnen het geladen bestand.\n"
            "LIF-bestanden kunnen meerdere opnames (series) bevatten;\n"
            "TIF-bestanden hebben doorgaans maar één serie."
        )
        self.cmb_series.currentIndexChanged.connect(self._series_changed)
        fv.addWidget(QLabel("Serie:"))
        fv.addWidget(self.cmb_series)
        lv.addWidget(grp_file)

        grp_view = QGroupBox("🎨  Weergave-instellingen")
        vv = QFormLayout(grp_view)
        self.cmb_channel = QComboBox()
        self.cmb_channel.setToolTip(
            "Kies welk fluorescentiekanaal je wilt bekijken.\n"
            "Elk kanaal komt overeen met een andere kleurstof of marker\n"
            "(bijv. DAPI voor kernen, GFP voor eiwitten)."
        )
        self.cmb_channel.currentIndexChanged.connect(self._refresh_image)
        vv.addRow("Kanaal:", self.cmb_channel)
        self.cmb_display = QComboBox()
        self.cmb_display.addItems(["Max Projectie", "Z-Slice"])
        self.cmb_display.setToolTip(
            "Max Projectie: toont de maximale pixelwaarde over alle Z-lagen.\n"
            "  → Geeft een volledig overzicht van alle structuren in het volume.\n\n"
            "Z-Slice: toont één specifieke Z-laag tegelijk.\n"
            "  → Gebruik de schuifbalk om door de lagen te navigeren."
        )
        self.cmb_display.currentIndexChanged.connect(self._toggle_view_mode)
        vv.addRow("Modus:", self.cmb_display)
        self.sld_z = QSlider(Qt.Horizontal)
        self.sld_z.setMinimum(0); self.sld_z.setMaximum(0)
        self.sld_z.setToolTip(
            "Schuif om door de Z-lagen (diepte) van de stack te bladeren.\n"
            "Alleen actief in de modus 'Z-Slice'."
        )
        self.sld_z.valueChanged.connect(self._refresh_image)
        self.sld_z.setEnabled(False)
        self.lbl_z = QLabel("Z: 0 / 0")
        vv.addRow(self.lbl_z, self.sld_z)
        lv.addWidget(grp_view)

        grp_cmap = QGroupBox("🌈  Kleurkaart")
        cv = QFormLayout(grp_cmap)
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(["hot", "gray", "inferno", "viridis",
                                 "magma", "plasma", "cividis", "turbo"])
        self.cmb_cmap.setToolTip(
            "Kies de kleurkaart voor de weergave van het beeld:\n"
            "  • hot      — zwart → rood → geel → wit (goed voor aggregaten)\n"
            "  • gray     — grijswaarden (standaard microscopie)\n"
            "  • inferno  — zwart → paars → oranje → wit\n"
            "  • viridis  — donkerblauw → groen → geel (kleurblindveilig)\n"
            "  • magma    — zwart → paars → roze → wit\n"
            "  • plasma   — blauw → paars → geel\n"
            "  • cividis  — blauw → groen → geel (kleurblindveilig)\n"
            "  • turbo    — regenboog met betere perceptie"
        )
        self.cmb_cmap.currentIndexChanged.connect(self._refresh_image)
        cv.addRow("Kleurkaart:", self.cmb_cmap)
        self.chk_autoscale = QCheckBox("Auto-schaal intensiteit")
        self.chk_autoscale.setChecked(True)
        self.chk_autoscale.setToolTip(
            "Als aangevinkt: past de helderheid automatisch aan op de\n"
            "minimum- en maximumwaarde van het zichtbare beeld.\n\n"
            "Als uitgevinkt: gebruikt een vaste schaal van 0 tot de\n"
            "maximale pixelwaarde in het beeld."
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
        filt = "Microscopie-bestanden (*.lif *.tif *.tiff);;Alle bestanden (*)"
        path, _ = QFileDialog.getOpenFileName(self, "Open bestand", "", filt)
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
            QMessageBox.critical(self, "Laadmelding", str(e))

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
#  TAB 2 — PRE-PROCESSING (uitgebreid)
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

        grp_quick = QGroupBox("⚡  Snelknoppen")
        qv = QVBoxLayout(grp_quick)
        btn_recommended = QPushButton("★  Aanbevolen pipeline (aggregaten)")
        btn_recommended.setObjectName("primary")
        btn_recommended.setToolTip(
            "Laadt de aanbevolen instellingen:\n"
            "• Achtergrondsubtractie AAN — Gaussiaan (σ=50)\n"
            "• Ruisonderdrukking AAN — Gaussiaan (σ=1.0)\n"
            "• Alle overige stappen UIT\n\n"
            "Optimaal startpunt voor eiwit-aggregaat analyse in confocale microscopie."
        )
        btn_recommended.clicked.connect(self._set_recommended)
        qv.addWidget(btn_recommended)
        btn_reset_all = QPushButton("↺  Alles terugzetten")
        btn_reset_all.setToolTip(
            "Zet alle pre-processing stappen terug naar de fabrieksinstellingen\n"
            "en toont het originele, onbewerkte beeld."
        )
        btn_reset_all.clicked.connect(self._reset)
        qv.addWidget(btn_reset_all)
        lv.addWidget(grp_quick)

        grp_steps = QGroupBox("🔧  Pre-processing stappen")
        sv = QVBoxLayout(grp_steps)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        iv = QVBoxLayout(inner)

        # Normalisatie
        self.chk_norm = QCheckBox("Percentiel-normalisatie")
        self.chk_norm.setChecked(True)
        self.chk_norm.setToolTip(
            "Schaalt de pixelintensiteiten zodat pmin% de donkerste\n"
            "en pmax% de helderste waarde wordt.\n\n"
            "Vermindert de invloed van extreme (uitbijter) pixels\n"
            "en maakt beelden van verschillende opnames vergelijkbaar."
        )
        iv.addWidget(self.chk_norm)
        pn = QHBoxLayout()
        pn.addWidget(QLabel("pmin:"))
        self.spn_pmin = QDoubleSpinBox()
        self.spn_pmin.setRange(0, 49); self.spn_pmin.setValue(1.0); self.spn_pmin.setSingleStep(0.5)
        self.spn_pmin.setToolTip(
            "Onderste percentielpunt (%) voor normalisatie.\n"
            "Pixels onder deze drempel worden op 0 gezet.\n"
            "Standaard: 1.0% — filtert ruis in de donkere achtergrond."
        )
        pn.addWidget(self.spn_pmin)
        pn.addWidget(QLabel("pmax:"))
        self.spn_pmax = QDoubleSpinBox()
        self.spn_pmax.setRange(51, 100); self.spn_pmax.setValue(99.9); self.spn_pmax.setSingleStep(0.5)
        self.spn_pmax.setToolTip(
            "Bovenste percentielpunt (%) voor normalisatie.\n"
            "Pixels boven deze drempel worden op 1 gezet.\n"
            "Standaard: 99.9% — voorkomt dat één heldere pixel\n"
            "de schaal verpest."
        )
        pn.addWidget(self.spn_pmax)
        iv.addLayout(pn)
        self._add_sep(iv)

        # Achtergrondsubtractie
        self.chk_bg = QCheckBox("Achtergrondsubtractie")
        self.chk_bg.setToolTip(
            "Verwijdert de diffuse achtergrondgloed uit het beeld.\n"
            "Belangrijk wanneer het beeld een ongelijkmatige verlichtingsachtergrond heeft.\n\n"
            "Aanbevolen bij aggregaat-analyse om vals-positieve detecties te vermijden."
        )
        iv.addWidget(self.chk_bg)
        bg_type_row = QHBoxLayout()
        bg_type_row.addWidget(QLabel("Methode:"))
        self.cmb_bg_method = QComboBox()
        self.cmb_bg_method.addItems(["Rolling Ball (morfologisch)", "Gaussiaan"])
        self.cmb_bg_method.setToolTip(
            "Rolling Ball (morfologisch):\n"
            "  Schat achtergrond via morfologische opening (wit tophat).\n"
            "  Goed voor lokale, ongelijkmatige achtergronden.\n\n"
            "Gaussiaan:\n"
            "  Past een grote Gaussiaanse blur toe als schatting van de achtergrond.\n"
            "  Snel en effectief bij geleidelijk variërende achtergronden.\n"
            "  Aanbevolen voor aggregaat-analyse (σ ≈ 50 px)."
        )
        bg_type_row.addWidget(self.cmb_bg_method)
        iv.addLayout(bg_type_row)
        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("Radius/σ (px):"))
        self.spn_bg_radius = QSpinBox()
        self.spn_bg_radius.setRange(5, 500); self.spn_bg_radius.setValue(30)
        self.spn_bg_radius.setToolTip(
            "Straal (in pixels) voor de achtergrondschatting.\n\n"
            "Rolling Ball: groter = grovere achtergrondschatting.\n"
            "Gaussiaan (σ): groter = meer blur, meer grote structuren worden als achtergrond gezien.\n\n"
            "Richtlijn: minstens 2–3× de maximale aggregaatgrootte.\n"
            "Aanbevolen bij Gaussiaan: σ = 50 px."
        )
        bg_row.addWidget(self.spn_bg_radius)
        iv.addLayout(bg_row)
        self._add_sep(iv)

        # Top-hat
        self.chk_tophat = QCheckBox("Top-Hat Filter")
        self.chk_tophat.setChecked(True)
        self.chk_tophat.setToolTip(
            "Versterkt kleine, heldere structuren (aggregaten) ten opzichte\n"
            "van de omgevende achtergrond.\n\n"
            "Werkt door de morfologische opening van het beeld af te trekken,\n"
            "waardoor alleen structuren kleiner dan de opgegeven radius overblijven."
        )
        iv.addWidget(self.chk_tophat)
        th_mode_row = QHBoxLayout()
        th_mode_row.addWidget(QLabel("Modus:"))
        self.cmb_tophat_mode = QComboBox()
        self.cmb_tophat_mode.addItems(["Enkelvoudig", "Multi-schaal (aanbevolen)"])
        self.cmb_tophat_mode.setCurrentIndex(1)
        self.cmb_tophat_mode.setToolTip(
            "Enkelvoudig:\n"
            "  Gebruikt één vaste straal voor de top-hat filter.\n\n"
            "Multi-schaal (aanbevolen):\n"
            "  Combineert drie stralen tegelijk: r-2, r en r+3.\n"
            "  Detecteert aggregaten van verschillende groottes in één stap.\n"
            "  Robuuster bij heterogene preparaten."
        )
        th_mode_row.addWidget(self.cmb_tophat_mode)
        iv.addLayout(th_mode_row)
        th_row = QHBoxLayout()
        th_row.addWidget(QLabel("Radius (px):"))
        self.spn_tophat = QSpinBox()
        self.spn_tophat.setRange(1, 50); self.spn_tophat.setValue(6)
        self.spn_tophat.setToolTip(
            "Straal van het structurerend element (schijf) in pixels.\n\n"
            "Kies een waarde iets groter dan de typische aggregaatstraal.\n"
            "Te klein: achtergrond wordt niet goed onderdrukt.\n"
            "Te groot: kleine aggregaten worden weggefilterd.\n"
            "Typische waarde: 4–10 px afhankelijk van de microscopie-resolutie."
        )
        th_row.addWidget(self.spn_tophat)
        iv.addLayout(th_row)
        info_th = QLabel("Multi-schaal: gebruikt r, r+3, r-2 tegelijk")
        info_th.setStyleSheet("color:#8b949e; font-size:11px;")
        iv.addWidget(info_th)
        self._add_sep(iv)

        # Denoise
        self.chk_denoise = QCheckBox("Ruisonderdrukking")
        self.chk_denoise.setToolTip(
            "Vermindert ruis in het beeld vóór segmentatie.\n"
            "Vermindert vals-positieve detecties door ruis-pieken.\n\n"
            "Let op: te veel onderdrukking kan kleine aggregaten vervagen."
        )
        iv.addWidget(self.chk_denoise)
        dn_mode_row = QHBoxLayout()
        dn_mode_row.addWidget(QLabel("Methode:"))
        self.cmb_denoise_mode = QComboBox()
        self.cmb_denoise_mode.addItems(["Gaussiaan", "Bilateral (behoudt randen) ★"])
        self.cmb_denoise_mode.setCurrentIndex(1)
        self.cmb_denoise_mode.setToolTip(
            "Gaussiaan:\n"
            "  Snelle, isotrope vervaging. Eenvoudig maar vervaagt ook randen.\n"
            "  Goed voor hoog-ruisige beelden waar randbehoud minder belangrijk is.\n\n"
            "Bilateral (★ aanbevolen):\n"
            "  Onderdrukt ruis terwijl scherpe randen (aggregaatgrenzen) behouden blijven.\n"
            "  Langzamer maar kwalitatief beter voor aggregaat-detectie."
        )
        dn_mode_row.addWidget(self.cmb_denoise_mode)
        iv.addLayout(dn_mode_row)
        dn_row = QHBoxLayout()
        dn_row.addWidget(QLabel("Sigma:"))
        self.spn_sigma = QDoubleSpinBox()
        self.spn_sigma.setRange(0.1, 10.0); self.spn_sigma.setValue(0.8); self.spn_sigma.setSingleStep(0.1)
        self.spn_sigma.setToolTip(
            "Sterkte van de ruisonderdrukking (standaardafwijking van de Gaussiaan).\n\n"
            "Gaussiaan: hogere sigma = meer vervaging.\n"
            "Bilateral: hogere sigma = groter ruimtelijk bereik van de filter.\n\n"
            "Typische waarden: 0.5–2.0.\n"
            "Begin laag (0.8) en verhoog alleen als er veel ruis zichtbaar is."
        )
        dn_row.addWidget(self.spn_sigma)
        iv.addLayout(dn_row)
        self._add_sep(iv)

        inner.setLayout(iv)
        scroll.setWidget(inner)
        sv.addWidget(scroll)
        lv.addWidget(grp_steps)

        btn_apply = QPushButton("▶  Pre-processing toepassen")
        btn_apply.setObjectName("primary")
        btn_apply.setToolTip(
            "Past alle aangevinkte pre-processing stappen toe op het huidige beeld\n"
            "in de volgorde: achtergrondsubtractie → top-hat → ruisonderdrukking → normalisatie.\n\n"
            "Het resultaat wordt rechts getoond en doorgegeven aan de segmentatie-tabs."
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
            QMessageBox.warning(self, "Geen beeld", "Laad eerst een beeld.")
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
        for ax, img, title in zip(axes, [before, after], ["Origineel", "Na pre-processing"]):
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
#  STATISTIEKEN-DIALOOG
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
            ax.set_title("Oppervlak (px²)", color="#79c0ff")
            ax.set_facecolor("#0d1117"); ax.tick_params(colors="#79c0ff")
            intns = [p.get("mean_intensity", 0) for p in result.properties]
            ax2 = fig.add_subplot(122)
            ax2.hist(intns, bins=30, color="#79c0ff", edgecolor="#161b22")
            ax2.set_title("Intensiteitsverdeling", color="#79c0ff")
            ax2.set_facecolor("#0d1117"); ax2.tick_params(colors="#79c0ff")
            fig.tight_layout()
        lyt.addWidget(canvas)

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lyt.addWidget(bb)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — CELLPOSE CELLICHAAM SEGMENTATIE
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

            self.progress.emit("⏳  Cellpose model laden…")
            model = cp_models.CellposeModel(gpu=use_gpu, model_type=self.model_type)
            
            img2d = self.img
            if img2d.ndim == 3:
                img2d = img2d.max(axis=0)

            self.progress.emit(
                f"⏳  Segmentatie op kanaal {self.kanaal} "
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

            self.progress.emit("⏳  Celkerngaten opvullen…")
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
                "⚠  Cellpose is niet geïnstalleerd.\n"
                "Installeer met:\n  pip install cellpose"
            )
            warn.setStyleSheet("color:#ff8800; font-weight:bold; padding:10px;")
            warn.setWordWrap(True)
            lv.addWidget(warn)

        grp_model = QGroupBox("🔬  Model & Kanaal")
        mf = QFormLayout(grp_model)

        self.spn_kanaal = QSpinBox()
        self.spn_kanaal.setRange(0, 15)
        self.spn_kanaal.setValue(2)
        mf.addRow("Segmentatie-kanaal:", self.spn_kanaal)

        self.cmb_model = QComboBox()
        self.cmb_model.addItems(["cyto2", "cyto", "nuclei", "cyto3"])
        self.cmb_model.setCurrentText("cyto2")
        self.cmb_model.setToolTip(
            "Kies het Cellpose-model passend bij je preparaat:\n\n"
            "  • cyto2   — verbeterd cytoplasma-model (aanbevolen voor cellen)\n"
            "  • cyto    — origineel cytoplasma-model\n"
            "  • nuclei  — geoptimaliseerd voor celkernen (DAPI/Hoechst)\n"
            "  • cyto3   — nieuwste generatie cytoplasma-model\n\n"
            "Probeer cyto2 als startpunt voor de meeste cellijnen."
        )
        mf.addRow("Model:", self.cmb_model)

        self.spn_diameter = QSpinBox()
        self.spn_diameter.setRange(0, 999)
        self.spn_diameter.setValue(80)
        self.spn_diameter.setSpecialValueText("Auto")
        self.spn_diameter.setToolTip(
            "Verwachte celdiameter in pixels.\n\n"
            "Stel in op 0 voor automatische schatting door Cellpose.\n"
            "Bij handmatige opgave: meet een representatieve cel in de viewer\n"
            "en vul de diameter in pixels in.\n\n"
            "Te klein: cellen worden gesplitst.\n"
            "Te groot: meerdere cellen worden samengevoegd."
        )
        mf.addRow("Celdiameter (px, 0=auto):", self.spn_diameter)

        lv.addWidget(grp_model)

        grp_thr = QGroupBox("⚙  Segmentatie-parameters")
        tf = QFormLayout(grp_thr)

        self.spn_flow = QDoubleSpinBox()
        self.spn_flow.setRange(0.1, 1.0)
        self.spn_flow.setSingleStep(0.05)
        self.spn_flow.setValue(0.8)
        self.spn_flow.setToolTip(
            "Flow threshold: maximaal toegestane fout in de optische stroomvelden.\n\n"
            "Lager (bijv. 0.4): accepteert meer imperfecte segmentaties → meer cellen\n"
            "Hoger (bijv. 0.9): strenger, alleen goed-gevormde maskers → minder cellen\n\n"
            "Standaard: 0.8 — goed startpunt voor de meeste preparaten.\n"
            "Verlaag als te weinig cellen worden gevonden."
        )
        tf.addRow("Flow threshold:", self.spn_flow)

        self.spn_cellprob = QDoubleSpinBox()
        self.spn_cellprob.setRange(-8.0, 6.0)
        self.spn_cellprob.setSingleStep(0.5)
        self.spn_cellprob.setValue(-4.0)
        self.spn_cellprob.setToolTip(
            "Cel-kansdrempel: minimale voorspelde kans om als cel te worden meegenomen.\n\n"
            "Lager (bijv. -6.0): meer pixels worden als cel gezien → grotere maskers\n"
            "Hoger (bijv. 0.0): alleen de meest zekere gebieden → kleinere/minder maskers\n\n"
            "Standaard: -4.0 — liberale instelling die ook zwak-gelabelde cellen meeneemt.\n"
            "Verhoog als te veel achtergrond als cel wordt gedetecteerd."
        )
        tf.addRow("Cellprob threshold:", self.spn_cellprob)

        btn_reset = QPushButton("↺  Herstel standaardwaarden")
        btn_reset.clicked.connect(self._reset_params)
        tf.addRow(btn_reset)

        lv.addWidget(grp_thr)

        grp_info = QGroupBox("ℹ  Werkwijze")
        iv = QVBoxLayout(grp_info)
        lbl_info = QLabel(
            "Cellpose berekent een <b>max-intensiteitsprojectie</b> over alle "
            "Z-slices op het gekozen kanaal en segmenteert daarop de cellichamen.\n\n"
            "Het resulterende binaire masker kan in verdere stappen worden gebruikt "
            "om achtergrond te filteren."
        )
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet("color:#8b949e; font-size:11px; padding:4px;")
        iv.addWidget(lbl_info)
        lv.addWidget(grp_info)

        self.btn_run = QPushButton("▶  Cellichamen segmenteren")
        self.btn_run.setObjectName("primary")
        self.btn_run.setEnabled(HAS_CELLPOSE)
        self.btn_run.setToolTip(
            "Start de Cellpose cellichaam-segmentatie op het huidige beeld.\n\n"
            "Cellpose berekent een max-projectie over alle Z-lagen en segmenteert\n"
            "de cellichamen automatisch op basis van het gekozen model.\n\n"
            "Het resulterende binaire masker wordt gebruikt om aggregaat-detecties\n"
            "te beperken tot het cellichaam (achtergrond wordt gemaskeerd).\n\n"
            "Vereist: pip install cellpose"
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

        grp_stats = QGroupBox("📊  Resultaat")
        sv = QVBoxLayout(grp_stats)
        self.txt_stats = QTextEdit()
        self.txt_stats.setReadOnly(True)
        self.txt_stats.setMinimumHeight(80)
        self.txt_stats.setMaximumHeight(140)
        sv.addWidget(self.txt_stats)
        lv.addWidget(grp_stats)

        btn_export = QPushButton("💾  Exporteer celmasker als TIFF")
        btn_export.setToolTip(
            "Slaat het gesegmenteerde celmasker op als TIFF-bestand.\n\n"
            "Het masker is binair: wit (255) = cellichaam, zwart (0) = achtergrond.\n"
            "Kan later opnieuw worden ingeladen of gebruikt voor batch-verwerking."
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
            QMessageBox.warning(self, "Geen beeld", "Laad eerst een beeld.")
            return

        diameter = self.spn_diameter.value() or None

        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.lbl_status.setText("Bezig…")
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

        self.lbl_status.setText(f"✅  Klaar — {n} cellen gedetecteerd in {t:.1f}s")
        self.txt_stats.setPlainText(
            f"Model          : {result['model']}\n"
            f"Diameter       : {diam} px\n"
            f"Flow threshold : {self.spn_flow.value():.2f}\n"
            f"Cellprob thr.  : {self.spn_cellprob.value():.1f}\n"
            f"Gedetect. cellen: {n}\n"
            f"Rekentijd      : {t:.2f} s\n"
            f"Masker dekt    : {self.cell_mask.mean()*100:.1f}% van het beeld"
        )

        self._display_result(result)
        self.mask_ready.emit(self.cell_mask)

    def _on_error(self, err: str):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self.lbl_status.setText("❌  Fout tijdens segmentatie")
        QMessageBox.critical(self, "Cellpose fout", err)

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
        axes[0].set_title("Max-projectie (invoer)", **TITLE)

        axes[1].imshow(mask, cmap="nipy_spectral", interpolation="nearest")
        axes[1].set_title(f"Cel maskers — {result['n_cells']} cellen", **TITLE)

        axes[2].imshow(img_masked, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        axes[2].set_title("Alleen cellichamen", **TITLE)

        for ax in axes:
            ax.axis("off")
            ax.set_facecolor(BG)

        self.canvas.fig.set_facecolor(BG)
        self.canvas.fig.tight_layout(pad=0.5)
        self.canvas.draw_idle()

    def _export_mask(self):
        if self.cell_mask is None:
            QMessageBox.warning(self, "Geen masker", "Voer eerst cellichaamdeterminatie uit.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Sla celmasker op", "cellpose_masker.tif",
            "TIFF (*.tif);;Alle bestanden (*)"
        )
        if not path:
            return
        tifffile.imwrite(path, self.cell_mask.astype(np.uint8) * 255)
        QMessageBox.information(self, "Opgeslagen", f"Celmasker opgeslagen:\n{path}")


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
                f"Geen model_fold*.pth bestanden gevonden in:\n{model_dir}"
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
                    "segmentation_models_pytorch is niet geïnstalleerd.\n"
                    "Installeer met:  pip install segmentation-models-pytorch"
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
#  TAB 4 — DEEP LEARNING SEGMENTATIE (Ensemble Only)
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
                "Installeer met:\n  pip install torch torchvision"
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

        grp_ens = QGroupBox("🗂  Ensemble Modellen (map met model_fold*.pth)")
        ev = QVBoxLayout(grp_ens)

        ens_path_row = QHBoxLayout()
        self.lbl_ensemble_path = QLabel("Geen map geselecteerd")
        self.lbl_ensemble_path.setStyleSheet("color:#8b949e; font-size:11px;")
        self.lbl_ensemble_path.setWordWrap(True)
        ens_path_row.addWidget(self.lbl_ensemble_path, stretch=1)
        btn_load_dir = QPushButton("📂  Laad modellenmap")
        btn_load_dir.clicked.connect(self._load_model_dir)
        ens_path_row.addWidget(btn_load_dir)
        ev.addLayout(ens_path_row)

        self.lbl_ensemble_info = QLabel("")
        self.lbl_ensemble_info.setStyleSheet("color:#58a6ff; font-size:11px;")
        self.lbl_ensemble_info.setWordWrap(True)
        ev.addWidget(self.lbl_ensemble_info)

        ens_thr_row = QHBoxLayout()
        self.chk_auto_thr = QCheckBox("Auto-threshold (gemiddeld uit modellen)")
        self.chk_auto_thr.setChecked(True)
        self.chk_auto_thr.setToolTip(
            "Als aangevinkt: gebruikt automatisch de optimale threshold\n"
            "die tijdens de training van elk model is bepaald.\n"
            "De uiteindelijke threshold is het gemiddelde over alle folds.\n\n"
            "Aanbevolen voor de meeste situaties."
        )
        self.chk_auto_thr.toggled.connect(self._on_auto_thr_toggled)
        ens_thr_row.addWidget(self.chk_auto_thr)
        ev.addLayout(ens_thr_row)

        ens_thr2_row = QHBoxLayout()
        ens_thr2_row.addWidget(QLabel("Vaste threshold:"))
        self.spn_ens_threshold = QDoubleSpinBox()
        self.spn_ens_threshold.setRange(0.01, 0.99)
        self.spn_ens_threshold.setValue(0.5)
        self.spn_ens_threshold.setSingleStep(0.05)
        self.spn_ens_threshold.setEnabled(False)
        self.spn_ens_threshold.setToolTip(
            "Handmatige threshold voor de ensemble-kanskaart (0.01–0.99).\n\n"
            "Pixels met een voorspelde kans ≥ threshold worden als aggregaat gelabeld.\n\n"
            "Lager (bijv. 0.3): meer/grotere detecties, meer vals-positieven.\n"
            "Hoger (bijv. 0.7): minder/kleinere detecties, minder vals-positieven.\n\n"
            "Alleen actief als 'Auto-threshold' is uitgevinkt."
        )
        ens_thr2_row.addWidget(self.spn_ens_threshold)
        ev.addLayout(ens_thr2_row)

        self.chk_ens_tta = QCheckBox("Test-Time Augmentation (TTA, 8×)")
        self.chk_ens_tta.setChecked(True)
        self.chk_ens_tta.setToolTip(
            "Middelt 8 augmentaties (4 rotaties + 2 spiegelingen).\n"
            "Verbetert kwaliteit maar is ~8× langzamer."
        )
        ev.addWidget(self.chk_ens_tta)

        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Device:"))
        self.cmb_device = QComboBox()
        self.cmb_device.addItems(["auto", "cpu", "cuda", "mps"])
        self.cmb_device.setToolTip(
            "Kies op welke hardware de inferentie wordt uitgevoerd:\n\n"
            "  • auto  — kiest automatisch GPU (cuda) als beschikbaar, anders cpu\n"
            "  • cpu   — gebruik de processor (langzamer, altijd beschikbaar)\n"
            "  • cuda  — gebruik een NVIDIA GPU (veel sneller, vereist CUDA-driver)\n"
            "  • mps   — gebruik Apple Silicon GPU (M1/M2/M3 Mac)\n\n"
            "Bij CUDA-fouten: stel in op 'cpu' als tijdelijke oplossing."
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
            "Minimaal oppervlak (in pixels²) van een gedetecteerd object.\n\n"
            "Objecten kleiner dan deze waarde worden verwijderd als ruis.\n\n"
            "Te laag: ruis-pieken worden meegenomen als vals-positieven.\n"
            "Te hoog: kleine echte aggregaten worden weggefilterd.\n\n"
            "Typische waarde: 5–50 px² afhankelijk van de microscopie-resolutie."
        )
        pf.addRow("Min oppervlak (px²):", self.spn_min_area)

        self.spn_max_area = QSpinBox()
        self.spn_max_area.setRange(1, 999999)
        self.spn_max_area.setValue(50000)
        self.spn_max_area.setToolTip(
            "Maximaal oppervlak (in pixels²) van een gedetecteerd object.\n\n"
            "Objecten groter dan deze waarde worden verwijderd.\n"
            "Voorkomt dat grote artefacten (bijv. dode cellen, debris) worden meegenomen.\n\n"
            "Stel hoog in als je ook grote aggregaatclusters wilt detecteren."
        )
        pf.addRow("Max oppervlak (px²):", self.spn_max_area)

        lv.addWidget(grp_post)

        grp_ov = QGroupBox("🖍  Overlay-opties")
        ovf = QFormLayout(grp_ov)
        self.chk_show_circles = QCheckBox("Teken contouren")
        self.chk_show_circles.setChecked(True)
        self.chk_show_circles.setToolTip(
            "Tekent de omtreklijn van elk gedetecteerd object over het beeld.\n"
            "Maakt de exacte grenzen van de segmentatie zichtbaar."
        )
        self.chk_show_numbers = QCheckBox("Toon nummers")
        self.chk_show_numbers.setChecked(True)
        self.chk_show_numbers.setToolTip(
            "Toont het ID-nummer van elk object in het centrum van de contour.\n"
            "Handig om specifieke objecten terug te vinden in de CSV-export."
        )
        self.chk_show_fill    = QCheckBox("Gevuld gebied")
        self.chk_show_fill.setChecked(True)
        self.chk_show_fill.setToolTip(
            "Kleurt het oppervlak van elk gedetecteerd object in met een semi-transparante kleur.\n"
            "Geeft een beter overzicht van de totale segmentatie dan alleen contouren."
        )
        self.cmb_circle_color = QComboBox()
        self.cmb_circle_color.addItems(["#00ffcc","#ff4466","#ffff00","#ffffff","#00aaff","#ff8800"])
        self.cmb_circle_color.setToolTip(
            "Kleur van de getekende contouren:\n"
            "  #00ffcc — cyaan (standaard, goed zichtbaar op donkere achtergrond)\n"
            "  #ff4466 — rood/roze\n"
            "  #ffff00 — geel\n"
            "  #ffffff — wit\n"
            "  #00aaff — blauw\n"
            "  #ff8800 — oranje"
        )
        self.spn_circle_lw  = QDoubleSpinBox()
        self.spn_circle_lw.setRange(0.3, 5); self.spn_circle_lw.setValue(1.2)
        self.spn_circle_lw.setToolTip(
            "Lijnbreedte van de getekende contouren in punten.\n\n"
            "Dunner (0.3–1.0): minder opvallend, meer detail zichtbaar.\n"
            "Dikker (2.0–5.0): beter zichtbaar bij kleine objecten of exportafbeeldingen."
        )
        self.spn_font_size  = QDoubleSpinBox()
        self.spn_font_size.setRange(3, 16); self.spn_font_size.setValue(6.5)
        self.spn_font_size.setToolTip(
            "Lettergrootte van de object-ID-nummers in punten.\n\n"
            "Pas aan op basis van de grootte van de objecten in het beeld:\n"
            "klein voor kleine aggregaten (4–6), groter voor cellen (8–12)."
        )
        self.cmb_cmap       = QComboBox()
        self.cmb_cmap.addItems(["hot","gray","inferno","magma","viridis","plasma"])
        self.cmb_cmap.setToolTip(
            "Kleurkaart voor de achtergrondafbeelding in de resultatenweergave.\n\n"
            "  • hot     — zwart → rood → wit (goed voor fluorescentiemicroscopie)\n"
            "  • gray    — grijswaarden\n"
            "  • inferno — zwart → paars → oranje → wit\n"
            "  • magma   — zwart → paars → roze → wit\n"
            "  • viridis — donkerblauw → groen → geel (kleurblindveilig)\n"
            "  • plasma  — blauw → paars → geel"
        )
        ovf.addWidget(self.chk_show_circles)
        ovf.addWidget(self.chk_show_numbers)
        ovf.addWidget(self.chk_show_fill)
        ovf.addRow("Kleur:",         self.cmb_circle_color)
        ovf.addRow("Lijnbreedte:",   self.spn_circle_lw)
        ovf.addRow("Lettergrootte:", self.spn_font_size)
        ovf.addRow("Achtergrond:",   self.cmb_cmap)
        lv.addWidget(grp_ov)

        self.btn_run_ensemble = QPushButton("▶  Ensemble Segmentatie uitvoeren")
        self.btn_run_ensemble.setObjectName("primary")
        self.btn_run_ensemble.setToolTip(
            "Start de ensemble deep learning segmentatie op het huidige beeld.\n\n"
            "Het algoritme:\n"
            "  1. Laadt alle model_fold*.pth bestanden uit de geselecteerde map\n"
            "  2. Voert inferentie uit op elk model (eventueel met TTA)\n"
            "  3. Middelt de kanskaarten van alle modellen\n"
            "  4. Drempelt de gemiddelde kanskaart (threshold)\n"
            "  5. Past morfologische filtering toe (min/max oppervlak)\n\n"
            "Let op: dit kan enkele minuten duren op een CPU."
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

        grp_stats = QGroupBox("📊  Statistieken")
        sv = QVBoxLayout(grp_stats)
        self.txt_stats = QTextEdit()
        self.txt_stats.setReadOnly(True)
        self.txt_stats.setMinimumHeight(90)
        sv.addWidget(self.txt_stats)
        lv.addWidget(grp_stats)

        btn_csv = QPushButton("💾  Exporteer CSV")
        btn_csv.setToolTip(
            "Exporteert de eigenschappen van alle gedetecteerde objecten naar een CSV-bestand.\n\n"
            "Kolommen per object:\n"
            "  • id, x, y         — identificatie en positie (centroïde)\n"
            "  • area_px2         — oppervlak in pixels²\n"
            "  • mean/max_intensity — gemiddelde en maximale pixelintensiteit\n"
            "  • eccentricity     — mate van ellipsvorm (0=cirkel, 1=lijnstuk)\n"
            "  • perimeter        — omtreklengte in pixels\n"
            "  • solidity         — vulgraad (convex hull)\n"
            "  • radius_px        — equivalente straal"
        )
        btn_csv.clicked.connect(self._export_csv)
        lv.addWidget(btn_csv)

        btn_img = QPushButton("🖼  Exporteer geannoteerd beeld")
        btn_img.setToolTip(
            "Slaat een afbeelding op met het originele beeld en het geannoteerde\n"
            "segmentatieresultaat naast elkaar (PNG of TIFF).\n\n"
            "De afbeelding toont de contouren en kleuring zoals ingesteld\n"
            "in de Overlay-opties."
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
            self, "Selecteer map met model_fold*.pth bestanden", ""
        )
        if not folder:
            return
        model_files = sorted(Path(folder).glob("model_fold*.pth"))
        if not model_files:
            QMessageBox.warning(
                self, "Geen modellen gevonden",
                f"Geen model_fold*.pth bestanden gevonden in:\n{folder}"
            )
            return
        self._model_dir = folder
        self.lbl_ensemble_path.setText(f"✅  {Path(folder).name}")
        self.lbl_ensemble_info.setText(f"{len(model_files)} modellen gevonden")

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
                        f"{len(model_files)} modellen  |  gem. threshold: {avg:.2f}"
                    )
            except Exception:
                pass

    def _run_ensemble(self):
        if not self._model_dir:
            QMessageBox.warning(self, "Geen map", "Laad eerst een modellenmap.")
            return
        if not HAS_SMP:
            QMessageBox.critical(
                self, "Package ontbreekt",
                "Installeer segmentation_models_pytorch:\n"
                "  pip install segmentation-models-pytorch"
            )
            return
        img = self._get_image()
        if img is None:
            QMessageBox.warning(self, "Geen beeld", "Laad en verwerk eerst een beeld.")
            return

        thr_override = None if self.chk_auto_thr.isChecked() \
                       else self.spn_ens_threshold.value()

        self.btn_run_ensemble.setEnabled(False)
        self.progress.setVisible(True)
        self.lbl_status.setText("Ensemble inferentie bezig…")

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
            f"Klaar: {result.n_objects} objecten  |  {result.time_seconds:.2f}s"
        )
        self._display_result(result)
        self._show_stats(result)
        self.result_ready.emit(result)

    def _on_error(self, err: str):
        self.btn_run_ensemble.setEnabled(True)
        self.progress.setVisible(False)
        self.lbl_status.setText("❌  Fout tijdens inferentie")
        QMessageBox.critical(self, "Deep Learning fout", err)

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
        axes[0].set_title("Pre-processed invoer", color="#79c0ff", fontsize=10)
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
            self.txt_stats.setPlainText(f"Aantal objecten: {result.n_objects}\nGeen eigenschappen.")
            return
        areas = [p.get("area_px2", 0) for p in result.properties]
        ints  = [p.get("mean_intensity", 0) for p in result.properties]
        rads  = [p.get("radius_px", 0) for p in result.properties]
        self.txt_stats.setPlainText(
            f"Methode:        {result.method_name}\n"
            f"Aantal obj.:    {result.n_objects}\n"
            f"Tijd:           {result.time_seconds:.3f} s\n"
            f"\n── Oppervlak (px²) ──\n"
            f"  Gemiddeld: {np.mean(areas):.1f}\n"
            f"  Mediaan:   {np.median(areas):.1f}\n"
            f"  Min/Max:   {np.min(areas):.0f} / {np.max(areas):.0f}\n"
            f"\n── Straal (px) ──\n"
            f"  Gemiddeld: {np.mean(rads):.1f}\n"
            f"  Mediaan:   {np.median(rads):.1f}\n"
            f"\n── Gemiddelde intensiteit ──\n"
            f"  Gemiddeld: {np.mean(ints):.4f}\n"
            f"  Mediaan:   {np.median(ints):.4f}\n"
        )

    def _export_csv(self):
        if not self.current_result or not self.current_result.properties:
            QMessageBox.warning(self, "Geen data", "Voer eerst segmentatie uit.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Sla CSV op", "", "CSV (*.csv)")
        if not path:
            return
        props = self.current_result.properties
        keys  = list(props[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(props)
        QMessageBox.information(self, "Opgeslagen", f"{len(props)} objecten → {path}")

    def _export_image(self):
        if self.current_result is None:
            QMessageBox.warning(self, "Geen resultaat", "Voer eerst segmentatie uit.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Sla geannoteerd beeld op", "",
            "PNG (*.png);;TIFF (*.tif)"
        )
        if not path:
            return
        import matplotlib.pyplot as plt
        img  = self._get_image()
        cmap = self.cmb_cmap.currentText()
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="#0d1117")
        axes[0].imshow(img, cmap=cmap, aspect="equal")
        axes[0].set_title("Invoer", color="#79c0ff"); axes[0].axis("off")
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
        QMessageBox.information(self, "Opgeslagen", f"Beeld opgeslagen: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — CORRIGEER (Ground-truth annotaties corrigeren)
# ═══════════════════════════════════════════════════════════════════════════════

class CorrectionTab(QWidget):
    """
    Werkwijze:
      1. Laad je handmatige ground-truth masker.
      2. De DL-detecties worden erover gelegd:
           GROEN  = DL-regio overlapt voldoende met GT  → al goedgekeurd
           ROOD   = DL-regio overlapt NIET met GT      → klik om toch goed te keuren
      3. Klik op een rode regio om hem alsnog groen (goedgekeurd) te maken.
         Je kunt ook op een groene regio klikken om hem terug rood te zetten.
      4. Stuur het gecorrigeerde masker naar Validatie.
    """
    corrected_mask_ready = pyqtSignal(object)

    _OVERLAP_THRESHOLD = 0.3   # min IoU-overlap om als TP te beschouwen

    def __init__(self, dl_tab, viewer_tab=None, parent=None):
        super().__init__(parent)
        self.dl_tab = dl_tab
        self.viewer_tab = viewer_tab

        self.gt_mask: Optional[np.ndarray] = None
        self.gt_path = ""

        # per DL-label-id: True = goedgekeurd (groen), False = afgekeurd (rood)
        self._region_status: Dict[int, bool] = {}
        # regio-properties cache
        self._dl_regions: List[Dict] = []

        # per GT-label-id: True = actief (blauw), False = verwijderd door gebruiker
        self._gt_region_status: Dict[int, bool] = {}
        # GT-regio-properties cache (gelabeld masker)
        self._gt_label_image: Optional[np.ndarray] = None
        self._gt_regions: List[Dict] = []

        self._cid = None   # matplotlib click-verbinding
        self._build_ui()

    # ──────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Actie-balk ──────────────────────────────────────────────────────
        grp_actions = QGroupBox("⚙️  Acties")
        av = QHBoxLayout(grp_actions)
        av.setSpacing(8)

        self.btn_load_gt = QPushButton("📂  Laad GT-masker")
        self.btn_load_gt.setObjectName("primary")
        self.btn_load_gt.setToolTip(
            "Laad een handmatig geannoteerd ground-truth masker (TIFF-bestand).\n\n"
            "Dit masker bevat de 'correcte' segmentatie waartegen de\n"
            "DL-detecties worden vergeleken.\n\n"
            "Wit (255) = aggregaat aanwezig, Zwart (0) = geen aggregaat."
        )
        self.btn_load_gt.clicked.connect(self._load_gt_mask)
        av.addWidget(self.btn_load_gt)

        self.btn_refresh = QPushButton("🔄  Vernieuw / haal DL-resultaat op")
        self.btn_refresh.setToolTip(
            "Haalt het laatste segmentatieresultaat op uit de Deep Learning-tab\n"
            "en vergelijkt dit automatisch met het geladen GT-masker.\n\n"
            "Groene contouren = DL-detectie overlapt met GT (terecht positief).\n"
            "Rode contouren = DL-detectie overlapt NIET met GT (mogelijk fout-positief).\n"
            "Blauwe contouren = GT-regio (handmatige annotatie)."
        )
        self.btn_refresh.clicked.connect(self._auto_classify_and_draw)
        self.btn_refresh.setEnabled(False)
        av.addWidget(self.btn_refresh)

        self.btn_reset = QPushButton("↺  Reset alle correcties")
        self.btn_reset.setToolTip(
            "Zet alle handmatige correcties terug naar de automatische classificatie.\n"
            "Alle groene/rode statussen worden opnieuw berekend op basis\n"
            "van de overlap met het GT-masker."
        )
        self.btn_reset.clicked.connect(self._reset_corrections)
        self.btn_reset.setEnabled(False)
        av.addWidget(self.btn_reset)

        self.btn_send = QPushButton("✅  Stuur gecorrigeerd masker naar Validatie")
        self.btn_send.setObjectName("primary")
        self.btn_send.setToolTip(
            "Stuurt het gecorrigeerde masker (na jouw aanpassingen) door\n"
            "naar de Validatie-tab voor kwantitatieve evaluatie.\n\n"
            "Goedgekeurde regio's + actieve GT-regio's worden samengevoegd\n"
            "tot het definitieve gecorrigeerde masker."
        )
        self.btn_send.clicked.connect(self._send_to_validation)
        self.btn_send.setEnabled(False)
        av.addWidget(self.btn_send)

        self.btn_save_mask = QPushButton("💾  Sla gecorrigeerd masker op")
        self.btn_save_mask.setToolTip(
            "Slaat het gecorrigeerde masker op als TIFF-bestand.\n"
            "Dit masker kan later opnieuw worden geladen als GT-masker\n"
            "of worden gebruikt voor verdere analyse."
        )
        self.btn_save_mask.clicked.connect(self._save_corrected_mask)
        self.btn_save_mask.setEnabled(False)
        av.addWidget(self.btn_save_mask)

        av.addStretch()
        root.addWidget(grp_actions)

        # ── Status-label ─────────────────────────────────────────────────────
        self.lbl_status = QLabel(
            "Stap 1: Laad een ground-truth masker.  "
            "Stap 2: Klik 'Vernieuw' om DL-detecties te klassificeren.  "
            "Stap 3: Klik op rode regio's (goedkeuren) of blauwe GT-regio's (verwijderen).  "
            "Stap 4: Stuur naar Validatie."
        )
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(
            "color:#8b949e; font-size:11px; padding:4px 8px;"
            "background:#161b22; border-radius:4px; border:1px solid #21262d;"
        )
        root.addWidget(self.lbl_status)

        # ── Hoofd-layout: links = legenda/teller, rechts = canvas ───────────
        mid = QHBoxLayout()
        mid.setSpacing(8)

        # links: legenda + teller
        left = QWidget()
        left.setFixedWidth(220)
        lv = QVBoxLayout(left)
        lv.setSpacing(8)

        grp_legend = QGroupBox("🎨  Legenda")
        lv2 = QVBoxLayout(grp_legend)
        for color, txt in [
            ("#00c850", "✔  Goedgekeurd (overlapt GT)"),
            ("#e03030", "✖  Niet goedgekeurd (klik om te corrigeren)"),
            ("#4488ff", "◌  GT-regio (blauw; klik om te verwijderen)"),
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

        grp_count = QGroupBox("📊  Teller")
        cv = QFormLayout(grp_count)
        self.lbl_n_green = QLabel("—")
        self.lbl_n_red   = QLabel("—")
        self.lbl_n_total = QLabel("—")
        self.lbl_n_gt_removed = QLabel("—")
        self.lbl_n_green.setStyleSheet("color:#3fb950; font-weight:bold;")
        self.lbl_n_red.setStyleSheet("color:#f85149; font-weight:bold;")
        self.lbl_n_gt_removed.setStyleSheet("color:#79c0ff; font-weight:bold;")
        cv.addRow("Goedgekeurd:",   self.lbl_n_green)
        cv.addRow("Afgekeurd:",     self.lbl_n_red)
        cv.addRow("Totaal DL:",     self.lbl_n_total)
        cv.addRow("GT verwijderd:", self.lbl_n_gt_removed)
        lv.addWidget(grp_count)

        grp_tip = QGroupBox("💡  Tip")
        tv = QVBoxLayout(grp_tip)
        tip_lbl = QLabel(
            "Klik op een <b style='color:#e03030'>rode</b> regio om hem "
            "<b style='color:#00c850'>groen</b> (goedgekeurd) te maken.<br><br>"
            "Klik op een <b style='color:#00c850'>groene</b> regio om hem "
            "terug <b style='color:#e03030'>rood</b> te zetten.<br><br>"
            "Klik op een <b style='color:#4488ff'>blauwe GT-regio</b> om hem "
            "te verwijderen uit het masker."
        )
        tip_lbl.setWordWrap(True)
        tip_lbl.setStyleSheet("color:#8b949e; font-size:11px;")
        tv.addWidget(tip_lbl)
        lv.addWidget(grp_tip)

        lv.addStretch()
        mid.addWidget(left)

        # rechts: canvas
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
    # Masker laden
    # ──────────────────────────────────────────────────────────────────────────
    def _load_gt_mask(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Laad Ground-Truth Masker", "",
            "Masker-bestanden (*.tif *.tiff *.png);;Alle bestanden (*)"
        )
        if not path:
            return
        try:
            mask = _load_gt_mask_from_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Fout bij laden", f"Kan masker niet laden:\n{e}")
            return
        if mask.sum() == 0:
            QMessageBox.warning(self, "Leeg masker",
                                "Het geladen masker bevat geen foreground-pixels.")
            return

        self.gt_mask = mask
        self.gt_path = path

        # Label de GT-regio's voor klik-detectie en verwijdering
        gt_labeled = label(mask.astype(bool))
        self._gt_label_image = gt_labeled
        self._gt_region_status = {}
        self._gt_regions = []
        for region in regionprops(gt_labeled):
            self._gt_region_status[region.label] = True   # standaard actief (blauw)
            self._gt_regions.append({
                "label": region.label,
                "cy": region.centroid[0],
                "cx": region.centroid[1],
                "area": region.area,
            })

        n_pos = int(mask.sum())
        pct   = 100.0 * n_pos / mask.size
        self.lbl_status.setText(
            f"✅  GT geladen: {Path(path).name}  |  Vorm: {mask.shape}  |  "
            f"Foreground: {n_pos:,} px ({pct:.1f}%)  |  "
            "Klik 'Vernieuw' om DL-detecties te laden."
        )
        self.lbl_status.setStyleSheet(
            "color:#3fb950; font-size:11px; padding:4px 8px;"
            "background:#0d2b0d; border-radius:4px; border:1px solid #238636;"
        )
        self.btn_refresh.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self._auto_classify_and_draw()

    # ──────────────────────────────────────────────────────────────────────────
    # Automatisch classificeren: groen als overlap >= drempel, anders rood
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
                "⚠  Geen DL-segmentatie beschikbaar. "
                "Voer eerst Deep Learning segmentatie uit (Tab 4)."
            )
            self.lbl_status.setStyleSheet(
                "color:#d29922; font-size:11px; padding:4px 8px;"
                "background:#2b1d0e; border-radius:4px; border:1px solid #9e6a03;"
            )
            self.btn_send.setEnabled(False)
            return

        # Zorg dat masker en label-afmeting overeenkomen
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
            # overlap = wat deel van de DL-regio in de GT valt
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
            f"📊  {len(self._region_status)} DL-regio's geladen  |  "
            f"Groen (goedgekeurd): {n_green}  |  Rood (klik om te corrigeren): {n_red}  |  "
            "Klik op een rode regio om hem goed te keuren."
        )
        self.lbl_status.setStyleSheet(
            "color:#79c0ff; font-size:11px; padding:4px 8px;"
            "background:#0d1b2b; border-radius:4px; border:1px solid #1f6feb;"
        )

        # Verbind klik-event
        if self._cid is not None:
            self.canvas_corr.mpl_disconnect(self._cid)
        self._cid = self.canvas_corr.mpl_connect("button_press_event", self._on_click)

    # ──────────────────────────────────────────────────────────────────────────
    # Canvas tekenen
    # ──────────────────────────────────────────────────────────────────────────
    def _draw(self, dl_label: Optional[np.ndarray]):
        ax = self.canvas_corr.axes
        ax.cla()
        ax.set_facecolor("#0d1117")
        ax.axis("off")

        if self.gt_mask is None:
            ax.text(0.5, 0.5, "Geen masker geladen",
                    ha="center", va="center", color="#484f58",
                    fontsize=13, transform=ax.transAxes)
            self.canvas_corr.draw_idle()
            return

        gt = self.gt_mask
        H, W = gt.shape

        # ── Achtergrond: echte microscopie-afbeelding ────────────────────────
        micro_img = None
        if self.viewer_tab is not None:
            micro_img = self.viewer_tab.get_current_image()

        if micro_img is not None:
            # Normaliseer naar [0,1] voor weergave
            img_show = micro_img.astype(np.float32)
            lo, hi = np.percentile(img_show, [1, 99])
            if hi > lo:
                img_show = np.clip((img_show - lo) / (hi - lo), 0, 1)
            else:
                img_show = np.zeros_like(img_show)
            ax.imshow(img_show, cmap="gray", aspect="equal",
                      interpolation="nearest", zorder=1)
        else:
            # Fallback: donkere achtergrond met GT-pixels licht aangeduid
            bg = np.zeros((H, W, 3), dtype=np.uint8)
            bg[gt.astype(bool)] = [50, 50, 70]
            ax.imshow(bg, aspect="equal", interpolation="nearest", zorder=1)

        # ── GT-contour (blauw gestippeld) — alleen actieve regio's ─────────────
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

        # ── DL-regio's als semi-transparante overlay ─────────────────────────
        if dl_label is not None and len(self._region_status) > 0:
            green_mask = np.zeros((H, W), dtype=bool)
            red_mask   = np.zeros((H, W), dtype=bool)

            for lbl_id, approved in self._region_status.items():
                region_px = (dl_label == lbl_id)
                if approved:
                    green_mask |= region_px
                else:
                    red_mask |= region_px

            # Gevulde vlakken (RGBA)
            rgba = np.zeros((H, W, 4), dtype=np.float32)
            rgba[green_mask] = [0.0,  0.78, 0.31, 0.35]   # groen
            rgba[red_mask]   = [0.87, 0.19, 0.19, 0.40]   # rood
            ax.imshow(rgba, aspect="equal", interpolation="nearest", zorder=3)

            # Contouren per regio
            for lbl_id, approved in self._region_status.items():
                color = "#00c850" if approved else "#e03030"
                contours = sk_measure.find_contours(
                    (dl_label == lbl_id).astype(float), 0.5)
                for c in contours:
                    ax.plot(c[:, 1], c[:, 0], color=color,
                            linewidth=1.4, zorder=5)

            # Nummers in centroid
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
            f"Corrigeer-weergave  |  "
            f"GT-contour (blauw)  +  DL-overlay  |  "
            f"Groen: {n_green}  ·  Rood: {n_red}  ·  GT verwijderd: {n_gt_removed}",
            color="#79c0ff", fontsize=10
        )
        self.canvas_corr.fig.tight_layout(pad=0.5)
        self.canvas_corr.draw_idle()

    # ──────────────────────────────────────────────────────────────────────────
    # Klik-handler: toggle status van aangeklikte regio
    # ──────────────────────────────────────────────────────────────────────────
    def _on_click(self, event):
        if event.inaxes != self.canvas_corr.axes:
            return
        if event.xdata is None or event.ydata is None:
            return

        cx_click = event.xdata
        cy_click = event.ydata

        # ── Zoek dichtstbijzijnde DL-regio-centroid ──────────────────────────
        best_dl_lbl  = None
        best_dl_dist = float("inf")
        for reg in self._dl_regions:
            dist = np.sqrt((reg["cx"] - cx_click) ** 2 + (reg["cy"] - cy_click) ** 2)
            radius = np.sqrt(reg["area"] / np.pi)
            if dist < max(radius * 1.5, 10) and dist < best_dl_dist:
                best_dl_dist = dist
                best_dl_lbl  = reg["label"]

        # ── Zoek dichtstbijzijnde GT-regio-centroid ──────────────────────────
        best_gt_lbl  = None
        best_gt_dist = float("inf")
        for reg in self._gt_regions:
            dist = np.sqrt((reg["cx"] - cx_click) ** 2 + (reg["cy"] - cy_click) ** 2)
            radius = np.sqrt(reg["area"] / np.pi)
            if dist < max(radius * 1.5, 10) and dist < best_gt_dist:
                best_gt_dist = dist
                best_gt_lbl  = reg["label"]

        # ── Bepaal welke regio aangeklikt werd (DL of GT, kleinste afstand) ──
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
            # Beide in de buurt: kies de dichtstbijzijnde
            if best_gt_dist < best_dl_dist:
                clicked_dl = False
            else:
                clicked_gt = False

        if clicked_dl:
            # Toggle DL-regio: rood ↔ groen
            self._region_status[best_dl_lbl] = not self._region_status[best_dl_lbl]
            self._draw(dl_label)
            self._update_counters()
            n_green = sum(1 for v in self._region_status.values() if v)
            n_red   = len(self._region_status) - n_green
            status  = "goedgekeurd ✔" if self._region_status[best_dl_lbl] else "afgekeurd ✖"
            self.lbl_status.setText(
                f"🖱  DL-regio {best_dl_lbl} → {status}  |  "
                f"Groen: {n_green}  ·  Rood: {n_red}"
            )

        elif clicked_gt:
            # Toggle GT-regio: actief (blauw) ↔ verwijderd
            was_active = self._gt_region_status.get(best_gt_lbl, True)
            self._gt_region_status[best_gt_lbl] = not was_active
            self._draw(dl_label)
            self._update_counters()
            n_gt_removed = sum(1 for v in self._gt_region_status.values() if not v)
            gt_status = "verwijderd ✖" if was_active else "hersteld ✔"
            self.lbl_status.setText(
                f"🖱  GT-regio {best_gt_lbl} → {gt_status}  |  "
                f"GT verwijderd: {n_gt_removed} / {len(self._gt_region_status)}"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Tellers bijwerken
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
    # Reset alle correcties terug naar auto-classificatie
    # ──────────────────────────────────────────────────────────────────────────
    def _reset_corrections(self):
        # Herstel ook verwijderde GT-regio's
        for lbl_id in self._gt_region_status:
            self._gt_region_status[lbl_id] = True
        self._auto_classify_and_draw()

    # ──────────────────────────────────────────────────────────────────────────
    # Gecorrigeerd masker bouwen en sturen naar ValidationTab
    # ──────────────────────────────────────────────────────────────────────────
    def _send_to_validation(self):
        if self.gt_mask is None:
            QMessageBox.warning(self, "Geen masker", "Laad eerst een GT-masker.")
            return

        result = getattr(self.dl_tab, "current_result", None)
        dl_label = result.label_image if result is not None else None

        corrected = self.gt_mask.copy().astype(np.uint8)

        # Verwijder GT-regio's die de gebruiker heeft uitgezet
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

            # Voeg goedgekeurde DL-regio's toe aan het gecorrigeerde masker
            for lbl_id, approved in self._region_status.items():
                if approved:
                    corrected[dl_label == lbl_id] = 1

        self.corrected_mask_ready.emit(corrected)
        n_gt_removed = sum(1 for v in self._gt_region_status.values() if not v)
        n_added = int(corrected.sum()) - int(self.gt_mask.sum())
        QMessageBox.information(
            self, "Verstuurd",
            f"Gecorrigeerd masker verstuurd naar Validatie.\n"
            f"Originele GT: {int(self.gt_mask.sum()):,} px\n"
            f"GT-regio's verwijderd: {n_gt_removed}\n"
            f"Gecorrigeerd: {int(corrected.sum()):,} px\n"
            f"Netto verschil: {n_added:+,} px"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Gecorrigeerd masker opslaan als TIFF
    # ──────────────────────────────────────────────────────────────────────────
    def _save_corrected_mask(self):
        if self.gt_mask is None:
            QMessageBox.warning(self, "Geen masker", "Laad eerst een GT-masker.")
            return

        result = getattr(self.dl_tab, "current_result", None)
        dl_label = result.label_image if result is not None else None

        # Bouw gecorrigeerd masker op (zelfde logica als _send_to_validation)
        corrected = self.gt_mask.copy().astype(np.uint8)

        # Verwijder GT-regio's die de gebruiker heeft uitgezet
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
            self, "Sla gecorrigeerd masker op",
            "gecorrigeerd_masker.tif",
            "TIFF (*.tif *.tiff);;PNG (*.png)"
        )
        if not path:
            return

        # Sla op als binair masker (0/255) zodat het universeel leesbaar is
        tifffile.imwrite(path, (corrected * 255).astype(np.uint8))

        n_pos = int(corrected.sum())
        QMessageBox.information(
            self, "Opgeslagen",
            f"Gecorrigeerd masker opgeslagen:\n{path}\n\n"
            f"Foreground-pixels: {n_pos:,}\n"
            f"Afmetingen: {corrected.shape[1]} × {corrected.shape[0]} px"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Slot: wanneer DL-tab een nieuw resultaat heeft
    # ──────────────────────────────────────────────────────────────────────────
    def _on_dl_updated(self, result):
        if self.gt_mask is not None:
            self._auto_classify_and_draw()
        self.lbl_status.setText(
            f"ℹ️  Nieuw DL-resultaat: {result.method_name} "
            f"({result.n_objects} objecten). "
            "Weergave automatisch bijgewerkt."
        )


class DoubleCanvas(FigureCanvas):
    """Matplotlib canvas met 2 subplots: [Ground Truth | Tab 4]."""

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
        subtitles  = ["Ground Truth", "TP=groen  FP=rood  FN=blauw"]

        for ax, overlay, title, sub in zip(axes, overlays, self.TITLES, subtitles):
            self._style_ax(ax, title)
            if overlay is not None:
                ax.imshow(overlay, aspect="equal", interpolation="nearest")
                ax.set_xlabel(sub, color="#8b949e", fontsize=8)
            else:
                ax.text(0.5, 0.5, "Nog niet\ngesegmenteerd",
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

        grp_actions = QGroupBox("⚙️  Acties")
        av = QHBoxLayout(grp_actions)
        av.setSpacing(8)

        self.btn_load_gt = QPushButton("📂  Laad Ground-Truth Masker")
        self.btn_load_gt.setObjectName("primary")
        self.btn_load_gt.setToolTip(
            "Laad een handmatig geannoteerd ground-truth masker (TIFF-bestand).\n\n"
            "Dit masker wordt gebruikt als referentie voor de validatie.\n"
            "Wit (255) = aggregaat aanwezig, Zwart (0) = geen aggregaat.\n\n"
            "Tip: het gecorrigeerde masker uit Tab 5 wordt automatisch ingeladen\n"
            "als je dat masker doorstuurt via de Corrigeer-tab."
        )
        self.btn_load_gt.clicked.connect(self._load_gt_mask)
        av.addWidget(self.btn_load_gt)

        self.btn_compare = QPushButton("📊  Vergelijk Resultaat")
        self.btn_compare.setEnabled(False)
        self.btn_compare.setToolTip(
            "Berekent validatiemetrieken door het DL-segmentatieresultaat\n"
            "te vergelijken met het geladen ground-truth masker.\n\n"
            "Berekende metrieken:\n"
            "  • F1/Dice  — harmonisch gemiddelde van precisie en recall\n"
            "  • IoU      — overlap gedeeld door de unie (Jaccard-index)\n"
            "  • Precisie — fractie van detecties die correct is\n"
            "  • Recall   — fractie van echte aggregaten die gevonden is\n"
            "  • TP/FP/FN — terecht positief / fout-positief / fout-negatief (pixels)"
        )
        self.btn_compare.clicked.connect(self._run_comparison)
        av.addWidget(self.btn_compare)

        self.btn_export = QPushButton("💾  Exporteer Rapport (CSV)")
        self.btn_export.setEnabled(False)
        self.btn_export.setToolTip(
            "Exporteert de validatiemetrieken naar een CSV-bestand.\n"
            "Handig voor het bijhouden van resultaten over meerdere beelden of runs."
        )
        self.btn_export.clicked.connect(self._export_csv)
        av.addWidget(self.btn_export)

        av.addStretch()
        root.addWidget(grp_actions)

        self.lbl_status = QLabel("Stap 1: Laad een ground-truth masker om te starten.")
        self.lbl_status.setStyleSheet(
            "color: #8b949e; font-size: 11px; padding: 4px 8px;"
            "background: #161b22; border-radius: 4px; border: 1px solid #21262d;"
        )
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        grp_metrics = QGroupBox("📈  Validatiemetrieken")
        mv = QVBoxLayout(grp_metrics)

        self.tbl_metrics = QTableWidget(6, 2)
        self.tbl_metrics.setHorizontalHeaderLabels(["Metriek", "Deep Learning (Ensemble)"])
        self.tbl_metrics.verticalHeader().setVisible(False)
        self.tbl_metrics.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_metrics.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl_metrics.setAlternatingRowColors(True)
        self.tbl_metrics.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_metrics.setMaximumHeight(230)
        self._populate_empty_table()
        mv.addWidget(self.tbl_metrics)
        root.addWidget(grp_metrics)

        grp_visual = QGroupBox("🖼️  Visuele Vergelijking  —  Ground Truth | Deep Learning")
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
            ("Precisie",         "—"),
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
            self, "Laad Ground-Truth Masker", "",
            "Masker-bestanden (*.tif *.tiff *.png);;Alle bestanden (*)"
        )
        if not path:
            return
        try:
            mask = _load_gt_mask_from_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Fout bij laden", f"Kan masker niet laden:\n{e}")
            return

        if mask.sum() == 0:
            QMessageBox.warning(
                self, "Leeg masker",
                "Het geladen masker bevat geen foreground-pixels."
            )
            return

        self.gt_mask = mask
        self.gt_path = path
        n_pos = int(mask.sum())
        pct   = 100.0 * n_pos / mask.size
        self.lbl_status.setText(
            f"✅  GT geladen: {Path(path).name}  |  Vorm: {mask.shape}  |  "
            f"Foreground: {n_pos:,} px ({pct:.1f}%)  |  "
            "Klik 'Vernieuw' om DL-detecties te laden."
        )
        self.lbl_status.setStyleSheet(
            "color:#3fb950; font-size:11px; padding:4px 8px;"
            "background:#0d2b0d; border-radius:4px; border:1px solid #238636;"
        )
        self.btn_refresh.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self._auto_classify_and_draw()

        self.lbl_status.setText(
            f"✅  Ground-truth geladen: {Path(path).name}  |  "
            f"Vorm: {mask.shape}  |  Foreground: {n_pos:,} px ({pct:.1f}%)"
        )
        self.lbl_status.setStyleSheet(
            "color: #3fb950; font-size: 11px; padding: 4px 8px;"
            "background: #0d2b0d; border-radius: 4px; border: 1px solid #238636;"
        )
        self.btn_compare.setEnabled(True)

    def _run_comparison(self):
        if self.gt_mask is None:
            QMessageBox.warning(self, "Geen ground-truth", "Laad eerst een ground-truth masker.")
            return

        dl_mask  = _result_to_binary_mask(getattr(self.dl_tab, "current_result", None))

        if dl_mask is None:
            QMessageBox.warning(
                self, "Geen segmentatie",
                "Voer eerst segmentatie uit in Tab 4 (Deep Learning)."
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
            ("Precisie",         _v(m_dl, "precision")),
            ("Recall",           _v(m_dl, "recall")),
            ("True Positives",   _v(m_dl, "tp")),
            ("False Positives",  _v(m_dl, "fp")),
        ]
        self._fill_table(rows, highlight=True)
        self.canvas.update_plots(gt, dl_mask)

        f1_dl  = m_dl.get("f1")
        if f1_dl is not None:
            verdict = f"Vergelijking succesvol — F1 = {f1_dl:.4f}."
        else:
            verdict = "Geen geldige metrieken beschikbaar."

        self.lbl_status.setText(f"📊  Vergelijking voltooid  |  {verdict}")
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
            self, "Exporteer validatierapport", "validatie_rapport.csv",
            "CSV-bestanden (*.csv)"
        )
        if not path:
            return
        keys   = ["f1", "iou", "precision", "recall", "tp", "fp", "fn"]
        labels = ["F1-score (Dice)", "IoU (Jaccard)", "Precisie", "Recall",
                  "True Positives", "False Positives", "False Negatives"]
        m_dl  = self._last_metrics["dl"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Metriek", "DeepLearning_Ensemble"])
            w.writerow(["Ground-truth masker", self.gt_path])
            w.writerow([])
            for key, label_txt in zip(keys, labels):
                dv = m_dl.get(key)
                w.writerow([
                    label_txt,
                    f"{dv:.6f}" if isinstance(dv, float) else (str(dv) if dv is not None else "N/A"),
                ])
        QMessageBox.information(self, "Geëxporteerd", f"Rapport opgeslagen:\n{path}")

    def _on_dl_updated(self, result):
        self.lbl_status.setText(
            f"ℹ️  Tab 4 bijgewerkt: {result.method_name} ({result.n_objects} objecten). "
            "Klik 'Vergelijk Resultaat' om te valideren."
        )

    def receive_corrected_mask(self, corrected_mask: np.ndarray):
        """Ontvangt het gecorrigeerde masker van de Corrigeer-tab."""
        self._corrected_mask = corrected_mask
        n_pos = int(corrected_mask.sum())
        pct   = 100.0 * n_pos / corrected_mask.size
        self.lbl_status.setText(
            f"✅  Gecorrigeerd GT-masker ontvangen van Corrigeer-tab  |  "
            f"Vorm: {corrected_mask.shape}  |  Foreground: {n_pos:,} px ({pct:.1f}%)  |  "
            "Klik 'Vergelijk Resultaat' om te valideren."
        )
        self.lbl_status.setStyleSheet(
            "color:#3fb950; font-size:11px; padding:4px 8px;"
            "background:#0d2b0d; border-radius:4px; border:1px solid #238636;"
        )
        # Gebruik het gecorrigeerde masker als gt_mask voor vergelijking
        self.gt_mask = corrected_mask
        self.gt_path = "(gecorrigeerd via Corrigeer-tab)"
        self.btn_compare.setEnabled(True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HOOFDVENSTER
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
        self.status.showMessage("Klaar — Open een .LIF of .TIF bestand om te starten.")
        self.setStatusBar(self.status)

        tb = QToolBar("Hoofd")
        tb.setIconSize(QSize(20, 20))
        tb.setMovable(False)
        tb.setStyleSheet("background:#161b22; spacing: 4px; padding: 4px 8px; border-bottom: 1px solid #21262d;")
        self.addToolBar(tb)

        for label, slot, tip in [
            ("📂  Open",        self._open_file,      "Open .lif of .tif bestand"),
            ("💾  Sla op",       self._save_result,    "Sla huidige segmentatie op"),
            ("📊  Statistieken", self._show_stats_dialog, "Toon gedetailleerde statistieken"),
            ("❓  Help",         self._show_help,      "Documentatie & tips"),
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
        self.tabs.addTab(self.cellpose_tab, "🔬  3. Cellichamen")
        self.tabs.addTab(self.dl_tab,       "🧠  4. Deep Learning")
        self.tabs.addTab(self.corr_tab,     "✏️  5. Corrigeer")
        self.tabs.addTab(self.val_tab,      "📊  6. Validatie")

    def _wire_signals(self):
        self.viewer_tab.stack_loaded.connect(
            lambda s: self.status.showMessage(
                f"Geladen: {s.name}  |  Z={s.z_count}, C={s.channel_count}, "
                f"{s.height}×{s.width} px"
            )
        )
        self.dl_tab.result_ready.connect(
            lambda r: self.status.showMessage(
                f"DL Segmentatie klaar: {r.method_name}  |  "
                f"N={r.n_objects} objecten  |  {r.time_seconds:.2f}s"
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
            QMessageBox.warning(self, "Geen resultaat", "Voer eerst segmentatie uit in Tab 4.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Sla labelbeeld op", "", "TIFF (*.tif);;PNG (*.png)"
        )
        if path:
            tifffile.imwrite(path, result.label_image.astype(np.uint16))
            QMessageBox.information(self, "Opgeslagen", f"Labelbeeld opgeslagen:\n{path}")

    def _show_stats_dialog(self):
        result = self.dl_tab.current_result
        if result is None:
            QMessageBox.information(self, "Geen resultaat", "Voer eerst segmentatie uit in Tab 4.")
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
      ✦ Handleiding
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
      ⚡ Aanbevolen werkstroom
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
              Laad hier een <b>.LIF</b> of <b>.TIF</b> bestand in en maak een
              selectie van de beelden die je wilt gebruiken voor de pre-processing.
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
              Klik hier op de
              <span style='background:#21262d; color:#f0f6fc;
                           border-radius:4px; padding:1px 6px;
                           font-size:11px;'>★ Aanbevolen pipeline</span>
              knop om de optimale beeldbewerking toe te passen.
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
              🔬 Cellichamen
            </span><br/>
            <span style='color:#c9d1d9; font-size:12px;'>
              Om de achtergrond te verwijderen kun je hier de cellichamen
              automatisch laten selecteren via <b>Cellpose</b>.
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
              Voer hier de aggregaat-segmentatie uit door de map met
              <b>Deep Learning modellen</b> in te laden
              (<code style='background:#0d1117; padding:1px 5px;
                            border-radius:3px; font-size:11px;'>model_fold*.pth</code>)
              en de ensemble predictie te starten.
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
              ✏️ Corrigeer
            </span><br/>
            <span style='color:#c9d1d9; font-size:12px;'>
              Laad hier je handmatige <b>ground-truth masker</b>. De DL-detecties
              worden automatisch <b style='color:#00c850'>groen</b> (overlapt GT)
              of <b style='color:#e03030'>rood</b> (overlapt GT niet) gekleurd.
              Klik op een rode regio om hem alsnog goed te keuren.
              Stuur het gecorrigeerde masker daarna door naar Validatie.
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
              📊 Validatie
            </span><br/>
            <span style='color:#c9d1d9; font-size:12px;'>
              Vergelijk het DL-resultaat met het (gecorrigeerde) ground-truth masker
              (F1/Dice, IoU, precisie &amp; recall). Het gecorrigeerde masker
              vanuit Tab 5 wordt hier automatisch ingeladen.
            </span>
          </div>
        </td>
      </tr>
    </table>

    <!-- ── TIPS ── -->
    <h2 style='color:#79c0ff; font-size:15px; font-weight:700;
               margin:0 0 12px 0; border-bottom:1px solid #21262d;
               padding-bottom:8px;'>
      💡 Handige tips
    </h2>

    <table width='100%' cellspacing='6' cellpadding='0'
           style='margin-bottom:20px;'>
      <tr>
        <td width='49%' valign='top'>
          <div style='background:#161b22; border:1px solid #21262d;
                      border-radius:8px; padding:12px 14px; height:100%;'>
            <p style='margin:0 0 5px 0; color:#f0c040; font-weight:700;
                      font-size:12px;'>⚠ Te veel detecties?</p>
            <p style='margin:0; color:#8b949e; font-size:11px; line-height:1.5;'>
              Verhoog de threshold of vergroot het minimale oppervlak
              (px²) in de post-processing instellingen.
            </p>
          </div>
        </td>
        <td width='2%'></td>
        <td width='49%' valign='top'>
          <div style='background:#161b22; border:1px solid #21262d;
                      border-radius:8px; padding:12px 14px; height:100%;'>
            <p style='margin:0 0 5px 0; color:#f0c040; font-weight:700;
                      font-size:12px;'>⚠ Trage segmentatie?</p>
            <p style='margin:0; color:#8b949e; font-size:11px; line-height:1.5;'>
              Schakel TTA uit voor snellere resultaten, of stel Device
              in op <i>cuda</i> als je een NVIDIA GPU hebt.
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
                      font-size:12px;'>⚠ CUDA-fout?</p>
            <p style='margin:0; color:#8b949e; font-size:11px; line-height:1.5;'>
              Stel Device in op <i>cpu</i> in Tab 4 om de GPU te omzeilen.
            </p>
          </div>
        </td>
        <td width='2%'></td>
        <td width='49%' valign='top'>
          <div style='background:#161b22; border:1px solid #21262d;
                      border-radius:8px; padding:12px 14px; height:100%;'>
            <p style='margin:0 0 5px 0; color:#f0c040; font-weight:700;
                      font-size:12px;'>⚠ Geen modellen gevonden?</p>
            <p style='margin:0; color:#8b949e; font-size:11px; line-height:1.5;'>
              Controleer of de bestandsnamen de vorm
              <code style='background:#0d1117; padding:1px 4px;
                            border-radius:3px;'>model_fold0.pth</code>
              hebben en de juiste map is geselecteerd.
            </p>
          </div>
        </td>
      </tr>
    </table>

    <!-- ── PAKKETTEN ── -->
    <h2 style='color:#79c0ff; font-size:15px; font-weight:700;
               margin:0 0 10px 0; border-bottom:1px solid #21262d;
               padding-bottom:8px;'>
      ⚙️ Benodigde pakketten
    </h2>
    <div style='background:#161b22; border:1px solid #21262d;
                border-radius:8px; padding:12px 16px; margin-bottom:8px;'>
      <p style='margin:0 0 4px 0; color:#58a6ff; font-size:11px;
                font-weight:700; text-transform:uppercase;
                letter-spacing:0.5px;'>Verplicht</p>
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
                letter-spacing:0.5px;'>.LIF bestanden (Leica)</p>
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
            "Welkom! — Open een .LIF of .TIF bestand via 📂 Open."
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