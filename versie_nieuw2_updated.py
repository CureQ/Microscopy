#!/usr/bin/env python3
"""
Protein Aggregate Analyzer — Verbeterde versie v3.0
=====================================================
Nieuw in v3.0 (fixes voor valse positieven):
  ★ CEL MASKER             — detecties buiten de cellen worden volledig verwijderd
  ★ Betere drempelwaarde   — percentiel berekend op alleen significante pixels (>p50)
  ★ Sterkere BG-suppressie — cell_sigma × 1.5 ipv × 0.5 voor echte cellichaamsuppressie
  ★ KNN ruis-guard         — z-score alleen berekend op pixels met voldoende intensiteit
  ★ Strengere fused drempel — minimum drempel verhoogd

Nieuw in v2.0:
  ★ Hot-colormap decoder  — laadt "hot" RGB TIF-bestanden correct als grijswaarden
  ★ Rolling-Ball v2        — verbeterde achtergrondsubtractie via morfologische opening
  ★ Bilateral denoise      — behoudt randen, dempt ruis
  ★ PUNCTA DETECTOR PRO    — aanbevolen methode (Tab 3, item 0)
  ★ WATERSHED SPLITTER     — verdeelt overlappende aggregaatclusters

Vereisten: PyQt5, numpy, scipy, scikit-image, matplotlib, tifffile
Optioneel:  readlif (voor .lif bestanden)

Auteur: Jazz Fust
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
from skimage.feature import blob_log, blob_dog, blob_doh
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

/* ── Splitter ─────────────────────────────────────────────────────────────── */
QSplitter::handle {
    background-color: #21262d;
    width: 2px;
    height: 2px;
}
QSplitter::handle:hover {
    background-color: #58a6ff;
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

/* ── Dialog button box ────────────────────────────────────────────────────── */
QDialogButtonBox QPushButton {
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
    """
    Converteer een RGB-beeld dat opgeslagen is met Matplotlib 'hot' colormap
    terug naar een enkelvoudig intensiteitskanaal [0, 1].

    Hot-colormap mapping:
      intensity [0.00, 0.33] → RGB = (3i,  0,   0)    => i = R/3
      intensity [0.33, 0.67] → RGB = (1,   3(i-1/3),  0)   => i = 1/3 + G/3
      intensity [0.67, 1.00] → RGB = (1,   1,   3(i-2/3))  => i = 2/3 + B/3
    """
    if rgb.ndim == 2:
        return rgb.astype(np.float32) / (rgb.max() + 1e-8)
    if rgb.ndim == 3 and rgb.shape[2] == 1:
        return rgb[:, :, 0].astype(np.float32) / (rgb.max() + 1e-8)
    if rgb.ndim == 3 and rgb.shape[2] >= 3:
        r = rgb[:, :, 0].astype(np.float32) / 255.0
        g = rgb[:, :, 1].astype(np.float32) / 255.0
        b = rgb[:, :, 2].astype(np.float32) / 255.0
        # Herstel intensiteit via omgekeerde hot-mapping
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
    # Fallback: gemiddelde over kleurkanalen
    return rgb.astype(np.float32).mean(axis=2) / 255.0


def is_hot_encoded(arr: np.ndarray) -> bool:
    """Detecteer of een RGB-beeld waarschijnlijk hot-colormap gecodeerd is."""
    if arr.ndim != 3 or arr.shape[2] < 3:
        return False
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    # Hot: blauw is bijna altijd nul of laag tenzij het erg helder is
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

        # ── Hot colormap detectie & decodering ──────────────────────────────
        decoded = False
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            if is_hot_encoded(arr):
                gray = decode_hot_colormap(arr)
                arr = gray[np.newaxis, np.newaxis]   # (1, 1, H, W)
                decoded = True
        if arr.ndim == 3 and arr.shape[2] in (3, 4) and not decoded:
            # Gewone RGB: neem maximale projectie over kleurkanalen
            gray = arr.max(axis=2).astype(np.float32)
            arr = gray[np.newaxis, np.newaxis]
        # ──────────────────────────────────────────────────────────────────

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
        """
        Verbeterde Rolling Ball via morfologische opening.
        Schat het cellichaamachtergrondniveau en subtracheert dit.
        Werkt beter dan eenvoudige Gaussiaan voor onregelmatige achtergrond.
        """
        img_f = img.astype(np.float32)
        # Begrens radius voor snelheid
        r = max(3, min(radius, min(img_f.shape) // 4, 40))
        from skimage.morphology import opening as morph_opening
        bg = morph_opening(img_f, disk(r))
        return np.clip(img_f - bg, 0, None).astype(np.float32)

    @staticmethod
    def subtract_background_gaussian(img: np.ndarray, sigma: float = 30.0) -> np.ndarray:
        """
        Snelle achtergrondsubtractie via Gaussiaans-gemiddelde.
        Goed voor geleidelijke verlichtingsonregelmatigheden.
        """
        img_f = img.astype(np.float32)
        bg = gaussian(img_f, sigma=sigma)
        return np.clip(img_f - bg * 0.95, 0, None).astype(np.float32)

    @staticmethod
    def tophat(img: np.ndarray, radius: int = 5) -> np.ndarray:
        """White Top-Hat: behoudt objecten kleiner dan de structurele element."""
        return white_tophat(img.astype(np.float32), disk(radius)).astype(np.float32)

    @staticmethod
    def multiscale_tophat(img: np.ndarray,
                          radii: Tuple[int, ...] = (3, 6, 10)) -> np.ndarray:
        """
        Nieuw: Multi-schaal Top-Hat.
        Combineert top-hat op meerdere schalen via maximum fusie.
        Detecteert aggregaten van verschillende groottes tegelijk.
        """
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
        """
        Nieuw: Bilateral filtering.
        Dempt ruis maar behoudt scherpe grenzen van aggregaten.
        Ideaal als pre-stap voor segmentatie.
        """
        img_f = np.clip(img.astype(np.float32), 0, 1)
        try:
            return denoise_bilateral(
                img_f, sigma_color=sigma_color, sigma_spatial=sigma_spatial,
                mode='reflect'
            ).astype(np.float32)
        except Exception:
            return gaussian(img_f, sigma=1.0).astype(np.float32)




# ═══════════════════════════════════════════════════════════════════════════════
#  SEGMENTATIE HULPFUNCTIES
# ═══════════════════════════════════════════════════════════════════════════════
def _knn_score_map(img_norm: np.ndarray, k: int = 10,
                   min_intensity_percentile: float = 5.0) -> np.ndarray:
    import math
    radius_fine   = max(5.0, math.sqrt(k / math.pi) * 2.5)
    radius_coarse = radius_fine * 2.5
    radius_xlarge = radius_fine * 6.0   # extra schaal voor grote aggregaten (30–80px)

    # v3.0 FIX: bereken een minimum intensiteitsdrempel.
    # In lege/zwarte gebieden is de lokale std ≈ 0, wat leidt tot enorme
    # z-scores door zelfs minieme ruis. Maskeer deze gebieden.
    min_thresh = np.percentile(img_norm, min_intensity_percentile)
    meaningful_mask = img_norm > max(min_thresh, 0.01)

    def _local_zscore(img, radius):
        local_mean = gaussian(img, sigma=radius)
        diff = img - local_mean
        local_var = gaussian(diff ** 2, sigma=radius)
        local_std = np.sqrt(np.clip(local_var, 0, None)) + 1e-6
        zscore = np.clip(diff / local_std, 0, None)
        # v3.0 FIX: zet z-score nul buiten het betekenisvolle gebied
        return zscore * meaningful_mask.astype(np.float32)

    z_fine   = _local_zscore(img_norm, radius_fine)
    z_coarse = _local_zscore(img_norm, radius_coarse)
    z_xlarge = _local_zscore(img_norm, radius_xlarge)
    return np.maximum(np.maximum(z_fine, z_coarse), z_xlarge).astype(np.float32)


def _blob_score_map(img_norm, min_sigma, max_sigma, threshold):
    blobs = blob_log(img_norm, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=10, threshold=threshold, overlap=0.4)
    score = np.zeros(img_norm.shape, dtype=np.float32)
    H, W = img_norm.shape
    for y, x, r in blobs:
        sigma_blob = max(1.0, r)
        half = int(np.ceil(sigma_blob * 3))
        y0, y1 = max(0, int(y) - half), min(H, int(y) + half + 1)
        x0, x1 = max(0, int(x) - half), min(W, int(x) + half + 1)
        ys = np.arange(y0, y1) - y
        xs = np.arange(x0, x1) - x
        YY, XX = np.meshgrid(ys, xs, indexing='ij')
        patch = np.exp(-(YY**2 + XX**2) / (2 * sigma_blob**2)).astype(np.float32)
        score[y0:y1, x0:x1] = np.maximum(score[y0:y1, x0:x1], patch)
    return score, blobs


def _multiscale_log_response(img: np.ndarray,
                              sigmas=(1.5, 2.5, 4.0, 6.0)) -> np.ndarray:
    """
    NIEUW: Multi-schaal genormaliseerde LoG-respons.
    Berekent σ²·LoG(img, σ) voor elke sigma en neemt de maximum respons.
    Dit is gevoeliger voor aggregaten op verschillende schalen dan een enkele LoG.
    """
    H, W = img.shape
    response = np.zeros((H, W), dtype=np.float32)
    for sigma in sigmas:
        # σ²-genormaliseerde LoG (schaal-invariant)
        lap = -filters.laplace(gaussian(img.astype(np.float32), sigma=sigma))
        lap_norm = lap * (sigma ** 2)
        lap_pos = np.clip(lap_norm, 0, None)
        response = np.maximum(response, lap_pos)
    return response


def _sharpness_map(img: np.ndarray, sigma_sharp: float = 1.0) -> np.ndarray:
    """
    Multi-schaal scherpheidskaart.
    Combineert fijne schaal (sigma_sharp) voor kleine puncta
    met grovere schaal (sigma_sharp*4) voor grote aggregaten.
    Grote aggregaten hebben een scherpe RAND, ook al is hun kern uniform.
    """
    img_f = img.astype(np.float32)

    # Fijne schaal: kleine puncta
    smooth_fine = gaussian(img_f, sigma=sigma_sharp * 3)
    diff_fine = img_f - smooth_fine
    var_fine = gaussian(diff_fine ** 2, sigma=sigma_sharp * 2)

    # Grove schaal: grote aggregaten — detecteer randen van grote blobs
    smooth_coarse = gaussian(img_f, sigma=sigma_sharp * 10)
    diff_coarse = img_f - smooth_coarse
    var_coarse = gaussian(diff_coarse ** 2, sigma=sigma_sharp * 6)

    # Neem het maximum: een pixel is 'scherp' als hij op ENIGE schaal scherp is
    combined = np.maximum(var_fine, var_coarse * 0.6)
    return np.sqrt(np.clip(combined, 0, None)).astype(np.float32)


def _local_background_suppression(img: np.ndarray,
                                   cell_sigma: float = 20.0,
                                   percentile_cut: float = 80.0) -> np.ndarray:
    """
    NIEUW: Cellichaam-achtergrond onderdrukking.
    Pixels die in een breed helder gebied liggen (cellichaam) worden
    naar nul gedrukt. Aggregaten zijn klein en scherp, niet breed en diffuus.
    """
    img_f = img.astype(np.float32)
    # Brede Gaussiaan schat het niveau van het cellichaam
    bg_broad = gaussian(img_f, sigma=cell_sigma)
    # Drempel: boven dit percentiel van het brede achtergrondsigmoid = cellichaam
    bg_thresh = np.percentile(bg_broad, percentile_cut)
    # Maak masker: cellichaam-pixels
    cell_mask = bg_broad > bg_thresh
    # Onderdruk cellichaampixels (fade out, niet hard knippen)
    suppression = np.where(cell_mask,
                           np.clip(1.0 - (bg_broad - bg_thresh) / (bg_broad.max() - bg_thresh + 1e-8), 0, 1),
                           1.0).astype(np.float32)
    return img_f * suppression


def _fiber_suppression_mask(img: np.ndarray,
                             fiber_sigma_range=(1.5, 4.0),
                             fiber_thresh_percentile: float = 75.0) -> np.ndarray:
    """
    Detecteert lineaire/vezelachtige structuren via Hessian-eigenwaarden (Frangi-filter).

    BELANGRIJK: grote ronde aggregaten worden BESCHERMD.
    Vezels: l2 << 0, |l1| << |l2|  → hoge vezelrespons
    Ronde blobs: l1 ≈ l2 << 0      → lage vezelrespons (hoge blob-achtigheid Rb)

    De blob_likeness ratio Rb = (l1/l2)² is hoog voor vezels (l1 ~ 0, l2 sterk)
    en laag voor ronde blobs (l1 ≈ l2). We filteren ALLEEN pixels met hoge Rb.
    """
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    img_f = img.astype(np.float32)
    fiber_response = np.zeros_like(img_f)
    blob_protect   = np.zeros_like(img_f)   # hoge waarde = waarschijnlijk rond aggregaat

    for sigma in np.linspace(fiber_sigma_range[0], fiber_sigma_range[1], 4):
        H_elems = hessian_matrix(img_f, sigma=sigma, order='rc')
        eigvals = hessian_matrix_eigvals(H_elems)
        l1, l2 = eigvals[0], eigvals[1]   # |l1| >= |l2|

        # Vezelrespons: alleen als BEIDE eigenwaarden niet gelijk zijn
        # Rb = (l2/l1)^2 — klein voor vezels, groot voor blobs
        denom = np.abs(l1) + 1e-8
        Rb = (l2 ** 2) / (denom ** 2)         # 0=vezel, 1=blob
        S2 = l1 ** 2 + l2 ** 2

        # Vezel: l2 negatief (donkere structuur op heldere bg → omgekeerd: heldere lijn)
        # Wij zoeken heldere structuren op donkere achtergrond: l1 < 0 EN l2 < 0
        # Vezel = |l1| >> |l2|  →  Rb klein
        # Blob  = |l1| ≈ |l2|  →  Rb groot (beschermen!)
        vezel = np.where(
            (l1 < 0) & (l2 < 0),
            (1.0 - Rb) * (1.0 - np.exp(-S2 / (2 * 0.005**2))),  # laag Rb = vezel
            0.0
        ).astype(np.float32)
        blob_score = np.where(
            (l1 < 0) & (l2 < 0),
            Rb,   # hoog Rb = ronde blob
            0.0
        ).astype(np.float32)

        fiber_response = np.maximum(fiber_response, vezel)
        blob_protect   = np.maximum(blob_protect,   blob_score)

    # Masker: hoge vezelrespons EN lage blob-bescherming
    thresh = np.percentile(fiber_response, fiber_thresh_percentile)
    fiber_mask = (fiber_response > thresh) & (blob_protect < 0.5)
    return fiber_mask


def _compute_cell_mask(img: np.ndarray,
                       min_cell_intensity_percentile: float = 15.0,
                       dilation_radius: int = 20) -> np.ndarray:
    """
    NIEUW v3.0: Bereken een masker dat aangeeft waar de cellen zich bevinden.

    Strategie:
    - Gebruik Otsu-drempel op het licht-geëgaliseerde beeld om celregio te vinden
    - Dilateer het masker zodat aggregaten aan de celrand niet worden gemist
    - Verwijder kleine ruis-eilandjes (min 500px²)

    Resultaat: binair masker, True = cel-regio
    """
    img_f = img.astype(np.float32)
    # Lichte normalisatie
    lo, hi = np.percentile(img_f, [1, 99.9])
    if hi > lo:
        img_f = np.clip((img_f - lo) / (hi - lo), 0, 1)

    # Zachte blur zodat ruis de drempel niet verstoort
    img_blur = gaussian(img_f, sigma=3.0)

    # Minimale intensiteitsdrempel: pixels die écht zwart/leeg zijn
    min_thresh = np.percentile(img_blur, min_cell_intensity_percentile)

    # Otsu binnen de niet-lege pixels
    signal_pixels = img_blur[img_blur > min_thresh]
    if len(signal_pixels) < 100:
        # Fallback: alles is cel
        return np.ones(img.shape, dtype=bool)

    try:
        otsu_thresh = threshold_otsu(signal_pixels)
        cell_mask = img_blur > (otsu_thresh * 0.5)  # iets lager dan Otsu voor randen
    except Exception:
        cell_mask = img_blur > min_thresh

    # Verwijder kleine ruis-eilandjes
    cell_mask = remove_small_objects(cell_mask, min_size=500)
    # Vul gaten binnen cellen
    cell_mask = ndi.binary_fill_holes(cell_mask)
    # Dilateer: zorg dat aggregaten aan de celrand niet worden gemist
    cell_mask = morphology.binary_dilation(cell_mask, disk(dilation_radius))

    return cell_mask


def _intensity_roundness_filter(label_img: np.ndarray,
                                  img: np.ndarray,
                                  min_area: int,
                                  max_area: int) -> np.ndarray:
    """
    Schaalbewuste nafilter gericht op cellichaam/vezel artefacten.

    Grote aggregaten (area > 200px²) krijgen soepelere drempels omdat:
    - Ze een hogere brede Gaussiaan-waarde hebben (zijn breed)
    - Hun contrast-ring relatief smal is t.o.v. hun oppervlak
    - Ze minder circulair kunnen zijn (clusters van meerdere aggregaten)
    """
    if label_img.max() == 0:
        return label_img
    out = label_img.copy()
    img_f = img.astype(np.float32)
    bg_broad = gaussian(img_f, sigma=25.0)
    bg_norm = bg_broad / (bg_broad.max() + 1e-8)

    for region in regionprops(label_img, intensity_image=img_f):
        area = region.area
        if not (min_area <= area <= max_area):
            out[label_img == region.label] = 0
            continue

        cy, cx = int(region.centroid[0]), int(region.centroid[1])

        # Schaalbewuste drempels: grote objecten krijgen meer speelruimte
        is_large = area > 200
        bg_cutoff    = 0.72 if is_large else 0.52   # grote aggregaten zijn breed & helder
        contrast_min = 0.06 if is_large else 0.12   # grote clusters: subtielere rand
        circ_min     = 0.15 if is_large else 0.28   # clusters zijn minder rond

        # Breed heldere achtergrond check
        local_bg_level = float(bg_norm[cy, cx])
        if local_bg_level > bg_cutoff:
            out[label_img == region.label] = 0
            continue

        # Contrast check: ring-breedte schaalt met objectgrootte
        mask_obj = (label_img == region.label)
        ring_radius = max(4, int(np.sqrt(area / np.pi) * 0.5))  # ~halve straal
        dilated = morphology.binary_dilation(mask_obj, disk(ring_radius))
        ring = dilated & ~mask_obj
        if ring.sum() > 0:
            ring_mean = float(img_f[ring].mean())
            obj_mean = float(region.mean_intensity)
            contrast = (obj_mean - ring_mean) / (ring_mean + 1e-8)
            if contrast < contrast_min:
                out[label_img == region.label] = 0
                continue

        # Compactheid: grote clusters hoeven minder rond te zijn
        perim = region.perimeter
        if perim > 0:
            circularity = (4 * np.pi * area) / (perim ** 2)
            if circularity < circ_min and area > 30:
                out[label_img == region.label] = 0
                continue

    return label(out > 0)


def _mask_to_result(mask, img, method_name, params, min_area, max_area,
                    t0, min_distance=3, close_radius=1):
    mask = remove_small_objects(mask.astype(bool), min_size=max(1, min_area))
    if close_radius > 0:
        mask = binary_closing(mask, disk(close_radius))

    temp_labels = label(mask)
    valid_mask  = np.zeros_like(mask, dtype=bool)
    for region in regionprops(temp_labels):
        if region.area <= max_area:
            valid_mask[temp_labels == region.label] = True

    lbl = label(valid_mask)
    lbl = _filter_by_morphology(lbl, min_area, max_area)
    props = _label_to_props(lbl, img)

    return SegmentationResult(
        method_name=method_name, params=params,
        label_image=lbl, n_objects=int(lbl.max()),
        properties=props, time_seconds=time.time() - t0
    )


def _watershed_split_labels(label_img: np.ndarray,
                             img: np.ndarray,
                             min_distance: int = 4) -> np.ndarray:
    """
    NIEUW: Watershed-gebaseerde splitsing van overlappende aggregaat-clusters.
    Grote labels (mogelijk meerdere samengeklonterde aggregaten) worden
    verdeeld op basis van lokale intensiteitspieken.
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed as wsh

    out = label_img.copy()
    for region in regionprops(label_img):
        if region.area < 2 * min_distance ** 2:
            continue  # te klein om te splitsen
        mask = (label_img == region.label)
        patch = img.astype(np.float32)
        # Lokale pieken in de intensiteit = aparte aggregaten
        distance = ndi.distance_transform_edt(mask)
        coords = peak_local_max(patch * mask, min_distance=min_distance,
                                labels=mask.astype(np.uint8))
        markers = np.zeros_like(mask, dtype=np.int32)
        for i, (y, x) in enumerate(coords):
            markers[y, x] = i + 1
        if markers.max() <= 1:
            continue  # slechts één piek, geen splitsing nodig
        split = wsh(-patch, markers=markers, mask=mask)
        # Schrijf gesplitste labels terug (met offset zodat labels uniek blijven)
        offset = out.max()
        for val in np.unique(split):
            if val == 0:
                continue
            out[split == val] = offset + val
    return label(out > 0)


def _filter_by_morphology(label_img, min_area, max_area):
    """
    Morfologische nafilter v2.2 — strenger dan v2.0.
    Verwijdert: objecten buiten groottebereik, draadachtige structuren,
    grillige holle vormen, en extreem platte objecten.
    """
    out = label_img.copy()
    for region in regionprops(label_img):
        area = region.area
        if not (min_area <= area <= max_area):
            out[label_img == region.label] = 0
            continue
        # Draad/vezel filter — strenger (was 0.93)
        if region.eccentricity > 0.90:
            out[label_img == region.label] = 0
            continue
        # Dikke vezel filter — strenger (was >0.87 & >120px²)
        if region.eccentricity > 0.82 and area > 80:
            out[label_img == region.label] = 0
            continue
        # Grillige rand filter
        if region.solidity < 0.65:
            out[label_img == region.label] = 0
            continue
        # Extreem plat object
        if area > 500:
            minor = region.minor_axis_length
            major = region.major_axis_length
            if major > 0 and minor / max(major, 1) < 0.25:
                out[label_img == region.label] = 0
                continue
    return label(out > 0)


def _blobs_to_props(blobs, img):
    props = []
    for i, (y, x, r) in enumerate(blobs):
        iy, ix = int(y), int(x)
        ir = max(1, int(r * 1.4))
        rr, cc = draw_disk((iy, ix), ir, shape=img.shape)
        if len(rr) == 0:
            continue
        region_vals = img[rr, cc]
        props.append({
            "id": i + 1, "y": iy, "x": ix, "radius_px": ir,
            "area_px2": len(rr),
            "mean_intensity": float(region_vals.mean()),
            "max_intensity":  float(region_vals.max()),
            "sigma": float(r),
        })
    return props


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
#  SEGMENTATIE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class SegmentationEngine:

    # ── 0. PUNCTA DETECTOR PRO  ★★ AANBEVOLEN ★★ ──────────────────────────────
    @staticmethod
    def puncta_detector_pro(img: np.ndarray,
                             # Schaalparameters
                             min_sigma: float = 1.2,
                             max_sigma: float = 20.0,
                             # Gevoeligheid
                             sensitivity: float = 0.035,
                             # Achtergrondonderdrukking
                             cell_bg_sigma: float = 22.0,
                             cell_bg_percentile: float = 75.0,
                             # Scherpheidsfilter
                             use_sharpness_gate: bool = True,
                             sharpness_percentile: float = 25.0,
                             # Watershed splitsing
                             use_watershed_split: bool = True,
                             min_split_distance: int = 4,
                             # Grootte filter
                             min_area: int = 5,
                             max_area: int = 3000) -> SegmentationResult:
        """
        ★★ PUNCTA DETECTOR PRO — de beste methode voor eiwit-aggregaten ★★

        Werking:
        ────────
        1. MULTI-SCHAAL ACHTERGROND-SUBTRACTIE
           Verwijdert brede diffuze achtergrond (cellichamen, autofluorescentie)
           via een gecombineerde rolling-ball + Gaussiaan aanpak.

        2. MULTI-SCHAAL LoG ENSEMBLE
           Berekent σ²-genormaliseerde LoG op 4 schalen simultaan.
           Detecteert aggregaten van 2–15 px diameter in één keer.
           Veel beter dan een enkele LoG-schaal!

        3. LOKALE ACHTERGROND Z-SCORE
           Elke pixel krijgt een z-score t.o.v. zijn locale omgeving.
           Pixels die statistisch significant helderder zijn dan hun buren
           worden als aggregaat-kandidaat gemarkeerd.

        4. SCORE FUSIE (weighted ensemble)
           LoG-respons × 0.65 + KNN-zscore × 0.35 → gecombineerde scorekaart.
           De twee methoden vullen elkaar aan:
           - LoG: excellent voor ronde blobs
           - KNN: pakt ook onregelmatige aggregaatvormen

        5. SCHERPHEIDS-GATE (optioneel)
           Verwerpt detecties in diffuze/vlakke gebieden.
           Echte aggregaten zijn scherpe, discrete structuren.

        6. CELLICHAAM-ONDERDRUKKING
           Pixels in brede heldere gebieden (cellichamen) worden
           onderdrukt vóór de drempelwaarde-stap.

        7. WATERSHED SPLITSING (optioneel)
           Groeperingen van dicht bij elkaar liggende aggregaten
           worden automatisch gesplitst op lokale intensiteitspieken.

        8. MORFOLOGISCHE NAFILTERING
           Eccentriciteit, soliditeit en grootte-filters verwijderen
           resterende valse positieven (draden, celranden, artefacten).
        """
        t0 = time.time()

        # ── Stap 1: Normaliseer & detecteer hot-colormap ──────────────────
        img_f = img.astype(np.float32)
        # Als het beeld al [0,1] bereik heeft (hot-gedecodeerd), niet opnieuw normaliseren
        if img_f.max() > 2.0:
            img_f = Preprocessor.normalize(img_f, pmin=1.0, pmax=99.9)

        # ── Stap 2: Multi-schaal achtergrond-subtractie ───────────────────
        # Factor 0.80 (was 0.92): minder agressief aftrekken zodat grote heldere
        # aggregaatclusters hun signaal behouden. Cellichamen worden alsnog
        # weggewerkt omdat die VEEL breder zijn dan de aggregaat-sigma.
        bg_broad = gaussian(img_f, sigma=cell_bg_sigma)
        img_nobg = np.clip(img_f - bg_broad * 0.80, 0, None)
        bg_mid = gaussian(img_f, sigma=cell_bg_sigma * 0.35)
        img_nobg2 = np.clip(img_f - bg_mid * 0.65, 0, None)
        img_clean = np.maximum(img_nobg, img_nobg2)
        img_clean = Preprocessor.normalize(img_clean, pmin=0.5, pmax=99.5)

        # ── Stap 2b: Hessian vezel/fiber onderdrukking ───────────────────
        # Vezels zijn lineaire structuren (anisotrope Hessian-eigenwaarden).
        # Aggregaten zijn bolvormig → we filteren vezelpixels weg vóór detectie.
        try:
            fiber_mask = _fiber_suppression_mask(
                img_clean, fiber_sigma_range=(1.5, 4.0),
                fiber_thresh_percentile=75.0
            )
            fiber_weight = np.where(fiber_mask, 0.04, 1.0).astype(np.float32)
            img_clean = img_clean * fiber_weight
            img_clean = Preprocessor.normalize(img_clean, pmin=0.5, pmax=99.5)
        except Exception:
            pass

        # ── Stap 3: Zachte denoise (bewaar puncta, verwijder ruis) ────────
        img_smooth = gaussian(img_clean, sigma=0.7)

        # ── Stap 4: Multi-schaal LoG ensemble ────────────────────────────
        # Grote aggregaten hebben een grote effectieve sigma.
        # Een aggregaat van 30px doorsnede heeft sigma ~ 30/(2*sqrt(2)) ~ 10px.
        # Voeg extra grote schalen toe zodat we ze niet missen.
        n_sigmas = 8
        sigmas = np.geomspace(min_sigma, max_sigma, n_sigmas)
        sigmas = np.concatenate([sigmas, [max_sigma * 1.4, max_sigma * 1.8, max_sigma * 2.2]])
        log_response = _multiscale_log_response(img_smooth, sigmas=sigmas)

        # ── Stap 5: KNN lokale z-score kaart ─────────────────────────────
        knn_raw = _knn_score_map(img_smooth, k=12)
        p99 = np.percentile(knn_raw, 99.5)
        knn_norm = np.clip(knn_raw / max(p99, 1e-6), 0, 1)

        # ── Stap 6: Normaliseer LoG-respons ───────────────────────────────
        p99_log = np.percentile(log_response, 99.5)
        log_norm = np.clip(log_response / max(p99_log, 1e-6), 0, 1)

        # ── Stap 7: Cellichaam-onderdrukking ─────────────────────────────
        # v3.0 FIX: gebruik cell_sigma * 1.5 (was * 0.5) zodat echte cellichamen
        # (60-200px breed) ook worden onderdrukt, niet alleen kleine structuren.
        bg_suppression = _local_background_suppression(
            img_smooth, cell_sigma=cell_bg_sigma * 1.5,
            percentile_cut=cell_bg_percentile
        )
        sup_n = Preprocessor.normalize(bg_suppression, pmin=1, pmax=99)

        # ── Stap 7b: CEL MASKER (NIEUW v3.0) ─────────────────────────────
        # Bereken waar de cellen zich bevinden. Detecties buiten de cellen
        # zijn per definitie fout-positieven en worden hier al uitgesloten.
        cell_mask = _compute_cell_mask(img_f, dilation_radius=20)

        # ── Stap 8: Gewogen score-fusie ───────────────────────────────────
        fused = 0.65 * log_norm + 0.35 * knn_norm
        fused = fused * np.clip(sup_n * 1.1, 0, 1)
        # Pas cel-masker toe: nul buiten de cellen
        fused = fused * cell_mask.astype(np.float32)

        # ── Stap 9: Drempelwaarde → binair masker ─────────────────────────
        # v3.0 FIX: gebruik percentiel > 50 als basis, zodat lege-achtergrond
        # ruis de drempelberekening niet omlaag trekt.
        signal_pixels = fused[fused > np.percentile(fused, 50)]
        if len(signal_pixels) > 0:
            p_thresh = np.percentile(signal_pixels, 100 * (1.0 - sensitivity * 2.5))
        else:
            p_thresh = sensitivity
        # Minimum drempel iets hoger zodat bijna-nul ruis wordt genegeerd
        mask = fused > max(p_thresh, sensitivity * 1.5)

        # ── Stap 10: Scherpheids-gate ─────────────────────────────────────
        # sigma_sharp verlaagd (0.7 was 1.2): scherper onderscheid aggregaat/vezel
        if use_sharpness_gate:
            sharp = _sharpness_map(img_clean, sigma_sharp=0.7)
            sharp_thresh = np.percentile(sharp[mask] if mask.sum() > 0 else sharp,
                                         sharpness_percentile)
            # Stricter: factor 0.7 was 0.5 — verwerpt meer diffuze detecties
            mask = mask & (sharp > sharp_thresh * 0.7)

        # ── Stap 11: Morfologische opschoning ─────────────────────────────
        mask = binary_closing(mask, disk(1))
        mask = remove_small_objects(mask, min_size=max(3, min_area))

        # ── Stap 12: Labeling ─────────────────────────────────────────────
        lbl = label(mask)

        # ── Stap 13: Grootte + morfologische filter ───────────────────────
        lbl = _filter_by_morphology(lbl, min_area, max_area)

        # ── Stap 14: Watershed splitsing van clusters ─────────────────────
        if use_watershed_split and lbl.max() > 0:
            lbl = _watershed_split_labels(lbl, img_clean,
                                          min_distance=min_split_distance)
            lbl = _filter_by_morphology(lbl, min_area, max_area)

        # ── Stap 15: Intensiteitscontrast + compactheid nafilter ──────────
        # Verwijdert resterende cellichaam/vezel artefacten via:
        # contrast t.o.v. omgeving, circularity, en brede achtergrondcheck
        if lbl.max() > 0:
            lbl = _intensity_roundness_filter(lbl, img_f, min_area, max_area)

        props = _label_to_props(lbl, img_f)
        return SegmentationResult(
            method_name="★★ Puncta Detector Pro",
            params=dict(min_sigma=min_sigma, max_sigma=max_sigma,
                        sensitivity=sensitivity, cell_bg_sigma=cell_bg_sigma,
                        use_sharpness_gate=use_sharpness_gate,
                        use_watershed_split=use_watershed_split),
            label_image=lbl, n_objects=int(lbl.max()),
            properties=props, time_seconds=time.time() - t0
        )

    # ── 1. Blob LoG ───────────────────────────────────────────────────────────
    @staticmethod
    def blob_log(img, min_sigma=1.5, max_sigma=7.0, num_sigma=10,
                 threshold=0.05, overlap=0.5, tophat_radius=0):
        t0 = time.time()
        img_norm = Preprocessor.normalize(img)
        if tophat_radius > 0:
            img_norm = white_tophat(img_norm, disk(tophat_radius))
            img_norm = Preprocessor.normalize(img_norm)
        blobs = blob_log(img_norm, min_sigma=min_sigma, max_sigma=max_sigma,
                         num_sigma=num_sigma, threshold=threshold, overlap=overlap)
        lbl = np.zeros(img.shape, dtype=np.int32)
        for i, (y, x, r) in enumerate(blobs):
            rr, cc = draw_disk((int(y), int(x)), max(1, int(r * 1.4)), shape=img.shape)
            lbl[rr, cc] = i + 1
        props = _blobs_to_props(blobs, img)
        return SegmentationResult(
            method_name="Blob LoG",
            params=dict(min_sigma=min_sigma, max_sigma=max_sigma,
                        num_sigma=num_sigma, threshold=threshold,
                        overlap=overlap, tophat_radius=tophat_radius),
            label_image=lbl, blob_list=blobs,
            n_objects=len(blobs), properties=props,
            time_seconds=time.time() - t0)

    # ── 2. Blob DoG ───────────────────────────────────────────────────────────
    @staticmethod
    def blob_dog(img, min_sigma=1.5, max_sigma=7.0, threshold=0.05, overlap=0.5,
                 tophat_radius=0):
        t0 = time.time()
        img_norm = Preprocessor.normalize(img)
        if tophat_radius > 0:
            img_norm = white_tophat(img_norm, disk(tophat_radius))
            img_norm = Preprocessor.normalize(img_norm)
        blobs = blob_dog(img_norm, min_sigma=min_sigma, max_sigma=max_sigma,
                         threshold=threshold, overlap=overlap)
        lbl = np.zeros(img.shape, dtype=np.int32)
        for i, (y, x, r) in enumerate(blobs):
            rr, cc = draw_disk((int(y), int(x)), max(1, int(r * 1.4)), shape=img.shape)
            lbl[rr, cc] = i + 1
        props = _blobs_to_props(blobs, img)
        return SegmentationResult(
            method_name="Blob DoG",
            params=dict(min_sigma=min_sigma, max_sigma=max_sigma,
                        threshold=threshold, overlap=overlap,
                        tophat_radius=tophat_radius),
            label_image=lbl, blob_list=blobs,
            n_objects=len(blobs), properties=props,
            time_seconds=time.time() - t0)

    # ── 3. KNN Intensity ──────────────────────────────────────────────────────
    @staticmethod
    def knn_intensity(img, k=10, z_score_thresh=2.5, min_area=5, max_area=2000):
        t0 = time.time()
        img_norm = Preprocessor.normalize(img).astype(np.float32)
        z_map = _knn_score_map(img_norm, k)
        mask = z_map > z_score_thresh
        return _mask_to_result(mask, img, "KNN Intensity",
                               dict(k=k, z_score_thresh=z_score_thresh),
                               min_area, max_area, t0)

    # ── 4. Combined Blob-LoG + KNN  ★ ────────────────────────────────────────
    @staticmethod
    def combined_blob_knn(img,
                          min_sigma: float = 1.5, max_sigma: float = 7.0,
                          blob_threshold: float = 0.04,
                          k: int = 15, z_score_thresh: float = 3.0,
                          min_area: int = 5, max_area: int = 1062,
                          weight_blob: float = 0.85,
                          fuse_threshold: float = 0.75) -> SegmentationResult:
        import math
        t0 = time.time()
        img_norm = Preprocessor.normalize(img).astype(np.float32)
        zero_frac = float((img_norm < 0.02).mean())
        if zero_frac > 0.25:
            img_proc = gaussian(img_norm, sigma=0.6).astype(np.float32)
        else:
            th_radius = max(3, int(max_sigma * 1.5))
            img_proc  = white_tophat(img_norm, disk(th_radius))
            img_proc  = Preprocessor.normalize(img_proc)
            img_proc  = gaussian(img_proc, sigma=0.6).astype(np.float32)
        blob_sc, _ = _blob_score_map(img_proc, min_sigma, max_sigma, blob_threshold)
        knn_raw = _knn_score_map(img_proc, k)
        p995 = np.percentile(knn_raw, 99.5)
        knn_sc = np.clip(knn_raw / p995, 0, 1) if p995 > 1e-6 else knn_raw.copy()
        fused = weight_blob * blob_sc + (1.0 - weight_blob) * knn_sc
        mask  = fused > fuse_threshold
        min_r = max(2, int(min_sigma * 1.0))
        return _mask_to_result(mask, img, "Combined Blob+KNN ★",
                               dict(min_sigma=min_sigma, max_sigma=max_sigma,
                                    blob_threshold=blob_threshold, k=k,
                                    weight_blob=weight_blob,
                                    fuse_threshold=fuse_threshold),
                               min_area, max_area, t0, min_distance=min_r)

    # ── 5. Otsu + Watershed ───────────────────────────────────────────────────
    @staticmethod
    def otsu_watershed(img, min_area=5, max_area=2000,
                       tophat_radius=5, gaussian_sigma=1.0):
        t0 = time.time()
        img_norm = Preprocessor.normalize(img).astype(np.float32)
        img_th = white_tophat(img_norm, disk(tophat_radius))
        img_th = gaussian(img_th, sigma=gaussian_sigma)
        thresh = threshold_otsu(img_th)
        mask = img_th > thresh
        return _mask_to_result(mask, img, "Otsu + Watershed",
                               dict(tophat_radius=tophat_radius,
                                    gaussian_sigma=gaussian_sigma),
                               min_area, max_area, t0)

    # ── 6. Multi-Otsu ─────────────────────────────────────────────────────────
    @staticmethod
    def multi_otsu(img, n_classes=3, min_area=5, max_area=2000, tophat_radius=5,
                   drempel_factor=0.80, cel_filter_radius=25, cel_filter_percentiel=85):
        t0 = time.time()
        img_norm = Preprocessor.normalize(img).astype(np.float32)
        if tophat_radius > 0:
            img_th = white_tophat(img_norm, disk(tophat_radius))
        else:
            img_th = img_norm.copy()
        img_th = gaussian(img_th, sigma=1.0)
        try:
            thresholds_3 = threshold_multiotsu(img_th, classes=3)
            drempel_basis = (thresholds_3[-2] + thresholds_3[-1]) / 2.0
            drempel = drempel_basis * drempel_factor
            mask = img_th > drempel
        except Exception:
            mask = img_th > threshold_otsu(img_th)
        try:
            r_cel = max(10, cel_filter_radius)
            lokaal_gem = gaussian(img_norm, sigma=r_cel)
            drempel_lokaal = np.percentile(lokaal_gem[mask], cel_filter_percentiel) \
                if mask.sum() > 0 else np.inf
            cel_artefact = mask & (lokaal_gem > drempel_lokaal)
            mask = mask & ~cel_artefact
        except Exception:
            pass
        return _mask_to_result(mask, img, f"Multi-Otsu ({n_classes} klassen)",
                               dict(n_classes=n_classes, tophat_radius=tophat_radius,
                                    drempel_factor=drempel_factor,
                                    cel_filter_radius=cel_filter_radius,
                                    cel_filter_percentiel=cel_filter_percentiel),
                               min_area, max_area, t0)

    # ── 7. Intensiteitspercentiel ─────────────────────────────────────────────
    @staticmethod
    def percentile_threshold(img, percentile=95.0, tophat_radius=5,
                              min_area=5, max_area=2000):
        t0 = time.time()
        img_norm = Preprocessor.normalize(img).astype(np.float32)
        img_th = white_tophat(img_norm, disk(tophat_radius))
        thresh = np.percentile(img_th, percentile)
        mask = img_th > thresh
        return _mask_to_result(mask, img, f"Percentiel ({percentile:.0f}%)",
                               dict(percentile=percentile, tophat_radius=tophat_radius),
                               min_area, max_area, t0)

    # ── 9. Adaptive Spot Enhancer  ★ ─────────────────────────────────────────
    @staticmethod
    def adaptive_spot_enhancer(img: np.ndarray,
                               # Schaalparameters (gebaseerd op annotatie-analyse)
                               # v4.2: min_radius verhoogd 1.2→1.5 (minder sub-pixel ruis)
                               #        max_radius verlaagd 8.0→7.0 (minder grote artefacten)
                               min_radius: float = 1.07,
                               max_radius: float = 8.57,
                               # Gevoeligheid — verhoogd (minder vals positieven)
                               # v4.2: 0.20→0.26 — sterker drempel, drastisch minder N
                               sensitivity: float = 0.411,
                               # DoG-ratio (σ2/σ1)
                               # v4.2: 1.6→1.7 — iets scherpere spot-selectie
                               dog_ratio: float = 1.68,
                               # Lokale achtergrond
                               # v4.2: 15→18 — grotere ring-filter, betere BG-schatting
                               local_bg_radius: int = 27,
                               # Morfologische nafilter
                               # v4.2: min_area 3→5 (geen sub-pixel ruis), max_area 500→400
                               min_area: int = 10,
                               max_area: int = 787,
                               # Ellips-tolerantie
                               # v4.2: 0.85→0.82 — iets strenger, minder vezels/draden
                               max_eccentricity: float = 0.70,
                               # Gebruik Hessian-blob-versterking
                               use_hessian_boost: bool = True) -> SegmentationResult:
        """
        ★ ADAPTIVE SPOT ENHANCER — geoptimaliseerd op basis van annotatieanalyse

        Gebaseerd op grondige analyse van 6 geannoteerde beelden (2590 objecten):
          • Mediaan radius: ~2 px  (p5=1.6 px, p95=5.8 px)
          • Hoge eccentriciteit (mediaan 0.79) — aggregaten zijn NIET altijd rond
          • Kleine oppervlakken (p50=12 px², p75=22 px², p95=106 px²)

        v4.2 wijzigingen (minder vals positieven):
          • sensitivity       0.20 → 0.26  (hogere drempel = minder N)
          • min_radius        1.2  → 1.5   (geen sub-pixel ruis)
          • max_radius        8.0  → 7.0   (minder grote artefacten)
          • dog_ratio         1.6  → 1.7   (scherpere spot-selectie)
          • local_bg_radius   15   → 18    (betere ring-filter)
          • min_area          3    → 5     (verwijdert 1-2 px ruis)
          • max_area          500  → 400   (geen grote cellichamen)
          • max_eccentricity  0.85 → 0.82  (strengere vezel-filter)
          • dilation_radius   8    → 5     (striktere celgrens)
          • p-drempel basis   p75  → p80   (alleen toppieken)
          • soliditeit        0.55 → 0.60  (minder grillige objecten)
          • intensiteitsratio 1.15 → 1.25  (sterkere contrast-eis)

        Werkwijze:
        ──────────
        1. LOKALE ACHTERGROND-NORMALISATIE
           Berekent voor elke pixel zijn lokale achtergrondniveau via een
           ring-filter (donut: groot disk - klein disk). Hierdoor worden
           zowel kleine stippen als grotere aggregaten gelijkmatig versterkt,
           ongeacht de globale verlichtingsverdeling.

        2. MULTI-SCHAAL DoG PYRADIDE (Difference of Gaussians)
           Berekent DoG over 6 schaalstappen van min_radius tot max_radius.
           DoG is bijzonder goed voor kleine compacte spots. Door de piramide
           worden aggregaten van 2–12 px diameter allemaal tegelijk gevangen.
           De σ2/σ1 ratio van 1.6 is geoptimaliseerd voor kleine biologische puncta.

        3. HESSIAN BLOB VERSTERKING (optioneel)
           De Hessian-matrix eigenwaarden detecteren blob-achtige structuren.
           Pixels waar beide eigenwaarden sterk negatief zijn (= helder bolletje
           op donkere achtergrond) krijgen een extra boost. Dit helpt juist bij
           de elliptische aggregaten die in de annotaties veel voorkomen.

        4. ADAPTIEVE DREMPELWAARDE
           Per regio (16×16 blokken) wordt de lokale drempel berekend.
           Dit compenseert voor inhomogene belichting van het microscoopveld.

        5. GROOTTE- EN VORM-NAFILTER
           Alleen objecten binnen het statistisch waargenomen bereik worden
           behouden. De eccentriciteitsdrempel is hoog (0.96) omdat de
           annotaties aantonen dat echte aggregaten ook elliptisch kunnen zijn.
        """
        import math
        t0 = time.time()

        img_f = img.astype(np.float32)

        # ── Stap 1: Normaliseer invoer ─────────────────────────────────────
        lo, hi = np.percentile(img_f, [1.0, 99.9])
        if hi > lo:
            img_f = np.clip((img_f - lo) / (hi - lo), 0.0, 1.0)

        # ── Stap 2: Lokale achtergrond-normalisatie via ring-filter ──────
        # Ring = grote disk - kleine disk → schat lokale achtergrond zonder
        # de spot zelf mee te nemen. Dit is robuuster dan een enkele Gaussiaan.
        from skimage.morphology import disk as _disk
        from skimage.filters import rank as _rank

        r_inner = max(1, int(max_radius * 0.8))
        r_outer = max(r_inner + 2, int(max_radius * 2.5))

        # Gebruik Gaussiaan als benadering van ring-filter (sneller, zelfde effect)
        bg_local = gaussian(img_f, sigma=r_outer)
        img_ring_sub = np.clip(img_f - bg_local * 0.85, 0.0, None)
        img_ring_sub = Preprocessor.normalize(img_ring_sub, pmin=0.5, pmax=99.5)

        # ── Stap 3: Multi-schaal DoG piramide ─────────────────────────────
        # DoG benadert de LoG maar is sneller en gevoeliger voor kleine spots.
        # Gebruik geomspace voor gelijkmatige dekking over het schaalgebied.
        n_scales = 6
        sigmas1 = np.geomspace(min_radius * 0.7, max_radius * 0.9, n_scales)
        sigmas2 = sigmas1 * dog_ratio

        dog_response = np.zeros_like(img_ring_sub)
        for s1, s2 in zip(sigmas1, sigmas2):
            g1 = gaussian(img_ring_sub, sigma=s1)
            g2 = gaussian(img_ring_sub, sigma=s2)
            dog = np.clip(g1 - g2, 0.0, None)
            dog_response = np.maximum(dog_response, dog)

        # ── Stap 4: Hessian blob versterking ──────────────────────────────
        hessian_score = np.zeros_like(img_ring_sub)
        if use_hessian_boost:
            try:
                from skimage.feature import hessian_matrix, hessian_matrix_eigvals
                # Gebruik de meest typische aggregaatschaal (~mediaan radius = 2 px)
                for hess_sigma in [min_radius * 1.2, (min_radius + max_radius) * 0.4]:
                    H_elems = hessian_matrix(img_ring_sub, sigma=hess_sigma, order='rc')
                    eigvals = hessian_matrix_eigvals(H_elems)
                    l1, l2 = eigvals[0], eigvals[1]
                    # Blob: beide eigenwaarden negatief → helder bolletje op donkere bg
                    blob_strength = np.where(
                        (l1 < 0) & (l2 < 0),
                        np.sqrt(l1 ** 2 + l2 ** 2),
                        0.0
                    ).astype(np.float32)
                    # Rondheid: l1 ≈ l2 → isotrope blob (hoge Rb)
                    denom = np.abs(l1) + 1e-8
                    Rb = (l2 / denom) ** 2  # 1 = perfect rond
                    # Aggregaten kunnen elliptisch zijn (Rb > 0.1 is al voldoende)
                    blob_score = blob_strength * np.clip(Rb * 2.0, 0.1, 1.0)
                    hessian_score = np.maximum(hessian_score, blob_score)

                # Normaliseer Hessian score
                p99h = np.percentile(hessian_score, 99.5)
                if p99h > 1e-8:
                    hessian_score = np.clip(hessian_score / p99h, 0.0, 1.0)
            except Exception:
                pass  # Hessian optioneel — geen fatale fout

        # ── Stap 5: Score-fusie DoG + Hessian ────────────────────────────
        p99_dog = np.percentile(dog_response, 99.5)
        if p99_dog > 1e-8:
            dog_norm = np.clip(dog_response / p99_dog, 0.0, 1.0)
        else:
            dog_norm = dog_response.copy()

        if use_hessian_boost:
            fused = 0.60 * dog_norm + 0.40 * hessian_score
        else:
            fused = dog_norm

        # ── Stap 6: Cel-masker (buiten cellen = nul) ──────────────────────
        # v4.2: dilation_radius verlaagd 8→5 — strengere celgrens, minder rand-artefacten
        cell_mask = _compute_cell_mask(img_f, dilation_radius=13)
        fused = fused * cell_mask.astype(np.float32)

        # ── Stap 7: Adaptieve drempelwaarde ──────────────────────────────
        # v4.2: percentiel verhoogd p75→p80 — alleen de sterkste signaalpieken
        # sensitivity * 1.2 (was * 1.5) — minder agressief naar beneden
        signal = fused[fused > np.percentile(fused, 80)]
        if len(signal) > 10:
            adaptive_thresh = np.percentile(signal, 100.0 * (1.0 - sensitivity * 1.2))
        else:
            adaptive_thresh = sensitivity
        thresh = max(adaptive_thresh, sensitivity)
        mask = fused > thresh

        # ── Stap 8: Morfologische opschoning ─────────────────────────────
        mask = binary_closing(mask, disk(1))
        mask = remove_small_objects(mask, min_size=max(2, min_area))

        # ── Stap 9: Labeling + morfologische nafilter ─────────────────────
        lbl = label(mask)

        # Aangepaste morfologische filter met strengere nafiltering (v4.1)
        out = lbl.copy()
        # Lokale achtergrond voor intensiteitscontrast-check
        bg_ref = gaussian(img_f, sigma=max_radius * 2.5)
        for region in regionprops(lbl):
            area = region.area
            if not (min_area <= area <= max_area):
                out[lbl == region.label] = 0
                continue
            # Eccentriciteitsdrempel — verlaagd van 0.96 → max_eccentricity (0.85)
            if region.eccentricity > max_eccentricity:
                out[lbl == region.label] = 0
                continue
            # Soliditeit-filter: verhoogd van 0.55 → 0.60 v4.2 (minder grillige artefacten)
            if region.solidity < 0.60:
                out[lbl == region.label] = 0
                continue
            # v4.2: intensiteitscontrast-drempel verhoogd 1.15→1.25
            # Een echt aggregaat moet merkbaar helderder zijn dan zijn omgeving.
            coords = region.coords
            spot_mean  = img_f[coords[:, 0], coords[:, 1]].mean()
            bg_mean    = bg_ref[coords[:, 0], coords[:, 1]].mean() + 1e-8
            if spot_mean / bg_mean < 1.28:
                out[lbl == region.label] = 0
                continue

        lbl = label(out > 0)

        # ── Stap 10: Watershed cluster-splitsing ──────────────────────────
        # Alleen voor grotere objecten (mogelijk clusters)
        if lbl.max() > 0:
            lbl = _watershed_split_labels(lbl, img_ring_sub, min_distance=3)
            # Nafilter na splitsing
            out2 = lbl.copy()
            for region in regionprops(lbl):
                if not (min_area <= region.area <= max_area):
                    out2[lbl == region.label] = 0
                    continue
                if region.eccentricity > max_eccentricity:
                    out2[lbl == region.label] = 0
            lbl = label(out2 > 0)

        props = _label_to_props(lbl, img.astype(np.float32))
        return SegmentationResult(
            method_name="★ Adaptive Spot Enhancer",
            params=dict(min_radius=min_radius, max_radius=max_radius,
                        sensitivity=sensitivity, dog_ratio=dog_ratio,
                        local_bg_radius=local_bg_radius,
                        use_hessian_boost=use_hessian_boost),
            label_image=lbl,
            n_objects=int(lbl.max()),
            properties=props,
            time_seconds=time.time() - t0
        )

    # ── 8. Custom Combination Builder ─────────────────────────────────────────
    @staticmethod
    def custom_combination(img,
                           use_blob_log=True, use_blob_dog=False,
                           use_knn=True, use_percentile=False,
                           use_tophat_otsu=False,
                           w_blob_log=1.0, w_blob_dog=1.0,
                           w_knn=1.0, w_percentile=1.0, w_tophat_otsu=1.0,
                           min_sigma=1.5, max_sigma=7.0,
                           blob_log_thresh=0.04, blob_dog_thresh=0.04,
                           knn_k=10, knn_z=2.0,
                           percentile=92.0, tophat_radius=5,
                           fuse_threshold=0.35,
                           logic="weighted_sum",
                           min_area=5, max_area=2000):
        t0 = time.time()
        img_norm = Preprocessor.normalize(img).astype(np.float32)
        h, w = img_norm.shape
        score_maps, weights = [], []

        if use_blob_log and w_blob_log > 0:
            sc, _ = _blob_score_map(img_norm, min_sigma, max_sigma, blob_log_thresh)
            score_maps.append(sc); weights.append(w_blob_log)
        if use_blob_dog and w_blob_dog > 0:
            blobs_d = blob_dog(img_norm, min_sigma=min_sigma, max_sigma=max_sigma,
                               threshold=blob_dog_thresh, overlap=0.5)
            sc = np.zeros((h, w), dtype=np.float32)
            for y, x, r in blobs_d:
                rr, cc = draw_disk((int(y), int(x)), max(1, int(r * 1.5)), shape=(h, w))
                sc[rr, cc] = 1.0
            score_maps.append(sc); weights.append(w_blob_dog)
        if use_knn and w_knn > 0:
            knn_z_map = _knn_score_map(img_norm, knn_k)
            sc = (knn_z_map > knn_z).astype(np.float32)
            score_maps.append(sc); weights.append(w_knn)
        if use_percentile and w_percentile > 0:
            img_th = white_tophat(img_norm, disk(tophat_radius))
            thr = np.percentile(img_th, percentile)
            sc = (img_th > thr).astype(np.float32)
            score_maps.append(sc); weights.append(w_percentile)
        if use_tophat_otsu and w_tophat_otsu > 0:
            img_th = white_tophat(img_norm, disk(tophat_radius))
            img_th = gaussian(img_th, sigma=1.0)
            thr = threshold_otsu(img_th)
            sc = (img_th > thr).astype(np.float32)
            score_maps.append(sc); weights.append(w_tophat_otsu)

        if not score_maps:
            mask = img_norm > threshold_otsu(img_norm)
        else:
            ws_arr = np.array(weights, dtype=np.float32)
            ws_arr = ws_arr / ws_arr.sum()
            if logic == "AND":
                mask = np.ones((h, w), dtype=bool)
                for sc in score_maps:
                    mask = mask & (sc > 0.3)
            elif logic == "OR":
                mask = np.zeros((h, w), dtype=bool)
                for sc in score_maps:
                    mask = mask | (sc > 0.3)
            else:
                fused = sum(wt * sc for wt, sc in zip(ws_arr, score_maps))
                mask = fused > fuse_threshold

        active = []
        if use_blob_log:    active.append("LoG")
        if use_blob_dog:    active.append("DoG")
        if use_knn:         active.append("KNN")
        if use_percentile:  active.append("Pct")
        if use_tophat_otsu: active.append("Otsu")
        name = f"Custom [{'+'.join(active)}] {logic}"
        return _mask_to_result(mask, img, name,
                               dict(logic=logic, fuse_threshold=fuse_threshold, active=active),
                               min_area, max_area, t0)


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER THREAD
# ═══════════════════════════════════════════════════════════════════════════════
class SegWorker(QThread):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func, self.args, self.kwargs = func, args, kwargs

    def run(self):
        try:
            self.finished.emit(self.func(*self.args, **self.kwargs))
        except Exception:
            self.error.emit(traceback.format_exc())


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
        self.btn_open.clicked.connect(self._open_file)
        fv.addWidget(self.btn_open)
        self.lbl_file = QLabel("Geen bestand geladen")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet("color:#58a6ff; font-size:11px;")
        fv.addWidget(self.lbl_file)
        self.cmb_series = QComboBox()
        self.cmb_series.currentIndexChanged.connect(self._series_changed)
        fv.addWidget(QLabel("Serie:"))
        fv.addWidget(self.cmb_series)
        lv.addWidget(grp_file)

        grp_view = QGroupBox("🎨  Weergave-instellingen")
        vv = QFormLayout(grp_view)
        self.cmb_channel = QComboBox()
        self.cmb_channel.currentIndexChanged.connect(self._refresh_image)
        vv.addRow("Kanaal:", self.cmb_channel)
        self.cmb_display = QComboBox()
        self.cmb_display.addItems(["Max Projectie", "Z-Slice"])
        self.cmb_display.currentIndexChanged.connect(self._toggle_view_mode)
        vv.addRow("Modus:", self.cmb_display)
        self.sld_z = QSlider(Qt.Horizontal)
        self.sld_z.setMinimum(0); self.sld_z.setMaximum(0)
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
        self.cmb_cmap.currentIndexChanged.connect(self._refresh_image)
        cv.addRow("Kleurkaart:", self.cmb_cmap)
        self.chk_autoscale = QCheckBox("Auto-schaal intensiteit")
        self.chk_autoscale.setChecked(True)
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

        # ── AANBEVOLEN pipeline knop ──────────────────────────────────────
        grp_quick = QGroupBox("⚡  Snelknoppen")
        qv = QVBoxLayout(grp_quick)
        btn_recommended = QPushButton("★  Aanbevolen pipeline (aggregaten)")
        btn_recommended.setObjectName("primary")
        btn_recommended.setToolTip(
            "Laadt de aanbevolen instellingen:\n"
            "• Achtergrondsubtractie AAN — Gaussiaan (σ=50)\n"
            "• Ruisonderdrukking AAN — Gaussiaan (σ=1.0)\n"
            "• Alle overige stappen UIT"
        )
        btn_recommended.clicked.connect(self._set_recommended)
        qv.addWidget(btn_recommended)
        btn_reset_all = QPushButton("↺  Alles terugzetten")
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
        iv.addWidget(self.chk_norm)
        pn = QHBoxLayout()
        pn.addWidget(QLabel("pmin:"))
        self.spn_pmin = QDoubleSpinBox()
        self.spn_pmin.setRange(0, 49); self.spn_pmin.setValue(1.0); self.spn_pmin.setSingleStep(0.5)
        pn.addWidget(self.spn_pmin)
        pn.addWidget(QLabel("pmax:"))
        self.spn_pmax = QDoubleSpinBox()
        self.spn_pmax.setRange(51, 100); self.spn_pmax.setValue(99.9); self.spn_pmax.setSingleStep(0.5)
        pn.addWidget(self.spn_pmax)
        iv.addLayout(pn)
        self._add_sep(iv)

        # Achtergrondsubtractie
        self.chk_bg = QCheckBox("Achtergrondsubtractie")
        iv.addWidget(self.chk_bg)
        bg_type_row = QHBoxLayout()
        bg_type_row.addWidget(QLabel("Methode:"))
        self.cmb_bg_method = QComboBox()
        self.cmb_bg_method.addItems(["Rolling Ball (morfologisch)", "Gaussiaan"])
        bg_type_row.addWidget(self.cmb_bg_method)
        iv.addLayout(bg_type_row)
        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("Radius/σ (px):"))
        self.spn_bg_radius = QSpinBox()
        self.spn_bg_radius.setRange(5, 500); self.spn_bg_radius.setValue(30)
        bg_row.addWidget(self.spn_bg_radius)
        iv.addLayout(bg_row)
        self._add_sep(iv)

        # Top-hat
        self.chk_tophat = QCheckBox("Top-Hat Filter")
        self.chk_tophat.setChecked(True)
        iv.addWidget(self.chk_tophat)
        th_mode_row = QHBoxLayout()
        th_mode_row.addWidget(QLabel("Modus:"))
        self.cmb_tophat_mode = QComboBox()
        self.cmb_tophat_mode.addItems(["Enkelvoudig", "Multi-schaal (aanbevolen)"])
        self.cmb_tophat_mode.setCurrentIndex(1)
        th_mode_row.addWidget(self.cmb_tophat_mode)
        iv.addLayout(th_mode_row)
        th_row = QHBoxLayout()
        th_row.addWidget(QLabel("Radius (px):"))
        self.spn_tophat = QSpinBox()
        self.spn_tophat.setRange(1, 50); self.spn_tophat.setValue(6)
        th_row.addWidget(self.spn_tophat)
        iv.addLayout(th_row)
        info_th = QLabel("Multi-schaal: gebruikt r, r+3, r-2 tegelijk")
        info_th.setStyleSheet("color:#8b949e; font-size:11px;")
        iv.addWidget(info_th)
        self._add_sep(iv)

        # Denoise
        self.chk_denoise = QCheckBox("Ruisonderdrukking")
        iv.addWidget(self.chk_denoise)
        dn_mode_row = QHBoxLayout()
        dn_mode_row.addWidget(QLabel("Methode:"))
        self.cmb_denoise_mode = QComboBox()
        self.cmb_denoise_mode.addItems(["Gaussiaan", "Bilateral (behoudt randen) ★"])
        self.cmb_denoise_mode.setCurrentIndex(1)
        dn_mode_row.addWidget(self.cmb_denoise_mode)
        iv.addLayout(dn_mode_row)
        dn_row = QHBoxLayout()
        dn_row.addWidget(QLabel("Sigma:"))
        self.spn_sigma = QDoubleSpinBox()
        self.spn_sigma.setRange(0.1, 10.0); self.spn_sigma.setValue(0.8); self.spn_sigma.setSingleStep(0.1)
        dn_row.addWidget(self.spn_sigma)
        iv.addLayout(dn_row)
        self._add_sep(iv)

        inner.setLayout(iv)
        scroll.setWidget(inner)
        sv.addWidget(scroll)
        lv.addWidget(grp_steps)

        btn_apply = QPushButton("▶  Pre-processing toepassen")
        btn_apply.setObjectName("primary")
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
        """
        Aanbevolen pipeline — alleen achtergrondsubtractie (Gaussiaan, σ=50)
        en ruisonderdrukking (Gaussiaan, σ=1.0).
        """
        self.chk_norm.setChecked(False)

        self.chk_bg.setChecked(True)
        self.cmb_bg_method.setCurrentIndex(1)     # Gaussiaan
        self.spn_bg_radius.setValue(50)

        self.chk_tophat.setChecked(False)

        self.chk_denoise.setChecked(True)
        self.cmb_denoise_mode.setCurrentIndex(0)  # Gaussiaan
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

        # Stap voor stap
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
#  TAB 3 — SEGMENTATIE
# ═══════════════════════════════════════════════════════════════════════════════
class SegmentationTab(QWidget):
    result_ready = pyqtSignal(object)

    METHOD_LIST = [
        "0. ★★ Puncta Detector Pro  (NIEUW — Aanbevolen)",
        "1. Blob LoG  (Laplaciaan van Gaussiaan)",
        "2. Blob DoG  (Verschil van Gaussianen)",
        "3. KNN Intensiteit",
        "4. ★  Combined Blob-LoG + KNN",
        "5. Otsu + Watershed",
        "6. Multi-Otsu (3 klassen)",
        "7. Intensiteitspercentiel",
        "8. ★ Adaptive Spot Enhancer  (DoG+Hessian, geopt. op annotaties)",
        "9. 🛠  Custom Combinatie Builder",
    ]

    def __init__(self, preprocess_tab: PreprocessTab):
        super().__init__()
        self.prep = preprocess_tab
        self.current_result: Optional[SegmentationResult] = None
        self.worker: Optional[SegWorker] = None
        self._build_ui()

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setSpacing(6)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(350)
        left_inner = QWidget()
        lv = QVBoxLayout(left_inner)
        lv.setSpacing(6)

        grp_m = QGroupBox("🔬  Segmentatiemethode")
        mv = QVBoxLayout(grp_m)
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(self.METHOD_LIST)
        self.cmb_method.setCurrentIndex(0)
        self.cmb_method.currentIndexChanged.connect(self._on_method_changed)
        mv.addWidget(self.cmb_method)
        lv.addWidget(grp_m)

        self.param_pages = QTabWidget()
        self.param_pages.setTabBarAutoHide(True)
        self.param_pages.tabBar().hide()

        # ── Pagina 0: PUNCTA DETECTOR PRO ────────────────────────────────
        pg_pro = QWidget()
        fv = QFormLayout(pg_pro)

        lbl_info = QLabel(
            "★★ Geoptimaliseerd voor confocale eiwit-aggregaten.\n"
            "Combineert multi-schaal LoG + KNN + cellichaamfilter\n"
            "en watershed-splitsing van clusters."
        )
        lbl_info.setStyleSheet("color:#58a6ff; font-size:11px; padding: 4px;")
        lbl_info.setWordWrap(True)
        fv.addRow(lbl_info)

        self.p_pro_minsig  = self._dbl(0.5, 10, 1.2, step=0.1)
        self.p_pro_maxsig  = self._dbl(2, 40, 20.0, step=0.5)
        self.p_pro_sens    = self._dbl(0.005, 0.5, 0.06, step=0.005)
        self.p_pro_bg_sig  = self._dbl(5, 80, 35.0, step=1.0)
        self.p_pro_bg_pct  = self._dbl(50, 99, 60.0, step=2.0)
        self.p_pro_sharp   = QCheckBox("Scherpheids-gate")
        self.p_pro_sharp.setChecked(True)
        self.p_pro_sharp_p = self._dbl(5, 60, 50.0, step=5.0)
        self.p_pro_wsh     = QCheckBox("Watershed cluster-splitsing")
        self.p_pro_wsh.setChecked(True)
        self.p_pro_wsh_d   = self._int(2, 20, 5)
        self.p_pro_mina    = self._int(1, 9999, 5)
        self.p_pro_maxa    = self._int(1, 99999, 3000)

        lbl_ms = QLabel("Min σ (px) — kleinste aggregaat:")
        lbl_ms.setToolTip("Kleinste detecteerbare aggregaatstraal in pixels.\n"
                          "Typisch 1–2 px voor kleine puncta.")
        fv.addRow(lbl_ms, self.p_pro_minsig)
        lbl_xs = QLabel("Max σ (px) — grootste aggregaat:")
        lbl_xs.setToolTip("Grootste detecteerbare aggregaatstraal in pixels.\n"
                          "Stel groter in bij grote aggregaten of clusters.")
        fv.addRow(lbl_xs, self.p_pro_maxsig)

        lbl_sens = QLabel("Gevoeligheid (lager = meer):")
        lbl_sens.setToolTip(
            "Drempelwaarde voor detectie.\n"
            "Lager (0.02) = meer maar ook meer valse positieven.\n"
            "Hoger (0.08) = minder maar zekerder detecties.\n"
            "Aanbevolen: 0.030–0.045"
        )
        fv.addRow(lbl_sens, self.p_pro_sens)

        lbl_bg = QLabel("Cel-achtergrond σ (px):")
        lbl_bg.setToolTip(
            "Hoe groot de Gaussiaan is die het cellichaam-niveau schat.\n"
            "Groter = ruimere definitie van 'cellichaam'.\n"
            "Typisch: 15–30 px voor 63x objectief."
        )
        fv.addRow(lbl_bg, self.p_pro_bg_sig)

        lbl_bgp = QLabel("Cel-onderdrukking percentiel:")
        lbl_bgp.setToolTip(
            "Gebieden waarvan het lokaal gemiddelde boven dit percentiel zit\n"
            "worden als cellichaamachtergrond beschouwd en onderdrukt.\n"
            "Verlaag (60–70) voor meer agressieve cellichaam-verwijdering."
        )
        fv.addRow(lbl_bgp, self.p_pro_bg_pct)

        fv.addRow(self.p_pro_sharp)
        lbl_shp = QLabel("Scherpheid percentiel (min):")
        lbl_shp.setToolTip(
            "Detecties in gebieden met lagere scherpheid dan dit percentiel\n"
            "worden verworpen. Aggregaten zijn scherp; cellichaamvlekken niet.\n"
            "Verhoog om meer cellichaam-artefacten te verwijderen."
        )
        fv.addRow(lbl_shp, self.p_pro_sharp_p)

        fv.addRow(self.p_pro_wsh)
        lbl_wshd = QLabel("Min. splitsingsafstand (px):")
        lbl_wshd.setToolTip(
            "Minimale afstand tussen twee aggregaatpieken om ze als apart\n"
            "te beschouwen bij watershed-splitsing."
        )
        fv.addRow(lbl_wshd, self.p_pro_wsh_d)

        fv.addRow("Min oppervlak px²:", self.p_pro_mina)
        fv.addRow("Max oppervlak px²:", self.p_pro_maxa)
        self.param_pages.addTab(pg_pro, "Pro")

        # ── Pagina 1: Blob LoG/DoG ────────────────────────────────────────
        pg_blob = QWidget(); fv = QFormLayout(pg_blob)
        self.p_blob_minsig = self._dbl(0.5, 20, 1.5)
        self.p_blob_maxsig = self._dbl(1,   50, 7.0)
        self.p_blob_numsig = self._int(3, 30, 10)
        self.p_blob_thr    = self._dbl(0.001, 1, 0.05, step=0.005)
        self.p_blob_ov     = self._dbl(0, 1, 0.5, step=0.05)
        self.p_blob_tophat = self._int(0, 30, 0)
        self.p_blob_mina   = self._int(1, 9999, 5)
        self.p_blob_maxa   = self._int(1, 99999, 2000)
        fv.addRow("Min σ (px):",            self.p_blob_minsig)
        fv.addRow("Max σ (px):",            self.p_blob_maxsig)
        fv.addRow("Aantal σ-stappen:",      self.p_blob_numsig)
        fv.addRow("Drempelwaarde:",         self.p_blob_thr)
        fv.addRow("Overlap (0–1):",         self.p_blob_ov)
        fv.addRow("Pre-tophat radius (0=uit):", self.p_blob_tophat)
        fv.addRow("Min oppervlak px²:",     self.p_blob_mina)
        fv.addRow("Max oppervlak px²:",     self.p_blob_maxa)
        self.param_pages.addTab(pg_blob, "Blob")

        # ── Pagina 2: KNN ─────────────────────────────────────────────────
        pg_knn = QWidget(); fv = QFormLayout(pg_knn)
        self.p_knn_k    = self._int(3, 100, 10)
        self.p_knn_z    = self._dbl(0.5, 10, 2.5, step=0.1)
        self.p_knn_mina = self._int(1, 9999, 5)
        self.p_knn_maxa = self._int(1, 99999, 2000)
        fv.addRow("k buren (spatiaal):", self.p_knn_k)
        fv.addRow("Intensiteit Z-drempel:", self.p_knn_z)
        fv.addRow("Min oppervlak px²:",  self.p_knn_mina)
        fv.addRow("Max oppervlak px²:",  self.p_knn_maxa)
        self.param_pages.addTab(pg_knn, "KNN")

        # ── Pagina 3: Combined Blob+KNN ───────────────────────────────────
        pg_comb = QWidget(); fv = QFormLayout(pg_comb)
        self.p_c_minsig = self._dbl(0.5, 20, 1.5)
        self.p_c_maxsig = self._dbl(1,   50, 7.0)
        self.p_c_bthr   = self._dbl(0.001, 1, 0.04, step=0.005)
        self.p_c_k      = self._int(3, 100, 15)
        self.p_c_zthr   = self._dbl(0.5, 10, 3.0, step=0.1)
        self.p_c_wblob  = self._dbl(0, 1, 0.85, step=0.05)
        self.p_c_fthr   = self._dbl(0.05, 0.95, 0.75, step=0.05)
        self.p_c_mina   = self._int(1, 9999, 5)
        self.p_c_maxa   = self._int(1, 99999, 1062)
        fv.addRow("Min σ (px):",        self.p_c_minsig)
        fv.addRow("Max σ (px):",        self.p_c_maxsig)
        fv.addRow("Blob drempel:",      self.p_c_bthr)
        fv.addRow("KNN k buren:",       self.p_c_k)
        fv.addRow("KNN Z-drempel:",     self.p_c_zthr)
        fv.addRow("Blob gewicht (0–1):", self.p_c_wblob)
        fv.addRow("Fusie-drempel:",     self.p_c_fthr)
        fv.addRow("Min oppervlak px²:", self.p_c_mina)
        fv.addRow("Max oppervlak px²:", self.p_c_maxa)
        self.param_pages.addTab(pg_comb, "Combined")

        # ── Pagina 4: Otsu ────────────────────────────────────────────────
        pg_otsu = QWidget(); fv = QFormLayout(pg_otsu)
        self.p_ot_tophat  = self._int(1, 30, 5)
        self.p_ot_gsigma  = self._dbl(0.1, 5, 1.0, step=0.1)
        self.p_ot_mina    = self._int(1, 9999, 5)
        self.p_ot_maxa    = self._int(1, 99999, 2000)
        fv.addRow("Top-hat radius:",    self.p_ot_tophat)
        fv.addRow("Gaussiaan σ:",       self.p_ot_gsigma)
        fv.addRow("Min oppervlak px²:", self.p_ot_mina)
        fv.addRow("Max oppervlak px²:", self.p_ot_maxa)
        self.param_pages.addTab(pg_otsu, "Otsu")

        # ── Pagina 5: Multi-Otsu ──────────────────────────────────────────
        pg_motsu = QWidget(); fv = QFormLayout(pg_motsu)
        self.p_mo_classes      = self._int(2, 5, 3)
        self.p_mo_tophat       = self._int(0, 30, 5)
        self.p_mo_drempel_fac  = self._dbl(0.5, 1.2, 0.80, step=0.05)
        self.p_mo_cel_radius   = self._int(5, 100, 25)
        self.p_mo_cel_pct      = self._dbl(50, 99, 85.0, step=2.0)
        self.p_mo_mina         = self._int(1, 9999, 5)
        self.p_mo_maxa         = self._int(1, 99999, 2000)
        fv.addRow("Klassen:",              self.p_mo_classes)
        fv.addRow("Top-hat radius:",       self.p_mo_tophat)
        fv.addRow("Drempel factor:",       self.p_mo_drempel_fac)
        fv.addRow("Cel-filter radius:",    self.p_mo_cel_radius)
        fv.addRow("Cel-filter percentiel:", self.p_mo_cel_pct)
        fv.addRow("Min oppervlak px²:",    self.p_mo_mina)
        fv.addRow("Max oppervlak px²:",    self.p_mo_maxa)
        self.param_pages.addTab(pg_motsu, "Multi-Otsu")

        # ── Pagina 6: Percentiel ──────────────────────────────────────────
        pg_pct = QWidget(); fv = QFormLayout(pg_pct)
        self.p_pct_pct    = self._dbl(50, 99.9, 95.0, step=0.5)
        self.p_pct_tophat = self._int(1, 50, 5)
        self.p_pct_mina   = self._int(1, 9999, 5)
        self.p_pct_maxa   = self._int(1, 99999, 2000)
        fv.addRow("Percentiel %:",      self.p_pct_pct)
        fv.addRow("Top-hat radius:",    self.p_pct_tophat)
        fv.addRow("Min oppervlak px²:", self.p_pct_mina)
        fv.addRow("Max oppervlak px²:", self.p_pct_maxa)
        self.param_pages.addTab(pg_pct, "Percentiel")

        # ── Pagina 7: Adaptive Spot Enhancer ─────────────────────────────
        pg_ase = QWidget()
        fv = QFormLayout(pg_ase)

        lbl_ase_info = QLabel(
            "★ Geoptimaliseerd op 2590 geannoteerde aggregaten. (v4.2)\n"
            "Mediaan radius ~2 px, eccentriciteit tot 0.82.\n"
            "v4.2: sensitivity 0.26, soliditeit 0.60, contrast 1.25,\n"
            "p80-drempel, dilation=5 → drastisch minder vals positieven."
        )
        lbl_ase_info.setStyleSheet("color:#58a6ff; font-size:11px; padding:4px;")
        lbl_ase_info.setWordWrap(True)
        fv.addRow(lbl_ase_info)

        self.p_ase_minr   = self._dbl(0.5, 10, 1.07, step=0.05)   # was 1.2
        self.p_ase_maxr   = self._dbl(2, 30, 8.57, step=0.1)      # was 8.0
        self.p_ase_sens   = self._dbl(0.05, 0.99, 0.411, step=0.01) # max verhoogd naar 0.99, default 0.411
        self.p_ase_dog    = self._dbl(1.2, 3.0, 1.68, step=0.02)  # was 1.6
        self.p_ase_bgr    = self._int(5, 60, 27)                  # was 15
        self.p_ase_hess   = QCheckBox("Hessian blob versterking")
        self.p_ase_hess.setChecked(True)
        self.p_ase_mina   = self._int(1, 999, 10)                 # was 3
        self.p_ase_maxa   = self._int(1, 9999, 787)               # was 500
        self.p_ase_maxecc = self._dbl(0.5, 0.99, 0.70, step=0.02) # was 0.85

        lbl_minr = QLabel("Min radius (px):")
        lbl_minr.setToolTip("Kleinste te detecteren aggregaatstraal.\n"
                            "Annotaties: p5 radius = 1.6 px → gebruik 1.0")
        fv.addRow(lbl_minr, self.p_ase_minr)

        lbl_maxr = QLabel("Max radius (px):")
        lbl_maxr.setToolTip("Grootste te detecteren aggregaatstraal.\n"
                            "Annotaties: p95 radius = 5.8 px → gebruik 6–10")
        fv.addRow(lbl_maxr, self.p_ase_maxr)

        lbl_sens = QLabel("Gevoeligheid:")
        lbl_sens.setToolTip("Drempelwaarde voor detectie.\n"
                            "0.15 = gevoelig (meer detecties), 0.25 = selectief (minder FP).\n"
                            "Standaard 0.20 geeft goede balans.")
        fv.addRow(lbl_sens, self.p_ase_sens)

        lbl_dog = QLabel("DoG ratio (σ2/σ1):")
        lbl_dog.setToolTip("Verhouding tussen de twee Gaussianen per schaalstap.\n"
                           "1.6 is optimaal voor kleine biologische puncta.")
        fv.addRow(lbl_dog, self.p_ase_dog)

        lbl_bgr = QLabel("Lokale BG radius (px):")
        lbl_bgr.setToolTip("Straal voor ring-achtergrond schatting.\n"
                           "Typisch 10–20 px voor 63x objectief.")
        fv.addRow(lbl_bgr, self.p_ase_bgr)

        fv.addRow(self.p_ase_hess)

        lbl_mina = QLabel("Min oppervlak px²:")
        lbl_mina.setToolTip("Annotaties: p5 area = 8 px² → gebruik 3–5")
        fv.addRow(lbl_mina, self.p_ase_mina)

        lbl_maxa = QLabel("Max oppervlak px²:")
        lbl_maxa.setToolTip("Annotaties: p95 area = 106 px² → gebruik 200–500")
        fv.addRow(lbl_maxa, self.p_ase_maxa)

        lbl_ecc = QLabel("Max eccentriciteit:")
        lbl_ecc.setToolTip("Annotaties tonen mediaan eccentriciteit 0.79.\n"
                           "v4.1: verlaagd naar 0.85 om lineaire artefacten/vezels te filteren.\n"
                           "Verhoog naar 0.92 als je elliptische aggregaten mist.")
        fv.addRow(lbl_ecc, self.p_ase_maxecc)

        self.param_pages.addTab(pg_ase, "ASE")

        # ── Pagina 8: Custom Builder ──────────────────────────────────────
        pg_cust = QWidget(); cv = QVBoxLayout(pg_cust)
        cv.addWidget(QLabel("✅ Selecteer scorekaarten:"))
        self.p_cu_use_blog = QCheckBox("Blob LoG");          self.p_cu_use_blog.setChecked(True)
        self.p_cu_use_bdog = QCheckBox("Blob DoG");          self.p_cu_use_bdog.setChecked(False)
        self.p_cu_use_knn  = QCheckBox("KNN Intensiteit");   self.p_cu_use_knn.setChecked(True)
        self.p_cu_use_pct  = QCheckBox("Intensiteitspercentiel"); self.p_cu_use_pct.setChecked(False)
        self.p_cu_use_otsu = QCheckBox("Otsu (Top-hat)");    self.p_cu_use_otsu.setChecked(False)
        for chk in [self.p_cu_use_blog, self.p_cu_use_bdog, self.p_cu_use_knn,
                    self.p_cu_use_pct, self.p_cu_use_otsu]:
            cv.addWidget(chk)
        cv.addWidget(QLabel("⚖  Gewichten:"))
        wf = QFormLayout()
        self.p_cu_w_blog = self._dbl(0, 5, 1.0, step=0.1)
        self.p_cu_w_bdog = self._dbl(0, 5, 1.0, step=0.1)
        self.p_cu_w_knn  = self._dbl(0, 5, 1.0, step=0.1)
        self.p_cu_w_pct  = self._dbl(0, 5, 1.0, step=0.1)
        self.p_cu_w_otsu = self._dbl(0, 5, 1.0, step=0.1)
        wf.addRow("w Blob LoG:", self.p_cu_w_blog)
        wf.addRow("w Blob DoG:", self.p_cu_w_bdog)
        wf.addRow("w KNN:",      self.p_cu_w_knn)
        wf.addRow("w Percentiel:", self.p_cu_w_pct)
        wf.addRow("w Otsu:",     self.p_cu_w_otsu)
        cv.addLayout(wf)
        cv.addWidget(QLabel("🔗 Fusie-logica:"))
        self.p_cu_logic = QComboBox()
        self.p_cu_logic.addItems(["weighted_sum", "AND", "OR"])
        cv.addWidget(self.p_cu_logic)
        sf = QFormLayout()
        self.p_cu_fuse_thr = self._dbl(0.05, 0.95, 0.35, step=0.05)
        self.p_cu_minsig   = self._dbl(0.5, 20, 1.5)
        self.p_cu_maxsig   = self._dbl(1,   50, 7.0)
        self.p_cu_knn_k    = self._int(3, 100, 10)
        self.p_cu_knn_z    = self._dbl(0.5, 10, 2.0, step=0.1)
        self.p_cu_pct      = self._dbl(50, 99.9, 92.0, step=1.0)
        self.p_cu_tophat   = self._int(1, 50, 5)
        self.p_cu_mina     = self._int(1, 9999, 5)
        self.p_cu_maxa     = self._int(1, 99999, 2000)
        sf.addRow("Fusie-drempel:", self.p_cu_fuse_thr)
        sf.addRow("Min σ:",         self.p_cu_minsig)
        sf.addRow("Max σ:",         self.p_cu_maxsig)
        sf.addRow("KNN k:",         self.p_cu_knn_k)
        sf.addRow("KNN Z-drempel:", self.p_cu_knn_z)
        sf.addRow("Percentiel %:",  self.p_cu_pct)
        sf.addRow("Top-hat radius:", self.p_cu_tophat)
        sf.addRow("Min oppervlak px²:", self.p_cu_mina)
        sf.addRow("Max oppervlak px²:", self.p_cu_maxa)
        cv.addLayout(sf)
        self.param_pages.addTab(pg_cust, "Custom")

        lv.addWidget(self.param_pages)
        self._on_method_changed(0)

        # Overlay-opties
        grp_ov = QGroupBox("🖍  Overlay-opties")
        ovf = QFormLayout(grp_ov)
        self.chk_show_circles = QCheckBox("Teken contouren"); self.chk_show_circles.setChecked(True)
        self.chk_show_numbers = QCheckBox("Toon nummers");    self.chk_show_numbers.setChecked(True)
        self.chk_show_fill    = QCheckBox("Gevuld gebied");   self.chk_show_fill.setChecked(True)
        self.cmb_circle_color = QComboBox()
        self.cmb_circle_color.addItems(["#00ffcc","#ff4466","#ffff00","#ffffff","#00aaff","#ff8800"])
        self.spn_circle_lw  = self._dbl(0.5, 5, 1.2, step=0.2)
        self.spn_font_size  = self._dbl(3, 16, 6.5, step=0.5)
        self.cmb_cmap_seg   = QComboBox()
        self.cmb_cmap_seg.addItems(["hot","gray","inferno","magma","viridis","plasma"])
        ovf.addWidget(self.chk_show_circles)
        ovf.addWidget(self.chk_show_numbers)
        ovf.addWidget(self.chk_show_fill)
        ovf.addRow("Kleur:",          self.cmb_circle_color)
        ovf.addRow("Lijnbreedte:",    self.spn_circle_lw)
        ovf.addRow("Lettergrootte:", self.spn_font_size)
        ovf.addRow("Achtergrond:",    self.cmb_cmap_seg)
        lv.addWidget(grp_ov)

        self.btn_run = QPushButton("▶  Segmentatie uitvoeren")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self._run)
        lv.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        lv.addWidget(self.progress)

        grp_stats = QGroupBox("📊  Statistieken")
        sv = QVBoxLayout(grp_stats)
        self.txt_stats = QTextEdit()
        self.txt_stats.setReadOnly(True)
        self.txt_stats.setMinimumHeight(90)
        sv.addWidget(self.txt_stats)
        lv.addWidget(grp_stats)

        btn_csv = QPushButton("💾  Exporteer CSV")
        btn_csv.clicked.connect(self._export_csv)
        lv.addWidget(btn_csv)

        btn_img = QPushButton("🖼  Exporteer geannoteerd beeld")
        btn_img.clicked.connect(self._export_image)
        lv.addWidget(btn_img)

        lv.addStretch()
        left_inner.setLayout(lv)
        left_scroll.setWidget(left_inner)
        main.addWidget(left_scroll)

        right_w = QWidget()
        rv = QVBoxLayout(right_w)
        rv.setSpacing(4)
        self.canvas_seg = MplCanvas(width=11, height=9)
        nav = NavigationToolbar(self.canvas_seg, self)
        nav.setStyleSheet("background:#f0f3f6; border-bottom: 1px solid #d0d7de; border-radius: 6px 6px 0 0;")
        rv.addWidget(nav)
        rv.addWidget(self.canvas_seg)
        main.addWidget(right_w)

    def _dbl(self, lo, hi, val, step=0.1):
        s = QDoubleSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setSingleStep(step)
        return s

    def _int(self, lo, hi, val, step=1):
        s = QSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setSingleStep(step)
        return s

    # Methode-index → parameter-pagina-index
    # 0=Pro, 1=Blob, 2=Blob, 3=KNN, 4=Combined, 5=Otsu, 6=Multi-Otsu, 7=Pct, 8=ASE, 9=Custom
    _METHOD_PAGE = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8}

    def _on_method_changed(self, idx):
        page = self._METHOD_PAGE.get(idx, 0)
        self.param_pages.setCurrentIndex(page)

    def _build_call(self, idx, img):
        e = SegmentationEngine
        if idx == 0:
            return e.puncta_detector_pro, dict(
                img=img,
                min_sigma=self.p_pro_minsig.value(),
                max_sigma=self.p_pro_maxsig.value(),
                sensitivity=self.p_pro_sens.value(),
                cell_bg_sigma=self.p_pro_bg_sig.value(),
                cell_bg_percentile=self.p_pro_bg_pct.value(),
                use_sharpness_gate=self.p_pro_sharp.isChecked(),
                sharpness_percentile=self.p_pro_sharp_p.value(),
                use_watershed_split=self.p_pro_wsh.isChecked(),
                min_split_distance=self.p_pro_wsh_d.value(),
                min_area=self.p_pro_mina.value(),
                max_area=self.p_pro_maxa.value()
            )
        elif idx == 1:
            return e.blob_log, dict(
                img=img, min_sigma=self.p_blob_minsig.value(),
                max_sigma=self.p_blob_maxsig.value(), num_sigma=self.p_blob_numsig.value(),
                threshold=self.p_blob_thr.value(), overlap=self.p_blob_ov.value(),
                tophat_radius=self.p_blob_tophat.value())
        elif idx == 2:
            return e.blob_dog, dict(
                img=img, min_sigma=self.p_blob_minsig.value(),
                max_sigma=self.p_blob_maxsig.value(), threshold=self.p_blob_thr.value(),
                overlap=self.p_blob_ov.value(), tophat_radius=self.p_blob_tophat.value())
        elif idx == 3:
            return e.knn_intensity, dict(
                img=img, k=self.p_knn_k.value(),
                z_score_thresh=self.p_knn_z.value(),
                min_area=self.p_knn_mina.value(), max_area=self.p_knn_maxa.value())
        elif idx == 4:
            return e.combined_blob_knn, dict(
                img=img,
                min_sigma=self.p_c_minsig.value(),
                max_sigma=self.p_c_maxsig.value(),
                blob_threshold=self.p_c_bthr.value(),
                k=self.p_c_k.value(),
                z_score_thresh=self.p_c_zthr.value(),
                weight_blob=self.p_c_wblob.value(),
                fuse_threshold=self.p_c_fthr.value(),
                min_area=self.p_c_mina.value(),
                max_area=self.p_c_maxa.value())
        elif idx == 5:
            return e.otsu_watershed, dict(
                img=img, tophat_radius=self.p_ot_tophat.value(),
                gaussian_sigma=self.p_ot_gsigma.value(),
                min_area=self.p_ot_mina.value(), max_area=self.p_ot_maxa.value())
        elif idx == 6:
            return e.multi_otsu, dict(
                img=img, n_classes=self.p_mo_classes.value(),
                tophat_radius=self.p_mo_tophat.value(),
                drempel_factor=self.p_mo_drempel_fac.value(),
                cel_filter_radius=self.p_mo_cel_radius.value(),
                cel_filter_percentiel=self.p_mo_cel_pct.value(),
                min_area=self.p_mo_mina.value(), max_area=self.p_mo_maxa.value())
        elif idx == 7:
            return e.percentile_threshold, dict(
                img=img, percentile=self.p_pct_pct.value(),
                tophat_radius=self.p_pct_tophat.value(),
                min_area=self.p_pct_mina.value(), max_area=self.p_pct_maxa.value())
        elif idx == 8:
            return e.adaptive_spot_enhancer, dict(
                img=img,
                min_radius=self.p_ase_minr.value(),
                max_radius=self.p_ase_maxr.value(),
                sensitivity=self.p_ase_sens.value(),
                dog_ratio=self.p_ase_dog.value(),
                local_bg_radius=self.p_ase_bgr.value(),
                use_hessian_boost=self.p_ase_hess.isChecked(),
                min_area=self.p_ase_mina.value(),
                max_area=self.p_ase_maxa.value(),
                max_eccentricity=self.p_ase_maxecc.value())
        elif idx == 9:
            return e.custom_combination, dict(
                img=img,
                use_blob_log=self.p_cu_use_blog.isChecked(),
                use_blob_dog=self.p_cu_use_bdog.isChecked(),
                use_knn=self.p_cu_use_knn.isChecked(),
                use_percentile=self.p_cu_use_pct.isChecked(),
                use_tophat_otsu=self.p_cu_use_otsu.isChecked(),
                w_blob_log=self.p_cu_w_blog.value(),
                w_blob_dog=self.p_cu_w_bdog.value(),
                w_knn=self.p_cu_w_knn.value(),
                w_percentile=self.p_cu_w_pct.value(),
                w_tophat_otsu=self.p_cu_w_otsu.value(),
                min_sigma=self.p_cu_minsig.value(),
                max_sigma=self.p_cu_maxsig.value(),
                blob_log_thresh=self.p_cu_w_blog.value(),
                blob_dog_thresh=self.p_cu_w_bdog.value(),
                knn_k=self.p_cu_knn_k.value(),
                knn_z=self.p_cu_knn_z.value(),
                percentile=self.p_cu_pct.value(),
                tophat_radius=self.p_cu_tophat.value(),
                fuse_threshold=self.p_cu_fuse_thr.value(),
                logic=self.p_cu_logic.currentText(),
                min_area=self.p_cu_mina.value(),
                max_area=self.p_cu_maxa.value())
        return e.otsu_watershed, dict(img=img)

    def _run(self):
        img = self.prep.get_image()
        if img is None:
            QMessageBox.warning(self, "Geen beeld", "Laad en verwerk eerst een beeld.")
            return
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        func, kwargs = self._build_call(self.cmb_method.currentIndex(), img)
        self.worker = SegWorker(func, **kwargs)
        self.worker.finished.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_result(self, result):
        self.current_result = result
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self._display_result(result)
        self._show_stats(result)
        self.result_ready.emit(result)

    def _on_error(self, err):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Segmentatiefout", err)

    def _display_result(self, result):
        img = self.prep.get_image()
        cmap = self.cmb_cmap_seg.currentText()
        circle_color = self.cmb_circle_color.currentText()
        show_circles = self.chk_show_circles.isChecked()
        show_numbers = self.chk_show_numbers.isChecked()
        show_fill    = self.chk_show_fill.isChecked()
        lw    = self.spn_circle_lw.value()
        fsize = self.spn_font_size.value()

        self.canvas_seg.fig.clf()
        axes = self.canvas_seg.fig.subplots(1, 2)

        axes[0].imshow(img, cmap=cmap, aspect="equal", interpolation="nearest")
        axes[0].set_title("Pre-processed invoer", color="#79c0ff", fontsize=10)
        axes[0].axis("off"); axes[0].set_facecolor("#0d1117")

        if show_fill and result.label_image is not None and result.label_image.max() > 0:
            overlay = overlay_labels_on_image(img, result.label_image, alpha=0.35)
            axes[1].imshow(overlay, aspect="equal", interpolation="nearest")
        else:
            axes[1].imshow(img, cmap=cmap, aspect="equal", interpolation="nearest")

        if show_circles and result.properties:
            draw_actual_contours(axes[1], result, circle_color=circle_color,
                                 text_color="#ffffff", circle_lw=lw,
                                 font_size=fsize, show_numbers=show_numbers)

        axes[1].set_title(
            f"{result.method_name}   N = {result.n_objects}   t = {result.time_seconds:.2f}s",
            color="#79c0ff", fontsize=10)
        axes[1].axis("off"); axes[1].set_facecolor("#0d1117")
        self.canvas_seg.fig.tight_layout(pad=0.5)
        self.canvas_seg.draw_idle()

    def _show_stats(self, result):
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
        path, _ = QFileDialog.getSaveFileName(self, "Sla geannoteerd beeld op", "",
                                              "PNG (*.png);;TIFF (*.tif)")
        if not path:
            return
        import matplotlib.pyplot as plt
        img = self.prep.get_image()
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="#0d1117")
        cmap = self.cmb_cmap_seg.currentText()
        axes[0].imshow(img, cmap=cmap, aspect="equal")
        axes[0].set_title("Invoer", color="#79c0ff"); axes[0].axis("off")
        if self.chk_show_fill.isChecked() and self.current_result.label_image is not None:
            overlay = overlay_labels_on_image(img, self.current_result.label_image)
            axes[1].imshow(overlay, aspect="equal")
        else:
            axes[1].imshow(img, cmap=cmap, aspect="equal")
        if self.chk_show_circles.isChecked():
            draw_actual_contours(axes[1], self.current_result,
                                 circle_color=self.cmb_circle_color.currentText(),
                                 show_numbers=self.chk_show_numbers.isChecked())
        axes[1].set_title(
            f"{self.current_result.method_name}  N={self.current_result.n_objects}",
            color="#79c0ff")
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig)
        QMessageBox.information(self, "Opgeslagen", f"Beeld opgeslagen: {path}")


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
#  DEEP LEARNING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class DeepLearningEngine:
    """
    Laadt een PyTorch .pth model en voert segmentatie uit op een 2D beeld.
    Het model moet een 2D segmentatie-uitvoer produceren (H, W) of (1, H, W)
    gegeven een invoer van (1, 1, H, W) of (1, C, H, W).
    """

    @staticmethod
    def run(img: np.ndarray,
            model_path: str,
            threshold: float = 0.5,
            tile_size: int = 512,
            overlap: int = 64,
            use_tta: bool = False,
            min_area: int = 5,
            max_area: int = 50000,
            device_name: str = "auto") -> SegmentationResult:
        t0 = time.time()

        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is niet geïnstalleerd.\n"
                "Installeer met: pip install torch torchvision"
            )

        # ── Device ──────────────────────────────────────────────────────────
        if device_name == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_name)

        # ── Model laden ─────────────────────────────────────────────────────
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Ondersteuning voor verschillende opslagformaten
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # Volledig model of raw state_dict opgeslagen als dict
                state_dict = checkpoint
        else:
            # Volledig model object
            model = checkpoint
            state_dict = None

        # Als alleen state_dict: bouw een generiek UNet-achtig model
        if state_dict is not None:
            model = DeepLearningEngine._build_generic_unet(state_dict, device)

        model = model.to(device)
        model.eval()

        # ── Beeld normaliseren ──────────────────────────────────────────────
        img_f = img.astype(np.float32)
        lo, hi = np.percentile(img_f, [1, 99.9])
        if hi > lo:
            img_f = np.clip((img_f - lo) / (hi - lo), 0, 1)

        H, W = img_f.shape

        # ── Inferentie (tiled voor grote beelden) ──────────────────────────
        prob_map = DeepLearningEngine._tiled_inference(
            model, img_f, device, tile_size, overlap, use_tta
        )

        # ── Drempel & post-processing ───────────────────────────────────────
        binary = prob_map >= threshold
        binary = remove_small_objects(binary, min_size=max(1, min_area))
        binary = binary_closing(binary, disk(1))

        lbl = label(binary)
        lbl = _filter_by_morphology(lbl, min_area, max_area)
        props = _label_to_props(lbl, img_f)

        return SegmentationResult(
            method_name=f"Deep Learning  (thr={threshold:.2f})",
            params=dict(model=model_path, threshold=threshold,
                        tile_size=tile_size, overlap=overlap, tta=use_tta),
            label_image=lbl,
            n_objects=int(lbl.max()),
            properties=props,
            time_seconds=time.time() - t0
        )

    @staticmethod
    def _tiled_inference(model, img_f: np.ndarray,
                         device, tile_size: int, overlap: int,
                         use_tta: bool) -> np.ndarray:
        """
        Voert het model uit in overlappende tegels zodat ook grote beelden
        verwerkt kunnen worden zonder geheugen-overflow.
        """
        import torch
        H, W = img_f.shape
        prob = np.zeros((H, W), dtype=np.float32)
        count = np.zeros((H, W), dtype=np.float32)

        step = max(1, tile_size - overlap)
        ys = list(range(0, H, step))
        xs = list(range(0, W, step))

        with torch.no_grad():
            for y0 in ys:
                for x0 in xs:
                    y1 = min(y0 + tile_size, H)
                    x1 = min(x0 + tile_size, W)
                    patch = img_f[y0:y1, x0:x1]
                    ph, pw = patch.shape

                    # Pad tot tile_size als nodig
                    pad_h = tile_size - ph
                    pad_w = tile_size - pw
                    if pad_h > 0 or pad_w > 0:
                        patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode='reflect')

                    tensor = torch.from_numpy(patch[np.newaxis, np.newaxis]).to(device)
                    out = DeepLearningEngine._forward_with_tta(model, tensor, use_tta)

                    # Verwerk uitvoer naar kanskaart [0,1]
                    out_np = DeepLearningEngine._to_prob(out, ph, pw)
                    prob[y0:y1, x0:x1] += out_np
                    count[y0:y1, x0:x1] += 1.0

        count = np.maximum(count, 1e-8)
        return prob / count

    @staticmethod
    def _forward_with_tta(model, tensor, use_tta: bool):
        """Test-Time Augmentation: middel het resultaat van 4 rotaties."""
        import torch
        if not use_tta:
            return model(tensor)
        results = []
        for k in range(4):
            rot = torch.rot90(tensor, k, dims=[2, 3])
            out = model(rot)
            # Draai terug
            out_back = torch.rot90(out, -k, dims=[-2, -1])
            results.append(out_back)
        return torch.stack(results).mean(0)

    @staticmethod
    def _to_prob(out, ph: int, pw: int) -> np.ndarray:
        """Converteer model-uitvoer naar een kanskaart [0,1] van grootte (ph, pw)."""
        import torch
        if isinstance(out, (list, tuple)):
            out = out[0]
        out = out.squeeze()  # verwijder batch- en kanaal-dimensies
        # Sigmoid als de waarden niet al in [0,1] zitten
        if out.min() < 0 or out.max() > 1:
            out = torch.sigmoid(out)
        out_np = out.cpu().numpy().astype(np.float32)
        # Bij multi-class uitvoer: neem de eerste positieve klasse
        if out_np.ndim == 3:
            out_np = out_np[1] if out_np.shape[0] > 1 else out_np[0]
        return out_np[:ph, :pw]

    @staticmethod
    def _build_generic_unet(state_dict: dict, device) -> "nn.Module":
        """
        Bouw een generieke UNet die de state_dict kan laden.
        Detecteert het aantal kanalen automatisch uit de eerste Conv2d laag.
        """
        import torch.nn as nn

        # Detecteer invoer/uitvoer kanalen
        in_ch, out_ch = 1, 1
        for k, v in state_dict.items():
            if "weight" in k and v.ndim == 4:
                in_ch = int(v.shape[1])
                break
        for k, v in reversed(list(state_dict.items())):
            if "weight" in k and v.ndim == 4:
                out_ch = int(v.shape[0])
                break

        class DoubleConv(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),
                    nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),
                )
            def forward(self, x): return self.net(x)

        class GenericUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc1 = DoubleConv(in_ch, 64)
                self.enc2 = DoubleConv(64, 128)
                self.enc3 = DoubleConv(128, 256)
                self.enc4 = DoubleConv(256, 512)
                self.pool = nn.MaxPool2d(2)
                self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
                self.dec3 = DoubleConv(512, 256)
                self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
                self.dec2 = DoubleConv(256, 128)
                self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.dec1 = DoubleConv(128, 64)
                self.out_conv = nn.Conv2d(64, out_ch, 1)

            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                e4 = self.enc4(self.pool(e3))
                d3 = self.dec3(torch.cat([self.up3(e4), e3], 1))
                d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
                d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
                return self.out_conv(d1)

        model = GenericUNet().to(device)
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            # Probeer strict=False als de architectuur iets afwijkt
            try:
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                raise RuntimeError(
                    f"Kan state_dict niet laden in het generieke UNet-model.\n"
                    f"Sla het volledige model op met torch.save(model, path) "
                    f"in plaats van alleen de state_dict.\n\nFout: {e}"
                )
        return model


# ═══════════════════════════════════════════════════════════════════════════════
#  ENSEMBLE PREPROCESSOR  (zelfde pipeline als training notebook v7)
# ═══════════════════════════════════════════════════════════════════════════════
class EnsemblePreprocessor:
    """
    Bouwt een 3-kanaals tensor (norm, tophat, DoG) met ImageNet-normalisatie,
    identiek aan de trainingsscript in het notebook.
    """
    @staticmethod
    def pct_norm(img: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(img, [1.0, 99.9])
        if hi <= lo:
            return np.zeros_like(img, dtype=np.float32)
        return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)

    @staticmethod
    def build(img: np.ndarray) -> np.ndarray:
        """img: 2D float array → 3×H×W float32 tensor klaar voor het model."""
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


# ═══════════════════════════════════════════════════════════════════════════════
#  ENSEMBLE INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class EnsembleEngine:
    """
    Laadt alle model_fold*.pth bestanden uit een map en combineert ze via
    ensemble-gemiddelde (identiek aan notebook predict_new_image).
    """

    @staticmethod
    def _predict_tta(model, processed: np.ndarray, device,
                     tile: int = 256, overlap: int = 32) -> np.ndarray:
        """Tiled TTA-inferentie met Hanning-venster (8 augmentaties)."""
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

        # ── Device ─────────────────────────────────────────────────────────
        if device_name == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_name)

        model_files = sorted(Path(model_dir).glob("model_fold*.pth"))
        if not model_files:
            raise FileNotFoundError(
                f"Geen model_fold*.pth bestanden gevonden in:\n{model_dir}"
            )

        # ── Preprocessing (3-kanaals, identiek aan training) ───────────────
        processed = EnsemblePreprocessor.build(img)

        all_proba: List[np.ndarray] = []
        all_thr:   List[float]      = []

        for mf in model_files:
            ckpt = torch.load(str(mf), map_location=device, weights_only=False)
            encoder  = ckpt.get("encoder", "resnet34")
            version  = ckpt.get("version", "v7")
            # v9-modellen zijn getraind met scSE-attention; v8 ook; oudere niet
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

            if use_tta:
                proba = EnsembleEngine._predict_tta(
                    model, processed, device, tile=tile_sz
                )
            else:
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
            method_name=(
                f"Ensemble DL  ({len(model_files)} folds, thr={thr:.2f})"
            ),
            params=dict(model_dir=model_dir, n_models=len(model_files),
                        threshold=thr, avg_thr=avg_thr),
            label_image=lbl,
            n_objects=int(lbl.max()),
            properties=props,
            time_seconds=time.time() - t0
        )


# ─── Ensemble Worker ───────────────────────────────────────────────────────────
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


# ─── Deep Learning Worker ──────────────────────────────────────────────────────
class DLWorker(QThread):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, img, model_path, threshold, tile_size, overlap,
                 use_tta, min_area, max_area, device_name):
        super().__init__()
        self.img        = img
        self.model_path = model_path
        self.threshold  = threshold
        self.tile_size  = tile_size
        self.overlap    = overlap
        self.use_tta    = use_tta
        self.min_area   = min_area
        self.max_area   = max_area
        self.device     = device_name

    def run(self):
        try:
            result = DeepLearningEngine.run(
                self.img, self.model_path,
                threshold=self.threshold,
                tile_size=self.tile_size,
                overlap=self.overlap,
                use_tta=self.use_tta,
                min_area=self.min_area,
                max_area=self.max_area,
                device_name=self.device
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3b — CELLPOSE CELLICHAAM SEGMENTATIE  (worker)
# ═══════════════════════════════════════════════════════════════════════════════

class CellposeWorker(QThread):
    """Voert Cellpose-segmentatie uit in een aparte thread (non-blocking UI)."""

    finished = pyqtSignal(object)   # levert CellposeResult af
    progress = pyqtSignal(str)      # statusberichten
    error    = pyqtSignal(str)      # traceback bij fout

    def __init__(self, img: np.ndarray, kanaal: int,
                 model_type: str, diameter,
                 flow_threshold: float, cellprob_threshold: float,
                 z_count: int, parent=None):
        super().__init__(parent)
        self.img               = img
        self.kanaal            = kanaal
        self.model_type        = model_type
        self.diameter          = diameter          # None = auto
        self.flow_threshold    = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.z_count           = z_count

    def run(self):
        try:
            import torch
            from cellpose import models as cp_models
            from scipy.ndimage import binary_fill_holes
            import time

            use_gpu = True          # altijd GPU op lokale machine, niet CPU! Slechte AI tool
            t0 = time.perf_counter()

            self.progress.emit("⏳  Cellpose model laden…")
            # Gebruik models.Cellpose (identiek aan de notebook) zodat cyto2
            # altijd via het officiële pad wordt geladen en gecached.
            model = cp_models.CellposeModel(gpu=use_gpu, model_type=self.model_type)
            
            # ── Max-intensiteitsprojectie ─────────────────────────────────────
            # img is 2D of 3D (Z, Y, X) afhankelijk van wat PreprocessTab levert.
            # We slaan de MIP altijd intern op; de UI meldt welk kanaal is gebruikt.
            img2d = self.img
            if img2d.ndim == 3:
                # Interpreteer als (Z, Y, X) of (Y, X, C) — pak de max over axis 0
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

            # ── Gaten (celkernen) opvullen per cel ───────────────────────────
            self.progress.emit("⏳  Celkerngaten opvullen…")
            mask_filled = np.zeros_like(mask)
            for cel_id in range(1, int(mask.max()) + 1):
                cel = mask == cel_id
                mask_filled[binary_fill_holes(cel)] = cel_id
            mask = mask_filled

            elapsed = time.perf_counter() - t0
            n_cells = int(mask.max())

            result = {
                "mask":     mask,               # 2D int array: 0=bg, 1..N=cel-id
                "binary":   (mask > 0),         # bool masker voor aggregaatfilter
                "n_cells":  n_cells,
                "elapsed":  elapsed,
                "model":    self.model_type,
                "diameter": self.diameter,
            }
            self.finished.emit(result)

        except Exception:
            self.error.emit(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3b — CELLPOSE CELLICHAAM SEGMENTATIE  (UI)
# ═══════════════════════════════════════════════════════════════════════════════

class CellposeTab(QWidget):
    """
    Tab voor cellichaamdeterminatie met Cellpose.

    Werkwijze:
      1. Gebruiker configureert kanaal, model en parameters.
      2. 'Segmenteer' start een CellposeWorker (QThread) — UI blijft responsief.
      3. Het resulterende masker wordt opgeslagen in self.cell_mask (bool 2D).
      4. Volgende tabs kunnen self.cell_mask opvragen via get_cell_mask().
    """

    mask_ready = pyqtSignal(object)   # emitteert cell_mask (np.ndarray bool)

    def __init__(self, preprocess_tab: "PreprocessTab", viewer_tab=None, parent=None):
        super().__init__(parent)
        self.prep        = preprocess_tab
        self.viewer      = viewer_tab  # originele afbeelding (tab 3 gebruikt origineel)
        self.cell_mask: Optional[np.ndarray] = None   # bool 2D — None = nog niet gesegmenteerd
        self._worker: Optional[CellposeWorker] = None
        self._build_ui()

    # ── Publieke API ───────────────────────────────────────────────────────────
    def get_cell_mask(self) -> Optional[np.ndarray]:
        """Geeft het binaire celmasker terug (True = binnen cellichaam).
        Geeft None als nog geen segmentatie is uitgevoerd."""
        return self.cell_mask

    def _get_image(self) -> Optional[np.ndarray]:
        """Tab 3 gebruikt altijd de originele (onbewerkte) afbeelding."""
        if self.viewer is not None:
            return self.viewer.get_current_image()
        return self.prep.get_image()

    # ── UI opbouw ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setSpacing(6)

        # ── Linker paneel (scroll) ─────────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(340)
        left_inner = QWidget()
        lv = QVBoxLayout(left_inner)
        lv.setSpacing(10)

        # Waarschuwing als Cellpose niet beschikbaar is
        if not HAS_CELLPOSE:
            warn = QLabel(
                "⚠  Cellpose is niet geïnstalleerd.\n"
                "Installeer met:\n"
                "  pip install cellpose"
            )
            warn.setStyleSheet("color:#ff8800; font-weight:bold; padding:10px;")
            warn.setWordWrap(True)
            lv.addWidget(warn)

        # ── Kanaal & model ─────────────────────────────────────────────────
        grp_model = QGroupBox("🔬  Model & Kanaal")
        mf = QFormLayout(grp_model)

        self.spn_kanaal = QSpinBox()
        self.spn_kanaal.setRange(0, 15)
        self.spn_kanaal.setValue(2)
        self.spn_kanaal.setToolTip(
            "Het kanaal (0-gebaseerd) waarop Cellpose de cellichamen detecteert.\n"
            "Kanaal 0 = eerste kanaal in het .lif bestand.\n"
            "Standaard: kanaal 2 (derde kanaal)."
        )
        mf.addRow("Segmentatie-kanaal:", self.spn_kanaal)

        self.cmb_model = QComboBox()
        self.cmb_model.addItems(["cyto2", "cyto", "nuclei", "cyto3"])
        self.cmb_model.setCurrentText("cyto2")
        self.cmb_model.setToolTip(
            "cyto2  — verbeterd model voor cellichamen (aanbevolen)\n"
            "cyto   — origineel cellichaammodel\n"
            "nuclei — voor DAPI/kernkleuring\n"
            "cyto3  — meest recente generatie (experimenteel)"
        )
        mf.addRow("Model:", self.cmb_model)

        self.spn_diameter = QSpinBox()
        self.spn_diameter.setRange(0, 999)
        self.spn_diameter.setValue(80)
        self.spn_diameter.setSpecialValueText("Auto")
        self.spn_diameter.setToolTip(
            "Geschatte celdiameter in pixels.\n"
            "80 = standaard (overeenkomstig notebook-instelling).\n"
            "0 = automatisch schatten (langzamer maar robuust).\n"
            "Meet een representatieve cel in Fiji voor een exacte waarde."
        )
        mf.addRow("Celdiameter (px, 0=auto):", self.spn_diameter)

        lv.addWidget(grp_model)

        # ── Drempelwaarden ─────────────────────────────────────────────────
        grp_thr = QGroupBox("⚙  Segmentatie-parameters")
        tf = QFormLayout(grp_thr)

        self.spn_flow = QDoubleSpinBox()
        self.spn_flow.setRange(0.1, 1.0)
        self.spn_flow.setSingleStep(0.05)
        self.spn_flow.setValue(0.8)
        self.spn_flow.setToolTip(
            "Flow threshold — bepaalt hoeveel van de celrand wordt meegenomen.\n"
            "Hoger (0.8–1.0) → meer uitlopers en dunne cytoplasmaranden.\n"
            "Lager (0.3–0.5) → alleen duidelijke, compacte celkernen.\n"
            "Standaard: 0.8"
        )
        tf.addRow("Flow threshold:", self.spn_flow)

        self.spn_cellprob = QDoubleSpinBox()
        self.spn_cellprob.setRange(-8.0, 6.0)
        self.spn_cellprob.setSingleStep(0.5)
        self.spn_cellprob.setValue(-4.0)
        self.spn_cellprob.setToolTip(
            "Celkans-drempel — hoe zwak een pixel nog als 'cel' wordt beschouwd.\n"
            "Lager (−4 tot −8) → ook lichtgrijze cytoplasmaranden worden meegenomen.\n"
            "Hoger (0 tot +6)  → alleen hoog-intensieve celgebieden.\n"
            "Standaard: −4.0"
        )
        tf.addRow("Cellprob threshold:", self.spn_cellprob)

        # Reset-knop naar standaardwaarden
        btn_reset = QPushButton("↺  Herstel standaardwaarden")
        btn_reset.setToolTip("Zet flow=0.8 en cellprob=−4.0 terug.")
        btn_reset.clicked.connect(self._reset_params)
        tf.addRow(btn_reset)

        lv.addWidget(grp_thr)

        # ── Uitleg celmasker ───────────────────────────────────────────────
        grp_info = QGroupBox("ℹ  Werkwijze")
        iv = QVBoxLayout(grp_info)
        lbl_info = QLabel(
            "Cellpose berekent een <b>max-intensiteitsprojectie</b> over alle "
            "Z-slices op het gekozen kanaal en segmenteert daarop de cellichamen.\n\n"
            "Het resulterende binaire masker wordt gebruikt in de <b>Segmentatie-tab</b> "
            "om aggregaten die buiten cellichamen vallen automatisch te verwijderen."
        )
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet("color:#8b949e; font-size:11px; padding:4px;")
        iv.addWidget(lbl_info)
        lv.addWidget(grp_info)

        # ── Run-knop & voortgang ───────────────────────────────────────────
        self.btn_run = QPushButton("▶  Cellichamen segmenteren")
        self.btn_run.setObjectName("primary")
        self.btn_run.setEnabled(HAS_CELLPOSE)
        self.btn_run.clicked.connect(self._run)
        lv.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)          # indeterminate spinner
        self.progress.setVisible(False)
        lv.addWidget(self.progress)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color:#58a6ff; font-size:11px;")
        self.lbl_status.setWordWrap(True)
        lv.addWidget(self.lbl_status)

        # ── Statistieken ───────────────────────────────────────────────────
        grp_stats = QGroupBox("📊  Resultaat")
        sv = QVBoxLayout(grp_stats)
        self.txt_stats = QTextEdit()
        self.txt_stats.setReadOnly(True)
        self.txt_stats.setMinimumHeight(80)
        self.txt_stats.setMaximumHeight(140)
        sv.addWidget(self.txt_stats)
        lv.addWidget(grp_stats)

        # ── Masker exporteren ──────────────────────────────────────────────
        btn_export = QPushButton("💾  Exporteer celmasker als TIFF")
        btn_export.setToolTip(
            "Slaat het 16-bit labelmasker op (elke cel een uniek ID).\n"
            "Bruikbaar als referentiemasker in andere programma's."
        )
        btn_export.clicked.connect(self._export_mask)
        lv.addWidget(btn_export)

        lv.addStretch()
        left_inner.setLayout(lv)
        left_scroll.setWidget(left_inner)
        main.addWidget(left_scroll)

        # ── Rechter canvas ─────────────────────────────────────────────────
        right_w = QWidget()
        rv = QVBoxLayout(right_w)
        rv.setSpacing(4)
        self.canvas = MplCanvas(width=11, height=9)
        nav = NavigationToolbar(self.canvas, self)
        nav.setStyleSheet(
            "background:#f0f3f6; border-bottom:1px solid #d0d7de; border-radius:6px 6px 0 0;"
        )
        rv.addWidget(nav)
        rv.addWidget(self.canvas)
        main.addWidget(right_w)

    # ── Slots ──────────────────────────────────────────────────────────────────
    def _reset_params(self):
        self.spn_flow.setValue(0.8)
        self.spn_cellprob.setValue(-4.0)
        self.spn_diameter.setValue(80)

    def _run(self):
        img = self._get_image()
        if img is None:
            QMessageBox.warning(
                self, "Geen beeld",
                "Laad eerst een beeld in de Viewer-tab en verwerk het in Pre-processing."
            )
            return

        diameter = self.spn_diameter.value() or None   # 0 → None (auto)

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

        # Origineel beeld voor weergave (altijd 2D max-proj)
        img2d = img.max(axis=0) if img.ndim == 3 else img

        # Normaliseer voor weergave
        lo, hi = np.percentile(img2d, [1, 99.9])
        img_norm = np.clip((img2d.astype(float) - lo) / max(hi - lo, 1e-6), 0, 1)

        # Cellichamen uitknippen: buiten masker = zwart
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
        # Sla het labelmasker op als uint16 (0=bg, 1..N=cel-id)
        # Voor het binaire masker: (cell_mask * 255).astype(uint8)
        tifffile.imwrite(path, self.cell_mask.astype(np.uint8) * 255)
        QMessageBox.information(self, "Opgeslagen", f"Celmasker opgeslagen:\n{path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — DEEP LEARNING SEGMENTATIE
# ═══════════════════════════════════════════════════════════════════════════════
class DeepLearningTab(QWidget):
    result_ready = pyqtSignal(object)

    def __init__(self, preprocess_tab: "PreprocessTab", viewer_tab=None):
        super().__init__()
        self.prep                   = preprocess_tab
        self.viewer                 = viewer_tab  # originele afbeelding (tab 5 gebruikt origineel)
        self.current_result: Optional[SegmentationResult] = None
        self.worker: Optional[DLWorker] = None
        self._ensemble_worker: Optional[EnsembleDLWorker] = None
        self._model_path            = ""
        self._model_dir             = ""
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _get_image(self) -> Optional[np.ndarray]:
        """Tab 5 gebruikt altijd de originele (onbewerkte) afbeelding."""
        if self.viewer is not None:
            return self.viewer.get_current_image()
        return self.prep.get_image()

    def _build_ui(self):
        main = QHBoxLayout(self)
        main.setSpacing(6)

        # ── Linker paneel ──────────────────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(360)
        left_inner  = QWidget()
        lv = QVBoxLayout(left_inner)
        lv.setSpacing(8)

        # Waarschuwing als torch niet beschikbaar
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

        # ── Ensemble modellenmap ───────────────────────────────────────────
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
        ens_thr2_row.addWidget(self.spn_ens_threshold)
        ev.addLayout(ens_thr2_row)

        self.chk_ens_tta = QCheckBox("Test-Time Augmentation (TTA, 8×)")
        self.chk_ens_tta.setChecked(True)
        self.chk_ens_tta.setToolTip(
            "Middelt 8 augmentaties (4 rotaties + 2 spiegelingen).\n"
            "Verbetert kwaliteit maar is ~8× langzamer."
        )
        ev.addWidget(self.chk_ens_tta)

        self.btn_run_ensemble = QPushButton("▶  Ensemble Segmentatie uitvoeren")
        self.btn_run_ensemble.setObjectName("primary")
        self.btn_run_ensemble.clicked.connect(self._run_ensemble)
        ev.addWidget(self.btn_run_ensemble)

        lv.addWidget(grp_ens)

        # ── Enkel model laden (enkelvoudig .pth) ───────────────────────────
        grp_model = QGroupBox("🤖  Deep Learning Model (.pth)")
        mv = QVBoxLayout(grp_model)

        path_row = QHBoxLayout()
        self.lbl_model_path = QLabel("Geen model geladen")
        self.lbl_model_path.setStyleSheet("color:#8b949e; font-size:11px;")
        self.lbl_model_path.setWordWrap(True)
        path_row.addWidget(self.lbl_model_path, stretch=1)
        btn_load = QPushButton("📂  Laad .pth")
        btn_load.clicked.connect(self._load_model)
        path_row.addWidget(btn_load)
        mv.addLayout(path_row)

        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Device:"))
        self.cmb_device = QComboBox()
        self.cmb_device.addItems(["auto", "cpu", "cuda", "mps"])
        self.cmb_device.setToolTip(
            "auto = CUDA als beschikbaar, anders CPU\n"
            "mps  = Apple Silicon (M1/M2)"
        )
        device_row.addWidget(self.cmb_device)
        mv.addLayout(device_row)

        lv.addWidget(grp_model)

        # ── Inferentie-parameters ──────────────────────────────────────────
        grp_inf = QGroupBox("⚙  Inferentie-parameters")
        fv = QFormLayout(grp_inf)

        self.spn_threshold = QDoubleSpinBox()
        self.spn_threshold.setRange(0.01, 0.99)
        self.spn_threshold.setValue(0.5)
        self.spn_threshold.setSingleStep(0.05)
        self.spn_threshold.setToolTip(
            "Kansdrempel voor segmentatie.\n"
            "Lager (0.3) = meer detecties, hogere gevoeligheid.\n"
            "Hoger (0.7) = minder maar zekerder detecties."
        )
        fv.addRow("Drempelwaarde (threshold):", self.spn_threshold)

        self.spn_tile = QSpinBox()
        self.spn_tile.setRange(64, 2048)
        self.spn_tile.setValue(512)
        self.spn_tile.setSingleStep(64)
        self.spn_tile.setToolTip(
            "Tegelgrootte in pixels.\n"
            "Groter = sneller maar meer geheugen.\n"
            "Verklein bij CUDA out-of-memory fouten."
        )
        fv.addRow("Tegelgrootte (tile size, px):", self.spn_tile)

        self.spn_overlap = QSpinBox()
        self.spn_overlap.setRange(0, 256)
        self.spn_overlap.setValue(64)
        self.spn_overlap.setSingleStep(16)
        self.spn_overlap.setToolTip(
            "Overlap tussen tegels in pixels.\n"
            "Groter = minder artefacten aan tegelranden.\n"
            "Aanbevolen: 10–15% van de tegelgrootte."
        )
        fv.addRow("Overlap (px):", self.spn_overlap)

        self.chk_tta = QCheckBox("Test-Time Augmentation (TTA)")
        self.chk_tta.setChecked(False)
        self.chk_tta.setToolTip(
            "Middelt het resultaat van 4 rotaties.\n"
            "Verbetert kwaliteit maar is 4× langzamer."
        )
        fv.addRow(self.chk_tta)

        lv.addWidget(grp_inf)

        # ── Post-processing ────────────────────────────────────────────────
        grp_post = QGroupBox("🔧  Post-processing")
        pf = QFormLayout(grp_post)

        self.spn_min_area = QSpinBox()
        self.spn_min_area.setRange(1, 9999)
        self.spn_min_area.setValue(5)
        pf.addRow("Min oppervlak (px²):", self.spn_min_area)

        self.spn_max_area = QSpinBox()
        self.spn_max_area.setRange(1, 999999)
        self.spn_max_area.setValue(50000)
        pf.addRow("Max oppervlak (px²):", self.spn_max_area)

        lv.addWidget(grp_post)

        # ── Overlay-opties ─────────────────────────────────────────────────
        grp_ov = QGroupBox("🖍  Overlay-opties")
        ovf = QFormLayout(grp_ov)
        self.chk_show_circles = QCheckBox("Teken contouren")
        self.chk_show_circles.setChecked(True)
        self.chk_show_numbers = QCheckBox("Toon nummers")
        self.chk_show_numbers.setChecked(True)
        self.chk_show_fill    = QCheckBox("Gevuld gebied")
        self.chk_show_fill.setChecked(True)
        self.cmb_circle_color = QComboBox()
        self.cmb_circle_color.addItems(["#00ffcc","#ff4466","#ffff00","#ffffff","#00aaff","#ff8800"])
        self.spn_circle_lw  = QDoubleSpinBox()
        self.spn_circle_lw.setRange(0.3, 5); self.spn_circle_lw.setValue(1.2)
        self.spn_font_size  = QDoubleSpinBox()
        self.spn_font_size.setRange(3, 16); self.spn_font_size.setValue(6.5)
        self.cmb_cmap       = QComboBox()
        self.cmb_cmap.addItems(["hot","gray","inferno","magma","viridis","plasma"])
        ovf.addWidget(self.chk_show_circles)
        ovf.addWidget(self.chk_show_numbers)
        ovf.addWidget(self.chk_show_fill)
        ovf.addRow("Kleur:",         self.cmb_circle_color)
        ovf.addRow("Lijnbreedte:",   self.spn_circle_lw)
        ovf.addRow("Lettergrootte:", self.spn_font_size)
        ovf.addRow("Achtergrond:",   self.cmb_cmap)
        lv.addWidget(grp_ov)

        # ── Uitvoer-knoppen ────────────────────────────────────────────────
        self.btn_run = QPushButton("▶  DL Segmentatie uitvoeren")
        self.btn_run.setObjectName("primary")
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

        grp_stats = QGroupBox("📊  Statistieken")
        sv = QVBoxLayout(grp_stats)
        self.txt_stats = QTextEdit()
        self.txt_stats.setReadOnly(True)
        self.txt_stats.setMinimumHeight(90)
        sv.addWidget(self.txt_stats)
        lv.addWidget(grp_stats)

        btn_csv = QPushButton("💾  Exporteer CSV")
        btn_csv.clicked.connect(self._export_csv)
        lv.addWidget(btn_csv)

        btn_img = QPushButton("🖼  Exporteer geannoteerd beeld")
        btn_img.clicked.connect(self._export_image)
        lv.addWidget(btn_img)

        lv.addStretch()
        left_inner.setLayout(lv)
        left_scroll.setWidget(left_inner)
        main.addWidget(left_scroll)

        # ── Rechter canvas ─────────────────────────────────────────────────
        right_w = QWidget()
        rv = QVBoxLayout(right_w)
        rv.setSpacing(4)
        self.canvas = MplCanvas(width=11, height=9)
        nav = NavigationToolbar(self.canvas, self)
        nav.setStyleSheet("background:#f0f3f6; border-bottom: 1px solid #d0d7de; border-radius: 6px 6px 0 0;")
        rv.addWidget(nav)
        rv.addWidget(self.canvas)
        main.addWidget(right_w)

    # ── Acties ─────────────────────────────────────────────────────────────────
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

        # Lees gemiddelde threshold uit de checkpoints
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

    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open PyTorch model", "",
            "PyTorch model (*.pth *.pt);;Alle bestanden (*)"
        )
        if not path:
            return
            
        self._model_path = path
        
        # --- NIEUW: Auto-load de perfecte threshold uit het v7 bestand! ---
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if "best_thr" in ckpt:
                opt_thr = ckpt["best_thr"]
                self.spn_threshold.setValue(opt_thr)
                
                # Update tile_size indien deze is opgeslagen (meestal 256)
                if "tile" in ckpt:
                    self.spn_tile.setValue(ckpt["tile"])
                    
                self.lbl_status.setText(f"Model geladen! Optimale drempel (thr={opt_thr:.2f}) automatisch ingesteld.")
            else:
                self.lbl_status.setText(f"Model geladen: {Path(path).name}")
        except Exception as e:
            self.lbl_status.setText(f"Model geselecteerd: {Path(path).name}")
            print(f"Kon metadata niet lezen: {e}")
        # ------------------------------------------------------------------

        self.lbl_model_path.setText(f"✅  {Path(path).name}")

    def _run(self):
        if not self._model_path:
            QMessageBox.warning(self, "Geen model", "Laad eerst een .pth modelbestand.")
            return
        img = self._get_image()
        if img is None:
            QMessageBox.warning(self, "Geen beeld", "Laad en verwerk eerst een beeld.")
            return
        if not HAS_TORCH:
            QMessageBox.critical(
                self, "PyTorch ontbreekt",
                "Installeer PyTorch met:\n  pip install torch torchvision"
            )
            return

        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.lbl_status.setText("Bezig met inferentie…")

        self.worker = DLWorker(
            img=img,
            model_path=self._model_path,
            threshold=self.spn_threshold.value(),
            tile_size=self.spn_tile.value(),
            overlap=self.spn_overlap.value(),
            use_tta=self.chk_tta.isChecked(),
            min_area=self.spn_min_area.value(),
            max_area=self.spn_max_area.value(),
            device_name=self.cmb_device.currentText()
        )
        self.worker.finished.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_result(self, result: SegmentationResult):
        self.current_result = result
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self.lbl_status.setText(
            f"Klaar: {result.n_objects} objecten gedetecteerd  |  {result.time_seconds:.2f}s"
        )
        self._display_result(result)
        self._show_stats(result)
        self.result_ready.emit(result)

    def _on_error(self, err: str):
        self.btn_run.setEnabled(True)
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
#  TAB 5 — VALIDATIE
# ═══════════════════════════════════════════════════════════════════════════════

# ── Hulp-canvas: 3 subplots naast elkaar ──────────────────────────────────────
class TripleCanvas(FigureCanvas):
    """Matplotlib canvas met 3 subplots: [Ground Truth | Tab 3 | Tab 4]."""

    OUTER_BG = "#161b22"
    INNER_BG = "#0d1117"
    TICK_CLR = "#c9d1d9"
    GRID_CLR = "#30363d"
    TITLES   = ["🏁  Ground Truth", "🔬  Tab 3 — Traditioneel", "🧠  Tab 4 — Deep Learning"]

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(14, 5), dpi=100, facecolor=self.OUTER_BG)
        self.axes = self.fig.subplots(1, 3)
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

    def update_plots(self, gt_mask, seg_mask, dl_mask):
        """
        Teken drie subplots:
          ax[0] → Ground-truth masker (groen)
          ax[1] → Tab 3 masker met TP/FP/FN overlay
          ax[2] → Tab 4 masker met TP/FP/FN overlay
        """
        self.fig.clf()
        axes = self.fig.subplots(1, 3)

        def _mask_rgb(pred, gt):
            if pred is None:
                return None
            p = pred.astype(bool)
            g = gt.astype(bool)
            rgb = np.zeros((*g.shape, 3), dtype=np.uint8)
            rgb[g & p]  = [0, 200, 80]    # TP — groen
            rgb[p & ~g] = [220, 50, 50]   # FP — rood
            rgb[g & ~p] = [50, 100, 220]  # FN — blauw
            return rgb

        gt_rgb = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        gt_rgb[gt_mask.astype(bool)] = [0, 200, 80]

        overlays   = [gt_rgb, _mask_rgb(seg_mask, gt_mask), _mask_rgb(dl_mask, gt_mask)]
        subtitles  = ["Ground Truth",
                      "TP=groen  FP=rood  FN=blauw",
                      "TP=groen  FP=rood  FN=blauw"]

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


# ── Metriekfuncties ────────────────────────────────────────────────────────────
def _compute_validation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Berekent pixel-niveau F1-score (Dice) en IoU (Jaccard) voor binaire maskers."""
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
    """Zet een SegmentationResult om naar een binair masker."""
    if result is None:
        return None
    if result.label_image is not None:
        return (result.label_image > 0).astype(np.uint8)
    return None


def _load_gt_mask_from_file(path: str) -> np.ndarray:
    """Laad een TIFF of PNG masker (2D of 3D) als binair masker."""
    img = tifffile.imread(path)
    if img.ndim == 3:
        img = img.max(axis=0)
    if img.ndim == 4:
        img = img.max(axis=(0, 1))
    return (img > 0).astype(np.uint8)


class ValidationTab(QWidget):
    """
    Tab 5 — Validatie
    =================
    Vergelijkt segmentaties van Tab 3 en Tab 4 met een geladen ground-truth masker.

    Parameters
    ----------
    seg_tab : SegmentationTab   — referentie naar Tab 3
    dl_tab  : DeepLearningTab   — referentie naar Tab 4
    """

    def __init__(self, seg_tab, dl_tab, parent=None):
        super().__init__(parent)
        self.seg_tab  = seg_tab
        self.dl_tab   = dl_tab
        self.gt_mask  = None
        self.gt_path  = ""
        self._last_metrics = None
        self._build_ui()

    # ── UI opbouw ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(10, 10, 10, 10)

        # ── Acties ────────────────────────────────────────────────────────
        grp_actions = QGroupBox("⚙️  Acties")
        av = QHBoxLayout(grp_actions)
        av.setSpacing(8)

        self.btn_load_gt = QPushButton("📂  Laad Ground-Truth Masker")
        self.btn_load_gt.setObjectName("primary")
        self.btn_load_gt.setToolTip(
            "Laad een binair masker (.tif / .png) als ground-truth referentie.\n"
            "Witte pixels = foreground (aggregaten), zwarte pixels = achtergrond."
        )
        self.btn_load_gt.clicked.connect(self._load_gt_mask)
        av.addWidget(self.btn_load_gt)

        self.btn_compare = QPushButton("📊  Vergelijk Methodes")
        self.btn_compare.setToolTip(
            "Vergelijk Tab 3 en Tab 4 met het ground-truth masker.\n"
            "Zorg dat beide tabs al gesegmenteerd zijn."
        )
        self.btn_compare.setEnabled(False)
        self.btn_compare.clicked.connect(self._run_comparison)
        av.addWidget(self.btn_compare)

        self.btn_export = QPushButton("💾  Exporteer Rapport (CSV)")
        self.btn_export.setToolTip("Sla de validatiemetrieken op als CSV-bestand.")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_csv)
        av.addWidget(self.btn_export)

        av.addStretch()
        root.addWidget(grp_actions)

        # ── Status ────────────────────────────────────────────────────────
        self.lbl_status = QLabel("Stap 1: Laad een ground-truth masker om te starten.")
        self.lbl_status.setStyleSheet(
            "color: #8b949e; font-size: 11px; padding: 4px 8px;"
            "background: #161b22; border-radius: 4px; border: 1px solid #21262d;"
        )
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        # ── Metriekentabel ────────────────────────────────────────────────
        grp_metrics = QGroupBox("📈  Validatiemetrieken")
        mv = QVBoxLayout(grp_metrics)

        self.tbl_metrics = QTableWidget(6, 3)
        self.tbl_metrics.setHorizontalHeaderLabels(
            ["Metriek", "Tab 3 — Traditioneel", "Tab 4 — Deep Learning"]
        )
        self.tbl_metrics.verticalHeader().setVisible(False)
        self.tbl_metrics.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl_metrics.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl_metrics.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.tbl_metrics.setAlternatingRowColors(True)
        self.tbl_metrics.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_metrics.setMaximumHeight(230)
        self._populate_empty_table()
        mv.addWidget(self.tbl_metrics)
        root.addWidget(grp_metrics)

        # ── Visuele vergelijking ──────────────────────────────────────────
        grp_visual = QGroupBox("🖼️  Visuele Vergelijking  —  Ground Truth | Tab 3 | Tab 4")
        vv = QVBoxLayout(grp_visual)

        self.canvas = TripleCanvas()
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

    # ── Tabel-helpers ──────────────────────────────────────────────────────────
    def _populate_empty_table(self):
        rows = [
            ("F1-score (Dice)",  "—", "—"),
            ("IoU (Jaccard)",    "—", "—"),
            ("Precisie",         "—", "—"),
            ("Recall",           "—", "—"),
            ("True Positives",   "—", "—"),
            ("False Positives",  "—", "—"),
        ]
        self._fill_table(rows, highlight=False)

    def _fill_table(self, rows, highlight=True):
        SCORE_ROWS = {0, 1, 2, 3}  # rijen waar hoger = beter

        for i, (metric, val_seg, val_dl) in enumerate(rows):
            item_metric = QTableWidgetItem(metric)
            item_metric.setFont(QFont("Segoe UI", 10, QFont.Bold))
            item_metric.setForeground(QColor("#79c0ff"))
            self.tbl_metrics.setItem(i, 0, item_metric)

            for col, val in enumerate([val_seg, val_dl], start=1):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                item.setFont(QFont("Cascadia Code", 10))

                if highlight and i in SCORE_ROWS and val not in ("—", "N/A"):
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

                self.tbl_metrics.setItem(i, col, item)

        self.tbl_metrics.resizeRowsToContents()

    # ── Acties ─────────────────────────────────────────────────────────────────
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
                "Het geladen masker bevat geen foreground-pixels.\n"
                "Controleer of het juiste bestand is geselecteerd."
            )
            return

        self.gt_mask = mask
        self.gt_path = path
        n_pos = int(mask.sum())
        pct   = 100.0 * n_pos / mask.size

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

        seg_mask = _result_to_binary_mask(getattr(self.seg_tab, "current_result", None))
        dl_mask  = _result_to_binary_mask(getattr(self.dl_tab,  "current_result", None))

        missing = []
        if seg_mask is None:
            missing.append("Tab 3 (Segmentatie)")
        if dl_mask is None:
            missing.append("Tab 4 (Deep Learning)")

        if len(missing) == 2:
            QMessageBox.warning(
                self, "Geen segmentaties",
                "Voer eerst segmentatie uit in Tab 3 en/of Tab 4."
            )
            return

        if missing:
            reply = QMessageBox.question(
                self, "Ontbrekende segmentatie",
                f"{', '.join(missing)} heeft nog geen resultaat.\n"
                "Doorgaan met de beschikbare resultaten?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        gt = self.gt_mask

        def _resize_if_needed(mask, target):
            if mask is None or mask.shape == target:
                return mask
            from skimage.transform import resize as sk_resize
            r = sk_resize(mask.astype(float), target,
                          order=0, anti_aliasing=False, preserve_range=True)
            return (r > 0.5).astype(np.uint8)

        seg_mask = _resize_if_needed(seg_mask, gt.shape)
        dl_mask  = _resize_if_needed(dl_mask,  gt.shape)

        def _metrics_or_na(mask):
            if mask is None:
                return {k: None for k in ("f1", "iou", "precision", "recall", "tp", "fp")}
            return _compute_validation_metrics(mask, gt)

        m_seg = _metrics_or_na(seg_mask)
        m_dl  = _metrics_or_na(dl_mask)

        def _v(m, key):
            v = m.get(key)
            if v is None:
                return "N/A"
            return f"{v:.4f}" if isinstance(v, float) else f"{v:,}"

        rows = [
            ("F1-score (Dice)",  _v(m_seg, "f1"),        _v(m_dl, "f1")),
            ("IoU (Jaccard)",    _v(m_seg, "iou"),       _v(m_dl, "iou")),
            ("Precisie",         _v(m_seg, "precision"), _v(m_dl, "precision")),
            ("Recall",           _v(m_seg, "recall"),    _v(m_dl, "recall")),
            ("True Positives",   _v(m_seg, "tp"),        _v(m_dl, "tp")),
            ("False Positives",  _v(m_seg, "fp"),        _v(m_dl, "fp")),
        ]
        self._fill_table(rows, highlight=True)

        self.canvas.update_plots(gt, seg_mask, dl_mask)

        f1_seg = m_seg.get("f1")
        f1_dl  = m_dl.get("f1")
        if f1_seg is not None and f1_dl is not None:
            if abs(f1_seg - f1_dl) < 0.005:
                verdict = "Tab 3 en Tab 4 presteren vrijwel gelijk."
            elif f1_seg > f1_dl:
                verdict = f"Tab 3 (Traditioneel) presteert beter  (ΔF1 = {f1_seg - f1_dl:+.4f})."
            else:
                verdict = f"Tab 4 (Deep Learning) presteert beter  (ΔF1 = {f1_dl - f1_seg:+.4f})."
        elif f1_seg is not None:
            verdict = f"Alleen Tab 3 beschikbaar — F1 = {f1_seg:.4f}."
        elif f1_dl is not None:
            verdict = f"Alleen Tab 4 beschikbaar — F1 = {f1_dl:.4f}."
        else:
            verdict = "Geen geldige metrieken beschikbaar."

        self.lbl_status.setText(f"📊  Vergelijking voltooid  |  {verdict}")
        self.lbl_status.setStyleSheet(
            "color: #79c0ff; font-size: 11px; padding: 4px 8px;"
            "background: #0d1b2b; border-radius: 4px; border: 1px solid #1f6feb;"
        )
        self._last_metrics = {"seg": m_seg, "dl": m_dl}
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
        m_seg = self._last_metrics["seg"]
        m_dl  = self._last_metrics["dl"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Metriek", "Tab3_Traditioneel", "Tab4_DeepLearning"])
            w.writerow(["Ground-truth masker", self.gt_path, ""])
            w.writerow([])
            for key, label_txt in zip(keys, labels):
                sv = m_seg.get(key)
                dv = m_dl.get(key)
                w.writerow([
                    label_txt,
                    f"{sv:.6f}" if isinstance(sv, float) else (str(sv) if sv is not None else "N/A"),
                    f"{dv:.6f}" if isinstance(dv, float) else (str(dv) if dv is not None else "N/A"),
                ])
        QMessageBox.information(self, "Geëxporteerd", f"Rapport opgeslagen:\n{path}")

    # ── Optionele slots voor live updates via signals ───────────────────────────
    def _on_seg_updated(self, result):
        self.lbl_status.setText(
            f"ℹ️  Tab 3 bijgewerkt: {result.method_name} ({result.n_objects} objecten). "
            "Klik 'Vergelijk Methodes' om te valideren."
        )

    def _on_dl_updated(self, result):
        self.lbl_status.setText(
            f"ℹ️  Tab 4 bijgewerkt: {result.method_name} ({result.n_objects} objecten). "
            "Klik 'Vergelijk Methodes' om te valideren."
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  HOOFDVENSTER
# ═══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Protein Aggregate Analyzer v4.2 — Puncta Detector Pro + Deep Learning")
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
        self.seg_tab      = SegmentationTab(self.prep_tab)
        self.dl_tab       = DeepLearningTab(self.prep_tab, viewer_tab=self.viewer_tab)
        self.val_tab      = ValidationTab(self.seg_tab, self.dl_tab)

        self.tabs.addTab(self.viewer_tab,   "🔭  1. Viewer")
        self.tabs.addTab(self.prep_tab,     "🔧  2. Pre-processing")
        self.tabs.addTab(self.cellpose_tab, "🔬  3. Cellichamen")
        self.tabs.addTab(self.seg_tab,      "🔍  4. Segmentatie")
        self.tabs.addTab(self.dl_tab,       "🧠  5. Deep Learning")
        self.tabs.addTab(self.val_tab,      "📊  6. Validatie")

    def _wire_signals(self):
        self.viewer_tab.stack_loaded.connect(
            lambda s: self.status.showMessage(
                f"Geladen: {s.name}  |  Z={s.z_count}, C={s.channel_count}, "
                f"{s.height}×{s.width} px"
            )
        )
        self.seg_tab.result_ready.connect(
            lambda r: self.status.showMessage(
                f"Segmentatie klaar: {r.method_name}  |  "
                f"N={r.n_objects} objecten  |  {r.time_seconds:.2f}s"
            )
        )
        self.dl_tab.result_ready.connect(
            lambda r: self.status.showMessage(
                f"DL Segmentatie klaar: {r.method_name}  |  "
                f"N={r.n_objects} objecten  |  {r.time_seconds:.2f}s"
            )
        )
        self.seg_tab.result_ready.connect(self.val_tab._on_seg_updated)
        self.dl_tab.result_ready.connect(self.val_tab._on_dl_updated)

    def _open_file(self):
        self.tabs.setCurrentIndex(0)
        self.viewer_tab._open_file()

    def _save_result(self):
        # Gebruik het resultaat van de actieve segmentatie-tab
        result = self.dl_tab.current_result if self.tabs.currentIndex() == 4 else self.seg_tab.current_result
        if result is None or result.label_image is None:
            QMessageBox.warning(self, "Geen resultaat", "Voer eerst segmentatie uit.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Sla labelbeeld op", "", "TIFF (*.tif);;PNG (*.png)"
        )
        if path:
            tifffile.imwrite(path, result.label_image.astype(np.uint16))
            QMessageBox.information(self, "Opgeslagen", f"Labelbeeld opgeslagen:\n{path}")

    def _show_stats_dialog(self):
        result = self.dl_tab.current_result if self.tabs.currentIndex() == 4 else self.seg_tab.current_result
        if result is None:
            QMessageBox.information(self, "Geen resultaat", "Voer eerst segmentatie uit.")
            return
        dlg = StatsDialog(result, self)
        dlg.exec_()

    def _show_help(self):
        txt = (
            "<h2>Protein Aggregate Analyzer v2.0 — Help</h2>"
            "<h3>Nieuw in v2.0</h3>"
            "<ul>"
            "<li><b>Hot-colormap decoder</b>: TIF-bestanden die opgeslagen zijn met "
            "hot-kleurpalet (RGB) worden automatisch gedecodeerd naar grijswaarden.</li>"
            "<li><b>Puncta Detector Pro ★★</b>: Nieuwe aanbevolen methode, specifiek "
            "ontworpen voor confocale eiwit-aggregaat detectie.</li>"
            "<li><b>Multi-schaal Top-Hat</b>: Detecteert aggregaten van meerdere "
            "groottes tegelijk (Tab 2).</li>"
            "<li><b>Bilateral denoise</b>: Behoudt scherpe aggregaatranden "
            "terwijl ruis wordt onderdrukt.</li>"
            "<li><b>Watershed splitsing</b>: Groeperingen van aggregaten worden "
            "automatisch gesplitst op intensiteitspieken.</li>"
            "</ul>"
            "<h3>Aanbevolen werkstroom</h3>"
            "<ol>"
            "<li><b>1. Viewer</b> — Laad bestand, controleer of beeld er goed uitziet.</li>"
            "<li><b>2. Pre-processing</b> — Klik '★ Aanbevolen pipeline' voor optimale instellingen. "
            "Of gebruik 'Enhance Puncta' voor de snelste aanpak.</li>"
            "<li><b>3. Segmentatie</b> — Kies '★★ Puncta Detector Pro' (index 0). "
            "Pas gevoeligheid aan: lager = meer detecties.</li>"
            "</ol>"
            "<h3>Puncta Detector Pro — parameters</h3>"
            "<ul>"
            "<li><b>Min/Max σ</b>: Groottebereik van aggregaten in pixels</li>"
            "<li><b>Gevoeligheid</b>: 0.02–0.05. Lager = meer/kleinere detecties</li>"
            "<li><b>Cel-achtergrond σ</b>: Hoe groot de cellichamen zijn (~15–30px)</li>"
            "<li><b>Scherpheids-gate</b>: Verwijdert diffuze vlekken (cellichamen)</li>"
            "<li><b>Watershed splitsing</b>: Splitst clusters van dichte aggregaten</li>"
            "</ul>"
            "<h3>Tips</h3>"
            "<ul>"
            "<li>Als te veel cellichaam wordt gedetecteerd: verhoog gevoeligheid "
            "(0.05–0.08) of verlaag cel-onderdrukking percentiel (60–70).</li>"
            "<li>Als aggregaten worden gemist: verlaag gevoeligheid (0.015–0.025).</li>"
            "<li>Kleine scherpe stippen: min σ = 1.0–1.5, max σ = 5–8</li>"
            "<li>Grotere aggregaten: min σ = 2.0, max σ = 12–15</li>"
            "</ul>"
        )
        dlg = QDialog(self)
        dlg.setWindowTitle("Help — v2.0")
        dlg.resize(700, 650)
        dlg.setStyleSheet(DARK_THEME)
        lyt = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setHtml(txt)
        lyt.addWidget(te)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        lyt.addWidget(bb)
        dlg.exec_()

    def _show_welcome(self):
        self.status.showMessage(
            "Welkom! v4.0 — Open een .LIF of .TIF bestand via 📂 Open."
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Protein Aggregate Analyzer v4.0")
    app.setOrganizationName("ConfocalLab")
    app.setStyle("Fusion")
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()