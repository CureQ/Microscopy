# ============================
# File: agg_segmentation.py
# Purpose: Segment mHTT aggregates (HA channel) in Huntington's disease microscopy data
# Supports LINA 1-plane model (resized inputs) or Cellpose
# Handles LIF/CZI files with missing metadata
# Adds optional Napari interactive 3D visualization
# ============================

import os
import numpy as np
from aicsimageio import AICSImage
from readlif.reader import LifFile
import scipy.ndimage as ndi
from skimage.transform import resize
import matplotlib.pyplot as plt

# Optional imports
try:
    from cellpose import models as cp_models
except ImportError:
    cp_models = None

try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None

try:
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    tk = None

try:
    import napari
except ImportError:
    napari = None

# ============================
# CONFIGURATION
# ============================
AGG_CHANNEL_NAME = "HA"
DEFAULT_HA_IDX = 2
USE_LINA = True
LINA_MODEL_PATH = "LINA/Models/PixelRegressionModel_1Plane.h5"
MODEL_TYPE = "cyto2"          # for Cellpose
MIN_AGG_AREA = 10
OUTPUT_DIR = "aggregate_results"
LINA_INPUT_SIZE = 352          # input size for 1-plane LINA model

# ============================
# IMAGE LOADING
# ============================
def load_image(filepath, channel_idx=None):
    """Load .lif or .czi image, return numpy array (Z, Y, X)."""
    ext = os.path.splitext(filepath)[1].lower()
    print(f"[INFO] Loading image: {filepath}")

    if ext == ".lif":
        lif = LifFile(filepath)
        series = lif.get_image(0)
        if channel_idx is None:
            try:
                channel_names = series.metadata['channel_names']
                if AGG_CHANNEL_NAME in channel_names:
                    channel_idx = channel_names.index(AGG_CHANNEL_NAME)
                else:
                    print(f"[WARNING] HA channel not found; defaulting to {DEFAULT_HA_IDX}")
                    channel_idx = DEFAULT_HA_IDX
            except (AttributeError, KeyError):
                print(f"[WARNING] Metadata unavailable; defaulting to {DEFAULT_HA_IDX}")
                channel_idx = DEFAULT_HA_IDX

        data = np.array([series.get_frame(z=z, t=0, c=channel_idx) for z in range(series.dims.z)])

    else:
        img = AICSImage(filepath)
        data = img.get_image_data("CZYX")
        if channel_idx is None:
            channel_idx = DEFAULT_HA_IDX
        if data.shape[0] > 1:
            data = data[channel_idx]
        if data.ndim == 3:  # already ZYX
            pass

    print(f"[INFO] Loaded image with shape {data.shape}")
    return data

# ============================
# PREPROCESSING
# ============================
def preprocess_slice(slice_img):
    """Normalize intensity and apply Gaussian smoothing."""
    if np.ptp(slice_img) == 0:
        return np.zeros_like(slice_img)
    norm = (slice_img - np.min(slice_img)) / np.ptp(slice_img)
    smooth = ndi.gaussian_filter(norm, sigma=1)
    return smooth

# ============================
# SEGMENTATION
# ============================
def segment_aggregates_per_zstack(stack):
    """Segment aggregates slice-by-slice using LINA 1-plane or Cellpose."""
    masks_z = []

    if USE_LINA:
        if load_model is None:
            raise ImportError("[ERROR] TensorFlow/Keras not available for LINA.")
        if not os.path.exists(LINA_MODEL_PATH):
            raise FileNotFoundError(f"[ERROR] LINA model not found: {LINA_MODEL_PATH}")

        print(f"[INFO] Loading LINA 1-plane model from {LINA_MODEL_PATH}...")
        model = load_model(LINA_MODEL_PATH)
        print("[INFO] Model loaded successfully.")

        for z, slice_img in enumerate(stack):
            print(f"   > Segmenting slice {z + 1}/{len(stack)}")
            img_pre = preprocess_slice(slice_img)
            input_img = resize(img_pre, (LINA_INPUT_SIZE, LINA_INPUT_SIZE), preserve_range=True, anti_aliasing=True)
            input_img = input_img[np.newaxis, :, :, np.newaxis]

            try:
                pred = model.predict(input_img)
                mask_bin = (pred[0, :, :, 0] > 0.5).astype(int)
                mask_bin = resize(mask_bin, slice_img.shape, order=0, preserve_range=True, anti_aliasing=False).astype(int)
            except Exception as e:
                print(f"[ERROR] Prediction failed on slice {z}: {e}")
                mask_bin = np.zeros_like(slice_img, dtype=int)

            masks_z.append(mask_bin)

    else:
        if cp_models is None:
            raise ImportError("[ERROR] Cellpose not installed.")
        print(f"[INFO] Using Cellpose ({MODEL_TYPE}) per Z-slice...")
        cp_model = cp_models.CellposeModel(gpu=False, model_type=MODEL_TYPE)

        for z, slice_img in enumerate(stack):
            print(f"   > Processing slice {z + 1}/{len(stack)} with Cellpose...")
            img_pre = preprocess_slice(slice_img)
            try:
                masks, _, _ = cp_model.eval(img_pre, channels=[0, 0])
            except Exception as e:
                print(f"[ERROR] Slice {z} segmentation failed: {e}")
                masks = np.zeros_like(img_pre, dtype=int)
            masks_z.append(masks)

    masks_z = np.array(masks_z)
    if masks_z.ndim == 2:
        masks_z = masks_z[np.newaxis, :, :]
    return masks_z

# ============================
# POSTPROCESSING
# ============================
def postprocess_masks(masks_z):
    """Clean masks and perform 3D labeling."""
    if masks_z.ndim == 2:
        masks_z = masks_z[np.newaxis, :, :]

    cleaned = np.zeros_like(masks_z)
    for z in range(masks_z.shape[0]):
        mask = masks_z[z] > 0
        mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
        labeled, n = ndi.label(mask)
        sizes = ndi.sum(mask, labeled, range(n + 1))
        remove_pixel = sizes[labeled] < MIN_AGG_AREA
        labeled[remove_pixel] = 0
        cleaned[z] = labeled > 0

    labeled_3d, num_features = ndi.label(cleaned)
    print(f"[INFO] Total 3D aggregates detected: {num_features}")
    return labeled_3d

# ============================
# SAVE RESULTS
# ============================
def save_results(masks_3d, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "aggregates_mask_3d.npy"), masks_3d)
    print(f"[INFO] Saved 3D mask → {output_dir}/aggregates_mask_3d.npy")

    # Save a preview image of middle Z slice
    mid_z = masks_3d.shape[0] // 2
    plt.figure(figsize=(6,6))
    plt.imshow(masks_3d[mid_z], cmap="magma")
    plt.title("Aggregate Segmentation (mid-Z slice)")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "aggregate_segmentation_preview.png"))
    plt.close()

# ============================
# NAPARI VIEWER
# ============================
def view_with_napari(img_stack, mask_stack):
    """Interactive 3D visualization with Napari."""
    if napari is None:
        print("[WARNING] Napari not installed. Skipping interactive view.")
        return
    viewer = napari.Viewer()
    viewer.add_image(img_stack, name="HA channel")
    viewer.add_labels(mask_stack, name="Aggregates")
    napari.run()

# ============================
# PIPELINE WRAPPER
# ============================
def run_aggregate_segmentation(filepath, channel_idx=None, return_mask=True, gui_parent=None, napari_view=False):
    """Full 3D aggregate segmentation pipeline with optional GUI/Napari visualization."""
    print("[INFO] Starting aggregate segmentation pipeline...")
    data = load_image(filepath, channel_idx=channel_idx)

    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    masks_z = segment_aggregates_per_zstack(data)
    masks_3d = postprocess_masks(masks_z)
    save_results(masks_3d)

    # Optional GUI viewer
    if gui_parent is not None and tk is not None:
        class ZStackViewer(tk.Toplevel):
            def __init__(self, parent, img_stack, mask_stack):
                super().__init__(parent)
                self.img_stack = img_stack
                self.mask_stack = mask_stack
                self.z = 0
                self.fig, self.ax = plt.subplots(figsize=(5,5))
                self.canvas = FigureCanvasTkAgg(self.fig, master=self)
                self.canvas.get_tk_widget().pack()
                btn_prev = tk.Button(self, text="Previous Z", command=self.prev_slice)
                btn_prev.pack(side="left")
                btn_next = tk.Button(self, text="Next Z", command=self.next_slice)
                btn_next.pack(side="right")
                self.show_slice()
            def show_slice(self):
                self.ax.clear()
                self.ax.imshow(self.img_stack[self.z], cmap="gray")
                self.ax.imshow(self.mask_stack[self.z], cmap="magma", alpha=0.5)
                self.ax.set_title(f"Slice {self.z+1}/{self.img_stack.shape[0]}")
                self.ax.axis("off")
                self.canvas.draw()
            def next_slice(self):
                if self.z < self.img_stack.shape[0]-1:
                    self.z += 1
                    self.show_slice()
            def prev_slice(self):
                if self.z > 0:
                    self.z -= 1
                    self.show_slice()
        ZStackViewer(gui_parent, data, masks_3d)

    # Optional Napari viewer
    if napari_view:
        view_with_napari(data, masks_3d)

    if return_mask:
        return masks_3d

# ============================
# CLI ENTRY POINT
# ============================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="3D Aggregate segmentation for HA channel mHTT images")
    parser.add_argument("filepath", help="Path to .lif or .czi microscopy file")
    parser.add_argument("--channel", type=int, default=None, help="Channel index to segment (optional)")
    parser.add_argument("--no-return", action="store_true", help="Don't return mask in memory")
    parser.add_argument("--napari", action="store_true", help="Launch Napari to view results interactively")
    args = parser.parse_args()

    run_aggregate_segmentation(args.filepath, 
                               channel_idx=args.channel, 
                               return_mask=not args.no_return,
                               napari_view=args.napari)