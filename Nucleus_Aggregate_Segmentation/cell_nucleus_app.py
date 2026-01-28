import os

import tifffile as tiff
from datetime import datetime

import sys
import tkinter as tk
from tkinter import messagebox, filedialog
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
from aicsimageio import AICSImage
from readlif.reader import LifFile
import ctypes

# --- ANALYSE IMPORTS ---
from skimage import measure, transform, filters, feature
from skimage.morphology import white_tophat, disk
from skimage.draw import disk as draw_disk
from skimage.morphology import dilation, disk

# --- CONFIGURATIE ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# --- SEGMENTATION IMPORT (Cellpose) ---
try:
    from segmentation import run_cellpose_and_show_full_lif
except ImportError:
    def run_cellpose_and_show_full_lif(*args, **kwargs):
        print("Segmentation module not found.")


def get_resource_path(relative_path: str) -> str:
    """Helper for PyInstaller resource loading."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# =====================
# Constants
# =====================
UI_CANVAS_WIDTH = 900
UI_CANVAS_HEIGHT = 700

COLOR_OPTIONS = {
    "Red": (255, 0, 0),
    "Green": (0, 255, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Magenta": (255, 0, 255),
    "Cyan": (0, 255, 255),
    "Orange": (255, 128, 0),
    "White": (255, 255, 255),
    "Gray": (128, 128, 128),
}

ORGANEL_OPTIONS = ["Nucleus", "A11", "CCT1", "HA", "Other"]
Z_PROCESSING_METHOD_OPTIONS = ["slice", "max_project", "mean_project"]


# ============================================================
# IMAGE READER CLASSES
# ============================================================
class LASXLifImage:
    def __init__(self, path: str):
        file = LifFile(path)
        img0 = file.get_image(0)
        self.num_channels = img0.channels
        self.num_z = img0.dims.z

        channel_z_lists = []
        for c in range(self.num_channels):
            channel_z_lists.append([np.array(i) for i in img0.get_iter_z(t=0, c=c)])

        all_channels = np.vstack(channel_z_lists)

        real_channels = []
        for c in range(self.num_channels):
            real_c = [all_channels[i] for i in range(c, len(all_channels), self.num_channels)]
            real_channels.append(np.stack(real_c, axis=0))

        self.data = np.stack(real_channels, axis=0)

    def get_z_stack(self, z_index: int):
        return [self.data[c][z_index] for c in range(self.num_channels)]

    def get_projection(self, method: str):
        if method == "max_project":
            return [np.max(self.data[c], axis=0) for c in range(self.num_channels)]
        elif method == "mean_project":
            return [np.mean(self.data[c], axis=0) for c in range(self.num_channels)]
        else:
            return self.get_z_stack(0)


class ImageReader:
    def read_all_channels(self, path: str, z_index: int = 0, method: str = "slice"):
        ext = os.path.splitext(path)[1].lower()

        if ext == ".lif":
            lif_img = LASXLifImage(path)
            return lif_img.get_projection(method) if method != "slice" else lif_img.get_z_stack(z_index)

        else:
            try:
                img_aics = AICSImage(path)
                num_channels = getattr(img_aics.dims, "C", 1)
                z_dim = getattr(img_aics.dims, "Z", 1)
                z_index = min(z_index, z_dim - 1)

                channel_arrays = []
                for c in range(num_channels):
                    arr = img_aics.get_image_data("ZYX", C=c, T=0, S=0)
                    if method == "max_project":
                        arr = np.max(arr, axis=0)
                    elif method == "mean_project":
                        arr = np.mean(arr, axis=0)
                    else:
                        arr = arr[z_index]
                    channel_arrays.append(arr)

                return channel_arrays

            except Exception:
                arr = np.array(Image.open(path))
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=0)
                elif arr.ndim == 3:
                    arr = np.moveaxis(arr, -1, 0)
                return [arr[i] for i in range(arr.shape[0])]


# ============================================================
# MAIN APPLICATION CLASS
# ============================================================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Microscopy Analysis Platform - All In One")
        self.geometry("1450x950")

        icon_path = get_resource_path("cureq_logo.png")
        if os.path.exists(icon_path):
            try:
                self.icon_image = Image.open(icon_path)
                self.photo_icon = ImageTk.PhotoImage(self.icon_image)
                self.wm_iconphoto(True, self.photo_icon)
                self.iconphoto(True, self.photo_icon)
            except Exception:
                pass

        self.reader = ImageReader()
        self.current_path = None
        self.current_z = 0
        self.z_slices = 1
        self.num_channels = 0
        self.projection_method = tk.StringVar(value="slice")
        self.loaded_channels_data = []

        self.active_channels = []
        self.channel_colors = []
        self.channel_label_vars = []
        self.channel_checkboxes = []

        self.tk_image = None
        self.overlay_images = {}   # voorkomt garbage collection van PhotoImages
        self.last_render = None    # info over laatste render (schaal/positie)
        self.scale_factor = 1.0

        # OVERLAY TAGS
        self.blob_tag = "blob_overlay"
        self.knn_tag = "knn_overlay"  # KNN pixel-based
        self.manual_tag = "manual_poly"
        self.drawing_tag = "drawing_line"

        self.saved_polygons = []
        self.polygon_active = False
        self.polygon_points = []
        self.temp_polygon_lines = []

        self.measure_active = False
        self.measure_start = None

        # ======================================
        # LAYOUT STRUCTURE
        # ======================================
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True)

        self.sidebar = ctk.CTkScrollableFrame(main_frame, width=380, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)

        title_lbl = ctk.CTkLabel(self.sidebar, text="CONTROLS", font=("Roboto Medium", 20))
        title_lbl.pack(pady=(20, 10))

        self.canvas_frame = ctk.CTkFrame(main_frame, fg_color="#101010")
        self.canvas_frame.pack(side="right", fill="both", expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#050505", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # ============================================================
        # GUI SECTIONS
        # ============================================================

        # 1. FILE SECTION
        self.create_section_frame("File & Z-Stack")

        self.load_btn = ctk.CTkButton(
            self.current_section, text="📂 Load Image File", command=self.load_image, height=35
        )
        self.load_btn.pack(pady=5, padx=10, fill="x")

        self.filename_label = ctk.CTkLabel(
            self.current_section, text="No file loaded", text_color="gray", wraplength=300
        )
        self.filename_label.pack(pady=(0, 10))

        self.z_label = ctk.CTkLabel(self.current_section, text="Z-Position: 0")
        self.z_label.pack(pady=(5, 0))

        self.z_slider = ctk.CTkSlider(
            self.current_section, from_=0, to=0, number_of_steps=1, command=self.update_z_slice
        )
        self.z_slider.pack(pady=5, padx=10, fill="x")

        self.proj_menu = ctk.CTkOptionMenu(
            self.current_section,
            values=Z_PROCESSING_METHOD_OPTIONS,
            variable=self.projection_method,
            command=self.on_projection_change,
        )
        self.proj_menu.pack(pady=(5, 15))

        # 2. CHANNELS SECTION
        self.create_section_frame("Channels & Labels")

        self.channel_container = ctk.CTkFrame(self.current_section, fg_color="transparent")
        self.channel_container.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(
            self.channel_container, text="Load an image to see channels...", text_color="gray"
        ).pack()

        # 3. VIEW SECTION
        self.create_section_frame("View Controls")

        self.compare_btn = ctk.CTkButton(
            self.current_section,
            text="👁️ Hold to View Original",
            font=("Roboto Medium", 14),
            fg_color="#D35400",
            hover_color="#A04000",
            height=40,
        )
        self.compare_btn.pack(pady=10, padx=10, fill="x")
        self.compare_btn.bind("<ButtonPress-1>", self.hide_overlays)
        self.compare_btn.bind("<ButtonRelease-1>", self.show_overlays)


        # 5. BLOB SECTION
        self.create_section_frame("Aggregates Detection: Blob")

        b_frame = ctk.CTkFrame(self.current_section, fg_color="transparent")
        b_frame.pack(fill="x", padx=5)

        ctk.CTkLabel(b_frame, text="MinSize:").grid(row=0, column=0, padx=2)
        self.min_sigma_var = tk.StringVar(value="1")
        ctk.CTkEntry(b_frame, width=40, textvariable=self.min_sigma_var).grid(row=0, column=1)

        ctk.CTkLabel(b_frame, text="MaxSize:").grid(row=0, column=2, padx=2)
        self.max_sigma_var = tk.StringVar(value="5")
        ctk.CTkEntry(b_frame, width=40, textvariable=self.max_sigma_var).grid(row=0, column=3)

        ctk.CTkLabel(b_frame, text="Threshold:").grid(row=1, column=0, padx=2, pady=5)

        self.thresh_val_label = ctk.CTkLabel(
            b_frame, text="0.05", font=("Arial", 12, "bold"), text_color="#27AE60"
        )
        self.thresh_val_label.grid(row=1, column=3, padx=2)

        self.thresh_slider = ctk.CTkSlider(
            b_frame,
            from_=0.00,
            to=1.0,
            number_of_steps=100,
            width=100,
            command=self.update_thresh_label,
        )
        self.thresh_slider.set(0.05)
        self.thresh_slider.grid(row=1, column=1, columnspan=2, pady=5)

        self.run_blob_btn = ctk.CTkButton(
            self.current_section,
            text="🟢 Run Blob Detector",
            fg_color="#27AE60",
            hover_color="#229954",
            text_color="black",
            command=self.run_blob_detection,
        )
        self.run_blob_btn.pack(pady=10, padx=10, fill="x")
        ctk.CTkButton(
            self.current_section,
            text="Clear Model Results",
            fg_color="#7D3C98",
            hover_color="#5B2C6F",
            text_color="white",
            command=self.clear_model_results,
        ).pack(pady=(0, 10), padx=10, fill="x")


        # 6. KNN INTENSITY (PIXEL-BASED)
        self.create_section_frame("Aggregates Detection: KNN Intensity")

        kn_frame = ctk.CTkFrame(self.current_section, fg_color="transparent")
        kn_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(kn_frame, text="Radius:").grid(row=0, column=0, padx=2, pady=2, sticky="w")
        self.kn_radius_var = tk.StringVar(value="10")
        ctk.CTkEntry(kn_frame, width=50, textvariable=self.kn_radius_var).grid(
            row=0, column=1, padx=2, pady=2
        )

        ctk.CTkLabel(kn_frame, text="Min radius:").grid(row=1, column=0, padx=2, pady=2, sticky="w")
        self.kn_radius_min_var = tk.StringVar(value="5")
        ctk.CTkEntry(kn_frame, width=50, textvariable=self.kn_radius_min_var).grid(
            row=1, column=1, padx=2, pady=2
        )

        ctk.CTkLabel(kn_frame, text="Max radius:").grid(row=2, column=0, padx=2, pady=2, sticky="w")
        self.kn_radius_max_var = tk.StringVar(value="40")
        ctk.CTkEntry(kn_frame, width=50, textvariable=self.kn_radius_max_var).grid(
            row=2, column=1, padx=2, pady=2
        )

        ctk.CTkLabel(kn_frame, text="K neighbors:").grid(
            row=0, column=2, padx=8, pady=2, sticky="w"
        )
        self.kn_k_var = tk.StringVar(value="30")
        ctk.CTkEntry(kn_frame, width=50, textvariable=self.kn_k_var).grid(
            row=0, column=3, padx=2, pady=2
        )

        ctk.CTkLabel(kn_frame, text="ΔIntensity:").grid(
            row=1, column=2, padx=8, pady=2, sticky="w"
        )
        self.kn_thresh_var = tk.StringVar(value="0.6")
        ctk.CTkEntry(kn_frame, width=50, textvariable=self.kn_thresh_var).grid(
            row=1, column=3, padx=2, pady=2
        )

        self.knn_btn = ctk.CTkButton(
            self.current_section,
            text="🔵 Run KNN Intensity",
            fg_color="#3498DB",
            hover_color="#2E86C1",
            text_color="white",
            command=self.run_knn_detector,
        )
        self.knn_btn.pack(pady=6, padx=10, fill="x")
        ctk.CTkButton(
            self.current_section,
            text="Clear Model Results",
            fg_color="#7D3C98",
            hover_color="#5B2C6F",
            text_color="white",
            command=self.clear_model_results,
        ).pack(pady=(0, 10), padx=10, fill="x")


        # 7. CELLPOSE SECTION
        self.create_section_frame("Nuclei Segmentation: Cellpose")
        cp_frame = ctk.CTkFrame(self.current_section, fg_color="transparent")
        cp_frame.pack(fill="x", padx=5)
        ctk.CTkLabel(cp_frame, text="Diameter:").pack(side="left", padx=5)
        self.diameter_var = tk.StringVar(value="250")
        self.diameter_entry = ctk.CTkEntry(cp_frame, textvariable=self.diameter_var, width=60)
        self.diameter_entry.pack(side="left")
        self.segment_btn = ctk.CTkButton(
            self.current_section,
            text="Run Cellpose (3D)",
            command=self.segment_nucleus,
            fg_color="#5B2C6F",
            hover_color="#4A235A",
        )
        self.segment_btn.pack(pady=10, padx=10, fill="x")

        # 8. MANUAL TOOLS
        self.create_section_frame("Manual Tools")
        grid_frame = ctk.CTkFrame(self.current_section, fg_color="transparent")
        grid_frame.pack(fill="x", padx=5, pady=5)
        self.measure_btn = ctk.CTkButton(
            grid_frame, text="📏 Measure", command=self.activate_measure, width=150
        )
        self.measure_btn.grid(row=0, column=0, padx=5, pady=5)
        self.polygon_btn = ctk.CTkButton(
            grid_frame, text="✏️ Draw Poly", command=self.activate_polygon_measure, width=150
        )
        self.polygon_btn.grid(row=0, column=1, padx=5, pady=5)
        self.clear_btn = ctk.CTkButton(
            self.current_section,
            text="🗑️ Clear All Drawings",
            fg_color="#E74C3C",
            hover_color="#C0392B",
            command=self.clear_drawings,
        )
        self.clear_btn.pack(pady=(5, 10), padx=10, fill="x")

    # ============================================================
    # HELPER UI METHODS
    # ============================================================
    def create_section_frame(self, title: str):
        frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="#2B2B2B",
            border_color="#404040",
            border_width=1,
            corner_radius=6,
        )
        frame.pack(fill="x", padx=10, pady=10)
        lbl = ctk.CTkLabel(
            frame, text=title, font=("Roboto Medium", 14), text_color="#E0E0E0"
        )
        lbl.pack(anchor="w", padx=10, pady=(8, 0))
        tk.Frame(frame, height=1, bg="#404040").pack(fill="x", padx=10, pady=5)
        self.current_section = frame

    def update_strictness_label(self, value):
        self.strictness_val_label.configure(text=f"{float(value):.1f}")

    def update_thresh_label(self, value):
        self.thresh_val_label.configure(text=f"{float(value):.2f}")

    def hide_overlays(self, event):
        self.canvas.itemconfigure(self.blob_tag, state="hidden")
        self.canvas.itemconfigure(self.knn_tag, state="hidden")
        self.canvas.itemconfigure(self.manual_tag, state="hidden")
        self.canvas.itemconfigure(self.drawing_tag, state="hidden")

    def show_overlays(self, event):
        self.canvas.itemconfigure(self.blob_tag, state="normal")
        self.canvas.itemconfigure(self.knn_tag, state="normal")
        self.canvas.itemconfigure(self.manual_tag, state="normal")
        self.canvas.itemconfigure(self.drawing_tag, state="normal")

    # ============================================================
    # IMAGE LOADING & DISPLAY
    # ============================================================
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.tif *.tiff *.png *.jpg *.jpeg *.lif")]
        )
        if not path:
            return
        self.current_path = path
        self.filename_label.configure(text=f"{os.path.basename(path)}")
        ext = os.path.splitext(path)[1].lower()
        if ext == ".lif":
            lif_img = LASXLifImage(path)
            self.num_channels, self.z_slices = lif_img.data.shape[0], lif_img.data.shape[1]
        else:
            try:
                img_aics = AICSImage(path)
                self.num_channels = getattr(img_aics.dims, "C", 1)
                self.z_slices = getattr(img_aics.dims, "Z", 1)
            except Exception:
                self.num_channels, self.z_slices = 1, 1
        self.setup_slider_visibility()
        self.create_channel_controls()
        self.display_image()

    def create_channel_controls(self):
        for widget in self.channel_container.winfo_children():
            widget.destroy()
        self.active_channels = []
        self.channel_label_vars = []
        self.channel_colors = []
        self.channel_checkboxes = []

        self.active_channels = [ctk.BooleanVar(value=True) for _ in range(self.num_channels)]
        self.channel_colors = [
            list(COLOR_OPTIONS.keys())[i % len(COLOR_OPTIONS)] for i in range(self.num_channels)
        ]

        for i in range(self.num_channels):
            f = ctk.CTkFrame(self.channel_container, fg_color="transparent")
            f.pack(fill="x", pady=2)
            cb = ctk.CTkCheckBox(
                f,
                text=f"Ch {i}",
                width=60,
                variable=self.active_channels[i],
                command=self.display_image,
            )
            cb.pack(side="left")
            self.channel_checkboxes.append(cb)

            dd_c = ctk.CTkOptionMenu(
                f,
                width=80,
                values=list(COLOR_OPTIONS.keys()),
                command=lambda v, idx=i: self.update_channel_color(idx, v),
            )
            dd_c.set(self.channel_colors[i])
            dd_c.pack(side="right", padx=2)

            label_var = tk.StringVar(value="Other")
            self.channel_label_vars.append(label_var)
            ctk.CTkOptionMenu(
                f, width=90, values=ORGANEL_OPTIONS, variable=label_var
            ).pack(side="right", padx=2)

    def on_projection_change(self, _):
        self.setup_slider_visibility()
        self.display_image()

    def setup_slider_visibility(self):
        if self.projection_method.get() == "slice" and self.z_slices > 1:
            self.z_label.configure(text=f"Z-Pos: {self.current_z}")
            self.z_slider.configure(
                from_=0,
                to=self.z_slices - 1,
                number_of_steps=max(self.z_slices - 1, 1),
            )
            self.z_label.pack(pady=(5, 0))
            self.z_slider.pack(pady=5, padx=10, fill="x")
        else:
            self.z_slider.pack_forget()
            self.z_label.pack_forget()

    def update_channel_color(self, idx, v):
        self.channel_colors[idx] = v
        self.display_image()

    def update_z_slice(self, v):
        self.current_z = int(float(v))
        self.z_label.configure(text=f"Z-Pos: {self.current_z}")
        self.display_image()

    def display_image(self):
        if not self.current_path:
            return
        try:
            self.loaded_channels_data = self.reader.read_all_channels(
                self.current_path, self.current_z, self.projection_method.get()
            )
        except Exception:
            return

        rgb = np.zeros((*self.loaded_channels_data[0].shape, 3), dtype=np.float32)
        for i, arr in enumerate(self.loaded_channels_data):
            if not self.active_channels[i].get():
                continue
            c = np.array(COLOR_OPTIONS[self.channel_colors[i]]) / 255.0
            arr = (arr.astype(np.float32) - arr.min()) / (np.ptp(arr) + 1e-8)
            rgb += np.expand_dims(arr, -1) * c

        self._render_to_canvas(rgb)

    def _render_to_canvas(self, rgb_data: np.ndarray):
        rgb_data = np.clip(rgb_data, 0, 1)
        img = Image.fromarray((rgb_data * 255).astype(np.uint8))

        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw <= 1:
            cw, ch = UI_CANVAS_WIDTH, UI_CANVAS_HEIGHT

        ir, cr = img.width / img.height, cw / ch
        if ir > cr:
            nw, nh = cw, int(cw / ir)
        else:
            nw, nh = int(ch * ir), ch

        self.scale_factor = nw / img.width
        self.tk_image = ImageTk.PhotoImage(img.resize((nw, nh), Image.Resampling.LANCZOS))

        self.canvas.delete("image_bg")
        self.canvas.create_image(
            cw // 2,
            ch // 2,
            anchor="center",
            image=self.tk_image,
            tags="image_bg",
        )
        self.canvas.tag_lower("image_bg")

        # --- onthoud render-parameters (nodig voor overlays) ---
        ox, oy = (cw - nw) // 2, (ch - nh) // 2
        self.last_render = {
            "cw": cw, "ch": ch,
            "nw": nw, "nh": nh,
            "ox": ox, "oy": oy,
            "orig_w": img.width, "orig_h": img.height
        }

        # overlays boven leggen
        self.canvas.tag_raise(self.blob_tag)
        self.canvas.tag_raise(self.knn_tag)
        self.canvas.tag_raise(self.manual_tag)
        self.canvas.tag_raise(self.drawing_tag)


    # ============================================================
    # MODEL RUNS
    # ============================================================
    def run_blob_detection(self):
        if not self.loaded_channels_data:
            return
        ha_idx = -1
        for i, var in enumerate(self.channel_label_vars):
            if var.get() == "HA":
                ha_idx = i
                break
        if ha_idx == -1:
            messagebox.showwarning("Error", "No 'HA' channel found.")
            return

        try:
            min_s = float(self.min_sigma_var.get())
            max_s = float(self.max_sigma_var.get())
            thresh = float(self.thresh_slider.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid Blob parameters.")
            return

        raw = self.loaded_channels_data[ha_idx]
        norm = (raw - raw.min()) / (np.ptp(raw) + 1e-8)

        blobs = feature.blob_log(norm, min_sigma=min_s, max_sigma=max_s, num_sigma=5, threshold=thresh)

        mask = np.zeros_like(norm, dtype=bool)

        for (y, x, sigma) in blobs:
            r = max(1, int(sigma * np.sqrt(2)))  # ongeveer dezelfde radius als je cirkel
            rr, cc = draw_disk((int(y), int(x)), r, shape=mask.shape)
            mask[rr, cc] = True
        
        # --- tekenen in GUI ---
        self.draw_mask_overlay(mask, (0, 255, 0), self.blob_tag, alpha=110)
        # --- export ---
        self._save_mask_tif(mask.astype(np.uint8), "BLOB")

        messagebox.showinfo("Blob Result", f"Found {len(blobs)} aggregates.")

    # ---------- KNN INTENSITY (PIXEL-BASED) ----------
    def run_knn_detector(self):
        if not self.loaded_channels_data:
            return

        ha_idx = -1
        for i, var in enumerate(self.channel_label_vars):
            if var.get() == "HA":
                ha_idx = i
                break
        if ha_idx == -1:
            messagebox.showwarning("Error", "No 'HA' channel found.")
            return

        try:
            radius = int(self.kn_radius_var.get())
            rmin = int(self.kn_radius_min_var.get())
            rmax = int(self.kn_radius_max_var.get())
            K = int(self.kn_k_var.get())
            thresh = float(self.kn_thresh_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid KNN pixel parameters.")
            return

        if rmin < 1 or rmax < 1 or rmin > rmax:
            messagebox.showerror("Error", "Min radius and max radius must be ≥ 1 and min ≤ max.")
            return
        if K < 1:
            messagebox.showerror("Error", "K must be ≥ 1.")
            return

        radius = max(rmin, min(radius, rmax))

        img = self.loaded_channels_data[ha_idx].astype(float)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        h, w = img.shape
        detected = []

        for y in range(radius, h - radius):
            for x in range(radius, w - radius):
                center = img[y, x]
                window = img[y-radius:y+radius+1, x-radius:x+radius+1].flatten()
                center_idx = len(window) // 2
                window_no_center = np.delete(window, center_idx)
                if window_no_center.size == 0:
                    continue
                K_eff = min(K, window_no_center.size)
                neighbors = np.sort(window_no_center)[:K_eff]
                if center - neighbors.mean() > thresh:
                    detected.append((y, x))

        # --- maak binaire mask op originele resolutie ---
        mask = np.zeros((h, w), dtype=bool)
        for (y, x) in detected:
            mask[y, x] = True

        # optioneel: maak puntjes wat groter zodat het echt "ingekleurd" oogt
        dil_r = max(1, radius // 4)  # speel hiermee
        mask = dilation(mask, disk(dil_r))
        
        # --- teken in GUI ----
        self.draw_mask_overlay(mask, (52, 152, 219), self.knn_tag, alpha=110)
        # --- export ---
        self._save_mask_tif(mask.astype(np.uint8), "KNN")

        messagebox.showinfo("KNN Pixels", f"Detected {len(detected)} high-intensity pixels.")

    # ============================================================
    # CLEAR MODEL OVERLAYS
    # ============================================================
    def clear_model_results(self):
        self.canvas.delete(self.blob_tag)
        self.canvas.delete(self.knn_tag)
        self.overlay_images.pop(self.blob_tag, None)
        self.overlay_images.pop(self.knn_tag, None)
        self.display_image()


    # ============================================================
    # DRAW HELPERS FOR MODEL RESULTS
    # ============================================================
    def draw_mask_overlay(self, mask: np.ndarray, rgb_color: tuple, tag: str, alpha: int = 120):
        """
        mask: 2D bool/0-1 array (originele beeldresolutie)
        rgb_color: (R,G,B)
        alpha: 0-255
        """
        self.canvas.delete(tag)

        if self.last_render is None:
            return

        # Resize mask naar het schermformaat (nearest = harde randen)
        nw, nh = self.last_render["nw"], self.last_render["nh"]
        mask_u8 = (mask.astype(np.uint8) * 255)
        mask_img = Image.fromarray(mask_u8, mode="L").resize((nw, nh), Image.Resampling.NEAREST)

        # Maak RGBA overlay array
        m = np.array(mask_img, dtype=np.uint8)  # 0..255
        overlay = np.zeros((nh, nw, 4), dtype=np.uint8)
        overlay[..., 0] = rgb_color[0]
        overlay[..., 1] = rgb_color[1]
        overlay[..., 2] = rgb_color[2]
        overlay[..., 3] = (m > 0).astype(np.uint8) * alpha  # vaste alpha waar mask==1

        overlay_img = Image.fromarray(overlay, mode="RGBA")
        tk_overlay = ImageTk.PhotoImage(overlay_img)

        # Bewaar reference zodat Tk 'm niet weggooit
        self.overlay_images[tag] = tk_overlay

        # Zelfde centrering als background
        cw, ch = self.last_render["cw"], self.last_render["ch"]
        self.canvas.create_image(
            cw // 2, ch // 2,
            anchor="center",
            image=tk_overlay,
            tags=tag
        )
    
    def _get_ha_index(self):
        for i, var in enumerate(self.channel_label_vars):
            if var.get() == "HA":
                return i
        return -1

    def _default_export_dir(self):
        # zelfde map als input bestand
        if self.current_path:
            return os.path.dirname(self.current_path)
        return os.getcwd()

    def _save_mask_tif(self, mask01: np.ndarray, method_name: str):
        """
        mask01: uint8/bool 2D array, values {0,1}
        Saved alongside the input file by default (or user-selected dir if you prefer).
        """
        if self.current_path is None:
            return

        out_dir = filedialog.askdirectory(initialdir=self._default_export_dir(), title="Choose export folder")
        if not out_dir:
            return

        base = os.path.splitext(os.path.basename(self.current_path))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"{base}_{method_name}_mask_{ts}.tif")

        mask01 = (mask01 > 0).astype(np.uint8)  # ensure 0/1
        tiff.imwrite(out_path, mask01, photometric="minisblack")
        messagebox.showinfo("Export", f"Saved mask:\n{out_path}")


    # ============================================================
    # CELLPOSE
    # ============================================================
    def segment_nucleus(self):
        if not self.current_path:
            return
        nuc_idx = -1
        for i, var in enumerate(self.channel_label_vars):
            if var.get() == "Nucleus":
                nuc_idx = i
                break
        if nuc_idx == -1:
            messagebox.showerror("Error", "Set channel to 'Nucleus'.")
            return
        try:
            d = int(self.diameter_var.get())
        except Exception:
            d = 250
        print(f"🧬 Segmenting C{nuc_idx}")
        run_cellpose_and_show_full_lif(
            self.current_path, channel_index=nuc_idx, gpu=False, diameter=d
        )

    # ============================================================
    # MANUAL TOOLS
    # ============================================================
    def activate_measure(self):
        self.measure_active = True
        self.polygon_active = False
        self.canvas.config(cursor="cross")
        self.canvas.bind("<Button-1>", self.start_measure)
        self.canvas.bind("<ButtonRelease-1>", self.end_measure)
        self.canvas.bind("<Motion>", self.update_measure_line)

    def start_measure(self, e):
        self.measure_start = (e.x, e.y)
        self.measure_line = self.canvas.create_line(
            e.x, e.y, e.x, e.y, fill="yellow", width=2
        )

    def update_measure_line(self, e):
        if self.measure_active and self.measure_start:
            self.canvas.coords(
                self.measure_line,
                self.measure_start[0],
                self.measure_start[1],
                e.x,
                e.y,
            )

    def end_measure(self, e):
        if not self.measure_active or not self.measure_start:
            return
        d = ((e.x - self.measure_start[0]) ** 2 + (e.y - self.measure_start[1]) ** 2) ** 0.5
        win = tk.Toplevel(self)
        win.geometry("300x100")
        tk.Label(win, text=f"{(d / self.scale_factor):.2f} px", font=("Arial", 24)).pack(
            expand=True
        )
        self.canvas.delete(self.measure_line)
        self.measure_start = None
        self.measure_active = False
        self.canvas.config(cursor="")
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.canvas.unbind("<Motion>")

    def activate_polygon_measure(self):
        self.polygon_active = True
        self.measure_active = False
        self.polygon_points = []
        self.temp_polygon_lines = []
        self.canvas.config(cursor="cross")
        self.canvas.bind("<Button-1>", self.add_polygon_point)
        self.canvas.bind("<Button-3>", self.finish_current_polygon)
        self.bind("<Escape>", lambda e: self.stop_drawing())

    def add_polygon_point(self, e):
        if not self.polygon_active:
            return
        self.polygon_points.append((e.x, e.y))
        if len(self.polygon_points) > 1:
            line = self.canvas.create_line(
                self.polygon_points[-2],
                self.polygon_points[-1],
                fill="yellow",
                dash=(2, 2),
                tags=self.drawing_tag,
            )
            self.temp_polygon_lines.append(line)

    def finish_current_polygon(self, e):
        if len(self.polygon_points) < 3:
            return
        for l in self.temp_polygon_lines:
            self.canvas.delete(l)
        self.temp_polygon_lines = []
        final = self.polygon_points + [self.polygon_points[0]]
        self.saved_polygons.append(final)
        coords = [c for p in final for c in p]
        self.canvas.create_line(coords, fill="cyan", width=2, tags=self.manual_tag)
        self.polygon_points = []

    def stop_drawing(self):
        self.polygon_points = []
        self.temp_polygon_lines = []
        self.polygon_active = False
        self.canvas.config(cursor="")
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<Button-3>")

    def clear_drawings(self):
        self.saved_polygons = []
        self.polygon_points = []
        self.temp_polygon_lines = []
        self.canvas.delete(self.manual_tag)
        self.canvas.delete(self.drawing_tag)
        self.display_image()


if __name__ == "__main__":
    app = App()
    app.mainloop()