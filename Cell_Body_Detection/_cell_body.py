import io
import os
from tkinter import filedialog

import customtkinter as ctk
import numpy as np
from aicsimageio import AICSImage
from cellpose import models
from CTkMessagebox import CTkMessagebox
from customtkinter import CTkImage
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


class cell_body_frame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.folder_path = ""
        self.data_path = ""
        self.original_image = None
        self.mask_array = None
        self.included_cells = set()
        self.boundary_color = ctk.StringVar(value="Green")

        self.settings_panel = ctk.CTkFrame(self, width=200)
        self.settings_panel.grid(row=0, column=0, sticky="ns")
        self.settings_panel.grid_propagate(False)
        self.create_settings_panel()

        self.viewer_panel = ctk.CTkFrame(self)
        self.viewer_panel.grid(row=0, column=1, sticky="nsew")
        self.viewer_panel.grid_rowconfigure(0, weight=1)
        self.viewer_panel.grid_columnconfigure(0, weight=1)

        self.image_label = ctk.CTkLabel(
            self.viewer_panel, text="Select an Image", anchor="center"
        )
        self.image_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.image_label.bind("<Button-1>", self.on_click)

        self.output_panel = ctk.CTkFrame(self, width=200)
        self.output_panel.grid(row=0, column=2, sticky="ns")
        self.output_panel.grid_propagate(False)
        self.create_output_panel()

    def create_settings_panel(self):
        ctk.CTkLabel(
            self.settings_panel,
            text="Settings",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(10, 5))

        # --- Navigation ---
        navigation_frame = ctk.CTkFrame(self.settings_panel, fg_color="transparent")
        navigation_frame.pack(padx=10, pady=(10, 10), fill="x")

        ctk.CTkLabel(
            navigation_frame, text="Navigation Options", font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w")
        ctk.CTkButton(
            self.settings_panel,
            text="Return to start screen",
            command=lambda: self.parent.show_frame(self.parent.home_frame),
        ).pack(padx=10, pady=(10, 15), fill="x")

        # --- Import Settings ---
        import_frame = ctk.CTkFrame(self.settings_panel, fg_color="transparent")
        import_frame.pack(padx=10, pady=(5, 0), fill="x")

        ctk.CTkLabel(
            import_frame, text="Import Settings", font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w")

        self.select_image_btn = ctk.CTkButton(
            import_frame, text="Select image", command=self.load_image
        )
        self.select_image_btn.pack(pady=(5, 5), fill="x")

        self.filename_label = ctk.CTkLabel(
            import_frame, text="No file selected", wraplength=180
        )
        self.filename_label.pack(pady=(0, 5), fill="x")

        ctk.CTkLabel(import_frame, text="Composite:", anchor="w").pack(
            pady=(5, 0), fill="x"
        )
        self.comp_option = ctk.CTkOptionMenu(
            import_frame, values=["Option 1", "Option 2", "Option 3"]
        )
        self.comp_option.pack(pady=(0, 5), fill="x")

        ctk.CTkLabel(import_frame, text="Stack:", anchor="w").pack(
            pady=(5, 0), fill="x"
        )
        self.stack_option = ctk.CTkOptionMenu(
            import_frame, values=["Stack 1", "Stack 2", "Stack 3"]
        )
        self.stack_option.pack(pady=(0, 5), fill="x")

        # --- Model Settings ---
        model_frame = ctk.CTkFrame(self.settings_panel, fg_color="transparent")
        model_frame.pack(padx=10, pady=(10, 0), fill="x")

        ctk.CTkLabel(
            model_frame, text="Model Settings", font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w")

        ctk.CTkLabel(model_frame, text="Diameter:", anchor="w").pack(
            pady=(5, 0), fill="x"
        )
        self.dia_entry = ctk.CTkEntry(model_frame)
        self.dia_entry.insert(0, "100")
        self.dia_entry.pack(pady=(0, 5), fill="x")

        self.segment_button = ctk.CTkButton(
            model_frame, text="Segment", command=self.run_segmentation, state="disabled"
        )
        self.segment_button.pack(pady=(5, 5), fill="x")

        # --- Display Options ---
        display_frame = ctk.CTkFrame(self.settings_panel, fg_color="transparent")
        display_frame.pack(padx=10, pady=(10, 10), fill="x")

        ctk.CTkLabel(
            display_frame, text="Display Options", font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w")

        self.show_original = ctk.BooleanVar(value=True)
        self.show_mask = ctk.BooleanVar(value=True)
        self.show_boundaries = ctk.BooleanVar(value=False)

        ctk.CTkCheckBox(
            display_frame,
            text="Show original",
            variable=self.show_original,
            command=self.update_display,
        ).pack(anchor="w")
        ctk.CTkCheckBox(
            display_frame,
            text="Show mask",
            variable=self.show_mask,
            command=self.update_display,
        ).pack(anchor="w")
        ctk.CTkCheckBox(
            display_frame,
            text="Show boundaries",
            variable=self.show_boundaries,
            command=self.update_display,
        ).pack(anchor="w")

        ctk.CTkLabel(display_frame, text="Boundary color:", anchor="w").pack(
            pady=(10, 0), anchor="w"
        )
        ctk.CTkOptionMenu(
            display_frame,
            variable=self.boundary_color,
            values=["Green", "Red", "Blue", "Yellow"],
            command=lambda _: self.update_display(),
        ).pack(fill="x")

    def create_output_panel(self):
        ctk.CTkLabel(
            self.output_panel, text="Output", font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)

        self.stats_label = ctk.CTkLabel(
            self.output_panel,
            text="Cell count:\nFound: ...\nSelected: ...",
            justify="left",
        )
        self.stats_label.pack(padx=10, pady=10)

        ctk.CTkButton(
            self.output_panel, text="Export", command=self.export_selected
        ).pack(side="bottom", padx=10, pady=10)
        ctk.CTkButton(
            self.output_panel, text="Export PDF", command=self.export_pdf
        ).pack(side="bottom", padx=10, pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Microscopy Image", "*.tif *.tiff *.lif *.png")]
        )

        if file_path:
            self.mask_array = None
            self.included_cells = set()

            self.data_path = file_path
            img_data = self.read_image(file_path)
            self.base_filename = os.path.splitext(os.path.basename(file_path))[0]
            self.filename_label.configure(text=self.base_filename)
            self.original_image = (
                img_data
                if isinstance(img_data, Image.Image)
                else Image.fromarray(img_data)
            )
            self.segment_button.configure(state="normal")
            self.update_display()

    def read_image(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in [".lif", ".tiff", ".tif"]:
            img = AICSImage(path)
            arr = img.get_image_data("YX", S=0, T=0, C=0, Z=0)
        else:
            img = Image.open(path)
            arr = img.convert("RGB")
        return arr

    def run_segmentation(self):
        try:
            model = models.CellposeModel(gpu=True)
            img = self.read_image(self.data_path)
            img_np = np.array(img) if isinstance(img, Image.Image) else img

            if img_np.ndim == 3 and img_np.shape[2] == 3:
                from skimage.color import rgb2gray

                img_np = rgb2gray(img_np)

            dia = float(self.dia_entry.get()) if self.dia_entry.get() else None
            masks, flows, styles = model.eval(img_np, diameter=dia)
            self.mask_array = masks
            self.included_cells = set(np.unique(masks)) - {0}
            self.update_display()

        except Exception as e:
            CTkMessagebox(title="Error", message=str(e), icon="cancel")

    def on_click(self, event):
        if self.mask_array is None:
            return

        x = int(event.x * self.mask_array.shape[1] / 1024)
        y = int(event.y * self.mask_array.shape[0] / 1024)

        if 0 <= x < self.mask_array.shape[1] and 0 <= y < self.mask_array.shape[0]:
            cell_id = self.mask_array[y, x]
            if cell_id != 0:
                if cell_id in self.included_cells:
                    self.included_cells.remove(cell_id)
                else:
                    self.included_cells.add(cell_id)
                self.update_display()

    def get_exact_boundaries(self, label_mask):
        padded = np.pad(label_mask, pad_width=1, mode="edge")
        boundaries = np.zeros_like(label_mask, dtype=bool)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = padded[
                1 + dy : 1 + dy + label_mask.shape[0],
                1 + dx : 1 + dx + label_mask.shape[1],
            ]
            boundaries |= label_mask != shifted
        return boundaries

    def update_display(self):
        base = Image.new("RGB", (1024, 1024), color="black")

        if self.original_image:
            img = self.original_image.resize((1024, 1024))
            if self.show_original.get():
                base = img.copy()

        if self.mask_array is not None:
            resized_mask = np.array(
                Image.fromarray(self.mask_array).resize(
                    (1024, 1024), resample=Image.NEAREST
                )
            )
            mask_rgb = np.zeros((1024, 1024, 3), dtype=np.uint8)
            rng = np.random.default_rng(seed=42)
            color_map = {
                cell_id: rng.integers(50, 256, size=3)
                for cell_id in np.unique(resized_mask)
                if cell_id != 0
            }

            for cell_id in self.included_cells:
                if cell_id == 0:
                    continue
                mask_rgb[resized_mask == cell_id] = color_map.get(cell_id, [255, 0, 0])

            mask_rgb_img = Image.fromarray(mask_rgb)

            if self.show_mask.get():
                mask_rgb_img = mask_rgb_img.convert("RGB")
                base = base.resize(mask_rgb_img.size).convert("RGB")
                base = Image.blend(base, mask_rgb_img, alpha=0.4)

            if self.show_boundaries.get():
                exact_boundaries = self.get_exact_boundaries(self.mask_array)
                boundary_mask = np.zeros_like(exact_boundaries)
                for cid in self.included_cells:
                    boundary_mask |= exact_boundaries & (self.mask_array == cid)
                boundary_resized = (
                    np.array(
                        Image.fromarray(boundary_mask.astype(np.uint8) * 255).resize(
                            (1024, 1024), resample=Image.NEAREST
                        )
                    )
                    > 0
                )

                base = base.convert("RGB")
                base_np = np.array(base)
                color_map = {
                    "Green": [0, 255, 0],
                    "Red": [255, 0, 0],
                    "Blue": [0, 0, 255],
                    "Yellow": [255, 255, 0],
                }
                color = color_map.get(self.boundary_color.get(), [0, 255, 0])
                base_np[boundary_resized] = color
                base = Image.fromarray(base_np)

            found = len(np.unique(self.mask_array)) - 1
            selected = len(self.included_cells)
            self.stats_label.configure(
                text=f"Cell count:\nFound: {found}\nSelected: {selected}"
            )

        self.tk_image = CTkImage(dark_image=base, size=(1024, 1024))
        self.image_label.configure(image=self.tk_image, text="")

    def create_overlay_image(self):
        base = self.original_image.resize((1024, 1024)).convert("RGB")
        base_np = np.array(base)
        boundaries = self.get_exact_boundaries(self.mask_array)
        boundary_mask = np.zeros_like(boundaries)
        for cid in self.included_cells:
            boundary_mask |= boundaries & (self.mask_array == cid)
        boundary_resized = (
            np.array(
                Image.fromarray(boundary_mask.astype(np.uint8) * 255).resize(
                    (1024, 1024), resample=Image.NEAREST
                )
            )
            > 0
        )
        color_map = {
            "Green": [0, 255, 0],
            "Red": [255, 0, 0],
            "Blue": [0, 0, 255],
            "Yellow": [255, 255, 0],
        }
        color = color_map.get(self.boundary_color.get(), [0, 255, 0])
        base_np[boundary_resized] = color
        return Image.fromarray(base_np)

    def export_selected(self):
        if self.mask_array is None:
            CTkMessagebox(title="Error", message="No mask to export.", icon="cancel")
            return

        export_mask = np.zeros_like(self.mask_array, dtype=np.uint16)
        for cid in self.included_cells:
            export_mask[self.mask_array == cid] = cid

        save_path = filedialog.asksaveasfilename(
            initialfile=f"{self.base_filename}_selected_cells",
            defaultextension=".tif",
            filetypes=[("TIFF", "*.tif"), ("NumPy array", "*.npy")],
        )

        if save_path:
            try:
                if save_path.endswith(".npy"):
                    np.save(save_path, export_mask)
                else:
                    Image.fromarray(export_mask).save(save_path)
                CTkMessagebox(
                    title="Success", message="Export completed.", icon="check"
                )
            except Exception as e:
                CTkMessagebox(title="Error", message=str(e), icon="cancel")

    def export_pdf(self):
        if self.original_image is None or self.mask_array is None:
            CTkMessagebox(
                title="Error", message="Missing image or masks.", icon="cancel"
            )
            return

        save_path = filedialog.asksaveasfilename(
            initialfile=f"{self.base_filename}_report",
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
        )

        if not save_path:
            return

        try:
            c = canvas.Canvas(save_path, pagesize=A4)
            width, height = A4

            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 50, "Cell Segmentation Report")

            c.setFont("Helvetica", 12)
            found = len(np.unique(self.mask_array)) - 1
            selected = len(self.included_cells)
            dia = self.dia_entry.get()
            c.drawString(50, height - 80, f"Detected cells: {found}")
            c.drawString(50, height - 100, f"Selected cells: {selected}")
            c.drawString(50, height - 120, f"Diameter parameter: {dia}")

            orig_img = self.original_image.resize((512, 512))
            orig_buf = io.BytesIO()
            orig_img.save(orig_buf, format="PNG")
            c.drawImage(ImageReader(orig_buf), 50, height - 650, width=256, height=256)
            c.drawString(50, height - 660, "Original image")

            overlay_img = self.create_overlay_image().resize((512, 512))
            overlay_buf = io.BytesIO()
            overlay_img.save(overlay_buf, format="PNG")
            c.drawImage(
                ImageReader(overlay_buf), 320, height - 650, width=256, height=256
            )
            c.drawString(320, height - 660, "Overlay (boundaries)")

            c.save()
            CTkMessagebox(title="Success", message="PDF exported.", icon="check")

        except Exception as e:
            CTkMessagebox(title="Error", message=str(e), icon="cancel")
