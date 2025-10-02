import customtkinter as ctk
import numpy as np
import tifffile
from skimage import morphology
from skimage.measure import label, regionprops
import napari
from tkinter import messagebox


class NapariPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        box_frame = ctk.CTkFrame(self, corner_radius=10)
        box_frame.pack(pady=20)

        # Başlık buraya
        title_label = ctk.CTkLabel(box_frame, text="3D View Settings", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=(10,5), padx=10)

        # Açıklama
        desc_label = ctk.CTkLabel(box_frame, text="Adjust the parameters below to visualize the 3D aggregates.", font=ctk.CTkFont(size=12), text_color="gray")
        desc_label.pack(pady=(0,10), padx=10)

        form_frame = ctk.CTkFrame(box_frame, corner_radius=5)
        form_frame.pack(pady=(0,15), padx=10)

        # Satır 0 - Threshold
        threshold_label = ctk.CTkLabel(form_frame, text="Brightness Threshold (0-1):")
        threshold_label.grid(row=0, column=0, sticky="w", padx=5, pady=10)

        info_thresh_btn = ctk.CTkButton(form_frame, text="i", width=20, height=20, fg_color="gray", hover_color="darkgray",
                                        command=self.show_threshold_info)
        info_thresh_btn.grid(row=0, column=1, sticky="w", pady=10)

        self.threshold_entry = ctk.CTkEntry(form_frame, width=120)
        self.threshold_entry.insert(0, "0.6")
        self.threshold_entry.grid(row=0, column=2, sticky="w", padx=10, pady=10)

        # Satır 1 - Min Size
        min_size_label = ctk.CTkLabel(form_frame, text="Minimum Aggregate Size (voxels):")
        min_size_label.grid(row=1, column=0, sticky="w", padx=5, pady=10)

        info_min_size_btn = ctk.CTkButton(form_frame, text="i", width=20, height=20, fg_color="gray", hover_color="darkgray",
                                        command=self.show_min_size_info)
        info_min_size_btn.grid(row=1, column=1, sticky="w", pady=10)

        self.min_size_entry = ctk.CTkEntry(form_frame, width=120)
        self.min_size_entry.insert(0, "10")
        self.min_size_entry.grid(row=1, column=2, sticky="w", padx=10, pady=10)

        # Sütun genişliklerini ayarlayarak hizalama yapabiliriz
        form_frame.grid_columnconfigure(0, weight=1)  # Label sütunu esnek genişlesin
        form_frame.grid_columnconfigure(1, minsize=25)  # i butonu sütunu sabit küçük
        form_frame.grid_columnconfigure(2, minsize=130) # Entry sütunu sabit geniş

        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(pady=10)

        run_btn = ctk.CTkButton(button_frame, text="Run 3D", command=self.run_napari_3d, width=150, height=35)
        run_btn.pack(pady=(0, 5))  # alttan biraz boşluk

        back_button = ctk.CTkButton(button_frame, text="Back", command=lambda: self.controller.show_frame("ChannelInfoPage"), width=150, height=35)
        back_button.pack()

    def show_threshold_info(self):
        messagebox.showinfo("Brightness Threshold (0-1):", 
                            "Threshold Ratio: The brightness cutoff value used to identify aggregates in the image. "
                            "Pixels with intensity above this ratio (relative to the max brightness) are considered aggregates. "
                            "For example, 0.6 means pixels brighter than 60% of the max value are selected.")

    def show_min_size_info(self):
        messagebox.showinfo("Minimum Aggregate Size (voxels):", 
                            "Minimum Size: The smallest size (in voxels) for a region to be counted as an aggregate. "
                            "This helps filter out small noise spots that are not true aggregates.")
        

    def get_settings(self):
        try:
            threshold = float(self.threshold_entry.get())
            min_size = int(self.min_size_entry.get())
            if not (0 < threshold <= 1):
                raise ValueError
            if min_size < 1:
                raise ValueError
            return threshold, min_size
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid threshold (0-1) and minimum size (>0).")
            return None, None

    def process_aggregates(self, img, threshold_ratio, min_size):
        selections = getattr(self.controller, "selections", [])
        if not selections:
            aggregate_channel = 2  # default channel if not assigned
        else:
            try:
                aggregate_channel = selections.index("Aggregate")
            except ValueError:
                aggregate_channel = 2

        aggregate_masks = []
        for z in range(img.shape[0]):
            slice_img = img[z, aggregate_channel, :, :]
            threshold = threshold_ratio * np.max(slice_img)
            bright_spots = slice_img > threshold
            clean_mask = morphology.remove_small_objects(bright_spots, min_size=min_size)
            aggregate_masks.append(clean_mask)

        aggregate_stack = np.stack(aggregate_masks, axis=0)
        labeled_aggregates, num_aggregates = label(aggregate_stack, return_num=True)
        return labeled_aggregates, num_aggregates

    def run_napari_3d(self):
        img = getattr(self.controller, "img", None)
        if img is None:
            messagebox.showerror("Error", "No image data found.")
            return

        threshold_ratio, min_size = self.get_settings()
        if threshold_ratio is None:
            return  # input error

        labeled_aggregates, num_aggregates = self.process_aggregates(img, threshold_ratio, min_size)

        selections = getattr(self.controller, "selections", [])
        # Find nuclei and cell body channels with fallback defaults
        try:
            nuclei_channel = selections.index("Nuclei")
        except (ValueError, AttributeError):
            nuclei_channel = 0
        try:
            cell_body_channel = selections.index("Cell body")
        except (ValueError, AttributeError):
            cell_body_channel = 1

        img_nuclei = img[:, nuclei_channel, :, :]
        img_cellbody = img[:, cell_body_channel, :, :]

        regions = regionprops(labeled_aggregates)
        points = []
        labels = []

        for region in regions:
            z, y, x = region.centroid
            points.append([z, y, x])
            labels.append(str(region.label))

        points = np.array(points)

        viewer = napari.Viewer(ndisplay=3)
        viewer.add_image(img_nuclei, name="Nucleus", colormap="blue", scale=(1,1,1))
        viewer.add_image(img_cellbody, name="Cell Body", colormap="gray", blending="additive", scale=(1,1,1))
        viewer.add_points(points, name='Aggregate IDs', size=3, face_color='red', text=labels, scale=(1,1,1))
        viewer.add_labels(labeled_aggregates, name="Aggregates", opacity=0.9, scale=(1,1,1))

        napari.run()
