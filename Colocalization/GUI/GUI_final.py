import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects
from scipy.stats import pearsonr, mannwhitneyu
import tifffile
from scipy.ndimage import label
import os
import datetime
import napari
from napari.utils.notifications import show_info

class ColocalizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Colocalization Analysis GUI")
        self.mhtt_path = None
        self.cct1_path = None
        self.mhtt_data = None
        self.cct1_data = None
        self.num_z_slices = 1
        self.mhtt_proj = None
        self.cct1_proj = None

        # Default parameters
        self.z_slice = 0
        self.mhtt_strictness = 4.0
        self.cct1_strictness = 4.5
        self.sigma = 3.5
        self.min_size = 20
        self.mhtt_color = 'green'
        self.cct1_color = 'red'
        self.show_overlap_only = False
        self.use_mip = True

        self.create_widgets()

    def create_widgets(self):
        # File selection
        file_frame = tk.Frame(self.root)
        file_frame.pack(pady=5)
        tk.Button(file_frame, text="Select mHTT file", command=self.load_mhtt).pack(side=tk.LEFT, padx=5)
        tk.Button(file_frame, text="Select CCT1 file", command=self.load_cct1).pack(side=tk.LEFT, padx=5)
        self.file_label = tk.Label(file_frame, text="No files loaded")
        self.file_label.pack(side=tk.LEFT, padx=10)

        # Top frame for parameter controls and output/plot
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.BOTH, expand=True)

        # Parameter controls on the left
        param_frame = tk.Frame(top_frame)
        param_frame.pack(side=tk.LEFT, anchor="n", padx=5, pady=5)

        self.z_slice_slider = tk.Scale(param_frame, from_=0, to=0, orient=tk.HORIZONTAL, label="Z-Slice", command=self.on_param_change)
        self.z_slice_slider.pack(fill=tk.X, padx=5)

        self.mhtt_strictness_slider = tk.Scale(param_frame, from_=1.0, to=10.0, resolution=0.5, orient=tk.HORIZONTAL, label="mHTT Factor", command=self.on_param_change)
        self.mhtt_strictness_slider.set(self.mhtt_strictness)
        self.mhtt_strictness_slider.pack(fill=tk.X, padx=5)

        self.cct1_strictness_slider = tk.Scale(param_frame, from_=1.0, to=10.0, resolution=0.5, orient=tk.HORIZONTAL, label="CCT1 Factor", command=self.on_param_change)
        self.cct1_strictness_slider.set(self.cct1_strictness)
        self.cct1_strictness_slider.pack(fill=tk.X, padx=5)

        self.sigma_slider = tk.Scale(param_frame, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, label="Gaussian σ", command=self.on_param_change)
        self.sigma_slider.set(self.sigma)
        self.sigma_slider.pack(fill=tk.X, padx=5)

        self.min_size_slider = tk.Scale(param_frame, from_=1, to=100, resolution=1, orient=tk.HORIZONTAL, label="Min Size", command=self.on_param_change)
        self.min_size_slider.set(self.min_size)
        self.min_size_slider.pack(fill=tk.X, padx=5)

        color_frame = tk.Frame(param_frame)
        color_frame.pack(fill=tk.X, pady=2)
        tk.Label(color_frame, text="mHTT Color:").pack(side=tk.LEFT)
        self.mhtt_color_var = tk.StringVar(value=self.mhtt_color)
        mhtt_color_menu = ttk.Combobox(color_frame, textvariable=self.mhtt_color_var, values=['red', 'green', 'blue', 'magenta', 'yellow', 'cyan'], width=8)
        mhtt_color_menu.pack(side=tk.LEFT, padx=5)
        mhtt_color_menu.bind("<<ComboboxSelected>>", self.on_param_change)

        tk.Label(color_frame, text="CCT1 Color:").pack(side=tk.LEFT)
        self.cct1_color_var = tk.StringVar(value=self.cct1_color)
        cct1_color_menu = ttk.Combobox(color_frame, textvariable=self.cct1_color_var, values=['red', 'green', 'blue', 'magenta', 'yellow', 'cyan'], width=8)
        cct1_color_menu.pack(side=tk.LEFT, padx=5)
        cct1_color_menu.bind("<<ComboboxSelected>>", self.on_param_change)

        self.overlap_var = tk.BooleanVar(value=self.show_overlap_only)
        overlap_check = tk.Checkbutton(param_frame, text="Show overlap only", variable=self.overlap_var, command=self.on_param_change)
        overlap_check.pack(anchor=tk.W)

        self.use_mip_var = tk.BooleanVar(value=self.use_mip)
        mip_check = tk.Checkbutton(param_frame, text="Use maximum intensity projection", variable=self.use_mip_var, command=self.on_param_change)
        mip_check.pack(anchor=tk.W)

        # Run analysis button below file_frame
        tk.Button(self.root, text="Run Colocalization Analysis", command=self.run_colocalization_analysis, bg="grey", fg="black").pack(pady=5)
        tk.Button(self.root, text="Show 3D Visualization", command=self.show_3d_visualization, bg="grey", fg="black").pack(pady=5)

        # Output and plot on the right of top_frame
        output_frame = tk.Frame(top_frame)
        output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        # Place matplotlib figure on top right
        self.fig = plt.Figure(figsize=(10, 5))
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=output_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # Output text below the plot
        self.output_text = tk.Text(output_frame, height=12, width=80)
        self.output_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def load_mhtt(self):
        path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif *.tiff")])
        if path:
            self.mhtt_path = path
            self.try_load_images()

    def load_cct1(self):
        path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif *.tiff")])
        if path:
            self.cct1_path = path
            self.try_load_images()

    def try_load_images(self):
        if self.mhtt_path and self.cct1_path:
            try:
                self.mhtt_data = tifffile.imread(self.mhtt_path)
                self.cct1_data = tifffile.imread(self.cct1_path)
                self.num_z_slices = self.mhtt_data.shape[0]
                self.mhtt_proj = np.max(self.mhtt_data, axis=0)
                self.cct1_proj = np.max(self.cct1_data, axis=0)
                self.z_slice_slider.config(to=self.num_z_slices-1)
                self.file_label.config(text=f"Loaded: {os.path.basename(self.mhtt_path)}, {os.path.basename(self.cct1_path)}")
                self.on_param_change()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load images: {e}")

    def on_param_change(self, event=None):
        if self.mhtt_data is None or self.cct1_data is None:
            return
        self.z_slice = self.z_slice_slider.get()
        self.mhtt_strictness = self.mhtt_strictness_slider.get()
        self.cct1_strictness = self.cct1_strictness_slider.get()
        self.sigma = self.sigma_slider.get()
        self.min_size = self.min_size_slider.get()
        self.mhtt_color = self.mhtt_color_var.get()
        self.cct1_color = self.cct1_color_var.get()
        self.show_overlap_only = self.overlap_var.get()
        self.use_mip = self.use_mip_var.get()
        self.update_visualization()

    def update_visualization(self):
        # Choose between MIP and z-slice
        if self.use_mip:
            mhtt_slice = self.mhtt_proj
            cct1_slice = self.cct1_proj
            slice_title = "Maximum Projection"
        else:
            mhtt_slice = self.mhtt_data[self.z_slice]
            cct1_slice = self.cct1_data[self.z_slice]
            slice_title = f"Z-Slice {self.z_slice}"

        # Normalize
        mhtt_norm_slice = (mhtt_slice - np.min(mhtt_slice)) / np.ptp(mhtt_slice)
        cct1_norm_slice = (cct1_slice - np.min(cct1_slice)) / np.ptp(cct1_slice)

        # Gaussian blur
        mhtt_smooth = gaussian(mhtt_norm_slice, sigma=self.sigma)
        cct1_smooth = gaussian(cct1_norm_slice, sigma=self.sigma)

        # Otsu threshold
        mhtt_otsu = threshold_otsu(mhtt_smooth)
        cct1_otsu = threshold_otsu(cct1_smooth)
        mhtt_threshold = min(1.0, mhtt_otsu * self.mhtt_strictness)
        cct1_threshold = min(1.0, cct1_otsu * self.cct1_strictness)

        # Masks
        mhtt_mask = mhtt_smooth > mhtt_threshold
        cct1_mask = cct1_smooth > cct1_threshold
        mhtt_mask = remove_small_objects(mhtt_mask, min_size=self.min_size)
        cct1_mask = remove_small_objects(cct1_mask, min_size=self.min_size)
        overlap_mask = mhtt_mask & cct1_mask
        # Count number of colocalized aggregates
        labeled_overlap, total_colocalized_aggregates = label(overlap_mask)
        num_colocalized_aggregates = total_colocalized_aggregates if self.show_overlap_only else 0

        # Composite image
        composite = np.zeros((mhtt_norm_slice.shape[0], mhtt_norm_slice.shape[1], 3))
        color_map = {'red': 0, 'green': 1, 'blue': 2, 'magenta': [0,2], 'yellow': [0,1], 'cyan': [1,2]}
        if not self.show_overlap_only:
            # mHTT
            if self.mhtt_color in ['red', 'green', 'blue']:
                composite[:,:,color_map[self.mhtt_color]] = mhtt_norm_slice * mhtt_mask
            else:
                for channel in color_map[self.mhtt_color]:
                    composite[:,:,channel] = mhtt_norm_slice * mhtt_mask
            # CCT1
            if self.cct1_color in ['red', 'green', 'blue']:
                composite[:,:,color_map[self.cct1_color]] = cct1_norm_slice * cct1_mask
            else:
                for channel in color_map[self.cct1_color]:
                    composite[:,:,channel] = cct1_norm_slice * cct1_mask
        else:
            if self.mhtt_color in ['red', 'green', 'blue'] and self.cct1_color in ['red', 'green', 'blue']:
                composite[:,:,color_map[self.mhtt_color]] = mhtt_norm_slice * overlap_mask
                composite[:,:,color_map[self.cct1_color]] = cct1_norm_slice * overlap_mask

        # Create raw overlay image (no thresholding, just colored channels)
        overlay_img = np.zeros((mhtt_norm_slice.shape[0], mhtt_norm_slice.shape[1], 3))
        color_map = {'red': 0, 'green': 1, 'blue': 2, 'magenta': [0,2], 'yellow': [0,1], 'cyan': [1,2]}
        # mHTT
        if self.mhtt_color in ['red', 'green', 'blue']:
            overlay_img[:,:,color_map[self.mhtt_color]] = mhtt_norm_slice
        else:
            for channel in color_map[self.mhtt_color]:
                overlay_img[:,:,channel] = mhtt_norm_slice
        # CCT1
        if self.cct1_color in ['red', 'green', 'blue']:
            overlay_img[:,:,color_map[self.cct1_color]] += cct1_norm_slice
        else:
            for channel in color_map[self.cct1_color]:
                overlay_img[:,:,channel] += cct1_norm_slice
        overlay_img = np.clip(overlay_img, 0, 1)

        # Plot thresholded composite (LEFT panel)
        self.ax1.clear()
        self.ax1.imshow(composite)
        title = f"{slice_title} - Thresholds: mHTT>{mhtt_threshold:.2f}, CCT1>{cct1_threshold:.2f}"
        if self.show_overlap_only:
            title += " (Overlap Only)"
        self.ax1.set_title(title, fontsize=10)
        self.ax1.axis('off')

        # Plot raw overlay (RIGHT panel, always ax2)
        self.ax2.clear()
        self.ax2.imshow(overlay_img)
        self.ax2.set_title(f"{slice_title} - Raw Overlay", fontsize=10)
        self.ax2.axis('off')

        self.fig.tight_layout()
        self.canvas.draw()

        # Metrics
        mhtt_area = np.sum(mhtt_mask)
        cct1_area = np.sum(cct1_mask)
        overlap_area = np.sum(overlap_mask)
        overlap_pct_mhtt = 100*overlap_area/mhtt_area if mhtt_area > 0 else 0
        overlap_pct_cct1 = 100*overlap_area/cct1_area if cct1_area > 0 else 0

        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, f"mHTT area: {mhtt_area} pixels\n")
        self.output_text.insert(tk.END, f"CCT1 area: {cct1_area} pixels\n")
        self.output_text.insert(tk.END, f"Overlap area: {overlap_area} pixels\n")
        self.output_text.insert(tk.END, f"Overlap percentage: {overlap_pct_mhtt:.2f}% of mHTT, {overlap_pct_cct1:.2f}% of CCT1\n")
        self.output_text.insert(tk.END, f"Colocalized aggregates: {num_colocalized_aggregates}\n")

        # Store for analysis
        self._last_masks = (mhtt_mask, cct1_mask, overlap_mask, mhtt_threshold, cct1_threshold, mhtt_norm_slice, cct1_norm_slice, slice_title)

    def run_colocalization_analysis(self):
        if not hasattr(self, "_last_masks"):
            messagebox.showwarning("Warning", "Please load images and update visualization first.")
            return
        mhtt_mask, cct1_mask, overlap_mask, mhtt_threshold, cct1_threshold, mhtt_norm_slice, cct1_norm_slice, slice_title = self._last_masks
        union_mask = mhtt_mask | cct1_mask

        # Pearson's
        mhtt_values = mhtt_norm_slice[union_mask]
        cct1_values = cct1_norm_slice[union_mask]
        pearson_corr, pearson_p = pearsonr(mhtt_values, cct1_values) if len(mhtt_values) > 1 else (0, 1)

        # Manders'
        m1 = np.sum(mhtt_norm_slice * overlap_mask) / np.sum(mhtt_norm_slice * mhtt_mask) if np.sum(mhtt_norm_slice * mhtt_mask) > 0 else 0
        m2 = np.sum(cct1_norm_slice * overlap_mask) / np.sum(cct1_norm_slice * cct1_mask) if np.sum(cct1_norm_slice * cct1_mask) > 0 else 0

        # Dice
        dice = 2 * np.sum(overlap_mask) / (np.sum(mhtt_mask) + np.sum(cct1_mask)) if (np.sum(mhtt_mask) + np.sum(cct1_mask)) > 0 else 0

        # Mann-Whitney U
        mhtt_inside = mhtt_norm_slice[mhtt_mask & ~cct1_mask].flatten()
        mhtt_overlap = mhtt_norm_slice[overlap_mask].flatten()
        if len(mhtt_inside) > 0 and len(mhtt_overlap) > 0:
            u_stat, p_value = mannwhitneyu(mhtt_inside, mhtt_overlap, alternative='two-sided')
        else:
            u_stat, p_value = 0, 1

        # Cohen's d
        if len(mhtt_inside) > 0 and len(mhtt_overlap) > 0:
            mean_in = np.mean(mhtt_overlap)
            mean_out = np.mean(mhtt_inside)
            std_in = np.std(mhtt_overlap)
            std_out = np.std(mhtt_inside)
            pooled_std = np.sqrt(((len(mhtt_overlap) - 1) * std_in**2 + (len(mhtt_inside) - 1) * std_out**2) / 
                                 (len(mhtt_overlap) + len(mhtt_inside) - 2)) if (len(mhtt_overlap) + len(mhtt_inside) - 2) > 0 else 0
            cohen_d = abs(mean_in - mean_out) / pooled_std if pooled_std > 0 else 0
        else:
            mean_in, mean_out, cohen_d = 0, 0, 0

        # IoU
        iou = np.sum(overlap_mask) / np.sum(union_mask) if np.sum(union_mask) > 0 else 0

        # Output
        self.output_text.insert(tk.END, "\n===== COLOCALIZATION ANALYSIS RESULTS =====\n")
        self.output_text.insert(tk.END, f"Parameters: mHTT factor={self.mhtt_strictness}, CCT1 factor={self.cct1_strictness}, sigma={self.sigma}, min_size={self.min_size}\n")
        self.output_text.insert(tk.END, f"Thresholds: mHTT>{mhtt_threshold:.4f}, CCT1>{cct1_threshold:.4f}\n")
        self.output_text.insert(tk.END, f"1. Pearson's Correlation: r = {pearson_corr:.4f} (p={pearson_p:.2e})\n")
        self.output_text.insert(tk.END, f"2. Manders' Coefficients: M1={m1:.4f}, M2={m2:.4f}\n")
        self.output_text.insert(tk.END, f"3. Dice Similarity: {dice:.4f}\n")
        self.output_text.insert(tk.END, f"4. Mann-Whitney U: U={u_stat:.2f}, p={p_value:.2e}\n")
        self.output_text.insert(tk.END, f"5. Cohen's d: d={cohen_d:.4f}, mean_in={mean_in:.2f}, mean_out={mean_out:.2f}\n")
        self.output_text.insert(tk.END, f"6. Intersection over Union (IoU): {iou:.4f}\n")

        # Compute full 3D masks for Napari visualization
        self.mhtt_mask_3d = np.zeros_like(self.mhtt_data, dtype=bool)
        self.cct1_mask_3d = np.zeros_like(self.cct1_data, dtype=bool)
        for z in range(self.num_z_slices):
            # Normalize slices
            mhtt_slice = self.mhtt_data[z]
            cct1_slice = self.cct1_data[z]
            mhtt_norm_slice = (mhtt_slice - np.min(mhtt_slice)) / np.ptp(mhtt_slice)
            cct1_norm_slice = (cct1_slice - np.min(cct1_slice)) / np.ptp(cct1_slice)
            # Gaussian blur
            mhtt_smooth = gaussian(mhtt_norm_slice, sigma=self.sigma)
            cct1_smooth = gaussian(cct1_norm_slice, sigma=self.sigma)
            # Otsu threshold
            mhtt_otsu = threshold_otsu(mhtt_smooth)
            cct1_otsu = threshold_otsu(cct1_smooth)
            mhtt_threshold = min(1.0, mhtt_otsu * self.mhtt_strictness)
            cct1_threshold = min(1.0, cct1_otsu * self.cct1_strictness)
            # Masks
            mhtt_mask_z = mhtt_smooth > mhtt_threshold
            cct1_mask_z = cct1_smooth > cct1_threshold
            mhtt_mask_z = remove_small_objects(mhtt_mask_z, min_size=self.min_size)
            cct1_mask_z = remove_small_objects(cct1_mask_z, min_size=self.min_size)
            # Store
            self.mhtt_mask_3d[z] = mhtt_mask_z
            self.cct1_mask_3d[z] = cct1_mask_z

        # PDF report
        report_dir = "colocalization_reports"
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = os.path.join(report_dir, f"colocalization_report_{timestamp}.pdf")

        # Visualization for PDF
        fig_results, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(mhtt_norm_slice, cmap='gray')
        axs[0].contour(mhtt_mask, colors='green', linewidths=0.5)
        axs[0].set_title(f"mHTT {slice_title}")
        axs[0].axis('off')
        axs[1].imshow(cct1_norm_slice, cmap='gray')
        axs[1].contour(cct1_mask, colors='red', linewidths=0.5)
        axs[1].set_title(f"CCT1 {slice_title}")
        axs[1].axis('off')
        overlap_img = np.zeros((mhtt_norm_slice.shape[0], mhtt_norm_slice.shape[1], 3))
        # Only show pixels where both masks overlap
        color_map = {'red': 0, 'green': 1, 'blue': 2, 'magenta': [0,2], 'yellow': [0,1], 'cyan': [1,2]}
        if self.mhtt_color in ['red', 'green', 'blue']:
            overlap_img[:, :, color_map[self.mhtt_color]] = mhtt_norm_slice * overlap_mask
        else:
            for channel in color_map[self.mhtt_color]:
                overlap_img[:, :, channel] = mhtt_norm_slice * overlap_mask
        if self.cct1_color in ['red', 'green', 'blue']:
            overlap_img[:, :, color_map[self.cct1_color]] = cct1_norm_slice * overlap_mask
        else:
            for channel in color_map[self.cct1_color]:
                overlap_img[:, :, channel] = cct1_norm_slice * overlap_mask
        overlap_img = np.clip(overlap_img, 0, 1)
        axs[2].imshow(overlap_img)
        axs[2].set_title("Colocalization Map")
        axs[2].axis('off')
        plt.tight_layout()

        with PdfPages(pdf_filename) as pdf:
            # Title page
            fig_title = plt.figure(figsize=(8.5, 11))
            fig_title.suptitle("Colocalization Analysis Report", fontsize=16, y=0.95)
            plt.figtext(0.5, 0.85, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                        ha='center', fontsize=12)
            plt.figtext(0.5, 0.80, f"mHTT Path: {os.path.basename(self.mhtt_path)}", ha='center', fontsize=10)
            plt.figtext(0.5, 0.77, f"CCT1 Path: {os.path.basename(self.cct1_path)}", ha='center', fontsize=10)
            plt.figtext(0.5, 0.72, f"Analysis Mode: {slice_title}", ha='center', fontsize=12, weight='bold')
            plt.figtext(0.5, 0.67, "Analysis Parameters:", ha='center', fontsize=12, weight='bold')
            plt.figtext(0.5, 0.63, f"mHTT threshold factor: {self.mhtt_strictness}", ha='center', fontsize=10)
            plt.figtext(0.5, 0.60, f"CCT1 threshold factor: {self.cct1_strictness}", ha='center', fontsize=10)
            plt.figtext(0.5, 0.57, f"Gaussian sigma: {self.sigma}", ha='center', fontsize=10)
            plt.figtext(0.5, 0.54, f"Min object size: {self.min_size} pixels", ha='center', fontsize=10)
            plt.figtext(0.5, 0.51, f"mHTT color: {self.mhtt_color}", ha='center', fontsize=10)
            plt.figtext(0.5, 0.48, f"CCT1 color: {self.cct1_color}", ha='center', fontsize=10)
            plt.axis('off')
            pdf.savefig(fig_title)
            plt.close(fig_title)

            # Results summary page
            fig_summary = plt.figure(figsize=(8.5, 11))
            fig_summary.suptitle("Colocalization Metrics Summary", fontsize=16, y=0.95)
            plt.figtext(0.1, 0.90, "Basic Metrics:", fontsize=12, weight='bold')
            plt.figtext(0.1, 0.87, f"mHTT area: {np.sum(mhtt_mask)} pixels", fontsize=10)
            plt.figtext(0.1, 0.84, f"CCT1 area: {np.sum(cct1_mask)} pixels", fontsize=10)
            plt.figtext(0.1, 0.81, f"Overlap area: {np.sum(overlap_mask)} pixels", fontsize=10)
            plt.figtext(0.1, 0.78, f"Overlap percentage: {100*np.sum(overlap_mask)/np.sum(mhtt_mask):.2f}% of mHTT, {100*np.sum(overlap_mask)/np.sum(cct1_mask):.2f}% of CCT1", fontsize=10)
            plt.figtext(0.1, 0.73, "1. Pearson's Correlation Coefficient:", fontsize=12, weight='bold')
            plt.figtext(0.1, 0.70, f"   r = {pearson_corr:.4f} (p-value: {pearson_p:.8e})", fontsize=10)
            plt.figtext(0.1, 0.65, "2. Manders' Coefficients:", fontsize=12, weight='bold')
            plt.figtext(0.1, 0.62, f"   M1 (fraction of mHTT overlapping with CCT1) = {m1:.4f}", fontsize=10)
            plt.figtext(0.1, 0.59, f"   M2 (fraction of CCT1 overlapping with mHTT) = {m2:.4f}", fontsize=10)
            plt.figtext(0.1, 0.54, "3. Dice Similarity Coefficient:", fontsize=12, weight='bold')
            plt.figtext(0.1, 0.51, f"   Dice = {dice:.4f}", fontsize=10)
            plt.figtext(0.1, 0.46, "4. Mann-Whitney U Test:", fontsize=12, weight='bold')
            plt.figtext(0.1, 0.43, f"   U = {u_stat:.2f}", fontsize=10)
            plt.figtext(0.1, 0.40, f"   p-value = {p_value:.8e}", fontsize=10)
            plt.figtext(0.1, 0.35, "5. Cohen's d (Effect Size):", fontsize=12, weight='bold')
            plt.figtext(0.1, 0.32, f"   d = {cohen_d:.4f}", fontsize=10)
            plt.figtext(0.1, 0.29, f"   Mean intensity in overlap: {mean_in:.2f}", fontsize=10)
            plt.figtext(0.1, 0.26, f"   Mean intensity outside overlap: {mean_out:.2f}", fontsize=10)
            plt.figtext(0.1, 0.21, "6. Intersection over Union (IoU):", fontsize=12, weight='bold')
            plt.figtext(0.1, 0.18, f"   IoU = {iou:.4f}", fontsize=10)
            plt.figtext(0.5, 0.05, f"Report saved to: {pdf_filename}", ha='center', fontsize=8, style='italic')
            plt.axis('off')
            pdf.savefig(fig_summary)
            plt.close(fig_summary)

            pdf.savefig(fig_results)
            plt.close(fig_results)

        self.output_text.insert(tk.END, f"\nPDF report generated and saved to: {pdf_filename}\n")

    def show_3d_visualization(self):
        try:
            viewer = napari.Viewer(ndisplay=3)

            # Add mHTT image
            if self.mhtt_data is not None:
                viewer.add_image(self.mhtt_data.astype(np.float32), name="mHTT", colormap="green", blending="additive")
                show_info("Added mHTT 3D image")

            # Add CCT1 image
            if self.cct1_data is not None:
                viewer.add_image(self.cct1_data.astype(np.float32), name="CCT1", colormap="red", blending="additive")
                show_info("Added CCT1 3D image")

            # Add masks if available
            if hasattr(self, 'mhtt_mask_3d'):
                viewer.add_labels(self.mhtt_mask_3d.astype(np.int32), name="mHTT Aggregates", opacity=0.5)
                show_info("Added mHTT 3D mask")
            if hasattr(self, 'cct1_mask_3d'):
                viewer.add_labels(self.cct1_mask_3d.astype(np.int32), name="CCT1 Aggregates", opacity=0.5)
                show_info("Added CCT1 3D mask")

            # Add overlay mask (overlap of mHTT and CCT1) if possible
            if hasattr(self, 'mhtt_mask_3d') and hasattr(self, 'cct1_mask_3d'):
                overlap_mask_3d = self.mhtt_mask_3d & self.cct1_mask_3d
                viewer.add_labels(overlap_mask_3d.astype(np.int32), name="Colocalization (Overlap)", opacity=0.8)
                show_info("Added Colocalization (Overlap) 3D mask")

        except Exception as e:
            print(f"Error in napari visualization: {e}")
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(self.mhtt_proj, cmap='gray')
            axs[0].set_title("mHTT Projection (fallback)")
            axs[1].imshow(self.cct1_proj, cmap='gray')
            axs[1].set_title("CCT1 Projection (fallback)")
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = ColocalizationApp(root)
    root.mainloop()
