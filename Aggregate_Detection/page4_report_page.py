import customtkinter as ctk
import threading
import numpy as np
import os
import traceback
from tkinter import messagebox
from scipy.ndimage import label
from utils import process_channel, process_aggregates, plot_gray_overlay_with_nuclei, analyze_aggregates_3d_with_2d_shape_features

class ReportPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.diameter_nuclei_var = ctk.StringVar(value="25")
        self.diameter_cellbody_var = ctk.StringVar(value="100")

        ctk.CTkLabel(self, text="Quantification Report", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)

        form_frame = ctk.CTkFrame(self)
        form_frame.pack(pady=10)

        ctk.CTkLabel(form_frame, text="Diameter (Nuclei):").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkEntry(form_frame, textvariable=self.diameter_nuclei_var).grid(row=0, column=1, padx=10, pady=5)

        ctk.CTkLabel(form_frame, text="Diameter (Cell Body):").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkEntry(form_frame, textvariable=self.diameter_cellbody_var).grid(row=1, column=1, padx=10, pady=5)

        self.start_btn = ctk.CTkButton(self, text="Run Analysis", command=self.start_analysis)
        self.start_btn.pack(pady=20)

        self.progress_label = ctk.CTkLabel(self, text="Progress: Waiting...")
        self.progress_label.pack(pady=5)

        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=15)

        self.pdf_btn = ctk.CTkButton(btn_frame, text="PDF Output", state="disabled", command=self.create_pdf)
        self.pdf_btn.grid(row=0, column=0, padx=10)

        self.csv_btn = ctk.CTkButton(btn_frame, text="CSV File", state="disabled", command=self.create_csv)
        self.csv_btn.grid(row=0, column=1, padx=10)

        self.back_btn = ctk.CTkButton(self, text="Back", command=lambda: controller.show_frame("ChannelInfoPage"))
        self.back_btn.pack(pady=20)

        self.img = None
        self.nuclei_masks = None
        self.cellbody_masks = None
        self.aggregate_masks = None
        self.labeled_aggregates = None

        os.makedirs("output", exist_ok=True)

    def start_analysis(self):
        self.start_btn.configure(state="disabled")
        self.pdf_btn.configure(state="disabled")
        self.csv_btn.configure(state="disabled")
        self.progress_label.configure(text="Running...")

        self.img = self.controller.img
        self.selections = self.controller.selections

        threading.Thread(target=self.run_analysis).start()

    # def run_analysis(self):
    #     try:
    #         sel = self.selections
    #         nuclei_idx = sel.index("Nuclei")
    #         agg_idx = sel.index("Aggregate")
    #         cellbody_idx = sel.index("Cell body")

    #         d_nuclei = int(self.diameter_nuclei_var.get())
    #         d_cell = int(self.diameter_cellbody_var.get())

    #         # --- Nuclei
    #         self.progress_label.configure(text="🔄 Segmenting nuclei...")
    #         print("[INFO] Segmenting nuclei...")
    #         self.nuclei_masks = process_channel(self.img, nuclei_idx, d_nuclei)
    #         print(f"[DONE] Nuclei detected in {len(self.nuclei_masks)} z-slices ✅")

    #         self.progress_label.configure(text="✅ Nuclei segmentation done.")

    #         # --- Cell body
    #         self.progress_label.configure(text="🔄 Segmenting cell bodies...")
    #         print("[INFO] Segmenting cell bodies...")
    #         self.cellbody_masks = process_channel(self.img, cellbody_idx, d_cell)
    #         print(f"[DONE] Cell bodies detected in {len(self.cellbody_masks)} z-slices ✅")

    #         self.progress_label.configure(text="✅ Cell body segmentation done.")

    #         # --- Aggregates
    #         self.progress_label.configure(text="🔄 Detecting aggregates...")
    #         print("[INFO] Detecting aggregates...")
    #         self.aggregate_masks = process_aggregates(self.img, agg_idx)
    #         agg_stack = np.stack(self.aggregate_masks)
    #         self.labeled_aggregates, num = label(agg_stack)
    #         print(f"[DONE] Found {num} aggregate regions across {agg_stack.shape[0]} slices ✅")

    #         self.progress_label.configure(text="✅ All steps complete.")
    #         self.pdf_btn.configure(state="normal")
    #         self.csv_btn.configure(state="normal")

    #     except Exception as e:
    #         traceback.print_exc()
    #         messagebox.showerror("Error", str(e))
    #     finally:
    #         self.start_btn.configure(state="normal")
    
    def update_progress(self, msg):
        self.progress_label.configure(text=msg)
        self.progress_label.update_idletasks()


    def run_analysis(self):
        try:
            sel = self.selections
            nuclei_idx = sel.index("Nuclei")
            agg_idx = sel.index("Aggregate")
            cellbody_idx = sel.index("Cell body")

            d_nuclei = int(self.diameter_nuclei_var.get())
            d_cell = int(self.diameter_cellbody_var.get())

            # === NUCLEI ===
            self.update_progress("🔬 Segmenting nuclei...")
            self.nuclei_masks = process_channel(self.img, nuclei_idx, d_nuclei)
            self.update_progress("✅ Nuclei detected")

            # === CELL BODY ===
            self.update_progress("🔬 Segmenting cell bodies...")
            self.cellbody_masks = process_channel(self.img, cellbody_idx, d_cell)
            self.update_progress("✅ Cell bodies detected")

            # === AGGREGATES ===
            self.update_progress("🔍 Detecting aggregates...")
            self.aggregate_masks = process_aggregates(self.img, agg_idx)
            agg_stack = np.stack(self.aggregate_masks)
            self.labeled_aggregates, num_labels = label(agg_stack)

            self.update_progress(f"✅ {num_labels} aggregates found in {agg_stack.shape[0]} slices")

            # === AGGREGATE LABEL PROGRESS ===
            for i in range(1, num_labels + 1):
                self.update_progress(f"🧠 Analyzing aggregates: {i}/{num_labels}")
                self.progress_label.update_idletasks()
                # Optionally wait or simulate per-label work here
                # time.sleep(0.05)  # if needed to visualize progress step-by-step

            self.update_progress("✅ All processing complete.")
            self.pdf_btn.configure(state="normal")
            self.csv_btn.configure(state="normal")

        except Exception as e:
            self.update_progress("❌ Error occurred.")
            messagebox.showerror("Error", str(e))

        finally:
            self.start_btn.configure(state="normal")

    def create_pdf(self):
        pdf_path = "output/aggregate_report.pdf"
        plot_gray_overlay_with_nuclei(self.img, self.cellbody_masks, self.nuclei_masks, self.labeled_aggregates, output_path=pdf_path)
        messagebox.showinfo("PDF Created", f"PDF saved to: {pdf_path}")

    def create_csv(self):
        csv_path = "output/aggregate_report.csv"
        df = analyze_aggregates_3d_with_2d_shape_features(self.img, self.cellbody_masks, self.nuclei_masks, self.labeled_aggregates)
        df.to_csv(csv_path, index=False)
        messagebox.showinfo("CSV Created", f"CSV saved to: {csv_path}")
    