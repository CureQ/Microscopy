import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from readlif.reader import LifFile
import numpy as np

class ColocalizationApp:
    def __init__(self, master):
        self.master = master
        self.master.title("mHTT-CCT1 Colocalization Tool")
        self.master.geometry("1000x700")

        self.file_path = None
        self.channels = {"nucleus": None, "mHTT": None, "CCT1": None}
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.master, text="Load .lif File").grid(row=0, column=0, pady=10, sticky="w")
        tk.Button(self.master, text="Browse", command=self.load_file).grid(row=0, column=1)

        self.channel_frame = tk.LabelFrame(self.master, text="Select Channels")
        self.channel_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        tk.Label(self.channel_frame, text="Nucleus Channel:").grid(row=0, column=0)
        self.nucleus_var = tk.IntVar()
        tk.Entry(self.channel_frame, textvariable=self.nucleus_var, width=5).grid(row=0, column=1)

        tk.Label(self.channel_frame, text="mHTT Channel:").grid(row=1, column=0)
        self.mhtt_var = tk.IntVar()
        tk.Entry(self.channel_frame, textvariable=self.mhtt_var, width=5).grid(row=1, column=1)

        tk.Label(self.channel_frame, text="CCT1 Channel:").grid(row=2, column=0)
        self.cct1_var = tk.IntVar()
        tk.Entry(self.channel_frame, textvariable=self.cct1_var, width=5).grid(row=2, column=1)

        self.param_frame = tk.LabelFrame(self.master, text="Analysis Parameters")
        self.param_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        tk.Label(self.param_frame, text="Nucleus Diameter (px):").grid(row=0, column=0)
        self.nucleus_size = tk.IntVar(value=100)
        tk.Entry(self.param_frame, textvariable=self.nucleus_size, width=6).grid(row=0, column=1)

        tk.Label(self.param_frame, text="Particle Diameter (px, odd):").grid(row=1, column=0)
        self.particle_size = tk.IntVar(value=15)
        tk.Entry(self.param_frame, textvariable=self.particle_size, width=6).grid(row=1, column=1)

        tk.Label(self.param_frame, text="Gaussian Sigma:").grid(row=2, column=0)
        self.sigma = tk.IntVar(value=2)
        tk.Entry(self.param_frame, textvariable=self.sigma, width=6).grid(row=2, column=1)

        tk.Button(self.master, text="Run Analysis", command=self.run_analysis, bg="#57CAC8", fg="white").grid(row=3, column=0, pady=20)
        tk.Button(self.master, text="Exit", command=self.master.quit).grid(row=3, column=1)

        self.canvas_frame = tk.LabelFrame(self.master, text="Image Preview")
        self.canvas_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

    def load_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("LIF files", "*.lif")])
        print("Loaded file:", self.file_path)

    def run_analysis(self):
        if not self.file_path:
            print("❌ No file selected.")
            return

        lif_file = LifFile(self.file_path)
        image = lif_file.get_image(0)

        nucleus_ch = self.nucleus_var.get() - 1
        mhtt_ch = self.mhtt_var.get() - 1
        cct1_ch = self.cct1_var.get() - 1

        try:
            z_mhtt = [np.array(z) for z in image.get_iter_z(t=0, c=mhtt_ch)]
            z_cct1 = [np.array(z) for z in image.get_iter_z(t=0, c=cct1_ch)]
            print("✅ Loaded image. mHTT Z-stack:", len(z_mhtt), "CCT1 Z-stack:", len(z_cct1))
        except Exception as e:
            print("❌ Failed to extract z-stacks:", e)
            return

        mhtt_proj = np.maximum.reduce(z_mhtt)
        cct1_proj = np.maximum.reduce(z_cct1)

        self.display_image(mhtt_proj, cct1_proj)

    def display_image(self, mhtt_img, cct1_img):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        try:
            overlay = np.zeros((*mhtt_img.shape, 3), dtype=np.uint8)
            overlay[..., 0] = (mhtt_img / np.max(mhtt_img) * 255).astype(np.uint8)
            overlay[..., 1] = (cct1_img / np.max(cct1_img) * 255).astype(np.uint8)
        except Exception as e:
            print("❌ Image normalization error:", e)
            return

        try:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(overlay)
            ax.set_title("mHTT (Red) + CCT1 (Green)")
            ax.axis('off')

            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
        except Exception as e:
            print("❌ Matplotlib display error:", e)


if __name__ == '__main__':
    root = tk.Tk()
    app = ColocalizationApp(root)
    root.mainloop()