import customtkinter as ctk
import numpy as np
import tifffile
from PIL import Image
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib import cm
from customtkinter import CTkImage


class ChannelInfoPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.image_labels = []
        self.option_menus = []
        self.selections = []
        self.choices = ["Select", "Cell body", "Nuclei", "Aggregate", "Other"]

        self.title = ctk.CTkLabel(self, text="Channel Assignment", font=ctk.CTkFont(size=20, weight="bold"))
        self.title.pack(pady=10)

        self.image_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.image_frame.pack(pady=20)

        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(pady=30, padx=30, fill="x")

        button_width = 220
        button_height = 45

        self.view3d_btn = ctk.CTkButton(button_frame, text="3D View", width=button_width, height=button_height, command=self.validate_and_go_3d)
        self.view3d_btn.pack(pady=10)

        self.report_btn = ctk.CTkButton(button_frame, text="Quantification Report", width=button_width, height=button_height, command=self.validate_and_go_report)
        self.report_btn.pack(pady=10)

        self.back_btn = ctk.CTkButton(button_frame, text="Back", width=button_width, height=button_height, command=lambda: controller.show_frame("FileSelectPage"))
        self.back_btn.pack(pady=10)

    def validate_and_go_3d(self):
        if self.validate_and_proceed():
            self.controller.show_frame("NapariPage")

    def validate_and_go_report(self):
        if self.validate_and_proceed():
            self.controller.show_frame("ReportPage")


    def tk_image_from_array(self, arr):
        im = Image.fromarray(arr).resize((150, 150))
        return CTkImage(light_image=im, size=im.size)

    def tkraise(self):
        super().tkraise()
        self.load_images()

    def load_images(self):
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        self.image_labels.clear()
        self.option_menus.clear()
        self.selections.clear()

        tif_path = self.controller.tif_path
        img = tifffile.imread(tif_path)

        print("▶️ Image shape:", img.shape)
        Z, C, Y, X = img.shape
        C = img.shape[1] if img.ndim == 4 else 1  # Handle (Z, C, Y, X) or (Z, Y, X)
        mid = Z // 2

        def array_to_ctkimage_safe(img_slice, cmap_name='magma', target_size=(200, 200)):
            img = img_slice.astype(np.float32)
            img -= img.min()
            if img.max() > 0:
                img /= img.max()
            cmap = cm.get_cmap(cmap_name)
            colored = (cmap(img)[:, :, :3] * 255).astype(np.uint8)
            pil_img = Image.fromarray(colored).resize(target_size)
            return CTkImage(light_image=pil_img, size=target_size)

        default_assignments = ["Nuclei", "Other", "Aggregate", "Cell body"]
        # default_assignments = []

        for c in range(C):
            img_slice = img[mid, c]
            photo = array_to_ctkimage_safe(img_slice)

            frame = ctk.CTkFrame(self.image_frame)
            frame.grid(row=0, column=c, padx=10)

            ch_label = ctk.CTkLabel(frame, text=f"Channel {c}", font=ctk.CTkFont(size=14))
            ch_label.pack(pady=(0, 5))

            img_label = ctk.CTkLabel(frame, image=photo, text="")
            img_label.image = photo
            img_label.pack()

            default_value = default_assignments[c] if c < len(default_assignments) else "Select"
            menu_var = ctk.StringVar(value=default_value)
            menu = ctk.CTkOptionMenu(frame, variable=menu_var, values=self.choices)
            menu.pack(pady=5)

            self.image_labels.append(img_label)
            self.option_menus.append(menu)
            self.selections.append(menu_var)

    def validate_and_proceed(self):
        selected = [s.get() for s in self.selections]

        if "Select" in selected:
            messagebox.showerror("Error", "Please select a type for each channel.")
            return

        if len(selected) != len(set(selected)):
            messagebox.showerror("Error", "Each channel type can only be selected once.")
            return

        # Seçimi controller'a kaydet
        self.controller.selections = selected
        return True
        # self.controller.show_frame("NapariPage")
        # self.controller.show_frame("ReportPage")  # 🟢 dikkat: bu isim doğru olmalı

