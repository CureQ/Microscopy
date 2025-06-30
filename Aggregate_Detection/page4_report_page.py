import customtkinter as ctk
from tkinter import messagebox

class ReportPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.title = ctk.CTkLabel(self, text="Aggregate Quantification Report", font=ctk.CTkFont(size=20, weight="bold"))
        self.title.pack(pady=20)

        # Model parametreleri kısmı (örnek olarak)
        param_frame = ctk.CTkFrame(self)
        param_frame.pack(pady=10)

        ctk.CTkLabel(param_frame, text="Model Parameter 1:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.param1_entry = ctk.CTkEntry(param_frame)
        self.param1_entry.grid(row=0, column=1, padx=5, pady=5)

        ctk.CTkLabel(param_frame, text="Model Parameter 2:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.param2_entry = ctk.CTkEntry(param_frame)
        self.param2_entry.grid(row=1, column=1, padx=5, pady=5)

        # Analizi Başlat butonu
        self.start_analysis_btn = ctk.CTkButton(self, text="Analizi Başlat", command=self.start_analysis)
        self.start_analysis_btn.pack(pady=20)

        # PDF ve CSV İndir butonları (başlangıçta pasif)
        self.pdf_btn = ctk.CTkButton(self, text="PDF İndir", state="disabled", command=self.download_pdf)
        self.pdf_btn.pack(pady=10)

        self.csv_btn = ctk.CTkButton(self, text="CSV İndir", state="disabled", command=self.download_csv)
        self.csv_btn.pack(pady=10)

    def start_analysis(self):
        # Buraya Cellpose + SAM modelini çalıştırma kodunu entegre edeceksin
        # Örnek olarak burada mesaj gösterelim ve butonları aktif yapalım
        messagebox.showinfo("Bilgi", "Analiz başlatıldı (burada model çalışacak).")

        # Model çalışması bittikten sonra butonları aktif etmek için çağır
        self.enable_buttons()

    def enable_buttons(self):
        self.pdf_btn.configure(state="normal")
        self.csv_btn.configure(state="normal")

    def download_pdf(self):
        # PDF dosyasını indir (veya oluştur)
        messagebox.showinfo("PDF İndir", "PDF indiriliyor... (Burada dosya işlemi yapılacak)")

    def download_csv(self):
        # CSV dosyasını indir (veya oluştur)
        messagebox.showinfo("CSV İndir", "CSV indiriliyor... (Burada dosya işlemi yapılacak)")
