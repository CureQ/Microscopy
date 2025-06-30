import customtkinter as ctk
from page1_file_select import FileSelectPage
from page2_channel_info import ChannelInfoPage
from page3_napari_page import NapariPage
from page4_report_page import ReportPage

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Aggregate Detection UI")
        self.geometry("1000x700")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.frames = {}
        container = ctk.CTkFrame(self)
        container.pack(fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        for F in (FileSelectPage, ChannelInfoPage, NapariPage, ReportPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        
        self.show_frame("FileSelectPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

if __name__ == "__main__":
    app = App()
    app.mainloop()