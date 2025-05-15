from tkinter import filedialog, messagebox
import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
import os

class cell_body_frame(ctk.CTkFrame):
    """
    Allows the user to select a microscopy image and detect cell bodies.
    """
    def __init__(self, parent):
        super().__init__(parent)

        self.parent=parent

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        """ TODO: Cell body detection stuff """

        self.folder_path=''
        self.data_path=''

        # Frame for folder and file selection
        data_location=ctk.CTkFrame(self)
        data_location.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        data_location.grid_columnconfigure(0, weight=1)

        # Label for folder selection
        folder_label=ctk.CTkLabel(master=data_location, text="Folder:")
        folder_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.btn_selectfolder=ctk.CTkButton(master=data_location, text="Select a folder", command=self.open_folder)
        self.btn_selectfolder.grid(row=0, column=1, padx=10, pady=10, sticky='nesw')

        # Label for file selection
        select_image_label=ctk.CTkLabel(master=data_location, text="File:")
        select_image_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')
        self.btn_selectrawfile=ctk.CTkButton(master=data_location, text="Select a file", command=self.open_image_file)
        self.btn_selectrawfile.grid(row=1, column=1, padx=10, pady=10, sticky='nesw')

        # Button to start the analysis (Does not work yet)
        start_analysis_button=ctk.CTkButton(self, text="Start analysis", command=self.start_analysis)
        start_analysis_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky='nesw')

        # Button to return to the start screen
        return_button=ctk.CTkButton(self, text="Return to start screen", command= lambda: parent.show_frame(self.parent.home_frame))
        return_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky='nesw')

    # Open a folder in the file dialog and show folder location inside the button
    def open_folder(self):
            results_folder = filedialog.askdirectory()
            if results_folder != '':
                self.btn_selectfolder.configure(text=results_folder)
                self.folder_path=results_folder
    
    # Open a .tif or .lif image in the file dialog and show image location inside the button
    def open_image_file(self):
            selected_file = filedialog.askopenfilename(filetypes=[("Microscopy Image", "*.tif")])
            if len(selected_file)!=0:
                self.btn_selectrawfile.configure(text=os.path.split(selected_file)[1])
                self.data_path=selected_file

    # TODO: This function can be used in the future to start the analysis
    def start_analysis(self):
        CTkMessagebox(title="Not available yet", message="This option is not available yet, \nbut is coming soon!")