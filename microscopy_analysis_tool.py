# Imports
import os
import json
import sys
import webbrowser
from pathlib import Path
from tkinter import *

# External libraries
import customtkinter as ctk
from CTkToolTip import *
from CTkColorPicker import *

# Import GUI components
from Cell_Body_Detection._cell_body import cell_body_frame
from Aggregate_Detection._aggregate import aggregate_frame
from Colocalization._colocalization import colocalization_frame

# Main application with all themes
class MainApp(ctk.CTk):
    """
    Control frame selection and hold 'global' variables.
    """
    def __init__(self):
        # Initialize GUI
        super().__init__()

        # Get icon - works for both normal and frozen
        relative_path="MEAlytics_logo.ico"
        try:
            self.base_path = sys._MEIPASS
        except Exception:
            source_path = Path(__file__).resolve()
            self.base_path = source_path.parent
        self.icon_path=os.path.join(self.base_path, relative_path)
        try:
            self.iconbitmap(self.icon_path)
        except Exception as error:
            print("Could not load in icon")
            print(error)

        # 'Global' variables
        self.tooltipwraplength=200

        # Colors
        self.gray_1 = '#333333'
        self.gray_2 = '#2b2b2b'
        self.gray_3 = "#3f3f3f"
        self.gray_4 = "#212121"
        self.gray_5 = "#696969"
        self.gray_6 = "#292929"
        self.entry_gray = "#565b5e"
        self.text_color = '#dce4ee'
        self.selected_color = "#125722"     # Green color to show something is selected
        self.unselected_color = "#570700"   # Red color to show somehting is not selected

        # Set theme from json
        theme_path=os.path.join(self.base_path, "theme.json")
        
        with open(theme_path, "r") as json_file:
            self.theme = json.load(json_file)

        ctk.set_default_color_theme(theme_path)
        ctk.set_appearance_mode("dark")

        base_color = self.theme["CTkButton"]["fg_color"][1]
        self.primary_1 = self.mix_color(base_color, self.gray_6, factor=0.9)
        self.primary_1 = self.adjust_color(self.primary_1, 1.5)

        # Initialize main frame
        self.home_frame=main_window
        self.show_frame(self.home_frame)

        print("Successfully launched Microscopy GUI")


    # Handle frame switching
    def show_frame(self, frame_class, *args, **kwargs):
        for widget in self.winfo_children():
            widget.destroy()
        frame = frame_class(self, *args, **kwargs)
        frame.pack(expand=True, fill="both") 
    
    def adjust_color(self, hex_color, factor):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[:2], 16)
        g = int(hex_color[2:4], 16) 
        b = int(hex_color[4:], 16)
        
        r = int(min(255, max(0, r * factor)))
        g = int(min(255, max(0, g * factor)))
        b = int(min(255, max(0, b * factor)))
        
        return f'#{r:02x}{g:02x}{b:02x}'

    def mix_color(self, hex_color1, hex_color2, factor):
        # Convert hex colors to RGB
        hex_color1 = hex_color1.lstrip('#')
        hex_color2 = hex_color2.lstrip('#')
        
        # Original color RGB
        r1 = int(hex_color1[:2], 16)
        g1 = int(hex_color1[2:4], 16)
        b1 = int(hex_color1[4:], 16)
        
        # Gray color RGB
        r2 = int(hex_color2[:2], 16)
        g2 = int(hex_color2[2:4], 16)
        b2 = int(hex_color2[4:], 16)
        
        # Mix colors based on factor
        r = int(r1 * (1-factor) + r2 * factor)
        g = int(g1 * (1-factor) + g2 * factor)
        b = int(b1 * (1-factor) + b2 * factor)
        
        return f'#{r:02x}{g:02x}{b:02x}'

    def set_theme(self, base_color):
        theme_path=os.path.join(self.base_path, "theme.json")

        with open(theme_path, "r") as json_file:
            theme = json.load(json_file)
        
        # Edit all relevant widgets
        theme["CTkButton"]["fg_color"]=["#3a7ebf", base_color]
        theme["CTkButton"]["hover_color"]=["#325882", self.adjust_color(base_color, factor=0.6)]

        theme["CTkCheckBox"]["fg_color"]=["#3a7ebf", base_color]
        theme["CTkCheckBox"]["hover_color"]=["#325882", self.adjust_color(base_color, factor=0.6)]

        theme["CTkEntry"]["border_color"]=["#325882", self.mix_color(base_color, self.entry_gray, factor=0.8)]

        theme["CTkComboBox"]["border_color"]=["#325882", self.mix_color(base_color, self.entry_gray, factor=0.5)]
        theme["CTkComboBox"]["button_color"]=["#325882", base_color]
        theme["CTkComboBox"]["button_hover_color"]=["#325882", self.mix_color(base_color, self.entry_gray, factor=0.5)]

        theme["CTkOptionMenu"]["fg_color"]=["#325882", self.mix_color(base_color, self.entry_gray, factor=0.5)]
        theme["CTkOptionMenu"]["button_color"]=["#325882", base_color]
        theme["CTkOptionMenu"]["button_hover_color"]=["#325882", self.mix_color(base_color, self.entry_gray, factor=0.5)]
        
        theme["CTkSlider"]["button_color"]=[base_color, base_color]
        theme["CTkSlider"]["button_hover_color"]=[self.adjust_color(base_color, factor=0.6), self.adjust_color(base_color, factor=0.6)]

        # Tabview buttons
        theme["CTkSegmentedButton"]["selected_color"]=["#3a7ebf", base_color]
        theme["CTkSegmentedButton"]["selected_hover_color"]=["#325882", self.adjust_color(base_color, factor=0.6)]
        
        self.primary_1 = self.mix_color(base_color, self.gray_6, factor=0.9)
        self.primary_1 = self.adjust_color(self.primary_1, 1.5)

        with open(theme_path, 'w') as json_file:
            json.dump(theme, json_file, indent=4)
        ctk.set_default_color_theme(theme_path)

        self.theme=theme
        self.show_frame(self.home_frame)


class main_window(ctk.CTkFrame):
    """
    Main window and landing page for the user.
    Allows the user to switch to different frames to perform different tasks.
    """
    def __init__(self, parent):
        super().__init__(parent)

        self.parent=parent

        parent.title("CureQ Microscopy tool")

        # Weights
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Frame for the sidebar buttons
        sidebarframe=ctk.CTkFrame(self)
        sidebarframe.grid(row=0, column=1, padx=5, pady=5, sticky='nesw')

        # Switch themes
        theme_switch = ctk.CTkButton(sidebarframe, text="Theme", command=self.colorpicker)
        theme_switch.grid(row=0, column=0, sticky='nesw', pady=10, padx=10)
        self.selected_color=parent.theme["CTkButton"]["fg_color"][1]

        cureq_button=ctk.CTkButton(master=sidebarframe, text="CureQ project", command=lambda: webbrowser.open_new("https://cureq.nl/"))
        cureq_button.grid(row=1, column=0, sticky='nesw', pady=10, padx=10)

        github_button=ctk.CTkButton(master=sidebarframe, text="GitHub", command=lambda: webbrowser.open_new("https://github.com/CureQ/Microscopy/"))
        github_button.grid(row=2, column=0, sticky='nesw', pady=10, padx=10)

        # Main button frame
        main_buttons_frame=ctk.CTkFrame(self)
        main_buttons_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nesw')
        main_buttons_frame.grid_columnconfigure(0, weight=1)
        main_buttons_frame.grid_columnconfigure(1, weight=1)
        main_buttons_frame.grid_rowconfigure(0, weight=1)
        main_buttons_frame.grid_rowconfigure(1, weight=1)

        # Go to Cell body frame
        cell_body_button = ctk.CTkButton(master=main_buttons_frame, text="Cell Body detection\n\n(Noah Christian Le Roy)", command=lambda: parent.show_frame(cell_body_frame), height=90, width=160)
        cell_body_button.grid(row=0, column=0, sticky='nesw', pady=10, padx=10)

        # Go to Aggregate frame
        aggregate_button = ctk.CTkButton(master=main_buttons_frame, text="Aggregate detection\n\n(Yunus Demir)", command=lambda: parent.show_frame(aggregate_frame), height=90, width=160)
        aggregate_button.grid(row=0, column=1, sticky='nesw', pady=10, padx=10)

        # Go to Colocalization frame
        colocalization_button = ctk.CTkButton(master=main_buttons_frame, text="Colocalization\n\n(Noah Wijnheijmer)", command=lambda: parent.show_frame(colocalization_frame), height=90, width=160)
        colocalization_button.grid(row=0, column=2, sticky='nesw', pady=10, padx=10)

    def colorpicker(self):
        popup=ctk.CTkToplevel(self)
        popup.title('Theme Selector')

        try:
            popup.after(250, lambda: popup.iconbitmap(os.path.join(self.parent.icon_path)))
        except Exception as error:
            print(error)
        
        def set_theme():
            self.parent.set_theme(self.selected_color)
            popup.destroy()
            self.parent.show_frame(main_window)

        def set_color(color):
            self.selected_color=color

        popup.grid_columnconfigure(0, weight=1)
        popup.grid_rowconfigure(0, weight=1)
        popup.grid_rowconfigure(1, weight=1)
        colorpicker = CTkColorPicker(popup, width=350, command=lambda e: set_color(e), initial_color=self.selected_color)
        colorpicker.grid(row=0, column=0, sticky='nesw', padx=5, pady=5)
        confirm_button=ctk.CTkButton(master=popup, text="Confirm", command=set_theme)
        confirm_button.grid(row=1, column=0, sticky='nesw', padx=5, pady=5)



# Launch microscopy GUI
def Microscopy_GUI():
    """
    Launches the graphical user interface (GUI) of the microscopy tool.

    Always launch the function with an "if __name__ == '__main__':" guard as follows:
        if __name__ == "__main__":
            Microscopy_GUI()
    """

    app = MainApp()
    app.mainloop()


if __name__ == "__main__":
    Microscopy_GUI()