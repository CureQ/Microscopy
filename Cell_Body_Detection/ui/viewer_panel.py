import customtkinter as ctk

from .. import constants
from ..utils.debug_logger import log
from .view_renderer import ImageViewRenderer


class ViewerPanel(ctk.CTkFrame):
    def __init__(self, parent, application_model, main_app_frame):
        super().__init__(parent)
        self.parent_frame = parent
        self.application_model = application_model
        self.main_app_frame = main_app_frame  # This is the cell_body_frame instance

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.image_canvas = ctk.CTkCanvas(
            self,
            bg=constants.COLOR_BLACK_STR,
            width=constants.UI_INITIAL_CANVAS_WIDTH,
            height=constants.UI_INITIAL_CANVAS_HEIGHT,
        )
        self.image_canvas.grid(row=0, column=0, sticky="nsew")

        self.image_view_renderer = ImageViewRenderer(
            self.image_canvas,
            self.application_model,
            self.main_app_frame,
        )

        self._bind_events()

    def _bind_events(self):
        log("ViewerPanel._bind_events execution started.", level="DEBUG")
        self.image_canvas.bind("<Button-1>", self.main_app_frame.handle_canvas_click)
        self.image_canvas.bind("<MouseWheel>", self.main_app_frame.handle_mouse_wheel)
        self.image_canvas.bind(
            "<Button-4>", self.main_app_frame.handle_mouse_wheel
        )  # Linux scroll up
        self.image_canvas.bind(
            "<Button-5>", self.main_app_frame.handle_mouse_wheel
        )  # Linux scroll down
        self.image_canvas.bind("<Configure>", self.main_app_frame.handle_canvas_resize)

    def get_canvas(self):
        return self.image_canvas

    def get_renderer(self):
        return self.image_view_renderer

    def update_display(self, quality="final"):
        """DEPRECATED: Should be called on main_app_frame"""
        log(
            "ViewerPanel.update_display is deprecated. Call on cell_body_frame.",
            level="WARNING",
        )
        self.main_app_frame.update_display(quality=quality)

    def get_canvas_dimensions(self):
        return self.image_canvas.winfo_width(), self.image_canvas.winfo_height()
