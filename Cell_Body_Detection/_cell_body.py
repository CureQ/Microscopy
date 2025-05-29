import customtkinter as ctk
import numpy as np
from CTkMessagebox import CTkMessagebox

from . import constants
from ._display_settings_controller import DisplaySettingsController
from ._drawing_controller import DrawingController
from ._file_io_controller import FileIOController
from ._view_models_and_renderer import ImageViewRenderer
from .application_model import ApplicationModel
from .segmentation_processing import run_cellpose_segmentation


# --- Main Application Frame ---
class cell_body_frame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_app = parent

        # --- Initialize ApplicationModel ---
        self.application_model = ApplicationModel()

        # --- Initialize Controllers with ApplicationModel ---
        self.drawing_controller = DrawingController(self, self.application_model)
        self.display_settings_controller = DisplaySettingsController(
            self.application_model, self
        )
        self.file_io_controller = FileIOController(
            self, self.application_model, self.display_settings_controller
        )

        self.image_view_renderer = None

        # UI elements
        self.image_canvas = None
        self.undo_button = None
        self.redo_button = None
        self.filename_label = None
        self.segment_button = None
        self.draw_mask_button = None
        self.dia_entry = None  # For segmentation diameter input
        self.stats_label = None
        self.resolution_label = None

        self.brightness_slider = None
        self.contrast_slider = None
        self.colormap_optionmenu = None
        self.colormap_variable = ctk.StringVar(value="None")

        # --- UI State Variables (CTk Variables) ---
        self.show_original_var = ctk.BooleanVar(
            value=self.application_model.display_state.show_original_image
        )
        self.show_original_var.trace_add(
            "write",
            lambda *args: self.application_model.set_view_option(
                "show_original_image", self.show_original_var.get()
            ),
        )

        self.show_mask_var = ctk.BooleanVar(
            value=self.application_model.display_state.show_cell_masks
        )
        self.show_mask_var.trace_add(
            "write",
            lambda *args: self.application_model.set_view_option(
                "show_cell_masks", self.show_mask_var.get()
            ),
        )

        self.show_boundaries_var = ctk.BooleanVar(
            value=self.application_model.display_state.show_cell_boundaries
        )
        self.show_boundaries_var.trace_add(
            "write",
            lambda *args: self.application_model.set_view_option(
                "show_cell_boundaries", self.show_boundaries_var.get()
            ),
        )

        self.boundary_color_var = ctk.StringVar(
            value=self.application_model.display_state.boundary_color_name
        )
        self.boundary_color_var.trace_add(
            "write",
            lambda *args: self.application_model.set_view_option(
                "boundary_color_name", self.boundary_color_var.get()
            ),
        )

        self.show_only_deselected_var = ctk.BooleanVar(
            value=self.application_model.display_state.show_deselected_masks_only
        )
        self.show_only_deselected_var.trace_add(
            "write",
            lambda *args: self.application_model.set_view_option(
                "show_deselected_masks_only", self.show_only_deselected_var.get()
            ),
        )

        self.show_cell_numbers_var = ctk.BooleanVar(
            value=self.application_model.display_state.show_cell_numbers
        )
        self.show_cell_numbers_var.trace_add(
            "write",
            lambda *args: self.application_model.set_view_option(
                "show_cell_numbers", self.show_cell_numbers_var.get()
            ),
        )

        # PDF Export Options
        self.pdf_opt_masks_only_var = ctk.BooleanVar(
            value=self.application_model.display_state.pdf_opt_masks_only
        )
        self.pdf_opt_masks_only_var.trace_add(
            "write",
            lambda *args: self.application_model.set_view_option(
                "pdf_opt_masks_only", self.pdf_opt_masks_only_var.get()
            ),
        )

        self.pdf_opt_boundaries_only_var = ctk.BooleanVar(
            value=self.application_model.display_state.pdf_opt_boundaries_only
        )
        self.pdf_opt_boundaries_only_var.trace_add(
            "write",
            lambda *args: self.application_model.set_view_option(
                "pdf_opt_boundaries_only", self.pdf_opt_boundaries_only_var.get()
            ),
        )

        self.pdf_opt_numbers_only_var = ctk.BooleanVar(
            value=self.application_model.display_state.pdf_opt_numbers_only
        )
        self.pdf_opt_numbers_only_var.trace_add(
            "write",
            lambda *args: self.application_model.set_view_option(
                "pdf_opt_numbers_only", self.pdf_opt_numbers_only_var.get()
            ),
        )

        self.pdf_opt_masks_boundaries_var = ctk.BooleanVar(
            value=self.application_model.display_state.pdf_opt_masks_boundaries
        )
        self.pdf_opt_masks_boundaries_var.trace_add(
            "write",
            lambda *args: self.application_model.set_view_option(
                "pdf_opt_masks_boundaries", self.pdf_opt_masks_boundaries_var.get()
            ),
        )

        self.pdf_opt_masks_numbers_var = ctk.BooleanVar(
            value=self.application_model.display_state.pdf_opt_masks_numbers
        )
        self.pdf_opt_masks_numbers_var.trace_add(
            "write",
            lambda *args: self.application_model.set_view_option(
                "pdf_opt_masks_numbers", self.pdf_opt_masks_numbers_var.get()
            ),
        )

        self.pdf_opt_boundaries_numbers_var = ctk.BooleanVar(
            value=self.application_model.display_state.pdf_opt_boundaries_numbers
        )
        self.pdf_opt_boundaries_numbers_var.trace_add(
            "write",
            lambda *args: self.application_model.set_view_option(
                "pdf_opt_boundaries_numbers", self.pdf_opt_boundaries_numbers_var.get()
            ),
        )

        self.pdf_opt_masks_boundaries_numbers_var = ctk.BooleanVar(
            value=self.application_model.display_state.pdf_opt_masks_boundaries_numbers
        )
        self.pdf_opt_masks_boundaries_numbers_var.trace_add(
            "write",
            lambda *args: self.application_model.set_view_option(
                "pdf_opt_masks_boundaries_numbers",
                self.pdf_opt_masks_boundaries_numbers_var.get(),
            ),
        )

        self._last_canvas_width = 0
        self._last_canvas_height = 0
        self.is_interacting = False
        self.interactive_render_timer_id = None
        self.final_render_delay_ms = constants.INTERACTIVE_RENDER_DELAY_MS

        # --- Subscribe to ApplicationModel updates ---
        self.application_model.subscribe(self.handle_model_update)

        self._setup_ui()

        # Initialize ImageViewRenderer
        self.image_view_renderer = ImageViewRenderer(
            self.image_canvas,
            self.application_model,
            self,
        )

        self._bind_events()
        self.update_history_buttons()

    def handle_model_update(self, change_type: str | None = None):
        """
        Called by ApplicationModel when its state changes.
        Updates the UI elements of cell_body_frame accordingly.
        """
        print(
            f"cell_body_frame.handle_model_update received change_type: {change_type}"
        )

        if change_type in ["image_loaded", "image_load_failed"]:
            print(
                f"Handling '{change_type}': Finalizing image load and resetting view."
            )
            self._finalize_image_load_view_reset()  # Handles pan/zoom reset, resolution label, segment button
            if self.filename_label:
                new_filename = (
                    self.application_model.base_filename
                    if self.application_model.base_filename
                    else constants.UI_TEXT_NO_FILE_SELECTED
                )
                self.filename_label.configure(text=new_filename)
                print(f"Filename label updated to: {new_filename}")

        if change_type in [
            "segmentation_updated",
            "mask_updated_user_drawn",
            "cell_selection_changed",
            "model_restored_undo",
            "model_restored_redo",
        ]:
            print(
                f"Handling '{change_type}': Updating display (final quality) and stats."
            )
            self.update_display(quality="final")

        if (
            change_type == "display_settings_changed"
            or change_type == "display_settings_reset"
        ):
            # Update sliders and colormap menu to reflect model state
            print(
                f"Handling '{change_type}': Updating display settings UI and re-rendering (final quality)."
            )
            if self.brightness_slider:
                brightness_val = self.application_model.display_state.brightness
                self.brightness_slider.set(brightness_val)
                print(f"Brightness slider set to: {brightness_val}")
            if self.contrast_slider:
                contrast_val = self.application_model.display_state.contrast
                self.contrast_slider.set(contrast_val)
                print(f"Contrast slider set to: {contrast_val}")
            if self.colormap_variable:
                colormap_val = (
                    self.application_model.display_state.colormap_name
                    if self.application_model.display_state.colormap_name
                    else "None"
                )
                self.colormap_variable.set(colormap_val)
                print(f"Colormap variable set to: {colormap_val}")
            self.update_display(quality="final")

        if change_type == "view_options_changed":
            print(
                f"Handling '{change_type}': View options changed, updating display (final quality)."
            )
            self.update_display(quality="final")

        if change_type == "pan_zoom_updated" or change_type == "pan_zoom_reset":
            print(
                f"Handling '{change_type}': Pan/zoom updated. Renderer should handle its update."
            )

        if change_type in [
            "history_updated",
            "model_restored_undo",
            "model_restored_redo",
            "image_loaded",
            "segmentation_updated",
            "mask_updated_user_drawn",
        ]:
            self.update_history_buttons()
            print(f"Handling '{change_type}': History buttons updated.")
            self._update_stats_label()
            print("Stats label updated.")

        # Synchronize CTk variables with model state (in case model was changed programmatically)
        # This ensures UI elements reflect the true state of the ApplicationModel.

        # Display options
        if (
            self.show_original_var.get()
            != self.application_model.display_state.show_original_image
        ):
            self.show_original_var.set(
                self.application_model.display_state.show_original_image
            )
        if (
            self.show_mask_var.get()
            != self.application_model.display_state.show_cell_masks
        ):
            self.show_mask_var.set(self.application_model.display_state.show_cell_masks)
        if (
            self.show_boundaries_var.get()
            != self.application_model.display_state.show_cell_boundaries
        ):
            self.show_boundaries_var.set(
                self.application_model.display_state.show_cell_boundaries
            )
        if (
            self.boundary_color_var.get()
            != self.application_model.display_state.boundary_color_name
        ):
            self.boundary_color_var.set(
                self.application_model.display_state.boundary_color_name
            )
        if (
            self.show_only_deselected_var.get()
            != self.application_model.display_state.show_deselected_masks_only
        ):
            self.show_only_deselected_var.set(
                self.application_model.display_state.show_deselected_masks_only
            )
        if (
            self.show_cell_numbers_var.get()
            != self.application_model.display_state.show_cell_numbers
        ):
            self.show_cell_numbers_var.set(
                self.application_model.display_state.show_cell_numbers
            )

        # PDF export options
        if (
            self.pdf_opt_masks_only_var.get()
            != self.application_model.display_state.pdf_opt_masks_only
        ):
            self.pdf_opt_masks_only_var.set(
                self.application_model.display_state.pdf_opt_masks_only
            )
        if (
            self.pdf_opt_boundaries_only_var.get()
            != self.application_model.display_state.pdf_opt_boundaries_only
        ):
            self.pdf_opt_boundaries_only_var.set(
                self.application_model.display_state.pdf_opt_boundaries_only
            )
        if (
            self.pdf_opt_numbers_only_var.get()
            != self.application_model.display_state.pdf_opt_numbers_only
        ):
            self.pdf_opt_numbers_only_var.set(
                self.application_model.display_state.pdf_opt_numbers_only
            )
        if (
            self.pdf_opt_masks_boundaries_var.get()
            != self.application_model.display_state.pdf_opt_masks_boundaries
        ):
            self.pdf_opt_masks_boundaries_var.set(
                self.application_model.display_state.pdf_opt_masks_boundaries
            )
        if (
            self.pdf_opt_masks_numbers_var.get()
            != self.application_model.display_state.pdf_opt_masks_numbers
        ):
            self.pdf_opt_masks_numbers_var.set(
                self.application_model.display_state.pdf_opt_masks_numbers
            )
        if (
            self.pdf_opt_boundaries_numbers_var.get()
            != self.application_model.display_state.pdf_opt_boundaries_numbers
        ):
            self.pdf_opt_boundaries_numbers_var.set(
                self.application_model.display_state.pdf_opt_boundaries_numbers
            )
        if (
            self.pdf_opt_masks_boundaries_numbers_var.get()
            != self.application_model.display_state.pdf_opt_masks_boundaries_numbers
        ):
            self.pdf_opt_masks_boundaries_numbers_var.set(
                self.application_model.display_state.pdf_opt_masks_boundaries_numbers
            )

        if (
            self.dia_entry
        ):  # Sync diameter entry if it's part of the model state (e.g. after undo/redo)
            current_model_diameter = self.application_model.segmentation_diameter
            if self.dia_entry.get() != current_model_diameter:
                self.dia_entry.delete(0, ctk.END)
                self.dia_entry.insert(0, current_model_diameter)
                print(f"Diameter entry synced to model value: {current_model_diameter}")

        print(
            f"Finished processing cell_body_frame.handle_model_update for change_type: {change_type}"
        )

    def _setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.settings_panel = ctk.CTkScrollableFrame(
            self, width=constants.UI_SIDEPANEL_WIDTH
        )
        self.settings_panel.grid(row=0, column=0, sticky="ns", padx=(5, 0), pady=5)
        self.create_settings_panel_widgets()

        self.viewer_panel = ctk.CTkFrame(self)
        self.viewer_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.viewer_panel.grid_rowconfigure(0, weight=1)
        self.viewer_panel.grid_columnconfigure(0, weight=1)
        # Crucial: self.image_canvas is created here
        self.image_canvas = ctk.CTkCanvas(
            self.viewer_panel,
            bg=constants.COLOR_BLACK_STR,
            width=constants.UI_INITIAL_CANVAS_WIDTH,
            height=constants.UI_INITIAL_CANVAS_HEIGHT,
        )
        self.image_canvas.grid(row=0, column=0, sticky="nsew")

        self.output_panel = ctk.CTkScrollableFrame(
            self, width=constants.UI_SIDEPANEL_WIDTH
        )
        self.output_panel.grid(row=0, column=2, sticky="ns", padx=(0, 5), pady=5)
        self.create_output_panel_widgets()

    def create_settings_panel_widgets(self):
        # --- Title ---
        ctk.CTkLabel(
            self.settings_panel,
            text=constants.UI_TEXT_SETTINGS_TITLE,  # "Settings"
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(10, 15), padx=10, fill="x")

        # --- Navigation ---
        navigation_frame = ctk.CTkFrame(self.settings_panel, fg_color="transparent")
        navigation_frame.pack(padx=10, pady=(0, 10), fill="x")
        ctk.CTkLabel(
            navigation_frame,
            text=constants.UI_TEXT_NAVIGATION,
            font=ctk.CTkFont(weight="bold"),  # "Navigation"
        ).pack(anchor="w")

        home_frame_ref = getattr(self.parent_app, "home_frame", None)
        nav_cmd = (
            (lambda: self.parent_app.show_frame(home_frame_ref))
            if home_frame_ref
            else (lambda: print("Home frame not found"))
        )
        ctk.CTkButton(
            navigation_frame,
            text=constants.UI_TEXT_RETURN_TO_START,
            command=nav_cmd,  # "Return to Start Screen"
        ).pack(pady=(5, 0), fill="x")

        # --- History ---
        history_frame = ctk.CTkFrame(self.settings_panel, fg_color="transparent")
        history_frame.pack(padx=10, pady=(10, 10), fill="x")
        ctk.CTkLabel(
            history_frame,
            text=constants.UI_TEXT_HISTORY,
            font=ctk.CTkFont(weight="bold"),  # "History"
        ).pack(anchor="w")
        self.undo_button = ctk.CTkButton(
            history_frame,
            text=constants.UI_TEXT_UNDO,
            command=self._handle_undo_action,
        )
        self.undo_button.pack(pady=(5, 5), fill="x")
        self.redo_button = ctk.CTkButton(
            history_frame,
            text=constants.UI_TEXT_REDO,
            command=self._handle_redo_action,
        )
        self.redo_button.pack(pady=(0, 5), fill="x")

        # --- Import Settings ---
        import_frame = ctk.CTkFrame(self.settings_panel, fg_color="transparent")
        import_frame.pack(padx=10, pady=(10, 10), fill="x")
        ctk.CTkLabel(
            import_frame,
            text=constants.UI_TEXT_IMPORT_SETTINGS,
            font=ctk.CTkFont(weight="bold"),  # "Import Settings"
        ).pack(anchor="w")
        self.select_image_btn = ctk.CTkButton(
            import_frame,
            text=constants.UI_TEXT_SELECT_IMAGE,
            command=self.file_io_controller.load_image,
        )
        self.select_image_btn.pack(pady=(5, 5), fill="x")
        self.filename_label = ctk.CTkLabel(
            import_frame,
            text=constants.UI_TEXT_NO_FILE_SELECTED,  # Filename label updated by model notification
            wraplength=constants.UI_FILENAME_LABEL_WRAPLENGTH,
        )
        self.filename_label.pack(pady=(0, 5), fill="x")

        self.resolution_label = ctk.CTkLabel(
            import_frame,
            text="Resolution: N/A",  # Initial text, updated by model notification
            justify="left",
            anchor="w",
        )
        self.resolution_label.pack(padx=10, pady=(5, 0), fill="x", anchor="n")

        # --- Model Settings ---
        model_frame = ctk.CTkFrame(self.settings_panel, fg_color="transparent")
        model_frame.pack(padx=10, pady=(10, 10), fill="x")
        ctk.CTkLabel(
            model_frame,
            text=constants.UI_TEXT_MODEL_SETTINGS,
            font=ctk.CTkFont(weight="bold"),  # "Model Settings"
        ).pack(anchor="w")
        ctk.CTkLabel(model_frame, text=constants.UI_TEXT_DIAMETER_LABEL).pack(
            anchor="w", pady=(5, 0)
        )  # "Diameter:"

        def only_allow_integers(P):
            return P.isdigit() or P == ""

        vcmd = self.register(only_allow_integers)

        self.dia_entry = ctk.CTkEntry(
            model_frame, validate="key", validatecommand=(vcmd, "%P")
        )
        self.dia_entry.insert(
            0, self.application_model.segmentation_diameter
        )  # Get from model
        # Add trace to update model when dia_entry changes
        self.dia_entry.bind(
            "<FocusOut>",
            lambda event: self.application_model.update_segmentation_diameter(
                self.dia_entry.get()
            ),
        )
        self.dia_entry.bind(
            "<Return>",
            lambda event: self.application_model.update_segmentation_diameter(
                self.dia_entry.get()
            ),
        )

        self.dia_entry.pack(pady=(0, 5), fill="x")
        self.segment_button = ctk.CTkButton(
            model_frame,
            text=constants.UI_TEXT_SEGMENT_BUTTON,
            command=self.run_segmentation,
            state="disabled",  # "Segment"
        )
        print("Segment button initialized, state: disabled")
        self.segment_button.pack(pady=(5, 5), fill="x")

        self.draw_mask_button = ctk.CTkButton(
            model_frame,
            text=constants.UI_TEXT_START_DRAWING_BUTTON,
            command=self.drawing_controller._start_drawing_mode,
        )
        self.draw_mask_button.pack(pady=(0, 0), fill="x")

        # --- Display Adjustments ---
        adjust_frame = ctk.CTkFrame(self.settings_panel, fg_color="transparent")
        adjust_frame.pack(padx=10, pady=(10, 10), fill="x")
        ctk.CTkLabel(
            adjust_frame,
            text=constants.UI_TEXT_DISPLAY_ADJUSTMENTS,
            font=ctk.CTkFont(weight="bold"),
        ).pack(anchor="w")

        # Brightness
        ctk.CTkLabel(adjust_frame, text=constants.UI_TEXT_BRIGHTNESS).pack(
            anchor="w", pady=(5, 0)
        )  # Placeholder
        self.brightness_slider = ctk.CTkSlider(
            adjust_frame,
            from_=0.1,  # Min brightness (avoid 0 as it can make image black)
            to=2.0,  # Max brightness
            number_of_steps=190,  # (2.0 - 0.1) / 0.01 = 190 steps for 0.01 increments
            command=self._handle_brightness_change,
        )
        self.brightness_slider.set(1.0)  # Default brightness
        self.brightness_slider.pack(fill="x", pady=(0, 5))

        # Contrast
        ctk.CTkLabel(adjust_frame, text=constants.UI_TEXT_CONTRAST).pack(
            anchor="w", pady=(5, 0)
        )
        self.contrast_slider = ctk.CTkSlider(
            adjust_frame,
            from_=0.1,  # Min contrast
            to=2.0,  # Max contrast
            number_of_steps=190,  # (2.0 - 0.1) / 0.01 = 190 steps
            command=self._handle_contrast_change,
        )
        self.contrast_slider.set(1.0)  # Default contrast
        self.contrast_slider.pack(fill="x", pady=(0, 5))

        # Colormap
        ctk.CTkLabel(adjust_frame, text=constants.UI_TEXT_COLORMAP).pack(
            anchor="w", pady=(5, 0)
        )
        self.colormap_optionmenu = ctk.CTkOptionMenu(
            adjust_frame,
            variable=self.colormap_variable,  # Initialized in __init__
            values=self.display_settings_controller.get_available_colormaps(),  # Controller provides list
            command=self._handle_colormap_change,  # This will call model.set_colormap
        )
        self.colormap_optionmenu.pack(fill="x", pady=(0, 10))
        self.colormap_optionmenu.configure(
            values=self.display_settings_controller.get_available_colormaps()
        )

        # Reset Button
        self.reset_display_button = ctk.CTkButton(
            adjust_frame,
            text=constants.UI_TEXT_RESET_DISPLAY_SETTINGS,  # Placeholder
            command=self._reset_display_settings_and_ui,
        )
        self.reset_display_button.pack(fill="x", pady=(5, 5))

        # --- Display Options ---
        display_frame = ctk.CTkFrame(self.settings_panel, fg_color="transparent")
        display_frame.pack(
            padx=10, pady=(10, 10), fill="x", expand=True, anchor="s"
        )  # Push to bottom
        ctk.CTkLabel(
            display_frame,
            text=constants.UI_TEXT_DISPLAY_OPTIONS,
            font=ctk.CTkFont(weight="bold"),  # "Display Options"
        ).pack(anchor="w")

        ctk.CTkCheckBox(
            display_frame,
            text=constants.UI_TEXT_SHOW_ORIGINAL_IMAGE,
            variable=self.show_original_var,  # Use the new CTk var
        ).pack(anchor="w", pady=2)
        ctk.CTkCheckBox(
            display_frame,
            text=constants.UI_TEXT_SHOW_CELL_MASKS,
            variable=self.show_mask_var,  # Use the new CTk var
        ).pack(anchor="w", pady=2)
        ctk.CTkCheckBox(
            display_frame,
            text=constants.UI_TEXT_SHOW_CELL_BOUNDARIES,
            variable=self.show_boundaries_var,  # Use the new CTk var
        ).pack(anchor="w", pady=2)
        ctk.CTkCheckBox(
            display_frame,
            text=constants.UI_TEXT_SHOW_CELL_NUMBERS,
            variable=self.show_cell_numbers_var,  # Use the new CTk var
        ).pack(anchor="w", pady=2)
        ctk.CTkLabel(display_frame, text=constants.UI_TEXT_DISPLAY_MODE_LABEL).pack(
            anchor="w", pady=(5, 0)
        )
        deselected_switch = ctk.CTkSwitch(
            display_frame,
            text=constants.UI_TEXT_SHOW_DESELECTED_ONLY,
            variable=self.show_only_deselected_var,  # Use the new CTk var
        )
        deselected_switch.pack(anchor="w", pady=(2, 5), padx=10)
        ctk.CTkLabel(display_frame, text=constants.UI_TEXT_BOUNDARY_COLOR_LABEL).pack(
            anchor="w", pady=(5, 0)
        )
        ctk.CTkOptionMenu(
            display_frame,
            variable=self.boundary_color_var,  # Use the new CTk var
            values=constants.AVAILABLE_BOUNDARY_COLORS,
        ).pack(fill="x", pady=(0, 5))

    def create_output_panel_widgets(self):
        ctk.CTkLabel(
            self.output_panel,
            text=constants.UI_TEXT_OUTPUT_PANEL_TITLE,
            font=ctk.CTkFont(size=16, weight="bold"),  # "Output"
        ).pack(pady=10, padx=10, fill="x")

        self.stats_label = ctk.CTkLabel(
            self.output_panel,
            text=constants.UI_TEXT_STATS_LABEL_DEFAULT,
            justify="left",
            anchor="w",
        )
        self.stats_label.pack(padx=10, pady=(5, 10), fill="x", anchor="n")

        export_frame = ctk.CTkFrame(self.output_panel, fg_color="transparent")
        export_frame.pack(padx=10, pady=10, fill="x", expand=True, anchor="s")

        ctk.CTkLabel(
            export_frame,
            text=constants.UI_TEXT_PDF_EXPORT_OPTIONS_LABEL,
            wraplength=constants.UI_FILENAME_LABEL_WRAPLENGTH,
        ).pack(  # "PDF Export Options (select multiple):"
            anchor="w", pady=(0, 2)
        )

        self.pdf_cb_masks_only = ctk.CTkCheckBox(
            export_frame,
            text=constants.UI_TEXT_PDF_MASKS_ONLY,
            variable=self.pdf_opt_masks_only_var,  # Use new CTk var
        )
        self.pdf_cb_masks_only.pack(anchor="w", pady=1)

        self.pdf_cb_boundaries_only = ctk.CTkCheckBox(
            export_frame,
            text=constants.UI_TEXT_PDF_BOUNDARIES_ONLY,
            variable=self.pdf_opt_boundaries_only_var,  # Use new CTk var
        )
        self.pdf_cb_boundaries_only.pack(anchor="w", pady=1)

        self.pdf_cb_numbers_only = ctk.CTkCheckBox(
            export_frame,
            text=constants.UI_TEXT_PDF_NUMBERS_ONLY,
            variable=self.pdf_opt_numbers_only_var,  # Use new CTk var
        )
        self.pdf_cb_numbers_only.pack(anchor="w", pady=1)

        self.pdf_cb_masks_boundaries = ctk.CTkCheckBox(
            export_frame,
            text=constants.UI_TEXT_PDF_MASKS_BOUNDARIES,
            variable=self.pdf_opt_masks_boundaries_var,  # Use new CTk var
        )
        self.pdf_cb_masks_boundaries.pack(anchor="w", pady=1)

        self.pdf_cb_masks_numbers = ctk.CTkCheckBox(
            export_frame,
            text=constants.UI_TEXT_PDF_MASKS_NUMBERS,
            variable=self.pdf_opt_masks_numbers_var,  # Use new CTk var
        )
        self.pdf_cb_masks_numbers.pack(anchor="w", pady=1)

        self.pdf_cb_boundaries_numbers = ctk.CTkCheckBox(
            export_frame,
            text=constants.UI_TEXT_PDF_BOUNDARIES_NUMBERS,
            variable=self.pdf_opt_boundaries_numbers_var,  # Use new CTk var
        )
        self.pdf_cb_boundaries_numbers.pack(anchor="w", pady=1)

        self.pdf_cb_masks_boundaries_numbers = ctk.CTkCheckBox(
            export_frame,
            text=constants.UI_TEXT_PDF_MASKS_BOUNDARIES_NUMBERS,
            variable=self.pdf_opt_masks_boundaries_numbers_var,  # Use new CTk var
        )
        self.pdf_cb_masks_boundaries_numbers.pack(anchor="w", pady=(1, 5))

        ctk.CTkButton(
            export_frame,
            text=constants.UI_TEXT_EXPORT_SELECTED_CELLS_BUTTON,
            command=self.file_io_controller.export_selected,
        ).pack(fill="x", pady=2)
        ctk.CTkButton(
            export_frame,
            text=constants.UI_TEXT_EXPORT_VIEW_AS_TIF_BUTTON,
            command=self.file_io_controller.export_current_view_as_tif,
        ).pack(fill="x", pady=2)
        ctk.CTkButton(
            export_frame,
            text=constants.UI_TEXT_EXPORT_PDF_REPORT_BUTTON,
            command=self.file_io_controller.export_pdf,
        ).pack(fill="x", pady=2)

    def _bind_events(self):
        self.image_canvas.bind("<Button-1>", self.handle_canvas_click)
        self.image_canvas.bind("<MouseWheel>", self.handle_mouse_wheel)
        self.image_canvas.bind("<Button-4>", self.handle_mouse_wheel)
        self.image_canvas.bind("<Button-5>", self.handle_mouse_wheel)

        self.image_canvas.bind("<Configure>", self.handle_canvas_resize)

        # Define common key binding logic
        def bind_keys(binder):
            binder("<Command-z>", self._handle_undo_shortcut)
            binder("<Control-z>", self._handle_undo_shortcut)
            binder("<Command-Shift-Z>", self._handle_redo_shortcut)
            binder("<Control-Shift-Z>", self._handle_redo_shortcut)
            binder("<Control-y>", self._handle_redo_shortcut)

            # Keyboard shortcuts for Zooming
            binder("<Key-plus>", self._handle_zoom_in_key)
            binder("<Key-equal>", self._handle_zoom_in_key)
            binder("<Key-minus>", self._handle_zoom_out_key)

            # Keyboard shortcuts for Panning
            binder("<Left>", self._handle_pan_left_key)
            binder("<Right>", self._handle_pan_right_key)
            binder("<Up>", self._handle_pan_up_key)
            binder("<Down>", self._handle_pan_down_key)

            # Shortcut for canceling drawing
            binder("<Escape>", self.drawing_controller._cancel_drawing_action)
            # Shortcut for finalizing drawing
            binder("<Return>", self.drawing_controller._handle_enter_key_press)
            binder("<KP_Enter>", self.drawing_controller._handle_enter_key_press)

        if hasattr(self.parent_app, "bind_all"):
            bind_keys(self.parent_app.bind_all)
            # Bind global click for focus management after other key binds
            self.parent_app.bind_all(
                "<Button-1>", self._handle_global_click_for_focus, add="+"
            )
        else:
            toplevel = self.winfo_toplevel()
            bind_keys(toplevel.bind_all)
            # Bind global click for focus management after other key binds
            toplevel.bind("<Button-1>", self._handle_global_click_for_focus, add="+")

    def update_history_buttons(self):
        can_undo_flag = False
        can_redo_flag = False

        if self.drawing_controller.drawing_mode_active:
            can_undo_flag = self.drawing_controller.can_undo_draw()
            can_redo_flag = self.drawing_controller.can_redo_draw()
        else:
            if self.application_model:  # Check if application_model exists
                can_undo_flag = self.application_model.can_undo()
                can_redo_flag = self.application_model.can_redo()

        if self.undo_button:
            self.undo_button.configure(state="normal" if can_undo_flag else "disabled")
            print(
                f"Undo button state set to: {'normal' if can_undo_flag else 'disabled'}"
            )
        if self.redo_button:
            self.redo_button.configure(state="normal" if can_redo_flag else "disabled")
            print(
                f"Redo button state set to: {'normal' if can_redo_flag else 'disabled'}"
            )

    def run_segmentation(self):
        if not self.application_model.image_data.original_image:
            print("Segmentation run skipped: No original image in model.")
            CTkMessagebox(
                title=constants.MSG_SEGMENTATION_ERROR_TITLE,
                message=constants.MSG_NO_IMAGE_FOR_SEGMENTATION,
                icon="warning",
            )
            return
        try:
            if self.drawing_controller.drawing_mode_active:
                print(
                    "Drawing mode active during segmentation run, stopping drawing mode."
                )
                self.drawing_controller._stop_drawing_mode()

            # Get image from ApplicationModel for segmentation
            img_for_segmentation = (
                self.application_model.image_data.original_image.convert("L")
            )
            img_np = np.array(img_for_segmentation)

            # Get diameter from ApplicationModel (synced from dia_entry)
            dia_text = self.application_model.segmentation_diameter
            dia = float(dia_text) if dia_text and dia_text.isdigit() else None
            print(f"Running segmentation with diameter: {dia}")

            # Run Cellpose segmentation
            masks, flows, styles = run_cellpose_segmentation(img_np, dia)

            if masks is not None:
                print(
                    f"Segmentation successful. {len(np.unique(masks)) - 1} masks found. Setting result in model."
                )
                self.application_model.set_segmentation_result(masks)
            else:
                # Optionally, show a more specific error if masks is None after call
                print("Segmentation failed: Cellpose returned no masks.")
                CTkMessagebox(
                    title=constants.MSG_SEGMENTATION_ERROR_TITLE,
                    message="Cellpose segmentation failed to produce masks.",
                    icon="cancel",
                )

        except Exception as e:
            print(f"Exception during segmentation: {str(e)}")
            CTkMessagebox(
                title=constants.MSG_SEGMENTATION_ERROR_TITLE,
                message=str(e),
                icon="cancel",
            )

    def handle_canvas_click(self, event):
        if self.drawing_controller.drawing_mode_active:
            # Delegate to DrawingController's method
            print(
                f"Canvas click: Delegating to DrawingController for draw at ({event.x}, {event.y})."
            )
            return self.drawing_controller.handle_canvas_click_for_draw(event)

        if (
            self.application_model.image_data.mask_array is None
            or self.application_model.image_data.original_image is None
        ):
            print("Canvas click ignored: No mask array or original image.")
            return

        canvas_x, canvas_y = event.x, event.y
        # Get pan/zoom from ApplicationModel
        zoom, pan_x, pan_y = self.application_model.pan_zoom_state.get_params()

        original_x = int((canvas_x - pan_x) / zoom)
        original_y = int((canvas_y - pan_y) / zoom)
        print(
            f"Canvas click at ({canvas_x}, {canvas_y}) translated to original coords ({original_x}, {original_y})."
        )

        # Get image dimensions from ApplicationModel
        orig_img_width = self.application_model.image_data.original_image.width
        orig_img_height = self.application_model.image_data.original_image.height

        if not (0 <= original_x < orig_img_width and 0 <= original_y < orig_img_height):
            print("Canvas click ignored: Click outside original image bounds.")
            return

        # Get mask_array shape from ApplicationModel
        mask_shape = self.application_model.image_data.mask_array.shape

        if not (0 <= original_y < mask_shape[0] and 0 <= original_x < mask_shape[1]):
            print("Canvas click ignored: Click outside mask array bounds.")
            return

        # Access mask_array from ApplicationModel
        cell_id = self.application_model.image_data.mask_array[original_y, original_x]
        print(f"Clicked on cell_id: {cell_id}.")

        if cell_id != 0:
            # Toggle cell inclusion on ApplicationModel
            print(f"Toggling inclusion for cell_id: {cell_id}.")
            self.application_model.toggle_cell_inclusion(cell_id)
        else:
            print("Clicked on background (cell_id 0), no action taken.")

    def handle_mouse_wheel(self, event):
        # Access original_image via ApplicationModel
        if self.application_model.image_data.original_image is None:
            print("Mouse wheel event ignored: No original image.")
            return "break"

        self.is_interacting = True
        if self.interactive_render_timer_id:
            self.after_cancel(self.interactive_render_timer_id)
            self.interactive_render_timer_id = None
            print("Mouse wheel: Canceled pending final render.")

        zoom_factor_multiplier = 0.0
        if event.num == 5 or event.delta < 0:  # Zoom out
            zoom_factor_multiplier = 1 / constants.KEY_ZOOM_STEP
        elif event.num == 4 or event.delta > 0:  # Zoom in
            zoom_factor_multiplier = constants.KEY_ZOOM_STEP

        print(
            f"Mouse wheel event: delta={event.delta}, num={event.num}, zoom_factor_multiplier={zoom_factor_multiplier}"
        )

        if zoom_factor_multiplier == 0.0:
            self.is_interacting = False
            print("Mouse wheel: No zoom change, breaking.")
            return "break"

        # Get current zoom and pan from ApplicationModel
        old_zoom_level, current_pan_x, current_pan_y = (
            self.application_model.pan_zoom_state.get_params()
        )
        potential_new_zoom_level = old_zoom_level * zoom_factor_multiplier

        current_min_zoom = 1.0
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        if (
            self.application_model.image_data.original_image
            and canvas_width > 1
            and canvas_height > 1
        ):
            # Calculate min_zoom_to_fit based on current image and canvas
            current_min_zoom = (
                self.application_model.pan_zoom_state._calculate_min_zoom_to_fit(
                    canvas_width,
                    canvas_height,
                    self.application_model.image_data.original_image.width,
                    self.application_model.image_data.original_image.height,
                )
            )

        # Update min_zoom_to_fit in the model as canvas might have changed
        self.application_model.pan_zoom_state.min_zoom_to_fit = current_min_zoom

        new_zoom_level = max(
            current_min_zoom, min(potential_new_zoom_level, constants.MAX_ZOOM_LEVEL)
        )

        if abs(new_zoom_level - old_zoom_level) < 1e-9:
            self.is_interacting = False
            print("Mouse wheel: Zoom level effectively unchanged.")
            if (
                abs(old_zoom_level - current_min_zoom) < 1e-9
            ):  # At min zoom and tried to zoom out
                print(
                    "Mouse wheel: At min zoom and tried to zoom out, finalizing view reset."
                )
                self._finalize_image_load_view_reset()  # This will reset pan/zoom in model and update display
            else:
                print("Mouse wheel: At max zoom or no change, scheduling final render.")
                self._schedule_final_render()  # e.g. at max zoom
            return "break"

        mouse_x_canvas = event.x
        mouse_y_canvas = event.y
        world_x = (mouse_x_canvas - current_pan_x) / old_zoom_level
        world_y = (mouse_y_canvas - current_pan_y) / old_zoom_level

        new_pan_x, new_pan_y = current_pan_x, current_pan_y
        if abs(new_zoom_level - current_min_zoom) < 1e-9:  # Zoomed to fit state
            # Reset pan and zoom to ensure perfect centering.
            # This will call application_model.reset_pan_zoom_for_image_view and notify
            print("Mouse wheel: Zoomed to fit state, finalizing view reset.")
            self._finalize_image_load_view_reset()
        else:  # Standard zoom, keep point under mouse fixed
            new_pan_x = mouse_x_canvas - (world_x * new_zoom_level)
            new_pan_y = mouse_y_canvas - (world_y * new_zoom_level)
            print(
                f"Mouse wheel: Standard zoom. New zoom: {new_zoom_level}, New pan: ({new_pan_x}, {new_pan_y})"
            )
            # Update model, which will notify renderer
            self.application_model.update_pan_zoom(new_zoom_level, new_pan_x, new_pan_y)
            self.update_display(quality="interactive")  # Local interactive update
            self._schedule_final_render()
        return "break"

    def handle_canvas_resize(self, event):
        new_width = event.width
        new_height = event.height
        print(f"Canvas resize event: New dimensions ({new_width}x{new_height})")

        # Avoid redundant updates if size hasn't actually changed meaningfully
        # or if the event is for initial widget creation before image load.
        if (
            (
                abs(new_width - self._last_canvas_width) < 2
                and abs(new_height - self._last_canvas_height) < 2
            )
            or self.application_model.image_data.original_image is None
        ):  # Check model for image
            print("Canvas resize: Negligible change or no image, skipping update.")
            self._last_canvas_width = new_width
            self._last_canvas_height = new_height
            return

        print("Canvas resize: Significant change, proceeding with update.")
        self._last_canvas_width = new_width
        self._last_canvas_height = new_height

        # Get image dimensions from ApplicationModel
        img_orig_width = self.application_model.image_data.original_image.width
        img_orig_height = self.application_model.image_data.original_image.height

        # Get current pan/zoom from ApplicationModel
        current_zoom, current_pan_x, current_pan_y = (
            self.application_model.pan_zoom_state.get_params()
        )

        # Calculate new min_zoom_to_fit and update it in the model
        new_min_zoom_to_fit = (
            self.application_model.pan_zoom_state._calculate_min_zoom_to_fit(
                new_width, new_height, img_orig_width, img_orig_height
            )
        )
        self.application_model.pan_zoom_state.min_zoom_to_fit = new_min_zoom_to_fit

        # Try to keep current view centered
        # Use previous canvas size for this calculation for stability
        prev_canvas_center_x = (
            self.image_canvas.winfo_width() - (new_width - self._last_canvas_width)
        ) / 2.0  # Approximate previous center X
        prev_canvas_center_y = (
            self.image_canvas.winfo_height() - (new_height - self._last_canvas_height)
        ) / 2.0  # Approximate previous center Y

        world_center_x = (prev_canvas_center_x - current_pan_x) / current_zoom
        world_center_y = (prev_canvas_center_y - current_pan_y) / current_zoom

        final_zoom = current_zoom
        final_pan_x, final_pan_y = current_pan_x, current_pan_y

        if (
            current_zoom <= new_min_zoom_to_fit + 1e-6
        ):  # If was fit or zoomed out, snap to new best fit
            final_zoom = new_min_zoom_to_fit
            zoomed_w = img_orig_width * final_zoom
            zoomed_h = img_orig_height * final_zoom
            final_pan_x = (new_width - zoomed_w) / 2.0
            final_pan_y = (new_height - zoomed_h) / 2.0
        else:  # User was zoomed in, keep zoom level, adjust pan
            final_pan_x = (new_width / 2.0) - (world_center_x * final_zoom)
            final_pan_y = (new_height / 2.0) - (world_center_y * final_zoom)
            final_zoom = max(
                final_zoom, new_min_zoom_to_fit
            )  # Ensure not less than new min_zoom

        # Update ApplicationModel
        self.application_model.update_pan_zoom(final_zoom, final_pan_x, final_pan_y)
        # Model notification will trigger renderer update.
        print(
            f"Canvas resize: Model pan/zoom updated to zoom={final_zoom}, pan=({final_pan_x},{final_pan_y}). Renderer will update."
        )

    def _handle_undo_action(self):
        if self.drawing_controller.drawing_mode_active:
            print("Undo action: Delegating to drawing controller.")
            self.drawing_controller._undo_draw_action()
        else:
            print("Undo action: Calling application model undo.")
            self.application_model.undo()  # Calls model's undo
            # Model notification will trigger display updates and history button updates.

    def _handle_redo_action(self):
        if self.drawing_controller.drawing_mode_active:
            print("Redo action: Delegating to drawing controller.")
            self.drawing_controller._redo_draw_action()
        else:
            print("Redo action: Calling application model redo.")
            self.application_model.redo()  # Calls model's redo
            # Model notification handles updates.

    def _handle_undo_shortcut(self, event=None):
        if self.drawing_controller.drawing_mode_active:
            print("Undo shortcut: Delegating to drawing controller.")
            self.drawing_controller._undo_draw_action()
            return "break"
        else:
            # Check can_undo via model, then call model.undo()
            if self.application_model.can_undo():
                print("Undo shortcut: Calling application model undo.")
                self.application_model.undo()
                # No direct display update, model notification handles it.
            else:
                print("Undo shortcut: Application model cannot undo.")
        return "break"

    def _handle_redo_shortcut(self, event=None):
        if self.drawing_controller.drawing_mode_active:
            print("Redo shortcut: Delegating to drawing controller.")
            self.drawing_controller._redo_draw_action()
            return "break"
        else:
            # Check can_redo via model, then call model.redo()
            if self.application_model.can_redo():
                print("Redo shortcut: Calling application model redo.")
                self.application_model.redo()
                # No direct display update, model notification handles it.
            else:
                print("Redo shortcut: Application model cannot redo.")
        return "break"

    def update_display(self, quality="final"):
        if self.image_view_renderer:  # Ensure renderer is initialized
            print(
                f"update_display called with quality: {quality}. Triggering renderer."
            )
            self.image_view_renderer.render(quality=quality)
        else:
            print("update_display called, but image_view_renderer not initialized.")

    def _handle_zoom_in_key(self, event=None):
        if self.application_model.image_data.original_image is None:
            print("Zoom in key: Ignored, no original image.")
            return "break"
        print("Zoom in key pressed.")

        zoom_factor = constants.KEY_ZOOM_STEP
        old_zoom, current_pan_x, current_pan_y = (
            self.application_model.pan_zoom_state.get_params()
        )

        new_zoom = old_zoom * zoom_factor

        if abs(new_zoom - old_zoom) < 0.0001:
            print("Zoom in key: Zoom level effectively unchanged.")
            return "break"

        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            print("Zoom in key: Canvas not ready.")
            return "break"

        center_x_canvas = canvas_width / 2.0
        center_y_canvas = canvas_height / 2.0
        img_x_at_canvas_center = (center_x_canvas - current_pan_x) / old_zoom
        img_y_at_canvas_center = (center_y_canvas - current_pan_y) / old_zoom

        new_pan_x = center_x_canvas - (img_x_at_canvas_center * new_zoom)
        new_pan_y = center_y_canvas - (img_y_at_canvas_center * new_zoom)

        # Update model
        self.application_model.update_pan_zoom(
            new_zoom, new_pan_x, new_pan_y, max_zoom_override=constants.MAX_ZOOM_LEVEL
        )

        # Check if we hit min_zoom_to_fit exactly (e.g., if image is smaller than canvas and we zoom in to fit)
        if (
            abs(
                self.application_model.pan_zoom_state.zoom_level
                - self.application_model.pan_zoom_state.min_zoom_to_fit
            )
            < 0.001
        ):
            print("Zoom in key: Hit min_zoom_to_fit exactly, recentering.")
            self._finalize_image_load_view_reset()  # Recenter

        self.update_display(quality="interactive")
        self._schedule_final_render()
        print(
            f"Zoom in key: Processed. New zoom: {self.application_model.pan_zoom_state.zoom_level}, Pan: ({self.application_model.pan_zoom_state.pan_x}, {self.application_model.pan_zoom_state.pan_y})"
        )
        return "break"

    def _handle_zoom_out_key(self, event=None):
        if self.application_model.image_data.original_image is None:
            print("Zoom out key: Ignored, no original image.")
            return "break"
        print("Zoom out key pressed.")

        zoom_factor = 1 / constants.KEY_ZOOM_STEP
        old_zoom, current_pan_x, current_pan_y = (
            self.application_model.pan_zoom_state.get_params()
        )
        min_zoom = self.application_model.pan_zoom_state.min_zoom_to_fit

        new_zoom = old_zoom * zoom_factor
        # Min zoom is handled by update_pan_zoom in model using min_zoom_to_fit

        if (
            abs(new_zoom - old_zoom) < 0.0001 and abs(new_zoom - min_zoom) < 0.0001
        ):  # Effectively no change and at min_zoom
            print(
                "Zoom out key: At min zoom and no effective change, ensuring centered."
            )
            self._finalize_image_load_view_reset()  # Ensure centered if already at min zoom
            self.update_display(
                quality="interactive"
            )  # Needed if _finalize did not change anything triggering full render
            self._schedule_final_render()
            return "break"
        elif abs(new_zoom - old_zoom) < 0.0001:
            print("Zoom out key: No effective zoom change.")
            self.update_display(quality="interactive")
            self._schedule_final_render()
            return "break"

        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            print("Zoom out key: Canvas not ready.")
            return "break"

        center_x_canvas = canvas_width / 2.0
        center_y_canvas = canvas_height / 2.0
        img_x_at_canvas_center = (center_x_canvas - current_pan_x) / old_zoom
        img_y_at_canvas_center = (center_y_canvas - current_pan_y) / old_zoom

        new_pan_x = center_x_canvas - (img_x_at_canvas_center * new_zoom)
        new_pan_y = center_y_canvas - (img_y_at_canvas_center * new_zoom)

        # Update model, this will also clamp to min_zoom
        self.application_model.update_pan_zoom(new_zoom, new_pan_x, new_pan_y)

        if abs(self.application_model.pan_zoom_state.zoom_level - min_zoom) < 0.001:
            self._finalize_image_load_view_reset()  # Recenter if we hit min zoom
            print("Zoom out key: Hit min_zoom_to_fit, recentered.")

        self.update_display(quality="interactive")
        self._schedule_final_render()
        print(
            f"Zoom out key: Processed. New zoom: {self.application_model.pan_zoom_state.zoom_level}, Pan: ({self.application_model.pan_zoom_state.pan_x}, {self.application_model.pan_zoom_state.pan_y})"
        )
        return "break"

    def _handle_pan_key(self, dx_factor=0, dy_factor=0):
        if self.application_model.image_data.original_image is None:
            print(
                f"Pan key (dx={dx_factor}, dy={dy_factor}): Ignored, no original image."
            )
            return "break"
        print(f"Pan key pressed: dx_factor={dx_factor}, dy_factor={dy_factor}")

        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        img_orig_width = self.application_model.image_data.original_image.width
        img_orig_height = self.application_model.image_data.original_image.height

        current_zoom, current_pan_x, current_pan_y = (
            self.application_model.pan_zoom_state.get_params()
        )
        zoomed_img_width = img_orig_width * current_zoom
        zoomed_img_height = img_orig_height * current_zoom

        pan_step = constants.PAN_STEP_PIXELS
        potential_pan_x = current_pan_x + (pan_step * dx_factor)
        potential_pan_y = current_pan_y + (pan_step * dy_factor)

        final_pan_x, final_pan_y = current_pan_x, current_pan_y

        if dx_factor != 0:  # Horizontal panning
            if (
                zoomed_img_width > canvas_width
            ):  # Only pan if image is larger than canvas
                min_allowed_pan_x = canvas_width - zoomed_img_width
                max_allowed_pan_x = 0
                final_pan_x = max(
                    min_allowed_pan_x, min(potential_pan_x, max_allowed_pan_x)
                )

        if dy_factor != 0:  # Vertical panning
            if (
                zoomed_img_height > canvas_height
            ):  # Only pan if image is larger than canvas
                min_allowed_pan_y = canvas_height - zoomed_img_height
                max_allowed_pan_y = 0
                final_pan_y = max(
                    min_allowed_pan_y, min(potential_pan_y, max_allowed_pan_y)
                )

        if (
            abs(final_pan_x - current_pan_x) < 1e-6
            and abs(final_pan_y - current_pan_y) < 1e-6
        ):
            # No actual change in pan, no need to update model or render
            print(
                f"Pan key: No change in pan values after clamping. Current: ({current_pan_x},{current_pan_y}), Final: ({final_pan_x},{final_pan_y})"
            )
            return "break"

        self.is_interacting = True
        if self.interactive_render_timer_id:
            self.after_cancel(self.interactive_render_timer_id)
            self.interactive_render_timer_id = None
            print("Pan key: Canceled pending final render.")

        self.application_model.update_pan_zoom(current_zoom, final_pan_x, final_pan_y)
        self.update_display(quality="interactive")
        self._schedule_final_render()
        print(f"Pan key: Processed. New pan: ({final_pan_x}, {final_pan_y})")
        return "break"

    def _handle_pan_left_key(self, event=None):
        return self._handle_pan_key(
            dx_factor=1
        )  # Positive factor for moving image right (view left)

    def _handle_pan_right_key(self, event=None):
        return self._handle_pan_key(
            dx_factor=-1
        )  # Negative factor for moving image left (view right)

    def _handle_pan_up_key(self, event=None):
        return self._handle_pan_key(
            dy_factor=1
        )  # Positive factor for moving image down (view up)

    def _handle_pan_down_key(self, event=None):
        return self._handle_pan_key(
            dy_factor=-1
        )  # Negative factor for moving image up (view down)

    # --- Method to handle global clicks for focus management ---
    def _handle_global_click_for_focus(self, event):
        if not (hasattr(self, "dia_entry") and self.dia_entry):
            # print("Global click: dia_entry does not exist, skipping focus management.") # Can be noisy
            return  # dia_entry doesn't exist, nothing to do.

        focused_widget = self.focus_get()
        if not focused_widget:
            # print("Global click: No widget has focus currently.") # Can be noisy
            return  # No widget has focus currently.

        # Step 1: Determine if dia_entry or its internal text input effectively has focus.
        dia_entry_effectively_has_focus = False
        if focused_widget == self.dia_entry:  # The CTkEntry frame itself
            dia_entry_effectively_has_focus = True
        elif (
            hasattr(self.dia_entry, "_entry")
            and focused_widget == self.dia_entry._entry
        ):  # The internal tk.Entry
            dia_entry_effectively_has_focus = True
        # If CTkEntry has other focusable parts, this check might need expansion,
        # or a generic descendant check for focused_widget within self.dia_entry.
        # For now, assuming the frame or its _entry widget are the key targets.

        if not dia_entry_effectively_has_focus:
            return  # dia_entry is not the one with focus.

        # Step 2: Determine if the click occurred on dia_entry or any of its children.
        clicked_widget = event.widget
        is_click_on_dia_entry_or_child = False
        w = clicked_widget
        while w:
            if w == self.dia_entry:
                is_click_on_dia_entry_or_child = True
                break
            # Optimization: Stop if we ascend beyond the cell_body_frame or toplevel.
            # This means the click was definitely outside dia_entry's hierarchical context *within* our UI part.
            if w == self or (
                hasattr(self, "winfo_toplevel") and w == self.winfo_toplevel()
            ):
                # print(f"Global click: Ascended to {w}, stopping parent search for dia_entry.") # Debug focus
                break
            try:
                if not hasattr(w, "winfo_parent"):
                    break
                parent_path = w.winfo_parent()
                if not parent_path or parent_path == ".":
                    break  # Reached root or no parent string
                w = w.nametowidget(parent_path)  # Ascend to parent
            except (
                Exception
            ):  # Catch errors if widget path is invalid (e.g., widget destroyed)
                break

        if not is_click_on_dia_entry_or_child:
            # dia_entry has focus, and the click was outside of it and its children.
            # Attempt to shift focus to the top-level window, deferred using after_idle.
            if hasattr(self, "winfo_toplevel"):
                self.after_idle(lambda: self.winfo_toplevel().focus_set())
                print(
                    f"Global click: dia_entry had focus ('{focused_widget}'), click on '{event.widget}' was outside. Shifting focus from dia_entry."
                )
            return "break"  # Crucial: stop further event processing for this click.

        return None

    def _finalize_image_load_view_reset(self):
        """
        Called when a new image is loaded or view needs reset.
        Uses ApplicationModel for data and triggers pan/zoom reset in model.
        Updates UI elements like resolution label and segment button state.
        """
        print("Finalizing image load view reset.")
        pil_image = self.application_model.image_data.original_image
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        if pil_image and canvas_width > 1 and canvas_height > 1:
            print(
                f"Resetting pan/zoom for image: {self.application_model.base_filename} on canvas {canvas_width}x{canvas_height}"
            )
            self.application_model.reset_pan_zoom_for_image_view(
                canvas_width, canvas_height
            )
            if self.segment_button:
                self.segment_button.configure(state="normal")
                print("Segment button state set to: normal")
            if self.resolution_label:
                res_text = f"Resolution: {pil_image.width} x {pil_image.height} px"
                self.resolution_label.configure(text=res_text)
                print(f"Resolution label updated: {res_text}")
        else:
            print("Resetting pan/zoom to default (no image or canvas not ready).")
            self.application_model.reset_pan_zoom_for_image_view()  # Resets to default if no image/canvas
            if self.segment_button:
                self.segment_button.configure(state="disabled")
                print("Segment button state set to: disabled (no image/canvas)")
            if self.resolution_label:
                self.resolution_label.configure(text="Resolution: N/A")
                print("Resolution label updated: N/A")

        self.update_history_buttons()  # Ensure history buttons reflect the reset state from model.
        print("History buttons updated after finalizing image load view reset.")

    def _schedule_final_render(self):
        if self.interactive_render_timer_id:
            self.after_cancel(self.interactive_render_timer_id)
            print("Scheduled final render: Canceled existing timer.")
        self.interactive_render_timer_id = self.after(
            self.final_render_delay_ms, lambda: self.update_display(quality="final")
        )
        print(f"Scheduled final render in {self.final_render_delay_ms} ms.")
        self.is_interacting = False

    def _update_stats_label(self):
        """Updates the statistics label based on data from ApplicationModel."""
        if self.stats_label:
            model_state = self.application_model
            total_cells_in_mask = 0
            if (
                model_state.image_data.mask_array is not None
                and model_state.image_data.mask_array.size > 0
            ):
                unique_ids_in_mask = np.unique(model_state.image_data.mask_array)
                total_cells_in_mask = len(unique_ids_in_mask[unique_ids_in_mask != 0])

            # Ensure user_drawn_cell_ids are reconciled with current mask
            actual_user_drawn_ids = model_state.image_data.user_drawn_cell_ids
            if model_state.image_data.mask_array is not None:
                actual_user_drawn_ids = (
                    model_state.image_data.user_drawn_cell_ids.intersection(
                        set(np.unique(model_state.image_data.mask_array))
                    )
                )

            user_drawn_count = len(actual_user_drawn_ids)
            model_found_count = total_cells_in_mask - user_drawn_count
            model_found_count = max(0, model_found_count)
            selected_count = len(model_state.image_data.included_cells)

            stats_text = (
                f"Cell count:\n"
                f"  Model Found: {model_found_count}\n"
                f"  User Drawn: {user_drawn_count}\n"
                f"  Total Unique: {total_cells_in_mask}\n"
                f"  Selected: {selected_count}"
            )
            self.stats_label.configure(text=stats_text)
        else:  # If stats_label is not yet created or None
            default_text = constants.UI_TEXT_STATS_LABEL_DEFAULT
            print(
                "Stats label not available for update, or model data missing for full stats."
            )
            if self.stats_label:  # Double check, though prior if should catch it
                self.stats_label.configure(text=default_text)
                print(f"Stats label set to default: '{default_text}'.")

    def _handle_brightness_change(self, value_str: str):  # Value from slider is string
        try:
            value = float(value_str)
            print(f"Brightness changed (UI): {value}")
            self.application_model.set_brightness(value)
        except ValueError:
            print(f"Invalid brightness value from UI: {value_str}")

    def _handle_contrast_change(self, value_str: str):  # Value from slider is string
        try:
            value = float(value_str)
            print(f"Contrast changed (UI): {value}")
            self.application_model.set_contrast(value)
        except ValueError:
            print(f"Invalid contrast value from UI: {value_str}")

    def _handle_colormap_change(self, choice: str):
        new_colormap = choice if choice != "None" else None
        print(f"Colormap changed (UI): {new_colormap}")
        self.application_model.set_colormap(new_colormap)

    def _reset_display_settings_and_ui(self):
        print("Reset display settings and UI requested.")
        self.application_model.reset_display_adjustments()