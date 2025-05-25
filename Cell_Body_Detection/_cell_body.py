import customtkinter as ctk
import numpy as np
from cellpose import models
from CTkMessagebox import CTkMessagebox

from . import constants
from ._drawing_controller import DrawingController
from ._file_io_controller import FileIOController
from ._view_models_and_renderer import (
    ImageViewModel,
    ImageViewRenderer,
    PanZoomModel,
)


class HistoryController:
    def __init__(self, image_view_model_ref, update_buttons_callback):
        self.image_view_model = image_view_model_ref
        self.undo_stack = []
        self.redo_stack = []
        self.update_buttons_callback = update_buttons_callback
        # Initialize with the current state as the baseline
        if self.image_view_model:
            self.undo_stack.append(self.image_view_model.get_snapshot_data())

    def record_state(self, is_initial_segmentation_state=False):
        snapshot = self.image_view_model.get_snapshot_data()

        if self.undo_stack:  # If undo_stack is not empty (i.e., there's at least the baseline state from __init__ or reset_history)
            self.redo_stack.clear()

        self.undo_stack.append(snapshot)
        if self.update_buttons_callback:
            self.update_buttons_callback()

    def undo(self):
        if not self.can_undo():  # Use the new can_undo logic
            return False

        # Pop the current state and move it to the redo stack
        current_state_snapshot = self.undo_stack.pop()
        self.redo_stack.append(current_state_snapshot)

        # The new top of the undo_stack is the state to restore
        previous_state_snapshot = self.undo_stack[-1]  # Peek
        self.image_view_model.restore_from_snapshot(previous_state_snapshot)

        if self.update_buttons_callback:
            self.update_buttons_callback()
        return True

    def redo(self):
        if not self.can_redo():
            return False

        # Pop the state to be redone
        state_to_restore_snapshot = self.redo_stack.pop()

        # Add this state back to the undo stack as it's now the current state
        self.undo_stack.append(state_to_restore_snapshot)
        self.image_view_model.restore_from_snapshot(state_to_restore_snapshot)

        if self.update_buttons_callback:
            self.update_buttons_callback()
        return True

    def reset_history(self):
        self.undo_stack = []
        self.redo_stack = []
        # After resetting, capture the current state as the new baseline
        if self.image_view_model:  # Ensure image_view_model is available
            self.undo_stack.append(self.image_view_model.get_snapshot_data())
        if self.update_buttons_callback:
            self.update_buttons_callback()

    def can_undo(self):
        # Can only undo if there is a previous state to revert to
        return len(self.undo_stack) > 1

    def can_redo(self):
        return bool(self.redo_stack)


# --- Main Application Frame ---


class cell_body_frame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_app = parent

        self.pan_zoom_model = PanZoomModel()
        self.image_view_model = ImageViewModel()
        self.history_controller = HistoryController(
            self.image_view_model, self.update_history_buttons
        )
        self.drawing_controller = DrawingController(self)
        self.file_io_controller = FileIOController(self)

        self.image_canvas = None
        self.undo_button = None
        self.redo_button = None
        self.filename_label = None
        self.segment_button = None
        self.draw_mask_button = None
        self.dia_entry = None
        self.stats_label = None
        self.resolution_label = None

        self.show_original = ctk.BooleanVar(value=True)
        self.show_mask = ctk.BooleanVar(value=True)
        self.show_boundaries = ctk.BooleanVar(value=False)
        self.boundary_color = ctk.StringVar(
            value=constants.DEFAULT_BOUNDARY_COLOR
        )  # "Green"
        self.show_only_deselected = ctk.BooleanVar(value=False)
        self.show_cell_numbers = ctk.BooleanVar(value=False)

        # PDF Export Options
        self.pdf_opt_masks_only = ctk.BooleanVar(value=False)
        self.pdf_opt_boundaries_only = ctk.BooleanVar(
            value=True
        )  # Default to boundaries only
        self.pdf_opt_numbers_only = ctk.BooleanVar(value=False)
        self.pdf_opt_masks_boundaries = ctk.BooleanVar(value=False)
        self.pdf_opt_masks_numbers = ctk.BooleanVar(value=False)
        self.pdf_opt_boundaries_numbers = ctk.BooleanVar(value=False)
        self.pdf_opt_masks_boundaries_numbers = ctk.BooleanVar(value=False)

        self.data_path = ""
        self.base_filename = ""

        self._last_canvas_width = 0  # For resize debouncing/check
        self._last_canvas_height = 0

        self.is_interacting = False  # Flag for active pan/zoom
        self.interactive_render_timer_id = None
        self.final_render_delay_ms = constants.INTERACTIVE_RENDER_DELAY_MS  # 250

        self._setup_ui()
        self.image_view_renderer = ImageViewRenderer(
            self.image_canvas, self.pan_zoom_model, self.image_view_model, self
        )
        self._bind_events()
        self.update_history_buttons()

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
            command=self._handle_undo_action,  # "Undo"
        )
        self.undo_button.pack(pady=(5, 5), fill="x")
        self.redo_button = ctk.CTkButton(
            history_frame,
            text=constants.UI_TEXT_REDO,
            command=self._handle_redo_action,  # "Redo"
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
            command=self.file_io_controller.load_image,  # MODIFIED
        )
        self.select_image_btn.pack(pady=(5, 5), fill="x")
        self.filename_label = ctk.CTkLabel(
            import_frame,
            text=constants.UI_TEXT_NO_FILE_SELECTED,
            wraplength=constants.UI_FILENAME_LABEL_WRAPLENGTH,  # "No file selected"
        )
        self.filename_label.pack(pady=(0, 5), fill="x")

        self.resolution_label = ctk.CTkLabel(
            import_frame,
            text="Resolution: N/A",  # Initial text
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
        self.dia_entry.insert(0, constants.DEFAULT_MODEL_DIAMETER)  # "100"
        self.dia_entry.pack(pady=(0, 5), fill="x")
        self.segment_button = ctk.CTkButton(
            model_frame,
            text=constants.UI_TEXT_SEGMENT_BUTTON,
            command=self.run_segmentation,
            state="disabled",  # "Segment"
        )
        self.segment_button.pack(pady=(5, 5), fill="x")

        self.draw_mask_button = ctk.CTkButton(
            model_frame,
            text=constants.UI_TEXT_START_DRAWING_BUTTON,
            command=self.drawing_controller._start_drawing_mode,
        )
        self.draw_mask_button.pack(pady=(0, 0), fill="x")

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
            text=constants.UI_TEXT_SHOW_ORIGINAL_IMAGE,  # "Show Original Image"
            variable=self.show_original,
            command=lambda: self.update_display(quality="final"),
        ).pack(anchor="w", pady=2)
        ctk.CTkCheckBox(
            display_frame,
            text=constants.UI_TEXT_SHOW_CELL_MASKS,  # "Show Cell Masks"
            variable=self.show_mask,
            command=lambda: self.update_display(quality="final"),
        ).pack(anchor="w", pady=2)
        ctk.CTkCheckBox(
            display_frame,
            text=constants.UI_TEXT_SHOW_CELL_BOUNDARIES,  # "Show Cell Boundaries"
            variable=self.show_boundaries,
            command=lambda: self.update_display(quality="final"),
        ).pack(anchor="w", pady=2)
        ctk.CTkCheckBox(
            display_frame,
            text=constants.UI_TEXT_SHOW_CELL_NUMBERS,  # "Show Cell Numbers"
            variable=self.show_cell_numbers,
            command=lambda: self.update_display(quality="final"),
        ).pack(anchor="w", pady=2)
        ctk.CTkLabel(display_frame, text=constants.UI_TEXT_DISPLAY_MODE_LABEL).pack(
            anchor="w", pady=(5, 0)
        )  # "Display Mode:"
        deselected_switch = ctk.CTkSwitch(
            display_frame,
            text=constants.UI_TEXT_SHOW_DESELECTED_ONLY,  # "Show Deselected Only"
            variable=self.show_only_deselected,
            command=lambda: self.update_display(quality="final"),
        )
        deselected_switch.pack(anchor="w", pady=(2, 5), padx=10)
        ctk.CTkLabel(
            display_frame, text=constants.UI_TEXT_BOUNDARY_COLOR_LABEL
        ).pack(  # "Boundary Color:"
            anchor="w", pady=(5, 0)
        )
        ctk.CTkOptionMenu(
            display_frame,
            variable=self.boundary_color,
            values=constants.AVAILABLE_BOUNDARY_COLORS,  # ["Green", "Red", "Blue", "Yellow"]
            command=lambda _: self.update_display(quality="final"),
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
        self.stats_label.pack(
            padx=10, pady=(5, 10), fill="x", anchor="n"
        )  # Adjusted pady

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
            variable=self.pdf_opt_masks_only,  # "Masks Only"
        )
        self.pdf_cb_masks_only.pack(anchor="w", pady=1)

        self.pdf_cb_boundaries_only = ctk.CTkCheckBox(
            export_frame,
            text=constants.UI_TEXT_PDF_BOUNDARIES_ONLY,
            variable=self.pdf_opt_boundaries_only,  # "Boundaries Only"
        )
        self.pdf_cb_boundaries_only.pack(anchor="w", pady=1)

        self.pdf_cb_numbers_only = ctk.CTkCheckBox(
            export_frame,
            text=constants.UI_TEXT_PDF_NUMBERS_ONLY,
            variable=self.pdf_opt_numbers_only,  # "Numbers Only"
        )
        self.pdf_cb_numbers_only.pack(anchor="w", pady=1)

        self.pdf_cb_masks_boundaries = ctk.CTkCheckBox(
            export_frame,
            text=constants.UI_TEXT_PDF_MASKS_BOUNDARIES,  # "Masks & Boundaries"
            variable=self.pdf_opt_masks_boundaries,
        )
        self.pdf_cb_masks_boundaries.pack(anchor="w", pady=1)

        self.pdf_cb_masks_numbers = ctk.CTkCheckBox(
            export_frame,
            text=constants.UI_TEXT_PDF_MASKS_NUMBERS,
            variable=self.pdf_opt_masks_numbers,  # "Masks & Numbers"
        )
        self.pdf_cb_masks_numbers.pack(anchor="w", pady=1)

        self.pdf_cb_boundaries_numbers = ctk.CTkCheckBox(
            export_frame,
            text=constants.UI_TEXT_PDF_BOUNDARIES_NUMBERS,  # "Boundaries & Numbers"
            variable=self.pdf_opt_boundaries_numbers,
        )
        self.pdf_cb_boundaries_numbers.pack(anchor="w", pady=1)

        self.pdf_cb_masks_boundaries_numbers = ctk.CTkCheckBox(
            export_frame,
            text=constants.UI_TEXT_PDF_MASKS_BOUNDARIES_NUMBERS,  # "Masks, Boundaries & Numbers"
            variable=self.pdf_opt_masks_boundaries_numbers,
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
            binder(
                "<Escape>", self.drawing_controller._cancel_drawing_action
            )  # MODIFIED
            # Shortcut for finalizing drawing
            binder(
                "<Return>", self.drawing_controller._handle_enter_key_press
            )  # MODIFIED
            binder(
                "<KP_Enter>", self.drawing_controller._handle_enter_key_press
            )  # MODIFIED

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
            if self.history_controller:  # Ensure history_controller is initialized
                can_undo_flag = self.history_controller.can_undo()
                can_redo_flag = self.history_controller.can_redo()

        if self.undo_button:
            self.undo_button.configure(state="normal" if can_undo_flag else "disabled")
        if self.redo_button:
            self.redo_button.configure(state="normal" if can_redo_flag else "disabled")

    def run_segmentation(self):
        if not self.image_view_model.original_image:
            CTkMessagebox(
                title=constants.MSG_SEGMENTATION_ERROR_TITLE,  # "Error"
                message=constants.MSG_NO_IMAGE_FOR_SEGMENTATION,  # "No image loaded for segmentation."
                icon="warning",
            )
            return
        try:
            # Ensure drawing mode is off before segmentation
            if self.drawing_controller.drawing_mode_active:
                self.drawing_controller._stop_drawing_mode()  # Stop drawing if active

            model = models.CellposeModel(gpu=True)
            img_for_segmentation = self.image_view_model.original_image.convert("L")
            img_np = np.array(img_for_segmentation)

            dia_text = (
                self.dia_entry.get()
                if self.dia_entry
                else constants.DEFAULT_MODEL_DIAMETER
            )  # "100"
            dia = float(dia_text) if dia_text else None

            masks, flows, styles = model.eval(img_np, diameter=dia)

            self.image_view_model.set_segmentation_result(masks)
            if self.image_view_renderer:
                self.image_view_renderer.invalidate_caches()  # Invalidate on new seg
            self.history_controller.record_state(is_initial_segmentation_state=True)

            self.update_display(quality="final")

        except Exception as e:
            CTkMessagebox(
                title=constants.MSG_SEGMENTATION_ERROR_TITLE,
                message=str(e),
                icon="cancel",
            )  # "Segmentation Error"

    def handle_canvas_click(self, event):
        if self.drawing_controller.drawing_mode_active:
            # Delegate to DrawingController's method
            return self.drawing_controller.handle_canvas_click_for_draw(event)

        if (
            self.image_view_model.mask_array is None
            or self.image_view_model.original_image is None
        ):
            return

        canvas_x, canvas_y = event.x, event.y
        zoom, pan_x, pan_y = self.pan_zoom_model.get_params()

        # Calculate coordinates on the original image plane
        original_x = int((canvas_x - pan_x) / zoom)
        original_y = int((canvas_y - pan_y) / zoom)

        orig_img_width = self.image_view_model.original_image.width
        orig_img_height = self.image_view_model.original_image.height

        # Check if the click (mapped to original image coords) is within the original image bounds
        if not (0 <= original_x < orig_img_width and 0 <= original_y < orig_img_height):
            return

        mask_shape = (
            self.image_view_model.mask_array.shape
        )  # (rows, cols) which is (height, width)

        # Assuming original_image (that was fed to segmentation) and mask_array have a 1:1 pixel correspondence.
        # Therefore, original_x maps to column index and original_y maps to row index in mask_array.
        mask_col_idx = original_x
        mask_row_idx = original_y

        # Ensure these indices are within the bounds of the mask_array
        if not (
            0 <= mask_row_idx < mask_shape[0] and 0 <= mask_col_idx < mask_shape[1]
        ):
            return

        cell_id = self.image_view_model.mask_array[mask_row_idx, mask_col_idx]

        if cell_id != 0:
            if self.image_view_renderer:
                self.image_view_renderer.invalidate_caches()
            self.image_view_model.toggle_cell_inclusion(cell_id)
            self.history_controller.record_state()
            self.update_display(quality="final")

    def handle_mouse_wheel(self, event):
        if self.image_view_model.original_image is None:
            return "break"

        self.is_interacting = True
        if self.interactive_render_timer_id:  # Cancel pending final render
            self.after_cancel(self.interactive_render_timer_id)
            self.interactive_render_timer_id = None

        zoom_factor_multiplier = 0.0
        if event.num == 5 or event.delta < 0:
            zoom_factor_multiplier = 1 / constants.KEY_ZOOM_STEP  # 0.9
        elif event.num == 4 or event.delta > 0:
            zoom_factor_multiplier = constants.KEY_ZOOM_STEP  # 1.1
        if zoom_factor_multiplier == 0.0:
            self.is_interacting = False  # No actual zoom occurred
            return "break"

        old_zoom_level = self.pan_zoom_model.zoom_level
        potential_new_zoom_level = old_zoom_level * zoom_factor_multiplier

        current_min_zoom = 1.0  # Default
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        if (
            self.image_view_model.original_image
            and canvas_width > 1
            and canvas_height > 1
        ):
            current_min_zoom = self.pan_zoom_model._calculate_min_zoom_to_fit(
                canvas_width,
                canvas_height,
                self.image_view_model.original_image.width,
                self.image_view_model.original_image.height,
            )
        # Update the stored min_zoom_to_fit in the model as canvas might have changed
        self.pan_zoom_model.min_zoom_to_fit = current_min_zoom

        new_zoom_level = max(
            current_min_zoom, min(potential_new_zoom_level, constants.MAX_ZOOM_LEVEL)
        )  # 5.0

        if abs(new_zoom_level - old_zoom_level) < 1e-9:
            self.is_interacting = False  # No actual zoom occurred
            # If we are at min_zoom and tried to zoom out further, ensure it's centered.
            if abs(old_zoom_level - current_min_zoom) < 1e-9:
                self._finalize_image_load_and_pan_reset()  # This will also update_display internally.
            else:  # Otherwise (e.g. at max zoom), just ensure final quality render if interaction was happening.
                self._schedule_final_render()
            return "break"

        # Zoom level DID change.
        # Calculate world coordinates of the mouse point on the original image
        # using OLD zoom and current pan (pan state before this zoom event's changes)
        mouse_x_canvas = event.x
        mouse_y_canvas = event.y
        world_x = (mouse_x_canvas - self.pan_zoom_model.pan_x) / old_zoom_level
        world_y = (mouse_y_canvas - self.pan_zoom_model.pan_y) / old_zoom_level

        # Apply the new zoom level to the model
        self.pan_zoom_model.zoom_level = new_zoom_level

        # Now, determine panning strategy.
        if abs(self.pan_zoom_model.zoom_level - current_min_zoom) < 1e-9:
            # We've zoomed to the "fit" state (or attempted to zoom past it).
            # Reset pan and zoom to ensure perfect centering.
            # _finalize_image_load_and_pan_reset will handle setting the zoom to min_zoom_to_fit
            # and centering the pan. It also calls update_display(quality="final").
            self._finalize_image_load_and_pan_reset()
        else:
            # Standard zoom: keep the point under the mouse fixed.
            self.pan_zoom_model.pan_x = mouse_x_canvas - (
                world_x * self.pan_zoom_model.zoom_level
            )
            self.pan_zoom_model.pan_y = mouse_y_canvas - (
                world_y * self.pan_zoom_model.zoom_level
            )

            self.update_display(quality="interactive")
            self._schedule_final_render()
        return "break"

    def handle_canvas_resize(self, event):
        new_width = event.width
        new_height = event.height

        # Avoid redundant updates if size hasn't actually changed meaningfully
        # or if the event is for initial widget creation before image load.
        if (
            abs(new_width - self._last_canvas_width) < 2
            and abs(new_height - self._last_canvas_height) < 2
        ) or self.image_view_model.original_image is None:
            # Store them anyway for the first <Configure> event after image load
            self._last_canvas_width = new_width
            self._last_canvas_height = new_height
            return

        self._last_canvas_width = new_width
        self._last_canvas_height = new_height

        img_orig_width = self.image_view_model.original_image.width
        img_orig_height = self.image_view_model.original_image.height

        # Calculate the new minimum zoom to fit the image in the new canvas size
        new_min_zoom_to_fit = self.pan_zoom_model._calculate_min_zoom_to_fit(
            new_width, new_height, img_orig_width, img_orig_height
        )
        self.pan_zoom_model.min_zoom_to_fit = (
            new_min_zoom_to_fit  # Update model's stored min_zoom
        )

        # Get current view center in world coordinates (original image pixels)
        # This tries to keep the current view centered after resize
        current_canvas_center_x = (
            self._last_canvas_width / 2.0
        )  # Use previous canvas size for this
        current_canvas_center_y = self._last_canvas_height / 2.0

        world_center_x = (
            current_canvas_center_x - self.pan_zoom_model.pan_x
        ) / self.pan_zoom_model.zoom_level
        world_center_y = (
            current_canvas_center_y - self.pan_zoom_model.pan_y
        ) / self.pan_zoom_model.zoom_level

        # Option: If current zoom is already "fit" or "zoomed out", snap to new best fit.
        if (
            self.pan_zoom_model.zoom_level <= 1.0 + 1e-6
        ):  # Add epsilon for float comparison
            self.pan_zoom_model.zoom_level = new_min_zoom_to_fit
            # Recenter pan for this new fit zoom
            zoomed_w = img_orig_width * self.pan_zoom_model.zoom_level
            zoomed_h = img_orig_height * self.pan_zoom_model.zoom_level
            self.pan_zoom_model.pan_x = (new_width - zoomed_w) / 2.0
            self.pan_zoom_model.pan_y = (new_height - zoomed_h) / 2.0
        else:
            # User was zoomed in. Keep current zoom_level, but adjust pan to keep
            # world_center_x, world_center_y at the new canvas center.
            self.pan_zoom_model.pan_x = (new_width / 2.0) - (
                world_center_x * self.pan_zoom_model.zoom_level
            )
            self.pan_zoom_model.pan_y = (new_height / 2.0) - (
                world_center_y * self.pan_zoom_model.zoom_level
            )
            # Ensure zoom is not less than the new min_zoom_to_fit even if user was zoomed in.
            # This can happen if they were zoomed in, but the window shrank drastically.
            self.pan_zoom_model.zoom_level = max(
                self.pan_zoom_model.zoom_level, new_min_zoom_to_fit
            )

        self.update_display(quality="final")

    def _handle_undo_action(self):
        if self.drawing_controller.drawing_mode_active:
            self.drawing_controller._undo_draw_action()
        else:
            if self.history_controller.undo():
                if self.image_view_renderer:
                    self.image_view_renderer.invalidate_caches()
                self.update_display(quality="final")

    def _handle_redo_action(self):
        if self.drawing_controller.drawing_mode_active:
            self.drawing_controller._redo_draw_action()
        else:
            if self.history_controller.redo():
                if self.image_view_renderer:
                    self.image_view_renderer.invalidate_caches()
                self.update_display(quality="final")

    def _handle_undo_shortcut(self, event=None):
        if self.drawing_controller.drawing_mode_active:
            self.drawing_controller._undo_draw_action()
            return "break"
        else:
            if self.history_controller.can_undo():
                if self.image_view_renderer:
                    self.image_view_renderer.invalidate_caches()
                self.history_controller.undo()
                self.update_display(quality="final")
        return "break"

    def _handle_redo_shortcut(self, event=None):
        if self.drawing_controller.drawing_mode_active:
            self.drawing_controller._redo_draw_action()
            return "break"
        else:
            if self.history_controller.can_redo():
                if self.image_view_renderer:
                    self.image_view_renderer.invalidate_caches()
                self.history_controller.redo()
                self.update_display(quality="final")
        return "break"

    def update_display(self, quality="final"):
        if self.image_view_renderer:  # Ensure renderer is initialized
            self.image_view_renderer.render(quality=quality)

    def _get_exact_boundaries(self, label_mask):
        if label_mask is None or label_mask.size == 0:
            return np.array([], dtype=bool)
        padded = np.pad(label_mask, pad_width=1, mode="constant", constant_values=0)
        boundaries = np.zeros_like(label_mask, dtype=bool)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = padded[
                1 + dy : 1 + dy + label_mask.shape[0],
                1 + dx : 1 + dx + label_mask.shape[1],
            ]
            is_diff = label_mask != shifted
            is_cell_boundary_with_bg = ((label_mask != 0) & (shifted == 0)) | (
                (label_mask == 0) & (shifted != 0)
            )
            is_cell_boundary_with_cell = (
                (label_mask != 0) & (shifted != 0) & (label_mask != shifted)
            )
            boundaries |= is_diff & (
                is_cell_boundary_with_bg | is_cell_boundary_with_cell
            )
        return boundaries

    def _handle_zoom_in_key(self, event=None):
        if self.image_view_model.original_image is None:
            return "break"

        zoom_factor = constants.KEY_ZOOM_STEP  # 1.1
        old_zoom = self.pan_zoom_model.zoom_level

        # Calculate potential new zoom and clamp it immediately
        new_zoom = old_zoom * zoom_factor
        new_zoom = min(new_zoom, constants.MAX_ZOOM_LEVEL)  # Max 5x zoom

        if abs(new_zoom - old_zoom) < 0.0001:  # If effectively no change
            return "break"

        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return "break"  # Avoid division by zero or nonsensical ops

        # Zoom towards the center of the canvas view
        center_x_canvas = canvas_width / 2.0
        center_y_canvas = canvas_height / 2.0

        # Image coordinates currently at the canvas center
        img_x_at_canvas_center = (
            center_x_canvas - self.pan_zoom_model.pan_x
        ) / old_zoom
        img_y_at_canvas_center = (
            center_y_canvas - self.pan_zoom_model.pan_y
        ) / old_zoom

        self.pan_zoom_model.zoom_level = new_zoom

        # Calculate new pan to keep that image point at the canvas center
        self.pan_zoom_model.pan_x = center_x_canvas - (
            img_x_at_canvas_center * new_zoom
        )
        self.pan_zoom_model.pan_y = center_y_canvas - (
            img_y_at_canvas_center * new_zoom
        )

        # If zoom hits exactly 1.0 (and min_zoom_to_fit is 1.0), it implies a desire to be perfectly centered.
        # _finalize_image_load_and_pan_reset also handles the min_zoom_to_fit < 1.0 cases correctly for centering.
        if (
            abs(self.pan_zoom_model.zoom_level - self.pan_zoom_model.min_zoom_to_fit)
            < 0.001
        ):
            # This covers both when min_zoom_to_fit is 1.0 and when it's < 1.0.
            # It will center the image appropriately for that zoom level.
            self._finalize_image_load_and_pan_reset()

        self.update_display(quality="interactive")
        self._schedule_final_render()
        return "break"

    def _handle_zoom_out_key(self, event=None):
        if self.image_view_model.original_image is None:
            return "break"

        zoom_factor = 1 / constants.KEY_ZOOM_STEP  # 1.1
        old_zoom = self.pan_zoom_model.zoom_level

        # Calculate potential new zoom and clamp it immediately
        new_zoom = old_zoom * zoom_factor
        new_zoom = max(self.pan_zoom_model.min_zoom_to_fit, new_zoom)

        if (
            abs(new_zoom - old_zoom) < 0.0001
        ):  # If effectively no change (e.g., already at min_zoom_to_fit)
            # If already at min_zoom_to_fit, ensure it's perfectly centered as per that fit.
            if (
                abs(
                    self.pan_zoom_model.zoom_level - self.pan_zoom_model.min_zoom_to_fit
                )
                < 0.001
            ):
                self._finalize_image_load_and_pan_reset()
            self.update_display(quality="interactive")
            self._schedule_final_render()
            return "break"

        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return "break"

        center_x_canvas = canvas_width / 2.0
        center_y_canvas = canvas_height / 2.0

        img_x_at_canvas_center = (
            center_x_canvas - self.pan_zoom_model.pan_x
        ) / old_zoom
        img_y_at_canvas_center = (
            center_y_canvas - self.pan_zoom_model.pan_y
        ) / old_zoom

        self.pan_zoom_model.zoom_level = new_zoom

        self.pan_zoom_model.pan_x = center_x_canvas - (
            img_x_at_canvas_center * new_zoom
        )
        self.pan_zoom_model.pan_y = center_y_canvas - (
            img_y_at_canvas_center * new_zoom
        )

        # If zoom reaches min_zoom_to_fit (either 1.0 or less), call _finalize_image_load_and_pan_reset.
        # This function is designed to set the zoom to min_zoom_to_fit and center the image correctly.
        if (
            abs(self.pan_zoom_model.zoom_level - self.pan_zoom_model.min_zoom_to_fit)
            < 0.001
        ):
            self._finalize_image_load_and_pan_reset()

        self.update_display(quality="interactive")
        self._schedule_final_render()
        return "break"

    # --- Add Arrow Key Panning Handlers ---
    def _handle_pan_left_key(self, event=None):
        if self.image_view_model.original_image is None:
            return "break"

        canvas_width = self.image_canvas.winfo_width()
        img_orig_width = self.image_view_model.original_image.width
        zoom = self.pan_zoom_model.zoom_level
        zoomed_img_width = img_orig_width * zoom

        if (
            zoomed_img_width <= canvas_width
        ):  # Image is fully visible or fits horizontally
            return "break"  # No horizontal panning needed

        self.is_interacting = True
        if self.interactive_render_timer_id:
            self.after_cancel(self.interactive_render_timer_id)
            self.interactive_render_timer_id = None

        pan_step = constants.PAN_STEP_PIXELS  # 50
        current_pan_x = self.pan_zoom_model.pan_x
        potential_pan_x = current_pan_x + pan_step

        # At this point, zoomed_img_width > canvas_width
        min_allowed_pan_x = canvas_width - zoomed_img_width
        max_allowed_pan_x = 0

        clamped_pan_x = max(min_allowed_pan_x, min(potential_pan_x, max_allowed_pan_x))

        self.pan_zoom_model.pan_x = clamped_pan_x
        self.update_display(quality="interactive")
        self._schedule_final_render()
        return "break"

    def _handle_pan_right_key(self, event=None):
        if self.image_view_model.original_image is None:
            return "break"

        canvas_width = self.image_canvas.winfo_width()
        img_orig_width = self.image_view_model.original_image.width
        zoom = self.pan_zoom_model.zoom_level
        zoomed_img_width = img_orig_width * zoom

        if (
            zoomed_img_width <= canvas_width
        ):  # Image is fully visible or fits horizontally
            return "break"  # No horizontal panning needed

        self.is_interacting = True
        if self.interactive_render_timer_id:
            self.after_cancel(self.interactive_render_timer_id)
            self.interactive_render_timer_id = None

        pan_step = constants.PAN_STEP_PIXELS  # 50
        current_pan_x = self.pan_zoom_model.pan_x
        potential_pan_x = current_pan_x - pan_step

        # At this point, zoomed_img_width > canvas_width
        min_allowed_pan_x = canvas_width - zoomed_img_width
        max_allowed_pan_x = 0

        clamped_pan_x = max(min_allowed_pan_x, min(potential_pan_x, max_allowed_pan_x))

        self.pan_zoom_model.pan_x = clamped_pan_x
        self.update_display(quality="interactive")
        self._schedule_final_render()
        return "break"

    def _handle_pan_up_key(self, event=None):
        if self.image_view_model.original_image is None:
            return "break"

        canvas_height = self.image_canvas.winfo_height()
        img_orig_height = self.image_view_model.original_image.height
        zoom = self.pan_zoom_model.zoom_level
        zoomed_img_height = img_orig_height * zoom

        if (
            zoomed_img_height <= canvas_height
        ):  # Image is fully visible or fits vertically
            return "break"  # No vertical panning needed

        self.is_interacting = True
        if self.interactive_render_timer_id:
            self.after_cancel(self.interactive_render_timer_id)
            self.interactive_render_timer_id = None

        pan_step = constants.PAN_STEP_PIXELS  # 50
        current_pan_y = self.pan_zoom_model.pan_y
        potential_pan_y = current_pan_y + pan_step

        # At this point, zoomed_img_height > canvas_height
        min_allowed_pan_y = canvas_height - zoomed_img_height
        max_allowed_pan_y = 0

        clamped_pan_y = max(min_allowed_pan_y, min(potential_pan_y, max_allowed_pan_y))

        self.pan_zoom_model.pan_y = clamped_pan_y
        self.update_display(quality="interactive")
        self._schedule_final_render()
        return "break"

    def _handle_pan_down_key(self, event=None):
        if self.image_view_model.original_image is None:
            return "break"

        canvas_height = self.image_canvas.winfo_height()
        img_orig_height = self.image_view_model.original_image.height
        zoom = self.pan_zoom_model.zoom_level
        zoomed_img_height = img_orig_height * zoom

        if (
            zoomed_img_height <= canvas_height
        ):  # Image is fully visible or fits vertically
            return "break"  # No vertical panning needed

        self.is_interacting = True
        if self.interactive_render_timer_id:
            self.after_cancel(self.interactive_render_timer_id)
            self.interactive_render_timer_id = None

        pan_step = constants.PAN_STEP_PIXELS  # 50
        current_pan_y = self.pan_zoom_model.pan_y
        potential_pan_y = current_pan_y - pan_step

        # At this point, zoomed_img_height > canvas_height
        min_allowed_pan_y = canvas_height - zoomed_img_height
        max_allowed_pan_y = 0

        clamped_pan_y = max(min_allowed_pan_y, min(potential_pan_y, max_allowed_pan_y))

        self.pan_zoom_model.pan_y = clamped_pan_y
        self.update_display(quality="interactive")
        self._schedule_final_render()
        return "break"

    # --- Method to handle global clicks for focus management ---
    def _handle_global_click_for_focus(self, event):
        if not (hasattr(self, "dia_entry") and self.dia_entry):
            return  # dia_entry doesn't exist, nothing to do.

        focused_widget = self.focus_get()
        if not focused_widget:
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
            return "break"  # Crucial: stop further event processing for this click.

        # If dia_entry didn't have focus, or if the click was on dia_entry/its child,
        # allow the event to be processed normally by other handlers.
        return None

    def _finalize_image_load_and_pan_reset(self):
        pil_image = self.image_view_model.original_image
        if pil_image:
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            img_width = pil_image.width
            img_height = pil_image.height
            # reset_for_new_image now calculates and sets the initial zoom to fit
            self.pan_zoom_model.reset_for_new_image(
                canvas_width, canvas_height, img_width, img_height
            )
            if self.segment_button:
                self.segment_button.configure(state="normal")
            if self.image_view_renderer:
                self.image_view_renderer.invalidate_caches()  # Invalidate on new image
            if self.resolution_label:
                self.resolution_label.configure(
                    text=f"Resolution: {pil_image.width} x {pil_image.height} px"
                )
        else:
            if self.segment_button:
                self.segment_button.configure(state="disabled")
            self.pan_zoom_model.reset_for_new_image()
            if self.image_view_renderer:
                self.image_view_renderer.invalidate_caches()
            if self.resolution_label:
                self.resolution_label.configure(text="Resolution: N/A")

        self.update_display(quality="final")  # First render is final quality
        self.update_history_buttons()

    def _schedule_final_render(self):
        if self.interactive_render_timer_id:
            self.after_cancel(self.interactive_render_timer_id)
        self.interactive_render_timer_id = self.after(
            self.final_render_delay_ms, lambda: self.update_display(quality="final")
        )
        self.is_interacting = False  # Reset interaction flag here

    def _update_stats_label(self, stats_text):
        if self.stats_label:
            self.stats_label.configure(text=stats_text)
