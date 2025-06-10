import threading

import customtkinter as ctk
import numpy as np
from CTkMessagebox import CTkMessagebox

from . import constants
from .controllers.display_settings_controller import DisplaySettingsController
from .controllers.drawing_controller import DrawingController
from .controllers.file_io_controller import FileIOController
from .model.application_model import ApplicationModel
from .processing.segmentation_processing import (
    build_segmentation_input_image_from_model,
    run_cellpose_segmentation,
)
from .ui.output_panel import OutputPanel
from .ui.settings_panel import SettingsPanel
from .ui.viewer_panel import ViewerPanel
from .utils.debug_logger import log


class cell_body_frame(ctk.CTkFrame):
    def __init__(self, parent):
        log("cell_body_frame.__init__ execution started.", level="INFO")
        super().__init__(parent)
        self.parent_app = parent

        # --- Initialize ApplicationModel ---
        self.application_model = ApplicationModel()

        # --- Initialize Controllers with ApplicationModel ---
        self.drawing_controller = DrawingController(self, self.application_model)
        self.display_settings_controller = DisplaySettingsController(
            self.application_model
        )
        self.file_io_controller = FileIOController(
            self, self.application_model, self.display_settings_controller
        )

        # --- Other instance variables ---
        self._last_canvas_width = 0
        self._last_canvas_height = 0
        self.is_interacting = False
        self.interactive_render_timer_id = None
        self.final_render_delay_ms = constants.INTERACTIVE_RENDER_DELAY_MS
        self.display_adjustment_render_timer_id = None

        # --- Setup UI layout and widgets ---
        self._setup_ui()

        # --- Subscribe to ApplicationModel updates ---
        self.application_model.subscribe(self.handle_model_update)

        # --- Bind events and update initial button states ---
        self._bind_events()
        self.settings_panel.update_history_buttons()
        log("Initial history button states updated.", level="DEBUG")
        log("cell_body_frame.__init__ execution finished.", level="INFO")

    def _setup_ui(self):
        log("_setup_ui: Started configuring main UI layout.", level="DEBUG")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.settings_panel = SettingsPanel(
            self,
            self.application_model,
            self.file_io_controller,
            self.drawing_controller,
            self.display_settings_controller,
        )
        self.settings_panel.grid(row=0, column=0, sticky="ns", padx=(5, 0), pady=5)

        self.viewer_panel = ViewerPanel(self, self.application_model, self)
        self.viewer_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.image_canvas = self.viewer_panel.get_canvas()  # For event handling
        self.image_view_renderer = self.viewer_panel.get_renderer()

        self.output_panel = OutputPanel(
            self, self.application_model, self.file_io_controller
        )
        self.output_panel.grid(row=0, column=2, sticky="ns", padx=(0, 5), pady=5)
        log("_setup_ui execution finished.", level="DEBUG")

    def handle_model_update(self, change_type: str | None = None):
        log(
            f"handle_model_update execution started. change_type: '{change_type}'",
            level="INFO",
        )

        if change_type == "image_loaded" or change_type == "image_load_failed":
            self._finalize_image_load_view_reset()
            self.settings_panel.update_image_dependent_widgets(
                change_type == "image_loaded"
            )
            self.output_panel.update_pdf_scale_bar_state()

        if change_type in ["pan_zoom_updated", "pan_zoom_reset"]:
            self._schedule_final_render()

        elif change_type in ["display_settings_changed", "display_settings_reset"]:
            self._schedule_display_adjustment_final_render()

        elif change_type == "view_options_changed":
            self._schedule_final_render()

        elif change_type in [
            "segmentation_updated",
            "cell_selection_changed",
            "mask_updated_user_drawn",
            "model_restored_undo",
            "model_restored_redo",
        ]:
            self.update_display(quality="final")

        if change_type in [
            "history_updated",
            "model_restored_undo",
            "model_restored_redo",
            "mask_updated_user_drawn",
            "segmentation_updated",
            "image_loaded",
        ]:
            self.settings_panel.update_history_buttons()

        if change_type in [
            "segmentation_updated",
            "cell_selection_changed",
            "mask_updated_user_drawn",
            "model_restored_undo",
            "model_restored_redo",
            "image_loaded",
        ]:
            self.output_panel.update_stats_label()

        if change_type in ["image_loaded", "pan_zoom_updated", "pan_zoom_reset"]:
            self.settings_panel.update_resolution_label_zoom()

        if change_type in [
            "model_restored_undo",
            "model_restored_redo",
            "segmentation_updated",
            "image_loaded",
            "display_settings_reset",
            "pan_zoom_reset",
            "channel_z_settings_changed",
            "channel_z_settings_reset",
        ]:
            log(f"Synchronizing UI variables with model state due to {change_type}")
            self.settings_panel.sync_ui_variables_with_model()
            self.output_panel.sync_ui_variables_with_model()

        if change_type in [
            "image_loaded",
            "image_load_failed",
            "channel_z_settings_changed",
            "channel_z_settings_reset",
        ]:
            self.settings_panel.update_channel_z_stack_controls()

        if change_type in [
            "image_loaded",
            "image_load_failed",
            "model_reset_for_new_image",
        ]:
            self.settings_panel.update_channel_z_frame_visibility()

        log(
            f"handle_model_update execution finished. change_type: '{change_type}'",
            level="INFO",
        )

    # --- Utility methods for value conversions ---
    def _convert_slider_to_model_value(self, slider_value_float: float) -> float:
        """Converts a slider value (1-100) to a model value (0.1-2.0)."""
        model_value = 0.0
        if slider_value_float <= 50.0:
            model_value = (
                0.1 + (slider_value_float - 1.0) * (0.9 / 49.0)
                if slider_value_float > 1.0
                else 0.1
            )
        else:
            model_value = 1.0 + (slider_value_float - 50.0) * (1.0 / 50.0)
        return max(0.1, min(model_value, 2.0))

    def _convert_model_to_slider_value(self, model_value_float: float) -> int:
        """Converts a model value (0.1-2.0) to a slider value (1-100)."""
        slider_val = 0
        if model_value_float <= 1.0:
            slider_val = (
                1.0 + (model_value_float - 0.1) * (49.0 / 0.9)
                if model_value_float >= 0.1
                else 1.0
            )
        else:
            slider_val = 50.0 + (model_value_float - 1.0) * (50.0 / 1.0)
        return round(max(1, min(slider_val, 100)))

    def _bind_events(self):
        log("_bind_events execution started.", level="DEBUG")

        toplevel = self.winfo_toplevel()

        toplevel.bind("<Command-z>", self._handle_undo_shortcut)
        toplevel.bind("<Control-z>", self._handle_undo_shortcut)
        toplevel.bind("<Command-Shift-Z>", self._handle_redo_shortcut)
        toplevel.bind("<Control-Shift-Z>", self._handle_redo_shortcut)
        toplevel.bind("<Control-y>", self._handle_redo_shortcut)
        toplevel.bind("<Key-plus>", self._handle_zoom_in_key)
        toplevel.bind("<Key-equal>", self._handle_zoom_in_key)
        toplevel.bind("<Key-minus>", self._handle_zoom_out_key)
        toplevel.bind("<Left>", self._handle_pan_left_key)
        toplevel.bind("<Right>", self._handle_pan_right_key)
        toplevel.bind("<Up>", self._handle_pan_up_key)
        toplevel.bind("<Down>", self._handle_pan_down_key)
        toplevel.bind("<Escape>", self.drawing_controller._cancel_drawing_action)
        toplevel.bind("<Return>", self.drawing_controller._handle_enter_key_press)
        toplevel.bind("<KP_Enter>", self.drawing_controller._handle_enter_key_press)

        self.image_canvas.bind(
            "<Button-1>", self._handle_canvas_click_for_entry_commit, add="+"
        )

        toplevel.bind("<Button-1>", self._handle_global_click_for_focus, add="+")

        log("_bind_events execution finished.", level="DEBUG")

    def run_segmentation(self):
        if not self.application_model.image_data.original_image:
            CTkMessagebox(
                title=constants.MSG_SEGMENTATION_ERROR_TITLE,
                message=constants.MSG_NO_IMAGE_FOR_SEGMENTATION,
                icon="warning",
            )
            return

        if self.drawing_controller.drawing_mode_active:
            self.drawing_controller._stop_drawing_mode()

        self.settings_panel.update_segmentation_in_progress(True)

        img_for_segmentation = build_segmentation_input_image_from_model(
            self.application_model,
            apply_image_adjustments=constants.APPLY_IMAGE_ADJUSTMENTS_BEFORE_SEGMENTATION,
        )
        img_np = np.array(img_for_segmentation)

        dia_value = self.application_model.segmentation_diameter
        dia = None
        if dia_value is not None:
            try:
                dia = int(dia_value)
            except (ValueError, TypeError):
                log(
                    f"Invalid diameter value encountered: '{dia_value}'. Cellpose will auto-detect diameter.",
                    level="WARNING",
                )
                dia = None

        thread = threading.Thread(
            target=self._segmentation_worker, args=(img_np, dia), daemon=True
        )
        thread.start()

    def _segmentation_worker(self, img_np: np.ndarray, dia: float | None):
        masks, error = None, None
        try:
            masks, _, _ = run_cellpose_segmentation(img_np, dia)
        except Exception as e:
            error = e
            log(f"Exception during segmentation worker: {str(e)}", level="ERROR")
        self.after(0, lambda: self._handle_segmentation_result(masks, error))

    def _handle_segmentation_result(
        self, masks: np.ndarray | None, error: Exception | None
    ):
        self.settings_panel.update_segmentation_in_progress(False)
        if error:
            CTkMessagebox(
                title=constants.MSG_SEGMENTATION_ERROR_TITLE,
                message=str(error),
                icon="cancel",
            )
            return
        if masks is not None and np.any(masks > 0):
            self.application_model.set_segmentation_result(masks)
        else:
            CTkMessagebox(
                title=constants.MSG_SEGMENTATION_ERROR_TITLE,
                message=constants.MSG_NO_MASKS_CREATED,
                icon="cancel",
            )

    def handle_canvas_click(self, event):
        if self.drawing_controller.drawing_mode_active:
            return self.drawing_controller.handle_canvas_click_for_draw(event)

        if self.application_model.image_data.mask_array is None:
            return

        zoom, pan_x, pan_y = self.application_model.pan_zoom_state.get_params()
        original_x = int((event.x - pan_x) / zoom)
        original_y = int((event.y - pan_y) / zoom)

        if not (
            0 <= original_x < self.application_model.image_data.original_image.width
            and 0
            <= original_y
            < self.application_model.image_data.original_image.height
        ):
            return

        cell_id = self.application_model.image_data.mask_array[original_y, original_x]
        if cell_id != 0:
            self.application_model.toggle_cell_inclusion(cell_id)

    def handle_mouse_wheel(self, event):
        if self.application_model.image_data.original_image is None:
            return "break"
        self.is_interacting = True
        if self.interactive_render_timer_id:
            self.after_cancel(self.interactive_render_timer_id)

        zoom_factor = 1.0
        if event.delta != 0:
            scroll_amount = event.delta * constants.MOUSE_WHEEL_ZOOM_SENSITIVITY
            zoom_factor = 1.0 + scroll_amount
            zoom_factor = max(0.5, min(zoom_factor, 2.0))
        elif event.num == 5:
            zoom_factor = 1 / constants.KEY_ZOOM_STEP  # Zoom out
        elif event.num == 4:
            zoom_factor = constants.KEY_ZOOM_STEP  # Zoom in

        if zoom_factor == 1.0:
            return "break"

        old_zoom, pan_x, pan_y = self.application_model.pan_zoom_state.get_params()
        canvas_w, canvas_h = self.viewer_panel.get_canvas_dimensions()
        min_zoom = self.application_model.pan_zoom_state._calculate_min_zoom_to_fit(
            canvas_w,
            canvas_h,
            self.application_model.image_data.original_image.width,
            self.application_model.image_data.original_image.height,
        )
        self.application_model.pan_zoom_state.min_zoom_to_fit = min_zoom

        new_zoom = max(min_zoom, min(old_zoom * zoom_factor, constants.MAX_ZOOM_LEVEL))

        if abs(new_zoom - old_zoom) < 1e-9:
            self.is_interacting = False
            if abs(old_zoom - min_zoom) < 1e-9:
                self._finalize_image_load_view_reset()
            else:
                self._schedule_final_render()
            return "break"

        world_x = (event.x - pan_x) / old_zoom
        world_y = (event.y - pan_y) / old_zoom

        if abs(new_zoom - min_zoom) < 1e-9:
            self._finalize_image_load_view_reset()
        else:
            new_pan_x = event.x - (world_x * new_zoom)
            new_pan_y = event.y - (world_y * new_zoom)
            self.application_model.update_pan_zoom(new_zoom, new_pan_x, new_pan_y)
        return "break"

    def handle_canvas_resize(self, event):
        new_width, new_height = event.width, event.height
        if (
            abs(new_width - self._last_canvas_width) < 2
            and abs(new_height - self._last_canvas_height) < 2
        ) or not self.application_model.image_data.original_image:
            self._last_canvas_width, self._last_canvas_height = new_width, new_height
            return

        self._last_canvas_width, self._last_canvas_height = new_width, new_height
        img_w, img_h = (
            self.application_model.image_data.original_image.width,
            self.application_model.image_data.original_image.height,
        )
        current_zoom, pan_x, pan_y = self.application_model.pan_zoom_state.get_params()

        new_min_zoom = self.application_model.pan_zoom_state._calculate_min_zoom_to_fit(
            new_width, new_height, img_w, img_h
        )
        self.application_model.pan_zoom_state.min_zoom_to_fit = new_min_zoom

        if current_zoom <= new_min_zoom + 1e-6:
            self._finalize_image_load_view_reset()
        else:
            prev_center_x = (
                self.image_canvas.winfo_width() - (new_width - self._last_canvas_width)
            ) / 2.0
            prev_center_y = (
                self.image_canvas.winfo_height()
                - (new_height - self._last_canvas_height)
            ) / 2.0
            world_center_x = (prev_center_x - pan_x) / current_zoom
            world_center_y = (prev_center_y - pan_y) / current_zoom
            new_pan_x = (new_width / 2.0) - (world_center_x * current_zoom)
            new_pan_y = (new_height / 2.0) - (world_center_y * current_zoom)
            self.application_model.update_pan_zoom(current_zoom, new_pan_x, new_pan_y)

    def _handle_undo_action(self):
        if self.drawing_controller.drawing_mode_active:
            self.drawing_controller._undo_draw_action()
        else:
            self.application_model.undo()

    def _handle_redo_action(self):
        if self.drawing_controller.drawing_mode_active:
            self.drawing_controller._redo_draw_action()
        else:
            self.application_model.redo()

    def _handle_undo_shortcut(self, event=None):
        self._handle_undo_action()
        return "break"

    def _handle_redo_shortcut(self, event=None):
        self._handle_redo_action()
        return "break"

    def update_display(self, quality="final"):
        if self.image_view_renderer:
            self.image_view_renderer.render(quality=quality)

    def _handle_zoom_in_key(self, event=None):
        if not self.application_model.image_data.original_image:
            return "break"
        old_zoom, pan_x, pan_y = self.application_model.pan_zoom_state.get_params()
        new_zoom = old_zoom * constants.KEY_ZOOM_STEP

        canvas_w, canvas_h = self.viewer_panel.get_canvas_dimensions()
        center_x, center_y = canvas_w / 2.0, canvas_h / 2.0
        img_x = (center_x - pan_x) / old_zoom
        img_y = (center_y - pan_y) / old_zoom
        new_pan_x = center_x - (img_x * new_zoom)
        new_pan_y = center_y - (img_y * new_zoom)

        self.application_model.update_pan_zoom(
            new_zoom, new_pan_x, new_pan_y, max_zoom_override=constants.MAX_ZOOM_LEVEL
        )
        if (
            abs(
                self.application_model.pan_zoom_state.zoom_level
                - self.application_model.pan_zoom_state.min_zoom_to_fit
            )
            < 0.001
        ):
            self._finalize_image_load_view_reset()

        self.update_display(quality="interactive")
        self._schedule_final_render()
        return "break"

    def _handle_zoom_out_key(self, event=None):
        if not self.application_model.image_data.original_image:
            return "break"
        old_zoom, pan_x, pan_y = self.application_model.pan_zoom_state.get_params()
        min_zoom = self.application_model.pan_zoom_state.min_zoom_to_fit
        new_zoom = old_zoom / constants.KEY_ZOOM_STEP

        if abs(new_zoom - old_zoom) < 0.0001 and abs(new_zoom - min_zoom) < 0.0001:
            self._finalize_image_load_view_reset()
            return "break"

        canvas_w, canvas_h = self.viewer_panel.get_canvas_dimensions()
        center_x, center_y = canvas_w / 2.0, canvas_h / 2.0
        img_x = (center_x - pan_x) / old_zoom
        img_y = (center_y - pan_y) / old_zoom
        new_pan_x = center_x - (img_x * new_zoom)
        new_pan_y = center_y - (img_y * new_zoom)

        self.application_model.update_pan_zoom(new_zoom, new_pan_x, new_pan_y)
        if abs(self.application_model.pan_zoom_state.zoom_level - min_zoom) < 0.001:
            self._finalize_image_load_view_reset()

        self.update_display(quality="interactive")
        self._schedule_final_render()
        return "break"

    def _handle_pan_key(self, dx_factor=0, dy_factor=0):
        if not self.application_model.image_data.original_image:
            return "break"

        canvas_w, canvas_h = self.viewer_panel.get_canvas_dimensions()
        img_w, img_h = (
            self.application_model.image_data.original_image.width,
            self.application_model.image_data.original_image.height,
        )
        zoom, pan_x, pan_y = self.application_model.pan_zoom_state.get_params()
        zoomed_w, zoomed_h = img_w * zoom, img_h * zoom

        pot_pan_x = pan_x + (constants.PAN_STEP_PIXELS * dx_factor)
        pot_pan_y = pan_y + (constants.PAN_STEP_PIXELS * dy_factor)

        final_pan_x, final_pan_y = pan_x, pan_y
        if dx_factor != 0 and zoomed_w > canvas_w:
            final_pan_x = max(canvas_w - zoomed_w, min(pot_pan_x, 0))
        if dy_factor != 0 and zoomed_h > canvas_h:
            final_pan_y = max(canvas_h - zoomed_h, min(pot_pan_y, 0))

        if abs(final_pan_x - pan_x) < 1e-6 and abs(final_pan_y - pan_y) < 1e-6:
            return "break"

        self.is_interacting = True
        if self.interactive_render_timer_id:
            self.after_cancel(self.interactive_render_timer_id)

        self.application_model.update_pan_zoom(zoom, final_pan_x, final_pan_y)
        self.update_display(quality="interactive")
        self._schedule_final_render()
        return "break"

    def _handle_pan_left_key(self, event=None):
        return self._handle_pan_key(dx_factor=1)

    def _handle_pan_right_key(self, event=None):
        return self._handle_pan_key(dx_factor=-1)

    def _handle_pan_up_key(self, event=None):
        return self._handle_pan_key(dy_factor=1)

    def _handle_pan_down_key(self, event=None):
        return self._handle_pan_key(dy_factor=-1)

    def _handle_canvas_click_for_entry_commit(self, event=None):
        self.settings_panel._commit_segmentation_diameter_from_entry()
        self.settings_panel._commit_scale_bar_microns_from_entry()

    def _handle_global_click_for_focus(self, event):
        # List of entry widgets that need special focus handling.
        # These widgets now live in the settings_panel.
        entry_widget_names = ["diameter_entry_field", "microns_entry_field"]

        for entry_name in entry_widget_names:
            entry_widget = getattr(self.settings_panel, entry_name, None)
            if not entry_widget:
                continue

            # Check if the entry widget (or its internal component) has focus
            focused_widget = self.focus_get()
            if not focused_widget:
                continue

            entry_has_focus = (focused_widget == entry_widget) or (
                hasattr(entry_widget, "_entry")
                and focused_widget == entry_widget._entry
            )

            if not entry_has_focus:
                continue

            # Check if the click was outside the entry widget by traversing up
            # from the clicked widget to see if it's a child of the entry widget.
            clicked_widget = event.widget
            is_click_on_entry = False
            w = clicked_widget
            while w:
                if w == entry_widget:
                    is_click_on_entry = True
                    break
                # Stop ascending if we reach the main frame or toplevel
                try:
                    if w == self or w == self.winfo_toplevel():
                        break
                    w = w.nametowidget(w.winfo_parent())
                except (KeyError, Exception):
                    break

            # If the entry had focus but the click was outside, shift focus away.
            # Using after_idle prevents race conditions and deadlocks.
            if not is_click_on_entry:
                self.after_idle(lambda: self.winfo_toplevel().focus_set())
                return "break"  # Stop further event processing for this click

        return None  # Allow event to propagate if no relevant entry was focused

    def _finalize_image_load_view_reset(self):
        pil_image = self.application_model.image_data.original_image
        canvas_width, canvas_height = self.viewer_panel.get_canvas_dimensions()

        if pil_image and canvas_width > 1 and canvas_height > 1:
            self.application_model.reset_pan_zoom_for_image_view(
                canvas_width, canvas_height
            )
        else:
            self.application_model.reset_pan_zoom_for_image_view(
                canvas_width, canvas_height
            )

        self.settings_panel.update_history_buttons()

    def _schedule_final_render(self):
        if self.interactive_render_timer_id:
            self.after_cancel(self.interactive_render_timer_id)
        self.interactive_render_timer_id = self.after(
            self.final_render_delay_ms, lambda: self.update_display(quality="final")
        )
        self.is_interacting = False

    def _handle_upload_mask_from_file(self):
        try:
            message = self.file_io_controller.upload_mask_from_file()
            if message:
                log(f"Mask upload: {message}", level="INFO")
        except Exception as e:
            log(f"Error in _handle_upload_mask_from_file: {str(e)}", level="ERROR")
            CTkMessagebox(
                title=constants.MSG_LOAD_ERROR_TITLE,
                message=f"Failed to load mask: {str(e)}",
                icon="cancel",
            )

    def set_all_buttons_state(self, enabled: bool, current_export_button=None):
        """Helper to enable/disable all export buttons and other relevant UI elements."""
        state = "normal" if enabled else "disabled"

        # Export buttons in OutputPanel
        if (
            self.output_panel.export_selected_button
            and self.output_panel.export_selected_button != current_export_button
        ):
            self.output_panel.export_selected_button.configure(state=state)
        if (
            self.output_panel.export_tif_button
            and self.output_panel.export_tif_button != current_export_button
        ):
            self.output_panel.export_tif_button.configure(state=state)
        if (
            self.output_panel.export_pdf_button
            and self.output_panel.export_pdf_button != current_export_button
        ):
            self.output_panel.export_pdf_button.configure(state=state)

        # Buttons in SettingsPanel
        if self.settings_panel.select_image_btn:
            self.settings_panel.select_image_btn.configure(state=state)
        if self.settings_panel.segment_button:
            self.settings_panel.segment_button.configure(state=state)
        if self.settings_panel.draw_mask_button:
            self.settings_panel.draw_mask_button.configure(state=state)
        if self.settings_panel.upload_mask_button:
            self.settings_panel.upload_mask_button.configure(state=state)
        if self.settings_panel.undo_button:
            self.settings_panel.undo_button.configure(state=state)
        if self.settings_panel.redo_button:
            self.settings_panel.redo_button.configure(state=state)

        # Entry fields in SettingsPanel
        if (
            hasattr(self.settings_panel, "diameter_entry_field")
            and self.settings_panel.diameter_entry_field
        ):
            self.settings_panel.diameter_entry_field.configure(state=state)
        if (
            hasattr(self.settings_panel, "microns_entry_field")
            and self.settings_panel.microns_entry_field
        ):
            # Also consider the scale lock
            if (
                state == "normal"
                and self.application_model.image_data.scale_conversion
                and self.application_model.image_data.scale_conversion.X
            ):
                self.settings_panel.microns_entry_field.configure(state="normal")
            else:
                self.settings_panel.microns_entry_field.configure(state="disabled")

    def _schedule_display_adjustment_final_render(self):
        if self.display_adjustment_render_timer_id:
            self.after_cancel(self.display_adjustment_render_timer_id)
        self.display_adjustment_render_timer_id = self.after(
            self.final_render_delay_ms, lambda: self.update_display(quality="final")
        )
