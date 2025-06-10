import numpy as np
from CTkMessagebox import CTkMessagebox
from skimage.draw import polygon

from .. import constants
from ..model.application_model import ApplicationModel
from ..utils.debug_logger import log


# --- Drawing Controller Class ---
class DrawingController:
    def __init__(self, parent_frame, application_model_ref: ApplicationModel):
        self.parent_frame = parent_frame  # Reference to cell_body_frame
        self.application_model = application_model_ref
        self.drawing_mode_active = False
        self.current_draw_points = []  # Stores original image coordinates for the polygon being drawn
        self.drawing_history_stack = []  # For undo/redo of points *during* an active drawing session
        self.drawing_history_pointer = -1  # Pointer for drawing_history_stack

    def _start_drawing_mode(self):
        if self.application_model.image_data.original_image is None:
            CTkMessagebox(
                title=constants.MSG_DRAWING_ERROR_TITLE,
                message=constants.MSG_DRAWING_LOAD_IMAGE_FIRST,
                icon="warning",
            )
            log(
                "DrawingController: Start drawing mode failed - no image loaded.",
                level="WARNING",
            )
            return

        self.drawing_mode_active = True
        self.current_draw_points = []
        self.drawing_history_stack = [
            []
        ]  # Start with an empty set of points for current drawing session
        self.drawing_history_pointer = 0
        log(
            "DrawingController: Drawing mode started. Polygon points and history reset."
        )

        if self.parent_frame.settings_panel.draw_mask_button:
            self.parent_frame.settings_panel.draw_mask_button.configure(
                text=constants.UI_TEXT_FINALIZE_POLYGON_BUTTON,
                command=self._try_finalize_drawing,
            )
            log("DrawingController: Draw mask button configured to 'Finalize Polygon'.")
        if self.parent_frame.settings_panel.segment_button:
            self.parent_frame.settings_panel.segment_button.configure(state="disabled")
            log("DrawingController: Segment button disabled.")
        if self.parent_frame.settings_panel.upload_mask_button:
            self.parent_frame.settings_panel.upload_mask_button.configure(
                state="disabled"
            )
            log("DrawingController: Upload mask button disabled.")
        if self.parent_frame.viewer_panel.image_canvas:
            self.parent_frame.viewer_panel.image_canvas.config(cursor="crosshair")
            log("DrawingController: Canvas cursor set to crosshair.")

        self.parent_frame.settings_panel.update_history_buttons()
        self.parent_frame.update_display(quality="interactive")

    def _stop_drawing_mode(self):
        self.drawing_mode_active = False
        self.drawing_history_stack = []
        self.drawing_history_pointer = -1
        log(
            "DrawingController: Drawing mode stopped. Polygon points and history cleared."
        )

        if self.parent_frame.settings_panel.draw_mask_button:
            self.parent_frame.settings_panel.draw_mask_button.configure(
                text=constants.UI_TEXT_START_DRAWING_BUTTON,
                command=self._start_drawing_mode,
            )
            log(
                "DrawingController: Draw mask button configured back to 'Start Drawing'."
            )
        if (
            self.parent_frame.settings_panel.segment_button
            and self.application_model.image_data.original_image is not None
        ):
            self.parent_frame.settings_panel.segment_button.configure(state="normal")
            log("DrawingController: Segment button re-enabled.")
        if (
            self.parent_frame.settings_panel.upload_mask_button
            and self.application_model.image_data.original_image is not None
        ):
            self.parent_frame.settings_panel.upload_mask_button.configure(
                state="normal"
            )
            log("DrawingController: Upload mask button re-enabled.")
        if self.parent_frame.viewer_panel.image_canvas:
            self.parent_frame.viewer_panel.image_canvas.config(cursor="")
            log("DrawingController: Canvas cursor reset.")

        self.parent_frame.settings_panel.update_history_buttons()
        self.parent_frame.update_display(quality="final")

    def _try_finalize_drawing(self):
        if not self.drawing_mode_active:
            log("DrawingController: Finalize drawing called but not in drawing mode.")
            return

        if len(self.current_draw_points) < 3:
            CTkMessagebox(
                title=constants.MSG_DRAWING_ERROR_TITLE,
                message=constants.MSG_DRAWING_NEED_MORE_POINTS,
                icon="warning",
            )
            log(
                "DrawingController: Finalize drawing failed - less than 3 points.",
                level="WARNING",
            )
            self._stop_drawing_mode()
            return

        log("DrawingController: Finalizing drawn mask.")
        self._finalize_drawn_mask()
        self._stop_drawing_mode()

    def _cancel_drawing_action(self, event=None):
        if self.drawing_mode_active:
            log(
                "DrawingController: Drawing action canceled by user (Escape key or explicit cancel)."
            )
            self.current_draw_points = []
            self._stop_drawing_mode()
            return "break"
        log("DrawingController: Cancel drawing action called but not in drawing mode.")
        return None

    def _handle_enter_key_press(self, event=None):
        if self.drawing_mode_active:
            log("DrawingController: Enter key pressed, attempting to finalize drawing.")
            self._try_finalize_drawing()
            return "break"
        log("DrawingController: Enter key press ignored, not in drawing mode.")
        return None

    def _finalize_drawn_mask(self):
        if not self.current_draw_points or len(self.current_draw_points) < 3:
            log(
                "DrawingController: _finalize_drawn_mask - No points or less than 3 points, skipping mask creation."
            )
            return

        image_data = self.application_model.image_data
        if image_data.original_image is None:
            log(
                "DrawingController: _finalize_drawn_mask - No original image, skipping mask creation."
            )
            return

        mask_shape_for_skimage = (
            image_data.original_image.height,
            image_data.original_image.width,
        )

        if image_data.mask_array is None:
            current_mask_array_np = np.zeros(
                mask_shape_for_skimage, dtype=np.dtype(constants.MASK_DTYPE_NAME)
            )
            new_cell_id = 1
        else:
            if (
                not image_data.mask_array.flags.writeable
                or image_data.mask_array.dtype != np.dtype(constants.MASK_DTYPE_NAME)
            ):
                current_mask_array_np = image_data.mask_array.astype(
                    np.dtype(constants.MASK_DTYPE_NAME)
                ).copy()
            else:
                current_mask_array_np = image_data.mask_array.copy()

            if current_mask_array_np.size == 0:
                current_mask_array_np = np.zeros(
                    mask_shape_for_skimage, dtype=np.dtype(constants.MASK_DTYPE_NAME)
                )
                new_cell_id = 1
            elif np.max(current_mask_array_np) == 0:
                new_cell_id = 1
            else:
                new_cell_id = np.max(current_mask_array_np) + 1

        points_r = np.array([p[1] for p in self.current_draw_points], dtype=np.double)
        points_c = np.array([p[0] for p in self.current_draw_points], dtype=np.double)

        rr, cc = polygon(points_r, points_c, shape=mask_shape_for_skimage)
        log(
            f"DrawingController: Polygon drawn with {len(self.current_draw_points)} points, creating mask for new cell ID: {new_cell_id}."
        )

        current_mask_array_np[rr, cc] = new_cell_id

        self.application_model.add_user_drawn_cell_mask(
            current_mask_array_np, new_cell_id
        )
        log(
            f"DrawingController: User-drawn mask with ID {new_cell_id} added to application model."
        )

        self.current_draw_points = []

    def _undo_draw_action(self):
        if self.drawing_history_pointer > 0:
            self.drawing_history_pointer -= 1
            self.current_draw_points = self.drawing_history_stack[
                self.drawing_history_pointer
            ].copy()
            log(
                f"DrawingController: Undo draw action. Pointer at {self.drawing_history_pointer}, {len(self.current_draw_points)} points."
            )
            self.parent_frame.settings_panel.update_history_buttons()
            self.parent_frame.update_display(quality="interactive")
        else:
            log("DrawingController: Cannot undo draw action - at beginning of history.")

    def _redo_draw_action(self):
        if self.drawing_history_pointer < len(self.drawing_history_stack) - 1:
            self.drawing_history_pointer += 1
            self.current_draw_points = self.drawing_history_stack[
                self.drawing_history_pointer
            ].copy()
            log(
                f"DrawingController: Redo draw action. Pointer at {self.drawing_history_pointer}, {len(self.current_draw_points)} points."
            )
            self.parent_frame.settings_panel.update_history_buttons()
            self.parent_frame.update_display(quality="interactive")
        else:
            log("DrawingController: Cannot redo draw action - at end of history.")

    def handle_canvas_click_for_draw(self, event):
        if self.application_model.image_data.original_image is None:
            log("DrawingController: Canvas click for draw ignored - no original image.")
            return "break"

        canvas_x, canvas_y = event.x, event.y
        zoom, pan_x, pan_y = self.application_model.pan_zoom_state.get_params()

        original_x = (canvas_x - pan_x) / zoom
        original_y = (canvas_y - pan_y) / zoom

        img_w = self.application_model.image_data.original_image.width
        img_h = self.application_model.image_data.original_image.height
        if not (0 <= original_x < img_w and 0 <= original_y < img_h):
            log(
                f"DrawingController: Click for draw at ({original_x},{original_y}) is outside image bounds. Ignoring."
            )
            return "break"

        self.current_draw_points.append((original_x, original_y))
        log(
            f"DrawingController: Point ({original_x}, {original_y}) added for drawing. Total points: {len(self.current_draw_points)}."
        )

        new_history_state = self.current_draw_points.copy()

        # Prune redo stack if new action is taken
        if self.drawing_history_pointer < len(self.drawing_history_stack) - 1:
            self.drawing_history_stack = self.drawing_history_stack[
                : self.drawing_history_pointer + 1
            ]

        self.drawing_history_stack.append(new_history_state)
        self.drawing_history_pointer += 1

        self.parent_frame.settings_panel.update_history_buttons()
        self.parent_frame.update_display(quality="interactive")

        return "break"

    def can_undo_draw(self):
        can = self.drawing_mode_active and self.drawing_history_pointer > 0
        return can

    def can_redo_draw(self):
        can = (
            self.drawing_mode_active
            and self.drawing_history_pointer < len(self.drawing_history_stack) - 1
        )
        return can
