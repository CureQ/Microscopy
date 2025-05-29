import numpy as np
from CTkMessagebox import CTkMessagebox
from skimage.draw import polygon

from . import constants
from .application_model import ApplicationModel


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
            print("DrawingController: Start drawing mode failed - no image loaded.")
            return

        self.drawing_mode_active = True
        self.current_draw_points = []
        self.drawing_history_stack = [
            []
        ]  # Start with an empty set of points for current drawing session
        self.drawing_history_pointer = 0
        print(
            "DrawingController: Drawing mode started. Polygon points and history reset."
        )

        if self.parent_frame.draw_mask_button:
            self.parent_frame.draw_mask_button.configure(
                text=constants.UI_TEXT_FINALIZE_POLYGON_BUTTON,
                command=self._try_finalize_drawing,
            )
            print(
                "DrawingController: Draw mask button configured to 'Finalize Polygon'."
            )
        if self.parent_frame.segment_button:
            self.parent_frame.segment_button.configure(state="disabled")
            print("DrawingController: Segment button disabled.")
        if self.parent_frame.image_canvas:
            self.parent_frame.image_canvas.config(cursor="crosshair")
            print("DrawingController: Canvas cursor set to crosshair.")

        self.parent_frame.update_history_buttons()
        self.parent_frame.update_display(quality="interactive")

    def _stop_drawing_mode(self):
        self.drawing_mode_active = False
        self.drawing_history_stack = []
        self.drawing_history_pointer = -1
        print(
            "DrawingController: Drawing mode stopped. Polygon points and history cleared."
        )

        if self.parent_frame.draw_mask_button:
            self.parent_frame.draw_mask_button.configure(
                text=constants.UI_TEXT_START_DRAWING_BUTTON,
                command=self._start_drawing_mode,
            )
            print(
                "DrawingController: Draw mask button configured back to 'Start Drawing'."
            )
        if (
            self.parent_frame.segment_button
            and self.application_model.image_data.original_image is not None
        ):
            self.parent_frame.segment_button.configure(state="normal")
            print("DrawingController: Segment button re-enabled.")
        if self.parent_frame.image_canvas:
            self.parent_frame.image_canvas.config(cursor="")
            print("DrawingController: Canvas cursor reset.")

        self.parent_frame.update_history_buttons()
        self.parent_frame.update_display(quality="final")

    def _try_finalize_drawing(self):
        if not self.drawing_mode_active:
            print("DrawingController: Finalize drawing called but not in drawing mode.")
            return

        if len(self.current_draw_points) < 3:
            CTkMessagebox(
                title=constants.MSG_DRAWING_ERROR_TITLE,
                message=constants.MSG_DRAWING_NEED_MORE_POINTS,
                icon="warning",
            )
            print("DrawingController: Finalize drawing failed - less than 3 points.")
            return

        print("DrawingController: Finalizing drawn mask.")
        self._finalize_drawn_mask()
        self._stop_drawing_mode()

    def _cancel_drawing_action(self, event=None):
        if self.drawing_mode_active:
            print(
                "DrawingController: Drawing action canceled by user (Escape key or explicit cancel)."
            )
            self.current_draw_points = []
            self._stop_drawing_mode()
            return "break"
        print(
            "DrawingController: Cancel drawing action called but not in drawing mode."
        )
        return None

    def _handle_enter_key_press(self, event=None):
        if self.drawing_mode_active:
            print(
                "DrawingController: Enter key pressed, attempting to finalize drawing."
            )
            self._try_finalize_drawing()
            return "break"
        print("DrawingController: Enter key press ignored, not in drawing mode.")
        return None

    def _finalize_drawn_mask(self):
        if not self.current_draw_points or len(self.current_draw_points) < 3:
            print(
                "DrawingController: _finalize_drawn_mask - No points or less than 3 points, skipping mask creation."
            )
            return

        image_data = self.application_model.image_data
        if image_data.original_image is None:
            print(
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
        print(
            f"DrawingController: Polygon drawn with {len(self.current_draw_points)} points, creating mask for new cell ID: {new_cell_id}."
        )

        current_mask_array_np[rr, cc] = new_cell_id

        self.application_model.add_user_drawn_cell_mask(
            current_mask_array_np, new_cell_id
        )
        print(
            f"DrawingController: User-drawn mask with ID {new_cell_id} added to application model."
        )

        self.current_draw_points = []

    def _undo_draw_action(self):
        if self.drawing_history_pointer > 0:
            self.drawing_history_pointer -= 1
            self.current_draw_points = self.drawing_history_stack[
                self.drawing_history_pointer
            ].copy()
            print(
                f"DrawingController: Undo draw action. Pointer at {self.drawing_history_pointer}, {len(self.current_draw_points)} points."
            )
            self.parent_frame.update_history_buttons()
            self.parent_frame.update_display(quality="interactive")
        else:
            print(
                "DrawingController: Cannot undo draw action - at beginning of history."
            )

    def _redo_draw_action(self):
        if self.drawing_history_pointer < len(self.drawing_history_stack) - 1:
            self.drawing_history_pointer += 1
            self.current_draw_points = self.drawing_history_stack[
                self.drawing_history_pointer
            ].copy()
            print(
                f"DrawingController: Redo draw action. Pointer at {self.drawing_history_pointer}, {len(self.current_draw_points)} points."
            )
            self.parent_frame.update_history_buttons()
            self.parent_frame.update_display(quality="interactive")
        else:
            print("DrawingController: Cannot redo draw action - at end of history.")

    def handle_canvas_click_for_draw(self, event):
        if self.application_model.image_data.original_image is None:
            print(
                "DrawingController: Canvas click for draw ignored - no original image."
            )
            return "break"

        canvas_x, canvas_y = event.x, event.y
        zoom, pan_x, pan_y = self.application_model.pan_zoom_state.get_params()

        original_x = (canvas_x - pan_x) / zoom
        original_y = (canvas_y - pan_y) / zoom

        img_w = self.application_model.image_data.original_image.width
        img_h = self.application_model.image_data.original_image.height
        if not (0 <= original_x < img_w and 0 <= original_y < img_h):
            print(
                f"DrawingController: Click for draw at ({original_x},{original_y}) is outside image bounds. Ignoring."
            )
            return "break"

        self.current_draw_points.append((original_x, original_y))
        print(
            f"DrawingController: Point ({original_x}, {original_y}) added for drawing. Total points: {len(self.current_draw_points)}."
        )

        new_points_state = self.current_draw_points.copy()
        if self.drawing_history_pointer < len(self.drawing_history_stack) - 1:
            self.drawing_history_stack = self.drawing_history_stack[
                : self.drawing_history_pointer + 1
            ]
            print(
                "DrawingController: Drawing history truncated for new point after undo."
            )

        self.drawing_history_stack.append(new_points_state)
        self.drawing_history_pointer = len(self.drawing_history_stack) - 1

        self.parent_frame.update_history_buttons()
        self.parent_frame.update_display(quality="interactive")
        print(
            f"DrawingController: Drawing history updated. Pointer at {self.drawing_history_pointer}."
        )
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
