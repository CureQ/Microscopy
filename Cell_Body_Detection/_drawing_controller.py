import numpy as np
from CTkMessagebox import CTkMessagebox
from skimage.draw import polygon

from . import constants


# --- Drawing Controller Class ---
class DrawingController:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame  # Reference to cell_body_frame
        self.drawing_mode_active = False
        self.current_draw_points = []  # Stores original image coordinates
        self.drawing_history_stack = []  # For undo/redo of points
        self.drawing_history_pointer = -1  # Pointer for drawing_history_stack

    def _start_drawing_mode(self):
        if self.parent_frame.image_view_model.original_image is None:
            CTkMessagebox(
                title=constants.MSG_DRAWING_ERROR_TITLE,
                message=constants.MSG_DRAWING_LOAD_IMAGE_FIRST,
                icon="warning",
            )
            return

        self.drawing_mode_active = True
        self.current_draw_points = []
        self.drawing_history_stack = [[]]  # Start with an empty set of points
        self.drawing_history_pointer = 0

        if self.parent_frame.draw_mask_button:
            self.parent_frame.draw_mask_button.configure(
                text=constants.UI_TEXT_FINALIZE_POLYGON_BUTTON,
                command=self._try_finalize_drawing,
            )
        if self.parent_frame.segment_button:
            self.parent_frame.segment_button.configure(state="disabled")
        if self.parent_frame.image_canvas:
            self.parent_frame.image_canvas.config(cursor="crosshair")

        self.parent_frame.update_history_buttons()
        self.parent_frame.update_display(quality="interactive")

    def _stop_drawing_mode(self):
        self.drawing_mode_active = False
        self.drawing_history_stack = []
        self.drawing_history_pointer = -1

        if self.parent_frame.draw_mask_button:
            self.parent_frame.draw_mask_button.configure(
                text=constants.UI_TEXT_START_DRAWING_BUTTON,
                command=self._start_drawing_mode,
            )
        if (
            self.parent_frame.segment_button
            and self.parent_frame.image_view_model.original_image is not None
        ):
            self.parent_frame.segment_button.configure(state="normal")
        if self.parent_frame.image_canvas:
            self.parent_frame.image_canvas.config(cursor="")

        self.parent_frame.update_history_buttons()
        self.parent_frame.update_display(quality="final")

    def _try_finalize_drawing(self):
        if not self.drawing_mode_active:
            return

        if len(self.current_draw_points) < 3:
            CTkMessagebox(
                title=constants.MSG_DRAWING_ERROR_TITLE,
                message=constants.MSG_DRAWING_NEED_MORE_POINTS,
                icon="warning",
            )
            self.current_draw_points = []
            self._stop_drawing_mode()
            return

        self._finalize_drawn_mask()
        self._stop_drawing_mode()  # Resets button and state

    def _cancel_drawing_action(self, event=None):
        if self.drawing_mode_active:
            self.current_draw_points = []  # Ensure points are cleared
            self._stop_drawing_mode()
            return "break"
        return None

    def _handle_enter_key_press(self, event=None):
        if self.drawing_mode_active:
            self._try_finalize_drawing()
            return "break"
        return None

    def _finalize_drawn_mask(self):
        if not self.current_draw_points or len(self.current_draw_points) < 3:
            return

        ivm = self.parent_frame.image_view_model
        if ivm.mask_array is None:
            if ivm.original_image is None:
                return
            mask_shape = (
                ivm.original_image.height,
                ivm.original_image.width,
            )
            ivm.mask_array = np.zeros(
                mask_shape, dtype=np.dtype(constants.MASK_DTYPE_NAME)
            )
            new_cell_id = 1
        else:
            if ivm.mask_array.size == 0:  # Empty mask array
                mask_shape = (  # Should ideally get from original_image
                    ivm.original_image.height,
                    ivm.original_image.width,
                )
                ivm.mask_array = np.zeros(
                    mask_shape, dtype=np.dtype(constants.MASK_DTYPE_NAME)
                )
                new_cell_id = 1
            elif np.max(ivm.mask_array) == 0:  # Only background
                new_cell_id = 1
            else:
                new_cell_id = np.max(ivm.mask_array) + 1

        mask_shape_for_skimage = (
            ivm.original_image.height,
            ivm.original_image.width,
        )
        points_r = np.array([p[1] for p in self.current_draw_points], dtype=np.double)
        points_c = np.array([p[0] for p in self.current_draw_points], dtype=np.double)

        rr, cc = polygon(points_r, points_c, shape=mask_shape_for_skimage)

        if not ivm.mask_array.flags.writeable:
            ivm.mask_array = ivm.mask_array.copy()
        if ivm.mask_array.dtype != np.dtype(constants.MASK_DTYPE_NAME):
            ivm.mask_array = ivm.mask_array.astype(np.dtype(constants.MASK_DTYPE_NAME))

        ivm.mask_array[rr, cc] = new_cell_id
        ivm.included_cells.add(new_cell_id)
        ivm.add_user_drawn_cell(new_cell_id)  # Mark as user-drawn

        if self.parent_frame.image_view_renderer:
            self.parent_frame.image_view_renderer.invalidate_caches()
        self.parent_frame.history_controller.record_state()  # Record this change

        self.parent_frame.update_display(quality="final")

    def _undo_draw_action(self):
        if self.drawing_history_pointer > 0:
            self.drawing_history_pointer -= 1
            self.current_draw_points = self.drawing_history_stack[
                self.drawing_history_pointer
            ].copy()
            self.parent_frame.update_history_buttons()
            self.parent_frame.update_display(quality="interactive")

    def _redo_draw_action(self):
        if self.drawing_history_pointer < len(self.drawing_history_stack) - 1:
            self.drawing_history_pointer += 1
            self.current_draw_points = self.drawing_history_stack[
                self.drawing_history_pointer
            ].copy()
            self.parent_frame.update_history_buttons()
            self.parent_frame.update_display(quality="interactive")

    def handle_canvas_click_for_draw(self, event):
        if self.parent_frame.image_view_model.original_image is None:
            return "break"

        canvas_x, canvas_y = event.x, event.y
        zoom, pan_x, pan_y = self.parent_frame.pan_zoom_model.get_params()

        original_x = (canvas_x - pan_x) / zoom
        original_y = (canvas_y - pan_y) / zoom

        img_w = self.parent_frame.image_view_model.original_image.width
        img_h = self.parent_frame.image_view_model.original_image.height
        if not (0 <= original_x < img_w and 0 <= original_y < img_h):
            return "break"

        self.current_draw_points.append((original_x, original_y))

        new_points_state = self.current_draw_points.copy()
        # Before appending new state, slice history if pointer is not at the end
        if self.drawing_history_pointer < len(self.drawing_history_stack) - 1:
            self.drawing_history_stack = self.drawing_history_stack[
                : self.drawing_history_pointer + 1
            ]

        self.drawing_history_stack.append(new_points_state)
        self.drawing_history_pointer = (
            len(self.drawing_history_stack) - 1
        )  # Pointer is now at the new state

        self.parent_frame.update_history_buttons()
        self.parent_frame.update_display(quality="interactive")
        return "break"

    def can_undo_draw(self):
        return self.drawing_history_pointer > 0

    def can_redo_draw(self):
        # Ensure pointer is less than the last valid index (len - 1)
        return self.drawing_history_pointer < len(self.drawing_history_stack) - 1
