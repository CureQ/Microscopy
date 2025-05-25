import customtkinter as ctk
import numpy as np
from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageTk,
)
from scipy import ndimage

from . import constants


class PanZoomModel:
    def __init__(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.last_drag_x = 0
        self.last_drag_y = 0
        self.min_zoom_to_fit = 1.0  # Zoom level to fit the current image

    def _calculate_min_zoom_to_fit(
        self, canvas_width, canvas_height, img_width, img_height
    ):
        if img_width == 0 or img_height == 0 or canvas_width == 0 or canvas_height == 0:
            return 1.0  # Default if no dimensions

        # Calculate zoom to fit width and height
        zoom_if_fit_width = canvas_width / img_width
        zoom_if_fit_height = canvas_height / img_height

        # Choose the smaller of the two to ensure the whole image fits
        return min(
            zoom_if_fit_width, zoom_if_fit_height, 1.0
        )  # Also ensure it doesn't go above 1.0 initially if image is smaller

    def reset_for_new_image(
        self, canvas_width=0, canvas_height=0, img_width=0, img_height=0
    ):
        if img_width > 0 and img_height > 0 and canvas_width > 1 and canvas_height > 1:
            self.min_zoom_to_fit = self._calculate_min_zoom_to_fit(
                canvas_width, canvas_height, img_width, img_height
            )
            self.zoom_level = self.min_zoom_to_fit

            # Center the image at this initial (potentially < 1.0) zoom level
            zoomed_img_width = img_width * self.zoom_level
            zoomed_img_height = img_height * self.zoom_level
            self.pan_x = (canvas_width - zoomed_img_width) / 2.0
            self.pan_y = (canvas_height - zoomed_img_height) / 2.0
        else:
            self.min_zoom_to_fit = 1.0
            self.zoom_level = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0

        self.last_drag_x = 0
        self.last_drag_y = 0

    def get_params(self):
        return self.zoom_level, self.pan_x, self.pan_y


class ImageViewModel:
    def __init__(self):
        self.original_image = None
        self.mask_array = None
        self.included_cells = set()
        self.user_drawn_cell_ids = set()  # For tracking user-drawn cells

    def reset_for_new_image(self):
        self.original_image = None
        self.mask_array = None
        self.included_cells = set()
        self.user_drawn_cell_ids.clear()  # Reset user-drawn IDs

    def set_image_data(self, original_image):
        self.original_image = original_image

    def set_segmentation_result(self, mask_array):
        self.mask_array = mask_array
        all_current_mask_ids = set()
        if mask_array is not None and mask_array.size > 0:
            all_current_mask_ids = set(np.unique(mask_array)) - {0}
            self.included_cells = (
                all_current_mask_ids.copy()
            )  # By default, include all found by segmentation
        else:
            self.included_cells = set()

        # Reconcile user_drawn_cell_ids: only keep those that still exist in the new mask_array
        self.user_drawn_cell_ids &= all_current_mask_ids

    def toggle_cell_inclusion(self, cell_id):
        if cell_id in self.included_cells:
            self.included_cells.remove(cell_id)
        else:
            self.included_cells.add(cell_id)

    def add_user_drawn_cell(self, cell_id):
        """Marks a cell ID as user-drawn."""
        self.user_drawn_cell_ids.add(cell_id)

    def get_snapshot_data(self):
        return {
            "mask_array": self.mask_array.copy()
            if self.mask_array is not None
            else None,
            "included_cells": self.included_cells.copy(),
            "user_drawn_cell_ids": self.user_drawn_cell_ids.copy(),  # Add to snapshot
        }

    def restore_from_snapshot(self, snapshot_data):
        self.mask_array = (
            snapshot_data["mask_array"].copy()
            if snapshot_data["mask_array"] is not None
            else None
        )
        self.included_cells = snapshot_data["included_cells"].copy()
        # Restore user_drawn_cell_ids, provide default if key is missing (for older states)
        self.user_drawn_cell_ids = snapshot_data.get(
            "user_drawn_cell_ids", set()
        ).copy()


class ImageViewRenderer:
    def __init__(
        self, canvas_ref, pan_zoom_model_ref, image_view_model_ref, cell_body_frame_ref
    ):
        self.image_canvas = canvas_ref
        self.pan_zoom_model = pan_zoom_model_ref
        self.image_view_model = image_view_model_ref
        self.parent_frame = (
            cell_body_frame_ref  # To access display options and _get_exact_boundaries
        )

        self.tk_image_on_canvas = None
        self._update_display_retry_id = None
        self.dilation_iterations = constants.DILATION_ITERATIONS_FOR_BOUNDARY_DISPLAY

        # Overlay Caching Attributes
        self._cached_full_res_mask_rgb = None
        self._cached_mask_ids_tuple_state = None
        self._cached_show_deselected_mask_state = None

        self._cached_full_res_boundary_pil = None
        self._cached_boundary_ids_tuple_state = None
        self._cached_show_deselected_boundary_state = None
        self._cached_boundary_color_state = None

        # Cache for cell numbering
        self._cached_cell_number_positions = None
        self._cached_cell_number_ids_tuple_state = None
        self._cached_cell_number_show_deselected_state = None

        # Drawing feedback attributes
        self.draw_feedback_color = (
            constants.DRAW_FEEDBACK_COLOR
        )  # For lines and intermediate points
        self.draw_first_point_color = constants.DRAW_FIRST_POINT_COLOR
        self.draw_last_point_color = constants.DRAW_LAST_POINT_COLOR
        self.draw_point_radius = constants.DRAW_POINT_RADIUS  # pixels on canvas

    def invalidate_caches(self):
        self._cached_full_res_mask_rgb = None
        self._cached_mask_ids_tuple_state = None
        self._cached_show_deselected_mask_state = None  # Clear renamed cache state

        self._cached_full_res_boundary_pil = None
        self._cached_boundary_ids_tuple_state = None
        self._cached_show_deselected_boundary_state = None  # Clear renamed cache state
        self._cached_boundary_color_state = None

        self._cached_cell_number_positions = None
        self._cached_cell_number_ids_tuple_state = None
        self._cached_cell_number_show_deselected_state = None

    def _dilate_boundary(self, bool_mask, iterations=1):
        """Simple dilation for a boolean mask using NumPy."""
        if not bool_mask.any() or iterations < 1:
            return bool_mask

        dilated_mask = bool_mask.copy()
        for _ in range(iterations):
            to_add = np.zeros_like(dilated_mask, dtype=bool)
            true_coords = np.argwhere(dilated_mask)
            for r, c in true_coords:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < dilated_mask.shape[0]
                            and 0 <= nc < dilated_mask.shape[1]
                        ):
                            to_add[nr, nc] = True
            dilated_mask |= to_add
        return dilated_mask

    def render(self, quality="final"):
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            if self._update_display_retry_id:
                self.image_canvas.after_cancel(self._update_display_retry_id)
            self._update_display_retry_id = self.image_canvas.after(
                constants.UI_RENDER_RETRY_DELAY_MS,
                lambda q=quality: self.render(quality=q),
            )
            return

        if self.image_view_model.original_image is None:
            self.image_canvas.delete("all")
            text_color = "white"
            if (
                hasattr(self.parent_frame, "_apply_appearance_mode")
                and hasattr(ctk, "ThemeManager")
                and ctk.ThemeManager.theme
            ):
                try:
                    text_color = self.parent_frame._apply_appearance_mode(
                        ctk.ThemeManager.theme["CTkLabel"]["text_color"]
                    )
                except KeyError:
                    pass
            self.image_canvas.create_text(
                canvas_width / 2,
                canvas_height / 2,
                text=constants.MSG_SELECT_IMAGE_PROMPT,  # "Select an Image"
                fill=text_color,
                font=ctk.CTkFont(size=16),
            )
            self.tk_image_on_canvas = None
            if self.parent_frame.stats_label:
                self.parent_frame.stats_label.configure(
                    text=constants.UI_TEXT_STATS_LABEL_DEFAULT
                )
            return

        zoom, pan_x, pan_y = self.pan_zoom_model.get_params()
        pil_original_image = self.image_view_model.original_image

        new_width = int(pil_original_image.width * zoom)
        new_height = int(pil_original_image.height * zoom)

        if new_width <= 0 or new_height <= 0:
            self.image_canvas.delete("all")
            self.tk_image_on_canvas = None
            return

        resample_filter = Image.LANCZOS if quality == "final" else Image.NEAREST
        zoomed_image = pil_original_image.resize(
            (max(1, new_width), max(1, new_height)), resample_filter
        )

        canvas_view_pil = Image.new(
            "RGB", (canvas_width, canvas_height), constants.COLOR_BLACK_STR
        )

        src_x1_on_zoomed_img = int(-pan_x) if pan_x < 0 else 0
        paste_dst_x_on_canvas = int(pan_x) if pan_x > 0 else 0
        src_y1_on_zoomed_img = int(-pan_y) if pan_y < 0 else 0
        paste_dst_y_on_canvas = int(pan_y) if pan_y > 0 else 0

        width_to_copy = min(
            zoomed_image.width - src_x1_on_zoomed_img,
            canvas_width - paste_dst_x_on_canvas,
        )
        height_to_copy = min(
            zoomed_image.height - src_y1_on_zoomed_img,
            canvas_height - paste_dst_y_on_canvas,
        )
        crop_box = None  # Initialize crop_box
        if width_to_copy > 0 and height_to_copy > 0:
            crop_box = (
                src_x1_on_zoomed_img,
                src_y1_on_zoomed_img,
                src_x1_on_zoomed_img + width_to_copy,
                src_y1_on_zoomed_img + height_to_copy,
            )
            cropped_visible_part = zoomed_image.crop(crop_box)
            canvas_view_pil.paste(
                cropped_visible_part, (paste_dst_x_on_canvas, paste_dst_y_on_canvas)
            )

        base_display_img = canvas_view_pil
        if not self.parent_frame.show_original.get():
            base_display_img = Image.new(
                "RGB", (canvas_width, canvas_height), constants.COLOR_BLACK_STR
            )

        current_mask_array = self.image_view_model.mask_array
        all_mask_ids = set()
        if current_mask_array is not None and current_mask_array.size > 0:
            unique_ids = np.unique(current_mask_array)
            all_mask_ids = set(unique_ids[unique_ids != 0])  # Exclude 0 background

        ids_to_process_for_display = set()
        show_deselected_mode = self.parent_frame.show_only_deselected.get()

        if show_deselected_mode:
            # If mode is active, ids_to_process are those in all_mask_ids but NOT in included_cells
            deselected_ids = all_mask_ids - self.image_view_model.included_cells
            ids_to_process_for_display = deselected_ids
        else:
            # Default mode: ids_to_process are those in included_cells (and also in all_mask_ids)
            ids_to_process_for_display = (
                self.image_view_model.included_cells.intersection(all_mask_ids)
            )

        current_ids_tuple_for_display = tuple(sorted(list(ids_to_process_for_display)))

        if current_mask_array is not None and current_mask_array.size > 0:
            # MASK OVERLAY
            if self.parent_frame.show_mask.get():
                if (
                    self._cached_full_res_mask_rgb is None
                    or self._cached_mask_ids_tuple_state
                    != current_ids_tuple_for_display
                    or self._cached_show_deselected_mask_state
                    != show_deselected_mode  # Use correct state var
                ):
                    self._cached_full_res_mask_rgb = Image.new(
                        "RGB", pil_original_image.size, constants.COLOR_BLACK_STR
                    )
                    rng = np.random.default_rng(seed=constants.RANDOM_SEED_MASKS)
                    unique_ids_for_colormap = np.unique(current_mask_array)
                    color_map = {
                        uid: tuple(rng.integers(50, 200, size=3))
                        for uid in unique_ids_for_colormap
                        if uid != 0
                    }
                    temp_mask_np = np.zeros(
                        (*current_mask_array.shape, 3), dtype=np.uint8
                    )
                    for cell_id_val in ids_to_process_for_display:
                        if cell_id_val != 0:
                            temp_mask_np[current_mask_array == cell_id_val] = (
                                color_map.get(
                                    cell_id_val, constants.COLOR_BLACK_RGB
                                )  # (255,0,0) fallback
                            )
                    self._cached_full_res_mask_rgb = Image.fromarray(temp_mask_np)
                    self._cached_mask_ids_tuple_state = current_ids_tuple_for_display
                    self._cached_show_deselected_mask_state = (
                        show_deselected_mode  # Store correct state var
                    )

                overlay_resample_filter = Image.NEAREST
                zoomed_mask_rgb = self._cached_full_res_mask_rgb.resize(
                    (max(1, new_width), max(1, new_height)),
                    resample=overlay_resample_filter,
                )
                if crop_box and width_to_copy > 0 and height_to_copy > 0:
                    cropped_mask = zoomed_mask_rgb.crop(crop_box)
                    temp_overlay = Image.new(
                        "RGB", (canvas_width, canvas_height), (0, 0, 0)
                    )
                    temp_overlay.paste(
                        cropped_mask, (paste_dst_x_on_canvas, paste_dst_y_on_canvas)
                    )
                    base_display_img = Image.blend(
                        base_display_img.convert("RGB"),
                        temp_overlay,
                        alpha=constants.MASK_BLEND_ALPHA,
                    )

            # BOUNDARY OVERLAY
            if self.parent_frame.show_boundaries.get():
                current_boundary_color_name = self.parent_frame.boundary_color.get()
                if (
                    self._cached_full_res_boundary_pil is None
                    or self._cached_boundary_ids_tuple_state
                    != current_ids_tuple_for_display
                    or self._cached_show_deselected_boundary_state
                    != show_deselected_mode  # Use correct state var
                    or self._cached_boundary_color_state != current_boundary_color_name
                ):
                    exact_boundaries = self.parent_frame._get_exact_boundaries(
                        current_mask_array
                    )
                    boundary_to_draw = np.zeros_like(exact_boundaries, dtype=bool)
                    for cid in ids_to_process_for_display:
                        boundary_to_draw |= exact_boundaries & (
                            current_mask_array == cid
                        )
                    if self.dilation_iterations > 0:
                        dilated_boundary_for_cache = self._dilate_boundary(
                            boundary_to_draw, iterations=self.dilation_iterations
                        )
                    else:
                        dilated_boundary_for_cache = boundary_to_draw
                    self._cached_full_res_boundary_pil = Image.fromarray(
                        dilated_boundary_for_cache.astype(np.uint8) * 255
                    )
                    self._cached_boundary_ids_tuple_state = (
                        current_ids_tuple_for_display
                    )
                    self._cached_show_deselected_boundary_state = (
                        show_deselected_mode  # Store correct state var
                    )
                    self._cached_boundary_color_state = current_boundary_color_name

                overlay_resample_filter = Image.NEAREST
                zoomed_boundary_pil = self._cached_full_res_boundary_pil.resize(
                    (max(1, new_width), max(1, new_height)),
                    resample=overlay_resample_filter,
                )
                if crop_box and width_to_copy > 0 and height_to_copy > 0:
                    cropped_boundary = zoomed_boundary_pil.crop(crop_box)
                    boundary_color_map_pil = constants.BOUNDARY_COLOR_MAP_PIL
                    chosen_color = boundary_color_map_pil.get(
                        current_boundary_color_name,
                        constants.BOUNDARY_COLOR_MAP_PIL["Green"],  # Default green
                    )
                    final_boundary_on_canvas_bool = Image.new(
                        "L", (canvas_width, canvas_height), 0
                    )
                    final_boundary_on_canvas_bool.paste(
                        cropped_boundary, (paste_dst_x_on_canvas, paste_dst_y_on_canvas)
                    )
                    base_np = np.array(base_display_img.convert("RGB"))
                    boundary_pixels_np = np.array(final_boundary_on_canvas_bool) > 0
                    base_np[boundary_pixels_np] = chosen_color
                    base_display_img = Image.fromarray(base_np)

        # --- START DRAWING FEEDBACK ---
        if (
            self.parent_frame.drawing_controller.drawing_mode_active
            and self.parent_frame.drawing_controller.current_draw_points
        ):
            # Ensure base_display_img is suitable for drawing (it should be RGB)
            if base_display_img.mode != "RGB":
                base_display_img = base_display_img.convert(
                    "RGB"
                )  # Should already be if overlays ran

            draw_on_canvas_view = ImageDraw.Draw(base_display_img)

            # Transform raw user-clicked points to canvas space for dot rendering
            # and for drawing lines connecting them directly.
            all_potential_canvas_points = []  # Stores all points transformed to canvas space
            visible_canvas_dots = []  # Store only points visible on canvas (list of dicts)

            for index, (orig_x, orig_y) in enumerate(
                self.parent_frame.drawing_controller.current_draw_points  # MODIFIED
            ):
                zoomed_pt_x = orig_x * zoom
                zoomed_pt_y = orig_y * zoom
                rel_zoomed_x = zoomed_pt_x - (src_x1_on_zoomed_img if crop_box else 0)
                rel_zoomed_y = zoomed_pt_y - (src_y1_on_zoomed_img if crop_box else 0)
                cv_x = rel_zoomed_x + paste_dst_x_on_canvas
                cv_y = rel_zoomed_y + paste_dst_y_on_canvas
                all_potential_canvas_points.append((cv_x, cv_y))

                if (
                    crop_box
                    and 0 <= rel_zoomed_x < crop_box[2] - crop_box[0]
                    and 0 <= rel_zoomed_y < crop_box[3] - crop_box[1]
                ):
                    visible_canvas_dots.append({"x": cv_x, "y": cv_y, "index": index})

            # Draw dots for actual user clicks (visible ones)
            if visible_canvas_dots:
                num_total_points = len(
                    self.parent_frame.drawing_controller.current_draw_points
                )  # MODIFIED
                for point_info in visible_canvas_dots:
                    pt_x, pt_y, current_index = (
                        point_info["x"],
                        point_info["y"],
                        point_info["index"],
                    )
                    point_color = self.draw_feedback_color
                    if num_total_points == 1 and current_index == 0:
                        point_color = self.draw_first_point_color
                    elif num_total_points > 1:
                        if current_index == 0:
                            point_color = self.draw_first_point_color
                        elif current_index == num_total_points - 1:
                            point_color = self.draw_last_point_color
                    draw_on_canvas_view.ellipse(
                        (
                            pt_x - self.draw_point_radius,
                            pt_y - self.draw_point_radius,
                            pt_x + self.draw_point_radius,
                            pt_y + self.draw_point_radius,
                        ),
                        fill=point_color,
                        outline=point_color,
                    )

            # Draw lines connecting ALL potential points (ImageDraw will clip)
            # This ensures lines extend off-canvas correctly if points are outside view.
            if len(all_potential_canvas_points) > 1:
                draw_on_canvas_view.line(
                    all_potential_canvas_points,
                    fill=self.draw_feedback_color,
                    width=constants.DRAW_FEEDBACK_LINE_WIDTH,
                )

            # Draw closing line based on TRUE first and last points of the WHOLE polygon
            if (
                len(self.parent_frame.drawing_controller.current_draw_points)
                >= 2  # MODIFIED
            ):  # Use original point count for this decision
                first_canvas_pt = all_potential_canvas_points[0]
                last_canvas_pt = all_potential_canvas_points[-1]
                draw_on_canvas_view.line(
                    [last_canvas_pt, first_canvas_pt],
                    fill=self.draw_feedback_color,
                    width=constants.DRAW_FEEDBACK_LINE_WIDTH,
                    joint="curve",
                )
        # --- END DRAWING FEEDBACK ---

        # --- START CELL NUMBERING ---
        if (
            self.parent_frame.show_cell_numbers.get()
            and current_mask_array is not None
            and current_mask_array.size > 0
            and ids_to_process_for_display
        ):
            if base_display_img.mode != "RGB":
                base_display_img = base_display_img.convert("RGB")
            draw_on_canvas_view = ImageDraw.Draw(base_display_img)

            # Determine font size based on zoom level - make it adaptive but not too small
            # Aim for a font size that's roughly 10-15px on the original image scale, then scaled by zoom.
            # This needs to be translated to a PIL font size for the canvas_view_pil image.
            # Let's try to make the numbers roughly 1/20th of the smaller dimension of a typical cell.
            # For now, a fixed size scaled by zoom, clamped.
            base_font_size_orig_img = (
                constants.CELL_NUMBERING_FONT_SIZE_ORIG_IMG
            )  # Approximate size on original image pixels
            # This font size is for drawing on the canvas_view_pil, which is at canvas resolution.
            # The numbers should appear consistent in size relative to the cells on the screen.
            # So, the size should scale with the zoom applied to the cell, but drawn on the unzoomed canvas view part.

            # The text is drawn on base_display_img, which is the canvas view.
            # We need to map cell centers (from original image coords) to canvas view coords.

            # Get boundary color for text
            current_boundary_color_name = self.parent_frame.boundary_color.get()
            boundary_color_map_pil = (
                constants.BOUNDARY_COLOR_MAP_PIL
            )  # Moved to constants
            text_color_tuple = boundary_color_map_pil.get(
                current_boundary_color_name,
                constants.BOUNDARY_COLOR_MAP_PIL["Green"],  # Default green
            )

            try:
                # Attempt to load a common system font. Size will be adjusted later.
                font = ImageFont.truetype(
                    constants.DEFAULT_FONT, size=constants.DEFAULT_FALLBACK_FONT_SIZE
                )  # Placeholder size
            except IOError:
                font = ImageFont.load_default()  # Fallback

            # --- Cell Position Calculation and Caching ---
            if (
                self._cached_cell_number_positions is None
                or self._cached_cell_number_ids_tuple_state
                != current_ids_tuple_for_display
                or self._cached_cell_number_show_deselected_state
                != show_deselected_mode
            ):
                # print("DEBUG: Recalculating cell number positions") # Optional debug
                cell_info_for_sorting = []
                img_h_orig, img_w_orig = (
                    pil_original_image.height,
                    pil_original_image.width,
                )

                for cell_id_val in ids_to_process_for_display:
                    if cell_id_val != 0:
                        single_cell_mask = current_mask_array == cell_id_val
                        if np.any(single_cell_mask):
                            rows, cols = np.where(single_cell_mask)
                            top_most = np.min(rows)
                            left_most_in_top_row = np.min(cols[rows == top_most])

                            # Attempt to find a point with a 10px margin from cell boundary
                            cell_boundary_margin = (
                                constants.CELL_CENTER_FIND_MARGIN
                            )  # 10
                            eroded_mask = ndimage.binary_erosion(
                                single_cell_mask,
                                iterations=cell_boundary_margin,
                                border_value=0,
                            )

                            chosen_cx, chosen_cy = -1.0, -1.0  # Initialize

                            if np.any(eroded_mask):
                                # Try to use a point from the eroded mask
                                cy_eroded_com, cx_eroded_com = ndimage.center_of_mass(
                                    eroded_mask
                                )
                                cy_eroded_idx, cx_eroded_idx = (
                                    int(round(cy_eroded_com)),
                                    int(round(cx_eroded_com)),
                                )
                                if (
                                    0 <= cy_eroded_idx < eroded_mask.shape[0]
                                    and 0 <= cx_eroded_idx < eroded_mask.shape[1]
                                    and eroded_mask[cy_eroded_idx, cx_eroded_idx]
                                ):
                                    chosen_cx, chosen_cy = cx_eroded_com, cy_eroded_com
                                else:
                                    # COM of eroded mask is outside eroded mask, use PoA of eroded mask
                                    dist_transform_eroded = (
                                        ndimage.distance_transform_edt(eroded_mask)
                                    )
                                    pole_y_eroded, pole_x_eroded = np.unravel_index(
                                        np.argmax(dist_transform_eroded),
                                        dist_transform_eroded.shape,
                                    )
                                    chosen_cx, chosen_cy = (
                                        float(pole_x_eroded),
                                        float(pole_y_eroded),
                                    )
                            else:
                                # Eroded mask is empty, fall back to original mask logic
                                cy_orig_com, cx_orig_com = ndimage.center_of_mass(
                                    single_cell_mask
                                )
                                cy_idx, cx_idx = (
                                    int(round(cy_orig_com)),
                                    int(round(cx_orig_com)),
                                )
                                if (
                                    0 <= cy_idx < single_cell_mask.shape[0]
                                    and 0 <= cx_idx < single_cell_mask.shape[1]
                                    and single_cell_mask[cy_idx, cx_idx]
                                ):
                                    chosen_cx, chosen_cy = cx_orig_com, cy_orig_com
                                else:
                                    dist_transform_orig = (
                                        ndimage.distance_transform_edt(single_cell_mask)
                                    )
                                    pole_y_orig, pole_x_orig = np.unravel_index(
                                        np.argmax(dist_transform_orig),
                                        dist_transform_orig.shape,
                                    )
                                    chosen_cx, chosen_cy = (
                                        float(pole_x_orig),
                                        float(pole_y_orig),
                                    )

                            final_cx_orig, final_cy_orig = chosen_cx, chosen_cy

                            # Clamp coordinates to be within original image bounds with an image boundary margin
                            margin = (
                                constants.IMAGE_BOUNDARY_MARGIN_FOR_NUMBER_PLACEMENT
                            )  # 10
                            # Ensure that the max coordinate for clamping is not less than the min coordinate
                            # If img_w_orig - 1 - margin < margin, it means the image is too small for the margin on both sides.
                            # In such a case, clamp to 'margin' from the top/left.

                            cx_min_clamp = margin
                            cx_max_clamp = max(
                                margin, img_w_orig - 1 - margin
                            )  # Ensure max_clamp is not less than min_clamp for small images

                            cy_min_clamp = margin
                            cy_max_clamp = max(
                                margin, img_h_orig - 1 - margin
                            )  # Ensure max_clamp is not less than min_clamp for small images

                            final_cx_orig = max(
                                cx_min_clamp, min(final_cx_orig, cx_max_clamp)
                            )
                            final_cy_orig = max(
                                cy_min_clamp, min(final_cy_orig, cy_max_clamp)
                            )

                            # Fallback for extremely small images where even the margin might be an issue
                            # (e.g., if img_w_orig < margin). The above max(margin, ...) handles this by preferring 'margin'.
                            # However, ensure it's still within overall 0 to img_dim-1 if somehow margin is > img_dim.
                            final_cx_orig = max(0, min(final_cx_orig, img_w_orig - 1))
                            final_cy_orig = max(0, min(final_cy_orig, img_h_orig - 1))

                            cell_info_for_sorting.append(
                                {
                                    "id": cell_id_val,
                                    "top": top_most,
                                    "left": left_most_in_top_row,
                                    "center_orig_x": final_cx_orig,
                                    "center_orig_y": final_cy_orig,
                                }
                            )

                self._cached_cell_number_positions = sorted(
                    cell_info_for_sorting, key=lambda c: (c["top"], c["left"])
                )
                self._cached_cell_number_ids_tuple_state = current_ids_tuple_for_display
                self._cached_cell_number_show_deselected_state = show_deselected_mode
            # --- End Cell Position Calculation and Caching ---

            # Draw numbers using cached and sorted positions
            sorted_cells_to_draw = self._cached_cell_number_positions
            if (
                not sorted_cells_to_draw
            ):  # Should not happen if ids_to_process_for_display was not empty
                return

            for i, cell_data in enumerate(sorted_cells_to_draw):
                display_number = str(i + 1)
                center_orig_x, center_orig_y = (
                    cell_data["center_orig_x"],
                    cell_data["center_orig_y"],
                )

                # Transform original image center to canvas view coordinates
                zoomed_center_x = center_orig_x * zoom
                zoomed_center_y = center_orig_y * zoom

                # Relative to the cropped portion of the zoomed image
                # src_x1_on_zoomed_img, src_y1_on_zoomed_img are from the main rendering logic for cropping
                rel_zoomed_center_x = zoomed_center_x - (
                    src_x1_on_zoomed_img if crop_box else 0
                )
                rel_zoomed_center_y = zoomed_center_y - (
                    src_y1_on_zoomed_img if crop_box else 0
                )

                # Final position on canvas_view_pil (paste_dst_x_on_canvas, paste_dst_y_on_canvas are also from main rendering)
                cv_text_x = rel_zoomed_center_x + paste_dst_x_on_canvas
                cv_text_y = rel_zoomed_center_y + paste_dst_y_on_canvas

                font_size_on_canvas = max(
                    constants.CELL_NUMBERING_FONT_SIZE_CANVAS_MIN,
                    int(
                        constants.CELL_NUMBERING_FONT_SIZE_ORIG_IMG * zoom
                    ),  # Scale base size by zoom, min 10
                )  # Scale base size by zoom, min 10
                font_size_on_canvas = min(
                    font_size_on_canvas,
                    constants.CELL_NUMBERING_FONT_SIZE_CANVAS_MAX,  # Max 30 to avoid huge numbers
                )  # Max 30 to avoid huge numbers
                try:
                    current_font = ImageFont.truetype(
                        constants.DEFAULT_FONT, size=font_size_on_canvas
                    )
                except IOError:
                    # Fallback if truetype fails (e.g. after first load_default)
                    # For load_default, size is not settable in the same way, so we might get fixed size.
                    current_font = ImageFont.load_default()

                # Get text bounding box to center it better
                try:
                    # For Pillow 10+ textbbox; for older textsize
                    if hasattr(draw_on_canvas_view.textbbox, "__call__"):  # Pillow 10+
                        bbox = draw_on_canvas_view.textbbox(
                            (cv_text_x, cv_text_y),
                            display_number,
                            font=current_font,
                            anchor="lt",
                        )  # Get bbox first
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        # Adjust cv_text_x, cv_text_y to center the text using its bbox
                        final_text_x = cv_text_x - text_width / 2
                        final_text_y = cv_text_y - text_height / 2
                    else:  # Older Pillow, use textsize and manual centering
                        text_width, text_height = draw_on_canvas_view.textsize(
                            display_number, font=current_font
                        )  # Deprecated
                        final_text_x = cv_text_x - text_width / 2
                        final_text_y = cv_text_y - text_height / 2
                except AttributeError:  # Fallback if textsize and textbbox are missing (very old PIL or other issue)
                    text_width, text_height = 10, 10  # Dummy values
                    final_text_x = cv_text_x - text_width / 2
                    final_text_y = cv_text_y - text_height / 2

                # Only draw if the center of the text (cv_text_x, cv_text_y) is within the visible canvas part.
                # ImageDraw will handle clipping the text if it partially goes off-screen.
                if 0 <= cv_text_x < canvas_width and 0 <= cv_text_y < canvas_height:
                    draw_on_canvas_view.text(
                        (final_text_x, final_text_y),
                        display_number,
                        fill=text_color_tuple,
                        font=current_font,
                    )
        # --- END CELL NUMBERING ---

        # Update stats label
        if self.parent_frame.stats_label:
            total_cells_in_mask = 0
            if (
                self.image_view_model.mask_array is not None
                and self.image_view_model.mask_array.size > 0
            ):
                # Exclude 0 (background) from unique cell IDs for total count
                unique_ids_in_mask = np.unique(self.image_view_model.mask_array)
                total_cells_in_mask = len(unique_ids_in_mask[unique_ids_in_mask != 0])

            user_drawn_count = len(self.image_view_model.user_drawn_cell_ids)
            # Ensure user_drawn_count doesn't exceed total_cells_in_mask, though theoretically it shouldn't if logic is correct.
            # This can happen if a user_drawn_cell_id somehow persists after masks are cleared/re-segmented without that ID.
            # The reconciliation in ImageViewModel should handle this, but as a safeguard:
            actual_user_drawn_ids_in_current_mask = (
                self.image_view_model.user_drawn_cell_ids.intersection(
                    set(np.unique(self.image_view_model.mask_array))
                    if self.image_view_model.mask_array is not None
                    else set()
                )
            )
            user_drawn_count = len(actual_user_drawn_ids_in_current_mask)

            model_found_count = total_cells_in_mask - user_drawn_count
            model_found_count = max(0, model_found_count)  # Ensure not negative

            selected_count = len(self.image_view_model.included_cells)

            stats_text = (
                f"Cell count:\n"
                f"  Model Found: {model_found_count}\n"
                f"  User Drawn: {user_drawn_count}\n"
                f"  Total Unique: {total_cells_in_mask}\n"
                f"  Selected: {selected_count}"
            )
            self.parent_frame.stats_label.configure(text=stats_text)

        if base_display_img.width > 0 and base_display_img.height > 0:
            self.tk_image_on_canvas = ImageTk.PhotoImage(base_display_img)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(
                0, 0, anchor="nw", image=self.tk_image_on_canvas
            )
        else:
            self.image_canvas.delete("all")
            self.tk_image_on_canvas = None
