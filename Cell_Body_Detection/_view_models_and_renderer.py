import customtkinter as ctk
import numpy as np
from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageTk,
)

from . import constants
from .application_model import ApplicationModel
from .image_overlay_processor import ImageOverlayProcessor


class ImageViewRenderer:
    def __init__(
        self,
        canvas_ref: ctk.CTkCanvas,
        application_model_ref: ApplicationModel,
        cell_body_frame_ref: ctk.CTkFrame,  # To access some UI elements like boundary_color or show_original
    ):
        self.image_canvas = canvas_ref
        self.application_model = application_model_ref
        self.parent_frame = cell_body_frame_ref

        self.overlay_processor = ImageOverlayProcessor(application_model_ref)

        self.tk_image_on_canvas = None
        self._update_display_retry_id = None

        self._cached_full_res_mask_rgb = None
        self._cached_mask_ids_tuple_state = None
        self._cached_show_deselected_mask_state = None

        self._cached_full_res_boundary_pil = None
        self._cached_boundary_ids_tuple_state = None
        self._cached_show_deselected_boundary_state = None
        self._cached_boundary_color_state = None

        self._cached_cell_number_positions = None
        self._cached_cell_number_ids_tuple_state = None
        self._cached_cell_number_show_deselected_state = None

        self.draw_feedback_color = constants.DRAW_FEEDBACK_COLOR
        self.draw_first_point_color = constants.DRAW_FIRST_POINT_COLOR
        self.draw_last_point_color = constants.DRAW_LAST_POINT_COLOR
        self.draw_point_radius = constants.DRAW_POINT_RADIUS

        # Subscribe to model changes
        self.application_model.subscribe(self.handle_model_update)

    def handle_model_update(self, change_type: str | None = None):
        """
        Called by the ApplicationModel when its state changes.
        Determines if a re-render is necessary.
        """
        print(
            f"ImageViewRenderer.handle_model_update received change_type: {change_type}"
        )

        if change_type in [
            "image_loaded",
            "segmentation_updated",
            "cell_selection_changed",
            "mask_updated_user_drawn",
            "pan_zoom_updated",
            "pan_zoom_reset",
            "display_settings_changed",
            "display_settings_reset",
            "view_options_changed",
            "model_restored_undo",
            "model_restored_redo",
            "history_updated",
        ]:
            if change_type == "pan_zoom_updated":
                print(
                    f"ImageViewRenderer: '{change_type}' detected, rendering with interactive quality."
                )
                self.render(quality="interactive")
            else:
                print(
                    f"ImageViewRenderer: '{change_type}' detected, rendering with final quality."
                )
                self.render(quality="final")

        # Specific cache invalidations based on change_type
        if change_type in [
            "segmentation_updated",
            "cell_selection_changed",
            "mask_updated_user_drawn",
            "model_restored_undo",
            "model_restored_redo",
        ]:
            print(
                f"ImageViewRenderer: '{change_type}' detected, invalidating all caches."
            )
            self.invalidate_caches()  # Selection or mask structure changed

        if change_type in [
            "view_options_changed",
            "display_settings_changed",
            "display_settings_reset",
        ]:
            print(
                f"ImageViewRenderer: '{change_type}' detected, checking and invalidating specific caches."
            )
            # If boundary color changes, boundary cache is invalid
            if (
                self._cached_boundary_color_state
                != self.application_model.display_state.boundary_color_name
            ):
                print(
                    "ImageViewRenderer: Boundary color changed, invalidating boundary cache."
                )
                self._cached_full_res_boundary_pil = None
            # If show_deselected changes, all caches that depend on it are invalid
            if (
                self._cached_show_deselected_mask_state
                != self.application_model.display_state.show_deselected_masks_only
            ):
                print(
                    "ImageViewRenderer: 'Show deselected masks only' changed, invalidating relevant caches."
                )
                self._cached_full_res_mask_rgb = None
                self._cached_full_res_boundary_pil = None
                self._cached_cell_number_positions = None

    def invalidate_caches(self):
        print("ImageViewRenderer: Invalidating all rendering caches.")
        self._cached_full_res_mask_rgb = None
        self._cached_mask_ids_tuple_state = None
        self._cached_show_deselected_mask_state = None

        self._cached_full_res_boundary_pil = None
        self._cached_boundary_ids_tuple_state = None
        self._cached_show_deselected_boundary_state = None
        self._cached_boundary_color_state = None

        self._cached_cell_number_positions = None
        self._cached_cell_number_ids_tuple_state = None
        self._cached_cell_number_show_deselected_state = None

    def render(self, quality="final"):
        print(f"ImageViewRenderer.render called with quality: {quality}")
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            print(
                f"ImageViewRenderer: Canvas not ready (width={canvas_width}, height={canvas_height}). Scheduling retry for render."
            )
            if self._update_display_retry_id:
                self.image_canvas.after_cancel(self._update_display_retry_id)
            self._update_display_retry_id = self.image_canvas.after(
                constants.UI_RENDER_RETRY_DELAY_MS,
                lambda q=quality: self.render(quality=q),
            )
            return

        # Fetch the (potentially processed) image from ApplicationModel
        pil_image_to_render = self.application_model.get_processed_image_for_display()

        if pil_image_to_render is None:
            self.image_canvas.delete("all")
            print(
                "ImageViewRenderer: No image to render. Displaying 'Select Image' prompt."
            )
            text_color = "white"  # Default
            # Accessing theme for text color if main app uses CTk Theming
            # This assumes parent_frame is the main app or has access to theme manager
            if (
                hasattr(self.parent_frame, "_apply_appearance_mode")
                and hasattr(ctk, "ThemeManager")
                and ctk.ThemeManager.theme
            ):
                try:
                    text_color = self.parent_frame._apply_appearance_mode(
                        ctk.ThemeManager.theme["CTkLabel"]["text_color"]
                    )
                except (KeyError, TypeError):
                    pass
            self.image_canvas.create_text(
                canvas_width / 2,
                canvas_height / 2,
                text=constants.MSG_SELECT_IMAGE_PROMPT,
                fill=text_color,
                font=ctk.CTkFont(size=16),
            )
            self.tk_image_on_canvas = None
            # Update stats label
            if hasattr(self.parent_frame, "_update_stats_label"):
                self.parent_frame._update_stats_label()
            elif (
                hasattr(self.parent_frame, "stats_label")
                and self.parent_frame.stats_label
            ):  # Direct access fallback
                self.parent_frame.stats_label.configure(
                    text=constants.UI_TEXT_STATS_LABEL_DEFAULT
                )
            return

        zoom, pan_x, pan_y = self.application_model.pan_zoom_state.get_params()
        print(f"ImageViewRenderer: Rendering with zoom={zoom}, pan=({pan_x},{pan_y})")

        new_width = int(pil_image_to_render.width * zoom)
        new_height = int(pil_image_to_render.height * zoom)

        if new_width <= 0 or new_height <= 0:
            print(
                f"ImageViewRenderer: Calculated new_width ({new_width}) or new_height ({new_height}) is <= 0. Clearing canvas."
            )
            self.image_canvas.delete("all")
            self.tk_image_on_canvas = None
            return

        resample_filter = Image.LANCZOS if quality == "final" else Image.NEAREST
        print(
            f"ImageViewRenderer: Resampling original image with filter: {'LANCZOS' if quality == 'final' else 'NEAREST'}."
        )
        # Ensure new_width and new_height are at least 1 for resize
        safe_new_width = max(1, new_width)
        safe_new_height = max(1, new_height)
        zoomed_image = pil_image_to_render.resize(
            (safe_new_width, safe_new_height), resample_filter
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
        # Use show_original_image from ApplicationModel's display_state
        if not self.application_model.display_state.show_original_image:
            base_display_img = Image.new(
                "RGB", (canvas_width, canvas_height), constants.COLOR_BLACK_STR
            )

        current_mask_array = (
            self.application_model.image_data.mask_array
        )  # Use ApplicationModel
        all_mask_ids = set()
        if current_mask_array is not None and current_mask_array.size > 0:
            unique_ids = np.unique(current_mask_array)
            all_mask_ids = set(unique_ids[unique_ids != 0])

        ids_to_process_for_display = set()
        # Use show_deselected_masks_only from ApplicationModel's display_state
        show_deselected_mode = (
            self.application_model.display_state.show_deselected_masks_only
        )

        if show_deselected_mode:
            deselected_ids = (
                all_mask_ids - self.application_model.image_data.included_cells
            )  # Use ApplicationModel
            ids_to_process_for_display = deselected_ids
        else:
            ids_to_process_for_display = (
                self.application_model.image_data.included_cells.intersection(
                    all_mask_ids
                )  # Use ApplicationModel
            )

        current_ids_tuple_for_display = tuple(sorted(list(ids_to_process_for_display)))

        if current_mask_array is not None and current_mask_array.size > 0:
            # MASK OVERLAY
            # Use show_cell_masks from ApplicationModel's display_state
            if self.application_model.display_state.show_cell_masks:
                print("ImageViewRenderer: Drawing cell masks.")
                if (
                    self._cached_full_res_mask_rgb is None
                    or self._cached_mask_ids_tuple_state
                    != current_ids_tuple_for_display
                    or self._cached_show_deselected_mask_state
                    != self.application_model.display_state.show_deselected_masks_only
                ):
                    print(
                        "ImageViewRenderer: Mask cache miss or invalid. Regenerating full-res mask layer via OverlayProcessor."
                    )
                    self._cached_full_res_mask_rgb = (
                        self.overlay_processor.draw_masks_on_pil(
                            base_pil_image=pil_image_to_render,  # Pass for size context
                            cell_ids_to_draw=ids_to_process_for_display,
                        )
                    )
                    # _cached_full_res_mask_rgb now stores the raw RGB mask layer.

                    self._cached_mask_ids_tuple_state = current_ids_tuple_for_display
                    self._cached_show_deselected_mask_state = (
                        self.application_model.display_state.show_deselected_masks_only
                    )

                # Now, _cached_full_res_mask_rgb holds the MASK LAYER (colors on black)
                overlay_resample_filter = Image.NEAREST
                zoomed_mask_layer_rgb = self._cached_full_res_mask_rgb.resize(
                    (safe_new_width, safe_new_height),
                    resample=overlay_resample_filter,
                )

                if crop_box and width_to_copy > 0 and height_to_copy > 0:
                    cropped_mask_layer = zoomed_mask_layer_rgb.crop(crop_box)
                    canvas_mask_layer = Image.new(
                        "RGB", (canvas_width, canvas_height), (0, 0, 0)
                    )
                    canvas_mask_layer.paste(
                        cropped_mask_layer,
                        (paste_dst_x_on_canvas, paste_dst_y_on_canvas),
                    )

                    if self.application_model.display_state.show_original_image:
                        # base_display_img currently holds the (processed) original image
                        base_display_img = Image.blend(
                            base_display_img.convert("RGB"),
                            canvas_mask_layer,
                            alpha=constants.MASK_BLEND_ALPHA,
                        )
                    else:
                        # base_display_img is already black if show_original_image is false.
                        # So, we just display the mask layer (which is colors on black).
                        base_display_img = canvas_mask_layer

            # BOUNDARY OVERLAY
            # Use show_cell_boundaries from ApplicationModel's display_state
            if self.application_model.display_state.show_cell_boundaries:
                # Use boundary_color_name from ApplicationModel's display_state
                print("ImageViewRenderer: Drawing cell boundaries.")
                current_boundary_color_name = (
                    self.application_model.display_state.boundary_color_name
                )

                # Get pre-calculated exact boundaries from ApplicationModel
                exact_boundaries = self.application_model.image_data.exact_boundaries

                if (
                    exact_boundaries
                    is not None  # Check if exact_boundaries are available
                    and (
                        self._cached_full_res_boundary_pil is None
                        or self._cached_boundary_ids_tuple_state
                        != current_ids_tuple_for_display
                        # Use show_deselected_masks_only from ApplicationModel for cache state
                        or self._cached_show_deselected_boundary_state
                        != self.application_model.display_state.show_deselected_masks_only
                        # Use boundary_color_name from ApplicationModel for cache state
                        or self._cached_boundary_color_state
                        != self.application_model.display_state.boundary_color_name
                    )
                ):
                    print(
                        "ImageViewRenderer: Boundary cache miss or invalid. Regenerating full-res boundary PIL via OverlayProcessor."
                    )
                    # Create a black base image of the correct full resolution size
                    black_base_for_boundary_extraction = Image.new(
                        "RGB", pil_image_to_render.size, (0, 0, 0)
                    )

                    # Ask the processor to draw boundaries on this black base
                    image_with_boundaries_drawn = self.overlay_processor.draw_boundaries_on_pil(
                        base_pil_image=black_base_for_boundary_extraction,  # Use the black base
                        cell_ids_to_draw=ids_to_process_for_display,
                    )

                    # To get a monochrome boundary mask (L mode):
                    # Convert the result to grayscale. Boundary pixels (colored) will be non-black.
                    # Non-boundary pixels (originally black) will remain black.
                    # Then convert to boolean where non-black is True (boundary).
                    gray_image_with_boundaries = image_with_boundaries_drawn.convert(
                        "L"
                    )
                    boundary_mask_np = (
                        np.array(gray_image_with_boundaries) > 0
                    )  # Threshold at 0 for any color

                    if np.any(boundary_mask_np):
                        self._cached_full_res_boundary_pil = Image.fromarray(
                            boundary_mask_np.astype(np.uint8) * 255, mode="L"
                        )
                    else:
                        self._cached_full_res_boundary_pil = Image.new(
                            "L", pil_image_to_render.size, 0
                        )

                    self._cached_boundary_ids_tuple_state = (
                        current_ids_tuple_for_display
                    )
                    self._cached_show_deselected_boundary_state = (
                        self.application_model.display_state.show_deselected_masks_only
                    )
                    self._cached_boundary_color_state = (
                        self.application_model.display_state.boundary_color_name
                    )  # Update cache

                # Check if _cached_full_res_boundary_pil is not None before proceeding to use it
                if self._cached_full_res_boundary_pil is not None:
                    print(
                        "ImageViewRenderer: Boundary cache hit. Drawing boundary overlay."
                    )
                    overlay_resample_filter = Image.NEAREST
                    zoomed_boundary_pil = self._cached_full_res_boundary_pil.resize(
                        (safe_new_width, safe_new_height),  # Use safe dimensions
                        resample=overlay_resample_filter,
                    )
                    if crop_box and width_to_copy > 0 and height_to_copy > 0:
                        cropped_boundary = zoomed_boundary_pil.crop(crop_box)
                        boundary_color_map_pil = constants.BOUNDARY_COLOR_MAP_PIL
                        # Use boundary_color_name from ApplicationModel
                        chosen_color = boundary_color_map_pil.get(
                            self.application_model.display_state.boundary_color_name,
                            constants.BOUNDARY_COLOR_MAP_PIL["Green"],
                        )
                        final_boundary_on_canvas_bool = Image.new(
                            "L", (canvas_width, canvas_height), 0
                        )
                        final_boundary_on_canvas_bool.paste(
                            cropped_boundary,
                            (paste_dst_x_on_canvas, paste_dst_y_on_canvas),
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
            if base_display_img.mode != "RGB":
                base_display_img = base_display_img.convert("RGB")
            print(
                "ImageViewRenderer: Drawing polygon feedback for active drawing mode."
            )

            draw_on_canvas_view = ImageDraw.Draw(base_display_img)

            # Transform raw user-clicked points to canvas space for dot rendering
            # and for drawing lines connecting them directly.
            all_potential_canvas_points = []  # Stores all points transformed to canvas space
            visible_canvas_dots = []  # Store only points visible on canvas (list of dicts)

            for index, (orig_x, orig_y) in enumerate(
                self.parent_frame.drawing_controller.current_draw_points
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
                )
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
                len(self.parent_frame.drawing_controller.current_draw_points) >= 2
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
            # Use show_cell_numbers from ApplicationModel's display_state
            self.application_model.display_state.show_cell_numbers
            and current_mask_array is not None
            and current_mask_array.size > 0
            and ids_to_process_for_display
        ):
            print("ImageViewRenderer: Drawing cell numbers.")
            if base_display_img.mode != "RGB":
                base_display_img = base_display_img.convert("RGB")
            draw_on_canvas_view = ImageDraw.Draw(base_display_img)

            current_boundary_color_name = (
                self.application_model.display_state.boundary_color_name
            )
            # For text color on canvas, use BOUNDARY_COLOR_MAP_PIL (same as boundaries)
            text_color_map_canvas = constants.BOUNDARY_COLOR_MAP_PIL
            text_color_tuple = text_color_map_canvas.get(
                current_boundary_color_name,
                constants.BOUNDARY_COLOR_MAP_PIL["Green"],
            )

            try:
                font = ImageFont.truetype(
                    constants.DEFAULT_FONT, size=constants.DEFAULT_FALLBACK_FONT_SIZE
                )
            except IOError:
                font = ImageFont.load_default()

            # --- Cell Position Calculation and Caching (using OverlayProcessor method) ---
            if (
                self._cached_cell_number_positions is None
                or self._cached_cell_number_ids_tuple_state
                != current_ids_tuple_for_display
                or self._cached_cell_number_show_deselected_state
                != self.application_model.display_state.show_deselected_masks_only
            ):
                print(
                    "ImageViewRenderer: Cell number positions cache miss or invalid. Recalculating via OverlayProcessor."
                )
                self._cached_cell_number_positions = (
                    self.overlay_processor._calculate_and_sort_cell_number_info(
                        ids_to_process_for_display
                    )
                )
                self._cached_cell_number_ids_tuple_state = current_ids_tuple_for_display
                self._cached_cell_number_show_deselected_state = (
                    self.application_model.display_state.show_deselected_masks_only
                )
            else:
                print("ImageViewRenderer: Cell number positions cache hit.")
            # --- End Cell Position Calculation and Caching ---

            sorted_cells_to_draw = self._cached_cell_number_positions
            if not sorted_cells_to_draw:
                # This means no cells to draw numbers for, skip the drawing loop
                pass  # base_display_img remains unchanged in this block
            else:
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
                        int(constants.CELL_NUMBERING_FONT_SIZE_ORIG_IMG * zoom),
                    )  # Scale base size by zoom, min 10
                    font_size_on_canvas = min(
                        font_size_on_canvas,
                        constants.CELL_NUMBERING_FONT_SIZE_CANVAS_MAX,
                    )  # Max 30 to avoid huge numbers
                    try:
                        current_font = ImageFont.truetype(
                            constants.DEFAULT_FONT, size=font_size_on_canvas
                        )
                    except IOError:
                        # Fallback if truetype fails (e.g. after first load_default)
                        current_font = ImageFont.load_default(size=font_size_on_canvas)

                    # Get text bounding box to center it better
                    try:
                        # For Pillow 10+ textbbox; for older textsize
                        if hasattr(
                            draw_on_canvas_view.textbbox, "__call__"
                        ):  # Pillow 10+
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
        if hasattr(
            self.parent_frame, "_update_stats_label"
        ):  # Check if parent_frame has the method
            self.parent_frame._update_stats_label()
        elif (
            hasattr(self.parent_frame, "stats_label") and self.parent_frame.stats_label
        ):  # Fallback for direct access
            # This block is essentially the same as above, could be refactored
            # For now, keeping it to ensure stats_label is updated if _update_stats_label is not present
            # This part calculates stats and directly configures the label, which is fine for a fallback.
            total_cells_in_mask = 0
            if (
                self.application_model.image_data.mask_array is not None
                and self.application_model.image_data.mask_array.size > 0
            ):
                unique_ids_in_mask = np.unique(
                    self.application_model.image_data.mask_array
                )
                total_cells_in_mask = len(unique_ids_in_mask[unique_ids_in_mask != 0])

            actual_user_drawn_ids_in_current_mask = (
                self.application_model.image_data.user_drawn_cell_ids.intersection(
                    set(np.unique(self.application_model.image_data.mask_array))
                    if self.application_model.image_data.mask_array is not None
                    else set()
                )
            )
            user_drawn_count = len(actual_user_drawn_ids_in_current_mask)

            model_found_count = total_cells_in_mask - user_drawn_count
            model_found_count = max(0, model_found_count)

            selected_count = len(self.application_model.image_data.included_cells)

            stats_text = (
                f"Cell count:\n"
                f"  Model Found: {model_found_count}\n"
                f"  User Drawn: {user_drawn_count}\n"
                f"  Total Unique: {total_cells_in_mask}\n"
                f"  Selected: {selected_count}"
            )
            self.parent_frame.stats_label.configure(text=stats_text)

        # Ensure base_display_img has valid dimensions before creating PhotoImage
        if base_display_img.width > 0 and base_display_img.height > 0:
            self.tk_image_on_canvas = ImageTk.PhotoImage(base_display_img)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(
                0, 0, anchor="nw", image=self.tk_image_on_canvas
            )
            print("ImageViewRenderer: Successfully updated canvas with new image.")
        else:
            # This case might happen if new_width/new_height was 0 and pil_image_to_render was not None
            # Or if base_display_img itself became 0x0 for some reason.
            print(
                f"ImageViewRenderer: base_display_img has invalid dimensions ({base_display_img.width}x{base_display_img.height}). Clearing canvas."
            )
            self.image_canvas.delete("all")
            self.tk_image_on_canvas = None
