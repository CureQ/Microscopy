import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from . import constants
from .application_model import ApplicationModel


class ImageOverlayProcessor:
    @staticmethod
    def dilate_boundary(bool_mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        """Simple dilation for a boolean mask using NumPy."""
        if not bool_mask.any() or iterations < 1:
            return bool_mask

        dilated_mask = bool_mask.copy()
        # Use 4-connectivity for simple dilation similar to the original method's effect
        struct = ndimage.generate_binary_structure(bool_mask.ndim, 1)
        dilated_mask = ndimage.binary_dilation(
            dilated_mask, structure=struct, iterations=iterations
        )
        return dilated_mask

    def __init__(self, application_model: ApplicationModel):
        self.application_model = application_model

    def _calculate_and_sort_cell_number_info(
        self, cell_ids_to_draw: set[int]
    ) -> list[dict]:
        """
        Calculates positions for cell numbers, sorted for consistent numbering.
        Logic adapted from ImageViewRenderer and FileIOController.
        Returns a list of dicts, each containing 'id', 'top', 'left',
        'center_orig_x', 'center_orig_y' for each cell_id in cell_ids_to_draw.
        """
        if (
            self.application_model.image_data.mask_array is None
            or not cell_ids_to_draw
            or self.application_model.image_data.original_image is None
        ):
            return []

        cell_info_for_sorting = []
        img_h_orig, img_w_orig = (
            self.application_model.image_data.original_image.height,
            self.application_model.image_data.original_image.width,
        )
        current_mask_array = self.application_model.image_data.mask_array

        for cell_id_val in cell_ids_to_draw:
            if cell_id_val == 0:
                continue

            single_cell_mask = current_mask_array == cell_id_val
            if not np.any(single_cell_mask):
                continue

            rows, cols = np.where(single_cell_mask)
            top_most = np.min(rows)
            left_most_in_top_row = np.min(cols[rows == top_most])

            cell_boundary_margin = constants.CELL_CENTER_FIND_MARGIN
            eroded_mask = ndimage.binary_erosion(
                single_cell_mask, iterations=cell_boundary_margin, border_value=0
            )

            chosen_cx, chosen_cy = -1.0, -1.0

            if np.any(eroded_mask):
                cy_eroded_com, cx_eroded_com = ndimage.center_of_mass(eroded_mask)
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
                    dist_transform_eroded = ndimage.distance_transform_edt(eroded_mask)
                    if np.any(dist_transform_eroded):
                        pole_y_eroded, pole_x_eroded = np.unravel_index(
                            np.argmax(dist_transform_eroded),
                            dist_transform_eroded.shape,
                        )
                        chosen_cx, chosen_cy = (
                            float(pole_x_eroded),
                            float(pole_y_eroded),
                        )

            if chosen_cx == -1.0:  # Fallback if eroded mask didn't yield a point
                cy_orig_com, cx_orig_com = ndimage.center_of_mass(single_cell_mask)
                cy_idx, cx_idx = int(round(cy_orig_com)), int(round(cx_orig_com))
                if (
                    0 <= cy_idx < single_cell_mask.shape[0]
                    and 0 <= cx_idx < single_cell_mask.shape[1]
                    and single_cell_mask[cy_idx, cx_idx]
                ):
                    chosen_cx, chosen_cy = cx_orig_com, cy_orig_com
                else:
                    dist_transform_orig = ndimage.distance_transform_edt(
                        single_cell_mask
                    )
                    if np.any(dist_transform_orig):
                        pole_y_orig, pole_x_orig = np.unravel_index(
                            np.argmax(dist_transform_orig), dist_transform_orig.shape
                        )
                        chosen_cx, chosen_cy = float(pole_x_orig), float(pole_y_orig)
                    else:
                        chosen_cx, chosen_cy = (
                            cx_orig_com,
                            cy_orig_com,
                        )  # Default to COM if all else fails

            margin = constants.IMAGE_BOUNDARY_MARGIN_FOR_NUMBER_PLACEMENT
            cx_min_clamp = margin
            cx_max_clamp = max(margin, img_w_orig - 1 - margin)
            cy_min_clamp = margin
            cy_max_clamp = max(margin, img_h_orig - 1 - margin)

            final_cx_orig = max(cx_min_clamp, min(chosen_cx, cx_max_clamp))
            final_cy_orig = max(cy_min_clamp, min(chosen_cy, cy_max_clamp))

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

        return sorted(cell_info_for_sorting, key=lambda c: (c["top"], c["left"]))

    def draw_masks_on_pil(
        self, base_pil_image: Image.Image, cell_ids_to_draw: set[int]
    ) -> Image.Image:
        """
        Generates a PIL Image representing only the colored mask layer for the specified cell IDs.
        The returned image will have RGB colors for masks on a black background.
        The base_pil_image is used ONLY for determining the output image size.
        """
        if (
            self.application_model.image_data.mask_array is None
            or not cell_ids_to_draw
            or base_pil_image is None  # Used for size context
        ):
            # Return a black image of the base_pil_image's size if no masks or context
            return Image.new(
                "RGB",
                base_pil_image.size if base_pil_image else (1, 1),
                constants.COLOR_BLACK_STR,
            )

        # Create an empty RGB image for the mask layer, matching base_pil_image size
        mask_layer_pil = Image.new(
            "RGB", base_pil_image.size, constants.COLOR_BLACK_STR
        )

        current_mask_array = self.application_model.image_data.mask_array

        rng = np.random.default_rng(seed=constants.RANDOM_SEED_MASKS)
        all_unique_mask_ids = np.unique(current_mask_array)
        color_map = {
            uid: tuple(rng.integers(50, 200, size=3))
            for uid in all_unique_mask_ids
            if uid != 0
        }

        temp_mask_np = np.zeros((*current_mask_array.shape, 3), dtype=np.uint8)
        any_mask_drawn = False
        for cell_id_val in cell_ids_to_draw:
            if cell_id_val != 0 and cell_id_val in color_map:
                mask_pixels = current_mask_array == cell_id_val
                if np.any(mask_pixels):
                    temp_mask_np[mask_pixels] = color_map[cell_id_val]
                    any_mask_drawn = True

        if not any_mask_drawn:
            # Return a black image of the correct size if no relevant masks were drawn
            return Image.new("RGB", base_pil_image.size, constants.COLOR_BLACK_STR)

        mask_layer_pil = Image.fromarray(temp_mask_np, "RGB")

        # Return only the mask layer, not blended
        return mask_layer_pil

    def blend_image_with_mask_layer(
        self, base_image: Image.Image, mask_layer: Image.Image, alpha: float
    ) -> Image.Image:
        """
        Blends a base image with a mask layer.
        Assumes mask_layer is an RGB image (colors on black) of the same size as base_image.
        """
        if base_image is None or mask_layer is None:
            return base_image

        if base_image.size != mask_layer.size:
            print(
                f"ImageOverlayProcessor Warning: Size mismatch in blend_image_with_mask_layer. Base: {base_image.size}, Mask Layer: {mask_layer.size}"
            )
            # Attempt to resize mask_layer to base_image size as a fallback, though this indicates an issue.
            # This shouldn't happen if draw_masks_on_pil used base_image for size context.
            mask_layer = mask_layer.resize(base_image.size, Image.NEAREST)

        base_image_rgb = base_image
        if base_image.mode != "RGB":
            base_image_rgb = base_image.convert("RGB")

        # Ensure mask_layer is also RGB
        mask_layer_rgb = mask_layer
        if mask_layer.mode != "RGB":
            mask_layer_rgb = mask_layer.convert("RGB")

        return Image.blend(base_image_rgb, mask_layer_rgb, alpha=alpha)

    def draw_boundaries_on_pil(
        self, base_pil_image: Image.Image, cell_ids_to_draw: set[int]
    ) -> Image.Image:
        if (
            self.application_model.image_data.mask_array is None
            or not cell_ids_to_draw
            or base_pil_image is None
        ):
            return base_pil_image

        current_mask_array = self.application_model.image_data.mask_array
        exact_boundaries = self.application_model.image_data.exact_boundaries

        if exact_boundaries is None or not exact_boundaries.any():
            return base_pil_image

        boundary_to_draw_on_image = np.zeros_like(exact_boundaries, dtype=bool)
        any_boundary_drawn = False

        for cid in cell_ids_to_draw:
            if cid != 0:
                cell_mask_region = current_mask_array == cid
                if np.any(cell_mask_region):
                    current_cell_boundary = exact_boundaries & cell_mask_region
                    if np.any(current_cell_boundary):
                        boundary_to_draw_on_image |= current_cell_boundary
                        any_boundary_drawn = True

        if not any_boundary_drawn:
            return base_pil_image

        dilation_iters = constants.DILATION_ITERATIONS_FOR_BOUNDARY_DISPLAY
        if dilation_iters > 0:
            dilated_boundaries = self.dilate_boundary(
                boundary_to_draw_on_image, iterations=dilation_iters
            )
        else:
            dilated_boundaries = boundary_to_draw_on_image

        if not np.any(dilated_boundaries):
            return base_pil_image

        output_image_np = np.array(base_pil_image.convert("RGB"))
        boundary_color_map_pil = constants.BOUNDARY_COLOR_MAP_PIL
        gui_boundary_color_name = (
            self.application_model.display_state.boundary_color_name
        )
        chosen_color_np = np.array(
            boundary_color_map_pil.get(
                gui_boundary_color_name, constants.BOUNDARY_COLOR_MAP_PIL["Green"]
            )
        )

        if dilated_boundaries.shape == output_image_np.shape[:2]:
            output_image_np[dilated_boundaries] = chosen_color_np
        else:
            print(
                f"ImageOverlayProcessor Warning: Shape mismatch for boundary drawing. Base: {output_image_np.shape[:2]}, Boundary: {dilated_boundaries.shape}"
            )
            return base_pil_image

        return Image.fromarray(output_image_np)

    def draw_numbers_on_pil(
        self, base_pil_image: Image.Image, cell_ids_to_draw: set[int], font_size: int
    ) -> Image.Image:
        if (
            self.application_model.image_data.mask_array is None
            or not cell_ids_to_draw
            or base_pil_image is None
            or self.application_model.image_data.original_image is None
        ):
            return base_pil_image

        image_to_draw_on = base_pil_image.copy()
        draw_on_pil = ImageDraw.Draw(image_to_draw_on)
        current_boundary_color_name = (
            self.application_model.display_state.boundary_color_name
        )
        text_color_map = constants.PDF_TEXT_NUMBER_COLOR_MAP
        text_color_tuple = text_color_map.get(
            current_boundary_color_name, constants.PDF_TEXT_NUMBER_COLOR_MAP["Green"]
        )

        try:
            font = ImageFont.truetype(constants.DEFAULT_FONT, size=font_size)
        except IOError:
            font = ImageFont.load_default(size=font_size)

        sorted_cell_info = self._calculate_and_sort_cell_number_info(cell_ids_to_draw)

        if not sorted_cell_info:
            return image_to_draw_on

        num_drawn_actually = 0
        for i, cell_data in enumerate(sorted_cell_info):
            display_number = str(i + 1)
            center_x_pil_orig = cell_data["center_orig_x"]
            center_y_pil_orig = cell_data["center_orig_y"]

            try:
                if hasattr(draw_on_pil.textbbox, "__call__"):
                    bbox = draw_on_pil.textbbox(
                        (center_x_pil_orig, center_y_pil_orig),
                        display_number,
                        font=font,
                        anchor="lt",
                    )
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    final_text_x = center_x_pil_orig - text_width / 2
                    final_text_y = center_y_pil_orig - text_height / 2
                else:
                    text_width, text_height = draw_on_pil.textsize(
                        display_number, font=font
                    )
                    final_text_x = center_x_pil_orig - text_width / 2
                    final_text_y = center_y_pil_orig - text_height / 2
            except AttributeError:
                text_width, text_height = 10, 10
                final_text_x = center_x_pil_orig - text_width / 2
                final_text_y = center_y_pil_orig - text_height / 2

            draw_on_pil.text(
                (final_text_x, final_text_y),
                display_number,
                fill=text_color_tuple,
                font=font,
            )
            num_drawn_actually += 1

        if num_drawn_actually == 0:
            # This implies sorted_cell_info was not empty but somehow no numbers were drawn
            # Return the original base_pil_image, not the copy, if nothing changed.
            return base_pil_image

        return image_to_draw_on
