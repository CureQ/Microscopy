import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .. import constants
from ..model.application_model import ApplicationModel
from ..utils.debug_logger import log


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
        log(
            f"ImageOverlayProcessor initialized with ApplicationModel: {application_model}",
            level="INFO",
        )

        # Cache fields for masks
        self._cached_full_res_mask_rgb = None
        self._cached_mask_ids_tuple_state = None
        self._cached_show_deselected_mask_state = None

        # Cache fields for boundaries (L mode)
        self._cached_full_res_boundary_L_pil = None
        self._cached_boundary_ids_tuple_state = None
        self._cached_boundary_show_deselected_state = None
        self._cached_boundary_image_size_state = None
        # self._cached_boundary_color_state = None # Color state is not needed for L-mode PIL cache

        # Cache fields for cell number positions
        self._cached_cell_number_positions: list[dict] | None = None
        self._cached_cell_number_ids_tuple_state: tuple | None = None
        self._cached_cell_number_show_deselected_state: bool | None = None

    def invalidate_mask_cache(self):
        """Invalidates the cached mask layer and its state."""
        log("ImageOverlayProcessor: Invalidating mask cache.", level="DEBUG")
        self._cached_full_res_mask_rgb = None
        self._cached_mask_ids_tuple_state = None
        self._cached_show_deselected_mask_state = None

    def invalidate_boundary_cache(self):
        """Invalidates the cached L-mode boundary PIL image and its state."""
        log("ImageOverlayProcessor: Invalidating boundary L PIL cache.", level="DEBUG")
        self._cached_full_res_boundary_L_pil = None
        self._cached_boundary_ids_tuple_state = None
        self._cached_boundary_show_deselected_state = None
        self._cached_boundary_image_size_state = None

    def invalidate_number_positions_cache(self):
        """Invalidates the cached cell number positions and their state."""
        log(
            "ImageOverlayProcessor: Invalidating cell number positions cache.",
            level="DEBUG",
        )
        self._cached_cell_number_positions = None
        self._cached_cell_number_ids_tuple_state = None
        self._cached_cell_number_show_deselected_state = None

    def is_mask_cache_current(
        self,
        base_pil_image_size: tuple[int, int],
        cell_ids_to_draw: set[int],
        show_deselected_masks_only: bool,
    ) -> bool:
        log(
            f"is_mask_cache_current: Checking. Current state - RGB cache: {'Exists' if self._cached_full_res_mask_rgb else 'None'}, IDs: {self._cached_mask_ids_tuple_state}, ShowDeselected: {self._cached_show_deselected_mask_state}, Size: {self._cached_full_res_mask_rgb.size if self._cached_full_res_mask_rgb else 'N/A'}",
            level="DEBUG",
        )
        log(
            f"is_mask_cache_current: Args - base_size: {base_pil_image_size}, cell_ids_count: {len(cell_ids_to_draw)}, show_deselected: {show_deselected_masks_only}",
            level="DEBUG",
        )
        if self._cached_full_res_mask_rgb is None:
            log(
                "is_mask_cache_current: Cache miss - _cached_full_res_mask_rgb is None.",
                level="DEBUG",
            )
            return False
        cell_ids_tuple = tuple(sorted(list(cell_ids_to_draw)))
        if self._cached_full_res_mask_rgb.size != base_pil_image_size:
            log(
                f"is_mask_cache_current: Cache miss - Size mismatch. Cached: {self._cached_full_res_mask_rgb.size}, Requested: {base_pil_image_size}",
                level="DEBUG",
            )
            return False
        if self._cached_mask_ids_tuple_state != cell_ids_tuple:
            log(
                f"is_mask_cache_current: Cache miss - IDs mismatch. Cached: {self._cached_mask_ids_tuple_state}, Requested: {cell_ids_tuple}",
                level="DEBUG",
            )
            return False
        if self._cached_show_deselected_mask_state != show_deselected_masks_only:
            log(
                f"is_mask_cache_current: Cache miss - Show deselected state mismatch. Cached: {self._cached_show_deselected_mask_state}, Requested: {show_deselected_masks_only}",
                level="DEBUG",
            )
            return False
        log("ImageOverlayProcessor: Mask cache is current.")
        return True

    def is_boundary_cache_current(
        self,
        image_size: tuple[int, int],
        cell_ids_to_draw: set[int],
        show_deselected_state: bool,
    ) -> bool:
        log(
            f"is_boundary_cache_current: Checking. Current state - L PIL cache: {'Exists' if self._cached_full_res_boundary_L_pil else 'None'}, IDs: {self._cached_boundary_ids_tuple_state}, ShowDeselected: {self._cached_boundary_show_deselected_state}, SizeState: {self._cached_boundary_image_size_state}, PILSize: {self._cached_full_res_boundary_L_pil.size if self._cached_full_res_boundary_L_pil else 'N/A'}",
            level="DEBUG",
        )
        log(
            f"is_boundary_cache_current: Args - image_size: {image_size}, cell_ids_count: {len(cell_ids_to_draw)}, show_deselected: {show_deselected_state}",
            level="DEBUG",
        )
        if self._cached_full_res_boundary_L_pil is None:
            log(
                "is_boundary_cache_current: Cache miss - _cached_full_res_boundary_L_pil is None.",
                level="DEBUG",
            )
            return False
        cell_ids_tuple = tuple(sorted(list(cell_ids_to_draw)))
        # Check actual PIL image size first, then the stored state for image_size.
        if self._cached_full_res_boundary_L_pil.size != image_size:
            log(
                f"is_boundary_cache_current: Cache miss - PIL size mismatch. Cached PIL: {self._cached_full_res_boundary_L_pil.size}, Requested: {image_size}",
                level="DEBUG",
            )
            return False
        if self._cached_boundary_image_size_state != image_size:
            log(
                f"is_boundary_cache_current: Cache miss - Cached image_size_state mismatch. Cached State: {self._cached_boundary_image_size_state}, Requested: {image_size}",
                level="DEBUG",
            )
            return False
        if self._cached_boundary_ids_tuple_state != cell_ids_tuple:
            log(
                f"is_boundary_cache_current: Cache miss - IDs mismatch. Cached: {self._cached_boundary_ids_tuple_state}, Requested: {cell_ids_tuple}",
                level="DEBUG",
            )
            return False
        if self._cached_boundary_show_deselected_state != show_deselected_state:
            log(
                f"is_boundary_cache_current: Cache miss - Show deselected state mismatch. Cached: {self._cached_boundary_show_deselected_state}, Requested: {show_deselected_state}",
                level="DEBUG",
            )
            return False
        log("ImageOverlayProcessor: Boundary cache is current.")
        return True

    def is_number_positions_cache_current(
        self, cell_ids_to_draw: set[int], show_deselected_state: bool
    ) -> bool:
        log(
            f"is_number_positions_cache_current: Checking. Current state - Positions: {'Exists' if self._cached_cell_number_positions is not None else 'None'}, IDs: {self._cached_cell_number_ids_tuple_state}, ShowDeselected: {self._cached_cell_number_show_deselected_state}",
            level="DEBUG",
        )
        log(
            f"is_number_positions_cache_current: Args - cell_ids_count: {len(cell_ids_to_draw)}, show_deselected: {show_deselected_state}",
            level="DEBUG",
        )
        if self._cached_cell_number_positions is None:
            log(
                "is_number_positions_cache_current: Cache miss - _cached_cell_number_positions is None.",
                level="DEBUG",
            )
            return False
        cell_ids_tuple = tuple(sorted(list(cell_ids_to_draw)))
        if self._cached_cell_number_ids_tuple_state != cell_ids_tuple:
            log(
                f"is_number_positions_cache_current: Cache miss - IDs mismatch. Cached: {self._cached_cell_number_ids_tuple_state}, Requested: {cell_ids_tuple}",
                level="DEBUG",
            )
            return False
        if self._cached_cell_number_show_deselected_state != show_deselected_state:
            log(
                f"is_number_positions_cache_current: Cache miss - Show deselected state mismatch. Cached: {self._cached_cell_number_show_deselected_state}, Requested: {show_deselected_state}",
                level="DEBUG",
            )
            return False
        log("ImageOverlayProcessor: Number positions cache is current.")
        return True

    def _calculate_and_sort_cell_number_info(
        self, cell_ids_to_draw: set[int]
    ) -> list[dict]:
        """
        Calculates positions for cell numbers, sorted for consistent numbering.
        Logic adapted from ImageViewRenderer and FileIOController.
        Returns a list of dicts, each containing 'id', 'top', 'left',
        'center_orig_x', 'center_orig_y' for each cell_id in cell_ids_to_draw.
        """
        log(
            f"_calculate_and_sort_cell_number_info: Entry. cell_ids_to_draw_count: {len(cell_ids_to_draw)}",
            level="DEBUG",
        )
        if (
            self.application_model.image_data.mask_array is None
            or not cell_ids_to_draw
            or self.application_model.image_data.original_image is None
        ):
            log(
                "_calculate_and_sort_cell_number_info: Prerequisites not met (no mask_array, no cell_ids_to_draw, or no original_image). Returning empty list.",
                level="DEBUG",
            )
            return []

        cell_info_for_sorting = []
        img_h_orig, img_w_orig = (
            self.application_model.image_data.original_image.height,
            self.application_model.image_data.original_image.width,
        )
        current_mask_array = self.application_model.image_data.mask_array

        for cell_id_val in cell_ids_to_draw:
            log(
                f"_calculate_and_sort_cell_number_info: Processing cell_id: {cell_id_val}",
                level="DEBUG",
            )
            if cell_id_val == 0:
                log(
                    "_calculate_and_sort_cell_number_info: Skipping cell_id 0.",
                    level="DEBUG",
                )
                continue

            single_cell_mask = current_mask_array == cell_id_val
            if not np.any(single_cell_mask):
                log(
                    f"_calculate_and_sort_cell_number_info: No pixels found for cell_id {cell_id_val} in mask_array. Skipping.",
                    level="DEBUG",
                )
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
                log(
                    f"_calculate_and_sort_cell_number_info: Cell {cell_id_val} - Eroded mask is not empty. Calculating center from eroded mask.",
                    level="DEBUG",
                )
                cy_eroded_com, cx_eroded_com = ndimage.center_of_mass(eroded_mask)
                cy_eroded_idx, cx_eroded_idx = (
                    int(round(cy_eroded_com)),
                    int(round(cx_eroded_com)),
                )
                log(
                    f"_calculate_and_sort_cell_number_info: Cell {cell_id_val} - Eroded COM (float): ({cx_eroded_com:.2f}, {cy_eroded_com:.2f}), Eroded COM (int_idx): ({cx_eroded_idx}, {cy_eroded_idx})",
                    level="DEBUG",
                )
                if (
                    0 <= cy_eroded_idx < eroded_mask.shape[0]
                    and 0 <= cx_eroded_idx < eroded_mask.shape[1]
                    and eroded_mask[cy_eroded_idx, cx_eroded_idx]
                ):
                    chosen_cx, chosen_cy = cx_eroded_com, cy_eroded_com
                    log(
                        f"_calculate_and_sort_cell_number_info: Cell {cell_id_val} - Center chosen based on eroded COM point being within the eroded mask.",
                        level="DEBUG",
                    )
                else:
                    log(
                        f"_calculate_and_sort_cell_number_info: Cell {cell_id_val} - Eroded COM point not within eroded mask. Trying pole of inaccessibility for eroded mask.",
                        level="DEBUG",
                    )
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
                        log(
                            f"_calculate_and_sort_cell_number_info: Cell {cell_id_val} - Center chosen based on eroded pole of inaccessibility: ({chosen_cx:.2f}, {chosen_cy:.2f})",
                            level="DEBUG",
                        )
                    else:
                        log(
                            f"_calculate_and_sort_cell_number_info: Cell {cell_id_val} - Eroded mask distance transform is all zero. No center found from eroded mask.",
                            level="DEBUG",
                        )
            else:
                log(
                    f"_calculate_and_sort_cell_number_info: Cell {cell_id_val} - Eroded mask is empty. Will use original mask for center calculation.",
                    level="DEBUG",
                )

            if chosen_cx == -1.0:  # Fallback if eroded mask didn't yield a point
                log(
                    f"_calculate_and_sort_cell_number_info: Cell {cell_id_val} - Fallback: Calculating center from original single_cell_mask.",
                    level="DEBUG",
                )
                cy_orig_com, cx_orig_com = ndimage.center_of_mass(single_cell_mask)
                cy_idx, cx_idx = int(round(cy_orig_com)), int(round(cx_orig_com))
                log(
                    f"_calculate_and_sort_cell_number_info: Cell {cell_id_val} - Original COM (float): ({cx_orig_com:.2f}, {cy_orig_com:.2f}), Original COM (int_idx): ({cx_idx}, {cy_idx})",
                    level="DEBUG",
                )
                if (
                    0 <= cy_idx < single_cell_mask.shape[0]
                    and 0 <= cx_idx < single_cell_mask.shape[1]
                    and single_cell_mask[cy_idx, cx_idx]
                ):
                    chosen_cx, chosen_cy = cx_orig_com, cy_orig_com
                    log(
                        f"_calculate_and_sort_cell_number_info: Cell {cell_id_val} - Center chosen based on original COM point being within the original mask.",
                        level="DEBUG",
                    )
                else:
                    log(
                        f"_calculate_and_sort_cell_number_info: Cell {cell_id_val} - Original COM point not within original mask. Trying pole of inaccessibility for original mask.",
                        level="DEBUG",
                    )
                    dist_transform_orig = ndimage.distance_transform_edt(
                        single_cell_mask
                    )
                    if np.any(dist_transform_orig):
                        pole_y_orig, pole_x_orig = np.unravel_index(
                            np.argmax(dist_transform_orig), dist_transform_orig.shape
                        )
                        chosen_cx, chosen_cy = float(pole_x_orig), float(pole_y_orig)
                        log(
                            f"_calculate_and_sort_cell_number_info: Cell {cell_id_val} - Center chosen based on original pole of inaccessibility: ({chosen_cx:.2f}, {chosen_cy:.2f})",
                            level="DEBUG",
                        )
                    else:
                        # This case should ideally not be reached if np.any(single_cell_mask) was true earlier.
                        log(
                            f"_calculate_and_sort_cell_number_info: Cell {cell_id_val} - Original mask distance transform is all zero. Defaulting to original COM. This is unexpected if mask was not empty.",
                            level="WARNING",
                        )
                        chosen_cx, chosen_cy = (
                            cx_orig_com,
                            cy_orig_com,
                        )  # Default to COM if all else fails

            log(
                f"_calculate_and_sort_cell_number_info: Cell {cell_id_val} - Chosen center before clamping: ({chosen_cx:.2f}, {chosen_cy:.2f})",
                level="DEBUG",
            )

            margin = constants.IMAGE_BOUNDARY_MARGIN_FOR_NUMBER_PLACEMENT
            cx_min_clamp = margin
            cx_max_clamp = max(margin, img_w_orig - 1 - margin)
            cy_min_clamp = margin
            cy_max_clamp = max(margin, img_h_orig - 1 - margin)
            log(
                f"_calculate_and_sort_cell_number_info: Cell {cell_id_val} - Clamping values - CX_min: {cx_min_clamp}, CX_max: {cx_max_clamp}, CY_min: {cy_min_clamp}, CY_max: {cy_max_clamp}",
                level="DEBUG",
            )

            final_cx_orig = max(cx_min_clamp, min(chosen_cx, cx_max_clamp))
            final_cy_orig = max(cy_min_clamp, min(chosen_cy, cy_max_clamp))
            log(
                f"_calculate_and_sort_cell_number_info: Cell {cell_id_val} - Center after margin clamping: ({final_cx_orig:.2f}, {final_cy_orig:.2f})",
                level="DEBUG",
            )

            final_cx_orig = max(0, min(final_cx_orig, img_w_orig - 1))
            final_cy_orig = max(0, min(final_cy_orig, img_h_orig - 1))
            log(
                f"_calculate_and_sort_cell_number_info: Cell {cell_id_val} - Final center after image boundary clamping (orig_coords): ({final_cx_orig:.2f}, {final_cy_orig:.2f}). Top-left for sorting: ({left_most_in_top_row}, {top_most})",
                level="DEBUG",
            )

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

    def get_cached_mask_layer_rgb(
        self,
        base_pil_image_size: tuple[int, int],
        cell_ids_to_draw: set[int],
        show_deselected_masks_only: bool,
    ) -> Image.Image:
        """
        Generates or retrieves from cache a PIL Image representing only the colored mask layer.
        The returned image will have RGB colors for masks on a black background.
        The base_pil_image_size is used for determining the output image size.
        """
        current_ids_tuple_for_cache = tuple(sorted(list(cell_ids_to_draw)))
        log(
            f"get_cached_mask_layer_rgb: Entry. base_pil_image_size: {base_pil_image_size}, cell_ids_to_draw_count: {len(cell_ids_to_draw)}, show_deselected: {show_deselected_masks_only}",
            level="DEBUG",
        )

        # Check cache
        if (
            self._cached_full_res_mask_rgb is not None
            and self._cached_mask_ids_tuple_state == current_ids_tuple_for_cache
            and self._cached_show_deselected_mask_state == show_deselected_masks_only
            and self._cached_full_res_mask_rgb.size
            == base_pil_image_size  # Ensure size matches
        ):
            log("ImageOverlayProcessor: Mask RGB layer cache hit.")
            return self._cached_full_res_mask_rgb

        log("ImageOverlayProcessor: Mask RGB layer cache miss or stale. Regenerating.")

        if (
            self.application_model.image_data.mask_array is None
            or base_pil_image_size is None
            or base_pil_image_size[0] <= 0
            or base_pil_image_size[1] <= 0
        ):
            img_size = (
                base_pil_image_size
                if base_pil_image_size
                and base_pil_image_size[0] > 0
                and base_pil_image_size[1] > 0
                else (1, 1)
            )
            log(
                f"get_cached_mask_layer_rgb: Prerequisites for mask generation not met or invalid base_pil_image_size. Returning black RGB image of size {img_size}.",
                level="WARNING",
            )
            self._cached_full_res_mask_rgb = Image.new(
                "RGB", img_size, constants.COLOR_BLACK_STR
            )
            self._cached_mask_ids_tuple_state = current_ids_tuple_for_cache
            self._cached_show_deselected_mask_state = show_deselected_masks_only
            return self._cached_full_res_mask_rgb

        current_mask_array = self.application_model.image_data.mask_array

        rng = np.random.default_rng(seed=constants.RANDOM_SEED_MASKS)
        all_unique_mask_ids = np.unique(current_mask_array)
        color_map = {
            uid: tuple(rng.integers(50, 200, size=3))
            for uid in all_unique_mask_ids
            if uid != 0
        }

        # Initialize temp_mask_np_for_pil based on base_pil_image_size.
        # Assumes current_mask_array.shape[:2] matches (base_pil_image_size[1], base_pil_image_size[0])
        temp_mask_np_for_pil = np.zeros(
            (base_pil_image_size[1], base_pil_image_size[0], 3), dtype=np.uint8
        )
        any_mask_drawn = False
        log(
            f"get_cached_mask_layer_rgb: Initialized temp_mask_np_for_pil with shape {temp_mask_np_for_pil.shape}. Color map has {len(color_map)} color entries. ids_for_actual_drawing_count: {len(cell_ids_to_draw)}",
            level="DEBUG",
        )

        ids_for_actual_drawing = cell_ids_to_draw

        for cell_id_val in ids_for_actual_drawing:
            if cell_id_val != 0 and cell_id_val in color_map:
                log(
                    f"get_cached_mask_layer_rgb: Drawing mask for cell_id {cell_id_val} with color {color_map[cell_id_val]}",
                    level="DEBUG",
                )
                mask_pixels_orig_res = (
                    current_mask_array == cell_id_val
                )  # This is at full original resolution
                if np.any(mask_pixels_orig_res):
                    if temp_mask_np_for_pil.shape[:2] == mask_pixels_orig_res.shape:
                        temp_mask_np_for_pil[mask_pixels_orig_res] = color_map[
                            cell_id_val
                        ]
                        any_mask_drawn = True
                        log(
                            f"get_cached_mask_layer_rgb: Applied mask for cell_id {cell_id_val}. any_mask_drawn set to True.",
                            level="DEBUG",
                        )
                    else:
                        log(
                            f"get_cached_mask_layer_rgb: Shape mismatch for cell_id {cell_id_val}. temp_mask_np shape: {temp_mask_np_for_pil.shape[:2]}, mask_pixels_orig_res shape: {mask_pixels_orig_res.shape}. Skipping draw for this cell.",
                            level="WARNING",
                        )
                else:
                    log(
                        f"get_cached_mask_layer_rgb: No pixels for cell_id {cell_id_val} in current_mask_array (np.any(mask_pixels_orig_res) is False). Skipping draw.",
                        level="DEBUG",
                    )
            elif cell_id_val == 0:
                log("get_cached_mask_layer_rgb: Skipping cell_id 0.", level="DEBUG")
            else:  # cell_id_val not in color_map (should not happen if all_unique_mask_ids was comprehensive and cell_id_val is from ids_for_actual_drawing)
                log(
                    f"get_cached_mask_layer_rgb: Cell ID {cell_id_val} is not 0 and not found in color_map. Skipping. This might indicate an issue with ids_for_actual_drawing or color_map generation.",
                    level="WARNING",
                )

        if not any_mask_drawn:
            log(
                "get_cached_mask_layer_rgb: No masks were actually drawn onto the numpy array. Returning new black RGB image.",
                level="DEBUG",
            )
            generated_mask_layer_pil = Image.new(
                "RGB", base_pil_image_size, constants.COLOR_BLACK_STR
            )
        else:
            log(
                "get_cached_mask_layer_rgb: Masks were drawn. Creating PIL image from numpy array.",
                level="DEBUG",
            )
            generated_mask_layer_pil = Image.fromarray(temp_mask_np_for_pil, "RGB")

        self._cached_full_res_mask_rgb = generated_mask_layer_pil
        self._cached_mask_ids_tuple_state = current_ids_tuple_for_cache
        self._cached_show_deselected_mask_state = show_deselected_masks_only

        log(
            f"ImageOverlayProcessor: Mask RGB layer cache regenerated and updated. Size: {self._cached_full_res_mask_rgb.size}",
            level="INFO",
        )
        return self._cached_full_res_mask_rgb

    def blend_image_with_mask_layer(
        self, base_image: Image.Image, mask_layer: Image.Image, alpha: float
    ) -> Image.Image:
        """
        Blends a base image with a mask layer.
        Assumes mask_layer is an RGB image (colors on black) of the same size as base_image.
        """
        log(
            f"blend_image_with_mask_layer: Entry. Base image size: {base_image.size if base_image else 'None'} (mode: {base_image.mode if base_image else 'N/A'}), Mask layer size: {mask_layer.size if mask_layer else 'None'} (mode: {mask_layer.mode if mask_layer else 'N/A'}), Alpha: {alpha}",
            level="DEBUG",
        )
        if base_image is None or mask_layer is None:
            log(
                "blend_image_with_mask_layer: Base or mask layer is None. Returning original base_image.",
                level="WARNING",
            )
            return base_image

        if base_image.size != mask_layer.size:
            log(
                f"ImageOverlayProcessor Warning: Size mismatch in blend_image_with_mask_layer. Base: {base_image.size}, Mask Layer: {mask_layer.size}. Resizing mask layer to base image size.",
                level="WARNING",
            )
            mask_layer = mask_layer.resize(base_image.size, Image.NEAREST)
            log(
                f"blend_image_with_mask_layer: Mask layer resized to {mask_layer.size}",
                level="DEBUG",
            )

        base_image_rgb = base_image
        if base_image.mode != "RGB":
            log(
                f"blend_image_with_mask_layer: Converting base_image from mode {base_image.mode} to RGB.",
                level="DEBUG",
            )
            base_image_rgb = base_image.convert("RGB")

        # Ensure mask_layer is also RGB
        mask_layer_rgb = mask_layer
        if mask_layer.mode != "RGB":
            log(
                f"blend_image_with_mask_layer: Converting mask_layer from mode {mask_layer.mode} to RGB.",
                level="DEBUG",
            )
            mask_layer_rgb = mask_layer.convert("RGB")

        blended_image = Image.blend(base_image_rgb, mask_layer_rgb, alpha=alpha)
        log(
            f"blend_image_with_mask_layer: Blending complete. Output image size: {blended_image.size}, mode: {blended_image.mode}. Exiting.",
            level="DEBUG",
        )
        return blended_image

    def _generate_full_res_boundary_mask_L_pil(
        self,
        image_size: tuple[int, int],
        cell_ids_to_draw: set[int],
    ) -> Image.Image:
        """
        Generates a full-resolution L-mode PIL image for boundaries.
        White boundaries (255) on black (0) background.
        image_size is (width, height).
        """
        log(
            f"_generate_full_res_boundary_mask_L_pil: Entry. image_size: {image_size}, cell_ids_to_draw_count: {len(cell_ids_to_draw)}",
            level="DEBUG",
        )
        if (
            self.application_model.image_data.mask_array is None
            or image_size is None
            or image_size[0] <= 0
            or image_size[1] <= 0
        ):
            final_size = (
                image_size
                if image_size and image_size[0] > 0 and image_size[1] > 0
                else (1, 1)
            )
            log(
                f"_generate_full_res_boundary_mask_L_pil: Prerequisites not met (no mask_array, or invalid image_size). Returning black L image of size {final_size}.",
                level="WARNING",
            )
            return Image.new("L", final_size, 0)

        current_mask_array = self.application_model.image_data.mask_array
        exact_boundaries_np = self.application_model.image_data.exact_boundaries

        if exact_boundaries_np is None or not exact_boundaries_np.any():
            return Image.new("L", image_size, 0)

        if current_mask_array.shape != exact_boundaries_np.shape:
            log(
                f"ImageOverlayProcessor Warning: Mismatch between mask_array shape {current_mask_array.shape} and exact_boundaries_np shape {exact_boundaries_np.shape}. This might lead to issues if IDs are misaligned when selecting boundaries.",
                level="WARNING",
            )
            pass

        boundary_to_draw_on_image_np = np.zeros_like(exact_boundaries_np, dtype=bool)
        any_boundary_drawn = False
        log(
            f"_generate_full_res_boundary_mask_L_pil: Initialized boundary_to_draw_on_image_np (boolean mask) with shape {boundary_to_draw_on_image_np.shape}",
            level="DEBUG",
        )

        for cid in cell_ids_to_draw:
            if cid != 0:
                log(
                    f"_generate_full_res_boundary_mask_L_pil: Processing cell_id {cid} for boundary inclusion.",
                    level="DEBUG",
                )
                if np.any(
                    current_mask_array == cid
                ):  # Check if cell_id actually exists in mask
                    cell_mask_region = current_mask_array == cid
                    current_cell_boundary = exact_boundaries_np & cell_mask_region
                    if np.any(current_cell_boundary):
                        boundary_to_draw_on_image_np |= current_cell_boundary
                        any_boundary_drawn = True
                        log(
                            f"_generate_full_res_boundary_mask_L_pil: Boundary for cell_id {cid} (sum: {np.sum(current_cell_boundary)}) added to collective boolean mask.",
                            level="DEBUG",
                        )
                    else:
                        log(
                            f"_generate_full_res_boundary_mask_L_pil: No exact boundary found for cell_id {cid} within its mask region (exact_boundaries_np & cell_mask_region was empty).",
                            level="DEBUG",
                        )
                else:
                    log(
                        f"_generate_full_res_boundary_mask_L_pil: Cell ID {cid} not found in current_mask_array. Cannot select boundary for this ID.",
                        level="DEBUG",
                    )

        if not any_boundary_drawn:
            log(
                f"_generate_full_res_boundary_mask_L_pil: No boundaries were drawn into the boolean mask for any selected cell ID. Returning black L image of size {image_size}.",
                level="DEBUG",
            )
            return Image.new("L", image_size, 0)

        dilation_iters = constants.DILATION_ITERATIONS_FOR_BOUNDARY_DISPLAY
        log(
            f"_generate_full_res_boundary_mask_L_pil: Dilation iterations for boundary display: {dilation_iters}",
            level="DEBUG",
        )
        if dilation_iters > 0:
            dilated_boundaries_np = self.dilate_boundary(
                boundary_to_draw_on_image_np, iterations=dilation_iters
            )
            log(
                f"_generate_full_res_boundary_mask_L_pil: Boundaries dilated. Original nnz: {np.count_nonzero(boundary_to_draw_on_image_np)}, Dilated nnz: {np.count_nonzero(dilated_boundaries_np)}",
                level="DEBUG",
            )
        else:
            dilated_boundaries_np = boundary_to_draw_on_image_np
            log(
                "_generate_full_res_boundary_mask_L_pil: No dilation performed as iterations <= 0.",
                level="DEBUG",
            )

        if not np.any(
            dilated_boundaries_np
        ):  # Should be redundant if any_boundary_drawn was true and dilation didn't remove everything
            log(
                f"_generate_full_res_boundary_mask_L_pil: Dilated (or original) boundaries_np is empty. Returning black L image of size {image_size}.",
                level="DEBUG",
            )
            return Image.new("L", image_size, 0)

        # Convert boolean numpy array to L mode PIL image
        log(
            f"_generate_full_res_boundary_mask_L_pil: Converting final boolean boundary mask (shape: {dilated_boundaries_np.shape}, nnz: {np.count_nonzero(dilated_boundaries_np)}) to L-mode PIL.",
            level="DEBUG",
        )
        boundary_L_pil = Image.fromarray(
            dilated_boundaries_np.astype(np.uint8) * 255, mode="L"
        )
        log(
            f"_generate_full_res_boundary_mask_L_pil: Generated L-mode PIL from numpy. Initial size: {boundary_L_pil.size}",
            level="DEBUG",
        )

        # Ensure final PIL is of the requested image_size
        if boundary_L_pil.size != image_size:
            log(
                f"ImageOverlayProcessor: Generated boundary L PIL size {boundary_L_pil.size} differs from requested image_size {image_size}. Resizing using NEAREST.",
                level="WARNING",
            )
            boundary_L_pil = boundary_L_pil.resize(
                image_size, Image.NEAREST
            )  # NEAREST for masks
            log(
                f"_generate_full_res_boundary_mask_L_pil: Resized L-mode PIL to final target size {image_size}.",
                level="DEBUG",
            )

        log(
            f"_generate_full_res_boundary_mask_L_pil: Exit. Returning L-mode PIL of size {boundary_L_pil.size}",
            level="DEBUG",
        )
        return boundary_L_pil

    def get_boundary_mask_L_pil(
        self,
        image_size: tuple[int, int],
        cell_ids_to_draw: set[int],
        show_deselected_state: bool,  # For cache key consistency with masks
    ) -> Image.Image:
        """
        Provides a cached L-mode PIL image for boundaries (white on black).
        image_size is (width, height).
        """
        current_ids_tuple_for_cache = tuple(sorted(list(cell_ids_to_draw)))
        log(
            f"get_boundary_mask_L_pil: Entry. image_size: {image_size}, cell_ids_to_draw_count: {len(cell_ids_to_draw)}, show_deselected_state: {show_deselected_state}",
            level="DEBUG",
        )

        if (
            self._cached_full_res_boundary_L_pil is not None
            and self._cached_boundary_ids_tuple_state == current_ids_tuple_for_cache
            and self._cached_boundary_show_deselected_state == show_deselected_state
            and self._cached_boundary_image_size_state == image_size
            and
            # Ensure the cached PIL image itself is not None, size check is implicit with image_size_state
            self._cached_full_res_boundary_L_pil.size == image_size
        ):
            log("ImageOverlayProcessor: Boundary L PIL cache hit.")
            return self._cached_full_res_boundary_L_pil

        log("ImageOverlayProcessor: Boundary L PIL cache miss or stale. Regenerating.")

        generated_boundary_L_pil = self._generate_full_res_boundary_mask_L_pil(
            image_size=image_size,
            cell_ids_to_draw=cell_ids_to_draw,
            # show_deselected_state=show_deselected_state # Pass if _generate needs it
        )

        # Update cache
        self._cached_full_res_boundary_L_pil = generated_boundary_L_pil
        self._cached_boundary_ids_tuple_state = current_ids_tuple_for_cache
        self._cached_boundary_show_deselected_state = show_deselected_state
        self._cached_boundary_image_size_state = image_size

        log("ImageOverlayProcessor: Boundary L PIL cache regenerated.")
        return self._cached_full_res_boundary_L_pil

    def draw_boundaries_on_pil(
        self, base_pil_image: Image.Image, cell_ids_to_draw: set[int]
    ) -> Image.Image:
        log(
            f"draw_boundaries_on_pil: Entry. Base image size: {base_pil_image.size if base_pil_image else 'None'} (mode: {base_pil_image.mode if base_pil_image else 'N/A'}), cell_ids_to_draw_count: {len(cell_ids_to_draw)}",
            level="DEBUG",
        )
        if (
            self.application_model.image_data.mask_array is None
            or not cell_ids_to_draw
            or base_pil_image is None
        ):
            log(
                "draw_boundaries_on_pil: Prerequisites not met (no mask_array, no cell_ids, or no base_image). Returning original base_pil_image.",
                level="DEBUG",
            )
            return base_pil_image

        current_mask_array = self.application_model.image_data.mask_array
        exact_boundaries = self.application_model.image_data.exact_boundaries

        if exact_boundaries is None or not exact_boundaries.any():
            log(
                "draw_boundaries_on_pil: No exact boundaries data available in model (exact_boundaries is None or empty). Returning original base_pil_image.",
                level="DEBUG",
            )
            return base_pil_image

        log(
            f"draw_boundaries_on_pil: current_mask_array shape: {current_mask_array.shape}, exact_boundaries shape: {exact_boundaries.shape}",
            level="DEBUG",
        )

        boundary_to_draw_on_image = np.zeros_like(exact_boundaries, dtype=bool)
        any_boundary_drawn = False
        log(
            f"draw_boundaries_on_pil: Initialized boundary_to_draw_on_image (boolean mask) with shape {boundary_to_draw_on_image.shape}",
            level="DEBUG",
        )

        for cid in cell_ids_to_draw:
            if cid != 0:
                log(
                    f"draw_boundaries_on_pil: Processing cell_id {cid} for boundary drawing.",
                    level="DEBUG",
                )
                cell_mask_region = current_mask_array == cid
                if np.any(cell_mask_region):
                    current_cell_boundary = exact_boundaries & cell_mask_region
                    if np.any(current_cell_boundary):
                        boundary_to_draw_on_image |= current_cell_boundary
                        any_boundary_drawn = True
                        log(
                            f"draw_boundaries_on_pil: Boundary for cell_id {cid} (sum: {np.sum(current_cell_boundary)}) added to boolean mask.",
                            level="DEBUG",
                        )
                    else:
                        log(
                            f"draw_boundaries_on_pil: No exact boundary found for cell_id {cid} within its mask region (exact_boundaries & cell_mask_region was empty).",
                            level="DEBUG",
                        )
                else:
                    log(
                        f"draw_boundaries_on_pil: Cell ID {cid} not found in current_mask_array. Skipping boundary selection for this ID.",
                        level="DEBUG",
                    )
            else:
                log("draw_boundaries_on_pil: Skipping cell_id 0.", level="DEBUG")

        if not any_boundary_drawn:
            log(
                "draw_boundaries_on_pil: No boundaries were selected for any provided cell ID. Returning original base_pil_image.",
                level="DEBUG",
            )
            return base_pil_image

        dilation_iters = constants.DILATION_ITERATIONS_FOR_BOUNDARY_DISPLAY
        log(
            f"draw_boundaries_on_pil: Dilation iterations for display: {dilation_iters}",
            level="DEBUG",
        )
        if dilation_iters > 0:
            dilated_boundaries = self.dilate_boundary(
                boundary_to_draw_on_image, iterations=dilation_iters
            )
            log(
                f"draw_boundaries_on_pil: Boundaries dilated for display. Original nnz: {np.count_nonzero(boundary_to_draw_on_image)}, Dilated nnz: {np.count_nonzero(dilated_boundaries)}",
                level="DEBUG",
            )
        else:
            dilated_boundaries = boundary_to_draw_on_image
            log(
                "draw_boundaries_on_pil: No dilation performed as iterations <= 0.",
                level="DEBUG",
            )

        if not np.any(dilated_boundaries):
            log(
                "draw_boundaries_on_pil: Dilated (or original) boundaries mask is empty. Returning original base_pil_image.",
                level="DEBUG",
            )
            return base_pil_image

        output_image_np = np.array(base_pil_image.convert("RGB"))
        log(
            f"draw_boundaries_on_pil: Converted base_pil_image to RGB numpy array. Shape: {output_image_np.shape}",
            level="DEBUG",
        )
        boundary_color_map_pil = constants.BOUNDARY_COLOR_MAP_PIL
        gui_boundary_color_name = (
            self.application_model.display_state.boundary_color_name
        )
        chosen_color_np = np.array(
            boundary_color_map_pil.get(
                gui_boundary_color_name,
                constants.BOUNDARY_COLOR_MAP_PIL[constants.BOUNDARY_COLOR_DEFAULT],
            )
        )
        log(
            f"draw_boundaries_on_pil: Boundary color name from display_state: '{gui_boundary_color_name}', chosen_color_np RGB tuple: {chosen_color_np.tolist()}",
            level="DEBUG",
        )

        if dilated_boundaries.shape == output_image_np.shape[:2]:
            output_image_np[dilated_boundaries] = chosen_color_np
            log(
                f"draw_boundaries_on_pil: Applied boundary color to output_image_np where dilated_boundaries is True (nnz: {np.count_nonzero(dilated_boundaries)}).",
                level="DEBUG",
            )
        else:
            log(
                f"ImageOverlayProcessor Warning: Shape mismatch for boundary drawing. Output_image_np shape: {output_image_np.shape[:2]}, Dilated_boundaries shape: {dilated_boundaries.shape}. Returning original base_pil_image.",
                level="WARNING",
            )
            return base_pil_image

        final_pil_image = Image.fromarray(output_image_np)
        log(
            f"draw_boundaries_on_pil: Exit. Returning new PIL image with boundaries. Size: {final_pil_image.size}, mode: {final_pil_image.mode}",
            level="DEBUG",
        )
        return final_pil_image

    def get_dynamic_cell_number_font_size(self, image_width, image_height):
        reference_size = 1024
        base_font_size = constants.CELL_NUMBERING_FONT_SIZE_ORIG_IMG
        scale = ((image_width * image_height) / (reference_size**2)) ** 0.5
        return int(round(base_font_size * scale))

    def draw_numbers_on_pil(
        self,
        base_pil_image: Image.Image,
        cell_ids_to_draw: set[int],
        font_size: int = None,
    ) -> Image.Image:
        if font_size is None:
            font_size = self.get_dynamic_cell_number_font_size(
                base_pil_image.width, base_pil_image.height
            )
        log(
            f"draw_numbers_on_pil: Entry. Base image size: {base_pil_image.size if base_pil_image else 'None'} (mode: {base_pil_image.mode if base_pil_image else 'N/A'}), cell_ids_to_draw_count: {len(cell_ids_to_draw)}, font_size: {font_size}",
            level="DEBUG",
        )
        if (
            self.application_model.image_data.mask_array is None
            or not cell_ids_to_draw
            or base_pil_image is None
            or self.application_model.image_data.original_image is None
        ):
            log(
                "draw_numbers_on_pil: Prerequisites not met (no mask_array, no cell_ids, no base_image, or no original_image in model). Returning original base_pil_image.",
                level="DEBUG",
            )
            return base_pil_image

        image_to_draw_on = base_pil_image.copy()
        log(
            f"draw_numbers_on_pil: Copied base_pil_image. New image_to_draw_on size: {image_to_draw_on.size}, mode: {image_to_draw_on.mode}",
            level="DEBUG",
        )
        draw_on_pil = ImageDraw.Draw(image_to_draw_on)
        current_boundary_color_name = (
            self.application_model.display_state.boundary_color_name
        )
        text_color_map = constants.BOUNDARY_COLOR_MAP_PIL
        text_color_tuple = text_color_map.get(
            current_boundary_color_name, constants.BOUNDARY_COLOR_MAP_PIL["Green"]
        )
        log(
            f"draw_numbers_on_pil: Text color for numbers (based on boundary color_name '{current_boundary_color_name}'): {text_color_tuple}",
            level="DEBUG",
        )

        try:
            font = ImageFont.truetype(constants.DEFAULT_FONT, size=font_size)
            log(
                f"draw_numbers_on_pil: Loaded font '{constants.DEFAULT_FONT}' with size {font_size}",
                level="DEBUG",
            )
        except IOError:
            log(
                f"draw_numbers_on_pil: Failed to load font '{constants.DEFAULT_FONT}'. Loading default PIL font with size {font_size}.",
                level="WARNING",
            )
            font = ImageFont.load_default(size=font_size)

        sorted_cell_info = self._calculate_and_sort_cell_number_info(cell_ids_to_draw)
        log(
            f"draw_numbers_on_pil: Called _calculate_and_sort_cell_number_info. Got {len(sorted_cell_info)} cells.",
            level="DEBUG",
        )

        if not sorted_cell_info:
            log(
                "draw_numbers_on_pil: No sorted_cell_info available (e.g., no valid cells or all were skipped). Returning original image_to_draw_on (the copy).",
                level="DEBUG",
            )
            return image_to_draw_on  # Return the copy

        num_drawn_actually = 0
        log(
            f"draw_numbers_on_pil: Starting to draw {len(sorted_cell_info)} numbers onto the image.",
            level="DEBUG",
        )
        for i, cell_data in enumerate(sorted_cell_info):
            display_number = str(i + 1)
            center_x_pil_orig = cell_data["center_orig_x"]
            center_y_pil_orig = cell_data["center_orig_y"]
            log(
                f"draw_numbers_on_pil: Drawing number '{display_number}' for cell_id {cell_data['id']} at original image coordinates (PIL coords): ({center_x_pil_orig:.2f}, {center_y_pil_orig:.2f})",
                level="DEBUG",
            )

            try:
                if hasattr(draw_on_pil.textbbox, "__call__"):  # Pillow 9.2.0+
                    bbox = draw_on_pil.textbbox(
                        (center_x_pil_orig, center_y_pil_orig),
                        display_number,
                        font=font,
                        anchor="lt",  # anchor typically means top-left of the bbox for the given point
                    )
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    # To center the text on (center_x_pil_orig, center_y_pil_orig), adjust by half width/height
                    final_text_x = center_x_pil_orig - text_width / 2
                    final_text_y = center_y_pil_orig - text_height / 2
                    log(
                        f"draw_numbers_on_pil: Number '{display_number}' using textbbox (Pillow 9.2.0+). bbox: {bbox}, text_width: {text_width:.2f}, text_height: {text_height:.2f}. Calculated final_text_x: {final_text_x:.2f}, final_text_y: {final_text_y:.2f}",
                        level="DEBUG",
                    )
                else:  # Older Pillow versions
                    text_width, text_height = draw_on_pil.textsize(
                        display_number, font=font
                    )
                    # To center the text on (center_x_pil_orig, center_y_pil_orig), adjust by half width/height
                    final_text_x = center_x_pil_orig - text_width / 2
                    final_text_y = center_y_pil_orig - text_height / 2
                    log(
                        f"draw_numbers_on_pil: Number '{display_number}' using textsize (Older Pillow). text_width: {text_width:.2f}, text_height: {text_height:.2f}. Calculated final_text_x: {final_text_x:.2f}, final_text_y: {final_text_y:.2f}",
                        level="DEBUG",
                    )
            except (
                AttributeError
            ):  # Fallback if textsize/textbbox is missing for some reason
                log(
                    f"draw_numbers_on_pil: AttributeError getting text size for number '{display_number}'. Using fallback size 10x10 and attempting to center.",
                    level="WARNING",
                )
                text_width, text_height = 10, 10  # Fallback size
                final_text_x = center_x_pil_orig - text_width / 2
                final_text_y = center_y_pil_orig - text_height / 2

            # Basic check if the calculated final_text_x, final_text_y is roughly within image bounds before drawing.
            # This isn't a perfect check for full visibility but avoids drawing way off-image.
            if (
                -text_width < final_text_x < base_pil_image.width
                and -text_height < final_text_y < base_pil_image.height
            ):
                draw_on_pil.text(
                    (final_text_x, final_text_y),
                    display_number,
                    fill=text_color_tuple,
                    font=font,
                )
                num_drawn_actually += 1
                log(
                    f"draw_numbers_on_pil: Drew number '{display_number}' at ({final_text_x:.2f}, {final_text_y:.2f}). num_drawn_actually: {num_drawn_actually}",
                    level="DEBUG",
                )
            else:
                log(
                    f"draw_numbers_on_pil: Number '{display_number}' for cell {cell_data['id']} at ({final_text_x:.2f}, {final_text_y:.2f}) was out of base_pil_image bounds ({base_pil_image.width}x{base_pil_image.height}). Skipping draw for this number.",
                    level="DEBUG",
                )

        if num_drawn_actually == 0:
            # This implies sorted_cell_info was not empty but somehow no numbers were drawn (e.g., all out of bounds)
            log(
                "draw_numbers_on_pil: No numbers were actually drawn (num_drawn_actually is 0), though sorted_cell_info was present. Returning original base_pil_image (if no copy was made or no changes to copy).",
                level="DEBUG",
            )
            return image_to_draw_on  # Return the copy

        log(
            f"draw_numbers_on_pil: Exit. Drew {num_drawn_actually} numbers. Returning modified image_to_draw_on. Size: {image_to_draw_on.size}, mode: {image_to_draw_on.mode}",
            level="DEBUG",
        )
        return image_to_draw_on

    def get_cached_cell_number_positions(
        self, cell_ids_to_draw: set[int], show_deselected_state: bool
    ) -> list[dict] | None:
        """
        Provides cached cell number positions.
        The 'show_deselected_state' is used for cache key consistency.
        The underlying _calculate_and_sort_cell_number_info assumes cell_ids_to_draw is already correctly filtered.
        Returns a list of dicts or None if no positions applicable.
        """
        current_ids_tuple_for_cache = tuple(sorted(list(cell_ids_to_draw)))
        log(
            f"get_cached_cell_number_positions: Entry. cell_ids_to_draw_count: {len(cell_ids_to_draw)}, show_deselected_state: {show_deselected_state}",
            level="DEBUG",
        )

        if (
            self._cached_cell_number_positions is not None  # Check if it's populated
            and self._cached_cell_number_ids_tuple_state == current_ids_tuple_for_cache
            and self._cached_cell_number_show_deselected_state == show_deselected_state
        ):
            log(
                f"ImageOverlayProcessor: Cell number positions cache hit. Returning {len(self._cached_cell_number_positions) if self._cached_cell_number_positions is not None else 'None'} cached positions.",
                level="DEBUG",
            )
            return self._cached_cell_number_positions

        log(
            "ImageOverlayProcessor: Cell number positions cache miss or stale. Regenerating.",
            level="DEBUG",
        )

        # Regenerate using the existing method
        # _calculate_and_sort_cell_number_info returns [] if not applicable, which is fine for caching.
        calculated_positions = self._calculate_and_sort_cell_number_info(
            cell_ids_to_draw
        )
        log(
            f"get_cached_cell_number_positions: _calculate_and_sort_cell_number_info returned {len(calculated_positions)} positions.",
            level="DEBUG",
        )

        # Update cache
        self._cached_cell_number_positions = calculated_positions
        self._cached_cell_number_ids_tuple_state = current_ids_tuple_for_cache
        self._cached_cell_number_show_deselected_state = show_deselected_state

        log(
            f"ImageOverlayProcessor: Cell number positions cache regenerated and updated. Cached {len(self._cached_cell_number_positions)} positions.",
            level="INFO",
        )
        return self._cached_cell_number_positions

    def generate_composite_overlay_image(
        self,
        base_pil_image: Image.Image | None,
        image_size_pil: tuple[int, int],
        display_state,
        ids_to_display: set[int],
        zoom: float,
        pan_x: float,
        pan_y: float,
        canvas_width: int,
        canvas_height: int,
        quality: str = "final",
    ) -> Image.Image:
        """
        Generates the final composite image to be displayed on the canvas.
        Handles base image, masks, boundaries, and numbers based on display_state.
        Applies zoom, pan, and cropping.
        """
        log(
            f"generate_composite_overlay_image: Entry. Quality: '{quality}', Base image size (expected): {image_size_pil}, Num IDs to display: {len(ids_to_display)}, Zoom: {zoom:.2f}, Pan: ({pan_x:.2f}, {pan_y:.2f}), Canvas: {canvas_width}x{canvas_height}",
            level="DEBUG",
        )

        resample_filter = Image.LANCZOS if quality == "final" else Image.NEAREST
        overlay_resample_filter = (
            Image.NEAREST
        )  # Overlays are usually best with NEAREST
        log(
            f"generate_composite_overlay_image: Resample filter for base image: {resample_filter}, Overlay resample filter: {overlay_resample_filter}",
            level="DEBUG",
        )

        safe_new_width = max(1, int(image_size_pil[0] * zoom))
        safe_new_height = max(1, int(image_size_pil[1] * zoom))
        log(
            f"generate_composite_overlay_image: Calculated zoomed image dimensions (safe_new_width, safe_new_height): ({safe_new_width}, {safe_new_height})",
            level="DEBUG",
        )

        # Determine cropping and pasting parameters
        src_x1_on_zoomed_img = int(-pan_x) if pan_x < 0 else 0
        paste_dst_x_on_canvas = int(pan_x) if pan_x > 0 else 0
        src_y1_on_zoomed_img = int(-pan_y) if pan_y < 0 else 0
        paste_dst_y_on_canvas = int(pan_y) if pan_y > 0 else 0

        width_to_copy = min(
            safe_new_width - src_x1_on_zoomed_img,
            canvas_width - paste_dst_x_on_canvas,
        )
        height_to_copy = min(
            safe_new_height - src_y1_on_zoomed_img,
            canvas_height - paste_dst_y_on_canvas,
        )
        log(
            f"generate_composite_overlay_image: Cropping/pasting params - src_x1_on_zoomed: {src_x1_on_zoomed_img}, paste_dst_x_on_canvas: {paste_dst_x_on_canvas}, src_y1_on_zoomed: {src_y1_on_zoomed_img}, paste_dst_y_on_canvas: {paste_dst_y_on_canvas}, width_to_copy: {width_to_copy}, height_to_copy: {height_to_copy}",
            level="DEBUG",
        )

        crop_box_on_zoomed = None
        if width_to_copy > 0 and height_to_copy > 0:
            crop_box_on_zoomed = (
                src_x1_on_zoomed_img,
                src_y1_on_zoomed_img,
                src_x1_on_zoomed_img + width_to_copy,
                src_y1_on_zoomed_img + height_to_copy,
            )
            log(
                f"generate_composite_overlay_image: Calculated crop_box_on_zoomed: {crop_box_on_zoomed}",
                level="DEBUG",
            )
        else:
            log(
                "generate_composite_overlay_image: width_to_copy or height_to_copy is <= 0. crop_box_on_zoomed will be None. Canvas will be black or only show overlays if they ignore crop_box.",
                level="DEBUG",
            )

        # Initialize the final display image for the canvas
        final_display_pil = Image.new(
            "RGB", (canvas_width, canvas_height), constants.COLOR_BLACK_STR
        )
        log(
            f"generate_composite_overlay_image: Initialized final_display_pil (canvas) as RGB black {canvas_width}x{canvas_height}",
            level="DEBUG",
        )

        # 1. Base Image
        if display_state.show_original_image and base_pil_image is not None:
            log(
                f"generate_composite_overlay_image: Step 1 - Base Image. show_original_image is True. Base image provided size: {base_pil_image.size}, mode: {base_pil_image.mode}",
                level="DEBUG",
            )
            if base_pil_image.size != image_size_pil:
                log(
                    f"ImageOverlayProcessor Warning: generate_composite_overlay_image - base_pil_image size {base_pil_image.size} differs from image_size_pil {image_size_pil}. Using base_pil_image content but image_size_pil for zoom calculations.",
                    level="WARNING",
                )

            zoomed_base_image = base_pil_image.resize(
                (safe_new_width, safe_new_height), resample_filter
            )
            log(
                f"generate_composite_overlay_image: Resized base_pil_image to zoomed dimensions: {zoomed_base_image.size}",
                level="DEBUG",
            )
            if crop_box_on_zoomed:
                cropped_visible_part = zoomed_base_image.crop(crop_box_on_zoomed)
                log(
                    f"generate_composite_overlay_image: Cropped visible part of zoomed_base_image. Size: {cropped_visible_part.size}",
                    level="DEBUG",
                )
                final_display_pil.paste(
                    cropped_visible_part, (paste_dst_x_on_canvas, paste_dst_y_on_canvas)
                )
                log(
                    f"generate_composite_overlay_image: Pasted cropped_visible_part onto final_display_pil at ({paste_dst_x_on_canvas}, {paste_dst_y_on_canvas})",
                    level="DEBUG",
                )
            else:
                log(
                    "generate_composite_overlay_image: crop_box_on_zoomed is None. Base image not pasted.",
                    level="DEBUG",
                )
        elif not display_state.show_original_image:
            log(
                "generate_composite_overlay_image: Step 1 - Base Image. show_original_image is False. Skipping base image drawing.",
                level="DEBUG",
            )
        elif base_pil_image is None:
            log(
                "generate_composite_overlay_image: Step 1 - Base Image. show_original_image is True but base_pil_image is None. Skipping base image drawing.",
                level="WARNING",
            )

        # 2. Mask Overlay
        if display_state.show_cell_masks:
            log(
                "generate_composite_overlay_image: Step 2 - Mask Overlay. show_cell_masks is True.",
                level="DEBUG",
            )
            mask_layer_full_res_rgb = self.get_cached_mask_layer_rgb(
                base_pil_image_size=image_size_pil,
                cell_ids_to_draw=ids_to_display,
                show_deselected_masks_only=display_state.show_deselected_masks_only,
            )
            if mask_layer_full_res_rgb:  # Should always return an image
                log(
                    f"generate_composite_overlay_image: Retrieved full_res_mask_rgb from cache/generation. Size: {mask_layer_full_res_rgb.size}, Mode: {mask_layer_full_res_rgb.mode}",
                    level="DEBUG",
                )
                zoomed_mask_layer_rgb = mask_layer_full_res_rgb.resize(
                    (safe_new_width, safe_new_height), overlay_resample_filter
                )
                log(
                    f"generate_composite_overlay_image: Resized full_res_mask_rgb to zoomed_mask_layer_rgb. Size: {zoomed_mask_layer_rgb.size}",
                    level="DEBUG",
                )

                if crop_box_on_zoomed:
                    cropped_mask_layer_for_canvas = zoomed_mask_layer_rgb.crop(
                        crop_box_on_zoomed
                    )
                    log(
                        f"generate_composite_overlay_image: Cropped zoomed_mask_layer_rgb to cropped_mask_layer_for_canvas. Size: {cropped_mask_layer_for_canvas.size}",
                        level="DEBUG",
                    )

                    canvas_mask_layer_rgb = Image.new(
                        "RGB",
                        (canvas_width, canvas_height),
                        (
                            0,
                            0,
                            0,
                        ),  # Black background for the canvas-sized mask fragment
                    )
                    canvas_mask_layer_rgb.paste(
                        cropped_mask_layer_for_canvas,
                        (paste_dst_x_on_canvas, paste_dst_y_on_canvas),
                    )
                    log(
                        f"generate_composite_overlay_image: Pasted cropped_mask_layer_for_canvas onto canvas_mask_layer_rgb at ({paste_dst_x_on_canvas}, {paste_dst_y_on_canvas})",
                        level="DEBUG",
                    )

                    if (
                        display_state.show_original_image and base_pil_image is not None
                    ):  # Check base_pil_image too, in case show_original_image was true but image was None
                        final_display_pil_rgb = final_display_pil.convert(
                            "RGB"
                        )  # Ensure final_display_pil is RGB for blend
                        final_display_pil = Image.blend(
                            final_display_pil_rgb,
                            canvas_mask_layer_rgb,  # This is already RGB
                            alpha=constants.MASK_BLEND_ALPHA,
                        )
                        log(
                            f"generate_composite_overlay_image: Blended canvas_mask_layer_rgb onto final_display_pil (which had base image). Alpha: {constants.MASK_BLEND_ALPHA}",
                            level="DEBUG",
                        )
                    else:
                        # If not showing original, or original was None, the mask layer (colors on black) becomes the base for this area
                        final_display_pil = canvas_mask_layer_rgb
                        log(
                            "generate_composite_overlay_image: Set final_display_pil to canvas_mask_layer_rgb (no base image was shown or available).",
                            level="DEBUG",
                        )
                else:
                    log(
                        "generate_composite_overlay_image: crop_box_on_zoomed is None. Mask layer not applied to canvas.",
                        level="DEBUG",
                    )
            else:
                log(
                    "generate_composite_overlay_image: get_cached_mask_layer_rgb returned None or empty image. Skipping mask overlay.",
                    level="WARNING",
                )
        else:
            log(
                "generate_composite_overlay_image: Step 2 - Mask Overlay. show_cell_masks is False. Skipping mask overlay.",
                level="DEBUG",
            )

        # 3. Boundary Overlay
        if display_state.show_cell_boundaries:
            log(
                "generate_composite_overlay_image: Step 3 - Boundary Overlay. show_cell_boundaries is True.",
                level="DEBUG",
            )
            boundary_L_pil_full_res = self.get_boundary_mask_L_pil(
                image_size=image_size_pil,
                cell_ids_to_draw=ids_to_display,
                show_deselected_state=display_state.show_deselected_masks_only,
            )
            if boundary_L_pil_full_res:  # Should always return an image
                log(
                    f"generate_composite_overlay_image: Retrieved full_res_boundary_L_pil from cache/generation. Size: {boundary_L_pil_full_res.size}, Mode: {boundary_L_pil_full_res.mode}",
                    level="DEBUG",
                )
                zoomed_boundary_L_pil = boundary_L_pil_full_res.resize(
                    (safe_new_width, safe_new_height), overlay_resample_filter
                )
                log(
                    f"generate_composite_overlay_image: Resized full_res_boundary_L_pil to zoomed_boundary_L_pil. Size: {zoomed_boundary_L_pil.size}",
                    level="DEBUG",
                )

                if crop_box_on_zoomed:
                    cropped_boundary_L_for_canvas_paste = zoomed_boundary_L_pil.crop(
                        crop_box_on_zoomed
                    )
                    log(
                        f"generate_composite_overlay_image: Cropped zoomed_boundary_L_pil to cropped_boundary_L_for_canvas_paste. Size: {cropped_boundary_L_for_canvas_paste.size}",
                        level="DEBUG",
                    )

                    boundary_on_canvas_L = Image.new(
                        "L",
                        (canvas_width, canvas_height),
                        0,  # Black background for the L-mode canvas mask
                    )
                    boundary_on_canvas_L.paste(
                        cropped_boundary_L_for_canvas_paste,
                        (paste_dst_x_on_canvas, paste_dst_y_on_canvas),
                    )
                    log(
                        f"generate_composite_overlay_image: Pasted cropped_boundary_L_for_canvas_paste onto boundary_on_canvas_L at ({paste_dst_x_on_canvas}, {paste_dst_y_on_canvas})",
                        level="DEBUG",
                    )

                    chosen_color_tuple = constants.BOUNDARY_COLOR_MAP_PIL.get(
                        display_state.boundary_color_name,
                        constants.BOUNDARY_COLOR_MAP_PIL["Green"],
                    )
                    log(
                        f"generate_composite_overlay_image: Boundary color name: '{display_state.boundary_color_name}', chosen RGB tuple: {chosen_color_tuple}",
                        level="DEBUG",
                    )

                    if quality == "interactive":
                        log(
                            "generate_composite_overlay_image: Boundary drawing method for 'interactive' quality: Alpha compositing.",
                            level="DEBUG",
                        )
                        boundary_on_canvas_rgba = Image.new(
                            "RGBA",
                            boundary_on_canvas_L.size,
                            (0, 0, 0, 0),  # Transparent canvas for RGBA boundary
                        )
                        boundary_color_img_rgba = Image.new(
                            "RGBA",
                            boundary_on_canvas_L.size,
                            (*chosen_color_tuple, 255),  # Solid color with full alpha
                        )
                        boundary_on_canvas_rgba.paste(
                            boundary_color_img_rgba,
                            mask=boundary_on_canvas_L,  # Use L mask to apply color
                        )
                        log(
                            "generate_composite_overlay_image: Created RGBA boundary layer (boundary_on_canvas_rgba) using L mask.",
                            level="DEBUG",
                        )

                        if final_display_pil.mode != "RGBA":
                            log(
                                f"generate_composite_overlay_image: Converting final_display_pil from mode {final_display_pil.mode} to RGBA for alpha_composite.",
                                level="DEBUG",
                            )
                            final_display_pil = final_display_pil.convert("RGBA")

                        final_display_pil = Image.alpha_composite(
                            final_display_pil, boundary_on_canvas_rgba
                        )
                        log(
                            "generate_composite_overlay_image: Alpha composited RGBA boundary layer onto final_display_pil.",
                            level="DEBUG",
                        )

                        # Special handling to convert back to RGB if conditions met (e.g. only boundaries on black)
                        if (
                            final_display_pil.mode == "RGBA"
                            and not display_state.show_cell_masks  # If masks are off
                            and (
                                not display_state.show_original_image
                                or base_pil_image is None
                            )  # And original image is off or was None
                        ):
                            log(
                                "generate_composite_overlay_image: Checking if RGBA result (boundaries on black) can be converted back to RGB.",
                                level="DEBUG",
                            )
                            # This checks if the image is effectively opaque after compositing boundaries on a (potentially) transparent black background.
                            # Create an opaque black background for comparison.
                            opaque_black_check_bg = Image.new(
                                "RGBA", final_display_pil.size, (0, 0, 0, 255)
                            )
                            # Composite the current RGBA image onto this opaque black. If all alpha pixels were already 255 or became 255,
                            # then it means the image has no true transparency left that matters.
                            test_composite = Image.alpha_composite(
                                opaque_black_check_bg, final_display_pil
                            )
                            alpha_min, alpha_max = test_composite.getextrema()[3]
                            if alpha_min == 255 and alpha_max == 255:
                                log(
                                    "generate_composite_overlay_image: All alpha values are 255 after test composite. Converting final_display_pil back to RGB.",
                                    level="DEBUG",
                                )
                                final_display_pil = final_display_pil.convert("RGB")
                            else:
                                log(
                                    f"generate_composite_overlay_image: RGBA result still has varying alpha values (min: {alpha_min}, max: {alpha_max}). Keeping as RGBA.",
                                    level="DEBUG",
                                )

                    else:  # Final quality
                        log(
                            "generate_composite_overlay_image: Boundary drawing method for 'final' quality: NumPy pixel manipulation.",
                            level="DEBUG",
                        )
                        boundary_pixels_to_color_np = np.array(boundary_on_canvas_L) > 0
                        if np.any(boundary_pixels_to_color_np):
                            log(
                                f"generate_composite_overlay_image: NumPy boundary mask (boundary_pixels_to_color_np) has {np.count_nonzero(boundary_pixels_to_color_np)} pixels to color.",
                                level="DEBUG",
                            )
                            if final_display_pil.mode != "RGB":
                                log(
                                    f"generate_composite_overlay_image: Converting final_display_pil from mode {final_display_pil.mode} to RGB for NumPy boundary coloring.",
                                    level="DEBUG",
                                )
                                final_display_pil = final_display_pil.convert("RGB")
                            final_display_np = np.array(final_display_pil)
                            final_display_np[boundary_pixels_to_color_np] = (
                                chosen_color_tuple
                            )
                            final_display_pil = Image.fromarray(final_display_np, "RGB")
                            log(
                                "generate_composite_overlay_image: Applied boundary color to final_display_pil using NumPy array manipulation.",
                                level="DEBUG",
                            )
                        else:
                            log(
                                "generate_composite_overlay_image: boundary_pixels_to_color_np is all False. No boundaries drawn with NumPy method.",
                                level="DEBUG",
                            )
                else:
                    log(
                        "generate_composite_overlay_image: crop_box_on_zoomed is None. Boundary layer not applied to canvas.",
                        level="DEBUG",
                    )
            else:  # boundary_L_pil_full_res is None or empty
                log(
                    "generate_composite_overlay_image: get_boundary_mask_L_pil returned None or empty image. Skipping boundary overlay.",
                    level="WARNING",
                )
        else:
            log(
                "generate_composite_overlay_image: Step 3 - Boundary Overlay. show_cell_boundaries is False. Skipping boundary overlay.",
                level="DEBUG",
            )

        # 4. Number Overlay
        if (
            quality != "interactive"
            and display_state.show_cell_numbers
            and ids_to_display
        ):
            log(
                f"generate_composite_overlay_image: Step 4 - Number Overlay. Quality is '{quality}', show_cell_numbers is True, {len(ids_to_display)} IDs to display.",
                level="DEBUG",
            )
            sorted_cells_positions = self.get_cached_cell_number_positions(
                cell_ids_to_draw=ids_to_display,
                show_deselected_state=display_state.show_deselected_masks_only,
            )
            if sorted_cells_positions:
                log(
                    f"generate_composite_overlay_image: Retrieved {len(sorted_cells_positions)} cell positions for numbering.",
                    level="DEBUG",
                )
                if final_display_pil.mode != "RGB":  # Ensure drawable for ImageDraw
                    log(
                        f"generate_composite_overlay_image: Converting final_display_pil from mode {final_display_pil.mode} to RGB for drawing numbers.",
                        level="DEBUG",
                    )
                    final_display_pil = final_display_pil.convert("RGB")
                draw_on_final_pil = ImageDraw.Draw(final_display_pil)

                text_color_tuple = constants.BOUNDARY_COLOR_MAP_PIL.get(  # Using boundary color map for numbers consistency
                    display_state.boundary_color_name,
                    constants.BOUNDARY_COLOR_MAP_PIL["Green"],
                )
                log(
                    f"generate_composite_overlay_image: Text color for numbers (from boundary color '{display_state.boundary_color_name}'): {text_color_tuple}",
                    level="DEBUG",
                )

                numbers_drawn_count = 0
                for i, cell_data in enumerate(sorted_cells_positions):
                    display_number = str(i + 1)
                    center_orig_x, center_orig_y = (
                        cell_data["center_orig_x"],
                        cell_data["center_orig_y"],
                    )
                    log(
                        f"generate_composite_overlay_image: Processing number '{display_number}' for cell_id {cell_data['id']}. Original center: ({center_orig_x:.2f}, {center_orig_y:.2f})",
                        level="DEBUG",
                    )

                    # Transform center to zoomed coordinates
                    zoomed_center_x = center_orig_x * zoom
                    zoomed_center_y = center_orig_y * zoom

                    # Relative to the cropped portion of the zoomed image
                    rel_zoomed_center_x = zoomed_center_x - src_x1_on_zoomed_img
                    rel_zoomed_center_y = zoomed_center_y - src_y1_on_zoomed_img

                    # Final position on canvas_view_pil (final_display_pil)
                    cv_text_x = rel_zoomed_center_x + paste_dst_x_on_canvas
                    cv_text_y = rel_zoomed_center_y + paste_dst_y_on_canvas

                    font_size_on_canvas = max(
                        constants.CELL_NUMBERING_FONT_SIZE_CANVAS_MIN,
                        int(
                            self.get_dynamic_cell_number_font_size(
                                image_size_pil[0], image_size_pil[1]
                            )
                            * zoom
                        ),
                    )

                    log(
                        f"generate_composite_overlay_image: Number '{display_number}' - Font size on canvas: {font_size_on_canvas} (zoom: {zoom:.2f})",
                        level="DEBUG",
                    )

                    try:
                        current_font = ImageFont.truetype(
                            constants.DEFAULT_FONT, size=font_size_on_canvas
                        )
                        log(
                            f"generate_composite_overlay_image: Loaded font '{constants.DEFAULT_FONT}' for numbers.",
                            level="DEBUG",
                        )
                    except IOError:
                        log(
                            f"generate_composite_overlay_image: Failed to load font '{constants.DEFAULT_FONT}'. Using default PIL font for numbers.",
                            level="WARNING",
                        )
                        current_font = ImageFont.load_default(size=font_size_on_canvas)

                    try:
                        if hasattr(
                            draw_on_final_pil.textbbox, "__call__"
                        ):  # Pillow 9.2.0+
                            bbox = draw_on_final_pil.textbbox(
                                (
                                    cv_text_x,
                                    cv_text_y,
                                ),  # Use calculated canvas center for bbox positioning
                                display_number,
                                font=current_font,
                                anchor="lt",  # Top-left of bbox at (cv_text_x, cv_text_y)
                            )
                            text_width, text_height = (
                                bbox[2] - bbox[0],
                                bbox[3] - bbox[1],
                            )
                            # To center the text at (cv_text_x, cv_text_y)
                            final_text_x, final_text_y = (
                                cv_text_x - text_width / 2,
                                cv_text_y - text_height / 2,
                            )
                            log(
                                f"generate_composite_overlay_image: Number '{display_number}' (Pillow 9.2.0+) - bbox: {bbox}, text_w/h: ({text_width:.2f},{text_height:.2f}), final_text_x/y: ({final_text_x:.2f},{final_text_y:.2f})",
                                level="DEBUG",
                            )
                        else:  # Older Pillow
                            text_width, text_height = draw_on_final_pil.textsize(
                                display_number, font=current_font
                            )
                            final_text_x, final_text_y = (
                                cv_text_x - text_width / 2,
                                cv_text_y - text_height / 2,
                            )
                            log(
                                f"generate_composite_overlay_image: Number '{display_number}' (Older Pillow) - text_w/h: ({text_width:.2f},{text_height:.2f}), final_text_x/y: ({final_text_x:.2f},{final_text_y:.2f})",
                                level="DEBUG",
                            )
                    except AttributeError:  # Fallback if textsize/textbbox is missing
                        log(
                            f"generate_composite_overlay_image: Number '{display_number}' - AttributeError for textbbox/textsize. Using fallback 10x10.",
                            level="WARNING",
                        )
                        text_width, text_height = 10, 10
                        final_text_x, final_text_y = (
                            cv_text_x - text_width / 2,
                            cv_text_y - text_height / 2,
                        )

                    # Check if the center of the text is visible on canvas
                    if 0 <= cv_text_x < canvas_width and 0 <= cv_text_y < canvas_height:
                        draw_on_final_pil.text(
                            (final_text_x, final_text_y),
                            display_number,
                            fill=text_color_tuple,
                            font=current_font,
                        )
                        numbers_drawn_count += 1
                        log(
                            f"generate_composite_overlay_image: Drew number '{display_number}' at canvas coords ({final_text_x:.2f}, {final_text_y:.2f}). Total numbers drawn: {numbers_drawn_count}",
                            level="DEBUG",
                        )
                    else:
                        log(
                            f"generate_composite_overlay_image: Number '{display_number}' center (cv_text_x,y: {cv_text_x:.2f},{cv_text_y:.2f}) is outside canvas ({canvas_width}x{canvas_height}). Skipping draw.",
                            level="DEBUG",
                        )
            else:
                log(
                    "generate_composite_overlay_image: No sorted_cells_positions available. Skipping number drawing.",
                    level="DEBUG",
                )
        elif quality == "interactive":
            log(
                "generate_composite_overlay_image: Step 4 - Number Overlay. Quality is 'interactive'. Skipping number drawing for performance.",
                level="DEBUG",
            )
        elif not display_state.show_cell_numbers:
            log(
                "generate_composite_overlay_image: Step 4 - Number Overlay. show_cell_numbers is False. Skipping number drawing.",
                level="DEBUG",
            )
        elif not ids_to_display:
            log(
                "generate_composite_overlay_image: Step 4 - Number Overlay. ids_to_display is empty. Skipping number drawing.",
                level="DEBUG",
            )

        log(
            f"ImageOverlayProcessor: generate_composite_overlay_image: Exit. Final composite image size: {final_display_pil.size}, mode: {final_display_pil.mode}",
            level="INFO",
        )
        return final_display_pil

    def draw_scale_bar(
        self,
        image_to_draw_on: Image.Image,
        effective_display_zoom: float,
        scale_conversion_obj,
        target_image_width: int,
        target_image_height: int,
        font_size: int = None,
    ) -> Image.Image:
        """
        Draws a scale bar on the given PIL Image.

        Args:
            image_to_draw_on: The PIL.Image object to draw on.
            effective_display_zoom: The zoom level at which this image is effectively being viewed (e.g., model.zoom for screen, or pixel_scale_factor for PDF).
            scale_conversion_obj: The scale conversion object (e.g., from model.image_data.scale_conversion).
            target_image_width: The width of the image_to_draw_on.
            target_image_height: The height of the image_to_draw_on.
            font_size: The font size for the scale bar label.

        Returns:
            The modified PIL.Image with the scale bar drawn.
        """
        # Get the micron length from the model's display_state
        target_microns_for_bar = int(
            self.application_model.display_state.scale_bar_microns
        )
        log(
            f"draw_scale_bar: Entry. Image to draw on size: {image_to_draw_on.size if image_to_draw_on else 'None'} (mode: {image_to_draw_on.mode if image_to_draw_on else 'N/A'}), effective_display_zoom: {effective_display_zoom:.2f}, target_image_width: {target_image_width}, target_image_height: {target_image_height}, target_microns: {target_microns_for_bar}",
            level="DEBUG",
        )
        if image_to_draw_on is None:
            log(
                "ImageOverlayProcessor.draw_scale_bar: No image_to_draw_on provided. Exiting.",
                level="WARNING",
            )
            return None  # Or raise error

        draw = ImageDraw.Draw(image_to_draw_on)
        log("draw_scale_bar: Initialized ImageDraw object.", level="DEBUG")

        # Scale bar properties
        bar_color = constants.SCALE_BAR_COLOR
        base_bar_height = constants.SCALE_BAR_HEIGHT

        if font_size is not None:
            font_size = int(font_size)
            log(
                f"draw_scale_bar: Using provided font_size: {font_size} (PDF export or override)",
                level="DEBUG",
            )
        else:
            font_size = 20
        bar_height_pixels = int(round(base_bar_height))
        log(
            f"draw_scale_bar: Using font_size: {font_size}, fixed bar_height_pixels: {bar_height_pixels} (base: {base_bar_height}). Effective_display_zoom: {effective_display_zoom:.2f} (used for bar length only).",
            level="DEBUG",
        )

        # Clamp values to prevent them from becoming too small or too large
        font_size = max(8, min(50, font_size))
        bar_height_pixels = max(1, min(15, bar_height_pixels))
        log(
            f"draw_scale_bar: Clamped font_size: {font_size}, bar_height_pixels: {bar_height_pixels}",
            level="DEBUG",
        )

        margin_x_pixels = 20  # Margin from the left edge
        margin_y_pixels = 20  # Margin from the bottom edge (from bottom of image to bottom of text+bar block)
        text_color = constants.SCALE_BAR_COLOR
        log(
            f"draw_scale_bar: Margins - X: {margin_x_pixels}, Y: {margin_y_pixels}. Text color: '{text_color}', Bar color: '{bar_color}'",
            level="DEBUG",
        )

        pil_font = None
        try:
            pil_font = ImageFont.truetype(constants.DEFAULT_FONT, font_size)
            log(
                f"draw_scale_bar: Successfully loaded font '{constants.DEFAULT_FONT}' with size {font_size}",
                level="DEBUG",
            )
        except IOError:
            log(
                f"ImageOverlayProcessor: Font '{constants.DEFAULT_FONT}' not found for scale bar (size {font_size}). Attempting to load default PIL font.",
                level="WARNING",
            )
            try:
                pil_font = ImageFont.load_default()
                # For newer Pillow that might take size in load_default itself or need set_size
                if hasattr(ImageFont, "load_default") and callable(
                    ImageFont.load_default
                ):
                    try:
                        pil_font = ImageFont.load_default(font_size)  # Try with size
                        log(
                            f"draw_scale_bar: Loaded default PIL font with size {font_size}.",
                            level="DEBUG",
                        )
                    except TypeError:  # If size argument is not supported
                        pil_font = ImageFont.load_default()
                        log(
                            "draw_scale_bar: Loaded default PIL font (no size arg). Will rely on inherent size or set_size if available.",
                            level="DEBUG",
                        )
                elif hasattr(pil_font, "set_size"):  # Check for older PIL versions
                    pil_font.set_size(font_size)
                    log(
                        f"draw_scale_bar: Set size of default PIL font to {font_size} using set_size().",
                        level="DEBUG",
                    )
                else:
                    log(
                        f"draw_scale_bar: Loaded default PIL font but could not set size to {font_size}. Using its default size.",
                        level="WARNING",
                    )
            except Exception as e_font:
                log(
                    f"ImageOverlayProcessor: Could not load default PIL font for scale bar: {e_font}",
                    level="ERROR",
                )
                # pil_font remains None

        if (
            not scale_conversion_obj
            or not hasattr(scale_conversion_obj, "X")
            or scale_conversion_obj.X == 0
        ):
            log(
                "ImageOverlayProcessor.draw_scale_bar: Scale conversion data (scale_conversion_obj or .X attribute) is not available or invalid (X is 0). Returning original image.",
                level="WARNING",
            )
            return image_to_draw_on  # Return original image if no scale

        log(
            f"draw_scale_bar: Scale conversion object X factor: {scale_conversion_obj.X}",
            level="DEBUG",
        )

        microns_per_pixel_orig = scale_conversion_obj.X

        # Calculate the length of the bar in original image pixels
        bar_length_orig_pixels = target_microns_for_bar / microns_per_pixel_orig
        log(
            f"draw_scale_bar: Calculated bar_length_orig_pixels: {bar_length_orig_pixels:.2f} (target_microns {target_microns_for_bar} / microns_per_pixel_orig {microns_per_pixel_orig})",
            level="DEBUG",
        )

        # Calculate the length of the bar in the target image's pixels (could be screen or PDF image)
        bar_length_on_target_image_pixels = (
            bar_length_orig_pixels * effective_display_zoom
        )
        log(
            f"draw_scale_bar: Calculated bar_length_on_target_image_pixels: {bar_length_on_target_image_pixels:.2f} (bar_length_orig_pixels {bar_length_orig_pixels:.2f} * effective_display_zoom {effective_display_zoom:.2f})",
            level="DEBUG",
        )

        if bar_length_on_target_image_pixels < 1:
            log(
                f"ImageOverlayProcessor.draw_scale_bar: Scale bar for {target_microns_for_bar}µm is too small to be visible ({bar_length_on_target_image_pixels:.2f}px). Returning original image.",
                level="INFO",
            )
            return image_to_draw_on

        actual_bar_y = target_image_height - margin_y_pixels - bar_height_pixels

        bar_x_start = margin_x_pixels
        bar_x_end = bar_x_start + bar_length_on_target_image_pixels
        log(
            f"draw_scale_bar: Bar coordinates - X_start: {bar_x_start}, X_end: {bar_x_end:.2f}, Y (actual_bar_y): {actual_bar_y}",
            level="DEBUG",
        )

        # Ensure bar does not exceed target image width
        if bar_x_end > target_image_width - margin_x_pixels:
            bar_x_end_orig = bar_x_end
            bar_x_end = target_image_width - margin_x_pixels
            log(
                f"draw_scale_bar: bar_x_end ({bar_x_end_orig:.2f}) exceeded target_image_width ({target_image_width}) minus margin. Adjusted bar_x_end to: {bar_x_end}",
                level="DEBUG",
            )

        # Draw the scale bar
        draw.line(
            [(bar_x_start, actual_bar_y), (bar_x_end, actual_bar_y)],
            fill=bar_color,
            width=bar_height_pixels,
        )
        log(
            f"draw_scale_bar: Drew scale bar line from ({bar_x_start},{actual_bar_y}) to ({bar_x_end:.2f},{actual_bar_y}) with width {bar_height_pixels}",
            level="DEBUG",
        )

        # Draw the label
        label_text = f"{target_microns_for_bar} \u00b5m"
        text_x = bar_x_start
        text_y = actual_bar_y - font_size - 5  # Position text above the bar (5px gap)
        log(
            f"draw_scale_bar: Label text: '{label_text}'. Calculated text_x: {text_x}, text_y: {text_y} (actual_bar_y {actual_bar_y} - font_size {font_size} - 5px gap)",
            level="DEBUG",
        )

        if pil_font:
            try:
                draw.text((text_x, text_y), label_text, fill=text_color, font=pil_font)
                log(
                    f"draw_scale_bar: Drew label text '{label_text}' at ({text_x},{text_y}) using loaded font.",
                    level="DEBUG",
                )
            except Exception as e_text_draw:
                log(
                    f"ImageOverlayProcessor.draw_scale_bar: Error drawing text with loaded font: {e_text_draw}. Attempting fallback draw.",
                    level="ERROR",
                )
                try:  # Fallback to drawing without specific font object if the loaded one failed for some reason
                    draw.text((text_x, text_y), label_text, fill=text_color)
                    log(
                        f"draw_scale_bar: Drew label text '{label_text}' at ({text_x},{text_y}) using fallback draw (no font object explicitly passed to text()).",
                        level="DEBUG",
                    )
                except Exception as e_text_draw_fallback:
                    log(
                        f"ImageOverlayProcessor.draw_scale_bar: Fallback text drawing also failed: {e_text_draw_fallback}",
                        level="ERROR",
                    )
        else:  # Fallback if font loading completely failed (pil_font is None)
            log(
                f"draw_scale_bar: pil_font is None. Attempting to draw label text '{label_text}' at ({text_x},{text_y}) using default draw handling for text.",
                level="WARNING",
            )
            try:
                draw.text((text_x, text_y), label_text, fill=text_color)
                log(
                    f"draw_scale_bar: Drew label text '{label_text}' at ({text_x},{text_y}) using default draw (pil_font was None).",
                    level="DEBUG",
                )
            except Exception as e_text_draw_no_font:
                log(
                    f"ImageOverlayProcessor.draw_scale_bar: Text drawing failed even with default handling (pil_font was None): {e_text_draw_no_font}",
                    level="ERROR",
                )

        log(
            f"ImageOverlayProcessor.draw_scale_bar: Exit. Scale bar drawn for {target_microns_for_bar}µm, final bar length on image: {bar_x_end - bar_x_start:.2f}px.",
            level="INFO",
        )
        return image_to_draw_on

    def get_final_composite_image(self, quality: str = "final") -> Image.Image | None:
        """
        Main method to get the fully composited image based on the current model state.
        """
        log(f"get_final_composite_image: Entry. Quality: '{quality}'", level="DEBUG")
        # 1. Get the base image (potentially processed)
        if quality == "final":
            log(
                "get_final_composite_image: Quality is 'final'. Attempting to get processed image for display.",
                level="DEBUG",
            )
            base_image_pil = self.application_model.get_processed_image_for_display()
            if base_image_pil is None:
                log(
                    "get_final_composite_image: Processed image is None. Falling back to original_image.",
                    level="DEBUG",
                )
                base_image_pil = self.application_model.image_data.original_image
        else:  # For "interactive" or other, just use original (or adapt as needed)
            log(
                f"get_final_composite_image: Quality is '{quality}'. Using original_image.",
                level="DEBUG",
            )
            base_image_pil = self.application_model.image_data.original_image

        if base_image_pil is None:
            log(
                "ImageOverlayProcessor: get_final_composite_image - No base image (original or processed) available. Returning None.",
                level="WARNING",
            )
            return None

        log(
            f"get_final_composite_image: Base image selected. Size: {base_image_pil.size}, Mode: {base_image_pil.mode}",
            level="DEBUG",
        )

        # Create a working copy
        composite_image_pil = base_image_pil.copy()
        log(
            f"get_final_composite_image: Copied base_image_pil to composite_image_pil. Size: {composite_image_pil.size}, Mode: {composite_image_pil.mode}",
            level="DEBUG",
        )

        # 2. Determine which cell IDs to draw for overlays
        all_mask_ids = (
            set(np.unique(self.application_model.image_data.mask_array)) - {0}
            if self.application_model.image_data.mask_array is not None
            else set()
        )
        included_cell_ids = self.application_model.image_data.included_cells
        log(
            f"get_final_composite_image: Determining cell IDs for overlays. All mask IDs (excluding 0): {len(all_mask_ids)}, Included (selected) cell IDs: {len(included_cell_ids)}",
            level="DEBUG",
        )

        cell_ids_to_draw_for_view = set()
        if self.application_model.display_state.show_deselected_masks_only:
            # Show only cells that are in all_mask_ids but NOT in included_cell_ids
            cell_ids_to_draw_for_view = all_mask_ids - included_cell_ids
            log(
                f"get_final_composite_image: show_deselected_masks_only is True. cell_ids_to_draw_for_view (deselected) count: {len(cell_ids_to_draw_for_view)}",
                level="DEBUG",
            )
        else:
            # Default: Show only the included (selected) cells
            cell_ids_to_draw_for_view = included_cell_ids
            log(
                f"get_final_composite_image: show_deselected_masks_only is False. cell_ids_to_draw_for_view (selected/included) count: {len(cell_ids_to_draw_for_view)}",
                level="DEBUG",
            )

        if not self.application_model.image_data.mask_array_exists():
            # If no masks, no overlays related to cells can be drawn, but base image can still be returned.
            log(
                "ImageOverlayProcessor: get_final_composite_image - No mask array exists in image_data. Returning current composite_image_pil (base image copy).",
                level="INFO",
            )
            return composite_image_pil

        # 3. Apply Masks
        if self.application_model.display_state.show_cell_masks:
            log(
                f"get_final_composite_image: Applying masks. Number of cell_ids_to_draw_for_view: {len(cell_ids_to_draw_for_view)}",
                level="DEBUG",
            )
            mask_layer_rgb = self.get_cached_mask_layer_rgb(
                base_pil_image_size=composite_image_pil.size,
                cell_ids_to_draw=cell_ids_to_draw_for_view,
                show_deselected_masks_only=self.application_model.display_state.show_deselected_masks_only,
            )
            if mask_layer_rgb:
                log(
                    f"get_final_composite_image: Retrieved mask_layer_rgb (size: {mask_layer_rgb.size}, mode: {mask_layer_rgb.mode}). Blending with composite_image_pil.",
                    level="DEBUG",
                )
                composite_image_pil = self.blend_image_with_mask_layer(
                    composite_image_pil, mask_layer_rgb, constants.MASK_BLEND_ALPHA
                )
                log(
                    f"get_final_composite_image: After blending masks, composite_image_pil size: {composite_image_pil.size}, mode: {composite_image_pil.mode}",
                    level="DEBUG",
                )
            else:
                log(
                    "get_final_composite_image: mask_layer_rgb was None (or empty). Masks not blended.",
                    level="WARNING",
                )
        else:
            log(
                "get_final_composite_image: display_state.show_cell_masks is False. Skipping mask application.",
                level="DEBUG",
            )

        # 4. Apply Boundaries
        if self.application_model.display_state.show_cell_boundaries:
            log(
                f"get_final_composite_image: Applying boundaries. Number of cell_ids_to_draw_for_view: {len(cell_ids_to_draw_for_view)}",
                level="DEBUG",
            )
            if composite_image_pil:  # Explicit check though it should be an image
                composite_image_pil = self.draw_boundaries_on_pil(
                    composite_image_pil, cell_ids_to_draw_for_view
                )
                log(
                    f"get_final_composite_image: After applying boundaries, composite_image_pil size: {composite_image_pil.size}, mode: {composite_image_pil.mode}",
                    level="DEBUG",
                )
            else:
                log(
                    "get_final_composite_image: composite_image_pil was None before drawing boundaries. This is unexpected if base image was present. Boundaries not drawn.",
                    level="WARNING",
                )
        else:
            log(
                "get_final_composite_image: display_state.show_cell_boundaries is False. Skipping boundary application.",
                level="DEBUG",
            )

        # 5. Apply Numbers
        if self.application_model.display_state.show_cell_numbers:
            log(
                f"get_final_composite_image: Applying numbers. Number of cell_ids_to_draw_for_view: {len(cell_ids_to_draw_for_view)}",
                level="DEBUG",
            )
            if composite_image_pil:  # Explicit check
                composite_image_pil = self.draw_numbers_on_pil(
                    composite_image_pil,
                    cell_ids_to_draw_for_view,
                    font_size=constants.CELL_NUMBERING_FONT_SIZE_ORIG_IMG,  # Use a full-res appropriate font size
                )
                log(
                    f"get_final_composite_image: After applying numbers, composite_image_pil size: {composite_image_pil.size}, mode: {composite_image_pil.mode}",
                    level="DEBUG",
                )
            else:
                log(
                    "get_final_composite_image: composite_image_pil was None before drawing numbers. This is unexpected. Numbers not drawn.",
                    level="WARNING",
                )
        else:
            log(
                "get_final_composite_image: display_state.show_cell_numbers is False. Skipping number application.",
                level="DEBUG",
            )

        log(
            f"ImageOverlayProcessor: get_final_composite_image: Exit. Returning final composite image. Size: {composite_image_pil.size if composite_image_pil else 'None'}, Mode: {composite_image_pil.mode if composite_image_pil else 'N/A'}",
            level="INFO",
        )
        return composite_image_pil

    def draw_diameter_aid(
        self,
        image_to_draw_on: Image.Image,
        effective_display_zoom: float,
        diameter_pixels: int,
        target_image_width: int,
        target_image_height: int,
    ) -> Image.Image:
        if not all(
            [
                image_to_draw_on,
                effective_display_zoom > 0,
                diameter_pixels > 0,
                target_image_width > 0,
                target_image_height > 0,
            ]
        ):
            return image_to_draw_on

        draw = ImageDraw.Draw(image_to_draw_on)
        radius_on_canvas = (diameter_pixels * effective_display_zoom) / 2
        center_x = target_image_width / 2
        center_y = target_image_height / 2

        box = [
            center_x - radius_on_canvas,
            center_y - radius_on_canvas,
            center_x + radius_on_canvas,
            center_y + radius_on_canvas,
        ]

        # Draw a dashed circle
        outline_color = constants.DIAMETER_AID_COLOR
        for i in range(0, 360, 15):
            draw.arc(
                box,
                start=i,
                end=i + 10,
                fill=outline_color,
                width=constants.DIAMETER_AID_WIDTH,
            )

        return image_to_draw_on
