import numpy as np
import scipy.ndimage as ndi
from cellpose import models
from PIL import Image, ImageEnhance

from .. import constants
from ..utils.debug_logger import log


def calculate_exact_boundaries_from_mask(
    mask_array: np.ndarray | None,
) -> np.ndarray | None:
    if mask_array is None or mask_array.size == 0:
        log(
            "calculate_exact_boundaries_from_mask: mask_array is None or empty, returning None."
        )  # Can be noisy
        return None
    if mask_array.ndim != 2:
        log(
            f"Warning: calculate_exact_boundaries_from_mask received a mask with unexpected dimensions: {mask_array.shape}. Returning None.",
            level="WARNING",
        )
        return None
    log(
        f"calculate_exact_boundaries_from_mask: Processing mask of shape {mask_array.shape}"
    )  # Can be noisy

    all_boundaries = np.zeros_like(mask_array, dtype=bool)
    unique_ids = np.unique(mask_array)
    struct = ndi.generate_binary_structure(mask_array.ndim, 1)  # 4-connectivity

    for cell_id in unique_ids:
        if cell_id == 0:  # Skip background
            continue

        single_cell_mask = mask_array == cell_id
        eroded_single_cell = ndi.binary_erosion(
            single_cell_mask, structure=struct, border_value=0
        )
        single_cell_boundary = single_cell_mask ^ eroded_single_cell
        all_boundaries |= single_cell_boundary
    return all_boundaries


def run_cellpose_segmentation(
    image_np: np.ndarray, diameter: float | None
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Runs Cellpose segmentation on the input image.

    Args:
        image_np: NumPy array of the image (should be 2D grayscale).
        diameter: Cell diameter for Cellpose.

    Returns:
        A tuple (masks, flows, styles). Masks can be None if segmentation fails.
    """
    try:
        # Determine if GPU is available/preferred (this could be a setting)
        use_gpu = True  # Placeholder - ideally from config or constants
        log(
            f"run_cellpose_segmentation: Initializing CellposeModel (GPU: {use_gpu}). Diameter: {diameter}"
        )
        model = models.CellposeModel(gpu=use_gpu)

        log(
            f"run_cellpose_segmentation: Evaluating image_np of shape {image_np.shape} with diameter {diameter}."
        )
        masks, flows, styles = model.eval(
            image_np,
            diameter=diameter,
        )
        if masks is not None:
            log(
                f"run_cellpose_segmentation: Cellpose evaluation successful. Found {len(np.unique(masks)) - 1 if masks is not None else 0} masks."
            )
        else:
            log(
                "run_cellpose_segmentation: Cellpose evaluation returned no masks.",
                level="WARNING",
            )
        return masks, flows, styles
    except Exception as e:
        log(f"Error during Cellpose segmentation: {e}", level="ERROR")
        return None, None, None


def build_segmentation_input_image_from_model(
    application_model,
    apply_image_adjustments=constants.APPLY_IMAGE_ADJUSTMENTS_BEFORE_SEGMENTATION,
):
    """
    Builds a grayscale PIL image for segmentation based on the current UI selections (channels, z-slice, etc.) in the application model.
    Handles both multi-layer (AICS) and single-layer images.
    If apply_image_adjustments is True, applies brightness and contrast from the model's display_state.
    """
    if (
        application_model.image_data.aics_image_obj
        and application_model.image_data.image_dims
    ):
        aics_img = application_model.image_data.aics_image_obj
        dims = application_model.image_data.image_dims
        z_method = application_model.display_state.z_processing_method
        current_z_idx = application_model.display_state.current_z_slice_index
        target_yx_shape = (dims.Y, dims.X)
        composite_rgb_float_array = np.zeros((*target_yx_shape, 3), dtype=np.float32)
        processed_any_channel = False
        for channel_config in application_model.display_state.display_channel_configs:
            source_channel_idx = channel_config.get("source_idx")
            is_active = channel_config.get("active_in_composite", False)
            display_color_0_255 = channel_config.get("color_tuple_0_255", (0, 0, 0))
            if (
                source_channel_idx is not None
                and is_active
                and 0 <= source_channel_idx < dims.C
            ):
                try:
                    channel_data_z_stack = aics_img.get_image_data(
                        "ZYX", C=source_channel_idx, T=0, S=0
                    )
                    processed_yx_plane = None
                    if dims.Z > 1:
                        if z_method == "max_project":
                            processed_yx_plane = np.max(channel_data_z_stack, axis=0)
                        elif z_method == "mean_project":
                            processed_yx_plane = np.mean(channel_data_z_stack, axis=0)
                        else:  # Slice (default or explicit)
                            target_z = max(0, min(current_z_idx, dims.Z - 1))
                            processed_yx_plane = channel_data_z_stack[target_z, :, :]
                    elif dims.Z == 1:
                        processed_yx_plane = channel_data_z_stack[0, :, :]
                    else:
                        if channel_data_z_stack.ndim == 2:
                            processed_yx_plane = channel_data_z_stack
                        else:
                            continue
                    if processed_yx_plane is not None:
                        p_min, p_max = (
                            float(np.min(processed_yx_plane)),
                            float(np.max(processed_yx_plane)),
                        )
                        if p_max > p_min:
                            normalized_yx_plane = (
                                processed_yx_plane.astype(np.float32) - p_min
                            ) / (p_max - p_min)
                        elif p_max == p_min and p_max > 0:
                            normalized_yx_plane = np.ones_like(
                                processed_yx_plane, dtype=np.float32
                            )
                        else:
                            normalized_yx_plane = np.zeros_like(
                                processed_yx_plane, dtype=np.float32
                            )
                        composite_rgb_float_array[:, :, 0] += normalized_yx_plane * (
                            display_color_0_255[0] / 255.0
                        )
                        composite_rgb_float_array[:, :, 1] += normalized_yx_plane * (
                            display_color_0_255[1] / 255.0
                        )
                        composite_rgb_float_array[:, :, 2] += normalized_yx_plane * (
                            display_color_0_255[2] / 255.0
                        )
                        processed_any_channel = True
                except Exception as e:
                    log(
                        f"Error processing source channel {source_channel_idx} for segmentation: {e}",
                        level="ERROR",
                    )
                    continue
        if not processed_any_channel:
            final_uint8_array = np.zeros((*target_yx_shape, 3), dtype=np.uint8)
        else:
            composite_rgb_float_array = np.clip(composite_rgb_float_array, 0.0, 1.0)
            final_uint8_array = (composite_rgb_float_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(final_uint8_array, mode="RGB")
        if apply_image_adjustments:
            brightness = application_model.display_state.brightness
            contrast = application_model.display_state.contrast
            if brightness != 1.0:
                pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness)
            if contrast != 1.0:
                pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast)
        return pil_img.convert("L")
    else:
        pil_img = application_model.image_data.original_image
        if apply_image_adjustments:
            brightness = application_model.display_state.brightness
            contrast = application_model.display_state.contrast
            if brightness != 1.0:
                pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness)
            if contrast != 1.0:
                pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast)
        return pil_img.convert("L")
