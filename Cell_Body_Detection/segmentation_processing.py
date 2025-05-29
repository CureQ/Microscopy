import numpy as np
import scipy.ndimage as ndi
from cellpose import models


def calculate_exact_boundaries_from_mask(
    mask_array: np.ndarray | None,
) -> np.ndarray | None:
    if mask_array is None or mask_array.size == 0:
        print(
            "calculate_exact_boundaries_from_mask: mask_array is None or empty, returning None."
        )  # Can be noisy
        return None
    if mask_array.ndim != 2:
        print(
            f"Warning: calculate_exact_boundaries_from_mask received a mask with unexpected dimensions: {mask_array.shape}. Returning None."
        )
        return None
    print(
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
        print(
            f"run_cellpose_segmentation: Initializing CellposeModel (GPU: {use_gpu}). Diameter: {diameter}"
        )
        model = models.CellposeModel(
            gpu=use_gpu
        )  # Consider making gpu configurable via constants or app settings
        # Cellpose expects a 2D image or a list of 2D images.

        print(
            f"run_cellpose_segmentation: Evaluating image_np of shape {image_np.shape} with diameter {diameter}."
        )
        masks, flows, styles = model.eval(
            image_np,
            diameter=diameter,
        )
        if masks is not None:
            print(
                f"run_cellpose_segmentation: Cellpose evaluation successful. Found {len(np.unique(masks)) - 1 if masks is not None else 0} masks."
            )
        else:
            print("run_cellpose_segmentation: Cellpose evaluation returned no masks.")
        return masks, flows, styles
    except Exception as e:
        print(f"Error during Cellpose segmentation: {e}")
        return None, None, None
