import numpy as np
from aicsimageio import AICSImage as AICSImageType
from PIL import Image

from ..processing.segmentation_processing import calculate_exact_boundaries_from_mask
from ..utils.debug_logger import log


class ImageDataModel:
    """
    Holds all data related to the core image and its segmentation.
    """

    def __init__(self):
        self.original_image: Image.Image | None = (
            None  # This will be the CURRENTLY PROCESSED 2D VIEW
        )
        self.aics_image_obj: AICSImageType | None = (
            None  # Stores the loaded AICSImage object
        )
        self.image_dims = None  # Stores AICSImage.dims or similar structure
        self.mask_array: np.ndarray | None = None
        self.exact_boundaries: np.ndarray | None = None
        self.included_cells: set[int] = set()
        self.user_drawn_cell_ids: set[int] = set()
        self.scale_conversion = None  # Placeholder for future scaling logic

    def mask_array_exists(self) -> bool:
        """Checks if the mask_array is not None and contains data."""
        return self.mask_array is not None and self.mask_array.size > 0

    def _update_exact_boundaries(self):
        self.exact_boundaries = calculate_exact_boundaries_from_mask(self.mask_array)
        log(
            f"ImageDataModel: Exact boundaries updated. Result is None: {self.exact_boundaries is None}",
            level="DEBUG",
        )

    def reset(self):
        log("ImageDataModel: Resetting data.", level="INFO")
        self.original_image = None
        self.aics_image_obj = None
        self.image_dims = None
        self.mask_array = None
        self.exact_boundaries = None
        self.included_cells = set()
        self.user_drawn_cell_ids.clear()
        self._update_exact_boundaries()
        self.scale_conversion = None  # Reset scale conversion if used

    def set_image_data(self, original_image: Image.Image | None):
        self.original_image = original_image

    def set_aics_image_obj(self, aics_obj: AICSImageType | None, dims=None):
        """Stores the AICSImage object and its dimensions."""
        self.aics_image_obj = aics_obj
        self.image_dims = dims
        log(
            f"ImageDataModel: AICS object {'set' if aics_obj else 'cleared'}. Dims: {dims}",
            level="DEBUG",
        )

    def set_scale_conversion(self, scale_conversion=None):
        """
        Sets the scale conversion factor for the image.
        This can be used to convert pixel coordinates to real-world units.
        """
        self.scale_conversion = scale_conversion
        log(
            f"ImageDataModel: Scale conversion set to {scale_conversion}", level="DEBUG"
        )

    def set_segmentation_result(self, mask_array: np.ndarray):
        self.mask_array = mask_array
        all_current_mask_ids = set()
        if mask_array is not None and mask_array.size > 0:
            all_current_mask_ids = set(np.unique(mask_array)) - {0}
            self.included_cells = all_current_mask_ids.copy()
        else:
            self.included_cells = set()

        # Reconcile user_drawn_cell_ids: only keep those that still exist in the new mask_array
        self.user_drawn_cell_ids &= all_current_mask_ids
        self._update_exact_boundaries()
        log(
            f"ImageDataModel: Segmentation result set. Mask shape: {mask_array.shape if mask_array is not None else 'None'}. Unique IDs in mask: {len(all_current_mask_ids) if all_current_mask_ids else 0}. Included cells: {len(self.included_cells)}",
            level="INFO",
        )

    def toggle_cell_inclusion(self, cell_id: int):
        if cell_id in self.included_cells:
            self.included_cells.remove(cell_id)
            log(
                f"ImageDataModel: Cell ID {cell_id} removed from included_cells.",
                level="DEBUG",
            )
        else:
            self.included_cells.add(cell_id)
            log(
                f"ImageDataModel: Cell ID {cell_id} added to included_cells.",
                level="DEBUG",
            )

    def add_user_drawn_cell(self, cell_id: int):
        self.user_drawn_cell_ids.add(cell_id)
        self.included_cells.add(cell_id)  # Ensure drawn cells are also included
        log(
            f"ImageDataModel: User-drawn cell ID {cell_id} added and included.",
            level="DEBUG",
        )

    def get_snapshot_data(self) -> dict:
        log("ImageDataModel: Getting snapshot data.", level="DEBUG")
        return {
            "mask_array": self.mask_array.copy()
            if self.mask_array is not None
            else None,
            "exact_boundaries": self.exact_boundaries.copy()
            if self.exact_boundaries is not None
            else None,
            "included_cells": self.included_cells.copy(),
            "user_drawn_cell_ids": self.user_drawn_cell_ids.copy(),
        }

    def restore_from_snapshot(self, snapshot_data: dict):
        self.mask_array = (
            snapshot_data["mask_array"].copy()
            if snapshot_data["mask_array"] is not None
            else None
        )
        log(
            f"ImageDataModel: Mask array restored from snapshot. Is None: {self.mask_array is None}",
            level="DEBUG",
        )

        # Restore exact_boundaries if present in snapshot, otherwise calculate
        exact_boundaries_from_snapshot = snapshot_data.get("exact_boundaries")
        if exact_boundaries_from_snapshot is not None:
            self.exact_boundaries = exact_boundaries_from_snapshot.copy()
        elif (
            self.mask_array is not None
        ):  # If not in snapshot but mask exists, recalculate
            self._update_exact_boundaries()
        else:  # No boundaries in snapshot and no mask
            self.exact_boundaries = None

        self.included_cells = snapshot_data["included_cells"].copy()
        self.user_drawn_cell_ids = snapshot_data.get(
            "user_drawn_cell_ids", set()
        ).copy()
        log("ImageDataModel: State restored from snapshot.", level="INFO")
