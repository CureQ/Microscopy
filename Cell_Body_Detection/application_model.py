import numpy as np
from PIL import Image

from .segmentation_processing import calculate_exact_boundaries_from_mask


class ImageDataModel:
    """
    Holds all data related to the core image and its segmentation.
    (Formerly parts of ImageViewModel)
    """

    def __init__(self):
        self.original_image: Image.Image | None = None
        self.mask_array: np.ndarray | None = None
        self.exact_boundaries: np.ndarray | None = None
        self.included_cells: set[int] = set()
        self.user_drawn_cell_ids: set[int] = set()

    def _update_exact_boundaries(self):
        self.exact_boundaries = calculate_exact_boundaries_from_mask(self.mask_array)

    def reset(self):
        self.original_image = None
        self.mask_array = None
        self.included_cells = set()
        self.user_drawn_cell_ids.clear()
        self._update_exact_boundaries()

    def set_image_data(self, original_image: Image.Image):
        self.original_image = original_image

    def set_segmentation_result(self, mask_array: np.ndarray):
        self.mask_array = mask_array
        all_current_mask_ids = set()
        if mask_array is not None and mask_array.size > 0:
            all_current_mask_ids = set(np.unique(mask_array)) - {0}
            # By default, include all found by segmentation, unless they were user-drawn
            # This behavior might need refinement: what if user drew, then segmented?
            # For now, let's assume new segmentation means new set of included cells.
            self.included_cells = all_current_mask_ids.copy()
        else:
            self.included_cells = set()

        # Reconcile user_drawn_cell_ids: only keep those that still exist in the new mask_array
        self.user_drawn_cell_ids &= all_current_mask_ids
        self._update_exact_boundaries()

    def toggle_cell_inclusion(self, cell_id: int):
        if cell_id in self.included_cells:
            self.included_cells.remove(cell_id)
        else:
            self.included_cells.add(cell_id)

    def add_user_drawn_cell(self, cell_id: int):
        self.user_drawn_cell_ids.add(cell_id)
        self.included_cells.add(cell_id)  # Ensure drawn cells are also included

    def get_snapshot_data(self) -> dict:
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


class PanZoomModel:
    """
    Holds the state for panning and zooming.
    (Formerly in _view_models_and_renderer.py)
    """

    def __init__(self):
        self.zoom_level: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self.min_zoom_to_fit: float = 1.0

    def _calculate_min_zoom_to_fit(
        self, canvas_width: int, canvas_height: int, img_width: int, img_height: int
    ) -> float:
        if img_width == 0 or img_height == 0 or canvas_width == 0 or canvas_height == 0:
            return 1.0
        zoom_if_fit_width = canvas_width / img_width
        zoom_if_fit_height = canvas_height / img_height
        return min(zoom_if_fit_width, zoom_if_fit_height, 1.0)

    def reset_for_new_image(
        self,
        canvas_width: int = 0,
        canvas_height: int = 0,
        img_width: int = 0,
        img_height: int = 0,
    ):
        if img_width > 0 and img_height > 0 and canvas_width > 1 and canvas_height > 1:
            self.min_zoom_to_fit = self._calculate_min_zoom_to_fit(
                canvas_width, canvas_height, img_width, img_height
            )
            self.zoom_level = self.min_zoom_to_fit
            zoomed_img_width = img_width * self.zoom_level
            zoomed_img_height = img_height * self.zoom_level
            self.pan_x = (canvas_width - zoomed_img_width) / 2.0
            self.pan_y = (canvas_height - zoomed_img_height) / 2.0
        else:
            self.min_zoom_to_fit = 1.0
            self.zoom_level = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0

    def get_params(self) -> tuple[float, float, float]:
        return self.zoom_level, self.pan_x, self.pan_y

    def set_zoom_level(
        self, zoom: float, min_zoom: float | None = None, max_zoom: float | None = None
    ):
        new_zoom = zoom
        if min_zoom is not None:
            new_zoom = max(min_zoom, new_zoom)
        if max_zoom is not None:
            new_zoom = min(max_zoom, new_zoom)
        self.zoom_level = new_zoom

    def set_pan(self, pan_x: float, pan_y: float):
        self.pan_x = pan_x
        self.pan_y = pan_y


class DisplayStateModel:
    """
    Holds the state for image display adjustments and view options.
    """

    def __init__(self):
        # Image adjustments
        self.brightness: float = 1.0
        self.contrast: float = 1.0
        self.colormap_name: str | None = None  # e.g., "viridis", "gray", None

        # View options (formerly in cell_body_frame)
        self.show_original_image: bool = True
        self.show_cell_masks: bool = True
        self.show_cell_boundaries: bool = False
        self.boundary_color_name: str = "Green"  # Default from constants
        self.show_deselected_masks_only: bool = False  # formerly show_only_deselected
        self.show_cell_numbers: bool = False

        # PDF Export Options (formerly in cell_body_frame)
        self.pdf_opt_masks_only: bool = False
        self.pdf_opt_boundaries_only: bool = True
        self.pdf_opt_numbers_only: bool = False
        self.pdf_opt_masks_boundaries: bool = False
        self.pdf_opt_masks_numbers: bool = False
        self.pdf_opt_boundaries_numbers: bool = False
        self.pdf_opt_masks_boundaries_numbers: bool = False
        # self.pdf_opt_include_original_image: bool = True # This was implicitly handled, maybe add explicitly

    def reset_image_adjustments(self):
        self.brightness = 1.0
        self.contrast = 1.0
        self.colormap_name = None


class ApplicationModel:
    """
    Central model for the application. Holds all application state
    and provides methods to modify and access it.
    Implements an observer pattern to notify views of changes.
    """

    def __init__(self):
        self.image_data = ImageDataModel()
        self.pan_zoom_state = PanZoomModel()
        self.display_state = DisplayStateModel()

        # File related state
        self.current_file_path: str | None = None
        self.base_filename: str | None = None
        self.segmentation_diameter: str = "100"  # Default from constants

        # History (Undo/Redo)
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []

        self._observers: list[callable] = []

    def subscribe(self, observer_callback: callable):
        if observer_callback not in self._observers:
            self._observers.append(observer_callback)

    def unsubscribe(self, observer_callback: callable):
        if observer_callback in self._observers:
            self._observers.remove(observer_callback)

    def notify_observers(self, change_type: str | None = None):
        """
        Notify all subscribed observers about a change.
        change_type can be used by observers to optimize updates.
        """
        print(f"Model notifying observers of change: {change_type}")  # For debugging
        for callback in self._observers:
            try:
                print(
                    f"Notifying observer: {callback.__qualname__ if hasattr(callback, '__qualname__') else callback} for change_type: {change_type}"
                )
                callback(change_type)
            except Exception as e:
                print(f"Error in observer callback: {callback} with error: {e}")

    # --- Image Data Methods ---
    def load_new_image(
        self, pil_image: Image.Image, file_path: str, base_filename: str
    ):
        self.image_data.reset()
        self.image_data.set_image_data(pil_image)
        self.current_file_path = file_path
        self.base_filename = base_filename
        self.reset_history()  # Resets history and records initial state
        print(f"Image loaded: {base_filename}, path: {file_path}")
        self.display_state.reset_image_adjustments()  # Reset brightness/contrast for new image
        self.notify_observers(change_type="image_loaded")

    def set_segmentation_result(self, mask_array: np.ndarray):
        self.image_data.set_segmentation_result(mask_array)
        self.record_state()  # Record after segmentation
        self.notify_observers(change_type="segmentation_updated")

    def toggle_cell_inclusion(self, cell_id: int):
        self.image_data.toggle_cell_inclusion(cell_id)
        self.record_state()
        self.notify_observers(change_type="cell_selection_changed")

    def add_user_drawn_cell_mask(
        self, mask_array_with_new_cell: np.ndarray, new_cell_id: int
    ):
        """
        More comprehensive update after drawing a new cell.
        The mask_array itself is updated by the drawing controller, then passed here.
        """
        self.image_data.mask_array = (
            mask_array_with_new_cell  # Assume controller created it correctly
        )
        self.image_data.add_user_drawn_cell(
            new_cell_id
        )  # Marks as user-drawn and adds to included
        self.image_data._update_exact_boundaries()  # EXPLICIT CALL after direct mask_array update
        self.record_state()
        self.notify_observers(change_type="mask_updated_user_drawn")

    # --- Pan/Zoom Methods ---
    def update_pan_zoom(
        self,
        zoom: float,
        pan_x: float,
        pan_y: float,
        min_zoom_override: float | None = None,
        max_zoom_override: float | None = None,
    ):
        effective_min_zoom = (
            min_zoom_override
            if min_zoom_override is not None
            else self.pan_zoom_state.min_zoom_to_fit
        )
        # MAX_ZOOM_LEVEL can be a global constant
        effective_max_zoom = (
            max_zoom_override if max_zoom_override is not None else 5.0
        )  # Replace with constants.MAX_ZOOM_LEVEL

        self.pan_zoom_state.set_zoom_level(zoom, effective_min_zoom, effective_max_zoom)
        self.pan_zoom_state.set_pan(pan_x, pan_y)
        # Pan/zoom usually results in 'interactive' updates, final update might be separate
        self.notify_observers(change_type="pan_zoom_updated")

    def reset_pan_zoom_for_image_view(self, canvas_width: int, canvas_height: int):
        if self.image_data.original_image:
            self.pan_zoom_state.reset_for_new_image(
                canvas_width,
                canvas_height,
                self.image_data.original_image.width,
                self.image_data.original_image.height,
            )
        else:
            self.pan_zoom_state.reset_for_new_image()
        self.notify_observers(change_type="pan_zoom_reset")

    # --- Display Settings Methods ---
    def set_brightness(self, value: float):
        self.display_state.brightness = value
        self.notify_observers(change_type="display_settings_changed")

    def set_contrast(self, value: float):
        self.display_state.contrast = value
        self.notify_observers(change_type="display_settings_changed")

    def set_colormap(self, colormap_name: str | None):
        self.display_state.colormap_name = colormap_name
        self.notify_observers(change_type="display_settings_changed")

    def reset_display_adjustments(self):
        self.display_state.reset_image_adjustments()
        self.notify_observers(change_type="display_settings_reset")

    def set_view_option(self, option_name: str, value: bool | str):
        if hasattr(self.display_state, option_name):
            setattr(self.display_state, option_name, value)
            self.notify_observers(change_type="view_options_changed")
        else:
            print(
                f"Warning: View option '{option_name}' not found in DisplayStateModel."
            )

    # --- History Methods ---
    def record_state(self):
        if not self.image_data.original_image:  # Don't record if no image
            print("No image, not recording state")
            return
        print("Recording state for ApplicationModel")
        snapshot = self.image_data.get_snapshot_data()

        if self.undo_stack:
            self.redo_stack.clear()
        self.undo_stack.append(snapshot)
        print(
            f"State recorded. Undo stack size: {len(self.undo_stack)}, Redo stack size: {len(self.redo_stack)}"
        )
        self.notify_observers(change_type="history_updated")

    def undo(self) -> bool:
        if not self.can_undo():
            print("Cannot undo: Undo stack has insufficient states.")
            return False
        print("Performing undo.")
        current_state_snapshot = self.undo_stack.pop()
        self.redo_stack.append(current_state_snapshot)

        previous_state_snapshot = self.undo_stack[-1]
        self.image_data.restore_from_snapshot(previous_state_snapshot)
        print(
            f"Undo successful. Restored to previous state. Undo stack size: {len(self.undo_stack)}, Redo stack size: {len(self.redo_stack)}"
        )
        self.notify_observers(change_type="model_restored_undo")
        return True

    def redo(self) -> bool:
        if not self.can_redo():
            print("Cannot redo: Redo stack is empty.")
            return False
        print("Performing redo.")
        state_to_restore_snapshot = self.redo_stack.pop()
        self.undo_stack.append(state_to_restore_snapshot)
        self.image_data.restore_from_snapshot(state_to_restore_snapshot)
        print(
            f"Redo successful. Restored to next state. Undo stack size: {len(self.undo_stack)}, Redo stack size: {len(self.redo_stack)}"
        )
        self.notify_observers(change_type="model_restored_redo")
        return True

    def reset_history(self):
        print("Resetting history.")
        self.undo_stack = []
        self.redo_stack = []
        if (
            self.image_data.original_image
        ):  # Only record initial state if an image is loaded
            print("Recording initial state after history reset.")
            self.record_state()  # Record current state as baseline after reset
        else:  # If no image, ensure stacks are empty and notify
            print(
                "No image present, history stacks cleared. Not recording initial state."
            )
            self.notify_observers(change_type="history_updated")

    def can_undo(self) -> bool:
        can = len(self.undo_stack) > 1
        # print(f"can_undo check: {can}") # Uncomment for debugging if needed
        return can

    def can_redo(self) -> bool:
        can = bool(self.redo_stack)
        # print(f"can_redo check: {can}") # Uncomment for debugging if needed
        return can

    # --- Getters for convenience (prefer specific getters if model grows complex) ---
    def get_full_state_for_view(self) -> dict:
        """
        Provides a comprehensive snapshot of data needed by views.
        This can be refined as needed.
        """
        return {
            "original_image": self.image_data.original_image,
            "mask_array": self.image_data.mask_array,
            "exact_boundaries": self.image_data.exact_boundaries,
            "included_cells": self.image_data.included_cells,
            "user_drawn_cell_ids": self.image_data.user_drawn_cell_ids,
            "zoom_level": self.pan_zoom_state.zoom_level,
            "pan_x": self.pan_zoom_state.pan_x,
            "pan_y": self.pan_zoom_state.pan_y,
            "min_zoom_to_fit": self.pan_zoom_state.min_zoom_to_fit,
            "brightness": self.display_state.brightness,
            "contrast": self.display_state.contrast,
            "colormap_name": self.display_state.colormap_name,
            "show_original_image": self.display_state.show_original_image,
            "show_cell_masks": self.display_state.show_cell_masks,
            "show_cell_boundaries": self.display_state.show_cell_boundaries,
            "boundary_color_name": self.display_state.boundary_color_name,
            "show_deselected_masks_only": self.display_state.show_deselected_masks_only,
            "show_cell_numbers": self.display_state.show_cell_numbers,
            "base_filename": self.base_filename,
            "current_file_path": self.current_file_path,
            "segmentation_diameter": self.segmentation_diameter,
            "pdf_options": {
                "masks_only": self.display_state.pdf_opt_masks_only,
                "boundaries_only": self.display_state.pdf_opt_boundaries_only,
                "numbers_only": self.display_state.pdf_opt_numbers_only,
                "masks_boundaries": self.display_state.pdf_opt_masks_boundaries,
                "masks_numbers": self.display_state.pdf_opt_masks_numbers,
                "boundaries_numbers": self.display_state.pdf_opt_boundaries_numbers,
                "masks_boundaries_numbers": self.display_state.pdf_opt_masks_boundaries_numbers,
            },
        }

    def get_processed_image_for_display(self) -> Image.Image | None:
        """
        Applies brightness, contrast, and colormap to the original image.
        This logic was in DisplaySettingsController.
        """
        if not self.image_data.original_image:
            return None

        # Ensure Pillow is imported for ImageEnhance, ImageOps, mcm
        try:
            import matplotlib.cm as mcm
            from PIL import ImageEnhance, ImageOps
        except ImportError:
            print(
                "Pillow or Matplotlib not fully available for image processing in get_processed_image_for_display."
            )
            return self.image_data.original_image  # Return unprocessed

        processed_image = self.image_data.original_image.copy()

        # Apply Brightness
        if self.display_state.brightness != 1.0:
            enhancer = ImageEnhance.Brightness(processed_image)
            processed_image = enhancer.enhance(self.display_state.brightness)

        # Apply Contrast
        if self.display_state.contrast != 1.0:
            enhancer = ImageEnhance.Contrast(processed_image)
            processed_image = enhancer.enhance(self.display_state.contrast)

        # Apply Colormap
        if (
            self.display_state.colormap_name
            and self.display_state.colormap_name != "None"
        ):
            # This logic is copied from DisplaySettingsController
            if processed_image.mode == "L" or processed_image.mode == "1":
                try:
                    cmap = mcm.get_cmap(self.display_state.colormap_name)
                    lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
                    img_gray = (
                        processed_image
                        if processed_image.mode == "L"
                        else processed_image.convert("L")
                    )
                    processed_image = Image.fromarray(lut[np.array(img_gray)]).convert(
                        "RGB"
                    )
                except Exception as e:
                    print(
                        f"Error applying colormap to L/1 image in get_processed_image_for_display: {e}"
                    )
                    # Keep original if colormap fails
            elif processed_image.mode in ["RGB", "RGBA"]:
                try:
                    # Grayscale first, then apply colormap if not simple grayscale/invert
                    if self.display_state.colormap_name not in [
                        "grayscale",
                        "invert_color",
                    ]:
                        img_gray = ImageOps.grayscale(processed_image.convert("RGB"))
                        cmap = mcm.get_cmap(self.display_state.colormap_name)
                        lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
                        colored_rgb = Image.fromarray(lut[np.array(img_gray)]).convert(
                            "RGB"
                        )
                        if processed_image.mode == "RGBA":
                            alpha = processed_image.split()[3]
                            processed_image = Image.merge(
                                "RGBA", (*colored_rgb.split(), alpha)
                            )
                        else:
                            processed_image = colored_rgb
                    elif self.display_state.colormap_name == "grayscale":
                        processed_image = ImageOps.grayscale(
                            processed_image.convert("RGB")
                        )
                    elif self.display_state.colormap_name == "invert_color":
                        rgb_part = processed_image.convert("RGB")
                        if (
                            rgb_part.mode == "RGB"
                        ):  # Ensure it's truly RGB before point() with LUT for 3 channels
                            lut_invert = [i for i in range(255, -1, -1)] * 3
                            rgb_part = rgb_part.point(lut_invert)
                            if processed_image.mode == "RGBA":
                                alpha = processed_image.split()[3]
                                processed_image = Image.merge(
                                    "RGBA", (*rgb_part.split(), alpha)
                                )
                            else:
                                processed_image = rgb_part
                except Exception as e:
                    print(
                        f"Error applying colormap to RGB/RGBA image in get_processed_image_for_display: {e}"
                    )
        return processed_image

    def update_segmentation_diameter(self, diameter_str: str):
        old_diameter = self.segmentation_diameter
        self.segmentation_diameter = diameter_str
        print(
            f"Segmentation diameter updated from '{old_diameter}' to '{diameter_str}'."
        )
        self.notify_observers(change_type="segmentation_params_changed")
