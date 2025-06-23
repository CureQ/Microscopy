import os

import numpy as np
from aicsimageio import AICSImage as AICSImageType
from PIL import Image

from .. import constants
from ..utils.debug_logger import log
from .display_state_model import DisplayStateModel
from .image_data_model import ImageDataModel
from .pan_zoom_model import PanZoomModel


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
        log(
            "ApplicationModel: Core data models (ImageData, PanZoom, DisplayState) initialized.",
            level="DEBUG",
        )

        # File related state
        self.current_file_path: str | None = None
        self.base_filename: str | None = None
        self.segmentation_diameter: str = constants.SEGMENTATION_DIAMETER_DEFAULT

        # These will be populated by UI and reflect available options from image_dims
        self.display_state.total_channels = 0
        self.display_state.total_z_slices = 0

        # History (Undo/Redo)
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []

        self._observers: list[callable] = []

        # Cache for processed base image
        self._cached_processed_image: Image.Image | None = None
        self._cached_processed_image_brightness_state: float | None = None
        self._cached_processed_image_contrast_state: float | None = None
        self._cached_processed_image_colormap_state: str | None = None
        self._cached_processed_image_original_ref: Image.Image | None = (
            None  # To detect if original_image changed
        )

        # Added status_message
        self.status_message: str = ""
        log("ApplicationModel: Initialized.", level="INFO")

    def _invalidate_processed_image_cache(self):
        log(
            "ApplicationModel: Invalidating processed image cache (B/C/Colormap part). Details below:",
            level="DEBUG",
        )
        log(
            f"  _cached_processed_image was: {'set' if self._cached_processed_image else 'None'}",
            level="DEBUG",
        )
        self._cached_processed_image = None
        log(
            f"  _cached_processed_image_brightness_state was: {self._cached_processed_image_brightness_state}",
            level="DEBUG",
        )
        self._cached_processed_image_brightness_state = None
        log(
            f"  _cached_processed_image_contrast_state was: {self._cached_processed_image_contrast_state}",
            level="DEBUG",
        )
        self._cached_processed_image_contrast_state = None
        log(
            f"  _cached_processed_image_colormap_state was: {self._cached_processed_image_colormap_state}",
            level="DEBUG",
        )
        self._cached_processed_image_colormap_state = None
        log(
            f"  _cached_processed_image_original_ref was: {'set' if self._cached_processed_image_original_ref else 'None'}",
            level="DEBUG",
        )
        self._cached_processed_image_original_ref = None
        log(
            "ApplicationModel: Cache (B/C/Colormap part) fully invalidated.",
            level="DEBUG",
        )

    def subscribe(self, observer_callback: callable):
        if observer_callback not in self._observers:
            self._observers.append(observer_callback)
            log(
                f"ApplicationModel: Observer {observer_callback.__qualname__ if hasattr(observer_callback, '__qualname__') else observer_callback} subscribed.",
                level="DEBUG",
            )
        else:
            log(
                f"ApplicationModel: Observer {observer_callback.__qualname__ if hasattr(observer_callback, '__qualname__') else observer_callback} already subscribed.",
                level="DEBUG",
            )

    def unsubscribe(self, observer_callback: callable):
        if observer_callback in self._observers:
            self._observers.remove(observer_callback)
            log(
                f"ApplicationModel: Observer {observer_callback.__qualname__ if hasattr(observer_callback, '__qualname__') else observer_callback} unsubscribed.",
                level="DEBUG",
            )
        else:
            log(
                f"ApplicationModel: Observer {observer_callback.__qualname__ if hasattr(observer_callback, '__qualname__') else observer_callback} not found for unsubscription.",
                level="DEBUG",
            )

    def notify_observers(self, change_type: str | None = None):
        """
        Notify all subscribed observers about a change.
        change_type can be used by observers to optimize updates.
        """
        log(f"Model notifying observers of change: {change_type}", level="DEBUG")
        for callback in self._observers:
            try:
                log(
                    f"Notifying observer: {callback.__qualname__ if hasattr(callback, '__qualname__') else callback} for change_type: {change_type}",
                    level="DEBUG",
                )
                callback(change_type)
            except Exception as e:
                log(
                    f"Error in observer callback: {callback} with error: {e}",
                    level="ERROR",
                )

    # --- Image Data Methods ---
    def load_new_image(
        self,
        pil_image: Image.Image | None,  # For non-AICS or fallback
        file_path: str,
        base_filename: str,
        scale_conversion=None,
        aics_image_obj_param: AICSImageType | None = None,  # New param
        aics_dims_param=None,  # New param
    ):
        self.image_data.reset()  # Reset first

        # If AICS object was passed from FileIOController, set it on image_data *after* reset
        if aics_image_obj_param:
            self.image_data.set_aics_image_obj(aics_image_obj_param, aics_dims_param)
            log(
                f"load_new_image: AICS object and dims assigned/updated from parameters. Dims: {self.image_data.image_dims}",
                level="INFO",
            )
        else:
            self.image_data.set_aics_image_obj(None, None)

        self.current_file_path = file_path
        self.base_filename = base_filename
        self.image_data.set_scale_conversion(scale_conversion)

        if (
            self.image_data.aics_image_obj
            and self.image_data.image_dims is not None
            and hasattr(self.image_data.image_dims, "X")
            and self.image_data.image_dims.X > 0
            and hasattr(self.image_data.image_dims, "Y")
            and self.image_data.image_dims.Y > 0
        ):
            dims = self.image_data.image_dims
            # Ensure C and Z are at least 1 if the attributes exist, otherwise default to 1.
            # AICSImage usually provides C, Z, T, X, Y, S.
            raw_c = dims.C if hasattr(dims, "C") else 1
            raw_z = dims.Z if hasattr(dims, "Z") else 1

            self.display_state.total_channels = int(raw_c)
            self.display_state.total_z_slices = int(raw_z)

            # Default Z settings
            if self.display_state.total_z_slices > 1:
                self.display_state.z_processing_method = "max_project"
            else:
                self.display_state.z_processing_method = "slice"
            self.display_state.current_z_slice_index = 0  # Always initialize to 0

            # Auto-map image channels to the display_channel_configs
            # and set them active if a source channel is assigned.
            num_img_channels = self.display_state.total_channels
            num_display_slots = len(self.display_state.display_channel_configs)

            for i in range(num_display_slots):
                config = self.display_state.display_channel_configs[i]
                if (
                    i < num_img_channels
                ):  # If there's an image channel available for this display slot
                    config["source_idx"] = i
                    config["active_in_composite"] = (
                        True  # Activate it by default if mapped
                    )
                    log(
                        f"  Auto-mapping display_slot[{i}] ('{config['name']}') to img_channel[{i}], active: True",
                        level="DEBUG",
                    )
                else:  # No image channel for this slot
                    config["source_idx"] = None
                    config["active_in_composite"] = (
                        False  # Deactivate if no source assigned
                    )
                    log(
                        f"  Deactivating display_slot[{i}] ('{config['name']}'), no img_channel available (img_channels: {num_img_channels})",
                        level="DEBUG",
                    )

            self._invalidate_processed_image_cache()
            initial_view = self.get_processed_image_for_display()
            if initial_view is None:
                log(
                    "Failed to generate initial 2D view from AICS object after load_new_image processing.",
                    level="ERROR",
                )

        elif pil_image:
            log(
                "ApplicationModel.load_new_image: Processing direct PIL image.",
                level="DEBUG",
            )
            self.image_data.set_image_data(pil_image)
            bands = pil_image.getbands()
            num_bands = len(bands)
            self.display_state.total_channels = num_bands
            self.display_state.total_z_slices = 1

            self.display_state.z_processing_method = "slice"
            self.display_state.current_z_slice_index = 0  # Ensure 0 for non-AICS

            if num_bands == 1:
                self.display_state.display_channel_configs[0]["source_idx"] = 0
            else:
                self.display_state.display_channel_configs[0]["source_idx"] = (
                    0 if num_bands > 0 else None
                )
                self.display_state.display_channel_configs[1]["source_idx"] = (
                    1 if num_bands > 1 else None
                )
                self.display_state.display_channel_configs[2]["source_idx"] = (
                    2 if num_bands > 2 else None
                )
            self._invalidate_processed_image_cache()
            self.image_data.original_image = self.get_processed_image_for_display()

        else:
            # This block is entered if:
            # 1. No AICS object was provided (self.image_data.aics_image_obj is None)
            #    OR self.image_data.image_dims evaluated to False.
            # AND
            # 2. No pil_image was provided directly.
            log_message = "load_new_image: No direct PIL image provided."
            if self.image_data.aics_image_obj and not self.image_data.image_dims:
                log_message += " AICS object IS present, but its dimensions are missing or invalid."
            elif not self.image_data.aics_image_obj:
                log_message += " No AICS object present either."
            else:  # AICS obj present, dims present but somehow skipped first if? (should not happen)
                log_message += " AICS object and dims seem present but conditions not met as expected."
            log(log_message, level="WARNING")

            self.image_data.original_image = None  # Ensure it's None
            self.display_state.total_channels = 0

        self.reset_history()
        log(
            f"Image loaded: {self.base_filename}, path: {self.current_file_path}. Current 2D view generated (if successful).",
            level="INFO",
        )
        self.notify_observers(change_type="image_loaded")

    def set_segmentation_result(self, mask_array: np.ndarray):
        self.image_data.set_segmentation_result(mask_array)
        self.record_state()  # Record after segmentation
        log(
            f"ApplicationModel: Segmentation result processed. Mask shape: {mask_array.shape if mask_array is not None else 'None'}.",
            level="INFO",
        )
        self.notify_observers(change_type="segmentation_updated")

    def toggle_cell_inclusion(self, cell_id: int):
        self.image_data.toggle_cell_inclusion(cell_id)
        self.record_state()
        log(f"ApplicationModel: Toggled cell inclusion for ID {cell_id}.", level="INFO")
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
        log(
            f"ApplicationModel: Added user-drawn cell mask for new ID {new_cell_id}.",
            level="INFO",
        )
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
        effective_max_zoom = max_zoom_override if max_zoom_override is not None else 5.0

        self.pan_zoom_state.set_zoom_level(zoom, effective_min_zoom, effective_max_zoom)
        self.pan_zoom_state.set_pan(pan_x, pan_y)
        log(
            f"ApplicationModel: Pan/zoom update. Zoom: {zoom:.4f}, Pan: ({pan_x:.2f},{pan_y:.2f}). Effective min_zoom: {effective_min_zoom:.4f}, max_zoom: {effective_max_zoom:.4f}",
            level="DEBUG",
        )
        self.notify_observers(change_type="pan_zoom_updated")

    def reset_pan_zoom_for_image_view(self, canvas_width: int, canvas_height: int):
        log(
            f"ApplicationModel: Pan/zoom reset requested for image view. Canvas: {canvas_width}x{canvas_height}",
            level="INFO",
        )
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
        if self.display_state.brightness != value:
            self.display_state.brightness = value
            self._invalidate_processed_image_cache()
            log(f"ApplicationModel: Brightness set to {value:.2f}.", level="DEBUG")
            self.notify_observers(change_type="display_settings_changed")

    def set_contrast(self, value: float):
        if self.display_state.contrast != value:
            self.display_state.contrast = value
            self._invalidate_processed_image_cache()
            log(f"ApplicationModel: Contrast set to {value:.2f}.", level="DEBUG")
            self.notify_observers(change_type="display_settings_changed")

    def set_colormap(self, colormap_name: str | None):
        if self.display_state.colormap_name != colormap_name:
            self.display_state.colormap_name = colormap_name
            self._invalidate_processed_image_cache()
            log(f"ApplicationModel: Colormap set to {colormap_name}.", level="DEBUG")
            self.notify_observers(change_type="display_settings_changed")

    def reset_display_adjustments(self):
        """Resets brightness, contrast, and colormap to their default values."""
        self.display_state.reset_image_adjustments()
        self._invalidate_processed_image_cache()  # Image appearance changes
        log(
            "ApplicationModel: Display adjustments (brightness, contrast, colormap) reset.",
            level="INFO",
        )
        self.notify_observers("display_settings_reset")
        self.record_state()  # Record state after resetting display adjustments

    def reset_channel_z_configurations(self):
        """Resets channel mapping and Z-stack processing to their default values."""
        log("ApplicationModel: Resetting channel and Z configurations.", level="INFO")
        self.display_state.reset_channel_z_defaults()
        self._invalidate_processed_image_cache()  # Channel/Z changes require image re-processing
        self.notify_observers("channel_z_settings_reset")
        self.record_state()

    def set_view_option(self, option_name: str, value: bool | str | int | None | dict):
        if (
            hasattr(self.display_state, option_name)
            and option_name != "display_channel_configs"
        ):  # Standard attribute
            current_value = getattr(self.display_state, option_name)
            if isinstance(current_value, bool) and not isinstance(value, bool):
                log(
                    f"DEFENSIVE: set_view_option: For '{option_name}', got non-bool value {value} (type: {type(value)}), casting to bool.",
                    level="WARNING",
                )
                value = bool(value)
            if current_value != value:
                setattr(self.display_state, option_name, value)
                log(
                    f"DisplayStateModel.{option_name} changed to {value} (type: {type(value)}). Verifying: {getattr(self.display_state, option_name)}",
                    level="DEBUG",
                )
                self._invalidate_processed_image_cache()

                if option_name.startswith("pdf_opt_"):
                    self.notify_observers(change_type="pdf_options_changed")
                elif option_name in ["z_processing_method", "current_z_slice_index"]:
                    self.notify_observers(change_type="channel_z_settings_changed")
                else:
                    self.notify_observers(change_type="view_options_changed")

        elif option_name == "display_channel_config_update" and isinstance(value, dict):
            config_index = value.get("config_index")
            new_source_idx = value.get("source_idx")  # Can be int or None
            new_active_state = value.get(
                "active_in_composite"
            )  # Can be bool or None (if not changing)
            new_color = value.get("color_tuple_0_255")  # Can be tuple or None

            if isinstance(config_index, int) and 0 <= config_index < len(
                self.display_state.display_channel_configs
            ):
                config_changed = False
                current_config = self.display_state.display_channel_configs[
                    config_index
                ]

                if (
                    "source_idx" in value
                ):  # Check if key exists, to allow setting to None
                    if current_config["source_idx"] != new_source_idx:
                        current_config["source_idx"] = new_source_idx
                        log(
                            f"DisplayStateModel.display_channel_configs[{config_index}] source_idx changed to {new_source_idx}",
                            level="DEBUG",
                        )
                        config_changed = True

                if isinstance(new_active_state, bool):
                    if current_config["active_in_composite"] != new_active_state:
                        current_config["active_in_composite"] = new_active_state
                        log(
                            f"DisplayStateModel.display_channel_configs[{config_index}] active_in_composite changed to {new_active_state}",
                            level="DEBUG",
                        )
                        config_changed = True

                if new_color is not None:
                    if tuple(
                        current_config.get("color_tuple_0_255", (0, 0, 0))
                    ) != tuple(new_color):
                        current_config["color_tuple_0_255"] = tuple(new_color)
                        log(
                            f"DisplayStateModel.display_channel_configs[{config_index}] color_tuple_0_255 changed to {new_color}",
                            level="DEBUG",
                        )
                        config_changed = True

                if config_changed:
                    self._invalidate_processed_image_cache()
                    self.notify_observers(
                        change_type="channel_z_settings_changed"
                    )  # This will trigger re-render and UI sync
            else:
                log(
                    f"Warning: Invalid 'config_index' ({config_index}) for display_channel_config_update.",
                    level="WARNING",
                )

        elif option_name == "segmentation_diameter":
            self.segmentation_diameter = value
            self.notify_observers("segmentation_params_changed")
            return

        else:
            log(
                f"Warning: View option '{option_name}' not found in DisplayStateModel or invalid payload for display_channel_config_update.",
                level="WARNING",
            )

    # --- History Methods ---
    def record_state(self):
        if not self.image_data.original_image:  # Don't record if no image
            log("No image, not recording state.", level="DEBUG")
            return
        log("Recording state for ApplicationModel.", level="INFO")
        snapshot = self.image_data.get_snapshot_data()

        if self.undo_stack:
            self.redo_stack.clear()
        self.undo_stack.append(snapshot)
        log(
            f"State recorded. Undo stack size: {len(self.undo_stack)}, Redo stack size: {len(self.redo_stack)}",
            level="DEBUG",
        )
        self.notify_observers(change_type="history_updated")

    def undo(self) -> bool:
        if not self.can_undo():
            log("Cannot undo: Undo stack has insufficient states.", level="INFO")
            return False
        log("Performing undo.", level="INFO")
        current_state_snapshot = self.undo_stack.pop()
        self.redo_stack.append(current_state_snapshot)

        previous_state_snapshot = self.undo_stack[-1]
        self.image_data.restore_from_snapshot(previous_state_snapshot)
        log(
            f"Undo successful. Restored to previous state. Undo stack size: {len(self.undo_stack)}, Redo stack size: {len(self.redo_stack)}",
            level="DEBUG",
        )
        self.notify_observers(change_type="model_restored_undo")
        return True

    def redo(self) -> bool:
        if not self.can_redo():
            log("Cannot redo: Redo stack is empty.", level="INFO")
            return False
        log("Performing redo.", level="INFO")
        state_to_restore_snapshot = self.redo_stack.pop()
        self.undo_stack.append(state_to_restore_snapshot)
        self.image_data.restore_from_snapshot(state_to_restore_snapshot)
        log(
            f"Redo successful. Restored to next state. Undo stack size: {len(self.undo_stack)}, Redo stack size: {len(self.redo_stack)}",
            level="DEBUG",
        )
        self.notify_observers(change_type="model_restored_redo")
        return True

    def reset_history(self):
        log("Resetting history.", level="INFO")
        self.undo_stack = []
        self.redo_stack = []
        if (
            self.image_data.original_image
        ):  # Only record initial state if an image is loaded
            log("Recording initial state after history reset.", level="DEBUG")
            self.record_state()  # Record current state as baseline after reset
        else:  # If no image, ensure stacks are empty and notify
            log(
                "No image present, history stacks cleared. Not recording initial state.",
                level="DEBUG",
            )
            self.notify_observers(change_type="history_updated")

    def can_undo(self) -> bool:
        can = len(self.undo_stack) > 1
        # log(f"can_undo check: {can}") # Uncomment for debugging if needed
        return can

    def can_redo(self) -> bool:
        can = bool(self.redo_stack)
        # log(f"can_redo check: {can}") # Uncomment for debugging if needed
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
            "scale_conversion": self.image_data.scale_conversion,
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
            "show_ruler": self.display_state.show_ruler,
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
        Applies channel selection, Z-stack processing, brightness, contrast, and colormap.
        Uses a cache for performance.
        """
        log(
            "get_processed_image_for_display: START. Current display state for AICS derivation (if applicable):",
            level="DEBUG",
        )
        log(
            f"  Z_method='{self.display_state.z_processing_method}', Z_slice={self.display_state.current_z_slice_index}",
            level="DEBUG",
        )
        log(f"  Total_Z={self.display_state.total_z_slices}", level="DEBUG")

        # This variable will hold the 2D PIL Image derived from current channel/Z settings
        # or the existing 2D PIL image if not AICS.
        pil_image_source_for_bc_colormap: Image.Image | None = None

        # --- Part 1: Derive a 2D PIL Image (pil_image_source_for_bc_colormap) ---
        if self.image_data.aics_image_obj and self.image_data.image_dims:
            log(
                "Deriving 2D PIL view from AICSImage object using flexible display channel configs.",
                level="DEBUG",
            )
            aics_img = self.image_data.aics_image_obj
            dims = self.image_data.image_dims
            z_method = self.display_state.z_processing_method
            current_z_idx = self.display_state.current_z_slice_index
            target_yx_shape = (dims.Y, dims.X)

            # Initialize a floating point accumulator for the composite RGB image
            composite_rgb_float_array = np.zeros(
                (*target_yx_shape, 3), dtype=np.float32
            )

            processed_any_channel = False

            for channel_config in self.display_state.display_channel_configs:
                source_channel_idx = channel_config.get("source_idx")
                is_active = channel_config.get("active_in_composite", False)
                display_color_0_255 = channel_config.get(
                    "color_tuple_0_255", (0, 0, 0)
                )  # Default to black if missing

                if (
                    source_channel_idx is not None
                    and is_active
                    and 0 <= source_channel_idx < dims.C
                ):
                    log(
                        f"Processing display channel: '{channel_config.get('name')}', source img channel: {source_channel_idx}, color: {display_color_0_255}",
                        level="DEBUG",
                    )
                    try:
                        # Get ZYX data for the specific source channel
                        # Assuming T=0, S=0 for simplicity in this context
                        channel_data_z_stack = aics_img.get_image_data(
                            "ZYX", C=source_channel_idx, T=0, S=0
                        )
                        processed_yx_plane = None

                        if dims.Z > 1:
                            if z_method == "max_project":
                                processed_yx_plane = np.max(
                                    channel_data_z_stack, axis=0
                                )
                            elif z_method == "mean_project":
                                processed_yx_plane = np.mean(
                                    channel_data_z_stack, axis=0
                                )
                            else:  # Slice (default or explicit)
                                target_z = max(0, min(current_z_idx, dims.Z - 1))
                                processed_yx_plane = channel_data_z_stack[
                                    target_z, :, :
                                ]
                        elif dims.Z == 1:
                            processed_yx_plane = channel_data_z_stack[0, :, :]
                        else:  # Should not happen if dims.Z is always >= 1 from loader
                            if channel_data_z_stack.ndim == 2:  # If it was YX already
                                processed_yx_plane = channel_data_z_stack
                            else:
                                log(
                                    f"Unexpected Z dimension ({dims.Z}) or array shape for source channel {source_channel_idx}. Skipping.",
                                    level="ERROR",
                                )
                                continue

                        if processed_yx_plane is not None:
                            # Normalize to 0.0 - 1.0 float
                            p_min, p_max = (
                                float(np.min(processed_yx_plane)),
                                float(np.max(processed_yx_plane)),
                            )
                            if p_max > p_min:
                                normalized_yx_plane = (
                                    processed_yx_plane.astype(np.float32) - p_min
                                ) / (p_max - p_min)
                            elif (
                                p_max == p_min and p_max > 0
                            ):  # Single non-zero value, map to 1.0
                                normalized_yx_plane = np.ones_like(
                                    processed_yx_plane, dtype=np.float32
                                )
                            else:  # All zeros or empty
                                normalized_yx_plane = np.zeros_like(
                                    processed_yx_plane, dtype=np.float32
                                )

                            # Additive blending with display color
                            composite_rgb_float_array[:, :, 0] += (
                                normalized_yx_plane * (display_color_0_255[0] / 255.0)
                            )
                            composite_rgb_float_array[:, :, 1] += (
                                normalized_yx_plane * (display_color_0_255[1] / 255.0)
                            )
                            composite_rgb_float_array[:, :, 2] += (
                                normalized_yx_plane * (display_color_0_255[2] / 255.0)
                            )
                            processed_any_channel = True
                        else:
                            log(
                                f"Z-processed plane was None for source channel {source_channel_idx}.",
                                level="WARNING",
                            )

                    except Exception as e:
                        log(
                            f"Error processing source channel {source_channel_idx} for display config '{channel_config.get('name')}': {e}",
                            level="ERROR",
                        )
                        continue  # Skip to next display channel config
                elif source_channel_idx is not None and not (
                    0 <= source_channel_idx < dims.C
                ):
                    log(
                        f"Skipping display channel '{channel_config.get('name')}': source_idx {source_channel_idx} is out of image channel bounds (0-{dims.C - 1}).",
                        level="WARNING",
                    )

            if not processed_any_channel and dims.C > 0:
                log(
                    "No active display channels were processed from AICS image, or all resulted in errors. Defaulting to black image.",
                    level="WARNING",
                )
                # Create a black uint8 array and then convert to PIL Image
                final_uint8_array = np.zeros((*target_yx_shape, 3), dtype=np.uint8)
                derived_pil_from_aics_logic = Image.fromarray(
                    final_uint8_array, mode="RGB"
                )
            elif (
                not processed_any_channel and dims.C == 0
            ):  # No channels in image to begin with
                log(
                    "AICS image has 0 channels. Cannot derive base image.",
                    level="WARNING",
                )
                final_uint8_array = np.zeros((*target_yx_shape, 3), dtype=np.uint8)
                derived_pil_from_aics_logic = Image.fromarray(
                    final_uint8_array, mode="RGB"
                )
            else:
                # Clip to 0.0 - 1.0 range after additive blending
                composite_rgb_float_array = np.clip(composite_rgb_float_array, 0.0, 1.0)
                # Scale to 0-255 and convert to uint8
                final_uint8_array = (composite_rgb_float_array * 255).astype(np.uint8)
                derived_pil_from_aics_logic = Image.fromarray(
                    final_uint8_array, mode="RGB"
                )

            # Critical check (remains useful)
            if derived_pil_from_aics_logic is not None and not isinstance(
                derived_pil_from_aics_logic, Image.Image
            ):
                log(
                    f"CRITICAL_ERROR: derived_pil_from_aics_logic is type {type(derived_pil_from_aics_logic)}, not PIL.Image! Defaulting to black.",
                    level="ERROR",
                )
                final_uint8_array = np.zeros((*target_yx_shape, 3), dtype=np.uint8)
                derived_pil_from_aics_logic = Image.fromarray(
                    final_uint8_array, mode="RGB"
                )

            pil_image_source_for_bc_colormap = derived_pil_from_aics_logic
            # Update self.image_data.original_image to this newly derived 2D PIL view
            # This is crucial for consistency and for non-AICS path if it relies on original_image being a PIL
            self.image_data.original_image = pil_image_source_for_bc_colormap

        elif self.image_data.original_image:
            log(
                "Processing non-AICS image (e.g., JPG, PNG) through existing logic. Needs refactor for display_channel_configs.",
                level="DEBUG",
            )
            source_object = self.image_data.original_image
            if isinstance(
                source_object, Image.Image
            ):  # It's already a PIL Image as expected
                log(
                    "get_processed_image_for_display: Using existing self.image_data.original_image (already PIL.Image) as source for B/C/Colormap.",
                    level="DEBUG",
                )
                pil_image_source_for_bc_colormap = source_object
            else:  # Unknown type, not PIL, not AICS
                log(
                    f"get_processed_image_for_display: ERROR - self.image_data.original_image is an unexpected type: {type(source_object)}. Cannot use as image source.",
                    level="ERROR",
                )
                pil_image_source_for_bc_colormap = None
                self.image_data.original_image = None  # Ensure original_image is None

        else:
            log(
                "get_processed_image_for_display: No image data source available (AICS or existing PIL original_image).",
                level="DEBUG",
            )
            self._cached_processed_image_original_ref = None
            self.image_data.original_image = None  # Ensure it's None here too
            self._invalidate_processed_image_cache()  # Should also clear _cached_processed_image
            return None

        if (
            self._cached_processed_image is not None
            and self._cached_processed_image_original_ref
            is pil_image_source_for_bc_colormap
            and self._cached_processed_image_brightness_state
            == self.display_state.brightness
            and self._cached_processed_image_contrast_state
            == self.display_state.contrast
            and self._cached_processed_image_colormap_state
            == self.display_state.colormap_name
        ):
            log(
                "ApplicationModel: Processed image cache (B/C/Colormap part) HIT. Keys:",
                level="DEBUG",
            )
            log(
                f"  _cached_img is set: {self._cached_processed_image is not None}",
                level="DEBUG",
            )
            log(
                f"  _cached_ref is pil_source: {self._cached_processed_image_original_ref is pil_image_source_for_bc_colormap} (ID _cached_ref: {id(self._cached_processed_image_original_ref)}, ID pil_source: {id(pil_image_source_for_bc_colormap)})",
                level="DEBUG",
            )
            log(
                f"  _cached_brightness ({self._cached_processed_image_brightness_state}) == current ({self.display_state.brightness}): {self._cached_processed_image_brightness_state == self.display_state.brightness}",
                level="DEBUG",
            )
            log(
                f"  _cached_contrast ({self._cached_processed_image_contrast_state}) == current ({self.display_state.contrast}): {self._cached_processed_image_contrast_state == self.display_state.contrast}",
                level="DEBUG",
            )
            log(
                f"  _cached_colormap ({self._cached_processed_image_colormap_state}) == current ({self.display_state.colormap_name}): {self._cached_processed_image_colormap_state == self.display_state.colormap_name}",
                level="DEBUG",
            )
            return self._cached_processed_image

        log(
            "ApplicationModel: Processed image cache (B/C/Colormap part) MISS or stale. Details of check:",
            level="DEBUG",
        )
        log(
            f"  _cached_img is set: {self._cached_processed_image is not None}",
            level="DEBUG",
        )
        log(
            f"  _cached_ref is pil_source: {self._cached_processed_image_original_ref is pil_image_source_for_bc_colormap} (ID _cached_ref: {id(self._cached_processed_image_original_ref if self._cached_processed_image_original_ref else 0)}, ID pil_source: {id(pil_image_source_for_bc_colormap if pil_image_source_for_bc_colormap else 0)})",
            level="DEBUG",
        )
        log(
            f"  _cached_brightness ({self._cached_processed_image_brightness_state}) == current ({self.display_state.brightness}): {self._cached_processed_image_brightness_state == self.display_state.brightness}",
            level="DEBUG",
        )
        log(
            f"  _cached_contrast ({self._cached_processed_image_contrast_state}) == current ({self.display_state.contrast}): {self._cached_processed_image_contrast_state == self.display_state.contrast}",
            level="DEBUG",
        )
        log(
            f"  _cached_colormap ({self._cached_processed_image_colormap_state}) == current ({self.display_state.colormap_name}): {self._cached_processed_image_colormap_state == self.display_state.colormap_name}",
            level="DEBUG",
        )
        log("Regenerating B/C/Colormap part.", level="DEBUG")

        if pil_image_source_for_bc_colormap is None:
            log(
                "ApplicationModel: Source PIL image for B/C/Colormap is None. Cannot generate final display image.",
                level="WARNING",
            )
            self.image_data.original_image = None
            self._invalidate_processed_image_cache()
            return None

        try:
            import matplotlib.cm as mcm  # type: ignore
            from PIL import ImageEnhance, ImageOps  # type: ignore
        except ImportError:
            log(
                "Pillow or Matplotlib not fully available for B/C/Colormap processing.",
                level="ERROR",
            )
            return pil_image_source_for_bc_colormap  # Return the unadjusted PIL image

        # Start processing for B/C/Colormap from this confirmed PIL image
        processed_image = pil_image_source_for_bc_colormap.copy()

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
            if processed_image is None:  # Should be caught above, but as a safeguard
                log(
                    "ApplicationModel: processed_image became None before colormap. Aborting colormap.",
                    level="ERROR",
                )
                self._cached_processed_image = None
                self._cached_processed_image_original_ref = (
                    pil_image_source_for_bc_colormap
                )
                return None

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
                    log(f"Error applying colormap to L/1 image: {e}", level="ERROR")
            elif processed_image.mode in ["RGB", "RGBA"]:
                try:
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
                        ).convert(
                            processed_image.mode
                            if processed_image.mode in ["L", "RGB", "RGBA"]
                            else "RGB"
                        )
                    elif self.display_state.colormap_name == "invert_color":
                        img_to_invert = processed_image.convert("RGB")
                        lut_invert = [i for i in range(255, -1, -1)] * 3
                        inverted_rgb = img_to_invert.point(lut_invert)
                        if processed_image.mode == "L":
                            processed_image = inverted_rgb.convert("L")
                        elif processed_image.mode == "RGBA":
                            alpha = processed_image.split()[3]
                            processed_image = Image.merge(
                                "RGBA", (*inverted_rgb.split(), alpha)
                            )
                        else:
                            processed_image = inverted_rgb
                except Exception as e:
                    log(
                        f"Error applying colormap to RGB/RGBA image: {e}", level="ERROR"
                    )

        # Update cache
        self._cached_processed_image = processed_image
        self._cached_processed_image_original_ref = pil_image_source_for_bc_colormap
        self._cached_processed_image_brightness_state = self.display_state.brightness
        self._cached_processed_image_contrast_state = self.display_state.contrast
        self._cached_processed_image_colormap_state = self.display_state.colormap_name

        log(
            "ApplicationModel: Processed image cache (B/C/Colormap part) regenerated.",
            level="DEBUG",
        )
        return processed_image

    def update_segmentation_diameter(self, diameter_str: str):
        old_diameter = self.segmentation_diameter
        self.segmentation_diameter = diameter_str
        log(
            f"Segmentation diameter updated from '{old_diameter}' to '{diameter_str}'.",
            level="INFO",
        )
        self.notify_observers(change_type="segmentation_params_changed")

    def set_status_message(self, message: str):
        """Sets a status message and notifies observers."""
        if self.status_message != message:
            self.status_message = message
            log(f"Status message set: {message}", level="INFO")
            self.notify_observers(change_type="status_message_changed")

    def clear_status_message(self):
        """Clears the status message and notifies observers."""
        if self.status_message != "":
            self.status_message = ""
            log("Status message cleared.", level="INFO")
            self.notify_observers(change_type="status_message_changed")

    def reset_for_new_image(self):
        """Resets the entire model to a state as if no image has been loaded."""
        log("ApplicationModel: Resetting state for new image.", level="INFO")

        self.image_data.reset()
        self.pan_zoom_state.reset_for_new_image()  # Resets to default pan/zoom

        self.display_state.reset_image_adjustments()
        self.display_state.reset_pdf_options()
        self.display_state.reset_channel_z_defaults()

        self.segmentation_diameter = constants.SEGMENTATION_DIAMETER_DEFAULT
        self._invalidate_processed_image_cache()
        self.reset_history()  # Clears undo/redo stack

        self.base_filename = None
        self.current_file_path = None
        self.image_load_error_message = None

        # Clear any existing status message from a previous operation
        self.clear_status_message()

        log("ApplicationModel: State reset complete.", level="INFO")
        self.notify_observers(
            change_type="model_reset_for_new_image"
        )  # Notify that a full reset occurred

    def set_image_load_error(self, file_path: str, error_message: str):
        """Called by FileIOController when image loading fails."""
        self.current_file_path = file_path
        self.base_filename = (
            os.path.splitext(os.path.basename(file_path))[0] if file_path else "Error"
        )
        self.image_load_error_message = error_message

        # Ensure image_data is also reset to a clean state
        self.image_data.reset()
        self._invalidate_processed_image_cache()

        log(
            f"ApplicationModel: Image load error set for '{self.base_filename}': {error_message}",
            level="ERROR",
        )
        self.notify_observers(change_type="image_load_failed")

    def set_scale_bar_microns(self, value: int):
        if self.display_state.scale_bar_microns != value:
            self.display_state.scale_bar_microns = int(value)
            log(f"ApplicationModel: Scale bar microns set to {value}.", level="DEBUG")
            self.notify_observers(change_type="display_settings_changed")
