import os

import numpy as np
from aicsimageio import AICSImage
from PIL import Image

from .. import constants
from ..utils.debug_logger import log
from .lasx_lif_image import LASXLifImage


class ImageReader:
    def _normalize_grayscale_to_uint8(
        self, arr_2d: np.ndarray, path: str
    ) -> np.ndarray:
        """Normalizes a 2D grayscale array to uint8."""
        log(
            f"_normalize_grayscale_to_uint8: Entry for {os.path.basename(path)}, input dtype: {arr_2d.dtype}",
            level="DEBUG",
        )
        if arr_2d.dtype == np.uint8:
            log(
                f"_normalize_grayscale_to_uint8: Already uint8 for {os.path.basename(path)}",
                level="DEBUG",
            )
            return arr_2d

        log(
            f"Normalizing grayscale image (dtype: {arr_2d.dtype}) to uint8 for {os.path.basename(path)}."
        )
        if np.issubdtype(arr_2d.dtype, np.floating):
            val_min, val_max = np.min(arr_2d), np.max(arr_2d)
            if val_max > val_min:
                data = (arr_2d - val_min) / (val_max - val_min) * 255.0
            else:  # Flat image
                data = np.zeros_like(arr_2d, dtype=np.float32)
            output_arr = data.astype(np.uint8)
            log(
                f"_normalize_grayscale_to_uint8: Normalized floating point to uint8 for {os.path.basename(path)}. Output dtype: {output_arr.dtype}",
                level="DEBUG",
            )
            return output_arr
        elif arr_2d.dtype == np.uint16:
            output_arr = (arr_2d / 257.0).astype(np.uint8)  # 65535 / 255 = 257
            log(
                f"_normalize_grayscale_to_uint8: Normalized uint16 to uint8 for {os.path.basename(path)}. Output dtype: {output_arr.dtype}",
                level="DEBUG",
            )
            return output_arr
        elif np.issubdtype(
            arr_2d.dtype, np.integer
        ):  # General integer case (includes int8, int16, int32, uint32 etc.)
            val_min, val_max = np.min(arr_2d), np.max(arr_2d)
            if val_max == val_min:  # Flat image
                # Return 0 if min/max is 0, otherwise 127 (mid-gray)
                output_val = 0 if val_min == 0 else 127
                output_arr = np.full_like(arr_2d, output_val, dtype=np.uint8)
                log(
                    f"_normalize_grayscale_to_uint8: Flat integer image, filled with {output_val} for {os.path.basename(path)}. Output dtype: {output_arr.dtype}",
                    level="DEBUG",
                )
                return output_arr
            # Use float64 for precision during normalization
            arr_f64 = arr_2d.astype(np.float64)
            data = (arr_f64 - val_min) / (val_max - val_min) * 255.0
            output_arr = np.clip(data, 0, 255).astype(
                np.uint8
            )  # Clip to ensure valid range
            log(
                f"_normalize_grayscale_to_uint8: Normalized general integer to uint8 for {os.path.basename(path)}. Output dtype: {output_arr.dtype}",
                level="DEBUG",
            )
            return output_arr
        else:
            log(
                f"Unhandled dtype {arr_2d.dtype} for grayscale normalization of {os.path.basename(path)}. Attempting max scaling.",
                level="WARNING",
            )
            arr_max_val = np.max(arr_2d)
            if arr_max_val > 0:
                output_arr = (arr_2d.astype(np.float64) / arr_max_val * 255.0).astype(
                    np.uint8
                )
                log(
                    f"_normalize_grayscale_to_uint8: Unhandled dtype, used max scaling. Output dtype: {output_arr.dtype}",
                    level="DEBUG",
                )
                return output_arr
            output_arr = arr_2d.astype(
                np.uint8
            )  # Or np.zeros_like(arr_2d, dtype=np.uint8)
            log(
                f"_normalize_grayscale_to_uint8: Unhandled dtype, max val is 0, converted to uint8. Output dtype: {output_arr.dtype}",
                level="DEBUG",
            )
            return output_arr

    def _normalize_3channel_to_uint8(
        self, arr_3channel: np.ndarray, path: str
    ) -> np.ndarray:
        """Normalizes a 3-channel (H, W, 3) array to uint8."""
        log(
            f"_normalize_3channel_to_uint8: Entry for {os.path.basename(path)}, input dtype: {arr_3channel.dtype}",
            level="DEBUG",
        )
        if arr_3channel.dtype == np.uint8:
            log(
                f"_normalize_3channel_to_uint8: Already uint8 for {os.path.basename(path)}",
                level="DEBUG",
            )
            return arr_3channel

        log(
            f"Normalizing 3-channel image (dtype: {arr_3channel.dtype}) to uint8 for {os.path.basename(path)}."
        )
        if np.issubdtype(arr_3channel.dtype, np.floating):
            val_min, val_max = (
                np.min(arr_3channel),
                np.max(arr_3channel),
            )  # Global min/max
            if val_max > val_min:
                data = (arr_3channel - val_min) / (val_max - val_min) * 255.0
            else:  # Flat image
                data = np.zeros_like(arr_3channel, dtype=np.float32)
            output_arr = np.clip(data, 0, 255).astype(np.uint8)
            log(
                f"_normalize_3channel_to_uint8: Normalized floating point to uint8 for {os.path.basename(path)}. Output dtype: {output_arr.dtype}",
                level="DEBUG",
            )
            return output_arr
        elif arr_3channel.dtype == np.uint16:
            output_arr = (arr_3channel / 257.0).astype(np.uint8)
            log(
                f"_normalize_3channel_to_uint8: Normalized uint16 to uint8 for {os.path.basename(path)}. Output dtype: {output_arr.dtype}",
                level="DEBUG",
            )
            return output_arr
        elif np.issubdtype(arr_3channel.dtype, np.integer):
            val_min, val_max = np.min(arr_3channel), np.max(arr_3channel)
            if val_max == val_min:
                output_val = 0 if val_min == 0 else 127
                output_arr = np.full_like(arr_3channel, output_val, dtype=np.uint8)
                log(
                    f"_normalize_3channel_to_uint8: Flat integer image, filled with {output_val} for {os.path.basename(path)}. Output dtype: {output_arr.dtype}",
                    level="DEBUG",
                )
                return output_arr
            arr_f64 = arr_3channel.astype(np.float64)
            data = (arr_f64 - val_min) / (val_max - val_min) * 255.0
            output_arr = np.clip(data, 0, 255).astype(np.uint8)
            log(
                f"_normalize_3channel_to_uint8: Normalized general integer to uint8 for {os.path.basename(path)}. Output dtype: {output_arr.dtype}",
                level="DEBUG",
            )
            return output_arr
        else:
            log(
                f"Unhandled dtype {arr_3channel.dtype} for 3-channel normalization of {os.path.basename(path)}. Attempting max scaling.",
                level="WARNING",
            )
            arr_max_val = np.max(arr_3channel)
            if arr_max_val > 0:
                data_scaled = (arr_3channel.astype(np.float64) / arr_max_val) * 255.0
                output_arr = np.clip(data_scaled, 0, 255).astype(np.uint8)
                log(
                    f"_normalize_3channel_to_uint8: Unhandled dtype, used max scaling. Output dtype: {output_arr.dtype}",
                    level="DEBUG",
                )
                return output_arr
            output_arr = arr_3channel.astype(
                np.uint8
            )  # Or np.zeros_like(arr_3channel, dtype=np.uint8)
            log(
                f"_normalize_3channel_to_uint8: Unhandled dtype, max val is 0, converted to uint8. Output dtype: {output_arr.dtype}",
                level="DEBUG",
            )
            return output_arr

    def _process_aics_image_data(
        self,
        img_aics: AICSImage,
        path: str,
        selected_channel_indices: list[int] | None = None,
        z_selection_params: dict | None = None,
    ):
        """Processes data from an AICSImage object to produce an image array and scale info."""
        log(
            f"_process_aics_image_data: Entry for {os.path.basename(path)}. Channels: {selected_channel_indices}, Z-params: {z_selection_params}",
            level="DEBUG",
        )
        arr = None
        scale_conversion = None
        try:
            scale_conversion = img_aics.physical_pixel_sizes
            log(
                f"_process_aics_image_data: Physical pixel sizes found for {os.path.basename(path)}: {scale_conversion}"
            )
        except AttributeError:
            scale_conversion = None
            log(
                f"_process_aics_image_data: No physical_pixel_sizes found for {os.path.basename(path)}. Using None.",
                level="WARNING",
            )

        channels_to_load = []
        if selected_channel_indices:
            channels_to_load = [
                c for c in selected_channel_indices if 0 <= c < img_aics.dims.C
            ]
            if not channels_to_load and img_aics.dims.C > 0:
                log(
                    f"Warning: Invalid selected_channel_indices ({selected_channel_indices}) for {os.path.basename(path)}. Defaulting to channel 0.",
                    level="WARNING",
                )
                channels_to_load = [0]
        else:
            if img_aics.dims.C > 0:
                channels_to_load = [
                    0
                ]  # Default to channel 0 if no selection and channels exist
                log(
                    f"_process_aics_image_data: No channels selected by user for {os.path.basename(path)}, defaulting to channel 0.",
                    level="INFO",
                )
            # If img_aics.dims.C is 0, channels_to_load remains empty.

        if (
            not channels_to_load and img_aics.dims.C > 0
        ):  # If still no channels to load but image has channels
            log(
                f"_process_aics_image_data: No valid channels to load for {os.path.basename(path)} after processing selections. Available C: {img_aics.dims.C}. Defaulting to channel 0.",
                level="WARNING",
            )
            channels_to_load = [
                0
            ]  # Force load channel 0 as a last resort if channels exist
        elif (
            not channels_to_load and img_aics.dims.C == 0
        ):  # No channels in image and none to load
            log(
                f"_process_aics_image_data: No channels available in image or to load for {os.path.basename(path)}. Available C: {img_aics.dims.C}",
                level="ERROR",
            )
            raise ValueError(
                f"No channels available or to load for {os.path.basename(path)} (available C: {img_aics.dims.C})."
            )

        log(
            f"_process_aics_image_data: Determined channels to load for {os.path.basename(path)}: {channels_to_load}",
            level="DEBUG",
        )

        processed_channels_data = []
        for c_idx in channels_to_load:
            # AICSImage loads ZYX C T S order. Request specific C, S, T. Get "ZYX" for a specific channel.
            channel_data_z_stack = img_aics.get_image_data("ZYX", S=0, T=0, C=c_idx)
            current_z_dim_size = img_aics.dims.Z
            default_z_index = current_z_dim_size // 2 if current_z_dim_size > 1 else 0
            processed_c_data_for_channel = None

            log(
                f"_process_aics_image_data: Ch {c_idx}, original Z-dim size: {current_z_dim_size} for {os.path.basename(path)}",
                level="DEBUG",
            )

            if current_z_dim_size > 1 and z_selection_params:
                z_type = z_selection_params.get("type")
                if z_type == "max_project":
                    processed_c_data_for_channel = np.max(channel_data_z_stack, axis=0)
                    log(
                        f"_process_aics_image_data: Ch {c_idx}, applied max_project for {os.path.basename(path)}",
                        level="DEBUG",
                    )
                elif z_type == "mean_project":
                    processed_c_data_for_channel = np.mean(channel_data_z_stack, axis=0)
                    log(
                        f"_process_aics_image_data: Ch {c_idx}, applied mean_project for {os.path.basename(path)}",
                        level="DEBUG",
                    )
                elif z_type == "slice":
                    z_idx_val = z_selection_params.get("value", default_z_index)
                    target_z_slice = max(0, min(z_idx_val, current_z_dim_size - 1))
                    log(
                        f"_process_aics_image_data: Ch {c_idx}, Z-slice: {target_z_slice} (req: {z_idx_val}) for {os.path.basename(path)}"
                    )
                    processed_c_data_for_channel = channel_data_z_stack[
                        target_z_slice, :, :
                    ]
                else:
                    log(
                        f"_process_aics_image_data: Ch {c_idx}, Z-slice: {default_z_index} (unknown Z type '{z_type}') for {os.path.basename(path)}.",
                        level="WARNING",
                    )
                    processed_c_data_for_channel = channel_data_z_stack[
                        default_z_index, :, :
                    ]
            else:
                log(
                    f"_process_aics_image_data: Ch {c_idx}, Z-slice: {default_z_index} (single Z or no Z params) for {os.path.basename(path)}."
                )
                processed_c_data_for_channel = channel_data_z_stack[
                    default_z_index, :, :
                ]
            processed_channels_data.append(processed_c_data_for_channel)

        num_loaded_channels = len(processed_channels_data)
        log(
            f"_process_aics_image_data: Number of channels processed into 2D arrays: {num_loaded_channels} for {os.path.basename(path)}",
            level="DEBUG",
        )
        if num_loaded_channels == 0:
            log(
                f"_process_aics_image_data: No channel data was processed for {os.path.basename(path)}.",
                level="ERROR",
            )
            raise ValueError(f"No channel data processed for {os.path.basename(path)}")
        elif num_loaded_channels == 1:
            arr = processed_channels_data[0]
            log(
                f"_process_aics_image_data: Single channel processed for {os.path.basename(path)}. Shape: {arr.shape}",
                level="DEBUG",
            )
        elif num_loaded_channels == 2:
            ch1, ch2 = processed_channels_data[0], processed_channels_data[1]
            if ch1.shape != ch2.shape:
                log(
                    f"_process_aics_image_data: Shape mismatch for 2-channel stacking: {ch1.shape} vs {ch2.shape} for {os.path.basename(path)}",
                    level="ERROR",
                )
                raise ValueError("Shape mismatch for 2-channel stacking.")
            zeros_ch = np.zeros_like(ch1, dtype=ch1.dtype)
            arr = np.stack((ch1, ch2, zeros_ch), axis=-1)
            log(
                f"_process_aics_image_data: Two channels processed and stacked with a zero channel for {os.path.basename(path)}. Shape: {arr.shape}",
                level="DEBUG",
            )
        elif num_loaded_channels >= 3:  # Handles 3 or more loaded channels
            arr = np.stack(processed_channels_data[:3], axis=-1)  # Take first 3 for RGB
            log(
                f"_process_aics_image_data: Three or more channels processed, stacked first 3 for RGB for {os.path.basename(path)}. Shape: {arr.shape}",
                level="DEBUG",
            )
            if num_loaded_channels > 3:
                log(
                    f"Loaded {num_loaded_channels} channels, using first 3 for RGB for {os.path.basename(path)}."
                )
        log(
            f"_process_aics_image_data: Exit for {os.path.basename(path)}. Array shape: {arr.shape if arr is not None else 'None'}, Scale info: {scale_conversion}",
            level="DEBUG",
        )
        return arr, scale_conversion

    def read_image_to_array(
        self,
        path,
        selected_channel_indices: list[int] | None = None,
        z_selection_params: dict | None = None,
    ):
        log(
            f"read_image_to_array: Entry for {os.path.basename(path)}. Channels: {selected_channel_indices}, Z-params: {z_selection_params}",
            level="DEBUG",
        )
        ext = os.path.splitext(path)[1].lower()
        arr = None
        scale_conversion = None  # Default to None if not found
        aics_image_obj_for_model = None

        if ext in [".tiff", ".tif"]:
            try:
                log(
                    f"Attempting to load TIFF {os.path.basename(path)} with AICSImage first to check for Z-slices."
                )
                img_aics = AICSImage(path)
                aics_image_obj_for_model = img_aics  # Tentatively assign
                log(
                    f"AICSImage loaded TIFF {os.path.basename(path)} successfully. Dims: {img_aics.dims if hasattr(img_aics, 'dims') else 'N/A'}",
                    level="DEBUG",
                )

                if (
                    hasattr(img_aics, "dims")
                    and hasattr(img_aics.dims, "Z")
                    and img_aics.dims.Z > 1
                ):
                    log(
                        f"Multi-Z stack TIFF detected (Z={img_aics.dims.Z}). Processing with AICSImage for {os.path.basename(path)}."
                    )
                    arr, scale_conversion = self._process_aics_image_data(
                        img_aics, path, selected_channel_indices, z_selection_params
                    )
                else:
                    # Z is 1, not available as expected, or AICSImage might have loaded it (e.g. OME-TIFF with Z=1)
                    # Try PIL for potentially better/simpler handling of flat TIFFs.
                    z_dim_for_log = (
                        img_aics.dims.Z
                        if hasattr(img_aics, "dims") and hasattr(img_aics.dims, "Z")
                        else "N/A"
                    )
                    log(
                        f"AICSImage loaded TIFF with Z={z_dim_for_log} for {os.path.basename(path)}. Attempting PIL for potentially simpler interpretation of flat TIFF."
                    )
                    try:
                        img_pil = Image.open(path)
                        arr = np.array(img_pil)
                        aics_image_obj_for_model = None  # PIL succeeded, so we don't need the AICS object for the model
                        scale_conversion = (
                            None  # PIL doesn't typically provide rich scale metadata
                        )
                        log(
                            f"Successfully loaded flat TIFF {os.path.basename(path)} with PIL. Shape: {arr.shape if arr is not None else 'None'}"
                        )
                    except Exception as pil_error_after_aics:
                        log(
                            f"PIL failed for TIFF {os.path.basename(path)} after AICSImage check (Z={z_dim_for_log}): {pil_error_after_aics}. Using AICSImage data.",
                            level="WARNING",
                        )
                        # Stick with AICSImage data since PIL failed, ensure aics_image_obj_for_model is still set
                        if not aics_image_obj_for_model:
                            aics_image_obj_for_model = (
                                img_aics  # Should still be img_aics from outer try
                            )
                            log(
                                f"Reinstated aics_image_obj_for_model for {os.path.basename(path)}",
                                level="DEBUG",
                            )
                        arr, scale_conversion = self._process_aics_image_data(
                            img_aics, path, selected_channel_indices, z_selection_params
                        )
                        log(
                            f"Processed with AICSImage after PIL fallback failure for {os.path.basename(path)}. Shape: {arr.shape if arr is not None else 'None'}",
                            level="DEBUG",
                        )

            except Exception as aics_initial_error:
                log(
                    f"AICSImage failed for TIFF {os.path.basename(path)} on initial load: {aics_initial_error}. Falling back to PIL.",
                    level="WARNING",
                )
                aics_image_obj_for_model = None  # Ensure it's None if AICS fails here
                try:
                    img_pil = Image.open(path)
                    arr = np.array(img_pil)
                    scale_conversion = None  # Reset as PIL is primary now
                    log(
                        f"Successfully loaded TIFF {os.path.basename(path)} with PIL after AICSImage initial failure. Shape: {arr.shape if arr is not None else 'None'}"
                    )
                except Exception as pil_final_error:
                    log(
                        f"PIL also failed for TIFF {os.path.basename(path)} after AICSImage failed: {pil_final_error}",
                        level="ERROR",
                    )
                    raise pil_final_error  # Re-raise the PIL error to be caught by load_image

        elif ext == ".lif":  # LIF files: Try AICSImage or LASX fix
            if constants.LIF_LASX_FIX:
                log(
                    f"LASX LIF fix enabled: Loading LIF {os.path.basename(path)} with LASXLifImage.",
                    level="INFO",
                )
                try:
                    lasx_img = LASXLifImage(path)
                    aics_image_obj_for_model = lasx_img
                    arr = lasx_img.get_image_data("ZYX", C=0)[0]  # (Y, X) of first Z
                    log(
                        f"LASX LIF fix: get_image_data('ZYX', C=0) returned array shape={lasx_img.get_image_data('ZYX', C=0).shape}, dtype={lasx_img.get_image_data('ZYX', C=0).dtype}, min={lasx_img.get_image_data('ZYX', C=0).min()}, max={lasx_img.get_image_data('ZYX', C=0).max()}",
                        level="DEBUG",
                    )
                    log(
                        f"LASX LIF fix: Using Z=0 for display, arr shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}",
                        level="DEBUG",
                    )
                    # Extract scale using AICSImage for metadata only
                    try:
                        aics_for_scale = AICSImage(path)
                        scale_conversion = aics_for_scale.physical_pixel_sizes
                        log(
                            f"LASX LIF fix: Extracted scale_conversion from AICSImage: {scale_conversion}",
                            level="INFO",
                        )
                    except Exception as scale_e:
                        scale_conversion = None
                        log(
                            f"LASX LIF fix: Failed to extract scale_conversion from AICSImage: {scale_e}",
                            level="WARNING",
                        )
                    log(
                        "LASX LIF fix: Converting array to PIL Image for display.",
                        level="DEBUG",
                    )
                    log(
                        f"Successfully loaded LIF {os.path.basename(path)} with LASXLifImage. Shape: {arr.shape if arr is not None else 'None'}",
                        level="DEBUG",
                    )
                    try:
                        pil_img = Image.fromarray(arr)
                        log(
                            f"LASX LIF fix: PIL image created. Mode: {pil_img.mode}, Size: {pil_img.size}",
                            level="DEBUG",
                        )
                    except Exception as pil_e:
                        log(
                            f"LASX LIF fix: Error converting array to PIL Image: {pil_e}",
                            level="ERROR",
                        )
                except Exception as e:
                    log(
                        f"LASXLifImage failed for LIF {os.path.basename(path)}: {e}.",
                        level="ERROR",
                    )
                    raise e
            else:
                log(
                    f"Attempting to load LIF {os.path.basename(path)} with AICSImage.",
                    level="INFO",
                )
                try:
                    img_aics = AICSImage(path)
                    aics_image_obj_for_model = img_aics
                    arr, scale_conversion = self._process_aics_image_data(
                        img_aics, path, selected_channel_indices, z_selection_params
                    )
                    log(
                        f"Successfully processed LIF {os.path.basename(path)} with AICSImage. Shape: {arr.shape if arr is not None else 'None'}",
                        level="DEBUG",
                    )
                except Exception as e:
                    log(
                        f"AICSImage failed for LIF {os.path.basename(path)}: {e}. LIF files are not typically readable by PIL.",
                        level="ERROR",
                    )
                    raise e  # Re-raise to be caught by load_image
        else:
            # For other extensions (jpg, png, etc.), use PIL
            log(
                f"Loading {os.path.basename(path)} (ext: {ext}) with PIL.", level="INFO"
            )
            try:
                log(f"Loading {os.path.basename(path)} with PIL.")
                img_pil = Image.open(path)
                arr = np.array(img_pil)
                log(
                    f"Successfully loaded {os.path.basename(path)} with PIL. Shape: {arr.shape if arr is not None else 'None'}",
                    level="DEBUG",
                )
            except Exception as e:
                log(f"PIL failed for {os.path.basename(path)}: {e}", level="ERROR")
                raise e  # Re-raise to be caught by load_image

        if arr is None:
            log(
                f"Array is None after all loading attempts for {os.path.basename(path)}",
                level="ERROR",
            )
            raise ValueError(f"Failed to load image array for {os.path.basename(path)}")

        log(
            f"read_image_to_array: Raw loaded array for {os.path.basename(path)}. Shape: {arr.shape}, Dtype: {arr.dtype}",
            level="DEBUG",
        )

        # Normalize array to be 3-channel uint8 for display
        if arr.ndim == 2:  # Grayscale
            log(
                f"read_image_to_array: Normalizing 2D grayscale for {os.path.basename(path)}.",
                level="DEBUG",
            )
            arr = self._normalize_grayscale_to_uint8(arr, path)
            arr = np.stack((arr,) * 3, axis=-1)
            log(
                f"Converted 2D array to 3-channel uint8 for {os.path.basename(path)}. Shape: {arr.shape}"
            )
        elif arr.ndim == 3:
            if arr.shape[2] == 1:  # Grayscale in 3D shape (e.g., C=1)
                log(
                    f"read_image_to_array: Normalizing 3D (C=1) grayscale for {os.path.basename(path)}.",
                    level="DEBUG",
                )
                arr = arr[:, :, 0]  # Reduce to 2D
                arr = self._normalize_grayscale_to_uint8(arr, path)
                arr = np.stack((arr,) * 3, axis=-1)
                log(
                    f"Converted 3D (C=1) array to 3-channel uint8 for {os.path.basename(path)}. Shape: {arr.shape}"
                )
            elif arr.shape[2] == 4:  # RGBA
                log(
                    f"RGBA image {os.path.basename(path)}. Converting to RGB by dropping alpha."
                )
                arr = arr[:, :, :3]  # Drop alpha, arr is now HWC with C=3
                log(
                    f"read_image_to_array: Converted RGBA to RGB for {os.path.basename(path)}. Shape: {arr.shape}",
                    level="DEBUG",
                )
                # Now arr falls into the C=3 case below for dtype check

            # After potential conversion from RGBA or single channel, if arr is 3-channel, ensure uint8
            if arr.ndim == 3 and arr.shape[2] == 3:  # Now arr should be 3-channel
                if arr.dtype != np.uint8:
                    log(
                        f"read_image_to_array: Normalizing 3-channel non-uint8 for {os.path.basename(path)}. Current dtype: {arr.dtype}",
                        level="DEBUG",
                    )
                    arr = self._normalize_3channel_to_uint8(arr, path)
                log(
                    f"Processed 3-channel array to uint8 for {os.path.basename(path)}. Shape: {arr.shape}, dtype: {arr.dtype}"
                )
            elif not (
                arr.ndim == 3 and arr.shape[2] == 3
            ):  # Should not happen if logic above is correct
                log(
                    f"Unexpected array shape after initial processing: {arr.shape} for {os.path.basename(path)}",
                    level="ERROR",
                )
                raise ValueError(
                    f"Unexpected array shape for display: {arr.shape}. Expected 3-channel RGB."
                )
        else:  # ndim not 2 or 3
            log(
                f"Unsupported image dimensions for display: {arr.shape} for {os.path.basename(path)}",
                level="ERROR",
            )
            raise ValueError(
                f"Unsupported image dimensions: {arr.shape}. Expected 2D grayscale or 3/4-channel color."
            )

        log(
            f"read_image_to_array returning array of shape {arr.shape}, dtype {arr.dtype} for {os.path.basename(path)}"
        )
        return arr, scale_conversion, aics_image_obj_for_model
