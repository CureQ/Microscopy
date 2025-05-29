import io
import os
from tkinter import filedialog

import customtkinter as ctk
import numpy as np
from aicsimageio import AICSImage
from CTkMessagebox import CTkMessagebox
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from . import constants
from .application_model import ApplicationModel
from .image_overlay_processor import ImageOverlayProcessor


class FileIOController:
    def __init__(
        self,
        parent_frame,
        application_model_ref: ApplicationModel,
        display_settings_controller_ref=None,
    ):
        self.parent_frame = parent_frame  # cell_body_frame instance
        self.application_model = application_model_ref
        self.display_settings_controller = display_settings_controller_ref
        self.overlay_processor = ImageOverlayProcessor(application_model_ref)

    def _read_image_to_array(
        self,
        path,
        selected_channel_indices: list[int] | None = None,
        z_selection_params: dict | None = None,
    ):
        ext = os.path.splitext(path)[1].lower()
        arr = None
        if ext in [".lif", ".tiff", ".tif"]:
            try:
                img_aics = AICSImage(path)

                # Determine channels to load
                channels_to_load = []
                if selected_channel_indices:
                    channels_to_load = [
                        c for c in selected_channel_indices if 0 <= c < img_aics.dims.C
                    ]
                    if (
                        not channels_to_load and img_aics.dims.C > 0
                    ):  # Invalid user selection, fallback
                        print(
                            f"Warning: Invalid selected_channel_indices ({selected_channel_indices}) for {path}. Defaulting to channel 0."
                        )
                        channels_to_load = [0]
                else:  # Default channel selection (primarily channel 0)
                    if img_aics.dims.C > 0:
                        channels_to_load = [0]
                        print(
                            f"_read_image_to_array: No channels selected by user for {path}, defaulting to channel 0."
                        )
                    else:
                        print(
                            f"_read_image_to_array: No channels available or selectable for {path}."
                        )

                if not channels_to_load:
                    print(
                        f"_read_image_to_array: No valid channels to load for {path} after processing selections. Available C: {img_aics.dims.C}"
                    )
                    raise ValueError(
                        f"No valid channels to load for {path} (available C: {img_aics.dims.C})."
                    )
                print(
                    f"_read_image_to_array: Determined channels to load for {path}: {channels_to_load}"
                )

                # Process Z dimension for each selected channel
                processed_channels_data = []
                for c_idx in channels_to_load:
                    # Get full Z-stack for the current channel
                    # AICSImage loads ZYX C T S order, but we request specific C, S, T.
                    # Here we get "ZYX" for a specific channel.
                    channel_data_z_stack = img_aics.get_image_data(
                        "ZYX", S=0, T=0, C=c_idx
                    )

                    current_z_dim_size = img_aics.dims.Z
                    default_z_index = (
                        current_z_dim_size // 2 if current_z_dim_size > 1 else 0
                    )

                    processed_c_data_for_channel = None
                    if current_z_dim_size > 1 and z_selection_params:
                        z_type = z_selection_params.get("type")
                        if z_type == "max_project":
                            processed_c_data_for_channel = np.max(
                                channel_data_z_stack, axis=0
                            )
                        elif z_type == "mean_project":
                            processed_c_data_for_channel = np.mean(
                                channel_data_z_stack, axis=0
                            )
                        elif z_type == "slice":
                            z_idx_val = z_selection_params.get("value", default_z_index)
                            target_z_slice = max(
                                0, min(z_idx_val, current_z_dim_size - 1)
                            )
                            print(
                                f"_read_image_to_array: Processing channel {c_idx}, Z-slice: {target_z_slice} (requested: {z_idx_val})"
                            )
                            processed_c_data_for_channel = channel_data_z_stack[
                                target_z_slice, :, :
                            ]
                        else:  # Unknown type, fallback to default Z slice
                            print(
                                f"_read_image_to_array: Processing channel {c_idx}, Z-slice: {default_z_index} (unknown Z type '{z_type}', fallback to default)."
                            )
                            processed_c_data_for_channel = channel_data_z_stack[
                                default_z_index, :, :
                            ]
                    else:  # Single Z-slice in image or no specific Z processing requested
                        print(
                            f"_read_image_to_array: Processing channel {c_idx}, Z-slice: {default_z_index} (single Z or no Z params)."
                        )
                        processed_c_data_for_channel = channel_data_z_stack[
                            default_z_index, :, :
                        ]

                    processed_channels_data.append(processed_c_data_for_channel)

                # Combine processed channels into 'arr'
                num_loaded_channels = len(processed_channels_data)
                if num_loaded_channels == 0:
                    # This case should ideally be caught by the 'if not channels_to_load' check earlier
                    print(
                        f"_read_image_to_array: ERROR - No channel data was processed for {path} despite earlier checks."
                    )
                    raise ValueError(f"No channel data processed for {path}")
                elif num_loaded_channels == 1:
                    arr = processed_channels_data[0]  # 2D array
                    print(
                        f"_read_image_to_array: Single channel {channels_to_load[0]} loaded for {path}."
                    )
                elif num_loaded_channels == 2:
                    # Combine 2 channels into a 3-channel image (e.g., C1 in Red, C2 in Green, Blue is zero)
                    ch1_data = processed_channels_data[0]
                    ch2_data = processed_channels_data[1]
                    if ch1_data.shape != ch2_data.shape:
                        print(
                            f"_read_image_to_array: ERROR - Shape mismatch for 2-channel stacking in {path}. Ch1: {ch1_data.shape}, Ch2: {ch2_data.shape}"
                        )
                        raise ValueError(
                            "Shape mismatch between processed channels for 2-channel stacking."
                        )
                    zeros_channel = np.zeros_like(ch1_data, dtype=ch1_data.dtype)
                    arr = np.stack((ch1_data, ch2_data, zeros_channel), axis=-1)
                    print(
                        f"_read_image_to_array: Two channels {channels_to_load[:2]} loaded and stacked into RGB for {path}."
                    )
                elif num_loaded_channels == 3:
                    arr = np.stack(processed_channels_data, axis=-1)
                    print(
                        f"_read_image_to_array: Three channels {channels_to_load[:3]} loaded and stacked into RGB for {path}."
                    )
                else:  # >= 4 channels selected by user (or somehow loaded)
                    # Take the first 3 channels for an RGB representation
                    arr = np.stack(processed_channels_data[:3], axis=-1)
                    print(
                        f"_read_image_to_array: {num_loaded_channels} channels loaded for {path}, taking first 3 {channels_to_load[:3]} for RGB."
                    )

            except Exception as e:
                print(f"AICSImage failed for {path}: {e}. Falling back to PIL.")
                img_pil = Image.open(path)
                arr = np.array(img_pil)
        else:
            img_pil = Image.open(path)
            arr = np.array(img_pil)

        if arr is not None:
            if arr.ndim == 2:
                if arr.dtype != np.uint8:
                    arr = (
                        (arr / arr.max() * 255).astype(np.uint8)
                        if arr.max() > 0
                        else arr.astype(np.uint8)
                    )
                arr = np.stack((arr,) * 3, axis=-1) if len(arr.shape) == 2 else arr
            elif arr.ndim == 3:
                if arr.shape[2] == 1:
                    arr = arr[:, :, 0]
                    if arr.dtype != np.uint8:
                        arr = (
                            (arr / arr.max() * 255).astype(np.uint8)
                            if arr.max() > 0
                            else arr.astype(np.uint8)
                        )
                    arr = np.stack((arr,) * 3, axis=-1)
                elif arr.shape[2] == 4:
                    arr = arr[:, :, :3]
                if arr.dtype != np.uint8 and arr.shape[2] == 3:
                    arr_max = np.percentile(arr, 99.9)  # Robust max for normalization
                    arr_min = np.percentile(arr, 0.1)  # Robust min
                    arr = (
                        np.clip((arr - arr_min) / (arr_max - arr_min + 1e-6), 0, 1)
                        * 255
                    )  # Normalize and scale
                    arr = arr.astype(np.uint8)
            else:
                raise ValueError(
                    f"Unsupported image dimensions: {arr.shape} for {path}"
                )
        print("arr returned")
        return arr

    def _prompt_channel_z_selection(self, img_dims, parent_window):
        print(
            f"Prompting for channel/Z selection. Image Dims C:{img_dims.C}, Z:{img_dims.Z}"
        )
        dialog = ctk.CTkToplevel(parent_window)
        dialog.title("Image Load Options")
        dialog.transient(parent_window)
        dialog.attributes("-topmost", True)

        results = {"channels": None, "z_params": None, "ok_pressed": False}

        main_frame = ctk.CTkFrame(dialog)
        main_frame.pack(padx=20, pady=20, expand=True, fill="both")

        # --- Channels Frame ---
        channel_vars = []
        if img_dims.C > 0:
            channels_frame = ctk.CTkFrame(main_frame)
            channels_frame.pack(pady=(0, 10), padx=10, fill="x")
            ctk.CTkLabel(
                channels_frame, text=f"Select Channels (Available: {img_dims.C}):"
            ).pack(anchor="w")
            ctk.CTkLabel(
                channels_frame,
                text="(Max 3 for RGB, 1 for grayscale will be used from selection)",
                font=("", 10),
            ).pack(anchor="w")

            checkbox_frame = ctk.CTkScrollableFrame(
                channels_frame, height=min(img_dims.C * 30, 120)
            )  # Limit height
            checkbox_frame.pack(fill="x", expand=True)

            for i in range(img_dims.C):
                var = ctk.IntVar(
                    value=1 if i == 0 else 0
                )  # Default select first channel
                cb = ctk.CTkCheckBox(
                    checkbox_frame, text=f"Channel {i + 1}", variable=var
                )
                cb.pack(anchor="w", padx=5)
                channel_vars.append(var)
        else:
            ctk.CTkLabel(main_frame, text="No channels detected in image.").pack(pady=5)

        # --- Z-Stack Frame ---
        z_slice_entry_var = ctk.StringVar()
        z_slice_entry = None  # To control its state

        if img_dims.Z > 1:
            z_stack_frame = ctk.CTkFrame(main_frame)
            z_stack_frame.pack(pady=(0, 10), padx=10, fill="x")
            ctk.CTkLabel(
                z_stack_frame,
                text=f"Z-Stack Processing (Available Slices: {img_dims.Z}):",
            ).pack(anchor="w")

            z_processing_var = ctk.StringVar(value="slice_middle")  # Default value

            def toggle_z_slice_entry_state():
                if z_processing_var.get() == "slice_specific":
                    if z_slice_entry:
                        z_slice_entry.configure(state="normal")
                else:
                    if z_slice_entry:
                        z_slice_entry.configure(state="disabled")

            ctk.CTkRadioButton(
                z_stack_frame,
                text="Middle Slice",
                variable=z_processing_var,
                value="slice_middle",
                command=toggle_z_slice_entry_state,
            ).pack(anchor="w")
            ctk.CTkRadioButton(
                z_stack_frame,
                text="Maximum Intensity Projection",
                variable=z_processing_var,
                value="max_project",
                command=toggle_z_slice_entry_state,
            ).pack(anchor="w")
            ctk.CTkRadioButton(
                z_stack_frame,
                text="Mean Intensity Projection",
                variable=z_processing_var,
                value="mean_project",
                command=toggle_z_slice_entry_state,
            ).pack(anchor="w")

            specific_slice_frame = ctk.CTkFrame(z_stack_frame)  # No internal padding
            specific_slice_frame.pack(fill="x", anchor="w")
            ctk.CTkRadioButton(
                specific_slice_frame,
                text="Specific Slice (0-indexed):",
                variable=z_processing_var,
                value="slice_specific",
                command=toggle_z_slice_entry_state,
            ).pack(side="left", anchor="w")

            z_slice_entry = ctk.CTkEntry(
                specific_slice_frame,
                textvariable=z_slice_entry_var,
                width=60,
                state="disabled",
            )
            z_slice_entry.pack(side="left", padx=(5, 0), anchor="w")
            z_slice_entry_var.set(
                str(img_dims.Z // 2)
            )  # Default to middle slice number

        # --- Buttons Frame ---
        buttons_frame = ctk.CTkFrame(main_frame)  # No internal padding
        buttons_frame.pack(pady=(10, 0), fill="x", side="bottom")

        def _on_ok():
            selected_channels_indices = []
            if img_dims.C > 0:
                selected_channels_indices = [
                    i for i, var in enumerate(channel_vars) if var.get() == 1
                ]
                print(
                    f"_prompt_channel_z_selection: User selected channel indices: {selected_channels_indices}"
                )
                if (
                    not selected_channels_indices
                ):  # If user deselects all, default to channel 0
                    CTkMessagebox(
                        title="Info",
                        message="No channels selected. Defaulting to Channel 1.",
                        icon="info",
                        parent=dialog,
                    )
                    selected_channels_indices = [0]
                    print(
                        "_prompt_channel_z_selection: No channels selected by user, defaulting to [0]."
                    )

                # Warn if more than 3 selected, but allow it (backend handles it)
                if len(selected_channels_indices) > 3:
                    CTkMessagebox(
                        title="Info",
                        message="More than 3 channels selected. The first 3 will be prioritized for RGB display.",
                        icon="info",
                        parent=dialog,
                    )
                    print(
                        f"_prompt_channel_z_selection: User selected {len(selected_channels_indices)} channels (more than 3). Prioritizing first 3."
                    )

            current_z_params = None
            if img_dims.Z > 1:
                z_choice = z_processing_var.get()
                print(
                    f"_prompt_channel_z_selection: User selected Z-processing: {z_choice}"
                )
                if z_choice == "slice_middle":
                    current_z_params = {"type": "slice", "value": img_dims.Z // 2}
                elif z_choice == "max_project":
                    current_z_params = {"type": "max_project"}
                elif z_choice == "mean_project":
                    current_z_params = {"type": "mean_project"}
                elif z_choice == "slice_specific":
                    try:
                        slice_val = int(z_slice_entry_var.get())
                        print(
                            f"_prompt_channel_z_selection: User specified Z-slice: {slice_val}"
                        )
                        if 0 <= slice_val < img_dims.Z:
                            current_z_params = {"type": "slice", "value": slice_val}
                        else:
                            CTkMessagebox(
                                title="Error",
                                message=f"Specific Z-slice must be between 0 and {img_dims.Z - 1}.",
                                icon="cancel",
                                parent=dialog,
                            )
                            print(
                                f"_prompt_channel_z_selection: Invalid Z-slice {slice_val}. Range is 0-{img_dims.Z - 1}."
                            )
                            return
                    except ValueError:
                        CTkMessagebox(
                            title="Error",
                            message="Invalid Z-slice number. Please enter a whole number.",
                            icon="cancel",
                            parent=dialog,
                        )
                        print(
                            f"_prompt_channel_z_selection: ValueError for Z-slice input: '{z_slice_entry_var.get()}'."
                        )
                        return
            elif img_dims.Z == 1:  # Single Z slice
                current_z_params = {"type": "slice", "value": 0}
                print("_prompt_channel_z_selection: Single Z-slice image, using Z=0.")
            # If img_dims.Z is 0 or less (should not happen for valid images), z_params remains None

            results["channels"] = selected_channels_indices
            results["z_params"] = current_z_params
            results["ok_pressed"] = True
            print(
                f"_prompt_channel_z_selection: OK pressed. Channels: {selected_channels_indices}, Z_params: {current_z_params}"
            )
            dialog.destroy()

        def _on_cancel():
            results["ok_pressed"] = False
            print("_prompt_channel_z_selection: Cancel pressed.")
            dialog.destroy()

        ok_button = ctk.CTkButton(buttons_frame, text="OK", command=_on_ok)
        ok_button.pack(side="right", padx=(10, 0))
        cancel_button = ctk.CTkButton(
            buttons_frame, text="Cancel", command=_on_cancel, fg_color="gray"
        )
        cancel_button.pack(side="right")

        dialog.protocol("WM_DELETE_WINDOW", _on_cancel)  # Handle window close button
        dialog.grab_set()  # Ensure modal
        parent_window.wait_window(dialog)  # Wait for dialog to close

        if results["ok_pressed"]:
            return results["channels"], results["z_params"]
        else:
            return None, None

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=constants.MICROSCOPY_IMG_FILETYPES
        )
        if not file_path:
            print("Load image: File dialog cancelled.")
            return

        print(f"Load image: File selected - {file_path}")
        base_filename_local = os.path.splitext(os.path.basename(file_path))[0]
        if self.parent_frame.filename_label:
            self.parent_frame.filename_label.configure(text=base_filename_local)
            print(f"Load image: Filename label updated to {base_filename_local}")

        img_data_arr = None
        s_ch_indices = None
        z_params = None

        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext in [".lif", ".tiff", ".tif"]:
                # Partially load to get dimensions for dialog
                try:
                    img_aics_dims_check = AICSImage(file_path)
                    dims = img_aics_dims_check.dims
                except Exception as e:
                    # Fallback if AICSImage fails even to read dims
                    CTkMessagebox(
                        title=constants.MSG_ERROR_LOADING_IMAGE_TITLE,
                        message=f"Could not read image metadata: {str(e)}. Attempting direct load.",
                        icon="warning",
                    )
                    print(
                        f"Load image: AICSImage failed to read metadata for {file_path}: {e}. Attempting direct load."
                    )
                    # Try to load with defaults if metadata read fails
                    img_data_arr = self._read_image_to_array(file_path, None, None)
                else:
                    if dims.C > 1 or dims.Z > 1:
                        print(
                            f"Load image: Multi-channel/Z-stack image ({file_path}). Prompting for selection."
                        )
                        s_ch_indices, z_params = self._prompt_channel_z_selection(
                            dims, self.parent_frame.winfo_toplevel()
                        )
                        if (
                            s_ch_indices is None and z_params is None
                        ):  # Dialog was cancelled
                            print(
                                f"Load image: Channel/Z selection dialog was cancelled for {file_path}."
                            )
                            return
                    else:  # Single channel, single Z, or no dialog needed
                        s_ch_indices = [0] if dims.C > 0 else []
                        z_params = (
                            {"type": "slice", "value": 0} if dims.Z > 0 else None
                        )  # Default z_params for single Z or no Z
                        print(
                            f"Load image: Single channel/Z-slice image or no selection needed for {file_path}. Using C:{s_ch_indices}, Z:{z_params}"
                        )

                    if (
                        s_ch_indices is not None
                    ):  # Proceed if not cancelled (even if defaults were set without dialog)
                        print(
                            f"Load image: Reading image data for {file_path} with C:{s_ch_indices}, Z:{z_params}"
                        )
                        img_data_arr = self._read_image_to_array(
                            file_path, s_ch_indices, z_params
                        )

            else:  # For other image types
                print(f"Load image: Reading standard image type {file_path}.")
                img_data_arr = self._read_image_to_array(file_path, None, None)

            if img_data_arr is None or img_data_arr.size == 0:
                # Check if img_data_arr was actually loaded (e.g. dialog cancel or actual read failure)
                if (
                    ext in [".lif", ".tiff", ".tif"]
                    and s_ch_indices is None
                    and z_params is None
                ):
                    # This means dialog was cancelled, and we already returned, so this path shouldn't be hit often.
                    # But as a safeguard if logic changes.
                    print(
                        f"Load image: Image loading cancelled or failed for {file_path} (dialog related)."
                    )
                    return
                print(
                    f"Load image: Failed to read image or image is empty for {file_path} after selection/processing."
                )
                raise ValueError(
                    "Failed to read image or image is empty after selection/processing."
                )

            pil_image = (
                Image.fromarray(img_data_arr)
                if not isinstance(img_data_arr, Image.Image)
                else img_data_arr
            )
            print(f"Loading image: {file_path} into ApplicationModel.")
            self.application_model.load_new_image(
                pil_image, file_path, base_filename_local
            )

        except Exception as e:
            print(f"Load image: Exception occurred while loading {file_path}: {str(e)}")
            CTkMessagebox(
                title=constants.MSG_ERROR_LOADING_IMAGE_TITLE,
                message=str(e),
                icon="cancel",
            )
            self.application_model.image_data.reset()
            self.application_model.current_file_path = None
            self.application_model.base_filename = None
            self.application_model.notify_observers("image_load_failed")

            if self.parent_frame.segment_button:
                self.parent_frame.segment_button.configure(state="disabled")

    def export_selected(self):
        if self.application_model.image_data.mask_array is None:
            CTkMessagebox(
                title=constants.MSG_EXPORT_ERROR_TITLE,
                message=constants.MSG_EXPORT_NO_MASK,
                icon="cancel",
            )
            print("Export selected cells: No mask array found.")
            return

        export_mask = np.zeros_like(
            self.application_model.image_data.mask_array,
            dtype=self.application_model.image_data.mask_array.dtype,
        )
        for cid in self.application_model.image_data.included_cells:
            export_mask[self.application_model.image_data.mask_array == cid] = cid

        save_path = filedialog.asksaveasfilename(
            initialfile=f"{self.application_model.base_filename}{constants.PDF_SELECTED_CELLS_FILENAME_PREFIX}",
            defaultextension=".tif",
            filetypes=constants.EXPORT_FILETYPES_TIF_NUMPY,
        )
        if save_path:
            try:
                print(f"Export selected cells: Saving to {save_path}")
                if save_path.endswith(".npy"):
                    np.save(save_path, export_mask)
                else:
                    Image.fromarray(export_mask).save(save_path)
                CTkMessagebox(
                    title=constants.MSG_EXPORT_SUCCESS_TITLE,
                    message=constants.MSG_EXPORT_COMPLETED,
                    icon="check",
                )
                print("Export selected cells: Success.")
            except Exception as e:
                CTkMessagebox(
                    title=constants.MSG_EXPORT_ERROR_TITLE,
                    message=constants.MSG_EXPORT_FAILED.format(error=str(e)),
                    icon="cancel",
                )
                print(f"Export selected cells: Failed - {str(e)}")
        else:
            print("Export selected cells: Save dialog cancelled.")

    def export_pdf(self):
        if self.application_model.image_data.original_image is None:
            CTkMessagebox(
                title=constants.MSG_PDF_EXPORT_ERROR_TITLE,
                message=constants.MSG_PDF_ORIGINAL_IMAGE_MISSING,
                icon="cancel",
            )
            print("Export PDF: Original image missing.")
            return

        save_path = filedialog.asksaveasfilename(
            initialfile=f"{self.application_model.base_filename}{constants.PDF_DEFAULT_FILENAME_PREFIX}",
            defaultextension=".pdf",
            filetypes=constants.EXPORT_FILETYPES_PDF,
        )
        if not save_path:
            print("Export PDF: Save dialog cancelled.")
            return

        print(f"Export PDF: Starting PDF export to {save_path}")
        TARGET_DPI = constants.PDF_TARGET_DPI

        try:
            img_original_pil = self.application_model.image_data.original_image
            mask_array = self.application_model.image_data.mask_array

            all_cell_ids_in_mask = (
                np.unique(mask_array[mask_array > 0])
                if mask_array is not None
                else np.array([])
            )
            selected_cell_ids_set = self.application_model.image_data.included_cells

            selected_cell_ids = list(selected_cell_ids_set)
            deselected_cell_ids = list(
                set(all_cell_ids_in_mask) - selected_cell_ids_set
            )

            # PDF setup
            pdf_canvas = canvas.Canvas(save_path, pagesize=A4)
            width, height = A4  # Page dimensions

            # --- Get the base image for PDF (potentially processed) ---
            current_original_image_pil = (
                self.application_model.get_processed_image_for_display()
            )

            if current_original_image_pil is None:  # Fallback to true original
                current_original_image_pil = (
                    self.application_model.image_data.original_image
                )

            # Check if the (potentially processed) original image is available, especially if it's requested for the PDF
            if (
                current_original_image_pil is None
                and self.parent_frame.pdf_opt_include_original_image.get()
            ):
                CTkMessagebox(
                    title=constants.MSG_PDF_EXPORT_ERROR_TITLE,
                    message=constants.MSG_PDF_ORIGINAL_IMAGE_MISSING,
                    icon="warning",
                )
                print(
                    "Export PDF: Original image (current_original_image_pil) is missing but was requested for PDF."
                )
                # Not returning here, to allow other PDF parts if possible

            # Title
            pdf_canvas.setFont(constants.PDF_FONT_TITLE, constants.PDF_FONT_TITLE_SIZE)
            pdf_canvas.drawString(
                constants.PDF_MARGIN,
                height - constants.PDF_MARGIN,
                "Cell Segmentation Report",
            )

            y_pos_stats = height - constants.PDF_MARGIN - constants.PDF_STATS_TOP_OFFSET
            details = [
                f"Source File: {self.application_model.base_filename}",
                f"Diameter parameter (gui): {self.parent_frame.dia_entry.get() if self.parent_frame.dia_entry else 'N/A'}",
                f"Detected cells: {len(np.unique(self.application_model.image_data.mask_array)) - 1 if self.application_model.image_data.mask_array is not None and self.application_model.image_data.mask_array.size > 0 else 0}",
                f"Selected cells: {len(self.application_model.image_data.included_cells)}",
            ]
            pdf_canvas.setFont(constants.PDF_FONT_BODY, constants.PDF_FONT_BODY_SIZE)
            for detail in details:
                pdf_canvas.drawString(constants.PDF_MARGIN, y_pos_stats, detail)
                y_pos_stats -= constants.PDF_LINE_HEIGHT

            img_title_space = (
                constants.PDF_LINE_HEIGHT * constants.PDF_TITLE_IMAGE_GAP_FACTOR
            )
            max_img_height_page1 = y_pos_stats - constants.PDF_MARGIN - img_title_space
            max_img_width_page1 = width - 2 * constants.PDF_MARGIN

            if current_original_image_pil:
                orig_pix_w, orig_pix_h = current_original_image_pil.size

                aspect_ratio = orig_pix_w / orig_pix_h if orig_pix_h > 0 else 1
                pdf_w_pts, pdf_h_pts = 0, 0
                if (max_img_width_page1 / aspect_ratio) <= max_img_height_page1:
                    pdf_w_pts = max_img_width_page1
                    pdf_h_pts = max_img_width_page1 / aspect_ratio
                else:
                    pdf_h_pts = max_img_height_page1
                    pdf_w_pts = max_img_height_page1 * aspect_ratio

                render_pix_w = int(pdf_w_pts * TARGET_DPI / 72.0)
                render_pix_h = int(pdf_h_pts * TARGET_DPI / 72.0)

                if render_pix_w > 0 and render_pix_h > 0:
                    resized_img_page1 = current_original_image_pil.resize(
                        (render_pix_w, render_pix_h), Image.LANCZOS
                    )
                    orig_buf = io.BytesIO()
                    resized_img_page1.save(
                        orig_buf,
                        format=constants.PDF_IMAGE_EXPORT_FORMAT,
                        dpi=(TARGET_DPI, TARGET_DPI),
                    )
                    orig_buf.seek(0)

                    # Corrected X coordinate for centering within content area
                    content_area_width = width - 2 * constants.PDF_MARGIN
                    img_x_page1 = (
                        constants.PDF_MARGIN + (content_area_width - pdf_w_pts) / 2
                    )
                    img_y_page1 = y_pos_stats - pdf_h_pts - img_title_space

                    pdf_canvas.drawImage(
                        ImageReader(orig_buf),
                        img_x_page1,
                        img_y_page1,
                        width=pdf_w_pts,
                        height=pdf_h_pts,
                    )
                    pdf_canvas.setFont(
                        constants.PDF_FONT_BODY, constants.PDF_FONT_SUBHEADER_SIZE
                    )
                    pdf_canvas.drawCentredString(
                        width / 2,
                        img_y_page1 - constants.PDF_LINE_HEIGHT,
                        "Original Image",
                    )
            pdf_canvas.showPage()

            images_to_export = []
            selected_cell_ids = self.application_model.image_data.included_cells.copy()

            if (
                self.application_model.image_data.mask_array is None
                or self.application_model.image_data.mask_array.size == 0
            ):
                any_overlay_option_selected = (
                    self.application_model.display_state.pdf_opt_masks_only
                    or self.application_model.display_state.pdf_opt_boundaries_only
                    or self.application_model.display_state.pdf_opt_numbers_only
                    or self.application_model.display_state.pdf_opt_masks_boundaries
                    or self.application_model.display_state.pdf_opt_masks_numbers
                    or self.application_model.display_state.pdf_opt_boundaries_numbers
                    or self.application_model.display_state.pdf_opt_masks_boundaries_numbers
                )
                if any_overlay_option_selected:
                    pdf_canvas.setFont(
                        constants.PDF_FONT_BODY, constants.PDF_FONT_BODY_SIZE
                    )
                    pdf_canvas.drawCentredString(
                        width / 2,
                        height / 2,
                        "Overlay images selected for PDF,",
                    )
                    pdf_canvas.drawCentredString(
                        width / 2,
                        height / 2 - constants.PDF_LINE_HEIGHT,
                        "but no cell segmentation data is available.",
                    )
                    pdf_canvas.showPage()
            else:
                # --- Image for Masks/Boundaries/Numbers Page ---
                # This will be the base onto which we draw masks, boundaries, numbers for the next page.
                # It should be the same (potentially processed) original image fetched earlier.
                page_base_image_pil = None
                if (
                    current_original_image_pil
                ):  # Use the already fetched (and potentially processed) image
                    page_base_image_pil = current_original_image_pil.copy()
                # If current_original_image_pil was None, page_base_image_pil will also be None.
                # The drawing functions _draw_masks_on_pil, etc., will need to handle a None base_pil_image gracefully
                # (e.g., by not drawing or drawing on a black placeholder if they create one).

                # Combined Image: Masks, Boundaries, Numbers (as selected)
                # Only create this page if there's something to draw (opt_masks, etc. are true)
                # AND if we have some base image to draw upon (page_base_image_pil is not None).
                # The first showPage() after the original image page has already prepared a new blank page.
                if page_base_image_pil and (
                    self.application_model.display_state.pdf_opt_masks_only
                    or self.application_model.display_state.pdf_opt_boundaries_only
                    or self.application_model.display_state.pdf_opt_numbers_only
                    or self.application_model.display_state.pdf_opt_masks_boundaries
                    or self.application_model.display_state.pdf_opt_masks_numbers
                    or self.application_model.display_state.pdf_opt_boundaries_numbers
                    or self.application_model.display_state.pdf_opt_masks_boundaries_numbers
                ):
                    y_coord = (
                        height - constants.PDF_MARGIN - constants.PDF_LINE_HEIGHT
                    )  # Reset y_coord for new page (which is current page)
                    pdf_canvas.setFont(
                        constants.PDF_FONT_BODY, constants.PDF_FONT_SUBHEADER_SIZE
                    )
                    pdf_canvas.drawString(
                        constants.PDF_MARGIN, y_coord, "Overlay Images"
                    )
                    y_coord -= (
                        constants.PDF_LINE_HEIGHT * constants.PDF_IMAGE_GAP_ABOVE_FACTOR
                    )

                    # Determine which cells to include based on "Show Deselected Masks Only"
                    # This logic will be encapsulated or passed to the overlay processor methods if they need it.
                    # For PDF, we generally want to show what the user *selected* for export,
                    # which is `selected_cell_ids` (all included_cells from the model).
                    # If a PDF option is to show *only deselected*, that would be a separate explicit flag.

                    ids_for_pdf_overlay = (
                        selected_cell_ids  # Use the selected cells for PDF overlays
                    )

                    if self.application_model.display_state.pdf_opt_masks_only:
                        # Create a pristine copy of the page base for this specific export item
                        current_export_item_img = page_base_image_pil.copy()
                        mask_layer = self.overlay_processor.draw_masks_on_pil(
                            current_export_item_img,  # For size context
                            ids_for_pdf_overlay,
                        )
                        img = self.overlay_processor.blend_image_with_mask_layer(
                            current_export_item_img,
                            mask_layer,
                            constants.MASK_BLEND_ALPHA,
                        )
                        if img:
                            images_to_export.append(("Cell Masks Only", img))

                    if self.application_model.display_state.pdf_opt_boundaries_only:
                        # Create a pristine copy of the page base
                        current_export_item_img = page_base_image_pil.copy()
                        img = self.overlay_processor.draw_boundaries_on_pil(
                            current_export_item_img, ids_for_pdf_overlay
                        )
                        if img:
                            images_to_export.append(("Cell Boundaries Only", img))

                    if self.application_model.display_state.pdf_opt_numbers_only:
                        current_export_item_img = page_base_image_pil.copy()
                        img = self.overlay_processor.draw_numbers_on_pil(
                            current_export_item_img,
                            ids_for_pdf_overlay,
                            font_size=constants.CELL_NUMBERING_FONT_SIZE_PDF,
                        )
                        if img:
                            images_to_export.append(("Cell Numbers Only", img))

                    if self.application_model.display_state.pdf_opt_masks_boundaries:
                        current_export_item_img = page_base_image_pil.copy()
                        # Apply Masks first by getting layer and blending
                        mask_layer = self.overlay_processor.draw_masks_on_pil(
                            current_export_item_img, ids_for_pdf_overlay
                        )
                        current_export_item_img = (
                            self.overlay_processor.blend_image_with_mask_layer(
                                current_export_item_img,
                                mask_layer,
                                constants.MASK_BLEND_ALPHA,
                            )
                        )
                        # Then apply boundaries to the already mask-blended image
                        current_export_item_img = (
                            self.overlay_processor.draw_boundaries_on_pil(
                                current_export_item_img, ids_for_pdf_overlay
                            )
                        )
                        if current_export_item_img:
                            images_to_export.append(
                                ("Masks & Boundaries", current_export_item_img)
                            )

                    if self.application_model.display_state.pdf_opt_masks_numbers:
                        current_export_item_img = page_base_image_pil.copy()
                        # Apply Masks
                        mask_layer = self.overlay_processor.draw_masks_on_pil(
                            current_export_item_img, ids_for_pdf_overlay
                        )
                        current_export_item_img = (
                            self.overlay_processor.blend_image_with_mask_layer(
                                current_export_item_img,
                                mask_layer,
                                constants.MASK_BLEND_ALPHA,
                            )
                        )
                        # Apply Numbers
                        current_export_item_img = (
                            self.overlay_processor.draw_numbers_on_pil(
                                current_export_item_img,
                                ids_for_pdf_overlay,
                                font_size=constants.CELL_NUMBERING_FONT_SIZE_PDF,
                            )
                        )
                        if current_export_item_img:
                            images_to_export.append(
                                ("Masks & Numbers", current_export_item_img)
                            )

                    if self.application_model.display_state.pdf_opt_boundaries_numbers:
                        current_export_item_img = page_base_image_pil.copy()
                        # Apply Boundaries
                        current_export_item_img = (
                            self.overlay_processor.draw_boundaries_on_pil(
                                current_export_item_img, ids_for_pdf_overlay
                            )
                        )
                        # Apply Numbers
                        current_export_item_img = (
                            self.overlay_processor.draw_numbers_on_pil(
                                current_export_item_img,
                                ids_for_pdf_overlay,
                                font_size=constants.CELL_NUMBERING_FONT_SIZE_PDF,
                            )
                        )
                        if current_export_item_img:
                            images_to_export.append(
                                ("Boundaries & Numbers", current_export_item_img)
                            )

                    if self.application_model.display_state.pdf_opt_masks_boundaries_numbers:
                        current_export_item_img = page_base_image_pil.copy()
                        # Apply Masks
                        mask_layer = self.overlay_processor.draw_masks_on_pil(
                            current_export_item_img, ids_for_pdf_overlay
                        )
                        current_export_item_img = (
                            self.overlay_processor.blend_image_with_mask_layer(
                                current_export_item_img,
                                mask_layer,
                                constants.MASK_BLEND_ALPHA,
                            )
                        )
                        # Apply Boundaries
                        current_export_item_img = (
                            self.overlay_processor.draw_boundaries_on_pil(
                                current_export_item_img, ids_for_pdf_overlay
                            )
                        )
                        # Apply Numbers
                        current_export_item_img = (
                            self.overlay_processor.draw_numbers_on_pil(
                                current_export_item_img,
                                ids_for_pdf_overlay,
                                font_size=constants.CELL_NUMBERING_FONT_SIZE_PDF,
                            )
                        )
                        if current_export_item_img:
                            images_to_export.append(
                                ("Masks, Boundaries & Numbers", current_export_item_img)
                            )

            num_images_to_export = len(images_to_export)
            img_idx = 0
            while img_idx < num_images_to_export:
                pdf_canvas.setFont(
                    constants.PDF_FONT_BODY, constants.PDF_FONT_SUBHEADER_SIZE
                )
                title, pil_img = images_to_export[img_idx]
                reserved_space_for_title_and_gap = (
                    constants.PDF_LINE_HEIGHT * constants.PDF_IMAGE_GAP_ABOVE_FACTOR
                )
                max_img_h_on_page = height - reserved_space_for_title_and_gap
                max_img_w_on_page = width - 2 * constants.PDF_MARGIN
                img_w_orig_pixels, img_h_orig_pixels = pil_img.size
                aspect_ratio_overlay = (
                    img_w_orig_pixels / img_h_orig_pixels
                    if img_h_orig_pixels > 0
                    else 1
                )
                pdf_w_pts_overlay, pdf_h_pts_overlay = 0, 0
                if (max_img_w_on_page / aspect_ratio_overlay) <= max_img_h_on_page:
                    pdf_w_pts_overlay = max_img_w_on_page
                    pdf_h_pts_overlay = max_img_w_on_page / aspect_ratio_overlay
                else:
                    pdf_h_pts_overlay = max_img_h_on_page
                    pdf_w_pts_overlay = max_img_h_on_page * aspect_ratio_overlay

                render_pix_w_overlay = int(pdf_w_pts_overlay * TARGET_DPI / 72.0)
                render_pix_h_overlay = int(pdf_h_pts_overlay * TARGET_DPI / 72.0)

                pil_img_copy_for_page = pil_img.copy()
                if render_pix_w_overlay > 0 and render_pix_h_overlay > 0:
                    resized_pil_overlay = pil_img_copy_for_page.resize(
                        (render_pix_w_overlay, render_pix_h_overlay), Image.LANCZOS
                    )
                    buf = io.BytesIO()
                    resized_pil_overlay.save(
                        buf,
                        format=constants.PDF_IMAGE_EXPORT_FORMAT,
                        dpi=(TARGET_DPI, TARGET_DPI),
                    )
                    buf.seek(0)
                    block_total_height = (
                        pdf_h_pts_overlay + reserved_space_for_title_and_gap
                    )
                    # Corrected X coordinate for centering within content area
                    content_area_width_overlay = (
                        width - 2 * constants.PDF_MARGIN
                    )  # Same as content_area_width
                    draw_x_image = (
                        constants.PDF_MARGIN
                        + (content_area_width_overlay - pdf_w_pts_overlay) / 2
                    )
                    # Y position calculation needs to ensure it's vertically centered too if desired, or placed consistently.
                    draw_y_image = (
                        constants.PDF_MARGIN + (height - block_total_height) / 2
                    )
                    gap_above_image = (
                        constants.PDF_LINE_HEIGHT * constants.PDF_IMAGE_GAP_ABOVE_FACTOR
                    )
                    title_baseline_y = (
                        draw_y_image + pdf_h_pts_overlay + gap_above_image
                    )
                    pdf_canvas.drawImage(
                        ImageReader(buf),
                        draw_x_image,
                        draw_y_image,
                        width=pdf_w_pts_overlay,
                        height=pdf_h_pts_overlay,
                    )
                    pdf_canvas.drawCentredString(width / 2, title_baseline_y, title)
                else:
                    pdf_canvas.drawCentredString(
                        width / 2, height / 2, f"Error rendering: {title}"
                    )
                img_idx += 1
                pdf_canvas.showPage()
            pdf_canvas.save()
            CTkMessagebox(
                title=constants.MSG_EXPORT_SUCCESS_TITLE,
                message=constants.MSG_PDF_EXPORTED_SUCCESS,
                icon="check",
            )
            print("Export PDF: Successfully exported.")
        except Exception as e:
            CTkMessagebox(
                title=constants.MSG_PDF_EXPORT_ERROR_TITLE,
                message=constants.MSG_PDF_EXPORT_FAILED.format(error=str(e)),
                icon="cancel",
            )
            print(f"Export PDF: Failed - {str(e)}")

    def export_current_view_as_tif(self):
        if not self.application_model.image_data.original_image:
            CTkMessagebox(
                title=constants.MSG_EXPORT_ERROR_TITLE,
                message=constants.MSG_NO_IMAGE_FOR_EXPORT,
                icon="warning",
            )
            print("Export current view as TIF: No original image loaded.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=constants.EXPORT_FILETYPES_TIF_NUMPY,
            title="Export Current View as TIF",
            initialfile=f"{self.application_model.base_filename if self.application_model.base_filename else 'view'}{constants.EXPORT_PREFIX_VIEW_TIF}.tif",
        )
        if not file_path:
            print("Export current view as TIF: Save dialog cancelled.")
            return

        print(f"Export current view as TIF: Starting export to {file_path}")
        try:
            base_img_for_export = (
                self.application_model.get_processed_image_for_display()
            )
            if base_img_for_export is None:
                base_img_for_export = self.application_model.image_data.original_image

            if base_img_for_export is None:
                CTkMessagebox(
                    title=constants.MSG_EXPORT_ERROR_TITLE,
                    message=constants.MSG_NO_IMAGE_FOR_EXPORT,
                    icon="warning",
                )
                print(
                    "Export current view as TIF: Base image for export is None even after fallback."
                )
                return

            # Start with the correct base: processed original or black
            if self.application_model.display_state.show_original_image:
                img_to_export_pil = base_img_for_export.copy()
                print(
                    "Export current view as TIF: Using processed/original image as base."
                )
            else:
                img_to_export_pil = Image.new(
                    "RGB", base_img_for_export.size, constants.COLOR_BLACK_STR
                )
                print(
                    "Export current view as TIF: Using black image as base (show_original_image is false)."
                )

            current_mask_array = self.application_model.image_data.mask_array
            all_mask_ids = set()
            if current_mask_array is not None and current_mask_array.size > 0:
                unique_ids = np.unique(current_mask_array)
                all_mask_ids = set(unique_ids[unique_ids != 0])

            ids_to_process_for_display = set()
            show_deselected_mode = (
                self.application_model.display_state.show_deselected_masks_only
            )

            if show_deselected_mode:
                deselected_ids = (
                    all_mask_ids - self.application_model.image_data.included_cells
                )
                ids_to_process_for_display = deselected_ids
            else:
                ids_to_process_for_display = (
                    self.application_model.image_data.included_cells.intersection(
                        all_mask_ids
                    )
                )

            # Apply Overlays Sequentially
            if (
                self.application_model.display_state.show_cell_masks
                and current_mask_array is not None
                and ids_to_process_for_display
            ):
                print("Export current view as TIF: Drawing masks.")
                # Get the mask layer (colors on black)
                mask_layer = self.overlay_processor.draw_masks_on_pil(
                    img_to_export_pil,  # Used for size context by processor
                    ids_to_process_for_display,
                )
                # Blend it with the current state of img_to_export_pil
                img_to_export_pil = self.overlay_processor.blend_image_with_mask_layer(
                    img_to_export_pil, mask_layer, constants.MASK_BLEND_ALPHA
                )

            if (
                self.application_model.display_state.show_cell_boundaries
                and current_mask_array is not None
                and ids_to_process_for_display
            ):
                print("Export current view as TIF: Drawing boundaries.")
                img_to_export_pil = self.overlay_processor.draw_boundaries_on_pil(
                    img_to_export_pil, ids_to_process_for_display
                )

            if (
                self.application_model.display_state.show_cell_numbers
                and current_mask_array is not None
                and ids_to_process_for_display
            ):
                print("Export current view as TIF: Drawing cell numbers.")
                img_to_export_pil = self.overlay_processor.draw_numbers_on_pil(
                    img_to_export_pil,
                    ids_to_process_for_display,
                    font_size=constants.CELL_NUMBERING_FONT_SIZE_ORIG_IMG,
                )

            img_to_export_pil.save(file_path, format="TIFF")
            CTkMessagebox(
                title=constants.MSG_EXPORT_SUCCESS_TITLE,
                message=constants.MSG_EXPORT_COMPLETED,
                icon="check",
            )
            print("Export current view as TIF: Successfully exported.")

        except Exception as e:
            error_message = constants.MSG_EXPORT_FAILED_PREFIX_VIEW_TIF + f" {str(e)}"
            CTkMessagebox(
                title=constants.MSG_EXPORT_ERROR_TITLE,
                message=error_message,
                icon="cancel",
            )
            print(f"Export current view as TIF: Failed - {str(e)}")
