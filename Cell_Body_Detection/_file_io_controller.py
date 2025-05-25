import io
import os
from tkinter import filedialog

import customtkinter as ctk
import numpy as np
from aicsimageio import AICSImage
from CTkMessagebox import CTkMessagebox
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from scipy import ndimage

from . import constants


class FileIOController:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame  # cell_body_frame instance

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
                            f"Warning: Invalid selected_channel_indices for {path}. Defaulting to channel 0."
                        )
                        channels_to_load = [0]
                else:  # Default channel selection (primarily channel 0)
                    if img_aics.dims.C > 0:
                        channels_to_load = [0]

                if not channels_to_load:
                    raise ValueError(
                        f"No valid channels to load for {path} (available C: {img_aics.dims.C})."
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
                            processed_c_data_for_channel = channel_data_z_stack[
                                target_z_slice, :, :
                            ]
                        else:  # Unknown type, fallback to default Z slice
                            processed_c_data_for_channel = channel_data_z_stack[
                                default_z_index, :, :
                            ]
                    else:  # Single Z-slice in image or no specific Z processing requested
                        processed_c_data_for_channel = channel_data_z_stack[
                            default_z_index, :, :
                        ]

                    processed_channels_data.append(processed_c_data_for_channel)

                # Combine processed channels into 'arr'
                num_loaded_channels = len(processed_channels_data)
                if num_loaded_channels == 0:
                    # This case should ideally be caught by the 'if not channels_to_load' check earlier
                    raise ValueError(f"No channel data processed for {path}")
                elif num_loaded_channels == 1:
                    arr = processed_channels_data[0]  # 2D array
                elif num_loaded_channels == 2:
                    # Combine 2 channels into a 3-channel image (e.g., C1 in Red, C2 in Green, Blue is zero)
                    ch1_data = processed_channels_data[0]
                    ch2_data = processed_channels_data[1]
                    if ch1_data.shape != ch2_data.shape:
                        raise ValueError(
                            "Shape mismatch between processed channels for 2-channel stacking."
                        )
                    zeros_channel = np.zeros_like(ch1_data, dtype=ch1_data.dtype)
                    arr = np.stack((ch1_data, ch2_data, zeros_channel), axis=-1)
                elif num_loaded_channels == 3:
                    arr = np.stack(processed_channels_data, axis=-1)
                else:  # >= 4 channels selected by user (or somehow loaded)
                    # Take the first 3 channels for an RGB representation
                    arr = np.stack(processed_channels_data[:3], axis=-1)

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
        return arr

    def _prompt_channel_z_selection(self, img_dims, parent_window):
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

                # Warn if more than 3 selected, but allow it (backend handles it)
                if len(selected_channels_indices) > 3:
                    CTkMessagebox(
                        title="Info",
                        message="More than 3 channels selected. The first 3 will be prioritized for RGB display.",
                        icon="info",
                        parent=dialog,
                    )

            current_z_params = None
            if img_dims.Z > 1:
                z_choice = z_processing_var.get()
                if z_choice == "slice_middle":
                    current_z_params = {"type": "slice", "value": img_dims.Z // 2}
                elif z_choice == "max_project":
                    current_z_params = {"type": "max_project"}
                elif z_choice == "mean_project":
                    current_z_params = {"type": "mean_project"}
                elif z_choice == "slice_specific":
                    try:
                        slice_val = int(z_slice_entry_var.get())
                        if 0 <= slice_val < img_dims.Z:
                            current_z_params = {"type": "slice", "value": slice_val}
                        else:
                            CTkMessagebox(
                                title="Error",
                                message=f"Specific Z-slice must be between 0 and {img_dims.Z - 1}.",
                                icon="cancel",
                                parent=dialog,
                            )
                            return
                    except ValueError:
                        CTkMessagebox(
                            title="Error",
                            message="Invalid Z-slice number. Please enter a whole number.",
                            icon="cancel",
                            parent=dialog,
                        )
                        return
            elif img_dims.Z == 1:  # Single Z slice
                current_z_params = {"type": "slice", "value": 0}
            # If img_dims.Z is 0 or less (should not happen for valid images), z_params remains None

            results["channels"] = selected_channels_indices
            results["z_params"] = current_z_params
            results["ok_pressed"] = True
            dialog.destroy()

        def _on_cancel():
            results["ok_pressed"] = False
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
            return

        self.parent_frame.data_path = file_path
        self.parent_frame.base_filename = os.path.splitext(os.path.basename(file_path))[
            0
        ]
        if self.parent_frame.filename_label:
            self.parent_frame.filename_label.configure(
                text=self.parent_frame.base_filename
            )

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
                    # Try to load with defaults if metadata read fails
                    img_data_arr = self._read_image_to_array(file_path, None, None)
                else:
                    if dims.C > 1 or dims.Z > 1:
                        s_ch_indices, z_params = self._prompt_channel_z_selection(
                            dims, self.parent_frame.winfo_toplevel()
                        )
                        if (
                            s_ch_indices is None and z_params is None
                        ):  # Dialog was cancelled
                            return
                    else:  # Single channel, single Z, or no dialog needed
                        s_ch_indices = [0] if dims.C > 0 else []
                        z_params = (
                            {"type": "slice", "value": 0} if dims.Z > 0 else None
                        )  # Default z_params for single Z or no Z

                    if (
                        s_ch_indices is not None
                    ):  # Proceed if not cancelled (even if defaults were set without dialog)
                        img_data_arr = self._read_image_to_array(
                            file_path, s_ch_indices, z_params
                        )

            else:  # For other image types
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
                    return
                raise ValueError(
                    "Failed to read image or image is empty after selection/processing."
                )

            pil_image = (
                Image.fromarray(img_data_arr)
                if not isinstance(img_data_arr, Image.Image)
                else img_data_arr
            )

            self.parent_frame.image_view_model.reset_for_new_image()
            self.parent_frame.image_view_model.set_image_data(pil_image)
            self.parent_frame.history_controller.reset_history()

            # Call the finalization method on the parent_frame
            self.parent_frame.image_canvas.after(
                50, self.parent_frame._finalize_image_load_and_pan_reset
            )

        except Exception as e:
            CTkMessagebox(
                title=constants.MSG_ERROR_LOADING_IMAGE_TITLE,
                message=str(e),
                icon="cancel",
            )
            self.parent_frame.image_view_model.reset_for_new_image()
            if self.parent_frame.segment_button:
                self.parent_frame.segment_button.configure(state="disabled")
            self.parent_frame.update_display()

    def export_selected(self):
        if self.parent_frame.image_view_model.mask_array is None:
            CTkMessagebox(
                title=constants.MSG_EXPORT_ERROR_TITLE,
                message=constants.MSG_EXPORT_NO_MASK,
                icon="cancel",
            )
            return

        export_mask = np.zeros_like(
            self.parent_frame.image_view_model.mask_array,
            dtype=self.parent_frame.image_view_model.mask_array.dtype,
        )
        for cid in self.parent_frame.image_view_model.included_cells:
            export_mask[self.parent_frame.image_view_model.mask_array == cid] = cid

        save_path = filedialog.asksaveasfilename(
            initialfile=f"{self.parent_frame.base_filename}{constants.PDF_SELECTED_CELLS_FILENAME_PREFIX}",
            defaultextension=".tif",
            filetypes=constants.EXPORT_FILETYPES_TIF_NUMPY,
        )
        if save_path:
            try:
                if save_path.endswith(".npy"):
                    np.save(save_path, export_mask)
                else:
                    Image.fromarray(export_mask).save(save_path)
                CTkMessagebox(
                    title=constants.MSG_EXPORT_SUCCESS_TITLE,
                    message=constants.MSG_EXPORT_COMPLETED,
                    icon="check",
                )
            except Exception as e:
                CTkMessagebox(
                    title=constants.MSG_EXPORT_ERROR_TITLE,
                    message=constants.MSG_EXPORT_FAILED.format(error=str(e)),
                    icon="cancel",
                )

    def export_pdf(self):
        if self.parent_frame.image_view_model.original_image is None:
            CTkMessagebox(
                title=constants.MSG_PDF_EXPORT_ERROR_TITLE,
                message=constants.MSG_PDF_ORIGINAL_IMAGE_MISSING,
                icon="cancel",
            )
            return

        save_path = filedialog.asksaveasfilename(
            initialfile=f"{self.parent_frame.base_filename}{constants.PDF_DEFAULT_FILENAME_PREFIX}",
            defaultextension=".pdf",
            filetypes=constants.EXPORT_FILETYPES_PDF,
        )
        if not save_path:
            return

        TARGET_DPI = constants.PDF_TARGET_DPI

        try:
            c = canvas.Canvas(save_path, pagesize=A4)
            width_pdf, height_pdf = A4
            margin = constants.PDF_MARGIN
            content_width = width_pdf - 2 * margin
            content_height = height_pdf - 2 * margin
            line_height = constants.PDF_LINE_HEIGHT

            c.setFont(constants.PDF_FONT_TITLE, constants.PDF_FONT_TITLE_SIZE)
            c.drawString(margin, height_pdf - margin, "Cell Segmentation Report")

            y_pos_stats = height_pdf - margin - constants.PDF_STATS_TOP_OFFSET
            details = [
                f"Source File: {self.parent_frame.base_filename}",
                f"Diameter parameter (gui): {self.parent_frame.dia_entry.get() if self.parent_frame.dia_entry else 'N/A'}",
                f"Detected cells: {len(np.unique(self.parent_frame.image_view_model.mask_array)) - 1 if self.parent_frame.image_view_model.mask_array is not None and self.parent_frame.image_view_model.mask_array.size > 0 else 0}",
                f"Selected cells: {len(self.parent_frame.image_view_model.included_cells)}",
            ]
            c.setFont(constants.PDF_FONT_BODY, constants.PDF_FONT_BODY_SIZE)
            for detail in details:
                c.drawString(margin, y_pos_stats, detail)
                y_pos_stats -= line_height

            img_title_space = line_height * constants.PDF_TITLE_IMAGE_GAP_FACTOR
            max_img_height_page1 = y_pos_stats - margin - img_title_space
            max_img_width_page1 = content_width

            if self.parent_frame.image_view_model.original_image:
                orig_img_pil_page1 = (
                    self.parent_frame.image_view_model.original_image.copy()
                )
                orig_pix_w, orig_pix_h = orig_img_pil_page1.size

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
                    resized_img_page1 = orig_img_pil_page1.resize(
                        (render_pix_w, render_pix_h), Image.LANCZOS
                    )
                    orig_buf = io.BytesIO()
                    resized_img_page1.save(
                        orig_buf,
                        format=constants.PDF_IMAGE_EXPORT_FORMAT,
                        dpi=(TARGET_DPI, TARGET_DPI),
                    )
                    orig_buf.seek(0)

                    img_x_page1 = margin + (content_width - pdf_w_pts) / 2
                    img_y_page1 = y_pos_stats - pdf_h_pts - img_title_space

                    c.drawImage(
                        ImageReader(orig_buf),
                        img_x_page1,
                        img_y_page1,
                        width=pdf_w_pts,
                        height=pdf_h_pts,
                    )
                    c.setFont(
                        constants.PDF_FONT_BODY, constants.PDF_FONT_SUBHEADER_SIZE
                    )
                    c.drawCentredString(
                        margin + content_width / 2,
                        img_y_page1 - line_height,
                        "Original Image",
                    )
            c.showPage()

            images_to_export = []
            selected_cell_ids = self.parent_frame.image_view_model.included_cells.copy()

            if (
                self.parent_frame.image_view_model.mask_array is None
                or self.parent_frame.image_view_model.mask_array.size == 0
            ):
                any_overlay_option_selected = (
                    self.parent_frame.pdf_opt_masks_only.get()
                    or self.parent_frame.pdf_opt_boundaries_only.get()
                    or self.parent_frame.pdf_opt_numbers_only.get()
                    or self.parent_frame.pdf_opt_masks_boundaries.get()
                    or self.parent_frame.pdf_opt_masks_numbers.get()
                    or self.parent_frame.pdf_opt_boundaries_numbers.get()
                    or self.parent_frame.pdf_opt_masks_boundaries_numbers.get()
                )
                if any_overlay_option_selected:
                    c.setFont("Helvetica", 12)
                    c.drawCentredString(
                        width_pdf / 2,
                        height_pdf / 2,
                        "Overlay images selected for PDF,",
                    )
                    c.drawCentredString(
                        width_pdf / 2,
                        height_pdf / 2 - line_height,
                        "but no cell segmentation data is available.",
                    )
                    c.showPage()
            else:
                base_img_for_overlays = (
                    self.parent_frame.image_view_model.original_image.copy().convert(
                        "RGB"
                    )
                )

                if self.parent_frame.pdf_opt_masks_only.get():
                    img = self._draw_masks_on_pil(
                        base_img_for_overlays.copy(), selected_cell_ids
                    )
                    if img:
                        images_to_export.append(("Cell Masks Only", img))

                if self.parent_frame.pdf_opt_boundaries_only.get():
                    img = self._draw_boundaries_on_pil(
                        base_img_for_overlays.copy(), selected_cell_ids
                    )
                    if img:
                        images_to_export.append(("Cell Boundaries Only", img))

                if self.parent_frame.pdf_opt_numbers_only.get():
                    img = self._draw_numbers_on_pil(
                        base_img_for_overlays.copy(), selected_cell_ids
                    )
                    if img:
                        images_to_export.append(("Cell Numbers Only", img))

                if self.parent_frame.pdf_opt_masks_boundaries.get():
                    img = base_img_for_overlays.copy()
                    img = self._draw_masks_on_pil(img, selected_cell_ids)
                    img = self._draw_boundaries_on_pil(img, selected_cell_ids)
                    if img:
                        images_to_export.append(("Masks & Boundaries", img))

                if self.parent_frame.pdf_opt_masks_numbers.get():
                    img = base_img_for_overlays.copy()
                    img = self._draw_masks_on_pil(img, selected_cell_ids)
                    img = self._draw_numbers_on_pil(img, selected_cell_ids)
                    if img:
                        images_to_export.append(("Masks & Numbers", img))

                if self.parent_frame.pdf_opt_boundaries_numbers.get():
                    img = base_img_for_overlays.copy()
                    img = self._draw_boundaries_on_pil(img, selected_cell_ids)
                    img = self._draw_numbers_on_pil(img, selected_cell_ids)
                    if img:
                        images_to_export.append(("Boundaries & Numbers", img))

                if self.parent_frame.pdf_opt_masks_boundaries_numbers.get():
                    img = base_img_for_overlays.copy()
                    img = self._draw_masks_on_pil(img, selected_cell_ids)
                    img = self._draw_boundaries_on_pil(img, selected_cell_ids)
                    img = self._draw_numbers_on_pil(img, selected_cell_ids)
                    if img:
                        images_to_export.append(("Masks, Boundaries & Numbers", img))

            num_images_to_export = len(images_to_export)
            img_idx = 0
            while img_idx < num_images_to_export:
                c.setFont(constants.PDF_FONT_BODY, constants.PDF_FONT_SUBHEADER_SIZE)
                title, pil_img = images_to_export[img_idx]
                reserved_space_for_title_and_gap = (
                    line_height * constants.PDF_TITLE_IMAGE_GAP_FACTOR
                )
                max_img_h_on_page = content_height - reserved_space_for_title_and_gap
                max_img_w_on_page = content_width
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
                    draw_y_image = margin + (content_height - block_total_height) / 2
                    draw_x_image = margin + (content_width - pdf_w_pts_overlay) / 2
                    gap_above_image = line_height * constants.PDF_IMAGE_GAP_ABOVE_FACTOR
                    title_baseline_y = (
                        draw_y_image + pdf_h_pts_overlay + gap_above_image
                    )
                    c.drawImage(
                        ImageReader(buf),
                        draw_x_image,
                        draw_y_image,
                        width=pdf_w_pts_overlay,
                        height=pdf_h_pts_overlay,
                    )
                    c.drawCentredString(width_pdf / 2, title_baseline_y, title)
                else:
                    c.drawCentredString(
                        width_pdf / 2, height_pdf / 2, f"Error rendering: {title}"
                    )
                img_idx += 1
                c.showPage()
            c.save()
            CTkMessagebox(
                title=constants.MSG_EXPORT_SUCCESS_TITLE,
                message=constants.MSG_PDF_EXPORTED_SUCCESS,
                icon="check",
            )
        except Exception as e:
            CTkMessagebox(
                title=constants.MSG_PDF_EXPORT_ERROR_TITLE,
                message=constants.MSG_PDF_EXPORT_FAILED.format(error=str(e)),
                icon="cancel",
            )

    def _draw_masks_on_pil(self, base_pil_image, cell_ids_to_draw):
        if (
            self.parent_frame.image_view_model.mask_array is None
            or not cell_ids_to_draw
        ):
            return base_pil_image

        mask_overlay_rgb_pil = Image.new(
            "RGB", base_pil_image.size, constants.COLOR_BLACK_STR
        )
        rng = np.random.default_rng(seed=constants.RANDOM_SEED_MASKS)
        all_unique_mask_ids = np.unique(self.parent_frame.image_view_model.mask_array)
        color_map = {
            uid: tuple(rng.integers(50, 200, size=3))
            for uid in all_unique_mask_ids
            if uid != 0
        }
        temp_mask_np = np.zeros(
            (*self.parent_frame.image_view_model.mask_array.shape, 3), dtype=np.uint8
        )
        any_mask_drawn = False
        for cell_id_val in cell_ids_to_draw:
            if cell_id_val != 0 and cell_id_val in color_map:
                mask_pixels = (
                    self.parent_frame.image_view_model.mask_array == cell_id_val
                )
                if np.any(mask_pixels):
                    temp_mask_np[mask_pixels] = color_map[cell_id_val]
                    any_mask_drawn = True
        if not any_mask_drawn:
            return base_pil_image
        mask_overlay_rgb_pil = Image.fromarray(temp_mask_np)
        if base_pil_image.mode != "RGB":
            base_pil_image = base_pil_image.convert("RGB")
        blended_image = Image.blend(
            base_pil_image, mask_overlay_rgb_pil, alpha=constants.MASK_BLEND_ALPHA
        )
        return blended_image

    def _draw_boundaries_on_pil(self, base_pil_image, cell_ids_to_draw):
        if (
            self.parent_frame.image_view_model.mask_array is None
            or not cell_ids_to_draw
        ):
            return base_pil_image

        current_mask_array = self.parent_frame.image_view_model.mask_array
        # Access _get_exact_boundaries via parent_frame
        exact_boundaries = self.parent_frame._get_exact_boundaries(current_mask_array)

        if exact_boundaries.size == 0:
            return base_pil_image

        boundary_to_draw_on_pdf = np.zeros_like(exact_boundaries, dtype=bool)
        any_boundary_drawn = False
        for cid in cell_ids_to_draw:
            if cid != 0:
                cell_mask_region = current_mask_array == cid
                if np.any(cell_mask_region):
                    boundary_to_draw_on_pdf |= exact_boundaries & cell_mask_region
                    any_boundary_drawn = True
        if not any_boundary_drawn:
            return base_pil_image
        output_image_np = np.array(base_pil_image.convert("RGB"))
        boundary_color_map_pil = constants.BOUNDARY_COLOR_MAP_PIL
        gui_boundary_color_name = self.parent_frame.boundary_color.get()
        chosen_color_np = np.array(
            boundary_color_map_pil.get(
                gui_boundary_color_name, constants.BOUNDARY_COLOR_MAP_PIL["Green"]
            )
        )
        if boundary_to_draw_on_pdf.shape == output_image_np.shape[:2]:
            output_image_np[boundary_to_draw_on_pdf] = chosen_color_np
        else:
            print(
                f"Warning: Shape mismatch for PDF boundary drawing. Base: {output_image_np.shape[:2]}, Boundary: {boundary_to_draw_on_pdf.shape}"
            )
            return base_pil_image
        return Image.fromarray(output_image_np)

    def _draw_numbers_on_pil(self, base_pil_image, cell_ids_to_draw):
        if (
            self.parent_frame.image_view_model.mask_array is None
            or not cell_ids_to_draw
            or self.parent_frame.image_view_model.original_image is None
        ):
            return base_pil_image

        draw_on_pil = ImageDraw.Draw(base_pil_image)
        current_boundary_color_name = self.parent_frame.boundary_color.get()
        boundary_color_map_text = constants.PDF_TEXT_NUMBER_COLOR_MAP
        text_color_tuple = boundary_color_map_text.get(
            current_boundary_color_name, constants.PDF_TEXT_NUMBER_COLOR_MAP["Green"]
        )
        base_font_size = constants.CELL_NUMBERING_FONT_SIZE_PDF
        try:
            font = ImageFont.truetype(constants.DEFAULT_FONT_BOLD, size=base_font_size)
        except IOError:
            try:
                font = ImageFont.truetype(constants.DEFAULT_FONT, size=base_font_size)
            except IOError:
                font = ImageFont.load_default()

        cell_info_for_sorting_pdf = []
        img_h_orig, img_w_orig = (
            self.parent_frame.image_view_model.original_image.height,
            self.parent_frame.image_view_model.original_image.width,
        )
        current_mask_array = self.parent_frame.image_view_model.mask_array
        num_drawn_actually = 0

        for cell_id_val in cell_ids_to_draw:
            if cell_id_val != 0:
                single_cell_mask = current_mask_array == cell_id_val
                if np.any(single_cell_mask):
                    rows, cols = np.where(single_cell_mask)
                    top_most = np.min(rows)
                    left_most_in_top_row = np.min(cols[rows == top_most])
                    cell_boundary_margin = constants.CELL_CENTER_FIND_MARGIN
                    eroded_mask = ndimage.binary_erosion(
                        single_cell_mask,
                        iterations=cell_boundary_margin,
                        border_value=0,
                    )
                    chosen_cx, chosen_cy = -1.0, -1.0
                    if np.any(eroded_mask):
                        cy_eroded_com, cx_eroded_com = ndimage.center_of_mass(
                            eroded_mask
                        )
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
                            dist_transform_eroded = ndimage.distance_transform_edt(
                                eroded_mask
                            )
                            if np.any(dist_transform_eroded):
                                pole_y_eroded, pole_x_eroded = np.unravel_index(
                                    np.argmax(dist_transform_eroded),
                                    dist_transform_eroded.shape,
                                )
                                chosen_cx, chosen_cy = (
                                    float(pole_x_eroded),
                                    float(pole_y_eroded),
                                )
                            else:
                                cy_orig_com_fb, cx_orig_com_fb = ndimage.center_of_mass(
                                    single_cell_mask
                                )
                                chosen_cx, chosen_cy = cx_orig_com_fb, cy_orig_com_fb
                    else:
                        cy_orig_com, cx_orig_com = ndimage.center_of_mass(
                            single_cell_mask
                        )
                        cy_idx, cx_idx = (
                            int(round(cy_orig_com)),
                            int(round(cx_orig_com)),
                        )
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
                                    np.argmax(dist_transform_orig),
                                    dist_transform_orig.shape,
                                )
                                chosen_cx, chosen_cy = (
                                    float(pole_x_orig),
                                    float(pole_y_orig),
                                )
                            else:
                                continue
                    final_cx_orig = max(0, min(chosen_cx, img_w_orig - 1))
                    final_cy_orig = max(0, min(chosen_cy, img_h_orig - 1))
                    cell_info_for_sorting_pdf.append(
                        {
                            "id": cell_id_val,
                            "top": top_most,
                            "left": left_most_in_top_row,
                            "center_orig_x": final_cx_orig,
                            "center_orig_y": final_cy_orig,
                        }
                    )
        if not cell_info_for_sorting_pdf:
            return base_pil_image
        sorted_cells_for_pdf_numbering = sorted(
            cell_info_for_sorting_pdf, key=lambda c: (c["top"], c["left"])
        )
        for i, cell_data in enumerate(sorted_cells_for_pdf_numbering):
            display_number = str(i + 1)
            center_x_pil = cell_data["center_orig_x"]
            center_y_pil = cell_data["center_orig_y"]
            try:
                if hasattr(draw_on_pil.textbbox, "__call__"):
                    bbox = draw_on_pil.textbbox(
                        (center_x_pil, center_y_pil),
                        display_number,
                        font=font,
                        anchor="lt",
                    )
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    final_text_x = center_x_pil - text_width / 2
                    final_text_y = center_y_pil - text_height / 2
                else:
                    text_width, text_height = draw_on_pil.textsize(
                        display_number, font=font
                    )
                    final_text_x = center_x_pil - text_width / 2
                    final_text_y = center_y_pil - text_height / 2
            except AttributeError:
                text_width, text_height = 10, 10
                final_text_x = center_x_pil - text_width / 2
                final_text_y = center_y_pil - text_height / 2
            draw_on_pil.text(
                (final_text_x, final_text_y),
                display_number,
                fill=text_color_tuple,
                font=font,
            )
            num_drawn_actually += 1
        if num_drawn_actually == 0:
            return base_pil_image
        return base_pil_image

    def export_current_view_as_tif(self):
        if self.parent_frame.image_view_model.original_image is None:
            CTkMessagebox(
                title=constants.MSG_EXPORT_ERROR_TITLE,  # "Export Error"
                message=constants.MSG_NO_IMAGE_FOR_EXPORT,  # "No image loaded to export."
                icon="cancel",
            )
            return

        original_pil = self.parent_frame.image_view_model.original_image
        cell_ids_for_drawing = []

        if (
            self.parent_frame.image_view_model.mask_array is not None
            and self.parent_frame.image_view_model.mask_array.size > 0
        ):
            all_mask_ids = np.unique(self.parent_frame.image_view_model.mask_array)
            all_cell_ids_set = {cid for cid in all_mask_ids if cid != 0}

            if self.parent_frame.show_only_deselected.get():
                cell_ids_for_drawing = list(
                    all_cell_ids_set - self.parent_frame.image_view_model.included_cells
                )
            else:
                cell_ids_for_drawing = list(
                    self.parent_frame.image_view_model.included_cells
                )

        # Initialize output_pil_image
        if self.parent_frame.show_original.get():
            output_pil_image = original_pil.copy().convert("RGB")
        else:
            output_pil_image = Image.new(
                "RGB", original_pil.size, constants.COLOR_BLACK_STR
            )

        # Apply overlays based on current view settings
        if self.parent_frame.show_mask.get():
            # _draw_masks_on_pil returns a blended image (new instance)
            output_pil_image = self._draw_masks_on_pil(
                output_pil_image.copy(), cell_ids_for_drawing
            )  # Pass a copy to avoid double blending if base is also mask

        if self.parent_frame.show_boundaries.get():
            # _draw_boundaries_on_pil modifies the image in place, so pass a copy if it's not the final step or if original is desired clean
            # For sequential application, we operate on the evolving output_pil_image
            output_pil_image = self._draw_boundaries_on_pil(
                output_pil_image, cell_ids_for_drawing
            )

        if self.parent_frame.show_cell_numbers.get():
            # _draw_numbers_on_pil modifies the image in place
            output_pil_image = self._draw_numbers_on_pil(
                output_pil_image, cell_ids_for_drawing
            )

        save_path = filedialog.asksaveasfilename(
            initialfile=f"{self.parent_frame.base_filename}_current_view",
            defaultextension=".tif",
            filetypes=[("TIFF Image", "*.tif *.tiff")],
        )

        if save_path:
            try:
                output_pil_image.save(save_path, format="TIFF", compression="tiff_lzw")
                CTkMessagebox(
                    title=constants.MSG_EXPORT_SUCCESS_TITLE,  # "Export Successful"
                    message=f"{constants.MSG_EXPORT_PREFIX_VIEW_TIF} exported successfully.",  # "Current view TIF"
                    icon="check",
                )
            except Exception as e:
                CTkMessagebox(
                    title=constants.MSG_EXPORT_ERROR_TITLE,  # "Export Error"
                    message=f"{constants.MSG_EXPORT_FAILED_PREFIX_VIEW_TIF} {str(e)}",  # "Failed to export current view TIF:"
                    icon="cancel",
                )
