import os
import traceback
from tkinter import filedialog

import numpy as np
from CTkMessagebox import CTkMessagebox
from PIL import Image

from .. import constants
from ..model.application_model import ApplicationModel
from ..processing.image_overlay_processor import ImageOverlayProcessor
from ..utils.debug_logger import log
from ..utils.image_reader import ImageReader
from ..utils.pdf_report_generator import PDFReportGenerator


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
        self.pdf_report_generator = PDFReportGenerator(
            application_model=self.application_model,
            overlay_processor=self.overlay_processor,
        )
        self.image_reader = ImageReader()
        log(
            f"FileIOController initialized for parent_frame: {parent_frame}",
            level="INFO",
        )

    def load_image(self):
        log("load_image execution started.", level="INFO")
        if self.parent_frame.settings_panel.filename_label:
            self.parent_frame.settings_panel.filename_label.configure(text="Loading...")
            self.parent_frame.update_idletasks()

        try:
            filepath = filedialog.askopenfilename(
                filetypes=constants.MICROSCOPY_IMG_FILETYPES
            )
            if not filepath:
                log("Load image: File dialog cancelled.")
                return

            log(f"Load image: File selected - {filepath}")
            base_filename_local = os.path.splitext(os.path.basename(filepath))[0]
            if self.parent_frame.settings_panel.filename_label:
                self.parent_frame.settings_panel.filename_label.configure(
                    text=base_filename_local
                )
                log(f"Load image: Filename label updated to {base_filename_local}")

            log("load_image: Setting status message 'Loading image...'", level="DEBUG")
            self.application_model.set_status_message("Loading image...")
            log("load_image: Resetting application model for new image.", level="DEBUG")
            self.application_model.reset_for_new_image()

            log(
                f"load_image: Calling _read_image_to_array for {os.path.basename(filepath)} with initial params.",
                level="DEBUG",
            )
            img_array, scale_info, aics_obj = self.image_reader.read_image_to_array(
                filepath, selected_channel_indices=None, z_selection_params=None
            )
            log(
                f"Image array loaded successfully. Shape: {img_array.shape}, Dtype: {img_array.dtype}"
            )

            pil_image_to_load = None
            if img_array is not None:
                try:
                    log(
                        f"load_image: Converting final numpy array to PIL Image for {os.path.basename(filepath)}.",
                        level="DEBUG",
                    )
                    pil_image_to_load = Image.fromarray(img_array)
                except Exception as e:
                    log(
                        f"Error converting final numpy array to PIL Image: {e}",
                        level="ERROR",
                    )
                    raise ValueError(
                        f"Could not convert loaded image data to displayable format: {e}"
                    )

            aics_dims = None
            if aics_obj:
                try:
                    log(
                        f"load_image: Retrieving dims from aics_obj for {os.path.basename(filepath)}.",
                        level="DEBUG",
                    )
                    aics_dims = aics_obj.dims
                except Exception as e:
                    log(f"Could not retrieve dims from aics_obj: {e}", level="WARNING")

            log(
                f"load_image: Calling application_model.load_new_image for {os.path.basename(filepath)}.",
                level="DEBUG",
            )
            self.application_model.load_new_image(
                pil_image=pil_image_to_load,
                file_path=filepath,
                base_filename=base_filename_local,
                scale_conversion=scale_info,
                aics_image_obj_param=aics_obj,
                aics_dims_param=aics_dims,
            )

            if self.parent_frame.settings_panel.filename_label:
                self.parent_frame.settings_panel.filename_label.configure(
                    text=self.application_model.base_filename
                )

            log(
                f"load_image: Triggering model update for 'image_loaded' for {os.path.basename(filepath)}.",
                level="DEBUG",
            )
            self.parent_frame.handle_model_update(change_type="image_loaded")
            log(
                f"load_image: Successfully loaded {os.path.basename(filepath)}.",
                level="INFO",
            )

        except Exception as e:
            log(f"Error loading image: {str(e)}", level="ERROR")
            if self.parent_frame.settings_panel.filename_label:
                self.parent_frame.settings_panel.filename_label.configure(
                    text=constants.UI_TEXT_LOAD_ERROR
                )
            CTkMessagebox(
                title=constants.MSG_LOAD_ERROR_TITLE,
                message=f"An error occurred while loading the image: {e}",
                icon="cancel",
            )
            log(
                "load_image: Error occurred, resetting application model.",
                level="DEBUG",
            )
            self.application_model.reset_for_new_image()
            log(
                f"load_image: Setting image load error in model for {os.path.basename(filepath)}.",
                level="DEBUG",
            )
            self.application_model.set_image_load_error(filepath, str(e))

            if self.parent_frame.settings_panel.filename_label:  # Reset filename label
                log("load_image: Resetting filename label due to error.", level="DEBUG")
                self.parent_frame.settings_panel.filename_label.configure(
                    text=constants.UI_TEXT_NO_FILE_SELECTED
                )

            log("load_image: Finally block reached.", level="DEBUG")
            if self.application_model.status_message == "Loading image...":
                log(
                    "load_image: Clearing 'Loading image...' status message.",
                    level="DEBUG",
                )
                self.application_model.clear_status_message()

    def get_processed_image_view(self) -> np.ndarray | None:
        """
        Generates a 2D or 3-channel RGB image view based on the current
        ApplicationModel display state (channels, Z-processing).
        This is called when display settings that affect the core image data change.
        """
        log("get_processed_image_view: Method called.", level="INFO")
        if not self.application_model.image_data.full_path:
            log("get_processed_image_view: No image path in model.", level="WARNING")
            return None

        full_path = self.application_model.image_data.full_path
        display_state = self.application_model.display_state
        log(
            f"get_processed_image_view: Current display state for {os.path.basename(full_path)}: z_method={display_state.z_processing_method}, z_slice={display_state.current_z_slice_index}",
            level="DEBUG",
        )

        selected_source_indices = []
        if self.application_model.image_data.aics_image_obj:
            log(
                f"get_processed_image_view: AICS object exists. Display channel configs: {display_state.display_channel_configs}",
                level="DEBUG",
            )
            for config in display_state.display_channel_configs:
                if (
                    config.get("active_in_composite", False)
                    and config.get("source_idx") is not None
                ):
                    selected_source_indices.append(config["source_idx"])
                    log(
                        f"get_processed_image_view: Config active, source_idx: {config['source_idx']} added.",
                        level="DEBUG",
                    )
            selected_source_indices = sorted(list(set(selected_source_indices)))
            log(
                f"get_processed_image_view: Active source indices (unique, sorted): {selected_source_indices} for {os.path.basename(full_path)}",
                level="DEBUG",
            )

            if (
                not selected_source_indices
                and self.application_model.image_data.aics_image_obj.dims.C > 0
            ):
                log(
                    "get_processed_image_view: No active display channels, defaulting to source channel 0 for AICS image.",
                    level="INFO",
                )
                selected_source_indices = [0]
        else:
            log(
                f"get_processed_image_view: No AICS object in model, or no channels to select. selected_source_indices remains {selected_source_indices} for {os.path.basename(full_path)}",
                level="DEBUG",
            )
        z_params = {
            "type": display_state.z_processing_method,
            "value": display_state.current_z_slice_index,
        }
        log(
            f"get_processed_image_view: Reading image {os.path.basename(full_path)} with channels={selected_source_indices}, z_params={z_params}"
        )

        try:
            log(
                f"get_processed_image_view: Calling _read_image_to_array for {os.path.basename(full_path)}.",
                level="DEBUG",
            )
            processed_arr, _, _ = self.image_reader.read_image_to_array(
                full_path,
                selected_channel_indices=selected_source_indices
                if self.application_model.image_data.aics_image_obj
                else None,  # Pass indices only if AICS
                z_selection_params=z_params
                if self.application_model.image_data.aics_image_obj
                else None,  # Pass Z params only if AICS
            )
            if processed_arr is not None:
                log(
                    f"get_processed_image_view: Successfully processed view. Shape: {processed_arr.shape}, Dtype: {processed_arr.dtype}"
                )
            else:
                log(
                    f"get_processed_image_view: _read_image_to_array returned None for {os.path.basename(full_path)}.",
                    level="WARNING",
                )
            return processed_arr
        except Exception as e:
            log(
                f"Error in get_processed_image_view for {os.path.basename(full_path)}: {e}",
                level="ERROR",
            )
            CTkMessagebox(
                title="Image Processing Error",
                message=f"Failed to update image view for '{os.path.basename(full_path)}':\n{str(e)}",
                icon="cancel",
            )
            return None

    def export_selected(self):
        log("export_selected: Method called.", level="INFO")
        if self.application_model.image_data.mask_array is None:
            log("Export selected cells: No mask array found.", level="ERROR")
            raise ValueError(constants.MSG_EXPORT_NO_MASK)

        log(
            f"export_selected: Creating export_mask. Included cells: {self.application_model.image_data.included_cells}",
            level="DEBUG",
        )
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
                log(f"Export selected cells: Saving to {save_path}")
                if save_path.endswith(".npy"):
                    log(
                        f"export_selected: Saving as .npy to {save_path}", level="DEBUG"
                    )
                    np.save(save_path, export_mask)
                else:
                    log(
                        f"export_selected: Saving as image to {save_path}",
                        level="DEBUG",
                    )
                    Image.fromarray(export_mask).save(save_path)
                log("Export selected cells: Success.")
                return constants.MSG_EXPORT_COMPLETED
            except Exception as e:
                log(f"Export selected cells: Failed - {str(e)}", level="ERROR")
                raise
        else:
            log("Export selected cells: Save dialog cancelled.")
            return None

    def export_current_view_as_tif(self):
        log("Export current view as TIF: Called", level="INFO")
        if self.application_model.image_data.original_image is None:
            log("Export TIF: No original image loaded.", level="ERROR")
            raise ValueError(constants.MSG_NO_IMAGE_FOR_EXPORT)

        current_view_pil = None
        try:
            log(
                "Export TIF: Retrieving canvas, zoom, and pan parameters.",
                level="DEBUG",
            )
            canvas_width = self.parent_frame.viewer_panel.image_canvas.winfo_width()
            canvas_height = self.parent_frame.viewer_panel.image_canvas.winfo_height()
            zoom, pan_x, pan_y = self.application_model.pan_zoom_state.get_params()
            log(
                f"Export TIF: Canvas WxH: {canvas_width}x{canvas_height}, Zoom: {zoom}, Pan: ({pan_x},{pan_y})",
                level="DEBUG",
            )

            log("Export TIF: Getting base image for view.", level="DEBUG")
            base_image_for_view = (
                self.application_model.get_processed_image_for_display()
            )
            if (
                base_image_for_view is None
            ):  # Should ideally not happen if original_image exists
                base_image_for_view = self.application_model.image_data.original_image

            if base_image_for_view is None:
                log(
                    "Export TIF: Base image for view is None even after fallback.",
                    level="ERROR",
                )
                raise ValueError("Could not retrieve base image for TIF export.")
            log(
                f"Export TIF: Base image for view obtained. Size: {base_image_for_view.size}",
                level="DEBUG",
            )

            # Determine which cell IDs to draw for the view
            log(
                "Export TIF: Determining cell IDs to display for the view.",
                level="DEBUG",
            )
            all_mask_ids = (
                set(np.unique(self.application_model.image_data.mask_array)) - {0}
                if self.application_model.image_data.mask_array is not None
                else set()
            )
            ids_to_display_for_view = set()
            if self.application_model.display_state.show_deselected_masks_only:
                ids_to_display_for_view = (
                    all_mask_ids - self.application_model.image_data.included_cells
                )
                log(
                    f"Export TIF: Showing deselected masks only. IDs: {ids_to_display_for_view}",
                    level="DEBUG",
                )
            else:
                ids_to_display_for_view = (
                    self.application_model.image_data.included_cells.intersection(
                        all_mask_ids
                    )
                )
                log(
                    f"Export TIF: Showing included/selected masks. IDs: {ids_to_display_for_view}",
                    level="DEBUG",
                )

            current_view_pil = self.overlay_processor.generate_composite_overlay_image(
                base_pil_image=base_image_for_view,
                image_size_pil=base_image_for_view.size,
                display_state=self.application_model.display_state,
                ids_to_display=ids_to_display_for_view,
                zoom=zoom,
                pan_x=pan_x,
                pan_y=pan_y,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                quality="final",
            )

            if current_view_pil is None:
                log(
                    "Export TIF: Failed to retrieve/generate the current view image using generate_composite_overlay_image.",
                    level="ERROR",
                )
                raise ValueError("Could not retrieve the current view for export.")
            log(
                f"Export TIF: Composite overlay image generated. Size: {current_view_pil.size}",
                level="DEBUG",
            )

            # --- Add Scale Bar if enabled ---
            if (
                self.application_model.display_state.show_ruler
                and self.application_model.image_data.scale_conversion
                and hasattr(self.application_model.image_data.scale_conversion, "X")
                and self.application_model.image_data.scale_conversion.X is not None
                and self.application_model.image_data.scale_conversion.X != 0
            ):
                log("Export TIF: Adding scale bar to current view image.")
                # current_view_pil is now the panned/zoomed view, matching the screen.
                # So, effective_display_zoom should be the canvas zoom level.
                current_view_pil_before_bar = (
                    current_view_pil  # For logging if it becomes None
                )
                current_view_pil = self.overlay_processor.draw_scale_bar(
                    image_to_draw_on=current_view_pil,
                    effective_display_zoom=zoom,
                    scale_conversion_obj=self.application_model.image_data.scale_conversion,
                    target_image_width=current_view_pil.width,
                    target_image_height=current_view_pil.height,
                )
                if (
                    current_view_pil is None
                ):  # Should not happen if initial current_view_pil was valid
                    log(
                        f"Export TIF: Image became None after attempting to draw scale bar. Initial image was: {current_view_pil_before_bar}",
                        level="ERROR",
                    )
                    raise ValueError("Failed to draw scale bar on TIF export image.")
            # --- End Scale Bar ---

        except Exception as e:
            log(f"Export TIF: Error retrieving current view - {str(e)}", level="ERROR")
            # Wrapping the original exception or just re-raising might be better
            # For now, creating a new ValueError to ensure a message is passed.
            raise ValueError(f"Error preparing view for TIF export: {str(e)}")

        save_path = filedialog.asksaveasfilename(
            initialfile=f"{self.application_model.base_filename}{constants.EXPORT_PREFIX_VIEW_TIF}",
            defaultextension=".tif",
            filetypes=[("TIFF", "*.tif"), ("PNG", "*.png")],
        )

        if save_path:
            try:
                log(f"Export TIF: Saving current view to {save_path}")
                current_view_pil.save(save_path)
                log("Export TIF: Successfully saved.")
                return (
                    constants.MSG_EXPORT_TIF_SUCCESS_DEFAULT
                )  # Use constant for success message
            except Exception as e:
                log(f"Export TIF: Failed to save - {str(e)}", level="ERROR")
                raise  # Re-raise
        else:
            log("Export TIF: Save dialog cancelled.")
            return None  # Indicates cancellation, _cell_body.py can choose to show no message or a specific one.

    def export_pdf(self):
        log(
            "FileIOController: Delegating PDF export to PDFReportGenerator.",
            level="INFO",
        )
        try:
            return self.pdf_report_generator.export_pdf()
        except Exception as e:
            log(
                f"FileIOController: Error during PDF export delegation: {str(e)}",
                level="ERROR",
            )
            raise

    def upload_mask_from_file(self):
        """
        Opens a file dialog to upload a mask file (.png, .tif/.tiff, .npy), validates it, and sets it as the current mask in the application model.
        Records the state for undo/redo. Returns a status message or raises on error.
        """

        if self.application_model.image_data.original_image is None:
            raise ValueError(constants.MSG_NO_IMAGE_FOR_EXPORT)

        file_path = filedialog.askopenfilename(
            title="Select Mask File",
            filetypes=[
                ("Mask Files", "*.tif *.tiff *.png *.npy"),
                ("TIFF", "*.tif *.tiff"),
                ("PNG", "*.png"),
                ("NumPy array", "*.npy"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return None

        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".tif", ".tiff", ".png"]:
                mask_img = Image.open(file_path)
                mask_np = np.array(mask_img)
            elif ext == ".npy":
                mask_np = np.load(file_path, allow_pickle=True)
                if not (
                    isinstance(mask_np, np.ndarray)
                    and np.issubdtype(mask_np.dtype, np.integer)
                ):
                    raise ValueError(constants.MSG_INVALID_MASK_NPY_FILE)
            else:
                raise ValueError(f"Unsupported file type: {ext}")

            mask_np = np.squeeze(mask_np)
            if mask_np.ndim == 3 and mask_np.shape[2] == 1:
                mask_np = mask_np[:, :, 0]
            if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                mask_np = mask_np[0, :, :]
            if not np.issubdtype(mask_np.dtype, np.integer):
                mask_np = mask_np.astype(np.int32)

            img = self.application_model.image_data.original_image
            expected_shape = (img.height, img.width)
            if mask_np.shape != expected_shape:
                raise ValueError(
                    "The uploaded mask does not match the current image size. Please upload a mask with the same dimensions as the loaded image."
                )

            self.application_model.set_segmentation_result(mask_np)
            log(
                f"Mask uploaded and applied successfully. Shape: {mask_np.shape}",
                level="INFO",
            )
            return "Mask uploaded and applied successfully."
        except Exception as e:
            log(
                f"Error loading mask: {str(e)}\n{traceback.format_exc()}", level="ERROR"
            )
            raise
