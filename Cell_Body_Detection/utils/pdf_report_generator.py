import io
from tkinter import filedialog

import numpy as np
from CTkMessagebox import CTkMessagebox
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from .. import constants
from ..model.application_model import ApplicationModel
from ..processing.image_overlay_processor import ImageOverlayProcessor
from ..utils.debug_logger import log


class PDFReportGenerator:
    def __init__(
        self,
        application_model: ApplicationModel,
        overlay_processor: ImageOverlayProcessor,
    ):
        self.application_model = application_model
        self.overlay_processor = overlay_processor

    def export_pdf(self):
        log("export_pdf: Method called.", level="INFO")
        if self.application_model.image_data.original_image is None:
            CTkMessagebox(
                title=constants.MSG_PDF_EXPORT_ERROR_TITLE,
                message=constants.MSG_PDF_ORIGINAL_IMAGE_MISSING,
                icon="cancel",
            )
            log("Export PDF: Original image missing.", level="WARNING")
            return

        save_path = filedialog.asksaveasfilename(
            initialfile=f"{self.application_model.base_filename}{constants.PDF_DEFAULT_FILENAME_PREFIX}",
            defaultextension=".pdf",
            filetypes=constants.EXPORT_FILETYPES_PDF,
        )
        if not save_path:
            log("Export PDF: Save dialog cancelled.")
            return None  # Handled by _cell_body.py

        # --- Pre-export checks for mask-dependent options ---
        log(
            "Export PDF: Performing pre-export checks for mask-dependent options.",
            level="DEBUG",
        )
        any_mask_dependent_pdf_option_selected = (
            self.application_model.display_state.pdf_opt_masks_only
            or self.application_model.display_state.pdf_opt_boundaries_only
            or self.application_model.display_state.pdf_opt_numbers_only
            or self.application_model.display_state.pdf_opt_masks_boundaries
            or self.application_model.display_state.pdf_opt_masks_numbers
            or self.application_model.display_state.pdf_opt_boundaries_numbers
            or self.application_model.display_state.pdf_opt_masks_boundaries_numbers
        )
        mask_array_available = self.application_model.image_data.mask_array_exists()

        if any_mask_dependent_pdf_option_selected and not mask_array_available:
            log(
                "Export PDF: Aborted - Overlay options selected, but no mask data is available. Options: "
                f"masks_only={self.application_model.display_state.pdf_opt_masks_only}, "
                f"boundaries_only={self.application_model.display_state.pdf_opt_boundaries_only}, "
                f"numbers_only={self.application_model.display_state.pdf_opt_numbers_only}, "
                f"masks_boundaries={self.application_model.display_state.pdf_opt_masks_boundaries}, "
                f"masks_numbers={self.application_model.display_state.pdf_opt_masks_numbers}, "
                f"boundaries_numbers={self.application_model.display_state.pdf_opt_boundaries_numbers}, "
                f"masks_boundaries_numbers={self.application_model.display_state.pdf_opt_masks_boundaries_numbers}",
                level="WARNING",
            )
            CTkMessagebox(
                title=constants.MSG_PDF_EXPORT_ERROR_TITLE,  # Or a more specific title
                message=constants.MSG_PDF_NO_MASK_FOR_OVERLAY_OPTIONS,  # New constant needed
                icon="warning",
            )
            return None  # Signal to _cell_body.py that operation didn't proceed to success/failure

        log(f"Export PDF: Starting PDF export to {save_path}")
        TARGET_DPI = constants.PDF_TARGET_DPI
        # overlay_data_missing_but_requested = False # No longer needed with stricter approach
        log(f"Export PDF: Target DPI set to {TARGET_DPI}", level="DEBUG")

        try:
            log("Export PDF: Retrieving image data and cell IDs.", level="DEBUG")
            mask_array = self.application_model.image_data.mask_array

            all_cell_ids_in_mask = (
                np.unique(mask_array[mask_array > 0])
                if mask_array is not None
                else np.array([])
            )
            selected_cell_ids_set = self.application_model.image_data.included_cells
            log(
                f"Export PDF: All cell IDs in mask: {all_cell_ids_in_mask.size}, Selected cell IDs: {len(selected_cell_ids_set)}",
                level="DEBUG",
            )

            selected_cell_ids = list(selected_cell_ids_set)

            # PDF setup
            pdf_canvas = canvas.Canvas(save_path, pagesize=A4)
            width, height = A4  # Page dimensions

            # --- Get the base image for PDF (potentially processed) ---
            log("Export PDF: Getting base image for PDF content.", level="DEBUG")
            current_original_image_pil = (
                self.application_model.get_processed_image_for_display()
            )

            if current_original_image_pil is None:  # Fallback to true original
                log(
                    "Export PDF: Processed image for display is None, falling back to true original_image.",
                    level="WARNING",
                )
                current_original_image_pil = (
                    self.application_model.image_data.original_image
                )

            if current_original_image_pil:
                log(
                    f"Export PDF: Base image for PDF content obtained. Size: {current_original_image_pil.size}",
                    level="DEBUG",
                )
            else:
                log(
                    "Export PDF: Base image for PDF content is None even after fallback.",
                    level="ERROR",
                )

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
                f"Diameter parameter: {self.application_model.segmentation_diameter}",
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
                    # --- Add Scale Bar to Original Image Page in PDF ---
                    if (
                        self.application_model.display_state.pdf_include_scale_bar
                        and self.application_model.image_data.scale_conversion
                        and hasattr(
                            self.application_model.image_data.scale_conversion, "X"
                        )
                        and self.application_model.image_data.scale_conversion.X
                        is not None
                        and self.application_model.image_data.scale_conversion.X != 0
                    ):
                        log("Export PDF: Adding scale bar to original image page.")
                        # The effective zoom for the scale bar is how much the original image was scaled to fit this PDF page.
                        effective_zoom_for_pdf_orig = (
                            resized_img_page1.width / current_original_image_pil.width
                        )
                        # Scale the font size for the scale bar label to match the numbers
                        scaled_font_size = int(
                            round(
                                constants.CELL_NUMBERING_FONT_SIZE_PDF
                                * effective_zoom_for_pdf_orig
                            )
                        )
                        # Make a mutable copy if resized_img_page1 is not already (it should be from resize)
                        image_for_bar_drawing = resized_img_page1.copy()

                        image_with_bar = self.overlay_processor.draw_scale_bar(
                            image_to_draw_on=image_for_bar_drawing,
                            effective_display_zoom=effective_zoom_for_pdf_orig,
                            scale_conversion_obj=self.application_model.image_data.scale_conversion,
                            target_image_width=image_for_bar_drawing.width,
                            target_image_height=image_for_bar_drawing.height,
                            font_size=scaled_font_size,  # Scaled to match numbers
                        )
                        if image_with_bar:
                            orig_buf_bar = io.BytesIO()
                            image_with_bar.save(
                                orig_buf_bar,
                                format=constants.PDF_IMAGE_EXPORT_FORMAT,
                                dpi=(TARGET_DPI, TARGET_DPI),
                            )
                            orig_buf_bar.seek(0)
                            # Redraw the image, now with the scale bar
                            pdf_canvas.drawImage(
                                ImageReader(orig_buf_bar),
                                img_x_page1,
                                img_y_page1,
                                width=pdf_w_pts,
                                height=pdf_h_pts,
                            )
                    # --- End Scale Bar for Original Image Page ---
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
            log(
                f"Export PDF: Preparing for overlay images. Number of selected cells for overlays: {len(selected_cell_ids)}",
                level="DEBUG",
            )

            if (
                self.application_model.image_data.mask_array is None
                or self.application_model.image_data.mask_array.size == 0
            ):
                log("Export PDF: No mask array available.", level="WARNING")
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
                    log(
                        "Export PDF: Overlay options selected, but no mask data. Adding info page.",
                        level="WARNING",
                    )
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
                if current_original_image_pil and (
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

                    ids_for_pdf_overlay = (
                        selected_cell_ids  # Use the selected cells for PDF overlays
                    )

                    if self.application_model.display_state.pdf_opt_masks_only:
                        current_export_item_img = current_original_image_pil.copy()
                        # Apply Masks only by getting layer and blending
                        mask_layer = self.overlay_processor.get_cached_mask_layer_rgb(
                            base_pil_image_size=current_export_item_img.size,
                            cell_ids_to_draw=ids_for_pdf_overlay,
                            show_deselected_masks_only=self.application_model.display_state.show_deselected_masks_only,
                        )
                        current_export_item_img = (
                            self.overlay_processor.blend_image_with_mask_layer(
                                current_export_item_img,
                                mask_layer,
                                constants.MASK_BLEND_ALPHA,
                            )
                        )
                        if current_export_item_img:
                            images_to_export.append(
                                (
                                    constants.UI_TEXT_PDF_MASKS_ONLY,
                                    current_export_item_img,
                                )
                            )

                    if self.application_model.display_state.pdf_opt_boundaries_only:
                        current_export_item_img = current_original_image_pil.copy()
                        img = self.overlay_processor.draw_boundaries_on_pil(
                            current_export_item_img, ids_for_pdf_overlay
                        )
                        if img:
                            images_to_export.append(
                                (constants.UI_TEXT_PDF_BOUNDARIES_ONLY, img)
                            )

                    if self.application_model.display_state.pdf_opt_numbers_only:
                        current_export_item_img = current_original_image_pil.copy()
                        img = self.overlay_processor.draw_numbers_on_pil(
                            current_export_item_img, ids_for_pdf_overlay
                        )
                        if img:
                            images_to_export.append(
                                (
                                    constants.UI_TEXT_PDF_NUMBERS_ONLY,
                                    img,
                                )
                            )

                    if self.application_model.display_state.pdf_opt_masks_boundaries:
                        current_export_item_img = current_original_image_pil.copy()
                        # Apply Masks first by getting layer and blending
                        mask_layer = self.overlay_processor.get_cached_mask_layer_rgb(
                            base_pil_image_size=current_export_item_img.size,
                            cell_ids_to_draw=ids_for_pdf_overlay,
                            show_deselected_masks_only=self.application_model.display_state.show_deselected_masks_only,
                        )
                        current_export_item_img = (
                            self.overlay_processor.blend_image_with_mask_layer(
                                current_export_item_img,
                                mask_layer,
                                constants.MASK_BLEND_ALPHA,
                            )
                        )
                        current_export_item_img = (
                            self.overlay_processor.draw_boundaries_on_pil(
                                current_export_item_img, ids_for_pdf_overlay
                            )
                        )
                        if current_export_item_img:
                            images_to_export.append(
                                (
                                    constants.UI_TEXT_PDF_MASKS_BOUNDARIES,
                                    current_export_item_img,
                                )
                            )

                    if self.application_model.display_state.pdf_opt_masks_numbers:
                        current_export_item_img = current_original_image_pil.copy()
                        # Apply Masks
                        mask_layer = self.overlay_processor.get_cached_mask_layer_rgb(
                            base_pil_image_size=current_export_item_img.size,
                            cell_ids_to_draw=ids_for_pdf_overlay,
                            show_deselected_masks_only=self.application_model.display_state.show_deselected_masks_only,
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
                                current_export_item_img, ids_for_pdf_overlay
                            )
                        )
                        if current_export_item_img:
                            images_to_export.append(
                                (
                                    constants.UI_TEXT_PDF_MASKS_NUMBERS,
                                    current_export_item_img,
                                )
                            )

                    if self.application_model.display_state.pdf_opt_boundaries_numbers:
                        current_export_item_img = current_original_image_pil.copy()
                        # Apply Boundaries
                        current_export_item_img = (
                            self.overlay_processor.draw_boundaries_on_pil(
                                current_export_item_img, ids_for_pdf_overlay
                            )
                        )
                        # Apply Numbers
                        img = self.overlay_processor.draw_numbers_on_pil(
                            current_export_item_img, ids_for_pdf_overlay
                        )
                        if img:
                            images_to_export.append(
                                (
                                    constants.UI_TEXT_PDF_BOUNDARIES_NUMBERS,
                                    img,
                                )
                            )

                    if self.application_model.display_state.pdf_opt_masks_boundaries_numbers:
                        current_export_item_img = current_original_image_pil.copy()
                        # Apply Masks
                        mask_layer = self.overlay_processor.get_cached_mask_layer_rgb(
                            base_pil_image_size=current_export_item_img.size,
                            cell_ids_to_draw=ids_for_pdf_overlay,
                            show_deselected_masks_only=self.application_model.display_state.show_deselected_masks_only,
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
                        img = self.overlay_processor.draw_numbers_on_pil(
                            current_export_item_img, ids_for_pdf_overlay
                        )
                        if img:
                            images_to_export.append(
                                (
                                    constants.UI_TEXT_PDF_MASKS_BOUNDARIES_NUMBERS,
                                    img,
                                )
                            )

            num_images_to_export = len(images_to_export)
            log(
                f"Export PDF: Total overlay images to add to PDF: {num_images_to_export}",
                level="INFO",
            )
            img_idx = 0
            while img_idx < num_images_to_export:
                log(
                    f"Export PDF: Processing overlay image {img_idx + 1}/{num_images_to_export} for PDF page.",
                    level="DEBUG",
                )
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
                    # --- Add Scale Bar to Overlay Image Pages in PDF ---
                    if (
                        self.application_model.display_state.pdf_include_scale_bar
                        and self.application_model.image_data.scale_conversion
                        and hasattr(
                            self.application_model.image_data.scale_conversion, "X"
                        )
                        and self.application_model.image_data.scale_conversion.X
                        is not None
                        and self.application_model.image_data.scale_conversion.X != 0
                    ):
                        log(f"Export PDF: Adding scale bar to overlay image: {title}")
                        # Effective zoom for the scale bar on these overlay pages.
                        # img_w_orig_pixels and img_h_orig_pixels are from the original full-res image.
                        effective_zoom_for_pdf_overlay = (
                            resized_pil_overlay.width / img_w_orig_pixels
                        )
                        # Scale the font size for the scale bar label to match the numbers
                        scaled_font_size_overlay = int(
                            round(
                                constants.CELL_NUMBERING_FONT_SIZE_PDF
                                * effective_zoom_for_pdf_overlay
                            )
                        )
                        # Make a mutable copy if resized_pil_overlay is not already (it should be from resize)
                        image_for_bar_drawing = resized_pil_overlay.copy()

                        image_with_bar = self.overlay_processor.draw_scale_bar(
                            image_to_draw_on=image_for_bar_drawing,
                            effective_display_zoom=effective_zoom_for_pdf_overlay,
                            scale_conversion_obj=self.application_model.image_data.scale_conversion,
                            target_image_width=image_for_bar_drawing.width,
                            target_image_height=image_for_bar_drawing.height,
                            font_size=scaled_font_size_overlay,  # Scaled to match numbers
                        )
                        if image_with_bar:
                            buf_bar_overlay = io.BytesIO()
                            image_with_bar.save(
                                buf_bar_overlay,
                                format=constants.PDF_IMAGE_EXPORT_FORMAT,
                                dpi=(TARGET_DPI, TARGET_DPI),
                            )
                            buf_bar_overlay.seek(0)
                            # Redraw the image, now with the scale bar
                            pdf_canvas.drawImage(
                                ImageReader(buf_bar_overlay),
                                draw_x_image,
                                draw_y_image,
                                width=pdf_w_pts_overlay,
                                height=pdf_h_pts_overlay,
                            )
                    # --- End Scale Bar for Overlay Image Pages ---
                    pdf_canvas.drawCentredString(width / 2, title_baseline_y, title)
                else:
                    pdf_canvas.drawCentredString(
                        width / 2, height / 2, f"Error rendering: {title}"
                    )
                img_idx += 1
                pdf_canvas.showPage()
                log(f"Export PDF: Page for overlay '{title}' completed.", level="DEBUG")
            pdf_canvas.save()
            log("Export PDF: Successfully exported.")
            return constants.MSG_PDF_EXPORTED_SUCCESS  # Standard success message

        except Exception as e:
            log(f"Export PDF: Failed - {str(e)}", level="ERROR")
            raise
