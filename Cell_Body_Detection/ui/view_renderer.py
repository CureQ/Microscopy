import threading

import customtkinter as ctk
import numpy as np
from PIL import (
    Image,
    ImageDraw,
    ImageTk,
)

from .. import constants
from ..model.application_model import ApplicationModel
from ..processing.image_overlay_processor import ImageOverlayProcessor
from ..utils.debug_logger import log


class ImageViewRenderer:
    def __init__(
        self,
        canvas_ref: ctk.CTkCanvas,
        application_model_ref: ApplicationModel,
        cell_body_frame_ref: ctk.CTkFrame,  # To access some UI elements like boundary_color or show_original
    ):
        log("ImageViewRenderer: Initializing.", level="INFO")
        self.image_canvas = canvas_ref
        self.application_model = application_model_ref
        self.parent_frame = cell_body_frame_ref

        self.overlay_processor = ImageOverlayProcessor(application_model_ref)

        self.tk_image_on_canvas = None
        self._update_display_retry_id = None
        self._final_render_debounce_id = None

        # Store last known states of relevant display settings for cache invalidation logic
        self._last_known_show_deselected_for_cache: bool | None = (
            None  # Unified for all caches dependent on this
        )
        self._last_known_boundary_color_name_for_render: str | None = (
            None  # Used to detect change for re-render, not L-cache invalidation
        )

        self.draw_feedback_color = constants.DRAW_FEEDBACK_COLOR
        self.draw_first_point_color = constants.DRAW_FIRST_POINT_COLOR
        self.draw_last_point_color = constants.DRAW_LAST_POINT_COLOR
        self.draw_point_radius = constants.DRAW_POINT_RADIUS

        self._cache_generation_active = False
        self._cache_generation_thread = None

        # Subscribe to model changes
        self.application_model.subscribe(self.handle_model_update)
        log(
            "ImageViewRenderer: Initialization complete. Subscribed to model updates.",
            level="DEBUG",
        )

    def handle_model_update(self, change_type: str | None = None):
        """
        Called by the ApplicationModel when its state changes.
        Determines if a re-render is necessary.
        """
        log(
            f"ImageViewRenderer.handle_model_update received change_type: {change_type}",
            level="DEBUG",
        )

        if change_type == "pdf_options_changed":
            log(
                "ImageViewRenderer: PDF options changed, no re-render of main canvas needed.",
                level="DEBUG",
            )
            return  # Explicitly do nothing for PDF option changes

        if change_type in [
            "image_loaded",
            "segmentation_updated",
            "cell_selection_changed",
            "mask_updated_user_drawn",
            "pan_zoom_updated",
            "pan_zoom_reset",
            "display_settings_changed",
            "display_settings_reset",
            "view_options_changed",
            "model_restored_undo",
            "model_restored_redo",
            "history_updated",
        ]:
            if change_type == "pan_zoom_updated":
                log(
                    f"ImageViewRenderer: '{change_type}' detected, rendering with interactive quality.",
                    level="DEBUG",
                )
                self.render(quality="interactive")
            elif change_type in [
                "display_settings_changed",
                "display_settings_reset",
                "view_options_changed",
            ]:
                log(
                    f"ImageViewRenderer: '{change_type}' detected, invalidating relevant caches and rendering with interactive quality (final render is debounced by UI).",
                    level="DEBUG",
                )
                self.render(quality="interactive")
                self._schedule_final_render()
            else:
                log(
                    f"ImageViewRenderer: '{change_type}' detected, rendering with final quality.",
                    level="DEBUG",
                )
                self.render(quality="final")
        elif change_type == "segmentation_params_changed":
            log(
                "ImageViewRenderer: 'segmentation_params_changed' detected. Re-rendering.",
                level="DEBUG",
            )
            self.render(quality="interactive")
            self._schedule_final_render()
        elif change_type == "channel_z_settings_changed":
            log(
                f"ImageViewRenderer: '{change_type}' detected, rendering with final quality.",
                level="DEBUG",
            )
            self.render(quality="final")
        else:
            log(
                f"ImageViewRenderer: Unhandled change_type '{change_type}', or no re-render needed for it.",
                level="DEBUG",
            )

        # Specific cache invalidations based on change_type
        if change_type in [
            "segmentation_updated",
            "cell_selection_changed",
            "mask_updated_user_drawn",
            "model_restored_undo",
            "model_restored_redo",
        ]:
            log(
                f"ImageViewRenderer: '{change_type}' detected, invalidating all caches.",
                level="DEBUG",
            )
            self.invalidate_caches()  # Selection or mask structure changed

        if change_type in [
            "view_options_changed",
            "display_settings_changed",
            "display_settings_reset",
        ]:
            log(
                f"ImageViewRenderer: '{change_type}' detected, checking and invalidating specific caches.",
                level="DEBUG",
            )
            current_boundary_color = (
                self.application_model.display_state.boundary_color_name
            )
            current_show_deselected = (
                self.application_model.display_state.show_deselected_masks_only
            )

            # Invalidate mask cache via processor if show_deselected_masks_only changed
            if self._last_known_show_deselected_for_cache != current_show_deselected:
                log(
                    "ImageViewRenderer: 'Show deselected masks only' changed, invalidating relevant caches in processor.",
                    level="DEBUG",
                )
                self.overlay_processor.invalidate_mask_cache()
                self.overlay_processor.invalidate_boundary_cache()
                self.overlay_processor.invalidate_number_positions_cache()  # Added
                # self._cached_cell_number_positions = None # Removed, was local

            # If boundary color changes, just note it. Re-render will use new color. L-mode boundary cache in processor is not affected.
            if (
                self._last_known_boundary_color_name_for_render
                != current_boundary_color
            ):
                log(
                    "ImageViewRenderer: Boundary color changed. Re-render will apply new color.",
                    level="DEBUG",
                )

            self._last_known_show_deselected_for_cache = current_show_deselected
            self._last_known_boundary_color_name_for_render = current_boundary_color

    def invalidate_caches(self):
        log("ImageViewRenderer: Invalidating all rendering caches.", level="DEBUG")
        self.overlay_processor.invalidate_mask_cache()
        self.overlay_processor.invalidate_boundary_cache()
        self.overlay_processor.invalidate_number_positions_cache()  # Added

    def _schedule_final_render(self):
        if self._final_render_debounce_id:
            self.image_canvas.after_cancel(self._final_render_debounce_id)
        self._final_render_debounce_id = self.image_canvas.after(
            constants.INTERACTIVE_RENDER_DELAY_MS,
            lambda: self.render(quality="final"),
        )

    # --- Helper to determine cache needs ---
    def _needs_cache_regeneration(
        self,
        current_ids_tuple_for_display: tuple,
        pil_image_to_render_size: tuple[int, int],
    ):
        needs_mask_worker_request = False
        if (
            self.application_model.display_state.show_cell_masks
            and pil_image_to_render_size != (0, 0)
            and not self.overlay_processor.is_mask_cache_current(
                base_pil_image_size=pil_image_to_render_size,
                cell_ids_to_draw=set(current_ids_tuple_for_display),
                show_deselected_masks_only=self.application_model.display_state.show_deselected_masks_only,
            )
        ):
            needs_mask_worker_request = True
            log("ImageViewRenderer: Mask cache regeneration needed.", level="DEBUG")

        needs_boundary_worker_request = False
        if (
            self.application_model.display_state.show_cell_boundaries
            and pil_image_to_render_size != (0, 0)
            and self.application_model.image_data.exact_boundaries is not None
            and not self.overlay_processor.is_boundary_cache_current(
                image_size=pil_image_to_render_size,
                cell_ids_to_draw=set(current_ids_tuple_for_display),
                show_deselected_state=self.application_model.display_state.show_deselected_masks_only,
            )
        ):
            needs_boundary_worker_request = True
            log("ImageViewRenderer: Boundary cache regeneration needed.", level="DEBUG")

        needs_numbers_worker_request = False
        if (
            self.application_model.display_state.show_cell_numbers
            and pil_image_to_render_size != (0, 0)
            and current_ids_tuple_for_display
            and not self.overlay_processor.is_number_positions_cache_current(
                cell_ids_to_draw=set(current_ids_tuple_for_display),
                show_deselected_state=self.application_model.display_state.show_deselected_masks_only,
            )
        ):
            needs_numbers_worker_request = True
            log(
                "ImageViewRenderer: Number positions cache regeneration needed.",
                level="DEBUG",
            )

        return {
            "mask": needs_mask_worker_request,
            "boundary": needs_boundary_worker_request,
            "numbers": needs_numbers_worker_request,
        }

    # --- Worker for cache generation ---
    def _threaded_cache_generator(
        self,
        current_ids_tuple_for_display_worker,
        pil_image_to_render_worker_copy,
        needs_mask_worker,
        needs_boundary_worker,
        needs_numbers_worker,
        current_display_state_for_worker,
    ):
        log(
            f"ImageViewRenderer (worker): Thread started. Needs mask: {needs_mask_worker}, boundary: {needs_boundary_worker}, numbers: {needs_numbers_worker}.",
            level="DEBUG",
        )
        log(
            f"ImageViewRenderer (worker): Operating with display state: show_deselected={current_display_state_for_worker.show_deselected_masks_only}, boundary_color='{current_display_state_for_worker.boundary_color_name}'.",
            level="DEBUG",
        )
        log(
            f"ImageViewRenderer (worker): Image size for cache gen: {pil_image_to_render_worker_copy.size if pil_image_to_render_worker_copy else 'None'}, Num IDs for display: {len(current_ids_tuple_for_display_worker)}",
            level="DEBUG",
        )

        try:
            if needs_mask_worker and pil_image_to_render_worker_copy:
                log(
                    "ImageViewRenderer (worker): Requesting mask RGB layer cache regeneration from OverlayProcessor.",
                    level="DEBUG",
                )
                # This call populates/updates the cache within overlay_processor
                self.overlay_processor.get_cached_mask_layer_rgb(
                    base_pil_image_size=pil_image_to_render_worker_copy.size,
                    cell_ids_to_draw=set(current_ids_tuple_for_display_worker),
                    show_deselected_masks_only=current_display_state_for_worker.show_deselected_masks_only,
                )
                log(
                    "ImageViewRenderer (worker): Mask RGB layer cache regeneration request completed with OverlayProcessor.",
                    level="DEBUG",
                )

            if needs_boundary_worker and pil_image_to_render_worker_copy:
                if self.application_model.image_data.exact_boundaries is not None:
                    log(
                        "ImageViewRenderer (worker): Requesting boundary L PIL cache regeneration from OverlayProcessor.",
                        level="DEBUG",
                    )
                    # This call populates/updates the cache within overlay_processor
                    self.overlay_processor.get_boundary_mask_L_pil(
                        image_size=pil_image_to_render_worker_copy.size,
                        cell_ids_to_draw=set(current_ids_tuple_for_display_worker),
                        show_deselected_state=current_display_state_for_worker.show_deselected_masks_only,
                    )
                    log(
                        "ImageViewRenderer (worker): Boundary L PIL cache regeneration request completed with OverlayProcessor.",
                        level="DEBUG",
                    )
                else:
                    log(
                        "ImageViewRenderer (worker): Exact boundaries became None. Skipping boundary cache generation.",
                        level="WARNING",
                    )

            if needs_numbers_worker and pil_image_to_render_worker_copy:
                log(
                    "ImageViewRenderer (worker): Requesting number positions cache regeneration from OverlayProcessor.",
                    level="DEBUG",
                )
                # This call populates/updates the cache within overlay_processor
                self.overlay_processor.get_cached_cell_number_positions(
                    cell_ids_to_draw=set(current_ids_tuple_for_display_worker),
                    show_deselected_state=current_display_state_for_worker.show_deselected_masks_only,
                )
                log(
                    "ImageViewRenderer (worker): Number positions cache regeneration request completed with OverlayProcessor.",
                    level="DEBUG",
                )

            # Update last known states after worker attempts cache generation
            self._last_known_show_deselected_for_cache = (
                current_display_state_for_worker.show_deselected_masks_only
            )
            self._last_known_boundary_color_name_for_render = (
                current_display_state_for_worker.boundary_color_name
            )

        except Exception as e:
            log(
                f"ImageViewRenderer (worker): Error during cache generation: {e}",
                level="ERROR",
            )
        finally:
            self._cache_generation_active = False
            log(
                "ImageViewRenderer (worker): Thread finished cache generation.",
                level="DEBUG",
            )
            if self.image_canvas:  # Ensure canvas still exists
                self.image_canvas.after(
                    0, lambda: self.render(quality="final", _is_post_worker_call=True)
                )
                log(
                    "ImageViewRenderer (worker): Scheduled final render after cache generation.",
                    level="DEBUG",
                )

    def render(self, quality="final", _is_post_worker_call=False):
        log(
            f"ImageViewRenderer.render called with quality: {quality}, post_worker: {_is_post_worker_call}",
            level="DEBUG",
        )
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            log(
                f"ImageViewRenderer: Canvas not ready (width={canvas_width}, height={canvas_height}). Scheduling retry for render.",
                level="DEBUG",
            )
            if self._update_display_retry_id:
                self.image_canvas.after_cancel(self._update_display_retry_id)
            self._update_display_retry_id = self.image_canvas.after(
                constants.UI_RENDER_RETRY_DELAY_MS,
                lambda q=quality: self.render(quality=q),
            )
            return

        # Fetch the (potentially processed) image from ApplicationModel
        pil_image_to_render = self.application_model.get_processed_image_for_display()
        log(
            f"ImageViewRenderer: Fetched pil_image_to_render (id: {id(pil_image_to_render)}, size: {pil_image_to_render.size if pil_image_to_render else 'None'}) for display.",
            level="DEBUG",
        )

        if pil_image_to_render is None:
            self.image_canvas.delete("all")
            log(
                "ImageViewRenderer: No image to render. Displaying 'Select Image' prompt.",
                level="INFO",
            )
            text_color = "white"  # Default
            # Accessing theme for text color if main app uses CTk Theming
            # This assumes parent_frame is the main app or has access to theme manager
            if (
                hasattr(self.parent_frame, "_apply_appearance_mode")
                and hasattr(ctk, "ThemeManager")
                and ctk.ThemeManager.theme
            ):
                try:
                    text_color = self.parent_frame._apply_appearance_mode(
                        ctk.ThemeManager.theme["CTkLabel"]["text_color"]
                    )
                except (KeyError, TypeError):
                    pass
            self.image_canvas.create_text(
                canvas_width / 2,
                canvas_height / 2,
                text=constants.MSG_SELECT_IMAGE_PROMPT,
                fill=text_color,
                font=ctk.CTkFont(size=16),
            )
            self.tk_image_on_canvas = None
            # Update stats label
            if hasattr(self.parent_frame, "_update_stats_label"):
                self.parent_frame._update_stats_label()
            elif (
                hasattr(self.parent_frame, "stats_label")
                and self.parent_frame.stats_label
            ):  # Direct access fallback
                self.parent_frame.stats_label.configure(
                    text=constants.UI_TEXT_STATS_LABEL_DEFAULT
                )
            return

        zoom, pan_x, pan_y = self.application_model.pan_zoom_state.get_params()
        log(
            f"ImageViewRenderer: Rendering with zoom={zoom}, pan=({pan_x},{pan_y})",
            level="DEBUG",
        )

        new_width = int(pil_image_to_render.width * zoom)
        new_height = int(pil_image_to_render.height * zoom)

        if new_width <= 0 or new_height <= 0:
            log(
                f"ImageViewRenderer: Calculated new_width ({new_width}) or new_height ({new_height}) is <= 0. Clearing canvas.",
                level="WARNING",
            )
            self.image_canvas.delete("all")
            self.tk_image_on_canvas = None
            return

        # --- Cache Regeneration Logic ---
        current_mask_array_for_ids = self.application_model.image_data.mask_array
        all_mask_ids_for_ids = set()
        if (
            current_mask_array_for_ids is not None
            and current_mask_array_for_ids.size > 0
        ):
            unique_ids_for_ids = np.unique(current_mask_array_for_ids)
            all_mask_ids_for_ids = set(unique_ids_for_ids[unique_ids_for_ids != 0])

        ids_to_process_for_display_local = set()
        show_deselected_mode_local = (
            self.application_model.display_state.show_deselected_masks_only
        )
        if show_deselected_mode_local:
            deselected_ids_local = (
                all_mask_ids_for_ids - self.application_model.image_data.included_cells
            )
            ids_to_process_for_display_local = deselected_ids_local
        else:
            ids_to_process_for_display_local = (
                self.application_model.image_data.included_cells.intersection(
                    all_mask_ids_for_ids
                )
            )

        current_ids_tuple_for_display = tuple(
            sorted(list(ids_to_process_for_display_local))
        )
        log(
            f"ImageViewRenderer: Determined current_ids_tuple_for_display (count: {len(current_ids_tuple_for_display)}) for overlays. show_deselected_mode_local: {show_deselected_mode_local}",
            level="DEBUG",
        )
        # --- End calculation for current_ids_tuple_for_display ---

        cache_needs_dict = self._needs_cache_regeneration(
            current_ids_tuple_for_display,
            pil_image_to_render.size if pil_image_to_render else (0, 0),
        )
        needs_any_cache_regeneration = any(cache_needs_dict.values())

        can_start_worker = (
            not _is_post_worker_call
            and needs_any_cache_regeneration
            and not self._cache_generation_active
        )

        if can_start_worker and quality == "final":  # Worker only for final quality
            self._cache_generation_active = True
            log(
                "ImageViewRenderer: Cache regeneration needed for FINAL render, starting worker thread.",
                level="INFO",
            )

            pil_image_copy_for_worker = (
                pil_image_to_render.copy() if pil_image_to_render else None
            )
            # Pass a snapshot of relevant display_state items to the worker
            current_display_state_snapshot = self.application_model.display_state

            self._cache_generation_thread = threading.Thread(
                target=self._threaded_cache_generator,
                args=(
                    current_ids_tuple_for_display,
                    pil_image_copy_for_worker,
                    cache_needs_dict["mask"],
                    cache_needs_dict["boundary"],
                    cache_needs_dict["numbers"],
                    current_display_state_snapshot,  # Pass the snapshot
                ),
                daemon=True,
            )
            self._cache_generation_thread.start()

            if quality == "final":
                log(
                    "ImageViewRenderer: Final render initiated cache generation. Displaying PREVIOUS image with TEXT placeholder and returning.",
                    level="DEBUG",
                )

                if pil_image_to_render is not None:
                    if self.tk_image_on_canvas:
                        text_width_approx = 150  # Approximate width for the backdrop
                        text_height_approx = 30  # Approximate height for the backdrop
                        rect_x1 = (canvas_width - text_width_approx) / 2
                        rect_y1 = (canvas_height - text_height_approx) / 2
                        rect_x2 = rect_x1 + text_width_approx
                        rect_y2 = rect_y1 + text_height_approx

                        # Ensure "updating_text_item" is a unique tag
                        self.image_canvas.delete("updating_text_item")
                        self.image_canvas.create_rectangle(
                            rect_x1,
                            rect_y1,
                            rect_x2,
                            rect_y2,
                            fill="#404040",  # Dark grey, somewhat transparent if possible by stipple or actual alpha in future
                            outline="",
                            tags="updating_text_item",
                        )

                        self.image_canvas.create_text(
                            canvas_width / 2,
                            canvas_height / 2,
                            text="Updating overlays...",
                            fill="white",
                            font=ctk.CTkFont(size=16),
                            tags="updating_text_item",
                        )
                    else:
                        # No previous tk_image_on_canvas, render a quick base placeholder.
                        self.image_canvas.delete("all")
                        # Synchronously render a simplified base image (panned/zoomed original or black)

                        l_new_width = 0
                        l_new_height = 0
                        if (
                            pil_image_to_render
                        ):  # Check again, though outer check exists
                            l_new_width = int(pil_image_to_render.width * zoom)
                            l_new_height = int(pil_image_to_render.height * zoom)

                        l_safe_new_width = max(1, l_new_width)
                        l_safe_new_height = max(1, l_new_height)

                        temp_placeholder_pil = None
                        if (
                            l_safe_new_width > 0 and l_safe_new_height > 0
                        ):  # Use local safe_new_width/height
                            try:
                                temp_zoomed_image = pil_image_to_render.resize(
                                    (l_safe_new_width, l_safe_new_height), Image.NEAREST
                                )
                                temp_placeholder_pil = Image.new(
                                    "RGB",
                                    (canvas_width, canvas_height),
                                    constants.COLOR_BLACK_STR,
                                )

                                l_src_x1 = int(-pan_x) if pan_x < 0 else 0
                                l_paste_dst_x = int(pan_x) if pan_x > 0 else 0
                                l_src_y1 = int(-pan_y) if pan_y < 0 else 0
                                l_paste_dst_y = int(pan_y) if pan_y > 0 else 0
                                l_w_copy = min(
                                    temp_zoomed_image.width - l_src_x1,
                                    canvas_width - l_paste_dst_x,
                                )
                                l_h_copy = min(
                                    temp_zoomed_image.height - l_src_y1,
                                    canvas_height - l_paste_dst_y,
                                )

                                if l_w_copy > 0 and l_h_copy > 0:
                                    l_crop_box = (
                                        l_src_x1,
                                        l_src_y1,
                                        l_src_x1 + l_w_copy,
                                        l_src_y1 + l_h_copy,
                                    )
                                    temp_cropped_visible = temp_zoomed_image.crop(
                                        l_crop_box
                                    )
                                    temp_placeholder_pil.paste(
                                        temp_cropped_visible,
                                        (l_paste_dst_x, l_paste_dst_y),
                                    )

                                if not self.application_model.display_state.show_original_image:
                                    log(
                                        "ImageViewRenderer: show_original_image is false, using black placeholder.",
                                        level="DEBUG",
                                    )
                                    temp_placeholder_pil = Image.new(
                                        "RGB",
                                        (canvas_width, canvas_height),
                                        constants.COLOR_BLACK_STR,
                                    )

                                if temp_placeholder_pil:
                                    self.tk_image_on_canvas = ImageTk.PhotoImage(
                                        temp_placeholder_pil
                                    )
                                    self.image_canvas.create_image(
                                        0, 0, anchor="nw", image=self.tk_image_on_canvas
                                    )
                                else:
                                    self.image_canvas.create_rectangle(
                                        0,
                                        0,
                                        canvas_width,
                                        canvas_height,
                                        fill=constants.COLOR_BLACK_STR,
                                        outline="",
                                    )

                            except Exception as e_sync_placeholder:
                                log(
                                    f"Error during sync placeholder render: {e_sync_placeholder}",
                                    level="ERROR",
                                )
                                self.image_canvas.create_rectangle(
                                    0,
                                    0,
                                    canvas_width,
                                    canvas_height,
                                    fill=constants.COLOR_BLACK_STR,
                                    outline="",
                                )  # Fallback black
                        else:
                            self.image_canvas.create_rectangle(
                                0,
                                0,
                                canvas_width,
                                canvas_height,
                                fill=constants.COLOR_BLACK_STR,
                                outline="",
                            )  # Fallback black

                        text_width_approx = 150
                        text_height_approx = 30
                        rect_x1 = (canvas_width - text_width_approx) / 2
                        rect_y1 = (canvas_height - text_height_approx) / 2
                        rect_x2 = rect_x1 + text_width_approx
                        rect_y2 = rect_y1 + text_height_approx
                        self.image_canvas.create_rectangle(
                            rect_x1,
                            rect_y1,
                            rect_x2,
                            rect_y2,
                            fill="#404040",
                            outline="",
                        )
                        self.image_canvas.create_text(
                            canvas_width / 2,
                            canvas_height / 2,
                            text="Updating overlays...",
                            fill="white",
                            font=ctk.CTkFont(size=16),
                        )

                return  # Return after displaying placeholder

        elif self._cache_generation_active and quality == "final":
            log(
                "ImageViewRenderer: Cache generation already active. Deferring this 'final' render call.",
                level="DEBUG",
            )
            return
        elif self._cache_generation_active and quality == "interactive":
            log(
                "ImageViewRenderer: Cache generation active. Interactive render will use current caches or skip.",
                level="DEBUG",
            )
        log(
            f"ImageViewRenderer: Calling generate_composite_overlay_image. Base img size: {pil_image_to_render.size if pil_image_to_render else 'None'}, IDs count: {len(current_ids_tuple_for_display)}, zoom: {zoom:.2f}, quality: {quality}",
            level="DEBUG",
        )
        base_display_img = self.overlay_processor.generate_composite_overlay_image(
            base_pil_image=pil_image_to_render,  # Full res (processed) original image
            image_size_pil=pil_image_to_render.size,  # Its size
            display_state=self.application_model.display_state,  # Pass the live display state object
            ids_to_display=set(
                current_ids_tuple_for_display
            ),  # The IDs that should be shown
            zoom=zoom,
            pan_x=pan_x,
            pan_y=pan_y,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            quality=quality,
        )

        # --- Ruler Drawing ---
        if (
            self.application_model.display_state.show_ruler
            and pil_image_to_render
            and self.application_model.image_data.scale_conversion
            and hasattr(self.application_model.image_data.scale_conversion, "X")
            and self.application_model.image_data.scale_conversion.X is not None
            and self.application_model.image_data.scale_conversion.X != 0
        ):
            log(
                f"ImageViewRenderer: Adding scale bar. Effective display zoom: {zoom:.2f}, Target microns: 10.0",
                level="DEBUG",
            )
            if base_display_img.mode != "RGB":  # Ensure image is RGB for color drawing
                base_display_img = base_display_img.convert("RGB")
            # Call the overlay processor to draw the scale bar
            base_display_img = self.overlay_processor.draw_scale_bar(
                image_to_draw_on=base_display_img,
                effective_display_zoom=zoom,  # Current canvas zoom
                scale_conversion_obj=self.application_model.image_data.scale_conversion,
                target_image_width=base_display_img.width,  # The final image canvas view width
                target_image_height=base_display_img.height,  # The final image canvas view height
            )
            log(
                f"ImageViewRenderer: Scale bar drawing complete. Image mode after: {base_display_img.mode}",
                level="DEBUG",
            )

        # --- Diameter Aid Drawing ---
        if (
            self.application_model.display_state.show_diameter_aid
            and pil_image_to_render
        ):
            diameter_val = self.application_model.segmentation_diameter
            if diameter_val:
                try:
                    diameter = int(diameter_val)
                    if diameter > 0:
                        log(
                            f"ImageViewRenderer: Adding diameter aid. Diameter: {diameter}",
                            level="DEBUG",
                        )
                        if base_display_img.mode != "RGB":
                            base_display_img = base_display_img.convert("RGB")

                        base_display_img = self.overlay_processor.draw_diameter_aid(
                            image_to_draw_on=base_display_img,
                            effective_display_zoom=zoom,
                            diameter_pixels=diameter,
                            target_image_width=base_display_img.width,
                            target_image_height=base_display_img.height,
                        )
                        log(
                            "ImageViewRenderer: Diameter aid drawing complete.",
                            level="DEBUG",
                        )
                except (ValueError, TypeError):
                    log(
                        f"ImageViewRenderer: Could not use diameter value '{diameter_val}' for aid.",
                        level="WARNING",
                    )

        if pil_image_to_render:
            local_new_width = int(pil_image_to_render.width * zoom)
            local_new_height = int(pil_image_to_render.height * zoom)
            local_safe_new_width = max(1, local_new_width)
            local_safe_new_height = max(1, local_new_height)

            src_x1_on_zoomed_img = int(-pan_x) if pan_x < 0 else 0
            paste_dst_x_on_canvas = int(pan_x) if pan_x > 0 else 0
            src_y1_on_zoomed_img = int(-pan_y) if pan_y < 0 else 0
            paste_dst_y_on_canvas = int(pan_y) if pan_y > 0 else 0

            width_to_copy = min(
                local_safe_new_width - src_x1_on_zoomed_img,
                canvas_width - paste_dst_x_on_canvas,
            )
            height_to_copy = min(
                local_safe_new_height - src_y1_on_zoomed_img,
                canvas_height - paste_dst_y_on_canvas,
            )
            crop_box = None
            if width_to_copy > 0 and height_to_copy > 0:
                crop_box = (
                    src_x1_on_zoomed_img,
                    src_y1_on_zoomed_img,
                    src_x1_on_zoomed_img + width_to_copy,
                    src_y1_on_zoomed_img + height_to_copy,
                )
        else:  # No pil_image_to_render, so drawing feedback likely won't map meaningfully
            src_x1_on_zoomed_img = 0  # Default values to prevent NameErrors if drawing mode is somehow active
            src_y1_on_zoomed_img = 0
            paste_dst_x_on_canvas = 0
            paste_dst_y_on_canvas = 0
            crop_box = None

        if (
            self.parent_frame.drawing_controller.drawing_mode_active
            and self.parent_frame.drawing_controller.current_draw_points
        ):
            if base_display_img.mode != "RGB":
                base_display_img = base_display_img.convert("RGB")
            log(
                "ImageViewRenderer: Drawing polygon feedback for active drawing mode.",
                level="DEBUG",
            )

            draw_on_canvas_view = ImageDraw.Draw(base_display_img)

            # Transform raw user-clicked points to canvas space for dot rendering
            # and for drawing lines connecting them directly.
            all_potential_canvas_points = []  # Stores all points transformed to canvas space
            visible_canvas_dots = []  # Store only points visible on canvas (list of dicts)

            for index, (orig_x, orig_y) in enumerate(
                self.parent_frame.drawing_controller.current_draw_points
            ):
                zoomed_pt_x = orig_x * zoom
                zoomed_pt_y = orig_y * zoom
                rel_zoomed_x = zoomed_pt_x - (src_x1_on_zoomed_img if crop_box else 0)
                rel_zoomed_y = zoomed_pt_y - (src_y1_on_zoomed_img if crop_box else 0)
                cv_x = rel_zoomed_x + paste_dst_x_on_canvas
                cv_y = rel_zoomed_y + paste_dst_y_on_canvas
                all_potential_canvas_points.append((cv_x, cv_y))

                if (
                    crop_box
                    and 0 <= rel_zoomed_x < crop_box[2] - crop_box[0]
                    and 0 <= rel_zoomed_y < crop_box[3] - crop_box[1]
                ):
                    visible_canvas_dots.append({"x": cv_x, "y": cv_y, "index": index})

            # Draw dots for actual user clicks (visible ones)
            if visible_canvas_dots:
                num_total_points = len(
                    self.parent_frame.drawing_controller.current_draw_points
                )
                for point_info in visible_canvas_dots:
                    pt_x, pt_y, current_index = (
                        point_info["x"],
                        point_info["y"],
                        point_info["index"],
                    )
                    point_color = self.draw_feedback_color
                    if num_total_points == 1 and current_index == 0:
                        point_color = self.draw_first_point_color
                    elif num_total_points > 1:
                        if current_index == 0:
                            point_color = self.draw_first_point_color
                        elif current_index == num_total_points - 1:
                            point_color = self.draw_last_point_color
                    draw_on_canvas_view.ellipse(
                        (
                            pt_x - self.draw_point_radius,
                            pt_y - self.draw_point_radius,
                            pt_x + self.draw_point_radius,
                            pt_y + self.draw_point_radius,
                        ),
                        fill=point_color,
                        outline=point_color,
                    )

            # Draw lines connecting ALL potential points (ImageDraw will clip)
            # This ensures lines extend off-canvas correctly if points are outside view.
            if len(all_potential_canvas_points) > 1:
                draw_on_canvas_view.line(
                    all_potential_canvas_points,
                    fill=self.draw_feedback_color,
                    width=constants.DRAW_FEEDBACK_LINE_WIDTH,
                )

            # Draw closing line based on TRUE first and last points of the WHOLE polygon
            if (
                len(self.parent_frame.drawing_controller.current_draw_points) >= 2
            ):  # Use original point count for this decision
                first_canvas_pt = all_potential_canvas_points[0]
                last_canvas_pt = all_potential_canvas_points[-1]
                draw_on_canvas_view.line(
                    [last_canvas_pt, first_canvas_pt],
                    fill=self.draw_feedback_color,
                    width=constants.DRAW_FEEDBACK_LINE_WIDTH,
                    joint="curve",
                )
        # --- END DRAWING FEEDBACK ---

        # Update stats label
        if hasattr(
            self.parent_frame, "_update_stats_label"
        ):  # Check if parent_frame has the method
            self.parent_frame._update_stats_label()
        elif (
            hasattr(self.parent_frame, "stats_label") and self.parent_frame.stats_label
        ):  # Fallback for direct access
            # This block is essentially the same as above, could be refactored
            # This part calculates stats and directly configures the label, which is fine for a fallback.
            total_cells_in_mask = 0
            if (
                self.application_model.image_data.mask_array is not None
                and self.application_model.image_data.mask_array.size > 0
            ):
                unique_ids_in_mask = np.unique(
                    self.application_model.image_data.mask_array
                )
                total_cells_in_mask = len(unique_ids_in_mask[unique_ids_in_mask != 0])

            actual_user_drawn_ids_in_current_mask = (
                self.application_model.image_data.user_drawn_cell_ids.intersection(
                    set(np.unique(self.application_model.image_data.mask_array))
                    if self.application_model.image_data.mask_array is not None
                    else set()
                )
            )
            user_drawn_count = len(actual_user_drawn_ids_in_current_mask)

            model_found_count = total_cells_in_mask - user_drawn_count
            model_found_count = max(0, model_found_count)

            selected_count = len(self.application_model.image_data.included_cells)

            stats_text = (
                f"Cell count:\n"
                f"  Model Found: {model_found_count}\n"
                f"  User Drawn: {user_drawn_count}\n"
                f"  Total Unique: {total_cells_in_mask}\n"
                f"  Selected: {selected_count}"
            )
            self.parent_frame.stats_label.configure(text=stats_text)

        # Ensure base_display_img has valid dimensions before creating PhotoImage
        if base_display_img.width > 0 and base_display_img.height > 0:
            self.tk_image_on_canvas = ImageTk.PhotoImage(base_display_img)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(
                0, 0, anchor="nw", image=self.tk_image_on_canvas
            )
            log(
                "ImageViewRenderer: Successfully updated canvas with new image.",
                level="INFO",
            )
        else:
            # This case might happen if new_width/new_height was 0 and pil_image_to_render was not None
            # Or if base_display_img itself became 0x0 for some reason.
            log(
                f"ImageViewRenderer: base_display_img has invalid dimensions ({base_display_img.width}x{base_display_img.height}). Clearing canvas.",
                level="WARNING",
            )
            self.image_canvas.delete("all")
            self.tk_image_on_canvas = None
