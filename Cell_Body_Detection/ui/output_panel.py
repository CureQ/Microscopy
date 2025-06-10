import threading

import customtkinter as ctk
import numpy as np
from CTkMessagebox import CTkMessagebox

from .. import constants
from ..utils.debug_logger import log
from .base_panel import BasePanel
from .tooltip import ToolTip


class OutputPanel(BasePanel):
    def __init__(self, parent, application_model, file_io_controller):
        super().__init__(parent, application_model, width=constants.UI_SIDEPANEL_WIDTH)
        self.file_io_controller = file_io_controller

        # --- UI Elements ---
        self.stats_label = None
        self.pdf_cb_masks_only = None
        self.pdf_cb_boundaries_only = None
        self.pdf_cb_numbers_only = None
        self.pdf_cb_masks_boundaries = None
        self.pdf_cb_masks_numbers = None
        self.pdf_cb_boundaries_numbers = None
        self.pdf_cb_masks_boundaries_numbers = None
        self.pdf_switch_include_scale_bar = None
        self.export_selected_button = None
        self.export_selected_progressbar = None
        self.export_tif_button = None
        self.export_tif_progressbar = None
        self.export_pdf_button = None
        self.export_pdf_progressbar = None

        # --- UI Variables ---
        self._setup_ui_variables()

        # --- Event Handlers for Scale Bar ---
        self._show_scale_info_error_pdf_handler = (
            lambda e: self._show_scale_info_error_pdf()
        )

        # --- Create Widgets ---
        self._create_widgets()

    def _setup_ui_variables(self):
        """Initializes all UI state variables (CTk Variables) for the output panel."""
        self.traced_ui_variables_map = []

        def add_to_map_and_return(var, path_tuple):
            self.traced_ui_variables_map.append((var, path_tuple))
            return var

        # --- PDF Export Options ---
        self.pdf_opt_masks_only_var = add_to_map_and_return(
            self._setup_traced_view_option_var(
                ("display_state", "pdf_opt_masks_only"),
                ctk.BooleanVar,
                "pdf_opt_masks_only",
            ),
            ("display_state", "pdf_opt_masks_only"),
        )
        self.pdf_opt_boundaries_only_var = add_to_map_and_return(
            self._setup_traced_view_option_var(
                ("display_state", "pdf_opt_boundaries_only"),
                ctk.BooleanVar,
                "pdf_opt_boundaries_only",
            ),
            ("display_state", "pdf_opt_boundaries_only"),
        )
        self.pdf_opt_numbers_only_var = add_to_map_and_return(
            self._setup_traced_view_option_var(
                ("display_state", "pdf_opt_numbers_only"),
                ctk.BooleanVar,
                "pdf_opt_numbers_only",
            ),
            ("display_state", "pdf_opt_numbers_only"),
        )
        self.pdf_opt_masks_boundaries_var = add_to_map_and_return(
            self._setup_traced_view_option_var(
                ("display_state", "pdf_opt_masks_boundaries"),
                ctk.BooleanVar,
                "pdf_opt_masks_boundaries",
            ),
            ("display_state", "pdf_opt_masks_boundaries"),
        )
        self.pdf_opt_masks_numbers_var = add_to_map_and_return(
            self._setup_traced_view_option_var(
                ("display_state", "pdf_opt_masks_numbers"),
                ctk.BooleanVar,
                "pdf_opt_masks_numbers",
            ),
            ("display_state", "pdf_opt_masks_numbers"),
        )
        self.pdf_opt_boundaries_numbers_var = add_to_map_and_return(
            self._setup_traced_view_option_var(
                ("display_state", "pdf_opt_boundaries_numbers"),
                ctk.BooleanVar,
                "pdf_opt_boundaries_numbers",
            ),
            ("display_state", "pdf_opt_boundaries_numbers"),
        )
        self.pdf_opt_masks_boundaries_numbers_var = add_to_map_and_return(
            self._setup_traced_view_option_var(
                ("display_state", "pdf_opt_masks_boundaries_numbers"),
                ctk.BooleanVar,
                "pdf_opt_masks_boundaries_numbers",
            ),
            ("display_state", "pdf_opt_masks_boundaries_numbers"),
        )
        self.pdf_include_scale_bar_var = ctk.BooleanVar(
            value=self.application_model.display_state.pdf_include_scale_bar
        )

    def _create_widgets(self):
        log("OutputPanel._create_widgets execution started.", level="DEBUG")
        ctk.CTkLabel(
            self,
            text=constants.UI_TEXT_OUTPUT_PANEL_TITLE,
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=10, padx=10, fill="x")

        self.stats_label = ctk.CTkLabel(
            self,
            text=constants.UI_TEXT_STATS_LABEL_DEFAULT,
            justify="left",
            anchor="w",
        )
        self.stats_label.pack(padx=10, pady=(5, 10), fill="x", anchor="n")
        ToolTip(self.stats_label, constants.TOOLTIP_STATS_LABEL)

        export_frame = ctk.CTkFrame(self, fg_color="transparent")
        export_frame.pack(padx=10, pady=10, fill="x", expand=True, anchor="s")

        ctk.CTkLabel(
            export_frame,
            text=constants.UI_TEXT_PDF_EXPORT_OPTIONS_LABEL,
            wraplength=constants.UI_FILENAME_LABEL_WRAPLENGTH,
        ).pack(anchor="w", pady=(0, 2))

        self.pdf_cb_masks_only = ctk.CTkCheckBox(
            export_frame,
            text=constants.UI_TEXT_PDF_MASKS_ONLY,
            variable=self.pdf_opt_masks_only_var,
        )
        self.pdf_cb_masks_only.pack(anchor="w", pady=1)
        ToolTip(self.pdf_cb_masks_only, constants.TOOLTIP_PDF_MASKS_ONLY)

        self.pdf_cb_boundaries_only = ctk.CTkCheckBox(
            export_frame,
            text=constants.UI_TEXT_PDF_BOUNDARIES_ONLY,
            variable=self.pdf_opt_boundaries_only_var,
        )
        self.pdf_cb_boundaries_only.pack(anchor="w", pady=1)
        ToolTip(self.pdf_cb_boundaries_only, constants.TOOLTIP_PDF_BOUNDARIES_ONLY)

        self.pdf_cb_numbers_only = ctk.CTkCheckBox(
            export_frame,
            text=constants.UI_TEXT_PDF_NUMBERS_ONLY,
            variable=self.pdf_opt_numbers_only_var,
        )
        self.pdf_cb_numbers_only.pack(anchor="w", pady=1)
        ToolTip(self.pdf_cb_numbers_only, constants.TOOLTIP_PDF_NUMBERS_ONLY)

        self.pdf_cb_masks_boundaries = ctk.CTkCheckBox(
            export_frame,
            text=constants.UI_TEXT_PDF_MASKS_BOUNDARIES,
            variable=self.pdf_opt_masks_boundaries_var,
        )
        self.pdf_cb_masks_boundaries.pack(anchor="w", pady=1)
        ToolTip(self.pdf_cb_masks_boundaries, constants.TOOLTIP_PDF_MASKS_BOUNDARIES)

        self.pdf_cb_masks_numbers = ctk.CTkCheckBox(
            export_frame,
            text=constants.UI_TEXT_PDF_MASKS_NUMBERS,
            variable=self.pdf_opt_masks_numbers_var,
        )
        self.pdf_cb_masks_numbers.pack(anchor="w", pady=1)
        ToolTip(self.pdf_cb_masks_numbers, constants.TOOLTIP_PDF_MASKS_NUMBERS)

        self.pdf_cb_boundaries_numbers = ctk.CTkCheckBox(
            export_frame,
            text=constants.UI_TEXT_PDF_BOUNDARIES_NUMBERS,
            variable=self.pdf_opt_boundaries_numbers_var,
        )
        self.pdf_cb_boundaries_numbers.pack(anchor="w", pady=1)
        ToolTip(
            self.pdf_cb_boundaries_numbers, constants.TOOLTIP_PDF_BOUNDARIES_NUMBERS
        )

        self.pdf_cb_masks_boundaries_numbers = ctk.CTkCheckBox(
            export_frame,
            text=constants.UI_TEXT_PDF_MASKS_BOUNDARIES_NUMBERS,
            variable=self.pdf_opt_masks_boundaries_numbers_var,
        )
        self.pdf_cb_masks_boundaries_numbers.pack(anchor="w", pady=(1, 5))
        ToolTip(
            self.pdf_cb_masks_boundaries_numbers,
            constants.TOOLTIP_PDF_MASKS_BOUNDARIES_NUMBERS,
        )

        self.pdf_switch_include_scale_bar = ctk.CTkSwitch(
            export_frame,
            text=constants.UI_TEXT_PDF_INCLUDE_SCALE_BAR,
            variable=self.pdf_include_scale_bar_var,
            onvalue=True,
            offvalue=False,
        )
        self.pdf_switch_include_scale_bar.pack(anchor="w", pady=(0, 8))
        ToolTip(
            self.pdf_switch_include_scale_bar, constants.TOOLTIP_PDF_INCLUDE_SCALE_BAR
        )
        self.pdf_include_scale_bar_var.trace_add(
            "write", self._on_pdf_include_scale_bar_change
        )
        self.update_pdf_scale_bar_state()

        # --- Export Buttons ---
        export_selected_action_frame = ctk.CTkFrame(
            export_frame, fg_color="transparent"
        )
        export_selected_action_frame.pack(fill="x", pady=2)
        self.export_selected_button = ctk.CTkButton(
            export_selected_action_frame,
            text=constants.UI_TEXT_EXPORT_SELECTED_CELLS_BUTTON,
            command=self._initiate_export_selected,
        )
        self.export_selected_button.pack(fill="x")
        ToolTip(self.export_selected_button, constants.TOOLTIP_EXPORT_SELECTED)
        self.export_selected_progressbar = ctk.CTkProgressBar(
            export_selected_action_frame, mode="indeterminate"
        )
        ToolTip(
            self.export_selected_progressbar, constants.TOOLTIP_EXPORT_SELECTED_PROGRESS
        )

        export_tif_action_frame = ctk.CTkFrame(export_frame, fg_color="transparent")
        export_tif_action_frame.pack(fill="x", pady=2)
        self.export_tif_button = ctk.CTkButton(
            export_tif_action_frame,
            text=constants.UI_TEXT_EXPORT_VIEW_AS_TIF_BUTTON,
            command=self._initiate_export_tif,
        )
        self.export_tif_button.pack(fill="x")
        ToolTip(self.export_tif_button, constants.TOOLTIP_EXPORT_TIF)
        self.export_tif_progressbar = ctk.CTkProgressBar(
            export_tif_action_frame, mode="indeterminate"
        )
        ToolTip(self.export_tif_progressbar, constants.TOOLTIP_EXPORT_TIF_PROGRESS)

        export_pdf_action_frame = ctk.CTkFrame(export_frame, fg_color="transparent")
        export_pdf_action_frame.pack(fill="x", pady=2)
        self.export_pdf_button = ctk.CTkButton(
            export_pdf_action_frame,
            text=constants.UI_TEXT_EXPORT_PDF_REPORT_BUTTON,
            command=self._initiate_export_pdf,
        )
        self.export_pdf_button.pack(fill="x")
        ToolTip(self.export_pdf_button, constants.TOOLTIP_EXPORT_PDF)
        self.export_pdf_progressbar = ctk.CTkProgressBar(
            export_pdf_action_frame, mode="indeterminate"
        )
        ToolTip(self.export_pdf_progressbar, constants.TOOLTIP_EXPORT_PDF_PROGRESS)

    def sync_ui_variables_with_model(self):
        """Synchronizes the panel's UI variables with the application model state."""
        super().sync_ui_variables_with_model()

        # Sync non-traced vars
        pdf_scale_bar_model = self.application_model.display_state.pdf_include_scale_bar
        if self.pdf_include_scale_bar_var.get() != pdf_scale_bar_model:
            self.pdf_include_scale_bar_var.set(pdf_scale_bar_model)

    def _on_pdf_include_scale_bar_change(self, *args):
        self.application_model.set_view_option(
            "pdf_include_scale_bar", self.pdf_include_scale_bar_var.get()
        )

    def update_pdf_scale_bar_state(self):
        """Disables the PDF scale bar switch if scale info is missing."""
        scale = self.application_model.image_data.scale_conversion
        scale_missing = not (scale and hasattr(scale, "X") and scale.X and scale.X != 0)

        if scale_missing:
            self.pdf_switch_include_scale_bar.deselect()
            self.pdf_switch_include_scale_bar.configure(state="disabled")
            self.pdf_switch_include_scale_bar.unbind("<Button-1>")
            self.pdf_switch_include_scale_bar.bind(
                "<Button-1>", self._show_scale_info_error_pdf_handler
            )
        else:
            self.pdf_switch_include_scale_bar.configure(state="normal")
            self.pdf_switch_include_scale_bar.unbind("<Button-1>")
            self.pdf_include_scale_bar_var.set(constants.PDF_INCLUDE_SCALE_BAR_DEFAULT)

    def update_stats_label(self):
        """Updates the statistics label based on data from ApplicationModel."""
        log("OutputPanel._update_stats_label execution started.", level="DEBUG")
        if not self.stats_label:
            return

        model_state = self.application_model
        total_cells_in_mask = 0
        if (
            model_state.image_data.mask_array is not None
            and model_state.image_data.mask_array.size > 0
        ):
            unique_ids_in_mask = np.unique(model_state.image_data.mask_array)
            total_cells_in_mask = len(unique_ids_in_mask[unique_ids_in_mask != 0])

        actual_user_drawn_ids = model_state.image_data.user_drawn_cell_ids
        if model_state.image_data.mask_array is not None:
            actual_user_drawn_ids = (
                model_state.image_data.user_drawn_cell_ids.intersection(
                    set(np.unique(model_state.image_data.mask_array))
                )
            )

        user_drawn_count = len(actual_user_drawn_ids)
        model_found_count = total_cells_in_mask - user_drawn_count
        model_found_count = max(0, model_found_count)
        selected_count = len(model_state.image_data.included_cells)

        stats_text = (
            f"Cell count:\n"
            f"  Model Found: {model_found_count}\n"
            f"  User Drawn: {user_drawn_count}\n"
            f"  Total Unique: {total_cells_in_mask}\n"
            f"  Selected: {selected_count}"
        )
        self.stats_label.configure(text=stats_text)

    def _show_scale_info_error_pdf(self, *_):
        CTkMessagebox(
            title="Scale Bar Unavailable",
            message=constants.MSG_SCALE_BAR_UNAVAILABLE,
            icon="warning",
        )

    def _initiate_export_process(
        self,
        export_function,
        button,
        progressbar,
        default_text,
        success_message_default,
        failure_title,
    ):
        if not self.application_model.image_data.original_image:
            CTkMessagebox(
                title=constants.MSG_EXPORT_ERROR_TITLE,
                message=constants.MSG_NO_IMAGE_FOR_EXPORT,
                icon="warning",
            )
            return

        # Special check for PDF export, which has an extra condition
        if export_function == self.file_io_controller.export_pdf:
            if (
                self.application_model.image_data.aics_image_obj
                and not self.application_model.image_data.original_image
            ):
                CTkMessagebox(
                    title=constants.MSG_EXPORT_ERROR_TITLE,
                    message="Multi-dimensional image is loaded but current view is not yet generated.",
                    icon="warning",
                )
                return

        button.configure(text=constants.UI_TEXT_EXPORTING)
        self.parent_frame.set_all_buttons_state(False, current_export_button=button)
        button.pack_configure(pady=(0, 5))
        progressbar.pack(fill="x", pady=(0, 0))
        progressbar.start()

        def worker():
            success, message = False, ""
            try:
                message = export_function()
                if message is None:
                    log(f"Export '{default_text}' cancelled by user.")
                else:
                    success = True
                    log(f"Export '{default_text}' successful: {message}")
            except Exception as e:
                message = f"Failed to export {default_text.lower()}: {str(e)}"
                log(message, level="ERROR")
            self.after(0, lambda: handle_result(success, message))

        def handle_result(success, message):
            progressbar.stop()
            progressbar.pack_forget()
            button.pack_configure(pady=(0, 0))
            button.configure(text=default_text)
            self.parent_frame.set_all_buttons_state(True)

            if success:
                CTkMessagebox(
                    title=constants.MSG_EXPORT_SUCCESS_TITLE,
                    message=message or success_message_default,
                    icon="check",
                )
            elif (
                message
            ):  # Only show message if it's not None (i.e., not a user cancellation)
                CTkMessagebox(title=failure_title, message=message, icon="cancel")

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _initiate_export_selected(self):
        self._initiate_export_process(
            export_function=self.file_io_controller.export_selected,
            button=self.export_selected_button,
            progressbar=self.export_selected_progressbar,
            default_text=constants.UI_TEXT_EXPORT_SELECTED_CELLS_BUTTON,
            success_message_default=constants.MSG_EXPORT_SELECTED_SUCCESS_DEFAULT,
            failure_title=constants.MSG_EXPORT_FAILED_TITLE,
        )

    def _initiate_export_tif(self):
        self._initiate_export_process(
            export_function=self.file_io_controller.export_current_view_as_tif,
            button=self.export_tif_button,
            progressbar=self.export_tif_progressbar,
            default_text=constants.UI_TEXT_EXPORT_VIEW_AS_TIF_BUTTON,
            success_message_default=constants.MSG_EXPORT_TIF_SUCCESS_DEFAULT,
            failure_title=constants.MSG_EXPORT_FAILED_TITLE,
        )

    def _initiate_export_pdf(self):
        self._initiate_export_process(
            export_function=self.file_io_controller.export_pdf,
            button=self.export_pdf_button,
            progressbar=self.export_pdf_progressbar,
            default_text=constants.UI_TEXT_EXPORT_PDF_REPORT_BUTTON,
            success_message_default=constants.MSG_EXPORT_PDF_SUCCESS_DEFAULT,
            failure_title=constants.MSG_EXPORT_FAILED_TITLE,
        )
