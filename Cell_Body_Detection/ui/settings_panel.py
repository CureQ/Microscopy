import tkinter.colorchooser as tk_colorchooser

import customtkinter as ctk
from CTkMessagebox import CTkMessagebox

from .. import constants
from ..utils.debug_logger import log
from .base_panel import BasePanel
from .tooltip import ToolTip


class SettingsPanel(BasePanel):
    def __init__(
        self,
        parent,
        application_model,
        file_io_controller,
        drawing_controller,
        display_settings_controller,
    ):
        super().__init__(parent, application_model, width=constants.UI_SIDEPANEL_WIDTH)
        self.parent_frame = parent
        self.file_io_controller = file_io_controller
        self.drawing_controller = drawing_controller
        self.display_settings_controller = display_settings_controller

        # --- UI Elements ---
        self.undo_button = None
        self.redo_button = None
        self.select_image_btn = None
        self.filename_label = None
        self.resolution_label = None
        self.adjust_channel_z_frame = None
        self.channel_z_stack_toggle_btn = None
        self.channel_z_stack_content_frame = None
        self.dynamic_channel_config_frame = None
        self.z_processing_method_menu = None
        self.z_slice_label = None
        self.z_slice_slider = None
        self.reset_channel_z_button = None
        self.adjust_frame = None
        self.brightness_slider = None
        self.contrast_slider = None
        self.colormap_optionmenu = None
        self.reset_display_button = None
        self.diameter_entry_field = None
        self.segment_button = None
        self.segment_progressbar = None
        self.draw_mask_button = None
        self.upload_mask_button = None
        self.show_ruler_cb = None
        self.microns_entry_field = None

        # --- UI Variables ---
        self._setup_ui_variables()

        # --- Event Handlers for Scale Bar ---
        self._show_scale_info_error_handler = lambda e: self._show_scale_info_error()
        self._show_scale_info_error_entry_handler = (
            lambda e: self._show_scale_info_error()
        )

        # --- Create Widgets ---
        self._create_widgets()
        self.update_scale_bar_state()

    def _setup_ui_variables(self):
        self.traced_ui_variables_map = []

        def add_to_map_and_return(var, path_tuple):
            self.traced_ui_variables_map.append((var, path_tuple))
            return var

        # --- Display Options ---
        self.show_original_var = add_to_map_and_return(
            self._setup_traced_view_option_var(
                ("display_state", "show_original_image"),
                ctk.BooleanVar,
                "show_original_image",
            ),
            ("display_state", "show_original_image"),
        )
        self.show_mask_var = add_to_map_and_return(
            self._setup_traced_view_option_var(
                ("display_state", "show_cell_masks"), ctk.BooleanVar, "show_cell_masks"
            ),
            ("display_state", "show_cell_masks"),
        )
        self.show_boundaries_var = add_to_map_and_return(
            self._setup_traced_view_option_var(
                ("display_state", "show_cell_boundaries"),
                ctk.BooleanVar,
                "show_cell_boundaries",
            ),
            ("display_state", "show_cell_boundaries"),
        )
        self.boundary_color_var = add_to_map_and_return(
            self._setup_traced_view_option_var(
                ("display_state", "boundary_color_name"),
                ctk.StringVar,
                "boundary_color_name",
            ),
            ("display_state", "boundary_color_name"),
        )
        self.show_only_deselected_var = add_to_map_and_return(
            self._setup_traced_view_option_var(
                ("display_state", "show_deselected_masks_only"),
                ctk.BooleanVar,
                "show_deselected_masks_only",
            ),
            ("display_state", "show_deselected_masks_only"),
        )
        self.show_cell_numbers_var = add_to_map_and_return(
            self._setup_traced_view_option_var(
                ("display_state", "show_cell_numbers"),
                ctk.BooleanVar,
                "show_cell_numbers",
            ),
            ("display_state", "show_cell_numbers"),
        )
        self.show_ruler_var = add_to_map_and_return(
            self._setup_traced_view_option_var(
                ("display_state", "show_ruler"), ctk.BooleanVar, "show_ruler"
            ),
            ("display_state", "show_ruler"),
        )
        self.show_diameter_aid_var = add_to_map_and_return(
            self._setup_traced_view_option_var(
                ("display_state", "show_diameter_aid"),
                ctk.BooleanVar,
                "show_diameter_aid",
            ),
            ("display_state", "show_diameter_aid"),
        )

        # --- Channel and Z-Stack Variables ---
        self.z_processing_method_var = add_to_map_and_return(
            self._setup_traced_view_option_var(
                ("display_state", "z_processing_method"),
                ctk.StringVar,
                "z_processing_method",
                initial_value_override="slice",
            ),
            ("display_state", "z_processing_method"),
        )
        self.current_z_slice_var = add_to_map_and_return(
            self._setup_traced_view_option_var(
                ("display_state", "current_z_slice_index"),
                ctk.IntVar,
                "current_z_slice_index",
                initial_value_override=constants.CURRENT_Z_SLICE_INDEX_DEFAULT,
            ),
            ("display_state", "current_z_slice_index"),
        )
        self.z_slice_label_var = ctk.StringVar(value=constants.UI_TEXT_Z_SLICE_NA)

        initial_diameter = self.application_model.segmentation_diameter
        self.diameter_var = ctk.StringVar(
            value=str(initial_diameter) if initial_diameter is not None else "30"
        )
        self.diameter_var.trace_add("write", self._on_diameter_change)

        self.colormap_variable = ctk.StringVar(value="None")

        self.scale_bar_microns_var = ctk.StringVar(
            value=str(int(self.application_model.display_state.scale_bar_microns))
        )
        self.scale_bar_microns_var.trace_add("write", self._on_microns_change)

    def _create_widgets(self):
        log("SettingsPanel._create_widgets execution started.", level="DEBUG")

        # --- Title ---
        ctk.CTkLabel(
            self,
            text=constants.UI_TEXT_SETTINGS_TITLE,
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(10, 15), padx=10, fill="x")

        # --- Navigation ---
        navigation_frame = ctk.CTkFrame(self, fg_color="transparent")
        navigation_frame.pack(padx=10, pady=(0, 10), fill="x")
        ctk.CTkLabel(
            navigation_frame,
            text=constants.UI_TEXT_NAVIGATION,
            font=ctk.CTkFont(weight="bold"),
        ).pack(anchor="w")
        home_frame_ref = getattr(self.parent_frame.parent_app, "home_frame", None)
        nav_cmd = (
            (lambda: self.parent_frame.parent_app.show_frame(home_frame_ref))
            if home_frame_ref
            else (lambda: log("Home frame not found", level="WARNING"))
        )
        ctk.CTkButton(
            navigation_frame,
            text=constants.UI_TEXT_RETURN_TO_START,
            command=nav_cmd,
        ).pack(pady=(5, 0), fill="x")

        # --- History ---
        history_frame = ctk.CTkFrame(self, fg_color="transparent")
        history_frame.pack(padx=10, pady=(10, 10), fill="x")
        ctk.CTkLabel(
            history_frame,
            text=constants.UI_TEXT_HISTORY,
            font=ctk.CTkFont(weight="bold"),
        ).pack(anchor="w")
        self.undo_button = ctk.CTkButton(
            history_frame,
            text=constants.UI_TEXT_UNDO,
            command=self.parent_frame._handle_undo_action,
        )
        self.undo_button.pack(pady=(5, 5), fill="x")
        ToolTip(self.undo_button, constants.TOOLTIP_UNDO)
        self.redo_button = ctk.CTkButton(
            history_frame,
            text=constants.UI_TEXT_REDO,
            command=self.parent_frame._handle_redo_action,
        )
        self.redo_button.pack(pady=(0, 5), fill="x")
        ToolTip(self.redo_button, constants.TOOLTIP_REDO)

        # --- Import Settings ---
        import_frame = ctk.CTkFrame(self, fg_color="transparent")
        import_frame.pack(padx=10, pady=(10, 10), fill="x")
        ctk.CTkLabel(
            import_frame,
            text=constants.UI_TEXT_IMPORT_SETTINGS,
            font=ctk.CTkFont(weight="bold"),
        ).pack(anchor="w")
        self.select_image_btn = ctk.CTkButton(
            import_frame,
            text=constants.UI_TEXT_SELECT_IMAGE,
            command=self.file_io_controller.load_image,
        )
        self.select_image_btn.pack(pady=(5, 5), fill="x")
        ToolTip(self.select_image_btn, constants.TOOLTIP_SELECT_IMAGE)
        self.filename_label = ctk.CTkLabel(
            import_frame,
            text=constants.UI_TEXT_NO_FILE_SELECTED,
            wraplength=constants.UI_FILENAME_LABEL_WRAPLENGTH,
        )
        self.filename_label.pack(pady=(0, 5), fill="x")
        self.resolution_label = ctk.CTkLabel(
            import_frame,
            text=constants.UI_TEXT_RESOLUTION_NA,
            justify="left",
            anchor="w",
        )
        self.resolution_label.pack(padx=10, pady=(5, 0), fill="x", anchor="n")

        # --- Channel & Z-Stack Controls ---
        self.adjust_channel_z_frame = ctk.CTkFrame(self, fg_color="transparent")
        ctk.CTkLabel(
            self.adjust_channel_z_frame, text="", font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w")

        channel_z_stack_folding_frame = ctk.CTkFrame(
            self.adjust_channel_z_frame, fg_color="transparent"
        )
        channel_z_stack_folding_frame.pack(fill="x", pady=(0, 0))

        self.channel_z_stack_expanded = ctk.BooleanVar(value=False)

        def toggle_channel_z_stack_controls():
            if self.channel_z_stack_expanded.get():
                self.channel_z_stack_content_frame.pack(fill="x", pady=(0, 0))
                self.channel_z_stack_toggle_btn.configure(
                    text=constants.UI_TEXT_EXPANDED_PREFIX
                    + constants.UI_TEXT_CHANNEL_Z_STACK_CONTROLS
                )
            else:
                self.channel_z_stack_content_frame.pack_forget()
                self.channel_z_stack_toggle_btn.configure(
                    text=constants.UI_TEXT_COLLAPSED_PREFIX
                    + constants.UI_TEXT_CHANNEL_Z_STACK_CONTROLS
                )

        self.channel_z_stack_toggle_btn = ctk.CTkButton(
            channel_z_stack_folding_frame,
            text=constants.UI_TEXT_COLLAPSED_PREFIX
            + constants.UI_TEXT_CHANNEL_Z_STACK_CONTROLS,
            command=lambda: self.channel_z_stack_expanded.set(
                not self.channel_z_stack_expanded.get()
            ),
            fg_color="transparent",
            hover=False,
            anchor="w",
        )
        self.channel_z_stack_toggle_btn.pack(fill="x", pady=(0, 0))
        self.channel_z_stack_content_frame = ctk.CTkFrame(
            channel_z_stack_folding_frame, fg_color="transparent"
        )

        ctk.CTkLabel(
            self.channel_z_stack_content_frame, text=constants.UI_TEXT_CHANNEL_MAPPING
        ).pack(anchor="w", pady=(5, 2))
        self.dynamic_channel_config_frame = ctk.CTkFrame(
            self.channel_z_stack_content_frame, fg_color="transparent"
        )
        self.dynamic_channel_config_frame.pack(fill="x", expand=True, pady=2)

        ctk.CTkLabel(
            self.channel_z_stack_content_frame,
            text=constants.UI_TEXT_Z_STACK_PROCESSING,
        ).pack(anchor="w", pady=(10, 2))
        self.z_processing_method_menu = ctk.CTkOptionMenu(
            self.channel_z_stack_content_frame,
            variable=self.z_processing_method_var,
            values=constants.Z_PROCESSING_METHOD_OPTIONS,
            command=lambda choice: self.display_settings_controller.set_channel_z_setting(
                "z_processing_method", choice
            ),
        )
        self.z_processing_method_menu.pack(fill="x", pady=(0, 5))
        self.z_processing_method_menu.configure(state="disabled")
        ToolTip(
            self.z_processing_method_menu, constants.TOOLTIP_Z_PROCESSING_METHOD_MENU
        )

        self.z_slice_label = ctk.CTkLabel(
            self.channel_z_stack_content_frame, textvariable=self.z_slice_label_var
        )
        self.z_slice_label.pack(anchor="w", pady=(5, 0))
        self.z_slice_slider = ctk.CTkSlider(
            self.channel_z_stack_content_frame,
            from_=0,
            to=1,
            number_of_steps=1,
            variable=self.current_z_slice_var,
            command=lambda value: self.display_settings_controller.set_channel_z_setting(
                "current_z_slice_index", int(value)
            ),
        )
        self.z_slice_slider.pack(fill="x", pady=(0, 10))
        self.z_slice_slider.configure(state="disabled")
        ToolTip(self.z_slice_slider, constants.TOOLTIP_Z_SLICE_SLIDER)

        self.reset_channel_z_button = ctk.CTkButton(
            self.channel_z_stack_content_frame,
            text=constants.UI_TEXT_RESET_CHANNEL_Z_SETTINGS,
            command=self.display_settings_controller.reset_channel_z_settings,
        )
        self.reset_channel_z_button.pack(fill="x", pady=(10, 5))
        ToolTip(self.reset_channel_z_button, constants.TOOLTIP_RESET_CHANNEL_Z_SETTINGS)
        self.channel_z_stack_expanded.trace_add(
            "write", lambda *args: toggle_channel_z_stack_controls()
        )
        toggle_channel_z_stack_controls()

        # --- Display Adjustments ---
        self.adjust_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.adjust_frame.pack(padx=10, pady=(2, 2), fill="x")

        display_adjustments_folding_frame = ctk.CTkFrame(
            self.adjust_frame, fg_color="transparent"
        )
        display_adjustments_folding_frame.pack(fill="x", pady=(0, 0))
        self.display_adjustments_expanded = ctk.BooleanVar(value=False)

        def toggle_display_adjustments():
            if self.display_adjustments_expanded.get():
                self.display_adjustments_content_frame.pack(fill="x", pady=(0, 0))
                self.display_adjustments_toggle_btn.configure(
                    text=constants.UI_TEXT_EXPANDED_PREFIX
                    + constants.UI_TEXT_DISPLAY_ADJUSTMENTS
                )
            else:
                self.display_adjustments_content_frame.pack_forget()
                self.display_adjustments_toggle_btn.configure(
                    text=constants.UI_TEXT_COLLAPSED_PREFIX
                    + constants.UI_TEXT_DISPLAY_ADJUSTMENTS
                )

        self.display_adjustments_toggle_btn = ctk.CTkButton(
            display_adjustments_folding_frame,
            text=constants.UI_TEXT_COLLAPSED_PREFIX
            + constants.UI_TEXT_DISPLAY_ADJUSTMENTS,
            command=lambda: self.display_adjustments_expanded.set(
                not self.display_adjustments_expanded.get()
            ),
            fg_color="transparent",
            hover=False,
            anchor="w",
        )
        self.display_adjustments_toggle_btn.pack(fill="x", pady=(0, 0))
        self.display_adjustments_content_frame = ctk.CTkFrame(
            display_adjustments_folding_frame, fg_color="transparent"
        )

        ctk.CTkLabel(
            self.display_adjustments_content_frame, text=constants.UI_TEXT_BRIGHTNESS
        ).pack(anchor="w", pady=(5, 0))
        self.brightness_slider = ctk.CTkSlider(
            self.display_adjustments_content_frame,
            from_=1,
            to=100,
            number_of_steps=99,
            command=self.display_settings_controller.set_brightness,
        )
        self.brightness_slider.set(50)
        self.brightness_slider.pack(fill="x", pady=(0, 5))
        ToolTip(self.brightness_slider, constants.TOOLTIP_BRIGHTNESS_SLIDER)

        ctk.CTkLabel(
            self.display_adjustments_content_frame, text=constants.UI_TEXT_CONTRAST
        ).pack(anchor="w", pady=(5, 0))
        self.contrast_slider = ctk.CTkSlider(
            self.display_adjustments_content_frame,
            from_=1,
            to=100,
            number_of_steps=99,
            command=self.display_settings_controller.set_contrast,
        )
        self.contrast_slider.set(50)
        self.contrast_slider.pack(fill="x", pady=(0, 5))
        ToolTip(self.contrast_slider, constants.TOOLTIP_CONTRAST_SLIDER)

        ctk.CTkLabel(
            self.display_adjustments_content_frame, text=constants.UI_TEXT_COLORMAP
        ).pack(anchor="w", pady=(5, 0))
        self.colormap_optionmenu = ctk.CTkOptionMenu(
            self.display_adjustments_content_frame,
            variable=self.colormap_variable,
            values=self.display_settings_controller.get_available_colormaps(),
            command=self.display_settings_controller.set_colormap,
        )
        self.colormap_optionmenu.pack(fill="x", pady=(0, 10))
        ToolTip(self.colormap_optionmenu, constants.TOOLTIP_COLORMAP_MENU)

        self.reset_display_button = ctk.CTkButton(
            self.display_adjustments_content_frame,
            text=constants.UI_TEXT_RESET_DISPLAY_SETTINGS,
            command=self.display_settings_controller.reset_display_settings,
        )
        self.reset_display_button.pack(fill="x", pady=(5, 5))
        ToolTip(self.reset_display_button, constants.TOOLTIP_RESET_DISPLAY_SETTINGS)

        self.display_adjustments_expanded.trace_add(
            "write", lambda *args: toggle_display_adjustments()
        )
        toggle_display_adjustments()

        # --- Mask Creation ---
        mask_creation_folding_frame = ctk.CTkFrame(self, fg_color="transparent")
        mask_creation_folding_frame.pack(padx=10, pady=(2, 2), fill="x")
        self.mask_creation_expanded = ctk.BooleanVar(value=False)

        def toggle_mask_creation():
            if self.mask_creation_expanded.get():
                self.mask_creation_content_frame.pack(fill="x", pady=(0, 0))
                self.mask_creation_toggle_btn.configure(
                    text=constants.UI_TEXT_EXPANDED_PREFIX
                    + constants.UI_TEXT_MASK_CREATION
                )
            else:
                self.mask_creation_content_frame.pack_forget()
                self.mask_creation_toggle_btn.configure(
                    text=constants.UI_TEXT_COLLAPSED_PREFIX
                    + constants.UI_TEXT_MASK_CREATION
                )

        self.mask_creation_toggle_btn = ctk.CTkButton(
            mask_creation_folding_frame,
            text=constants.UI_TEXT_COLLAPSED_PREFIX + constants.UI_TEXT_MASK_CREATION,
            command=lambda: self.mask_creation_expanded.set(
                not self.mask_creation_expanded.get()
            ),
            fg_color="transparent",
            hover=False,
            anchor="w",
        )
        self.mask_creation_toggle_btn.pack(fill="x", pady=(0, 0))
        self.mask_creation_content_frame = ctk.CTkFrame(
            mask_creation_folding_frame, fg_color="transparent"
        )

        # --- Model Settings (Diameter) ---
        model_settings_folding_frame = ctk.CTkFrame(
            self.mask_creation_content_frame, fg_color="transparent"
        )
        model_settings_folding_frame.pack(fill="x", pady=(0, 0))
        self.model_settings_expanded = ctk.BooleanVar(value=False)

        def toggle_model_settings():
            if self.model_settings_expanded.get():
                self.model_settings_content_frame.pack(fill="x", pady=(0, 0))
                self.model_settings_toggle_btn.configure(
                    text=constants.UI_TEXT_MODEL_SETTINGS_EXPANDED
                )
            else:
                self.model_settings_content_frame.pack_forget()
                self.model_settings_toggle_btn.configure(
                    text=constants.UI_TEXT_MODEL_SETTINGS_COLLAPSED
                )

        self.model_settings_toggle_btn = ctk.CTkButton(
            model_settings_folding_frame,
            text=constants.UI_TEXT_MODEL_SETTINGS_COLLAPSED,
            command=lambda: self.model_settings_expanded.set(
                not self.model_settings_expanded.get()
            ),
            fg_color="transparent",
            hover=False,
            anchor="w",
        )
        self.model_settings_toggle_btn.pack(fill="x", pady=(0, 0))
        self.model_settings_content_frame = ctk.CTkFrame(
            model_settings_folding_frame, fg_color="transparent"
        )

        ctk.CTkLabel(
            self.model_settings_content_frame, text=constants.UI_TEXT_DIAMETER_LABEL
        ).pack(anchor="w", pady=(5, 0))
        vcmd = self.register(lambda P: P.isdigit() or P == "")
        self.diameter_entry_field = ctk.CTkEntry(
            self.model_settings_content_frame,
            validate="key",
            validatecommand=(vcmd, "%P"),
            textvariable=self.diameter_var,
        )
        self.diameter_entry_field.bind(
            "<FocusOut>", self._commit_segmentation_diameter_from_entry
        )
        self.diameter_entry_field.bind(
            "<Return>", self._commit_segmentation_diameter_from_entry
        )
        self.diameter_entry_field.pack(pady=(0, 5), fill="x")
        ToolTip(self.diameter_entry_field, constants.TOOLTIP_DIAMETER_ENTRY)

        show_diameter_aid_cb = ctk.CTkCheckBox(
            self.model_settings_content_frame,
            text=constants.UI_TEXT_SHOW_DIAMETER_AID,
            variable=self.show_diameter_aid_var,
        )
        show_diameter_aid_cb.pack(anchor="w", pady=2)
        ToolTip(show_diameter_aid_cb, constants.TOOLTIP_SHOW_DIAMETER_AID)

        self.model_settings_expanded.trace_add(
            "write", lambda *args: toggle_model_settings()
        )
        toggle_model_settings()

        segment_action_frame = ctk.CTkFrame(
            self.mask_creation_content_frame, fg_color="transparent"
        )
        segment_action_frame.pack(pady=(5, 5), fill="x")
        self.segment_button = ctk.CTkButton(
            segment_action_frame,
            text=constants.UI_TEXT_SEGMENT_BUTTON,
            command=self.parent_frame.run_segmentation,
            state="disabled",
        )
        self.segment_button.pack(fill="x", pady=(0, 0))
        ToolTip(self.segment_button, constants.TOOLTIP_SEGMENT_BUTTON)
        self.segment_progressbar = ctk.CTkProgressBar(
            segment_action_frame, mode="indeterminate"
        )

        self.draw_mask_button = ctk.CTkButton(
            self.mask_creation_content_frame,
            text=constants.UI_TEXT_START_DRAWING_BUTTON,
            command=self.drawing_controller._start_drawing_mode,
            state="disabled",
        )
        self.draw_mask_button.pack(pady=(0, 0), fill="x")
        ToolTip(self.draw_mask_button, constants.TOOLTIP_DRAW_MASK_BUTTON)

        self.upload_mask_button = ctk.CTkButton(
            self.mask_creation_content_frame,
            text=constants.UI_TEXT_UPLOAD_MASK_BUTTON,
            command=self.parent_frame._handle_upload_mask_from_file,
            state="disabled",
        )
        self.upload_mask_button.pack(pady=(5, 0), fill="x")
        ToolTip(self.upload_mask_button, constants.TOOLTIP_UPLOAD_MASK_BUTTON)

        self.mask_creation_expanded.trace_add(
            "write", lambda *args: toggle_mask_creation()
        )
        toggle_mask_creation()

        # --- Display Options ---
        display_frame = ctk.CTkFrame(self, fg_color="transparent")
        display_frame.pack(padx=10, pady=(10, 10), fill="x", expand=True, anchor="s")
        ctk.CTkLabel(
            display_frame,
            text=constants.UI_TEXT_DISPLAY_OPTIONS,
            font=ctk.CTkFont(weight="bold"),
        ).pack(anchor="w")

        show_original_cb = ctk.CTkCheckBox(
            display_frame,
            text=constants.UI_TEXT_SHOW_ORIGINAL_IMAGE,
            variable=self.show_original_var,
        )
        show_original_cb.pack(anchor="w", pady=2)
        ToolTip(show_original_cb, constants.TOOLTIP_SHOW_ORIGINAL)
        show_masks_cb = ctk.CTkCheckBox(
            display_frame,
            text=constants.UI_TEXT_SHOW_CELL_MASKS,
            variable=self.show_mask_var,
        )
        show_masks_cb.pack(anchor="w", pady=2)
        ToolTip(show_masks_cb, constants.TOOLTIP_SHOW_MASKS)

        boundaries_row = ctk.CTkFrame(display_frame, fg_color="transparent")
        boundaries_row.pack(fill="x", pady=2)
        show_boundaries_cb = ctk.CTkCheckBox(
            boundaries_row,
            text=constants.UI_TEXT_SHOW_CELL_BOUNDARIES,
            variable=self.show_boundaries_var,
            width=150,
        )
        show_boundaries_cb.pack(side="left", anchor="w")
        ToolTip(show_boundaries_cb, constants.TOOLTIP_SHOW_BOUNDARIES)
        boundary_color_menu = ctk.CTkOptionMenu(
            boundaries_row,
            variable=self.boundary_color_var,
            values=constants.AVAILABLE_BOUNDARY_COLORS,
            width=100,
        )
        boundary_color_menu.pack(side="left", padx=(8, 0), fill="x", expand=True)
        ToolTip(boundary_color_menu, constants.TOOLTIP_BOUNDARY_COLOR)

        show_numbers_cb = ctk.CTkCheckBox(
            display_frame,
            text=constants.UI_TEXT_SHOW_CELL_NUMBERS,
            variable=self.show_cell_numbers_var,
        )
        show_numbers_cb.pack(anchor="w", pady=2)
        ToolTip(show_numbers_cb, constants.TOOLTIP_SHOW_CELL_NUMBERS)

        ruler_row = ctk.CTkFrame(display_frame, fg_color="transparent")
        ruler_row.pack(fill="x", pady=2)
        self.show_ruler_cb = ctk.CTkCheckBox(
            ruler_row,
            text=constants.UI_TEXT_SHOW_RULER,
            variable=self.show_ruler_var,
            width=150,
        )
        self.show_ruler_cb.pack(side="left", anchor="center", pady=0, padx=(0, 8))
        ToolTip(self.show_ruler_cb, constants.TOOLTIP_SHOW_RULER)

        vcmd_microns = self.register(lambda P: P.isdigit() or P == "")
        self.microns_entry_field = ctk.CTkEntry(
            ruler_row,
            textvariable=self.scale_bar_microns_var,
            validate="key",
            validatecommand=(vcmd_microns, "%P"),
            width=100,
        )
        self.microns_entry_field.pack(side="left", anchor="center", padx=(0, 4), pady=0)
        ToolTip(self.microns_entry_field, constants.TOOLTIP_SCALE_BAR_MICRONS)
        ctk.CTkLabel(ruler_row, text=constants.UI_TEXT_MICRONS_LABEL).pack(
            side="left", anchor="center", padx=(0, 0), pady=0
        )
        self.microns_entry_field.bind(
            "<FocusOut>", self._commit_scale_bar_microns_from_entry
        )
        self.microns_entry_field.bind(
            "<Return>", self._commit_scale_bar_microns_from_entry
        )

        self.update_scale_bar_state()

        show_deselected_switch = ctk.CTkSwitch(
            display_frame,
            text=constants.UI_TEXT_SHOW_DESELECTED_ONLY,
            variable=self.show_only_deselected_var,
        )
        show_deselected_switch.pack(anchor="w", pady=2)
        ToolTip(show_deselected_switch, constants.TOOLTIP_SHOW_DESELECTED_ONLY)

        self.update_channel_z_frame_visibility()

    def sync_ui_variables_with_model(self):
        super().sync_ui_variables_with_model()

        # Sync display adjustment sliders and colormap
        model_brightness = self.application_model.display_state.brightness
        slider_b_val = self.parent_frame._convert_model_to_slider_value(
            model_brightness
        )
        self.brightness_slider.set(slider_b_val)

        model_contrast = self.application_model.display_state.contrast
        slider_c_val = self.parent_frame._convert_model_to_slider_value(model_contrast)
        self.contrast_slider.set(slider_c_val)

        model_colormap = self.application_model.display_state.colormap_name or "None"
        self.colormap_variable.set(model_colormap)

        model_diameter = self.application_model.segmentation_diameter
        if model_diameter is not None:
            current_diameter_str = str(model_diameter)
            if self.diameter_var.get() != current_diameter_str:
                self.diameter_var.set(current_diameter_str)

        current_dia = self.diameter_var.get()
        if not (current_dia and current_dia.isdigit() and int(current_dia) > 0):
            model_dia = self.application_model.segmentation_diameter
            self.diameter_var.set(str(model_dia) if model_dia is not None else "30")

        self.update_channel_z_stack_controls()
        self.update_scale_bar_state()

    def update_history_buttons(self):
        can_undo, can_redo = False, False
        if self.drawing_controller.drawing_mode_active:
            can_undo = self.drawing_controller.can_undo_draw()
            can_redo = self.drawing_controller.can_redo_draw()
        else:
            can_undo = self.application_model.can_undo()
            can_redo = self.application_model.can_redo()

        self.undo_button.configure(state="normal" if can_undo else "disabled")
        self.redo_button.configure(state="normal" if can_redo else "disabled")

    def update_segmentation_in_progress(self, in_progress):
        if in_progress:
            self.segment_button.configure(state="disabled", text="Segmenting...")
            self.draw_mask_button.configure(state="disabled")
            self.select_image_btn.configure(state="disabled")
            self.upload_mask_button.configure(state="disabled")
            self.segment_button.pack_configure(pady=(0, 5))
            self.segment_progressbar.pack(fill="x", pady=(0, 0))
            self.segment_progressbar.start()
        else:
            self.segment_button.configure(
                state="normal"
                if self.application_model.image_data.original_image
                else "disabled",
                text=constants.UI_TEXT_SEGMENT_BUTTON,
            )
            self.draw_mask_button.configure(state="normal")
            self.select_image_btn.configure(state="normal")
            self.upload_mask_button.configure(state="normal")
            self.segment_progressbar.stop()
            self.segment_progressbar.pack_forget()
            self.segment_button.pack_configure(pady=(0, 0))

    def update_image_dependent_widgets(self, image_loaded):
        state = "normal" if image_loaded else "disabled"
        self.segment_button.configure(state=state)
        self.draw_mask_button.configure(state=state)
        self.upload_mask_button.configure(state=state)
        self.filename_label.configure(
            text=self.application_model.base_filename
            or constants.UI_TEXT_NO_FILE_SELECTED
        )

        pil_image = self.application_model.image_data.original_image
        if pil_image:
            self.resolution_label.configure(
                text=f"Resolution: {pil_image.width} x {pil_image.height} px"
            )
        else:
            self.resolution_label.configure(text=constants.UI_TEXT_RESOLUTION_NA)

        self.update_channel_z_frame_visibility()
        self.update_scale_bar_state()

    def update_resolution_label_zoom(self):
        pil_image = self.application_model.image_data.original_image
        zoom = self.application_model.pan_zoom_state.zoom_level
        if pil_image:
            res_text = (
                f"Resolution: {pil_image.width} x {pil_image.height} px @ {zoom:.2f}x"
            )
            self.resolution_label.configure(text=res_text)
        else:
            self.resolution_label.configure(text=constants.UI_TEXT_RESOLUTION_NA)

    def update_channel_z_stack_controls(self):
        model_display_state = self.application_model.display_state
        total_img_channels = model_display_state.total_channels
        total_z_slices = model_display_state.total_z_slices

        preset_colors = constants.CHANNEL_PRESET_COLORS
        preset_color_map = {rgb: name for name, rgb in preset_colors}
        color_dropdown_options = [name for name, rgb in preset_colors] + [
            constants.UI_TEXT_CUSTOM_COLOR
        ]

        for widget in self.dynamic_channel_config_frame.winfo_children():
            widget.destroy()

        for config_idx in range(total_img_channels):
            if config_idx >= len(model_display_state.display_channel_configs):
                break
            channel_config = model_display_state.display_channel_configs[config_idx]
            row_frame = ctk.CTkFrame(
                self.dynamic_channel_config_frame, fg_color="transparent"
            )
            row_frame.pack(fill="x", pady=1)

            ctk.CTkLabel(
                row_frame, text=f"Channel {config_idx}", width=80, anchor="w"
            ).pack(side="left", padx=(0, 5))

            color_rgb = tuple(channel_config.get("color_tuple_0_255", (128, 128, 128)))
            dropdown_value = preset_color_map.get(
                color_rgb, constants.UI_TEXT_CUSTOM_COLOR
            )
            color_var = ctk.StringVar(value=dropdown_value)

            def on_color_dropdown_change(choice, idx=config_idx):
                if choice == constants.UI_TEXT_CUSTOM_COLOR:
                    self._handle_dynamic_channel_color_change(idx)
                else:
                    rgb = dict(preset_colors)[choice]
                    self.application_model.set_view_option(
                        "display_channel_config_update",
                        {"config_index": idx, "color_tuple_0_255": rgb},
                    )
                self.update_channel_z_stack_controls()

            color_dropdown = ctk.CTkOptionMenu(
                row_frame,
                variable=color_var,
                values=color_dropdown_options,
                command=lambda choice, idx=config_idx: on_color_dropdown_change(
                    choice, idx
                ),
                width=130,
            )
            color_dropdown.pack(side="left", padx=(0, 5))
            ToolTip(color_dropdown, constants.TOOLTIP_CHANNEL_COLOR_DROPDOWN)

            active_var = ctk.BooleanVar(
                value=channel_config.get("active_in_composite", False)
            )
            checkbox = ctk.CTkCheckBox(
                row_frame,
                text="",
                variable=active_var,
                command=lambda state=active_var,
                idx=config_idx: self.display_settings_controller.update_channel_config(
                    {
                        "config_index": idx,
                        "active_in_composite": state.get(),
                    }
                ),
                width=20,
            )
            checkbox.pack(side="left", padx=(0, 0))
            ToolTip(checkbox, constants.TOOLTIP_CHANNEL_ACTIVE_CHECKBOX)

            if color_var.get() == constants.UI_TEXT_CUSTOM_COLOR:
                hex_color = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"
                color_btn = ctk.CTkButton(
                    row_frame,
                    text=" ",
                    width=24,
                    fg_color=hex_color,
                    command=lambda idx=config_idx: self._handle_dynamic_channel_color_change(
                        idx
                    ),
                )
                color_btn.pack(side="left", padx=(5, 0))
                ToolTip(color_btn, constants.TOOLTIP_CHANNEL_COLOR_PICKER)

        self.z_processing_method_menu.configure(
            state="normal" if total_z_slices > 0 else "disabled"
        )

        is_multidim = total_img_channels > 1 or total_z_slices > 1
        self.reset_channel_z_button.configure(
            state="normal"
            if self.application_model.image_data.original_image and is_multidim
            else "disabled"
        )

        current_z = int(model_display_state.current_z_slice_index)
        total_z = int(total_z_slices)
        label_text = constants.UI_TEXT_Z_SLICE_NA

        if total_z > 0 and model_display_state.z_processing_method == "slice":
            self.z_slice_label.pack(anchor="w", pady=(5, 0))
            self.z_slice_slider.pack(fill="x", pady=(0, 10))
            slider_to = max(0, total_z - 1)
            self.z_slice_slider.configure(
                state="normal" if total_z > 1 else "disabled",
                from_=0,
                to=slider_to,
                number_of_steps=max(1, total_z - 1),
            )
            label_text = f"Z-Slice: {current_z} / {slider_to}"
        else:
            self.z_slice_label.pack_forget()
            self.z_slice_slider.pack_forget()
            self.z_slice_slider.configure(
                state="disabled", from_=0, to=0, number_of_steps=1
            )
            if total_z > 0:
                label_text = (
                    f"Z-Slice: (N/A for {model_display_state.z_processing_method})"
                )

        self.z_slice_label_var.set(label_text)

    def _handle_dynamic_channel_color_change(self, config_index: int):
        initial_color = "#%02x%02x%02x" % tuple(
            self.application_model.display_state.display_channel_configs[config_index][
                "color_tuple_0_255"
            ]
        )
        color_code, _ = tk_colorchooser.askcolor(
            initialcolor=initial_color, title="Select Channel Color"
        )
        if color_code:
            rgb_tuple = tuple(int(round(c)) for c in color_code)
            self.application_model.set_view_option(
                "display_channel_config_update",
                {"config_index": config_index, "color_tuple_0_255": rgb_tuple},
            )
        self.update_channel_z_stack_controls()

    def update_channel_z_frame_visibility(self):
        total_channels = self.application_model.display_state.total_channels
        total_z_slices = self.application_model.display_state.total_z_slices
        should_show = (total_z_slices > 1) or (total_channels > 3)

        if should_show:
            if not self.adjust_channel_z_frame.winfo_ismapped():
                pack_before_widget = (
                    self.adjust_frame or self.mask_creation_folding_frame
                )
                self.adjust_channel_z_frame.pack(
                    padx=10, pady=(0, 2), fill="x", before=pack_before_widget
                )
        else:
            if self.adjust_channel_z_frame.winfo_ismapped():
                self.adjust_channel_z_frame.pack_forget()

    def _on_diameter_change(self, *args):
        def do_live_update():
            val = self.diameter_var.get()
            try:
                diameter = int(val)
                if diameter >= 0:
                    if self.application_model.segmentation_diameter != diameter:
                        self.display_settings_controller.set_segmentation_diameter(
                            diameter
                        )
            except ValueError:
                pass  # Allow invalid values (like empty string) during typing.

        self.after(1, do_live_update)

    def _on_microns_change(self, *args):
        def do_live_update():
            val = self.scale_bar_microns_var.get()
            try:
                microns = int(val)
                if microns >= 0:
                    if (
                        self.application_model.display_state.scale_bar_microns
                        != microns
                    ):
                        self.display_settings_controller.set_scale_bar_microns(microns)
            except ValueError:
                pass  # Allow invalid values (like empty string) during typing.

        self.after(1, do_live_update)

    def _commit_segmentation_diameter_from_entry(self, event=None):
        def do_final_commit():
            val = self.diameter_var.get()
            try:
                diameter = int(val)
                if diameter >= 0:
                    return  # Value is valid, do nothing.
            except (ValueError, TypeError):
                # Value is not a valid int, fall through to reset
                pass

            # Reset if input is invalid (empty, non-integer, or negative).
            model_val = self.application_model.segmentation_diameter
            new_val = "30"  # A safe default for diameter.
            if model_val is not None:
                new_val = str(model_val)

            if self.diameter_var.get() != new_val:
                self.diameter_var.set(new_val)

        self.after(1, do_final_commit)

    def _commit_scale_bar_microns_from_entry(self, event=None):
        def do_final_commit():
            val = self.scale_bar_microns_var.get()
            try:
                microns = int(val)
                if microns >= 0:
                    return  # Value is valid, do nothing.
            except (ValueError, TypeError):
                # Value is not a valid int, fall through to reset
                pass

            # Reset if input is invalid (empty, non-integer, or negative).
            model_val = self.application_model.display_state.scale_bar_microns
            new_val = ""
            if model_val is not None:
                new_val = str(int(model_val))

            if self.scale_bar_microns_var.get() != new_val:
                self.scale_bar_microns_var.set(new_val)

        self.after(1, do_final_commit)

    def update_scale_bar_state(self):
        scale = self.application_model.image_data.scale_conversion
        scale_missing = not (scale and hasattr(scale, "X") and scale.X and scale.X != 0)

        if scale_missing:
            self.show_ruler_cb.configure(state="disabled")
            self.microns_entry_field.configure(state="disabled")
            self.show_ruler_cb.unbind("<Button-1>")
            self.microns_entry_field.unbind("<FocusIn>")
            self.show_ruler_cb.bind("<Button-1>", self._show_scale_info_error_handler)
            self.microns_entry_field.bind(
                "<FocusIn>", self._show_scale_info_error_entry_handler
            )
            self.show_ruler_var.set(False)
        else:
            self.show_ruler_cb.configure(state="normal")
            self.microns_entry_field.configure(state="normal")
            self.show_ruler_cb.unbind("<Button-1>")
            self.microns_entry_field.unbind("<FocusIn>")
            self.show_ruler_var.set(constants.SHOW_RULER_DEFAULT)

    def _show_scale_info_error(self, *_):
        CTkMessagebox(
            title="Scale Bar Unavailable",
            message=constants.MSG_SCALE_BAR_UNAVAILABLE,
            icon="warning",
        )
