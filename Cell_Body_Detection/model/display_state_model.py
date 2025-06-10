from .. import constants
from ..utils.debug_logger import log


class DisplayStateModel:
    """
    Holds the state for image display adjustments and view options.
    """

    DEFAULT_DISPLAY_CHANNEL_CONFIGS = [
        {
            "name": "Red",
            "color_tuple_0_255": (255, 0, 0),
            "source_idx": None,
            "active_in_composite": True,
        },
        {
            "name": "Green",
            "color_tuple_0_255": (0, 255, 0),
            "source_idx": None,
            "active_in_composite": True,
        },
        {
            "name": "Blue",
            "color_tuple_0_255": (0, 0, 255),
            "source_idx": None,
            "active_in_composite": True,
        },
        {
            "name": "Yellow",
            "color_tuple_0_255": (255, 255, 0),
            "source_idx": None,
            "active_in_composite": False,
        },
        {
            "name": "Magenta",
            "color_tuple_0_255": (255, 0, 255),
            "source_idx": None,
            "active_in_composite": False,
        },
        {
            "name": "Cyan",
            "color_tuple_0_255": (0, 255, 255),
            "source_idx": None,
            "active_in_composite": False,
        },
        {
            "name": "Gray",
            "color_tuple_0_255": (255, 255, 255),
            "source_idx": None,
            "active_in_composite": False,
        },
    ]

    def __init__(self):
        # Image adjustments
        self.brightness: float = constants.BRIGHTNESS_DEFAULT
        self.contrast: float = constants.CONTRAST_DEFAULT
        self.colormap_name: str | None = constants.COLORMAP_DEFAULT
        self.scale_bar_microns: int = constants.SCALE_BAR_MICRONS_DEFAULT

        # View options
        self.show_original_image: bool = constants.SHOW_ORIGINAL_IMAGE_DEFAULT
        self.show_cell_masks: bool = constants.SHOW_CELL_MASKS_DEFAULT
        self.show_cell_boundaries: bool = constants.SHOW_CELL_BOUNDARIES_DEFAULT
        self.boundary_color_name: str = constants.BOUNDARY_COLOR_DEFAULT
        self.show_deselected_masks_only: bool = (
            constants.SHOW_DESELECTED_MASKS_ONLY_DEFAULT
        )
        self.show_cell_numbers: bool = constants.SHOW_CELL_NUMBERS_DEFAULT
        self.show_ruler: bool = constants.SHOW_RULER_DEFAULT
        self.show_diameter_aid: bool = constants.SHOW_DIAMETER_AID_DEFAULT

        # PDF Export Options
        self.pdf_opt_masks_only: bool = constants.PDF_OPT_MASKS_ONLY_DEFAULT
        self.pdf_opt_boundaries_only: bool = constants.PDF_OPT_BOUNDARIES_ONLY_DEFAULT
        self.pdf_opt_numbers_only: bool = constants.PDF_OPT_NUMBERS_ONLY_DEFAULT
        self.pdf_opt_masks_boundaries: bool = constants.PDF_OPT_MASKS_BOUNDARIES_DEFAULT
        self.pdf_opt_masks_numbers: bool = constants.PDF_OPT_MASKS_NUMBERS_DEFAULT
        self.pdf_opt_boundaries_numbers: bool = (
            constants.PDF_OPT_BOUNDARIES_NUMBERS_DEFAULT
        )
        self.pdf_opt_masks_boundaries_numbers: bool = (
            constants.PDF_OPT_MASKS_BOUNDARIES_NUMBERS_DEFAULT
        )
        self.pdf_include_scale_bar: bool = constants.PDF_INCLUDE_SCALE_BAR_DEFAULT

        # Z-Stack Processing
        self.z_processing_method: str = constants.Z_PROCESSING_METHOD_DEFAULT
        self.current_z_slice_index: int = constants.CURRENT_Z_SLICE_INDEX_DEFAULT
        self.display_channel_configs: list[dict] = [
            config.copy() for config in self.DEFAULT_DISPLAY_CHANNEL_CONFIGS
        ]
        log("DisplayStateModel: Initialized.", level="DEBUG")

    def reset_image_adjustments(self):
        self.brightness = constants.BRIGHTNESS_DEFAULT
        self.contrast = constants.CONTRAST_DEFAULT
        self.colormap_name = constants.COLORMAP_DEFAULT
        self.scale_bar_microns = constants.SCALE_BAR_MICRONS_DEFAULT

    def reset_pdf_options(self):
        log("DisplayStateModel: Resetting PDF options.", level="INFO")
        self.pdf_opt_masks_only = constants.PDF_OPT_MASKS_ONLY_DEFAULT
        self.pdf_opt_boundaries_only = constants.PDF_OPT_BOUNDARIES_ONLY_DEFAULT
        self.pdf_opt_numbers_only = constants.PDF_OPT_NUMBERS_ONLY_DEFAULT
        self.pdf_opt_masks_boundaries = constants.PDF_OPT_MASKS_BOUNDARIES_DEFAULT
        self.pdf_opt_masks_numbers = constants.PDF_OPT_MASKS_NUMBERS_DEFAULT
        self.pdf_opt_boundaries_numbers = constants.PDF_OPT_BOUNDARIES_NUMBERS_DEFAULT
        self.pdf_opt_masks_boundaries_numbers = (
            constants.PDF_OPT_MASKS_BOUNDARIES_NUMBERS_DEFAULT
        )
        self.pdf_include_scale_bar = constants.PDF_INCLUDE_SCALE_BAR_DEFAULT

    def reset_channel_z_defaults(self):
        log("DisplayStateModel: Resetting channel and Z defaults.", level="INFO")
        self.z_processing_method = constants.Z_PROCESSING_METHOD_DEFAULT
        self.current_z_slice_index = constants.CURRENT_Z_SLICE_INDEX_DEFAULT

        # Reset display_channel_configs to a fresh copy of defaults, ensuring all source_idx are None initially
        self.display_channel_configs = []
        for default_config in self.DEFAULT_DISPLAY_CHANNEL_CONFIGS:
            new_config = default_config.copy()
            new_config["source_idx"] = None  # Explicitly None
            # active_in_composite will be taken from default_config initially,
            # then potentially overridden if an image is loaded and channels are mapped.
            self.display_channel_configs.append(new_config)

        # If image dimensions are known (i.e., an image is loaded),
        # apply image-specific defaults for channel mapping and Z-slice.
        if hasattr(self, "total_channels") and self.total_channels > 0:
            log(
                f"DisplayStateModel: Applying image-specific defaults: total_C={self.total_channels}, total_Z={getattr(self, 'total_z_slices', 0)}",
                level="DEBUG",
            )
            num_img_channels = self.total_channels
            num_display_slots = len(self.display_channel_configs)

            for i in range(num_display_slots):
                config = self.display_channel_configs[i]
                if i < num_img_channels:
                    config["source_idx"] = i
                    config["active_in_composite"] = True
                else:
                    config["source_idx"] = None
                    config["active_in_composite"] = False

            # Set Z-processing method based on image Z-slices
            if hasattr(self, "total_z_slices") and self.total_z_slices > 1:
                self.z_processing_method = constants.Z_PROCESSING_METHOD_DEFAULT
            else:
                self.z_processing_method = "slice"
        else:
            log(
                "DisplayStateModel: No image-specific dimensions (total_channels) found. Using absolute defaults.",
                level="DEBUG",
            )
            # Ensure all are inactive if no image context
            for config in self.display_channel_configs:
                config["source_idx"] = None
                config["active_in_composite"] = (
                    False  # If truly no image, nothing should be active by default
                )
            self.current_z_slice_index = constants.CURRENT_Z_SLICE_INDEX_DEFAULT
