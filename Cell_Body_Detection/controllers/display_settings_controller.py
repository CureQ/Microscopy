import matplotlib

from .. import constants
from ..model.application_model import ApplicationModel
from ..utils.debug_logger import log


class DisplaySettingsController:
    def __init__(self, application_model_ref: ApplicationModel):
        """
        Initializes the DisplaySettingsController.

        Args:
            application_model_ref: A reference to the ApplicationModel instance.
        """
        self.application_model = application_model_ref

    def _convert_slider_to_model_value(self, slider_value: float) -> float:
        """Converts a slider value (1-100) to a model value (e.g., 0-2)."""
        min_val = 0.0
        max_val = 2.0
        return ((float(slider_value) - 1) / 99) * (max_val - min_val) + min_val

    def set_brightness(self, value: str):
        """Sets the brightness level in the ApplicationModel."""
        model_value = self._convert_slider_to_model_value(float(value))
        log(
            f"DisplaySettingsController: Setting brightness to {model_value} (slider: {value})"
        )
        self.application_model.set_brightness(model_value)

    def set_contrast(self, value: str):
        """Sets the contrast level in the ApplicationModel."""
        model_value = self._convert_slider_to_model_value(float(value))
        log(
            f"DisplaySettingsController: Setting contrast to {model_value} (slider: {value})"
        )
        self.application_model.set_contrast(model_value)

    def set_colormap(self, colormap_name: str | None):
        """Sets the colormap in the ApplicationModel."""
        log(f"DisplaySettingsController: Setting colormap to {colormap_name}")
        self.application_model.set_colormap(colormap_name)

    def reset_display_settings(self):
        """Resets display settings to their default values in the ApplicationModel."""
        log("DisplaySettingsController: Resetting display adjustments to original.")
        self.application_model.reset_display_adjustments()

    def set_channel_z_setting(self, key: str, value):
        """Sets a specific channel or Z-stack setting in the model."""
        log(f"DisplaySettingsController: Setting {key} to {value}")
        self.application_model.set_view_option(key, value)

    def update_channel_config(self, payload: dict):
        """Updates a specific display channel's configuration."""
        log(
            f"DisplaySettingsController: Updating channel config with payload: {payload}"
        )
        self.application_model.set_view_option("display_channel_config_update", payload)

    def reset_channel_z_settings(self):
        """Resets all channel and Z-stack configurations to their defaults."""
        log("DisplaySettingsController: Resetting channel and Z-stack settings.")
        self.application_model.reset_channel_z_configurations()

    def set_segmentation_diameter(self, diameter: int):
        """Sets the segmentation diameter in the ApplicationModel."""
        log(f"DisplaySettingsController: Setting diameter to {diameter}")
        self.application_model.set_view_option("segmentation_diameter", diameter)

    def set_scale_bar_microns(self, microns: int):
        """Sets the scale bar length in microns in the ApplicationModel."""
        log(f"DisplaySettingsController: Setting scale bar microns to {microns}")
        self.application_model.set_scale_bar_microns(microns)

    def get_available_colormaps(self) -> list[str]:
        """Returns a list of available colormap names from matplotlib."""
        colormaps = list(matplotlib.colormaps())
        colormap_options = constants.UI_IMAGE_COLORMAP_OPTIONS
        available_set = set(colormaps) & set(colormap_options)
        sorted_list = sorted(list(available_set))
        return ["None"] + sorted_list
