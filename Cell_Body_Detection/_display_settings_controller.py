import matplotlib

from . import constants
from .application_model import ApplicationModel


class DisplaySettingsController:
    def __init__(self, application_model_ref: ApplicationModel, cell_body_frame_ref):
        """
        Initializes the DisplaySettingsController.

        Args:
            application_model_ref: A reference to the ApplicationModel instance.
            cell_body_frame_ref: A reference to the cell_body_frame instance (might still be needed for UI updates not covered by model notifications, or can be removed if not).
        """
        self.application_model = application_model_ref
        self.cell_body_frame = cell_body_frame_ref

    def set_brightness(self, value: float):
        """Sets the brightness level in the ApplicationModel."""
        print(f"DisplaySettingsController: Setting brightness to {value}")
        self.application_model.set_brightness(float(value))
        # The model will notify observers (like ImageViewRenderer) to update the display.

    def set_contrast(self, value: float):
        """Sets the contrast level in the ApplicationModel."""
        print(f"DisplaySettingsController: Setting contrast to {value}")
        self.application_model.set_contrast(float(value))
        # Model notifies observers.

    def set_colormap(self, colormap_name: str | None):
        """Sets the colormap in the ApplicationModel."""
        print(f"DisplaySettingsController: Setting colormap to {colormap_name}")
        self.application_model.set_colormap(colormap_name)
        # Model notifies observers.

    def reset_to_original(self):
        """Resets display settings to their default values in the ApplicationModel."""
        print("DisplaySettingsController: Resetting display adjustments to original.")
        self.application_model.reset_display_adjustments()
        # Model notifies observers.

    def get_available_colormaps(self) -> list[str]:
        """Returns a list of available colormap names from matplotlib."""
        colormaps = list(matplotlib.colormaps())
        colormap_options = constants.UI_IMAGE_COLORMAP_OPTIONS
        # Ensure "None" is always an option and that the list is sorted and unique.
        # Combine, deduplicate, and sort, then add "None" at the beginning.
        available_set = set(colormaps) & set(colormap_options)
        sorted_list = sorted(list(available_set))
        return ["None"] + sorted_list
