import customtkinter as ctk

from ..model.application_model import ApplicationModel
from ..utils.debug_logger import log


class BasePanel(ctk.CTkScrollableFrame):
    def __init__(self, parent, application_model: ApplicationModel, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent_frame = parent
        self.application_model = application_model
        self.traced_ui_variables_map = []

    def _setup_traced_view_option_var(
        self,
        model_attribute_path: tuple,
        ctk_var_type,
        option_key_name: str,
        initial_value_override=None,
    ):
        log(
            f"BasePanel._setup_traced_view_option_var: model_path={model_attribute_path}, key='{option_key_name}'",
            level="TRACE",
        )

        if initial_value_override is not None:
            initial_model_value = initial_value_override
        else:
            current_object = self.application_model
            for attr in model_attribute_path:
                current_object = getattr(current_object, attr)
            initial_model_value = current_object

        variable = ctk_var_type(value=initial_model_value)

        def trace_callback(*args, var=variable, key=option_key_name):
            value = bool(var.get()) if isinstance(var, ctk.BooleanVar) else var.get()
            self.application_model.set_view_option(key, value)

        variable.trace_add("write", trace_callback)
        return variable

    def sync_ui_variables_with_model(self):
        """Synchronizes the panel's UI variables with the application model state."""
        log(
            f"{self.__class__.__name__}.sync_ui_variables_with_model: Synchronizing variables."
        )
        for ui_var, model_attr_path in self.traced_ui_variables_map:
            current_model_val_obj = self.application_model
            for attr_name in model_attr_path:
                current_model_val_obj = getattr(current_model_val_obj, attr_name)

            value_to_set = (
                bool(current_model_val_obj)
                if isinstance(ui_var, ctk.BooleanVar)
                else current_model_val_obj
            )
            if ui_var.get() != value_to_set:
                ui_var.set(value_to_set)
