from ..utils.debug_logger import log


class PanZoomModel:
    """
    Holds the state for panning and zooming.
    """

    def __init__(self):
        self.zoom_level: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self.min_zoom_to_fit: float = 1.0
        log("PanZoomModel: Initialized.", level="DEBUG")

    def _calculate_min_zoom_to_fit(
        self, canvas_width: int, canvas_height: int, img_width: int, img_height: int
    ) -> float:
        if img_width == 0 or img_height == 0 or canvas_width == 0 or canvas_height == 0:
            return 1.0
        zoom_if_fit_width = canvas_width / img_width
        zoom_if_fit_height = canvas_height / img_height
        return min(zoom_if_fit_width, zoom_if_fit_height, 1.0)

    def reset_for_new_image(
        self,
        canvas_width: int = 0,
        canvas_height: int = 0,
        img_width: int = 0,
        img_height: int = 0,
    ):
        if img_width > 0 and img_height > 0 and canvas_width > 1 and canvas_height > 1:
            self.min_zoom_to_fit = self._calculate_min_zoom_to_fit(
                canvas_width, canvas_height, img_width, img_height
            )
            self.zoom_level = self.min_zoom_to_fit
            zoomed_img_width = img_width * self.zoom_level
            zoomed_img_height = img_height * self.zoom_level
            self.pan_x = (canvas_width - zoomed_img_width) / 2.0
            self.pan_y = (canvas_height - zoomed_img_height) / 2.0
            log(
                f"PanZoomModel: Reset for new image. Canvas: {canvas_width}x{canvas_height}, Img: {img_width}x{img_height}. Min zoom: {self.min_zoom_to_fit:.4f}, Zoom: {self.zoom_level:.4f}, Pan: ({self.pan_x:.2f}, {self.pan_y:.2f})",
                level="INFO",
            )
        else:
            self.min_zoom_to_fit = 1.0
            self.zoom_level = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0
            log(
                f"PanZoomModel: Reset to default (no image/canvas). Min zoom: {self.min_zoom_to_fit}, Zoom: {self.zoom_level}, Pan: ({self.pan_x}, {self.pan_y})",
                level="INFO",
            )

    def get_params(self) -> tuple[float, float, float]:
        log(
            f"PanZoomModel: get_params called. Returning zoom={self.zoom_level:.4f}, pan=({self.pan_x:.2f},{self.pan_y:.2f})",
            level="DEBUG",
        )
        return self.zoom_level, self.pan_x, self.pan_y

    def set_zoom_level(
        self, zoom: float, min_zoom: float | None = None, max_zoom: float | None = None
    ):
        new_zoom = zoom
        if min_zoom is not None:
            new_zoom = max(min_zoom, new_zoom)
        if max_zoom is not None:
            new_zoom = min(max_zoom, new_zoom)
        self.zoom_level = new_zoom
        log(
            f"PanZoomModel: Zoom level set to {self.zoom_level:.4f} (requested: {zoom:.4f}, min_in: {min_zoom}, max_in: {max_zoom})",
            level="DEBUG",
        )

    def set_pan(self, pan_x: float, pan_y: float):
        self.pan_x = pan_x
        self.pan_y = pan_y
        log(
            f"PanZoomModel: Pan set to ({self.pan_x:.2f}, {self.pan_y:.2f})",
            level="DEBUG",
        )
