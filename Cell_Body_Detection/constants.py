# Cell_Body_Detection/constants.py

# --- Fonts ---
DEFAULT_FONT = "arial.ttf"
DEFAULT_FONT_BOLD = "arialbd.ttf"

# --- UI Defaults ---
DEFAULT_MODEL_DIAMETER = "100"
DEFAULT_BOUNDARY_COLOR = "Green"
AVAILABLE_BOUNDARY_COLORS = ["Green", "Red", "Blue", "Yellow"]

# --- Drawing Feedback Colors ---
DRAW_FEEDBACK_COLOR = "magenta"
DRAW_FIRST_POINT_COLOR = "cyan"
DRAW_LAST_POINT_COLOR = "orange"
DRAW_POINT_RADIUS = 3  # pixels on canvas

# --- PDF Export ---
PDF_TARGET_DPI = 300.0
PDF_DEFAULT_FILENAME_PREFIX = "_report"
PDF_SELECTED_CELLS_FILENAME_PREFIX = "_selected_cells"

# --- Zoom & Pan ---
MAX_ZOOM_LEVEL = 5.0
INTERACTIVE_RENDER_DELAY_MS = 250  # For pan/zoom final render
PAN_STEP_PIXELS = 50

# --- Alpha Values ---
MASK_BLEND_ALPHA = 0.4

# --- Drawing Feedback ---
DRAW_FEEDBACK_LINE_WIDTH = 2

# --- PDF Layout ---
PDF_MARGIN = 50
PDF_LINE_HEIGHT = 18
PDF_STATS_TOP_OFFSET = 30
PDF_TITLE_IMAGE_GAP_FACTOR = 1.5  # Multiplier for line_height
PDF_IMAGE_GAP_ABOVE_FACTOR = 0.5  # Multiplier for line_height

# --- Messages ---
MSG_NO_IMAGE_LOADED = "No image loaded."
MSG_SELECT_IMAGE_PROMPT = "Select an Image"
MSG_ERROR_LOADING_IMAGE_TITLE = "Error Loading Image"
MSG_SEGMENTATION_ERROR_TITLE = "Segmentation Error"
MSG_NO_IMAGE_FOR_SEGMENTATION = "No image loaded for segmentation."
MSG_DRAWING_ERROR_TITLE = "Drawing Error"
MSG_DRAWING_LOAD_IMAGE_FIRST = "Please load an image first."
MSG_DRAWING_NEED_MORE_POINTS = "Need at least 3 points to create a polygon."
MSG_EXPORT_ERROR_TITLE = "Error"
MSG_EXPORT_NO_MASK = "No mask to export."
MSG_EXPORT_SUCCESS_TITLE = "Success"
MSG_EXPORT_COMPLETED = "Export completed."
MSG_EXPORT_FAILED = "Export failed: {error}"  # Placeholder for error string
MSG_PDF_EXPORT_ERROR_TITLE = "PDF Export Error"
MSG_PDF_EXPORT_FAILED = "PDF creation failed: {error}"
MSG_PDF_ORIGINAL_IMAGE_MISSING = "Original image missing for PDF report."
MSG_PDF_EXPORTED_SUCCESS = "PDF exported."
MSG_NO_IMAGE_FOR_EXPORT = "No image loaded to export."
# MSG_EXPORT_PREFIX_VIEW_TIF = "Current view TIF" # This is a UI message string
MSG_EXPORT_FAILED_PREFIX_VIEW_TIF = "Failed to export current view TIF:"

# --- UI Text ---
UI_TEXT_SETTINGS_TITLE = "Settings"
UI_TEXT_NAVIGATION = "Navigation"
UI_TEXT_RETURN_TO_START = "Return to Start Screen"
UI_TEXT_HISTORY = "History"
UI_TEXT_UNDO = "Undo"
UI_TEXT_REDO = "Redo"
UI_TEXT_IMPORT_SETTINGS = "Import Settings"
UI_TEXT_SELECT_IMAGE = "Select Image"
UI_TEXT_NO_FILE_SELECTED = "No file selected"
UI_TEXT_MODEL_SETTINGS = "Model Settings"
UI_TEXT_DIAMETER_LABEL = "Diameter:"
UI_TEXT_SEGMENT_BUTTON = "Segment"
UI_TEXT_START_DRAWING_BUTTON = "Start Drawing"
UI_TEXT_FINALIZE_POLYGON_BUTTON = "Finalize Polygon"
UI_TEXT_DISPLAY_OPTIONS = "Display Options"
UI_TEXT_SHOW_ORIGINAL_IMAGE = "Show Original Image"
UI_TEXT_SHOW_CELL_MASKS = "Show Cell Masks"
UI_TEXT_SHOW_CELL_BOUNDARIES = "Show Cell Boundaries"
UI_TEXT_SHOW_CELL_NUMBERS = "Show Cell Numbers"
UI_TEXT_DISPLAY_MODE_LABEL = "Display Mode:"
UI_TEXT_SHOW_DESELECTED_ONLY = "Show Deselected Masks"  # For switch
UI_TEXT_BOUNDARY_COLOR_LABEL = "Boundary Color:"
UI_TEXT_OUTPUT_PANEL_TITLE = "Output"
UI_TEXT_STATS_LABEL_DEFAULT = ""  # Needs formatting in use
UI_TEXT_PDF_EXPORT_OPTIONS_LABEL = "PDF Export Options (select multiple):"
UI_TEXT_PDF_MASKS_ONLY = "Masks Only"
UI_TEXT_PDF_BOUNDARIES_ONLY = "Boundaries Only"
UI_TEXT_PDF_NUMBERS_ONLY = "Numbers Only"
UI_TEXT_PDF_MASKS_BOUNDARIES = "Masks & Boundaries"
UI_TEXT_PDF_MASKS_NUMBERS = "Masks & Numbers"
UI_TEXT_PDF_BOUNDARIES_NUMBERS = "Boundaries & Numbers"
UI_TEXT_PDF_MASKS_BOUNDARIES_NUMBERS = "Masks, Boundaries & Numbers"
UI_TEXT_PDF_INCLUDE_ORIGINAL = "Include Original Image"
UI_TEXT_EXPORT_SELECTED_CELLS_BUTTON = "Export Selected Cells as Mask File"
UI_TEXT_EXPORT_PDF_REPORT_BUTTON = "Export PDF Report"
UI_TEXT_EXPORT_VIEW_AS_TIF_BUTTON = "Export Current View as Image"

# --- Colors (general, if not covered by specific UI elements) ---
COLOR_BLACK_RGB = (0, 0, 0)
COLOR_WHITE_RGB = (255, 255, 255)  # Example

BOUNDARY_COLOR_MAP_PIL = {
    "Green": (0, 255, 0),
    "Red": (255, 0, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
}

# Versions for text on PDF numbers, for better contrast
PDF_TEXT_NUMBER_COLOR_MAP = {
    "Green": (0, 255, 0),
    "Red": (255, 0, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
}

# --- Filetypes ---
MICROSCOPY_IMG_FILETYPES = [
    ("Microscopy Image", "*.tif *.tiff *.lif *.png *.jpg *.jpeg")
]
EXPORT_FILETYPES_TIF_NUMPY = [("TIFF", "*.tif"), ("PNG", "*.png"), ("NumPy array", "*.npy")]
EXPORT_FILETYPES_PDF = [("PDF files", "*.pdf")]

# --- Keyboard Shortcut Pan/Zoom ---
KEY_ZOOM_STEP = 1.1

# --- Misc ---
CELL_NUMBERING_FONT_SIZE_ORIG_IMG = (
    15  # Base font size for cell numbers on original image scale
)
CELL_NUMBERING_FONT_SIZE_CANVAS_MIN = 10
CELL_NUMBERING_FONT_SIZE_CANVAS_MAX = 30
CELL_NUMBERING_FONT_SIZE_PDF = 20  # For PDF export

CELL_CENTER_FIND_MARGIN = 10  # Margin for finding cell center for numbering/PDF
IMAGE_BOUNDARY_MARGIN_FOR_NUMBER_PLACEMENT = (
    10  # Margin from image edge for number placement
)

DILATION_ITERATIONS_FOR_BOUNDARY_DISPLAY = 1  # Default for renderer

# --- Fallback/Placeholder Values ---
DEFAULT_FALLBACK_FONT_SIZE = 12

# --- UI Specific Dimensions ---
UI_SIDEPANEL_WIDTH = 300
UI_INITIAL_CANVAS_WIDTH = 1024
UI_INITIAL_CANVAS_HEIGHT = 1024
UI_FILENAME_LABEL_WRAPLENGTH = 250

# --- PDF Font Settings ---
PDF_FONT_TITLE = "Helvetica-Bold"
PDF_FONT_TITLE_SIZE = 16
PDF_FONT_BODY = "Helvetica"
PDF_FONT_BODY_SIZE = 10
PDF_FONT_SUBHEADER_SIZE = 12  # For image titles etc.
PDF_IMAGE_EXPORT_FORMAT = "PNG"

# --- Miscellaneous ---
RANDOM_SEED_MASKS = 42
UI_RENDER_RETRY_DELAY_MS = 50
COLOR_BLACK_STR = "black"  # For PIL Image.new
MASK_DTYPE_NAME = "int32"  # To be used with np.dtype()

# --- UI Text for Display Adjustments ---
UI_TEXT_DISPLAY_ADJUSTMENTS = "Image Adjustments"
UI_TEXT_BRIGHTNESS = "Brightness:"
UI_TEXT_CONTRAST = "Contrast:"
UI_TEXT_COLORMAP = "Colormap:"
UI_TEXT_RESET_DISPLAY_SETTINGS = "Reset Image Adjustments"

# --- PDF Export Drawing & Layout ---
PDF_CELL_IMAGE_PADDING = 20  # Pixels around cell for individual cell images

# --- File Export Prefixes ---
EXPORT_PREFIX_VIEW_TIF = "_current_view"

# --- Colormap Options ---
UI_IMAGE_COLORMAP_OPTIONS = [
    "gray",       # Standard for raw microscopy images
    "bone",       # Softer grayscale, good for medical imaging
    "cividis",    # Colorblind-friendly, perceptually uniform
    "viridis",    # High contrast, perceptually uniform
    "plasma",     # More vibrant alternative to viridis
    "magma",      # Darker tones, good for depth/intensity
    "inferno",    # High contrast, useful for bright features
    "twilight",   # Good for low-light microscopy
]