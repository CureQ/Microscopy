# =====================
# Fonts
# =====================
DEFAULT_FONT = "arial.ttf"

# =====================
# Colors
# =====================
COLOR_BLACK_STR = "black"  # For PIL Image.new

BOUNDARY_COLOR_MAP_PIL = {
    "Green": (0, 255, 0),
    "Red": (255, 0, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Magenta": (255, 0, 255),
    "Cyan": (0, 255, 255),
    "Gray": (255, 255, 255),
}

CHANNEL_PRESET_COLORS = [
    ("Red         ", (255, 0, 0)),
    ("Green       ", (0, 255, 0)),
    ("Blue        ", (0, 0, 255)),
    ("Yellow      ", (255, 255, 0)),
    ("Magenta     ", (255, 0, 255)),
    ("Cyan        ", (0, 255, 255)),
    ("Gray        ", (255, 255, 255)),
]

# =====================
# UI Defaults & States
# =====================
AVAILABLE_BOUNDARY_COLORS = [
    "Red",
    "Green",
    "Blue",
    "Yellow",
    "Magenta",
    "Cyan",
    "Gray",
]
Z_PROCESSING_METHOD_OPTIONS = ["slice", "max_project", "mean_project"]

# UI Default States (centralized for all toggles, checkboxes, and options)
SHOW_ORIGINAL_IMAGE_DEFAULT = True  # Show original image overlay
SHOW_CELL_MASKS_DEFAULT = False  # Show cell masks overlay
SHOW_CELL_BOUNDARIES_DEFAULT = True  # Show cell boundaries/contours
BOUNDARY_COLOR_DEFAULT = "Green"  # Default boundary color
SHOW_DESELECTED_MASKS_ONLY_DEFAULT = False  # Show only deselected masks
SHOW_CELL_NUMBERS_DEFAULT = True  # Show cell numbers/IDs
SHOW_RULER_DEFAULT = True  # Show scale bar (already present)
SHOW_DIAMETER_AID_DEFAULT = False  # Show diameter aid circle on canvas

# PDF Export Option Defaults
PDF_OPT_MASKS_ONLY_DEFAULT = True
PDF_OPT_BOUNDARIES_ONLY_DEFAULT = True
PDF_OPT_NUMBERS_ONLY_DEFAULT = True
PDF_OPT_MASKS_BOUNDARIES_DEFAULT = True
PDF_OPT_MASKS_NUMBERS_DEFAULT = True
PDF_OPT_BOUNDARIES_NUMBERS_DEFAULT = True
PDF_OPT_MASKS_BOUNDARIES_NUMBERS_DEFAULT = True
PDF_INCLUDE_SCALE_BAR_DEFAULT = True  # Already present

# Z-Stack and Channel Defaults
Z_PROCESSING_METHOD_DEFAULT = "max_project"
CURRENT_Z_SLICE_INDEX_DEFAULT = 0

# Segmentation and Display Adjustments
SEGMENTATION_DIAMETER_DEFAULT = "100"
APPLY_IMAGE_ADJUSTMENTS_BEFORE_SEGMENTATION = True
BRIGHTNESS_DEFAULT = 1.0
CONTRAST_DEFAULT = 1.0
COLORMAP_DEFAULT = None  # or "None"
SCALE_BAR_MICRONS_DEFAULT = 30

# UI Text Labels
UI_TEXT_SETTINGS_TITLE = "Settings"
UI_TEXT_NAVIGATION = "Navigation"
UI_TEXT_RETURN_TO_START = "Return to Start Screen"
UI_TEXT_HISTORY = "History"
UI_TEXT_UNDO = "Undo"
UI_TEXT_REDO = "Redo"
UI_TEXT_IMPORT_SETTINGS = "Import File"
UI_TEXT_SELECT_IMAGE = "Select Image"
UI_TEXT_NO_FILE_SELECTED = "No file selected"
UI_TEXT_MODEL_SETTINGS = "Model Settings"
UI_TEXT_DIAMETER_LABEL = "Diameter:"
UI_TEXT_SEGMENT_BUTTON = "Segment (Model)"
UI_TEXT_START_DRAWING_BUTTON = "Start Drawing (Manual)"
UI_TEXT_FINALIZE_POLYGON_BUTTON = "Finalize Mask"
UI_TEXT_DISPLAY_OPTIONS = "Overlay Options"
UI_TEXT_SHOW_ORIGINAL_IMAGE = "Show Original Image"
UI_TEXT_SHOW_CELL_MASKS = "Show Cell Masks"
UI_TEXT_SHOW_CELL_BOUNDARIES = "Show Cell Outlines"
UI_TEXT_SHOW_CELL_NUMBERS = "Show Cell IDs"
UI_TEXT_DISPLAY_MODE_LABEL = "Display Mode:"
UI_TEXT_SHOW_RULER = "Show Scale Bar"
UI_TEXT_SHOW_DESELECTED_ONLY = "Show Deselected Masks"
UI_TEXT_BOUNDARY_COLOR_LABEL = "Cell Outline Color:"
UI_TEXT_OUTPUT_PANEL_TITLE = "Output"
UI_TEXT_STATS_LABEL_DEFAULT = ""  # Needs formatting in use
UI_TEXT_PDF_EXPORT_OPTIONS_LABEL = "PDF Export Options (select multiple):"
UI_TEXT_PDF_MASKS_ONLY = "Masks Only"
UI_TEXT_PDF_BOUNDARIES_ONLY = "Outlines Only"
UI_TEXT_PDF_NUMBERS_ONLY = "IDs Only"
UI_TEXT_PDF_MASKS_BOUNDARIES = "Masks & Outlines"
UI_TEXT_PDF_MASKS_NUMBERS = "Masks & IDs"
UI_TEXT_PDF_BOUNDARIES_NUMBERS = "Outlines & IDs"
UI_TEXT_PDF_MASKS_BOUNDARIES_NUMBERS = "Masks, Outlines & IDs"
UI_TEXT_PDF_INCLUDE_ORIGINAL = "Include Original Image"
UI_TEXT_EXPORT_SELECTED_CELLS_BUTTON = "Export Selected Cells as Segmentation Mask"
UI_TEXT_EXPORT_PDF_REPORT_BUTTON = "Export PDF Report"
UI_TEXT_EXPORT_VIEW_AS_TIF_BUTTON = "Export View as TIF"
UI_TEXT_UPLOAD_MASK_BUTTON = "Upload (From File)"
UI_TEXT_BRIGHTNESS = "Brightness:"
UI_TEXT_CONTRAST = "Contrast:"
UI_TEXT_COLORMAP = "Colormap:"
UI_TEXT_RESET_DISPLAY_SETTINGS = "Reset Image Adjustments"
UI_TEXT_COLLAPSED_PREFIX = "► "
UI_TEXT_EXPANDED_PREFIX = "▼ "
UI_TEXT_MODEL_SETTINGS_COLLAPSED = "   ► Model Settings"
UI_TEXT_MODEL_SETTINGS_EXPANDED = "   ▼ Model Settings"
UI_TEXT_CHANNEL_Z_STACK_CONTROLS = "Channel & Z-Stack Controls"
UI_TEXT_MASK_CREATION = "Mask Creation"
UI_TEXT_DISPLAY_ADJUSTMENTS = "Image Adjustments"
UI_TEXT_RESET_CHANNEL_Z_SETTINGS = "Reset Channel/Z Settings"
UI_TEXT_MICRONS_LABEL = "µm"
UI_TEXT_EXPORTING = "Exporting..."
UI_TEXT_RESOLUTION_NA = "Resolution: N/A"
UI_TEXT_Z_SLICE_NA = "Z-Slice: N/A"
UI_TEXT_PDF_INCLUDE_SCALE_BAR = "Include scale bar in all PDF images"
UI_TEXT_CHANNEL_MAPPING = "Channel Mapping:"
UI_TEXT_Z_STACK_PROCESSING = "Z-Stack Processing:"
UI_TEXT_CUSTOM_COLOR = "Custom color"
UI_TEXT_SHOW_DIAMETER_AID = "Show Diameter Aid"

# UI Specific Dimensions
UI_SIDEPANEL_WIDTH = 300
UI_INITIAL_CANVAS_WIDTH = 1024
UI_INITIAL_CANVAS_HEIGHT = 1024
UI_FILENAME_LABEL_WRAPLENGTH = 250

# =====================
# Drawing & Feedback
# =====================
DRAW_FEEDBACK_COLOR = "magenta"
DRAW_FIRST_POINT_COLOR = "cyan"
DRAW_LAST_POINT_COLOR = "orange"
DRAW_POINT_RADIUS = 3  # pixels on canvas
DRAW_FEEDBACK_LINE_WIDTH = 2


# =====================
# Scale Bar
# =====================
SCALE_BAR_COLOR = "white"
SCALE_BAR_HEIGHT = 5.0


# =====================
# Diameter Aid
# =====================
DIAMETER_AID_COLOR = "red"
DIAMETER_AID_WIDTH = 2


# =====================
# PDF Export
# =====================
PDF_TARGET_DPI = 300.0
PDF_DEFAULT_FILENAME_PREFIX = "_report"
PDF_SELECTED_CELLS_FILENAME_PREFIX = "_segmentation_mask"
PDF_MARGIN = 50
PDF_LINE_HEIGHT = 18
PDF_STATS_TOP_OFFSET = 30
PDF_TITLE_IMAGE_GAP_FACTOR = 1.5  # Multiplier for line_height
PDF_IMAGE_GAP_ABOVE_FACTOR = 0.5  # Multiplier for line_height
PDF_FONT_TITLE = "Helvetica-Bold"
PDF_FONT_TITLE_SIZE = 16
PDF_FONT_BODY = "Helvetica"
PDF_FONT_BODY_SIZE = 10
PDF_FONT_SUBHEADER_SIZE = 12  # For image titles etc.
PDF_IMAGE_EXPORT_FORMAT = "PNG"

# =====================
# File Types & Export
# =====================
MICROSCOPY_IMG_FILETYPES = [
    ("Microscopy Image", "*.tif *.tiff *.lif *.png *.jpg *.jpeg")
]
EXPORT_FILETYPES_TIF_NUMPY = [
    ("TIFF", "*.tif"),
    ("PNG", "*.png"),
    ("NumPy array", "*.npy"),
]
EXPORT_FILETYPES_PDF = [("PDF files", "*.pdf")]
EXPORT_PREFIX_VIEW_TIF = "_current_view"

# =====================
# Image Processing Defaults
# =====================
MAX_ZOOM_LEVEL = 3.0
INTERACTIVE_RENDER_DELAY_MS = 250  # For pan/zoom final render
PAN_STEP_PIXELS = 50
MASK_BLEND_ALPHA = 0.4
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

# Colormap Options
UI_IMAGE_COLORMAP_OPTIONS = [
    "gray",  # Standard for raw microscopy images
    "bone",  # Softer grayscale, good for medical imaging
    "cividis",  # Colorblind-friendly, perceptually uniform
    "viridis",  # High contrast, perceptually uniform
    "plasma",  # More vibrant alternative to viridis
    "magma",  # Darker tones, good for depth/intensity
    "inferno",  # High contrast, useful for bright features
    "twilight",  # Good for low-light microscopy
]

# =====================
# Miscellaneous
# =====================
RANDOM_SEED_MASKS = 42
UI_RENDER_RETRY_DELAY_MS = 50
MASK_DTYPE_NAME = "int32"  # To be used with np.dtype()
LIF_LASX_FIX = True  # If True, use special loading for LASX-created LIF files (see _file_io_controller.py)
KEY_ZOOM_STEP = 1.1
MOUSE_WHEEL_ZOOM_SENSITIVITY = 0.02

# =====================
# Messages
# =====================
UI_TEXT_LOAD_ERROR = "Load Error"
MSG_LOAD_ERROR_TITLE = "Load Error"
UI_TEXT_NO_IMAGE_LOADED = "No image loaded."
MSG_SELECT_IMAGE_PROMPT = "Select an Image"
MSG_SEGMENTATION_ERROR_TITLE = "Segmentation Error"
MSG_NO_IMAGE_FOR_SEGMENTATION = "No image loaded for segmentation."
MSG_DRAWING_ERROR_TITLE = "Drawing Error"
MSG_DRAWING_LOAD_IMAGE_FIRST = "Please load an image first."
MSG_DRAWING_NEED_MORE_POINTS = "Need at least 3 points to create a polygon."
MSG_EXPORT_ERROR_TITLE = "Export Error"
MSG_EXPORT_NO_MASK = "No segmentation mask found to export."
MSG_EXPORT_SUCCESS_TITLE = "Export Successful"
MSG_EXPORT_FAILED_TITLE = "Export Failed"
MSG_EXPORT_COMPLETED = "Export completed successfully."
MSG_PDF_EXPORT_ERROR_TITLE = "PDF Export Error"
MSG_PDF_ORIGINAL_IMAGE_MISSING = "Original image is missing, cannot generate full PDF."
MSG_PDF_EXPORTED_SUCCESS = "PDF report exported successfully."
MSG_PDF_NO_MASK_FOR_OVERLAY_OPTIONS = "Cannot generate selected PDF overlays (masks, outlines, IDs) because no segmentation data is available. Please run segmentation or draw masks first."
MSG_NO_IMAGE_FOR_EXPORT = "No image loaded to export from."
MSG_INVALID_MASK_NPY_FILE = "The selected .npy file does not contain a valid mask array. This file was most likely not created by this application. Please upload a numeric mask array exported from this app or a compatible tool."
MSG_NO_MASKS_CREATED = "Segmentation completed, but no masks were created. Please check your image and parameters."
MSG_EXPORT_SELECTED_SUCCESS_DEFAULT = "Selected cells exported successfully."
MSG_EXPORT_TIF_SUCCESS_DEFAULT = "View exported as TIF successfully."
MSG_EXPORT_PDF_SUCCESS_DEFAULT = "PDF report exported successfully."
MSG_SCALE_BAR_UNAVAILABLE = "Scale bar options are unavailable because this image does not contain scale information. Please load a LIF file with scale metadata to use these features."

# =====================
# Tooltip Descriptions for UI Elements
# =====================
UI_TOOLTIP_BACKGROUND_COLOR = "#ffffff"
UI_TOOLTIP_FOREGROUND_COLOR = "#000000"
UI_TOOLTIP_FONT_NAME = "arial"
UI_TOOLTIP_FONT_SIZE = 12

TOOLTIP_SELECT_IMAGE = "Select a microscopy image file to load into the application."
TOOLTIP_UNDO = "Undo the last action."
TOOLTIP_REDO = "Redo the last undone action."
TOOLTIP_SEGMENT_BUTTON = (
    "Run automatic cell segmentation using the current diameter settings."
)
TOOLTIP_DRAW_MASK_BUTTON = "Manually draw a cell mask on the image."
TOOLTIP_UPLOAD_MASK_BUTTON = "Upload a mask file from your computer."
TOOLTIP_BRIGHTNESS_SLIDER = "Adjust the image brightness."
TOOLTIP_CONTRAST_SLIDER = "Adjust the image contrast."
TOOLTIP_COLORMAP_MENU = "Change the color mapping of the image display."
TOOLTIP_DIAMETER_ENTRY = "Set the expected cell diameter for segmentation."
TOOLTIP_EXPORT_SELECTED = "Export a mask containing only the selected cells."
TOOLTIP_EXPORT_TIF = "Export the current view as a TIF image."
TOOLTIP_EXPORT_PDF = "Export a PDF report of the current analysis."
TOOLTIP_SHOW_ORIGINAL = "Toggle display of the original image."
TOOLTIP_SHOW_MASKS = "Toggle display of cell masks."
TOOLTIP_SHOW_BOUNDARIES = "Toggle display of cell outlines."
TOOLTIP_BOUNDARY_COLOR = "Select the color for cell outlines."
TOOLTIP_SHOW_CELL_NUMBERS = "Toggle display of cell IDs."
TOOLTIP_SHOW_RULER = "Show or hide the scale bar."
TOOLTIP_SCALE_BAR_MICRONS = "Set the length of the scale bar in microns."
TOOLTIP_PDF_INCLUDE_SCALE_BAR = "Include a scale bar in all exported PDF images."
TOOLTIP_PDF_EXPORT_OPTIONS = "Select which overlays to include in the PDF export."
TOOLTIP_CHANNEL_MAPPING = "Assign image channels to display colors."
TOOLTIP_Z_STACK_PROCESSING = "Choose how to process Z-stack images."
TOOLTIP_Z_SLICE_SLIDER = "Select the Z-slice to display."
TOOLTIP_RESET_CHANNEL_Z = "Reset channel and Z-stack settings to default."
TOOLTIP_SHOW_DESELECTED_ONLY = "Show only the deselected cell masks."
TOOLTIP_EXPORT_SELECTED_PROGRESS = "Progress bar for exporting selected cells."
TOOLTIP_EXPORT_TIF_PROGRESS = "Progress bar for exporting TIF image."
TOOLTIP_EXPORT_PDF_PROGRESS = "Progress bar for exporting PDF report."
TOOLTIP_PDF_MASKS_ONLY = "Export only the cell masks in the PDF."
TOOLTIP_PDF_BOUNDARIES_ONLY = "Export only the cell outlines in the PDF."
TOOLTIP_PDF_NUMBERS_ONLY = "Export only the cell IDs in the PDF."
TOOLTIP_PDF_MASKS_BOUNDARIES = "Export both cell masks and outlines in the PDF."
TOOLTIP_PDF_MASKS_NUMBERS = "Export both cell masks and IDs in the PDF."
TOOLTIP_PDF_BOUNDARIES_NUMBERS = "Export both cell outlines and IDs in the PDF."
TOOLTIP_PDF_MASKS_BOUNDARIES_NUMBERS = (
    "Export cell masks, outlines, and IDs in the PDF."
)
TOOLTIP_RESET_DISPLAY_SETTINGS = (
    "Reset brightness, contrast, and colormap to default values."
)
TOOLTIP_RESET_CHANNEL_Z_SETTINGS = (
    "Reset channel and Z-stack controls to their default configuration."
)
TOOLTIP_CHANNEL_COLOR_PICKER = "Pick a custom color for this channel."
TOOLTIP_CHANNEL_ACTIVE_CHECKBOX = (
    "Toggle whether this channel is active in the composite image."
)
TOOLTIP_CHANNEL_COLOR_DROPDOWN = "Select a preset or custom color for this channel."
TOOLTIP_Z_PROCESSING_METHOD_MENU = "Select the method for Z-stack processing."
TOOLTIP_Z_SLICE_LABEL = "Shows the current Z-slice index."
TOOLTIP_HISTORY_LABEL = "Shows the history of actions for undo/redo."
TOOLTIP_STATS_LABEL = "Shows statistics about the current segmentation and selection."
TOOLTIP_RESOLUTION_LABEL = "Shows the resolution and zoom level of the loaded image."
TOOLTIP_SHOW_DIAMETER_AID = (
    "Show a circle representing the current segmentation diameter on the image."
)
