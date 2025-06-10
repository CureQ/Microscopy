# Cell Body Detection - User Guide

This guide will help you use the Cell Body Detection application to analyze microscopy images and identify cell bodies in your samples.

## What This Application Does

The Cell Body Detection application helps you:
- **Load microscopy images** from various file formats (especially `.lif`, `.tif`, `.png`, `.jpg`)
- **Automatically detect cell bodies** using AI-powered segmentation
- **Manually draw or edit cell boundaries** when needed
- **Select which cells to include** in your analysis
- **Export results** as images, data files, or comprehensive PDF reports

## Getting Started

### Installation

1. **Download and install Python** (if not already installed) from [python.org](https://python.org)
2. **Open a terminal or command prompt**
3. **Install the application**:
   ```
   pip install -r requirements.txt
   ```
4. **Run the application**:
   ```
   python microscopy_analysis_tool.py
   ```

The application window will open, ready for use.

## Understanding the Interface

The application has three main areas:

### Left Panel (Settings)
- **File selection** and image loading
- **Segmentation controls** for automatic cell detection
- **Manual drawing tools** for custom cell boundaries
- **Display options** to customize what you see
- **Image adjustments** for brightness, contrast, and color

### Center Panel (Image Viewer)
- **Main image display** where you view your microscopy data
- **Interactive zoom and pan** - use mouse wheel to zoom, click and drag to pan
- **Cell selection** - click on cells to include/exclude them from analysis

### Right Panel (Output)
- **Statistics** about detected cells
- **Export options** for saving your results

## Step-by-Step Workflow

### 1. Load Your Image

1. Click **"Select Image"** in the left panel
2. Browse to your microscopy image file
3. Select the file and click "Open"
4. Wait for the image to load and appear in the center viewer

**Supported formats**: `.lif` (Leica), `.tif/.tiff`, `.png`, `.jpg/.jpeg`

### 2. Adjust Image Display (Optional)

If your image is too dark, too bright, or needs color adjustment:

1. **Brightness**: Use the brightness slider to make the image lighter or darker
2. **Contrast**: Use the contrast slider to enhance detail
3. **Colormap**: Choose a different color scheme from the dropdown
4. **Reset**: Click "Reset Image Adjustments" to return to original settings

For multi-channel images (like `.lif` files):
- **Channel Controls**: Toggle individual channels on/off
- **Z-Stack Processing**: Choose how to handle 3D image stacks
  - *Slice*: View one layer at a time
  - *Max Project*: Combine all layers showing brightest pixels
  - *Mean Project*: Combine all layers using average intensity

### 3. Detect Cell Bodies

#### Automatic Detection (Recommended)

1. **Set the diameter**: Enter the approximate cell diameter in pixels
   - If unsure, start with 50-100 pixels
   - You can check the "Show Diameter Aid" option to see a circle of this size on your image
2. Click **"Segment (Model)"**
3. Wait for the AI to process your image (this may take a minute)
4. Cell boundaries will appear on your image

#### Upload Previous Results (From File)

If you have previously saved mask files from this or other compatible applications:

1. Click **"Upload (From File)"**
2. Browse to your mask file
3. Select a compatible file format:
   - `.tif` or `.tiff`: Mask images saved from previous sessions
   - `.png`: Mask images in PNG format
   - `.npy`: NumPy array files (technical format from this application)
4. Click "Open"
5. The mask will be loaded and cell boundaries will appear

**Important**: The uploaded mask must match the dimensions of your current image exactly.

#### Manual Drawing (For Custom Areas)

1. Click **"Start Drawing (Manual)"**
2. Click points around the edge of a cell to create a polygon
3. Press **Enter** or click **"Finalize Mask"** when done
4. Press **Escape** to cancel drawing

### 4. Review and Select Cells

- **Green boundaries**: Currently selected cells (included in analysis)
- **Red boundaries**: Deselected cells (excluded from analysis)
- **Click on any cell** to toggle its selection
- **Cell numbers**: Small numbers show cell IDs (can be toggled on/off)

### 5. Customize Display Options

In the "Overlay Options" section:

- **Show Original Image**: Toggle the background image on/off
- **Show Cell Masks**: Show filled cell areas in color
- **Show Cell Outlines**: Show cell boundary lines
- **Show Cell IDs**: Display numbers for each cell
- **Show Scale Bar**: Display a measurement bar (if image has scale information)
- **Cell Outline Color**: Change the color of cell boundaries

### 6. Export Your Results

Choose from several export options:

#### Export Selected Cells as Segmentation Mask
- Creates a data file containing only the cells you've selected
- Useful for further analysis in other software
- **Save these files to reload later** using the "Upload (From File)" feature
- Formats: `.tif`, `.png`, or `.npy` (NumPy array)

#### Export View as TIF
- Saves exactly what you see on screen as an image file
- Includes all your display settings and overlays
- Good for presentations or publications

#### Export PDF Report
- Creates a comprehensive report with multiple views
- Select which overlays to include:
  - *Masks Only*: Just the filled cell areas
  - *Outlines Only*: Just the cell boundaries
  - *IDs Only*: Just the cell numbers
  - *Combined views*: Various combinations of the above
- Includes statistics and image information

## Tips for Best Results

### Image Quality
- **Use high-contrast images** for better automatic detection
- **Adjust brightness/contrast** before segmentation for optimal results
- **For multi-channel images**, select the channel that shows cell boundaries most clearly

### Segmentation Parameters
- **Start with default diameter** (100 pixels) and adjust based on results
- **Too small diameter**: Cells may be over-segmented (split into multiple pieces)
- **Too large diameter**: Multiple cells may be grouped together
- **Re-run segmentation** with different parameters if needed

### Manual Corrections
- **Use manual drawing** for cells the AI missed
- **Click to deselect** cells that were incorrectly detected
- **Draw custom boundaries** around irregular or touching cells
- **Upload previous masks** if you've already analyzed similar images

### Working with Large Images
- **Use zoom** to inspect cells closely
- **Pan around** the image to check all areas
- **The application remembers** your zoom and pan settings

### Working with Previous Results
- **Save your mask files** using "Export Selected Cells as Segmentation Mask"
- **Reload masks later** using "Upload (From File)" for the same or similar images
- **Share masks** with colleagues by sending the exported mask files
- **Use consistent file naming** to keep track of different analysis sessions

## Keyboard Shortcuts

- **Ctrl+Z** (Cmd+Z on Mac): Undo last action
- **Ctrl+Shift+Z** (Cmd+Shift+Z on Mac): Redo action
- **+/-**: Zoom in/out
- **Arrow keys**: Pan image
- **Escape**: Cancel current drawing
- **Enter**: Finalize current drawing

## Troubleshooting

### Common Issues

**"No masks were created" after segmentation**
- Try adjusting the diameter parameter
- Check if your image has sufficient contrast
- Consider adjusting brightness/contrast before segmentation

**Application runs slowly**
- Large images may take time to process
- Close other programs to free up memory
- Consider working with smaller image regions

**Can't see cell boundaries clearly**
- Adjust the cell outline color
- Toggle different overlay options
- Adjust image brightness and contrast

**Export fails**
- Make sure you have write permissions to the destination folder
- Check that you have enough disk space
- Try a different file format

**"Invalid mask file" error when uploading**
- Make sure the mask file dimensions match your current image
- Check that the file was created by this application or a compatible tool
- Verify the file isn't corrupted
- Try a different file format (.tif instead of .npy, for example)

### Getting Help

If you encounter issues:
1. Check that your image file isn't corrupted
2. Try restarting the application
3. Ensure your image is in a supported format
4. For technical support, include your image file format and a description of the problem

## File Format Notes

### .lif Files (Leica)
- **Best supported format** with automatic scale information
- **Multi-channel support** with individual channel controls
- **Z-stack support** for 3D image processing

### .tif/.tiff Files
- **Good general support** for microscopy images
- May include scale information depending on how they were saved

### .png/.jpg Files
- **Basic support** for standard image formats
- No scale information available
- Best for simple, single-channel images

## Best Practices

1. **Save your work regularly** by exporting results
2. **Keep original images** - the application doesn't modify source files
3. **Document your settings** - note diameter and other parameters used
4. **Use consistent parameters** within an experiment for comparable results
5. **Review results carefully** - always check that segmentation looks reasonable
6. **Export multiple formats** if you need data for different purposes

---