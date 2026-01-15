"""
FUNCTIONALITY:
- Load .lif images (notebook-friendly picker) and select the aggregate channel
- Apply channel renaming/mapping and fetch the aggregate ZYX volume
- Build a cell mask via max projection + Otsu + morphology
- Segment aggregates with percentile thresholding inside the cell mask and size filters
- Visualize aggregates in 2D (projection overlay) and 3D scatter plots
- Run the end-to-end aggregate pipeline and expose helper functions
"""

import numpy as np
import matplotlib.pyplot as plt
from aicsimageio import AICSImage
from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
)
from skimage.measure import label, regionprops
from notebook_file_picker import ensure_local_file



def select_and_load_lif():
    """
    Selecting and loading a LIF file.

    Input:
    - User picks a .lif file via the file dialog

    Functions:
    - Opens a file dialog to choose a preferred file
    - Saves the chosen LIF file and prints its dimensions

    Output:
    - Loaded image object (AICSImage)

    Additional information:
    - Dimensions:
      T: Time (one frame per picture)
      C: Channels
      Z: Layers
      Y: Height
      X: Width
    """
    lif_path = ensure_local_file(prompt_for_file=True, accept=".lif", description="Upload LIF")
    img = AICSImage(lif_path)
    print(f"Loaded LIF: {lif_path}")
    print(f"Image shape (T, C, Z, Y, X): {img.shape}")
    return img


def select_aggregate_channel(img, aggregate_name="aggregate", prompt_for_names=True, channel_map=None):
    """
    Select the aggregate channel from a LIF image.

    Input:
    - img: loaded AICSImage
    - aggregate_name: channel name to match (case-insensitive)
    - prompt_for_names: whether to prompt for channel renaming
    - channel_map: optional mapping original -> new channel names

    Functions:
    - Apply the provided channel map or prompt to rename channels
    - List available channels for visibility
    - Return the index of the first channel matching aggregate_name

    Output:
    - Index of the selected aggregate channel, or None if missing

    Additional information:
    - Logs a message and returns None when no match is found
    """
    original_channels = [str(ch) for ch in img.channel_names]
    print("\nOriginal channels:")
    for i, ch in enumerate(original_channels):
        print(f"[{i}] {ch}")
    if channel_map:
        new_channels = [channel_map.get(ch, ch) for ch in original_channels]
    else:
        print("\nAvailable channels:")
        for i, ch in enumerate(original_channels):
            print(f"[{i}] {ch}")
        if prompt_for_names:
            print("\nRename channels (press Enter to keep the original name):")
            new_channels = []
            for i, ch in enumerate(original_channels):
                new_name = input(f"New name for channel {i} (currently '{ch}'): ").strip()
                if new_name == "":
                    new_name = ch
                new_channels.append(new_name)
        else:
            new_channels = original_channels

    print("\nFinal channel names:")
    for i, ch in enumerate(new_channels):
        print(f"[{i}] {ch}")
    matches = [i for i, ch in enumerate(new_channels) if ch.lower() == aggregate_name.lower()]
    if not matches:
        print(f"No channel named '{aggregate_name}'.")
        return None

    idx = matches[0]
    print(f"Selected aggregate channel: {idx} ('{new_channels[idx]}')\n")
    return idx


def load_aggregate_volume(img, aggregate_index):
    """
    Return the ZYX volume for the chosen aggregate channel.

    Input:
    - img: loaded AICSImage
    - aggregate_index: channel index to extract

    Functions:
    - Read the 3D image data in ZYX order for the selected channel
    - Print the volume shape

    Output:
    - 3D numpy array (Z, Y, X) for the aggregate channel

    Additional information:
    - Uses the first timepoint (T=0) by default
    """
    volume = img.get_image_data("ZYX", C=aggregate_index, T=0)
    print(f"Aggregate volume loaded (Z,Y,X): {volume.shape}")
    return volume


def create_cell_mask(volume, min_cell_size=20000):
    """
    Make a 3D cell mask by thresholding a max projection.

    Input:
    - volume: aggregate intensity volume (Z, Y, X)
    - min_cell_size: minimum area to keep in the 2D mask

    Functions:
    - Compute a max projection and threshold with Otsu
    - Remove small objects and fill small holes
    - Repeat the cleaned 2D mask across all Z slices

    Output:
    - Boolean 3D cell mask aligned to the input volume

    Additional information:
    - Uses a fixed hole-fill area threshold of 5000 pixels
    """
    print("\n--- Creating cellmask ---")
    proj = volume.max(axis=0)
    thresh = threshold_otsu(proj)
    mask2D = proj > thresh
    mask2D = remove_small_objects(mask2D, min_size=min_cell_size)
    mask2D = remove_small_holes(mask2D, area_threshold=5000)
    cell_mask = np.repeat(mask2D[np.newaxis, :, :], volume.shape[0], axis=0)
    print("Cellmask created")
    return cell_mask


def segment_aggregates_intensity(volume, cell_mask, perc=99.9, min_size=100, max_size=50000):
    """
    Intensity-based aggregate segmentation inside the cell mask.

    Input:
    - volume: aggregate intensity volume (Z, Y, X)
    - cell_mask: boolean mask limiting the search area
    - perc: percentile for high-intensity thresholding
    - min_size: minimum voxel count per aggregate
    - max_size: maximum voxel count allowed

    Functions:
    - Extract intensities inside the cell mask and compute a percentile threshold
    - Threshold the volume, remove small components, and drop oversized ones
    - Return the cleaned aggregate mask and the threshold used

    Output:
    - Boolean mask of segmented aggregates
    - Threshold value (float) used for segmentation

    Additional information:
    - Returns an empty mask and None when the cell mask contains no voxels
    """
    print("\n--- Intensity-based aggregate segmentation ---")
    cell_values = volume[cell_mask > 0]
    print(f"Voxels inside cell mask: {cell_values.size}")
    if cell_values.size == 0:
        return np.zeros_like(volume, dtype=bool), None

    thr = np.percentile(cell_values, perc)
    print(f"Threshold (percentile {perc}): {thr}")
    high = (volume >= thr) & (cell_mask > 0)

    high = remove_small_objects(high, min_size=min_size, connectivity=3)

    labeled = label(high)
    cleaned = np.zeros_like(high, dtype=bool)
    for prop in regionprops(labeled):
        if prop.area <= max_size:
            cleaned[labeled == prop.label] = True

    print("Intensity segmentation complete")
    return cleaned, thr


def show_projection_with_mask(volume, mask, title="Projectie met aggregaten"):
    """
    Show a max projection with the aggregate mask overlay.

    Input:
    - volume: aggregate intensity volume (Z, Y, X)
    - mask: boolean aggregate mask
    - title: optional plot title

    Functions:
    - Compute a max projection of the volume
    - Scatter mask voxels on top of the projection

    Output:
    - Displayed matplotlib figure

    Additional information:
    - Uses red scatter points at (X, Y) positions of the mask
    """
    proj = volume.max(axis=0)
    plt.figure(figsize=(9, 9))
    plt.imshow(proj, cmap="gray")
    plt.title(title)
    plt.xlabel("X-pixels")
    plt.ylabel("Y-pixels")
    coords = np.column_stack(np.where(mask))
    plt.scatter(coords[:, 2], coords[:, 1], s=12, c="red")
    plt.show()


def plot_aggregates_3d(mask, z_scale=25, point_size=8, color='lime'):
    """
    Plot aggregates in 3D as scatter points.

    Input:
    - mask: boolean aggregate mask (Z, Y, X)
    - z_scale: scaling factor applied to Z for display
    - point_size: scatter point size
    - color: scatter point color

    Functions:
    - Label aggregates and extract voxel coordinates
    - Scale Z coordinates for visual compression/expansion
    - Scatter each aggregate's voxels in 3D

    Output:
    - Displayed matplotlib 3D scatter plot

    Additional information:
    - Uses regionprops to gather component coordinates
    """
    labeled = label(mask)
    props = regionprops(labeled)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for prop in props:
        coords = prop.coords  # (Z,Y,X)
        X = coords[:, 2]
        Y = coords[:, 1]
        Z = coords[:, 0] * z_scale
        ax.scatter(X, Y, Z, s=point_size, color=color)
    Zmax = (mask.shape[0] - 1) * z_scale
    Xmax = mask.shape[2] - 1
    Ymax = mask.shape[1] - 1
    ax.set_xlim(0, Xmax)
    ax.set_ylim(0, Ymax)
    ax.set_zlim(0, Zmax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (scaled)")
    plt.title("3D aggregaten")
    plt.show()


def run_aggregaten_pipeline(
    lif_path=None,
    prompt_for_file=True,
    prompt_for_channel_names=True,
    channel_map=None,
    aggregate_name="aggregate",
    min_cell_size=20000,
    perc=99.9,
    min_size=100,
    max_size=50000,
    show_plots=False,
):
    """
    Run the aggregate segmentation pipeline end-to-end.

    Input:
    - lif_path: optional path to a .lif file
    - prompt_for_file: whether to open a file picker
    - prompt_for_channel_names: whether to prompt for channel renaming
    - channel_map: optional channel name mapping
    - aggregate_name: channel name to match
    - min_cell_size: minimum 2D area for the cell mask
    - perc: percentile used for thresholding aggregates
    - min_size: minimum aggregate size (voxels)
    - max_size: maximum aggregate size (voxels)
    - show_plots: whether to display 2D/3D visualizations

    Functions:
    - Load the image (prompt or path)
    - Select the aggregate channel and load its volume
    - Build a cell mask and run intensity-based segmentation
    - Optionally show projection and 3D plots

    Output:
    - Dictionary with volume, cell_mask, aggregates_mask, and threshold

    Additional information:
    - Returns None early when file selection or channel selection fails
    """
    if prompt_for_file or not lif_path:
        img = select_and_load_lif()
    else:
        img = AICSImage(lif_path)
        print(f"Loaded LIF: {lif_path}")
        print(f"Image shape (T, C, Z, Y, X): {img.shape}")
    if img is None:
        return None

    agg_idx = select_aggregate_channel(
        img,
        aggregate_name=aggregate_name,
        prompt_for_names=prompt_for_channel_names,
        channel_map=channel_map,
    )
    if agg_idx is None:
        return None

    volume = load_aggregate_volume(img, agg_idx)
    cell_mask = create_cell_mask(volume, min_cell_size=min_cell_size)
    filtered_input = volume * cell_mask
    aggregates_mask, thr = segment_aggregates_intensity(
        filtered_input,
        cell_mask,
        perc=perc,
        min_size=min_size,
        max_size=max_size,
    )

    if show_plots:
        show_projection_with_mask(volume, aggregates_mask)
        plot_aggregates_3d(aggregates_mask, z_scale=30)

    return {
        "volume": volume,
        "cell_mask": cell_mask,
        "aggregates_mask": aggregates_mask,
        "threshold": thr,
    }


__all__ = [
    "run_aggregaten_pipeline",
    "select_and_load_lif",
    "select_aggregate_channel",
    "load_aggregate_volume",
    "create_cell_mask",
    "segment_aggregates_intensity",
    "show_projection_with_mask",
    "plot_aggregates_3d",
]
