"""
FUNCTIONALITY:
- Load .lif images and choose the nucleus channel via dialog
- Segment nuclei, merge nearby detections, and compute convex hulls in physical space
- Export voxel coordinates to Excel/CSV and save/load pipeline outputs
- Voxelize hulls back to masks and generate visual QA plots
- Provide an interactive demo to run the whole flow manually
"""

import sys, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog
from aicsimageio import AICSImage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import ball, binary_dilation, binary_opening, remove_small_objects
from skimage.segmentation import relabel_sequential
from sklearn.neighbors import KDTree

#---------- LOADING FILE ----------------

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

    Tk().withdraw()
    lif_path = filedialog.askopenfilename(title='Select a lif-file', filetypes=[('Leica Image File', '*.lif')])
    if not lif_path:
        print('NO FILE SELECTED.')
        return None

    img = AICSImage(lif_path)
    img.file_path = lif_path
    print(f'Lif-file loaded: {lif_path}')
    print(f'Dimensions (T, C, Z, Y, X): {img.shape}')
    return img

#---------- RENAME CHANNELS AND SELECT NUCLEUS ----------------

def rename_and_select_nucleus(img):
    """
    Renaming channel names and selecting the nucleus channel.

    Input:
    - Loaded LIF image

    Functions:
    - Retrieve original channel names
    - Allow manual channel name overrides
    - Show updated channel names
    - Find the channel named "Nucleus"
    - Create a mapping from old to new names

    Output:
    - Nucleus channel name
    - Nucleus channel index
    - Channel name mapping
    """
    # Retrieve original channel names
    original_channels = [str(ch) for ch in img.channel_names]
    print("Original channel names:")
    for i, ch in enumerate(original_channels):
        print(f"  [{i}] {ch}")

    # Manually overwrite channel names
    print("\nEnter a new name for the channel (press Enter to keep the current name):\n")
    new_channels = []
    for i, ch in enumerate(original_channels):
        new_name = input(f"Name for channel {i} (currently: '{ch}'): ").strip()
        if not new_name:
            new_name = ch
        new_channels.append(new_name)

    # Show new channel names
    print("\nUpdated channel names:")
    for i, ch in enumerate(new_channels):
        print(f"  [{i}] {ch}")

    # Find channel name "Nucleus"
    match = [i for i, ch in enumerate(new_channels) if ch.lower() == "nucleus"]
    if match:
        nucleus_index = match[0]
        nucleus_name = new_channels[nucleus_index]
        print(f"\nChosen channel: '{nucleus_name}' (index {nucleus_index})")
    else:
        nucleus_index = None
        nucleus_name = None
        print("\nNo channel found with name 'Nucleus'.")

    # Create a mapping old -> new
    channel_map = dict(zip(original_channels, new_channels))

    return nucleus_name, nucleus_index, channel_map

#---------- NAMING VOLUME ----------------

def nucleus_volumes(img, selected_channel_name, selected_channel_index, intensity_threshold=None):
    """
    Extracting the nucleus volume from the selected channel.

    Input:
    - Loaded LIF image
    - Selected nucleus channel name
    - Selected nucleus channel index
    - Optional manual intensity threshold (currently unused)

    Functions:
    - Validate that a channel index is provided
    - Retrieve 3D volume data for the selected channel
    - Report the volume shape

    Output:
    - 3D numpy array (Z, Y, X) for the nucleus channel, or None if unavailable
    """
    if selected_channel_index is None:
        print("No valid channel selected. Cannot retrieve volume.")
        return None

    # Retrieve 3D data
    volume = img.get_image_data("ZYX", C=selected_channel_index, T=0)
    print(f"Volume shape: {volume.shape} (Z, Y, X)")

    return volume

#---------- SEGMENT AND MERGE NUCLEI ----------------

def segment_and_merge(volume, min_size=500, merge_distance=15, z_compression_factor=None):
    """
    Segmenting nuclei and merging nearby objects.

    Input:
    - 3D volume (Z, Y, X)
    - min_size: minimum voxel count to keep an object
    - merge_distance: maximum distance (voxels) to merge neighboring nuclei
    - z_compression_factor: optional compression factor for Z visualization

    Functions:
    - Ask for a Z compression factor when not provided
    - Threshold the volume with Otsu and clean the mask
    - Label objects and merge those within the merge distance
    - Relabel and visualize the merged result in 3D with compressed Z

    Output:
    - Labeled volume after merging
    """

    # Z-compression input
    if z_compression_factor is None:
        try:
            z_compression_factor = float(
                input("Enter desired Z compression (e.g. 0.3 = flatter, 1.0 = no compression): ")
            )
        except ValueError:
            z_compression_factor = 0.3
            print("Invalid input, default value 0.3 used.")

    print(f"\nZ compression set to x{z_compression_factor}")
    print("\nSegmentation started...")

    # Otsu threshold
    thresh = threshold_otsu(volume)
    print(f"Otsu threshold = {thresh:.2f}")

    mask = volume > thresh
    mask = binary_opening(mask, ball(1))
    mask = binary_dilation(mask, ball(1))
    mask = remove_small_objects(mask, min_size=min_size)

    labeled = label(mask)
    props = regionprops(labeled)
    print(f"Initial objects found: {len(props)}")

    # Bounding boxes
    boxes = []
    for p in props:
        z1, y1, x1, z2, y2, x2 = p.bbox
        boxes.append((p.label, z1, y1, x1, z2, y2, x2))

    merge_pairs = []

    # Overlap test
    for i in range(len(boxes)):
        lbl1, z1a, y1a, x1a, z1b, y1b, x1b = boxes[i]

        for j in range(i + 1, len(boxes)):
            lbl2, z2a, y2a, x2a, z2b, y2b, x2b = boxes[j]

            if (z1a > z2b + merge_distance or z2a > z1b + merge_distance or
                y1a > y2b + merge_distance or y2a > y1b + merge_distance or
                x1a > x2b + merge_distance or x2a > x1b + merge_distance):
                continue

            coords1 = props[lbl1 - 1].coords
            coords2 = props[lbl2 - 1].coords

            tree1 = KDTree(coords1)
            if tree1.query_radius(coords2, r=merge_distance, count_only=True).sum() > 0:
                merge_pairs.append((lbl1, lbl2))

    print(f"Merge pairs found: {len(merge_pairs)}")

    # Union-find
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        parent[find(a)] = find(b)

    for a, b in merge_pairs:
        union(a, b)

    groups = {}
    for p in props:
        root = find(p.label)
        groups.setdefault(root, set()).add(p.label)

    print(f"Unique merge groups: {len(groups)}")

    # Merge labels
    merged = labeled.copy()
    for root, lbls in groups.items():
        base = min(lbls)
        for lbl in lbls:
            merged[merged == lbl] = base

    final_labels, _, _ = relabel_sequential(merged)
    print(f"Final object count: {final_labels.max()}")

    # 3D plot
    plt.close("all")
    z, y, x = np.nonzero(final_labels)
    z_scaled = z * z_compression_factor

    # Unique colors
    cmap = plt.cm.get_cmap("tab20", final_labels.max())
    colors = cmap(final_labels[z, y, x] - 1)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        x, y, z_scaled,
        s=0.1,
        c=colors,
        alpha=0.85
    )

    # Adding label numbers
    final_props = regionprops(final_labels)
    for p in final_props:
        cz, cy, cx = p.centroid  # (z, y, x)
        ax.text(
            cx, cy, cz * z_compression_factor,
            str(p.label),
            color="black",
            fontsize=10,
            ha="center"
        )

    ax.set_box_aspect((1, 1, z_compression_factor))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel(f"Z (x{z_compression_factor:.2f})")
    ax.set_title("3D nuclei segmentation with merging (compressed Z)")
    plt.tight_layout()

    fig.canvas.draw()
    sys.stdout.flush()
    time.sleep(0.1)
    plt.show(block=False)

    return final_labels

#---------- RETREIVING VOXEL SIZE ----------------

def get_voxel_size_from_img(img, default=(1.0, 1.0, 1.0)):
    """
    Retrieving physical voxel sizes from the image metadata.

    Input:
    - Loaded LIF image
    - Default voxel size fallback in (z, y, x)

    Functions:
    - Read physical pixel sizes from metadata
    - Return absolute values to handle negative spacing
    - Fall back to the provided default when metadata is missing

    Output:
    - Tuple with voxel sizes (z, y, x)
    """
    pps = getattr(img, "physical_pixel_sizes", None)
    if pps is None or (pps.Z is None or pps.Y is None or pps.X is None):
        return default
    return (abs(float(pps.Z)), abs(float(pps.Y)), abs(float(pps.X)))

#---------- CREATING CONVEX HULL ----------------

def compute_convex_hulls_phys(labeled_volume, voxel_size_xyz, min_points=50):
    """
    Computing convex hulls in physical coordinates.

    Input:
    - Labeled volume (3D array)
    - Voxel sizes in (x, y, z)
    - Minimum voxel count required to build a hull

    Functions:
    - Convert labeled voxels to physical coordinates
    - Build a convex hull per label
    - Collect hull properties (volume, area, centroid)

    Output:
    - List of (label, hull) pairs
    - List of property dictionaries per label
    """
    vx, vy, vz = voxel_size_xyz  # using (x, y, z) order here
    hulls, props = [], []

    unique_labels = np.unique(labeled_volume)
    unique_labels = unique_labels[unique_labels > 0]

    for lbl in unique_labels:
        # coords z,y,x -> convert to physical (x,y,z)
        zyxs = np.column_stack(np.nonzero(labeled_volume == lbl))
        if zyxs.shape[0] < min_points:
            continue

        # convert to (x,y,z) and scale to physical units
        xs = zyxs[:, 2] * vx
        ys = zyxs[:, 1] * vy
        zs = zyxs[:, 0] * vz
        pts_xyz = np.column_stack((xs, ys, zs))  # (N, 3) in (x,y,z)

        try:
            hull = ConvexHull(pts_xyz)
            hulls.append((int(lbl), hull))
            props.append({
                "label": int(lbl),
                "n_points": zyxs.shape[0],
                "hull_volume_phys": hull.volume,
                "hull_area_phys": hull.area,
                "centroid_phys_xyz": pts_xyz.mean(axis=0)
            })
        except Exception as e:
            print(f"Hull failed for label {lbl}: {e}")

    print(f"{len(hulls)} hulls computed in physical (x, y, z) space.")
    return hulls, props

#---------- PLOTING CONVEX HULL WITH COMPRESSION ----------------

def plot_convex_hulls_phys(labeled_volume, hulls, voxel_size_xyz, z_compression_factor=0.1, alpha_points=0.08):
    """
    Plotting convex hulls in physical space with compressed Z.

    Input:
    - Labeled volume
    - List of convex hulls
    - Voxel sizes in (x, y, z)
    - Z compression factor for visualization
    - Alpha value for the voxel point cloud

    Functions:
    - Convert labeled voxels to physical coordinates
    - Compress Z for visualization
    - Scatter all voxels colored by label
    - Draw convex hull surfaces per label

    Output:
    - 3D matplotlib figure
    """
    vx, vy, vz = voxel_size_xyz

    # All voxel points (physical)
    z, y, x = np.nonzero(labeled_volume)
    X = x * vx
    Y = y * vy
    Z = z * vz

    # Z compression
    Zc = Z * z_compression_factor

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Point cloud (labels)
    ax.scatter(
        X, Y, Zc,
        s=0.05,
        c=labeled_volume[z, y, x],
        cmap="tab10",
        alpha=alpha_points
    )

    # Colors for hulls
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(hulls))))

    # Draw hulls
    for i, (lbl, hull) in enumerate(hulls):

        # Copy physical points and compress Z
        pts = hull.points.copy()
        pts[:, 2] *= z_compression_factor

        for tri_idx in hull.simplices:
            tri = Poly3DCollection([pts[tri_idx]], alpha=0.35, edgecolor="k")
            tri.set_facecolor(colors[i % len(colors)])
            ax.add_collection3d(tri)

    # Same aspect settings as segmentation plot
    ax.set_box_aspect((1, 1, z_compression_factor))
    ax.set_proj_type('ortho')
    ax.set_zlim(Zc.min(), Zc.max())

    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")
    ax.set_zlabel(f"Z (um x{z_compression_factor})")
    ax.set_title("3D convex hulls - physical space (compressed Z)")

    plt.tight_layout()
    plt.show()


def export_coordinates_to_excel(
    labeled_volume,
    source_path,
    voxel_size_xyz=None,
    output_dir="Excel_data_contour",
    suffix="_convex_coord",
    include_microns=True,
):
    """
    Save voxel coordinates to Excel (or CSV) per nucleus label.

    Input:
    - labeled_volume: 3D labeled array
    - source_path: path to the source image for naming
    - voxel_size_xyz: optional voxel spacing (x, y, z) for micron columns
    - output_dir: directory where the spreadsheet will be written
    - suffix: filename suffix to append
    - include_microns: whether to add x_um, y_um, z_um columns

    Functions:
    - Extract voxel coordinates and their labels
    - Convert to micrometers when spacing is provided
    - Write an Excel file; fall back to CSV if Excel writing fails

    Output:
    - Path to the saved Excel or CSV file

    Additional information:
    - Creates the output directory when it does not exist
    - Skips writing when no labeled voxels are present
    """
    z, y, x = np.nonzero(labeled_volume)
    if len(x) == 0:
        print("No labeled voxels found; skipping coordinate export.")
        return None

    labels = labeled_volume[z, y, x]
    data = {
        "x": x.astype(int),
        "y": y.astype(int),
        "z": z.astype(int),
        "nucleus_label": labels.astype(int),
    }

    if include_microns and voxel_size_xyz is not None:
        vx, vy, vz = voxel_size_xyz
        data["x_um"] = x * vx
        data["y_um"] = y * vy
        data["z_um"] = z * vz

    df = pd.DataFrame(data).sort_values(["nucleus_label", "z", "y", "x"])

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    base_name = Path(source_path).stem if source_path else "convex3"
    excel_path = output_dir_path / f"{base_name}{suffix}.xlsx"

    try:
        df.to_excel(excel_path, index=False)
        print(f"Coordinates saved to: {excel_path}")
        return excel_path
    except Exception as exc:
        print(f"Excel export failed ({exc}); saving CSV instead.")
        csv_path = output_dir_path / f"{base_name}{suffix}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Coordinates saved to CSV: {csv_path}")
        return csv_path


def save_results_npz(path, labels, voxel_size_xyz, channel_map=None):
    """
    Save key pipeline outputs to a compressed .npz file.

    Input:
    - path: destination file path
    - labels: labeled volume array
    - voxel_size_xyz: tuple with voxel spacing (x, y, z)
    - channel_map: optional channel name mapping dictionary

    Functions:
    - Package labels, voxel sizes, and channel map into a single archive
    - Use numpy.savez_compressed for compact storage

    Output:
    - Written .npz file at the provided path

    Additional information:
    - Stores an empty mapping when channel_map is None
    """
    np.savez_compressed(
        path,
        labels=labels,
        voxel_size_xyz=np.asarray(voxel_size_xyz, dtype=float),
        channel_map=channel_map if channel_map is not None else {},
    )


def load_results_npz(path):
    """
    Load pipeline outputs saved with save_results_npz.

    Input:
    - path: path to the .npz archive

    Functions:
    - Read the compressed archive with numpy.load
    - Unpack labels, voxel_size_xyz, and channel_map

    Output:
    - Dictionary containing labels, voxel_size_xyz, and channel_map

    Additional information:
    - Channel map is returned as a standard dict; defaults to {} if missing
    """
    data = np.load(path, allow_pickle=True)
    return {
        "labels": data["labels"],
        "voxel_size_xyz": tuple(data["voxel_size_xyz"]),
        "channel_map": data["channel_map"].item() if "channel_map" in data else {},
    }


def voxelize_hull_to_mask(hull, voxel_size_xyz, volume_shape, pad=1, epsilon=1e-9):
    """
    Convert a convex hull in physical space back to a voxel mask.

    Input:
    - hull: scipy ConvexHull object (physical coordinates)
    - voxel_size_xyz: voxel spacing (x, y, z)
    - volume_shape: target mask shape (Z, Y, X)
    - pad: padding voxels added around the hull bounds
    - epsilon: tolerance for inside-outside checks

    Functions:
    - Compute a bounding box on the voxel grid
    - Evaluate hull half-space equations on voxel centers
    - Assemble and return a boolean mask clipped to the volume

    Output:
    - Boolean mask of the hull on the provided grid

    Additional information:
    - Returns an all-false mask when bounds fall outside the volume
    """
    vx, vy, vz = voxel_size_xyz
    pts = hull.points

    x_min, y_min, z_min = pts.min(axis=0)
    x_max, y_max, z_max = pts.max(axis=0)

    x0 = max(int(np.floor(x_min / vx)) - pad, 0)
    x1 = min(int(np.ceil(x_max / vx)) + pad, volume_shape[2] - 1)
    y0 = max(int(np.floor(y_min / vy)) - pad, 0)
    y1 = min(int(np.ceil(y_max / vy)) + pad, volume_shape[1] - 1)
    z0 = max(int(np.floor(z_min / vz)) - pad, 0)
    z1 = min(int(np.ceil(z_max / vz)) + pad, volume_shape[0] - 1)

    if x0 > x1 or y0 > y1 or z0 > z1:
        return np.zeros(volume_shape, dtype=bool)

    zs, ys, xs = np.mgrid[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
    X = xs * vx
    Y = ys * vy
    Z = zs * vz

    coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    inside = np.ones(coords.shape[0], dtype=bool)
    for A, B, C, D in hull.equations:
        inside &= (A * coords[:, 0] + B * coords[:, 1] + C * coords[:, 2] + D <= epsilon)
        if not inside.any():
            break

    mask_sub = inside.reshape(X.shape)
    mask = np.zeros(volume_shape, dtype=bool)
    mask[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = mask_sub
    return mask


#---------- CONVEX HULL PIPELINE ----------------

def run_convhull_pipeline(
    lif_path=None,
    prompt_for_file=True,
    nucleus_name="Nucleus",
    channel_map=None,
    prompt_for_channel_names=True,
    min_size=500,
    merge_distance=30,
    z_compression_factor=None,
    min_hull_points=50,
    voxel_size_default=(1.0, 1.0, 1.0),
    plot_results=True,
):
    """
    Running the full nuclei segmentation and convex-hull workflow.

    Input:
    - lif_path: optional path to a .lif file
    - prompt_for_file: open a file dialog when True (lif_path is ignored)
    - nucleus_name: channel name to use as nucleus when bypassing prompts
    - channel_map: optional dict old -> new channel names to avoid prompts
    - prompt_for_channel_names: ask the user to rename channels when True
    - min_size: minimum voxel count to keep a segmented object
    - merge_distance: distance (voxels) to merge neighboring nuclei
    - z_compression_factor: optional compression factor for Z visualization
    - min_hull_points: minimum voxel count required to build a hull
    - voxel_size_default: fallback voxel sizes (z, y, x)
    - plot_results: whether to display the convex-hull plot

    Functions:
    - Load a LIF image (dialog or path)
    - Apply channel naming (mapping or prompt)
    - Extract the nucleus volume
    - Segment and merge nuclei
    - Compute convex hulls in physical space and optionally plot

    Output:
    - Dictionary with image, labels, voxel sizes, hulls, and properties
    """
    # Load image (prefer dialog when requested)
    if prompt_for_file:
        img = select_and_load_lif()
    elif lif_path:
        img = AICSImage(lif_path)
        img.file_path = lif_path
        print(f"Lif-file loaded: {lif_path}")
        print(f"Dimensions (T, C, Z, Y, X): {img.shape}")
    else:
        print("No file path provided and prompt_for_file is False; aborting pipeline.")
        return None

    if img is None:
        print("No image available; aborting pipeline.")
        return None

    # Channel naming strategy
    selected_channel_name = None
    selected_channel_index = None

    if channel_map:
        updated_channels = [channel_map.get(str(ch), str(ch)) for ch in img.channel_names]
        for idx, ch in enumerate(updated_channels):
            if ch.lower() == nucleus_name.lower():
                selected_channel_name = ch
                selected_channel_index = idx
                break
        print("Applied provided channel mapping.")
    elif prompt_for_channel_names:
        selected_channel_name, selected_channel_index, channel_map = rename_and_select_nucleus(img)
    else:
        updated_channels = [str(ch) for ch in img.channel_names]
        for idx, ch in enumerate(updated_channels):
            if ch.lower() == nucleus_name.lower():
                selected_channel_name = ch
                selected_channel_index = idx
                break
        channel_map = dict(zip(updated_channels, updated_channels))
        print("Using existing channel names without prompting.")

    if selected_channel_index is None:
        print("No nucleus channel selected; aborting pipeline.")
        return None

    # Volume extraction
    nucleus_volume = nucleus_volumes(
        img,
        selected_channel_name,
        selected_channel_index,
    )
    if nucleus_volume is None:
        return None

    # Segmentation and merging
    final_labels = segment_and_merge(
        nucleus_volume,
        min_size=min_size,
        merge_distance=merge_distance,
        z_compression_factor=z_compression_factor,
    )

    # Physical sizes and hulls
    vz, vy, vx = get_voxel_size_from_img(img, default=voxel_size_default)
    voxel_size_xyz = (vx, vy, vz)
    hulls, props = compute_convex_hulls_phys(
        final_labels,
        voxel_size_xyz,
        min_points=min_hull_points,
    )

    coords_excel_path = export_coordinates_to_excel(
        final_labels,
        getattr(img, "file_path", lif_path),
        output_dir="Excel_data_contour",
        suffix="_convex_coord",
        voxel_size_xyz=voxel_size_xyz,
        include_microns=True,
    )

    if plot_results:
        plot_convex_hulls_phys(
            final_labels,
            hulls,
            voxel_size_xyz,
            z_compression_factor=0.1 if z_compression_factor is None else z_compression_factor,
        )

    return {
        "image": img,
        "nucleus_volume": nucleus_volume,
        "labels": final_labels,
        "voxel_size_xyz": voxel_size_xyz,
        "hulls": hulls,
        "properties": props,
        "channel_map": channel_map,
        "coords_excel_path": coords_excel_path,
    }


def _interactive_demo():
    """
    Run the original interactive flow when executing this file directly.

    Input:
    - None (prompts the user interactively)

    Functions:
    - Load a LIF file via dialog
    - Select nucleus channel, segment, and merge nuclei
    - Compute and plot convex hulls
    - Export voxel coordinates with micron columns

    Output:
    - Displays plots and writes coordinate files; returns None

    Additional information:
    - Serves as a quick manual test of the convex hull pipeline
    """
    img = select_and_load_lif()
    if img is None:
        return

    selected_channel_name, selected_channel_index, _ = rename_and_select_nucleus(img)
    nucleus_volume = nucleus_volumes(img, selected_channel_name, selected_channel_index)
    if nucleus_volume is None:
        return

    final_labels = segment_and_merge(
        nucleus_volume,
        merge_distance=30,
        min_size=500,
        z_compression_factor=None,
    )

    vz, vy, vx = get_voxel_size_from_img(img)
    voxel_size_xyz = (vx, vy, vz)
    hulls, _ = compute_convex_hulls_phys(final_labels, voxel_size_xyz)

    plot_convex_hulls_phys(final_labels, hulls, voxel_size_xyz)
    export_coordinates_to_excel(
        final_labels,
        getattr(img, "file_path", None),
        voxel_size_xyz=voxel_size_xyz,
        include_microns=True,
    )


if __name__ == "__main__":
    _interactive_demo()
