"""
FUNCTIONALITY:
- Load .lif images and select the nucleus channel
- Segment nuclei, merge nearby detections, and compute convex hulls in physical space
- Fit ellipses per slice, stabilize orientations, and interpolate missing slices
- Rasterize hulls/ellipses to masks, compute IoU, and export summary plots
- Provide interactive visual QA (3D plots, slice overlays, IoU grids) and a demo flow
"""

import math
import numpy as np
import sys, time
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from aicsimageio import AICSImage
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, ball, binary_opening, binary_dilation
from skimage.measure import label, regionprops
from skimage.draw import ellipse
from sklearn.neighbors import KDTree
from skimage.segmentation import relabel_sequential


def _flip_over_x_axis(y_values):
    """Mirror y coordinates to flip visuals over the x-axis."""
    return -y_values


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

    # Open file drawer
    Tk().withdraw()
    lif_path = filedialog.askopenfilename(
        title="Select a lif-file",
        filetypes=[("Leica Image File", "*.lif")]
    )
    if not lif_path:
        print("NO FILE SELECTED.")
        return None

    # Save chosen file
    img = AICSImage(lif_path)
    img.file_path = lif_path

    # Display lif-file and its dimensions
    print(f"Lif-file loaded: {lif_path}")
    print(f"Dimensions (T, C, Z, Y, X): {img.shape}")
    return img

def rename_and_select_nucleus(img):
    """
    Rename channels and pick the nucleus channel.

    Input:
    - Loaded LIF image

    Functions:
    - List original channel names
    - Prompt for optional manual renaming
    - Display the updated names
    - Locate the channel named "Nucleus"
    - Build a mapping of original -> new names

    Output:
    - Selected nucleus channel name (or None)
    - Selected nucleus channel index (or None)
    - Dict with channel name mapping
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


def nucleus_volumes(img, selected_channel_name, selected_channel_index, intensity_threshold=None):
    """
    Retrieve the nucleus volume for a chosen channel.

    Input:
    - Loaded LIF image
    - Selected nucleus channel name
    - Selected nucleus channel index
    - Optional manual intensity threshold (currently unused)

    Functions:
    - Check that a valid channel index exists
    - Read the 3D ZYX stack for the nucleus channel
    - Print the resulting volume shape

    Output:
    - 3D numpy array (Z, Y, X) for the nucleus channel, or None when missing

    Additional information:
    - intensity_threshold is present for future use and not applied now
    """
    if selected_channel_index is None:
        print("No valid channel selected. Cannot retrieve volume.")
        return None

    # Retrieve 3D data
    volume = img.get_image_data("ZYX", C=selected_channel_index, T=0)
    print(f"Volume shape: {volume.shape} (Z, Y, X)")

    return volume

def segment_and_merge(volume, min_size=500, merge_distance=15, z_compression_factor=None):
    """
    Segment nuclei and merge nearby detections.

    Input:
    - 3D nucleus volume (Z, Y, X)
    - min_size: minimum voxel count to keep an object
    - merge_distance: maximum voxel distance to merge neighbors
    - z_compression_factor: optional Z compression for plots

    Functions:
    - Request Z compression when not provided
    - Apply Otsu thresholding and basic morphological cleanup
    - Label nuclei, detect close neighbors, and union overlapping labels
    - Relabel sequentially and render a compressed 3D scatter preview

    Output:
    - Labeled 3D array after merging
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
    z_idx, y_idx, x_idx = np.nonzero(final_labels)
    colors = final_labels[z_idx, y_idx, x_idx] - 1

    y_plot = _flip_over_x_axis(y_idx)
    z_scaled = z_idx * z_compression_factor

    # Unique colors
    cmap = plt.cm.get_cmap("tab20", final_labels.max())
    colors = cmap(colors)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        x_idx, y_plot, z_scaled,
        s=0.1,
        c=colors,
        alpha=0.85
    )

    # Adding label numbers
    final_props = regionprops(final_labels)
    for p in final_props:
        cz, cy, cx = p.centroid  # (z, y, x)
        cy = _flip_over_x_axis(cy)
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

def get_voxel_size_from_img(img, default=(1.0, 1.0, 1.0)):
    """
    Read physical voxel spacing from image metadata.

    Input:
    - Loaded LIF image
    - default: fallback voxel size tuple (z, y, x)

    Functions:
    - Access physical pixel sizes on the image object
    - Convert values to absolute floats to avoid negative spacing
    - Use the provided default when metadata is incomplete

    Output:
    - Tuple with voxel sizes (z, y, x)
    """
    pps = getattr(img, "physical_pixel_sizes", None)
    if pps is None or (pps.Z is None or pps.Y is None or pps.X is None):
        return default
    return (abs(float(pps.Z)), abs(float(pps.Y)), abs(float(pps.X)))

def compute_convex_hulls_phys(labeled_volume, voxel_size_xyz, min_points=50):
    """
    Build convex hulls for labeled nuclei in physical units.

    Input:
    - Labeled volume (3D array)
    - Voxel sizes in (x, y, z)
    - min_points: minimum voxel count required per hull

    Functions:
    - Convert labeled voxels to physical (x, y, z) coordinates
    - Create a convex hull per label when enough points exist
    - Store per-label properties such as volume, area, and centroid

    Output:
    - List of (label, hull) pairs
    - List of dictionaries with hull properties
    """
    vx, vy, vz = voxel_size_xyz  # (x, y, z) order
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
        pts_xyz = np.column_stack((xs, ys, zs))

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


def voxelize_hull_to_mask(hull, voxel_size_xyz, volume_shape, pad=1, epsilon=1e-9):
    """
    Rasterize a convex hull back onto a voxel grid.

    Input:
    - hull: scipy ConvexHull object in physical coordinates
    - voxel_size_xyz: voxel spacing (x, y, z)
    - volume_shape: target mask shape (Z, Y, X)
    - pad: padding voxels around the hull bounding box
    - epsilon: tolerance for inside-outside checks

    Functions:
    - Compute hull bounds and crop to the requested volume
    - Evaluate hull half-space equations on the voxel centers
    - Assemble a boolean mask covering the hull interior

    Output:
    - Boolean 3D mask for the given hull
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


def hulls_to_mask(hulls, voxel_size_xyz, volume_shape, pad=1):
    """
    Combine multiple convex hulls into one mask.

    Input:
    - hulls: list of (label, hull) pairs
    - voxel_size_xyz: voxel spacing (x, y, z)
    - volume_shape: target mask shape (Z, Y, X)
    - pad: padding voxels around each hull

    Functions:
    - Voxelize each hull on the target grid
    - Union all hull masks into a single boolean array

    Output:
    - Boolean mask containing all hull volumes
    """
    hull_mask = np.zeros(volume_shape, dtype=bool)
    for _, hull in hulls:
        hull_mask |= voxelize_hull_to_mask(hull, voxel_size_xyz, volume_shape, pad=pad)
    return hull_mask

def plot_convex_hulls_phys(labeled_volume, hulls, voxel_size_xyz, z_compression_factor=0.1, alpha_points=0.08):
    """
    Plot convex hulls in physical space with a compressed Z axis.

    Input:
    - Labeled volume
    - hulls: list of convex hulls
    - Voxel sizes in (x, y, z)
    - z_compression_factor: Z scaling for visualization
    - alpha_points: transparency for the voxel cloud

    Functions:
    - Convert labeled voxels to physical coordinates
    - Apply Z compression for visualization
    - Render a colored point cloud per label
    - Draw convex hull surfaces on top

    Output:
    - Displayed 3D matplotlib figure
    """
    vx, vy, vz = voxel_size_xyz

    # All voxel points
    z_idx, y_idx, x_idx = np.nonzero(labeled_volume)
    X = x_idx * vx
    Y = _flip_over_x_axis(y_idx) * vy
    Z = z_idx * vz

    # Z compression
    Zc = Z * z_compression_factor

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Point cloud (labels)
    ax.scatter(
        X, Y, Zc,
        s=0.05,
        c=labeled_volume[z_idx, y_idx, x_idx],
        cmap="tab10",
        alpha=alpha_points
    )

    # Colors for hulls
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(hulls))))

    # Draw hulls
    for i, (lbl, hull) in enumerate(hulls):

        # Copy physical points and compress Z
        pts = hull.points.copy()
        pts[:, 1] = _flip_over_x_axis(pts[:, 1])
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
    ax.set_title("3D convex hulls (compressed Z)")

    plt.tight_layout()
    plt.show(block=True)


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
    Run the full nuclei segmentation and convex-hull workflow.

    Input:
    - lif_path: optional path to a .lif file
    - prompt_for_file: open a file dialog when True (lif_path ignored)
    - nucleus_name: channel name used as nucleus when skipping prompts
    - channel_map: optional dict of old -> new names to avoid prompts
    - prompt_for_channel_names: ask the user to rename channels when True
    - min_size: minimum voxel count to keep a segmented object
    - merge_distance: voxel distance to merge neighboring nuclei
    - z_compression_factor: optional Z compression for visualization
    - min_hull_points: minimum voxel count required to build a hull
    - voxel_size_default: fallback voxel sizes (z, y, x)
    - plot_results: whether to display convex-hull plots

    Functions:
    - Load a LIF image from a dialog or provided path
    - Apply channel naming via mapping or interactive prompts
    - Extract, segment, and merge nucleus volumes
    - Compute convex hulls in physical space and generate summary plots

    Output:
    - Dictionary containing image, volumes, labels, voxel sizes, hulls, properties, and channel map
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

    print("\nGenerating convex-hull summary plots...")

    # Labeling per z-layer
    Z_layers = final_labels.shape[0]
    labels_per_layer = []

    for z in range(Z_layers):
        slab = final_labels[z, :, :]
        unique_labels = np.unique(slab)
        unique_labels = unique_labels[unique_labels > 0]
        labels_per_layer.append(len(unique_labels))

    # Otsu thresholding
    global_otsu_threshold = threshold_otsu(nucleus_volume)

    # 2D plots
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
    layers = list(range(Z_layers))

    # Bar plot: nuclei per layer
    bars = ax3.bar(layers, labels_per_layer, color='skyblue', alpha=0.7)
    ax3.set_xlabel('Layer (Z)')
    ax3.set_ylabel('Number of nuclei (labels)')
    ax3.set_title('Nuclei per Z-layer')
    ax3.grid(True, alpha=0.3)

    for bar, count in zip(bars, labels_per_layer):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')

    # Line plot: global threshold
    ax4.plot(layers,
            [global_otsu_threshold]*Z_layers,
            'o-', color='red', linewidth=2, markersize=6)
    ax4.fill_between(layers,
                    [global_otsu_threshold]*Z_layers,
                    alpha=0.3, color='red')

    ax4.set_xlabel('Layer (Z)')
    ax4.set_ylabel('Otsu threshold')
    ax4.set_title('Global Threshold (same for all Z)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
        

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
    }


def _augment_ellipse_with_units(e, voxel_size_xy=None):
    """
    Add physical units to ellipse measurements when voxel sizes are known.

    Input:
    - e: ellipse dictionary with pixel-based measurements
    - voxel_size_xy: optional spacing (x, y) in micrometers

    Functions:
    - Copy pixel-based center and radii
    - Convert center and radii to micrometers using voxel spacing
    - Derive major and minor diameters in micrometers

    Output:
    - Updated ellipse dictionary that includes micrometer fields

    Additional information:
    - Returns the original ellipse unchanged when voxel_size_xy is missing
    """
    if voxel_size_xy is None:
        return e

    vx, vy = voxel_size_xy  # (x, y) spacing in micrometers
    e["voxel_size_xy"] = (vx, vy)

    # Center positions in micrometers
    e["cx_um"] = e["cx"] * vx
    e["cy_um"] = e["cy"] * vy

    # Scale radii along the oriented axes
    phi = e["phi"]
    major_scale = math.sqrt((vx * math.cos(phi))**2 + (vy * math.sin(phi))**2)
    minor_phi = phi + math.pi / 2.0
    minor_scale = math.sqrt((vx * math.cos(minor_phi))**2 + (vy * math.sin(minor_phi))**2)

    e["a_um"] = e["a"] * major_scale
    e["b_um"] = e["b"] * minor_scale
    e["major_diameter_um"] = 2.0 * e["a_um"]
    e["minor_diameter_um"] = 2.0 * e["b_um"]
    return e

def _stabilize_ellipse_orientations(ellipses):
    """
    Avoid random 180 flips in orientation between adjacent Z-slices.

    Regionprops can return equivalent angles that differ by π (same ellipse,
    opposite direction). For each label, pick the orientation that is closest
    to the previous slice so the angle varies smoothly with Z.
    """
    if not ellipses:
        return []

    def wrap_half_pi(angle):
        # Map to [-pi/2, pi/2)
        return ((angle + math.pi / 2.0) % math.pi) - math.pi / 2.0

    stabilized = []
    by_label = {}
    for e in ellipses:
        by_label.setdefault(e["label"], []).append(e)

    for lbl, el_list in by_label.items():
        el_list = sorted(el_list, key=lambda e: e["z"])
        prev_phi = wrap_half_pi(el_list[0]["phi"])
        el_list[0]["phi"] = prev_phi
        stabilized.append(el_list[0])

        for e in el_list[1:]:
            cand = wrap_half_pi(e["phi"])
            # Choose the equivalent angle (phi or phi±pi) closest to the previous one
            options = [cand, cand + math.pi, cand - math.pi]
            best = min(options, key=lambda a: abs(a - prev_phi))
            e["phi"] = best
            prev_phi = best
            stabilized.append(e)

    return sorted(stabilized, key=lambda e: (e["label"], e["z"]))

def extract_ellipses_from_labels(
    final_labels,
    min_area_px=5,
    max_area_px=None,
    voxel_size_xy=None,
):
    """
    Derive ellipse fits per nucleus and per Z-layer.

    Input:
    - final_labels: labeled 3D nuclei volume
    - min_area_px: minimum area per ellipse in pixels
    - max_area_px: optional maximum area filter
    - voxel_size_xy: optional spacing (x, y) for physical units

    Functions:
    - Iterate over each Z slice and run regionprops on labels
    - Filter ellipses by pixel area
    - Store centroid, radii, rotation, and area
    - Add physical units when voxel spacing is provided

    Output:
    - List of ellipse dictionaries
    - Console summary of ellipse counts per Z-layer
    """
    Z, Y, X = final_labels.shape

    ellipses = []
    ellipses_per_z = np.zeros(Z, dtype=int)

    for z in range(Z):
        slice_labels = final_labels[z]

        # If something is in this layer
        if not np.any(slice_labels):
            continue

        # Regionprops on the original labels
        regions = regionprops(slice_labels)

        for r in regions:
            area = r.area
            if area < min_area_px:
                continue
            if max_area_px is not None and area > max_area_px:
                continue

            lbl = int(r.label)
            cy, cx = r.centroid
            a = r.major_axis_length / 2.0
            b = r.minor_axis_length / 2.0
            if a == 0 or b == 0:
                continue

            # Robust orientation: compute major-axis direction from PCA on (x=cols, y=rows)
            coords = r.coords
            xs = coords[:, 1]
            ys = coords[:, 0]
            xs_c = xs - xs.mean()
            ys_c = ys - ys.mean()
            cov = np.cov(np.stack([xs_c, ys_c]), bias=True)
            evals, evecs = np.linalg.eigh(cov)  # ascending eigenvalues
            vx_major, vy_major = evecs[:, np.argmax(evals)]
            phi = math.atan2(vy_major, vx_major)  # CCW from +x (cols) axis
            # Wrap to [-pi/2, pi/2) to avoid 180° flips
            if phi >= math.pi / 2.0:
                phi -= math.pi
            if phi < -math.pi / 2.0:
                phi += math.pi
            # Angle from the row (y) axis, as expected by skimage.draw.ellipse
            phi_row = math.atan2(vx_major, vy_major)

            ellipses.append(_augment_ellipse_with_units({
                "label": lbl,
                "z": int(z),
                "cx": float(cx),
                "cy": float(cy),
                "a": float(a),
                "b": float(b),
                "phi": float(phi),
                "phi_row": float(phi_row),
                "area_px": float(area),
                "source": "measured",
            }, voxel_size_xy=voxel_size_xy))
            ellipses_per_z[z] += 1

    print(f"\nMeasured ellipses (per slice): {len(ellipses)}")
    print("Ellipses per Z-layer (before interpolation):")
    for z in range(Z):
        print(f"  Z={z:2d}: {ellipses_per_z[z]} ellipses")

    return _stabilize_ellipse_orientations(ellipses)

def densify_ellipses_by_interpolation(ellipses, Z):
    """
    Fill in missing Z-slices by interpolating ellipse parameters.

    Input:
    - ellipses: list of measured ellipse dictionaries
    - Z: total number of slices in the volume

    Functions:
    - Group ellipses per nucleus label and sort by Z
    - Keep measured ellipses intact
    - Linearly interpolate centers, radii, rotation, and area for missing slices
    - Preserve voxel size metadata when present

    Output:
    - List with measured and interpolated ellipse dictionaries
    - Console summary of ellipse counts per Z-layer after interpolation
    """
    if not ellipses:
        return []

    voxel_size_xy = None
    # Group per label
    by_label = {}
    for e in ellipses:
        if voxel_size_xy is None and isinstance(e, dict):
            voxel_size_xy = e.get("voxel_size_xy")
        by_label.setdefault(e["label"], []).append(e)

    dense_ellipses = []

    for lbl, el_list in by_label.items():
        # sort for z
        el_list = sorted(el_list, key=lambda e: e["z"])
        zs_meas = np.array([e["z"] for e in el_list], dtype=int)

        # Keep measured ellips
        dense_ellipses.extend(el_list)

        z_min = zs_meas.min()
        z_max = zs_meas.max()

        # Fill gaps in z-layer
        for z in range(z_min, z_max + 1):
            if z in zs_meas:
                continue  

            lower_idxs = np.where(zs_meas < z)[0]
            upper_idxs = np.where(zs_meas > z)[0]
            if len(lower_idxs) == 0 or len(upper_idxs) == 0:
                continue  

            z1_idx = lower_idxs[-1]
            z2_idx = upper_idxs[0]

            e1 = el_list[z1_idx]
            e2 = el_list[z2_idx]

            z1 = e1["z"]
            z2 = e2["z"]
            if z2 == z1:
                continue

            t = (z - z1) / (z2 - z1)

            def lerp(v1, v2):
                return (1.0 - t) * v1 + t * v2

            cx = lerp(e1["cx"], e2["cx"])
            cy = lerp(e1["cy"], e2["cy"])
            a = lerp(e1["a"], e2["a"])
            b = lerp(e1["b"], e2["b"])
            phi = lerp(e1["phi"], e2["phi"])

            dense_ellipses.append(_augment_ellipse_with_units({
                "label": lbl,
                "z": int(z),
                "cx": float(cx),
                "cy": float(cy),
                "a": float(a),
                "b": float(b),
                "phi": float(phi),
                "area_px": float(lerp(e1["area_px"], e2["area_px"])),
                "source": "interp",
            }, voxel_size_xy=voxel_size_xy))

    dense_ellipses = sorted(dense_ellipses, key=lambda e: (e["label"], e["z"]))

    ellipses_per_z = np.zeros(Z, dtype=int)
    for e in dense_ellipses:
        ellipses_per_z[e["z"]] += 1

    print(f"\nTotal ellipses after interpolation: {len(dense_ellipses)}")
    print("Ellipses per Z-layer (after interpolation):")
    for z in range(Z):
        print(f"  Z={z:2d}: {ellipses_per_z[z]} ellipses")

    return dense_ellipses


def ellipses_to_mask(ellipses, volume_shape):
    """
    Rasterize ellipse definitions onto a 3D mask.

    Input:
    - ellipses: list of ellipse dictionaries with Z indices
    - volume_shape: target mask shape (Z, Y, X)

    Functions:
    - Loop through ellipses and draw each ellipse on its Z slice
    - Skip ellipses that fall outside the provided Z range

    Output:
    - Boolean mask containing all rasterized ellipses
    """
    mask = np.zeros(volume_shape, dtype=bool)
    Z, Y, X = volume_shape

    # Global rotation tweak (radians) to flip/align ellipses with the convex hulls
    rotation_offset_rad = math.pi / 2.0  # adjust if further fine-tuning is needed

    for e in ellipses:
        z_idx = e["z"]
        if z_idx < 0 or z_idx >= Z:
            continue

        # Use per-ellipse orientation measured from the row (Y) axis.
        # Falls back to converting from phi (measured from +X) when needed.
        rotation_from_row_axis = e.get("phi_row", e["phi"] - math.pi / 2.0) + rotation_offset_rad
        rr, cc = ellipse(
            r=e["cy"],
            c=e["cx"],
            r_radius=e["b"],
            c_radius=e["a"],
            shape=(Y, X),
            rotation=rotation_from_row_axis,
        )
        mask[z_idx, rr, cc] = True

    return mask


def compute_iou(convex_mask, ellipse_mask):
    """
    Compute intersection over union (IoU) between two masks.

    Input:
    - convex_mask: boolean 3D mask from convex hulls
    - ellipse_mask: boolean 3D mask from ellipses

    Functions:
    - Validate matching shapes
    - Calculate IoU per Z-slice
    - Calculate overall 3D IoU

    Output:
    - Array with per-layer IoU values (NaN when undefined)
    - Single float with total IoU (NaN when undefined)
    """
    if convex_mask.shape != ellipse_mask.shape:
        raise ValueError(f"Mask shapes differ: {convex_mask.shape} vs {ellipse_mask.shape}")

    per_layer = []
    Z = convex_mask.shape[0]
    for z in range(Z):
        inter = np.logical_and(convex_mask[z], ellipse_mask[z]).sum()
        union = np.logical_or(convex_mask[z], ellipse_mask[z]).sum()
        if union == 0:
            per_layer.append(np.nan)
        else:
            per_layer.append(inter / union)

    total_inter = np.logical_and(convex_mask, ellipse_mask).sum()
    total_union = np.logical_or(convex_mask, ellipse_mask).sum()
    total_iou = float(total_inter / total_union) if total_union > 0 else np.nan

    return np.array(per_layer), total_iou


def plot_convex_and_ellipses_3d(
    hulls,
    ellipses,
    voxel_size_xyz,
    total_iou,
    z_compression_factor=0.1,
    ellipse_rotation_offset=0.0,
):
    """
    Show convex hull surfaces and ellipse rings together in 3D.

    Input:
    - hulls: list of (label, hull) pairs
    - ellipses: list of ellipse dictionaries
    - voxel_size_xyz: voxel spacing (x, y, z)
    - total_iou: overall IoU value to display
    - z_compression_factor: Z scaling for visualization
    - ellipse_rotation_offset: optional rotation (radians) applied to ellipse rings to match hull orientation

    Functions:
    - Render convex hull surfaces with compressed Z
    - Draw ellipse contours at their respective Z positions (with optional rotation correction)
    - Annotate the plot title with the IoU value

    Output:
    - Displayed 3D matplotlib figure combining hulls and ellipses
    """
    vx, vy, vz = voxel_size_xyz
    plt.close("all")
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Draw convex hull surfaces
    for lbl, hull in hulls:
        pts = hull.points.copy()
        pts[:, 1] = _flip_over_x_axis(pts[:, 1])
        pts[:, 2] *= z_compression_factor  # compress z
        faces = []
        for tri_idx in hull.simplices:
            tri = pts[tri_idx]
            faces.append(tri)
        poly = Poly3DCollection(faces, alpha=0.35, edgecolor="#7bbce5", facecolor="#9ad5f5", linewidths=0.3)
        ax.add_collection3d(poly)

    # Draw ellipse rings
    t = np.linspace(0, 2 * np.pi, 120)
    for e in ellipses:
        cx = e.get("cx_um", e["cx"] * vx)
        cy = e.get("cy_um", e["cy"] * vy)
        a = e.get("a_um", e["a"] * vx)
        b = e.get("b_um", e["b"] * vy)
        phi = e["phi"] + ellipse_rotation_offset
        z_idx = e["z"]

        xs = cx + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
        ys = cy + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)
        zs = np.full_like(xs, z_idx * vz * z_compression_factor, dtype=float)
        ys = _flip_over_x_axis(ys)
        ax.plot(xs, ys, zs, color="#1f6f3d", linewidth=1.6, alpha=0.95)

    ax.set_box_aspect((1, 1, z_compression_factor))
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_zlabel(f"Z (µm x{z_compression_factor})")
    iou_text = "n/a" if np.isnan(total_iou) else f"{total_iou:.3f}"
    ax.set_title(f"Convex hulls (light blue) vs ellipse rings (green) — 3D IoU = {iou_text}")
    ax.legend(handles=[
        plt.Line2D([0], [0], color="#9ad5f5", lw=6, label="Convex hull"),
        plt.Line2D([0], [0], color="#1f6f3d", lw=2.5, label="Ellipse ring"),
    ], loc="upper right")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.show()


def plot_iou_grid_overlays(convex_mask, ellipse_mask, per_layer_iou, cols=5):
    """
    Display per-layer overlays of hull and ellipse masks with IoU labels.

    Input:
    - convex_mask: boolean 3D mask from convex hulls
    - ellipse_mask: boolean 3D mask from ellipses
    - per_layer_iou: array of IoU values per Z-layer
    - cols: number of columns in the grid

    Functions:
    - Build RGB overlays for each slice highlighting hull, ellipse, and overlap
    - Arrange overlays in a grid with slice-specific IoU in the title
    - Add a shared legend for color meaning

    Output:
    - Displayed matplotlib grid of 2D overlays
    """
    Z, Y, X = convex_mask.shape
    rows = math.ceil(Z / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), facecolor="white")
    fig.patch.set_facecolor("white")
    axes = np.atleast_2d(axes)

    for idx in range(rows * cols):
        ax = axes.flat[idx]
        if idx >= Z:
            ax.axis("off")
            continue

        overlay = np.zeros((Y, X, 3), dtype=float)
        hull_slice = convex_mask[idx]
        ellipse_slice = ellipse_mask[idx]
        inter_slice = np.logical_and(hull_slice, ellipse_slice)

        overlay[..., 2] = hull_slice * 0.6
        overlay[..., 1] = ellipse_slice * 0.7       
        overlay[..., 0] = inter_slice * 0.7         

        iou_val = per_layer_iou[idx]
        label = "n/a" if np.isnan(iou_val) else f"{iou_val:.3f}"

        ax.imshow(overlay)
        ax.set_title(f"Z={idx} IoU={label}", fontsize=9)
        ax.axis("off")
        ax.set_facecolor("white")

    # Shared legend for colors
    legend_handles = [
        plt.Line2D([0], [0], color="blue", lw=4, label="Convex hull"),
        plt.Line2D([0], [0], color="green", lw=4, label="Ellipses"),
        plt.Line2D([0], [0], color="gray", lw=4, label="Overlap"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="gray",
    )

    plt.tight_layout()
    plt.show()

def plot_iou_results(per_layer_iou, total_iou, convex_mask, ellipse_mask):
    """
    Visualize IoU metrics and a projection overlay.

    Input:
    - per_layer_iou: array of IoU values per Z-layer
    - total_iou: overall 3D IoU value
    - convex_mask: boolean 3D mask from convex hulls
    - ellipse_mask: boolean 3D mask from ellipses

    Functions:
    - Plot IoU per slice with a line indicating total IoU
    - Create a max-projection overlay of both masks
    - Add legends explaining color coding

    Output:
    - Displayed matplotlib figure with IoU plot and projection overlay
    """
    Z = convex_mask.shape[0]
    layers = np.arange(Z)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    fig.patch.set_facecolor("white")

    axes[0].plot(layers, per_layer_iou, "o-", color="steelblue")
    axes[0].axhline(total_iou, color="orange", linestyle="--", label=f"3D IoU = {total_iou:.3f}")
    axes[0].set_xlabel("Z-layer")
    axes[0].set_ylabel("IoU")
    axes[0].set_title("IoU per Z-layer")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Max projection overlay
    convex_proj = convex_mask.any(axis=0)
    ellipse_proj = ellipse_mask.any(axis=0)
    inter_proj = convex_proj & ellipse_proj

    overlay = np.zeros(convex_proj.shape + (3,), dtype=float)
    overlay[..., 0] = ellipse_proj
    overlay[..., 1] = inter_proj    
    overlay[..., 2] = convex_proj   

    axes[1].imshow(overlay)
    axes[1].set_title("Max projection overlay")
    axes[1].axis("off")
    axes[0].set_facecolor("white")
    axes[1].set_facecolor("white")

    # Legend for colors
    legend_handles = [
        plt.Line2D([0], [0], color="red", lw=4, label="Ellipses"),
        plt.Line2D([0], [0], color="blue", lw=4, label="Convex hull"),
        plt.Line2D([0], [0], color="gray", lw=4, label="Overlap"),
    ]
    axes[1].legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="gray",
    )

    plt.tight_layout()
    plt.show()


def plot_image_hull_ellipse_slice(
    nucleus_volume,
    convex_mask,
    ellipse_mask,
    z_layer,
    window_percentiles=(1, 99),
):

    Z = nucleus_volume.shape[0]
    if z_layer < 0 or z_layer >= Z:
        print(f"Z layer {z_layer} out of range (0..{Z - 1}).")
        return

    img = nucleus_volume[z_layer].astype(float)
    if window_percentiles is not None and img.size > 0:
        nonzero = img[img > 0]
        base = nonzero if nonzero.size > 0 else img.ravel()
        vmin, vmax = np.percentile(base, window_percentiles)
        if vmax == vmin:
            vmin, vmax = img.min(), img.max()
    else:
        vmin, vmax = img.min(), img.max()

    plt.figure(figsize=(6, 6), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)

    hull_slice = convex_mask[z_layer]
    ellipse_slice = ellipse_mask[z_layer]

    # Contours for hull (blue) and ellipse (green)
    ax.contour(hull_slice, levels=[0.5], colors=["blue"], linewidths=1.5, alpha=0.9)
    ax.contour(ellipse_slice, levels=[0.5], colors=["green"], linewidths=1.5, alpha=0.9)

    ax.set_title(f"Nucleus signal with hull (blue) and ellipse (green) — Z={z_layer}")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_image_hull_ellipse_grid(
    nucleus_volume,
    convex_mask,
    ellipse_mask,
    cols=5,
    window_percentiles=(1, 99),
):
    
    Z, Y, X = nucleus_volume.shape
    rows = math.ceil(Z / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), facecolor="white")
    axes = np.atleast_2d(axes)

    # Shared intensity window for consistent contrast
    img_flat = nucleus_volume.astype(float).ravel()
    if window_percentiles is not None and img_flat.size > 0:
        vmin, vmax = np.percentile(img_flat, window_percentiles)
        if vmax == vmin:
            vmin, vmax = img_flat.min(), img_flat.max()
    else:
        vmin, vmax = img_flat.min(), img_flat.max()

    for idx in range(rows * cols):
        ax = axes.flat[idx]
        if idx >= Z:
            ax.axis("off")
            continue

        ax.imshow(nucleus_volume[idx], cmap="gray", vmin=vmin, vmax=vmax)
        hull_slice = convex_mask[idx]
        ellipse_slice = ellipse_mask[idx]
        ax.contour(hull_slice, levels=[0.5], colors=["blue"], linewidths=1.0, alpha=0.9)
        ax.contour(ellipse_slice, levels=[0.5], colors=["green"], linewidths=1.0, alpha=0.9)
        ax.set_title(f"Z={idx}", fontsize=9)
        ax.axis("off")
        ax.set_facecolor("white")

    plt.tight_layout()
    plt.show()

def plot_ellipses_3d(
    ellipses,
    voxel_size_xyz=None,
    z_compression_factor=0.1,
    n_ellipse_points=100,
    ellipse_rotation_offset=0.0,
):
    """
    Plot ellipse contours in 3D at their respective Z positions.

    Input:
    - ellipses: list of ellipse dictionaries
    - voxel_size_xyz: optional spacing (x, y, z)
    - z_compression_factor: Z scaling for visualization
    - n_ellipse_points: number of points used to draw each ellipse
    - ellipse_rotation_offset: optional rotation (radians) applied to ellipse rings to match hull orientation

    Functions:
    - Color ellipses by label
    - Convert pixel or micrometer values to 3D coordinates
    - Draw ellipse rings at the correct Z height with optional compression and rotation correction

    Output:
    - Displayed 3D matplotlib figure with ellipse contours

    Additional information:
    - Prints a message and returns when no ellipses are provided
    """
    if not ellipses:
        print("No ellips to plot.")
        return

    plt.close("all")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    vx, vy, vz = (1.0, 1.0, 1.0) if voxel_size_xyz is None else voxel_size_xyz
    labels = sorted(set(e["label"] for e in ellipses))
    cmap = plt.get_cmap("tab10")
    label_to_color = {lbl: cmap(i % 10) for i, lbl in enumerate(labels)}

    t = np.linspace(0, 2 * np.pi, n_ellipse_points)

    for e in ellipses:
        cx = e.get("cx_um", e["cx"] * vx)
        cy = e.get("cy_um", e["cy"] * vy)
        a = e.get("a_um", e["a"] * vx)
        b = e.get("b_um", e["b"] * vy)
        phi = e["phi"] + ellipse_rotation_offset
        z_idx = e["z"]
        lbl = e["label"]

        xs = cx + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
        ys = cy + a * np.cos(t) * np.sin(phi) + b * np.cos(phi) * np.sin(t)
        zs = np.full_like(xs, z_idx * vz * z_compression_factor, dtype=float)
        ys = _flip_over_x_axis(ys)

        ax.plot(xs, ys, zs, color="#1f6f3d", linewidth=1.6)

    ax.set_box_aspect((1, 1, z_compression_factor))
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel(f"z (µm x{z_compression_factor:.2f})")
    ax.set_title("3D ellips contours")
    plt.tight_layout()
    plt.show()
    


def show_slice_and_ellipses(final_labels, ellipses, z_layer):
    """
    Show a single Z-layer with its nuclei mask and ellipses for visual QA.

    Input:
    - final_labels: labeled nuclei volume
    - ellipses: list of ellipse dictionaries
    - z_layer: slice index to display

    Functions:
    - Validate the requested Z index
    - Plot the binary mask for the selected slice
    - Overlay ellipse contours that belong to the slice

    Output:
    - Displayed matplotlib figure of the requested slice

    Additional information:
    - Prints the number of ellipses drawn for quick inspection
    """
    Z, Y, X = final_labels.shape
    if z_layer < 0 or z_layer >= Z:
        print(f"Z laag {z_layer} buiten bereik (0..{Z - 1}).")
        return

    mask = final_labels[z_layer] > 0

    plt.figure(figsize=(5, 5), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")
    plt.imshow(mask, cmap="gray")

    t = np.linspace(0, 2 * np.pi, 100)

    count_here = 0
    for e in ellipses:
        if e["z"] != z_layer:
            continue

        count_here += 1
        cx = e["cx"]
        cy = e["cy"]
        a = e["a"]
        b = e["b"]
        phi = e["phi"]

        xs = cx + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
        ys = cy + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)

        plt.plot(xs, ys, color="#1f6f3d", linewidth=1.6)

    print(f"Slice Z={z_layer}: {count_here} ellips drawn.")
    plt.title(f"Z layer {z_layer} with ellips")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    #plt.pause(0.001) # Turn on when skipping plot

def show_all_slices_with_ellipses(final_labels, ellipses, cols=5):
    """
    Grid view of every Z-layer with ellipses overlaid.
    Helps verify orientation/placement consistency across the stack.
    """
    Z, Y, X = final_labels.shape
    rows = math.ceil(Z / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), facecolor="white")
    fig.patch.set_facecolor("white")
    axes = np.atleast_2d(axes)

    t = np.linspace(0, 2 * np.pi, 100)
    ellipses_by_z = {}
    for e in ellipses:
        ellipses_by_z.setdefault(e["z"], []).append(e)

    for idx in range(rows * cols):
        ax = axes.flat[idx]
        if idx >= Z:
            ax.axis("off")
            continue

        mask = final_labels[idx] > 0
        ax.imshow(mask, cmap="gray")

        for e in ellipses_by_z.get(idx, []):
            cx = e["cx"]
            cy = e["cy"]
            a = e["a"]
            b = e["b"]
            phi = e["phi"]

            xs = cx + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
            ys = cy + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)
            ax.plot(xs, ys, color="#1f6f3d", linewidth=1.2)

        ax.set_title(f"Z={idx}")
        ax.axis("off")
        ax.set_facecolor("white")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1) LIF bestand kiezen en laden
    img = select_and_load_lif()
    if img is None:
        raise SystemExit("Geen LIF bestand gekozen, script stopt.")

    # 2) Kanaalnamen instellen en nucleus-kanaal kiezen
    selected_channel_name, selected_channel_index, channel_map = rename_and_select_nucleus(img)

    # 3) Nucleus volume pakken
    nucleus_volume = nucleus_volumes(img, selected_channel_name, selected_channel_index)
    if nucleus_volume is None:
        raise SystemExit("Geen nucleus volume gevonden, script stopt.")

    # 4) Z-compressie (alleen visualisatie) en segmentatie
    try:
        z_comp = float(
            input("Enter desired Z compression for 3D plots (e.g. 0.3 = flatter, 1.0 = no compression): ")
        )
    except ValueError:
        z_comp = 1.0
        print("Invalid input, default value 1.0 used.")

    final_labels = segment_and_merge(
        nucleus_volume,
        merge_distance=30,
        min_size=300,
        z_compression_factor=z_comp,
    )

    # 5) Pixel- en micron-afmetingen ophalen
    vz, vy, vx = get_voxel_size_from_img(img)
    voxel_size_xyz = (vx, vy, vz)

    # 5) Echte ellipsen uit labels halen
    base_ellipses = extract_ellipses_from_labels(
        final_labels,
        min_area_px=5,
        max_area_px=None,
        voxel_size_xy=(vx, vy),
    )

    # 6) Ellipsen dichter maken door interpolatie in Z en één keer tonen
    Z = final_labels.shape[0]
    dense_ellipses = densify_ellipses_by_interpolation(base_ellipses, Z)
    ellipses_for_plot = dense_ellipses

    plot_ellipses_3d(
        ellipses_for_plot,
        voxel_size_xyz=voxel_size_xyz,
        z_compression_factor=z_comp,
        n_ellipse_points=80,
        ellipse_rotation_offset=0.0,
    )

    # Multiplot of all layers to inspect ellipse placement/orientation
    show_all_slices_with_ellipses(final_labels, ellipses_for_plot, cols=5)

    show_slice_and_ellipses(final_labels, ellipses_for_plot, z_layer=5)

    # 9) IoU tussen convex hull volume en ellipse volume (per Z en totaal)
    hulls, _ = compute_convex_hulls_phys(final_labels, voxel_size_xyz)

    # 10) Standalone convex hull visualization (physical space)
    plot_convex_hulls_phys(
        final_labels,
        hulls,
        voxel_size_xyz,
        z_compression_factor=z_comp,
    )

    convex_mask = hulls_to_mask(hulls, voxel_size_xyz, final_labels.shape, pad=1)
    ellipse_mask = ellipses_to_mask(ellipses_for_plot, final_labels.shape)

    per_layer_iou, total_iou = compute_iou(convex_mask, ellipse_mask)

    # Print IoU as a simple list of numbers (4 decimals), NaN when undefined
    print("\nIoU per Z-layer:")
    for z, iou in enumerate(per_layer_iou):
        label = f"Z={z:02d}"
        if np.isnan(iou):
            print(f"{label}: n/a (no overlap or union)")
        else:
            print(f"{label}: {iou:.4f}")
    if np.isnan(total_iou):
        print("\nTotal 3D IoU: n/a (no overlap or union)")
    else:
        print(f"\nTotal 3D IoU: {total_iou:.4f}")

    plot_iou_results(per_layer_iou, total_iou, convex_mask, ellipse_mask)

    # 11) Extra visualisaties: gecombineerde 3D plot en IoU grid
    plot_convex_and_ellipses_3d(
        hulls,
        ellipses_for_plot,
        voxel_size_xyz,
        total_iou,
        z_compression_factor=z_comp,
    )
    plot_iou_grid_overlays(convex_mask, ellipse_mask, per_layer_iou, cols=5)

    # 12) Visual check: raw image with hull/ellipse contours on a chosen slice
    mid_z = final_labels.shape[0] // 2
    plot_image_hull_ellipse_slice(
        nucleus_volume,
        convex_mask,
        ellipse_mask,
        z_layer=mid_z,
        window_percentiles=(1, 99),
    )

    # 13) Grid: raw images with hull/ellipse overlays for all layers
    plot_image_hull_ellipse_grid(
        nucleus_volume,
        convex_mask,
        ellipse_mask,
        cols=5,
        window_percentiles=(1, 99),
    )
