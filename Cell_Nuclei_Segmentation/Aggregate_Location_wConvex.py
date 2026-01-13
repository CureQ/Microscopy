"""
FUNCTIONALITY:
- Dynamically load the convex-hull pipeline from Convex_Hull.py
- Run convex-hull and aggregate segmentation on the same LIF file
- Collect channel mappings once and reuse for both pipelines
- Plot convex hulls alongside aggregates in 3D with optional mirroring
- Provide a CLI flow to pick channels and render overlays
"""

import importlib.util
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import binary_dilation
from Aggregate_Detection import run_aggregaten_pipeline
from notebook_file_picker import ensure_local_file


def load_convex_module():
    """
    Dynamically import Convex_Hull.py to reuse its pipeline functions.

    Input:
    - None (uses the current file location to find Convex_Hull.py)

    Functions:
    - Locate Convex_Hull.py next to this script
    - Build a module spec and execute the module
    - Return the loaded module object for reuse

    Output:
    - Imported module object exposing convex-hull helpers

    Additional information:
    - Raises FileNotFoundError when Convex_Hull.py is missing
    - Raises ImportError if the module spec cannot be created or executed
    """
    module_path = Path(__file__).with_name("Convex_Hull.py")
    if not module_path.exists():
        raise FileNotFoundError(f"Convex_Hull.py not found next to {__file__}")

    spec = importlib.util.spec_from_file_location("convex", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not create spec for Convex_Hull.py")

    convex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(convex)
    return convex


convex = load_convex_module()
run_convhull_pipeline = convex.run_convhull_pipeline


def plot_hulls_and_aggregates(hulls, voxel_size_xyz, agg_mask, nuclei_mask, z_compression=0.1, mirror_xy=True):
    """
    Plot convex hulls (blue) and aggregates (green/red if touching nuclei) in 3D.

    Input:
    - hulls: list of (label, hull) pairs
    - voxel_size_xyz: voxel spacing tuple (x, y, z)
    - agg_mask: boolean aggregate mask
    - nuclei_mask: boolean nuclei mask to test touching
    - z_compression: factor to compress Z for visualization
    - mirror_xy: whether to mirror X/Y coordinates for display

    Functions:
    - Mirror coordinates when requested to align with image space
    - Draw convex hull wireframes with compressed Z
    - Dilate nuclei to flag touching aggregates and color voxels red/green
    - Scatter aggregate voxels in physical units

    Output:
    - Displayed matplotlib 3D figure combining hulls and aggregates

    Additional information:
    - Touching test uses a 1-voxel dilation; adjust iterations if needed
    """
    vx, vy, vz = voxel_size_xyz  # (x, y, z) spacing

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Determine extents in physical units for mirroring
    max_x = 0.0
    max_y = 0.0
    if hulls:
        for _, hull in hulls:
            pts = hull.points
            max_x = max(max_x, pts[:, 0].max())
            max_y = max(max_y, pts[:, 1].max())
    z_idx, y_idx, x_idx = np.nonzero(agg_mask)
    if len(x_idx):
        max_x = max(max_x, (x_idx * vx).max())
        max_y = max(max_y, (y_idx * vy).max())

    def mirror_points(xs, ys):
        if not mirror_xy:
            return xs, ys
        return max_x - xs, max_y - ys

    # Draw hulls as wireframes
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, max(1, len(hulls))))
    for i, (lbl, hull) in enumerate(hulls):
        pts = hull.points.copy()  # (x, y, z)
        pts[:, 0], pts[:, 1] = mirror_points(pts[:, 0], pts[:, 1])
        pts[:, 2] *= z_compression
        for tri_idx in hull.simplices:
            tri = Poly3DCollection([pts[tri_idx]], alpha=0.15, edgecolor="k", linewidths=0.5)
            tri.set_facecolor(colors[i % len(colors)])
            ax.add_collection3d(tri)

    # Touching test: dilate nuclei mask by 1 voxel
    dilated_nuclei = binary_dilation(nuclei_mask, iterations=1)
    touching = agg_mask & dilated_nuclei

    # Aggregates in physical units
    z_idx, y_idx, x_idx = np.nonzero(agg_mask)
    if len(x_idx):
        X = x_idx * vx
        Y = y_idx * vy
        X, Y = mirror_points(X, Y)
        Z = z_idx * vz * z_compression
        # Color per-voxel based on touching
        touching_flat = touching[z_idx, y_idx, x_idx]
        colors = np.where(touching_flat, "red", "lime")
        ax.scatter(X, Y, Z, s=5, c=colors, alpha=0.8, label="Aggregates")

    # Axis settings
    ax.set_xlabel("X (units)")
    ax.set_ylabel("Y (units)")
    ax.set_zlabel(f"Z (compressed x{z_compression})")
    ax.set_title("Convex hulls (blue) and aggregates (green/red)")
    ax.set_box_aspect((1, 1, z_compression))
    plt.tight_layout()
    plt.show()


def main():
    """
    Run convex-hull and aggregate pipelines on a shared LIF and plot overlays.

    Input:
    - None (prompts the user for file selection and channel choices)

    Functions:
    - Pick a .lif file and build a channel mapping once
    - Ask for nucleus and aggregate channel indices
    - Execute convex-hull and aggregate pipelines with shared mapping
    - Validate matching shapes and render a combined 3D plot

    Output:
    - Displays plots; returns None

    Additional information:
    - Aborts early when file selection, channel selection, or shape checks fail
    """
    lif_path = ensure_local_file(prompt_for_file=True, accept=".lif", description="Upload shared LIF")

    from aicsimageio import AICSImage
    img = AICSImage(lif_path)
    original_channels = [str(ch) for ch in img.channel_names]
    print("\nAvailable channels:")
    for i, ch in enumerate(original_channels):
        print(f"[{i}] {ch}")
    print("\nRename channels (press Enter to keep the current name):")
    new_channels = []
    for i, ch in enumerate(original_channels):
        new_name = input(f"New name for channel {i} (currently '{ch}'): ").strip()
        if new_name == "":
            new_name = ch
        new_channels.append(new_name)
    channel_map = dict(zip(original_channels, new_channels))

    print("\nRenamed channels:")
    for i, ch in enumerate(new_channels):
        print(f"[{i}] {ch}")
    try:
        nuc_idx = int(input("Index for nucleus channel: ").strip())
        agg_idx = int(input("Index for aggregate channel: ").strip())
    except Exception:
        print("Invalid indices.")
        return
    nucleus_name = new_channels[nuc_idx]
    aggregate_name = new_channels[agg_idx]

    convex_res = run_convhull_pipeline(
        lif_path=lif_path,
        prompt_for_file=False,
        prompt_for_channel_names=False,
        channel_map=channel_map,
        nucleus_name=nucleus_name,
    )
    if convex_res is None:
        print("Convex pipeline aborted.")
        return

    agg_res = run_aggregaten_pipeline(
        lif_path=lif_path,
        prompt_for_file=False,
        prompt_for_channel_names=False,
        channel_map=channel_map,
        aggregate_name=aggregate_name,
        show_plots=False,
    )
    if agg_res is None:
        print("Aggregate pipeline aborted.")
        return

    nuclei_labels = convex_res["labels"]
    agg_mask = agg_res["aggregates_mask"]

    if nuclei_labels.shape != agg_mask.shape:
        print(f"Shape mismatch: nuclei {nuclei_labels.shape} vs aggregates {agg_mask.shape}; cannot overlay reliably.")
        return

    voxel_size_xyz = convex_res["voxel_size_xyz"]
    plot_hulls_and_aggregates(convex_res["hulls"], voxel_size_xyz, agg_mask, nuclei_labels > 0, z_compression=0.1)


if __name__ == "__main__":
    main()
