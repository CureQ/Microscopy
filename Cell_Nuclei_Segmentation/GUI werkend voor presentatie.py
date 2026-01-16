"""
3D Nuclei + Aggregates GUI (Tkinter)

GOAL
- Load a Leica .lif microscopy file (AICSImageIO via your helper)
- Run nuclei segmentation and derive per-slice ellipses (contour representation)
- Run aggregate segmentation in a chosen channel
- Provide multiple 3D visualizations:
  1) Nuclei convex hulls (point cloud + hull surface triangles)
  2) Nuclei ellipse contours (your original plot_ellipses_3d)
  3) Aggregates vs nuclei convex hulls (inside/outside using nuclei labels)
  4) Aggregates + contours (filled ellipses + aggregate coloring using contour geometry)
  5) Aggregates vs contours (lines) (aggregate coloring using contour geometry)

NOTES
- This script assumes the following modules exist in your project:
  - Convex_Ellips_Comparing.py
  - Aggregate_Detection.py
- The GUI is intentionally kept simple (LabelFrames + Buttons) so it can be reused in other GUIs.
"""

# =========================
# Standard library / GUI
# =========================
import tkinter as tk                     # Tkinter GUI framework
from tkinter import messagebox           # Tkinter message dialogs (error/info/warning)

# =========================
# Numeric + plotting
# =========================
import numpy as np                       # Fast array operations
import matplotlib.pyplot as plt          # Matplotlib plotting (standalone windows)

# Embed Matplotlib figures inside Tkinter (for the preview panel)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# 3D polygon support (for convex hull faces + filled ellipse surfaces)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 2D point-in-polygon test (to classify aggregates inside/outside contours)
from matplotlib.path import Path

# Convex hull computation in 3D
from scipy.spatial import ConvexHull


# =========================
# Pipeline imports (your code)
# =========================
from Convex_Ellips_Comparing import (
    select_and_load_lif,                 # your file picker + AICSImage loader
    nucleus_volumes,                     # load nucleus channel into volume
    segment_and_merge,                   # nuclei segmentation -> labeled volume
    get_voxel_size_from_img,             # read voxel spacing from image metadata
    extract_ellipses_from_labels,        # fit ellipse per label per z-slice
    densify_ellipses_by_interpolation,   # densify ellipse representation across z
    plot_ellipses_3d,                    # your existing ellipse plotting function
    plot_convex_hulls_phys,              # your existing hull plotting function (not used for the button here)
)

from Aggregate_Detection import (
    load_aggregate_volume,               # load aggregate channel volume
    create_cell_mask,                    # build cell mask from intensity projection
    segment_aggregates_intensity,        # percentile-based thresholding + size filtering
    show_projection_with_mask,           # 2D diagnostic overlay
    plot_aggregates_3d,                  # 3D diagnostic scatter
)


# =========================
# Global state (shared between button callbacks)
# =========================
img = None                               # AICSImage object after loading a .lif
final_labels = None                      # nuclei segmentation result (Z,Y,X), integers (0=background)
voxel_size_xyz = None                    # (vx, vy, vz) in micrometers, from metadata

ellipses_raw = None                      # list of ellipse dictionaries from extract_ellipses_from_labels
ellipses_dense = None                    # densified ellipse data for the original plot_ellipses_3d button
contour_cache_by_z = None                # dict: z -> list of Nx2 arrays in pixel coordinates (x,y)

aggregates_mask = None                   # boolean mask (Z,Y,X) for segmented aggregates
preview_canvas = None                    # Tkinter widget that holds the Matplotlib preview figure


# =========================
# GUI window setup
# =========================
root = tk.Tk()                           # Create the main application window
root.title("3D Nuclei Analysis")         # Window title
root.geometry("1000x700")                # Window size in pixels (width x height)


# =========================
# Tkinter "state variables"
# (These are bound to dropdowns and can be read with .get())
# =========================
nucleus_var = tk.StringVar()             # selected nucleus channel name
a11_var = tk.StringVar()                 # placeholder (not used in callbacks currently)
ha_var = tk.StringVar()                  # selected aggregate channel name (e.g., Huntingtin)
cct1_var = tk.StringVar()                # placeholder (not used in callbacks currently)
zscale_var = tk.StringVar(value="0.1")   # Z scaling factor used for 3D visualization

dropdowns = []                           # list of Tk OptionMenu widgets (so we can update menu items after load)


# =========================
# GUI layout containers
# =========================
left = tk.Frame(root, width=340)         # left control panel
left.pack(side="left", fill="y", padx=10)

right = tk.Frame(root, bd=2, relief="sunken")  # right panel to show preview image
right.pack(side="right", fill="both", expand=True, padx=10, pady=10)


# =========================
# Utility functions
# =========================
def clear_preview():
    """
    Remove the current preview figure from the right-hand GUI panel.
    """
    global preview_canvas
    if preview_canvas:
        preview_canvas.get_tk_widget().destroy()
        preview_canvas = None


def zscale():
    """
    Read the Z-scale dropdown and return it as a float.
    Falls back to 0.1 when parsing fails.
    """
    try:
        return float(zscale_var.get())
    except Exception:
        return 0.1


def mask_to_phys(mask):
    """
    Convert a boolean voxel mask in (Z,Y,X) into physical coordinates (X,Y,Z).

    Why:
    - Matplotlib 3D needs X/Y/Z vectors
    - We want micrometer units instead of pixel indices
    - We apply the same Y-axis flip convention used elsewhere in your plots
      so the orientation is consistent.

    Inputs
    - mask: boolean array of shape (Z,Y,X)

    Outputs
    - X, Y, Z: 1D arrays (same length) in micrometers,
      where Z is multiplied by the GUI zscale() factor
    """
    vx, vy, vz = voxel_size_xyz           # micrometers per pixel (x,y) and per slice (z)
    z, y, x = np.nonzero(mask)            # indices of True voxels
    X = x * vx                            # convert x indices to micrometers
    Y = -y * vy                           # flip Y axis, then convert to micrometers
    Z = z * vz * zscale()                 # convert z indices to micrometers and apply compression
    return X, Y, Z


def get_polys_for_z(z, cache):
    """
    Convenience accessor for the contour cache:
    returns a list of 2D polygons for a given z-slice.
    """
    if cache is None:
        return []
    return cache.get(int(z), [])


# =========================
# Convex hull visualization (button output)
# =========================
def plot_convex_hulls_button_style():
    """
    Plot nuclei segmentation as:
    - a colored point cloud (label IDs),
    - plus convex hull surface triangles for each nucleus.

    This function is intentionally explicit and self-contained so
    the output is stable (visible hulls + readable points), regardless
    of how plot_convex_hulls_phys behaves internally.

    Preconditions
    - final_labels must exist (run nuclei analysis first)
    - voxel_size_xyz must exist (computed during nuclei analysis)
    """
    if final_labels is None or voxel_size_xyz is None:
        messagebox.showerror("Error", "Run nuclei analysis first")
        return

    vx, vy, vz = voxel_size_xyz            # voxel spacing (um)
    zs = zscale()                          # Z compression/scale factor (for display)

    # Extract all non-background labeled voxels to show the point cloud
    z_idx, y_idx, x_idx = np.nonzero(final_labels)
    if len(z_idx) == 0:
        messagebox.showwarning("Warning", "No labeled voxels to plot.")
        return

    # Convert voxel indices to physical coordinates
    X = x_idx * vx
    Y = -y_idx * vy
    Z = z_idx * vz * zs

    # Create a new 3D figure
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the point cloud with colors by nucleus label
    ax.scatter(
        X, Y, Z,
        s=0.05,                             # very small points for dense clouds
        c=final_labels[z_idx, y_idx, x_idx],# label values used as color source
        cmap="tab10",
        alpha=0.10,                         # transparency to avoid saturation
    )

    # Compute convex hulls for each label and plot their faces
    hulls = []
    for lbl in np.unique(final_labels):
        if lbl == 0:
            continue                        # skip background
        z, y, x = np.where(final_labels == lbl)
        if len(z) < 4:
            continue                        # need >=4 non-coplanar points for a 3D hull
        pts = np.column_stack([
            x * vx,
            -y * vy,
            z * vz * zs
        ])
        try:
            hulls.append(ConvexHull(pts))
        except Exception:
            # ConvexHull can fail if points are degenerate
            continue

    if not hulls:
        messagebox.showwarning("Warning", "No convex hulls found.")
        return

    # Add hull triangles as translucent polygons
    for h in hulls:
        for tri in h.simplices:
            poly = Poly3DCollection(
                [h.points[tri]],
                alpha=0.25,                 # face visibility
                edgecolor="k",              # black edges
                linewidths=0.2
            )
            ax.add_collection3d(poly)

    # Axis labels and aesthetics
    ax.set_title("3D convex hulls (compressed Z)")
    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um, flipped)")
    ax.set_zlabel(f"Z (um x{zs})")

    # Equal-ish aspect ratio with Z compressed
    ax.set_box_aspect((1, 1, max(0.02, zs)))

    plt.tight_layout()
    plt.show()


# =========================
# Button callback: Load .lif + preview
# =========================
def load_lif_and_preview():
    """
    Load a LIF image using your file picker and show a quick preview:
    - max intensity projection of channel C=0 over Z
    - displayed inside the GUI right panel
    """
    global img

    # Open file picker and load image
    img = select_and_load_lif()
    if img is None:
        return

    # Clear old preview image from the GUI
    clear_preview()

    # Read channel names from the image
    ch = list(img.channel_names)

    # Set default dropdown selections
    nucleus_var.set(ch[0] if ch else "None")
    a11_var.set("None")
    ha_var.set(ch[2] if len(ch) > 2 else "None")
    cct1_var.set("None")

    # Update dropdown menus with the actual channel names
    for d in dropdowns:
        d["menu"].delete(0, "end")
        for c in ["None"] + ch:
            d["menu"].add_command(
                label=c,
                command=lambda v=d.var, x=c: v.set(x)
            )

    # Load a volume for preview using the first channel (C=0), timepoint 0 (T=0)
    vol = img.get_image_data("ZYX", C=0, T=0)

    # Compute a max projection across Z for preview
    max_proj = np.max(vol, axis=0)

    # Create a Matplotlib Figure to embed in Tkinter
    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.imshow(max_proj, cmap="gray")
    ax.set_title("Max projection (C=0)")
    ax.axis("off")

    # Embed the figure into the Tkinter right panel
    global preview_canvas
    preview_canvas = FigureCanvasTkAgg(fig, master=right)
    preview_canvas.draw()
    preview_canvas.get_tk_widget().pack(fill="both", expand=True)


# =========================
# Button callback: Run nuclei analysis
# =========================
def run_analysis():
    """
    Run the nuclei analysis pipeline:
    - load nucleus channel volume
    - run segmentation -> labeled volume (final_labels)
    - read voxel size from metadata (voxel_size_xyz)
    - extract ellipses per label/slice (ellipses_raw)
    - densify ellipse representation for legacy plotting (ellipses_dense)

    Results are stored in global variables used by plotting buttons.
    """
    global final_labels, voxel_size_xyz, ellipses_raw, ellipses_dense, contour_cache_by_z

    if img is None:
        messagebox.showerror("Error", "Load a LIF first")
        return

    nuc = nucleus_var.get()
    if nuc == "None":
        messagebox.showerror("Error", "Select nucleus channel")
        return

    # Load the nucleus volume (Z,Y,X)
    volume = nucleus_volumes(img, nuc, img.channel_names.index(nuc))

    # Segment nuclei into a labeled mask (0=background, 1..N labels)
    final_labels = segment_and_merge(volume, z_compression_factor=zscale())

    # Read voxel spacing from the image metadata
    voxel_size_xyz = get_voxel_size_from_img(img)

    # Extract ellipse parameters from labels
    vx, vy, vz = voxel_size_xyz
    ellipses_raw = extract_ellipses_from_labels(final_labels, voxel_size_xy=(vx, vy))

    # Densify for your original contour plotting function
    ellipses_dense = densify_ellipses_by_interpolation(ellipses_raw, Z=final_labels.shape[0])

    # Reset contour cache (will be created on demand for contour-based overlay plots)
    contour_cache_by_z = None

    # Debug/feedback
    if not ellipses_raw:
        messagebox.showwarning("Warning", "No ellipses extracted from labels.")
    else:
        print("ellipses_raw count:", len(ellipses_raw))

    messagebox.showinfo("Done", "Nuclei analysis finished")


# =========================
# Button callback: Segment aggregates
# =========================
def segment_aggregates():
    """
    Segment aggregates from the selected aggregate channel using your intensity-based pipeline:
    - Load aggregate volume
    - Build a cell mask (search region)
    - Threshold by percentile within the mask + size filtering

    The resulting aggregates_mask is stored globally.
    """
    global aggregates_mask

    if img is None:
        messagebox.showerror("Error", "Load a LIF first")
        return

    agg = ha_var.get()
    if agg == "None":
        messagebox.showerror("Error", "Select aggregate channel")
        return

    # Find the channel index by its name
    idx = img.channel_names.index(agg)

    # Load the aggregate volume (Z,Y,X)
    vol = load_aggregate_volume(img, idx)

    # Create a broad cell mask (restrict where aggregates may exist)
    cell_mask = create_cell_mask(vol, min_cell_size=20)

    # Segment aggregates based on intensity percentile inside the cell mask
    aggregates_mask, thr = segment_aggregates_intensity(
        vol * cell_mask,                   # restrict intensity to cell region
        cell_mask,
        perc=99.5,
        min_size=10,
        max_size=50000,
    )

    # Print some diagnostics
    n = np.count_nonzero(aggregates_mask)
    print("Aggregate voxels:", n, "threshold:", thr)

    if n == 0:
        messagebox.showwarning("Warning", "No aggregates found")
        return

    # Optional diagnostics plots from your aggregate module
    show_projection_with_mask(vol, aggregates_mask)
    plot_aggregates_3d(aggregates_mask, z_scale=30)


# =========================
# Aggregates vs nuclei convex hulls (reference-style)
# =========================
def overlay_vs_hulls():
    """
    Visualize aggregates classified as inside/outside nuclei using the nuclei label mask:
    - inside: aggregates where final_labels > 0
    - outside: aggregates where final_labels == 0

    Then show convex hull surfaces for each nucleus.

    Preconditions
    - final_labels exists (run nuclei analysis)
    - aggregates_mask exists (run aggregate segmentation)
    """
    if aggregates_mask is None or final_labels is None or voxel_size_xyz is None:
        messagebox.showerror("Error", "Run nuclei + aggregate segmentation first")
        return

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Classify aggregates using label membership
    inside = aggregates_mask & (final_labels > 0)
    outside = aggregates_mask & ~(final_labels > 0)

    # Plot aggregate voxels as points
    ax.scatter(*mask_to_phys(inside), s=4, c="green", alpha=1.0, label="inside nucleus")
    ax.scatter(*mask_to_phys(outside), s=4, c="red", alpha=1.0, label="outside nucleus")

    # Compute and plot convex hulls per nucleus label
    hulls = []
    vx, vy, vz = voxel_size_xyz
    for lbl in np.unique(final_labels):
        if lbl == 0:
            continue
        z, y, x = np.where(final_labels == lbl)
        if len(z) < 4:
            continue
        pts = np.column_stack([x * vx, -y * vy, z * vz * zscale()])
        hulls.append(ConvexHull(pts))

    for h in hulls:
        for tri in h.simplices:
            poly = Poly3DCollection([h.points[tri]], alpha=0.07)
            ax.add_collection3d(poly)

    ax.set_title("Aggregates vs Nuclei (Convex hulls)")
    ax.legend()
    plt.show()


# =========================
# Custom ellipse plotting that also caches per-slice polygons
# =========================
def plot_ellipses_3d_2(
    ellipses,
    voxel_size_xyz=None,
    z_compression_factor=0.1,
    n_ellipse_points=160,
    ellipse_rotation_offset=0.0,
    fill_alpha=0.16,
    line_alpha=0.85,
    linewidth=1.1,
):
    """
    Draw filled ellipses (one per ellipse dict) and build a cache of 2D polygons per Z slice.

    Why:
    - We want a plot where ellipses are visible as surfaces (filled)
    - We want to reuse the polygon coordinates for point-in-polygon classification of aggregates

    Inputs
    - ellipses: list of dicts with at least keys: label, z, cx/cy/a/b/phi (+ optional *_um)
    - voxel_size_xyz: (vx, vy, vz) in micrometers
    - z_compression_factor: visual scaling for Z
    - n_ellipse_points: number of points per ellipse ring
    - ellipse_rotation_offset: optional extra rotation in radians
    - fill_alpha: fill transparency for ellipse surfaces
    - line_alpha: transparency for ellipse outlines
    - linewidth: width of ellipse outlines

    Outputs
    - fig, ax: Matplotlib objects (so we can add aggregates after)
    - cache: dict mapping z -> list of polygon point arrays in pixel coords (Nx2)
    """
    if not ellipses:
        print("No ellipses to plot.")
        return None, None, {}

    vx, vy, vz = (1.0, 1.0, 1.0) if voxel_size_xyz is None else voxel_size_xyz

    # Map nucleus label -> color for consistent coloring across ellipses
    labels = sorted(set(e["label"] for e in ellipses))
    cmap = plt.get_cmap("tab10")
    label_to_color = {lbl: cmap(i % 10) for i, lbl in enumerate(labels)}

    # Parameterization of ellipse boundary
    t = np.linspace(0, 2 * np.pi, n_ellipse_points)

    # Create plot
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    cache = {}  # z -> list of Nx2 polygons (pixel space)

    for e in ellipses:
        # Ellipse center and radii in micrometers (if provided) else convert from pixels
        cx_um = e.get("cx_um", e["cx"] * vx)
        cy_um = e.get("cy_um", e["cy"] * vy)
        a_um = e.get("a_um", e["a"] * vx)
        b_um = e.get("b_um", e["b"] * vy)

        # Rotation and slice position
        phi = e["phi"] + ellipse_rotation_offset
        z_idx = int(e["z"])
        lbl = e["label"]
        color = label_to_color[lbl]

        # Parametric ellipse in 2D (micrometers)
        xs_um = cx_um + a_um * np.cos(t) * np.cos(phi) - b_um * np.sin(t) * np.sin(phi)
        ys_um = cy_um + a_um * np.cos(t) * np.sin(phi) + b_um * np.cos(phi) * np.sin(t)

        # Z position of this ellipse in the 3D plot
        zs_um = np.full_like(xs_um, z_idx * vz * z_compression_factor, dtype=float)

        # Apply Y flip for the 3D plot, to match your convention
        ys_um_flipped = -ys_um

        # Convert ellipse boundary back to pixel coordinates for point-in-polygon checks
        xs_pix = xs_um / vx
        ys_pix = ys_um / vy
        pts2d = np.column_stack([xs_pix, ys_pix])

        # Store polygon points for this z-slice
        cache.setdefault(z_idx, []).append(pts2d)

        # Build a filled polygon surface in 3D
        poly3d = np.column_stack([xs_um, ys_um_flipped, zs_um])
        poly = Poly3DCollection([poly3d], alpha=fill_alpha)
        poly.set_facecolor(color)
        poly.set_edgecolor("none")
        ax.add_collection3d(poly)

        # Draw the ellipse outline
        ax.plot(xs_um, ys_um_flipped, zs_um, color=color, alpha=line_alpha, linewidth=linewidth)

    # Label axes
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm, flipped)")
    ax.set_zlabel(f"z (µm x{z_compression_factor:.2f})")
    ax.set_title("3D ellipse contours (filled)")

    # Aspect ratio with compressed Z
    ax.set_box_aspect((1, 1, max(0.02, z_compression_factor)))
    plt.tight_layout()

    return fig, ax, cache


# =========================
# Aggregates + contours (filled) using contour-based classification
# =========================
def aggregates_plus_contours_filled():
    """
    Show a single 3D plot that contains:
    - Filled ellipses (nucleus contours) per z-slice
    - Aggregates plotted in green (inside contours) or red (outside contours)

    IMPORTANT:
    - The inside/outside classification is based on contour polygons (not convex hulls).
    - The contour polygons come from plot_ellipses_3d_2's cache.
    """
    global contour_cache_by_z

    if ellipses_raw is None or voxel_size_xyz is None:
        messagebox.showerror("Error", "Run nuclei analysis first")
        return
    if aggregates_mask is None:
        messagebox.showerror("Error", "Run aggregate segmentation first")
        return

    vx, vy, vz = voxel_size_xyz
    zs = zscale()

    # Create the filled ellipse plot and retrieve the per-z polygon cache
    fig, ax, cache = plot_ellipses_3d_2(
        ellipses=ellipses_raw,
        voxel_size_xyz=voxel_size_xyz,
        z_compression_factor=zs,
        n_ellipse_points=160,
        fill_alpha=0.16,
        line_alpha=0.85,
        linewidth=1.1,
    )
    if ax is None:
        return

    contour_cache_by_z = cache

    # Get aggregate voxel coordinates
    z_idx, y_idx, x_idx = np.nonzero(aggregates_mask)
    if len(z_idx) == 0:
        messagebox.showwarning("Warning", "No aggregate voxels to plot.")
        return

    # Determine inside/outside per aggregate voxel
    inside = np.zeros_like(z_idx, dtype=bool)

    # Only evaluate slices that actually contain aggregate voxels
    for z in np.unique(z_idx):
        polys = get_polys_for_z(int(z), contour_cache_by_z)
        if not polys:
            continue

        sel = (z_idx == z)
        pts_xy = np.column_stack([x_idx[sel], y_idx[sel]])

        # If multiple polygons exist in this slice, consider inside if inside ANY polygon
        inside_z = np.zeros(pts_xy.shape[0], dtype=bool)
        for poly_pts in polys:
            inside_z |= Path(poly_pts).contains_points(pts_xy)
        inside[sel] = inside_z

    # Convert aggregate voxels to physical coordinates for plotting
    Xin = x_idx[inside] * vx
    Yin = -(y_idx[inside] * vy)
    Zin = (z_idx[inside] * vz) * zs

    Xout = x_idx[~inside] * vx
    Yout = -(y_idx[~inside] * vy)
    Zout = (z_idx[~inside] * vz) * zs

    # Plot aggregates on top of the filled ellipses
    ax.scatter(Xin, Yin, Zin, s=6, c="green", alpha=0.9, label="Aggregate inside contour")
    ax.scatter(Xout, Yout, Zout, s=6, c="red", alpha=0.9, label="Aggregate outside contour")

    ax.legend()
    plt.show()


# =========================
# Aggregates vs contours (lines) using the cached polygons
# =========================
def overlay_vs_contours_lines():
    """
    Plot aggregates as green/red (inside/outside contours) and draw contours as line loops.

    This plot uses contour_cache_by_z. If it doesn't exist yet, it is built
    by calling plot_ellipses_3d_2 with "invisible" settings (alpha=0) and
    then immediately closing that figure.
    """
    if aggregates_mask is None or voxel_size_xyz is None:
        messagebox.showerror("Error", "Run nuclei + aggregate segmentation first")
        return
    if ellipses_raw is None:
        messagebox.showerror("Error", "Run nuclei analysis first")
        return

    global contour_cache_by_z
    if contour_cache_by_z is None:
        # Build the cache without showing an ellipse plot
        _, _, contour_cache_by_z = plot_ellipses_3d_2(
            ellipses=ellipses_raw,
            voxel_size_xyz=voxel_size_xyz,
            z_compression_factor=zscale(),
            n_ellipse_points=140,
            fill_alpha=0.0,
            line_alpha=0.0,
            linewidth=1.0,
        )
        plt.close("all")

    # Aggregate voxel coordinates
    z_idx, y_idx, x_idx = np.nonzero(aggregates_mask)
    inside = np.zeros_like(z_idx, dtype=bool)

    # Point-in-polygon test per z-slice
    for z in np.unique(z_idx):
        polys = get_polys_for_z(int(z), contour_cache_by_z)
        if not polys:
            continue

        sel = (z_idx == z)
        pts_xy = np.column_stack([x_idx[sel], y_idx[sel]])

        inside_z = np.zeros(pts_xy.shape[0], dtype=bool)
        for poly_pts in polys:
            inside_z |= Path(poly_pts).contains_points(pts_xy)
        inside[sel] = inside_z

    vx, vy, vz = voxel_size_xyz
    zs = zscale()

    # Create plot
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Plot aggregates
    ax.scatter(
        x_idx[inside] * vx,
        -y_idx[inside] * vy,
        z_idx[inside] * vz * zs,
        c="green",
        s=4,
        alpha=1.0,
        label="inside contour",
    )
    ax.scatter(
        x_idx[~inside] * vx,
        -y_idx[~inside] * vy,
        z_idx[~inside] * vz * zs,
        c="red",
        s=4,
        alpha=1.0,
        label="outside contour",
    )

    # Plot contour loops as lines
    for z, polys in contour_cache_by_z.items():
        for poly_pts in polys:
            ax.plot(
                poly_pts[:, 0] * vx,
                -poly_pts[:, 1] * vy,
                np.full(poly_pts.shape[0], z * vz * zs, dtype=float),
                linewidth=1.0,
                alpha=0.35,
            )

    ax.set_title("Aggregates vs Nuclei (Contours, lines)")
    ax.legend()
    plt.show()


# =========================
# GUI widgets
# =========================

# Button: load file + preview
tk.Button(
    left,
    text="1) Load LIF + preview",
    command=load_lif_and_preview
).pack(fill="x", pady=5)

# Group: channel dropdowns
settings = tk.LabelFrame(left, text="Channels")
settings.pack(fill="x", pady=10)

def add_dropdown(label_text, variable):
    """
    Create a label + dropdown pair inside the Channels frame.
    The actual menu items are filled after loading the LIF.
    """
    tk.Label(settings, text=label_text).pack(anchor="w")
    menu = tk.OptionMenu(settings, variable, "")
    menu.pack(fill="x")
    menu.var = variable
    dropdowns.append(menu)

add_dropdown("Nucleus", nucleus_var)
add_dropdown("A11", a11_var)
add_dropdown("Huntingtin", ha_var)
add_dropdown("CCT1", cct1_var)

# Z-scale dropdown (visual only: used to compress/scale Z in 3D)
tk.Label(settings, text="Z-scale").pack(anchor="w")
tk.OptionMenu(settings, zscale_var, "1.0", "0.5", "0.2", "0.1", "0.05").pack(fill="x")

# Button: run nuclei pipeline
tk.Button(
    settings,
    text="2) Run nuclei analysis",
    bg="#d0ffd0",
    command=run_analysis
).pack(fill="x", pady=10)

# Group: base visualizations (nuclei only)
viz_basic = tk.LabelFrame(left, text="Visualisation")
viz_basic.pack(fill="x")

# Convex hull plot (stable button output)
tk.Button(
    viz_basic,
    text="Convex hulls",
    command=plot_convex_hulls_button_style
).pack(fill="x", pady=2)

# Original ellipse plot (your existing function)
tk.Button(
    viz_basic,
    text="Contours / ellipses",
    command=lambda: plot_ellipses_3d(
        ellipses_dense,
        voxel_size_xyz=voxel_size_xyz,
        z_compression_factor=zscale()
    )
).pack(fill="x", pady=2)

# Group: aggregate segmentation controls
agg = tk.LabelFrame(left, text="Aggregates")
agg.pack(fill="x", pady=10)

tk.Button(
    agg,
    text="Segment aggregates",
    command=segment_aggregates
).pack(fill="x", pady=2)

# Group: aggregate-focused visualization (placed under Aggregates)
viz_adv = tk.LabelFrame(left, text="Visualisation (Aggregates)")
viz_adv.pack(fill="x")

tk.Button(
    viz_adv,
    text="Aggregates vs Convex hulls",
    command=overlay_vs_hulls
).pack(fill="x", pady=2)

tk.Button(
    viz_adv,
    text="Aggregates + Contours (filled)",
    command=aggregates_plus_contours_filled
).pack(fill="x", pady=2)

tk.Button(
    viz_adv,
    text="Aggregates vs Contours (lines)",
    command=overlay_vs_contours_lines
).pack(fill="x", pady=2)

# Start the Tkinter event loop (blocks until the window is closed)
root.mainloop()
