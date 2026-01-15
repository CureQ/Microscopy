import tkinter as tk
from tkinter import messagebox
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

# =========================
# Pipeline functies
# =========================
from Convex_Ellips_Comparing import (
    select_and_load_lif,
    nucleus_volumes,
    segment_and_merge,
    get_voxel_size_from_img,
    extract_ellipses_from_labels,
    densify_ellipses_by_interpolation,
    plot_ellipses_3d,
    plot_convex_hulls_phys
)
from Aggregate_Detection import (
        select_aggregate_channel,
        load_aggregate_volume,
        create_cell_mask,
        segment_aggregates_intensity,
        show_projection_with_mask,
        plot_aggregates_3d)


from scipy.spatial import ConvexHull
import numpy as np

def compute_convex_hulls_phys(labeled_volume, voxel_size_xyz):
    """
    Bereken ConvexHull objecten per label in een gelabeld volume.
    Geeft een lijst van (label, ConvexHull) tuples terug.
    """
    vx, vy, vz = voxel_size_xyz
    hulls = []

    labels = np.unique(labeled_volume)
    for label_id in labels:
        if label_id == 0:
            continue  # 0 = achtergrond

        # Alle voxel indices van dit label
        z_idx, y_idx, x_idx = np.where(labeled_volume == label_id)
        if len(z_idx) < 4:
            # Te weinig punten voor ConvexHull
            continue

        # Zet om naar fysieke coordinaten
        pts = np.column_stack([
            x_idx * vx,
            y_idx * vy,
            z_idx * vz
        ])

        # Maak ConvexHull object
        hull = ConvexHull(pts)

        # Voeg toe als tuple (label, hull)
        hulls.append((label_id, hull))

    return hulls

def _flip_over_x_axis(y_values):
    """Mirror y coordinates to flip visuals over the x-axis."""
    return -y_values





# =========================
# Globale data
# =========================
img = None
final_labels = None
voxel_size_xyz = None
ellipses_dense = None
preview_canvas = None

# =========================
# GUI setup
# =========================
root = tk.Tk()
root.title("3D Nuclei Analysis")
root.geometry("1000x600")

# =========================
# GUI state vars
# =========================
nucleus_var = tk.StringVar()
a11_var = tk.StringVar()
ha_var = tk.StringVar()
cct1_var = tk.StringVar()
cct1_var = tk.StringVar()
zscale_var = tk.StringVar(value="0.1")  # Default Z-scale

dropdown_menus = []

# =========================
# Layout
# =========================
left = tk.Frame(root, width=300)
left.pack(side="left", fill="y", padx=10)

right = tk.Frame(root, bd=2, relief="sunken")
right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

# =========================
# Helpers
# =========================
def clear_preview():
    global preview_canvas
    if preview_canvas:
        preview_canvas.get_tk_widget().destroy()
        preview_canvas = None

# =========================
# Load + preview
# =========================
def load_lif_and_preview():
    global img
    img = select_and_load_lif()
    if img is None:
        return

    clear_preview()

    print("img inhoud", img.channel_names)
    channel_names = img.channel_names
    nucleus_var.set(channel_names[0])
    a11_var.set("None")
    ha_var.set(channel_names[2])
    cct1_var.set("None")

    update_dropdowns(channel_names)

    vol = img.get_image_data("ZYX", C=0, T=0)
    max_proj = np.max(vol, axis=0)

    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.imshow(max_proj, cmap="gray")
    ax.set_title("Max projection")
    ax.axis("off")

    global preview_canvas
    preview_canvas = FigureCanvasTkAgg(fig, master=right)
    preview_canvas.draw()
    preview_canvas.get_tk_widget().pack(fill="both", expand=True)

# =========================
# Dropdown update
# =========================
def update_dropdowns(channel_names):
    for menu in dropdown_menus:
        menu["menu"].delete(0, "end")
        for ch in ["None"] + channel_names:
            menu["menu"].add_command(
                label=ch,
                command=lambda v=menu.var, c=ch: v.set(c)
            )

# =========================
# RUN ANALYSE (GEEN PLOTS)
# =========================
def run_full_analysis():
    global final_labels, voxel_size_xyz, ellipses_dense

    if img is None:
        messagebox.showerror("Fout", "Geen LIF bestand geladen")
        return

    nucleus_channel = nucleus_var.get()
    if nucleus_channel == "None":
        messagebox.showerror("Fout", "Selecteer nucleus kanaal")
        return

    # Segmentatie
    volume = nucleus_volumes(
        img,
        nucleus_channel,
        img.channel_names.index(nucleus_channel)
    )
    final_labels = segment_and_merge(volume, z_compression_factor=float(zscale_var.get()))

    voxel_size_xyz = get_voxel_size_from_img(img)

    # Ellipsen extraheren
    vx, vy, vz = voxel_size_xyz
    ellipses = extract_ellipses_from_labels(final_labels, voxel_size_xy=(vx, vy))
    ellipses_dense = densify_ellipses_by_interpolation(ellipses, Z=final_labels.shape[0])

    messagebox.showinfo("Analyse", "Analyse voltooid (plots pas bij knoppen)")

# =========================
# Visualisaties
# =========================
def show_convex_hulls():
    """
    Robust plotting van convex hulls, ongeacht output van compute_convex_hulls_phys.
    """
    if final_labels is None:
        messagebox.showerror("Fout", "Run eerst analyse")
        return

    if voxel_size_xyz is None:
        messagebox.showerror("Fout", "Voxelgrootte niet beschikbaar")
        return

    try:
        zscale = float(zscale_var.get())
    except ValueError:
        zscale = 1.0
        messagebox.showwarning("Waarschuwing", "Ongeldige Z-schaal, gebruik 1.0")

    vx, vy, vz = voxel_size_xyz
    voxel_size_scaled = (vx, vy, vz * zscale)

    try:
        # Bereken hulls
        hulls = compute_convex_hulls_phys(final_labels, voxel_size_scaled)

        if not hulls:
            messagebox.showwarning("Waarschuwing", "Geen convex hulls gevonden")
            return

        # Zorg dat we altijd (label,hull) tuples hebben
        if isinstance(hulls[0], tuple) and len(hulls[0]) == 2:
            hulls_tuples = hulls
        else:
            hulls_tuples = [(i+1, h) for i, h in enumerate(hulls)]
        print("Zscale var", zscale_var)
        # Plot hulls
        plot_convex_hulls_phys(
            final_labels,
            hulls_tuples,
            voxel_size_scaled,
            z_compression_factor=float(zscale_var.get())
        )

    except Exception as e:
        messagebox.showerror("Fout", f"Convex hull plotting mislukt:\n{e}")

def show_contours():
    if ellipses_dense is None:
        messagebox.showerror("Fout", "Run eerst analyse")
        return

    try:
        zscale = float(zscale_var.get())
    except ValueError:
        zscale = 1.0

    plot_ellipses_3d(
        ellipses_dense,
        voxel_size_xyz=voxel_size_xyz,
        z_compression_factor=float(zscale_var.get())
    )

def segment_aggregates():
    if img is None:
        messagebox.showerror("Fout", "Geen LIF bestand geladen")
        return

    print(ha_var.get())
    aggregate_channel = select_aggregate_channel(img, aggregate_name=ha_var.get())
    if aggregate_channel is None:
        return

    # Hier zou de segmentatiecode voor aggregaten komen
    messagebox.showinfo("Aggregates", f"Aggregate kanaal geselecteerd: {aggregate_channel}")

    volume = load_aggregate_volume(img, aggregate_channel)
    cell_mask = create_cell_mask(volume, min_cell_size=20)
    filtered_input = volume * cell_mask
    aggregates_mask, thr = segment_aggregates_intensity(
        filtered_input,
        cell_mask,
        perc=99.9,
        min_size=100,
        max_size=50000,
    )

    show_projection_with_mask(volume, aggregates_mask)
    plot_aggregates_3d(aggregates_mask, z_scale=30)


# =========================
# LEFT PANEL
# =========================
tk.Button(left, text="1️⃣ Load LIF + preview", command=load_lif_and_preview)\
    .pack(fill="x", pady=5)

settings = tk.LabelFrame(left, text="Instellingen")
settings.pack(fill="x", pady=10)

def dropdown(parent, label, var):
    tk.Label(parent, text=label).pack(anchor="w")
    menu = tk.OptionMenu(parent, var, "")
    menu.pack(fill="x")
    menu.var = var
    dropdown_menus.append(menu)

dropdown(settings, "Nucleus", nucleus_var)
dropdown(settings, "A11", a11_var)
dropdown(settings, "Huntingtin", ha_var)
dropdown(settings, "CCT1", cct1_var)

# Z-SCALE DROPDOWN
tk.Label(settings, text="Z-schaal (compressie)").pack(anchor="w", pady=(8, 0))
tk.OptionMenu(
    settings,
    zscale_var,
    "1.0", "0.5", "0.2", "0.1", "0.05"
).pack(fill="x")

tk.Button(
    settings,
    text="2️⃣ RUN ANALYSE",
    bg="#d0ffd0",
    command=run_full_analysis
).pack(fill="x", pady=10)

viz = tk.LabelFrame(left, text="Visualisaties")
viz.pack(fill="x")

tk.Button(viz, text="Convex Hulls", command=show_convex_hulls)\
    .pack(fill="x", pady=3)

tk.Button(viz, text="Contouren / Ellipsen", command=show_contours)\
    .pack(fill="x", pady=3)


aggregate_frame = tk.LabelFrame(left, text="Aggregates")
aggregate_frame.pack(fill="x")

tk.Button(aggregate_frame, text="Segment aggregates", command=segment_aggregates)\
    .pack(fill="x", pady=3)

tk.Button(aggregate_frame, text="Localize aggregates", command=segment_aggregates)\
    .pack(fill="x", pady=3)

root.mainloop()
