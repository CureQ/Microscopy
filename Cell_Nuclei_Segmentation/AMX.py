import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# =========================
# Pipeline functies (komen uit jouw bestaande module)
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

from scipy.spatial import ConvexHull


# =========================
# Convex hull berekening (lokaal in deze GUI)
# =========================
def compute_convex_hulls_phys_local(labeled_volume, voxel_size_xyz):
    """
    Bereken ConvexHull objecten per label in een gelabeld volume.
    Geeft een lijst van (label, ConvexHull) tuples terug.
    """
    vx, vy, vz = voxel_size_xyz
    hulls = []

    labels = np.unique(labeled_volume)
    for label_id in labels:
        if label_id == 0:
            continue  # achtergrond

        z_idx, y_idx, x_idx = np.where(labeled_volume == label_id)
        if len(z_idx) < 4:
            continue

        pts = np.column_stack([
            x_idx * vx,
            y_idx * vy,
            z_idx * vz
        ])

        try:
            hull = ConvexHull(pts)
        except Exception:
            continue

        hulls.append((int(label_id), hull))

    return hulls


# =========================
# Aggregaten helpers
# =========================
def get_channel_volume(img, channel_name):
    """Laad kanaalvolume als ZYX numpy array."""
    c = img.channel_names.index(channel_name)
    return img.get_image_data("ZYX", C=c, T=0)


def simple_aggregate_mask(volume_zyx, percentile=99.5):
    """
    Simpele aggregate-mask: voxels boven een hoge percentieldrempel.
    Werkt vaak goed als aggregates bright puncta zijn.
    """
    thr = np.percentile(volume_zyx, percentile)
    return volume_zyx > thr


def labels_overlapping_mask(final_labels, mask, min_fraction=0.05):
    """
    Bepaal welke label-IDs overlap hebben met mask.
    min_fraction = overlap_voxels / totale_label_voxels
    """
    if final_labels.shape != mask.shape:
        raise ValueError("Mask en final_labels hebben verschillende vorm")

    flat = final_labels.ravel()
    mflat = mask.ravel()

    counts = np.bincount(flat)
    if counts.size == 0:
        return set()
    counts[0] = 0

    overlap = np.bincount(flat[mflat])
    if overlap.shape[0] < counts.shape[0]:
        overlap = np.pad(overlap, (0, counts.shape[0] - overlap.shape[0]))

    frac = np.zeros_like(counts, dtype=float)
    nonzero = counts > 0
    frac[nonzero] = overlap[nonzero] / counts[nonzero]

    positive = np.where(frac >= min_fraction)[0]
    positive = positive[positive != 0]
    return set(map(int, positive))


def filter_ellipses_by_labels(ellipses_dense, keep_labels):
    """
    Filter ellipses_dense op label-ids.
    Ondersteunt dict en list-of-tuples.
    Als formaat onbekend is, geef ongewijzigd terug.
    """
    if keep_labels is None:
        return ellipses_dense

    if isinstance(ellipses_dense, dict):
        return {k: v for k, v in ellipses_dense.items() if int(k) in keep_labels}

    if isinstance(ellipses_dense, list) and len(ellipses_dense) > 0:
        first = ellipses_dense[0]
        if isinstance(first, tuple) and len(first) == 2:
            return [(int(k), v) for (k, v) in ellipses_dense if int(k) in keep_labels]

    return ellipses_dense


def safe_float(var, default=1.0):
    try:
        return float(var.get())
    except Exception:
        return default


# =========================
# Globale data
# =========================
img = None
final_labels = None
voxel_size_xyz = None
ellipses_dense = None
preview_canvas = None

aggregate_positive_labels = None
aggregate_mask = None


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

# Z-compressie: editable (typen) + presets
zscale_var = tk.StringVar(value="0.1")

# Aggregaat filter
aggregate_channel_var = tk.StringVar(value="None")
agg_overlap_var = tk.StringVar(value="0.05")   # 0-1
agg_percentile_var = tk.StringVar(value="99.5")  # drempel voor mask

dropdown_menus = []


# =========================
# Layout
# =========================
left = tk.Frame(root, width=320)
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


def update_dropdowns(channel_names):
    for menu in dropdown_menus:
        menu["menu"].delete(0, "end")
        for ch in ["None"] + list(channel_names):
            menu["menu"].add_command(
                label=ch,
                command=lambda v=menu.var, c=ch: v.set(c)
            )


def dropdown(parent, label, var):
    tk.Label(parent, text=label).pack(anchor="w")
    menu = tk.OptionMenu(parent, var, "")
    menu.pack(fill="x")
    menu.var = var
    dropdown_menus.append(menu)
    return menu


# =========================
# Load + preview
# =========================
def load_lif_and_preview():
    global img

    img = select_and_load_lif()
    if img is None:
        return

    clear_preview()

    channel_names = img.channel_names
    if not channel_names:
        messagebox.showerror("Fout", "Geen kanalen gevonden in LIF")
        return

    nucleus_var.set(channel_names[0])
    a11_var.set("None")
    ha_var.set("None")
    cct1_var.set("None")
    aggregate_channel_var.set("None")

    update_dropdowns(channel_names)

    # Preview: max projection van C=0 (of je kunt nucleus kanaal kiezen, maar dit is simpel)
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
# RUN ANALYSE (geen plots)
# =========================
def run_full_analysis():
    global final_labels, voxel_size_xyz, ellipses_dense
    global aggregate_positive_labels, aggregate_mask

    if img is None:
        messagebox.showerror("Fout", "Geen LIF bestand geladen")
        return

    nucleus_channel = nucleus_var.get()
    if not nucleus_channel or nucleus_channel == "None":
        messagebox.showerror("Fout", "Selecteer nucleus kanaal")
        return

    # Segmentatie
    volume = nucleus_volumes(
        img,
        nucleus_channel,
        img.channel_names.index(nucleus_channel)
    )
    final_labels = segment_and_merge(volume)

    voxel_size_xyz = get_voxel_size_from_img(img)
    vx, vy, vz = voxel_size_xyz

    # Ellipsen
    ellipses = extract_ellipses_from_labels(final_labels, voxel_size_xy=(vx, vy))
    ellipses_dense = densify_ellipses_by_interpolation(ellipses, Z=final_labels.shape[0])

    # Aggregaten filter voorbereiden (optioneel)
    agg_ch = aggregate_channel_var.get()
    if agg_ch and agg_ch != "None":
        try:
            agg_vol = get_channel_volume(img, agg_ch)

            p = safe_float(agg_percentile_var, default=99.5)
            if p < 50:
                p = 50.0
            if p > 99.99:
                p = 99.99

            aggregate_mask = simple_aggregate_mask(agg_vol, percentile=p)

            min_frac = safe_float(agg_overlap_var, default=0.05)
            if min_frac < 0:
                min_frac = 0.0
            if min_frac > 1:
                min_frac = 1.0

            aggregate_positive_labels = labels_overlapping_mask(
                final_labels, aggregate_mask, min_fraction=min_frac
            )
        except Exception as e:
            aggregate_positive_labels = None
            aggregate_mask = None
            messagebox.showwarning("Waarschuwing", f"Aggregaten filter niet gezet:\n{e}")
    else:
        aggregate_positive_labels = None
        aggregate_mask = None

    msg = "Analyse voltooid."
    if aggregate_positive_labels is not None:
        msg += f"\nAggregate-positieve nuclei: {len(aggregate_positive_labels)}"
    messagebox.showinfo("Analyse", msg)


# =========================
# Visualisaties
# =========================
def show_convex_hulls():
    if final_labels is None:
        messagebox.showerror("Fout", "Run eerst analyse")
        return
    if voxel_size_xyz is None:
        messagebox.showerror("Fout", "Voxelgrootte niet beschikbaar")
        return

    zscale = safe_float(zscale_var, default=1.0)
    if zscale <= 0:
        zscale = 1.0
        messagebox.showwarning("Waarschuwing", "Z-schaal moet > 0 zijn, gebruik 1.0")

    try:
        # Belangrijk: géén vz*zscale hier, alleen in de plot compressie
        hulls_tuples = compute_convex_hulls_phys_local(final_labels, voxel_size_xyz)

        if not hulls_tuples:
            messagebox.showwarning("Waarschuwing", "Geen convex hulls gevonden")
            return

        if aggregate_positive_labels is not None:
            hulls_tuples = [(lab_id, h) for (lab_id, h) in hulls_tuples if lab_id in aggregate_positive_labels]

        if not hulls_tuples:
            messagebox.showwarning("Waarschuwing", "Geen objecten over na aggregaten-filter")
            return

        plot_convex_hulls_phys(
            final_labels,
            hulls_tuples,
            voxel_size_xyz,
            z_compression_factor=zscale
        )
    except Exception as e:
        messagebox.showerror("Fout", f"Convex hull plotting mislukt:\n{e}")


def show_contours():
    if ellipses_dense is None:
        messagebox.showerror("Fout", "Run eerst analyse")
        return
    if voxel_size_xyz is None:
        messagebox.showerror("Fout", "Voxelgrootte niet beschikbaar")
        return

    zscale = safe_float(zscale_var, default=1.0)
    if zscale <= 0:
        zscale = 1.0

    data = ellipses_dense
    if aggregate_positive_labels is not None:
        data = filter_ellipses_by_labels(data, aggregate_positive_labels)
        if isinstance(data, dict) and len(data) == 0:
            messagebox.showwarning("Waarschuwing", "Geen ellipsen over na aggregaten-filter")
            return
        if isinstance(data, list) and len(data) == 0:
            messagebox.showwarning("Waarschuwing", "Geen ellipsen over na aggregaten-filter")
            return

    try:
        plot_ellipses_3d(
            data,
            voxel_size_xyz=voxel_size_xyz,
            z_compression_factor=zscale
        )
    except Exception as e:
        messagebox.showerror("Fout", f"Contour plotting mislukt:\n{e}")


# =========================
# LEFT PANEL
# =========================
tk.Button(left, text="1) Load LIF + preview", command=load_lif_and_preview).pack(fill="x", pady=5)

settings = tk.LabelFrame(left, text="Instellingen")
settings.pack(fill="x", pady=10)

dropdown(settings, "Nucleus", nucleus_var)
dropdown(settings, "A11", a11_var)
dropdown(settings, "Huntingtin", ha_var)
dropdown(settings, "CCT1", cct1_var)

# Aggregaat filter UI
agg_frame = tk.LabelFrame(left, text="Aggregaten filter (optioneel)")
agg_frame.pack(fill="x", pady=10)

tk.Label(agg_frame, text="Kanaal").pack(anchor="w")
agg_menu = tk.OptionMenu(agg_frame, aggregate_channel_var, "")
agg_menu.pack(fill="x")
agg_menu.var = aggregate_channel_var
dropdown_menus.append(agg_menu)

tk.Label(agg_frame, text="Min overlap fractie (0-1)").pack(anchor="w", pady=(6, 0))
tk.Entry(agg_frame, textvariable=agg_overlap_var).pack(fill="x")

tk.Label(agg_frame, text="Intensity percentiel (50-99.99)").pack(anchor="w", pady=(6, 0))
tk.Entry(agg_frame, textvariable=agg_percentile_var).pack(fill="x")

# Z-schaal: editable combobox
tk.Label(settings, text="Z-schaal (compressie)").pack(anchor="w", pady=(8, 0))
zscale_combo = ttk.Combobox(
    settings,
    textvariable=zscale_var,
    values=("1.0", "0.5", "0.2", "0.1", "0.05"),
    state="normal"
)
zscale_combo.pack(fill="x")

tk.Button(
    settings,
    text="2) RUN ANALYSE",
    bg="#d0ffd0",
    command=run_full_analysis
).pack(fill="x", pady=10)

viz = tk.LabelFrame(left, text="Visualisaties")
viz.pack(fill="x")

tk.Button(viz, text="Convex Hulls", command=show_convex_hulls).pack(fill="x", pady=3)
tk.Button(viz, text="Contouren / Ellipsen", command=show_contours).pack(fill="x", pady=3)

root.mainloop()
