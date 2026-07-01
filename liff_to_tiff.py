"""
LIF → TIFF Batch & Single Converter (met Hot colormap)
Inclusief fix voor te lange bestandsnamen (Windows 260 char limit).
"""

import sys
import os
import argparse
import glob
import numpy as np

try:
    from readlif.reader import LifFile
except ImportError:
    sys.exit("❌  Installeer readlif eerst:  pip install readlif")

try:
    import tifffile
except ImportError:
    sys.exit("❌  Installeer tifffile eerst:  pip install tifffile")

try:
    import matplotlib
    matplotlib.use("Agg")  # geen GUI nodig
except ImportError:
    sys.exit("❌  Installeer matplotlib eerst:  pip install matplotlib")


# ─────────────────────────────────────────────
# Hulpfuncties
# ─────────────────────────────────────────────

def list_images(lif: LifFile):
    print("\n📂  Series gevonden in LIF-bestand:")
    print(f"  {'#':>3}  {'Naam':<40}  {'Kanalen':>7}  {'Z-lagen':>7}  {'XY (px)':>14}")
    print("  " + "─" * 76)
    for i, img in enumerate(lif.get_iter_image()):
        dims = img.dims
        ch   = img.channels
        name = img.name or f"serie_{i}"
        print(f"  {i:>3}  {name:<40}  {ch:>7}  {dims.z:>7}  {dims.x}×{dims.y}")
    print()

def get_image_data(img, channel: int) -> np.ndarray:
    frames = []
    for z in range(img.dims.z):
        frame = img.get_frame(z=z, t=0, c=channel, m=0)
        frames.append(np.array(frame))
    return np.stack(frames, axis=0)

def max_projection(stack: np.ndarray) -> np.ndarray:
    return stack.max(axis=0)

def apply_hot_colormap(data: np.ndarray) -> np.ndarray:
    d = data.astype(np.float32)
    # Bepaal de absolute maximale helderheid op basis van het datatype
    if data.dtype == np.uint8:
        absolute_max = 255.0     # 8-bit beelden
    elif data.dtype == np.uint16:
        absolute_max = 65535.0   # 16-bit beelden
    else:
        # Veilige terugvaloptie mocht het format afwijken
        absolute_max = d.max() if d.max() > 0 else 1.0
 
    # Bereken de helderheid ten opzichte van het absolute maximum
    norm = d / absolute_max
    norm = np.clip(norm, 0.0, 1.0) # Zorgt dat we netjes tussen 0% en 100% blijven
    # Pas de Hot colormap toe
    colormap = matplotlib.colormaps.get_cmap("hot")
    colored  = (colormap(norm)[:, :, :3] * 255).astype(np.uint8)
    return colored

def save_tiff(data: np.ndarray, path: str, hot: bool = True):
    if hot:
        if data.ndim == 2:
            colored = apply_hot_colormap(data)
        else:
            colored = data
        tifffile.imwrite(path, colored, photometric="rgb")
        print(f"  ✅  Opgeslagen (hot RGB): {os.path.basename(path)}")
    else:
        if data.dtype == np.uint16:
            out = data
        else:
            d_min, d_max = data.min(), data.max()
            if d_max > d_min:
                out = ((data.astype(np.float32) - d_min) /
                       (d_max - d_min) * 65535).astype(np.uint16)
            else:
                out = data.astype(np.uint16)
        tifffile.imwrite(path, out, photometric="minisblack")
        print(f"  ✅  Opgeslagen (grijs 16-bit): {os.path.basename(path)}")

def process_lif_file(lif_path, uitvoer_map, channel, modus, gebruik_hot, serie_idx=None):
    """Verwerkt een specifiek LIF bestand. Als serie_idx None is, doe dan alles."""
    try:
        lif = LifFile(lif_path)
    except Exception as e:
        print(f"❌  Kan {os.path.basename(lif_path)} niet openen: {e}")
        return

    images = list(lif.get_iter_image())
    if not images:
        print(f"⚠️  Geen series in {os.path.basename(lif_path)}")
        return

    series_to_process = [images[serie_idx]] if serie_idx is not None else images

    for i, img in enumerate(series_to_process):
        actual_serie_idx = serie_idx if serie_idx is not None else i
        
        if channel >= img.channels:
            print(f"⚠️  Kanaal {channel} ontbreekt in '{img.name}' (LIF: {os.path.basename(lif_path)}). Overgeslagen.")
            continue

        lif_naam   = os.path.splitext(os.path.basename(lif_path))[0]
        serie_naam = (img.name or f"serie{actual_serie_idx}").replace(" ", "_").replace("/", "-")
        
        # --- FIX VOOR TE LANGE BESTANDSnamen ---
        # Voorkom dat de serienaam en bestandsnaam dubbel in de titel komen als ze (deels) hetzelfde zijn
        if lif_naam == serie_naam:
            basis = f"{lif_naam}__ch{channel}"
        elif lif_naam in serie_naam:
            basis = f"{serie_naam}__ch{channel}"
        else:
            basis = f"{lif_naam}__{serie_naam}__ch{channel}"
            
        kleur_tag  = "hot" if gebruik_hot else "gray"
        basis = f"{basis}__{kleur_tag}"

        # Harde limiet: als de basisnaam nog steeds meer dan 150 tekens is, knip hem dan af
        if len(basis) > 150:
            basis = basis[:150] + "_kort"
        # ---------------------------------------

        print(f"\n⏳  Ophalen: {lif_naam} -> {serie_naam} (kanaal {channel})")
        stack = get_image_data(img, channel)

        if modus == "1" or modus == "max":
            proj = max_projection(stack)
            pad  = os.path.join(uitvoer_map, f"{basis}__maxproj.tiff")
            save_tiff(proj, pad, hot=gebruik_hot)
        elif modus == "2" or modus == "perlayer":
            for z in range(stack.shape[0]):
                pad = os.path.join(uitvoer_map, f"{basis}__z{z:04d}.tiff")
                save_tiff(stack[z], pad, hot=gebruik_hot)


# ─────────────────────────────────────────────
# Interactieve modus
# ─────────────────────────────────────────────

def interactive_mode():
    print("=" * 60)
    print("   LIF → TIFF Converter (Enkel & Batch Modus)")
    print("=" * 60)

    invoer = input("\n📁  Pad naar .LIF bestand OF map met .LIF bestanden: ").strip().strip('"').strip("'")
    if not os.path.exists(invoer):
        sys.exit(f"❌  Niet gevonden: {invoer}")

    is_dir = os.path.isdir(invoer)

    if is_dir:
        lif_files = glob.glob(os.path.join(invoer, "*.lif"))
        if not lif_files:
            sys.exit(f"❌  Geen .LIF bestanden gevonden in map: {invoer}")
        print(f"  📂  {len(lif_files)} .LIF bestand(en) gevonden in de map.")
        kanaal = int(input("\n🎨  Welk kanaal verwerken we voor alle bestanden? (bijv. 0): ").strip())
        serie_idx = None # Batch mode: verwerk alle series
    else:
        lif = LifFile(invoer)
        images = list(lif.get_iter_image())
        if not images:
            sys.exit("❌  Geen series gevonden in dit LIF-bestand.")
        list_images(lif)
        
        if len(images) == 1:
            serie_idx = 0
            print("  (Automatisch serie 0 geselecteerd.)")
        else:
            serie_idx = int(input(f"🔢  Welke serie? (0–{len(images)-1}): ").strip())
        
        n_channels = images[serie_idx].channels
        if n_channels == 1:
            kanaal = 0
            print("  (Automatisch kanaal 0 geselecteerd.)")
        else:
            kanaal = int(input(f"\n🎨  Welk kanaal? (0–{n_channels-1}): ").strip())

    print("\n📐  Z-modus:")
    print("  [1]  Max-projectie  (alle lagen samengevat → 1 TIFF)")
    print("  [2]  Per z-laag     (elke laag apart als TIFF)")
    modus = input("  Keuze (1 of 2): ").strip()

    print("\n🌡️   Kleurinstelling:")
    print("  [1]  Hot colormap  (zwart → rood → geel → wit)  ← standaard")
    print("  [2]  Grijswaarde   (16-bit, voor kwantitatieve analyse)")
    kleur_keuze = input("  Keuze (1 of 2, standaard=1): ").strip()
    gebruik_hot = kleur_keuze != "2"

    standaard_map = invoer if is_dir else os.path.dirname(os.path.abspath(invoer))
    uitvoer_map   = input(f"\n💾  Uitvoermap [{standaard_map}]: ").strip()
    if not uitvoer_map:
        uitvoer_map = standaard_map
    os.makedirs(uitvoer_map, exist_ok=True)

    print("\n🚀  Start conversie...")
    if is_dir:
        for f in lif_files:
            process_lif_file(f, uitvoer_map, kanaal, modus, gebruik_hot, serie_idx=None)
    else:
        process_lif_file(invoer, uitvoer_map, kanaal, modus, gebruik_hot, serie_idx=serie_idx)

    print("\n🎉  Klaar!\n")


# ─────────────────────────────────────────────
# Command-line modus
# ─────────────────────────────────────────────

def cli_mode():
    p = argparse.ArgumentParser(description="LIF → TIFF Batch Converter")
    p.add_argument("invoer",        help="Pad naar het .LIF bestand OF map met .LIF bestanden")
    p.add_argument("--info",        action="store_true", help="Toon series-overzicht en stop (alleen bij enkel bestand)")
    p.add_argument("--serie",       type=int, default=None, help="Serie-index (standaard: verwerk alles in de map, of serie 0 bij enkel bestand)")
    p.add_argument("--kanaal",      type=int, default=0, help="Kanaal-index (standaard: 0)")
    p.add_argument("--modus",       choices=["max", "perlayer"], default="max", help="'max' = max-projectie, 'perlayer' = per z-laag")
    p.add_argument("--gray",        action="store_true", help="Grijswaarde 16-bit output")
    p.add_argument("--uitvoer",     default=None, help="Uitvoermap")
    args = p.parse_args()

    if not os.path.exists(args.invoer):
        sys.exit(f"❌  Niet gevonden: {args.invoer}")

    is_dir = os.path.isdir(args.invoer)
    gebruik_hot = not args.gray
    uitvoer_map = args.uitvoer or (args.invoer if is_dir else os.path.dirname(os.path.abspath(args.invoer)))
    os.makedirs(uitvoer_map, exist_ok=True)

    if is_dir:
        lif_files = glob.glob(os.path.join(args.invoer, "*.lif"))
        print(f"📂  Start batch conversie voor {len(lif_files)} bestanden...")
        for f in lif_files:
            process_lif_file(f, uitvoer_map, args.kanaal, args.modus, gebruik_hot, serie_idx=args.serie)
    else:
        if args.info:
            list_images(LifFile(args.invoer))
            return
        serie = args.serie if args.serie is not None else 0
        process_lif_file(args.invoer, uitvoer_map, args.kanaal, args.modus, gebruik_hot, serie_idx=serie)

    print("\n🎉  Klaar!")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        cli_mode()