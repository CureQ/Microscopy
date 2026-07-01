"""
Image_16_verwerk.py
====================
Gecombineerd script voor Image_16:
  1. Detecteer de GELE annotatielijnen via G-kanaal verschil (geannoteerd vs ruw)
  2. Sla het contour-masker op (masker_controle)
  3. Vul de contourringen op tot gevulde binaire maskers (masker_filled)
  4. Verwijder de annotaties uit de ruwe afbeelding via inpainting (zonder_annotaties)

De annotaties zijn GELE/ORANJE ringen getekend rondom eiwitaggregaten.
Detectie: pixels waar het G-kanaal in de geannoteerde afbeelding significant
          hoger is dan in de ruwe afbeelding.

Invoer:
    Image_16_geannoteerd.tif   – de afbeelding met gele contourringen
    Image_16_ruwe.tiff         – de originele ruwe afbeelding

Uitvoer:
    Image_16_masker_controle.tiff   – binaire contour-masker (0/255)
    Image_16_masker_filled.tiff     – gevuld binair masker (0/255)
    Image_16_zonder_annotaties.tiff – inpainted schone afbeelding
    Image_16_preview.png            – visuele controle overlay

Gebruik:
    python Image_16_verwerk.py

Vereisten:
    pip install numpy tifffile opencv-python scipy matplotlib
"""

import os
import numpy as np
import tifffile
import cv2
from scipy import ndimage

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────
#  Configuratie – pas hier de paden aan
# ─────────────────────────────────────────────

INPUT_DIR = r"C:\Users\fustjar\Downloads\Image_16_combi"

GEANNOTEERD_BESTAND = os.path.join(INPUT_DIR, "Image_16_geannoteerd.tif")
RUWE_BESTAND        = os.path.join(INPUT_DIR, "Image_16_ruwe.tiff")

OUTPUT_DIR = INPUT_DIR

# Drempel voor G-kanaal verschil: hoger = minder ruis, minder annotaties
# 30 werkt goed voor deze afbeelding; verlaag naar 20 als je te weinig mist
G_DIFF_DREMPEL = 30

# Minimum pixels per annotatie-component (kleiner wordt als ruis verwijderd)
MIN_SIZE = 8

# Minimum pixels per interieurregio bij het vullen
MIN_INTERIOR = 4

# Preview opslaan?
SAVE_PREVIEW = True


# ─────────────────────────────────────────────
#  Stap 1: Gele annotaties detecteren
# ─────────────────────────────────────────────

def maak_masker_controle(geannoteerd_pad, ruwe_pad, g_diff_drempel=30, min_size=8):
    """
    Detecteert de gele annotatielijnen door het G-kanaal van de geannoteerde
    afbeelding te vergelijken met het G-kanaal van de ruwe afbeelding.

    Gele pixels hebben: R hoog + G hoog + B laag.
    In de ruwe afbeelding (alleen rood signaal): G ≈ 0.
    Dus: annotatie-pixels = pixels waar G_geannoteerd >> G_ruw.
    """
    img = tifffile.imread(geannoteerd_pad)
    ruwe = tifffile.imread(ruwe_pad)

    # G-kanaal verschil
    g_diff = img[:, :, 1].astype(np.int32) - ruwe[:, :, 1].astype(np.int32)
    mask = (g_diff > g_diff_drempel).astype(np.uint8) * 255

    # Kleine ruis-componenten verwijderen
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_size:
            clean[labels == i] = 255

    n_px = int(np.count_nonzero(clean))
    n_comp, _ = cv2.connectedComponents(clean)
    print(f"  Annotatie-pixels: {n_px}, componenten: {n_comp - 1}")
    return clean, img, ruwe


# ─────────────────────────────────────────────
#  Stap 2: Contouren vullen → gevuld masker
# ─────────────────────────────────────────────

def contours_to_filled_masks(contour_mask, min_size=8, min_interior_size=4):
    """
    Vult de gedetecteerde contourringen op naar gevulde maskers.

    Per verbonden component:
    - Gesloten ring → directe fill_holes
    - Meerdere ringen → per ring via interieurdetectie + dilatatie
    - Open boog / punt → convex hull of morfologische sluiting
    """
    h, w = contour_mask.shape
    result = np.zeros((h, w), dtype=np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(contour_mask)

    k5      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_size:
            continue

        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        x  = int(stats[i, cv2.CC_STAT_LEFT])
        y  = int(stats[i, cv2.CC_STAT_TOP])

        pad = 15
        x1 = max(0, x - pad);       y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad);  y2 = min(h, y + bh + pad)
        crop = (labels[y1:y2, x1:x2] == i).astype(np.uint8)

        # Sluit kleine gaten in de ring voor betere fill_holes
        crop_closed = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, k5)

        filled    = ndimage.binary_fill_holes(crop_closed).astype(np.uint8)
        interiors = (filled - crop_closed).astype(np.uint8)
        n_int_raw, int_labels_crop = cv2.connectedComponents(interiors)

        valid_ints = [
            j for j in range(1, n_int_raw)
            if np.count_nonzero(int_labels_crop == j) >= min_interior_size
        ]
        n_int = len(valid_ints)

        if n_int == 0:
            # Open boog of punt: gebruik convex hull of morfologische sluiting
            crop_big = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, k_close)
            crop_filled = ndimage.binary_fill_holes(crop_big).astype(np.uint8)
            result[y1:y2, x1:x2] = np.maximum(result[y1:y2, x1:x2], crop_filled)

        elif n_int == 1:
            # Één ring: directe fill
            result[y1:y2, x1:x2] = np.maximum(result[y1:y2, x1:x2], filled)

        else:
            # Meerdere ringen: separeer per interieur
            for j in valid_ints:
                single_int  = (int_labels_crop == j).astype(np.uint8)
                grown       = cv2.dilate(single_int, k5, iterations=2)
                ring_mask   = np.maximum(single_int, grown & crop_closed)
                ring_filled = ndimage.binary_fill_holes(ring_mask).astype(np.uint8)
                result[y1:y2, x1:x2] = np.maximum(result[y1:y2, x1:x2], ring_filled)

    return (result > 0).astype(np.uint8) * 255


# ─────────────────────────────────────────────
#  Stap 3: Inpainting → schone afbeelding
# ─────────────────────────────────────────────

def inpaint_afbeelding(ruwe_img_rgb, mask):
    """
    Verwijdert de annotatiepixels via inpainting (TELEA).
    Verwacht numpy array in RGB volgorde.
    """
    # cv2.inpaint werkt met BGR
    ruwe_bgr = cv2.cvtColor(ruwe_img_rgb, cv2.COLOR_RGB2BGR)
    resultaat_bgr = cv2.inpaint(ruwe_bgr, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(resultaat_bgr, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────
#  Preview
# ─────────────────────────────────────────────

def save_preview(geannoteerd, masker_controle, masker_filled, schone_img, out_path):
    if not HAS_MPL:
        print("  (matplotlib niet beschikbaar, preview overgeslagen)")
        return

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    axes[0].imshow(geannoteerd)
    axes[0].set_title("Geannoteerd (input)")
    axes[0].axis("off")

    axes[1].imshow(masker_controle, cmap="gray")
    axes[1].set_title("Masker controle (contouren)")
    axes[1].axis("off")

    axes[2].imshow(masker_filled, cmap="gray")
    axes[2].set_title("Gevuld masker")
    axes[2].axis("off")

    axes[3].imshow(schone_img)
    axes[3].imshow(masker_filled, alpha=0.4, cmap="cool", vmin=0, vmax=255)
    axes[3].set_title("Overlay (schoon + masker)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Preview opgeslagen: {os.path.basename(out_path)}")


# ─────────────────────────────────────────────
#  Hoofd-pipeline
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Image_16_verwerk.py")
    print("  Gele annotaties → masker_controle → gevuld masker")
    print("=" * 60)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    masker_controle_pad = os.path.join(OUTPUT_DIR, "Image_16_masker_controle.tiff")
    filled_pad          = os.path.join(OUTPUT_DIR, "Image_16_masker_filled.tiff")
    zonder_pad          = os.path.join(OUTPUT_DIR, "Image_16_zonder_annotaties.tiff")
    preview_pad         = os.path.join(OUTPUT_DIR, "Image_16_preview.png")

    # ── Stap 1: masker_controle ──────────────────────────────────────
    print("[1/3] Gele annotaties detecteren (G-kanaal verschil)...")
    masker, geannoteerd_img, ruwe_img = maak_masker_controle(
        GEANNOTEERD_BESTAND,
        RUWE_BESTAND,
        g_diff_drempel=G_DIFF_DREMPEL,
        min_size=MIN_SIZE,
    )
    tifffile.imwrite(masker_controle_pad, masker, compression="deflate")
    print(f"  Opgeslagen: Image_16_masker_controle.tiff\n")

    # ── Stap 2: contouren vullen ─────────────────────────────────────
    print("[2/3] Contourringen vullen naar masker...")
    binary = (masker > 0).astype(np.uint8)

    masker_filled = contours_to_filled_masks(
        binary,
        min_size=MIN_SIZE,
        min_interior_size=MIN_INTERIOR,
    )

    n_filled = int(np.count_nonzero(masker_filled))
    n_inst_raw, _ = cv2.connectedComponents(masker_filled)
    n_instances = n_inst_raw - 1

    print(f"  Masker-px: {n_filled}  |  Instanties: {n_instances}")
    tifffile.imwrite(filled_pad, masker_filled, compression="deflate")
    print(f"  Opgeslagen: Image_16_masker_filled.tiff\n")

    # ── Stap 3: inpainting op ruwe afbeelding ───────────────────────
    print("[3/3] Annotaties verwijderen uit ruwe afbeelding (inpainting)...")
    schone_img = inpaint_afbeelding(ruwe_img, masker)
    tifffile.imwrite(zonder_pad, schone_img, compression="deflate")
    print(f"  Opgeslagen: Image_16_zonder_annotaties.tiff\n")

    # ── Preview ─────────────────────────────────────────────────────
    if SAVE_PREVIEW:
        print("  Preview genereren...")
        save_preview(geannoteerd_img, masker, masker_filled, schone_img, preview_pad)

    print()
    print("=" * 60)
    print("  Klaar! Uitvoerbestanden:")
    print(f"    Image_16_masker_controle.tiff")
    print(f"    Image_16_masker_filled.tiff")
    print(f"    Image_16_zonder_annotaties.tiff")
    if SAVE_PREVIEW:
        print(f"    Image_16_preview.png")
    print("=" * 60)


if __name__ == "__main__":
    main()