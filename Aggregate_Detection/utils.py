import numpy as np
from skimage import morphology, measure
from scipy.ndimage import label, binary_dilation
import matplotlib.pyplot as plt
import pandas as pd
from cellpose import models, io

io.logger_setup() # run this to get printing of progress

model = models.CellposeModel(gpu=True)

model = models.CellposeModel(gpu=True)

def process_channel(img, channel_idx, diameter, threshold_factor=0.4):
    channel_img = img[:, channel_idx, :, :]
    masks = []
    for z_slice in range(channel_img.shape[0]):
        slice_img = channel_img[z_slice, :, :]

        # Burada kendi modelin eval çağrısı olmalı, aşağıdaki yer tutucu
        mask, _, _ = model.eval(
            slice_img,
            batch_size=8,
            flow_threshold=threshold_factor,
            cellprob_threshold=0.0,
            diameter=diameter
        )
        masks.append(mask)
    return masks

def process_aggregates(img, channel_idx=2):
    aggregate_masks = []
    for z_slice in range(img.shape[0]):
        slice_img = img[z_slice, channel_idx, :, :]
        threshold = 0.6 * np.max(slice_img)
        bright_spots = slice_img > threshold
        clean_mask = morphology.remove_small_objects(bright_spots, min_size=10)
        aggregate_masks.append(clean_mask)
    return aggregate_masks

def plot_gray_overlay_with_nuclei(img, cellbody_masks, nuclei_masks, labeled_aggregates, output_path="output/aggregate_report.pdf"):
    from matplotlib.backends.backend_pdf import PdfPages

    def draw_contours(ax, mask, color):
        for region in measure.regionprops(mask):
            for coords in measure.find_contours(mask == region.label, 0.5):
                ax.plot(coords[:, 1], coords[:, 0], color=color, linewidth=0.5, alpha=0.5)

    with PdfPages(output_path) as pdf:
        for z in range(img.shape[0]):
            base_img = img[z, 2]
            cell_mask = cellbody_masks[z]
            nuclei_mask = nuclei_masks[z]
            agg_mask = labeled_aggregates[z]

            fig, axs = plt.subplots(1, 2, figsize=(14, 7))
            axs[0].imshow(base_img, cmap='magma')
            axs[0].set_title(f"Original Z{z}", fontsize=14)
            axs[0].axis('off')

            axs[1].imshow(base_img, cmap='gray')
            draw_contours(axs[1], cell_mask, 'gray')
            draw_contours(axs[1], nuclei_mask, 'white')

            red_overlay = np.zeros((*agg_mask.shape, 4))
            red_overlay[agg_mask > 0] = [1, 0, 0, 0.7]
            axs[1].imshow(red_overlay)

            axs[1].set_title(f"Overlay Z{z} | Cells: {len(np.unique(cell_mask))-1} | "
                             f"Nuclei: {len(np.unique(nuclei_mask))-1} | "
                             f"Aggs: {len(measure.regionprops(agg_mask))}", fontsize=14)
            axs[1].axis('off')

            fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def categorize_shape_descriptor(eccentricity, circularity):
    if eccentricity > 0.85 and circularity < 0.4:
        return "elongated"
    elif 0.6 < eccentricity <= 0.85 or 0.4 <= circularity < 0.7:
        return "irregular"
    else:
        return "circular"

def categorize_solidity(sol):
    return "smooth" if sol >= 0.9 else "irregular"

def analyze_aggregates_3d_with_2d_shape_features(img, cellbody_masks, nuclei_masks, labeled_aggregates, perinuclear_margin=10):
    labeled_3d, num_features = label(labeled_aggregates > 0)

    nucleus_channel_idx = 0
    aggregate_channel_idx = 2
    cellbody_channel_idx = 3

    all_results = []

    for label_id in range(1, num_features + 1):
        coords_3d = np.argwhere(labeled_3d == label_id)
        volume = coords_3d.shape[0]  # Voxel count

        z_slices = np.unique(coords_3d[:, 0])

        eccs, circs, solids = [], [], []
        overlaps_nuc, overlaps_peri = [], []
        intensities = {'aggregate': [], 'nucleus': [], 'cellbody': []}

        for z in z_slices:
            mask_agg_2d = (labeled_3d[z] == label_id).astype(np.uint8)
            props = measure.regionprops(mask_agg_2d, intensity_image=img[z, aggregate_channel_idx])

            if not props:
                continue

            region = props[0]
            coords = region.coords

            try:
                circ = 4 * np.pi * region.area / (region.perimeter**2 + 1e-6)
                circularity = circ if circ <= 1.2 else np.nan
            except:
                circularity = np.nan

            circs.append(circularity)
            eccs.append(region.eccentricity)
            solids.append(region.solidity)

            nuc_mask = nuclei_masks[z].astype(bool)
            cell_mask = cellbody_masks[z].astype(bool)

            nuclei_dilated = binary_dilation(nuc_mask, iterations=perinuclear_margin)
            perinuclear_mask = nuclei_dilated & (~nuc_mask)

            overlap_nuc = np.sum(nuc_mask[coords[:, 0], coords[:, 1]]) / region.area
            overlap_peri = np.sum(perinuclear_mask[coords[:, 0], coords[:, 1]]) / region.area

            overlaps_nuc.append(overlap_nuc)
            overlaps_peri.append(overlap_peri)

            for ch_name, ch_idx in zip(['aggregate', 'nucleus', 'cellbody'],
                                       [aggregate_channel_idx, nucleus_channel_idx, cellbody_channel_idx]):
                mean_val = np.mean(img[z, ch_idx, coords[:, 0], coords[:, 1]])
                intensities[ch_name].append(mean_val)

        res = {
            'aggregate_3d_label': label_id,
            'z_start': int(z_slices.min()),
            'z_end': int(z_slices.max()),
            'volume_voxels': volume,
            'mean_eccentricity': round(np.nanmean(eccs), 2),
            'mean_solidity': round(np.nanmean(solids), 2),
            'mean_circularity': round(np.nanmean(circs), 2),
            'mean_overlap_nucleus': round(np.nanmean(overlaps_nuc), 2),
            'mean_overlap_perinuclear': round(np.nanmean(overlaps_peri), 2)
        }

        for k in intensities:
            res[f'mean_intensity_{k}'] = round(np.nanmean(intensities[k]), 2)

        if res['mean_overlap_nucleus'] > 0.5:
            res['location'] = 'nuclear'
        elif res['mean_overlap_perinuclear'] > 0.5:
            res['location'] = 'perinuclear'
        else:
            res['location'] = 'cytoplasmic'

        all_results.append(res)

    df = pd.DataFrame(all_results)

    df["solidity"] = df["mean_solidity"].apply(categorize_solidity)
    df["shape"] = df.apply(
        lambda row: categorize_shape_descriptor(row["mean_eccentricity"], row["mean_circularity"]), axis=1
    )
    df["intensity"] = (df["mean_intensity_aggregate"] / 65535).round(2)

    df_filtered = df[[
        "aggregate_3d_label", "z_start", "z_end", "volume_voxels", "location",
        "shape", "solidity", "intensity"
    ]]

    # df_filtered.to_csv(save_path, index=False)
    # print(f"[CSV] Saved to: {save_path}")
    return df_filtered
