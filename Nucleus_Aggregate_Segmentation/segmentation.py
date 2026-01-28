import numpy as np
from cellpose import models
import napari
from readlif.reader import LifFile
from aicsimageio import AICSImage
from scipy.ndimage import gaussian_filter
 
def run_cellpose_and_show_full_lif(
    lif_path, channel_index=0, gpu=False, diameter=250,
    flow_threshold=0.4, cellprob_threshold=0.0
):
    """
    Voert Cellpose-segmentatie uit op het gekozen kanaal van een LIF (of ander bestand)
    en toont het resultaat samen met alle kanalen in Napari.
    """
    print(f"📂 Bestand openen: {lif_path}")
    ext = lif_path.lower().split(".")[-1]
 
    # ---- LIF-bestand
    if ext == "lif":
        file = LifFile(lif_path)
        img0 = file.get_image(0)
        num_channels = img0.channels
        num_z = img0.dims.z
        print(f"📸 LIF: {num_channels} kanalen, {num_z} Z-slices")
 
        # 🔹 Lees ALLE kanalen in voor visualisatie
        full_stack = []
        for c in range(num_channels):
            z_planes = [np.array(z, dtype=np.float32) for z in img0.get_iter_z(t=0, c=c)]
            full_stack.append(np.stack(z_planes, axis=0))
        full_stack = np.stack(full_stack, axis=1)  # (Z, C, Y, X)
        print(f"🖼️ Volledige stack shape: {full_stack.shape}")
 
        # 🔹 Lees alleen nucleus channel voor segmentatie
        nucleus_stack = full_stack[:, channel_index, :, :].copy()
 
    # ---- Overige formaten
    else:
        img = AICSImage(lif_path)
        num_channels = getattr(img.dims, "C", 1)
        num_z = getattr(img.dims, "Z", 1)
        print(f"📸 AICS: {num_channels} kanalen, {num_z} Z-slices")
 
        full_stack = img.get_image_data("ZCYX", T=0).astype(np.float32)
        nucleus_stack = full_stack[:, channel_index, :, :].copy()
 
    # ---- Normalisatie + smoothing nucleus channel
    for i in range(nucleus_stack.shape[0]):
        plane = nucleus_stack[i]
        plane = gaussian_filter(plane, sigma=0.8)
        if plane.ptp() > 0:
            plane = (plane - plane.min()) / plane.ptp()
        nucleus_stack[i] = plane
 
    print(f"🧬 Stack shape nucleus: {nucleus_stack.shape}")
 
    # ---- Cellpose segmentatie
    model = models.CellposeModel(gpu=gpu, model_type='nuclei')
    masks_list = []
    for i, slice_i in enumerate(nucleus_stack):
        print(f"⚙️ Segmenteren slice {i+1}/{len(nucleus_stack)} ...")
        masks, flows, styles = model.eval(
            slice_i,
            diameter=diameter,
            channels=[0, 0],
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold
        )
        masks_list.append((masks > 0).astype(np.uint8))
    masks_3d = np.stack(masks_list, axis=0)
    print(f"✅ Segmentatie voltooid: shape={masks_3d.shape}")
 
    # ---- Normaliseer alle kanalen voor Napari-weergave
    full_stack_norm = []
    for c in range(num_channels):
        channel = full_stack[:, c, :, :]
        ch_norm = np.zeros_like(channel)
        for i in range(channel.shape[0]):
            plane = channel[i]
            if plane.ptp() > 0:
                plane = (plane - plane.min()) / plane.ptp()
            ch_norm[i] = plane
        full_stack_norm.append(ch_norm)
    full_stack_norm = np.stack(full_stack_norm, axis=1)  # (Z, C, Y, X)
 
    # ---- Toon in Napari
    viewer = napari.Viewer()
 
    # Voeg elk kanaal apart toe zodat je ze individueel kunt togglen
    for c in range(num_channels):
        viewer.add_image(
            full_stack_norm[:, c, :, :],
            name=f"Channel {c}",
            colormap='gray',
            blending='additive',
            visible=True
        )
 
    # Overlay nucleus-mask
    viewer.add_labels(masks_3d, name="Nucleus mask", opacity=0.5)
    viewer.dims.ndisplay = 3
    napari.run()