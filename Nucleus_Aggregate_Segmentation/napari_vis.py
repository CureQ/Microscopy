# cellpose_slices_napari.py
import numpy as np
from cellpose import models, io
import napari
 
# --- 1. Laad het TIF-bestand ---
img = io.imread("Cell_nucleus/test1.tif")
 
# --- 2. Initialiseer het model ---
model = models.CellposeModel(gpu=False, model_type='nuclei')
 
# --- 3. Voor elke slice segmentatie uitvoeren ---
all_masks = []
 
for i in range(img.shape[0]):  # door alle Z-slices
    slice_i = img[i]
    masks, flows, styles = model.eval(
        slice_i,
        diameter=250,
        channels=[0, 0],
        flow_threshold=0.4,
        cellprob_threshold=0.0
    )
    all_masks.append(masks)
 
# --- 4. Combineer alle slices tot één 3D array ---
masks_3d = np.stack(all_masks, axis=0)
 
# --- 5. Toon resultaten in Napari ---
viewer = napari.Viewer()
viewer.add_image(img, name="Microscopy", colormap='gray')
viewer.add_labels(masks_3d, name="Cellpose masks")
 
viewer.dims.ndisplay = 3
 
napari.run()
 
print("✅ Segmentatie voltooid en getoond in Napari (niet opgeslagen).")