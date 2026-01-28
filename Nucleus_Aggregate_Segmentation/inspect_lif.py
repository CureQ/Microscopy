import os
from readlif.reader import LifFile

lif_path = r"C:\Users\berka\OneDrive - HvA\Desktop\Projects_4\Integraal Project\Images\E35\E35.lif"

lif = LifFile(lif_path)

print("===== LIF METADATA =====")
print("Number of datasets:", len(lif.get_image_list()))
print()

for i, ds in enumerate(lif.get_image_list()):
    print(f"[DATASET {i}]")
    print("Name:", ds.name)
    print("Dimensions (TZCYX):", ds.dims)     # dims tuple
    print("Channels:", ds.channels)
    print("-----------------------")
