from aicsimageio import AICSImage
import tifffile

file_path = "C:/Users/jesse/Documents/HvA/CureQ/Microscopy/tiff_files/tiff_files20240131_CKR_E10B_mHtt-HAQ25Doxy96H_HA-star580_CCT1-star635P_A11-star460L_nucspotlive488_1_channel1.tif"
# tifffile.imread(file_path)
print(AICSImage(file_path))
