"""
LASXLifImage: A minimal AICSImage-like wrapper for LASX-created LIF files.

- Loads a LIF file using readlif.reader.LifFile and extracts per-channel, per-z numpy arrays.
- Provides .dims (with .C, .Z, .Y, .X) and .get_image_data("ZYX", C=channel, T=0, S=0).
- Only supports single timepoint (T=0) and single scene (S=0).
- Used for the LASX fix in microscopy analysis tool.
"""

import numpy as np
from readlif.reader import LifFile

from ..utils.debug_logger import log


class LASXLifDims:
    def __init__(self, C, Z, Y, X):
        self.C = C
        self.Z = Z
        self.Y = Y
        self.X = X


class LASXLifImage:
    def __init__(self, path):
        file = LifFile(path)
        img0 = file.get_image(0)
        num_channels = img0.channels
        num_z = img0.dims.z
        log(
            f"LASXLifImage: Detected {num_channels} channels, {num_z} z-slices in file '{path}'.",
            level="INFO",
        )
        channel_z_lists = []
        for c in range(num_channels):
            channel_z_lists.append([np.array(i) for i in img0.get_iter_z(t=0, c=c)])
            log(
                f"LASXLifImage: c{c} list shape: {[arr.shape for arr in channel_z_lists[-1]]}",
                level="DEBUG",
            )
        all_channels = np.vstack(channel_z_lists)
        log(
            f"LASXLifImage: all_channels shape after vstack: {all_channels.shape}",
            level="DEBUG",
        )
        real_channels = []
        for c in range(num_channels):
            real_c = [
                all_channels[i] for i in range(c, len(all_channels), num_channels)
            ]
            log(
                f"LASXLifImage: realChannel_{c} length: {len(real_c)}, shape: {real_c[0].shape if real_c else 'EMPTY'}",
                level="DEBUG",
            )
            real_channels.append(np.stack(real_c, axis=0))
        self.data = np.stack(real_channels, axis=0)  # (C, Z, Y, X)
        self.dims = LASXLifDims(
            C=self.data.shape[0],
            Z=self.data.shape[1],
            Y=self.data.shape[2],
            X=self.data.shape[3],
        )
        log(
            f"LASXLifImage: Final data shape: {self.data.shape}, dtype: {self.data.dtype}",
            level="INFO",
        )

    def get_image_data(self, order, C=0, T=0, S=0):
        # Only support order="ZYX", T=0, S=0
        log(
            f"LASXLifImage.get_image_data called: order={order}, C={C}, T={T}, S={S}",
            level="DEBUG",
        )
        if order != "ZYX":
            raise NotImplementedError("LASXLifImage only supports order='ZYX'")
        if T != 0 or S != 0:
            raise NotImplementedError("LASXLifImage only supports T=0, S=0")
        # Return (Z, Y, X) for the requested channel
        arr = self.data[C]
        log(
            f"LASXLifImage.get_image_data: Returning array for channel {C}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}",
            level="DEBUG",
        )
        return arr  # shape (Z, Y, X)
