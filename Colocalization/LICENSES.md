# Third‑Party Licences

This repository depends on several **open‑source** libraries and pre‑trained models. The table below lists each direct dependency, its upstream location, and the licence that governs its distribution and use.

| Component                        | Upstream / Homepage                                                                                          | Licence                | Notes                          |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------- | ------------------------------ |
| **Python 3.x**                   | [https://www.python.org](https://www.python.org)                                                             | PSF 2.0                | Permissive                     |
| **NumPy**                        | [https://github.com/numpy/numpy](https://github.com/numpy/numpy)                                             | BSD 3‑Clause           | Permissive                     |
| **SciPy**                        | [https://github.com/scipy/scipy](https://github.com/scipy/scipy)                                             | BSD 3‑Clause           | Permissive                     |
| **pandas**                       | [https://github.com/pandas-dev/pandas](https://github.com/pandas-dev/pandas)                                 | BSD 3‑Clause           | Permissive                     |
| **Matplotlib**                   | [https://github.com/matplotlib/matplotlib](https://github.com/matplotlib/matplotlib)                         | Matplotlib (BSD‑style) | Permissive                     |
| **scikit‑image**                 | [https://github.com/scikit-image/scikit-image](https://github.com/scikit-image/scikit-image)                 | BSD 3‑Clause           | Permissive                     |
| **scikit‑learn**                 | [https://github.com/scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)                 | BSD 3‑Clause           | Permissive                     |
| **OpenCV‑Python**                | [https://github.com/opencv/opencv-python](https://github.com/opencv/opencv-python)                           | Apache 2.0             | Permissive                     |
| **AICSImageIO**                  | [https://github.com/AllenCellModeling/aicsimageio](https://github.com/AllenCellModeling/aicsimageio)         | BSD 3‑Clause           | Permissive                     |
| **tifffile**                     | [https://github.com/cgohlke/tifffile](https://github.com/cgohlke/tifffile)                                   | BSD 3‑Clause           | Permissive                     |
| **napari**                       | [https://github.com/napari/napari](https://github.com/napari/napari)                                         | BSD 3‑Clause           | GUI viewer                     |
| **PyTorch**                      | [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)                                     | BSD 3‑Clause           | Permissive                     |
| **TensorFlow**                   | [https://github.com/tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)                         | Apache 2.0             | Permissive                     |
| **Keras**                        | [https://github.com/keras-team/keras](https://github.com/keras-team/keras)                                   | Apache 2.0             | Permissive                     |
| **nnU‑Net**                      | [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)                                     | Apache 2.0             | Permissive                     |
| **Cellpose**                     | [https://github.com/MouseLand/cellpose](https://github.com/MouseLand/cellpose)                               | MIT                    | Permissive                     |
| **CellposeSAM**                  | same as above                                                                                                | MIT                    | Inherits Cellpose licence      |
| **Segment Anything Model (SAM)** | [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) | Apache 2.0             | Used via CellposeSAM           |
| **CUDA Toolkit (runtime)**       | [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)                   | NVIDIA EULA            | Proprietary; not redistributed |

## Compatibility

All licences listed above are *permissive* and compatible with this project’s main licence (MIT for source code, CC‑BY 4.0 for documentation). The proprietary NVIDIA CUDA runtime is **not distributed** in this repository; users install it separately under NVIDIA’s terms.

## Copyright

© 2025 Noah Wijnheijmer.
Original code: MIT Licence (see `LICENSE`).
Documentation (including this file): CC‑BY 4.0.

---

*Last updated: 4 June 2025*

