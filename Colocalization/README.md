# CCT1–mHTT Colocalization Analysis

This repository accompanies the master's thesis *"Quantitative Colocalization Analysis of CCT1 and Mutant Huntingtin Aggregates in Fluorescence Microscopy Images"*, conducted at the Amsterdam University of Applied Sciences in collaboration with the Prinses Máxima Centrum.

The goal of this project is to quantify spatial colocalization between mutant Huntingtin (mHTT) aggregates and the CCT1 chaperone using 3D fluorescence microscopy data. The pipeline combines both traditional image analysis techniques and deep learning models (e.g., nnU-Net) to segment structures of interest and compute overlap metrics such as Pearson’s coefficient, Manders’ M1/M2, and Dice similarity coefficient.

---

## 📊 Methodology Overview

The analysis pipeline is structured as follows:

1. **Image Loading**: Supports multi-channel .tiff and .lif files with z-stacks.
2. **Segmentation**: Cellpose (for cells), nnU-Net (for mHTT and CCT1), and threshold-based alternatives.
3. **Colocalization Analysis**: Pearson’s, Manders’ M1/M2, and Dice coefficient, optionally visualized.
4. **Output**: Visual overlays, 3D renderings, and tabular summary files per image.

---

## 🔧 Requirements

- Python 3.9+
- Jupyter
- Libraries: `numpy`, `scikit-image`, `matplotlib`, `opencv-python`, `cellpose`, `nnunet`, `napari`, `pandas`

Environment files or setup instructions are added alongside the analysis in requirements.txt files in future commits. Ensure CUDA support if running nnU-Net on GPU.

---
### 🔬 Some Core Notebook elements
- **`GUI_final.py`**
The Final GUI implementation. This is the core product.	
- **`logboek.md`**  
  Weekly development and research log documenting the iterative process, meetings, and decisions.
- **`Zelfevaluatieformulier_eindverslag.pdf`**  
  Self-assessment form for evaluating learning outcomes in the context of the graduation project.
- **`USER_GUIDE.md`**  
  Practical guide for using the colocalization analysis tool. Includes setup instructions, input requirements, interface walkthrough, and interpretation tips for output metrics.
- **`nnUnet_final.py`**
Implements learned segmentation through an adapted version of the nnUnet pipline.
---
## 🧬 Research Use Only Disclaimer

This software tool, including its graphical user interface (GUI), source code, pretrained models, and documentation, was developed as part of the MSc thesis *“Colocalization Patterns of Mutant Huntingtin and the CCT1 Chaperone”* by N. Wijnheijmer at the Amsterdam University of Applied Sciences.

### 🔬 For Research Purposes Only

This tool is intended strictly for **non-clinical, academic research use** in biomedical image analysis. It is specifically designed to assist in the **quantitative colocalization analysis** of mutant Huntingtin (mHTT) aggregates and CCT1 chaperone signal in 3D fluorescence microscopy images.

- **Not for diagnostic use.** This software must not be used for clinical decision-making or patient care.
- **No therapeutic claims.** While the tool analyzes protein colocalization patterns with potential biological implications, it does not make or support therapeutic inferences.

### ⚖️ Legal Compliance

This software falls under the scientific research exemption defined in **Article 2(6) of the EU Artificial Intelligence Act**. It is not classified as a high-risk or clinical AI system.

- All datasets used are anonymized and derived from research collaborations (e.g., Prinses Máxima Centrum). No personal data is used or stored.
- Compliance with **Recital 26 of the EU General Data Protection Regulation (GDPR)** is maintained, as all image data are non-identifiable and non-personal.

### 📜 Licensing and Third-party Components

The tool integrates open-source packages including, but not limited to: `nnU-Net`, `Cellpose`, `Napari`, `scikit-image`, and `matplotlib`. All components are used in accordance with their respective licenses (e.g., MIT, Apache 2.0). A complete license list is provided in the `LICENSES.md` file.

### 🧪 Intended Audience

This tool is intended for:

- Researchers and technicians at the Prinses Máxima Centrum and collaborating labs
- Biomedical scientists interested in chaperone-aggregate interactions
- AI researchers exploring explainable, biologically grounded segmentation workflows

Users are expected to independently verify and interpret analysis results.

### ❗ Disclaimer of Warranty

No guarantee is given as to the accuracy, completeness, or reliability of the results. The developers, academic supervisors, and affiliated institutions accept no liability for any misuse or misinterpretation of the tool’s output. 

Important: The learned segmentation model (nnU-Net) used in this pipeline was trained on a limited dataset of approximately 100 images from two specific experiments. As such, its performance may not automatically generalize to new microscopy data acquired under different imaging conditions, biological samples, or experimental protocols.

While the pipeline automates several steps of the colocalization analysis, users must be aware that model outputs are not guaranteed to be correct without expert verification. Segmentation errors, overfitting to the training set, and biases in the learned representations may lead to misleading biological interpretations if results are accepted uncritically.

To mitigate this risk:
	•	Visual inspection of the segmentation overlays is mandatory before quantitative results are exported.
	•	Users are strongly advised to cross-check automated outputs against biological plausibility and, where available, against expert-annotated reference data.
	•	Any extension of this pipeline to new datasets should include additional validation of model performance.

This pipeline follows a Human-in-the-Loop (HITL) design philosophy: the user retains responsibility for interpreting and validating all outputs, and automated results should be treated as decision-support, not as ground truth.

---

For support, feedback, or citation requests, please refer to the [User Guide](./USER_GUIDE.md) or contact the corresponding author: noah.wijnheijmer@hva.nl.


## 📍 Citation

If you use or adapt this pipeline, please cite:

> Wijnheijmer, N. (2025). *Colocalization patterns of mutant Huntingtin and the CCT1 chaperone* [Master’s thesis, Amsterdam University of Applied Sciences].

---

## 📬 Contact

For questions or collaborations, contact:  
📧 noah.wijnheijmer@hva.nl

---
