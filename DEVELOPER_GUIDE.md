# CureQ Microscopy — Developer Guide

> **Author:** Camiel Jongejeugd — HVA 2025–2026  
> **Purpose:** Help new developers understand the codebase and explain how to make changes correctly.

---

## 1. Folder Structure

```
CureQ/
│
├── microscopy_application.py      ← Entry point — run this to start the app
├── DEVELOPER_GUIDE.md             ← This document
├── USER_GUIDE.md                  ← End-user instructions
├── final_requirements.txt         ← All required Python packages
│
├── core/                          ← Shared base components
│   ├── globals.py                 ← Capability flags (HAS_TORCH, CELLPOSE_AVAILABLE)
│   │                                 and shared constants (BIOLOGICAL_LABELS, SEG_MODE_*)
│   ├── state.py                   ← AppState: central data store
│   ├── workers.py                 ← BaseWorker: base for background tasks
│   ├── theme.py                   ← QSS styling, BTN_* button style constants
│   └── ui_components.py           ← Reusable widgets (ImageCanvas, StatusLabel, etc.)
│
├── utils/                         ← Helper functions
│   ├── channel_utils.py           ← Fill and read channel dropdowns
│   ├── image_utils.py             ← Load and normalise images (_to_zcyx, etc.)
│   └── segmentation_utils.py      ← Cellpose helpers, _cb_postprocess, model cache
│
├── tabs/                          ← One file per tab in the application
│   ├── image_viewer.py            ← Tab 1: load and view images
│   ├── preprocessing.py           ← Tab 2: pre-processing pipeline
│   ├── cell_body_nuclei.py        ← Tab 3: cell body and nucleus segmentation
│   ├── aggregate_detection.py     ← Tab 4: aggregate detection (deep learning)
│   ├── cell_region_analysis.py    ← Tab 5: region analysis + statistics
│   ├── colocalization.py          ← Tab 6: mHTT / CCT1 colocalisation
│   └── results.py                 ← Tab 7: results overview
│
└── models/                        ← AI model weights (.pth files)
    ├── model_fold01 2.pth
    └── ... (10 folds total)
```

---

## 2. How the Application Works

### 2.1 Central Data Store: AppState (`core/state.py`)

All tabs share **one** `AppState` object. It stores:
- The loaded image (`raw_image`, `preprocessed_image`)
- Channel information (`channel_mappings`)
- Derived channels (`nucleus_mask`, `cell_body_mask`, `aggregates`, `region_*`)

**Observer pattern** — tabs subscribe to changes:

```python
state.subscribe(self._on_state_change)

def _on_state_change(self, what: str):
    if what == "raw_image":
        self._refresh_ui()
```

**Saving a derived channel:**
```python
state.set_derived_channel("my_channel", numpy_array, "my_tab")
# All subscribed tabs automatically receive: _on_state_change("derived_channels")
```

### 2.2 Background Tasks: BaseWorker (`core/workers.py`)

Heavy computations run in a separate thread so the UI stays responsive.

```python
class MyWorker(BaseWorker):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self._data = data

    def run_task(self):
        self.signals.progress.emit("Processing...")
        self.signals.progress_percent.emit(50)
        result = compute_something(self._data)
        self.signals.progress_percent.emit(100)
        return result
```

**Starting a worker in a tab:**
```python
self._worker = MyWorker(data, parent=self)
self._worker.signals.progress.connect(self._stat_lbl.info)
self._worker.signals.result.connect(self._on_result)
self._worker.signals.error.connect(lambda m: QMessageBox.critical(self, "Error", m))
self._worker.signals.finished.connect(lambda: self._run_btn.setEnabled(True))
self._worker.start()
```

### 2.3 Structure of a Tab Class

Every tab follows the same pattern:

```python
class MyTab(QWidget):

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self._state  = state
        self._result = None
        self._worker = None
        self._build_ui()
        state.subscribe(self._on_state_change)

    def _build_ui(self):
        root = QHBoxLayout(self)
        ls, _, ll = make_scroll_widget()   # left settings panel
        # ... add widgets to ll ...
        root.addWidget(ls)
        # ... add canvas / right panel ...

    def _on_state_change(self, what: str):
        if what in ("raw_image", "derived_channels"):
            self._update_buttons()

    def _run(self):
        self._run_btn.setEnabled(False)
        self._worker = MyWorker(data, parent=self)
        self._worker.signals.result.connect(self._on_result)
        self._worker.signals.finished.connect(
            lambda: self._run_btn.setEnabled(True))
        self._worker.start()

    def _on_result(self, r):
        self._result = r
        self._state.set_derived_channel("my_output", r["mask"], "my_tab")
```

---

## 3. Common Modifications

### 3.1 Add a New Tab

| Step | File | What to do |
|------|------|------------|
| 1 | `tabs/my_tab.py` | Create `MyTab(QWidget)` and `MyWorker(BaseWorker)` |
| 2 | `microscopy_application.py` | `from tabs.my_tab import MyTab` |
| 3 | `microscopy_application.py` | Add `MyTab(state)` to `MainWindow._build_tabs()` |
| 4 | `core/ui_components.py` | Add help text in `TAB_HELP_TEXT["My Tab"]` |

### 3.2 Add a New Pre-processing Step

| Step | File | What to do |
|------|------|------------|
| 1 | `tabs/preprocessing.py` | Add logic in `Preprocessor.process()` |
| 2 | `tabs/preprocessing.py` | Add `QCheckBox` + parameters in `_build_ui()` |
| 3 | `tabs/preprocessing.py` | Pass the value via `PreprocessingWorker.__init__()` |

### 3.3 Add a New Segmentation Method

| Step | File | What to do |
|------|------|------------|
| 1 | `utils/segmentation_utils.py` | Write `run_my_method(plane, params) -> np.ndarray` |
| 2 | `tabs/cell_body_nuclei.py` | Create a Worker class that calls it |
| 3 | `tabs/cell_body_nuclei.py` | Add a button in `_build_ui()` |
| 4 | — | Save result: `state.set_derived_channel("cell_body_mask", mask, "tab")` |

### 3.4 Add a New Statistic in Cell Region Analysis

| Step | File | What to do |
|------|------|------------|
| 1 | `tabs/cell_region_analysis.py` | Add computation in `_populate_cell_stats()` |
| 2 | `tabs/cell_region_analysis.py` | Add column to `_cell_stats_table` (header + value) |
| 3 | Automatic | `_export_cell_csv()` exports all columns automatically |

### 3.5 Add a New Canvas Overlay

| Step | File | What to do |
|------|------|------------|
| 1 | `tabs/cell_region_analysis.py` | Add `QCheckBox` in `_build_ui()` under "Display" |
| 2 | `tabs/cell_region_analysis.py` | Connect: `cb.stateChanged.connect(self._refresh_overlay)` |
| 3 | `tabs/cell_region_analysis.py` | Add rendering in `_refresh_overlay()` before `ax.set_title(...)` |

### 3.6 Add a New AI Model for Aggregate Detection

| Step | File | What to do |
|------|------|------------|
| 1 | `models/` | Save model as `model_fold[XX].pth` |
| 2 | `tabs/aggregate_detection.py` | Update `EnsembleEngine.load_models()` |
| 3 | `tabs/aggregate_detection.py` | Update `EnsembleEngine.predict()` aggregation |

### 3.7 Support a New Image File Format

| Step | File | What to do |
|------|------|------------|
| 1 | `tabs/image_viewer.py` | Add `try/except` block in `_load_file()` |
| 2 | `utils/image_utils.py` | Use `_to_zcyx(array)` to convert to `(Z, C, Y, X)` |
| 3 | `tabs/image_viewer.py` | `self._state.raw_image = array` + `self._state.notify("raw_image")` |

---

## 4. Data Flow Between Tabs

```
ImageViewerTab
    │  state.raw_image  →  notify("raw_image")
    ↓
PreprocessingTab
    │  state.set_derived_channel("preprocessed_*")
    ↓
CellBodyNucleiTab
    │  state.set_derived_channel("cell_body_mask")
    │  state.set_derived_channel("nucleus_mask")
    ↓
AggregateDetectionTab
    │  state.set_derived_channel("aggregates")
    ↓
CellRegionAnalysisTab
    │  Reads:  nucleus_mask, cell_body_mask, aggregates
    │  Writes: region_nucleus, region_perinuclear,
    │          region_cytoplasm, region_periphery
    ↓
ColocalizationTab
    │  Reads:  raw_image (mHTT and CCT1 channels)
    ↓
ResultsTab
    │  Reads:  all derived channels for summary
```

---

## 5. Important Files and Functions

| Function / Class | File | Description |
|---|---|---|
| `AppState` | `core/state.py` | Central data store |
| `BaseWorker` | `core/workers.py` | Base class for all background workers |
| `_populate_channel_combo()` | `utils/channel_utils.py` | Fill a QComboBox with raw channels |
| `_populate_channel_combo_full()` | `utils/channel_utils.py` | Fill with raw + derived channels |
| `_get_combo_image()` | `utils/channel_utils.py` | Get 2D max-projection for selected channel |
| `_to_zcyx()` | `utils/image_utils.py` | Reshape array to standard (Z, C, Y, X) |
| `_cb_postprocess()` | `utils/segmentation_utils.py` | Remove small objects from mask |
| `_CELLPOSE_MODEL_CACHE` | `utils/segmentation_utils.py` | Global dict caching loaded Cellpose models |
| `make_scroll_widget()` | `core/ui_components.py` | Create a scrollable side panel |
| `section(title)` | `core/ui_components.py` | Create a titled QGroupBox |
| `BTN_RUN` / `BTN_PURPLE` | `core/theme.py` | Button style sheet constants |
| `CELLPOSE_AVAILABLE` | `core/globals.py` | True if Cellpose is installed |
| `HAS_TORCH` | `core/globals.py` | True if PyTorch is installed |

---

## 6. Naming Conventions

| Pattern | Meaning | Example |
|---------|---------|---------|
| `_build_ui()` | Builds all widgets | Present in every tab |
| `_on_state_change(what)` | Reacts to AppState changes | Present in every tab |
| `_run()` | Starts background analysis | Connected to the run button |
| `_on_result(r)` | Processes worker result | `r` is a dict with numpy arrays |
| `[Name]Worker` | Background task class | `CellBodyWorker`, `CellRegionWorker` |
| `[Name]Tab` | Tab UI class | `PreprocessingTab`, `ColocalizationTab` |
| `region_*` | Region mask channel name | `region_nucleus`, `region_cytoplasm` |

---

## 7. Dependencies

```
PyQt5          → GUI framework
matplotlib     → Canvases embedded in Qt
numpy          → All array operations
scipy          → Statistics and image filters
scikit-image   → Segmentation post-processing, regionprops
pandas         → CSV table export
tifffile       → TIFF file I/O
Pillow         → PNG/JPG fallback loading
cellpose       → Cell and nucleus segmentation (requires PyTorch)
torch          → PyTorch backend for cellpose + DL ensemble
segmentation-models-pytorch → U-Net architecture for aggregate detection
timm           → Encoder backbone for U-Net
einops         → Tensor operations in DL models
aicsimageio    → CZI/LIF/ND2 file loading
readlif        → Leica .lif files
```

---

## 8. Running the Application

```bash
conda activate microscopy
cd "/path/to/CureQ"
python microscopy_application.py
```

---

## 9. Common Mistakes

| Error | Cause | Solution |
|-------|-------|----------|
| UI freezes during analysis | Computation in main thread | Use a `BaseWorker` subclass |
| `notify()` triggers nothing | Tab not subscribed | Add `state.subscribe(self._on_state_change)` in `__init__` |
| Channel combo is empty | `_populate_channel_combo_full` not imported | Add it to `from utils.channel_utils import (...)` |
| `NameError: not defined` | Class not imported in module file | Add it to the `from PyQt5.QtWidgets import (...)` block |
| App crashes at startup | Import error in a tab | Check all `from ... import ...` statements |
| Mask has wrong dimensions | Forgot `_to_zcyx()` | Always use `_to_zcyx()` after loading an image |

---

