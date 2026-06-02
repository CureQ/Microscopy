# CureQ Microscopy — User Guide

### Step 1 Create a new environment 
Open a terminal (Command Prompt / PowerShell on Windows, Terminal on Mac/Linux):

```bash
conda create -n microscopy python=3.11
```
activate the environment 
```bash
conda activate microscopy
```
you should now see 
(microscopy)

### Step 2 — Install dependencies

Before installing anything, make sure you are in the correct folder.
Put these files in the same folder:
models map
final.py
Q_logo.jpg
final_requirements.txt

Then open a terminal (Command Prompt / PowerShell on Windows, Terminal on Mac/Linux) and navigate to the folder:
Mac / Linux:
cd /path/to/your/project/folder
Windows:
cd C:\path\to\your\project\folder

```bash
pip install -r final_requirements.txt
```

```bash
# 2a. PyTorch — Mac Apple Silicon (MPS):
pip install torch torchvision

# 2b. PyTorch — CPU only (any machine, slower):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2c. PyTorch — Windows/Linux NVIDIA GPU (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

```

### Step 3 — Run the app
Navigate to the folder

Then run:
```bash
python final.py
```