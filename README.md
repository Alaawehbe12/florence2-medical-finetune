# Florence-2 Medical Fine-Tuning
This repository provides a framework for fine-tuning Florence-2 on medical imaging tasks using the MIMIC-CXR dataset. It supports LoRA adapters (PEFT) to enable lightweight, high-performance fine-tuning for medical captioning tasks.

## Dataset
We use the MIMIC-CXR dataset, which contains chest X-ray images and radiology reports. Each report has an **Impression** (short summary) and a **Detailed Report** (full description). Data is organized in JSONL format with two types of captions: **caption** (impression) and **detailed caption** (full report). Example entry: `{"image": "images/img1.png", "prefix": "Chest X-ray", "suffix": "Mild bilateral pneumonia"}`. Here, `image` is a relative path from `dataset/images/`, `prefix` provides context for Florence-2, and `suffix` is the target caption.

Dataset structure:
```
dataset/
├─ images/         # Chest X-ray images
├─ train.jsonl     # training data
└─ val.jsonl       # validation data
```
Repository Structure:
```
florence2-medical-finetune/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ dataset/
│  ├─ images/
│  ├─ train.jsonl
│  └─ val.jsonl
├─ src/
│  ├─ train.py      # Main training script
│  ├─ dataset.py    # Dataset loader
│  └─ utils.py      # Helper functions
├─ configs/
│  └─ config.yaml   # Hyperparameters and paths
└─ checkpoints/     # Saved models
```




## Installation
1. Clone the repo: `git clone https://github.com/yourusername/florence2-medical-finetune.git && cd florence2-medical-finetune`
2. Install dependencies: `pip install -r requirements.txt`
3. Place MIMIC-CXR images in `dataset/images/` and ensure JSONL files (`train.jsonl` and `val.jsonl`) are properly formatted.

## Configuration
All paths, hyperparameters, and LoRA settings are in `configs/config.yaml`:
```yaml
train_json: "dataset/train.jsonl"
val_json: "dataset/val.jsonl"
image_dir: "dataset/images"
model_name: "microsoft/Florence-2-large"
processor_name: "microsoft/Florence-2-large"
batch_size: 4
epochs: 50
lr: 5e-6
lora_target_modules:
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
  - "lm_head"
```
## Training

Run the training script: python src/train.py. LoRA adapters are applied automatically. Checkpoints are saved in checkpoints/epoch_{n}. Mixed-precision training is enabled if GPU is available.

##  Validation & Evaluation

A simple validation loop computes average loss per epoch. Optionally, generate predictions and compare against impressions or detailed reports.

##  Utilities

src/utils.py contains helper functions: seed setting for reproducibility, safe image loading, checkpoint directory creation, JSON saving, and loss computation.

##  Notes

Ensure images in dataset/images/ match paths in JSONL files.

Use prefix to provide context for Florence-2 (e.g., "Chest X-ray").

Use suffix to choose which caption to train on (impression or detailed report).

## References

MIMIC-CXR Dataset

Florence-2 Model

PEFT / LoRA
