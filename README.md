# Florence-2 Medical Fine-Tuning

This repository provides a framework for fine-tuning Florence-2 on medical imaging tasks, leveraging the MIMIC-CXR dataset. Florence-2 is a vision-language model capable of performing multiple vision-language tasks, including image captioning, object detection, and detailed image description.

## Fine-Tuning Modes

Florence-2 can be adapted for different types of medical image understanding through specific prefix-based modes in the fine-tuning process:

<CAPTION>: Generates concise, general captions describing the main findings in a medical image. Ideal for brief reports or image annotations.

<DETAILED_CAPTION>: Produces a more comprehensive description, capturing subtle findings, anatomical details, and abnormalities. Useful for radiology reports requiring more context.

<od> (Object Detection): Focuses on detecting and localizing abnormalities or regions of interest within medical images. Can be combined with captions to provide contextualized detection outputs.

Other Prefixes: Custom prefixes can be defined to guide the model toward specialized tasks, such as highlighting specific organs, pathologies, or procedural details.

## Dataset
We use the MIMIC-CXR dataset, which contains chest X-ray images and radiology reports. Each report has an **Impression** (short summary) and a **Detailed Report** (full description). Data is organized in JSONL format with two types of captions: **caption** (impression) and **detailed caption** (findings). Example entry: `{"image": "test_24506.png", "prefix": "<CAPTION>", "suffix": "Interval placement of a left-sided chest tube with tip in the left lung apex. A right chest tube is now seen with side port within the thorax and tip in the apex. The endotracheal tube and enteric tube are unchanged} `. Here, `image` is a relative path from `dataset/images/`, `prefix` provides context for Florence-2, and `suffix` is the target caption.
Also good to 

Dataset structure:
```
dataset/
├─ images/         # Chest X-ray images use the link below to download the images data !! 
├─ train.jsonl     # training data <caption> target
├─ test.jsonl      # testing data <caption> target
├─ val.jsonl       # validation data <caption> target
├─ test_detailed_caption.jsonl  # testing data <DETAILED_CAPTION> target
├─ train_detailed_caption.jsonl # training data <DETAILED_CAPTION> target
└─ val_detailed_caption.jsonl   # validataion data <DETAILED_CAPTION> target

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

[MIMIC-CXR Dataset](https://huggingface.co/datasets/itsanmolgupta/mimic-cxr-dataset)

[Florence-2 Model](https://huggingface.co/microsoft/Florence-2-base)

[PEFT / LoRA](https://huggingface.co/docs/peft/en/package_reference/lora)
