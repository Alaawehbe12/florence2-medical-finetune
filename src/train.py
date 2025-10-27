import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor, AdamW, get_scheduler
from peft import LoraConfig, get_peft_model
import yaml
from src.dataset import MedicalDataset
from tqdm import tqdm

# ------------------------------
# Load Config
# ------------------------------
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ------------------------------
# Dataset & DataLoader
# ------------------------------
train_dataset = MedicalDataset(cfg["train_json"], cfg["image_dir"])
val_dataset = MedicalDataset(cfg["val_json"], cfg["image_dir"])

def collate_fn(batch):
    prefixes, suffixes, images = zip(*batch)
    inputs = processor(text=list(prefixes), images=list(images), return_tensors="pt", padding=True)
    
    # tokenize labels
    tokenized_labels = processor.tokenizer(
        list(suffixes),
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    labels = tokenized_labels["input_ids"].clone()
    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    labels[labels == pad_token_id] = -100

    # Move tensors to device
    batch_out = {k: v.to(DEVICE) for k, v in inputs.items()}
    batch_out["labels"] = labels.to(DEVICE)
    return batch_out

train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=4)

# ------------------------------
# Model + Processor
# ------------------------------
print("Loading model:", cfg["model_name"])
model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], trust_remote_code=True)
processor = AutoProcessor.from_pretrained(cfg["processor_name"], trust_remote_code=True)

# ------------------------------
# LoRA Adapter
# ------------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=cfg.get("lora_target_modules", ["q_proj","k_proj","v_proj","o_proj","lm_head"]),
    task_type="CAUSAL_LM",
    lora_dropout=0.01
)
model = get_peft_model(model, lora_config)
model.to(DEVICE)
model.print_trainable_parameters()

# ------------------------------
# Training Function
# ------------------------------
def train():
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"])
    num_training_steps = cfg["epochs"] * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type=="cuda"))

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        for batch in pbar:
            labels = batch.pop("labels")
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE.type=="cuda")):
                outputs = model(**batch, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            torch.cuda.empty_cache()

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch+1} - Avg Train Loss: {avg_loss:.4f}")

        # ------------------------------
        # Save checkpoint
        # ------------------------------
        checkpoint_dir = os.path.join("checkpoints", f"epoch_{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        try:
            model.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)
        except Exception as e:
            print("Warning: failed to save checkpoint:", e)
        print(f"Checkpoint saved to {checkpoint_dir}")

        # ------------------------------
        # Optional validation loop
        # ------------------------------
        validate(epoch)

# ------------------------------
# Validation function (simple)
# ------------------------------
def validate(epoch_idx):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            labels = batch.pop("labels")
            outputs = model(**batch, labels=labels)
            val_loss += outputs.loss.item()
    avg_val_loss = val_loss / max(len(val_loader),1)
    print(f"[Validation] Epoch {epoch_idx+1} - Avg Loss: {avg_val_loss:.4f}")
    model.train()

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    train()

