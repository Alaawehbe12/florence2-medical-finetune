import os
import json
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple

class MedicalDataset(Dataset):
    def __init__(self, jsonl_file: str, image_dir: str):
        self.image_dir = image_dir
        self.entries = self._load_entries(jsonl_file)

    def _load_entries(self, jsonl_file):
        entries = []
        with open(jsonl_file, "r") as f:
            for line in f:
                data = json.loads(line)
                img_path = os.path.join(self.image_dir, data["image"])
                if os.path.exists(img_path):
                    data["image"] = img_path
                    entries.append(data)
        print(f"Loaded {len(entries)} entries from {jsonl_file}")
        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx) -> Tuple[str, str, Image.Image]:
        entry = self.entries[idx]
        image = Image.open(entry["image"]).convert("RGB")
        return entry.get("prefix", ""), entry.get("suffix", ""), image

