
import os
from datasets import load_dataset
from pathlib import Path
import soundfile as sf

print("Loading English Dialects dataset...")
ds = load_dataset("ylacombe/english_dialects", split="train", streaming=True)

datasets_dir = Path("datasets/accent_non_native")
datasets_dir.mkdir(parents=True, exist_ok=True)

count = 0
max_samples = 150

# Target dialects for "non-native" feel (regional accents)
target_dialects = ["scottish", "irish", "welsh", "northern"]

for sample in ds:
    dialect = sample.get("dialect", "").lower()
    
    if any(d in dialect for d in target_dialects) and count < max_samples:
        audio = sample["audio"]
        out_path = datasets_dir / f"{dialect}_{count}.wav"
        
        sf.write(str(out_path), audio["array"], audio["sampling_rate"])
        count += 1
        print(f"Saved {out_path.name} (total: {count})")
    
    if count >= max_samples:
        break

print(f"Downloaded {count} dialect samples")
