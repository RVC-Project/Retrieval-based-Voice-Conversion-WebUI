
import os
from datasets import load_dataset
from pathlib import Path
import soundfile as sf

print("Loading Genshin Voice dataset...")
ds = load_dataset("simon3000/genshin-voice", split="train", streaming=True)

datasets_dir = Path("datasets/anime_airy")
datasets_dir.mkdir(parents=True, exist_ok=True)

count = 0
max_samples = 150

# Target characters with airy/cute voices (English)
target_chars = ["paimon", "barbara", "kokomi", "nahida", "klee", "qiqi", "diona"]

for sample in ds:
    speaker = str(sample.get("speaker", "")).lower()
    lang = sample.get("language", "")
    
    # Only English samples
    if lang != "en":
        continue
        
    if any(char in speaker for char in target_chars) and count < max_samples:
        audio = sample["audio"]
        out_path = datasets_dir / f"{speaker.replace(' ', '_')}_{count}.wav"
        
        sf.write(str(out_path), audio["array"], audio["sampling_rate"])
        count += 1
        print(f"Saved {out_path.name} (total: {count})")
    
    if count >= max_samples:
        break

print(f"Downloaded {count} anime voice samples")
