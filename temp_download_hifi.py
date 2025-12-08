
import os
from datasets import load_dataset
from pathlib import Path
import soundfile as sf

print("Loading Hi-Fi TTS dataset...")
ds = load_dataset("MikhailT/hifi-tts", split="train", streaming=True)

datasets_dir = Path("datasets")

# Speaker ID to voice type mapping for HiFi-TTS
# HiFi has 10 speakers total
voice_map = {
    "92": "male_low",      # Deep male
    "6097": "male_mid",    # Mid male
    "6670": "female_low",  # Lower female
    "6671": "female_high", # Higher female
    "8051": "singing_male",
    "9017": "singing_female",
}

counts = {k: 0 for k in set(voice_map.values())}
max_per_type = 100

for sample in ds:
    speaker = str(sample.get("speaker", ""))
    
    if speaker in voice_map:
        voice_type = voice_map[speaker]
        
        if counts[voice_type] < max_per_type:
            out_dir = datasets_dir / voice_type
            out_dir.mkdir(parents=True, exist_ok=True)
            
            audio = sample["audio"]
            out_path = out_dir / f"hifi_{speaker}_{counts[voice_type]}.wav"
            
            sf.write(str(out_path), audio["array"], audio["sampling_rate"])
            counts[voice_type] += 1
            print(f"Saved {out_path.name} ({voice_type}: {counts[voice_type]})")
    
    if all(c >= max_per_type for c in counts.values()):
        break

print(f"Final counts: {counts}")
