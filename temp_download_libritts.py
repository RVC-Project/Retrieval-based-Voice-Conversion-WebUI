
import os
from datasets import load_dataset
from pathlib import Path

print("Loading LibriTTS dataset (this may take a while)...")
ds = load_dataset("mythicinfinity/libritts", "clean", split="train.clean.100", streaming=True)

# Sample speakers - take first 50 samples per target voice type
# LibriTTS speaker IDs are in the 'speaker_id' column
target_speakers = {
    "male_low": ["19", "26", "1272"],      # Deep male voices
    "male_mid": ["32", "40", "1089"],      # Mid-range male
    "female_low": ["87", "103", "1284"],   # Lower female
    "female_high": ["121", "237", "3570"], # Higher female
}

datasets_dir = Path("datasets")
counts = {k: 0 for k in target_speakers}
max_per_type = 100  # Max samples per voice type

for sample in ds:
    speaker = str(sample.get("speaker_id", ""))
    
    for voice_type, speakers in target_speakers.items():
        if speaker in speakers and counts[voice_type] < max_per_type:
            out_dir = datasets_dir / voice_type
            out_dir.mkdir(parents=True, exist_ok=True)
            
            audio = sample["audio"]
            out_path = out_dir / f"{speaker}_{sample['id']}.wav"
            
            # Save audio
            import soundfile as sf
            sf.write(str(out_path), audio["array"], audio["sampling_rate"])
            
            counts[voice_type] += 1
            print(f"Saved {out_path.name} ({voice_type}: {counts[voice_type]})")
    
    # Check if we have enough
    if all(c >= max_per_type for c in counts.values()):
        print("Collected enough samples!")
        break

print(f"Final counts: {counts}")
