"""
Simplified dataset downloader - downloads voice data directly.
"""
import os
import sys
from pathlib import Path

# Ensure we have datasets library
try:
    from datasets import load_dataset
    import soundfile as sf
except ImportError:
    print("Installing required packages...")
    os.system("pip install datasets soundfile")
    from datasets import load_dataset
    import soundfile as sf

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

def download_libritts():
    """Download LibriTTS samples for male/female voices."""
    print("\n=== Downloading LibriTTS (Male/Female Voices) ===")
    
    ds = load_dataset("mythicinfinity/libritts", "clean", split="train.clean.100", streaming=True)
    
    # Speaker ID to voice type mapping
    speaker_voice_map = {
        "19": "male_low",
        "26": "male_low", 
        "1272": "male_low",
        "32": "male_mid",
        "40": "male_mid",
        "1089": "male_mid",
        "87": "female_low",
        "103": "female_low",
        "1284": "female_low",
        "121": "female_high",
        "237": "female_high",
        "3570": "female_high",
    }
    
    counts = {}
    for vt in set(speaker_voice_map.values()):
        counts[vt] = 0
        (DATASETS_DIR / vt).mkdir(parents=True, exist_ok=True)
    
    max_per_type = 100
    
    for sample in ds:
        speaker = str(sample.get("speaker_id", ""))
        
        if speaker in speaker_voice_map:
            voice_type = speaker_voice_map[speaker]
            
            if counts[voice_type] < max_per_type:
                out_dir = DATASETS_DIR / voice_type
                audio = sample["audio"]
                out_path = out_dir / f"libritts_{speaker}_{counts[voice_type]}.wav"
                
                sf.write(str(out_path), audio["array"], audio["sampling_rate"])
                counts[voice_type] += 1
                print(f"  {voice_type}: {counts[voice_type]}/{max_per_type}", end="\r")
        
        if all(c >= max_per_type for c in counts.values()):
            break
    
    print(f"\nLibriTTS complete: {counts}")

def download_dialects():
    """Download English dialect samples."""
    print("\n=== Downloading English Dialects (Accents) ===")
    
    out_dir = DATASETS_DIR / "accent_non_native"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    max_samples = 150
    configs = ["scottish_male", "scottish_female", "irish_male", "welsh_male", "welsh_female", "northern_male"]
    
    for config in configs:
        if count >= max_samples:
            break
        try:
            ds = load_dataset("ylacombe/english_dialects", config, split="train", streaming=True)
            for sample in ds:
                if count >= max_samples:
                    break
                audio = sample["audio"]
                out_path = out_dir / f"dialect_{config}_{count}.wav"
                sf.write(str(out_path), audio["array"], audio["sampling_rate"])
                count += 1
                print(f"  accent_non_native: {count}/{max_samples}", end="\r")
        except Exception as e:
            print(f"  Error with {config}: {e}")
    
    print(f"\nDialects complete: {count} samples")

def download_genshin():
    """Download Genshin voices for anime style."""
    print("\n=== Downloading Genshin Voices (Anime Style) ===")
    
    ds = load_dataset("simon3000/genshin-voice", split="train", streaming=True)
    
    out_dir = DATASETS_DIR / "anime_airy"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    max_samples = 150
    target_chars = ["paimon", "barbara", "kokomi", "nahida", "klee", "qiqi", "diona"]
    
    for sample in ds:
        speaker = str(sample.get("speaker", "")).lower()
        lang = sample.get("language", "")
        
        if lang != "en":
            continue
        
        if any(char in speaker for char in target_chars) and count < max_samples:
            audio = sample["audio"]
            clean_speaker = speaker.replace(" ", "_").replace("/", "_")
            out_path = out_dir / f"genshin_{clean_speaker}_{count}.wav"
            
            sf.write(str(out_path), audio["array"], audio["sampling_rate"])
            count += 1
            print(f"  anime_airy: {count}/{max_samples}", end="\r")
        
        if count >= max_samples:
            break
    
    print(f"\nGenshin complete: {count} samples")

def download_hifi():
    """Download Hi-Fi TTS for singing voices."""
    print("\n=== Downloading Hi-Fi TTS (Singing/High Quality) ===")
    
    ds = load_dataset("MikhailT/hifi-tts", "clean", split="train", streaming=True)
    
    # Map speakers to voice types
    speaker_map = {
        "92": "singing_male",
        "6097": "singing_male",
        "6670": "singing_female",
        "6671": "singing_female",
    }
    
    counts = {}
    for vt in set(speaker_map.values()):
        counts[vt] = 0
        (DATASETS_DIR / vt).mkdir(parents=True, exist_ok=True)
    
    max_per_type = 100
    
    for sample in ds:
        speaker = str(sample.get("speaker", ""))
        
        if speaker in speaker_map:
            voice_type = speaker_map[speaker]
            
            if counts[voice_type] < max_per_type:
                out_dir = DATASETS_DIR / voice_type
                audio = sample["audio"]
                out_path = out_dir / f"hifi_{speaker}_{counts[voice_type]}.wav"
                
                sf.write(str(out_path), audio["array"], audio["sampling_rate"])
                counts[voice_type] += 1
                print(f"  {voice_type}: {counts[voice_type]}/{max_per_type}", end="\r")
        
        if all(c >= max_per_type for c in counts.values()):
            break
    
    print(f"\nHi-Fi TTS complete: {counts}")

def print_summary():
    """Print download summary."""
    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    
    voice_types = [
        "male_low", "male_mid", "female_low", "female_high",
        "anime_airy", "accent_non_native", "singing_male", "singing_female",
        "child", "elderly"
    ]
    
    total = 0
    for vt in voice_types:
        vt_dir = DATASETS_DIR / vt
        if vt_dir.exists():
            files = list(vt_dir.glob("*.wav"))
            count = len(files)
            total += count
            status = "✓" if count > 0 else "✗"
            print(f"  {status} {vt}: {count} files")
        else:
            print(f"  ✗ {vt}: 0 files")
    
    print(f"\nTotal: {total} audio files downloaded")
    print("\nNote: 'child' and 'elderly' need manual data - not available in these datasets.")

def main():
    print("=" * 50)
    print("RVC Voice Dataset Downloader")
    print("=" * 50)
    print(f"Output: {DATASETS_DIR}")
    
    # Create all directories
    for vt in ["male_low", "male_mid", "female_low", "female_high", 
               "anime_airy", "accent_non_native", "singing_male", "singing_female",
               "child", "elderly"]:
        (DATASETS_DIR / vt).mkdir(parents=True, exist_ok=True)
    
    # Download each dataset
    try:
        download_libritts()
    except Exception as e:
        print(f"Error downloading LibriTTS: {e}")
    
    try:
        download_dialects()
    except Exception as e:
        print(f"Error downloading Dialects: {e}")
    
    try:
        download_genshin()
    except Exception as e:
        print(f"Error downloading Genshin: {e}")
    
    try:
        download_hifi()
    except Exception as e:
        print(f"Error downloading Hi-Fi TTS: {e}")
    
    print_summary()
    
    print("\n" + "=" * 50)
    print("NEXT STEPS")
    print("=" * 50)
    print("1. Review downloaded files in datasets/ folder")
    print("2. Train models: python tools/train_batch.py --voice male_low")
    print("3. Run experiments: python tools/run_experiments_batch.py")

if __name__ == "__main__":
    main()
