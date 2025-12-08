# Voice Datasets

This directory contains the audio datasets for training custom RVC models.

## Structure

Each subdirectory corresponds to a specific voice type:

- `male_low/`: Bass/Baritone male voices
- `male_mid/`: Tenor/Mid-range male voices
- `female_low/`: Alto/Contralto female voices
- `female_high/`: Soprano/High-range female voices
- `anime_airy/`: Breath/Airy anime-style voices
- `accent_non_native/`: Voices with distinct non-native accents
- `singing_male/`: Male singing vocals
- `singing_female/`: Female singing vocals
- `child/`: Child voices
- `elderly/`: Elderly voices

## How to Add Data

1.  **Collect Audio**: Gather 10-15 minutes of clean, single-speaker audio for the desired category.
2.  **Place Files**: Put the raw audio files (mp3, wav, etc.) into a temporary folder or directly here.
3.  **Process**: Use the provided tool to normalize and split the audio.

```bash
# Example: Processing a raw file into the male_low dataset
python tools/audio_preprocessor.py -i raw_audio/my_voice.mp3 -o datasets/male_low
```

## Requirements

- **Format**: WAV (will be converted automatically)
- **Sample Rate**: 40kHz or 48kHz (will be converted automatically)
- **Channels**: Mono (will be converted automatically)
- **Quality**: No background noise, music, or reverb. Use UVR5 to clean if necessary.
