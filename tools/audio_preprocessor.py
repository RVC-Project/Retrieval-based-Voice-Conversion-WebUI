import os
import argparse
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm

def process_audio(input_path, output_dir, sr=40000, min_silence_len=500, silence_thresh=-40, chunk_len=10000):
    """
    Process audio file: convert to wav, normalize, remove silence, split into chunks
    """
    filename = os.path.basename(input_path).split('.')[0]
    
    print(f"Processing {input_path}...")
    
    # Load audio
    try:
        audio = AudioSegment.from_file(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return

    # Normalize
    audio = audio.normalize()
    
    # Split on silence
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=100
    )
    
    # Combine small chunks to reach target length
    output_chunks = []
    current_chunk = AudioSegment.empty()
    
    for chunk in chunks:
        if len(current_chunk) + len(chunk) < chunk_len:
            current_chunk += chunk
        else:
            output_chunks.append(current_chunk)
            current_chunk = chunk
    
    if len(current_chunk) > 0:
        output_chunks.append(current_chunk)
        
    # Save chunks
    os.makedirs(output_dir, exist_ok=True)
    
    for i, chunk in enumerate(output_chunks):
        # Convert to target sample rate
        chunk = chunk.set_frame_rate(sr).set_channels(1)
        
        # Export
        out_name = f"{filename}_{i:03d}.wav"
        out_path = os.path.join(output_dir, out_name)
        chunk.export(out_path, format="wav")
        
    print(f"Saved {len(output_chunks)} chunks to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Audio Dataset Preprocessor for RVC")
    parser.add_argument("--input", "-i", required=True, help="Input file or directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--sr", type=int, default=40000, help="Target sample rate (default: 40000)")
    parser.add_argument("--len", type=int, default=10000, help="Target chunk length in ms (default: 10000)")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        process_audio(args.input, args.output, sr=args.sr, chunk_len=args.len)
    elif os.path.isdir(args.input):
        files = [f for f in os.listdir(args.input) if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg'))]
        for f in tqdm(files):
            process_audio(os.path.join(args.input, f), args.output, sr=args.sr, chunk_len=args.len)

if __name__ == "__main__":
    main()
