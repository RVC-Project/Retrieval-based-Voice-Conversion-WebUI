import itertools
import argparse
import os
import sys
import json
import time
from scipy.io import wavfile

# Add root to path
now_dir = os.getcwd()
sys.path.append(now_dir)

from dotenv import load_dotenv
from configs.config import Config
from infer.modules.vc.modules import VC

def main():
    parser = argparse.ArgumentParser(description="Run RVC inference across a parameter grid.")
    parser.add_argument("--model_name", required=True, help="Name of the model (must be in assets/weights)")
    parser.add_argument("--input_path", required=True, help="Path to reference audio file")
    parser.add_argument("--index_path", default="", help="Path to .index file")
    parser.add_argument("--output_dir", default="experiments", help="Base directory for output")
    parser.add_argument("--f0up_key", type=int, default=0, help="Pitch shift (semitones)")
    
    args = parser.parse_args()

    # Load config and VC
    load_dotenv()
    config = Config()
    vc = VC(config)
    vc.get_vc(args.model_name)

    # Define Grid
    # You can modify this grid in the code or make it configurable via JSON later
    grid = {
        "f0method": ["rmvpe", "pm"], # "harvest", "crepe" are slower
        "index_rate": [0.0, 0.5, 0.75, 1.0],
        "filter_radius": [3],
        "rms_mix_rate": [0.25, 1.0],
        "protect": [0.33],
        "resample_sr": [0], # 0 means no resampling
    }

    # Prepare output directory
    model_slug = os.path.splitext(args.model_name)[0]
    audio_slug = os.path.splitext(os.path.basename(args.input_path))[0]
    timestamp = int(time.time())
    experiment_dir = os.path.join(args.output_dir, model_slug, audio_slug, str(timestamp))
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"Starting Grid Search Experiment")
    print(f"Model: {args.model_name}")
    print(f"Input: {args.input_path}")
    print(f"Output: {experiment_dir}")

    # Generate combinations
    keys = grid.keys()
    values = grid.values()
    combinations = list(itertools.product(*values))
    
    results = []

    total = len(combinations)
    print(f"Total combinations to run: {total}")

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"[{i+1}/{total}] Running with {params}")

        # Construct output filename
        # e.g. rmvpe_idx0.5_rms1.0.wav
        filename_parts = [f"{k}{v}" for k, v in params.items()]
        filename = "_".join(filename_parts) + ".wav"
        output_path = os.path.join(experiment_dir, filename)

        # Run Inference
        try:
            info, opt = vc.vc_single(
                0, # sid
                args.input_path,
                args.f0up_key,
                None, # f0_file
                params["f0method"],
                args.index_path,
                None, # file_index2
                params["index_rate"],
                params["filter_radius"],
                params["resample_sr"],
                params["rms_mix_rate"],
                params["protect"]
            )
            
            if "Success" in info:
                tgt_sr, audio_opt = opt
                wavfile.write(output_path, tgt_sr, audio_opt)
                results.append({
                    "params": params,
                    "output_file": filename,
                    "status": "success"
                })
            else:
                print(f"Error: {info}")
                results.append({
                    "params": params,
                    "status": "failed",
                    "error": info
                })

        except Exception as e:
            print(f"Exception: {e}")
            results.append({
                "params": params,
                "status": "error",
                "error": str(e)
            })

    # Save metadata
    metadata_path = os.path.join(experiment_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "model": args.model_name,
            "input_path": args.input_path,
            "f0up_key": args.f0up_key,
            "grid": grid,
            "results": results
        }, f, indent=2)

    print(f"Experiment completed. Results saved to {experiment_dir}")

if __name__ == "__main__":
    main()
