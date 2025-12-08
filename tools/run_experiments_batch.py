import os
import subprocess
import sys

# Add root to path
now_dir = os.getcwd()
sys.path.append(now_dir)

def main():
    # List of voices as defined in Task 7/10
    voices = [
        'male_low', 'male_mid', 'female_low', 'female_high',
        'anime_airy', 'accent_non_native', 'singing_male', 'singing_female',
        'child', 'elderly'
    ]

    # Base paths
    weights_dir = os.path.join(now_dir, "assets", "weights")
    datasets_dir = os.path.join(now_dir, "datasets")
    experiments_dir = os.path.join(now_dir, "experiments")
    
    # Path to test_grid.py
    test_grid_script = os.path.join(now_dir, "tools", "test_grid.py")

    print(f"Starting Batch Experiments for {len(voices)} voices...")

    for voice in voices:
        print(f"\n--- Processing Voice: {voice} ---")
        
        # Check for model
        model_name = f"{voice}.pth"
        model_path = os.path.join(weights_dir, model_name)
        
        if not os.path.exists(model_path):
            print(f"Skipping {voice}: Model not found at {model_path}")
            continue

        # Check for test audio
        # We need a reference audio to run inference on.
        # Ideally, we should have a 'test_samples' folder or use a file from the dataset itself (held out).
        # For now, let's look for a 'test.wav' in the voice's dataset folder, or a global test file.
        
        # Strategy: Look for 'test.wav' in dataset dir, else take the first wav file found.
        voice_dataset_dir = os.path.join(datasets_dir, voice)
        input_audio = None
        
        if os.path.exists(voice_dataset_dir):
            potential_files = [f for f in os.listdir(voice_dataset_dir) if f.endswith(".wav")]
            if "test.wav" in potential_files:
                input_audio = os.path.join(voice_dataset_dir, "test.wav")
            elif len(potential_files) > 0:
                input_audio = os.path.join(voice_dataset_dir, potential_files[0])
        
        if not input_audio:
            print(f"Skipping {voice}: No input audio found in {voice_dataset_dir}")
            continue

        # Check for index file (optional but recommended)
        # Usually located in logs/{voice}/added_*.index
        # We need to find it.
        logs_dir = os.path.join(now_dir, "logs", voice)
        index_path = ""
        if os.path.exists(logs_dir):
            for f in os.listdir(logs_dir):
                if f.startswith("added_") and f.endswith(".index"):
                    index_path = os.path.join(logs_dir, f)
                    break
        
        print(f"Model: {model_name}")
        print(f"Input: {input_audio}")
        print(f"Index: {index_path if index_path else 'None'}")

        # Run test_grid.py
        cmd = [
            sys.executable, test_grid_script,
            "--model_name", model_name,
            "--input_path", input_audio,
            "--output_dir", experiments_dir
        ]
        
        if index_path:
            cmd.extend(["--index_path", index_path])

        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully ran experiments for {voice}")
        except subprocess.CalledProcessError as e:
            print(f"Error running experiments for {voice}: {e}")

    print("\nBatch Experiments Completed.")

if __name__ == "__main__":
    main()
