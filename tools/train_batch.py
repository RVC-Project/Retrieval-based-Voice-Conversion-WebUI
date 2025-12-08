import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config

def run_command(cmd, cwd=None):
    print(f"Running: {cmd}")
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Stream output
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            
    rc = process.poll()
    return rc

def train_voice_model(voice_name, dataset_path, epochs=50, batch_size=8, sample_rate="40k", version="v2", gpu_id="0"):
    """
    Automates the RVC training pipeline for a single voice model.
    """
    print(f"\n{'='*50}")
    print(f"Starting training for: {voice_name}")
    print(f"{'='*50}\n")
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(root_dir, "logs", voice_name)
    
    # 1. Preprocessing
    print("\n[Step 1/4] Preprocessing Data...")
    cmd_preprocess = f"python infer/modules/train/preprocess.py \"{dataset_path}\" {sample_rate.replace('k','000')} 2 \"{logs_dir}\" False 3.0"
    if run_command(cmd_preprocess, cwd=root_dir) != 0:
        print("Error in preprocessing")
        return False

    # 2. Feature Extraction
    print("\n[Step 2/4] Extracting Features...")
    # F0 extraction (rmvpe_gpu)
    cmd_f0 = f"python infer/modules/train/extract/extract_f0_rmvpe.py 1 0 0 \"{logs_dir}\" True"
    if run_command(cmd_f0, cwd=root_dir) != 0:
        print("Error in F0 extraction")
        return False
        
    # Feature extraction (HuBERT)
    cmd_feat = f"python infer/modules/train/extract_feature_print.py {gpu_id} 1 0 0 \"{logs_dir}\" {version} False"
    if run_command(cmd_feat, cwd=root_dir) != 0:
        print("Error in feature extraction")
        return False

    # 3. Training Model
    print("\n[Step 3/4] Training Model...")
    # Determine pretrained models
    if version == "v1":
        pg = f"assets/pretrained/f0G{sample_rate}.pth"
        pd = f"assets/pretrained/f0D{sample_rate}.pth"
    else:
        pg = f"assets/pretrained_v2/f0G{sample_rate}.pth"
        pd = f"assets/pretrained_v2/f0D{sample_rate}.pth"
        
    cmd_train = (
        f"python infer/modules/train/train.py -e \"{voice_name}\" -sr {sample_rate} -f0 1 -bs {batch_size} "
        f"-g {gpu_id} -te {epochs} -se 10 -pg \"{pg}\" -pd \"{pd}\" -l 0 -c 0 -sw 1 -v {version}"
    )
    
    if run_command(cmd_train, cwd=root_dir) != 0:
        print("Error in training")
        return False

    # 4. Training Index
    print("\n[Step 4/4] Training Index...")
    
    index_script = f"""
import sys
import os
import numpy as np
import faiss
from sklearn.cluster import MiniBatchKMeans

exp_dir = "{logs_dir}"
version = "{version}"
feature_dir = os.path.join(exp_dir, "3_feature256" if version == "v1" else "3_feature768")

if not os.path.exists(feature_dir):
    print("Feature dir not found")
    sys.exit(1)

listdir_res = list(os.listdir(feature_dir))
if len(listdir_res) == 0:
    print("No features found")
    sys.exit(1)

npys = []
for name in sorted(listdir_res):
    phone = np.load(os.path.join(feature_dir, name))
    npys.append(phone)

big_npy = np.concatenate(npys, 0)
big_npy_idx = np.arange(big_npy.shape[0])
np.random.shuffle(big_npy_idx)
big_npy = big_npy[big_npy_idx]

if big_npy.shape[0] > 2e5:
    big_npy = (
        MiniBatchKMeans(
            n_clusters=10000,
            batch_size=256 * 8,
            compute_labels=False,
            init="random",
        )
        .fit(big_npy)
        .cluster_centers_
    )

np.save(os.path.join(exp_dir, "total_fea.npy"), big_npy)
n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
index = faiss.index_factory(256 if version == "v1" else 768, "IVF%s,Flat" % n_ivf)
index_ivf = faiss.extract_index_ivf(index)
index_ivf.nprobe = 1
index.train(big_npy)
faiss.write_index(
    index,
    os.path.join(exp_dir, f"trained_IVF{{n_ivf}}_Flat_nprobe_1_{voice_name}_{version}.index")
)

batch_size_add = 8192
for i in range(0, big_npy.shape[0], batch_size_add):
    index.add(big_npy[i : i + batch_size_add])

faiss.write_index(
    index,
    os.path.join(exp_dir, f"added_IVF{{n_ivf}}_Flat_nprobe_1_{voice_name}_{version}.index")
)
print("Index training complete")
"""
    
    # Write temp script
    with open("temp_index_train.py", "w") as f:
        f.write(index_script)
        
    if run_command("python temp_index_train.py", cwd=root_dir) != 0:
        print("Error in index training")
        os.remove("temp_index_train.py")
        return False
        
    os.remove("temp_index_train.py")
    print(f"\nSuccessfully trained model for {voice_name}!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Batch Train RVC Models")
    parser.add_argument("--voice", type=str, help="Specific voice name to train (folder name in datasets/)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    
    args = parser.parse_args()
    
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
    
    if args.voice:
        voices = [args.voice]
    else:
        voices = [d for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
        
    print(f"Found {len(voices)} voices to train: {voices}")
    
    for voice in voices:
        dataset_path = os.path.join(datasets_dir, voice)
        # Check if dataset has files
        if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
            print(f"Skipping {voice} - no data found")
            continue
            
        train_voice_model(voice, dataset_path, epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
