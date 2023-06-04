import glob
import os
import shutil

MODELNAME = 'sacks'
# source_dir = f'sway/models/{MODELNAME}/'
source_dir = f'logs/{MODELNAME}/'

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

# List all files in the source directory
files = os.listdir(source_dir)
# Filter files starting with 'G_' and 'D_'
g_files = [f for f in files if ("G_") in f]
d_files = [f for f in files if ("D_") in f]
model_files = [f for f in files if f.startswith(MODELNAME) and has_numbers(f) and "G_" not in f and "D_" not in f]

if model_files:
  model_max = max(model_files, key=lambda x: (x.split(MODELNAME)[-1].split(".")[0]))

if model_files:
  MODELSTEP = (model_max.split(MODELNAME)[1].split(".")[0])
else:
  MODELSTEP = "no"

#@markdown If this stops working for whatever reason please spam ping me as much as possible at discord.gg/aihub
print("MODELSTEP:", MODELSTEP)

# Loading for subsequent training? (Will load G and D files, which are huge but not used for inference)
# Probably not useful, unless I'm trying to finetune an existing model
GDLoading = False
if GDLoading:
    g_files = [f for f in files if ("G_") in f]
    g_max = max(g_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    GNumber = (g_max.split("_")[2].split(".")[0])
    shutil.move(f"{source_dir}/D_{GNumber}.pth", 
                f"logs/{MODELNAME}/G_{MODELSTEP}.pth")
    
    shutil.move(f"{source_dir}/G_{MODELSTEP}.pth", 
                f"logs/{MODELNAME}/D_{MODELSTEP}.pth")

os.makedirs(f"logs/{MODELNAME}/", exist_ok=True)
for pattern in ['*.index', '*.npy']:
    for file in glob.glob(f"{source_dir}/{pattern}"):
        shutil.move(file, f"logs/{MODELNAME}/")

for pattern in [f'{MODELNAME}*.pth', f'{MODELNAME}{MODELSTEP}.pth']:
    for file in glob.glob(f"{source_dir}/{pattern}"):
        shutil.move(file, f"weights/{MODELNAME}.pth")
