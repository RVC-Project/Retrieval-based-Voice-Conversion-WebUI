import os
from src.configuration import weight_root

def get_voice_weights():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)

    return names