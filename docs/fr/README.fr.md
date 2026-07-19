<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
Un framework simple et facile à utiliser pour la conversion du timbre vocal / le changement de voix.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange
)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>

[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Models-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)


[**Journal de mise à jour**](./Changelog_FR.md) | [**FAQ**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E8%A7%A3%E7%AD%94) | [**AutoDL·Formation d'un chanteur AI pour 5 centimes**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/Autodl%E8%AE%AD%E7%BB%83RVC%C2%B7AI%E6%AD%8C%E6%89%8B%E6%95%99%E7%A8%8B) | [**Enregistrement des expériences comparatives**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/%E5%AF%B9%E7%85%A7%E5%AE%9E%E9%AA%8C%C2%B7%E5%AE%9E%E9%AA%8C%E8%AE%B0%E5%BD%95) | [**Démonstration en ligne**](https://huggingface.co/spaces/Ricecake123/RVC-demo)

</div>

------

[**English**](../en/README.en.md) | [ **中文简体**](../../README.md) | [**日本語**](../jp/README.ja.md) | [**한국어**](../kr/README.ko.md) ([**韓國語**](../kr/README.ko.han.md)) | [**Français**](../fr/README.fr.md) | [**Turc**](../tr/README.tr.md) | [**Português**](../pt/README.pt.md)

Cliquez ici pour voir notre [vidéo de démonstration](https://www.bilibili.com/video/BV1pm4y1z7Gm/) !

> Conversion vocale en temps réel avec RVC : [w-okada/voice-changer](https://github.com/w-okada/voice-changer)

> Le modèle de base est formé avec près de 50 heures de données VCTK de haute qualité et open source. Aucun souci concernant les droits d'auteur, n'hésitez pas à l'utiliser.

> Attendez-vous au modèle de base RVCv3 : plus de paramètres, plus de données, de meilleurs résultats, une vitesse d'inférence presque identique, et nécessite moins de données pour la formation.

## Introduction
Ce dépôt a les caractéristiques suivantes :
+ Utilise le top1 pour remplacer les caractéristiques de la source d'entrée par les caractéristiques de l'ensemble d'entraînement pour éliminer les fuites de timbre vocal.
+ Peut être formé rapidement même sur une carte graphique relativement moins performante.
+ Obtient de bons résultats même avec peu de données pour la formation (il est recommandé de collecter au moins 10 minutes de données vocales avec un faible bruit de fond).
+ Peut changer le timbre vocal en fusionnant des modèles (avec l'aide de l'onglet ckpt-merge).
+ Interface web simple et facile à utiliser.
+ Peut appeler le modèle UVR5 pour séparer rapidement la voix et l'accompagnement.
+ Utilise l'algorithme de pitch vocal le plus avancé [InterSpeech2023-RMVPE](#projets-référencés) pour éliminer les problèmes de voix muette. Meilleurs résultats, plus rapide que crepe_full, et moins gourmand en ressources.
+ Les systèmes AMD/Intel utilisent les dépendances CPU ; Windows peut utiliser DirectML et Linux utilise le CPU.

## Configuration de l'environnement

Cette branche cible **Python 3.12 x64**. Exécutez toutes les commandes depuis la racine du dépôt. Ubuntu 24.04 x86_64 est recommandé.

### Ubuntu 24.04

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev ffmpeg unzip libsndfile1 libportaudio2

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Windows

Installez Python 3.12 x64, puis créez un environnement virtuel :

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

### Choisir les dépendances selon le matériel

| Matériel | Installation |
| --- | --- |
| CPU, AMD, Intel | Utiliser `requirments_cpu_py312.txt` ; Windows peut utiliser DirectML, Linux utilise le CPU |
| NVIDIA RTX série 50 | Installer d'abord Torch CUDA 12.8, puis `requirments_cu128_py312.txt` |
| NVIDIA antérieure à la série RTX 50 | Installer d'abord Torch CUDA 11.8, puis `requirments_cu118_py312.txt` |

#### CPU, AMD, Intel

```bash
python -m pip install -r requirments_cpu_py312.txt
```

#### NVIDIA RTX série 50 : installation en deux étapes

```bash
python -m pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu128_py312.txt
```

#### NVIDIA antérieure à la série RTX 50 : installation en deux étapes

```bash
python -m pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu118_py312.txt
```

Vérifiez Torch et CUDA :

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
```


### Sources de paquets

Les trois fichiers `requirments_*.txt` définissent leurs sources en haut. Pour utiliser les sources officielles, remplacez uniquement `--index-url` et `--extra-index-url`, en conservant les versions, les suffixes CUDA et l'ordre des deux étapes.

| Default mirror | Official source |
| --- | --- |
| `https://mirrors.pku.edu.cn/pypi/simple` | `https://pypi.org/simple` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cpu` | `https://download.pytorch.org/whl/cpu` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cu118` | `https://download.pytorch.org/whl/cu118` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cu128` | `https://download.pytorch.org/whl/cu128` |

## Modèles et répertoires d'exécution

Le WebUI crée automatiquement les répertoires d'exécution. Téléchargez les modèles depuis le [dépôt de modèles Hugging Face](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main) et conservez cette structure :

```text
assets/
├── hubert_base/
│   ├── config.json
│   ├── preprocessor_config.json
│   └── pytorch_model.bin
├── rmvpe/rmvpe.pt
├── pretrained/
├── pretrained_v2/
├── uvr5_weights/
├── weights/        # user RVC .pth models
└── indices/        # user .index files
logs/
└── mute/           # training silence samples

# Exact paths used by the code
assets/hubert_base/config.json
assets/hubert_base/preprocessor_config.json
assets/hubert_base/pytorch_model.bin
assets/rmvpe/rmvpe.pt
assets/pretrained/*.pth
assets/pretrained_v2/*.pth
assets/uvr5_weights/*
assets/weights/*.pth
assets/indices/*.index
logs/mute/*
```

### Télécharger les modèles

```bash
python -m pip install --upgrade huggingface_hub

# Required for inference and feature extraction
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "hubert_base/*" --local-dir assets
hf download lj1995/VoiceConversionWebUI rmvpe.pt --revision main \
  --local-dir assets/rmvpe

# Required for v1/v2 training
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "pretrained/*" "pretrained_v2/*" --local-dir assets
hf download lj1995/VoiceConversionWebUI mute.zip --revision main \
  --local-dir .model-downloads
python -m zipfile -e .model-downloads/mute.zip logs

# Required only for UVR5 vocal separation
hf download lj1995/VoiceConversionWebUI --revision main \
  --include "uvr5_weights/*" --local-dir assets
```

Les environnements Windows AMD/Intel DirectML nécessitent aussi :

```bash
hf download lj1995/VoiceConversionWebUI rmvpe.onnx --revision main \
  --local-dir assets/rmvpe
```


### FFmpeg

La commande Ubuntu précédente installe FFmpeg. Sous Windows, placez ces fichiers à la racine du dépôt :

- [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffmpeg.exe?download=true)
- [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffprobe.exe?download=true)

## Démarrer le WebUI

```bash
python webui.py
```

Serveur Ubuntu sans interface graphique :

```bash
python webui.py --noautoopen
```

Le port par défaut est `7865`. Placez les modèles `.pth` dans `assets/weights/` et les fichiers `.index` dans `assets/indices/`.

## Crédits
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
+ [Extraction de la hauteur vocale : RMVPE](https://github.com/Dream-High/RMVPE)
  + Le modèle pré-entraîné a été formé et testé par [yxlllc](https://github.com/yxlllc/RMVPE) et [RVC-Boss](https://github.com/RVC-Boss).

## Remerciements à tous les contributeurs pour leurs efforts
<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>
