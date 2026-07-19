
<div align="center">

<h1>Çekme Temelli Ses Dönüşümü Web Arayüzü</h1>
Basit ve kullanımı kolay bir ses tınısı dönüştürme / ses değiştirici çerçevesi.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange
)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>

[![Lisans](https://img.shields.io/github/license/RVC-Project/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Models-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)


</div>

------
[**Değişiklik Geçmişi**](./Changelog_TR.md) | [**SSS (Sıkça Sorulan Sorular)**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/SSS-(Sıkça-Sorulan-Sorular))

[**İngilizce**](../en/README.en.md) | [**中文简体**](../../README.md) | [**日本語**](../jp/README.ja.md) | [**한국어**](../kr/README.ko.md) ([**韓國語**](../kr/README.ko.han.md)) | [**Français**](../fr/README.fr.md) | [**Türkçe**](../tr/README.tr.md) | [**Português**](../pt/README.pt.md)

Burada [Demo Video'muzu](https://www.bilibili.com/video/BV1pm4y1z7Gm/) izleyebilirsiniz!

RVC Kullanarak Gerçek Zamanlı Ses Dönüşüm Yazılımı: [w-okada/voice-changer](https://github.com/w-okada/voice-changer)

> Ön eğitim modeli için veri kümesi neredeyse 50 saatlik yüksek kaliteli VCTK açık kaynak veri kümesini kullanır.

> Yüksek kaliteli lisanslı şarkı veri setleri telif hakkı ihlali olmadan kullanımınız için eklenecektir.

> Lütfen daha büyük parametrelere, daha fazla eğitim verisine sahip RVCv3'ün ön eğitimli temel modeline göz atın; daha iyi sonuçlar, değişmeyen çıkarsama hızı ve daha az eğitim verisi gerektirir.

## Özet
Bu depo aşağıdaki özelliklere sahiptir:
+ Ton sızıntısını en aza indirmek için kaynak özelliğini en iyi çıkarımı kullanarak eğitim kümesi özelliği ile değiştirme;
+ Kolay ve hızlı eğitim, hatta nispeten zayıf grafik kartlarında bile;
+ Az miktarda veriyle bile nispeten iyi sonuçlar alın (>=10 dakika düşük gürültülü konuşma önerilir);
+ Timbraları değiştirmek için model birleştirmeyi destekleme (ckpt işleme sekmesi-> ckpt birleştir);
+ Kullanımı kolay Web arayüzü;
+ UVR5 modelini kullanarak hızla vokalleri ve enstrümanları ayırma.
+ En güçlü Yüksek tiz Ses Çıkarma Algoritması [InterSpeech2023-RMVPE](#Krediler) sessiz ses sorununu önlemek için kullanılır. En iyi sonuçları (önemli ölçüde) sağlar ve Crepe_full'den daha hızlı çalışır, hatta daha düşük kaynak tüketimi sağlar.
+ AMD/Intel sistemleri CPU bağımlılıklarını kullanır; Windows DirectML, Linux CPU kullanabilir.

## Ortamın Hazırlanması

Bu dal **Python 3.12 x64** için hazırlanmıştır. Tüm komutları depo kökünde çalıştırın. Ubuntu 24.04 x86_64 önerilir.

### Ubuntu 24.04

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev ffmpeg unzip libsndfile1 libportaudio2

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### Windows

Python 3.12 x64 kurduktan sonra sanal ortam oluşturun:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

### Donanıma göre bağımlılık seçimi

| Donanım | Kurulum |
| --- | --- |
| CPU, AMD, Intel | `requirments_cpu_py312.txt` kullanın; Windows DirectML, Linux CPU kullanabilir |
| NVIDIA RTX 50 serisi | Önce CUDA 12.8 Torch, ardından `requirments_cu128_py312.txt` |
| RTX 50 serisinden önceki NVIDIA | Önce CUDA 11.8 Torch, ardından `requirments_cu118_py312.txt` |

#### CPU, AMD, Intel

```bash
python -m pip install -r requirments_cpu_py312.txt
```

#### NVIDIA RTX 50 serisi: iki aşama

```bash
python -m pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu128_py312.txt
```

#### RTX 50 serisinden önceki NVIDIA: iki aşama

```bash
python -m pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118 \
  --extra-index-url https://pypi.org/simple
python -m pip install -r requirments_cu118_py312.txt
```

Torch ve CUDA durumunu doğrulayın:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
```

Program NVIDIA GPU belleğini ve hesaplama yeteneğini de denetler. Yaklaşık 4 GiB altındaki veya SM 5.3 altındaki kartlar CPU yolunu kullanır.

### Paket kaynakları

Üç `requirments_*.txt` dosyasının başında paket kaynakları yer alır. Resmî kaynakları kullanmak için yalnızca `--index-url` ve `--extra-index-url` satırlarını değiştirin; sürümleri, CUDA eklerini ve iki aşamalı sırayı koruyun.

| Default mirror | Official source |
| --- | --- |
| `https://mirrors.pku.edu.cn/pypi/simple` | `https://pypi.org/simple` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cpu` | `https://download.pytorch.org/whl/cpu` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cu118` | `https://download.pytorch.org/whl/cu118` |
| `https://mirrors.nju.edu.cn/pytorch/whl/cu128` | `https://download.pytorch.org/whl/cu128` |

## Modeller ve çalışma dizinleri

WebUI çalışma dizinlerini otomatik oluşturur. Modelleri [Hugging Face model deposundan](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main) indirin ve şu yapıyı koruyun:

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

### Modelleri indirme

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

Windows AMD/Intel DirectML ortamlarında ayrıca şu dosya gerekir:

```bash
hf download lj1995/VoiceConversionWebUI rmvpe.onnx --revision main \
  --local-dir assets/rmvpe
```

Eski `hubert_base.pt` dosyası bu dalda kullanılmaz. Güncel kod `assets/hubert_base/` altındaki Transformers modelini kullanır. FCPE modeli `torchfcpe` paketine dahildir.

### FFmpeg

Yukarıdaki Ubuntu komutu FFmpeg'i kurar. Windows'ta şu dosyaları depo köküne yerleştirin:

- [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffmpeg.exe?download=true)
- [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffprobe.exe?download=true)

## WebUI'yi başlatma

```bash
python webui.py
```

Grafik arayüzü olmayan Ubuntu sunucusu:

```bash
python webui.py --noautoopen
```

Varsayılan bağlantı noktası `7865`'tir. `.pth` modellerini `assets/weights/`, `.index` dosyalarını `assets/indices/` içine yerleştirin.

## Krediler
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
+ [Vokal ton çıkarma:RMVPE](https://github.com/Dream-High/RMVPE)
  + Ön eğitimli model [yxlllc](https://github.com/yxlllc/RMVPE) ve [RVC-Boss](https://github.com/RVC-Boss) tarafından eğitilip test edilmiştir.

## Katkıda Bulunan Herkese Teşekkürler
<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>
