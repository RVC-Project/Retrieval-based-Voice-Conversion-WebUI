# Retrieval-based-Voice-Conversion-WebUI

<div align="center">

<h1>Retrieval Tabanlı Ses Dönüşümü Web Arayüzü</h1>
Kolay kullanılabilen VITS tabanlı bir Ses Dönüşümü çerçevesi.<br><br>

[![madewithlove](https://forthebadge.com/images/badges/built-with-love.svg)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
  
<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>
  
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/Retrieval_based_Voice_Conversion_WebUI.ipynb)
[![Licence](https://img.shields.io/github/license/RVC-Project/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

</div>

------
[**Değişiklik Kaydı**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/Changelog_TR.md) | [**SSS (Sıkça Sorulan Sorular)**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/FAQ-(Frequently-Asked-Questions)) 

[**English**](./README.en.md) | [**中文简体**](../README.md) | [**日本語**](./README.ja.md) | [**한국어**](./README.ko.md) ([**韓國語**](./README.ko.han.md)) | [**Türkçe**](./README.tr.md)

Demo Videosu için [buraya](https://www.bilibili.com/video/BV1pm4y1z7Gm/) bakın!

RVC kullanarak Gerçek Zamanlı Ses Dönüşümü Yazılımı: [w-okada/voice-changer](https://github.com/w-okada/voice-changer)

> RVC kullanan çevrimiçi bir demo: Vocal'i Akustik Gitar sesine dönüştüren demo: https://huggingface.co/spaces/lj1995/vocal2guitar

> Vocal2Guitar demo videosu: https://www.bilibili.com/video/BV19W4y1D7tT/

> Ön eğitim modeli için neredeyse 50 saatlik yüksek kaliteli VCTK açık kaynaklı veri kümesi kullanılmıştır.

> Lisanslı yüksek kaliteli şarkı veri kümesi, telif hakkı ihlali endişesi olmadan kullanımınız için sırayla eklenecektir.

## Özet
Bu depo aşağıdaki özelliklere sahiptir:
+ Top1 geri alım kullanarak kaynak özelliğini eğitim seti özelliğiyle değiştirerek ses tonu sızmasını azaltma;
+ Kolay ve hızlı eğitim, hatta göreceli olarak zayıf grafik kartlarında bile;
+ Az miktarda veri ile bile (en az 10 dakika düşük gürültülü konuşma tavsiye edilir) oldukça iyi sonuçlar elde etme;
+ Timbrları değiştirmek için model birleştirmeyi destekleme (ckpt işleme sekmesinde ckpt birleştirme kullanma);
+ Kolay kullanımlı Webui arayüzü;
+ UVR5 modelini kullanarak hızlı bir şekilde vokalleri ve enstrümanları ayırma.
+ En güçlü Yüksek Tiz Ses Ayıklama Algoritması [InterSpeech2023-RMVPE](#Teşekkürler) sessiz ses sorununu önlemek için kullanılması. En iyi sonuçları (önemli ölçüde) sağlar ve Crepe_full'dan daha düşük kaynak tüketimiyle daha hızlıdır.

## Ortamı Hazırlama
Aşağıdaki komutlar Python sürümü 3.8 veya daha yüksek olan ortamda çalıştırılmalıdır.

(Windows/Linux)
Önce pip aracılığıyla ana bağımlılıkları yükleyin:
```bash
# PyTorch ile ilgili temel bağımlılıkları yükleyin, kuruluysa atlayın
# Referans: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

#Windows + Nvidia Ampere Mimarisi(RTX30xx) için, deneyime göre https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/21 adresindeki cuda sürümüne göre pytorch'a karşılık gelen cuda sürümünü belirtmeniz gerekebilir
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

Sonra poetry kullanarak diğer bağımlılıkları yükleyebilirsiniz:
```bash
# Poetry bağımlılık yönetim aracını yükleyin, kuruluysa atlayın
# Referans: https://python-poetry.org/docs/#installation
curl -sSL https://install.python-poetry.org | python3 -

# Proje bağımlılıklarını yükleyin
poetry install
```

Bunun yerine pip kullanarak da yükleyebilirsiniz:
```bash
pip install -r requirements.txt
```

------
Mac kullanıcıları bağımlılıkları `run.sh` üzerinden yükleyebilir:
```bash
sh ./run.sh
```

## Diğer Ön-Modellerin Hazırlanması
RVC'n

in çıkarım ve eğitim için diğer ön-modellere ihtiyacı vardır.

Onları [Huggingface alanımızdan](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/) indirmeniz gerekmektedir.

İşte RVC'nin ihtiyaç duyduğu Diğer Ön-Modellerin ve diğer dosyaların listesi:
```bash
hubert_base.pt

./pretrained 

./uvr5_weights

V2 sürümü modelini test etmek istiyorsanız (v2 sürümü modeli girişi 256 boyutlu 9 katmanlı Hubert+final_proj'dan 768 boyutlu 12 katmanlı Hubert'ın özelliğine ve 3 dönem ayrımına değiştirilmiştir), ek özellikleri indirmeniz gerekecektir.

./pretrained_v2

#Eğer Windows kullanıyorsanız, FFmpeg yüklü değilse bu dictionariyaya da ihtiyacınız olabilir, FFmpeg yüklüyse atlayın
ffmpeg.exe
```
Daha sonra bu komutu kullanarak Webui'yi başlatabilirsiniz:
```bash
python infer-web.py
```
Windows veya macOS kullanıyorsanız, RVC-beta.7z'yi indirip çıkarabilir ve Webui'yi başlatmak için windows'ta `go-web.bat` veya macOS'te `sh ./run.sh` kullanarak RVC'yi doğrudan kullanabilirsiniz.

Ayrıca, RVC hakkında bir rehber de bulunmaktadır ve ihtiyacınız varsa buna göz atabilirsiniz.

## Teşekkürler
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
+ [Vocal pitch extraction:RMVPE](https://github.com/Dream-High/RMVPE)
  + Ön eğitimli model [yxlllc](https://github.com/yxlllc/RMVPE) ve [RVC-Boss](https://github.com/RVC-Boss) tarafından eğitilmiş ve test edilmiştir.
  
## Tüm katkıda bulunanlara teşekkürler
<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>