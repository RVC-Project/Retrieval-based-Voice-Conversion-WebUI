
<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
Framework konversi suara yang mudah digunakan berdasarkan VITS.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange
)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>
  
[![RVC v1](https://img.shields.io/badge/RVCv1-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/tools/ipynb/v1.ipynb)
[![RVC v2](https://img.shields.io/badge/RVCv2-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/tools/ipynb/v2.ipynb)
[![Licence](https://img.shields.io/github/license/RVC-Project/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

[**Changelog**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/Changelog_EN.md) | [**FAQ (Pertanyaan yang Sering Diajukan)**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/FAQ-(Frequently-Asked-Questions)) 

[**English**](../en/README.en.md) | [**中文简体**](../../README.md) | [**日本語**](../jp/README.ja.md) | [**한국어**](../kr/README.ko.md) ([**韓國語**](../kr/README.ko.han.md)) | [**Français**](../fr/README.fr.md) | [**Türkçe**](../tr/README.tr.md) | [**Português**](../pt/README.pt.md)

</div>

> Model dasar dilatih menggunakan hampir 50 jam set data pelatihan VCTK sumber terbuka berkualitas tinggi. Oleh karena itu, tidak ada masalah hak cipta, silakan gunakan dengan bebas.

> Nantikan model dasar RVCv3 dengan parameter yang lebih besar, dataset yang lebih besar, efek yang lebih baik, kecepatan inferensi yang lebih cepat secara dasar, dan jumlah data latihan yang lebih sedikit yang dibutuhkan.

> Ada [downloader satu-klik](https://github.com/RVC-Project/RVC-Models-Downloader) untuk model/paket integrasi/alat. Selamat mencoba.

<table>
   <tr>
		<td align="center">Pelatihan dan Inferensi Webui</td>
		<td align="center">Antarmuka Grafis Pengubahan Suara Real-time</td>
	</tr>
  <tr>
		<td align="center"><img src="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/assets/129054828/092e5c12-0d49-4168-a590-0b0ef6a4f630"></td>
    <td align="center"><img src="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/assets/129054828/730b4114-8805-44a1-ab1a-04668f3c30a6"></td>
	</tr>
	<tr>
		<td align="center">go-web.bat</td>
		<td align="center">go-realtime-gui.bat</td>
	</tr>
  <tr>
    <td align="center">Anda bebas memilih tindakan yang ingin Anda lakukan.</td>
		<td align="center">Kami telah mencapai latensi end-to-end sebesar 170ms. Dengan menggunakan perangkat input dan output ASIO, kami berhasil mencapai latensi end-to-end sebesar 90ms, tetapi ini sangat bergantung pada dukungan driver perangkat keras.</td>
	</tr>
</table>

## Fitur:
+ Kurangi kebocoran nada dengan mengganti fitur sumber ke fitur set-pelatihan menggunakan top1 retrieval;
+ Pelatihan mudah + cepat, bahkan pada kartu grafis yang buruk;
+ Pelatihan dengan jumlah data yang sedikit (>=10 menit bicara rendah bising disarankan);
+ Fusi model untuk mengubah timbre (menggunakan tab proses ckpt->fusi ckpt);
+ WebUI yang mudah digunakan;
+ Model UVR5 untuk memisahkan vokal dan instrumen dengan cepat;
+ Algoritma Ekstraksi Suara Pitch Tinggi [InterSpeech2023-RMVPE](#Credits) untuk mencegah masalah suara yang membisu. Memberikan hasil terbaik (secara signifikan) dan lebih cepat dengan konsumsi sumber daya yang lebih rendah daripada Crepe_full;
+ Dukungan akselerasi kartu grafis AMD/Intel;
+ Dukungan akselerasi kartu grafis Intel ARC dengan IPEX.

Lihat [Video Demo](https://www.bilibili.com/video/BV1pm4y1z7Gm/) kami di sini!

## Konfigurasi Lingkungan
### Batasan Versi Python
> Disarankan untuk menggunakan conda untuk mengelola lingkungan Python.

> Untuk alasan batasan versi, silakan lihat [bug](https://github.com/facebookresearch/fairseq/issues/5012) ini.

```bash
python --version # 3.8 <= Python < 3.11
```

### Instalasi & Skrip Awal Ketergantungan Satu-Klik Linux/MacOS
Dengan menjalankan `run

.sh` di direktori root proyek, Anda dapat mengonfigurasi lingkungan virtual `venv`, secara otomatis menginstal dependensi yang diperlukan, dan memulai program utama dengan satu klik.
```bash
sh ./run.sh
```

### Instalasi Manual Ketergantungan
1. Instal `pytorch` dan dependensi intinya, lewati jika sudah terinstal. Lihat: https://pytorch.org/get-started/locally/
	```bash
	pip install torch torchvision torchaudio
	```
2. Jika Anda menggunakan arsitektur Nvidia Ampere (RTX30xx) di Windows, sesuai pengalaman #21, Anda perlu menentukan versi cuda yang sesuai dengan pytorch.
	```bash
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
	```
3. Instal dependensi yang sesuai sesuai dengan kartu grafis Anda sendiri.
- Nvidia GPU
	```bash
	pip install -r requirements.txt
	```
- AMD/Intel GPU
	```bash
	pip install -r requirements-dml.txt
	```
- AMD ROCM (Linux)
	```bash
	pip install -r requirements-amd.txt
	```
- Intel IPEX (Linux)
	```bash
	pip install -r requirements-ipex.txt
	```

## Persiapan Berkas Lainnya
### 1. Aset
> RVC memerlukan beberapa model yang terletak di folder `assets` untuk inferensi dan pelatihan.
#### Periksa/Unduh Otomatis (Default)
> Secara default, RVC dapat secara otomatis memeriksa integritas sumber daya yang diperlukan saat program utama dimulai.

> Bahkan jika sumber daya tidak lengkap, program akan tetap mulai.

- Jika Anda ingin mengunduh semua sumber daya, tambahkan parameter `--update`.
- Jika Anda ingin melewati pemeriksaan integritas sumber daya saat mulai, tambahkan parameter `--nocheck`.

#### Unduh Secara Manual
> Semua berkas sumber daya berada di [Hugging Face space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

> Anda dapat menemukan beberapa skrip untuk mengunduhnya di folder `tools`

> Anda juga dapat menggunakan [downloader satu-klik](https://github.com/RVC-Project/RVC-Models-Downloader) untuk model/paket integrasi/alat

Berikut adalah daftar yang mencakup nama semua model pra dan berkas lain yang diperlukan oleh RVC.

- ./assets/hubert/hubert_base.pt
	```bash
	rvcmd assets/hubert # Perintah RVC-Models-Downloader
	```
- ./assets/pretrained
	```bash
	rvcmd assets/v1 # Perintah RVC-Models-Downloader
	```
- ./assets/uvr5_weights
	```bash
	rvcmd assets/uvr5 # Perintah RVC-Models-Downloader
	```
Jika Anda ingin menggunakan versi v2 dari model, Anda perlu mengunduh sumber daya tambahan di

- ./assets/pretrained_v2
	```bash
	rvcmd assets/v2 # Perintah RVC-Models-Downloader
	```

### 2. Instalasi alat ffmpeg
Jika `ffmpeg` dan `ffprobe` sudah terinstal, Anda dapat melewatkan langkah ini.
#### Ubuntu/Debian
```bash
sudo apt install ffmpeg
```
#### MacOS
```bash
brew install ffmpeg
```
#### Windows
Setelah diunduh, letakkan di direktori root.
```bash
rvcmd tools/ffmpeg # Perintah RVC-Models-Downloader
```
- [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe)

- [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe)

### 3. Unduh berkas yang diperlukan untuk algoritma ekstraksi pitch suara vokal rmvpe

Jika Anda ingin menggunakan algoritma ekstraksi pitch suara RMVPE terbaru, Anda perlu mengunduh parameter model ekstraksi pitch dan menempatkannya di `assets/rmvpe`.

- [rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt)
	```bash
	rvcmd assets/rmvpe # Perintah RVC-Models-Downloader
	```

#### Unduh lingkungan DML RMVPE (opsional, untuk GPU AMD/Intel)

- [rmvpe.onnx](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.onnx)
	```bash
	rvcmd assets/rmvpe # Perintah RVC-Models-Downloader
	```

### 4. AMD ROCM (opsional, hanya untuk Linux)

Jika Anda ingin menjalankan RVC di sistem Linux berbasis teknologi ROCM milik AMD, harap pertama instal driver yang diperlukan [di sini](https://rocm.docs.amd.com/en/latest/deploy/linux/os-native/install.html).

Jika Anda menggunakan Arch Linux, Anda dapat menggunakan pacman untuk menginstal driver yang diperlukan.
````
pacman -S rocm-hip-sdk rocm-opencl-sdk
````
Untuk beberapa model kartu grafis, Anda mungkin perlu mengonfigurasi variabel lingkungan berikut (seperti: RX6700XT).
````
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
````
Juga, pastikan pengguna saat ini Anda berada dalam grup pengguna `render` dan `video`.
````
sudo usermod -aG render $USERNAME
sudo usermod -aG video $USERNAME
````
## Memulai
### Mulai Langsung
Gunakan perintah berikut untuk memulai WebUI.
```bash
python infer-web.py
```
### Linux/MacOS
```bash
./run.sh
```
### Untuk pengguna I-card yang perlu menggunakan teknologi IPEX (hanya untuk Linux)
```bash
source /opt/intel/oneapi/setvars.sh
./run.sh
```
### Menggunakan Paket Integrasi (Pengguna Windows)
Unduh dan ekstrak `RVC-beta.7z`. Setelah diekstraksi, klik dua kali `go-web.bat` untuk memulai program dengan satu klik.
```bash
rvcmd packs/general/latest # Perintah RVC-Models-Downloader
```

## Kredit
+ [ContentVec](https://github.com/auspicious3000/contentvec/)


+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
+ [Ekstraksi Suara Pitch Vokal:RMVPE](https://github.com/Dream-High/RMVPE)
  + Model pra-dilatih dilatih dan diuji oleh [yxlllc](https://github.com/yxlllc/RMVPE) dan [RVC-Boss](https://github.com/RVC-Boss).

## Terima kasih kepada semua kontributor atas upaya mereka
<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>

