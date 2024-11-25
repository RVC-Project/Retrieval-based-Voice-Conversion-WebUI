<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
一個基於VITS的簡單易用的變聲框架<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange
)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/Retrieval_based_Voice_Conversion_WebUI.ipynb)
[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

[**更新日誌**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/Changelog_CN.md) | [**常見問題解答**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E8%A7%A3%E7%AD%94) | [**AutoDL·5毛錢訓練AI歌手**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/Autodl%E8%AE%AD%E7%BB%83RVC%C2%B7AI%E6%AD%8C%E6%89%8B%E6%95%99%E7%A8%8B) | [**對照實驗記錄**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/Autodl%E8%AE%AD%E7%BB%83RVC%C2%B7AI%E6%AD%8C%E6%89%8B%E6%95%99%E7%A8%8B](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/%E5%AF%B9%E7%85%A7%E5%AE%9E%E9%AA%8C%C2%B7%E5%AE%9E%E9%AA%8C%E8%AE%B0%E5%BD%95)) | [**在線示範**](https://modelscope.cn/studios/FlowerCry/RVCv2demo)

[**English**](./docs/en/README.en.md) | [**中文簡體**](./README.md) | [**中文正體**](README.tw.md) | [**日本語**](./docs/jp/README.ja.md) | [**한국어**](./docs/kr/README.ko.md) ([**韓國語**](./docs/kr/README.ko.han.md)) | [**Français**](./docs/fr/README.fr.md) | [**Türkçe**](./docs/tr/README.tr.md) | [**Português**](./docs/pt/README.pt.md)

</div>

> 底模使用接近50小時的開源高品質VCTK訓練集訓練，無版權方面的顧慮，請大家放心使用

> 請期待RVCv3的底模，參數更大，數據更大，效果更好，基本持平的推理速度，需要訓練數據量更少。

<table>
   <tr>
        <td align="center">訓練推理界面</td>
        <td align="center">即時變聲界面</td>
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
    <td align="center">可以自由選擇想要執行的操作。</td>
        <td align="center">我們已經實現端到端170ms延遲。如使用ASIO輸入輸出設備，已能實現端到端90ms延遲，但非常依賴硬體驅動支持。</td>
    </tr>
</table>

## 簡介
本倉庫具有以下特點
+ 使用top1檢索替換輸入源特徵為訓練集特徵來杜絕音色洩漏
+ 即便在相對較差的顯示卡上也能快速訓練
+ 使用少量數據進行訓練也能得到較好結果(推薦至少收集10分鐘低底噪語音數據)
+ 可以透過模型融合來改變音色(借助ckpt處理選項卡中的ckpt-merge)
+ 簡單易用的網頁界面
+ 可調用UVR5模型來快速分離人聲和伴奏
+ 使用最先進的[人聲音高提取算法InterSpeech2023-RMVPE](#參考項目)根絕啞音問題。效果最好（顯著地）但比crepe_full更快、資源占用更小
+ A卡I卡加速支持

點此查看我們的[示範影片](https://www.bilibili.com/video/BV1pm4y1z7Gm/) !

## 環境配置
以下指令需在 Python 版本大於3.8的環境中執行。  

### Windows/Linux/MacOS等平台通用方法
下列方法任選其一。
#### 1. 通過 pip 安裝依賴
1. 安裝Pytorch及其核心依賴，若已安裝則跳過。參考自: https://pytorch.org/get-started/locally/
```bash
pip install torch torchvision torchaudio
```
2. 如果是 win 系統 + Nvidia Ampere 架構(RTX30xx)，根據 #21 的經驗，需要指定 pytorch 對應的 cuda 版本
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```
3. 根據自己的顯示卡安裝對應依賴
- N卡
```bash
pip install -r requirements.txt
```
- A卡/I卡
```bash
pip install -r requirements-dml.txt
```
- A卡ROCM(Linux)
```bash
pip install -r requirements-amd.txt
```
- I卡IPEX(Linux)
```bash
pip install -r requirements-ipex.txt
```

#### 2. 通過 poetry 來安裝依賴
安裝 Poetry 依賴管理工具，若已安裝則跳過。參考自: https://python-poetry.org/docs/#installation
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

通過 Poetry 安裝依賴時，python 建議使用 3.7-3.10 版本，其餘版本在安裝 llvmlite==0.39.0 時會出現衝突
```bash
poetry init -n
poetry env use "path to your python.exe"
poetry run pip install -r requirments.txt
```

### MacOS
可以通過 `run.sh` 來安裝依賴
```bash
sh ./run.sh
```

## 其他預模型準備
RVC需要其他一些預模型來推理和訓練。

你可以從我們的[Hugging Face space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)下載到這些模型。

### 1. 下載 assets
以下是一份清單，包括了所有RVC所需的預模型和其他文件的名稱。你可以在`tools`文件夾找到下載它們的腳本。

- ./assets/hubert/hubert_base.pt

- ./assets/pretrained 

- ./assets/uvr5_weights

想使用v2版本模型的話，需要額外下載

- ./assets/pretrained_v2

### 2. 安裝 ffmpeg
若ffmpeg和ffprobe已安裝則跳過。

#### Ubuntu/Debian 用戶
```bash
sudo apt install ffmpeg
```
#### MacOS 用戶
```bash
brew install ffmpeg
```
#### Windows 用戶
下載後放置在根目錄。
- 下載[ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe)

- 下載[ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe)

### 3. 下載 rmvpe 人聲音高提取算法所需文件

如果你想使用最新的RMVPE人聲音高提取算法，則你需要下載音高提取模型參數並放置於RVC根目錄。

- 下載[rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt)

#### 下載 rmvpe 的 dml 環境(可選, A卡/I卡用戶)

- 下載[rmvpe.onnx](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.onnx)

### 4. AMD顯示卡Rocm(可選, 僅Linux)

如果你想基於AMD的Rocm技術在Linux系統上運行RVC，請先在[這裡](https://rocm.docs.amd.com/en/latest/deploy/linux/os-native/install.html)安裝所需的驅動。

若你使用的是Arch Linux，可以使用pacman來安裝所需驅動：
````
pacman -S rocm-hip-sdk rocm-opencl-sdk
````
對於某些型號的顯示卡，你可能需要額外配置如下的環境變數（如：RX6700XT）：
````
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
````
同時確保你的當前用戶處於`render`與`video`用戶組內：
````
sudo usermod -aG render $USERNAME
sudo usermod -aG video $USERNAME
````

## 開始使用
### 直接啟動
使用以下指令來啟動 WebUI
```bash
python infer-web.py
```

若先前使用 Poetry 安裝依賴，則可以透過以下方式啟動WebUI
```bash
poetry run python infer-web.py
```

### 使用整合包
下載並解壓`RVC-beta.7z`
#### Windows 用戶
雙擊`go-web.bat`
#### MacOS 用戶
```bash
sh ./run.sh
```
### 對於需要使用IPEX技術的I卡用戶(僅Linux)
```bash
source /opt/intel/oneapi/setvars.sh
```

## 參考項目
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
+ [Vocal pitch extraction:RMVPE](https://github.com/Dream-High/RMVPE)
  + The pretrained model is trained and tested by [yxlllc](https://github.com/yxlllc/RMVPE) and [RVC-Boss](https://github.com/RVC-Boss).

## 感謝所有貢獻者作出的努力
<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>
