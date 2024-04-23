<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
一个基于VITS的简单易用的变声框架<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange
)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>

[![RVC v1](https://img.shields.io/badge/RVCv1-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/tools/ipynb/v1.ipynb)
[![RVC v2](https://img.shields.io/badge/RVCv2-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/tools/ipynb/v2.ipynb)
[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

[**更新日志**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/Changelog_CN.md) | [**常见问题解答**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E8%A7%A3%E7%AD%94) | [**AutoDL·5毛钱训练AI歌手**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/Autodl%E8%AE%AD%E7%BB%83RVC%C2%B7AI%E6%AD%8C%E6%89%8B%E6%95%99%E7%A8%8B) | [**对照实验记录**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/Autodl%E8%AE%AD%E7%BB%83RVC%C2%B7AI%E6%AD%8C%E6%89%8B%E6%95%99%E7%A8%8B](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/%E5%AF%B9%E7%85%A7%E5%AE%9E%E9%AA%8C%C2%B7%E5%AE%9E%E9%AA%8C%E8%AE%B0%E5%BD%95)) | [**在线演示**](https://modelscope.cn/studios/FlowerCry/RVCv2demo)

[**English**](./docs/en/README.en.md) | [**中文简体**](./README.md) | [**日本語**](./docs/jp/README.ja.md) | [**한국어**](./docs/kr/README.ko.md) ([**韓國語**](./docs/kr/README.ko.han.md)) | [**Français**](./docs/fr/README.fr.md) | [**Türkçe**](./docs/tr/README.tr.md) | [**Português**](./docs/pt/README.pt.md)

</div>

> 底模使用接近50小时的开源高质量VCTK训练集训练，无版权方面的顾虑，请大家放心使用

> 请期待RVCv3的底模，参数更大，数据集更大，效果更好，基本持平的推理速度，需要训练数据量更少。

> 由于某些地区无法直连Hugging Face，即使设法成功访问，速度也十分缓慢，特推出模型/整合包/工具的一键下载器，欢迎试用：[RVC-Models-Downloader](https://github.com/RVC-Project/RVC-Models-Downloader)

<table>
   <tr>
		<td align="center">训练推理界面</td>
		<td align="center">实时变声界面</td>
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
    <td align="center">可以自由选择想要执行的操作。</td>
	<td align="center">我们已经实现端到端170ms延迟。如使用ASIO输入输出设备，已能实现端到端90ms延迟，但非常依赖硬件驱动支持。</td>
	</tr>
</table>

## 简介
本仓库具有以下特点
+ 使用top1检索替换输入源特征为训练集特征来杜绝音色泄漏
+ 即便在相对较差的显卡上也能快速训练
+ 使用少量数据进行训练也能得到较好结果(推荐至少收集10分钟低底噪语音数据)
+ 可以通过模型融合来改变音色(借助ckpt处理选项卡中的ckpt-merge)
+ 简单易用的网页界面
+ 可调用UVR5模型来快速分离人声和伴奏
+ 使用最先进的[人声音高提取算法InterSpeech2023-RMVPE](#参考项目)根绝哑音问题，效果更好，运行更快，资源占用更少
+ A卡I卡加速支持

点此查看我们的[演示视频](https://www.bilibili.com/video/BV1pm4y1z7Gm/) !

## 环境配置
### Python 版本限制
> 建议使用 conda 管理 Python 环境

> 版本限制原因参见此[bug](https://github.com/facebookresearch/fairseq/issues/5012)

```bash
python --version # 3.8 <= Python < 3.11
```

### Linux/MacOS 一键依赖安装启动脚本
执行项目根目录下`run.sh`即可一键配置`venv`虚拟环境、自动安装所需依赖并启动主程序。
```bash
sh ./run.sh
```

### 手动安装依赖
1. 安装`pytorch`及其核心依赖，若已安装则跳过。参考自: https://pytorch.org/get-started/locally/
	```bash
	pip install torch torchvision torchaudio
	```
2. 如果是 win 系统 + Nvidia Ampere 架构(RTX30xx)，根据 #21 的经验，需要指定 pytorch 对应的 CUDA 版本
	```bash
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
	```
3. 根据自己的显卡安装对应依赖
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

## 其他资源准备
### 1. assets
> RVC需要位于`assets`文件夹下的一些模型资源进行推理和训练。
#### 自动检查/下载资源(默认)
> 默认情况下，RVC可在主程序启动时自动检查所需资源的完整性。

> 即使资源不完整，程序也将继续启动。

- 如果您希望下载所有资源，请添加`--update`参数
- 如果您希望跳过启动时的资源完整性检查，请添加`--nocheck`参数

#### 手动下载资源
> 所有资源文件均位于[Hugging Face space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

> 你可以在`tools`文件夹找到下载它们的脚本

> 你也可以使用模型/整合包/工具的一键下载器：[RVC-Models-Downloader](https://github.com/RVC-Project/RVC-Models-Downloader)

以下是一份清单，包括了所有RVC所需的预模型和其他文件的名称。

- ./assets/hubert/hubert_base.pt
	```bash
	rvcmd assets/hubert # RVC-Models-Downloader command
	```
- ./assets/pretrained
	```bash
	rvcmd assets/v1 # RVC-Models-Downloader command
	```
- ./assets/uvr5_weights
	```bash
	rvcmd assets/uvr5 # RVC-Models-Downloader command
	```
想使用v2版本模型的话，需要额外下载

- ./assets/pretrained_v2
	```bash
	rvcmd assets/v2 # RVC-Models-Downloader command
	```

### 2. 安装 ffmpeg 工具
若已安装`ffmpeg`和`ffprobe`则可跳过此步骤。

#### Ubuntu/Debian 用户
```bash
sudo apt install ffmpeg
```
#### MacOS 用户
```bash
brew install ffmpeg
```
#### Windows 用户
下载后放置在根目录。
```bash
rvcmd tools/ffmpeg # RVC-Models-Downloader command
```
- 下载[ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe)

- 下载[ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe)

### 3. 下载 rmvpe 人声音高提取算法所需文件

如果你想使用最新的RMVPE人声音高提取算法，则你需要下载音高提取模型参数并放置于`assets/rmvpe`。

- 下载[rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt)
	```bash
	rvcmd assets/rmvpe # RVC-Models-Downloader command
	```

#### 下载 rmvpe 的 dml 环境(可选, A卡/I卡用户)

- 下载[rmvpe.onnx](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.onnx)
	```bash
	rvcmd assets/rmvpe # RVC-Models-Downloader command
	```

### 4. AMD显卡Rocm(可选, 仅Linux)

如果你想基于AMD的Rocm技术在Linux系统上运行RVC，请先在[这里](https://rocm.docs.amd.com/en/latest/deploy/linux/os-native/install.html)安装所需的驱动。

若你使用的是Arch Linux，可以使用pacman来安装所需驱动：
````
pacman -S rocm-hip-sdk rocm-opencl-sdk
````
对于某些型号的显卡，你可能需要额外配置如下的环境变量（如：RX6700XT）：
````
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
````
同时确保你的当前用户处于`render`与`video`用户组内：
````
sudo usermod -aG render $USERNAME
sudo usermod -aG video $USERNAME
````

## 开始使用
### 直接启动
使用以下指令来启动 WebUI
```bash
python infer-web.py
```
### Linux/MacOS 用户
```bash
./run.sh
```
### 对于需要使用IPEX技术的I卡用户(仅Linux)
```bash
source /opt/intel/oneapi/setvars.sh
./run.sh
```
### 使用整合包 (Windows 用户)
下载并解压`RVC-beta.7z`，解压后双击`go-web.bat`即可一键启动。
```bash
rvcmd packs/general/latest # RVC-Models-Downloader command
```

## 参考项目
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
+ [Vocal pitch extraction:RMVPE](https://github.com/Dream-High/RMVPE)
  + The pretrained model is trained and tested by [yxlllc](https://github.com/yxlllc/RMVPE) and [RVC-Boss](https://github.com/RVC-Boss).

## 感谢所有贡献者作出的努力
<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>
