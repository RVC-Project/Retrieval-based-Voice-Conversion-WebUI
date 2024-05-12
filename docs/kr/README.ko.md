<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
VITS 기반의 간단하고 사용하기 쉬운 음성 변환 프레임워크.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>

[![RVC v1](https://img.shields.io/badge/RVCv1-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/tools/ipynb/v1.ipynb)
[![RVC v2](https://img.shields.io/badge/RVCv2-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/tools/ipynb/v2.ipynb)
[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

[**업데이트 로그**](./Changelog_KO.md) | [**자주 묻는 질문**](./faq_ko.md) | [**AutoDL·5원으로 AI 가수 훈련**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/Autodl%E8%AE%AD%E7%BB%83RVC%C2%B7AI%E6%AD%8C%E6%89%8B%E6%95%99%E7%A8%8B) | [**대조 실험 기록**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/%E5%AF%B9%E7%85%A7%E5%AE%9E%E9%AA%8C%C2%B7%E5%AE%9E%E9%AA%8C%E8%AE%B0%E5%BD%95) | [**온라인 데모**](https://modelscope.cn/studios/FlowerCry/RVCv2demo)

[**English**](../en/README.en.md) | [**中文简体**](../../README.md) | [**日本語**](../jp/README.ja.md) | [**한국어**](../kr/README.ko.md) ([**韓國語**](../kr/README.ko.han.md)) | [**Français**](../fr/README.fr.md) | [**Türkçe**](../tr/README.tr.md) | [**Português**](../pt/README.pt.md)

</div>

> 기본 모델은 50시간 가량의 고퀄리티 오픈 소스 VCTK 데이터셋을 사용하였으므로, 저작권상의 염려가 없으니 안심하고 사용하시기 바랍니다.

> 더 큰 매개변수, 더 큰 데이터, 더 나은 효과, 기본적으로 동일한 추론 속도, 더 적은 양의 훈련 데이터가 필요한 RVCv3의 기본 모델을 기대해 주십시오.

> 특정 지역에서 Hugging Face에 직접 연결할 수 없는 경우가 있으며, 성공적으로 연결해도 속도가 매우 느릴 수 있으므로, 모델/통합 패키지/도구의 일괄 다운로더를 특별히 소개합니다. [RVC-Models-Downloader](https://github.com/RVC-Project/RVC-Models-Downloader)

<table>
   <tr>
		<td align="center">훈련 및 추론 인터페이스</td>
		<td align="center">실시간 음성 변환 인터페이스</td>
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
    <td align="center">원하는 작업을 자유롭게 선택할 수 있습니다.</td>
		<td align="center">우리는 이미 끝에서 끝까지 170ms의 지연을 실현했습니다. ASIO 입력 및 출력 장치를 사용하면 끝에서 끝까지 90ms의 지연을 달성할 수 있지만, 이는 하드웨어 드라이버 지원에 매우 의존적입니다.</td>
	</tr>
</table>

## 소개

본 프로젝트는 다음과 같은 특징을 가지고 있습니다:

- top1 검색을 이용하여 입력 음색 특징을 훈련 세트 음색 특징으로 대체하여 음색의 누출을 방지
- 상대적으로 낮은 성능의 GPU에서도 빠른 훈련 가능
- 적은 양의 데이터로 훈련해도 좋은 결과를 얻을 수 있음 (최소 10분 이상의 저잡음 음성 데이터를 사용하는 것을 권장)
- 모델 융합을 통한 음색의 변조 가능 (ckpt 처리 탭->ckpt 병합 선택)
- 사용하기 쉬운 WebUI (웹 인터페이스)
- UVR5 모델을 이용하여 목소리와 배경음악의 빠른 분리;
- 최첨단 [음성 피치 추출 알고리즘 InterSpeech2023-RMVPE](#参考项目)을 사용하여 무성음 문제를 해결합니다. 효과는 최고(압도적)이며 crepe_full보다 더 빠르고 리소스 사용이 적음
- A카드와 I카드 가속을 지원

해당 프로젝트의 [데모 비디오](https://www.bilibili.com/video/BV1pm4y1z7Gm/)를 확인해보세요!

## 환경 설정

다음 명령은 Python 버전이 3.8 이상인 환경에서 실행해야 합니다.

### Windows/Linux/MacOS 등 플랫폼 공통 방법

아래 방법 중 하나를 선택하세요.

#### 1. pip를 통한 의존성 설치

1. Pytorch 및 의존성 모듈 설치, 이미 설치되어 있으면 생략. 참조: https://pytorch.org/get-started/locally/

```bash
pip install torch torchvision torchaudio
```

2. win 시스템 + Nvidia Ampere 아키텍처(RTX30xx) 사용 시, #21의 사례에 따라 pytorch에 해당하는 cuda 버전을 지정

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

3. 자신의 그래픽 카드에 맞는 의존성 설치

- N카드

```bash
pip install -r requirements.txt
```

- A카드/I카드

```bash
pip install -r requirements-dml.txt
```

- A카드ROCM(Linux)

```bash
pip install -r requirements-amd.txt
```

- I카드IPEX(Linux)

```bash
pip install -r requirements-ipex.txt
```

#### 2. poetry를 통한 의존성 설치

Poetry 의존성 관리 도구 설치, 이미 설치된 경우 생략. 참조: https://python-poetry.org/docs/#installation

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

poetry를 통한 의존성 설치

```bash
poetry install
```

### MacOS

`run.sh`를 통해 의존성 설치 가능

```bash
sh ./run.sh
```

<!--

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

translate to Korean
-->

## 기타 사전 훈련된 모델 준비

### assets

> RVC는 추론과 훈련을 위해 assets 폴더 하위에 사전 훈련된 모델이 필요합니다.

#### 자동 검사/다운로드 리소스(기본값)

> 기본적으로 RVC는 시작할 때 필요한 리소스의 무결성을 자동으로 확인할 수 있습니다.

> 리소스가 불완전하더라도 프로그램은 계속 실행됩니다.

- 모든 리소스를 다운로드하려면 `--update` 매개변수를 추가하세요
- 시작 시 리소스 무결성 검사를 건너뛰려면 `--nocheck` 매개변수를 추가하세요

#### 리소스 수동 다운로드

> 모든 리소스 파일은 [Hugging Face space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)에 있습니다.

> 이들을 다운로드하는 스크립트는 `tools` 폴더에서 찾을 수 있습니다.

> 모델/통합 패키지/도구의 일괄 다운로더를 사용할 수도 있습니다: [RVC-Models-Downloader](https://github.com/RVC-Project/RVC-Models-Downloader)

다음은 RVC에 필요한 모든 사전 훈련된 모델과 기타 파일의 목록입니다.

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

v2 버전 모델을 사용하려면 추가로 다음을 다운로드해야 합니다.

- ./assets/pretrained_v2
  ```bash
  rvcmd assets/v2 # RVC-Models-Downloader command
  ```

### 2. ffmpeg 설치

`ffmpeg`와 `ffprobe`가 이미 설치되어 있다면 건너뜁니다.

#### Ubuntu/Debian 사용자

```bash
sudo apt install ffmpeg
```

#### MacOS 사용자

```bash
brew install ffmpeg
```

#### Windows 사용자

다운로드 후 루트 디렉토리에 배치.

- [ffmpeg.exe 다운로드](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe)

- [ffprobe.exe 다운로드](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe)

### 3. RMVPE 인간 음성 피치 추출 알고리즘에 필요한 파일 다운로드

최신 RMVPE 인간 음성 피치 추출 알고리즘을 사용하려면 음피치 추출 모델 매개변수를 다운로드하고 RVC 루트 디렉토리에 배치해야 합니다.

- [rmvpe.pt 다운로드](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt)

#### dml 환경의 RMVPE 다운로드(선택사항, A카드/I카드 사용자)

- [rmvpe.onnx 다운로드](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.onnx)

### 4. AMD 그래픽 카드 Rocm(선택사항, Linux만 해당)

Linux 시스템에서 AMD의 Rocm 기술을 기반으로 RVC를 실행하려면 [여기](https://rocm.docs.amd.com/en/latest/deploy/linux/os-native/install.html)에서 필요한 드라이버를 먼저 설치하세요.

Arch Linux를 사용하는 경우 pacman을 사용하여 필요한 드라이버를 설치할 수 있습니다.

```
pacman -S rocm-hip-sdk rocm-opencl-sdk
```

일부 모델의 그래픽 카드(예: RX6700XT)의 경우, 다음과 같은 환경 변수를 추가로 설정해야 할 수 있습니다.

```
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

동시에 현재 사용자가 `render` 및 `video` 사용자 그룹에 속해 있는지 확인하세요.

```
sudo usermod -aG render $USERNAME
sudo usermod -aG video $USERNAME
```

## 시작하기

### 직접 시작

다음 명령어로 WebUI를 시작하세요

```bash
python infer-web.py
```

### 통합 패키지 사용

`RVC-beta.7z`를 다운로드하고 압축 해제

#### Windows 사용자

`go-web.bat` 더블 클릭

#### MacOS 사용자

```bash
sh ./run.sh
```

### IPEX 기술이 필요한 I카드 사용자를 위한 지침(Linux만 해당)

```bash
source /opt/intel/oneapi/setvars.sh
```

## 참조 프로젝트

- [ContentVec](https://github.com/auspicious3000/contentvec/)
- [VITS](https://github.com/jaywalnut310/vits)
- [HIFIGAN](https://github.com/jik876/hifi-gan)
- [Gradio](https://github.com/gradio-app/gradio)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [Vocal pitch extraction:RMVPE](https://github.com/Dream-High/RMVPE)
  - 사전 훈련된 모델은 [yxlllc](https://github.com/yxlllc/RMVPE)와 [RVC-Boss](https://github.com/RVC-Boss)에 의해 훈련되고 테스트되었습니다.

## 모든 기여자들의 노력에 감사드립니다

<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>
