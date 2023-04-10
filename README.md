<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
一个基于VITS的简单易用的语音转换（变声器）框架。<br><br>

[![madewithlove](https://forthebadge.com/images/badges/built-with-love.svg)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI)

<br>

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/liujing04/Retrieval-based-Voice-Conversion-WebUI/blob/main/Retrieval_based_Voice_Conversion_WebUI.ipynb)
[![Licence](https://img.shields.io/github/license/liujing04/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/blob/main/%E4%BD%BF%E7%94%A8%E9%9C%80%E9%81%B5%E5%AE%88%E7%9A%84%E5%8D%8F%E8%AE%AE-LICENSE.txt)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-blue.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

</div>

------

[**更新日志**](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/blob/main/Changelog_CN.md)

[**English**](./README_en.md) | [**中文简体**](./README.md)

> 点此查看我们的[演示视频](https://www.bilibili.com/video/BV1pm4y1z7Gm/) !

> 使用了RVC的实时语音转换: [w-okada/voice-changer](https://github.com/w-okada/voice-changer)

## 简介
本仓库具有以下特点:
+ 使用top1特征模型检索来杜绝音色泄漏；
+ 即便在相对较差的显卡上也能快速训练;
+ 使用少量数据进行训练也能得到较好结果;
+ 可以通过模型融合来改变音色;
+ 简单易用的WebUI界面;
+ 可调用UVR5模型来快速分离人声和伴奏。
+ 底模训练集使用接近50小时的高质量VCTK开源，后续会陆续加入高质量有授权歌声训练集供大家放心使用。
## 环境配置
我们推荐你使用poetry来配置环境。

以下指令需在Python版本大于3.8的环境当中执行:
```bash
# 安装Pytorch及其核心依赖，若已安装则跳过
# 参考自: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

如果是win系统+30系显卡，根据https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/issues/21的经验，需要指定pytorch对应的cuda版本

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# 安装 Poetry 依赖管理工具, 若已安装则跳过
# 参考自: https://python-poetry.org/docs/#installation
curl -sSL https://install.python-poetry.org | python3 -

# 通过poetry安装依赖
poetry install
```

你也可以通过pip来安装依赖：

**注意**: `MacOS`下`faiss 1.7.2`版本会导致抛出段错误，请将`requirements.txt`的对应条目改为`faiss-cpu==1.7.0`

```bash
pip install -r requirements.txt
```

## 其他预模型准备
RVC需要其他的一些预模型来推理和训练。

你可以从我们的[Huggingface space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)下载到这些模型。

以下是一份清单，包括了所有RVC所需的预模型和其他文件的名称:
```bash
hubert_base.pt

./pretrained 

./uvr5_weights

#如果你正在使用Windows，则你可能需要这个文件夹，若FFmpeg已安装则跳过
./ffmpeg
```
之后使用以下指令来调用Webui:
```bash
python infer-web.py
```
如果你正在使用Windows，你可以直接下载并解压`RVC-beta.7z` 来使用RVC，运行`go-web.bat`来启动WebUI。

我们将在两周内推出一个英文版本的WebUI.

仓库内还有一份`小白简易教程.doc`以供参考。

## 参考项目
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
## 感谢所有贡献者作出的努力
<a href="https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=liujing04/Retrieval-based-Voice-Conversion-WebUI" />
</a>

