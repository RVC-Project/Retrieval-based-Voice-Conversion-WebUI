<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
一个基于VITS的简单易用的语音转换（变声器）框架<br><br>

[![madewithlove](https://forthebadge.com/images/badges/built-with-love.svg)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/Retrieval_based_Voice_Conversion_WebUI.ipynb)
[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

[**更新日志**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/Changelog_CN.md) | [**常见问题解答**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E8%A7%A3%E7%AD%94) | [**AutoDL·5毛钱训练AI歌手**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/Autodl%E8%AE%AD%E7%BB%83RVC%C2%B7AI%E6%AD%8C%E6%89%8B%E6%95%99%E7%A8%8B) | [**对照实验记录**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/Autodl%E8%AE%AD%E7%BB%83RVC%C2%B7AI%E6%AD%8C%E6%89%8B%E6%95%99%E7%A8%8B](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/%E5%AF%B9%E7%85%A7%E5%AE%9E%E9%AA%8C%C2%B7%E5%AE%9E%E9%AA%8C%E8%AE%B0%E5%BD%95)) | [**在线演示**](https://huggingface.co/spaces/Ricecake123/RVC-demo)

</div>

------

[**English**](./docs/README.en.md) | [**中文简体**](./README.md) | [**日本語**](./docs/README.ja.md) | [**한국어**](./docs/README.ko.md) ([**韓國語**](./docs/README.ko.han.md))

点此查看我们的[演示视频](https://www.bilibili.com/video/BV1pm4y1z7Gm/) !

> 使用了RVC的实时语音转换: [w-okada/voice-changer](https://github.com/w-okada/voice-changer)

> 使用了RVC变声器训练的人声转木吉他模型在线demo ：https://huggingface.co/spaces/lj1995/vocal2guitar

> RVC人声转吉他效果展示视频 ：https://www.bilibili.com/video/BV19W4y1D7tT/

> 底模使用接近50小时的开源高质量VCTK训练集训练，无版权方面的顾虑，请大家放心使用

> 后续会陆续加入高质量有授权歌声训练集训练底模

## 简介
本仓库具有以下特点
+ 使用top1检索替换输入源特征为训练集特征来杜绝音色泄漏
+ 即便在相对较差的显卡上也能快速训练
+ 使用少量数据进行训练也能得到较好结果(推荐至少收集10分钟低底噪语音数据)
+ 可以通过模型融合来改变音色(借助ckpt处理选项卡中的ckpt-merge)
+ 简单易用的网页界面
+ 可调用UVR5模型来快速分离人声和伴奏
+ 使用最先进的[人声音高提取算法InterSpeech2023-RMVPE](#参考项目)根绝哑音问题。效果最好（显著地）但比crepe_full更快、资源占用更小

## 环境配置
可以使用poetry配置环境。

以下指令需在Python版本大于3.8的环境中执行:
```bash
# 安装Pytorch及其核心依赖，若已安装则跳过
# 参考自: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

#如果是win系统+Nvidia Ampere架构(RTX30xx)，根据 #21 的经验，需要指定pytorch对应的cuda版本
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# 安装 Poetry 依赖管理工具, 若已安装则跳过
# 参考自: https://python-poetry.org/docs/#installation
curl -sSL https://install.python-poetry.org | python3 -

# 通过poetry安装依赖
poetry install
```

你也可以通过pip来安装依赖：
```bash
pip install -r requirements.txt
```

## 其他预模型准备
RVC需要其他一些预模型来推理和训练。

你可以从我们的[Hugging Face space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)下载到这些模型。

以下是一份清单，包括了所有RVC所需的预模型和其他文件的名称:
```bash
hubert_base.pt

./pretrained 

./uvr5_weights

想测试v2版本模型的话，需要额外下载

./pretrained_v2 

如果你正在使用Windows，则你可能需要这个文件，若ffmpeg和ffprobe已安装则跳过; ubuntu/debian 用户可以通过apt install ffmpeg来安装这2个库

./ffmpeg

https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe

./ffprobe

https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe

如果你想使用最新的RMVPE人声音高提取算法，则你需要下载音高提取模型参数并放置于RVC根目录

https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt

```
之后使用以下指令来启动WebUI:
```bash
python infer-web.py
```
如果你正在使用Windows，你可以直接下载并解压`RVC-beta.7z`，运行`go-web.bat`以启动WebUI。

仓库内还有一份`小白简易教程.doc`以供参考。

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
