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

---

[**English**](./docs/README.en.md) | [**中文简体**](./README.md) | [**日本語**](./docs/README.ja.md) | [**한국어**](./docs/README.ko.md) ([**韓國語**](./docs/README.ko.han.md))

点此查看我们的[演示视频](https://www.bilibili.com/video/BV1pm4y1z7Gm/) !


> 使用了 RVC 的实时语音转换: [w-okada/voice-changer](https://github.com/w-okada/voice-changer)


> 使用了RVC变声器训练的人声转木吉他模型在线demo ：https://huggingface.co/spaces/lj1995/vocal2guitar

> RVC人声转吉他效果展示视频 ：https://www.bilibili.com/video/BV19W4y1D7tT/

> 底模使用接近50小时的开源高质量VCTK训练集训练，无版权方面的顾虑，请大家放心使用



> 后续会陆续加入高质量有授权歌声训练集训练底模

## 简介

**此为fork仓库，相较于原版，不会进行过多的更新**
但是这个仓库有命令行工具，方便进行批量调用或者不方便查看WebUI的机器上运行

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


推荐使用 poetry 配置环境。

以下指令需在 Python 版本大于 3.8 的环境中执行:

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

你也可以通过 pip 来安装依赖：

```bash
pip install -r requirements.txt
```


> 请提前安装好对应版本的 torch 以及 torchaudio、torchvision

**注意**

1. 英特尔`MacOS`下使用 pip 安装`faiss 1.7.0`以上版本会导致抛出段错误，在手动安装时，如需安装最新版，请使用`conda`；如只能使用`pip`，请指定使用`1.7.0`版本。
2. `MacOS`下如`faiss`安装失败，可尝试通过`brew`安装`Swig`

```bash
brew install swig
```

## 其他预模型准备

RVC 需要其他一些预模型来推理和训练。

你可以从我们的[Hugging Face space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)下载到这些模型。

以下是一份清单，包括了所有 RVC 所需的预模型和其他文件的名称:

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

之后使用以下指令来启动 WebUI:

```bash
python infer-web.py
```

如果你正在使用 Windows，你可以直接下载并解压 `RVC-beta.7z`，运行 `go-web.bat` 以启动 WebUI。

仓库内还有一份 `小白简易教程.doc` 以供参考。

---


## 无 WebUI 推理(CLI)

示例命令:

```bash
python -O infer_cli.py .\raw\whd.wav -i .\logs\nahida\added_IVF3204_Flat_nprobe_1_v1.index -f harvest -o .\output\whd_nahida_e200.wav -m .\weights\nahida_e200.pth  -ir 0.7 -d cuda:0 -fp
```

简化版：

```bash
python -O infer_cli.py.\raw\whd.wav -i .\logs\nahida\added_IVF3104_Flat_nprobe_1_v1.index -m .\weights\nahida_e200.pth  -ir 0.7 -fp
```

> `-O` 参数是防止程序无法运行加上的，最好别省略

帮助:

```bash
python infer_cli.py -h
```

参数:

```bash
usage: infer_cli.py [-h] [-m MODEL] [-i INDEX] [-f F0METHOD] [-k KEY] [-ir INDEX_RATE] [-d DEVICE] [-fp] [-o OUTPUT] [-fr FILTER_RADIUS] [-s TGT_SR] [-rs RESAMPLE_SR] [-rms RMS_MIX_RATE] [-v VERSION] input_audio

positional arguments:
  input_audio           输入的音频文件路径

options:
  -h,   --help          show this help message and exit
  -m    --model
                        \weights路径下的pth文件 必填，无默认值
  -i    --index
                        索引文件路径 必填，无默认值
  -f    --f0method
                        f0的方法，可选‘pm’ ‘harvest’ 默认harvest
  -k    --key           升降调 默认0
  -ir   --index_rate
                        索引比例（取值0~1，越趋近于1理论上音色泄露更少）默认0.7
  -d    --device
                        使用的设备，默认 cuda:0 输入格式 <device_name>:<device_id>
  -fp,  --is_half       是否使用半精度运算 如果添加该flag，则使用fp16
                        半精度训练；若没有，则使用fp32单精度模式
  -o    --output        输出路径，可以不填写 若不填写则自动生成
  -fr   --filter_radius
                        对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音 默认3
  -s    --tgt_sr
                        目标采样率 默认48000
  -rs   --resample_sr
                        后处理重采样至最终采样率，0为不进行重采样 默认0
  -rms  --rms_mix_rate
                        输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络 默认1
  -v    --version
                        模型版本, 默认为 v1
```

### 特殊说明

输出文件路径可不填, 若不指定则会将生成的文件按照如下规则命名并置于`output/`文件夹中。

```
output/<input_filename>_<model_name>_<index_rate>_<f0up_keys>_<f0method>_<time>.wav
```

## 无 WebUI 训练（CLI）

示例命令：
```bash
python train_cli.py preprocess D:\Data\Retrieval-based-Voice-Conversion-WebUI\dataset\nahida_main nahida    # 预处理数据

python train_cli.py f0 D:\Data\Retrieval-based-Voice-Conversion-WebUI\dataset\nahida_main nahida    # 提取f0

python train_cli.py train D:\Data\Retrieval-based-Voice-Conversion-WebUI\dataset\nahida_main nahida    # 正式训练
```
 
帮助：

```bash
usage: train_cli.py [-h] [-sr SAMPLE_RATE] [-np N_P] [-gpus GPUS] [-if IF_F0] [-v {v1,v2}] [-f {pm,harvest,dio}] [-s SPK_ID] [-se SAVE_EPOCH] [-e TOTAL_EPOCH] [-bs BATCH_SIZE] [-sl IF_SAVE_LATEST] [-cg IF_CACHE_GPU]
                    [-sew IF_SAVE_EVERY_WEIGHTS] [-pg PRETRAINED_G] [-pd PRETRAINED_D]
                    {preprocess,f0,train} dataset_path exp_dir

positional arguments:
  dataset_path          数据集路径
  exp_dir               实验名称
  mode                  执行任务的类型 可选：preprocess f0 train

options:
  -h, --help            show this help message and exit
  -sr SAMPLE_RATE, --sample_rate SAMPLE_RATE
                        可选值32k 40k 48k 默认48k
  -np N_P, --n_p N_P    number of process cpus 可用于处理的cpu数量。默认为所有可用的cpu
  -gpus GPUS, --gpus GPUS
                        显卡选择 格式：0-1-2 表示使用卡0和卡1和卡2 默认0
  -if IF_F0, --if_f0 IF_F0
                        模型是否带音高指导(唱歌一定要, 语音可以不要) 默认True
  -v {v1,v2}, --version {v1,v2}
                        版本(目前仅40k支持了v2) 默认v1
  -f {pm,harvest,dio}, --f0method {pm,harvest,dio}
                        选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢 默认harvest
  -s SPK_ID, --spk_id SPK_ID
                        speaker id 默认0（不要更改）
  -se SAVE_EPOCH, --save_epoch SAVE_EPOCH
                        保存频率 默认5
  -e TOTAL_EPOCH, --total_epoch TOTAL_EPOCH
                        总训练轮数total_epoch 默认40
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        每张显卡的batch_size 默认值会根据显存大小自动计算。
  -sl IF_SAVE_LATEST, --if_save_latest IF_SAVE_LATEST
                        是否仅保存最新的ckpt文件以节省硬盘空间 默认True
  -cg IF_CACHE_GPU, --if_cache_gpu IF_CACHE_GPU
                        是否缓存所有训练集至显存. 10min以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速 默认False
  -sew IF_SAVE_EVERY_WEIGHTS, --if_save_every_weights IF_SAVE_EVERY_WEIGHTS
                        是否在每次保存时间点将最终小模型保存至weights文件夹 默认True
  -pg PRETRAINED_G, --pretrained_g PRETRAINED_G
                        加载预训练底模G路径 默认pretrained/f0G48k.pth
  -pd PRETRAINED_D, --pretrained_d PRETRAINED_D
                        加载预训练底模D路径 默认pretrained/f0D48k.pth

```


其中，所有的可选参数都已经设定了默认值，只需要设定数据集路径和实验名称即可。

### 关于训练的Tips

+ 直接把一整个长的wav文件放在指定目录即可，程序会自动分片并重新采样
+ 你可能需要改的参数：
  + pretrained_d 和 pretrained_g 将它们改成和采样率相匹配的名称：如32k填入pretrained/f0D32k.pth
  + total_epoch 总共训练轮数。通常情况下40轮即可出效果，如果训练集数据质量高，可以调大至200epoch以减少可能的音色泄露，并可能解决生成出来的声音中电流声的问题
  + save_epoch 一定要设置成 total_epoch 的某一个因数！！！不然你的模型训练完了最后的一个epoch但是你会发现没有保存下来的文件！！


## 关于此项目的Tips

- 最终的模型文件为 `weights/` 目录下的 pth 文件。`logs/实验名称/` 下的 pth 文件只用于记录训练状态，不可直接使用。
- 如果你的训练运行完成但是没有在 `weights/` 目录下生成相应的模型文件，请使用 webui 中的 `ckpt处理` 选项卡中 `模型提取` 功能生成最终的模型。
- **使用之前一定记得先看看**[**常见问题解答**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/常见问题解答)
- 推荐将数据集保存在`dataset/实验名称/`下，将推理的输入音频放在`raw/`目录下。

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
