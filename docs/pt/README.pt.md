<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
Uma estrutura de conversão de voz fácil de usar baseada em VITS.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange
)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>
  
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/Retrieval_based_Voice_Conversion_WebUI.ipynb)
[![Licence](https://img.shields.io/github/license/RVC-Project/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

</div>

------
[**Changelog**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/Changelog_EN.md) | [**FAQ (Frequently Asked Questions)**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/FAQ-(Frequently-Asked-Questions)) 

[**English**](../en/README.en.md) | [**中文简体**](../../README.md) | [**日本語**](../jp/README.ja.md) | [**한국어**](../kr/README.ko.md) ([**韓國語**](../kr/README.ko.han.md)) | [**Türkçe**](../tr/README.tr.md) | [**Português**](../pt/README.pt.md)


Confira nosso [Vídeo de demonstração](https://www.bilibili.com/video/BV1pm4y1z7Gm/) aqui!

Treinamento/Inferência WebUI：go-web.bat
![Traduzido](https://github.com/RafaelGodoyEbert/Retrieval-based-Voice-Conversion-WebUI/assets/78083427/0b894d87-565a-432c-8b5b-45e4a65d5d17)

GUI de conversão de voz em tempo real：go-realtime-gui.bat
![image](https://github.com/RafaelGodoyEbert/Retrieval-based-Voice-Conversion-WebUI/assets/78083427/d172e3e5-35f4-4876-9530-c28246919e9e)


> O dataset para o modelo de pré-treinamento usa quase 50 horas de conjunto de dados de código aberto VCTK de alta qualidade.

> Dataset de músicas licenciadas de alta qualidade serão adicionados ao conjunto de treinamento, um após o outro, para seu uso, sem se preocupar com violação de direitos autorais.

> Aguarde o modelo básico pré-treinado do RVCv3, que possui parâmetros maiores, mais dados de treinamento, melhores resultados, velocidade de inferência inalterada e requer menos dados de treinamento para treinamento.

## Resumo
Este repositório possui os seguintes recursos:
+ Reduza o vazamento de tom substituindo o recurso de origem pelo recurso de conjunto de treinamento usando a recuperação top1;
+ Treinamento fácil e rápido, mesmo em placas gráficas relativamente ruins;
+ Treinar com uma pequena quantidade de dados também obtém resultados relativamente bons (>=10min de áudio com baixo ruído recomendado);
+ Suporta fusão de modelos para alterar timbres (usando guia de processamento ckpt-> mesclagem ckpt);
+ Interface Webui fácil de usar;
+ Use o modelo UVR5 para separar rapidamente vocais e instrumentos.
+ Use o mais poderoso algoritmo de extração de voz de alta frequência [InterSpeech2023-RMVPE](#Credits) para evitar o problema de som mudo. Fornece os melhores resultados (significativamente) e é mais rápido, com consumo de recursos ainda menor que o Crepe_full.
+ Suporta aceleração de placas gráficas AMD/Intel.
+ Aceleração de placas gráficas Intel ARC com suporte para IPEX.

## Preparando o ambiente
Os comandos a seguir precisam ser executados no ambiente Python versão 3.8 ou superior.

(Windows/Linux)
Primeiro instale as dependências principais através do pip:
```bash
# Instale as dependências principais relacionadas ao PyTorch, pule se instaladas
# Referência: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

#Para arquitetura Windows + Nvidia Ampere (RTX30xx), você precisa especificar a versão cuda correspondente ao pytorch de acordo com a experiência de https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/ 21
#pip instalar tocha torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

#Para placas Linux + AMD, você precisa usar as seguintes versões do pytorch:
#pip instalar tocha torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```

Então pode usar poesia para instalar as outras dependências:
```bash
# Instale a ferramenta de gerenciamento de dependências Poetry, pule se instalada
# Referência: https://python-poetry.org/docs/#installation
curl -sSL https://install.python-poetry.org | python3 -

#Instale as dependências do projeto
poetry install
```

Você também pode usar pip para instalá-los:
```bash

for Nvidia graphics cards
  pip install -r requirements.txt

for AMD/Intel graphics cards on Windows (DirectML)：
  pip install -r requirements-dml.txt

for Intel ARC graphics cards on Linux / WSL using Python 3.10: 
  pip install -r requirements-ipex.txt

for AMD graphics cards on Linux (ROCm):
  pip install -r requirements-amd.txt
```

------
Usuários de Mac podem instalar dependências via `run.sh`:
```bash
sh ./run.sh
```

## Preparação de outros Pré-modelos
RVC requer outros pré-modelos para inferir e treinar.

```bash
#Baixe todos os modelos necessários em https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/
python tools/download_models.py
```

Ou apenas baixe-os você mesmo em nosso [Huggingface space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/).

Aqui está uma lista de pré-modelos e outros arquivos que o RVC precisa:
```bash
./assets/hubert/hubert_base.pt

./assets/pretrained 

./assets/uvr5_weights

Downloads adicionais são necessários se você quiser testar a versão v2 do modelo.

./assets/pretrained_v2

Se você deseja testar o modelo da versão v2 (o modelo da versão v2 alterou a entrada do recurso dimensional 256 do Hubert + final_proj de 9 camadas para o recurso dimensional 768 do Hubert de 12 camadas e adicionou 3 discriminadores de período), você precisará baixar recursos adicionais

./assets/pretrained_v2

#Se você estiver usando Windows, também pode precisar desses dois arquivos, pule se FFmpeg e FFprobe estiverem instalados
ffmpeg.exe

https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe

ffprobe.exe

https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe

Se quiser usar o algoritmo de extração de tom vocal SOTA RMVPE mais recente, você precisa baixar os pesos RMVPE e colocá-los no diretório raiz RVC

https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt

    Para usuários de placas gráficas AMD/Intel, você precisa baixar:

    https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.onnx

```

Os usuários de placas gráficas Intel ARC precisam executar o comando `source /opt/intel/oneapi/setvars.sh` antes de iniciar o Webui.

Em seguida, use este comando para iniciar o Webui:
```bash
python infer-web.py
```

Se estiver usando Windows ou macOS, você pode baixar e extrair `RVC-beta.7z` para usar RVC diretamente usando `go-web.bat` no Windows ou `sh ./run.sh` no macOS para iniciar o Webui.

## Suporte ROCm para placas gráficas AMD (somente Linux)
Para usar o ROCm no Linux, instale todos os drivers necessários conforme descrito [aqui](https://rocm.docs.amd.com/en/latest/deploy/linux/os-native/install.html).

No Arch use pacman para instalar o driver:
````
pacman -S rocm-hip-sdk rocm-opencl-sdk
````

Talvez você também precise definir estas variáveis de ambiente (por exemplo, em um RX6700XT):
````
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
````
Verifique também se seu usuário faz parte do grupo `render` e `video`:
````
sudo usermod -aG render $USERNAME
sudo usermod -aG video $USERNAME
````
Depois disso, você pode executar o WebUI:
```bash
python infer-web.py
```

## Credits
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
+ [Vocal pitch extraction:RMVPE](https://github.com/Dream-High/RMVPE)
  + The pretrained model is trained and tested by [yxlllc](https://github.com/yxlllc/RMVPE) and [RVC-Boss](https://github.com/RVC-Boss).
  
## Thanks to all contributors for their efforts
<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>

