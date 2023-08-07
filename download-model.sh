#!/bin/bash

git clone https://huggingface.co/lj1995/VoiceConversionWebUI

cp -r VoiceConversionWebUI/pretrained/* pretrained
cp -r VoiceConversionWebUI/pretrained_v2/* pretrained_v2
cp -r VoiceConversionWebUI/uvr5_weights/* uvr5_weights
cp -r VoiceConversionWebUI/hubert_base.pt .
cp -r VoiceConversionWebUI/rmvpe.pt .

rm -rf VoiceConversionWebUI