#!/bin/bash

echo working dir is $(pwd)
echo downloading requirement aria2 check.

if command -v aria2c &> /dev/null
then
    echo "aria2c command found"
else
    echo failed. please install aria2
    sleep 5
    exit 1
fi

d32="f0D32k.pth"
d40="f0D40k.pth"
d48="f0D48k.pth"
g32="f0G32k.pth"
g40="f0G40k.pth"
g48="f0G48k.pth"

d40v2="f0D40k.pth"
g40v2="f0G40k.pth"

dld32="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D32k.pth"
dld40="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D40k.pth"
dld48="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D48k.pth"
dlg32="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G32k.pth"
dlg40="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G40k.pth"
dlg48="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G48k.pth"

dld40v2="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth"
dlg40v2="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth"

hp2_all="HP2_all_vocals.pth"
hp3_all="HP3_all_vocals.pth"
hp5_only="HP5_only_main_vocal.pth"
VR_DeEchoAggressive="VR-DeEchoAggressive.pth"
VR_DeEchoDeReverb="VR-DeEchoDeReverb.pth"
VR_DeEchoNormal="VR-DeEchoNormal.pth"
onnx_dereverb="vocals.onnx"
rmvpe="rmvpe.pt"

dlhp2_all="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2_all_vocals.pth"
dlhp3_all="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP3_all_vocals.pth"
dlhp5_only="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5_only_main_vocal.pth"
dlVR_DeEchoAggressive="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoAggressive.pth"
dlVR_DeEchoDeReverb="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoDeReverb.pth"
dlVR_DeEchoNormal="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/VR-DeEchoNormal.pth"
dlonnx_dereverb="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx"
dlrmvpe="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"

hb="hubert_base.pt"

dlhb="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"

echo dir check start.

if [ -d "./assets/pretrained" ]; then
    echo dir ./assets/pretrained checked.
else
    echo failed. generating dir ./assets/pretrained.
    mkdir pretrained
fi

if [ -d "./assets/pretrained_v2" ]; then
    echo dir ./assets/pretrained_v2 checked.
else
    echo failed. generating dir ./assets/pretrained_v2.
    mkdir pretrained_v2
fi

if [ -d "./assets/uvr5_weights" ]; then
    echo dir ./assets/uvr5_weights checked.
else
    echo failed. generating dir ./assets/uvr5_weights.
    mkdir uvr5_weights
fi

if [ -d "./assets/uvr5_weights/onnx_dereverb_By_FoxJoy" ]; then
    echo dir ./assets/uvr5_weights/onnx_dereverb_By_FoxJoy checked.
else
    echo failed. generating dir ./assets/uvr5_weights/onnx_dereverb_By_FoxJoy.
    mkdir uvr5_weights/onnx_dereverb_By_FoxJoy
fi

echo dir check finished.

echo required files check start.

echo checking D32k.pth
if [ -f "./assets/pretrained/D32k.pth" ]; then
    echo D32k.pth in ./assets/pretrained checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D32k.pth -d ./assets/pretrained -o D32k.pth
        if [ -f "./assets/pretrained/D32k.pth" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking D40k.pth
if [ -f "./assets/pretrained/D40k.pth" ]; then
    echo D40k.pth in ./assets/pretrained checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D40k.pth -d ./assets/pretrained -o D40k.pth
        if [ -f "./assets/pretrained/D40k.pth" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking D40k.pth
if [ -f "./assets/pretrained_v2/D40k.pth" ]; then
    echo D40k.pth in ./assets/pretrained_v2 checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k.pth -d ./assets/pretrained_v2 -o D40k.pth
        if [ -f "./assets/pretrained_v2/D40k.pth" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking D48k.pth
if [ -f "./assets/pretrained/D48k.pth" ]; then
    echo D48k.pth in ./assets/pretrained checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D48k.pth -d ./assets/pretrained -o D48k.pth
        if [ -f "./assets/pretrained/D48k.pth" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking G32k.pth
if [ -f "./assets/pretrained/G32k.pth" ]; then
    echo G32k.pth in ./assets/pretrained checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G32k.pth -d ./assets/pretrained -o G32k.pth
        if [ -f "./assets/pretrained/G32k.pth" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking G40k.pth
if [ -f "./assets/pretrained/G40k.pth" ]; then
    echo G40k.pth in ./assets/pretrained checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G40k.pth -d ./assets/pretrained -o G40k.pth
        if [ -f "./assets/pretrained/G40k.pth" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking G40k.pth
if [ -f "./assets/pretrained_v2/G40k.pth" ]; then
    echo G40k.pth in ./assets/pretrained_v2 checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G40k.pth -d ./assets/pretrained_v2 -o G40k.pth
        if [ -f "./assets/pretrained_v2/G40k.pth" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking G48k.pth
if [ -f "./assets/pretrained/G48k.pth" ]; then
    echo G48k.pth in ./assets/pretrained checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G48k.pth -d ./assets/pretrained -o G48k.pth
        if [ -f "./assets/pretrained/G48k.pth" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking $d32
if [ -f "./assets/pretrained/$d32" ]; then
    echo $d32 in ./assets/pretrained checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M $dld32 -d ./assets/pretrained -o $d32
        if [ -f "./assets/pretrained/$d32" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking $d40
if [ -f "./assets/pretrained/$d40" ]; then
    echo $d40 in ./assets/pretrained checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M $dld40 -d ./assets/pretrained -o $d40
        if [ -f "./assets/pretrained/$d40" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking $d40v2
if [ -f "./assets/pretrained_v2/$d40v2" ]; then
    echo $d40v2 in ./assets/pretrained_v2 checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M $dld40v2 -d ./assets/pretrained_v2 -o $d40v2
        if [ -f "./assets/pretrained_v2/$d40v2" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking $d48
if [ -f "./assets/pretrained/$d48" ]; then
    echo $d48 in ./assets/pretrained checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M $dld48 -d ./assets/pretrained -o $d48
        if [ -f "./assets/pretrained/$d48" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking $g32
if [ -f "./assets/pretrained/$g32" ]; then
    echo $g32 in ./assets/pretrained checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M $dlg32 -d ./assets/pretrained -o $g32
        if [ -f "./assets/pretrained/$g32" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking $g40
if [ -f "./assets/pretrained/$g40" ]; then
    echo $g40 in ./assets/pretrained checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M $dlg40 -d ./assets/pretrained -o $g40
        if [ -f "./assets/pretrained/$g40" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking $g40v2
if [ -f "./assets/pretrained_v2/$g40v2" ]; then
    echo $g40v2 in ./assets/pretrained_v2 checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M $dlg40v2 -d ./assets/pretrained_v2 -o $g40v2
        if [ -f "./assets/pretrained_v2/$g40v2" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking $g48
if [ -f "./assets/pretrained/$g48" ]; then
    echo $g48 in ./assets/pretrained checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M $dlg48 -d ./assets/pretrained -o $g48
        if [ -f "./assets/pretrained/$g48" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking $hp2_all
if [ -f "./assets/uvr5_weights/$hp2_all" ]; then
    echo $hp2_all in ./assets/uvr5_weights checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M $dlhp2_all -d ./assets/uvr5_weights -o $hp2_all
        if [ -f "./assets/uvr5_weights/$hp2_all" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking $hp3_all
if [ -f "./assets/uvr5_weights/$hp3_all" ]; then
    echo $hp3_all in ./assets/uvr5_weights checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M $dlhp3_all -d ./assets/uvr5_weights -o $hp3_all
        if [ -f "./assets/uvr5_weights/$hp3_all" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking $hp5_only
if [ -f "./assets/uvr5_weights/$hp5_only" ]; then
    echo $hp5_only in ./assets/uvr5_weights checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M $dlhp5_only -d ./assets/uvr5_weights -o $hp5_only
        if [ -f "./assets/uvr5_weights/$hp5_only" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking $VR_DeEchoAggressive
if [ -f "./assets/uvr5_weights/$VR_DeEchoAggressive" ]; then
    echo $VR_DeEchoAggressive in ./assets/uvr5_weights checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M $dlVR_DeEchoAggressive -d ./assets/uvr5_weights -o $VR_DeEchoAggressive
        if [ -f "./assets/uvr5_weights/$VR_DeEchoAggressive" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking $VR_DeEchoDeReverb
if [ -f "./assets/uvr5_weights/$VR_DeEchoDeReverb" ]; then
    echo $VR_DeEchoDeReverb in ./assets/uvr5_weights checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M $dlVR_DeEchoDeReverb -d ./assets/uvr5_weights -o $VR_DeEchoDeReverb
        if [ -f "./assets/uvr5_weights/$VR_DeEchoDeReverb" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking $VR_DeEchoNormal
if [ -f "./assets/uvr5_weights/$VR_DeEchoNormal" ]; then
    echo $VR_DeEchoNormal in ./assets/uvr5_weights checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M $dlVR_DeEchoNormal -d ./assets/uvr5_weights -o $VR_DeEchoNormal
        if [ -f "./assets/uvr5_weights/$VR_DeEchoNormal" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking $onnx_dereverb
if [ -f "./assets/uvr5_weights/onnx_dereverb_By_FoxJoy/$onnx_dereverb" ]; then
    echo $onnx_dereverb in ./assets/uvr5_weights/onnx_dereverb_By_FoxJoy checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M $dlonnx_dereverb -d ./assets/uvr5_weights/onnx_dereverb_By_FoxJoy -o $onnx_dereverb
        if [ -f "./assets/uvr5_weights/onnx_dereverb_By_FoxJoy/$onnx_dereverb" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking $rmvpe
if [ -f "./assets/rmvpe/$rmvpe" ]; then
    echo $rmvpe in ./assets/rmvpe checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M $dlrmvpe -d ./assets/rmvpe -o $rmvpe
        if [ -f "./assets/rmvpe/$rmvpe" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo checking $hb
if [ -f "./assets/hubert/$hb" ]; then
    echo $hb in ./assets/hubert/pretrained checked.
else
    echo failed. starting download from huggingface.
    if command -v aria2c &> /dev/null; then
        aria2c --console-log-level=error -c -x 16 -s 16 -k 1M $dlhb -d ./assets/hubert/ -o $hb
        if [ -f "./assets/hubert/$hb" ]; then
            echo download successful.
        else
            echo please try again!
            exit 1
        fi
    else
        echo aria2c command not found. Please install aria2c and try again.
        exit 1
    fi
fi

echo required files check finished.
