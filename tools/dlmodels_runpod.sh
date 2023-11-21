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


rmvpe="rmvpe.pt"

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

echo dir check finished.

echo required files check start.

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
