#!/bin/sh

printf "working dir is %s\n" "$PWD"
echo "downloading requirement aria2 check."

if command -v aria2c > /dev/null 2>&1
then
    echo "aria2 command found"
else
    echo "failed. please install aria2"
    exit 1
fi

echo "dir check start."

check_dir() {
    [ -d "$1" ] && printf "dir %s checked\n" "$1" || \
    printf "failed. generating dir %s\n" "$1" && mkdir -p "$1"
}

check_dir "./assets/pretrained"
check_dir "./assets/pretrained_v2"
check_dir "./assets/uvr5_weights"
check_dir "./assets/uvr5_weights/onnx_dereverb_By_FoxJoy"

echo "dir check finished."

echo "required files check start."
check_file_pretrained() {
  printf "checking %s\n" "$2"
  if [ -f "./assets/""$1""/""$2""" ]; then
      printf "%s in ./assets/%s checked.\n" "$2" "$1" 
  else
      echo failed. starting download from huggingface.
      if command -v aria2c > /dev/null 2>&1; then
          aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"$1"/"$2" -d ./assets/"$1" -o "$2"
          [ -f "./assets/""$1""/""$2""" ] && echo "download successful." || { echo "please try again!" && exit 1; }
      else
          echo "aria2c command not found. Please install aria2c and try again."
          exit 1
      fi
  fi
}

check_file_special() {
  printf "checking %s\n" "$2"
  if [ -f "./assets/""$1""/""$2""" ]; then
      printf "%s in ./assets/%s checked.\n" "$2" "$1" 
  else
      echo failed. starting download from huggingface.
      if command -v aria2c > /dev/null 2>&1; then
          aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"$2" -d ./assets/"$1" -o "$2"
          [ -f "./assets/""$1""/""$2""" ] && echo "download successful." || { echo "please try again!" && exit 1; }
      else
          echo "aria2c command not found. Please install aria2c and try again."
          exit 1
      fi
  fi
}

check_file_pretrained pretrained D32k.pth
check_file_pretrained pretrained D40k.pth
check_file_pretrained pretrained D48k.pth
check_file_pretrained pretrained G32k.pth
check_file_pretrained pretrained G40k.pth
check_file_pretrained pretrained G48k.pth
check_file_pretrained pretrained_v2 f0D40k.pth
check_file_pretrained pretrained_v2 f0G40k.pth
check_file_pretrained pretrained_v2 D40k.pth
check_file_pretrained pretrained_v2 G40k.pth
check_file_pretrained uvr5_weights HP2_all_vocals.pth
check_file_pretrained uvr5_weights HP3_all_vocals.pth
check_file_pretrained uvr5_weights HP5_only_main_vocal.pth
check_file_pretrained uvr5_weights VR-DeEchoAggressive.pth
check_file_pretrained uvr5_weights VR-DeEchoDeReverb.pth
check_file_pretrained uvr5_weights VR-DeEchoNormal.pth
check_file_pretrained uvr5_weights "onnx_dereverb_By_FoxJoy/vocals.onnx"
check_file_special rmvpe rmvpe.pt
check_file_special hubert hubert_base.pt

echo "required files check finished."
