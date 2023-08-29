Instructions and tips for RVC training
======================================
This TIPS explains how data training is done.

# Training flow
I will explain along the steps in the training tab of the GUI.

## step1
Set the experiment name here. 

You can also set here whether the model should take pitch into account.
If the model doesn't consider pitch, the model will be lighter, but not suitable for singing.

Data for each experiment is placed in `/logs/your-experiment-name/`.

## step2a
Loads and preprocesses audio.

### load audio
If you specify a folder with audio, the audio files in that folder will be read automatically.
For example, if you specify `C:Users\hoge\voices`, `C:Users\hoge\voices\voice.mp3` will be loaded, but `C:Users\hoge\voices\dir\voice.mp3` will Not loaded.

Since ffmpeg is used internally for reading audio, if the extension is supported by ffmpeg, it will be read automatically.
After converting to int16 with ffmpeg, convert to float32 and normalize between -1 to 1.

### denoising
The audio is smoothed by scipy's filtfilt.

### Audio Split
First, the input audio is divided by detecting parts of silence that last longer than a certain period (max_sil_kept=5 seconds?). After splitting the audio on silence, split the audio every 4 seconds with an overlap of 0.3 seconds. For audio separated within 4 seconds, after normalizing the volume, convert the wav file to `/logs/your-experiment-name/0_gt_wavs` and then convert it to 16k sampling rate to `/logs/your-experiment-name/1_16k_wavs ` as a wav file.

## step2b
### Extract pitch
Extract pitch information from wav files. Extract the pitch information (=f0) using the method built into parselmouth or pyworld and save it in `/logs/your-experiment-name/2a_f0`. Then logarithmically convert the pitch information to an integer between 1 and 255 and save it in `/logs/your-experiment-name/2b-f0nsf`.

### Extract feature_print
Convert the wav file to embedding in advance using HuBERT. Read the wav file saved in `/logs/your-experiment-name/1_16k_wavs`, convert the wav file to 256-dimensional features with HuBERT, and save in npy format in `/logs/your-experiment-name/3_feature256`.

## step3
train the model.
### Glossary for Beginners
In deep learning, the data set is divided and the learning proceeds little by little. In one model update (step), batch_size data are retrieved and predictions and error corrections are performed. Doing this once for a dataset counts as one epoch.

Therefore, the learning time is the learning time per step x (the number of data in the dataset / batch size) x the number of epochs. In general, the larger the batch size, the more stable the learning becomes (learning time per step ÷ batch size) becomes smaller, but it uses more GPU memory. GPU RAM can be checked with the nvidia-smi command. Learning can be done in a short time by increasing the batch size as much as possible according to the machine of the execution environment.

### Specify pretrained model
RVC starts training the model from pretrained weights instead of from 0, so it can be trained with a small dataset.

By default

- If you consider pitch, it loads `rvc-location/pretrained/f0G40k.pth` and `rvc-location/pretrained/f0D40k.pth`. 
- If you don't consider pitch, it loads `rvc-location/pretrained/f0G40k.pth` and `rvc-location/pretrained/f0D40k.pth`. 

When learning, model parameters are saved in `logs/your-experiment-name/G_{}.pth` and `logs/your-experiment-name/D_{}.pth` for each save_every_epoch, but by specifying this path, you can start learning. You can restart or start training from model weights learned in a different experiment.

### learning index
RVC saves the HuBERT feature values used during training, and during inference, searches for feature values that are similar to the feature values used during learning to perform inference. In order to perform this search at high speed, the index is learned in advance.
For index learning, we use the approximate neighborhood search library faiss. Read the feature value of `logs/your-experiment-name/3_feature256` and use it to learn the index, and save it as `logs/your-experiment-name/add_XXX.index`.

(From the 20230428update version, it is read from the index, and saving / specifying is no longer necessary.)

### Button description
- Train model: After executing step2b, press this button to train the model.
- Train feature index: After training the model, perform index learning.
- One-click training: step2b, model training and feature index training all at once.