## Q1:ffmpeg error/utf8 error.
It is most likely not a FFmpeg issue, but rather an audio path issue;

FFmpeg may encounter an error when reading paths containing special characters like spaces and (), which may cause an FFmpeg error; and when the training set's audio contains Chinese paths, writing it into filelist.txt may cause a utf8 error.<br>

## Q2:Cannot find index file after "One-click Training".
If it displays "Training is done. The program is closed," then the model has been trained successfully, and the subsequent errors are fake;

The lack of an 'added' index file after One-click training may be due to the training set being too large, causing the addition of the index to get stuck; this has been resolved by using batch processing to add the index, which solves the problem of memory overload when adding the index. As a temporary solution, try clicking the "Train Index" button again.<br>

## Q3:Cannot find the model in “Inferencing timbre” after training
Click “Refresh timbre list” and check again; if still not visible, check if there are any errors during training and send screenshots of the console, web UI, and logs/experiment_name/*.log to the developers for further analysis.<br>

## Q4:How to share a model/How to use others' models?
The pth files stored in rvc_root/logs/experiment_name are not meant for sharing or inference, but for storing the experiment checkpoits for reproducibility and further training. The model to be shared should be the 60+MB pth file in the weights folder;

In the future, weights/exp_name.pth and logs/exp_name/added_xxx.index will be merged into a single weights/exp_name.zip file to eliminate the need for manual index input; so share the zip file, not the pth file, unless you want to continue training on a different machine;

Copying/sharing the several hundred MB pth files from the logs folder to the weights folder for forced inference may result in errors such as missing f0, tgt_sr, or other keys. You need to use the ckpt tab at the bottom to manually or automatically (if the information is found in the logs/exp_name), select whether to include pitch infomation and target audio sampling rate options and then extract the smaller model. After extraction, there will be a 60+ MB pth file in the weights folder, and you can refresh the voices to use it.<br>

## Q5:Connection Error.
You may have closed the console (black command line window).<br>

## Q6:WebUI popup 'Expecting value: line 1 column 1 (char 0)'.
Please disable system LAN proxy/global proxy and then refresh.<br>

## Q7:How to train and infer without the WebUI?
Training script:<br>
You can run training in WebUI first, and the command-line versions of dataset preprocessing and training will be displayed in the message window.<br>

Inference script:<br>
https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/myinfer.py<br>


e.g.<br>

runtime\python.exe myinfer.py 0 "E:\codes\py39\RVC-beta\todo-songs\1111.wav" "E:\codes\py39\logs\mi-test\added_IVF677_Flat_nprobe_7.index" harvest "test.wav" "weights/mi-test.pth" 0.6 cuda:0 True<br>


f0up_key=sys.argv[1]<br>
input_path=sys.argv[2]<br>
index_path=sys.argv[3]<br>
f0method=sys.argv[4]#harvest or pm<br>
opt_path=sys.argv[5]<br>
model_path=sys.argv[6]<br>
index_rate=float(sys.argv[7])<br>
device=sys.argv[8]<br>
is_half=bool(sys.argv[9])<br>

## Q8:Cuda error/Cuda out of memory.
There is a small chance that there is a problem with the CUDA configuration or the device is not supported; more likely, there is not enough memory (out of memory).<br>

For training, reduce the batch size (if reducing to 1 is still not enough, you may need to change the graphics card); for inference, adjust the x_pad, x_query, x_center, and x_max settings in the config.py file as needed. 4G or lower memory cards (e.g. 1060(3G) and various 2G cards) can be abandoned, while 4G memory cards still have a chance.<br>

## Q9:How many total_epoch are optimal?
If the training dataset's audio quality is poor and the noise floor is high, 20-30 epochs are sufficient. Setting it too high won't improve the audio quality of your low-quality training set.<br>

If the training set audio quality is high, the noise floor is low, and there is sufficient duration, you can increase it. 200 is acceptable (since training is fast, and if you're able to prepare a high-quality training set, your GPU likely can handle a longer training duration without issue).<br>

## Q10:How much training set duration is needed?

A dataset of around 10min to 50min is recommended.<br>

With guaranteed high sound quality and low bottom noise, more can be added if the dataset's timbre is uniform.<br>

For a high-level training set (lean + distinctive tone), 5min to 10min is fine.<br>

There are some people who have trained successfully with 1min to 2min data, but the success is not reproducible by others and is not very informative. <br>This requires that the training set has a very distinctive timbre (e.g. a high-frequency airy anime girl sound) and the quality of the audio is high;
Data of less than 1min duration has not been successfully attempted so far. This is not recommended.<br>


## Q11:What is the index rate for and how to adjust it?
If the tone quality of the pre-trained model and inference source is higher than that of the training set, they can bring up the tone quality of the inference result, but at the cost of a possible tone bias towards the tone of the underlying model/inference source rather than the tone of the training set, which is generally referred to as "tone leakage".<br>

The index rate is used to reduce/resolve the timbre leakage problem. If the index rate is set to 1, theoretically there is no timbre leakage from the inference source and the timbre quality is more biased towards the training set. If the training set has a lower sound quality than the inference source, then a higher index rate may reduce the sound quality. Turning it down to 0 does not have the effect of using retrieval blending to protect the training set tones.<br>

If the training set has good audio quality and long duration, turn up the total_epoch, when the model itself is less likely to refer to the inferred source and the pretrained underlying model, and there is little "tone leakage", the index_rate is not important and you can even not create/share the index file.<br>

## Q12:How to choose the gpu when inferring?
In the config.py file, select the card number after "device cuda:".<br>

The mapping between card number and graphics card can be seen in the graphics card information section of the training tab.<br>

## Q13:How to use the model saved in the middle of training?
Save via model extraction at the bottom of the ckpt processing tab.

## Q14:File/memory error(when training)?
Too many processes and your memory is not enough. You may fix it by:

1、decrease the input in field "Threads of CPU".

2、pre-cut trainset to shorter audio files.

## Q15: How to continue training using more data

step1: put all wav data to path2.

step2: exp_name2+path2 -> process dataset and extract feature.

step3: copy the latest G and D file of exp_name1 (your previous experiment) into exp_name2 folder.

step4: click "train the model", and it will continue training from the beginning of your previous exp model epoch.

## Q16: error about llvmlite.dll

OSError: Could not load shared object file: llvmlite.dll

FileNotFoundError: Could not find module lib\site-packages\llvmlite\binding\llvmlite.dll (or one of its dependencies). Try using the full path with constructor syntax.

The issue will happen in windows, install https://aka.ms/vs/17/release/vc_redist.x64.exe and it will be fixed.

## Q17: RuntimeError: The expanded size of the tensor (17280) must match the existing size (0) at non-singleton dimension 1.  Target sizes: [1, 17280].  Tensor sizes: [0]

Delete the wav files whose size is significantly smaller than others, and that won't happen again. Than click "train the model"and "train the index".

## Q18: RuntimeError: The size of tensor a (24) must match the size of tensor b (16) at non-singleton dimension 2

Do not change the sampling rate and then continue training. If it is necessary to change, the exp name should be changed and the model will be trained from scratch. You can also copy the pitch and features (0/1/2/2b folders) extracted last time to accelerate the training process.

