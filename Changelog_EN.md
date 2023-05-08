### 2023-04-09
- Fixed training parameters to improve GPU utilization rate: A100 increased from 25% to around 90%, V100: 50% to around 90%, 2060S: 60% to around 85%, P40: 25% to around 95%; significantly improved training speed
- Changed parameter: total batch_size is now per GPU batch_size
- Changed total_epoch: maximum limit increased from 100 to 1000; default increased from 10 to 20
- Fixed issue of ckpt extraction recognizing pitch incorrectly, causing abnormal inference
- Fixed issue of distributed training saving ckpt for each rank
- Applied nan feature filtering for feature extraction
- Fixed issue with silent input/output producing random consonants or noise (old models need to retrain with a new dataset)

### 2023-04-16 Update
- Added local real-time voice changing mini-GUI, start by double-clicking go-realtime-gui.bat
- Applied filtering for frequency bands below 50Hz during training and inference
- Lowered the minimum pitch extraction of pyworld from the default 80 to 50 for training and inference, allowing male low-pitched voices between 50-80Hz not to be muted
- WebUI supports changing languages according to system locale (currently supporting en_US, ja_JP, zh_CN, zh_HK, zh_SG, zh_TW; defaults to en_US if not supported)
- Fixed recognition of some GPUs (e.g., V100-16G recognition failure, P4 recognition failure)

### 2023-04-28 Update
- Upgraded faiss index settings for faster speed and higher quality
- Removed dependency on total_npy; future model sharing will not require total_npy input
- Unlocked restrictions for the 16-series GPUs, providing 4GB inference settings for 4GB VRAM GPUs
- Fixed bug in UVR5 vocal accompaniment separation for certain audio formats
- Real-time voice changing mini-GUI now supports non-40k and non-lazy pitch models

### Future Plans:
Features:
- Add option: extract small models for each epoch save
- Add option: export additional mp3 to the specified path during inference
- Support multi-person training tab (up to 4 people)

Base model:
- Collect breathing wav files to add to the training dataset to fix the issue of distorted breath sounds
- We are currently training a base model with an extended singing dataset, which will be released in the future
- Upgrade discriminator
- Upgrade self-supervised feature structure
