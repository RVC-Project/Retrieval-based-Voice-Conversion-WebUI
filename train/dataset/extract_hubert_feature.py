import os
import sys
import traceback

device = sys.argv[1]
n_part = int(sys.argv[2])
i_part = int(sys.argv[3])
if len(sys.argv) == 7:
    exp_dir = sys.argv[4]
    version = sys.argv[5]
    is_half = sys.argv[6].lower() == "true"
else:
    i_gpu = sys.argv[4]
    exp_dir = sys.argv[5]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    version = sys.argv[6]
    is_half = sys.argv[7].lower() == "true"
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from configs.config import get_device_dtype_sm
from infer.hubert import (
    HUBERT_MODEL_PATH,
    extract_hubert_features,
    hubert_audio_requires_normalization,
    load_hubert_model,
)
from i18n.i18n import I18nAuto
from tools.progress import should_report

i18n = I18nAuto()

if "privateuseone" not in device:
    device = "cpu"
    if torch.cuda.is_available():
        selected_device, selected_dtype, _, _ = get_device_dtype_sm(0)
        device = str(selected_device)
        is_half = is_half and selected_dtype == torch.float16
else:
    import torch_directml

    device = torch_directml.device(torch_directml.default_device())


f = open("%s/extract_f0_feature.log" % exp_dir, "a", encoding="utf8")


def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


model_path = str(HUBERT_MODEL_PATH)
wavPath = "%s/1_16k_wavs" % exp_dir
outPath = (
    "%s/3_feature256" % exp_dir if version == "v1" else "%s/3_feature768" % exp_dir
)
os.makedirs(outPath, exist_ok=True)


# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


assigned_files = [
    file
    for file in sorted(os.listdir(wavPath))[i_part::n_part]
    if file.endswith(".wav")
]
todo = [
    file
    for file in assigned_files
    if not os.path.exists(
        "%s/%s.npy" % (outPath, os.path.splitext(file)[0])
    )
]
skipped = len(assigned_files) - len(todo)
if len(todo) == 0:
    printt(i18n("[HuBERT特征] 无待处理音频，已全部跳过：%s") % skipped)
    raise SystemExit(0)


printt(i18n("[HuBERT特征] 正在加载模型：%s") % model_path)
if os.access(model_path, os.F_OK) == False:
    printt(
        i18n("[HuBERT特征][失败] 模型不存在：%s")
        % model_path
    )
    raise SystemExit(1)
model = load_hubert_model(device, is_half and device != "cpu")
normalize_audio = hubert_audio_requires_normalization()
printt(
    i18n("[HuBERT特征] 设备：%s | 待处理：%s | 已跳过：%s")
    % (device, len(todo), skipped)
)

success = 0
failed = 0
for idx, file in enumerate(todo):
    try:
        wav_path = "%s/%s" % (wavPath, file)
        out_path = "%s/%s.npy" % (outPath, os.path.splitext(file)[0])
        if os.path.exists(out_path):
            skipped += 1
            continue

        feats = readwave(wav_path, normalize=normalize_audio)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        source = (
            feats.half().to(device)
            if is_half and device != "cpu"
            else feats.to(device)
        )
        with torch.no_grad():
            feats = extract_hubert_features(
                model,
                source,
                version,
                padding_mask=padding_mask.to(device),
            )

        feats = feats.squeeze(0).float().cpu().numpy()
        if np.isnan(feats).sum() == 0:
            np.save(out_path, feats, allow_pickle=False)
            success += 1
            if should_report(idx, len(todo), max(1, (12 + n_part - 1) // n_part)):
                printt(
                    i18n("[HuBERT特征] 进度：%s/%s | 成功：%s | 失败：%s | %s | %s")
                    % (idx + 1, len(todo), success, failed, file, feats.shape)
                )
        else:
            failed += 1
            printt(i18n("[HuBERT特征][失败] %s 包含NaN") % file)
    except Exception:
        failed += 1
        printt(i18n("[HuBERT特征][失败] %s\n%s") % (file, traceback.format_exc()))
printt(
    i18n("[HuBERT特征] 完成 | 成功：%s | 跳过：%s | 失败：%s")
    % (success, skipped, failed)
)
