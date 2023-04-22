import os

import faiss
import ffmpeg
import numpy as np


def train_index(exp_dir1):
    '''train and save faiss index
    
    Args:
        exp_dir1(string): Relative path where index is stored
    '''
    exp_dir = "%s/logs/%s" % (os.getcwd(), exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = "%s/3_feature256" % (exp_dir)

    if os.path.exists(feature_dir) == False:
        return "请先进行特征提取!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "请先进行特征提取！"

    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    np.save("%s/total_fea.npy" % exp_dir, big_npy)

    # use recommended parameter in https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    N = big_npy.shape[0]
    dim = big_npy.shape[1]
    if 4 * np.rint(np.sqrt(N)) * 30 > N:
        n_ivf = N // 30
    else:
        n_ivf = -(-N // 256)
        for x in range(4, 18, 2):
            K = x * np.rint(np.sqrt(N)).astype(int)
            if 30 * K <= N <= 256 * K:
                n_ivf = K
                break
    index_string = "IVF%s,PQ%sx4fs,RFlat" % (n_ivf, -(-dim//2))
    index_name = index_string.replace(",", "_")

    
    infos = []
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(dim, index_string)
    infos.append("training")
    yield "\n".join(infos)
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_%s.index" % (exp_dir, index_name),
    )
    infos.append("adding")
    yield "\n".join(infos)
    index.add(big_npy)
    faiss.write_index(
        index,
        "%s/added_%s.index" % (exp_dir, index_name),
    )
    infos.append("成功构建索引, added_%s.index" % (index_name))
    yield "\n".join(infos)


def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
