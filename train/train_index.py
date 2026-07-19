import os
import platform
import sys
import traceback
import glob

import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from i18n.i18n import I18nAuto
from tools.progress import should_report


i18n = I18nAuto()


exp_name = sys.argv[1]
version = sys.argv[2]
outside_index_root = sys.argv[3]
n_cpu = int(sys.argv[4])
exp_dir = os.path.join("logs", exp_name)
feature_dir = os.path.join(
    exp_dir, "3_feature256" if version == "v1" else "3_feature768"
)
log_path = os.path.join(exp_dir, "train_index.log")
os.makedirs(exp_dir, exist_ok=True)


def log(message):
    print(message, flush=True)
    with open(log_path, "a", encoding="utf8") as f:
        f.write(str(message) + "\n")


with open(log_path, "w", encoding="utf8"):
    pass


def newest_index(pattern):
    paths = [path for path in glob.glob(pattern) if os.path.isfile(path)]
    return max(paths, key=os.path.getmtime) if paths else ""


def link_added_index(added_path):
    added_name = os.path.basename(added_path)
    try:
        os.makedirs(outside_index_root, exist_ok=True)
        source = os.path.abspath(added_path)
        outside_root = os.path.abspath(outside_index_root)
        if os.path.commonpath([source, outside_root]) == outside_root:
            log(i18n("[索引训练] 外部索引链接已存在：%s") % source)
            return
        target = os.path.abspath(
            os.path.join(outside_index_root, "%s_%s" % (exp_name, added_name))
        )
        if os.path.lexists(target):
            try:
                if os.path.samefile(source, target):
                    log(i18n("[索引训练] 外部索引链接已存在：%s") % target)
                    return
            except (FileNotFoundError, OSError):
                pass
            os.unlink(target)
        if platform.system() == "Windows":
            os.link(source, target)
        else:
            os.symlink(source, target)
        log(i18n("[索引训练] 已链接索引到外部目录：%s") % outside_index_root)
    except Exception:
        log(
            i18n("[索引训练][失败] 无法链接索引到外部目录：%s\n%s")
            % (outside_index_root, traceback.format_exc())
        )


existing_trained_path = newest_index(
    os.path.join(
        exp_dir,
        "trained_IVF*_Flat_nprobe_*_%s_%s.index" % (exp_name, version),
    )
)
existing_added_path = newest_index(
    os.path.join(exp_dir, "added_IVF*_Flat_nprobe_*_%s_%s.index" % (exp_name, version))
)
if not existing_added_path:
    existing_added_path = newest_index(
        os.path.join(
            outside_index_root,
            "%s_*added_IVF*_Flat_nprobe_*_%s_%s.index"
            % (exp_name, exp_name, version),
        )
    )
if existing_added_path:
    if existing_trained_path:
        log(
            i18n("[索引训练][跳过] trained索引已存在：%s")
            % os.path.basename(existing_trained_path)
        )
    log(
        i18n("[索引训练][跳过] added索引已存在：%s")
        % os.path.basename(existing_added_path)
    )
    link_added_index(existing_added_path)
    raise SystemExit(0)

if not os.path.isdir(feature_dir) or not os.listdir(feature_dir):
    log(i18n("[索引训练][失败] 请先进行特征提取"))
    raise SystemExit(1)

features = []
for name in sorted(os.listdir(feature_dir)):
    features.append(np.load(os.path.join(feature_dir, name)))

big_npy = np.concatenate(features, 0)
big_npy = big_npy[np.random.permutation(big_npy.shape[0])]
if big_npy.shape[0] > 200000:
    log(i18n("[索引训练] 正在将%s条特征聚类为10000个中心") % big_npy.shape[0])
    try:
        big_npy = MiniBatchKMeans(
            n_clusters=10000,
            verbose=False,
            batch_size=256 * n_cpu,
            compute_labels=False,
            init="random",
        ).fit(big_npy).cluster_centers_
    except Exception:
        log(i18n("[索引训练][失败] 聚类失败，将使用原始特征继续\n%s") % traceback.format_exc())

n_ivf = max(1, min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39))
log(i18n("[索引训练] 特征形状：%s | IVF数量：%s") % (big_npy.shape, n_ivf))
if existing_trained_path:
    trained_path = existing_trained_path
    index = faiss.read_index(trained_path)
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    log(
        i18n("[索引训练][跳过] trained索引已存在：%s")
        % os.path.basename(trained_path)
    )
else:
    index = faiss.index_factory(
        256 if version == "v1" else 768, "IVF%s,Flat" % n_ivf
    )
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    trained_path = os.path.join(
        exp_dir,
        "trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_name, version),
    )
    log(i18n("[索引训练] 正在训练索引"))
    index.train(big_npy)
    faiss.write_index(index, trained_path)

log(i18n("[索引训练] 正在写入特征向量"))
starts = list(range(0, big_npy.shape[0], 8192))
for batch_index, start in enumerate(starts):
    index.add(big_npy[start : start + 8192])
    if should_report(batch_index, len(starts), 10):
        log(
            i18n("[索引训练] 写入进度：%s/%s")
            % (batch_index + 1, len(starts))
        )

added_name = "added_IVF%s_Flat_nprobe_%s_%s_%s.index" % (
    n_ivf,
    index_ivf.nprobe,
    exp_name,
    version,
)
added_path = os.path.join(exp_dir, added_name)
faiss.write_index(index, added_path)
log(i18n("[索引训练] 成功构建索引：%s") % added_name)
link_added_index(added_path)
