"""
格式：直接cid为自带的index位；aid放不下了，通过字典来查，反正就5w个
"""
import os
import traceback
import logging

logger = logging.getLogger(__name__)

from multiprocessing import cpu_count

import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# ###########如果是原始特征要先写save
n_cpu = 0
if n_cpu == 0:
    n_cpu = cpu_count()
inp_root = r"./logs/anz/3_feature768"
npys = []
listdir_res = list(os.listdir(inp_root))
for name in sorted(listdir_res):
    phone = np.load("%s/%s" % (inp_root, name))
    npys.append(phone)
big_npy = np.concatenate(npys, 0)
big_npy_idx = np.arange(big_npy.shape[0])
np.random.shuffle(big_npy_idx)
big_npy = big_npy[big_npy_idx]
logger.debug(big_npy.shape)  # (6196072, 192)#fp32#4.43G
if big_npy.shape[0] > 2e5:
    # if(1):
    info = "Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0]
    logger.info(info)
    try:
        big_npy = (
            MiniBatchKMeans(
                n_clusters=10000,
                verbose=True,
                batch_size=256 * n_cpu,
                compute_labels=False,
                init="random",
            )
            .fit(big_npy)
            .cluster_centers_
        )
    except:
        info = traceback.format_exc()
        logger.warning(info)

np.save("tools/infer/big_src_feature_mi.npy", big_npy)

##################train+add
# big_npy=np.load("/bili-coeus/jupyter/jupyterhub-liujing04/vits_ch/inference_f0/big_src_feature_mi.npy")
n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
index = faiss.index_factory(768, "IVF%s,Flat" % n_ivf)  # mi
logger.info("Training...")
index_ivf = faiss.extract_index_ivf(index)  #
index_ivf.nprobe = 1
index.train(big_npy)
faiss.write_index(
    index, "tools/infer/trained_IVF%s_Flat_baseline_src_feat_v2.index" % (n_ivf)
)
logger.info("Adding...")
batch_size_add = 8192
for i in range(0, big_npy.shape[0], batch_size_add):
    index.add(big_npy[i : i + batch_size_add])
faiss.write_index(
    index, "tools/infer/added_IVF%s_Flat_mi_baseline_src_feat.index" % (n_ivf)
)
"""
大小（都是FP32）
big_src_feature 2.95G
    (3098036, 256)
big_emb         4.43G
    (6196072, 192)
big_emb双倍是因为求特征要repeat后再加pitch

"""
