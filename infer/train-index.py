'''
格式：直接cid为自带的index位；aid放不下了，通过字典来查，反正就5w个
'''
import faiss,numpy as np,os

# ###########如果是原始特征要先写save
inp_root=r"E:\codes\py39\dataset\mi\2-co256"
npys=[]
for name in sorted(list(os.listdir(inp_root))):
    phone=np.load("%s/%s"%(inp_root,name))
    npys.append(phone)
big_npy=np.concatenate(npys,0)
print(big_npy.shape)#(6196072, 192)#fp32#4.43G

num_expand = 16  # this is tunable params
batch_size = 1024  # this is tunable params

index = faiss.index_factory(256, "IVF512,PQ128x4fs,RFlat")
original_norm = np.maximum(np.linalg.norm(big_npy, ord=2, axis=1, keepdims=True), 1e-9)
big_npy /= original_norm
index.train(big_npy)
index.add(big_npy)
dist, neighbor = index.search(big_npy, num_expand)

expand_arrays = []
ixs = np.arange(big_npy.shape[0])
for i in range(-(-big_npy.shape[0]//batch_size)):
    ix = ixs[i*batch_size:(i+1)*batch_size]
    weight = np.power(np.einsum("nd,nmd->nm", big_npy[ix], big_npy[neighbor[ix]]), 3.)
    expand_arrays.append(np.sum(big_npy[neighbor[ix]] * np.expand_dims(weight, axis=2),axis=1))
big_npy = np.concatenate(expand_arrays, axis=0)

# normalize index version
big_npy = big_npy / np.maximum(np.linalg.norm(big_npy, ord=2, axis=1, keepdims=True), 1e-9)
np.save("infer/big_src_feature_mi.npy", big_npy * original_norm)

# not normalize index version
# big_npy = big_npy / np.maximum(np.linalg.norm(big_npy, ord=2, axis=1, keepdims=True), 1e-9) * original_norm
# np.save("infer/big_src_feature_mi.npy", big_npy)

##################train+add
# big_npy=np.load("/bili-coeus/jupyter/jupyterhub-liujing04/vits_ch/inference_f0/big_src_feature_mi.npy")
print(big_npy.shape)
index = faiss.index_factory(256, "IVF512,Flat")#mi
print("training")
index_ivf = faiss.extract_index_ivf(index)#
index_ivf.nprobe = 9
index.train(big_npy)
faiss.write_index(index, 'infer/trained_IVF512_Flat_mi_baseline_src_feat.index')
print("adding")
index.add(big_npy)
faiss.write_index(index,"infer/added_IVF512_Flat_mi_baseline_src_feat.index")
'''
大小（都是FP32）
big_src_feature 2.95G
    (3098036, 256)
big_emb         4.43G
    (6196072, 192)
big_emb双倍是因为求特征要repeat后再加pitch

'''