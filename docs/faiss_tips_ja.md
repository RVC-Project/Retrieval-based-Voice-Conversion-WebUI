faiss tuning TIPS
==================
# about faiss
faissはfacebook researchの開発する、密なベクトルに対する近傍探索をまとめたライブラリで、多くの近似近傍探索の手法を効率的に実装しています。
近似近傍探索はある程度精度を犠牲にしながら高速に類似するベクトルを探します。

## faiss in RVC
RVCではHuBERTで変換した特徴量のEmbeddingに対し、学習データから生成されたEmbeddingと類似するものを検索し、混ぜることでより元の音声に近い変換を実現しています。ただ、この検索は愚直に行うと時間がかかるため、近似近傍探索を用いることで高速な変換を実現しています。

# 実装のoverview
モデルが配置されている '/logs/your-experiment/3_feature256'には各音声データからHuBERTで抽出された特徴量が配置されています。
ここからnpyファイルをファイル名でソートした順番で読み込み、ベクトルを連結してbig_npyを作成します。(このベクトルのshapeは[N, 256]です。)
big_npyを/logs/your-experiment/total_fea.npyとして保存した後、faissを学習させます。

2023/04/18時点ではfaissのindex factoryの機能を用いて、L2距離に基づくIVFを用いています。
IVFの分割数(n_ivf)はN//39で、n_probeはint(np.power(n_ivf, 0.3))が採用されています。(infer-web.pyのtrain_index周りを探してください。)

本Tipsではまずこれらのパラメータの意味を解説し、その後よりよいindexを作成するための開発者向けアドバイスを書きます。

# 手法の解説
## index factory
index factoryは複数の近似近傍探索の手法を繋げるパイプラインをstringで表記するfaiss独自の記法です。
これにより、index factoryの文字列を変更するだけで様々な近似近傍探索の手法を試せます。
RVCでは以下のように使われています。

```python
index = faiss.index_factory(256, "IVF%s,Flat" % n_ivf)
```
index_factoryの引数のうち、1つ目はベクトルの次元数、2つ目はindex factoryの文字列で、3つ目には用いる距離を指定することができます。

より詳細な記法については
https://github.com/facebookresearch/faiss/wiki/The-index-factory

## 距離指標
embeddingの類似度として用いられる代表的な指標として以下の二つがあります。

- ユークリッド距離(METRIC_L2)
- 内積(METRIC_INNER_PRODUCT)

ユークリッド距離では各次元において二乗の差をとり、全次元の差を足してから平方根をとります。これは日常的に用いる2次元、3次元での距離と同じです。
内積はこのままでは類似度の指標として用いず、一般的にはL2ノルムで正規化してから内積をとるコサイン類似度を用います。

どちらがよいかは場合によりますが、word2vec等で得られるembeddingやArcFace等で学習した類似画像検索のモデルではコサイン類似度が用いられることが多いです。ベクトルXに対してl2正規化をnumpyで行う場合は、0 divisionを避けるために十分に小さな値をepsとして以下のコードで可能です。

```python
X_normed = X / np.maximum(eps, np.linalg.norm(X, ord=2, axis=-1, keepdims=True))
```

また、index factoryには第3引数に渡す値を選ぶことで計算に用いる距離指標を変更できます。

```python
index = faiss.index_factory(dimention, text, faiss.METRIC_INNER_PRODUCT)
```

## IVF
IVF(Inverted file indexes)は全文検索における転置インデックスと似たようなアルゴリズムです。
学習時には検索対象に対してkmeansでクラスタリングを行い、クラスタ中心を用いてボロノイ分割を行います。各データ点には一つずつクラスタが割り当てられるので、クラスタからデータ点を逆引きする辞書を作成します。

例えば以下のようにクラスタが割り当てられた場合
|index|クラスタ|
|-----|-------|
|1|A|
|2|B|
|3|A|
|4|C|
|5|B|

作成される転置インデックスは以下のようになります。

|クラスタ|index|
|-------|-----|
|A|1, 3|
|B|2, 5|
|C|4|

検索時にはまずクラスタからn_probe個のクラスタを検索し、次にそれぞれのクラスタに属するデータ点について距離を計算します。

# 推奨されるパラメータ
indexの選び方については公式にガイドラインがあるので、それに準じて説明します。
https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

1M以下のデータセットにおいては4bit-PQが2023年4月時点ではfaissで利用できる最も効率的な手法です。
これをIVFと組み合わせ、4bit-PQで候補を絞り、最後に正確な指標で距離を再計算するには以下のindex factoryを用いることで記載できます。

```python
index = faiss.index_factory(256, "IVF1024,PQ128x4fs,RFlat")
```

## IVFの推奨パラメータ
IVFの数が多すぎる場合、たとえばデータ数の数だけIVFによる粗量子化を行うと、これは愚直な全探索と同じになり効率が悪いです。
1M以下の場合ではIVFの値はデータ点の数Nに対して4*sqrt(N) ~ 16*sqrt(N)に推奨しています。

n_probeはn_probeの数に比例して計算時間が増えるので、精度と相談して適切に選んでください。個人的にはRVCにおいてそこまで精度は必要ないと思うのでn_probe = 1で良いと思います。

## FastScan
FastScanは直積量子化で大まかに距離を近似するのを、レジスタ内で行うことにより高速に行うようにした手法です。
直積量子化は学習時にd次元ごと(通常はd=2)に独立してクラスタリングを行い、クラスタ同士の距離を事前計算してlookup tableを作成します。予測時はlookup tableを見ることで各次元の距離をO(1)で計算できます。
そのため、PQの次に指定する数字は通常ベクトルの半分の次元を指定します。

FastScanに関するより詳細な説明は公式のドキュメントを参照してください。
https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)

## RFlat
RFlatはFastScanで計算した大まかな距離を、index factoryの第三引数で指定した正確な距離で再計算する指示です。
k個の近傍を取得する際は、k*k_factor個の点について再計算が行われます。

# Embeddingに関するテクニック
## alpha query expansion
クエリ拡張は検索で使われるテクニックで、例えば全文検索では入力された検索文に単語を幾つか追加することで検索精度を上げることがあります。ベクトル検索にもいくつか提唱されていて、その内追加の学習がいらず効果が高い手法としてα-query expansionが知られています。論文では[Attention-Based Query Expansion Learning](https://arxiv.org/abs/2007.08019)などで紹介されていて、[kaggleのshopeeコンペの2位の解法](https://www.kaggle.com/code/lyakaap/2nd-place-solution/notebook)にも用いられていました。

α-query expansionはあるベクトルに対し、近傍のベクトルを類似度のα乗した重みで足し合わせることでできます。いかにコードの例を張ります。big_npyをα query expansionしたものに置き換えます。

```python
alpha = 3.
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
    weight = np.power(np.einsum("nd,nmd->nm", big_npy[ix], big_npy[neighbor[ix]]), alpha)
    expand_arrays.append(np.sum(big_npy[neighbor[ix]] * np.expand_dims(weight, axis=2),axis=1))
big_npy = np.concatenate(expand_arrays, axis=0)

# normalize index version
big_npy = big_npy / np.maximum(np.linalg.norm(big_npy, ord=2, axis=1, keepdims=True), 1e-9)
```

これは、検索を行うクエリにも、検索対象のDBにも適応可能なテクニックです。

## MiniBatch KMeansによるembeddingの圧縮
total_fea.npyが大きすぎる場合、KMeansを用いてベクトルを小さくすることが可能です。
以下のコードで、embeddingの圧縮が可能です。n_clustersは圧縮したい大きさを指定し、batch_sizeは256 * CPUのコア数を指定することでCPUの並列化の恩恵を十分に得ることができます。

```python
import multiprocessing
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=10000, batch_size=256 * multiprocessing.cpu_count(), init="random")
kmeans.fit(big_npy)
sample_npy = kmeans.cluster_centers_
```