Facebook AI Similarity Search (Faiss) 팁
==================
# Faiss에 대하여
Faiss 는 Facebook Research가 개발하는, 고밀도 벡터 이웃 검색 라이브러리입니다. 근사 근접 탐색법 (Approximate Neigbor Search)은 약간의 정확성을 희생하여 유사 벡터를 고속으로 찾습니다.

## RVC에 있어서 Faiss
RVC에서는 HuBERT로 변환한 feature의 embedding을 위해 훈련 데이터에서 생성된 embedding과 유사한 embadding을 검색하고 혼합하여 원래의 음성에 더욱 가까운 변환을 달성합니다. 그러나, 이 탐색법은 단순히 수행하면 시간이 다소 소모되므로, 근사 근접 탐색법을 통해 고속 변환을 가능케 하고 있습니다.

# 구현 개요
모델이 위치한 `/logs/your-experiment/3_feature256`에는 각 음성 데이터에서 HuBERT가 추출한 feature들이 있습니다. 여기에서 파일 이름별로 정렬된 npy 파일을 읽고, 벡터를 연결하여 big_npy ([N, 256] 모양의 벡터) 를 만듭니다. big_npy를 `/logs/your-experiment/total_fea.npy`로 저장한 후, Faiss로 학습시킵니다.

2023/04/18 기준으로, Faiss의 Index Factory 기능을 이용해, L2 거리에 근거하는 IVF를 이용하고 있습니다. IVF의 분할수(n_ivf)는 N//39로, n_probe는 int(np.power(n_ivf, 0.3))가 사용되고 있습니다. (infer-web.py의 train_index 주위를 찾으십시오.)

이 팁에서는 먼저 이러한 매개 변수의 의미를 설명하고, 개발자가 추후 더 나은 index를 작성할 수 있도록 하는 조언을 작성합니다.

# 방법의 설명
## Index factory
index factory는 여러 근사 근접 탐색법을 문자열로 연결하는 pipeline을 문자열로 표기하는 Faiss만의 독자적인 기법입니다. 이를 통해 index factory의 문자열을 변경하는 것만으로 다양한 근사 근접 탐색을 시도해 볼 수 있습니다. RVC에서는 다음과 같이 사용됩니다:

```python
index = Faiss.index_factory(256, "IVF%s,Flat" % n_ivf)
```
`index_factory`의 인수들 중 첫 번째는 벡터의 차원 수이고, 두번째는 index factory 문자열이며, 세번째에는 사용할 거리를 지정할 수 있습니다.

기법의 보다 자세한 설명은 https://github.com/facebookresearch/Faiss/wiki/The-index-factory 를 확인해 주십시오.

## 거리에 대한 index
embedding의 유사도로서 사용되는 대표적인 지표로서 이하의 2개가 있습니다.

- 유클리드 거리 (METRIC_L2)
- 내적(内積) (METRIC_INNER_PRODUCT)

유클리드 거리에서는 각 차원에서 제곱의 차를 구하고, 각 차원에서 구한 차를 모두 더한 후 제곱근을 취합니다. 이것은 일상적으로 사용되는 2차원, 3차원에서의 거리의 연산법과 같습니다. 내적은 그 값을 그대로 유사도 지표로 사용하지 않고, L2 정규화를 한 이후 내적을 취하는 코사인 유사도를 사용합니다.

어느 쪽이 더 좋은지는 경우에 따라 다르지만, word2vec에서 얻은 embedding 및 ArcFace를 활용한 이미지 검색 모델은 코사인 유사성이 이용되는 경우가 많습니다. numpy를 사용하여 벡터 X에 대해 L2 정규화를 하고자 하는 경우, 0 division을 피하기 위해 충분히 작은 값을 eps로 한 뒤 이하에 코드를 활용하면 됩니다.

```python
X_normed = X / np.maximum(eps, np.linalg.norm(X, ord=2, axis=-1, keepdims=True))
```

또한, `index factory`의 3번째 인수에 건네주는 값을 선택하는 것을 통해 계산에 사용하는 거리 index를 변경할 수 있습니다.

```python
index = Faiss.index_factory(dimention, text, Faiss.METRIC_INNER_PRODUCT)
```

## IVF
IVF (Inverted file indexes)는 역색인 탐색법과 유사한 알고리즘입니다. 학습시에는 검색 대상에 대해 k-평균 군집법을 실시하고 클러스터 중심을 이용해 보로노이 분할을 실시합니다. 각 데이터 포인트에는 클러스터가 할당되므로, 클러스터에서 데이터 포인트를 조회하는 dictionary를 만듭니다.

예를 들어, 클러스터가 다음과 같이 할당된 경우
|index|Cluster|
|-----|-------|
|1|A|
|2|B|
|3|A|
|4|C|
|5|B|

IVF 이후의 결과는 다음과 같습니다:

|cluster|index|
|-------|-----|
|A|1, 3|
|B|2, 5|
|C|4|

탐색 시, 우선 클러스터에서 `n_probe`개의 클러스터를 탐색한 다음, 각 클러스터에 속한 데이터 포인트의 거리를 계산합니다.

# 권장 매개변수
index의 선택 방법에 대해서는 공식적으로 가이드 라인이 있으므로, 거기에 준해 설명합니다.
https://github.com/facebookresearch/Faiss/wiki/Guidelines-to-choose-an-index

1M 이하의 데이터 세트에 있어서는 4bit-PQ가 2023년 4월 시점에서는 Faiss로 이용할 수 있는 가장 효율적인 수법입니다. 이것을 IVF와 조합해, 4bit-PQ로 후보를 추려내고, 마지막으로 이하의 index factory를 이용하여 정확한 지표로 거리를 재계산하면 됩니다.

```python
index = Faiss.index_factory(256, "IVF1024,PQ128x4fs,RFlat")
```

## IVF 권장 매개변수
IVF의 수가 너무 많으면, 가령 데이터 수의 수만큼 IVF로 양자화(Quantization)를 수행하면, 이것은 완전탐색과 같아져 효율이 나빠지게 됩니다. 1M 이하의 경우 IVF 값은 데이터 포인트 수 N에 대해 4sqrt(N) ~ 16sqrt(N)를 사용하는 것을 권장합니다.

n_probe는 n_probe의 수에 비례하여 계산 시간이 늘어나므로 정확도와 시간을 적절히 균형을 맞추어 주십시오. 개인적으로 RVC에 있어서 그렇게까지 정확도는 필요 없다고 생각하기 때문에 n_probe = 1이면 된다고 생각합니다.

## FastScan
FastScan은 직적 양자화를 레지스터에서 수행함으로써 거리의 고속 근사를 가능하게 하는 방법입니다.직적 양자화는 학습시에 d차원마다(보통 d=2)에 독립적으로 클러스터링을 실시해, 클러스터끼리의 거리를 사전 계산해 lookup table를 작성합니다. 예측시는 lookup table을 보면 각 차원의 거리를 O(1)로 계산할 수 있습니다. 따라서 PQ 다음에 지정하는 숫자는 일반적으로 벡터의 절반 차원을 지정합니다.

FastScan에 대한 자세한 설명은 공식 문서를 참조하십시오.
https://github.com/facebookresearch/Faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)

## RFlat
RFlat은 FastScan이 계산한 대략적인 거리를 index factory의 3번째 인수로 지정한 정확한 거리로 다시 계산하라는 인스트럭션입니다. k개의 근접 변수를 가져올 때 k*k_factor개의 점에 대해 재계산이 이루어집니다.

# Embedding 테크닉
## Alpha 쿼리 확장
퀴리 확장이란 탐색에서 사용되는 기술로, 예를 들어 전문 탐색 시, 입력된 검색문에 단어를 몇 개를 추가함으로써 검색 정확도를 올리는 방법입니다. 백터 탐색을 위해서도 몇가지 방법이 제안되었는데, 그 중 α-쿼리 확장은 추가 학습이 필요 없는 매우 효과적인 방법으로 알려져 있습니다. [Attention-Based Query Expansion Learning](https://arxiv.org/abs/2007.08019)와 [2nd place solution of kaggle shopee competition](https://www.kaggle.com/code/lyakaap/2nd-place-solution/notebook) 논문에서 소개된 바 있습니다..

α-쿼리 확장은 한 벡터에 인접한 벡터를 유사도의 α곱한 가중치로 더해주면 됩니다. 코드로 예시를 들어 보겠습니다. big_npy를 α query expansion로 대체합니다.

```python
alpha = 3.
index = Faiss.index_factory(256, "IVF512,PQ128x4fs,RFlat")
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

# index version 정규화
big_npy = big_npy / np.maximum(np.linalg.norm(big_npy, ord=2, axis=1, keepdims=True), 1e-9)
```

위 테크닉은 탐색을 수행하는 쿼리에도, 탐색 대상 DB에도 적응 가능한 테크닉입니다.

## MiniBatch KMeans에 의한 embedding 압축

total_fea.npy가 너무 클 경우 K-means를 이용하여 벡터를 작게 만드는 것이 가능합니다. 이하 코드로 embedding의 압축이 가능합니다. n_clusters에 압축하고자 하는 크기를 지정하고 batch_size에 256 * CPU의 코어 수를 지정함으로써 CPU 병렬화의 혜택을 충분히 얻을 수 있습니다.

```python
import multiprocessing
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=10000, batch_size=256 * multiprocessing.cpu_count(), init="random")
kmeans.fit(big_npy)
sample_npy = kmeans.cluster_centers_
```