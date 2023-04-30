faiss tuning TIPS
==================
# about faiss
faiss is a library of neighborhood searches for dense vectors, developed by facebook research, which efficiently implements many approximate neighborhood search methods.
Approximate Neighbor Search finds similar vectors quickly while sacrificing some accuracy.

## faiss in RVC
In RVC, for the embedding of features converted by HuBERT, we search for embeddings similar to the embedding generated from the training data and mix them to achieve a conversion that is closer to the original speech. However, since this search takes time if performed naively, high-speed conversion is realized by using approximate neighborhood search.

# implementation overview
In '/logs/your-experiment/3_feature256' where the model is located, features extracted by HuBERT from each voice data are located.
From here we read the npy files in order sorted by filename and concatenate the vectors to create big_npy. (This vector has shape [N, 256].)
After saving big_npy as /logs/your-experiment/total_fea.npy, train it with faiss.

In this article, I will explain the meaning of these parameters.

# Explanation of the method
## index factory
An index factory is a unique faiss notation that expresses a pipeline that connects multiple approximate neighborhood search methods as a string.
This allows you to try various approximate neighborhood search methods simply by changing the index factory string.
In RVC it is used like this:

```python
index = faiss.index_factory(256, "IVF%s,Flat" % n_ivf)
```
Among the arguments of index_factory, the first is the number of dimensions of the vector, the second is the index factory string, and the third is the distance to use.

For more detailed notation
https://github.com/facebookresearch/faiss/wiki/The-index-factory

## index for distance
There are two typical indexes used as similarity of embedding as follows.

- Euclidean distance (METRIC_L2)
- inner product (METRIC_INNER_PRODUCT)

Euclidean distance takes the squared difference in each dimension, sums the differences in all dimensions, and then takes the square root. This is the same as the distance in 2D and 3D that we use on a daily basis.
The inner product is not used as an index of similarity as it is, and the cosine similarity that takes the inner product after being normalized by the L2 norm is generally used.

Which is better depends on the case, but cosine similarity is often used in embedding obtained by word2vec and similar image retrieval models learned by ArcFace. If you want to do l2 normalization on vector X with numpy, you can do it with the following code with eps small enough to avoid 0 division.

```python
X_normed = X / np.maximum(eps, np.linalg.norm(X, ord=2, axis=-1, keepdims=True))
```

Also, for the index factory, you can change the distance index used for calculation by choosing the value to pass as the third argument.

```python
index = faiss.index_factory(dimention, text, faiss.METRIC_INNER_PRODUCT)
```

## IVF
IVF (Inverted file indexes) is an algorithm similar to the inverted index in full-text search.
During learning, the search target is clustered with kmeans, and Voronoi partitioning is performed using the cluster center. Each data point is assigned a cluster, so we create a dictionary that looks up the data points from the clusters.

For example, if clusters are assigned as follows
|index|Cluster|
|-----|-------|
|1|A|
|2|B|
|3|A|
|4|C|
|5|B|

The resulting inverted index looks like this:

|cluster|index|
|-------|-----|
|A|1, 3|
|B|2, 5|
|C|4|

When searching, we first search n_probe clusters from the clusters, and then calculate the distances for the data points belonging to each cluster.

# recommend parameter
There are official guidelines on how to choose an index, so I will explain accordingly.
https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

For datasets below 1M, 4bit-PQ is the most efficient method available in faiss as of April 2023.
Combining this with IVF, narrowing down the candidates with 4bit-PQ, and finally recalculating the distance with an accurate index can be described by using the following index factory.

```python
index = faiss.index_factory(256, "IVF1024,PQ128x4fs,RFlat")
```

## Recommended parameters for IVF
Consider the case of too many IVFs. For example, if coarse quantization by IVF is performed for the number of data, this is the same as a naive exhaustive search and is inefficient.
For 1M or less, IVF values are recommended between 4*sqrt(N) ~ 16*sqrt(N) for N number of data points.

Since the calculation time increases in proportion to the number of n_probes, please consult with the accuracy and choose appropriately. Personally, I don't think RVC needs that much accuracy, so n_probe = 1 is fine.

## FastScan
FastScan is a method that enables high-speed approximation of distances by Cartesian product quantization by performing them in registers.
Cartesian product quantization performs clustering independently for each d dimension (usually d = 2) during learning, calculates the distance between clusters in advance, and creates a lookup table. At the time of prediction, the distance of each dimension can be calculated in O(1) by looking at the lookup table.
So the number you specify after PQ usually specifies half the dimension of the vector.

For a more detailed description of FastScan, please refer to the official documentation.
https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)

## RFlat
RFlat is an instruction to recalculate the rough distance calculated by FastScan with the exact distance specified by the third argument of index factory.
When getting k neighbors, k*k_factor points are recalculated.
