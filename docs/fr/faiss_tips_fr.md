Conseils de réglage pour faiss
==================
# À propos de faiss
faiss est une bibliothèque de recherches de voisins pour les vecteurs denses, développée par Facebook Research, qui implémente efficacement de nombreuses méthodes de recherche de voisins approximatifs.
La recherche de voisins approximatifs trouve rapidement des vecteurs similaires tout en sacrifiant une certaine précision.

## faiss dans RVC
Dans RVC, pour l'incorporation des caractéristiques converties par HuBERT, nous recherchons des incorporations similaires à l'incorporation générée à partir des données d'entraînement et les mixons pour obtenir une conversion plus proche de la parole originale. Cependant, cette recherche serait longue si elle était effectuée de manière naïve, donc une conversion à haute vitesse est réalisée en utilisant une recherche de voisinage approximatif.

# Vue d'ensemble de la mise en œuvre
Dans '/logs/votre-expérience/3_feature256' où le modèle est situé, les caractéristiques extraites par HuBERT de chaque donnée vocale sont situées.
À partir de là, nous lisons les fichiers npy dans un ordre trié par nom de fichier et concaténons les vecteurs pour créer big_npy. (Ce vecteur a la forme [N, 256].)
Après avoir sauvegardé big_npy comme /logs/votre-expérience/total_fea.npy, nous l'entraînons avec faiss.

Dans cet article, j'expliquerai la signification de ces paramètres.

# Explication de la méthode
## Usine d'index
Une usine d'index est une notation unique de faiss qui exprime un pipeline qui relie plusieurs méthodes de recherche de voisinage approximatif sous forme de chaîne.
Cela vous permet d'essayer diverses méthodes de recherche de voisinage approximatif simplement en changeant la chaîne de l'usine d'index.
Dans RVC, elle est utilisée comme ceci :

```python
index = faiss.index_factory(256, "IVF%s,Flat" % n_ivf)
```

Parmi les arguments de index_factory, le premier est le nombre de dimensions du vecteur, le second est la chaîne de l'usine d'index, et le troisième est la distance à utiliser.

Pour une notation plus détaillée :
https://github.com/facebookresearch/faiss/wiki/The-index-factory

## Index pour la distance
Il existe deux index typiques utilisés comme similarité de l'incorporation comme suit :

- Distance euclidienne (METRIC_L2)
- Produit intérieur (METRIC_INNER_PRODUCT)

La distance euclidienne prend la différence au carré dans chaque dimension, somme les différences dans toutes les dimensions, puis prend la racine carrée. C'est la même chose que la distance en 2D et 3D que nous utilisons au quotidien.
Le produit intérieur n'est pas utilisé comme index de similarité tel quel, et la similarité cosinus qui prend le produit intérieur après avoir été normalisé par la norme L2 est généralement utilisée.

Lequel est le mieux dépend du cas, mais la similarité cosinus est souvent utilisée dans l'incorporation obtenue par word2vec et des modèles de récupération d'images similaires appris par ArcFace. Si vous voulez faire une normalisation l2 sur le vecteur X avec numpy, vous pouvez le faire avec le code suivant avec eps suffisamment petit pour éviter une division par 0.

```python
X_normed = X / np.maximum(eps, np.linalg.norm(X, ord=2, axis=-1, keepdims=True))
```

De plus, pour l'usine d'index, vous pouvez changer l'index de distance utilisé pour le calcul en choisissant la valeur à passer comme troisième argument.

```python
index = faiss.index_factory(dimention, texte, faiss.METRIC_INNER_PRODUCT)
```

## IVF
IVF (Inverted file indexes) est un algorithme similaire à l'index inversé dans la recherche en texte intégral.
Lors de l'apprentissage, la cible de recherche est regroupée avec kmeans, et une partition de Voronoi est effectuée en utilisant le centre du cluster. Chaque point de données est attribué à un cluster, nous créons donc un dictionnaire qui permet de rechercher les points de données à partir des clusters.

Par exemple, si des clusters sont attribués comme suit :
|index|Cluster|
|-----|-------|
|1|A|
|2|B|
|3|A|
|4|C|
|5|B|

L'index inversé résultant ressemble à ceci :

|cluster|index|
|-------|-----|
|A|1, 3|
|B|2, 5|
|C|4|

Lors de la recherche, nous recherchons d'abord n_probe clusters parmi les clusters, puis nous calculons les distances pour les points de données appartenant à chaque cluster.

# Recommandation de paramètre
Il existe des directives officielles sur la façon de choisir un index, je vais donc expliquer en conséquence.
https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

Pour les ensembles de données inférieurs à 1M, 4bit-PQ est la méthode la plus efficace disponible dans faiss en avril 2023.
En combinant cela avec IVF, en réduisant les candidats avec 4bit-PQ, et enfin en recalculant la distance avec un index précis, on peut le décrire en utilisant l'usine d'index suivante.

```python
index = faiss.index_factory(256, "IVF1024,PQ128x4fs,RFlat")
```

## Paramètres recommandés pour IVF
Considérez le cas de trop d'IVF. Par exemple, si une quantification grossière par IVF est effectuée pour le nombre de données, cela revient à une recherche exhaustive naïve et est inefficace.
Pour 1M ou moins, les valeurs IVF sont recommandées entre 4*sqrt(N) ~ 16*sqrt(N) pour N nombre de points de données.

Comme le temps de calcul augmente proportionnellement au nombre de n_probes, veuillez consulter la précision et choisir de manière appropriée. Personnellement, je ne pense pas que RVC ait besoin de tant de précision, donc n_probe = 1 est bien.

## FastScan
FastScan est une méthode qui permet d'approximer rapidement les distances par quantification de produit cartésien en les effectuant dans les registres.
La quantification du produit cartésien effectue un regroupement indépendamment

 pour chaque dimension d (généralement d = 2) pendant l'apprentissage, calcule la distance entre les clusters à l'avance, et crée une table de recherche. Au moment de la prédiction, la distance de chaque dimension peut être calculée en O(1) en consultant la table de recherche.
Le nombre que vous spécifiez après PQ spécifie généralement la moitié de la dimension du vecteur.

Pour une description plus détaillée de FastScan, veuillez consulter la documentation officielle.
https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)

## RFlat
RFlat est une instruction pour recalculer la distance approximative calculée par FastScan avec la distance exacte spécifiée par le troisième argument de l'usine d'index.
Lors de l'obtention de k voisins, k*k_factor points sont recalculés.
