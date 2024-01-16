pONTAS de afinação FAISS
==================
# sobre faiss
faiss é uma biblioteca de pesquisas de vetores densos na área, desenvolvida pela pesquisa do facebook, que implementa com eficiência muitos métodos de pesquisa de área aproximada.
A Pesquisa Aproximada de área encontra vetores semelhantes rapidamente, sacrificando alguma precisão.

## faiss em RVC
No RVC, para a incorporação de recursos convertidos pelo HuBERT, buscamos incorporações semelhantes à incorporação gerada a partir dos dados de treinamento e as misturamos para obter uma conversão mais próxima do discurso original. No entanto, como essa pesquisa leva tempo se realizada de forma ingênua, a conversão de alta velocidade é realizada usando a pesquisa aproximada de área.

# visão geral da implementação
Em '/logs/nome-do-seu-modelo/3_feature256', onde o modelo está localizado, os recursos extraídos pelo HuBERT de cada dado de voz estão localizados.
A partir daqui, lemos os arquivos npy ordenados por nome de arquivo e concatenamos os vetores para criar big_npy. (Este vetor tem a forma [N, 256].)
Depois de salvar big_npy as /logs/nome-do-seu-modelo/total_fea.npy, treine-o com faiss.

Neste artigo, explicarei o significado desses parâmetros.

# Explicação do método
## Fábrica de Index
Uma fábrica de Index é uma notação faiss exclusiva que expressa um pipeline que conecta vários métodos de pesquisa de área aproximados como uma string.
Isso permite que você experimente vários métodos aproximados de pesquisa de área simplesmente alterando a cadeia de caracteres de fábrica do Index.
No RVC é usado assim:

```python
index = faiss.index_factory(256, "IVF%s,Flat" % n_ivf)
```
Entre os argumentos de index_factory, o primeiro é o número de dimensões do vetor, o segundo é a string de fábrica do Index e o terceiro é a distância a ser usada.

Para uma notação mais detalhada
https://github.com/facebookresearch/faiss/wiki/The-index-factory

## Construção de Index
Existem dois Indexs típicos usados como similaridade de incorporação da seguinte forma.

- Distância euclidiana (MÉTRICA_L2)
- Produto interno (METRIC_INNER_PRODUCT)

A distância euclidiana toma a diferença quadrática em cada dimensão, soma as diferenças em todas as dimensões e, em seguida, toma a raiz quadrada. Isso é o mesmo que a distância em 2D e 3D que usamos diariamente.
O produto interno não é usado como um Index de similaridade como é, e a similaridade de cosseno que leva o produto interno depois de ser normalizado pela norma L2 é geralmente usada.

O que é melhor depende do caso, mas a similaridade de cosseno é frequentemente usada na incorporação obtida pelo word2vec e modelos de recuperação de imagem semelhantes aprendidos pelo ArcFace. Se você quiser fazer a normalização l2 no vetor X com numpy, você pode fazê-lo com o seguinte código com eps pequeno o suficiente para evitar a divisão 0.

```python
X_normed = X / np.maximum(eps, np.linalg.norm(X, ord=2, axis=-1, keepdims=True))
```

Além disso, para a Construção de Index, você pode alterar o Index de distância usado para cálculo escolhendo o valor a ser passado como o terceiro argumento.

```python
index = faiss.index_factory(dimention, text, faiss.METRIC_INNER_PRODUCT)
```

## FI
IVF (Inverted file indexes) é um algoritmo semelhante ao Index invertido na pesquisa de texto completo.
Durante o aprendizado, o destino da pesquisa é agrupado com kmeans e o particionamento Voronoi é realizado usando o centro de cluster. A cada ponto de dados é atribuído um cluster, por isso criamos um dicionário que procura os pontos de dados dos clusters.

Por exemplo, se os clusters forem atribuídos da seguinte forma
|index|Cluster|
|-----|-------|
|1|A|
|2|B|
|3|A|
|4|C|
|5|B|

O Index invertido resultante se parece com isso:

| cluster | Index |
|-------|-----|
| A | 1, 3 |
| B | 2 5 |
| C | 4 |

Ao pesquisar, primeiro pesquisamos n_probe clusters dos clusters e, em seguida, calculamos as distâncias para os pontos de dados pertencentes a cada cluster.

# Parâmetro de recomendação
Existem diretrizes oficiais sobre como escolher um Index, então vou explicar de
acordo. https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

Para conjuntos de dados abaixo de 1M, o 4bit-PQ é o método mais eficiente disponível no faiss em abril de 2023.
Combinando isso com a fertilização in vitro, estreitando os candidatos com 4bit-PQ e, finalmente, recalcular a distância com um Index preciso pode ser descrito usando a seguinte fábrica de Indexs.

```python
index = faiss.index_factory(256, "IVF1024,PQ128x4fs,RFlat")
```

## Parâmetros recomendados para FIV
Considere o caso de muitas FIVs. Por exemplo, se a quantização grosseira por FIV for realizada para o número de dados, isso é o mesmo que uma pesquisa exaustiva ingênua e é ineficiente.
Para 1M ou menos, os valores de FIV são recomendados entre 4*sqrt(N) ~ 16*sqrt(N) para N número de pontos de dados.

Como o tempo de cálculo aumenta proporcionalmente ao número de n_sondas, consulte a precisão e escolha adequadamente. Pessoalmente, não acho que o RVC precise de tanta precisão, então n_probe = 1 está bem.

## FastScan
O FastScan é um método que permite a aproximação de alta velocidade de distâncias por quantização de produto cartesiano, realizando-as em registros.
A quantização cartesiana do produto executa o agrupamento independentemente para cada dimensão d (geralmente d = 2) durante o aprendizado, calcula a distância entre os agrupamentos com antecedência e cria uma tabela de pesquisa. No momento da previsão, a distância de cada dimensão pode ser calculada em O(1) olhando para a tabela de pesquisa.
Portanto, o número que você especifica após PQ geralmente especifica metade da dimensão do vetor.

Para uma descrição mais detalhada do FastScan, consulte a documentação oficial.
https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)

## RFlat
RFlat é uma instrução para recalcular a distância aproximada calculada pelo FastScan com a distância exata especificada pelo terceiro argumento da Construção de Index.
Ao obter áreas k, os pontos k*k_factor são recalculados.
