
# faiss Ayar İpuçları
==================

# faiss Hakkında
faiss, yoğun vektörler için komşuluk aramalarının bir kütüphanesidir ve birçok yaklaşık komşuluk arama yöntemini verimli bir şekilde uygular. Facebook araştırma tarafından geliştirilen faiss, benzer vektörleri hızlı bir şekilde bulurken bazı doğruluğu feda eder.

## RVC'de faiss Kullanımı
RVC'de, HuBERT tarafından dönüştürülen özelliklerin gömülmesi için eğitim verisinden oluşturulan gömme ile benzer gömlemeleri ararız ve bunları karıştırarak orijinal konuşmaya daha yakın bir dönüşüm elde ederiz. Ancak bu arama basitçe yapıldığında zaman alır, bu nedenle yaklaşık komşuluk araması kullanarak yüksek hızlı dönüşüm sağlanır.

# Uygulama Genel Bakış
Modelin bulunduğu '/logs/your-experiment/3_feature256' dizininde, her ses verisinden HuBERT tarafından çıkarılan özellikler bulunur.
Buradan, dosya adına göre sıralanmış npy dosyalarını okuyarak vektörleri birleştirip büyük_npy'yi oluştururuz. (Bu vektörün şekli [N, 256] şeklindedir.)
Büyük_npy'yi /logs/your-experiment/total_fea.npy olarak kaydettikten sonra, onu faiss ile eğitiriz.

Bu makalede, bu parametrelerin anlamını açıklayacağım.

# Yöntemin Açıklaması
## İndeks Fabrikası
Bir indeks fabrikası, birden fazla yaklaşık komşuluk arama yöntemini bir dizi olarak bağlayan benzersiz bir faiss gösterimidir. Bu, indeks fabrikası dizesini değiştirerek basitçe çeşitli yaklaşık komşuluk arama yöntemlerini denemenizi sağlar.
RVC'de bunu şu şekilde kullanırız:

```python
index = faiss.index_factory(256, "IVF%s,Flat" % n_ivf)
```
index_factory'nin argümanları arasında ilk vektör boyutu, ikinci indeks fabrikası dizesi ve üçüncü kullanılacak mesafe yer alır.

Daha ayrıntılı gösterim için
https://github.com/facebookresearch/faiss/wiki/The-index-factory

## Mesafe İçin İndeks
Aşağıdaki gibi gömme benzerliği olarak kullanılan iki tipik indeks bulunur.

- Öklidyen mesafe (METRIC_L2)
- iç çarpım (METRIC_INNER_PRODUCT)

Öklidyen mesafe, her boyutta karesel farkı alır, tüm boyutlardaki farkları toplar ve ardından karekök alır. Bu, günlük hayatta kullandığımız 2D ve 3D'deki mesafeye benzer.
İç çarpım, çoğunlukla L2 norm ile normalize edildikten sonra iç çarpımı alan ve genellikle kosinüs benzerliği olarak kullanılan bir benzerlik göstergesi olarak kullanılır.

Hangisinin daha iyi olduğu duruma bağlıdır, ancak kosinüs benzerliği genellikle word2vec tarafından elde edilen gömme ve ArcFace tarafından öğrenilen benzer görüntü alım modellerinde kullanılır. Vektör X'i numpy ile l2 normalize yapmak isterseniz, 0 bölme hatasından kaçınmak için yeterince küçük bir eps ile şu kodu kullanabilirsiniz:

```python
X_normed = X / np.maximum(eps, np.linalg.norm(X, ord=2, axis=-1, keepdims=True))
```

Ayrıca, indeks fabrikası için üçüncü argüman olarak geçirilecek değeri seçerek hesaplamada kullanılan mesafe indeksini değiştirebilirsiniz.

```python
index = faiss.index_factory(dimention, text, faiss.METRIC_INNER_PRODUCT)
```

## IVF
IVF (Ters dosya indeksleri), tam metin aramasındaki ters indeksle benzer bir algoritmadır.
Öğrenme sırasında, arama hedefi kmeans ile kümelendirilir ve küme merkezi kullanılarak Voronoi bölütleme gerçekleştirilir. Her veri noktasına bir küme atanır, bu nedenle veri noktalarını kümeden arayan bir sözlük oluştururuz.

Örneğin, kümelere aşağıdaki gibi atanmışsa
|index|Cluster|
|-----|-------|
|1|A|
|2|B|
|3|A|
|4|C|
|5|B|

Elde edilen ters indeks şu şekildedir:

|cluster|index|
|-------|-----|
|A|1, 3|
|B|2, 5|
|C|4|

Arama yaparken, önce kümeden n_probe küme ararız ve ardından her küme için ait veri noktalarının mesafelerini hesaplarız.

# Tavsiye Edilen Parametreler
Resmi olarak nasıl bir indeks seçileceği konusunda rehberler bulunmaktadır, bu nedenle buna uygun olarak açıklayacağım.
https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

1M'den düşük veri kümeleri için, N sayısı için 4bit-PQ, Nisan 2023 itibariyle faiss'de mevcut en verimli yöntemdir.
Bunu IVF ile birleştirerek adayları 4bit-PQ ile daraltmak ve nihayet doğru bir indeksle mesafeyi yeniden hesaplamak, aşağıdaki indeks fabrikas

ını kullanarak açıklanabilir.

```python
index = faiss.index_factory(256, "IVF1024,PQ128x4fs,RFlat")
```

## IVF İçin Tavsiye Edilen Parametreler
Çok sayıda IVF durumunu düşünün. Örneğin, veri sayısı için IVF tarafından kabaca nicelleme yapılırsa, bu basit bir tükenmez arama ile aynıdır ve verimsizdir.
1M veya daha az için IVF değerleri, N veri noktaları için 4*sqrt(N) ~ 16*sqrt(N) arasında tavsiye edilir.

Hesaplama süresi n_probes sayısına orantılı olarak arttığından, doğrulukla danışmanlık yapın ve uygun şekilde seçin. Kişisel olarak, RVC'nin bu kadar doğruluk gerektirmediğini düşünüyorum, bu nedenle n_probe = 1 uygundur.

## FastScan
FastScan, bunları kaydedicilerde gerçekleştirerek onları Kartez ürünü nicelleme ile hızlı yaklaşık mesafe sağlayan bir yöntemdir.
Kartez ürünü nicelleme öğrenme sırasında her d boyut için (genellikle d = 2) kümeleme yapar, küme merkezlerini önceden hesaplar ve küme merkezleri arasındaki mesafeyi hesaplar ve bir arama tablosu oluşturur. Tahmin yaparken, her boyutun mesafesi arama tablosuna bakarak O(1) hesaplanabilir.
PQ sonrası belirttiğiniz sayı genellikle vektörün yarısı olan boyutu belirtir.

FastScan hakkında daha ayrıntılı açıklama için lütfen resmi belgelere başvurun.
https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)

## RFlat
RFlat, FastScan ile hesaplanan kesirli mesafeyi indeks fabrikasının üçüncü argümanı tarafından belirtilen doğru mesafe ile yeniden hesaplamak için bir talimattır.
k komşuları alırken, k*k_factor nokta yeniden hesaplanır.
