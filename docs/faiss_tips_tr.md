faiss ayarları hakkında ipuçları
=============================

# faiss hakkında
faiss, facebook araştırma tarafından geliştirilen, yoğun vektörler için yakınsaklık aramaları için bir kütüphanedir ve birçok yaklaşık yakınsaklık arama yöntemini verimli bir şekilde uygular.
Yaklaşık Yakınsaklık Arama, biraz doğruluktan ödün vererek benzer vektörleri hızlı bir şekilde bulur.

## RVC'de faiss
RVC'de, HuBERT tarafından dönüştürülen özelliklerin gömülmesi için eğitim verilerinden oluşturulan gömülmelerle benzer gömülmeleri arar ve onları karıştırarak orijinal konuşmaya daha yakın bir dönüşüm elde ederiz. Ancak, bu arama zaman alıyorsa, yaklaşık yakınsaklık arama kullanarak yüksek hızlı dönüşüm elde edilir.

# Uygulama genel bakışı
Modelin bulunduğu '/logs/your-experiment/3_feature256' dizininde, her ses verisinden HuBERT tarafından çıkarılan özellikler bulunur.
Burası, dosya adına göre sıralanmış npy dosyalarını okuyarak vektörleri birleştirerek büyük npy oluşturur. (Bu vektörün şekli [N, 256].)
Büyük npy, /logs/your-experiment/total_fea.npy olarak kaydedildikten sonra faiss ile eğitilir.

Bu makalede, bu parametrelerin anlamını açıklayacağım.

# Yöntemin Açıklaması
## indeks fabrikası
Bir indeks fabrikası, birden çok yaklaşık yakınsaklık arama yöntemini bir dize olarak bağlayan benzersiz bir faiss gösterimidir.
Bu, indeks fabrikası dizesini değiştirerek kolayca çeşitli yaklaşık yakınsaklık arama yöntemlerini denemenize olanak tanır.
RVC'de bunu şu şekilde kullanıyoruz:

```python
index = faiss.index_factory(256, "IVF%s,Flat" % n_ivf)
```
index_factory'nin argümanları arasında ilk olarak vektörün boyutu, ikinci olarak indeks fabrikası dizesi ve üçüncü olarak kullanılacak mesafe bulunur.

Daha ayrıntılı gösterim için
https://github.com/facebookresearch/faiss/wiki/The-index-factory

## mesafe için indeks
Aşağıda gömülmenin benzerliğinde kullanılan iki tipik indeks bulunur.

- Öklidyen mesafesi (METRIC_L2)
- iç çarpım (METRIC_INNER_PRODUCT)

Öklidyen mesafesi, her boyutta kare farkı alır, tüm boyutlardaki farkları toplar ve ardından karekökünü alır. Bu, günlük hayatta kullandığımız 2D ve 3D'deki mesafeyle aynıdır.
İç çarpım, doğrudan bir benzerlik indeksi olarak kullanılmaz, genellikle L2 normuyla normalize edildikten sonra iç çarpım alınan kosinüs benzerliği kullanılır.

Hangisinin daha iyi olduğu duruma bağlıdır, ancak word2vec tarafından elde edilen gömülme ve ArcFace ile öğrenilmiş benzer görüntü arama modellerinde genellikle kosinüs benzerliği kullanılır. numpy ile X vektörüne l2 normalizasyonu yapmak için aşağıdaki kodu eps değerini sıfıra bölme hatasından kaçınmak için yeterince küçük bir değerle kullanabilirsiniz.

```python
X_normed = X / np.maximum(eps, np.linalg.norm(X, ord=2, axis=-1, keepdims=True))
```

Ayrıca, indeks fabrikasında hesaplama için kullanılan mesafe indeksini üçüncü argüman olarak geçerek hesaplanan mesafeyi değiştirebilirsiniz.

```python
index = faiss.index_factory(dimention, text, faiss.METRIC_INNER_PRODUCT)
```

## IVF
IVF (Ters dosya indeksleri), tam metin aramasındaki ters indekse benzer bir algoritmadır.
Öğrenme sırasında, arama hedefi kmeans ile kümeleme yapılır ve küme merkezi ile Voronoi bölümlenmesi yapılır. Her veri noktası bir kümeye atanır, bu nedenle veri noktalarını kümelelerden arayan bir sözlük oluştururuz.

Örneğin, kümeler şu şekilde atanırsa:
|index|Küme|
|-----|-------|
|1|A|
|2|B|
|3|A|
|4|C|
|5|B|

Sonuçta elde edilen ters indeks aşağıdaki gibi görünecektir:

|küme|index|
|-------|-----|
|A|1, 3|
|B|2, 5|
|C|4|

Arama yaparken, önce kümelerden n_probe kümeleri arar ve ardından her kümeye ait veri noktalarının mesafesini hesaplar.

# Önerilen parametreler
Önerilen bir indeks seçme konusunda resmi yönergeler bulunur, bu nedenle buna göre açıklayacağım.
https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

1M'den küçük veri kümeleri için, Nisan 2023 itibarıyla faiss tarafından mevcut olan en verimli yöntem 4bit-PQ'dir.
Bunu IVF ile birleştirerek, 4bit-PQ ile adayları daraltabilir ve nihayetinde doğru bir indeksle mesafeyi yeniden hesaplayarak aşağıdaki indeks fabrikasını kullanarak tanımlayabiliriz.



```python
index = faiss.index_factory(256, "IVF1024,PQ128x4fs,RFlat")
```

## IVF için Önerilen Parametreler
Çok fazla IVF'nin olduğu durumu düşünün. Örneğin, IVF tarafından verilerin sayısı için kalın nicelleme yapıldığında, bu, basit bir tam arama ile aynıdır ve verimsizdir.
1M veya daha az için IVF değerleri, veri noktalarının N sayısı için 4*sqrt(N) ~ 16*sqrt(N) arasında önerilir.

n_probes sayısı arttıkça hesaplama süresi arttığından, doğruluk ile danışın ve uygun şekilde seçin. Kişisel olarak RVC'nin bu kadar hassas olmasını gerektiren bir durum olmadığını düşünüyorum, bu nedenle n_probe = 1 yeterlidir.

## FastScan
FastScan, bunları kayıtlarda gerçekleştirerek onları kartez ürün kuantizasyonu ile yüksek hızda mesafeye yaklaşık olarak yapılmasını sağlayan bir yöntemdir.
Kartez ürün kuantizasyonu, öğrenme sırasında her d boyut için (genellikle d = 2) bağımsız olarak kümeleme yapar, küme merkezleri arasındaki mesafeyi önceden hesaplar ve bir arama tablosu oluşturur. Tahmin sırasında her boyutun mesafesi, arama tablosuna bakarak O(1) olarak hesaplanabilir.
Bu nedenle PQ'dan sonra belirttiğiniz sayı genellikle vektörün yarısı olarak belirtir.

FastScan hakkında daha ayrıntılı bilgi için lütfen resmi belgelere başvurun.
https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)

## RFlat
RFlat, FastScan ile hesaplanan yaklaşık mesafeyi indeks fabrikasının üçüncü argümanı ile belirtilen tam mesafe ile yeniden hesaplamak için bir talimattır.
K-en yakın komşuyu alırken, k*k_factor kadar nokta yeniden hesaplanır.