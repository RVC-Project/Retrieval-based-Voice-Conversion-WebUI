
### 2023-08-13
1- Düzenli hata düzeltmeleri
- Minimum toplam epoch sayısını 1 olarak değiştirin ve minimum toplam epoch sayısını 2 olarak değiştirin
- Ön eğitim modellerini kullanmama nedeniyle oluşan eğitim hatalarını düzeltin
- Eşlik eden vokallerin ayrılmasından sonra grafik belleğini temizleyin
- Faiss kaydetme yolu mutlak yoldan göreli yola değiştirilmiştir
- Boşluk içeren yolu destekleyin (hem eğitim kümesi yolu hem de deney adı desteklenir ve artık hata rapor edilmez)
- Filelist, zorunlu utf8 kodlamasını iptal eder
- Gerçek zamanlı ses değişikliği sırasında faiss aramasından kaynaklanan CPU tüketim sorununu çözün

2- Temel güncellemeler
- Geçerli en güçlü açık kaynak vokal ton çıkarma modeli RMVPE'yi eğitin ve RVC eğitimi, çevrimdışı/gerçek zamanlı çıkarım için kullanın, PyTorch/Onnx/DirectML destekler
- Pytorch_DML aracılığıyla AMD ve Intel grafik kartları için destek ekleyin

(1) Gerçek zamanlı ses değişimi (2) Çıkarım (3) Vokal eşlik ayrımı (4) Şu anda desteklenmeyen eğitim, CPU eğitimine geçiş yapacaktır; Onnx_Dml ile gpu için RMVPE çıkarımını destekler


### 2023-06-18
- Yeni ön eğitilmiş v2 modeller: 32k ve 48k
- F0 modeli çıkarım hatalarını düzeltme
- Eğitim kümesi 1 saati aşarsa, özelliği şekil açısından küçültmek için otomatik minibatch-kmeans yapın, böylece indeks eğitimi, eklemesi ve araması çok daha hızlı olur.
- Bir oyunca vokal2guitar huggingface alanı sağlama
- Aykırı kısa kesim eğitim kümesi seslerini otomatik olarak silme
- Onnx dışa aktarma sekmesi

Başarısız deneyler:
- ~~Özellik çıkarımı: zamansal özellik çıkarımı ekleme: etkili değil~~
- ~~Özellik çıkarımı: PCAR boyut azaltma ekleme: arama daha yavaş~~
- ~~Eğitim sırasında rastgele veri artırma: etkili değil~~

Yapılacaklar listesi:
- ~~Vocos-RVC (küçük vokoder): etkili değil~~
- ~~Eğitim için Crepe desteği: RMVPE ile değiştirildi~~
- ~~Yarı hassas Crepe çıkarımı: RMVPE ile değiştirildi. Ve zor gerçekleştirilebilir.~~
- F0 düzenleyici desteği

### 2023-05-28
- v2 jupyter notebook, korece değişiklik günlüğü, bazı çevre gereksinimlerini düzeltme
- Sesli olmayan ünsüz ve nefes koruma modu ekleme
- Crepe-full ton algılama desteği ekleme
- UVR5 vokal ayrımı: yankı kaldırma modelleri ve yankı kaldırma modelleri destekleme
- İndeks adında deney adı ve sürüm ekleme
- Toplu ses dönüşüm işleme ve UVR5 vokal ayrımı sırasında çıkış seslerinin ihracat formatını kullanıcıların manuel olarak seçmelerine olanak tanıma
- v1 32k model eğitimi artık desteklenmiyor

### 2023-05-13
- Tek tıklamayla paketin eski sürümündeki çalışma zamanındaki gereksiz kodları temizleme: lib.infer_pack ve uvr5_pack
- Eğitim seti ön işleme içindeki sahte çoklu işlem hatasını düzeltme
- Harvest ton tanıma algoritması için ortanca filtre yarıçap ayarı ekleme
- Çıkış sesi için örnek alma örneği için yeniden örnekleme desteği ekleme
- Eğitim için "n_cpu" çoklu işlem ayarı, "f0 çıkarma" yerine "veri ön işleme ve f0 çıkarma" için değiştirildi
- Günlükler klasörü altındaki indeks yollarını otomatik olarak tespit etme ve bir açılır liste işlevi sağlama
- Sekme sayfasına "Sıkça Sorulan Sorular ve Cevaplar"ı ekleme (ayrıca github RVC wiki'ye de bakabilirsiniz)
- Çıkarım sırasında aynı giriş sesi yolunu kullanırken harvest tonunu önbelleğe alma (amaç: harvest ton çıkarımı kullanırken, tüm işlem hattı uzun ve tekrarlayan bir ton çıkarım işlemi geçirecektir. Önbellekleme kullanılmazsa, farklı timbre, indeks ve ton ortanca filtreleme yarıçapı ayarlarıyla deney yapan kullanıcılar, ilk çıkarım sonrası çok acı verici bir bekleme süreci yaşayacaktır)

### 2023-05-14
- Girişin hacim zarfını çıktının hacim zarfıyla karıştırmak veya değiştirmek için girişin hacim zarfını kullanma (problemi "giriş sessizleştirme ve çıktı küçük

 amplitüdlü gürültü" sorununu hafifletebilir. Giriş sesi arka plan gürültüsü yüksekse, açık olması önerilmez ve varsayılan olarak açık değildir (1 varsayılan olarak kapalı olarak kabul edilir)
- Belirli bir frekansta filtreleme uygulama eğitim ve çıkarım için 50Hz'nin altındaki frekans bantları için
- Pyworld'un varsayılan 80'den 50'ye minimum ton çıkarma sınırlamasını eğitim ve çıkarım için düşürme, 50-80Hz arasındaki erkek alçak seslerin sessizleştirilmemesine izin verme
- WebUI, sistem yereli diline göre dil değiştirme (şu anda en_US, ja_JP, zh_CN, zh_HK, zh_SG, zh_TW'yi destekliyor; desteklenmeyen durumda varsayılan olarak en_US'ye geçer)
- Belirli bir giriş sesi yolunu kullanırken harvest tonunu önbelleğe alma (amaç: harvest ton çıkarma kullanırken, tüm işlem hattı uzun ve tekrarlayan bir ton çıkarma süreci geçirecektir. Önbellekleme kullanılmazsa, farklı timbre, indeks ve ton ortanca filtreleme yarıçapı ayarlarıyla deney yapan kullanıcılar, ilk çıkarım sonrası çok acı verici bir bekleme süreci yaşayacaktır)

### 2023-04-09 Güncellemesi
- GPU kullanım oranını artırmak için eğitim parametrelerini düzeltme: A100, %25'ten yaklaşık %90'a, V100: %50'den yaklaşık %90'a, 2060S: %60'tan yaklaşık %85'e, P40: %25'ten yaklaşık %95'e; eğitim hızını önemli ölçüde artırma
- Parametre değişti: toplam_batch_size artık GPU başına batch_size
- Toplam_epoch değişti: maksimum sınırı 1000'e yükseltildi; varsayılan 10'dan 20'ye yükseltildi
- ckpt çıkarımı ile çalma tanıma hatasını düzeltme, anormal çıkarım oluşturan
- Dağıtılmış eğitimde her sıra için ckpt kaydetme sorununu düzeltme
- Özellik çıkarımı için NaN özellik filtrelemesi uygulama
- Sessiz giriş/çıkışın rastgele ünsüzler veya gürültü üretme sorununu düzeltme (eski modeller yeni bir veri kümesiyle tekrar eğitilmelidir)

### 2023-04-16 Güncellemesi
- Yerel gerçek zamanlı ses değiştirme mini-GUI'si ekleme, çift tıklayarak go-realtime-gui.bat ile başlayın
- Eğitim ve çıkarım sırasında 50Hz'nin altındaki frekans bantlarını filtreleme uygulama
- Pyworld'deki varsayılan 80'den 50'ye minimum ton çıkarma sınırlamasını eğitim ve çıkarım için düşürme, 50-80Hz arasındaki erkek alçak seslerin sessizleştirilmemesine izin verme
- WebUI, sistem yereli diline göre dil değiştirme (şu anda en_US, ja_JP, zh_CN, zh_HK, zh_SG, zh_TW'yi destekliyor; desteklenmeyen durumda varsayılan olarak en_US'ye geçer)
- Bazı GPU'ların tanınmasını düzeltme (örneğin, V100-16G tanınmama sorunu, P4 tanınmama sorunu)

### 2023-04-28 Güncellemesi
- Daha hızlı hız ve daha yüksek kalite için faiss indeks ayarlarını yükseltme
- Toplam_npy bağımlılığını kaldırma; gelecekteki model paylaşımları için total_npy girdisi gerekmeyecek
- 16-serisi GPU'lar için kısıtlamaları açma, 4GB VRAM GPU'lar için 4GB çıkarım ayarları sağlama
- Belirli ses biçimlerine yönelik UVR5 vokal eşlik ayrımındaki hata düzeltme
- Gerçek zamanlı ses değiştirme mini-GUI şimdi 40k dışı ve tembel ton modellerini destekler

### Gelecekteki Planlar:
Özellikler:
- Her epoch kaydetmek için küçük modeller çıkar seçeneğini ekleme
- Çıkarım sırasında çıkış seslerini belirtilen yolda ekstra mp3 olarak kaydetme seçeneğini ekleme
- Birden fazla kişinin eğitim sekmesini destekleme (en fazla 4 kişiye kadar)
  
Temel model:
- Bozuk nefes seslerinin sorununu düzeltmek için nefes alma wav dosyalarını eğitim veri kümesine eklemek
- Şu anda genişletilmiş bir şarkı veri kümesiyle temel model eğitimi yapıyoruz ve gelecekte yayınlanacak
