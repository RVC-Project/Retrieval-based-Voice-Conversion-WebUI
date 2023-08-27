### 2023-06-18
- Yeni önceden eğitilmiş v2 modelleri: 32k ve 48k
- F0 olmayan model çıkarımlarındaki hatalar düzeltildi
- Eğitim kümesi 1 saatini aşarsa, özelliğin boyutunu azaltmak için otomatik minibatch-kmeans yapılır, böylece indeks eğitimi, ekleme ve arama işlemleri çok daha hızlı olur.
- Oyuncak sesden gitar huggingface alanı sağlanır
- Aykırı kısa kesme eğitim kümesi sesleri otomatik olarak silinir
- Onnx dışa aktarma sekmesi

Başarısız deneyler:
- ~~Özellik çıkarımı: zamansal özellik çıkarımı ekleme: etkili değil~~
- ~~Özellik çıkarımı: PCAR boyut indirgeme ekleme: arama daha da yavaş~~
- ~~Eğitimde rastgele veri artırma: etkili değil~~

Yapılacaklar listesi:
- Vocos-RVC (küçük vokoder)
- Eğitim için Crepe desteği
- Yarı hassas Crepe çıkarımı
- F0 düzenleyici desteği

### 2023-05-28
- v2 jupyter not defteri eklendi, korece değişiklik günlüğü eklendi, bazı ortam gereksinimleri düzeltildi
- Sesli olmayan ünsüz ve nefes koruma modu eklendi
- Crepe-full pitch algılama desteği eklendi
- UVR5 vokal ayırma: dereverb ve de-echo modellerini destekler
- İndeksin adında deney adı ve sürümünü ekleyin
- Toplu ses dönüşüm işlemi ve UVR5 vokal ayırma sırasında çıktı seslerinin ihracat formatını manuel olarak seçme desteği eklendi
- v1 32k model eğitimi artık desteklenmiyor

### 2023-05-13
- Tek tıklamalı paketin eski sürümündeki gereksiz kodlar temizlendi: lib.infer_pack ve uvr5_pack
- Eğitim kümesi ön işlemesinde sahte çok işlem hatası düzeltildi
- Harvest pitch algı algoritması için median filtre yarıçapı ayarlama eklendi
- Ses ihracatı için yeniden örnekleme desteği eklendi
- Eğitimde "n_cpu" için çoklu işlem ayarı "f0 çıkarma" dan "veri ön işleme ve f0 çıkarma" olarak değiştirildi
- İndex yolu otomatik olarak algılanır ve açılır liste işlevi sağlanır
- Sekme sayfasında "Sık Sorulan Sorular ve Cevaplar" eklendi (ayrıca github RVC wiki'ye bakabilirsiniz)
- Çıkarım sırasında, aynı giriş sesi yolu kullanıldığında harvest pitch önbelleğe alınır (amaç: harvest pitch çıkarma kullanılırken, tüm işlem süreci uzun ve tekrarlayan bir pitch çıkarma sürecinden geçer. Önbellek kullanılmazsa, farklı timbre, index ve pitch median filtre yarıçapı ayarlarıyla deney yapan kullanıcılar ilk çıkarımın ardından çok acı verici bir bekleme süreci yaşayacaktır)

### 2023-05-14
- Girişin ses hacmini çıkışın ses hacmiyle karıştırma veya değiştirme seçeneği eklendi ( "giriş sessiz ve çıkış düşük amplitütlü gürültü" sorununu hafifletmeye yardımcı olur. Giriş sesinin arka plan gürültüsü yüksekse, önerilmez ve varsayılan olarak kapalıdır (1 kapalı olarak düşünülebilir)
- Çıkarılan küçük modellerin belirli bir sıklıkta kaydedilmesini destekler (farklı epoch altındaki performansı görmek istiyorsanız, ancak tüm büyük kontrol noktalarını kaydetmek istemiyor ve her seferinde ckpt-processing ile küçük modelleri manuel olarak çıkarmak istemiyorsanız, bu özellik oldukça pratik olacaktır)
- Sunucunun genel proxy'sinin neden olduğu "bağlantı hataları" sorununu, çevre değişkenleri ayarlayarak çözer
- Önceden eğitilmiş v2 modelleri destekler (şu anda sadece 40k sürümleri test için kamuya açıktır ve diğer iki örnekleme hızı henüz tam olarak eğitilmemiştir)
- İnferans öncesi aşırı ses hacmi 1'i aşmasını engeller
- Eğitim kümesinin ayarlarını hafifçe düzeltildi

#######################

Geçmiş değişiklik günlükleri:

### 2023-04-09
- GPU kullanım oranını artırmak için eğitim parametreleri düzeltilerek: A100% 25'ten yaklaşık 90'a, V100: %50'den yaklaşık 90'a, 2060S: %60'dan yaklaşık 85'e, P40: %25'ten yaklaşık 95'e; eğitim hızı önemli ölçüde artırıldı
- Parametre değiştirildi: toplam batch_size artık her GPU için batch_size
- Toplam_epoch değiştirildi: maksimum sınır 100'den 1000'e yükseltildi; varsayılan 10'dan 20'ye yükseltildi
- Ckpt çıkarımı sırasında pitch yanlış tanıma nedeniyle oluşan anormal çıkarım sorunu

 düzeltildi
- Dağıtılmış eğitimde her sıra için ckpt kaydetme sorunu düzeltildi
- Özellik çıkarımında nan özellik filtreleme uygulandı
- Giriş/çıkış sessiz üretildiğinde rastgele ünsüzler veya gürültü üretme sorunu düzeltildi (eski modeller yeni bir veri kümesiyle yeniden eğitilmelidir)

### 2023-04-16 Güncellemesi
- Yerel gerçek zamanlı ses değiştirme mini-GUI eklendi, go-realtime-gui.bat dosyasını çift tıklatarak başlayın
- Eğitim ve çıkarımda 50Hz'nin altındaki frekans bantları için filtreleme uygulandı
- Eğitim ve çıkarımda pyworld'ün varsayılan 80'den 50'ye düşürüldü, böylece 50-80Hz aralığındaki erkek düşük perdeli seslerin sessiz kalmaması sağlandı
- WebUI, sistem yereli diline göre dil değiştirme desteği ekledi (şu anda en_US, ja_JP, zh_CN, zh_HK, zh_SG, zh_TW'yi desteklemektedir; desteklenmezse varsayılan olarak en_US kullanılır)
- Bazı GPU'ların tanınmasında sorun giderildi (örneğin, V100-16G tanınma hatası, P4 tanınma hatası)

### 2023-04-28 Güncellemesi
- Daha hızlı hız ve daha yüksek kalite için faiss indeks ayarları yükseltildi
- total_npy bağımlılığı kaldırıldı; gelecekteki model paylaşımı total_npy girişi gerektirmeyecek
- 16 serisi GPU'lar için kısıtlamalar kaldırıldı, 4GB VRAM GPU'ları için 4GB çıkarım ayarları sağlanıyor
- Belirli ses biçimleri için UVR5 vokal eşlik ayırma hatası düzeltildi
- Gerçek zamanlı ses değiştirme mini-GUI, 40k dışında ve tembelleştirilmemiş pitch modellerini destekler hale geldi

### Gelecek Planlar:
Özellikler:
- Her epoch kaydetmek için küçük modelleri çıkarma seçeneği ekle
- Çıkarım sırasında çıktı sesleri için belirli bir yola ekstra mp3'leri kaydetme seçeneği ekle
- Birden çok kişi eğitim sekmesini destekle (en fazla 4 kişiye kadar)