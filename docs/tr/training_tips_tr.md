RVC Eğitimi için Talimatlar ve İpuçları
===========================================

Bu TIPS, veri eğitiminin nasıl yapıldığını açıklar.

# Eğitim Süreci
Eğitim sekmesinde adımları takip ederek açıklayacağım.

## Adım 1
Burada deney adını ayarlayın.

Ayrıca burada modelin pitch'i dikkate alıp almayacağını da belirtebilirsiniz.
Eğer model pitch'i dikkate almazsa, model daha hafif olacak ancak şarkı söyleme için uygun olmayacaktır.

Her deney için veriler `/logs/deney-adınız/` klasörüne yerleştirilir.

## Adım 2a
Ses yüklenir ve ön işlem yapılır.

### Ses yükleme
Ses içeren bir klasörü belirtirseniz, o klasördeki ses dosyaları otomatik olarak okunacaktır.
Örneğin, `C:Kullanıcılar\hoge\sese` gibi bir klasör belirtirseniz, `C:Kullanıcılar\hoge\sese\voice.mp3` yüklenecek, ancak `C:Kullanıcılar\hoge\sese\klasör\voice.mp3` yüklenecektir.

Ses okumak için dahili olarak ffmpeg kullanıldığından, uzantı ffmpeg tarafından destekleniyorsa otomatik olarak okunacaktır.
ffmpeg ile int16'ya dönüştürüldükten sonra, float32'ye çevrilir ve -1 ile 1 arasında normalize edilir.

### Gürültü Temizleme
Ses, scipy'nin filtfilt fonksiyonu ile düzeltilir.

### Ses Ayırma
Önceki işlemlerin ardından giriş sesi, belirli bir süreden (max_sil_kept=5 saniye?) daha uzun süren sessiz bölümleri algılayarak bölünür. Ses sessizlik üzerinde bölündükten sonra, sesi her 4 saniyede bir 0.3 saniyelik bir örtüşme ile bölünür. 4 saniye içinde ayrılan ses için, sesin ses düzeyi normalize edildikten sonra wav dosyasına çevrilir ve `/logs/deney-adınız/0_gt_wavs` klasörüne kaydedilir ve ardından 16k örnekleme hızında `/logs/deney-adınız/1_16k_wavs` klasörüne kaydedilir.

## Adım 2b
### Pitch (Ton Yüksekliği) Çıkarma
Wav dosyalarından pitch bilgisi çıkarılır. Parselmouth veya pyworld tarafından sağlanan yöntem kullanılarak pitch bilgisi (=f0) çıkarılır ve `/logs/deney-adınız/2a_f0` klasöründe kaydedilir. Daha sonra pitch bilgisi logaritmik olarak 1 ile 255 arasında bir tamsayıya dönüştürülür ve `/logs/deney-adınız/2b-f0nsf` klasöründe kaydedilir.

### Özelliklerin Çıkartılması
Wav dosyası, HuBERT kullanılarak önceden gömme olarak çıkartılır. `/logs/deney-adınız/1_16k_wavs` klasöründe kaydedilen wav dosyası okunur, 256 boyutlu özelliklere HuBERT kullanılarak dönüştürülür ve `/logs/deney-adınız/3_feature256` klasöründe npy formatında kaydedilir.

## Adım 3
Modeli eğitin.
### Yeni Başlayanlar İçin Terimler
Derin öğrenmede, veri kümesi bölmeye ve öğrenmeye azar azar devam eder. Bir model güncellemesinde (adım), batch_size veri alınır ve tahminler ve hata düzeltmeleri yapılır. Bunun bir veri kümesi için bir kez yapılması bir epoch olarak sayılır.

Bu nedenle, öğrenme süresi adım başına öğrenme süresi x (veri kümesindeki veri sayısı / batch boyutu) x epoch sayısıdır. Genel olarak, batch boyutu ne kadar büyükse, öğrenme daha istikrarlı olur (adım başına öğrenme süresi ÷ batch boyutu) daha küçük olur, ancak daha fazla GPU belleği kullanır. GPU RAM, nvidia-smi komutu ile kontrol edilebilir. Makineye göre mümkün olduğunca batch boyutunu artırarak kısa sürede öğrenme yapılabilir.

### Önceden Eğitilmiş Modeli Belirtme
RVC, modeli 0'dan değil önceden eğitilmiş ağırlıklardan başlayarak eğitmeye başlar, bu nedenle küçük bir veri kümesiyle eğitilebilir.

Varsayılan olarak

- Eğer pitch'i dikkate alıyorsanız, `rvc-konumu/pretrained/f0G40k.pth` ve `rvc-konumu/pretrained/f0D40k.pth` yüklenir.
- Eğer pitch'i dikkate almıyorsanız, `rvc-konumu/pretrained/f0G40k.pth` ve `rvc-konumu/pretrained/f0D40k.pth` yüklenir.

Eğitim sırasında, model parametreleri `logs/deney-adınız/G_{}.pth` ve `logs/deney-adınız/D_{}.pth` olarak her save_every_epoch için kaydedilir, ancak bu yolu belirterek eğitimi başlatabilirsiniz. Farklı bir deneyde öğrenilen model ağırlıklarından eğitime yeniden başlatabilir veya yeni başlatabilirsiniz.

### Öğrenme İndeksi
RVC, eğitim sırasında kullanılan HuBERT özellik değerlerini kaydeder ve çıkarım sırasında eğitim sırasında kullanılan özellik değerlerine ben

zer özellik değerlerini aramak için çıkarım yapar. Bu aramayı yüksek hızda gerçekleştirmek için indeksi önceden öğrenir.
İndeks öğrenimi için, yaklaşık komşuluk arama kütüphanesi faiss kullanılır. `/logs/deney-adınız/3_feature256` klasöründe kaydedilen özellik değerini okuyarak indeks öğrenimi yapılır ve `logs/deney-adınız/add_XXX.index` olarak kaydedilir.

(20230428 güncelleme sürümünden itibaren, indeks okunur ve kaydetme / belirtme artık gerekli değildir.)

### Buton açıklamaları
- Modeli Eğit: Adım 2b'yi tamamladıktan sonra, modeli eğitmek için bu düğmeye basın.
- Özellik İndeksini Eğit: Model eğitimini tamamladıktan sonra, indeks öğrenimini yapmak için bu düğmeye basın.
- Tek Tıkla Eğitim: Adım 2b, model eğitimi ve özellik indeksi eğitimi hepsi bir arada.