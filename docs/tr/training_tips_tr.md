## RVC Eğitimi için Talimatlar ve İpuçları
======================================
Bu TALİMAT, veri eğitiminin nasıl yapıldığını açıklamaktadır.

# Eğitim Akışı
Eğitim sekmesindeki adımları takip ederek açıklayacağım.

## Adım 1
Deney adını burada belirleyin.

Ayrıca burada modelin pitch'i dikkate alıp almayacağını da belirleyebilirsiniz.
Eğer model pitch'i dikkate almazsa, model daha hafif olacak, ancak şarkı söyleme için uygun olmayacaktır.

Her deney için veriler `/logs/your-experiment-name/` dizinine yerleştirilir.

## Adım 2a
Ses yüklenir ve ön işleme yapılır.

### Ses Yükleme
Ses içeren bir klasör belirtirseniz, bu klasördeki ses dosyaları otomatik olarak okunur.
Örneğin, `C:Users\hoge\voices` belirtirseniz, `C:Users\hoge\voices\voice.mp3` yüklenecek, ancak `C:Users\hoge\voices\dir\voice.mp3` yüklenmeyecektir.

Ses okumak için dahili olarak ffmpeg kullanıldığından, uzantı ffmpeg tarafından destekleniyorsa otomatik olarak okunacaktır.
ffmpeg ile int16'ya dönüştürüldükten sonra float32'ye dönüştürülüp -1 ile 1 arasında normalize edilir.

### Gürültü Temizleme
Ses scipy'nin filtfilt işlevi ile yumuşatılır.

### Ses Ayırma
İlk olarak, giriş sesi belirli bir süreden (max_sil_kept=5 saniye?) daha uzun süren sessiz kısımları tespit ederek böler. Sessizlik üzerinde ses bölündükten sonra sesi 4 saniyede bir 0.3 saniyelik bir örtüşme ile böler. 4 saniye içinde ayrılan sesler için ses normalleştirildikten sonra wav dosyası olarak `/logs/your-experiment-name/0_gt_wavs`'a, ardından 16 kHz örnekleme hızına dönüştürülerek `/logs/your-experiment-name/1_16k_wavs` olarak kaydedilir.

## Adım 2b
### Pitch Çıkarımı
Wav dosyalarından pitch bilgisi çıkarılır. ParSelMouth veya PyWorld'e dahili olarak yerleştirilmiş yöntemi kullanarak pitch bilgisi (=f0) çıkarılır ve `/logs/your-experiment-name/2a_f0` dizinine kaydedilir. Ardından pitch bilgisi logaritmik olarak 1 ile 255 arasında bir tamsayıya dönüştürülüp `/logs/your-experiment-name/2b-f0nsf` dizinine kaydedilir.

### Özellik Çıkarımı
HuBERT'i kullanarak önceden gömme olarak wav dosyasını çıkarır. `/logs/your-experiment-name/1_16k_wavs`'a kaydedilen wav dosyasını okuyarak, wav dosyasını 256 boyutlu HuBERT özelliklerine dönüştürür ve npy formatında `/logs/your-experiment-name/3_feature256` dizinine kaydeder.

## Adım 3
Modeli eğit.
### Başlangıç Seviyesi Sözlüğü
Derin öğrenmede, veri kümesi bölmeye ve öğrenmeye adım adım devam eder. Bir model güncellemesinde (adım), batch_size veri alınır ve tahminler ve hata düzeltmeleri yapılır. Bunun bir defa bir veri kümesi için yapılması bir dönem olarak sayılır.

Bu nedenle, öğrenme zamanı adım başına öğrenme zamanı x (veri kümesindeki veri sayısı / batch boyutu) x dönem sayısıdır. Genel olarak, batch boyutu ne kadar büyükse, öğrenme daha istikrarlı hale gelir (adım başına öğrenme süresi ÷ batch boyutu) küçülür, ancak daha fazla GPU belleği kullanır. GPU RAM'ı nvidia-smi komutu ile kontrol edilebilir. Çalışma ortamının makinesine göre batch boyutunu mümkün olduğunca artırarak öğrenme süresini kısa sürede yapabilirsiniz.

### Önceden Eğitilmiş Modeli Belirtme
RVC, modeli 0'dan değil önceden eğitilmiş ağırlıklardan başlatarak eğitir, bu nedenle küçük bir veri kümesi ile eğitilebilir.

Varsayılan olarak

- Eğer pitch'i dikkate alıyorsanız, `rvc-location/pretrained/f0G40k.pth` ve `rvc-location/pretrained/f0D40k.pth` yüklenir.
- Eğer pitch'i dikkate almıyorsanız, yine `rvc-location/pretrained/f0G40k.pth` ve `rvc-location/pretrained/f0D40k.pth` yüklenir.

Öğrenirken model parametreleri her save_every_epoch için `logs/your-experiment-name/G_{}.pth` ve `logs/your-experiment-name/D_{}.pth` olarak kaydedilir, ancak bu yolu belirterek öğrenmeye başlayabilirsiniz. Farklı bir deneyde öğrenilen model ağırlıklarından öğrenmeye yeniden başlayabilir veya eğitimi başlatabilirsiniz.

### Öğrenme İndeksi
RVC, eğitim sırasında kullanılan HuBERT özellik değerlerini kaydeder ve çıkarım sırasında, öğrenme sırasında kullanılan özellik değerlerine benzer özellik değerlerini arayarak çıkarım yapar. Bu aramayı yüksek hızda gerçekleştirebilmek için indeks öğrenilir.
İndeks öğrenimi için yaklaş

ık komşuluk arama kütüphanesi faiss kullanılır. `/logs/your-experiment-name/3_feature256`'daki özellik değerini okur ve indeksi öğrenmek için kullanır, `logs/your-experiment-name/add_XXX.index` olarak kaydedilir.

(20230428 güncelleme sürümünden itibaren indeks okunur ve kaydetmek/belirtmek artık gerekli değildir.)

### Düğme Açıklaması
- Modeli Eğit: Adım 2b'yi çalıştırdıktan sonra, modeli eğitmek için bu düğmeye basın.
- Özellik İndeksini Eğit: Modeli eğittikten sonra, indeks öğrenme işlemi yapın.
- Tek Tıklamayla Eğitim: Adım 2b, model eğitimi ve özellik indeks eğitimini bir arada yapar.
