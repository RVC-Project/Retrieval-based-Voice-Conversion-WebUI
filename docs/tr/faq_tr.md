## Q1: FFmpeg Hatası/UTF8 Hatası
Büyük olasılıkla bu bir FFmpeg sorunu değil, daha çok ses dosyası yolunda bir sorun;

FFmpeg, boşluklar ve () gibi özel karakterler içeren yolları okurken bir hata ile karşılaşabilir; ve eğitim setinin ses dosyaları Çin karakterleri içeriyorsa, bunlar filelist.txt'ye yazıldığında utf8 hatasına neden olabilir.<br>

## Q2: "Tek Tıklamayla Eğitim" Sonrası İndeks Dosyası Bulunamıyor
Eğer "Eğitim tamamlandı. Program kapatıldı." mesajını görüyorsa, model başarıyla eğitilmiş demektir ve sonraki hatalar sahte;

"Added" dizini oluşturulduğu halde "Tek Tıklamayla Eğitim" sonrası indeks dosyası bulunamıyorsa, bu genellikle eğitim setinin çok büyük olmasından kaynaklanabilir ve indeksin eklenmesi sıkışabilir. Bu sorun indeks eklerken bellek yükünü azaltmak için toplu işlem yaparak çözülmüştür. Geçici bir çözüm olarak, "Eğitim İndeksini Eğit" düğmesine tekrar tıklamayı deneyin.<br>

## Q3: Eğitim Sonrası "Tonlama İnceleniyor" Bölümünde Model Bulunamıyor
"Lanetleme İstemi Listesini Yenile" düğmesine tıklayarak tekrar kontrol edin; hala görünmüyorsa, eğitim sırasında herhangi bir hata olup olmadığını kontrol edin ve geliştiricilere daha fazla analiz için konsol, web arayüzü ve logs/experiment_name/*.log ekran görüntülerini gönderin.<br>

## Q4: Bir Model Nasıl Paylaşılır/Başkalarının Modelleri Nasıl Kullanılır?
rvc_root/logs/experiment_name dizininde saklanan pth dosyaları paylaşım veya çıkarım için değildir, bunlar deney checkpoint'larıdır ve çoğaltılabilirlik ve daha fazla eğitim için saklanır. Paylaşılacak olan model, weights klasöründeki 60+MB'lık pth dosyası olmalıdır;

Gelecekte, weights/exp_name.pth ve logs/exp_name/added_xxx.index birleştirilerek tek bir weights/exp_name.zip dosyasına dönüştürülecek ve manuel indeks girişi gereksinimini ortadan kaldıracaktır; bu nedenle pth dosyasını değil, farklı bir makinede eğitime devam etmek istemezseniz zip dosyasını paylaşın;

Çıkarılmış modelleri zorlama çıkarım için logs klasöründen weights klasörüne birkaç yüz MB'lık pth dosyalarını kopyalamak/paylaşmak, eksik f0, tgt_sr veya diğer anahtarlar gibi hatalara neden olabilir. Smaller modeli manuel veya otomatik olarak çıkarmak için alttaki ckpt sekmesini kullanmanız gerekmektedir (eğer bilgi logs/exp_name içinde bulunuyorsa), pitch bilgisini ve hedef ses örnekleme oranı seçeneklerini seçmeli ve ardından daha küçük modele çıkarmalısınız. Çıkardıktan sonra weights klasöründe 60+ MB'lık bir pth dosyası olacaktır ve sesleri yeniden güncelleyebilirsiniz.<br>

## Q5: Bağlantı Hatası
Büyük ihtimalle konsolu (siyah komut satırı penceresi) kapatmış olabilirsiniz.<br>

## Q6: Web Arayüzünde 'Beklenen Değer: Satır 1 Sütun 1 (Karakter 0)' Hatası
Lütfen sistem LAN proxy/global proxy'sini devre dışı bırakın ve ardından sayfayı yenileyin.<br>

## Q7: WebUI Olmadan Nasıl Eğitim Yapılır ve Tahmin Yapılır?
Eğitim komut dosyası:<br>
Önce WebUI'de eğitimi çalıştırabilirsiniz, ardından veri seti önişleme ve eğitiminin komut satırı sürümleri mesaj penceresinde görüntülenecektir.<br>

Tahmin komut dosyası:<br>
https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/myinfer.py<br>


örn:<br>

runtime\python.exe myinfer.py 0 "E:\codes\py39\RVC-beta\todo-songs\1111.wav" "E:\codes\py39\logs\mi-test\added_IVF677_Flat_nprobe_7.index" harvest "test.wav" "weights/mi-test.pth" 0.6 cuda:0 True<br>


f0up_key=sys.argv[1]<br>
input_path=sys.argv[2]<br>
index_path=sys.argv[3]<br>
f0method=sys.argv[4]#harvest or pm<br>
opt_path=sys.argv[5]<br>
model_path=sys.argv[6]<br>
index_rate=float(sys.argv[7])<br>
device=sys.argv[8]<br>
is_half=bool(sys.argv[9])<br>

## Q8: Cuda Hatası/Cuda Bellek Yetersizliği
Küçük bir ihtimalle CUDA konfigürasyonunda bir problem olabilir veya cihaz desteklenmiyor olabilir; daha muhtemel olarak yetersiz bellek olabilir (bellek yetersizliği).<br>

Eğitim için toplu işlem boyutunu azaltın (1'e indirgemek yeterli değilse, grafik kartını değiştirmeniz gerekebilir); çıkarım için ise config.py dosyasındaki x_pad, x_query, x_center ve x_max ayarlarını ihtiyaca göre düzenleyin. 4GB veya daha düşük bellekli kartlar (örneğin 1060(3G) ve çeşit

li 2GB kartlar) terk edilebilir, 4GB bellekli kartlar hala bir şansı vardır.<br>

## Q9: Optimal Olarak Kaç total_epoch Gerekli?
Eğitim veri setinin ses kalitesi düşük ve gürültü seviyesi yüksekse, 20-30 dönem yeterlidir. Fazla yüksek bir değer belirlemek, düşük kaliteli eğitim setinizin ses kalitesini artırmaz.<br>

Eğitim setinin ses kalitesi yüksek, gürültü seviyesi düşük ve yeterli süre varsa, bu değeri artırabilirsiniz. 200 kabul edilebilir bir değerdir (çünkü eğitim hızlıdır ve yüksek kaliteli bir eğitim seti hazırlayabiliyorsanız, GPU'nuz muhtemelen uzun bir eğitim süresini sorunsuz bir şekilde yönetebilir).<br>

## Q10: Kaç Dakika Eğitim Verisi Süresi Gerekli?

10 ila 50 dakika arası bir veri seti önerilir.<br>

Garantili yüksek ses kalitesi ve düşük arka plan gürültüsü varsa, veri setinin tonlaması homojen ise daha fazlası eklenebilir.<br>

Yüksek seviyede bir eğitim seti (zarif ve belirgin tonlama), 5 ila 10 dakika arası uygundur.<br>

1 ila 2 dakika veri ile başarılı bir şekilde eğitim yapan bazı insanlar olsa da, başarı diğerleri tarafından tekrarlanabilir değil ve çok bilgilendirici değil. Bu, eğitim setinin çok belirgin bir tonlamaya sahip olmasını (örneğin yüksek frekansta havadar bir anime kız sesi gibi) ve ses kalitesinin yüksek olmasını gerektirir; 1 dakikadan daha kısa süreli veri denenmemiştir ve önerilmez.<br>


## Q11: İndeks Oranı Nedir ve Nasıl Ayarlanır?
Eğer önceden eğitilmiş model ve tahmin kaynağının ton kalitesi, eğitim setinden daha yüksekse, tahmin sonucunun ton kalitesini yükseltebilirler, ancak altta yatan modelin/tahmin kaynağının tonu yerine eğitim setinin tonuna yönelik olası bir ton önyargısıyla sonuçlanır, bu genellikle "ton sızıntısı" olarak adlandırılır.<br>

İndeks oranı, ton sızıntı sorununu azaltmak/çözmek için kullanılır. İndeks oranı 1 olarak ayarlandığında, teorik olarak tahmin kaynağından ton sızıntısı olmaz ve ton kalitesi daha çok eğitim setine yönelik olur. Eğer eğitim seti, tahmin kaynağından daha düşük ses kalitesine sahipse, daha yüksek bir indeks oranı ses kalitesini azaltabilir. Oranı 0'a düşürmek, eğitim seti tonlarını korumak için getirme karıştırmasını kullanmanın etkisine sahip değildir.<br>

Eğer eğitim seti iyi ses kalitesine ve uzun süreye sahipse, total_epoch'u artırın. Model, tahmin kaynağına ve önceden eğitilmiş alt modeline daha az başvurduğunda ve "ton sızıntısı" daha az olduğunda, indeks oranı önemli değil ve hatta indeks dosyası oluşturmak/paylaşmak gerekli değildir.<br>

## Q12: Tahmin Yaparken Hangi GPU'yu Seçmeli?
config.py dosyasında "device cuda:" ardından kart numarasını seçin.<br>

Kart numarası ile grafik kartı arasındaki eşleme, eğitim sekmesinin grafik kartı bilgileri bölümünde görülebilir.<br>

## Q13: Eğitimin Ortasında Kaydedilen Model Nasıl Kullanılır?
Kaydetme işlemini ckpt işleme sekmesinin altında yer alan model çıkarımı ile yapabilirsiniz.

## Q14: Dosya/Bellek Hatası (Eğitim Sırasında)?
Çok fazla işlem ve yetersiz bellek olabilir. Bu sorunu düzeltebilirsiniz:

1. "CPU İş Parçacıkları" alanındaki girişi azaltarak.

2. Eğitim verisini daha kısa ses dosyalarına önceden keserek.

## Q15: Daha Fazla Veri Kullanarak Eğitime Nasıl Devam Edilir?

Adım 1: Tüm wav verilerini path2 dizinine yerleştirin.

Adım 2: exp_name2+path2 -> veri setini önişleme ve özellik çıkarma.

Adım 3: exp_name1 (önceki deneyinizin) en son G ve D dosyalarını exp_name2 klasörüne kopyalayın.

Adım 4: "modeli eğit" düğmesine tıklayın ve önceki deneyinizin model döneminden başlayarak eğitime devam edecektir.
