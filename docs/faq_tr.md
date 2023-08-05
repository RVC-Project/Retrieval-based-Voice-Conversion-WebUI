## Soru 1: FFmpeg hatası/utf8 hatası.
Muhtemelen bir FFmpeg sorunu değil, ses yolunda bir sorun var;

FFmpeg, boşluklar ve () gibi özel karakterler içeren yolları okurken bir hata ile karşılaşabilir ve FFmpeg hatası oluşturabilir; ve eğitim setinin sesleri Çince yollar içeriyorsa, bunları filelist.txt'ye yazmak utf8 hatasına neden olabilir.

## Soru 2: "Tek Tıklamayla Eğitim" sonrasında indeks dosyası bulunamıyor.
"Training is done. The program is closed" şeklinde görüntüleniyorsa, model başarılı bir şekilde eğitilmiş demektir ve sonraki hatalar yanıltıcı olabilir;

Tek tıklamalı eğitim sonrasında "added" indeks dosyasının eksik olması, eğitim setinin çok büyük olmasından kaynaklanabilir ve indeksin eklenmesinin takılmasına neden olabilir; bunun çözümü, indeksi eklerken bellek aşımı sorununu çözen toplu işlemi kullanmaktır. Geçici bir çözüm olarak, "Train Index" düğmesine tekrar tıklamayı deneyin.

## Soru 3: Eğitim sonrasında "Timbre Inferencing" bölümünde model bulunamıyor
"Refresh timbre list"e tıklayın ve tekrar kontrol edin; hala görünmüyorsa, eğitim sırasında hatalar olup olmadığını kontrol edin ve geliştiricilere ek analiz için konsol, web UI ve logs/experiment_name/*.log ekran görüntüleri gönderin.

## Soru 4: Bir modeli nasıl paylaşabilirim/Başkalarının modellerini nasıl kullanabilirim?
rvc_root/logs/experiment_name klasöründe depolanan pth dosyaları, paylaşım veya çıkarım için değil, yeniden üretilebilirlik ve daha fazla eğitim için deney kontrol noktalarını depolamak içindir. Paylaşılacak model, weights klasöründeki 60+MB pth dosyası olmalıdır;

Gelecekte, weights/exp_name.pth ve logs/exp_name/added_xxx.index birleştirilerek, manuel indeks girişi gerektirmeyen bir tek weights/exp_name.zip dosyası oluşturulacak; bu nedenle, farklı bir makinede eğitime devam etmek istemiyorsanız, pth dosyasını değil zip dosyasını paylaşın;

Logs klasöründen weights klasörüne birkaç yüz MB'lık pth dosyalarını zorlama çıkarım için kopyalamak/paylaşmak, eksik f0, tgt_sr veya diğer anahtarlar gibi hatalara neden olabilir. Alt kısımdaki ckpt sekmesini kullanarak manuel veya otomatik olarak (bilgiler logs/exp_name'de bulunuyorsa) ton infomasyonu ve hedef ses örnekleme hızı seçmeyi deneyin ve ardından daha küçük modeli çıkarın. Çıkarıldıktan sonra weights klasöründe 60+ MB'lık bir pth dosyası olacak ve sesleri yenileyerek kullanabilirsiniz.

## Soru 5: Bağlantı Hatası.
Muhtemelen konsolu (siyah komut satırı penceresini) kapattınız.

## Soru 6: WebUI'de 'Expecting value: line 1 column 1 (char 0)' hatası.
Sistem LAN proxy/global proxy'yi devre dışı bırakın ve sonra yenileyin.

## Soru 7: WebUI olmadan nasıl eğitilir ve sonuçlandırılır?
Eğitim betiği:
Eğitimi WebUI'de çalıştırabilirsiniz, ve mesaj penceresinde veri seti ön işleme ve eğitiminin komut satırı sürümleri gösterilecektir.

Sonuçlandırma betiği:
https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/myinfer.py

Örneğin:

```bash
runtime\python.exe myinfer.py 0 "E:\codes\py39\RVC-beta\todo-songs\1111.wav" "E:\codes\py39\logs\mi-test\added_IVF677_Flat_nprobe_7.index" harvest "test.wav" "weights/mi-test.pth" 0.6 cuda:0 True
```

f0up_key=sys.argv[1]
input_path=sys.argv[2]
index_path=sys.argv[3]
f0method=sys.argv[4]#harvest or pm
opt_path=sys.argv[5]
model_path=sys.argv[6]
index_rate=float(sys.argv[7])
device=sys.argv[8]
is_half=bool(sys.argv[9])

## Soru 8: Cuda hatası/Cuda bellek dışı.
Küçük bir olasılıkla CUDA yapılandırmasında bir sorun olabilir veya cihaz desteklenmiyor olabilir; daha olası bir şekilde, yeterli belleğiniz yoktur (bellek dışı).

Eğitim için, toplu boyutunu azaltın (1'e düşürmek hala yeterli değilse, grafik kartını değiştirmeniz gerekebilir); sonuçlandırma için, config.py dosyasında x_pad, x_query, x_center ve x_max ayarlarını ihtiyaca göre ayarlayın. 4G veya daha düşük bellekli kartlar (örn. 1060(3G) ve çeşitli 2G kartlar) terk edilebilir, ancak 4G bellekli kartların hala bir şansı vardır.

## Soru 9: Optimal kaç total_epoch kullanmalıyım?
Eğitim veri setinin ses kalitesi düşük ve gürültü seviyesi yüksekse, 20-30 epoch yeterlidir. Çok yüksek bir değer ayarlamak, düşük kaliteli eğitim setinizin ses kalitesini artırmaz.

Eğitim setinin ses kalitesi yüksek, gürültü sevi

yesi düşük ve yeterli süresi varsa, artırabilirsiniz. 200 kabul edilebilir (çünkü eğitim hızlıdır ve yüksek kaliteli bir eğitim seti hazırlayabiliyorsanız, GPU'nuz muhtemelen sorunsuz bir şekilde daha uzun bir eğitim süresini işleyebilir).

## Soru 10: Ne kadar eğitim verisi süresine ihtiyacım var?
Yaklaşık 10 dakika ile 50 dakika arasında bir veri seti önerilir.

Sağlam ses kalitesi ve düşük taban gürültü garantiliyse, veri seti seslerinin homojen olması durumunda daha fazla ekleyebilirsiniz.

Yüksek seviye bir eğitim seti için (düzgün + belirgin bir ton), 5 dakika ile 10 dakika arasında yeterlidir.

1 dakika ile 2 dakika veriyle başarıyla eğitim yapan bazı insanlar var, ancak başarı başkaları tarafından tekrarlanabilir değil ve çok bilgi verici değil. Bu, eğitim setinin çok belirgin bir tona sahip olmasını (örneğin yüksek frekanslı havadar anime kız sesi gibi) ve ses kalitesinin yüksek olmasını gerektirir; 1 dakikadan daha kısa veriler şu ana kadar başarılı bir şekilde deneme yapılmamıştır. Bu önerilmez.

## Soru 11: İndeks oranı nedir ve nasıl ayarlanır?
Önceden eğitilmiş modelin ve çıkarım kaynağının ton kalitesi, eğitim setinin ton kalitesinden daha yüksekse, bunlar çıkarım sonucunun ton kalitesini artırabilir, ancak eğitim setinin tonuna göre değil, genellikle "ton sızıntısı" olarak adlandırılan eğitim setinin tonuna göre bir ton eğilimine yol açabilir.

İndeks oranı, ton sızıntı sorununu azaltmak/çözmek için kullanılır. İndeks oranı 1 olarak ayarlandığında, teorik olarak çıkarım kaynağından hiç ton sızıntısı olmaz ve ton kalitesi daha çok eğitim setine yönlendirilir. Eğitim seti, çıkarım kaynağından ses kalitesi açısından daha düşükse, daha yüksek bir indeks oranı ses kalitesini azaltabilir. 0'a indirildiğinde, eğitim seti tonlarını korumak için çıkarım karışımı kullanma etkisi yoktur.

Eğitim seti iyi ses kalitesine sahipse ve uzun süreliyse, total_epoch'ı artırın, modelin kendi başına çıkarım kaynağına ve önceden eğitilmiş temel modeline başvurma olasılığı azaldığında ve "ton sızıntısı" çok az olduğunda, indeks oranı önemli değildir ve hatta indeks dosyası oluşturmak/paylaşmak zorunda kalmazsınız.

## Soru 12: Çıkarırken hangi gpu'yu seçmeliyim?
config.py dosyasında, "device cuda:" dan sonra kart numarasını seçin.

Kart numarası ile grafik kartı arasındaki eşleştirmeyi eğitim sekmesinin grafik kartı bilgisi bölümünde görebilirsiniz.

## Soru 13: Eğitimin ortasında kaydedilen modeli nasıl kullanabilirim?
Çıkartma modeli, ckpt processing sekmesinin alt kısmında kaydedin.

## Soru 14: Dosya/bellek hatası (eğitim sırasında)?
Çok fazla işlem ve belleğiniz yeterli değil. Bunun düzeltilmesi için:

1. "Threads of CPU" alanında girişi azaltın.
2. Eğitim setini daha kısa ses dosyalarına önceden kesin.

## Soru 15: Daha fazla veri kullanarak nasıl eğitime devam ederim?
Adım 1: Tüm wav verilerini path2'ye koyun.
Adım 2: exp_name2+path2 -> veri kümesini işleyin ve özellik çıkarın.
Adım 3: exp_name1 (önceki deneyiminiz) en son G ve D dosyalarını exp_name2 klasörüne kopyalayın.
Adım 4: "train the model" düğmesine tıklayın ve önceki deneyiminiz model epoğunun başlangıcından itibaren eğitime devam edecektir.