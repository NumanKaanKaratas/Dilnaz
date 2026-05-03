# Dilnaz

**Dil:** [English](README.md) | Türkçe

Dilnaz, bir sonraki yazı parçasını doğrudan token olarak tahmin etmek yerine bir sonraki anlam dağılımını tahmin etmeyi hedefleyen iki aşamalı bir semantic language modeling araştırma projesidir.

Temel fikir şudur: klasik dil modelleri çoğunlukla "sıradaki token ne olmalı?" sorusunu öğrenir. Dilnaz ise önce "sıradaki anlam ne olmalı?" sorusunu öğrenir, sonra bu anlamı yazıya döker. Bu ayrım özellikle çok dilli genelleme, yüzey biçimi değişiklikleri, eş anlamlı kelimeler ve yazım farklılıkları için daha esnek bir temsil alanı oluşturmayı amaçlar.

## Amaç

Dilnaz'ın uzun vadeli hedefi, metni yalnızca karakter veya token dizisi olarak değil, anlam akışı olarak modellemektir.

Örneğin `araba`, `otomobil`, `car` ve başka dillerdeki yakın karşılıklar yüzeyde farklıdır; ancak aynı veya çok yakın bir kavrama işaret ederler. Dilnaz bu tür parçaları mümkün olduğunca aynı semantic uzayın yakın bölgelerine yerleştirmeyi, ikinci aşamada ise bu semantic uzayda bir sonraki anlamı tahmin etmeyi hedefler.

Bu mimaride yazım biçimi tamamen yok sayılmaz. Dil modeli geçmişteki dil, biçim ve bağlam sinyallerini kullanarak hangi anlamın hangi yüzey biçimiyle yazılması gerektiğini de öğrenir. Yani önce anlam tahmin edilir, sonra bu anlam bulunduğu bağlama uygun yüzey biçimine çevrilir.

## Mimari

Dilnaz iki ana modelden oluşur:

```text
surface text
  -> HybridTokenizer
  -> DIL: surface/context -> semantic distribution
  -> NAZ: semantic sequence -> next semantic distribution
  -> DIL renderer: semantic distribution -> surface text
```

### DIL

`DIL`, surface ile semantic uzay arasındaki çift yönlü köprüdür.

Görevleri:

- hybrid tokenizer çıktısını ve sol bağlamı okuyarak semantic latent dağılım üretmek
- her parça için `mean + log_std` dağılımı oluşturmak
- latent dağılımdan tekrar byte/surface düzeyinde yazı üretmek
- NLLB teacher'dan gelen çok dilli semantic geometriye yaklaşmak

DIL bir VAE benzeri akış kullanır:

```text
surface/context -> encoder -> mean, log_std -> sampled latent -> renderer -> surface
```

Eğitim kayıpları:

- reconstruction cross entropy
- length loss
- KL loss
- NLLB grouped layer geometry loss
- mean geometry loss
- variance regularizer

DIL'in amacı yalnızca ezberci bir autoencoder olmak değildir. Reconstruction yüzey biçimini korur; NLLB distillation ise latent uzayın semantic olarak anlamlı kalmasını sağlar.

### NAZ

`NAZ`, DIL'in ürettiği semantic dağılımlar üzerinde çalışan ikinci aşama modeldir.

Görevi:

```text
meaning_1, meaning_2, meaning_3 -> meaning_4
```

Yani NAZ'ın hedefi token id değildir. Hedef, frozen DIL encoder tarafından üretilen bir sonraki parçanın `target_mean + target_log_std` dağılımıdır.

NAZ generation sırasında yazıyı tekrar encode ederek döngüye sokmaz. Prompt yalnızca başlangıçta DIL encoder'dan geçirilir. Sonrasında NAZ kendi ürettiği semantic dağılımı tekrar input olarak kullanır:

```text
prompt surface -> DIL encoder once -> initial semantic states
NAZ -> next_mean, next_log_std
NAZ -> next_mean, next_log_std
...
generated means -> DIL renderer -> text
```

Bu semantic-loop tasarımının amacı, generation sırasında yüzey yazım hatalarının tekrar semantic input'a taşınmasını engellemek ve uzun üretimlerde modeli anlam akışı üzerinde tutmaktır.

## NAZ Semantic Backbone

NAZ backbone tamamen Dilnaz'a ait native bir semantic backbone'dur. Dış bir transformer model wrapper'ı kullanılmaz.

Blok düzeni:

```text
L0  SemanticDeltaMixer
L1  SemanticDeltaMixer
L2  SemanticDeltaMixer
L3  SemanticGlobalAttention
tekrar...
```

Bu yapı iki farklı ihtiyacı birleştirir:

- `SemanticDeltaMixer`: uzun semantic akışı ucuz ve recurrent state ile taşımak
- `SemanticGlobalAttention`: belirli aralıklarla tüm bağlama doğrudan bakmak

Backbone bileşenleri:

- `ZeroCenteredRMSNorm`
- `PartialRotaryEmbedding`
- `SemanticDeltaMixer`
- `SemanticGlobalAttention`
- `GatedFeedForward`
- `NazBackboneCache`

Bu tasarım token vocabulary veya LM head üzerine kurulmaz. NAZ'ın giriş ve çıkış dili semantic dağılımlardır.

## Tokenizer

Dilnaz, hybrid surface tokenizer kullanır.

Amaç, hem byte düzeyinde güvenli kapsama alanı sağlamak hem de sık görülen yüzey parçalarını daha verimli temsil etmektir.

Özellikler:

- byte fallback ile OOV riskini azaltır
- surface vocabulary ile sık parçaları kompakt tutar
- her segment `max_word_bytes` genişliğinde modele verilir
- boşluk ve noktalama ayrımları korunur
- tokenizer vocab dosyası DIL checkpoint içine kopyalanır

Varsayılan vocab kaynağı:

```text
dilnaz/tokenization/hybrid_surface_vocab.json
```

Bu tokenizer'ın görevi semantic hedefi belirlemek değildir. Tokenizer yalnızca yüzey metni modelin okuyabileceği parçalara böler. Semantic hizalama DIL encoder ve NLLB distillation ile öğrenilir.

## NLLB Neden Kullanılıyor?

NLLB, çok dilli semantic yakınlık için teacher olarak kullanılır.

Dilnaz'ın hedefi yalnızca Türkçe token dizilerini ezberlemek değildir. Aynı veya yakın anlamların farklı dillerde ve farklı yazım biçimlerinde semantic uzayda yakın durması istenir. NLLB encoder temsilleri bu amaç için güçlü bir başlangıç öğretmeni sağlar.

DIL eğitiminde NLLB doğrudan decoder olarak kullanılmaz. NLLB'den alınan encoder layer temsilleri grouped geometry loss ile DIL latent uzayına aktarılır.

Kullanılan teacher fikri:

```text
NLLB hidden layers -> grouped semantic geometry -> DIL layer vectors + mean
```

Bu sayede DIL, yalnızca hedef kelimeyi geri yazmayı değil, kelimeler ve bağlam parçaları arasındaki semantic ilişki geometrisini de öğrenir.

## Diğer Yaklaşımlardan Farkı

Dilnaz'ın ana farkı, autoregressive hedefin discrete token değil continuous semantic distribution olmasıdır.

Klasik akış:

```text
past tokens -> next token id
```

Dilnaz akışı:

```text
past meanings -> next meaning distribution -> renderer -> surface text
```

Bu ayrım birkaç önemli sonuç doğurur:

- aynı anlama gelen farklı yüzey biçimleri semantic olarak yakın temsil edilebilir
- generation akışı yazım biçiminden önce anlam planına odaklanır
- decoder/render aşaması semantic üretimden ayrıdır
- ikinci model yüzey token ezberi yerine semantic geçişleri öğrenir
- ileride çok dilli eğitimde diller arası semantic aktarım daha doğal hale gelebilir

Bu proje, tokenizer-free veya yalnızca byte-level bir sistem değildir. Yüzey bilgisi korunur; ancak nihai autoregressive hedef anlam dağılımıdır.

## Eğitim Akışı

Önerilen sıra:

1. DIL eğitilir.
2. DIL frozen tutulur.
3. NAZ, frozen DIL'in ürettiği `mean/log_std` hedefleriyle eğitilir.
4. Interface sırasında prompt DIL ile encode edilir, NAZ semantic-loop üretir, DIL renderer final text üretir.

### DIL Eğitimi

```powershell
cd D:\Projects\Dilnaz\dilnaz\train

python train_dil.py `
  --train-file ../../TrainDatas/Test1.txt `
  --output-dir ../../checkpoints/Dil `
  --max-steps 50000 `
  --batch-size 1024 `
  --log-every 50 `
  --checkpoint-every 5000 `
  --data-mode resident
```

### NAZ Eğitimi

```powershell
cd D:\Projects\Dilnaz\dilnaz\train

python train_naz.py `
  --train-file ../../TrainDatas/Test1.txt `
  --dil-checkpoint-dir ../../checkpoints/Dil `
  --output-dir ../../checkpoints/Naz `
  --max-steps 30000 `
  --batch-size 8 `
  --sequence-length 256 `
  --log-every 50 `
  --data-mode resident
```

`resident` mode küçük ve orta ölçekli deneylerde hızlıdır. Büyük veride `streaming` mode kullanılabilir. Streaming modunda DIL encoder her batch için çalışır; resident modda frozen DIL semantic dağılımları başta cache'lenir.

## Interface

### DIL İnceleme

```powershell
cd D:\Projects\Dilnaz\dilnaz\train

python interface_dil.py `
  --checkpoint-dir ../../checkpoints/Dil `
  --text "Yahudi toplulukları ile olan irtibatlarının oldukça azalması."
```

Bu arayüz DIL'in reconstruction kalitesini, similarity matrix çıktısını ve latent swap davranışını kontrol etmek için kullanılır.

### NAZ Üretim

```powershell
cd D:\Projects\Dilnaz\dilnaz\train

python interface_naz.py `
  --checkpoint-dir ../../checkpoints/Naz `
  --max-new-tokens 512 `
  --num-samples 8 `
  --text "Yahudiler, dünyanın dört bir tarafına dağılmış topluluklardan"
```

NAZ interface semantic-loop kullanır. Generated tokenlar tekrar DIL encoder'a sokulmaz; DIL encoder yalnızca prompt için çalışır.

## Compile ve Performans

Compile sistemi full model compile etmez. Yalnızca saf tensor core'lar compile edilir:

- `DilEncoderCore`
- `DilDecoderRenderer`
- `NazStudentCore`

Bu yaklaşım checkpoint, tokenizer, random sampling, cache objeleri ve loss bookkeeping gibi Python ağırlıklı parçaları compile grafiğinin dışında tutar.

Varsayılan CUDA compile modu:

```text
reduce-overhead
```

CPU tarafında compile varsayılan olarak kapalıdır.

## Checkpoint Kontratı

Dilnaz geriye dönük checkpoint uyumluluğu taşımaz. Mimari kontrat değiştiğinde checkpoint formatı kırılır ve modeller sıfırdan eğitilir.

Güncel kontrat:

```text
DIL format_version = 7
NAZ format_version = 10
```

Bu tercih bilinçlidir. Proje araştırma aşamasında olduğu için eski mimari dallarını canlı tutmak yerine aktif mimariyi temiz ve doğrudan tutmak önceliklidir.

## Proje Yapısı

```text
dilnaz/
  models/
    configuration_dil.py
    configuration_naz.py
    modeling_dil.py
    modeling_naz.py
    naz_backbone/
  tokenization/
    hybrid_surface_vocab.json
  train/
    train_dil.py
    train_naz.py
    interface_dil.py
    interface_naz.py
    dil_data.py
    naz_data.py
    byte_trainer_utils.py
```

## Geliştirme İlkeleri

- semantic hedef token hedefinden ayrıdır
- DIL ve NAZ görevleri ayrı tutulur
- DIL semantic uzayı NLLB teacher geometry ile şekillenir
- NAZ yalnızca frozen DIL semantic dağılımlarını öğrenir
- generation sırasında semantic-loop korunur
- eski mimari uyumluluk kodu taşınmaz
- dış backbone wrapper kullanılmaz

## Yol Haritası

Kısa vadeli hedefler:

- DIL reconstruction kalitesini daha büyük ve çeşitli veriyle güçlendirmek
- NAZ semantic-loop repetition davranışını ölçmek
- uzun context eğitimlerini daha büyük corpus üzerinde test etmek
- multilingual prompt ve continuation testleri yapmak

Orta vadeli hedefler:

- daha güçlü semantic memory mekanizmaları eklemek
- uzun contextte topic/anchor takibini geliştirmek
- renderer hızını ve batch decode verimini artırmak
- semantic uzayda daha kontrollü sampling stratejileri denemek

Uzun vadeli hedef:

```text
surface language modeling -> semantic language modeling
```

Dilnaz'ın nihai amacı, modelin önce ne söylemek istediğinin anlamını öğrenmesi, ardından bu anlamı bağlama ve dile uygun şekilde yazıya dökmesidir.
