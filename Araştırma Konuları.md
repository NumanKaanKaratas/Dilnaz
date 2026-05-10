# 🔍 RAG Sistemleri & AI Mimarisi Yenilikleri — Araştırma Raporu
**Tarih:** 10 Mayıs 2026 | **Kaynak sayısı:** 20+ | **Derinlik:** Derin

---

## 📋 Yönetici Özeti

Bu rapor iki temel soruyu yanıtlıyor:
1. **En iyi RAG sistemi nedir?** — 2026 itibarıyla "en iyi" diye tek bir sistem yok; kullanım senaryosuna göre farklı liderler var. Açık kaynak tarafında **LightRAG** ve **RAGFlow** öne çıkıyor; kurumsal tarafta **LangGraph** + **Pinecone** kombinasyonu. En güncel trend: **Adaptive RAG** (sorgu karmaşıklığına göre yönlendirme).
2. **RAG eğitim sürecinde kullanılıyor mu?** — Evet, ama çok az şirket bunu yapıyor. **REALM**, **RETRO**, **Atlas** gibi modeller retrieval'ı doğrudan *eğitim* sırasına entegre etmiştir.
3. **MLP & Linear Attention yenilikleri** — MoE'nin dominant hale gelmesi, DeepSeek'in MLA'sı, SSM/Mamba hibrit mimariler ve Log-Linear Attention en öne çıkan yenilikler.

---

## 🗺️ BÖLÜM 1 — EN İYİ RAG SİSTEMLERİ (2025-2026)

### 1.1 RAG Türlerine Göre Sınıflandırma

| Tür | Nasıl Çalışır | Güçlü Olduğu Yer |
|-----|--------------|-----------------|
| **Naive RAG** | Chunk → Embed → Vector Search → Generate | Hızlı, basit soru-cevap |
| **Advanced RAG** | HyDE, reranker, hybrid search ekler | Genel kurumsal kullanım |
| **GraphRAG** | Entity → Knowledge Graph → Traversal | Multi-hop, ilişkisel sorgular |
| **Agentic RAG** | LLM araçları sıralı çağırır | Karmaşık çok adımlı görevler |
| **Adaptive RAG** | Query classifier → en uygun pipeline | 2026'nın best practice'i |

### 1.2 Framework Karşılaştırma Tablosu

| Framework | Tür | Güçlü Yönler | Zayıf Yönler | Puan |
|-----------|-----|-------------|-------------|------|
| **LightRAG** | Graph+Vector | 6100x token verimliliği, ~80ms gecikme, EMNLP 2025 | Büyük corpus'ta index yavaş | 🟢 9.2/10 |
| **RAGFlow** | Agent+RAG | Document parsing (tablo/görsel), Docker kurulumu kolay | GPU gereksinimi | 🟢 9.0/10 |
| **LangGraph** | Agentic | En iyi ajan orkestrasyon, HITL, checkpointing | Karmaşık öğrenme eğrisi | 🟢 8.8/10 |
| **Microsoft GraphRAG** | Graph | Global insight, community detection, LazyGraphRAG (0.1% index maliyeti) | Yüksek index maliyeti | 🟢 8.5/10 |
| **LlamaIndex** | E2E | Multimodal (metin/görsel/ses), geniş ekosistem | Modüler değil | 🟡 8.3/10 |
| **FlashRAG** | Araştırma | 17 SOTA algoritma, 36 benchmark dataset | Prodüksiyon için zayıf | 🟡 7.8/10 |
| **Pinecone** | Vector DB | Sub-100ms, otomatik sharding | Fiyat, vendor lock-in | 🟡 7.5/10 |

### 1.3 Kullanım Senaryosuna Göre Tavsiye

```
Senaryo → Tavsiye

📄 Basit doküman QA          → LightRAG veya LlamaIndex
🕸️ İlişkisel/multi-hop sorgu  → Microsoft GraphRAG veya LightRAG
🤖 Ajan iş akışları           → LangGraph + RAGFlow
🏢 Kurumsal entegrasyon       → LangGraph + Pinecone
🔬 Araştırma/deney            → FlashRAG
📊 Büyük corpus               → RAGFlow (Docker) veya GraphRAG (LazyGraphRAG)
🚀 2026 best practice         → Adaptive RAG (query classifier ile hybrid pipeline)
```

---

## 🎓 BÖLÜM 2 — RAG EĞİTİM SÜRECİNDE KULLANILIYOR MU?

### 2.1 Kısa Cevap: EVET — ama standart değil

Klasik RAG yalnızca *inference* (cevap üretme) sırasında çalışır — model ağırlıkları sabittir. Ancak bir grup araştırma modeli retrieval'ı doğrudan **eğitim döngüsüne** entegre etmiştir:

### 2.2 Eğitim-Sırasında Retrieval Kullanan Modeller

| Model | Kurum | Nasıl Çalışır |
|-------|-------|--------------|
| **REALM (2020)** | Google | MLM pretraining + retriever joint eğitimi. Retriever ve language model birlikte gradient descent ile optimize edilir |
| **RAG (Lewis 2020)** | Meta | Retriever + generator joint training; dokümanlar latent variable olarak modellenir |
| **RETRO (2022)** | DeepMind | Sequence chunk'larını eğitim sırasında en yakın komşularla birleştiren chunked cross-attention |
| **Atlas (2023)** | Meta | Few-shot learning ile retrieval; FiD reader + contriever retriever birlikte fine-tune |
| **RAFT (2024)** | Microsoft | Retrieval-Augmented Fine-Tuning — model hem ilgili hem ilgisiz dokümanlarla eğitilerek "dikkat dağıtıcıya" dayanıklı hale getirilir |

### 2.3 Neden Yaygınlaşmadı?

- **Maliyet**: Her eğitim adımında retrieval index sorgulama astronomik GPU/disk I/O yükü oluşturur
- **Teknik zorluk**: Frozen index vs. güncellenen index kararı; gradient'in retriever'a akması zor
- **Alternatif**: Modern büyük modeller (Llama 4, Qwen3, DeepSeek-V3) zaten trilyonlarca token ile ön-eğitim aldığından parametrik bellek yeterince güçlü

### 2.4 2025-2026'daki Yeni Yön: ExpRAG & Experience Retrieval

Yeni bir trend olarak ajan sistemleri kendi **geçmiş deneyimlerini** bir corpus olarak depolayıp, sonraki görevlerde bu deneyimleri retrieve edebiliyor. Bu "Retrieval-Augmented Agents" ya da ExpRAG olarak adlandırılıyor ve bir tür training-time retrieval'ın pratik versiyonu sayılabilir.

---

## ⚡ BÖLÜM 3 — MLP & LİNEAR ATTENTİON YENİLİKLERİ (2025-2026)

### 3.1 Attention Kısmındaki Büyük Yenilikler

#### 🔵 Multi-Head Latent Attention (MLA) — DeepSeek
DeepSeek-V2/V3/R1 ile tanıtılan en büyük attention yeniliği:
- Q, K, V matrislerini **low-rank latent vektöre** sıkıştırıyor
- KV cache boyutunu dramatik azaltıyor (LLaMA-2-7B'de **%93 sıkıştırma → 10.6x hız artışı**)
- Decoupled RoPE ile pozisyonel bilgi ayrı tutulur
- Artık Kimi K2, GLM-5, Ling 2.5 gibi modeller de MLA'ya geçiş yapıyor

| Metod | KV Cache | Parametre | Performans |
|-------|---------|-----------|-----------|
| MHA (klasik) | Tam boyut | Yüksek | Referans |
| GQA | ~4-8x küçük | Orta | ~Aynı |
| MQA | ~8x küçük | Düşük | Düşük |
| **MLA** | **~93% küçük** | **Orta** | **MHA seviyesi** |

#### 🟢 Linear Attention & Hybrid Mimariler
Quadratic O(N²) → Linear O(N) complexity geçişi için en öne çıkan yaklaşımlar:

| Model/Yöntem | Kurum | Özellik |
|-------------|-------|---------|
| **Mamba / Mamba-2** | CMU+Princeton | Selective SSM; 5x Transformer'dan hızlı inference |
| **RWKV-7** | RWKV Foundation | RNN tabanlı, linear attention; dil modellemesinde Transformer eşdeğeri |
| **GatedDeltaNet** | Çeşitli | Delta rule + gating; recall'da Mamba'yı geçiyor |
| **Kimi Linear** | Moonshot | Hibrit mimari; expressive + efficient |
| **SSE (Sparse State Expansion)** | ByteDance | Row-sparse update; linear+hybrid için state kapasite sorununun çözümü |
| **Log-Linear Attention** | Arxiv 2025 | Linear ve full softmax arasında orta yol; fixed state boyutu sınırlamasını aşıyor |

#### 🟡 Hybrid Mimari: 2026'nın Sonucu

Saf linear modeller bazı "exact retrieval" görevlerinde hâlâ Transformer'ın gerisinde kalıyor. Pratik çözüm:

```
Hybrid = Linear Attention (büyük çoğunluk) + Softmax Attention (stratejik konumlar)

Önerilen oran (ByteDance araştırması): Linear:Full = 3:1 ile 6:1
```

NVIDIA'nın araştırması da SSM'in büyük bölümde, attention'ın ise "exact copying" gerektiren yerlerde kullanılması gerektiğini gösteriyor.

---

### 3.2 MLP Kısmındaki Büyük Yenilikler

#### 🔴 Mixture of Experts (MoE) — Dominant Trend

2025'te MoE standart hale geldi. Temel fikir: FFN bloklarını sparse activated expert gruplarına bölmek.

| Model | Toplam Param | Aktif Param | Expert Sayısı | Strateji |
|-------|------------|------------|--------------|---------|
| **DeepSeek-V3** | ~670B | 37B | 256 | 9 aktif, küçük expert |
| **Llama 4 Maverick** | ~400B | ~17B | ~128 | 2 aktif, büyük expert |
| **Qwen3-235B** | 235B | 22B | ~128 | MoE + dense hibrit |
| **Jamba** | ~52B | ~12B | 16 | Mamba + MoE birleşimi |

**DeepSeek yaklaşımı**: Çok sayıda küçük expert (256 expert, 9 aktif) → ince uzmanlaşma, daha iyi yük dengesi  
**Meta yaklaşımı**: Az sayıda büyük expert (2 aktif) → basit routing, daha stabil eğitim

#### 🟠 Gizli MoE (Secret MoE) — 2025 Keşfi

Harvard/MIT araştırması dense LLM'lerdeki MLP katmanlarının aslında *gizlice* sparse computation yaptığını gösterdi:
- Nöronlar nadiren ateşleniyor (yüksek sparsity)
- Bu aktivasyon dağılımı Sparse Autoencoder (SAE) yapısına uyuyor
- Yani dense modeller bile MoE gibi davranıyor!

#### 🟣 KAN (Kolmogorov-Arnold Networks) + MLP Hibrit

MLP-KAN: MoE çerçevesinde MLP (representation learning) + KAN (function learning) birleşimi. Araştırma aşamasında ama 4 farklı domain benchmark'ta rekabetçi sonuçlar verdi.

#### 🔵 SwiGLU / GeGLU — Artık Standart

Neredeyse tüm yeni modeller klasik ReLU'yu bırakıp Gated Linear Unit varyantlarına geçti:
```
SwiGLU(x) = Swish(xW₁) ⊗ (xW₂)
```
Bu FFN kapasitesini artırırken parametre sayısını sabit tutuyor.

---

## 📊 BÖLÜM 4 — EN BÜYÜK ŞİRKETLER NE YAPIYOR?

| Şirket | Attention Yeniliği | MLP/FFN Yeniliği | Öne Çıkan Model |
|--------|------------------|-----------------|----------------|
| **DeepSeek** | MLA (KV cache %93 küçültme) | Fine-grained MoE (256 expert) | DeepSeek-V3, R1 |
| **Alibaba** | Hybrid Attention (linear+full) | Ultra-sparse MoE + MTP | Qwen3-Next |
| **Meta** | GQA → Hibrit araştırma | MoE (az büyük expert) | Llama 4 Scout/Maverick |
| **Google DeepMind** | RecurrentGemma/Griffin | Dense + araştırma | Gemma 3 |
| **Moonshot (Kimi)** | MLA + Linear Attention hybrid | MoE | Kimi K2 |
| **ByteDance** | SSE (Sparse State Expansion) | Fine-grained MoE | Seed serisi |
| **AI21 Labs** | Jamba: Mamba + Transformer | MoE | Jamba 2 |
| **Microsoft** | — | GraphRAG + hibrit araştırma | LazyGraphRAG, Azure AI |

---

## 🏆 BÖLÜM 5 — PUANLAMA / ÖNEMİ SIRALAMASI

### RAG Sistemleri (Genel Kullanım)
```
1. 🥇 LightRAG          — 9.2/10  (hız + graph + EMNLP 2025)
2. 🥈 RAGFlow           — 9.0/10  (agent entegrasyonu, belge ayrıştırma)
3. 🥉 LangGraph         — 8.8/10  (kurumsal ajan orkestrasyon)
4.    MS GraphRAG        — 8.5/10  (global insight, LazyGraphRAG)
5.    LlamaIndex         — 8.3/10  (multimodal, geniş ekosistem)
```

### AI Mimari Yenilikleri (Etki Büyüklüğü)
```
1. 🥇 MoE (Mixture of Experts)        — 10/10  (2025'te standart oldu)
2. 🥈 MLA (Multi-Head Latent Attention)— 9.5/10 (inference devrimsel)
3. 🥉 SSM/Mamba + Hybrid              — 9.0/10 (uzun context sorunu çözüyor)
4.    SwiGLU/GeGLU                     — 8.5/10 (artık her modelde var)
5.    Log-Linear Attention             — 8.0/10 (araştırma ama çok umut verici)
6.    KAN+MLP Hybrid                   — 6.5/10 (erken aşama)
```

---

## 💡 TAVSİYELER

**RAG kurmak istiyorsanız:**
- Başlangıç için → **LightRAG** (kolay deploy, güçlü graph-vector hybrid)
- Kurumsal ortam → **LangGraph** + **Pinecone/Milvus** + **RAGFlow**
- İlişkisel sorgular → **Microsoft GraphRAG** (LazyGraphRAG ile index maliyeti %99 düştü)
- 2026 best practice → **Adaptive RAG**: query classifier → Naive/Advanced/Graph/Agentic routing

**AI mimarisi araştırıyorsanız:**
- MoE kaçınılmaz — özellikle **fine-grained MoE** (DeepSeek yaklaşımı) artık norm
- Attention'da **MLA** veya **GQA** olmazsa olmaz (pure MHA artık eski)
- Uzun context → **Hybrid** (Linear:Full = 3:1–6:1) en dengeli seçim
- Eğitim sırasında retrieval istiyorsanız → **RAFT** veya **Atlas** yaklaşımı

---

## 🔗 KAYNAKLAR VE LİNKLER

| Kaynak | Link |
|--------|------|
| LightRAG (EMNLP 2025) | https://github.com/HKUDS/LightRAG |
| Microsoft GraphRAG | https://github.com/microsoft/graphrag |
| RAGFlow | https://github.com/infiniflow/ragflow |
| LangGraph | https://github.com/langchain-ai/langgraph |
| FlashRAG | https://github.com/RUC-NLPIR/FlashRAG |
| Mamba Paper | https://arxiv.org/abs/2312.00752 |
| Mamba-3 (ICLR 2026) | https://openreview.net/pdf?id=HwCvaJOiCj |
| MLA (DeepSeek) | https://arxiv.org/abs/2502.07864 |
| Log-Linear Attention | https://arxiv.org/pdf/2506.04761 |
| SSE (ByteDance) | https://arxiv.org/pdf/2507.16577 |
| MoE Survey | https://arxiv.org/abs/2407.06204 |
| REALM Paper | https://dl.acm.org/doi/abs/10.5555/3524938.3525306 |
| Atlas Paper | https://arxiv.org/abs/2208.03299 |
| RAG Techniques 2026 | https://blog.starmorph.com/blog/rag-techniques-compared-best-practices-guide |

---

# 🔬 Derin Dalış Araştırma Raporu — 5 İleri Seviye Konu
**Tarih:** 10 Mayıs 2026 | **Kaynak sayısı:** 25+ | **Derinlik:** Maksimum

---

## 📋 Kapsam

1. HippoRAG 2 — RAG + Continual Learning
2. Gated DeltaNet — Linear Attention'da Recall Çözümü
3. Speculative Decoding + MoE — Inference Hız Devrimi
4. RAFT — Eğitim Sırasında Retrieval'ın Pratik Hali
5. Qwen3-Next — Hibrit Mimari Üçgeni: Attention + MoE + MTP

---

## 🧠 KONU 1 — HippoRAG 2: RAG'dan Belleğe

**Paper:** "From RAG to Memory: Non-Parametric Continual Learning for LLMs"
**Kurum:** Ohio State University + UIUC
**Yayın:** arXiv Feb 2025 → ICML 2025 (Proceedings, PMLR 267)
**GitHub:** https://github.com/OSU-NLP-Group/HippoRAG

### Problem Neydi?

Standart RAG sistemleri vector retrieval'a dayanıyor — yani embedding benzerliği. Bu:
- **Factual memory** (basit gerçek sorular): İyi çalışıyor ✅
- **Sense-making** (büyük, karmaşık bağlamı anlama): Zayıf ❌
- **Associativity** (birbirine bağlı bilgileri multi-hop ile bulma): Çok zayıf ❌

GraphRAG ve benzeri "structure-augmented" sistemler sense-making'i iyileştiriyor ama factual memory'i bozuyordu — garip bir ödünleşim.

### HippoRAG 2 Nasıl Çalışıyor?

İnsan hippocampusunu (hafıza merkezini) modelliyor. 3 katmanlı mimari:

```
[Dokümanlar]
     ↓
[Knowledge Graph oluşturma — entity + relation extraction]
     ↓
[Derin passage entegrasyonu] ← YENİ: pasajlar da graf düğümü olarak eklendi
     ↓
[Personalized PageRank (PPR) algoritması]
     ↓
[LLM online kullanımı ile yanıt üretimi]
```

**HippoRAG 1'e göre farkları:**
| Bileşen | HippoRAG 1 | HippoRAG 2 |
|---------|-----------|-----------|
| Passage entegrasyonu | Graf kenarlarında | Graf düğümü olarak tam entegrasyon |
| LLM kullanımı | Sadece indexleme | Online (runtime) da LLM kullanımı |
| Factual memory | Standart | Korunuyor + iyileştirildi |

### Sonuçlar

| Görev | Metrik | HippoRAG 2 vs SOTA |
|-------|--------|-------------------|
| Associative memory (multi-hop) | MuSiQue, 2Wiki, HotpotQA | **+7%** SOTA embedding model üzerinde |
| Sense-making | NarrativeQA | GraphRAG/LightRAG'ı geçiyor |
| Factual memory | NaturalQuestions, PopQA | Korunuyor (önceki graph sistemleri burada düşüyordu) |

**Indexleme maliyeti:** GraphRAG, RAPTOR, LightRAG'a kıyasla çok daha az kaynak — çünkü LLM sadece online (query sırasında) kullanılıyor.

### Continual Learning Bağlantısı

Parametrik continual learning (yani ağırlıkları güncelleyerek sürekli öğrenme) LLM'lerde catastrophic forgetting nedeniyle çok zor. HippoRAG 2 **non-parametric** bir çözüm sunuyor: yeni bilgi sadece grafı genişletiyor, model ağırlıkları hiç değişmiyor. Bu saf bir continual learning değil ama pratik ve ölçeklenebilir bir yaklaşım.

### Kullanım Senaryoları

- Zaman içinde büyüyen knowledge base (düzenli yeni doküman ekleme)
- Multi-hop soru-cevap (araştırma asistanları, akademik)
- Uzun hikaye / roman anlama
- Dinamik kurumsal bilgi tabanı

---

## ⚡ KONU 2 — Gated DeltaNet: Linear Attention'da Recall Sorununun Çözümü

**Paper:** "Gated Delta Networks: Improving Mamba2 with Delta Rule"
**Yazarlar:** Songlin Yang, Jan Kautz, Ali Hatamizadeh (NVIDIA)
**arXiv:** 2412.06464 | ICLR 2025
**GitHub:** https://github.com/sustcsonglin/flash-linear-attention

### Sorun: Linear Attention Neden Recall'da Başarısız?

Linear attention, KV cache'i sabit boyutlu bir durum matrisiyle değiştirir:
```
S_t = S_{t-1} + v_t ⊗ k_t    (standart linear attention güncelleme)
o_t = S_t · q_t               (output)
```

Sorun: Sonsuz bilgi sabit boyutlu bir matrise sıkıştırılamaz. Uzun dizilerde eski bilgiler yeni bilgiler tarafından "ezilir" → recall başarısız olur.

### İki Komplementer Mekanizma

**1. Gating (Mamba2'den):** Belleği hızlıca silmek için:
```
S_t = α_t · S_{t-1} + β_t · v_t ⊗ k_t
```
α_t: scalar forget gate — ne kadar unutacağız?
β_t: update gate — ne kadar yeni bilgi yazacağız?

**2. Delta Rule (DeltaNet'ten):** Hassas bellek güncellemesi için:
```
S_t = S_{t-1} + β_t · (v_t - S_{t-1}k_t) ⊗ k_t
```
Bu "hata düzeltme" mantığı: mevcut durum ne kadar yanlış tahmin ediyor, o farkı öğren.
Widrow-Hoff öğrenme kuralından geliyor — ilişkisel bellek için tasarlanmış.

### Gated DeltaNet = Birleşim

```python
# Gated Delta Rule:
S_t = α_t · S_{t-1} + β_t · (v_t - S_{t-1}·k_t) · k_t^T

# Yorumu:
# α_t  → ne kadar eski belleği koru
# (v_t - S_{t-1}·k_t) → tahmin hatası (error signal)
# β_t  → bu hatadan ne kadar öğren
```

**Çok güzel bir özellik:** α_t gating, belleğin üzerine "L2 regularization" etkisi yaratıyor — belirsiz ya da eski bilgileri zamanla silindirir.

### Benchmark Sonuçları

| Model | Language Modeling | In-Context Retrieval | Long Context |
|-------|-----------------|---------------------|-------------|
| Mamba2 | İyi | Zayıf | Orta |
| DeltaNet | Orta | İyi | Orta |
| **Gated DeltaNet** | **En iyi** | **En iyi** | **En iyi** |
| Transformer | Referans | Referans | Referans |

Hybrid versiyonlar (GatedDeltaNet + sliding window attention) Transformer baseline'ı aşıyor.

### Endüstri Adaptasyonu

**Qwen3-Next** her 4 katmandan 3'ünü Gated DeltaNet olarak kullanıyor.
**Kimi Linear**: GDN'i temel alarak Kimi Delta Attention (KDA) geliştirdi — daha ince gating mekanizması ile.
**vLLM**: Triton kernel desteği eklendi.

### FPGA Donanım Sonucu (2026)

H100 GPU üzerindeki GDN decode, standart Transformer'dan *daha* memory-bound — yani GPU'dan tam verim alamıyorsunuz. FPGA implementasyonu ise H100'den 60x düşük latency ile GDN decode yapabildiğini gösterdi. Bu, GDN'in özelleşmiş donanımla gelecekte çok daha verimli hale gelebileceğine işaret ediyor.

---

## 🚀 KONU 3 — Speculative Decoding + MoE: Inference Hız Devrimi

### Speculative Decoding Nedir?

LLM inference token-by-token çalışır — her token ayrı bir forward pass. Bu yavaş.

Speculative Decoding şöyle çalışır:
```
1. Küçük/hızlı bir draft model N token üretir (örn. 5 token)
2. Büyük hedef model bu 5 tokeni PARALEL olarak doğrular
3. Eğer draft modelin tahminleri doğruysa, 5 token tek seferde kabul edilir
4. İlk yanlış tokenden itibaren büyük modelin çıktısı kullanılır
```

Bu sayede ortalama **1.5x–3.5x speedup** elde ediliyor — çıktı kalitesi değişmiyor.

### MoE + Speculative Decoding: Sorun ve Çözüm

**Temel sorun:** MoE'de her token farklı expert'leri aktive eder. 5 draft token, 5x daha fazla expert aktive edebilir → GPU memory pressure ve I/O darboğazı.

```
Örnek — Mixtral 8x7B:
- Tek token: 2 expert aktive (8'den)
- 5 draft token: 10 expert'e kadar aktive olabilir
- Veri hareketi 5x artıyor → hız avantajı kayboluyor
```

**2025 Çözümleri:**

### SpecMoEOff (Nanjing Üniversitesi, Ağustos 2025)
- MoE offloading + speculative decoding'i birleştiriyor
- GPU ve CPU'yu roofline analizi ile optimize ediyor
- **Sonuç: 2.5x throughput artışı** (SOTA MoE offloading tekniklerine göre)

### MoESD (arXiv Mayıs 2025)
- Teorik analiz: MoE modeller orta batch size'larda speculative decoding'den dense modellerden **daha çok** yararlanıyor
- MoE ne kadar sparse olursa, SD'nin etkili olduğu batch size aralığı o kadar genişliyor
- **Sonuç: Qwen2-57B-A14B'de 2.29x speedup**

### Cascade (vLLM, Haziran 2025)
- "Utility-driven" yaklaşım: her iteration'da `speculation utility = token_gains / verification_cost` hesaplar
- Utility < 1 ise speculation'ı devre dışı bırakır
- **Sonuç: 7-14% throughput artışı**, maksimum 5% yavaşlama garantisi

### MoE-Spec (2025)
- Expert budgeting: her layer için önceden expert kullanımını tahmin eder
- EAGLE-3 baseline'ı üzerinden **%10-30 throughput artışı**

### Kritik Insight: Batch Size Eşiği

```
Küçük batch (batch=1): SD genellikle faydalı değil MoE için
Orta batch (4-32):     SD + MoE çok faydalı ← altın nokta
Büyük batch:           Memory bottleneck → diğer teknikler daha iyi
```

### Qwen3-Next ile MTP (Multi-Token Prediction) Farkı

Speculative decoding ayrı bir draft model gerektirir. MTP ise modelin **kendi içindeki** ek prediction head'leri kullanır. Qwen3-Next'te MTP, ek GPU belleği olmadan 100+ token/saniye hız sağlıyor — pratikliği açısından speculative decoding'den üstün.

---

## 🎓 KONU 4 — RAFT: Eğitim Sırasında Retrieval'ın En Pratik Versiyonu

**Paper:** "RAFT: Adapting Language Model to Domain Specific RAG"
**Kurum:** Microsoft (Zhang et al., 2024)
**arXiv:** 2403.10131

### RAFT Nedir?

RAFT (Retrieval-Augmented Fine-Tuning) klasik RAG ile fine-tuning'i birleştiren bir eğitim tarifi. REALM/RETRO gibi "training sırasında retrieval yapan" modellerden farklı — burada retrieval eğitim öncesinde gerçekleşiyor, eğitim sırasında değil.

Ama kullanım sonucu aynı: model gerçek RAG ortamına daha hazır hale geliyor.

### RAFT Eğitim Kurgusu

```
[Eğitim verisi hazırlama]
     ↓
Her soru için:
  - 1 oracle (doğru) doküman
  - K adet distractor (ilgisiz, yanıltıcı) doküman

[Fine-tuning]
  - Model: "Doğru cevabı bul + Chain-of-Thought ile gerekçelendir"
  - Görevi: Hem doğru kaynağa odaklanmayı öğren
  - Hem de distractorları görmezden gelmeyi öğren
```

**Anahtar yenilik:** Model sadece "cevabı bul" değil "distractorları filtrele ve oracle'a odaklan" öğreniyor. Bu gerçek RAG kullanımını simüle ediyor — çünkü gerçek retrieval her zaman ilgili sonuçları getirmiyor.

### Chain-of-Thought Entegrasyonu

```
Soru: "X firmasının 2023 geliri nedir?"
Dokümanlar: [doğru finansal rapor] + [3 ilgisiz rapor]

RAFT hedef cevap formatı:
"<CITATION>Doküman 1</CITATION>'e göre [analiz...] 
Dokümanlar 2, 3, 4 bu soruyla ilgisiz çünkü [...] 
Bu nedenle cevap: $X milyardır."
```

### Pratik Sonuçlar

| Model | Domain | RAFT vs. Baseline RAG |
|-------|--------|----------------------|
| Llama2-7B | Domain QA | Önemli iyileşme |
| 7B model | EDA (Elektronik Tasarım Otomasyonu) | GPT-3.5 RAG'ı geçiyor |
| ALoFTRAG (LoRA+RAFT) | 20 dataset, 26 dil | Ortalama citation +8.3%, cevap +3.0% |

**Önemli bulgu:** Yeni nesil modellerde (Llama3-8B, Mistral-7B) RAFT'ın etkisi daha az belirgin — çünkü bu modeller zaten daha iyi instruction-following yapıyor. RAFT özellikle küçük, eski modellerde çok etkili.

### RAFT Varyantları Ekosistemi (2025)

```
RAFT (2024)
├── ALoFTRAG (Ocak 2025) — otomatik, LoRA, etiketsiz data, 26 dil
├── CRAFT (2024)         — compute-efficient, LoRA adapter switching
├── RbFT (Ocak 2025)     — adversarial/counterfactual retrieval'a dayanıklılık
└── GraphRAFT (Nisan 2025) — knowledge graph + RAFT, Cypher/SPARQL üretimi
```

### RAFT vs. Klasik Fine-Tuning vs. RAG

| Yöntem | Bilgi Güncelliği | Domain Adaptasyonu | Maliyet |
|--------|-----------------|-------------------|---------|
| Saf RAG | ✅ Anlık | ❌ Genel model kısıtları | Düşük |
| Fine-Tuning | ❌ Donmuş | ✅ Güçlü | Orta |
| **RAFT** | ✅ RAG ile dinamik | ✅ Distractor-robust | Orta |
| REALM/RETRO | ✅ Eğitimde entegre | ✅ Güçlü | Çok yüksek |

### Ne Zaman RAFT Kullanmalısınız?

✅ Domain-specific QA (tıp, hukuk, finans, EDA)
✅ Retrieval sonuçları gürültülü olduğunda (düşük precision retriever)
✅ Küçük modeli büyük model performansına taşımak istiyorsanız
✅ Gizlilik gereksinimleri olan yerelde çalışan sistemler (ALoFTRAG)
❌ Her zaman güncel bilgi gerektiren uygulamalar (saf RAG daha uygun)
❌ Çok büyük SOTA modeller (zaten instruction-following yeterli)

---

## 🏗️ KONU 5 — Qwen3-Next: Hibrit Mimari Üçgeni

**Model:** Qwen3-Next-80B-A3B (Instruct + Thinking varyantları)
**Kurum:** Alibaba Qwen Team
**Yayın:** Eylül 2025 | Apache 2.0 lisansı
**Lisans:** Ticari + akademik kullanım serbest

### Neden Önemli?

Dense modeller büyüdükçe inference maliyeti orantılı artıyor. Qwen3-Next bunu kırdı:
- **80 milyar** toplam parametre
- Ancak yalnızca **3 milyar** aktif per token (**%96.25 sparsity!**)
- Performans Qwen3-32B'ye eşit veya üstün, eğitim maliyeti 10x daha az

### Mimari Üçgeni

```
┌─────────────────────────────────────────┐
│           QWEN3-NEXT MİMARİSİ           │
│                                         │
│  [Hybrid Attention] + [Ultra-Sparse MoE] + [MTP] │
└─────────────────────────────────────────┘
```

### Bileşen 1: Hybrid Attention

**48 katmanın 36'sı:** Gated DeltaNet (linear attention)
**48 katmanın 12'si:** GQA (Grouped Query Attention) — tam softmax attention

```
Katman dağılımı: [GDN, GDN, GDN, GQA, GDN, GDN, GDN, GQA, ...]
Oran: 3:1 (Linear : Full)
```

**GDN katmanları:** Uzun context'i verimli işliyor, sabit bellek
**GQA katmanları:** Kesin retrieval, güçlü bağlam anlama, tam dikkat

Ek teknik detaylar:
- Head dimension: 256 (standart 128'in 2x'i → daha zengin representation)
- Rotary embedding: Sadece dimension'ların %25'ine uygulanıyor (partial RoPE)
- Context: 262,144 token (eğitim), 1 milyon token'a YaRN ile genişletilebilir

### Bileşen 2: Ultra-Sparse MoE

**512 routed + 1 shared expert**
**Her token için: 10 routed + 1 shared = 11 expert aktive**

```
Aktif oran: 10/512 = %1.95 → yani her forward pass'ta 
parametrelerin yalnızca ~%3.75'i kullanılıyor
```

**Dual-track tasarım:**
```
Input Token
    ↓
    ├── [Router] → Top-10 Specialist Experts → specialist output
    └── [Shared Expert] → her token için çalışır → general output
    ↓
Weighted Combination → Final Output
```

Shared expert: "genel pratisyen" — dil modellemesinin temel kalıpları
Routed experts: "uzmanlar" — belirli görev/domain bilgisi

**Expert yük dengeleme:** Router initialization sırasında parametreler normalize ediliyor → erken eğitimde tüm expertler eşit şans alıyor.

### Bileşen 3: Multi-Token Prediction (MTP)

Standard autoregressive decoding: 1 token / 1 forward pass

MTP: Model aynı anda birden fazla token tahmin ediyor:

```
[Mevcut durum] → [token_t, token_{t+1}, token_{t+2}] parallel prediction
                          ↓
                 Doğrulama (speculative verification)
                          ↓
                 Kabul edilen tokenler output'a eklenir
```

**Qwen3-Next öneri:** Inference sırasında 2 token tahmin et (2-token lookahead)

**Etki:**
- Eğitim: Model daha iyi "ileriye bakma" öğreniyor → daha az token ile daha iyi convergence
- Inference: 100+ token/saniye → DeepSeek R1'in 20-30 token/saniyesinden 3-5x hızlı

### Eğitim Detayları

- 15 trilyon token (Qwen3'ün 36T corpus'undan uniform sampling)
- Reinforcement learning: GSPO (Group Sampling Policy Optimization)
- Stability: Zero-centered + weight-decayed layernorm

### Performans Tablosu

| Karşılaştırma | 4K token input | 128K token input |
|--------------|---------------|-----------------|
| vs Qwen3-30B-A3B | Eşit hız | 3x daha hızlı |
| vs Qwen3-32B (dense) | 3x daha hızlı | 10x daha hızlı |
| Parametre verimliliği | 80B total / 3B aktif | %96.25 sparsity |

**Donanım:** NVIDIA Blackwell (NVLink 5.0, 1.8TB/s) ile optimize edilmiş — 512 expert routing için inter-GPU bandwidth kritik.

### Neden Bu Mimari Önemli?

Qwen3-Next, üç ayrı araştırma çizgisinin birleşimi:

```
Gated DeltaNet → Uzun context + linear scaling
    +
Ultra-sparse MoE → Parametre kapasitesi / compute ayrışması
    +
Multi-Token Prediction → Paralel inference + daha iyi eğitim
    =
Qwen3-Next: Dense model performansı, küçük model maliyeti
```

Bu, "Context Length Scaling ve Total Parameter Scaling" ikili trendinin pratik çözümü.

---

## 📊 GENEL PUANLAMA VE KARŞILAŞTIRMA

### Olgunluk ve Pratik Kullanılabilirlik

| Konu | Olgunluk | Prodüksiyon Hazırlığı | Araştırma Etkisi |
|------|---------|----------------------|----------------|
| HippoRAG 2 | 🟢 ICML 2025, GitHub mevcut | 🟡 Orta (indexleme maliyeti) | 🟢 Yüksek |
| Gated DeltaNet | 🟢 ICLR 2025, Qwen3-Next'te üretimde | 🟢 Yüksek | 🟢 Çok Yüksek |
| Spec. Dec + MoE | 🟡 Aktif araştırma | 🟡 Cascade ile kısmi | 🟢 Yüksek |
| RAFT | 🟢 2024'ten beri kurumsal | 🟢 Yüksek | 🟡 Orta-Yüksek |
| Qwen3-Next | 🟢 Apache 2.0, vLLM destekli | 🟢 Çok Yüksek | 🟢 Çok Yüksek |

### Birbirleriyle İlişkisi

```
HippoRAG 2 ──────┐
                  ├─── RAG Sistemlerini İyileştiriyor
RAFT ────────────┘

Gated DeltaNet ──┐
                  ├─── Qwen3-Next'in Temel Taşı
MoE + MTP ───────┘

Speculative Dec. ─── MoE ile birlikte inference optimize ediyor
```

---

## 💡 TAVSİYELER

**Eğer RAG sistemi geliştiriyorsanız:**
→ HippoRAG 2'yi multi-hop senaryolar için, RAFT'ı domain adaptasyonu için birleştirin.

**Eğer inference maliyeti sorunsa:**
→ Önce Qwen3-Next benzeri bir model seçin (aktif parametre sayısı az). Sonra MTP veya Speculative Decoding ekleyin.

**Eğer uzun context kritikse:**
→ Gated DeltaNet tabanlı hibrit mimari (3:1 linear:full) kullanın. Qwen3-Next bunu zaten yapıyor.

**Eğer LLM eğitiyorsanız:**
→ MTP her zaman açık olsun (ücretsiz hız + daha iyi pretraining convergence). Ultra-sparse MoE + hybrid attention: en verimli parametre/compute oranı.

---

## 🔗 TÜM KAYNAKLAR

| Konu | Kaynak | Link |
|------|--------|------|
| HippoRAG 2 | arXiv + ICML 2025 | https://arxiv.org/abs/2502.14802 |
| HippoRAG 2 | GitHub | https://github.com/OSU-NLP-Group/HippoRAG |
| Gated DeltaNet | arXiv | https://arxiv.org/abs/2412.06464 |
| Gated DeltaNet | Kimi Linear paper | https://arxiv.org/pdf/2510.26692 |
| SpecMoEOff | arXiv | https://arxiv.org/abs/2508.21706 |
| MoESD | arXiv | https://arxiv.org/abs/2505.19645 |
| Cascade (vLLM SD) | arXiv | https://arxiv.org/abs/2506.20675 |
| RAFT | arXiv | https://arxiv.org/abs/2403.10131 |
| ALoFTRAG | arXiv | https://arxiv.org/abs/2501.11929 |
| Qwen3-Next HF | HuggingFace | https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking |
| Qwen3-Next vLLM | vLLM Blog | https://blog.vllm.ai/2025/09/11/qwen3-next.html |
| Qwen3-Next NVIDIA | NVIDIA Dev Blog | https://developer.nvidia.com/blog/new-open-source-qwen3-next-models... |

---
