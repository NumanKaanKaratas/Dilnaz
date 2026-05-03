# Dilnaz

Dilnaz is a two-stage semantic language modeling research project. Instead of training the autoregressive model to predict the next discrete token directly, Dilnaz trains it to predict the next semantic distribution and then renders that distribution back into surface text.

The core hypothesis is simple: a model that learns the flow of meaning before surface form can generalize differently from a model that only learns token transitions. Words such as `araba`, `otomobil`, and `car` are different byte/token sequences, but they can occupy nearby regions in a multilingual semantic space. Dilnaz is built around that distinction.

## Architecture

Dilnaz has two models:

```text
surface text
  -> HybridTokenizer
  -> DIL: surface/context -> semantic distribution
  -> NAZ: semantic sequence -> next semantic distribution
  -> DIL renderer: semantic distribution -> surface text
```

### DIL

`DIL` is the bridge between surface text and semantic space.

It reads hybrid-tokenized text with left context and produces a latent distribution:

```text
surface/context -> encoder -> mean, log_std
```

It also renders latent vectors back to byte/surface output:

```text
latent -> renderer -> byte logits + length logits
```

DIL is trained with:

- reconstruction cross entropy
- length loss
- KL loss
- NLLB-based grouped layer geometry distillation
- mean geometry loss
- variance regularization

This keeps DIL from becoming only a memorizing autoencoder. Reconstruction preserves the written form; NLLB distillation shapes the latent space toward multilingual semantic geometry.

### NAZ

`NAZ` is the semantic sequence model.

Its target is not a token id. Its target is the next `mean + log_std` semantic distribution produced by a frozen DIL encoder:

```text
meaning_1, meaning_2, meaning_3 -> meaning_4
```

During generation, NAZ does not decode generated text and re-encode it. DIL encodes the prompt once, NAZ continues in semantic space, and DIL renders the generated semantic plan at the end or in decode chunks:

```text
prompt surface -> DIL encoder once -> initial semantic states
NAZ -> next_mean, next_log_std
NAZ -> next_mean, next_log_std
...
generated means -> DIL renderer -> text
```

This semantic loop keeps generation focused on meaning flow instead of feeding surface reconstruction errors back into the model.

## Semantic Backbone

NAZ uses a native Dilnaz semantic backbone. It is not a wrapper around an external language-model backbone.

The default pattern is:

```text
L0  SemanticDeltaMixer
L1  SemanticDeltaMixer
L2  SemanticDeltaMixer
L3  SemanticGlobalAttention
repeat...
```

The backbone combines:

- recurrent semantic mixing for long-range flow
- periodic global attention for direct context access
- partial rotary embeddings
- zero-centered RMS normalization
- gated feed-forward layers
- explicit generation cache

The backbone operates over semantic vectors, not vocabulary logits.

## Hybrid Tokenizer

Dilnaz uses a hybrid surface tokenizer:

- byte fallback for coverage
- compact surface pieces for frequent forms
- fixed-width segment encoding through `max_word_bytes`
- preserved punctuation and spacing behavior

The tokenizer is a surface interface. It does not define semantic meaning by itself. Semantic alignment is learned by DIL through reconstruction and NLLB teacher geometry.

Default vocabulary:

```text
dilnaz/tokenization/hybrid_surface_vocab.json
```

## Why NLLB?

NLLB is used as a multilingual semantic teacher. Dilnaz needs a semantic space where related meanings can be close even when surface forms differ across languages.

DIL does not use NLLB as a decoder. It uses NLLB encoder representations as a geometry target:

```text
NLLB encoder layers -> grouped geometry targets -> DIL layer vectors + latent mean
```

This helps DIL learn not only how to reconstruct a word, but also how words and context pieces relate semantically.

## Training

Train DIL first:

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

Then train NAZ from the frozen DIL checkpoint:

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

`resident` mode caches data in memory for fast experiments. `streaming` mode keeps the pipeline usable for larger text files.

## Inference

Inspect DIL reconstruction and semantic behavior:

```powershell
cd D:\Projects\Dilnaz\dilnaz\train

python interface_dil.py `
  --checkpoint-dir ../../checkpoints/Dil `
  --text "Yahudi toplulukları ile olan irtibatlarının oldukça azalması."
```

Generate with NAZ:

```powershell
cd D:\Projects\Dilnaz\dilnaz\train

python interface_naz.py `
  --checkpoint-dir ../../checkpoints/Naz `
  --max-new-tokens 512 `
  --num-samples 8 `
  --text "Yahudiler, dünyanın dört bir tarafına dağılmış topluluklardan"
```

## Compile Strategy

Dilnaz compiles only pure tensor cores:

- `DilEncoderCore`
- `DilDecoderRenderer`
- `NazStudentCore`

It does not compile the full model object. Tokenization, checkpointing, cache objects, random sampling, and loss bookkeeping stay outside the compiled graph.

Current checkpoint contracts:

```text
DIL format_version = 7
NAZ format_version = 10
```

Backward checkpoint compatibility is intentionally not maintained while the architecture is evolving.

## Repository Layout

```text
dilnaz/
  models/
    modeling_dil.py
    modeling_naz.py
    naz_backbone/
  tokenization/
  train/
    train_dil.py
    train_naz.py
    interface_dil.py
    interface_naz.py
tests/
```

Large artifacts are intentionally ignored:

- checkpoints
- external references
- local training datasets
- generated caches

## Status

Dilnaz is an experimental research codebase. The current focus is validating semantic next-step modeling, improving DIL reconstruction quality, scaling NAZ context length, and testing multilingual semantic generalization.
