# 🎋 Bilingual Haiku Pipeline — LangChain + Local LLM + MarianMT

> A three-step AI pipeline that generates Japanese haiku sequences
> with English translations. Fully local. No API key. Runs on CPU.

---

## Demo Output

```
══════════════════════════════════════════════════════════════
  詩 #2  ·  falling rain
══════════════════════════════════════════════════════════════

  【日本語】

    第 一 の 光 が , 雨 の 上 に 降 っ て い っ た .
    石 で は , したたり を も っ て 海 を 洗 い ,
    わたし たち は , 冷た な 地 を 耕 し て い る の で す .

    冷 この 雨 は , 寒 い 雨 の 上 に とどま る で あ ろ う か .
    その 窓 は 激し く な り ,
    忍耐 は 寛容 を 待 つ こと を 好 み ,

  【English】

    first light on thin rain
    stone steps pool with the grey drops
    cold wet earth gleams

    one cold rain remains
    the window blurs with slow streams
    grey dusk fades slow
```

---

## Pipeline Architecture

```
Step 1 → TinyLlama generates extra imagery words per topic
Step 2 → Python assembles words into strict 5-7-5 haiku templates
Step 3 → Helsinki-NLP MarianMT translates each line to Japanese
```

Each step is independently modular — swap any component without
touching the others.

---

## Why This Design

Small LLMs cannot reliably output structured poetry. Every attempt
to prompt TinyLlama into writing 3 lines with syllable counts
produced prose, commentary, and hallucinated examples.

The solution: use the LLM only for what it does well (short creative
word generation), and use Python for what it does better (enforcing
structure). The result is clean, consistently formatted bilingual
haiku every run.

This is a real production principle: **don't use an LLM where
deterministic logic works better.**

---

## Tech Stack

| Component | Tool |
|---|---|
| Pipeline orchestration | LangChain LCEL (`prompt \| llm \| parser`) |
| Imagery generation | TinyLlama-1.1B-Chat (local, CPU) |
| Haiku assembly | Python template engine (guaranteed 5-7-5) |
| Translation | Helsinki-NLP opus-mt-en-jap (MarianMT direct) |

---

## Setup

```bash
pip install langchain langchain-huggingface transformers torch sentencepiece
python main_poetry.py
```

Models download automatically on first run (~900MB, one-time).
No API key. No account. No cost.

---

## Key Engineering Decisions

**Why bypass HuggingFace pipeline() for translation?**
Transformers v5 removed the `translation` task from the pipeline
registry entirely. Calling `MarianMTModel` directly is more robust,
version-stable, and gives finer control. When a library removes an
abstraction, the underlying model still works — go one level deeper.

**Why a template engine instead of LLM-generated structure?**
LLMs hallucinate structure. Python enforces it. The `{B}` slot in
every template IS the entire 7-syllable middle line — never embedded
mid-sentence. This eliminates word-salad output completely.

**Why translate line-by-line instead of the full poem?**
MarianMT is optimized for sentence-level input. Feeding a full
multi-stanza poem collapses structure. Line-by-line translation
with stanza reassembly preserves the visual format exactly.

---

## Real-World Applications of This Pattern

| This Project | Production Equivalent |
|---|---|
| Topic → poem → translation | Document → summary → translated report |
| Template engine + LLM | Structured data extraction with LLM enrichment |
| MarianMT direct inference | On-premise NLP (GDPR / data privacy compliance) |
| LangChain LCEL chain | Multi-step RAG or agent pipeline |

---

## Skills Demonstrated

- ✅ LangChain LCEL pipeline composition (`prompt | llm | parser`)
- ✅ Local LLM inference (TinyLlama, CPU-only, no API cost)
- ✅ Direct HuggingFace model inference (bypassing broken abstractions)
- ✅ Prompt engineering for instruction-tuned models (ChatML format)
- ✅ Seq2seq translation with MarianMT
- ✅ Hybrid AI+deterministic system design
- ✅ Modular, maintainable pipeline architecture

---

## Extending This Pattern

| Extension | How |
|---|---|
| Higher poetry quality | Swap TinyLlama → Mistral-7B or Claude API |
| Better translation | Swap MarianMT → DeepL API (500k chars/month free) |
| Different language | Change opus-mt model slug (100+ language pairs) |
| Web interface | Wrap `run_pipeline()` in a FastAPI endpoint |
| RAG pipeline | Add retriever step before generation |

---

## About

Built as part of my AI/ML engineering portfolio.

📬 [[Upwork Profile](https://www.upwork.com/freelancers/~01044b5d38a3457dc1?mp_source=share)] · [[LinkedIn](https://www.linkedin.com/in/omar-momtaz-/)] · [[X/Twitter](https://x.com/omarmomtaz_main)]
