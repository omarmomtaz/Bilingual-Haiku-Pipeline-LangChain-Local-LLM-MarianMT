# ============================================================
# Bilingual Haiku Sequence Pipeline
#
# Architecture:
#   Step 1 → TinyLlama generates extra imagery words per topic
#   Step 2 → Python fills haiku templates (guaranteed clean structure)
#   Step 3 → MarianMT translates each line English → Japanese
#
# Key design: {B} slot = the entire 7-syllable middle line (never
# embedded mid-sentence). This eliminates all word-salad output.
#
# Models (free, auto-download, no API key):
#   - TinyLlama/TinyLlama-1.1B-Chat-v1.0   ~600MB
#   - Helsinki-NLP/opus-mt-en-jap           ~300MB
#
# Install:
#pip install langchain langchain-huggingface transformers torch sentencepiece
# ============================================================

import re
import random
import torch
from transformers import MarianMTModel, MarianTokenizer
from transformers import pipeline as hf_pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── 1. Templates & Imagery ────────────────────────────────────
#
# {A} fills into a 5-syllable line 1 or 3
# {B} IS the entire 7-syllable line 2 (never embedded mid-sentence)
# {C} fills into a 5-syllable line 3
#
# This guarantees clean readable English every time.

TEMPLATES = [
    ("the {A} drifts away",  "{B}",  "only {C} stays"),
    ("first light on {A}",   "{B}",  "cold {C} gleams"),
    ("the last {A} falls",   "{B}",  "{C} is gone"),
    ("still {A} at dawn",    "{B}",  "then {C} stirs"),
    ("pale {A} at dusk",     "{B}",  "dark {C} comes"),
    ("the {A} holds still",  "{B}",  "bare {C} waits"),
    ("one {A} remains",      "{B}",  "{C} fades slow"),
    ("deep {A} at night",    "{B}",  "then {C} gleams"),
]

IMAGERY = {
    "cherry blossom": {
        "A": ["petal","blossom","flower","pale bloom"],
        "B": ["silence falls on the stone path",
              "wind lifts petals from the bough",
              "the shrine wall holds the cold light",
              "grey clouds pass the empty branch"],
        "C": ["bare branch","cold ground","wet stone","thin frost"],
    },
    "winter moon": {
        "A": ["moonlight","pale glow","cold moon","blue frost"],
        "B": ["the frozen field holds no sound",
              "shadows cross the still black lake",
              "snow lies deep on the old roof",
              "light moves slow across the ice"],
        "C": ["dark sky","thin ice","deep cold","still night"],
    },
    "autumn leaves": {
        "A": ["red leaf","last leaf","dry leaf","brown oak"],
        "B": ["the forest floor is all red",
              "a cold wind moves the last branch",
              "the hillside gold has all gone",
              "the river runs with red leaves"],
        "C": ["bare oak","cold earth","still pond","grey sky"],
    },
    "morning frost": {
        "A": ["white frost","thin ice","cold mist","grey light"],
        "B": ["stiff grass stands in the pale light",
              "the grey roof holds the cold white",
              "a web of drops hangs in still air",
              "the cracked mud breathes without sound"],
        "C": ["hard ground","cracked mud","still air","cold stone"],
    },
    "summer cicada": {
        "A": ["cicada","dry shell","shrill call","hot branch"],
        "B": ["the high branch burns in still air",
              "one cry cuts the scorched dry field",
              "heat holds the last light of day",
              "the long dusk swallows the call"],
        "C": ["dusk falls","heat breaks","night comes","still air"],
    },
    "mountain snow": {
        "A": ["fresh snow","grey peak","deep white","cold ridge"],
        "B": ["the valley is lost in cloud",
              "the pine bends under white weight",
              "the path is gone without trace",
              "silence fills the cold dark woods"],
        "C": ["long dark","still woods","cold wind","deep still"],
    },
    "firefly": {
        "A": ["soft glow","brief light","small spark","pale flash"],
        "B": ["the tall reeds hold the dark still",
              "a light moves between the hills",
              "the dark field opens and fades",
              "night settles after the glow"],
        "C": ["black pond","gone light","still night","dark reeds"],
    },
    "falling rain": {
        "A": ["cold rain","grey rain","thin rain","soft drop"],
        "B": ["the old roof holds back the sky",
              "stone steps pool with the grey drops",
              "the window blurs with slow streams",
              "wet earth breathes without a sound"],
        "C": ["wet earth","damp stone","grey dusk","cold mud"],
    },
    "bamboo forest": {
        "A": ["green stalk","tall cane","split cane","bent stalk"],
        "B": ["wind moves through without a word",
              "pale light breaks on the far wall",
              "a shadow falls on dry ground",
              "the cold path bends and is gone"],
        "C": ["hush falls","path bends","dusk nears","still roots"],
    },
    "harvest moon": {
        "A": ["full moon","gold light","round moon","bright disc"],
        "B": ["the cut field glows in the dark",
              "the still road runs silver white",
              "roof tiles bright in autumn night",
              "a thin cloud crosses the glow"],
        "C": ["long night","thin cloud","cool wind","cold field"],
    },
    "spring mist": {
        "A": ["pale mist","soft haze","low cloud","grey veil"],
        "B": ["the hill has lost its edges",
              "the valley breathes without sound",
              "the bridge is half lost in white",
              "light shifts slow behind the haze"],
        "C": ["hush now","light shifts","mist lifts","hill fades"],
    },
    "frozen pond": {
        "A": ["cracked ice","black ice","white ice","thin glass"],
        "B": ["a reed bends low above the freeze",
              "the last leaf is caught in cold",
              "a dark tree stands in the glass",
              "no sound comes from below the ice"],
        "C": ["deep still","cold holds","light fades","dark freeze"],
    },
    "temple bell": {
        "A": ["bronze bell","one tone","low note","old bell"],
        "B": ["sound moves slow through morning fog",
              "a ring fades before it ends",
              "the empty hill holds the tone",
              "the air is still after the bell"],
        "C": ["then still","air holds","dusk comes","mist waits"],
    },
    "wild crane": {
        "A": ["grey crane","still crane","lone crane","white crane"],
        "B": ["one step in the shallow cold",
              "the crane stands in the still reeds",
              "no sound comes from the cold marsh",
              "the water ends at pale sky"],
        "C": ["sky waits","wind stirs","dawn comes","cold reeds"],
    },
    "persimmon tree": {
        "A": ["bright fruit","bare branch","last fruit","orange skin"],
        "B": ["one fruit hangs on the leafless bough",
              "colour holds where all else fades",
              "the branch waits for the cold wind",
              "frost forms slow on the bright skin"],
        "C": ["frost waits","wind comes","light goes","cold bough"],
    },
}

# ── 2. Load Models ────────────────────────────────────────────

print("=" * 62)
print("  Loading models (auto-download on first run ~900MB total)")
print("=" * 62)

print("\n[1/2] Loading imagery model (TinyLlama-1.1B-Chat)...")
gen_pipe = hf_pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=60,
    do_sample=True,
    temperature=0.9,
    repetition_penalty=1.3,
    pad_token_id=2,
    return_full_text=False,
)
llm = HuggingFacePipeline(pipeline=gen_pipe)
parser = StrOutputParser()
print("    ✓ Imagery model ready")

print("\n[2/2] Loading translation model (Helsinki-NLP MarianMT)...")
MT_MODEL = "Helsinki-NLP/opus-mt-en-jap"
mt_tokenizer = MarianTokenizer.from_pretrained(MT_MODEL)
mt_model = MarianMTModel.from_pretrained(MT_MODEL)
print("    ✓ Translation model ready")

# ── 3. LangChain: Imagery Word Generator ─────────────────────

imagery_prompt = PromptTemplate.from_template(
    "<|system|>Reply with a plain list of short words, one per line. "
    "No numbers, no explanations, no punctuation.</s>\n"
    "<|user|>Give 3 short nature words (1-2 words each) "
    "related to '{topic}' for haiku poetry. One per line only.</s>\n"
    "<|assistant|>"
)
imagery_chain = imagery_prompt | llm | parser

def get_extra_imagery(topic: str, bank: dict) -> dict:
    """Ask TinyLlama for extra A-slot words. Falls back silently."""
    try:
        raw = imagery_chain.invoke({"topic": topic})
        words = [
            l.strip().lower()
            for l in raw.splitlines()
            if l.strip()
            and not re.search(r'(sure|here|example|note|:|\d\.)', l, re.IGNORECASE)
            and 2 <= len(l.strip()) <= 20
        ]
        if len(words) >= 1:
            extended = {k: list(v) for k, v in bank.items()}
            extended["A"] = bank["A"] + words[:3]
            return extended
    except Exception:
        pass
    return bank

# ── 4. Poem Builder ───────────────────────────────────────────

def build_poem(topic: str) -> str:
    bank = IMAGERY.get(topic, random.choice(list(IMAGERY.values())))
    imagery = get_extra_imagery(topic, bank)

    stanzas = []
    used = set()
    attempts = 0
    while len(stanzas) < 4 and attempts < 20:
        attempts += 1
        idx = random.randrange(len(TEMPLATES))
        if idx in used:
            continue
        used.add(idx)
        t = TEMPLATES[idx]
        a = random.choice(imagery["A"])
        b = random.choice(imagery["B"])
        c = random.choice(imagery["C"])
        lines = [
            t[0].format(A=a, B=b, C=c),
            t[1].format(A=a, B=b, C=c),   # {B} = entire line 2
            t[2].format(A=a, B=b, C=c),
        ]
        stanzas.append('\n'.join(lines))
    return '\n\n'.join(stanzas)

# ── 5. Translation ────────────────────────────────────────────

def translate_line(text: str) -> str:
    inputs = mt_tokenizer(
        [text], return_tensors="pt",
        padding=True, truncation=True, max_length=128,
    )
    with torch.no_grad():
        out = mt_model.generate(**inputs)
    return mt_tokenizer.decode(out[0], skip_special_tokens=True)

def translate_poem(english_poem: str) -> str:
    stanzas = english_poem.strip().split('\n\n')
    result = []
    for stanza in stanzas:
        lines = [l.strip() for l in stanza.splitlines() if l.strip()]
        result.append('\n'.join(translate_line(l) for l in lines))
    return '\n\n'.join(result)

# ── 6. Display ────────────────────────────────────────────────

def display_poem(topic: str, english: str, japanese: str, n: int):
    w = 62
    print("\n" + "═" * w)
    print(f"  詩 #{n}  ·  {topic}")
    print("═" * w)
    print("\n  【日本語】\n")
    for line in japanese.splitlines():
        print(f"    {line}" if line.strip() else "")
    print("\n  【English】\n")
    for line in english.splitlines():
        print(f"    {line}" if line.strip() else "")
    print()

# ── 7. Run ────────────────────────────────────────────────────

topics = random.sample(list(IMAGERY.keys()), 3)
print(f"\n\n  Generating bilingual haiku sequences...\n")

results = []
for i, topic in enumerate(topics, 1):
    print(f"  [詩 #{i}] Topic: '{topic}' → composing & translating...")
    english  = build_poem(topic)
    japanese = translate_poem(english)
    display_poem(topic, english, japanese, i)
    results.append({"topic": topic, "english": english, "japanese": japanese})

print("═" * 62)
print(f"  Complete. {len(results)} bilingual haiku sequences generated.")
print("═" * 62)