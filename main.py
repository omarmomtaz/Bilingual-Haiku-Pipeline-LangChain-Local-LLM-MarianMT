# pip install langchain langchain-huggingface transformers torch langchain-community

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain, SequentialChain
from transformers import pipeline

print("Loading AI model... (first time takes 1-2 minutes)")

# Hugging Face text generation pipeline
hf_pipeline = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=50,
    temperature=0.7
)

# Wrap it for LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)

print("Model loaded! Creating LangChain...\n")

# ============================================
# CHAIN 1: Generate a random topic
topic_prompt = PromptTemplate(
    input_variables=[],
    template="""Generate one random topic word for a haiku.
Examples: ocean, mountain, sunset, autumn, cherry blossom.
Just write one word, nothing else.

Topic:"""
)

topic_chain = LLMChain(
    llm=llm,
    prompt=topic_prompt,
    output_key="topic"
)

# ============================================
# CHAIN 2: Write a haiku about the topic
haiku_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""Write a beautiful haiku (3 lines: 5-7-5 syllables) about: {topic}

Haiku:"""
)

haiku_chain = LLMChain(
    llm=llm,
    prompt=haiku_prompt,
    output_key="haiku"
)

# ============================================
# SEQUENTIAL CHAIN: Combine both chains
full_chain = SequentialChain(
    chains=[topic_chain, haiku_chain],
    input_variables=[],
    output_variables=["topic", "haiku"],
    verbose=True  # Shows the chain execution steps
)

# ============================================
# RUN THE CHAIN!
print("="*60)
print("LANGCHAIN DEMO: Topic Generator → Haiku Writer")
print("="*60)
print("\nRunning two-step chain...\n")

# RUN the full chain
result = full_chain.invoke({})

# Display results
print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(f"\n📝 Generated Topic: {result['topic'].strip()}")
print(f"\n🎋 Haiku:\n{result['haiku'].strip()}")
print("\n" + "="*60)

# Run it multiple times to see different results
print("\n\nGenerating 2 more haikus...\n")

for i in range(2):
    print(f"\n--- Haiku #{i+2} ---")
    result = full_chain.invoke({})
    print(f"Topic: {result['topic'].strip()}")
    print(f"Haiku:\n{result['haiku'].strip()}")
    print("-" * 40)
