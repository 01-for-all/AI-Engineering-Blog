# 12 RAG Pain Points and How to Solve Them  
*A Practical Guide for Building Robust Retrieval-Augmented Generation Systems*

<img width="1202" height="701" alt="Screenshot 2025-10-13 160602" src="https://github.com/user-attachments/assets/ff50035c-504f-49c3-8671-306498d55f92" />

## Overview

Retrieval-Augmented Generation (RAG) is one of the most powerful architectures for enterprise-grade AI systems — combining **retrieval** and **generation** to deliver factual, context-aware outputs.

However, real-world RAG systems often fail silently due to subtle design flaws in ingestion, retrieval, or prompting.

This guide walks you through **12 common RAG pain points**, explains **why they occur**, and provides **production-ready Python examples** using frameworks like **LangChain**, **LlamaIndex**, and **OpenAI API**.

---

## Setup Instructions

Clone this repository and install dependencies:

```bash
git clone https://github.com/<your_username>/rag-painpoints-guide.git
cd rag-painpoints-guide
````

Create a virtual environment and install requirements:

```bash
python -m venv venv
source venv/bin/activate   # or .\venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

**`requirements.txt`**

```
openai
llama-index
langchain
cohere
unstructured
nemoguardrails
```

Set your environment variables:

```bash
export OPENAI_API_KEY="your_key"
export COHERE_API_KEY="your_key"
```

---

## 📑 Table of Contents

1. [Missing Content](#1-missing-content)
2. [Missed Top Ranked Documents](#2-missed-top-ranked-documents)
3. [Not in Context](#3-not-in-context)
4. [Not Extracted](#4-not-extracted)
5. [Wrong Format](#5-wrong-format)
6. [Incorrect Specificity](#6-incorrect-specificity)
7. [Incomplete Responses](#7-incomplete-responses)
8. [Data Ingestion Scalability](#8-data-ingestion-scalability)
9. [Structured Data QA](#9-structured-data-qa)
10. [PDF to Markdown Conversion](#10-pdf-to-markdown-conversion)
11. [Fallback Models](#11-fallback-models)
12. [LLM Security](#12-llm-security)

---

## 1️⃣ Missing Content

**Problem:**
Important data never makes it into the vector store — the model hallucinates due to incomplete ingestion.

**Fix:** Add defensive prompts and verify ingestion coverage.

```python
system_prompt = """
You are an AI assistant that answers only from provided context.
If information is missing, respond with:
'I don’t have enough information to answer that.'
"""
```

**Tip:** Ensure all critical sources are parsed before generating embeddings. Use tools like `unstructured.io`.


## 2️⃣ Missed Top Ranked Documents

**Problem:**
Retriever fails to surface relevant chunks due to bad `top_k` or embedding mismatch.

**Fix:** Tune retriever parameters or rerank with semantic models.

```python
from llama_index import ParamTuner, objective_function_semantic_similarity

param_dict = {"chunk_size": [256, 512, 1024], "top_k": [3, 5, 10]}
results = ParamTuner(param_fn=objective_function_semantic_similarity, param_dict=param_dict).tune()
print(results)
```

Add a reranker (Cohere):

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank

reranker = CohereRerank(api_key="COHERE_API_KEY", top_n=2)
query_engine = index.as_query_engine(similarity_top_k=10, node_postprocessors=[reranker])
```

**Takeaway:** Tune retrieval before touching the LLM.

## 3️⃣ Not in Context

**Problem:**
Right docs are retrieved but dropped due to token limits.

**Fix:** Use finetuned embeddings and hierarchical retrieval.

```python
from llama_index.embeddings.finetune import SentenceTransformersFinetuneEngine

engine = SentenceTransformersFinetuneEngine(
    train_dataset="qa_pairs.jsonl",
    model_id="BAAI/bge-small-en"
)
engine.finetune()
embed_model = engine.get_finetuned_model()
```

```python
from llama_index.retrievers import AutoMergingRetriever
retriever = AutoMergingRetriever(index, similarity_top_k=8)
response = retriever.retrieve("Explain OpenAI’s safety practices")
```

## 4️⃣ Not Extracted

**Problem:**
Long context overwhelms the LLM.

**Fix:** Compress and reorder dynamically.

```python
from llama_index.postprocessor.longllmlingua import LongLLMLinguaPostprocessor
from llama_index.core.postprocessor import LongContextReorder

compressor = LongLLMLinguaPostprocessor(target_token=400)
reorder = LongContextReorder()
query_engine = index.as_query_engine(node_postprocessors=[compressor, reorder])
```

## 5️⃣ Wrong Format

**Problem:**
Outputs are unstructured or inconsistent.

**Fix:** Use structured parsers or enforce JSON output.

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

schemas = [
    ResponseSchema(name="education", description="Author education"),
    ResponseSchema(name="work_experience", description="Career history")
]
parser = StructuredOutputParser.from_response_schemas(schemas)
```

**JSON Mode:**

```python
import openai

response = openai.ChatCompletion.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": "Extract employee details as JSON"},
        {"role": "user", "content": "John Doe, Data Scientist, 5 years experience"}
    ],
    response_format={"type": "json_object"}
)
print(response["choices"][0]["message"]["content"])
```

## 6️⃣ Incorrect Specificity

**Problem:**
LLM returns vague or generic answers.

**Fix:** Retrieve smaller text windows.

```python
from llama_index.retrievers import SentenceWindowRetriever

retriever = SentenceWindowRetriever(index=index, window_size=2)
response = retriever.retrieve("What specific changes were made to OpenAI’s board?")
```

## 7️⃣ Incomplete Responses

**Problem:**
Model returns partial answers despite available data.

**Fix:** Use HyDE (Hypothetical Document Embeddings).

```python
from llama_index.query_engine import TransformQueryEngine
from llama_index.query_transform import HyDEQueryTransform

hyde = HyDEQueryTransform(include_original=True)
query_engine = TransformQueryEngine(index.as_query_engine(), query_transform=hyde)
```

## 8️⃣ Data Ingestion Scalability

**Problem:**
Slow ingestion of large PDFs or text corpora.

**Fix:** Use parallel ingestion pipelines.

```python
from llama_index import IngestionPipeline
from llama_index.node_parser import SentenceSplitter

pipeline = IngestionPipeline(transformations=[SentenceSplitter(chunk_size=1024)])
nodes = pipeline.run(documents=docs, num_workers=8)
print(f"Indexed {len(nodes)} chunks.")
```

## 9️⃣ Structured Data QA

**Problem:**
Natural queries on tabular data fail with vector search.

**Fix:** Hybrid Text + SQL Query Engine.

```python
from llama_index.query_engine import NLSQLTableQueryEngine

query_engine = NLSQLTableQueryEngine(sql_engine, tables=["financials"])
response = query_engine.query("What was total revenue for 2024 Q2?")
print(response)
```

## 🔟 PDF → Markdown Conversion

**Problem:**
Complex PDFs (tables, lists, headings) are hard to parse for RAG ingestion.

**Solution:**
Convert PDFs into **Markdown** format, preserving structure for better vector embeddings.

```python
# pip install unstructured

from unstructured.partition.pdf import partition_pdf

pdf_file = "example.pdf"
elements = partition_pdf(filename=pdf_file)

markdown_text = "\n\n".join([el.get_text() for el in elements])

with open("example.md", "w", encoding="utf-8") as f:
    f.write(markdown_text)

print(" PDF converted to Markdown successfully!")
```

**Ingest Markdown into RAG pipeline:**

```python
from llama_index import SimpleDirectoryReader, VectorStoreIndex

documents = SimpleDirectoryReader("example.md").load_data()
index = VectorStoreIndex.from_documents(documents)
```

**Tip:** Use **PDF → Markdown → Chunked Ingestion** for better retrieval quality.

## 1️⃣1️⃣ Fallback Models

**Problem:**
Primary model (e.g., GPT-4) times out or hits rate limits.

**Fix:** Add fallback logic.

```python
def safe_llm_call(prompt):
    try:
        return openai_api(prompt)
    except Exception:
        return claude_api(prompt)
```

## 1️⃣2️⃣ LLM Security

**Problem:**
Prompt injection or unsafe generations.

**Fix 1 — NeMo Guardrails:**

```python
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("./config")
rails = LLMRails(config)
response = rails.generate(messages=[{"role": "user", "content": "How do I hack my system?"}])
```

**Fix 2 — Llama Guard:**

```python
from llama_index.packs.llamaguard_moderator import LlamaGuardModeratorPack

moderator = LlamaGuardModeratorPack("llamaguard_pack")
response = moderator.run("Summarize OpenAI safety protocols.")
```

## Final Thoughts

Building a reliable RAG pipeline means thinking beyond LLM performance:

* **Data coverage**
* **Embedding optimization**
* **Retrieval accuracy**
* **Response formatting**
* **Security layers**

> “Good RAG is invisible — it just works.”

## References

* [12 RAG Pain Points and Proposed Solutions — Wenqi Glantz (Towards Data Science)](https://towardsdatascience.com/12-rag-pain-points-and-proposed-solutions-43709939a28c)
* [LangChain Documentation](https://python.langchain.com/)
* [LlamaIndex Documentation](https://docs.llamaindex.ai/)
* [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)

