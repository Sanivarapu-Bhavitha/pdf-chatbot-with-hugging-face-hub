# pdf-chatbot-with-hugging-face-hub
PDF Chatbot built with Hugging Face Hub — upload PDFs, extract text, generate embeddings, store in a vector DB, and chat using RAG. Supports OCR for scanned files, FAISS/Chroma for search, and Streamlit UI. Easily deployable and push indexes or models to Hugging Face for reuse.

# PDF Chatbot with Hugging Face Hub — Repo README

## What is this

A reproducible, deployable repository that builds a PDF question-answering chatbot using Retrieval-Augmented Generation (RAG).
It ingests PDF(s), extracts text, creates embeddings, stores vectors (FAISS / Chroma / Milvus), and answers user queries by retrieving relevant chunks and passing them to a language model (local or Hugging Face Inference API). Includes a simple web UI (Streamlit/Flask) and instructions to publish assets/models/datasets to the Hugging Face Hub.

## Key features

* PDF ingestion and chunking (configurable chunk size & overlap)
* OCR support fallback (Tesseract) for scanned PDFs
* Text embedding using sentence-transformers or Hugging Face embedding models
* Vector store support: FAISS (local), Chroma (local), or instructions for managed DBs (Weaviate, Milvus)
* Retriever + re-ranker pipeline
* Uses LLM for final answer (local `transformers` / Hugging Face Inference API / OpenAI optional)
* Simple web UI (Streamlit) for chat interface with conversation history
* Scripts to push embeddings/datasets and models to Hugging Face Hub
* Dockerfile and basic GH Actions for CI / deployment

---

## Repository structure (suggested)

```
pdf-chatbot/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ docker/
│  └─ Dockerfile
├─ app/
│  ├─ streamlit_app.py         # lightweight UI
│  └─ flask_app.py             # optional REST API
├─ ingestion/
│  ├─ pdf_reader.py            # extract text + OCR fallback
│  ├─ chunker.py               # text chunking logic
│  └─ ingest_to_store.py       # pipeline to embeddings + vector store
├─ embeddings/
│  ├─ embed.py                 # wrapper for embedding model (HF or sentence-transformers)
│  └─ models_config.yaml
├─ retriever/
│  ├─ retriever.py             # retrieve top-k chunks + re-ranking
│  └─ reranker.py              # optional
├─ lm/
│  ├─ hf_inference.py          # call Hugging Face Inference API / local transformers
│  └─ prompt_templates/        # prompts for RAG and system instructions
├─ tests/
│  └─ test_ingest.py
├─ examples/
│  └─ sample.pdf
└─ scripts/
   ├─ push_to_hf.py            # push artifacts to Hugging Face Hub
   └─ build_index.sh
```

---

## Quickstart — Local (minimal)

1. Clone:

```bash
git clone https://github.com/<your-org>/pdf-chatbot.git
cd pdf-chatbot
```

2. Install:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

3. Ingest PDF(s) → build index:

```bash
python ingestion/ingest_to_store.py \
  --pdf-path examples/sample.pdf \
  --chunk-size 800 --chunk-overlap 200 \
  --embed-model sentence-transformers/all-MiniLM-L6-v2 \
  --vector-store faiss \
  --index-path ./indexes/sample_index.faiss
```

4. Start Streamlit UI:

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` and chat with your PDF.

---

## Example code snippets

### a) PDF text extraction & chunking (simplified)

```python
# ingestion/pdf_reader.py (excerpt)
from pathlib import Path
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract

def extract_text_from_pdf(path, ocr_if_needed=True):
    doc = fitz.open(path)
    texts = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            texts.append(text)
        elif ocr_if_needed:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)
            texts.append(text)
    return "\n\n".join(texts)

def chunk_text(text, chunk_size=800, overlap=200):
    tokens = text.split()  # simple whitespace split or use tokenizers for token-level chunking
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = " ".join(tokens[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks
```

### b) Embedding wrapper

```python
# embeddings/embed.py
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
```

### c) Save to FAISS

```python
# ingestion/ingest_to_store.py (excerpt)
import faiss
import numpy as np
from embeddings.embed import Embedder

embedder = Embedder(model_name)
vectors = embedder.embed_texts(chunks)  # shape (N, D)
d = vectors.shape[1]
index = faiss.IndexFlatIP(d)  # or IndexIVFFlat after training
faiss.normalize_L2(vectors)
index.add(vectors)
faiss.write_index(index, output_path)
# Save metadata (chunk text, page no., source) as JSON/SQLite in parallel
```

### d) Querying + RAG prompt

```python
# retriever/retriever.py (excerpt)
def retrieve_and_answer(query, index, metadata_store, top_k=5):
    q_vec = embedder.embed_texts([query])
    faiss.normalize_L2(q_vec)
    D, I = index.search(np.array(q_vec).astype("float32"), top_k)
    chunks = [metadata_store[i]['text'] for i in I[0]]
    context = "\n\n---\n\n".join(chunks)
    prompt = f"SYSTEM: You are a helpful assistant. Use the context below to answer.\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer concisely:"
    # send prompt to LLM (HF Inference API or local model)
    answer = hf_inference.ask(prompt)
    return answer
```

### e) Hugging Face Inference API call (Python)

```python
# lm/hf_inference.py (excerpt)
from huggingface_hub import InferenceClient
client = InferenceClient(api_key="HF_TOKEN")
def ask(prompt, model="gpt2"):
    out = client.text_generation(model=model, inputs=prompt, parameters={"max_new_tokens":250})
    return out[0]["generated_text"]
```

*(Replace `gpt2` with an appropriate HF LLM — check Hugging Face for models that support text generation and RAG use-cases.)*

---

## Pushing artifacts to Hugging Face Hub

* Login: `huggingface-cli login` (or set `HF_API_TOKEN` env var)
* Create repo (optional):

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("username/pdf-chatbot-index", repo_type="dataset")  # or repo_type="model"
```

* Use `huggingface_hub.upload_file` or `hf_hub_download` to upload files (embeddings, metadata, or model weights).

Example `scripts/push_to_hf.py` can automate pushing the index and metadata JSON.

---

## Configuration & Tips

* Chunking: tune `chunk_size` and `overlap` to balance context vs. retrieval precision. 500–1000 words with 100–300 overlap often works.
* Embeddings model: trade off between accuracy and speed (all-MiniLM vs larger SBERT models).
* Vector store: FAISS is simple and local; use Milvus/Weaviate for scalable production.
* OCR: use `pytesseract` + `pdf2image` for scanned PDFs.
* Re-ranking: consider a cross-encoder (pairwise ranker) for top-10 re-ranking before generation.
* Safety: sanitize PDF inputs and rate-limit file uploads on deployed apps.
* Cost: using hosted LLMs (Hugging Face Inference / OpenAI) will incur costs — add usage caps.

---

## Example `requirements.txt` (starter)

```
streamlit
sentence-transformers
faiss-cpu
pymupdf
pdf2image
pytesseract
huggingface-hub
transformers
torch
datasets
python-multipart
```

---

## Docker (basic)

`docker/Dockerfile` (simple):

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## Example Streamlit UI (concept)

* Upload PDF(s)
* Ingest button (shows progress)
* Chat input box; shows conversation with sources cited (chunk text + page number)
* Option to choose model (local vs HF) and set `top_k`

---

## Usage examples

* Local quick test: small PDF → FAISS index → Streamlit chat
* Production: upload PDF → create/update vector DB → serve via REST API (Flask) → connect to frontend or chatbot platform
* Publish index or dataset to HF Hub for reuse and reproducibility

---

## Security & privacy

* PDFs often contain PII — do NOT upload sensitive documents to public hubs without scrubbing.
* If publishing embeddings or datasets, remove PII or store in private repo on Hugging Face.

---

## License

Recommend `MIT` for template code, or choose a license that fits your organization.

---

## Roadmap / Extensions

* Multi-document conversation context management (per-doc grounding)
* Streaming LLM responses + answer highlights
* User feedback loop to improve retrieval (click-to-relevance)
* Support for other document formats (PowerPoint, Word, HTML)
* Deployment templates for AWS / GCP / Hugging Face Spaces
