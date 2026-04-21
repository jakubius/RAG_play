# RAG (Retrieval-Augmented Generation) System

A complete RAG pipeline for PDF document Q&A that combines semantic search with local LLM generation. Ask questions about your PDF documents and get answers grounded in the actual content, preventing hallucinations.

## 📋 Table of Contents

- [Overview](#overview)
- [Workflow Architecture](#workflow-architecture)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [How RAG Works](#how-rag-works)
- [Configuration](#configuration)
- [Performance Optimization](#performance-optimization)

---

## Overview

This RAG system performs three main operations:

1. **Indexing** (one-time): Extract PDF text → split into chunks → convert to embeddings → store in vector DB
2. **Querying** (per question): Convert question to embedding → retrieve similar chunks → generate answer with LLM
3. **Web Interface**: Flask app serving a chat UI for easy interaction

### Why RAG?

- **Accurate Answers**: Grounds responses in actual document content
- **No Hallucinations**: LLM can only reference what's in the PDF
- **Transparent Source**: Shows which page the answer came from
- **Offline**: Uses local models (no external API calls)

---

## Workflow Architecture

### 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG SYSTEM COMPONENTS                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐       │
│  │   PDF Input  │      │   Embedding  │      │ Vector Store │       │
│  │              │─────▶│   Model      │─────▶│  (Chroma)    │       │
│  │ self_driving │  (1) │ nomic-embed  │  (2) │   Database   │       │
│  │ portfolio.pdf│      │              │      │              │       │
│  └──────────────┘      └──────────────┘      └──────────────┘       │
│                                                         ▲             │
│                                                         │             │
│  ┌──────────────┐      ┌──────────────┐      ┌────────┴──────┐      │
│  │   Web UI     │      │  Flask App   │      │  Chroma Server│      │
│  │  (Browser)   │◀─────│              │──────│  (Port 8000)   │      │
│  │              │  (6) │  /chat route │  (5) │                │      │
│  └──────────────┘      └──────────────┘      └────────────────┘      │
│         ▲                      ▲                                       │
│         │                      │                                       │
│         │              ┌───────┴────────┐                             │
│         │              │                │                             │
│         │              │    Question    │      ┌──────────────┐       │
│         │              │   Embedding    │─────▶│   LLM Model  │       │
│         │              │   (via model)  │  (4) │  (Ollama)    │       │
│         │              └────────────────┘      │ gemma3:1b    │       │
│         │                     ▲                └──────────────┘       │
│         │                     │                        ▲              │
│         │                     │                        │              │
│         │              ┌──────┴─────────────────────────┤             │
│         │              │                                │             │
│         └──────────────┤  Context + Question  ──────────┘             │
│                        │  (Retrieved chunks)                          │
│                        │  (System prompt)                             │
│                        └─────────────────────                         │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 📦 INDEXING PIPELINE (One-Time Setup)

**When**: First time you run the application
**Duration**: ~30-60 seconds (done once, then cached)

```
┌──────────────────────────────────────────────────────────────────────┐
│                     STEP 1: PDF TEXT EXTRACTION                       │
│                Reads PDF and extracts text from each page              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  PDF File (self_driving_portfolio.pdf)                               │
│       │                                                               │
│       ▼                                                               │
│  ┌─────────────────────────────────────────────┐                     │
│  │ extract_pdf_pages() [PyMuPDF/fitz]          │                     │
│  │ - Opens PDF file                            │                     │
│  │ - Iterates through each page                │                     │
│  │ - Extracts text using get_text()            │                     │
│  │ INPUT: PDF path                             │                     │
│  │ OUTPUT: List of {page_num, text}            │                     │
│  └─────────────────────────────────────────────┘                     │
│       │                                                               │
│       ▼                                                               │
│  ┌─────────────────────────────────────────────┐                     │
│  │ render_pdf_pages() [PyMuPDF/fitz]           │                     │
│  │ - Renders each page as PNG image            │                     │
│  │ - Stores images for visual reference        │                     │
│  │ - Caches to page_images/ directory          │                     │
│  │ OUTPUT: page_num → image_path mapping       │                     │
│  └─────────────────────────────────────────────┘                     │
│       │                                                               │
│       ▼                                                               │
│  Pages: [                                                            │
│    {page_num: 1, text: "Introduction..."},                           │
│    {page_num: 2, text: "Self-driving cars..."},                      │
│    ...                                                               │
│  ]                                                                    │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                     STEP 2: TEXT CHUNKING                            │
│              Splits long text into smaller, overlapping pieces        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Extracted Pages → RecursiveCharacterTextSplitter                    │
│                                                                        │
│  ┌─────────────────────────────────────────────┐                     │
│  │ build_chunks()                              │                     │
│  │ - Uses RecursiveCharacterTextSplitter       │                     │
│  │ - chunk_size = 256 tokens (~50-60 words)    │                     │
│  │ - chunk_overlap = 32 tokens (12% overlap)   │                     │
│  │ - Preserves text continuity across chunks   │                     │
│  │ OUTPUT: List of chunks + metadata           │                     │
│  └─────────────────────────────────────────────┘                     │
│       │                                                               │
│       ▼                                                               │
│                                                                        │
│  Example:                                                             │
│  ┌──────────────────────────────────────────────────────┐            │
│  │ Chunk 1: "Introduction. Self-driving vehicles...     │            │
│  │          are the future of transportation. They..."  │◀───┐       │
│  └──────────────────────────────────────────────────────┘    │       │
│                                                              Overlap   │
│  ┌──────────────────────────────────────────────────────┐    │       │
│  │ Chunk 2: "...are the future of transportation. They  │◀───┘       │
│  │          will reduce accidents. Autonomous systems..."│            │
│  └──────────────────────────────────────────────────────┘            │
│                                                                        │
│  Result: ~1000-5000 chunks depending on document size               │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                 STEP 3: EMBEDDING GENERATION                         │
│       Converts text chunks to 768-dimensional numerical vectors      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Text Chunks                                                         │
│       │                                                               │
│       ▼                                                               │
│  ┌─────────────────────────────────────────────┐                     │
│  │ SentenceTransformer Model                   │                     │
│  │ Model: nomic-ai/nomic-embed-text-v1.5       │                     │
│  │                                             │                     │
│  │ "Self-driving vehicles reduce accidents"   │                     │
│  │         │                                   │                     │
│  │         ▼                                   │                     │
│  │ [0.234, -0.156, 0.789, ..., 0.432]        │  ← 768 dimensions   │
│  │ (numerically represents meaning)           │                     │
│  │                                             │                     │
│  │ Similar texts get similar vectors!         │                     │
│  └─────────────────────────────────────────────┘                     │
│       │                                                               │
│       ▼                                                               │
│  Embeddings List: [[vec1], [vec2], ..., [vecN]]                    │
│  (one 768-dim vector per chunk)                                     │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│              STEP 4: STORE IN VECTOR DATABASE                        │
│  Save chunks + embeddings + metadata for fast semantic search        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Chroma Vector Database (localhost:8000)                             │
│                                                                        │
│  ┌─ Collection: "pdf_rag" ──────────────────────────────────────┐   │
│  │                                                               │   │
│  │  ID        │ Document (Text)        │ Embedding    │ Metadata│   │
│  │  ──────────┼────────────────────────┼──────────────┼─────────│   │
│  │ chunk_0    │ "Self-driving vehicles │ [0.234, ...]│page: 1  │   │
│  │            │  are the future..."    │             │         │   │
│  │  ──────────┼────────────────────────┼──────────────┼─────────│   │
│  │ chunk_1    │ "...reduce accidents   │ [0.156, ...]│page: 1  │   │
│  │            │  and improve safety"   │             │         │   │
│  │  ──────────┼────────────────────────┼──────────────┼─────────│   │
│  │ chunk_2    │ "Autonomous systems    │ [0.789, ...]│page: 2  │   │
│  │            │  use sensors and AI"   │             │         │   │
│  │  ...       │ ...                    │ ...          │ ...     │   │
│  │                                                               │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  ✅ Indexing Complete! Data persists across sessions                │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 💬 QUERY PIPELINE (Per Question)

**When**: Every time user asks a question
**Duration**: ~2-5 seconds

```
┌────────────────────────────────────────────────────────────────────────┐
│              STEP 5A: EMBED THE QUESTION                               │
│         Convert user's question to same embedding space as chunks      │
├────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  User Question (from Web UI)                                           │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │ "How do self-driving cars prevent accidents?"               │      │
│  └──────────────────────────────────────────────────────────────┘      │
│       │                                                                  │
│       ▼                                                                  │
│  SentenceTransformer Model (same one from indexing)                    │
│       │                                                                  │
│       ▼                                                                  │
│  Question Embedding = [0.245, -0.167, 0.801, ..., 0.419]              │
│  (768-dimensional vector representing question meaning)                │
│                                                                          │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│           STEP 5B: SEMANTIC SEARCH (RETRIEVAL)                         │
│     Query vector database to find most similar chunks (top_k=4)        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Question Embedding: [0.245, -0.167, 0.801, ...]                      │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ Chroma Vector Database Query                               │       │
│  │ collection.query(                                          │       │
│  │   query_embeddings=[question_embedding],                  │       │
│  │   n_results=4  # TOP_K                                    │       │
│  │ )                                                          │       │
│  │                                                             │       │
│  │ Algorithm: Cosine Similarity                              │       │
│  │ Find chunks with embeddings closest to question           │       │
│  │ in 768-dimensional vector space                           │       │
│  └─────────────────────────────────────────────────────────────┘       │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ RETRIEVAL RESULTS (Sorted by Similarity Score)             │       │
│  ├─────────────────────────────────────────────────────────────┤       │
│  │                                                             │       │
│  │ 🥇 Rank 1 (Score: 0.87)                                    │       │
│  │    Text: "Autonomous vehicles use radar and LIDAR sensors  │       │
│  │          to detect obstacles. This prevents collisions..."│       │
│  │    Page: 3                                                 │       │
│  │                                                             │       │
│  │ 🥈 Rank 2 (Score: 0.81)                                    │       │
│  │    Text: "AI systems continuously analyze the surrounding  │       │
│  │          environment and decide the safest driving path..."│       │
│  │    Page: 4                                                 │       │
│  │                                                             │       │
│  │ 🥉 Rank 3 (Score: 0.78)                                    │       │
│  │    Text: "Emergency braking systems can stop the vehicle   │       │
│  │          in milliseconds, faster than human reaction..."  │       │
│  │    Page: 5                                                 │       │
│  │                                                             │       │
│  │ 4️⃣  Rank 4 (Score: 0.75)                                   │       │
│  │    Text: "Self-driving cars reduce human errors which are  │       │
│  │          the cause of 90% of car accidents..."            │       │
│  │    Page: 2                                                 │       │
│  │                                                             │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                                                                          │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│           STEP 5C: PREPARE CONTEXT FOR LLM                             │
│      Join retrieved chunks into a single context string               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Retrieved Chunks + Retrieved Metadata                                 │
│       │                                                                  │
│       ▼                                                                  │
│  ┌────────────────────────────────────────────────────────────┐        │
│  │ Context String (formatted for readability):                │        │
│  │                                                            │        │
│  │ Autonomous vehicles use radar and LIDAR sensors to detect │        │
│  │ obstacles. This prevents collisions...                    │        │
│  │                                                            │        │
│  │ ---                                                        │        │
│  │                                                            │        │
│  │ AI systems continuously analyze the surrounding           │        │
│  │ environment and decide the safest driving path...         │        │
│  │                                                            │        │
│  │ ---                                                        │        │
│  │                                                            │        │
│  │ Emergency braking systems can stop the vehicle in          │        │
│  │ milliseconds, faster than human reaction...               │        │
│  │                                                            │        │
│  │ ---                                                        │        │
│  │                                                            │        │
│  │ Self-driving cars reduce human errors which are the cause │        │
│  │ of 90% of car accidents...                                │        │
│  └────────────────────────────────────────────────────────────┘        │
│                                                                          │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│        STEP 5D: GENERATE ANSWER WITH LLM                               │
│     Send context + question to Ollama for answer generation            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │ Ollama Local LLM (gemma3:1b)                             │          │
│  │                                                          │          │
│  │ Input Messages:                                          │          │
│  │ ┌────────────────────────────────────────────────────┐  │          │
│  │ │ System: "You are a helpful RAG assistant. Answer   │  │          │
│  │ │         only from the provided context. If the     │  │          │
│  │ │         answer is not in the context, say you      │  │          │
│  │ │         couldn't find it in the document."         │  │          │
│  │ └────────────────────────────────────────────────────┘  │          │
│  │                                                          │          │
│  │ ┌────────────────────────────────────────────────────┐  │          │
│  │ │ User: "Context:                                    │  │          │
│  │ │        [4 chunks retrieved above]                  │  │          │
│  │ │                                                    │  │          │
│  │ │        Question: How do self-driving cars prevent │  │          │
│  │ │        accidents?"                                │  │          │
│  │ └────────────────────────────────────────────────────┘  │          │
│  │                                                          │          │
│  │ Processing:                                              │          │
│  │ - LLM reads system prompt (stay grounded in context)    │          │
│  │ - LLM reads retrieved chunk context                     │          │
│  │ - LLM generates answer based ONLY on context            │          │
│  │ - LLM outputs response token-by-token                   │          │
│  │                                                          │          │
│  └──────────────────────────────────────────────────────────┘          │
│       │                                                                  │
│       ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │ LLM RESPONSE (Grounded in Context)                       │          │
│  │                                                          │          │
│  │ "Self-driving cars prevent accidents through several    │          │
│  │  key mechanisms:                                         │          │
│  │                                                          │          │
│  │  1. Sensor Technology: They use radar and LIDAR sensors │          │
│  │     to detect obstacles and other vehicles, allowing    │          │
│  │     them to react faster than humans.                   │          │
│  │                                                          │          │
│  │  2. AI Decision-Making: Autonomous systems continuously │          │
│  │     analyze the environment and determine the safest    │          │
│  │     driving path.                                       │          │
│  │                                                          │          │
│  │  3. Emergency Response: Emergency braking can stop the  │          │
│  │     vehicle in milliseconds, much faster than human     │          │
│  │     reaction time.                                      │          │
│  │                                                          │          │
│  │  4. Error Reduction: By eliminating human errors which  │          │
│  │     cause 90% of accidents, autonomous vehicles        │          │
│  │     significantly improve safety."                     │          │
│  │                                                          │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                          │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│        STEP 5E: RETURN VISUAL REFERENCE                                │
│     Extract page numbers and provide image of source material          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Retrieved Chunk Metadata: [page: 3, page: 4, page: 5, page: 2]      │
│       │                                                                  │
│       ▼                                                                  │
│  Select First Page Number: 3                                           │
│       │                                                                  │
│       ▼                                                                  │
│  Lookup Image Path: page_images/page_3.png                            │
│       │                                                                  │
│       ▼                                                                  │
│  Return to Web UI:                                                      │
│  {                                                                       │
│    "answer": "Self-driving cars prevent accidents through...",        │
│    "page": 3,                                                          │
│    "image_url": "/page-image/3"  ← User can click to see original    │
│  }                                                                       │
│                                                                          │
│  ✅ Question fully processed and answered!                             │
│                                                                          │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 🌐 WEB INTERFACE FLOW

```
┌────────────────────────────────────────────────────────────────┐
│                    WEB APPLICATION FLOW                         │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Browser (User Interface)                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. Page loads: GET /                                   │  │
│  │     ↓                                                    │  │
│  │  2. Flask serves index.html (chat interface)            │  │
│  │     ↓                                                    │  │
│  │  3. User types question and clicks "Ask"               │  │
│  │     ↓                                                    │  │
│  │  4. JavaScript sends: POST /chat                        │  │
│  │     {question: "How do self-driving cars...?"}         │  │
│  └──────────────────────────────────────────────────────────┘  │
│       │                                                          │
│       ▼                                                          │
│  Flask Server (web_app.py)                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  @app.route("/chat", methods=["POST"])                 │  │
│  │  ├─ Extract question from JSON request                 │  │
│  │  ├─ Validate question is not empty                     │  │
│  │  ├─ Call: ask_rag(question, COLLECTION, MODEL, ...)   │  │
│  │  │   ↓                                                  │  │
│  │  │   This triggers the full RAG query pipeline!        │  │
│  │  │   ↓                                                  │  │
│  │  ├─ Receive: (answer, page_num, image_path)            │  │
│  │  ├─ Build response JSON                                │  │
│  │  └─ Return: {answer, page, image_url}                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│       │                                                          │
│       ▼                                                          │
│  Browser (Display Results)                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Answer displayed to user:                             │  │
│  │  "Self-driving cars prevent accidents through..."       │  │
│  │                                                          │  │
│  │  [Display PDF page 3 as visual reference]              │  │
│  │  GET /page-image/3  ← Fetches PNG image               │  │
│  │                                                          │  │
│  │  User can continue asking questions                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- Ollama (for local LLM inference)
- Chroma (vector database server)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Start External Services

**Terminal 1 - Start Ollama (LLM Server)**
```bash
# First install Ollama from https://ollama.ai
ollama serve
# Pull the model if not already downloaded
ollama pull gemma3:1b
```

**Terminal 2 - Start Chroma (Vector Database)**
```bash
pip install chromadb
chroma run --host localhost --port 8000
```

### Step 3: Run the Application

**Terminal 3 - Start RAG Application**
```bash
python web_app.py
```

Your browser will automatically open to `http://localhost:5000`

---

## Usage

### Web Interface

1. **Ask Questions**: Type any question about the PDF in the chat box
2. **Get Answers**: RAG retrieves relevant content and generates grounded answers
3. **See Sources**: Visual reference shows which page the answer came from
4. **Continue Conversation**: Ask follow-up questions

### Example Questions for `self_driving_portfolio.pdf`

- "How do self-driving cars prevent accidents?"
- "What sensors do autonomous vehicles use?"
- "Explain the AI decision-making process in self-driving cars"
- "What are the safety benefits of autonomous vehicles?"

---

## How RAG Works

### The RAG Process (Step-by-Step)

#### 1. **Embeddings = Numerical Meaning**

Embeddings convert text into numerical vectors that capture semantic meaning:

```
Text: "Self-driving cars use AI systems"
      ↓
Embedding: [0.234, -0.156, 0.789, ..., 0.432]  (768 numbers)
      ↓
Similar texts get similar embeddings, enabling semantic search!
```

#### 2. **Semantic Search ≠ Keyword Search**

```
Keyword Search (Simple):
  Question: "How do cars prevent accidents?"
  Results: Only pages containing exact words "cars" + "prevent"

Semantic Search (RAG):
  Question: "How do cars prevent accidents?"
  Results: Pages about safety, sensors, emergency braking
           (even if exact words don't match!)
```

#### 3. **Context Prevents Hallucinations**

```
❌ Without RAG:
   Q: "How do autonomous vehicles prevent accidents?"
   A: "They use quantum tunneling and teleportation..." ← HALLUCINATION

✅ With RAG:
   Q: "How do autonomous vehicles prevent accidents?"
   System Prompt: "Answer only from the provided context"
   Context: "[Retrieved chunks about sensors, AI, safety]"
   A: "They use radar/LIDAR sensors, AI decision-making,
       and fast emergency braking..." ← GROUNDED IN DOCUMENT
```

---

## Configuration

Edit these values in `rag_core.py` to tune RAG performance:

### Document Configuration
```python
PDF_PATH = "self_driving_portfolio.pdf"  # PDF to analyze
IMAGES_DIR = "page_images"               # Where to save page images
COLLECTION_NAME = "pdf_rag"              # Vector DB collection name
```

### Embedding Model (Step 3 of indexing)
```python
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"  # Current: good balance
# Alternatives:
# - "all-MiniLM-L6-v2"   # Faster, less accurate
# - "all-mpnet-base-v2"  # More accurate, slower
```

### LLM Configuration (Step 5D of query)
```python
OLLAMA_MODEL = "gemma3:1b"  # 1B parameters = lightweight
OLLAMA_HOST = "http://localhost:11434"  # Ollama server address
```

### Chunking Strategy (Step 2 of indexing) ⚠️ **High Impact on Quality**
```python
chunk_size = 256        # Chunk size in tokens (~50-60 words)
                        # Try 128 (focused), 512 (more context), 1024 (full sections)
chunk_overlap = 32      # Overlap between chunks
                        # Increase to 64-128 for better boundary preservation
```

### Retrieval Parameters (Step 5B of query)
```python
TOP_K = 4               # Number of chunks to retrieve
                        # Try 3-10: More = more context but slower
```

### System Prompt (Step 5D of query) ⚠️ **Controls LLM Behavior**
```python
SYSTEM_PROMPT = """You are a helpful RAG assistant.
Answer only from the provided context.
If the answer is not in the context, say you couldn't find it.
Be concise but useful."""
```

---

## Performance Optimization

### Indexing Speed (One-Time)

**Current**: ~30-60 seconds for 20-page PDF

**Optimize**:
- Increase `chunk_size` (fewer chunks = faster indexing)
- Use faster embedding model: `all-MiniLM-L6-v2`
- Skip image rendering if not needed (comment out `render_pdf_pages()`)

### Query Speed (Per Question)

**Current**: ~2-5 seconds per question

**Optimize**:
- Reduce `TOP_K` (fewer chunks = faster retrieval)
- Use smaller LLM: `tinyllama`, `phi` (1B vs 2-3B parameters)
- Keep `chunk_size` reasonable (avoid too many chunks)

### Retrieval Quality (Accuracy)

**Improve**:
- Experiment with `chunk_size` (try 512 or 1024)
- Increase `TOP_K` (retrieve more context)
- Use better embedding model: `all-mpnet-base-v2`
- Implement query expansion (rephrase question multiple ways)
- Add metadata to chunks (section titles, document structure)
- Re-rank results using cross-encoders

### Memory Usage

**Reduce**:
- Use smaller embedding model
- Process documents in batches
- Clear vector DB cache periodically
- Use lighter LLM model (`tinyllama` instead of `gemma3`)

---

## Troubleshooting

### "Connection refused to Chroma"
- Ensure Chroma is running: `chroma run --host localhost --port 8000`

### "Ollama refused connection"
- Ensure Ollama is running: `ollama serve`
- Check `OLLAMA_HOST` is correct in `rag_core.py`

### "Slow query times"
- Reduce `TOP_K` parameter
- Use smaller embedding model
- Use smaller LLM model

### "Poor answer quality"
- Increase `TOP_K` to retrieve more context
- Increase `chunk_size` for more complete information
- Check if question is actually covered in the PDF
- Try different embedding model for better semantic understanding

---

## Architecture Summary

```
Component                Purpose                    Technology
──────────────────────────────────────────────────────────────
PDF Input               Document source            PyMuPDF (fitz)
Text Extraction         Extract page text          PyMuPDF
Text Chunking          Split into pieces           LangChain
Embedding Model        Text → Numbers              SentenceTransformer
Vector Database        Store + search              Chroma
Web Server             User interface              Flask
LLM                    Answer generation           Ollama
```

---

## Future Improvements

- [ ] Hybrid search (semantic + keyword)
- [ ] Query expansion and rephrasing
- [ ] Multi-turn conversation history
- [ ] Confidence scores for answers
- [ ] Result re-ranking using cross-encoders
- [ ] Document metadata filtering
- [ ] Caching for frequent questions
- [ ] User feedback mechanism
- [ ] Fine-tuned embedding models
- [ ] Production WSGI server (Gunicorn)

---

## License

MIT

