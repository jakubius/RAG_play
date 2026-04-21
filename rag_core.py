"""
RAG (Retrieval-Augmented Generation) Pipeline for PDF Documents

This module implements a complete RAG system that:
1. Extracts text and renders images from PDFs
2. Splits document text into chunks (pieces)
3. Converts chunks into numerical embeddings (vector representations)
4. Stores embeddings in a vector database (Chroma)
5. When a question is asked, retrieves relevant chunks
6. Sends the question + relevant context to an LLM (Ollama) for answer generation

RAG improves LLM accuracy by grounding responses in actual document content,
preventing hallucinations and ensuring answers are based on the provided material.

POTENTIAL IMPROVEMENTS FOR RAG EFFECTIVENESS:
- Experiment with chunk_size and chunk_overlap (256/32 may not be optimal)
- Add metadata like document title, section headings to improve retrieval
- Implement re-ranking of retrieved chunks using cross-encoders
- Add filtering by document metadata in semantic search
- Use hybrid search combining semantic + keyword-based retrieval
- Implement query expansion or multi-query techniques
- Add feedback mechanism to track which chunks were most helpful
- Consider dense passage retrieval or fine-tuned embedding models
- Implement caching for frequently asked questions
"""

import os
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# ============================================================================
# CONFIGURATION: Adjust these values to optimize RAG performance
# ============================================================================

PDF_PATH = "self_driving_portfolio.pdf"  # Path to the PDF document to analyze
IMAGES_DIR = "page_images"  # Directory where PDF pages will be saved as PNG images
COLLECTION_NAME = "pdf_rag"  # Name of the vector database collection in Chroma

# Chroma vector database connection details
CHROMA_HOST = "localhost"  # Chroma server address (must be running separately)
CHROMA_PORT = 8000  # Chroma server port

# Embedding model: Converts text chunks into 768-dimensional vectors
# These vectors enable semantic similarity search
# Alternatives: "all-MiniLM-L6-v2" (faster), "all-mpnet-base-v2" (more accurate)
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"

# LLM model for generating answers from context
# Running locally via Ollama (no API costs, private, faster for small models)
OLLAMA_MODEL = "gemma3:1b"  # 1B params = lightweight but capable

# TOP_K: Number of most relevant chunks to retrieve for context
# Higher = more context but slower; Lower = faster but might miss relevant info
# Range: 3-10 is typical. More for complex questions, less for simpler ones.
TOP_K = 4

# System prompt defines the assistant's behavior and constraints
# Key instruction: "Answer only from the provided context" prevents hallucinations
SYSTEM_PROMPT = """You are a helpful RAG assistant.
Answer only from the provided context.
If the answer is not in the context, say you couldn't find it in the document.
Be concise but useful.
"""

# ============================================================================
# STEP 1: PDF TEXT EXTRACTION
# ============================================================================

def extract_pdf_pages(path: str):
    """
    Extract plain text from each PDF page.

    Process:
    1. Open the PDF file using PyMuPDF (fitz)
    2. Iterate through each page
    3. Extract text using optical character recognition (get_text())
    4. Store each page's text with its page number

    Returns: List of dicts with format {'page_num': int, 'text': str}

    Limitations & Improvements:
    - get_text() may lose formatting (tables, columns)
    - For complex PDFs, consider using layout-aware extraction
    - Could add image captions or metadata as separate chunks
    """
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({"page_num": i + 1, "text": text})
    doc.close()
    return pages


# ============================================================================
# STEP 1b: PDF PAGE RENDERING (VISUAL REFERENCE)
# ============================================================================

def render_pdf_pages(path: str, out_dir: str, zoom: float = 2.0):
    """
    Convert PDF pages to PNG images (for visual reference in UI).

    Process:
    1. Create output directory if it doesn't exist
    2. For each PDF page:
       - Render page as high-resolution image (zoom=2.0 = 2x quality)
       - Save as PNG to disk (skip if already cached)
       - Store mapping of page_number -> image_path
    3. Return dictionary for quick image lookups by page number

    Why this matters in RAG:
    - When retrieving relevant text, we can show the original page visually
    - Helps users verify the context and see surrounding content
    - Improves trust by showing "where" the answer came from

    Parameters:
    - zoom: Resolution multiplier (2.0 = 200% quality, slower but clearer)

    Returns: Dict mapping page_num -> absolute_path_to_image

    Optimization opportunities:
    - Cache images to avoid re-rendering (already does this!)
    - Could use lower zoom for faster startup, higher for detail
    - Could generate thumbnails for preview, full res on demand
    """
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(path)
    image_paths = {}
    matrix = fitz.Matrix(zoom, zoom)  # Scaling matrix for high-resolution rendering

    for i, page in enumerate(doc):
        image_path = os.path.abspath(os.path.join(out_dir, f"page_{i+1}.png"))
        if not os.path.exists(image_path):  # Skip if already cached
            pix = page.get_pixmap(matrix=matrix)  # Render page to image
            pix.save(image_path)  # Save image to disk
        image_paths[i + 1] = image_path

    doc.close()
    return image_paths


# ============================================================================
# STEP 2: TEXT CHUNKING (BREAKING DOCUMENTS INTO RETRIEVABLE PIECES)
# ============================================================================

def build_chunks(pages):
    """
    Split document text into overlapping chunks for vector embedding.

    Why chunking matters in RAG:
    - Embeddings work best on moderate-length text (not entire documents)
    - Smaller chunks = more precise retrieval but more overhead
    - Need overlap to preserve context at chunk boundaries

    Chunking Strategy (RecursiveCharacterTextSplitter):
    - Splits text hierarchically: paragraphs > sentences > characters
    - Preserves logical boundaries when possible
    - More intelligent than simple character splitting

    Current settings:
    - chunk_size=256: ~50-60 words per chunk
    - chunk_overlap=32: 12% overlap between chunks
    - Chunk overlap helps catch relevant content split at boundaries

    Process:
    1. For each page, split text into chunks
    2. Filter out empty chunks
    3. Create metadata (page number) for each chunk
    4. Return parallel lists of chunk texts and metadata

    Returns: Tuple of (chunk_texts list, metadatas list)

    ⚠️  TUNING OPPORTUNITIES (Major impact on RAG quality):
    - chunk_size=256 is small; try 512-1024 for document-level context
    - chunk_size=128 for more focused retrieval (less noise)
    - Increase overlap (64) to better handle boundary cases
    - Could use semantic chunking to split at topic boundaries
    - Could add overlapping windows (5-sentence sliding window)
    - Consider chunk_size relative to question length
    - Test with actual queries to find optimal chunk size!
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    chunk_texts = []
    metadatas = []

    for page in pages:
        splits = splitter.split_text(page["text"])
        for chunk in splits:
            if chunk.strip():  # Skip empty chunks
                chunk_texts.append(chunk)
                metadatas.append({"page": page["page_num"]})

    return chunk_texts, metadatas


# ============================================================================
# STEP 3: VECTOR DATABASE MANAGEMENT (CHROMA)
# ============================================================================

def get_collection():
    """
    Connect to Chroma vector database and get or create a collection.

    Chroma stores embeddings (vectors) and documents in a searchable index.

    Returns: Chroma collection object for storing/querying embeddings

    What happens:
    - Connects to Chroma server at localhost:8000
    - Gets existing collection named "pdf_rag" or creates it if new
    - Collection acts like a table in a database

    Note: Chroma server must be running separately!
    Start with: chroma run --host localhost --port 8000
    """
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


# ============================================================================
# STEP 4: INDEXING (CONVERTING TEXT TO EMBEDDINGS AND STORING)
# ============================================================================

def ensure_indexed(collection, model):
    """
    One-time indexing: Convert all chunks to embeddings and store in Chroma.

    RAG Pipeline Step 1 - Indexing (done once):
    1. Check if collection already indexed (skip if yes)
    2. Extract all text from PDF
    3. Split into chunks
    4. Convert each chunk to embedding (768-dimensional vector)
    5. Store chunks + embeddings + metadata in Chroma

    Embeddings explained:
    - Text is converted to numbers using SentenceTransformer
    - Similar texts get similar numbers (vectors)
    - This enables semantic similarity search

    Performance notes:
    - If collection already has data, returns immediately (no reindexing)
    - Indexing is slow (~time proportional to document size)
    - But only happens once, then reused for all queries
    - Cached result shared across Q&A sessions

    Why embeddings enable RAG:
    - When user asks a question, convert question to embedding
    - Find chunks with most similar embeddings (semantic search)
    - Those similar chunks are the most relevant context
    - Send that context + question to LLM
    """
    if collection.count() > 0:  # Skip if already indexed
        return

    pages = extract_pdf_pages(PDF_PATH)
    chunks, metadatas = build_chunks(pages)
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()  # Convert to vectors

    # Store everything in Chroma for fast retrieval later
    collection.add(
        documents=chunks,  # The actual text chunks
        embeddings=embeddings,  # Numerical vectors for similarity search
        metadatas=metadatas,  # Associated data (page numbers)
        ids=[f"chunk_{i}" for i in range(len(chunks))]  # Unique IDs for tracking
    )


# ============================================================================
# STEP 5: RAG QUERY (THE CORE Q&A PIPELINE)
# ============================================================================

def ask_rag(question: str, collection, model, image_paths, top_k: int = TOP_K):
    """
    Complete RAG pipeline: Answer question using document context.

    RAG Pipeline Step 2 - Query & Generation:

    STEP 5a: EMBED THE QUESTION
    - Convert user question to embedding using same model as chunks
    - Question embedding has same dimension as chunk embeddings
    - Allows apples-to-apples comparison

    STEP 5b: SEMANTIC SEARCH (RETRIEVAL)
    - Query Chroma database for top_k most similar chunks
    - Similarity = how close embeddings are in vector space
    - Returns documents + metadata (page numbers) + distances
    - No keyword matching, pure semantic similarity!

    STEP 5c: PREPARE CONTEXT
    - Join retrieved chunks with separators for readability
    - Create context string to send to LLM
    - LLM sees: System prompt + Context + Question

    STEP 5d: GENERATE ANSWER (LLM)
    - Send context + question to Ollama (local LLM)
    - System prompt steers LLM to only use context
    - LLM generates answer grounded in relevant chunks
    - This (context + LLM) is what makes RAG work!

    STEP 5e: RETURN VISUAL REFERENCES
    - Extract page numbers from retrieved chunks metadata
    - Get first matching page image for visual reference
    - User can see "where" the answer came from

    Returns: Tuple of (answer_text, best_page_number, image_path)

    ⚠️  IMPROVEMENT IDEAS FOR RETRIEVAL QUALITY:
    - Batch queries if asking multiple questions
    - Re-rank results after retrieval (cross-encoder)
    - Try different top_k values (more context ≠ always better)
    - Filter results by relevance score (optional threshold)
    - Implement query expansion: rephrase question multiple ways
    - Log which chunks were actually used vs retrieved
    - Add relevance scores to answer metadata
    """
    # Step 5a: Convert question to embedding
    query_embedding = model.encode([question])[0].tolist()

    # Step 5b: Semantic search - find most similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Step 5c: Prepare context from retrieved chunks
    docs = results["documents"][0]  # Extract chunk texts
    metas = results["metadatas"][0]  # Extract metadata (page numbers)
    context = "\n\n---\n\n".join(docs)  # Join with separators for clarity

    # Step 5d: Send to LLM with context and question
    ollama_client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    response = ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},  # Constrains LLM behavior
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )

    # Step 5e: Extract and return visual reference
    page_hits = [m.get("page") for m in metas if m and "page" in m]
    best_page = page_hits[0] if page_hits else None  # First matching page
    best_image = image_paths.get(best_page) if best_page else None

    return response["message"]["content"], best_page, best_image


# ============================================================================
# INITIALIZATION: One-time setup that loads all components
# ============================================================================

def initialize_rag():
    """
    Initialize the complete RAG system in the correct order.

    This function:
    1. Loads the embedding model from HuggingFace (first time is slow)
    2. Renders PDF pages as images (for visual reference)
    3. Connects to Chroma vector database
    4. Indexes document (first time only)
    5. Returns all components ready for querying

    Returns: Tuple of (embedding_model, chroma_collection, image_path_dict)

    Performance:
    - First run: ~30-60 seconds (indexing is slow)
    - Subsequent runs: ~5 seconds (index cached in Chroma)

    This function is called once at startup to set up the RAG system.
    """
    # Load embedding model from HuggingFace Hub
    # trust_remote_code=True allows custom code in the model
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)

    # Render all PDF pages as PNG images
    image_paths = render_pdf_pages(PDF_PATH, IMAGES_DIR)

    # Connect to Chroma database
    collection = get_collection()

    # One-time indexing: convert text to embeddings and store
    ensure_indexed(collection, model)

    # Return all components needed for ask_rag() to work
    return model, collection, image_paths