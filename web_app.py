"""
RAG Web Interface - Flask Web App for PDF Document Q&A

This module provides a web UI for the RAG system using Flask.
Features:
- RESTful API endpoints for chat
- Web interface (HTML/CSS/JS in templates/)
- Image serving for PDF pages
- Automatic browser opening on startup

The core RAG logic is in rag_core.py; this just provides the UI layer.
"""

import os
import time
import threading
import traceback
import webbrowser
from flask import Flask, request, jsonify, render_template, send_file
from rag_core import initialize_rag, ask_rag

# ============================================================================
# WEB SERVER CONFIGURATION
# ============================================================================

APP_HOST = "127.0.0.1"  # Localhost only (not accessible from network)
APP_PORT = 5000  # Default Flask port

# ============================================================================
# FLASK APP SETUP AND GLOBAL STATE
# ============================================================================

app = Flask(__name__)

# Global variables to hold RAG components after initialization
# These persist across requests, initialized once at startup
MODEL = None  # SentenceTransformer embedding model
COLLECTION = None  # Chroma vector database collection
IMAGE_PATHS = None  # Mapping of page numbers to image file paths


# ============================================================================
# BROWSER AUTOMATION: Attempt to open Opera browser (can try Firefox, Chrome)
# ============================================================================

def open_opera(url: str):
    """
    Attempt to open a URL in Opera browser, fallback to default browser.

    Why separate function:
    - Allows specifying preferred browser (Opera in this case)
    - Degrades gracefully to default browser if Opera not found
    - Cleaner error handling for browser unavailability

    Platform support:
    - macOS: "open -a Opera" command
    - Windows/Linux: "opera" command

    Returns: Opens browser window with the URL
    """
    candidates = [
        "open -a Opera %s",  # macOS - explicitly open Opera app
        "opera %s"  # Windows/Linux - call opera command
    ]
    for candidate in candidates:
        try:
            browser = webbrowser.get(candidate)
            browser.open(url)
            return
        except webbrowser.Error:
            continue
    # Fallback: use default system browser if Opera not available
    webbrowser.open(url)


# ============================================================================
# FLASK ROUTES: API endpoints for the web interface
# ============================================================================

@app.route("/")
def home():
    """
    Serve the main web interface (HTML page).

    Returns: Rendered HTML from templates/index.html
    The HTML file contains the chat UI and JavaScript for interaction.
    """
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Core RAG endpoint: Process user question and return answer.

    Request format (JSON):
    {
        "question": "What is the main topic of this document?"
    }

    Process:
    1. Extract question from request
    2. Validate question is not empty
    3. Call ask_rag() from rag_core to:
       - Embed the question
       - Retrieve relevant document chunks
       - Generate answer using LLM
    4. Return answer + page reference + image URL

    Response format (JSON):
    {
        "answer": "The document discusses...",
        "page": 5,
        "image_url": "/page-image/5"
    }

    Error handling:
    - Returns 400 if question is missing or empty
    - Returns 500 if any error occurs during RAG processing

    ⚠️  IMPROVEMENT OPPORTUNITIES:
    - Add request validation (max question length)
    - Add rate limiting to prevent abuse
    - Log all queries for analytics/improvement
    - Add confidence scores to answers
    - Support multi-turn conversation history
    - Allow filtering by date range or sections
    """
    try:
        data = request.get_json()
        question = (data or {}).get("question", "").strip()
        if not question:
            return jsonify({"error": "Question is required"}), 400

        # Call the core RAG function from rag_core.py
        answer, best_page, best_image = ask_rag(question, COLLECTION, MODEL, IMAGE_PATHS)

        # Build image URL if image found (for web client)
        image_url = f"/page-image/{best_page}" if best_page and best_image else None

        return jsonify({
            "answer": answer,
            "page": best_page,
            "image_url": image_url
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/page-image/<int:page_num>")
def page_image(page_num):
    """
    Serve a specific PDF page as PNG image.

    URL path: /page-image/5 -> returns PNG of page 5

    Process:
    1. Look up page number in IMAGE_PATHS mapping
    2. Check if file exists on disk
    3. Serve file to web client

    This allows the web UI to display the original PDF pages
    alongside the RAG answers, helping users verify the source.

    Returns:
    - PNG image file if found (200)
    - "Image not found" message if page not indexed (404)
    """
    image_path = IMAGE_PATHS.get(page_num)
    if not image_path or not os.path.exists(image_path):
        return "Image not found", 404
    return send_file(image_path)


# ============================================================================
# STARTUP AUTOMATION: Open browser in background after server starts
# ============================================================================

def open_browser_later():
    """
    Open the web interface in browser after giving server time to start.

    Why separate thread:
    - Flask app takes ~1-2 seconds to start
    - Can't open browser before server is ready
    - Threading allows both to happen in parallel

    Process:
    1. Sleep for 1.5 seconds (give server time to start)
    2. Open browser to localhost:5000
    3. User sees UI automatically (better UX)

    Daemon thread:
    - If this thread is running when main thread exits, app exits anyway
    - No need to explicitly shut down browser thread
    """
    time.sleep(1.5)  # Wait for Flask server to be ready
    open_opera(f"http://{APP_HOST}:{APP_PORT}")


# ============================================================================
# MAIN ENTRY POINT: Initialize RAG and start web server
# ============================================================================

def main():
    """
    Main entry point: Initialize RAG system and start Flask web server.

    Startup sequence:
    1. Initialize RAG system (load models, index document)
       - This sets global variables: MODEL, COLLECTION, IMAGE_PATHS
       - First run is slow (~30-60 sec), subsequent runs ~5 sec
    2. Start browser-opening thread (daemon)
    3. Run Flask development server
       - Hot reload disabled (debug=False) for stability
       - Accessible at http://127.0.0.1:5000

    Global variables:
    - MODEL: Initialized by initialize_rag() - used in /chat route
    - COLLECTION: Chroma vector DB - used in /chat route
    - IMAGE_PATHS: Dict mapping page nums to image paths - used in both routes

    Error handling:
    - Catches and prints full traceback if anything goes wrong
    - Waits for user input before closing (better UX for errors)

    ⚠️  PRODUCTION IMPROVEMENTS (not for development):
    - Use production WSGI server (Gunicorn/uWSGI) instead of Flask dev server
    - Add logging framework instead of print()
    - Add request authentication/authorization
    - Use environment variables for configuration
    - Add monitoring and alerting
    - Use connection pooling for Chroma/Ollama
    """
    global MODEL, COLLECTION, IMAGE_PATHS
    try:
        print("Initializing RAG...")
        # Load all components: embedding model, vector DB, images
        MODEL, COLLECTION, IMAGE_PATHS = initialize_rag()

        # Start daemon thread to open browser after server is ready
        threading.Thread(target=open_browser_later, daemon=True).start()

        # Run Flask web server (blocking)
        app.run(host=APP_HOST, port=APP_PORT, debug=False)
    except Exception:
        print("\nAn error occurred:\n")
        traceback.print_exc()
        input("\nPress Enter to close...")

if __name__ == "__main__":
    main()