# RAG with Gemini and ChromaDB

A simple Retrieval-Augmented Generation (RAG) script that:
- Loads local text files from `data/news_articles/`.
- Splits them into chunks.
- Generates embeddings with Gemini.
- Stores them in a persistent ChromaDB collection.
- Retrieves relevant chunks and answers a question with Gemini.

## Project Structure

- `app.py` - Main script (ingest, embed, store, query, generate).
- `news_articles/` - Place your `.txt` files here.
- `chroma_persistent_storage/` - Persistent ChromaDB storage.
- `.env` - Environment variables (API key).
- `venv/` - Local virtual environment (optional).

## Prerequisites

- Python 3.10+ recommended
- A Gemini API key

## Setup

1) Create and activate a virtual environment (optional)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install google-genai chromadb python-dotenv
```

3) Set your API key

Create a `.env` file in the project root:

```env
API_KEY=YOUR_GEMINI_API_KEY
```

## Run

```powershell
python app.py
```

## How It Works

1) **Load documents**: reads all `.txt` files in `news_articles/`.
2) **Chunking**: splits each document into 1000-character chunks with 20-character overlap.
3) **Embedding**: uses `gemini-embedding-001` to create vectors for each chunk.
4) **Upsert**: stores chunk text and embeddings in ChromaDB.
5) **Query**: retrieves the most relevant chunks for a question.
6) **Generate**: sends retrieved context to `gemini-2.5-flash` to answer.

## Configuration

- Change the collection name in `app.py`:
  - `collection_name = "document_qa_collection"`
- Adjust chunk sizes in `split_text(text, chunk_size, chunk_overlap)`.
- Change the question in the `__main__` block.

## Notes

- The script currently **re-embeds and upserts** every run. For larger datasets, consider checking whether chunks already exist before re-embedding.
- ChromaDB persists data under `chroma_persistent_storage/`.

## Troubleshooting

- **"environment variable must be set"**: Make sure `.env` has `API_KEY` and `load_dotenv()` is called before client creation.
- **Validation errors in generate_content**: The SDK expects a string for `contents`, not role-based dicts.
- **Empty results**: Ensure there are `.txt` files inside `news_articles/`.


