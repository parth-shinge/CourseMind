# Lecture-to-Notes + RAG Learning Assistant

A course-support tool featuring two coupled pipelines sharing a single knowledge base:

1. **Video → Notes**: Converts lecture videos (uploaded files or YouTube links) into concise, structured, timestamp-traceable notes using a local speech-to-text model + LLM summarization.
2. **RAG Assistant**: Indexes those notes into a vector store and answers student questions grounded *only* in the course material, with lecture and timestamp citations.

The two pipelines share a knowledge base so the assistant's answers stay grounded in what was actually taught, and update automatically as new lectures are ingested.

---

## Prerequisites

- **Python 3.11+** (tested with 3.11.x)
- **API Key**: A Gemini API Key (default) or Anthropic API Key
- **FFmpeg** (required by faster-whisper for audio extraction): [install guide](https://ffmpeg.org/download.html)

---

## Setup

### 1. Clone and enter the repository

```bash
git clone <repo-url>
cd Lecture2RAG
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
# Windows PowerShell:
Copy-Item .env.example .env

# macOS / Linux:
cp .env.example .env
```

Edit `.env` and set your API key(s):

```env
LLM_PROVIDER=gemini          # or "anthropic"
GEMINI_API_KEY=your_key_here
ANTHROPIC_API_KEY=            # leave blank if using Gemini

# Optional model overrides (defaults shown):
# GEMINI_MODEL=gemini-3.5-flash
# ANTHROPIC_MODEL=claude-sonnet-4-6
# WHISPER_MODEL_SIZE=small
# WHISPER_DEVICE=cpu
```

> **Note on `GEMINI_MODEL`:** Gemini model names rotate and deprecate frequently. If the default model stops working, check available models with `client.models.list()` or visit the [Gemini API docs](https://ai.google.dev/gemini-api/docs/models). The `GEMINI_MODEL` env var lets you override the default without editing code.

### 5. Verify setup

```bash
python -c "from app.config import settings; settings.ensure_dirs(); print('Config OK! Provider:', settings.LLM_PROVIDER)"
```

---

## Running the App

### Streamlit UI (recommended)

```bash
streamlit run app/main.py
```

This opens a browser with three tabs:
- **📥 Ingest** — Upload a video or paste a YouTube URL to generate notes
- **📝 Notes** — Browse and review generated lecture notes
- **💬 Chat** — Ask questions and get grounded answers with citations

### CLI Modules

Each pipeline module is independently runnable:

```bash
# Download / validate a video
python -m app.pipeline.video_ingest "https://youtube.com/watch?v=..."

# Transcribe a local video/audio file
python -m app.pipeline.transcribe path/to/video.mp4

# Generate notes from a saved transcript
python -m app.pipeline.summarize <lecture_id> --title "Lecture Title"

# Chunk and index notes into the vector store
python -m app.rag.vector_store index <lecture_id>

# Query the vector store directly
python -m app.rag.vector_store query "What is acceleration?"

# Full RAG: retrieve + generate answer
python -m app.rag.generator "What is Newton's second law?"

# View pilot-study log statistics
python -m app.utils.log_summary
```

---

## Sample Data

The `data/samples/` directory contains a small synthetic demo (3 transcript segments, 2 note sections) so you can explore the Notes tab and test retrieval without running the full ingestion pipeline.

To load the sample into the active data directories:

```bash
# Windows PowerShell:
Copy-Item data\samples\sample_transcript.json data\transcripts\sample_demo.json
Copy-Item data\samples\sample_notes.json data\notes\sample_demo.json
Copy-Item data\samples\sample_notes.md data\notes\sample_demo.md

# macOS / Linux:
cp data/samples/sample_transcript.json data/transcripts/sample_demo.json
cp data/samples/sample_notes.json data/notes/sample_demo.json
cp data/samples/sample_notes.md data/notes/sample_demo.md
```

Then index the sample for Chat tab retrieval:

```bash
python -m app.rag.vector_store index sample_demo
```

> **Note:** The Chat tab's answer generation still requires a valid API key, but the Notes tab and vector store retrieval work immediately with the sample data.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests are a mix of:
- **Unit tests** with mocked LLM calls (no API key needed)
- **Integration tests** using real local models (sentence-transformers, ChromaDB)

---

## Project Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for the complete architecture specification, module contracts, data formats, and build roadmap.

```
Lecture2RAG/
├── ARCHITECTURE.md        # Single source of truth for contracts
├── README.md              # This file
├── requirements.txt
├── .env.example           # Template for .env (real .env is gitignored)
├── app/
│   ├── main.py            # Streamlit entry point
│   ├── config.py          # Settings from .env
│   ├── llm_client.py      # Pluggable LLM wrapper (Gemini / Anthropic)
│   ├── pipeline/          # Video → Notes pipeline
│   │   ├── video_ingest.py
│   │   ├── transcribe.py
│   │   ├── summarize.py
│   │   └── notes_store.py
│   ├── rag/               # RAG pipeline
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   └── generator.py
│   ├── ui/                # Streamlit tab components
│   └── utils/             # Logging and helpers
├── data/
│   ├── samples/           # Checked-in demo data
│   ├── transcripts/       # Generated transcript JSONs
│   ├── notes/             # Generated notes (md + json)
│   ├── vector_db/         # ChromaDB persistent storage
│   └── logs/              # Pilot-study query/ingest logs
└── tests/
```

---

## Known Limitations

| Area | Limitation |
|------|-----------|
| **Gemini free-tier quota** | The free tier has rate limits (~15 RPM, ~1M TPM). Long lectures with many sections may hit quota limits during summarization. The LLM client retries on 429/503 errors with exponential backoff, but sustained high usage may require a paid API key. |
| **Thinking-token budget** | Gemini 3.x "thinking" models (e.g., `gemini-3.5-flash`) use `max_output_tokens` for *both* thinking and visible output. This means a section that needs 2000 tokens of output may require a 4000+ token budget. The summarizer handles this automatically with retry-and-double logic, but very long sections may still truncate. |
| **English-only transcription** | faster-whisper auto-detects the spoken language, but the summarization prompts and notes format are English-oriented. Non-English lectures will transcribe but may produce lower-quality notes. |
| **Single-course scope** | Designed as a single-course pilot. All lectures share one ChromaDB collection. Multi-course support would require collection-per-course isolation. |
| **Local embedding model** | Uses `all-MiniLM-L6-v2` (384-dim). Sufficient for course-scale data (~hundreds of chunks) but not optimized for very large corpora. |
| **No authentication** | The Streamlit UI has no login or access control. Intended for local/pilot use only. |
