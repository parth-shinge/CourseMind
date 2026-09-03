# Sample Data

This directory contains a small synthetic demo dataset for the Lecture-to-Notes + RAG Assistant.

**Contents:**
- `sample_transcript.json` — 3 transcript segments about Newton's Laws of Motion
- `sample_notes.json` — Structured notes metadata (2 sections) matching ARCHITECTURE.md Section 6
- `sample_notes.md` — Human-readable markdown rendering of the notes

**Purpose:** Lets anyone clone the repo and immediately try the Notes tab and retrieval without running the full ingestion pipeline or needing an API key for transcription/summarization.

**Usage:**

1. Copy the sample files into the active data directories:

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

2. Index the sample for Chat tab retrieval:

   ```bash
   python -m app.rag.vector_store index sample_demo
   ```

3. Launch the Streamlit app and explore:

   ```bash
   streamlit run app/main.py
   ```

> **Note:** The Chat tab's answer generation requires a valid LLM API key. The Notes tab works immediately with the copied sample data.

**This data is synthetic** — it was written by hand as a minimal demo and is not derived from any copyrighted lecture recording.
