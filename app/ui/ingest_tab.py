"""Streamlit UI tab: Video Ingestion & Processing.

Implementation in Phase 5.  Logging added in Phase 6.
"""

import time
import traceback
import streamlit as st
from datetime import date

from app.config import settings
from app.utils import sanitize_lecture_id
from app.pipeline.video_ingest import download_video
from app.pipeline.transcribe import transcribe
from app.pipeline.summarize import generate_notes
from app.pipeline.notes_store import save_notes
from app.rag.chunker import chunk_notes
from app.rag.vector_store import upsert
from app.utils.logger import log_ingestion


def render_ingest_tab() -> None:
    """Render the video ingestion tab."""
    st.header("📥 Video Ingestion & Transcription")
    st.write("Upload a video or enter a YouTube/Drive link to generate notes.")

    # ── Input mode ──────────────────────────────────────────────────────────
    input_mode = st.radio("Input Method", ["URL / Local Path", "File Upload"])

    file_upload = None
    url_or_path = ""

    if input_mode == "File Upload":
        file_upload = st.file_uploader(
            "Upload video file", type=["mp4", "webm", "mp3", "wav", "m4a"]
        )
    else:
        url_or_path = st.text_input("Enter Video URL or Local Path")

    title = st.text_input("Lecture Title (Required)")
    source_url = st.text_input("Source URL (Optional)")
    lecture_date = st.date_input("Date (Optional)", value=date.today())

    if "confirm_ingest" not in st.session_state:
        st.session_state.confirm_ingest = False

    def _click_confirm() -> None:
        st.session_state.confirm_ingest = True

    # ── Primary button ───────────────────────────────────────────────────────
    if st.button("Process Video"):
        if not title:
            st.error("Lecture Title is required.")
            return
        if input_mode == "File Upload" and not file_upload:
            st.error("Please upload a file.")
            return
        if input_mode == "URL / Local Path" and not url_or_path:
            st.error("Please provide a URL or local path.")
            return

        lecture_id = sanitize_lecture_id(title)

        # Warn if already processed
        expected_json = settings.NOTES_DIR / f"{lecture_id}.json"
        if expected_json.exists():
            st.warning(
                f"Lecture notes for '{lecture_id}' already exist. "
                "Processing again will overwrite."
            )

        st.warning("⚠️ This will use your daily LLM quota. Continue?")
        st.button("Confirm", on_click=_click_confirm)

    # ── Confirmed pipeline run ───────────────────────────────────────────────
    if st.session_state.confirm_ingest:
        lecture_id = sanitize_lecture_id(title)
        st.session_state.confirm_ingest = False  # reset so re-runs don't re-trigger

        source_label = url_or_path or (file_upload.name if file_upload else "uploaded_file")

        meta = {
            "lecture_id": lecture_id,
            "title": title,
            "source": source_url or source_label,
            "date": lecture_date.isoformat(),
        }

        stage_durations: dict = {}
        ingest_success = False
        ingest_error: str | None = None

        with st.status("Processing Lecture...", expanded=True) as status:
            try:
                # ── Stage 1: Download ────────────────────────────────────
                st.write("📥 Downloading / verifying video...")
                t0 = time.perf_counter()
                if input_mode == "File Upload" and file_upload is not None:
                    target_dir = settings.RAW_VIDEOS_DIR
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_path = target_dir / file_upload.name
                    with open(target_path, "wb") as f:
                        f.write(file_upload.getbuffer())
                    video_path = str(target_path)
                else:
                    video_path = download_video(url_or_path)
                stage_durations["download"] = time.perf_counter() - t0

                # ── Stage 2: Transcribe ──────────────────────────────────
                st.write("🎙️ Transcribing audio (this may take a while)...")
                t0 = time.perf_counter()
                segments = transcribe(video_path)
                stage_durations["transcribe"] = time.perf_counter() - t0

                # ── Stage 3: Summarise ───────────────────────────────────
                st.write("📝 Generating structured notes via LLM...")
                t0 = time.perf_counter()
                notes = generate_notes(segments, meta)
                stage_durations["summarize"] = time.perf_counter() - t0

                # ── Stage 4: Save ────────────────────────────────────────
                st.write("💾 Saving notes to disk...")
                save_notes(lecture_id, notes["sections"], notes["markdown"], notes["meta"])

                # ── Stage 5: Chunk + Index ───────────────────────────────
                st.write("🔍 Chunking & indexing into vector store...")
                t0 = time.perf_counter()
                chunks = chunk_notes(lecture_id, notes["sections"])
                upsert(chunks)
                stage_durations["index"] = time.perf_counter() - t0

                ingest_success = True
                status.update(
                    label="✅ Lecture processed successfully!",
                    state="complete",
                    expanded=False,
                )
                st.success(f"Done! Lecture ID: **{lecture_id}**")
                with st.expander("Preview Notes"):
                    st.markdown(notes["markdown"])

            except Exception as e:
                ingest_error = traceback.format_exc()
                status.update(label="❌ Processing failed", state="error", expanded=True)
                st.error(f"An error occurred: {e}")

            finally:
                # Always log — even on partial failure so timing data is preserved
                log_ingestion(
                    lecture_id=lecture_id,
                    source=source_label,
                    stage_durations=stage_durations,
                    success=ingest_success,
                    error=ingest_error,
                )
