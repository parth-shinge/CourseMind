"""Streamlit UI tab: Structured Notes Viewer.

Implementation in Phase 5.
"""

import json
import streamlit as st
from pathlib import Path

from app.config import settings
from app.pipeline.notes_store import load_notes


def render_notes_tab() -> None:
    """Render the structured notes viewer tab."""
    st.header("📝 Lecture Notes")
    st.write("Browse and review generated notes by lecture topic and timestamp.")

    notes_dir = settings.NOTES_DIR
    if not notes_dir.exists():
        st.info("No notes found. Go to the Ingest tab to process a video.")
        return

    json_files = list(notes_dir.glob("*.json"))
    if not json_files:
        st.info("No notes found. Go to the Ingest tab to process a video.")
        return

    # Load metadata for the selectbox
    lectures = {}
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                meta = json.load(f)
                title = meta.get("title", jf.stem)
                date_str = meta.get("date", "")
                display_name = f"{title} ({date_str})" if date_str else title
                lectures[display_name] = meta.get("lecture_id", jf.stem)
        except Exception:
            pass

    if not lectures:
        st.info("No valid notes metadata found.")
        return

    selected_display = st.selectbox("Select a lecture", list(lectures.keys()))
    if selected_display:
        lecture_id = lectures[selected_display]
        try:
            markdown_content, _ = load_notes(lecture_id)
            with st.container(border=True):
                st.markdown(markdown_content)
        except Exception as e:
            st.error(f"Error loading notes for {lecture_id}: {e}")
