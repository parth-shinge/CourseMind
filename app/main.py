"""Streamlit entry point for CourseMind (Lecture-to-Notes + RAG Learning Assistant).

Provides a multi-tab interface:
- Tab 1: Video Ingest & Transcribe
- Tab 2: Structured Notes Viewer
- Tab 3: RAG Assistant Chat
"""

import streamlit as st

from app.config import settings
from app.ui.ingest_tab import render_ingest_tab
from app.ui.notes_tab import render_notes_tab
from app.ui.chat_tab import render_chat_tab


def main() -> None:
    st.set_page_config(
        page_title="CourseMind",
        page_icon="🎓",
        layout="wide",
    )

    settings.ensure_dirs()

    st.title("🎓 CourseMind")
    st.caption("Course-support tool: Video to Notes and Grounded RAG Assistant")

    tab_ingest, tab_notes, tab_chat = st.tabs(
        ["📥 Ingest", "📝 Notes", "💬 Chat"]
    )

    with tab_ingest:
        render_ingest_tab()

    with tab_notes:
        render_notes_tab()

    with tab_chat:
        render_chat_tab()


if __name__ == "__main__":
    main()
