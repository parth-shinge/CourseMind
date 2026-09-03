"""Streamlit UI tab: Grounded RAG Assistant Chat.

Implementation in Phase 5.  Logging added in Phase 6.
"""

import time
import streamlit as st

from app.rag.generator import answer
from app.rag.vector_store import query as vs_query
from app.rag.retriever import RELEVANCE_THRESHOLD
from app.utils.logger import log_query

def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS for display."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_chat_tab() -> None:
    """Render the grounded RAG chat assistant tab."""
    st.header("💬 Course Assistant")
    st.write("Ask questions about the course material and view timestamp citations.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        section_title = src.get("section_title", "Unknown")
                        lecture_id = src.get("lecture_id", "Unknown")
                        timestamp = _format_timestamp(src.get("timestamp_start", 0))
                        st.markdown(f"- 📖 {section_title} — Lecture: {lecture_id} (at {timestamp})")

    if prompt := st.chat_input("Ask a question about the course..."):
        # User message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching for answers..."):
                try:
                    t0 = time.perf_counter()

                    # Retrieve: raw query → relevance filter
                    raw_chunks = vs_query(prompt, k=5)
                    chunks = [
                        (chunk, score)
                        for chunk, score in raw_chunks
                        if score <= RELEVANCE_THRESHOLD
                    ]
                    filtered_out_count = len(raw_chunks) - len(chunks)

                    # Generate answer
                    response = answer(prompt, chunks)
                    latency = time.perf_counter() - t0

                    # used_llm is False when generator short-circuits (no chunks)
                    used_llm = bool(chunks)

                    # Log — silent-fail internally, never crashes the UI
                    log_query(
                        question=prompt,
                        retrieved_chunks=chunks,
                        filtered_out_count=filtered_out_count,
                        answer=response["answer"],
                        sources=response.get("sources", []),
                        latency_seconds=latency,
                        used_llm=used_llm,
                    )

                    st.markdown(response["answer"])

                    if response.get("sources"):
                        with st.expander("Sources"):
                            for src in response["sources"]:
                                section_title = src.get("section_title", "Unknown")
                                lecture_id = src.get("lecture_id", "Unknown")
                                timestamp = _format_timestamp(src.get("timestamp_start", 0))
                                st.markdown(f"- 📖 {section_title} — Lecture: {lecture_id} (at {timestamp})")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", []),
                    })
                except Exception as e:
                    st.error(f"Error processing your question: {e}")
