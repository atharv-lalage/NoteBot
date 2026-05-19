import os
import json
import time
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ─────────────────────────── Config ───────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHAT_MODEL = "gemini-2.5-flash"            # Latest stable model (May 2026)
EMBED_MODEL = "gemini-embedding-001"      # Gemini native embeddings
MAX_RETRIES = 3

st.set_page_config(page_title="NoteBot", page_icon="📒", layout="wide")
st.title("📒 NoteBot")
st.caption("RAG-powered study assistant · Gemini 2.5 Flash · Streaming · Structured Output")

if not GEMINI_API_KEY:
    st.error("⚠️ Set `GEMINI_API_KEY` in your `.env` file")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# ─────────────────────── Session State ────────────────────────
for key, default in {
    "messages": [],
    "chunks": [],
    "embeddings": None,
    "ready": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────── Safe API Call Helper ─────────────────────

def safe_api_call(fn, *args, **kwargs):
    """Retry API calls on 429/503 errors with backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except (ClientError, ServerError) as e:
            if attempt < MAX_RETRIES - 1:
                wait = (attempt + 1) * 3
                st.warning(f"⏳ API busy, retrying in {wait}s... (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise e

# ─────────────────────── Embedding Utils ──────────────────────

def embed_texts(texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """Batch embed using Gemini Embedding API (batches of 100)."""
    all_embs = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        result = safe_api_call(
            client.models.embed_content,
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task),
        )
        all_embs.extend([e.values for e in result.embeddings])
    return np.array(all_embs)


def retrieve(query: str, k: int = 4) -> tuple[list[str], list[float]]:
    """Top-k retrieval via cosine similarity."""
    q_emb = embed_texts([query], task="RETRIEVAL_QUERY")[0]
    embs = st.session_state.embeddings
    norms = np.linalg.norm(embs, axis=1) * np.linalg.norm(q_emb)
    scores = embs @ q_emb / np.where(norms == 0, 1, norms)
    top = np.argsort(scores)[::-1][:k]
    return [st.session_state.chunks[i] for i in top], [float(scores[i]) for i in top]

# ─────────────────────── PDF Processing ───────────────────────

def process_pdf(file) -> list[str]:
    reader = PdfReader(file)
    text = "".join(p.extract_text() or "" for p in reader.pages)
    if not text.strip():
        raise ValueError("No text could be extracted from this PDF")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)

# ─────────────────────────── Sidebar ──────────────────────────
with st.sidebar:
    st.header("📂 Upload Notes")
    file = st.file_uploader("Choose a PDF", type="pdf")

    if file and not st.session_state.ready:
        with st.spinner("Extracting & embedding..."):
            try:
                chunks = process_pdf(file)
                embeddings = embed_texts(chunks)
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.ready = True
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.ready:
        st.success(f"✅ {len(st.session_state.chunks)} chunks indexed")
        if st.button("🗑️ Reset"):
            st.session_state.clear()
            st.rerun()

# ─────────────────────────── Tabs ─────────────────────────────
tab_chat, tab_summary, tab_flash = st.tabs(["💬 Chat", "📊 Summary", "📇 Flashcards"])

# ═══════════════════ TAB 1: RAG Chat + Streaming ══════════════
with tab_chat:
    if not st.session_state.ready:
        st.info("👈 Upload a PDF to start chatting with your notes")
    else:
        # Render history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📄 Sources"):
                        for i, (s, sc) in enumerate(zip(msg["sources"], msg["scores"])):
                            st.caption(f"**[{sc:.2f}]** {s[:250]}...")

        # Chat input
        if query := st.chat_input("Ask about your notes..."):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # Retrieve relevant chunks
            sources, scores = retrieve(query, k=4)
            context = "\n\n---\n\n".join(sources)

            # Build prompt with system instruction + conversation history
            history = "\n".join(
                f"{m['role'].upper()}: {m['content']}"
                for m in st.session_state.messages[-6:]
            )

            prompt = f"""You are NoteBot, an AI study assistant. Answer based ONLY on the provided context.
If the answer isn't in the context, say "I couldn't find this in your notes."
Be concise. Use bullet points for lists. Reference specific parts when possible.

CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

QUESTION: {query}"""

            # ✨ Streaming response (token-by-token)
            with st.chat_message("assistant"):
                try:
                    placeholder = st.empty()
                    full_response = ""
                    for chunk in client.models.generate_content_stream(
                        model=CHAT_MODEL, contents=prompt
                    ):
                        if chunk.text:
                            full_response += chunk.text
                            placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)

                    with st.expander("📄 Sources"):
                        for i, (s, sc) in enumerate(zip(sources, scores)):
                            st.caption(f"**[{sc:.2f}]** {s[:250]}...")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources,
                        "scores": scores,
                    })
                except (ClientError, ServerError) as e:
                    st.error(f"⚠️ API error: {e}. Please wait a moment and try again.")

# ═══════════════ TAB 2: Summary (Structured Output) ═══════════
with tab_summary:
    if not st.session_state.ready:
        st.info("👈 Upload a PDF to generate a summary")
    else:
        if st.button("🧠 Generate Summary"):
            with st.spinner("Analyzing..."):
                sample = "\n\n".join(st.session_state.chunks[:20])
                prompt = f"""Analyze these notes and return a JSON object:
{{
  "title": "detected title/subject",
  "summary": "comprehensive 3-4 sentence summary",
  "key_topics": ["topic1", "topic2"],
  "difficulty": "Beginner | Intermediate | Advanced",
  "study_tips": ["tip1", "tip2", "tip3"]
}}

NOTES:
{sample}"""

                # ✨ Structured Output — Gemini JSON mode
                try:
                    resp = safe_api_call(
                        client.models.generate_content,
                        model=CHAT_MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                        ),
                    )
                    data = json.loads(resp.text)
                    st.subheader(data.get("title", "Summary"))
                    st.write(data.get("summary", ""))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🏷️ Key Topics**")
                        for t in data.get("key_topics", []):
                            st.markdown(f"- {t}")
                    with col2:
                        st.markdown(f"**📈 Difficulty:** {data.get('difficulty', 'N/A')}")
                        st.markdown("**💡 Study Tips**")
                        for t in data.get("study_tips", []):
                            st.markdown(f"- {t}")
                except (ClientError, ServerError) as e:
                    st.error(f"⚠️ API error: {e}. Please wait and try again.")
                except json.JSONDecodeError:
                    st.error("Failed to parse structured output")
                    st.code(resp.text)

# ═══════════════ TAB 3: Flashcards (Structured Output) ════════
with tab_flash:
    if not st.session_state.ready:
        st.info("👈 Upload a PDF to generate flashcards")
    else:
        n = st.slider("Number of flashcards", 3, 15, 5)
        if st.button("🎴 Generate Flashcards"):
            with st.spinner("Creating flashcards..."):
                sample = "\n\n".join(st.session_state.chunks[:15])
                prompt = f"""Generate exactly {n} flashcards from these notes as JSON:
{{
  "flashcards": [
    {{"question": "...", "answer": "...", "difficulty": "easy|medium|hard"}}
  ]
}}

NOTES:
{sample}"""

                # ✨ Structured Output — Gemini JSON mode
                try:
                    resp = safe_api_call(
                        client.models.generate_content,
                        model=CHAT_MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                        ),
                    )
                    parsed = json.loads(resp.text)
                    cards = parsed.get("flashcards", parsed) if isinstance(parsed, dict) else parsed
                    for i, c in enumerate(cards):
                        icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(
                            c.get("difficulty", ""), "⚪"
                        )
                        with st.expander(f"{icon} Card {i+1}: {c['question'][:60]}"):
                            st.markdown(f"**Q:** {c['question']}")
                            st.divider()
                            st.markdown(f"**A:** {c['answer']}")
                except (ClientError, ServerError) as e:
                    st.error(f"⚠️ API error: {e}. Please wait and try again.")
                except json.JSONDecodeError:
                    st.error("Failed to parse structured output")
                    st.code(resp.text)