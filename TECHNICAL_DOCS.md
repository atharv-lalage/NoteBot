# 📒 NoteBot — Technical Documentation

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How RAG Works (Step by Step)](#how-rag-works-step-by-step)
- [Code Walkthrough](#code-walkthrough)
- [GenAI Concepts Used](#genai-concepts-used)
- [Interview Q&A](#interview-qa)

---

## Overview

**NoteBot** is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF notes and interact with them through three AI-powered features:

1. **Chat** — Ask questions, get answers grounded in your notes
2. **Summary** — Auto-analyze the document structure and difficulty
3. **Flashcards** — Generate study cards with difficulty tagging

The app is built with **Streamlit** (frontend) and **Google Gemini 2.5 Flash** (LLM + Embeddings).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER                                 │
│                   Uploads PDF + Asks Questions               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   STREAMLIT FRONTEND                         │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐          │
│  │ 💬 Chat  │  │ 📊 Summary   │  │ 📇 Flashcards │          │
│  └────┬─────┘  └──────┬───────┘  └───────┬───────┘          │
└───────┼───────────────┼──────────────────┼──────────────────┘
        │               │                  │
        ▼               ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE                             │
│                                                              │
│  1. PDF Text Extraction (PyPDF2)                             │
│  2. Chunking (RecursiveCharacterTextSplitter)                │
│  3. Embedding (Gemini Embedding 001)                         │
│  4. Storage (NumPy arrays in session state)                  │
│  5. Retrieval (Cosine similarity, top-4)                     │
│  6. Generation (Gemini 2.5 Flash with context)               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
        │                                    │
        ▼                                    ▼
┌──────────────────┐              ┌──────────────────────┐
│  Gemini Embedding │              │  Gemini 2.5 Flash    │
│  001 API          │              │  (Generation API)    │
│  - Batch embed    │              │  - Streaming         │
│  - Task types     │              │  - JSON mode         │
└──────────────────┘              └──────────────────────┘
```

---

## How RAG Works (Step by Step)

RAG (Retrieval-Augmented Generation) is the technique behind tools like ChatGPT with file uploads, Perplexity AI, and Notion AI. Here's how NoteBot implements it:

### Step 1: Document Ingestion
```
PDF File → PyPDF2 → Raw Text String
```
We extract all text from every page of the uploaded PDF using `PyPDF2.PdfReader`.

### Step 2: Chunking
```
Raw Text → RecursiveCharacterTextSplitter → List of Chunks
```
We split the text into **800-character chunks** with **200-character overlap**. Why?
- LLMs have context limits — we can't send the entire document
- Overlap ensures sentences at chunk boundaries aren't cut in half
- The splitter tries to break at natural points: `\n\n` → `\n` → `. ` → ` `

### Step 3: Embedding
```
List of Chunks → Gemini Embedding API → NumPy Array of Vectors
```
Each chunk is converted into a **768-dimensional vector** (a list of numbers) that captures its **semantic meaning**. Similar concepts will have similar vectors.

We use **batch embedding** — sending up to 100 chunks per API call instead of one at a time. This is 100x more efficient.

### Step 4: Retrieval
```
User Query → Embed Query → Cosine Similarity → Top-4 Chunks
```
When the user asks a question:
1. The query is embedded using the same model (but with `task_type="RETRIEVAL_QUERY"`)
2. We compute **cosine similarity** between the query vector and all chunk vectors
3. The top 4 most similar chunks are returned as "context"

**Cosine similarity formula:**
```
similarity = dot(A, B) / (||A|| × ||B||)
```
Score ranges from -1 (opposite) to 1 (identical).

### Step 5: Generation
```
System Prompt + Context Chunks + Chat History + Question → Gemini 2.5 Flash → Answer
```
The retrieved chunks are injected into a prompt with system instructions telling the model to only answer from the given context. This **grounds** the response and prevents hallucination.

### Step 6: Streaming
```
Response tokens → Streamed one-by-one to the UI → Real-time display
```
Instead of waiting for the full response, we stream tokens using `generate_content_stream`, displaying each token as it arrives.

---

## Code Walkthrough

### File: `app.py` (282 lines)

| Line Range | Section | What It Does |
|-----------|---------|-------------|
| 1–11 | **Imports** | Core libraries: Streamlit, Gemini SDK, NumPy, PyPDF2, LangChain |
| 13–29 | **Config & Init** | Loads API key, initializes Gemini client, sets up page |
| 31–39 | **Session State** | Initializes `messages`, `chunks`, `embeddings`, `ready` flags |
| 41–54 | **Retry Helper** | `safe_api_call()` — retries API calls 3 times with backoff on 429/503 |
| 56–80 | **Embedding & Retrieval** | `embed_texts()` for batch embedding, `retrieve()` for cosine search |
| 82–93 | **PDF Processing** | `process_pdf()` — extract text and split into chunks |
| 95–116 | **Sidebar** | File upload, embedding progress, reset button |
| 118–187 | **Chat Tab** | Multi-turn RAG chat with streaming and source attribution |
| 189–237 | **Summary Tab** | Structured JSON output for document analysis |
| 239–282 | **Flashcards Tab** | Structured JSON output for Q&A card generation |

### Key Design Decisions

| Decision | Why |
|----------|-----|
| **NumPy for vector store** (not ChromaDB/FAISS) | Shows understanding of how similarity search actually works under the hood |
| **Session state for storage** | No external database needed, keeps it simple for a study tool |
| **Batch embedding** | Shows API optimization knowledge (100 chunks per call vs 1) |
| **Separate task types** | `RETRIEVAL_DOCUMENT` for chunks, `RETRIEVAL_QUERY` for queries — Gemini optimizes embeddings differently |
| **200-char overlap** | Prevents information loss at chunk boundaries |
| **Top-4 retrieval** | Balances context quality vs. token usage |

---

## GenAI Concepts Used

### 1. RAG (Retrieval-Augmented Generation)
Instead of relying on the LLM's training data, we **retrieve relevant context** from the user's own documents and inject it into the prompt. This eliminates hallucination for document-specific questions.

### 2. Embeddings
Text is converted into dense numerical vectors that capture **semantic meaning**. "What is photosynthesis?" and "How do plants make food?" would have similar embeddings even though they share few words.

### 3. Cosine Similarity
A mathematical measure of how similar two vectors are, regardless of their magnitude. We use it to find which document chunks are most relevant to the user's question.

### 4. Streaming
The `generate_content_stream` API returns response tokens one at a time, allowing real-time display. This improves perceived latency — users see the answer being "typed" instead of waiting for the full response.

### 5. Structured Output (JSON Mode)
By setting `response_mime_type="application/json"`, we force the LLM to return valid JSON. This is critical for features like Summary and Flashcards where we need to parse the response programmatically.

### 6. Prompt Engineering
The system prompt constrains the model to only answer from provided context. This includes:
- **System instructions** (role definition, rules)
- **Context injection** (retrieved chunks)
- **Conversation history** (last 3 turns for multi-turn awareness)

### 7. Retry with Exponential Backoff
The `safe_api_call` function catches 429 (rate limit) and 503 (server busy) errors, waits progressively longer (3s, 6s, 9s), and retries. This is a production-grade pattern used in all real API integrations.

---

## Interview Q&A

### Q1: What is RAG and why did you use it?
**Answer:** RAG stands for Retrieval-Augmented Generation. Instead of asking an LLM to answer from its training data (which may be outdated or hallucinate), I retrieve relevant chunks from the user's uploaded PDF and inject them into the prompt as context. This grounds the response in actual data. I used it because NoteBot needs to answer questions about *specific* user-uploaded notes, not general knowledge.

### Q2: How does the embedding process work in your project?
**Answer:** I use Google's `gemini-embedding-001` model to convert each text chunk into a 768-dimensional vector. These vectors capture semantic meaning — so similar concepts get similar vectors even if they use different words. I batch-embed up to 100 chunks per API call for efficiency, and store the resulting vectors as a NumPy array in Streamlit's session state.

### Q3: Why did you choose cosine similarity over other distance metrics?
**Answer:** Cosine similarity measures the angle between two vectors, not their magnitude. This is important because embedding vectors can have varying lengths depending on the text. Cosine similarity normalizes for this, giving a score between -1 and 1 where 1 means identical direction. It's the industry standard for text embeddings and is computationally efficient with NumPy's dot product.

### Q4: What is structured output and where did you use it?
**Answer:** Structured output forces the LLM to return valid JSON by setting `response_mime_type="application/json"` in the API config. I used it in two places: the Summary tab (returns title, topics, difficulty, tips as JSON) and the Flashcards tab (returns an array of Q&A objects). This is much more reliable than generating free text and trying to parse it with regex.

### Q5: How does streaming work in your chat feature?
**Answer:** Instead of calling `generate_content()` which waits for the full response, I use `generate_content_stream()` which returns an iterator of response chunks. I loop through these chunks, appending each token to a Streamlit placeholder, creating a real-time typing effect. This reduces perceived latency from seconds to milliseconds for the first token.

### Q6: What is chunking and why is overlap important?
**Answer:** Chunking splits a large document into smaller pieces (800 characters in my case) so each piece fits within the LLM's context window. Overlap (200 characters) is critical because without it, a sentence at the boundary of two chunks would be split in half, losing its meaning. The 200-char overlap ensures continuity. I use LangChain's `RecursiveCharacterTextSplitter` which tries to split at natural boundaries like paragraph breaks first.

### Q7: How do you handle API rate limits and errors?
**Answer:** I implemented a `safe_api_call` wrapper function that catches `429 RESOURCE_EXHAUSTED` and `503 UNAVAILABLE` errors from the Gemini API. It retries up to 3 times with progressive backoff (3s, 6s, 9s delays). If all retries fail, it raises the error to the UI which shows a user-friendly error message instead of crashing. This is a standard production pattern called exponential backoff.

### Q8: Why didn't you use a vector database like ChromaDB or Pinecone?
**Answer:** For a study assistant handling single PDFs at a time, an in-memory NumPy array is simpler and faster. I implemented cosine similarity manually to demonstrate understanding of *how* vector search works under the hood, rather than abstracting it away behind a library. For a production system with millions of documents, I would use ChromaDB or FAISS for persistence and faster approximate nearest neighbor search.

### Q9: What is the difference between RETRIEVAL_DOCUMENT and RETRIEVAL_QUERY task types?
**Answer:** Gemini's embedding API optimizes vectors differently based on the task type. `RETRIEVAL_DOCUMENT` is used when embedding the document chunks — it optimizes for being *found*. `RETRIEVAL_QUERY` is used when embedding the user's question — it optimizes for *finding*. Using the correct task types improves retrieval accuracy compared to using the same type for both.

### Q10: How do you maintain conversation context across multiple questions?
**Answer:** I use Streamlit's `st.session_state.messages` to store the full conversation history (role, content, sources, scores). When building the prompt for a new question, I include the last 6 messages (3 user-assistant pairs) as conversation history. This allows the model to understand follow-up questions like "explain that in more detail" or "what about the second point?"

### Q11: What would you improve if you had more time?
**Answer:** Three things:
1. **Hybrid search** — combine BM25 keyword search with semantic search for better retrieval quality, especially for exact terms
2. **Persistent storage** — use ChromaDB so users don't re-embed PDFs every session
3. **RAG evaluation** — add metrics like faithfulness and relevancy scoring using frameworks like RAGAS to measure answer quality

### Q12: Why Gemini 2.5 Flash specifically?
**Answer:** Gemini 2.5 Flash is Google's latest stable model (May 2026) optimized for low-latency, high-volume tasks. It's the best price-performance model in the Gemini family — fast enough for streaming responses, smart enough for structured JSON output, and has a 1M token context window. The Flash variant is ideal for a study assistant where speed matters more than maximum reasoning depth.

---

## Technologies & Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.35.0 | Web UI framework |
| `google-genai` | ≥1.50.0 | Gemini LLM & Embedding API |
| `langchain-text-splitters` | ≥1.0.0 | Document chunking |
| `PyPDF2` | ≥3.0.0 | PDF text extraction |
| `numpy` | ≥2.0.0 | Vector math & cosine similarity |
| `python-dotenv` | ≥1.0.0 | Environment variable loading |
