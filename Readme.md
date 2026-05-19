# 📒 NoteBot — AI-Powered RAG Study Assistant

An intelligent study assistant that lets you **chat with your PDF notes** using a Retrieval-Augmented Generation (RAG) pipeline powered by **Google Gemini 2.5 Flash**.

Upload any PDF → NoteBot chunks it, embeds it, and lets you ask questions, generate summaries, and create flashcards — all grounded in your actual notes.

---

## ✨ GenAI Features Demonstrated

| Feature | Implementation | Why It Matters |
|---------|---------------|----------------|
| **RAG Pipeline** | Chunking → Embedding → Cosine Retrieval → Generation | Core pattern behind ChatGPT plugins, Perplexity, etc. |
| **Streaming Responses** | `generate_content_stream` — token-by-token output | Real-time UX, industry standard for LLM apps |
| **Structured Output** | `response_mime_type="application/json"` | Reliable, parseable AI responses (no regex hacks) |
| **Multi-Turn Chat** | `st.session_state` conversation history | Context-aware follow-up questions |
| **Batch Embeddings** | Single API call per 100 chunks | 10x fewer API calls vs one-at-a-time |
| **Source Attribution** | Retrieved chunks + similarity scores shown | Transparency & trust in AI answers |
| **Retry with Backoff** | Auto-retry on 429/503 with exponential delay | Production-grade error resilience |
| **Prompt Engineering** | System instructions + context injection | Grounded, hallucination-resistant answers |

---

## 🧩 App Tabs

### 💬 Chat — RAG Q&A with Streaming
Ask questions about your notes. Answers are streamed token-by-token and grounded in retrieved chunks with similarity scores.

### 📊 Summary — Structured Document Analysis
One-click AI summary returning structured JSON with: title, summary, key topics, difficulty level, and study tips.

### 📇 Flashcards — AI-Generated Study Cards
Auto-generate Q&A flashcards with difficulty tagging (easy/medium/hard) from your notes.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Google Gemini 2.5 Flash |
| **Embeddings** | Gemini Embedding 001 |
| **Frontend** | Streamlit |
| **Retrieval** | Cosine similarity (NumPy) |
| **Text Splitting** | LangChain `RecursiveCharacterTextSplitter` |
| **PDF Parsing** | PyPDF2 |

---

## 🚀 Setup & Run

```bash
# 1. Clone
git clone https://github.com/your-username/note-bot.git
cd note-bot

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# Edit .env → add your Gemini API key from https://aistudio.google.com/

# 5. Run
streamlit run app.py
```

---

## 📁 Project Structure

```
NoteBot/
├── app.py              # Main application (282 lines)
│                        # - RAG pipeline (embed → retrieve → generate)
│                        # - Streaming chat with memory
│                        # - Structured output for summary & flashcards
│                        # - Retry logic for API resilience
├── requirements.txt    # Clean dependencies (6 packages)
├── .env                # Your API key (git-ignored)
├── .env.example        # API key template
├── .gitignore          # Ignores .env, venv/, __pycache__/
└── README.md           # This file
```

---

## 📄 License

Academic / Demo purposes.
