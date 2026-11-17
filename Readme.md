# 📒 NoteBot — AI-Powered RAG Study Assistant

An intelligent study assistant that lets you **chat with your PDF notes** using a Retrieval-Augmented Generation (RAG) pipeline powered by **Google Gemini 3.5 Flash Lite**.

Upload any PDF → NoteBot chunks it, embeds it, and lets you ask questions, generate summaries, and create flashcards — all grounded in your actual notes.

👉 **[Read the Technical Documentation & FAQ](TECHNICAL_DOCS.md)** for architecture diagrams, RAG step-by-step pipeline specs, and technical design choices.

👉 **[Read the Week 1 Requirements and Design](docs/requirements-and-design.md)** for the problem statement, scope, requirements, diagrams, schema, and evaluation plan.

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
| **LLM** | Google Gemini 3.5 Flash Lite |
| **Embeddings** | Gemini Embedding 001 |
| **Frontend** | React + Vite |
| **Backend API** | FastAPI |
| **Retrieval** | PostgreSQL pgvector cosine search |
| **Text Splitting** | LangChain `RecursiveCharacterTextSplitter` |
| **PDF Parsing** | PyMuPDF (page-aware extraction) |
| **Persistence** | PostgreSQL + pgvector + SQLAlchemy |

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
# Optional: set CHAT_MODEL=gemini-3.5-flash-lite

# 5. Run the API
uvicorn app:app --reload

# 6. Run the React frontend in another terminal
cd frontend
npm install
npm run dev
```

### Persistent local database and full stack

The application includes PostgreSQL with pgvector through Docker Compose:

```bash
docker compose up --build
```

This persists documents, pages, chunks, embeddings, conversations, messages,
and source links in the `postgres_data` volume. To run migrations manually:

```bash
docker compose up -d db
alembic upgrade head
```

Retrieval uses pgvector cosine search with candidate over-retrieval, a
configurable similarity threshold, document filtering, neighboring chunks, and
page/chunk citation metadata.

Chat answers use versioned grounded prompts, conversation-aware query rewriting,
strict refusal behavior, and application-generated page citations. Unsupported
questions return: `I could not find enough information in your uploaded notes to
answer this question.`

---

## 📁 Project Structure

```
NoteBot/
├── app.py              # FastAPI entry point
├── app/
│   ├── config.py       # Centralized settings
│   ├── ingestion/      # PDF loading and chunking
│   ├── retrieval/      # Embeddings, vector store, retrieval
│   ├── generation/     # Gemini client, prompts, answer generation
│   ├── services/       # Chat, summary, flashcard, and quiz interfaces
│   └── api.py           # FastAPI backend endpoints
├── frontend/             # React + Vite frontend
├── tests/               # Automated tests
├── evaluation/          # Evaluation assets
├── docs/                # Requirements and project design
├── Dockerfile
├── docker-compose.yml
│                        # - Retry logic for API resilience
├── TECHNICAL_DOCS.md   # Architecture, RAG step-by-step & Technical FAQ
├── requirements.txt    # Clean dependencies (6 packages)
├── .env                # Your API key (git-ignored)
├── .env.example        # API key template
├── .gitignore          # Ignores .env, venv/, __pycache__/
└── README.md           # Main project overview
```

PDF uploads are validated for type and size, hashed with SHA-256 for duplicate
detection, extracted page by page, and tracked through `uploaded → processing →
ready` or `failed`. Scanned PDFs without selectable text receive a clear upload
error.

---

## 📄 License

Academic / Demo purposes.
