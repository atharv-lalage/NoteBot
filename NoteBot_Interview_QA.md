# NoteBot Interview Q&A

### 1. Tell me about this project
**NoteBot** is an AI-powered study assistant designed to help students or professionals interact directly with their PDF notes.

In simple terms, you can upload a PDF document, and the application allows you to do three main things:
1. **Chat with your notes:** You can ask questions, and the AI will find the exact information from your document to answer you.
2. **Generate Summaries:** With one click, it provides a structured breakdown of the document, including the main title, a brief summary, key topics, and study tips.
3. **Create Flashcards:** It can automatically generate Q&A flashcards from the text, categorizing them by difficulty to help you study.

**Under the hood:**
It is built using a **React/Vite frontend** and a **FastAPI backend**. The "brain" of the project is powered by **Google's Gemini 2.5 Flash API**. To make the AI answer questions accurately based *only* on the uploaded document, it uses a technique called **RAG (Retrieval-Augmented Generation)**.

---

### 2. Give architecture of this project and architecture diagram
NoteBot follows a classic **Retrieval-Augmented Generation (RAG)** architecture. It consists of a frontend interface that communicates with a backend data processing pipeline, which in turn integrates with Google's Gemini APIs for AI capabilities.

1. **Frontend (React/Vite):** The user interface where users upload PDFs and interact.
2. **Document Ingestion & Chunking:** PyMuPDF extracts text page by page, and LangChain's `RecursiveCharacterTextSplitter` breaks it into chunks.
3. **Embedding & Storage:** Text chunks are sent to the `Gemini Embedding 001` API to be converted into numerical vectors, which are temporarily stored in memory using NumPy.
4. **Retrieval:** The user's question is converted into an embedding, and Cosine Similarity finds the top 4 matching text chunks.
5. **Generation:** The retrieved chunks, the question, and chat history are sent to **Gemini 2.5 Flash**, which streams back the answer.

---

### 3. Give HLD and LLD of project
**High-Level Design (HLD):**
1. **Presentation Layer (Frontend):** React/Vite application.
2. **Data Processing Pipeline (Ingestion):** PyMuPDF, LangChain Text Splitter.
3. **Vector Representation & Storage:** NumPy retrieval adapter backed by PostgreSQL/pgvector.
4. **External AI Services (The "Brain"):** Google Gemini API.

**Low-Level Design (LLD):**
1. **Data Structures & State Management:** `st.session_state` stores chunks, embeddings (NumPy array), and messages.
2. **Core Algorithms & Functions:** `process_pdf` (chunking), `embed_texts` (batching API calls), `retrieve` (Cosine Similarity search), and `safe_api_call` (Exponential Backoff).
3. **Prompt Engineering & Generation Flow:** Concatenates context, uses streaming responses, and structured JSON outputs for Summaries/Flashcards.

---

### 4. What problem does NoteBot solve?
NoteBot solves the problem of information overload and inefficient studying when dealing with dense documents. It addresses:
1. **Time-consuming manual search:** Ask a question in natural language to find information instantly.
2. **AI Hallucinations:** Uses RAG to strictly lock the AI's answers *only* to the facts present in your uploaded notes.
3. **Study Prep Friction:** Automates the creation of structured summaries and difficulty-graded flashcards.

---

### 5. Why did you build NoteBot?
**1. The Practical Reason:** I wanted a solution where you could just "talk" to your textbook and have the tedious prep work—like summarizing and creating study cards—automated for you.
**2. The Technical Reason:** I wanted to build a comprehensive, end-to-end demonstration of modern Generative AI engineering patterns. I wanted to implement core concepts from scratch—like calculating Cosine Similarity using pure NumPy—and showcase advanced capabilities like real-time streaming and Structured JSON Outputs.

---

### 6. What are the main features of NoteBot?
1. **Interactive Document Chat (RAG):** Chat with notes with real-time streaming answers.
2. **Transparent Source Attribution:** Shows exactly which chunks of the PDF it used.
3. **Automated Structured Summaries:** One-click JSON breakdown (Title, Summary, Topics, Difficulty, Tips).
4. **Instant AI Flashcards:** Auto-generates Q&A study cards tagged by difficulty.
5. **Production-Grade Reliability:** Exponential backoff and retry logic handles API limits gracefully.

---

### 7. Walk me through what happens when a user uploads a PDF
1. **Text Extraction:** PyMuPDF reads the file and extracts page-aware text.
2. **Intelligent Chunking:** LangChain's `RecursiveCharacterTextSplitter` breaks the document down into blocks of 800 characters with a 200-character overlap.
3. **Vector Embedding:** The chunks are sent to the Gemini Embedding API in batches of 100 to get 768-dimensional mathematical vectors.
4. **State Storage:** The original text chunks and vectors are persisted in PostgreSQL with pgvector.

---

### 8. Walk me through what happens when a user asks a question
1. **Query Embedding:** The question is sent to the Gemini Embedding API.
2. **Similarity Search (Retrieval):** Uses Cosine Similarity via NumPy to compare the question's vector against all chunk vectors, retrieving the top 4 most relevant chunks.
3. **Prompt Construction:** A prompt is built combining system instructions, the retrieved context, chat history, and the question.
4. **Streaming Generation:** The prompt is sent to Gemini 2.5 Flash, which streams the answer token-by-token.

---

### 9. Explain the complete RAG pipeline in your project
**Phase 1: Ingestion**
Extraction (PyMuPDF) → Chunking (LangChain) → Embedding (Gemini API) → Storage (NumPy / Session State).
**Phase 2: Retrieval & Generation**
Query Embedding (Gemini API) → Similarity Search (Cosine Similarity) → Prompt Engineering (Context + Question) → Generation (Gemini 2.5 Flash Streaming).

---

### 10. What technologies did you use and why?
1. **React/Vite:** For a responsive, component-based study workspace.
2. **Google Gemini API:** Fast latency (streaming), massive context window, and native JSON mode.
3. **NumPy:** For manual, fast in-memory Cosine Similarity search without heavy DB dependencies.
4. **LangChain:** For intelligent, semantic text chunking.
5. **PyMuPDF:** For reliable, page-aware PDF text extraction and validation.

---

### 11. What was your contribution to the project?
As the sole developer, I owned the project end-to-end:
1. **AI Pipeline Engineering:** Designed the RAG pipeline, manually implemented Cosine Similarity using pure NumPy, and optimized Gemini API calls with batching.
2. **Frontend Development:** Built the React UI, connected it to FastAPI, and implemented real-time response streaming.
3. **Reliability & Documentation:** Added exponential backoff retry logic and authored comprehensive technical documentation.

---

### 12. What was the most challenging part of building NoteBot?
1. **Ensuring Reliable API Interactions:** Handling rate limits (429/503 errors) when batching embeddings required building a custom `safe_api_call` wrapper with Exponential Backoff.
2. **Forcing Parseable Data:** Getting the AI to return clean JSON without markdown wrappers for flashcards required implementing strict Structured Output via Gemini's JSON mode.
3. **Math from Scratch:** Implementing the Cosine Similarity vector search mathematically using pure NumPy instead of a pre-built vector DB.

---

### 13. What are embeddings?
Embeddings are a way to translate human language into numbers (vectors) that a computer can understand. They capture the **semantic meaning** of text, not just exact words. For example, "Photosynthesis" and "plants making food from sunlight" will have mathematically similar embeddings even if they don't share identical words.

---

### 14. Why cosine similarity?
Cosine Similarity measures the *angle* (direction) between two vectors rather than their magnitude (length). This is crucial for text because a short question and a long explanatory paragraph can have the exact same core meaning. Cosine similarity recognizes they are pointing in the same semantic direction, ignoring their difference in word count.

---

### 15. Why chunking + 200 overlap?
**Chunking:** AI models have context limits, and feeding massive texts can lead to the AI losing information in the middle. Chunks of 800 characters provide focused context.
**Overlap:** A 200-character overlap prevents vital sentences from being cut in half at chunk boundaries, ensuring continuous context is always preserved for the AI.

---

### 16. Why top-4 retrieval?
It's the "Goldilocks" zone. Top-1 might miss information spread across multiple paragraphs. Top-20 introduces too much irrelevant noise, increasing the chance of hallucinations, consuming more tokens, and slowing down the response. Top-4 (~3,200 characters) provides enough breadth for a comprehensive answer while keeping speed high and cost low.

---

### 17. Why Gemini 2.5 Flash?
It is heavily optimized for speed and low latency, which is essential for a streaming chat experience. It natively supports strict Structured Output (JSON mode) for the summaries and flashcards, and it has a massive 1-million-token context window that easily scales.

---

### 18. Why did you implement NumPy retrieval instead of FAISS/Chroma?
1. **Scale:** For a single-document session, creating an in-memory NumPy array is vastly faster and simpler than spinning up an external database like ChromaDB.
2. **Technical Depth:** I wanted to build the search from scratch to demonstrate I deeply understand the linear algebra and math behind how semantic search actually works, rather than treating vector search like a magic black box.

---

### 19. What would you improve in NoteBot?
1. **Hybrid Search:** Combine semantic vector search with traditional keyword search (BM25) to catch specific names or acronyms.
2. **Persistent Storage:** Add a backend and persistent vector DB (like ChromaDB or PostgreSQL + pgvector) so users can permanently save documents.
3. **Automated Evaluation:** Integrate an evaluation framework like RAGAS to programmatically score answers for faithfulness and relevance.
