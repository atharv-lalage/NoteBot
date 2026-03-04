import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
from PyPDF2 import PdfReader
import time
import numpy as np
from dotenv import load_dotenv

from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.header("NoteBot (Gemini Lite)")

if not GEMINI_API_KEY:
    st.error("⚠️ API Key not found!")
    st.stop()

# Initialize client GLOBALLY at top level
client = genai.Client(api_key=GEMINI_API_KEY)

# --- Embedding function ---
def get_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type)
    )
    return result.embeddings[0].values

# --- Simple Vector Store ---
class SimpleVectorStore:
    def __init__(self):
        self.texts = []
        self.embeddings = []

    def add_texts(self, texts: list[str]):
        for text in texts:
            emb = get_embedding(text, task_type="RETRIEVAL_DOCUMENT")
            self.texts.append(text)
            self.embeddings.append(emb)

    def similarity_search(self, query: str, k: int = 4) -> list[str]:
        query_emb = np.array(get_embedding(query, task_type="RETRIEVAL_QUERY"))
        scores = []
        for emb in self.embeddings:
            emb_arr = np.array(emb)
            score = np.dot(query_emb, emb_arr) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb_arr)
            )
            scores.append(score)
        top_k = np.argsort(scores)[::-1][:k]
        return [self.texts[i] for i in top_k]

with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader("Upload notes PDF", type="pdf")

if file is not None:
    with st.spinner("Processing PDF..."):
        try:
            my_pdf = PdfReader(file)
            text = ""
            for page in my_pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted

            if not text:
                st.error("Could not extract text.")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_text(text)
            st.write(f"✅ Extracted {len(text)} characters, {len(chunks)} chunks.")

            vector_store = SimpleVectorStore()
            progress_bar = st.progress(0)

            for i, chunk in enumerate(chunks):
                vector_store.add_texts([chunk])
                progress_bar.progress((i + 1) / len(chunks))
                time.sleep(0.1)

            st.success("✅ Database Ready! Ask your question below.")

        except Exception as e:
            st.error(f"Error during processing: {e}")
            st.stop()

    user_query = st.text_input("Type your query here")

    if user_query:
        try:
            relevant_chunks = vector_store.similarity_search(user_query, k=4)
            context_text = "\n\n".join(relevant_chunks)

            prompt = f"""You are my assistant tutor. Answer the question based on the following context.
If the answer is not in the context, simply say "I don't know Atharv".

Context:
{context_text}

Question:
{user_query}
"""
            with st.spinner("Thinking..."):
                response = client.models.generate_content(
                    model="gemini-flash-latest",
                    contents=prompt
                )
                st.write(response.text)  # <-- .text for clean output

        except Exception as e:
            st.error(f"An error occurred: {e}")