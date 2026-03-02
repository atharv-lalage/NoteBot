import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
from PyPDF2 import PdfReader
import time
from dotenv import load_dotenv

# --- NEW SDK (not deprecated) ---
from google import genai
from google.genai import types

# --- LangChain Imports ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.header("NoteBot (Gemini Lite)")

if not GEMINI_API_KEY:
    st.error("⚠️ API Key not found! Make sure .env file has GEMINI_API_KEY.")
    st.stop()

# --- Custom Embeddings class using NEW google-genai SDK ---
class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            result = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        return result.embeddings[0].values

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
                st.error("Could not extract text. The PDF might be an image/scan.")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_text(text)
            st.write(f"✅ Extracted {len(text)} characters, {len(chunks)} chunks.")

            # Use NEW custom embeddings class
            embeddings = GeminiEmbeddings(api_key=GEMINI_API_KEY)

            vector_store = None
            batch_size = 10
            progress_bar = st.progress(0)

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                if vector_store is None:
                    vector_store = FAISS.from_texts(batch, embeddings)
                else:
                    vector_store.add_texts(batch)

                progress = min((i + batch_size) / len(chunks), 1.0)
                progress_bar.progress(progress)
                time.sleep(0.5)

            st.success("✅ Database Ready! Ask your question below.")

        except Exception as e:
            st.error(f"Error during processing: {e}")
            st.stop()

    user_query = st.text_input("Type your query here")

    if user_query:
        try:
            docs = vector_store.similarity_search(user_query)
            context_text = "\n\n".join([doc.page_content for doc in docs])

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-lite",
                google_api_key=GEMINI_API_KEY,
                temperature=0.3
            )

            prompt_template = ChatPromptTemplate.from_template(
                """You are my assistant tutor. Answer the question based on the following context.
                If the answer is not in the context, simply say "I don't know Atharv".

                Context:
                {context}

                Question:
                {input}
                """
            )

            with st.spinner("Thinking..."):
                messages = prompt_template.format_messages(context=context_text, input=user_query)
                response = llm.invoke(messages)
                st.write(response.content)

        except Exception as e:
            st.error(f"An error occurred: {e}")