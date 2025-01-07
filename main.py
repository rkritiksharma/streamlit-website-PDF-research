import os
import streamlit as st
import pickle
import google.generativeai as genai  # Gemini API
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import pdfplumber

# Gemini API Key Setup
GEMINI_API_KEY = "AIzaSyCCw-l2HgxXQMMvDdN_CEMB2SqYwLMbuP0"
genai.configure(api_key=GEMINI_API_KEY)

# --- Set Page Configuration ---
st.set_page_config(
    page_title="News Research Bot 📚",
    page_icon="🔍",
    layout="wide",
)

# Custom CSS
st.markdown(
    """
    <style>
        .main-title { font-size: 3.5rem; color: #4CAF50; text-align: center; }
        .answer-box { background-color: #E8F8F5; padding: 1rem; border-radius: 5px; }
        .source-box { background-color: #FEF5E7; padding: 0.8rem; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True
)

# Title
st.markdown('<h1 class="main-title">🚀 News Research Tool 📊</h1>', unsafe_allow_html=True)

# Sidebar for URLs and File Upload
st.sidebar.markdown("### 📜 Input Article URLs or PDF")
urls = [st.sidebar.text_input(f"📝 URL {i+1}") for i in range(3)]
pdf_file = st.sidebar.file_uploader("📄 Upload PDF", type="pdf")
process_clicked = st.sidebar.button("🔍 Process URLs/PDF")

# FAISS Storage Path
file_path = st.sidebar.text_input("📂 FAISS File Path:", "faiss_store_gemini.pkl")

# Load URLs
def load_data_with_retry(urls, retries=3):
    data = []
    for url in urls:
        for attempt in range(retries):
            try:
                loader = UnstructuredURLLoader(urls=[url])
                documents = loader.load()
                data.extend(documents)
                st.success(f"✅ Loaded data from {url}")
                break
            except Exception as e:
                st.error(f"Retry {attempt+1}: Failed to load {url}. Error: {e}")
    return data

# Load PDF
def load_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        data = []
        for page in pdf.pages:
            text = page.extract_text()
            # Create Document object from extracted text
            if text:
                data.append(Document(page_content=text))
        return data

if process_clicked:
    valid_urls = [url for url in urls if url]
    if not valid_urls and not pdf_file:
        st.error("⚠️ Provide at least one URL or upload a PDF!")
    else:
        st.info("⏳ Loading and processing data...")
        
        # Load data from URLs
        data = []
        if valid_urls:
            data = load_data_with_retry(valid_urls)
        
        # Load data from PDF
        if pdf_file:
            pdf_data = load_pdf(pdf_file)
            data.extend(pdf_data)

        if data:
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
            docs = text_splitter.split_documents(data)

            # Embed and store in FAISS
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)

            # Save the FAISS index
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore, f)
            st.success(f"✅ FAISS index saved at: {file_path}")
        else:
            st.error("❌ Failed to load data.")

# Query Section
st.markdown("### 🔎 Ask a Question")
query = st.text_input("💡 Enter your question here:")

if query:
    if os.path.exists(file_path):
        # Load FAISS index
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        # Retrieve relevant context
        retriever = vectorstore.as_retriever()
        relevant_docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # Query Gemini API
        st.info("🔍 Generating answer using Gemini...")
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(f"Context: {context}\n\nQuestion: {query}")

        # Display the answer
        st.markdown("### 🧠 Answer")
        st.write(response.text)

        # Display Sources
        st.markdown("### 📚 Sources")
        seen_sources = set()
        for i, doc in enumerate(relevant_docs, start=1):
            source = doc.metadata.get('source', 'PDF')
            if source not in seen_sources:
                st.write(f"{i}. {source}")
                seen_sources.add(source)
            
    else:
        st.error(f"❌ FAISS index not found at {file_path}. Process URLs first.")

# Footer Section
st.markdown(
    """
    <div class="footer" style="text-align:center; font-family:Arial; font-size:16px;">
        🛠️ <b style="color:#2196F3;">Developed by RK</b> |
        ✅ <b style="color:#4CAF50;">Powered by Gemini Pro, Hugging Face, LangChain & FAISS</b> |
        🚀 <b style="color:#FF9800;">Built with Streamlit</b> |
    </div>
    """,
    unsafe_allow_html=True
)
