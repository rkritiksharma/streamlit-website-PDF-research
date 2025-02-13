import os
import streamlit as st
import pickle
import config  # Import your config module
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import pdfplumber
import pytesseract  # OCR for scanned PDFs
from PIL import Image
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# ✅ Securely Fetch API Key
if not config.GEMINI_API_KEY:
    st.error("⚠️ Please set the GEMINI_API_KEY in the .env file!")
    st.stop()

genai.configure(api_key=config.GEMINI_API_KEY)

# ✅ Sidebar for User Input
st.sidebar.markdown("### 📜 Input Article URLs or PDF")
urls = [st.sidebar.text_input(f"📝 URL {i+1}") for i in range(3)]
pdf_file = st.sidebar.file_uploader("📄 Upload PDF", type="pdf")
process_clicked = st.sidebar.button("🔍 Process URLs/PDF")

# ✅ FAISS Storage Path from Config
file_path = config.FAISS_FILE_PATH

# ✅ Load URLs with Retry Logic
def load_data_with_retry(urls, retries=3):
    data = []
    for url in urls:
        if not url.strip():
            continue
        for attempt in range(retries):
            try:
                loader = UnstructuredURLLoader(urls=[url])
                documents = loader.load()
                data.extend(documents)
                st.success(f"✅ Loaded data from {url}")
                break
            except Exception as e:
                st.warning(f"Retry {attempt+1}: Failed to load {url}. Error: {e}")
    return data

# ✅ Load PDF with Improved OCR Handling
def load_pdf(pdf_file):
    data = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    image = page.to_image(resolution=300)
                    text = pytesseract.image_to_string(Image.open(image))
                
                if text:
                    data.append(Document(page_content=text))
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
    return data

# ✅ Process URLs and PDFs
if process_clicked:
    valid_urls = [url for url in urls if url]
    if not valid_urls and not pdf_file:
        st.error("⚠️ Provide at least one URL or upload a PDF!")
    else:
        st.info("⏳ Processing documents...")

        data = []
        if valid_urls:
            data.extend(load_data_with_retry(valid_urls))

        if pdf_file:
            data.extend(load_pdf(pdf_file))

        if data:
            with st.spinner("🔍 Creating Embeddings..."):
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = text_splitter.split_documents(data)

                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = FAISS.from_documents(docs, embeddings)

                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore, f)
                
                st.success(f"✅ FAISS index saved at: {file_path}")
        else:
            st.error("❌ No valid data found. Check your URLs or PDF file.")

# ✅ Query Section
st.markdown("### 🔎 Ask a Question")
query = st.text_input("💡 Enter your question here:")

if query:
    if os.path.exists(file_path):
        with st.spinner("🔍 Searching..."):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)

            llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

            template = """
            Use the following context to answer the question at the end.
            If the context doesn't contain the answer, say "I cannot answer this based on the provided information."
            Be concise and cite the sources.

            Context: {context}

            Question: {question}
            """
            PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True,
            )

            result = qa_chain({"query": query})

            # ✅ Display Answer
            st.markdown("### 🧠 Answer")
            st.write(result["result"])

            # ✅ Display Sources
            st.markdown("### 📚 Sources")
            seen_sources = set()
            for i, doc in enumerate(result["source_documents"], start=1):
                source = doc.metadata.get('source', 'Unknown Source')
                if source not in seen_sources:
                    st.write(f"{i}. {source}")
                    seen_sources.add(source)

    else:
        st.error(f"❌ FAISS index not found at {file_path}. Process URLs first.")

# ✅ Footer
st.markdown(
    """
    <div style="text-align: center; font-size: 16px;">
        🛠️ <b style="color: #2196F3;">Developed by RK</b> |
        ✅ <b style="color: #4CAF50;">Powered by Gemini Pro, LangChain & FAISS</b> |
        🚀 <b style="color: #FF9800;">Built with Streamlit</b>
    </div>
    """,
    unsafe_allow_html=True,
)
