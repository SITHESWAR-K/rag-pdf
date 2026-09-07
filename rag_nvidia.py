import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="PDF Assistant (NVIDIA)", layout="wide", page_icon="📄")
st.title("📄 PDF Question Answering System")

# 1. API Key retrieval (from Streamlit Secrets or sidebar)
api_key = st.secrets.get("NVIDIA_API_KEY", os.getenv("NVIDIA_API_KEY"))
if not api_key:
    api_key = st.sidebar.text_input("Enter NVIDIA API Key", type="password")

if not api_key:
    st.warning("Please configure your NVIDIA API key in Secrets or enter it in the sidebar.")
    st.stop()

# 2. Initialize Models
embedder = NVIDIAEmbeddings(
    model="nvidia/nemotron-3-embed-1b",
    nvidia_api_key=api_key
)

llm = ChatNVIDIA(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    max_tokens=1024,
    nvidia_api_key=api_key
)

# 3. Session State Management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "last_file" not in st.session_state:
    st.session_state.last_file = None

# 4. Sidebar Upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file and st.session_state.last_file != uploaded_file.name:
        with st.spinner("Indexing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
            chunks = splitter.split_documents(pages)

            vectorstore = FAISS.from_documents(chunks, embedder)
            st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            st.session_state.last_file = uploaded_file.name
            st.session_state.messages = []

            os.remove(tmp_path)
            st.success(f"Indexed {len(pages)} pages ({len(chunks)} chunks)!")

# 5. Chat Interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask a question about your PDF..."):
    if not st.session_state.retriever:
        st.warning("Please upload a PDF in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docs = st.session_state.retriever.invoke(user_query)
                context = "\n\n---\n\n".join([
                    f"[Page {d.metadata.get('page', 0) + 1}]: {d.page_content}"
                    for d in docs
                ])

                prompt = (
                    "Answer the question strictly using the provided context below. "
                    "Cite page numbers where available. If not in the context, say you don't know.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {user_query}\nAnswer:"
                )

                response = llm.invoke(prompt)
                answer = response.content
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
