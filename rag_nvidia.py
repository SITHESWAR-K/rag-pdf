import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Page Configuration
st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄", layout="wide")
st.title("📄 PDF Assistant (NVIDIA NIM + RAG)")

load_dotenv()

# Streamlit Cloud reads secrets from st.secrets, local runs read from .env
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") or st.secrets.get("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    st.error("Missing `NVIDIA_API_KEY`. Add it to your `.env` or Streamlit Secrets.")
    st.stop()

DOCS_DIR = "docs"
INDEX_PATH = "faiss_index"

# 2. Cached Vector Store Pipeline
@st.cache_resource(show_spinner="Initializing Embeddings & Vector Store...")
def load_or_create_vectorstore():
    # Updated active embedding model (nv-embedqa-e5-v5 is deprecated)
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nemotron-3-embed-1b",
        api_key=NVIDIA_API_KEY
    )

    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        return None

    loader = PyPDFDirectoryLoader(DOCS_DIR)
    docs = loader.load()
    if not docs:
        return None

    # Chunker tailored for code, Q&A, and technical test cases
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\nQuestion", "\n\nQ", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)
    return vectorstore

vectorstore = load_or_create_vectorstore()

if not vectorstore:
    st.warning(f"No documents found in the `{DOCS_DIR}` folder. Please add your PDF files.")
    st.stop()

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# 3. LLM Setup
llm = ChatNVIDIA(
    model="openai/gpt-oss-20b",
    api_key=NVIDIA_API_KEY,
    temperature=0.0,
    max_tokens=1024
)

system_prompt = (
    "You are an expert technical assistant. Answer the user's question "
    "using ONLY the facts, programming questions, constraints, and test cases provided "
    "in the context below.\n\n"
    "STRICT INSTRUCTIONS:\n"
    "1. Do not fabricate, assume, or infer details not explicitly stated.\n"
    "2. If the context does not contain the answer, reply: 'The document does not contain sufficient information.'\n"
    "3. Maintain exact fidelity to code logic, inputs, and outputs.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(
        f"[Page {d.metadata.get('page', '?')}]:\n{d.page_content.strip()}"
        for d in docs
    )

rag_chain = prompt | llm | StrOutputParser()

# 4. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Retrieved Document Chunks"):
                for src in msg["sources"]:
                    st.caption(f"**Page {src.metadata.get('page', '?')}**")
                    st.text(src.page_content)

# Handle user query
if user_query := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching document & generating response..."):
            retrieved_docs = retriever.invoke(user_query)
            context_str = format_docs(retrieved_docs)

            answer = rag_chain.invoke({
                "context": context_str,
                "question": user_query
            })

            st.markdown(answer)

            with st.expander("Retrieved Document Chunks"):
                for doc in retrieved_docs:
                    st.caption(f"**Page {doc.metadata.get('page', '?')}**")
                    st.text(doc.page_content)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved_docs
    })
