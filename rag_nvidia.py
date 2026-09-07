import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Page Configuration
st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄", layout="wide")
st.title("📄 PDF Assistant (Upload & Ask)")

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") or st.secrets.get("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    st.error("Missing `NVIDIA_API_KEY`. Add it to your `.env` file or Streamlit Secrets.")
    st.stop()

# 2. Models Setup
@st.cache_resource
def get_embeddings():
    return NVIDIAEmbeddings(
        model="nvidia/nemotron-3-embed-1b",
        api_key=NVIDIA_API_KEY
    )

@st.cache_resource
def get_llm():
    return ChatNVIDIA(
        model="openai/gpt-oss-20b",
        api_key=NVIDIA_API_KEY,
        temperature=0.0,
        max_tokens=1024
    )

embeddings = get_embeddings()
llm = get_llm()

# 3. Sidebar: Runtime File Upload & Processing
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None

# Rebuild the vectorstore only when a new file is uploaded
if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_filename:
    with st.spinner(f"Processing and indexing '{uploaded_file.name}'..."):
        # Write the in-memory uploaded file to a temporary file on disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=250,
                separators=["\n\nQuestion", "\n\nQ", "\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(docs)

            # Store the FAISS index directly in session state
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
            st.session_state.last_uploaded_filename = uploaded_file.name
            st.session_state.messages = []  # Reset chat history for the new document
            st.success(f"Indexed {len(chunks)} chunks from '{uploaded_file.name}'.")

        finally:
            # Clean up the temporary file from the disk
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

# Prompt user if no file has been uploaded yet
if st.session_state.vectorstore is None:
    st.info("👈 Please upload a PDF file from the sidebar to begin asking questions.")
    st.stop()

# 4. RAG Chain Setup
retriever = st.session_state.vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

system_prompt = (
    "You are an expert technical assistant. Answer the user's question "
    "using ONLY the facts, programming questions, constraints, and test cases provided "
    "in the context below.\n\n"
    "STRICT INSTRUCTIONS:\n"
    "1. Do not fabricate, assume, or infer details not explicitly stated.\n"
    "2. If the context does not contain the answer, reply: 'The provided document does not contain sufficient information to answer this question.'\n"
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

# 5. Chat Interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Retrieved Document Chunks"):
                for src in msg["sources"]:
                    st.caption(f"**Page {src.metadata.get('page', '?')}**")
                    st.text(src.page_content)

if user_query := st.chat_input(f"Ask anything about {st.session_state.last_uploaded_filename}..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching document & generating answer..."):
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
