import os
import hashlib
import pickle
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# 1. Page & Layout Setup
st.set_page_config(
    page_title="Advanced PDF RAG Assistant",
    page_icon="🧠",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Cache directory for persistent vector store
CACHE_DIR = ".faiss_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# 2. Sidebar: Configuration & API Key Management
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Secure API Key lookup: .env -> st.secrets -> sidebar input
    env_api_key = os.getenv("NVIDIA_API_KEY") or (
        st.secrets.get("NVIDIA_API_KEY") if hasattr(st, "secrets") and "NVIDIA_API_KEY" in st.secrets else None
    )
    
    if env_api_key:
        api_key = env_api_key
        st.success("NVIDIA API Key loaded securely.", icon="🔒")
    else:
        api_key = st.text_input(
            "NVIDIA API Key",
            type="password",
            placeholder="nvapi-...",
            help="Get your free API key at https://build.nvidia.com"
        )
        if not api_key:
            st.info("Please enter your NVIDIA API Key or set it in `.env`.", icon="ℹ️")

    st.markdown("---")
    st.subheader("🛠️ RAG Pipeline Settings")
    
    # Model Selection
    available_models = [
        "google/gemma-4-31b-it",
        "meta/llama-3.3-70b-instruct",
        "mistralai/mistral-large-2-instruct"
    ]
    selected_model = st.selectbox("LLM Generator", available_models, index=0)
    
    # Retrieval Tuning
    use_hybrid = st.toggle("Hybrid Search (BM25 + FAISS)", value=True, help="Combines exact keyword matching with dense semantic search via Reciprocal Rank Fusion.")
    use_reranker = st.toggle("NVIDIA NIM Reranking", value=True, help="Uses NVIDIA cross-encoder reranker to refine top candidate chunks.")
    top_k = st.slider("Context Chunks (Top-K)", min_value=2, max_value=8, value=4)

    st.markdown("---")
    st.subheader("📂 Document Management")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# Stop execution early if no API Key is provided
if not api_key:
    st.warning("⚠️ **Missing NVIDIA API Key**. Add `NVIDIA_API_KEY` to your `.env` file or enter it in the sidebar to proceed.")
    st.stop()

# 3. Model Initializers (Cached)
@st.cache_resource(show_spinner=False)
def get_embeddings(key: str):
    return NVIDIAEmbeddings(
        model="nvidia/nemotron-3-embed-1b",
        api_key=key
    )

@st.cache_resource(show_spinner=False)
def get_llm(model_name: str, key: str):
    return ChatNVIDIA(
        model=model_name,
        api_key=key,
        temperature=0.1,
        max_tokens=1500
    )

@st.cache_resource(show_spinner=False)
def get_reranker(key: str):
    try:
        from langchain_nvidia_ai_endpoints import NVIDIARerank
        return NVIDIARerank(
            model="nvidia/llama-3.2-nv-rerankqa-1b-v2",
            api_key=key,
            top_n=top_k
        )
    except Exception:
        return None

embeddings = get_embeddings(api_key)
llm = get_llm(selected_model, api_key)
reranker = get_reranker(api_key) if use_reranker else None

# 4. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []

if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None

if "doc_summary" not in st.session_state:
    st.session_state.doc_summary = None

# 5. Persistent Indexing & Document Ingestion
def get_file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = get_file_hash(file_bytes)
    file_cache_path = os.path.join(CACHE_DIR, file_hash)

    # Check if a new file has been uploaded
    if st.session_state.current_file_hash != file_hash:
        st.session_state.current_file_hash = file_hash
        st.session_state.messages = []
        st.session_state.doc_summary = None

        chunks = []
        loaded_from_cache = False

        # Attempt to load from persistent cache
        if os.path.exists(file_cache_path) and os.path.exists(os.path.join(file_cache_path, "chunks.pkl")):
            try:
                with st.spinner("Loading cached vector index from disk..."):
                    vectorstore = FAISS.load_local(
                        file_cache_path,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    with open(os.path.join(file_cache_path, "chunks.pkl"), "rb") as f:
                        chunks = pickle.load(f)
                    loaded_from_cache = True
            except Exception as e:
                st.warning(f"Could not load cache: {e}. Rebuilding index...")

        # If not cached, extract, split, and persist
        if not loaded_from_cache:
            with st.spinner(f"Extracting and indexing '{uploaded_file.name}'..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file_bytes)
                    tmp_file_path = tmp_file.name

                try:
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        separators=["\n\nQuestion", "\n\nQ", "\n\n", "\n", " ", ""]
                    )
                    chunks = text_splitter.split_documents(docs)

                    # Enrich metadata with chunk IDs
                    for i, chunk in enumerate(chunks):
                        chunk.metadata["chunk_id"] = i

                    # Build and save FAISS vectorstore
                    vectorstore = FAISS.from_documents(chunks, embeddings)
                    vectorstore.save_local(file_cache_path)

                    # Save chunks for BM25 and quick inspection
                    with open(os.path.join(file_cache_path, "chunks.pkl"), "wb") as f:
                        pickle.dump(chunks, f)

                finally:
                    if os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)

        st.session_state.doc_chunks = chunks

        # Build Hybrid (BM25 + FAISS) or Pure Vector Retriever
        faiss_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k * 2 if use_reranker else top_k}
        )

        if use_hybrid and chunks:
            bm25_retriever = BM25Retriever.from_documents(chunks)
            bm25_retriever.k = top_k * 2 if use_reranker else top_k
            st.session_state.retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.4, 0.6]
            )
        else:
            st.session_state.retriever = faiss_retriever

        cache_status_msg = "⚡ Loaded from disk cache" if loaded_from_cache else "🔨 Indexed & saved to cache"
        st.sidebar.success(f"{cache_status_msg} ({len(chunks)} chunks)")

# Quick summary trigger in sidebar
with col_btn2:
    if uploaded_file and st.button("Summary", use_container_width=True, help="Generate an executive summary of the document"):
        if st.session_state.doc_chunks:
            with st.spinner("Generating document summary..."):
                sample_text = "\n\n".join([c.page_content for c in st.session_state.doc_chunks[:6]])
                summary_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an expert analyst. Provide a clear, structured summary of this document and list 3 suggested questions a user could ask about it."),
                    ("human", "Document preview:\n{text}")
                ])
                summary_chain = summary_prompt | llm | StrOutputParser()
                summary_res = summary_chain.invoke({"text": sample_text})
                st.session_state.doc_summary = summary_res

# 6. Guardrail: Ensure Document is Uploaded
if not uploaded_file or st.session_state.retriever is None:
    st.markdown(
        """
        ## 🧠 Welcome to the Advanced PDF RAG Assistant
        
        This assistant is powered by **NVIDIA NIM**, featuring:
        * **Hybrid Search**: Dense semantic search (Nemotron embeddings) fused with sparse keyword search (BM25).
        * **Cross-Encoder Reranker**: Precision ranking powered by NVIDIA NIM rerank models.
        * **Conversational Memory**: Multi-turn contextual follow-ups.
        * **Token Streaming**: Instant real-time responses.
        * **Persistent Disk Caching**: Fast reloads with zero redundant embedding computations.

        ---
        👈 **To get started, please upload a PDF file from the sidebar.**
        """
    )
    st.stop()

# 7. Document Summary Card (if generated)
if st.session_state.doc_summary:
    with st.expander("📑 Document Executive Summary & Suggested Questions", expanded=True):
        st.markdown(st.session_state.doc_summary)

# 8. Conversational Chains & Logic
def format_docs(docs):
    formatted = []
    for d in docs:
        page = d.metadata.get("page", 0)
        # Adjust 0-indexed page to 1-indexed for human readability
        page_num = page + 1 if isinstance(page, int) else page
        formatted.append(f"[Page {page_num}]:\n{d.page_content.strip()}")
    return "\n\n".join(formatted)

def retrieve_and_rerank(query: str, base_retriever, reranker_model, final_k: int):
    """Retrieves documents with hybrid retriever and optionally reranks via NVIDIA NIM."""
    initial_docs = base_retriever.invoke(query)
    if reranker_model and initial_docs:
        try:
            compressed = reranker_model.compress_documents(query=query, documents=initial_docs)
            return compressed[:final_k]
        except Exception:
            # Graceful fallback if reranker API has unexpected latency or issues
            return initial_docs[:final_k]
    return initial_docs[:final_k]

# Conversational Query Reformulation (Contextualization)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a chat history and the latest user question which might reference context "
        "in the chat history, formulate a standalone question that can be understood "
        "without the chat history. Do NOT answer the question, just reformulate it if needed "
        "and otherwise return it as is."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])
contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()

# Final Answer Generation Prompt
qa_system_prompt = (
    "You are an expert technical assistant. Answer the user's question "
    "using ONLY the facts, programming questions, constraints, and test cases provided "
    "in the context below.\n\n"
    "STRICT RULES:\n"
    "1. Do not fabricate, assume, or infer details not explicitly stated in the context.\n"
    "2. If the context does not contain the answer, reply: 'The provided document does not contain sufficient information to answer this question.'\n"
    "3. Maintain exact fidelity to code logic, variables, inputs, and outputs.\n\n"
    "Context:\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])
qa_chain = qa_prompt | llm | StrOutputParser()

# 9. Chat Display History
# Convert session messages to LangChain message objects for context
langchain_history = []
for msg in st.session_state.messages:
    if msg["role"] == "user":
        langchain_history.append(HumanMessage(content=msg["content"]))
    else:
        langchain_history.append(AIMessage(content=msg["content"]))

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"🔍 Retrieved Sources ({len(msg['sources'])})", expanded=False):
                for idx, src in enumerate(msg["sources"], start=1):
                    page = src.metadata.get("page", 0)
                    page_num = page + 1 if isinstance(page, int) else page
                    st.markdown(f"**Source #{idx} — Page {page_num}**")
                    st.code(src.page_content, language="text")

# 10. User Query & Streaming Response Flow
if user_query := st.chat_input(f"Ask about '{uploaded_file.name}'..."):
    # Append user question to history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        # Step A: Reformulate query if multi-turn history exists
        if len(langchain_history) > 0:
            with st.spinner("Understanding question in conversation context..."):
                standalone_query = contextualize_chain.invoke({
                    "chat_history": langchain_history,
                    "question": user_query
                })
        else:
            standalone_query = user_query

        # Step B: Retrieval + Re-ranking
        with st.spinner("Searching document & ranking relevance..."):
            retrieved_docs = retrieve_and_rerank(
                query=standalone_query,
                base_retriever=st.session_state.retriever,
                reranker_model=reranker if use_reranker else None,
                final_k=top_k
            )
            context_text = format_docs(retrieved_docs)

        # Step C: Stream Answer Generation
        stream = qa_chain.stream({
            "context": context_text,
            "chat_history": langchain_history,
            "question": user_query
        })
        
        full_answer = st.write_stream(stream)

        # Step D: Display Retrieved Sources
        if retrieved_docs:
            with st.expander(f"🔍 Retrieved Sources ({len(retrieved_docs)})", expanded=False):
                for idx, doc in enumerate(retrieved_docs, start=1):
                    page = doc.metadata.get("page", 0)
                    page_num = page + 1 if isinstance(page, int) else page
                    st.markdown(f"**Source #{idx} — Page {page_num}**")
                    st.code(doc.page_content, language="text")

    # Append assistant response with its retrieved sources to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": retrieved_docs
    })
