import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    print("Error: NVIDIA_API_KEY is not set in your .env file.")
    sys.exit(1)

DOCS_DIR = "docs"
INDEX_PATH = "faiss_index"

# 1. Models Setup
# High-performing retrieval and generation models from NVIDIA NIM
embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    api_key=NVIDIA_API_KEY
)

llm = ChatNVIDIA(
    model="meta/llama-3.1-70b-instruct",
    api_key=NVIDIA_API_KEY,
    temperature=0.0,
    max_tokens=1024
)

# 2. Vector Store & Ingestion Pipeline
def get_vectorstore():
    """Loads existing FAISS index or ingests PDFs with optimized chunking."""
    if os.path.exists(INDEX_PATH):
        print("Loading existing FAISS index from disk...")
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"Directory '{DOCS_DIR}' was empty. Place your PDF inside and rerun.")
        sys.exit(1)

    print(f"Reading PDFs from '{DOCS_DIR}'...")
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    docs = loader.load()

    if not docs:
        print(f"No PDFs found in '{DOCS_DIR}'.")
        sys.exit(1)

    # Coding questions & I/O examples require larger chunk sizes and tailored separators
    # so problem statements, constraints, and test cases do not get split across chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\nQuestion", "\n\nQ", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} pages.")

    print("Generating embeddings and building FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print(f"Vector store saved to '{INDEX_PATH}'.")
    return vectorstore

def format_docs(docs):
    """Formats retrieved Document objects into a clean contextual block."""
    formatted_chunks = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "?")
        formatted_chunks.append(
            f"[Document {i} | Source: {source} (Page {page})]\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted_chunks)

# 3. Grounded Strict Prompt
# Rules explicitly stop the model from fabricating I/O cases or using training knowledge
system_prompt = (
    "You are an expert technical evaluation assistant. Answer the user's question "
    "using ONLY the facts, programming questions, constraints, and test cases provided "
    "in the context below.\n\n"
    "STRICT INSTRUCTIONS:\n"
    "1. Do not fabricate, assume, or infer inputs, outputs, or problem statements not explicitly written in the context.\n"
    "2. If the answer or specific question number is not present in the context, respond strictly with: "
    "'The provided document does not contain sufficient information to answer this question.'\n"
    "3. When presenting code logic, inputs, or outputs, maintain exact fidelity to the provided context.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

def main():
    vectorstore = get_vectorstore()
    
    # Retrieve top 5 most relevant chunks to capture multi-part problems
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # LCEL pipeline
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n--- RAG System Ready (Type 'exit' to quit) ---")
    while True:
        try:
            query = input("\nEnter your question: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                break

            # Diagnostic Step: Inspect retrieved context for verification
            retrieved_docs = retriever.invoke(query)
            print(f"\n[Debug] Retrieved {len(retrieved_docs)} chunks:")
            for idx, doc in enumerate(retrieved_docs, 1):
                preview = doc.page_content.replace("\n", " ")[:120]
                page = doc.metadata.get("page", "?")
                print(f"  ({idx}) Page {page}: {preview}...")

            print("\nGenerating Answer...")
            answer = rag_chain.invoke(query)
            print(f"\nAnswer:\n{answer}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
