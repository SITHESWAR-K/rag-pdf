import os
import sys
import warnings

warnings.filterwarnings("ignore")

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Verify NVIDIA API Key
if not os.environ.get("NVIDIA_API_KEY"):
    print("[!] NVIDIA_API_KEY is not set.")
    os.environ["NVIDIA_API_KEY"] = input("Enter your NVIDIA API key (nvapi-...): ").strip()

# 2. Initialize NVIDIA Models
embedder = NVIDIAEmbeddings(
    model="nvidia/nemotron-3-embed-1b"
)

llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    temperature=0.2,
    max_tokens=1024
)

# 3. Ingest Documents from './docs'
DOCS_DIR = "./docs"
os.makedirs(DOCS_DIR, exist_ok=True)

print(f"[*] Scanning '{DOCS_DIR}' for documents (.pdf, .txt)...")
loaders = [
    DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader),
    DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader),
]

raw_docs = []
for loader in loaders:
    try:
        raw_docs.extend(loader.load())
    except Exception as e:
        print(f"[!] Warning reading loader {loader}: {e}")

# Fallback text if the docs folder is empty
if not raw_docs:
    print(f"[*] No documents found in '{DOCS_DIR}'. Creating sample text.")
    from langchain_core.documents import Document
    raw_docs = [
        Document(
            page_content="RAG pipelines combine semantic vector retrieval with LLM generation.",
            metadata={"source": "default_notes.txt"}
        )
    ]

# 4. Chunk Documents and Build FAISS Index
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
chunks = splitter.split_documents(raw_docs)
print(f"[*] Indexing {len(chunks)} text chunks into FAISS...")

vectorstore = FAISS.from_documents(chunks, embedder)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Build RAG Chain
def format_docs(docs):
    formatted = []
    for d in docs:
        source = d.metadata.get("source", "Unknown")
        formatted.append(f"Source: {source}\nContent: {d.page_content}")
    return "\n\n---\n\n".join(formatted)

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an assistant answering questions strictly based on the context below. "
        "Cite the document source where relevant. If the answer is not in the context, say you do not know.\n\n"
        "Context:\n{context}"
    ),
    ("human", "{question}")
])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# 6. Runtime Interactive Loop
print("\n==================================================")
print(" RAG System Ready. Type your question below.")
print(" Type 'exit' or 'quit' to end the session.")
print("==================================================\n")

while True:
    try:
        query = input("Ask a question: ").strip()

        # Exit conditions
        if not query:
            continue
        if query.lower() in ["exit", "quit", "q"]:
            print("\nExiting session.")
            sys.exit(0)

        print("\nThinking...")
        response = rag_chain.invoke(query)
        print(f"\nAnswer:\n{response}\n")
        print("-" * 50)

    except KeyboardInterrupt:
        print("\nSession interrupted. Exiting.")
        sys.exit(0)
    except Exception as err:
        print(f"\n[!] Error during execution: {err}\n")
