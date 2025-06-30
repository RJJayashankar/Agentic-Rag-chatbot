from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List

def embed_and_store_chunks(chunks: List[Document], persist_directory: str = "./chroma_db") -> Chroma:
    print(f"Embedding {len(chunks)} chunks with HuggingFace...")
    
    # ✅ Initialize local embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # ✅ Store chunks in Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    vectorstore.persist()
    print("Embedding and storage complete.")
    return vectorstore
