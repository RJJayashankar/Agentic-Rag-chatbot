from pdf_loader_chunker import load_and_chunk_pdf
from embed_store import embed_and_store_chunks
from agent import graph  # <- Compiled LangGraph
import os
from dotenv import load_dotenv

load_dotenv()

def ingest():
    pdf_path = "data/CIS-MS-ISAC-NIST-Cybersecurity-Framework-Policy-Template-Guide-2024.pdf"
    chunks = load_and_chunk_pdf(pdf_path)
    vectorstore = embed_and_store_chunks(chunks, persist_directory="./chroma_db")

def chat():
    print("Cybersecurity RAG Agent")
    print("Type your question or 'exit' to quit.\n")

    while True:
        user_query = input("Ask: ")
        if user_query.strip().lower() in {"exit", "quit"}:
            print("Exiting. Stay secure!")
            break

        try:
            result = graph.invoke({"query": user_query})  
            print("\nAgentic RAG Response:\n")
            print(result["response"])
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    

    chat()
