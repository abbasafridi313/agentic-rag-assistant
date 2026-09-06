import os
from dotenv import load_dotenv
from supabase import create_client
from fastembed import TextEmbedding

load_dotenv()

# Supabase connect
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

# Free embedding model load karo (pehli baar chalao to download hoga)
print("Loading embedding model...")
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Simple chunking function (500 characters ke chunks, thora overlap ke saath)
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def ingest_document(title, content):
    # 1. Document save karo
    doc_response = supabase.table("documents").insert({
        "title": title,
        "content": content
    }).execute()
    document_id = doc_response.data[0]["id"]
    print(f"Document saved: {title} (id: {document_id})")

    # 2. Chunks banao
    chunks = chunk_text(content)
    print(f"Created {len(chunks)} chunks")

    # 3. Har chunk ka embedding banao aur save karo
    embeddings = list(model.embed(chunks))
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        supabase.table("chunks").insert({
            "document_id": document_id,
            "content": chunk,
            "embedding": embedding.tolist()
        }).execute()
        print(f"Chunk {i+1}/{len(chunks)} saved")

    print("Ingestion complete!")

if __name__ == "__main__":
    sample_text = """
    Agentic RAG systems combine retrieval-augmented generation with autonomous decision-making.
    Unlike traditional RAG, an agentic system can decide when to retrieve more information,
    which tools to use, and how to verify its own answers before responding to the user.
    This makes the system more reliable for complex, multi-step questions.
    """
    ingest_document("Intro to Agentic RAG", sample_text)