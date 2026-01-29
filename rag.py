import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
VECTOR_DIR = "vectorstore"
COLLECTION = 'support_docs'

def embed_query(query):
    res = client.embeddings.create(model="text-embedding-3-small", input=[query])
    return res.data[0].embedding


def retrieve(query, k: int = 6):
    chroma = chromadb.PersistentClient(path=VECTOR_DIR)  # connect to Chroma DB on disk
    col = chroma.get_collection(COLLECTION)  # open collection where ingest.py stored chunks
    q_emb = embed_query(query)  # embed the question

    res = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({
            "text": doc,
            "meta": meta,
            "distance": float(dist),
        })
    hits.sort(key=lambda x: x["distance"])
    return hits

def build_context(hits):
    blocks = []
    for i, h in enumerate(hits, start=1):
        src = h["meta"].get("source", "unknown")
        page = h["meta"].get("page", None)

        # If pdf page exists -> show "file p.N", else just show filename
        cite = f"{src} p.{page}" if page else f"{src}"

        blocks.append(f"[{i}] SOURCE: {cite}\n{h['text']}")
    return "\n\n".join(blocks)


def is_answer_found(hits, threshold: float = 0.35):
    if not hits:
        return False
    return hits[0]["distance"] <= threshold


# quick manual test (run: python rag.py)
if __name__ == "__main__":
    q = "What battery type i need for battery replacement for Remote key"
    hits = retrieve(q, k=5)
    print("Top distances:", [h["distance"] for h in hits[:3]])
    print("\n--- CONTEXT PREVIEW ---\n")
    print(build_context(hits[:2]))
