import os
import json
import chromadb

from dotenv import load_dotenv
from openai import OpenAI

from tickets import tool_create_github_ticket

load_dotenv()

PERSIST_DIR = "vectorstore"
COLLECTION = "support_docs"
EMBED_MODEL = "text-embedding-3-small"

COMPANY_NAME = os.getenv("COMPANY_NAME", "Hyundai")
COMPANY_EMAIL = os.getenv("COMPANY_EMAIL", "ipinfo@hyundai.com")
COMPANY_PHONE = os.getenv("COMPANY_PHONE", "+000 00 000 00 00")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")

# retrieval behavior
TOP_K = 5
NOT_FOUND_DISTANCE = 0.35  # larger distance => less similar


#  Helpers
def get_collection():
    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})


def embed_query(client: OpenAI, text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def format_citations(metadatas: list[dict]) -> str:
    # unique (source,page)
    seen = set()
    items = []
    for m in metadatas:
        src = m.get("source", "unknown")
        page = m.get("page", "?")
        key = (src, page)
        if key not in seen:
            seen.add(key)
            items.append(f"{src} (page {page})")
    if not items:
        return ""
    return "Sources: " + ", ".join(items)


def tool_search_docs(openai_client: OpenAI, question) -> dict:
    col = get_collection()
    q_emb = embed_query(openai_client, question)

    res = col.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    best_dist = dists[0] if dists else 999.0

    return {
        "question": question,
        "best_distance": best_dist,
        "not_found_threshold": NOT_FOUND_DISTANCE,
        "chunks": [
            {"text": d, "source": (m or {}).get("source"), "page": (m or {}).get("page"), "distance": dist}
            for d, m, dist in zip(docs, metas, dists)
        ],
        "citations": [{"source": (m or {}).get("source"), "page": (m or {}).get("page")} for m in metas],
    }


#  Tool calling loop
def run_agent(openai_client: OpenAI, messages: list[dict]) -> str:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "Search in the internal PDF knowledge base and return relevant chunks with sources and pages.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "User question to search for."}
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_ticket",
                "description": "Create a support ticket when the answer is not found or user asks to create a ticket.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "email", "title", "description"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    while True:
        resp = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools,
        )

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        # final answer
        if not tool_calls:
            return msg.content or ""

        # tool calls
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in tool_calls],
            }
        )

        for tc in tool_calls:
            fn = tc.function.name
            args = json.loads(tc.function.arguments or "{}")

            if fn == "search_docs":
                out = tool_search_docs(openai_client, args["question"])
            elif fn == "create_ticket":
                out = tool_create_github_ticket(
                    name=args["name"],
                    email=args["email"],
                    title=args["title"],
                    description=args["description"],
                )
            else:
                out = {"ok": False, "error": f"Unknown tool: {fn}"}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(out),
                }
            )
