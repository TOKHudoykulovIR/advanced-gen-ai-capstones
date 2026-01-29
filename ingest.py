from dotenv import load_dotenv # load environment variables from .env
from pypdf import PdfReader # read PDF files
from openai import OpenAI # OpenAI client

import chromadb # vector database
import tiktoken # tokenize text by tokens
import hashlib # create unique IDs for chunks
import os # work with folders and files
import re # text cleaning (regex)


load_dotenv()  # load variables from .env file

DOCS_DIR = "files" # folder with PDF documents
PERSIST_DIR = "vectorstore" # folder where vectors will be stored
COLLECTION = "support_docs" # collection name in ChromaDB
EMBED_MODEL = "text-embedding-3-small" # embedding model name


def clean_text(text) -> str:
    """
    Cleans text from extra spaces and invisible symbols
    """
    text = (text or "").replace("\x00", " ")  # remove null bytes
    text = re.sub(r"\s+", " ", text)  # replace many spaces with one
    return text.strip()  # remove leading/trailing spaces


def create_id(text) -> str:
    """
    Creates a unique ID for each text chunk
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_by_tokens(text, max_tokens=260, overlap=40) -> list:
    """
    Splits text into chunks based on token count
    overlap -> how many tokens from the previous chunk we repeat in the next chunk
    """
    encoder = tiktoken.get_encoding("cl100k_base")  # tokenizer
    tokens = encoder.encode(text)  # convert text to tokens

    chunks = []
    step = max_tokens - overlap
    i = 0

    # loop through tokens
    while i < len(tokens):
        chunk_tokens = tokens[i:i + max_tokens]  # take part of tokens
        chunks.append(encoder.decode(chunk_tokens))  # convert tokens back to text
        i += step

    return chunks


def read_pdf_pages(path: str) -> list:
    """
    Reads text from each page of a PDF
    Returns list of (page_number, text)
    """
    reader = PdfReader(path)
    pages = []

    # go through each page
    for index, page in enumerate(reader.pages):
        text = clean_text(page.extract_text())  # extract text from page

        if text:
            pages.append((index + 1, text))  # pages start from 1

    return pages


def get_collection():
    """
    Creates or loads ChromaDB collection
    """
    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    return client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})


def embed_texts(client: OpenAI, texts: list) -> list:
    """
    Converts text chunks into vector embeddings
    """
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]  # return list of vectors


def ingest():
    """
    Main ingestion function
    """
    os.makedirs(DOCS_DIR, exist_ok=True)

    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]  # get all PDF files

    if not pdf_files:
        raise RuntimeError("No PDF files found in /files")

    collection = get_collection()
    openai_client = OpenAI()

    ids = []
    documents = []
    metadata = []

    # loop through each PDF
    for filename in pdf_files:
        file_path = os.path.join(DOCS_DIR, filename)

        # read PDF pages
        for page_number, text in read_pdf_pages(file_path):
            chunks = split_by_tokens(text)  # split page text into chunks

            for index, chunk in enumerate(chunks):
                chunk = clean_text(chunk)

                if not chunk:
                    continue

                uid = create_id(f"{filename}-{page_number}-{index}")  # create unique ID

                ids.append(uid)
                documents.append(chunk)

                # store source info for citations
                metadata.append({
                    "source": filename,
                    "page": page_number
                })

    # upload data in batches
    batch_size = 128

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_meta = metadata[i:i + batch_size]

        embeddings = embed_texts(openai_client, batch_docs)

        # save vectors to ChromaDB
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=embeddings
        )

    print(f"Ingested {len(documents)} chunks from {len(pdf_files)} PDFs")

if __name__ == "__main__":
    ingest()
