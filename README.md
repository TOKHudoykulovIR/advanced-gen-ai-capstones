# Customer Support RAG System

This project is a Customer Support web application that answers questions from PDF documents and creates support tickets when an answer is not found.

## Step 1. Fill environment variables

`.env` file in the project root, fill it:

## Step 2. Install dependencies

pip install -r requirements.txt

## Step 4. Run ingestion

This step reads PDFs, splits them into chunks, creates embeddings, and stores them in ChromaDB.

python ingest.py

After this step, the `vectorstore/` folder will be created.

## Step 5. Run chat application

Start the web chat interface:

streamlit run chat.py

Open in browser:
http://localhost:8501


## Usage example:
<img width="833" height="937" alt="image_2026-01-29_21-24-34" src="https://github.com/user-attachments/assets/8fc3b782-1331-41cd-904e-b59813cd73bb" />

---
---
---

<img width="1237" height="124" alt="image_2026-01-29_21-28-15" src="https://github.com/user-attachments/assets/02cd9413-a3ee-429c-b8c4-bf97b587b690" />
