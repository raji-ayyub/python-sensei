
# Python learning assistant API

A FastAPI-powered backend that answers tax-related questions using an AI assistant with document retrieval (RAG) and conversational memory.

The assistant reads tax documents (PDFs), stores embeddings, retrieves relevant context, and responds using an LLM with tool support and persistent conversation history.

---

## Features

- FastAPI REST API
- AI-powered question answering
- PDF document ingestion
- Vector search using Chroma
- Conversational memory with SQLite (LangGraph checkpointer)
- Tool-based retrieval (RAG)
- Session-based conversation tracking
- CORS enabled (frontend ready)

---

## Tech Stack

- **FastAPI** – API framework
- **LangChain + LangGraph** – Agent & workflow orchestration
- **OpenAI** – LLM & embeddings
- **ChromaDB** – Vector storage
- **SQLite** – Conversation memory
- **pdfplumber** – PDF text extraction

---

## Project Structure

```

.
├── main.py                # FastAPI entry point
├── app.py                 # TaxAssistant logic (LLM, RAG, graph)
├── files/
│   ├── *.pdf              # Tax documents (input)
│   └── docstore/          # Chroma vector database
├── tax_files/
│   └── conversations.db   # SQLite conversation memory
├── .env                   # Environment variables
├── requirements.txt
└── README.md

````

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/raji-ayyub/rag_python_informant
cd rag_python_informant
````

---

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Environment variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
HOST=0.0.0.0
PORT=8000
```

---

### 5. Add PDF documents

Place your tax-related PDF files inside the `files/` directory.

These documents will be:

* Loaded
* Chunked
* Embedded
* Stored in ChromaDB automatically

---

## Running the API

```bash
python main.py
```

The API will be available at:

```
http://localhost:8000
```

---

## API Endpoints

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "healthy",
  "service": "Python guru is up and running"
}
```

---

### Ask a Question

```http
POST /ask
```

Request body:

```json
{
  "question": "I need to learn python, give me a start point"
}
```

Response:

```json
{
  "success": true,
  "user_id": "user_1",
  "question": "What is fastapi",
  "answer": "..."
}
```

Each `user_id` maintains its own conversation memory.

---

## How It Works (High Level)

1. PDFs are loaded and converted into text
2. Text is chunked and embedded
3. Embeddings are stored in ChromaDB
4. A LangGraph agent:

   * Decides when to use tools
   * Retrieves relevant document context
   * Answers using the LLM
5. Conversations are persisted using SQLite

---

## Notes

* SQLite is used for durability in development and small deployments
* The assistant instance is created once and reused
* CORS is fully open by default (adjust for production)
* Authentication is currently disabled but can be added

---

## Future Improvements

* Postgres checkpointer
* Authentication & user management
* Streaming responses
* Better retrieval ranking
* Deployment with Docker

