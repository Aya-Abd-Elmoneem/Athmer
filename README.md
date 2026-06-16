<div align="center">

<img src="Frontend/public/athmer.png" alt="Athmer Logo" width="120" />

# Athmer

### An AI-Powered Agricultural Intelligence Platform for the Egyptian Agricultural Bank

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.124-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC244C?style=flat)](https://qdrant.tech)
[![MongoDB](https://img.shields.io/badge/MongoDB-7-47A248?style=flat&logo=mongodb&logoColor=white)](https://mongodb.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Technologies Used](#-technologies-used)
- [Requirements](#-requirements)
- [Installation & Setup](#-installation--setup)
- [Project Structure](#-project-structure)
- [API](#-api)
- [RAG Pipeline](#-rag-pipeline)
- [Contributing](#-contributing)

---

## 🌾 Overview

**Athmer** is a multi-stakeholder AI platform serving the Egyptian agricultural ecosystem:

| User | Role | Interface |
|------|------|----------|
| 🌱 **Farmer** | Crop disease detection via images + bank service inquiries | Chatbot + Image Upload |
| 🏦 **Agricultural Bank** | Banking services + RAG-powered customer support | Dashboard + Chatbot |
| 🏛️ **Ministry** | Agricultural monitoring and crop analytics | Analytics Dashboard |

The system is built on a **RAG (Retrieval-Augmented Generation)** pipeline that retrieves knowledge from agricultural bank documents and generates accurate Arabic responses.

---

## ✨ Features

### 🤖 Advanced RAG Pipeline
- **Intent Detection** — detects user intent before entering RAG (greetings, out-of-scope queries, banking questions)
- **Query Rewriting** — converts Egyptian Arabic into formal Arabic for better retrieval
- **Query Expansion** — generates alternative query formulations for improved retrieval
- **Similarity Thresholding** — rejects low-confidence results instead of returning hallucinated answers

### 🔌 Multi-Provider Architecture
- **Generation:** Groq (llama-3.1-8b-instant) / OpenAI / Ollama
- **Embedding:** BAAI/bge-m3 / SentenceTransformer / Cohere multilingual
- **Vector DB:** Qdrant (local or cloud)

### 🌍 Full Arabic Support
- Arabic + Egyptian dialect system prompts
- Arabic-aware chunking strategy
- Fully RTL UI support

---

## 🏗️ System Architecture


Frontend (React)
│ HTTP (Axios)
▼
Backend (FastAPI - NLP Controller)
│
├── Intent Detection
├── Query Rewriting
├── Query Expansion
├── Vector Search (Qdrant)
└── LLM Generation (Groq)
│
▼
Databases:

Qdrant (Vector Store)
MongoDB (Metadata)
File Storage

---

## 🛠️ Technologies Used

### Backend
| Tech | Version | Purpose |
|------|--------|--------|
| FastAPI | 0.124 | API Framework |
| Motor | 3.7 | Async MongoDB Driver |
| Qdrant Client | 1.10 | Vector Database |
| SentenceTransformers | 5.0 | BGE-M3 Embeddings |
| Groq | latest | LLM Generation |
| LangChain | 1.3 | Document Processing |
| PyMuPDF | 1.27 | PDF Parsing |
| python-docx | 1.2 | Word Parsing |

### Frontend
| Tech | Purpose |
|------|--------|
| React 18 + Vite | UI Framework |
| React Router v6 | Routing |
| Zustand | State Management |
| Axios | API Calls |
| Tajawal / Cairo | Arabic Fonts |

### Infrastructure
| Tech | Purpose |
|------|--------|
| MongoDB 7 | Document Storage |
| Qdrant | Vector Search |
| Docker Compose | Containerization |

---

## 📦 Requirements

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Groq API Key (free from https://console.groq.com)

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/athmer.git
cd athmer
2. Start MongoDB
cd Backend/docker
docker compose up -d
3. Backend Setup
cd Backend/src

cp .env.example .env

Configure .env:

GENERATION_BACKEND=GROQ
EMBEDDING_BACKEND=SENTENCE_TRANSFORMER

GROQ_API_KEY=your_api_key
GENERATION_MODEL_ID=llama-3.1-8b-instant

EMBEDDING_MODEL_ID=BAAI/bge-m3
EMBEDDING_MODEL_SIZE=1024

INPUT_DEFAULT_MAX_CHARACTERS=4000
GENERATION_DEFAULT_MAX_TOKENS=600
GENERATION_DEFAULT_TEMPERATURE=0.1

MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=Athmer

VECTOR_DB_BACKEND=QDRANT
VECTOR_DB_PATH=assets/database/qdrant
VECTOR_DB_DISTANCE_METHOD=cosine

PRIMARY_LANG=ar
DEFAULT_LANG=ar

Install dependencies:

pip install -r requirements.txt

Run server:

uvicorn main:app --reload --host 0.0.0.0 --port 8000

⚠️ On first run, the system will download the BAAI/bge-m3 model (~2.3GB).

4. Frontend Setup
cd Frontend
npm install
npm run dev

Frontend runs at:

http://localhost:5173
5. Build Vector Index
POST /data/upload/{project_id}
POST /data/process/{project_id}
POST /nlp/index/push/{project_id}
📁 Project Structure
Backend/
├── docker/
├── src/
│   ├── main.py
│   ├── controllers/
│   │   ├── NLPController.py
│   ├── stores/
│   │   ├── llm/
│   │   ├── vectordb/
│   ├── routes/
│   └── assets/

Frontend/
├── public/
├── src/
│   ├── pages/
│   ├── api/
│   └── store/
📡 API

Base URL:

http://localhost:8000/api/v1
Endpoints
Method	Endpoint	Description
GET	/	Health check
POST	/data/upload/{project_id}	Upload file
POST	/data/process/{project_id}	Process documents
POST	/nlp/index/push/{project_id}	Build index
POST	/nlp/index/answer/{project_id}	Ask chatbot
Example Request
curl -X POST "http://localhost:8000/api/v1/nlp/index/answer/3" \
-H "Content-Type: application/json" \
-d '{"query": "What are the requirements for seasonal crop loans?", "limit": 5}'
🔄 RAG Pipeline
User Query
    │
    ▼
Intent Detection
    │
    ├── Greeting → Direct Response
    ├── Out of Scope → Reject
    └── Valid Query
          │
          ▼
Query Rewriting
          │
          ▼
Query Expansion
          │
          ▼
Vector Search (Qdrant)
          │
          ▼
LLM Generation (Groq)
          │
          ▼
Final Answer
🔐 Role System
Email Pattern	Role
farmer@...	Farmer
bank@...	Bank Employee
ministry@...	Ministry Employee
🤝 Contributing
git checkout -b feature/your-feature
git commit -m "feat: add new feature"
git push origin feature/your-feature
Commit Rules
feat: new feature
fix: bug fix
docs: documentation
refactor: code restructuring
perf: performance improvement
📄 License

This project is licensed under the MIT License.

<div align="center">

Built with 🌾 for Egyptian agriculture

Athmer

</div> ```
