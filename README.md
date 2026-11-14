<div align="center">

# 🧠 Nzelu AI

**Context-Grounded Retrieval-Augmented Generation (RAG) Study Companion**

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=googlebard&logoColor=white)](https://ai.google.dev/)
[![Pinecone](https://img.shields.io/badge/Pinecone-000000?style=for-the-badge&logo=pinecone&logoColor=white)](https://www.pinecone.io/)

</div>

---

## 🎯 Overview

Derived from the Nyanja word ***Nzelu***—meaning *Knowledge*—Nzelu AI is a lightweight Retrieval-Augmented Generation (RAG) assistant designed to live directly within a student's or researcher's documentation space. Acting as an empathetic and precise study companion, it extracts domain-specific context from uploaded knowledge bases to deliver contextually grounded, low-latency answers enriched with real-world analogies and case studies—without hallucination.

<div align="center">
  <img src="screenshots/interface.png" alt="Nzelu AI Chatbot Interface" width="85%" />
  <br />
  <em>Nzelu AI Split-Screen Study Workspace</em>
</div>

---

## 🏗️ System Architecture & Engineering Blueprint

Nzelu AI splits the traditional LLM bottleneck into two independent asynchronous pipelines: an **Ingestion Pipeline** and a **Runtime Execution Pipeline**.

<div align="center">
  <img src="screenshots/plan.png" alt="Nzelu AI System Architecture Plan" width="85%" />
  <br />
  <em>End-to-End Ingestion and Retrieval Architecture</em>
</div>

### 1. The Ingestion & Vector Dense Space Pipeline

* **Document Processing:** Uploaded documents are parsed and passed through a structured `RecursiveCharacterTextSplitter`. Chunk sizes are configured to **1,000 tokens** with a **200-token overlap** to preserve semantic context boundaries across text splits.
* **Mathematical Vector Representation:** Text chunks are transformed into dense numerical vectors using Google's `gemini-embedding-001` / `gemini-embedding-2-preview` models (truncated to 768 dimensions).
* **Vector Indexing:** High-dimensional embeddings are indexed into a serverless Pinecone database cluster utilizing **Cosine Similarity** for fast semantic matching.

<div align="center">
  <img src="screenshots/cos1.png" alt="Cosine Similarity Visualization 1" width="46%" />
  &nbsp;&nbsp;
  <img src="screenshots/cos2.png" alt="Cosine Similarity Visualization 2" width="46%" />
  <br />
  <em>Vector Space Distribution & Cosine Similarity Indexing</em>
</div>

### 2. Runtime Retrieval & Execution Pipeline

* **Similarity Score Threshold Filtering:** Unlike naive RAG pipelines that fetch arbitrary data using a blind K-nearest neighbor (K-NN) search, Nzelu AI implements an advanced `similarity_score_threshold` set strictly to `0.3`–`0.4` alongside document metadata filtering. If no document meets this relevance bar, the system cuts off irrelevant noise to prevent the LLM from hallucinating.
* **Stateful Memory Orchestration:** Leveraging `streamlit.session_state`, the execution thread preserves isolated, file-scoped sliding windows of typed message classes (`HumanMessage`, `SystemMessage`, `AIMessage`).
* **Dynamic Context Injection:** The system programmatically serializes the raw page content fetched by the vector store, formats it cleanly into a dynamic runtime system prompt block, and executes an atomic, context-aware inference request via the `gemini-2.5-flash` model.

---

## 🛠️ Tech Stack & Key Technical Decisions

| Component | Technology | Reason |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Eliminates heavy frontend state management overhead, enabling rapid prototyping of stateful reactive components using native session threading. |
| **Orchestrator** | LangChain | Provides a unified, modular abstraction layer with predictable object schemas, ensuring seamless swaps between models and vector database clients. |
| **Vector Database** | Pinecone | Offloads multi-dimensional vector indexing to a fully managed serverless cloud service, maintaining sub-millisecond retrieval speeds without consuming local memory. |
| **LLM & Embeddings** | Google Gemini | Combines `gemini-2.5-flash` for low-latency reasoning with Gemini embeddings for high-precision semantic retrieval. |
| **Language** | Python 3.13 | Leverages the latest native interpreter optimizations, enhanced garbage collection, and faster package execution. |

---

## 🚀 Quick Start

### Prerequisites
Obtain two API keys before starting:
1. **Google Gemini API Key** (from Google AI Studio)
2. **Pinecone API Key** (with a 768-dimension serverless index)

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/joel-musonda/nzeluai-rag.git
   cd nzeluai-rag
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY="your_gemini_api_key"
   PINECONE_API_KEY="your_pinecone_api_key"
   PINECONE_INDEX_NAME="edu-app"
   ```

4. **Run the application**
   ```bash
   python ingestion.py
   python retrieval.py
   streamlit run chatbot_rag.py
   ```