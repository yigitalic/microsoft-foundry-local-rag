# 🤖 Local RAG AI Assistant with Microsoft Foundry Local

A fully offline, privacy-first Retrieval-Augmented Generation (RAG) knowledge assistant running entirely on your local device. 

Developed during the **Microsoft Summer School / Internship** program, this project utilizes **Microsoft Foundry Local** for local model inference and **SQLite** as a serverless local vector store.

---

## 🌟 Key Features

*   **100% Offline & Private:** Runs entirely on-device with zero cloud dependencies, zero external network calls, and zero API costs.
*   **On-Device Models:**
    *   **Embedding Model:** `qwen3-embedding-0.6b` (produces 1024-dimensional semantic vectors).
    *   **LLM (Generative Chat Model):** `phi-4-mini` (3.8B parameters, optimized for on-device reasoning and tool calling).
*   **Automatic Hardware Acceleration:** Automatically utilizes local hardware (CPU, GPU, or NPU) via ONNX Runtime GenAI.
*   **Modern Streamlit Web UI:**
    *   Sleek dark theme UI with custom premium aesthetics.
    *   **Drag-and-Drop Ingestion:** Upload new `.txt` files directly from the UI, chunk them, generate embeddings, and write them to SQLite in real-time.
    *   **Source Citations:** Transparently view similarity scores and matched context chunks for every response.
*   **Smart Memory Management:** Models are loaded into memory only during inference and automatically unloaded (`unload()`) afterwards to conserve RAM/VRAM.

---

## 📐 System Architecture

All components run on a single machine:

```
[Local Document (.txt)]
       │
       ▼ (ingestion.py / UI Uploader)
[Text Chunking (by paragraph)]
       │
       ▼ (qwen3-embedding-0.6b)
[Vector Embeddings (1024 d)]
       │
       ▼
[SQLite Database (knowledge_base.db)]
       ▲
       │ (retrieval.py - Cosine Similarity)
[User Search Query] ◄───► [Streamlit UI Chat] ───► [Phi-4-Mini LLM] ───► [Final Citated Answer]
```

---

## 📂 Project Structure

*   `app_streamlit.py`: The main Streamlit web application interface.
*   `ingestion.py`: Document parser and SQLite database loading script.
*   `retrieval.py`: Semantic search execution (Query Embedding -> SQLite Cosine Similarity match).
*   `database.py`: SQLite initialization, read, and write operations.
*   `vector_search.py`: Local embedding generator and mathematical cosine similarity function.
*   `documents/`: Directory containing source knowledge base text files.
*   `requirements.txt`: Project dependencies list.

---

## 🚀 Getting Started

### Prerequisites
*   **Python 3.11** or higher.
*   **Git** installed.

### Installation & Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yigitalic/microsoft-foundry-local-rag.git
   cd microsoft-foundry-local-rag
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the initial ingestion script** (to load the sample files into SQLite):
   ```bash
   python ingestion.py
   ```

5. **Start the Streamlit application:**
   ```bash
   streamlit run app_streamlit.py
   ```

6. Open `http://localhost:8501` in your browser.

---

## 📊 Evaluation & Testing

The system has been evaluated with targeted local test cases:

*   **Factual Matching:** Questions regarding staj schedules are accurately answered using context from `staj_rehberi.txt` (Cosine similarity ~`0.53`).
*   **Hallucination Prevention:** When asked about questions outside the database, the agent successfully fallbacks to *"I cannot find this information in my database"* rather than fabricating facts.
*   **Performance:** Inference responses are generated locally within ~4-10 seconds on consumer CPU hardware.
