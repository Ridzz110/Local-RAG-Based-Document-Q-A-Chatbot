
# RAG-Based Document Chatbot [![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/ridzzzzzzzzzzzzzz/local-rag-based-document-qa-chatbot)

A Retrieval-augmented generation (RAG) chatbot for querying your own documents. This project allows you to upload documents (PDF, DOCX, TXT), prepare them, and interact with them. Works great for research and study session ;)

## Features

+ Supports multiple document formats: PDF, DOCX, TXT.

+ Documents are chunked and stored in a vector store (FAISS).

+ Uses all-MiniLM-L6-v2 embeddings for document chunks.

+ Retrieves relevant chunks based on queries via similarity search.

+ Uses Groq for LLM integration (llama-3.3-70b-versatile)
  
+ App metrics and monitoring logs to keep track of performance.
  
+ Demo deployed on Hugging Face spaces

+ Evaluated on RAGAS Framework resulting in perfect scores

## How It Works

### Document Preparation

+ Upload your documents in DOCX, PDF, or TXT format.

+ Documents are chunked along with source information and IDs.

### Embedding Creation

+ Each chunk is converted into vector embeddings using the all-MiniLM-L6-v2 model.

+ Embeddings are stored in a FAISS vector store for efficient retrieval.

### Query Retrieval

+ When a query is entered, a similarity search retrieves the most relevant chunks from the vector store.

### Answer Generation

+ Retrieved chunks are passed as context to a prompt.

+ The prompt is sent to the via an API call to llama-3.3-70b-versatile, which generates the final answer.

## Requirements

All Python dependencies are listed in requirements.txt.

### Key packages include:

+ gradio for the UI

+ faiss for vector store

+ sentence-transformers for embeddings

+ Groq for LLM integration
  
+ HuggingFace spaces for deployement

## Evaluation

RAG pipeline evaluated using the **RAGAS framework** on a sample document with domain-specific questions.

| Metric | Score |
|---|---|
| Faithfulness | 1.00 |
| Context Recall | 1.00 |
| Factual Correctness | 0.87 |
| Semantic Similarity | 0.89 |

> Faithfulness and Context Recall both scored a perfect 1.0 — the pipeline never hallucinated and consistently retrieved relevant context.

Run evaluation on your own documents:
1. Index your documents by running the app and uploading them first
2. Update the `test_cases` list in `src/eval.py` with questions and ground truths relevant to your documents
3. Then run:
```
python src/eval.py
```

## Setup Instructions

Clone the repository:
```
git clone https://github.com/Ridzz110/RAG-Based-Document-Chatbot.git
cd RAG-Based-Document-Chatbot
```
Set up Python virtual environment:
```
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```
Install Python dependencies:
```
pip install -r requirements.txt
```

Run the Gradio interface:
```
python src/app.py
```

This will start the RAG chatbot UI.


## Future Improvements.
+ Implement docker.
+ User Sessions persistance.

## Author

**Rida Batool**
AI Undergraduate | Aspiring ML Engineer

GitHub: [https://github.com/Ridzz110](https://github.com/Ridzz110)
LinkedIn: [https://www.linkedin.com/in/ridabatool110](https://www.linkedin.com/in/ridabatool110)
