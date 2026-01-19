from typing import List, Dict
import numpy as np  
from sentence_transformers import SentenceTransformer 

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32

def load_embedding_model() -> SentenceTransformer:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model

def embed_document(documents: List[Dict], model:SentenceTransformer) -> Dict[str, List]:

    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    embeddings = model.encode(
        texts,
        batch_size = BATCH_SIZE,
        show_progress_bar = True,
        convert_to_numpy = True,
        normalize_embeddings = True

    )
    return{
        "texts": texts,
        "metadatas": metadatas,
        "embeddings": embeddings
    }

