import os
import pickle
from typing import Dict
import faiss
import numpy as np
import shutil

index_dir="vector_store"
index_path = os.path.join(index_dir,"index.faiss")
meta_path = os.path.join(index_dir, "metadata.pkl")

def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def save_index(index: faiss.Index, data:Dict):
    if os.path.exists("vector_store"):
        shutil.rmtree("vector_store")
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(data, f)

def load_index() -> tuple:
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("FAISS index or metadata not found.")
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        data = pickle.load(f)
    return index, data
