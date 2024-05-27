from typing import List
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
class CustomEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        query_embeddings = self.model.encode(text)
        return query_embeddings.tolist()