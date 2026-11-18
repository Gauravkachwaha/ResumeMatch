import os
from typing import List, Union
from dotenv import load_dotenv

load_dotenv()


class EmbeddingGenerator:
    """
    Unified embedding generator supporting local HuggingFace SentenceTransformers (default),
    OpenAI, and Cohere embeddings based on environment variables.
    """

    def __init__(self, provider: str = None, model_name: str = None):
        self.provider = provider or os.getenv("EMBEDDING_PROVIDER", "sentence-transformers").lower()
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.cohere_key = os.getenv("COHERE_API_KEY", "").strip()

        # Resolve provider based on keys if default requested but keys exist
        if self.provider == "openai" and self.openai_key:
            import openai
            self.client = openai.OpenAI(api_key=self.openai_key)
            self.model_name = model_name or "text-embedding-3-small"
        elif self.provider == "cohere" and self.cohere_key:
            import cohere
            self.client = cohere.Client(api_key=self.cohere_key)
            self.model_name = model_name or "embed-english-v3.0"
        else:
            # Default to local sentence-transformers
            from sentence_transformers import SentenceTransformer
            self.provider = "sentence-transformers"
            self.model = SentenceTransformer(self.model_name)

    def encode(self, text: str) -> List[float]:
        """
        Encode a single string into an embedding vector.
        """
        if not text or not text.strip():
            # Return zero vector of appropriate dimension (384 for MiniLM)
            dim = 384 if "MiniLM" in self.model_name else 1536
            return [0.0] * dim

        if self.provider == "openai":
            response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
            return response.data[0].embedding
        elif self.provider == "cohere":
            response = self.client.embed(
                texts=[text],
                model=self.model_name,
                input_type="search_document"
            )
            return response.embeddings[0]
        else:
            return self.model.encode(text, convert_to_numpy=True).tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a batch of strings into embedding vectors.
        """
        if not texts:
            return []

        cleaned_texts = [t if t and t.strip() else " " for t in texts]

        if self.provider == "openai":
            response = self.client.embeddings.create(
                input=cleaned_texts,
                model=self.model_name
            )
            return [item.embedding for item in response.data]
        elif self.provider == "cohere":
            response = self.client.embed(
                texts=cleaned_texts,
                model=self.model_name,
                input_type="search_document"
            )
            return response.embeddings
        else:
            embeddings = self.model.encode(cleaned_texts, convert_to_numpy=True)
            return embeddings.tolist()
