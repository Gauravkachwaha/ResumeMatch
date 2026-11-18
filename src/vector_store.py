import os
import sys
from typing import List, Dict, Any, Optional

# Anaconda / Windows DLL PATH fix for sqlite3 & chromadb
if sys.platform == "win32":
    for p in [r"C:\Anaconda\DLLs", r"C:\Anaconda\Library\bin", r"C:\Anaconda"]:
        if os.path.exists(p):
            try:
                os.add_dll_directory(p)
            except Exception:
                pass
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

import chromadb


from src.embeddings import EmbeddingGenerator


class ResumeVectorStore:
    """
    Persistent ChromaDB vector store for resume chunks and structured candidate metadata.
    """

    def __init__(self, db_path: str = "chroma_db", collection_name: str = "resumes"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_generator = EmbeddingGenerator()

        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def reset_collection(self):
        """
        Delete and recreate the resumes collection.
        """
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_resumes(self, processed_resumes: List[Dict[str, Any]]):
        """
        Embed and store processed resumes with structured metadata into ChromaDB.
        """
        documents = []
        embeddings = []
        ids = []
        metadatas = []

        counter = 0

        for resume in processed_resumes:
            filename = resume.get("filename", "")
            category = resume.get("category", "")
            resume_path = resume.get("path", "")
            meta = resume.get("metadata", {})

            candidate_name = meta.get("candidate_name", "Candidate")
            skills_str = ", ".join(meta.get("skills", []))
            years_exp = float(meta.get("total_years_experience", 0.0))
            education = meta.get("highest_education", "Bachelor's")

            chunks = resume.get("chunks", [])

            for idx, chunk in enumerate(chunks):
                if isinstance(chunk, dict):
                    chunk_text = chunk["text"]
                    section = chunk.get("section", "GENERAL")
                else:
                    chunk_text = str(chunk)
                    section = "GENERAL"

                documents.append(chunk_text)
                ids.append(f"{filename}_{section}_{idx}_{counter}")

                metadatas.append({
                    "filename": filename,
                    "category": category,
                    "resume_path": resume_path,
                    "candidate_name": candidate_name,
                    "skills": skills_str,
                    "years_of_experience": years_exp,
                    "highest_education": education,
                    "section": section,
                    "chunk_index": idx
                })

                counter += 1

        if documents:
            # Batch embedding generation
            embeddings = self.embedding_generator.encode_batch(documents)

            # ChromaDB batch add
            batch_size = 500
            for i in range(0, len(documents), batch_size):
                self.collection.add(
                    documents=documents[i:i+batch_size],
                    embeddings=embeddings[i:i+batch_size],
                    ids=ids[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )

        print(f"Indexed {len(documents)} chunks from {len(processed_resumes)} resumes in ChromaDB.")

    def search_vector(self, query: str, top_k: int = 15, where_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform vector similarity search against ChromaDB.
        """
        query_embedding = self.embedding_generator.encode(query)

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = self.collection.query(**query_kwargs)
        return results

    def get_all_records(self) -> Dict[str, Any]:
        """
        Retrieve all documents and metadata from collection (used for BM25 hybrid indexing).
        """
        return self.collection.get(include=["documents", "metadatas"])