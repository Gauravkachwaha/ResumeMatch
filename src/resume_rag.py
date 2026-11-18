import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.document_processor import DocumentProcessor
from src.text_processor import TextProcessor
from src.metadata_extractor import MetadataExtractor
from src.vector_store import ResumeVectorStore


class ResumeRAGPipeline:
    """
    Part A — Document Processing Pipeline for Resume RAG system.
    Orchestrates:
    1. Multi-format file loading (PDF, DOCX, TXT)
    2. Intelligent section-aware chunking
    3. Structured metadata extraction (candidate name, skills, experience, education)
    4. Vector store persistence in ChromaDB
    """

    def __init__(self, db_path: str = "chroma_db"):
        self.doc_processor = DocumentProcessor()
        self.text_processor = TextProcessor()
        self.metadata_extractor = MetadataExtractor()
        self.vector_store = ResumeVectorStore(db_path=db_path)

    def process_resume(self, file_path: str, category: str = "Uncategorized") -> Dict[str, Any]:
        """
        Process a single resume file end-to-end.
        """
        path = Path(file_path)
        raw_text = self.doc_processor.extract_text(str(path))
        cleaned_text = self.text_processor.clean_text(raw_text)
        metadata = self.metadata_extractor.extract_metadata(cleaned_text, filename=path.name)
        chunks = self.text_processor.chunk_text(cleaned_text)

        return {
            "filename": path.name,
            "category": category,
            "path": str(path),
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "metadata": metadata,
            "chunks": chunks
        }

    def ingest_directory(
        self,
        resumes_dir: str = "data/selected_resumes",
        output_json: Optional[str] = "outputs/processed_resumes.json",
        reset_db: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Scan directory for resumes, process them, save output JSON, and index into ChromaDB.
        """
        dir_path = Path(resumes_dir)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {resumes_dir}")

        processed_resumes = []
        files_to_process = []

        # Recursively find all supported resume files
        for ext in ["*.pdf", "*.docx", "*.doc", "*.txt"]:
            files_to_process.extend(list(dir_path.rglob(ext)))

        print(f"Found {len(files_to_process)} resume files to process.")

        for file_p in files_to_process:
            category = file_p.parent.name if file_p.parent != dir_path else "Uncategorized"
            try:
                processed = self.process_resume(str(file_p), category=category)
                processed_resumes.append(processed)
                print(f"  [OK] Processed: {file_p.name} [{category}] ({len(processed['chunks'])} chunks)")
            except Exception as e:
                print(f"  [FAIL] Failed to process {file_p.name}: {e}")

        # Save processed metadata JSON artifact
        if output_json:
            out_path = Path(output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(processed_resumes, f, indent=2, ensure_ascii=False)
            print(f"\nSaved processed resume artifacts to {output_json}")

        # Index into ChromaDB
        if reset_db:
            self.vector_store.reset_collection()
        
        self.vector_store.add_resumes(processed_resumes)
        return processed_resumes


if __name__ == "__main__":
    pipeline = ResumeRAGPipeline()
    pipeline.ingest_directory()
