import json
from pathlib import Path

from src.text_processor import TextProcessor


INPUT_FILE = Path("outputs/extracted_resumes.json")
OUTPUT_FILE = Path("outputs/processed_resumes.json")


def main():

    processor = TextProcessor()

    # Load extracted resumes
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        resumes = json.load(f)

    processed_resumes = []

    for resume in resumes:

        cleaned_text = processor.clean_text(resume["text"])

        chunks = processor.chunk_text(
            cleaned_text,
            chunk_size=500,
            overlap=100
        )

        processed_resumes.append({
            "filename": resume["filename"],
            "category": resume["category"],
            "path": resume["path"],
            "text": cleaned_text,
            "chunks": chunks
        })

    # Save processed resumes
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            processed_resumes,
            f,
            indent=2,
            ensure_ascii=False
        )

    total_chunks = sum(
        len(resume["chunks"])
        for resume in processed_resumes
    )

    print("====================================")
    print(f"RESUMES PROCESSED: {len(processed_resumes)}")
    print(f"TOTAL CHUNKS: {total_chunks}")
    print(f"OUTPUT: {OUTPUT_FILE}")
    print("====================================")


if __name__ == "__main__":
    main()