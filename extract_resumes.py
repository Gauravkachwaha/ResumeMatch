from pathlib import Path
import json

from src.document_processor import DocumentProcessor


# Paths
RESUME_DIR = Path("data/selected_resumes")
OUTPUT_DIR = Path("outputs")
OUTPUT_FILE = OUTPUT_DIR / "extracted_resumes.json"


def main():
    processor = DocumentProcessor()

    OUTPUT_DIR.mkdir(exist_ok=True)

    resumes = []

    # Go through each job category
    for category_dir in RESUME_DIR.iterdir():

        if not category_dir.is_dir():
            continue

        category = category_dir.name

        print(f"\nProcessing category: {category}")

        # Find all PDFs
        pdf_files = list(category_dir.glob("*.pdf"))

        for pdf_path in pdf_files:

            print(f"  Processing: {pdf_path.name}")

            try:
                text = processor.extract_text(str(pdf_path))

                resumes.append({
                    "filename": pdf_path.name,
                    "category": category,
                    "path": str(pdf_path),
                    "text": text
                })

                print(f"  Extracted: {len(text)} characters")

            except Exception as e:
                print(f"  ERROR: {e}")


    # Save extracted data
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(resumes, f, indent=2, ensure_ascii=False)

    print("\n====================================")
    print(f"TOTAL RESUMES PROCESSED: {len(resumes)}")
    print(f"OUTPUT FILE: {OUTPUT_FILE}")
    print("====================================")


if __name__ == "__main__":
    main()