import json

from src.vector_store import ResumeVectorStore


INPUT_FILE = "outputs/processed_resumes.json"


def main():

    # Load processed resumes
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        resumes = json.load(f)

    print(f"Loaded {len(resumes)} resumes.")

    # Create vector store
    vector_store = ResumeVectorStore()

    # Add chunks
    vector_store.add_resumes(resumes)


if __name__ == "__main__":
    main()