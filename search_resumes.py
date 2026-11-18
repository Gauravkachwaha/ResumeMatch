from src.vector_store import ResumeVectorStore


def main():

    vector_store = ResumeVectorStore()

    job_description = """
    We are looking for a Java developer with experience
    in Spring Boot, REST APIs, MySQL and microservices.
    """

    results = vector_store.search(
        job_description,
        top_k=5
    )

    print("\nTOP MATCHING CHUNKS")
    print("====================")

    for i, document in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][i]

        print(f"\nRank: {i + 1}")
        print(f"Resume: {metadata['filename']}")
        print(f"Category: {metadata['category']}")
        print(f"Chunk: {metadata['chunk_index']}")
        print(f"Text: {document}")


if __name__ == "__main__":
    main()