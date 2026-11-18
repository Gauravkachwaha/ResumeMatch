from pathlib import Path
from collections import Counter


# Root directory containing all resume folders
RESUME_DIR = Path("data/resumes")


def find_resumes():
    """
    Recursively find all PDF resumes inside data/resumes/
    """

    pdf_files = list(RESUME_DIR.rglob("*.pdf"))

    return pdf_files


def main():

    print("=" * 60)
    print("RESUME DATASET INSPECTION")
    print("=" * 60)

    pdf_files = find_resumes()

    print(f"\nTotal PDF resumes found: {len(pdf_files)}")

    if not pdf_files:
        print("\nNo PDF files found.")
        print("Check that your dataset is inside:")
        print("data/resumes/")
        return

    # Count resumes based on their immediate parent folder
    categories = Counter()

    for pdf in pdf_files:
        category = pdf.parent.name
        categories[category] += 1

    print("\n" + "=" * 60)
    print("RESUMES BY CATEGORY")
    print("=" * 60)

    for category, count in categories.most_common():
        print(f"{category:<35} {count:>5}")

    print("\n" + "=" * 60)
    print("SAMPLE FILES")
    print("=" * 60)

    for pdf in pdf_files[:10]:
        print(pdf)

    print("\n" + "=" * 60)
    print("DATASET CHECK")
    print("=" * 60)

    if len(pdf_files) >= 30:
        print("✅ Dataset contains 30+ resumes.")
    else:
        print("⚠️ Dataset contains fewer than 30 resumes.")

    print("\nDone.")


if __name__ == "__main__":
    main()