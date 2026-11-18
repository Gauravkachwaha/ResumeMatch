from pathlib import Path
import pymupdf
import shutil


SOURCE_DIR = Path("data/resumes")
TARGET_DIR = Path("data/selected_resumes")

TARGET_COUNT = 50


def has_text(pdf_path):
    """
    Check whether a PDF contains selectable text.
    """

    try:

        document = pymupdf.open(pdf_path)

        text_length = 0

        for page in document:

            text = page.get_text()

            if text:
                text_length += len(text.strip())

        document.close()

        return text_length >= 100

    except Exception:

        return False


def main():

    print("=" * 70)
    print("SEARCHING FOR TEXT-BASED RESUMES")
    print("=" * 70)

    pdf_files = list(
        SOURCE_DIR.rglob("*.pdf")
    )

    print(
        f"\nTotal PDFs available: "
        f"{len(pdf_files)}"
    )

    text_resumes = []

    for index, pdf_path in enumerate(
        pdf_files,
        start=1
    ):

        if has_text(pdf_path):

            text_resumes.append(pdf_path)

            print(
                f"\rText PDFs found: "
                f"{len(text_resumes)}",
                end=""
            )

            if len(text_resumes) >= TARGET_COUNT:
                break

    print("\n")

    print(
        f"Found {len(text_resumes)} "
        f"text-based resumes."
    )

    if len(text_resumes) < TARGET_COUNT:

        print(
            "\nWARNING: Could not find "
            f"{TARGET_COUNT} text resumes."
        )

    # -------------------------------------------------
    # Clear old selected dataset
    # -------------------------------------------------

    if TARGET_DIR.exists():

        shutil.rmtree(TARGET_DIR)

    TARGET_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------
    # Copy selected resumes
    # -------------------------------------------------

    for index, pdf_path in enumerate(
        text_resumes,
        start=1
    ):

        # Keep original category
        category = pdf_path.parent.name

        category_dir = (
            TARGET_DIR / category
        )

        category_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        destination = (
            category_dir /
            pdf_path.name
        )

        shutil.copy2(
            pdf_path,
            destination
        )

        print(
            f"{index:02d}. "
            f"{category} / "
            f"{pdf_path.name}"
        )

    print("\n" + "=" * 70)
    print("DATASET CREATION COMPLETE")
    print("=" * 70)

    print(
        f"\nSelected resumes: "
        f"{len(text_resumes)}"
    )

    print(
        f"Location: "
        f"{TARGET_DIR}"
    )


if __name__ == "__main__":
    main()