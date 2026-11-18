from pathlib import Path
import shutil
import random


# Original Kaggle dataset
SOURCE_DIR = Path("data/resumes")

# Our curated dataset
TARGET_DIR = Path("data/selected_resumes")

# Number of resumes we want from each category
RESUMES_PER_CATEGORY = 5

# Categories selected for our project
CATEGORIES = {
    "Python Developer": "Python Developer",
    "Data Science": "Data Science",
    "Java Developer": "Java Developer resumes",
    "React Developer": "React Developer",
    "SQL Developer": "SQL Developer resumes",
    "DevOps Engineer": "DevOps Engineer",
    "Testing": "Testing",
    "Business Analyst": "Business Analyst resumes",
    "SAP Developer": "SAP Developer",
    "Network Security": "Network Security Engineer resumes",
}


def find_category_folder(category_name):
    """
    Search recursively for a folder with the given name.
    """

    matches = [
        path
        for path in SOURCE_DIR.rglob("*")
        if path.is_dir() and path.name.lower() == category_name.lower()
    ]

    if not matches:
        return None

    return matches[0]


def select_resumes():

    random.seed(42)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    total_selected = 0

    print("=" * 60)
    print("CREATING CURATED RESUME DATASET")
    print("=" * 60)

    for project_category, source_category in CATEGORIES.items():

        source_folder = find_category_folder(source_category)

        if source_folder is None:
            print(f"\n❌ Category not found: {source_category}")
            continue

        pdf_files = list(source_folder.glob("*.pdf"))

        if len(pdf_files) < RESUMES_PER_CATEGORY:
            print(
                f"\n⚠️ {source_category}: "
                f"only {len(pdf_files)} resumes available"
            )
            selected = pdf_files
        else:
            selected = random.sample(
                pdf_files,
                RESUMES_PER_CATEGORY
            )

        # Create category folder
        target_category = TARGET_DIR / project_category
        target_category.mkdir(parents=True, exist_ok=True)

        for pdf in selected:

            destination = target_category / pdf.name

            shutil.copy2(pdf, destination)

            total_selected += 1

        print(
            f"\n{project_category:<25} "
            f"{len(selected)} resumes"
        )

    print("\n" + "=" * 60)
    print(f"TOTAL RESUMES SELECTED: {total_selected}")
    print("=" * 60)


if __name__ == "__main__":
    select_resumes()