import sys
import os
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

def run_verification():
    print("==================================================")
    print("         PHASE 1 VERIFICATION SUITE              ")
    print("==================================================")
    
    errors = []

    # 1. Verify Existing RAG Modules
    modules_to_test = [
        "src.document_processor",
        "src.text_processor",
        "src.metadata_extractor",
        "src.embeddings",
        "src.vector_store",
        "src.resume_rag",
        "src.job_matcher",
    ]

    for mod_name in modules_to_test:
        try:
            __import__(mod_name)
            print(f"[OK] Successfully imported: {mod_name}")
        except Exception as e:
            err_msg = f"[FAIL] Failed to import {mod_name}: {e}"
            print(err_msg)
            errors.append(err_msg)

    # 2. Verify OpenRouter Configuration Helper
    try:
        from src.llm_config import get_openrouter_llm
        llm = get_openrouter_llm()
        print(f"[OK] Successfully initialized OpenRouter LLM helper (Model: {llm.model_name}, Base URL: {llm.openai_api_base})")
    except Exception as e:
        err_msg = f"[FAIL] OpenRouter helper initialization failed: {e}"
        print(err_msg)
        errors.append(err_msg)

    # 3. Verify .env.example
    env_example = root_dir / ".env.example"
    if env_example.exists():
        content = env_example.read_text(encoding="utf-8")
        if "OPENROUTER_API_KEY=" in content:
            print("[OK] .env.example correctly defines OPENROUTER_API_KEY")
        else:
            errors.append("[FAIL] OPENROUTER_API_KEY missing from .env.example")
    else:
        errors.append("[FAIL] .env.example file not found")

    # 4. Verify .gitignore
    gitignore = root_dir / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text(encoding="utf-8")
        if ".env" in content:
            print("[OK] .gitignore protects .env file")
        else:
            errors.append("[FAIL] .env not found in .gitignore")

    # 5. Verify No Secret Hardcoding in Code
    for py_file in root_dir.glob("src/*.py"):
        text = py_file.read_text(encoding="utf-8")
        if "sk-or-v1-" in text or "sk-proj-" in text:
            errors.append(f"[FAIL] Hardcoded API key pattern found in {py_file.name}")

    print("==================================================")
    if not errors:
        print("RESULT: ALL PHASE 1 VERIFICATIONS PASSED SUCCESSFULLY!")
    else:
        print(f"RESULT: {len(errors)} VERIFICATION ERRORS FOUND:")
        for err in errors:
            print(f" - {err}")
    print("==================================================")

if __name__ == "__main__":
    run_verification()
