import json
from pathlib import Path


cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Resume RAG & Hybrid Job Matching Engine\n",
            "\n",
            "This notebook demonstrates the complete end-to-end **Resume RAG + Job Matching Engine**.\n",
            "\n",
            "### Pipeline Architecture:\n",
            "1. **Multi-Format Document Ingestion**: Parsing resumes from PDF, DOCX, and TXT formats with OCR fallback.\n",
            "2. **Intelligent Section-Aware Chunking**: Structuring resume text into section boundaries (`EXPERIENCE`, `EDUCATION`, `SKILLS`, `PROJECTS`, `CERTIFICATIONS`).\n",
            "3. **Structured Metadata Extraction**: Extracting candidate names, recognized technical skills, years of experience, and highest education.\n",
            "4. **Vector DB Indexing**: Generating embeddings with `sentence-transformers/all-MiniLM-L6-v2` and persisting in ChromaDB.\n",
            "5. **Hybrid Matching Engine**: Combining BM25 keyword matching + vector similarity, candidate hard-filtering, 0-100 weighted scoring, and match reasoning.\n",
            "6. **Performance Benchmarking**: Measuring Precision@K, Recall@K, and end-to-end query latency."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import time\n",
            "import json\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from pathlib import Path\n",
            "\n",
            "# Ensure project root is in sys.path\n",
            "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
            "if str(PROJECT_ROOT) not in sys.path:\n",
            "    sys.path.insert(0, str(PROJECT_ROOT))\n",
            "\n",
            "# Custom Module Imports\n",
            "from src.document_processor import DocumentProcessor\n",
            "from src.text_processor import TextProcessor\n",
            "from src.metadata_extractor import MetadataExtractor\n",
            "from src.embeddings import EmbeddingGenerator\n",
            "from src.vector_store import ResumeVectorStore\n",
            "from src.resume_rag import ResumeRAGPipeline\n",
            "from src.job_matcher import JobMatcher\n",
            "\n",
            "sns.set_theme(style=\"whitegrid\")\n",
            "print(\"[OK] Core pipeline modules imported successfully!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Document Parsing & Text Extraction\n",
            "Testing text extraction on a sample PDF resume."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "doc_processor = DocumentProcessor()\n",
            "sample_pdf = str(PROJECT_ROOT / \"data/selected_resumes/Python Developer/103.pdf\")\n",
            "raw_text = doc_processor.extract_text(sample_pdf)\n",
            "\n",
            "print(f\"File: {sample_pdf}\")\n",
            "print(f\"Extracted Character Count: {len(raw_text)}\")\n",
            "print(\"\\n--- Extracted Text Sample (First 350 chars) ---\")\n",
            "print(raw_text[:350])"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Section-Aware Chunking (Before vs. After)\n",
            "Demonstrating naive fixed-size character chunking vs. our intelligent section-header boundary chunking."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "text_processor = TextProcessor()\n",
            "cleaned_text = text_processor.clean_text(raw_text)\n",
            "\n",
            "# Naive character/word splitting\n",
            "words = cleaned_text.split()\n",
            "naive_chunks = [\" \".join(words[i:i+100]) for i in range(0, len(words), 80)]\n",
            "\n",
            "# Section-aware chunking\n",
            "section_chunks = text_processor.chunk_text(cleaned_text, max_words=200)\n",
            "\n",
            "print(\"=== NAIVE FIXED-SIZE SLIDING WINDOW CHUNKING (Sample Chunk 1) ===\")\n",
            "print(naive_chunks[0][:220] + \"...\\n\")\n",
            "\n",
            "print(\"=== INTELLIGENT SECTION-AWARE CHUNKING (Sample Chunks) ===\")\n",
            "for idx, chunk in enumerate(section_chunks[:3]):\n",
            "    print(f\"--- Chunk {idx+1} [Section: {chunk['section']}] ---\")\n",
            "    print(chunk['text'][:220] + \"...\\n\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Structured Metadata Extraction\n",
            "Extracting candidate name, technical skills taxonomy, total years of experience, and highest education."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "extractor = MetadataExtractor()\n",
            "metadata = extractor.extract_metadata(cleaned_text, filename=Path(sample_pdf).name)\n",
            "\n",
            "print(\"Extracted Structured Metadata:\")\n",
            "print(json.dumps(metadata, indent=2))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Vector Store & RAG Ingestion Pipeline\n",
            "Ingesting all 50 selected resumes into persistent ChromaDB storage."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "pipeline = ResumeRAGPipeline(db_path=str(PROJECT_ROOT / \"chroma_db\"))\n",
            "t0 = time.time()\n",
            "processed_resumes = pipeline.ingest_directory(resumes_dir=str(PROJECT_ROOT / \"data/selected_resumes\"), reset_db=True)\n",
            "t_ingest = time.time() - t0\n",
            "\n",
            "print(f\"\\nIngested {len(processed_resumes)} resumes into ChromaDB in {t_ingest:.2f} seconds.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Job Matching Engine Execution\n",
            "Executing hybrid matching for a sample Job Description and outputting exact target JSON structure."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "matcher = JobMatcher(vector_store=pipeline.vector_store)\n",
            "\n",
            "# Load sample JD\n",
            "jd_file = str(PROJECT_ROOT / \"data/job_descriptions/jd_1_python_developer.json\")\n",
            "with open(jd_file, \"r\") as f:\n",
            "    sample_jd = json.load(f)\n",
            "\n",
            "print(f\"Job Role: {sample_jd['title']}\")\n",
            "print(f\"Must Have Skills: {sample_jd['must_have_skills']}\")\n",
            "print(f\"Required Experience: {sample_jd['required_experience_years']} years\\n\")\n",
            "\n",
            "match_result = matcher.match(sample_jd, top_k=5)\n",
            "print(\"=== TARGET OUTPUT JSON STRUCTURE ===\")\n",
            "print(json.dumps(match_result, indent=2))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Performance Evaluation & Benchmarking\n",
            "Evaluating Retrieval Accuracy (**Precision@K** and **Recall@K**) and measuring **Latency** across all 5 test Job Descriptions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "jd_dir = PROJECT_ROOT / \"data/job_descriptions\"\n",
            "jd_files = sorted(list(jd_dir.glob(\"*.json\")))\n",
            "\n",
            "precision_k = {5: [], 10: []}\n",
            "recall_k = {5: [], 10: []}\n",
            "latency_records = []\n",
            "\n",
            "for jd_path in jd_files:\n",
            "    with open(jd_path, \"r\") as f:\n",
            "        jd_data = json.load(f)\n",
            "    \n",
            "    target_category = jd_data.get(\"category\", \"\")\n",
            "    \n",
            "    # Measure Latency for K=5 and K=10\n",
            "    t_start = time.time()\n",
            "    res_5 = matcher.match(jd_data, top_k=5)\n",
            "    t_q5 = (time.time() - t_start) * 1000\n",
            "    \n",
            "    t_start = time.time()\n",
            "    res_10 = matcher.match(jd_data, top_k=10)\n",
            "    t_q10 = (time.time() - t_start) * 1000\n",
            "    \n",
            "    latency_records.append({\n",
            "        \"Job Title\": jd_data[\"title\"],\n",
            "        \"Query Latency @ K=5 (ms)\": round(t_q5, 2),\n",
            "        \"Query Latency @ K=10 (ms)\": round(t_q10, 2)\n",
            "    })\n",
            "    \n",
            "    # Calculate Relevant Candidates (Ground truth: 5 candidates exist in target category)\n",
            "    total_relevant_in_db = 5.0\n",
            "    \n",
            "    # K=5\n",
            "    matches_5 = res_5[\"top_matches\"]\n",
            "    rel_5 = sum(1 for m in matches_5 if target_category.lower() in m[\"resume_path\"].lower() or target_category.lower() in m[\"candidate_name\"].lower())\n",
            "    precision_k[5].append(rel_5 / 5.0)\n",
            "    recall_k[5].append(rel_5 / total_relevant_in_db)\n",
            "    \n",
            "    # K=10\n",
            "    matches_10 = res_10[\"top_matches\"]\n",
            "    rel_10 = sum(1 for m in matches_10 if target_category.lower() in m[\"resume_path\"].lower() or target_category.lower() in m[\"candidate_name\"].lower())\n",
            "    precision_k[10].append(rel_10 / 10.0)\n",
            "    recall_k[10].append(rel_10 / total_relevant_in_db)\n",
            "\n",
            "print(\"=== RETRIEVAL ACCURACY RESULTS ===\")\n",
            "print(f\"Mean Precision@5 : {np.mean(precision_k[5]):.2%}\")\n",
            "print(f\"Mean Recall@5    : {np.mean(recall_k[5]):.2%}\")\n",
            "print(f\"Mean Precision@10: {np.mean(precision_k[10]):.2%}\")\n",
            "print(f\"Mean Recall@10   : {np.mean(recall_k[10]):.2%}\")\n",
            "\n",
            "df_latency = pd.DataFrame(latency_records)\n",
            "print(\"\\n=== LATENCY BENCHMARK SUMMARY (ms) ===\")\n",
            "print(df_latency.to_string(index=False))"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Visualize Performance Metrics and Latency Graphs\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Subplot 1: Precision@K & Recall@K\n",
            "metrics_df = pd.DataFrame({\n",
            "    'Metric': ['Precision@5', 'Recall@5', 'Precision@10', 'Recall@10'],\n",
            "    'Score (%)': [np.mean(precision_k[5])*100, np.mean(recall_k[5])*100, np.mean(precision_k[10])*100, np.mean(recall_k[10])*100]\n",
            "})\n",
            "\n",
            "sns.barplot(data=metrics_df, x='Metric', y='Score (%)', palette='crest', ax=axes[0])\n",
            "axes[0].set_ylim(0, 105)\n",
            "axes[0].set_title('Retrieval Accuracy Metrics', fontsize=13, fontweight='bold')\n",
            "for p in axes[0].patches:\n",
            "    axes[0].annotate(f\"{p.get_height():.1f}%\", (p.get_x() + p.get_width() / 2., p.get_height() - 8),\n",
            "                     ha='center', va='center', color='white', fontweight='bold')\n",
            "\n",
            "# Subplot 2: Latency by Job Description\n",
            "sns.barplot(data=df_latency, x='Job Title', y='Query Latency @ K=10 (ms)', palette='viridis', ax=axes[1])\n",
            "axes[1].set_title('Query Latency by Job Role (ms)', fontsize=13, fontweight='bold')\n",
            "axes[1].tick_params(axis='x', rotation=25)\n",
            "for p in axes[1].patches:\n",
            "    axes[1].annotate(f\"{p.get_height():.1f}ms\", (p.get_x() + p.get_width() / 2., p.get_height() - 3),\n",
            "                     ha='center', va='center', color='white', fontweight='bold')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
]

notebook_json = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

notebook_path = Path("notebooks/resume_job_matching.ipynb")
notebook_path.parent.mkdir(parents=True, exist_ok=True)

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=2)

print(f"Generated notebook at {notebook_path}")
