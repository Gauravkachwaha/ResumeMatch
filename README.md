# AI Resume Job Matcher Agent — Stateful LangGraph Recruiter Assistant

An intelligent, stateful AI Recruiter Agent built with **LangGraph**, **OpenRouter (`openrouter/free`)**, **Streamlit**, **ChromaDB**, **SentenceTransformers (`all-MiniLM-L6-v2`)**, and **BM25**.

The agent automates resume parsing, hybrid retrieval, deterministic candidate scoring (0–100), candidate comparison, ranking explainability, interview question generation, multi-round candidate screening (100 -> 10 -> 3), and grounded hire/no-hire recommendations.

---

## 🌟 System Architecture

```text
                                  Streamlit Web Interface (app.py)
                                                │
                                                ▼
                                   Public Invocation API (run_agent)
                                                │
                                                ▼
                                    LangGraph State Machine Engine
                                           (src/matching_agent.py)
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
          Intent Classifier               RAG & Job Matcher              Recruiter Tools
       (src/intent_classifier.py)      (src/resume_rag.py & job_matcher.py)  (src/agent_tools.py)
                 │                              │                              │
                 ▼                              ▼                              ▼
          OpenRouter LLM                ChromaDB Vector Store           Grounded Synthesis &
        (openrouter/free)              & BM25 Keyword Search           Question Generator
```

### System Responsibility Breakdown

* **LangGraph (`src/matching_agent.py`)**: Stateful workflow orchestration, intent-based conditional routing, `AgentState` management, and `MemorySaver` checkpointer persistence across turns.
* **OpenRouter LLM (`openrouter/free`)**: Intent classification, qualitative evidence analysis, natural-language candidate comparisons, ranking explanations, interview question generation, and grounded hire/no-hire decision rationale.
* **Deterministic RAG & JobMatcher (`src/resume_rag.py` & `src/job_matcher.py`)**: Local embeddings (`all-MiniLM-L6-v2`), ChromaDB vector search, BM25 keyword search, structured metadata extraction, and 0–100 candidate scoring (50% Vector + 30% BM25 & Skills + 20% Exp Ratio).
* **Streamlit UI (`app.py`)**: Recruiter presentation layer for chat interaction, candidate cards, comparison tables, interview question cards, multi-round funnel visualization, and state debugging.

---

## 🚀 Key Features

1. **Natural-Language Query Understanding**: Interprets recruiter prompts and extracts job title, mandatory must-have skills, optional nice-to-have skills, and experience criteria.
2. **Hybrid Retrieval & Scoring**: Combines dense vector similarity (`sentence-transformers/all-MiniLM-L6-v2`) with BM25 keyword matching and experience ratios.
3. **Stateful Conversation Memory**: Preserves context, past searches, and refined requirements across multi-turn sessions via LangGraph `MemorySaver` checkpointer using a stable `thread_id`.
4. **Candidate Comparison Tool**: Generates side-by-side comparative matrices over candidate skills, experience, strengths, and trade-offs.
5. **Ranking Explainability**: Explains why Candidate A ranked higher than Candidate B using factual score deltas and resume evidence.
6. **Tailored Interview Questions**: Produces role-specific technical and gap-probing interview questions referencing actual resume evidence.
7. **Iterative Requirement Refinement**: Updates criteria when recruiter provides feedback (*"Make Docker mandatory"*) while preserving previous mandatory skills.
8. **Multi-Round Candidate Screening Funnel**:
   - **Round 1 (Initial Screening)**: Narrows ~100 raw resumes down to Top 10 shortlisted matches.
   - **Round 2 (Deep Analysis)**: Conducts qualitative evidence evaluation across Top 10 matches and selects Top 3 finalists.
   - **Round 3 (Final Evaluation)**: Audits Top 3 finalists against mandatory must-have skills and issues grounded **HIRE** or **NO-HIRE** recommendations.

---

## 📁 Repository Structure

```text
resume-rag-job-matcher-agent/
├── app.py                         # Streamlit Web Application Interface
├── README.md                      # Complete project documentation
├── requirements.txt               # Pinned dependencies
├── .env.example                   # Template for OPENROUTER_API_KEY
├── .gitignore                     # Git ignore rules
│
├── src/                           # Core Source Package
│   ├── agent_state.py             # AgentState TypedDict & CandidateMatch schemas
│   ├── matching_agent.py          # LangGraph StateGraph, router & run_agent API
│   ├── intent_classifier.py       # OpenRouter intent classifier & response schemas
│   ├── agent_nodes.py             # LangGraph search, ranking & tool wrappers
│   ├── agent_tools.py             # Candidate comparison, ranking explanation & interview tools
│   ├── screening_nodes.py         # Multi-round screening pipeline nodes (Rounds 1-3)
│   ├── llm_config.py              # OpenRouter LLM initialization helper
│   ├── document_processor.py      # PDF, DOCX, TXT document parser & OCR
│   ├── text_processor.py          # Section-aware chunker & text cleaner
│   ├── metadata_extractor.py      # Candidate name, skill, experience extractor
│   ├── embeddings.py              # Local SentenceTransformers wrapper
│   ├── vector_store.py            # ChromaDB persistent store & query interface
│   ├── resume_rag.py              # Document ingestion pipeline
│   └── job_matcher.py             # Hybrid job matching engine
│
├── docs/                          # Architecture & Documentation Deliverables
│   ├── architecture.md            # Detailed system architecture specification
│   ├── state_machine.mmd          # LangGraph Mermaid state machine diagram
│   ├── demo_script.md             # 5–6 minute live demonstration walkthrough
│   └── assignment_checklist.md    # Assignment requirement verification matrix
│
├── tests/                         # Test Scenarios & End-to-End Flows
│   └── conversation_flows.md      # 6 verified end-to-end conversation flows
│
├── scratch/                       # Automated Test Suites (Phases 2 - 6)
│   ├── test_phase2.py             # Intent Classifier & State schema tests (11/11 pass)
│   ├── test_phase3.py             # RAG & Ranking Integration tests (5/5 pass)
│   ├── test_phase4.py             # Agent Tools & Explainability tests (5/5 pass)
│   ├── test_phase5.py             # LangGraph Workflow & Memory tests (8/8 pass)
│   ├── test_phase6.py             # Multi-Round Screening tests (8/8 pass)
│   └── test_all_phases.py         # Master test runner (37/37 pass)
│
├── data/                          # Dataset
│   ├── job_descriptions/          # Sample job description specs
│   └── selected_resumes/          # 50 PDF resumes across technical categories
│
└── chroma_db/                     # Persistent ChromaDB vector database
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites

* Python 3.10+
* Git

### 2. Clone Repository & Create Virtual Environment

```bash
git clone https://github.com/Siddhant20104111/resume-rag-job-matcher-agent.git
cd resume-rag-job-matcher-agent

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate virtual environment (Linux / macOS)
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Open `.env` and insert your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

---

## 🏃 Running the Application

### 1. Index Resumes

Before your first search, process the resumes in `data/selected_resumes/` and build the local ChromaDB index:

```bash
python -m src.resume_rag
```

> Running this command again rebuilds the resume collection from the files in that folder.

### 2. Launch Streamlit Web UI

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### 3. Run Automated Test Suite

To run all 37 integration tests across Phases 2 through 6:

```bash
python scratch/test_all_phases.py
```

---

## 💬 Example Conversational Queries

1. **New Search**:
   > `"Find Python developers with 3+ years experience in FastAPI and PostgreSQL."`
2. **Side-by-Side Comparison**:
   > `"Compare the top 3 candidates side by side."`
3. **Ranking Explanation**:
   > `"Why did the top candidate rank higher than the second candidate?"`
4. **Requirement Refinement**:
   > `"Actually, make Docker mandatory."`
5. **Tailored Interview Questions**:
   > `"Generate interview questions for the top candidate."`
6. **Multi-Round Screening**:
   > `"Run a full multi-round screening process on the top Python developer candidates."`

---

## 🔮 Realistic System Limitations

1. **LLM Evidence Dependency**: OpenRouter LLM qualitative analysis relies on candidate resume text extracted during document ingestion. Incomplete resume text may result in unlisted skills being marked as unavailable.
2. **Decision Support**: Final HIRE / NO-HIRE recommendations serve as executive decision support for recruiters and hiring managers, not autonomous hiring decisions.
3. **OpenRouter Rate Limits**: Free tier model availability (`openrouter/free`) may occasionally experience latency or rate limits. The system automatically falls back to deterministic rule-based analysis if the API is temporarily unreachable.
