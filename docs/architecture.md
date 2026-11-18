# AI Resume Job Matcher Agent — System Architecture & Design Specification

## System Overview

The **AI Resume Job Matcher Agent** is a stateful recruiter assistant built using **LangGraph**, **ChromaDB**, **sentence-transformers (`all-MiniLM-L6-v2`)**, **BM25**, and **OpenRouter (`openrouter/free`)**.

The system automates resume parsing, hybrid retrieval, candidate scoring (0–100), side-by-side candidate comparison, ranking explainability, interview question generation, multi-round screening (100 -> 10 -> 3), and final hire/no-hire recommendations with human feedback loop support.

---

## Component Architecture

```
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

---

## AgentState Specification (`src/agent_state.py`)

```python
class AgentState(TypedDict, total=False):
    messages: Sequence[BaseMessage]             # Conversation history reducer
    raw_jd: str                                 # Latest user text prompt or job description
    parsed_jd: Dict[str, Any]                   # Standardized requirement spec (title, skills, exp)
    filters: Dict[str, Any]                     # Active metadata search filters
    ranked_candidates: List[CandidateMatch]     # Active shortlist of ranked candidates
    top_k: int                                  # Maximum candidates to evaluate
    report: str                                 # Executive report output
    human_feedback: Optional[str]               # Recruiter feedback text
    selected_candidate: Optional[str]           # Candidate focus identifier
    interview_questions: Dict[str, List[str]]   # Tailored questions by candidate
    screening_results: Dict[str, Any]           # Screening funnel output
    final_recommendation: Optional[str]         # Final hire decision summary
    current_intent: IntentType                  # Classified intent of active turn
    next_action: str                            # Routing state key

    # Phase 6 Multi-Round Screening Tracking Fields
    screening_round: int                        # 0: Not started, 1: Round 1, 2: Round 2, 3: Round 3
    screening_status: str                       # "NOT_STARTED", "INITIAL_SCREENING", "DEEP_ANALYSIS", "FINAL_EVALUATION", "COMPLETED", "FAILED"
    initial_candidates: List[CandidateMatch]    # Round 1: Top 10 shortlisted
    deep_analysis_results: List[Dict[str, Any]] # Round 2: Qualitative evidence evaluation
    finalists: List[CandidateMatch]             # Round 2 -> 3: Top 3 finalists
    final_recommendations: List[Dict[str, Any]] # Round 3: HIRE / NO-HIRE decisions
```

---

## Multi-Round Screening Funnel Pipeline

```text
Round 1: Initial Screen (Top 10)
  ├── Input: ~100 Raw Resumes in ChromaDB
  ├── Engine: Hybrid Search (50% Vector + 30% BM25 & Skills + 20% Exp Ratio)
  └── Output: state.initial_candidates (Top 10 Shortlist)

Round 2: Deep Qualitative Analysis (Top 3 Finalists)
  ├── Input: Top 10 Shortlist + Resume Evidence Excerpts
  ├── Engine: OpenRouter LLM Qualitative Assessment
  └── Output: state.finalists (Top 3 Finalists)

Round 3: Final Hire/No-Hire Evaluation
  ├── Input: Top 3 Finalists + Must-Have Requirements Audit
  ├── Hard Rule: Mandatory skill missing -> Explicit NO-HIRE decision
  └── Output: state.final_recommendations (HIRE / NO-HIRE + Confidence & Rationale)
```

---

## Memory & State Persistence

State persistence across conversational turns is powered by LangGraph's `MemorySaver` checkpointer using a stable `thread_id` UUID. This allows:
1. Re-using previous candidate search results when performing follow-up queries (e.g. *"Compare the top 3"*).
2. Preserving previous must-have skills when refining criteria (e.g. *"Make Docker mandatory"*).
3. Maintaining multi-round screening status across turns.
