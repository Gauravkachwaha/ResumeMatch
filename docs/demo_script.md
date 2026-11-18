# AI Resume Job Matcher Agent — 5 to 6 Minute Live Demo Script

## Demo Overview
* **Goal**: Demonstrate the conversational AI Recruiter Agent using LangGraph, hybrid RAG, candidate comparison, ranking explainability, interview question generation, multi-round screening, and human feedback loops.
* **Target Audience**: Technical Assessors & Recruiters.
* **Duration**: 5 minutes 30 seconds.

---

## Step-by-Step Script Walkthrough

### 0:00 – 0:30 | Introduction & Problem Overview
* **Action**: Launch application using `streamlit run app.py`. Point to the header and sidebar configuration.
* **Script**:
  > "Hello! Welcome to the demo of our AI Resume Job Matcher & Recruiter Agent. Hiring teams struggle with screening hundreds of resumes, assessing qualitative fit, and explaining why one candidate ranked above another. Our agent solves this using LangGraph state machines, ChromaDB vector search, BM25 keyword matching, and OpenRouter LLMs."

---

### 0:30 – 1:15 | System Architecture Overview
* **Action**: Expand the **LangGraph State Machine Visualizer** in the sidebar.
* **Script**:
  > "Under the hood, the system is orchestrating an `AgentState` object using LangGraph. When a recruiter sends a query, our `intent_classifier` node categorizes the user's intent and routes execution dynamically to specialized graph nodes—such as resume search, candidate ranking, side-by-side comparison, or multi-round screening—backed by a `MemorySaver` checkpointer for stateful multi-turn memory."

---

### 1:15 – 2:15 | Basic Search & Match Reporting
* **Action**: Click quick action **"🔍 Find Python Devs"** or type:
  `"Find Python developers with 3+ years experience in FastAPI and PostgreSQL."`
* **Script**:
  > "Let's perform a natural-language candidate search. The agent extracts structured requirements—identifying Python, FastAPI, and PostgreSQL as must-have skills—queries ChromaDB and BM25, and invokes our deterministic `JobMatcher` engine to rank candidates out of 100. As you can see, Candidate Jessica Claire ranks #1 with a match score of 74/100."

---

### 2:15 – 3:00 | Conversational State Reuse (Compare & Explain)
* **Action**: Type: `"Compare the top 3 candidates side by side."` then `"Why did the top candidate rank higher than the second candidate?"`
* **Script**:
  > "Notice that when we ask the agent to compare candidates or explain ranking differences, it doesn't re-run a costly RAG search. It reuses the active candidate pool in `AgentState` via `thread_id`. The ranking explanation tool breaks down exact score deltas, skill coverage differences, and resume evidence."

---

### 3:00 – 3:45 | Human Feedback & Requirement Refinement
* **Action**: Type: `"Actually, make Docker mandatory."`
* **Script**:
  > "Now let's demonstrate the Human Feedback Loop. When we ask to make Docker mandatory, our requirement refinement node preserves previous skills like Python and FastAPI, adds Docker as a mandatory skill, re-runs retrieval and re-ranks candidates in real-time."

---

### 3:45 – 5:00 | Multi-Round Candidate Screening Funnel
* **Action**: Click quick action **"🏆 Full Screening"** or type:
  `"Run a full screening process on the top Python candidates."`
* **Script**:
  > "Finally, let's look at our Multi-Round Screening Funnel.
  > In **Round 1**, the agent retrieves the Top 10 shortlisted candidates out of ~100 raw resumes.
  > In **Round 2**, OpenRouter conducts deep qualitative analysis over resume evidence excerpts.
  > In **Round 3**, the agent evaluates finalists against mandatory must-have requirements. If a candidate lacks a mandatory skill, it enforces a strict NO-HIRE decision with explicit reasoning."

---

### 5:00 – 5:30 | Conclusion & Summary
* **Action**: Open the **Agent State Debugger** tab to show state inspection.
* **Script**:
  > "In summary, our system combines deterministic RAG search with grounded LLM qualitative reasoning and stateful graph orchestration. Thank you!"
