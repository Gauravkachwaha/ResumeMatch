# End-to-End Conversation Flow Verification Suite

This document records 6 verified end-to-end conversation scenarios executed against the compiled LangGraph agent (`run_agent`).

---

## Flow 1: Natural Language Search & Shortlisting

* **Turn 1 (User)**: `"Find Python developers with 3+ years experience in FastAPI and PostgreSQL."`
* **Agent Routing**: `START -> intent_classifier (new_search) -> parse_and_refine -> search_resumes -> rank_candidates -> generate_report -> human_feedback -> END`
* **State Updates**:
  * `parsed_jd`: `{"title": "Python Developer", "must_have_skills": ["Python", "FastAPI", "PostgreSQL"], "required_experience_years": 3.0}`
  * `ranked_candidates`: 10 matches retrieved and scored out of 100.
* **Verified Output**: Executive Match Report displaying Top 10 candidate match scores, matched skills, and evidence excerpts.

---

## Flow 2: Stateful Side-by-Side Candidate Comparison

* **Turn 1 (User)**: `"Find Python developers with 3+ years experience."`
* **Turn 2 (User)**: `"Compare the top 3 candidates side by side."` (Same `thread_id`)
* **Agent Routing**: `START -> intent_classifier (compare_candidates) -> compare_candidates_node -> human_feedback -> END`
* **State Verification**: Reuses existing `ranked_candidates` in `AgentState` without executing a new RAG search.
* **Verified Output**: Structured side-by-side comparison matrix covering Match Scores, Experience Years, Matched Skills, and Key Trade-offs.

---

## Flow 3: Ranking Explainability & Evidence Traceability

* **Turn 1 (User)**: Candidate search executed.
* **Turn 2 (User)**: `"Why did Candidate 1 rank higher than Candidate 2?"` (Same `thread_id`)
* **Agent Routing**: `START -> intent_classifier (explain_ranking) -> explain_ranking_node -> human_feedback -> END`
* **State Verification**: Identifies score differential (`74 vs 40`) and compares skill match vectors.
* **Verified Output**: 3 to 4 grounded bullet points explaining score delta based on JobMatcher evidence and resume excerpts.

---

## Flow 4: Iterative Requirement Refinement & Re-Ranking

* **Turn 1 (User)**: `"Find Python developers with 3+ years experience."`
* **Turn 2 (User)**: `"Actually, make Docker mandatory."` (Same `thread_id`)
* **Agent Routing**: `START -> intent_classifier (refine_requirements) -> parse_and_refine -> search_resumes -> rank_candidates -> generate_report -> human_feedback -> END`
* **State Verification**: Preserves existing `must_have_skills` (`Python`) and appends `Docker` as a mandatory skill. Re-runs search and re-ranks candidate pool.
* **Verified Output**: Updated candidate match report displaying modified ranking reflecting Docker requirement.

---

## Flow 5: Tailored Interview Question Generation

* **Turn 1 (User)**: Candidate search executed.
* **Turn 2 (User)**: `"Generate interview questions for the top candidate."` (Same `thread_id`)
* **Agent Routing**: `START -> intent_classifier (generate_interview_questions) -> generate_interview_questions_node -> human_feedback -> END`
* **State Verification**: Reads top candidate's matched skills, missing skills, and resume excerpts.
* **Verified Output**: 4 to 6 technical and gap-probing interview questions referencing actual candidate experience.

---

## Flow 6: Multi-Round Screening Funnel (Round 1 -> 2 -> 3 -> Hire/No-Hire)

* **Turn 1 (User)**: `"Run a full multi-round screening process on the top Python developer candidates."`
* **Agent Routing**: `START -> intent_classifier (screen_candidates) -> initial_screening_node (Round 1: Top 10) -> deep_analysis_node (Round 2: Qualitative Analysis) -> final_evaluation_node (Round 3: Must-Have Audit) -> screening_report_node -> human_feedback -> END`
* **State Verification**:
  * `screening_round`: `3`
  * `screening_status`: `"COMPLETED"`
  * `initial_candidates`: Top 10 shortlisted.
  * `finalists`: Top 3 selected.
  * `final_recommendations`: Structured `HIRE` or `NO-HIRE` decision with rationale per finalist.
* **Verified Output**: Comprehensive multi-stage screening funnel report.
