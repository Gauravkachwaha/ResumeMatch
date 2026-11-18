from typing import TypedDict, List, Dict, Any, Optional, Literal, Sequence, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# Supported Intent Types for Phase 2 and agent routing
IntentType = Literal[
    "new_search",
    "refine_requirements",
    "compare_candidates",
    "explain_ranking",
    "generate_interview_questions",
    "screen_candidates",
    "final_recommendation",
    "general_question"
]


class CandidateMatch(TypedDict, total=False):
    """
    State structure representing a candidate match result.
    Fully compatible with the existing JobMatcher output dictionary,
    adding optional fields for downstream LLM reasoning and screening.
    """
    candidate_name: str
    resume_path: str
    match_score: int              # Deterministic 0-100 score from JobMatcher
    matched_skills: List[str]
    missing_skills: List[str]
    experience_years: float
    relevant_excerpts: List[str]
    reasoning: str                 # Hybrid matcher explanation string
    recommendation: Optional[str]  # LLM recommendation e.g. "Strong Hire", "Consider"


class AgentState(TypedDict, total=False):
    """
    LangGraph state schema for the AI Recruiter Agent.
    Tracks conversation history, requirements, active candidates, intent, tools, and multi-round screening.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Conversation history using LangGraph reducer
    raw_jd: str                                               # Latest raw text request or JD
    parsed_jd: Dict[str, Any]                                 # Structured requirements (title, skills, exp)
    filters: Dict[str, Any]                                   # Active candidate metadata filters
    ranked_candidates: List[CandidateMatch]                   # Pool of ranked candidate matches
    top_k: int                                                # Max candidates to retrieve/evaluate
    report: str                                               # Latest assistant output / executive summary
    human_feedback: Optional[str]                             # Human feedback / recruiter input
    selected_candidate: Optional[str]                         # Selected candidate focus (e.g. for questions)
    interview_questions: Dict[str, List[str]]                 # Generated interview questions by candidate
    screening_results: Dict[str, Any]                         # Multi-round screening funnel output
    final_recommendation: Optional[str]                       # Final hire/no-hire decision rationale
    current_intent: IntentType                                # Classified user intent of current turn
    next_action: str                                          # Graph routing key for conditional edges

    # Phase 6 Multi-Round Screening Funnel Tracking Fields
    screening_round: int                                      # 0: Not started, 1: Round 1, 2: Round 2, 3: Round 3
    screening_status: str                                     # "NOT_STARTED", "INITIAL_SCREENING", "DEEP_ANALYSIS", "FINAL_EVALUATION", "COMPLETED", "FAILED"
    initial_candidates: List[CandidateMatch]                  # Round 1: Top 10 candidates from RAG + JobMatcher
    deep_analysis_results: List[Dict[str, Any]]               # Round 2: LLM qualitative analysis for Top 10
    finalists: List[CandidateMatch]                           # Round 2 -> Round 3: Top 3 selected finalists
    final_recommendations: List[Dict[str, Any]]               # Round 3: HIRE / NO-HIRE recommendations per finalist



def create_candidate_match(
    candidate_name: str,
    resume_path: str,
    match_score: int,
    matched_skills: List[str],
    relevant_excerpts: List[str],
    reasoning: str,
    missing_skills: Optional[List[str]] = None,
    experience_years: float = 0.0,
    recommendation: Optional[str] = None
) -> CandidateMatch:
    """
    Helper function to construct a valid CandidateMatch dictionary.
    Adapts existing JobMatcher dictionary results into candidate match state.
    """
    return {
        "candidate_name": candidate_name,
        "resume_path": resume_path,
        "match_score": match_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills or [],
        "experience_years": experience_years,
        "relevant_excerpts": relevant_excerpts,
        "reasoning": reasoning,
        "recommendation": recommendation
    }
