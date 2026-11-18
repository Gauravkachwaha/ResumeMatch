import os
import json
from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from src.agent_state import AgentState, CandidateMatch, create_candidate_match
from src.vector_store import ResumeVectorStore
from src.job_matcher import JobMatcher
from src.intent_classifier import IntentClassifier
from src.llm_config import get_openrouter_llm
from src.agent_tools import compare_candidates, explain_ranking, generate_interview_questions, resolve_candidate


# Shared instances for vector store, job matcher & intent classifier
_vector_store_instance: Optional[ResumeVectorStore] = None
_job_matcher_instance: Optional[JobMatcher] = None
_intent_classifier_instance: Optional[IntentClassifier] = None


def get_vector_store() -> ResumeVectorStore:
    """Singleton helper to obtain shared ResumeVectorStore instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = ResumeVectorStore()
    return _vector_store_instance


def get_job_matcher() -> JobMatcher:
    """Singleton helper to obtain shared JobMatcher instance."""
    global _job_matcher_instance
    if _job_matcher_instance is None:
        vs = get_vector_store()
        _job_matcher_instance = JobMatcher(vector_store=vs)
    return _job_matcher_instance


def get_intent_classifier() -> IntentClassifier:
    """Singleton helper to obtain shared IntentClassifier instance."""
    global _intent_classifier_instance
    if _intent_classifier_instance is None:
        _intent_classifier_instance = IntentClassifier()
    return _intent_classifier_instance


def intent_classifier_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Intent Classifier.
    Extracts the latest user message and uses the OpenRouter IntentClassifier to classify intent.
    Updates state["current_intent"] and state["raw_jd"].
    """
    messages = state.get("messages", [])
    raw_jd = state.get("raw_jd", "").strip()

    # Extract user input from latest HumanMessage if available
    user_input = ""
    if messages:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                user_input = msg.content
                break

    if not user_input:
        user_input = raw_jd

    if not user_input:
        return {
            "current_intent": "general_question",
            "report": "Please provide a job description or prompt."
        }

    classifier = get_intent_classifier()
    res = classifier.classify(user_input, history=messages)
    intent_val = res.get("intent", "general_question")

    return {
        "current_intent": intent_val,
        "raw_jd": user_input
    }


def parse_and_refine_requirements_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Requirement Parsing & Refinement.
    - For new_search: Converts raw text into structured parsed_jd.
    - For refine_requirements: Preserves previous parsed_jd and updates criteria using OpenRouter LLM.
    """
    intent = state.get("current_intent", "new_search")
    raw_jd = state.get("raw_jd", "")
    existing_parsed = state.get("parsed_jd", {})

    matcher = get_job_matcher()

    if intent == "refine_requirements" and existing_parsed:
        # LLM-assisted refinement preserving previous requirements
        sys_prompt = (
            "You are an AI Recruiter Requirement Refiner.\n"
            "Your task is to update existing job requirements based on the user's refinement prompt.\n"
            "CRITICAL RULE: PRESERVE all existing skills, title, and experience criteria UNLESS the user explicitly asks to remove or change them.\n"
            "Output JSON format: {\"title\": \"...\", \"must_have_skills\": [...], \"nice_to_have_skills\": [...], \"required_experience_years\": float, \"description\": \"...\"}"
        )
        curr_str = json.dumps(existing_parsed)
        user_prompt = f"Current Requirements: {curr_str}\nRefinement Request: {raw_jd}\n\nUpdate the requirements JSON."

        try:
            llm = get_openrouter_llm(temperature=0.0)
            resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
            content = resp.content.strip()
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()
            updated_parsed = json.loads(content)
        except Exception:
            # Deterministic fallback requirement updating
            updated_parsed = dict(existing_parsed)
            # Check for skill additions in raw_jd
            new_skills = matcher.metadata_extractor.extract_skills(raw_jd)
            must_have = list(updated_parsed.get("must_have_skills", []))
            for s in new_skills:
                if s not in must_have:
                    must_have.append(s)
            updated_parsed["must_have_skills"] = must_have
            updated_parsed["description"] = f"{existing_parsed.get('description', '')} (Refinement: {raw_jd})"

        return {"parsed_jd": updated_parsed}
    else:
        # Standard requirement parsing
        parsed = matcher.parse_job_description(raw_jd)
        return {"parsed_jd": parsed}


def search_resumes_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Resume Search.
    Executes vector query over ChromaDB using standardized parsed_jd requirements.
    """
    parsed_jd = state.get("parsed_jd", {})
    raw_jd = state.get("raw_jd", "").strip()
    filters = state.get("filters", {})
    top_k = state.get("top_k", 10) or 10

    if not parsed_jd and not raw_jd:
        return {
            "report": "Error: No job requirements provided for resume search.",
            "ranked_candidates": []
        }

    matcher = get_job_matcher()
    if not parsed_jd:
        parsed_jd = matcher.parse_job_description(raw_jd)

    vector_store = get_vector_store()
    jd_text = parsed_jd.get("description", raw_jd)

    try:
        vector_store.search_vector(jd_text, top_k=top_k, where_filter=filters if filters else None)
    except Exception as e:
        return {
            "report": f"RAG Search Error: {e}",
            "ranked_candidates": []
        }

    return {"parsed_jd": parsed_jd}


def rank_candidates_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Candidate Ranking.
    Runs hybrid matching engine (ChromaDB vector + BM25 keyword + experience ratio + penalties).
    Updates state["ranked_candidates"].
    """
    parsed_jd = state.get("parsed_jd", {})
    raw_jd = state.get("raw_jd", "").strip()
    top_k = state.get("top_k", 10) or 10

    if not parsed_jd and not raw_jd:
        return {"ranked_candidates": [], "report": "Error: Missing requirements for candidate ranking."}

    matcher = get_job_matcher()
    jd_input = parsed_jd if parsed_jd else raw_jd

    try:
        match_result = matcher.match(jd_input=jd_input, top_k=top_k)
    except Exception as e:
        return {"ranked_candidates": [], "report": f"JobMatcher Execution Failure: {e}"}

    raw_matches = match_result.get("top_matches", [])
    if not raw_matches:
        return {"ranked_candidates": [], "report": "No matching candidates were found."}

    must_have_lower = [s.lower() for s in parsed_jd.get("must_have_skills", [])]
    adapted_candidates: List[CandidateMatch] = []

    for raw_cand in raw_matches:
        c_skills = raw_cand.get("matched_skills", [])
        c_skills_lower = [s.lower() for s in c_skills]
        missing_skills = [s for s in must_have_lower if s not in c_skills_lower]

        match_obj = create_candidate_match(
            candidate_name=raw_cand.get("candidate_name", "Candidate"),
            resume_path=raw_cand.get("resume_path", ""),
            match_score=raw_cand.get("match_score", 0),
            matched_skills=c_skills,
            missing_skills=missing_skills,
            relevant_excerpts=raw_cand.get("relevant_excerpts", []),
            reasoning=raw_cand.get("reasoning", "")
        )
        adapted_candidates.append(match_obj)

    adapted_candidates = adapted_candidates[:top_k]
    return {"ranked_candidates": adapted_candidates}


def generate_report_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Report Generation.
    Synthesizes executive recruiter candidate report based on state["ranked_candidates"] and state["parsed_jd"].
    """
    ranked = state.get("ranked_candidates", [])
    parsed_jd = state.get("parsed_jd", {})

    title = parsed_jd.get("title", "Job Role")
    must_have = parsed_jd.get("must_have_skills", [])

    if not ranked:
        report_text = f"Executive Recruiter Report for {title}\n\nNo candidate resumes matched the required criteria."
        return {
            "report": report_text,
            "messages": [AIMessage(content=report_text)]
        }

    lines = [
        f"### 📋 Executive Candidate Match Report: {title}",
        f"**Evaluated Candidates**: Top {len(ranked)} matches retrieved.",
        f"**Must-Have Skills**: {', '.join(must_have) if must_have else 'Unspecified'}",
        "---"
    ]

    for i, cand in enumerate(ranked, 1):
        name = cand.get("candidate_name", "Candidate")
        score = cand.get("match_score", 0)
        exp = cand.get("experience_years", 0.0)
        matched = cand.get("matched_skills", [])
        missing = cand.get("missing_skills", [])
        reasoning = cand.get("reasoning", "")

        lines.append(f"#### {i}. {name} — Match Score: {score}/100")
        lines.append(f"- **Matched Skills**: {', '.join(matched) if matched else 'None'}")
        if missing:
            lines.append(f"- **Missing Skills**: {', '.join(missing)}")
        lines.append(f"- **Experience**: ~{exp} years")
        lines.append(f"- **Match Rationale**: {reasoning}")
        lines.append("")

    report_text = "\n".join(lines).strip()
    return {
        "report": report_text,
        "messages": [AIMessage(content=report_text)]
    }


def compare_candidates_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Candidate Comparison Tool Wrapper.
    Reuses existing ranked_candidates from state without executing a new RAG search.
    """
    ranked = state.get("ranked_candidates", [])
    if not ranked:
        msg = "No candidates are available to compare. Please run a candidate search first."
        return {"report": msg, "messages": [AIMessage(content=msg)]}

    # Extract target candidates from user prompt or default to top 2/3
    user_prompt = state.get("raw_jd", "").lower()
    target_ids = ["0", "1"]  # default top 2

    if "top 3" in user_prompt or "3" in user_prompt and len(ranked) >= 3:
        target_ids = ["0", "1", "2"]

    res = compare_candidates(target_ids, state)
    if "error" in res:
        msg = res["error"]
        return {"report": msg, "messages": [AIMessage(content=msg)]}

    cmp_info = res.get("comparison", {})
    summary = cmp_info.get("summary", "")
    best = cmp_info.get("best_match", "N/A")

    report_text = f"### 📊 Candidate Comparison Analysis\n**Recommended Choice**: {best}\n\n{summary}"
    return {
        "report": report_text,
        "messages": [AIMessage(content=report_text)]
    }


def explain_ranking_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Ranking Explanation Tool Wrapper.
    Explains why Candidate 1 ranked higher than Candidate 2 based on JobMatcher scores & evidence.
    """
    ranked = state.get("ranked_candidates", [])
    if len(ranked) < 2:
        msg = "At least two candidates are required in state to explain ranking differences."
        return {"report": msg, "messages": [AIMessage(content=msg)]}

    res = explain_ranking("0", "1", state)
    if "error" in res:
        msg = res["error"]
        return {"report": msg, "messages": [AIMessage(content=msg)]}

    exp_text = res.get("explanation", "")
    h_name = res.get("higher_candidate", "")
    l_name = res.get("lower_candidate", "")

    report_text = f"### 💡 Ranking Explanation: {h_name} vs {l_name}\n\n{exp_text}"
    return {
        "report": report_text,
        "messages": [AIMessage(content=report_text)]
    }


def generate_interview_questions_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Interview Question Generator Wrapper.
    Generates tailored technical & behavioral questions for target candidate.
    """
    ranked = state.get("ranked_candidates", [])
    if not ranked:
        msg = "No candidates available in state to generate interview questions."
        return {"report": msg, "messages": [AIMessage(content=msg)]}

    cand_id = state.get("selected_candidate", "0")
    res = generate_interview_questions(cand_id, state)

    if "error" in res:
        msg = res["error"]
        return {"report": msg, "messages": [AIMessage(content=msg)]}

    c_name = res.get("candidate_name", "Candidate")
    questions = res.get("questions", [])

    lines = [f"### 🎯 Tailored Interview Questions: {c_name}"]
    for i, q in enumerate(questions, 1):
        lines.append(f"**Q{i} [{q.get('area', 'General')}]**: {q.get('question', '')}")
        lines.append(f"*Purpose*: {q.get('purpose', '')}\n")

    report_text = "\n".join(lines)
    return {
        "interview_questions": {c_name: [q.get("question", "") for q in questions]},
        "report": report_text,
        "messages": [AIMessage(content=report_text)]
    }


def general_response_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph Node: General Conversational Assistant.
    Answers general questions about current search/candidate context without fabricating facts.
    """
    raw_jd = state.get("raw_jd", "")
    ranked = state.get("ranked_candidates", [])
    parsed = state.get("parsed_jd", {})

    cand_summary = ", ".join([c.get("candidate_name", "") for c in ranked[:3]]) if ranked else "None"
    context_text = f"Role: {parsed.get('title', 'N/A')}. Active Top Candidates: {cand_summary}."

    sys_prompt = "You are a helpful AI Recruiter Assistant. Answer the user's question using the active candidate context. Do not invent unmentioned facts."
    human_prompt = f"Context: {context_text}\nUser Question: {raw_jd}"

    try:
        llm = get_openrouter_llm(temperature=0.2)
        resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
        ans = resp.content.strip()
    except Exception as e:
        ans = f"Assistant Response for '{raw_jd}': Currently evaluating candidate context for {parsed.get('title', 'Role')}."

    return {
        "report": ans,
        "messages": [AIMessage(content=ans)]
    }


def human_feedback_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Human Feedback Loop Handler.
    Updates state with recruiter feedback and prepares next turn state.
    """
    raw_jd = state.get("raw_jd", "")
    return {
        "human_feedback": raw_jd,
        "next_action": "await_user_input"
    }


def screening_placeholder_node(state: AgentState) -> Dict[str, Any]:
    """Phase 6 Placeholder Node: Multi-round Candidate Screening Funnel."""
    msg = "Multi-round candidate screening workflow placeholder (Phase 6 feature)."
    return {"report": msg, "messages": [AIMessage(content=msg)]}


def recommendation_placeholder_node(state: AgentState) -> Dict[str, Any]:
    """Phase 6 Placeholder Node: Final Hire / No-Hire Recommendation System."""
    msg = "Final hire / no-hire recommendation system placeholder (Phase 6 feature)."
    return {"report": msg, "messages": [AIMessage(content=msg)]}
