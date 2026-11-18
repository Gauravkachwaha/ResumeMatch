import json
import re
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from src.agent_state import AgentState, CandidateMatch
from src.llm_config import get_openrouter_llm


def extract_requirements(jd: str) -> Dict[str, Any]:
    """
    Tool: extract_requirements
    Extracts structured job requirements (title, must_have_skills, nice_to_have_skills, required_experience_years)
    from a raw natural language job description string.
    Distinguishes must-have requirements from nice-to-have requirements.
    """
    from src.agent_nodes import get_job_matcher
    matcher = get_job_matcher()
    return matcher.parse_job_description(jd)



def resolve_candidate(candidate_id: str, candidates: List[CandidateMatch]) -> Optional[CandidateMatch]:
    """
    Helper function to resolve a candidate identifier against a list of CandidateMatch objects.
    Supports matching by:
    1. Integer index ("0", "1", "2")
    2. Text ordinal index ("top 1", "candidate 1", "1")
    3. Resume Path or Filename (exact or substring match)
    4. Candidate Name (case-insensitive exact or substring match)
    """
    if not candidates or candidate_id is None:
        return None

    cid_str = str(candidate_id).strip()
    cid_lower = cid_str.lower()

    # 1. Match by exact integer index e.g. "0", "1", "2"
    if cid_str.isdigit():
        idx = int(cid_str)
        if 0 <= idx < len(candidates):
            return candidates[idx]

    # 2. Match by "top X" or "candidate X" ordinal
    idx_match = re.search(r'(?:top|candidate|rank)?\s*#?\s*(\d+)', cid_lower)
    if idx_match:
        try:
            num = int(idx_match.group(1))
            # 1-indexed to 0-indexed conversion if user asks for candidate 1
            if 1 <= num <= len(candidates):
                return candidates[num - 1]
            elif 0 <= num < len(candidates):
                return candidates[num]
        except ValueError:
            pass

    # 3. Match by Resume Path or Filename
    for cand in candidates:
        rpath = cand.get("resume_path", "").lower()
        if cid_lower in rpath or rpath.endswith(cid_lower):
            return cand

    # 4. Match by Candidate Name
    for cand in candidates:
        cname = cand.get("candidate_name", "").lower()
        if cid_lower == cname or cid_lower in cname or cname in cid_lower:
            return cand

    return None



def compare_candidates(candidate_ids: List[str], state: AgentState) -> Dict[str, Any]:
    """
    Tool 1: compare_candidates
    Compares selected candidates using factual data stored in AgentState.
    Grounded in JobMatcher scores, matched/missing skills, experience, and resume excerpts.
    Uses OpenRouter LLM only for natural-language synthesis over factual evidence.
    """
    ranked_candidates = state.get("ranked_candidates", [])
    parsed_jd = state.get("parsed_jd", {})

    if not ranked_candidates:
        return {"error": "No candidates are currently available for comparison. Please run a candidate search first."}

    if not candidate_ids or len(candidate_ids) < 2:
        return {"error": "At least two candidates are required for comparison."}

    # Resolve candidate objects
    resolved_candidates: List[CandidateMatch] = []
    missing_ids: List[str] = []

    for cid in candidate_ids:
        cand = resolve_candidate(cid, ranked_candidates)
        if cand and cand not in resolved_candidates:
            resolved_candidates.append(cand)
        elif not cand:
            missing_ids.append(cid)

    if missing_ids:
        return {
            "error": f"Could not find candidate(s): {', '.join(missing_ids)}. Available candidates: {[c.get('candidate_name') for c in ranked_candidates]}"
        }

    if len(resolved_candidates) < 2:
        return {"error": "At least two distinct valid candidates are required for comparison."}

    # Build factual data summary for LLM grounding
    must_have = parsed_jd.get("must_have_skills", [])
    req_exp = parsed_jd.get("required_experience_years", 0)

    facts_summary = []
    structured_candidates = []

    for cand in resolved_candidates:
        c_name = cand.get("candidate_name", "Candidate")
        score = cand.get("match_score", 0)
        exp = cand.get("experience_years", 0.0)
        matched = cand.get("matched_skills", [])
        missing = cand.get("missing_skills", [])
        excerpts = cand.get("relevant_excerpts", [])
        reasoning = cand.get("reasoning", "")

        facts_summary.append(
            f"Candidate: {c_name}\n"
            f"- Match Score (JobMatcher): {score}/100\n"
            f"- Relevant Experience: ~{exp} years (Required: {req_exp} yrs)\n"
            f"- Matched Skills: {', '.join(matched) if matched else 'None'}\n"
            f"- Missing Skills: {', '.join(missing) if missing else 'None'}\n"
            f"- Match Reasoning: {reasoning}\n"
            f"- Evidence Excerpts: {' | '.join(excerpts[:2]) if excerpts else 'None'}\n"
        )

        structured_candidates.append({
            "name": c_name,
            "match_score": score,
            "experience_years": exp,
            "matched_skills": matched,
            "missing_skills": missing,
            "strengths": [f"Matched {len(matched)} key skills", f"{exp} years experience"],
            "gaps": [f"Missing {s}" for s in missing] if missing else ["No major skill gaps"]
        })

    facts_context = "\n".join(facts_summary)
    jd_context = f"Job Title: {parsed_jd.get('title', 'Role')}\nMust Have: {', '.join(must_have)}"

    # Grounded LLM Call
    system_prompt = (
        "You are an AI Recruiter Analyst. Compare the provided candidates side-by-side.\n"
        "CRITICAL RULE: Base your comparison STRICTLY on the provided candidate facts below. "
        "Do NOT invent, fabricate, or assume any projects, skills, or experience not explicitly listed in the facts.\n"
        "Provide a concise, professional comparison highlighting key differences, strengths, gaps, and the recommended top choice."
    )

    human_prompt = (
        f"Job Context:\n{jd_context}\n\n"
        f"Candidate Factual Evidence:\n{facts_context}\n\n"
        "Please provide:\n"
        "1. Overall Comparison & Best Match\n"
        "2. Key Differences in Skills & Experience\n"
        "3. Strengths & Gaps per candidate"
    )

    try:
        llm = get_openrouter_llm(temperature=0.1)
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        comparison_text = response.content.strip()
    except Exception as e:
        comparison_text = f"Factual comparison generated based on JobMatcher scores.\n\n" + "\n".join(facts_summary)

    # Sort structured candidates by match score descending
    structured_candidates.sort(key=lambda x: x["match_score"], reverse=True)
    best_match = structured_candidates[0]["name"] if structured_candidates else "N/A"

    return {
        "candidates": structured_candidates,
        "comparison": {
            "best_match": best_match,
            "summary": comparison_text
        }
    }


def explain_ranking(higher_candidate_id: str, lower_candidate_id: str, state: AgentState) -> Dict[str, Any]:
    """
    Tool 2: explain_ranking
    Explains why Candidate A ranked higher than Candidate B based on JobMatcher score,
    matched requirements, missing skills, experience, and factual resume evidence.
    """
    ranked_candidates = state.get("ranked_candidates", [])
    parsed_jd = state.get("parsed_jd", {})

    if not ranked_candidates:
        return {"error": "No ranked candidates are available. Please run a search first."}

    higher_cand = resolve_candidate(higher_candidate_id, ranked_candidates)
    lower_cand = resolve_candidate(lower_candidate_id, ranked_candidates)

    if not higher_cand:
        return {"error": f"Candidate '{higher_candidate_id}' not found."}
    if not lower_cand:
        return {"error": f"Candidate '{lower_candidate_id}' not found."}

    # Ensure higher_cand actually has the higher score (swap if user passed them in reverse order)
    if higher_cand.get("match_score", 0) < lower_cand.get("match_score", 0):
        higher_cand, lower_cand = lower_cand, higher_cand

    name_h = higher_cand.get("candidate_name", "Higher Candidate")
    score_h = higher_cand.get("match_score", 0)
    exp_h = higher_cand.get("experience_years", 0.0)
    matched_h = higher_cand.get("matched_skills", [])
    missing_h = higher_cand.get("missing_skills", [])
    reasoning_h = higher_cand.get("reasoning", "")

    name_l = lower_cand.get("candidate_name", "Lower Candidate")
    score_l = lower_cand.get("match_score", 0)
    exp_l = lower_cand.get("experience_years", 0.0)
    matched_l = lower_cand.get("matched_skills", [])
    missing_l = lower_cand.get("missing_skills", [])
    reasoning_l = lower_cand.get("reasoning", "")

    score_diff = score_h - score_l

    # Extract key factual differences
    unique_skills_h = [s for s in matched_h if s not in matched_l]
    unique_skills_l = [s for s in matched_l if s not in matched_h]

    # Grounded LLM natural language synthesis
    system_prompt = (
        "You are an AI Recruiter Analyst. Explain clearly why the higher-scoring candidate ranked above the lower-scoring candidate.\n"
        "CRITICAL RULE: Rely strictly on the scores, skills, experience, and reasoning provided. Do NOT fabricate outside facts."
    )

    facts_text = (
        f"Higher Candidate: {name_h} (Score: {score_h}/100, Exp: {exp_h} yrs)\n"
        f"- Matched Skills: {', '.join(matched_h)}\n"
        f"- Missing Skills: {', '.join(missing_h) if missing_h else 'None'}\n"
        f"- Matcher Reasoning: {reasoning_h}\n\n"
        f"Lower Candidate: {name_l} (Score: {score_l}/100, Exp: {exp_l} yrs)\n"
        f"- Matched Skills: {', '.join(matched_l)}\n"
        f"- Missing Skills: {', '.join(missing_l) if missing_l else 'None'}\n"
        f"- Matcher Reasoning: {reasoning_l}\n"
    )

    human_prompt = f"Job Context: {parsed_jd.get('title', 'Role')}\n\nFactual Evidence:\n{facts_text}\n\nExplain why {name_h} ranked higher than {name_l} in 3 to 4 clear bullet points."

    try:
        llm = get_openrouter_llm(temperature=0.1)
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        explanation_text = response.content.strip()
    except Exception as e:
        explanation_text = (
            f"{name_h} scored {score_h}/100 compared to {name_l}'s {score_l}/100.\n"
            f"Key factors:\n"
            f"1. Score difference of {score_diff} points.\n"
            f"2. {name_h} matched skills not present in {name_l}'s profile: {', '.join(unique_skills_h) if unique_skills_h else 'Higher overall relevance'}.\n"
            f"3. Experience: {name_h} ({exp_h} yrs) vs {name_l} ({exp_l} yrs)."
        )

    return {
        "higher_candidate": name_h,
        "higher_score": score_h,
        "lower_candidate": name_l,
        "lower_score": score_l,
        "score_difference": score_diff,
        "key_differences": {
            "skills_only_in_higher": unique_skills_h,
            "skills_only_in_lower": unique_skills_l,
            "experience_difference_years": round(exp_h - exp_l, 1)
        },
        "explanation": explanation_text
    }


def generate_interview_questions(candidate_id: str, state: AgentState) -> Dict[str, Any]:
    """
    Tool 3: generate_interview_questions
    Generates candidate-specific technical, behavioral, and gap-probing interview questions
    based on job requirements, matched skills, missing skills, and resume evidence.
    Prevents generic questions and avoids inventing unmentioned candidate projects.
    """
    ranked_candidates = state.get("ranked_candidates", [])
    parsed_jd = state.get("parsed_jd", {})

    if not ranked_candidates:
        return {"error": "No candidates available to generate interview questions. Please run a search first."}

    cand = resolve_candidate(candidate_id, ranked_candidates)
    if not cand:
        return {"error": f"Candidate '{candidate_id}' not found in current ranked candidates."}

    c_name = cand.get("candidate_name", "Candidate")
    matched_skills = cand.get("matched_skills", [])
    missing_skills = cand.get("missing_skills", [])
    exp = cand.get("experience_years", 0.0)
    excerpts = cand.get("relevant_excerpts", [])
    title = parsed_jd.get("title", "Job Role")

    facts_summary = (
        f"Candidate Name: {c_name}\n"
        f"Target Role: {title}\n"
        f"Experience: ~{exp} years\n"
        f"Matched Skills: {', '.join(matched_skills) if matched_skills else 'None'}\n"
        f"Missing Skills / Gaps: {', '.join(missing_skills) if missing_skills else 'None'}\n"
        f"Resume Excerpts: {' | '.join(excerpts) if excerpts else 'None'}\n"
    )

    system_prompt = (
        "You are an expert technical interviewer.\n"
        "Generate 4 to 6 candidate-specific interview questions based strictly on the provided candidate facts and resume evidence.\n"
        "RULES:\n"
        "1. DO NOT ask generic questions like 'What is Python?' or 'What is FastAPI?'.\n"
        "2. Ask tailored questions probing their specific experience, architecture decisions, or skill gaps.\n"
        "3. DO NOT fabricate or invent candidate projects not present in the facts.\n"
        "4. Output JSON in the format: {\"candidate_name\": \"...\", \"questions\": [{\"area\": \"...\", \"question\": \"...\", \"purpose\": \"...\"}]}"
    )

    human_prompt = f"Candidate Evidence:\n{facts_summary}\n\nGenerate structured interview questions."

    try:
        llm = get_openrouter_llm(temperature=0.2)
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        content = response.content.strip()

        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()

        parsed_json = json.loads(content)
        questions_list = parsed_json.get("questions", [])
    except Exception as e:
        # Fallback structured question generation if JSON parsing or API fails
        questions_list = []
        for skill in matched_skills[:2]:
            questions_list.append({
                "area": skill,
                "question": f"Based on your resume experience with {skill}, describe how you designed or optimized a production system using {skill}.",
                "purpose": f"Verify depth of hands-on expertise in {skill}."
            })
        for gap in missing_skills[:2]:
            questions_list.append({
                "area": f"Gap: {gap}",
                "question": f"The position requires {gap}, which is not explicitly listed on your resume. What is your experience level or familiarity with {gap}?",
                "purpose": f"Probe knowledge gap in required skill {gap}."
            })
        if not questions_list:
            questions_list.append({
                "area": "Technical Architecture",
                "question": f"Describe a challenging technical project you delivered as a {title}.",
                "purpose": "Assess technical problem-solving and project execution."
            })

    return {
        "candidate_id": candidate_id,
        "candidate_name": c_name,
        "questions": questions_list,
        "summary": f"Generated {len(questions_list)} tailored interview questions for {c_name}."
    }
