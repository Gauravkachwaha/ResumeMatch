import json
import os
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

from src.agent_state import AgentState, CandidateMatch
from src.agent_nodes import search_resumes_node, rank_candidates_node
from src.llm_config import get_openrouter_llm
from src.agent_tools import generate_interview_questions


# Configurable Screening Funnel Limits
INITIAL_SCREENING_LIMIT = 10
FINALIST_LIMIT = 3


def initial_screening_node(state: AgentState) -> Dict[str, Any]:
    """
    Round 1: Initial Screening Node.
    Executes hybrid RAG + JobMatcher pipeline to retrieve and rank the Top 10 initial candidates
    out of the raw resume database.
    """
    raw_jd = state.get("raw_jd", "")
    parsed_jd = state.get("parsed_jd", {})

    if not parsed_jd and not raw_jd:
        return {
            "screening_status": "FAILED",
            "report": "Screening Error: No job requirements provided for screening."
        }

    # 1. Execute Search & Ranking with top_k = INITIAL_SCREENING_LIMIT
    search_state = dict(state)
    search_state["top_k"] = INITIAL_SCREENING_LIMIT
    
    search_res = search_resumes_node(search_state)
    search_state.update(search_res)

    rank_res = rank_candidates_node(search_state)
    ranked = rank_res.get("ranked_candidates", [])

    # Purely deterministic Python slicing for Top 10
    top_10 = ranked[:INITIAL_SCREENING_LIMIT]

    return {
        "screening_round": 1,
        "screening_status": "INITIAL_SCREENING",
        "parsed_jd": search_state.get("parsed_jd", parsed_jd),
        "initial_candidates": top_10,
        "ranked_candidates": top_10
    }


def deep_analysis_node(state: AgentState) -> Dict[str, Any]:
    """
    Round 2: Deep Analysis Node.
    Uses OpenRouter LLM (openrouter/free) to conduct evidence-based qualitative analysis of the Top 10 candidates.
    Evaluates demonstrated skills, missing skills, experience relevance, resume evidence, gaps, risks, and interview focus.
    Deterministically selects Top 3 finalists for Round 3.
    """
    initial_candidates = state.get("initial_candidates", state.get("ranked_candidates", []))
    parsed_jd = state.get("parsed_jd", {})

    if not initial_candidates:
        return {
            "screening_status": "FAILED",
            "report": "Screening Error: No initial candidates available for deep analysis."
        }

    top_10 = initial_candidates[:INITIAL_SCREENING_LIMIT]
    must_have = parsed_jd.get("must_have_skills", [])
    title = parsed_jd.get("title", "Job Role")
    req_exp = parsed_jd.get("required_experience_years", 0)

    deep_results: List[Dict[str, Any]] = []
    llm = get_openrouter_llm(temperature=0.1)

    sys_prompt = (
        "You are an AI Senior Recruiter conducting Round 2 Deep Analysis.\n"
        "Analyze the provided candidate evidence against the job requirements.\n"
        "STRICT RULE: Rely ONLY on the candidate evidence provided. Do NOT fabricate outside projects or unmentioned experience.\n"
        "Output JSON matching: {\"candidate_name\": \"...\", \"demonstrated_skills\": [...], \"missing_skills\": [...], \"experience_relevance\": \"...\", \"strengths\": [...], \"gaps\": [...], \"risks\": [...], \"qualitative_assessment\": \"...\"}"
    )

    for cand in top_10:
        c_name = cand.get("candidate_name", "Candidate")
        score = cand.get("match_score", 0)
        exp = cand.get("experience_years", 0.0)
        matched = cand.get("matched_skills", [])
        missing = cand.get("missing_skills", [])
        excerpts = cand.get("relevant_excerpts", [])
        reasoning = cand.get("reasoning", "")

        facts = (
            f"Candidate: {c_name}\n"
            f"Job Title: {title}\n"
            f"JobMatcher Score: {score}/100\n"
            f"Experience: ~{exp} yrs (Required: {req_exp} yrs)\n"
            f"Matched Skills: {', '.join(matched) if matched else 'None'}\n"
            f"Missing Skills: {', '.join(missing) if missing else 'None'}\n"
            f"Resume Excerpts: {' | '.join(excerpts) if excerpts else 'None'}\n"
            f"Matcher Rationale: {reasoning}"
        )

        try:
            resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=facts)])
            content = resp.content.strip()
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()
            parsed_analysis = json.loads(content)
            parsed_analysis["initial_score"] = score
            parsed_analysis["candidate_name"] = c_name
        except Exception as e:
            # Deterministic grounded fallback analysis
            parsed_analysis = {
                "candidate_name": c_name,
                "initial_score": score,
                "demonstrated_skills": matched,
                "missing_skills": missing,
                "experience_relevance": f"{exp} years of relevant experience vs {req_exp} required.",
                "strengths": [f"Demonstrated {len(matched)} key skills: {', '.join(matched[:3])}"],
                "gaps": [f"Missing required skill: {s}" for s in missing] if missing else ["No critical skill gaps identified"],
                "risks": [f"Missing skill {s}" for s in missing] if missing else ["None"],
                "qualitative_assessment": reasoning
            }

        deep_results.append(parsed_analysis)

    # Deterministic Python selection of Top 3 finalists (ordered by JobMatcher score & requirement coverage)
    finalists = top_10[:FINALIST_LIMIT]

    return {
        "screening_round": 2,
        "screening_status": "DEEP_ANALYSIS",
        "deep_analysis_results": deep_results,
        "finalists": finalists
    }


def final_evaluation_node(state: AgentState) -> Dict[str, Any]:
    """
    Round 3: Final Evaluation Node.
    Evaluates Top 3 finalists and generates explicit HIRE / NO-HIRE recommendations.
    CRITICAL RULE: Candidates missing mandatory must-have requirements receive NO-HIRE with explicit warnings.
    """
    finalists = state.get("finalists", [])
    parsed_jd = state.get("parsed_jd", {})
    deep_results = state.get("deep_analysis_results", [])

    if not finalists:
        return {
            "screening_status": "FAILED",
            "report": "Screening Error: No finalists available for final evaluation."
        }

    must_have = parsed_jd.get("must_have_skills", [])
    must_have_lower = [s.lower() for s in must_have]
    title = parsed_jd.get("title", "Job Role")

    final_recs: List[Dict[str, Any]] = []
    llm = get_openrouter_llm(temperature=0.1)

    sys_prompt = (
        "You are an Executive Hiring Authority conducting Round 3 Final Evaluation.\n"
        "Evaluate the candidate for a final HIRE or NO-HIRE recommendation.\n"
        "CRITICAL RULES:\n"
        "1. Must-have requirements are MANDATORY. A candidate missing core must-have skills MUST be marked NO-HIRE.\n"
        "2. Base your decision STRICTLY on the provided evidence. Do NOT fabricate missing facts.\n"
        "3. Output JSON matching: {\"candidate_name\": \"...\", \"recommendation\": \"HIRE\" | \"NO-HIRE\", \"confidence\": 0-100, \"reasoning\": \"...\", \"strengths\": [...], \"gaps\": [...], \"risks\": [...], \"interview_focus_areas\": [...]}"
    )

    for cand in finalists:
        c_name = cand.get("candidate_name", "Candidate")
        score = cand.get("match_score", 0)
        exp = cand.get("experience_years", 0.0)
        matched = cand.get("matched_skills", [])
        matched_lower = [s.lower() for s in matched]
        missing = [s for s in must_have_lower if s not in matched_lower]

        # Check corresponding deep analysis
        deep_info = next((d for d in deep_results if d.get("candidate_name") == c_name), {})

        evidence = (
            f"Candidate: {c_name}\n"
            f"Target Role: {title}\n"
            f"JobMatcher Score: {score}/100\n"
            f"Must-Have Skills Required: {', '.join(must_have)}\n"
            f"Candidate Matched Skills: {', '.join(matched)}\n"
            f"Candidate Missing Must-Have Skills: {', '.join(missing) if missing else 'None'}\n"
            f"Experience: ~{exp} yrs\n"
            f"Deep Analysis Assessment: {deep_info.get('qualitative_assessment', '')}"
        )

        try:
            resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=evidence)])
            content = resp.content.strip()
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()
            eval_dict = json.loads(content)
            
            # Enforce hard must-have rule deterministically if LLM hallucinated a HIRE for missing must-have
            if missing and eval_dict.get("recommendation") == "HIRE":
                eval_dict["recommendation"] = "NO-HIRE"
                eval_dict["reasoning"] += f" (Overridden to NO-HIRE due to missing mandatory skills: {', '.join(missing)})"
        except Exception:
            # Deterministic fallback evaluation logic
            if missing or score < 50:
                rec = "NO-HIRE"
                reason = f"Candidate missing mandatory required skill(s): {', '.join(missing)}." if missing else "Match score below acceptable threshold."
                conf = 90
            else:
                rec = "HIRE"
                reason = f"Candidate satisfies all must-have requirements ({', '.join(matched)}) with {exp} years experience."
                conf = 85

            eval_dict = {
                "candidate_name": c_name,
                "recommendation": rec,
                "confidence": conf,
                "reasoning": reason,
                "strengths": [f"Matched {s}" for s in matched],
                "gaps": [f"Missing {s}" for s in missing],
                "risks": [f"Unverified proficiency in {s}" for s in missing] if missing else ["None"],
                "interview_focus_areas": [f"Verify hands-on experience in {s}" for s in matched[:2]]
            }

        eval_dict["candidate_name"] = c_name
        eval_dict["match_score"] = score
        final_recs.append(eval_dict)

    return {
        "screening_round": 3,
        "screening_status": "COMPLETED",
        "final_recommendations": final_recs
    }


def screening_report_node(state: AgentState) -> Dict[str, Any]:
    """
    Multi-Round Screening Report Node.
    Formats a transparent, executive multi-stage screening report covering:
    - Round 1: Initial Top 10 Shortlist
    - Round 2: Deep Analysis Findings
    - Round 3: Top 3 Final Recommendations (HIRE / NO-HIRE)
    """
    parsed_jd = state.get("parsed_jd", {})
    initial_cands = state.get("initial_candidates", [])
    deep_results = state.get("deep_analysis_results", [])
    final_recs = state.get("final_recommendations", [])

    title = parsed_jd.get("title", "Job Role")
    must_have = parsed_jd.get("must_have_skills", [])

    lines = [
        f"# 🏆 Multi-Round Candidate Screening Funnel Report",
        f"**Role**: {title} | **Must-Have Criteria**: {', '.join(must_have) if must_have else 'Standard'}",
        "---",
        "## 📊 Screening Funnel Summary",
        f"- **Initial Candidate Pool Evaluated**: ~100 resumes in vector database",
        f"- **Round 1 Initial Shortlist**: Top {len(initial_cands)} candidates selected via hybrid vector + BM25 search",
        f"- **Round 2 Deep Analysis**: Top {len(deep_results)} candidates evaluated for evidence & skill gaps",
        f"- **Round 3 Final Evaluation**: Top {len(final_recs)} finalists evaluated for final decision",
        "---",
        "## 🥇 Round 3 Final Recommendations (Top Finalists)"
    ]

    for rec in final_recs:
        name = rec.get("candidate_name", "Candidate")
        decision = rec.get("recommendation", "NO-HIRE")
        score = rec.get("match_score", 0)
        reasoning = rec.get("reasoning", "")
        strengths = rec.get("strengths", [])
        gaps = rec.get("gaps", [])
        focus = rec.get("interview_focus_areas", [])

        icon = "✅" if decision == "HIRE" else "❌"
        lines.append(f"### {icon} {name} — Recommendation: **{decision}** (Initial Match Score: {score}/100)")
        lines.append(f"**Decision Rationale**: {reasoning}")
        if strengths:
            lines.append(f"- **Key Strengths**: {', '.join(strengths)}")
        if gaps:
            lines.append(f"- **Identified Gaps**: {', '.join(gaps)}")
        if focus:
            lines.append(f"- **Interview Focus Areas**: {', '.join(focus)}")
        lines.append("")

    lines.append("---")
    lines.append("## 🔍 Round 1 & 2 Shortlist Breakdown")
    for cand in initial_cands:
        lines.append(f"- **{cand.get('candidate_name')}** — Match Score: {cand.get('match_score')}/100 | Skills: {', '.join(cand.get('matched_skills', []))}")

    report_text = "\n".join(lines).strip()
    return {
        "report": report_text,
        "screening_results": {
            "summary": report_text,
            "final_recommendations": final_recs
        },
        "messages": [AIMessage(content=report_text)]
    }
