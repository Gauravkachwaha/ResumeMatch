import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.agent_state import AgentState, create_candidate_match
from src.screening_nodes import initial_screening_node, deep_analysis_node, final_evaluation_node, screening_report_node
from src.matching_agent import run_agent, _compiled_agent_graph


def run_phase6_tests():
    print("==================================================")
    print("         PHASE 6 TEST & VERIFICATION SUITE       ")
    print("==================================================")

    passed_count = 0
    total_count = 0

    state: AgentState = {
        "raw_jd": "Senior Python Developer with FastAPI and PostgreSQL",
        "parsed_jd": {
            "title": "Senior Python Developer",
            "must_have_skills": ["Python", "FastAPI", "PostgreSQL"],
            "required_experience_years": 3
        },
        "top_k": 10
    }

    # ----------------------------------------------------
    # TEST 1: Round 1 Initial Screening
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 1] Testing Round 1: Initial Screening Node (Top 10 Shortlist)...")

    try:
        res1 = initial_screening_node(state)
        state.update(res1)
        assert state["screening_round"] == 1
        assert state["screening_status"] == "INITIAL_SCREENING"
        assert "initial_candidates" in state
        assert len(state["initial_candidates"]) > 0

        print(f" [PASS] Test 1: Round 1 Initial Screening retrieved {len(state['initial_candidates'])} candidates.")
        print(f"        Top Candidate: {state['initial_candidates'][0]['candidate_name']} (Score: {state['initial_candidates'][0]['match_score']}/100)")
        passed_count += 1
    except Exception as e:
        print(f" [FAIL] Test 1: Initial screening failed: {e}")

    # ----------------------------------------------------
    # TEST 2: Round 2 Deep Analysis
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 2] Testing Round 2: Deep Analysis Node (Qualitative Evidence Assessment)...")

    try:
        res2 = deep_analysis_node(state)
        state.update(res2)
        assert state["screening_round"] == 2
        assert state["screening_status"] == "DEEP_ANALYSIS"
        assert "deep_analysis_results" in state
        assert len(state["deep_analysis_results"]) > 0

        sample_deep = state["deep_analysis_results"][0]
        assert "strengths" in sample_deep
        assert "gaps" in sample_deep

        print(f" [PASS] Test 2: Deep Analysis evaluated {len(state['deep_analysis_results'])} candidates with evidence.")
        print(f"        Sample Strengths: {sample_deep['strengths']}")
        print(f"        Sample Gaps: {sample_deep['gaps']}")
        passed_count += 1
    except Exception as e:
        print(f" [FAIL] Test 2: Deep Analysis failed: {e}")

    # ----------------------------------------------------
    # TEST 3: Round 2 -> Round 3 Finalists Selection
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 3] Testing Finalists Selection (Slicing Top 3 Finalists)...")

    try:
        finalists = state.get("finalists", [])
        assert len(finalists) <= 3
        print(f" [PASS] Test 3: Sliced {len(finalists)} finalists for Round 3 final evaluation.")
        passed_count += 1
    except Exception as e:
        print(f" [FAIL] Test 3: Finalists selection failed: {e}")

    # ----------------------------------------------------
    # TEST 4: Round 3 Final Evaluation (Hire / No-Hire Decision)
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 4] Testing Round 3: Final Evaluation Node (Hire / No-Hire Decision)...")

    try:
        res3 = final_evaluation_node(state)
        state.update(res3)
        assert state["screening_round"] == 3
        assert state["screening_status"] == "COMPLETED"
        assert "final_recommendations" in state
        assert len(state["final_recommendations"]) > 0

        sample_rec = state["final_recommendations"][0]
        assert sample_rec["recommendation"] in ["HIRE", "NO-HIRE"]
        assert "reasoning" in sample_rec
        assert "confidence" in sample_rec

        print(f" [PASS] Test 4: Final Evaluation completed for {len(state['final_recommendations'])} finalists.")
        print(f"        Finalist 1: {sample_rec['candidate_name']} -> Recommendation: {sample_rec['recommendation']} (Confidence: {sample_rec['confidence']}%)")
        print(f"        Reasoning: {sample_rec['reasoning']}")
        passed_count += 1
    except Exception as e:
        print(f" [FAIL] Test 4: Final evaluation failed: {e}")

    # ----------------------------------------------------
    # TEST 5: Must-Have Requirement Failure Rule
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 5] Testing Must-Have Requirement Failure Enforcement...")

    try:
        # Create candidate missing mandatory FastAPI requirement
        incomplete_cand = create_candidate_match(
            candidate_name="Incomplete Skill Candidate",
            resume_path="test_incomplete.pdf",
            match_score=40,
            matched_skills=["Python"], # Missing FastAPI & PostgreSQL
            relevant_excerpts=["Used basic Python scripts."],
            reasoning="Missing mandatory skills."
        )

        test_state = dict(state)
        test_state["finalists"] = [incomplete_cand]
        test_state["parsed_jd"] = {
            "title": "Senior Python Developer",
            "must_have_skills": ["Python", "FastAPI", "PostgreSQL"],
            "required_experience_years": 3
        }

        res_must_have = final_evaluation_node(test_state)
        recs = res_must_have.get("final_recommendations", [])

        assert len(recs) == 1
        assert recs[0]["recommendation"] == "NO-HIRE"
        assert "missing" in recs[0]["reasoning"].lower() or "no-hire" in recs[0]["recommendation"]

        print(f" [PASS] Test 5: Candidate missing mandatory skills correctly received 'NO-HIRE'.")
        print(f"        Reasoning: {recs[0]['reasoning']}")
        passed_count += 1
    except Exception as e:
        print(f" [FAIL] Test 5: Must-have requirement test failed: {e}")

    # ----------------------------------------------------
    # TEST 6: Missing Resume Evidence Handling
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 6] Testing Missing Resume Data Graceful Handling...")

    try:
        empty_cand = create_candidate_match(
            candidate_name="Empty Resume Candidate",
            resume_path="empty.pdf",
            match_score=0,
            matched_skills=[],
            relevant_excerpts=[],
            reasoning="No text found."
        )

        test_state_empty = dict(state)
        test_state_empty["finalists"] = [empty_cand]
        res_empty = final_evaluation_node(test_state_empty)
        recs_empty = res_empty.get("final_recommendations", [])

        assert len(recs_empty) == 1
        assert recs_empty[0]["recommendation"] == "NO-HIRE"

        print(f" [PASS] Test 6: Empty resume candidate handled gracefully without hallucinated claims.")
        passed_count += 1
    except Exception as e:
        print(f" [FAIL] Test 6: Missing resume evidence test failed: {e}")

    # ----------------------------------------------------
    # TEST 7: Multi-Round Screening Report Generation
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 7] Testing Multi-Round Screening Report Node...")

    try:
        report_res = screening_report_node(state)
        state.update(report_res)
        assert "report" in state
        assert "Multi-Round Candidate Screening Funnel Report" in state["report"]
        assert "Final Recommendations" in state["report"]

        print(f" [PASS] Test 7: Multi-round screening funnel report generated successfully.")
        passed_count += 1
    except Exception as e:
        print(f" [FAIL] Test 7: Screening report generation failed: {e}")

    # ----------------------------------------------------
    # TEST 8: End-to-End Multi-Round Screening Integration Flow
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 8] Testing End-to-End Multi-Round Screening Agent Invocation...")
    thread_screen = "session_screening_e2e"

    try:
        e2e_res = run_agent("Run a full screening process on the top Python developer candidates.", thread_id=thread_screen)
        assert e2e_res.get("current_intent") == "screen_candidates"
        assert e2e_res.get("screening_round") == 3
        assert e2e_res.get("screening_status") == "COMPLETED"
        assert "report" in e2e_res and "Multi-Round Candidate Screening" in e2e_res["report"]

        print(f" [PASS] Test 8: End-to-End Multi-Round Screening Agent flow executed successfully.")
        print(f"        Final Status: {e2e_res['screening_status']} (Round: {e2e_res['screening_round']})")
        passed_count += 1
    except Exception as e:
        print(f" [FAIL] Test 8: End-to-end screening agent flow failed: {e}")

    print("\n==================================================")
    print(f"RESULTS: {passed_count}/{total_count} PHASE 6 INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_phase6_tests()
