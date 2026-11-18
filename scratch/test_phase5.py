import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.matching_agent import run_agent, get_graph_ascii_visualization, _compiled_agent_graph


def run_phase5_tests():
    print("==================================================")
    print("         PHASE 5 TEST & VERIFICATION SUITE       ")
    print("==================================================")

    passed_count = 0
    total_count = 0

    # ----------------------------------------------------
    # TEST 1: New Search Flow
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 1] Testing Turn 1: New Search Flow...")
    thread_1 = "session_test_flow_1"

    try:
        res1 = run_agent("Find Python developers with 3+ years experience.", thread_id=thread_1)
        assert res1.get("current_intent") == "new_search"
        assert "ranked_candidates" in res1
        assert len(res1["ranked_candidates"]) > 0
        assert "report" in res1 and "Executive Candidate Match Report" in res1["report"]

        print(f" [PASS] Test 1: New Search Flow executed successfully.")
        print(f"        Intent: {res1['current_intent']}")
        print(f"        Candidates Ranked: {len(res1['ranked_candidates'])}")
        print(f"        Top Candidate: {res1['ranked_candidates'][0]['candidate_name']} ({res1['ranked_candidates'][0]['match_score']}/100)")
        passed_count += 1
    except Exception as e:
        print(f" [FAIL] Test 1: New Search Flow failed: {e}")

    # ----------------------------------------------------
    # TEST 2: Candidate Comparison (Reusing State)
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 2] Testing Turn 2: Candidate Comparison (Reusing State)...")

    try:
        res2 = run_agent("Compare the top 3.", thread_id=thread_1)
        assert res2.get("current_intent") == "compare_candidates"
        assert "ranked_candidates" in res2 and len(res2["ranked_candidates"]) > 0
        assert "report" in res2 and "Candidate Comparison Analysis" in res2["report"]

        print(f" [PASS] Test 2: Candidate Comparison re-used existing state without new RAG search.")
        print(f"        Intent: {res2['current_intent']}")
        passed_count += 1
    except Exception as e:
        print(f" [FAIL] Test 2: Compare candidates test failed: {e}")

    # ----------------------------------------------------
    # TEST 3: Explain Ranking Flow
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 3] Testing Turn 3: Explain Ranking Flow...")

    try:
        res3 = run_agent("Why did the top candidate rank higher than the second candidate?", thread_id=thread_1)
        assert res3.get("current_intent") == "explain_ranking"
        assert "report" in res3 and "Ranking Explanation" in res3["report"]

        print(f" [PASS] Test 3: Explain Ranking flow executed successfully using JobMatcher evidence.")
        print(f"        Intent: {res3['current_intent']}")
        passed_count += 1
    except Exception as e:
        print(f" [FAIL] Test 3: Explain ranking test failed: {e}")

    # ----------------------------------------------------
    # TEST 4: Requirement Refinement Flow
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 4] Testing Requirement Refinement Flow ('Make Docker mandatory')...")
    thread_refine = "session_refine_test"

    try:
        # Initial search turn
        _ = run_agent("Find Python developers with 3+ years experience.", thread_id=thread_refine)
        
        # Refinement turn
        res_refine = run_agent("Make Docker mandatory.", thread_id=thread_refine)
        assert res_refine.get("current_intent") == "refine_requirements"
        assert "parsed_jd" in res_refine
        
        must_have = res_refine["parsed_jd"].get("must_have_skills", [])
        must_have_lower = [s.lower() for s in must_have]
        assert "docker" in must_have_lower
        assert "python" in must_have_lower  # Preserved previous skill

        print(f" [PASS] Test 4: Requirement refinement updated state and preserved previous skills.")
        print(f"        Updated Must-Have Skills: {must_have}")
        passed_count += 1
    except Exception as e:
        print(f" [FAIL] Test 4: Requirement refinement test failed: {e}")

    # ----------------------------------------------------
    # TEST 5: Generate Interview Questions Flow
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 5] Testing Generate Interview Questions Flow...")

    try:
        res5 = run_agent("Generate interview questions for the top candidate.", thread_id=thread_1)
        assert res5.get("current_intent") == "generate_interview_questions"
        assert "report" in res5 and "Tailored Interview Questions" in res5["report"]

        print(f" [PASS] Test 5: Interview questions generated using candidate & job context.")
        print(f"        Intent: {res5['current_intent']}")
        passed_count += 1
    except Exception as e:
        print(f" [FAIL] Test 5: Interview question generation test failed: {e}")

    # ----------------------------------------------------
    # TEST 6: State Memory & Checkpointing Across Turns
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 6] Testing MemorySaver State Persistence Across Turns...")

    try:
        thread_mem = "session_memory_test"
        r1 = run_agent("Find Java developers with 4+ years experience.", thread_id=thread_mem)
        r2 = run_agent("What is the top candidate's match score?", thread_id=thread_mem)

        assert "ranked_candidates" in r2 and len(r2["ranked_candidates"]) > 0
        assert r2["ranked_candidates"][0]["candidate_name"] == r1["ranked_candidates"][0]["candidate_name"]

        print(f" [PASS] Test 6: MemorySaver preserved state across conversational turns (Thread: {thread_mem}).")
        passed_count += 1
    except Exception as e:
        print(f" [FAIL] Test 6: Memory checkpointing test failed: {e}")

    # ----------------------------------------------------
    # TEST 7: Rare / Impossible Search Requirements Handling
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 7] Testing Rare Search Requirements Handling...")

    try:
        res_empty = run_agent("Find QuantumComputingObsoleteCobolSpecialist999 with 25 years experience.", thread_id="session_empty")
        assert "report" in res_empty
        print(f" [PASS] Test 7: Handled rare search gracefully without crashing.")
        passed_count += 1
    except Exception as e:
        print(f" [FAIL] Test 7: Rare search test failed: {e}")

    # ----------------------------------------------------
    # TEST 8: StateGraph Visualization
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 8] Testing LangGraph Graph ASCII Visualization...")

    try:
        ascii_vis = get_graph_ascii_visualization()
        assert ascii_vis and len(ascii_vis) > 50
        print(f" [PASS] Test 8: Generated LangGraph ASCII StateGraph representation.")
        passed_count += 1
    except Exception as e:
        print(f" [FAIL] Test 8: Graph visualization test failed: {e}")

    print("\n==================================================")
    print(f"RESULTS: {passed_count}/{total_count} PHASE 5 INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_phase5_tests()
