import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.agent_state import AgentState, CandidateMatch
from src.agent_nodes import search_resumes_node, rank_candidates_node
from src.job_matcher import JobMatcher


def run_phase3_tests():
    print("==================================================")
    print("         PHASE 3 TEST & VERIFICATION SUITE       ")
    print("==================================================")

    passed_count = 0
    total_count = 0

    # ----------------------------------------------------
    # TEST 1: New Search & Ranking Pipeline Flow
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 1] Testing New Search Pipeline (search_resumes_node -> rank_candidates_node)...")
    
    state: AgentState = {
        "raw_jd": "Looking for a Senior Python Developer with FastAPI, PostgreSQL, and 3+ years experience.",
        "parsed_jd": {
            "title": "Senior Python Developer",
            "must_have_skills": ["Python", "FastAPI", "PostgreSQL"],
            "required_experience_years": 3
        },
        "top_k": 10
    }

    try:
        # Step A: Search node
        search_res = search_resumes_node(state)
        state.update(search_res)
        assert "parsed_jd" in state
        assert state["parsed_jd"]["title"] == "Senior Python Developer"

        # Step B: Rank node
        rank_res = rank_candidates_node(state)
        state.update(rank_res)
        assert "ranked_candidates" in state
        ranked = state["ranked_candidates"]
        
        if len(ranked) > 0:
            print(f" [PASS] Test 1: Retrieved and ranked {len(ranked)} candidates.")
            print(f"        Top Candidate: {ranked[0]['candidate_name']} (Score: {ranked[0]['match_score']}/100)")
            passed_count += 1
        else:
            print(" [FAIL] Test 1: Expected ranked candidates, but list was empty.")
    except Exception as e:
        print(f" [FAIL] Test 1: Pipeline execution error: {e}")

    # ----------------------------------------------------
    # TEST 2: Top-K Constraint Enforcement
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 2] Testing Top-K Constraint Enforcement (top_k=3)...")

    state_top_k: AgentState = {
        "raw_jd": "Python Developer",
        "parsed_jd": {
            "title": "Python Developer",
            "must_have_skills": ["Python"],
            "required_experience_years": 1
        },
        "top_k": 3
    }

    try:
        search_res = search_resumes_node(state_top_k)
        state_top_k.update(search_res)
        rank_res = rank_candidates_node(state_top_k)
        state_top_k.update(rank_res)

        ranked = state_top_k.get("ranked_candidates", [])
        if len(ranked) <= 3:
            print(f" [PASS] Test 2: Returned {len(ranked)} candidates (<= top_k=3 limit).")
            passed_count += 1
        else:
            print(f" [FAIL] Test 2: Returned {len(ranked)} candidates, exceeding top_k limit of 3.")
    except Exception as e:
        print(f" [FAIL] Test 2: Top-K test failed: {e}")

    # ----------------------------------------------------
    # TEST 3: Zero Candidate Match / Rare Requirement Handling
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 3] Testing Rare/Zero Candidate Match Handling...")

    state_rare: AgentState = {
        "raw_jd": "Looking for a specialist in NonExistentSkillX999 and ObsoleteLang888.",
        "parsed_jd": {
            "title": "Rare Skill Specialist",
            "must_have_skills": ["NonExistentSkillX999", "ObsoleteLang888"],
            "required_experience_years": 15
        },
        "top_k": 5
    }

    try:
        search_res = search_resumes_node(state_rare)
        state_rare.update(search_res)
        rank_res = rank_candidates_node(state_rare)
        state_rare.update(rank_res)

        # The system should not crash even if scores are low or penalized
        ranked = state_rare.get("ranked_candidates", [])
        print(f" [PASS] Test 3: System handled rare query gracefully without crashing (Returned {len(ranked)} candidates).")
        passed_count += 1
    except Exception as e:
        print(f" [FAIL] Test 3: Zero/Rare match test crashed: {e}")

    # ----------------------------------------------------
    # TEST 4: Existing JobMatcher Verification
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 4] Verifying Integration with Existing JobMatcher...")

    try:
        matcher = JobMatcher()
        raw_result = matcher.match(
            jd_input={"title": "Data Scientist", "must_have_skills": ["Python", "Machine Learning"], "required_experience_years": 2},
            top_k=2
        )
        assert "top_matches" in raw_result
        print(f" [PASS] Test 4: Existing JobMatcher successfully invoked directly (Returned {len(raw_result['top_matches'])} matches).")
        passed_count += 1
    except Exception as e:
        print(f" [FAIL] Test 4: Existing JobMatcher call failed: {e}")

    # ----------------------------------------------------
    # TEST 5: State Update & CandidateMatch Structure Verification
    # ----------------------------------------------------
    total_count += 1
    print("\n[Test 5] Verifying State Population with CandidateMatch TypedDict...")

    if "ranked_candidates" in state and len(state["ranked_candidates"]) > 0:
        cand_item = state["ranked_candidates"][0]
        required_keys = ["candidate_name", "resume_path", "match_score", "matched_skills", "relevant_excerpts", "reasoning"]
        missing = [k for k in required_keys if k not in cand_item]

        if not missing:
            print(f" [PASS] Test 5: state['ranked_candidates'] fully populated with all keys: {required_keys}")
            passed_count += 1
        else:
            print(f" [FAIL] Test 5: Missing keys in candidate state object: {missing}")
    else:
        print(" [FAIL] Test 5: Candidate state was not populated.")

    print("\n==================================================")
    print(f"RESULTS: {passed_count}/{total_count} PHASE 3 TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_phase3_tests()
