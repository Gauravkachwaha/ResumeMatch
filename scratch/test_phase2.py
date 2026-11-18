import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from langchain_core.messages import HumanMessage, AIMessage
from src.agent_state import AgentState, CandidateMatch, IntentType, create_candidate_match
from src.intent_classifier import IntentClassifier


def run_phase2_tests():
    print("==================================================")
    print("         PHASE 2 TEST & VERIFICATION SUITE       ")
    print("==================================================")
    
    passed_count = 0
    total_count = 0

    # 1. Test AgentState & CandidateMatch Construction
    total_count += 1
    try:
        match_item = create_candidate_match(
            candidate_name="Alice Smith",
            resume_path="data/resumes/alice.pdf",
            match_score=85,
            matched_skills=["Python", "FastAPI"],
            relevant_excerpts=["Built REST APIs in Python"],
            reasoning="Strong skill match."
        )
        assert match_item["candidate_name"] == "Alice Smith"
        assert match_item["match_score"] == 85

        state: AgentState = {
            "messages": [],
            "raw_jd": "Senior Python Developer",
            "parsed_jd": {"title": "Python Dev", "must_have_skills": ["Python"]},
            "filters": {},
            "ranked_candidates": [match_item],
            "top_k": 5,
            "report": "",
            "current_intent": "new_search",
            "next_action": "search_resumes"
        }
        assert state["current_intent"] == "new_search"
        print("[OK] Test 1: AgentState & CandidateMatch structures valid.")
        passed_count += 1
    except Exception as e:
        print(f"[FAIL] Test 1: AgentState/CandidateMatch failed: {e}")

    # 2. Test Intent Classifier Initialization
    total_count += 1
    try:
        classifier = IntentClassifier(temperature=0.0)
        print("[OK] Test 2: IntentClassifier initialized successfully.")
        passed_count += 1
    except Exception as e:
        print(f"[FAIL] Test 2: IntentClassifier initialization failed: {e}")
        return

    # 3. Test 8 Required Example Inputs
    test_cases = [
        ("Find me candidates with React and 3+ years experience.", "new_search"),
        ("Actually make Docker mandatory and remove AWS.", "refine_requirements"),
        ("Compare the top 3 candidates side by side.", "compare_candidates"),
        ("Why did John rank higher than Jane?", "explain_ranking"),
        ("Generate interview questions for John.", "generate_interview_questions"),
        ("Screen the top 10 candidates more deeply.", "screen_candidates"),
        ("Should we hire John?", "final_recommendation"),
        ("What does John's current project involve?", "general_question"),
    ]

    print("\nRunning Intent Classification Tests (Part 6):")
    print("--------------------------------------------------")

    for i, (prompt, expected_intent) in enumerate(test_cases, 1):
        total_count += 1
        res = classifier.classify(prompt)
        classified = res.get("intent")
        reason = res.get("reason", "")

        if classified == expected_intent:
            print(f" [PASS] Example {i}: \"{prompt}\"\n        -> Expected: {expected_intent} | Got: {classified}\n        -> Reason: {reason}\n")
            passed_count += 1
        else:
            print(f" [FAIL] Example {i}: \"{prompt}\"\n        -> Expected: {expected_intent} | Got: {classified}\n        -> Reason: {reason}\n")

    # 4. Test Intent Classification with Conversation History (Part 7)
    total_count += 1
    print("Running Conversation History Test (Part 7):")
    print("--------------------------------------------------")
    history = [
        HumanMessage(content="Find Java developers with 3+ years experience."),
        AIMessage(content="Found 10 matching Java candidates with Spring Boot and SQL experience.")
    ]
    followup_prompt = "Compare the top 3."
    res_history = classifier.classify(followup_prompt, history=history)
    classified_hist = res_history.get("intent")

    if classified_hist == "compare_candidates":
        print(f" [PASS] Follow-up Prompt: \"{followup_prompt}\"\n        -> Context: [Java Dev Search -> 10 candidates]\n        -> Got: {classified_hist} (Reason: {res_history.get('reason')})\n")
        passed_count += 1
    else:
        print(f" [FAIL] Follow-up Prompt: \"{followup_prompt}\"\n        -> Expected: compare_candidates | Got: {classified_hist}\n")

    print("==================================================")
    print(f"RESULTS: {passed_count}/{total_count} TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_phase2_tests()
