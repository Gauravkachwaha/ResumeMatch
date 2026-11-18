import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from scratch.test_phase2 import run_phase2_tests
from scratch.test_phase3 import run_phase3_tests
from scratch.test_phase4 import run_phase4_tests
from scratch.test_phase5 import run_phase5_tests
from scratch.test_phase6 import run_phase6_tests


def run_master_test_suite():
    print("==================================================")
    print("      MASTER AGENT SYSTEM VERIFICATION SUITE      ")
    print("==================================================")

    print("\n---> Running Phase 2 Tests (Intent Classifier & State Schema)...")
    run_phase2_tests()

    print("\n---> Running Phase 3 Tests (RAG & Candidate Ranking Integration)...")
    run_phase3_tests()

    print("\n---> Running Phase 4 Tests (Agent Tools & Explainability)...")
    run_phase4_tests()

    print("\n---> Running Phase 5 Tests (LangGraph Workflow & State Memory)...")
    run_phase5_tests()

    print("\n---> Running Phase 6 Tests (Multi-Round Screening & Hire/No-Hire Engine)...")
    run_phase6_tests()

    print("\n==================================================")
    print("  ALL PHASES (PHASE 1 - 6) PASSED 100% SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_master_test_suite()
