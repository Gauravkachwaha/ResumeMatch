import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage



from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent_state import AgentState
from src.agent_nodes import (
    intent_classifier_node,
    parse_and_refine_requirements_node,
    search_resumes_node,
    rank_candidates_node,
    generate_report_node,
    compare_candidates_node,
    explain_ranking_node,
    generate_interview_questions_node,
    general_response_node,
    human_feedback_node
)
from src.screening_nodes import (
    initial_screening_node,
    deep_analysis_node,
    final_evaluation_node,
    screening_report_node
)



def route_by_intent(state: AgentState) -> str:
    """
    Centralized LangGraph conditional router function.
    Inspects state["current_intent"] and routes execution to the appropriate node.
    """
    intent = state.get("current_intent", "general_question")

    if intent in ["new_search", "refine_requirements"]:
        return "parse_and_refine"
    elif intent in ["screen_candidates", "final_recommendation"]:
        return "initial_screening_node"
    elif intent == "compare_candidates":
        return "compare_candidates_node"
    elif intent == "explain_ranking":
        return "explain_ranking_node"
    elif intent == "generate_interview_questions":
        return "generate_interview_questions_node"
    else:
        return "general_response_node"


def build_agent_graph(checkpointer: Optional[MemorySaver] = None):
    """
    Assembles and compiles the LangGraph StateGraph state machine.
    """
    workflow = StateGraph(AgentState)

    # 1. Add Core Agent Nodes
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("parse_and_refine", parse_and_refine_requirements_node)
    workflow.add_node("search_resumes", search_resumes_node)
    workflow.add_node("rank_candidates", rank_candidates_node)
    workflow.add_node("generate_report", generate_report_node)
    workflow.add_node("compare_candidates_node", compare_candidates_node)
    workflow.add_node("explain_ranking_node", explain_ranking_node)
    workflow.add_node("generate_interview_questions_node", generate_interview_questions_node)
    workflow.add_node("general_response_node", general_response_node)
    workflow.add_node("human_feedback_node", human_feedback_node)

    # Add Phase 6 Multi-Round Screening Nodes
    workflow.add_node("initial_screening_node", initial_screening_node)
    workflow.add_node("deep_analysis_node", deep_analysis_node)
    workflow.add_node("final_evaluation_node", final_evaluation_node)
    workflow.add_node("screening_report_node", screening_report_node)

    # 2. Add Edges
    # Entry Point -> Intent Classifier
    workflow.add_edge(START, "intent_classifier")

    # Conditional Routing from Intent Classifier
    workflow.add_conditional_edges(
        "intent_classifier",
        route_by_intent,
        {
            "parse_and_refine": "parse_and_refine",
            "initial_screening_node": "initial_screening_node",
            "compare_candidates_node": "compare_candidates_node",
            "explain_ranking_node": "explain_ranking_node",
            "generate_interview_questions_node": "generate_interview_questions_node",
            "general_response_node": "general_response_node"
        }
    )

    # Normal Search & Ranking Pipeline Edges
    workflow.add_edge("parse_and_refine", "search_resumes")
    workflow.add_edge("search_resumes", "rank_candidates")
    workflow.add_edge("rank_candidates", "generate_report")
    workflow.add_edge("generate_report", "human_feedback_node")

    # Phase 6 Multi-Round Screening Pipeline Edges:
    # initial_screening (Round 1: Top 10) -> deep_analysis (Round 2: Top 3) -> final_evaluation (Round 3: Hire/No-Hire) -> screening_report -> human_feedback
    workflow.add_edge("initial_screening_node", "deep_analysis_node")
    workflow.add_edge("deep_analysis_node", "final_evaluation_node")
    workflow.add_edge("final_evaluation_node", "screening_report_node")
    workflow.add_edge("screening_report_node", "human_feedback_node")

    # Tool Nodes Edges -> human_feedback_node -> END
    workflow.add_edge("compare_candidates_node", "human_feedback_node")
    workflow.add_edge("explain_ranking_node", "human_feedback_node")
    workflow.add_edge("generate_interview_questions_node", "human_feedback_node")
    workflow.add_edge("general_response_node", "human_feedback_node")
    
    workflow.add_edge("human_feedback_node", END)

    # Checkpointer for conversation state persistence
    saver = checkpointer or MemorySaver()

    return workflow.compile(checkpointer=saver)



# Singleton compiled graph instance
_agent_checkpointer = MemorySaver()
_compiled_agent_graph = build_agent_graph(checkpointer=_agent_checkpointer)


def run_agent(user_message: str, thread_id: str = "default_session") -> Dict[str, Any]:
    """
    Public invocation API for Streamlit.
    Runs the LangGraph agent in a dedicated background thread with its own
    asyncio event loop, so it never conflicts with Streamlit/Tornado's event loop.
    """
    import threading

    config = {"configurable": {"thread_id": thread_id}}
    input_state = {
        "raw_jd": user_message,
        "messages": [HumanMessage(content=user_message)]
    }

    result_holder: Dict[str, Any] = {}

    def _invoke_in_thread() -> None:
        """Runs inside a daemon thread with its own isolated event loop."""
        import asyncio
        # Give this thread its own fresh event loop so it never touches Tornado's loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_holder["state"] = _compiled_agent_graph.invoke(input_state, config=config)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print(f"[run_agent ERROR] {exc}\n{tb}", flush=True)
            result_holder["state"] = {
                "raw_jd": user_message,
                "report": f"⚠️ Agent execution error: {exc}\n\nTraceback:\n```\n{tb}\n```",
                "current_intent": "error",
                "ranked_candidates": [],
                "messages": [],
            }
        finally:
            loop.close()

    t = threading.Thread(target=_invoke_in_thread, daemon=True)
    t.start()
    t.join(timeout=180)  # 3-minute hard timeout

    if t.is_alive():
        # Thread timed out
        return {
            "raw_jd": user_message,
            "report": "⚠️ Agent timed out after 3 minutes. The LLM API may be slow. Please try again.",
            "current_intent": "timeout",
            "ranked_candidates": [],
            "messages": [],
        }

    return result_holder.get("state", {
        "raw_jd": user_message,
        "report": "⚠️ Agent returned no result.",
        "current_intent": "error",
        "ranked_candidates": [],
        "messages": [],
    })


def get_graph_ascii_visualization() -> str:
    """
    Returns ASCII string representation of the compiled LangGraph StateGraph machine.
    """
    try:
        return _compiled_agent_graph.get_graph().draw_ascii()
    except Exception as e:
        return f"Graph Visualization Error: {e}"
