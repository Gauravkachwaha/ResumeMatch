import json
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from src.agent_state import IntentType
from src.llm_config import get_openrouter_llm


class IntentResponse(BaseModel):
    """
    Pydantic schema for structured intent classification output.
    """
    intent: IntentType = Field(
        ...,
        description="The classified intent of the user's request. Must be one of the allowed intent values."
    )
    reason: str = Field(
        ...,
        description="Brief natural language explanation for why this intent was selected."
    )


ALLOWED_INTENTS = [
    "new_search",
    "refine_requirements",
    "compare_candidates",
    "explain_ranking",
    "generate_interview_questions",
    "screen_candidates",
    "final_recommendation",
    "general_question"
]


SYSTEM_PROMPT = """You are an AI Recruiter Intent Classifier.
Your task is to analyze the user's latest input (and conversation history, if provided) and determine the user's intent.

You MUST classify the intent into EXACTLY ONE of the following 8 allowed categories:
1. "new_search": Initial request or brand new search for candidates with specific skills, experience, or job description (e.g., "Find me Python developers with 3+ years experience").
2. "refine_requirements": Modifying, adding, removing, or tweaking job requirements or filters for an existing search (e.g., "Make Docker mandatory", "Remove AWS", "Change min experience to 5 years").
3. "compare_candidates": Asking to compare candidates side-by-side or analyze trade-offs between candidates (e.g., "Compare the top 3 candidates side by side").
4. "explain_ranking": Asking why a specific candidate ranked higher than another, or why a candidate received a certain score (e.g., "Why did John rank higher than Jane?").
5. "generate_interview_questions": Asking to generate interview questions, technical questions, or screening questions for candidate(s) (e.g., "Generate interview questions for John").
6. "screen_candidates": Asking to run multi-round screening, evaluate candidates more deeply, or filter candidate pool (e.g., "Screen the top 10 candidates more deeply").
7. "final_recommendation": Asking for a final hire / no-hire recommendation or decision advice (e.g., "Should we hire John?", "Give me the final hire recommendation").
8. "general_question": Any general question, clarification, or conversational input that does not fit the above categories (e.g., "What does John's current project involve?").

IMPORTANT RULES:
- Never invent a new intent category outside the 8 listed above.
- Respond strictly in valid JSON format matching the schema: {"intent": "<intent_string>", "reason": "<explanation_string>"}.
"""


class IntentClassifier:
    """
    Intent Classifier module utilizing OpenRouter LLM (openrouter/free)
    to classify natural language recruiter requests into structured intents.
    """

    def __init__(self, temperature: float = 0.0):
        self.llm = get_openrouter_llm(temperature=temperature)

    def classify(self, user_input: str, history: Optional[List[BaseMessage]] = None) -> Dict[str, Any]:
        """
        Classify the intent of user_input taking optional conversation history into account.
        Returns a dict: {"intent": <IntentType>, "reason": str}
        """
        if not user_input or not user_input.strip():
            return {
                "intent": "general_question",
                "reason": "Empty input provided."
            }

        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Append relevant conversation history if available
        if history:
            for msg in history[-4:]:
                messages.append(msg)

        # Append current user prompt
        prompt_content = f"Classify the following user request:\n\"{user_input}\""
        messages.append(HumanMessage(content=prompt_content))

        # Try structured output via LLM
        try:
            structured_llm = self.llm.with_structured_output(IntentResponse)
            result = structured_llm.invoke(messages)

            if isinstance(result, IntentResponse):
                intent_val = result.intent if result.intent in ALLOWED_INTENTS else "general_question"
                return {
                    "intent": intent_val,
                    "reason": result.reason
                }
            elif isinstance(result, dict):
                intent_val = result.get("intent", "general_question")
                if intent_val not in ALLOWED_INTENTS:
                    intent_val = "general_question"
                return {
                    "intent": intent_val,
                    "reason": result.get("reason", "Classified intent.")
                }
        except Exception:
            # Fallback to direct raw invoke + JSON extraction if with_structured_output is unsupported by free endpoint
            try:
                raw_response = self.llm.invoke(messages)
                content = raw_response.content.strip()

                if "```" in content:
                    content = content.split("```")[1].replace("json", "").strip()

                parsed = json.loads(content)
                intent_val = parsed.get("intent", "general_question")
                if intent_val not in ALLOWED_INTENTS:
                    intent_val = "general_question"
                return {
                    "intent": intent_val,
                    "reason": parsed.get("reason", "Parsed from raw response.")
                }
            except Exception as inner_err:
                return self._rule_based_fallback(user_input, f"LLM classification unavailable ({inner_err}). Used fallback classifier.")

        return {
            "intent": "general_question",
            "reason": "Default fallback classification."
        }

    def _rule_based_fallback(self, text: str, reason_prefix: str = "") -> Dict[str, Any]:
        """
        Deterministic rule-based fallback for intent classification when LLM API is unavailable.
        """
        t = text.lower()
        if any(w in t for w in ["compare", "side by side", "versus", "vs"]):
            intent = "compare_candidates"
        elif any(w in t for w in ["why did", "why rank", "rank higher", "ranked higher", "scoring explanation"]):
            intent = "explain_ranking"
        elif any(w in t for w in ["interview question", "questions for", "technical questions", "screening questions"]):
            intent = "generate_interview_questions"
        elif any(w in t for w in ["screen", "screening", "round 1", "round 2", "funnel"]):
            intent = "screen_candidates"
        elif any(w in t for w in ["should we hire", "hire recommendation", "final decision", "hire/no-hire", "hire john"]):
            intent = "final_recommendation"
        elif any(w in t for w in ["make", "mandatory", "remove", "change", "actually", "require", "add skill"]):
            intent = "refine_requirements"
        elif any(w in t for w in ["find", "search", "looking for", "developer", "engineer", "experience", "need a"]):
            intent = "new_search"
        else:
            intent = "general_question"

        return {
            "intent": intent,
            "reason": f"{reason_prefix} Rule-matched keyword in user text."
        }
