import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_openrouter_llm(temperature: float = 0.2) -> ChatOpenAI:
    """
    Initialize and return a ChatOpenAI client configured to route requests
    to OpenRouter using the openrouter/free model.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model_name = os.getenv("LLM_MODEL", "openrouter/free")
    base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")

    if not api_key:
        print("[WARNING] OPENROUTER_API_KEY is not set in environment variables. Please set it in your .env file.")

    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key or "missing_key",
        openai_api_base=base_url,
        temperature=temperature,
        default_headers={
            "HTTP-Referer": "https://github.com/Siddhant20104111/resume-rag-job-matcher-agent",
            "X-Title": "Resume RAG Job Matcher Agent"
        }
    )
