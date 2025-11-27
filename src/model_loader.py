import os
from dotenv import load_dotenv

from llama_index.llms.google_genai import GoogleGenAI

from src.config import (
    LLM_MODEL,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_REPETITION_PENALTY
)


# Load environment variables from the .env file
load_dotenv()


def initialise_llm() -> GoogleGenAI:
    """Initialises the GoogleGenAI LLM with core parameters from config."""

    api_key: str | None = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Make sure it's set in your .env file."
        )

    return GoogleGenAI(
        api_key=api_key,
        model=LLM_MODEL,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        repetition_penalty=LLM_REPETITION_PENALTY
    )