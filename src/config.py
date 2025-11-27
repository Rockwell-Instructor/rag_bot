# --- LLM Model Configuration ---

# Model Selection
LLM_MODEL: str = "gemini-2.5-flash-lite"
# Token Generation Parameters
LLM_MAX_NEW_TOKENS: int = 768
LLM_TEMPERATURE: float = 0.01
LLM_TOP_P: float = 0.95
LLM_REPETITION_PENALTY: float = 1.03
# Memory Config
MEMORY_TOKEN_LIMIT: int = 1500
# -- System Prompt --
LLM_SYSTEM_PROMPT: str = (
    "You are a helpful chatbot. Be friendly and conversational."
)