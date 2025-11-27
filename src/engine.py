from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.llms.google_genai import GoogleGenAI

from src.config import LLM_SYSTEM_PROMPT, MEMORY_TOKEN_LIMIT
from src.model_loader import initialise_llm


def main_chat_loop() -> None:
    """Main chat loop to ask a question to the LLM and print the answer."""

    llm: GoogleGenAI = initialise_llm()

    memory = ChatSummaryMemoryBuffer.from_defaults(
        chat_history=[],
        llm=llm, 
        token_limit=MEMORY_TOKEN_LIMIT
    )

    conversation: SimpleChatEngine = SimpleChatEngine.from_defaults(
        llm=llm,
        memory=memory,
        system_prompt=LLM_SYSTEM_PROMPT
    )

    conversation.chat_repl()
