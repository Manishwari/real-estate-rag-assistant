"""
LLM generation layer. Defaults to Groq (free tier, fast Llama models —
no cost to run this assignment) but is written against a tiny common
interface so swapping in OpenAI or Anthropic is a one-line change.

Set the backend via the GENERATOR_BACKEND env var / Streamlit secret:
    "groq"      (default) - needs GROQ_API_KEY      https://console.groq.com
    "openai"              - needs OPENAI_API_KEY
    "anthropic"           - needs ANTHROPIC_API_KEY
"""
from __future__ import annotations

import os

SYSTEM_PROMPT = """You are a helpful, precise Real Estate AI Assistant for a property \
platform. Answer the user's question using ONLY the information in the provided context \
chunks below, which come from the company's real project brochures, policies, FAQs, RERA \
documents, and other official records.

Rules:
1. Base your answer strictly on the provided context. Do not invent prices, dates, RERA \
numbers, or any other fact that is not present in the context.
2. If the context does not contain enough information to answer confidently, say so plainly \
(e.g. "I don't have that information in the knowledge base") instead of guessing.
3. When multiple context chunks are relevant, synthesize them into one coherent answer.
4. Keep answers concise and well-organized (use short paragraphs or bullet points when helpful).
5. Use the conversation history to resolve follow-up questions and pronouns (e.g. "what about \
its possession date?" refers to whatever project was last discussed).
6. Do not mention "chunks" or "context" to the user — just answer naturally, as a knowledgeable \
real estate assistant would.
"""


def _build_messages(question: str, context_blocks: list[str], history: list[dict]) -> list[dict]:
    context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no relevant documents found)"
    user_turn = (
        f"Context from the knowledge base:\n\n{context_text}\n\n"
        f"---\n\nUser question: {question}"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # keep the last few turns of real conversation for multi-turn memory
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": user_turn})
    return messages


def _generate_groq(question: str, context_blocks: list[str], history: list[dict]) -> str:
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    client = Groq(api_key=api_key)
    messages = _build_messages(question, context_blocks, history)
    resp = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=messages,
        temperature=0.2,
        max_tokens=800,
    )
    return resp.choices[0].message.content


def _generate_openai(question: str, context_blocks: list[str], history: list[dict]) -> str:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)
    messages = _build_messages(question, context_blocks, history)
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0.2,
        max_tokens=800,
    )
    return resp.choices[0].message.content


def _generate_anthropic(question: str, context_blocks: list[str], history: list[dict]) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    client = anthropic.Anthropic(api_key=api_key)
    messages = _build_messages(question, context_blocks, history)
    system = messages[0]["content"]
    convo = messages[1:]
    resp = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        system=system,
        messages=convo,
        max_tokens=800,
    )
    return "".join(block.text for block in resp.content if block.type == "text")


_BACKENDS = {
    "groq": _generate_groq,
    "openai": _generate_openai,
    "anthropic": _generate_anthropic,
}


def generate_answer(question: str, context_blocks: list[str], history: list[dict]) -> str:
    backend = os.environ.get("GENERATOR_BACKEND", "groq").lower()
    fn = _BACKENDS.get(backend)
    if fn is None:
        raise ValueError(f"Unknown GENERATOR_BACKEND '{backend}'. Choose from {list(_BACKENDS)}.")
    return fn(question, context_blocks, history)
