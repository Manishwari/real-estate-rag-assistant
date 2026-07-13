"""
Orchestrates retrieval + generation into a single `answer()` call, and
handles the "no relevant information" case gracefully instead of letting
the LLM hallucinate.
"""
from __future__ import annotations

from .generator import generate_answer
from .retriever import Retriever

RELEVANCE_THRESHOLD = 0.20
RETRIEVAL_HISTORY_TURNS = 2

NO_INFO_MESSAGE = (
    "I don't have information about that in the knowledge base I have access to. "
    "I can help with questions about our project brochures, pricing & payment plans, "
    "floor plans, amenities, location guides, RERA documentation, registration & "
    "possession process, home loans, and our policies (privacy, terms, cancellation "
    "& refund). Could you rephrase your question or ask about one of these topics?"
)


def _build_retrieval_query(question: str, history: list[dict]) -> str:
    recent_user_turns = [m["content"] for m in history if m["role"] == "user"][-RETRIEVAL_HISTORY_TURNS:]
    if not recent_user_turns:
        return question
    return " ".join(recent_user_turns) + " " + question


class RAGChain:
    def __init__(self, retriever: Retriever, top_k: int = 5):
        self.retriever = retriever
        self.top_k = top_k

    def answer(self, question: str, history: list[dict] | None = None) -> dict:
        history = history or []
        retrieval_query = _build_retrieval_query(question, history)
        results = self.retriever.query(retrieval_query, top_k=self.top_k)
        relevant = [r for r in results if r["relevance_score"] >= RELEVANCE_THRESHOLD]

        if not relevant:
            return {"answer": NO_INFO_MESSAGE, "sources": [], "grounded": False}

        context_blocks = [
            f"[Source: {r['source']} | Category: {r['category']}]\n{r['text']}" for r in relevant
        ]
        answer_text = generate_answer(question, context_blocks, history)

        best_per_doc: dict[str, dict] = {}
        for r in relevant:
            key = r["doc_id"]
            if key not in best_per_doc or r["relevance_score"] > best_per_doc[key]["relevance_score"]:
                best_per_doc[key] = r
        sources = sorted(best_per_doc.values(), key=lambda r: -r["relevance_score"])

        return {
            "answer": answer_text,
            "sources": [
                {
                    "source": s["source"],
                    "category": s["category"],
                    "relevance_score": s["relevance_score"],
                }
                for s in sources
            ],
            "grounded": True,
        }