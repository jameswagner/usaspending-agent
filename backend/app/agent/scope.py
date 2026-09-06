"""The cheap in-scope pre-filter gate ask() runs before starting the (much
more expensive) tool-calling loop.
"""
from __future__ import annotations

from langsmith import traceable

from .singletons import MODEL, _get_client, _get_retriever

SCOPE_CLASSIFIER_PROMPT = (
    "You classify whether a user question is in scope for a USASpending.gov "
    "assistant: federal spending, budgets, obligations/outlays, contracts, "
    "grants/financial assistance, awards, recipients, federal agencies, or "
    "USASpending.gov data/fields/API concepts. You are given the single "
    "most relevant passage a retrieval system found for this question, "
    "which may or may not actually be relevant — a weak or unrelated "
    "passage does NOT mean the question is out of scope, since many "
    "in-scope questions (e.g. live spending-data lookups) have no good "
    "match in this retrieval corpus at all. Use the passage only as "
    "supporting evidence when it looks genuinely on-topic; ignore it if it "
    "looks irrelevant. Respond with only YES or NO, nothing else."
)


@traceable(run_type="retriever", name="scope_classifier_context")
def _get_top_passage(question: str) -> str:
    results = _get_retriever().retrieve(question, top_k=1)
    if not results:
        return "No relevant passage was found for this question."
    top = results[0]
    return f"Most relevant passage found (rerank score {top['rerank_score']:.2f}):\n{top['text']}"


@traceable(run_type="llm", name="scope_classifier")
def _is_in_scope(question: str) -> bool:
    """Cheap pre-filter gate: only start the (much more expensive) tool-
    calling loop if the question is plausibly in-scope for this app's whole
    domain, instead of relying on the system prompt alone to stop the model
    from answering off-topic questions from its own knowledge.

    Deliberately broader than "the guide has relevant content" — a
    live-data question (e.g. an agency lookup) can be legitimately in scope
    without matching anything in the static guide; the prompt explicitly
    tells the classifier not to penalize a weak/irrelevant retrieved
    passage, for exactly this reason.

    Runs retrieval before the classifier call: measured 97.2% accuracy this
    way vs. 89.9% for the bare-question baseline, on a 71-question
    hand-reviewed labeled set (agent/dev_tools/calibrate_scope_classifier.py,
    see BACKLOG.md for the full comparison, including why majority-vote
    alone doesn't help). If the tool loop later also calls search_guide for
    the same question, retrieval runs again - deliberately not engineered
    around, since it's local (no LLM cost) and cheap over this corpus's
    size.

    No temperature control is available on this model - sampling
    parameters are deprecated/rejected outright by the current API, with
    no direct replacement - so this call is inherently non-deterministic
    regardless of the above.
    """
    context = _get_top_passage(question)
    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=5,
        system=SCOPE_CLASSIFIER_PROMPT,
        messages=[{"role": "user", "content": f"Question: {question}\n\n{context}"}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return text.strip().upper().startswith("YES")
