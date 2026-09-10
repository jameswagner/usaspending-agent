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
    "looks irrelevant. You may also be given the immediately preceding "
    "conversation turn(s) - use them only to understand what a follow-up "
    "question that doesn't restate its subject is actually asking about "
    "(e.g. 'what about NASA?' after a budget question is a budget "
    "question about NASA); still classify based on the CURRENT question's "
    "own topic, not the prior turns' topic alone. Respond with only YES "
    "or NO, nothing else."
)


@traceable(run_type="retriever", name="scope_classifier_context")
def _get_top_passage(question: str) -> str:
    results = _get_retriever().retrieve(question, top_k=1)
    if not results:
        return "No relevant passage was found for this question."
    top = results[0]
    return f"Most relevant passage found (rerank score {top['rerank_score']:.2f}):\n{top['text']}"


def _render_recent_exchanges(recent_messages: list, max_exchanges: int = 2) -> str:
    """Render up to the last few human-question/final-answer exchanges
    from a LangGraph conversation's message list, for folding into the
    scope classifier's context. A single human turn can produce more than
    one AIMessage (an intermediate tool-calling one, content a list of
    content blocks, then a final plain-text one) plus ToolMessages in
    between - only the question and the final plain-text answer per turn
    are useful context here, so everything else is skipped."""
    exchanges: list[tuple[str, str]] = []
    pending_question: str | None = None
    pending_answer: str | None = None
    for message in recent_messages:
        message_type = getattr(message, "type", None)
        if message_type == "human":
            if pending_question is not None:
                exchanges.append((pending_question, pending_answer or ""))
            pending_question = str(message.content)
            pending_answer = None
        elif message_type == "ai" and isinstance(message.content, str) and message.content:
            pending_answer = message.content
    if pending_question is not None:
        exchanges.append((pending_question, pending_answer or ""))

    return "\n".join(f"Prior turn: Q: {q}\nA: {a}" for q, a in exchanges[-max_exchanges:])


@traceable(run_type="llm", name="scope_classifier")
def _is_in_scope(question: str, recent_messages: list | None = None) -> bool:
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

    recent_messages (issue #62) is an optional LangGraph conversation
    message list (see _ask_langgraph) - when given, the last couple of
    question/answer exchanges are rendered and folded into this same
    call's context. Confirmed live this correctly resolves a follow-up
    that keeps a topical token ("What about FY2023?" after a budget
    question - still classifies YES, still calls the right tool for the
    right year). Confirmed live it does NOT reliably resolve a fully
    generic follow-up with no topical token at all (e.g. "Was that a
    lot?" after the same budget question classified NO, 3/3 repeats,
    even with the prior exchange correctly rendered into its context) -
    a real, demonstrated limit of folding raw history into one classifier
    call, not a wiring bug. Ships without calibration data, unlike the
    bare-question figure above; a future calibrate_scope_classifier.py-
    style pass extending its labeled set with real multi-turn follow-ups
    - or a query-rewriting step, deliberately not built here - would be
    the next move if this limit matters in practice.
    """
    context = _get_top_passage(question)
    user_content = f"Question: {question}\n\n{context}"
    if recent_messages:
        history_block = _render_recent_exchanges(recent_messages)
        if history_block:
            user_content = f"{history_block}\n\n{user_content}"
    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=5,
        system=SCOPE_CLASSIFIER_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return text.strip().upper().startswith("YES")
