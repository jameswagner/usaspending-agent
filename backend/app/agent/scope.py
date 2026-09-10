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

# Genuinely different task from SCOPE_CLASSIFIER_PROMPT above, not a
# variant of it - the prior turn was ALREADY confirmed in scope, so this
# defaults to YES (continuation) and only says NO for a clear topic
# pivot, rather than re-litigating scope from a blank slate on every
# follow-up. No retrieved passage here - "does this match the corpus"
# isn't the question for a follow-up; "did the conversation change
# subject" is.
FOLLOWUP_SCOPE_CLASSIFIER_PROMPT = (
    "You are given the most recent turn(s) of a conversation that was "
    "ALREADY confirmed in scope for a USASpending.gov federal spending "
    "assistant, plus a new follow-up question. Classify whether the "
    "follow-up continues that conversation or is a complete pivot to an "
    "unrelated topic. Default to YES (still in scope): a terse reaction, "
    "clarification, or comparison that only makes sense in light of what "
    "was just discussed (e.g. 'was that a lot?', 'why?', 'is that "
    "normal?', 'what about last year?') is a continuation even though it "
    "doesn't restate the subject itself. Answer NO only if the follow-up "
    "is clearly, entirely unrelated to the ongoing conversation - a real "
    "subject change (e.g. asking about the weather, a recipe, or the "
    "capital of a country), not just a short or vague phrasing of a "
    "continuation. Respond with only YES or NO, nothing else."
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
    message list (see _ask_langgraph). When there's a renderable prior
    exchange, this switches entirely to FOLLOWUP_SCOPE_CLASSIFIER_PROMPT:
    the prior turn already passed this same gate, so the question isn't
    "does this match the corpus" (no retrieval happens for this branch at
    all) but "did the conversation pivot away" - defaulting to YES and
    only rejecting a clear subject change. This is a real, deliberate
    fix, not a tweak: an earlier version tried folding history into the
    *original* from-scratch prompt (still asking "is this in scope" cold)
    and confirmed live it failed a fully generic follow-up with no
    topical token ("Was that a lot?" after a budget question classified
    NO, 3/3 repeats, even with the correct prior exchange present) - the
    problem was the framing, not the missing context. With the
    continuation-by-default framing, the same case classifies YES, and a
    genuine pivot ("what's the capital of France?" as a follow-up) still
    correctly classifies NO - both confirmed live, not assumed. Ships
    without calibration data either way; a future
    calibrate_scope_classifier.py-style pass extending its labeled set
    with real multi-turn examples would give this a measured accuracy
    figure the way the bare-question path already has.
    """
    if recent_messages:
        history_block = _render_recent_exchanges(recent_messages)
        if history_block:
            response = _get_client().messages.create(
                model=MODEL,
                max_tokens=5,
                system=FOLLOWUP_SCOPE_CLASSIFIER_PROMPT,
                messages=[{"role": "user", "content": f"{history_block}\n\nNew question: {question}"}],
            )
            text = next((b.text for b in response.content if b.type == "text"), "")
            return text.strip().upper().startswith("YES")

    context = _get_top_passage(question)
    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=5,
        system=SCOPE_CLASSIFIER_PROMPT,
        messages=[{"role": "user", "content": f"Question: {question}\n\n{context}"}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return text.strip().upper().startswith("YES")
