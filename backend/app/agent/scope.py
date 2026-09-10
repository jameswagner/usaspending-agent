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
# folds in two YES conditions instead of one: a genuine continuation of
# the prior turn, OR a brand new in-domain question that has nothing to
# do with the prior turn's specific subject (e.g. a different agency
# entirely) - both are in scope, only a real domain pivot isn't. An
# earlier version only had the continuation condition and, confirmed
# live, incorrectly rejected every new-but-in-domain question asked
# right after an in-scope one (e.g. asking about the Department of
# Education immediately after a NASA question) - it had conflated "not
# connected to the prior turn" with "off domain," which are not the same
# thing. No retrieved passage here - unlike the bare-question path, a
# full-sentence in-domain question doesn't need retrieval as a crutch to
# be recognized as such.
FOLLOWUP_SCOPE_CLASSIFIER_PROMPT = (
    "You classify whether a user's question is in scope for a "
    "USASpending.gov assistant: federal spending, budgets, "
    "obligations/outlays, contracts, grants/financial assistance, awards, "
    "recipients, federal agencies, or USASpending.gov data/fields/API "
    "concepts. You are also given the immediately preceding turn(s) of "
    "the conversation, which were already confirmed in scope.\n\n"
    "Answer YES if EITHER is true:\n"
    "1. The question is itself about this domain, even if it concerns a "
    "completely different agency, recipient, or topic than the prior turn "
    "(e.g. asking about a different agency's spending right after a "
    "question about NASA is still in scope - a new in-domain question, "
    "not a domain pivot).\n"
    "2. The question is a short reaction, clarification, or comparison "
    "that only makes sense in light of the prior turn (e.g. 'was that a "
    "lot?', 'why?', 'is that normal?', 'what about last year?') - judge "
    "these as a continuation of an already in-scope conversation, not "
    "against their own words alone.\n\n"
    "Answer NO only if the question is clearly about something else "
    "entirely, unconnected to federal spending/budgets/awards/agencies in "
    "any way (e.g. the weather, a recipe, general trivia, a coding "
    "question). Respond with only YES or NO, nothing else."
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
    exchange, this switches entirely to FOLLOWUP_SCOPE_CLASSIFIER_PROMPT -
    see that constant's own comment for why it needs two YES conditions
    (continuation, or an independently new in-domain question), not just
    one. Went through two live-tested iterations before landing here:
    (1) folding history into the original from-scratch prompt failed a
    fully generic follow-up with no topical token ("Was that a lot?"
    classified NO, 3/3, even with correct history present); (2) a
    continuation-only rewrite fixed that but then incorrectly rejected
    every new-but-unrelated-entity in-domain question ("How much did the
    Department of Education spend on Pell Grants?" right after a NASA
    question classified NO, 5/5). All of the following are confirmed
    live against the final version, repeated 3x each: "Was that a lot?"
    (continuation, no topic token) -> YES; "What about FY2023?"
    (continuation, weak anchor) -> YES; a new Department of
    Education/Boeing question (unrelated entity, still in-domain) -> YES;
    "capital of France?" / a recipe question (genuine domain pivot) ->
    NO. Ships without calibration data either way; a future
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
