"""Verify which Provence candidates provide new evidence for a Judge gap."""

import json
import logging


LOGGER = logging.getLogger(__name__)
SYSTEM_PROMPT = """You check proposed additions to the evidence for a multi-hop question.
Use only the supplied text. The candidate's source title and the current evidence may resolve pronouns, entity names or links between facts. The stated gaps are claims about what is missing, not established facts.

Keep a candidate if it provides a new fact that directly addresses a stated gap or establishes a missing link needed to resolve it. Verify that the fact applies to the right entity and relation. A similar name, a shared topic, or a fact already present in the evidence is insufficient. A candidate can help resolve a gap even when another fact is still needed. Do not answer the question.

Put the JSON array of candidate numbers to keep on the first line, for example [1, 3]. Return [] if none qualify."""

JOINT_SYSTEM_PROMPT = """You check proposed additions to the evidence for a multi-hop question.
Use only the supplied text. Use titles and current evidence to resolve references and distinguish entities. Treat the stated gaps as claims about what is missing, not facts. Do not answer the question.

Evaluate the candidates together with the current evidence. Trace how a candidate can identify an intermediate entity and another candidate can give the requested fact about that same entity. Keep every sentence needed for a supported chain, even if one sentence would not address the gap on its own. Prefer a chain that resolves the gap over one that only progresses toward it. If the gap cannot be resolved, keep facts that establish a missing link toward it. Exclude redundant facts and facts that merely resemble an entity or topic without connecting to the chain.

For example, if the gap asks when an author's first novel was released, one candidate may identify that novel and a second may give its date. Keep both if the date refers to the novel identified by the first.

Put the JSON array of candidate numbers to keep on the first line, for example [1, 3]. Return [] if none qualify."""

VERIFIER_PROMPTS = {"sentence": SYSTEM_PROMPT, "joint": JOINT_SYSTEM_PROMPT}


def _parse_selection(output, count):
    lines = output.strip().splitlines() if isinstance(output, str) else []
    if lines and lines[0].strip() in {"```", "```json"}:
        lines = lines[1:]
    try:
        ids = json.loads(lines[0]) if lines else None
    except ValueError:
        return None
    if not isinstance(ids, list) or any(type(i) is not int or not 1 <= i <= count for i in ids):
        return None
    return set(ids)


def verify_history_candidates(call_reasoner, requests, *, mode="sentence"):
    """Return decisions for (question, gap text, context, [(title, sentence), ...])."""
    tasks = []
    for question, gap_text, context, candidates in requests:
        numbered = "\n".join(
            f"{i}. [{title}] {sentence}"
            for i, (title, sentence) in enumerate(candidates, start=1)
        )
        tasks.append(
            f"Question: {question}\nMissing information: {gap_text}\n"
            f"Current evidence:\n{context}\nProposed additions:\n{numbered}"
        )

    outputs = call_reasoner(
        system_prefixes=VERIFIER_PROMPTS[mode],
        task_contents=tasks,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=max(16, 4 * max(len(request[3]) for request in requests)),
        do_sample=False,
    )
    decisions = [
        _parse_selection(output, len(request[3]))
        for request, output in zip(requests, outputs, strict=True)
    ]
    invalid = sum(decision is None for decision in decisions)
    if invalid:
        LOGGER.warning("History repair verifier returned %d invalid selections. Using the original candidates.", invalid)
    return decisions
