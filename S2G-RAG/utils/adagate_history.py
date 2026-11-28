"""AdaGATE utility and adaptive capacity for historical evidence sentences."""

import re


def _terms(text):
    return {word for word in re.findall(r"[^\W_]+", text.casefold()) if len(word) > 3 or word.isdigit()}


def rank_history_sentences(question, gap, context, documents, candidate_budget):
    """Use selected evidence as a proxy for AdaGATE's entity ledger."""
    question_terms = _terms(question)
    gap_terms = _terms(gap)
    evidence_terms = _terms(context)
    scored = []
    for doc_id, (title, sentences) in enumerate(documents):
        title_terms = _terms(title)
        for sentence_id, sentence in enumerate(sentences, 1):
            terms = _terms(sentence)
            if not terms:
                continue
            source_terms = terms | title_terms
            gap_coverage = len(gap_terms & source_terms) / max(len(gap_terms), 1)
            relevance = len(question_terms & source_terms) / max(len(question_terms), 1)
            if not (gap_coverage or relevance):
                continue
            novelty = len(terms - evidence_terms) / len(terms)
            redundancy = len(terms & evidence_terms) / len(terms)
            utility = 0.35 * gap_coverage + 0.15 * novelty - 0.15 * redundancy + 0.15 * relevance
            scored.append((utility, (doc_id, sentence_id)))

    scored.sort(key=lambda item: item[0], reverse=True)
    drops = [left[0] - right[0] for left, right in zip(scored, scored[1:])]
    largest_drop = max(drops, default=0)
    if largest_drop > 0:
        # Keep the group above the steepest drop, plus AdaGATE's two-item buffer.
        scored = scored[:drops.index(largest_drop) + 3]
    return [location for _, location in scored[:candidate_budget]]
