"""Map Provence output to source sentences and allocate the evidence budget."""

import re
import unicodedata
from collections import Counter
from itertools import groupby, zip_longest

from utils.history_repair_filter import verify_history_candidates


PROVENCE_MODEL = "naver/provence-reranker-debertav3-v1"
PROVENCE_REVISION = "ef49e233e3c6e50efc476c68f1390f8a63add4d4"


def load_provence(model_name, device):
    """Load the same checkpoint for every Provence selector variant."""
    import torch
    from transformers import AutoModel

    revision = PROVENCE_REVISION if model_name == PROVENCE_MODEL else None
    return AutoModel.from_pretrained(
        model_name, revision=revision, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device).eval()


def _comparable_words(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", text))


def _source_ids(sentences_by_doc, pruned_docs):
    """Match retained sentences in order within each document."""
    candidates = []
    documents = [(i, sentences) for i, sentences in enumerate(sentences_by_doc) if sentences]
    for (doc_id, sentences), pruned in zip(documents, pruned_docs, strict=True):
        remaining = " " + _comparable_words(pruned) + " "
        selected = []
        for sentence_id, sentence in enumerate(sentences, start=1):
            words = _comparable_words(sentence)
            match = remaining.find(f" {words} ") if words else -1
            if match >= 0:
                selected.append((doc_id, sentence_id))
                remaining = remaining[match + len(words) + 1:]
        candidates.append(selected)
    return [item for row in zip_longest(*candidates) for item in row if item is not None]


def _allocate_budget(ranked_groups, budget, reserve_per_fact):
    """Use the same global ranking for both variants, with optional fact reservations."""
    counts = Counter()
    best_rank = {}
    for ranked in ranked_groups:
        counts.update(ranked)
        for rank, sentence in enumerate(ranked):
            best_rank[sentence] = min(rank, best_rank.get(sentence, rank))
    pooled = sorted(counts, key=lambda sentence: (-counts[sentence], best_rank[sentence], sentence))

    selected = set()
    if reserve_per_fact:
        for ranked in ranked_groups:
            if len(selected) == budget:
                break
            if ranked and selected.isdisjoint(ranked):
                selected.add(ranked[0])
    for sentence in pooled:
        if len(selected) == budget:
            break
        selected.add(sentence)
    return [sentence for sentence in pooled if sentence in selected]


def select_provence_sentence_ids(
    provence, queries_batch, texts_batch, split_sentences, *,
    return_top_k=6, max_sents_per_doc=40, reserve_per_fact=False,
):
    """Prune the same source documents for each query and return per-document IDs."""
    sentences_batch = [
        [
            split_sentences(text)[:max_sents_per_doc]
            if text and text.strip() != "No results found." else []
            for text in texts
        ]
        for texts in texts_batch
    ]
    return _select_from_sentences(
        provence, queries_batch, sentences_batch, return_top_k, reserve_per_fact
    )


def _select_from_sentences(provence, queries_batch, sentences_batch, budget, reserve_per_fact=False):
    selected_batch = [[[] for _ in documents] for documents in sentences_batch]
    ranked_batch = _rank_from_sentences(
        provence, queries_batch, sentences_batch, budget, reserve_per_fact
    )
    for selected_docs, ranked in zip(selected_batch, ranked_batch, strict=True):
        for doc_id, sentence_id in sorted(ranked):
            selected_docs[doc_id].append(sentence_id)
    return selected_batch


def _rank_from_sentences(provence, queries_batch, sentences_batch, budget, reserve_per_fact=False):
    """Return ranked (document ID, sentence ID) selections before formatting by source."""
    questions, contexts, owners = [], [], []
    for index, (queries, documents) in enumerate(zip(queries_batch, sentences_batch, strict=True)):
        context = [" ".join(sentences) for sentences in documents if sentences]
        if context:
            for query in queries:
                questions.append(query)
                contexts.append(context)
                owners.append(index)
    if not questions:
        return [[] for _ in sentences_batch]

    pruned = provence.process(
        question=questions, context=contexts, title=None, reorder=False, batch_size=16
    )["pruned_context"]
    ranked_batch = [[] for _ in sentences_batch]
    for index, pruned_docs in zip(owners, pruned, strict=True):
        ranked_batch[index].append(_source_ids(sentences_batch[index], pruned_docs))
    return [_allocate_budget(groups, budget, reserve_per_fact) for groups in ranked_batch]


def _history_gap_text(gaps):
    phrases = []
    for gap in gaps[:3]:
        if not isinstance(gap, dict):
            continue
        phrase = gap.get("description") or " ".join(
            str(gap[field]) for field in ("target", "slot") if gap.get(field)
        )
        if words := " ".join(str(phrase).split()):
            phrases.append(words)
    return " ".join(phrases)


def _history_candidates(history, context, split_sentences):
    """Keep source sentences that have not already been exposed to the Judge."""
    seen = set()
    for block in context.split("\n\n---\n\n"):
        header, _, body = block.partition("\n")
        if header.startswith("[") and header.endswith("] EVIDENCE:"):
            title = _comparable_words(header[1:-len("] EVIDENCE:")])
            seen.update(
                (title, _comparable_words(sentence))
                for sentence in re.split(r"(?:^|\n)- ", body)
                if sentence
            )

    candidates = []
    for title, text in dict.fromkeys(history):
        if not text or text.strip() == "No results found.":
            continue
        title = title or "N/A"
        title_key = _comparable_words(title)
        sentences = []
        for sentence in split_sentences(text)[:40]:
            words = _comparable_words(sentence)
            key = (title_key, words)
            if words and key not in seen:
                seen.add(key)
                sentences.append(sentence)
        if sentences:
            candidates.append((title, sentences))
    return candidates


def recover_history_evidence(
    provence, questions, verdicts, gaps, histories, contexts, split_sentences, *, budget,
    candidate_budget=None, use_gap=True, verifier=None, verification_mode="sentence",
):
    """Return (batch index, query, evidence block) for successful historical lookups."""
    lookups = []
    for index, verdict in enumerate(verdicts):
        if verdict or not gaps[index] or not histories[index]:
            continue
        gap_text = _history_gap_text(gaps[index])
        if not gap_text:
            continue
        candidates = _history_candidates(histories[index], contexts[index], split_sentences)
        if candidates:
            question = " ".join(questions[index].split())
            query = f"{question} Missing information: {gap_text}"
            lookups.append((index, query, gap_text, candidates))

    if not lookups:
        return []
    ranked_batch = _rank_from_sentences(
        provence,
        [[query if use_gap else questions[index]] for index, query, _, _ in lookups],
        [[sentences for _, sentences in candidates] for _, _, _, candidates in lookups],
        budget if candidate_budget is None else candidate_budget,
    )
    proposals = [(lookup, ranked) for lookup, ranked in zip(lookups, ranked_batch, strict=True) if ranked]

    decisions = [None] * len(proposals)
    if verifier is not None and proposals:
        decisions = verify_history_candidates(verifier, [
            (questions[index], gap_text, contexts[index], _verifier_candidates(candidates, ranked))
            for (index, _, gap_text, candidates), ranked in proposals
        ], mode=verification_mode)

    recovered = []
    for ((index, query, _, candidates), ranked), decision in zip(proposals, decisions, strict=True):
        accepted = _accepted_history_ids(ranked, decision, budget)
        evidence = []
        for doc_id, locations in groupby(accepted, key=lambda item: item[0]):
            title, sentences = candidates[doc_id]
            evidence.append(f"[{title}] EVIDENCE:\n- " + "\n- ".join(
                sentences[sentence_id - 1] for _, sentence_id in locations
            ))
        if evidence:
            recovered.append((index, query, "\n\n---\n\n".join(evidence)))
    return recovered


def _verifier_candidates(documents, ranked):
    return [
        (documents[doc_id][0], documents[doc_id][1][sentence_id - 1])
        for doc_id, sentence_id in sorted(ranked)
    ]


def _accepted_history_ids(ranked, decision, budget):
    """Apply the final cap in Provence order and format the chosen IDs in source order."""
    if decision is None:
        return sorted(ranked[:budget])
    accepted = {location for number, location in enumerate(sorted(ranked), 1) if number in decision}
    return sorted([location for location in ranked if location in accepted][:budget])
