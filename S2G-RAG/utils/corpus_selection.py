import os


SUPPORTED_DATASETS = ("hotpotqa", "triviaqa", "2wikimultihopqa")

BM25_CORPUS_FILES = {
    "hotpotqa": (
        "bm25_search/corpus/corpus.pkl",
        "bm25_search/corpus/retriever_settings.pkl",
    ),
    "triviaqa": (
        "bm25_search/corpus/triviaqa_corpus.pkl",
        "bm25_search/corpus/triviaqa_retriever_settings.pkl",
    ),
    "2wikimultihopqa": (
        "bm25_search/corpus/2wiki_corpus.pkl",
        "bm25_search/corpus/2wiki_retriever_settings.pkl",
    ),
}

DATASET_RETRIEVAL_PROTOCOL = {
    "hotpotqa": "ircot_hotpotqa_fullwiki",
    "triviaqa": "triviaqa_entity_pages_search_results_union",
    "2wikimultihopqa": "ircot_2wikimultihopqa_context_union",
}


def infer_dataset_name(dataset_name=None, input_path=None, experiment_name=None):
    """
    Resolve the dataset name from an explicit argument or a known filename pattern.
    显式传入参数、文件路径或者实验名称自动识别数据集名称
    Args:
        dataset_name: The name of the dataset to resolve.
        input_path: The path to the input file.
        experiment_name: The name of the experiment.
    Returns:
        return: 标准化后的数据集名字
    """
    if dataset_name:
        normalized = dataset_name.strip().lower()
        if normalized in SUPPORTED_DATASETS:
            return normalized
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Expected one of: {', '.join(SUPPORTED_DATASETS)}"
        )

    haystacks = [input_path or "", experiment_name or ""]
    for candidate in SUPPORTED_DATASETS:
        for haystack in haystacks:
            if candidate in os.path.basename(str(haystack)).lower():
                return candidate

    if "2wiki" in " ".join(haystacks).lower():
        return "2wikimultihopqa"

    raise ValueError(
        "Could not infer dataset_name from input_path/experiment_name. "
        "Please pass --dataset_name explicitly."
    )


def get_bm25_resource_paths(dataset_name):
    """Return the corpus/settings files for the requested dataset."""
    resolved = infer_dataset_name(dataset_name=dataset_name)
    return BM25_CORPUS_FILES[resolved]


def get_retrieval_cache_tag(dataset_name):
    """Return a stable cache tag for datasets that share the same retrieval corpus."""
    resolved = infer_dataset_name(dataset_name=dataset_name)
    return DATASET_RETRIEVAL_PROTOCOL[resolved]
