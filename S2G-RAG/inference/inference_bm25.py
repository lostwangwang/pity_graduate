# -*- coding: utf-8 -*-
import argparse
import json
import os
import pickle
import re
import sys
import unicodedata
from itertools import zip_longest

import pandas as pd
import torch
from accelerate import Accelerator
from openai import OpenAI
from pyserini.search.lucene import LuceneSearcher as SimpleSearcher

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.evaluation import exact_match_multiple, normalize_answer
from utils.prompt_template import force_answer_prompt, TRIVIAQA_SELECTOR_SYSTEM_PROMPT, TRIVIAQA_FORCE_ANSWER_PROMPT, SELECTOR_SYSTEM_PROMPT, SUFF_SYSTEM_PROMPT
from utils.text_processing import (
    extract_evidence_global_ids,
    extract_final_answer_and_rationale,
    extract_gap_items,
    safe_literal_eval,
)
from utils.corpus_selection import get_bm25_resource_paths, infer_dataset_name
from utils.provence_selector import PROVENCE_MODEL, load_provence, recover_history_evidence

try:
    import pysbd

    _PYSBD_OK = True
    _PYSBD_SEGMENTER = pysbd.Segmenter(language="en", clean=False)
except Exception:
    pysbd = None
    _PYSBD_OK = False
    _PYSBD_SEGMENTER = None

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_JSON_RE = re.compile(r"\{.*\}", re.S)
CSV_COLUMNS = [
    "ID",
    "Turn",
    "Reasoner Task Content",
    "Evidence Context So Far",
    "Raw Retrieved Docs",
    "Gate Output",
    "SuffMissingFacts",
    "Reasoner Answer",
    "Gold Answer",
    "Correct Answer",
    "Gold Retrieved Docs",
    "Retrieved Titles So Far",
    "Correct Retrieval",
]

accelerator = None
tokenizer = None
model = None
gate_tokenizer = None
gate_model = None
searcher = None
corpus = None
MODEL = None


def get_selector_system_prompt(dataset_name):
    """Choose a selector prompt that matches the dataset's evidence style."""
    if str(dataset_name or "").strip().lower() == "triviaqa":
        return TRIVIAQA_SELECTOR_SYSTEM_PROMPT
    return SELECTOR_SYSTEM_PROMPT


def get_answer_system_prompt(dataset_name):
    """Choose an answer-generation prompt that matches the dataset's answer style."""
    if str(dataset_name or "").strip().lower() == "triviaqa":
        return TRIVIAQA_FORCE_ANSWER_PROMPT
    return force_answer_prompt


def safe_json_load(text: str):
    """Parse JSON robustly, including fenced code blocks."""
    if not text:
        return None

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        match = _JSON_RE.search(text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


def split_wiki_sentences(text: str):
    """Split Wikipedia-like text into sentences."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n+", text)
    sentences = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        cjk_count = len(_CJK_RE.findall(paragraph))
        if cjk_count > 0 and (cjk_count / max(1, len(paragraph))) > 0.02:
            parts = re.split(r"(?<=[。！？!?])", paragraph)
            sentences.extend(x.strip() for x in parts if x and x.strip())
            continue

        if _PYSBD_OK:
            parts = _PYSBD_SEGMENTER.segment(paragraph)
            sentences.extend(x.strip() for x in parts if x and x.strip())
        else:
            parts = re.split(r"(?<=[.!?])\s+|[\n]+", paragraph)
            sentences.extend(x.strip() for x in parts if x and x.strip())

    return sentences


def concat_raw_retrieved_docs(titles, texts):
    """Concatenate raw retrieved documents before sentence selection."""
    blocks = []
    for title, raw_text in zip(titles or [], texts or []):
        if not raw_text:
            continue
        blocks.append(f"[{title or 'N/A'}]\n{raw_text.strip()}")
    return "\n\n---\n\n".join(blocks)


def call_reasoner_batch(
    system_prefixes,
    task_contents,
    gpt=False,
    temperature=0.6,
    top_p=0.9,
    max_new_tokens=256,
    do_sample=True,
):
    """Run the main reasoner model and return decoded outputs."""
    if isinstance(system_prefixes, str):
        messages = [
            [
                {"role": "system", "content": system_prefixes},
                {"role": "user", "content": task_content},
            ]
            for task_content in task_contents
        ]
    else:
        messages = [
            [
                {"role": "system", "content": system_prefix},
                {"role": "user", "content": task_content},
            ]
            for task_content, system_prefix in zip(task_contents, system_prefixes)
        ]

    if gpt:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Please set OPENAI_API_KEY.")

        client = OpenAI(
            base_url="https://api.openai.com/v1",
            api_key=api_key,
        )

        outputs = []
        for message in messages:
            response = client.chat.completions.create(
                model=MODEL,
                messages=message,
                temperature=temperature,
                max_tokens=max_new_tokens,
                stop=None,
            )
            outputs.append(response.choices[0].message.content)
        return outputs

    prompts = [
        tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        for msgs in messages
    ]
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(accelerator.device)

    with torch.no_grad():
        generated = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = generated[:, prompt_len:]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return [text.strip() for text in decoded]


def bm25_search_batch(query_texts, all_past_doc_ids, k=2, remove_repeat_docs=True):
    """Run BM25 search with optional exact-doc de-duplication."""
    past_doc_id_sets = [set(x) for x in all_past_doc_ids] if remove_repeat_docs else [set() for _ in all_past_doc_ids]

    titles_batch = []
    texts_batch = []
    doc_ids_batch = []

    for query_index, query_text in enumerate(query_texts):
        hits = searcher.search(str(query_text or "").strip(), k=max(k, 50))
        filtered_doc_ids = []
        for hit in hits:
            doc_id = hit.docid
            if doc_id not in corpus:
                continue
            if remove_repeat_docs and doc_id in past_doc_id_sets[query_index]:
                continue
            filtered_doc_ids.append(doc_id)
            if len(filtered_doc_ids) >= k:
                break

        top_titles = []
        top_texts = []
        top_doc_ids = []
        for doc_id in filtered_doc_ids:
            top_doc_ids.append(doc_id)
            top_titles.append(corpus[doc_id].get("title", "No results found."))
            top_texts.append(corpus[doc_id].get("text", "No results found."))

        if not top_titles:
            top_doc_ids = [""]
            top_titles = ["No results found."]
            top_texts = ["No results found."]

        titles_batch.append(top_titles)
        texts_batch.append(top_texts)
        doc_ids_batch.append(top_doc_ids)

    return titles_batch, texts_batch, doc_ids_batch


def build_suff_user_prompt(question: str, context: str) -> str:
    return f"""QUESTION:
{question}

CONTEXT:
{context}

Please ONLY use the above CONTEXT as evidence.
Decide whether it is sufficient, and if not, list the missing facts
in the required JSON format."""


def call_suff_gate_batch(questions, evidence_contexts, gate_model_obj, gate_tokenizer_obj, max_new_tokens=256):
    """批量调用本地 S2G-Judge，判断当前累计证据是否足以回答问题。

    参数：
        questions：原始问题列表。
        evidence_contexts：与问题一一对应的累计证据上下文 C_t。
        gate_model_obj：执行生成的 S2G-Judge 模型。
        gate_tokenizer_obj：模型配套的分词器，用于编码输入和解码输出。
        max_new_tokens：每条样本最多生成的 token 数。

    返回两个按输入顺序排列的列表：
        verdicts：每条样本的证据充足性判定 s_t（布尔值）。
        missing_facts_batch：每条样本的结构化缺失事实 G_t，供下一轮检索构造查询。
    """
    messages_batch = []
    # 1. 按顺序配对问题和证据。系统消息规定判定规则和 JSON 格式，用户消息提供本条样本。
    for question, context in zip(questions, evidence_contexts):
        messages_batch.append(
            [
                {"role": "system", "content": SUFF_SYSTEM_PROMPT},
                {"role": "user", "content": build_suff_user_prompt(question, context)},
            ]
        )

    # 2. 按模型的聊天模板拼接提示文本，并在末尾标记助手回复的起点。
    prompts = [
        gate_tokenizer_obj.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        for msgs in messages_batch
    ]

    # 3. 将提示文本编码为 token ID，填充至批内同一长度，并按分词器配置截断长输入。转为 PyTorch 张量后移到推理设备，attention_mask 标记哪些位置是填充。
    inputs = gate_tokenizer_obj(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(accelerator.device)

    # 4. 本地模型逐 token 生成回复。关闭梯度计算以减少显存开销，并关闭随机采样。
    with torch.no_grad():
        outputs = gate_model_obj.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # 5. 输出序列以原始输入开头，按填充后的输入宽度切掉前缀，只解码新生成的回复。
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, prompt_len:]
    texts = gate_tokenizer_obj.batch_decode(generated_ids, skip_special_tokens=True)

    verdicts = []
    missing_facts_batch = []

    # 6. 将模型生成的文本逐条解析为 JSON，再提取判定结果和缺失事实。
    for assistant_text in texts:
        parsed = safe_json_load((assistant_text or "").strip())
        # 解析失败或结果不是 JSON 对象时，按证据不足处理，并保留该样本的输出位置。
        if not isinstance(parsed, dict):
            verdicts.append(False)
            missing_facts_batch.append([])
            continue

        # 缺少 sufficient 字段时默认判为证据不足。缺失事实兼容 gap_items 等字段名。
        sufficient = bool(parsed.get("sufficient", False))
        missing_facts = extract_gap_items(parsed)
        # 证据已充足，或缺失事实不是列表时，统一返回空列表。
        if sufficient or not isinstance(missing_facts, list):
            missing_facts = []

        verdicts.append(sufficient)
        missing_facts_batch.append(missing_facts)

    return verdicts, missing_facts_batch


def build_query_from_missing(question, missing_facts, max_facts=None, dataset_name=None):
    """
    Build a retrieval query from missing facts.
    根据Gate识别出来的缺失事实，构造下一轮检索用的查询子句
    逻辑: 把原始问题 + 缺失事实拼接成 BM25 检索query
    参数:
        question: 原始问题
        missing_facts: Sufficiency Gate识别出来的缺失事实列表, 每个元素是dict
        max_facts: 最大拼接事实数量，默认3(TriviaQA用1)
        dataset_name: 数据集名字，用于决定max_facts
    """
    # 设置默认max_facts: TriviaQA只取一条缺失事实； 其它数据集默认最多取3条
    if max_facts is None:
        max_facts = 1 if str(dataset_name or "").strip().lower() == "triviaqa" else 3

    # 如果gate没有识别出缺失事实，直接返回原问题作为检索query
    if not missing_facts:
        return question

    # 用来收集事实短语
    phrases = []
    # 只取前max_facts条缺失事实
    for item in missing_facts[:max_facts]:
        # 异常保护: 不是字典就跳过
        if not isinstance(item, dict):
            continue

        # 从缺失事实dict里取出三个字段: target(实体)、slot(属性)、description(描述)
        target = (item.get("target") or "").strip()
        slot = (item.get("slot") or "").strip()
        description = (item.get("description") or "").strip()

        # 优先使用target + slot组合(结构化事实)
        if target and slot:
            phrases.append(f"{target} {slot}")
        # 如果没有target+slot, 就用自然语言描述
        elif description:
            phrases.append(description)
    # 如果收集到了事实短语: 原问题 + 这些短语拼接；否则返回原问题
    return question if not phrases else question + " " + " ".join(phrases)


def should_force_first_retrieval(turn, evidence_context, verdict):
    """
    强制首轮检索: 解决一个边界情况——第一轮证据为空，但gate模型误判证据充足
    :param turn: 当前是第几轮
    :param evidence_context: 目前累积拿到的证据上下文
    :param verdict: gate的充足性
    """
    return turn == 0 and bool(verdict) and not str(evidence_context or "").strip()


def format_missing_facts_for_selector(missing_facts, max_facts=3) -> str:
    """Format missing-fact hints for the selector prompt."""
    if not missing_facts:
        return "None."

    lines = []
    for item in missing_facts[:max_facts]:
        if not isinstance(item, dict):
            continue

        category = (item.get("category") or "").strip()
        target = (item.get("target") or "").strip()
        slot = (item.get("slot") or "").strip()
        description = (item.get("description") or "").strip()

        core = f"{target} {slot}".strip() if target and slot else description
        if core:
            lines.append(f"- [{category or 'other'}] {core}")

    return "\n".join(lines) if lines else "None."


def concat_and_pick_sentences_batch(
    questions,
    titles_batch,
    texts_batch,
    missing_facts_batch=None,
    dataset_name=None,
    gpt=False,
    return_top_k=6,
    max_sents_per_doc=40,
):
    """Select sentence-level evidence from retrieved documents."""
    tasks = []
    index_maps = []

    if missing_facts_batch is None:
        missing_facts_batch = [[] for _ in questions]

    for i, (titles, texts) in enumerate(zip(titles_batch, texts_batch)):
        question = questions[i]
        missing_facts = missing_facts_batch[i]

        sentences = []
        for doc_idx, (title, raw_text) in enumerate(zip(titles, texts)):
            if not raw_text or raw_text.strip() == "No results found.":
                continue

            parts = split_wiki_sentences(raw_text)
            if max_sents_per_doc is not None:
                parts = parts[:max_sents_per_doc]

            for local_sid, sentence_text in enumerate(parts, start=1):
                sentences.append((doc_idx, local_sid, title or "N/A", sentence_text))

        if not sentences:
            tasks.append(None)
            index_maps.append({})
            continue

        numbered_lines = []
        id_map = {}
        for global_id, (doc_idx, local_sid, title, sentence_text) in enumerate(sentences, start=1):
            numbered_lines.append(f"[{global_id}] ({title} | s#{local_sid}) {sentence_text}")
            id_map[global_id] = (doc_idx, local_sid)

        hints = format_missing_facts_for_selector(missing_facts, max_facts=3)
        user_prompt = (
            f"ORIGINAL QUESTION:\n{question}\n\n"
            f"MISSING FACTS TO FILL:\n{hints}\n\n"
            f"NUMBERED SENTENCES FROM RETRIEVED DOCUMENTS:\n"
            + "\n".join(numbered_lines)
            + "\n\n"
            + f"You may select up to {return_top_k} sentences.\n"
            + 'Return ONLY JSON with "evidence_global_ids".'
        )

        tasks.append(user_prompt)
        index_maps.append(id_map)

    tasks_to_call = [task for task in tasks if task is not None]
    outputs = []
    if tasks_to_call:
        outputs = call_reasoner_batch(
            system_prefixes=get_selector_system_prompt(dataset_name),
            task_contents=tasks_to_call,
            gpt=gpt,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=64,
            do_sample=False,
        )

    picked_ids_batch = []
    output_ptr = 0

    for i, titles in enumerate(titles_batch):
        if tasks[i] is None:
            picked_ids_batch.append([[] for _ in titles])
            continue

        raw_output = (outputs[output_ptr] or "").strip()
        output_ptr += 1

        parsed = safe_json_load(raw_output)
        if not isinstance(parsed, dict):
            parsed = {}

        global_ids = []
        values = extract_evidence_global_ids(parsed)
        if isinstance(values, list):
            for value in values:
                try:
                    value = int(value)
                    if value >= 1:
                        global_ids.append(value)
                except Exception:
                    pass

        per_doc = [[] for _ in titles]
        for global_id in global_ids:
            if global_id not in index_maps[i]:
                continue
            doc_idx, local_sid = index_maps[i][global_id]
            if 0 <= doc_idx < len(per_doc):
                per_doc[doc_idx].append(int(local_sid))

        picked_ids_batch.append([sorted(set(x)) for x in per_doc])

    return picked_ids_batch


def merge_evidence_only(titles, texts, evidence_ids_per_doc):
    """
    Keep only selected evidence sentences in a merged block.
    只保留筛选出来的证据句子，组装成一段格式化的证据文本块
    输入是一批文档，每个文档附带选中的句子下标；输出是拼接好、带标题的证据字符串
    参数：
        titles: 文档标题列表
        texts: 对应文档的原始正文文本列表
        evidence_ids_per_doc: 每个文档里被选中的证据句子编号列表（1-based）
    """
    blocks = [] # 用来存放每个文档组装后的证据块

    # 遍历每一篇文档: 标题、原文、该文档选中的句子编号
    for title, raw_text, sentence_ids in zip(titles, texts, evidence_ids_per_doc):
        # 原文为空 / 没有选中任何句子，直接跳过这篇文档
        if not raw_text or not sentence_ids:
            continue

        # 把维基文档原文切分成句子列表
        sentences = split_wiki_sentences(raw_text)
        chosen = []
        for index in sentence_ids:
            # 编号从1开始，做越界检查
            if 1 <= index <= len(sentences):
                chosen.append(sentences[index - 1]) # 转成Python 0-based下标取出句子

        # 如果这篇文档中选中句子，格式化
        if chosen:
            blocks.append(f"[{title or 'N/A'}] EVIDENCE:\n- " + "\n- ".join(chosen))

    # 多个文档的证据块
    return "\n\n---\n\n".join(blocks)


def select_provence_sentence_ids(
    provence, questions, texts_batch, return_top_k=6, max_sents_per_doc=40
):
    """Map Provence's pruned text back to source sentence IDs."""
    def comparable_words(text):
        text = unicodedata.normalize("NFKC", text).casefold()
        return " ".join(re.findall(r"\w+", text))

    sentences_batch = [
        [
            split_wiki_sentences(text)[:max_sents_per_doc]
            if text and text.strip() != "No results found." else []
            for text in texts
        ]
        for texts in texts_batch
    ]
    selected_batch = [[[] for _ in documents] for documents in sentences_batch]
    active = [index for index, documents in enumerate(sentences_batch) if any(documents)]
    if not active:
        return selected_batch

    contexts = [
        [" ".join(sentences) for sentences in sentences_batch[index] if sentences]
        for index in active
    ]
    pruned_batch = provence.process(
        question=[questions[index] for index in active],
        context=contexts,
        title=None,
        reorder=False,
        batch_size=16,
    )["pruned_context"]

    for index, pruned_documents in zip(active, pruned_batch, strict=True):
        candidates = []
        pruned_documents = iter(pruned_documents)
        for doc_index, sentences in enumerate(sentences_batch[index]):
            if not sentences:
                continue
            remaining = " " + comparable_words(next(pruned_documents)) + " "
            selected = []
            for sentence_id, original in enumerate(sentences, start=1):
                words = comparable_words(original)
                match = remaining.find(f" {words} ") if words else -1
                if match >= 0:
                    selected.append((doc_index, sentence_id))
                    remaining = remaining[match + len(words) + 1:]
            candidates.append(selected)

        # Keep early evidence from each document before taking additional sentences.
        ranked = (item for row in zip_longest(*candidates) for item in row if item)
        for _, (doc_index, sentence_id) in zip(range(return_top_k), ranked):
            selected_batch[index][doc_index].append(sentence_id)

    return selected_batch


def append_evidence_context(previous: str, new_text: str, sep: str = "\n\n---\n\n") -> str:
    """Append new evidence text while avoiding exact duplication."""
    previous = (previous or "").strip()
    new_text = (new_text or "").strip()

    if not new_text:
        return previous
    if not previous:
        return new_text
    if new_text in previous:
        return previous
    return previous + sep + new_text


def update_task_with_evidence(task_content: str, query: str, evidence_block: str) -> str:
    """Append retrieved evidence to the task content."""
    evidence_block = (evidence_block or "").strip()
    if not evidence_block:
        return task_content
    return f"{task_content}Query: {query}\nRetrieved Document: {evidence_block}\n"


def save_results_to_csv(df: pd.DataFrame, filename: str):
    """Append results to CSV with a stable column order."""
    df = df.reindex(columns=CSV_COLUMNS)
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode="a", header=False, index=False)


def _load_gold_answers(row):
    """
    读取标准答案, 兼容两种别名: 新版"Answer" / 旧版"Answers
    :param row: pandas的一行数据(DataFrame的一行)
    :return: list格式的标准答案列表
    """
    # 如果这一行存在旧列名, "Answers", 优先读取它
    if "Answers" in row.index:
        # safe_literal_eval: 安全地解析字符串为Python对象(比如列表、字典)
        return safe_literal_eval(row["Answers"])
    # 如果这一行存在新列名, "Answer", 读取它
    if "Answer" in row.index:
        value = row["Answer"]
        # 解析为列表, 如果已经是列表则直接返回
        parsed = safe_literal_eval(value)
        return parsed if isinstance(parsed, list) else [value]
    # 如果两种都没找到, 返回空列表
    return []


def _load_gold_documents(row):
    """Load gold document titles from the CSV row."""
    if "Documents" not in row.index:
        return []
    return safe_literal_eval(row["Documents"])


def main_batch(
    task_contents,
    question_type,
    ids,
    batch_gold_answers,
    gold_retrieved_docs,
    checkpoint_file,
    max_turns,
    dataset_name,
    gpt,
    top_docs,
    remove_repeat_docs,
    batch_questions=None,
    write_csv=True,
    provence=None,
    history_repair_top_k=0,
):
    """
    Run one batched multi-turn RAG loop.
    单批次多轮自适应RAG主循环
    核心流程: 对一批样本、循环执行【充足性Gate判断 -> 需要则构造子查询+BM25检索 -> 精选证据 -> 更新上下文】
    Gate判断信息足够 / 达到最大轮次时，送入reasoner生成最终答案
    每一轮都会保存样本状态(Gate输出、已检索文档、证据、预测答案、对错标记)
    参数:
        task_contents: 每条样本的reasoner任务内容(question + context)
        question_type: 问题类型，传给get_answer_system_prompt（比如multi-hop / single-hop）
        ids: 样本ID列表
        batch_gold_answers: 标准答案列表
        gold_retrieved_docs: 标准相关文档列表
        checkpoint_file: 结果保存CSV文件路径
        max_turns: 单条问题最大检索轮次
        dataset_name: 数据集名字
        gpt: bool，是否使用gpt api作为reasoner，否则本地Llama
        top_docs: 每一轮检索返回top多少个文档
        remove_repeat_docs: 是否过滤掉重复检索文档
        batch_questions: 问题列表
        write_csv: bool，是否写入CSV文件
        provence: 溯源模块开关，不为None时使用provence做句子筛选，否则用concat_and_pick_sentences_batch
    """
    # 没传问题, 就填充空字符串
    if batch_questions is None:
        batch_questions = [""] * len(task_contents)

    turn = 0 # 当前轮次计数
    results = [] # 存放所有轮次的记录，每一轮每条样本生成一条记录
    all_past_titles = [[] for _ in task_contents] # 每个样本: 历史已经检索过的文档标题
    all_past_doc_ids = [[] for _ in task_contents] # 每个样本: 历史已经检索过的文档ID, 去重用
    evidence_context_so_far = [""] * len(task_contents) # 每个样本: 经过筛选后的证据上下文(送入reasoner的精简上下文)
    raw_retrieved_concat_so_far = [""] * len(task_contents) # 每个样本: 原始检索文档全文拼接(留作日志，不直接喂给模型)
    history_docs = [[] for _ in task_contents] if history_repair_top_k else None

    while turn <= max_turns and task_contents: # 循环执行，直到信息充足或达到最大轮次
        # ====== Step 1: 调用sufficiency gate批量判断信息是否充足 ======
        # verdicts: bool列表 True=信息足够， 可以回答； false=信息不足，需要继续检索
        # missing_facts_list: Gate识别出来还缺少的事实，用来构造下一轮检索子问题
        verdicts, missing_facts_list = call_suff_gate_batch(
            batch_questions,
            evidence_context_so_far,
            gate_model,
            gate_tokenizer,
        )

        # 强制逻辑: 首轮强制检索，无视Gate说“信息不足”
        for i, verdict in enumerate(verdicts):
            if should_force_first_retrieval(turn, evidence_context_so_far[i], verdict):
                verdicts[i] = False
                missing_facts_list[i] = []

        if history_docs is not None and 0 < turn < max_turns:
            recovered = recover_history_evidence(
                provence, batch_questions, verdicts, missing_facts_list, history_docs,
                evidence_context_so_far, split_wiki_sentences, budget=history_repair_top_k,
            )
            if recovered:
                repair_indices = [i for i, _, _ in recovered]
                for i, query, evidence in recovered:
                    evidence_context_so_far[i] = append_evidence_context(evidence_context_so_far[i], evidence)
                    task_contents[i] = update_task_with_evidence(task_contents[i], f"[History repair] {query}", evidence)
                repaired_verdicts, repaired_gaps = call_suff_gate_batch(
                    [batch_questions[i] for i in repair_indices],
                    [evidence_context_so_far[i] for i in repair_indices],
                    gate_model, gate_tokenizer,
                )
                for i, verdict, gaps in zip(repair_indices, repaired_verdicts, repaired_gaps, strict=True):
                    verdicts[i], missing_facts_list[i] = verdict, gaps

        # 划分两类样本
        # done_idx: 本轮可以停止检索，生成答案的样本(Gate说信息足够了 / 已经耗尽了max_turns最大轮次)
        done_idx = [i for i, verdict in enumerate(verdicts) if verdict or turn == max_turns]
        # need_retrieve_idx: 本轮需要继续检索的样本(Gate说信息不足，但还没达到max_turns最大轮次)
        need_retrieve_idx = [i for i, verdict in enumerate(verdicts) if (not verdict) and turn < max_turns]

        # ====== Step 2: 需要检索的样本，执行BM25检索 + 证据筛选 ======
        if need_retrieve_idx: # 如果还有需要继续检索的样本
            # 根据Gate输出的缺失事实，构造子查询(下一轮检索query)
            sub_queries = [
                build_query_from_missing(
                    batch_questions[i],
                    missing_facts_list[i],
                    dataset_name=dataset_name,
                )
                for i in need_retrieve_idx
            ]
            # 取出这批待检索样本的历史文档id，用于去重
            sub_past_doc_ids = [all_past_doc_ids[i] for i in need_retrieve_idx] # 每个样本: 本轮之前已经检索过的文档ID, 去重用

            # BM25批量检索，返回文档标题、正文、文档id
            sub_titles, sub_texts, sub_doc_ids = bm25_search_batch(
                sub_queries,
                sub_past_doc_ids,
                k=top_docs,
                remove_repeat_docs=remove_repeat_docs,
            )

            # 把本轮新检索到的文档，追加到该样本的历史记录里(去重)
            for local_index, global_index in enumerate(need_retrieve_idx):
                if history_docs is not None:
                    history_docs[global_index].extend(zip(sub_titles[local_index], sub_texts[local_index]))
                for doc_id, title in zip(sub_doc_ids[local_index], sub_titles[local_index]):
                    if doc_id and doc_id != "No results found." and doc_id not in all_past_doc_ids[global_index]:
                        all_past_doc_ids[global_index].append(doc_id)
                    if title and title != "No results found." and title not in all_past_titles[global_index]:
                        all_past_titles[global_index].append(title)

                # 拼接本轮原始检索文档文本，追加到raw记录(原始全文，用于日志分析)
                new_raw = concat_raw_retrieved_docs(sub_titles[local_index], sub_texts[local_index])
                if not new_raw:
                    continue
                if raw_retrieved_concat_so_far[global_index]:
                    raw_retrieved_concat_so_far[global_index] += "\n\n---\n\n" + new_raw
                else:
                    raw_retrieved_concat_so_far[global_index] = new_raw

            retrieval_questions = [batch_questions[i] for i in need_retrieve_idx]
            # ====== Step 3: 证据筛选 ======
            # 如果没传provence，则用concat_and_pick_sentences_batch做句子筛选
            if provence is None or history_repair_top_k:
                # 用concat_and_pick_sentences_batch做句子筛选
                sub_evidence_ids = concat_and_pick_sentences_batch(
                    questions=retrieval_questions,
                    titles_batch=sub_titles,
                    texts_batch=sub_texts,
                    missing_facts_batch=[missing_facts_list[i] for i in need_retrieve_idx],
                    dataset_name=dataset_name,
                    gpt=gpt,
                    return_top_k=6,
                    max_sents_per_doc=40,
                )
            else:
                # 如果传了provence，则用select_provence_sentence_ids做句子筛选
                sub_evidence_ids = select_provence_sentence_ids(
                    provence, retrieval_questions, sub_texts
                )
                
            
            # 合并选中句子，更新evidence上下文和prompt
            for local_index, global_index in enumerate(need_retrieve_idx):
                merged = merge_evidence_only(
                    sub_titles[local_index],
                    sub_texts[local_index],
                    sub_evidence_ids[local_index],
                )
                # 把本轮精选证据追加到历史证据上下文
                evidence_context_so_far[global_index] = append_evidence_context(
                    evidence_context_so_far[global_index],
                    merged,
                )
                # 更新送入reasoner的task prompt(把新证据塞进去)
                task_contents[global_index] = update_task_with_evidence(
                    task_contents[global_index],
                    sub_queries[local_index],
                    merged,
                )

        # ====== Step 4: 本轮可以停止检索的样本，送入reasoner生成最终答案 ======
        predicted_answers = {} # 每个样本: 本轮预测的答案
        correct_answer_flags = {} # 每个样本: 本轮预测答案是否正确

        if done_idx:
            done_tasks = [task_contents[i] for i in done_idx]
            
            # 批量调用reasoner(gpt/本地Llama)生成最终答案
            done_outputs = call_reasoner_batch(
                system_prefixes=get_answer_system_prompt(dataset_name),
                task_contents=done_tasks,
                gpt=gpt,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=128,
                do_sample=False,
            )

            # 解析输出，提取答案，归一化, 计算Exact Match(多标答匹配)
            for local_index, global_index in enumerate(done_idx):
                pred_answer, _ = extract_final_answer_and_rationale(
                    done_outputs[local_index],
                    question_type,
                )
                pred_answer_norm = normalize_answer(pred_answer)
                predicted_answers[global_index] = pred_answer_norm
                correct_answer_flags[global_index] = bool(
                    exact_match_multiple(pred_answer_norm, batch_gold_answers[global_index])
                )

        # ====== Step 5: 记录本轮结果，准备进入下一轮 ======
        for i in range(len(task_contents)):
            # 提取本轮所有历史检索过的文档标题，去重
            unique_titles = list(dict.fromkeys(all_past_titles[i]))
            # Correct Retrieval: 所有gold文档标题都被检索到才算true
            correct_retrieval = all(
                exact_match_multiple(gold_title, unique_titles)
                for gold_title in gold_retrieved_docs[i]
            )

            pred_answer_norm = predicted_answers.get(i, "")
            correct_answer = correct_answer_flags.get(i, False) if pred_answer_norm else False

            results.append(
                {
                    "ID": ids[i],
                    "Turn": turn + 1,
                    "Reasoner Task Content": task_contents[i],
                    "Evidence Context So Far": evidence_context_so_far[i],
                    "Raw Retrieved Docs": raw_retrieved_concat_so_far[i],
                    "Gate Output": bool(verdicts[i]),
                    "SuffMissingFacts": missing_facts_list[i],
                    "Reasoner Answer": pred_answer_norm,
                    "Gold Answer": batch_gold_answers[i],
                    "Correct Answer": correct_answer,
                    "Gold Retrieved Docs": gold_retrieved_docs[i],
                    "Retrieved Titles So Far": unique_titles,
                    "Correct Retrieval": correct_retrieval,
                }
            )

        # 如果本轮没有需要继续检索的样本，直接跳出while循环
        if not need_retrieve_idx:
            break
        
        # ====== Step 6: 过滤列表，只保留need_retrieve_idx对应的样本，进入下一轮循环 ======
        # 只把还需要继续检索的样本留在列表里，其它样本丢弃(已经生成答案，不再参与后续轮次)
        task_contents = [task_contents[i] for i in need_retrieve_idx]
        ids = [ids[i] for i in need_retrieve_idx]
        batch_gold_answers = [batch_gold_answers[i] for i in need_retrieve_idx]
        gold_retrieved_docs = [gold_retrieved_docs[i] for i in need_retrieve_idx]
        batch_questions = [batch_questions[i] for i in need_retrieve_idx]
        all_past_titles = [all_past_titles[i] for i in need_retrieve_idx]
        all_past_doc_ids = [all_past_doc_ids[i] for i in need_retrieve_idx]
        evidence_context_so_far = [evidence_context_so_far[i] for i in need_retrieve_idx]
        raw_retrieved_concat_so_far = [raw_retrieved_concat_so_far[i] for i in need_retrieve_idx]
        if history_docs is not None:
            history_docs = [history_docs[i] for i in need_retrieve_idx]
        turn += 1

    # 可选: 本轮结果写入csv
    if write_csv and checkpoint_file:
        save_results_to_csv(pd.DataFrame(results, columns=CSV_COLUMNS), checkpoint_file)

    return results


def run_system(
    input_file,
    output_file,
    question_type,
    log_file,
    dataset_name,
    batch_size=8,
    max_turns=4,
    gpt=False,
    top_docs=6,
    remove_repeat_docs=False,
    start_index=0,
    end_index=5000,
    provence=None,
    history_repair_top_k=0,
):
    """
    Run the end-to-end evaluation loop.
    端到端评估主循环函数
    核心逻辑: 分批读取评测数据集，批量执行多轮RAG推理(带sufficiency gate判断是否继续检索)
    保存每轮结果，最后统计: 答案正确率、检索正确率、平均轮次，写入日志
    参数说明：
        input_file: 输入评测csv文件路径（包含问题、ID、标答等）
        output_file: 输出结果csv路径，保存每条样本每一轮的推理记录
        question_type: 问题类型，传给main_batch（比如multi-hop / single-hop）
        log_file: 汇总指标日志文件路径
        dataset_name: 数据集名字，用于日志标记
        batch_size: 推理批大小
        max_turns: 单条问题最大检索轮次（gate多次判断，最多循环几轮检索）
        gpt: bool，是否使用gpt api作为reasoner，否则本地Llama
        top_docs: 每一轮检索返回top多少个文档
        remove_repeat_docs: 是否过滤掉重复检索文档
        start_index: 数据集切片起始行，方便断点续跑、分片评测
        end_index: 数据集切片终止行
        provence: 应该是 provenance，溯源相关参数，传给main_batch
    """
    # 读取评测数据集
    data = pd.read_csv(input_file)
    total_rows = len(data)
    # 修正起始下标，不能小于0
    start = max(0, int(start_index or 0))
    # 如果没指定end_index就读到末尾，否则不超过总行数
    end = total_rows if end_index is None else min(total_rows, int(end_index))
    # 如果起始下标大于等于终止下标，则报错
    if start >= end:
        print(f"[Subset] Empty slice: start={start}, end={end}, total={total_rows}")
        return

    # 切片数据集，重置索引
    data = data.iloc[start:end].reset_index(drop=True)
    print(f"[Subset] Using rows [{start}, {end}) -> {len(data)} rows")

    # 按batch_size遍历，分批处理
    for offset in range(0, len(data), batch_size):
        # 取出当前batch
        batch = data.iloc[offset : offset + batch_size]

        # 提取当前批次每条样本的ID、问题、标准答案、标准相关文档
        batch_ids = [row["ID"] for _, row in batch.iterrows()]
        batch_questions = [row["Question"] for _, row in batch.iterrows()]
        batch_gold_answers = [_load_gold_answers(row) for _, row in batch.iterrows()]
        batch_gold_docs = [_load_gold_documents(row) for _, row in batch.iterrows()]

        # 构建每条样本的reasoner任务内容
        task_contents = [
            f"Question: {question}\n" if gpt else f"Question: {question}\nContext:\n"
            for question in batch_questions
        ]

        # 调用主推理函数，执行多轮RAG推理
        # 调用核心推理函数 main_batch，执行多轮RAG推理(带sufficiency gate判断是否继续检索)
        # main_batch内部会循环多轮检索+gate判断，直到信息充足或者达到max_turns
        batch_results = main_batch(
            task_contents=task_contents,
            question_type=question_type,
            ids=batch_ids,
            batch_gold_answers=batch_gold_answers,
            gold_retrieved_docs=batch_gold_docs,
            checkpoint_file=output_file,
            max_turns=max_turns,
            dataset_name=dataset_name,
            gpt=gpt,
            top_docs=top_docs,
            remove_repeat_docs=remove_repeat_docs,
            batch_questions=batch_questions,
            write_csv=False, # 不在main_batch内部写文件，最后统一写入output_file
            provence=provence,
            history_repair_top_k=history_repair_top_k,
        )

        if batch_results: # 如果batch_results非空，则写入output_file
            save_results_to_csv(pd.DataFrame(batch_results, columns=CSV_COLUMNS), output_file)

    if os.path.exists(output_file): # 如果output_file存在，则读取
        df = pd.read_csv(output_file)
    else:
        df = pd.DataFrame(columns=CSV_COLUMNS) # 如果output_file不存在，则创建空DataFrame

    if len(df):
        df_sorted = df.sort_values(["ID", "Turn"])
        df_final = df_sorted.groupby("ID", as_index=False).tail(1)
    else:
        df_final = pd.DataFrame(columns=CSV_COLUMNS)

    # 指标计算
    acc = float(df_final["Correct Answer"].mean()) if len(df_final) else 0.0 # 答案正确率
    ret = float(df_final["Correct Retrieval"].mean()) if len(df_final) else 0.0 # 检索正确率
    avg_turn = float(df_final["Turn"].mean()) if len(df_final) else 0.0 # 平均检索轮次

    # 日志目录不存在则创建，追加写入日志
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True) # 创建日志文件目录
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n===== Final Evaluation =====\n")
        f.write(f"Num rows (all turns): {len(df)}\n") # 所有轮次的总记录数(一条问题有多轮)
        f.write(f"Num questions (final rows): {len(df_final)}\n") # 有效问题数量()
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Correct Answer (mean): {acc:.4f}\n") # 答案准确率 
        f.write(f"Correct Retrieval (mean): {ret:.4f}\n") # 检索准确率
        f.write(f"Average Turn: {avg_turn:.2f}\n") # 平均需要几轮检索才能给出答案
        f.write("============================================\n")

    print("Generation complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-turn RAG with a sufficiency gate and evidence selector."
    )
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["hotpotqa", "triviaqa", "2wikimultihopqa"],
        default=None,
    )
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--question_type", type=str, choices=["MCQ", "OEQ", "MATH"], default="OEQ")
    parser.add_argument("--gpt", action="store_true", default=False)
    parser.add_argument("--top_docs", type=int, default=6)
    parser.add_argument("--remove_repeat_docs", action="store_true", default=False)
    parser.add_argument(
        "--evidence_selector", choices=["s2g", "provence"], default="s2g",
        help="Use the original S2G sentence extractor or Provence.",
    )
    parser.add_argument(
        "--provence_model", default=PROVENCE_MODEL,
        help="Provence Hugging Face model ID or path to a local checkpoint.",
    )
    parser.add_argument(
        "--history_repair", action="store_true",
        help="Use Provence to recover missing evidence from earlier retrieval turns.",
    )
    parser.add_argument("--history_repair_top_k", type=int, default=3)
    args = parser.parse_args()
    if args.history_repair and args.evidence_selector != "s2g":
        parser.error("--history_repair requires --evidence_selector s2g")
    if args.history_repair and args.history_repair_top_k < 1:
        parser.error("--history_repair_top_k must be positive")
    return args


def load_corpus_and_searcher(dataset_name):
    """
    加载语料库、检索参数，初始化BM25检索器
    """
    # 根据数据集名字，拿到corpus(语料)和index配置文件的路径
    corpus_path, settings_path = get_bm25_resource_paths(dataset_name)

    # 读取预先保存好的完整语料(pickle序列化文件)
    with open(corpus_path, "rb") as f:
        loaded_corpus = pickle.load(f)
    
    # 读取BM25索引的配置信息(k1, b, 索引文件夹路径等)
    with open(settings_path, "rb") as f:
        settings = pickle.load(f)

    # 从保存好的索引目录加载BM25检索器(pyserini的SimpleSearcher类)
    searcher_obj = SimpleSearcher(settings["index_dir"])
    # 设置BM25超参数 k1, b: 如果配置里没有，默认k1=0.9, b=0.4
    searcher_obj.set_bm25(settings.get("k1", 0.9), settings.get("b", 0.4))
    return loaded_corpus, searcher_obj


def load_reasoner_and_gate(args):
    """
    Load the main reasoner and the sufficiency gate.
    加载主推理模型(reasoner)和充足性判断门控模型(sufficiency gate)
    充足性门控: 用来判断当前已有信息是否足够回答问题，是RAG里常见的判断模块
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    global MODEL # 全局变量，标记使用的模型名称(仅gpt分支用到)

    # ====分支1: 使用OpenAI接口(gpt-40-mini)作为reasoner
    if args.gpt:
        if not os.getenv("CHATANYWHERE_API_KEY"):
            raise ValueError("CHATANYWHERE_API_KEY not set.")
        MODEL = "gpt-4o-mini"
        loaded_tokenizer = None # 调用API，不需要本地tokenizer
        loaded_model = None # 调用API，不需要本地模型权重
    else:
        # 从环境变量读取Llama模型本地路径
        model_id = os.getenv("LLAMA_PATH")
        if not model_id: # 如果环境变量未设置，则报错
            raise ValueError("LLAMA_PATH not set.")
        # 自动选择精度: GPU支持bf16就用bf16，否则回退fp16
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 # 根据GPU支持情况选择精度
        # 加载reasoner的分词器， padding在左侧(生成类模型常用left_padding)
        loaded_tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        # 如果分词器没有pad_token, 就用eos_token充当pad_token(Llama原生无pad_token)
        if loaded_tokenizer.pad_token is None:
            loaded_tokenizer.pad_token = loaded_tokenizer.eos_token

        # 加载reasoner的模型权重
        loaded_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True, # 低内存占用优化
            device_map="auto",
        )
        # 同步模型config里的pad_token_id，保证和tokenizer一致
        loaded_model.config.pad_token_id = loaded_tokenizer.pad_token_id
    
    # ======== 下面固定加载 sufficiency gate(充足性判别模型, LoRA微调版)
    suff_base_path = os.getenv("SUFF_BASE_PATH")
    if not suff_base_path:
        raise ValueError("SUFF_BASE_PATH must be set.")

    # gate模型同样自动选精度: GPU支持bf16就用bf16，否则回退fp16
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # gate专属分词器, use_fast=False, 加载gate模型的分词器， padding在左侧(生成类模型常用left_padding)
    loaded_gate_tokenizer = AutoTokenizer.from_pretrained(
        suff_base_path,
        padding_side="left",
        use_fast=False,
    )
    # gate分词器补pad token
    if loaded_gate_tokenizer.pad_token is None:
        loaded_gate_tokenizer.pad_token = loaded_gate_tokenizer.eos_token

    # 先加载gate的底座基础模型
    gate_base_model = AutoModelForCausalLM.from_pretrained(
        suff_base_path,
        torch_dtype=compute_dtype,
    )

    # 读取gate的LoRA权重路径
    suff_lora_path = os.getenv("SUFF_LORA_PATH")
    if not suff_lora_path:
        raise ValueError("SUFF_LORA_PATH must be set.")

    print(f"[INFO] Loading sufficiency LoRA from {suff_lora_path}")
    # 把LoRA适配器挂载到底座模型上，得到gate模型
    loaded_gate_model = PeftModel.from_pretrained(gate_base_model, suff_lora_path)
    loaded_gate_model.config.pad_token_id = loaded_gate_tokenizer.pad_token_id

    # accelerate 包装(适配多卡、混合精度训练/推理)
    if args.gpt:
        # GPT模式: reasoner是API，只需要prepare gate模型即可
        loaded_gate_model = accelerator.prepare(loaded_gate_model)
    else:
        # 本地Llama模式: reasoner + gate两个模型都交给accelerate管理
        loaded_model, loaded_gate_model = accelerator.prepare(loaded_model, loaded_gate_model)
        loaded_model.eval() # reasoner设为评估模式, 关闭dropout等训练组件

    loaded_gate_model.eval() # gate模型设为评估模式, 关闭dropout等训练组件
    # 返回: reasoner tokenizer, reasoner model, gate tokenizer, gate model
    return loaded_tokenizer, loaded_model, loaded_gate_tokenizer, loaded_gate_model


def main():
    global accelerator, tokenizer, model, gate_tokenizer, gate_model, searcher, corpus

    args = parse_args()

    # 根据输入路径、实验名自动推断数据集名称(比如2wikimultihopqa / hotpotqa)
    if args.input_path is None:
        args.input_path = f"data/original/{args.experiment_name}_dev.csv"
    if args.output_path is None:
        args.output_path = f"predictions/{args.experiment_name}_predictions.csv"
    if args.log_path is None:
        args.log_path = f"logs/{args.experiment_name}_log.txt"

    # 根据输入路径、实验名自动推断数据集名称(比如2wikimultihopqa / hotpotqa)
    args.dataset_name = infer_dataset_name(
        dataset_name=args.dataset_name,
        input_path=args.input_path,
        experiment_name=args.experiment_name,
    )

    # 初始化accelerate工具: 负责多卡/单卡、混合精度、设备分发
    accelerator = Accelerator()
    # 设置CUDA内存分配配置: 允许扩展的段(segments)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # 清空CUDA缓存: 确保每次运行前都是干净的
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 加载语料库corpus + 检索器searcher(BM25 / E5稠密检索在这里初始化)
    corpus, searcher = load_corpus_and_searcher(args.dataset_name)
    
    # 加载回答推理主模型(reasoner) + s2g-judge门控判断机制(gate_model), 以及各自tokenizer
    tokenizer, model, gate_tokenizer, gate_model = load_reasoner_and_gate(args)

    # 证据抽取器
    provence = None
    if args.evidence_selector == "provence" or args.history_repair:
        provence = load_provence(args.provence_model, accelerator.device)

    # 正式启动S2G-RAG主推理循环
    run_system(
        input_file=args.input_path, # 输入数据集csv
        output_file=args.output_path, # 输出结果csv
        question_type=args.question_type, # 问题类型
        log_file=args.log_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        max_turns=args.max_turns,
        gpt=args.gpt, # 是否调用GPT
        top_docs=args.top_docs, # 每一轮检索返回top-k个文档
        remove_repeat_docs=args.remove_repeat_docs, # 是否去重文档
        start_index=args.start_index, # 从哪一行开始跑数据
        end_index=args.end_index, # 到哪一行结束
        provence=provence,
        history_repair_top_k=args.history_repair_top_k if args.history_repair else 0,
    )


if __name__ == "__main__":
    main()
