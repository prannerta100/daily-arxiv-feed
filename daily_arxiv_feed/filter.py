import hashlib
import json
import logging
from pathlib import Path

from openai import OpenAI

from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.llm import chat, parse_json_response

logger = logging.getLogger(__name__)

FILTER_SYSTEM_PROMPT = """You are a research paper filter for a senior NLP researcher. From this batch of papers, identify ANY that could plausibly match these categories:

1. **Multilingual NLP** — paper's core contribution involves multilingual or cross-lingual NLP
2. **Multilingual speech processing** — paper's core contribution involves multilingual or cross-lingual speech/audio
3. **Foundation model theory insights** — reveals genuine mechanistic insights into how large models work, or a clever trick with strong evidence
4. **LLM evaluation** — proposes genuinely novel automatic evaluation methodology or reveals counter-intuitive evaluation findings

This is a FIRST PASS — mark papers as candidates if they have a reasonable chance of fitting. A second pass will do the final selection.

Respond with JSON: {"papers": [{"arxiv_id": "...", "selected": true/false, "categories": [1,2,...], "reasoning": "brief explanation"}]}"""

RANK_SYSTEM_PROMPT = """You are doing the FINAL selection of arxiv papers for a senior NLP researcher's daily 2-page digest. You have a list of candidate papers that passed initial filtering. You must now pick the BEST ones under strict quotas:

- Category 1 (Multilingual NLP): pick the TOP 3-4 most novel/impactful. Reject papers where multilingual is incidental.
- Category 2 (Multilingual speech): pick the TOP 1-2. Reject papers where multilingual is incidental.
- Category 3 (Foundation model theory): pick AT MOST 1 — only if it reveals something genuinely new about model internals. Most days pick 0.
- Category 4 (LLM evaluation): pick AT MOST 1 — only if evaluation methodology is truly novel. Most days pick 0.

TOTAL must be 5-8 papers. You are reading a 2-page digest — every paper must earn its space.

You will receive candidate papers with their abstracts and initial reasoning. Rank ruthlessly.

Respond with JSON: {"papers": [{"arxiv_id": "...", "selected": true/false, "categories": [1,2,...], "reasoning": "why this made or didn't make the final cut"}]}"""


def build_filter_prompt(papers: list[Paper]) -> str:
    lines = []
    for p in papers:
        lines.append(f"--- arxiv_id: {p.arxiv_id} ---")
        lines.append(f"Title: {p.title}")
        lines.append(f"Abstract: {p.abstract}")
        lines.append("")
    return "\n".join(lines)


def _build_rank_prompt(candidates: list[dict], paper_map: dict[str, Paper]) -> str:
    lines = []
    for c in candidates:
        p = paper_map.get(c["arxiv_id"])
        if not p:
            continue
        lines.append(f"--- arxiv_id: {c['arxiv_id']} ---")
        lines.append(f"Title: {p.title}")
        lines.append(f"Categories: {c['categories']}")
        lines.append(f"Initial reasoning: {c['reasoning']}")
        lines.append(f"Abstract: {p.abstract}")
        lines.append("")
    return "\n".join(lines)


def parse_filter_response(response: str) -> list[dict]:
    data = parse_json_response(response)
    return data["papers"]


def filter_papers(client: OpenAI, papers: list[Paper], *,
                  filter_system_prompt: str = FILTER_SYSTEM_PROMPT,
                  rank_system_prompt: str = RANK_SYSTEM_PROMPT,
                  max_papers: int | None = None, batch_size: int = 20,
                  checkpoint_path: Path | None = None) -> tuple[list[Paper], list[dict]]:
    if not papers:
        return [], []

    if batch_size<1:
        raise ValueError("Filter batch size must be positive")
    all_decisions = []
    completed_batches = 0
    fingerprint = hashlib.sha256(json.dumps({"prompt":filter_system_prompt,"batch_size":batch_size,
        "papers":[[p.arxiv_id,p.title,p.abstract] for p in papers]},ensure_ascii=False).encode()).hexdigest()
    if checkpoint_path and checkpoint_path.exists():
        try:
            cached = json.loads(checkpoint_path.read_text())
            if cached["fingerprint"]==fingerprint:
                all_decisions = cached["decisions"]
                completed_batches = cached["completed_batches"]
                logger.info("Resuming filter after %d completed batches",completed_batches)
        except (ValueError,KeyError):
            logger.warning("Ignoring invalid filter checkpoint %s",checkpoint_path)

    for i in range(completed_batches*batch_size, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        user_prompt = build_filter_prompt(batch)
        response = chat(client=client, system=filter_system_prompt, user=user_prompt, json_mode=True)
        returned = {d["arxiv_id"]:d for d in parse_filter_response(response)}
        # Candidate-only profiles need not spend tokens listing unrelated papers.
        # Record omissions as intentional non-nominations, not scorer failures.
        decisions = [returned.get(p.arxiv_id,{"arxiv_id":p.arxiv_id,"selected":False,
                     "categories":[],"reasoning":"Not nominated in first-pass selection"}) for p in batch]
        all_decisions.extend(decisions)
        if checkpoint_path:
            checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
            temporary = checkpoint_path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"fingerprint":fingerprint,
                "completed_batches":i//batch_size+1,"decisions":all_decisions},indent=2))
            temporary.replace(checkpoint_path)
        logger.info("Filtered batch %d/%d",i//batch_size+1,(len(papers)+batch_size-1)//batch_size)

    paper_map = {p.arxiv_id: p for p in papers}
    candidates = [d for d in all_decisions if d.get("selected") and d.get("arxiv_id") in paper_map]
    logger.info("First pass: %d candidates from %d papers", len(candidates), len(papers))

    if not candidates:
        return [], all_decisions

    rank_prompt = _build_rank_prompt(candidates, paper_map)
    rank_response = chat(client=client, system=rank_system_prompt, user=rank_prompt, json_mode=True)
    rank_decisions = parse_filter_response(rank_response)

    candidate_ids = {d["arxiv_id"] for d in candidates}
    final_ids = list(dict.fromkeys(d["arxiv_id"] for d in rank_decisions
                                  if d.get("selected") and d.get("arxiv_id") in candidate_ids))
    if max_papers is not None:
        final_ids = final_ids[:max_papers]
    selected = [paper_map[aid] for aid in final_ids]

    for d in all_decisions:
        rank_match = next((r for r in rank_decisions if r["arxiv_id"] == d["arxiv_id"]), None)
        if rank_match:
            d["selected"] = rank_match["selected"]
            d["reasoning"] = rank_match["reasoning"]
            d["categories"] = rank_match["categories"]
        d["selected"] = d.get("arxiv_id") in final_ids

    logger.info("Final selection: %d papers", len(selected))
    return selected, all_decisions
