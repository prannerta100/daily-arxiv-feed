import logging

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


def filter_papers(client: OpenAI, papers: list[Paper]) -> tuple[list[Paper], list[dict]]:
    if not papers:
        return [], []

    batch_size = 20
    all_decisions = []

    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        user_prompt = build_filter_prompt(batch)
        response = chat(client=client, system=FILTER_SYSTEM_PROMPT, user=user_prompt, json_mode=True)
        decisions = parse_filter_response(response)
        all_decisions.extend(decisions)

    candidates = [d for d in all_decisions if d.get("selected")]
    logger.info("First pass: %d candidates from %d papers", len(candidates), len(papers))

    if not candidates:
        return [], all_decisions

    paper_map = {p.arxiv_id: p for p in papers}
    rank_prompt = _build_rank_prompt(candidates, paper_map)
    rank_response = chat(client=client, system=RANK_SYSTEM_PROMPT, user=rank_prompt, json_mode=True)
    rank_decisions = parse_filter_response(rank_response)

    final_ids = {d["arxiv_id"] for d in rank_decisions if d.get("selected")}
    selected = [paper_map[aid] for aid in final_ids if aid in paper_map]

    for d in all_decisions:
        rank_match = next((r for r in rank_decisions if r["arxiv_id"] == d["arxiv_id"]), None)
        if rank_match:
            d["selected"] = rank_match["selected"]
            d["reasoning"] = rank_match["reasoning"]
            d["categories"] = rank_match["categories"]

    logger.info("Final selection: %d papers", len(selected))
    return selected, all_decisions
