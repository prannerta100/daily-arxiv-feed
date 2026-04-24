import json
import logging

from openai import OpenAI

from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.llm import chat

logger = logging.getLogger(__name__)

FILTER_SYSTEM_PROMPT = """You are a research paper filter for a senior NLP researcher. You will receive a list of arxiv papers with their abstracts. Select ONLY papers that match one or more of these categories:

1. **Multilingual NLP** — papers explicitly addressing multilingual or cross-lingual natural language processing. The paper must have a multilingual or cross-lingual component as a core contribution, not just multilingual evaluation tacked on.

2. **Multilingual speech processing** — papers addressing speech or audio processing in multilingual or cross-lingual contexts. Must involve multiple languages in the speech/audio domain.

3. **Foundation model theory insights** — papers presenting genuine mechanistic insights into how large foundation models work, or a demonstrably clever trick with strong evidence. Be VERY selective — most papers do NOT qualify. Only select if the paper reveals something genuinely new about why or how these models behave. Expect at most 1 paper per day in this category.

4. **LLM evaluation** — papers with novel AUTOMATIC evaluation methods or surprising insights into LLM-as-judge approaches. NOT yet-another-benchmark papers. Must present genuinely new evaluation methodology or counter-intuitive findings about evaluation. Expect at most 1 paper per day in this category.

Be strict. Most papers should NOT be selected. When in doubt, reject.

Respond with JSON: {"papers": [{"arxiv_id": "...", "selected": true/false, "categories": [1,2,...], "reasoning": "brief explanation"}]}"""


def build_filter_prompt(papers: list[Paper]) -> str:
    lines = []
    for p in papers:
        lines.append(f"--- arxiv_id: {p.arxiv_id} ---")
        lines.append(f"Title: {p.title}")
        lines.append(f"Abstract: {p.abstract}")
        lines.append("")
    return "\n".join(lines)


def parse_filter_response(response: str) -> list[dict]:
    data = json.loads(response)
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

    selected_ids = {d["arxiv_id"] for d in all_decisions if d.get("selected")}
    paper_map = {p.arxiv_id: p for p in papers}
    selected = [paper_map[aid] for aid in selected_ids if aid in paper_map]

    logger.info("Filtered %d/%d papers", len(selected), len(papers))
    return selected, all_decisions
