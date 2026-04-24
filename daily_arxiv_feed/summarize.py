import json
import logging

from openai import OpenAI

from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.llm import chat

logger = logging.getLogger(__name__)

SUMMARIZE_SYSTEM_PROMPT = """You are summarizing an arxiv paper for a senior NLP researcher's daily digest. The digest has a strict 2-page limit, so be concise but accurate.

You will be given the paper's title, abstract, and which interest category it matched. More relevant/insightful papers deserve more detail; less relevant ones can be brief.

Explain difficult concepts in plain language — do not just restate jargon from the abstract. The reader should understand the key idea without having read the paper.

Respond with JSON:
{
  "one_line_takeaway": "The 'so what' — why this paper matters, in one sentence",
  "key_contribution": "What they did (2-3 sentences max)",
  "method": "How they did it (2-3 sentences max)",
  "most_important_result": "The headline finding (1-2 sentences)"
}"""


def build_summarize_prompt(paper: Paper, decisions: list[dict]) -> str:
    decision = next((d for d in decisions if d["arxiv_id"] == paper.arxiv_id), None)
    cats = decision["categories"] if decision else []
    reasoning = decision["reasoning"] if decision else ""
    cat_names = {1: "Multilingual NLP", 2: "Multilingual Speech", 3: "Foundation Model Theory", 4: "LLM Evaluation"}
    matched = ", ".join(cat_names.get(c, str(c)) for c in cats)

    return f"""arxiv_id: {paper.arxiv_id}
Title: {paper.title}
Authors: {', '.join(paper.authors)}
Matched categories: {matched}
Selection reasoning: {reasoning}

Abstract:
{paper.abstract}"""


def summarize_papers(client: OpenAI, papers: list[Paper], decisions: list[dict]) -> list[dict]:
    results = []
    for paper in papers:
        user_prompt = build_summarize_prompt(paper, decisions)
        response = chat(client=client, system=SUMMARIZE_SYSTEM_PROMPT, user=user_prompt, json_mode=True)
        summary = json.loads(response)
        results.append({
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors,
            "url": paper.url,
            "categories": next((d["categories"] for d in decisions if d["arxiv_id"] == paper.arxiv_id), []),
            "summary": summary,
        })
        logger.info("Summarized: %s", paper.arxiv_id)
    return results
