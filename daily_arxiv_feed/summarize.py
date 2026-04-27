import logging

from openai import OpenAI

from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.llm import chat, parse_json_response

logger = logging.getLogger(__name__)

SUMMARIZE_SYSTEM_PROMPT = """You are summarizing an arxiv paper for a senior NLP researcher's daily 2-page digest. Space is extremely limited — you MUST be brief.

Rules:
- Explain concepts in plain language, not jargon from the abstract
- Every sentence must carry information — no filler, no hedging, no "interestingly"
- The reader should understand the key idea WITHOUT reading the paper
- More important papers get slightly more detail; minor ones get bare minimum

Respond with JSON:
{
  "one_line_takeaway": "One punchy sentence: why this paper matters",
  "key_contribution": "What they did (1-2 sentences, max 50 words)",
  "method": "How they did it (1-2 sentences, max 35 words)",
  "most_important_result": "The headline number or finding (1-2 sentences, max 35 words)"
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
        summary = parse_json_response(response)
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
