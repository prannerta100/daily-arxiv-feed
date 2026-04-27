import logging

from openai import OpenAI

from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.llm import chat, parse_json_response

logger = logging.getLogger(__name__)

SUMMARIZE_SYSTEM_PROMPT = """You are explaining an arxiv paper to a smart colleague over coffee. They haven't read it. You have 30 seconds.

Write like a human, not like an abstract compressor. Use natural sentences a person would actually say.

BAD: "Identifies a post-answer newline activation as a second-order confidence signal predicting error detection beyond logprobs"
GOOD: "They found a hidden signal inside LLMs that reveals when the model knows it got something wrong — even when its stated confidence says otherwise."

Rules:
- Write in plain, natural English — no stacking nouns or jargon from the abstract
- Never name specific models, datasets, or benchmarks unless the paper IS about that model/dataset
- Never list numbers, scores, or metrics — just say whether it worked well or not
- Each field is exactly one clear sentence
- If you wouldn't say it out loud to a colleague, rewrite it

Respond with JSON:
{
  "one_line_takeaway": "Why should I care about this paper? (one sentence, max 20 words)",
  "key_contribution": "What did they actually do? (one sentence, max 30 words)",
  "method": "How did they do it? (one sentence, max 25 words)",
  "most_important_result": "Did it work? What happened? (one sentence, max 25 words)"
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
