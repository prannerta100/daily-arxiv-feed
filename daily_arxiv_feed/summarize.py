import logging

from openai import OpenAI

from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.llm import chat, parse_json_response

logger = logging.getLogger(__name__)

SUMMARIZE_SYSTEM_PROMPT = """You are explaining an arxiv paper to a smart colleague. They haven't read it. Your job is to give them genuine understanding of the core idea — what insight drives this work and why it matters.

Write like a human, not like an abstract compressor. Narrate the crux. If the idea is deep or subtle, take the space you need — there is no hard word limit. If the idea is simple, keep it short. Match length to the complexity of the insight.

BAD: "Identifies a post-answer newline activation as a second-order confidence signal predicting error detection beyond logprobs"
GOOD: "They found a hidden signal inside LLMs — the activation pattern right after the model finishes its answer — that reveals when the model knows it got something wrong, even when its stated confidence says otherwise. This matters because it means we can catch errors without asking the model to self-evaluate."

Rules:
- Narrate the actual insight — what's the core idea and why does it work?
- Use plain, natural English — no stacking nouns or jargon from the abstract
- Specific model names, dataset names, and benchmark scores are fine when they add understanding, but don't list them mechanically
- Each field should be 2-4 complete sentences written in full prose
- NEVER omit subjects from sentences (bad: "Identifies X..." good: "They identify X...")
- NEVER compress by removing articles, spaces, or connectives
- NEVER write in telegram/shorthand style — write how you'd actually explain it out loud
- one_line_takeaway: minimum 20 words, a complete thought
- key_contribution, method, most_important_result: minimum 50 words each

Respond with JSON:
{
  "one_line_takeaway": "Why should I care about this paper? (full sentence, 20+ words)",
  "key_contribution": "What did they actually do? Narrate the core insight. (50+ words, full prose)",
  "method": "How does it work? Explain the key mechanism or approach. (50+ words, full prose)",
  "most_important_result": "What happened when they tried it? What did we learn? (50+ words, full prose)"
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
