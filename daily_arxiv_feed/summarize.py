import logging

from openai import OpenAI

from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.llm import chat, parse_json_response
from daily_arxiv_feed.excerpts import (
    PROSE_RULES, abstract_excerpt, budget_instruction, clean_abstract,
    fits_budget, summary_limits,
)

logger = logging.getLogger(__name__)

SUMMARIZE_SYSTEM_PROMPT = "You edit scientific abstracts for a readable research digest.\n\n" + PROSE_RULES


def build_summarize_prompt(paper: Paper, decisions: list[dict], limits: dict | None = None) -> str:
    decision = next((d for d in decisions if d["arxiv_id"] == paper.arxiv_id), None)
    cats = decision["categories"] if decision else []
    reasoning = decision["reasoning"] if decision else ""
    cat_names = {1: "Multilingual NLP", 2: "Multilingual Speech", 3: "Foundation Model Theory", 4: "LLM Evaluation"}
    matched = ", ".join(cat_names.get(c, str(c)) for c in cats)

    return f"""arxiv_id: {paper.arxiv_id}
Title: {paper.title}
Authors: {', '.join(paper.authors)}
Matched categories: {matched}
Selection context (not evidence for factual claims): {reasoning}

{budget_instruction(limits or summary_limits())}

Abstract:
{clean_abstract(paper.abstract)}"""


def generate_excerpt(client: OpenAI, paper: Paper, limits: dict, feedback: list[str] | None = None) -> tuple[dict, str]:
    prompt = build_summarize_prompt(paper, [], limits)
    if feedback:
        prompt += "\nCorrect these issues:\n" + "\n".join(feedback)
    for attempt in range(2):
        response = chat(client=client, system=SUMMARIZE_SYSTEM_PROMPT, user=prompt, json_mode=True)
        try:
            candidate = parse_json_response(response)
            if set(candidate) != {"excerpt"} or not isinstance(candidate["excerpt"], str):
                raise ValueError('Return exactly {"excerpt": "one paragraph"}')
            candidate = {"excerpt": " ".join(candidate["excerpt"].split())}
            if not fits_budget(candidate, limits):
                raise ValueError(budget_instruction(limits))
            return candidate, "compressed_abstract"
        except (ValueError, TypeError) as e:
            logger.warning("Invalid excerpt for %s (attempt %d): %s", paper.arxiv_id, attempt + 1, e)
            prompt += f"\nThe previous response was rejected: {e}. Rewrite by omitting secondary details."
    return abstract_excerpt(paper.abstract, limits), "abstract_sentences"


def summarize_papers(client: OpenAI, papers: list[Paper], decisions: list[dict],
                     previous: list[dict] | None = None) -> list[dict]:
    results = []
    previous_map = {s["arxiv_id"]: s for s in (previous or [])}
    for paper in papers:
        limits = summary_limits(previous_map.get(paper.arxiv_id))
        summary, source = generate_excerpt(client, paper, limits)
        results.append({
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors,
            "url": paper.url,
            "categories": next((d["categories"] for d in decisions if d["arxiv_id"] == paper.arxiv_id), []),
            "summary": summary,
            "summary_limits": limits,
            "summary_source": source,
        })
        logger.info("Summarized: %s", paper.arxiv_id)
    return results
