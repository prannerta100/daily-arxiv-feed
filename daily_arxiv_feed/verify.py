import logging

from openai import OpenAI

from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.llm import chat, parse_json_response
from daily_arxiv_feed.summarize import generate_excerpt
from daily_arxiv_feed.excerpts import abstract_excerpt, clean_abstract, fits_budget, summary_limits, summary_text

logger = logging.getLogger(__name__)

VERIFY_SYSTEM_PROMPT = """You are verifying a summary of an arxiv paper against its original abstract.

Check both faithfulness and readability. Every factual claim must be supported by
the abstract or title. The title can establish a paper or benchmark name.
Flag invented mechanisms, causal explanations, stronger conclusions,
incorrect numbers/comparisons, and omitted qualifications that change meaning.
Also flag telegram-style fragments, missing subjects or verbs, unexplained acronym
chains, and compressed noun piles that a researcher must decode rather than read.
Flag a central named mechanism that is mentioned without explaining what it does.
Do not require more detail: selective coverage of the abstract is intentional.
Omitting an independent result, model size, or comparison is not itself an error.
An excerpt should read as connected, complete sentences in one compressed abstract.

Respond with JSON:
{"passed": true/false, "issues": ["specific factual or readability issues, empty if passed"]}"""


def _verify_one(client: OpenAI, abstract: str, summary: dict, title: str = "") -> dict:
    user_prompt = f"""Title: {title}

Original abstract:
{clean_abstract(abstract)}

Summary to verify:
{summary_text(summary)}"""

    response = chat(client=client, system=VERIFY_SYSTEM_PROMPT, user=user_prompt, json_mode=True)
    return parse_json_response(response)


def _regenerate_summary(client: OpenAI, paper: Paper, feedback: list[str], limits: dict) -> tuple[dict, str]:
    return generate_excerpt(client, paper, limits, feedback)


def verify_summaries(client: OpenAI, papers: list[Paper], summaries: list[dict]) -> list[dict]:
    paper_map = {p.arxiv_id: p for p in papers}
    results = []

    for entry in summaries:
        paper = paper_map.get(entry["arxiv_id"])
        if not paper:
            results.append(entry)
            continue

        limits = entry.get("summary_limits") or summary_limits()
        verification = _verify_one(client, paper.abstract, entry["summary"], paper.title)
        if not fits_budget(entry["summary"], limits):
            verification = {"passed": False, "issues": verification["issues"] + ["Excerpt exceeds its length budget."]}

        if verification["passed"]:
            results.append({**entry, "summary_verification": {"status":"passed", "attempts":1}})
            logger.info("Verified OK: %s", entry["arxiv_id"])
            continue

        logger.warning("Verification failed for %s: %s", entry["arxiv_id"], verification["issues"])
        new_summary, source = _regenerate_summary(client, paper, verification["issues"], limits)
        entry_copy = {**entry, "summary": new_summary, "summary_source": source, "summary_limits": limits}
        entry_copy.pop("verification_warning", None)

        re_verification = _verify_one(client, paper.abstract, new_summary, paper.title)
        entry_copy["summary_verification"] = {"status":"passed", "attempts":2,
                                                "initial_issues":verification["issues"]}
        if not re_verification["passed"]:
            # Do not publish the same unsupported interpretation with a warning.
            entry_copy["summary"] = abstract_excerpt(paper.abstract, limits)
            entry_copy["summary_source"] = "abstract_sentences"
            entry_copy["summary_verification"]["status"] = "source_sentence_fallback"
            entry_copy["summary_verification"]["rewrite_issues"] = re_verification["issues"]
            logger.warning("Re-verification failed for %s; using complete abstract sentences", entry["arxiv_id"])
        else:
            logger.info("Re-verification passed for %s", entry["arxiv_id"])

        results.append(entry_copy)

    return results
