import json
import logging

from openai import OpenAI

from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.llm import chat
from daily_arxiv_feed.summarize import SUMMARIZE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

VERIFY_SYSTEM_PROMPT = """You are verifying a summary of an arxiv paper against its original abstract.

Check for:
- Factual inconsistencies between the summary and abstract
- Claims in the summary not supported by the abstract
- Misrepresented results or conclusions
- Exaggerated or understated findings

Respond with JSON:
{"passed": true/false, "issues": ["list of specific issues found, empty if passed"]}"""


def _verify_one(client: OpenAI, abstract: str, summary: dict) -> dict:
    user_prompt = f"""Original abstract:
{abstract}

Summary to verify:
- Takeaway: {summary['one_line_takeaway']}
- Key contribution: {summary['key_contribution']}
- Method: {summary['method']}
- Most important result: {summary['most_important_result']}"""

    response = chat(client=client, system=VERIFY_SYSTEM_PROMPT, user=user_prompt, json_mode=True)
    return json.loads(response)


def _regenerate_summary(client: OpenAI, paper: Paper, feedback: list[str]) -> dict:
    regen_prompt = f"""Regenerate a summary for this paper. Your previous summary had these issues:
{chr(10).join('- ' + issue for issue in feedback)}

Fix these issues. Be accurate to the abstract.

Title: {paper.title}
Abstract: {paper.abstract}

Respond with JSON:
{{"one_line_takeaway": "...", "key_contribution": "...", "method": "...", "most_important_result": "..."}}"""

    response = chat(client=client, system=SUMMARIZE_SYSTEM_PROMPT, user=regen_prompt, json_mode=True)
    return json.loads(response)


def verify_summaries(client: OpenAI, papers: list[Paper], summaries: list[dict]) -> list[dict]:
    paper_map = {p.arxiv_id: p for p in papers}
    results = []

    for entry in summaries:
        paper = paper_map.get(entry["arxiv_id"])
        if not paper:
            results.append(entry)
            continue

        verification = _verify_one(client, paper.abstract, entry["summary"])

        if verification["passed"]:
            results.append(entry)
            logger.info("Verified OK: %s", entry["arxiv_id"])
            continue

        logger.warning("Verification failed for %s: %s", entry["arxiv_id"], verification["issues"])
        new_summary = _regenerate_summary(client, paper, verification["issues"])
        entry_copy = {**entry, "summary": new_summary}

        re_verification = _verify_one(client, paper.abstract, new_summary)
        if not re_verification["passed"]:
            entry_copy["verification_warning"] = "Summary may contain inaccuracies: " + "; ".join(re_verification["issues"])
            logger.warning("Re-verification still failed for %s, adding warning", entry["arxiv_id"])
        else:
            logger.info("Re-verification passed for %s", entry["arxiv_id"])

        results.append(entry_copy)

    return results
