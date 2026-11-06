"""Short, source-checked reading aids shared by both digests."""
import hashlib
import json
import logging

from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.llm import chat, parse_json_response

logger = logging.getLogger(__name__)
MAX_WORDS = 16
MAX_CHARS = 110

GENERATE_PROMPT = f"""Write one short, readable sentence per paper for an italic
one-line summary above its abstract. State the main contribution or finding in
plain language with a subject and verb, not a slogan or acronym/noun pile.
Avoid unexplained acronyms and opaque method names: describe what the method does
instead. Prefer a concrete subject to an empty 'They'. For example, write
'Learned models reduce the measurements needed for electronic-structure simulations.'
rather than 'They use ML shadows to remove QC-AFQMC bottlenecks'.
Use at most {MAX_WORDS} words AND {MAX_CHARS} characters including spaces.
Aim for 10-12 words and under 90 characters to leave room below both limits.
Use only the supplied title and abstract. Preserve essential qualifications;
do not turn simulation or measured-data analysis into a hardware demonstration,
or a benchmark improvement into a general advantage. Omit numbers if their scope
cannot fit. Avoid hype, bullets, labels, TeX, and line breaks. These are reading
aids; the abstract below provides the details. Return JSON with exactly one row
per supplied ID: {{"papers":[{{"arxiv_id":"...","one_line_takeaway":"..."}}]}}.
"""

VERIFY_PROMPT = """Check each proposed one-line summary against its supplied
title and original abstract. Every claim must be supported and appropriately
qualified. Distinguish hardware experiments, measured-data analysis, simulation
and theory; reject overstated comparisons and invented explanations. Check that
it is a readable sentence with a subject and verb rather than compressed jargon.
Selective coverage is intentional: do not demand every detail or secondary result.
Return exactly one row for every supplied ID as JSON:
{"papers":[{"arxiv_id":"...","passed":true,"issues":[]}]}.
"""


def _fingerprint(paper: Paper) -> str:
    return hashlib.sha256(json.dumps(
        [paper.title, paper.abstract, GENERATE_PROMPT, VERIFY_PROMPT],
        ensure_ascii=False,
    ).encode()).hexdigest()


def _valid(text) -> bool:
    return (isinstance(text, str) and bool(text.strip())
            and "\n" not in text and "\r" not in text
            and len(text) <= MAX_CHARS and len(text.split()) <= MAX_WORDS
            and text.endswith((".", "!", "?")))


def _rows(response: str, expected: set[str]) -> dict:
    rows = parse_json_response(response)["papers"]
    by_id = {row["arxiv_id"]: row for row in rows}
    if len(rows) != len(by_id) or set(by_id) != expected:
        raise ValueError("One-line response must contain each supplied ID exactly once")
    return by_id


def add_takeaways(client, papers: list[Paper], entries: list[dict]) -> list[dict]:
    """Add verified lines without changing excerpts, abstracts or their budgets."""
    paper_map = {paper.arxiv_id: paper for paper in papers}
    results = [dict(entry) for entry in entries]
    pending = []
    for entry in results:
        paper = paper_map[entry["arxiv_id"]]
        audit = entry.get("takeaway_verification", {})
        if not (_valid(entry.get("one_line_takeaway"))
                and audit.get("status") == "passed"
                and audit.get("source_fingerprint") == _fingerprint(paper)):
            pending.append(entry)
    feedback = {}
    for attempt in range(1, 4):
        if not pending:
            return results
        inputs = [{"arxiv_id":entry["arxiv_id"],
                   "title":paper_map[entry["arxiv_id"]].title,
                   "abstract":paper_map[entry["arxiv_id"]].abstract,
                   "previous_issues":feedback.get(entry["arxiv_id"], [])}
                  for entry in pending]
        expected = {entry["arxiv_id"] for entry in pending}
        try:
            generated = _rows(chat(client=client, system=GENERATE_PROMPT,
                                  user=json.dumps(inputs, ensure_ascii=False), json_mode=True), expected)
            valid = []
            for item in inputs:
                line = generated[item["arxiv_id"]].get("one_line_takeaway")
                if _valid(line):
                    valid.append({**item, "one_line_takeaway":line})
                else:
                    feedback[item["arxiv_id"]] = [
                        f"Rejected line: {line!r}. Use a complete sentence ending in punctuation, "
                        f"at most {MAX_WORDS} words and {MAX_CHARS} characters. "
                        "Shorten to 10-12 words and under 90 characters; omit secondary details."]
                    logger.warning("Rejected one-line format for %s: %r", item["arxiv_id"], line)
            checks = _rows(chat(client=client, system=VERIFY_PROMPT,
                               user=json.dumps(valid, ensure_ascii=False), json_mode=True),
                           {item["arxiv_id"] for item in valid}) if valid else {}
            remaining = []
            for entry in pending:
                aid = entry["arxiv_id"]
                check = checks.get(aid, {})
                if check.get("passed") is True and check.get("issues") == []:
                    entry["one_line_takeaway"] = generated[aid]["one_line_takeaway"]
                    entry["takeaway_verification"] = {
                        "status":"passed", "attempts":attempt,
                        "source_fingerprint":_fingerprint(paper_map[aid]),
                    }
                else:
                    feedback[aid] = check.get("issues") or feedback.get(aid) or ["Verification did not pass."]
                    remaining.append(entry)
            pending = remaining
        except (KeyError, TypeError, ValueError) as error:
            feedback = {entry["arxiv_id"]:[str(error)] for entry in pending}
            logger.warning("Invalid one-line response (attempt %d): %s", attempt, error)
    if pending:
        raise ValueError(f"Could not verify one-line summaries: {feedback}")
    return results
