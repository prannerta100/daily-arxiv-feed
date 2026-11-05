import logging

from openai import OpenAI

from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.llm import chat, parse_json_response
from daily_arxiv_feed.excerpts import clean_abstract

logger = logging.getLogger(__name__)

MAX_OVERVIEW_WORDS = 100
MAX_OVERVIEW_CHARS = 700

OVERVIEW_SYSTEM_PROMPT = """Describe the two or three dominant research themes in
today's supplied paper titles and abstract snippets. Write 2-3 connected, complete
sentences in plain English, at most 100 words AND 700 characters including spaces.
Explain what researchers are working on, rather than listing acronyms or titles.
No slash-separated topic chains, plus-sign shorthand, parenthetical name lists,
or unexplained specialist abbreviations. Select fewer themes if necessary.
Do not estimate paper counts or percentages, invent findings from truncated
abstracts, or claim changes relative to earlier days you have not been given.

Respond with JSON: {"overview": "your overview text here"}"""


def generate_overview(client: OpenAI, all_papers: list[Paper]) -> str:
    lines = []
    for p in all_papers:
        lines.append(f"- {p.title}: {clean_abstract(p.abstract)[:300]}")
    user_prompt = f"Today's {len(all_papers)} arxiv papers:\n\n" + "\n".join(lines)

    for attempt in range(2):
        response = chat(client=client, system=OVERVIEW_SYSTEM_PROMPT, user=user_prompt, json_mode=True)
        data = parse_json_response(response)
        text = " ".join(data["overview"].split())
        if len(text.split()) <= MAX_OVERVIEW_WORDS and len(text) <= MAX_OVERVIEW_CHARS:
            return text
        user_prompt += "\nThe previous overview exceeded its limits. Cover fewer themes in complete sentences."
    logger.warning("Overview exceeded its length limits twice; omitting it")
    return ""
