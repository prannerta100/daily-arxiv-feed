import logging

from openai import OpenAI

from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.llm import chat, parse_json_response

logger = logging.getLogger(__name__)

OVERVIEW_SYSTEM_PROMPT = """You are analyzing today's arxiv submissions to identify what's hot in the field. Given a list of ALL paper titles and abstracts from today, write a brief (2-4 sentences) overview of the dominant themes and trends you see. What topics are getting the most attention? Any surprising clusters or emerging directions?

Be specific — name the actual topics and rough counts (e.g., "At least 8 papers on..." or "A noticeable cluster around..."). Don't be generic.

Respond with JSON: {"overview": "your overview text here"}"""


def generate_overview(client: OpenAI, all_papers: list[Paper]) -> str:
    lines = []
    for p in all_papers:
        lines.append(f"- {p.title}: {p.abstract[:200]}")
    user_prompt = f"Today's {len(all_papers)} arxiv papers:\n\n" + "\n".join(lines)

    response = chat(client=client, system=OVERVIEW_SYSTEM_PROMPT, user=user_prompt, json_mode=True)
    data = parse_json_response(response)
    return data["overview"]
