import json
import os
import logging
import re
import time
import random

from openai import OpenAI

logger = logging.getLogger(__name__)

MODEL = "gpt-5.2"
BASE_URL = "https://llm-proxy.us-east-2.int.infra.intelligence.webex.com/openai/v1"
RETRYABLE_STATUSES = {429, 500, 502, 503}
MAX_RETRIES = 5


def get_client() -> OpenAI:
    token = os.environ.get("WEBEX_TOKEN")
    if not token:
        raise ValueError("WEBEX_TOKEN environment variable is not set")
    return OpenAI(
        base_url=BASE_URL,
        api_key=token,
        default_headers={"x-cisco-app": "daily-arxiv-feed"},
    )


def chat(
    client: OpenAI,
    system: str,
    user: str,
    json_mode: bool = False,
    temperature: float = 0.2,
) -> str:
    kwargs: dict = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status in RETRYABLE_STATUSES and attempt < MAX_RETRIES:
                wait = min(2 ** attempt + random.random(), 60)
                logger.warning("LLM call failed (status %s), retry %d/%d in %.1fs", status, attempt + 1, MAX_RETRIES, wait)
                time.sleep(wait)
            else:
                raise


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _extract_json_text(text: str) -> str:
    text = text.strip()
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()
    start = text.index("{")
    end = text.rindex("}") + 1
    return text[start:end]


def parse_json_response(text: str) -> dict:
    raw = _extract_json_text(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Attempt repair: fix unescaped control characters inside string values
        repaired = re.sub(r'[\x00-\x1f]+', ' ', raw)
        return json.loads(repaired)
