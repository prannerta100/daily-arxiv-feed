"""Shared prose format, hard length limits, and source-sentence fallback."""
import math
import re

DEFAULT_MAX_WORDS = 130
DEFAULT_MAX_CHARS = 1000
MAX_GROWTH = 1.15
LEGACY_FIELDS = ("one_line_takeaway", "key_contribution", "method", "most_important_result")

PROSE_RULES = """Write a compressed abstract: one coherent paragraph of complete,
connected sentences. State the problem or question, what the authors did, and the
main finding or limitation, as space permits. Aim for 80-110 words in 3-5 sentences,
staying below the hard limits. Do not fill the available space just because it exists.
Use only information supported by the supplied abstract. Retain important scope,
comparisons and qualifications; distinguish percentage points from relative gains.
Select the most informative claims rather than trying to include every detail.
You may reuse or lightly edit clear sentences from the abstract. If a mechanism
is unclear, omit it rather than inventing an explanation or copying opaque jargon.
Use explicit subjects and verbs. Explain an essential unfamiliar term in ordinary
words or omit it. Avoid unexplained acronym chains, noun piles, slash/plus/arrow
shorthand, bullet points, headings, numbered lists, and parenthetical score lists.
Avoid introducing abbreviations for concepts already expressed clearly in words:
use "speech recognition" rather than ASR and "this harm" rather than a new TFGBV acronym.
Describe what a named mechanism does, rather than merely calling it a new notion
or framework. Replace lists of formal properties with their practical meaning;
if the abstract does not explain the mechanism, describe its stated behavior
without repeating an unexplained method name. Use at most TWO quantitative performance comparisons, usually just
one. Do not enumerate model-by-model scores, every metric, or secondary release
and documentation details. For example, say that adapters reduced text loss
without improving comprehension, then report the main gain from fuller training;
do not list every model size and its score change.
Do not add a separate takeaway, promotional significance claim, or repeated conclusion.
Shorten by removing secondary details and repetition, never articles or connectives.
Return JSON with exactly one field: {"excerpt": "the paragraph"}."""


def summary_text(summary: dict) -> str:
    if isinstance(summary.get("excerpt"), str):
        return " ".join(summary["excerpt"].split())
    return " ".join(str(summary.get(k, "")).strip() for k in LEGACY_FIELDS).strip()


def summary_limits(previous: dict | None = None) -> dict:
    """Bound new work and keep reruns within 15% of their original display text."""
    limits = {"max_words": DEFAULT_MAX_WORDS, "max_chars": DEFAULT_MAX_CHARS}
    if previous:
        if previous.get("summary_limits"):
            return {k: min(v, int(previous["summary_limits"][k])) for k, v in limits.items()}
        text = summary_text(previous.get("summary", previous))
        if text:
            limits["max_words"] = min(limits["max_words"], math.floor(len(text.split()) * MAX_GROWTH))
            limits["max_chars"] = min(limits["max_chars"], math.floor(len(text) * MAX_GROWTH))
    return limits


def fits_budget(summary: dict, limits: dict) -> bool:
    text = summary_text(summary)
    return bool(text) and len(text.split()) <= limits["max_words"] and len(text) <= limits["max_chars"]


def budget_instruction(limits: dict) -> str:
    return (f"Hard limits for the ENTIRE paragraph: at most {limits['max_words']} words "
            f"AND {limits['max_chars']} characters including spaces. Aim below both limits. "
            "A shorter clear paragraph is preferable to cramming in more facts.")


def clean_abstract(abstract: str) -> str:
    # RSS includes an announcement header; it is not part of the abstract.
    if abstract.lstrip().startswith("arXiv:") and "Abstract:" in abstract:
        abstract = abstract.split("Abstract:", 1)[1]
    abstract = abstract.replace(r"\%", "%")
    abstract = re.sub(r"\\([a-zA-Z]+)(?:\{\})?", r"\1", abstract)
    return " ".join(abstract.split())


def abstract_excerpt(abstract: str, limits: dict) -> dict:
    """Select whole source sentences in source order; never clip a sentence."""
    text = clean_abstract(abstract)
    # Protect common abbreviations from being mistaken for sentence boundaries.
    protected = re.sub(r"\b(?:e\.g\.|i\.e\.|et al\.|Fig\.|Dr\.)",
                       lambda m: m.group().replace(".", "\u2024"), text)
    sentences = [s.replace("\u2024", ".") for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", protected)]
    candidates = [i for i, s in enumerate(sentences) if s.endswith((".", "!", "?"))
                  and not re.search(r"https?://|www\.", s)]
    # Keep the opening context and a finding near the end before filling details.
    priority = candidates[:1] + candidates[-1:] + candidates[1:-1]
    chosen = []
    for i in dict.fromkeys(priority):
        proposed = sorted(chosen + [i])
        candidate = {"excerpt": " ".join(sentences[j] for j in proposed)}
        if fits_budget(candidate, limits):
            chosen = proposed
    if not chosen:
        # An explicit failure is preferable to an invented or cut-off summary.
        raise ValueError("No complete abstract sentence fits the excerpt budget")
    return {"excerpt": " ".join(sentences[i] for i in chosen)}
