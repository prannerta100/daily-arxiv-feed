import logging
import re
import subprocess
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = PROJECT_ROOT / "templates"

CATEGORY_NAMES = {
    1: "Multilingual NLP",
    2: "Multilingual Speech Processing",
    3: "Foundation Model Theory",
    4: "LLM Evaluation",
}

_LATEX_SPECIAL_RE = re.compile(r'([&%$#_{}~^\\])')
_LATEX_SPECIAL_MAP = {
    '&': r'\&',
    '%': r'\%',
    '$': r'\$',
    '#': r'\#',
    '_': r'\_',
    '{': r'\{',
    '}': r'\}',
    '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}',
    '\\': r'\textbackslash{}',
}

_UNICODE_NORMALIZE = {
    '‘': '`',       # left single quote
    '’': "'",       # right single quote / apostrophe
    '“': '``',      # left double quote
    '”': "''",      # right double quote
    '–': '--',      # en-dash
    '—': '---',     # em-dash
    '…': '...',     # ellipsis
    ' ': '~',       # non-breaking space
}
_UNICODE_NORMALIZE_RE = re.compile('[' + ''.join(_UNICODE_NORMALIZE.keys()) + ']')


def _normalize_unicode(text: str) -> str:
    return _UNICODE_NORMALIZE_RE.sub(lambda m: _UNICODE_NORMALIZE[m.group()], text)


def _latex_escape(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = _normalize_unicode(text)
    return _LATEX_SPECIAL_RE.sub(lambda m: _LATEX_SPECIAL_MAP[m.group(1)], text)


_LATEX_CMD_RE = re.compile(r'\\[a-zA-Z]+\{[^}]*\}|\\["\'^`~uvHtcdb]\{[^}]*\}|\\["\'^`~uvHtcdb][a-zA-Z]')


def _latex_escape_preserving_commands(text: str) -> str:
    """Escape special chars but preserve existing LaTeX commands like \\"o, \\'{e}."""
    if not isinstance(text, str):
        text = str(text)
    parts = _LATEX_CMD_RE.split(text)
    commands = _LATEX_CMD_RE.findall(text)
    result = []
    for i, part in enumerate(parts):
        result.append(_latex_escape(part))
        if i < len(commands):
            result.append(commands[i])
    return "".join(result)


def _latex_passthrough(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text


def render_latex(summaries: list[dict], date: str, overview: str = "") -> str:
    """Render summaries into LaTeX using the neurips_digest template.

    Args:
        summaries: List of paper summary dicts with arxiv_id, title, authors,
                   url, categories, and summary fields.
        date: Date string for the digest header.
        overview: Optional field overview text.

    Returns:
        LaTeX source code as a string.
    """
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        block_start_string=r"\BLOCK{",
        block_end_string="}",
        variable_start_string=r"\VAR{",
        variable_end_string="}",
        comment_start_string=r"\#{",
        comment_end_string="}",
    )
    env.filters["e"] = _latex_escape
    env.filters["raw"] = _latex_passthrough
    env.filters["arxiv"] = _latex_escape_preserving_commands

    template = env.get_template("neurips_digest.tex")

    summaries = [dict(s, authors_str=", ".join(s["authors"])) for s in summaries]

    # Group papers by primary category
    grouped: dict[int, list[dict]] = {}
    for s in summaries:
        primary_cat = s["categories"][0] if s["categories"] else 0
        grouped.setdefault(primary_cat, []).append(s)

    sections = []
    for cat_id in sorted(grouped.keys()):
        sections.append({
            "name": CATEGORY_NAMES.get(cat_id, "Other"),
            "papers": grouped[cat_id],
        })

    return template.render(
        date=date,
        paper_count=len(summaries),
        sections=sections,
        overview=overview,
    )


def compile_pdf(tex_content: str, output_dir: Path, filename: str) -> Path:
    """Compile LaTeX to PDF using tectonic.

    Args:
        tex_content: LaTeX source code string.
        output_dir: Directory to write .tex and .pdf files.
        filename: Base filename (without extension).

    Returns:
        Path to the generated PDF.

    Raises:
        RuntimeError: If compilation fails or PDF is not generated.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / f"{filename}.tex"
    pdf_path = output_dir / f"{filename}.pdf"

    tex_path.write_text(tex_content, encoding="utf-8")

    result = subprocess.run(
        ["tectonic", str(tex_path.resolve())],
        capture_output=True,
        text=True,
        cwd=str(output_dir.resolve()),
        timeout=300,
    )
    if result.returncode != 0:
        logger.error("tectonic failed:\n%s", result.stderr)
        raise RuntimeError(f"tectonic compilation failed: {result.stderr}")

    if not pdf_path.exists():
        raise RuntimeError(f"PDF not found at {pdf_path} after compilation")

    return pdf_path


def get_page_count(pdf_path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode == 0:
        match = re.search(r"Pages:\s+(\d+)", result.stdout)
        if match:
            return int(match.group(1))
    logger.warning("pdfinfo failed, estimating page count from file size")
    size_kb = pdf_path.stat().st_size / 1024
    return max(1, int(size_kb / 6))
