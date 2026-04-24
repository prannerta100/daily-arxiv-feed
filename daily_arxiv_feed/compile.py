import logging
import subprocess
import re
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

LATEX_SPECIAL = {
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


def _latex_escape(text: str) -> str:
    """Escape LaTeX special characters."""
    if not isinstance(text, str):
        text = str(text)
    # Escape backslash first to avoid double-escaping
    text = text.replace('\\', r'\textbackslash{}')
    for char, replacement in LATEX_SPECIAL.items():
        if char != '\\':
            text = text.replace(char, replacement)
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

    template = env.get_template("neurips_digest.tex")

    # Pre-process: join authors into a string
    for s in summaries:
        s["authors_str"] = ", ".join(s["authors"])

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
        ["tectonic", str(tex_path)],
        capture_output=True,
        text=True,
        cwd=str(output_dir),
    )
    if result.returncode != 0:
        logger.error("tectonic failed:\n%s", result.stderr)
        raise RuntimeError(f"tectonic compilation failed: {result.stderr}")

    if not pdf_path.exists():
        raise RuntimeError(f"PDF not found at {pdf_path} after compilation")

    return pdf_path


def get_page_count(pdf_path: Path) -> int:
    """Extract page count from PDF using strings command.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Number of pages in the PDF.
    """
    result = subprocess.run(
        ["strings", str(pdf_path)],
        capture_output=True,
        text=True,
    )
    matches = re.findall(r"/Count\s+(\d+)", result.stdout)
    if matches:
        return max(int(m) for m in matches)
    count = result.stdout.count("/Page")
    return max(count - 1, 1)
