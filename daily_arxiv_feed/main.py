import json
import logging
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from daily_arxiv_feed.llm import get_client, chat, parse_json_response
from daily_arxiv_feed.fetch import fetch_papers
from daily_arxiv_feed.filter import filter_papers
from daily_arxiv_feed.summarize import summarize_papers
from daily_arxiv_feed.verify import verify_summaries
from daily_arxiv_feed.overview import generate_overview
from daily_arxiv_feed.compile import render_latex, compile_pdf, get_page_count

logger = logging.getLogger(__name__)

MAX_PAGES = 3
MAX_COMPRESS_ITERATIONS = 3

COMPRESS_SYSTEM_PROMPT = """You are compressing paper summaries to fit a strict 2-page PDF limit. The current digest is too long. Rewrite ALL summaries to be shorter while preserving the key insight and most important result for each paper. Use tighter prose — fewer words, same information density. Do not drop any paper.

Respond with JSON: {"summaries": [{"arxiv_id": "...", "one_line_takeaway": "...", "key_contribution": "...", "method": "...", "most_important_result": "..."}]}"""


def _write_staging(staging_dir: Path, filename: str, data) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / filename).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _compress_summaries(client, summaries: list[dict]) -> list[dict]:
    lines = []
    for s in summaries:
        lines.append(f"arxiv_id: {s['arxiv_id']}")
        lines.append(f"Title: {s['title']}")
        for k, v in s["summary"].items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    user_prompt = "\n".join(lines)

    for attempt in range(2):
        try:
            response = chat(client=client, system=COMPRESS_SYSTEM_PROMPT, user=user_prompt, json_mode=True)
            compressed = parse_json_response(response)
            break
        except Exception:
            if attempt == 1:
                logger.warning("Compression failed twice, keeping original summaries")
                return summaries
            logger.warning("Compression JSON parse failed, retrying")

    summary_map = {c["arxiv_id"]: c for c in compressed["summaries"]}
    for s in summaries:
        if s["arxiv_id"] in summary_map:
            c = summary_map[s["arxiv_id"]]
            s["summary"] = {
                "one_line_takeaway": c["one_line_takeaway"],
                "key_contribution": c["key_contribution"],
                "method": c["method"],
                "most_important_result": c["most_important_result"],
            }
    return summaries


def _push_pdf(pdf_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    pdf_path = pdf_path.resolve()
    tex_path = pdf_path.with_suffix(".tex")
    files_to_push = [str(pdf_path.relative_to(repo_root))]
    if tex_path.exists():
        files_to_push.append(str(tex_path.relative_to(repo_root)))

    date = pdf_path.stem
    try:
        subprocess.run(["git", "add"] + files_to_push, cwd=repo_root, check=True)
        subprocess.run(
            ["git", "commit", "-m", f"digest: {date}"],
            cwd=repo_root, check=True,
        )
        subprocess.run(["git", "push"], cwd=repo_root, check=True)
        logger.info("Pushed %s to remote", ", ".join(files_to_push))
    except subprocess.CalledProcessError as e:
        logger.warning("Git push failed: %s", e)


def run_pipeline(output_dir: Path | None = None, date: str | None = None) -> Path | None:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "output"
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    staging_dir = output_dir / "staging" / date

    client = get_client()

    logger.info("=== Stage 1: Fetch ===")
    papers = fetch_papers()
    if not papers:
        logger.warning("No papers fetched. Exiting.")
        return None
    _write_staging(staging_dir, "01_fetched.json", [asdict(p) for p in papers])
    logger.info("Fetched %d papers", len(papers))

    logger.info("=== Stage 2: Filter ===")
    selected, all_decisions = filter_papers(client, papers)
    _write_staging(staging_dir, "02_filtered.json", all_decisions)
    if not selected:
        logger.warning("No papers selected. Exiting.")
        return None
    logger.info("Selected %d papers", len(selected))

    logger.info("=== Stage 2.5: Field Overview ===")
    try:
        overview = generate_overview(client, papers)
    except Exception:
        logger.exception("Overview generation failed, continuing without it")
        overview = ""

    logger.info("=== Stage 3: Summarize ===")
    summaries = summarize_papers(client, selected, all_decisions)
    _write_staging(staging_dir, "03_summaries.json", summaries)

    logger.info("=== Stage 4: Verify ===")
    verified = verify_summaries(client, selected, summaries)
    _write_staging(staging_dir, "04_verified.json", verified)

    logger.info("=== Stage 5: Compile ===")
    tex = render_latex(verified, date, overview=overview)
    pdf_path = compile_pdf(tex, output_dir, date)
    pages = get_page_count(pdf_path)
    logger.info("Initial PDF: %d pages", pages)

    for i in range(MAX_COMPRESS_ITERATIONS):
        if pages <= MAX_PAGES:
            break
        logger.info("PDF is %d pages (max %d), compressing (iteration %d)...", pages, MAX_PAGES, i + 1)
        verified = _compress_summaries(client, verified)
        _write_staging(staging_dir, "04_verified.json", verified)
        tex = render_latex(verified, date, overview=overview)
        pdf_path = compile_pdf(tex, output_dir, date)
        pages = get_page_count(pdf_path)
        logger.info("After compression: %d pages", pages)

    if pages > MAX_PAGES:
        logger.warning("PDF still %d pages after %d compression attempts", pages, MAX_COMPRESS_ITERATIONS)

    logger.info("=== Stage 6: Push to remote ===")
    _push_pdf(pdf_path)

    logger.info("Done! PDF at %s", pdf_path)
    return pdf_path


def main():
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"{today}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    try:
        result = run_pipeline()
        if result:
            logging.info("Pipeline complete: %s", result)
        else:
            logging.info("Pipeline complete: no PDF generated today")
    except Exception:
        logging.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
