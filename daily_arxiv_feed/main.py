import argparse
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from daily_arxiv_feed.llm import get_client
from daily_arxiv_feed.fetch import Paper, fetch_papers
from daily_arxiv_feed.filter import filter_papers
from daily_arxiv_feed.summarize import generate_excerpt, summarize_papers
from daily_arxiv_feed.excerpts import fits_budget, summary_limits, summary_text
from daily_arxiv_feed.verify import verify_summaries
from daily_arxiv_feed.overview import generate_overview
from daily_arxiv_feed.compile import render_latex, compile_pdf, get_page_count

logger = logging.getLogger(__name__)

MAX_PAGES = 3
MAX_COMPRESS_ITERATIONS = 3


def _write_staging(staging_dir: Path, filename: str, data) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / filename).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _compress_summaries(client, summaries: list[dict], papers: list[Paper]) -> list[dict]:
    """Shorten from source abstracts, keeping complete prose and explicit limits."""
    paper_map = {p.arxiv_id: p for p in papers}
    results = []
    for s in summaries:
        current = summary_text(s["summary"])
        original_limits = s.get("summary_limits") or summary_limits()
        limits = {"max_words": min(original_limits["max_words"], max(70, int(len(current.split()) * .85))),
                  "max_chars": min(original_limits["max_chars"], max(450, int(len(current) * .85)))}
        if limits["max_words"] >= len(current.split()) and limits["max_chars"] >= len(current):
            results.append(s)
            continue
        summary, source = generate_excerpt(client, paper_map[s["arxiv_id"]], limits,
            ["The PDF exceeds its page budget. Select fewer claims; preserve full sentences and readability."])
        results.append({**s, "summary": summary, "summary_limits": limits, "summary_source": source})
    return results


def _compile_digest(client, papers, summaries, output_dir, date, overview):
    staging_dir = output_dir / "staging" / date
    for iteration in range(MAX_COMPRESS_ITERATIONS + 1):
        tex = render_latex(summaries, date, overview=overview)
        pdf_path = compile_pdf(tex, output_dir, date)
        pages = get_page_count(pdf_path)
        logger.info("PDF: %d pages (limit %d)", pages, MAX_PAGES)
        if pages <= MAX_PAGES or iteration == MAX_COMPRESS_ITERATIONS:
            break
        summaries = _compress_summaries(client, summaries, papers)
        # Compression changes claims too; verify the text readers will see.
        summaries = verify_summaries(client, papers, summaries)
        _write_staging(staging_dir, "04_verified.json", summaries)
    if pages > MAX_PAGES:
        logger.warning("PDF still %d pages after %d compression attempts", pages, MAX_COMPRESS_ITERATIONS)
    return pdf_path, summaries


def _push_pdf(pdf_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    pdf_path = pdf_path.resolve()
    tex_path = pdf_path.with_suffix(".tex")
    if not pdf_path.is_relative_to(repo_root):
        logger.warning("PDF %s is outside the repo; skipping git push", pdf_path)
        return
    files_to_push = [str(pdf_path.relative_to(repo_root))]
    if tex_path.exists():
        files_to_push.append(str(tex_path.relative_to(repo_root)))

    date = pdf_path.stem
    label = "quantum digest" if pdf_path.parent.name=="quantum" else "digest"
    try:
        subprocess.run(["git", "add"] + files_to_push, cwd=repo_root, check=True)
        subprocess.run(
            ["git", "commit", "-m", f"{label}: {date}"],
            cwd=repo_root, check=True,
        )
        subprocess.run(["git", "push"], cwd=repo_root, check=True)
        logger.info("Pushed %s to remote", ", ".join(files_to_push))
    except subprocess.CalledProcessError as e:
        logger.warning("Git push failed: %s", e)


def run_pipeline(output_dir: Path | None = None, date: str | None = None, *, publish: bool = True) -> Path | None:
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
    _write_staging(staging_dir, "05_overview.json", {"overview": overview})

    logger.info("=== Stage 3: Summarize ===")
    previous_path = staging_dir / "04_verified.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    summaries = summarize_papers(client, selected, all_decisions, previous=previous)
    _write_staging(staging_dir, "03_summaries.json", summaries)

    logger.info("=== Stage 4: Verify ===")
    verified = verify_summaries(client, selected, summaries)
    _write_staging(staging_dir, "04_verified.json", verified)

    logger.info("=== Stage 5: Compile ===")
    pdf_path, verified = _compile_digest(client, selected, verified, output_dir, date, overview)

    if publish:
        logger.info("=== Stage 6: Push to remote ===")
        _push_pdf(pdf_path)

    logger.info("Done! PDF at %s", pdf_path)
    return pdf_path


def run_daily_digests(output_dir: Path | None = None, date: str | None = None, *,
                      publish: bool = True, open_pdfs: bool = False) -> dict[str, Path | None]:
    """Complete NLP before starting any quantum fetch/filter work."""
    from daily_arxiv_feed.quantum import run_quantum_pipeline
    output_dir = output_dir or Path(__file__).resolve().parent.parent / "output"
    date = date or datetime.now().strftime("%Y-%m-%d")
    logger.info("=== Daily sequence 1/2: NLP ===")
    nlp_pdf = run_pipeline(output_dir=output_dir,date=date,publish=publish)
    if open_pdfs and nlp_pdf:
        subprocess.run(["open",str(nlp_pdf)],check=True)
    # No executor, background process, or concurrent API requests: the first call
    # must return (including compilation) before this second pipeline can start.
    logger.info("=== Daily sequence 2/2: quantum ===")
    quantum_pdf = run_quantum_pipeline(output_dir=output_dir/"quantum",date=date,publish=publish)
    if open_pdfs and quantum_pdf:
        subprocess.run(["open",str(quantum_pdf)],check=True)
    return {"nlp":nlp_pdf,"quantum":quantum_pdf}


def rebuild_digest(date: str, output_dir: Path | None = None) -> Path:
    """Rebuild cached selections locally; this path never commits or pushes."""
    output_dir = output_dir or Path(__file__).resolve().parent.parent / "output"
    staging = output_dir / "staging" / date
    previous = json.loads((staging / "04_verified.json").read_text())
    all_papers = [Paper(**p) for p in json.loads((staging / "01_fetched.json").read_text())]
    paper_map = {p.arxiv_id: p for p in all_papers}
    selected = [paper_map[s["arxiv_id"]] for s in previous]
    decisions = [{"arxiv_id":s["arxiv_id"], "categories":s["categories"], "reasoning":"Previously selected"}
                 for s in previous]
    backup = staging / "readability-before"
    backup.mkdir(exist_ok=True)
    for path in [staging / "03_summaries.json", staging / "04_verified.json",
                 output_dir / f"{date}.tex", output_dir / f"{date}.pdf"]:
        if path.exists() and not (backup / path.name).exists():
            shutil.copy2(path, backup / path.name)
    client = get_client()
    overview_file = staging / "05_overview.json"
    if overview_file.exists():
        overview = json.loads(overview_file.read_text())["overview"]
    else:
        overview = generate_overview(client, all_papers)
        _write_staging(staging, "05_overview.json", {"overview":overview})
    summaries = summarize_papers(client, selected, decisions, previous=previous)
    _write_staging(staging, "03_summaries.json", summaries)
    verified = verify_summaries(client, selected, summaries)
    _write_staging(staging, "04_verified.json", verified)
    pdf, verified = _compile_digest(client, selected, verified, output_dir, date, overview)
    comparison = []
    baseline = json.loads((backup / "04_verified.json").read_text())
    for old,new in zip(baseline,verified,strict=True):
        assert old["arxiv_id"] == new["arxiv_id"]
        before, after = summary_text(old["summary"]), summary_text(new["summary"])
        assert fits_budget(new["summary"],summary_limits(old))
        comparison.append({"arxiv_id":new["arxiv_id"], "title":new["title"],
            "before":before,"after":after,"before_words":len(before.split()),"after_words":len(after.split()),
            "before_chars":len(before),"after_chars":len(after),"source":new.get("summary_source"),
            "limits":new["summary_limits"]})
    _write_staging(staging,"06_readability_comparison.json",comparison)
    return pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", metavar="YYYY-MM-DD", help="Rewrite cached digest locally without fetching, committing or pushing")
    parser.add_argument("--digest",choices=["all","nlp","quantum"],
                        help="Default: NLP then quantum; --rebuild defaults to NLP for compatibility")
    parser.add_argument("--date",metavar="YYYY-MM-DD",help="Output date for a fresh run")
    parser.add_argument("--no-push",action="store_true",help="Generate local PDFs without committing or pushing")
    parser.add_argument("--open",action="store_true",dest="open_pdfs",help="Open each PDF as it completes")
    args = parser.parse_args()
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
        digest = args.digest or ("nlp" if args.rebuild else "all")
        if args.rebuild:
            if digest=="quantum":
                from daily_arxiv_feed.quantum import rebuild_quantum_digest
                result = rebuild_quantum_digest(args.rebuild)
            elif digest=="all":
                from daily_arxiv_feed.quantum import rebuild_quantum_digest
                result = {"nlp":rebuild_digest(args.rebuild)}
                result["quantum"] = rebuild_quantum_digest(args.rebuild)
            else:
                result = rebuild_digest(args.rebuild)
        elif digest=="all":
            result = run_daily_digests(date=args.date,publish=not args.no_push,open_pdfs=args.open_pdfs)
        elif digest=="quantum":
            from daily_arxiv_feed.quantum import run_quantum_pipeline
            result = run_quantum_pipeline(date=args.date,publish=not args.no_push)
        else:
            result = run_pipeline(date=args.date,publish=not args.no_push)
        if args.open_pdfs and digest!="all" and result:
            subprocess.run(["open",str(result)],check=True)
        if result:
            logging.info("Pipeline complete: %s", result)
        else:
            logging.info("Pipeline complete: no PDF generated today")
    except Exception:
        logging.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
