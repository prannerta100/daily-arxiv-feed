"""Quantum selection and full author abstracts with separate short reading aids."""
import hashlib
import json
import logging
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from daily_arxiv_feed.compile import compile_pdf, get_page_count, render_latex
from daily_arxiv_feed.fetch import fetch_papers
from daily_arxiv_feed.filter import filter_papers
from daily_arxiv_feed.llm import get_client
from daily_arxiv_feed.takeaways import add_takeaways

logger = logging.getLogger(__name__)
QUANTUM_CATEGORIES = ["quant-ph", "cs.LG", "physics.app-ph"]
QUANTUM_SECTIONS = {
    1: "Machine Learning for Quantum Control and Pulse Optimization",
    2: "Machine Learning for Error Correction, Mitigation and Fault Tolerance",
    3: "Notable Quantum Computing Research",
}
MAX_QUANTUM_PAPERS = 8

EXPERIMENT_PREFERENCE = """
The reader has an experimental and implementation-oriented research inclination.
Actively seek working hardware demonstrations, experiments on physical devices,
methods tested on measured device data, and useful implementations with credible
benchmarks. Include calibration, real-time feedback, decoder latency/throughput,
resource overhead, robustness, reproducibility and practical constraints when
supported by the abstract. Useful demonstrated progress need not claim a grand
theoretical breakthrough. Do not discard experimental work simply because it is
not ML: meaningful hardware/control/QEC implementations can enter section 3.
Keep strong theory eligible, especially when it enables experiments, but do not
let elegant theory or speculative scalability crowd out strong practical work.
When suitable candidates exist, aim to include at least two papers with physical
experiments or measured-device-data validation across the digest. This is a soft
balance preference, not a reason to admit weak papers or invent evidence.
Treat hardware experiments, measured-data validation, realistic noisy simulation,
idealized numerical benchmarks and theory as distinct forms of evidence. State
the actual evidence type in each selection reason; distinguish experimental device
data from running the proposed method on hardware. Consider released code only
when explicitly mentioned. Never describe a simulation as an experiment on a
physical device. Concrete implementations and strong realistic simulations are
valuable too; do not equate implementation with a theorem or a proposed design.
"""

FILTER_PROMPT = """Select candidates for a quantum computing research digest.
Prioritize these interests, in order, while allowing major general quantum work:
1. Machine learning for quantum control and pulse optimization: reinforcement
   learning, Bayesian optimization, learned models or other ML used to design
   pulses, calibrate gates, suppress leakage, control qubits or prepare states.
   Hardware constraints, noise robustness, sample efficiency and experiments matter.
2. Machine learning for quantum error correction, error mitigation and fault
   tolerance: learned decoders, syndrome/noise inference, adaptive correction,
   mitigation and practical routes to reliable logical operations. Distinguish
   mitigation from correction and fault tolerance; do not conflate their claims.
3. Genuinely interesting, potentially impactful general quantum computing:
   substantial algorithms, complexity results, quantum hardware, logical qubits,
   codes, architectures or enabling experiments, even when they do not use ML.
This first pass should be inclusive about plausible matches. Exclude classical
ML/control unrelated to quantum computing, generic quantum-inspired optimization,
and routine quantum-ML classifiers on small classical datasets unless their
abstract establishes a substantial quantum-computing advance. Judge evidence in
the supplied abstract, not marketing words or speculation about eventual impact.
Respond with JSON: {"papers": [{"arxiv_id":"...", "selected":true,
"categories":[1], "reasoning":"brief evidence-based selection reason"}]}.
Return only plausible candidates. Omit unrelated papers entirely; do not spend
output tokens listing obvious rejections. An empty papers list is valid.
""" + EXPERIMENT_PREFERENCE

RANK_PROMPT = """Make the final, ordered selection for a quantum computing digest
whose entries will contain the FULL author abstracts, not generated summaries.
Use these primary sections:
1. ML for quantum control and pulse optimization.
2. ML for error correction, mitigation and fault tolerance.
3. Notable general quantum computing research.
Aim for 2-3 strong papers in each ML-focused section and 1-2 strong general
papers when the candidates warrant it; these are preferences, NOT minimum quotas.
Select at most 8 total, and fewer (including none) on a weak day. Do not fill space
with irrelevant papers. Give focused ML work priority, but retain genuine major
general advances. Read the actual abstract for mechanisms, comparisons, physical
constraints, scaling and experimental or theoretical evidence. Do not infer that
simulated results are hardware demonstrations, or that error mitigation gives
fault tolerance. A purely non-ML control/QEC paper belongs in section 3 if notable.
Order selected papers from strongest to weakest. Put their primary section first
in categories; use only category numbers 1, 2 or 3. Reject duplicates and routine
or speculative work. Do not rewrite any abstract.
The word 'learning' or 'learnability' alone does not establish an ML contribution.
Parameter-identifiability theorems, tomography protocols and conventional optimal
control without an actual ML/statistical-learning method belong in section 3,
even when they are useful to calibration. Do not move them into a priority section
to fill its preferred count. It is fine to select only one or zero pulse/control
ML papers when that is all the evidence supports.
Apply the same evidence rule to section 2: it requires an actual ML method,
learned decoder/model, statistical inference or ML-driven code design. Purely
analytical decoding-complexity results, conventional decoders and non-ML code
constructions belong in section 3, regardless of their fault-tolerance relevance.
Respond with JSON: {"papers":[{"arxiv_id":"...", "selected":true,
"categories":[1], "reasoning":"why it earns a place, based on abstract evidence"}]}.
""" + EXPERIMENT_PREFERENCE


def full_abstract(text: str) -> str:
    """Remove only the RSS announcement wrapper, not author text or TeX."""
    return re.sub(r"\A\s*arXiv:[^\n]*\n\s*Abstract:\s*", "", text, count=1).strip()


def rebuild_quantum_digest(date: str, output_dir: Path | None = None) -> Path:
    """Retypeset cached full abstracts without LLM calls, commits or pushes."""
    from daily_arxiv_feed.main import _write_staging
    output_dir = output_dir or Path(__file__).resolve().parent.parent / "output" / "quantum"
    staging = output_dir / "staging" / date
    entries = json.loads((staging/"03_full_abstracts.json").read_text())
    papers = {p["arxiv_id"]:p for p in json.loads((staging/"01_fetched.json").read_text())}
    for entry in entries:
        source = full_abstract(papers[entry["arxiv_id"]]["abstract"])
        if entry["abstract"]!=source or entry["abstract_sha256"]!=hashlib.sha256(source.encode()).hexdigest():
            raise ValueError(f"Cached full abstract differs from source: {entry['arxiv_id']}")
    tex = render_latex(entries,date,category_names=QUANTUM_SECTIONS,template_name="quantum_digest.tex")
    pdf = compile_pdf(tex,output_dir,date)
    manifest = json.loads((staging/"04_manifest.json").read_text())
    manifest.update(pdf=str(pdf),pages=get_page_count(pdf),rendered_full_abstracts=len(entries),
                    selected_papers=len(entries),status="complete")
    _write_staging(staging,"04_manifest.json",manifest)
    return pdf


def run_quantum_pipeline(output_dir: Path | None = None, date: str | None = None,
                         *, publish: bool = True) -> Path | None:
    # Import locally so the NLP module remains usable without a circular import.
    from daily_arxiv_feed.main import _push_pdf, _write_staging
    output_dir = output_dir or Path(__file__).resolve().parent.parent / "output" / "quantum"
    date = date or datetime.now().strftime("%Y-%m-%d")
    staging = output_dir / "staging" / date
    logger.info("=== Quantum: fetch ===")
    papers = fetch_papers(categories=QUANTUM_CATEGORIES)
    _write_staging(staging, "01_fetched.json", [asdict(p) for p in papers])
    eligible = [p for p in papers if full_abstract(p.abstract)]
    manifest = {"digest":"quantum", "date":date, "source_categories":QUANTUM_CATEGORIES,
                "fetched_unique_papers":len(papers), "nonempty_abstract_papers":len(eligible),
                "excluded_empty_abstract_papers":len(papers)-len(eligible),
                "selected_papers":None, "rendered_full_abstracts":0,"status":"selecting",
                "format":"unabridged author abstracts; announcement wrapper removed only"}
    if not eligible:
        manifest.update(selected_papers=0,status="no_eligible_papers")
        _write_staging(staging,"04_manifest.json",manifest)
        logger.info("Quantum: no papers with nonempty abstracts")
        return None
    client = get_client()
    _write_staging(staging,"04_manifest.json",manifest)
    try:
        selected, decisions = filter_papers(client, eligible, filter_system_prompt=FILTER_PROMPT,
            rank_system_prompt=RANK_PROMPT, max_papers=MAX_QUANTUM_PAPERS,batch_size=40,
            checkpoint_path=staging/"02_filter_checkpoint.json")
    except Exception as error:
        manifest.update(status="selection_failed",error_type=type(error).__name__)
        _write_staging(staging,"04_manifest.json",manifest)
        raise
    _write_staging(staging,"02_filtered.json",decisions)
    if not selected:
        manifest.update(selected_papers=0,status="no_qualifying_selection")
        _write_staging(staging,"04_manifest.json",manifest)
        logger.info("Quantum: no qualifying papers selected from %d eligible papers",len(eligible))
        return None
    decision_map = {d["arxiv_id"]:d for d in decisions}
    entries = []
    for paper in selected:
        abstract = full_abstract(paper.abstract)
        categories = [c for c in decision_map[paper.arxiv_id]["categories"] if c in QUANTUM_SECTIONS]
        if not categories:
            raise ValueError(f"Quantum selection has no valid section: {paper.arxiv_id}")
        entries.append({"arxiv_id":paper.arxiv_id,"title":paper.title,"authors":paper.authors,
                        "url":paper.url,"categories":categories,"abstract":abstract,
                        "abstract_sha256":hashlib.sha256(abstract.encode()).hexdigest()})
    manifest.update(selected_papers=len(selected),eligible_not_selected=len(eligible)-len(selected),
                    status="adding_takeaways")
    _write_staging(staging,"04_manifest.json",manifest)
    try:
        entries = add_takeaways(client, selected, entries)
    except Exception as error:
        manifest.update(status="takeaway_failed",error_type=type(error).__name__)
        _write_staging(staging,"04_manifest.json",manifest)
        raise
    _write_staging(staging,"03_full_abstracts.json",entries)
    manifest.update(selected_papers=len(selected),eligible_not_selected=len(eligible)-len(selected),status="rendering")
    _write_staging(staging,"04_manifest.json",manifest)
    # The separate reading aid never changes or shortens the author abstract.
    tex = render_latex(entries,date,category_names=QUANTUM_SECTIONS,template_name="quantum_digest.tex")
    try:
        pdf = compile_pdf(tex,output_dir,date)
    except Exception as error:
        manifest.update(status="render_failed",error_type=type(error).__name__)
        _write_staging(staging,"04_manifest.json",manifest)
        raise
    manifest.update(selected_papers=len(selected),rendered_full_abstracts=len(entries),status="complete",
                    eligible_not_selected=len(eligible)-len(selected),pdf=str(pdf),pages=get_page_count(pdf))
    _write_staging(staging,"04_manifest.json",manifest)
    if publish:
        _push_pdf(pdf)
    logger.info("Quantum: rendered %d/%d selected full abstracts from %d eligible papers; PDF %s",
                len(entries),len(selected),len(eligible),pdf)
    return pdf
