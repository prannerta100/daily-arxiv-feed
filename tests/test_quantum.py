import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from daily_arxiv_feed.abstract_rendering import latex_abstract
from daily_arxiv_feed.compile import render_latex
from daily_arxiv_feed.fetch import Paper, fetch_papers, fetch_via_rss
from daily_arxiv_feed.filter import filter_papers
from daily_arxiv_feed.main import run_daily_digests
from daily_arxiv_feed.quantum import full_abstract, rebuild_quantum_digest, run_quantum_pipeline


def paper(aid="id",abstract="The authors optimize quantum gates."):
    return Paper(aid,"Quantum gates",["Author"],abstract,["quant-ph"],"https://arxiv.org/abs/"+aid)


def test_full_abstract_removes_only_announcement_preserving_source_tex_and_tail():
    body=r"We optimize $A^{-1}$ with 99\% fidelity. " + "Details remain. "*200 + "Final limitation."
    assert full_abstract("arXiv:idv1 Announce Type: new\nAbstract: "+body)==body
    assert full_abstract(body)==body


def test_custom_fetch_categories_do_not_change_nlp_defaults():
    from daily_arxiv_feed.fetch import CATEGORIES
    before=CATEGORIES.copy()
    with patch("daily_arxiv_feed.fetch._retry_request") as request, \
         patch("daily_arxiv_feed.fetch.feedparser.parse",return_value=MagicMock(entries=[])):
        fetch_via_rss(categories=["quant-ph","physics.app-ph"])
    assert [c.args[0] for c in request.call_args_list]==[
        "https://rss.arxiv.org/rss/quant-ph","https://rss.arxiv.org/rss/physics.app-ph"]
    assert CATEGORIES==before


def test_custom_categories_survive_source_fallback():
    with patch("daily_arxiv_feed.fetch.fetch_via_rss",side_effect=RuntimeError("Unavailable")), \
         patch("daily_arxiv_feed.fetch.fetch_via_api",return_value=[paper()]) as api:
        assert fetch_papers(categories=["quant-ph"])[0].arxiv_id=="id"
    api.assert_called_once_with(categories=["quant-ph"])


def test_filter_keeps_ranking_order_caps_and_rejects_invented_ids():
    papers=[paper(str(i)) for i in range(3)]
    initial={"papers":[{"arxiv_id":p.arxiv_id,"selected":True,"categories":[1],"reasoning":"Quantum control"} for p in papers]}
    ranked={"papers":[{"arxiv_id":aid,"selected":True,"categories":[1],"reasoning":"Strong"} for aid in ["invented","2","0","1"]]}
    with patch("daily_arxiv_feed.filter.chat",side_effect=[json.dumps(initial),json.dumps(ranked)]):
        selected,decisions=filter_papers(MagicMock(),papers,max_papers=2)
    assert [p.arxiv_id for p in selected]==["2","0"]
    assert {d['arxiv_id'] for d in decisions if d['selected']}=={"2","0"}


def test_filter_resumes_completed_batches_after_connection_failure(tmp_path):
    papers=[paper('0'),paper('1')]
    decision=lambda aid: {'arxiv_id':aid,'selected':True,'categories':[1],'reasoning':'Control'}
    checkpoint=tmp_path/'checkpoint.json'
    with patch('daily_arxiv_feed.filter.chat',side_effect=[json.dumps({'papers':[decision('0')]}),RuntimeError('Connection dropped')]):
        with pytest.raises(RuntimeError):
            filter_papers(MagicMock(),papers,batch_size=1,checkpoint_path=checkpoint)
    assert json.loads(checkpoint.read_text())['completed_batches']==1
    with patch('daily_arxiv_feed.filter.chat',side_effect=[
        json.dumps({'papers':[decision('1')]}),json.dumps({'papers':[decision('1'),decision('0')]})]) as chat:
        selected,_=filter_papers(MagicMock(),papers,batch_size=1,checkpoint_path=checkpoint)
    assert chat.call_count==2  # one remaining batch, then ranking
    assert [p.arxiv_id for p in selected]==['1','0']
    assert 'arxiv_id: 1' in chat.call_args_list[0].kwargs['user']


def test_failed_selection_is_not_reported_as_zero_qualifying_papers(tmp_path):
    with patch('daily_arxiv_feed.quantum.fetch_papers',return_value=[paper()]), \
         patch('daily_arxiv_feed.quantum.get_client'), \
         patch('daily_arxiv_feed.quantum.filter_papers',side_effect=RuntimeError('Connection dropped')):
        with pytest.raises(RuntimeError):
            run_quantum_pipeline(tmp_path,'date',publish=False)
    manifest=json.loads((tmp_path/'staging/date/04_manifest.json').read_text())
    assert manifest['status']=='selection_failed'
    assert manifest['selected_papers'] is None
    assert manifest['rendered_full_abstracts']==0


def test_quantum_pipeline_preserves_full_long_abstract_and_separate_outputs(tmp_path):
    body=r"We study $\lambda I + X^T X$. "+"The full methodological detail remains. "*100+"Final qualification."
    papers=[paper("one","arXiv:onev1 Announce Type: new\nAbstract: "+body),paper("empty","")]
    decisions=[{"arxiv_id":"one","selected":True,"categories":[1],"reasoning":"ML optimizes gates"}]
    destination=tmp_path/'quantum'
    pdf=destination/'2026-11-05.pdf'
    with patch("daily_arxiv_feed.quantum.fetch_papers",return_value=papers), \
         patch("daily_arxiv_feed.quantum.get_client"), \
         patch("daily_arxiv_feed.quantum.add_takeaways",side_effect=lambda client,papers,entries: [
             {**entry,"one_line_takeaway":"The method improves quantum gate control."} for entry in entries]), \
         patch("daily_arxiv_feed.quantum.filter_papers",return_value=([papers[0]],decisions)) as select, \
         patch("daily_arxiv_feed.quantum.render_latex",return_value="tex") as render, \
         patch("daily_arxiv_feed.quantum.compile_pdf",return_value=pdf), \
         patch("daily_arxiv_feed.quantum.get_page_count",return_value=7), \
         patch("daily_arxiv_feed.main._push_pdf") as push, \
         patch("daily_arxiv_feed.main._compress_summaries",side_effect=AssertionError("Must not compress")), \
         patch("daily_arxiv_feed.main.summarize_papers",side_effect=AssertionError("Must not summarize")):
        assert run_quantum_pipeline(destination,"2026-11-05",publish=False)==pdf
    push.assert_not_called()
    assert select.call_args.args[1]==[papers[0]]
    assert select.call_args.kwargs['max_papers']==8
    entry=render.call_args.args[0][0]
    assert entry['abstract']==body
    assert entry['one_line_takeaway']=="The method improves quantum gate control."
    assert entry['abstract_sha256']==hashlib.sha256(body.encode()).hexdigest()
    manifest=json.loads((destination/'staging/2026-11-05/04_manifest.json').read_text())
    assert manifest['fetched_unique_papers']==2
    assert manifest['nonempty_abstract_papers']==1
    assert manifest['excluded_empty_abstract_papers']==1
    assert manifest['rendered_full_abstracts']==manifest['selected_papers']==1
    assert manifest['pages']==7


def test_serial_order_includes_nlp_compilation_and_open_before_quantum(tmp_path):
    events=[]
    nlp_pdf=tmp_path/'2026-11-05.pdf'
    quantum_pdf=tmp_path/'quantum/2026-11-05.pdf'
    def nlp(**kwargs):
        events.append('nlp_start')
        nlp_pdf.write_bytes(b'completed PDF')
        events.append('nlp_compiled')
        return nlp_pdf
    def quantum(**kwargs):
        assert nlp_pdf.exists()
        assert kwargs['output_dir']==tmp_path/'quantum'
        events.append('quantum_start')
        return quantum_pdf
    def open_pdf(args,**kwargs):
        events.append('open_nlp' if args[1]==str(nlp_pdf) else 'open_quantum')
    with patch("daily_arxiv_feed.main.run_pipeline",side_effect=nlp), \
         patch("daily_arxiv_feed.quantum.run_quantum_pipeline",side_effect=quantum), \
         patch("daily_arxiv_feed.main.subprocess.run",side_effect=open_pdf):
        outputs=run_daily_digests(tmp_path,"2026-11-05",publish=False,open_pdfs=True)
    assert events==['nlp_start','nlp_compiled','open_nlp','quantum_start','open_quantum']
    assert outputs=={'nlp':nlp_pdf,'quantum':quantum_pdf}


def test_nlp_failure_prevents_quantum_from_starting(tmp_path):
    with patch("daily_arxiv_feed.main.run_pipeline",side_effect=RuntimeError("NLP compilation failed")), \
         patch("daily_arxiv_feed.quantum.run_quantum_pipeline") as quantum:
        with pytest.raises(RuntimeError): run_daily_digests(tmp_path)
    quantum.assert_not_called()


def test_empty_nlp_day_still_allows_quantum(tmp_path):
    with patch("daily_arxiv_feed.main.run_pipeline",return_value=None), \
         patch("daily_arxiv_feed.quantum.run_quantum_pipeline",return_value=tmp_path/'q.pdf') as quantum:
        outputs=run_daily_digests(tmp_path,publish=False)
    quantum.assert_called_once()
    assert outputs['nlp'] is None


def test_abstract_math_is_typeset_and_unknown_commands_cannot_execute():
    text=latex_abstract(r"We use $\lambda I + X^T X$ at 99% fidelity, with α ≥ 0.")
    assert r"$\lambda I + X^T X$" in text
    assert r"99\%" in text
    assert r"\ensuremath{\alpha}" in text
    assert r"\ensuremath{\geq}" in text
    malicious=latex_abstract(r"Outside \input{/etc/passwd} and $\input{/etc/passwd}$.")
    assert r"\input{" not in malicious
    assert "input" in malicious


def test_quantum_template_uses_full_abstracts_and_quantum_sections():
    body="Opening sentence. "+"Detail remains. "*200+"Final qualification."
    row={'arxiv_id':'id','title':'Quantum','authors':['A'],'url':'url','categories':[1],'abstract':body}
    tex=render_latex([row],"date",category_names={1:'Pulse Optimization'},template_name='quantum_digest.tex')
    assert 'Quantum Computing Digest' in tex
    assert 'Pulse Optimization' in tex
    assert body in tex
    assert 'Multilingual NLP' not in tex


def test_quantum_rebuild_rejects_altered_abstract_before_compiling(tmp_path):
    staging=tmp_path/'staging/date'
    staging.mkdir(parents=True)
    (staging/'01_fetched.json').write_text(json.dumps([{'arxiv_id':'id','abstract':'Original abstract.'}]))
    (staging/'03_full_abstracts.json').write_text(json.dumps([{'arxiv_id':'id','abstract':'Changed abstract.','abstract_sha256':'invalid'}]))
    with patch('daily_arxiv_feed.quantum.compile_pdf') as compile_pdf:
        with pytest.raises(ValueError,match='differs from source'):
            rebuild_quantum_digest('date',tmp_path)
    compile_pdf.assert_not_called()
