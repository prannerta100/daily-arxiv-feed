import json
from unittest.mock import MagicMock, patch

from daily_arxiv_feed.excerpts import abstract_excerpt, fits_budget, summary_limits, summary_text
from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.summarize import generate_excerpt
from daily_arxiv_feed.verify import verify_summaries
from daily_arxiv_feed.main import _compress_summaries, _compile_digest


def paper(abstract="The authors compare two languages. Their method improves translation."):
    return Paper("1234.56789", "Translation", ["A"], abstract, ["cs.CL"], "url")


def entry(text, **kwargs):
    p = paper()
    return {"arxiv_id":p.arxiv_id,"title":p.title,"authors":p.authors,"url":p.url,
            "categories":[1],"summary":{"excerpt":text},**kwargs}


def test_word_and_character_limits_are_independent():
    assert not fits_budget({"excerpt":"longword longword"},{"max_words":10,"max_chars":10})
    assert not fits_budget({"excerpt":"a b c d"},{"max_words":3,"max_chars":100})
    assert fits_budget({"excerpt":"a b c."},{"max_words":3,"max_chars":6})


def test_reruns_do_not_compound_length_allowance():
    previous = entry("word "*100)
    limits = summary_limits(previous)
    assert limits["max_words"] <= 115
    assert limits["max_chars"] <= int(len(summary_text(previous["summary"]))*1.15)
    next_entry = entry("word "*110,summary_limits=limits)
    assert summary_limits(next_entry)==limits


def test_fallback_selects_whole_source_sentences_and_removes_rss_header():
    source = "arXiv:1234.56789v1 Announce Type: new\nAbstract: The study compares languages. The authors train a model. Translation improves."
    result=abstract_excerpt(source,{"max_words":8,"max_chars":70})
    assert result=={"excerpt":"The study compares languages. Translation improves."}
    assert fits_budget(result,{"max_words":8,"max_chars":70})


def test_overlong_response_retries_then_uses_source_without_truncation():
    p=paper()
    limits={"max_words":12,"max_chars":100}
    with patch("daily_arxiv_feed.summarize.chat",return_value=json.dumps({"excerpt":"detail "*80})) as call:
        result,source=generate_excerpt(MagicMock(),p,limits)
    assert call.call_count==2
    assert source=="abstract_sentences"
    assert fits_budget(result,limits)
    assert result["excerpt"].endswith(".")


def test_rewrite_keeps_original_budget_and_falls_back_after_failed_verification():
    p=paper()
    limits={"max_words":15,"max_chars":120}
    original=entry("Models solve every language.",summary_limits=limits)
    fail=json.dumps({"passed":False,"issues":["The abstract does not support this conclusion."]})
    with patch("daily_arxiv_feed.verify.chat",return_value=fail):
        with patch("daily_arxiv_feed.summarize.chat",return_value=json.dumps({"excerpt":"The authors claim universal translation."})) as rewrite:
            out=verify_summaries(MagicMock(),[p],[original])[0]
    assert out["summary_source"]=="abstract_sentences"
    assert "universal" not in summary_text(out["summary"])
    assert fits_budget(out["summary"],limits)
    assert "15 words" in rewrite.call_args.kwargs["user"]


def test_compression_uses_source_and_does_not_mutate_input():
    p=paper()
    original=entry("A readable sentence. "*25,summary_limits={"max_words":130,"max_chars":1000})
    original_text=summary_text(original["summary"])
    with patch("daily_arxiv_feed.main.generate_excerpt",return_value=({"excerpt":"The method improves translation."},"compressed_abstract")) as generate:
        out=_compress_summaries(MagicMock(),[original],[p])
    assert summary_text(original["summary"])==original_text
    assert out[0]["arxiv_id"]==p.arxiv_id
    assert generate.call_args.args[1].abstract==p.abstract
    assert out[0]["summary_limits"]["max_words"] < len(original_text.split())


def test_compressed_text_is_verified_before_final_render(tmp_path):
    p=paper()
    original=[entry("Original text.")]
    compressed=[entry("Shorter text.")]
    corrected=[entry("Faithful final text.")]
    with patch("daily_arxiv_feed.main.render_latex",return_value="tex") as render, \
         patch("daily_arxiv_feed.main.compile_pdf",return_value=tmp_path/"digest.pdf"), \
         patch("daily_arxiv_feed.main.get_page_count",side_effect=[4,2]), \
         patch("daily_arxiv_feed.main._compress_summaries",return_value=compressed), \
         patch("daily_arxiv_feed.main.verify_summaries",return_value=corrected) as verify:
        _,out=_compile_digest(MagicMock(),[p],original,tmp_path,"date","")
    assert verify.call_args.args[2]==compressed
    assert render.call_args.args[0]==corrected
    assert out==corrected
