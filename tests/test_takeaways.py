import json
from unittest.mock import MagicMock, patch

import pytest

from daily_arxiv_feed.compile import render_latex
from daily_arxiv_feed.fetch import Paper
from daily_arxiv_feed.takeaways import add_takeaways, MAX_CHARS, MAX_WORDS


def paper(aid="id", abstract="In simulation, the decoder corrects more errors."):
    return Paper(aid, "A learned decoder", ["Author"], abstract, ["quant-ph"], "url")


def generated(line, aid="id"):
    return json.dumps({"papers":[{"arxiv_id":aid,"one_line_takeaway":line}]})


def checked(passed=True, issues=None):
    return json.dumps({"papers":[{"arxiv_id":"id","passed":passed,"issues":issues or []}]})


def test_rejects_hardware_overstatement_then_verifies_replacement_and_preserves_body():
    entry={"arxiv_id":"id","abstract":paper().abstract,"summary":{"excerpt":"Existing text."}}
    replies=[generated("The decoder corrects errors on hardware."),
             checked(False,["The abstract reports simulation, not hardware."]),
             generated("A learned decoder corrects more errors in simulation."),checked()]
    with patch("daily_arxiv_feed.takeaways.chat",side_effect=replies) as chat:
        result=add_takeaways(MagicMock(),[paper()],[entry])[0]
    assert result['takeaway_verification']['attempts']==2
    assert 'simulation' in result['one_line_takeaway']
    assert result['abstract']==entry['abstract']
    assert result['summary']==entry['summary']
    assert 'hardware' in chat.call_args_list[2].kwargs['user']
    assert 'one_line_takeaway' not in entry  # does not mutate source entries


def test_reuses_only_verified_lines_for_unchanged_sources():
    with patch("daily_arxiv_feed.takeaways.chat",side_effect=[generated("The decoder improves correction in simulation."),checked()]):
        entries=add_takeaways(MagicMock(),[paper()],[{"arxiv_id":"id"}])
    with patch("daily_arxiv_feed.takeaways.chat",side_effect=AssertionError("Must use cache")):
        assert add_takeaways(MagicMock(),[paper()],entries)==entries
    changed=paper(abstract="A revised abstract reports a different result.")
    with patch("daily_arxiv_feed.takeaways.chat",side_effect=[generated("The revised study reports a different result."),checked()]) as chat:
        updated=add_takeaways(MagicMock(),[changed],entries)
    assert chat.call_count==2
    assert updated[0]['takeaway_verification']['source_fingerprint']!=entries[0]['takeaway_verification']['source_fingerprint']


@pytest.mark.parametrize("invalid",["word "*(MAX_WORDS+1)+"end.","x"*MAX_CHARS+".","A line.\nAnother line."])
def test_invalid_length_or_linebreaks_cannot_reach_verification(invalid):
    with patch("daily_arxiv_feed.takeaways.chat",return_value=generated(invalid)) as chat:
        with pytest.raises(ValueError,match="Could not verify"):
            add_takeaways(MagicMock(),[paper()],[{"arxiv_id":"id"}])
    assert chat.call_count==3  # rejected before any verification call


def test_missing_ids_cannot_be_silently_published():
    with patch("daily_arxiv_feed.takeaways.chat",return_value='{"papers":[]}'):
        with pytest.raises(ValueError,match="each supplied ID"):
            add_takeaways(MagicMock(),[paper()],[{"arxiv_id":"id"}])


@pytest.mark.parametrize("quantum",[False,True])
def test_both_templates_render_italic_line_before_unchanged_body(quantum):
    body="The full original text remains. "*100
    entry={"arxiv_id":"id","title":"Title","authors":["Author"],"url":"url","categories":[1],
           "one_line_takeaway":"The method improves accuracy & efficiency.",
           "abstract":body,"summary":{"excerpt":body}}
    tex=render_latex([entry],"date",template_name="quantum_digest.tex" if quantum else "neurips_digest.tex")
    italic=r"\textit{The method improves accuracy \& efficiency.}"
    assert italic in tex
    assert body in tex
    assert tex.index(italic)<tex.index(body)
