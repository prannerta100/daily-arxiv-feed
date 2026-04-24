from daily_arxiv_feed.compile import render_latex


def test_render_latex_produces_valid_tex():
    summaries = [{
        "arxiv_id": "2504.1",
        "title": "Test Paper",
        "authors": ["Author A", "Author B"],
        "url": "https://arxiv.org/abs/2504.1",
        "categories": [1],
        "summary": {
            "one_line_takeaway": "Key insight here.",
            "key_contribution": "They did X.",
            "method": "Using Y approach.",
            "most_important_result": "Z improved by 10%.",
        },
    }]
    tex = render_latex(summaries, "2026-04-24", overview="Multilingual NLP is hot today.")
    assert "Test Paper" in tex
    assert "2026-04-24" in tex
    assert "Key insight here" in tex
    assert "arxiv" in tex
    assert "Field Overview" in tex


def test_render_latex_groups_by_category():
    summaries = [
        {
            "arxiv_id": "2504.1", "title": "P1", "authors": ["A"], "url": "u",
            "categories": [1],
            "summary": {"one_line_takeaway": "T1", "key_contribution": "C1", "method": "M1", "most_important_result": "R1"},
        },
        {
            "arxiv_id": "2504.2", "title": "P2", "authors": ["B"], "url": "u",
            "categories": [3],
            "summary": {"one_line_takeaway": "T2", "key_contribution": "C2", "method": "M2", "most_important_result": "R2"},
        },
    ]
    tex = render_latex(summaries, "2026-04-24")
    assert "Multilingual NLP" in tex
    assert "Foundation Model Theory" in tex


def test_render_latex_escapes_summary_special_chars():
    summaries = [{
        "arxiv_id": "2504.1",
        "title": "Paper Title",
        "authors": ["Author A"],
        "url": "https://arxiv.org/abs/2504.1",
        "categories": [1],
        "summary": {
            "one_line_takeaway": "100% improvement & more",
            "key_contribution": "Uses $special chars#",
            "method": "M",
            "most_important_result": "R",
        },
    }]
    tex = render_latex(summaries, "2026-04-24")
    assert "\\%" in tex
    assert "\\&" in tex
    assert "\\$" in tex
    assert "\\#" in tex


def test_render_latex_preserves_latex_in_titles():
    summaries = [{
        "arxiv_id": "2504.1",
        "title": 'Fr\\"obe et al.',
        "authors": ['Fr\\"obe'],
        "url": "https://arxiv.org/abs/2504.1",
        "categories": [1],
        "summary": {
            "one_line_takeaway": "T",
            "key_contribution": "C",
            "method": "M",
            "most_important_result": "R",
        },
    }]
    tex = render_latex(summaries, "2026-04-24")
    assert 'Fr\\"obe' in tex


def test_render_latex_escapes_special_chars_in_titles():
    summaries = [{
        "arxiv_id": "2504.1",
        "title": "R&D for 50% Improvement",
        "authors": ["Author A & Author B"],
        "url": "https://arxiv.org/abs/2504.1",
        "categories": [1],
        "summary": {
            "one_line_takeaway": "T",
            "key_contribution": "C",
            "method": "M",
            "most_important_result": "R",
        },
    }]
    tex = render_latex(summaries, "2026-04-24")
    assert r"R\&D" in tex
    assert r"50\%" in tex
    assert r"Author A \& Author B" in tex
