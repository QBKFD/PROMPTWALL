"""The README's result tables are generated from results/results.json; this test
fails if someone edits a number by hand or forgets to regenerate after a new run."""

from pathlib import Path

from eval.make_charts import load, render_readme

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_generated_blocks_match_results():
    text = README.read_text()
    assert render_readme(text, load()) == text, "run `uv run python -m eval.make_charts`"


def test_readme_has_no_em_or_en_dashes():
    text = README.read_text()
    em_dash, en_dash = chr(0x2014), chr(0x2013)
    assert em_dash not in text and en_dash not in text
