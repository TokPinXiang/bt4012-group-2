"""Input contract (prediction files) and frontmatter definitions."""

import pandas as pd
import pytest

from conftest import FLAG, SAME, T1, T2, T3
from src.agent.specs import load_agent_spec, parse_frontmatter
from src.agent.toolkit import CaseToolkit
from src.agent.validation import validate_predictions

GOOD = pd.DataFrame({
    "tx_id": [T1, T2, T3, FLAG, SAME],
    "split": ["train", "train", "train", "validation", "validation"],
    "score": [0.1, 0.8, 0.7, 0.95, 0.2],
    "predicted_label": [0, 1, 1, 1, 0],
    "model_name": "xgb",
})


def check(tmp_path, db_path, df):
    p = tmp_path / "predictions_x.csv"
    df.to_csv(p, index=False)
    rep = validate_predictions(p, db_path)
    return rep, {i.check for i in rep.issues if i.level == "error"}


def test_good_file_passes(tmp_path, db_path):
    rep, errors = check(tmp_path, db_path, GOOD)
    assert rep.ok and not errors, rep.render()


@pytest.mark.parametrize("mutate, expected", [
    (lambda d: d.drop(columns=["predicted_label"]), "columns"),
    (lambda d: d.assign(tx_id=d.tx_id.str.replace("HI-Small:", "")), "row_schema"),
    (lambda d: d.assign(score=d.score * 10), "row_schema"),
    (lambda d: d.assign(split="valid"), "row_schema"),
    (lambda d: pd.concat([d, d.iloc[[0]]]), "unique_tx"),
    (lambda d: d.assign(tx_id=d.tx_id.replace(T1, "HI-Small:12345678")), "tx_exists"),
    (lambda d: d.assign(split=d.split.replace("validation", "test")), "split_matches"),
    (lambda d: d.assign(predicted_label=[1, 0, 1, 1, 0]), "single_threshold"),
    (lambda d: d.assign(model_name=["a", "b", "a", "a", "a"]), "one_model"),
])
def test_bad_files_are_caught(tmp_path, db_path, mutate, expected):
    rep, errors = check(tmp_path, db_path, mutate(GOOD.copy()))
    assert not rep.ok and expected in errors, rep.render()


def test_real_definitions_load():
    agent, tools = load_agent_spec("investigator", toolkit_cls=CaseToolkit)
    assert agent.submit_tool in tools and set(agent.tools) <= set(tools)
    assert "investigator" in agent.prompt.lower()


def test_frontmatter_errors(tmp_path):
    with pytest.raises(ValueError):
        parse_frontmatter("no frontmatter here")
    (tmp_path / "agents").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "agents" / "a.md").write_text(
        "---\nname: a\ndescription: d\nmodel: m\ntools: [ghost]\nsubmit_tool: ghost\n---\nprompt", encoding="utf-8")
    with pytest.raises(ValueError, match="no definition"):
        load_agent_spec("a", directory=tmp_path)
