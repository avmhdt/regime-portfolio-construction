"""FR-15: the AST scan rejects hmmlearn calls that condition on the whole sequence."""

from __future__ import annotations

from pathlib import Path

import pytest

from regport.regime import guard

SRC = Path(__file__).resolve().parents[2] / "src"
HMM_PATH = Path("src/regport/regime/hmm.py")
OTHER_PATH = Path("src/regport/opt/erc.py")


@pytest.mark.parametrize("name", ["predict", "predict_proba", "score_samples"])
def test_banned_everywhere(name: str) -> None:
    findings = guard.scan_source(f"model.{name}(x)\n", OTHER_PATH)
    assert [f.message for f in findings] == [f"banned call .{name}()"]


def test_banned_through_getattr() -> None:
    findings = guard.scan_source('getattr(model, "predict_proba")(x)\n', OTHER_PATH)
    assert findings and "getattr" in findings[0].message


def test_decode_banned_where_hmmlearn_is_imported() -> None:
    source = "from hmmlearn.hmm import GaussianHMM\nGaussianHMM().decode(x)\n"
    findings = guard.scan_source(source, HMM_PATH)
    assert [f.message for f in findings] == ["banned call .decode()"]


def test_bytes_decode_allowed_without_hmmlearn() -> None:
    assert guard.scan_source("b'x'.decode('utf-8')\n", OTHER_PATH) == []


def test_fit_and_score_allowed_in_hmm_module() -> None:
    source = "import hmmlearn.hmm\nm = hmmlearn.hmm.GaussianHMM()\nm.fit(x)\nm.score(x)\n"
    assert guard.scan_source(source, HMM_PATH) == []


def test_hmmlearn_import_outside_hmm_module() -> None:
    findings = guard.scan_source("import hmmlearn\n", OTHER_PATH)
    assert findings and "only be imported" in findings[0].message
    findings = guard.scan_source("from hmmlearn import hmm\n", OTHER_PATH)
    assert findings and "only be imported" in findings[0].message


def test_main_reports_and_exits(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("m.predict(x)\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("m.predict(x)\n", encoding="utf-8")
    assert guard.main([str(tmp_path)]) == 1
    assert "bad.py:1: banned call .predict()" in capsys.readouterr().out
    good = tmp_path / "sub" / "good.py"
    good.parent.mkdir()
    good.write_text("b'x'.decode()\n", encoding="utf-8")
    assert guard.main([str(good)]) == 0


def test_src_is_clean() -> None:
    """The real source tree has no banned calls (AC-W-01, scan part)."""
    findings = guard.scan_paths([SRC])
    assert findings == [], "\n".join(map(str, findings))
