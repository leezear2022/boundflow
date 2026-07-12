"""Contract test for workload-family LOO retry-factor calibration."""

import json

from scripts.calibrate_phase7a_pr11_retry_factor import main
from tests.test_phase7a_pr11_barrier_eval_runner import _write


def test_retry_factor_calibration_emits_selection_and_manifest(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    calibration = tmp_path / "calibration.jsonl"
    out_dir = tmp_path / "out"
    _write(first, workload="first", scale=1.0)
    _write(second, workload="second", scale=1.2)
    calibration.write_text(first.read_text() + second.read_text())

    result = main(
        [
            "--calibration",
            str(calibration),
            "--out-dir",
            str(out_dir),
            "--factors",
            "1.0,1.2",
            "--ridges",
            "0.000001,0.001",
        ]
    )

    assert result == 0
    selection = json.loads((out_dir / "selection.json").read_text())
    assert selection["selected_factor"] in {1.0, 1.2}
    assert selection["selected_ridge"] in {0.000001, 0.001}
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["selected_factor"] == selection["selected_factor"]
    assert manifest["selected_ridge"] == selection["selected_ridge"]
    assert manifest["outputs"]["raw.jsonl"]
