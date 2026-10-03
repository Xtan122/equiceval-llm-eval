import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_run_comparison_cli(tmp_path):
    output = tmp_path / "report.json"
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    proc = subprocess.run(
        [sys.executable, "-m", "expeval.run_comparison",
         "--pairs", str(ROOT / "data" / "sample_pairs.json"),
         "--output", str(output)],
        cwd=ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(output.read_text())
    assert set(report["methods"]) == {"equiceval", "ra"}
    assert report["methods"]["equiceval"]["metrics"]["n_total"] == 16
    assert len(report["methods"]["ra"]["decisions"]) == 16
