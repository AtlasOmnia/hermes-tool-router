from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "scoring.py"
spec = importlib.util.spec_from_file_location("router_benchmark_scoring", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules["router_benchmark_scoring"] = module
spec.loader.exec_module(module)


def test_scoring_computes_required_recall_and_precision():
    records = [
        ({"web"}, {"web"}),
        ({"file", "terminal"}, {"file"}),
        (set(), set()),
    ]
    result = module.score_routes(records)
    assert result["required_recall"] == 2 / 3
    assert result["toolset_precision"] == 1.0
    assert result["exact_set_accuracy"] == 2 / 3
