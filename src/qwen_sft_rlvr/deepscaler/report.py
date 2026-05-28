from __future__ import annotations

import json
from pathlib import Path


class DeepScaleRReport:
    BENCHMARKS = ["aime_2024", "math500", "amc_2023", "minerva_math", "olympiad_bench"]

    def compare(
        self,
        eval_root: str,
        runs: list[str],
        output_dir: str,
        benchmarks: list[str] | None = None,
    ) -> dict:
        root = Path(eval_root)
        selected = benchmarks or self._common_benchmarks(root, runs)
        result = {run: self._load_run(root, run, selected) for run in runs}
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (out / "comparison.md").write_text(self._markdown(result, selected), encoding="utf-8")
        return result

    def _common_benchmarks(self, root: Path, runs: list[str]) -> list[str]:
        available = []
        for benchmark in self.BENCHMARKS:
            if all((root / run / benchmark / "metrics.json").exists() for run in runs):
                available.append(benchmark)
        if not available:
            raise FileNotFoundError(f"No common benchmark metrics found for runs: {runs}")
        return available

    def _load_run(self, root: Path, run: str, benchmarks: list[str]) -> dict:
        metrics = {}
        for benchmark in benchmarks:
            path = root / run / benchmark / "metrics.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing metrics file: {path}")
            metrics[benchmark] = json.loads(path.read_text(encoding="utf-8"))
        return metrics

    def _markdown(self, result: dict, benchmarks: list[str]) -> str:
        lines = ["# DeepScaleR Benchmark Comparison", ""]
        headers = ["run", *benchmarks, "average"]
        lines.append("|" + "|".join(headers) + "|")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for run, metrics in result.items():
            values = [metrics[b]["pass_at_1"] for b in benchmarks]
            cells = [run, *[f"{v:.4f}" for v in values], f"{sum(values) / len(values):.4f}"]
            lines.append("|" + "|".join(cells) + "|")
        return "\n".join(lines) + "\n"
