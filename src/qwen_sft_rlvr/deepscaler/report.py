from __future__ import annotations

import json
from pathlib import Path


class DeepScaleRReport:
    BENCHMARKS = ["aime_2024", "math500", "amc_2023", "minerva_math", "olympiad_bench"]

    def compare(self, eval_root: str, runs: list[str], output_dir: str) -> dict:
        result = {run: self._load_run(Path(eval_root), run) for run in runs}
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (out / "comparison.md").write_text(self._markdown(result), encoding="utf-8")
        return result

    def _load_run(self, root: Path, run: str) -> dict:
        metrics = {}
        for benchmark in self.BENCHMARKS:
            path = root / run / benchmark / "metrics.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing metrics file: {path}")
            metrics[benchmark] = json.loads(path.read_text(encoding="utf-8"))
        return metrics

    def _markdown(self, result: dict) -> str:
        lines = ["# DeepScaleR Benchmark Comparison", ""]
        headers = ["run", *self.BENCHMARKS, "average"]
        lines.append("|" + "|".join(headers) + "|")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for run, metrics in result.items():
            values = [metrics[b]["pass_at_1"] for b in self.BENCHMARKS]
            cells = [run, *[f"{v:.4f}" for v in values], f"{sum(values) / len(values):.4f}"]
            lines.append("|" + "|".join(cells) + "|")
        return "\n".join(lines) + "\n"
