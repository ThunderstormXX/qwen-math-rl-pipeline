#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
TARGET="${DEEPSCALER_BENCH_RAW_DIR:-data/raw/deepscaler/benchmarks}"
BASE_URL="${DEEPSCALER_BENCH_RAW_URL:-https://raw.githubusercontent.com/applese233/deepscaler/main/deepscaler/data/test}"
mkdir -p "${TARGET}"
echo "[deepscaler] Downloading benchmark JSON files to ${TARGET}"

for file in aime.json math500.json amc.json minerva.json olympiad_bench.json; do
  echo "[deepscaler] ${file}"
  curl -L --fail "${BASE_URL}/${file}" -o "${TARGET}/${file}"
done

echo "[deepscaler] Benchmark download complete"
