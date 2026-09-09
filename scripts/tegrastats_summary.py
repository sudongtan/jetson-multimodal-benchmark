"""Parse a tegrastats log and merge peak RAM / average power into a result JSON.

Run on the host, after capturing tegrastats around a benchmark run:
    tegrastats --interval 200 --logfile /tmp/tegra.log &
    TEGRA_PID=$!
    <run the benchmark>
    kill $TEGRA_PID
    python3 scripts/tegrastats_summary.py /tmp/tegra.log results/llm-benchmarks/some_run.json

Line format (JetPack 6.2.2, Orin Nano), one sample per line:
    09-09-2026 09:46:29 RAM 2454/7607MB (lfb 2x4MB) SWAP 0/3804MB (cached 0MB)
    CPU [...] GR3D_FREQ 0% ... VDD_IN 3410mW/3410mW VDD_CPU_GPU_CV 560mW/560mW VDD_SOC 1083mW/1082mW
VDD_IN is instant/average-since-tegrastats-started mW; since tegrastats is started right
before and killed right after each benchmark run, the last line's average is a close
proxy for that run's average power draw.
"""
import argparse
import json
import re
import sys
from pathlib import Path

RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")
VDD_IN_RE = re.compile(r"VDD_IN (\d+)mW/(\d+)mW")


def summarize(log_path: Path):
    peak_ram_mb = 0
    last_avg_power_mw = None

    for line in log_path.read_text().splitlines():
        ram_match = RAM_RE.search(line)
        if ram_match:
            peak_ram_mb = max(peak_ram_mb, int(ram_match.group(1)))

        power_match = VDD_IN_RE.search(line)
        if power_match:
            last_avg_power_mw = int(power_match.group(2))

    if peak_ram_mb == 0 or last_avg_power_mw is None:
        raise ValueError(f"no RAM/VDD_IN samples found in {log_path}")

    return {
        "peak_memory_mb": peak_ram_mb,
        "avg_power_w": last_avg_power_mw / 1000.0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", help="tegrastats --logfile output for this run")
    ap.add_argument("result_json", help="benchmark result JSON to merge peak_memory_mb/avg_power_w into")
    args = ap.parse_args()

    stats = summarize(Path(args.log_path))

    result_path = Path(args.result_json)
    result = json.loads(result_path.read_text())
    result.update(stats)
    result_path.write_text(json.dumps(result, indent=2))
    print(f"{result_path}: {stats}")
