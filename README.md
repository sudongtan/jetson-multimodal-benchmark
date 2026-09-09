# Jetson Orin Nano: Perception / LLM / VLM Optimization & Benchmarking

A reproducible benchmark suite for vision, LLM, and VLM inference on a Jetson
Orin Nano Super (8GB), across quantization levels, power modes, and
concurrency — plus a combined perception-to-language demo pipeline tying the
results together.

Hardware and one-time environment setup (flashing, power modes, Docker/SSD
configuration, the NVIDIA container runtime) are documented in
[SETUP.md](SETUP.md).

## Status

| Phase | Goal | Status |
|---|---|---|
| Vision (YOLOv8n, TensorRT) | Accuracy vs. speed across FP32/FP16/INT8 | ✅ Done |
| LLM (llama.cpp vs. MLC vs. TensorRT-Edge-LLM) | Framework/quantization comparison | 🚧 In progress |
| VLM (LLaVA/VILA) | Image+text latency & memory | Planned |
| Combined pipeline | Camera → detection → language, live demo | Planned |
| Systematic sweep | Power mode × concurrency × quantization | Planned |

## Vision benchmarks

YOLOv8n exported to TensorRT at three precisions and benchmarked on-device
(FPS, latency percentiles, mAP against `coco128.yaml`):

![Accuracy vs. speed](docs/vision-accuracy-vs-speed.png)

| Precision | FPS | Latency (mean) | mAP50-95 |
|---|---|---|---|
| FP32 | ~85 | ~11.7 ms | 0.4503 |
| FP16 | ~124 | ~8.1 ms | 0.4505 |
| INT8 | ~139 | ~7.2 ms | 0.4415 |

**Takeaway:** FP16 gives ~45% more throughput than FP32 for effectively no
accuracy loss — the clear default for this model/board combination. INT8
pushes another ~12% faster for a small (~2%) mAP drop, worth it only if the
extra headroom matters more than that last bit of accuracy.

Reproduce with:
```bash
cd vision-benchmarks
DATA=coco128.yaml bash run_all.sh
python3 plot_results.py
```
See [vision-benchmarks/run_all.sh](vision-benchmarks/run_all.sh) for what
this runs under the hood (export → TensorRT engine build → benchmark, per
precision, inside the `ultralytics/ultralytics:latest-jetson-jetpack6`
container).

## LLM benchmarks

One model — **Qwen2.5-1.5B-Instruct** — deployed three ways
(llama.cpp, MLC, TensorRT-Edge-LLM), at matched Q4 / Q8 / FP16
quantization levels wherever the framework supports it, each measuring
decode tokens/sec, time-to-first-token (TTFT), and peak memory/power.

**Status:** harness is built, not yet run — see [Status](#status) above.
Once `results/llm-benchmarks/*.json` are populated this section gets the
same chart + summary table treatment as the vision benchmarks.

Reproduce with:
```bash
sudo nvpmodel -m 2 && sudo jetson_clocks   # MAXN_SUPER, re-apply after reboot
bash llm-benchmarks/run_all.sh             # downloads models, quantizes/compiles, benchmarks all 3 x quant sweeps
python3 llm-benchmarks/plot_results.py
```

## Repo structure

```
vision-benchmarks/    # YOLOv8n: PyTorch -> ONNX -> TensorRT, FP16/INT8 comparison
llm-benchmarks/       # LLM via llama.cpp, MLC, TensorRT-Edge-LLM
vlm-benchmarks/       # VLM (LLaVA/VILA) benchmarking (planned)
combined-pipeline/    # perception -> language demo (planned)
results/              # raw JSON benchmark output, one file per run
scripts/              # measurement harness, automation
docs/                 # charts and writeup
```

## Methodology

Every run logs model, precision/quantization, latency percentiles (mean,
p50, p95), throughput, and accuracy to `results/` as JSON — see
[vision-benchmarks/benchmark.py](vision-benchmarks/benchmark.py) for the
vision harness. Later phases add power mode and concurrency sweeps, plus
tokens-per-joule as an efficiency metric for LLM/VLM runs.
