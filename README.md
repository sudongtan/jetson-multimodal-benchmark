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
| LLM (llama.cpp vs. MLC) | Framework/quantization comparison | ✅ Done |
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

One model — **Qwen2.5-1.5B-Instruct** — deployed two ways (llama.cpp,
MLC) at matched Q4 / Q8 / FP16 quantization levels wherever the framework
supports it, each measuring decode tokens/sec, time-to-first-token, and
peak memory/power.

![LLM framework comparison](docs/llm-framework-comparison.png)

| Framework | Quant | tokens/sec (decode) |
|---|---|---|
| llama.cpp | Q4_K_M | 32.0 |
| llama.cpp | Q8_0 | 22.5 |
| llama.cpp | FP16 | 13.1 |
| MLC | q4f16_1 | 60.6 |
| MLC | q0f16 (FP16) | 29.7 |

**Takeaway:** MLC is ~2x faster than llama.cpp at every matched
quantization level — its AOT-compiled TVM kernels outperform llama.cpp's
more general-purpose GGUF kernels. But that speed has a cost: MLC needs a
model- and GPU-specific compile step that's slower to set up and more
version-fragile (new model architectures often need a newer MLC release,
which can lag behind or drop support for older JetPack versions), while
llama.cpp runs any GGUF file directly with no separate compile step and
picks up new model releases almost immediately via the community's GGUF
conversions. In practice that's a real speed-vs-robustness tradeoff, not
a clear win for either framework.

**Why no TensorRT-Edge-LLM:** attempted as a third framework via
jetson-containers' `tensorrt_edgellm` package, which has no prebuilt image
and builds a ~30-package dependency chain from source. That build hit
several out-of-memory failures caused by build tools defaulting to
parallelism levels the board's 8GB can't support; each was root-caused and
fixed (constraining build concurrency/memory, capping a compiler's own
job count, disabling an irrelevant GPU-architecture target). Even with
those fixed, TensorRT-LLM's engine-build workflow itself is not a practical
fit here — its ahead-of-time compilation phase is memory-intensive and,
per NVIDIA's own TensorRT-LLM engineers, can't fall back to swap when it
runs out of memory; they recommend other backends on hardware this
constrained. It's kept as an optional, separate script
(`run_tensorrt_edgellm.sh`) rather than part of the baseline above, which
already meets this phase's goal on its own.

Reproduce with:
```bash
sudo nvpmodel -m 2 && sudo jetson_clocks   # MAXN_SUPER, re-apply after reboot
bash llm-benchmarks/run_all.sh             # llama.cpp + MLC, Q4/Q8/FP16 sweep
python3 llm-benchmarks/plot_results.py

# optional stretch goal, not part of the baseline above:
bash llm-benchmarks/run_tensorrt_edgellm.sh
```

## Repo structure

```
vision-benchmarks/    # YOLOv8n: PyTorch -> ONNX -> TensorRT, FP16/INT8 comparison
llm-benchmarks/       # LLM via llama.cpp, MLC (TensorRT-Edge-LLM optional, see LLM benchmarks below)
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
