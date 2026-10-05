# Phase 14 — Serving & Optimization: Latency & Memory Benchmarking

First sub-step of Phase 14. Measures real request latency and GPU memory usage for the FastAPI backend's two endpoints, replacing the informal single-sample timings from Phase 13 with proper distributions.

## Method

A lightweight `GET /debug/memory` endpoint was added, reporting `torch.cuda.memory_allocated()`, `max_memory_allocated()`, and `memory_reserved()` from inside the server process — the only way to get an accurate GPU memory figure, since the benchmarking client is a separate process and cannot query the server's CUDA state directly.

Against the running server (not in-process — this measures the real deployed path, including HTTP and FastAPI overhead), `scripts/benchmark_api.py` sent: one initial request (labeled separately, since it may carry residual warmup from server start rather than representing a true isolated cold start), then 50 sequential `POST /predict` requests and 20 sequential `POST /explain` requests, using the same test-split image and clinical fields used throughout this project's verification work. Latency was measured per-request with `time.perf_counter()` around the full request/response cycle; mean, median, p95, p99, min, and max are reported for each endpoint. Memory was read once before and once after the full benchmark run.

## Results

| Metric | `/predict` (n=50) | `/explain` (n=20) |
|---|---|---|
| Mean | 173.7 ms | 1288.6 ms |
| Median | 167.0 ms | 1267.0 ms |
| p95 | 226.1 ms | 1427.8 ms |
| p99 | 241.5 ms | 1428.9 ms |
| Min | 146.7 ms | 1181.3 ms |
| Max | 241.5 ms | 1428.9 ms |

First request (post-server-start): 412.0 ms — noticeably higher than the warm `/predict` median (167.0 ms), consistent with some one-time warmup cost (CUDA context setup, first-call overhead) that subsequent requests don't pay.

**GPU memory**: peak allocated across the entire 70-request benchmark run was 205.7 MB, with 232 MB reserved. On a 6GB VRAM card, this leaves substantial headroom — memory is not a binding constraint for this system at the tested request pattern (sequential, single in-flight request at a time).

## Interpretation

**`/predict` is fast and tightly distributed** — median 167ms with a narrow spread (min 147ms to max 242ms, only two mild outliers above the main cluster, visible in the boxplot). This is well within interactive-use territory: two full forward passes (E04 and E08) plus clinical feature encoding, completed in well under a quarter of a second at the slowest observed case.

**`/explain` is roughly 7–8x slower than `/predict`, as expected given its workload** (5 Grad-CAM backward passes + 5 separate SHAP `DeepExplainer` constructions, one per label, each evaluated against a 50-sample background) — but it is **consistently** bounded, not prone to occasional severe slowdowns. The gap between median (1267ms) and p95 (1428ms) is modest (~160ms), and the boxplot shows no extreme outliers, only a moderately wide interquartile range. This is a predictable cost a user or frontend can design around (e.g. a loading indicator with a roughly-known duration), not an unreliable one.

**Memory is not currently a bottleneck.** 205.7 MB peak against 6GB available VRAM means there is no indication this system would need memory optimization before scaling to, for example, concurrent requests — though this benchmark only tested sequential single-request load, not concurrency (see Limitations).

## Limitations

- All requests were sent **sequentially, one at a time** — this measures single-request latency, not throughput or behavior under concurrent load. A production deployment serving multiple simultaneous users would need a separate concurrency benchmark (e.g. multiple simultaneous requests) to characterize queueing behavior and whether memory headroom holds up under parallel `/explain` calls specifically, since SHAP's `DeepExplainer` construction is the most expensive part of that endpoint.
- The "first request" measurement is explicitly not a true isolated cold start (a fresh server process measuring only its very first request with nothing before it) — it reflects "first request after this benchmark script started," which may follow other activity on the same server process. Reported as a labeled, caveated data point, not presented as a rigorous cold-start figure.
- `max_allocated_mb` is cumulative since server process start, not reset between the `/predict` and `/explain` phases of the benchmark — so the reported peak (205.7 MB) reflects the worst point across the whole run, which is the practically useful number for capacity planning, but does not isolate which endpoint alone drove that peak.
- Benchmarked on this project's specific hardware (GTX 1660 Ti, 6GB VRAM); these absolute numbers would differ on other hardware, though the relative pattern (`/explain` costing several times more than `/predict`, memory being comfortably within budget) would likely hold directionally.

Full artifacts: `outputs/benchmarks/api_latency_benchmark.json` (raw stats), `outputs/benchmarks/latency_distribution.png` (boxplot).
