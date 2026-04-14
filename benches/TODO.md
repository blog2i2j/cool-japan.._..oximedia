# oximedia-benchmarks TODO

**Version: 0.1.3**
**Status as of: 2026-04-15**

Criterion-based performance regression suite for the OxiMedia ecosystem. The
`oximedia-benchmarks` crate (`publish = false`) aggregates benchmarks that
exercise the container, codec, audio, computer-vision, filter-graph, I/O, and
FFmpeg-compat layers, plus quality/metering/format-probe/deduplication utility
benches, backed by a shared `helpers/` module for deterministic synthetic data
generation. Two convenience scripts (`run_benchmarks.sh`, `compare_ffmpeg.sh`)
drive full-suite runs and head-to-head comparisons with a locally installed
FFmpeg 6.1 binary, and Criterion's native HTML reports are written under
`target/criterion/`.

## Current Status

### Registered `[[bench]]` targets (Cargo.toml)

The workspace member declares 11 Criterion harnesses. All of them set
`harness = false` so Criterion owns `main()`:

| Bench name              | Source file                | Coverage                                            |
|-------------------------|----------------------------|-----------------------------------------------------|
| `container_bench`       | `container_bench.rs`       | Matroska/WebM, Ogg, FLAC, MP4, WAV demuxing + probe |
| `codec_bench`           | `codec_bench.rs`           | AV1 header/tile/transform/CDEF, VP8/VP9 decode      |
| `audio_bench`           | `audio_bench.rs`           | FLAC/Opus/Vorbis decode, PCM, resample, filters     |
| `cv_bench`              | `cv_bench.rs`              | Haar, YOLO/NMS, Sobel, Canny, blur, optical flow    |
| `graph_bench`           | `graph_bench.rs`           | Filter graph build/topology/linear+branching runs   |
| `io_bench`              | `io_bench.rs`              | Memory source, BitReader, Exp-Golomb, buffers       |
| `compat_ffmpeg_bench`   | `compat_ffmpeg_bench.rs`   | `oximedia_compat_ffmpeg::parse_and_translate`       |
| `quality_metrics`       | `quality_metrics.rs`       | PSNR + SSIM across broadcast frame sizes            |
| `audio_metering`        | `audio_metering.rs`        | EBU R128 / K-weighted loudness + true peak          |
| `format_probe`          | `format_probe.rs`          | Magic-byte probing across 42 container variants     |
| `dedup_hash`            | `dedup_hash.rs`            | sha2 / xxhash-rust deduplication hashers            |

### Legacy / orphaned `.rs` files

- `codec_benchmark.rs` (~33 KB): detailed AV1/VP8/VP9 encoder benches, but it
  is **not** listed under `[[bench]]`. Cargo will not compile it, yet it still
  carries `criterion_group!` / `criterion_main!`, so contributors may mistake
  it for an active suite.
- `filter_benchmark.rs` (~30 KB): scaling / color-conversion / deinterlace /
  audio filter benches — also not registered in Cargo.toml. It contains
  useful work that is simply unreachable today.

### Shared helpers

- `helpers/mod.rs` — YUV 4:2:0 frame generator, synthetic container headers,
  test-data path resolver (`test_data_dir()`, `test_file()`), with inline unit
  tests. Included as a `mod helpers;` from every bench that uses it.

### Documentation

- `README.md` — quick start, per-suite description, comparing baselines, CI
  notes. Currently describes only 6 suites and still talks about "4 main
  benchmark files" in some sections.
- `BENCHMARKS.md` — performance targets vs FFmpeg 6.1 (container, AV1/VP8/VP9
  decode, Opus/FLAC/Vorbis decode, OpenCV comparison) and methodology.
- `IMPLEMENTATION.md` — historical implementation notes; still claims
  "**Benchmark Suites**: 4 (Container, Codec, Audio, CV)" and "~2,400 SLOC",
  which is stale.

### Orchestration scripts

- `run_benchmarks.sh` — releases results under `benchmark_results/`, supports
  `--quick`, `--bench <name>`, `--save-baseline`, `--baseline`, `--compare`.
  Currently hard-codes only `container_bench`, `codec_bench`, `audio_bench`,
  and `cv_bench` in its full-suite loop; the 7 newer targets are skipped
  unless run explicitly.
- `compare_ffmpeg.sh` — auto-generates test clips via FFmpeg `lavfi`, runs
  `ffmpeg -benchmark` for demux/decode/audio-decode, emits markdown
  comparison report.

## Completed

- [x] Criterion harness wired end-to-end with HTML report output under
  `target/criterion/`.
- [x] Container, codec, audio, cv, graph, io, compat-ffmpeg suites landed and
  compile against workspace dependencies.
- [x] Utility benches (quality_metrics, audio_metering, format_probe,
  dedup_hash) integrated as first-class `[[bench]]` targets.
- [x] `publish = false` set; crate excluded from any `cargo publish` path.
- [x] Workspace inheritance for `version`, `authors`, `edition`, `license`,
  `rust-version`, `homepage`, `repository`.
- [x] `[features]` gating for heavy dependencies: `codec` pulls in
  `oximedia-codec` with `av1`, `vp8`, `vp9`; `cv` pulls in `oximedia-cv`.
- [x] Shared `helpers/` module, reused across all suites.
- [x] `run_benchmarks.sh` baseline save/compare flow compatible with
  Criterion's `--save-baseline` / `--baseline` semantics.
- [x] `compare_ffmpeg.sh` produces timestamped markdown reports under
  `benchmark_results/`.
- [x] Workspace dependencies kept in sync with the root `Cargo.toml`
  (`oximedia-core`, `oximedia-io`, `oximedia-container`, `oximedia-codec`,
  `oximedia-audio`, `oximedia-graph`, `oximedia-cv`,
  `oximedia-compat-ffmpeg`, `oximedia-quality`, `oximedia-metering`,
  `oximedia-dedup`).

## Enhancements

- [ ] Reconcile legacy files: either register `codec_benchmark.rs` and
  `filter_benchmark.rs` as `[[bench]]` entries (e.g.
  `codec_encode_bench`, `filter_bench`) or fold their useful cases into the
  existing `codec_bench.rs` / `graph_bench.rs` and delete the originals so
  contributors cannot accidentally edit dead code.
- [ ] Update `run_benchmarks.sh` so its full-suite loop runs all 11
  registered benches (today it only invokes four). Add a plain `cargo bench
  --manifest-path benches/Cargo.toml` fallback that exercises every target.
- [ ] Refresh `README.md` and `IMPLEMENTATION.md` to match the current
  target count (drop "4 main benchmark files", update the file-structure
  tree, and list the utility suites).
- [ ] Add bench targets for the 0.1.2 / 0.1.3 crate families:
  - [ ] `hdr_bench` — PQ/HLG conversion, tone mapping, HDR10+ SEI builder.
  - [ ] `spatial_bench` — HOA rotation, HRTF convolution, VBAP panning.
  - [ ] `cache_bench` — LRU / ARC / tiered cache hit paths + Bloom filter.
  - [ ] `stream_bench` — BOLA ABR ladder, segment packager throughput.
  - [ ] `video_bench` — motion estimation, deinterlace, scene detection.
  - [ ] `cdn_bench` — edge routing, geo selection, failover decision.
  - [ ] `neural_bench` — tensor ops, conv2d, batch norm hot paths.
  - [ ] `vr360_bench` — equirectangular↔cubemap, EAC, bicubic/Lanczos.
  - [ ] `analytics_bench` — session aggregation, retention curves, A/B split.
  - [ ] `caption_gen_bench` — Knuth-Plass line breaking, speaker diarization.
  - [ ] `pipeline_bench` — end-to-end declarative pipeline (parse → plan →
        execute) on the DSL defined in `oximedia-pipeline`.
- [ ] GPU-specific benches behind a `gpu-bench` feature that pulls in
  `oximedia-gpu` and `oximedia-accel` (bilateral filter, histogram, affine
  transform, DCT compute shader, texture pool LRU).
- [ ] SIMD-specific benches in `oximedia-simd` and `oximedia-accel` that run
  the same kernel through scalar, SSE4.2, AVX2, AVX-512 / VNNI (`sad_8x8_vnni`,
  `sad_4x4_vnni`) and print per-lane throughput with Criterion's
  `BenchmarkGroup::throughput`.
- [ ] NMOS discovery / routing latency benches in `oximedia-routing`
  (`CachedFlowGraph` rebuild, IS-04 fetch, IS-05 connection setup) using the
  `MockNmosRegistry` harness.
- [ ] End-to-end pipeline bench: demux (container) → decode (codec) → filter
  (graph) → encode (codec) → mux (container) over a deterministic 10-second
  clip to catch cross-crate regressions.
- [ ] Memory profiling hooks: capture jemalloc `stats.allocated`, peak RSS,
  and allocation counts alongside the existing throughput numbers (feature
  gated to avoid a Pure-Rust violation on platforms without jemalloc).
- [ ] CI integration: publish Criterion HTML reports as a GitHub Pages
  artifact and wire `critcmp` so PRs surface a comment with delta
  summaries; alert on >5% regressions, fail at >20% (the current README
  mentions the thresholds but the workflow is not checked in).
- [ ] Comparison baselines: document FFmpeg 6.1, OpenCV 4.9, and OpenH264
  build flags used in `compare_ffmpeg.sh` (threads, codec, preset) so
  numbers are reproducible; capture `ffmpeg -buildconf` in the report.
- [ ] `quality_bench` — pair `oximedia-quality` VMAF / SSIM / PSNR with the
  `oximedia-transcode` preset ladder so each preset's visual-quality vs
  encode-time trade-off is tracked per release.
- [ ] `cargo-criterion` integration to enable machine-readable JSON output
  for downstream tooling (regression dashboard, `iai-callgrind` replay).

## Known Issues / Gaps

- **Stale suite count**: `IMPLEMENTATION.md` and parts of `README.md` advertise
  "4 suites" / "4 main benchmark files", while Cargo.toml now defines 11.
  Top-level project documentation still refers to "4 criterion benchmark
  suites in `benches/`" from the 0.1.2 milestone.
- **Dead source files**: `codec_benchmark.rs` and `filter_benchmark.rs` are
  not reachable from any `[[bench]]` entry; they compile in isolation with
  Criterion but are silently skipped by `cargo bench`.
- **Partial script coverage**: `run_benchmarks.sh` only loops over
  `container_bench`, `codec_bench`, `audio_bench`, `cv_bench` in its full-
  suite branch. `graph_bench`, `io_bench`, `compat_ffmpeg_bench`,
  `quality_metrics`, `audio_metering`, `format_probe`, `dedup_hash` are
  reachable only via `--bench <name>` or a bare `cargo bench` invocation.
- **Help text drift**: the `--help` output of `run_benchmarks.sh` lists only
  four available benchmark suites; it omits the seven newer targets.
- **Feature gating**: `codec` and `cv` features are off by default, so a
  plain `cargo bench --manifest-path benches/Cargo.toml` currently fails
  to compile the codec-dependent portions of `codec_bench.rs`. Either make
  the features default or adjust the scripts to pass
  `--features "codec cv"`.
- **CI workflow missing**: `README.md` references
  `.github/workflows/benchmarks.yml`, but no such file is committed.
- **Reliance on system FFmpeg**: `compare_ffmpeg.sh` hard-depends on
  `ffmpeg`/`ffprobe`/`bc` being on `$PATH` and on GNU `stat`/`lscpu`/`free`
  on Linux (macOS falls back). A preflight check and a dockerised
  reference image would make numbers comparable across machines.

## Benchmark Targets / Performance Goals

Detailed throughput goals vs FFmpeg 6.1, dav1d, libvpx, and OpenCV 4.9 live
in `BENCHMARKS.md`. High-level categories already covered there:

- Container demuxing (Matroska/WebM, Ogg, FLAC, MP4, WAV) in MB/s.
- AV1 / VP8 / VP9 decode at 720p / 1080p / 4K in FPS.
- Audio decode (FLAC, Opus, Vorbis) in realtime factor.
- Computer vision (Sobel, Gaussian blur, RGB↔gray) in FPS.

Any new benches added under the Enhancements list should extend
`BENCHMARKS.md` with matching target tables so regressions have an
explicit acceptance bar.

## Future (Post-0.2.0)

| Item                                      | Notes                                                                                       |
|-------------------------------------------|---------------------------------------------------------------------------------------------|
| Continuous benchmarking dashboard         | Host Criterion + `critcmp` JSON history, track trend lines per bench, alert on regressions. |
| Criterion + flamegraph integration        | `--profile-time` + `cargo flamegraph --bench` wired into `run_benchmarks.sh`.               |
| `cargo-codspeed` adoption                 | Off-box measurement for reproducible numbers irrespective of runner load.                   |
| `iai-callgrind` micro-benches             | Instruction-count benches for AV1 transforms, SIMD kernels, bit readers.                    |
| Regression-bot automation                 | Bot posts comment on PRs with delta + sparkline; auto-blocks merges on >20% regressions.    |
| Cross-platform bench matrix               | x86_64 (AVX2 / AVX-512), aarch64 (NEON / SVE), Linux / macOS / Windows.                     |
| WASM benchmark adapter                    | Replay a subset of CPU benches inside `oximedia-wasm` via `wasm-bindgen-test` timers.       |
| GPU cross-vendor bench                    | Run `oximedia-gpu` suites on NVIDIA / AMD / Apple / Intel via the wgpu backend.             |

*Last updated: 2026-04-15 - v0.1.3, oximedia-benchmarks summary*
