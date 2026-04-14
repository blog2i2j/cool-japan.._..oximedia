# oximedia-cli — Command-Line Interface TODO

**Version: 0.1.3**
**Status as of: 2026-04-15**

The `oximedia-cli` crate is the primary user-facing entry point to the OxiMedia
Sovereign Media Framework. It ships **two binaries** from a single crate — the
main `oximedia` multitool with ~85 domain subcommands covering the entire FFmpeg
+ OpenCV feature surface in patent-free, pure-Rust form, and the `oximedia-ff`
drop-in that accepts raw FFmpeg-style argv and executes each translated job
through the native transcode engine. Every subcommand is wired through clap
derive, returns `anyhow::Result<()>`, respects a global `--json` / `--no-color`
/ `-v/-vv/-vvv` / `-q` flag set, and delegates real work to the ~101 workspace
crates. The whole crate weighs in at ~50,900 lines of Rust across 89 source
files (counted on 2026-04-15 via `wc -l src/*.rs src/bin/*.rs`).

---

## Current Status

### Binary targets (`Cargo.toml`)

| Binary         | Entry point                | Purpose                                                  |
|----------------|----------------------------|----------------------------------------------------------|
| `oximedia`     | `src/main.rs` (775 lines)  | Primary multitool; dispatches all domain subcommands.    |
| `oximedia-ff`  | `src/bin/oximedia-ff.rs`   | FFmpeg drop-in — argv → `oximedia-compat-ffmpeg` → exec. |

Both share code via the thin `src/lib.rs` (33 lines) which publicly re-exports
`presets`, `progress`, and `transcode` so the `oximedia-ff` binary does not
duplicate transcode plumbing.

### Source layout (89 files, ~50,900 SLOC)

- **Entry**: `main.rs` (775), `lib.rs` (33), `bin/oximedia-ff.rs` (348).
- **Command surface**: `commands.rs` (1442) — defines the top-level `Commands`
  enum (~85 variants) and shared `MonitorCommand`, `RestoreCommand`,
  `CaptionsCommand`, `PresetCommand` sub-enums; `handlers.rs` (829) — shared
  handler functions (`probe_file`, `show_info`, `show_version`, logging init,
  monitor/restore/captions/preset dispatch).
- **Domain modules (`*_cmd.rs`)**: 77 files, one per top-level subcommand group
  (aaf, access, align, archive, archivepro, audio, audiopost, auto, batch,
  calibrate, captions, clips, cloud, collab, color, conform, dedup, denoise,
  distributed, dolbyvision, drm, edl, farm, ffcompat, filter, forensics, gaming,
  graphics, image, imf, loudness, lut, mam, mir, mixer, monitor, multicam, ndi,
  normalize, optimize, package, playlist, playout, plugin, profiler, proxy, qc,
  quality, recommend, renderfarm, repair, restore, review, rights, routing,
  scaling, scopes, search, stabilize, stream, subtitle, switcher, timecode,
  timeline, timesync, tui, vfx, videoip, virtual, watermark, workflow).
- **Shared helpers**: `analyze.rs`, `batch.rs`, `benchmark.rs`, `concat.rs`,
  `extract.rs`, `metadata.rs`, `progress.rs`, `scene.rs`, `thumbnail.rs`,
  `transcode.rs`, `validate.rs`, plus `presets/` (builtin, custom, device,
  streaming, validate, web) and `sprite/` (generate, output, timestamps, utils).

### Top-level subcommand taxonomy (from `commands::Commands`)

- **Probing & inspection** — `probe`, `info`, `version`, `validate`, `analyze`,
  `benchmark`, `metadata`, `forensics`.
- **Transcoding & muxing** — `transcode` (alias `convert`), `extract`, `batch`,
  `concat`, `thumbnail`, `sprite`, `package` (HLS/DASH), `optimize`,
  `batch-engine` (SQLite-backed persistent queue).
- **Audio** — `audio`, `loudness`, `normalize`, `mixer`, `audiopost`,
  `mir` (music-info retrieval), `align`.
- **Video processing** — `scene`, `scopes`, `denoise`, `stabilize`, `scaling`,
  `filter`, `lut`, `color`, `dolby-vision`, `multicam`, `vfx`, `image`,
  `graphics`.
- **Subtitles & captions** — `subtitle`, `captions`, `timecode`.
- **Broadcast & production** — `playout`, `switcher`, `ndi`, `videoip` (RTP/
  SRT/RIST), `calibrate`, `virtual`, `routing`, `timeline`, `timesync`, `edl`,
  `playlist`, `conform`, `qc`.
- **Archival & asset management** — `archive`, `archive-pro`, `mam`, `clips`,
  `proxy`, `dedup`, `search`, `watermark`, `drm`, `rights`, `access`,
  `repair`, `restore`.
- **Collaboration & workflow** — `collab`, `review`, `workflow`, `auto`,
  `recommend`, `imf`, `aaf`.
- **Infrastructure** — `distributed`, `farm`, `renderfarm`, `cloud`, `monitor`,
  `profiler`, `stream`, `plugin`, `quality`, `gaming`.
- **Interop & UX** — `ffcompat` (alias `ff`), `tui`, `preset`.

### Global flags (`Cli` in `main.rs`)

- `-v / --verbose` (repeatable, `ArgAction::Count` → `-v`, `-vv`, `-vvv`).
- `-q / --quiet` — suppress everything except errors.
- `--no-color` — disables `colored::control` overrides.
- `--json` — propagates into every subcommand that has a structured-output path.

### FFmpeg compatibility story

Two complementary layers both route through `oximedia-compat-ffmpeg`:

1. **In-tool**: `oximedia ff <ffmpeg-args>` (`Commands::Ffcompat`, alias `ff`)
   handled by `src/ffcompat_cmd.rs` (373 lines). Supports `--dry-run` / `--plan`
   to print translated jobs without executing.
2. **Drop-in replacement**: `oximedia-ff` binary (348 lines) — a standalone
   argv→translate→execute pipeline that can be symlinked as `ffmpeg` so legacy
   scripts transparently retarget onto OxiMedia. Honours `-h/--help`,
   `-version/--version`, `--dry-run`, `--plan` and prints FFmpeg-style
   diagnostics (`warning:`, `info:`, `error:` prefixes with hints).

Both rely on `oximedia-compat-ffmpeg`'s `parse_and_translate()` which emits
`DiagnosticKind::{PatentCodecSubstituted, UnknownOptionIgnored, FilterNotSupported,
UnsupportedFeature, Info, Warning, Error}` so H.264/H.265/AAC invocations are
auto-rewritten into AV1/VP9/Opus and reported back transparently.

### Interactive TUI (`tui_cmd.rs`, 521 lines)

Three-tab ratatui interface (`Files`, `Commands`, `About`) with crossterm input,
panic-safe terminal restoration, and a working file browser that lists cwd
entries by size. Keyboard: `q`/`Ctrl+C` quit, `Tab`/`→`/`←`/`Shift+Tab` change
tabs, `↑`/`↓` navigate, `Enter` show details.

---

## Completed `[x]`

- [x] Two-binary crate layout — `oximedia` (primary) and `oximedia-ff` (FFmpeg
  drop-in) both shipping from the same `oximedia-cli` crate.
- [x] Core inspection suite — `probe` (text/json/csv, chapters, per-stream,
  metadata dump, content hash, quality snapshot), `info`, `version`.
- [x] Full transcode pipeline — `transcode` with FFmpeg-compatible aliases
  (`-i`, `-c:v`, `-c:a`, `-b:v`, `-b:a`, `-vf`, `-af`, `-ss`, `-t`, `-r`, `-y`),
  two-pass, CRF, preset names, resume, stream mapping, loudness normalize hook.
- [x] Frame / thumbnail / sprite generation — `extract`, `thumbnail` (single /
  multiple / grid / auto), `sprite` with WebVTT + JSON manifest, configurable
  sampling strategy (uniform / scene-based / keyframe-only / smart) and layout
  modes.
- [x] Batch processing — both the ad-hoc `batch` subcommand (TOML config, `-j`
  jobs, `--continue-on-error`, `--dry-run`) and the persistent SQLite-backed
  `batch-engine submit/status/list/cancel/report` command.
- [x] 0.1.2 additions: `loudness` (EBU R128 analyze / check / standards / info),
  `quality` (PSNR/SSIM/BRISQUE/NIQE etc. compare/analyze/list/explain),
  `dedup` (scan/report/clean/hash/compare), `timecode` (convert/calculate/
  validate/burn-in), `normalize` (analyze/process/check/targets),
  `batch-engine`, `scopes`, `workflow`, `version`.
- [x] ~85 domain subcommand modules wired end-to-end through `commands.rs` →
  `main.rs` match arms → `*_cmd.rs::handle_*_command` handlers, every one
  honouring the global `--json` flag.
- [x] Broadcast & live production coverage — `playout`, `switcher`, `ndi`,
  `videoip`, `multicam`, `routing`, `virtual`, `calibrate`, `timesync`.
- [x] MAM / archival coverage — `mam`, `search`, `dedup`, `archive`, `archive-pro`,
  `proxy`, `clips`, `review`, `drm`, `rights`, `access`, `watermark`, `repair`,
  `restore`.
- [x] Production post coverage — `timeline`, `edl`, `conform`, `qc`, `vfx`,
  `graphics`, `image`, `audiopost`, `mixer`, `mir`, `subtitle`, `captions`,
  `color`, `lut`, `dolby-vision`.
- [x] Infrastructure — `distributed`, `farm`, `renderfarm`, `cloud`, `monitor`,
  `profiler`, `plugin`, `stream`, `quality`, `recommend`, `scaling`, `optimize`.
- [x] FFmpeg compatibility layer — both `oximedia ff <args>` and the standalone
  `oximedia-ff` binary, both wired to `oximedia-compat-ffmpeg::parse_and_translate`
  with patent-codec auto-substitution, diagnostic forwarding, and `--dry-run`.
- [x] Interactive TUI — `oximedia tui` launches a three-tab ratatui UI with
  cwd file browser, command reference with descriptions, and about panel.
- [x] Shared `handlers.rs` — `init_logging` respects `-v` count and `-q`,
  `probe_file`, `show_info`, `show_version` with feature-gated build info.
- [x] Coloured error output — every failure is reported via `colored` as
  `Error: …` in red + `Caused by: …` chain from `anyhow::Error::source()`.
- [x] Progress reporting harness — `progress.rs` (397 lines) built on
  `indicatif` for transcode / batch / analyze long-running operations.
- [x] Preset system — `presets/` module (builtin, custom, device, streaming,
  validate, web) plus `preset list/show/create/template/import/export/remove`
  subcommand; preset doctest defects fixed during 0.1.2.
- [x] Bug fixes shipped during 0.1.2 — archive_cmd compile errors, farm_cmd
  async issues, search_cmd missing types, and the preset-module doctest.
- [x] `build.rs` minimised to `fn main() {}` after workspace-level linker
  script (`.cargo/config.toml`) took over glibc 2.38+ `__isoc23_*` symbol
  compat (keeps crate-specific build config available for the future).
- [x] WASM / non-CLI surface explicitly NOT linked — the crate only compiles
  for native targets by design; `oximedia-wasm` handles the browser story.

---

## Enhancements `[ ]`

### Shell integration & packaging

- [ ] Generate `bash`/`zsh`/`fish`/`powershell`/`elvish` completions with
  `clap_complete` under a new `completion` subcommand or a `build.rs` hook.
- [ ] Generate a Unix man page (`oximedia.1`) for every top-level subcommand
  via `clap_mangen` and install it from the release pipeline.
- [ ] Add `oximedia doctor` — environment diagnostics (Rust toolchain, SIMD
  feature detection via `oximedia-simd::CpuFeatures`, wgpu adapter list, CUDA
  driver presence via `oxionnx-cuda::CudaContext::try_new()`, plugin search
  paths, config-dir discovery via `dirs`).
- [ ] `cargo-dist` / GitHub Actions pipeline producing signed pre-built binaries
  (x86_64-linux-gnu, x86_64-linux-musl, aarch64-linux, x86_64-darwin,
  aarch64-darwin, x86_64-windows-msvc) plus shasums and SBOMs.
- [ ] Platform packaging — Homebrew formula, Windows `winget`/Scoop manifest,
  Debian/Ubuntu `.deb`, RPM for Fedora/RHEL, Arch Linux AUR `PKGBUILD`.

### Output & piping

- [ ] Audit every `*_cmd.rs` that accepts `cli.json` and ensure it actually
  emits strict JSON on stdout with no colour codes, so it can be piped into
  `jq`. Candidates most in need: `scopes_cmd`, `graphics_cmd`, `timeline_cmd`,
  `workflow_cmd`, `vfx_cmd`, `review_cmd`, `collab_cmd`.
- [ ] Add NDJSON streaming output (`--ndjson`) for `probe`, `quality`,
  `loudness`, `monitor`, and long-running subcommands so downstream tooling
  can start processing before the run completes.
- [ ] Standardise exit codes across handlers (0 ok / 1 runtime error /
  2 invalid args / 3 integrity failure / 4 patent-codec refusal) — today
  everything collapses to 1 via `std::process::exit(1)` in `main.rs`.
- [ ] `--log-format json` option that wires `tracing-subscriber`'s JSON layer
  for machine-readable logs in server deployments.

### `oximedia-ff` / `ffcompat` coverage

- [ ] Expand the FFmpeg flag surface handled by `oximedia-compat-ffmpeg`:
  `-filter_complex`, `-map_metadata`, `-ss` combined with `-to`, input-side
  `-f` format forcing, `-hide_banner`, `-stats`, `-progress`, `-nostats`,
  `-loglevel`, `-threads`, `-g`/`-keyint_min`, hardware accel flags (`-hwaccel`).
- [ ] Document and test the 75+ codec / 30 format mappings exposed by
  `oximedia-compat-ffmpeg` against real-world FFmpeg invocations.
- [ ] Add an `oximedia-ff --explain <args>` mode that prints the translation
  table (input flag → OxiMedia option) without executing.
- [ ] Ship an opt-in `oximedia-cv2` companion binary that layers the existing
  `oximedia-compat-cv2` (BGR ordering, 150+ OpenCV constants) in the same way
  `oximedia-ff` layers on `oximedia-compat-ffmpeg`.

### TUI polish (`tui_cmd.rs`)

- [ ] Replace the placeholder file-info string with a real mini-probe render
  (resolution, codec, duration, bitrate) by calling into the existing
  `handlers::probe_file` path.
- [ ] Add an actionable "run command" tab — currently `Commands` tab only lists
  names + descriptions; letting the user prefill arguments and dispatch into
  the real subcommand would close the loop.
- [ ] Mouse support (crossterm `EnableMouseCapture`), PgUp/PgDn for long lists,
  `/` for incremental search.
- [ ] Persist cwd navigation so `Enter` on a directory descends into it.

### Subcommand-level gaps (direct `grep` findings)

- [ ] `captions_cmd.rs:114` — `generate` currently writes a placeholder caption
  track; wire to the real ASR pipeline (OxiMedia has speech alignment in
  `oximedia-caption-gen`).
- [ ] `restore_cmd.rs:80,210` — audio path assumes raw PCM f32 LE; video path
  writes the input through unchanged. Wire to `oximedia-restore` decoders.
- [ ] `conform_cmd.rs:183` — copy-without-transform stub; implement the
  conform transforms (frame-rate, colour-space, loudness) end-to-end.
- [ ] `cloud_cmd.rs:151` — cost estimates use a simplified model; query live
  provider APIs (S3, GCS, R2, B2) via `oximedia-storage`.
- [ ] `scopes_cmd.rs:283,838` — the scope renderer currently synthesises a
  gradient placeholder frame when the full demux/decode chain is unavailable;
  wire to `oximedia-container` + `oximedia-codec` for real pixel data.
- [ ] `normalize_cmd.rs:282,288` — two-pass loudness processor writes a stub
  marker on pass 2; thread real decoded audio through and encode the output.
- [ ] `proxy_cmd.rs:302` — proxy generation writes a placeholder file; invoke
  the real low-bitrate transcode path.
- [ ] `metadata.rs:353` — timestamp formatter is a seconds-since-epoch stub;
  format via `chrono` (already a dep) as RFC 3339.
- [ ] `timeline_cmd.rs:763,966` — `generate_otio_placeholder` is the fallback
  when the real OTIO serializer isn't selected; fill in full OTIO coverage.
- [ ] `extract.rs:296` — all frame extracts currently fall back to PPM as a
  lossless stand-in; finish the PNG/JPEG output encoders.

### Testing

- [ ] Crate-level `tests/` directory — none exists yet. Add integration tests
  that spawn each `oximedia <subcommand> --help` to catch clap regressions
  and `oximedia <subcommand> --json` smoke tests on tiny fixture files.
- [ ] End-to-end test for `oximedia-ff` against a golden set of real FFmpeg
  command lines (both passing and patent-codec-substituted).
- [ ] Snapshot tests for `oximedia probe --format json` on fixtures shipped
  by `oximedia-io`.
- [ ] `assert_cmd` + `predicates` based CLI test harness that exercises
  every top-level variant of `Commands` at least once.
- [ ] Add an `examples/` directory demonstrating shell-script workflows that
  chain subcommands (transcode → loudness check → package → upload via cloud).

### Performance & UX

- [ ] Lazy subcommand loading — the single `main.rs` match is 500+ arms; as
  the CLI grows, consider switching to `clap_derive` with `#[command(flatten)]`
  into per-domain enums so `--help` per-domain is faster to render.
- [ ] Colour-aware `--no-color` auto-detection using the `CLICOLOR` /
  `NO_COLOR` / `TERM=dumb` environment variables (colored already does some
  of this — double-check the `--no-color` flag still wins).
- [ ] Parallelise `batch` using `rayon` work-stealing across files rather than
  the current round-robin; respect `-j 0` as "num_cpus".
- [ ] Streaming `--progress json` output suitable for `ffmpeg-progress-yuv`
  style consumers and GUI frontends.

### Documentation

- [ ] Expand the crate-level doc comment in `main.rs` (currently ~20 lines of
  examples) to cover the 85-subcommand taxonomy.
- [ ] Generate a static HTML reference from clap `--help` output and publish
  to `docs.oximedia.rs` / GitHub Pages.
- [ ] Document the plugin search paths honoured by `plugin_cmd` and the
  format of `$OXIMEDIA_PRESET_DIR` / `$OXIMEDIA_PLUGIN_DIR`.

---

## Known Issues / Gaps

- The `monitor_cmd.rs` module (430 lines) exists on disk but the `Monitor`
  variant is dispatched via `handlers::handle_monitor_command`. Audit whether
  the handler fully delegates into `monitor_cmd` or still carries duplicate
  logic in `handlers.rs` (which is 829 lines — at the 2000-line refactor
  boundary but worth splitting sooner).
- `commands.rs` at 1442 lines concentrates all clap derives; consider an
  `splitrs`-style split by domain (broadcast/mam/post/audio/video) to stay
  well under the 2000-line refactor ceiling as more subcommands are added.
- Placeholder/stub behaviours enumerated above under "Subcommand-level gaps"
  — each one is a working code path but short of the ecosystem's
  production-grade capability.
- No dedicated `tests/` directory; test coverage lives inside each
  `*_cmd.rs` via `#[cfg(test)]` modules. At least `loudness_cmd`, `batch_cmd`,
  `quality_cmd`, `normalize_cmd` have inline tests writing scratch files
  under `std::env::temp_dir()` per policy.

---

## Future (Post-0.2.0)

| Item                                | Rationale                                                  |
|-------------------------------------|------------------------------------------------------------|
| Shell completions (bash/zsh/fish)   | Tab-complete on 85 subcommands is a huge discoverability win. |
| Man pages via `clap_mangen`         | Offline reference; required by most Linux distro policies. |
| Signed pre-built binaries           | `cargo-dist` + GitHub Actions for all Tier-1 targets.      |
| Homebrew / winget / apt / AUR       | Distribution-channel parity with FFmpeg / MPV.             |
| Full interactive TUI mode           | Run subcommands from inside the TUI, not just browse them. |
| Expanded `oximedia-ff` flag surface | Cover `-filter_complex`, `-progress`, `-hwaccel`, metadata. |
| `oximedia-cv2` companion binary     | Opt-in OpenCV drop-in layered on `oximedia-compat-cv2`.    |
| `oximedia doctor` diagnostic        | Single-command environment audit for support channels.    |
| NDJSON streaming output             | First-class machine-readable progress everywhere.          |
| `assert_cmd`-driven integration CI  | Guarantees every subcommand stays wired as features grow.  |
| Docs site (docs.oximedia.rs)        | Auto-generated from clap metadata on every release.        |
| Crate split under `commands.rs`     | Keep files < 2000 SLOC as the subcommand count climbs.     |

---

*Last updated: 2026-04-15 — v0.1.3, oximedia-cli summary*
