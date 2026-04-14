# oximedia — Facade Crate TODO

**Version: 0.1.3**
**Status as of: 2026-04-15**

The `oximedia` facade is the single cargo-add entry point into the entire OxiMedia
ecosystem. It re-exports approximately 103 workspace library crates behind
individual Cargo feature flags, groups them into feature-gated modules, provides
a curated `prelude` for common types, and anchors a shared integration test
suite that exercises cross-subsystem behaviour. The facade is the canonical way
for downstream users to consume OxiMedia: `oximedia = { version = "0.1.3",
features = ["..."] }` replaces pulling in dozens of sibling crates manually.

---

## Current Status

- **Source size**
  - `src/lib.rs` — 1451 lines (per-subsystem module declarations, feature-flag
    documentation table, always-on re-exports, `#![forbid(unsafe_code)]`,
    `#![warn(missing_docs)]`).
  - `src/prelude.rs` — 689 lines (most-used items per feature, hand-picked for
    each enabled subsystem).
  - `tests/integration.rs` — 1224 lines (cross-feature integration suite).
  - `Cargo.toml` — 610 lines (dependency block + features + `[[example]]`
    sections).
- **Feature flags** — 101 defined features in `Cargo.toml`:
  - `default = []` (lean build, core modules only).
  - 99 per-crate features of the form `<name> = ["dep:oximedia-<name>"]`
    gating a single workspace library crate each.
  - `full` meta-feature enabling every optional feature (one entry per crate;
    `full` is the only fan-out meta-feature).
- **Re-exported crates** — 103 workspace libraries:
  - 4 always-on: `oximedia-core`, `oximedia-io`, `oximedia-container`,
    `oximedia-cv` (exposed via `lib.rs` plus `pub use oximedia_cv as cv;`).
  - 99 feature-gated optional crates, each in its own `#[cfg(feature = "…")]
    pub mod <name> { pub use oximedia_<name>::*; }` block.
- **Examples** (17, all registered in `Cargo.toml` `[[example]]` tables and
  living in `/Users/kitasan/work/oximedia/examples/`):
  - Always-on: `probe_file`, `corner_detection`, `optical_flow`,
    `face_detection`, `image_processing`, `decode_video`.
  - Feature-gated: `audio_metering` (`metering`), `quality_assessment`
    (`quality`), `timecode_operations` (`timecode`), `dedup_detection`
    (`dedup`), `workflow_pipeline` (`workflow`), `video_scopes` (`scopes`),
    `shot_detection` (`shots`), `nmos_registry` (`routing`),
    `color_pipeline` (`colormgmt` + `lut`),
    `media_pipeline` (`quality` + `metering` + `transcode` + `timecode` +
    `workflow` + `archive`),
    `nmos_server_demo` (`routing`).
- **Integration test categories** (`tests/integration.rs`) — 6 modules:
  - `core_tests` — always-compiled: `OxiError`/`OxiResult` construction,
    `probe_format` on Matroska, MP4, empty, and garbage buffers.
  - `quality_tests` — gated on `quality`: PSNR/SSIM/MS-SSIM on synthetic
    Gray8 and YUV420P frames, dimension-mismatch errors, no-reference blur,
    metric-type classification.
  - `timecode_tests` — gated on `timecode`: SMPTE 12M LTC/VITC round-trip,
    drop/non-drop-frame arithmetic at all standard rates.
  - `metering_tests` — gated on `metering`: EBU R128 integrated loudness,
    true-peak, loudness range.
  - `archive_tests` — gated on `archive`: checksum, fixity, verification config.
  - `combined_tests` — gated on `search` **and** `quality`: cross-subsystem
    faceted search filtered by quality scores.

---

## Completed

- [x] Facade crate stood up at workspace root (`/Users/kitasan/work/oximedia/oximedia/`).
- [x] Always-on core re-exports published: `CodecId`, `MediaType`, `OxiError`,
      `OxiResult`, `PixelFormat`, `Rational`, `SampleFormat`, `Timestamp`,
      `BitReader`, `FileSource`, `MediaSource`, `MemorySource`, `probe_format`,
      `ContainerFormat`, `Demuxer`, `Metadata`, `Packet`, `PacketFlags`,
      `ProbeResult`, `StreamInfo`, `CodecParams`, plus `pub use oximedia_cv as cv`.
- [x] 99 optional subsystem modules wired in `lib.rs`, each behind its own
      feature flag with a crate-level doc comment.
- [x] Feature-flag documentation table in the crate-level `//!` doc covers
      audio, video, graph, effects, net, metering, normalize, quality, metadata-ext,
      timecode, workflow, batch, monitor, lut, colormgmt, transcode, subtitle,
      captions, archive, dedup, search, mam, scene, shots, scopes, vfx, image-ext,
      watermark, mir, recommend, playlist, playout, rights, review, restore, repair,
      multicam, stabilize, cloud, edl, ndi, imf, aaf, timesync, forensics, accel,
      simd, switcher, timeline, optimize, profiler, renderfarm, storage, collab,
      gaming, virtual-prod, access, conform, convert, automation, clips, proxy,
      presets, calibrate, denoise, align, analysis, audiopost, qc, jobs, auto,
      edit, routing, audio-analysis, gpu, packager, drm, archive-pro, distributed,
      farm, dolbyvision, mixer, scaling, graphics, videoip, compat-ffmpeg, plugin,
      server, hdr, spatial, cache, stream, video-proc, cdn, neural, vr360,
      analytics, caption-gen.
- [x] `prelude` module with 689 lines of curated type imports per feature,
      including alias renames to avoid collisions (`AafEditRate` vs `ImfEditRate`,
      `BatchRetryPolicy` vs `WorkflowRetryPolicy`, `ConversionQualityMode` vs
      `QualityMode`, `FarmJobId` vs `RenderJobId`, `TimelineClip` vs `EditClip`,
      etc.).
- [x] `full` meta-feature that turns on every optional feature; matches the
      table in the crate doc.
- [x] 17 worked examples registered under `[[example]]` with correct
      `required-features` declarations for feature-gated builds.
- [x] 6-module integration test suite (`tests/integration.rs`) exercising
      always-on probing plus five feature-gated subsystems.
- [x] `#![forbid(unsafe_code)]` enforced; no `unwrap()` in facade source
      (prelude and module declarations are pure re-exports).
- [x] `dev-dependencies` limited to `tokio` (`macros` + `rt-multi-thread`),
      `serde_json`, and `uuid` — no heavyweight dev deps leaked into the facade.
- [x] Workspace compiles cleanly to `wasm32-unknown-unknown` when the facade is
      used with only always-on features (WASM-specific surface lives in the
      separate `oximedia-wasm` crate).

---

## Enhancements

### Feature flags and Cargo.toml

- [ ] Add an `image-transform` row to the feature-flag table in `src/lib.rs`
      crate-level doc (lines 36-136); the feature exists (`Cargo.toml` line 423,
      `lib.rs` line 1442) but is missing from the documented matrix.
- [ ] Add `oximedia-pipeline` to `Cargo.toml`, declare a `pipeline` feature
      flag, and expose `pub mod pipeline` in `lib.rs`; the crate is listed in
      project memory as a 0.1.2 addition but is not reachable through the
      facade.
- [ ] Define a `minimal` feature preset distinct from `default = []` that
      enables only `audio` + `video` + `metadata-ext` for quick-start users
      who need basic decoding without pulling the `full` tree.
- [ ] Introduce category meta-features to mirror the prelude grouping: e.g.
      `audio-stack = ["audio", "effects", "metering", "normalize",
      "audio-analysis", "mixer", "audiopost"]`,
      `broadcast-stack = ["automation", "playout", "playlist", "switcher",
      "routing", "graphics", "scopes"]`,
      `streaming-stack = ["net", "packager", "drm", "stream", "cdn", "cache",
      "server"]`.
- [ ] Document the implicit `normalize -> metering` activation
      (`Cargo.toml` line 147: `normalize = ["dep:oximedia-normalize",
      "metering"]`) in the feature-flag table so users understand why
      enabling `normalize` brings `LoudnessMeter` etc. into scope.
- [ ] Split optional dependency table and feature list into two halves of
      `Cargo.toml` with a divider comment so the file is easier to scan at
      610 lines.

### Prelude coverage

- [ ] Normalize the newer prelude entries (`prelude.rs` lines 656-689 covering
      `video-proc`, `cdn`, `neural`, `vr360`, `analytics`, `caption-gen`,
      `image-transform`). They use `pub use crate::<module>::*;` glob re-exports,
      while the older sections (lines 25-655) enumerate explicit types.
      Pick curated type sets to match the established API-surface contract.
- [ ] Re-export the always-on `oximedia_cv` top-level facade alias into the
      prelude (currently only `oximedia::cv::*` works; adding
      `pub use crate::cv as cv;` or re-exporting high-profile CV primitives
      would save one import for most users).
- [ ] Add explicit prelude re-exports for `oximedia-pipeline` once the crate
      is wired (see above): expected `PipelineBuilder`, `PipelineGraph`,
      `PipelineError`, `PipelineResult`.
- [ ] Audit alias names for alphabetisation consistency: the file currently
      mixes `ReviewError`/`ReviewResult` style with `Error as
      <Prefix>Error`/`Result as <Prefix>Result` patterns; standardise the two.

### Documentation and discoverability

- [ ] Write at least three runnable doctest blocks in `src/lib.rs` (probe +
      dedup, transcode + quality assessment, prelude quick-start) so that
      `cargo test --doc --features ...` exercises the public surface.
- [ ] Add a feature-matrix table to the crate README (not to this TODO) that
      cross-references features against subsystem crates, so users pick
      feature flags without reading `Cargo.toml`.
- [ ] Extend `cargo doc` cross-link coverage: every `pub mod <name>` in
      `lib.rs` should `[link]` to the underlying crate's top-level page so
      `rustdoc` navigation surfaces the child crate docs.
- [ ] Add a "Cookbook" section in the crate-level doc pointing to each of
      the 17 worked examples by filename and required feature flags.

### Integration tests

- [ ] Add integration-test modules for the 11 newest workspace crates
      (covered in `lib.rs` but untested in `integration.rs`):
      `hdr_tests` (HDR10+ SEI round-trip), `spatial_tests` (HOA encode/decode,
      HRTF binaural render), `cache_tests` (LRU eviction invariant, tiered
      promotion), `stream_tests` (ABR ladder switching, QoE health score),
      `video_proc_tests` (scene cut on synthetic stripe sequence, 3:2 pulldown
      detection), `cdn_tests` (edge selection under simulated latency matrix),
      `neural_tests` (conv2d output shape, scene classifier confidence),
      `vr360_tests` (equirectangular↔cubemap round-trip PSNR),
      `analytics_tests` (retention curve monotonicity, A/B significance),
      `caption_gen_tests` (Knuth-Plass line budget, WCAG 2.1 ≥4.5:1 contrast),
      `image_transform_tests` (resize + rotate + color convert identity).
- [ ] Grow the `combined_tests` module beyond `search` + `quality` to
      exercise other cross-feature paths: `transcode` + `normalize` + `qc`,
      `playlist` + `playout` + `automation`, `routing` + `videoip` + `ndi`.
- [ ] Add a compile-only test harness that iterates `cargo check
      --no-default-features --features <one-at-a-time>` for every feature,
      proving each flag is independently buildable.
- [ ] Add a single smoke test under `full` that imports `oximedia::prelude::*`
      and touches one symbol from each prelude section, so renames in child
      crates break CI immediately.

### Examples

- [ ] Add examples for the crates that lack one: `hdr_tone_map`,
      `spatial_binaural_render`, `abr_streaming`, `cdn_failover`,
      `neural_scene_classify`, `vr360_projection`, `analytics_session`,
      `caption_wcag_compliance`.
- [ ] Convert `media_pipeline.rs` (currently a single-run example) into a
      tutorial-style example with inline comments explaining each of its six
      required features.
- [ ] Add an `ffmpeg_translate_demo.rs` example gated on `compat-ffmpeg`
      that parses a real FFmpeg command-line and prints the translated
      `TranscodeConfig`.

---

## Known Issues / Gaps

- `image-transform` feature is documented by module (line 1442) and prelude
  (line 689) but is **not** listed in the feature-flag table in the crate-level
  `//!` doc that renders on docs.rs. The table jumps from `caption-gen` straight
  to `full`.
- The `prelude` is inconsistent: older sections list individual types while
  the last seven sections (`video-proc`, `cdn`, `neural`, `vr360`, `analytics`,
  `caption-gen`, `image-transform`) glob-re-export entire modules, making the
  prelude's API contract brittle (anything added to the child crate leaks in).
- `normalize` silently activates `metering` (the only compound non-`full`
  feature in the file). This is deliberate but undocumented in the feature
  table, and users enabling `normalize` will find `LoudnessMeter` etc. in
  scope without asking.
- `oximedia-pipeline` is listed in project memory as part of the 0.1.2
  additions but has no `Cargo.toml` entry, no feature flag, and no module —
  downstream users cannot reach it through the facade.
- Integration tests cover only 6 of 99 optional feature crates (≈6%).
  The other 93 feature-gated subsystems have no facade-level integration
  coverage, relying entirely on their home crates' test suites.
- `#![warn(missing_docs)]` is set but per-feature modules often have a single
  `//!` blurb and then `pub use oximedia_<name>::*;`, so the rendered facade
  docs are thin for downstream readers who land on `oximedia::<module>` first.

---

## Future (Post-0.2.0)

| Item | Target | Notes |
|------|--------|-------|
| Trim re-export surface | 0.2.0 | Move glob re-exports to curated per-type lists; freeze prelude as the stability boundary. |
| Category meta-features | 0.2.0 | `audio-stack`, `broadcast-stack`, `streaming-stack`, `post-stack`, `ml-stack` for faster cargo-add. |
| Per-feature `docs.rs` build matrix | 0.2.0 | Configure `[package.metadata.docs.rs]` with `all-features = true` plus a feature-subset CI job. |
| Workspace-wide semver check | 0.2.x | Automate `cargo-semver-checks` against each feature permutation on every release tag. |
| Facade benchmarks | 0.2.x | Criterion suite that wires several features together (decode → scale → encode) to detect regressions across crate boundaries. |
| `no_std` probing layer | 0.3.0 | Expose a strictly `no_std`/`alloc`-compatible subset of always-on re-exports for embedded pipelines. |
| Plugin-discovery helper | 0.3.0 | Surface `plugin::PluginRegistry` from the facade with auto-registration of workspace codec plugins. |
| WASM facade parity | 0.3.0 | Re-introduce a `wasm` feature that mirrors a curated subset of `oximedia-wasm` behind the same flag name. |

---

*Last updated: 2026-04-15 — v0.1.3, facade crate summary; 1451-line lib.rs, 689-line prelude.rs, 1224-line integration.rs, 610-line Cargo.toml; 101 features (99 per-crate + `default` + `full`); 103 re-exported crates (4 always-on + 99 optional); 17 registered examples; 6 integration test categories.*
