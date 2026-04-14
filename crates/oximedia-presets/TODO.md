# oximedia-presets TODO

## Current Status
- 30 modules covering comprehensive encoding preset library with 200+ presets
- Categories: platform (YouTube, Vimeo, Facebook, Instagram, TikTok, Twitter, LinkedIn), broadcast (ATSC, DVB, ISDB), streaming (HLS, DASH, Smooth, RTMP, SRT), archive (lossless, mezzanine), mobile (iOS, Android), web (HTML5, progressive), social (stories, reels, feed), quality tiers, codec profiles (AV1, VP9, VP8, Opus, H.264, HEVC)
- Key types: PresetLibrary, PresetRegistry, PresetMetadata, Preset, AbrLadder, OptimalPreset, BitrateRange
- Additional modules: preset_benchmark, preset_chain, preset_diff, preset_export, preset_import, preset_manager, preset_metadata, preset_override, preset_resolver, preset_scoring, preset_tags, preset_versioning, color_preset, delivery_preset, ingest_preset

## Enhancements
- [x] Add preset inheritance in `PresetLibrary` — derive presets from base presets with overrides
- [x] Implement `PresetRegistry` fuzzy search (Levenshtein distance) for typo-tolerant lookups
- [x] Extend `OptimalPreset::select()` to consider resolution and frame rate, not just bitrate
- [ ] Add platform spec auto-update mechanism in `platform` modules (fetch latest requirements)
- [x] Implement preset compatibility matrix — check if source media matches preset requirements
- [x] Extend `preset_chain` to validate chained preset compatibility (output format of N matches input of N+1)
- [x] Add `preset_scoring` weight customization per use-case (latency-sensitive vs quality-sensitive)

## New Features
- [x] Add Twitch streaming presets in `platform` module (low-latency, different ingest servers)
- [x] Implement per-scene adaptive preset selection based on content complexity analysis
- [x] Add AV1 film grain synthesis presets for archival/restoration workflows
- [x] Implement preset A/B comparison tool in `preset_benchmark` (encode same source, compare metrics)
- [x] Add FLAC/Opus audio-only presets for podcast and music distribution
- [x] Implement preset recommendation from source media analysis (resolution, noise, motion)
- [x] Add Cinema DCP (Digital Cinema Package) presets for theatrical distribution
- [x] Implement user preset sharing via import/export with signature verification

## Performance
- [x] Lazy-load preset modules — only instantiate presets for requested categories (`LazyPresetCategory` with `OnceLock`)
- [x] Cache `PresetLibrary::new()` initialization since it loads all 200+ presets eagerly (`PresetLibrary::global()` singleton via `OnceLock`)
- [x] Optimize `PresetLibrary::search()` with a pre-built inverted index on name/description tokens
- [x] Use `Arc<Preset>` in `PresetRegistry` to avoid cloning preset configs during lookup

## Testing
- [x] Add validation tests ensuring all platform presets meet their respective platform requirements (wave3_tests.rs)
- [x] Test `OptimalPreset` selection with edge cases (zero bitrate, u64::MAX bitrate) (wave3_tests.rs)
- [x] Add round-trip tests for `preset_export` and `preset_import` (export -> import -> compare) (wave3_tests.rs)
- [x] Test `AbrLadder` generation for all streaming protocols with expected rung counts (wave3_tests.rs)
- [x] Verify `preset_diff` correctly identifies all parameter changes between preset versions (wave3_tests.rs)

## Documentation
- [ ] Add preset selection guide — decision tree for choosing the right preset by use case
- [ ] Document ABR ladder design principles with recommended bitrate/resolution combinations
- [ ] Add platform-specific encoding guidelines with links to official specifications
