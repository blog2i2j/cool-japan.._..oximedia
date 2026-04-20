# oximedia-compat-ffmpeg TODO

## Current Status
- 10 source files; FFmpeg CLI argument compatibility layer
- Key features: argument parsing (FfmpegArgs, GlobalOptions, InputSpec, OutputSpec), codec mapping (80+ mappings: libx264->av1, aac->opus, etc.), filter graph parsing, stream specifiers, diagnostics, translator (parse_and_translate -> TranslateResult)
- Modules: arg_parser, argument_builder, codec_map, codec_mapping, diagnostics, filter_graph, filter_lex, stream_spec, translator

## Enhancements
- [x] Extend `codec_map.rs` to cover all common FFmpeg codec aliases (e.g., h264_nvenc, hevc_amf -> av1 equivalents)
- [ ] Add support for `-filter_complex` multi-input/output filter graph parsing in `filter_lex.rs`
- [ ] Improve `diagnostics.rs` with suggestion-based error messages ("did you mean..." for mistyped codecs)
- [ ] Extend `stream_spec.rs` to handle complex stream specifiers like `0:v:0`, `0:a:#0x1100`
- [ ] Add `-map` flag with negative mapping support (e.g., `-map 0 -map -0:s`) in `arg_parser.rs`
- [ ] Implement `-ss` / `-to` / `-t` seeking/duration options in `arg_parser.rs`
- [ ] Add `-preset` / `-tune` / `-profile` translation in `codec_mapping.rs`
- [x] APV aliases added to codec_map.rs + codec_mapping.rs — Slice A of /ultra Wave 3 (2026-04-17) DONE
- [x] FFmpeg compat Wave 3: filter_complex, -map stream_spec, -ss/-to/-t, ffprobe output — Slice G of /ultra Wave 3 (2026-04-17)

## Wave 4 Progress (2026-04-18)
- [x] codec-map-cache: OnceLock singleton for codec_map + codec_mapping registries — Wave 4 Slice E
- [x] encoder-quality-args: -preset/-tune/-profile:v parsing → EncoderQualityOptions — Wave 4 Slice E
- [x] filter-shorthand: -vf/-af parsing → single-chain FilterGraph (reuses filter_complex AST) — Wave 4 Slice E
- [x] two-pass: -pass 1/-pass 2 → PassPhase::First/Second with JSON stats file — Wave 4 Slice E

## New Features
- [ ] Implement `ffprobe`-compatible output mode (JSON/XML/CSV format info)
- [ ] Add `-vf` / `-af` shorthand filter chain parsing alongside `-filter_complex`
- [ ] Implement batch mode translation for converting multiple files in one invocation
- [ ] Add `-movflags +faststart` and similar muxer option translation in `translator.rs`
- [ ] Implement `-hwaccel` option translation to OxiMedia GPU pipeline flags
- [ ] Add support for concat protocol (`concat:file1|file2`) and concat demuxer syntax
- [ ] Implement two-pass encoding translation (`-pass 1` / `-pass 2`) in `translator.rs`
- [ ] Add `-metadata` tag translation for title, artist, comment fields

## Performance
- [ ] Cache parsed codec map in `codec_map.rs` to avoid repeated HashMap construction
- [ ] Optimize `filter_lex.rs` parser with zero-copy string slicing instead of String allocation
- [ ] Pre-compile regex patterns in `arg_parser.rs` for repeated argument parsing

## Testing
- [ ] Add test suite covering 50+ real-world FFmpeg command lines and their expected translations
- [ ] Test `filter_lex.rs` with complex filter graphs (split, overlay, amix chains)
- [ ] Add round-trip test: build arguments with `argument_builder.rs`, parse back, verify equivalence
- [ ] Test diagnostic output formatting matches FFmpeg-style warning/error format
- [ ] Add fuzz testing for `arg_parser.rs` with random argument combinations
- [ ] Test codec mapping completeness: verify all patent-free codecs have FFmpeg aliases

## Documentation
- [ ] Add FFmpeg-to-OxiMedia command translation examples in crate docs
- [ ] Document supported and unsupported FFmpeg options with migration notes
- [ ] Add filter graph syntax reference showing supported filter names
