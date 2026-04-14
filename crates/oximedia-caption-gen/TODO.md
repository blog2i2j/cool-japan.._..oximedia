# oximedia-caption-gen TODO

## Current Status
- 4 modules: `alignment` (speech-to-caption), `line_breaking` (greedy + Knuth-Plass DP), `wcag` (WCAG 2.1 compliance), `diarization` (speaker metadata + crosstalk)
- Zero external dependencies beyond `thiserror` (pure Rust, minimal footprint)
- Exports: `WordTimestamp`, `TranscriptSegment`, `CaptionBlock`, `LineBreakConfig`, `WcagChecker`, `Speaker`, `SpeakerTurn`

## Enhancements
- [ ] Add word-level confidence scores to `alignment::WordTimestamp` for quality-aware caption display
- [ ] Extend `line_breaking::optimal_break` with language-aware hyphenation (CJK line break rules)
- [ ] Add `wcag` check for maximum simultaneous on-screen caption count (accessibility guideline)
- [ ] Improve `diarization::merge_consecutive_turns` with configurable silence gap threshold
- [ ] Add `alignment::split_long_segments` support for splitting at sentence boundaries (not just duration)
- [ ] Extend `line_breaking::LineBreakConfig` with maximum characters-per-line constraint
- [ ] Add reading speed validation for different target audiences (children vs adults) in `wcag`
- [ ] Improve `diarization::CrosstalkDetector` with adjustable overlap tolerance percentage

## New Features
- [x] Add `forced_narrative` module for identifying and marking forced narrative subtitles (FN/SDH)
- [x] Implement `caption_format_adapter` module converting `CaptionBlock` to SRT/VTT/TTML output strings
- [ ] Add `punctuation_restoration` module for adding punctuation to raw ASR transcript output
- [ ] Implement `language_detect` module to auto-detect transcript language for locale-aware line breaking
- [x] Add `caption_timing_adjuster` module for shifting/stretching caption timings to match edited video
- [ ] Implement `caption_diff` module for comparing two caption tracks and highlighting differences
- [ ] Add `style_generator` module suggesting font size, position, colors based on video content analysis
- [ ] Implement `multi_language` module for bilingual caption layout (primary + secondary language)

## Performance
- [ ] Optimize `optimal_break` DP algorithm with Knuth-Plass SMAWK speedup for O(n) line breaking
- [ ] Add batch processing API to `align_to_frames` for processing multiple segments in one call
- [ ] Cache CPS (characters-per-second) computation results in `line_breaking` for repeated re-breaks
- [ ] Use string interning for `Speaker` labels in `diarization` to reduce allocation in large transcripts

## Testing
- [ ] Add test for `alignment::build_caption_blocks` with overlapping word timestamps
- [ ] Test `line_breaking::optimal_break` against known Knuth-Plass reference outputs
- [ ] Add WCAG compliance test suite with real-world caption samples that pass/fail each criterion
- [ ] Test `diarization::assign_speakers_to_blocks` with 5+ simultaneous speakers
- [ ] Add property-based test: `merge_short_segments` output has no segment shorter than min duration
- [ ] Test `greedy_break` vs `optimal_break` produce identical results for single-line captions
- [ ] Add round-trip test: split then merge segments should preserve total text content

## Documentation
- [ ] Document WCAG 2.1 conformance levels (A, AA, AAA) and which checks map to each level
- [ ] Add example showing complete pipeline: word timestamps -> alignment -> line breaking -> blocks
- [ ] Document `diarization` module usage with speaker identification metadata from ASR systems
