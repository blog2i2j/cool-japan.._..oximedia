# oximedia-container TODO

## Current Status
- 95+ source files; container format demuxing and muxing
- Demuxers: Matroska/WebM, Ogg, FLAC, WAV, MP4, MPEG-TS, SRT, WebVTT, Y4M
- Muxers: Matroska/WebM, Ogg, FLAC, WAV, MPEG-TS, Y4M, CMAF
- Features: format probing (magic bytes), seeking, metadata (read/write/edit), chapters (Matroska/MP4), attachments, cue points, streaming (HLS/DASH), fragmented MP4, PTS/DTS management, track selection
- Modules: demux/, mux/, metadata/, chapters/, fragment/, streaming/, data/ (telemetry/GPS), edit/, tracks/, attach/, cue/, etc.

## Enhancements
- [ ] Complete MP4 muxer (currently only demuxer exists in `demux/mp4/`; no `mux/mp4/`)
- [ ] Extend `demux/matroska/parser.rs` with full support for Matroska v4 elements (Block Addition Mappings)
- [ ] Add sample-accurate seeking in `seek.rs` for all container formats (not just keyframe-based)
- [ ] Extend `metadata/editor.rs` with batch tag operations (copy all tags between files)
- [ ] Improve `streaming/mux.rs` with CMAF low-latency chunked transfer encoding
- [ ] Add edit list support in `edit_list.rs` for gapless audio and trimmed video playback
- [ ] Extend `demux/mpegts/` with SCTE-35 ad insertion marker parsing
- [ ] Add `container_probe.rs` detailed stream analysis (bitrate distribution, keyframe interval)

## New Features
- [x] Implement CAF (Core Audio Format) demuxer/muxer for Apple ecosystem compatibility
- [ ] Add MPEG-DASH segment template generation in `streaming/`
- [ ] Implement sample grouping and random access point indexing in `sample_table.rs`
- [x] Add subtitle stream muxing support for Matroska (WebVTT/ASS embedded subtitles)
- [ ] Implement `timecode/track.rs` SMPTE timecode track reading for professional workflows
- [x] Add multi-angle support in Matroska via `tracks/selector.rs`
- [ ] Implement fragmented MP4 (fMP4) live ingest support in `fragment/`
- [ ] Add MKV attachment extraction and insertion CLI in `attach/matroska.rs`

## Performance
- [ ] Implement buffered async I/O in `demux/buffer.rs` with configurable read-ahead size
- [ ] Add memory-mapped I/O option for large file demuxing via `mmap` feature gate
- [ ] Optimize EBML variable-length integer parsing in `demux/matroska/ebml.rs` with lookup tables
- [ ] Cache cluster/cue positions in `cue/` for faster random-access seeking in Matroska
- [ ] Parallelize mux writing in `mux/matroska/writer.rs` for multi-stream interleaving
- [ ] Optimize `pts_dts.rs` timestamp rewriting with batch processing

## Testing
- [ ] Add conformance tests for Matroska demuxer against Matroska test suite files
- [ ] Test MP4 demuxer with fragmented MP4 (fMP4) and progressive MP4 variants
- [ ] Add round-trip test: demux -> mux -> demux -> verify packet-level equality for all formats
- [ ] Test MPEG-TS demuxer with real broadcast streams containing multiple programs
- [ ] Add stress test for `streaming/demux.rs` with simulated live stream input (continuous data)
- [ ] Test `metadata/vorbis.rs` Vorbis comment handling with edge cases (empty tags, unicode)

## Documentation
- [ ] Document container format support matrix (which codecs in which containers)
- [ ] Add seeking strategy documentation (keyframe vs. sample-accurate vs. byte-offset)
- [ ] Document streaming output modes (HLS, DASH, CMAF) with configuration examples
