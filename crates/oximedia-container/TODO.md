# oximedia-container TODO

## Current Status
- 95+ source files; container format demuxing and muxing
- Demuxers: Matroska/WebM, Ogg, FLAC, WAV, MP4, MPEG-TS, SRT, WebVTT, Y4M
- Muxers: Matroska/WebM, Ogg, FLAC, WAV, MPEG-TS, Y4M, CMAF
- Features: format probing (magic bytes), seeking, metadata (read/write/edit), chapters (Matroska/MP4), attachments, cue points, streaming (HLS/DASH), fragmented MP4, PTS/DTS management, track selection
- Modules: demux/, mux/, metadata/, chapters/, fragment/, streaming/, data/ (telemetry/GPS), edit/, tracks/, attach/, cue/, etc.

## Enhancements
- [x] mp4-muxer — muxer exists at mux/mp4/ (5 files: basic.rs, simple.rs, mod.rs, facade.rs, writer.rs); fMP4 fragmented mode tracked by Wave 3 Slice D
- [ ] Extend `demux/matroska/parser.rs` with full support for Matroska v4 elements (Block Addition Mappings)
- [ ] Add sample-accurate seeking in `seek.rs` for all container formats (not just keyframe-based)
- [x] Extend `metadata/editor.rs` with batch tag operations (copy all tags between files)
- [ ] Improve `streaming/mux.rs` with CMAF low-latency chunked transfer encoding
- [ ] Add edit list support in `edit_list.rs` for gapless audio and trimmed video playback
- [x] Extend `demux/mpegts/` with SCTE-35 ad insertion marker parsing and mux emission
- [ ] Add `container_probe.rs` detailed stream analysis (bitrate distribution, keyframe interval)

## Wave 2 Progress (2026-04-17)
- [x] AVI container: muxer (mux/avi/writer.rs) + demuxer (demux/avi/reader.rs) implemented — MJPEG-only, ≤1 GB. Wave 2 Slice F (2026-04-17). Follow-ups: OpenDML >1GB, audio streams, non-MJPEG codecs.
- [x] MKV MJPEG support: V_MJPEG codec ID now emitted correctly. Wave 2 Slice B.
- [x] MKV APV support: V_MS/VFW/FOURCC with BITMAPINFOHEADER CodecPrivate (biCompression=apv1). Wave 2 Slice B.

## Wave 3 Progress (2026-04-17)
- [x] AVI v3: OpenDML >1 GB super-index, PCM audio track, H264/RGB24 codec FourCCs — Slice C of /ultra Wave 3 (2026-04-17)
- [x] MP4 muxer gap-fill: fMP4 fragmented mode (moof+mdat+sidx), AV1 av1C, MJPEG/APV codec arms — Slice D of /ultra Wave 3 (2026-04-17)
- [x] Matroska+streaming: sample-accurate seek (Cues walk+decode-skip), gapless elst, DASH SegmentTemplate — Slice F of /ultra Wave 3 (2026-04-17)

## Wave 4 Progress (2026-04-18)
- [x] mkv-v4-blockadd: BlockAdditionMappingType 4 (HDR10+) + 5 (DV EL) emit+parse — Wave 4 Slice D
- [x] seek-sample-accurate-all: extend MKV pattern to MP4 (stss+stts+ctts) + AVI (idx1/super-index) — Wave 4 Slice D
- [x] cmaf-ll-chunked: CmafChunkedMode with moof+mdat per chunk, styp=cmfl, prft boxes — Wave 4 Slice D

## Wave 5 Progress (2026-04-18)
- [x] scte35-parse-emit: `parse_splice_info_section()` free function + `emit_time_signal/splice_null/splice_insert` — demux/mpegts/scte35 + mux/mpegts/scte35, re-exported from crate root — Wave 5 Slice D
- [x] metadata-batch: `BatchMetadataUpdate` builder + `BatchResult` in `metadata/batch.rs`, re-exported from crate root — Wave 5 Slice D
- [x] integration tests: `tests/scte35_parse.rs` (9 tests) + `tests/metadata_batch.rs` (12 tests) smoke-testing public re-exports — Wave 5 Slice D

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
