//! LL-DASH chunked transfer encoding configuration and assembly.
//!
//! Low-Latency DASH uses chunked transfer encoding to deliver partial segments
//! before the full segment duration has elapsed.  This module provides:
//!
//! - [`LlDashStreamConfig`] — millisecond-granularity latency parameters
//! - [`DashChunk`] — a single partial-segment data unit
//! - [`ChunkedSegmentAssembler`] — server-side chunk accumulator that
//!   detects when all chunks for a segment have arrived
//! - [`DashRepresentation`] — bandwidth/codec descriptor for MPD generation
//! - [`generate_ll_dash_mpd`] — produce a minimal LL-DASH MPD XML string
//!
//! # Relation to `dash::ll_dash`
//!
//! The existing [`crate::dash::ll_dash`] module operates at the
//! CMAF/second-resolution level and manages a sliding window of full segments.
//! This module works at the *millisecond* level and focuses on the chunked
//! delivery format exposed via HTTP/2 chunked transfer encoding.

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

// ─── Configuration ────────────────────────────────────────────────────────────

/// LL-DASH chunked transfer encoding configuration.
///
/// All time values are in **milliseconds** for fine-grained latency control.
#[derive(Debug, Clone)]
pub struct LlDashStreamConfig {
    /// Total duration of one segment in milliseconds (typically 2000 ms).
    pub segment_duration_ms: u64,
    /// Duration of each chunk in milliseconds (typically 500 ms).
    ///
    /// Must divide `segment_duration_ms` evenly.
    pub chunk_duration_ms: u64,
    /// How early (in ms) chunks are made available before segment end.
    ///
    /// Should be less than `segment_duration_ms`.
    pub availability_time_offset_ms: u64,
    /// CDN/origin base URL for segment requests.
    pub service_location: String,
    /// Target end-to-end latency in milliseconds (typically 3000 ms).
    pub target_latency_ms: u64,
}

impl LlDashStreamConfig {
    /// Standard LL-DASH configuration suitable for live broadcast.
    ///
    /// segment=2000 ms, chunk=500 ms, ATO=1800 ms, latency=3000 ms.
    #[must_use]
    pub fn default_config() -> Self {
        Self {
            segment_duration_ms: 2000,
            chunk_duration_ms: 500,
            availability_time_offset_ms: 1800,
            service_location: String::new(),
            target_latency_ms: 3000,
        }
    }

    /// Ultra-low-latency configuration for real-time or interactive media.
    ///
    /// segment=1000 ms, chunk=250 ms, ATO=750 ms, latency=1500 ms.
    #[must_use]
    pub fn ultra_low_latency() -> Self {
        Self {
            segment_duration_ms: 1000,
            chunk_duration_ms: 250,
            availability_time_offset_ms: 750,
            service_location: String::new(),
            target_latency_ms: 1500,
        }
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - `chunk_duration_ms` is zero
    /// - `chunk_duration_ms` does not evenly divide `segment_duration_ms`
    /// - `availability_time_offset_ms` ≥ `segment_duration_ms`
    pub fn validate(&self) -> Result<(), String> {
        if self.chunk_duration_ms == 0 {
            return Err("chunk_duration_ms must be greater than zero".to_owned());
        }
        if self.segment_duration_ms % self.chunk_duration_ms != 0 {
            return Err(format!(
                "chunk_duration_ms ({}) must evenly divide segment_duration_ms ({})",
                self.chunk_duration_ms, self.segment_duration_ms
            ));
        }
        if self.availability_time_offset_ms >= self.segment_duration_ms {
            return Err(format!(
                "availability_time_offset_ms ({}) must be less than segment_duration_ms ({})",
                self.availability_time_offset_ms, self.segment_duration_ms
            ));
        }
        Ok(())
    }

    /// Returns how many chunks make up one complete segment.
    #[must_use]
    pub fn expected_chunks_per_segment(&self) -> u32 {
        if self.chunk_duration_ms == 0 {
            return 1;
        }
        (self.segment_duration_ms / self.chunk_duration_ms) as u32
    }
}

// ─── DashChunk ────────────────────────────────────────────────────────────────

/// A single partial-segment chunk delivered via chunked transfer encoding.
#[derive(Debug, Clone)]
pub struct DashChunk {
    /// Index of the parent segment (monotonically increasing).
    pub segment_index: u64,
    /// Index of this chunk within the segment (0-based).
    pub chunk_index: u32,
    /// Raw payload bytes (CMAF `moof`+`mdat` or similar).
    pub data: Vec<u8>,
    /// Whether this is the last chunk in the segment.
    pub is_final: bool,
    /// Nominal duration of this chunk in milliseconds.
    pub duration_ms: u64,
}

impl DashChunk {
    /// Create a new chunk.  `is_final` defaults to `false`.
    #[must_use]
    pub fn new(segment_index: u64, chunk_index: u32, data: Vec<u8>, duration_ms: u64) -> Self {
        Self {
            segment_index,
            chunk_index,
            data,
            is_final: false,
            duration_ms,
        }
    }

    /// Mark this chunk as the final chunk of the segment.
    #[must_use]
    pub fn mark_final(mut self) -> Self {
        self.is_final = true;
        self
    }
}

// ─── ChunkedSegmentAssembler ──────────────────────────────────────────────────

/// Server-side assembler that accumulates [`DashChunk`]s and detects
/// when a complete segment has been received.
///
/// Tracks multiple in-flight segments simultaneously to handle out-of-order
/// chunk delivery gracefully.
pub struct ChunkedSegmentAssembler {
    config: LlDashStreamConfig,
    /// Map from segment_index → ordered chunks received so far.
    segments: HashMap<u64, Vec<DashChunk>>,
}

impl ChunkedSegmentAssembler {
    /// Create a new assembler using the given configuration.
    #[must_use]
    pub fn new(config: LlDashStreamConfig) -> Self {
        Self {
            config,
            segments: HashMap::new(),
        }
    }

    /// Add a chunk.  Returns `true` if the segment is now complete.
    ///
    /// A segment is considered complete when either:
    /// - `chunk.is_final` is `true`, **or**
    /// - the number of received chunks equals `expected_chunks_per_segment()`.
    pub fn add_chunk(&mut self, chunk: DashChunk) -> bool {
        let seg_idx = chunk.segment_index;
        let is_final = chunk.is_final;
        let entry = self.segments.entry(seg_idx).or_default();
        entry.push(chunk);
        let expected = self.config.expected_chunks_per_segment() as usize;
        is_final || entry.len() >= expected
    }

    /// Returns `true` if all chunks for `segment_index` have been received.
    #[must_use]
    pub fn is_segment_complete(&self, segment_index: u64) -> bool {
        let expected = self.config.expected_chunks_per_segment() as usize;
        match self.segments.get(&segment_index) {
            None => false,
            Some(chunks) => chunks.len() >= expected || chunks.iter().any(|c| c.is_final),
        }
    }

    /// Assemble all chunks for `segment_index` into a contiguous byte buffer.
    ///
    /// Returns `None` if no chunks for that segment have been received.
    #[must_use]
    pub fn get_segment_data(&self, segment_index: u64) -> Option<Vec<u8>> {
        let chunks = self.segments.get(&segment_index)?;
        if chunks.is_empty() {
            return None;
        }
        // Sort by chunk_index to ensure correct ordering.
        let mut ordered: Vec<&DashChunk> = chunks.iter().collect();
        ordered.sort_by_key(|c| c.chunk_index);
        let total: usize = ordered.iter().map(|c| c.data.len()).sum();
        let mut buf = Vec::with_capacity(total);
        for c in ordered {
            buf.extend_from_slice(&c.data);
        }
        Some(buf)
    }

    /// Returns the number of chunks received so far for `segment_index`.
    #[must_use]
    pub fn chunks_received(&self, segment_index: u64) -> usize {
        self.segments.get(&segment_index).map_or(0, Vec::len)
    }

    /// Returns the expected number of chunks per segment based on the config.
    #[must_use]
    pub fn expected_chunks_per_segment(&self) -> u32 {
        self.config.expected_chunks_per_segment()
    }
}

// ─── DashRepresentation ──────────────────────────────────────────────────────

/// A DASH representation descriptor for MPD generation.
#[derive(Debug, Clone)]
pub struct DashRepresentation {
    /// Unique representation ID string.
    pub id: String,
    /// Bandwidth in bits per second.
    pub bandwidth_bps: u64,
    /// Frame/video width in pixels.
    pub width: u32,
    /// Frame/video height in pixels.
    pub height: u32,
    /// RFC 6381 codec string, e.g. `"avc1.640028"` or `"av01.0.08M.08"`.
    pub codecs: String,
}

// ─── MPD generation ───────────────────────────────────────────────────────────

/// Generate a minimal LL-DASH MPD XML document.
///
/// The MPD is of type `dynamic`, includes a `<ServiceDescription>` with
/// `<Latency>` and `<PlaybackRate>` elements, and a `<SegmentTemplate>` with
/// `availabilityTimeOffset` set from the config.
///
/// `duration_secs` is the total presentation duration (use `0.0` for live
/// content with an unknown end time — the field is then omitted).
#[must_use]
pub fn generate_ll_dash_mpd(
    config: &LlDashStreamConfig,
    duration_secs: f64,
    representations: &[DashRepresentation],
) -> String {
    let mut xml = String::with_capacity(2048);

    let ato_secs = config.availability_time_offset_ms as f64 / 1000.0;
    let target_latency_ms = config.target_latency_ms;
    let seg_dur_secs = config.segment_duration_ms as f64 / 1000.0;
    let chunk_dur_secs = config.chunk_duration_ms as f64 / 1000.0;

    // XML declaration + MPD opening
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\"\n");
    xml.push_str("     xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n");
    xml.push_str("     type=\"dynamic\"\n");
    let _ = writeln!(xml, "     minimumUpdatePeriod=\"PT{chunk_dur_secs:.3}S\"");
    let _ = writeln!(xml, "     minBufferTime=\"PT{seg_dur_secs:.3}S\"");
    let _ = writeln!(
        xml,
        "     suggestedPresentationDelay=\"PT{:.3}S\"",
        target_latency_ms as f64 / 1000.0
    );

    // Optional media presentation duration (omit for indefinite live)
    if duration_secs > 0.0 {
        let _ = writeln!(
            xml,
            "     mediaPresentationDuration=\"PT{duration_secs:.3}S\""
        );
    }

    xml.push_str(
        "     profiles=\"urn:mpeg:dash:profile:isoff-live:2011,urn:mpeg:dash:profile:cmaf:2019\">\n",
    );

    // ServiceDescription element
    xml.push_str("  <ServiceDescription id=\"0\">\n");
    let _ = writeln!(
        xml,
        "    <Latency target=\"{target_latency_ms}\" min=\"{}\" max=\"{}\"/>",
        (target_latency_ms as f64 * 0.5) as u64,
        target_latency_ms * 3
    );
    xml.push_str("    <PlaybackRate min=\"0.96\" max=\"1.04\"/>\n");
    xml.push_str("  </ServiceDescription>\n");

    // Period
    xml.push_str("  <Period id=\"0\" start=\"PT0S\">\n");
    xml.push_str("    <AdaptationSet mimeType=\"video/mp4\" contentType=\"video\">\n");

    // SegmentTemplate with availabilityTimeOffset
    let _ = writeln!(
        xml,
        "      <SegmentTemplate timescale=\"90000\"\n         media=\"segment_$Number$.m4s\"\n         initialization=\"init.mp4\"\n         availabilityTimeOffset=\"{ato_secs:.3}\">"
    );
    xml.push_str("        <SegmentTimeline/>\n");
    xml.push_str("      </SegmentTemplate>\n");

    // Representations
    for rep in representations {
        let _ = writeln!(
            xml,
            "      <Representation id=\"{}\" bandwidth=\"{}\" width=\"{}\" height=\"{}\" codecs=\"{}\"/>",
            rep.id, rep.bandwidth_bps, rep.width, rep.height, rep.codecs
        );
    }

    xml.push_str("    </AdaptationSet>\n");
    xml.push_str("  </Period>\n");
    xml.push_str("</MPD>\n");

    xml
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> LlDashStreamConfig {
        LlDashStreamConfig::default_config()
    }

    // 1. Validate catches bad chunk/segment ratio
    #[test]
    fn test_validate_bad_ratio() {
        let mut cfg = default_cfg();
        cfg.chunk_duration_ms = 300; // 2000 / 300 is not integer
        assert!(cfg.validate().is_err());
    }

    // 2. Validate catches ATO >= segment
    #[test]
    fn test_validate_ato_too_large() {
        let mut cfg = default_cfg();
        cfg.availability_time_offset_ms = 2000; // equal to segment_duration_ms
        assert!(cfg.validate().is_err());
    }

    // 3. Validate catches zero chunk duration
    #[test]
    fn test_validate_zero_chunk() {
        let mut cfg = default_cfg();
        cfg.chunk_duration_ms = 0;
        assert!(cfg.validate().is_err());
    }

    // 4. expected_chunks_per_segment correct
    #[test]
    fn test_expected_chunks_per_segment() {
        let cfg = default_cfg(); // 2000 / 500 = 4
        assert_eq!(cfg.expected_chunks_per_segment(), 4);
    }

    // 5. ultra_low_latency chunks per segment
    #[test]
    fn test_ull_chunks_per_segment() {
        let cfg = LlDashStreamConfig::ultra_low_latency(); // 1000 / 250 = 4
        assert_eq!(cfg.expected_chunks_per_segment(), 4);
    }

    // 6. add_chunk completes segment on final flag
    #[test]
    fn test_add_chunk_completes_on_final() {
        let cfg = default_cfg();
        let mut asm = ChunkedSegmentAssembler::new(cfg);
        let chunk = DashChunk::new(0, 0, vec![1, 2, 3], 500).mark_final();
        let done = asm.add_chunk(chunk);
        assert!(done);
    }

    // 7. add_chunk completes segment when expected count reached
    #[test]
    fn test_add_chunk_completes_on_count() {
        let cfg = default_cfg(); // expects 4 chunks
        let mut asm = ChunkedSegmentAssembler::new(cfg);
        for i in 0..3u32 {
            let done = asm.add_chunk(DashChunk::new(1, i, vec![i as u8], 500));
            assert!(!done, "should not be done after chunk {i}");
        }
        let done = asm.add_chunk(DashChunk::new(1, 3, vec![3], 500));
        assert!(done);
    }

    // 8. get_segment_data assembles chunks in order
    #[test]
    fn test_get_segment_data_assembles() {
        let cfg = default_cfg();
        let mut asm = ChunkedSegmentAssembler::new(cfg);
        // Push out of order
        asm.add_chunk(DashChunk::new(2, 1, vec![0xBB], 500));
        asm.add_chunk(DashChunk::new(2, 0, vec![0xAA], 500));
        let data = asm.get_segment_data(2).expect("should have data");
        assert_eq!(data, vec![0xAA, 0xBB]);
    }

    // 9. is_segment_complete
    #[test]
    fn test_is_segment_complete() {
        let cfg = default_cfg();
        let mut asm = ChunkedSegmentAssembler::new(cfg);
        assert!(!asm.is_segment_complete(0));
        for i in 0..4u32 {
            asm.add_chunk(DashChunk::new(0, i, vec![i as u8], 500));
        }
        assert!(asm.is_segment_complete(0));
    }

    // 10. chunks_received tracks per-segment
    #[test]
    fn test_chunks_received() {
        let cfg = default_cfg();
        let mut asm = ChunkedSegmentAssembler::new(cfg);
        assert_eq!(asm.chunks_received(5), 0);
        asm.add_chunk(DashChunk::new(5, 0, vec![1], 500));
        asm.add_chunk(DashChunk::new(5, 1, vec![2], 500));
        assert_eq!(asm.chunks_received(5), 2);
    }

    // 11. Multiple segments tracked independently
    #[test]
    fn test_multiple_segments() {
        let cfg = default_cfg();
        let mut asm = ChunkedSegmentAssembler::new(cfg);
        asm.add_chunk(DashChunk::new(0, 0, vec![0xAA], 500));
        asm.add_chunk(DashChunk::new(1, 0, vec![0xBB], 500));
        assert_eq!(asm.chunks_received(0), 1);
        assert_eq!(asm.chunks_received(1), 1);
    }

    // 12. MPD contains availabilityTimeOffset
    #[test]
    fn test_mpd_contains_ato() {
        let cfg = default_cfg(); // ATO = 1800 ms = 1.800 s
        let reps = vec![DashRepresentation {
            id: "1".to_owned(),
            bandwidth_bps: 2_000_000,
            width: 1920,
            height: 1080,
            codecs: "avc1.640028".to_owned(),
        }];
        let mpd = generate_ll_dash_mpd(&cfg, 0.0, &reps);
        assert!(mpd.contains("availabilityTimeOffset=\"1.800\""));
    }

    // 13. MPD contains ServiceDescription
    #[test]
    fn test_mpd_contains_service_description() {
        let cfg = default_cfg();
        let mpd = generate_ll_dash_mpd(&cfg, 0.0, &[]);
        assert!(mpd.contains("ServiceDescription"));
        assert!(mpd.contains("Latency"));
        assert!(mpd.contains("PlaybackRate"));
    }

    // 14. chunk mark_final sets flag
    #[test]
    fn test_chunk_mark_final() {
        let c = DashChunk::new(0, 0, vec![], 500).mark_final();
        assert!(c.is_final);
    }

    // 15. assembler expected_chunks_per_segment delegates to config
    #[test]
    fn test_assembler_expected_chunks() {
        let cfg = LlDashStreamConfig::ultra_low_latency();
        let asm = ChunkedSegmentAssembler::new(cfg);
        assert_eq!(asm.expected_chunks_per_segment(), 4);
    }
}
