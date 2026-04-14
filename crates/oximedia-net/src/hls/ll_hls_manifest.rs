//! LL-HLS manifest builder for CMAF partial segments.
//!
//! This module provides higher-level manifest construction types that map
//! directly to the LL-HLS specification (Apple RFC 8216bis) focusing on
//! the partial segment delivery model:
//!
//! - [`PartialSegment`] — a sub-segment chunk with duration, URI and
//!   independent (keyframe) flag
//! - [`HlsSegment`] — a complete media segment carrying optional parts
//! - [`LlHlsManifest`] — a live LL-HLS media playlist that renders to M3U8
//!
//! The [`LlHlsManifest::build_m3u8`] method emits all required LL-HLS
//! extension tags in one call.
//!
//! # Example
//!
//! ```
//! use oximedia_net::hls::ll_hls_manifest::{
//!     HlsSegment, LlHlsManifest, PartialSegment,
//! };
//!
//! let mut manifest = LlHlsManifest::new(6.0, 0.2);
//!
//! let mut seg = HlsSegment::new(0, "seg0.ts", 6.0);
//! seg.parts.push(PartialSegment::new(0, 0, 200, "part0.mp4", true));
//! seg.parts.push(PartialSegment::new(0, 1, 200, "part1.mp4", false));
//! manifest.segments.push(seg);
//!
//! manifest.preload_hint = Some("part_next.mp4".to_owned());
//! let m3u8 = manifest.build_m3u8();
//! assert!(m3u8.contains("#EXT-X-SERVER-CONTROL"));
//! assert!(m3u8.contains("#EXT-X-PART-INF"));
//! assert!(m3u8.contains("#EXT-X-PRELOAD-HINT"));
//! ```

use std::fmt::Write as FmtWrite;

// ─── PartialSegment ───────────────────────────────────────────────────────────

/// A partial segment (sub-chunk) within an LL-HLS stream.
///
/// Corresponds to an `#EXT-X-PART` tag in the M3U8 playlist.
#[derive(Debug, Clone)]
pub struct PartialSegment {
    /// Media sequence number of the parent complete segment.
    pub sequence: u64,
    /// Zero-based index of this part within `sequence`.
    pub part_index: u32,
    /// Nominal duration of this part in milliseconds.
    pub duration_ms: u32,
    /// URI identifying the partial media resource.
    pub uri: String,
    /// Whether this part starts with a keyframe (IDR), enabling independent
    /// playback start from this point.
    pub independent: bool,
}

impl PartialSegment {
    /// Create a new partial segment descriptor.
    #[must_use]
    pub fn new(
        sequence: u64,
        part_index: u32,
        duration_ms: u32,
        uri: impl Into<String>,
        independent: bool,
    ) -> Self {
        Self {
            sequence,
            part_index,
            duration_ms,
            uri: uri.into(),
            independent,
        }
    }

    /// Duration of this part in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        self.duration_ms as f64 / 1000.0
    }

    /// Renders this part as an `#EXT-X-PART` tag line.
    #[must_use]
    pub fn to_tag(&self) -> String {
        let mut tag = format!(
            "#EXT-X-PART:DURATION={:.5},URI=\"{}\"",
            self.duration_secs(),
            self.uri
        );
        if self.independent {
            tag.push_str(",INDEPENDENT=YES");
        }
        tag
    }
}

// ─── HlsSegment ───────────────────────────────────────────────────────────────

/// A complete HLS media segment, optionally carrying partial sub-segments.
///
/// In LL-HLS mode, each complete segment is preceded by its `#EXT-X-PART`
/// lines so clients can start playback before the segment is fully available.
#[derive(Debug, Clone)]
pub struct HlsSegment {
    /// Monotonically increasing media sequence number.
    pub sequence: u64,
    /// URI of the complete segment media resource.
    pub uri: String,
    /// Duration of the complete segment in seconds (for `#EXTINF`).
    pub duration_secs: f64,
    /// Partial segments that make up this complete segment.
    pub parts: Vec<PartialSegment>,
}

impl HlsSegment {
    /// Create a new HLS segment descriptor.
    #[must_use]
    pub fn new(sequence: u64, uri: impl Into<String>, duration_secs: f64) -> Self {
        Self {
            sequence,
            uri: uri.into(),
            duration_secs,
            parts: Vec::new(),
        }
    }

    /// Renders all `#EXT-X-PART` tags followed by the `#EXTINF` + URI pair.
    #[must_use]
    pub fn to_tags(&self) -> String {
        let mut out = String::new();
        for part in &self.parts {
            let _ = writeln!(out, "{}", part.to_tag());
        }
        let _ = writeln!(out, "#EXTINF:{:.5},", self.duration_secs);
        let _ = writeln!(out, "{}", self.uri);
        out
    }
}

// ─── LlHlsManifest ────────────────────────────────────────────────────────────

/// An LL-HLS media playlist ready for M3U8 serialization.
///
/// Provides all required RFC 8216bis extension tags:
/// - `#EXT-X-SERVER-CONTROL` with `CAN-BLOCK-RELOAD=YES` and
///   `PART-HOLD-BACK` set to three times the part target duration
/// - `#EXT-X-PART-INF:PART-TARGET=…` derived from `part_target_duration_secs`
/// - One `#EXT-X-PART` tag per [`PartialSegment`] inside each [`HlsSegment`]
/// - An optional `#EXT-X-PRELOAD-HINT:TYPE=PART,URI="…"` at the end
///
/// # Server-Control
///
/// `can_skip_until` is emitted as `CAN-SKIP-UNTIL` when positive.  Setting it
/// to zero suppresses the attribute.
#[derive(Debug, Clone)]
pub struct LlHlsManifest {
    /// Complete segments in the sliding playlist window.
    pub segments: Vec<HlsSegment>,
    /// In-progress partial segments for the current open segment (not yet
    /// part of a complete segment).
    pub partial_segments: Vec<PartialSegment>,
    /// Optional URI of the next expected partial segment (preload hint).
    pub preload_hint: Option<String>,
    /// Delta playlist skip threshold in seconds (`CAN-SKIP-UNTIL`).
    /// A value ≤ 0.0 suppresses the attribute.
    pub can_skip_until: f32,

    // ── Internal timing parameters ────────────────────────────────────────
    /// Target duration of complete segments in seconds (`#EXT-X-TARGETDURATION`).
    target_duration_secs: f64,
    /// Nominal part duration in seconds used for `#EXT-X-PART-INF` and
    /// `PART-HOLD-BACK` calculation.
    part_target_duration_secs: f64,
    /// First media sequence number in the window.
    media_sequence: u64,
}

impl LlHlsManifest {
    /// Create a new manifest with the given full-segment and part target durations.
    ///
    /// `target_duration_secs` — e.g., 6.0
    /// `part_target_duration_secs` — e.g., 0.2
    #[must_use]
    pub fn new(target_duration_secs: f64, part_target_duration_secs: f64) -> Self {
        Self {
            segments: Vec::new(),
            partial_segments: Vec::new(),
            preload_hint: None,
            can_skip_until: 0.0,
            target_duration_secs,
            part_target_duration_secs,
            media_sequence: 0,
        }
    }

    /// Set the media sequence number (first segment in the window).
    pub fn set_media_sequence(&mut self, seq: u64) {
        self.media_sequence = seq;
    }

    /// Returns `PART-HOLD-BACK` value: three times the part target duration.
    #[must_use]
    pub fn part_hold_back(&self) -> f64 {
        self.part_target_duration_secs * 3.0
    }

    /// Build the complete M3U8 playlist string.
    ///
    /// Emits all required LL-HLS extension tags in RFC 8216bis order:
    ///
    /// 1. `#EXTM3U` + `#EXT-X-VERSION:9`
    /// 2. `#EXT-X-TARGETDURATION`
    /// 3. `#EXT-X-MEDIA-SEQUENCE`
    /// 4. `#EXT-X-PART-INF:PART-TARGET=…`
    /// 5. `#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK=…[,CAN-SKIP-UNTIL=…]`
    /// 6. Complete segments with their `#EXT-X-PART` + `#EXTINF` tags
    /// 7. In-progress `#EXT-X-PART` tags for the current open segment
    /// 8. `#EXT-X-PRELOAD-HINT:TYPE=PART,URI="…"` (if set)
    #[must_use]
    pub fn build_m3u8(&self) -> String {
        let mut out = String::with_capacity(4096);

        // Header
        out.push_str("#EXTM3U\n");
        out.push_str("#EXT-X-VERSION:9\n");
        let _ = writeln!(
            out,
            "#EXT-X-TARGETDURATION:{}",
            self.target_duration_secs.ceil() as u64
        );
        let _ = writeln!(out, "#EXT-X-MEDIA-SEQUENCE:{}", self.media_sequence);

        // Part info
        let _ = writeln!(
            out,
            "#EXT-X-PART-INF:PART-TARGET={:.5}",
            self.part_target_duration_secs
        );

        // Server control
        let part_hold_back = self.part_hold_back();
        let mut sc = format!(
            "#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK={part_hold_back:.3}"
        );
        if self.can_skip_until > 0.0 {
            let _ = write!(sc, ",CAN-SKIP-UNTIL={:.1}", self.can_skip_until);
        }
        let _ = writeln!(out, "{sc}");

        // Complete segments
        for seg in &self.segments {
            out.push_str(&seg.to_tags());
        }

        // In-progress partial segments (current open segment)
        for part in &self.partial_segments {
            let _ = writeln!(out, "{}", part.to_tag());
        }

        // Preload hint at end of playlist
        if let Some(hint_uri) = &self.preload_hint {
            let _ = writeln!(
                out,
                "#EXT-X-PRELOAD-HINT:TYPE=PART,URI=\"{hint_uri}\""
            );
        }

        out
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manifest() -> LlHlsManifest {
        LlHlsManifest::new(6.0, 0.2)
    }

    // 1. PartialSegment duration_secs converts from ms correctly
    #[test]
    fn test_partial_segment_duration_secs() {
        let ps = PartialSegment::new(0, 0, 200, "part0.mp4", true);
        assert!((ps.duration_secs() - 0.2).abs() < 1e-9);
    }

    // 2. PartialSegment to_tag contains EXT-X-PART and DURATION
    #[test]
    fn test_partial_segment_to_tag_basic() {
        let ps = PartialSegment::new(1, 2, 200, "part2.mp4", false);
        let tag = ps.to_tag();
        assert!(tag.contains("#EXT-X-PART"));
        assert!(tag.contains("DURATION=0.20000"));
        assert!(tag.contains("part2.mp4"));
    }

    // 3. PartialSegment to_tag includes INDEPENDENT=YES when flagged
    #[test]
    fn test_partial_segment_independent_flag() {
        let ps = PartialSegment::new(0, 0, 200, "keypart.mp4", true);
        assert!(ps.to_tag().contains("INDEPENDENT=YES"));
    }

    // 4. PartialSegment without independent does not include INDEPENDENT
    #[test]
    fn test_partial_segment_no_independent() {
        let ps = PartialSegment::new(0, 1, 200, "p1.mp4", false);
        assert!(!ps.to_tag().contains("INDEPENDENT=YES"));
    }

    // 5. HlsSegment to_tags emits EXTINF and URI
    #[test]
    fn test_hls_segment_to_tags_extinf() {
        let seg = HlsSegment::new(0, "seg0.ts", 6.0);
        let tags = seg.to_tags();
        assert!(tags.contains("#EXTINF:6.00000,"));
        assert!(tags.contains("seg0.ts"));
    }

    // 6. HlsSegment to_tags emits parts before EXTINF
    #[test]
    fn test_hls_segment_parts_before_extinf() {
        let mut seg = HlsSegment::new(0, "seg0.ts", 6.0);
        seg.parts.push(PartialSegment::new(0, 0, 200, "p0.mp4", true));
        let tags = seg.to_tags();
        let part_pos = tags.find("#EXT-X-PART").expect("EXT-X-PART present");
        let extinf_pos = tags.find("#EXTINF").expect("EXTINF present");
        assert!(part_pos < extinf_pos, "parts must precede EXTINF");
    }

    // 7. LlHlsManifest::new sets part_hold_back correctly
    #[test]
    fn test_manifest_part_hold_back() {
        let m = make_manifest();
        assert!((m.part_hold_back() - 0.6).abs() < 1e-9);
    }

    // 8. build_m3u8 contains required header tags
    #[test]
    fn test_build_m3u8_header_tags() {
        let m = make_manifest();
        let out = m.build_m3u8();
        assert!(out.starts_with("#EXTM3U\n"), "must start with #EXTM3U");
        assert!(out.contains("#EXT-X-VERSION:9"));
        assert!(out.contains("#EXT-X-TARGETDURATION:6"));
        assert!(out.contains("#EXT-X-MEDIA-SEQUENCE:0"));
    }

    // 9. build_m3u8 contains EXT-X-PART-INF with PART-TARGET
    #[test]
    fn test_build_m3u8_part_inf() {
        let m = make_manifest();
        let out = m.build_m3u8();
        assert!(out.contains("#EXT-X-PART-INF:PART-TARGET=0.20000"));
    }

    // 10. build_m3u8 contains SERVER-CONTROL with CAN-BLOCK-RELOAD=YES
    #[test]
    fn test_build_m3u8_server_control() {
        let m = make_manifest();
        let out = m.build_m3u8();
        assert!(out.contains("#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES"));
        assert!(out.contains("PART-HOLD-BACK=0.600"));
    }

    // 11. build_m3u8 emits CAN-SKIP-UNTIL when set
    #[test]
    fn test_build_m3u8_can_skip_until() {
        let mut m = make_manifest();
        m.can_skip_until = 24.0;
        let out = m.build_m3u8();
        assert!(out.contains("CAN-SKIP-UNTIL=24.0"));
    }

    // 12. build_m3u8 suppresses CAN-SKIP-UNTIL when zero
    #[test]
    fn test_build_m3u8_no_skip_until_when_zero() {
        let m = make_manifest();
        let out = m.build_m3u8();
        assert!(!out.contains("CAN-SKIP-UNTIL"));
    }

    // 13. build_m3u8 includes segment EXTINF
    #[test]
    fn test_build_m3u8_segment_extinf() {
        let mut m = make_manifest();
        m.segments.push(HlsSegment::new(0, "seg0.ts", 6.0));
        let out = m.build_m3u8();
        assert!(out.contains("#EXTINF:"));
        assert!(out.contains("seg0.ts"));
    }

    // 14. build_m3u8 includes in-progress partial segments
    #[test]
    fn test_build_m3u8_partial_segments() {
        let mut m = make_manifest();
        m.partial_segments
            .push(PartialSegment::new(1, 0, 200, "part_current.mp4", true));
        let out = m.build_m3u8();
        assert!(out.contains("#EXT-X-PART"));
        assert!(out.contains("part_current.mp4"));
    }

    // 15. build_m3u8 includes EXT-X-PRELOAD-HINT at end
    #[test]
    fn test_build_m3u8_preload_hint() {
        let mut m = make_manifest();
        m.preload_hint = Some("next_part.mp4".to_owned());
        let out = m.build_m3u8();
        assert!(out.contains("#EXT-X-PRELOAD-HINT:TYPE=PART,URI=\"next_part.mp4\""));
    }

    // 16. Preload hint appears after segments in the output
    #[test]
    fn test_preload_hint_after_segments() {
        let mut m = make_manifest();
        m.segments.push(HlsSegment::new(0, "seg0.ts", 6.0));
        m.preload_hint = Some("hint.mp4".to_owned());
        let out = m.build_m3u8();
        let seg_pos = out.find("seg0.ts").expect("segment uri present");
        let hint_pos = out.find("#EXT-X-PRELOAD-HINT").expect("hint present");
        assert!(hint_pos > seg_pos, "preload hint must appear after segment");
    }

    // 17. Media sequence is respected
    #[test]
    fn test_media_sequence_respected() {
        let mut m = make_manifest();
        m.set_media_sequence(42);
        let out = m.build_m3u8();
        assert!(out.contains("#EXT-X-MEDIA-SEQUENCE:42"));
    }

    // 18. Multiple segments all appear in output
    #[test]
    fn test_multiple_segments_in_output() {
        let mut m = make_manifest();
        for i in 0..3u64 {
            m.segments
                .push(HlsSegment::new(i, format!("seg{i}.ts"), 6.0));
        }
        let out = m.build_m3u8();
        for i in 0..3 {
            assert!(out.contains(&format!("seg{i}.ts")));
        }
    }

    // 19. Segment with multiple parts emits parts in order
    #[test]
    fn test_segment_parts_order() {
        let mut m = make_manifest();
        let mut seg = HlsSegment::new(0, "seg0.ts", 6.0);
        seg.parts.push(PartialSegment::new(0, 0, 200, "p0.mp4", true));
        seg.parts.push(PartialSegment::new(0, 1, 200, "p1.mp4", false));
        seg.parts.push(PartialSegment::new(0, 2, 200, "p2.mp4", false));
        m.segments.push(seg);
        let out = m.build_m3u8();
        let p0 = out.find("p0.mp4").expect("p0");
        let p1 = out.find("p1.mp4").expect("p1");
        let p2 = out.find("p2.mp4").expect("p2");
        assert!(p0 < p1 && p1 < p2);
    }

    // 20. No preload hint: tag absent from output
    #[test]
    fn test_no_preload_hint_absent() {
        let m = make_manifest();
        assert!(!m.build_m3u8().contains("#EXT-X-PRELOAD-HINT"));
    }
}
