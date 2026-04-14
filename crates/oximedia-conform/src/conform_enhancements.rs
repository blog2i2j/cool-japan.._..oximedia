//! Enhanced conform utilities: PDF stub export, XML import, hash matching,
//! gap detection, multi-cam conform, AAF stub export, partial filename matching,
//! and frame accuracy validation.

#![allow(dead_code)]

use crate::types::ConformClip;
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// ConformReport PDF stub
// ---------------------------------------------------------------------------

/// A minimal ASCII-text "PDF" stub for conform reports.
///
/// A real PDF requires complex binary structure. This stub emits a minimal
/// PDF/ASCII representation that embeds the report text. It is suitable for
/// testing and archival stubs where a real PDF renderer is not available.
pub fn to_pdf_stub(session_name: &str, clip_count: usize, matched_count: usize) -> Vec<u8> {
    // Build a minimal, valid ASCII PDF that contains the report text.
    // Structure: header, one page, content stream with the text, cross-reference table.
    let body = format!(
        "OxiMedia Conform Report\nSession: {session_name}\nClips: {clip_count}\nMatched: {matched_count}\n"
    );

    // We'll build a minimal PDF manually (no external deps).
    let content_stream = format!(
        "BT\n/F1 12 Tf\n50 750 Td\n({}) Tj\n50 730 Td\n(Session: {}) Tj\n50 710 Td\n(Clips: {}) Tj\n50 690 Td\n(Matched: {}) Tj\nET\n",
        "OxiMedia Conform Report",
        session_name.replace('(', "\\(").replace(')', "\\)"),
        clip_count,
        matched_count,
    );

    let stream_len = content_stream.len();

    let mut pdf = String::new();
    pdf.push_str("%PDF-1.4\n");

    // Object 1: Catalog
    let obj1_offset = pdf.len();
    pdf.push_str("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    // Object 2: Pages
    let obj2_offset = pdf.len();
    pdf.push_str("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    // Object 3: Page
    let obj3_offset = pdf.len();
    pdf.push_str(
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    );

    // Object 4: Content stream
    let obj4_offset = pdf.len();
    pdf.push_str(&format!(
        "4 0 obj\n<< /Length {stream_len} >>\nstream\n{content_stream}endstream\nendobj\n"
    ));

    // Object 5: Font
    let obj5_offset = pdf.len();
    pdf.push_str(
        "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    );

    // Cross-reference table
    let xref_offset = pdf.len();
    pdf.push_str("xref\n0 6\n");
    pdf.push_str("0000000000 65535 f \n");
    pdf.push_str(&format!("{obj1_offset:010} 00000 n \n"));
    pdf.push_str(&format!("{obj2_offset:010} 00000 n \n"));
    pdf.push_str(&format!("{obj3_offset:010} 00000 n \n"));
    pdf.push_str(&format!("{obj4_offset:010} 00000 n \n"));
    pdf.push_str(&format!("{obj5_offset:010} 00000 n \n"));

    // Trailer
    pdf.push_str(&format!(
        "trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ));

    // Embed report text as a comment at end (for easy extraction / debugging)
    pdf.push_str(&format!("% Report body: {body}"));

    pdf.into_bytes()
}

// ---------------------------------------------------------------------------
// ResolveXmlImport
// ---------------------------------------------------------------------------

/// Parsed clip data from a Resolve/Premiere XML timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct XmlClip {
    /// Clip name.
    pub name: String,
    /// Source in-point (frames or timecode string).
    pub src_in: String,
    /// Source out-point.
    pub src_out: String,
    /// Record in-point.
    pub rec_in: String,
    /// Record out-point.
    pub rec_out: String,
}

/// Errors that can occur during XML import.
#[derive(Debug)]
pub enum XmlImportError {
    /// The XML is malformed or missing required attributes.
    MalformedXml(String),
}

impl std::fmt::Display for XmlImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MalformedXml(msg) => write!(f, "Malformed XML: {msg}"),
        }
    }
}

impl std::error::Error for XmlImportError {}

/// Minimal XML parser that extracts `<clip>` elements and their attributes.
///
/// This is a purpose-built parser for the simple `<clip name="..." srcIn="..."
/// srcOut="..." recIn="..." recOut="..."/>` format used by DaVinci Resolve and
/// Adobe Premiere in their EDL-companion XML exports. It does **not** depend on
/// any external XML library.
pub struct ResolveXmlImport;

impl ResolveXmlImport {
    /// Parse `<clip>` elements from an XML string.
    ///
    /// Each `<clip>` element must have `name`, `srcIn`, `srcOut`, `recIn`, and
    /// `recOut` attributes. Missing elements are skipped; missing required
    /// attributes on a present element produce an error.
    ///
    /// # Errors
    ///
    /// Returns `XmlImportError::MalformedXml` if a `<clip>` element is missing a
    /// required attribute.
    pub fn parse(xml: &str) -> Result<Vec<XmlClip>, XmlImportError> {
        let mut clips = Vec::new();

        // We scan for every `<clip` token and extract attributes manually.
        let mut remaining = xml;
        while let Some(start) = remaining.find("<clip") {
            remaining = &remaining[start..];

            // Find end of this element (either `/>` or `>` for non-self-closing)
            let end = remaining.find('>').ok_or_else(|| {
                XmlImportError::MalformedXml("Unclosed <clip element".to_string())
            })?;

            let element_src = &remaining[..=end];
            remaining = &remaining[end + 1..];

            // Extract all attributes from the element string
            let name = Self::extract_attr(element_src, "name")
                .ok_or_else(|| XmlImportError::MalformedXml("Missing 'name' on <clip>".to_string()))?;
            let src_in = Self::extract_attr(element_src, "srcIn")
                .ok_or_else(|| XmlImportError::MalformedXml("Missing 'srcIn' on <clip>".to_string()))?;
            let src_out = Self::extract_attr(element_src, "srcOut")
                .ok_or_else(|| XmlImportError::MalformedXml("Missing 'srcOut' on <clip>".to_string()))?;
            let rec_in = Self::extract_attr(element_src, "recIn")
                .ok_or_else(|| XmlImportError::MalformedXml("Missing 'recIn' on <clip>".to_string()))?;
            let rec_out = Self::extract_attr(element_src, "recOut")
                .ok_or_else(|| XmlImportError::MalformedXml("Missing 'recOut' on <clip>".to_string()))?;

            clips.push(XmlClip {
                name,
                src_in,
                src_out,
                rec_in,
                rec_out,
            });
        }

        Ok(clips)
    }

    /// Extract an attribute value from an element string like `<clip name="foo" ...>`.
    fn extract_attr(element: &str, attr: &str) -> Option<String> {
        // Pattern: `attr="value"` or `attr='value'`
        let needle_dq = format!("{attr}=\"");
        let needle_sq = format!("{attr}='");

        if let Some(pos) = element.find(&needle_dq) {
            let after = &element[pos + needle_dq.len()..];
            let end = after.find('"')?;
            return Some(after[..end].to_string());
        }
        if let Some(pos) = element.find(&needle_sq) {
            let after = &element[pos + needle_sq.len()..];
            let end = after.find('\'')?;
            return Some(after[..end].to_string());
        }
        None
    }
}

// ---------------------------------------------------------------------------
// ContentHashMatcher
// ---------------------------------------------------------------------------

/// A pair of source and target clips that share an identical content hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashMatch {
    /// Key from the source map whose hash was matched.
    pub source_key: String,
    /// Key from the target map whose hash was matched.
    pub target_key: String,
    /// The shared content hash.
    pub hash: [u8; 32],
}

/// Matches clips between two collections using a 32-byte content hash.
pub struct ContentHashMatcher;

impl ContentHashMatcher {
    /// Match clips whose 32-byte content hashes are identical.
    ///
    /// For every `(source_key, hash)` pair this method looks up the same hash in
    /// `target_hashes`. When a match is found a [`HashMatch`] entry is appended to
    /// the result. Each source key produces at most one match (the first target
    /// found with the same hash).
    #[must_use]
    pub fn match_clips(
        source_hashes: &HashMap<String, [u8; 32]>,
        target_hashes: &HashMap<String, [u8; 32]>,
    ) -> Vec<HashMatch> {
        // Build a reverse index: hash -> target_key for O(1) lookup.
        let mut target_index: HashMap<[u8; 32], &str> = HashMap::new();
        for (key, hash) in target_hashes {
            target_index.entry(*hash).or_insert(key.as_str());
        }

        let mut matches = Vec::new();
        for (source_key, hash) in source_hashes {
            if let Some(&target_key) = target_index.get(hash) {
                matches.push(HashMatch {
                    source_key: source_key.clone(),
                    target_key: target_key.to_string(),
                    hash: *hash,
                });
            }
        }

        // Sort for deterministic output
        matches.sort_by(|a, b| a.source_key.cmp(&b.source_key));
        matches
    }
}

// ---------------------------------------------------------------------------
// TimelineGapDetector
// ---------------------------------------------------------------------------

/// A gap detected between two adjacent clips on the timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct Gap {
    /// Frame number at which the gap starts (exclusive end of preceding clip).
    pub start_frame: u64,
    /// Frame number at which the gap ends (inclusive start of following clip).
    pub end_frame: u64,
    /// Gap duration in frames.
    pub duration_frames: u64,
    /// Gap duration in seconds.
    pub duration_secs: f64,
}

/// Detects holes between clips on a timeline.
pub struct TimelineGapDetector;

impl TimelineGapDetector {
    /// Find gaps between clips on a timeline.
    ///
    /// A gap exists whenever the `offset_s` of the next clip is greater than the
    /// `offset_s + duration_s` of the preceding clip.
    ///
    /// # Arguments
    ///
    /// * `clips` – Clips in any order; they are sorted by `offset_s` internally.
    /// * `fps`   – Frames per second used to convert gap durations to frame counts.
    ///
    /// # Returns
    ///
    /// A sorted list of [`Gap`] values. Empty if there are no holes.
    #[must_use]
    pub fn find_gaps(clips: &[ConformClip], fps: f32) -> Vec<Gap> {
        if clips.len() < 2 || fps <= 0.0 {
            return Vec::new();
        }

        // Sort clips by their timeline offset.
        let mut sorted: Vec<&ConformClip> = clips.iter().collect();
        sorted.sort_by(|a, b| a.offset_s.partial_cmp(&b.offset_s).unwrap_or(std::cmp::Ordering::Equal));

        let mut gaps = Vec::new();
        let fps_f64 = f64::from(fps);

        for window in sorted.windows(2) {
            let prev = window[0];
            let next = window[1];

            let prev_end_s = prev.offset_s + prev.duration_s;
            let gap_secs = next.offset_s - prev_end_s;

            // Only report positive gaps (negative means overlap, zero means adjacent)
            if gap_secs > 1e-9 {
                let start_frame = (prev_end_s * fps_f64).round() as u64;
                let end_frame = (next.offset_s * fps_f64).round() as u64;
                let duration_frames = end_frame.saturating_sub(start_frame);

                gaps.push(Gap {
                    start_frame,
                    end_frame,
                    duration_frames,
                    duration_secs: gap_secs,
                });
            }
        }

        gaps
    }
}

// ---------------------------------------------------------------------------
// MultiCamConformer
// ---------------------------------------------------------------------------

/// A pair of primary and secondary clips that have been synchronised.
#[derive(Debug, Clone, PartialEq)]
pub struct SyncedPair {
    /// Index into the `primary` slice of the matching primary clip.
    pub primary_index: usize,
    /// Index into the `secondary` slice of the matching secondary clip.
    pub secondary_index: usize,
    /// Offset in frames between the two clips at the sync point.
    pub offset_frames: i64,
}

/// Multi-camera conform utility that aligns secondary cameras to a primary
/// camera using a common sync point.
pub struct MultiCamConformer;

impl MultiCamConformer {
    /// Synchronise secondary camera clips to primary camera clips.
    ///
    /// For each primary clip the algorithm finds the secondary clip that best
    /// straddles the `sync_point` (record-in offset in frames). The resulting
    /// pairs indicate which clips overlap at the sync point and by how many
    /// frames.
    ///
    /// # Arguments
    ///
    /// * `primary`    – Clips from the primary / sync reference camera.
    /// * `secondary`  – Clips from the secondary camera.
    /// * `sync_point` – Timeline position (in frames from timeline start) used
    ///                  as the synchronisation anchor.
    ///
    /// # Returns
    ///
    /// A vec of [`SyncedPair`] entries for every primary clip that has a
    /// corresponding secondary clip at the sync point.
    #[must_use]
    pub fn sync_cameras(
        primary: &[ConformClip],
        secondary: &[ConformClip],
        sync_point: u64,
    ) -> Vec<SyncedPair> {
        // Helper: convert offset_s to frame number (approximate, 25 fps default
        // since ConformClip does not carry fps). The conformer just uses frame
        // arithmetic on the stored second values scaled to 1000 for sub-second
        // precision without losing generality.
        let to_frame = |secs: f64| -> i64 { (secs * 1000.0).round() as i64 };
        let sync_frame = sync_point as i64;

        let mut pairs = Vec::new();

        for (pi, pclip) in primary.iter().enumerate() {
            let p_start = to_frame(pclip.offset_s);
            let p_end = to_frame(pclip.offset_s + pclip.duration_s);

            // The primary clip must span the sync point.
            if p_start > sync_frame || p_end < sync_frame {
                continue;
            }

            // Find the secondary clip that also spans the sync point, choosing
            // the closest one by distance to sync point at its midpoint.
            let best = secondary
                .iter()
                .enumerate()
                .filter_map(|(si, sclip)| {
                    let s_start = to_frame(sclip.offset_s);
                    let s_end = to_frame(sclip.offset_s + sclip.duration_s);
                    if s_start <= sync_frame && s_end >= sync_frame {
                        let offset = s_start - p_start;
                        Some((si, offset))
                    } else {
                        None
                    }
                })
                .min_by_key(|(_, offset)| offset.abs());

            if let Some((si, offset_frames)) = best {
                pairs.push(SyncedPair {
                    primary_index: pi,
                    secondary_index: si,
                    offset_frames,
                });
            }
        }

        pairs
    }
}

// ---------------------------------------------------------------------------
// AafConformExport
// ---------------------------------------------------------------------------

/// Exports conform data to an AAF-stub byte sequence.
///
/// A full AAF file requires the Microsoft Structured Storage format plus
/// hundreds of AAF SDK classes. This stub produces a placeholder byte sequence
/// that identifies itself as an OxiMedia AAF stub. Real AAF generation should
/// be delegated to `oximedia-aaf` once that crate is available.
pub struct AafConformExport;

impl AafConformExport {
    /// Produce a placeholder AAF byte sequence for the given clips.
    ///
    /// The returned bytes begin with `OAAF` (OxiMedia AAF stub magic) followed
    /// by a version byte, the clip count as a little-endian u32, and an ASCII
    /// text summary.
    #[must_use]
    pub fn to_aaf_stub(clips: &[ConformClip]) -> Vec<u8> {
        // Magic: 'O', 'A', 'A', 'F'
        const MAGIC: &[u8] = b"OAAF";
        const VERSION: u8 = 1;

        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.push(VERSION);

        // Clip count as little-endian u32
        let clip_count = clips.len() as u32;
        buf.extend_from_slice(&clip_count.to_le_bytes());

        // ASCII clip summary (name, offset, duration)
        let summary = clips
            .iter()
            .map(|c| format!("{}@{:.3}+{:.3}", c.name, c.offset_s, c.duration_s))
            .collect::<Vec<_>>()
            .join(";");

        // Length-prefixed summary string (u32 LE length + UTF-8 bytes)
        let summary_bytes = summary.as_bytes();
        buf.extend_from_slice(&(summary_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(summary_bytes);

        // Placeholder comment
        let comment = b"full AAF requires oximedia-aaf";
        buf.extend_from_slice(&(comment.len() as u32).to_le_bytes());
        buf.extend_from_slice(comment);

        buf
    }
}

// ---------------------------------------------------------------------------
// PartialFileMatcher
// ---------------------------------------------------------------------------

/// Matches source and target file paths by their filename stem (basename
/// without the file extension).
pub struct PartialFileMatcher;

impl PartialFileMatcher {
    /// Match source and target paths by filename stem.
    ///
    /// Returns a vector of `(source_index, target_index)` pairs where the stem
    /// of the source filename equals the stem of the target filename. Each source
    /// index appears at most once.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_conform::conform_enhancements::PartialFileMatcher;
    ///
    /// let sources = ["clip_001.mov", "clip_002.r3d"];
    /// let targets = ["clip_001.mxf", "clip_003.mp4"];
    /// let pairs = PartialFileMatcher::match_by_filename_stem(&sources, &targets);
    /// assert_eq!(pairs, vec![(0, 0)]);
    /// ```
    #[must_use]
    pub fn match_by_filename_stem(sources: &[&str], targets: &[&str]) -> Vec<(usize, usize)> {
        // Build index: stem -> first target index
        let mut target_index: HashMap<String, usize> = HashMap::new();
        for (ti, &target) in targets.iter().enumerate() {
            let stem = Self::stem(target);
            target_index.entry(stem).or_insert(ti);
        }

        let mut pairs = Vec::new();
        for (si, &source) in sources.iter().enumerate() {
            let stem = Self::stem(source);
            if let Some(&ti) = target_index.get(&stem) {
                pairs.push((si, ti));
            }
        }

        pairs
    }

    /// Extract the filename stem from a path string.
    fn stem(path: &str) -> String {
        let path = Path::new(path);
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(path.to_str().unwrap_or(""));

        // Remove the last extension
        match filename.rfind('.') {
            Some(dot_pos) if dot_pos > 0 => filename[..dot_pos].to_string(),
            _ => filename.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// FrameAccuracyValidator
// ---------------------------------------------------------------------------

/// Validates whether an actual frame number is within an acceptable tolerance
/// of the expected frame number.
pub struct FrameAccuracyValidator;

impl FrameAccuracyValidator {
    /// Check whether `actual` is within `tolerance_frames` of `expected`.
    ///
    /// # Arguments
    ///
    /// * `expected`         – The reference frame number.
    /// * `actual`           – The measured / computed frame number.
    /// * `tolerance_frames` – Maximum allowable deviation in frames (inclusive).
    ///
    /// # Returns
    ///
    /// `true` if `|expected - actual| <= tolerance_frames`.
    #[must_use]
    pub fn check(expected: u64, actual: u64, tolerance_frames: u32) -> bool {
        let diff = if expected >= actual {
            expected - actual
        } else {
            actual - expected
        };
        diff <= u64::from(tolerance_frames)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── to_pdf_stub ───────────────────────────────────────────────────────

    #[test]
    fn test_pdf_stub_starts_with_pdf_header() {
        let bytes = to_pdf_stub("TestSession", 10, 8);
        let s = String::from_utf8_lossy(&bytes);
        assert!(s.starts_with("%PDF-1.4"), "should start with PDF header");
    }

    #[test]
    fn test_pdf_stub_contains_session_name() {
        let bytes = to_pdf_stub("MyConform", 5, 3);
        let s = String::from_utf8_lossy(&bytes);
        assert!(s.contains("MyConform"), "should contain session name");
    }

    #[test]
    fn test_pdf_stub_ends_with_eof() {
        let bytes = to_pdf_stub("S", 1, 1);
        let s = String::from_utf8_lossy(&bytes);
        assert!(s.contains("%%EOF"), "should contain %%EOF marker");
    }

    // ── ResolveXmlImport ─────────────────────────────────────────────────

    #[test]
    fn test_resolve_xml_parse_single_clip() {
        let xml = r#"<timeline><clip name="A001C001" srcIn="00:01:00:00" srcOut="00:01:30:00" recIn="00:00:00:00" recOut="00:00:30:00"/></timeline>"#;
        let clips = ResolveXmlImport::parse(xml).expect("parse should succeed");
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].name, "A001C001");
        assert_eq!(clips[0].src_in, "00:01:00:00");
        assert_eq!(clips[0].src_out, "00:01:30:00");
        assert_eq!(clips[0].rec_in, "00:00:00:00");
        assert_eq!(clips[0].rec_out, "00:00:30:00");
    }

    #[test]
    fn test_resolve_xml_parse_multiple_clips() {
        let xml = r#"
<timeline>
  <clip name="A001C001" srcIn="01:00:00:00" srcOut="01:00:10:00" recIn="00:00:00:00" recOut="00:00:10:00"/>
  <clip name="A001C002" srcIn="01:00:15:00" srcOut="01:00:25:00" recIn="00:00:10:00" recOut="00:00:20:00"/>
</timeline>
"#;
        let clips = ResolveXmlImport::parse(xml).expect("parse should succeed");
        assert_eq!(clips.len(), 2);
        assert_eq!(clips[0].name, "A001C001");
        assert_eq!(clips[1].name, "A001C002");
    }

    #[test]
    fn test_resolve_xml_parse_empty_timeline() {
        let xml = "<timeline></timeline>";
        let clips = ResolveXmlImport::parse(xml).expect("parse should succeed");
        assert!(clips.is_empty());
    }

    #[test]
    fn test_resolve_xml_parse_missing_attr_returns_error() {
        let xml = r#"<clip name="X" srcIn="0" srcOut="1" recIn="0"/>"#; // missing recOut
        let result = ResolveXmlImport::parse(xml);
        assert!(result.is_err());
    }

    // ── ContentHashMatcher ───────────────────────────────────────────────

    fn make_hash(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    #[test]
    fn test_hash_match_identical_hashes() {
        let mut sources = HashMap::new();
        sources.insert("clip_a.mov".to_string(), make_hash(1));
        sources.insert("clip_b.mov".to_string(), make_hash(2));

        let mut targets = HashMap::new();
        targets.insert("clip_a.mxf".to_string(), make_hash(1));
        targets.insert("clip_c.mxf".to_string(), make_hash(3));

        let matches = ContentHashMatcher::match_clips(&sources, &targets);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].source_key, "clip_a.mov");
        assert_eq!(matches[0].target_key, "clip_a.mxf");
        assert_eq!(matches[0].hash, make_hash(1));
    }

    #[test]
    fn test_hash_match_no_matches() {
        let mut sources = HashMap::new();
        sources.insert("a".to_string(), make_hash(10));

        let mut targets = HashMap::new();
        targets.insert("b".to_string(), make_hash(20));

        let matches = ContentHashMatcher::match_clips(&sources, &targets);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_hash_match_multiple() {
        let mut sources = HashMap::new();
        sources.insert("s1".to_string(), make_hash(5));
        sources.insert("s2".to_string(), make_hash(6));

        let mut targets = HashMap::new();
        targets.insert("t1".to_string(), make_hash(5));
        targets.insert("t2".to_string(), make_hash(6));

        let matches = ContentHashMatcher::match_clips(&sources, &targets);
        assert_eq!(matches.len(), 2);
    }

    // ── TimelineGapDetector ──────────────────────────────────────────────

    fn make_clip(name: &str, offset_s: f64, duration_s: f64) -> ConformClip {
        ConformClip {
            name: name.to_string(),
            offset_s,
            duration_s,
            src_in_s: 0.0,
            src_out_s: duration_s,
        }
    }

    #[test]
    fn test_gap_found_between_non_adjacent_clips() {
        let clips = vec![
            make_clip("A", 0.0, 10.0),  // ends at 10s
            make_clip("B", 15.0, 5.0),  // starts at 15s → gap 10s..15s
        ];
        let gaps = TimelineGapDetector::find_gaps(&clips, 25.0);
        assert_eq!(gaps.len(), 1);
        let gap = &gaps[0];
        assert_eq!(gap.start_frame, 250); // 10s * 25fps
        assert_eq!(gap.end_frame, 375);   // 15s * 25fps
        assert_eq!(gap.duration_frames, 125); // 5s * 25fps
        assert!((gap.duration_secs - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_no_gap_adjacent_clips() {
        let clips = vec![
            make_clip("A", 0.0, 10.0),
            make_clip("B", 10.0, 5.0),
        ];
        let gaps = TimelineGapDetector::find_gaps(&clips, 25.0);
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_gap_unsorted_clips() {
        // Pass clips in reverse order – detector should sort them
        let clips = vec![
            make_clip("B", 20.0, 5.0),
            make_clip("A", 0.0, 5.0),  // ends at 5s, next starts at 20s → gap
        ];
        let gaps = TimelineGapDetector::find_gaps(&clips, 25.0);
        assert_eq!(gaps.len(), 1);
        assert!((gaps[0].duration_secs - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_no_gaps_single_clip() {
        let clips = vec![make_clip("A", 0.0, 10.0)];
        let gaps = TimelineGapDetector::find_gaps(&clips, 25.0);
        assert!(gaps.is_empty());
    }

    // ── MultiCamConformer ────────────────────────────────────────────────

    #[test]
    fn test_sync_cameras_basic() {
        let primary = vec![make_clip("P1", 0.0, 60.0)];
        let secondary = vec![make_clip("S1", 0.0, 60.0)];
        let pairs = MultiCamConformer::sync_cameras(&primary, &secondary, 1000);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].primary_index, 0);
        assert_eq!(pairs[0].secondary_index, 0);
    }

    #[test]
    fn test_sync_cameras_no_overlap() {
        let primary = vec![make_clip("P1", 0.0, 10.0)];    // frames 0..10000
        let secondary = vec![make_clip("S1", 20.0, 10.0)]; // frames 20000..30000
        // sync_point=500 is within primary but not secondary
        let pairs = MultiCamConformer::sync_cameras(&primary, &secondary, 500);
        assert!(pairs.is_empty());
    }

    // ── AafConformExport ─────────────────────────────────────────────────

    #[test]
    fn test_aaf_stub_magic() {
        let clips = vec![make_clip("C1", 0.0, 10.0)];
        let bytes = AafConformExport::to_aaf_stub(&clips);
        assert!(bytes.starts_with(b"OAAF"));
    }

    #[test]
    fn test_aaf_stub_clip_count() {
        let clips = vec![make_clip("A", 0.0, 5.0), make_clip("B", 5.0, 5.0)];
        let bytes = AafConformExport::to_aaf_stub(&clips);
        // bytes[4] = VERSION, bytes[5..9] = clip_count LE u32
        let count = u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_aaf_stub_empty_clips() {
        let bytes = AafConformExport::to_aaf_stub(&[]);
        assert!(bytes.starts_with(b"OAAF"));
        let count = u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]);
        assert_eq!(count, 0);
    }

    // ── PartialFileMatcher ───────────────────────────────────────────────

    #[test]
    fn test_partial_filename_match_across_extensions() {
        let sources = ["clip_001.mov", "clip_002.r3d", "clip_003.ari"];
        let targets = ["clip_001.mxf", "clip_002.mxf", "clip_004.mp4"];
        let pairs = PartialFileMatcher::match_by_filename_stem(&sources, &targets);
        assert_eq!(pairs.len(), 2);
        assert!(pairs.contains(&(0, 0)));
        assert!(pairs.contains(&(1, 1)));
    }

    #[test]
    fn test_partial_filename_match_no_match() {
        let sources = ["alpha.mov"];
        let targets = ["beta.mxf"];
        let pairs = PartialFileMatcher::match_by_filename_stem(&sources, &targets);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_partial_filename_match_with_paths() {
        let sources = ["/media/raw/interview.r3d"];
        let targets = ["/deliver/grade/interview.mxf"];
        let pairs = PartialFileMatcher::match_by_filename_stem(&sources, &targets);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (0, 0));
    }

    #[test]
    fn test_partial_filename_no_extension() {
        let sources = ["clip_no_ext"];
        let targets = ["clip_no_ext"];
        let pairs = PartialFileMatcher::match_by_filename_stem(&sources, &targets);
        assert_eq!(pairs.len(), 1);
    }

    // ── FrameAccuracyValidator ───────────────────────────────────────────

    #[test]
    fn test_frame_accuracy_exact_match() {
        assert!(FrameAccuracyValidator::check(100, 100, 0));
    }

    #[test]
    fn test_frame_accuracy_within_tolerance() {
        assert!(FrameAccuracyValidator::check(100, 102, 2));
        assert!(FrameAccuracyValidator::check(100, 98, 2));
    }

    #[test]
    fn test_frame_accuracy_exceeds_tolerance() {
        assert!(!FrameAccuracyValidator::check(100, 103, 2));
        assert!(!FrameAccuracyValidator::check(100, 97, 2));
    }

    #[test]
    fn test_frame_accuracy_zero_tolerance_fails() {
        assert!(!FrameAccuracyValidator::check(100, 101, 0));
    }
}
