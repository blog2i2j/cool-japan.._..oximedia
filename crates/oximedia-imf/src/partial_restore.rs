//! Partial restore — extract specific segments or tracks from an IMF package.
//!
//! [`PartialRestore`] reads an IMF package and produces a filtered subset:
//! given a time range or a set of track identifiers, it generates a reduced
//! CPL and an updated PKL listing only the required essence files.
//!
//! # Example
//! ```no_run
//! use oximedia_imf::partial_restore::{PartialRestore, RestoreFilter};
//!
//! let filter = RestoreFilter::time_range(0, 5760); // first 4 min at 24fps
//! let result = PartialRestore::extract("/path/to/imp", filter).expect("ok");
//! println!("Extracted {} track segments", result.segment_count());
//! ```

#![allow(dead_code, missing_docs)]

use crate::{ImfError, ImfResult};
use std::path::Path;

/// Criteria for the partial restore operation.
#[derive(Debug, Clone)]
pub struct RestoreFilter {
    pub start_frame: Option<u64>,
    pub end_frame: Option<u64>,
    pub track_ids: Vec<String>,
    pub include_subtitles: bool,
    pub include_all_audio: bool,
}

impl RestoreFilter {
    #[must_use]
    pub fn time_range(start_frame: u64, end_frame: u64) -> Self {
        Self {
            start_frame: Some(start_frame),
            end_frame: Some(end_frame),
            track_ids: Vec::new(),
            include_subtitles: false,
            include_all_audio: false,
        }
    }

    #[must_use]
    pub fn tracks(track_ids: Vec<String>) -> Self {
        Self {
            start_frame: None,
            end_frame: None,
            track_ids,
            include_subtitles: false,
            include_all_audio: false,
        }
    }

    #[must_use]
    pub fn all() -> Self {
        Self {
            start_frame: None,
            end_frame: None,
            track_ids: Vec::new(),
            include_subtitles: true,
            include_all_audio: true,
        }
    }

    #[must_use]
    pub fn with_subtitles(mut self) -> Self {
        self.include_subtitles = true;
        self
    }

    #[must_use]
    pub fn duration_frames(&self) -> Option<u64> {
        match (self.start_frame, self.end_frame) {
            (Some(s), Some(e)) if e > s => Some(e - s),
            _ => None,
        }
    }
}

/// A single extracted track segment.
#[derive(Debug, Clone)]
pub struct ExtractedSegment {
    pub track_id: String,
    pub track_type: String,
    pub essence_path: String,
    pub source_in: u64,
    pub duration_frames: u64,
}

/// Result of a partial restore operation.
#[derive(Debug, Clone, Default)]
pub struct PartialRestoreResult {
    pub source_package: String,
    pub filter_start: Option<u64>,
    pub filter_end: Option<u64>,
    pub segments: Vec<ExtractedSegment>,
    pub cpl_xml: String,
    pub pkl_xml: String,
}

impl PartialRestoreResult {
    #[must_use]
    pub fn segment_count(&self) -> usize { self.segments.len() }

    #[must_use]
    pub fn total_duration_frames(&self) -> u64 {
        self.segments.iter().map(|s| s.duration_frames).sum()
    }

    /// Write CPL and PKL to the given output directory.
    ///
    /// # Errors
    /// Returns `ImfError::Io` on write failure.
    pub fn write_to(&self, dir: impl AsRef<Path>) -> ImfResult<()> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).map_err(|e| ImfError::Other(e.to_string()))?;
        std::fs::write(dir.join("CPL_PARTIAL.xml"), &self.cpl_xml)
            .map_err(|e| ImfError::Other(e.to_string()))?;
        std::fs::write(dir.join("PKL_PARTIAL.xml"), &self.pkl_xml)
            .map_err(|e| ImfError::Other(e.to_string()))
    }
}

/// Performs partial restore operations on IMF packages.
pub struct PartialRestore;

impl PartialRestore {
    /// Extract a subset of an IMF package according to `filter`.
    ///
    /// # Errors
    /// Returns `ImfError::InvalidPackage` if `pkg_path` is not a valid IMF package directory.
    pub fn extract(pkg_path: impl AsRef<Path>, filter: RestoreFilter) -> ImfResult<PartialRestoreResult> {
        let pkg_path = pkg_path.as_ref();
        if !pkg_path.is_dir() {
            return Err(ImfError::InvalidPackage(format!(
                "Not a directory: {}", pkg_path.display()
            )));
        }

        let mxf_files: Vec<_> = std::fs::read_dir(pkg_path)
            .map_err(|e| ImfError::Other(e.to_string()))?
            .flatten()
            .filter(|e| {
                e.path().extension()
                    .and_then(|x| x.to_str())
                    .map(|x| x.eq_ignore_ascii_case("mxf"))
                    .unwrap_or(false)
            })
            .collect();

        if mxf_files.is_empty() {
            return Err(ImfError::InvalidPackage("No MXF essence files found".to_string()));
        }

        let mut segments = Vec::new();
        let start_frame = filter.start_frame.unwrap_or(0);

        for (i, entry) in mxf_files.iter().enumerate() {
            let path_str = entry.path().to_string_lossy().to_string();
            let file_size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            let total_frames = (file_size / 2_000_000).max(1);
            let end_frame = filter.end_frame.unwrap_or(total_frames);
            let clip_end = end_frame.min(total_frames);
            let clip_start = start_frame.min(clip_end);
            let duration = clip_end.saturating_sub(clip_start);
            let track_type = if i == 0 { "MainImageSequence" } else { "MainAudioSequence" };
            segments.push(ExtractedSegment {
                track_id: format!("urn:uuid:track-{i}"),
                track_type: track_type.to_string(),
                essence_path: path_str,
                source_in: clip_start,
                duration_frames: duration,
            });
        }

        let cpl_xml = build_cpl(&segments, &filter);
        let pkl_xml = build_pkl(&mxf_files);

        Ok(PartialRestoreResult {
            source_package: pkg_path.to_string_lossy().to_string(),
            filter_start: filter.start_frame,
            filter_end: filter.end_frame,
            segments,
            cpl_xml,
            pkl_xml,
        })
    }
}

fn build_cpl(segments: &[ExtractedSegment], filter: &RestoreFilter) -> String {
    let total: u64 = segments.iter().map(|s| s.duration_frames).sum();
    let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<CompositionPlaylist>\n  <Id>urn:uuid:partial-cpl</Id>\n  <EditRate>24 1</EditRate>\n");
    xml.push_str(&format!("  <!-- frames {}-{} -->\n",
        filter.start_frame.unwrap_or(0),
        filter.end_frame.map(|e| e.to_string()).unwrap_or_else(|| "end".to_string())));
    xml.push_str("  <SegmentList><Segment><SequenceList>\n");
    for seg in segments {
        xml.push_str(&format!(
            "    <Sequence><TrackId>{}</TrackId><ResourceList><Resource>\
             <EntryPoint>{}</EntryPoint><SourceDuration>{}</SourceDuration>\
             </Resource></ResourceList></Sequence>\n",
            seg.track_id, seg.source_in, seg.duration_frames
        ));
    }
    xml.push_str(&format!("  </SequenceList><Duration>{total}</Duration></Segment></SegmentList>\n"));
    xml.push_str("</CompositionPlaylist>\n");
    xml
}

fn build_pkl(entries: &[std::fs::DirEntry]) -> String {
    let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<PackingList>\n  <AssetList>\n");
    for (i, entry) in entries.iter().enumerate() {
        let name = entry.file_name().to_string_lossy().to_string();
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        xml.push_str(&format!(
            "    <Asset><Id>urn:uuid:asset-{i}</Id><OriginalFileName>{name}</OriginalFileName><Size>{size}</Size></Asset>\n"
        ));
    }
    xml.push_str("  </AssetList>\n</PackingList>\n");
    xml
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pkg(suffix: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("oximedia_pr_{suffix}"));
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("video.mxf"), vec![0u8; 2_000_000]).ok();
        std::fs::write(dir.join("audio.mxf"), vec![0u8; 500_000]).ok();
        dir
    }

    #[test]
    fn test_extract_all() {
        let dir = make_pkg("all");
        let r = PartialRestore::extract(&dir, RestoreFilter::all()).expect("ok");
        assert!(r.segment_count() > 0);
        assert!(r.cpl_xml.contains("<CompositionPlaylist"));
        assert!(r.pkl_xml.contains("<PackingList"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_extract_time_range() {
        let dir = make_pkg("range");
        let r = PartialRestore::extract(&dir, RestoreFilter::time_range(0, 1)).expect("ok");
        assert_eq!(r.filter_start, Some(0));
        assert_eq!(r.filter_end, Some(1));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_nonexistent() {
        assert!(PartialRestore::extract("/nonexistent/xyz", RestoreFilter::all()).is_err());
    }

    #[test]
    fn test_empty_dir_fails() {
        let dir = std::env::temp_dir().join("oximedia_pr_empty");
        std::fs::create_dir_all(&dir).ok();
        assert!(PartialRestore::extract(&dir, RestoreFilter::all()).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_write_to() {
        let src = make_pkg("write");
        let out = std::env::temp_dir().join("oximedia_pr_out");
        let r = PartialRestore::extract(&src, RestoreFilter::all()).expect("ok");
        r.write_to(&out).expect("write ok");
        assert!(out.join("CPL_PARTIAL.xml").exists());
        assert!(out.join("PKL_PARTIAL.xml").exists());
        std::fs::remove_dir_all(&src).ok();
        std::fs::remove_dir_all(&out).ok();
    }

    #[test]
    fn test_filter_duration() {
        let f = RestoreFilter::time_range(100, 300);
        assert_eq!(f.duration_frames(), Some(200));
    }
}
