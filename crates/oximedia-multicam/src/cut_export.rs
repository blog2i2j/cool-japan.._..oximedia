//! Multi-camera cut list exporter — EDL (Edit Decision List) format.
//!
//! Generates a simplified CMX 3600-compatible EDL string from a list of cut
//! tuples `(offset_ms, camera_id, timecode_ms)`.  The output is suitable for
//! import into non-linear editors that understand basic EDL syntax.
//!
//! # Example
//!
//! ```rust
//! use oximedia_multicam::cut_export::MulticamCutExporter;
//!
//! let cuts = [(0, 1, 0), (5000, 2, 5000), (12000, 1, 12000)];
//! let edl = MulticamCutExporter::to_edl(&cuts);
//! assert!(edl.contains("TITLE"));
//! assert!(edl.contains("001"));
//! ```

/// Exports multi-camera cut lists in CMX 3600 EDL format.
pub struct MulticamCutExporter;

impl MulticamCutExporter {
    /// Convert a slice of cut tuples into a CMX 3600 EDL string.
    ///
    /// Each element of `cuts` is `(offset_ms, camera_id, timecode_ms)`:
    ///
    /// * `offset_ms`    — timeline position of the cut (milliseconds).
    /// * `camera_id`    — camera / angle identifier.
    /// * `timecode_ms`  — source timecode at the cut point (milliseconds).
    #[must_use]
    pub fn to_edl(cuts: &[(u64, u64, u64)]) -> String {
        let mut out = String::new();
        out.push_str("TITLE: MULTICAM EDL\n");
        out.push_str("FCM: NON-DROP FRAME\n\n");

        for (idx, &(offset_ms, camera_id, tc_ms)) in cuts.iter().enumerate() {
            let event = idx + 1;
            let reel = format!("CAM{camera_id:03}");
            let src_in = ms_to_tc(tc_ms, 25);
            let src_out = if idx + 1 < cuts.len() {
                let dur = cuts[idx + 1].0.saturating_sub(offset_ms);
                ms_to_tc(tc_ms + dur, 25)
            } else {
                // Last cut: use a nominal 5-second out point.
                ms_to_tc(tc_ms + 5_000, 25)
            };
            let rec_in = ms_to_tc(offset_ms, 25);
            let rec_out = if idx + 1 < cuts.len() {
                ms_to_tc(cuts[idx + 1].0, 25)
            } else {
                ms_to_tc(offset_ms + 5_000, 25)
            };

            out.push_str(&format!(
                "{event:03}  {reel:<8} V     C        {src_in} {src_out} {rec_in} {rec_out}\n"
            ));
        }

        out
    }

    /// Convert a cut list to a simple CSV string with columns:
    /// `event,camera_id,offset_ms,timecode_ms`.
    #[must_use]
    pub fn to_csv(cuts: &[(u64, u64, u64)]) -> String {
        let mut out = String::from("event,camera_id,offset_ms,timecode_ms\n");
        for (idx, &(offset_ms, camera_id, tc_ms)) in cuts.iter().enumerate() {
            out.push_str(&format!("{},{},{},{}\n", idx + 1, camera_id, offset_ms, tc_ms));
        }
        out
    }

    /// Return the total number of cuts in the list.
    #[must_use]
    pub fn cut_count(cuts: &[(u64, u64, u64)]) -> usize {
        cuts.len()
    }

    /// Compute the total programme duration (ms) from the cut list.
    ///
    /// The duration is the distance from the first cut offset to the last cut
    /// offset plus a default 5-second tail for the final cut.
    #[must_use]
    pub fn total_duration_ms(cuts: &[(u64, u64, u64)]) -> u64 {
        match (cuts.first(), cuts.last()) {
            (Some(&(first_off, _, _)), Some(&(last_off, _, _))) => {
                (last_off - first_off) + 5_000
            }
            _ => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert milliseconds to a SMPTE timecode string `HH:MM:SS:FF` at the given
/// integer frame rate.
fn ms_to_tc(ms: u64, fps: u64) -> String {
    let total_frames = ms * fps / 1_000;
    let frames = total_frames % fps;
    let total_seconds = total_frames / fps;
    let seconds = total_seconds % 60;
    let total_minutes = total_seconds / 60;
    let minutes = total_minutes % 60;
    let hours = total_minutes / 60;
    format!("{hours:02}:{minutes:02}:{seconds:02}:{frames:02}")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_edl_basic() {
        let cuts = [(0, 1, 0), (5_000, 2, 5_000), (12_000, 1, 12_000)];
        let edl = MulticamCutExporter::to_edl(&cuts);
        assert!(edl.contains("TITLE: MULTICAM EDL"), "EDL should have title");
        assert!(edl.contains("FCM"), "EDL should have FCM line");
        assert!(edl.contains("001"), "First event should be numbered 001");
        assert!(edl.contains("002"), "Second event should be numbered 002");
        assert!(edl.contains("003"), "Third event should be numbered 003");
    }

    #[test]
    fn test_to_edl_empty() {
        let cuts: [(u64, u64, u64); 0] = [];
        let edl = MulticamCutExporter::to_edl(&cuts);
        assert!(edl.contains("TITLE"), "Empty cut list should still produce header");
    }

    #[test]
    fn test_to_edl_single_cut() {
        let cuts = [(1_000, 3, 0)];
        let edl = MulticamCutExporter::to_edl(&cuts);
        assert!(edl.contains("CAM003"), "Camera ID should appear as reel name");
    }

    #[test]
    fn test_to_csv_basic() {
        let cuts = [(0, 1, 0), (5_000, 2, 5_000)];
        let csv = MulticamCutExporter::to_csv(&cuts);
        assert!(csv.starts_with("event,camera_id,offset_ms,timecode_ms\n"));
        assert!(csv.contains("1,1,0,0"), "First row should match cut data");
        assert!(csv.contains("2,2,5000,5000"), "Second row should match cut data");
    }

    #[test]
    fn test_cut_count() {
        let cuts = [(0u64, 1u64, 0u64), (5_000, 2, 5_000), (12_000, 1, 12_000)];
        assert_eq!(MulticamCutExporter::cut_count(&cuts), 3);
    }

    #[test]
    fn test_total_duration_ms() {
        let cuts = [(0u64, 1u64, 0u64), (10_000, 2, 10_000)];
        assert_eq!(MulticamCutExporter::total_duration_ms(&cuts), 15_000);
    }

    #[test]
    fn test_total_duration_ms_empty() {
        let cuts: [(u64, u64, u64); 0] = [];
        assert_eq!(MulticamCutExporter::total_duration_ms(&cuts), 0);
    }

    #[test]
    fn test_ms_to_tc_one_second() {
        assert_eq!(ms_to_tc(1_000, 25), "00:00:01:00");
    }

    #[test]
    fn test_ms_to_tc_one_minute() {
        assert_eq!(ms_to_tc(60_000, 25), "00:01:00:00");
    }

    #[test]
    fn test_ms_to_tc_one_hour() {
        assert_eq!(ms_to_tc(3_600_000, 25), "01:00:00:00");
    }

    #[test]
    fn test_ms_to_tc_half_second() {
        // 500 ms × 25 fps / 1000 = 12.5 → 12 frames
        assert_eq!(ms_to_tc(500, 25), "00:00:00:12");
    }
}
