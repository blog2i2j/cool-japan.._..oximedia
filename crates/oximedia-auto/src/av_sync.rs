//! Audio-visual synchronisation validation.
//!
//! [`AvSyncValidator`] cross-references audio peak timestamps with visual cut
//! timestamps and reports any pairs that are not temporally aligned within a
//! configurable tolerance.
//!
//! # Algorithm
//!
//! For each audio peak the validator finds the closest visual cut. If the
//! absolute time difference exceeds `tolerance_ms`, a [`SyncIssue`] is
//! recorded. The validation also runs in the reverse direction: for each
//! visual cut the closest audio peak is found and checked against the same
//! tolerance. Duplicate pairs are de-duplicated so each pair appears at most
//! once in the result.
//!
//! # Use cases
//!
//! - Verify that music edit points align with visual cuts (music video QC).
//! - Validate lip-sync by cross-referencing audio transients with mouth-open
//!   visual events.
//! - Post-render QC to confirm automated A/V sync was preserved.
//!
//! # Example
//!
//! ```rust
//! use oximedia_auto::av_sync::{AvSyncValidator, SyncIssue};
//!
//! let audio_peaks  = vec![0_u64, 1_000, 2_000, 3_000];
//! let visual_cuts  = vec![0_u64, 1_005, 2_100, 3_000]; // 3rd cut 100 ms late
//! let tolerance_ms = 50_u64;
//!
//! let issues = AvSyncValidator::check(&audio_peaks, &visual_cuts, tolerance_ms);
//!
//! // Only the 3rd pair (2000 vs 2100, diff = 100 ms) should be flagged.
//! assert_eq!(issues.len(), 1);
//! assert_eq!(issues[0].audio_peak_ms, 2_000);
//! assert_eq!(issues[0].visual_cut_ms, 2_100);
//! assert_eq!(issues[0].delta_ms, 100);
//! ```

#![allow(dead_code, clippy::cast_sign_loss, clippy::cast_possible_wrap)]

// ---------------------------------------------------------------------------
// SyncIssue
// ---------------------------------------------------------------------------

/// A single A/V synchronisation discrepancy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncIssue {
    /// Timestamp of the audio peak (ms from start of stream).
    pub audio_peak_ms: u64,
    /// Timestamp of the closest visual cut (ms from start of stream).
    pub visual_cut_ms: u64,
    /// Absolute time difference `|audio_peak_ms − visual_cut_ms|` in ms.
    pub delta_ms: u64,
    /// Tolerance threshold that was exceeded (ms).
    pub tolerance_ms: u64,
}

impl SyncIssue {
    /// Short human-readable description of this sync issue.
    #[must_use]
    pub fn description(&self) -> String {
        format!(
            "A/V sync issue: audio peak at {}ms, visual cut at {}ms (delta {}ms, tolerance {}ms)",
            self.audio_peak_ms, self.visual_cut_ms, self.delta_ms, self.tolerance_ms
        )
    }
}

impl std::fmt::Display for SyncIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.description())
    }
}

// ---------------------------------------------------------------------------
// AvSyncValidator
// ---------------------------------------------------------------------------

/// Audio-visual synchronisation validator.
///
/// All methods are pure functions; no instance state is required.
pub struct AvSyncValidator;

impl AvSyncValidator {
    /// Validate A/V synchronisation between audio peaks and visual cuts.
    ///
    /// For each audio peak, finds the nearest visual cut. If the difference
    /// exceeds `tolerance_ms`, a [`SyncIssue`] is recorded. The same check
    /// is then run for each visual cut → nearest audio peak, and any new
    /// violations are appended.
    ///
    /// The returned list is sorted by `audio_peak_ms` ascending.
    ///
    /// # Arguments
    ///
    /// * `audio_peaks` — sorted or unsorted list of audio peak timestamps (ms).
    /// * `visual_cuts` — sorted or unsorted list of visual cut timestamps (ms).
    /// * `tolerance_ms` — maximum allowed discrepancy (inclusive).
    ///
    /// Returns an empty `Vec` when either input list is empty or all pairs
    /// are within tolerance.
    #[must_use]
    pub fn check(audio_peaks: &[u64], visual_cuts: &[u64], tolerance_ms: u64) -> Vec<SyncIssue> {
        if audio_peaks.is_empty() || visual_cuts.is_empty() {
            return Vec::new();
        }

        // Use a set to de-duplicate (audio_peak, visual_cut) pairs.
        let mut seen: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::new();
        let mut issues: Vec<SyncIssue> = Vec::new();

        // Pass 1: for each audio peak, find the closest visual cut.
        for &ap in audio_peaks {
            if let Some(&vc) = Self::nearest(ap, visual_cuts) {
                let delta = ap.abs_diff(vc);
                if delta > tolerance_ms && seen.insert((ap, vc)) {
                    issues.push(SyncIssue {
                        audio_peak_ms: ap,
                        visual_cut_ms: vc,
                        delta_ms: delta,
                        tolerance_ms,
                    });
                }
            }
        }

        // Pass 2: for each visual cut, find the closest audio peak.
        for &vc in visual_cuts {
            if let Some(&ap) = Self::nearest(vc, audio_peaks) {
                let delta = vc.abs_diff(ap);
                if delta > tolerance_ms && seen.insert((ap, vc)) {
                    issues.push(SyncIssue {
                        audio_peak_ms: ap,
                        visual_cut_ms: vc,
                        delta_ms: delta,
                        tolerance_ms,
                    });
                }
            }
        }

        // Sort by audio_peak_ms for deterministic output.
        issues.sort_by_key(|i| (i.audio_peak_ms, i.visual_cut_ms));
        issues
    }

    /// Find the element in `timestamps` closest (by absolute difference) to
    /// `target`.  Returns `None` for empty input.
    fn nearest(target: u64, timestamps: &[u64]) -> Option<&u64> {
        timestamps.iter().min_by_key(|&&t| t.abs_diff(target))
    }

    /// Report whether a sequence is considered "in sync" (no issues detected).
    #[must_use]
    pub fn is_in_sync(audio_peaks: &[u64], visual_cuts: &[u64], tolerance_ms: u64) -> bool {
        Self::check(audio_peaks, visual_cuts, tolerance_ms).is_empty()
    }

    /// Compute the maximum A/V delta across all nearest-pair matches.
    ///
    /// Returns `None` when either input is empty.
    #[must_use]
    pub fn max_delta_ms(audio_peaks: &[u64], visual_cuts: &[u64]) -> Option<u64> {
        if audio_peaks.is_empty() || visual_cuts.is_empty() {
            return None;
        }
        audio_peaks
            .iter()
            .filter_map(|&ap| Self::nearest(ap, visual_cuts).map(|&vc| ap.abs_diff(vc)))
            .max()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_audio_peaks_returns_empty() {
        let issues = AvSyncValidator::check(&[], &[1_000, 2_000], 50);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_empty_visual_cuts_returns_empty() {
        let issues = AvSyncValidator::check(&[1_000, 2_000], &[], 50);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_perfectly_synced_no_issues() {
        let peaks = vec![0_u64, 1_000, 2_000, 3_000];
        let cuts = vec![0_u64, 1_000, 2_000, 3_000];
        let issues = AvSyncValidator::check(&peaks, &cuts, 50);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_within_tolerance_no_issues() {
        let peaks = vec![0_u64, 1_000, 2_000];
        let cuts = vec![30_u64, 1_040, 2_050]; // all within 50 ms
        let issues = AvSyncValidator::check(&peaks, &cuts, 50);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_single_out_of_sync_pair() {
        let peaks = vec![0_u64, 1_000, 2_000, 3_000];
        let cuts = vec![0_u64, 1_005, 2_100, 3_000];
        let issues = AvSyncValidator::check(&peaks, &cuts, 50);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].audio_peak_ms, 2_000);
        assert_eq!(issues[0].visual_cut_ms, 2_100);
        assert_eq!(issues[0].delta_ms, 100);
    }

    #[test]
    fn test_multiple_issues() {
        let peaks = vec![0_u64, 500, 1_000];
        let cuts = vec![200_u64, 700, 1_300]; // all differ by 200 ms > 50 ms tolerance
        let issues = AvSyncValidator::check(&peaks, &cuts, 50);
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_is_in_sync_true() {
        let peaks = vec![0_u64, 1_000];
        let cuts = vec![10_u64, 990];
        assert!(AvSyncValidator::is_in_sync(&peaks, &cuts, 50));
    }

    #[test]
    fn test_is_in_sync_false() {
        let peaks = vec![0_u64, 1_000];
        let cuts = vec![0_u64, 1_200]; // 200 ms gap
        assert!(!AvSyncValidator::is_in_sync(&peaks, &cuts, 100));
    }

    #[test]
    fn test_max_delta_ms() {
        let peaks = vec![0_u64, 1_000, 2_000];
        let cuts = vec![0_u64, 1_050, 2_300]; // deltas: 0, 50, 300
        let max = AvSyncValidator::max_delta_ms(&peaks, &cuts);
        assert_eq!(max, Some(300));
    }

    #[test]
    fn test_max_delta_empty_returns_none() {
        assert_eq!(AvSyncValidator::max_delta_ms(&[], &[1_000]), None);
        assert_eq!(AvSyncValidator::max_delta_ms(&[1_000], &[]), None);
    }

    #[test]
    fn test_issues_sorted_by_audio_peak() {
        let peaks = vec![2_000_u64, 0, 1_000];
        let cuts = vec![2_500_u64, 500, 1_500];
        let issues = AvSyncValidator::check(&peaks, &cuts, 50);
        for w in issues.windows(2) {
            assert!(
                w[0].audio_peak_ms <= w[1].audio_peak_ms,
                "Issues not sorted: {} > {}",
                w[0].audio_peak_ms,
                w[1].audio_peak_ms
            );
        }
    }

    #[test]
    fn test_sync_issue_display() {
        let issue = SyncIssue {
            audio_peak_ms: 1_000,
            visual_cut_ms: 1_150,
            delta_ms: 150,
            tolerance_ms: 50,
        };
        let s = format!("{issue}");
        assert!(s.contains("1000"));
        assert!(s.contains("150"));
    }
}
