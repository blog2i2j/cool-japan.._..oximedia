//! Multi-track alignment for synchronizing multiple audio/video tracks.
//!
//! This module provides tools for aligning multiple tracks to a common reference,
//! using cross-correlation of feature vectors to find the best time offset.

#![allow(dead_code)]

/// Anchor point tying a frame to a feature vector for alignment purposes.
#[derive(Debug, Clone)]
pub struct AlignmentAnchor {
    /// ID of the track this anchor belongs to.
    pub track_id: String,
    /// Frame index within the track.
    pub frame_idx: u64,
    /// Feature vector extracted from this frame.
    pub feature_vector: Vec<f32>,
}

impl AlignmentAnchor {
    /// Create a new alignment anchor.
    #[must_use]
    pub fn new(track_id: impl Into<String>, frame_idx: u64, feature_vector: Vec<f32>) -> Self {
        Self {
            track_id: track_id.into(),
            frame_idx,
            feature_vector,
        }
    }
}

/// Method used to align two tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignMethod {
    /// Cross-correlation of audio waveforms.
    AudioCorrelation,
    /// Matching of visual feature vectors.
    VisualFeature,
    /// Explicit sync markers (clapperboard, flash, …).
    Marker,
    /// LTC/VITC timecode comparison.
    Timecode,
    /// Manually specified offset.
    Manual,
}

/// Result of aligning one track to a reference track.
#[derive(Debug, Clone)]
pub struct TrackAlignment {
    /// ID of the reference (anchor) track.
    pub reference_id: String,
    /// ID of the track that was aligned.
    pub aligned_id: String,
    /// Frame offset to apply: `aligned_frame + offset_frames = reference_frame`.
    pub offset_frames: i64,
    /// Confidence of the alignment in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Method that produced the alignment.
    pub method: AlignMethod,
}

impl TrackAlignment {
    /// Create a new track alignment result.
    #[must_use]
    pub fn new(
        reference_id: impl Into<String>,
        aligned_id: impl Into<String>,
        offset_frames: i64,
        confidence: f32,
        method: AlignMethod,
    ) -> Self {
        Self {
            reference_id: reference_id.into(),
            aligned_id: aligned_id.into(),
            offset_frames,
            confidence,
            method,
        }
    }
}

/// Aligns multiple tracks to a designated reference track using feature
/// cross-correlation.
#[derive(Debug, Default)]
pub struct MultitrackAligner {
    /// ID of the reference track.
    reference_id: Option<String>,
}

impl MultitrackAligner {
    /// Create a new aligner (no reference set yet).
    #[must_use]
    pub fn new() -> Self {
        Self { reference_id: None }
    }

    /// Set the reference track ID.
    pub fn set_reference(&mut self, track_id: &str) {
        self.reference_id = Some(track_id.to_owned());
    }

    /// Align `track_id` to the reference using the supplied anchors.
    ///
    /// For each anchor belonging to `track_id`, the method searches for the
    /// anchor from the reference track with the matching frame index (or the
    /// closest one) and accumulates cross-correlation evidence.  The offset
    /// with the highest aggregate correlation peak is returned.
    ///
    /// # Panics
    /// Panics if no reference track has been set via [`Self::set_reference`].
    #[must_use]
    pub fn align_track(&self, track_id: &str, anchors: &[AlignmentAnchor]) -> TrackAlignment {
        let reference_id = self
            .reference_id
            .as_deref()
            .expect("Reference track must be set before calling align_track");

        // Separate anchors by track.
        let ref_anchors: Vec<&AlignmentAnchor> = anchors
            .iter()
            .filter(|a| a.track_id == reference_id)
            .collect();
        let tgt_anchors: Vec<&AlignmentAnchor> =
            anchors.iter().filter(|a| a.track_id == track_id).collect();

        if ref_anchors.is_empty() || tgt_anchors.is_empty() {
            return TrackAlignment::new(reference_id, track_id, 0, 0.0, AlignMethod::VisualFeature);
        }

        // Compute cross-correlation for each reference/target pair and vote.
        let mut best_offset: i64 = 0;
        let mut best_score: f64 = f64::NEG_INFINITY;
        let mut total_pairs = 0usize;

        for ref_anchor in &ref_anchors {
            for tgt_anchor in &tgt_anchors {
                let corr = cross_correlate(&ref_anchor.feature_vector, &tgt_anchor.feature_vector);
                // The peak position in the cross-correlation array gives the lag.
                if let Some((peak_lag, peak_val)) = corr
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                {
                    // Convert cross-correlation lag to frame offset.
                    let n = ref_anchor.feature_vector.len();
                    let signed_lag = peak_lag as i64 - (n as i64 - 1);
                    let frame_offset =
                        tgt_anchor.frame_idx as i64 - ref_anchor.frame_idx as i64 + signed_lag;

                    if f64::from(*peak_val) > best_score {
                        best_score = f64::from(*peak_val);
                        best_offset = frame_offset;
                    }
                    total_pairs += 1;
                }
            }
        }

        // Confidence is based on peak value normalised to [0, 1].
        let confidence = if total_pairs > 0 {
            (best_score as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };

        TrackAlignment::new(
            reference_id,
            track_id,
            best_offset,
            confidence,
            AlignMethod::VisualFeature,
        )
    }
}

/// Compute the normalised cross-correlation of two equal-length (or different-
/// length) sequences.
///
/// The output has length `len(a) + len(b) - 1`.  Each value is the
/// zero-lag-normalised dot product at that lag, clamped to `[-1, 1]`.
#[must_use]
pub fn cross_correlate(a: &[f32], b: &[f32]) -> Vec<f32> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }

    let na = a.len();
    let nb = b.len();
    let out_len = na + nb - 1;

    // Precompute norms for normalisation.
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = norm_a * norm_b;

    let mut result = vec![0.0f32; out_len];

    for lag in 0..out_len {
        let mut sum = 0.0f32;
        // lag = 0 → b starts at position -(nb-1) relative to a.
        // At lag `lag`, b[j] aligns with a[lag + j - (nb - 1)].
        for j in 0..nb {
            let a_idx = lag + j;
            if a_idx >= nb - 1 && a_idx - (nb - 1) < na {
                let ai = a_idx - (nb - 1);
                sum += a[ai] * b[j];
            }
        }
        result[lag] = if denom > 1e-10 {
            (sum / denom).clamp(-1.0, 1.0)
        } else {
            0.0
        };
    }

    result
}

/// Pairwise frame-offset matrix between a set of tracks.
#[derive(Debug, Clone)]
pub struct AlignmentMatrix {
    /// Track IDs (row/column labels).
    pub tracks: Vec<String>,
    /// `offsets[i][j]` = frame offset of track `j` relative to track `i`.
    pub offsets: Vec<Vec<i64>>,
}

impl AlignmentMatrix {
    /// Create a new alignment matrix from a list of pairwise [`TrackAlignment`]s.
    ///
    /// The matrix is filled symmetrically: if `alignment.offset_frames` is the
    /// offset from `reference_id` to `aligned_id`, then `[ref_idx][aln_idx] =
    /// offset` and `[aln_idx][ref_idx] = -offset`.
    #[must_use]
    pub fn from_alignments(track_ids: &[&str], alignments: &[TrackAlignment]) -> Self {
        let n = track_ids.len();
        let tracks: Vec<String> = track_ids.iter().map(|s| (*s).to_string()).collect();
        let mut offsets = vec![vec![0i64; n]; n];

        let idx = |id: &str| tracks.iter().position(|t| t == id);

        for aln in alignments {
            if let (Some(ri), Some(ai)) = (idx(&aln.reference_id), idx(&aln.aligned_id)) {
                offsets[ri][ai] = aln.offset_frames;
                offsets[ai][ri] = -aln.offset_frames;
            }
        }

        Self { tracks, offsets }
    }

    /// Compute a single global offset for each track (relative to track 0)
    /// using a least-squares approach.
    ///
    /// The method averages all available pairwise constraints for each track,
    /// propagating offsets transitively through the matrix.  Track 0 is fixed
    /// at offset 0.
    #[must_use]
    pub fn compute_global_offsets(&self) -> Vec<i64> {
        let n = self.tracks.len();
        if n == 0 {
            return Vec::new();
        }

        // Initialise with direct offsets from track 0.
        let mut global: Vec<f64> = (0..n).map(|j| self.offsets[0][j] as f64).collect();
        global[0] = 0.0;

        // Iteratively refine: for each track i, average all constraints
        // `global[j] + offsets[j][i]` over all j != i.
        // Two passes is enough for a dense matrix; more for sparse.
        for _ in 0..3 {
            let prev = global.clone();
            for i in 1..n {
                let mut sum = 0.0f64;
                let mut count = 0usize;
                for j in 0..n {
                    if j != i && self.offsets[j][i] != 0 {
                        sum += prev[j] + self.offsets[j][i] as f64;
                        count += 1;
                    }
                }
                if count > 0 {
                    global[i] = sum / count as f64;
                }
            }
        }

        global.iter().map(|&v| v.round() as i64).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── cross_correlate ───────────────────────────────────────────────────────

    #[test]
    fn test_cross_correlate_empty() {
        assert!(cross_correlate(&[], &[1.0]).is_empty());
        assert!(cross_correlate(&[1.0], &[]).is_empty());
    }

    #[test]
    fn test_cross_correlate_identical() {
        let a = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let corr = cross_correlate(&a, &a);
        // Peak should be at the centre lag (zero lag).
        let peak_idx = corr
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).expect("max_by should succeed"))
            .map(|(i, _)| i)
            .expect("test expectation failed");
        let zero_lag = a.len() - 1; // centre lag index
        assert_eq!(peak_idx, zero_lag, "peak should be at zero lag");
    }

    #[test]
    fn test_cross_correlate_output_length() {
        let a = vec![1.0f32; 5];
        let b = vec![1.0f32; 3];
        let corr = cross_correlate(&a, &b);
        assert_eq!(corr.len(), a.len() + b.len() - 1);
    }

    #[test]
    fn test_cross_correlate_values_in_range() {
        let a: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..8).map(|i| (8 - i) as f32).collect();
        let corr = cross_correlate(&a, &b);
        for &v in &corr {
            assert!(v >= -1.0 && v <= 1.0, "value {v} out of [-1, 1]");
        }
    }

    #[test]
    fn test_cross_correlate_zero_signal() {
        let a = vec![0.0f32; 4];
        let b = vec![1.0f32; 4];
        let corr = cross_correlate(&a, &b);
        assert!(
            corr.iter().all(|&v| v == 0.0),
            "zero signal should yield zero correlation"
        );
    }

    // ── AlignmentAnchor ───────────────────────────────────────────────────────

    #[test]
    fn test_anchor_creation() {
        let anchor = AlignmentAnchor::new("cam_a", 42, vec![1.0, 2.0, 3.0]);
        assert_eq!(anchor.track_id, "cam_a");
        assert_eq!(anchor.frame_idx, 42);
        assert_eq!(anchor.feature_vector.len(), 3);
    }

    // ── TrackAlignment ────────────────────────────────────────────────────────

    #[test]
    fn test_track_alignment_fields() {
        let aln = TrackAlignment::new("ref", "tgt", -5, 0.9, AlignMethod::AudioCorrelation);
        assert_eq!(aln.reference_id, "ref");
        assert_eq!(aln.aligned_id, "tgt");
        assert_eq!(aln.offset_frames, -5);
        assert!((aln.confidence - 0.9).abs() < f32::EPSILON);
        assert_eq!(aln.method, AlignMethod::AudioCorrelation);
    }

    // ── MultitrackAligner ─────────────────────────────────────────────────────

    #[test]
    fn test_aligner_no_anchors() {
        let mut aligner = MultitrackAligner::new();
        aligner.set_reference("ref");
        let anchors: Vec<AlignmentAnchor> = vec![];
        let result = aligner.align_track("tgt", &anchors);
        assert_eq!(result.offset_frames, 0);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_aligner_identical_features() {
        let mut aligner = MultitrackAligner::new();
        aligner.set_reference("ref");

        let fv = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let anchors = vec![
            AlignmentAnchor::new("ref", 10, fv.clone()),
            AlignmentAnchor::new("tgt", 10, fv.clone()),
        ];
        let result = aligner.align_track("tgt", &anchors);
        // With identical features at the same frame, offset should be near 0.
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_aligner_sets_reference() {
        let mut aligner = MultitrackAligner::new();
        assert!(aligner.reference_id.is_none());
        aligner.set_reference("master");
        assert_eq!(aligner.reference_id.as_deref(), Some("master"));
    }

    // ── AlignmentMatrix ───────────────────────────────────────────────────────

    #[test]
    fn test_alignment_matrix_identity() {
        let tracks = vec!["a", "b", "c"];
        let alignments: Vec<TrackAlignment> = vec![];
        let matrix = AlignmentMatrix::from_alignments(&tracks, &alignments);
        assert_eq!(matrix.tracks.len(), 3);
        // All offsets default to 0.
        for row in &matrix.offsets {
            for &v in row {
                assert_eq!(v, 0);
            }
        }
    }

    #[test]
    fn test_alignment_matrix_symmetric() {
        let tracks = vec!["a", "b"];
        let alignments = vec![TrackAlignment::new("a", "b", 10, 0.9, AlignMethod::Manual)];
        let matrix = AlignmentMatrix::from_alignments(&tracks, &alignments);
        assert_eq!(matrix.offsets[0][1], 10);
        assert_eq!(matrix.offsets[1][0], -10);
    }

    #[test]
    fn test_compute_global_offsets_empty() {
        let matrix = AlignmentMatrix {
            tracks: vec![],
            offsets: vec![],
        };
        assert!(matrix.compute_global_offsets().is_empty());
    }

    #[test]
    fn test_compute_global_offsets_single() {
        let matrix = AlignmentMatrix {
            tracks: vec!["a".to_string()],
            offsets: vec![vec![0]],
        };
        let offsets = matrix.compute_global_offsets();
        assert_eq!(offsets, vec![0]);
    }

    #[test]
    fn test_compute_global_offsets_two_tracks() {
        let tracks = vec!["a", "b"];
        let alignments = vec![TrackAlignment::new("a", "b", 5, 0.9, AlignMethod::Manual)];
        let matrix = AlignmentMatrix::from_alignments(&tracks, &alignments);
        let global = matrix.compute_global_offsets();
        // Track 0 (a) is reference → offset 0; track 1 (b) should be +5.
        assert_eq!(global[0], 0);
        assert_eq!(global[1], 5);
    }
}
