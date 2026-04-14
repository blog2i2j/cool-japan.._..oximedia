//! Interlace / progressive field detection for video frames.
//!
//! Interlaced video stores two *fields* per frame — the odd lines (top field)
//! and the even lines (bottom field) — captured at different moments in time.
//! When video is displayed progressively the two fields are combined, causing
//! visible *combing* artefacts on moving objects.
//!
//! This module detects whether a frame (or a sequence of frames) is interlaced
//! and, if so, determines the field order (TFF = top-field-first, BFF =
//! bottom-field-first).
//!
//! ## Algorithms
//!
//! * **Combing score** — measures the inter-line difference within a field vs
//!   across fields.  High values indicate combing artefacts typical of
//!   interlaced content.
//! * **Field difference metric (FDM)** — compares adjacent line pairs to detect
//!   the alternating motion pattern that interlaced content produces.
//! * **TFF / BFF discriminator** — computes separate combing scores for the
//!   top and bottom field pairings to determine which field order was used.
//! * **Sequence classifier** — accumulates per-frame decisions over a window
//!   of frames to produce a stable content-type decision.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during interlace detection.
#[derive(Debug, Error)]
pub enum InterlaceDetectError {
    /// The frame buffer is smaller than `width × height`.
    #[error("buffer length {got} < required {expected}")]
    BufferTooSmall {
        /// Actual buffer length.
        got: usize,
        /// Required buffer length (`width × height`).
        expected: usize,
    },
    /// Dimensions are too small for meaningful analysis (need at least 8 rows).
    #[error("frame height {height} is too small (minimum 8)")]
    HeightTooSmall {
        /// Frame height.
        height: u32,
    },
    /// Frame width must be at least 1.
    #[error("frame width {width} is zero")]
    ZeroWidth {
        /// Frame width.
        width: u32,
    },
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Field order for interlaced content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldOrder {
    /// Top field (odd-numbered lines) contains the earlier moment in time.
    TopFieldFirst,
    /// Bottom field (even-numbered lines) contains the earlier moment in time.
    BottomFieldFirst,
    /// Field order could not be determined (insufficient evidence).
    Unknown,
}

/// Classification of a single frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameType {
    /// Progressive frame — no combing detected.
    Progressive,
    /// Interlaced frame — combing artefacts detected.
    Interlaced,
    /// Detection was inconclusive.
    Uncertain,
}

/// Detailed metrics for a single-frame interlace analysis.
#[derive(Debug, Clone)]
pub struct FrameAnalysis {
    /// Overall combing score in the range [0, 1].
    /// Values > [`DEFAULT_COMBING_THRESHOLD`] suggest interlaced content.
    pub combing_score: f64,
    /// Combing score when treating lines 0,2,4,… as the top field.
    pub top_field_score: f64,
    /// Combing score when treating lines 1,3,5,… as the bottom field.
    pub bottom_field_score: f64,
    /// Classification result.
    pub frame_type: FrameType,
    /// Inferred field order (meaningful only when `frame_type == Interlaced`).
    pub field_order: FieldOrder,
}

/// Default combing threshold above which a frame is classified as interlaced.
pub const DEFAULT_COMBING_THRESHOLD: f64 = 0.15;

// ---------------------------------------------------------------------------
// Public API — single-frame analysis
// ---------------------------------------------------------------------------

/// Analyse a single grayscale frame for interlacing artefacts.
///
/// # Parameters
///
/// * `frame` — grayscale pixel data in raster order, one byte per sample.
/// * `width` / `height` — frame dimensions.
/// * `threshold` — combing-score threshold; use [`DEFAULT_COMBING_THRESHOLD`]
///   if unsure.
///
/// # Errors
///
/// Returns [`InterlaceDetectError`] if validation fails.
pub fn analyse_frame(
    frame: &[u8],
    width: u32,
    height: u32,
    threshold: f64,
) -> Result<FrameAnalysis, InterlaceDetectError> {
    let w = width as usize;
    let h = height as usize;

    if width == 0 {
        return Err(InterlaceDetectError::ZeroWidth { width });
    }
    if height < 8 {
        return Err(InterlaceDetectError::HeightTooSmall { height });
    }
    if frame.len() < w * h {
        return Err(InterlaceDetectError::BufferTooSmall {
            got: frame.len(),
            expected: w * h,
        });
    }

    let top_field_score = combing_score_for_field(frame, w, h, FieldParity::Even);
    let bottom_field_score = combing_score_for_field(frame, w, h, FieldParity::Odd);
    let combing_score = top_field_score.max(bottom_field_score);

    let frame_type = if combing_score > threshold {
        FrameType::Interlaced
    } else if combing_score < threshold * 0.5 {
        FrameType::Progressive
    } else {
        FrameType::Uncertain
    };

    let field_order = if frame_type == FrameType::Interlaced {
        classify_field_order(top_field_score, bottom_field_score)
    } else {
        FieldOrder::Unknown
    };

    Ok(FrameAnalysis {
        combing_score,
        top_field_score,
        bottom_field_score,
        frame_type,
        field_order,
    })
}

// ---------------------------------------------------------------------------
// Public API — sequence classifier
// ---------------------------------------------------------------------------

/// Accumulates per-frame decisions over a sliding window to produce a stable
/// content-type verdict.
#[derive(Debug, Clone)]
pub struct InterlaceClassifier {
    /// Combing threshold passed to [`analyse_frame`].
    pub threshold: f64,
    /// Window size (number of frames to retain).
    pub window: usize,
    // Ring buffer of per-frame combing scores.
    scores: std::collections::VecDeque<f64>,
    // Ring buffer of per-frame field-order votes.
    field_votes: std::collections::VecDeque<FieldOrder>,
}

impl InterlaceClassifier {
    /// Create a new classifier.
    ///
    /// * `threshold` — combing score threshold (use [`DEFAULT_COMBING_THRESHOLD`]).
    /// * `window`    — number of recent frames to include in the decision.
    pub fn new(threshold: f64, window: usize) -> Self {
        let cap = window.max(1);
        Self {
            threshold,
            window: cap,
            scores: std::collections::VecDeque::with_capacity(cap),
            field_votes: std::collections::VecDeque::with_capacity(cap),
        }
    }

    /// Submit a new frame for analysis and update internal state.
    ///
    /// # Errors
    ///
    /// Propagates any [`InterlaceDetectError`] from [`analyse_frame`].
    pub fn push_frame(
        &mut self,
        frame: &[u8],
        width: u32,
        height: u32,
    ) -> Result<FrameAnalysis, InterlaceDetectError> {
        let analysis = analyse_frame(frame, width, height, self.threshold)?;

        // Maintain sliding window.
        if self.scores.len() >= self.window {
            self.scores.pop_front();
            self.field_votes.pop_front();
        }
        self.scores.push_back(analysis.combing_score);
        self.field_votes.push_back(analysis.field_order);

        Ok(analysis)
    }

    /// Return the current content-type decision based on accumulated history.
    pub fn current_verdict(&self) -> ContentType {
        if self.scores.is_empty() {
            return ContentType::Unknown;
        }

        let mean: f64 = self.scores.iter().sum::<f64>() / self.scores.len() as f64;
        let interlaced_count = self.scores.iter().filter(|&&s| s > self.threshold).count();
        let interlace_fraction = interlaced_count as f64 / self.scores.len() as f64;

        if interlace_fraction > 0.6 {
            let order = self.dominant_field_order();
            ContentType::Interlaced(order)
        } else if mean < self.threshold * 0.5 && interlace_fraction < 0.2 {
            ContentType::Progressive
        } else {
            ContentType::Unknown
        }
    }

    /// Return the dominant (most frequent) non-Unknown field order across the window.
    fn dominant_field_order(&self) -> FieldOrder {
        let mut tff = 0usize;
        let mut bff = 0usize;
        for &fo in &self.field_votes {
            match fo {
                FieldOrder::TopFieldFirst => tff += 1,
                FieldOrder::BottomFieldFirst => bff += 1,
                FieldOrder::Unknown => {}
            }
        }
        if tff > bff {
            FieldOrder::TopFieldFirst
        } else if bff > tff {
            FieldOrder::BottomFieldFirst
        } else {
            FieldOrder::Unknown
        }
    }
}

/// Stable content-type decision from [`InterlaceClassifier`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    /// Progressive content (no combing).
    Progressive,
    /// Interlaced content with detected field order.
    Interlaced(FieldOrder),
    /// Verdict is not yet available or evidence is ambiguous.
    Unknown,
}

// ---------------------------------------------------------------------------
// Public free functions
// ---------------------------------------------------------------------------

/// Compute the raw combing score for a frame without classification.
///
/// The score is in [0, 1].  Higher values indicate more combing artefacts.
///
/// # Errors
///
/// Returns [`InterlaceDetectError`] on validation failure.
pub fn combing_score(frame: &[u8], width: u32, height: u32) -> Result<f64, InterlaceDetectError> {
    let w = width as usize;
    let h = height as usize;
    if width == 0 {
        return Err(InterlaceDetectError::ZeroWidth { width });
    }
    if height < 8 {
        return Err(InterlaceDetectError::HeightTooSmall { height });
    }
    if frame.len() < w * h {
        return Err(InterlaceDetectError::BufferTooSmall {
            got: frame.len(),
            expected: w * h,
        });
    }
    let even = combing_score_for_field(frame, w, h, FieldParity::Even);
    let odd = combing_score_for_field(frame, w, h, FieldParity::Odd);
    Ok(even.max(odd))
}

/// Compute the field difference metric (FDM) between two frames.
///
/// A high FDM between adjacent frames suggests that the content is interlaced —
/// even lines from `frame_a` closely match even lines from `frame_b` but odd
/// lines show large differences (or vice versa).
///
/// Returns the absolute difference between even-line MAD and odd-line MAD,
/// normalised to [0, 1].
///
/// # Errors
///
/// Returns [`InterlaceDetectError`] on validation failure.
pub fn field_difference_metric(
    frame_a: &[u8],
    frame_b: &[u8],
    width: u32,
    height: u32,
) -> Result<f64, InterlaceDetectError> {
    let w = width as usize;
    let h = height as usize;
    if width == 0 {
        return Err(InterlaceDetectError::ZeroWidth { width });
    }
    if height < 8 {
        return Err(InterlaceDetectError::HeightTooSmall { height });
    }
    let expected = w * h;
    if frame_a.len() < expected {
        return Err(InterlaceDetectError::BufferTooSmall {
            got: frame_a.len(),
            expected,
        });
    }
    if frame_b.len() < expected {
        return Err(InterlaceDetectError::BufferTooSmall {
            got: frame_b.len(),
            expected,
        });
    }

    let (even_mad, odd_mad) = inter_frame_field_mad(frame_a, frame_b, w, h);
    Ok((even_mad - odd_mad).abs())
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Identifies which rows belong to each field.
#[derive(Clone, Copy)]
enum FieldParity {
    /// Rows 0, 2, 4, … (top field in TFF content).
    Even,
    /// Rows 1, 3, 5, … (bottom field in TFF content).
    Odd,
}

/// Compute combing score for the given field parity.
///
/// For each row belonging to `parity`, compute the mean absolute difference
/// to the next row.  Compare that to the mean absolute difference between rows
/// that belong to the *same* field (two rows apart).  A high ratio of
/// cross-field to intra-field differences indicates combing.
fn combing_score_for_field(frame: &[u8], w: usize, h: usize, parity: FieldParity) -> f64 {
    let start: usize = match parity {
        FieldParity::Even => 0,
        FieldParity::Odd => 1,
    };

    // Cross-field diff: row[r] vs row[r+1] (different field)
    let mut cross_sum = 0u64;
    let mut cross_n = 0u64;
    // Intra-field diff: row[r] vs row[r+2] (same field)
    let mut intra_sum = 0u64;
    let mut intra_n = 0u64;

    let mut r = start;
    while r + 1 < h {
        // Cross-field: r vs r+1
        let row_a = &frame[r * w..(r + 1) * w];
        let row_b = &frame[(r + 1) * w..(r + 2) * w];
        for (&a, &b) in row_a.iter().zip(row_b.iter()) {
            cross_sum += (a as i32 - b as i32).unsigned_abs() as u64;
        }
        cross_n += w as u64;

        // Intra-field: r vs r+2 (if in bounds)
        if r + 2 < h {
            let row_c = &frame[(r + 2) * w..(r + 3) * w];
            for (&a, &c) in row_a.iter().zip(row_c.iter()) {
                intra_sum += (a as i32 - c as i32).unsigned_abs() as u64;
            }
            intra_n += w as u64;
        }

        r += 2;
    }

    if cross_n == 0 || intra_n == 0 {
        return 0.0;
    }

    let cross_mad = cross_sum as f64 / (cross_n as f64 * 255.0);
    let intra_mad = intra_sum as f64 / (intra_n as f64 * 255.0);

    // Combing score: how much more is cross vs intra?
    if intra_mad < f64::EPSILON {
        if cross_mad > f64::EPSILON {
            1.0
        } else {
            0.0
        }
    } else {
        ((cross_mad - intra_mad) / intra_mad).clamp(0.0, 1.0)
    }
}

/// Compute per-field MAD between two frames.
///
/// Returns `(even_mad, odd_mad)` normalised to [0, 1].
fn inter_frame_field_mad(frame_a: &[u8], frame_b: &[u8], w: usize, h: usize) -> (f64, f64) {
    let (mut even_sum, mut even_n) = (0u64, 0u64);
    let (mut odd_sum, mut odd_n) = (0u64, 0u64);

    for row in 0..h {
        let ra = &frame_a[row * w..(row + 1) * w];
        let rb = &frame_b[row * w..(row + 1) * w];
        let mad: u64 = ra
            .iter()
            .zip(rb.iter())
            .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as u64)
            .sum();
        if row % 2 == 0 {
            even_sum += mad;
            even_n += w as u64;
        } else {
            odd_sum += mad;
            odd_n += w as u64;
        }
    }

    let even_mad = if even_n > 0 {
        even_sum as f64 / (even_n as f64 * 255.0)
    } else {
        0.0
    };
    let odd_mad = if odd_n > 0 {
        odd_sum as f64 / (odd_n as f64 * 255.0)
    } else {
        0.0
    };
    (even_mad, odd_mad)
}

/// Determine field order from per-field combing scores.
fn classify_field_order(top_score: f64, bottom_score: f64) -> FieldOrder {
    let diff = (top_score - bottom_score).abs();
    if diff < 0.01 {
        // Scores are too similar to distinguish.
        FieldOrder::Unknown
    } else if top_score > bottom_score {
        // Top field shows more combing → top field is the *later* field in time
        // which is consistent with Bottom-Field-First ordering.
        FieldOrder::BottomFieldFirst
    } else {
        FieldOrder::TopFieldFirst
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn progressive_frame(w: u32, h: u32) -> Vec<u8> {
        // Smooth gradient: no combing expected.
        (0..(w * h) as usize).map(|i| (i % 256) as u8).collect()
    }

    fn interlaced_frame(w: u32, h: u32) -> Vec<u8> {
        // Simulate combing: even lines bright, odd lines dark.
        (0..(w * h) as usize)
            .map(|i| {
                let row = i / w as usize;
                if row % 2 == 0 {
                    220
                } else {
                    30
                }
            })
            .collect()
    }

    // 1. Flat frame → combing_score near zero
    #[test]
    fn test_flat_frame_not_interlaced() {
        let frame: Vec<u8> = vec![128u8; 16 * 16];
        let score = combing_score(&frame, 16, 16).unwrap();
        // Flat → all diffs zero → score 0
        assert!(score < 0.01, "flat frame should score near 0, got {score}");
    }

    // 2. Interlaced-pattern frame → high combing score
    #[test]
    fn test_interlaced_pattern_high_score() {
        let frame = interlaced_frame(32, 32);
        let score = combing_score(&frame, 32, 32).unwrap();
        assert!(
            score > DEFAULT_COMBING_THRESHOLD,
            "interlaced frame should exceed threshold, got {score}"
        );
    }

    // 3. Progressive ramp → classified as Progressive or Uncertain
    #[test]
    fn test_progressive_frame_classified_correctly() {
        let frame = progressive_frame(32, 32);
        let analysis = analyse_frame(&frame, 32, 32, DEFAULT_COMBING_THRESHOLD).unwrap();
        assert_ne!(
            analysis.frame_type,
            FrameType::Interlaced,
            "smooth ramp should not be classified as interlaced"
        );
    }

    // 4. Height too small returns error
    #[test]
    fn test_height_too_small_error() {
        let frame = vec![0u8; 16 * 4];
        let err = combing_score(&frame, 16, 4);
        assert!(matches!(
            err,
            Err(InterlaceDetectError::HeightTooSmall { .. })
        ));
    }

    // 5. Buffer too small returns error
    #[test]
    fn test_buffer_too_small_error() {
        let short = vec![0u8; 8];
        let err = combing_score(&short, 16, 16);
        assert!(matches!(
            err,
            Err(InterlaceDetectError::BufferTooSmall { .. })
        ));
    }

    // 6. Zero width returns error
    #[test]
    fn test_zero_width_error() {
        let frame = vec![0u8; 16];
        let err = combing_score(&frame, 0, 16);
        assert!(matches!(err, Err(InterlaceDetectError::ZeroWidth { .. })));
    }

    // 7. field_difference_metric: identical frames → near zero
    #[test]
    fn test_fdm_identical_frames_near_zero() {
        let frame = progressive_frame(16, 16);
        let fdm = field_difference_metric(&frame, &frame, 16, 16).unwrap();
        assert!(
            fdm < 0.05,
            "identical frames should have near-zero FDM, got {fdm}"
        );
    }

    // 8. InterlaceClassifier: interlaced frames → interlaced verdict
    #[test]
    fn test_classifier_interlaced_verdict() {
        let mut clf = InterlaceClassifier::new(DEFAULT_COMBING_THRESHOLD, 8);
        let frame = interlaced_frame(32, 32);
        for _ in 0..8 {
            clf.push_frame(&frame, 32, 32).unwrap();
        }
        let verdict = clf.current_verdict();
        assert!(
            matches!(verdict, ContentType::Interlaced(_)),
            "expected Interlaced verdict, got {verdict:?}"
        );
    }

    // 9. InterlaceClassifier: progressive frames → Progressive verdict
    #[test]
    fn test_classifier_progressive_verdict() {
        let mut clf = InterlaceClassifier::new(DEFAULT_COMBING_THRESHOLD, 8);
        let frame = vec![128u8; 32 * 32]; // flat → low score
        for _ in 0..8 {
            clf.push_frame(&frame, 32, 32).unwrap();
        }
        let verdict = clf.current_verdict();
        assert_eq!(
            verdict,
            ContentType::Progressive,
            "expected Progressive verdict, got {verdict:?}"
        );
    }

    // 10. analyse_frame: combing scores are in [0, 1]
    #[test]
    fn test_combing_scores_in_range() {
        let frame = interlaced_frame(32, 32);
        let analysis = analyse_frame(&frame, 32, 32, DEFAULT_COMBING_THRESHOLD).unwrap();
        assert!(analysis.combing_score >= 0.0 && analysis.combing_score <= 1.0);
        assert!(analysis.top_field_score >= 0.0 && analysis.top_field_score <= 1.0);
        assert!(analysis.bottom_field_score >= 0.0 && analysis.bottom_field_score <= 1.0);
    }
}
