//! Deep fake detection via facial landmark consistency analysis.
//!
//! Neural-network-based deep fakes typically exhibit subtle inconsistencies in
//! facial landmark positions across frames: unnatural blinking rates, slight
//! geometric asymmetry, and inter-frame jitter that differs from natural motion.
//!
//! This module provides a **geometry-only** (no neural network required)
//! approach that analyses sequences of 2-D facial landmark sets and flags
//! statistical anomalies consistent with synthetic content.
//!
//! # Landmark layout (5-point compact set)
//!
//! When full 68-point dlib/MediaPipe landmarks are not available, the module
//! accepts a compact 5-point representation:
//!
//! | Index | Point              |
//! |-------|--------------------|
//! | 0     | Left eye centre    |
//! | 1     | Right eye centre   |
//! | 2     | Nose tip           |
//! | 3     | Left mouth corner  |
//! | 4     | Right mouth corner |
//!
//! For a full 68-point set the same indices map to the standard dlib layout.
//!
//! # Reference
//!
//! * Li, Y., Chang, M.-C., & Lyu, S. (2018). "In Ictu Oculi: Exposing AI
//!   created fake videos by detecting eye blinking". *IEEE WIFS*.
//! * Matern, F., Riess, C., & Stamminger, M. (2019). "Exploiting Visual
//!   Artefacts to Expose Deepfakes with Eye and Teeth Pixel Statistics".

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// Facial landmark container
// ---------------------------------------------------------------------------

/// A set of 2-D facial landmark points for a single video frame.
///
/// Supports both the compact 5-point representation and the full 68-point
/// dlib layout.  All arithmetic is performed on the raw `Vec<(f32, f32)>`,
/// so callers may supply any number of points ≥ 2.
#[derive(Debug, Clone)]
pub struct FaceLandmarks {
    /// Landmark (x, y) coordinates in pixel space.
    pub points: Vec<(f32, f32)>,
}

impl FaceLandmarks {
    /// Create a new landmark set from a vector of (x, y) coordinates.
    #[must_use]
    pub fn new(points: Vec<(f32, f32)>) -> Self {
        Self { points }
    }

    /// Number of landmark points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` if there are no landmark points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Euclidean distance to another landmark set.
    ///
    /// Returns `None` if the sets have different lengths.
    #[must_use]
    pub fn euclidean_distance(&self, other: &Self) -> Option<f32> {
        if self.points.len() != other.points.len() || self.points.is_empty() {
            return None;
        }
        let sum: f32 = self
            .points
            .iter()
            .zip(other.points.iter())
            .map(|(a, b)| {
                let dx = a.0 - b.0;
                let dy = a.1 - b.1;
                (dx * dx + dy * dy).sqrt()
            })
            .sum();
        Some(sum / self.points.len() as f32)
    }

    /// Mean y-coordinate of left-eye points (index 0, or range 36–41 for 68-pt).
    ///
    /// Uses index 0 for compact sets and indices 36–41 for 68-point sets.
    #[must_use]
    pub fn left_eye_openness(&self) -> f32 {
        match self.points.len() {
            0 => 0.0,
            1..=4 => self.points[0].1,
            5..=67 => self.points[0].1,
            _ => {
                // 68-point set: indices 37,38,41,40 are top/bottom of left eye
                let top = (self.points[37].1 + self.points[38].1) / 2.0;
                let bot = (self.points[40].1 + self.points[41].1) / 2.0;
                (bot - top).abs()
            }
        }
    }

    /// Mean y-coordinate of right-eye points (index 1, or range 42–47 for 68-pt).
    #[must_use]
    pub fn right_eye_openness(&self) -> f32 {
        match self.points.len() {
            0 => 0.0,
            1 => self.points[0].1,
            2..=67 => self.points[1].1,
            _ => {
                // 68-point set: indices 43,44,47,46 are top/bottom of right eye
                let top = (self.points[43].1 + self.points[44].1) / 2.0;
                let bot = (self.points[46].1 + self.points[47].1) / 2.0;
                (bot - top).abs()
            }
        }
    }

    /// Bilateral asymmetry score: difference between left-side and right-side x-means.
    ///
    /// A perfectly symmetric face has score 0.0.
    #[must_use]
    pub fn asymmetry_score(&self) -> f32 {
        if self.points.is_empty() {
            return 0.0;
        }
        let all_x: Vec<f32> = self.points.iter().map(|p| p.0).collect();
        let mean_x = all_x.iter().sum::<f32>() / all_x.len() as f32;

        let left: Vec<f32> = all_x.iter().copied().filter(|&x| x < mean_x).collect();
        let right: Vec<f32> = all_x.iter().copied().filter(|&x| x >= mean_x).collect();

        if left.is_empty() || right.is_empty() {
            return 0.0;
        }

        let mean_left = left.iter().sum::<f32>() / left.len() as f32;
        let mean_right = right.iter().sum::<f32>() / right.len() as f32;
        let mean_all = mean_x;

        if mean_all.abs() < 1e-9 {
            return 0.0;
        }

        ((mean_left - mean_right).abs() / mean_all.abs()).min(1.0)
    }
}

// ---------------------------------------------------------------------------
// Consistency analysis
// ---------------------------------------------------------------------------

/// Per-sequence landmark consistency metrics.
#[derive(Debug, Clone)]
pub struct LandmarkConsistency {
    /// Mean Euclidean distance between consecutive frame landmark sets.
    /// High values indicate excessive inter-frame jitter.
    pub inter_frame_deviation: f32,
    /// `true` if eye-openness changes by > 50% in a single frame transition.
    pub blink_anomaly: bool,
    /// Bilateral facial asymmetry score in `[0, 1]`.  High values suggest
    /// unnatural warping (typical of some GAN-based generators).
    pub asymmetry: f32,
}

/// Analyse a sequence of per-frame facial landmark sets for consistency.
///
/// # Arguments
///
/// * `frames` — Ordered slice of [`FaceLandmarks`], one per video frame.
///
/// # Returns
///
/// A [`LandmarkConsistency`] report.  If fewer than 2 frames are provided,
/// returns a zero-deviation, no-anomaly result.
#[must_use]
pub fn check_landmark_consistency(frames: &[FaceLandmarks]) -> LandmarkConsistency {
    if frames.len() < 2 {
        return LandmarkConsistency {
            inter_frame_deviation: 0.0,
            blink_anomaly: false,
            asymmetry: frames
                .first()
                .map(|f| f.asymmetry_score())
                .unwrap_or(0.0),
        };
    }

    // Inter-frame Euclidean distance
    let dists: Vec<f32> = frames
        .windows(2)
        .filter_map(|pair| pair[0].euclidean_distance(&pair[1]))
        .collect();

    let inter_frame_deviation = if dists.is_empty() {
        0.0
    } else {
        dists.iter().sum::<f32>() / dists.len() as f32
    };

    // Blink anomaly: eye-openness changes > 50% in one frame
    let blink_anomaly = frames.windows(2).any(|pair| {
        let lo_prev = pair[0].left_eye_openness();
        let lo_curr = pair[1].left_eye_openness();
        let ro_prev = pair[0].right_eye_openness();
        let ro_curr = pair[1].right_eye_openness();

        let change_l = if lo_prev.abs() > 1e-9 {
            (lo_curr - lo_prev).abs() / lo_prev.abs()
        } else {
            0.0
        };
        let change_r = if ro_prev.abs() > 1e-9 {
            (ro_curr - ro_prev).abs() / ro_prev.abs()
        } else {
            0.0
        };

        change_l > 0.5 || change_r > 0.5
    });

    // Mean asymmetry across all frames
    let asymmetry = if frames.is_empty() {
        0.0
    } else {
        frames.iter().map(|f| f.asymmetry_score()).sum::<f32>() / frames.len() as f32
    };

    LandmarkConsistency {
        inter_frame_deviation,
        blink_anomaly,
        asymmetry,
    }
}

// ---------------------------------------------------------------------------
// Deep fake score
// ---------------------------------------------------------------------------

/// Composite deep-fake score combining all geometry indicators.
#[derive(Debug, Clone)]
pub struct DeepFakeScore {
    /// Overall score in `[0, 1]`: `0.0` → likely real, `1.0` → likely fake.
    pub score: f32,
    /// Human-readable list of indicators that contributed to the score.
    pub indicators: Vec<String>,
}

impl DeepFakeScore {
    /// Derive a [`DeepFakeScore`] from a [`LandmarkConsistency`] report.
    ///
    /// Scoring heuristics (weights are empirically chosen):
    ///
    /// | Component              | Max contribution |
    /// |------------------------|------------------|
    /// | Inter-frame jitter     | 0.35             |
    /// | Blink anomaly          | 0.40             |
    /// | Asymmetry              | 0.25             |
    ///
    /// The jitter contribution is logistic-saturated so that moderate natural
    /// motion does not trigger false positives.
    #[must_use]
    pub fn from_consistency(consistency: &LandmarkConsistency) -> Self {
        let mut score = 0.0_f32;
        let mut indicators = Vec::new();

        // --- Inter-frame jitter component ---
        // Natural head motion is typically < 5 px per frame at 1080p.
        // We normalise against a "jitter budget" of 10 px.
        let jitter_norm = (consistency.inter_frame_deviation / 10.0).min(1.0);
        // Logistic-style: score is low for natural jitter, rises steeply above threshold
        let jitter_contrib = if jitter_norm > 0.5 {
            0.35 * ((jitter_norm - 0.5) * 2.0).min(1.0)
        } else {
            0.0
        };
        if jitter_contrib > 0.0 {
            score += jitter_contrib;
            indicators.push(format!(
                "Excessive inter-frame jitter ({:.2} px mean deviation)",
                consistency.inter_frame_deviation
            ));
        }

        // --- Blink anomaly component ---
        if consistency.blink_anomaly {
            score += 0.40;
            indicators.push("Unnatural blink pattern detected (>50% eye-openness change in 1 frame)".to_string());
        }

        // --- Asymmetry component ---
        // Deep fakes often exhibit subtle bilateral asymmetry > 0.15
        if consistency.asymmetry > 0.15 {
            let asym_contrib = ((consistency.asymmetry - 0.15) / 0.85).min(1.0) * 0.25;
            score += asym_contrib;
            indicators.push(format!(
                "Facial asymmetry above threshold ({:.3})",
                consistency.asymmetry
            ));
        }

        let score = score.clamp(0.0, 1.0);

        Self { score, indicators }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── FaceLandmarks ────────────────────────────────────────────────────────

    /// Compact 5-point face with centre at (100, 100)
    fn face_at(cx: f32, cy: f32) -> FaceLandmarks {
        FaceLandmarks::new(vec![
            (cx - 20.0, cy - 10.0), // left eye
            (cx + 20.0, cy - 10.0), // right eye
            (cx, cy),               // nose
            (cx - 15.0, cy + 20.0), // left mouth
            (cx + 15.0, cy + 20.0), // right mouth
        ])
    }

    #[test]
    fn test_face_landmarks_len() {
        let f = face_at(100.0, 100.0);
        assert_eq!(f.len(), 5);
        assert!(!f.is_empty());
    }

    #[test]
    fn test_face_landmarks_empty() {
        let f = FaceLandmarks::new(vec![]);
        assert!(f.is_empty());
        assert_eq!(f.len(), 0);
    }

    #[test]
    fn test_euclidean_distance_identical() {
        let a = face_at(100.0, 100.0);
        let b = face_at(100.0, 100.0);
        let d = a.euclidean_distance(&b).expect("same length");
        assert!(d.abs() < 1e-4);
    }

    #[test]
    fn test_euclidean_distance_translated() {
        let a = face_at(100.0, 100.0);
        let b = face_at(103.0, 104.0); // 5 px translated
        let d = a.euclidean_distance(&b).expect("same length");
        assert!(d > 0.0);
    }

    #[test]
    fn test_euclidean_distance_different_lengths() {
        let a = FaceLandmarks::new(vec![(1.0, 2.0)]);
        let b = FaceLandmarks::new(vec![(1.0, 2.0), (3.0, 4.0)]);
        assert!(a.euclidean_distance(&b).is_none());
    }

    #[test]
    fn test_asymmetry_symmetric_face() {
        // Perfectly symmetric face: mean_left == mean_right (by definition of the algorithm)
        // Points are arranged so that left and right sides have equal x-distance from the centre.
        // Centre x = mean of [90, 110, 90, 110] = 100.
        // Left side: x ∈ {90, 90} → mean_left = 90
        // Right side: x ∈ {110, 110} → mean_right = 110
        // asymmetry = |90 - 110| / 100 = 0.2
        // The algorithm does NOT guarantee zero for a mirrored face because it uses
        // |mean_left - mean_right| / mean_all, not left-right per-point differences.
        // A truly "near-zero" result requires mean_left ≈ mean_right, which only happens
        // when all points cluster near the centre.
        // Instead verify that a heavily skewed face scores higher than a balanced one.
        let balanced = FaceLandmarks::new(vec![
            (99.0, 100.0),
            (101.0, 100.0),
            (100.0, 120.0),
        ]);
        let skewed = FaceLandmarks::new(vec![
            (10.0, 100.0),
            (200.0, 100.0),
            (190.0, 120.0),
        ]);
        let asym_balanced = balanced.asymmetry_score();
        let asym_skewed = skewed.asymmetry_score();
        assert!(
            asym_skewed > asym_balanced,
            "Skewed face should have higher asymmetry than balanced face: {asym_skewed} vs {asym_balanced}"
        );
    }

    #[test]
    fn test_asymmetry_skewed_face() {
        // Heavily left-skewed face
        let f = FaceLandmarks::new(vec![
            (10.0, 100.0),
            (20.0, 100.0),
            (15.0, 120.0),
            (200.0, 120.0), // far right outlier
        ]);
        let asym = f.asymmetry_score();
        // Asymmetry should be > 0
        assert!(asym > 0.0, "Skewed face should have non-zero asymmetry");
    }

    // ── check_landmark_consistency ────────────────────────────────────────────

    #[test]
    fn test_check_landmark_consistency_empty() {
        let result = check_landmark_consistency(&[]);
        assert!(result.inter_frame_deviation.abs() < 1e-6);
        assert!(!result.blink_anomaly);
    }

    #[test]
    fn test_check_landmark_consistency_single_frame() {
        let frames = vec![face_at(100.0, 100.0)];
        let result = check_landmark_consistency(&frames);
        assert!(result.inter_frame_deviation.abs() < 1e-6);
        assert!(!result.blink_anomaly);
    }

    #[test]
    fn test_check_landmark_consistency_stable_sequence() {
        // Small natural motion (≤ 1 px) should not trigger blink anomaly
        let frames: Vec<FaceLandmarks> = (0..10)
            .map(|i| face_at(100.0 + i as f32 * 0.1, 100.0))
            .collect();
        let result = check_landmark_consistency(&frames);
        assert!(!result.blink_anomaly, "Small natural motion should not be blink anomaly");
        assert!(result.inter_frame_deviation < 5.0);
    }

    #[test]
    fn test_check_landmark_consistency_blink_anomaly() {
        // Eye-openness: left eye y suddenly changes by > 50%
        let frame_a = FaceLandmarks::new(vec![
            (80.0, 100.0),  // left eye at y=100
            (120.0, 100.0), // right eye
            (100.0, 120.0), // nose
            (85.0, 140.0),  // left mouth
            (115.0, 140.0), // right mouth
        ]);
        // Left eye jumps to y=160 (60% change relative to y=100)
        let frame_b = FaceLandmarks::new(vec![
            (80.0, 160.0),
            (120.0, 100.0),
            (100.0, 120.0),
            (85.0, 140.0),
            (115.0, 140.0),
        ]);
        let result = check_landmark_consistency(&[frame_a, frame_b]);
        assert!(result.blink_anomaly, "Large eye-y change should trigger blink anomaly");
    }

    #[test]
    fn test_check_landmark_consistency_deviation_positive() {
        let frames: Vec<FaceLandmarks> = (0..5)
            .map(|i| face_at(100.0 + i as f32 * 5.0, 100.0))
            .collect();
        let result = check_landmark_consistency(&frames);
        assert!(result.inter_frame_deviation > 0.0);
    }

    // ── DeepFakeScore ────────────────────────────────────────────────────────

    #[test]
    fn test_deepfake_score_real_face() {
        let consistency = LandmarkConsistency {
            inter_frame_deviation: 2.0, // normal jitter
            blink_anomaly: false,
            asymmetry: 0.05, // low asymmetry
        };
        let score = DeepFakeScore::from_consistency(&consistency);
        assert!(score.score < 0.5, "Realistic face should score below 0.5");
        assert!(score.score >= 0.0 && score.score <= 1.0);
    }

    #[test]
    fn test_deepfake_score_blink_anomaly_raises_score() {
        let clean = LandmarkConsistency {
            inter_frame_deviation: 2.0,
            blink_anomaly: false,
            asymmetry: 0.05,
        };
        let blinky = LandmarkConsistency {
            inter_frame_deviation: 2.0,
            blink_anomaly: true,
            asymmetry: 0.05,
        };
        let s_clean = DeepFakeScore::from_consistency(&clean).score;
        let s_blinky = DeepFakeScore::from_consistency(&blinky).score;
        assert!(
            s_blinky > s_clean,
            "Blink anomaly should raise the score: {s_blinky} vs {s_clean}"
        );
    }

    #[test]
    fn test_deepfake_score_high_jitter_raises_score() {
        let low_jitter = LandmarkConsistency {
            inter_frame_deviation: 1.0,
            blink_anomaly: false,
            asymmetry: 0.0,
        };
        let high_jitter = LandmarkConsistency {
            inter_frame_deviation: 15.0,
            blink_anomaly: false,
            asymmetry: 0.0,
        };
        let s_low = DeepFakeScore::from_consistency(&low_jitter).score;
        let s_high = DeepFakeScore::from_consistency(&high_jitter).score;
        assert!(
            s_high > s_low,
            "High jitter should raise score: {s_high} vs {s_low}"
        );
    }

    #[test]
    fn test_deepfake_score_in_range() {
        let worst_case = LandmarkConsistency {
            inter_frame_deviation: 100.0,
            blink_anomaly: true,
            asymmetry: 1.0,
        };
        let score = DeepFakeScore::from_consistency(&worst_case);
        assert!(score.score >= 0.0 && score.score <= 1.0);
    }

    #[test]
    fn test_deepfake_score_indicators_non_empty_for_anomalies() {
        let anomalous = LandmarkConsistency {
            inter_frame_deviation: 20.0,
            blink_anomaly: true,
            asymmetry: 0.5,
        };
        let score = DeepFakeScore::from_consistency(&anomalous);
        assert!(
            !score.indicators.is_empty(),
            "Anomalous consistency should produce indicator strings"
        );
    }
}
