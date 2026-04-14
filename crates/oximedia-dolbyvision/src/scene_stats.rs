//! Scene-level PQ (Perceptual Quantizer) statistics for Dolby Vision.
//!
//! [`DvSceneStats`] accumulates per-frame minimum, mid-tone (median
//! approximation), and maximum PQ code values across all frames that belong to
//! a single scene.  The summary statistics are used to drive shot-level L1
//! metadata and scene-boundary–triggered trim decisions.
//!
//! # PQ Code Range
//!
//! PQ codes are 12-bit values in the range 0–4095 as used by the Dolby Vision
//! L1 metadata block (`min_pq`, `mid_pq`, `max_pq`).

/// Accumulates per-frame PQ statistics for a Dolby Vision scene.
///
/// Call [`DvSceneStats::add_frame`] once per frame, then retrieve the
/// scene-level summary with [`DvSceneStats::scene_summary`].
#[derive(Debug, Clone)]
pub struct DvSceneStats {
    /// Minimum PQ value across all frames (scene shadow level).
    scene_min: u16,
    /// Running sum of per-frame `mid_pq` values (for average computation).
    mid_sum: u64,
    /// Maximum PQ value across all frames (scene peak level).
    scene_max: u16,
    /// Number of frames accumulated.
    frame_count: u64,
}

impl DvSceneStats {
    /// Create a new, empty scene statistics collector.
    ///
    /// Before any frames are added, the min is initialised to `u16::MAX` and
    /// the max to `0` so the first frame sets correct bounds.
    #[must_use]
    pub fn new() -> Self {
        Self {
            scene_min: u16::MAX,
            mid_sum: 0,
            scene_max: 0,
            frame_count: 0,
        }
    }

    /// Accumulate statistics for one frame.
    ///
    /// # Arguments
    ///
    /// * `min_pq` — shadow level of this frame (12-bit PQ, 0–4095).
    /// * `mid_pq` — mid-tone level of this frame (12-bit PQ, 0–4095).
    /// * `max_pq` — peak level of this frame (12-bit PQ, 0–4095).
    pub fn add_frame(&mut self, min_pq: u16, mid_pq: u16, max_pq: u16) {
        // Clamp to valid 12-bit PQ range.
        let min_pq = min_pq.min(4095);
        let mid_pq = mid_pq.min(4095);
        let max_pq = max_pq.min(4095);

        if min_pq < self.scene_min {
            self.scene_min = min_pq;
        }
        if max_pq > self.scene_max {
            self.scene_max = max_pq;
        }
        self.mid_sum = self.mid_sum.saturating_add(u64::from(mid_pq));
        self.frame_count += 1;
    }

    /// Return scene-level PQ summary: `(scene_min, scene_avg_mid, scene_max)`.
    ///
    /// - `scene_min` — minimum PQ value seen across all frames.
    /// - `scene_avg_mid` — arithmetic mean of per-frame `mid_pq` values,
    ///   rounded to the nearest integer.
    /// - `scene_max` — maximum PQ value seen across all frames.
    ///
    /// Returns `(0, 0, 0)` when no frames have been added.
    #[must_use]
    pub fn scene_summary(&self) -> (u16, u16, u16) {
        if self.frame_count == 0 {
            return (0, 0, 0);
        }
        let avg_mid =
            ((self.mid_sum + self.frame_count / 2) / self.frame_count).min(4095) as u16;
        let scene_min = if self.scene_min == u16::MAX {
            0
        } else {
            self.scene_min
        };
        (scene_min, avg_mid, self.scene_max)
    }

    /// Number of frames accumulated in this scene.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Reset all accumulated statistics, ready for the next scene.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for DvSceneStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_summary() {
        let stats = DvSceneStats::new();
        assert_eq!(stats.scene_summary(), (0, 0, 0));
        assert_eq!(stats.frame_count(), 0);
    }

    #[test]
    fn test_single_frame() {
        let mut stats = DvSceneStats::new();
        stats.add_frame(100, 500, 2000);
        let (min, avg, max) = stats.scene_summary();
        assert_eq!(min, 100);
        assert_eq!(avg, 500);
        assert_eq!(max, 2000);
    }

    #[test]
    fn test_multiple_frames_min_max() {
        let mut stats = DvSceneStats::new();
        stats.add_frame(200, 600, 3000);
        stats.add_frame(50, 700, 4000);
        stats.add_frame(300, 400, 2500);
        let (min, avg, max) = stats.scene_summary();
        assert_eq!(min, 50, "scene min should be 50");
        assert_eq!(max, 4000, "scene max should be 4000");
        // avg of 600 + 700 + 400 = 1700 / 3 ≈ 567
        assert_eq!(avg, 567, "avg mid should be 567");
    }

    #[test]
    fn test_pq_clamp() {
        let mut stats = DvSceneStats::new();
        // Values exceeding 12-bit range should be clamped.
        stats.add_frame(0, 0, 5000);
        let (_, _, max) = stats.scene_summary();
        assert_eq!(max, 4095, "max should be clamped to 4095");
    }

    #[test]
    fn test_reset() {
        let mut stats = DvSceneStats::new();
        stats.add_frame(100, 500, 2000);
        assert_eq!(stats.frame_count(), 1);
        stats.reset();
        assert_eq!(stats.frame_count(), 0);
        assert_eq!(stats.scene_summary(), (0, 0, 0));
    }

    #[test]
    fn test_default() {
        let stats = DvSceneStats::default();
        assert_eq!(stats.frame_count(), 0);
    }
}
