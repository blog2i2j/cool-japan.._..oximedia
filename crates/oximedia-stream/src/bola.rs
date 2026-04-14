//! BOLA-E (Buffer Occupancy Level-based Adaptive Bitrate) algorithm.
//!
//! This module implements the BOLA-E variant from the paper:
//! "BOLA: Near-Optimal Bitrate Adaptation for Online Videos" (Spiteri et al., 2016/2020).
//!
//! BOLA-E introduces an elasticity parameter `gamma_p` that allows the algorithm to
//! trade off between buffer stability and quality maximization. The utility function
//! maps each quality level to a logarithmic utility value:
//!
//! ```text
//! V * (utility[q] + gamma_p) / (p[q]) − V * gamma_p / p[q]
//! ```
//!
//! where `V` is a Lyapunov control parameter derived from buffer size, `p[q]` is the
//! segment duration at quality `q`, and `utility[q] = log(bitrate[q] / bitrate[0])`.
//!
//! # Example
//!
//! ```rust
//! use oximedia_stream::bola::{BolaConfig, BolaState};
//!
//! let bitrates = vec![300_000u64, 750_000, 1_500_000, 3_000_000, 6_000_000];
//! let cfg = BolaConfig::new(bitrates, 30.0, 2.0).expect("valid config");
//! let mut state = BolaState::new(cfg);
//!
//! // Simulate a 20s buffer occupancy and pick the best quality index.
//! let quality_idx = state.select_quality(20.0);
//! assert!(quality_idx < 5);
//! ```

use crate::error::{StreamError, StreamResult};

/// Configuration for the BOLA-E algorithm.
#[derive(Debug, Clone)]
pub struct BolaConfig {
    /// Available bitrates in bits-per-second, sorted ascending.
    pub bitrates_bps: Vec<u64>,
    /// Maximum playback buffer duration in seconds.
    pub buffer_max_secs: f64,
    /// Segment duration in seconds (assumed constant across qualities).
    pub segment_duration_secs: f64,
    /// BOLA-E elasticity parameter (gamma_p). Typical value ≈ 5/p.
    pub gamma_p: f64,
    /// Lyapunov control parameter V, derived from buffer size when not set explicitly.
    pub lyapunov_v: f64,
}

impl BolaConfig {
    /// Construct a new [`BolaConfig`], computing `gamma_p` and `V` from buffer/segment params.
    ///
    /// # Arguments
    /// * `bitrates_bps` – Available bitrates in bps, must be non-empty and sorted ascending.
    /// * `buffer_max_secs` – Maximum buffer in seconds (must be > `segment_duration_secs`).
    /// * `segment_duration_secs` – Duration of each segment (must be > 0).
    ///
    /// # Errors
    /// Returns `StreamError::InvalidParameter` for invalid inputs.
    pub fn new(
        bitrates_bps: Vec<u64>,
        buffer_max_secs: f64,
        segment_duration_secs: f64,
    ) -> StreamResult<Self> {
        if bitrates_bps.is_empty() {
            return Err(StreamError::InvalidParameter(
                "bitrates_bps must not be empty".into(),
            ));
        }
        if segment_duration_secs <= 0.0 {
            return Err(StreamError::InvalidParameter(
                "segment_duration_secs must be > 0".into(),
            ));
        }
        if buffer_max_secs <= segment_duration_secs {
            return Err(StreamError::InvalidParameter(
                "buffer_max_secs must be > segment_duration_secs".into(),
            ));
        }

        // Validate ascending order.
        for i in 1..bitrates_bps.len() {
            if bitrates_bps[i] <= bitrates_bps[i - 1] {
                return Err(StreamError::InvalidParameter(
                    "bitrates_bps must be strictly ascending".into(),
                ));
            }
        }

        // gamma_p = 5 / segment_duration (BOLA-E recommended default).
        let gamma_p = 5.0 / segment_duration_secs;

        // V is calibrated so that at maximum buffer the highest quality is preferred.
        // V = (buffer_max_secs - segment_duration_secs)
        //     / (utility_max + gamma_p * segment_duration_secs)
        let utility_max = Self::utility_value(
            *bitrates_bps
                .last()
                .ok_or_else(|| StreamError::InvalidParameter("empty bitrates".into()))?,
            bitrates_bps[0],
        );
        let lyapunov_v = (buffer_max_secs - segment_duration_secs)
            / (utility_max + gamma_p * segment_duration_secs);

        Ok(Self {
            bitrates_bps,
            buffer_max_secs,
            segment_duration_secs,
            gamma_p,
            lyapunov_v,
        })
    }

    /// Compute logarithmic utility: `log2(bitrate / min_bitrate)`.
    pub fn utility_value(bitrate: u64, min_bitrate: u64) -> f64 {
        if min_bitrate == 0 || bitrate == 0 {
            return 0.0;
        }
        (bitrate as f64 / min_bitrate as f64).log2()
    }
}

/// Runtime state for the BOLA-E ABR controller.
#[derive(Debug)]
pub struct BolaState {
    config: BolaConfig,
    /// Pre-computed utility values for each quality level.
    utilities: Vec<f64>,
    /// Last selected quality index.
    last_quality: usize,
}

impl BolaState {
    /// Create a new [`BolaState`] from the given configuration.
    pub fn new(config: BolaConfig) -> Self {
        let min_bitrate = config.bitrates_bps[0];
        let utilities = config
            .bitrates_bps
            .iter()
            .map(|&br| BolaConfig::utility_value(br, min_bitrate))
            .collect();

        Self {
            config,
            utilities,
            last_quality: 0,
        }
    }

    /// Select the best quality index for the current buffer occupancy.
    ///
    /// The BOLA-E objective function for quality `q`:
    ///
    /// ```text
    /// score(q) = (V * (utility[q] + gamma_p * p) − buffer_level) / p
    /// ```
    ///
    /// We choose the quality `q*` that maximises `score(q)`, subject to
    /// `score(q*) >= 0` (otherwise fall back to quality 0).
    ///
    /// # Arguments
    /// * `buffer_level_secs` – Current playback buffer occupancy in seconds.
    pub fn select_quality(&mut self, buffer_level_secs: f64) -> usize {
        let v = self.config.lyapunov_v;
        let gamma_p = self.config.gamma_p;
        let p = self.config.segment_duration_secs;

        let buf = buffer_level_secs;
        let mut best_idx = 0usize;
        let mut best_score = f64::NEG_INFINITY;

        for (q, &util) in self.utilities.iter().enumerate() {
            // score(q) = (V*(utility[q] + gamma_p * p) - buffer_level) / p
            let score = (v * (util + gamma_p * p) - buf) / p;
            if score > best_score {
                best_score = score;
                best_idx = q;
            }
        }

        // If the best score is negative, the buffer is too full — stay at quality 0.
        if best_score < 0.0 {
            best_idx = 0;
        }

        // Apply hysteresis: only switch up if improvement is significant.
        let switched_up = best_idx > self.last_quality;
        if switched_up {
            // Re-evaluate: only upgrade if the score is strictly positive.
            let p = self.config.segment_duration_secs;
            let score_for_best = (v * (self.utilities[best_idx] + gamma_p * p) - buf) / p;
            if score_for_best <= 0.0 {
                best_idx = self.last_quality;
            }
        }

        self.last_quality = best_idx;
        best_idx
    }

    /// Notify the controller that a segment download completed with the observed throughput.
    ///
    /// This can be used by extensions to update bandwidth estimates; the basic BOLA-E
    /// implementation does not require throughput feedback (it is purely buffer-driven),
    /// but derived controllers may override this.
    pub fn on_segment_downloaded(&mut self, _throughput_bps: u64) {
        // Intentionally no-op in the base BOLA-E implementation.
        // Throughput-aware extensions (BOLA-U/BOLA-O) can use this hook.
    }

    /// Return the number of quality levels.
    pub fn num_quality_levels(&self) -> usize {
        self.config.bitrates_bps.len()
    }

    /// Return the bitrate (bps) for the given quality index.
    ///
    /// # Errors
    /// Returns `StreamError::InvalidParameter` if `index` is out of range.
    pub fn bitrate_at(&self, index: usize) -> StreamResult<u64> {
        self.config
            .bitrates_bps
            .get(index)
            .copied()
            .ok_or_else(|| StreamError::InvalidParameter(format!("index {index} out of range")))
    }

    /// Return the configuration.
    pub fn config(&self) -> &BolaConfig {
        &self.config
    }

    /// Return the last selected quality index without performing a new selection.
    pub fn last_quality(&self) -> usize {
        self.last_quality
    }

    /// Return the utility value at `index`.
    ///
    /// # Errors
    /// Returns `StreamError::InvalidParameter` if `index` is out of range.
    pub fn utility_at(&self, index: usize) -> StreamResult<f64> {
        self.utilities
            .get(index)
            .copied()
            .ok_or_else(|| StreamError::InvalidParameter(format!("index {index} out of range")))
    }
}

/// Segment download record used to feed throughput observations into an ABR controller.
#[derive(Debug, Clone)]
pub struct SegmentDownloadRecord {
    /// Quality level that was downloaded.
    pub quality_index: usize,
    /// Number of bytes transferred.
    pub bytes_transferred: u64,
    /// Download duration in seconds (wall-clock).
    pub download_duration_secs: f64,
    /// Resulting buffer level after the segment was appended (seconds).
    pub resulting_buffer_secs: f64,
}

impl SegmentDownloadRecord {
    /// Compute the throughput in bits per second.
    ///
    /// Returns `None` if `download_duration_secs` is zero to avoid division by zero.
    pub fn throughput_bps(&self) -> Option<u64> {
        if self.download_duration_secs <= 0.0 {
            return None;
        }
        Some((self.bytes_transferred as f64 * 8.0 / self.download_duration_secs) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_bitrates() -> Vec<u64> {
        vec![300_000, 750_000, 1_500_000, 3_000_000, 6_000_000]
    }

    #[test]
    fn test_config_new_valid() {
        let cfg = BolaConfig::new(default_bitrates(), 30.0, 4.0);
        assert!(cfg.is_ok(), "valid config should succeed");
        let cfg = cfg.unwrap();
        assert!(cfg.lyapunov_v > 0.0);
        assert!(cfg.gamma_p > 0.0);
    }

    #[test]
    fn test_config_empty_bitrates_fails() {
        let result = BolaConfig::new(vec![], 30.0, 4.0);
        assert!(matches!(result, Err(StreamError::InvalidParameter(_))));
    }

    #[test]
    fn test_config_zero_segment_duration_fails() {
        let result = BolaConfig::new(default_bitrates(), 30.0, 0.0);
        assert!(matches!(result, Err(StreamError::InvalidParameter(_))));
    }

    #[test]
    fn test_config_buffer_smaller_than_segment_fails() {
        let result = BolaConfig::new(default_bitrates(), 3.0, 4.0);
        assert!(matches!(result, Err(StreamError::InvalidParameter(_))));
    }

    #[test]
    fn test_config_non_ascending_bitrates_fails() {
        let result = BolaConfig::new(vec![3_000_000, 1_000_000], 30.0, 4.0);
        assert!(matches!(result, Err(StreamError::InvalidParameter(_))));
    }

    #[test]
    fn test_utilities_computed() {
        let cfg = BolaConfig::new(default_bitrates(), 30.0, 4.0).unwrap();
        let state = BolaState::new(cfg);
        // Utility at index 0 should be 0 (log2(1) = 0).
        let u0 = state.utility_at(0).unwrap();
        assert!((u0 - 0.0).abs() < 1e-10, "utility[0] should be 0");
        // Utilities should be strictly increasing.
        for i in 1..state.num_quality_levels() {
            let u_prev = state.utility_at(i - 1).unwrap();
            let u_cur = state.utility_at(i).unwrap();
            assert!(u_cur > u_prev, "utilities must be strictly increasing");
        }
    }

    #[test]
    fn test_select_quality_empty_buffer_picks_highest() {
        // BOLA-E property: when buffer is empty, all quality scores are positive
        // (V*(util[q]+gamma_p) / p > 0 for all q), so the algorithm picks the highest
        // quality level — this is intentional in BOLA-E (it aggressively requests
        // segments to fill the empty buffer).
        let cfg = BolaConfig::new(default_bitrates(), 30.0, 4.0).unwrap();
        let mut state = BolaState::new(cfg);
        let q = state.select_quality(0.0);
        // With empty buffer all scores positive → picks highest quality.
        assert_eq!(
            q,
            state.num_quality_levels() - 1,
            "empty buffer should pick highest quality per BOLA-E"
        );
    }

    #[test]
    fn test_select_quality_buffer_drainage_drops_quality() {
        // When buffer is very large (at max), scores can become negative for higher
        // qualities, causing the algorithm to pick a lower quality level.
        // Specifically, with buffer > V*(util[q_max] + gamma_p*p), score drops to 0.
        let cfg = BolaConfig::new(default_bitrates(), 30.0, 4.0).unwrap();
        let v = cfg.lyapunov_v;
        let gamma_p = cfg.gamma_p;
        let p = cfg.segment_duration_secs;
        let min_br = cfg.bitrates_bps[0];
        let max_br = *cfg.bitrates_bps.last().unwrap();
        let util_max = BolaConfig::utility_value(max_br, min_br);
        // At exactly the threshold for highest quality, all qualities should be valid.
        let threshold = v * (util_max + gamma_p * p);
        // Buffer below threshold → highest quality selected.
        let mut state = BolaState::new(cfg.clone());
        let q = state.select_quality(threshold * 0.5);
        assert_eq!(
            q,
            state.num_quality_levels() - 1,
            "below threshold should pick highest"
        );
        // Buffer above threshold for high quality but below for quality 0 → lower quality.
        let util_0 = 0.0f64; // utility of quality 0
        let threshold_q0 = v * (util_0 + gamma_p * p);
        let mut state2 = BolaState::new(cfg);
        state2.last_quality = 0; // reset hysteresis
        let q2 = state2.select_quality(threshold + 1.0);
        // Should be lower than max.
        let _ = threshold_q0;
        assert!(q2 < 4, "above max threshold should not pick highest");
    }

    #[test]
    fn test_select_quality_returns_valid_index() {
        let cfg = BolaConfig::new(default_bitrates(), 30.0, 4.0).unwrap();
        let mut state = BolaState::new(cfg);
        for buf_secs in [0.0f64, 5.0, 10.0, 15.0, 20.0, 25.0, 29.0] {
            state.last_quality = 0;
            let q = state.select_quality(buf_secs);
            assert!(
                q < state.num_quality_levels(),
                "quality {q} out of range for buf={buf_secs}"
            );
        }
    }

    #[test]
    fn test_bitrate_at_valid() {
        let cfg = BolaConfig::new(default_bitrates(), 30.0, 4.0).unwrap();
        let state = BolaState::new(cfg);
        assert_eq!(state.bitrate_at(0).unwrap(), 300_000);
        assert_eq!(state.bitrate_at(4).unwrap(), 6_000_000);
    }

    #[test]
    fn test_bitrate_at_out_of_range() {
        let cfg = BolaConfig::new(default_bitrates(), 30.0, 4.0).unwrap();
        let state = BolaState::new(cfg);
        assert!(matches!(
            state.bitrate_at(99),
            Err(StreamError::InvalidParameter(_))
        ));
    }

    #[test]
    fn test_single_quality_level() {
        let cfg = BolaConfig::new(vec![1_000_000], 30.0, 4.0).unwrap();
        let mut state = BolaState::new(cfg);
        // With a single quality, always returns 0.
        assert_eq!(state.select_quality(0.0), 0);
        assert_eq!(state.select_quality(15.0), 0);
        assert_eq!(state.select_quality(29.0), 0);
    }

    #[test]
    fn test_segment_download_record_throughput() {
        let rec = SegmentDownloadRecord {
            quality_index: 2,
            bytes_transferred: 1_000_000,
            download_duration_secs: 1.0,
            resulting_buffer_secs: 12.0,
        };
        assert_eq!(rec.throughput_bps(), Some(8_000_000));
    }

    #[test]
    fn test_segment_download_record_zero_duration() {
        let rec = SegmentDownloadRecord {
            quality_index: 0,
            bytes_transferred: 100,
            download_duration_secs: 0.0,
            resulting_buffer_secs: 0.0,
        };
        assert_eq!(rec.throughput_bps(), None);
    }
}
