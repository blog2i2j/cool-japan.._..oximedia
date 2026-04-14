//! Camera tracking data filtering and smoothing for virtual production.
//!
//! Provides four filtering modes for 6-DOF camera tracking streams:
//!
//! - **PassThrough** – no processing; useful for latency testing.
//! - **LowPass** – first-order IIR (exponential moving average), controlled
//!   by a cutoff frequency and sample rate.
//! - **Kalman** – per-component 1-D scalar Kalman filter (process + measurement
//!   noise parameters) applied independently to each of the seven components
//!   (3 position + 4 quaternion).  The quaternion is re-normalised after each
//!   update.
//! - **OneEuro** – adaptive low-pass filter that reduces jitter at slow motion
//!   while keeping latency low at fast motion.  Based on the original paper:
//!   *Casiez et al., "1€ Filter: A Simple Speed-Based Low-Pass Filter for
//!   Noisy Input in Interactive Systems", CHI 2012.*
//!
//! # Example
//! ```rust
//! use oximedia_virtual::tracking_filter::{TrackingFilter, TrackingFrame, FilterMode};
//!
//! let mut filter = TrackingFilter::new(FilterMode::PassThrough);
//! let frame = TrackingFrame {
//!     frame: 0,
//!     position: [1.0, 2.0, 3.0],
//!     rotation_quat: [0.0, 0.0, 0.0, 1.0],
//!     confidence: 1.0,
//! };
//! let out = filter.filter(&frame);
//! assert_eq!(out.position, frame.position);
//! ```

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single sample of 6-DOF camera tracking data.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TrackingFrame {
    /// Monotonically increasing frame counter.
    pub frame: u64,
    /// World-space position in metres `[x, y, z]`.
    pub position: [f32; 3],
    /// Orientation as a unit quaternion `[x, y, z, w]`.
    pub rotation_quat: [f32; 4],
    /// Tracking confidence in `[0, 1]`; 1.0 = full confidence.
    pub confidence: f32,
}

/// Selects the algorithm used by [`TrackingFilter`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum FilterMode {
    /// Data passes through without modification.
    PassThrough,

    /// First-order IIR low-pass filter.
    ///
    /// The smoothing coefficient α is derived from `cutoff_hz` and
    /// `sample_rate_hz`:
    /// ```text
    /// tau = 1 / (2π · cutoff_hz)
    /// dt  = 1 / sample_rate_hz
    /// α   = dt / (tau + dt)     (standard first-order IIR)
    /// ```
    LowPass {
        /// Low-pass cutoff frequency in Hz.
        cutoff_hz: f32,
        /// Expected input sample rate in Hz.
        sample_rate_hz: f32,
    },

    /// Independent scalar Kalman filter applied to each component.
    ///
    /// Each component is modelled as a constant-position state with
    /// Gaussian process and measurement noise.
    Kalman {
        /// Process noise variance Q (motion model uncertainty).
        process_noise: f32,
        /// Measurement noise variance R (sensor uncertainty).
        measurement_noise: f32,
    },

    /// One-Euro adaptive low-pass filter.
    ///
    /// Adjusts the effective cutoff frequency based on the speed of the
    /// signal: slow-moving signals use a low cutoff (heavy smoothing) while
    /// fast-moving signals use a high cutoff (low latency).
    OneEuro {
        /// Minimum cutoff frequency in Hz (applied when signal is stationary).
        min_cutoff: f32,
        /// Speed coefficient β: higher values reduce latency for fast motion.
        beta: f32,
    },
}

/// Jitter report produced by [`JitterMetrics::compute`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JitterReport {
    /// Root-mean-square frame-to-frame position displacement (metres).
    pub rms_position_jitter: f32,
    /// Maximum frame-to-frame position displacement (metres).
    pub max_position_jitter: f32,
    /// Root-mean-square frame-to-frame rotation quaternion angular distance
    /// (in radians, using the great-circle distance on the unit 4-sphere).
    pub rms_rotation_jitter: f32,
}

// ---------------------------------------------------------------------------
// Internal filter state
// ---------------------------------------------------------------------------

/// 1-D scalar Kalman filter state.
#[derive(Debug, Clone)]
struct ScalarKalman {
    /// Current state estimate.
    x: f32,
    /// Estimate covariance.
    p: f32,
    /// Process noise Q.
    q: f32,
    /// Measurement noise R.
    r: f32,
    /// Whether the filter has been initialised.
    initialised: bool,
}

impl ScalarKalman {
    fn new(q: f32, r: f32) -> Self {
        Self {
            x: 0.0,
            p: 1.0,
            q,
            r,
            initialised: false,
        }
    }

    /// Update with a new measurement and return the filtered estimate.
    fn update(&mut self, measurement: f32) -> f32 {
        if !self.initialised {
            self.x = measurement;
            self.p = 1.0;
            self.initialised = true;
            return self.x;
        }

        // Predict
        let p_pred = self.p + self.q;

        // Kalman gain
        let k = p_pred / (p_pred + self.r);

        // Update
        self.x += k * (measurement - self.x);
        self.p = (1.0 - k) * p_pred;

        self.x
    }
}

// ---------------------------------------------------------------------------
// One-Euro internal state for a single component
// ---------------------------------------------------------------------------

/// One-Euro filter state for a single scalar component.
#[derive(Debug, Clone)]
struct OneEuroScalar {
    /// Minimum cutoff frequency (Hz).
    min_cutoff: f32,
    /// Speed coefficient β.
    beta: f32,
    /// Derivative cutoff frequency (Hz); fixed at 1 Hz.
    d_cutoff: f32,
    /// Previous filtered value.
    x_prev: Option<f32>,
    /// Previous filtered derivative.
    dx_prev: f32,
}

impl OneEuroScalar {
    fn new(min_cutoff: f32, beta: f32) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff: 1.0,
            x_prev: None,
            dx_prev: 0.0,
        }
    }

    /// Compute the IIR smoothing coefficient α for a given cutoff and sample rate.
    #[inline]
    fn alpha(cutoff_hz: f32, sample_rate_hz: f32) -> f32 {
        if sample_rate_hz <= 0.0 || cutoff_hz <= 0.0 {
            return 1.0;
        }
        let tau = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
        let dt = 1.0 / sample_rate_hz;
        dt / (tau + dt)
    }

    /// Filter a new raw measurement at the given sample rate.
    fn filter(&mut self, raw: f32, sample_rate_hz: f32) -> f32 {
        let x_prev = match self.x_prev {
            Some(v) => v,
            None => {
                self.x_prev = Some(raw);
                return raw;
            }
        };

        // Estimate derivative.
        let dx = (raw - x_prev) * sample_rate_hz;

        // Filter the derivative.
        let a_d = Self::alpha(self.d_cutoff, sample_rate_hz);
        let dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev;
        self.dx_prev = dx_hat;

        // Adapt cutoff to signal speed.
        let cutoff = self.min_cutoff + self.beta * dx_hat.abs();

        // Filter the signal.
        let a = Self::alpha(cutoff, sample_rate_hz);
        let x_hat = a * raw + (1.0 - a) * x_prev;
        self.x_prev = Some(x_hat);

        x_hat
    }
}

// ---------------------------------------------------------------------------
// IIR low-pass state for a single component
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct IirScalar {
    prev: Option<f32>,
    alpha: f32,
}

impl IirScalar {
    fn new(cutoff_hz: f32, sample_rate_hz: f32) -> Self {
        let alpha = if sample_rate_hz > 0.0 && cutoff_hz > 0.0 {
            let tau = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
            let dt = 1.0 / sample_rate_hz;
            dt / (tau + dt)
        } else {
            1.0
        };
        Self { prev: None, alpha }
    }

    fn filter(&mut self, raw: f32) -> f32 {
        let out = match self.prev {
            None => raw,
            Some(p) => self.alpha * raw + (1.0 - self.alpha) * p,
        };
        self.prev = Some(out);
        out
    }
}

// ---------------------------------------------------------------------------
// Normalise a quaternion (fallback to identity if degenerate)
// ---------------------------------------------------------------------------

fn quat_normalise(q: [f32; 4]) -> [f32; 4] {
    let len_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if len_sq < 1e-12 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let inv = 1.0 / len_sq.sqrt();
    [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
}

// ---------------------------------------------------------------------------
// Internal per-filter state union
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum FilterState {
    PassThrough,
    LowPass {
        pos: [IirScalar; 3],
        rot: [IirScalar; 4],
    },
    Kalman {
        pos: [ScalarKalman; 3],
        rot: [ScalarKalman; 4],
    },
    OneEuro {
        pos: [OneEuroScalar; 3],
        rot: [OneEuroScalar; 4],
        sample_rate_hz: f32,
    },
}

// ---------------------------------------------------------------------------
// Public TrackingFilter
// ---------------------------------------------------------------------------

/// Stateful camera tracking filter.
///
/// Create one instance per tracking stream and call [`TrackingFilter::filter`]
/// for every incoming [`TrackingFrame`] in order.
pub struct TrackingFilter {
    mode: FilterMode,
    state: FilterState,
}

impl TrackingFilter {
    /// Construct a new filter with the specified mode.
    #[must_use]
    pub fn new(mode: FilterMode) -> Self {
        let state = Self::build_state(&mode);
        Self { mode, state }
    }

    /// Reset internal filter state (e.g. after a cut or tracking loss).
    pub fn reset(&mut self) {
        self.state = Self::build_state(&self.mode);
    }

    /// Apply the filter to one incoming frame and return the smoothed result.
    ///
    /// Frames must be supplied in chronological order; calling this method
    /// out-of-order produces undefined smoothing behaviour.
    pub fn filter(&mut self, frame: &TrackingFrame) -> TrackingFrame {
        match &mut self.state {
            FilterState::PassThrough => frame.clone(),

            FilterState::LowPass { pos, rot } => {
                let filtered_pos = [
                    pos[0].filter(frame.position[0]),
                    pos[1].filter(frame.position[1]),
                    pos[2].filter(frame.position[2]),
                ];
                let q_raw = frame.rotation_quat;
                let filtered_rot_raw = [
                    rot[0].filter(q_raw[0]),
                    rot[1].filter(q_raw[1]),
                    rot[2].filter(q_raw[2]),
                    rot[3].filter(q_raw[3]),
                ];
                TrackingFrame {
                    frame: frame.frame,
                    position: filtered_pos,
                    rotation_quat: quat_normalise(filtered_rot_raw),
                    confidence: frame.confidence,
                }
            }

            FilterState::Kalman { pos, rot } => {
                let filtered_pos = [
                    pos[0].update(frame.position[0]),
                    pos[1].update(frame.position[1]),
                    pos[2].update(frame.position[2]),
                ];
                let q_raw = frame.rotation_quat;
                let filtered_rot_raw = [
                    rot[0].update(q_raw[0]),
                    rot[1].update(q_raw[1]),
                    rot[2].update(q_raw[2]),
                    rot[3].update(q_raw[3]),
                ];
                TrackingFrame {
                    frame: frame.frame,
                    position: filtered_pos,
                    rotation_quat: quat_normalise(filtered_rot_raw),
                    confidence: frame.confidence,
                }
            }

            FilterState::OneEuro {
                pos,
                rot,
                sample_rate_hz,
            } => {
                let sr = *sample_rate_hz;
                let filtered_pos = [
                    pos[0].filter(frame.position[0], sr),
                    pos[1].filter(frame.position[1], sr),
                    pos[2].filter(frame.position[2], sr),
                ];
                let q_raw = frame.rotation_quat;
                let filtered_rot_raw = [
                    rot[0].filter(q_raw[0], sr),
                    rot[1].filter(q_raw[1], sr),
                    rot[2].filter(q_raw[2], sr),
                    rot[3].filter(q_raw[3], sr),
                ];
                TrackingFrame {
                    frame: frame.frame,
                    position: filtered_pos,
                    rotation_quat: quat_normalise(filtered_rot_raw),
                    confidence: frame.confidence,
                }
            }
        }
    }

    // -----------------------------------------------------------------------

    fn build_state(mode: &FilterMode) -> FilterState {
        match mode {
            FilterMode::PassThrough => FilterState::PassThrough,

            FilterMode::LowPass {
                cutoff_hz,
                sample_rate_hz,
            } => FilterState::LowPass {
                pos: std::array::from_fn(|_| IirScalar::new(*cutoff_hz, *sample_rate_hz)),
                rot: std::array::from_fn(|_| IirScalar::new(*cutoff_hz, *sample_rate_hz)),
            },

            FilterMode::Kalman {
                process_noise,
                measurement_noise,
            } => FilterState::Kalman {
                pos: std::array::from_fn(|_| ScalarKalman::new(*process_noise, *measurement_noise)),
                rot: std::array::from_fn(|_| ScalarKalman::new(*process_noise, *measurement_noise)),
            },

            FilterMode::OneEuro { min_cutoff, beta } => {
                // Default sample rate of 60 Hz; updated dynamically if needed.
                // For the stateless construction we use 60 Hz as a placeholder.
                FilterState::OneEuro {
                    pos: std::array::from_fn(|_| OneEuroScalar::new(*min_cutoff, *beta)),
                    rot: std::array::from_fn(|_| OneEuroScalar::new(*min_cutoff, *beta)),
                    sample_rate_hz: 60.0,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// JitterMetrics
// ---------------------------------------------------------------------------

/// Compute jitter statistics from a sequence of tracking frames.
pub struct JitterMetrics;

impl JitterMetrics {
    /// Compute position and rotation jitter metrics from a slice of frames.
    ///
    /// Jitter is defined as the frame-to-frame displacement (position) or
    /// angular distance (rotation quaternion). Returns a zero-valued
    /// [`JitterReport`] if fewer than two frames are supplied.
    #[must_use]
    pub fn compute(frames: &[TrackingFrame]) -> JitterReport {
        if frames.len() < 2 {
            return JitterReport {
                rms_position_jitter: 0.0,
                max_position_jitter: 0.0,
                rms_rotation_jitter: 0.0,
            };
        }

        let n = (frames.len() - 1) as f32;
        let mut sum_pos_sq = 0.0_f32;
        let mut max_pos = 0.0_f32;
        let mut sum_rot_sq = 0.0_f32;

        for pair in frames.windows(2) {
            let a = &pair[0];
            let b = &pair[1];

            // Position displacement.
            let dp = [
                b.position[0] - a.position[0],
                b.position[1] - a.position[1],
                b.position[2] - a.position[2],
            ];
            let pos_dist = (dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2]).sqrt();
            sum_pos_sq += pos_dist * pos_dist;
            max_pos = max_pos.max(pos_dist);

            // Rotation angular distance: 2 * acos(|a·b|) on unit 4-sphere.
            let dot = (a.rotation_quat[0] * b.rotation_quat[0]
                + a.rotation_quat[1] * b.rotation_quat[1]
                + a.rotation_quat[2] * b.rotation_quat[2]
                + a.rotation_quat[3] * b.rotation_quat[3])
                .abs()
                .clamp(0.0, 1.0);
            let rot_dist = 2.0 * dot.acos();
            sum_rot_sq += rot_dist * rot_dist;
        }

        JitterReport {
            rms_position_jitter: (sum_pos_sq / n).sqrt(),
            max_position_jitter: max_pos,
            rms_rotation_jitter: (sum_rot_sq / n).sqrt(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_frame(frame: u64) -> TrackingFrame {
        TrackingFrame {
            frame,
            position: [0.0, 0.0, 0.0],
            rotation_quat: [0.0, 0.0, 0.0, 1.0],
            confidence: 1.0,
        }
    }

    // -----------------------------------------------------------------------
    // PassThrough
    // -----------------------------------------------------------------------

    #[test]
    fn test_passthrough_unchanged_position() {
        let mut f = TrackingFilter::new(FilterMode::PassThrough);
        let frame = TrackingFrame {
            frame: 0,
            position: [1.0, 2.0, 3.0],
            rotation_quat: [0.0, 0.0, 0.0, 1.0],
            confidence: 0.9,
        };
        let out = f.filter(&frame);
        assert_eq!(out.position, frame.position);
        assert_eq!(out.rotation_quat, frame.rotation_quat);
        assert_eq!(out.confidence, frame.confidence);
    }

    #[test]
    fn test_passthrough_unchanged_rotation() {
        let mut f = TrackingFilter::new(FilterMode::PassThrough);
        let frame = TrackingFrame {
            frame: 7,
            position: [0.0; 3],
            rotation_quat: [0.707, 0.0, 0.707, 0.0],
            confidence: 1.0,
        };
        let out = f.filter(&frame);
        // rotation_quat passed through as-is (no normalisation in passthrough)
        assert!((out.rotation_quat[0] - 0.707).abs() < 1e-5);
        assert!((out.rotation_quat[2] - 0.707).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // Low-pass: first frame returned as-is, subsequent frames smoothed
    // -----------------------------------------------------------------------

    #[test]
    fn test_lowpass_first_frame_passthrough() {
        let mut f = TrackingFilter::new(FilterMode::LowPass {
            cutoff_hz: 5.0,
            sample_rate_hz: 60.0,
        });
        let frame = TrackingFrame {
            frame: 0,
            position: [3.0, -1.0, 2.5],
            rotation_quat: [0.0, 0.0, 0.0, 1.0],
            confidence: 1.0,
        };
        let out = f.filter(&frame);
        // First output must equal input exactly.
        assert_eq!(out.position, frame.position);
    }

    #[test]
    fn test_lowpass_smoothes_step_change() {
        let mut f = TrackingFilter::new(FilterMode::LowPass {
            cutoff_hz: 2.0,
            sample_rate_hz: 60.0,
        });

        // Warm up at zero.
        for i in 0..30_u64 {
            f.filter(&identity_frame(i));
        }

        // Step to 10 m.
        let step_frame = TrackingFrame {
            frame: 30,
            position: [10.0, 0.0, 0.0],
            rotation_quat: [0.0, 0.0, 0.0, 1.0],
            confidence: 1.0,
        };
        let out = f.filter(&step_frame);

        // Filtered output must be strictly between 0 and 10 (it's smoothed).
        assert!(
            out.position[0] > 0.0 && out.position[0] < 10.0,
            "smoothed position should be between 0 and 10, got {}",
            out.position[0]
        );
    }

    // -----------------------------------------------------------------------
    // Kalman: reduces jitter
    // -----------------------------------------------------------------------

    #[test]
    fn test_kalman_converges_to_stationary_position() {
        let mut f = TrackingFilter::new(FilterMode::Kalman {
            process_noise: 1e-3,
            measurement_noise: 0.1,
        });

        // Feed 60 frames with small white noise around (5, 0, 0).
        // We use a deterministic pseudo-noise sequence.
        let mut pos_x_sum = 0.0_f32;
        let noises = [0.05, -0.03, 0.07, -0.06, 0.02, -0.04, 0.08, -0.01];
        for i in 0..60_u64 {
            let noise = noises[(i as usize) % noises.len()];
            let frame = TrackingFrame {
                frame: i,
                position: [5.0 + noise, 0.0, 0.0],
                rotation_quat: [0.0, 0.0, 0.0, 1.0],
                confidence: 1.0,
            };
            let out = f.filter(&frame);
            if i >= 30 {
                pos_x_sum += out.position[0];
            }
        }

        let avg = pos_x_sum / 30.0;
        assert!(
            (avg - 5.0).abs() < 0.15,
            "Kalman average should be close to 5.0, got {avg}"
        );
    }

    #[test]
    fn test_kalman_reduces_rms_jitter() {
        // Generate noisy frames and compare jitter before/after Kalman.
        let noisy_frames: Vec<TrackingFrame> = (0..120_u64)
            .map(|i| {
                let noise = if i % 2 == 0 { 0.1 } else { -0.1 };
                TrackingFrame {
                    frame: i,
                    position: [noise, 0.0, 0.0],
                    rotation_quat: [0.0, 0.0, 0.0, 1.0],
                    confidence: 1.0,
                }
            })
            .collect();

        let raw_jitter = JitterMetrics::compute(&noisy_frames).rms_position_jitter;

        let mut f = TrackingFilter::new(FilterMode::Kalman {
            process_noise: 1e-4,
            measurement_noise: 0.05,
        });
        let filtered_frames: Vec<TrackingFrame> =
            noisy_frames.iter().map(|fr| f.filter(fr)).collect();
        let filtered_jitter = JitterMetrics::compute(&filtered_frames).rms_position_jitter;

        assert!(
            filtered_jitter < raw_jitter,
            "Kalman should reduce jitter: {filtered_jitter} < {raw_jitter}"
        );
    }

    // -----------------------------------------------------------------------
    // One-Euro: reacts to fast motion with less lag than pure low-pass
    // -----------------------------------------------------------------------

    #[test]
    fn test_one_euro_first_frame_passthrough() {
        let mut f = TrackingFilter::new(FilterMode::OneEuro {
            min_cutoff: 1.0,
            beta: 0.007,
        });
        let frame = TrackingFrame {
            frame: 0,
            position: [7.0, -2.0, 1.0],
            rotation_quat: [0.0, 0.0, 0.0, 1.0],
            confidence: 1.0,
        };
        let out = f.filter(&frame);
        assert_eq!(out.position, frame.position, "first frame must pass through");
    }

    #[test]
    fn test_one_euro_smoothes_slow_jitter() {
        let mut f = TrackingFilter::new(FilterMode::OneEuro {
            min_cutoff: 1.0,
            beta: 0.0,
        });

        // Warm up with 30 identity frames.
        for i in 0..30_u64 {
            f.filter(&identity_frame(i));
        }

        // Apply a small jitter and check output is between original and zero.
        let jitter = TrackingFrame {
            frame: 30,
            position: [0.5, 0.0, 0.0],
            rotation_quat: [0.0, 0.0, 0.0, 1.0],
            confidence: 1.0,
        };
        let out = f.filter(&jitter);
        assert!(
            out.position[0] < 0.5,
            "One-Euro should smooth small jitter: {}",
            out.position[0]
        );
    }

    #[test]
    fn test_one_euro_fast_motion_lower_lag_than_lowpass() {
        // For a large fast step, One-Euro (high beta) should track faster
        // than an equivalent low-pass filter.
        let mut lp = TrackingFilter::new(FilterMode::LowPass {
            cutoff_hz: 2.0,
            sample_rate_hz: 60.0,
        });
        let mut oe = TrackingFilter::new(FilterMode::OneEuro {
            min_cutoff: 2.0,
            beta: 1.0, // aggressive speed adaptation
        });

        // Warm up both at zero.
        for i in 0..30_u64 {
            lp.filter(&identity_frame(i));
            oe.filter(&identity_frame(i));
        }

        // Large fast step.
        let step = TrackingFrame {
            frame: 30,
            position: [100.0, 0.0, 0.0],
            rotation_quat: [0.0, 0.0, 0.0, 1.0],
            confidence: 1.0,
        };
        let lp_out = lp.filter(&step);
        let oe_out = oe.filter(&step);

        // One-Euro with high β should react faster (higher output for large step).
        assert!(
            oe_out.position[0] >= lp_out.position[0],
            "One-Euro should track fast motion better: oe={} lp={}",
            oe_out.position[0],
            lp_out.position[0]
        );
    }

    // -----------------------------------------------------------------------
    // Jitter metrics
    // -----------------------------------------------------------------------

    #[test]
    fn test_jitter_metrics_empty_or_single() {
        let report = JitterMetrics::compute(&[]);
        assert_eq!(report.rms_position_jitter, 0.0);
        assert_eq!(report.max_position_jitter, 0.0);
        assert_eq!(report.rms_rotation_jitter, 0.0);

        let single = [identity_frame(0)];
        let r2 = JitterMetrics::compute(&single);
        assert_eq!(r2.rms_position_jitter, 0.0);
    }

    #[test]
    fn test_jitter_metrics_static_signal_zero_jitter() {
        let frames: Vec<TrackingFrame> = (0..10)
            .map(|i| TrackingFrame {
                frame: i,
                position: [1.0, 2.0, 3.0],
                rotation_quat: [0.0, 0.0, 0.0, 1.0],
                confidence: 1.0,
            })
            .collect();
        let report = JitterMetrics::compute(&frames);
        assert!(
            report.rms_position_jitter < 1e-6,
            "static signal: rms={}",
            report.rms_position_jitter
        );
        assert!(report.max_position_jitter < 1e-6);
    }

    #[test]
    fn test_jitter_metrics_known_displacement() {
        // Two frames: (0,0,0) → (3,4,0) → distance = 5.
        let frames = vec![
            TrackingFrame {
                frame: 0,
                position: [0.0, 0.0, 0.0],
                rotation_quat: [0.0, 0.0, 0.0, 1.0],
                confidence: 1.0,
            },
            TrackingFrame {
                frame: 1,
                position: [3.0, 4.0, 0.0],
                rotation_quat: [0.0, 0.0, 0.0, 1.0],
                confidence: 1.0,
            },
        ];
        let report = JitterMetrics::compute(&frames);
        assert!(
            (report.rms_position_jitter - 5.0).abs() < 1e-4,
            "expected 5.0, got {}",
            report.rms_position_jitter
        );
        assert!(
            (report.max_position_jitter - 5.0).abs() < 1e-4
        );
    }

    #[test]
    fn test_jitter_rotation_nonzero_when_rotating() {
        // 90-degree rotation around Z: quat = [0, 0, sin(π/4), cos(π/4)]
        let half = std::f32::consts::FRAC_PI_4;
        let frames = vec![
            TrackingFrame {
                frame: 0,
                position: [0.0; 3],
                rotation_quat: [0.0, 0.0, 0.0, 1.0],
                confidence: 1.0,
            },
            TrackingFrame {
                frame: 1,
                position: [0.0; 3],
                rotation_quat: [0.0, 0.0, half.sin(), half.cos()],
                confidence: 1.0,
            },
        ];
        let report = JitterMetrics::compute(&frames);
        assert!(
            report.rms_rotation_jitter > 0.0,
            "rotation jitter should be non-zero"
        );
    }
}
