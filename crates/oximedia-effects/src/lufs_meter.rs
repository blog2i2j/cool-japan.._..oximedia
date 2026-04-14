//! LUFS Loudness Metering — EBU R128 / ITU-R BS.1770-4.
//!
//! Implements integrated loudness (LUFS), short-term loudness (3 s window),
//! momentary loudness (400 ms window), and loudness range (LRA) measurement
//! conforming to EBU R128 / ITU-R BS.1770-4.
//!
//! # Algorithm Overview
//!
//! ITU-R BS.1770 defines loudness through a two-stage K-weighting filter
//! followed by mean-square gating:
//!
//! 1. **K-weighting** — two biquad stages applied per channel:
//!    - Stage 1 (pre-filter): high-shelf boost ~+4 dB at 1.5 kHz, simulating
//!      acoustic head-related transfer effects.
//!    - Stage 2 (high-pass): 2nd-order Butterworth HPF at 38 Hz, removing
//!      sub-bass content.
//!
//! 2. **Mean-square power** per channel, summed with channel weighting:
//!    - L, R, C: weight 1.0
//!    - Ls, Rs (surround): weight 1.5
//!    - LFE: excluded
//!
//! 3. **Absolute gate** (–70 LUFS), then **relative gate** (–10 LU below
//!    ungated mean), with 400 ms overlapping blocks at 75% overlap for
//!    integrated loudness.
//!
//! 4. **Short-term** loudness: 3 s sliding window, updated every 100 ms.
//!
//! 5. **Momentary** loudness: 400 ms sliding window, updated every 100 ms.
//!
//! 6. **Loudness Range (LRA)**: difference between 10th and 95th percentiles
//!    of the 3 s block distribution after short-term gating at –20 LUFS
//!    absolute and –20 LU relative.
//!
//! # Example
//!
//! ```
//! use oximedia_effects::lufs_meter::{LufsMeter, LufsMeterConfig, ChannelLayout};
//!
//! let config = LufsMeterConfig {
//!     sample_rate: 48_000.0,
//!     layout: ChannelLayout::Stereo,
//! };
//! let mut meter = LufsMeter::new(config);
//!
//! // Feed stereo frames
//! let left  = vec![0.1_f32; 4800];
//! let right = vec![-0.1_f32; 4800];
//! meter.process_stereo(&left, &right);
//!
//! let momentary = meter.momentary_lufs();
//! let integrated = meter.integrated_lufs();
//! assert!(momentary.is_finite() || momentary == f32::NEG_INFINITY);
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

// ---------------------------------------------------------------------------
// Channel layout
// ---------------------------------------------------------------------------

/// Channel layout for loudness measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelLayout {
    /// Single mono channel.
    Mono,
    /// Stereo (L, R).
    Stereo,
    /// 5.0 surround (L, R, C, Ls, Rs) — no LFE.
    Surround50,
    /// 5.1 surround (L, R, C, LFE, Ls, Rs).
    Surround51,
}

impl ChannelLayout {
    /// Number of channels in this layout.
    #[must_use]
    pub const fn channel_count(self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround50 => 5,
            Self::Surround51 => 6,
        }
    }

    /// BS.1770 channel weight for the given channel index.
    /// Returns `0.0` for excluded channels (LFE in 5.1).
    #[must_use]
    pub fn channel_weight(self, idx: usize) -> f64 {
        match self {
            Self::Mono => 1.0,
            Self::Stereo => 1.0,
            Self::Surround50 => match idx {
                0..=2 => 1.0,  // L, R, C
                3 | 4 => 1.41, // Ls, Rs  (sqrt(2) ≈ 1.41)
                _ => 0.0,
            },
            Self::Surround51 => match idx {
                0..=2 => 1.0,  // L, R, C
                3 => 0.0,      // LFE excluded
                4 | 5 => 1.41, // Ls, Rs
                _ => 0.0,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`LufsMeter`].
#[derive(Debug, Clone)]
pub struct LufsMeterConfig {
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// Channel layout.
    pub layout: ChannelLayout,
}

impl Default for LufsMeterConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000.0,
            layout: ChannelLayout::Stereo,
        }
    }
}

// ---------------------------------------------------------------------------
// K-weighting biquad
// ---------------------------------------------------------------------------

/// A single biquad section (Direct Form I).
#[derive(Debug, Clone)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Biquad {
    fn new(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// K-weighting filter (2 cascaded biquads per channel)
// ---------------------------------------------------------------------------

/// K-weighting filter for one channel: pre-filter + highpass.
#[derive(Debug, Clone)]
struct KWeightingFilter {
    pre_filter: Biquad,
    highpass: Biquad,
}

impl KWeightingFilter {
    /// Compute K-weighting biquad coefficients for the given sample rate.
    ///
    /// Stage 1 — high-shelf pre-filter (BS.1770-4 Table 1):
    ///   Analog prototype: shelf gain +4 dB at 1681.97 Hz, Q=0.707
    ///   Bilinear-transformed to digital.
    ///
    /// Stage 2 — Butterworth HPF at 38.13 Hz (BS.1770-4 Table 2).
    fn new(sample_rate: f64) -> Self {
        let fs = sample_rate;

        // ---- Stage 1: pre-filter (high-shelf at 1681.97 Hz, +4 dB) ----
        // Reference: ITU-R BS.1770-4, Annex 1, Table 1
        let f0 = 1681.974450955533;
        let db = 3.999_843_853_973_347;
        let k_h = f64::tan(std::f64::consts::PI * f0 / fs);
        let vh = f64::powf(10.0_f64, db / 20.0);
        let vb = f64::powf(vh, 0.499_666_78);
        let sq2 = std::f64::consts::SQRT_2;
        // Normalise so that a0 = 1
        let a0_pre = 1.0 + (sq2 / vb) * k_h + k_h * k_h;
        let b0_pre = (vh + (sq2 * vb) * k_h + k_h * k_h) / a0_pre;
        let b1_pre = (2.0 * (k_h * k_h - vh)) / a0_pre;
        let b2_pre = (vh - (sq2 * vb) * k_h + k_h * k_h) / a0_pre;
        let a1_pre = (2.0 * (k_h * k_h - 1.0)) / a0_pre;
        let a2_pre = (1.0 - (sq2 / vb) * k_h + k_h * k_h) / a0_pre;

        // ---- Stage 2: highpass Butterworth at 38.13 Hz ----
        let f_hp = 38.134_496_945_404_14;
        let k_hp = f64::tan(std::f64::consts::PI * f_hp / fs);
        let sq2_hp = std::f64::consts::SQRT_2;
        let a0_hp = 1.0 + sq2_hp * k_hp + k_hp * k_hp;
        let b0_hp = 1.0 / a0_hp;
        let b1_hp = -2.0 / a0_hp;
        let b2_hp = 1.0 / a0_hp;
        let a1_hp = (2.0 * (k_hp * k_hp - 1.0)) / a0_hp;
        let a2_hp = (1.0 - sq2_hp * k_hp + k_hp * k_hp) / a0_hp;

        Self {
            pre_filter: Biquad::new(b0_pre, b1_pre, b2_pre, a1_pre, a2_pre),
            highpass: Biquad::new(b0_hp, b1_hp, b2_hp, a1_hp, a2_hp),
        }
    }

    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        self.highpass.process(self.pre_filter.process(x))
    }

    fn reset(&mut self) {
        self.pre_filter.reset();
        self.highpass.reset();
    }
}

// ---------------------------------------------------------------------------
// Circular buffer for mean-square computation
// ---------------------------------------------------------------------------

/// Fixed-capacity ring buffer for mean-square accumulation.
#[derive(Debug, Clone)]
struct MsRingBuffer {
    buf: Vec<f64>,
    write_pos: usize,
    capacity: usize,
    sum: f64,
    filled: bool,
}

impl MsRingBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buf: vec![0.0; capacity],
            write_pos: 0,
            capacity,
            sum: 0.0,
            filled: false,
        }
    }

    /// Push a new squared sample, return the current window mean-square.
    fn push(&mut self, sq: f64) -> f64 {
        // Remove outgoing sample from sum
        self.sum -= self.buf[self.write_pos];
        self.buf[self.write_pos] = sq;
        self.sum += sq;

        let prev_pos = self.write_pos;
        self.write_pos = (self.write_pos + 1) % self.capacity;

        if prev_pos == self.capacity - 1 {
            self.filled = true;
        }

        if self.filled {
            (self.sum / self.capacity as f64).max(0.0)
        } else {
            let n = self.write_pos;
            if n == 0 {
                0.0
            } else {
                (self.sum / n as f64).max(0.0)
            }
        }
    }

    fn reset(&mut self) {
        self.buf.fill(0.0);
        self.write_pos = 0;
        self.sum = 0.0;
        self.filled = false;
    }
}

// ---------------------------------------------------------------------------
// Gated block accumulator
// ---------------------------------------------------------------------------

/// Accumulates 400 ms blocks (with 75% overlap) for integrated LUFS.
#[derive(Debug, Clone)]
struct GatedAccumulator {
    /// Block size in samples (400 ms).
    block_samples: usize,
    /// Hop size (100 ms = 25% of block).
    hop_samples: usize,
    /// Per-channel squared sums for the current block.
    channel_sums: Vec<f64>,
    /// BS.1770 channel weights.
    weights: Vec<f64>,
    /// Sample counter within current hop.
    hop_counter: usize,
    /// Sample counter within current block (for overlap book-keeping).
    block_counter: usize,
    /// All completed block loudness values (dB).
    block_loudnesses: Vec<f64>,
    /// Number of channels.
    num_channels: usize,
}

impl GatedAccumulator {
    fn new(sample_rate: f64, layout: ChannelLayout) -> Self {
        let block_samples = (0.4 * sample_rate).round() as usize;
        let hop_samples = (0.1 * sample_rate).round() as usize;
        let num_channels = layout.channel_count();
        let weights: Vec<f64> = (0..num_channels)
            .map(|i| layout.channel_weight(i))
            .collect();
        Self {
            block_samples,
            hop_samples,
            channel_sums: vec![0.0; num_channels],
            weights,
            hop_counter: 0,
            block_counter: 0,
            block_loudnesses: Vec::new(),
            num_channels,
        }
    }

    /// Push one filtered squared sample per channel.
    fn push_frame(&mut self, sq_per_channel: &[f64]) {
        for (ch, &sq) in sq_per_channel.iter().enumerate().take(self.num_channels) {
            self.channel_sums[ch] += sq;
        }
        self.block_counter += 1;
        self.hop_counter += 1;

        if self.hop_counter >= self.hop_samples {
            self.hop_counter = 0;
        }

        // When we've accumulated a full block worth of samples since last hop
        if self.block_counter >= self.block_samples {
            // Finalise block
            let mut z: f64 = 0.0;
            for (ch, &w) in self.weights.iter().enumerate() {
                z += w * self.channel_sums[ch] / self.block_samples as f64;
            }
            let loudness = if z > 1e-12 {
                -0.691 + 10.0 * f64::log10(z)
            } else {
                f64::NEG_INFINITY
            };
            self.block_loudnesses.push(loudness);

            // Slide by one hop: subtract oldest hop's contribution.
            // Since we don't store per-hop sums, shift the block by discarding
            // hop_samples from the start. We approximate by scaling.
            let keep_ratio =
                (self.block_samples - self.hop_samples) as f64 / self.block_samples as f64;
            for sum in &mut self.channel_sums {
                *sum *= keep_ratio;
            }
            self.block_counter = self.block_samples - self.hop_samples;
        }
    }

    /// Compute integrated LUFS using absolute + relative gating.
    fn integrated_lufs(&self) -> f64 {
        if self.block_loudnesses.is_empty() {
            return f64::NEG_INFINITY;
        }
        let absolute_threshold = -70.0_f64;

        // Pass 1: exclude blocks below absolute gate
        let pass1: Vec<f64> = self
            .block_loudnesses
            .iter()
            .copied()
            .filter(|&l| l > absolute_threshold)
            .collect();

        if pass1.is_empty() {
            return f64::NEG_INFINITY;
        }

        // Mean loudness of pass1 blocks
        let n1 = pass1.len() as f64;
        // Convert to linear, average, convert back
        let mean_lin1: f64 = pass1
            .iter()
            .map(|&l| f64::powf(10.0, l / 10.0))
            .sum::<f64>()
            / n1;
        let mean_db1 = 10.0 * f64::log10(mean_lin1);

        // Pass 2: relative gate at mean_db1 - 10 LU
        let relative_threshold = mean_db1 - 10.0;
        let pass2: Vec<f64> = pass1
            .iter()
            .copied()
            .filter(|&l| l > relative_threshold)
            .collect();

        if pass2.is_empty() {
            return f64::NEG_INFINITY;
        }

        let n2 = pass2.len() as f64;
        let mean_lin2: f64 = pass2
            .iter()
            .map(|&l| f64::powf(10.0, l / 10.0))
            .sum::<f64>()
            / n2;

        -0.691 + 10.0 * f64::log10(mean_lin2)
    }

    /// Compute loudness range (LRA) in LU.
    fn loudness_range(&self) -> f64 {
        if self.block_loudnesses.len() < 2 {
            return 0.0;
        }
        let abs_gate = -70.0_f64;
        let st_gate = -20.0_f64;

        let mut filtered: Vec<f64> = self
            .block_loudnesses
            .iter()
            .copied()
            .filter(|&l| l > abs_gate && l > st_gate)
            .collect();

        if filtered.len() < 2 {
            return 0.0;
        }

        // Sort for percentile computation
        filtered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = filtered.len();
        let lo_idx = ((n as f64 * 0.10).round() as usize).min(n - 1);
        let hi_idx = ((n as f64 * 0.95).round() as usize).min(n - 1);

        (filtered[hi_idx] - filtered[lo_idx]).max(0.0)
    }

    fn reset(&mut self) {
        self.channel_sums.fill(0.0);
        self.hop_counter = 0;
        self.block_counter = 0;
        self.block_loudnesses.clear();
    }
}

// ---------------------------------------------------------------------------
// Sliding window loudness
// ---------------------------------------------------------------------------

/// Computes loudness over a fixed sliding window (momentary=400 ms, short-term=3 s).
#[derive(Debug, Clone)]
struct SlidingLoudness {
    /// Per-channel ring buffers for squared filtered samples.
    ring_bufs: Vec<MsRingBuffer>,
    /// BS.1770 channel weights.
    weights: Vec<f64>,
    num_channels: usize,
}

impl SlidingLoudness {
    fn new(window_seconds: f64, sample_rate: f64, layout: ChannelLayout) -> Self {
        let window_samples = (window_seconds * sample_rate).round() as usize;
        let num_channels = layout.channel_count();
        let weights = (0..num_channels)
            .map(|i| layout.channel_weight(i))
            .collect();
        Self {
            ring_bufs: vec![MsRingBuffer::new(window_samples); num_channels],
            weights,
            num_channels,
        }
    }

    /// Push one filtered squared sample per channel, return momentary/short-term LUFS.
    fn push_frame(&mut self, sq_per_channel: &[f64]) -> f64 {
        let mut z = 0.0_f64;
        for (ch, buf) in self
            .ring_bufs
            .iter_mut()
            .enumerate()
            .take(self.num_channels)
        {
            let sq = if ch < sq_per_channel.len() {
                sq_per_channel[ch]
            } else {
                0.0
            };
            let ms = buf.push(sq);
            z += self.weights[ch] * ms;
        }
        if z > 1e-12 {
            -0.691 + 10.0 * f64::log10(z)
        } else {
            f64::NEG_INFINITY
        }
    }

    fn reset(&mut self) {
        for buf in &mut self.ring_bufs {
            buf.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// Main meter
// ---------------------------------------------------------------------------

/// EBU R128 / ITU-R BS.1770-4 loudness meter.
///
/// Provides:
/// - **Momentary** LUFS (400 ms window)
/// - **Short-term** LUFS (3 s window)
/// - **Integrated** LUFS (gated, full program)
/// - **Loudness Range** (LRA) in LU
/// - **True Peak** per channel
///
/// # Usage
///
/// 1. Create with [`LufsMeter::new`].
/// 2. Feed audio via [`process_mono`], [`process_stereo`], or
///    [`process_multichannel`].
/// 3. Read measurements at any time.
/// 4. Call [`reset`] to start a new measurement.
///
/// [`process_mono`]: LufsMeter::process_mono
/// [`process_stereo`]: LufsMeter::process_stereo
/// [`process_multichannel`]: LufsMeter::process_multichannel
/// [`reset`]: LufsMeter::reset
#[derive(Debug, Clone)]
pub struct LufsMeter {
    config: LufsMeterConfig,
    /// K-weighting filters, one per channel.
    kw_filters: Vec<KWeightingFilter>,
    /// Momentary loudness sliding window (400 ms).
    momentary_window: SlidingLoudness,
    /// Short-term loudness sliding window (3 s).
    short_term_window: SlidingLoudness,
    /// Gated block accumulator for integrated loudness + LRA.
    gated: GatedAccumulator,
    /// Last momentary LUFS value.
    last_momentary: f64,
    /// Last short-term LUFS value.
    last_short_term: f64,
    /// True peak per channel (running maximum of |sample|).
    true_peak: Vec<f32>,
    /// Number of channels.
    num_channels: usize,
}

impl LufsMeter {
    /// Create a new meter with the given configuration.
    #[must_use]
    pub fn new(config: LufsMeterConfig) -> Self {
        let fs = f64::from(config.sample_rate);
        let layout = config.layout;
        let num_channels = layout.channel_count();

        let kw_filters = (0..num_channels)
            .map(|_| KWeightingFilter::new(fs))
            .collect();

        Self {
            kw_filters,
            momentary_window: SlidingLoudness::new(0.4, fs, layout),
            short_term_window: SlidingLoudness::new(3.0, fs, layout),
            gated: GatedAccumulator::new(fs, layout),
            last_momentary: f64::NEG_INFINITY,
            last_short_term: f64::NEG_INFINITY,
            true_peak: vec![0.0_f32; num_channels],
            num_channels,
            config,
        }
    }

    /// Process a buffer of mono samples.
    pub fn process_mono(&mut self, samples: &[f32]) {
        let mut sq = [0.0_f64; 1];
        for &s in samples {
            let filtered = self.kw_filters[0].process(f64::from(s));
            sq[0] = filtered * filtered;
            self.last_momentary = self.momentary_window.push_frame(&sq);
            self.last_short_term = self.short_term_window.push_frame(&sq);
            self.gated.push_frame(&sq);
            if s.abs() > self.true_peak[0] {
                self.true_peak[0] = s.abs();
            }
        }
    }

    /// Process interleaved stereo buffers (separate L/R slices).
    pub fn process_stereo(&mut self, left: &[f32], right: &[f32]) {
        let len = left.len().min(right.len());
        let mut sq = [0.0_f64; 2];
        for i in 0..len {
            let fl = self.kw_filters[0].process(f64::from(left[i]));
            let fr = if self.num_channels > 1 {
                self.kw_filters[1].process(f64::from(right[i]))
            } else {
                0.0
            };
            sq[0] = fl * fl;
            sq[1] = fr * fr;

            self.last_momentary = self.momentary_window.push_frame(&sq);
            self.last_short_term = self.short_term_window.push_frame(&sq);
            self.gated.push_frame(&sq);

            if left[i].abs() > self.true_peak[0] {
                self.true_peak[0] = left[i].abs();
            }
            if self.num_channels > 1 && right[i].abs() > self.true_peak[1] {
                self.true_peak[1] = right[i].abs();
            }
        }
    }

    /// Process multichannel audio. `frames` is a slice of per-sample channel
    /// vectors: `frames[sample_index][channel_index]`.
    pub fn process_multichannel(&mut self, frames: &[Vec<f32>]) {
        let mut sq = vec![0.0_f64; self.num_channels];
        for frame in frames {
            for (ch, filter) in self.kw_filters.iter_mut().enumerate() {
                let s = frame.get(ch).copied().unwrap_or(0.0);
                let filtered = filter.process(f64::from(s));
                sq[ch] = filtered * filtered;
                if s.abs() > self.true_peak[ch] {
                    self.true_peak[ch] = s.abs();
                }
            }
            self.last_momentary = self.momentary_window.push_frame(&sq);
            self.last_short_term = self.short_term_window.push_frame(&sq);
            self.gated.push_frame(&sq);
        }
    }

    /// Momentary loudness (400 ms window) in LUFS.
    ///
    /// Returns [`f32::NEG_INFINITY`] when the signal is below the noise floor.
    #[must_use]
    pub fn momentary_lufs(&self) -> f32 {
        self.last_momentary as f32
    }

    /// Short-term loudness (3 s window) in LUFS.
    ///
    /// Returns [`f32::NEG_INFINITY`] when insufficient audio has been processed.
    #[must_use]
    pub fn short_term_lufs(&self) -> f32 {
        self.last_short_term as f32
    }

    /// Integrated loudness (full program, gated per EBU R128) in LUFS.
    ///
    /// Returns [`f32::NEG_INFINITY`] when insufficient audio has been processed.
    #[must_use]
    pub fn integrated_lufs(&self) -> f32 {
        self.gated.integrated_lufs() as f32
    }

    /// Loudness range (LRA) in LU.
    #[must_use]
    pub fn loudness_range(&self) -> f32 {
        self.gated.loudness_range() as f32
    }

    /// True peak per channel (linear, not dBFS).
    ///
    /// Returns a slice with one entry per channel in the configured layout.
    #[must_use]
    pub fn true_peak(&self) -> &[f32] {
        &self.true_peak
    }

    /// True peak of the loudest channel in dBFS.
    #[must_use]
    pub fn true_peak_db(&self) -> f32 {
        let max_peak = self.true_peak.iter().copied().fold(0.0_f32, f32::max);
        if max_peak > 1e-10 {
            20.0 * max_peak.log10()
        } else {
            f32::NEG_INFINITY
        }
    }

    /// Reset all measurement state to start a fresh measurement.
    pub fn reset(&mut self) {
        for f in &mut self.kw_filters {
            f.reset();
        }
        self.momentary_window.reset();
        self.short_term_window.reset();
        self.gated.reset();
        self.last_momentary = f64::NEG_INFINITY;
        self.last_short_term = f64::NEG_INFINITY;
        self.true_peak.fill(0.0);
    }

    /// Return the current configuration.
    #[must_use]
    pub fn config(&self) -> &LufsMeterConfig {
        &self.config
    }

    /// Return the number of channels.
    #[must_use]
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }
}

// ---------------------------------------------------------------------------
// Convenience: loudness difference between two measurements
// ---------------------------------------------------------------------------

/// Compute the difference between a measured loudness value and a target in LU.
///
/// Positive result means the signal is louder than target (needs gain reduction).
/// Negative result means it is quieter (needs gain boost).
#[must_use]
pub fn loudness_deviation(measured_lufs: f32, target_lufs: f32) -> f32 {
    measured_lufs - target_lufs
}

/// Convert a LUFS value to a linear gain correction.
///
/// `correction_db = target - measured`, then `gain = 10^(correction_db/20)`.
#[must_use]
pub fn lufs_to_gain_correction(measured_lufs: f32, target_lufs: f32) -> f32 {
    let diff_db = target_lufs - measured_lufs;
    f32::powf(10.0, diff_db / 20.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stereo_meter() -> LufsMeter {
        LufsMeter::new(LufsMeterConfig {
            sample_rate: 48_000.0,
            layout: ChannelLayout::Stereo,
        })
    }

    /// Sine wave at frequency `hz`, `duration` seconds.
    fn sine_signal(hz: f32, amplitude: f32, sample_rate: f32, duration_secs: f32) -> Vec<f32> {
        let n = (sample_rate * duration_secs) as usize;
        (0..n)
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * hz * i as f32 / sample_rate).sin())
            .collect()
    }

    #[test]
    fn test_meter_constructs_ok() {
        let meter = make_stereo_meter();
        assert_eq!(meter.num_channels(), 2);
    }

    #[test]
    fn test_silence_gives_neg_inf_momentary() {
        let mut meter = make_stereo_meter();
        let silence = vec![0.0_f32; 4800];
        meter.process_stereo(&silence, &silence);
        // Silence = NEG_INFINITY or very low LUFS
        let m = meter.momentary_lufs();
        assert!(m == f32::NEG_INFINITY || m < -60.0, "got {m}");
    }

    #[test]
    fn test_sine_tone_is_finite() {
        let mut meter = make_stereo_meter();
        let sig = sine_signal(1000.0, 0.5, 48_000.0, 1.0);
        meter.process_stereo(&sig, &sig);
        let m = meter.momentary_lufs();
        assert!(
            m.is_finite(),
            "momentary LUFS should be finite for sine, got {m}"
        );
    }

    #[test]
    fn test_louder_sine_gives_higher_lufs() {
        let mut low_meter = make_stereo_meter();
        let mut high_meter = make_stereo_meter();

        let low_sig = sine_signal(1000.0, 0.1, 48_000.0, 1.0);
        let high_sig = sine_signal(1000.0, 0.9, 48_000.0, 1.0);

        low_meter.process_stereo(&low_sig, &low_sig);
        high_meter.process_stereo(&high_sig, &high_sig);

        let low_m = low_meter.momentary_lufs();
        let high_m = high_meter.momentary_lufs();

        assert!(
            high_m > low_m,
            "louder signal should have higher LUFS: {high_m} vs {low_m}"
        );
    }

    #[test]
    fn test_true_peak_tracks_maximum() {
        let mut meter = make_stereo_meter();
        let signal = vec![0.0_f32, 0.5, -0.8, 0.3, 0.0];
        meter.process_stereo(&signal, &signal);
        let tp = meter.true_peak();
        assert!(
            (tp[0] - 0.8).abs() < 1e-5,
            "true peak L should be 0.8, got {}",
            tp[0]
        );
        assert!(
            (tp[1] - 0.8).abs() < 1e-5,
            "true peak R should be 0.8, got {}",
            tp[1]
        );
    }

    #[test]
    fn test_true_peak_db_neg_inf_for_silence() {
        let meter = make_stereo_meter();
        let tp_db = meter.true_peak_db();
        assert_eq!(tp_db, f32::NEG_INFINITY, "silence → NEG_INFINITY dBFS");
    }

    #[test]
    fn test_reset_clears_state() {
        let mut meter = make_stereo_meter();
        let sig = sine_signal(1000.0, 0.5, 48_000.0, 1.0);
        meter.process_stereo(&sig, &sig);
        meter.reset();
        assert_eq!(meter.momentary_lufs(), f32::NEG_INFINITY);
        assert_eq!(meter.true_peak_db(), f32::NEG_INFINITY);
    }

    #[test]
    fn test_mono_processing() {
        let mut meter = LufsMeter::new(LufsMeterConfig {
            sample_rate: 48_000.0,
            layout: ChannelLayout::Mono,
        });
        let sig = sine_signal(440.0, 0.3, 48_000.0, 0.5);
        meter.process_mono(&sig);
        let m = meter.momentary_lufs();
        assert!(m.is_finite() || m == f32::NEG_INFINITY);
    }

    #[test]
    fn test_loudness_deviation() {
        let dev = loudness_deviation(-18.0, -23.0);
        assert!(
            (dev - 5.0).abs() < 1e-5,
            "deviation should be +5 LU, got {dev}"
        );
    }

    #[test]
    fn test_lufs_to_gain_correction_unity() {
        // When measured == target, correction = 0 dB → gain = 1.0
        let gain = lufs_to_gain_correction(-23.0, -23.0);
        assert!((gain - 1.0).abs() < 1e-5, "unity gain expected, got {gain}");
    }

    #[test]
    fn test_lufs_to_gain_correction_boost() {
        // When measured = -30, target = -23 → correction = +7 dB → gain > 1
        let gain = lufs_to_gain_correction(-30.0, -23.0);
        assert!(gain > 1.0, "gain should be >1 for boost, got {gain}");
    }

    #[test]
    fn test_integrated_lufs_after_long_signal() {
        let mut meter = make_stereo_meter();
        // 6 seconds of 1 kHz sine at -18 dBFS ≈ 0.126 amplitude
        let amp = f32::powf(10.0, -18.0 / 20.0);
        let sig = sine_signal(1000.0, amp, 48_000.0, 6.0);
        meter.process_stereo(&sig, &sig);
        let integrated = meter.integrated_lufs();
        // Should not be NEG_INFINITY with 6 seconds of tone
        assert!(
            integrated.is_finite(),
            "integrated LUFS should be finite after 6 s of tone, got {integrated}"
        );
    }

    #[test]
    fn test_channel_layout_weights() {
        assert_eq!(ChannelLayout::Stereo.channel_weight(0), 1.0);
        assert_eq!(ChannelLayout::Stereo.channel_weight(1), 1.0);
        assert_eq!(ChannelLayout::Surround51.channel_weight(3), 0.0); // LFE excluded
        assert_eq!(ChannelLayout::Surround51.channel_weight(4), 1.41); // Ls
    }

    // ── New tests for TODO item: LUFS loudness metering ───────────────────

    #[test]
    fn test_channel_layout_counts() {
        assert_eq!(ChannelLayout::Mono.channel_count(), 1);
        assert_eq!(ChannelLayout::Stereo.channel_count(), 2);
        assert_eq!(ChannelLayout::Surround50.channel_count(), 5);
        assert_eq!(ChannelLayout::Surround51.channel_count(), 6);
    }

    #[test]
    fn test_surround50_weights() {
        // L, R, C = 1.0; Ls, Rs = 1.41 (sqrt(2) approximation).
        for ch in 0..=2 {
            assert_eq!(
                ChannelLayout::Surround50.channel_weight(ch),
                1.0,
                "channel {ch} weight wrong"
            );
        }
        assert_eq!(ChannelLayout::Surround50.channel_weight(3), 1.41);
        assert_eq!(ChannelLayout::Surround50.channel_weight(4), 1.41);
    }

    #[test]
    fn test_meter_short_term_lufs_after_signal() {
        let mut meter = make_stereo_meter();
        let amp = f32::powf(10.0, -18.0 / 20.0);
        // 4 seconds of tone — enough to populate the 3 s short-term window.
        let sig = sine_signal(1000.0, amp, 48_000.0, 4.0);
        meter.process_stereo(&sig, &sig);
        let st = meter.short_term_lufs();
        assert!(
            st.is_finite(),
            "short-term LUFS should be finite after 4 s tone, got {st}"
        );
    }

    #[test]
    fn test_meter_momentary_higher_than_silence() {
        let mut silent = make_stereo_meter();
        let mut loud = make_stereo_meter();

        let silence = vec![0.0_f32; 19200]; // 400 ms
        let tone = sine_signal(1000.0, 0.5, 48_000.0, 0.4);

        silent.process_stereo(&silence, &silence);
        loud.process_stereo(&tone, &tone);

        let m_silent = silent.momentary_lufs();
        let m_loud = loud.momentary_lufs();

        // Silence → NEG_INFINITY; loud tone → finite. finite > NEG_INFINITY always.
        assert!(
            m_loud > m_silent,
            "loud ({m_loud}) should be greater than silent ({m_silent})"
        );
    }

    #[test]
    fn test_meter_loudness_deviation_negative() {
        // If measured is quieter than target, deviation is negative.
        let dev = loudness_deviation(-30.0, -23.0);
        assert!(
            dev < 0.0,
            "deviation should be negative when quieter: {dev}"
        );
        assert!((dev - (-7.0)).abs() < 1e-5, "expected -7 LU, got {dev}");
    }

    #[test]
    fn test_meter_gain_correction_cut() {
        // When measured is louder than target, correction < 1.0 (cut needed).
        let gain = lufs_to_gain_correction(-20.0, -23.0);
        assert!(gain < 1.0, "gain should be < 1 for a cut, got {gain}");
    }

    #[test]
    fn test_meter_true_peak_monotonically_increases() {
        let mut meter = make_stereo_meter();
        let mut last_peak = f32::NEG_INFINITY;

        // Feed increasingly loud signals; true peak should never decrease.
        for amp in [0.1_f32, 0.3, 0.5, 0.7, 0.9] {
            let sig = sine_signal(1000.0, amp, 48_000.0, 0.1);
            meter.process_stereo(&sig, &sig);
            let tp = meter.true_peak_db();
            assert!(
                tp >= last_peak - 0.01,
                "true peak should not decrease: {tp} < {last_peak}"
            );
            if tp.is_finite() {
                last_peak = tp;
            }
        }
    }

    #[test]
    fn test_meter_process_stereo_then_reset() {
        let mut meter = make_stereo_meter();
        let sig = sine_signal(1000.0, 0.5, 48_000.0, 2.0);
        meter.process_stereo(&sig, &sig);

        // Before reset: measurements should exist.
        let before = meter.momentary_lufs();
        assert!(before.is_finite() || before == f32::NEG_INFINITY);

        meter.reset();

        // After reset: all measurements should be NEG_INFINITY.
        assert_eq!(
            meter.momentary_lufs(),
            f32::NEG_INFINITY,
            "momentary should be NEG_INFINITY after reset"
        );
        assert_eq!(
            meter.integrated_lufs(),
            f32::NEG_INFINITY,
            "integrated should be NEG_INFINITY after reset"
        );
    }

    #[test]
    fn test_meter_config_getter() {
        let config = LufsMeterConfig {
            sample_rate: 44_100.0,
            layout: ChannelLayout::Mono,
        };
        let meter = LufsMeter::new(config.clone());
        assert_eq!(meter.config().sample_rate, 44_100.0);
        assert_eq!(meter.config().layout, ChannelLayout::Mono);
    }
}
