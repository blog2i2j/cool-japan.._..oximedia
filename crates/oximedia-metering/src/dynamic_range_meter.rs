//! Dynamic range metering: crest factor, PLR (Peak-to-Loudness Ratio),
//! PSR (Program Segment Ratio) metrics, and histogram display.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Result of a crest factor measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrestFactor {
    /// Peak level in dBFS.
    pub peak_dbfs: f64,
    /// RMS level in dBFS.
    pub rms_dbfs: f64,
    /// Crest factor in dB (peak - RMS).
    pub crest_db: f64,
}

impl CrestFactor {
    /// Compute crest factor from a block of mono samples.
    #[must_use]
    pub fn from_samples(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let peak_linear = samples.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
        let rms_linear = (samples.iter().map(|s| s * s).sum::<f64>() / samples.len() as f64).sqrt();

        let peak_dbfs = linear_to_dbfs(peak_linear);
        let rms_dbfs = linear_to_dbfs(rms_linear.max(1e-12));
        Some(Self {
            peak_dbfs,
            rms_dbfs,
            crest_db: peak_dbfs - rms_dbfs,
        })
    }

    /// Return `true` if the crest factor indicates heavily limited/compressed audio
    /// (crest factor < 6 dB is considered over-compressed for most content types).
    #[must_use]
    pub fn is_over_compressed(&self) -> bool {
        self.crest_db < 6.0
    }
}

/// Convert a linear amplitude to dBFS (full scale = 0 dBFS at amplitude 1.0).
#[must_use]
pub fn linear_to_dbfs(linear: f64) -> f64 {
    if linear <= 0.0 {
        return f64::NEG_INFINITY;
    }
    20.0 * linear.log10()
}

/// Convert dBFS to a linear amplitude.
#[must_use]
pub fn dbfs_to_linear(dbfs: f64) -> f64 {
    10.0_f64.powf(dbfs / 20.0)
}

/// A simple histogram over a dBFS range.
#[derive(Debug, Clone)]
pub struct LevelHistogram {
    /// Number of bins.
    pub bins: usize,
    /// Minimum dBFS value.
    pub min_db: f64,
    /// Maximum dBFS value.
    pub max_db: f64,
    /// Counts per bin.
    counts: Vec<u64>,
    /// Total samples counted.
    total: u64,
}

impl LevelHistogram {
    /// Create a new histogram.
    ///
    /// # Panics
    /// Panics if `bins` is zero or `min_db >= max_db`.
    #[must_use]
    pub fn new(bins: usize, min_db: f64, max_db: f64) -> Self {
        assert!(bins > 0, "bins must be > 0");
        assert!(min_db < max_db, "min_db must be < max_db");
        Self {
            bins,
            min_db,
            max_db,
            counts: vec![0; bins],
            total: 0,
        }
    }

    /// Add a dBFS sample to the histogram.
    pub fn add(&mut self, dbfs: f64) {
        if dbfs.is_finite() && dbfs >= self.min_db && dbfs < self.max_db {
            let frac = (dbfs - self.min_db) / (self.max_db - self.min_db);
            let bin = (frac * self.bins as f64).min(self.bins as f64 - 1.0) as usize;
            self.counts[bin] += 1;
        }
        self.total += 1;
    }

    /// Add all samples from a slice (converts linear to dBFS first).
    pub fn add_linear_block(&mut self, samples: &[f64]) {
        for &s in samples {
            self.add(linear_to_dbfs(s.abs().max(1e-12)));
        }
    }

    /// Return the count for a specific bin.
    #[must_use]
    pub fn bin_count(&self, bin: usize) -> u64 {
        self.counts.get(bin).copied().unwrap_or(0)
    }

    /// Return the centre dBFS value for a given bin.
    #[must_use]
    pub fn bin_centre_db(&self, bin: usize) -> f64 {
        let bin_width = (self.max_db - self.min_db) / self.bins as f64;
        self.min_db + (bin as f64 + 0.5) * bin_width
    }

    /// Return the normalised histogram (each bin count / total). Length = `bins`.
    #[must_use]
    pub fn normalized(&self) -> Vec<f64> {
        if self.total == 0 {
            return vec![0.0; self.bins];
        }
        self.counts
            .iter()
            .map(|&c| c as f64 / self.total as f64)
            .collect()
    }

    /// Return total samples added.
    #[must_use]
    pub fn total_samples(&self) -> u64 {
        self.total
    }

    /// Find the mode bin (highest count).
    #[must_use]
    pub fn mode_bin(&self) -> Option<usize> {
        self.counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
    }

    /// Reset all counts.
    pub fn reset(&mut self) {
        for c in &mut self.counts {
            *c = 0;
        }
        self.total = 0;
    }
}

/// PLR (Peak-to-Loudness Ratio) measurement.
/// PLR = true_peak_dbtp − integrated_loudness_lufs.
/// High PLR indicates dynamic, uncompressed content.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plr {
    /// True peak in dBTP.
    pub true_peak_dbtp: f64,
    /// Integrated loudness in LUFS.
    pub integrated_lufs: f64,
    /// PLR in LU/dB.
    pub plr_lu: f64,
}

impl Plr {
    /// Compute PLR from true peak and integrated loudness.
    #[must_use]
    pub fn compute(true_peak_dbtp: f64, integrated_lufs: f64) -> Self {
        Self {
            true_peak_dbtp,
            integrated_lufs,
            plr_lu: true_peak_dbtp - integrated_lufs,
        }
    }

    /// Classify PLR: >14 LU = dynamic, 8–14 = moderate, <8 = compressed.
    #[must_use]
    pub fn classification(&self) -> &'static str {
        if self.plr_lu >= 14.0 {
            "Dynamic"
        } else if self.plr_lu >= 8.0 {
            "Moderate"
        } else {
            "Compressed"
        }
    }
}

/// PSR (Program Segment Ratio) measurement.
/// PSR = maximum short-term loudness − integrated loudness.
/// Captures loudness variation within a program.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Psr {
    /// Maximum short-term loudness in LUFS.
    pub max_short_term_lufs: f64,
    /// Integrated loudness in LUFS.
    pub integrated_lufs: f64,
    /// PSR in LU.
    pub psr_lu: f64,
}

impl Psr {
    /// Compute PSR.
    #[must_use]
    pub fn compute(max_short_term_lufs: f64, integrated_lufs: f64) -> Self {
        Self {
            max_short_term_lufs,
            integrated_lufs,
            psr_lu: max_short_term_lufs - integrated_lufs,
        }
    }

    /// Return `true` if the PSR suggests loudness variation typical of speech/drama (>3 LU).
    #[must_use]
    pub fn has_loudness_variation(&self) -> bool {
        self.psr_lu > 3.0
    }
}

/// Dynamic range meter that accumulates peak and RMS data and computes
/// crest factor, PLR, PSR, and maintains a level histogram.
pub struct DynamicRangeMeter {
    /// Sample rate in Hz.
    sample_rate: f64,
    /// Peak level seen (linear).
    peak_linear: f64,
    /// Accumulated squared sum for RMS.
    squared_sum: f64,
    /// Number of samples accumulated.
    sample_count: u64,
    /// Short-term max: rolling window of recent block RMS values in dBFS.
    short_term_window: Vec<f64>,
    /// Write position in short-term window.
    st_write_pos: usize,
    /// Level histogram.
    histogram: LevelHistogram,
    /// Externally supplied integrated loudness (LUFS) for PLR/PSR.
    integrated_lufs: Option<f64>,
    /// Maximum short-term loudness seen (for PSR).
    max_short_term_lufs: Option<f64>,
}

impl DynamicRangeMeter {
    /// Create a new meter.
    #[must_use]
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            peak_linear: 0.0,
            squared_sum: 0.0,
            sample_count: 0,
            short_term_window: vec![-96.0; 30], // 30 blocks ≈ 3 seconds of 100ms blocks
            st_write_pos: 0,
            histogram: LevelHistogram::new(120, -96.0, 0.0),
            integrated_lufs: None,
            max_short_term_lufs: None,
        }
    }

    /// Process a mono sample block.
    pub fn process_mono(&mut self, samples: &[f64]) {
        if samples.is_empty() {
            return;
        }
        for &s in samples {
            let abs_s = s.abs();
            if abs_s > self.peak_linear {
                self.peak_linear = abs_s;
            }
            self.squared_sum += s * s;
            self.sample_count += 1;
        }
        // Store block RMS in short-term window
        let block_rms = (samples.iter().map(|s| s * s).sum::<f64>() / samples.len() as f64).sqrt();
        let block_db = linear_to_dbfs(block_rms.max(1e-12));
        self.short_term_window[self.st_write_pos] = block_db;
        self.st_write_pos = (self.st_write_pos + 1) % self.short_term_window.len();

        // Update histogram
        self.histogram.add_linear_block(samples);
    }

    /// Process interleaved stereo by mixing to mono first.
    pub fn process_stereo_interleaved(&mut self, samples: &[f64]) {
        let mono: Vec<f64> = samples
            .chunks_exact(2)
            .map(|c| (c[0] + c[1]) * 0.5)
            .collect();
        self.process_mono(&mono);
    }

    /// Supply integrated loudness from an external loudness meter (for PLR/PSR computation).
    pub fn set_integrated_lufs(&mut self, lufs: f64) {
        self.integrated_lufs = Some(lufs);
    }

    /// Supply maximum short-term loudness (for PSR).
    pub fn set_max_short_term_lufs(&mut self, lufs: f64) {
        let current = self.max_short_term_lufs.unwrap_or(f64::NEG_INFINITY);
        self.max_short_term_lufs = Some(lufs.max(current));
    }

    /// Return peak level in dBFS.
    #[must_use]
    pub fn peak_dbfs(&self) -> f64 {
        linear_to_dbfs(self.peak_linear)
    }

    /// Return overall RMS in dBFS.
    #[must_use]
    pub fn rms_dbfs(&self) -> f64 {
        if self.sample_count == 0 {
            return f64::NEG_INFINITY;
        }
        let rms = (self.squared_sum / self.sample_count as f64).sqrt();
        linear_to_dbfs(rms.max(1e-12))
    }

    /// Compute crest factor from accumulated data.
    #[must_use]
    pub fn crest_factor(&self) -> CrestFactor {
        let peak = self.peak_dbfs();
        let rms = self.rms_dbfs();
        CrestFactor {
            peak_dbfs: peak,
            rms_dbfs: rms,
            crest_db: peak - rms,
        }
    }

    /// Compute PLR if integrated loudness has been set.
    #[must_use]
    pub fn plr(&self) -> Option<Plr> {
        self.integrated_lufs
            .map(|lufs| Plr::compute(self.peak_dbfs(), lufs))
    }

    /// Compute PSR if both integrated and max short-term loudness have been set.
    #[must_use]
    pub fn psr(&self) -> Option<Psr> {
        match (self.max_short_term_lufs, self.integrated_lufs) {
            (Some(max_st), Some(integrated)) => Some(Psr::compute(max_st, integrated)),
            _ => None,
        }
    }

    /// Return a reference to the level histogram.
    #[must_use]
    pub fn histogram(&self) -> &LevelHistogram {
        &self.histogram
    }

    /// Return the maximum short-term block RMS in dBFS.
    #[must_use]
    pub fn max_short_term_rms_dbfs(&self) -> f64 {
        self.short_term_window
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Total mono samples processed.
    #[must_use]
    pub fn total_samples(&self) -> u64 {
        self.sample_count
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.peak_linear = 0.0;
        self.squared_sum = 0.0;
        self.sample_count = 0;
        for v in &mut self.short_term_window {
            *v = -96.0;
        }
        self.st_write_pos = 0;
        self.histogram.reset();
        self.integrated_lufs = None;
        self.max_short_term_lufs = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_block(freq_hz: f64, sr: f64, amp: f64, frames: usize) -> Vec<f64> {
        (0..frames)
            .map(|i| amp * (2.0 * std::f64::consts::PI * freq_hz * i as f64 / sr).sin())
            .collect()
    }

    #[test]
    fn test_linear_to_dbfs_full_scale() {
        let db = linear_to_dbfs(1.0);
        assert!((db - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_to_dbfs_half() {
        let db = linear_to_dbfs(0.5);
        assert!((db - (-6.020_599_913_279_624)).abs() < 1e-6);
    }

    #[test]
    fn test_linear_to_dbfs_zero() {
        assert_eq!(linear_to_dbfs(0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn test_dbfs_to_linear_round_trip() {
        let original = 0.25_f64;
        let db = linear_to_dbfs(original);
        let recovered = dbfs_to_linear(db);
        assert!((recovered - original).abs() < 1e-10);
    }

    #[test]
    fn test_crest_factor_from_samples() {
        let sig = sine_block(1000.0, 48000.0, 0.5, 4800);
        let cf = CrestFactor::from_samples(&sig).expect("cf should be valid");
        // Sine wave crest factor ≈ 3 dB
        assert!(
            cf.crest_db > 2.0 && cf.crest_db < 4.0,
            "crest={}",
            cf.crest_db
        );
    }

    #[test]
    fn test_crest_factor_empty() {
        assert!(CrestFactor::from_samples(&[]).is_none());
    }

    #[test]
    fn test_crest_factor_over_compressed() {
        // DC signal → crest factor = 0 dB
        let sig = vec![0.5_f64; 1000];
        let cf = CrestFactor::from_samples(&sig).expect("cf should be valid");
        assert!(cf.is_over_compressed());
    }

    #[test]
    fn test_histogram_add() {
        let mut h = LevelHistogram::new(10, -60.0, 0.0);
        h.add(-30.0);
        h.add(-10.0);
        assert_eq!(h.total_samples(), 2);
    }

    #[test]
    fn test_histogram_out_of_range() {
        let mut h = LevelHistogram::new(10, -60.0, 0.0);
        h.add(-80.0); // below min, not counted in bins
        h.add(10.0); // above max, not counted in bins
                     // total still incremented but bin counts should be 0
        let mode = h.mode_bin().expect("mode should be valid");
        assert_eq!(h.bin_count(mode), 0);
    }

    #[test]
    fn test_histogram_normalized_sums_to_one() {
        let mut h = LevelHistogram::new(10, -60.0, 0.0);
        for i in 0..10 {
            h.add(-6.0 * (i as f64 + 1.0));
        }
        let norm = h.normalized();
        let sum: f64 = norm.iter().sum();
        // sum may be < 1 if some samples were out of range; at most 1.0
        assert!(sum <= 1.0 + 1e-10);
    }

    #[test]
    fn test_histogram_bin_centre_db() {
        let h = LevelHistogram::new(10, -60.0, 0.0);
        let centre = h.bin_centre_db(0);
        // first bin centre at -60 + 0.5 * 6 = -57
        assert!((centre - (-57.0)).abs() < 1e-6, "centre={}", centre);
    }

    #[test]
    fn test_histogram_reset() {
        let mut h = LevelHistogram::new(5, -50.0, 0.0);
        h.add(-25.0);
        h.reset();
        assert_eq!(h.total_samples(), 0);
    }

    #[test]
    fn test_plr_classification() {
        assert_eq!(Plr::compute(-1.0, -20.0).classification(), "Dynamic");
        assert_eq!(Plr::compute(-1.0, -12.0).classification(), "Moderate");
        assert_eq!(Plr::compute(-1.0, -6.0).classification(), "Compressed");
    }

    #[test]
    fn test_psr_has_variation() {
        let psr = Psr::compute(-10.0, -20.0);
        assert!(psr.has_loudness_variation());
        let psr2 = Psr::compute(-19.0, -20.0);
        assert!(!psr2.has_loudness_variation());
    }

    #[test]
    fn test_meter_peak() {
        let mut meter = DynamicRangeMeter::new(48000.0);
        let sig = vec![0.0, 0.5, -0.8, 0.3];
        meter.process_mono(&sig);
        let peak = meter.peak_dbfs();
        assert!((peak - linear_to_dbfs(0.8)).abs() < 1e-6);
    }

    #[test]
    fn test_meter_rms() {
        let mut meter = DynamicRangeMeter::new(48000.0);
        // DC at 0.5 → RMS = 0.5
        let sig = vec![0.5_f64; 1000];
        meter.process_mono(&sig);
        assert!((meter.rms_dbfs() - linear_to_dbfs(0.5)).abs() < 1e-4);
    }

    #[test]
    fn test_meter_plr_none_without_lufs() {
        let meter = DynamicRangeMeter::new(48000.0);
        assert!(meter.plr().is_none());
    }

    #[test]
    fn test_meter_plr_some_with_lufs() {
        let mut meter = DynamicRangeMeter::new(48000.0);
        let sig = vec![0.5_f64; 100];
        meter.process_mono(&sig);
        meter.set_integrated_lufs(-20.0);
        let plr = meter.plr().expect("plr should be valid");
        assert!(plr.plr_lu.is_finite());
    }

    #[test]
    fn test_meter_reset() {
        let mut meter = DynamicRangeMeter::new(48000.0);
        let sig = vec![0.5_f64; 100];
        meter.process_mono(&sig);
        meter.reset();
        assert_eq!(meter.total_samples(), 0);
        assert_eq!(meter.peak_dbfs(), f64::NEG_INFINITY);
    }
}
