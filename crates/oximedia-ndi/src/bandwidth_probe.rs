//! Bandwidth probing for adaptive quality selection in NDI streams.
//!
//! This module provides tools for measuring available network bandwidth and
//! recommending appropriate video quality levels based on observed throughput,
//! round-trip time (RTT), and packet loss.
//!
//! The [`BandwidthProber`] accumulates measurements and applies an Exponentially
//! Weighted Moving Average (EMA) over the configured history window, then maps
//! the smoothed estimate to a [`QualityLevel`] while accounting for a safety
//! margin to absorb short-term fluctuations.

#![allow(dead_code)]

/// Probing configuration for the [`BandwidthProber`].
#[derive(Debug, Clone)]
pub struct ProbeConfig {
    /// Maximum number of recent measurements kept in the history ring buffer.
    pub window_size: usize,
    /// EMA smoothing factor α ∈ (0, 1].  Higher values react faster to changes.
    pub ema_alpha: f32,
    /// Fraction of the EMA estimate used as the effective bandwidth when
    /// selecting a quality level.  E.g. `0.85` means 85 % of measured bps.
    pub safety_margin: f32,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            window_size: 20,
            ema_alpha: 0.2,
            safety_margin: 0.85,
        }
    }
}

/// A single bandwidth probe result snapshot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbeResult {
    /// Measured throughput in bits per second.
    pub measured_bps: u64,
    /// Round-trip time in microseconds.
    pub rtt_us: u64,
    /// Fraction of packets lost ∈ [0, 1].
    pub packet_loss_rate: f32,
    /// Unix timestamp (seconds) when this measurement was taken.
    pub timestamp_secs: u64,
}

/// Discrete quality tiers for adaptive NDI streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualityLevel {
    /// 360p — very low bandwidth conditions.
    UltraLow,
    /// 540p — low bandwidth conditions.
    Low,
    /// 720p — medium bandwidth conditions.
    Medium,
    /// 1080p — high bandwidth conditions.
    High,
    /// 4K UHD — ultra-high bandwidth conditions.
    Ultra,
}

impl QualityLevel {
    /// Minimum target bitrate in bits per second for this quality level.
    ///
    /// These thresholds are conservative NDI estimates that account for the
    /// low-latency, uncompressed-friendly nature of the protocol.
    pub fn target_bps(&self) -> u64 {
        match self {
            Self::UltraLow => 2_000_000,   //   2 Mbps  (360p)
            Self::Low => 8_000_000,         //   8 Mbps  (540p)
            Self::Medium => 20_000_000,     //  20 Mbps  (720p)
            Self::High => 50_000_000,       //  50 Mbps  (1080p)
            Self::Ultra => 150_000_000,     // 150 Mbps  (4K)
        }
    }

    /// Human-readable label for this quality level.
    pub fn label(&self) -> &'static str {
        match self {
            Self::UltraLow => "360p",
            Self::Low => "540p",
            Self::Medium => "720p",
            Self::High => "1080p",
            Self::Ultra => "4K",
        }
    }

    /// Select the highest quality level whose [`target_bps`] does not exceed
    /// `available_bps`.  Returns [`QualityLevel::UltraLow`] when bandwidth is
    /// below the minimum threshold.
    pub fn from_bps(available_bps: u64) -> Self {
        if available_bps >= Self::Ultra.target_bps() {
            Self::Ultra
        } else if available_bps >= Self::High.target_bps() {
            Self::High
        } else if available_bps >= Self::Medium.target_bps() {
            Self::Medium
        } else if available_bps >= Self::Low.target_bps() {
            Self::Low
        } else {
            Self::UltraLow
        }
    }
}

/// Adaptive bandwidth prober that accumulates network measurements and
/// recommends a [`QualityLevel`] using an EMA-smoothed estimate.
#[derive(Debug, Clone)]
pub struct BandwidthProber {
    config: ProbeConfig,
    history: Vec<ProbeResult>,
    /// Current EMA estimate in bits per second.
    ema_bps: f64,
    /// Whether at least one measurement has been recorded.
    initialized: bool,
}

impl BandwidthProber {
    /// Create a new prober with the given configuration.
    pub fn new(config: ProbeConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            ema_bps: 0.0,
            initialized: false,
        }
    }

    /// Add a new measurement from a completed probe cycle.
    ///
    /// # Arguments
    ///
    /// * `bytes_sent`      — Number of bytes transmitted during the probe.
    /// * `duration_us`     — Duration of the probe interval in microseconds.
    /// * `losses`          — Number of packets confirmed lost.
    /// * `total_packets`   — Total number of packets sent in this probe.
    pub fn add_measurement(
        &mut self,
        bytes_sent: u64,
        duration_us: u64,
        losses: u32,
        total_packets: u32,
    ) {
        if duration_us == 0 {
            return;
        }

        let measured_bps = (bytes_sent.saturating_mul(8)).saturating_mul(1_000_000)
            / duration_us;

        let packet_loss_rate = if total_packets > 0 {
            (losses as f32) / (total_packets as f32)
        } else {
            0.0_f32
        };

        // Approximate RTT: we don't have it directly here, so store 0.
        // Callers can supply richer data via add_probe_result.
        let result = ProbeResult {
            measured_bps,
            rtt_us: 0,
            packet_loss_rate,
            timestamp_secs: current_unix_secs(),
        };

        self.push_result(result, measured_bps);
    }

    /// Add a fully populated [`ProbeResult`] directly.
    ///
    /// This method gives callers full control over all measurement fields,
    /// which is useful when RTT is obtained from a separate ping mechanism.
    pub fn add_probe_result(&mut self, result: ProbeResult) {
        let measured_bps = result.measured_bps;
        self.push_result(result, measured_bps);
    }

    fn push_result(&mut self, result: ProbeResult, measured_bps: u64) {
        // Update EMA
        let sample = measured_bps as f64;
        if self.initialized {
            let alpha = self.config.ema_alpha as f64;
            self.ema_bps = alpha * sample + (1.0 - alpha) * self.ema_bps;
        } else {
            self.ema_bps = sample;
            self.initialized = true;
        }

        // Maintain ring-buffer-style history
        self.history.push(result);
        if self.history.len() > self.config.window_size {
            self.history.remove(0);
        }
    }

    /// Return the current EMA-smoothed bandwidth estimate in bits per second.
    ///
    /// Returns `0` if no measurements have been added yet.
    pub fn estimate_bps(&self) -> u64 {
        if self.initialized {
            self.ema_bps as u64
        } else {
            0
        }
    }

    /// Recommend a [`QualityLevel`] based on the current bandwidth estimate
    /// after applying the configured safety margin.
    ///
    /// The effective available bandwidth is `estimate_bps * safety_margin`,
    /// which provides headroom for short-term fluctuations and protocol
    /// overhead.
    pub fn recommend_quality(&self) -> QualityLevel {
        let effective_bps =
            (self.estimate_bps() as f64 * self.config.safety_margin as f64) as u64;
        QualityLevel::from_bps(effective_bps)
    }

    /// Read-only view of the measurement history (most recent at the end).
    pub fn history(&self) -> &[ProbeResult] {
        &self.history
    }

    /// Returns the probe configuration.
    pub fn config(&self) -> &ProbeConfig {
        &self.config
    }
}

impl Default for BandwidthProber {
    fn default() -> Self {
        Self::new(ProbeConfig::default())
    }
}

/// Best-effort current Unix time in seconds.
///
/// Falls back to zero on platforms where `SystemTime` is unavailable.
fn current_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Convert Mbps to bytes/sec.
    fn mbps_to_bytes(mbps: u64) -> u64 {
        mbps * 125_000
    }

    /// Simulate a 1-second probe at `mbps` Mbps with no losses.
    fn probe(prober: &mut BandwidthProber, mbps: u64) {
        prober.add_measurement(mbps_to_bytes(mbps), 1_000_000, 0, 100);
    }

    // ── QualityLevel ─────────────────────────────────────────────────────────

    #[test]
    fn quality_level_ordering() {
        assert!(QualityLevel::UltraLow < QualityLevel::Low);
        assert!(QualityLevel::Low < QualityLevel::Medium);
        assert!(QualityLevel::Medium < QualityLevel::High);
        assert!(QualityLevel::High < QualityLevel::Ultra);
    }

    #[test]
    fn quality_level_target_bps_ascending() {
        let levels = [
            QualityLevel::UltraLow,
            QualityLevel::Low,
            QualityLevel::Medium,
            QualityLevel::High,
            QualityLevel::Ultra,
        ];
        for w in levels.windows(2) {
            assert!(w[0].target_bps() < w[1].target_bps());
        }
    }

    #[test]
    fn quality_level_labels_non_empty() {
        for level in [
            QualityLevel::UltraLow,
            QualityLevel::Low,
            QualityLevel::Medium,
            QualityLevel::High,
            QualityLevel::Ultra,
        ] {
            assert!(!level.label().is_empty());
        }
    }

    #[test]
    fn quality_level_from_bps_zero_is_ultra_low() {
        assert_eq!(QualityLevel::from_bps(0), QualityLevel::UltraLow);
    }

    #[test]
    fn quality_level_from_bps_exact_thresholds() {
        assert_eq!(
            QualityLevel::from_bps(QualityLevel::Ultra.target_bps()),
            QualityLevel::Ultra
        );
        assert_eq!(
            QualityLevel::from_bps(QualityLevel::High.target_bps()),
            QualityLevel::High
        );
        assert_eq!(
            QualityLevel::from_bps(QualityLevel::Medium.target_bps()),
            QualityLevel::Medium
        );
    }

    // ── BandwidthProber basic ─────────────────────────────────────────────────

    #[test]
    fn prober_empty_returns_zero() {
        let p = BandwidthProber::default();
        assert_eq!(p.estimate_bps(), 0);
        assert!(p.history().is_empty());
    }

    #[test]
    fn prober_single_measurement_sets_estimate() {
        let mut p = BandwidthProber::default();
        // 100 Mbps probe
        probe(&mut p, 100);
        assert!(p.estimate_bps() > 0);
    }

    // ── Low bandwidth → UltraLow ──────────────────────────────────────────────

    #[test]
    fn low_bandwidth_recommends_ultra_low() {
        let mut p = BandwidthProber::new(ProbeConfig {
            ema_alpha: 1.0, // immediate convergence
            safety_margin: 1.0,
            window_size: 10,
        });
        // 1 Mbps — below UltraLow threshold (2 Mbps)
        probe(&mut p, 1);
        assert_eq!(p.recommend_quality(), QualityLevel::UltraLow);
    }

    // ── High bandwidth → Ultra ───────────────────────────────────────────────

    #[test]
    fn high_bandwidth_recommends_ultra() {
        let mut p = BandwidthProber::new(ProbeConfig {
            ema_alpha: 1.0,
            safety_margin: 1.0,
            window_size: 10,
        });
        // 200 Mbps — above Ultra threshold (150 Mbps)
        probe(&mut p, 200);
        assert_eq!(p.recommend_quality(), QualityLevel::Ultra);
    }

    // ── EMA convergence ───────────────────────────────────────────────────────

    #[test]
    fn ema_converges_after_many_samples() {
        let target_mbps: u64 = 60; // expected to land in High range
        let mut p = BandwidthProber::new(ProbeConfig {
            ema_alpha: 0.3,
            safety_margin: 1.0,
            window_size: 50,
        });
        for _ in 0..50 {
            probe(&mut p, target_mbps);
        }
        let estimate_mbps = p.estimate_bps() / 1_000_000;
        // Should be within 5 % of target after 50 samples with α = 0.3
        assert!(
            estimate_mbps >= target_mbps * 95 / 100 && estimate_mbps <= target_mbps * 105 / 100,
            "estimate {estimate_mbps} Mbps not close to target {target_mbps} Mbps"
        );
    }

    // ── Safety margin applied ─────────────────────────────────────────────────

    #[test]
    fn safety_margin_downgrades_quality() {
        // 60 Mbps raw → High without margin.
        // With 80 % margin → 48 Mbps → Medium (< 50 Mbps threshold).
        let mut p_no_margin = BandwidthProber::new(ProbeConfig {
            ema_alpha: 1.0,
            safety_margin: 1.0,
            window_size: 10,
        });
        let mut p_margin = BandwidthProber::new(ProbeConfig {
            ema_alpha: 1.0,
            safety_margin: 0.8,
            window_size: 10,
        });
        probe(&mut p_no_margin, 60);
        probe(&mut p_margin, 60);

        let q_no = p_no_margin.recommend_quality();
        let q_yes = p_margin.recommend_quality();
        // With full margin we expect High; with 80% we expect Medium or lower.
        assert_eq!(q_no, QualityLevel::High);
        assert!(q_yes <= QualityLevel::Medium);
    }

    // ── History ring-buffer ───────────────────────────────────────────────────

    #[test]
    fn history_capped_at_window_size() {
        let window = 5_usize;
        let mut p = BandwidthProber::new(ProbeConfig {
            window_size: window,
            ..ProbeConfig::default()
        });
        for _ in 0..20 {
            probe(&mut p, 50);
        }
        assert_eq!(p.history().len(), window);
    }

    // ── Packet loss recorded ──────────────────────────────────────────────────

    #[test]
    fn packet_loss_rate_recorded() {
        let mut p = BandwidthProber::default();
        // 10 losses out of 100 packets → 10 %
        p.add_measurement(125_000_000, 1_000_000, 10, 100);
        let last = p.history().last().expect("history should have one entry");
        assert!((last.packet_loss_rate - 0.1).abs() < 1e-5);
    }

    // ── Zero-duration guard ───────────────────────────────────────────────────

    #[test]
    fn zero_duration_probe_ignored() {
        let mut p = BandwidthProber::default();
        p.add_measurement(1_000_000, 0, 0, 100);
        assert_eq!(p.estimate_bps(), 0);
        assert!(p.history().is_empty());
    }

    // ── add_probe_result ──────────────────────────────────────────────────────

    #[test]
    fn add_probe_result_direct() {
        let mut p = BandwidthProber::new(ProbeConfig {
            ema_alpha: 1.0,
            safety_margin: 1.0,
            window_size: 10,
        });
        let result = ProbeResult {
            measured_bps: 200_000_000, // 200 Mbps
            rtt_us: 1_000,
            packet_loss_rate: 0.0,
            timestamp_secs: 1_700_000_000,
        };
        p.add_probe_result(result);
        assert_eq!(p.recommend_quality(), QualityLevel::Ultra);
        assert_eq!(p.history()[0].rtt_us, 1_000);
    }

    // ========================================================================
    // New tests (8+)
    // ========================================================================

    #[test]
    fn probe_config_default_values() {
        let cfg = ProbeConfig::default();
        assert_eq!(cfg.window_size, 20);
        assert!((cfg.ema_alpha - 0.2).abs() < 1e-5);
        assert!((cfg.safety_margin - 0.85).abs() < 1e-5);
    }

    #[test]
    fn quality_level_medium_range() {
        // Exactly at Medium threshold → Medium
        assert_eq!(QualityLevel::from_bps(20_000_000), QualityLevel::Medium);
        // Just below High threshold → still Medium
        assert_eq!(QualityLevel::from_bps(49_999_999), QualityLevel::Medium);
    }

    #[test]
    fn quality_level_low_range() {
        assert_eq!(QualityLevel::from_bps(8_000_000), QualityLevel::Low);
        assert_eq!(QualityLevel::from_bps(19_999_999), QualityLevel::Low);
    }

    #[test]
    fn ema_reacts_to_bandwidth_drop() {
        let mut p = BandwidthProber::new(ProbeConfig {
            ema_alpha: 0.5,
            safety_margin: 1.0,
            window_size: 50,
        });
        // Start at 100 Mbps
        for _ in 0..10 {
            probe(&mut p, 100);
        }
        let high_estimate = p.estimate_bps();
        // Drop to 10 Mbps
        for _ in 0..10 {
            probe(&mut p, 10);
        }
        let low_estimate = p.estimate_bps();
        assert!(
            low_estimate < high_estimate,
            "EMA should decrease: was {} now {}",
            high_estimate,
            low_estimate
        );
    }

    #[test]
    fn prober_default_recommends_ultra_low() {
        let p = BandwidthProber::default();
        // No measurements → 0 bps → UltraLow
        assert_eq!(p.recommend_quality(), QualityLevel::UltraLow);
    }

    #[test]
    fn no_packet_loss_yields_zero_rate() {
        let mut p = BandwidthProber::default();
        p.add_measurement(1_000_000, 1_000_000, 0, 1000);
        let last = p.history().last();
        assert!(last.is_some());
        if let Some(result) = last {
            assert!((result.packet_loss_rate - 0.0).abs() < 1e-5);
        }
    }

    #[test]
    fn zero_total_packets_yields_zero_loss_rate() {
        let mut p = BandwidthProber::default();
        p.add_measurement(1_000_000, 1_000_000, 0, 0);
        let last = p.history().last();
        assert!(last.is_some());
        if let Some(result) = last {
            assert!((result.packet_loss_rate - 0.0).abs() < 1e-5);
        }
    }

    #[test]
    fn high_loss_rate_recorded() {
        let mut p = BandwidthProber::default();
        // 50% packet loss
        p.add_measurement(125_000_000, 1_000_000, 50, 100);
        let last = p.history().last();
        assert!(last.is_some());
        if let Some(result) = last {
            assert!((result.packet_loss_rate - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn probe_result_equality() {
        let r1 = ProbeResult {
            measured_bps: 100_000,
            rtt_us: 500,
            packet_loss_rate: 0.01,
            timestamp_secs: 1000,
        };
        let r2 = ProbeResult {
            measured_bps: 100_000,
            rtt_us: 500,
            packet_loss_rate: 0.01,
            timestamp_secs: 1000,
        };
        assert_eq!(r1, r2);
    }

    #[test]
    fn quality_level_clone_and_copy() {
        let level = QualityLevel::High;
        let copied = level;
        let cloned = level.clone();
        assert_eq!(level, copied);
        assert_eq!(level, cloned);
    }

    #[test]
    fn config_accessor_returns_configured_values() {
        let cfg = ProbeConfig {
            window_size: 42,
            ema_alpha: 0.7,
            safety_margin: 0.9,
        };
        let p = BandwidthProber::new(cfg);
        assert_eq!(p.config().window_size, 42);
        assert!((p.config().ema_alpha - 0.7).abs() < 1e-5);
        assert!((p.config().safety_margin - 0.9).abs() < 1e-5);
    }
}
