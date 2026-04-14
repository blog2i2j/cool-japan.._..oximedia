//! Network stream QC — validate live RTMP/SRT/HLS streams in real-time.
//!
//! This module provides QC analysis for live and on-demand network streams.
//! It supports validation of stream health metrics: bitrate stability, packet
//! loss, jitter, segment availability (HLS), and stream continuity.
//!
//! # Architecture
//!
//! The `StreamQcAnalyzer` collects `StreamSample` observations over a
//! sliding window and produces a `StreamQcReport` containing findings with
//! severity levels. Callers drive the polling loop or hook into a notification
//! system; the analyzer itself is stateless per call.

#![allow(dead_code)]

use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// Stream protocol type
// ─────────────────────────────────────────────────────────────────────────────

/// Network streaming protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamProtocol {
    /// Real-Time Messaging Protocol.
    Rtmp,
    /// Secure Reliable Transport.
    Srt,
    /// HTTP Live Streaming (Apple HLS).
    Hls,
    /// MPEG-DASH (Dynamic Adaptive Streaming over HTTP).
    Dash,
    /// WebRTC real-time communication.
    WebRtc,
    /// Raw UDP/RTP stream.
    Rtp,
}

impl std::fmt::Display for StreamProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rtmp => write!(f, "RTMP"),
            Self::Srt => write!(f, "SRT"),
            Self::Hls => write!(f, "HLS"),
            Self::Dash => write!(f, "DASH"),
            Self::WebRtc => write!(f, "WebRTC"),
            Self::Rtp => write!(f, "RTP"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stream sample
// ─────────────────────────────────────────────────────────────────────────────

/// A single measurement sample collected from a live stream.
#[derive(Debug, Clone)]
pub struct StreamSample {
    /// Monotonic timestamp of the sample (seconds since stream start).
    pub timestamp_secs: f64,
    /// Received bitrate during this sample interval in kbps.
    pub bitrate_kbps: f64,
    /// Number of lost packets in this interval (if reported by the protocol).
    pub packets_lost: u32,
    /// Total packets received in this interval.
    pub packets_received: u32,
    /// Round-trip time in milliseconds (SRT/WebRTC/RTP).
    pub rtt_ms: Option<f64>,
    /// Jitter in milliseconds (SRT/WebRTC/RTP).
    pub jitter_ms: Option<f64>,
    /// Whether the stream segment/keyframe arrived on time (HLS/DASH).
    pub segment_on_time: Option<bool>,
    /// Audio/video sync offset measured at this point (ms, positive = audio leads).
    pub av_offset_ms: Option<f64>,
}

impl StreamSample {
    /// Creates a minimal stream sample.
    #[must_use]
    pub fn new(timestamp_secs: f64, bitrate_kbps: f64) -> Self {
        Self {
            timestamp_secs,
            bitrate_kbps,
            packets_lost: 0,
            packets_received: 0,
            rtt_ms: None,
            jitter_ms: None,
            segment_on_time: None,
            av_offset_ms: None,
        }
    }

    /// Adds packet loss information.
    #[must_use]
    pub fn with_packet_stats(mut self, lost: u32, received: u32) -> Self {
        self.packets_lost = lost;
        self.packets_received = received;
        self
    }

    /// Adds RTT and jitter measurements.
    #[must_use]
    pub fn with_network_stats(mut self, rtt_ms: f64, jitter_ms: f64) -> Self {
        self.rtt_ms = Some(rtt_ms);
        self.jitter_ms = Some(jitter_ms);
        self
    }

    /// Returns the packet loss rate as a fraction [0, 1].
    #[must_use]
    pub fn packet_loss_rate(&self) -> f64 {
        let total = self.packets_lost + self.packets_received;
        if total == 0 {
            return 0.0;
        }
        f64::from(self.packets_lost) / f64::from(total)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QC configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for stream QC analysis.
#[derive(Debug, Clone)]
pub struct StreamQcConfig {
    /// Streaming protocol (informational, affects threshold defaults).
    pub protocol: StreamProtocol,
    /// Expected average bitrate in kbps (0 = not enforced).
    pub expected_bitrate_kbps: f64,
    /// Maximum allowed bitrate deviation (fraction: 0.20 = ±20 %).
    pub bitrate_tolerance: f64,
    /// Maximum acceptable packet loss rate [0, 1].
    pub max_packet_loss_rate: f64,
    /// Maximum acceptable jitter in milliseconds.
    pub max_jitter_ms: f64,
    /// Maximum acceptable RTT in milliseconds.
    pub max_rtt_ms: f64,
    /// Maximum acceptable A/V sync offset in milliseconds.
    pub max_av_offset_ms: f64,
    /// Minimum fraction of segments that must arrive on time (HLS/DASH).
    pub min_on_time_fraction: f64,
    /// Window size (number of samples) for rolling analysis.
    pub window_size: usize,
}

impl StreamQcConfig {
    /// Default configuration for an RTMP live stream.
    #[must_use]
    pub fn rtmp_default() -> Self {
        Self {
            protocol: StreamProtocol::Rtmp,
            expected_bitrate_kbps: 4000.0,
            bitrate_tolerance: 0.20,
            max_packet_loss_rate: 0.01,
            max_jitter_ms: 30.0,
            max_rtt_ms: 200.0,
            max_av_offset_ms: 45.0,
            min_on_time_fraction: 0.95,
            window_size: 30,
        }
    }

    /// Default configuration for an SRT live stream.
    #[must_use]
    pub fn srt_default() -> Self {
        Self {
            protocol: StreamProtocol::Srt,
            expected_bitrate_kbps: 6000.0,
            bitrate_tolerance: 0.15,
            max_packet_loss_rate: 0.005,
            max_jitter_ms: 20.0,
            max_rtt_ms: 150.0,
            max_av_offset_ms: 40.0,
            min_on_time_fraction: 0.98,
            window_size: 30,
        }
    }

    /// Default configuration for an HLS on-demand or live stream.
    #[must_use]
    pub fn hls_default() -> Self {
        Self {
            protocol: StreamProtocol::Hls,
            expected_bitrate_kbps: 0.0, // varies per rendition
            bitrate_tolerance: 0.25,
            max_packet_loss_rate: 0.0,
            max_jitter_ms: 0.0,
            max_rtt_ms: 500.0,
            max_av_offset_ms: 45.0,
            min_on_time_fraction: 0.95,
            window_size: 10,
        }
    }
}

impl Default for StreamQcConfig {
    fn default() -> Self {
        Self::rtmp_default()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Findings
// ─────────────────────────────────────────────────────────────────────────────

/// Severity level for stream QC findings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StreamFindingSeverity {
    /// Informational.
    Info,
    /// Warning — degraded but acceptable.
    Warning,
    /// Error — outside specification.
    Error,
    /// Critical — stream is unusable.
    Critical,
}

impl std::fmt::Display for StreamFindingSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// A single stream QC finding.
#[derive(Debug, Clone)]
pub struct StreamFinding {
    /// Severity of the finding.
    pub severity: StreamFindingSeverity,
    /// Short rule code.
    pub code: &'static str,
    /// Human-readable message.
    pub message: String,
    /// Optional timestamp (seconds from stream start) where the issue occurred.
    pub at_secs: Option<f64>,
    /// Optional measured value associated with the finding.
    pub measured_value: Option<f64>,
}

impl StreamFinding {
    /// Creates a new stream finding.
    #[must_use]
    pub fn new(
        severity: StreamFindingSeverity,
        code: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            code,
            message: message.into(),
            at_secs: None,
            measured_value: None,
        }
    }

    /// Attaches a timestamp to the finding.
    #[must_use]
    pub fn at(mut self, secs: f64) -> Self {
        self.at_secs = Some(secs);
        self
    }

    /// Attaches a measured value to the finding.
    #[must_use]
    pub fn with_value(mut self, v: f64) -> Self {
        self.measured_value = Some(v);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stream QC report
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated stream QC report produced by [`StreamQcAnalyzer::analyze`].
#[derive(Debug, Clone, Default)]
pub struct StreamQcReport {
    /// Protocol being monitored.
    pub protocol: Option<StreamProtocol>,
    /// Total number of samples analyzed.
    pub sample_count: usize,
    /// Mean bitrate across all samples (kbps).
    pub mean_bitrate_kbps: f64,
    /// Peak bitrate observed (kbps).
    pub peak_bitrate_kbps: f64,
    /// Minimum bitrate observed (kbps).
    pub min_bitrate_kbps: f64,
    /// Mean packet loss rate.
    pub mean_packet_loss_rate: f64,
    /// Peak jitter observed (ms).
    pub peak_jitter_ms: f64,
    /// Peak RTT observed (ms).
    pub peak_rtt_ms: f64,
    /// All findings from the analysis.
    pub findings: Vec<StreamFinding>,
    /// Whether the stream passed all checks.
    pub passed: bool,
}

impl StreamQcReport {
    /// Creates an empty report.
    #[must_use]
    pub fn new() -> Self {
        Self {
            passed: true,
            ..Default::default()
        }
    }

    /// Returns all findings at or above the given severity.
    #[must_use]
    pub fn findings_at_least(&self, severity: StreamFindingSeverity) -> Vec<&StreamFinding> {
        self.findings
            .iter()
            .filter(|f| f.severity >= severity)
            .collect()
    }

    /// Returns true if any error or critical finding exists.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.findings
            .iter()
            .any(|f| f.severity >= StreamFindingSeverity::Error)
    }

    fn add_finding(&mut self, f: StreamFinding) {
        if f.severity >= StreamFindingSeverity::Error {
            self.passed = false;
        }
        self.findings.push(f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Analyzer
// ─────────────────────────────────────────────────────────────────────────────

/// Analyzes a window of stream samples and produces a `StreamQcReport`.
#[derive(Debug, Clone)]
pub struct StreamQcAnalyzer {
    config: StreamQcConfig,
    /// Rolling sample window.
    window: VecDeque<StreamSample>,
}

impl StreamQcAnalyzer {
    /// Creates a new analyzer with the given configuration.
    #[must_use]
    pub fn new(config: StreamQcConfig) -> Self {
        let cap = config.window_size.max(1);
        Self {
            config,
            window: VecDeque::with_capacity(cap),
        }
    }

    /// Pushes a new sample into the rolling window, evicting the oldest if needed.
    pub fn push_sample(&mut self, sample: StreamSample) {
        if self.window.len() >= self.config.window_size {
            self.window.pop_front();
        }
        self.window.push_back(sample);
    }

    /// Analyzes the current window and returns a report.
    ///
    /// This is a pure function over the accumulated samples — it does not
    /// modify the window.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&self) -> StreamQcReport {
        let mut report = StreamQcReport::new();
        report.protocol = Some(self.config.protocol);
        report.sample_count = self.window.len();

        if self.window.is_empty() {
            report.add_finding(StreamFinding::new(
                StreamFindingSeverity::Info,
                "STRM-000",
                "No stream samples available for analysis",
            ));
            return report;
        }

        // ── Bitrate statistics ──────────────────────────────────────────────
        let mut sum_br = 0.0_f64;
        let mut max_br = f64::MIN;
        let mut min_br = f64::MAX;

        for s in &self.window {
            sum_br += s.bitrate_kbps;
            if s.bitrate_kbps > max_br {
                max_br = s.bitrate_kbps;
            }
            if s.bitrate_kbps < min_br {
                min_br = s.bitrate_kbps;
            }
        }

        let mean_br = sum_br / self.window.len() as f64;
        report.mean_bitrate_kbps = mean_br;
        report.peak_bitrate_kbps = max_br;
        report.min_bitrate_kbps = min_br;

        if self.config.expected_bitrate_kbps > 0.0 {
            let expected = self.config.expected_bitrate_kbps;
            let tolerance = self.config.bitrate_tolerance;
            let lower = expected * (1.0 - tolerance);
            let upper = expected * (1.0 + tolerance);

            if mean_br < lower {
                report.add_finding(
                    StreamFinding::new(
                        StreamFindingSeverity::Error,
                        "STRM-010",
                        format!(
                            "Mean bitrate {mean_br:.0} kbps is below expected lower bound {lower:.0} kbps"
                        ),
                    )
                    .with_value(mean_br),
                );
            } else if mean_br > upper {
                report.add_finding(
                    StreamFinding::new(
                        StreamFindingSeverity::Warning,
                        "STRM-011",
                        format!(
                            "Mean bitrate {mean_br:.0} kbps exceeds expected upper bound {upper:.0} kbps"
                        ),
                    )
                    .with_value(mean_br),
                );
            }
        }

        // ── Packet loss ─────────────────────────────────────────────────────
        let mut total_lost = 0u64;
        let mut total_received = 0u64;
        for s in &self.window {
            total_lost += u64::from(s.packets_lost);
            total_received += u64::from(s.packets_received);
        }
        let total_packets = total_lost + total_received;
        if total_packets > 0 {
            let loss_rate = total_lost as f64 / total_packets as f64;
            report.mean_packet_loss_rate = loss_rate;

            if loss_rate > self.config.max_packet_loss_rate {
                let severity = if loss_rate > self.config.max_packet_loss_rate * 5.0 {
                    StreamFindingSeverity::Critical
                } else {
                    StreamFindingSeverity::Error
                };
                report.add_finding(
                    StreamFinding::new(
                        severity,
                        "STRM-020",
                        format!(
                            "Packet loss rate {:.2}% exceeds threshold {:.2}%",
                            loss_rate * 100.0,
                            self.config.max_packet_loss_rate * 100.0
                        ),
                    )
                    .with_value(loss_rate * 100.0),
                );
            }
        }

        // ── Jitter ──────────────────────────────────────────────────────────
        let mut peak_jitter = 0.0_f64;
        for s in &self.window {
            if let Some(j) = s.jitter_ms {
                if j > peak_jitter {
                    peak_jitter = j;
                }
            }
        }
        report.peak_jitter_ms = peak_jitter;

        if self.config.max_jitter_ms > 0.0 && peak_jitter > self.config.max_jitter_ms {
            report.add_finding(
                StreamFinding::new(
                    StreamFindingSeverity::Warning,
                    "STRM-030",
                    format!(
                        "Peak jitter {peak_jitter:.1} ms exceeds threshold {:.1} ms",
                        self.config.max_jitter_ms
                    ),
                )
                .with_value(peak_jitter),
            );
        }

        // ── RTT ─────────────────────────────────────────────────────────────
        let mut peak_rtt = 0.0_f64;
        for s in &self.window {
            if let Some(r) = s.rtt_ms {
                if r > peak_rtt {
                    peak_rtt = r;
                }
            }
        }
        report.peak_rtt_ms = peak_rtt;

        if self.config.max_rtt_ms > 0.0 && peak_rtt > self.config.max_rtt_ms {
            report.add_finding(
                StreamFinding::new(
                    StreamFindingSeverity::Warning,
                    "STRM-031",
                    format!(
                        "Peak RTT {peak_rtt:.1} ms exceeds threshold {:.1} ms",
                        self.config.max_rtt_ms
                    ),
                )
                .with_value(peak_rtt),
            );
        }

        // ── A/V sync ────────────────────────────────────────────────────────
        for s in &self.window {
            if let Some(offset) = s.av_offset_ms {
                if offset.abs() > self.config.max_av_offset_ms {
                    report.add_finding(
                        StreamFinding::new(
                            StreamFindingSeverity::Error,
                            "STRM-040",
                            format!(
                                "A/V sync offset {offset:.1} ms exceeds threshold {:.1} ms at {:.2}s",
                                self.config.max_av_offset_ms, s.timestamp_secs
                            ),
                        )
                        .at(s.timestamp_secs)
                        .with_value(offset.abs()),
                    );
                }
            }
        }

        // ── HLS/DASH segment timeliness ─────────────────────────────────────
        let on_time_samples: Vec<bool> = self
            .window
            .iter()
            .filter_map(|s| s.segment_on_time)
            .collect();

        if !on_time_samples.is_empty() {
            let on_time_count = on_time_samples.iter().filter(|&&v| v).count();
            let fraction = on_time_count as f64 / on_time_samples.len() as f64;
            if fraction < self.config.min_on_time_fraction {
                report.add_finding(
                    StreamFinding::new(
                        StreamFindingSeverity::Error,
                        "STRM-050",
                        format!(
                            "Only {:.1}% of segments arrived on time (threshold: {:.1}%)",
                            fraction * 100.0,
                            self.config.min_on_time_fraction * 100.0
                        ),
                    )
                    .with_value(fraction * 100.0),
                );
            }
        }

        report
    }

    /// Returns the number of samples currently in the window.
    #[must_use]
    pub fn window_len(&self) -> usize {
        self.window.len()
    }

    /// Clears all samples from the window.
    pub fn clear(&mut self) {
        self.window.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stable_samples(count: usize, bitrate_kbps: f64) -> Vec<StreamSample> {
        (0..count)
            .map(|i| StreamSample::new(i as f64, bitrate_kbps))
            .collect()
    }

    #[test]
    fn test_clean_stream_passes() {
        let config = StreamQcConfig {
            expected_bitrate_kbps: 4000.0,
            ..StreamQcConfig::rtmp_default()
        };
        let mut analyzer = StreamQcAnalyzer::new(config);
        for s in make_stable_samples(10, 4000.0) {
            analyzer.push_sample(s);
        }
        let report = analyzer.analyze();
        assert!(
            report.passed,
            "Clean stream should pass: {:?}",
            report.findings
        );
    }

    #[test]
    fn test_low_bitrate_triggers_error() {
        let config = StreamQcConfig {
            expected_bitrate_kbps: 4000.0,
            bitrate_tolerance: 0.10,
            ..StreamQcConfig::rtmp_default()
        };
        let mut analyzer = StreamQcAnalyzer::new(config);
        for s in make_stable_samples(10, 1000.0) {
            analyzer.push_sample(s);
        }
        let report = analyzer.analyze();
        assert!(!report.passed);
        assert!(report.findings.iter().any(|f| f.code == "STRM-010"));
    }

    #[test]
    fn test_packet_loss_error() {
        let config = StreamQcConfig {
            max_packet_loss_rate: 0.01,
            ..StreamQcConfig::rtmp_default()
        };
        let mut analyzer = StreamQcAnalyzer::new(config);
        let sample = StreamSample::new(0.0, 4000.0).with_packet_stats(100, 900); // 10% loss
        analyzer.push_sample(sample);
        let report = analyzer.analyze();
        assert!(report.has_errors());
    }

    #[test]
    fn test_av_sync_error() {
        let config = StreamQcConfig {
            max_av_offset_ms: 45.0,
            ..StreamQcConfig::rtmp_default()
        };
        let mut analyzer = StreamQcAnalyzer::new(config);
        let mut sample = StreamSample::new(5.0, 4000.0);
        sample.av_offset_ms = Some(200.0);
        analyzer.push_sample(sample);
        let report = analyzer.analyze();
        assert!(report.findings.iter().any(|f| f.code == "STRM-040"));
    }

    #[test]
    fn test_window_eviction() {
        let mut config = StreamQcConfig::rtmp_default();
        config.window_size = 3;
        let mut analyzer = StreamQcAnalyzer::new(config);
        for i in 0..5u64 {
            analyzer.push_sample(StreamSample::new(i as f64, 4000.0));
        }
        assert_eq!(analyzer.window_len(), 3);
    }

    #[test]
    fn test_empty_window_returns_info() {
        let analyzer = StreamQcAnalyzer::new(StreamQcConfig::default());
        let report = analyzer.analyze();
        assert!(report.findings.iter().any(|f| f.code == "STRM-000"));
    }

    #[test]
    fn test_hls_segment_timeliness() {
        let config = StreamQcConfig {
            min_on_time_fraction: 0.95,
            ..StreamQcConfig::hls_default()
        };
        let mut analyzer = StreamQcAnalyzer::new(config);
        // 7 on time, 3 late → 70% on time
        for i in 0..10usize {
            let mut s = StreamSample::new(i as f64 * 2.0, 2000.0);
            s.segment_on_time = Some(i < 7);
            analyzer.push_sample(s);
        }
        let report = analyzer.analyze();
        assert!(report.findings.iter().any(|f| f.code == "STRM-050"));
    }

    #[test]
    fn test_packet_loss_rate() {
        let s = StreamSample::new(0.0, 1000.0).with_packet_stats(10, 90);
        assert!((s.packet_loss_rate() - 0.1).abs() < 1e-9);
    }
}
