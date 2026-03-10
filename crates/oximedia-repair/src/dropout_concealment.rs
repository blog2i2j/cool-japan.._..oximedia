//! Dropout detection and concealment for video and audio streams.
//!
//! A "dropout" is a short segment of missing or completely corrupted data.
//! This module provides tools to characterize dropout patterns and apply
//! concealment strategies to minimise perceptual impact.

#![allow(dead_code)]

/// Describes a single dropout event.
#[derive(Debug, Clone, PartialEq)]
pub struct DropoutPattern {
    /// Frame index (video) or sample offset (audio) where the dropout begins.
    pub start: u64,
    /// Length of the dropout expressed as a number of frames / sample blocks.
    pub length: u64,
    /// Estimated severity: 0.0 (barely visible) .. 1.0 (fully black/silent).
    pub severity: f64,
    /// Nominal frame rate used to convert length to milliseconds.
    pub frame_rate: f64,
}

impl DropoutPattern {
    /// Create a new dropout pattern descriptor.
    pub fn new(start: u64, length: u64, severity: f64, frame_rate: f64) -> Self {
        Self {
            start,
            length,
            severity: severity.clamp(0.0, 1.0),
            frame_rate: if frame_rate <= 0.0 { 25.0 } else { frame_rate },
        }
    }

    /// Duration of the dropout in milliseconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_ms(&self) -> f64 {
        (self.length as f64 / self.frame_rate) * 1000.0
    }

    /// Returns `true` if the dropout is considered a "flash" (≤ 2 frames).
    pub fn is_flash(&self) -> bool {
        self.length <= 2
    }

    /// Returns `true` if the dropout is long (> 1 second equivalent).
    pub fn is_long(&self) -> bool {
        self.duration_ms() > 1000.0
    }
}

/// Concealment method applied to fill a dropout region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcealmentMethod {
    /// Freeze the last good frame / repeat the last good sample block.
    Freeze,
    /// Blend the last good and next good frames together.
    Blend,
    /// Interpolate pixel values from surrounding valid frames.
    Interpolate,
    /// Fill with neutral (black frame / digital silence).
    NeutralFill,
    /// Use a pre-rendered error frame / noise pattern.
    ErrorPattern,
}

impl ConcealmentMethod {
    /// Quality rating in [0, 100]; higher is perceptually better.
    pub fn quality_rating(&self) -> u8 {
        match self {
            Self::Interpolate => 90,
            Self::Blend => 75,
            Self::Freeze => 60,
            Self::ErrorPattern => 30,
            Self::NeutralFill => 10,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Freeze => "freeze",
            Self::Blend => "blend",
            Self::Interpolate => "interpolate",
            Self::NeutralFill => "neutral_fill",
            Self::ErrorPattern => "error_pattern",
        }
    }

    /// Returns `true` if this method can be applied without the next good
    /// frame (useful when the dropout extends to end-of-stream).
    pub fn requires_next_frame(&self) -> bool {
        matches!(self, Self::Blend | Self::Interpolate)
    }
}

/// Aggregated statistics from a concealment pass.
#[derive(Debug, Clone, Default)]
pub struct ConcealmentStatistics {
    /// Total dropouts encountered.
    pub dropouts_found: usize,
    /// Dropouts that were successfully concealed.
    pub dropouts_concealed: usize,
    /// Dropouts where concealment was not possible.
    pub dropouts_failed: usize,
    /// Total frames / sample blocks replaced.
    pub frames_replaced: u64,
    /// Average severity of all dropouts detected.
    pub average_severity: f64,
}

impl ConcealmentStatistics {
    /// Fraction of dropouts that were successfully concealed.
    #[allow(clippy::cast_precision_loss)]
    pub fn concealment_rate(&self) -> f64 {
        if self.dropouts_found == 0 {
            return 1.0;
        }
        self.dropouts_concealed as f64 / self.dropouts_found as f64
    }
}

/// Engine that scans a list of dropout patterns and applies concealment.
#[derive(Debug, Clone)]
pub struct DropoutConcealer {
    preferred_method: ConcealmentMethod,
    fallback_method: ConcealmentMethod,
    /// Maximum duration (ms) the concealer will attempt to conceal.
    max_conceal_ms: f64,
}

impl DropoutConcealer {
    /// Create a new concealer.
    pub fn new(preferred_method: ConcealmentMethod) -> Self {
        Self {
            preferred_method,
            fallback_method: ConcealmentMethod::Freeze,
            max_conceal_ms: 5000.0,
        }
    }

    /// Set the fallback method used when the preferred method cannot be applied.
    pub fn with_fallback(mut self, fallback: ConcealmentMethod) -> Self {
        self.fallback_method = fallback;
        self
    }

    /// Set the maximum dropout duration the engine will attempt to conceal.
    pub fn with_max_conceal_ms(mut self, ms: f64) -> Self {
        self.max_conceal_ms = ms;
        self
    }

    /// Choose the concealment method for a particular dropout.
    fn select_method(&self, dropout: &DropoutPattern, has_next_frame: bool) -> ConcealmentMethod {
        let method = if !has_next_frame && self.preferred_method.requires_next_frame() {
            self.fallback_method
        } else {
            self.preferred_method
        };
        // Downgrade for very long dropouts.
        if dropout.is_long() && method.requires_next_frame() {
            self.fallback_method
        } else {
            method
        }
    }

    /// Simulate concealment of a single frame region described by `dropout`.
    /// Returns the method applied, or `None` if the dropout was too long.
    pub fn conceal_frame(
        &self,
        dropout: &DropoutPattern,
        has_next_frame: bool,
    ) -> Option<ConcealmentMethod> {
        if dropout.duration_ms() > self.max_conceal_ms {
            return None;
        }
        Some(self.select_method(dropout, has_next_frame))
    }

    /// Process a list of dropouts and return updated statistics.
    pub fn statistics(&self, dropouts: &[DropoutPattern]) -> ConcealmentStatistics {
        let mut stats = ConcealmentStatistics {
            dropouts_found: dropouts.len(),
            ..Default::default()
        };

        let mut severity_sum = 0.0_f64;

        for dropout in dropouts {
            severity_sum += dropout.severity;
            // Assume next frame is available unless it's the last in stream.
            let has_next = true;
            if let Some(_method) = self.conceal_frame(dropout, has_next) {
                stats.dropouts_concealed += 1;
                stats.frames_replaced += dropout.length;
            } else {
                stats.dropouts_failed += 1;
            }
        }

        if !dropouts.is_empty() {
            #[allow(clippy::cast_precision_loss)]
            {
                stats.average_severity = severity_sum / dropouts.len() as f64;
            }
        }

        stats
    }
}

impl Default for DropoutConcealer {
    fn default() -> Self {
        Self::new(ConcealmentMethod::Interpolate)
    }
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dropout(start: u64, length: u64, severity: f64) -> DropoutPattern {
        DropoutPattern::new(start, length, severity, 25.0)
    }

    #[test]
    fn test_dropout_duration_ms() {
        // 25 fps, 25 frames → 1000 ms.
        let d = make_dropout(0, 25, 0.5);
        assert!((d.duration_ms() - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn test_dropout_is_flash() {
        assert!(make_dropout(0, 1, 0.9).is_flash());
        assert!(make_dropout(0, 2, 0.9).is_flash());
        assert!(!make_dropout(0, 3, 0.5).is_flash());
    }

    #[test]
    fn test_dropout_is_long() {
        // 26 frames at 25 fps = 1040 ms > 1000 ms.
        assert!(make_dropout(0, 26, 0.5).is_long());
        assert!(!make_dropout(0, 24, 0.5).is_long());
    }

    #[test]
    fn test_dropout_severity_clamped() {
        let d = DropoutPattern::new(0, 5, 2.5, 25.0);
        assert!((d.severity - 1.0).abs() < 1e-9);
        let d2 = DropoutPattern::new(0, 5, -0.5, 25.0);
        assert!((d2.severity - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_method_quality_ratings() {
        assert!(
            ConcealmentMethod::Interpolate.quality_rating()
                > ConcealmentMethod::Blend.quality_rating()
        );
        assert!(
            ConcealmentMethod::Blend.quality_rating() > ConcealmentMethod::Freeze.quality_rating()
        );
        assert!(
            ConcealmentMethod::Freeze.quality_rating()
                > ConcealmentMethod::ErrorPattern.quality_rating()
        );
        assert!(
            ConcealmentMethod::ErrorPattern.quality_rating()
                > ConcealmentMethod::NeutralFill.quality_rating()
        );
    }

    #[test]
    fn test_method_requires_next_frame() {
        assert!(ConcealmentMethod::Blend.requires_next_frame());
        assert!(ConcealmentMethod::Interpolate.requires_next_frame());
        assert!(!ConcealmentMethod::Freeze.requires_next_frame());
        assert!(!ConcealmentMethod::NeutralFill.requires_next_frame());
    }

    #[test]
    fn test_method_name() {
        assert_eq!(ConcealmentMethod::Freeze.name(), "freeze");
        assert_eq!(ConcealmentMethod::Blend.name(), "blend");
        assert_eq!(ConcealmentMethod::NeutralFill.name(), "neutral_fill");
    }

    #[test]
    fn test_conceal_frame_within_limit() {
        let concealer = DropoutConcealer::new(ConcealmentMethod::Blend);
        let dropout = make_dropout(0, 5, 0.5); // 200 ms
        let result = concealer.conceal_frame(&dropout, true);
        assert!(result.is_some());
    }

    #[test]
    fn test_conceal_frame_exceeds_limit() {
        let concealer = DropoutConcealer::new(ConcealmentMethod::Blend).with_max_conceal_ms(100.0);
        let dropout = make_dropout(0, 10, 0.5); // 400 ms > 100 ms
        assert!(concealer.conceal_frame(&dropout, true).is_none());
    }

    #[test]
    fn test_conceal_frame_fallback_no_next_frame() {
        let concealer = DropoutConcealer::new(ConcealmentMethod::Interpolate)
            .with_fallback(ConcealmentMethod::Freeze);
        let dropout = make_dropout(0, 3, 0.5);
        let method = concealer
            .conceal_frame(&dropout, false)
            .expect("unexpected None/Err");
        assert_eq!(method, ConcealmentMethod::Freeze);
    }

    #[test]
    fn test_statistics_all_concealed() {
        let concealer = DropoutConcealer::default();
        let dropouts = vec![make_dropout(0, 2, 0.3), make_dropout(50, 3, 0.5)];
        let stats = concealer.statistics(&dropouts);
        assert_eq!(stats.dropouts_found, 2);
        assert_eq!(stats.dropouts_concealed, 2);
        assert_eq!(stats.dropouts_failed, 0);
    }

    #[test]
    fn test_statistics_concealment_rate() {
        let stats = ConcealmentStatistics {
            dropouts_found: 4,
            dropouts_concealed: 3,
            dropouts_failed: 1,
            ..Default::default()
        };
        assert!((stats.concealment_rate() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_statistics_empty_dropouts() {
        let concealer = DropoutConcealer::default();
        let stats = concealer.statistics(&[]);
        assert_eq!(stats.dropouts_found, 0);
        assert!((stats.concealment_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_statistics_average_severity() {
        let concealer = DropoutConcealer::default();
        let dropouts = vec![make_dropout(0, 1, 0.4), make_dropout(10, 1, 0.6)];
        let stats = concealer.statistics(&dropouts);
        assert!((stats.average_severity - 0.5).abs() < 1e-9);
    }
}
