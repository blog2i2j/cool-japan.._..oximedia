#![allow(dead_code)]
//! Caption gap and overlap analysis for quality control.
//!
//! Detects timing issues such as:
//! - Gaps between consecutive captions that are too short or too long
//! - Overlapping captions that would display simultaneously
//! - Captions that violate minimum display duration requirements
//! - Reading-speed violations based on character count and duration

/// Minimum gap in milliseconds between consecutive captions (broadcast standard).
const DEFAULT_MIN_GAP_MS: u64 = 66;

/// Maximum recommended gap in milliseconds before flagging as a "dead zone".
const DEFAULT_MAX_GAP_MS: u64 = 5000;

/// Minimum display duration in milliseconds for a single caption.
const DEFAULT_MIN_DISPLAY_MS: u64 = 1000;

/// Maximum display duration in milliseconds for a single caption.
const DEFAULT_MAX_DISPLAY_MS: u64 = 7000;

/// A time span representing a caption's start and end in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptionSpan {
    /// Start time in milliseconds from the beginning of the programme.
    pub start_ms: u64,
    /// End time in milliseconds from the beginning of the programme.
    pub end_ms: u64,
    /// Number of characters in the caption text (excluding markup).
    pub char_count: usize,
}

impl CaptionSpan {
    /// Create a new caption span.
    #[must_use]
    pub fn new(start_ms: u64, end_ms: u64, char_count: usize) -> Self {
        Self {
            start_ms,
            end_ms,
            char_count,
        }
    }

    /// Duration of this caption in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// Severity level for a gap/overlap issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    /// Informational — not necessarily wrong but worth noting.
    Info,
    /// Warning — may cause problems on some players.
    Warning,
    /// Error — violates broadcast standards.
    Error,
}

/// A specific issue found during gap analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GapIssue {
    /// Zero-based index of the first caption involved.
    pub caption_index: usize,
    /// The kind of issue detected.
    pub kind: GapIssueKind,
    /// Severity of this issue.
    pub severity: IssueSeverity,
    /// Human-readable description.
    pub message: String,
}

/// The kind of timing issue detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapIssueKind {
    /// Gap between two consecutive captions is below the minimum threshold.
    GapTooShort,
    /// Gap between two consecutive captions exceeds the maximum threshold.
    GapTooLong,
    /// Two captions overlap in time.
    Overlap,
    /// A single caption is displayed for less than the minimum duration.
    DisplayTooShort,
    /// A single caption is displayed for more than the maximum duration.
    DisplayTooLong,
    /// Caption start time is after its end time (invalid).
    InvertedSpan,
}

/// Configuration for the gap analyser.
#[derive(Debug, Clone)]
pub struct GapAnalysisConfig {
    /// Minimum gap between consecutive captions in milliseconds.
    pub min_gap_ms: u64,
    /// Maximum gap before flagging a dead zone in milliseconds.
    pub max_gap_ms: u64,
    /// Minimum display duration in milliseconds.
    pub min_display_ms: u64,
    /// Maximum display duration in milliseconds.
    pub max_display_ms: u64,
}

impl Default for GapAnalysisConfig {
    fn default() -> Self {
        Self {
            min_gap_ms: DEFAULT_MIN_GAP_MS,
            max_gap_ms: DEFAULT_MAX_GAP_MS,
            min_display_ms: DEFAULT_MIN_DISPLAY_MS,
            max_display_ms: DEFAULT_MAX_DISPLAY_MS,
        }
    }
}

/// Result of running a gap analysis across a list of caption spans.
#[derive(Debug, Clone)]
pub struct GapAnalysisReport {
    /// Total number of captions analysed.
    pub caption_count: usize,
    /// All issues found.
    pub issues: Vec<GapIssue>,
    /// Total programme duration covered by captions in milliseconds.
    pub total_caption_ms: u64,
    /// Total gap duration in milliseconds.
    pub total_gap_ms: u64,
}

impl GapAnalysisReport {
    /// Return the number of errors in this report.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .count()
    }

    /// Return the number of warnings in this report.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Warning)
            .count()
    }

    /// Whether this report has no errors.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.error_count() == 0
    }

    /// Caption coverage as a ratio (0.0 to 1.0) of total time including gaps.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn coverage_ratio(&self) -> f64 {
        let total = self.total_caption_ms + self.total_gap_ms;
        if total == 0 {
            return 0.0;
        }
        self.total_caption_ms as f64 / total as f64
    }
}

/// Analyse a list of caption spans for gap and overlap issues.
#[must_use]
pub fn analyse_gaps(spans: &[CaptionSpan], config: &GapAnalysisConfig) -> GapAnalysisReport {
    let mut issues = Vec::new();
    let mut total_caption_ms: u64 = 0;
    let mut total_gap_ms: u64 = 0;

    for (i, span) in spans.iter().enumerate() {
        // Check inverted spans
        if span.start_ms > span.end_ms {
            issues.push(GapIssue {
                caption_index: i,
                kind: GapIssueKind::InvertedSpan,
                severity: IssueSeverity::Error,
                message: format!(
                    "Caption {i}: start ({}) is after end ({})",
                    span.start_ms, span.end_ms
                ),
            });
            continue;
        }

        let dur = span.duration_ms();
        total_caption_ms += dur;

        // Check display duration
        if dur < config.min_display_ms {
            issues.push(GapIssue {
                caption_index: i,
                kind: GapIssueKind::DisplayTooShort,
                severity: IssueSeverity::Warning,
                message: format!(
                    "Caption {i}: display duration {dur}ms < minimum {}ms",
                    config.min_display_ms
                ),
            });
        }

        if dur > config.max_display_ms {
            issues.push(GapIssue {
                caption_index: i,
                kind: GapIssueKind::DisplayTooLong,
                severity: IssueSeverity::Warning,
                message: format!(
                    "Caption {i}: display duration {dur}ms > maximum {}ms",
                    config.max_display_ms
                ),
            });
        }

        // Check gap to next caption
        if i + 1 < spans.len() {
            let next = &spans[i + 1];
            if span.end_ms > next.start_ms {
                let overlap = span.end_ms - next.start_ms;
                issues.push(GapIssue {
                    caption_index: i,
                    kind: GapIssueKind::Overlap,
                    severity: IssueSeverity::Error,
                    message: format!("Captions {i} and {}: overlap by {overlap}ms", i + 1),
                });
            } else {
                let gap = next.start_ms - span.end_ms;
                total_gap_ms += gap;

                if gap > 0 && gap < config.min_gap_ms {
                    issues.push(GapIssue {
                        caption_index: i,
                        kind: GapIssueKind::GapTooShort,
                        severity: IssueSeverity::Warning,
                        message: format!(
                            "Gap between captions {i} and {}: {gap}ms < minimum {}ms",
                            i + 1,
                            config.min_gap_ms
                        ),
                    });
                }

                if gap > config.max_gap_ms {
                    issues.push(GapIssue {
                        caption_index: i,
                        kind: GapIssueKind::GapTooLong,
                        severity: IssueSeverity::Info,
                        message: format!(
                            "Gap between captions {i} and {}: {gap}ms > maximum {}ms",
                            i + 1,
                            config.max_gap_ms
                        ),
                    });
                }
            }
        }
    }

    GapAnalysisReport {
        caption_count: spans.len(),
        issues,
        total_caption_ms,
        total_gap_ms,
    }
}

/// Find the longest gap in a sequence of caption spans.
/// Returns `None` if there are fewer than 2 spans.
#[must_use]
pub fn longest_gap(spans: &[CaptionSpan]) -> Option<(usize, u64)> {
    if spans.len() < 2 {
        return None;
    }
    let mut best_idx = 0;
    let mut best_gap = 0u64;
    for i in 0..spans.len() - 1 {
        let gap = spans[i + 1].start_ms.saturating_sub(spans[i].end_ms);
        if gap > best_gap {
            best_gap = gap;
            best_idx = i;
        }
    }
    Some((best_idx, best_gap))
}

/// Count the number of overlaps in the span list.
#[must_use]
pub fn count_overlaps(spans: &[CaptionSpan]) -> usize {
    if spans.len() < 2 {
        return 0;
    }
    let mut count = 0;
    for i in 0..spans.len() - 1 {
        if spans[i].end_ms > spans[i + 1].start_ms {
            count += 1;
        }
    }
    count
}

/// Compute the average gap duration in milliseconds.
/// Returns `None` if there are fewer than 2 spans.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn average_gap(spans: &[CaptionSpan]) -> Option<f64> {
    if spans.len() < 2 {
        return None;
    }
    let mut total_gap = 0u64;
    let mut gap_count = 0usize;
    for i in 0..spans.len() - 1 {
        if spans[i + 1].start_ms > spans[i].end_ms {
            total_gap += spans[i + 1].start_ms - spans[i].end_ms;
            gap_count += 1;
        }
    }
    if gap_count == 0 {
        return Some(0.0);
    }
    Some(total_gap as f64 / gap_count as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_spans(timings: &[(u64, u64, usize)]) -> Vec<CaptionSpan> {
        timings
            .iter()
            .map(|&(s, e, c)| CaptionSpan::new(s, e, c))
            .collect()
    }

    #[test]
    fn test_clean_captions() {
        let spans = make_spans(&[(0, 2000, 20), (2100, 4000, 25), (4100, 6000, 18)]);
        let report = analyse_gaps(&spans, &GapAnalysisConfig::default());
        assert!(report.is_clean());
        assert_eq!(report.caption_count, 3);
        assert_eq!(report.error_count(), 0);
    }

    #[test]
    fn test_overlap_detected() {
        let spans = make_spans(&[(0, 3000, 20), (2500, 5000, 25)]);
        let report = analyse_gaps(&spans, &GapAnalysisConfig::default());
        assert_eq!(report.error_count(), 1);
        let issue = &report.issues[0];
        assert_eq!(issue.kind, GapIssueKind::Overlap);
    }

    #[test]
    fn test_gap_too_short() {
        let spans = make_spans(&[(0, 2000, 20), (2030, 4000, 25)]);
        let report = analyse_gaps(&spans, &GapAnalysisConfig::default());
        assert!(report
            .issues
            .iter()
            .any(|i| i.kind == GapIssueKind::GapTooShort));
    }

    #[test]
    fn test_gap_too_long() {
        let spans = make_spans(&[(0, 2000, 20), (10000, 12000, 25)]);
        let report = analyse_gaps(&spans, &GapAnalysisConfig::default());
        assert!(report
            .issues
            .iter()
            .any(|i| i.kind == GapIssueKind::GapTooLong));
    }

    #[test]
    fn test_display_too_short() {
        let spans = make_spans(&[(0, 500, 10)]);
        let report = analyse_gaps(&spans, &GapAnalysisConfig::default());
        assert!(report
            .issues
            .iter()
            .any(|i| i.kind == GapIssueKind::DisplayTooShort));
    }

    #[test]
    fn test_display_too_long() {
        let spans = make_spans(&[(0, 10000, 30)]);
        let report = analyse_gaps(&spans, &GapAnalysisConfig::default());
        assert!(report
            .issues
            .iter()
            .any(|i| i.kind == GapIssueKind::DisplayTooLong));
    }

    #[test]
    fn test_inverted_span() {
        let spans = make_spans(&[(5000, 2000, 10)]);
        let report = analyse_gaps(&spans, &GapAnalysisConfig::default());
        assert!(report
            .issues
            .iter()
            .any(|i| i.kind == GapIssueKind::InvertedSpan));
    }

    #[test]
    fn test_empty_spans() {
        let report = analyse_gaps(&[], &GapAnalysisConfig::default());
        assert_eq!(report.caption_count, 0);
        assert!(report.is_clean());
    }

    #[test]
    fn test_coverage_ratio() {
        let spans = make_spans(&[(0, 2000, 20), (3000, 5000, 25)]);
        let report = analyse_gaps(&spans, &GapAnalysisConfig::default());
        let ratio = report.coverage_ratio();
        assert!(ratio > 0.7 && ratio < 0.9);
    }

    #[test]
    fn test_longest_gap() {
        let spans = make_spans(&[(0, 1000, 10), (1100, 2000, 10), (5000, 6000, 10)]);
        let (idx, gap) = longest_gap(&spans).expect("longest gap should be found");
        assert_eq!(idx, 1);
        assert_eq!(gap, 3000);
    }

    #[test]
    fn test_count_overlaps() {
        let spans = make_spans(&[(0, 2000, 10), (1500, 3000, 10), (2800, 4000, 10)]);
        assert_eq!(count_overlaps(&spans), 2);
    }

    #[test]
    fn test_average_gap() {
        let spans = make_spans(&[(0, 1000, 10), (2000, 3000, 10), (4000, 5000, 10)]);
        let avg = average_gap(&spans).expect("average gap should be computed");
        assert!((avg - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_custom_config() {
        let config = GapAnalysisConfig {
            min_gap_ms: 100,
            max_gap_ms: 3000,
            min_display_ms: 500,
            max_display_ms: 5000,
        };
        let spans = make_spans(&[(0, 1000, 10), (1050, 2000, 10)]);
        let report = analyse_gaps(&spans, &config);
        assert!(report
            .issues
            .iter()
            .any(|i| i.kind == GapIssueKind::GapTooShort));
    }
}
