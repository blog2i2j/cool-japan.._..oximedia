//! Caption style guide rule enforcement.
//!
//! This module validates [`CaptionBlock`] sequences against configurable style
//! guide rules covering:
//!
//! - Maximum characters per line
//! - Maximum lines per caption
//! - Reading speed limits (words per minute / characters per second) for
//!   different target audiences
//! - Minimum and maximum display duration
//! - Inter-caption gap requirements
//! - Sentence capitalisation and trailing punctuation
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_caption_gen::caption_style_guide::{StyleGuide, StyleViolation, Audience};
//! use oximedia_caption_gen::{CaptionBlock, CaptionPosition};
//!
//! let block = CaptionBlock {
//!     id: 1,
//!     start_ms: 0,
//!     end_ms: 3000,
//!     lines: vec!["Hello, world!".to_string()],
//!     speaker_id: None,
//!     position: CaptionPosition::Bottom,
//! };
//! let guide = StyleGuide::broadcast();
//! let violations = guide.check_block(&block);
//! assert!(violations.is_empty());
//! ```

use crate::alignment::CaptionBlock;

// ─── Audience ─────────────────────────────────────────────────────────────────

/// Target audience that drives reading-speed thresholds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Audience {
    /// Children (typically ≤ 8 years): slower reading speed.
    Children,
    /// General adult audience.
    Adult,
    /// Professional broadcast (EBU/BBC style guide).
    Broadcast,
    /// Custom audience with explicit CPS (characters-per-second) limit.
    Custom { max_cps: u8 },
}

impl Audience {
    /// Maximum allowed characters-per-second for this audience.
    pub fn max_cps(&self) -> f64 {
        match self {
            Audience::Children => 10.0,
            Audience::Adult => 17.0,
            Audience::Broadcast => 20.0,
            Audience::Custom { max_cps } => f64::from(*max_cps),
        }
    }
}

// ─── StyleGuide ───────────────────────────────────────────────────────────────

/// A complete caption style guide configuration.
#[derive(Debug, Clone)]
pub struct StyleGuide {
    /// Maximum characters per line (including spaces).
    pub max_chars_per_line: usize,
    /// Maximum number of lines in a single caption.
    pub max_lines: usize,
    /// Minimum display duration in milliseconds.
    pub min_duration_ms: u64,
    /// Maximum display duration in milliseconds.
    pub max_duration_ms: u64,
    /// Minimum gap between consecutive captions in milliseconds.
    pub min_gap_ms: u64,
    /// Target audience (drives reading-speed check).
    pub audience: Audience,
    /// Require each caption to start with a capital letter.
    pub require_capitalisation: bool,
    /// Require each caption to end with sentence-terminal punctuation (`.!?`).
    /// When `false`, trailing punctuation is optional.
    pub require_terminal_punctuation: bool,
    /// Maximum characters per line for the *shorter* line in a two-line block
    /// (line balance check).  `None` disables the check.
    pub max_short_line_chars: Option<usize>,
}

impl StyleGuide {
    /// Pre-set for broadcast delivery (EBU R 37 / BBC subtitle guidelines).
    pub fn broadcast() -> Self {
        Self {
            max_chars_per_line: 42,
            max_lines: 2,
            min_duration_ms: 833, // ~5 frames at 24 fps
            max_duration_ms: 7_000,
            min_gap_ms: 80,
            audience: Audience::Broadcast,
            require_capitalisation: true,
            require_terminal_punctuation: false,
            max_short_line_chars: Some(40),
        }
    }

    /// Pre-set for streaming platforms (Netflix subtitle style guide).
    pub fn streaming() -> Self {
        Self {
            max_chars_per_line: 42,
            max_lines: 2,
            min_duration_ms: 833,
            max_duration_ms: 7_000,
            min_gap_ms: 83, // ~2 frames at 24 fps per Netflix spec
            audience: Audience::Adult,
            require_capitalisation: true,
            require_terminal_punctuation: false,
            max_short_line_chars: None,
        }
    }

    /// Pre-set for children's educational content.
    pub fn children() -> Self {
        Self {
            max_chars_per_line: 32,
            max_lines: 2,
            min_duration_ms: 1_200,
            max_duration_ms: 5_500,
            min_gap_ms: 120,
            audience: Audience::Children,
            require_capitalisation: true,
            require_terminal_punctuation: true,
            max_short_line_chars: Some(28),
        }
    }

    // ─── Validation ──────────────────────────────────────────────────────────

    /// Validate a single [`CaptionBlock`] against this style guide.
    ///
    /// Returns a (possibly empty) list of [`StyleViolation`] values.
    pub fn check_block(&self, block: &CaptionBlock) -> Vec<StyleViolation> {
        let mut violations = Vec::new();

        // Line count.
        if block.lines.len() > self.max_lines {
            violations.push(StyleViolation::TooManyLines {
                block_id: block.id,
                actual: block.lines.len(),
                max: self.max_lines,
            });
        }

        // Per-line character count.
        for (line_idx, line) in block.lines.iter().enumerate() {
            let chars = line.chars().count();
            if chars > self.max_chars_per_line {
                violations.push(StyleViolation::LineTooLong {
                    block_id: block.id,
                    line_index: line_idx,
                    actual: chars,
                    max: self.max_chars_per_line,
                });
            }
        }

        // Display duration.
        let duration_ms = block.duration_ms();
        if duration_ms < self.min_duration_ms {
            violations.push(StyleViolation::DurationTooShort {
                block_id: block.id,
                actual_ms: duration_ms,
                min_ms: self.min_duration_ms,
            });
        }
        if duration_ms > self.max_duration_ms {
            violations.push(StyleViolation::DurationTooLong {
                block_id: block.id,
                actual_ms: duration_ms,
                max_ms: self.max_duration_ms,
            });
        }

        // Reading speed (CPS).
        let total_chars: usize = block.lines.iter().map(|l| l.chars().count()).sum();
        if duration_ms > 0 {
            let cps = total_chars as f64 / (duration_ms as f64 / 1000.0);
            let max_cps = self.audience.max_cps();
            if cps > max_cps {
                violations.push(StyleViolation::ReadingSpeedExceeded {
                    block_id: block.id,
                    actual_cps: cps,
                    max_cps,
                });
            }
        }

        // Capitalisation.
        if self.require_capitalisation {
            if let Some(first_line) = block.lines.first() {
                if let Some(first_char) = first_line.chars().next() {
                    if first_char.is_alphabetic() && !first_char.is_uppercase() {
                        violations
                            .push(StyleViolation::MissingCapitalisation { block_id: block.id });
                    }
                }
            }
        }

        // Terminal punctuation.
        if self.require_terminal_punctuation {
            if let Some(last_line) = block.lines.last() {
                let last_char = last_line.chars().next_back();
                match last_char {
                    Some('.' | '!' | '?') => {}
                    _ => {
                        violations.push(StyleViolation::MissingTerminalPunctuation {
                            block_id: block.id,
                        });
                    }
                }
            }
        }

        // Line balance: if there are exactly two lines, check the shorter one.
        if let Some(max_short) = self.max_short_line_chars {
            if block.lines.len() == 2 {
                let lens: Vec<usize> = block.lines.iter().map(|l| l.chars().count()).collect();
                let shorter = lens[0].min(lens[1]);
                if shorter > max_short {
                    violations.push(StyleViolation::ShortLineImbalance {
                        block_id: block.id,
                        shorter_len: shorter,
                        max_short,
                    });
                }
            }
        }

        violations
    }

    /// Validate an entire caption track (ordered slice of blocks).
    ///
    /// In addition to per-block checks, this also validates inter-caption gap
    /// requirements between consecutive blocks.
    pub fn check_track(&self, blocks: &[CaptionBlock]) -> Vec<StyleViolation> {
        let mut violations = Vec::new();

        for block in blocks {
            violations.extend(self.check_block(block));
        }

        // Gap checks between consecutive blocks.
        for pair in blocks.windows(2) {
            let a = &pair[0];
            let b = &pair[1];
            if b.start_ms > a.end_ms {
                let gap_ms = b.start_ms - a.end_ms;
                if gap_ms < self.min_gap_ms {
                    violations.push(StyleViolation::GapTooShort {
                        before_id: a.id,
                        after_id: b.id,
                        actual_ms: gap_ms,
                        min_ms: self.min_gap_ms,
                    });
                }
            } else if b.start_ms < a.end_ms {
                // Overlap.
                violations.push(StyleViolation::CaptionOverlap {
                    block_id_a: a.id,
                    block_id_b: b.id,
                });
            }
        }

        violations
    }

    /// Returns a compliance score in [0.0, 1.0] where 1.0 means fully compliant.
    ///
    /// Score = `1 - violations / (blocks * checks_per_block)`.
    pub fn compliance_score(&self, blocks: &[CaptionBlock]) -> f64 {
        if blocks.is_empty() {
            return 1.0;
        }
        let violations = self.check_track(blocks).len();
        // Estimate max possible checks: 6 per-block + (n-1) gap checks.
        let max_checks = blocks.len() * 6 + blocks.len().saturating_sub(1);
        if max_checks == 0 {
            return 1.0;
        }
        let ratio = violations as f64 / max_checks as f64;
        (1.0 - ratio).max(0.0)
    }
}

// ─── StyleViolation ───────────────────────────────────────────────────────────

/// A single style guide violation.
#[derive(Debug, Clone, PartialEq)]
pub enum StyleViolation {
    /// A line contains more characters than allowed.
    LineTooLong {
        block_id: u32,
        line_index: usize,
        actual: usize,
        max: usize,
    },
    /// The block has more lines than allowed.
    TooManyLines {
        block_id: u32,
        actual: usize,
        max: usize,
    },
    /// The block is displayed for less than the minimum duration.
    DurationTooShort {
        block_id: u32,
        actual_ms: u64,
        min_ms: u64,
    },
    /// The block is displayed for longer than the maximum duration.
    DurationTooLong {
        block_id: u32,
        actual_ms: u64,
        max_ms: u64,
    },
    /// The reading speed exceeds the audience's threshold.
    ReadingSpeedExceeded {
        block_id: u32,
        actual_cps: f64,
        max_cps: f64,
    },
    /// The first character of the caption is not capitalised.
    MissingCapitalisation { block_id: u32 },
    /// The last line does not end with sentence-terminal punctuation.
    MissingTerminalPunctuation { block_id: u32 },
    /// Two consecutive captions overlap in time.
    CaptionOverlap { block_id_a: u32, block_id_b: u32 },
    /// The gap between two consecutive captions is shorter than required.
    GapTooShort {
        before_id: u32,
        after_id: u32,
        actual_ms: u64,
        min_ms: u64,
    },
    /// In a two-line block, the shorter line exceeds the balance limit.
    ShortLineImbalance {
        block_id: u32,
        shorter_len: usize,
        max_short: usize,
    },
}

impl StyleViolation {
    /// The caption block ID associated with this violation.
    pub fn block_id(&self) -> u32 {
        match self {
            StyleViolation::LineTooLong { block_id, .. } => *block_id,
            StyleViolation::TooManyLines { block_id, .. } => *block_id,
            StyleViolation::DurationTooShort { block_id, .. } => *block_id,
            StyleViolation::DurationTooLong { block_id, .. } => *block_id,
            StyleViolation::ReadingSpeedExceeded { block_id, .. } => *block_id,
            StyleViolation::MissingCapitalisation { block_id } => *block_id,
            StyleViolation::MissingTerminalPunctuation { block_id } => *block_id,
            StyleViolation::CaptionOverlap { block_id_a, .. } => *block_id_a,
            StyleViolation::GapTooShort { before_id, .. } => *before_id,
            StyleViolation::ShortLineImbalance { block_id, .. } => *block_id,
        }
    }

    /// Human-readable description of the violation.
    pub fn description(&self) -> String {
        match self {
            StyleViolation::LineTooLong { block_id, line_index, actual, max } =>
                format!("Block {block_id} line {line_index}: {actual} chars > max {max}"),
            StyleViolation::TooManyLines { block_id, actual, max } =>
                format!("Block {block_id}: {actual} lines > max {max}"),
            StyleViolation::DurationTooShort { block_id, actual_ms, min_ms } =>
                format!("Block {block_id}: {actual_ms}ms < min {min_ms}ms"),
            StyleViolation::DurationTooLong { block_id, actual_ms, max_ms } =>
                format!("Block {block_id}: {actual_ms}ms > max {max_ms}ms"),
            StyleViolation::ReadingSpeedExceeded { block_id, actual_cps, max_cps } =>
                format!("Block {block_id}: {actual_cps:.1} CPS > max {max_cps:.1} CPS"),
            StyleViolation::MissingCapitalisation { block_id } =>
                format!("Block {block_id}: caption does not begin with a capital letter"),
            StyleViolation::MissingTerminalPunctuation { block_id } =>
                format!("Block {block_id}: caption does not end with terminal punctuation"),
            StyleViolation::CaptionOverlap { block_id_a, block_id_b } =>
                format!("Blocks {block_id_a} and {block_id_b} overlap in time"),
            StyleViolation::GapTooShort { before_id, after_id, actual_ms, min_ms } =>
                format!("Gap between blocks {before_id} and {after_id}: {actual_ms}ms < min {min_ms}ms"),
            StyleViolation::ShortLineImbalance { block_id, shorter_len, max_short } =>
                format!("Block {block_id}: shorter line has {shorter_len} chars > balance limit {max_short}"),
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::CaptionPosition;

    fn make_block(id: u32, start_ms: u64, end_ms: u64, lines: Vec<&str>) -> CaptionBlock {
        CaptionBlock {
            id,
            start_ms,
            end_ms,
            lines: lines.into_iter().map(|s| s.to_string()).collect(),
            speaker_id: None,
            position: CaptionPosition::Bottom,
        }
    }

    // ── per-block checks ──────────────────────────────────────────────────────

    #[test]
    fn valid_broadcast_block_no_violations() {
        let guide = StyleGuide::broadcast();
        let block = make_block(1, 0, 2000, vec!["Hello, world!"]);
        assert!(guide.check_block(&block).is_empty());
    }

    #[test]
    fn line_too_long_detected() {
        let guide = StyleGuide::broadcast(); // max 42 chars
        let long_line = "A".repeat(50);
        let block = make_block(1, 0, 3000, vec![&long_line]);
        let v = guide.check_block(&block);
        assert!(v
            .iter()
            .any(|v| matches!(v, StyleViolation::LineTooLong { .. })));
    }

    #[test]
    fn too_many_lines_detected() {
        let guide = StyleGuide::broadcast(); // max 2 lines
        let block = make_block(1, 0, 4000, vec!["Line one.", "Line two.", "Line three."]);
        let v = guide.check_block(&block);
        assert!(v
            .iter()
            .any(|v| matches!(v, StyleViolation::TooManyLines { .. })));
    }

    #[test]
    fn duration_too_short_detected() {
        let guide = StyleGuide::broadcast(); // min 833ms
        let block = make_block(1, 0, 200, vec!["Hi."]);
        let v = guide.check_block(&block);
        assert!(v
            .iter()
            .any(|v| matches!(v, StyleViolation::DurationTooShort { .. })));
    }

    #[test]
    fn duration_too_long_detected() {
        let guide = StyleGuide::broadcast(); // max 7000ms
        let block = make_block(1, 0, 10_000, vec!["Hello."]);
        let v = guide.check_block(&block);
        assert!(v
            .iter()
            .any(|v| matches!(v, StyleViolation::DurationTooLong { .. })));
    }

    #[test]
    fn reading_speed_exceeded_for_children() {
        let guide = StyleGuide::children(); // max 10 CPS
                                            // 100 chars / 1 second = 100 CPS — well over the limit.
        let long_text = "A".repeat(100);
        let block = make_block(1, 0, 1000, vec![&long_text]);
        let v = guide.check_block(&block);
        assert!(v
            .iter()
            .any(|v| matches!(v, StyleViolation::ReadingSpeedExceeded { .. })));
    }

    #[test]
    fn missing_capitalisation_detected() {
        let guide = StyleGuide::broadcast();
        let block = make_block(1, 0, 2000, vec!["hello world"]);
        let v = guide.check_block(&block);
        assert!(v
            .iter()
            .any(|v| matches!(v, StyleViolation::MissingCapitalisation { .. })));
    }

    #[test]
    fn terminal_punctuation_required_by_children_guide() {
        let guide = StyleGuide::children();
        let block = make_block(1, 0, 2000, vec!["Hello world"]);
        let v = guide.check_block(&block);
        assert!(v
            .iter()
            .any(|v| matches!(v, StyleViolation::MissingTerminalPunctuation { .. })));
    }

    // ── track-level checks ────────────────────────────────────────────────────

    #[test]
    fn gap_too_short_detected() {
        let guide = StyleGuide::broadcast(); // min gap 80ms
        let blocks = vec![
            make_block(1, 0, 1000, vec!["First."]),
            make_block(2, 1010, 2000, vec!["Second."]), // gap = 10ms < 80ms
        ];
        let v = guide.check_track(&blocks);
        assert!(v
            .iter()
            .any(|v| matches!(v, StyleViolation::GapTooShort { .. })));
    }

    #[test]
    fn overlap_detected() {
        let guide = StyleGuide::broadcast();
        let blocks = vec![
            make_block(1, 0, 1500, vec!["First."]),
            make_block(2, 1000, 2000, vec!["Second."]), // overlaps
        ];
        let v = guide.check_track(&blocks);
        assert!(v
            .iter()
            .any(|v| matches!(v, StyleViolation::CaptionOverlap { .. })));
    }

    #[test]
    fn compliance_score_perfect() {
        let guide = StyleGuide::broadcast();
        let blocks = vec![
            make_block(1, 0, 2000, vec!["Hello, world!"]),
            make_block(2, 2100, 4000, vec!["This is a test."]),
        ];
        let score = guide.compliance_score(&blocks);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn compliance_score_partial() {
        let guide = StyleGuide::broadcast();
        // One block has a very short duration.
        let blocks = vec![
            make_block(1, 0, 100, vec!["Hi."]),
            make_block(2, 300, 2000, vec!["This is fine."]),
        ];
        let score = guide.compliance_score(&blocks);
        assert!(score < 1.0, "score should be < 1.0 due to violation");
        assert!(score >= 0.0);
    }

    #[test]
    fn audience_max_cps_ordering() {
        assert!(Audience::Children.max_cps() < Audience::Adult.max_cps());
        assert!(Audience::Adult.max_cps() < Audience::Broadcast.max_cps());
    }

    #[test]
    fn custom_audience_respected() {
        let guide = StyleGuide {
            audience: Audience::Custom { max_cps: 5 },
            max_chars_per_line: 80,
            max_lines: 3,
            min_duration_ms: 0,
            max_duration_ms: 60_000,
            min_gap_ms: 0,
            require_capitalisation: false,
            require_terminal_punctuation: false,
            max_short_line_chars: None,
        };
        // 50 chars / 1 second = 50 CPS > 5 max
        let block = make_block(1, 0, 1000, vec![&"A".repeat(50)]);
        let v = guide.check_block(&block);
        assert!(v
            .iter()
            .any(|v| matches!(v, StyleViolation::ReadingSpeedExceeded { .. })));
    }

    #[test]
    fn violation_description_non_empty() {
        let v = StyleViolation::LineTooLong {
            block_id: 3,
            line_index: 0,
            actual: 50,
            max: 42,
        };
        assert!(!v.description().is_empty());
        assert!(v.description().contains("50"));
    }
}
