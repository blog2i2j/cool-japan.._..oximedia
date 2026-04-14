//! Audio mixing down-mix table — channel layout reduction rules.
//!
//! This module provides a complete, standards-derived down-mix coefficient table
//! for reducing multi-channel audio to fewer channels without clipping artefacts
//! and with perceptually correct loudness relationships.
//!
//! # Standards references
//!
//! - ITU-R BS.775-3 — multichannel stereophonic sound systems
//! - ATSC A/52 (Dolby AC-3) centre mix level / surround mix level definitions
//! - EBU R128 — loudness normalisation (used to derive output trim suggestions)
//!
//! # Supported conversions
//!
//! | Source layout | Target layouts             |
//! |---------------|----------------------------|
//! | 7.1           | 5.1, stereo, mono          |
//! | 5.1           | 5.0, stereo, mono          |
//! | 5.0           | stereo, mono               |
//! | 4.0 quad      | stereo, mono               |
//! | 2.1           | stereo, mono               |
//! | stereo        | mono                       |
//!
//! Each conversion is expressed as a [`DownmixRule`] — a list of
//! [`DownmixCoefficient`] entries that map `(input_channel, output_channel)` pairs
//! to linear mix coefficients.

use std::fmt;

// ──────────────────────────────────────────────────────────────────────────────
// Channel layout enum
// ──────────────────────────────────────────────────────────────────────────────

/// Standard multi-channel audio layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DownmixLayout {
    /// Single mono channel.
    Mono,
    /// Two stereo channels: L, R.
    Stereo,
    /// 2.1: L, R, LFE.
    TwoPointOne,
    /// 4.0 quad: L, R, Ls, Rs.
    Quad,
    /// 5.0 surround: L, R, C, Ls, Rs.
    FivePointZero,
    /// 5.1 surround: L, R, C, LFE, Ls, Rs.
    FivePointOne,
    /// 7.1 surround: L, R, C, LFE, Lss, Rss, Lrs, Rrs.
    SevenPointOne,
}

impl DownmixLayout {
    /// Number of channels in this layout.
    #[must_use]
    pub fn channel_count(self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::TwoPointOne => 3,
            Self::Quad => 4,
            Self::FivePointZero => 5,
            Self::FivePointOne => 6,
            Self::SevenPointOne => 8,
        }
    }

    /// Human-readable label (e.g. `"5.1"`).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Mono => "mono",
            Self::Stereo => "stereo",
            Self::TwoPointOne => "2.1",
            Self::Quad => "4.0",
            Self::FivePointZero => "5.0",
            Self::FivePointOne => "5.1",
            Self::SevenPointOne => "7.1",
        }
    }

    /// Whether this layout carries a dedicated LFE sub-woofer channel.
    #[must_use]
    pub fn has_lfe(self) -> bool {
        matches!(self, Self::TwoPointOne | Self::FivePointOne | Self::SevenPointOne)
    }
}

impl fmt::Display for DownmixLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Down-mix coefficient
// ──────────────────────────────────────────────────────────────────────────────

/// A single routing entry: mix `input_channel` into `output_channel` with
/// `gain` (linear amplitude, 1.0 = 0 dBFS, 0.707 ≈ −3 dB, etc.).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DownmixCoefficient {
    /// Zero-based index of the source channel (in the input layout).
    pub input_channel: usize,
    /// Zero-based index of the destination channel (in the output layout).
    pub output_channel: usize,
    /// Linear amplitude coefficient applied to the source sample.
    pub gain: f32,
}

impl DownmixCoefficient {
    /// Construct a new coefficient entry.
    #[must_use]
    pub fn new(input_channel: usize, output_channel: usize, gain: f32) -> Self {
        Self { input_channel, output_channel, gain }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Down-mix rule
// ──────────────────────────────────────────────────────────────────────────────

/// A complete down-mix rule from one layout to another.
///
/// Contains the ordered list of [`DownmixCoefficient`] entries that fully describe
/// the mixing matrix for a specific `(source, target)` layout pair.
#[derive(Debug, Clone)]
pub struct DownmixRule {
    /// Source layout.
    pub source: DownmixLayout,
    /// Target layout.
    pub target: DownmixLayout,
    /// Mix coefficients (may overlap: multiple inputs → same output, or one input → many outputs).
    pub coefficients: Vec<DownmixCoefficient>,
    /// Optional overall output trim in dB to compensate for loudness increase.
    pub output_trim_db: f32,
}

impl DownmixRule {
    /// Apply this rule to an interleaved PCM buffer.
    ///
    /// `input`  — interleaved f32 samples, `frame_count * source.channel_count()` elements.
    /// `output` — pre-zeroed f32 buffer, `frame_count * target.channel_count()` elements.
    ///
    /// The trim gain (converted from `output_trim_db`) is applied to every output sample.
    pub fn apply(&self, input: &[f32], output: &mut [f32], frame_count: usize) {
        let in_ch = self.source.channel_count();
        let out_ch = self.target.channel_count();
        let trim = db_to_linear(self.output_trim_db);

        for frame in 0..frame_count {
            for coeff in &self.coefficients {
                let in_pos = frame * in_ch + coeff.input_channel;
                let out_pos = frame * out_ch + coeff.output_channel;
                if in_pos < input.len() && out_pos < output.len() {
                    output[out_pos] += input[in_pos] * coeff.gain;
                }
            }
        }

        // Apply output trim to compensate for loudness increase.
        if (trim - 1.0).abs() > 1e-6 {
            for s in output.iter_mut() {
                *s *= trim;
            }
        }
    }

    /// Validate internal consistency: all channel indices must be within range.
    ///
    /// # Errors
    ///
    /// Returns an error string if any coefficient references an out-of-range channel.
    pub fn validate(&self) -> Result<(), String> {
        let max_in = self.source.channel_count();
        let max_out = self.target.channel_count();
        for (i, c) in self.coefficients.iter().enumerate() {
            if c.input_channel >= max_in {
                return Err(format!(
                    "coefficient[{i}]: input_channel {} >= source channel count {max_in}",
                    c.input_channel
                ));
            }
            if c.output_channel >= max_out {
                return Err(format!(
                    "coefficient[{i}]: output_channel {} >= target channel count {max_out}",
                    c.output_channel
                ));
            }
        }
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Errors
// ──────────────────────────────────────────────────────────────────────────────

/// Errors from the down-mix table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownmixError {
    /// No rule exists for the requested `(source, target)` conversion.
    NoRuleAvailable {
        /// Source layout that was requested.
        source: DownmixLayout,
        /// Target layout that was requested.
        target: DownmixLayout,
    },
    /// The source layout has fewer channels than the target — use up-mix instead.
    SourceSmallerThanTarget {
        /// Source layout.
        source: DownmixLayout,
        /// Target layout.
        target: DownmixLayout,
    },
}

impl fmt::Display for DownmixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoRuleAvailable { source, target } => write!(
                f,
                "no down-mix rule from {source} → {target}",
            ),
            Self::SourceSmallerThanTarget { source, target } => write!(
                f,
                "source layout {source} ({} ch) has fewer channels than target {target} ({} ch)",
                source.channel_count(),
                target.channel_count(),
            ),
        }
    }
}

impl std::error::Error for DownmixError {}

// ──────────────────────────────────────────────────────────────────────────────
// Down-mix table
// ──────────────────────────────────────────────────────────────────────────────

/// Channel-layout coefficient table for audio down-mixing.
///
/// Call [`DownmixTable::standard()`] to obtain a table pre-populated with
/// ITU-R BS.775/ATSC A/52–derived coefficients for all common reductions.
pub struct DownmixTable {
    rules: Vec<DownmixRule>,
}

impl Default for DownmixTable {
    fn default() -> Self {
        Self::standard()
    }
}

impl DownmixTable {
    /// Construct an empty table (no rules loaded).
    #[must_use]
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a custom down-mix rule.
    pub fn add_rule(&mut self, rule: DownmixRule) {
        self.rules.push(rule);
    }

    /// Look up a rule for a `(source, target)` layout pair.
    ///
    /// # Errors
    ///
    /// - [`DownmixError::SourceSmallerThanTarget`] — if `source` has fewer channels.
    /// - [`DownmixError::NoRuleAvailable`] — if no rule exists for this pair.
    pub fn get(
        &self,
        source: DownmixLayout,
        target: DownmixLayout,
    ) -> Result<&DownmixRule, DownmixError> {
        if source.channel_count() < target.channel_count() {
            return Err(DownmixError::SourceSmallerThanTarget { source, target });
        }
        if source == target {
            // Identity: no down-mix needed — caller should use passthrough.
            return self
                .rules
                .iter()
                .find(|r| r.source == source && r.target == target)
                .ok_or(DownmixError::NoRuleAvailable { source, target });
        }
        self.rules
            .iter()
            .find(|r| r.source == source && r.target == target)
            .ok_or(DownmixError::NoRuleAvailable { source, target })
    }

    /// Returns `true` if a rule exists for `(source, target)`.
    #[must_use]
    pub fn has_rule(&self, source: DownmixLayout, target: DownmixLayout) -> bool {
        self.rules.iter().any(|r| r.source == source && r.target == target)
    }

    /// Returns the number of rules in the table.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Build the standard ITU-R BS.775 / ATSC A/52 down-mix coefficient table.
    ///
    /// Channel index conventions per layout:
    ///
    /// | Layout | Indices                                         |
    /// |--------|-------------------------------------------------|
    /// | 7.1    | 0=L, 1=R, 2=C, 3=LFE, 4=Lss, 5=Rss, 6=Lrs, 7=Rrs |
    /// | 5.1    | 0=L, 1=R, 2=C, 3=LFE, 4=Ls,  5=Rs             |
    /// | 5.0    | 0=L, 1=R, 2=C, 3=Ls,  4=Rs                     |
    /// | Quad   | 0=L, 1=R, 2=Ls, 3=Rs                            |
    /// | 2.1    | 0=L, 1=R, 2=LFE                                 |
    /// | Stereo | 0=L, 1=R                                        |
    /// | Mono   | 0=M                                             |
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn standard() -> Self {
        // ITU centre mix level  = 0.707 (−3.01 dB)
        // ITU surround mix level = 0.707 (−3.01 dB)
        // Equal-power stereo mix = 0.707 per channel (L+R → mono at −3 dB)
        const C_MIX: f32 = 0.707_107; // centre to L/R
        const S_MIX: f32 = 0.707_107; // surround to L/R
        const EQ_PWR: f32 = 0.707_107; // equal-power mono mix
        const FULL: f32 = 1.0; // passthrough

        let mut table = Self::new();

        // ── 7.1 → 5.1 ─────────────────────────────────────────────────────────
        // L→L, R→R, C→C, LFE→LFE,
        // Lss+Lrs → Ls (−3 dB each), Rss+Rrs → Rs (−3 dB each)
        table.add_rule(DownmixRule {
            source: DownmixLayout::SevenPointOne,
            target: DownmixLayout::FivePointOne,
            coefficients: vec![
                DownmixCoefficient::new(0, 0, FULL),    // L   → L
                DownmixCoefficient::new(1, 1, FULL),    // R   → R
                DownmixCoefficient::new(2, 2, FULL),    // C   → C
                DownmixCoefficient::new(3, 3, FULL),    // LFE → LFE
                DownmixCoefficient::new(4, 4, S_MIX),   // Lss → Ls (−3 dB)
                DownmixCoefficient::new(6, 4, S_MIX),   // Lrs → Ls (−3 dB)
                DownmixCoefficient::new(5, 5, S_MIX),   // Rss → Rs (−3 dB)
                DownmixCoefficient::new(7, 5, S_MIX),   // Rrs → Rs (−3 dB)
            ],
            output_trim_db: 0.0,
        });

        // ── 7.1 → Stereo ──────────────────────────────────────────────────────
        // L → Lo, R → Ro, C → Lo+Ro (−3 dB), LFE ignored,
        // Lss+Lrs → Lo (−3 dB each), Rss+Rrs → Ro (−3 dB each)
        table.add_rule(DownmixRule {
            source: DownmixLayout::SevenPointOne,
            target: DownmixLayout::Stereo,
            coefficients: vec![
                DownmixCoefficient::new(0, 0, FULL),    // L   → Lo
                DownmixCoefficient::new(1, 1, FULL),    // R   → Ro
                DownmixCoefficient::new(2, 0, C_MIX),   // C   → Lo
                DownmixCoefficient::new(2, 1, C_MIX),   // C   → Ro
                // LFE (ch 3) intentionally omitted
                DownmixCoefficient::new(4, 0, S_MIX),   // Lss → Lo
                DownmixCoefficient::new(5, 1, S_MIX),   // Rss → Ro
                DownmixCoefficient::new(6, 0, S_MIX),   // Lrs → Lo
                DownmixCoefficient::new(7, 1, S_MIX),   // Rrs → Ro
            ],
            output_trim_db: -3.0,
        });

        // ── 7.1 → Mono ────────────────────────────────────────────────────────
        table.add_rule(DownmixRule {
            source: DownmixLayout::SevenPointOne,
            target: DownmixLayout::Mono,
            coefficients: vec![
                DownmixCoefficient::new(0, 0, EQ_PWR),  // L
                DownmixCoefficient::new(1, 0, EQ_PWR),  // R
                DownmixCoefficient::new(2, 0, C_MIX),   // C
                // LFE omitted
                DownmixCoefficient::new(4, 0, S_MIX * EQ_PWR), // Lss
                DownmixCoefficient::new(5, 0, S_MIX * EQ_PWR), // Rss
                DownmixCoefficient::new(6, 0, S_MIX * EQ_PWR), // Lrs
                DownmixCoefficient::new(7, 0, S_MIX * EQ_PWR), // Rrs
            ],
            output_trim_db: -3.0,
        });

        // ── 5.1 → Stereo ──────────────────────────────────────────────────────
        // ITU-R BS.775 equation:
        //   Lo = L + C×0.707 + Ls×0.707
        //   Ro = R + C×0.707 + Rs×0.707
        table.add_rule(DownmixRule {
            source: DownmixLayout::FivePointOne,
            target: DownmixLayout::Stereo,
            coefficients: vec![
                DownmixCoefficient::new(0, 0, FULL),    // L   → Lo
                DownmixCoefficient::new(1, 1, FULL),    // R   → Ro
                DownmixCoefficient::new(2, 0, C_MIX),   // C   → Lo (−3 dB)
                DownmixCoefficient::new(2, 1, C_MIX),   // C   → Ro (−3 dB)
                // LFE (ch 3) intentionally omitted
                DownmixCoefficient::new(4, 0, S_MIX),   // Ls  → Lo (−3 dB)
                DownmixCoefficient::new(5, 1, S_MIX),   // Rs  → Ro (−3 dB)
            ],
            output_trim_db: -3.0,
        });

        // ── 5.1 → Mono ────────────────────────────────────────────────────────
        table.add_rule(DownmixRule {
            source: DownmixLayout::FivePointOne,
            target: DownmixLayout::Mono,
            coefficients: vec![
                DownmixCoefficient::new(0, 0, EQ_PWR),
                DownmixCoefficient::new(1, 0, EQ_PWR),
                DownmixCoefficient::new(2, 0, C_MIX),
                // LFE omitted
                DownmixCoefficient::new(4, 0, S_MIX * EQ_PWR),
                DownmixCoefficient::new(5, 0, S_MIX * EQ_PWR),
            ],
            output_trim_db: -3.0,
        });

        // ── 5.0 → Stereo ──────────────────────────────────────────────────────
        // 5.0: L=0, R=1, C=2, Ls=3, Rs=4
        table.add_rule(DownmixRule {
            source: DownmixLayout::FivePointZero,
            target: DownmixLayout::Stereo,
            coefficients: vec![
                DownmixCoefficient::new(0, 0, FULL),
                DownmixCoefficient::new(1, 1, FULL),
                DownmixCoefficient::new(2, 0, C_MIX),
                DownmixCoefficient::new(2, 1, C_MIX),
                DownmixCoefficient::new(3, 0, S_MIX),
                DownmixCoefficient::new(4, 1, S_MIX),
            ],
            output_trim_db: -3.0,
        });

        // ── 5.0 → Mono ────────────────────────────────────────────────────────
        table.add_rule(DownmixRule {
            source: DownmixLayout::FivePointZero,
            target: DownmixLayout::Mono,
            coefficients: vec![
                DownmixCoefficient::new(0, 0, EQ_PWR),
                DownmixCoefficient::new(1, 0, EQ_PWR),
                DownmixCoefficient::new(2, 0, C_MIX),
                DownmixCoefficient::new(3, 0, S_MIX * EQ_PWR),
                DownmixCoefficient::new(4, 0, S_MIX * EQ_PWR),
            ],
            output_trim_db: -3.0,
        });

        // ── Quad → Stereo ─────────────────────────────────────────────────────
        // Quad: L=0, R=1, Ls=2, Rs=3
        table.add_rule(DownmixRule {
            source: DownmixLayout::Quad,
            target: DownmixLayout::Stereo,
            coefficients: vec![
                DownmixCoefficient::new(0, 0, FULL),
                DownmixCoefficient::new(1, 1, FULL),
                DownmixCoefficient::new(2, 0, S_MIX),
                DownmixCoefficient::new(3, 1, S_MIX),
            ],
            output_trim_db: -3.0,
        });

        // ── Quad → Mono ───────────────────────────────────────────────────────
        table.add_rule(DownmixRule {
            source: DownmixLayout::Quad,
            target: DownmixLayout::Mono,
            coefficients: vec![
                DownmixCoefficient::new(0, 0, EQ_PWR),
                DownmixCoefficient::new(1, 0, EQ_PWR),
                DownmixCoefficient::new(2, 0, S_MIX * EQ_PWR),
                DownmixCoefficient::new(3, 0, S_MIX * EQ_PWR),
            ],
            output_trim_db: -3.0,
        });

        // ── 2.1 → Stereo ──────────────────────────────────────────────────────
        // LFE is dropped; L and R pass through unchanged.
        table.add_rule(DownmixRule {
            source: DownmixLayout::TwoPointOne,
            target: DownmixLayout::Stereo,
            coefficients: vec![
                DownmixCoefficient::new(0, 0, FULL),
                DownmixCoefficient::new(1, 1, FULL),
                // LFE (ch 2) dropped
            ],
            output_trim_db: 0.0,
        });

        // ── 2.1 → Mono ────────────────────────────────────────────────────────
        table.add_rule(DownmixRule {
            source: DownmixLayout::TwoPointOne,
            target: DownmixLayout::Mono,
            coefficients: vec![
                DownmixCoefficient::new(0, 0, EQ_PWR),
                DownmixCoefficient::new(1, 0, EQ_PWR),
                // LFE dropped
            ],
            output_trim_db: 0.0,
        });

        // ── Stereo → Mono ─────────────────────────────────────────────────────
        table.add_rule(DownmixRule {
            source: DownmixLayout::Stereo,
            target: DownmixLayout::Mono,
            coefficients: vec![
                DownmixCoefficient::new(0, 0, EQ_PWR),
                DownmixCoefficient::new(1, 0, EQ_PWR),
            ],
            output_trim_db: 0.0,
        });

        table
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper
// ──────────────────────────────────────────────────────────────────────────────

/// Convert a gain in dB to a linear amplitude factor.
#[inline]
#[must_use]
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Layout helpers ─────────────────────────────────────────────────────────

    #[test]
    fn test_channel_counts() {
        assert_eq!(DownmixLayout::Mono.channel_count(), 1);
        assert_eq!(DownmixLayout::Stereo.channel_count(), 2);
        assert_eq!(DownmixLayout::TwoPointOne.channel_count(), 3);
        assert_eq!(DownmixLayout::Quad.channel_count(), 4);
        assert_eq!(DownmixLayout::FivePointZero.channel_count(), 5);
        assert_eq!(DownmixLayout::FivePointOne.channel_count(), 6);
        assert_eq!(DownmixLayout::SevenPointOne.channel_count(), 8);
    }

    #[test]
    fn test_has_lfe_correct() {
        assert!(DownmixLayout::TwoPointOne.has_lfe());
        assert!(DownmixLayout::FivePointOne.has_lfe());
        assert!(DownmixLayout::SevenPointOne.has_lfe());
        assert!(!DownmixLayout::Stereo.has_lfe());
        assert!(!DownmixLayout::Quad.has_lfe());
        assert!(!DownmixLayout::FivePointZero.has_lfe());
    }

    #[test]
    fn test_layout_display() {
        assert_eq!(DownmixLayout::FivePointOne.to_string(), "5.1");
        assert_eq!(DownmixLayout::SevenPointOne.to_string(), "7.1");
    }

    // ── Standard table lookup ──────────────────────────────────────────────────

    #[test]
    fn test_standard_table_has_71_to_51_rule() {
        let table = DownmixTable::standard();
        assert!(table.has_rule(DownmixLayout::SevenPointOne, DownmixLayout::FivePointOne));
    }

    #[test]
    fn test_standard_table_has_51_to_stereo_rule() {
        let table = DownmixTable::standard();
        assert!(table.has_rule(DownmixLayout::FivePointOne, DownmixLayout::Stereo));
    }

    #[test]
    fn test_standard_table_has_stereo_to_mono_rule() {
        let table = DownmixTable::standard();
        assert!(table.has_rule(DownmixLayout::Stereo, DownmixLayout::Mono));
    }

    #[test]
    fn test_no_upmix_rule_available() {
        let table = DownmixTable::standard();
        let err = table
            .get(DownmixLayout::Mono, DownmixLayout::Stereo)
            .expect_err("mono → stereo is an up-mix");
        assert!(matches!(
            err,
            DownmixError::SourceSmallerThanTarget { .. }
        ));
    }

    #[test]
    fn test_missing_rule_returns_error() {
        let table = DownmixTable::new(); // empty
        let err = table
            .get(DownmixLayout::FivePointOne, DownmixLayout::Stereo)
            .expect_err("empty table has no rules");
        assert!(matches!(err, DownmixError::NoRuleAvailable { .. }));
    }

    // ── Coefficient validation ─────────────────────────────────────────────────

    #[test]
    fn test_all_standard_rules_validate_cleanly() {
        let table = DownmixTable::standard();
        for rule in &table.rules {
            assert!(
                rule.validate().is_ok(),
                "rule {:?}→{:?} failed validation: {:?}",
                rule.source,
                rule.target,
                rule.validate()
            );
        }
    }

    #[test]
    fn test_validation_catches_out_of_range_input() {
        let bad_rule = DownmixRule {
            source: DownmixLayout::Stereo,
            target: DownmixLayout::Mono,
            coefficients: vec![DownmixCoefficient::new(99, 0, 1.0)], // ch 99 doesn't exist
            output_trim_db: 0.0,
        };
        assert!(bad_rule.validate().is_err());
    }

    #[test]
    fn test_validation_catches_out_of_range_output() {
        let bad_rule = DownmixRule {
            source: DownmixLayout::Stereo,
            target: DownmixLayout::Mono,
            coefficients: vec![DownmixCoefficient::new(0, 5, 1.0)], // output ch 5 doesn't exist in mono
            output_trim_db: 0.0,
        };
        assert!(bad_rule.validate().is_err());
    }

    // ── apply() correctness ────────────────────────────────────────────────────

    #[test]
    fn test_stereo_to_mono_apply() {
        let table = DownmixTable::standard();
        let rule = table
            .get(DownmixLayout::Stereo, DownmixLayout::Mono)
            .expect("rule exists");

        // 1 frame: L=1.0, R=1.0
        let input = [1.0_f32, 1.0_f32];
        let mut output = [0.0_f32; 1];
        rule.apply(&input, &mut output, 1);

        // Expected: (1.0 * 0.707 + 1.0 * 0.707) * trim(0 dB) ≈ 1.414
        let expected = 2.0_f32 * 0.707_107;
        assert!(
            (output[0] - expected).abs() < 1e-5,
            "mono output {:.6} != {:.6}",
            output[0],
            expected
        );
    }

    #[test]
    fn test_51_to_stereo_centre_split() {
        let table = DownmixTable::standard();
        let rule = table
            .get(DownmixLayout::FivePointOne, DownmixLayout::Stereo)
            .expect("rule exists");

        // 1 frame: only centre channel set to 1.0, all others 0.0
        // 5.1: L=0, R=1, C=2, LFE=3, Ls=4, Rs=5
        let input = [0.0_f32, 0.0, 1.0, 0.0, 0.0, 0.0];
        let mut output = [0.0_f32; 2];
        rule.apply(&input, &mut output, 1);

        // C spreads to Lo and Ro at 0.707 each, then trim of −3 dB (0.707 linear)
        let trim = 10.0_f32.powf(-3.0 / 20.0);
        let expected = 0.707_107_f32 * trim;
        assert!(
            (output[0] - expected).abs() < 1e-5,
            "Lo {:.6} != {:.6}",
            output[0],
            expected
        );
        assert!(
            (output[1] - expected).abs() < 1e-5,
            "Ro {:.6} != {:.6}",
            output[1],
            expected
        );
    }

    #[test]
    fn test_71_to_51_channel_count() {
        let table = DownmixTable::standard();
        let rule = table
            .get(DownmixLayout::SevenPointOne, DownmixLayout::FivePointOne)
            .expect("rule exists");
        assert_eq!(rule.target.channel_count(), 6);
    }

    // ── Error display ──────────────────────────────────────────────────────────

    #[test]
    fn test_error_display_no_rule() {
        let err = DownmixError::NoRuleAvailable {
            source: DownmixLayout::FivePointOne,
            target: DownmixLayout::Quad,
        };
        let s = err.to_string();
        assert!(s.contains("5.1"));
        assert!(s.contains("4.0"));
    }

    #[test]
    fn test_error_display_upmix() {
        let err = DownmixError::SourceSmallerThanTarget {
            source: DownmixLayout::Mono,
            target: DownmixLayout::Stereo,
        };
        let s = err.to_string();
        assert!(s.contains("mono"));
        assert!(s.contains("stereo"));
    }

    // ── db_to_linear ───────────────────────────────────────────────────────────

    #[test]
    fn test_db_to_linear_zero_db() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_db_to_linear_minus_6db() {
        // −6 dB ≈ 0.501
        assert!((db_to_linear(-6.0) - 0.501_187).abs() < 1e-4);
    }
}
