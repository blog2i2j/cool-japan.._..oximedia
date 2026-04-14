//! DASH MPD (Media Presentation Description) incremental update helper.
//!
//! [`MpdUpdater`] takes a base MPD XML string and provides a fluent API for
//! appending `<Period>` elements and serialising the final document.  It does
//! not depend on an XML parser and instead performs simple string manipulation
//! to keep the crate pure-Rust with minimal dependencies.
//!
//! # Example
//!
//! ```rust
//! use oximedia_stream::dash_mpd_updater::MpdUpdater;
//!
//! let base = r#"<?xml version="1.0"?><MPD type="dynamic"></MPD>"#;
//! let mut updater = MpdUpdater::new(base);
//! updater.add_period(6.0);
//! updater.add_period(6.0);
//! let mpd = updater.build();
//! assert!(mpd.contains("period_"));
//! ```

// ─── PeriodInfo ───────────────────────────────────────────────────────────────

/// Describes a single MPEG-DASH `<Period>` to be appended to the MPD.
#[derive(Debug, Clone)]
pub struct PeriodInfo {
    /// Unique period identifier, e.g. `"period_0"`.
    pub id: String,
    /// Period duration in seconds (ISO 8601 duration `PT{n}S` format).
    pub duration_s: f64,
    /// Optional media presentation start offset in seconds from the MPD start.
    pub start_s: Option<f64>,
}

impl PeriodInfo {
    /// Render the period as a minimal DASH `<Period>` XML element.
    #[must_use]
    pub fn to_xml(&self) -> String {
        let duration = format_iso_duration(self.duration_s);
        let start_attr = match self.start_s {
            Some(s) if s > 0.0 => format!(" start=\"{}\"", format_iso_duration(s)),
            _ => String::new(),
        };
        format!(
            "<Period id=\"{}\"{start_attr} duration=\"{duration}\"/>",
            self.id
        )
    }
}

/// Format a duration in seconds as an ISO 8601 duration string (`PT…S`).
fn format_iso_duration(seconds: f64) -> String {
    if seconds.fract().abs() < 1e-9 {
        format!("PT{}S", seconds as u64)
    } else {
        // Use three decimal places for sub-second precision.
        format!("PT{:.3}S", seconds)
    }
}

// ─── MpdUpdater ───────────────────────────────────────────────────────────────

/// Incrementally builds a DASH MPD by appending `<Period>` elements.
///
/// # Lifecycle
///
/// 1. Construct with [`MpdUpdater::new`] supplying a base MPD string.
/// 2. Call [`add_period`] once per available DASH period.
/// 3. Call [`Self::build`] to get the final MPD XML string.
///
/// [`add_period`]: MpdUpdater::add_period
pub struct MpdUpdater {
    /// Base MPD XML (everything before the closing `</MPD>` tag, or the full
    /// document if it has no `</MPD>` close).
    base: String,
    /// Accumulated periods.
    periods: Vec<PeriodInfo>,
    /// Monotonically increasing period index used to generate unique IDs.
    next_period_index: u64,
    /// Cumulative elapsed time in seconds (used to compute period start offsets).
    elapsed_s: f64,
}

impl MpdUpdater {
    /// Create a new updater from a base MPD document.
    ///
    /// The `base_mpd` should be a valid DASH MPD XML string.  Existing
    /// `<Period>` elements are preserved: [`build`] inserts new periods before
    /// the `</MPD>` closing tag.
    ///
    /// [`build`]: MpdUpdater::build
    pub fn new(base_mpd: impl Into<String>) -> Self {
        Self {
            base: base_mpd.into(),
            periods: Vec::new(),
            next_period_index: 0,
            elapsed_s: 0.0,
        }
    }

    /// Append a new period of `duration_s` seconds to the MPD.
    ///
    /// The period is assigned a sequential ID (`"period_0"`, `"period_1"`, …)
    /// and its `start` offset is derived from the cumulative duration of all
    /// previously added periods.
    pub fn add_period(&mut self, duration_s: f64) {
        let id = format!("period_{}", self.next_period_index);
        let start_s = if self.elapsed_s > 0.0 {
            Some(self.elapsed_s)
        } else {
            None
        };
        self.periods.push(PeriodInfo {
            id,
            duration_s,
            start_s,
        });
        self.elapsed_s += duration_s;
        self.next_period_index += 1;
    }

    /// Return the number of periods added so far.
    pub fn period_count(&self) -> usize {
        self.periods.len()
    }

    /// Return the cumulative duration of all added periods in seconds.
    pub fn total_duration_s(&self) -> f64 {
        self.elapsed_s
    }

    /// Build the final MPD XML string.
    ///
    /// New periods are inserted before the `</MPD>` closing tag when present.
    /// If the base document has no `</MPD>`, periods are appended at the end.
    #[must_use]
    pub fn build(&self) -> String {
        if self.periods.is_empty() {
            return self.base.clone();
        }

        let period_xml: String = self
            .periods
            .iter()
            .map(|p| p.to_xml())
            .collect::<Vec<_>>()
            .join("\n");

        // Find the </MPD> closing tag and insert before it.
        if let Some(pos) = self.base.rfind("</MPD>") {
            let (before, after) = self.base.split_at(pos);
            format!("{before}\n{period_xml}\n{after}")
        } else {
            // No closing tag — append at the end.
            format!("{}\n{period_xml}", self.base)
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<MPD type="dynamic" profiles="urn:mpeg:dash:profile:isoff-live:2011">
</MPD>"#;

    #[test]
    fn test_build_with_no_periods_returns_base() {
        let updater = MpdUpdater::new(BASE);
        assert_eq!(updater.build(), BASE);
    }

    #[test]
    fn test_add_period_increments_count() {
        let mut updater = MpdUpdater::new(BASE);
        updater.add_period(6.0);
        updater.add_period(6.0);
        assert_eq!(updater.period_count(), 2);
    }

    #[test]
    fn test_build_contains_period_ids() {
        let mut updater = MpdUpdater::new(BASE);
        updater.add_period(6.0);
        updater.add_period(4.0);
        let mpd = updater.build();
        assert!(mpd.contains("period_0"), "should contain period_0: {mpd}");
        assert!(mpd.contains("period_1"), "should contain period_1: {mpd}");
    }

    #[test]
    fn test_build_inserts_before_closing_tag() {
        let mut updater = MpdUpdater::new(BASE);
        updater.add_period(6.0);
        let mpd = updater.build();
        // Period XML must appear before the </MPD> closing tag
        let period_pos = mpd.find("period_0").expect("period_0 in output");
        let close_pos = mpd.find("</MPD>").expect("</MPD> in output");
        assert!(period_pos < close_pos, "period must precede </MPD>");
    }

    #[test]
    fn test_total_duration_accumulates() {
        let mut updater = MpdUpdater::new(BASE);
        updater.add_period(6.0);
        updater.add_period(4.5);
        assert!((updater.total_duration_s() - 10.5).abs() < 1e-9);
    }

    #[test]
    fn test_iso_duration_formatting_integer() {
        assert_eq!(format_iso_duration(6.0), "PT6S");
        assert_eq!(format_iso_duration(60.0), "PT60S");
    }

    #[test]
    fn test_iso_duration_formatting_fractional() {
        let s = format_iso_duration(6.5);
        assert!(s.starts_with("PT6.5"), "expected PT6.5... got {s}");
    }

    #[test]
    fn test_period_start_offset_for_second_period() {
        let mut updater = MpdUpdater::new(BASE);
        updater.add_period(6.0);
        updater.add_period(6.0);
        // Second period should have start="PT6S"
        let mpd = updater.build();
        assert!(
            mpd.contains("start=\"PT6S\""),
            "second period must have start offset: {mpd}"
        );
    }

    #[test]
    fn test_no_mpd_close_tag_appends_to_end() {
        let base = "<MPD>";
        let mut updater = MpdUpdater::new(base);
        updater.add_period(6.0);
        let mpd = updater.build();
        assert!(mpd.starts_with("<MPD>"));
        assert!(mpd.contains("period_0"));
    }
}
