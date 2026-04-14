//! Weighted retention curves accounting for viewer demographics.
//!
//! Standard retention curves treat every viewer equally, which can obscure
//! important demographic differences.  This module lets callers attach a
//! numeric *weight* to each viewing session (e.g. an inverse-frequency weight
//! for demographic rebalancing, or an importance weight for a specific audience
//! segment) and produces a retention curve in which each second is the
//! *weighted fraction* of viewers who were still watching at that point.
//!
//! ## Key types
//! - [`DemographicWeight`] — a named demographic category and its importance
//!   weight.
//! - [`WeightedSession`] — pairs a [`ViewerSession`] with an explicit weight.
//! - [`WeightedRetentionCurve`] — the per-second weighted-retention result,
//!   together with population-level statistics.
//! - [`WeightedRetentionConfig`] — controls how weights are applied and how
//!   edge cases are handled.
//!
//! ## Algorithm
//! For each second `t` in `[0, content_duration_s)`:
//!
//! ```text
//! weighted_retention(t) = Σ(weight_i * watched_t_i)  /  Σ weight_i
//! ```
//!
//! where `watched_t_i ∈ {0, 1}` is whether viewer `i` had their playhead at
//! second `t`.

use crate::error::AnalyticsError;
use crate::session::{build_playback_map, ViewerSession};

// ─── Demographic descriptor ───────────────────────────────────────────────────

/// A named demographic category paired with a relative importance weight.
///
/// Weights do not need to sum to one — the computation normalises them
/// internally.  A weight of `0.0` means the category is entirely ignored.
#[derive(Debug, Clone, PartialEq)]
pub struct DemographicWeight {
    /// Identifier for the demographic category (e.g. `"18-24"`, `"mobile"`).
    pub category: String,
    /// Non-negative importance weight for this category.
    pub weight: f64,
}

impl DemographicWeight {
    /// Create a new demographic weight entry.
    pub fn new(category: impl Into<String>, weight: f64) -> Self {
        Self {
            category: category.into(),
            weight,
        }
    }
}

// ─── Weighted session ─────────────────────────────────────────────────────────

/// A viewer session together with an explicit sampling / importance weight.
#[derive(Debug, Clone)]
pub struct WeightedSession<'a> {
    /// Reference to the underlying session.
    pub session: &'a ViewerSession,
    /// Non-negative weight applied to this session in the retention computation.
    /// Use `1.0` for uniform (unweighted) treatment.
    pub weight: f64,
    /// Optional demographic category tag for reporting purposes.
    pub demographic: Option<String>,
}

impl<'a> WeightedSession<'a> {
    /// Construct a uniformly-weighted session (weight = 1.0).
    pub fn uniform(session: &'a ViewerSession) -> Self {
        Self {
            session,
            weight: 1.0,
            demographic: None,
        }
    }

    /// Construct a weighted session with an explicit weight.
    pub fn with_weight(session: &'a ViewerSession, weight: f64) -> Self {
        Self {
            session,
            weight,
            demographic: None,
        }
    }

    /// Attach a demographic category label.
    pub fn with_demographic(mut self, category: impl Into<String>) -> Self {
        self.demographic = Some(category.into());
        self
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for weighted retention curve computation.
#[derive(Debug, Clone)]
pub struct WeightedRetentionConfig {
    /// When `true`, weights are normalised so that they sum to 1.0 before
    /// the retention is computed.  When `false`, raw weights are used (the
    /// denominator is the raw weight sum).  Both produce the same retention
    /// values; the option controls whether `total_weight` is exposed as-is or
    /// as 1.0.  Defaults to `true`.
    pub normalise_weights: bool,
    /// Resolution of the retention curve in seconds.  Defaults to 1.
    /// A value of `N` means one retention sample every `N` seconds.
    pub resolution_s: u64,
    /// If `true`, sessions with a weight ≤ 0.0 are silently ignored.
    /// If `false`, they raise [`AnalyticsError::InvalidInput`].
    /// Defaults to `true`.
    pub ignore_zero_weight_sessions: bool,
}

impl Default for WeightedRetentionConfig {
    fn default() -> Self {
        Self {
            normalise_weights: true,
            resolution_s: 1,
            ignore_zero_weight_sessions: true,
        }
    }
}

// ─── Result type ──────────────────────────────────────────────────────────────

/// Per-second weighted retention value.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedRetentionPoint {
    /// Position in seconds from the start of the content.
    pub position_s: u64,
    /// Weighted fraction of viewers still watching (0.0 – 1.0).
    pub retention: f64,
    /// Sum of weights of viewers who were watching at this second.
    pub weighted_viewers: f64,
}

/// Result of a weighted retention curve computation.
#[derive(Debug, Clone)]
pub struct WeightedRetentionCurve {
    /// Per-second (or per-resolution-step) retention values.
    pub points: Vec<WeightedRetentionPoint>,
    /// Total normalised weight of all sessions (1.0 if `normalise_weights`).
    pub total_weight: f64,
    /// Number of sessions included (after filtering zero-weight ones).
    pub session_count: usize,
    /// Weighted average watch time in seconds.
    pub weighted_avg_watch_s: f64,
    /// Weighted completion rate (fraction reaching ≥95 % of content).
    pub weighted_completion_rate: f64,
}

impl WeightedRetentionCurve {
    /// Retention value at the given position in seconds, or `None` if out of range.
    pub fn at_second(&self, position_s: u64) -> Option<f64> {
        self.points
            .iter()
            .find(|p| p.position_s == position_s)
            .map(|p| p.retention)
    }

    /// Identify seconds where retention drops by more than `threshold` compared
    /// to the previous point.  Returns `(position_s, drop_magnitude)` pairs.
    pub fn drop_off_points(&self, threshold: f64) -> Vec<(u64, f64)> {
        self.points
            .windows(2)
            .filter_map(|w| {
                let drop = w[0].retention - w[1].retention;
                if drop > threshold {
                    Some((w[1].position_s, drop))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Weighted retention area: sum of retention values across all points,
    /// normalised by the number of points.  Useful as a single-number summary.
    pub fn area_under_curve(&self) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        let total: f64 = self.points.iter().map(|p| p.retention).sum();
        total / self.points.len() as f64
    }
}

// ─── Core computation ─────────────────────────────────────────────────────────

/// Compute a weighted retention curve from a set of viewer sessions.
///
/// Each session carries an explicit `weight` (inverse-frequency, demographic
/// importance, etc.).  The result is a per-second (or per-resolution-step)
/// weighted fraction of viewers who were still watching.
///
/// # Errors
/// - [`AnalyticsError::InsufficientData`] — no sessions remain after filtering.
/// - [`AnalyticsError::InvalidInput`] — negative weight encountered and
///   `ignore_zero_weight_sessions` is `false`.
/// - [`AnalyticsError::InvalidInput`] — `content_duration_ms` is zero.
pub fn compute_weighted_retention(
    sessions: &[WeightedSession<'_>],
    content_duration_ms: u64,
    config: &WeightedRetentionConfig,
) -> Result<WeightedRetentionCurve, AnalyticsError> {
    if content_duration_ms == 0 {
        return Err(AnalyticsError::InvalidInput(
            "content_duration_ms must be greater than zero".to_string(),
        ));
    }

    let resolution = config.resolution_s.max(1);

    // ── Filter / validate weights ─────────────────────────────────────────────

    // When not ignoring zero/negative weights, check for negatives first.
    if !config.ignore_zero_weight_sessions {
        for ws in sessions {
            if ws.weight < 0.0 {
                return Err(AnalyticsError::InvalidInput(format!(
                    "negative weight {} for session {}",
                    ws.weight, ws.session.session_id
                )));
            }
        }
    }

    let valid_sessions: Vec<&WeightedSession<'_>> = sessions
        .iter()
        .filter(|ws| {
            if config.ignore_zero_weight_sessions {
                ws.weight > 0.0
            } else {
                ws.weight >= 0.0
            }
        })
        .collect();

    if valid_sessions.is_empty() {
        return Err(AnalyticsError::InsufficientData(
            "no sessions with positive weight".to_string(),
        ));
    }

    let raw_weight_sum: f64 = valid_sessions.iter().map(|ws| ws.weight).sum();
    let scale = if config.normalise_weights && raw_weight_sum > 0.0 {
        1.0 / raw_weight_sum
    } else {
        1.0
    };
    let total_weight = if config.normalise_weights {
        1.0
    } else {
        raw_weight_sum
    };

    // ── Build playback maps ───────────────────────────────────────────────────
    let maps_and_weights: Vec<_> = valid_sessions
        .iter()
        .map(|ws| {
            let map = build_playback_map(ws.session, content_duration_ms);
            (map, ws.weight * scale)
        })
        .collect();

    let content_duration_s = (content_duration_ms + 999) / 1000;
    let num_points = content_duration_s.div_ceil(resolution) as usize;

    // ── Per-step retention ────────────────────────────────────────────────────
    let mut points = Vec::with_capacity(num_points);
    let completion_threshold_s = (content_duration_ms as f64 * 0.95 / 1000.0) as usize;

    for step in 0..num_points {
        let position_s = step as u64 * resolution;
        let pos_idx = position_s as usize;

        let weighted_viewers: f64 = maps_and_weights
            .iter()
            .filter_map(|(map, w)| {
                if map.positions_watched.get(pos_idx).copied().unwrap_or(false) {
                    Some(*w)
                } else {
                    None
                }
            })
            .sum();

        let retention = if total_weight > 0.0 {
            weighted_viewers / total_weight
        } else {
            0.0
        };

        points.push(WeightedRetentionPoint {
            position_s,
            retention,
            weighted_viewers,
        });
    }

    // ── Weighted average watch time ───────────────────────────────────────────
    let weighted_avg_watch_s: f64 = maps_and_weights
        .iter()
        .map(|(map, w)| {
            let watched_s = map.positions_watched.iter().filter(|&&v| v).count();
            watched_s as f64 * w
        })
        .sum::<f64>()
        / total_weight;

    // ── Weighted completion rate ──────────────────────────────────────────────
    let weighted_completion_rate: f64 = maps_and_weights
        .iter()
        .map(|(map, w)| {
            let completed = map
                .positions_watched
                .get(completion_threshold_s)
                .copied()
                .unwrap_or(false);
            if completed {
                *w
            } else {
                0.0
            }
        })
        .sum::<f64>()
        / total_weight;

    Ok(WeightedRetentionCurve {
        points,
        total_weight,
        session_count: valid_sessions.len(),
        weighted_avg_watch_s,
        weighted_completion_rate,
    })
}

/// Compute weighted retention curves broken down by demographic category.
///
/// Returns one curve per unique demographic category found in `sessions`,
/// plus an `"all"` aggregate curve that uses all sessions combined.
///
/// Sessions without a demographic label are grouped under `"untagged"`.
pub fn compute_retention_by_demographic(
    sessions: &[WeightedSession<'_>],
    content_duration_ms: u64,
    config: &WeightedRetentionConfig,
) -> Vec<(String, WeightedRetentionCurve)> {
    // Collect unique categories.
    let mut categories: std::collections::HashSet<String> = std::collections::HashSet::new();
    for ws in sessions {
        let cat = ws
            .demographic
            .clone()
            .unwrap_or_else(|| "untagged".to_string());
        categories.insert(cat);
    }

    let mut results: Vec<(String, WeightedRetentionCurve)> = Vec::new();

    // Per-category curves.
    for category in &categories {
        let subset: Vec<_> = sessions
            .iter()
            .filter(|ws| {
                let cat = ws.demographic.as_deref().unwrap_or("untagged");
                cat == category
            })
            .map(|ws| WeightedSession {
                session: ws.session,
                weight: ws.weight,
                demographic: ws.demographic.clone(),
            })
            .collect();

        if let Ok(curve) = compute_weighted_retention(&subset, content_duration_ms, config) {
            results.push((category.clone(), curve));
        }
    }

    // Aggregate "all" curve.
    if let Ok(all_curve) = compute_weighted_retention(sessions, content_duration_ms, config) {
        results.push(("all".to_string(), all_curve));
    }

    results.sort_by(|a, b| a.0.cmp(&b.0));
    results
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{PlaybackEvent, ViewerSession};

    fn make_session(id: &str, content_id: &str, watch_ms: u64) -> ViewerSession {
        ViewerSession {
            session_id: id.to_string(),
            user_id: None,
            content_id: content_id.to_string(),
            started_at_ms: 0,
            events: vec![
                PlaybackEvent::Play { timestamp_ms: 0 },
                PlaybackEvent::End {
                    position_ms: watch_ms,
                    watch_duration_ms: watch_ms,
                },
            ],
        }
    }

    const CONTENT_MS: u64 = 10_000; // 10 seconds

    #[test]
    fn uniform_weight_matches_unweighted_intuition() {
        // All 3 viewers watch the full 10 s → retention should be ~1.0 at second 0.
        let s1 = make_session("s1", "c1", CONTENT_MS);
        let s2 = make_session("s2", "c1", CONTENT_MS);
        let s3 = make_session("s3", "c1", CONTENT_MS);
        let sessions = vec![
            WeightedSession::uniform(&s1),
            WeightedSession::uniform(&s2),
            WeightedSession::uniform(&s3),
        ];
        let config = WeightedRetentionConfig::default();
        let curve = compute_weighted_retention(&sessions, CONTENT_MS, &config)
            .expect("should compute curve");
        assert!(
            curve.points[0].retention > 0.9,
            "retention at t=0 should be high"
        );
        assert_eq!(curve.session_count, 3);
        assert!((curve.total_weight - 1.0).abs() < 1e-10);
    }

    #[test]
    fn higher_weight_dominates_retention() {
        // s1 watches full 10 s with weight=9, s2 watches 2 s with weight=1.
        // At second 5, s1 is watching, s2 is not.
        // Retention at second 5 = 9/(9+1) = 0.9.
        let s1 = make_session("s1", "c1", CONTENT_MS);
        let s2 = make_session("s2", "c1", 2_000); // only 2 s
        let sessions = vec![
            WeightedSession::with_weight(&s1, 9.0),
            WeightedSession::with_weight(&s2, 1.0),
        ];
        let config = WeightedRetentionConfig {
            normalise_weights: false,
            ..Default::default()
        };
        let curve = compute_weighted_retention(&sessions, CONTENT_MS, &config)
            .expect("should compute curve");
        let ret_at_5 = curve.at_second(5).expect("should have second 5");
        assert!((ret_at_5 - 0.9).abs() < 0.05, "ret_at_5={ret_at_5}");
    }

    #[test]
    fn zero_weight_sessions_filtered_by_default() {
        let s1 = make_session("s1", "c1", CONTENT_MS);
        let s2 = make_session("s2", "c1", CONTENT_MS);
        let sessions = vec![
            WeightedSession::with_weight(&s1, 1.0),
            WeightedSession::with_weight(&s2, 0.0), // zero weight → filtered
        ];
        let config = WeightedRetentionConfig::default();
        let curve = compute_weighted_retention(&sessions, CONTENT_MS, &config)
            .expect("should still work after filtering zero-weight session");
        assert_eq!(curve.session_count, 1);
    }

    #[test]
    fn all_zero_weight_sessions_returns_error() {
        let s1 = make_session("s1", "c1", CONTENT_MS);
        let sessions = vec![WeightedSession::with_weight(&s1, 0.0)];
        let config = WeightedRetentionConfig::default();
        let result = compute_weighted_retention(&sessions, CONTENT_MS, &config);
        assert!(result.is_err());
    }

    #[test]
    fn zero_content_duration_returns_error() {
        let s1 = make_session("s1", "c1", 5_000);
        let sessions = vec![WeightedSession::uniform(&s1)];
        let config = WeightedRetentionConfig::default();
        let result = compute_weighted_retention(&sessions, 0, &config);
        assert!(result.is_err());
    }

    #[test]
    fn drop_off_points_detected() {
        // s1 watches full 10 s, s2 drops off after 3 s (both weight=1).
        let s1 = make_session("s1", "c1", CONTENT_MS);
        let s2 = make_session("s2", "c1", 3_000);
        let sessions = vec![WeightedSession::uniform(&s1), WeightedSession::uniform(&s2)];
        let config = WeightedRetentionConfig::default();
        let curve = compute_weighted_retention(&sessions, CONTENT_MS, &config).expect("curve");
        // At second 3 → 0.5 retention, second 4 → 0.5 (s1 only), so no big drop.
        // The boundary where s2 drops creates a visible drop.
        let drops = curve.drop_off_points(0.3);
        // At least one drop should be detected near second 3-4.
        assert!(!drops.is_empty(), "expected at least one drop-off point");
    }

    #[test]
    fn area_under_curve_full_retention() {
        // If everyone watches everything, AUC ≈ 1.0.
        let s1 = make_session("s1", "c1", CONTENT_MS);
        let s2 = make_session("s2", "c1", CONTENT_MS);
        let sessions = vec![WeightedSession::uniform(&s1), WeightedSession::uniform(&s2)];
        let config = WeightedRetentionConfig::default();
        let curve = compute_weighted_retention(&sessions, CONTENT_MS, &config).expect("curve");
        let auc = curve.area_under_curve();
        assert!(auc > 0.8, "auc={auc}");
    }

    #[test]
    fn demographic_breakdown_produces_per_group_curves() {
        let s1 = make_session("s1", "c1", CONTENT_MS);
        let s2 = make_session("s2", "c1", 5_000);
        let s3 = make_session("s3", "c1", CONTENT_MS);
        let sessions = vec![
            WeightedSession::uniform(&s1).with_demographic("18-24"),
            WeightedSession::uniform(&s2).with_demographic("25-34"),
            WeightedSession::uniform(&s3).with_demographic("18-24"),
        ];
        let config = WeightedRetentionConfig::default();
        let breakdown = compute_retention_by_demographic(&sessions, CONTENT_MS, &config);
        // Should have "18-24", "25-34", and "all".
        let category_names: Vec<&str> = breakdown.iter().map(|(c, _)| c.as_str()).collect();
        assert!(category_names.contains(&"18-24"));
        assert!(category_names.contains(&"25-34"));
        assert!(category_names.contains(&"all"));
    }

    #[test]
    fn resolution_reduces_number_of_points() {
        let s1 = make_session("s1", "c1", CONTENT_MS);
        let sessions_1s = vec![WeightedSession::uniform(&s1)];
        let sessions_2s = vec![WeightedSession::uniform(&s1)];
        let config_1s = WeightedRetentionConfig {
            resolution_s: 1,
            ..Default::default()
        };
        let config_2s = WeightedRetentionConfig {
            resolution_s: 2,
            ..Default::default()
        };
        let curve_1s =
            compute_weighted_retention(&sessions_1s, CONTENT_MS, &config_1s).expect("1s curve");
        let curve_2s =
            compute_weighted_retention(&sessions_2s, CONTENT_MS, &config_2s).expect("2s curve");
        assert!(
            curve_1s.points.len() > curve_2s.points.len(),
            "1s resolution should yield more points than 2s"
        );
    }

    #[test]
    fn weighted_avg_watch_time_plausible() {
        // s1 watches 10 s (weight 3), s2 watches 5 s (weight 1).
        // Weighted avg = (3*10 + 1*5) / (3+1) = 35/4 = 8.75 s.
        let s1 = make_session("s1", "c1", CONTENT_MS);
        let s2 = make_session("s2", "c1", 5_000);
        let sessions = vec![
            WeightedSession::with_weight(&s1, 3.0),
            WeightedSession::with_weight(&s2, 1.0),
        ];
        let config = WeightedRetentionConfig {
            normalise_weights: false,
            ..Default::default()
        };
        let curve = compute_weighted_retention(&sessions, CONTENT_MS, &config).expect("curve");
        // The playback map-based watch time may differ slightly from the End-event
        // watch_duration_ms due to rounding to seconds; allow some tolerance.
        assert!(
            curve.weighted_avg_watch_s > 5.0 && curve.weighted_avg_watch_s <= 10.0,
            "weighted_avg_watch_s={}",
            curve.weighted_avg_watch_s
        );
    }
}
