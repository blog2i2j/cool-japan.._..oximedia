//! Engagement scoring model for media content.
//!
//! Computes a weighted engagement score from viewer session data, models score
//! trends over time with linear regression, and ranks content by engagement.

use crate::session::{build_playback_map, PlaybackEvent, ViewerSession};

// ─── Score model ──────────────────────────────────────────────────────────────

/// Decomposed components of an engagement score (each in 0.0 – 1.0).
#[derive(Debug, Clone, PartialEq)]
pub struct EngagementComponents {
    /// Ratio of average watch time to content duration (capped at 1.0).
    pub watch_time_score: f32,
    /// Fraction of sessions that reached ≥95 % completion.
    pub completion_score: f32,
    /// Fraction of sessions that rewatched any segment.
    pub rewatch_score: f32,
    /// Placeholder for social interaction data (always 0.5 until social data
    /// is available in sessions).
    pub social_score: f32,
    /// Penalty term proportional to the forward-seek rate (lower is better).
    pub seek_forward_penalty: f32,
}

/// Weights controlling the relative importance of each engagement component.
#[derive(Debug, Clone, PartialEq)]
pub struct EngagementWeights {
    pub watch_time: f32,
    pub completion: f32,
    pub rewatch: f32,
    pub social: f32,
    /// Multiplicative penalty factor for forward seeks.  A value of 1.0 means
    /// each forward seek as a fraction of total events subtracts directly from
    /// the score.
    pub forward_seek_penalty: f32,
}

impl EngagementWeights {
    /// All five components equally weighted at 0.2.
    pub fn default() -> Self {
        Self {
            watch_time: 0.2,
            completion: 0.2,
            rewatch: 0.2,
            social: 0.2,
            forward_seek_penalty: 0.2,
        }
    }
}

/// Final engagement score for a piece of content.
#[derive(Debug, Clone, PartialEq)]
pub struct ContentEngagementScore {
    pub content_id: String,
    /// Overall score in 0.0 – 1.0.
    pub score: f32,
    pub components: EngagementComponents,
}

// ─── Core computation ─────────────────────────────────────────────────────────

/// Compute an engagement score for a content item from its viewer sessions.
///
/// Returns a score of `0.0` when `sessions` is empty or `content_duration_ms`
/// is zero.  The `content_id` is taken from the first session's `content_id`.
pub fn compute_engagement(
    sessions: &[ViewerSession],
    content_duration_ms: u64,
    weights: &EngagementWeights,
) -> ContentEngagementScore {
    let content_id = sessions
        .first()
        .map(|s| s.content_id.clone())
        .unwrap_or_default();

    if sessions.is_empty() || content_duration_ms == 0 {
        return ContentEngagementScore {
            content_id,
            score: 0.0,
            components: EngagementComponents {
                watch_time_score: 0.0,
                completion_score: 0.0,
                rewatch_score: 0.0,
                social_score: 0.5,
                seek_forward_penalty: 0.0,
            },
        };
    }

    let n = sessions.len() as f64;
    let completion_threshold_ms = (content_duration_ms as f64 * 0.95) as u64;

    let mut total_watch_ms: u64 = 0;
    let mut completion_count: u32 = 0;
    let mut rewatch_count: u32 = 0;
    let mut total_events: u32 = 0;
    let mut forward_seek_count: u32 = 0;

    for session in sessions {
        // Watch time: prefer the End event's watch_duration_ms.
        let session_watch_ms = session.events.iter().fold(0u64, |acc, e| match e {
            PlaybackEvent::End {
                watch_duration_ms, ..
            } => acc.max(*watch_duration_ms),
            _ => acc,
        });
        total_watch_ms += session_watch_ms;

        // Completion: did the session reach ≥ 95 % of the content?
        let map = build_playback_map(session, content_duration_ms);
        let completion_sec = (completion_threshold_ms / 1000) as usize;
        if map
            .positions_watched
            .get(completion_sec)
            .copied()
            .unwrap_or(false)
        {
            completion_count += 1;
        }

        // Rewatch: any second watched more than once means the session included a seek-back.
        // We detect this by checking for backward seek events.
        let has_rewatch = session
            .events
            .iter()
            .any(|e| matches!(e, PlaybackEvent::Seek { from_ms, to_ms } if to_ms < from_ms));
        if has_rewatch {
            rewatch_count += 1;
        }

        // Forward seek penalty.
        for event in &session.events {
            total_events += 1;
            if let PlaybackEvent::Seek { from_ms, to_ms } = event {
                if to_ms > from_ms {
                    forward_seek_count += 1;
                }
            }
        }
    }

    let avg_watch_ms = total_watch_ms as f64 / n;
    let watch_time_score = (avg_watch_ms / content_duration_ms as f64).min(1.0) as f32;
    let completion_score = completion_count as f32 / sessions.len() as f32;
    let rewatch_score = rewatch_count as f32 / sessions.len() as f32;
    let social_score: f32 = 0.5; // placeholder

    let seek_forward_penalty = if total_events > 0 {
        forward_seek_count as f32 / total_events as f32
    } else {
        0.0
    };

    // Weighted score:
    //   score = w_watch * watch_time_score
    //         + w_completion * completion_score
    //         + w_rewatch * rewatch_score
    //         + w_social * social_score
    //         - w_penalty * seek_forward_penalty
    // Clamped to [0.0, 1.0].
    let raw_score = weights.watch_time * watch_time_score
        + weights.completion * completion_score
        + weights.rewatch * rewatch_score
        + weights.social * social_score
        - weights.forward_seek_penalty * seek_forward_penalty;

    let score = raw_score.max(0.0).min(1.0);

    ContentEngagementScore {
        content_id,
        score,
        components: EngagementComponents {
            watch_time_score,
            completion_score,
            rewatch_score,
            social_score,
            seek_forward_penalty,
        },
    }
}

// ─── Trend analysis ───────────────────────────────────────────────────────────

/// A time-series of engagement scores for a content item.
#[derive(Debug, Clone)]
pub struct EngagementTrend {
    /// Pairs of (timestamp_ms, engagement_score).
    pub scores_over_time: Vec<(i64, f32)>,
}

impl EngagementTrend {
    /// Compute the linear-regression slope of the score series.
    ///
    /// Returns `0.0` if the series has fewer than two points or if the
    /// denominator is zero.
    pub fn slope(&self) -> f32 {
        linear_regression_slope(&self.scores_over_time)
    }
}

/// Compute the least-squares linear regression slope of the given (x, y) data.
///
/// `slope = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)`
///
/// Returns `0.0` when the denominator is zero (all x values identical) or when
/// there are fewer than two data points.
pub fn linear_regression_slope(points: &[(i64, f32)]) -> f32 {
    let n = points.len();
    if n < 2 {
        return 0.0;
    }

    // Use f64 for numerical stability with large timestamp values.
    let n_f = n as f64;
    let mut sum_x: f64 = 0.0;
    let mut sum_y: f64 = 0.0;
    let mut sum_xy: f64 = 0.0;
    let mut sum_x2: f64 = 0.0;

    for &(x, y) in points {
        let xf = x as f64;
        let yf = y as f64;
        sum_x += xf;
        sum_y += yf;
        sum_xy += xf * yf;
        sum_x2 += xf * xf;
    }

    let denom = n_f * sum_x2 - sum_x * sum_x;
    if denom.abs() < f64::EPSILON {
        return 0.0;
    }

    ((n_f * sum_xy - sum_x * sum_y) / denom) as f32
}

// ─── Time-series decomposition ────────────────────────────────────────────────

/// A period used for seasonal decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeasonalPeriod {
    /// 7-day weekly seasonality.
    Weekly,
    /// 30-day monthly seasonality.
    Monthly,
    /// Custom period length (number of observations per cycle).
    Custom(usize),
}

impl SeasonalPeriod {
    /// Return the integer period length (number of observations per cycle).
    pub fn length(&self) -> usize {
        match self {
            SeasonalPeriod::Weekly => 7,
            SeasonalPeriod::Monthly => 30,
            SeasonalPeriod::Custom(n) => *n,
        }
    }
}

/// Result of additive time-series decomposition: y = trend + seasonal + residual.
///
/// All three components have the same length as the input series.
#[derive(Debug, Clone)]
pub struct DecomposedSeries {
    /// Smoothed trend component (centered moving average).
    pub trend: Vec<f64>,
    /// Seasonal component (mean deviation for each seasonal phase).
    pub seasonal: Vec<f64>,
    /// Residual = observed − trend − seasonal.
    pub residual: Vec<f64>,
    /// Original observed values.
    pub observed: Vec<f64>,
    /// Period used for decomposition.
    pub period: usize,
}

/// Decompose a time-series into trend + seasonal + residual components.
///
/// Uses classical additive decomposition (STL-style but without LOESS):
///
/// 1. **Trend**: centered moving average with window = `period`.
/// 2. **Seasonal**: for each phase position in [0, period), compute the mean
///    of `(observed − trend)` across all cycles; then centre by subtracting
///    the mean of the seasonal indices.
/// 3. **Residual**: `observed − trend − seasonal`.
///
/// For positions at the edges of the series where the centered moving average
/// cannot be computed, the trend is interpolated linearly.
///
/// Returns `None` when the series has fewer than `2 * period` points.
pub fn decompose_time_series(
    series: &[(i64, f32)],
    period: SeasonalPeriod,
) -> Option<DecomposedSeries> {
    let n = series.len();
    let p = period.length();
    if p == 0 || n < 2 * p {
        return None;
    }

    let y: Vec<f64> = series.iter().map(|&(_, v)| v as f64).collect();

    // ── Step 1: Centered moving average (trend) ───────────────────────────────
    let half = p / 2;
    let mut trend = vec![f64::NAN; n];

    for i in half..n.saturating_sub(half) {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);
        let window = &y[start..end];
        trend[i] = window.iter().sum::<f64>() / window.len() as f64;
    }

    // Interpolate NaN edges linearly from the first/last computed values.
    if let Some(first_valid) = trend.iter().position(|v| !v.is_nan()) {
        let val = trend[first_valid];
        for i in 0..first_valid {
            trend[i] = val;
        }
    }
    if let Some(last_valid) = trend.iter().rposition(|v| !v.is_nan()) {
        let val = trend[last_valid];
        for i in (last_valid + 1)..n {
            trend[i] = val;
        }
    }
    // Linear interpolation between known valid points (fill interior NaNs).
    let mut start = None;
    for i in 0..n {
        if !trend[i].is_nan() {
            if let Some(s) = start {
                // Interpolate from s to i.
                let t_s = trend[s];
                let t_e = trend[i];
                for j in (s + 1)..i {
                    let t = (j - s) as f64 / (i - s) as f64;
                    trend[j] = t_s + t * (t_e - t_s);
                }
                start = None;
            }
        } else if start.is_none() {
            start = Some(if i == 0 { 0 } else { i - 1 });
        }
    }

    // ── Step 2: Seasonal indices ──────────────────────────────────────────────
    // detrended[i] = y[i] − trend[i]
    let detrended: Vec<f64> = y
        .iter()
        .zip(trend.iter())
        .map(|(&yi, &ti)| yi - ti)
        .collect();

    // Average detrended values for each phase position.
    let mut phase_sums = vec![0.0f64; p];
    let mut phase_counts = vec![0u32; p];
    for (i, &d) in detrended.iter().enumerate() {
        let phase = i % p;
        phase_sums[phase] += d;
        phase_counts[phase] += 1;
    }
    let mut phase_means: Vec<f64> = phase_sums
        .iter()
        .zip(phase_counts.iter())
        .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
        .collect();

    // Centre seasonal indices so they sum to zero.
    let phase_mean: f64 = phase_means.iter().sum::<f64>() / p as f64;
    for v in &mut phase_means {
        *v -= phase_mean;
    }

    let seasonal: Vec<f64> = (0..n).map(|i| phase_means[i % p]).collect();

    // ── Step 3: Residual ──────────────────────────────────────────────────────
    let residual: Vec<f64> = y
        .iter()
        .zip(trend.iter())
        .zip(seasonal.iter())
        .map(|((&yi, &ti), &si)| yi - ti - si)
        .collect();

    Some(DecomposedSeries {
        trend,
        seasonal,
        residual,
        observed: y,
        period: p,
    })
}

// ─── Exponential Moving Average ───────────────────────────────────────────────

/// Smoothing factor and configuration for exponential moving average (EMA).
///
/// EMA: `EMA(0) = y(0)`, `EMA(i) = alpha * y(i) + (1 - alpha) * EMA(i-1)`.
#[derive(Debug, Clone, PartialEq)]
pub struct EmaConfig {
    /// Smoothing factor `alpha ∈ (0.0, 1.0]`.
    pub alpha: f64,
}

impl EmaConfig {
    /// Build from explicit `alpha`. Returns `None` when `alpha ∉ (0.0, 1.0]`.
    pub fn with_alpha(alpha: f64) -> Option<Self> {
        if alpha > 0.0 && alpha <= 1.0 {
            Some(Self { alpha })
        } else {
            None
        }
    }

    /// Build from span N using `alpha = 2 / (N + 1)`. Returns `None` for span 0.
    pub fn from_span(span: usize) -> Option<Self> {
        if span == 0 {
            return None;
        }
        Some(Self {
            alpha: 2.0 / (span as f64 + 1.0),
        })
    }
}

impl Default for EmaConfig {
    fn default() -> Self {
        Self { alpha: 0.2 }
    }
}

/// Result of an EMA computation over an engagement score time-series.
#[derive(Debug, Clone)]
pub struct EmaResult {
    /// EMA-smoothed values aligned 1-to-1 with the input series.
    pub smoothed: Vec<f64>,
    /// The smoothing factor `alpha` applied.
    pub alpha: f64,
    /// Linear-regression slope of the smoothed series.
    pub trend_slope: f64,
}

impl EmaResult {
    /// Most recent smoothed value.
    pub fn last_smoothed(&self) -> f64 {
        self.smoothed.last().copied().unwrap_or(0.0)
    }

    /// First smoothed value (seeded from the first observation).
    pub fn first_smoothed(&self) -> f64 {
        self.smoothed.first().copied().unwrap_or(0.0)
    }

    /// Infer the trend direction from the EMA's slope.
    pub fn trend_direction(&self, epsilon: f64) -> TrendDirection {
        TrendDirection::from_slope(self.trend_slope, epsilon)
    }
}

/// Trend direction inferred from slope analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    /// Score is growing over time.
    Growing,
    /// Score is declining over time.
    Declining,
    /// No discernible trend.
    Flat,
}

impl TrendDirection {
    /// Classify a slope value.
    pub fn from_slope(slope: f64, epsilon: f64) -> Self {
        if slope > epsilon {
            Self::Growing
        } else if slope < -epsilon {
            Self::Declining
        } else {
            Self::Flat
        }
    }
}

/// Compute the exponential moving average of an engagement score series.
///
/// Returns `None` when the series is empty or `alpha ∉ (0.0, 1.0]`.
pub fn exponential_moving_average(series: &[(i64, f32)], config: &EmaConfig) -> Option<EmaResult> {
    if series.is_empty() || config.alpha <= 0.0 || config.alpha > 1.0 {
        return None;
    }

    let alpha = config.alpha;
    let one_minus = 1.0 - alpha;

    let mut smoothed = Vec::with_capacity(series.len());
    let mut prev = f64::from(series[0].1);
    smoothed.push(prev);

    for &(_, y) in &series[1..] {
        let ema = alpha * f64::from(y) + one_minus * prev;
        smoothed.push(ema);
        prev = ema;
    }

    let indexed: Vec<(i64, f32)> = smoothed
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as i64, v as f32))
        .collect();
    let trend_slope = f64::from(linear_regression_slope(&indexed));

    Some(EmaResult {
        smoothed,
        alpha,
        trend_slope,
    })
}

// ─── Ranking ──────────────────────────────────────────────────────────────────

/// Ranks and recommends content items by their engagement score.
pub struct ContentRanker;

impl ContentRanker {
    /// Sort `scores` by engagement descending and return `(content_id, score)`
    /// pairs.
    pub fn rank_by_engagement<'a>(scores: &'a [ContentEngagementScore]) -> Vec<(&'a str, f32)> {
        let mut ranked: Vec<_> = scores
            .iter()
            .map(|s| (s.content_id.as_str(), s.score))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{PlaybackEvent, ViewerSession};

    fn full_watch_session(id: &str, content_ms: u64) -> ViewerSession {
        ViewerSession {
            session_id: id.to_string(),
            user_id: None,
            content_id: "content_a".to_string(),
            started_at_ms: 0,
            events: vec![
                PlaybackEvent::Play { timestamp_ms: 0 },
                PlaybackEvent::End {
                    position_ms: content_ms,
                    watch_duration_ms: content_ms,
                },
            ],
        }
    }

    fn partial_watch_session(id: &str, watch_ms: u64, _content_ms: u64) -> ViewerSession {
        ViewerSession {
            session_id: id.to_string(),
            user_id: None,
            content_id: "content_a".to_string(),
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

    fn session_with_forward_seek(id: &str, content_ms: u64) -> ViewerSession {
        ViewerSession {
            session_id: id.to_string(),
            user_id: None,
            content_id: "content_a".to_string(),
            started_at_ms: 0,
            events: vec![
                PlaybackEvent::Play { timestamp_ms: 0 },
                PlaybackEvent::Seek {
                    from_ms: 3000,
                    to_ms: 7000,
                },
                PlaybackEvent::End {
                    position_ms: content_ms,
                    watch_duration_ms: content_ms / 2,
                },
            ],
        }
    }

    fn session_with_backward_seek(id: &str, content_ms: u64) -> ViewerSession {
        ViewerSession {
            session_id: id.to_string(),
            user_id: None,
            content_id: "content_a".to_string(),
            started_at_ms: 0,
            events: vec![
                PlaybackEvent::Play { timestamp_ms: 0 },
                PlaybackEvent::Seek {
                    from_ms: 7000,
                    to_ms: 3000,
                },
                PlaybackEvent::End {
                    position_ms: content_ms,
                    watch_duration_ms: content_ms,
                },
            ],
        }
    }

    // ── compute_engagement ───────────────────────────────────────────────────

    #[test]
    fn engagement_empty_sessions() {
        let weights = EngagementWeights::default();
        let score = compute_engagement(&[], 10_000, &weights);
        assert_eq!(score.score, 0.0);
    }

    #[test]
    fn engagement_zero_duration() {
        let sessions = vec![full_watch_session("s1", 10_000)];
        let weights = EngagementWeights::default();
        let score = compute_engagement(&sessions, 0, &weights);
        assert_eq!(score.score, 0.0);
    }

    #[test]
    fn engagement_full_watch_high_score() {
        let sessions: Vec<_> = (0..10)
            .map(|i| full_watch_session(&format!("s{i}"), 10_000))
            .collect();
        let weights = EngagementWeights::default();
        let score = compute_engagement(&sessions, 10_000, &weights);
        // watch_time=1.0, completion=1.0, rewatch=0.0, social=0.5, penalty=0.0
        // = 0.2*1 + 0.2*1 + 0.2*0 + 0.2*0.5 - 0.2*0 = 0.5
        assert!((score.score - 0.5).abs() < 0.05, "score={}", score.score);
    }

    #[test]
    fn engagement_partial_watch_lower_score() {
        let sessions: Vec<_> = (0..10)
            .map(|i| partial_watch_session(&format!("s{i}"), 3_000, 10_000))
            .collect();
        let weights = EngagementWeights::default();
        let full = compute_engagement(
            &(0..10)
                .map(|i| full_watch_session(&format!("s{i}"), 10_000))
                .collect::<Vec<_>>(),
            10_000,
            &weights,
        );
        let partial = compute_engagement(&sessions, 10_000, &weights);
        assert!(
            partial.score < full.score,
            "partial={} full={}",
            partial.score,
            full.score
        );
    }

    #[test]
    fn engagement_components_watch_time_capped() {
        // Watch time = 2x content duration → capped at 1.0.
        let sessions = vec![partial_watch_session("s1", 20_000, 10_000)];
        let weights = EngagementWeights::default();
        let score = compute_engagement(&sessions, 10_000, &weights);
        assert!(score.components.watch_time_score <= 1.0);
    }

    #[test]
    fn engagement_rewatch_detected() {
        let sessions = vec![session_with_backward_seek("s1", 10_000)];
        let weights = EngagementWeights::default();
        let score = compute_engagement(&sessions, 10_000, &weights);
        assert!((score.components.rewatch_score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn engagement_forward_seek_penalty() {
        let no_seek: Vec<_> = (0..5)
            .map(|i| full_watch_session(&format!("s{i}"), 10_000))
            .collect();
        let with_seek: Vec<_> = (0..5)
            .map(|i| session_with_forward_seek(&format!("s{i}"), 10_000))
            .collect();
        let weights = EngagementWeights::default();
        let score_clean = compute_engagement(&no_seek, 10_000, &weights);
        let score_seeky = compute_engagement(&with_seek, 10_000, &weights);
        assert!(
            score_seeky.score <= score_clean.score,
            "seeky={} clean={}",
            score_seeky.score,
            score_clean.score
        );
    }

    #[test]
    fn engagement_social_score_placeholder() {
        let sessions = vec![full_watch_session("s1", 5_000)];
        let weights = EngagementWeights::default();
        let score = compute_engagement(&sessions, 5_000, &weights);
        assert!((score.components.social_score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn engagement_content_id_from_first_session() {
        let sessions = vec![full_watch_session("s1", 10_000)];
        let weights = EngagementWeights::default();
        let score = compute_engagement(&sessions, 10_000, &weights);
        assert_eq!(score.content_id, "content_a");
    }

    #[test]
    fn engagement_weights_default_sum_to_one() {
        let w = EngagementWeights::default();
        let sum = w.watch_time + w.completion + w.rewatch + w.social + w.forward_seek_penalty;
        assert!((sum - 1.0).abs() < 1e-6);
    }

    // ── linear_regression_slope ──────────────────────────────────────────────

    #[test]
    fn slope_perfectly_increasing() {
        // y = x (in tiny units): (0,0.0),(1,1.0),(2,2.0),(3,3.0)
        let points = vec![(0i64, 0.0f32), (1, 1.0), (2, 2.0), (3, 3.0)];
        let slope = linear_regression_slope(&points);
        assert!((slope - 1.0).abs() < 1e-4, "slope={slope}");
    }

    #[test]
    fn slope_perfectly_decreasing() {
        let points = vec![(0i64, 3.0f32), (1, 2.0), (2, 1.0), (3, 0.0)];
        let slope = linear_regression_slope(&points);
        assert!((slope + 1.0).abs() < 1e-4, "slope={slope}");
    }

    #[test]
    fn slope_flat() {
        let points = vec![(0i64, 0.5f32), (1, 0.5), (2, 0.5), (3, 0.5)];
        let slope = linear_regression_slope(&points);
        assert!(slope.abs() < 1e-6, "slope={slope}");
    }

    #[test]
    fn slope_single_point_returns_zero() {
        let points = vec![(100i64, 0.8f32)];
        assert_eq!(linear_regression_slope(&points), 0.0);
    }

    #[test]
    fn slope_two_points() {
        let points = vec![(0i64, 0.0f32), (10, 1.0)];
        let slope = linear_regression_slope(&points);
        assert!((slope - 0.1).abs() < 1e-5, "slope={slope}");
    }

    #[test]
    fn engagement_trend_slope_method() {
        let trend = EngagementTrend {
            scores_over_time: vec![(0, 0.3), (1_000, 0.6), (2_000, 0.9)],
        };
        let slope = trend.slope();
        assert!(slope > 0.0, "expected positive slope, got {slope}");
    }

    // ── ContentRanker ────────────────────────────────────────────────────────

    #[test]
    fn ranker_sorted_descending() {
        let scores = vec![
            ContentEngagementScore {
                content_id: "a".to_string(),
                score: 0.4,
                components: EngagementComponents {
                    watch_time_score: 0.4,
                    completion_score: 0.4,
                    rewatch_score: 0.0,
                    social_score: 0.5,
                    seek_forward_penalty: 0.0,
                },
            },
            ContentEngagementScore {
                content_id: "b".to_string(),
                score: 0.9,
                components: EngagementComponents {
                    watch_time_score: 0.9,
                    completion_score: 0.9,
                    rewatch_score: 0.1,
                    social_score: 0.5,
                    seek_forward_penalty: 0.0,
                },
            },
            ContentEngagementScore {
                content_id: "c".to_string(),
                score: 0.6,
                components: EngagementComponents {
                    watch_time_score: 0.6,
                    completion_score: 0.6,
                    rewatch_score: 0.0,
                    social_score: 0.5,
                    seek_forward_penalty: 0.0,
                },
            },
        ];
        let ranked = ContentRanker::rank_by_engagement(&scores);
        assert_eq!(ranked[0].0, "b");
        assert_eq!(ranked[1].0, "c");
        assert_eq!(ranked[2].0, "a");
    }

    #[test]
    fn ranker_empty_input() {
        let ranked = ContentRanker::rank_by_engagement(&[]);
        assert!(ranked.is_empty());
    }

    #[test]
    fn ranker_single_item() {
        let scores = vec![ContentEngagementScore {
            content_id: "only".to_string(),
            score: 0.7,
            components: EngagementComponents {
                watch_time_score: 0.7,
                completion_score: 0.7,
                rewatch_score: 0.0,
                social_score: 0.5,
                seek_forward_penalty: 0.0,
            },
        }];
        let ranked = ContentRanker::rank_by_engagement(&scores);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].0, "only");
    }

    // ── EMA tests ────────────────────────────────────────────────────────────

    #[test]
    fn ema_empty_series_returns_none() {
        assert!(exponential_moving_average(&[], &EmaConfig::default()).is_none());
    }

    #[test]
    fn ema_alpha_one_equals_original_series() {
        // alpha=1.0 → EMA(i) = y(i).
        let config = EmaConfig::with_alpha(1.0).expect("valid");
        let series = vec![(0i64, 0.1f32), (1, 0.5), (2, 0.9), (3, 0.3)];
        let result = exponential_moving_average(&series, &config).expect("result");
        assert_eq!(result.smoothed.len(), series.len());
        for (i, &(_, y)) in series.iter().enumerate() {
            // f64::from(f32) then back: use 1e-6 tolerance for f32 → f64 conversion.
            assert!(
                (result.smoothed[i] - f64::from(y)).abs() < 1e-6,
                "index {i}: ema={} y={}",
                result.smoothed[i],
                y
            );
        }
    }

    #[test]
    fn ema_smooths_noisy_signal() {
        let series: Vec<(i64, f32)> = (0i64..20)
            .map(|i| (i, if i % 2 == 0 { 0.9 } else { 0.1 }))
            .collect();
        let config = EmaConfig::from_span(5).expect("valid span");
        let result = exponential_moving_average(&series, &config).expect("result");
        let last = result.last_smoothed();
        assert!(
            last > 0.2 && last < 0.8,
            "smoothed last={last} should be near 0.5"
        );
    }

    #[test]
    fn ema_seeded_with_first_observation() {
        // seed = y(0) = 0.7; second EMA = 0.5 * 0.1 + 0.5 * 0.7 = 0.4
        let series = vec![(0i64, 0.7f32), (1, 0.1)];
        let config = EmaConfig::with_alpha(0.5).expect("valid");
        let result = exponential_moving_average(&series, &config).expect("result");
        // f64::from(0.7f32) is ~0.699999988; use 1e-6 tolerance.
        assert!(
            (result.first_smoothed() - f64::from(0.7f32)).abs() < 1e-9,
            "first_smoothed={} expected {}",
            result.first_smoothed(),
            f64::from(0.7f32)
        );
        // EMA[1] = 0.5 * f64::from(0.1f32) + 0.5 * f64::from(0.7f32)
        let expected = 0.5 * f64::from(0.1f32) + 0.5 * f64::from(0.7f32);
        assert!(
            (result.smoothed[1] - expected).abs() < 1e-9,
            "smoothed[1]={} expected {expected}",
            result.smoothed[1]
        );
    }

    #[test]
    fn ema_from_span_produces_valid_alpha() {
        let config = EmaConfig::from_span(9).expect("valid");
        assert!((config.alpha - 0.2).abs() < 1e-12);
    }

    #[test]
    fn ema_from_span_zero_returns_none() {
        assert!(EmaConfig::from_span(0).is_none());
    }

    #[test]
    fn ema_with_invalid_alpha_returns_none() {
        assert!(EmaConfig::with_alpha(0.0).is_none());
        assert!(EmaConfig::with_alpha(-0.1).is_none());
        assert!(EmaConfig::with_alpha(1.1).is_none());
    }

    #[test]
    fn ema_trend_slope_positive_for_growing_series() {
        let series: Vec<(i64, f32)> = (0i64..10).map(|i| (i, i as f32 * 0.1)).collect();
        let config = EmaConfig::with_alpha(0.3).expect("valid");
        let result = exponential_moving_average(&series, &config).expect("result");
        assert!(result.trend_slope > 0.0, "slope={}", result.trend_slope);
        assert_eq!(result.trend_direction(1e-6), TrendDirection::Growing);
    }

    #[test]
    fn ema_trend_direction_declining() {
        let series: Vec<(i64, f32)> = (0i64..10).map(|i| (i, 1.0f32 - i as f32 * 0.1)).collect();
        let config = EmaConfig::with_alpha(0.3).expect("valid");
        let result = exponential_moving_average(&series, &config).expect("result");
        assert_eq!(result.trend_direction(1e-6), TrendDirection::Declining);
    }

    #[test]
    fn ema_trend_direction_flat_for_constant_series() {
        let series: Vec<(i64, f32)> = (0i64..10).map(|i| (i, 0.5f32)).collect();
        let config = EmaConfig::with_alpha(0.3).expect("valid");
        let result = exponential_moving_average(&series, &config).expect("result");
        assert_eq!(result.trend_direction(1e-6), TrendDirection::Flat);
    }

    #[test]
    fn ema_result_alpha_stored_correctly() {
        let series = vec![(0i64, 0.5f32), (1, 0.6)];
        let config = EmaConfig::with_alpha(0.4).expect("valid");
        let result = exponential_moving_average(&series, &config).expect("result");
        assert!((result.alpha - 0.4).abs() < 1e-12);
    }
}
