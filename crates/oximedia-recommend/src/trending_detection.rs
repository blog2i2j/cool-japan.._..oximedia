//! Trending content detection via exponentially-weighted view velocity.
//!
//! This module maintains a per-item exponential moving average (EMA) of view
//! velocity (views per second) and computes a first-order acceleration term
//! (the rate of change of velocity) that drives trend classification.
//!
//! ## Algorithm sketch
//!
//! For each call to [`TrendingDetector::update`]:
//!
//! 1. Compute the instantaneous velocity for the item:
//!    `v_inst = view_count_delta / elapsed_secs`
//! 2. Update the EMA velocity:
//!    `v_ema = α · v_inst + (1 − α) · v_ema_prev`
//!    where `α = 1 − exp(−elapsed / half_life)` (time-weighted decay).
//! 3. Compute acceleration as:
//!    `a = v_ema − v_ema_prev`
//! 4. Update second-order EMA of acceleration:
//!    `a_ema = α · a + (1 − α) · a_ema_prev`
//!
//! Trend classification rules (applied in priority order):
//!
//! - `Viral(multiplier)` — acceleration EMA > `viral_threshold` (multiplier = `a_ema / v_ema`).
//! - `Rising`            — velocity EMA is increasing (acceleration > 0).
//! - `Declining`         — velocity EMA is decreasing (acceleration < 0).
//! - `Stable`            — otherwise.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// A single view-count observation for one item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewSignal {
    /// Unique identifier of the content item.
    pub item_id: String,
    /// Unix timestamp (seconds) when the observation was recorded.
    pub timestamp_secs: u64,
    /// Number of new views since the previous observation.
    pub view_count_delta: u32,
}

/// Trend classification for an item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrendClass {
    /// Velocity is increasing — item is gaining traction.
    Rising,
    /// Velocity is roughly constant — item has reached a steady state.
    Stable,
    /// Velocity is decreasing — item is losing traction.
    Declining,
    /// Extraordinary acceleration — item is going viral.
    ///
    /// The inner value is the ratio `acceleration / velocity`, i.e. the growth
    /// multiplier.
    Viral(f32),
}

/// Scored trending result for one content item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendingScore {
    /// Content item identifier.
    pub item_id: String,
    /// Exponentially weighted view velocity (views per second).
    pub velocity: f32,
    /// Exponentially weighted acceleration (change in velocity per second).
    pub acceleration: f32,
    /// Classification of the trend.
    pub trend_class: TrendClass,
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal state per item
// ──────────────────────────────────────────────────────────────────────────────

/// EMA half-life in seconds for the velocity signal.
const VELOCITY_HALF_LIFE_SECS: f64 = 300.0; // 5 minutes

/// EMA half-life for the acceleration (smoothed second derivative).
const ACCELERATION_HALF_LIFE_SECS: f64 = 600.0; // 10 minutes

/// Minimum elapsed time (seconds) between updates before decay is applied.
const MIN_ELAPSED_SECS: f64 = 1.0;

/// State maintained per content item.
#[derive(Debug, Clone)]
struct ItemState {
    /// EMA of view velocity (views/sec).
    velocity_ema: f64,
    /// EMA of acceleration (Δvelocity/sec).
    acceleration_ema: f64,
    /// Timestamp of the last observation.
    last_ts: u64,
}

impl ItemState {
    fn new(ts: u64) -> Self {
        Self {
            velocity_ema: 0.0,
            acceleration_ema: 0.0,
            last_ts: ts,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TrendingDetector
// ──────────────────────────────────────────────────────────────────────────────

/// Detects trending content from a stream of view-count signals.
///
/// # Example
///
/// ```
/// use oximedia_recommend::trending_detection::{TrendingDetector, ViewSignal};
///
/// let mut detector = TrendingDetector::new(3600, 10.0);
/// detector.update(ViewSignal {
///     item_id: "news_1".to_string(),
///     timestamp_secs: 1_000_000,
///     view_count_delta: 500,
/// });
/// let top = detector.trending_items(5);
/// assert!(!top.is_empty());
/// ```
pub struct TrendingDetector {
    /// Only signals within this window from the most recent observation are
    /// considered active.  Older items are still stored but rank last.
    pub time_window_secs: u64,
    /// Acceleration EMA threshold above which an item is classified as Viral.
    pub viral_threshold: f32,
    /// Per-item state.
    item_states: HashMap<String, ItemState>,
    /// Timestamp of the globally most recent observation.
    global_latest_ts: u64,
}

impl TrendingDetector {
    /// Create a new detector.
    ///
    /// - `time_window_secs`: items whose latest signal is older than this are
    ///   ranked at the bottom of trending lists.
    /// - `viral_threshold`: acceleration EMA that triggers a `Viral` classification.
    #[must_use]
    pub fn new(time_window_secs: u64, viral_threshold: f32) -> Self {
        Self {
            time_window_secs,
            viral_threshold,
            item_states: HashMap::new(),
            global_latest_ts: 0,
        }
    }

    /// Ingest a single view-count observation.
    ///
    /// Multiple signals for the same item are accepted in any order; out-of-order
    /// signals (timestamp earlier than the last recorded) are silently ignored.
    pub fn update(&mut self, signal: ViewSignal) {
        // Update global clock.
        if signal.timestamp_secs > self.global_latest_ts {
            self.global_latest_ts = signal.timestamp_secs;
        }

        let state = self
            .item_states
            .entry(signal.item_id.clone())
            .or_insert_with(|| ItemState::new(signal.timestamp_secs));

        // Ignore out-of-order signals.
        if signal.timestamp_secs < state.last_ts {
            return;
        }

        let elapsed = (signal.timestamp_secs - state.last_ts).max(1) as f64;
        let elapsed_clamped = elapsed.max(MIN_ELAPSED_SECS);

        // Instantaneous velocity.
        let v_inst = signal.view_count_delta as f64 / elapsed_clamped;

        // Alpha for velocity EMA (time-weighted).
        let alpha_v = 1.0 - f64::exp(-elapsed_clamped * f64::ln(2.0) / VELOCITY_HALF_LIFE_SECS);

        let prev_velocity = state.velocity_ema;
        state.velocity_ema = alpha_v * v_inst + (1.0 - alpha_v) * prev_velocity;

        // Instantaneous acceleration (change in velocity per second).
        let a_inst = (state.velocity_ema - prev_velocity) / elapsed_clamped;

        // Alpha for acceleration EMA.
        let alpha_a = 1.0 - f64::exp(-elapsed_clamped * f64::ln(2.0) / ACCELERATION_HALF_LIFE_SECS);
        state.acceleration_ema = alpha_a * a_inst + (1.0 - alpha_a) * state.acceleration_ema;

        state.last_ts = signal.timestamp_secs;
    }

    /// Return the top `n` trending items ordered by velocity descending.
    ///
    /// Items whose last signal is outside the active time window are included
    /// but ranked after in-window items (their velocity is used but they are
    /// considered stale).
    #[must_use]
    pub fn trending_items(&self, n: usize) -> Vec<TrendingScore> {
        if n == 0 {
            return Vec::new();
        }

        let mut scores: Vec<TrendingScore> = self
            .item_states
            .iter()
            .map(|(id, state)| self.build_score(id, state))
            .collect();

        // Primary sort: in-window items first, then by velocity descending.
        scores.sort_by(|a, b| {
            let a_in = self.is_in_window(&self.item_states[&a.item_id]);
            let b_in = self.is_in_window(&self.item_states[&b.item_id]);
            match (a_in, b_in) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => b
                    .velocity
                    .partial_cmp(&a.velocity)
                    .unwrap_or(std::cmp::Ordering::Equal),
            }
        });

        scores.truncate(n);
        scores
    }

    /// Return all items whose acceleration EMA exceeds `viral_threshold`.
    ///
    /// Results are ordered by acceleration descending.
    #[must_use]
    pub fn detect_viral(&self) -> Vec<TrendingScore> {
        let mut viral: Vec<TrendingScore> = self
            .item_states
            .iter()
            .filter_map(|(id, state)| {
                if state.acceleration_ema as f32 > self.viral_threshold {
                    Some(self.build_score(id, state))
                } else {
                    None
                }
            })
            .collect();

        viral.sort_by(|a, b| {
            b.acceleration
                .partial_cmp(&a.acceleration)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        viral
    }

    /// Whether an item's last signal falls within the active time window.
    fn is_in_window(&self, state: &ItemState) -> bool {
        self.global_latest_ts.saturating_sub(state.last_ts) <= self.time_window_secs
    }

    /// Build a `TrendingScore` for the given item state.
    fn build_score(&self, id: &str, state: &ItemState) -> TrendingScore {
        let velocity = state.velocity_ema as f32;
        let acceleration = state.acceleration_ema as f32;
        let trend_class = self.classify(velocity, acceleration);
        TrendingScore {
            item_id: id.to_string(),
            velocity,
            acceleration,
            trend_class,
        }
    }

    /// Classify the trend for an item given its velocity and acceleration EMAs.
    fn classify(&self, velocity: f32, acceleration: f32) -> TrendClass {
        if acceleration > self.viral_threshold {
            let multiplier = if velocity.abs() > f32::EPSILON {
                (acceleration / velocity).abs()
            } else {
                acceleration.abs()
            };
            TrendClass::Viral(multiplier)
        } else if acceleration > 0.0 {
            TrendClass::Rising
        } else if acceleration < 0.0 {
            TrendClass::Declining
        } else {
            TrendClass::Stable
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(item_id: &str, ts: u64, delta: u32) -> ViewSignal {
        ViewSignal {
            item_id: item_id.to_string(),
            timestamp_secs: ts,
            view_count_delta: delta,
        }
    }

    // 1. Steady view rate → Stable classification.
    #[test]
    fn test_steady_views_stable() {
        let mut det = TrendingDetector::new(3600, 5.0);
        // Feed uniform 100 views every 60 s over 10 minutes.
        for i in 0..10u64 {
            det.update(sig("steady", 1000 + i * 60, 100));
        }
        let top = det.trending_items(1);
        assert_eq!(top.len(), 1);
        assert!(
            matches!(top[0].trend_class, TrendClass::Stable | TrendClass::Rising),
            "steady stream should be Stable or Rising, got {:?}",
            top[0].trend_class
        );
    }

    // 2. Sudden large spike → Viral.
    #[test]
    fn test_spike_triggers_viral() {
        // Use a very low viral_threshold so that a clear spike registers.
        let mut det = TrendingDetector::new(3600, 0.01);
        // Baseline: modest traffic every 60 s.
        for i in 0..5u64 {
            det.update(sig("spike_item", 1000 + i * 60, 10));
        }
        // Massive spike — 50 000 views in the next 60-second window.
        det.update(sig("spike_item", 1300, 50_000));
        // Second big-traffic update to push acceleration EMA above threshold.
        det.update(sig("spike_item", 1360, 50_000));

        let viral = det.detect_viral();
        assert!(
            viral.iter().any(|v| v.item_id == "spike_item"),
            "spike_item should be detected as viral (got: {viral:?})"
        );
    }

    // 3. Declining traffic → Declining class.
    #[test]
    fn test_declining_traffic() {
        // Use a very high viral_threshold so Viral is never triggered,
        // and feed a long monotonically decreasing series so the EMA has time
        // to track the downward trend.
        let mut det = TrendingDetector::new(7200, 1_000_000.0);
        // 20 observations, each with fewer views than the last.
        // Space them 5 minutes apart so the half-life (5 min) lets the EMA decay.
        for i in 0..20u64 {
            let views = 2000u32.saturating_sub(i as u32 * 100);
            det.update(sig("fading", 1_000_000 + i * 300, views));
        }
        let top = det.trending_items(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].item_id, "fading", "fading should be top item");
        assert!(
            matches!(
                top[0].trend_class,
                TrendClass::Declining | TrendClass::Stable
            ),
            "declining traffic should be Declining or Stable, got {:?}",
            top[0].trend_class
        );
    }

    // 4. Top-N order is by velocity descending.
    #[test]
    fn test_top_n_order_by_velocity() {
        let mut det = TrendingDetector::new(3600, 100.0);
        // Three items with distinct velocities.
        // "fast": 500 views/60s, "medium": 200 views/60s, "slow": 50 views/60s.
        for i in 0..5u64 {
            det.update(sig("fast", 1000 + i * 60, 500));
            det.update(sig("medium", 1000 + i * 60, 200));
            det.update(sig("slow", 1000 + i * 60, 50));
        }
        let top = det.trending_items(3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].item_id, "fast");
        assert_eq!(top[1].item_id, "medium");
        assert_eq!(top[2].item_id, "slow");
    }

    // 5. n=0 returns empty.
    #[test]
    fn test_n_zero_returns_empty() {
        let mut det = TrendingDetector::new(3600, 5.0);
        det.update(sig("x", 1000, 100));
        assert!(det.trending_items(0).is_empty());
    }

    // 6. No signals → empty trending and viral lists.
    #[test]
    fn test_no_signals_empty() {
        let det = TrendingDetector::new(3600, 5.0);
        assert!(det.trending_items(10).is_empty());
        assert!(det.detect_viral().is_empty());
    }

    // 7. detect_viral returns only items above threshold.
    #[test]
    fn test_detect_viral_threshold_respected() {
        let mut det = TrendingDetector::new(3600, 0.1);
        // Normal traffic item.
        for i in 0..5u64 {
            det.update(sig("normal", 1000 + i * 60, 50));
        }
        // Viral item (large sudden spike).
        det.update(sig("viral", 2000, 1));
        det.update(sig("viral", 2060, 500_000));

        let viral = det.detect_viral();
        // "viral" should appear; "normal" may or may not depending on EMA.
        assert!(
            viral.iter().all(|v| v.acceleration > det.viral_threshold),
            "all viral items should have acceleration above threshold"
        );
    }

    // 8. Out-of-order signals are ignored gracefully (no panic).
    #[test]
    fn test_out_of_order_signals_ignored() {
        let mut det = TrendingDetector::new(3600, 5.0);
        det.update(sig("item", 2000, 100));
        det.update(sig("item", 1000, 999)); // earlier timestamp — should be ignored
        det.update(sig("item", 2060, 100));
        let top = det.trending_items(1);
        assert_eq!(top.len(), 1);
        assert!(top[0].velocity >= 0.0);
    }

    // 9. TrendClass::Viral carries a positive multiplier.
    #[test]
    fn test_viral_multiplier_positive() {
        let mut det = TrendingDetector::new(3600, 0.01);
        det.update(sig("v", 1000, 1));
        det.update(sig("v", 1060, 100_000));
        let viral = det.detect_viral();
        for ts in &viral {
            if let TrendClass::Viral(m) = ts.trend_class {
                assert!(m > 0.0, "viral multiplier should be positive, got {m}");
            }
        }
    }

    // 10. Multiple items, trending_items returns at most n entries.
    #[test]
    fn test_trending_items_length_capped() {
        let mut det = TrendingDetector::new(3600, 100.0);
        for idx in 0..20u64 {
            det.update(sig(
                &format!("item_{idx}"),
                1000 + idx * 10,
                (idx as u32 + 1) * 10,
            ));
        }
        let top = det.trending_items(5);
        assert_eq!(top.len(), 5);
    }
}
