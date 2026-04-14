//! Comprehensive SLA (Service Level Agreement) tracking with tier definitions,
//! uptime calculation, breach detection, and reporting-period aggregation.
//!
//! This module goes beyond the basic [`crate::sla`] primitives and provides
//! production-grade SLA management:
//!
//! - **`SlaTier`** — named service tiers (Bronze / Silver / Gold / Platinum) each with
//!   a defined minimum uptime percentage and maximum allowed downtime budget.
//! - **`UptimeWindow`** — a contiguous reporting period (e.g. calendar month) that
//!   accumulates "up" and "down" intervals and computes the achieved uptime.
//! - **`SlaBreachEvent`** — a recorded breach with its start/end timestamps, affected
//!   tier, and the delta between achieved and required uptime.
//! - **`SlaTracker`** — the top-level manager that ties everything together: register
//!   tier assignments per service, feed interval observations, detect breaches,
//!   and generate per-period SLA reports.

#![allow(dead_code)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// SLA Tier Definitions
// ---------------------------------------------------------------------------

/// Named SLA tier used to categorise services by their contractual commitments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SlaTier {
    /// Bronze — 99.0 % monthly uptime (~7.3 h downtime/month).
    Bronze,
    /// Silver — 99.5 % monthly uptime (~3.65 h downtime/month).
    Silver,
    /// Gold — 99.9 % monthly uptime (~43.8 min downtime/month).
    Gold,
    /// Platinum — 99.99 % monthly uptime (~4.38 min downtime/month).
    Platinum,
}

impl SlaTier {
    /// Required uptime percentage (0–100) for this tier.
    #[must_use]
    pub fn required_uptime_pct(self) -> f64 {
        match self {
            Self::Bronze => 99.0,
            Self::Silver => 99.5,
            Self::Gold => 99.9,
            Self::Platinum => 99.99,
        }
    }

    /// Allowed downtime budget in seconds for a 30-day calendar month.
    #[must_use]
    pub fn monthly_downtime_budget_secs(self) -> f64 {
        let month_secs = 30.0 * 24.0 * 3600.0;
        month_secs * (1.0 - self.required_uptime_pct() / 100.0)
    }

    /// Human-readable tier name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Bronze => "Bronze",
            Self::Silver => "Silver",
            Self::Gold => "Gold",
            Self::Platinum => "Platinum",
        }
    }

    /// Returns `true` if `achieved_pct` satisfies this tier's requirement.
    #[must_use]
    pub fn is_satisfied(self, achieved_pct: f64) -> bool {
        achieved_pct >= self.required_uptime_pct()
    }
}

impl std::fmt::Display for SlaTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// Uptime Window
// ---------------------------------------------------------------------------

/// A contiguous reporting window (e.g. a calendar month) during which uptime
/// observations are accumulated.
///
/// Time is represented as seconds since an arbitrary epoch (e.g. Unix epoch).
/// Each call to [`UptimeWindow::record_interval`] adds either "up" or "down"
/// time to the window's running totals.
#[derive(Debug, Clone)]
pub struct UptimeWindow {
    /// Inclusive start of the reporting window (seconds).
    pub start_secs: f64,
    /// Exclusive end of the reporting window (seconds).
    pub end_secs: f64,
    /// Total seconds classified as "up" within this window.
    up_secs: f64,
    /// Total seconds classified as "down" within this window.
    down_secs: f64,
}

impl UptimeWindow {
    /// Create a new empty window spanning `[start_secs, end_secs)`.
    ///
    /// # Panics (debug only)
    ///
    /// Does not panic; if `start_secs >= end_secs` the window is treated as
    /// having zero duration and will always report 100 % uptime.
    #[must_use]
    pub fn new(start_secs: f64, end_secs: f64) -> Self {
        Self {
            start_secs,
            end_secs: end_secs.max(start_secs),
            up_secs: 0.0,
            down_secs: 0.0,
        }
    }

    /// Total window duration in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        (self.end_secs - self.start_secs).max(0.0)
    }

    /// Record an interval `[interval_start, interval_end)` as either up or down.
    ///
    /// The interval is clipped to the window boundaries before being added.
    pub fn record_interval(&mut self, interval_start: f64, interval_end: f64, is_up: bool) {
        let clipped_start = interval_start.max(self.start_secs);
        let clipped_end = interval_end.min(self.end_secs);
        let duration = (clipped_end - clipped_start).max(0.0);
        if is_up {
            self.up_secs += duration;
        } else {
            self.down_secs += duration;
        }
    }

    /// Achieved uptime percentage (0–100) for this window.
    ///
    /// Returns 100.0 if the window has zero duration.
    #[must_use]
    pub fn uptime_pct(&self) -> f64 {
        let total = self.duration_secs();
        if total < 1e-12 {
            return 100.0;
        }
        // Up seconds may not cover the full window; uncovered time is treated
        // as unmeasured and counted as "up" (benefit of the doubt).
        let measured = self.up_secs + self.down_secs;
        if measured < 1e-12 {
            return 100.0;
        }
        (self.up_secs / measured * 100.0).clamp(0.0, 100.0)
    }

    /// Downtime in seconds accumulated in this window.
    #[must_use]
    pub fn down_secs(&self) -> f64 {
        self.down_secs
    }

    /// Uptime in seconds accumulated in this window.
    #[must_use]
    pub fn up_secs(&self) -> f64 {
        self.up_secs
    }

    /// Remaining downtime budget given a target tier.
    ///
    /// Positive values mean budget still available; negative means over-budget.
    #[must_use]
    pub fn remaining_downtime_budget(&self, tier: SlaTier) -> f64 {
        let budget = self.duration_secs() * (1.0 - tier.required_uptime_pct() / 100.0);
        budget - self.down_secs
    }
}

// ---------------------------------------------------------------------------
// SLA Breach Event
// ---------------------------------------------------------------------------

/// A recorded SLA breach within a reporting window.
#[derive(Debug, Clone)]
pub struct SlaBreachEvent {
    /// Identifier of the affected service.
    pub service_id: String,
    /// The tier whose requirement was breached.
    pub tier: SlaTier,
    /// Start of the reporting period in which the breach occurred (seconds).
    pub period_start_secs: f64,
    /// End of the reporting period (seconds).
    pub period_end_secs: f64,
    /// The uptime percentage that was actually achieved.
    pub achieved_pct: f64,
    /// The uptime percentage required by the tier.
    pub required_pct: f64,
    /// When the breach was detected (seconds since epoch).
    pub detected_at_secs: f64,
}

impl SlaBreachEvent {
    /// Shortfall: how many percentage points below the requirement.
    #[must_use]
    pub fn shortfall_pct(&self) -> f64 {
        (self.required_pct - self.achieved_pct).max(0.0)
    }

    /// Returns `true` if the shortfall is greater than zero.
    #[must_use]
    pub fn is_breach(&self) -> bool {
        self.achieved_pct < self.required_pct
    }
}

// ---------------------------------------------------------------------------
// SLA Report
// ---------------------------------------------------------------------------

/// Aggregated SLA report for a single service over a single reporting period.
#[derive(Debug, Clone)]
pub struct SlaReport {
    /// Identifier of the service.
    pub service_id: String,
    /// Contracted tier.
    pub tier: SlaTier,
    /// The reporting window this report covers.
    pub window: UptimeWindow,
    /// Whether the SLA was met.
    pub sla_met: bool,
    /// All breach events recorded during this window.
    pub breaches: Vec<SlaBreachEvent>,
}

impl SlaReport {
    /// Achieved uptime percentage.
    #[must_use]
    pub fn uptime_pct(&self) -> f64 {
        self.window.uptime_pct()
    }

    /// Number of breach events recorded.
    #[must_use]
    pub fn breach_count(&self) -> usize {
        self.breaches.len()
    }
}

// ---------------------------------------------------------------------------
// Per-service State
// ---------------------------------------------------------------------------

/// Internal state for a single tracked service.
#[derive(Debug)]
struct ServiceState {
    tier: SlaTier,
    /// Accumulating window for the current reporting period.
    current_window: UptimeWindow,
    /// Historical SLA reports (one per closed period).
    history: Vec<SlaReport>,
    /// Breach events accumulated so far in the current period.
    current_breaches: Vec<SlaBreachEvent>,
}

// ---------------------------------------------------------------------------
// SlaTracker
// ---------------------------------------------------------------------------

/// Top-level SLA tracker.
///
/// Manages SLA tier assignments for multiple services, accepts uptime/downtime
/// interval observations, detects breaches, and produces per-period reports.
///
/// # Usage
///
/// ```
/// use oximedia_monitor::sla_tracker::{SlaTracker, SlaTier};
///
/// let mut tracker = SlaTracker::new();
/// tracker.register_service("video-encoder", SlaTier::Gold, 0.0, 86_400.0);
///
/// // Record that the service was up for the first 23 hours.
/// tracker.record_interval("video-encoder", 0.0, 82_800.0, true);
/// // Record 1 hour of downtime.
/// tracker.record_interval("video-encoder", 82_800.0, 86_400.0, false);
///
/// let report = tracker.close_period("video-encoder", 1.0).unwrap();
/// println!("Uptime: {:.3}%  SLA met: {}", report.uptime_pct(), report.sla_met);
/// ```
#[derive(Debug)]
pub struct SlaTracker {
    services: HashMap<String, ServiceState>,
}

impl SlaTracker {
    /// Create a new, empty SLA tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            services: HashMap::new(),
        }
    }

    /// Register a service with a given tier and initial reporting window.
    ///
    /// If the service is already registered its tier is updated and a new
    /// current window is started.
    pub fn register_service(
        &mut self,
        service_id: impl Into<String>,
        tier: SlaTier,
        window_start_secs: f64,
        window_end_secs: f64,
    ) {
        let id = service_id.into();
        let state = ServiceState {
            tier,
            current_window: UptimeWindow::new(window_start_secs, window_end_secs),
            history: Vec::new(),
            current_breaches: Vec::new(),
        };
        self.services.insert(id, state);
    }

    /// Record an up/down interval for a service.
    ///
    /// Returns `false` if the service is not registered.
    pub fn record_interval(
        &mut self,
        service_id: &str,
        interval_start: f64,
        interval_end: f64,
        is_up: bool,
    ) -> bool {
        if let Some(state) = self.services.get_mut(service_id) {
            state
                .current_window
                .record_interval(interval_start, interval_end, is_up);
            true
        } else {
            false
        }
    }

    /// Close the current reporting period for a service, generate a report,
    /// archive it in history, and start a new window immediately following.
    ///
    /// `detected_at_secs` is the wall-clock time at which you are closing the
    /// period (used to timestamp any breach events).
    ///
    /// Returns `None` if the service is not registered.
    pub fn close_period(
        &mut self,
        service_id: &str,
        detected_at_secs: f64,
    ) -> Option<SlaReport> {
        let state = self.services.get_mut(service_id)?;
        let achieved = state.current_window.uptime_pct();
        let required = state.tier.required_uptime_pct();
        let sla_met = achieved >= required;

        // Collect any breach that occurred in this window.
        let mut breaches = std::mem::take(&mut state.current_breaches);
        if !sla_met {
            breaches.push(SlaBreachEvent {
                service_id: service_id.to_string(),
                tier: state.tier,
                period_start_secs: state.current_window.start_secs,
                period_end_secs: state.current_window.end_secs,
                achieved_pct: achieved,
                required_pct: required,
                detected_at_secs,
            });
        }

        let window = state.current_window.clone();
        let report = SlaReport {
            service_id: service_id.to_string(),
            tier: state.tier,
            window: window.clone(),
            sla_met,
            breaches: breaches.clone(),
        };

        state.history.push(report.clone());

        // Start a fresh window for the next period, using the previous end as
        // the new start.  End is set to 2× the previous window's duration away.
        let prev_duration = window.duration_secs();
        let new_start = window.end_secs;
        let new_end = new_start + prev_duration.max(1.0);
        state.current_window = UptimeWindow::new(new_start, new_end);
        state.current_breaches = Vec::new();

        Some(report)
    }

    /// Return a snapshot report for the current (still-open) period without
    /// closing it.
    ///
    /// Returns `None` if the service is not registered.
    #[must_use]
    pub fn current_report(&self, service_id: &str) -> Option<SlaReport> {
        let state = self.services.get(service_id)?;
        let achieved = state.current_window.uptime_pct();
        let required = state.tier.required_uptime_pct();
        Some(SlaReport {
            service_id: service_id.to_string(),
            tier: state.tier,
            window: state.current_window.clone(),
            sla_met: achieved >= required,
            breaches: state.current_breaches.clone(),
        })
    }

    /// Number of historical closed periods for a service.
    #[must_use]
    pub fn closed_period_count(&self, service_id: &str) -> usize {
        self.services
            .get(service_id)
            .map_or(0, |s| s.history.len())
    }

    /// Retrieve all closed-period reports for a service (oldest first).
    #[must_use]
    pub fn history(&self, service_id: &str) -> Vec<&SlaReport> {
        self.services
            .get(service_id)
            .map_or_else(Vec::new, |s| s.history.iter().collect())
    }

    /// Number of registered services.
    #[must_use]
    pub fn service_count(&self) -> usize {
        self.services.len()
    }

    /// Returns `true` if the service is registered.
    #[must_use]
    pub fn contains_service(&self, service_id: &str) -> bool {
        self.services.contains_key(service_id)
    }

    /// Return the tier assigned to a service.
    #[must_use]
    pub fn tier_of(&self, service_id: &str) -> Option<SlaTier> {
        self.services.get(service_id).map(|s| s.tier)
    }

    /// Compute aggregate breach statistics across all services and all periods.
    ///
    /// Returns a `(total_periods, total_breaches)` tuple.
    #[must_use]
    pub fn aggregate_stats(&self) -> (usize, usize) {
        let mut total_periods = 0usize;
        let mut total_breaches = 0usize;
        for state in self.services.values() {
            total_periods += state.history.len();
            total_breaches += state.history.iter().filter(|r| !r.sla_met).count();
        }
        (total_periods, total_breaches)
    }

    /// Return a flat list of all breach events across all services and periods.
    #[must_use]
    pub fn all_breaches(&self) -> Vec<&SlaBreachEvent> {
        self.services
            .values()
            .flat_map(|s| s.history.iter().flat_map(|r| r.breaches.iter()))
            .collect()
    }
}

impl Default for SlaTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Convenience: 30-day window in seconds.
    const MONTH: f64 = 30.0 * 24.0 * 3600.0;

    fn make_tracker_with_gold() -> SlaTracker {
        let mut t = SlaTracker::new();
        t.register_service("svc", SlaTier::Gold, 0.0, MONTH);
        t
    }

    // --- SlaTier tests ---

    #[test]
    fn test_tier_ordering() {
        assert!(SlaTier::Bronze < SlaTier::Platinum);
        assert!(SlaTier::Gold < SlaTier::Platinum);
    }

    #[test]
    fn test_tier_required_uptime() {
        assert!((SlaTier::Gold.required_uptime_pct() - 99.9).abs() < 1e-9);
        assert!((SlaTier::Platinum.required_uptime_pct() - 99.99).abs() < 1e-9);
    }

    #[test]
    fn test_tier_monthly_budget() {
        // Gold: 99.9 % → 0.1 % downtime → ~2592 s in 30 days
        let budget = SlaTier::Gold.monthly_downtime_budget_secs();
        assert!((budget - 2592.0).abs() < 1.0);
    }

    #[test]
    fn test_tier_is_satisfied() {
        assert!(SlaTier::Gold.is_satisfied(99.95));
        assert!(!SlaTier::Gold.is_satisfied(99.8));
    }

    #[test]
    fn test_tier_display() {
        assert_eq!(SlaTier::Platinum.to_string(), "Platinum");
    }

    // --- UptimeWindow tests ---

    #[test]
    fn test_window_uptime_pct_all_up() {
        let mut w = UptimeWindow::new(0.0, 3600.0);
        w.record_interval(0.0, 3600.0, true);
        assert!((w.uptime_pct() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_window_uptime_pct_half_up() {
        let mut w = UptimeWindow::new(0.0, 3600.0);
        w.record_interval(0.0, 1800.0, true);
        w.record_interval(1800.0, 3600.0, false);
        assert!((w.uptime_pct() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_window_clip_to_boundaries() {
        let mut w = UptimeWindow::new(1000.0, 2000.0);
        // Interval extends beyond window; should be clipped to window width.
        w.record_interval(500.0, 2500.0, true);
        assert!((w.up_secs() - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn test_window_empty_is_100pct() {
        let w = UptimeWindow::new(0.0, 3600.0);
        assert!((w.uptime_pct() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_window_remaining_budget_positive() {
        let mut w = UptimeWindow::new(0.0, MONTH);
        // Only 100 s down; Gold budget is ~2592 s
        w.record_interval(0.0, 100.0, false);
        w.record_interval(100.0, MONTH, true);
        let remaining = w.remaining_downtime_budget(SlaTier::Gold);
        assert!(remaining > 0.0, "should have budget remaining; got {remaining}");
    }

    #[test]
    fn test_window_remaining_budget_negative() {
        let mut w = UptimeWindow::new(0.0, MONTH);
        // 10 000 s down; well above Gold budget of ~2592 s
        w.record_interval(0.0, 10_000.0, false);
        w.record_interval(10_000.0, MONTH, true);
        let remaining = w.remaining_downtime_budget(SlaTier::Gold);
        assert!(remaining < 0.0, "should be over budget; got {remaining}");
    }

    // --- SlaTracker tests ---

    #[test]
    fn test_tracker_register_and_count() {
        let mut t = SlaTracker::new();
        t.register_service("a", SlaTier::Gold, 0.0, MONTH);
        t.register_service("b", SlaTier::Silver, 0.0, MONTH);
        assert_eq!(t.service_count(), 2);
        assert!(t.contains_service("a"));
    }

    #[test]
    fn test_tracker_tier_of() {
        let t = make_tracker_with_gold();
        assert_eq!(t.tier_of("svc"), Some(SlaTier::Gold));
        assert_eq!(t.tier_of("missing"), None);
    }

    #[test]
    fn test_tracker_full_uptime_no_breach() {
        let mut t = make_tracker_with_gold();
        t.record_interval("svc", 0.0, MONTH, true);
        let report = t.close_period("svc", MONTH).expect("close should succeed");
        assert!(report.sla_met);
        assert_eq!(report.breach_count(), 0);
        assert!((report.uptime_pct() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_tracker_downtime_exceeds_budget_causes_breach() {
        let mut t = make_tracker_with_gold();
        // Record 10 000 s downtime (> Gold budget of ~2592 s).
        t.record_interval("svc", 0.0, 10_000.0, false);
        t.record_interval("svc", 10_000.0, MONTH, true);
        let report = t.close_period("svc", MONTH).expect("close should succeed");
        assert!(!report.sla_met);
        assert_eq!(report.breach_count(), 1);
        assert!(report.breaches[0].shortfall_pct() > 0.0);
    }

    #[test]
    fn test_tracker_history_accumulates() {
        let mut t = make_tracker_with_gold();
        t.record_interval("svc", 0.0, MONTH, true);
        t.close_period("svc", MONTH).expect("close period 1");
        // Next period (auto-started).
        t.record_interval("svc", MONTH, 2.0 * MONTH, true);
        t.close_period("svc", 2.0 * MONTH).expect("close period 2");
        assert_eq!(t.closed_period_count("svc"), 2);
    }

    #[test]
    fn test_tracker_current_report_does_not_close() {
        let mut t = make_tracker_with_gold();
        t.record_interval("svc", 0.0, 1000.0, false);
        let snap = t.current_report("svc").expect("current_report should succeed");
        assert!(snap.uptime_pct() < 100.0);
        // Period is still open — history is empty.
        assert_eq!(t.closed_period_count("svc"), 0);
    }

    #[test]
    fn test_tracker_aggregate_stats() {
        let mut t = SlaTracker::new();
        t.register_service("a", SlaTier::Gold, 0.0, MONTH);
        t.register_service("b", SlaTier::Bronze, 0.0, MONTH);
        // Service a: full uptime — no breach.
        t.record_interval("a", 0.0, MONTH, true);
        t.close_period("a", MONTH).expect("close a");
        // Service b: massive downtime — breach.
        t.record_interval("b", 0.0, MONTH, false);
        t.close_period("b", MONTH).expect("close b");

        let (periods, breaches) = t.aggregate_stats();
        assert_eq!(periods, 2);
        assert_eq!(breaches, 1);
    }

    #[test]
    fn test_tracker_all_breaches() {
        let mut t = make_tracker_with_gold();
        t.record_interval("svc", 0.0, MONTH, false);
        t.close_period("svc", MONTH).expect("close");
        let breaches = t.all_breaches();
        assert_eq!(breaches.len(), 1);
        assert_eq!(breaches[0].tier, SlaTier::Gold);
    }

    #[test]
    fn test_tracker_record_unknown_service_returns_false() {
        let mut t = SlaTracker::new();
        let ok = t.record_interval("ghost", 0.0, 100.0, true);
        assert!(!ok);
    }

    #[test]
    fn test_breach_shortfall_calculation() {
        let breach = SlaBreachEvent {
            service_id: "svc".to_string(),
            tier: SlaTier::Gold,
            period_start_secs: 0.0,
            period_end_secs: MONTH,
            achieved_pct: 99.5,
            required_pct: 99.9,
            detected_at_secs: MONTH,
        };
        assert!((breach.shortfall_pct() - 0.4).abs() < 1e-9);
        assert!(breach.is_breach());
    }

    #[test]
    fn test_close_period_unknown_service_returns_none() {
        let mut t = SlaTracker::new();
        assert!(t.close_period("ghost", 0.0).is_none());
    }
}
