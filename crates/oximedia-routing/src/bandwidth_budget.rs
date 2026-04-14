#![allow(dead_code)]

//! Bandwidth budgeting for media routing.
//!
//! Tracks available link bandwidth and allocates capacity to media
//! streams, preventing over-subscription on routing links.
//!
//! # Threshold Warnings
//!
//! `BandwidthBudget` supports utilization threshold monitoring via
//! `UtilizationThreshold` and `BandwidthAlert`. Register an alert
//! callback with `on_alert()`; it is fired once on each state
//! transition (`Normal → Warning → Critical` and back), not on every
//! check.

use std::sync::Arc;

/// Unit of bandwidth measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandwidthUnit {
    /// Bits per second.
    Bps,
    /// Kilobits per second.
    Kbps,
    /// Megabits per second.
    Mbps,
    /// Gigabits per second.
    Gbps,
}

impl BandwidthUnit {
    /// Convert a value in this unit to bits per second.
    #[allow(clippy::cast_precision_loss)]
    pub fn to_bps(self, value: f64) -> f64 {
        match self {
            Self::Bps => value,
            Self::Kbps => value * 1_000.0,
            Self::Mbps => value * 1_000_000.0,
            Self::Gbps => value * 1_000_000_000.0,
        }
    }

    /// Convert bits per second to this unit.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_bps(self, bps: f64) -> f64 {
        match self {
            Self::Bps => bps,
            Self::Kbps => bps / 1_000.0,
            Self::Mbps => bps / 1_000_000.0,
            Self::Gbps => bps / 1_000_000_000.0,
        }
    }
}

/// An allocation request for a stream.
#[derive(Debug, Clone)]
pub struct AllocationRequest {
    /// Stream identifier.
    pub stream_id: String,
    /// Required bandwidth in bits per second.
    pub required_bps: f64,
    /// Priority (lower number = higher priority).
    pub priority: u32,
}

impl AllocationRequest {
    /// Create a new allocation request.
    pub fn new(stream_id: impl Into<String>, required_bps: f64, priority: u32) -> Self {
        Self {
            stream_id: stream_id.into(),
            required_bps,
            priority,
        }
    }
}

/// Outcome of an allocation attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocationResult {
    /// Successfully allocated.
    Granted,
    /// Not enough bandwidth remaining.
    Denied,
    /// Already allocated for this stream.
    AlreadyAllocated,
}

/// Record of an active allocation.
#[derive(Debug, Clone)]
struct ActiveAllocation {
    stream_id: String,
    allocated_bps: f64,
    priority: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Threshold / Alert types
// ─────────────────────────────────────────────────────────────────────────────

/// Utilization thresholds that trigger bandwidth alerts.
///
/// Percentages are in the range 0.0–100.0.
#[derive(Debug, Clone, Copy)]
pub struct UtilizationThreshold {
    /// Utilization percentage (0–100) at which a `Warning` alert fires.
    pub warning_pct: f64,
    /// Utilization percentage (0–100) at which a `Critical` alert fires.
    pub critical_pct: f64,
}

impl Default for UtilizationThreshold {
    fn default() -> Self {
        Self {
            warning_pct: 80.0,
            critical_pct: 95.0,
        }
    }
}

/// An alert emitted when link utilization crosses a threshold boundary.
#[derive(Debug, Clone)]
pub enum BandwidthAlert {
    /// Utilization crossed the warning threshold.
    Warning {
        /// Link identifier.
        link_id: String,
        /// Current utilization percentage.
        utilization_pct: f64,
    },
    /// Utilization crossed the critical threshold.
    Critical {
        /// Link identifier.
        link_id: String,
        /// Current utilization percentage.
        utilization_pct: f64,
    },
    /// Utilization dropped below the warning threshold — alert cleared.
    Cleared {
        /// Link identifier.
        link_id: String,
    },
}

/// Callback type for bandwidth alerts.
///
/// Uses `Arc` so the callback can be cloned alongside the budget struct.
pub type AlertCallback = Arc<dyn Fn(BandwidthAlert) + Send + Sync>;

// ─────────────────────────────────────────────────────────────────────────────
// Alert state (for hysteresis)
// ─────────────────────────────────────────────────────────────────────────────

/// Internal alert state used for hysteresis: only fire on transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum AlertState {
    /// Utilization is below the warning threshold.
    #[default]
    Normal,
    /// Utilization is at or above the warning threshold but below critical.
    Warning,
    /// Utilization is at or above the critical threshold.
    Critical,
}

// ─────────────────────────────────────────────────────────────────────────────
// BandwidthBudget
// ─────────────────────────────────────────────────────────────────────────────

/// Manages a fixed bandwidth budget for a link or path.
///
/// `Clone` is derived; the alert callback is shared via `Arc`.
#[derive(Clone)]
pub struct BandwidthBudget {
    /// Identifier used in alert payloads.
    link_id: String,
    /// Total capacity in bps.
    capacity_bps: f64,
    /// Active allocations.
    allocations: Vec<ActiveAllocation>,
    /// Optional utilization thresholds.
    threshold: Option<UtilizationThreshold>,
    /// Optional alert callback (Arc for Clone + Send + Sync).
    alert_callback: Option<AlertCallback>,
    /// Hysteresis state — only emit on state transitions.
    alert_state: AlertState,
}

impl std::fmt::Debug for BandwidthBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BandwidthBudget")
            .field("link_id", &self.link_id)
            .field("capacity_bps", &self.capacity_bps)
            .field("allocations", &self.allocations)
            .field("threshold", &self.threshold)
            .field(
                "alert_callback",
                &self.alert_callback.as_ref().map(|_| "<callback>"),
            )
            .field("alert_state", &self.alert_state)
            .finish()
    }
}

impl BandwidthBudget {
    /// Create a new budget with the given capacity.
    pub fn new(capacity: f64, unit: BandwidthUnit) -> Self {
        Self::with_id("", capacity, unit)
    }

    /// Create a new budget with the given capacity and link identifier.
    pub fn with_id(link_id: impl Into<String>, capacity: f64, unit: BandwidthUnit) -> Self {
        Self {
            link_id: link_id.into(),
            capacity_bps: unit.to_bps(capacity),
            allocations: Vec::new(),
            threshold: None,
            alert_callback: None,
            alert_state: AlertState::Normal,
        }
    }

    /// Total capacity in bps.
    pub fn capacity_bps(&self) -> f64 {
        self.capacity_bps
    }

    /// Currently allocated bandwidth in bps.
    pub fn allocated_bps(&self) -> f64 {
        self.allocations.iter().map(|a| a.allocated_bps).sum()
    }

    /// Remaining available bandwidth in bps.
    pub fn available_bps(&self) -> f64 {
        (self.capacity_bps - self.allocated_bps()).max(0.0)
    }

    /// Utilization as a fraction 0.0..=1.0.
    #[allow(clippy::cast_precision_loss)]
    pub fn utilization(&self) -> f64 {
        if self.capacity_bps <= 0.0 {
            return 0.0;
        }
        (self.allocated_bps() / self.capacity_bps).min(1.0)
    }

    /// Utilization as a percentage 0.0..=100.0.
    pub fn utilization_pct(&self) -> f64 {
        self.utilization() * 100.0
    }

    /// Number of active allocations.
    pub fn allocation_count(&self) -> usize {
        self.allocations.len()
    }

    /// Configure utilization thresholds for alert firing.
    pub fn set_thresholds(&mut self, threshold: UtilizationThreshold) {
        self.threshold = Some(threshold);
        // Re-evaluate current state without firing (just update internal state).
        self.alert_state = self.compute_alert_state(self.utilization_pct());
    }

    /// Register an alert callback.  The callback will be called once on each
    /// threshold state transition.
    pub fn on_alert(&mut self, callback: AlertCallback) {
        self.alert_callback = Some(callback);
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Determine the new alert state given a utilization percentage and the
    /// current threshold configuration.
    fn compute_alert_state(&self, pct: f64) -> AlertState {
        match &self.threshold {
            None => AlertState::Normal,
            Some(t) if pct >= t.critical_pct => AlertState::Critical,
            Some(t) if pct >= t.warning_pct => AlertState::Warning,
            _ => AlertState::Normal,
        }
    }

    /// Compare the new alert state against the stored one and, on a transition,
    /// fire the callback.
    fn check_and_fire_alert(&mut self) {
        let pct = self.utilization_pct();
        let new_state = self.compute_alert_state(pct);

        if new_state == self.alert_state {
            return; // no transition — hysteresis prevents repeated firing
        }

        self.alert_state = new_state;

        if let Some(cb) = &self.alert_callback {
            let alert = match new_state {
                AlertState::Normal => BandwidthAlert::Cleared {
                    link_id: self.link_id.clone(),
                },
                AlertState::Warning => BandwidthAlert::Warning {
                    link_id: self.link_id.clone(),
                    utilization_pct: pct,
                },
                AlertState::Critical => BandwidthAlert::Critical {
                    link_id: self.link_id.clone(),
                    utilization_pct: pct,
                },
            };
            cb(alert);
        }
    }

    // ── Public allocation API ─────────────────────────────────────────────────

    /// Try to allocate bandwidth for a stream.
    pub fn allocate(&mut self, request: &AllocationRequest) -> AllocationResult {
        // Check for duplicate
        if self
            .allocations
            .iter()
            .any(|a| a.stream_id == request.stream_id)
        {
            return AllocationResult::AlreadyAllocated;
        }

        if request.required_bps > self.available_bps() {
            return AllocationResult::Denied;
        }

        self.allocations.push(ActiveAllocation {
            stream_id: request.stream_id.clone(),
            allocated_bps: request.required_bps,
            priority: request.priority,
        });

        self.check_and_fire_alert();
        AllocationResult::Granted
    }

    /// Release an allocation by stream id. Returns the released bps or `None`.
    pub fn release(&mut self, stream_id: &str) -> Option<f64> {
        if let Some(pos) = self
            .allocations
            .iter()
            .position(|a| a.stream_id == stream_id)
        {
            let removed = self.allocations.remove(pos);
            self.check_and_fire_alert();
            Some(removed.allocated_bps)
        } else {
            None
        }
    }

    /// Release all allocations.
    pub fn release_all(&mut self) {
        self.allocations.clear();
        self.check_and_fire_alert();
    }

    /// Check if a stream has an active allocation.
    pub fn is_allocated(&self, stream_id: &str) -> bool {
        self.allocations.iter().any(|a| a.stream_id == stream_id)
    }

    /// Try to preempt the lowest-priority allocation to make room.
    /// Returns the preempted stream id if successful.
    pub fn preempt_lowest_priority(&mut self, request: &AllocationRequest) -> Option<String> {
        if self.allocations.is_empty() {
            return None;
        }

        // Find lowest priority (highest number) that is lower priority than request
        let worst_idx = self
            .allocations
            .iter()
            .enumerate()
            .filter(|(_, a)| a.priority > request.priority)
            .max_by_key(|(_, a)| a.priority)
            .map(|(i, _)| i);

        if let Some(idx) = worst_idx {
            let freed_bps = self.allocations[idx].allocated_bps;
            let freed_id = self.allocations[idx].stream_id.clone();
            self.allocations.remove(idx);

            // Now check if there's room
            if request.required_bps <= self.available_bps() {
                self.allocations.push(ActiveAllocation {
                    stream_id: request.stream_id.clone(),
                    allocated_bps: request.required_bps,
                    priority: request.priority,
                });
                self.check_and_fire_alert();
                return Some(freed_id);
            }
            // Put it back if still not enough room
            let restore_priority = self.allocations.len() as u32;
            self.allocations.push(ActiveAllocation {
                stream_id: freed_id,
                allocated_bps: freed_bps,
                priority: restore_priority,
            });
        }
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BudgetAllocator
// ─────────────────────────────────────────────────────────────────────────────

/// Allocator that manages budgets across multiple links.
#[derive(Clone)]
pub struct BudgetAllocator {
    budgets: Vec<(String, BandwidthBudget)>,
}

impl std::fmt::Debug for BudgetAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BudgetAllocator")
            .field(
                "budgets",
                &self.budgets.iter().map(|(n, _)| n).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl BudgetAllocator {
    /// Create a new allocator.
    pub fn new() -> Self {
        Self {
            budgets: Vec::new(),
        }
    }

    /// Add a link budget.
    pub fn add_link(&mut self, name: impl Into<String>, budget: BandwidthBudget) {
        let name = name.into();
        self.budgets.push((name, budget));
    }

    /// Number of managed links.
    pub fn link_count(&self) -> usize {
        self.budgets.len()
    }

    /// Get a budget by link name.
    pub fn get_budget(&self, name: &str) -> Option<&BandwidthBudget> {
        self.budgets.iter().find(|(n, _)| n == name).map(|(_, b)| b)
    }

    /// Get a mutable budget by link name.
    pub fn get_budget_mut(&mut self, name: &str) -> Option<&mut BandwidthBudget> {
        self.budgets
            .iter_mut()
            .find(|(n, _)| n == name)
            .map(|(_, b)| b)
    }

    /// Total capacity across all links.
    pub fn total_capacity_bps(&self) -> f64 {
        self.budgets.iter().map(|(_, b)| b.capacity_bps()).sum()
    }

    /// Total available across all links.
    pub fn total_available_bps(&self) -> f64 {
        self.budgets.iter().map(|(_, b)| b.available_bps()).sum()
    }

    /// Set utilization thresholds for a specific link.
    pub fn set_thresholds_for_link(&mut self, link_id: &str, threshold: UtilizationThreshold) {
        if let Some(budget) = self.get_budget_mut(link_id) {
            budget.set_thresholds(threshold);
        }
    }

    /// Register an alert callback for a specific link.
    pub fn on_alert_for_link(&mut self, link_id: &str, callback: AlertCallback) {
        if let Some(budget) = self.get_budget_mut(link_id) {
            budget.on_alert(callback);
        }
    }
}

impl Default for BudgetAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn test_bandwidth_unit_to_bps() {
        assert!((BandwidthUnit::Kbps.to_bps(1.0) - 1000.0).abs() < 0.01);
        assert!((BandwidthUnit::Mbps.to_bps(1.0) - 1_000_000.0).abs() < 0.01);
        assert!((BandwidthUnit::Gbps.to_bps(1.0) - 1_000_000_000.0).abs() < 0.01);
    }

    #[test]
    fn test_bandwidth_unit_from_bps() {
        assert!((BandwidthUnit::Mbps.from_bps(1_000_000.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_new_budget_capacity() {
        let b = BandwidthBudget::new(10.0, BandwidthUnit::Gbps);
        assert!((b.capacity_bps() - 10_000_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_allocate_success() {
        let mut b = BandwidthBudget::new(100.0, BandwidthUnit::Mbps);
        let req = AllocationRequest::new("s1", 50_000_000.0, 1);
        assert_eq!(b.allocate(&req), AllocationResult::Granted);
        assert_eq!(b.allocation_count(), 1);
    }

    #[test]
    fn test_allocate_denied() {
        let mut b = BandwidthBudget::new(10.0, BandwidthUnit::Mbps);
        let req = AllocationRequest::new("big", 20_000_000.0, 1);
        assert_eq!(b.allocate(&req), AllocationResult::Denied);
    }

    #[test]
    fn test_allocate_duplicate() {
        let mut b = BandwidthBudget::new(100.0, BandwidthUnit::Mbps);
        let req = AllocationRequest::new("s1", 10_000_000.0, 1);
        b.allocate(&req);
        assert_eq!(b.allocate(&req), AllocationResult::AlreadyAllocated);
    }

    #[test]
    fn test_release() {
        let mut b = BandwidthBudget::new(100.0, BandwidthUnit::Mbps);
        let req = AllocationRequest::new("s1", 30_000_000.0, 1);
        b.allocate(&req);
        let freed = b.release("s1");
        assert!(freed.is_some());
        assert!((freed.expect("should succeed in test") - 30_000_000.0).abs() < 1.0);
        assert_eq!(b.allocation_count(), 0);
    }

    #[test]
    fn test_release_nonexistent() {
        let mut b = BandwidthBudget::new(100.0, BandwidthUnit::Mbps);
        assert!(b.release("ghost").is_none());
    }

    #[test]
    fn test_available_bps() {
        let mut b = BandwidthBudget::new(100.0, BandwidthUnit::Mbps);
        let req = AllocationRequest::new("s1", 40_000_000.0, 1);
        b.allocate(&req);
        assert!((b.available_bps() - 60_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_utilization() {
        let mut b = BandwidthBudget::new(100.0, BandwidthUnit::Mbps);
        let req = AllocationRequest::new("s1", 50_000_000.0, 1);
        b.allocate(&req);
        assert!((b.utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_utilization_zero_capacity() {
        let b = BandwidthBudget::new(0.0, BandwidthUnit::Bps);
        assert!((b.utilization()).abs() < 0.01);
    }

    #[test]
    fn test_release_all() {
        let mut b = BandwidthBudget::new(100.0, BandwidthUnit::Mbps);
        b.allocate(&AllocationRequest::new("a", 10_000_000.0, 1));
        b.allocate(&AllocationRequest::new("b", 20_000_000.0, 2));
        b.release_all();
        assert_eq!(b.allocation_count(), 0);
    }

    #[test]
    fn test_is_allocated() {
        let mut b = BandwidthBudget::new(100.0, BandwidthUnit::Mbps);
        b.allocate(&AllocationRequest::new("s1", 10_000_000.0, 1));
        assert!(b.is_allocated("s1"));
        assert!(!b.is_allocated("s2"));
    }

    #[test]
    fn test_budget_allocator_multi_link() {
        let mut alloc = BudgetAllocator::new();
        alloc.add_link("link1", BandwidthBudget::new(10.0, BandwidthUnit::Gbps));
        alloc.add_link("link2", BandwidthBudget::new(1.0, BandwidthUnit::Gbps));
        assert_eq!(alloc.link_count(), 2);
        assert!((alloc.total_capacity_bps() - 11_000_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_budget_allocator_get_budget() {
        let mut alloc = BudgetAllocator::new();
        alloc.add_link("primary", BandwidthBudget::new(10.0, BandwidthUnit::Gbps));
        assert!(alloc.get_budget("primary").is_some());
        assert!(alloc.get_budget("missing").is_none());
    }

    #[test]
    fn test_preempt_lowest_priority() {
        let mut b = BandwidthBudget::new(100.0, BandwidthUnit::Mbps);
        // Fill up with low priority
        b.allocate(&AllocationRequest::new("low", 80_000_000.0, 10));
        // High priority request
        let req = AllocationRequest::new("high", 80_000_000.0, 1);
        let preempted = b.preempt_lowest_priority(&req);
        assert_eq!(preempted, Some("low".to_string()));
        assert!(b.is_allocated("high"));
        assert!(!b.is_allocated("low"));
    }

    // ── Threshold / alert tests ───────────────────────────────────────────────

    /// Helper: build a budget that fires alerts into an atomic counter.
    /// Returns (budget, warning_count, critical_count, cleared_count).
    fn make_instrumented_budget(
        capacity_mbps: f64,
    ) -> (
        BandwidthBudget,
        Arc<AtomicU32>,
        Arc<AtomicU32>,
        Arc<AtomicU32>,
    ) {
        let warning_count = Arc::new(AtomicU32::new(0));
        let critical_count = Arc::new(AtomicU32::new(0));
        let cleared_count = Arc::new(AtomicU32::new(0));

        let wc = Arc::clone(&warning_count);
        let cc = Arc::clone(&critical_count);
        let clc = Arc::clone(&cleared_count);

        let mut budget = BandwidthBudget::with_id("test-link", capacity_mbps, BandwidthUnit::Mbps);
        budget.set_thresholds(UtilizationThreshold {
            warning_pct: 80.0,
            critical_pct: 95.0,
        });
        budget.on_alert(Arc::new(move |alert| match alert {
            BandwidthAlert::Warning { .. } => {
                wc.fetch_add(1, Ordering::SeqCst);
            }
            BandwidthAlert::Critical { .. } => {
                cc.fetch_add(1, Ordering::SeqCst);
            }
            BandwidthAlert::Cleared { .. } => {
                clc.fetch_add(1, Ordering::SeqCst);
            }
        }));

        (budget, warning_count, critical_count, cleared_count)
    }

    #[test]
    fn test_threshold_warning_fires_on_crossing() {
        let (mut b, warn, crit, _cleared) = make_instrumented_budget(100.0);

        // 81 Mbps → 81% → should cross warning threshold
        b.allocate(&AllocationRequest::new("s1", 81_000_000.0, 1));
        assert_eq!(warn.load(Ordering::SeqCst), 1, "warning should fire once");
        assert_eq!(crit.load(Ordering::SeqCst), 0, "critical should not fire");
    }

    #[test]
    fn test_threshold_critical_fires_on_crossing() {
        let (mut b, warn, crit, _cleared) = make_instrumented_budget(100.0);

        // 96 Mbps → 96% → critical
        b.allocate(&AllocationRequest::new("s1", 96_000_000.0, 1));
        // Should have gone directly Normal → Critical (warning threshold also passed)
        assert_eq!(crit.load(Ordering::SeqCst), 1, "critical should fire once");
        // Warning was skipped because we jumped straight to critical
        assert_eq!(warn.load(Ordering::SeqCst), 0, "warning was skipped");
    }

    #[test]
    fn test_threshold_cleared_fires_on_release_below_warning() {
        let (mut b, _warn, _crit, cleared) = make_instrumented_budget(100.0);

        b.allocate(&AllocationRequest::new("s1", 81_000_000.0, 1));
        assert_eq!(cleared.load(Ordering::SeqCst), 0);

        // Release → utilization drops to 0 → cleared
        b.release("s1");
        assert_eq!(
            cleared.load(Ordering::SeqCst),
            1,
            "cleared should fire once"
        );
    }

    #[test]
    fn test_threshold_hysteresis_no_repeated_alerts() {
        let (mut b, warn, _crit, _cleared) = make_instrumented_budget(100.0);

        // First crossing
        b.allocate(&AllocationRequest::new("s1", 81_000_000.0, 1));
        assert_eq!(warn.load(Ordering::SeqCst), 1);

        // Second allocation while still in warning zone — no new alert
        b.allocate(&AllocationRequest::new("s2", 5_000_000.0, 2));
        // Still warning (86%), no new transition
        assert_eq!(warn.load(Ordering::SeqCst), 1, "no repeated warning");
    }

    #[test]
    fn test_threshold_warning_to_critical_transition() {
        let (mut b, warn, crit, _cleared) = make_instrumented_budget(100.0);

        // Cross into warning first
        b.allocate(&AllocationRequest::new("s1", 82_000_000.0, 1));
        assert_eq!(warn.load(Ordering::SeqCst), 1);
        assert_eq!(crit.load(Ordering::SeqCst), 0);

        // Now push into critical
        b.allocate(&AllocationRequest::new("s2", 14_000_000.0, 2));
        // 96% → critical transition
        assert_eq!(crit.load(Ordering::SeqCst), 1, "critical should fire");
        assert_eq!(warn.load(Ordering::SeqCst), 1, "warning not re-fired");
    }

    #[test]
    fn test_threshold_critical_to_warning_to_normal() {
        let (mut b, warn, crit, cleared) = make_instrumented_budget(100.0);

        // ── Step 1: allocate 96% → Critical ──────────────────────────────────
        b.allocate(&AllocationRequest::new("s1", 96_000_000.0, 1));
        assert_eq!(
            crit.load(Ordering::SeqCst),
            1,
            "critical fires on first allocation"
        );

        // ── Step 2: release → 0% → Cleared (Critical → Normal) ───────────────
        b.release("s1");
        assert_eq!(
            cleared.load(Ordering::SeqCst),
            1,
            "cleared fires on release from critical"
        );

        // ── Step 3: allocate 82% → Warning (Normal → Warning) ────────────────
        b.allocate(&AllocationRequest::new("s2", 82_000_000.0, 1));
        assert_eq!(
            warn.load(Ordering::SeqCst),
            1,
            "warning fires after critical cleared"
        );

        // ── Step 4: release → 0% → Cleared (Warning → Normal) ────────────────
        // Releasing from warning also fires Cleared, so total cleared = 2.
        b.release("s2");
        assert_eq!(
            cleared.load(Ordering::SeqCst),
            2,
            "cleared fires again on release from warning"
        );
    }

    #[test]
    fn test_budget_allocator_per_link_thresholds() {
        let mut alloc = BudgetAllocator::new();
        alloc.add_link(
            "primary",
            BandwidthBudget::with_id("primary", 100.0, BandwidthUnit::Mbps),
        );
        alloc.set_thresholds_for_link(
            "primary",
            UtilizationThreshold {
                warning_pct: 80.0,
                critical_pct: 95.0,
            },
        );
        let fired = Arc::new(AtomicU32::new(0));
        let fc = Arc::clone(&fired);
        alloc.on_alert_for_link(
            "primary",
            Arc::new(move |_| {
                fc.fetch_add(1, Ordering::SeqCst);
            }),
        );

        if let Some(budget) = alloc.get_budget_mut("primary") {
            budget.allocate(&AllocationRequest::new("s1", 85_000_000.0, 1));
        }
        assert_eq!(fired.load(Ordering::SeqCst), 1, "per-link alert fired");
    }

    #[test]
    fn test_no_alert_without_thresholds() {
        // Without calling set_thresholds, no alerts should fire
        let fired = Arc::new(AtomicU32::new(0));
        let fc = Arc::clone(&fired);
        let mut b = BandwidthBudget::new(100.0, BandwidthUnit::Mbps);
        b.on_alert(Arc::new(move |_| {
            fc.fetch_add(1, Ordering::SeqCst);
        }));

        b.allocate(&AllocationRequest::new("s1", 99_000_000.0, 1));
        assert_eq!(
            fired.load(Ordering::SeqCst),
            0,
            "no alert without thresholds"
        );
    }

    #[test]
    fn test_utilization_pct_method() {
        let mut b = BandwidthBudget::new(100.0, BandwidthUnit::Mbps);
        b.allocate(&AllocationRequest::new("s1", 75_000_000.0, 1));
        assert!((b.utilization_pct() - 75.0).abs() < 0.01);
    }
}
