//! Rights-renewal scheduling and status tracking.
//!
//! When a rights window is approaching expiry, rights managers need to know
//! which assets are eligible for renewal, in what priority order, and what
//! the renewal cost estimate is.  This module provides:
//!
//! - [`RenewalCandidate`] — a rights record flagged as approaching expiry.
//! - [`RenewalStatus`] — lifecycle state: `Pending`, `InProgress`, `Renewed`,
//!   `Lapsed`, `NotRequired`.
//! - [`RenewalConfig`] — look-ahead window and cost-estimation parameters.
//! - [`RenewalScheduler`] — assembles candidates from a set of rights records
//!   and tracks their lifecycle through to completion.
//!
//! # Design notes
//!
//! All timestamps are Unix seconds.  The scheduler is purely in-memory and
//! does not require a database; persistence is the caller's responsibility.
//!
//! Priority scoring is additive:
//! - High-revenue assets score higher (configurable weight).
//! - Active rights score higher than suspended ones.
//! - Imminently-expiring rights score higher than those expiring further out.
//!
//! # Example
//!
//! ```rust
//! use oximedia_rights::rights_renewal::{RenewalConfig, RenewalScheduler, RenewalStatus};
//!
//! let config = RenewalConfig {
//!     lookahead_days: 30,
//!     cost_per_day_usd: 5.0,
//!     revenue_weight: 1.5,
//! };
//! let mut scheduler = RenewalScheduler::new(config);
//!
//! let now = 0_u64;
//! // Add a record expiring in 15 days from now (within the 30-day window)
//! scheduler.add_record("rec-1", "asset-A", "Alice Media", true, now + 15 * 86_400, 1_000.0);
//!
//! let candidates = scheduler.due_for_renewal(now);
//! assert_eq!(candidates.len(), 1);
//! ```

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ── RenewalStatus ─────────────────────────────────────────────────────────────

/// Lifecycle state of a renewal candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenewalStatus {
    /// The renewal has been identified but no action has been taken.
    Pending,
    /// A renewal request has been submitted and is being processed.
    InProgress,
    /// The rights have been successfully renewed.
    Renewed,
    /// The rights expired without renewal.
    Lapsed,
    /// The asset no longer requires renewal (e.g. contract terminated).
    NotRequired,
}

impl RenewalStatus {
    /// Whether this status represents a terminal state (no further transitions
    /// are expected).
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Renewed | Self::Lapsed | Self::NotRequired)
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Pending => "Pending",
            Self::InProgress => "In Progress",
            Self::Renewed => "Renewed",
            Self::Lapsed => "Lapsed",
            Self::NotRequired => "Not Required",
        }
    }
}

// ── RenewalConfig ─────────────────────────────────────────────────────────────

/// Configuration for the renewal scheduler.
#[derive(Debug, Clone)]
pub struct RenewalConfig {
    /// How many days before expiry a record is flagged as a renewal candidate.
    pub lookahead_days: u64,
    /// Estimated cost per day of rights extension in USD (used for cost
    /// estimates only — not authoritative).
    pub cost_per_day_usd: f64,
    /// Multiplier applied to the revenue score component when computing
    /// priority.  Values > 1.0 favour high-revenue assets.
    pub revenue_weight: f64,
}

impl Default for RenewalConfig {
    fn default() -> Self {
        Self {
            lookahead_days: 30,
            cost_per_day_usd: 10.0,
            revenue_weight: 1.0,
        }
    }
}

impl RenewalConfig {
    /// Lookahead window in seconds.
    #[must_use]
    pub fn lookahead_seconds(&self) -> u64 {
        self.lookahead_days * 86_400
    }
}

// ── RenewalCandidate ──────────────────────────────────────────────────────────

/// A rights record that is eligible for renewal.
#[derive(Debug, Clone)]
pub struct RenewalCandidate {
    /// Rights record identifier.
    pub record_id: String,
    /// Asset to which the rights apply.
    pub asset_id: String,
    /// Rights holder name.
    pub holder: String,
    /// Whether the rights are currently active.
    pub active: bool,
    /// Expiry timestamp (Unix seconds).
    pub expires_at: u64,
    /// Estimated monthly revenue attributable to this asset (USD).
    pub monthly_revenue_usd: f64,
    /// Current lifecycle status.
    pub status: RenewalStatus,
    /// Priority score (higher = more urgent to renew).  Computed by
    /// [`RenewalScheduler`]; 0.0 for manually-inserted candidates.
    pub priority_score: f64,
    /// Estimated cost to renew for one year (USD).
    pub estimated_annual_cost_usd: f64,
    /// Unix timestamp when the renewal was last actioned (or 0 if never).
    pub last_actioned_at: u64,
    /// Free-text notes recorded by the rights manager.
    pub notes: String,
}

impl RenewalCandidate {
    /// Days remaining until expiry from `now`.  Returns 0 if already expired.
    #[must_use]
    pub fn days_remaining(&self, now: u64) -> u64 {
        if now >= self.expires_at {
            0
        } else {
            (self.expires_at - now) / 86_400
        }
    }

    /// Whether the record has already expired at `now`.
    #[must_use]
    pub fn is_expired(&self, now: u64) -> bool {
        now >= self.expires_at
    }

    /// Whether the status allows transitioning to `new_status`.
    ///
    /// Valid transitions:
    /// - `Pending` → `InProgress`, `NotRequired`
    /// - `InProgress` → `Renewed`, `Lapsed`, `Pending` (rollback)
    /// - Terminal states (`Renewed`, `Lapsed`, `NotRequired`) → none
    #[must_use]
    pub fn can_transition_to(&self, new_status: RenewalStatus) -> bool {
        if self.status.is_terminal() {
            return false;
        }
        match (self.status, new_status) {
            (RenewalStatus::Pending, RenewalStatus::InProgress)
            | (RenewalStatus::Pending, RenewalStatus::NotRequired)
            | (RenewalStatus::InProgress, RenewalStatus::Renewed)
            | (RenewalStatus::InProgress, RenewalStatus::Lapsed)
            | (RenewalStatus::InProgress, RenewalStatus::Pending) => true,
            _ => false,
        }
    }
}

// ── RenewalScheduler ──────────────────────────────────────────────────────────

/// Assembles renewal candidates from rights records and tracks their lifecycle.
#[derive(Debug)]
pub struct RenewalScheduler {
    config: RenewalConfig,
    /// Keyed by record_id.
    candidates: HashMap<String, RenewalCandidate>,
}

impl RenewalScheduler {
    /// Create a new scheduler with the given configuration.
    #[must_use]
    pub fn new(config: RenewalConfig) -> Self {
        Self {
            config,
            candidates: HashMap::new(),
        }
    }

    /// Create a scheduler with the default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(RenewalConfig::default())
    }

    // ── Record management ─────────────────────────────────────────────────────

    /// Register a rights record.
    ///
    /// If a candidate with the same `record_id` already exists it is
    /// replaced.  The priority score and estimated annual cost are computed
    /// automatically based on the configuration.
    pub fn add_record(
        &mut self,
        record_id: impl Into<String>,
        asset_id: impl Into<String>,
        holder: impl Into<String>,
        active: bool,
        expires_at: u64,
        monthly_revenue_usd: f64,
    ) {
        let record_id = record_id.into();
        let annual_cost =
            self.config.cost_per_day_usd * 365.0;
        // Priority: inverse of days remaining (closer = higher priority)
        // + revenue contribution.  We use a reference "now" of 0; the score
        // is recomputed lazily in `due_for_renewal`.
        let candidate = RenewalCandidate {
            record_id: record_id.clone(),
            asset_id: asset_id.into(),
            holder: holder.into(),
            active,
            expires_at,
            monthly_revenue_usd,
            status: RenewalStatus::Pending,
            priority_score: 0.0,
            estimated_annual_cost_usd: annual_cost,
            last_actioned_at: 0,
            notes: String::new(),
        };
        self.candidates.insert(record_id, candidate);
    }

    /// Remove a record from the scheduler.
    pub fn remove_record(&mut self, record_id: &str) -> Option<RenewalCandidate> {
        self.candidates.remove(record_id)
    }

    // ── Status lifecycle ──────────────────────────────────────────────────────

    /// Transition a candidate to a new status.
    ///
    /// Returns `Ok(())` on success, `Err` with a description if the
    /// transition is invalid or the record doesn't exist.
    pub fn transition(
        &mut self,
        record_id: &str,
        new_status: RenewalStatus,
        now: u64,
        notes: impl Into<String>,
    ) -> crate::Result<()> {
        let candidate = self
            .candidates
            .get_mut(record_id)
            .ok_or_else(|| crate::RightsError::NotFound(record_id.to_string()))?;

        if !candidate.can_transition_to(new_status) {
            return Err(crate::RightsError::InvalidOperation(format!(
                "Cannot transition '{}' from {:?} to {:?}",
                record_id, candidate.status, new_status
            )));
        }

        candidate.status = new_status;
        candidate.last_actioned_at = now;
        let note = notes.into();
        if !note.is_empty() {
            if !candidate.notes.is_empty() {
                candidate.notes.push('\n');
            }
            candidate.notes.push_str(&note);
        }
        Ok(())
    }

    // ── Query ─────────────────────────────────────────────────────────────────

    /// All candidates that fall within the lookahead window at `now` and are
    /// not yet in a terminal state, sorted by priority score descending.
    #[must_use]
    pub fn due_for_renewal(&mut self, now: u64) -> Vec<&RenewalCandidate> {
        let window_end = now.saturating_add(self.config.lookahead_seconds());
        let revenue_weight = self.config.revenue_weight;

        // Recompute priority scores.
        for c in self.candidates.values_mut() {
            let days_left = c.days_remaining(now) as f64 + 1.0; // +1 avoids div/0
            let urgency = 1.0 / days_left;
            let revenue_score = c.monthly_revenue_usd * revenue_weight / 1_000.0;
            c.priority_score = urgency + revenue_score;
        }

        let mut due: Vec<&RenewalCandidate> = self
            .candidates
            .values()
            .filter(|c| {
                !c.status.is_terminal() && c.expires_at > now && c.expires_at <= window_end
            })
            .collect();

        due.sort_by(|a, b| {
            b.priority_score
                .partial_cmp(&a.priority_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        due
    }

    /// All candidates already expired at `now` that are not yet in a terminal
    /// state, sorted by expiry ascending.
    #[must_use]
    pub fn lapsed_candidates(&self, now: u64) -> Vec<&RenewalCandidate> {
        let mut lapsed: Vec<&RenewalCandidate> = self
            .candidates
            .values()
            .filter(|c| c.is_expired(now) && !c.status.is_terminal())
            .collect();
        lapsed.sort_by_key(|c| c.expires_at);
        lapsed
    }

    /// All candidates by status.
    #[must_use]
    pub fn by_status(&self, status: RenewalStatus) -> Vec<&RenewalCandidate> {
        self.candidates
            .values()
            .filter(|c| c.status == status)
            .collect()
    }

    /// Total number of registered candidates.
    #[must_use]
    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }

    /// Estimated total annual renewal cost for all candidates currently due.
    #[must_use]
    pub fn estimated_renewal_cost(&mut self, now: u64) -> f64 {
        self.due_for_renewal(now)
            .iter()
            .map(|c| c.estimated_annual_cost_usd)
            .sum()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: u64 = 10 * 86_400; // start at 10 days so we can subtract a day
    const DAY: u64 = 86_400;

    fn make_scheduler() -> RenewalScheduler {
        let config = RenewalConfig {
            lookahead_days: 30,
            cost_per_day_usd: 10.0,
            revenue_weight: 1.0,
        };
        RenewalScheduler::new(config)
    }

    fn populate(s: &mut RenewalScheduler) {
        // Expires in 10 days — within 30-day window
        s.add_record("r1", "asset-A", "Alice", true, BASE + 10 * DAY, 500.0);
        // Expires in 20 days — within window
        s.add_record("r2", "asset-B", "Bob", true, BASE + 20 * DAY, 200.0);
        // Expires in 40 days — outside 30-day window
        s.add_record("r3", "asset-C", "Carol", true, BASE + 40 * DAY, 100.0);
        // Already expired
        s.add_record("r4", "asset-D", "Dave", false, BASE - DAY, 50.0);
    }

    // ── RenewalStatus ──

    #[test]
    fn test_status_terminal() {
        assert!(RenewalStatus::Renewed.is_terminal());
        assert!(RenewalStatus::Lapsed.is_terminal());
        assert!(RenewalStatus::NotRequired.is_terminal());
        assert!(!RenewalStatus::Pending.is_terminal());
        assert!(!RenewalStatus::InProgress.is_terminal());
    }

    #[test]
    fn test_status_label() {
        assert_eq!(RenewalStatus::Pending.label(), "Pending");
        assert_eq!(RenewalStatus::Renewed.label(), "Renewed");
    }

    // ── RenewalConfig ──

    #[test]
    fn test_config_lookahead_seconds() {
        let cfg = RenewalConfig {
            lookahead_days: 7,
            ..Default::default()
        };
        assert_eq!(cfg.lookahead_seconds(), 7 * 86_400);
    }

    // ── RenewalCandidate ──

    #[test]
    fn test_days_remaining() {
        let mut s = make_scheduler();
        s.add_record("r", "a", "h", true, BASE + 5 * DAY, 0.0);
        let c = s.candidates.get("r").expect("record should exist");
        assert_eq!(c.days_remaining(BASE), 5);
        assert_eq!(c.days_remaining(BASE + 5 * DAY), 0);
    }

    #[test]
    fn test_is_expired() {
        let mut s = make_scheduler();
        s.add_record("r", "a", "h", true, BASE + DAY, 0.0);
        let c = s.candidates.get("r").expect("record should exist");
        assert!(!c.is_expired(BASE));
        assert!(c.is_expired(BASE + DAY));
    }

    #[test]
    fn test_valid_transitions() {
        let mut s = make_scheduler();
        populate(&mut s);
        let c = s.candidates.get("r1").expect("r1 should exist");
        assert!(c.can_transition_to(RenewalStatus::InProgress));
        assert!(c.can_transition_to(RenewalStatus::NotRequired));
        assert!(!c.can_transition_to(RenewalStatus::Renewed)); // skip InProgress
    }

    #[test]
    fn test_terminal_blocks_all_transitions() {
        let mut s = make_scheduler();
        s.add_record("r", "a", "h", true, BASE + DAY, 0.0);
        s.transition("r", RenewalStatus::InProgress, 1, "").expect("transition should succeed");
        s.transition("r", RenewalStatus::Renewed, 2, "").expect("transition should succeed");
        let c = s.candidates.get("r").expect("record should exist");
        assert!(!c.can_transition_to(RenewalStatus::Pending));
    }

    // ── RenewalScheduler — due_for_renewal ──

    #[test]
    fn test_due_for_renewal_count() {
        let mut s = make_scheduler();
        populate(&mut s);
        // r1 (10 days) and r2 (20 days) within 30-day window;
        // r3 is 40 days away; r4 already expired
        let due = s.due_for_renewal(BASE);
        assert_eq!(due.len(), 2);
    }

    #[test]
    fn test_due_sorted_by_priority_descending() {
        let mut s = make_scheduler();
        populate(&mut s);
        let due = s.due_for_renewal(BASE);
        // r1 expires sooner → higher urgency → higher priority
        let priorities: Vec<f64> = due.iter().map(|c| c.priority_score).collect();
        for pair in priorities.windows(2) {
            assert!(pair[0] >= pair[1], "not sorted descending: {:?}", priorities);
        }
    }

    #[test]
    fn test_lapsed_candidates() {
        let mut s = make_scheduler();
        populate(&mut s);
        let lapsed = s.lapsed_candidates(BASE);
        assert_eq!(lapsed.len(), 1);
        assert_eq!(lapsed[0].record_id, "r4");
    }

    #[test]
    fn test_terminal_excluded_from_due() {
        let mut s = make_scheduler();
        populate(&mut s);
        s.transition("r1", RenewalStatus::InProgress, 1, "filed")
            .expect("transition should succeed");
        s.transition("r1", RenewalStatus::Renewed, 2, "done")
            .expect("transition should succeed");
        let due = s.due_for_renewal(BASE);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].record_id, "r2");
    }

    #[test]
    fn test_by_status() {
        let mut s = make_scheduler();
        populate(&mut s);
        let pending = s.by_status(RenewalStatus::Pending);
        assert_eq!(pending.len(), 4); // all 4 start as Pending
    }

    #[test]
    fn test_transition_invalid_returns_err() {
        let mut s = make_scheduler();
        s.add_record("r", "a", "h", true, BASE + DAY, 0.0);
        // Pending → Renewed is not a valid direct transition
        let result = s.transition("r", RenewalStatus::Renewed, 1, "");
        assert!(result.is_err());
    }

    #[test]
    fn test_transition_not_found_returns_err() {
        let mut s = make_scheduler();
        let result = s.transition("nonexistent", RenewalStatus::InProgress, 1, "");
        assert!(result.is_err());
    }

    #[test]
    fn test_estimated_renewal_cost() {
        let config = RenewalConfig {
            lookahead_days: 30,
            cost_per_day_usd: 10.0,
            revenue_weight: 1.0,
        };
        let mut s = RenewalScheduler::new(config);
        // 2 records within window, annual cost = 365 * 10 = 3650 each
        s.add_record("r1", "a", "h", true, BASE + 10 * DAY, 0.0);
        s.add_record("r2", "b", "h", true, BASE + 20 * DAY, 0.0);
        let cost = s.estimated_renewal_cost(BASE);
        let expected = 2.0 * 365.0 * 10.0;
        assert!((cost - expected).abs() < 1e-6);
    }

    #[test]
    fn test_remove_record() {
        let mut s = make_scheduler();
        s.add_record("r", "a", "h", true, BASE + DAY, 0.0);
        assert_eq!(s.candidate_count(), 1);
        let removed = s.remove_record("r");
        assert!(removed.is_some());
        assert_eq!(s.candidate_count(), 0);
    }
}
