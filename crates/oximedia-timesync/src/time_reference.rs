//! Time reference source management for multi-source clock synchronization.
//!
//! This module provides types for representing external time reference sources
//! (GPS, PTP, NTP, etc.) and a manager that selects the best available reference
//! using configurable priority and quality criteria.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// The type of time reference source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ReferenceSource {
    /// GPS / GNSS disciplined oscillator (highest accuracy).
    Gps,
    /// PTP (IEEE 1588) grandmaster clock.
    Ptp,
    /// NTP stratum-1 server.
    NtpStratum1,
    /// NTP stratum-2 or higher server.
    NtpStratum2Plus,
    /// SMPTE LTC timecode derived reference.
    Ltc,
    /// Internal oscillator (free-running, lowest accuracy).
    Internal,
}

impl ReferenceSource {
    /// Returns the default priority for this source (lower number = higher priority).
    pub fn default_priority(self) -> u8 {
        match self {
            Self::Gps => 1,
            Self::Ptp => 2,
            Self::NtpStratum1 => 3,
            Self::NtpStratum2Plus => 4,
            Self::Ltc => 5,
            Self::Internal => 255,
        }
    }

    /// Returns a human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Gps => "GPS",
            Self::Ptp => "PTP",
            Self::NtpStratum1 => "NTP/S1",
            Self::NtpStratum2Plus => "NTP/S2+",
            Self::Ltc => "LTC",
            Self::Internal => "Internal",
        }
    }

    /// Returns whether this source is expected to provide sub-microsecond accuracy.
    pub fn is_high_precision(self) -> bool {
        matches!(self, Self::Gps | Self::Ptp)
    }
}

/// Quality descriptor for a time reference measurement.
#[derive(Debug, Clone, Copy)]
pub struct ReferenceQuality {
    /// Estimated uncertainty / accuracy in nanoseconds (lower is better).
    pub accuracy_ns: u64,
    /// Root-mean-square jitter observed over recent samples, in nanoseconds.
    pub jitter_ns: u64,
    /// Whether the source is currently considered locked/valid.
    pub locked: bool,
}

impl ReferenceQuality {
    /// Create a quality descriptor for a locked source.
    pub fn locked(accuracy_ns: u64, jitter_ns: u64) -> Self {
        Self {
            accuracy_ns,
            jitter_ns,
            locked: true,
        }
    }

    /// Create a quality descriptor for an unlocked (invalid) source.
    pub fn unlocked() -> Self {
        Self {
            accuracy_ns: u64::MAX,
            jitter_ns: u64::MAX,
            locked: false,
        }
    }

    /// Compute a scalar score (lower is better) for reference selection.
    ///
    /// Uses a weighted sum of accuracy and jitter.
    pub fn score(&self) -> u64 {
        if !self.locked {
            return u64::MAX;
        }
        self.accuracy_ns.saturating_add(self.jitter_ns / 2)
    }
}

/// A snapshot of a time reference at a given instant.
#[derive(Debug, Clone)]
pub struct TimeReference {
    /// The type of source providing this reference.
    pub source: ReferenceSource,
    /// Current offset of the local clock from this reference, in nanoseconds.
    pub offset_ns: i64,
    /// Quality metrics for this reference.
    pub quality: ReferenceQuality,
    /// When this snapshot was taken.
    pub captured_at: Instant,
    /// Optional display label (e.g., hostname of an NTP server).
    pub label: Option<String>,
}

impl TimeReference {
    /// Create a new time reference snapshot.
    pub fn new(source: ReferenceSource, offset_ns: i64, quality: ReferenceQuality) -> Self {
        Self {
            source,
            offset_ns,
            quality,
            captured_at: Instant::now(),
            label: None,
        }
    }

    /// Attach an optional label to this reference.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// How long ago this snapshot was captured.
    pub fn age(&self) -> Duration {
        self.captured_at.elapsed()
    }

    /// Whether this reference is fresh (captured within `max_age`).
    pub fn is_fresh(&self, max_age: Duration) -> bool {
        self.age() <= max_age
    }

    /// Whether this reference is locked and fresh.
    pub fn is_usable(&self, max_age: Duration) -> bool {
        self.quality.locked && self.is_fresh(max_age)
    }
}

/// Manages multiple time reference sources and selects the best one.
#[derive(Debug)]
pub struct TimeReferenceManager {
    /// Known references, keyed by source type.
    references: HashMap<ReferenceSource, TimeReference>,
    /// Optional priority overrides (lower = higher priority).
    priority_overrides: HashMap<ReferenceSource, u8>,
    /// Maximum age before a reference is considered stale.
    max_reference_age: Duration,
    /// Currently selected reference source.
    selected: Option<ReferenceSource>,
}

impl TimeReferenceManager {
    /// Create a new manager with the given staleness threshold.
    pub fn new(max_reference_age: Duration) -> Self {
        Self {
            references: HashMap::new(),
            priority_overrides: HashMap::new(),
            max_reference_age,
            selected: None,
        }
    }

    /// Override the priority for a specific source.
    ///
    /// Lower values mean higher priority. Use `0` to prefer over even GPS.
    pub fn set_priority(&mut self, source: ReferenceSource, priority: u8) {
        self.priority_overrides.insert(source, priority);
    }

    /// Update or insert a reference snapshot, then reselect the best source.
    pub fn update(&mut self, reference: TimeReference) {
        self.references.insert(reference.source, reference);
        self.reselect();
    }

    /// Remove a reference source (e.g., after disconnect), then reselect.
    pub fn remove(&mut self, source: ReferenceSource) {
        self.references.remove(&source);
        if self.selected == Some(source) {
            self.selected = None;
        }
        self.reselect();
    }

    /// Force a re-selection of the best available reference.
    pub fn reselect(&mut self) {
        let max_age = self.max_reference_age;
        self.selected = self
            .references
            .values()
            .filter(|r| r.is_usable(max_age))
            .min_by_key(|r| {
                let priority = self
                    .priority_overrides
                    .get(&r.source)
                    .copied()
                    .unwrap_or_else(|| r.source.default_priority());
                // Primary sort by priority; secondary by quality score.
                (priority, r.quality.score())
            })
            .map(|r| r.source);
    }

    /// Get the currently selected best reference, if any.
    pub fn selected(&self) -> Option<&TimeReference> {
        self.selected.and_then(|s| self.references.get(&s))
    }

    /// Get a reference by source type.
    pub fn get(&self, source: ReferenceSource) -> Option<&TimeReference> {
        self.references.get(&source)
    }

    /// Returns how many references are currently tracked.
    pub fn reference_count(&self) -> usize {
        self.references.len()
    }

    /// Returns how many references are currently usable.
    pub fn usable_count(&self) -> usize {
        let max_age = self.max_reference_age;
        self.references
            .values()
            .filter(|r| r.is_usable(max_age))
            .count()
    }

    /// Returns the offset of the selected reference, or `None` if none selected.
    pub fn selected_offset_ns(&self) -> Option<i64> {
        self.selected().map(|r| r.offset_ns)
    }

    /// Returns the quality of the selected reference, or `None` if none selected.
    pub fn selected_quality(&self) -> Option<ReferenceQuality> {
        self.selected().map(|r| r.quality)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_source_priority_ordering() {
        assert!(ReferenceSource::Gps.default_priority() < ReferenceSource::Ptp.default_priority());
        assert!(
            ReferenceSource::Ptp.default_priority()
                < ReferenceSource::NtpStratum1.default_priority()
        );
        assert_eq!(ReferenceSource::Internal.default_priority(), 255);
    }

    #[test]
    fn test_reference_source_name() {
        assert_eq!(ReferenceSource::Gps.name(), "GPS");
        assert_eq!(ReferenceSource::Ptp.name(), "PTP");
        assert_eq!(ReferenceSource::NtpStratum1.name(), "NTP/S1");
        assert_eq!(ReferenceSource::Ltc.name(), "LTC");
        assert_eq!(ReferenceSource::Internal.name(), "Internal");
    }

    #[test]
    fn test_reference_source_high_precision() {
        assert!(ReferenceSource::Gps.is_high_precision());
        assert!(ReferenceSource::Ptp.is_high_precision());
        assert!(!ReferenceSource::NtpStratum1.is_high_precision());
        assert!(!ReferenceSource::Internal.is_high_precision());
    }

    #[test]
    fn test_reference_quality_locked_score() {
        let q = ReferenceQuality::locked(100, 50);
        assert!(q.locked);
        assert_eq!(q.score(), 125); // 100 + 50/2
    }

    #[test]
    fn test_reference_quality_unlocked_score() {
        let q = ReferenceQuality::unlocked();
        assert!(!q.locked);
        assert_eq!(q.score(), u64::MAX);
    }

    #[test]
    fn test_time_reference_is_fresh() {
        let q = ReferenceQuality::locked(100, 0);
        let r = TimeReference::new(ReferenceSource::Ptp, 500, q);
        assert!(r.is_fresh(Duration::from_secs(10)));
        std::thread::sleep(Duration::from_millis(1));
        assert!(!r.is_fresh(Duration::from_nanos(0)));
    }

    #[test]
    fn test_time_reference_with_label() {
        let q = ReferenceQuality::locked(200, 10);
        let r = TimeReference::new(ReferenceSource::NtpStratum1, 0, q).with_label("pool.ntp.org");
        assert_eq!(r.label.as_deref(), Some("pool.ntp.org"));
    }

    #[test]
    fn test_time_reference_is_usable() {
        let q = ReferenceQuality::locked(100, 0);
        let r = TimeReference::new(ReferenceSource::Gps, 0, q);
        assert!(r.is_usable(Duration::from_secs(60)));
    }

    #[test]
    fn test_manager_selects_best_priority() {
        let mut mgr = TimeReferenceManager::new(Duration::from_secs(60));
        let q_ptp = ReferenceQuality::locked(500, 100);
        let q_gps = ReferenceQuality::locked(50, 10);
        mgr.update(TimeReference::new(ReferenceSource::Ptp, 100, q_ptp));
        mgr.update(TimeReference::new(ReferenceSource::Gps, 5, q_gps));
        let sel = mgr.selected().expect("should succeed in test");
        assert_eq!(sel.source, ReferenceSource::Gps);
    }

    #[test]
    fn test_manager_reference_count() {
        let mut mgr = TimeReferenceManager::new(Duration::from_secs(60));
        assert_eq!(mgr.reference_count(), 0);
        mgr.update(TimeReference::new(
            ReferenceSource::NtpStratum1,
            0,
            ReferenceQuality::locked(1000, 200),
        ));
        assert_eq!(mgr.reference_count(), 1);
    }

    #[test]
    fn test_manager_remove_selected_clears_selection() {
        let mut mgr = TimeReferenceManager::new(Duration::from_secs(60));
        mgr.update(TimeReference::new(
            ReferenceSource::Ptp,
            0,
            ReferenceQuality::locked(100, 10),
        ));
        assert!(mgr.selected().is_some());
        mgr.remove(ReferenceSource::Ptp);
        assert!(mgr.selected().is_none());
    }

    #[test]
    fn test_manager_priority_override() {
        let mut mgr = TimeReferenceManager::new(Duration::from_secs(60));
        mgr.update(TimeReference::new(
            ReferenceSource::Gps,
            0,
            ReferenceQuality::locked(50, 5),
        ));
        mgr.update(TimeReference::new(
            ReferenceSource::Ptp,
            0,
            ReferenceQuality::locked(200, 20),
        ));
        // Override: prefer PTP over GPS
        mgr.set_priority(ReferenceSource::Ptp, 0);
        mgr.reselect();
        let sel = mgr.selected().expect("should succeed in test");
        assert_eq!(sel.source, ReferenceSource::Ptp);
    }

    #[test]
    fn test_manager_usable_count() {
        let mut mgr = TimeReferenceManager::new(Duration::from_secs(60));
        mgr.update(TimeReference::new(
            ReferenceSource::Ptp,
            0,
            ReferenceQuality::locked(100, 10),
        ));
        mgr.update(TimeReference::new(
            ReferenceSource::Internal,
            0,
            ReferenceQuality::unlocked(),
        ));
        assert_eq!(mgr.usable_count(), 1);
    }

    #[test]
    fn test_manager_selected_offset_and_quality() {
        let mut mgr = TimeReferenceManager::new(Duration::from_secs(60));
        mgr.update(TimeReference::new(
            ReferenceSource::Gps,
            42,
            ReferenceQuality::locked(10, 1),
        ));
        assert_eq!(mgr.selected_offset_ns(), Some(42));
        let q = mgr.selected_quality().expect("should succeed in test");
        assert!(q.locked);
    }

    #[test]
    fn test_manager_get_by_source() {
        let mut mgr = TimeReferenceManager::new(Duration::from_secs(60));
        mgr.update(TimeReference::new(
            ReferenceSource::Ltc,
            77,
            ReferenceQuality::locked(5000, 100),
        ));
        let r = mgr
            .get(ReferenceSource::Ltc)
            .expect("should succeed in test");
        assert_eq!(r.offset_ns, 77);
    }

    #[test]
    fn test_reference_quality_score_no_overflow() {
        let q = ReferenceQuality::locked(u64::MAX / 2, u64::MAX / 2);
        // Should saturate, not panic
        let _ = q.score();
    }
}
