//! Worker capacity reservations for scheduled high-priority jobs.
//!
//! [`ResourceReservation`] lets operators pre-allocate a number of worker
//! slots for a named high-priority job that will be submitted at a future
//! scheduled time.  A [`ReservationRegistry`] manages the full set of active
//! reservations and enforces that the number of reserved slots never exceeds
//! the available farm capacity.

#![allow(dead_code)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ReservationStatus
// ---------------------------------------------------------------------------

/// Lifecycle state of a [`ResourceReservation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReservationStatus {
    /// Reservation is active; slots are held for the named job.
    Active,
    /// Reservation has been consumed — the job was submitted and dispatched.
    Consumed,
    /// Reservation was cancelled before the job arrived.
    Cancelled,
    /// Reservation expired because the job never arrived within the window.
    Expired,
}

// ---------------------------------------------------------------------------
// ResourceReservation
// ---------------------------------------------------------------------------

/// A pre-allocated capacity reservation for a future render job.
#[derive(Debug, Clone)]
pub struct ResourceReservation {
    /// Stable identifier for this reservation.
    pub id: u64,
    /// Human-readable label (e.g. job name or client name).
    pub label: String,
    /// Number of worker slots reserved.
    pub slots: u32,
    /// Unix timestamp (seconds) from which the reservation is valid.
    pub start_at: i64,
    /// Unix timestamp (seconds) at which the reservation expires if unused.
    pub expires_at: i64,
    /// Current lifecycle status.
    pub status: ReservationStatus,
}

impl ResourceReservation {
    /// Creates a new active reservation.
    #[must_use]
    pub fn new(
        id: u64,
        label: impl Into<String>,
        slots: u32,
        start_at: i64,
        expires_at: i64,
    ) -> Self {
        Self {
            id,
            label: label.into(),
            slots,
            start_at,
            expires_at,
            status: ReservationStatus::Active,
        }
    }

    /// Returns `true` when this reservation is still holding capacity at the
    /// given Unix timestamp.
    #[must_use]
    pub fn is_active_at(&self, now: i64) -> bool {
        self.status == ReservationStatus::Active && now >= self.start_at && now < self.expires_at
    }

    /// Marks the reservation as consumed.
    pub fn consume(&mut self) {
        self.status = ReservationStatus::Consumed;
    }

    /// Cancels the reservation, releasing the held slots.
    pub fn cancel(&mut self) {
        self.status = ReservationStatus::Cancelled;
    }
}

// ---------------------------------------------------------------------------
// ReservationRegistry
// ---------------------------------------------------------------------------

/// Manages a pool of worker-capacity reservations for the render farm.
pub struct ReservationRegistry {
    reservations: HashMap<u64, ResourceReservation>,
    /// Total worker capacity of the farm.
    total_capacity: u32,
    next_id: u64,
}

impl ReservationRegistry {
    /// Creates a registry for a farm with `total_capacity` workers.
    #[must_use]
    pub fn new(total_capacity: u32) -> Self {
        Self {
            reservations: HashMap::new(),
            total_capacity,
            next_id: 1,
        }
    }

    /// Attempts to create a new reservation.
    ///
    /// Returns `Ok(reservation_id)` when there is sufficient unreserved
    /// capacity at the requested time window, or `Err(String)` describing why
    /// the reservation was rejected.
    ///
    /// The check is pessimistic: it sums all *active* reservations that
    /// overlap with `[start_at, expires_at)` regardless of whether they are
    /// currently consuming workers.
    pub fn reserve(
        &mut self,
        label: impl Into<String>,
        slots: u32,
        start_at: i64,
        expires_at: i64,
    ) -> Result<u64, String> {
        if slots == 0 {
            return Err("slots must be greater than zero".to_string());
        }
        if expires_at <= start_at {
            return Err("expires_at must be after start_at".to_string());
        }
        if slots > self.total_capacity {
            return Err(format!(
                "requested {slots} slots but farm capacity is only {}",
                self.total_capacity
            ));
        }

        // Sum active reservations that overlap with the requested window.
        let already_reserved: u32 = self
            .reservations
            .values()
            .filter(|r| {
                r.status == ReservationStatus::Active
                    && r.start_at < expires_at
                    && r.expires_at > start_at
            })
            .map(|r| r.slots)
            .sum();

        let available = self.total_capacity.saturating_sub(already_reserved);
        if slots > available {
            return Err(format!(
                "insufficient capacity: requested {slots}, available {available}"
            ));
        }

        let id = self.next_id;
        self.next_id += 1;
        let reservation = ResourceReservation::new(id, label, slots, start_at, expires_at);
        self.reservations.insert(id, reservation);
        Ok(id)
    }

    /// Cancels an existing reservation.
    ///
    /// Returns `true` when the reservation existed and was cancelled.
    pub fn cancel(&mut self, id: u64) -> bool {
        if let Some(r) = self.reservations.get_mut(&id) {
            r.cancel();
            true
        } else {
            false
        }
    }

    /// Marks a reservation as consumed (job dispatched).
    ///
    /// Returns `true` when the reservation existed and was in an active state.
    pub fn consume(&mut self, id: u64) -> bool {
        if let Some(r) = self.reservations.get_mut(&id) {
            if r.status == ReservationStatus::Active {
                r.consume();
                return true;
            }
        }
        false
    }

    /// Expires all reservations whose window has passed at `now`.
    pub fn expire_old(&mut self, now: i64) {
        for r in self.reservations.values_mut() {
            if r.status == ReservationStatus::Active && now >= r.expires_at {
                r.status = ReservationStatus::Expired;
            }
        }
    }

    /// Returns the number of worker slots reserved at the given timestamp.
    #[must_use]
    pub fn reserved_at(&self, now: i64) -> u32 {
        self.reservations
            .values()
            .filter(|r| r.is_active_at(now))
            .map(|r| r.slots)
            .sum()
    }

    /// Returns available (unreserved) capacity at the given timestamp.
    #[must_use]
    pub fn available_at(&self, now: i64) -> u32 {
        self.total_capacity.saturating_sub(self.reserved_at(now))
    }

    /// All reservations (any status).
    #[must_use]
    pub fn all(&self) -> impl Iterator<Item = &ResourceReservation> {
        self.reservations.values()
    }

    /// Number of active reservations.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.reservations
            .values()
            .filter(|r| r.status == ReservationStatus::Active)
            .count()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reserve_and_retrieve() {
        let mut reg = ReservationRegistry::new(100);
        let id = reg
            .reserve("job_a", 10, 1000, 2000)
            .expect("reserve should succeed");
        assert_eq!(reg.active_count(), 1);
        assert_eq!(reg.reserved_at(1500), 10);
        assert_eq!(reg.available_at(1500), 90);
        let _ = id;
    }

    #[test]
    fn test_reserve_exceeds_capacity_rejected() {
        let mut reg = ReservationRegistry::new(50);
        let result = reg.reserve("big_job", 60, 0, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_double_reserve_overlap_rejected_when_full() {
        let mut reg = ReservationRegistry::new(100);
        reg.reserve("a", 80, 0, 500).expect("first reserve");
        let result = reg.reserve("b", 30, 0, 500);
        assert!(
            result.is_err(),
            "overlapping reservation that exceeds capacity should fail"
        );
    }

    #[test]
    fn test_non_overlapping_reservations_both_succeed() {
        let mut reg = ReservationRegistry::new(100);
        reg.reserve("a", 80, 0, 500).expect("first reserve");
        let result = reg.reserve("b", 80, 500, 1000);
        assert!(
            result.is_ok(),
            "non-overlapping windows should both be granted"
        );
    }

    #[test]
    fn test_cancel_releases_capacity() {
        let mut reg = ReservationRegistry::new(100);
        let id = reg.reserve("a", 80, 0, 500).expect("reserve");
        reg.cancel(id);
        // After cancellation, new reservation should succeed
        let result = reg.reserve("b", 90, 0, 500);
        assert!(result.is_ok());
    }

    #[test]
    fn test_consume_marks_consumed() {
        let mut reg = ReservationRegistry::new(100);
        let id = reg.reserve("job", 20, 0, 500).expect("reserve");
        assert!(reg.consume(id));
        let r = reg
            .all()
            .find(|r| r.id == id)
            .expect("reservation should exist");
        assert_eq!(r.status, ReservationStatus::Consumed);
    }

    #[test]
    fn test_expire_old() {
        let mut reg = ReservationRegistry::new(100);
        let id = reg.reserve("job", 20, 0, 100).expect("reserve");
        reg.expire_old(200);
        let r = reg.all().find(|r| r.id == id).expect("should exist");
        assert_eq!(r.status, ReservationStatus::Expired);
    }

    #[test]
    fn test_reserved_at_outside_window_is_zero() {
        let mut reg = ReservationRegistry::new(100);
        reg.reserve("job", 50, 1000, 2000).expect("reserve");
        // Before window
        assert_eq!(reg.reserved_at(500), 0);
        // After window
        assert_eq!(reg.reserved_at(3000), 0);
    }

    #[test]
    fn test_zero_slots_rejected() {
        let mut reg = ReservationRegistry::new(100);
        assert!(reg.reserve("bad", 0, 0, 100).is_err());
    }

    #[test]
    fn test_invalid_window_rejected() {
        let mut reg = ReservationRegistry::new(100);
        assert!(reg.reserve("bad", 5, 500, 100).is_err());
    }
}
