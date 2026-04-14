//! Calibration schedule — track when devices are due for re-calibration.
//!
//! Professional colour pipelines require regular re-calibration of cameras
//! and displays. This module maintains a registry of devices with their last
//! calibration date and the maximum allowed interval between calibrations.
//! It provides pass/fail status, overdue detection, and priority-sorted work
//! lists for calibration technicians.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Device types ─────────────────────────────────────────────────────────────

/// The kind of device in the calibration schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceKind {
    /// A cinema or broadcast camera.
    Camera,
    /// A reference display monitor.
    Display,
    /// A printing device.
    Printer,
    /// A densitometer or spectrophotometer.
    MeasurementDevice,
    /// Other / custom device.
    Other,
}

impl std::fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Camera => write!(f, "Camera"),
            Self::Display => write!(f, "Display"),
            Self::Printer => write!(f, "Printer"),
            Self::MeasurementDevice => write!(f, "MeasurementDevice"),
            Self::Other => write!(f, "Other"),
        }
    }
}

// ─── Calibration status ───────────────────────────────────────────────────────

/// Current calibration status for a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationStatus {
    /// Calibration is current (within the allowed interval).
    Current,
    /// Calibration is due soon (within the warning window).
    DueSoon,
    /// Calibration is overdue.
    Overdue,
    /// Device has never been calibrated.
    NeverCalibrated,
}

impl std::fmt::Display for CalibrationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Current => write!(f, "Current"),
            Self::DueSoon => write!(f, "DueSoon"),
            Self::Overdue => write!(f, "Overdue"),
            Self::NeverCalibrated => write!(f, "NeverCalibrated"),
        }
    }
}

// ─── Device entry ─────────────────────────────────────────────────────────────

/// A device registered in the calibration schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceEntry {
    /// Unique device identifier.
    pub device_id: String,
    /// Human-readable device name.
    pub name: String,
    /// Type of device.
    pub kind: DeviceKind,
    /// When the device was last calibrated (None if never).
    pub last_calibrated: Option<DateTime<Utc>>,
    /// Maximum interval between calibrations in days.
    pub interval_days: u32,
    /// Number of days before the due date at which `DueSoon` is flagged.
    pub warning_days: u32,
    /// Optional notes.
    pub notes: Option<String>,
}

impl DeviceEntry {
    /// Create a new device entry.
    ///
    /// `interval_days` — maximum number of days between calibrations.
    /// `warning_days`  — days before the due date to flag as DueSoon.
    #[must_use]
    pub fn new(
        device_id: impl Into<String>,
        name: impl Into<String>,
        kind: DeviceKind,
        interval_days: u32,
    ) -> Self {
        Self {
            device_id: device_id.into(),
            name: name.into(),
            kind,
            last_calibrated: None,
            interval_days,
            warning_days: 7,
            notes: None,
        }
    }

    /// Set the warning window.
    #[must_use]
    pub fn with_warning_days(mut self, days: u32) -> Self {
        self.warning_days = days;
        self
    }

    /// Set notes.
    #[must_use]
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }

    /// Record a calibration at `when`.
    pub fn record_calibration(&mut self, when: DateTime<Utc>) {
        self.last_calibrated = Some(when);
    }

    /// Record calibration as of now (UTC).
    pub fn record_calibration_now(&mut self) {
        self.last_calibrated = Some(Utc::now());
    }

    /// Compute the due date (last_calibrated + interval_days), or None if
    /// never calibrated.
    #[must_use]
    pub fn due_date(&self) -> Option<DateTime<Utc>> {
        self.last_calibrated
            .map(|lc| lc + Duration::days(i64::from(self.interval_days)))
    }

    /// Compute the calibration status relative to `now`.
    #[must_use]
    pub fn status_at(&self, now: DateTime<Utc>) -> CalibrationStatus {
        match self.last_calibrated {
            None => CalibrationStatus::NeverCalibrated,
            Some(lc) => {
                let due = lc + Duration::days(i64::from(self.interval_days));
                let warn = due - Duration::days(i64::from(self.warning_days));
                if now >= due {
                    CalibrationStatus::Overdue
                } else if now >= warn {
                    CalibrationStatus::DueSoon
                } else {
                    CalibrationStatus::Current
                }
            }
        }
    }

    /// How many days until (positive) or since (negative) the due date.
    #[must_use]
    pub fn days_until_due(&self, now: DateTime<Utc>) -> Option<i64> {
        self.due_date().map(|due| (due - now).num_days())
    }
}

// ─── CalibrationSchedule ─────────────────────────────────────────────────────

/// A registry of devices and their calibration schedules.
pub struct CalibrationSchedule {
    devices: HashMap<String, DeviceEntry>,
}

impl CalibrationSchedule {
    /// Create an empty schedule.
    #[must_use]
    pub fn new() -> Self {
        Self {
            devices: HashMap::new(),
        }
    }

    /// Register a device.
    pub fn register(&mut self, entry: DeviceEntry) {
        self.devices.insert(entry.device_id.clone(), entry);
    }

    /// Remove a device by ID. Returns the removed entry, if any.
    pub fn unregister(&mut self, device_id: &str) -> Option<DeviceEntry> {
        self.devices.remove(device_id)
    }

    /// Look up a device by ID.
    #[must_use]
    pub fn device(&self, device_id: &str) -> Option<&DeviceEntry> {
        self.devices.get(device_id)
    }

    /// Mutable access to a device.
    pub fn device_mut(&mut self, device_id: &str) -> Option<&mut DeviceEntry> {
        self.devices.get_mut(device_id)
    }

    /// Number of registered devices.
    #[must_use]
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Return all devices with `Overdue` or `NeverCalibrated` status at `now`.
    #[must_use]
    pub fn overdue_devices(&self, now: DateTime<Utc>) -> Vec<&DeviceEntry> {
        self.devices
            .values()
            .filter(|d| {
                matches!(
                    d.status_at(now),
                    CalibrationStatus::Overdue | CalibrationStatus::NeverCalibrated
                )
            })
            .collect()
    }

    /// Return all devices with `DueSoon` status at `now`.
    #[must_use]
    pub fn due_soon_devices(&self, now: DateTime<Utc>) -> Vec<&DeviceEntry> {
        self.devices
            .values()
            .filter(|d| d.status_at(now) == CalibrationStatus::DueSoon)
            .collect()
    }

    /// Produce a work list sorted by urgency: Overdue/NeverCalibrated first,
    /// DueSoon next, then Current (ascending days until due).
    #[must_use]
    pub fn work_list(&self, now: DateTime<Utc>) -> Vec<WorkListEntry> {
        let mut entries: Vec<WorkListEntry> = self
            .devices
            .values()
            .map(|d| WorkListEntry {
                device_id: d.device_id.clone(),
                name: d.name.clone(),
                kind: d.kind,
                status: d.status_at(now),
                days_until_due: d.days_until_due(now),
            })
            .collect();

        entries.sort_by(|a, b| {
            urgency_score(a.status)
                .cmp(&urgency_score(b.status))
                .then_with(|| {
                    let da = a.days_until_due.unwrap_or(i64::MIN);
                    let db = b.days_until_due.unwrap_or(i64::MIN);
                    da.cmp(&db)
                })
        });

        entries
    }

    /// Check whether all devices are currently calibrated.
    #[must_use]
    pub fn all_current(&self, now: DateTime<Utc>) -> bool {
        self.devices
            .values()
            .all(|d| d.status_at(now) == CalibrationStatus::Current)
    }
}

impl Default for CalibrationSchedule {
    fn default() -> Self {
        Self::new()
    }
}

/// An entry in the sorted work list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkListEntry {
    /// Device ID.
    pub device_id: String,
    /// Device name.
    pub name: String,
    /// Device kind.
    pub kind: DeviceKind,
    /// Current calibration status.
    pub status: CalibrationStatus,
    /// Days until (or since, if negative) the due date.
    pub days_until_due: Option<i64>,
}

fn urgency_score(status: CalibrationStatus) -> u8 {
    match status {
        CalibrationStatus::NeverCalibrated => 0,
        CalibrationStatus::Overdue => 1,
        CalibrationStatus::DueSoon => 2,
        CalibrationStatus::Current => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn fixed_now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 3, 15, 12, 0, 0).unwrap()
    }

    fn calibrated_entry(interval_days: u32, days_ago: i64) -> DeviceEntry {
        let mut e = DeviceEntry::new("D1", "Monitor A", DeviceKind::Display, interval_days);
        let now = fixed_now();
        e.record_calibration(now - Duration::days(days_ago));
        e
    }

    #[test]
    fn test_status_never_calibrated() {
        let e = DeviceEntry::new("D", "Cam", DeviceKind::Camera, 30);
        assert_eq!(e.status_at(fixed_now()), CalibrationStatus::NeverCalibrated);
    }

    #[test]
    fn test_status_current() {
        let e = calibrated_entry(30, 10); // calibrated 10 days ago, due in 20
        assert_eq!(e.status_at(fixed_now()), CalibrationStatus::Current);
    }

    #[test]
    fn test_status_due_soon() {
        let mut e = calibrated_entry(30, 25); // due in 5 days, warning=7
        e.warning_days = 7;
        assert_eq!(e.status_at(fixed_now()), CalibrationStatus::DueSoon);
    }

    #[test]
    fn test_status_overdue() {
        let e = calibrated_entry(30, 35); // overdue by 5 days
        assert_eq!(e.status_at(fixed_now()), CalibrationStatus::Overdue);
    }

    #[test]
    fn test_days_until_due_positive() {
        let e = calibrated_entry(30, 10);
        let days = e.days_until_due(fixed_now()).unwrap();
        assert_eq!(days, 20);
    }

    #[test]
    fn test_days_until_due_negative() {
        let e = calibrated_entry(30, 35);
        let days = e.days_until_due(fixed_now()).unwrap();
        assert_eq!(days, -5);
    }

    #[test]
    fn test_days_until_due_never_calibrated() {
        let e = DeviceEntry::new("D", "Cam", DeviceKind::Camera, 30);
        assert!(e.days_until_due(fixed_now()).is_none());
    }

    #[test]
    fn test_due_date_none_when_never_calibrated() {
        let e = DeviceEntry::new("D", "Cam", DeviceKind::Camera, 30);
        assert!(e.due_date().is_none());
    }

    #[test]
    fn test_schedule_register_and_count() {
        let mut sched = CalibrationSchedule::new();
        let e = DeviceEntry::new("D1", "Monitor", DeviceKind::Display, 30);
        sched.register(e);
        assert_eq!(sched.device_count(), 1);
    }

    #[test]
    fn test_schedule_unregister() {
        let mut sched = CalibrationSchedule::new();
        sched.register(DeviceEntry::new("D1", "X", DeviceKind::Display, 30));
        let removed = sched.unregister("D1");
        assert!(removed.is_some());
        assert_eq!(sched.device_count(), 0);
    }

    #[test]
    fn test_schedule_overdue_devices() {
        let mut sched = CalibrationSchedule::new();
        sched.register(calibrated_entry(30, 10).into_id("current"));
        sched.register(calibrated_entry(30, 35).into_id("overdue"));
        sched.register({
            let mut e = DeviceEntry::new("never", "Never", DeviceKind::Camera, 30);
            e
        });
        let overdue = sched.overdue_devices(fixed_now());
        assert_eq!(overdue.len(), 2); // overdue + never
    }

    #[test]
    fn test_schedule_due_soon_devices() {
        let mut sched = CalibrationSchedule::new();
        let mut e = calibrated_entry(30, 25);
        e.device_id = "due_soon".to_string();
        e.warning_days = 7;
        sched.register(e);
        let soon = sched.due_soon_devices(fixed_now());
        assert_eq!(soon.len(), 1);
    }

    #[test]
    fn test_schedule_all_current() {
        let mut sched = CalibrationSchedule::new();
        sched.register(calibrated_entry(30, 5).into_id("A"));
        sched.register(calibrated_entry(30, 8).into_id("B"));
        assert!(sched.all_current(fixed_now()));
    }

    #[test]
    fn test_schedule_not_all_current_when_overdue() {
        let mut sched = CalibrationSchedule::new();
        sched.register(calibrated_entry(30, 5).into_id("A"));
        sched.register(calibrated_entry(30, 40).into_id("B_overdue"));
        assert!(!sched.all_current(fixed_now()));
    }

    #[test]
    fn test_work_list_ordering() {
        let mut sched = CalibrationSchedule::new();
        sched.register(calibrated_entry(30, 5).into_id("current"));
        sched.register(calibrated_entry(30, 40).into_id("overdue"));
        let mut never = DeviceEntry::new("never", "X", DeviceKind::Camera, 30);
        never.device_id = "never".to_string();
        sched.register(never);

        let list = sched.work_list(fixed_now());
        assert_eq!(list.len(), 3);
        // First entry should be NeverCalibrated or Overdue (urgency 0/1)
        assert!(matches!(
            list[0].status,
            CalibrationStatus::NeverCalibrated | CalibrationStatus::Overdue
        ));
        // Last should be Current
        assert_eq!(list[2].status, CalibrationStatus::Current);
    }

    #[test]
    fn test_device_kind_display() {
        assert_eq!(DeviceKind::Camera.to_string(), "Camera");
        assert_eq!(DeviceKind::Display.to_string(), "Display");
        assert_eq!(DeviceKind::Printer.to_string(), "Printer");
    }

    #[test]
    fn test_calibration_status_display() {
        assert_eq!(CalibrationStatus::Current.to_string(), "Current");
        assert_eq!(CalibrationStatus::Overdue.to_string(), "Overdue");
        assert_eq!(CalibrationStatus::NeverCalibrated.to_string(), "NeverCalibrated");
    }
}

// ─── Test helper ─────────────────────────────────────────────────────────────

impl DeviceEntry {
    /// Test helper: override `device_id`.
    #[cfg(test)]
    fn into_id(mut self, id: &str) -> Self {
        self.device_id = id.to_string();
        self
    }
}
