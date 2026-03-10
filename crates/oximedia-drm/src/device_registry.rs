#![allow(dead_code)]
//! Device registry for DRM playback authorization.
//!
//! Tracks registered devices per user, enforcing per-user device limits
//! and providing deauthorization to free up slots.

use std::collections::HashMap;
use std::time::SystemTime;

/// Category of a registered device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceType {
    /// Smartphone or tablet.
    Mobile,
    /// Laptop or desktop computer.
    Desktop,
    /// Smart TV or set-top box.
    Television,
    /// Games console.
    GameConsole,
    /// Web browser (no dedicated app).
    WebBrowser,
}

impl DeviceType {
    /// Returns `true` if this device type is considered a mobile device.
    pub fn is_mobile(&self) -> bool {
        matches!(self, DeviceType::Mobile)
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            DeviceType::Mobile => "Mobile",
            DeviceType::Desktop => "Desktop",
            DeviceType::Television => "Television",
            DeviceType::GameConsole => "Game Console",
            DeviceType::WebBrowser => "Web Browser",
        }
    }
}

/// Metadata for a single registered device.
#[derive(Debug, Clone)]
pub struct DeviceRecord {
    /// Globally unique device identifier (e.g. hardware fingerprint hash).
    pub device_id: String,
    /// Human-readable display name chosen by the user.
    pub display_name: String,
    /// Type of device.
    pub device_type: DeviceType,
    /// User who owns this record.
    pub user_id: String,
    /// When the device was first registered.
    pub registered_at: SystemTime,
    /// Whether the device is currently authorized for playback.
    pub authorized: bool,
}

impl DeviceRecord {
    /// Create a new, authorized device record.
    pub fn new(
        device_id: impl Into<String>,
        display_name: impl Into<String>,
        device_type: DeviceType,
        user_id: impl Into<String>,
    ) -> Self {
        Self {
            device_id: device_id.into(),
            display_name: display_name.into(),
            device_type,
            user_id: user_id.into(),
            registered_at: SystemTime::now(),
            authorized: true,
        }
    }

    /// Returns `true` when the device record is currently authorized.
    pub fn is_authorized(&self) -> bool {
        self.authorized
    }
}

/// Registry that stores device records and enforces per-user device limits.
#[derive(Debug)]
pub struct DeviceRegistry {
    /// All records, keyed by device_id.
    records: HashMap<String, DeviceRecord>,
    /// Maximum number of simultaneously-authorized devices per user.
    max_devices_per_user: usize,
}

impl DeviceRegistry {
    /// Create a new registry with the given per-user device cap.
    pub fn new(max_devices_per_user: usize) -> Self {
        Self {
            records: HashMap::new(),
            max_devices_per_user,
        }
    }

    /// Register a new device for a user.
    ///
    /// Returns `Err(String)` if the user has already reached the device limit
    /// (counting only authorized records).
    pub fn register(&mut self, record: DeviceRecord) -> Result<(), String> {
        let authorized_count = self.authorized_count(&record.user_id);
        if authorized_count >= self.max_devices_per_user {
            return Err(format!(
                "User '{}' has reached the device limit of {}",
                record.user_id, self.max_devices_per_user
            ));
        }
        self.records.insert(record.device_id.clone(), record);
        Ok(())
    }

    /// Deauthorize a device by its ID. Returns `true` if the record existed.
    pub fn deauthorize(&mut self, device_id: &str) -> bool {
        if let Some(record) = self.records.get_mut(device_id) {
            record.authorized = false;
            true
        } else {
            false
        }
    }

    /// Remove a device record entirely. Returns `true` if it existed.
    pub fn remove(&mut self, device_id: &str) -> bool {
        self.records.remove(device_id).is_some()
    }

    /// Count of authorized devices for the given user.
    pub fn authorized_count(&self, user_id: &str) -> usize {
        self.records
            .values()
            .filter(|r| r.user_id == user_id && r.authorized)
            .count()
    }

    /// Retrieve a device record by ID.
    pub fn get(&self, device_id: &str) -> Option<&DeviceRecord> {
        self.records.get(device_id)
    }

    /// All authorized devices for a user.
    pub fn authorized_devices(&self, user_id: &str) -> Vec<&DeviceRecord> {
        self.records
            .values()
            .filter(|r| r.user_id == user_id && r.authorized)
            .collect()
    }

    /// Total records in the registry (authorized + deauthorized).
    pub fn total_count(&self) -> usize {
        self.records.len()
    }
}

impl Default for DeviceRegistry {
    fn default() -> Self {
        Self::new(5)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(device_id: &str, user_id: &str, dtype: DeviceType) -> DeviceRecord {
        DeviceRecord::new(device_id, "Test Device", dtype, user_id)
    }

    // DeviceType tests

    #[test]
    fn test_mobile_is_mobile() {
        assert!(DeviceType::Mobile.is_mobile());
    }

    #[test]
    fn test_desktop_is_not_mobile() {
        assert!(!DeviceType::Desktop.is_mobile());
    }

    #[test]
    fn test_television_is_not_mobile() {
        assert!(!DeviceType::Television.is_mobile());
    }

    #[test]
    fn test_device_type_name_non_empty() {
        for dt in &[
            DeviceType::Mobile,
            DeviceType::Desktop,
            DeviceType::Television,
            DeviceType::GameConsole,
            DeviceType::WebBrowser,
        ] {
            assert!(!dt.name().is_empty());
        }
    }

    // DeviceRecord tests

    #[test]
    fn test_new_record_is_authorized() {
        let r = make_record("d1", "u1", DeviceType::Desktop);
        assert!(r.is_authorized());
    }

    #[test]
    fn test_deauthorized_record() {
        let mut r = make_record("d2", "u1", DeviceType::Mobile);
        r.authorized = false;
        assert!(!r.is_authorized());
    }

    // DeviceRegistry tests

    #[test]
    fn test_register_single_device() {
        let mut reg = DeviceRegistry::new(3);
        let r = make_record("d1", "u1", DeviceType::Desktop);
        assert!(reg.register(r).is_ok());
        assert_eq!(reg.authorized_count("u1"), 1);
    }

    #[test]
    fn test_register_exceeds_limit() {
        let mut reg = DeviceRegistry::new(2);
        reg.register(make_record("d1", "u1", DeviceType::Desktop))
            .expect("register should succeed");
        reg.register(make_record("d2", "u1", DeviceType::Mobile))
            .expect("register should succeed");
        let result = reg.register(make_record("d3", "u1", DeviceType::WebBrowser));
        assert!(result.is_err());
    }

    #[test]
    fn test_deauthorize_frees_slot() {
        let mut reg = DeviceRegistry::new(2);
        reg.register(make_record("d1", "u1", DeviceType::Desktop))
            .expect("register should succeed");
        reg.register(make_record("d2", "u1", DeviceType::Mobile))
            .expect("register should succeed");
        assert!(reg.deauthorize("d1"));
        // Now only 1 authorized, so registration should succeed
        assert!(reg
            .register(make_record("d3", "u1", DeviceType::Television))
            .is_ok());
    }

    #[test]
    fn test_deauthorize_nonexistent_returns_false() {
        let mut reg = DeviceRegistry::new(3);
        assert!(!reg.deauthorize("no_device"));
    }

    #[test]
    fn test_authorized_count_excludes_deauthorized() {
        let mut reg = DeviceRegistry::new(5);
        reg.register(make_record("d1", "u2", DeviceType::Desktop))
            .expect("register should succeed");
        reg.register(make_record("d2", "u2", DeviceType::Mobile))
            .expect("register should succeed");
        reg.deauthorize("d1");
        assert_eq!(reg.authorized_count("u2"), 1);
    }

    #[test]
    fn test_remove_device() {
        let mut reg = DeviceRegistry::new(3);
        reg.register(make_record("d1", "u1", DeviceType::Desktop))
            .expect("register should succeed");
        assert!(reg.remove("d1"));
        assert_eq!(reg.total_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_returns_false() {
        let mut reg = DeviceRegistry::new(3);
        assert!(!reg.remove("ghost"));
    }

    #[test]
    fn test_get_device() {
        let mut reg = DeviceRegistry::new(3);
        reg.register(make_record("d1", "u1", DeviceType::Television))
            .expect("register should succeed");
        let record = reg.get("d1").expect("record should exist");
        assert_eq!(record.device_id, "d1");
    }

    #[test]
    fn test_authorized_devices_list() {
        let mut reg = DeviceRegistry::new(5);
        reg.register(make_record("d1", "u3", DeviceType::Desktop))
            .expect("register should succeed");
        reg.register(make_record("d2", "u3", DeviceType::Mobile))
            .expect("register should succeed");
        reg.register(make_record("d3", "u4", DeviceType::Television))
            .expect("register should succeed");
        reg.deauthorize("d1");
        let devices = reg.authorized_devices("u3");
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].device_id, "d2");
    }

    #[test]
    fn test_different_users_independent_limits() {
        let mut reg = DeviceRegistry::new(1);
        reg.register(make_record("d1", "u1", DeviceType::Desktop))
            .expect("register should succeed");
        // u2 should be unaffected by u1's usage
        assert!(reg
            .register(make_record("d2", "u2", DeviceType::Mobile))
            .is_ok());
    }
}
