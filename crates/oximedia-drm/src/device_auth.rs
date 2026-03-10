//! Device registration and authorization for DRM.
//!
//! Provides:
//! - [`DeviceType`]: categories of end-user devices
//! - [`DeviceRegistration`]: a registered device record
//! - [`DeviceLimit`]: maximum device and stream counts per account
//! - [`DeviceManager`]: manages a collection of device registrations

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// DeviceType
// ---------------------------------------------------------------------------

/// Categories of end-user playback devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// Smartphone or feature phone.
    Mobile,
    /// Tablet device.
    Tablet,
    /// Desktop or laptop computer.
    Desktop,
    /// Smart TV or connected television.
    SmartTv,
    /// Dedicated streaming device (e.g., streaming stick, set-top box).
    StreamingDevice,
}

impl DeviceType {
    /// Returns `true` if the device type is a mobile device (phone or tablet).
    #[must_use]
    pub fn is_mobile(self) -> bool {
        matches!(self, DeviceType::Mobile | DeviceType::Tablet)
    }

    /// Typical maximum supported resolution height (in pixels) for this device type.
    #[must_use]
    pub fn typical_max_resolution(self) -> u32 {
        match self {
            DeviceType::Mobile => 1080,
            DeviceType::Tablet => 1080,
            DeviceType::Desktop => 2160,
            DeviceType::SmartTv => 2160,
            DeviceType::StreamingDevice => 2160,
        }
    }

    /// Human-readable name of the device type.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            DeviceType::Mobile => "Mobile",
            DeviceType::Tablet => "Tablet",
            DeviceType::Desktop => "Desktop",
            DeviceType::SmartTv => "Smart TV",
            DeviceType::StreamingDevice => "Streaming Device",
        }
    }
}

// ---------------------------------------------------------------------------
// DeviceRegistration
// ---------------------------------------------------------------------------

/// A record of a registered device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceRegistration {
    /// Unique device identifier (e.g., a UUID string or hardware fingerprint).
    pub device_id: String,
    /// Category of the device.
    pub device_type: DeviceType,
    /// Unix epoch timestamp (seconds) when the device was registered.
    pub registered_epoch: u64,
    /// Whether the device is currently active.
    pub is_active: bool,
}

impl DeviceRegistration {
    /// Create a new active registration.
    #[must_use]
    pub fn new(device_id: impl Into<String>, device_type: DeviceType, epoch: u64) -> Self {
        Self {
            device_id: device_id.into(),
            device_type,
            registered_epoch: epoch,
            is_active: true,
        }
    }

    /// Returns `true` if the registration is valid (non-empty ID and active).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.device_id.is_empty() && self.is_active
    }

    /// Deactivate the device (mark it as no longer authorised).
    pub fn deactivate(&mut self) {
        self.is_active = false;
    }

    /// Re-activate the device.
    pub fn activate(&mut self) {
        self.is_active = true;
    }
}

// ---------------------------------------------------------------------------
// DeviceLimit
// ---------------------------------------------------------------------------

/// Account-level device and stream limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceLimit {
    /// Maximum number of registered devices allowed on the account.
    pub max_devices: u32,
    /// Maximum number of simultaneously active streams.
    pub max_concurrent_streams: u32,
}

impl DeviceLimit {
    /// Standard plan: 5 devices, 2 concurrent streams.
    #[must_use]
    pub fn standard() -> Self {
        Self {
            max_devices: 5,
            max_concurrent_streams: 2,
        }
    }

    /// Premium plan: 10 devices, 4 concurrent streams.
    #[must_use]
    pub fn premium() -> Self {
        Self {
            max_devices: 10,
            max_concurrent_streams: 4,
        }
    }

    /// Family plan: 6 devices, 6 concurrent streams.
    #[must_use]
    pub fn family() -> Self {
        Self {
            max_devices: 6,
            max_concurrent_streams: 6,
        }
    }
}

// ---------------------------------------------------------------------------
// DeviceManager
// ---------------------------------------------------------------------------

/// Manages device registrations for a single account.
pub struct DeviceManager {
    /// All registered devices (active and inactive).
    pub registrations: Vec<DeviceRegistration>,
    /// Account-level limits.
    pub limit: DeviceLimit,
}

impl DeviceManager {
    /// Create a new device manager with standard limits.
    #[must_use]
    pub fn new() -> Self {
        Self {
            registrations: Vec::new(),
            limit: DeviceLimit::standard(),
        }
    }

    /// Create with a specific limit.
    #[must_use]
    pub fn with_limit(limit: DeviceLimit) -> Self {
        Self {
            registrations: Vec::new(),
            limit,
        }
    }

    /// Register a new device.  Returns `true` on success, `false` if the
    /// active device limit has been reached or the device ID is already registered.
    pub fn register(&mut self, registration: DeviceRegistration) -> bool {
        // Reject if already registered
        if self
            .registrations
            .iter()
            .any(|r| r.device_id == registration.device_id)
        {
            return false;
        }
        // Reject if at capacity
        if !self.can_register() {
            return false;
        }
        self.registrations.push(registration);
        true
    }

    /// Deregister (remove) a device by its ID.  Returns `true` if found and removed.
    pub fn deregister(&mut self, device_id: &str) -> bool {
        if let Some(pos) = self
            .registrations
            .iter()
            .position(|r| r.device_id == device_id)
        {
            self.registrations.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Deactivate a device without removing it from the list.
    pub fn deactivate(&mut self, device_id: &str) -> bool {
        if let Some(reg) = self
            .registrations
            .iter_mut()
            .find(|r| r.device_id == device_id)
        {
            reg.deactivate();
            true
        } else {
            false
        }
    }

    /// Return the number of currently active registrations.
    #[must_use]
    pub fn active_count(&self) -> u32 {
        self.registrations
            .iter()
            .filter(|r| r.is_active)
            .count()
            .try_into()
            .unwrap_or(u32::MAX)
    }

    /// Returns `true` if a new device can be registered (active count < limit).
    #[must_use]
    pub fn can_register(&self) -> bool {
        self.active_count() < self.limit.max_devices
    }

    /// Find a registration by device ID.
    #[must_use]
    pub fn find_device(&self, device_id: &str) -> Option<&DeviceRegistration> {
        self.registrations.iter().find(|r| r.device_id == device_id)
    }

    /// Return all active registrations.
    #[must_use]
    pub fn active_devices(&self) -> Vec<&DeviceRegistration> {
        self.registrations.iter().filter(|r| r.is_active).collect()
    }

    /// Total number of registrations (active + inactive).
    #[must_use]
    pub fn total_registered(&self) -> usize {
        self.registrations.len()
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn mobile_reg(id: &str) -> DeviceRegistration {
        DeviceRegistration::new(id, DeviceType::Mobile, 1_000_000)
    }

    // ---- DeviceType ----

    #[test]
    fn test_device_type_is_mobile_true() {
        assert!(DeviceType::Mobile.is_mobile());
        assert!(DeviceType::Tablet.is_mobile());
    }

    #[test]
    fn test_device_type_is_mobile_false() {
        assert!(!DeviceType::Desktop.is_mobile());
        assert!(!DeviceType::SmartTv.is_mobile());
        assert!(!DeviceType::StreamingDevice.is_mobile());
    }

    #[test]
    fn test_device_type_typical_max_resolution() {
        assert_eq!(DeviceType::Mobile.typical_max_resolution(), 1080);
        assert_eq!(DeviceType::Desktop.typical_max_resolution(), 2160);
        assert_eq!(DeviceType::SmartTv.typical_max_resolution(), 2160);
    }

    #[test]
    fn test_device_type_name() {
        assert_eq!(DeviceType::Mobile.name(), "Mobile");
        assert_eq!(DeviceType::SmartTv.name(), "Smart TV");
    }

    // ---- DeviceRegistration ----

    #[test]
    fn test_registration_valid_when_active() {
        let r = mobile_reg("device-1");
        assert!(r.is_valid());
    }

    #[test]
    fn test_registration_invalid_when_empty_id() {
        let r = DeviceRegistration::new("", DeviceType::Mobile, 0);
        assert!(!r.is_valid());
    }

    #[test]
    fn test_registration_deactivate() {
        let mut r = mobile_reg("device-2");
        assert!(r.is_active);
        r.deactivate();
        assert!(!r.is_active);
        assert!(!r.is_valid());
    }

    #[test]
    fn test_registration_reactivate() {
        let mut r = mobile_reg("device-3");
        r.deactivate();
        r.activate();
        assert!(r.is_active);
        assert!(r.is_valid());
    }

    // ---- DeviceLimit ----

    #[test]
    fn test_standard_limit() {
        let l = DeviceLimit::standard();
        assert_eq!(l.max_devices, 5);
        assert_eq!(l.max_concurrent_streams, 2);
    }

    #[test]
    fn test_premium_limit() {
        let l = DeviceLimit::premium();
        assert_eq!(l.max_devices, 10);
        assert_eq!(l.max_concurrent_streams, 4);
    }

    // ---- DeviceManager ----

    #[test]
    fn test_device_manager_register_success() {
        let mut m = DeviceManager::new();
        let ok = m.register(mobile_reg("dev-a"));
        assert!(ok);
        assert_eq!(m.active_count(), 1);
    }

    #[test]
    fn test_device_manager_register_duplicate_rejected() {
        let mut m = DeviceManager::new();
        m.register(mobile_reg("dev-x"));
        let ok = m.register(mobile_reg("dev-x"));
        assert!(!ok);
        assert_eq!(m.active_count(), 1);
    }

    #[test]
    fn test_device_manager_register_at_capacity() {
        let mut m = DeviceManager::with_limit(DeviceLimit {
            max_devices: 2,
            max_concurrent_streams: 1,
        });
        assert!(m.register(mobile_reg("d-1")));
        assert!(m.register(mobile_reg("d-2")));
        assert!(!m.register(mobile_reg("d-3")), "Should be at capacity");
    }

    #[test]
    fn test_device_manager_deregister() {
        let mut m = DeviceManager::new();
        m.register(mobile_reg("d-del"));
        let ok = m.deregister("d-del");
        assert!(ok);
        assert_eq!(m.active_count(), 0);
    }

    #[test]
    fn test_device_manager_deregister_nonexistent() {
        let mut m = DeviceManager::new();
        assert!(!m.deregister("ghost"));
    }

    #[test]
    fn test_device_manager_deactivate() {
        let mut m = DeviceManager::new();
        m.register(mobile_reg("d-off"));
        m.deactivate("d-off");
        assert_eq!(m.active_count(), 0);
        assert_eq!(m.total_registered(), 1);
    }

    #[test]
    fn test_device_manager_find_device() {
        let mut m = DeviceManager::new();
        m.register(mobile_reg("d-find"));
        let found = m.find_device("d-find");
        assert!(found.is_some());
        assert_eq!(found.expect("device should be found").device_id, "d-find");
    }

    #[test]
    fn test_device_manager_find_device_not_found() {
        let m = DeviceManager::new();
        assert!(m.find_device("nonexistent").is_none());
    }

    #[test]
    fn test_device_manager_active_devices() {
        let mut m = DeviceManager::new();
        m.register(mobile_reg("d-1"));
        m.register(mobile_reg("d-2"));
        m.deactivate("d-1");
        let active = m.active_devices();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].device_id, "d-2");
    }

    #[test]
    fn test_device_manager_can_register_false_at_limit() {
        let mut m = DeviceManager::with_limit(DeviceLimit {
            max_devices: 1,
            max_concurrent_streams: 1,
        });
        m.register(mobile_reg("solo"));
        assert!(!m.can_register());
    }
}
