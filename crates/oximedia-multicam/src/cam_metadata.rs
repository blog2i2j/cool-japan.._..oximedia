//! Camera metadata registry: sensor profiles and camera type classification.

#![allow(dead_code)]

use std::collections::HashMap;

/// Classification of camera sensor type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CameraType {
    /// Consumer-grade phone camera
    Phone,
    /// Consumer camcorder
    Consumer,
    /// Professional broadcast camera
    Broadcast,
    /// Cinema camera (S35, full-frame, large format)
    Cinema,
    /// Action / sports camera (small body)
    Action,
    /// PTZ robotic camera
    Ptz,
    /// Other / unknown
    Other(String),
}

impl CameraType {
    /// Returns the approximate sensor diagonal in millimetres for well-known types.
    ///
    /// Returns `None` for `Other` or categories without a canonical size.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn sensor_size_mm(&self) -> Option<f64> {
        match self {
            Self::Phone => Some(8.0),
            Self::Consumer => Some(11.3),
            Self::Action => Some(9.6),
            Self::Broadcast => Some(16.0),
            Self::Cinema => Some(43.3),
            Self::Ptz => Some(14.1),
            Self::Other(_) => None,
        }
    }

    /// Returns `true` if the type is a known professional tier (broadcast or cinema).
    #[must_use]
    pub fn is_professional(&self) -> bool {
        matches!(self, Self::Broadcast | Self::Cinema)
    }
}

/// Metadata describing a single camera body.
#[derive(Debug, Clone)]
pub struct CameraMetadata {
    /// Human-readable model name.
    pub model: String,
    /// Manufacturer name.
    pub manufacturer: String,
    /// Camera type classification.
    pub camera_type: CameraType,
    /// Horizontal resolution in pixels.
    pub width_px: u32,
    /// Vertical resolution in pixels.
    pub height_px: u32,
    /// Maximum frame rate in fps.
    pub max_fps: f32,
}

impl CameraMetadata {
    /// Creates a new `CameraMetadata` entry.
    pub fn new(
        model: impl Into<String>,
        manufacturer: impl Into<String>,
        camera_type: CameraType,
        width_px: u32,
        height_px: u32,
        max_fps: f32,
    ) -> Self {
        Self {
            model: model.into(),
            manufacturer: manufacturer.into(),
            camera_type,
            width_px,
            height_px,
            max_fps,
        }
    }

    /// Returns `true` when the camera supports at least 3840 × 2160 (4K UHD).
    #[must_use]
    pub fn is_4k(&self) -> bool {
        self.width_px >= 3840 && self.height_px >= 2160
    }

    /// Returns `true` when the camera supports 8K (7680 × 4320) or higher.
    #[must_use]
    pub fn is_8k(&self) -> bool {
        self.width_px >= 7680 && self.height_px >= 4320
    }

    /// Returns the pixel count (total megapixels, rounded).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn megapixels(&self) -> f64 {
        (f64::from(self.width_px) * f64::from(self.height_px)) / 1_000_000.0
    }
}

/// In-memory registry of known camera models.
#[derive(Debug, Default)]
pub struct CameraRegistry {
    cameras: HashMap<String, CameraMetadata>,
}

impl CameraRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a camera model.
    ///
    /// If a model with the same name already exists it is replaced and the old
    /// value is returned.
    pub fn register(&mut self, meta: CameraMetadata) -> Option<CameraMetadata> {
        self.cameras.insert(meta.model.clone(), meta)
    }

    /// Looks up a camera by exact model name.
    #[must_use]
    pub fn find_by_model(&self, model: &str) -> Option<&CameraMetadata> {
        self.cameras.get(model)
    }

    /// Returns all cameras matching a given `CameraType`.
    #[must_use]
    pub fn find_by_type(&self, camera_type: &CameraType) -> Vec<&CameraMetadata> {
        self.cameras
            .values()
            .filter(|m| &m.camera_type == camera_type)
            .collect()
    }

    /// Returns the total number of registered cameras.
    #[must_use]
    pub fn count(&self) -> usize {
        self.cameras.len()
    }

    /// Returns `true` if no cameras are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cameras.is_empty()
    }

    /// Removes a camera by model name.  Returns the removed entry if present.
    pub fn remove(&mut self, model: &str) -> Option<CameraMetadata> {
        self.cameras.remove(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cam(model: &str, ct: CameraType, w: u32, h: u32, fps: f32) -> CameraMetadata {
        CameraMetadata::new(model, "TestMfg", ct, w, h, fps)
    }

    #[test]
    fn test_phone_sensor_size() {
        assert_eq!(CameraType::Phone.sensor_size_mm(), Some(8.0));
    }

    #[test]
    fn test_cinema_sensor_size() {
        assert_eq!(CameraType::Cinema.sensor_size_mm(), Some(43.3));
    }

    #[test]
    fn test_other_sensor_size_none() {
        assert_eq!(CameraType::Other("custom".into()).sensor_size_mm(), None);
    }

    #[test]
    fn test_is_professional_broadcast() {
        assert!(CameraType::Broadcast.is_professional());
    }

    #[test]
    fn test_is_professional_phone() {
        assert!(!CameraType::Phone.is_professional());
    }

    #[test]
    fn test_is_4k_true() {
        let cam = make_cam("A", CameraType::Cinema, 3840, 2160, 60.0);
        assert!(cam.is_4k());
    }

    #[test]
    fn test_is_4k_false_hd() {
        let cam = make_cam("B", CameraType::Consumer, 1920, 1080, 30.0);
        assert!(!cam.is_4k());
    }

    #[test]
    fn test_is_8k() {
        let cam = make_cam("C", CameraType::Cinema, 8192, 4320, 24.0);
        assert!(cam.is_8k());
    }

    #[test]
    fn test_megapixels() {
        let cam = make_cam("D", CameraType::Broadcast, 1920, 1080, 50.0);
        let mp = cam.megapixels();
        assert!((mp - 2.0736).abs() < 0.001);
    }

    #[test]
    fn test_registry_register_and_count() {
        let mut reg = CameraRegistry::new();
        reg.register(make_cam("CamA", CameraType::Action, 4096, 2160, 120.0));
        reg.register(make_cam("CamB", CameraType::Broadcast, 1920, 1080, 60.0));
        assert_eq!(reg.count(), 2);
    }

    #[test]
    fn test_registry_find_by_model() {
        let mut reg = CameraRegistry::new();
        reg.register(make_cam("ModelX", CameraType::Ptz, 1920, 1080, 30.0));
        assert!(reg.find_by_model("ModelX").is_some());
        assert!(reg.find_by_model("Unknown").is_none());
    }

    #[test]
    fn test_registry_find_by_type() {
        let mut reg = CameraRegistry::new();
        reg.register(make_cam("CineCam1", CameraType::Cinema, 4096, 3072, 30.0));
        reg.register(make_cam("CineCam2", CameraType::Cinema, 6144, 3160, 24.0));
        reg.register(make_cam("PhoneCam", CameraType::Phone, 4032, 3024, 60.0));
        let cinema = reg.find_by_type(&CameraType::Cinema);
        assert_eq!(cinema.len(), 2);
    }

    #[test]
    fn test_registry_replace_existing() {
        let mut reg = CameraRegistry::new();
        reg.register(make_cam("Dup", CameraType::Consumer, 1920, 1080, 30.0));
        let old = reg.register(make_cam("Dup", CameraType::Consumer, 3840, 2160, 60.0));
        assert!(old.is_some());
        assert_eq!(reg.count(), 1);
        assert!(reg
            .find_by_model("Dup")
            .expect("multicam test operation should succeed")
            .is_4k());
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = CameraRegistry::new();
        reg.register(make_cam("Del", CameraType::Action, 1920, 1080, 120.0));
        let removed = reg.remove("Del");
        assert!(removed.is_some());
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_is_empty() {
        let reg = CameraRegistry::new();
        assert!(reg.is_empty());
    }
}
