#![allow(dead_code)]
//! Camera labeling and identification system for multi-camera productions.
//!
//! Provides a structured way to label cameras with human-readable names,
//! short codes, tally colors, and positional descriptions. Labels can be
//! formatted for different contexts (switcher display, metadata embed, etc.).

use std::collections::HashMap;
use std::fmt;

/// Color used for tally light / UI indicators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TallyColor {
    /// Red tally – camera is on-air / program.
    Red,
    /// Green tally – camera is on preview.
    Green,
    /// Amber / yellow tally – camera is cued.
    Amber,
    /// Blue tally – ISO recording.
    Blue,
    /// No tally active.
    Off,
}

impl fmt::Display for TallyColor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Red => write!(f, "RED"),
            Self::Green => write!(f, "GREEN"),
            Self::Amber => write!(f, "AMBER"),
            Self::Blue => write!(f, "BLUE"),
            Self::Off => write!(f, "OFF"),
        }
    }
}

/// Role a camera plays in the production.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CameraRole {
    /// Wide / establishing shot.
    Wide,
    /// Medium shot.
    Medium,
    /// Close-up / tight shot.
    CloseUp,
    /// Jib / crane camera.
    Jib,
    /// Steadicam / handheld roving camera.
    Roving,
    /// Robotic / PTZ camera.
    Robotic,
    /// Specialty camera (e.g., underwater, lipstick).
    Specialty,
}

impl fmt::Display for CameraRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wide => write!(f, "WIDE"),
            Self::Medium => write!(f, "MED"),
            Self::CloseUp => write!(f, "CU"),
            Self::Jib => write!(f, "JIB"),
            Self::Roving => write!(f, "ROV"),
            Self::Robotic => write!(f, "PTZ"),
            Self::Specialty => write!(f, "SPL"),
        }
    }
}

/// A label assigned to a single camera in a multi-camera production.
#[derive(Debug, Clone)]
pub struct CamLabel {
    /// Numeric index (0-based).
    pub index: usize,
    /// Short code shown on switcher buttons (e.g. "CAM1").
    pub short_code: String,
    /// Full descriptive name (e.g. "Camera 1 – Stage Left Wide").
    pub full_name: String,
    /// Role of this camera.
    pub role: CameraRole,
    /// Current tally state.
    pub tally: TallyColor,
    /// Operator name, if assigned.
    pub operator: Option<String>,
    /// Free-form notes.
    pub notes: String,
}

impl CamLabel {
    /// Create a new camera label with minimal required fields.
    #[must_use]
    pub fn new(index: usize, short_code: &str, full_name: &str, role: CameraRole) -> Self {
        Self {
            index,
            short_code: short_code.to_owned(),
            full_name: full_name.to_owned(),
            role,
            tally: TallyColor::Off,
            operator: None,
            notes: String::new(),
        }
    }

    /// Return a compact switcher-friendly label, e.g. "CAM1 \[WIDE\]".
    #[must_use]
    pub fn switcher_label(&self) -> String {
        format!("{} [{}]", self.short_code, self.role)
    }

    /// Return a metadata-embed label including operator if known.
    #[must_use]
    pub fn metadata_label(&self) -> String {
        match &self.operator {
            Some(op) => format!("{} ({}) – {}", self.full_name, op, self.role),
            None => format!("{} – {}", self.full_name, self.role),
        }
    }

    /// Set the tally color.
    pub fn set_tally(&mut self, color: TallyColor) {
        self.tally = color;
    }

    /// Check whether the camera is currently on-air (red tally).
    #[must_use]
    pub fn is_on_air(&self) -> bool {
        self.tally == TallyColor::Red
    }

    /// Check whether the camera is currently on preview (green tally).
    #[must_use]
    pub fn is_on_preview(&self) -> bool {
        self.tally == TallyColor::Green
    }
}

impl fmt::Display for CamLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.switcher_label())
    }
}

/// Registry that manages labels for all cameras in a production.
#[derive(Debug, Clone)]
pub struct LabelRegistry {
    /// Labels keyed by camera index.
    labels: HashMap<usize, CamLabel>,
    /// Production name.
    production: String,
}

impl LabelRegistry {
    /// Create a new empty registry for a given production.
    #[must_use]
    pub fn new(production: &str) -> Self {
        Self {
            labels: HashMap::new(),
            production: production.to_owned(),
        }
    }

    /// Register a camera label. Returns the previous label if the index was already used.
    pub fn register(&mut self, label: CamLabel) -> Option<CamLabel> {
        self.labels.insert(label.index, label)
    }

    /// Remove a label by index.
    pub fn remove(&mut self, index: usize) -> Option<CamLabel> {
        self.labels.remove(&index)
    }

    /// Look up a label by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&CamLabel> {
        self.labels.get(&index)
    }

    /// Mutably look up a label by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut CamLabel> {
        self.labels.get_mut(&index)
    }

    /// Number of registered cameras.
    #[must_use]
    pub fn count(&self) -> usize {
        self.labels.len()
    }

    /// Production name.
    #[must_use]
    pub fn production(&self) -> &str {
        &self.production
    }

    /// Iterate over all labels in index order.
    #[must_use]
    pub fn labels_sorted(&self) -> Vec<&CamLabel> {
        let mut v: Vec<&CamLabel> = self.labels.values().collect();
        v.sort_by_key(|l| l.index);
        v
    }

    /// Find the first camera with a given role.
    #[must_use]
    pub fn find_by_role(&self, role: CameraRole) -> Option<&CamLabel> {
        self.labels.values().find(|l| l.role == role)
    }

    /// Set all tallies to Off.
    pub fn clear_tallies(&mut self) {
        for label in self.labels.values_mut() {
            label.tally = TallyColor::Off;
        }
    }

    /// Set on-air camera (red tally). All others go to Off unless they are Green.
    pub fn set_on_air(&mut self, index: usize) {
        for (idx, label) in &mut self.labels {
            if *idx == index {
                label.tally = TallyColor::Red;
            } else if label.tally == TallyColor::Red {
                label.tally = TallyColor::Off;
            }
        }
    }

    /// Return indices of cameras matching a role.
    #[must_use]
    pub fn indices_for_role(&self, role: CameraRole) -> Vec<usize> {
        let mut out: Vec<usize> = self
            .labels
            .values()
            .filter(|l| l.role == role)
            .map(|l| l.index)
            .collect();
        out.sort_unstable();
        out
    }

    /// Generate a summary string for all cameras.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut lines: Vec<String> = Vec::new();
        lines.push(format!("Production: {}", self.production));
        for label in self.labels_sorted() {
            lines.push(format!(
                "  {} – {} (tally: {})",
                label.short_code, label.full_name, label.tally
            ));
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cam_label_new() {
        let label = CamLabel::new(0, "CAM1", "Camera 1", CameraRole::Wide);
        assert_eq!(label.index, 0);
        assert_eq!(label.short_code, "CAM1");
        assert_eq!(label.role, CameraRole::Wide);
        assert_eq!(label.tally, TallyColor::Off);
    }

    #[test]
    fn test_switcher_label() {
        let label = CamLabel::new(0, "CAM1", "Camera 1", CameraRole::Wide);
        assert_eq!(label.switcher_label(), "CAM1 [WIDE]");
    }

    #[test]
    fn test_metadata_label_no_operator() {
        let label = CamLabel::new(1, "CAM2", "Camera 2", CameraRole::CloseUp);
        assert_eq!(label.metadata_label(), "Camera 2 – CU");
    }

    #[test]
    fn test_metadata_label_with_operator() {
        let mut label = CamLabel::new(2, "CAM3", "Camera 3", CameraRole::Jib);
        label.operator = Some("John".to_owned());
        assert_eq!(label.metadata_label(), "Camera 3 (John) – JIB");
    }

    #[test]
    fn test_tally_state() {
        let mut label = CamLabel::new(0, "CAM1", "Camera 1", CameraRole::Wide);
        assert!(!label.is_on_air());
        assert!(!label.is_on_preview());

        label.set_tally(TallyColor::Red);
        assert!(label.is_on_air());

        label.set_tally(TallyColor::Green);
        assert!(label.is_on_preview());
        assert!(!label.is_on_air());
    }

    #[test]
    fn test_display() {
        let label = CamLabel::new(0, "CAM1", "Camera 1", CameraRole::Roving);
        assert_eq!(format!("{label}"), "CAM1 [ROV]");
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = LabelRegistry::new("Test Show");
        reg.register(CamLabel::new(0, "A", "Cam A", CameraRole::Wide));
        assert_eq!(reg.count(), 1);
        assert!(reg.get(0).is_some());
        assert!(reg.get(99).is_none());
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = LabelRegistry::new("Show");
        reg.register(CamLabel::new(0, "A", "Cam A", CameraRole::Wide));
        let removed = reg.remove(0);
        assert!(removed.is_some());
        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn test_registry_find_by_role() {
        let mut reg = LabelRegistry::new("Show");
        reg.register(CamLabel::new(0, "A", "Cam A", CameraRole::Wide));
        reg.register(CamLabel::new(1, "B", "Cam B", CameraRole::Jib));
        let found = reg.find_by_role(CameraRole::Jib);
        assert!(found.is_some());
        assert_eq!(
            found.expect("multicam test operation should succeed").index,
            1
        );
    }

    #[test]
    fn test_registry_clear_tallies() {
        let mut reg = LabelRegistry::new("Show");
        let mut label = CamLabel::new(0, "A", "Cam A", CameraRole::Wide);
        label.set_tally(TallyColor::Red);
        reg.register(label);
        reg.clear_tallies();
        assert_eq!(
            reg.get(0)
                .expect("multicam test operation should succeed")
                .tally,
            TallyColor::Off
        );
    }

    #[test]
    fn test_registry_set_on_air() {
        let mut reg = LabelRegistry::new("Show");
        reg.register(CamLabel::new(0, "A", "Cam A", CameraRole::Wide));
        reg.register(CamLabel::new(1, "B", "Cam B", CameraRole::CloseUp));
        reg.set_on_air(0);
        assert_eq!(
            reg.get(0)
                .expect("multicam test operation should succeed")
                .tally,
            TallyColor::Red
        );
        assert_eq!(
            reg.get(1)
                .expect("multicam test operation should succeed")
                .tally,
            TallyColor::Off
        );

        // Switch on-air to camera 1
        reg.set_on_air(1);
        assert_eq!(
            reg.get(0)
                .expect("multicam test operation should succeed")
                .tally,
            TallyColor::Off
        );
        assert_eq!(
            reg.get(1)
                .expect("multicam test operation should succeed")
                .tally,
            TallyColor::Red
        );
    }

    #[test]
    fn test_registry_indices_for_role() {
        let mut reg = LabelRegistry::new("Show");
        reg.register(CamLabel::new(0, "A", "Cam A", CameraRole::Wide));
        reg.register(CamLabel::new(2, "C", "Cam C", CameraRole::Wide));
        reg.register(CamLabel::new(1, "B", "Cam B", CameraRole::CloseUp));
        let wides = reg.indices_for_role(CameraRole::Wide);
        assert_eq!(wides, vec![0, 2]);
    }

    #[test]
    fn test_registry_summary() {
        let mut reg = LabelRegistry::new("Big Show");
        reg.register(CamLabel::new(0, "CAM1", "Camera 1", CameraRole::Wide));
        let s = reg.summary();
        assert!(s.contains("Big Show"));
        assert!(s.contains("CAM1"));
    }

    #[test]
    fn test_tally_color_display() {
        assert_eq!(format!("{}", TallyColor::Red), "RED");
        assert_eq!(format!("{}", TallyColor::Off), "OFF");
    }

    #[test]
    fn test_camera_role_display() {
        assert_eq!(format!("{}", CameraRole::Robotic), "PTZ");
        assert_eq!(format!("{}", CameraRole::Specialty), "SPL");
    }
}
