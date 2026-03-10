//! Sync markers and cue points for multi-camera production.

use crate::{AngleId, FrameNumber};

/// Marker type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerType {
    /// Sync point marker
    SyncPoint,
    /// Cue point for switching
    CuePoint,
    /// Scene boundary
    SceneBoundary,
    /// Quality issue marker
    QualityIssue,
    /// Note/comment
    Note,
    /// Custom marker
    Custom,
}

/// Sync marker
#[derive(Debug, Clone)]
pub struct SyncMarker {
    /// Marker identifier
    pub id: u64,
    /// Frame number
    pub frame: FrameNumber,
    /// Angles affected
    pub angles: Vec<AngleId>,
    /// Marker type
    pub marker_type: MarkerType,
    /// Label/description
    pub label: String,
    /// Color (RGBA)
    pub color: [u8; 4],
}

impl SyncMarker {
    /// Create a new sync marker
    #[must_use]
    pub fn new(id: u64, frame: FrameNumber, marker_type: MarkerType) -> Self {
        Self {
            id,
            frame,
            angles: Vec::new(),
            marker_type,
            label: String::new(),
            color: [255, 0, 0, 255], // Red by default
        }
    }

    /// Add angle to marker
    pub fn add_angle(&mut self, angle: AngleId) {
        if !self.angles.contains(&angle) {
            self.angles.push(angle);
        }
    }

    /// Remove angle from marker
    pub fn remove_angle(&mut self, angle: AngleId) {
        self.angles.retain(|&a| a != angle);
    }

    /// Check if marker affects angle
    #[must_use]
    pub fn affects_angle(&self, angle: AngleId) -> bool {
        self.angles.contains(&angle)
    }

    /// Set label
    pub fn set_label(&mut self, label: String) {
        self.label = label;
    }

    /// Set color
    pub fn set_color(&mut self, color: [u8; 4]) {
        self.color = color;
    }
}

/// Marker manager
#[derive(Debug)]
pub struct MarkerManager {
    /// All markers
    markers: Vec<SyncMarker>,
    /// Next marker ID
    next_id: u64,
}

impl MarkerManager {
    /// Create a new marker manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            markers: Vec::new(),
            next_id: 1,
        }
    }

    /// Add a marker
    pub fn add_marker(&mut self, mut marker: SyncMarker) -> u64 {
        marker.id = self.next_id;
        self.next_id += 1;
        let id = marker.id;
        self.markers.push(marker);
        id
    }

    /// Create and add a sync point marker
    pub fn add_sync_point(
        &mut self,
        frame: FrameNumber,
        angles: Vec<AngleId>,
        label: String,
    ) -> u64 {
        let mut marker = SyncMarker::new(0, frame, MarkerType::SyncPoint);
        marker.angles = angles;
        marker.label = label;
        marker.color = [0, 255, 0, 255]; // Green
        self.add_marker(marker)
    }

    /// Create and add a cue point marker
    pub fn add_cue_point(&mut self, frame: FrameNumber, angle: AngleId, label: String) -> u64 {
        let mut marker = SyncMarker::new(0, frame, MarkerType::CuePoint);
        marker.angles = vec![angle];
        marker.label = label;
        marker.color = [0, 0, 255, 255]; // Blue
        self.add_marker(marker)
    }

    /// Get marker by ID
    #[must_use]
    pub fn get_marker(&self, id: u64) -> Option<&SyncMarker> {
        self.markers.iter().find(|m| m.id == id)
    }

    /// Get mutable marker by ID
    pub fn get_marker_mut(&mut self, id: u64) -> Option<&mut SyncMarker> {
        self.markers.iter_mut().find(|m| m.id == id)
    }

    /// Remove marker by ID
    pub fn remove_marker(&mut self, id: u64) -> bool {
        if let Some(pos) = self.markers.iter().position(|m| m.id == id) {
            self.markers.remove(pos);
            true
        } else {
            false
        }
    }

    /// Get all markers
    #[must_use]
    pub fn markers(&self) -> &[SyncMarker] {
        &self.markers
    }

    /// Get markers at frame
    #[must_use]
    pub fn markers_at_frame(&self, frame: FrameNumber) -> Vec<&SyncMarker> {
        self.markers.iter().filter(|m| m.frame == frame).collect()
    }

    /// Get markers in frame range
    #[must_use]
    pub fn markers_in_range(&self, start: FrameNumber, end: FrameNumber) -> Vec<&SyncMarker> {
        self.markers
            .iter()
            .filter(|m| m.frame >= start && m.frame < end)
            .collect()
    }

    /// Get markers by type
    #[must_use]
    pub fn markers_by_type(&self, marker_type: MarkerType) -> Vec<&SyncMarker> {
        self.markers
            .iter()
            .filter(|m| m.marker_type == marker_type)
            .collect()
    }

    /// Get markers affecting angle
    #[must_use]
    pub fn markers_for_angle(&self, angle: AngleId) -> Vec<&SyncMarker> {
        self.markers
            .iter()
            .filter(|m| m.affects_angle(angle))
            .collect()
    }

    /// Clear all markers
    pub fn clear(&mut self) {
        self.markers.clear();
    }

    /// Get marker count
    #[must_use]
    pub fn count(&self) -> usize {
        self.markers.len()
    }

    /// Get next marker after frame
    #[must_use]
    pub fn next_marker_after(&self, frame: FrameNumber) -> Option<&SyncMarker> {
        self.markers
            .iter()
            .filter(|m| m.frame > frame)
            .min_by_key(|m| m.frame)
    }

    /// Get previous marker before frame
    #[must_use]
    pub fn previous_marker_before(&self, frame: FrameNumber) -> Option<&SyncMarker> {
        self.markers
            .iter()
            .filter(|m| m.frame < frame)
            .max_by_key(|m| m.frame)
    }

    /// Export markers to CSV
    #[must_use]
    pub fn export_csv(&self) -> String {
        let mut csv = String::from("ID,Frame,Type,Label,Angles\n");

        for marker in &self.markers {
            let type_str = match marker.marker_type {
                MarkerType::SyncPoint => "SyncPoint",
                MarkerType::CuePoint => "CuePoint",
                MarkerType::SceneBoundary => "SceneBoundary",
                MarkerType::QualityIssue => "QualityIssue",
                MarkerType::Note => "Note",
                MarkerType::Custom => "Custom",
            };

            let angles_str = marker
                .angles
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(";");

            csv.push_str(&format!(
                "{},{},{},\"{}\",{}\n",
                marker.id, marker.frame, type_str, marker.label, angles_str
            ));
        }

        csv
    }

    /// Sort markers by frame
    pub fn sort_by_frame(&mut self) {
        self.markers.sort_by_key(|m| m.frame);
    }
}

impl Default for MarkerManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Marker color presets
pub struct MarkerColors;

impl MarkerColors {
    /// Red marker
    pub const RED: [u8; 4] = [255, 0, 0, 255];
    /// Green marker
    pub const GREEN: [u8; 4] = [0, 255, 0, 255];
    /// Blue marker
    pub const BLUE: [u8; 4] = [0, 0, 255, 255];
    /// Yellow marker
    pub const YELLOW: [u8; 4] = [255, 255, 0, 255];
    /// Orange marker
    pub const ORANGE: [u8; 4] = [255, 165, 0, 255];
    /// Purple marker
    pub const PURPLE: [u8; 4] = [128, 0, 128, 255];
    /// White marker
    pub const WHITE: [u8; 4] = [255, 255, 255, 255];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_marker_creation() {
        let marker = SyncMarker::new(1, 100, MarkerType::SyncPoint);
        assert_eq!(marker.id, 1);
        assert_eq!(marker.frame, 100);
        assert_eq!(marker.marker_type, MarkerType::SyncPoint);
    }

    #[test]
    fn test_add_remove_angle() {
        let mut marker = SyncMarker::new(1, 100, MarkerType::SyncPoint);
        marker.add_angle(0);
        marker.add_angle(1);
        assert_eq!(marker.angles.len(), 2);

        marker.remove_angle(0);
        assert_eq!(marker.angles.len(), 1);
        assert_eq!(marker.angles[0], 1);
    }

    #[test]
    fn test_affects_angle() {
        let mut marker = SyncMarker::new(1, 100, MarkerType::SyncPoint);
        marker.add_angle(0);

        assert!(marker.affects_angle(0));
        assert!(!marker.affects_angle(1));
    }

    #[test]
    fn test_marker_manager_creation() {
        let manager = MarkerManager::new();
        assert_eq!(manager.count(), 0);
        assert_eq!(manager.next_id, 1);
    }

    #[test]
    fn test_add_marker() {
        let mut manager = MarkerManager::new();
        let marker = SyncMarker::new(0, 100, MarkerType::SyncPoint);
        let id = manager.add_marker(marker);

        assert_eq!(id, 1);
        assert_eq!(manager.count(), 1);
    }

    #[test]
    fn test_add_sync_point() {
        let mut manager = MarkerManager::new();
        let id = manager.add_sync_point(100, vec![0, 1], "Sync 1".to_string());

        assert_eq!(id, 1);
        let marker = manager
            .get_marker(id)
            .expect("multicam test operation should succeed");
        assert_eq!(marker.marker_type, MarkerType::SyncPoint);
        assert_eq!(marker.label, "Sync 1");
    }

    #[test]
    fn test_remove_marker() {
        let mut manager = MarkerManager::new();
        let id = manager.add_sync_point(100, vec![0], "Test".to_string());

        assert!(manager.remove_marker(id));
        assert_eq!(manager.count(), 0);
    }

    #[test]
    fn test_markers_at_frame() {
        let mut manager = MarkerManager::new();
        manager.add_sync_point(100, vec![0], "First".to_string());
        manager.add_sync_point(100, vec![1], "Second".to_string());
        manager.add_sync_point(200, vec![0], "Third".to_string());

        let markers = manager.markers_at_frame(100);
        assert_eq!(markers.len(), 2);
    }

    #[test]
    fn test_markers_in_range() {
        let mut manager = MarkerManager::new();
        manager.add_sync_point(100, vec![0], "First".to_string());
        manager.add_sync_point(150, vec![0], "Second".to_string());
        manager.add_sync_point(200, vec![0], "Third".to_string());

        let markers = manager.markers_in_range(100, 200);
        assert_eq!(markers.len(), 2);
    }

    #[test]
    fn test_markers_by_type() {
        let mut manager = MarkerManager::new();
        manager.add_sync_point(100, vec![0], "Sync".to_string());
        manager.add_cue_point(200, 0, "Cue".to_string());

        let sync_markers = manager.markers_by_type(MarkerType::SyncPoint);
        assert_eq!(sync_markers.len(), 1);

        let cue_markers = manager.markers_by_type(MarkerType::CuePoint);
        assert_eq!(cue_markers.len(), 1);
    }

    #[test]
    fn test_markers_for_angle() {
        let mut manager = MarkerManager::new();
        manager.add_sync_point(100, vec![0, 1], "Both".to_string());
        manager.add_sync_point(200, vec![0], "Zero only".to_string());

        let markers = manager.markers_for_angle(0);
        assert_eq!(markers.len(), 2);

        let markers = manager.markers_for_angle(1);
        assert_eq!(markers.len(), 1);
    }

    #[test]
    fn test_next_previous_marker() {
        let mut manager = MarkerManager::new();
        manager.add_sync_point(100, vec![0], "First".to_string());
        manager.add_sync_point(200, vec![0], "Second".to_string());
        manager.add_sync_point(300, vec![0], "Third".to_string());

        let next = manager.next_marker_after(150);
        assert_eq!(
            next.expect("multicam test operation should succeed").frame,
            200
        );

        let prev = manager.previous_marker_before(250);
        assert_eq!(
            prev.expect("multicam test operation should succeed").frame,
            200
        );
    }

    #[test]
    fn test_export_csv() {
        let mut manager = MarkerManager::new();
        manager.add_sync_point(100, vec![0, 1], "Test".to_string());

        let csv = manager.export_csv();
        assert!(csv.contains("ID,Frame,Type,Label,Angles"));
        assert!(csv.contains("SyncPoint"));
    }

    #[test]
    fn test_sort_by_frame() {
        let mut manager = MarkerManager::new();
        manager.add_sync_point(300, vec![0], "Third".to_string());
        manager.add_sync_point(100, vec![0], "First".to_string());
        manager.add_sync_point(200, vec![0], "Second".to_string());

        manager.sort_by_frame();
        assert_eq!(manager.markers()[0].frame, 100);
        assert_eq!(manager.markers()[1].frame, 200);
        assert_eq!(manager.markers()[2].frame, 300);
    }
}
