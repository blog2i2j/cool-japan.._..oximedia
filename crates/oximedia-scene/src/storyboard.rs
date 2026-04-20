//! Scene storyboard generation.
//!
//! Provides types for organising video frames into storyboard panels and
//! exporting summaries suitable for editorial review.

/// A single frame entry in a storyboard panel.
#[derive(Debug, Clone)]
pub struct StoryboardFrame {
    /// Frame index in the source media.
    pub frame_idx: u64,
    /// Whether this frame is a key frame (scene cut, I-frame, etc.).
    pub is_key_frame: bool,
    /// Optional path to a thumbnail image for this frame.
    pub thumbnail_path: String,
    /// Free-form editorial notes.
    pub notes: String,
}

impl StoryboardFrame {
    /// Create a new `StoryboardFrame`.
    #[must_use]
    pub fn new(
        frame_idx: u64,
        is_key_frame: bool,
        thumbnail_path: impl Into<String>,
        notes: impl Into<String>,
    ) -> Self {
        Self {
            frame_idx,
            is_key_frame,
            thumbnail_path: thumbnail_path.into(),
            notes: notes.into(),
        }
    }

    /// Returns `true` when both `thumbnail_path` and `notes` are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.thumbnail_path.is_empty() && self.notes.is_empty()
    }
}

/// A storyboard panel representing a single shot or scene.
#[derive(Debug, Clone)]
pub struct StoryboardPanel {
    /// Shot identifier this panel corresponds to.
    pub shot_id: u64,
    /// Frames included in this panel.
    pub frames: Vec<StoryboardFrame>,
    /// Editorial direction note for this panel.
    pub direction: String,
}

impl StoryboardPanel {
    /// Create a new empty `StoryboardPanel`.
    #[must_use]
    pub fn new(shot_id: u64, direction: impl Into<String>) -> Self {
        Self {
            shot_id,
            frames: Vec::new(),
            direction: direction.into(),
        }
    }

    /// Number of frames in this panel.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// References to frames marked as key frames.
    #[must_use]
    pub fn key_frames(&self) -> Vec<&StoryboardFrame> {
        self.frames.iter().filter(|f| f.is_key_frame).collect()
    }

    /// Add a frame to the panel.
    pub fn add_frame(&mut self, frame: StoryboardFrame) {
        self.frames.push(frame);
    }
}

/// A complete storyboard composed of multiple panels.
#[derive(Debug, Clone)]
pub struct Storyboard {
    /// Title of the storyboard.
    pub title: String,
    /// Ordered list of panels.
    pub panels: Vec<StoryboardPanel>,
    /// Creation timestamp in Unix epoch milliseconds.
    pub created_ms: u64,
}

impl Storyboard {
    /// Create a new empty `Storyboard`.
    #[must_use]
    pub fn new(title: impl Into<String>, created_ms: u64) -> Self {
        Self {
            title: title.into(),
            panels: Vec::new(),
            created_ms,
        }
    }

    /// Add a panel to the storyboard.
    pub fn add_panel(&mut self, panel: StoryboardPanel) {
        self.panels.push(panel);
    }

    /// Number of panels.
    #[must_use]
    pub fn panel_count(&self) -> usize {
        self.panels.len()
    }

    /// Total number of frames across all panels.
    #[must_use]
    pub fn total_frames(&self) -> usize {
        self.panels.iter().map(StoryboardPanel::frame_count).sum()
    }

    /// Generate a plain-text summary of this storyboard.
    #[must_use]
    pub fn export_summary(&self) -> String {
        format!(
            "Storyboard: {}\nPanels: {}\nTotal frames: {}\nCreated: {}ms",
            self.title,
            self.panel_count(),
            self.total_frames(),
            self.created_ms,
        )
    }
}

/// Utility for selecting evenly-spaced key frame indices.
pub struct KeyFrameSelector;

impl KeyFrameSelector {
    /// Select `n_keyframes` evenly-spaced frame indices from a clip of
    /// `total_frames` frames (indices are in `[0, total_frames)`).
    ///
    /// Returns an empty vector when `n_keyframes == 0` or `total_frames == 0`.
    #[must_use]
    pub fn select(total_frames: u64, n_keyframes: usize) -> Vec<u64> {
        if n_keyframes == 0 || total_frames == 0 {
            return Vec::new();
        }
        let n = n_keyframes as u64;
        if n >= total_frames {
            return (0..total_frames).collect();
        }
        (0..n)
            .map(|i| i * (total_frames - 1) / (n - 1).max(1))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. StoryboardFrame::is_empty – truly empty
    #[test]
    fn test_frame_is_empty() {
        let f = StoryboardFrame::new(0, false, "", "");
        assert!(f.is_empty());
    }

    // 2. StoryboardFrame::is_empty – has thumbnail path
    #[test]
    fn test_frame_not_empty_thumbnail() {
        let thumb = std::env::temp_dir()
            .join("oximedia-scene-storyboard-thumb.jpg")
            .to_string_lossy()
            .into_owned();
        let f = StoryboardFrame::new(0, false, thumb, "");
        assert!(!f.is_empty());
    }

    // 3. StoryboardFrame::is_empty – has notes
    #[test]
    fn test_frame_not_empty_notes() {
        let f = StoryboardFrame::new(0, false, "", "Close-up of hero");
        assert!(!f.is_empty());
    }

    // 4. StoryboardPanel::frame_count
    #[test]
    fn test_panel_frame_count() {
        let mut panel = StoryboardPanel::new(0, "Wide shot");
        panel.add_frame(StoryboardFrame::new(10, true, "", ""));
        panel.add_frame(StoryboardFrame::new(20, false, "", ""));
        assert_eq!(panel.frame_count(), 2);
    }

    // 5. StoryboardPanel::key_frames – filter correctly
    #[test]
    fn test_panel_key_frames() {
        let mut panel = StoryboardPanel::new(1, "Action");
        panel.add_frame(StoryboardFrame::new(0, true, "", ""));
        panel.add_frame(StoryboardFrame::new(5, false, "", ""));
        panel.add_frame(StoryboardFrame::new(10, true, "", ""));
        let kf = panel.key_frames();
        assert_eq!(kf.len(), 2);
        assert_eq!(kf[0].frame_idx, 0);
        assert_eq!(kf[1].frame_idx, 10);
    }

    // 6. StoryboardPanel::key_frames – none
    #[test]
    fn test_panel_no_key_frames() {
        let mut panel = StoryboardPanel::new(2, "");
        panel.add_frame(StoryboardFrame::new(1, false, "", ""));
        assert!(panel.key_frames().is_empty());
    }

    // 7. Storyboard::add_panel and panel_count
    #[test]
    fn test_storyboard_panel_count() {
        let mut sb = Storyboard::new("Test", 0);
        sb.add_panel(StoryboardPanel::new(0, ""));
        sb.add_panel(StoryboardPanel::new(1, ""));
        assert_eq!(sb.panel_count(), 2);
    }

    // 8. Storyboard::total_frames – sums panels
    #[test]
    fn test_storyboard_total_frames() {
        let mut sb = Storyboard::new("T", 0);
        let mut p1 = StoryboardPanel::new(0, "");
        p1.add_frame(StoryboardFrame::new(0, true, "", ""));
        p1.add_frame(StoryboardFrame::new(1, false, "", ""));
        let mut p2 = StoryboardPanel::new(1, "");
        p2.add_frame(StoryboardFrame::new(10, true, "", ""));
        sb.add_panel(p1);
        sb.add_panel(p2);
        assert_eq!(sb.total_frames(), 3);
    }

    // 9. Storyboard::total_frames – empty
    #[test]
    fn test_storyboard_total_frames_empty() {
        let sb = Storyboard::new("Empty", 0);
        assert_eq!(sb.total_frames(), 0);
    }

    // 10. Storyboard::export_summary contains title
    #[test]
    fn test_storyboard_export_summary_title() {
        let sb = Storyboard::new("My Film", 12345);
        let s = sb.export_summary();
        assert!(s.contains("My Film"));
    }

    // 11. Storyboard::export_summary contains panel/frame counts
    #[test]
    fn test_storyboard_export_summary_counts() {
        let mut sb = Storyboard::new("X", 0);
        let mut p = StoryboardPanel::new(0, "");
        p.add_frame(StoryboardFrame::new(0, true, "", ""));
        sb.add_panel(p);
        let s = sb.export_summary();
        assert!(s.contains('1')); // panels and frames both = 1
    }

    // 12. KeyFrameSelector::select – evenly spaced
    #[test]
    fn test_keyframe_selector_basic() {
        let indices = KeyFrameSelector::select(100, 5);
        assert_eq!(indices.len(), 5);
        // First and last should be 0 and 99
        assert_eq!(*indices.first().expect("should succeed in test"), 0);
        assert_eq!(*indices.last().expect("should succeed in test"), 99);
    }

    // 13. KeyFrameSelector::select – zero keyframes
    #[test]
    fn test_keyframe_selector_zero_n() {
        let indices = KeyFrameSelector::select(100, 0);
        assert!(indices.is_empty());
    }

    // 14. KeyFrameSelector::select – zero total frames
    #[test]
    fn test_keyframe_selector_zero_frames() {
        let indices = KeyFrameSelector::select(0, 5);
        assert!(indices.is_empty());
    }

    // 15. KeyFrameSelector::select – n >= total_frames returns all
    #[test]
    fn test_keyframe_selector_n_exceeds_frames() {
        let indices = KeyFrameSelector::select(3, 10);
        assert_eq!(indices.len(), 3);
        assert_eq!(indices, vec![0, 1, 2]);
    }

    // 16. KeyFrameSelector::select – single keyframe returns first frame
    #[test]
    fn test_keyframe_selector_single() {
        let indices = KeyFrameSelector::select(50, 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
    }
}
