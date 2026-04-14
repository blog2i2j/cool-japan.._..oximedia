//! Scene and source switching logic for live streaming.
//!
//! Provides a stateful [`SceneSwitcher`] that manages named scenes composed of
//! individually toggleable [`SceneSource`] elements, supports several
//! [`TransitionType`] variants, and can queue future switches via
//! [`SceneSwitcher::schedule_switch`].

use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can arise from scene-switcher operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SwitcherError {
    /// A scene with the requested name was not found.
    #[error("scene not found: '{0}'")]
    SceneNotFound(String),
    /// A scene with the same name already exists.
    #[error("scene already exists: '{0}'")]
    SceneAlreadyExists(String),
    /// A source with the requested id was not found in the scene.
    #[error("source not found: id={0}")]
    SourceNotFound(u32),
    /// Cannot remove the currently active scene.
    #[error("cannot remove the active scene: '{0}'")]
    CannotRemoveActive(String),
    /// Attempted to switch while no scenes are registered.
    #[error("no scenes registered")]
    NoScenes,
}

// ---------------------------------------------------------------------------
// TransitionType
// ---------------------------------------------------------------------------

/// Direction used by the Slide transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlideDir {
    /// Scene slides in from the left.
    Left,
    /// Scene slides in from the right.
    Right,
    /// Scene slides in from the top.
    Up,
    /// Scene slides in from the bottom.
    Down,
}

impl SlideDir {
    /// Returns the opposite slide direction.
    #[must_use]
    pub fn opposite(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
            Self::Up => Self::Down,
            Self::Down => Self::Up,
        }
    }
}

/// How a scene transition is rendered.
#[derive(Debug, Clone, PartialEq)]
pub enum TransitionType {
    /// Instant cut — no transition frames.
    Cut,
    /// Cross-fade over `duration_ms` milliseconds.
    Fade {
        /// Duration of the fade in milliseconds.
        duration_ms: u32,
    },
    /// Slide in `direction` over `duration_ms` milliseconds.
    Slide {
        /// Slide direction.
        direction: SlideDir,
        /// Duration of the slide in milliseconds.
        duration_ms: u32,
    },
    /// A pre-rendered stinger video plays over the cut.
    Stinger,
}

impl TransitionType {
    /// Estimated duration in milliseconds (0 for [`Cut`](TransitionType::Cut)).
    #[must_use]
    pub fn duration_ms(&self) -> u32 {
        match self {
            Self::Cut | Self::Stinger => 0,
            Self::Fade { duration_ms } | Self::Slide { duration_ms, .. } => *duration_ms,
        }
    }

    /// Returns `true` for transitions that require blending frames.
    #[must_use]
    pub fn requires_blend(&self) -> bool {
        matches!(self, Self::Fade { .. } | Self::Slide { .. })
    }
}

// ---------------------------------------------------------------------------
// SceneSource
// ---------------------------------------------------------------------------

/// A single source within a scene (camera, game capture, browser, etc.).
#[derive(Debug, Clone)]
pub struct SceneSource {
    /// Unique identifier within the scene.
    pub id: u32,
    /// Human-readable name.
    pub name: String,
    /// Whether this source is currently visible.
    pub visible: bool,
    /// Audio volume in the range 0.0 – 1.0 (values are clamped on construction).
    pub volume: f32,
}

impl SceneSource {
    /// Construct a new source.
    ///
    /// `volume` is clamped to [0.0, 1.0].
    #[must_use]
    pub fn new(id: u32, name: impl Into<String>, visible: bool, volume: f32) -> Self {
        Self {
            id,
            name: name.into(),
            visible,
            volume: volume.clamp(0.0, 1.0),
        }
    }

    /// Update the volume, clamping to [0.0, 1.0].
    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 1.0);
    }
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

/// A named collection of sources that can be switched to.
#[derive(Debug, Clone)]
pub struct Scene {
    /// Scene name (unique within the switcher).
    pub name: String,
    /// Ordered list of sources in this scene.
    pub sources: Vec<SceneSource>,
    /// Default transition used when switching *to* this scene.
    pub transition_type: TransitionType,
}

impl Scene {
    /// Create a scene with no sources and a [`Cut`](TransitionType::Cut) transition.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            sources: Vec::new(),
            transition_type: TransitionType::Cut,
        }
    }

    /// Add a source to the scene.
    pub fn add_source(&mut self, source: SceneSource) {
        self.sources.push(source);
    }

    /// Find a source by id, returning a mutable reference.
    #[must_use]
    pub fn source_mut(&mut self, id: u32) -> Option<&mut SceneSource> {
        self.sources.iter_mut().find(|s| s.id == id)
    }

    /// Number of visible sources.
    #[must_use]
    pub fn visible_source_count(&self) -> usize {
        self.sources.iter().filter(|s| s.visible).count()
    }
}

// ---------------------------------------------------------------------------
// SwitchEvent
// ---------------------------------------------------------------------------

/// Record of a scene switch that occurred or is scheduled.
#[derive(Debug, Clone)]
pub struct SwitchEvent {
    /// Name of the scene that was active before the switch.
    pub from_scene: Option<String>,
    /// Name of the scene switched to.
    pub to_scene: String,
    /// Transition used for this switch.
    pub transition: TransitionType,
    /// Frame index at which the switch occurred.
    pub frame: u64,
}

// ---------------------------------------------------------------------------
// ScheduledSwitch
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ScheduledSwitch {
    target_scene: String,
    at_frame: u64,
    transition: TransitionType,
}

// ---------------------------------------------------------------------------
// SceneSwitcher
// ---------------------------------------------------------------------------

/// Live scene switcher with transition support and scheduled switching.
///
/// Scenes are keyed by name. At most one scene is active at a time; calling
/// [`switch_to`](SceneSwitcher::switch_to) records a [`SwitchEvent`] and
/// updates the active scene immediately (transition rendering is left to the
/// caller's compositor).
pub struct SceneSwitcher {
    scenes: HashMap<String, Scene>,
    active_scene: Option<String>,
    history: Vec<SwitchEvent>,
    scheduled: Vec<ScheduledSwitch>,
}

impl Default for SceneSwitcher {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneSwitcher {
    /// Create a new, empty switcher.
    #[must_use]
    pub fn new() -> Self {
        Self {
            scenes: HashMap::new(),
            active_scene: None,
            history: Vec::new(),
            scheduled: Vec::new(),
        }
    }

    // -- Scene management ---------------------------------------------------

    /// Register a new scene.
    ///
    /// # Errors
    ///
    /// Returns [`SwitcherError::SceneAlreadyExists`] if a scene with that name
    /// already exists.
    pub fn add_scene(&mut self, scene: Scene) -> Result<(), SwitcherError> {
        if self.scenes.contains_key(&scene.name) {
            return Err(SwitcherError::SceneAlreadyExists(scene.name));
        }
        self.scenes.insert(scene.name.clone(), scene);
        Ok(())
    }

    /// Remove a scene by name.
    ///
    /// # Errors
    ///
    /// - [`SwitcherError::SceneNotFound`] when the name is unknown.
    /// - [`SwitcherError::CannotRemoveActive`] when trying to remove the
    ///   currently active scene.
    pub fn remove_scene(&mut self, name: &str) -> Result<Scene, SwitcherError> {
        if self.active_scene.as_deref() == Some(name) {
            return Err(SwitcherError::CannotRemoveActive(name.to_string()));
        }
        self.scenes
            .remove(name)
            .ok_or_else(|| SwitcherError::SceneNotFound(name.to_string()))
    }

    /// Borrow a scene by name.
    #[must_use]
    pub fn scene(&self, name: &str) -> Option<&Scene> {
        self.scenes.get(name)
    }

    /// Mutably borrow a scene by name.
    #[must_use]
    pub fn scene_mut(&mut self, name: &str) -> Option<&mut Scene> {
        self.scenes.get_mut(name)
    }

    // -- Switching ----------------------------------------------------------

    /// Switch to the named scene immediately using its own default transition.
    ///
    /// Records a [`SwitchEvent`] in the history. If the named scene is already
    /// active the call is a no-op (returns `Ok(())` without recording an event).
    ///
    /// # Errors
    ///
    /// Returns [`SwitcherError::SceneNotFound`] if the target scene does not
    /// exist.
    pub fn switch_to(&mut self, name: &str, frame: u64) -> Result<(), SwitcherError> {
        if !self.scenes.contains_key(name) {
            return Err(SwitcherError::SceneNotFound(name.to_string()));
        }
        if self.active_scene.as_deref() == Some(name) {
            return Ok(());
        }
        let transition = self
            .scenes
            .get(name)
            .map(|s| s.transition_type.clone())
            .unwrap_or(TransitionType::Cut);
        let event = SwitchEvent {
            from_scene: self.active_scene.clone(),
            to_scene: name.to_string(),
            transition,
            frame,
        };
        self.history.push(event);
        self.active_scene = Some(name.to_string());
        Ok(())
    }

    /// Switch to the named scene using an explicit transition type.
    ///
    /// # Errors
    ///
    /// Returns [`SwitcherError::SceneNotFound`] if the target scene is unknown.
    pub fn switch_to_with(
        &mut self,
        name: &str,
        transition: TransitionType,
        frame: u64,
    ) -> Result<(), SwitcherError> {
        if !self.scenes.contains_key(name) {
            return Err(SwitcherError::SceneNotFound(name.to_string()));
        }
        if self.active_scene.as_deref() == Some(name) {
            return Ok(());
        }
        let event = SwitchEvent {
            from_scene: self.active_scene.clone(),
            to_scene: name.to_string(),
            transition,
            frame,
        };
        self.history.push(event);
        self.active_scene = Some(name.to_string());
        Ok(())
    }

    /// Returns the name of the currently active scene, if any.
    #[must_use]
    pub fn current_scene(&self) -> Option<&str> {
        self.active_scene.as_deref()
    }

    // -- Scheduling ---------------------------------------------------------

    /// Schedule a switch to `name` at frame `at_frame`.
    ///
    /// The switch is NOT applied until [`SceneSwitcher::tick`] is called with a
    /// frame number ≥ `at_frame`.
    ///
    /// # Errors
    ///
    /// Returns [`SwitcherError::SceneNotFound`] if the target scene is unknown.
    pub fn schedule_switch(
        &mut self,
        name: &str,
        at_frame: u64,
        transition: TransitionType,
    ) -> Result<(), SwitcherError> {
        if !self.scenes.contains_key(name) {
            return Err(SwitcherError::SceneNotFound(name.to_string()));
        }
        self.scheduled.push(ScheduledSwitch {
            target_scene: name.to_string(),
            at_frame,
            transition,
        });
        // Keep earliest-first for predictable determinism
        self.scheduled.sort_by_key(|s| s.at_frame);
        Ok(())
    }

    /// Process scheduled switches up to and including `current_frame`.
    ///
    /// Switches are applied in chronological order. Returns the list of
    /// [`SwitchEvent`]s that were fired.
    pub fn tick(&mut self, current_frame: u64) -> Vec<SwitchEvent> {
        let mut fired = Vec::new();
        loop {
            let Some(first) = self.scheduled.first() else {
                break;
            };
            if first.at_frame > current_frame {
                break;
            }
            let sw = self.scheduled.remove(0);
            if self.active_scene.as_deref() != Some(&sw.target_scene)
                && self.scenes.contains_key(&sw.target_scene)
            {
                let event = SwitchEvent {
                    from_scene: self.active_scene.clone(),
                    to_scene: sw.target_scene.clone(),
                    transition: sw.transition,
                    frame: sw.at_frame,
                };
                self.active_scene = Some(sw.target_scene);
                self.history.push(event.clone());
                fired.push(event);
            }
        }
        fired
    }

    // -- History / introspection --------------------------------------------

    /// Full switch history for this session.
    #[must_use]
    pub fn history(&self) -> &[SwitchEvent] {
        &self.history
    }

    /// Number of scenes registered.
    #[must_use]
    pub fn scene_count(&self) -> usize {
        self.scenes.len()
    }

    /// Number of scheduled switches pending.
    #[must_use]
    pub fn pending_switch_count(&self) -> usize {
        self.scheduled.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scene(name: &str) -> Scene {
        let mut s = Scene::new(name);
        s.add_source(SceneSource::new(1, "Game Capture", true, 1.0));
        s.add_source(SceneSource::new(2, "Webcam", false, 0.8));
        s
    }

    fn make_fade_scene(name: &str) -> Scene {
        let mut s = Scene::new(name);
        s.transition_type = TransitionType::Fade { duration_ms: 300 };
        s
    }

    // -- Scene management --

    #[test]
    fn test_add_and_count_scenes() {
        let mut sw = SceneSwitcher::new();
        sw.add_scene(make_scene("Main")).expect("add Main");
        sw.add_scene(make_scene("BRB")).expect("add BRB");
        assert_eq!(sw.scene_count(), 2);
    }

    #[test]
    fn test_add_duplicate_scene_fails() {
        let mut sw = SceneSwitcher::new();
        sw.add_scene(make_scene("Main")).expect("first add");
        let err = sw.add_scene(make_scene("Main")).expect_err("duplicate");
        assert!(matches!(err, SwitcherError::SceneAlreadyExists(_)));
    }

    #[test]
    fn test_remove_scene() {
        let mut sw = SceneSwitcher::new();
        sw.add_scene(make_scene("Main")).expect("add");
        let removed = sw.remove_scene("Main").expect("remove");
        assert_eq!(removed.name, "Main");
        assert_eq!(sw.scene_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_scene_fails() {
        let mut sw = SceneSwitcher::new();
        let err = sw.remove_scene("Ghost").expect_err("not found");
        assert!(matches!(err, SwitcherError::SceneNotFound(_)));
    }

    #[test]
    fn test_cannot_remove_active_scene() {
        let mut sw = SceneSwitcher::new();
        sw.add_scene(make_scene("Main")).expect("add");
        sw.switch_to("Main", 0).expect("switch");
        let err = sw.remove_scene("Main").expect_err("should fail");
        assert!(matches!(err, SwitcherError::CannotRemoveActive(_)));
    }

    // -- Switching --

    #[test]
    fn test_switch_to_updates_active() {
        let mut sw = SceneSwitcher::new();
        sw.add_scene(make_scene("Main")).expect("add");
        sw.add_scene(make_scene("BRB")).expect("add");
        sw.switch_to("Main", 0).expect("switch");
        assert_eq!(sw.current_scene(), Some("Main"));
        sw.switch_to("BRB", 100).expect("switch");
        assert_eq!(sw.current_scene(), Some("BRB"));
    }

    #[test]
    fn test_switch_to_nonexistent_fails() {
        let mut sw = SceneSwitcher::new();
        let err = sw.switch_to("Ghost", 0).expect_err("not found");
        assert!(matches!(err, SwitcherError::SceneNotFound(_)));
    }

    #[test]
    fn test_switch_to_same_scene_is_noop() {
        let mut sw = SceneSwitcher::new();
        sw.add_scene(make_scene("Main")).expect("add");
        sw.switch_to("Main", 0).expect("switch");
        sw.switch_to("Main", 1).expect("noop");
        assert_eq!(sw.history().len(), 1);
    }

    #[test]
    fn test_switch_records_history() {
        let mut sw = SceneSwitcher::new();
        sw.add_scene(make_scene("Main")).expect("add");
        sw.add_scene(make_scene("BRB")).expect("add");
        sw.switch_to("Main", 0).expect("switch");
        sw.switch_to("BRB", 60).expect("switch");
        assert_eq!(sw.history().len(), 2);
        let ev = &sw.history()[1];
        assert_eq!(ev.from_scene.as_deref(), Some("Main"));
        assert_eq!(ev.to_scene, "BRB");
        assert_eq!(ev.frame, 60);
    }

    #[test]
    fn test_switch_to_with_explicit_transition() {
        let mut sw = SceneSwitcher::new();
        sw.add_scene(make_scene("Main")).expect("add");
        sw.add_scene(make_scene("Cam")).expect("add");
        sw.switch_to("Main", 0).expect("switch");
        sw.switch_to_with("Cam", TransitionType::Fade { duration_ms: 500 }, 60)
            .expect("switch");
        let ev = &sw.history()[1];
        assert_eq!(ev.transition, TransitionType::Fade { duration_ms: 500 });
    }

    #[test]
    fn test_transition_duration_ms() {
        assert_eq!(TransitionType::Cut.duration_ms(), 0);
        assert_eq!(TransitionType::Stinger.duration_ms(), 0);
        assert_eq!(TransitionType::Fade { duration_ms: 250 }.duration_ms(), 250);
        assert_eq!(
            TransitionType::Slide {
                direction: SlideDir::Left,
                duration_ms: 400
            }
            .duration_ms(),
            400
        );
    }

    #[test]
    fn test_transition_requires_blend() {
        assert!(!TransitionType::Cut.requires_blend());
        assert!(!TransitionType::Stinger.requires_blend());
        assert!(TransitionType::Fade { duration_ms: 100 }.requires_blend());
        assert!(TransitionType::Slide {
            direction: SlideDir::Right,
            duration_ms: 200
        }
        .requires_blend());
    }

    #[test]
    fn test_scene_default_transition_used_on_switch() {
        let mut sw = SceneSwitcher::new();
        sw.add_scene(make_scene("Main")).expect("add");
        sw.add_scene(make_fade_scene("Cam")).expect("add");
        sw.switch_to("Main", 0).expect("switch to main");
        sw.switch_to("Cam", 60).expect("switch to cam");
        let ev = &sw.history()[1];
        assert_eq!(ev.transition, TransitionType::Fade { duration_ms: 300 });
    }

    // -- Scheduling --

    #[test]
    fn test_schedule_switch_fires_on_tick() {
        let mut sw = SceneSwitcher::new();
        sw.add_scene(make_scene("Main")).expect("add");
        sw.add_scene(make_scene("BRB")).expect("add");
        sw.switch_to("Main", 0).expect("switch");
        sw.schedule_switch("BRB", 300, TransitionType::Cut)
            .expect("schedule");
        assert_eq!(sw.pending_switch_count(), 1);

        let fired = sw.tick(299);
        assert!(fired.is_empty());
        assert_eq!(sw.current_scene(), Some("Main"));

        let fired = sw.tick(300);
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].to_scene, "BRB");
        assert_eq!(sw.current_scene(), Some("BRB"));
        assert_eq!(sw.pending_switch_count(), 0);
    }

    #[test]
    fn test_schedule_switch_nonexistent_fails() {
        let mut sw = SceneSwitcher::new();
        let err = sw
            .schedule_switch("Ghost", 100, TransitionType::Cut)
            .expect_err("not found");
        assert!(matches!(err, SwitcherError::SceneNotFound(_)));
    }

    #[test]
    fn test_multiple_scheduled_switches_ordered() {
        let mut sw = SceneSwitcher::new();
        sw.add_scene(make_scene("A")).expect("add");
        sw.add_scene(make_scene("B")).expect("add");
        sw.add_scene(make_scene("C")).expect("add");
        sw.switch_to("A", 0).expect("switch");
        // Schedule out of order
        sw.schedule_switch("C", 200, TransitionType::Cut)
            .expect("schedule C");
        sw.schedule_switch("B", 100, TransitionType::Cut)
            .expect("schedule B");

        let fired = sw.tick(150);
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].to_scene, "B");

        let fired = sw.tick(250);
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].to_scene, "C");
    }

    // -- Scene source operations --

    #[test]
    fn test_scene_source_visibility() {
        let mut sw = SceneSwitcher::new();
        sw.add_scene(make_scene("Main")).expect("add");
        let scene = sw.scene_mut("Main").expect("scene");
        assert_eq!(scene.visible_source_count(), 1); // only source 1 is visible
        scene.source_mut(2).expect("source 2").visible = true;
        assert_eq!(scene.visible_source_count(), 2);
    }

    #[test]
    fn test_scene_source_volume_clamped() {
        let src = SceneSource::new(1, "Test", true, 1.5);
        assert!((src.volume - 1.0).abs() < f32::EPSILON);
        let src_low = SceneSource::new(2, "Low", true, -0.3);
        assert!((src_low.volume - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_slide_dir_opposite() {
        assert_eq!(SlideDir::Left.opposite(), SlideDir::Right);
        assert_eq!(SlideDir::Right.opposite(), SlideDir::Left);
        assert_eq!(SlideDir::Up.opposite(), SlideDir::Down);
        assert_eq!(SlideDir::Down.opposite(), SlideDir::Up);
    }
}
