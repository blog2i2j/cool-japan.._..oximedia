//! CG (Character Generator) graphics overlay module.
//!
//! Provides lower thirds, full-frame titles, logo bugs, and animated CG
//! elements for live production graphics pipelines.  Each overlay is
//! described by a logical descriptor; actual pixel rendering is delegated
//! to the downstream compositor.
//!
//! # Design
//!
//! - `OverlayLayer` represents a single CG element (lower-third, title, logo).
//! - `GraphicsOverlayManager` manages a z-ordered stack of layers and handles
//!   scheduling of animations (in/hold/out durations).
//! - All coordinates are normalised to the range `[0.0, 1.0]` relative to the
//!   frame dimensions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors produced by graphics overlay operations.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum GraphicsOverlayError {
    /// The requested layer ID does not exist.
    #[error("overlay layer {0} not found")]
    LayerNotFound(usize),
    /// An overlay with this ID is already registered.
    #[error("overlay layer {0} already exists")]
    LayerAlreadyExists(usize),
    /// An animation parameter is out of the valid range.
    #[error("invalid animation parameter: {0}")]
    InvalidParameter(String),
    /// The graphics channel is full.
    #[error("graphics channel capacity ({0}) exceeded")]
    CapacityExceeded(usize),
}

// ── Animation / transition style ─────────────────────────────────────────────

/// How a CG element transitions on or off screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CgTransitionStyle {
    /// Instantaneous cut.
    Cut,
    /// Linear fade in/out via alpha.
    Fade,
    /// Horizontal slide from the left edge.
    SlideLeft,
    /// Horizontal slide from the right edge.
    SlideRight,
    /// Vertical slide from the bottom.
    SlideUp,
    /// Vertical slide from the top.
    SlideDown,
    /// Scale-up (appear from a small central region).
    ScaleUp,
}

impl Default for CgTransitionStyle {
    fn default() -> Self {
        Self::Fade
    }
}

/// Timing descriptor for a single CG element animation phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CgTiming {
    /// Duration of the in-animation (element appearing).
    pub animate_in: Duration,
    /// Duration for which the element holds at full visibility.
    pub hold: Duration,
    /// Duration of the out-animation (element disappearing).
    pub animate_out: Duration,
}

impl CgTiming {
    /// Create a simple timing with equal in/out durations.
    pub fn new(animate_in: Duration, hold: Duration, animate_out: Duration) -> Self {
        Self {
            animate_in,
            hold,
            animate_out,
        }
    }

    /// Total lifecycle duration.
    pub fn total_duration(&self) -> Duration {
        self.animate_in + self.hold + self.animate_out
    }

    /// Create an instant cut with the given hold duration.
    pub fn instant(hold: Duration) -> Self {
        Self::new(Duration::ZERO, hold, Duration::ZERO)
    }

    /// Create a short fade (250 ms each side) with the given hold duration.
    pub fn quick_fade(hold: Duration) -> Self {
        Self::new(Duration::from_millis(250), hold, Duration::from_millis(250))
    }
}

// ── Content descriptors ───────────────────────────────────────────────────────

/// A lower-third template — the most common CG element in broadcast production.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowerThirdContent {
    /// Primary line text (e.g. subject name).
    pub primary: String,
    /// Secondary line text (e.g. title / affiliation).
    pub secondary: Option<String>,
    /// Optional logo or icon ID to display alongside the text.
    pub logo_id: Option<String>,
    /// Background bar colour as RGBA (each 0–255).
    pub bar_colour: [u8; 4],
    /// Text colour as RGBA.
    pub text_colour: [u8; 4],
    /// Vertical position as fraction of frame height (0.0 = top, 1.0 = bottom).
    pub v_position: f32,
    /// Horizontal margin as fraction of frame width.
    pub h_margin: f32,
}

impl LowerThirdContent {
    /// Create a standard lower-third.
    pub fn new(primary: impl Into<String>) -> Self {
        Self {
            primary: primary.into(),
            secondary: None,
            logo_id: None,
            bar_colour: [20, 80, 180, 220],
            text_colour: [255, 255, 255, 255],
            v_position: 0.82,
            h_margin: 0.04,
        }
    }

    /// Add secondary line text.
    pub fn with_secondary(mut self, text: impl Into<String>) -> Self {
        self.secondary = Some(text.into());
        self
    }

    /// Set the bar colour.
    pub fn with_bar_colour(mut self, rgba: [u8; 4]) -> Self {
        self.bar_colour = rgba;
        self
    }

    /// Set vertical position.
    pub fn at_position(mut self, v_pos: f32) -> Self {
        self.v_position = v_pos.clamp(0.0, 1.0);
        self
    }
}

/// A full-frame title overlay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TitleContent {
    /// Main title text.
    pub headline: String,
    /// Optional subtitle text below the headline.
    pub subtitle: Option<String>,
    /// Background opacity (0.0 = transparent, 1.0 = fully opaque black).
    pub background_alpha: f32,
    /// Title font size in normalised units (1.0 = 10% of frame height).
    pub font_size: f32,
    /// Text colour.
    pub colour: [u8; 4],
}

impl TitleContent {
    /// Create a full-screen title.
    pub fn new(headline: impl Into<String>) -> Self {
        Self {
            headline: headline.into(),
            subtitle: None,
            background_alpha: 0.6,
            font_size: 1.0,
            colour: [255, 255, 255, 255],
        }
    }

    /// Add subtitle.
    pub fn with_subtitle(mut self, text: impl Into<String>) -> Self {
        self.subtitle = Some(text.into());
        self
    }
}

/// A logo bug (station ID / watermark) overlay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogoBugContent {
    /// Asset ID referencing the logo image in the media pool.
    pub asset_id: String,
    /// Normalised X position of the logo's top-left corner.
    pub x: f32,
    /// Normalised Y position of the logo's top-left corner.
    pub y: f32,
    /// Normalised width of the logo.
    pub width: f32,
    /// Normalised height of the logo.
    pub height: f32,
    /// Opacity (0.0 to 1.0).
    pub opacity: f32,
}

impl LogoBugContent {
    /// Standard bottom-right logo bug.
    pub fn bottom_right(asset_id: impl Into<String>) -> Self {
        Self {
            asset_id: asset_id.into(),
            x: 0.86,
            y: 0.04,
            width: 0.10,
            height: 0.08,
            opacity: 0.85,
        }
    }

    /// Top-left logo bug (common for sporting events).
    pub fn top_left(asset_id: impl Into<String>) -> Self {
        Self {
            asset_id: asset_id.into(),
            x: 0.04,
            y: 0.04,
            width: 0.10,
            height: 0.08,
            opacity: 0.85,
        }
    }
}

/// The content variant held by an overlay layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OverlayContent {
    /// Lower-third name/title strap.
    LowerThird(LowerThirdContent),
    /// Full-frame title card.
    Title(TitleContent),
    /// Station-ID logo bug / watermark.
    LogoBug(LogoBugContent),
    /// Raw CG markup string (renderer-specific DSL).
    Custom(String),
}

impl OverlayContent {
    /// Human-readable type name.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::LowerThird(_) => "LowerThird",
            Self::Title(_) => "Title",
            Self::LogoBug(_) => "LogoBug",
            Self::Custom(_) => "Custom",
        }
    }
}

// ── Playback state machine ────────────────────────────────────────────────────

/// Lifecycle state of a single overlay layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LayerState {
    #[default]
    /// Layer is loaded but not yet visible.
    Ready,
    /// Layer is animating on-screen.
    AnimatingIn,
    /// Layer is fully visible and holding.
    Live,
    /// Layer is animating off-screen.
    AnimatingOut,
    /// Layer has completed its out animation and is no longer visible.
    Done,
}

/// Runtime state tracked per layer (not serialised — rebuilt on load).
#[derive(Debug, Default)]
struct LayerRuntime {
    state: LayerState,
    phase_started_at: Option<Instant>,
}

impl LayerRuntime {
    fn new() -> Self {
        Self {
            state: LayerState::Ready,
            phase_started_at: None,
        }
    }

    fn phase_elapsed(&self) -> Duration {
        self.phase_started_at
            .map(|t| t.elapsed())
            .unwrap_or(Duration::ZERO)
    }
}

// ── Overlay layer ─────────────────────────────────────────────────────────────

/// A single CG overlay layer registered in the graphics channel.
#[derive(Debug, Serialize, Deserialize)]
pub struct OverlayLayer {
    /// Unique identifier.
    pub id: usize,
    /// Z-order (higher = drawn on top).
    pub z_order: i32,
    /// The graphical content.
    pub content: OverlayContent,
    /// Transition style.
    pub transition: CgTransitionStyle,
    /// Timing configuration.
    pub timing: CgTiming,
    /// Whether to auto-advance through the full animate-in → hold → animate-out
    /// lifecycle automatically when `tick()` is called.
    pub auto_run: bool,
    #[serde(skip)]
    runtime: LayerRuntime,
}

impl OverlayLayer {
    /// Create a new overlay layer.
    pub fn new(id: usize, content: OverlayContent, timing: CgTiming) -> Self {
        Self {
            id,
            z_order: 0,
            content,
            transition: CgTransitionStyle::default(),
            timing,
            auto_run: true,
            runtime: LayerRuntime::new(),
        }
    }

    /// Current lifecycle state.
    pub fn state(&self) -> LayerState {
        self.runtime.state
    }

    /// Trigger the animate-in phase.
    pub fn animate_in(&mut self) {
        if self.runtime.state == LayerState::Ready || self.runtime.state == LayerState::Done {
            self.runtime.state = LayerState::AnimatingIn;
            self.runtime.phase_started_at = Some(Instant::now());
        }
    }

    /// Trigger the animate-out phase.
    pub fn animate_out(&mut self) {
        if self.runtime.state == LayerState::Live {
            self.runtime.state = LayerState::AnimatingOut;
            self.runtime.phase_started_at = Some(Instant::now());
        }
    }

    /// Advance the state machine according to elapsed time.
    /// Returns `true` when the layer transitions to a new state.
    pub fn tick(&mut self) -> bool {
        let elapsed = self.runtime.phase_elapsed();
        match self.runtime.state {
            LayerState::AnimatingIn => {
                if elapsed >= self.timing.animate_in {
                    self.runtime.state = LayerState::Live;
                    self.runtime.phase_started_at = Some(Instant::now());
                    return true;
                }
            }
            LayerState::Live => {
                if self.auto_run && elapsed >= self.timing.hold {
                    self.runtime.state = LayerState::AnimatingOut;
                    self.runtime.phase_started_at = Some(Instant::now());
                    return true;
                }
            }
            LayerState::AnimatingOut => {
                if elapsed >= self.timing.animate_out {
                    self.runtime.state = LayerState::Done;
                    self.runtime.phase_started_at = None;
                    return true;
                }
            }
            LayerState::Ready | LayerState::Done => {}
        }
        false
    }

    /// Returns the current alpha (0.0–1.0) based on animation progress.
    pub fn current_alpha(&self) -> f32 {
        match self.runtime.state {
            LayerState::Ready | LayerState::Done => 0.0,
            LayerState::Live => 1.0,
            LayerState::AnimatingIn => {
                let dur = self.timing.animate_in;
                if dur == Duration::ZERO {
                    return 1.0;
                }
                let ratio = self.runtime.phase_elapsed().as_secs_f32()
                    / dur.as_secs_f32().max(f32::EPSILON);
                ratio.clamp(0.0, 1.0)
            }
            LayerState::AnimatingOut => {
                let dur = self.timing.animate_out;
                if dur == Duration::ZERO {
                    return 0.0;
                }
                let ratio = self.runtime.phase_elapsed().as_secs_f32()
                    / dur.as_secs_f32().max(f32::EPSILON);
                (1.0 - ratio).clamp(0.0, 1.0)
            }
        }
    }

    /// Set the z-order.
    pub fn with_z_order(mut self, z: i32) -> Self {
        self.z_order = z;
        self
    }

    /// Set transition style.
    pub fn with_transition(mut self, style: CgTransitionStyle) -> Self {
        self.transition = style;
        self
    }
}

// ── Graphics overlay manager ──────────────────────────────────────────────────

/// Manages all CG overlay layers in a single graphics channel.
///
/// Layers are stored in z-order (lowest first) for compositor traversal.
pub struct GraphicsOverlayManager {
    layers: HashMap<usize, OverlayLayer>,
    capacity: usize,
}

impl GraphicsOverlayManager {
    /// Create a manager with the given maximum layer capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            layers: HashMap::new(),
            capacity,
        }
    }

    /// Add a layer.
    pub fn add_layer(&mut self, layer: OverlayLayer) -> Result<(), GraphicsOverlayError> {
        if self.layers.contains_key(&layer.id) {
            return Err(GraphicsOverlayError::LayerAlreadyExists(layer.id));
        }
        if self.layers.len() >= self.capacity {
            return Err(GraphicsOverlayError::CapacityExceeded(self.capacity));
        }
        self.layers.insert(layer.id, layer);
        Ok(())
    }

    /// Remove a layer.
    pub fn remove_layer(&mut self, id: usize) -> Result<OverlayLayer, GraphicsOverlayError> {
        self.layers
            .remove(&id)
            .ok_or(GraphicsOverlayError::LayerNotFound(id))
    }

    /// Get a reference to a layer.
    pub fn layer(&self, id: usize) -> Option<&OverlayLayer> {
        self.layers.get(&id)
    }

    /// Get a mutable reference to a layer.
    pub fn layer_mut(&mut self, id: usize) -> Option<&mut OverlayLayer> {
        self.layers.get_mut(&id)
    }

    /// Number of registered layers.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Trigger animate-in for a specific layer.
    pub fn take_in(&mut self, id: usize) -> Result<(), GraphicsOverlayError> {
        let layer = self
            .layers
            .get_mut(&id)
            .ok_or(GraphicsOverlayError::LayerNotFound(id))?;
        layer.animate_in();
        Ok(())
    }

    /// Trigger animate-out for a specific layer.
    pub fn take_out(&mut self, id: usize) -> Result<(), GraphicsOverlayError> {
        let layer = self
            .layers
            .get_mut(&id)
            .ok_or(GraphicsOverlayError::LayerNotFound(id))?;
        layer.animate_out();
        Ok(())
    }

    /// Advance all layers' state machines.  Removes layers in `Done` state
    /// if `prune_done` is `true`.
    pub fn tick_all(&mut self, prune_done: bool) {
        for layer in self.layers.values_mut() {
            layer.tick();
        }
        if prune_done {
            self.layers.retain(|_, l| l.state() != LayerState::Done);
        }
    }

    /// Return layer IDs sorted by z-order (ascending — lowest drawn first).
    pub fn z_ordered_ids(&self) -> Vec<usize> {
        let mut ids: Vec<(i32, usize)> = self.layers.values().map(|l| (l.z_order, l.id)).collect();
        ids.sort_by_key(|&(z, _)| z);
        ids.into_iter().map(|(_, id)| id).collect()
    }

    /// Return IDs of all currently visible (not `Ready` or `Done`) layers.
    pub fn visible_layer_ids(&self) -> Vec<usize> {
        self.layers
            .values()
            .filter(|l| {
                matches!(
                    l.state(),
                    LayerState::AnimatingIn | LayerState::Live | LayerState::AnimatingOut
                )
            })
            .map(|l| l.id)
            .collect()
    }

    /// Convenience: build and immediately animate-in a lower-third.
    pub fn show_lower_third(
        &mut self,
        id: usize,
        content: LowerThirdContent,
        timing: CgTiming,
    ) -> Result<(), GraphicsOverlayError> {
        let mut layer = OverlayLayer::new(id, OverlayContent::LowerThird(content), timing);
        layer.animate_in();
        self.add_layer(layer)
    }

    /// Convenience: build and immediately animate-in a title card.
    pub fn show_title(
        &mut self,
        id: usize,
        content: TitleContent,
        timing: CgTiming,
    ) -> Result<(), GraphicsOverlayError> {
        let mut layer = OverlayLayer::new(id, OverlayContent::Title(content), timing);
        layer.animate_in();
        self.add_layer(layer)
    }

    /// Convenience: add a persistent logo bug (hold = max, no auto-out).
    pub fn set_logo_bug(
        &mut self,
        id: usize,
        content: LogoBugContent,
    ) -> Result<(), GraphicsOverlayError> {
        let timing = CgTiming::quick_fade(Duration::from_secs(u32::MAX as u64));
        let mut layer = OverlayLayer::new(id, OverlayContent::LogoBug(content), timing);
        layer.auto_run = false;
        layer.animate_in();
        self.add_layer(layer)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn quick_timing() -> CgTiming {
        CgTiming::instant(Duration::from_millis(100))
    }

    fn manager() -> GraphicsOverlayManager {
        GraphicsOverlayManager::new(16)
    }

    #[test]
    fn test_add_and_count_layers() {
        let mut mgr = manager();
        let lt = LowerThirdContent::new("Jane Smith").with_secondary("Reporter");
        mgr.show_lower_third(1, lt, quick_timing())
            .expect("add layer");
        assert_eq!(mgr.layer_count(), 1);
    }

    #[test]
    fn test_add_duplicate_id_errors() {
        let mut mgr = manager();
        let lt1 = LowerThirdContent::new("A");
        let lt2 = LowerThirdContent::new("B");
        mgr.show_lower_third(1, lt1, quick_timing())
            .expect("first ok");
        let err = mgr.show_lower_third(1, lt2, quick_timing());
        assert_eq!(err, Err(GraphicsOverlayError::LayerAlreadyExists(1)));
    }

    #[test]
    fn test_remove_layer() {
        let mut mgr = manager();
        let lt = LowerThirdContent::new("Test");
        mgr.show_lower_third(2, lt, quick_timing()).expect("add");
        mgr.remove_layer(2).expect("remove");
        assert_eq!(mgr.layer_count(), 0);
    }

    #[test]
    fn test_remove_missing_layer_errors() {
        let mut mgr = manager();
        let result = mgr.remove_layer(99);
        assert!(result.is_err());
        match result {
            Err(GraphicsOverlayError::LayerNotFound(id)) => assert_eq!(id, 99),
            _ => panic!("unexpected result"),
        }
    }

    #[test]
    fn test_take_in_transitions_layer_to_animating_in() {
        let mut mgr = manager();
        let lt = LowerThirdContent::new("Live");
        let timing = CgTiming::new(
            Duration::from_millis(500),
            Duration::from_secs(5),
            Duration::from_millis(500),
        );
        let layer = OverlayLayer::new(3, OverlayContent::LowerThird(lt), timing);
        mgr.add_layer(layer).expect("add");
        mgr.take_in(3).expect("animate in");
        assert_eq!(
            mgr.layer(3).expect("exists").state(),
            LayerState::AnimatingIn
        );
    }

    #[test]
    fn test_capacity_exceeded_errors() {
        let mut mgr = GraphicsOverlayManager::new(1);
        let lt1 = LowerThirdContent::new("A");
        let lt2 = LowerThirdContent::new("B");
        mgr.show_lower_third(1, lt1, quick_timing()).expect("first");
        let err = mgr.show_lower_third(2, lt2, quick_timing());
        assert_eq!(err, Err(GraphicsOverlayError::CapacityExceeded(1)));
    }

    #[test]
    fn test_z_ordered_ids() {
        let mut mgr = manager();
        let mut a = OverlayLayer::new(10, OverlayContent::Custom("a".into()), quick_timing())
            .with_z_order(2);
        let mut b = OverlayLayer::new(11, OverlayContent::Custom("b".into()), quick_timing())
            .with_z_order(0);
        let mut c = OverlayLayer::new(12, OverlayContent::Custom("c".into()), quick_timing())
            .with_z_order(1);
        a.animate_in();
        b.animate_in();
        c.animate_in();
        mgr.add_layer(a).expect("a");
        mgr.add_layer(b).expect("b");
        mgr.add_layer(c).expect("c");
        let ids = mgr.z_ordered_ids();
        assert_eq!(ids[0], 11); // z=0 first
        assert_eq!(ids[1], 12); // z=1
        assert_eq!(ids[2], 10); // z=2
    }

    #[test]
    fn test_visible_layer_ids_shows_animating_in() {
        let mut mgr = manager();
        let lt = LowerThirdContent::new("On air");
        let timing = CgTiming::new(
            Duration::from_millis(200),
            Duration::from_secs(10),
            Duration::from_millis(200),
        );
        let mut layer = OverlayLayer::new(5, OverlayContent::LowerThird(lt), timing);
        layer.animate_in();
        mgr.add_layer(layer).expect("add");
        let visible = mgr.visible_layer_ids();
        assert!(visible.contains(&5));
    }

    #[test]
    fn test_logo_bug_convenience() {
        let mut mgr = manager();
        let logo = LogoBugContent::bottom_right("station_logo");
        mgr.set_logo_bug(20, logo).expect("set logo");
        let layer = mgr.layer(20).expect("exists");
        assert_eq!(layer.state(), LayerState::AnimatingIn);
        assert!(!layer.auto_run);
    }

    #[test]
    fn test_layer_alpha_ready_is_zero() {
        let lt = LowerThirdContent::new("Hidden");
        let layer = OverlayLayer::new(0, OverlayContent::LowerThird(lt), quick_timing());
        assert_eq!(layer.current_alpha(), 0.0);
        assert_eq!(layer.state(), LayerState::Ready);
    }

    #[test]
    fn test_layer_tick_cut_transition_to_live() {
        let lt = LowerThirdContent::new("Fast");
        // Zero animate_in so tick immediately moves to Live
        let timing = CgTiming::new(Duration::ZERO, Duration::from_secs(10), Duration::ZERO);
        let mut layer = OverlayLayer::new(0, OverlayContent::LowerThird(lt), timing);
        layer.animate_in();
        assert_eq!(layer.state(), LayerState::AnimatingIn);
        layer.tick();
        assert_eq!(layer.state(), LayerState::Live);
    }

    #[test]
    fn test_lower_third_builder() {
        let lt = LowerThirdContent::new("Dr. Alice")
            .with_secondary("Chief Scientist")
            .with_bar_colour([0, 0, 0, 200])
            .at_position(0.75);
        assert_eq!(lt.primary, "Dr. Alice");
        assert_eq!(lt.secondary.as_deref(), Some("Chief Scientist"));
        assert_eq!(lt.v_position, 0.75);
    }

    #[test]
    fn test_title_builder() {
        let title = TitleContent::new("Breaking News").with_subtitle("Live from Studio A");
        assert_eq!(title.headline, "Breaking News");
        assert!(title.subtitle.is_some());
    }

    #[test]
    fn test_cg_timing_total_duration() {
        let t = CgTiming::new(
            Duration::from_millis(250),
            Duration::from_secs(5),
            Duration::from_millis(250),
        );
        assert_eq!(t.total_duration(), Duration::from_millis(5500));
    }

    #[test]
    fn test_tick_all_prune_done() {
        let mut mgr = manager();
        // Use cut timing with zero durations so tick instantly completes all phases
        let timing = CgTiming::new(Duration::ZERO, Duration::ZERO, Duration::ZERO);
        let mut layer = OverlayLayer::new(99, OverlayContent::Custom("x".into()), timing);
        layer.animate_in();
        mgr.add_layer(layer).expect("add");
        // Tick until Done
        for _ in 0..10 {
            mgr.tick_all(false);
        }
        assert_eq!(mgr.layer(99).expect("exists").state(), LayerState::Done);
        mgr.tick_all(true); // now prune
        assert_eq!(mgr.layer_count(), 0);
    }
}
