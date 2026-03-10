//! Transition library for the production switcher.
//!
//! Provides wipe patterns, push/squeeze effects, DVE-based transitions,
//! and custom transition path definitions.

#![allow(dead_code)]

/// A 2-D vector used to describe push/squeeze directions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    /// Horizontal component.
    pub x: f32,
    /// Vertical component.
    pub y: f32,
}

impl Vec2 {
    /// Create a new vector.
    #[must_use]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Left direction.
    #[must_use]
    pub fn left() -> Self {
        Self::new(-1.0, 0.0)
    }

    /// Right direction.
    #[must_use]
    pub fn right() -> Self {
        Self::new(1.0, 0.0)
    }

    /// Up direction.
    #[must_use]
    pub fn up() -> Self {
        Self::new(0.0, -1.0)
    }

    /// Down direction.
    #[must_use]
    pub fn down() -> Self {
        Self::new(0.0, 1.0)
    }

    /// Magnitude (length) of the vector.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Normalise the vector to unit length (no-op for zero vector).
    #[must_use]
    pub fn normalised(&self) -> Self {
        let m = self.magnitude();
        if m < 1e-6 {
            *self
        } else {
            Self::new(self.x / m, self.y / m)
        }
    }
}

/// A wipe pattern shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WipeShape {
    /// Horizontal split wipe.
    HorizontalSplit,
    /// Vertical split wipe.
    VerticalSplit,
    /// Diagonal wipe from top-left to bottom-right.
    DiagonalTopLeft,
    /// Diagonal wipe from top-right to bottom-left.
    DiagonalTopRight,
    /// Circular expanding wipe.
    Circle,
    /// Diamond-shaped wipe.
    Diamond,
    /// Four-corner box wipe.
    Box,
    /// Venetian-blind horizontal wipe.
    BlindsHorizontal,
    /// Venetian-blind vertical wipe.
    BlindsVertical,
}

/// Configuration for a wipe transition.
#[derive(Debug, Clone)]
pub struct WipeConfig {
    /// Wipe shape.
    pub shape: WipeShape,
    /// Angle offset in degrees (for diagonal / custom wipes).
    pub angle_deg: f32,
    /// Border width (0.0 = no border, 1.0 = full frame).
    pub border_width: f32,
    /// Whether the wipe direction is reversed.
    pub reversed: bool,
    /// Whether to add a soft edge.
    pub soft_edge: bool,
    /// Soft edge width as a fraction of the frame.
    pub soft_edge_width: f32,
}

impl WipeConfig {
    /// Create a simple wipe with default settings.
    #[must_use]
    pub fn new(shape: WipeShape) -> Self {
        Self {
            shape,
            angle_deg: 0.0,
            border_width: 0.0,
            reversed: false,
            soft_edge: false,
            soft_edge_width: 0.02,
        }
    }

    /// Create a bordered wipe.
    #[must_use]
    pub fn with_border(mut self, width: f32) -> Self {
        self.border_width = width;
        self
    }

    /// Enable soft edge.
    #[must_use]
    pub fn with_soft_edge(mut self, width: f32) -> Self {
        self.soft_edge = true;
        self.soft_edge_width = width;
        self
    }

    /// Reverse the wipe direction.
    #[must_use]
    pub fn reversed(mut self) -> Self {
        self.reversed = true;
        self
    }
}

/// Configuration for a push transition.
#[derive(Debug, Clone)]
pub struct PushConfig {
    /// Direction of the push.
    pub direction: Vec2,
    /// Duration in frames.
    pub duration_frames: u32,
    /// Easing curve (0.0 = linear, 1.0 = full ease).
    pub ease: f32,
}

impl PushConfig {
    /// Create a push to the left.
    #[must_use]
    pub fn left(duration_frames: u32) -> Self {
        Self {
            direction: Vec2::left(),
            duration_frames,
            ease: 0.5,
        }
    }

    /// Create a push to the right.
    #[must_use]
    pub fn right(duration_frames: u32) -> Self {
        Self {
            direction: Vec2::right(),
            duration_frames,
            ease: 0.5,
        }
    }

    /// Create a push upward.
    #[must_use]
    pub fn up(duration_frames: u32) -> Self {
        Self {
            direction: Vec2::up(),
            duration_frames,
            ease: 0.5,
        }
    }

    /// Create a push downward.
    #[must_use]
    pub fn down(duration_frames: u32) -> Self {
        Self {
            direction: Vec2::down(),
            duration_frames,
            ease: 0.5,
        }
    }
}

/// Configuration for a squeeze (shrink) transition.
#[derive(Debug, Clone)]
pub struct SqueezeConfig {
    /// Direction the outgoing image squeezes towards.
    pub direction: Vec2,
    /// Duration in frames.
    pub duration_frames: u32,
    /// Whether to simultaneously expand the incoming image (true) or reveal
    /// it underneath (false).
    pub expand_incoming: bool,
}

impl SqueezeConfig {
    /// Create a squeeze to the left.
    #[must_use]
    pub fn left(duration_frames: u32) -> Self {
        Self {
            direction: Vec2::left(),
            duration_frames,
            expand_incoming: true,
        }
    }
}

/// A keyframe on a DVE transition path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DveKeyframe {
    /// Normalised time (0.0 = start, 1.0 = end).
    pub t: f32,
    /// X position in screen-normalised coordinates (-1.0 to 1.0).
    pub x: f32,
    /// Y position in screen-normalised coordinates (-1.0 to 1.0).
    pub y: f32,
    /// Scale (1.0 = full size).
    pub scale: f32,
    /// Rotation in degrees.
    pub rotation_deg: f32,
}

impl DveKeyframe {
    /// Create a new DVE keyframe.
    #[must_use]
    pub fn new(t: f32, x: f32, y: f32, scale: f32, rotation_deg: f32) -> Self {
        Self {
            t,
            x,
            y,
            scale,
            rotation_deg,
        }
    }
}

/// A DVE (Digital Video Effect) transition path.
#[derive(Debug, Clone)]
pub struct DvePath {
    /// Ordered keyframes (by `t`).
    pub keyframes: Vec<DveKeyframe>,
    /// Duration in frames.
    pub duration_frames: u32,
}

impl DvePath {
    /// Create an empty DVE path.
    #[must_use]
    pub fn new(duration_frames: u32) -> Self {
        Self {
            keyframes: Vec::new(),
            duration_frames,
        }
    }

    /// Add a keyframe (keyframes must be added in time order).
    pub fn add_keyframe(&mut self, kf: DveKeyframe) {
        self.keyframes.push(kf);
    }

    /// Interpolate position at normalised time `t` using linear interpolation.
    /// Returns `(x, y, scale, rotation_deg)` or `None` if no keyframes exist.
    #[must_use]
    pub fn interpolate(&self, t: f32) -> Option<(f32, f32, f32, f32)> {
        if self.keyframes.is_empty() {
            return None;
        }
        if self.keyframes.len() == 1 {
            let kf = &self.keyframes[0];
            return Some((kf.x, kf.y, kf.scale, kf.rotation_deg));
        }

        // Find surrounding keyframes
        let after = self.keyframes.iter().position(|kf| kf.t >= t);
        match after {
            None => {
                let last = self.keyframes.last()?;
                Some((last.x, last.y, last.scale, last.rotation_deg))
            }
            Some(0) => {
                let first = &self.keyframes[0];
                Some((first.x, first.y, first.scale, first.rotation_deg))
            }
            Some(i) => {
                let a = &self.keyframes[i - 1];
                let b = &self.keyframes[i];
                let span = b.t - a.t;
                let alpha = if span < 1e-6 { 0.0 } else { (t - a.t) / span };
                let lerp = |va: f32, vb: f32| va + (vb - va) * alpha;
                Some((
                    lerp(a.x, b.x),
                    lerp(a.y, b.y),
                    lerp(a.scale, b.scale),
                    lerp(a.rotation_deg, b.rotation_deg),
                ))
            }
        }
    }
}

/// The kind of transition in the library.
#[derive(Debug, Clone)]
pub enum TransitionKind {
    /// Wipe-based transition.
    Wipe(WipeConfig),
    /// Push transition.
    Push(PushConfig),
    /// Squeeze transition.
    Squeeze(SqueezeConfig),
    /// DVE path transition.
    Dve(DvePath),
}

/// A named entry in the transition library.
#[derive(Debug, Clone)]
pub struct TransitionEntry {
    /// Entry identifier.
    pub id: u32,
    /// Display name.
    pub name: String,
    /// Transition kind.
    pub kind: TransitionKind,
    /// Default duration in frames (can be overridden at playback).
    pub default_duration_frames: u32,
    /// Favourite flag for quick access.
    pub favourite: bool,
}

impl TransitionEntry {
    /// Create a new transition entry.
    #[must_use]
    pub fn new(
        id: u32,
        name: impl Into<String>,
        kind: TransitionKind,
        default_duration_frames: u32,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            kind,
            default_duration_frames,
            favourite: false,
        }
    }

    /// Mark as favourite.
    pub fn set_favourite(&mut self, fav: bool) {
        self.favourite = fav;
    }
}

/// The transition library holding all available transitions.
#[derive(Debug, Default)]
pub struct TransitionLibrary {
    /// All stored transition entries.
    entries: Vec<TransitionEntry>,
}

impl TransitionLibrary {
    /// Create an empty library.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a transition entry.
    pub fn add(&mut self, entry: TransitionEntry) {
        self.entries.push(entry);
    }

    /// Find by ID.
    #[must_use]
    pub fn find(&self, id: u32) -> Option<&TransitionEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Find mutably by ID.
    #[must_use]
    pub fn find_mut(&mut self, id: u32) -> Option<&mut TransitionEntry> {
        self.entries.iter_mut().find(|e| e.id == id)
    }

    /// All favourited entries.
    #[must_use]
    pub fn favourites(&self) -> Vec<&TransitionEntry> {
        self.entries.iter().filter(|e| e.favourite).collect()
    }

    /// Total number of entries.
    #[must_use]
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Filter entries by kind variant name.
    #[must_use]
    pub fn wipe_entries(&self) -> Vec<&TransitionEntry> {
        self.entries
            .iter()
            .filter(|e| matches!(e.kind, TransitionKind::Wipe(_)))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// New types: TransitionType, WipeDirection, TransitionDef, TransitionLibrary2,
//            TransitionMix, TransitionWipe
// ---------------------------------------------------------------------------

/// High-level transition type for the switcher library.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionType {
    /// An instantaneous cut.
    Cut,
    /// A dissolve/mix between two sources.
    Mix,
    /// A wipe revealing the incoming source.
    Wipe,
    /// A digital video effect transition.
    DVE,
    /// A sting/animation transition.
    Sting,
}

impl TransitionType {
    /// Returns `true` for all transition types that have a configurable duration.
    ///
    /// `Cut` is instantaneous and therefore returns `false`.
    #[must_use]
    pub fn has_duration(&self) -> bool {
        !matches!(self, Self::Cut)
    }
}

/// Direction of a wipe transition.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WipeDirection {
    /// Wipe from right to left (revealing left side).
    Left,
    /// Wipe from left to right (revealing right side).
    Right,
    /// Wipe from bottom to top (revealing top side).
    Up,
    /// Wipe from top to bottom (revealing bottom side).
    Down,
    /// Wipe from bottom-right to top-left.
    TopLeft,
    /// Wipe from top-left to bottom-right.
    BottomRight,
}

impl WipeDirection {
    /// Returns the angle in degrees corresponding to this wipe direction.
    ///
    /// Angles follow screen coordinates: 0° = right, 90° = down.
    #[must_use]
    pub fn angle_degrees(&self) -> f32 {
        match self {
            Self::Right => 0.0,
            Self::Down => 90.0,
            Self::Left => 180.0,
            Self::Up => 270.0,
            Self::TopLeft => 315.0,
            Self::BottomRight => 135.0,
        }
    }
}

/// A named transition definition stored in the switcher's transition library.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TransitionDef {
    /// Unique identifier.
    pub id: u32,
    /// Human-readable name.
    pub name: String,
    /// The type of this transition.
    pub transition_type: TransitionType,
    /// Default duration in frames when not overridden.
    pub default_duration_frames: u32,
    /// Whether this transition supports a softness/feathering control.
    pub supports_softness: bool,
}

impl TransitionDef {
    /// Returns `true` if this is a `Cut` transition (instantaneous).
    #[must_use]
    pub fn is_cut(&self) -> bool {
        self.transition_type == TransitionType::Cut
    }
}

/// A library of named `TransitionDef` entries with auto-incrementing IDs.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct TransitionDefLibrary {
    /// All stored definitions.
    pub transitions: Vec<TransitionDef>,
    next_id: u32,
}

impl TransitionDefLibrary {
    /// Create an empty library.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new transition definition, assigning it the next available ID.
    pub fn register(&mut self, mut def: TransitionDef) {
        def.id = self.next_id;
        self.next_id += 1;
        self.transitions.push(def);
    }

    /// Find all definitions matching a given `TransitionType`.
    #[must_use]
    pub fn find_by_type(&self, t: &TransitionType) -> Vec<&TransitionDef> {
        self.transitions
            .iter()
            .filter(|d| &d.transition_type == t)
            .collect()
    }

    /// Return the first registered `Mix` transition, if any.
    #[must_use]
    pub fn default_mix(&self) -> Option<&TransitionDef> {
        self.transitions
            .iter()
            .find(|d| d.transition_type == TransitionType::Mix)
    }
}

/// Compute functions for a mix (dissolve) transition.
pub struct TransitionMix;

impl TransitionMix {
    /// Compute the alpha (0.0 → 1.0) for a linear mix at normalised progress `p`.
    ///
    /// `p` should be in [0.0, 1.0].
    #[must_use]
    pub fn compute_alpha(progress: f32) -> f32 {
        progress.clamp(0.0, 1.0)
    }
}

/// Compute functions for a wipe transition.
pub struct TransitionWipe;

impl TransitionWipe {
    /// Compute a binary wipe mask for a frame of `width × height` pixels.
    ///
    /// Each element is `1.0` where the incoming source is fully visible and
    /// `0.0` where the outgoing source is visible.  `progress` is in [0.0, 1.0].
    ///
    /// For `Left` / `Right` wipes the mask splits along a vertical boundary;
    /// for `Up` / `Down` along a horizontal boundary; diagonal wipes use a
    /// 45°-angle threshold.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn compute_mask(
        progress: f32,
        direction: &WipeDirection,
        width: u32,
        height: u32,
    ) -> Vec<f32> {
        let p = progress.clamp(0.0, 1.0);
        let w = width as usize;
        let h = height as usize;
        let mut mask = vec![0.0f32; w * h];

        for row in 0..h {
            for col in 0..w {
                let visible = match direction {
                    WipeDirection::Right => (col as f32) / (w as f32) < p,
                    WipeDirection::Left => (col as f32) / (w as f32) > 1.0 - p,
                    WipeDirection::Down => (row as f32) / (h as f32) < p,
                    WipeDirection::Up => (row as f32) / (h as f32) > 1.0 - p,
                    WipeDirection::BottomRight => {
                        f32::midpoint((col as f32) / (w as f32), (row as f32) / (h as f32)) < p
                    }
                    WipeDirection::TopLeft => {
                        f32::midpoint((col as f32) / (w as f32), (row as f32) / (h as f32))
                            > 2.0 * (1.0 - p)
                    }
                };
                mask[row * w + col] = if visible { 1.0 } else { 0.0 };
            }
        }
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec2_magnitude_unit() {
        let v = Vec2::right();
        assert!((v.magnitude() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec2_normalised() {
        let v = Vec2::new(3.0, 4.0);
        let n = v.normalised();
        assert!((n.magnitude() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec2_zero_normalise_no_panic() {
        let v = Vec2::new(0.0, 0.0);
        let n = v.normalised();
        assert_eq!(n.x, 0.0);
        assert_eq!(n.y, 0.0);
    }

    #[test]
    fn test_wipe_config_defaults() {
        let w = WipeConfig::new(WipeShape::Circle);
        assert_eq!(w.shape, WipeShape::Circle);
        assert!(!w.reversed);
        assert!(!w.soft_edge);
    }

    #[test]
    fn test_wipe_config_with_border() {
        let w = WipeConfig::new(WipeShape::Diamond).with_border(0.05);
        assert!((w.border_width - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_wipe_config_reversed() {
        let w = WipeConfig::new(WipeShape::HorizontalSplit).reversed();
        assert!(w.reversed);
    }

    #[test]
    fn test_push_config_directions() {
        let p = PushConfig::left(25);
        assert_eq!(p.direction.x, -1.0);
        let p2 = PushConfig::right(25);
        assert_eq!(p2.direction.x, 1.0);
        let p3 = PushConfig::up(25);
        assert_eq!(p3.direction.y, -1.0);
        let p4 = PushConfig::down(25);
        assert_eq!(p4.direction.y, 1.0);
    }

    #[test]
    fn test_dve_path_no_keyframes_returns_none() {
        let path = DvePath::new(30);
        assert!(path.interpolate(0.5).is_none());
    }

    #[test]
    fn test_dve_path_single_keyframe() {
        let mut path = DvePath::new(30);
        path.add_keyframe(DveKeyframe::new(0.0, 0.5, 0.5, 1.0, 0.0));
        let (x, y, s, r) = path.interpolate(0.5).expect("should succeed in test");
        assert!((x - 0.5).abs() < 1e-6);
        assert!((y - 0.5).abs() < 1e-6);
        assert!((s - 1.0).abs() < 1e-6);
        assert!((r - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dve_path_interpolation_midpoint() {
        let mut path = DvePath::new(30);
        path.add_keyframe(DveKeyframe::new(0.0, 0.0, 0.0, 1.0, 0.0));
        path.add_keyframe(DveKeyframe::new(1.0, 2.0, 2.0, 0.5, 90.0));
        let (x, y, s, r) = path.interpolate(0.5).expect("should succeed in test");
        assert!((x - 1.0).abs() < 1e-5);
        assert!((y - 1.0).abs() < 1e-5);
        assert!((s - 0.75).abs() < 1e-5);
        assert!((r - 45.0).abs() < 1e-5);
    }

    #[test]
    fn test_dve_path_clamp_before_start() {
        let mut path = DvePath::new(30);
        path.add_keyframe(DveKeyframe::new(0.5, 1.0, 1.0, 1.0, 0.0));
        let (x, _, _, _) = path.interpolate(0.0).expect("should succeed in test");
        assert!((x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_transition_entry_creation() {
        let entry = TransitionEntry::new(
            1,
            "Circle Wipe",
            TransitionKind::Wipe(WipeConfig::new(WipeShape::Circle)),
            25,
        );
        assert_eq!(entry.id, 1);
        assert!(!entry.favourite);
    }

    #[test]
    fn test_transition_library_add_and_find() {
        let mut lib = TransitionLibrary::new();
        let entry = TransitionEntry::new(
            1,
            "H-Split",
            TransitionKind::Wipe(WipeConfig::new(WipeShape::HorizontalSplit)),
            30,
        );
        lib.add(entry);
        assert_eq!(lib.count(), 1);
        assert!(lib.find(1).is_some());
        assert!(lib.find(99).is_none());
    }

    #[test]
    fn test_transition_library_favourites() {
        let mut lib = TransitionLibrary::new();
        let mut e1 = TransitionEntry::new(1, "Fav", TransitionKind::Push(PushConfig::left(25)), 25);
        e1.set_favourite(true);
        lib.add(e1);
        lib.add(TransitionEntry::new(
            2,
            "Not Fav",
            TransitionKind::Push(PushConfig::right(25)),
            25,
        ));
        assert_eq!(lib.favourites().len(), 1);
    }

    #[test]
    fn test_transition_library_wipe_filter() {
        let mut lib = TransitionLibrary::new();
        lib.add(TransitionEntry::new(
            1,
            "Wipe",
            TransitionKind::Wipe(WipeConfig::new(WipeShape::Box)),
            25,
        ));
        lib.add(TransitionEntry::new(
            2,
            "Push",
            TransitionKind::Push(PushConfig::left(25)),
            25,
        ));
        assert_eq!(lib.wipe_entries().len(), 1);
    }

    // --- TransitionType tests ---

    #[test]
    fn test_transition_type_cut_has_no_duration() {
        assert!(!TransitionType::Cut.has_duration());
    }

    #[test]
    fn test_transition_type_mix_has_duration() {
        assert!(TransitionType::Mix.has_duration());
    }

    #[test]
    fn test_transition_type_wipe_has_duration() {
        assert!(TransitionType::Wipe.has_duration());
    }

    #[test]
    fn test_transition_type_dve_has_duration() {
        assert!(TransitionType::DVE.has_duration());
    }

    #[test]
    fn test_transition_type_sting_has_duration() {
        assert!(TransitionType::Sting.has_duration());
    }

    // --- WipeDirection tests ---

    #[test]
    fn test_wipe_direction_right_angle() {
        assert!((WipeDirection::Right.angle_degrees() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_wipe_direction_down_angle() {
        assert!((WipeDirection::Down.angle_degrees() - 90.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_wipe_direction_left_angle() {
        assert!((WipeDirection::Left.angle_degrees() - 180.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_wipe_direction_up_angle() {
        assert!((WipeDirection::Up.angle_degrees() - 270.0).abs() < f32::EPSILON);
    }

    // --- TransitionDef / TransitionDefLibrary tests ---

    #[test]
    fn test_transition_def_is_cut() {
        let def = TransitionDef {
            id: 0,
            name: "Cut".to_string(),
            transition_type: TransitionType::Cut,
            default_duration_frames: 0,
            supports_softness: false,
        };
        assert!(def.is_cut());
    }

    #[test]
    fn test_transition_def_is_not_cut() {
        let def = TransitionDef {
            id: 0,
            name: "Dissolve".to_string(),
            transition_type: TransitionType::Mix,
            default_duration_frames: 25,
            supports_softness: true,
        };
        assert!(!def.is_cut());
    }

    #[test]
    fn test_transition_def_library_register_and_find_by_type() {
        let mut lib = TransitionDefLibrary::new();
        lib.register(TransitionDef {
            id: 0,
            name: "Dissolve".to_string(),
            transition_type: TransitionType::Mix,
            default_duration_frames: 25,
            supports_softness: true,
        });
        lib.register(TransitionDef {
            id: 0,
            name: "Hard Cut".to_string(),
            transition_type: TransitionType::Cut,
            default_duration_frames: 0,
            supports_softness: false,
        });
        let mixes = lib.find_by_type(&TransitionType::Mix);
        assert_eq!(mixes.len(), 1);
        assert_eq!(mixes[0].name, "Dissolve");
    }

    #[test]
    fn test_transition_def_library_default_mix() {
        let mut lib = TransitionDefLibrary::new();
        assert!(lib.default_mix().is_none());
        lib.register(TransitionDef {
            id: 0,
            name: "Mix".to_string(),
            transition_type: TransitionType::Mix,
            default_duration_frames: 30,
            supports_softness: false,
        });
        assert!(lib.default_mix().is_some());
    }

    // --- TransitionMix::compute_alpha ---

    #[test]
    fn test_transition_mix_alpha_at_zero() {
        assert!((TransitionMix::compute_alpha(0.0) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_transition_mix_alpha_at_one() {
        assert!((TransitionMix::compute_alpha(1.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_transition_mix_alpha_clamps() {
        assert!((TransitionMix::compute_alpha(1.5) - 1.0).abs() < f32::EPSILON);
        assert!((TransitionMix::compute_alpha(-0.5) - 0.0).abs() < f32::EPSILON);
    }

    // --- TransitionWipe::compute_mask ---

    #[test]
    fn test_transition_wipe_mask_all_zero_at_start() {
        let mask = TransitionWipe::compute_mask(0.0, &WipeDirection::Right, 4, 4);
        assert!(mask.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_transition_wipe_mask_all_one_at_end() {
        let mask = TransitionWipe::compute_mask(1.0, &WipeDirection::Right, 4, 4);
        assert!(mask.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_transition_wipe_mask_correct_length() {
        let mask = TransitionWipe::compute_mask(0.5, &WipeDirection::Down, 8, 6);
        assert_eq!(mask.len(), 8 * 6);
    }
}
