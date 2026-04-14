//! Visual transition effects for broadcast graphics.
//!
//! Provides frame-level alpha/offset computation for common broadcast transitions:
//! - **CrossDissolve**: linear or eased opacity cross-fade between two layers
//! - **Wipe**: directional reveal (left→right, right→left, top→bottom, bottom→top)
//! - **DipToColor**: fade-out to a solid colour then fade-in
//! - **Push**: outgoing frame slides out while incoming slides in
//! - **Slide**: incoming frame slides in over the static outgoing frame
//! - **Iris**: circular iris-wipe expand / contract
//! - **PageFlip**: horizontal page-turn reveal (column-level phase offset)
//!
//! Every effect exposes a single [`TransitionEffect::sample`] method that,
//! given normalised progress `t ∈ [0.0, 1.0]`, returns a [`TransitionFrame`]
//! describing how to composite the outgoing and incoming layers.
//!
//! # Example
//!
//! ```
//! use oximedia_graphics::transition_effect::{TransitionEffect, CrossDissolve, EasingCurve};
//!
//! let effect = TransitionEffect::CrossDissolve(CrossDissolve {
//!     easing: EasingCurve::EaseInOut,
//! });
//! let frame = effect.sample(0.5);
//! assert!((frame.outgoing_alpha - 0.5).abs() < 1e-4);
//! assert!((frame.incoming_alpha - 0.5).abs() < 1e-4);
//! ```

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// EasingCurve
// ─────────────────────────────────────────────────────────────────────────────

/// Easing curve applied to a transition's normalised progress value `t`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EasingCurve {
    /// Constant velocity — no easing.
    Linear,
    /// Smooth acceleration and deceleration (cosine interpolation).
    EaseInOut,
    /// Accelerates from zero.
    EaseIn,
    /// Decelerates to zero.
    EaseOut,
    /// Cubic ease in-out (`smoothstep`).
    SmoothStep,
    /// Quintic ease in-out (`smootherstep`).
    SmootherStep,
}

impl EasingCurve {
    /// Apply this easing curve to a normalised time value `t ∈ [0, 1]`.
    #[must_use]
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            EasingCurve::Linear => t,
            EasingCurve::EaseInOut => 0.5 - (PI * t).cos() * 0.5,
            EasingCurve::EaseIn => 1.0 - (PI * 0.5 * t).cos(),
            EasingCurve::EaseOut => (PI * 0.5 * t).sin(),
            EasingCurve::SmoothStep => t * t * (3.0 - 2.0 * t),
            EasingCurve::SmootherStep => t * t * t * (t * (t * 6.0 - 15.0) + 10.0),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Direction
// ─────────────────────────────────────────────────────────────────────────────

/// Axis-aligned wipe / push direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    /// Reveal left-to-right.
    LeftToRight,
    /// Reveal right-to-left.
    RightToLeft,
    /// Reveal top-to-bottom.
    TopToBottom,
    /// Reveal bottom-to-top.
    BottomToTop,
}

// ─────────────────────────────────────────────────────────────────────────────
// TransitionFrame
// ─────────────────────────────────────────────────────────────────────────────

/// Compositing parameters for a single frame of a transition.
///
/// All fields are in normalised coordinate space (`[0, 1]` over width/height).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransitionFrame {
    /// Opacity of the outgoing layer (`1.0` = fully visible).
    pub outgoing_alpha: f32,
    /// Opacity of the incoming layer (`0.0` = fully transparent).
    pub incoming_alpha: f32,
    /// Horizontal translation offset for the outgoing layer (normalised width).
    pub outgoing_offset_x: f32,
    /// Vertical translation offset for the outgoing layer (normalised height).
    pub outgoing_offset_y: f32,
    /// Horizontal translation offset for the incoming layer.
    pub incoming_offset_x: f32,
    /// Vertical translation offset for the incoming layer.
    pub incoming_offset_y: f32,
    /// Normalised wipe edge position in `[0, 1]`.  Only meaningful for wipe and
    /// iris effects; `0.0` otherwise.
    pub wipe_edge: f32,
    /// Colour to mix for dip-to-colour transitions.  `None` for all other effects.
    pub dip_color: Option<[f32; 4]>,
    /// Per-column phase offset for page-flip effect (empty for other effects).
    pub page_flip_phases: Vec<f32>,
}

impl TransitionFrame {
    fn zero() -> Self {
        Self {
            outgoing_alpha: 1.0,
            incoming_alpha: 0.0,
            outgoing_offset_x: 0.0,
            outgoing_offset_y: 0.0,
            incoming_offset_x: 0.0,
            incoming_offset_y: 0.0,
            wipe_edge: 0.0,
            dip_color: None,
            page_flip_phases: Vec::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Individual effect parameter structs
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-dissolve (opacity fade) between outgoing and incoming layers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossDissolve {
    /// Easing applied to the progress value.
    pub easing: EasingCurve,
}

/// Directional wipe revealing the incoming layer behind the outgoing layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Wipe {
    /// Wipe direction.
    pub direction: Direction,
    /// Easing applied to the wipe edge position.
    pub easing: EasingCurve,
    /// Soft-edge feather width as a fraction of frame width/height (`0.0` = hard edge).
    pub feather: f32,
}

/// Dip-to-colour: outgoing fades out to a solid colour, then incoming fades in.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DipToColor {
    /// RGBA colour to dip through (values `[0, 1]`).
    pub color: [f32; 4],
    /// Fraction of total duration spent on the fade-out phase.
    /// The remaining `1 - fade_out_fraction` is the fade-in phase.
    pub fade_out_fraction: f32,
}

/// Push: outgoing slides out while incoming slides in from the opposite edge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Push {
    /// Direction the outgoing frame moves toward.
    pub direction: Direction,
    /// Easing applied to the position.
    pub easing: EasingCurve,
}

/// Slide: incoming frame slides in over the stationary outgoing frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Slide {
    /// Direction from which the incoming frame enters.
    pub direction: Direction,
    /// Easing applied to the slide position.
    pub easing: EasingCurve,
}

/// Circular iris wipe expanding from the centre to reveal the incoming layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Iris {
    /// Centre of the iris as normalised `(x, y)` coordinates.
    pub center_x: f32,
    /// Centre Y.
    pub center_y: f32,
    /// Easing applied to the iris radius.
    pub easing: EasingCurve,
    /// Feather width as a fraction of the maximum radius.
    pub feather: f32,
}

/// Page-flip effect: the outgoing layer appears to turn like a page, revealing
/// the incoming layer.  The flip proceeds column-by-column from left to right.
///
/// The returned [`TransitionFrame::page_flip_phases`] contains one normalised
/// phase value per column in `[0, 1]` space; renderers should use it to decide
/// which source to show and the cosine fold angle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PageFlip {
    /// Number of discrete columns the renderer divides the frame into.
    pub columns: u32,
    /// Easing applied to the overall progress.
    pub easing: EasingCurve,
}

// ─────────────────────────────────────────────────────────────────────────────
// TransitionEffect — main enum
// ─────────────────────────────────────────────────────────────────────────────

/// A composable visual transition effect.
///
/// Call [`TransitionEffect::sample`] with `t ∈ [0.0, 1.0]` to obtain the
/// [`TransitionFrame`] parameters for any point during the transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TransitionEffect {
    /// Opacity cross-fade.
    CrossDissolve(CrossDissolve),
    /// Directional wipe.
    Wipe(Wipe),
    /// Dip through a solid colour.
    DipToColor(DipToColor),
    /// Simultaneous push slide.
    Push(Push),
    /// One-sided slide.
    Slide(Slide),
    /// Circular iris.
    Iris(Iris),
    /// Page-flip.
    PageFlip(PageFlip),
}

impl TransitionEffect {
    /// Sample the transition at normalised progress `t ∈ [0.0, 1.0]`.
    ///
    /// `t = 0.0` is the very start (fully outgoing) and `t = 1.0` is the end
    /// (fully incoming).
    #[must_use]
    pub fn sample(&self, t: f32) -> TransitionFrame {
        let t = t.clamp(0.0, 1.0);
        match self {
            TransitionEffect::CrossDissolve(p) => sample_cross_dissolve(p, t),
            TransitionEffect::Wipe(p) => sample_wipe(p, t),
            TransitionEffect::DipToColor(p) => sample_dip_to_color(p, t),
            TransitionEffect::Push(p) => sample_push(p, t),
            TransitionEffect::Slide(p) => sample_slide(p, t),
            TransitionEffect::Iris(p) => sample_iris(p, t),
            TransitionEffect::PageFlip(p) => sample_page_flip(p, t),
        }
    }

    /// Duration hint in seconds for the effect at a given frame rate (informational only).
    ///
    /// The actual transition timing is controlled by the caller.
    #[must_use]
    pub fn default_duration_secs(&self) -> f32 {
        match self {
            TransitionEffect::CrossDissolve(_) => 1.0,
            TransitionEffect::Wipe(_) => 0.75,
            TransitionEffect::DipToColor(_) => 1.5,
            TransitionEffect::Push(_) => 0.5,
            TransitionEffect::Slide(_) => 0.5,
            TransitionEffect::Iris(_) => 1.0,
            TransitionEffect::PageFlip(_) => 1.2,
        }
    }

    /// Total number of frames for `fps` frames per second and default duration.
    #[must_use]
    pub fn default_frame_count(&self, fps: f32) -> u32 {
        ((self.default_duration_secs() * fps).ceil() as u32).max(1)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sampling implementations
// ─────────────────────────────────────────────────────────────────────────────

fn sample_cross_dissolve(p: &CrossDissolve, t: f32) -> TransitionFrame {
    let et = p.easing.apply(t);
    TransitionFrame {
        outgoing_alpha: 1.0 - et,
        incoming_alpha: et,
        ..TransitionFrame::zero()
    }
}

fn sample_wipe(p: &Wipe, t: f32) -> TransitionFrame {
    let et = p.easing.apply(t);
    TransitionFrame {
        outgoing_alpha: 1.0,
        incoming_alpha: 1.0,
        wipe_edge: et,
        ..TransitionFrame::zero()
    }
}

fn sample_dip_to_color(p: &DipToColor, t: f32) -> TransitionFrame {
    // Phase 1: outgoing fades to colour
    // Phase 2: colour fades to incoming
    let fo = p.fade_out_fraction.clamp(0.01, 0.99);
    let (outgoing_alpha, incoming_alpha, dip_alpha) = if t <= fo {
        let local = t / fo;
        let a = 1.0 - local;
        (a, 0.0, local)
    } else {
        let local = (t - fo) / (1.0 - fo);
        let a = local;
        (0.0, a, 1.0 - local)
    };
    // Encode dip colour with its own alpha blended in
    let mut color = p.color;
    color[3] *= dip_alpha;
    TransitionFrame {
        outgoing_alpha,
        incoming_alpha,
        dip_color: Some(color),
        ..TransitionFrame::zero()
    }
}

fn direction_offset(direction: Direction, progress: f32) -> (f32, f32) {
    match direction {
        Direction::LeftToRight => (-progress, 0.0),
        Direction::RightToLeft => (progress, 0.0),
        Direction::TopToBottom => (0.0, -progress),
        Direction::BottomToTop => (0.0, progress),
    }
}

fn incoming_start_offset(direction: Direction) -> (f32, f32) {
    match direction {
        Direction::LeftToRight => (1.0, 0.0),
        Direction::RightToLeft => (-1.0, 0.0),
        Direction::TopToBottom => (0.0, 1.0),
        Direction::BottomToTop => (0.0, -1.0),
    }
}

fn sample_push(p: &Push, t: f32) -> TransitionFrame {
    let et = p.easing.apply(t);
    let (ox, oy) = direction_offset(p.direction, et);
    let (isx, isy) = incoming_start_offset(p.direction);
    let ix = isx - isx * et;
    let iy = isy - isy * et;
    TransitionFrame {
        outgoing_alpha: 1.0,
        incoming_alpha: 1.0,
        outgoing_offset_x: ox,
        outgoing_offset_y: oy,
        incoming_offset_x: ix,
        incoming_offset_y: iy,
        ..TransitionFrame::zero()
    }
}

fn sample_slide(p: &Slide, t: f32) -> TransitionFrame {
    let et = p.easing.apply(t);
    let (isx, isy) = incoming_start_offset(p.direction);
    let ix = isx - isx * et;
    let iy = isy - isy * et;
    TransitionFrame {
        outgoing_alpha: 1.0,
        incoming_alpha: 1.0,
        incoming_offset_x: ix,
        incoming_offset_y: iy,
        ..TransitionFrame::zero()
    }
}

fn sample_iris(p: &Iris, t: f32) -> TransitionFrame {
    let et = p.easing.apply(t);
    TransitionFrame {
        outgoing_alpha: 1.0,
        incoming_alpha: 1.0,
        // wipe_edge encodes the iris radius in [0, 1] normalised to the maximum radius
        wipe_edge: et,
        ..TransitionFrame::zero()
    }
}

fn sample_page_flip(p: &PageFlip, t: f32) -> TransitionFrame {
    let et = p.easing.apply(t);
    let cols = p.columns.max(1) as usize;
    // Each column i starts its flip at t_start = i/(cols+1) and ends at t_end = (i+1)/(cols+1).
    // phase[i] is local progress of that column [0,1].
    let phases: Vec<f32> = (0..cols)
        .map(|i| {
            let t_start = i as f32 / cols as f32;
            let t_end = (i + 1) as f32 / cols as f32;
            if et <= t_start {
                0.0_f32
            } else if et >= t_end {
                1.0_f32
            } else {
                (et - t_start) / (t_end - t_start)
            }
        })
        .collect();
    TransitionFrame {
        outgoing_alpha: 1.0,
        incoming_alpha: 1.0,
        page_flip_phases: phases,
        ..TransitionFrame::zero()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TransitionSequencer — time-managed effect runner
// ─────────────────────────────────────────────────────────────────────────────

/// Manages a sequence of transitions with per-effect durations.
///
/// Advances time through [`TransitionSequencer::advance`] and queries the
/// current compositing parameters via [`TransitionSequencer::current_frame`].
#[derive(Debug, Clone)]
pub struct TransitionSequencer {
    effects: Vec<(TransitionEffect, f32)>, // (effect, duration_secs)
    current_index: usize,
    elapsed: f32,
    finished: bool,
}

impl TransitionSequencer {
    /// Create a new sequencer from a list of `(effect, duration_seconds)` pairs.
    ///
    /// Returns an error if `effects` is empty or any duration is non-positive.
    pub fn new(effects: Vec<(TransitionEffect, f32)>) -> Result<Self, SequencerError> {
        if effects.is_empty() {
            return Err(SequencerError::EmptySequence);
        }
        for (i, (_, dur)) in effects.iter().enumerate() {
            if *dur <= 0.0 {
                return Err(SequencerError::InvalidDuration { index: i });
            }
        }
        Ok(Self {
            effects,
            current_index: 0,
            elapsed: 0.0,
            finished: false,
        })
    }

    /// Advance the sequencer by `delta` seconds.
    pub fn advance(&mut self, delta_secs: f32) {
        if self.finished {
            return;
        }
        self.elapsed += delta_secs.max(0.0);
        // Consume completed effects
        loop {
            if self.current_index >= self.effects.len() {
                self.finished = true;
                break;
            }
            let duration = self.effects[self.current_index].1;
            if self.elapsed >= duration {
                self.elapsed -= duration;
                self.current_index += 1;
            } else {
                break;
            }
        }
    }

    /// Returns the [`TransitionFrame`] for the current moment in the sequence.
    #[must_use]
    pub fn current_frame(&self) -> TransitionFrame {
        if self.finished || self.current_index >= self.effects.len() {
            // All transitions complete — show incoming at 100 %
            return TransitionFrame {
                outgoing_alpha: 0.0,
                incoming_alpha: 1.0,
                ..TransitionFrame::zero()
            };
        }
        let (effect, duration) = &self.effects[self.current_index];
        let t = (self.elapsed / duration).clamp(0.0, 1.0);
        effect.sample(t)
    }

    /// Returns `true` when all transitions in the sequence are complete.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Total duration of all effects in seconds.
    #[must_use]
    pub fn total_duration(&self) -> f32 {
        self.effects.iter().map(|(_, d)| *d).sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SequencerError
// ─────────────────────────────────────────────────────────────────────────────

/// Error type for [`TransitionSequencer`] construction.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SequencerError {
    /// The effect list was empty.
    #[error("transition sequence must contain at least one effect")]
    EmptySequence,
    /// A duration was zero or negative.
    #[error("effect at index {index} has a non-positive duration")]
    InvalidDuration {
        /// Index of the invalid effect.
        index: usize,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn easing_linear_identity() {
        assert!((EasingCurve::Linear.apply(0.0) - 0.0).abs() < 1e-6);
        assert!((EasingCurve::Linear.apply(0.5) - 0.5).abs() < 1e-6);
        assert!((EasingCurve::Linear.apply(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn easing_clamps_out_of_range() {
        assert!((EasingCurve::SmoothStep.apply(-0.5) - 0.0).abs() < 1e-6);
        assert!((EasingCurve::SmoothStep.apply(1.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cross_dissolve_midpoint() {
        let effect = TransitionEffect::CrossDissolve(CrossDissolve {
            easing: EasingCurve::Linear,
        });
        let frame = effect.sample(0.5);
        assert!((frame.outgoing_alpha - 0.5).abs() < 1e-5);
        assert!((frame.incoming_alpha - 0.5).abs() < 1e-5);
    }

    #[test]
    fn cross_dissolve_boundaries() {
        let effect = TransitionEffect::CrossDissolve(CrossDissolve {
            easing: EasingCurve::Linear,
        });
        let start = effect.sample(0.0);
        assert!((start.outgoing_alpha - 1.0).abs() < 1e-5);
        assert!((start.incoming_alpha - 0.0).abs() < 1e-5);
        let end = effect.sample(1.0);
        assert!((end.outgoing_alpha - 0.0).abs() < 1e-5);
        assert!((end.incoming_alpha - 1.0).abs() < 1e-5);
    }

    #[test]
    fn wipe_edge_progresses() {
        let effect = TransitionEffect::Wipe(Wipe {
            direction: Direction::LeftToRight,
            easing: EasingCurve::Linear,
            feather: 0.0,
        });
        let f0 = effect.sample(0.0);
        let f1 = effect.sample(1.0);
        assert!(f0.wipe_edge < f1.wipe_edge);
        assert!((f1.wipe_edge - 1.0).abs() < 1e-5);
    }

    #[test]
    fn push_offsets_are_opposite() {
        let effect = TransitionEffect::Push(Push {
            direction: Direction::LeftToRight,
            easing: EasingCurve::Linear,
        });
        let frame = effect.sample(0.5);
        // outgoing moves left (-x), incoming arrives from right (+x approaching 0)
        assert!(frame.outgoing_offset_x < 0.0);
        assert!(frame.incoming_offset_x > 0.0);
    }

    #[test]
    fn slide_only_moves_incoming() {
        let effect = TransitionEffect::Slide(Slide {
            direction: Direction::TopToBottom,
            easing: EasingCurve::Linear,
        });
        let frame = effect.sample(0.5);
        assert!((frame.outgoing_offset_x).abs() < 1e-6);
        assert!((frame.outgoing_offset_y).abs() < 1e-6);
        assert!(frame.incoming_offset_y > 0.0);
    }

    #[test]
    fn dip_to_color_midpoint_has_color() {
        let effect = TransitionEffect::DipToColor(DipToColor {
            color: [0.0, 0.0, 0.0, 1.0],
            fade_out_fraction: 0.5,
        });
        let frame = effect.sample(0.5);
        assert!(frame.dip_color.is_some());
    }

    #[test]
    fn page_flip_phases_length_matches_columns() {
        let cols = 20_u32;
        let effect = TransitionEffect::PageFlip(PageFlip {
            columns: cols,
            easing: EasingCurve::Linear,
        });
        let frame = effect.sample(0.5);
        assert_eq!(frame.page_flip_phases.len(), cols as usize);
    }

    #[test]
    fn sequencer_advances_through_effects() {
        let effects = vec![
            (
                TransitionEffect::CrossDissolve(CrossDissolve {
                    easing: EasingCurve::Linear,
                }),
                1.0_f32,
            ),
            (
                TransitionEffect::Wipe(Wipe {
                    direction: Direction::LeftToRight,
                    easing: EasingCurve::Linear,
                    feather: 0.0,
                }),
                1.0_f32,
            ),
        ];
        let mut seq = TransitionSequencer::new(effects).expect("valid sequence");
        assert!(!seq.is_finished());
        assert!((seq.total_duration() - 2.0).abs() < 1e-6);
        seq.advance(1.5);
        // Should now be in second effect at t=0.5
        let frame = seq.current_frame();
        assert!((frame.wipe_edge - 0.5).abs() < 1e-4);
        seq.advance(1.0);
        assert!(seq.is_finished());
    }

    #[test]
    fn sequencer_empty_returns_error() {
        let result = TransitionSequencer::new(vec![]);
        assert!(matches!(result, Err(SequencerError::EmptySequence)));
    }

    #[test]
    fn sequencer_invalid_duration_returns_error() {
        let effects = vec![(
            TransitionEffect::CrossDissolve(CrossDissolve {
                easing: EasingCurve::Linear,
            }),
            -1.0_f32,
        )];
        let result = TransitionSequencer::new(effects);
        assert!(matches!(
            result,
            Err(SequencerError::InvalidDuration { index: 0 })
        ));
    }

    #[test]
    fn iris_radius_grows() {
        let effect = TransitionEffect::Iris(Iris {
            center_x: 0.5,
            center_y: 0.5,
            easing: EasingCurve::Linear,
            feather: 0.05,
        });
        let r0 = effect.sample(0.0).wipe_edge;
        let r1 = effect.sample(1.0).wipe_edge;
        assert!(r1 > r0);
    }

    #[test]
    fn default_frame_count_positive() {
        let effect = TransitionEffect::CrossDissolve(CrossDissolve {
            easing: EasingCurve::Linear,
        });
        assert!(effect.default_frame_count(60.0) > 0);
    }
}
