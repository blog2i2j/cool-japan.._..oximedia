#![allow(dead_code)]
//! Priority-based camera angle selection with weighted scoring.
//!
//! Assigns dynamic priorities to camera angles based on configurable criteria
//! (composition quality, audio activity, motion, face detection) and selects
//! the highest-priority angle for automatic switching.

use std::collections::HashMap;

/// A single scoring criterion used to evaluate an angle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Criterion {
    /// How well the shot is composed (rule of thirds, framing).
    CompositionQuality,
    /// Level of audio activity from this angle's microphone.
    AudioActivity,
    /// Amount of motion detected in the frame.
    MotionLevel,
    /// Whether faces are detected and well-framed.
    FaceDetection,
    /// Speaker detection (who is currently talking).
    SpeakerDetection,
    /// How long since this angle was last on-air (variety bonus).
    ShotVariety,
    /// Manual operator boost.
    OperatorBoost,
}

/// Weight and current value for a single criterion.
#[derive(Debug, Clone, Copy)]
pub struct CriterionScore {
    /// Normalised value in [0.0, 1.0].
    pub value: f64,
    /// Weight multiplier for this criterion.
    pub weight: f64,
}

impl CriterionScore {
    /// Create a new criterion score.
    #[must_use]
    pub fn new(value: f64, weight: f64) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
            weight: weight.max(0.0),
        }
    }

    /// Weighted contribution of this score.
    #[must_use]
    pub fn weighted(&self) -> f64 {
        self.value * self.weight
    }
}

/// Per-angle priority record holding all criterion scores.
#[derive(Debug, Clone)]
pub struct AnglePriority {
    /// Camera angle identifier.
    pub angle_id: usize,
    /// Per-criterion scores.
    scores: HashMap<Criterion, CriterionScore>,
    /// Manual override priority (if set, bypasses normal scoring).
    manual_override: Option<f64>,
    /// Whether this angle is locked out (excluded from selection).
    pub locked_out: bool,
}

impl AnglePriority {
    /// Create a new angle priority record with no scores.
    #[must_use]
    pub fn new(angle_id: usize) -> Self {
        Self {
            angle_id,
            scores: HashMap::new(),
            manual_override: None,
            locked_out: false,
        }
    }

    /// Set a criterion score.
    pub fn set_score(&mut self, criterion: Criterion, value: f64, weight: f64) {
        self.scores
            .insert(criterion, CriterionScore::new(value, weight));
    }

    /// Get the score for a criterion.
    #[must_use]
    pub fn get_score(&self, criterion: Criterion) -> Option<&CriterionScore> {
        self.scores.get(&criterion)
    }

    /// Set a manual override priority.
    pub fn set_manual_override(&mut self, priority: f64) {
        self.manual_override = Some(priority);
    }

    /// Clear the manual override.
    pub fn clear_manual_override(&mut self) {
        self.manual_override = None;
    }

    /// Compute the total priority for this angle.
    ///
    /// If a manual override is set, it is returned directly.
    /// If the angle is locked out, 0.0 is returned.
    /// Otherwise the weighted sum of all criterion scores is returned.
    pub fn total_priority(&self) -> f64 {
        if self.locked_out {
            return 0.0;
        }
        if let Some(ovr) = self.manual_override {
            return ovr;
        }
        self.scores.values().map(CriterionScore::weighted).sum()
    }

    /// Number of criteria set.
    #[must_use]
    pub fn criteria_count(&self) -> usize {
        self.scores.len()
    }
}

/// Minimum dwell time before a camera switch is allowed.
#[derive(Debug, Clone, Copy)]
pub struct DwellConstraint {
    /// Minimum frames an angle must stay on-air before switching away.
    pub min_frames: u64,
    /// Maximum frames an angle can stay on-air before a mandatory switch.
    pub max_frames: Option<u64>,
}

impl Default for DwellConstraint {
    fn default() -> Self {
        Self {
            min_frames: 75, // 3 seconds at 25fps
            max_frames: None,
        }
    }
}

/// Result of the priority selection process.
#[derive(Debug, Clone)]
pub struct SelectionResult {
    /// The chosen angle.
    pub angle_id: usize,
    /// The computed priority of the chosen angle.
    pub priority: f64,
    /// Priorities of all evaluated angles (sorted descending).
    pub all_priorities: Vec<(usize, f64)>,
    /// Whether the selection was constrained by dwell time.
    pub dwell_constrained: bool,
}

/// Priority-based angle selector that evaluates all angles and picks the best.
#[derive(Debug)]
pub struct AnglePrioritySelector {
    /// Per-angle priority records.
    angles: HashMap<usize, AnglePriority>,
    /// Dwell constraint settings.
    pub dwell: DwellConstraint,
    /// Currently on-air angle.
    current_angle: Option<usize>,
    /// Number of frames the current angle has been on-air.
    frames_on_air: u64,
    /// Hysteresis threshold — new angle must beat current by this margin.
    pub hysteresis: f64,
}

impl AnglePrioritySelector {
    /// Create a new selector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            angles: HashMap::new(),
            dwell: DwellConstraint::default(),
            current_angle: None,
            frames_on_air: 0,
            hysteresis: 0.1,
        }
    }

    /// Register a camera angle.
    pub fn register_angle(&mut self, angle_id: usize) {
        self.angles
            .entry(angle_id)
            .or_insert_with(|| AnglePriority::new(angle_id));
    }

    /// Get a mutable reference to an angle's priority record.
    pub fn angle_mut(&mut self, angle_id: usize) -> Option<&mut AnglePriority> {
        self.angles.get_mut(&angle_id)
    }

    /// Get a reference to an angle's priority record.
    #[must_use]
    pub fn angle(&self, angle_id: usize) -> Option<&AnglePriority> {
        self.angles.get(&angle_id)
    }

    /// Number of registered angles.
    #[must_use]
    pub fn angle_count(&self) -> usize {
        self.angles.len()
    }

    /// Evaluate all angles and select the best one.
    ///
    /// Returns `None` if no angles are registered.
    pub fn select(&mut self) -> Option<SelectionResult> {
        if self.angles.is_empty() {
            return None;
        }

        let mut priorities: Vec<(usize, f64)> = self
            .angles
            .values()
            .map(|a| (a.angle_id, a.total_priority()))
            .collect();
        priorities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best_id = priorities[0].0;
        let best_priority = priorities[0].1;

        // Apply dwell and hysteresis constraints
        let dwell_constrained;
        let selected_id;

        if let Some(current) = self.current_angle {
            if self.frames_on_air < self.dwell.min_frames {
                // Still within minimum dwell — keep current angle
                dwell_constrained = true;
                selected_id = current;
            } else if let Some(max) = self.dwell.max_frames {
                if self.frames_on_air >= max {
                    // Exceeded max dwell — force switch
                    dwell_constrained = false;
                    selected_id = best_id;
                    self.frames_on_air = 0;
                    self.current_angle = Some(best_id);
                } else {
                    dwell_constrained = false;
                    selected_id =
                        self.apply_hysteresis(current, best_id, best_priority, &priorities);
                }
            } else {
                dwell_constrained = false;
                selected_id = self.apply_hysteresis(current, best_id, best_priority, &priorities);
            }
        } else {
            dwell_constrained = false;
            selected_id = best_id;
            self.current_angle = Some(best_id);
        }

        self.frames_on_air += 1;

        let selected_priority = priorities
            .iter()
            .find(|(id, _)| *id == selected_id)
            .map_or(0.0, |(_, p)| *p);

        Some(SelectionResult {
            angle_id: selected_id,
            priority: selected_priority,
            all_priorities: priorities,
            dwell_constrained,
        })
    }

    /// Apply hysteresis: only switch if the best angle beats the current by more
    /// than the hysteresis margin.
    fn apply_hysteresis(
        &mut self,
        current: usize,
        best_id: usize,
        best_priority: f64,
        priorities: &[(usize, f64)],
    ) -> usize {
        let current_priority = priorities
            .iter()
            .find(|(id, _)| *id == current)
            .map_or(0.0, |(_, p)| *p);

        if best_id != current && (best_priority - current_priority) > self.hysteresis {
            self.current_angle = Some(best_id);
            self.frames_on_air = 0;
            best_id
        } else {
            current
        }
    }

    /// Get the current on-air angle.
    #[must_use]
    pub fn current_angle(&self) -> Option<usize> {
        self.current_angle
    }

    /// Get the number of frames the current angle has been on-air.
    #[must_use]
    pub fn frames_on_air(&self) -> u64 {
        self.frames_on_air
    }

    /// Force-set the on-air angle (manual override).
    pub fn force_angle(&mut self, angle_id: usize) {
        self.current_angle = Some(angle_id);
        self.frames_on_air = 0;
    }
}

impl Default for AnglePrioritySelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_criterion_score_clamp() {
        let s = CriterionScore::new(1.5, 2.0);
        assert!((s.value - 1.0).abs() < f64::EPSILON);
        assert!((s.weight - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_criterion_score_weighted() {
        let s = CriterionScore::new(0.8, 1.5);
        let expected = 0.8 * 1.5;
        assert!((s.weighted() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_angle_priority_total() {
        let mut ap = AnglePriority::new(0);
        ap.set_score(Criterion::CompositionQuality, 0.9, 1.0);
        ap.set_score(Criterion::AudioActivity, 0.5, 2.0);
        // 0.9*1.0 + 0.5*2.0 = 1.9
        assert!((ap.total_priority() - 1.9).abs() < 1e-10);
    }

    #[test]
    fn test_angle_priority_locked_out() {
        let mut ap = AnglePriority::new(0);
        ap.set_score(Criterion::CompositionQuality, 0.9, 1.0);
        ap.locked_out = true;
        assert!((ap.total_priority()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_angle_priority_manual_override() {
        let mut ap = AnglePriority::new(0);
        ap.set_score(Criterion::CompositionQuality, 0.9, 1.0);
        ap.set_manual_override(5.0);
        assert!((ap.total_priority() - 5.0).abs() < f64::EPSILON);
        ap.clear_manual_override();
        assert!((ap.total_priority() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_selector_empty_returns_none() {
        let mut sel = AnglePrioritySelector::new();
        assert!(sel.select().is_none());
    }

    #[test]
    fn test_selector_single_angle() {
        let mut sel = AnglePrioritySelector::new();
        sel.register_angle(0);
        sel.angle_mut(0)
            .expect("multicam test operation should succeed")
            .set_score(Criterion::CompositionQuality, 0.8, 1.0);
        let result = sel
            .select()
            .expect("multicam test operation should succeed");
        assert_eq!(result.angle_id, 0);
    }

    #[test]
    fn test_selector_picks_highest() {
        let mut sel = AnglePrioritySelector::new();
        sel.hysteresis = 0.0;
        sel.dwell.min_frames = 0;
        sel.register_angle(0);
        sel.register_angle(1);
        sel.angle_mut(0)
            .expect("multicam test operation should succeed")
            .set_score(Criterion::CompositionQuality, 0.3, 1.0);
        sel.angle_mut(1)
            .expect("multicam test operation should succeed")
            .set_score(Criterion::CompositionQuality, 0.9, 1.0);
        let result = sel
            .select()
            .expect("multicam test operation should succeed");
        assert_eq!(result.angle_id, 1);
    }

    #[test]
    fn test_selector_dwell_constraint() {
        let mut sel = AnglePrioritySelector::new();
        sel.dwell.min_frames = 5;
        sel.hysteresis = 0.0;
        sel.register_angle(0);
        sel.register_angle(1);
        sel.angle_mut(0)
            .expect("multicam test operation should succeed")
            .set_score(Criterion::AudioActivity, 0.5, 1.0);
        sel.angle_mut(1)
            .expect("multicam test operation should succeed")
            .set_score(Criterion::AudioActivity, 1.0, 1.0);

        // First select picks angle 1 (best)
        let r = sel
            .select()
            .expect("multicam test operation should succeed");
        assert_eq!(r.angle_id, 1);

        // Now make angle 0 the best
        sel.angle_mut(0)
            .expect("multicam test operation should succeed")
            .set_score(Criterion::AudioActivity, 1.0, 1.0);
        sel.angle_mut(1)
            .expect("multicam test operation should succeed")
            .set_score(Criterion::AudioActivity, 0.1, 1.0);

        // Should stay on angle 1 due to dwell constraint
        let r = sel
            .select()
            .expect("multicam test operation should succeed");
        assert!(r.dwell_constrained);
        assert_eq!(r.angle_id, 1);
    }

    #[test]
    fn test_selector_hysteresis() {
        let mut sel = AnglePrioritySelector::new();
        sel.dwell.min_frames = 0;
        sel.hysteresis = 0.5;
        sel.register_angle(0);
        sel.register_angle(1);
        sel.angle_mut(0)
            .expect("multicam test operation should succeed")
            .set_score(Criterion::MotionLevel, 0.7, 1.0);
        sel.angle_mut(1)
            .expect("multicam test operation should succeed")
            .set_score(Criterion::MotionLevel, 0.8, 1.0);

        // First select: picks angle 1
        let _ = sel
            .select()
            .expect("multicam test operation should succeed");
        assert_eq!(sel.current_angle(), Some(1));

        // Angle 0 only marginally better — hysteresis prevents switch
        sel.angle_mut(0)
            .expect("multicam test operation should succeed")
            .set_score(Criterion::MotionLevel, 0.85, 1.0);
        let _ = sel
            .select()
            .expect("multicam test operation should succeed");
        assert_eq!(sel.current_angle(), Some(1));
    }

    #[test]
    fn test_selector_force_angle() {
        let mut sel = AnglePrioritySelector::new();
        sel.register_angle(0);
        sel.register_angle(1);
        sel.force_angle(0);
        assert_eq!(sel.current_angle(), Some(0));
        assert_eq!(sel.frames_on_air(), 0);
    }

    #[test]
    fn test_criteria_count() {
        let mut ap = AnglePriority::new(0);
        assert_eq!(ap.criteria_count(), 0);
        ap.set_score(Criterion::FaceDetection, 1.0, 1.0);
        ap.set_score(Criterion::SpeakerDetection, 0.5, 1.0);
        assert_eq!(ap.criteria_count(), 2);
    }

    #[test]
    fn test_dwell_constraint_default() {
        let d = DwellConstraint::default();
        assert_eq!(d.min_frames, 75);
        assert!(d.max_frames.is_none());
    }
}
