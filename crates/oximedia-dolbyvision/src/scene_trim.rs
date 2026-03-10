//! Dolby Vision scene-level trim metadata
//!
//! Scene trim metadata describes per-scene tone mapping parameters for adapting
//! content to displays with different peak luminances.

/// Trim target parameters for a specific display luminance
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct TrimTarget {
    /// Target display peak luminance in nits
    pub target_nits: f32,
    /// Target mid-contrast (scene-level)
    pub target_mid_contrast: f32,
    /// Trim slope (0.0–2.0, 1.0 = neutral)
    pub trim_slope: f32,
    /// Trim offset (-1.0–1.0, 0.0 = neutral)
    pub trim_offset: f32,
    /// Trim power (0.0–2.0, 1.0 = neutral)
    pub trim_power: f32,
}

impl TrimTarget {
    /// Create a trim target for a common display luminance level.
    ///
    /// Supported display levels: 100, 400, 1000, 4000 nits.
    /// Other values produce sensible defaults.
    #[must_use]
    pub fn for_display(display_nits: f32) -> Self {
        // Rule-based trim values for common display targets
        if (display_nits - 100.0).abs() < 1.0 {
            Self {
                target_nits: 100.0,
                target_mid_contrast: 0.6,
                trim_slope: 0.85,
                trim_offset: -0.05,
                trim_power: 1.1,
            }
        } else if (display_nits - 400.0).abs() < 1.0 {
            Self {
                target_nits: 400.0,
                target_mid_contrast: 0.75,
                trim_slope: 0.95,
                trim_offset: -0.02,
                trim_power: 1.05,
            }
        } else if (display_nits - 1000.0).abs() < 1.0 {
            Self {
                target_nits: 1000.0,
                target_mid_contrast: 1.0,
                trim_slope: 1.0,
                trim_offset: 0.0,
                trim_power: 1.0,
            }
        } else if (display_nits - 4000.0).abs() < 1.0 {
            Self {
                target_nits: 4000.0,
                target_mid_contrast: 1.1,
                trim_slope: 1.05,
                trim_offset: 0.02,
                trim_power: 0.98,
            }
        } else {
            // Generic: linearly interpolate slope between 100-nit and 4000-nit anchors
            let t = (display_nits.ln() - 100_f32.ln()) / (4000_f32.ln() - 100_f32.ln());
            let t = t.clamp(0.0, 1.0);
            Self {
                target_nits: display_nits,
                target_mid_contrast: 0.6 + t * 0.5,
                trim_slope: 0.85 + t * 0.2,
                trim_offset: -0.05 + t * 0.07,
                trim_power: 1.1 - t * 0.12,
            }
        }
    }
}

/// Scene-level trim metadata covering a range of frames
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SceneTrimMetadata {
    /// Unique scene identifier
    pub scene_id: u32,
    /// First frame index of the scene (inclusive)
    pub first_frame: u64,
    /// Last frame index of the scene (inclusive)
    pub last_frame: u64,
    /// Trim targets for various display luminances
    pub trims: Vec<TrimTarget>,
}

impl SceneTrimMetadata {
    /// Create a new `SceneTrimMetadata`
    #[must_use]
    pub fn new(scene_id: u32, first_frame: u64, last_frame: u64) -> Self {
        Self {
            scene_id,
            first_frame,
            last_frame,
            trims: Vec::new(),
        }
    }

    /// Add a trim target to this scene
    pub fn add_trim(&mut self, trim: TrimTarget) {
        self.trims.push(trim);
    }

    /// Interpolate trim values for the given target luminance.
    ///
    /// Uses linear interpolation between the two nearest known trim targets.
    /// Returns the nearest target if interpolation is not possible.
    #[must_use]
    pub fn interpolate_trim(&self, target_nits: f32) -> TrimTarget {
        if self.trims.is_empty() {
            return TrimTarget::for_display(target_nits);
        }

        if self.trims.len() == 1 {
            return self.trims[0].clone();
        }

        // Find the two surrounding targets
        let mut sorted = self.trims.clone();
        sorted.sort_by(|a, b| {
            a.target_nits
                .partial_cmp(&b.target_nits)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Below the lowest target
        if target_nits <= sorted[0].target_nits {
            return sorted[0].clone();
        }
        // Above the highest target
        let last = sorted
            .last()
            .expect("sorted is non-empty: element 0 accessed above");
        if target_nits >= last.target_nits {
            return last.clone();
        }

        // Find the pair to interpolate between
        for i in 0..sorted.len() - 1 {
            let lo = &sorted[i];
            let hi = &sorted[i + 1];
            if target_nits >= lo.target_nits && target_nits <= hi.target_nits {
                let t = (target_nits - lo.target_nits) / (hi.target_nits - lo.target_nits);
                return TrimTarget {
                    target_nits,
                    target_mid_contrast: lo.target_mid_contrast
                        + t * (hi.target_mid_contrast - lo.target_mid_contrast),
                    trim_slope: lo.trim_slope + t * (hi.trim_slope - lo.trim_slope),
                    trim_offset: lo.trim_offset + t * (hi.trim_offset - lo.trim_offset),
                    trim_power: lo.trim_power + t * (hi.trim_power - lo.trim_power),
                };
            }
        }

        // Fallback
        sorted[0].clone()
    }
}

/// Library of scene trim metadata indexed by scene/frame
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct TrimMetadataLibrary {
    scenes: Vec<SceneTrimMetadata>,
}

impl TrimMetadataLibrary {
    /// Create an empty library
    #[must_use]
    pub fn new() -> Self {
        Self { scenes: Vec::new() }
    }

    /// Add scene trim metadata to the library
    pub fn add_scene(&mut self, metadata: SceneTrimMetadata) {
        self.scenes.push(metadata);
    }

    /// Get the interpolated trim for the given frame index and display luminance.
    ///
    /// Returns `None` if no scene covers the requested frame.
    #[must_use]
    pub fn get_trim_at_frame(&self, frame: u64, display_nits: f32) -> Option<TrimTarget> {
        let scene = self
            .scenes
            .iter()
            .find(|s| frame >= s.first_frame && frame <= s.last_frame)?;
        Some(scene.interpolate_trim(display_nits))
    }

    /// Return a reference to the internal scene list (useful for validation)
    #[must_use]
    pub fn scenes(&self) -> &[SceneTrimMetadata] {
        &self.scenes
    }
}

/// Validates trim metadata continuity across scenes
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct TrimValidator;

impl TrimValidator {
    /// Check for frame gaps or overlaps in the library.
    ///
    /// Returns a list of error/warning messages (empty = valid).
    #[must_use]
    pub fn check_continuity(library: &TrimMetadataLibrary) -> Vec<String> {
        let mut messages = Vec::new();
        let mut sorted = library.scenes().to_vec();
        sorted.sort_by_key(|s| s.first_frame);

        for i in 0..sorted.len() {
            let scene = &sorted[i];

            // Check that first_frame <= last_frame
            if scene.first_frame > scene.last_frame {
                messages.push(format!(
                    "Scene {}: first_frame ({}) > last_frame ({})",
                    scene.scene_id, scene.first_frame, scene.last_frame
                ));
            }

            if i + 1 < sorted.len() {
                let next = &sorted[i + 1];
                let expected_next = scene.last_frame + 1;

                if next.first_frame > expected_next {
                    messages.push(format!(
                        "Gap between scene {} (last={}) and scene {} (first={}): frames {} to {} missing",
                        scene.scene_id,
                        scene.last_frame,
                        next.scene_id,
                        next.first_frame,
                        expected_next,
                        next.first_frame - 1,
                    ));
                } else if next.first_frame < expected_next {
                    messages.push(format!(
                        "Overlap between scene {} (last={}) and scene {} (first={})",
                        scene.scene_id, scene.last_frame, next.scene_id, next.first_frame,
                    ));
                }
            }
        }

        messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim_target_for_100_nits() {
        let t = TrimTarget::for_display(100.0);
        assert!((t.target_nits - 100.0).abs() < f32::EPSILON);
        assert!(t.trim_slope < 1.0);
    }

    #[test]
    fn test_trim_target_for_1000_nits() {
        let t = TrimTarget::for_display(1000.0);
        assert!((t.trim_slope - 1.0).abs() < f32::EPSILON);
        assert!((t.trim_offset).abs() < f32::EPSILON);
    }

    #[test]
    fn test_trim_target_for_4000_nits() {
        let t = TrimTarget::for_display(4000.0);
        assert!(t.trim_slope > 1.0);
    }

    #[test]
    fn test_trim_target_for_400_nits() {
        let t = TrimTarget::for_display(400.0);
        assert!((t.target_nits - 400.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_scene_trim_metadata_creation() {
        let mut scene = SceneTrimMetadata::new(1, 0, 99);
        scene.add_trim(TrimTarget::for_display(1000.0));
        assert_eq!(scene.scene_id, 1);
        assert_eq!(scene.trims.len(), 1);
    }

    #[test]
    fn test_interpolate_trim_single() {
        let mut scene = SceneTrimMetadata::new(1, 0, 99);
        scene.add_trim(TrimTarget::for_display(1000.0));
        let result = scene.interpolate_trim(500.0);
        assert!((result.trim_slope - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_interpolate_trim_between_100_1000() {
        let mut scene = SceneTrimMetadata::new(1, 0, 99);
        scene.add_trim(TrimTarget::for_display(100.0));
        scene.add_trim(TrimTarget::for_display(1000.0));

        let mid = scene.interpolate_trim(550.0);
        // Should be between 100-nit slope (0.85) and 1000-nit slope (1.0)
        assert!(
            mid.trim_slope > 0.85 && mid.trim_slope < 1.0,
            "slope={}",
            mid.trim_slope
        );
    }

    #[test]
    fn test_trim_metadata_library_add_and_get() {
        let mut lib = TrimMetadataLibrary::new();
        let mut scene = SceneTrimMetadata::new(0, 0, 23);
        scene.add_trim(TrimTarget::for_display(1000.0));
        lib.add_scene(scene);

        let result = lib.get_trim_at_frame(10, 1000.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_trim_metadata_library_miss() {
        let lib = TrimMetadataLibrary::new();
        let result = lib.get_trim_at_frame(10, 1000.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_continuity_no_gaps_no_overlaps() {
        let mut lib = TrimMetadataLibrary::new();
        lib.add_scene(SceneTrimMetadata::new(0, 0, 23));
        lib.add_scene(SceneTrimMetadata::new(1, 24, 47));
        lib.add_scene(SceneTrimMetadata::new(2, 48, 71));

        let msgs = TrimValidator::check_continuity(&lib);
        assert!(msgs.is_empty(), "Expected no issues, got: {msgs:?}");
    }

    #[test]
    fn test_continuity_detects_gap() {
        let mut lib = TrimMetadataLibrary::new();
        lib.add_scene(SceneTrimMetadata::new(0, 0, 23));
        lib.add_scene(SceneTrimMetadata::new(1, 30, 53)); // gap at 24-29

        let msgs = TrimValidator::check_continuity(&lib);
        assert!(!msgs.is_empty());
        assert!(
            msgs[0].contains("Gap"),
            "Expected gap message, got: {}",
            msgs[0]
        );
    }

    #[test]
    fn test_continuity_detects_overlap() {
        let mut lib = TrimMetadataLibrary::new();
        lib.add_scene(SceneTrimMetadata::new(0, 0, 30));
        lib.add_scene(SceneTrimMetadata::new(1, 25, 50)); // overlap at 25-30

        let msgs = TrimValidator::check_continuity(&lib);
        assert!(!msgs.is_empty());
        assert!(
            msgs[0].contains("Overlap"),
            "Expected overlap message, got: {}",
            msgs[0]
        );
    }
}

// ── Spec-required types ───────────────────────────────────────────────────────

/// A per-scene tone-mapping trim with lift/gain/gamma controls.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct SceneTrim {
    /// Unique scene identifier.
    pub scene_id: u32,
    /// First frame of this scene (inclusive).
    pub start_frame: u64,
    /// Last frame of this scene (inclusive).
    pub end_frame: u64,
    /// Lift adjustment (black point offset, typically −0.5–0.5).
    pub lift: f32,
    /// Gain multiplier (typically 0.0–4.0).
    pub gain: f32,
    /// Gamma exponent (typically 0.5–2.0).
    pub gamma: f32,
}

impl SceneTrim {
    /// Number of frames covered by this scene.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame) + 1
    }

    /// Apply lift/gain/gamma to a normalised PQ value (0.0–1.0).
    ///
    /// Formula: `((pq + lift) * gain) ^ gamma`, clamped to [0, 1].
    #[must_use]
    pub fn apply_to_pq(&self, pq: f32) -> f32 {
        let v = (pq + self.lift) * self.gain;
        v.max(0.0).powf(self.gamma).min(1.0)
    }
}

/// Trim mode describes the DV specification level used for the trim metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum TrimMode {
    /// Level 2 trim: single-pass, global.
    Level2,
    /// Level 3 trim: mid-tone detail control.
    Level3,
    /// Level 8 trim: target-display-specific.
    Level8,
}

impl TrimMode {
    /// Processing complexity score (higher = more metadata to carry).
    #[must_use]
    pub const fn complexity(&self) -> u32 {
        match self {
            Self::Level2 => 1,
            Self::Level3 => 2,
            Self::Level8 => 4,
        }
    }
}

/// An ordered database of `SceneTrim` entries.
#[derive(Default, Debug)]
#[allow(dead_code)]
pub struct SceneTrimDatabase {
    /// All trims in order of `start_frame`.
    pub trims: Vec<SceneTrim>,
}

impl SceneTrimDatabase {
    /// Create an empty database.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a scene trim.
    pub fn add(&mut self, trim: SceneTrim) {
        self.trims.push(trim);
    }

    /// Find the trim whose frame range covers `frame`, if any.
    #[must_use]
    pub fn find_for_frame(&self, frame: u64) -> Option<&SceneTrim> {
        self.trims
            .iter()
            .find(|t| frame >= t.start_frame && frame <= t.end_frame)
    }

    /// Arithmetic mean of `gain` across all trims; 0.0 if empty.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_gain(&self) -> f32 {
        if self.trims.is_empty() {
            return 0.0;
        }
        self.trims.iter().map(|t| t.gain).sum::<f32>() / self.trims.len() as f32
    }

    /// Number of trims in the database.
    #[must_use]
    pub fn trim_count(&self) -> usize {
        self.trims.len()
    }
}

/// Validates a `SceneTrim` against allowed parameter ranges.
#[allow(dead_code)]
pub struct TrimValidation;

impl TrimValidation {
    /// Returns a list of validation error messages (empty = valid).
    ///
    /// Ranges checked:
    /// - `lift`: [−0.5, 0.5]
    /// - `gain`: [0.0, 4.0]
    /// - `gamma`: [0.5, 2.0]
    #[must_use]
    pub fn check(trim: &SceneTrim) -> Vec<String> {
        let mut errors = Vec::new();
        if !(-0.5..=0.5).contains(&trim.lift) {
            errors.push(format!("lift out of range [-0.5, 0.5]: {}", trim.lift));
        }
        if !(0.0..=4.0).contains(&trim.gain) {
            errors.push(format!("gain out of range [0.0, 4.0]: {}", trim.gain));
        }
        if !(0.5..=2.0).contains(&trim.gamma) {
            errors.push(format!("gamma out of range [0.5, 2.0]: {}", trim.gamma));
        }
        errors
    }
}

#[cfg(test)]
mod spec_tests {
    use super::*;

    fn neutral_trim(scene_id: u32, start: u64, end: u64) -> SceneTrim {
        SceneTrim {
            scene_id,
            start_frame: start,
            end_frame: end,
            lift: 0.0,
            gain: 1.0,
            gamma: 1.0,
        }
    }

    #[test]
    fn test_scene_trim_duration() {
        let t = neutral_trim(0, 10, 19);
        assert_eq!(t.duration_frames(), 10);
    }

    #[test]
    fn test_scene_trim_duration_single_frame() {
        let t = neutral_trim(0, 5, 5);
        assert_eq!(t.duration_frames(), 1);
    }

    #[test]
    fn test_apply_to_pq_neutral() {
        let t = neutral_trim(0, 0, 23);
        assert!((t.apply_to_pq(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_apply_to_pq_gain_doubles() {
        let t = SceneTrim {
            scene_id: 0,
            start_frame: 0,
            end_frame: 10,
            lift: 0.0,
            gain: 2.0,
            gamma: 1.0,
        };
        assert!((t.apply_to_pq(0.4) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_apply_to_pq_clamps_to_one() {
        let t = SceneTrim {
            scene_id: 0,
            start_frame: 0,
            end_frame: 10,
            lift: 0.0,
            gain: 4.0,
            gamma: 1.0,
        };
        assert_eq!(t.apply_to_pq(1.0), 1.0);
    }

    #[test]
    fn test_trim_mode_complexity_ordering() {
        assert!(TrimMode::Level8.complexity() > TrimMode::Level2.complexity());
        assert!(TrimMode::Level3.complexity() > TrimMode::Level2.complexity());
    }

    #[test]
    fn test_scene_trim_database_add_and_count() {
        let mut db = SceneTrimDatabase::new();
        db.add(neutral_trim(0, 0, 23));
        db.add(neutral_trim(1, 24, 47));
        assert_eq!(db.trim_count(), 2);
    }

    #[test]
    fn test_scene_trim_database_find_for_frame_hit() {
        let mut db = SceneTrimDatabase::new();
        db.add(neutral_trim(7, 100, 199));
        let result = db.find_for_frame(150);
        assert!(result.is_some());
        assert_eq!(result.expect("test expectation failed").scene_id, 7);
    }

    #[test]
    fn test_scene_trim_database_find_for_frame_miss() {
        let mut db = SceneTrimDatabase::new();
        db.add(neutral_trim(0, 0, 23));
        assert!(db.find_for_frame(50).is_none());
    }

    #[test]
    fn test_scene_trim_database_average_gain_empty() {
        let db = SceneTrimDatabase::new();
        assert_eq!(db.average_gain(), 0.0);
    }

    #[test]
    fn test_scene_trim_database_average_gain() {
        let mut db = SceneTrimDatabase::new();
        db.add(SceneTrim {
            scene_id: 0,
            start_frame: 0,
            end_frame: 10,
            lift: 0.0,
            gain: 2.0,
            gamma: 1.0,
        });
        db.add(SceneTrim {
            scene_id: 1,
            start_frame: 11,
            end_frame: 20,
            lift: 0.0,
            gain: 4.0,
            gamma: 1.0,
        });
        assert!((db.average_gain() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_trim_validation_valid() {
        let t = neutral_trim(0, 0, 10);
        assert!(TrimValidation::check(&t).is_empty());
    }

    #[test]
    fn test_trim_validation_lift_out_of_range() {
        let t = SceneTrim {
            scene_id: 0,
            start_frame: 0,
            end_frame: 10,
            lift: 1.0,
            gain: 1.0,
            gamma: 1.0,
        };
        let errs = TrimValidation::check(&t);
        assert!(!errs.is_empty());
        assert!(errs[0].contains("lift"));
    }

    #[test]
    fn test_trim_validation_gain_out_of_range() {
        let t = SceneTrim {
            scene_id: 0,
            start_frame: 0,
            end_frame: 10,
            lift: 0.0,
            gain: 5.0,
            gamma: 1.0,
        };
        let errs = TrimValidation::check(&t);
        assert!(errs.iter().any(|e| e.contains("gain")));
    }

    #[test]
    fn test_trim_validation_gamma_out_of_range() {
        let t = SceneTrim {
            scene_id: 0,
            start_frame: 0,
            end_frame: 10,
            lift: 0.0,
            gain: 1.0,
            gamma: 0.1,
        };
        let errs = TrimValidation::check(&t);
        assert!(errs.iter().any(|e| e.contains("gamma")));
    }
}
