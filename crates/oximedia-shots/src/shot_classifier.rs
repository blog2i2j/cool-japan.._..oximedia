//! Shot role classifier: identifies the **cinematic purpose** of a shot.
//!
//! While `ShotType` (ECU → ELS) describes the **framing size** and `CoverageType`
//! describes the **production coverage pattern**, this module classifies the
//! **narrative role** a shot plays in an edit:
//!
//! | [`CinematicRole`]   | Typical usage                                           |
//! |---------------------|---------------------------------------------------------|
//! | `CloseUp`           | Emphasis on emotion, detail, or dialogue reaction       |
//! | `MediumShot`        | Conversation, action at intermediate distance           |
//! | `WideShot`          | Establishing context, crowd, environment                |
//! | `Insert`            | Tight detail cut-in to a specific object or action      |
//! | `Cutaway`           | B-roll that briefly departs from the primary action     |
//! | `PointOfView`       | Subjective shot from a character's perspective          |
//! | `ReactionShot`      | Character response to action in a previous shot         |
//!
//! # Algorithm
//!
//! [`ShotClassifier`] derives a feature vector from the frame:
//!
//! 1. **Face-size ratio** — proxy for subject proximity (simulated via
//!    skin-tone segmentation + largest connected-component bounding box).
//! 2. **Overall edge density** — high density = complex scene (wide shot) or
//!    detail insert; low density = clean close-up or reaction shot.
//! 3. **Central-region edge concentration** — ratio of central to peripheral
//!    edge density; high concentration → insert or close-up.
//! 4. **Motion magnitude** (optional, requires two frames) — high motion →
//!    unlikely to be a reaction shot; near-zero motion → candidate for reaction.
//! 5. **Horizontal symmetry score** — symmetric frames are typical of
//!    point-of-view and master shots.
//!
//! The features are combined via weighted rules into a confidence distribution
//! over all seven roles, and the highest-scoring role is returned together with
//! its confidence and the full distribution for downstream blending.

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::{FrameBuffer, GrayImage};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The cinematic role (narrative purpose) of a shot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CinematicRole {
    /// Tight framing on a subject — emotion, dialogue, detail emphasis.
    CloseUp,
    /// Mid-range framing — conversation or action at intermediate distance.
    MediumShot,
    /// Wide or establishing framing — environment, context, crowd.
    WideShot,
    /// Tight cut-in to a specific prop or detail (not a person's face).
    Insert,
    /// Break-away shot of something other than the primary action (B-roll).
    Cutaway,
    /// Subjective shot from a character's point of view.
    PointOfView,
    /// Character reaction to an event in a preceding shot.
    ReactionShot,
}

impl CinematicRole {
    /// Return a short human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::CloseUp => "Close-Up",
            Self::MediumShot => "Medium Shot",
            Self::WideShot => "Wide Shot",
            Self::Insert => "Insert",
            Self::Cutaway => "Cutaway",
            Self::PointOfView => "Point of View",
            Self::ReactionShot => "Reaction Shot",
        }
    }

    /// Return a brief description of the role's typical use.
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::CloseUp => "Emphasis on emotion, expression, or detail",
            Self::MediumShot => "Standard dialogue or action framing",
            Self::WideShot => "Environmental context or establishing framing",
            Self::Insert => "Tight detail cut-in to a specific object",
            Self::Cutaway => "B-roll departing from the primary action",
            Self::PointOfView => "Subjective perspective of a character",
            Self::ReactionShot => "Character response to prior action",
        }
    }

    /// Total number of defined roles.
    pub const COUNT: usize = 7;

    /// All roles in declaration order.
    #[must_use]
    pub const fn all() -> [Self; Self::COUNT] {
        [
            Self::CloseUp,
            Self::MediumShot,
            Self::WideShot,
            Self::Insert,
            Self::Cutaway,
            Self::PointOfView,
            Self::ReactionShot,
        ]
    }
}

impl std::fmt::Display for CinematicRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// Feature vector
// ---------------------------------------------------------------------------

/// Internal feature vector extracted from a frame (and optionally a prior frame).
#[derive(Debug, Clone, Copy)]
struct FrameFeatures {
    /// Estimated face/person area as a fraction of the total frame area [0, 1].
    face_ratio: f32,
    /// Fraction of edge pixels in the whole frame [0, 1].
    overall_edge_density: f32,
    /// Ratio of central-region edge density to full-frame edge density [0, ∞).
    ///
    /// Values > 1 indicate edges are concentrated near the centre (insert/CU).
    central_edge_concentration: f32,
    /// Inter-frame optical flow magnitude normalised to [0, 1].
    ///
    /// `None` when no prior frame is available.
    motion_magnitude: Option<f32>,
    /// Horizontal symmetry score [0, 1] (1 = perfectly symmetric).
    symmetry_score: f32,
    /// Proportion of the frame occupied by skin-tone pixels [0, 1].
    skin_ratio: f32,
    /// Low-frequency energy fraction — high → smooth gradients (sky, flat BG)
    /// typical of wide or POV shots.
    low_freq_fraction: f32,
}

// ---------------------------------------------------------------------------
// Confidence distribution
// ---------------------------------------------------------------------------

/// Per-role confidence scores.  All values are in [0, 1]; they do *not*
/// necessarily sum to 1.0 because the roles are not mutually exclusive from a
/// feature standpoint.
#[derive(Debug, Clone, Copy)]
pub struct RoleConfidences {
    /// Confidence for each role in the same order as [`CinematicRole::all`].
    scores: [f32; CinematicRole::COUNT],
}

impl RoleConfidences {
    fn new(scores: [f32; CinematicRole::COUNT]) -> Self {
        Self { scores }
    }

    /// Get the confidence for a specific role.
    #[must_use]
    pub fn get(&self, role: CinematicRole) -> f32 {
        match role {
            CinematicRole::CloseUp => self.scores[0],
            CinematicRole::MediumShot => self.scores[1],
            CinematicRole::WideShot => self.scores[2],
            CinematicRole::Insert => self.scores[3],
            CinematicRole::Cutaway => self.scores[4],
            CinematicRole::PointOfView => self.scores[5],
            CinematicRole::ReactionShot => self.scores[6],
        }
    }

    /// Return the role with the highest confidence score.
    #[must_use]
    pub fn best_role(&self) -> (CinematicRole, f32) {
        let mut best_idx = 0;
        let mut best_score = self.scores[0];
        for (i, &s) in self.scores.iter().enumerate().skip(1) {
            if s > best_score {
                best_score = s;
                best_idx = i;
            }
        }
        (CinematicRole::all()[best_idx], best_score)
    }

    /// Return all roles sorted by confidence descending.
    #[must_use]
    pub fn ranked(&self) -> Vec<(CinematicRole, f32)> {
        let roles = CinematicRole::all();
        let mut pairs: Vec<(CinematicRole, f32)> = roles
            .iter()
            .zip(self.scores.iter())
            .map(|(&r, &s)| (r, s))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    }
}

// ---------------------------------------------------------------------------
// Classification result
// ---------------------------------------------------------------------------

/// Full result of shot role classification.
#[derive(Debug, Clone)]
pub struct ShotClassification {
    /// The most likely cinematic role for this shot.
    pub role: CinematicRole,
    /// Confidence of the primary role [0, 1].
    pub confidence: f32,
    /// Per-role confidence distribution.
    pub confidences: RoleConfidences,
    /// True if this shot was classified using motion information from a prior frame.
    pub used_motion: bool,
}

// ---------------------------------------------------------------------------
// Classifier configuration
// ---------------------------------------------------------------------------

/// Configuration for [`ShotClassifier`].
#[derive(Debug, Clone)]
pub struct ShotClassifierConfig {
    /// Face/skin ratio below which the shot is unlikely to be a close-up of a
    /// person (used for insert and cutaway detection).
    pub person_absent_threshold: f32,
    /// Face/skin ratio above which the shot is likely a close-up or reaction shot.
    pub close_up_face_threshold: f32,
    /// Overall edge density above which the frame is considered visually complex.
    pub complex_edge_threshold: f32,
    /// Central edge concentration ratio above which an insert is likely.
    pub insert_concentration_threshold: f32,
    /// Motion magnitude below which a shot is considered static (reaction/POV candidate).
    pub static_motion_threshold: f32,
    /// Symmetry score above which a POV or wide establishing shot is indicated.
    pub symmetry_threshold: f32,
    /// Low-frequency fraction above which the background is smooth (POV / wide).
    pub smooth_bg_threshold: f32,
}

impl Default for ShotClassifierConfig {
    fn default() -> Self {
        Self {
            person_absent_threshold: 0.03,
            close_up_face_threshold: 0.20,
            complex_edge_threshold: 0.12,
            insert_concentration_threshold: 1.6,
            static_motion_threshold: 0.05,
            symmetry_threshold: 0.75,
            smooth_bg_threshold: 0.65,
        }
    }
}

// ---------------------------------------------------------------------------
// Classifier
// ---------------------------------------------------------------------------

/// Shot role classifier.
///
/// Create with [`ShotClassifier::new`] or [`ShotClassifier::default`], then call
/// [`ShotClassifier::classify`] (single frame) or
/// [`ShotClassifier::classify_with_prior`] (with a preceding frame for motion).
pub struct ShotClassifier {
    config: ShotClassifierConfig,
}

impl Default for ShotClassifier {
    fn default() -> Self {
        Self::new(ShotClassifierConfig::default())
    }
}

impl ShotClassifier {
    /// Create a new classifier with the given configuration.
    #[must_use]
    pub fn new(config: ShotClassifierConfig) -> Self {
        Self { config }
    }

    /// Classify the cinematic role of a single frame.
    ///
    /// # Errors
    ///
    /// Returns [`ShotError::InvalidFrame`] if the frame has fewer than 3 channels
    /// or is empty.
    pub fn classify(&self, frame: &FrameBuffer) -> ShotResult<ShotClassification> {
        self.classify_with_prior(frame, None)
    }

    /// Classify the cinematic role of a frame, optionally using a prior frame
    /// to compute motion magnitude.
    ///
    /// # Errors
    ///
    /// Returns [`ShotError::InvalidFrame`] if the frame has fewer than 3 channels
    /// or is empty.
    pub fn classify_with_prior(
        &self,
        frame: &FrameBuffer,
        prior: Option<&FrameBuffer>,
    ) -> ShotResult<ShotClassification> {
        let (h, w, ch) = frame.dim();
        if ch < 3 {
            return Err(ShotError::InvalidFrame(
                "Frame must have at least 3 channels".to_string(),
            ));
        }
        if h == 0 || w == 0 {
            return Err(ShotError::InvalidFrame("Frame is empty".to_string()));
        }

        let features = self.extract_features(frame, prior)?;
        let confidences = self.score_roles(&features);
        let (role, confidence) = confidences.best_role();

        Ok(ShotClassification {
            role,
            confidence,
            confidences,
            used_motion: features.motion_magnitude.is_some(),
        })
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &ShotClassifierConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Feature extraction
    // -----------------------------------------------------------------------

    fn extract_features(
        &self,
        frame: &FrameBuffer,
        prior: Option<&FrameBuffer>,
    ) -> ShotResult<FrameFeatures> {
        let gray = to_grayscale(frame);
        let edges = compute_sobel(&gray);

        let overall_edge_density = mean_edge_density(&edges);
        let central_edge_concentration = central_concentration(&edges);
        let face_ratio = estimate_face_ratio(frame, &gray);
        let skin_ratio = estimate_skin_ratio(frame);
        let symmetry_score = compute_horizontal_symmetry(&gray);
        let low_freq_fraction = compute_low_freq_fraction(&gray);

        let motion_magnitude = if let Some(prev) = prior {
            Some(estimate_motion(prev, frame)?)
        } else {
            None
        };

        Ok(FrameFeatures {
            face_ratio,
            overall_edge_density,
            central_edge_concentration,
            motion_magnitude,
            symmetry_score,
            skin_ratio,
            low_freq_fraction,
        })
    }

    // -----------------------------------------------------------------------
    // Scoring
    // -----------------------------------------------------------------------

    /// Convert a feature vector into per-role confidence scores.
    fn score_roles(&self, f: &FrameFeatures) -> RoleConfidences {
        let cfg = &self.config;

        // ── CloseUp ──────────────────────────────────────────────────────────
        // High face/skin presence, moderate edge density, concentrated centre.
        let close_up = {
            let face_score = sigmoid(f.face_ratio, cfg.close_up_face_threshold, 10.0);
            let skin_score = sigmoid(f.skin_ratio, 0.15, 8.0);
            let edge_ok = 1.0 - sigmoid(f.overall_edge_density, cfg.complex_edge_threshold, 5.0);
            (face_score * 0.45 + skin_score * 0.35 + edge_ok * 0.20).clamp(0.0, 1.0)
        };

        // ── MediumShot ────────────────────────────────────────────────────────
        // Moderate face presence, moderate edge density, neither extreme.
        let medium_shot = {
            // Bell-shaped: score peaks when face_ratio ≈ 0.10
            let face_score = bell(f.face_ratio, 0.10, 0.08);
            // Edge density near mid-range
            let edge_score = bell(f.overall_edge_density, 0.08, 0.06);
            (face_score * 0.55 + edge_score * 0.45).clamp(0.0, 1.0)
        };

        // ── WideShot ──────────────────────────────────────────────────────────
        // Low face ratio, high overall edge density (complex scene), low central concentration.
        let wide_shot = {
            let no_face = 1.0 - sigmoid(f.face_ratio, cfg.person_absent_threshold, 8.0);
            let complex = sigmoid(f.overall_edge_density, cfg.complex_edge_threshold, 6.0);
            let low_centre = 1.0
                - sigmoid(
                    f.central_edge_concentration,
                    cfg.insert_concentration_threshold,
                    4.0,
                );
            (no_face * 0.35 + complex * 0.40 + low_centre * 0.25).clamp(0.0, 1.0)
        };

        // ── Insert ────────────────────────────────────────────────────────────
        // No face, high central edge concentration, moderate total edge density.
        let insert = {
            let no_face = 1.0 - sigmoid(f.face_ratio, cfg.person_absent_threshold, 10.0);
            let concentrated = sigmoid(
                f.central_edge_concentration,
                cfg.insert_concentration_threshold,
                5.0,
            );
            let moderate_edges = bell(f.overall_edge_density, 0.10, 0.07);
            (no_face * 0.40 + concentrated * 0.40 + moderate_edges * 0.20).clamp(0.0, 1.0)
        };

        // ── Cutaway ───────────────────────────────────────────────────────────
        // No face, high motion (cut from main action), high edge complexity, low symmetry.
        let cutaway = {
            let no_face = 1.0 - sigmoid(f.face_ratio, cfg.person_absent_threshold, 8.0);
            let complex = sigmoid(f.overall_edge_density, cfg.complex_edge_threshold, 5.0);
            let asymmetric = 1.0 - sigmoid(f.symmetry_score, cfg.symmetry_threshold, 6.0);
            let motion_boost = f
                .motion_magnitude
                .map(|m| sigmoid(m, 0.15, 5.0) * 0.20)
                .unwrap_or(0.0);
            (no_face * 0.35 + complex * 0.25 + asymmetric * 0.20 + motion_boost + 0.0)
                .clamp(0.0, 1.0)
        };

        // ── PointOfView ───────────────────────────────────────────────────────
        // High symmetry, smooth background (lens-forward), low face ratio (camera
        // *is* the character's eyes), moderate motion (walking / looking).
        let point_of_view = {
            let symmetric = sigmoid(f.symmetry_score, cfg.symmetry_threshold, 8.0);
            let smooth = sigmoid(f.low_freq_fraction, cfg.smooth_bg_threshold, 6.0);
            let no_face = 1.0 - sigmoid(f.face_ratio, cfg.person_absent_threshold, 8.0);
            let motion_ok = f
                .motion_magnitude
                .map(|m| {
                    // POV often has mild motion; penalise both static and very high motion
                    bell(m, 0.12, 0.10)
                })
                .unwrap_or(0.5);
            (symmetric * 0.35 + smooth * 0.25 + no_face * 0.20 + motion_ok * 0.20).clamp(0.0, 1.0)
        };

        // ── ReactionShot ──────────────────────────────────────────────────────
        // Face present, static or near-static frame, low edge density overall
        // (clean background), moderate skin ratio.
        let reaction_shot = {
            let has_face = sigmoid(f.face_ratio, cfg.person_absent_threshold, 10.0);
            let static_frame = f
                .motion_magnitude
                .map(|m| 1.0 - sigmoid(m, cfg.static_motion_threshold, 8.0))
                .unwrap_or(0.5);
            let clean_bg = 1.0 - sigmoid(f.overall_edge_density, cfg.complex_edge_threshold, 5.0);
            let skin_present = sigmoid(f.skin_ratio, 0.10, 8.0);
            (has_face * 0.35 + static_frame * 0.25 + clean_bg * 0.20 + skin_present * 0.20)
                .clamp(0.0, 1.0)
        };

        RoleConfidences::new([
            close_up,
            medium_shot,
            wide_shot,
            insert,
            cutaway,
            point_of_view,
            reaction_shot,
        ])
    }
}

// ---------------------------------------------------------------------------
// Mathematical helpers
// ---------------------------------------------------------------------------

/// Sigmoid function centred at `mid` with steepness `k`.
///
/// Returns values in (0, 1).  Output is ~0.5 when `x == mid`.
#[inline]
fn sigmoid(x: f32, mid: f32, k: f32) -> f32 {
    1.0 / (1.0 + (-(k * (x - mid))).exp())
}

/// Gaussian bell function centred at `mu` with standard deviation `sigma`.
///
/// Returns 1.0 at the centre and decays towards 0 at the tails.
#[inline]
fn bell(x: f32, mu: f32, sigma: f32) -> f32 {
    if sigma < f32::EPSILON {
        return if (x - mu).abs() < f32::EPSILON {
            1.0
        } else {
            0.0
        };
    }
    let z = (x - mu) / sigma;
    (-0.5 * z * z).exp()
}

// ---------------------------------------------------------------------------
// Image processing primitives
// ---------------------------------------------------------------------------

/// Convert an RGB `FrameBuffer` to a grayscale `GrayImage` using BT.601 weights.
fn to_grayscale(frame: &FrameBuffer) -> GrayImage {
    let (h, w, _) = frame.dim();
    let mut gray = GrayImage::zeros(h, w);
    for y in 0..h {
        for x in 0..w {
            let r = f32::from(frame.get(y, x, 0));
            let g = f32::from(frame.get(y, x, 1));
            let b = f32::from(frame.get(y, x, 2));
            gray.set(y, x, (r * 0.299 + g * 0.587 + b * 0.114) as u8);
        }
    }
    gray
}

/// Compute a Sobel-magnitude edge image from a grayscale image.
fn compute_sobel(gray: &GrayImage) -> GrayImage {
    let (h, w) = gray.dim();
    let mut edges = GrayImage::zeros(h, w);
    if h < 3 || w < 3 {
        return edges;
    }
    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let gx = -i32::from(gray.get(y - 1, x - 1))
                - 2 * i32::from(gray.get(y, x - 1))
                - i32::from(gray.get(y + 1, x - 1))
                + i32::from(gray.get(y - 1, x + 1))
                + 2 * i32::from(gray.get(y, x + 1))
                + i32::from(gray.get(y + 1, x + 1));
            let gy = -i32::from(gray.get(y - 1, x - 1))
                - 2 * i32::from(gray.get(y - 1, x))
                - i32::from(gray.get(y - 1, x + 1))
                + i32::from(gray.get(y + 1, x - 1))
                + 2 * i32::from(gray.get(y + 1, x))
                + i32::from(gray.get(y + 1, x + 1));
            let mag = ((gx * gx + gy * gy) as f32).sqrt();
            edges.set(
                y,
                x,
                (mag / std::f32::consts::SQRT_2 / 255.0 * 255.0).min(255.0) as u8,
            );
        }
    }
    edges
}

/// Mean edge pixel value normalised to [0, 1].
fn mean_edge_density(edges: &GrayImage) -> f32 {
    let (h, w) = edges.dim();
    if h == 0 || w == 0 {
        return 0.0;
    }
    let sum: u64 = (0..h)
        .flat_map(|y| (0..w).map(move |x| u64::from(edges.get(y, x))))
        .sum();
    sum as f32 / ((h * w) as f32 * 255.0)
}

/// Ratio of central-region mean edge density to full-frame mean edge density.
///
/// The central region is the inner 50 % (quarter-point to three-quarter-point
/// in both axes).  Values > 1 indicate edge concentration near the centre.
fn central_concentration(edges: &GrayImage) -> f32 {
    let (h, w) = edges.dim();
    if h < 4 || w < 4 {
        return 1.0;
    }
    let y0 = h / 4;
    let y1 = (3 * h / 4).max(y0 + 1);
    let x0 = w / 4;
    let x1 = (3 * w / 4).max(x0 + 1);

    let mut central_sum = 0u64;
    let central_count = ((y1 - y0) * (x1 - x0)) as u64;
    for y in y0..y1 {
        for x in x0..x1 {
            central_sum += u64::from(edges.get(y, x));
        }
    }
    let global_density = mean_edge_density(edges);
    if global_density < f32::EPSILON {
        return 1.0;
    }
    let central_density = central_sum as f32 / (central_count as f32 * 255.0);
    central_density / global_density
}

/// Estimate the face/person presence as a fraction of the frame area.
///
/// This is a heuristic using skin-tone detection in the upper-centre region of
/// the frame (where faces tend to appear) combined with region-based weighting.
fn estimate_face_ratio(frame: &FrameBuffer, gray: &GrayImage) -> f32 {
    let (h, w, _) = frame.dim();
    if h == 0 || w == 0 {
        return 0.0;
    }

    // Search in upper 60 % of frame, centre 80 %
    let y_end = (h * 6 / 10).max(1);
    let x_start = w / 10;
    let x_end = (9 * w / 10).max(x_start + 1);

    let mut skin_count = 0u32;
    let total = ((y_end) * (x_end - x_start)) as f32;
    if total < 1.0 {
        return 0.0;
    }

    for y in 0..y_end {
        for x in x_start..x_end {
            if is_skin(frame.get(y, x, 0), frame.get(y, x, 1), frame.get(y, x, 2)) {
                skin_count += 1;
            }
        }
    }

    // Also require some texture (non-flat regions) in the same area
    let mut textured_skin = 0u32;
    for y in 1..(y_end.saturating_sub(1)) {
        for x in (x_start + 1)..(x_end.saturating_sub(1)) {
            if is_skin(frame.get(y, x, 0), frame.get(y, x, 1), frame.get(y, x, 2))
                && u32::from(gray.get(y, x)) > 20
            {
                textured_skin += 1;
            }
        }
    }

    let raw_ratio = skin_count as f32 / total;
    let texture_ratio = textured_skin as f32 / total;

    // Weight: raw presence + texture confirmation
    (raw_ratio * 0.6 + texture_ratio * 0.4).min(1.0)
}

/// Estimate the global skin-tone fraction of the frame.
fn estimate_skin_ratio(frame: &FrameBuffer) -> f32 {
    let (h, w, _) = frame.dim();
    if h == 0 || w == 0 {
        return 0.0;
    }
    let mut count = 0u32;
    let total = (h * w) as f32;
    for y in 0..h {
        for x in 0..w {
            if is_skin(frame.get(y, x, 0), frame.get(y, x, 1), frame.get(y, x, 2)) {
                count += 1;
            }
        }
    }
    count as f32 / total
}

/// Skin-tone classifier based on RGB thresholds (Kovac et al., 2003 approximation).
///
/// Returns `true` if the pixel is likely to be skin.
#[inline]
fn is_skin(r: u8, g: u8, b: u8) -> bool {
    let r = u32::from(r);
    let g = u32::from(g);
    let b = u32::from(b);
    // Classic RGB skin rule (slightly relaxed for diverse tones)
    r > 95
        && g > 40
        && b > 20
        && r > g
        && r > b
        && r.saturating_sub(g) > 15
        && (r.max(g).max(b)).saturating_sub(r.min(g).min(b)) > 15
}

/// Compute horizontal (left–right) symmetry score in [0, 1].
///
/// Divides the grayscale image into left and right halves, mirrors the right,
/// and computes the normalised mean absolute difference (lower = more symmetric,
/// converted to score = 1 - MAD).
fn compute_horizontal_symmetry(gray: &GrayImage) -> f32 {
    let (h, w) = gray.dim();
    if h == 0 || w < 2 {
        return 0.5;
    }
    let half_w = w / 2;
    let mut diff_sum = 0u64;
    let total = (h * half_w) as u64;
    if total == 0 {
        return 0.5;
    }
    for y in 0..h {
        for x in 0..half_w {
            let left = u32::from(gray.get(y, x));
            let right = u32::from(gray.get(y, w - 1 - x));
            diff_sum += left.abs_diff(right) as u64;
        }
    }
    let mad = diff_sum as f32 / (total as f32 * 255.0);
    (1.0 - mad).clamp(0.0, 1.0)
}

/// Estimate the fraction of energy in the low-frequency component of the image.
///
/// Uses a simple 8×8 block-mean downscale: smooth images (sky, flat backgrounds)
/// lose very little energy when downscaled, while complex images lose a lot.
fn compute_low_freq_fraction(gray: &GrayImage) -> f32 {
    let (h, w) = gray.dim();
    if h < 8 || w < 8 {
        return 0.5;
    }
    let bh = h / 8;
    let bw = w / 8;

    // Full-res energy
    let mut full_energy = 0.0_f64;
    for y in 0..h {
        for x in 0..w {
            let v = f64::from(gray.get(y, x));
            full_energy += v * v;
        }
    }

    // Low-res (block-mean) energy, upsampled
    let mut low_energy = 0.0_f64;
    for by in 0..8 {
        for bx in 0..8 {
            let y_start = by * bh;
            let y_end = (y_start + bh).min(h);
            let x_start = bx * bw;
            let x_end = (x_start + bw).min(w);
            let n = ((y_end - y_start) * (x_end - x_start)) as f64;
            if n < 1.0 {
                continue;
            }
            let mut block_sum = 0.0_f64;
            for y in y_start..y_end {
                for x in x_start..x_end {
                    block_sum += f64::from(gray.get(y, x));
                }
            }
            let mean = block_sum / n;
            low_energy += mean * mean * n;
        }
    }

    if full_energy < f64::EPSILON {
        return 0.5;
    }
    (low_energy / full_energy).clamp(0.0, 1.0) as f32
}

/// Estimate normalised inter-frame motion magnitude.
///
/// Uses a simple block-based mean absolute difference (MAD) approach:
/// the frame is divided into 8×8 macro-blocks and the MAD across all blocks
/// is averaged and normalised to [0, 1].
///
/// # Errors
///
/// Returns an error if either frame has fewer than 3 channels or is empty.
fn estimate_motion(prev: &FrameBuffer, curr: &FrameBuffer) -> ShotResult<f32> {
    let (ph, pw, pc) = prev.dim();
    let (ch, cw, cc) = curr.dim();
    if pc < 3 || cc < 3 {
        return Err(ShotError::InvalidFrame(
            "Motion estimation requires 3-channel frames".to_string(),
        ));
    }
    if ph == 0 || pw == 0 || ch == 0 || cw == 0 {
        return Ok(0.0);
    }

    let h = ph.min(ch);
    let w = pw.min(cw);

    let prev_gray = to_grayscale(prev);
    let curr_gray = to_grayscale(curr);

    let block_h = (h / 8).max(1);
    let block_w = (w / 8).max(1);
    let mut total_mad = 0.0_f64;
    let mut block_count = 0u32;

    let mut y = 0;
    while y + block_h <= h {
        let mut x = 0;
        while x + block_w <= w {
            let mut diff_sum = 0u64;
            let n = (block_h * block_w) as u64;
            for by in 0..block_h {
                for bx in 0..block_w {
                    let pv = u32::from(prev_gray.get(y + by, x + bx));
                    let cv = u32::from(curr_gray.get(y + by, x + bx));
                    diff_sum += pv.abs_diff(cv) as u64;
                }
            }
            total_mad += diff_sum as f64 / (n as f64 * 255.0);
            block_count += 1;
            x += block_w;
        }
        y += block_h;
    }

    if block_count == 0 {
        return Ok(0.0);
    }
    Ok((total_mad / f64::from(block_count)) as f32)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────────────

    fn solid_frame(r: u8, g: u8, b: u8, h: usize, w: usize) -> FrameBuffer {
        let mut frame = FrameBuffer::zeros(h, w, 3);
        for y in 0..h {
            for x in 0..w {
                frame.set(y, x, 0, r);
                frame.set(y, x, 1, g);
                frame.set(y, x, 2, b);
            }
        }
        frame
    }

    /// Create a frame that looks like a skin-tone face filling most of the frame.
    fn face_frame(h: usize, w: usize) -> FrameBuffer {
        // r=200,g=150,b=120 — satisfies Kovac skin rule
        solid_frame(200, 150, 120, h, w)
    }

    /// Create a frame with random-looking high-frequency content (edges everywhere).
    fn noisy_frame(h: usize, w: usize) -> FrameBuffer {
        let mut frame = FrameBuffer::zeros(h, w, 3);
        for y in 0..h {
            for x in 0..w {
                let v = ((y + x) % 2 * 255) as u8;
                frame.set(y, x, 0, v);
                frame.set(y, x, 1, v);
                frame.set(y, x, 2, v);
            }
        }
        frame
    }

    // ── CinematicRole ──────────────────────────────────────────────────────────

    #[test]
    fn test_cinematic_role_labels_are_non_empty() {
        for role in CinematicRole::all() {
            assert!(!role.label().is_empty(), "label must not be empty");
            assert!(
                !role.description().is_empty(),
                "description must not be empty"
            );
        }
    }

    #[test]
    fn test_cinematic_role_display() {
        let role = CinematicRole::CloseUp;
        assert_eq!(format!("{role}"), "Close-Up");
    }

    #[test]
    fn test_cinematic_role_count() {
        assert_eq!(CinematicRole::all().len(), CinematicRole::COUNT);
    }

    // ── RoleConfidences ────────────────────────────────────────────────────────

    #[test]
    fn test_role_confidences_get() {
        let scores = [0.9, 0.5, 0.3, 0.1, 0.2, 0.4, 0.7];
        let rc = RoleConfidences::new(scores);
        assert!((rc.get(CinematicRole::CloseUp) - 0.9).abs() < 1e-6);
        assert!((rc.get(CinematicRole::ReactionShot) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_role_confidences_best_role() {
        let scores = [0.1, 0.2, 0.9, 0.3, 0.4, 0.5, 0.6];
        let rc = RoleConfidences::new(scores);
        let (best, score) = rc.best_role();
        assert_eq!(best, CinematicRole::WideShot);
        assert!((score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_role_confidences_ranked_order() {
        let scores = [0.5, 0.7, 0.3, 0.8, 0.1, 0.6, 0.2];
        let rc = RoleConfidences::new(scores);
        let ranked = rc.ranked();
        assert_eq!(ranked.len(), CinematicRole::COUNT);
        // First should be Insert (0.8)
        assert_eq!(ranked[0].0, CinematicRole::Insert);
        // Scores should be descending
        for i in 0..ranked.len() - 1 {
            assert!(ranked[i].1 >= ranked[i + 1].1);
        }
    }

    // ── ShotClassifier ─────────────────────────────────────────────────────────

    #[test]
    fn test_classifier_default_construction() {
        let _c = ShotClassifier::default();
    }

    #[test]
    fn test_classify_returns_ok_for_valid_frame() {
        let classifier = ShotClassifier::default();
        let frame = face_frame(64, 64);
        let result = classifier.classify(&frame);
        assert!(result.is_ok(), "classify should succeed on valid frame");
    }

    #[test]
    fn test_classify_error_on_insufficient_channels() {
        let classifier = ShotClassifier::default();
        let frame = FrameBuffer::zeros(64, 64, 1);
        assert!(
            classifier.classify(&frame).is_err(),
            "1-channel frame must be rejected"
        );
    }

    #[test]
    fn test_classify_error_on_empty_frame() {
        let classifier = ShotClassifier::default();
        let frame = FrameBuffer::zeros(0, 0, 3);
        assert!(
            classifier.classify(&frame).is_err(),
            "empty frame must be rejected"
        );
    }

    #[test]
    fn test_confidence_is_bounded() {
        let classifier = ShotClassifier::default();
        for frame in [
            face_frame(64, 64),
            noisy_frame(64, 64),
            solid_frame(50, 100, 200, 64, 64),
        ] {
            let result = classifier
                .classify(&frame)
                .expect("classify should succeed");
            assert!(
                (0.0..=1.0).contains(&result.confidence),
                "confidence must be in [0, 1], got {}",
                result.confidence
            );
        }
    }

    #[test]
    fn test_skin_heavy_frame_favours_close_up_or_reaction() {
        let classifier = ShotClassifier::default();
        let frame = face_frame(80, 80);
        let result = classifier
            .classify(&frame)
            .expect("classify should succeed");
        // A frame dominated by skin tones should score high for close-up or reaction
        let cu = result.confidences.get(CinematicRole::CloseUp);
        let rx = result.confidences.get(CinematicRole::ReactionShot);
        assert!(
            cu > 0.4 || rx > 0.4,
            "skin-heavy frame should score ≥0.4 for CloseUp ({cu:.2}) or ReactionShot ({rx:.2})"
        );
    }

    #[test]
    fn test_noisy_frame_favours_wide_or_cutaway() {
        let classifier = ShotClassifier::default();
        let frame = noisy_frame(80, 80);
        let result = classifier
            .classify(&frame)
            .expect("classify should succeed");
        let wide = result.confidences.get(CinematicRole::WideShot);
        let cut = result.confidences.get(CinematicRole::Cutaway);
        assert!(
            wide > 0.3 || cut > 0.3,
            "noisy frame should score ≥0.3 for WideShot ({wide:.2}) or Cutaway ({cut:.2})"
        );
    }

    #[test]
    fn test_classify_with_prior_uses_motion() {
        let classifier = ShotClassifier::default();
        let frame_a = solid_frame(100, 100, 100, 64, 64);
        let frame_b = solid_frame(180, 180, 180, 64, 64);
        let result = classifier
            .classify_with_prior(&frame_b, Some(&frame_a))
            .expect("classify_with_prior should succeed");
        assert!(
            result.used_motion,
            "should report motion when prior frame is provided"
        );
    }

    #[test]
    fn test_classify_without_prior_no_motion() {
        let classifier = ShotClassifier::default();
        let frame = face_frame(64, 64);
        let result = classifier
            .classify(&frame)
            .expect("classify should succeed");
        assert!(
            !result.used_motion,
            "should not report motion without prior frame"
        );
    }

    #[test]
    fn test_all_roles_have_non_negative_confidence() {
        let classifier = ShotClassifier::default();
        let frame = noisy_frame(64, 64);
        let result = classifier
            .classify(&frame)
            .expect("classify should succeed");
        for role in CinematicRole::all() {
            let s = result.confidences.get(role);
            assert!(s >= 0.0, "confidence for {role} must be ≥ 0");
        }
    }

    #[test]
    fn test_config_accessor() {
        let cfg = ShotClassifierConfig {
            close_up_face_threshold: 0.30,
            ..ShotClassifierConfig::default()
        };
        let classifier = ShotClassifier::new(cfg);
        assert!((classifier.config().close_up_face_threshold - 0.30).abs() < f32::EPSILON);
    }

    // ── Mathematical helpers ────────────────────────────────────────────────────

    #[test]
    fn test_sigmoid_midpoint() {
        let s = sigmoid(0.5, 0.5, 10.0);
        assert!((s - 0.5).abs() < 1e-5, "sigmoid at midpoint should be 0.5");
    }

    #[test]
    fn test_sigmoid_monotone() {
        let s1 = sigmoid(0.1, 0.5, 10.0);
        let s2 = sigmoid(0.9, 0.5, 10.0);
        assert!(s1 < s2, "sigmoid should be monotonically increasing");
    }

    #[test]
    fn test_bell_peak() {
        let b = bell(0.5, 0.5, 0.1);
        assert!((b - 1.0).abs() < 1e-5, "bell at centre should be 1.0");
    }

    #[test]
    fn test_bell_tails() {
        let b = bell(1.5, 0.5, 0.1);
        assert!(b < 0.01, "bell far from centre should be near 0");
    }

    // ── Image primitives ───────────────────────────────────────────────────────

    #[test]
    fn test_to_grayscale_uniform_rgb() {
        let frame = solid_frame(100, 100, 100, 10, 10);
        let gray = to_grayscale(&frame);
        assert_eq!(gray.get(5, 5), 100);
    }

    #[test]
    fn test_mean_edge_density_flat_image() {
        let gray = GrayImage::zeros(64, 64);
        let edges = compute_sobel(&gray);
        let density = mean_edge_density(&edges);
        assert!(density < f32::EPSILON, "flat image should have zero edges");
    }

    #[test]
    fn test_horizontal_symmetry_symmetric_frame() {
        // A solid frame is perfectly symmetric
        let frame = solid_frame(128, 128, 128, 32, 32);
        let gray = to_grayscale(&frame);
        let score = compute_horizontal_symmetry(&gray);
        assert!(score > 0.99, "solid frame symmetry should be ~1.0");
    }

    #[test]
    fn test_low_freq_fraction_smooth_image() {
        let frame = solid_frame(128, 128, 128, 64, 64);
        let gray = to_grayscale(&frame);
        let lff = compute_low_freq_fraction(&gray);
        assert!(
            lff > 0.90,
            "smooth image should have high low-freq fraction, got {lff}"
        );
    }

    #[test]
    fn test_estimate_motion_identical_frames() {
        let frame = solid_frame(120, 120, 120, 64, 64);
        let motion = estimate_motion(&frame, &frame).expect("motion estimation should succeed");
        assert!(
            motion < 0.01,
            "identical frames should have near-zero motion"
        );
    }

    #[test]
    fn test_estimate_motion_different_frames() {
        let prev = solid_frame(0, 0, 0, 64, 64);
        let curr = solid_frame(255, 255, 255, 64, 64);
        let motion = estimate_motion(&prev, &curr).expect("motion estimation should succeed");
        assert!(
            motion > 0.5,
            "black→white transition should have high motion"
        );
    }

    #[test]
    fn test_is_skin_positive() {
        // Typical Caucasian skin tone
        assert!(is_skin(200, 150, 120), "should be classified as skin");
    }

    #[test]
    fn test_is_skin_negative_blue() {
        assert!(!is_skin(50, 50, 200), "blue pixel should not be skin");
    }

    #[test]
    fn test_central_concentration_uniform_edges() {
        let frame = noisy_frame(64, 64);
        let gray = to_grayscale(&frame);
        let edges = compute_sobel(&gray);
        let conc = central_concentration(&edges);
        // For a uniformly noisy image, concentration should be ~1.0
        assert!(
            conc > 0.5,
            "uniform edge distribution → concentration ≈ 1.0"
        );
    }

    #[test]
    fn test_insert_role_detectable_when_centre_sharp() {
        // Build a frame with only the central region sharp (high contrast checkerboard)
        // and the periphery flat — this should push insert score up.
        let h = 80;
        let w = 80;
        let mut frame = solid_frame(128, 100, 80, h, w);
        // Centre 40x40 filled with alternating black/white for high edge density
        for y in 20..60 {
            for x in 20..60 {
                let v = ((y + x) % 2 * 255) as u8;
                frame.set(y, x, 0, v);
                frame.set(y, x, 1, v);
                frame.set(y, x, 2, v);
            }
        }
        // No skin (all grey periphery, B&W centre) → no face presence
        let classifier = ShotClassifier::default();
        let result = classifier.classify(&frame).expect("should succeed");
        let insert_score = result.confidences.get(CinematicRole::Insert);
        // Insert should receive meaningful confidence
        assert!(
            insert_score > 0.25,
            "centre-sharp, no-face frame should score ≥ 0.25 for Insert (got {insert_score:.2})"
        );
    }

    #[test]
    fn test_wide_shot_role_for_complex_scene() {
        // High overall edge density and no faces → wide or cutaway
        let classifier = ShotClassifier::default();
        let frame = noisy_frame(64, 64);
        let result = classifier.classify(&frame).expect("should succeed");
        let wide = result.confidences.get(CinematicRole::WideShot);
        let cut = result.confidences.get(CinematicRole::Cutaway);
        assert!(
            wide > 0.2 || cut > 0.2,
            "complex no-face frame should favour Wide or Cutaway"
        );
    }
}
