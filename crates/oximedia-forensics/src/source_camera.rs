#![allow(dead_code)]
//! Source camera identification via sensor fingerprinting.
//!
//! This module identifies the source camera of an image using sensor
//! noise fingerprints (PRNU), lens aberration patterns, color filter
//! array (CFA) interpolation artifacts, and other device-specific
//! characteristics.

use std::collections::HashMap;

/// Camera sensor type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SensorType {
    /// CCD sensor.
    Ccd,
    /// CMOS sensor (rolling shutter).
    CmosRolling,
    /// CMOS sensor (global shutter).
    CmosGlobal,
    /// Back-side illuminated CMOS.
    BsiCmos,
    /// Foveon (stacked RGB layers).
    Foveon,
    /// Unknown sensor type.
    Unknown,
}

impl SensorType {
    /// Return a human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Ccd => "CCD",
            Self::CmosRolling => "CMOS (Rolling)",
            Self::CmosGlobal => "CMOS (Global)",
            Self::BsiCmos => "BSI CMOS",
            Self::Foveon => "Foveon",
            Self::Unknown => "Unknown",
        }
    }
}

/// Color Filter Array pattern type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfaPattern {
    /// Standard Bayer RGGB pattern.
    BayerRggb,
    /// Bayer BGGR pattern.
    BayerBggr,
    /// Bayer GRBG pattern.
    BayerGrbg,
    /// Bayer GBRG pattern.
    BayerGbrg,
    /// X-Trans pattern (Fujifilm).
    XTrans,
    /// No CFA (Foveon, monochrome).
    None,
}

impl CfaPattern {
    /// Return a human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::BayerRggb => "Bayer RGGB",
            Self::BayerBggr => "Bayer BGGR",
            Self::BayerGrbg => "Bayer GRBG",
            Self::BayerGbrg => "Bayer GBRG",
            Self::XTrans => "X-Trans",
            Self::None => "None",
        }
    }
}

/// CFA demosaicing artifact analysis result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CfaAnalysis {
    /// Detected CFA pattern.
    pub pattern: CfaPattern,
    /// Strength of CFA artifacts (higher = more visible).
    pub artifact_strength: f64,
    /// Confidence of the detection (0.0..1.0).
    pub confidence: f64,
}

impl CfaAnalysis {
    /// Create a new CFA analysis result.
    #[must_use]
    pub fn new(pattern: CfaPattern, artifact_strength: f64, confidence: f64) -> Self {
        Self {
            pattern,
            artifact_strength,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Check if CFA artifacts are detectable.
    #[must_use]
    pub fn artifacts_present(&self) -> bool {
        self.artifact_strength > 0.01 && self.confidence > 0.5
    }
}

/// Lens aberration fingerprint for camera identification.
#[derive(Debug, Clone, PartialEq)]
pub struct LensFingerprint {
    /// Radial distortion coefficients [k1, k2, k3].
    pub radial_coeffs: [f64; 3],
    /// Chromatic aberration magnitude (pixels).
    pub chromatic_aberration: f64,
    /// Vignetting falloff factor.
    pub vignetting_falloff: f64,
    /// Confidence of the lens identification.
    pub confidence: f64,
}

impl LensFingerprint {
    /// Create a new lens fingerprint.
    #[must_use]
    pub fn new(
        radial_coeffs: [f64; 3],
        chromatic_aberration: f64,
        vignetting_falloff: f64,
    ) -> Self {
        Self {
            radial_coeffs,
            chromatic_aberration,
            vignetting_falloff,
            confidence: 0.0,
        }
    }

    /// Compute the similarity to another lens fingerprint.
    #[must_use]
    pub fn similarity(&self, other: &Self) -> f64 {
        // Weighted distance in parameter space
        let mut dist = 0.0;
        for i in 0..3 {
            dist += (self.radial_coeffs[i] - other.radial_coeffs[i]).powi(2);
        }
        dist += (self.chromatic_aberration - other.chromatic_aberration).powi(2) * 0.1;
        dist += (self.vignetting_falloff - other.vignetting_falloff).powi(2) * 0.1;
        let distance = dist.sqrt();

        // Convert distance to similarity (0..1)
        1.0 / (1.0 + distance * 10.0)
    }

    /// Check if distortion is minimal (possibly a prime lens or corrected).
    #[must_use]
    pub fn is_low_distortion(&self) -> bool {
        self.radial_coeffs.iter().all(|c| c.abs() < 0.001)
    }
}

/// Camera identification match result.
#[derive(Debug, Clone)]
pub struct CameraMatch {
    /// Camera make/model string.
    pub camera_model: String,
    /// Match confidence (0.0..1.0).
    pub confidence: f64,
    /// PRNU correlation score.
    pub prnu_score: f64,
    /// CFA pattern match score.
    pub cfa_score: f64,
    /// Lens fingerprint similarity score.
    pub lens_score: f64,
}

impl CameraMatch {
    /// Create a new camera match result.
    #[must_use]
    pub fn new(camera_model: &str, confidence: f64) -> Self {
        Self {
            camera_model: camera_model.to_string(),
            confidence: confidence.clamp(0.0, 1.0),
            prnu_score: 0.0,
            cfa_score: 0.0,
            lens_score: 0.0,
        }
    }

    /// Compute the weighted overall score.
    #[must_use]
    pub fn weighted_score(&self, prnu_weight: f64, cfa_weight: f64, lens_weight: f64) -> f64 {
        let total_weight = prnu_weight + cfa_weight + lens_weight;
        if total_weight < 1e-10 {
            return 0.0;
        }
        (self.prnu_score * prnu_weight
            + self.cfa_score * cfa_weight
            + self.lens_score * lens_weight)
            / total_weight
    }
}

/// Database of known camera fingerprints.
#[derive(Debug, Clone)]
pub struct CameraDatabase {
    /// Known camera entries indexed by model name.
    entries: HashMap<String, CameraEntry>,
}

/// A single camera entry in the database.
#[derive(Debug, Clone)]
pub struct CameraEntry {
    /// Camera model name.
    pub model: String,
    /// Sensor type.
    pub sensor_type: SensorType,
    /// Expected CFA pattern.
    pub cfa_pattern: CfaPattern,
    /// Reference lens fingerprint.
    pub lens_fingerprint: Option<LensFingerprint>,
    /// Native resolution (width, height).
    pub native_resolution: (u32, u32),
}

impl CameraEntry {
    /// Create a new camera entry.
    #[must_use]
    pub fn new(
        model: &str,
        sensor_type: SensorType,
        cfa_pattern: CfaPattern,
        resolution: (u32, u32),
    ) -> Self {
        Self {
            model: model.to_string(),
            sensor_type,
            cfa_pattern,
            lens_fingerprint: None,
            native_resolution: resolution,
        }
    }
}

impl CameraDatabase {
    /// Create a new empty camera database.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Add a camera entry to the database.
    pub fn add_entry(&mut self, entry: CameraEntry) {
        self.entries.insert(entry.model.clone(), entry);
    }

    /// Look up a camera by model name.
    #[must_use]
    pub fn lookup(&self, model: &str) -> Option<&CameraEntry> {
        self.entries.get(model)
    }

    /// Return the number of entries in the database.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the database is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Find the best matching camera for a given CFA analysis and lens fingerprint.
    #[must_use]
    pub fn find_best_match(
        &self,
        cfa: &CfaAnalysis,
        lens: &LensFingerprint,
    ) -> Option<CameraMatch> {
        let mut best: Option<CameraMatch> = None;

        for entry in self.entries.values() {
            let cfa_score = if entry.cfa_pattern == cfa.pattern {
                cfa.confidence
            } else {
                0.0
            };

            let lens_score = entry
                .lens_fingerprint
                .as_ref()
                .map_or(0.0, |ref_lens| lens.similarity(ref_lens));

            let overall = cfa_score * 0.4 + lens_score * 0.6;

            let should_replace = match &best {
                Some(current) => overall > current.confidence,
                None => overall > 0.1,
            };

            if should_replace {
                let mut m = CameraMatch::new(&entry.model, overall);
                m.cfa_score = cfa_score;
                m.lens_score = lens_score;
                best = Some(m);
            }
        }

        best
    }
}

impl Default for CameraDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive source camera identification result.
#[derive(Debug, Clone)]
pub struct SourceCameraResult {
    /// Best camera match (if any).
    pub best_match: Option<CameraMatch>,
    /// All candidate matches sorted by confidence.
    pub candidates: Vec<CameraMatch>,
    /// CFA analysis result.
    pub cfa_analysis: Option<CfaAnalysis>,
    /// Lens fingerprint.
    pub lens_fingerprint: Option<LensFingerprint>,
    /// Detected sensor type.
    pub sensor_type: SensorType,
    /// Textual findings.
    pub findings: Vec<String>,
}

impl SourceCameraResult {
    /// Create a new empty result.
    #[must_use]
    pub fn new() -> Self {
        Self {
            best_match: None,
            candidates: Vec::new(),
            cfa_analysis: None,
            lens_fingerprint: None,
            sensor_type: SensorType::Unknown,
            findings: Vec::new(),
        }
    }

    /// Add a finding.
    pub fn add_finding(&mut self, finding: &str) {
        self.findings.push(finding.to_string());
    }

    /// Whether a camera was identified.
    #[must_use]
    pub fn is_identified(&self) -> bool {
        self.best_match
            .as_ref()
            .map_or(false, |m| m.confidence > 0.5)
    }
}

impl Default for SourceCameraResult {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensor_type_labels() {
        assert_eq!(SensorType::Ccd.label(), "CCD");
        assert_eq!(SensorType::CmosRolling.label(), "CMOS (Rolling)");
        assert_eq!(SensorType::BsiCmos.label(), "BSI CMOS");
        assert_eq!(SensorType::Unknown.label(), "Unknown");
    }

    #[test]
    fn test_cfa_pattern_labels() {
        assert_eq!(CfaPattern::BayerRggb.label(), "Bayer RGGB");
        assert_eq!(CfaPattern::XTrans.label(), "X-Trans");
        assert_eq!(CfaPattern::None.label(), "None");
    }

    #[test]
    fn test_cfa_analysis_artifacts_present() {
        let cfa = CfaAnalysis::new(CfaPattern::BayerRggb, 0.1, 0.8);
        assert!(cfa.artifacts_present());

        let cfa2 = CfaAnalysis::new(CfaPattern::BayerRggb, 0.001, 0.8);
        assert!(!cfa2.artifacts_present());
    }

    #[test]
    fn test_cfa_analysis_low_confidence() {
        let cfa = CfaAnalysis::new(CfaPattern::BayerRggb, 0.5, 0.3);
        assert!(!cfa.artifacts_present());
    }

    #[test]
    fn test_lens_fingerprint_self_similarity() {
        let lf = LensFingerprint::new([0.01, -0.02, 0.001], 0.5, 0.8);
        assert!((lf.similarity(&lf) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lens_fingerprint_different() {
        let lf1 = LensFingerprint::new([0.01, -0.02, 0.001], 0.5, 0.8);
        let lf2 = LensFingerprint::new([0.1, -0.2, 0.05], 2.0, 0.3);
        let sim = lf1.similarity(&lf2);
        assert!(sim < 0.5);
        assert!(sim > 0.0);
    }

    #[test]
    fn test_lens_low_distortion() {
        let lf = LensFingerprint::new([0.0001, -0.0005, 0.0002], 0.5, 0.8);
        assert!(lf.is_low_distortion());

        let lf2 = LensFingerprint::new([0.01, -0.02, 0.001], 0.5, 0.8);
        assert!(!lf2.is_low_distortion());
    }

    #[test]
    fn test_camera_match_weighted_score() {
        let mut m = CameraMatch::new("TestCam", 0.8);
        m.prnu_score = 0.9;
        m.cfa_score = 0.7;
        m.lens_score = 0.8;
        let score = m.weighted_score(1.0, 1.0, 1.0);
        assert!((score - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_camera_match_zero_weights() {
        let m = CameraMatch::new("TestCam", 0.8);
        assert!((m.weighted_score(0.0, 0.0, 0.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_camera_database_add_lookup() {
        let mut db = CameraDatabase::new();
        db.add_entry(CameraEntry::new(
            "Canon EOS R5",
            SensorType::CmosRolling,
            CfaPattern::BayerRggb,
            (8192, 5464),
        ));
        assert_eq!(db.len(), 1);
        assert!(!db.is_empty());
        assert!(db.lookup("Canon EOS R5").is_some());
        assert!(db.lookup("Nikon Z9").is_none());
    }

    #[test]
    fn test_camera_database_empty() {
        let db = CameraDatabase::new();
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_camera_database_find_best_match() {
        let mut db = CameraDatabase::new();
        let mut entry = CameraEntry::new(
            "Canon EOS R5",
            SensorType::CmosRolling,
            CfaPattern::BayerRggb,
            (8192, 5464),
        );
        entry.lens_fingerprint = Some(LensFingerprint::new([0.01, -0.02, 0.001], 0.5, 0.8));
        db.add_entry(entry);

        let cfa = CfaAnalysis::new(CfaPattern::BayerRggb, 0.1, 0.9);
        let lens = LensFingerprint::new([0.01, -0.02, 0.001], 0.5, 0.8);
        let best = db.find_best_match(&cfa, &lens);
        assert!(best.is_some());
        assert_eq!(
            best.expect("test expectation failed").camera_model,
            "Canon EOS R5"
        );
    }

    #[test]
    fn test_source_camera_result_not_identified() {
        let result = SourceCameraResult::new();
        assert!(!result.is_identified());
    }

    #[test]
    fn test_source_camera_result_identified() {
        let mut result = SourceCameraResult::new();
        result.best_match = Some(CameraMatch::new("TestCam", 0.8));
        assert!(result.is_identified());
    }

    #[test]
    fn test_source_camera_result_findings() {
        let mut result = SourceCameraResult::new();
        result.add_finding("CFA pattern detected: Bayer RGGB");
        result.add_finding("Lens distortion matches Canon 24-70mm");
        assert_eq!(result.findings.len(), 2);
    }
}
