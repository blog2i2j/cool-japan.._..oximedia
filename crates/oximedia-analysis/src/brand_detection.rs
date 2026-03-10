//! Logo and brand detection for video content.
//!
//! This module provides tools to detect brand logos and measure brand exposure
//! in video streams:
//! - **Color Signature Extraction** - 8-bin normalized color histogram
//! - **Brand Matching** - Cosine similarity between color signatures
//! - **Exposure Reporting** - Frame counts, screen time, confidence

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────
// BrandMatch
// ─────────────────────────────────────────────────────────────

/// A single brand detection event in one frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrandMatch {
    /// Name of the detected brand
    pub brand_name: String,
    /// Detection confidence (0.0–1.0)
    pub confidence: f32,
    /// Bounding region `(x, y, width, height)` in pixels
    pub region: (u32, u32, u32, u32),
    /// Frame index where this match was found
    pub frame_idx: u64,
}

impl BrandMatch {
    /// Returns true if the match exceeds the given confidence threshold.
    #[must_use]
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

// ─────────────────────────────────────────────────────────────
// BrandTemplate
// ─────────────────────────────────────────────────────────────

/// A brand template stored in the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrandTemplate {
    /// Brand / logo name
    pub name: String,
    /// 64-bit feature hash for quick rejection
    pub feature_hash: u64,
    /// 8-bin normalized color histogram signature
    pub color_signature: [f32; 8],
}

impl BrandTemplate {
    /// Create a new brand template.
    #[must_use]
    pub fn new(name: impl Into<String>, feature_hash: u64, color_signature: [f32; 8]) -> Self {
        Self {
            name: name.into(),
            feature_hash,
            color_signature,
        }
    }

    /// Compute the L2 norm of the color signature.
    #[must_use]
    pub fn signature_norm(&self) -> f32 {
        self.color_signature
            .iter()
            .map(|&v| v * v)
            .sum::<f32>()
            .sqrt()
    }
}

// ─────────────────────────────────────────────────────────────
// BrandDatabase
// ─────────────────────────────────────────────────────────────

/// A simple in-memory database of brand templates.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BrandDatabase {
    /// Stored templates
    pub templates: Vec<BrandTemplate>,
}

impl BrandDatabase {
    /// Create an empty database.
    #[must_use]
    pub fn new() -> Self {
        Self {
            templates: Vec::new(),
        }
    }

    /// Add a template to the database.
    pub fn add_template(&mut self, template: BrandTemplate) {
        self.templates.push(template);
    }

    /// Find a template by exact brand name (case-insensitive).
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<&BrandTemplate> {
        let lower = name.to_lowercase();
        self.templates
            .iter()
            .find(|t| t.name.to_lowercase() == lower)
    }

    /// Returns the number of templates in the database.
    #[must_use]
    pub fn count(&self) -> usize {
        self.templates.len()
    }

    /// Returns true if the database is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────
// ColorSignatureExtractor
// ─────────────────────────────────────────────────────────────

/// Extracts an 8-bin normalized color histogram from an RGB patch.
pub struct ColorSignatureExtractor;

impl ColorSignatureExtractor {
    /// Extract an 8-bin normalized color histogram from an RGB patch.
    ///
    /// * `rgb_patch` – interleaved RGB values in \[0, 1\]
    /// * `width` – patch width in pixels
    /// * `height` – patch height in pixels
    ///
    /// Returns a normalized 8-element array (bins sum to ≤ 1.0 per channel pair).
    #[must_use]
    pub fn extract(rgb_patch: &[f32], width: u32, height: u32) -> [f32; 8] {
        let pixel_count = (width * height) as usize;
        if rgb_patch.len() < pixel_count * 3 || pixel_count == 0 {
            return [0.0; 8];
        }

        // 8 bins: split across R (4 bins) and G (2 bins) and B (2 bins)
        // Bin assignment: combine quantised R and G
        let mut counts = [0u32; 8];

        for i in 0..pixel_count {
            let r = rgb_patch[i * 3].clamp(0.0, 1.0);
            let g = rgb_patch[i * 3 + 1].clamp(0.0, 1.0);
            let b = rgb_patch[i * 3 + 2].clamp(0.0, 1.0);

            // Quantise each channel to 2 bins and combine into a 3-bit index
            let ri = usize::from(r >= 0.5);
            let gi = usize::from(g >= 0.5);
            let bi = usize::from(b >= 0.5);
            let bin = (ri << 2) | (gi << 1) | bi;
            counts[bin] += 1;
        }

        let total = pixel_count as f32;
        let mut sig = [0.0f32; 8];
        for (i, &c) in counts.iter().enumerate() {
            sig[i] = c as f32 / total;
        }
        sig
    }
}

// ─────────────────────────────────────────────────────────────
// BrandMatcher
// ─────────────────────────────────────────────────────────────

/// Matches a region color signature against a `BrandDatabase`.
pub struct BrandMatcher;

impl BrandMatcher {
    /// Match a region signature against all templates using cosine similarity.
    ///
    /// Returns the best `BrandMatch` if cosine similarity exceeds 0.80.
    #[must_use]
    pub fn match_region(region_signature: &[f32; 8], db: &BrandDatabase) -> Option<BrandMatch> {
        if db.is_empty() {
            return None;
        }

        let mut best_sim = 0.0f32;
        let mut best_template: Option<&BrandTemplate> = None;

        let region_norm = region_signature.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if region_norm < 1e-8 {
            return None;
        }

        for template in &db.templates {
            let t_norm = template.signature_norm();
            if t_norm < 1e-8 {
                continue;
            }
            let dot: f32 = region_signature
                .iter()
                .zip(template.color_signature.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            let sim = (dot / (region_norm * t_norm)).clamp(0.0, 1.0);
            if sim > best_sim {
                best_sim = sim;
                best_template = Some(template);
            }
        }

        const MATCH_THRESHOLD: f32 = 0.80;
        best_template
            .filter(|_| best_sim >= MATCH_THRESHOLD)
            .map(|t| {
                BrandMatch {
                    brand_name: t.name.clone(),
                    confidence: best_sim,
                    region: (0, 0, 0, 0), // caller fills in actual coordinates
                    frame_idx: 0,
                }
            })
    }

    /// Match a region with explicit position and frame information.
    #[must_use]
    pub fn match_region_at(
        region_signature: &[f32; 8],
        db: &BrandDatabase,
        region: (u32, u32, u32, u32),
        frame_idx: u64,
    ) -> Option<BrandMatch> {
        Self::match_region(region_signature, db).map(|mut m| {
            m.region = region;
            m.frame_idx = frame_idx;
            m
        })
    }
}

// ─────────────────────────────────────────────────────────────
// BrandExposureReport
// ─────────────────────────────────────────────────────────────

/// Aggregate brand exposure report across multiple frames.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrandExposureReport {
    /// Brand name
    pub brand_name: String,
    /// Total number of frames in the video
    pub total_frames: u64,
    /// Cumulative on-screen time in seconds
    pub screen_time_secs: f64,
    /// Average detection confidence
    pub avg_confidence: f32,
    /// Frame indices where the brand was most prominent
    pub prominent_frames: Vec<u64>,
}

impl BrandExposureReport {
    /// Build a report from a list of `BrandMatch` events and frame rate.
    ///
    /// * `matches` – all matches for this brand
    /// * `total_frames` – total number of frames in the video
    /// * `fps` – frames per second
    #[must_use]
    pub fn from_matches(
        brand_name: impl Into<String>,
        matches: &[BrandMatch],
        total_frames: u64,
        fps: f64,
    ) -> Self {
        if matches.is_empty() {
            return Self {
                brand_name: brand_name.into(),
                total_frames,
                screen_time_secs: 0.0,
                avg_confidence: 0.0,
                prominent_frames: Vec::new(),
            };
        }

        let screen_time_secs = matches.len() as f64 / fps.max(1.0);
        let avg_confidence =
            matches.iter().map(|m| m.confidence).sum::<f32>() / matches.len() as f32;

        // Select frames with confidence above average as prominent
        let mut prominent_frames: Vec<u64> = matches
            .iter()
            .filter(|m| m.confidence >= avg_confidence)
            .map(|m| m.frame_idx)
            .collect();
        prominent_frames.sort_unstable();
        prominent_frames.dedup();

        Self {
            brand_name: brand_name.into(),
            total_frames,
            screen_time_secs,
            avg_confidence,
            prominent_frames,
        }
    }

    /// Returns the fraction of total frames where this brand was visible.
    #[must_use]
    pub fn exposure_ratio(&self) -> f64 {
        if self.total_frames == 0 {
            return 0.0;
        }
        self.prominent_frames.len() as f64 / self.total_frames as f64
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signature(dominant_bin: usize) -> [f32; 8] {
        let mut sig = [0.0f32; 8];
        sig[dominant_bin] = 1.0;
        sig
    }

    // ── BrandTemplate ─────────────────────────────────────────

    #[test]
    fn test_brand_template_norm() {
        let mut sig = [0.0f32; 8];
        sig[0] = 0.6;
        sig[1] = 0.8;
        let t = BrandTemplate::new("Test", 0, sig);
        let norm = t.signature_norm();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    // ── BrandDatabase ─────────────────────────────────────────

    #[test]
    fn test_database_add_and_count() {
        let mut db = BrandDatabase::new();
        assert_eq!(db.count(), 0);
        db.add_template(BrandTemplate::new("Nike", 0x1234, make_signature(3)));
        assert_eq!(db.count(), 1);
    }

    #[test]
    fn test_database_find_by_name() {
        let mut db = BrandDatabase::new();
        db.add_template(BrandTemplate::new("Nike", 0, make_signature(3)));
        assert!(db.find_by_name("Nike").is_some());
        assert!(db.find_by_name("nike").is_some()); // case-insensitive
        assert!(db.find_by_name("Adidas").is_none());
    }

    #[test]
    fn test_database_is_empty() {
        let db = BrandDatabase::new();
        assert!(db.is_empty());
    }

    // ── ColorSignatureExtractor ───────────────────────────────

    #[test]
    fn test_extract_uniform_red() {
        // 2×2 patch of pure red (r=1, g=0, b=0): 4 pixels × 3 channels = 12 values
        // Each pixel: [1.0, 0.0, 0.0]
        let mut patch = Vec::with_capacity(12);
        for _ in 0..4 {
            patch.extend_from_slice(&[1.0f32, 0.0, 0.0]);
        }
        let sig = ColorSignatureExtractor::extract(&patch, 2, 2);
        // bin 4 = (ri=1, gi=0, bi=0) → (1<<2)|(0<<1)|0 = 4
        assert!((sig[4] - 1.0).abs() < 1e-6);
        let total: f32 = sig.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_extract_empty_patch() {
        let sig = ColorSignatureExtractor::extract(&[], 0, 0);
        assert_eq!(sig, [0.0; 8]);
    }

    #[test]
    fn test_extract_normalised() {
        let patch: Vec<f32> = (0..9).map(|i| (i % 3) as f32 * 0.5).collect();
        let sig = ColorSignatureExtractor::extract(&patch, 3, 1);
        let total: f32 = sig.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    // ── BrandMatcher ─────────────────────────────────────────

    #[test]
    fn test_match_identical_signature() {
        let sig = make_signature(5);
        let mut db = BrandDatabase::new();
        db.add_template(BrandTemplate::new("Brand", 0, sig));
        let result = BrandMatcher::match_region(&sig, &db);
        assert!(result.is_some());
        let m = result.expect("expected successful result");
        assert_eq!(m.brand_name, "Brand");
        assert!((m.confidence - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_match_different_signature() {
        let sig_brand = make_signature(2);
        let sig_query = make_signature(7);
        let mut db = BrandDatabase::new();
        db.add_template(BrandTemplate::new("Brand", 0, sig_brand));
        let result = BrandMatcher::match_region(&sig_query, &db);
        assert!(result.is_none()); // cosine = 0
    }

    #[test]
    fn test_match_empty_db() {
        let sig = make_signature(0);
        let db = BrandDatabase::new();
        let result = BrandMatcher::match_region(&sig, &db);
        assert!(result.is_none());
    }

    #[test]
    fn test_match_region_at() {
        let sig = make_signature(3);
        let mut db = BrandDatabase::new();
        db.add_template(BrandTemplate::new("TestBrand", 0, sig));
        let result = BrandMatcher::match_region_at(&sig, &db, (10, 20, 50, 50), 42);
        assert!(result.is_some());
        let m = result.expect("expected successful result");
        assert_eq!(m.frame_idx, 42);
        assert_eq!(m.region, (10, 20, 50, 50));
    }

    // ── BrandExposureReport ───────────────────────────────────

    #[test]
    fn test_exposure_report_empty_matches() {
        let report = BrandExposureReport::from_matches("Nike", &[], 1000, 30.0);
        assert_eq!(report.screen_time_secs, 0.0);
        assert_eq!(report.avg_confidence, 0.0);
        assert!(report.prominent_frames.is_empty());
    }

    #[test]
    fn test_exposure_report_screen_time() {
        let matches: Vec<BrandMatch> = (0..30)
            .map(|i| BrandMatch {
                brand_name: "Nike".to_string(),
                confidence: 0.9,
                region: (0, 0, 50, 50),
                frame_idx: i,
            })
            .collect();
        let report = BrandExposureReport::from_matches("Nike", &matches, 1000, 30.0);
        assert!((report.screen_time_secs - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_exposure_ratio() {
        let matches: Vec<BrandMatch> = (0..10)
            .map(|i| BrandMatch {
                brand_name: "X".to_string(),
                confidence: 0.95,
                region: (0, 0, 1, 1),
                frame_idx: i,
            })
            .collect();
        let report = BrandExposureReport::from_matches("X", &matches, 100, 30.0);
        let ratio = report.exposure_ratio();
        assert!(ratio >= 0.0 && ratio <= 1.0);
    }

    #[test]
    fn test_brand_match_is_confident() {
        let m = BrandMatch {
            brand_name: "A".to_string(),
            confidence: 0.85,
            region: (0, 0, 1, 1),
            frame_idx: 0,
        };
        assert!(m.is_confident(0.80));
        assert!(!m.is_confident(0.90));
    }
}
