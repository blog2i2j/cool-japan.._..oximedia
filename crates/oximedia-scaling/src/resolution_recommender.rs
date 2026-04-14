//! Adaptive resolution recommendation for video encoding.
//!
//! Maps content complexity metrics to optimal output resolutions given a
//! target bitrate budget and a device profile.  The design follows a
//! three-stage pipeline:
//!
//! 1. **Content analysis** — compute a `ComplexityScore` from perceptual
//!    metrics (spatial detail, temporal motion, noise level).
//! 2. **Bitrate mapping** — translate `(bitrate, complexity)` into a set of
//!    candidate resolutions using codec-specific efficiency curves.
//! 3. **Device matching** — filter the candidates against the capabilities
//!    described by a `DeviceProfile` (max resolution, max bitrate, codec
//!    support).
//!
//! The public entry point is `ResolutionRecommender::recommend`.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use thiserror::Error;

/// Errors that can arise during recommendation.
#[derive(Debug, Error, PartialEq)]
pub enum RecommenderError {
    /// Target bitrate was zero or negative.
    #[error("target bitrate {kbps} kbps is not positive")]
    InvalidBitrate {
        /// The invalid bitrate value (kbps).
        kbps: f64,
    },
    /// The source resolution has a zero axis.
    #[error("source resolution {width}x{height} has a zero axis")]
    InvalidSourceResolution {
        /// Source width.
        width: u32,
        /// Source height.
        height: u32,
    },
    /// No resolution candidate is compatible with the device profile.
    #[error("no compatible resolution found for device profile '{device}'")]
    NoCompatibleResolution {
        /// Name of the device profile that had no match.
        device: String,
    },
}

// ── Complexity scoring ─────────────────────────────────────────────────────────

/// Content complexity score derived from perceptual metrics.
///
/// All fields are normalised to the range **[0.0, 1.0]**.
#[derive(Debug, Clone, Copy)]
pub struct ComplexityScore {
    /// Spatial detail level: 0.0 = flat/simple, 1.0 = highly textured.
    pub spatial: f64,
    /// Temporal motion level: 0.0 = static, 1.0 = fast motion throughout.
    pub temporal: f64,
    /// Noise level: 0.0 = clean, 1.0 = very noisy (grain, sensor noise).
    pub noise: f64,
}

impl ComplexityScore {
    /// Create a new complexity score with values clamped to [0, 1].
    pub fn new(spatial: f64, temporal: f64, noise: f64) -> Self {
        Self {
            spatial: spatial.clamp(0.0, 1.0),
            temporal: temporal.clamp(0.0, 1.0),
            noise: noise.clamp(0.0, 1.0),
        }
    }

    /// Weighted aggregate score.
    ///
    /// Spatial and temporal contribute 40 % each; noise contributes 20 %.
    pub fn aggregate(&self) -> f64 {
        (self.spatial * 0.4 + self.temporal * 0.4 + self.noise * 0.2).clamp(0.0, 1.0)
    }

    /// Compute a complexity score from a flat luminance buffer using simple
    /// statistics.
    ///
    /// This is a lightweight heuristic — for production use a proper VIF/VMAF
    /// analysis should be preferred.
    ///
    /// - Spatial: gradient magnitude (Sobel-like 2×2).
    /// - Temporal: mean absolute difference between two consecutive frames.
    /// - Noise: local variance estimate.
    pub fn from_luma_frames(
        current: &[u8],
        previous: Option<&[u8]>,
        width: u32,
        height: u32,
    ) -> Self {
        if current.is_empty() || width == 0 || height == 0 {
            return Self::new(0.0, 0.0, 0.0);
        }

        let w = width as usize;
        let h = height as usize;
        let n = (w * h).min(current.len());

        // Spatial: average horizontal gradient magnitude.
        let mut grad_sum = 0u64;
        let mut grad_count = 0u64;
        for y in 0..h {
            for x in 0..w.saturating_sub(1) {
                let idx0 = y * w + x;
                let idx1 = y * w + x + 1;
                if idx1 < n {
                    let diff = (current[idx0] as i32 - current[idx1] as i32).unsigned_abs();
                    grad_sum += u64::from(diff);
                    grad_count += 1;
                }
            }
        }
        let spatial_raw = if grad_count > 0 {
            grad_sum as f64 / (grad_count as f64 * 255.0)
        } else {
            0.0
        };
        // Scale: typical natural images have mean gradient ~5–30/255 → map to [0,1]
        let spatial = (spatial_raw * 8.0).clamp(0.0, 1.0);

        // Temporal: mean absolute difference with previous frame.
        let temporal = if let Some(prev) = previous {
            if prev.len() >= n {
                let diff_sum: u64 = current[..n]
                    .iter()
                    .zip(prev[..n].iter())
                    .map(|(&a, &b)| u64::from((a as i32 - b as i32).unsigned_abs()))
                    .sum();
                let mean_diff = diff_sum as f64 / (n as f64 * 255.0);
                (mean_diff * 10.0).clamp(0.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Noise: local variance in 4×4 blocks.
        let block = 4usize;
        let mut var_sum = 0.0f64;
        let mut var_blocks = 0u32;
        let mut y = 0;
        while y + block <= h {
            let mut x = 0;
            while x + block <= w {
                let mut block_sum = 0u64;
                let mut block_sq = 0u64;
                for by in 0..block {
                    for bx in 0..block {
                        let v = current[(y + by) * w + (x + bx)] as u64;
                        block_sum += v;
                        block_sq += v * v;
                    }
                }
                let count = (block * block) as u64;
                let mean = block_sum as f64 / count as f64;
                let variance = (block_sq as f64 / count as f64) - mean * mean;
                var_sum += variance;
                var_blocks += 1;
                x += block;
            }
            y += block;
        }
        let noise_raw = if var_blocks > 0 {
            var_sum / (var_blocks as f64 * 255.0 * 255.0)
        } else {
            0.0
        };
        let noise = (noise_raw * 20.0).clamp(0.0, 1.0);

        Self::new(spatial, temporal, noise)
    }
}

impl Default for ComplexityScore {
    fn default() -> Self {
        Self::new(0.5, 0.3, 0.1)
    }
}

// ── Resolution candidates ──────────────────────────────────────────────────────

/// A candidate output resolution with associated quality metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolutionCandidate {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Minimum bitrate (kbps) required to encode this resolution at acceptable quality.
    pub min_bitrate_kbps: u32,
    /// Recommended bitrate (kbps) for good quality at this resolution.
    pub recommended_bitrate_kbps: u32,
    /// Human-readable name (e.g. `"1080p"`, `"720p"`).
    pub label: &'static str,
}

impl ResolutionCandidate {
    /// Total pixel count.
    pub fn pixels(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Pixel density relative to 1080p (1920×1080 = 2_073_600 px).
    pub fn relative_density(&self) -> f64 {
        self.pixels() as f64 / 2_073_600.0
    }
}

/// Standard AVC/HEVC resolution ladder with bitrate guidance.
pub fn standard_ladder() -> Vec<ResolutionCandidate> {
    vec![
        ResolutionCandidate {
            width: 426,
            height: 240,
            min_bitrate_kbps: 150,
            recommended_bitrate_kbps: 300,
            label: "240p",
        },
        ResolutionCandidate {
            width: 640,
            height: 360,
            min_bitrate_kbps: 300,
            recommended_bitrate_kbps: 600,
            label: "360p",
        },
        ResolutionCandidate {
            width: 854,
            height: 480,
            min_bitrate_kbps: 500,
            recommended_bitrate_kbps: 1000,
            label: "480p",
        },
        ResolutionCandidate {
            width: 1280,
            height: 720,
            min_bitrate_kbps: 1000,
            recommended_bitrate_kbps: 2500,
            label: "720p",
        },
        ResolutionCandidate {
            width: 1920,
            height: 1080,
            min_bitrate_kbps: 2000,
            recommended_bitrate_kbps: 5000,
            label: "1080p",
        },
        ResolutionCandidate {
            width: 2560,
            height: 1440,
            min_bitrate_kbps: 4000,
            recommended_bitrate_kbps: 10000,
            label: "1440p",
        },
        ResolutionCandidate {
            width: 3840,
            height: 2160,
            min_bitrate_kbps: 8000,
            recommended_bitrate_kbps: 20000,
            label: "4K",
        },
    ]
}

// ── Device profiles ────────────────────────────────────────────────────────────

/// Codec supported by a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportedCodec {
    /// H.264 / AVC.
    H264,
    /// H.265 / HEVC.
    H265,
    /// VP9 (patent-free).
    Vp9,
    /// AV1 (patent-free).
    Av1,
}

/// Device capability profile.
#[derive(Debug, Clone)]
pub struct DeviceProfile {
    /// Human-readable device name.
    pub name: String,
    /// Maximum resolution this device can decode efficiently.
    pub max_width: u32,
    /// Maximum height this device can decode efficiently.
    pub max_height: u32,
    /// Maximum bitrate (kbps) this device's connection can sustain.
    pub max_bitrate_kbps: u32,
    /// Supported codecs (at least one must match).
    pub codecs: Vec<SupportedCodec>,
}

impl DeviceProfile {
    /// Create a new device profile.
    pub fn new(
        name: impl Into<String>,
        max_width: u32,
        max_height: u32,
        max_bitrate_kbps: u32,
    ) -> Self {
        Self {
            name: name.into(),
            max_width,
            max_height,
            max_bitrate_kbps,
            codecs: vec![
                SupportedCodec::H264,
                SupportedCodec::Vp9,
                SupportedCodec::Av1,
            ],
        }
    }

    /// Pre-defined profile for mobile devices (720p, 4 Mbps).
    pub fn mobile() -> Self {
        Self::new("Mobile", 1280, 720, 4000)
    }

    /// Pre-defined profile for desktop browsers (4K, 40 Mbps).
    pub fn desktop() -> Self {
        Self::new("Desktop", 3840, 2160, 40000)
    }

    /// Pre-defined profile for smart TV / streaming devices (4K, 25 Mbps).
    pub fn smart_tv() -> Self {
        Self::new("SmartTV", 3840, 2160, 25000)
    }

    /// Pre-defined profile for low-bandwidth connections (480p, 1 Mbps).
    pub fn low_bandwidth() -> Self {
        Self::new("LowBandwidth", 854, 480, 1000)
    }

    /// Returns `true` if the given resolution fits within this profile's limits.
    pub fn can_display(&self, width: u32, height: u32) -> bool {
        width <= self.max_width && height <= self.max_height
    }
}

// ── Recommender ────────────────────────────────────────────────────────────────

/// Configuration for the [`ResolutionRecommender`].
#[derive(Debug, Clone)]
pub struct RecommenderConfig {
    /// Factor by which to inflate the minimum bitrate threshold when content
    /// complexity is high.  Default: 1.5.
    pub complexity_headroom: f64,
    /// Whether to allow downscaling below the source resolution.
    /// When `false`, the source resolution is used as an upper bound.
    pub allow_upscale: bool,
}

impl Default for RecommenderConfig {
    fn default() -> Self {
        Self {
            complexity_headroom: 1.5,
            allow_upscale: false,
        }
    }
}

/// Recommendation result returned by `ResolutionRecommender::recommend`.
#[derive(Debug, Clone)]
pub struct Recommendation {
    /// The recommended output resolution.
    pub candidate: ResolutionCandidate,
    /// Effective bitrate budget (kbps) after applying complexity headroom.
    pub effective_bitrate_kbps: f64,
    /// The complexity score that drove the decision.
    pub complexity: ComplexityScore,
    /// Explanation string for logging/debugging.
    pub rationale: String,
}

/// Recommends an output resolution given content complexity, a target bitrate,
/// and an optional device profile.
pub struct ResolutionRecommender {
    config: RecommenderConfig,
    ladder: Vec<ResolutionCandidate>,
}

impl ResolutionRecommender {
    /// Create a new recommender using the standard resolution ladder.
    pub fn new(config: RecommenderConfig) -> Self {
        Self {
            config,
            ladder: standard_ladder(),
        }
    }

    /// Create a recommender with a custom resolution ladder.
    pub fn with_ladder(config: RecommenderConfig, ladder: Vec<ResolutionCandidate>) -> Self {
        Self { config, ladder }
    }

    /// Recommend the best output resolution.
    ///
    /// # Arguments
    /// - `target_bitrate_kbps` — available bitrate budget.
    /// - `src_width` / `src_height` — source content dimensions.
    /// - `complexity` — pre-computed complexity score.
    /// - `device` — optional device profile to restrict candidates.
    pub fn recommend(
        &self,
        target_bitrate_kbps: f64,
        src_width: u32,
        src_height: u32,
        complexity: ComplexityScore,
        device: Option<&DeviceProfile>,
    ) -> Result<Recommendation, RecommenderError> {
        if target_bitrate_kbps <= 0.0 {
            return Err(RecommenderError::InvalidBitrate {
                kbps: target_bitrate_kbps,
            });
        }
        if src_width == 0 || src_height == 0 {
            return Err(RecommenderError::InvalidSourceResolution {
                width: src_width,
                height: src_height,
            });
        }

        // Inflate minimum bitrate thresholds by complexity headroom.
        let headroom = 1.0 + self.config.complexity_headroom * complexity.aggregate();
        let effective_bitrate = target_bitrate_kbps / headroom;

        // Build a list of eligible candidates.
        let source_pixels = u64::from(src_width) * u64::from(src_height);

        let eligible: Vec<&ResolutionCandidate> = self
            .ladder
            .iter()
            .filter(|c| {
                // Bitrate must meet the minimum threshold.
                target_bitrate_kbps >= c.min_bitrate_kbps as f64
                    // Effective bitrate must meet minimum.
                    && effective_bitrate >= c.min_bitrate_kbps as f64
                    // Do not upscale unless explicitly allowed.
                    && (self.config.allow_upscale || c.pixels() <= source_pixels)
                    // Device resolution limit.
                    && device.map_or(true, |d| d.can_display(c.width, c.height))
                    // Device bitrate limit.
                    && device.map_or(true, |d| {
                        c.min_bitrate_kbps <= d.max_bitrate_kbps
                    })
            })
            .collect();

        // Pick the highest resolution among eligible candidates.
        let best = eligible
            .into_iter()
            .max_by_key(|c| c.pixels())
            .ok_or_else(|| RecommenderError::NoCompatibleResolution {
                device: device.map_or_else(|| "none".into(), |d| d.name.clone()),
            })?;

        let rationale =
            format!(
            "Selected {} ({}x{}) — effective_bitrate={:.0} kbps, complexity={:.2}, headroom={:.2}x",
            best.label, best.width, best.height, effective_bitrate, complexity.aggregate(), headroom
        );

        Ok(Recommendation {
            candidate: best.clone(),
            effective_bitrate_kbps: effective_bitrate,
            complexity,
            rationale,
        })
    }

    /// Recommend a ranked list of up to `max_count` candidates (best first).
    pub fn recommend_ranked(
        &self,
        target_bitrate_kbps: f64,
        src_width: u32,
        src_height: u32,
        complexity: ComplexityScore,
        device: Option<&DeviceProfile>,
        max_count: usize,
    ) -> Result<Vec<Recommendation>, RecommenderError> {
        if target_bitrate_kbps <= 0.0 {
            return Err(RecommenderError::InvalidBitrate {
                kbps: target_bitrate_kbps,
            });
        }
        if src_width == 0 || src_height == 0 {
            return Err(RecommenderError::InvalidSourceResolution {
                width: src_width,
                height: src_height,
            });
        }

        let headroom = 1.0 + self.config.complexity_headroom * complexity.aggregate();
        let effective_bitrate = target_bitrate_kbps / headroom;
        let source_pixels = u64::from(src_width) * u64::from(src_height);

        let mut eligible: Vec<&ResolutionCandidate> = self
            .ladder
            .iter()
            .filter(|c| {
                target_bitrate_kbps >= c.min_bitrate_kbps as f64
                    && effective_bitrate >= c.min_bitrate_kbps as f64
                    && (self.config.allow_upscale || c.pixels() <= source_pixels)
                    && device.map_or(true, |d| d.can_display(c.width, c.height))
                    && device.map_or(true, |d| c.min_bitrate_kbps <= d.max_bitrate_kbps)
            })
            .collect();

        // Sort descending by resolution.
        eligible.sort_by(|a, b| b.pixels().cmp(&a.pixels()));

        let results = eligible
            .into_iter()
            .take(max_count)
            .map(|c| {
                let rationale = format!(
                    "{} ({}x{}) — effective={:.0} kbps",
                    c.label, c.width, c.height, effective_bitrate
                );
                Recommendation {
                    candidate: c.clone(),
                    effective_bitrate_kbps: effective_bitrate,
                    complexity,
                    rationale,
                }
            })
            .collect();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_complexity() -> ComplexityScore {
        ComplexityScore::new(0.3, 0.2, 0.1)
    }

    // ── ComplexityScore tests ─────────────────────────────────────────────────

    #[test]
    fn test_complexity_clamping() {
        let c = ComplexityScore::new(2.0, -1.0, 0.5);
        assert_eq!(c.spatial, 1.0);
        assert_eq!(c.temporal, 0.0);
        assert_eq!(c.noise, 0.5);
    }

    #[test]
    fn test_complexity_aggregate_zero() {
        let c = ComplexityScore::new(0.0, 0.0, 0.0);
        assert_eq!(c.aggregate(), 0.0);
    }

    #[test]
    fn test_complexity_aggregate_one() {
        let c = ComplexityScore::new(1.0, 1.0, 1.0);
        assert!((c.aggregate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_complexity_aggregate_weighted() {
        // spatial=1, temporal=0, noise=0 → 0.4
        let c = ComplexityScore::new(1.0, 0.0, 0.0);
        assert!((c.aggregate() - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_from_luma_frames_empty() {
        let c = ComplexityScore::from_luma_frames(&[], None, 0, 0);
        assert_eq!(c.spatial, 0.0);
        assert_eq!(c.temporal, 0.0);
    }

    #[test]
    fn test_from_luma_frames_flat() {
        // Flat grey image → near-zero spatial complexity.
        let luma = vec![128u8; 64 * 64];
        let c = ComplexityScore::from_luma_frames(&luma, None, 64, 64);
        assert!(
            c.spatial < 0.05,
            "flat image should have low spatial: {}",
            c.spatial
        );
    }

    #[test]
    fn test_from_luma_frames_gradient() {
        // Image with strong horizontal gradient.
        let mut luma = vec![0u8; 64 * 64];
        for y in 0..64usize {
            for x in 0..64usize {
                luma[y * 64 + x] = x as u8 * 4;
            }
        }
        let c = ComplexityScore::from_luma_frames(&luma, None, 64, 64);
        assert!(
            c.spatial > 0.1,
            "gradient image should have higher spatial: {}",
            c.spatial
        );
    }

    // ── DeviceProfile tests ───────────────────────────────────────────────────

    #[test]
    fn test_device_can_display() {
        let d = DeviceProfile::mobile();
        assert!(d.can_display(1280, 720));
        assert!(!d.can_display(1920, 1080));
    }

    #[test]
    fn test_device_profiles_predefined() {
        let mobile = DeviceProfile::mobile();
        assert_eq!(mobile.max_width, 1280);
        let desktop = DeviceProfile::desktop();
        assert_eq!(desktop.max_width, 3840);
    }

    // ── Recommender tests ─────────────────────────────────────────────────────

    #[test]
    fn test_recommend_basic_1080p() {
        let rec = ResolutionRecommender::new(RecommenderConfig::default());
        let r = rec
            .recommend(5000.0, 1920, 1080, simple_complexity(), None)
            .unwrap();
        assert_eq!(r.candidate.label, "1080p");
    }

    #[test]
    fn test_recommend_low_bitrate_picks_lower_res() {
        let rec = ResolutionRecommender::new(RecommenderConfig::default());
        let r = rec
            .recommend(400.0, 1920, 1080, simple_complexity(), None)
            .unwrap();
        // At 400 kbps only 240p or 360p should be eligible.
        assert!(
            r.candidate.min_bitrate_kbps <= 400,
            "candidate min_bitrate {} should be ≤ 400",
            r.candidate.min_bitrate_kbps
        );
    }

    #[test]
    fn test_recommend_zero_bitrate_error() {
        let rec = ResolutionRecommender::new(RecommenderConfig::default());
        let err = rec
            .recommend(0.0, 1920, 1080, simple_complexity(), None)
            .unwrap_err();
        assert!(matches!(err, RecommenderError::InvalidBitrate { .. }));
    }

    #[test]
    fn test_recommend_zero_source_error() {
        let rec = ResolutionRecommender::new(RecommenderConfig::default());
        let err = rec
            .recommend(5000.0, 0, 1080, simple_complexity(), None)
            .unwrap_err();
        assert!(matches!(
            err,
            RecommenderError::InvalidSourceResolution { .. }
        ));
    }

    #[test]
    fn test_recommend_with_mobile_device_caps_at_720p() {
        let rec = ResolutionRecommender::new(RecommenderConfig::default());
        let mobile = DeviceProfile::mobile();
        let r = rec
            .recommend(10000.0, 1920, 1080, simple_complexity(), Some(&mobile))
            .unwrap();
        assert!(
            r.candidate.width <= 1280,
            "mobile should cap at 720p, got {}",
            r.candidate.label
        );
    }

    #[test]
    fn test_recommend_ranked_returns_sorted() {
        let rec = ResolutionRecommender::new(RecommenderConfig::default());
        let ranked = rec
            .recommend_ranked(20000.0, 3840, 2160, simple_complexity(), None, 3)
            .unwrap();
        assert!(!ranked.is_empty());
        for w in ranked.windows(2) {
            assert!(
                w[0].candidate.pixels() >= w[1].candidate.pixels(),
                "results should be sorted descending"
            );
        }
    }

    #[test]
    fn test_recommend_no_upscale_by_default() {
        // Source is 480p — 720p should not be recommended.
        let rec = ResolutionRecommender::new(RecommenderConfig::default());
        let r = rec
            .recommend(5000.0, 854, 480, simple_complexity(), None)
            .unwrap();
        assert!(
            r.candidate.pixels() <= u64::from(854_u32) * u64::from(480_u32),
            "should not upscale above 480p source, got {}",
            r.candidate.label
        );
    }

    #[test]
    fn test_standard_ladder_ordered() {
        let ladder = standard_ladder();
        for w in ladder.windows(2) {
            assert!(
                w[0].pixels() <= w[1].pixels(),
                "ladder should be in ascending resolution order"
            );
        }
    }
}
