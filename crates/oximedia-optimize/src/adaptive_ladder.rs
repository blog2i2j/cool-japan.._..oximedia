//! Adaptive bitrate ladder optimization for streaming workflows.
//!
//! This module provides tools to optimize ABR (Adaptive Bitrate) ladder configurations
//! for HLS, DASH, and similar streaming protocols.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A single rung of an adaptive bitrate ladder.
#[derive(Debug, Clone, PartialEq)]
pub struct LadderRung {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Target bitrate in kilobits per second.
    pub bitrate_kbps: u32,
    /// Codec identifier (e.g. "h264", "h265", "av1").
    pub codec: String,
    /// Target frame rate.
    pub frame_rate: f32,
}

impl LadderRung {
    /// Returns a human-readable resolution name (e.g. "1080p", "720p").
    #[must_use]
    pub fn resolution_name(&self) -> String {
        match self.height {
            h if h >= 2160 => "4K".to_string(),
            h if h >= 1440 => "1440p".to_string(),
            h if h >= 1080 => "1080p".to_string(),
            h if h >= 720 => "720p".to_string(),
            h if h >= 480 => "480p".to_string(),
            h if h >= 360 => "360p".to_string(),
            h if h >= 240 => "240p".to_string(),
            _ => format!("{}p", self.height),
        }
    }

    /// Returns `true` if this rung is HD (720p or higher).
    #[must_use]
    pub fn is_hd(&self) -> bool {
        self.height >= 720
    }

    /// Returns the bits per pixel at this rung's settings.
    #[must_use]
    pub fn bits_per_pixel(&self) -> f32 {
        let pixels_per_frame = (self.width * self.height) as f32;
        let bits_per_sec = (self.bitrate_kbps * 1000) as f32;
        if self.frame_rate > 0.0 && pixels_per_frame > 0.0 {
            bits_per_sec / (pixels_per_frame * self.frame_rate)
        } else {
            0.0
        }
    }
}

/// Optimizer that selects an appropriate set of ladder rungs for a given source.
#[derive(Debug, Clone)]
pub struct LadderOptimizer {
    /// Minimum number of rungs to keep in the ladder.
    pub min_rungs: u8,
    /// Maximum number of rungs allowed.
    pub max_rungs: u8,
    /// Bits-per-pixel threshold below which a rung is considered too complex to include.
    pub complexity_threshold: f32,
}

impl Default for LadderOptimizer {
    fn default() -> Self {
        Self {
            min_rungs: 2,
            max_rungs: 6,
            complexity_threshold: 0.05,
        }
    }
}

impl LadderOptimizer {
    /// Filters ladder rungs, keeping those whose bits-per-pixel exceeds the
    /// complexity threshold. Always retains at least `min_rungs` entries.
    #[must_use]
    pub fn optimize_ladder(
        &self,
        _source_bitrate_kbps: u32,
        rungs: &[LadderRung],
    ) -> Vec<LadderRung> {
        let mut filtered: Vec<LadderRung> = rungs
            .iter()
            .filter(|r| r.bits_per_pixel() >= self.complexity_threshold)
            .cloned()
            .collect();

        // Ensure minimum rung count by taking from the original list if needed
        if filtered.len() < self.min_rungs as usize && !rungs.is_empty() {
            for rung in rungs.iter().rev() {
                if filtered.len() >= self.min_rungs as usize {
                    break;
                }
                if !filtered.iter().any(|r| r.height == rung.height) {
                    filtered.push(rung.clone());
                }
            }
        }

        // Respect max_rungs limit
        filtered.truncate(self.max_rungs as usize);
        filtered
    }

    /// Returns the recommended number of rungs based on content complexity (0.0–1.0).
    #[must_use]
    pub fn recommended_rung_count(&self, complexity: f32) -> u8 {
        let range = (self.max_rungs - self.min_rungs) as f32;
        let count = self.min_rungs as f32 + complexity.clamp(0.0, 1.0) * range;
        count.round() as u8
    }
}

/// Model of available bandwidth used to select the best rung.
#[derive(Debug, Clone)]
pub struct BandwidthModel {
    /// Peak bandwidth in Mbit/s.
    pub peak_mbps: f32,
    /// Average (mean) bandwidth in Mbit/s.
    pub average_mbps: f32,
    /// 95th-percentile bandwidth in Mbit/s.
    pub percentile_95: f32,
}

impl BandwidthModel {
    /// Selects the highest ladder rung whose bitrate fits within the average bandwidth.
    #[must_use]
    pub fn select_rung<'a>(&self, ladder: &'a [LadderRung]) -> Option<&'a LadderRung> {
        let avg_kbps = (self.average_mbps * 1000.0) as u32;
        ladder
            .iter()
            .filter(|r| r.bitrate_kbps <= avg_kbps)
            .max_by_key(|r| r.bitrate_kbps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rung(width: u32, height: u32, bitrate_kbps: u32, fps: f32) -> LadderRung {
        LadderRung {
            width,
            height,
            bitrate_kbps,
            codec: "h264".to_string(),
            frame_rate: fps,
        }
    }

    #[test]
    fn test_resolution_name_4k() {
        let rung = make_rung(3840, 2160, 15000, 30.0);
        assert_eq!(rung.resolution_name(), "4K");
    }

    #[test]
    fn test_resolution_name_1080p() {
        let rung = make_rung(1920, 1080, 5000, 30.0);
        assert_eq!(rung.resolution_name(), "1080p");
    }

    #[test]
    fn test_resolution_name_720p() {
        let rung = make_rung(1280, 720, 3000, 30.0);
        assert_eq!(rung.resolution_name(), "720p");
    }

    #[test]
    fn test_resolution_name_480p() {
        let rung = make_rung(854, 480, 1500, 30.0);
        assert_eq!(rung.resolution_name(), "480p");
    }

    #[test]
    fn test_is_hd_true() {
        let rung = make_rung(1280, 720, 3000, 30.0);
        assert!(rung.is_hd());
    }

    #[test]
    fn test_is_hd_false() {
        let rung = make_rung(854, 480, 1500, 30.0);
        assert!(!rung.is_hd());
    }

    #[test]
    fn test_bits_per_pixel_positive() {
        let rung = make_rung(1280, 720, 3000, 30.0);
        let bpp = rung.bits_per_pixel();
        assert!(bpp > 0.0, "bits_per_pixel should be positive");
    }

    #[test]
    fn test_bits_per_pixel_zero_fps() {
        let rung = make_rung(1280, 720, 3000, 0.0);
        assert_eq!(rung.bits_per_pixel(), 0.0);
    }

    #[test]
    fn test_optimizer_default() {
        let opt = LadderOptimizer::default();
        assert_eq!(opt.min_rungs, 2);
        assert_eq!(opt.max_rungs, 6);
    }

    #[test]
    fn test_optimize_ladder_filters_low_bpp() {
        let opt = LadderOptimizer {
            min_rungs: 1,
            max_rungs: 6,
            complexity_threshold: 10.0, // very high threshold — most will be filtered
        };
        let rungs = vec![
            make_rung(1920, 1080, 5000, 30.0),
            make_rung(1280, 720, 3000, 30.0),
        ];
        // bits_per_pixel for these will be far below 10.0, so all get filtered but min_rungs=1
        let result = opt.optimize_ladder(8000, &rungs);
        assert!(!result.is_empty());
        assert!(result.len() <= 6);
    }

    #[test]
    fn test_optimize_ladder_keeps_min_rungs() {
        let opt = LadderOptimizer {
            min_rungs: 2,
            max_rungs: 6,
            complexity_threshold: 999.0, // absurdly high so all are filtered
        };
        let rungs = vec![
            make_rung(1920, 1080, 5000, 30.0),
            make_rung(1280, 720, 3000, 30.0),
            make_rung(854, 480, 1500, 30.0),
        ];
        let result = opt.optimize_ladder(8000, &rungs);
        assert!(result.len() >= 2);
    }

    #[test]
    fn test_optimize_ladder_respects_max_rungs() {
        let opt = LadderOptimizer {
            min_rungs: 1,
            max_rungs: 2,
            complexity_threshold: 0.0,
        };
        let rungs = vec![
            make_rung(3840, 2160, 15000, 30.0),
            make_rung(1920, 1080, 5000, 30.0),
            make_rung(1280, 720, 3000, 30.0),
            make_rung(854, 480, 1500, 30.0),
        ];
        let result = opt.optimize_ladder(20000, &rungs);
        assert!(result.len() <= 2);
    }

    #[test]
    fn test_recommended_rung_count_min() {
        let opt = LadderOptimizer::default(); // min=2, max=6
        assert_eq!(opt.recommended_rung_count(0.0), 2);
    }

    #[test]
    fn test_recommended_rung_count_max() {
        let opt = LadderOptimizer::default();
        assert_eq!(opt.recommended_rung_count(1.0), 6);
    }

    #[test]
    fn test_bandwidth_model_select_rung() {
        let model = BandwidthModel {
            peak_mbps: 10.0,
            average_mbps: 4.0, // 4000 kbps average
            percentile_95: 8.0,
        };
        let ladder = vec![
            make_rung(854, 480, 1500, 30.0),
            make_rung(1280, 720, 3000, 30.0),
            make_rung(1920, 1080, 6000, 30.0), // too high
        ];
        let rung = model.select_rung(&ladder);
        assert!(rung.is_some());
        assert_eq!(rung.expect("rung should be found").height, 720);
    }

    #[test]
    fn test_bandwidth_model_no_rung_fits() {
        let model = BandwidthModel {
            peak_mbps: 0.5,
            average_mbps: 0.1, // 100 kbps — too low for any rung
            percentile_95: 0.4,
        };
        let ladder = vec![make_rung(854, 480, 1500, 30.0)];
        assert!(model.select_rung(&ladder).is_none());
    }
}
