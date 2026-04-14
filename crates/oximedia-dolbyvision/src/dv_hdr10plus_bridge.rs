//! Dolby Vision to HDR10+ metadata conversion bridge.
//!
//! Converts DV RPU Level 1 / Level 2 metadata into HDR10+ SEI messages,
//! mapping DV tone mapping curves to HDR10+ bezier-curve approximations
//! and targeted system display parameters.

use crate::{Level1Metadata, Level2Metadata};

// ── HDR10+ SEI structures ────────────────────────────────────────────────────

/// HDR10+ SEI payload (simplified ST 2094-40 dynamic metadata).
#[derive(Debug, Clone)]
pub struct Hdr10PlusSei {
    /// Country code (always 0xB5 for USA / SMPTE).
    pub country_code: u8,
    /// Terminal provider code (Samsung HDR10+ = 0x003C).
    pub terminal_provider_code: u16,
    /// Terminal provider oriented code (HDR10+ = 0x0001).
    pub terminal_provider_oriented_code: u16,
    /// Application identifier (4 = HDR10+).
    pub application_identifier: u8,
    /// Application version.
    pub application_version: u8,
    /// Number of windows (typically 1 for full-frame).
    pub num_windows: u8,
    /// Per-window metadata.
    pub windows: Vec<Hdr10PlusWindow>,
    /// Targeted system display maximum luminance (in units of 1 cd/m^2).
    pub targeted_system_display_maximum_luminance: u32,
    /// Targeted system display actual peak luminance flag.
    pub targeted_system_display_actual_peak_luminance_flag: bool,
    /// Mastering display actual peak luminance flag.
    pub mastering_display_actual_peak_luminance_flag: bool,
}

impl Hdr10PlusSei {
    /// Create a default HDR10+ SEI with one full-frame window.
    #[must_use]
    pub fn new_single_window(window: Hdr10PlusWindow, target_max_nits: u32) -> Self {
        Self {
            country_code: 0xB5,
            terminal_provider_code: 0x003C,
            terminal_provider_oriented_code: 0x0001,
            application_identifier: 4,
            application_version: 1,
            num_windows: 1,
            windows: vec![window],
            targeted_system_display_maximum_luminance: target_max_nits,
            targeted_system_display_actual_peak_luminance_flag: false,
            mastering_display_actual_peak_luminance_flag: false,
        }
    }
}

/// Per-window HDR10+ metadata.
#[derive(Debug, Clone)]
pub struct Hdr10PlusWindow {
    /// Maximum scene-frame-average content light level (in 0.0001 cd/m^2 units).
    pub maxscl: [u32; 3],
    /// Average maximum RGB value of the scene (in 0.0001 cd/m^2 units).
    pub average_maxrgb: u32,
    /// Number of distribution maxrgb percentiles.
    pub num_distribution_maxrgb_percentiles: u8,
    /// Percentile values (index, percentile).
    pub distribution_maxrgb_percentiles: Vec<DistributionPercentile>,
    /// Fraction bright pixels (0.0-1.0, stored as u16 / 1000).
    pub fraction_bright_pixels: u16,
    /// Tone mapping flag (true = bezier curve present).
    pub tone_mapping_flag: bool,
    /// Knee point x (in 0.0001 units, 0-1).
    pub knee_point_x: u16,
    /// Knee point y (in 0.0001 units, 0-1).
    pub knee_point_y: u16,
    /// Number of bezier curve anchors.
    pub num_bezier_curve_anchors: u8,
    /// Bezier curve anchor points (0.0-1.0 in 0.0001 units).
    pub bezier_curve_anchors: Vec<u16>,
}

impl Default for Hdr10PlusWindow {
    fn default() -> Self {
        Self {
            maxscl: [0; 3],
            average_maxrgb: 0,
            num_distribution_maxrgb_percentiles: 0,
            distribution_maxrgb_percentiles: Vec::new(),
            fraction_bright_pixels: 0,
            tone_mapping_flag: false,
            knee_point_x: 0,
            knee_point_y: 0,
            num_bezier_curve_anchors: 0,
            bezier_curve_anchors: Vec::new(),
        }
    }
}

/// Distribution percentile entry for HDR10+ maxRGB histogram.
#[derive(Debug, Clone, Copy)]
pub struct DistributionPercentile {
    /// Percentile index (0-100).
    pub percentage: u8,
    /// Percentile value (in 0.0001 cd/m^2 units).
    pub percentile: u32,
}

// ── Conversion bridge ────────────────────────────────────────────────────────

/// Converter from Dolby Vision RPU metadata to HDR10+ SEI messages.
///
/// Maps DV Level 1 (min/max/avg PQ) and optional Level 2 (trim) metadata
/// into HDR10+ dynamic tone mapping structures.
#[derive(Debug, Clone)]
pub struct DvToHdr10PlusBridge {
    /// Target system display maximum luminance (nits) for HDR10+ output.
    pub target_max_nits: u32,
    /// Source mastering display maximum luminance (nits).
    pub source_max_nits: u32,
    /// Source mastering display minimum luminance (nits, scaled by 10000).
    pub source_min_nits_x10000: u32,
    /// Number of bezier curve anchors to generate for tone mapping approximation.
    pub num_bezier_anchors: u8,
}

impl Default for DvToHdr10PlusBridge {
    fn default() -> Self {
        Self {
            target_max_nits: 1000,
            source_max_nits: 4000,
            source_min_nits_x10000: 50, // 0.005 nits
            num_bezier_anchors: 9,
        }
    }
}

impl DvToHdr10PlusBridge {
    /// Create a new bridge with the specified target and source luminance.
    #[must_use]
    pub fn new(target_max_nits: u32, source_max_nits: u32) -> Self {
        Self {
            target_max_nits,
            source_max_nits,
            ..Default::default()
        }
    }

    /// Convert DV Level 1 metadata into an HDR10+ SEI message.
    ///
    /// When `level2` is provided, the trim parameters are used to refine
    /// the bezier tone mapping curve.
    #[must_use]
    pub fn convert(
        &self,
        level1: &Level1Metadata,
        level2: Option<&Level2Metadata>,
    ) -> Hdr10PlusSei {
        let max_nits = pq_to_linear_nits(level1.max_pq);
        let min_nits = pq_to_linear_nits(level1.min_pq);
        let avg_nits = pq_to_linear_nits(level1.avg_pq);

        // Convert to 0.0001 cd/m^2 units
        let max_scl_value = (max_nits * 10000.0) as u32;
        let avg_maxrgb = (avg_nits * 10000.0) as u32;

        // Build distribution percentiles from L1 data
        let percentiles = self.build_percentiles(min_nits, avg_nits, max_nits);

        // Build bezier tone mapping curve
        let (knee_x, knee_y, anchors) = self.build_bezier_curve(level1, level2);

        let window = Hdr10PlusWindow {
            maxscl: [max_scl_value, max_scl_value, max_scl_value],
            average_maxrgb: avg_maxrgb,
            num_distribution_maxrgb_percentiles: percentiles.len() as u8,
            distribution_maxrgb_percentiles: percentiles,
            fraction_bright_pixels: self.estimate_bright_fraction(level1),
            tone_mapping_flag: true,
            knee_point_x: knee_x,
            knee_point_y: knee_y,
            num_bezier_curve_anchors: anchors.len() as u8,
            bezier_curve_anchors: anchors,
        };

        Hdr10PlusSei::new_single_window(window, self.target_max_nits)
    }

    /// Build distribution percentiles from L1 luminance data.
    fn build_percentiles(
        &self,
        min_nits: f64,
        avg_nits: f64,
        max_nits: f64,
    ) -> Vec<DistributionPercentile> {
        // Standard 9 percentiles at fixed intervals
        let percentages = [1u8, 5, 10, 25, 50, 75, 90, 95, 99];
        let range = max_nits - min_nits;

        percentages
            .iter()
            .map(|&pct| {
                let t = f64::from(pct) / 100.0;
                // Use a distribution model biased toward the average
                let value = if t <= 0.5 {
                    // Lower half: interpolate from min to avg
                    let local_t = t * 2.0;
                    min_nits + local_t * (avg_nits - min_nits)
                } else {
                    // Upper half: interpolate from avg to max
                    let local_t = (t - 0.5) * 2.0;
                    avg_nits + local_t * (max_nits - avg_nits)
                };
                let clamped = value.clamp(0.0, range + min_nits);
                DistributionPercentile {
                    percentage: pct,
                    percentile: (clamped * 10000.0) as u32,
                }
            })
            .collect()
    }

    /// Build bezier tone mapping curve from DV metadata.
    ///
    /// Returns (knee_point_x, knee_point_y, anchor_values).
    /// Values are in 0-10000 range (representing 0.0-1.0 in fixed point).
    fn build_bezier_curve(
        &self,
        level1: &Level1Metadata,
        level2: Option<&Level2Metadata>,
    ) -> (u16, u16, Vec<u16>) {
        let source_max = self.source_max_nits as f64;
        let target_max = self.target_max_nits as f64;

        // Knee point: where tone mapping starts to compress
        // Typically around the target display capability relative to source
        let knee_ratio = (target_max / source_max).clamp(0.0, 1.0);

        // Apply L2 trim adjustments if available
        let (slope_adj, offset_adj, power_adj) = if let Some(l2) = level2 {
            let slope = f64::from(l2.trim_slope) / f64::from(1i16 << 12);
            let offset = f64::from(l2.trim_offset) / f64::from(1i16 << 12);
            let power = f64::from(l2.trim_power) / f64::from(1i16 << 12);
            (slope.clamp(0.1, 4.0), offset.clamp(-1.0, 1.0), power.clamp(0.1, 4.0))
        } else {
            (1.0, 0.0, 1.0)
        };

        // Knee point position influenced by content's max PQ
        let content_max_ratio =
            pq_to_linear_nits(level1.max_pq) / source_max.max(1.0);
        let knee_x_f = (knee_ratio * content_max_ratio.clamp(0.3, 1.0) * slope_adj)
            .clamp(0.1, 0.95);
        let knee_y_f = self.tone_map_value(knee_x_f, slope_adj, offset_adj, power_adj);

        let knee_x = (knee_x_f * 10000.0).clamp(0.0, 10000.0) as u16;
        let knee_y = (knee_y_f * 10000.0).clamp(0.0, 10000.0) as u16;

        // Generate bezier anchors between knee point and (1,1)
        let num_anchors = self.num_bezier_anchors.max(1) as usize;
        let mut anchors = Vec::with_capacity(num_anchors);

        for i in 0..num_anchors {
            let t = (i as f64 + 1.0) / (num_anchors as f64 + 1.0);
            let x = knee_x_f + t * (1.0 - knee_x_f);
            let y = self.tone_map_value(x, slope_adj, offset_adj, power_adj);
            anchors.push((y * 10000.0).clamp(0.0, 10000.0) as u16);
        }

        (knee_x, knee_y, anchors)
    }

    /// Apply polynomial tone mapping (approximating DV's internal curve).
    ///
    /// Maps input (0..1 normalized to source range) to output (0..1 normalized to target).
    fn tone_map_value(&self, x: f64, slope: f64, offset: f64, power: f64) -> f64 {
        let source_max = self.source_max_nits as f64;
        let target_max = self.target_max_nits as f64;
        let ratio = target_max / source_max.max(1.0);

        // Reinhard-style curve with DV trim adjustments
        let input = (x * slope + offset).clamp(0.0, 1.0);
        let mapped = if ratio >= 1.0 {
            // Target >= source: linear pass-through
            input
        } else {
            // Compress highlights: modified Reinhard
            let numerator = input;
            let denominator = input + (1.0 - ratio) * input.powf(power);
            if denominator > f64::EPSILON {
                numerator / denominator
            } else {
                0.0
            }
        };

        mapped.clamp(0.0, 1.0)
    }

    /// Estimate fraction of bright pixels from L1 metadata.
    fn estimate_bright_fraction(&self, level1: &Level1Metadata) -> u16 {
        let avg_nits = pq_to_linear_nits(level1.avg_pq);
        let max_nits = pq_to_linear_nits(level1.max_pq);
        let target = self.target_max_nits as f64;

        if max_nits <= target || max_nits < f64::EPSILON {
            // No pixels exceed target
            return 0;
        }

        // Rough estimate: fraction above target based on avg/max ratio
        let bright_threshold = target / max_nits.max(1.0);
        let estimated = 1.0 - bright_threshold;
        let adjusted = estimated * (avg_nits / max_nits.max(1.0));

        (adjusted.clamp(0.0, 1.0) * 1000.0) as u16
    }
}

/// Convert PQ code (0-4095) to linear luminance in nits.
fn pq_to_linear_nits(pq_code: u16) -> f64 {
    const M1_INV: f64 = 1.0 / 0.159_301_758_113_479_8;
    const M2_INV: f64 = 1.0 / 78.843_750;
    const C1: f64 = 0.835_937_5;
    const C2: f64 = 18.851_562_5;
    const C3: f64 = 18.6875;

    let pq_norm = f64::from(pq_code) / 4095.0;
    let v = pq_norm.powf(M2_INV);
    let y = ((v - C1).max(0.0) / (C2 - C3 * v).max(f64::EPSILON)).powf(M1_INV);
    y * 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_l1(min_pq: u16, max_pq: u16, avg_pq: u16) -> Level1Metadata {
        Level1Metadata {
            min_pq,
            max_pq,
            avg_pq,
        }
    }

    fn make_l2(slope: i16, offset: i16, power: i16) -> Level2Metadata {
        Level2Metadata {
            target_display_index: 0,
            trim_slope: slope,
            trim_offset: offset,
            trim_power: power,
            trim_chroma_weight: 1 << 12,
            trim_saturation_gain: 1 << 12,
            ms_weight: 1 << 12,
            target_mid_contrast: 0,
            clip_trim: 0,
            saturation_vector_field: Vec::new(),
            hue_vector_field: Vec::new(),
        }
    }

    #[test]
    fn test_basic_conversion() {
        let bridge = DvToHdr10PlusBridge::default();
        let l1 = make_l1(62, 3696, 2048);
        let sei = bridge.convert(&l1, None);

        assert_eq!(sei.country_code, 0xB5);
        assert_eq!(sei.application_identifier, 4);
        assert_eq!(sei.num_windows, 1);
        assert_eq!(sei.targeted_system_display_maximum_luminance, 1000);
    }

    #[test]
    fn test_conversion_with_l2_trim() {
        let bridge = DvToHdr10PlusBridge::default();
        let l1 = make_l1(62, 3696, 2048);
        let l2 = make_l2(1 << 12, 0, 1 << 12); // identity trim
        let sei = bridge.convert(&l1, Some(&l2));

        assert!(sei.windows[0].tone_mapping_flag);
        assert!(!sei.windows[0].bezier_curve_anchors.is_empty());
    }

    #[test]
    fn test_bezier_anchors_count() {
        let bridge = DvToHdr10PlusBridge {
            num_bezier_anchors: 5,
            ..Default::default()
        };
        let l1 = make_l1(62, 3696, 2048);
        let sei = bridge.convert(&l1, None);

        assert_eq!(sei.windows[0].num_bezier_curve_anchors, 5);
        assert_eq!(sei.windows[0].bezier_curve_anchors.len(), 5);
    }

    #[test]
    fn test_bezier_anchors_monotonic() {
        let bridge = DvToHdr10PlusBridge::default();
        let l1 = make_l1(62, 4079, 2500);
        let sei = bridge.convert(&l1, None);

        let anchors = &sei.windows[0].bezier_curve_anchors;
        for w in anchors.windows(2) {
            assert!(
                w[1] >= w[0],
                "anchors should be monotonically non-decreasing: {} < {}",
                w[1],
                w[0]
            );
        }
    }

    #[test]
    fn test_maxscl_reflects_l1_max() {
        let bridge = DvToHdr10PlusBridge::default();
        let l1 = make_l1(0, 4095, 2048);
        let sei = bridge.convert(&l1, None);

        // max_pq = 4095 -> 10000 nits -> maxscl = 10000 * 10000
        assert!(sei.windows[0].maxscl[0] > 0);
        assert_eq!(sei.windows[0].maxscl[0], sei.windows[0].maxscl[1]);
    }

    #[test]
    fn test_percentiles_ordered() {
        let bridge = DvToHdr10PlusBridge::default();
        let l1 = make_l1(62, 3696, 1500);
        let sei = bridge.convert(&l1, None);

        let percs = &sei.windows[0].distribution_maxrgb_percentiles;
        assert!(!percs.is_empty());
        for w in percs.windows(2) {
            assert!(
                w[1].percentile >= w[0].percentile,
                "percentiles should be non-decreasing: {} < {}",
                w[1].percentile,
                w[0].percentile
            );
        }
    }

    #[test]
    fn test_dark_scene_low_bright_fraction() {
        let bridge = DvToHdr10PlusBridge::default();
        // Dark scene: max PQ below target
        let l1 = make_l1(0, 2000, 1000);
        let sei = bridge.convert(&l1, None);

        // All content below target -> fraction should be 0
        assert_eq!(
            sei.windows[0].fraction_bright_pixels, 0,
            "dark scene should have 0 bright fraction"
        );
    }

    #[test]
    fn test_bright_scene_nonzero_bright_fraction() {
        let bridge = DvToHdr10PlusBridge::new(1000, 4000);
        // Bright scene: max PQ at 4000 nits, avg at ~2000 nits
        let l1 = make_l1(62, 4079, 3500);
        let sei = bridge.convert(&l1, None);

        assert!(
            sei.windows[0].fraction_bright_pixels > 0,
            "bright scene should have nonzero bright fraction"
        );
    }

    #[test]
    fn test_l2_slope_affects_curve() {
        let bridge = DvToHdr10PlusBridge::default();
        let l1 = make_l1(62, 3696, 2048);

        // High slope trim
        let l2_high = make_l2(2 << 12, 0, 1 << 12);
        let sei_high = bridge.convert(&l1, Some(&l2_high));

        // Low slope trim
        let l2_low = make_l2(1 << 11, 0, 1 << 12); // 0.5x slope
        let sei_low = bridge.convert(&l1, Some(&l2_low));

        // Different slopes should produce different knee points
        let kx_high = sei_high.windows[0].knee_point_x;
        let kx_low = sei_low.windows[0].knee_point_x;
        // They may differ (or coincidentally match, but generally won't with these values)
        assert!(
            kx_high != kx_low || sei_high.windows[0].knee_point_y != sei_low.windows[0].knee_point_y,
            "different L2 slopes should produce different curves"
        );
    }

    #[test]
    fn test_pq_to_linear_nits_boundary() {
        let zero = pq_to_linear_nits(0);
        assert!(zero.abs() < 0.01, "PQ 0 should map to ~0 nits, got {zero}");

        let max = pq_to_linear_nits(4095);
        assert!(
            (max - 10000.0).abs() < 100.0,
            "PQ 4095 should map to ~10000 nits, got {max}"
        );
    }

    #[test]
    fn test_hdr10plus_sei_structure() {
        let window = Hdr10PlusWindow::default();
        let sei = Hdr10PlusSei::new_single_window(window, 1000);
        assert_eq!(sei.country_code, 0xB5);
        assert_eq!(sei.terminal_provider_code, 0x003C);
        assert_eq!(sei.terminal_provider_oriented_code, 0x0001);
        assert_eq!(sei.application_identifier, 4);
        assert_eq!(sei.application_version, 1);
        assert_eq!(sei.num_windows, 1);
    }

    #[test]
    fn test_target_passthrough_when_source_smaller() {
        // Source max <= target max: should pass through nearly linearly
        let bridge = DvToHdr10PlusBridge::new(4000, 1000);
        let l1 = make_l1(62, 3696, 2048);
        let sei = bridge.convert(&l1, None);

        // Knee point y should be close to knee point x (near-linear)
        let kx = sei.windows[0].knee_point_x as f64 / 10000.0;
        let ky = sei.windows[0].knee_point_y as f64 / 10000.0;
        assert!(
            (ky - kx).abs() < 0.3,
            "near-linear mapping expected: kx={kx}, ky={ky}"
        );
    }
}
