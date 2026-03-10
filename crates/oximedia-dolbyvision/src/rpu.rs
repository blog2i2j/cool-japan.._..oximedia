//! RPU (Reference Processing Unit) header and data structures.
//!
//! This module defines the core RPU structures for Dolby Vision metadata.

use crate::{DolbyVisionError, Profile, Result};
use bitflags::bitflags;

/// RPU header containing format and configuration information.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RpuHeader {
    /// RPU type (0 = metadata, 1 = reserved)
    pub rpu_type: u8,

    /// RPU format (0 = standard, 1 = extended)
    pub rpu_format: u16,

    /// VDR sequence info present flag
    pub vdr_seq_info_present: bool,

    /// VDR sequence information
    pub vdr_seq_info: Option<VdrSeqInfo>,

    /// Picture index
    pub picture_index: u16,

    /// Change flags
    pub change_flags: ChangeFlags,

    /// NLQ parameter prediction flag
    pub nlq_param_pred_flag: bool,

    /// Number of NLQ parameters minus 1
    pub num_nlq_param_predictors: u8,

    /// Component order (0 = RGB, 1 = GBR, 2 = BGR, etc.)
    pub component_order: u8,

    /// Coefficient data type (0 = fixed point, 1 = floating point)
    pub coef_data_type: u8,

    /// Coefficient log2 denominator
    pub coef_log2_denom: u8,

    /// Mapping color space (0 = YCbCr, 1 = RGB, 2 = IPT)
    pub mapping_color_space: u8,

    /// Mapping chroma format (0 = 4:2:0, 1 = 4:2:2, 2 = 4:4:4)
    pub mapping_chroma_format: u8,

    /// Number of mapping pivots minus 2
    pub num_pivots_minus_2: u8,

    /// Prediction pivot value
    pub pred_pivot_value: u16,
}

impl RpuHeader {
    /// Create default header for the given profile.
    #[must_use]
    pub fn default_for_profile(profile: Profile) -> Self {
        Self {
            rpu_type: 0,
            rpu_format: 0,
            vdr_seq_info_present: true,
            vdr_seq_info: Some(VdrSeqInfo::default_for_profile(profile)),
            picture_index: 0,
            change_flags: ChangeFlags::empty(),
            nlq_param_pred_flag: false,
            num_nlq_param_predictors: 0,
            component_order: match profile {
                Profile::Profile5 => 2, // IPT order
                _ => 0,                 // RGB order
            },
            coef_data_type: 0, // Fixed point
            coef_log2_denom: 14,
            mapping_color_space: match profile {
                Profile::Profile5 => 2, // IPT
                _ => 1,                 // RGB
            },
            mapping_chroma_format: 2, // 4:4:4
            num_pivots_minus_2: 0,
            pred_pivot_value: 0,
        }
    }
}

bitflags! {
    /// Change flags indicating which metadata blocks have changed.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct ChangeFlags: u16 {
        /// VDR metadata changed
        const VDR_CHANGED = 1 << 0;
        /// Mapping changed
        const MAPPING_CHANGED = 1 << 1;
        /// Color correction changed
        const COLOR_CORRECTION_CHANGED = 1 << 2;
        /// NLQ changed
        const NLQ_CHANGED = 1 << 3;
    }
}

/// VDR (Vizio Display Management) sequence information.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VdrSeqInfo {
    /// VDR DM (Display Management) metadata ID
    pub vdr_dm_metadata_id: u8,

    /// Scene refresh flag
    pub scene_refresh_flag: u8,

    /// YCbCr to RGB flag
    pub ycbcr_to_rgb_flag: bool,

    /// Coefficient data type
    pub coef_data_type: u8,

    /// Coefficient log2 denominator
    pub coef_log2_denom: u8,

    /// VDR bit depth
    pub vdr_bit_depth: u8,

    /// BL bit depth (base layer)
    pub bl_bit_depth: u8,

    /// EL bit depth (enhancement layer)
    pub el_bit_depth: u8,

    /// Original bit depth
    pub source_bit_depth: u8,
}

impl VdrSeqInfo {
    /// Create default VDR sequence info for the given profile.
    #[must_use]
    pub fn default_for_profile(profile: Profile) -> Self {
        Self {
            vdr_dm_metadata_id: 0,
            scene_refresh_flag: 0,
            ycbcr_to_rgb_flag: !matches!(profile, Profile::Profile5),
            coef_data_type: 0,
            coef_log2_denom: 14,
            vdr_bit_depth: 12,
            bl_bit_depth: 10,
            el_bit_depth: 8,
            source_bit_depth: 10,
        }
    }
}

/// VDR DM (Vizio Display Management) data.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VdrDmData {
    /// Affected DM metadata ID
    pub affected_dm_metadata_id: u8,

    /// Current DM metadata ID
    pub current_dm_metadata_id: u8,

    /// Scene refresh flag
    pub scene_refresh_flag: u8,

    /// YCbCr to RGB conversion matrix
    pub ycbcr_to_rgb_matrix: Option<ColorMatrix>,

    /// RGB to LMS conversion matrix
    pub rgb_to_lms_matrix: Option<ColorMatrix>,

    /// Signal EOTF (Electro-Optical Transfer Function)
    pub signal_eotf: u16,

    /// Signal EOTF parameter 0
    pub signal_eotf_param0: u16,

    /// Signal EOTF parameter 1
    pub signal_eotf_param1: u16,

    /// Signal EOTF parameter 2
    pub signal_eotf_param2: u32,

    /// Signal bit depth
    pub signal_bit_depth: u8,

    /// Signal color space
    pub signal_color_space: u8,

    /// Signal chroma format
    pub signal_chroma_format: u8,

    /// Signal full range flag
    pub signal_full_range_flag: u8,

    /// Source minimum PQ
    pub source_min_pq: u16,

    /// Source maximum PQ
    pub source_max_pq: u16,

    /// Source diagonal
    pub source_diagonal: u16,

    /// Reshaping curves
    pub reshaping_curves: Vec<ReshapingCurve>,

    /// NLQ (Non-Linear Quantization) parameters
    pub nlq_params: Vec<NlqParams>,
}

impl Default for VdrDmData {
    fn default() -> Self {
        Self {
            affected_dm_metadata_id: 0,
            current_dm_metadata_id: 0,
            scene_refresh_flag: 0,
            ycbcr_to_rgb_matrix: None,
            rgb_to_lms_matrix: None,
            signal_eotf: 0,
            signal_eotf_param0: 0,
            signal_eotf_param1: 0,
            signal_eotf_param2: 0,
            signal_bit_depth: 10,
            signal_color_space: 0,
            signal_chroma_format: 0,
            signal_full_range_flag: 0,
            source_min_pq: 0,
            source_max_pq: 4095,
            source_diagonal: 42,
            reshaping_curves: Vec::new(),
            nlq_params: Vec::new(),
        }
    }
}

/// Color transformation matrix (3x3).
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColorMatrix {
    /// Matrix coefficients \[row\]\[col\]
    pub matrix: [[i32; 3]; 3],
}

impl ColorMatrix {
    /// Create identity matrix.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            matrix: [[1 << 14, 0, 0], [0, 1 << 14, 0], [0, 0, 1 << 14]],
        }
    }

    /// BT.2020 YCbCr to RGB matrix (scaled by 2^14).
    #[must_use]
    pub const fn bt2020_ycbcr_to_rgb() -> Self {
        Self {
            matrix: [[16384, 0, 24543], [16384, -2752, -8869], [16384, 28688, 0]],
        }
    }

    /// BT.709 YCbCr to RGB matrix (scaled by 2^14).
    #[must_use]
    pub const fn bt709_ycbcr_to_rgb() -> Self {
        Self {
            matrix: [[16384, 0, 25803], [16384, -3073, -7722], [16384, 30372, 0]],
        }
    }

    /// BT.2020 RGB to LMS matrix (IPT color space, scaled by 2^14).
    #[must_use]
    pub const fn bt2020_rgb_to_lms() -> Self {
        Self {
            matrix: [[6610, 8192, 1582], [2766, 12298, 1320], [82, 820, 15482]],
        }
    }
}

impl Default for ColorMatrix {
    fn default() -> Self {
        Self::identity()
    }
}

/// Reshaping curve for tone mapping.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReshapingCurve {
    /// Pivot values (input)
    pub pivots: Vec<u16>,

    /// Mapping values (output)
    pub mapping_idc: Vec<u8>,

    /// Polynomial mapping flag
    pub poly_order_minus1: Vec<u8>,

    /// Polynomial coefficients
    pub poly_coef: Vec<Vec<i64>>,

    /// MMR (Min/Max/Avg RGB) order
    pub mmr_order_minus1: u8,

    /// MMR coefficients
    pub mmr_coef: Vec<i64>,
}

impl Default for ReshapingCurve {
    fn default() -> Self {
        Self {
            pivots: vec![0, 4095],
            mapping_idc: vec![0],
            poly_order_minus1: vec![0],
            poly_coef: vec![vec![0, 1 << 14]],
            mmr_order_minus1: 0,
            mmr_coef: vec![1 << 14, 0, 0],
        }
    }
}

/// NLQ (Non-Linear Quantization) parameters.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NlqParams {
    /// NLQ offset
    pub nlq_offset: u16,

    /// VDR in max
    pub vdr_in_max: u64,

    /// Linear deadzone slope
    pub linear_deadzone_slope: u64,

    /// Linear deadzone threshold
    pub linear_deadzone_threshold: u64,
}

impl Default for NlqParams {
    fn default() -> Self {
        Self {
            nlq_offset: 0,
            vdr_in_max: 0,
            linear_deadzone_slope: 0,
            linear_deadzone_threshold: 0,
        }
    }
}

/// Mapping method for tone mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MappingMethod {
    /// Polynomial mapping
    Polynomial = 0,
    /// MMR (Min/Max/Avg RGB) mapping
    Mmr = 1,
}

impl MappingMethod {
    /// Create from numeric value.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Polynomial),
            1 => Some(Self::Mmr),
            _ => None,
        }
    }
}

/// EOTF (Electro-Optical Transfer Function) type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Eotf {
    /// BT.1886 (SDR)
    Bt1886 = 0,
    /// PQ (Perceptual Quantizer, ST 2084)
    Pq = 1,
    /// HLG (Hybrid Log-Gamma)
    Hlg = 2,
    /// Linear
    Linear = 3,
}

impl Eotf {
    /// Create from numeric value.
    #[must_use]
    pub const fn from_u16(value: u16) -> Option<Self> {
        match value {
            0 => Some(Self::Bt1886),
            1 => Some(Self::Pq),
            2 => Some(Self::Hlg),
            3 => Some(Self::Linear),
            _ => None,
        }
    }

    /// Get default EOTF for profile.
    #[must_use]
    pub const fn default_for_profile(profile: Profile) -> Self {
        match profile {
            Profile::Profile8_4 => Self::Hlg,
            _ => Self::Pq,
        }
    }
}

/// Profile-specific RPU data for Profile 5.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Profile5Data {
    /// IPT to RGB conversion matrix
    pub ipt_to_rgb_matrix: ColorMatrix,

    /// RGB primaries
    pub rgb_primaries: [[u16; 2]; 3],

    /// White point
    pub white_point: [u16; 2],

    /// Display mastering metadata present
    pub dm_metadata_present: bool,

    /// Maximum display mastering luminance
    pub max_display_mastering_luminance: u32,

    /// Minimum display mastering luminance
    pub min_display_mastering_luminance: u32,

    /// Maximum content light level
    pub max_cll: u16,

    /// Maximum frame-average light level
    pub max_fall: u16,
}

/// Profile-specific RPU data for Profile 7.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Profile7Data {
    /// MEL (Metadata Enhancement Layer) present
    pub mel_present: bool,

    /// Enhancement layer bit depth
    pub el_bit_depth: u8,

    /// Enhancement layer spatial scalability flag
    pub el_spatial_resampling_filter_flag: bool,

    /// Disable residual flag
    pub disable_residual_flag: bool,

    /// EL type
    pub el_type: u8,
}

/// Profile-specific RPU data for Profile 8.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Profile8Data {
    /// Extended mapping present
    pub ext_mapping_present: bool,

    /// Extended mapping level
    pub ext_mapping_level: u8,
}

/// Profile-specific RPU data for Profile 8.1.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Profile8_1Data {
    /// Low-latency mode
    pub low_latency_mode: bool,

    /// Line-by-line processing flag
    pub line_by_line_flag: bool,
}

/// Profile-specific RPU data for Profile 8.4.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Profile8_4Data {
    /// HLG OOTF (Opto-Optical Transfer Function) gamma
    pub hlg_ootf_gamma: u16,

    /// HLG system gamma
    pub hlg_system_gamma: u16,

    /// Scene luminance scale
    pub scene_luminance_scale: u16,
}

/// Extension block for future use.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExtensionBlock {
    /// Extension block length
    pub length: u16,

    /// Extension block level
    pub level: u8,

    /// Extension block data
    pub data: Vec<u8>,
}

/// Parse variable-length unsigned integer (Exp-Golomb coding).
pub fn parse_ue(bits: &mut dyn Iterator<Item = bool>) -> Result<u32> {
    let mut leading_zeros = 0;
    while let Some(bit) = bits.next() {
        if bit {
            break;
        }
        leading_zeros += 1;
        if leading_zeros > 31 {
            return Err(DolbyVisionError::InvalidPayload(
                "Exp-Golomb code too long".to_string(),
            ));
        }
    }

    if leading_zeros == 0 {
        return Ok(0);
    }

    let mut value = 1u32;
    for _ in 0..leading_zeros {
        value <<= 1;
        if let Some(bit) = bits.next() {
            if bit {
                value |= 1;
            }
        } else {
            return Err(DolbyVisionError::InvalidPayload(
                "Unexpected end of bitstream".to_string(),
            ));
        }
    }

    Ok(value - 1)
}

/// Parse variable-length signed integer (Exp-Golomb coding).
pub fn parse_se(bits: &mut dyn Iterator<Item = bool>) -> Result<i32> {
    let ue = parse_ue(bits)?;
    let sign = if ue & 1 == 0 { -1 } else { 1 };
    Ok(sign * ((ue as i32 + 1) >> 1))
}

// ── New RPU types ─────────────────────────────────────────────────────────────

/// RPU transport container type.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RpuType {
    /// Bare Annex B start codes, no emulation prevention.
    StraightAnnexB,
    /// Standard H.264/H.265 bitstream with emulation prevention bytes.
    EmulationPrevented,
    /// Raw RPU payload with no NAL framing.
    RpuOnly,
}

impl RpuType {
    /// Returns `true` when emulation prevention bytes are present.
    #[must_use]
    pub const fn has_emulation_prevention(self) -> bool {
        matches!(self, Self::EmulationPrevented)
    }
}

/// Colorimetry encoding mode for the RPU data.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorimetryMode {
    /// Standard YCbCr (BT.2020 primaries).
    Ycbcr,
    /// ICtCp (perceptually uniform, used in Profile 5).
    ICtCp,
    /// RGB scene-linear light.
    Rgb,
}

impl ColorimetryMode {
    /// Returns `true` when the mode operates in scene-linear light (RGB only).
    #[must_use]
    pub const fn is_scene_linear(self) -> bool {
        matches!(self, Self::Rgb)
    }
}

/// Compact RPU header carrying transport and spatial flags.
///
/// This is a simplified header distinct from [`RpuHeader`] which carries the
/// full Dolby Vision bitstream header.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct CompactRpuHeader {
    /// RPU transport type.
    pub rpu_type: RpuType,
    /// RPU format field (0 = standard).
    pub rpu_format: u8,
    /// Whether the enhancement layer uses a spatial resampling filter.
    pub el_spatial_resampling_filter_flag: bool,
    /// Whether residual coding is disabled (base-layer-only mode).
    pub disable_residual_flag: bool,
}

impl CompactRpuHeader {
    /// Returns `true` when the stream carries only the base layer (no EL).
    #[must_use]
    pub fn is_base_layer_only(&self) -> bool {
        self.disable_residual_flag
    }
}

/// RPU payload data accompanying a [`CompactRpuHeader`].
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct RpuPayload {
    /// The header for this RPU.
    pub header: CompactRpuHeader,
    /// Size of the display management data block in bytes.
    pub dm_data_size: u32,
    /// Whether VDR display management metadata is present.
    pub vdr_dm_metadata_present: bool,
}

impl RpuPayload {
    /// Returns `true` when dynamic (per-frame) metadata is present.
    #[must_use]
    pub fn has_dynamic_metadata(&self) -> bool {
        self.vdr_dm_metadata_present && self.dm_data_size > 0
    }
}

/// Write variable-length unsigned integer (Exp-Golomb coding).
#[allow(dead_code)]
pub fn write_ue(value: u32) -> Vec<bool> {
    if value == 0 {
        return vec![true];
    }

    let value_plus_1 = value + 1;
    let bit_length = 32 - value_plus_1.leading_zeros();
    let leading_zeros = bit_length - 1;

    let mut bits = Vec::with_capacity((leading_zeros * 2 + 1) as usize);

    // Write leading zeros
    for _ in 0..leading_zeros {
        bits.push(false);
    }

    // Write value bits
    for i in (0..bit_length).rev() {
        bits.push((value_plus_1 >> i) & 1 == 1);
    }

    bits
}

/// Write variable-length signed integer (Exp-Golomb coding).
#[allow(dead_code)]
pub fn write_se(value: i32) -> Vec<bool> {
    let ue = if value <= 0 {
        (-value * 2) as u32
    } else {
        (value * 2 - 1) as u32
    };
    write_ue(ue)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp_golomb_ue() {
        let test_cases = vec![
            (vec![true], 0),
            (vec![false, true, false], 1),
            (vec![false, true, true], 2),
            (vec![false, false, true, false, false], 3),
        ];

        for (bits, expected) in test_cases {
            let mut iter = bits.into_iter();
            let result = parse_ue(&mut iter).expect("result should be valid");
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_exp_golomb_se() {
        let test_cases = vec![
            (vec![true], 0),
            (vec![false, true, false], 1),
            (vec![false, true, true], -1),
            (vec![false, false, true, false, false], 2),
            (vec![false, false, true, false, true], -2),
        ];

        for (bits, expected) in test_cases {
            let mut iter = bits.into_iter();
            let result = parse_se(&mut iter).expect("result should be valid");
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_write_ue() {
        assert_eq!(write_ue(0), vec![true]);
        assert_eq!(write_ue(1), vec![false, true, false]);
        assert_eq!(write_ue(2), vec![false, true, true]);
        assert_eq!(write_ue(3), vec![false, false, true, false, false]);
    }

    #[test]
    fn test_write_se() {
        assert_eq!(write_se(0), vec![true]);
        assert_eq!(write_se(1), vec![false, true, false]);
        assert_eq!(write_se(-1), vec![false, true, true]);
        assert_eq!(write_se(2), vec![false, false, true, false, false]);
        assert_eq!(write_se(-2), vec![false, false, true, false, true]);
    }

    #[test]
    fn test_color_matrix() {
        let identity = ColorMatrix::identity();
        assert_eq!(identity.matrix[0][0], 1 << 14);
        assert_eq!(identity.matrix[1][1], 1 << 14);
        assert_eq!(identity.matrix[2][2], 1 << 14);

        let bt2020 = ColorMatrix::bt2020_ycbcr_to_rgb();
        assert_eq!(bt2020.matrix[0][0], 16384);
    }

    #[test]
    fn test_eotf() {
        assert_eq!(Eotf::from_u16(0), Some(Eotf::Bt1886));
        assert_eq!(Eotf::from_u16(1), Some(Eotf::Pq));
        assert_eq!(Eotf::from_u16(2), Some(Eotf::Hlg));
        assert_eq!(Eotf::from_u16(3), Some(Eotf::Linear));
        assert_eq!(Eotf::from_u16(99), None);

        assert_eq!(Eotf::default_for_profile(Profile::Profile8_4), Eotf::Hlg);
        assert_eq!(Eotf::default_for_profile(Profile::Profile8), Eotf::Pq);
    }

    #[test]
    fn test_mapping_method() {
        assert_eq!(MappingMethod::from_u8(0), Some(MappingMethod::Polynomial));
        assert_eq!(MappingMethod::from_u8(1), Some(MappingMethod::Mmr));
        assert_eq!(MappingMethod::from_u8(99), None);
    }

    #[test]
    fn test_vdr_seq_info() {
        let info = VdrSeqInfo::default_for_profile(Profile::Profile8);
        assert_eq!(info.vdr_bit_depth, 12);
        assert_eq!(info.bl_bit_depth, 10);
        assert!(info.ycbcr_to_rgb_flag);
    }

    #[test]
    fn test_rpu_header() {
        let header = RpuHeader::default_for_profile(Profile::Profile8);
        assert_eq!(header.rpu_type, 0);
        assert_eq!(header.rpu_format, 0);
        assert!(header.vdr_seq_info_present);
    }
}

// ── Unit tests for new RPU types ──────────────────────────────────────────────

#[cfg(test)]
mod rpu_type_tests {
    use super::*;

    fn compact_header(
        rpu_type: RpuType,
        rpu_format: u8,
        el_spatial: bool,
        disable_residual: bool,
    ) -> CompactRpuHeader {
        CompactRpuHeader {
            rpu_type,
            rpu_format,
            el_spatial_resampling_filter_flag: el_spatial,
            disable_residual_flag: disable_residual,
        }
    }

    fn payload(header: CompactRpuHeader, dm_size: u32, vdr_present: bool) -> RpuPayload {
        RpuPayload {
            header,
            dm_data_size: dm_size,
            vdr_dm_metadata_present: vdr_present,
        }
    }

    #[test]
    fn test_rpu_type_straight_no_emulation() {
        assert!(!RpuType::StraightAnnexB.has_emulation_prevention());
    }

    #[test]
    fn test_rpu_type_emulation_prevented_true() {
        assert!(RpuType::EmulationPrevented.has_emulation_prevention());
    }

    #[test]
    fn test_rpu_type_rpu_only_no_emulation() {
        assert!(!RpuType::RpuOnly.has_emulation_prevention());
    }

    #[test]
    fn test_colorimetry_rgb_is_scene_linear() {
        assert!(ColorimetryMode::Rgb.is_scene_linear());
    }

    #[test]
    fn test_colorimetry_ycbcr_not_scene_linear() {
        assert!(!ColorimetryMode::Ycbcr.is_scene_linear());
    }

    #[test]
    fn test_colorimetry_ictcp_not_scene_linear() {
        assert!(!ColorimetryMode::ICtCp.is_scene_linear());
    }

    #[test]
    fn test_compact_header_base_layer_only_true() {
        let h = compact_header(RpuType::RpuOnly, 0, false, true);
        assert!(h.is_base_layer_only());
    }

    #[test]
    fn test_compact_header_base_layer_only_false() {
        let h = compact_header(RpuType::StraightAnnexB, 0, true, false);
        assert!(!h.is_base_layer_only());
    }

    #[test]
    fn test_rpu_payload_has_dynamic_metadata_true() {
        let h = compact_header(RpuType::EmulationPrevented, 0, false, false);
        let p = payload(h, 128, true);
        assert!(p.has_dynamic_metadata());
    }

    #[test]
    fn test_rpu_payload_has_dynamic_metadata_no_vdr() {
        let h = compact_header(RpuType::EmulationPrevented, 0, false, false);
        let p = payload(h, 128, false);
        assert!(!p.has_dynamic_metadata());
    }

    #[test]
    fn test_rpu_payload_has_dynamic_metadata_zero_size() {
        let h = compact_header(RpuType::EmulationPrevented, 0, false, false);
        let p = payload(h, 0, true);
        assert!(!p.has_dynamic_metadata());
    }

    #[test]
    fn test_rpu_payload_rpu_format_stored() {
        let h = compact_header(RpuType::StraightAnnexB, 1, false, false);
        assert_eq!(h.rpu_format, 1);
    }
}
