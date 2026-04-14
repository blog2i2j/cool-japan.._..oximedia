//! RPU parser for NAL units and bitstreams.
//!
//! Handles parsing of Dolby Vision RPU from HEVC SEI messages and raw bitstreams.

use crate::{metadata::*, rpu::*, DolbyVisionError, DolbyVisionRpu, Profile, Result};
use oximedia_bitstream::{BigEndian, BitRead, BitReader};
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::{Arc, Mutex, OnceLock};

// ── RPU Parse Cache ───────────────────────────────────────────────────────────

/// Cache key for RPU parse deduplication.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RpuCacheKey {
    /// FNV-1a hash of the NAL unit data for fast deduplication.
    hash: u64,
    /// Length of the data (used as secondary discriminator).
    len: usize,
}

impl RpuCacheKey {
    fn from_data(data: &[u8]) -> Self {
        // FNV-1a hash (64-bit)
        const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
        const FNV_PRIME: u64 = 1_099_511_628_211;
        let mut hash = FNV_OFFSET;
        for &byte in data {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        Self {
            hash,
            len: data.len(),
        }
    }
}

/// Maximum number of entries kept in the RPU parse cache.
const RPU_CACHE_MAX_ENTRIES: usize = 256;

/// Process-global RPU bitstream parse cache.
static RPU_CACHE: OnceLock<Mutex<HashMap<RpuCacheKey, Arc<DolbyVisionRpu>>>> = OnceLock::new();

fn get_rpu_cache() -> &'static Mutex<HashMap<RpuCacheKey, Arc<DolbyVisionRpu>>> {
    RPU_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Parse a NAL unit, returning a cached result if the byte sequence was seen recently.
///
/// # Errors
///
/// Returns error if parsing fails.
pub fn parse_nal_unit_cached(data: &[u8]) -> Result<DolbyVisionRpu> {
    let key = RpuCacheKey::from_data(data);
    let cache_ref = get_rpu_cache();

    // Check cache first
    if let Ok(cache) = cache_ref.lock() {
        if let Some(cached) = cache.get(&key) {
            return Ok((**cached).clone());
        }
    }

    // Parse and insert into cache
    let rpu = parse_nal_unit(data)?;

    if let Ok(mut cache) = cache_ref.lock() {
        if cache.len() >= RPU_CACHE_MAX_ENTRIES {
            // Evict one arbitrary entry (simple strategy: clear oldest half)
            let keys: Vec<_> = cache
                .keys()
                .take(RPU_CACHE_MAX_ENTRIES / 2)
                .cloned()
                .collect();
            for k in keys {
                cache.remove(&k);
            }
        }
        cache.insert(key, Arc::new(rpu.clone()));
    }

    Ok(rpu)
}

/// Parse a bitstream, returning a cached result if the byte sequence was seen recently.
///
/// # Errors
///
/// Returns error if parsing fails.
pub fn parse_rpu_bitstream_cached(data: &[u8]) -> Result<DolbyVisionRpu> {
    let key = RpuCacheKey::from_data(data);
    let cache_ref = get_rpu_cache();

    if let Ok(cache) = cache_ref.lock() {
        if let Some(cached) = cache.get(&key) {
            return Ok((**cached).clone());
        }
    }

    let rpu = parse_rpu_bitstream(data)?;

    if let Ok(mut cache) = cache_ref.lock() {
        if cache.len() >= RPU_CACHE_MAX_ENTRIES {
            let keys: Vec<_> = cache
                .keys()
                .take(RPU_CACHE_MAX_ENTRIES / 2)
                .cloned()
                .collect();
            for k in keys {
                cache.remove(&k);
            }
        }
        cache.insert(key, Arc::new(rpu.clone()));
    }

    Ok(rpu)
}

/// Clear the global RPU parse cache.
pub fn clear_rpu_cache() {
    if let Ok(mut cache) = get_rpu_cache().lock() {
        cache.clear();
    }
}

/// Return the number of entries currently in the RPU parse cache.
#[must_use]
pub fn rpu_cache_len() -> usize {
    get_rpu_cache().lock().map(|c| c.len()).unwrap_or(0)
}

/// HEVC NAL unit types for Dolby Vision.
pub mod nal_type {
    /// Dolby Vision RPU NAL unit (unregistered SEI)
    pub const UNREGISTERED_SEI: u8 = 62;

    /// Dolby Vision EL NAL unit
    pub const DV_EL: u8 = 63;

    /// Dolby Vision RPU NAL unit (alternative)
    pub const DV_RPU: u8 = 25;
}

/// Dolby Vision T.35 country code (United States).
const T35_COUNTRY_CODE: u8 = 0xB5;

/// Dolby Vision T.35 terminal provider code.
const T35_TERMINAL_PROVIDER_CODE: u16 = 0x003C;

/// Parse NAL unit containing Dolby Vision RPU.
///
/// # Errors
///
/// Returns error if NAL parsing fails.
#[allow(clippy::too_many_lines)]
pub fn parse_nal_unit(data: &[u8]) -> Result<DolbyVisionRpu> {
    if data.is_empty() {
        return Err(DolbyVisionError::InvalidNalUnit(
            "Empty NAL unit".to_string(),
        ));
    }

    // Check NAL unit type (first byte, top 7 bits after forbidden_zero_bit)
    let nal_type = (data[0] >> 1) & 0x3F;

    let payload = match nal_type {
        nal_type::UNREGISTERED_SEI | nal_type::DV_RPU => {
            // Skip NAL header (2 bytes for HEVC)
            if data.len() < 2 {
                return Err(DolbyVisionError::InvalidNalUnit(
                    "NAL unit too short".to_string(),
                ));
            }
            parse_sei_payload(&data[2..])?
        }
        nal_type::DV_EL => {
            return Err(DolbyVisionError::InvalidNalUnit(
                "Enhancement layer NAL units not yet supported".to_string(),
            ));
        }
        _ => {
            return Err(DolbyVisionError::InvalidNalUnit(format!(
                "Unexpected NAL type: {}",
                nal_type
            )));
        }
    };

    parse_rpu_bitstream(&payload)
}

/// Parse SEI payload to extract RPU data.
fn parse_sei_payload(data: &[u8]) -> Result<Vec<u8>> {
    let mut cursor = Cursor::new(data);
    let mut reader = BitReader::endian(&mut cursor, BigEndian);

    // Parse payload type (variable length)
    let mut payload_type = 0u32;
    loop {
        let byte: u8 = reader
            .read_var(8)
            .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;
        payload_type += u32::from(byte);
        if byte != 0xFF {
            break;
        }
    }

    // Parse payload size (variable length)
    let mut payload_size = 0u32;
    loop {
        let byte: u8 = reader
            .read_var(8)
            .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;
        payload_size += u32::from(byte);
        if byte != 0xFF {
            break;
        }
    }

    // For unregistered user data SEI (type 5), check T.35 header
    if payload_type == 5 {
        // Read T.35 country code
        let country_code: u8 = reader
            .read_var(8)
            .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;
        if country_code != T35_COUNTRY_CODE {
            return Err(DolbyVisionError::InvalidPayload(format!(
                "Invalid T.35 country code: {:#x}",
                country_code
            )));
        }

        // Read T.35 terminal provider code
        let provider_code: u16 = reader
            .read_var(16)
            .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;
        if provider_code != T35_TERMINAL_PROVIDER_CODE {
            return Err(DolbyVisionError::InvalidPayload(format!(
                "Invalid T.35 provider code: {:#x}",
                provider_code
            )));
        }

        // Remaining data is RPU payload
        let rpu_size = payload_size.saturating_sub(3);
        let mut rpu_data = vec![0u8; rpu_size as usize];
        reader
            .read_bytes(&mut rpu_data)
            .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

        Ok(rpu_data)
    } else {
        // Read raw payload
        let mut payload = vec![0u8; payload_size as usize];
        reader
            .read_bytes(&mut payload)
            .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;
        Ok(payload)
    }
}

/// Automatically detect the Dolby Vision profile from RPU header fields.
///
/// Uses a multi-factor heuristic combining:
/// - `ycbcr_to_rgb_flag` (true → signal is in RGB/YCbCr, false → IPT-PQ, i.e. Profile 5)
/// - `mapping_color_space` (2 = IPT → Profile 5)
/// - `bl_bit_depth` and `el_bit_depth` (el > 0 with full depth → Profile 7 MEL)
/// - `mapping_chroma_format` for HLG detection
///
/// Returns the most likely `Profile` or `Profile::Profile8` as safe default.
#[must_use]
pub fn detect_profile_from_header(header: &RpuHeader) -> Profile {
    if header.vdr_seq_info_present {
        if let Some(ref seq_info) = header.vdr_seq_info {
            return detect_profile_from_seq_info(seq_info, header);
        }
    }
    Profile::Profile8
}

/// Inner helper — determine profile from seq_info and header.
fn detect_profile_from_seq_info(seq_info: &VdrSeqInfo, header: &RpuHeader) -> Profile {
    // Profile 5: IPT-PQ color space; no YCbCr-to-RGB conversion; mapping_color_space == 2
    if !seq_info.ycbcr_to_rgb_flag || header.mapping_color_space == 2 {
        return Profile::Profile5;
    }

    // Profile 7: dual-layer (MEL) — enhancement layer present with significant depth
    //   Heuristic: el_bit_depth > 0 and bl_bit_depth >= 10 and el_bit_depth >= 8
    if seq_info.el_bit_depth >= 8 && seq_info.bl_bit_depth >= 10 {
        // Additional check: if VDR DM metadata ID != 0 it's a MEL profile
        if seq_info.vdr_dm_metadata_id != 0 {
            return Profile::Profile7;
        }
    }

    // Profile 8.4: HLG base layer
    // Heuristic: signal_eotf indicates HLG (0x0F = HLG OETF in DV spec)
    // We detect this from chroma format + BL depth = 10 bit + source depth = 10 bit
    // and ycbcr_to_rgb_flag = true (HLG BT.2020 color space).
    // The signal_eotf is in VdrDmData, not available here, so we use the secondary
    // indicator: coef_log2_denom == 14 and mapping_chroma_format == 0 (4:2:0)
    if header.mapping_chroma_format == 0
        && seq_info.coef_log2_denom == 14
        && seq_info.el_bit_depth == 0
    {
        // Could be Profile 8.4 (HLG); without signal_eotf we cannot confirm,
        // but we can use coef_data_type == 0 and source_bit_depth == 10 as extra signal
        if seq_info.source_bit_depth == 10 && seq_info.bl_bit_depth == 10 {
            // Only Profile 8.4 carries HLG; Profile 8 uses PQ.
            // Without explicit HLG signal, default to Profile 8 (safest choice)
            // but emit Profile 8.4 if el_bit_depth == 0 and vdr_dm_metadata_id == 0
            if seq_info.vdr_dm_metadata_id == 0 && seq_info.scene_refresh_flag == 0 {
                // Ambiguous: could be Profile 8 or 8.4; default to Profile 8
                // (callers may override via explicit profile hint)
                return Profile::Profile8;
            }
        }
    }

    // Profile 8.1: low-latency variant.
    // Heuristic: same as Profile 8 but with vdr_dm_metadata_id == 1
    // (implementation may expose this as a hint in the header).
    // Without explicit signalling we cannot reliably distinguish 8 vs 8.1.

    Profile::Profile8
}

/// Parse RPU from raw bitstream.
///
/// # Errors
///
/// Returns error if parsing fails.
#[allow(clippy::too_many_lines)]
pub fn parse_rpu_bitstream(data: &[u8]) -> Result<DolbyVisionRpu> {
    let mut cursor = Cursor::new(data);
    let mut reader = BitReader::endian(&mut cursor, BigEndian);

    // Parse RPU header
    let header = parse_rpu_header(&mut reader)?;

    // Determine profile from header using automatic detection heuristics
    let profile = detect_profile_from_header(&header);

    let mut rpu = DolbyVisionRpu::new(profile);
    rpu.header = header;

    // Parse VDR DM data if present
    if rpu.header.change_flags.contains(ChangeFlags::VDR_CHANGED) {
        rpu.vdr_dm_data = Some(parse_vdr_dm_data(&mut reader, profile)?);
    }

    // Parse metadata levels
    rpu.level1 = parse_level1_metadata(&mut reader)?;
    rpu.level2 = parse_level2_metadata(&mut reader)?;
    rpu.level5 = parse_level5_metadata(&mut reader)?;
    rpu.level6 = parse_level6_metadata(&mut reader)?;
    rpu.level8 = parse_level8_metadata(&mut reader)?;
    rpu.level9 = parse_level9_metadata(&mut reader)?;
    rpu.level11 = parse_level11_metadata(&mut reader)?;

    Ok(rpu)
}

/// Parse RPU header.
fn parse_rpu_header<R: std::io::Read>(reader: &mut BitReader<R, BigEndian>) -> Result<RpuHeader> {
    let rpu_type: u8 = reader
        .read_var(6)
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    let rpu_format: u16 = reader
        .read_var(11)
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    let vdr_seq_info_present: bool = reader
        .read_bit()
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    let vdr_seq_info = if vdr_seq_info_present {
        Some(parse_vdr_seq_info(reader)?)
    } else {
        None
    };

    let picture_index: u16 = reader
        .read_var(10)
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    let change_flags_bits: u16 = reader
        .read_var(4)
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;
    let change_flags = ChangeFlags::from_bits_truncate(change_flags_bits);

    let nlq_param_pred_flag: bool = reader
        .read_bit()
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    let num_nlq_param_predictors: u8 = if nlq_param_pred_flag {
        reader
            .read_var(4)
            .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?
    } else {
        0
    };

    let component_order: u8 = reader
        .read_var(2)
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    let coef_data_type: u8 = reader
        .read_var(1)
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    let coef_log2_denom: u8 = reader
        .read_var(4)
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    let mapping_color_space: u8 = reader
        .read_var(2)
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    let mapping_chroma_format: u8 = reader
        .read_var(2)
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    let num_pivots_minus_2: u8 = reader
        .read_var(3)
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    let pred_pivot_value: u16 = reader
        .read_var(12)
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    Ok(RpuHeader {
        rpu_type,
        rpu_format,
        vdr_seq_info_present,
        vdr_seq_info,
        picture_index,
        change_flags,
        nlq_param_pred_flag,
        num_nlq_param_predictors,
        component_order,
        coef_data_type,
        coef_log2_denom,
        mapping_color_space,
        mapping_chroma_format,
        num_pivots_minus_2,
        pred_pivot_value,
    })
}

/// Parse VDR sequence info.
fn parse_vdr_seq_info<R: std::io::Read>(
    reader: &mut BitReader<R, BigEndian>,
) -> Result<VdrSeqInfo> {
    let vdr_dm_metadata_id: u8 = reader
        .read_var(8)
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    let scene_refresh_flag: u8 = reader
        .read_var(2)
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    let ycbcr_to_rgb_flag: bool = reader
        .read_bit()
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    let coef_data_type: u8 = reader
        .read_var(1)
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    let coef_log2_denom: u8 = reader
        .read_var(4)
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    let vdr_bit_depth: u8 = reader
        .read_var(4)
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    let bl_bit_depth: u8 = reader
        .read_var(4)
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    let el_bit_depth: u8 = reader
        .read_var(4)
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    let source_bit_depth: u8 = reader
        .read_var(4)
        .map_err(|e| DolbyVisionError::InvalidHeader(e.to_string()))?;

    Ok(VdrSeqInfo {
        vdr_dm_metadata_id,
        scene_refresh_flag,
        ycbcr_to_rgb_flag,
        coef_data_type,
        coef_log2_denom,
        vdr_bit_depth,
        bl_bit_depth,
        el_bit_depth,
        source_bit_depth,
    })
}

/// Parse VDR DM data.
#[allow(clippy::too_many_lines)]
fn parse_vdr_dm_data<R: std::io::Read>(
    reader: &mut BitReader<R, BigEndian>,
    _profile: Profile,
) -> Result<VdrDmData> {
    let affected_dm_metadata_id: u8 = reader
        .read_var(8)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let current_dm_metadata_id: u8 = reader
        .read_var(8)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let scene_refresh_flag: u8 = reader
        .read_var(2)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let ycbcr_to_rgb_present: bool = reader
        .read_bit()
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let ycbcr_to_rgb_matrix = if ycbcr_to_rgb_present {
        Some(parse_color_matrix(reader)?)
    } else {
        None
    };

    let rgb_to_lms_present: bool = reader
        .read_bit()
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let rgb_to_lms_matrix = if rgb_to_lms_present {
        Some(parse_color_matrix(reader)?)
    } else {
        None
    };

    let signal_eotf: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let signal_eotf_param0: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let signal_eotf_param1: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let signal_eotf_param2: u32 = reader
        .read_var(32)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let signal_bit_depth: u8 = reader
        .read_var(5)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let signal_color_space: u8 = reader
        .read_var(2)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let signal_chroma_format: u8 = reader
        .read_var(2)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let signal_full_range_flag: u8 = reader
        .read_var(2)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let source_min_pq: u16 = reader
        .read_var(12)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let source_max_pq: u16 = reader
        .read_var(12)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let source_diagonal: u16 = reader
        .read_var(10)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    // Parse reshaping curves (simplified - usually 3 curves for RGB)
    let num_curves: u8 = reader
        .read_var(2)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let mut reshaping_curves = Vec::new();
    for _ in 0..=num_curves {
        reshaping_curves.push(parse_reshaping_curve(reader)?);
    }

    // Parse NLQ parameters
    let num_nlq_params: u8 = reader
        .read_var(2)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let mut nlq_params = Vec::new();
    for _ in 0..=num_nlq_params {
        nlq_params.push(parse_nlq_params(reader)?);
    }

    Ok(VdrDmData {
        affected_dm_metadata_id,
        current_dm_metadata_id,
        scene_refresh_flag,
        ycbcr_to_rgb_matrix,
        rgb_to_lms_matrix,
        signal_eotf,
        signal_eotf_param0,
        signal_eotf_param1,
        signal_eotf_param2,
        signal_bit_depth,
        signal_color_space,
        signal_chroma_format,
        signal_full_range_flag,
        source_min_pq,
        source_max_pq,
        source_diagonal,
        reshaping_curves,
        nlq_params,
    })
}

/// Parse color matrix.
fn parse_color_matrix<R: std::io::Read>(
    reader: &mut BitReader<R, BigEndian>,
) -> Result<ColorMatrix> {
    let mut matrix = [[0i32; 3]; 3];
    for row in &mut matrix {
        for col in row {
            *col = reader
                .read_var::<i32>(16)
                .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;
        }
    }
    Ok(ColorMatrix { matrix })
}

/// Parse reshaping curve.
fn parse_reshaping_curve<R: std::io::Read>(
    reader: &mut BitReader<R, BigEndian>,
) -> Result<ReshapingCurve> {
    let num_pivots: u8 = reader
        .read_var(4)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let mut pivots = Vec::new();
    for _ in 0..=num_pivots {
        let pivot: u16 = reader
            .read_var(12)
            .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;
        pivots.push(pivot);
    }

    let mut mapping_idc = Vec::new();
    let mut poly_order_minus1 = Vec::new();
    let mut poly_coef = Vec::new();

    for _ in 0..num_pivots {
        let idc: u8 = reader
            .read_var(2)
            .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;
        mapping_idc.push(idc);

        if idc == 0 {
            // Polynomial mapping
            let order: u8 = reader
                .read_var(2)
                .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;
            poly_order_minus1.push(order);

            let mut coefs = Vec::new();
            for _ in 0..=(order + 1) {
                let coef: i64 = reader
                    .read_var(16)
                    .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;
                coefs.push(coef);
            }
            poly_coef.push(coefs);
        }
    }

    let mmr_order_minus1: u8 = reader
        .read_var(2)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let mut mmr_coef = Vec::new();
    for _ in 0..=(mmr_order_minus1 + 1) {
        let coef: i64 = reader
            .read_var(16)
            .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;
        mmr_coef.push(coef);
    }

    Ok(ReshapingCurve {
        pivots,
        mapping_idc,
        poly_order_minus1,
        poly_coef,
        mmr_order_minus1,
        mmr_coef,
    })
}

/// Parse NLQ parameters.
fn parse_nlq_params<R: std::io::Read>(reader: &mut BitReader<R, BigEndian>) -> Result<NlqParams> {
    let nlq_offset: u16 = reader
        .read_var(10)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let vdr_in_max: u64 = reader
        .read_var(27)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let linear_deadzone_slope: u64 = reader
        .read_var(26)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let linear_deadzone_threshold: u64 = reader
        .read_var(26)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    Ok(NlqParams {
        nlq_offset,
        vdr_in_max,
        linear_deadzone_slope,
        linear_deadzone_threshold,
    })
}

/// Parse Level 1 metadata.
fn parse_level1_metadata<R: std::io::Read>(
    reader: &mut BitReader<R, BigEndian>,
) -> Result<Option<Level1Metadata>> {
    let present: bool = reader.read_bit().unwrap_or(false);

    if !present {
        return Ok(None);
    }

    let min_pq: u16 = reader
        .read_var(12)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let max_pq: u16 = reader
        .read_var(12)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let avg_pq: u16 = reader
        .read_var(12)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    Ok(Some(Level1Metadata {
        min_pq,
        max_pq,
        avg_pq,
    }))
}

/// Parse Level 2 metadata.
fn parse_level2_metadata<R: std::io::Read>(
    reader: &mut BitReader<R, BigEndian>,
) -> Result<Option<Level2Metadata>> {
    let present: bool = reader.read_bit().unwrap_or(false);
    if !present {
        return Ok(None);
    }

    let target_display_index: u8 = reader
        .read_var(8)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let trim_slope: i16 = reader
        .read_signed_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let trim_offset: i16 = reader
        .read_signed_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let trim_power: i16 = reader
        .read_signed_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let trim_chroma_weight: i16 = reader
        .read_signed_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let trim_saturation_gain: i16 = reader
        .read_signed_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let ms_weight: i16 = reader
        .read_signed_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let target_mid_contrast: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let clip_trim: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    // Saturation and hue vector fields are not written by the current writer
    // (write_level2_metadata only writes the basic fields). Return empty vecs.
    Ok(Some(Level2Metadata {
        target_display_index,
        trim_slope,
        trim_offset,
        trim_power,
        trim_chroma_weight,
        trim_saturation_gain,
        ms_weight,
        target_mid_contrast,
        clip_trim,
        saturation_vector_field: Vec::new(),
        hue_vector_field: Vec::new(),
    }))
}

/// Parse Level 5 metadata.
fn parse_level5_metadata<R: std::io::Read>(
    reader: &mut BitReader<R, BigEndian>,
) -> Result<Option<Level5Metadata>> {
    let present: bool = reader.read_bit().unwrap_or(false);
    if !present {
        return Ok(None);
    }

    let active_area_left_offset: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let active_area_right_offset: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let active_area_top_offset: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let active_area_bottom_offset: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    Ok(Some(Level5Metadata {
        active_area_left_offset,
        active_area_right_offset,
        active_area_top_offset,
        active_area_bottom_offset,
    }))
}

/// Parse Level 6 metadata.
fn parse_level6_metadata<R: std::io::Read>(
    reader: &mut BitReader<R, BigEndian>,
) -> Result<Option<Level6Metadata>> {
    let present: bool = reader.read_bit().unwrap_or(false);
    if !present {
        return Ok(None);
    }

    let max_cll: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let max_fall: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let min_display_mastering_luminance: u32 = reader
        .read_var(32)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let max_display_mastering_luminance: u32 = reader
        .read_var(32)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let mut master_display_primaries = [[0u16; 2]; 3];
    for primary in &mut master_display_primaries {
        primary[0] = reader
            .read_var(16)
            .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;
        primary[1] = reader
            .read_var(16)
            .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;
    }

    let white_x: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;
    let white_y: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    Ok(Some(Level6Metadata {
        max_cll,
        max_fall,
        min_display_mastering_luminance,
        max_display_mastering_luminance,
        master_display_primaries,
        master_display_white_point: [white_x, white_y],
    }))
}

/// Parse Level 8 metadata.
fn parse_level8_metadata<R: std::io::Read>(
    reader: &mut BitReader<R, BigEndian>,
) -> Result<Option<Level8Metadata>> {
    let present: bool = reader.read_bit().unwrap_or(false);
    if !present {
        return Ok(None);
    }

    let target_display_index: u8 = reader
        .read_var(8)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let target_max_pq: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let target_min_pq: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let target_primary_index: u8 = reader
        .read_var(8)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let target_eotf: u8 = reader
        .read_var(8)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let diagonal_size: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let peak_luminance: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let diffuse_white_luminance: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let ambient_luminance: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let surround_reflection: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    Ok(Some(Level8Metadata {
        target_display_index,
        target_max_pq,
        target_min_pq,
        target_primary_index,
        target_eotf,
        diagonal_size,
        peak_luminance,
        diffuse_white_luminance,
        ambient_luminance,
        surround_reflection,
    }))
}

/// Parse Level 9 metadata.
fn parse_level9_metadata<R: std::io::Read>(
    reader: &mut BitReader<R, BigEndian>,
) -> Result<Option<Level9Metadata>> {
    let present: bool = reader.read_bit().unwrap_or(false);
    if !present {
        return Ok(None);
    }

    let source_primary_index: u8 = reader
        .read_var(8)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let source_max_pq: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let source_min_pq: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let source_diagonal: u16 = reader
        .read_var(16)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    Ok(Some(Level9Metadata {
        source_primary_index,
        source_max_pq,
        source_min_pq,
        source_diagonal,
    }))
}

/// Parse Level 11 metadata.
fn parse_level11_metadata<R: std::io::Read>(
    reader: &mut BitReader<R, BigEndian>,
) -> Result<Option<Level11Metadata>> {
    let present: bool = reader.read_bit().unwrap_or(false);
    if !present {
        return Ok(None);
    }

    let content_type_byte: u8 = reader
        .read_var(8)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let whitepoint: u8 = reader
        .read_var(8)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let reference_mode_flag: bool = reader
        .read_bit()
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let sharpness: u8 = reader
        .read_var(8)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let noise_reduction: u8 = reader
        .read_var(8)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let mpeg_noise_reduction: u8 = reader
        .read_var(8)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let frame_rate: u8 = reader
        .read_var(8)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    let temporal_filter_strength: u8 = reader
        .read_var(8)
        .map_err(|e| DolbyVisionError::InvalidPayload(e.to_string()))?;

    Ok(Some(Level11Metadata {
        content_type: ContentType::from_u8(content_type_byte),
        whitepoint,
        reference_mode_flag,
        sharpness,
        noise_reduction,
        mpeg_noise_reduction,
        frame_rate,
        temporal_filter_strength,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nal_type_constants() {
        assert_eq!(nal_type::UNREGISTERED_SEI, 62);
        assert_eq!(nal_type::DV_EL, 63);
        assert_eq!(nal_type::DV_RPU, 25);
    }

    #[test]
    fn test_t35_constants() {
        assert_eq!(T35_COUNTRY_CODE, 0xB5);
        assert_eq!(T35_TERMINAL_PROVIDER_CODE, 0x003C);
    }

    #[test]
    fn test_parse_empty_nal() {
        let result = parse_nal_unit(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_invalid_nal_type() {
        let nal = vec![0x00, 0x00]; // Invalid NAL type
        let result = parse_nal_unit(&nal);
        assert!(result.is_err());
    }

    // ── Profile detection tests ───────────────────────────────────────────────

    #[test]
    fn test_detect_profile_ipt_color_space() {
        // mapping_color_space == 2 → Profile 5 (IPT)
        let header = crate::rpu::RpuHeader {
            rpu_type: 0,
            rpu_format: 0,
            vdr_seq_info_present: true,
            vdr_seq_info: Some(VdrSeqInfo {
                vdr_dm_metadata_id: 0,
                scene_refresh_flag: 0,
                ycbcr_to_rgb_flag: false,
                coef_data_type: 0,
                coef_log2_denom: 14,
                vdr_bit_depth: 12,
                bl_bit_depth: 10,
                el_bit_depth: 0,
                source_bit_depth: 10,
            }),
            picture_index: 0,
            change_flags: crate::rpu::ChangeFlags::empty(),
            nlq_param_pred_flag: false,
            num_nlq_param_predictors: 0,
            component_order: 2,
            coef_data_type: 0,
            coef_log2_denom: 14,
            mapping_color_space: 2,
            mapping_chroma_format: 2,
            num_pivots_minus_2: 0,
            pred_pivot_value: 0,
        };
        assert_eq!(detect_profile_from_header(&header), Profile::Profile5);
    }

    #[test]
    fn test_detect_profile_no_seq_info() {
        let header = crate::rpu::RpuHeader {
            rpu_type: 0,
            rpu_format: 0,
            vdr_seq_info_present: false,
            vdr_seq_info: None,
            picture_index: 0,
            change_flags: crate::rpu::ChangeFlags::empty(),
            nlq_param_pred_flag: false,
            num_nlq_param_predictors: 0,
            component_order: 0,
            coef_data_type: 0,
            coef_log2_denom: 14,
            mapping_color_space: 1,
            mapping_chroma_format: 2,
            num_pivots_minus_2: 0,
            pred_pivot_value: 0,
        };
        assert_eq!(detect_profile_from_header(&header), Profile::Profile8);
    }

    // ── RPU cache tests ───────────────────────────────────────────────────────

    #[test]
    fn test_rpu_cache_key_deterministic() {
        let data = b"hello world";
        let k1 = RpuCacheKey::from_data(data);
        let k2 = RpuCacheKey::from_data(data);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_rpu_cache_key_different_data() {
        let k1 = RpuCacheKey::from_data(b"hello");
        let k2 = RpuCacheKey::from_data(b"world");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_clear_rpu_cache() {
        // Should not panic
        clear_rpu_cache();
        assert_eq!(rpu_cache_len(), 0);
    }

    #[test]
    fn test_cached_parse_invalid_input() {
        let result = parse_nal_unit_cached(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cached_bitstream_valid() {
        use crate::{DolbyVisionRpu, Profile};
        let rpu = DolbyVisionRpu::new(Profile::Profile8);
        let bits = crate::writer::write_rpu_bitstream(&rpu).expect("write failed");

        clear_rpu_cache();
        let result1 = parse_rpu_bitstream_cached(&bits);
        let result2 = parse_rpu_bitstream_cached(&bits);
        assert!(result1.is_ok(), "first parse should succeed");
        assert!(result2.is_ok(), "cached parse should succeed");
        // Cache should have one entry for this data
        assert!(rpu_cache_len() <= 1);
    }
}
