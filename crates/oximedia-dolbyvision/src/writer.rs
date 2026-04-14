//! RPU writer for creating NAL units and bitstreams.
//!
//! Handles writing of Dolby Vision RPU to HEVC SEI messages and raw bitstreams.

use crate::{metadata::*, rpu::*, DolbyVisionError, DolbyVisionRpu, Profile, Result};
use oximedia_bitstream::{BigEndian, BitWrite, BitWriter};
use std::io::Cursor;

/// Dolby Vision T.35 country code (United States).
const T35_COUNTRY_CODE: u8 = 0xB5;

/// Dolby Vision T.35 terminal provider code.
const T35_TERMINAL_PROVIDER_CODE: u16 = 0x003C;

/// HEVC NAL unit type for unregistered SEI.
const NAL_TYPE_SEI_UNREGISTERED: u8 = 39;

/// SEI payload type for unregistered user data.
const SEI_TYPE_USER_DATA_UNREGISTERED: u8 = 5;

/// Write RPU to NAL unit bytes (including NAL header).
///
/// # Errors
///
/// Returns error if writing fails.
pub fn write_nal_unit(rpu: &DolbyVisionRpu) -> Result<Vec<u8>> {
    let rpu_payload = write_rpu_bitstream(rpu)?;
    let sei_payload = create_sei_payload(&rpu_payload)?;

    // Create HEVC NAL unit header (2 bytes)
    let mut nal_data = Vec::new();

    // NAL header byte 1: forbidden_zero_bit (1) + nal_unit_type (6) + nuh_layer_id (6 bits, upper)
    // forbidden_zero_bit = 0, nal_unit_type = 39 (SEI), nuh_layer_id = 0
    let byte1 = (NAL_TYPE_SEI_UNREGISTERED << 1) & 0x7E;
    nal_data.push(byte1);

    // NAL header byte 2: nuh_layer_id (lower 5 bits) + nuh_temporal_id_plus1 (3 bits)
    // nuh_layer_id = 0, nuh_temporal_id_plus1 = 1
    let byte2 = 0x01;
    nal_data.push(byte2);

    // Add SEI payload
    nal_data.extend_from_slice(&sei_payload);

    // Add emulation prevention if needed
    let nal_data = add_emulation_prevention(&nal_data);

    Ok(nal_data)
}

/// Write RPU to raw bitstream (without NAL wrapper).
///
/// # Errors
///
/// Returns error if writing fails.
pub fn write_rpu_bitstream(rpu: &DolbyVisionRpu) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();
    let mut cursor = Cursor::new(&mut buffer);
    let mut writer = BitWriter::endian(&mut cursor, BigEndian);

    // Write RPU header
    write_rpu_header(&mut writer, &rpu.header)?;

    // Write VDR DM data if present and changed
    if rpu.header.change_flags.contains(ChangeFlags::VDR_CHANGED) {
        if let Some(ref vdr_dm_data) = rpu.vdr_dm_data {
            write_vdr_dm_data(&mut writer, vdr_dm_data, rpu.profile)?;
        }
    }

    // Write metadata levels
    write_level1_metadata(&mut writer, rpu.level1.as_ref())?;
    write_level2_metadata(&mut writer, rpu.level2.as_ref())?;
    write_level5_metadata(&mut writer, rpu.level5.as_ref())?;
    write_level6_metadata(&mut writer, rpu.level6.as_ref())?;
    write_level8_metadata(&mut writer, rpu.level8.as_ref())?;
    write_level9_metadata(&mut writer, rpu.level9.as_ref())?;
    write_level11_metadata(&mut writer, rpu.level11.as_ref())?;

    // Byte align
    writer.byte_align().map_err(DolbyVisionError::Io)?;

    Ok(buffer)
}

/// Create SEI payload with T.35 header.
fn create_sei_payload(rpu_data: &[u8]) -> Result<Vec<u8>> {
    let mut payload = Vec::new();
    let mut cursor = Cursor::new(&mut payload);
    let mut writer = BitWriter::endian(&mut cursor, BigEndian);

    // Write SEI payload type (variable length)
    let payload_type = SEI_TYPE_USER_DATA_UNREGISTERED;
    write_sei_size(&mut writer, u32::from(payload_type))?;

    // Calculate payload size: T.35 header (3 bytes) + RPU data
    let payload_size = 3 + rpu_data.len();
    write_sei_size(&mut writer, payload_size as u32)?;

    // Write T.35 country code
    writer
        .write_var(8, T35_COUNTRY_CODE)
        .map_err(DolbyVisionError::Io)?;

    // Write T.35 terminal provider code
    writer
        .write_var(16, T35_TERMINAL_PROVIDER_CODE)
        .map_err(DolbyVisionError::Io)?;

    // Write RPU data
    writer.write_bytes(rpu_data).map_err(DolbyVisionError::Io)?;

    // Add trailing bits (0x80 for byte alignment)
    writer.write_var(8, 0x80u8).map_err(DolbyVisionError::Io)?;

    Ok(payload)
}

/// Write SEI size using variable-length encoding.
fn write_sei_size<W: std::io::Write>(
    writer: &mut BitWriter<W, BigEndian>,
    mut size: u32,
) -> Result<()> {
    while size > 255 {
        writer.write_var(8, 0xFFu8).map_err(DolbyVisionError::Io)?;
        size -= 255;
    }
    writer
        .write_var(8, size as u8)
        .map_err(DolbyVisionError::Io)?;
    Ok(())
}

/// Write RPU header.
fn write_rpu_header<W: std::io::Write>(
    writer: &mut BitWriter<W, BigEndian>,
    header: &RpuHeader,
) -> Result<()> {
    writer
        .write_var(6, header.rpu_type)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(11, header.rpu_format)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_bit(header.vdr_seq_info_present)
        .map_err(DolbyVisionError::Io)?;

    if let Some(ref seq_info) = header.vdr_seq_info {
        write_vdr_seq_info(writer, seq_info)?;
    }

    writer
        .write_var(10, header.picture_index)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(4, header.change_flags.bits())
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_bit(header.nlq_param_pred_flag)
        .map_err(DolbyVisionError::Io)?;

    if header.nlq_param_pred_flag {
        writer
            .write_var(4, header.num_nlq_param_predictors)
            .map_err(DolbyVisionError::Io)?;
    }

    writer
        .write_var(2, header.component_order)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(1, header.coef_data_type)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(4, header.coef_log2_denom)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(2, header.mapping_color_space)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(2, header.mapping_chroma_format)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(3, header.num_pivots_minus_2)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(12, header.pred_pivot_value)
        .map_err(DolbyVisionError::Io)?;

    Ok(())
}

/// Write VDR sequence info.
fn write_vdr_seq_info<W: std::io::Write>(
    writer: &mut BitWriter<W, BigEndian>,
    seq_info: &VdrSeqInfo,
) -> Result<()> {
    writer
        .write_var(8, seq_info.vdr_dm_metadata_id)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(2, seq_info.scene_refresh_flag)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_bit(seq_info.ycbcr_to_rgb_flag)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(1, seq_info.coef_data_type)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(4, seq_info.coef_log2_denom)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(4, seq_info.vdr_bit_depth)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(4, seq_info.bl_bit_depth)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(4, seq_info.el_bit_depth)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(4, seq_info.source_bit_depth)
        .map_err(DolbyVisionError::Io)?;

    Ok(())
}

/// Write VDR DM data.
#[allow(clippy::too_many_lines)]
fn write_vdr_dm_data<W: std::io::Write>(
    writer: &mut BitWriter<W, BigEndian>,
    vdr_dm_data: &VdrDmData,
    _profile: Profile,
) -> Result<()> {
    writer
        .write_var(8, vdr_dm_data.affected_dm_metadata_id)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(8, vdr_dm_data.current_dm_metadata_id)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(2, vdr_dm_data.scene_refresh_flag)
        .map_err(DolbyVisionError::Io)?;

    let ycbcr_to_rgb_present = vdr_dm_data.ycbcr_to_rgb_matrix.is_some();
    writer
        .write_bit(ycbcr_to_rgb_present)
        .map_err(DolbyVisionError::Io)?;

    if let Some(ref matrix) = vdr_dm_data.ycbcr_to_rgb_matrix {
        write_color_matrix(writer, matrix)?;
    }

    let rgb_to_lms_present = vdr_dm_data.rgb_to_lms_matrix.is_some();
    writer
        .write_bit(rgb_to_lms_present)
        .map_err(DolbyVisionError::Io)?;

    if let Some(ref matrix) = vdr_dm_data.rgb_to_lms_matrix {
        write_color_matrix(writer, matrix)?;
    }

    writer
        .write_var(16, vdr_dm_data.signal_eotf)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(16, vdr_dm_data.signal_eotf_param0)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(16, vdr_dm_data.signal_eotf_param1)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(32, vdr_dm_data.signal_eotf_param2)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(5, vdr_dm_data.signal_bit_depth)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(2, vdr_dm_data.signal_color_space)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(2, vdr_dm_data.signal_chroma_format)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(2, vdr_dm_data.signal_full_range_flag)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(12, vdr_dm_data.source_min_pq)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(12, vdr_dm_data.source_max_pq)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(10, vdr_dm_data.source_diagonal)
        .map_err(DolbyVisionError::Io)?;

    // Write reshaping curves
    let num_curves = vdr_dm_data.reshaping_curves.len().saturating_sub(1);
    writer
        .write_var(2, num_curves as u8)
        .map_err(DolbyVisionError::Io)?;

    for curve in &vdr_dm_data.reshaping_curves {
        write_reshaping_curve(writer, curve)?;
    }

    // Write NLQ parameters
    let num_nlq_params = vdr_dm_data.nlq_params.len().saturating_sub(1);
    writer
        .write_var(2, num_nlq_params as u8)
        .map_err(DolbyVisionError::Io)?;

    for params in &vdr_dm_data.nlq_params {
        write_nlq_params(writer, params)?;
    }

    Ok(())
}

/// Write color matrix.
fn write_color_matrix<W: std::io::Write>(
    writer: &mut BitWriter<W, BigEndian>,
    matrix: &ColorMatrix,
) -> Result<()> {
    for row in &matrix.matrix {
        for &col in row {
            writer
                .write_signed_var(16, col)
                .map_err(DolbyVisionError::Io)?;
        }
    }
    Ok(())
}

/// Write reshaping curve.
fn write_reshaping_curve<W: std::io::Write>(
    writer: &mut BitWriter<W, BigEndian>,
    curve: &ReshapingCurve,
) -> Result<()> {
    let num_pivots = curve.pivots.len().saturating_sub(1);
    writer
        .write_var(4, num_pivots as u8)
        .map_err(DolbyVisionError::Io)?;

    for &pivot in &curve.pivots {
        writer.write_var(12, pivot).map_err(DolbyVisionError::Io)?;
    }

    for (i, &idc) in curve.mapping_idc.iter().enumerate() {
        writer.write_var(2, idc).map_err(DolbyVisionError::Io)?;

        if idc == 0 && i < curve.poly_order_minus1.len() {
            // Polynomial mapping
            let order = curve.poly_order_minus1[i];
            writer.write_var(2, order).map_err(DolbyVisionError::Io)?;

            if i < curve.poly_coef.len() {
                for &coef in &curve.poly_coef[i] {
                    writer
                        .write_signed_var(16, coef)
                        .map_err(DolbyVisionError::Io)?;
                }
            }
        }
    }

    writer
        .write_var(2, curve.mmr_order_minus1)
        .map_err(DolbyVisionError::Io)?;

    for &coef in &curve.mmr_coef {
        writer
            .write_signed_var(16, coef)
            .map_err(DolbyVisionError::Io)?;
    }

    Ok(())
}

/// Write NLQ parameters.
fn write_nlq_params<W: std::io::Write>(
    writer: &mut BitWriter<W, BigEndian>,
    params: &NlqParams,
) -> Result<()> {
    writer
        .write_var(10, params.nlq_offset)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(27, params.vdr_in_max)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(26, params.linear_deadzone_slope)
        .map_err(DolbyVisionError::Io)?;

    writer
        .write_var(26, params.linear_deadzone_threshold)
        .map_err(DolbyVisionError::Io)?;

    Ok(())
}

/// Write Level 1 metadata.
fn write_level1_metadata<W: std::io::Write>(
    writer: &mut BitWriter<W, BigEndian>,
    metadata: Option<&Level1Metadata>,
) -> Result<()> {
    writer
        .write_bit(metadata.is_some())
        .map_err(DolbyVisionError::Io)?;

    if let Some(meta) = metadata {
        writer
            .write_var(12, meta.min_pq)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(12, meta.max_pq)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(12, meta.avg_pq)
            .map_err(DolbyVisionError::Io)?;
    }

    Ok(())
}

/// Write Level 2 metadata.
fn write_level2_metadata<W: std::io::Write>(
    writer: &mut BitWriter<W, BigEndian>,
    metadata: Option<&Level2Metadata>,
) -> Result<()> {
    writer
        .write_bit(metadata.is_some())
        .map_err(DolbyVisionError::Io)?;

    if let Some(meta) = metadata {
        writer
            .write_var(8, meta.target_display_index)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_signed_var(16, meta.trim_slope)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_signed_var(16, meta.trim_offset)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_signed_var(16, meta.trim_power)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_signed_var(16, meta.trim_chroma_weight)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_signed_var(16, meta.trim_saturation_gain)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_signed_var(16, meta.ms_weight)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(16, meta.target_mid_contrast)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(16, meta.clip_trim)
            .map_err(DolbyVisionError::Io)?;
    }

    Ok(())
}

/// Write Level 5 metadata.
fn write_level5_metadata<W: std::io::Write>(
    writer: &mut BitWriter<W, BigEndian>,
    metadata: Option<&Level5Metadata>,
) -> Result<()> {
    writer
        .write_bit(metadata.is_some())
        .map_err(DolbyVisionError::Io)?;

    if let Some(meta) = metadata {
        writer
            .write_var(16, meta.active_area_left_offset)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(16, meta.active_area_right_offset)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(16, meta.active_area_top_offset)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(16, meta.active_area_bottom_offset)
            .map_err(DolbyVisionError::Io)?;
    }

    Ok(())
}

/// Write Level 6 metadata.
fn write_level6_metadata<W: std::io::Write>(
    writer: &mut BitWriter<W, BigEndian>,
    metadata: Option<&Level6Metadata>,
) -> Result<()> {
    writer
        .write_bit(metadata.is_some())
        .map_err(DolbyVisionError::Io)?;

    if let Some(meta) = metadata {
        writer
            .write_var(16, meta.max_cll)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(16, meta.max_fall)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(32, meta.min_display_mastering_luminance)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(32, meta.max_display_mastering_luminance)
            .map_err(DolbyVisionError::Io)?;

        for primary in &meta.master_display_primaries {
            writer
                .write_var(16, primary[0])
                .map_err(DolbyVisionError::Io)?;
            writer
                .write_var(16, primary[1])
                .map_err(DolbyVisionError::Io)?;
        }

        writer
            .write_var(16, meta.master_display_white_point[0])
            .map_err(DolbyVisionError::Io)?;
        writer
            .write_var(16, meta.master_display_white_point[1])
            .map_err(DolbyVisionError::Io)?;
    }

    Ok(())
}

/// Write Level 8 metadata.
fn write_level8_metadata<W: std::io::Write>(
    writer: &mut BitWriter<W, BigEndian>,
    metadata: Option<&Level8Metadata>,
) -> Result<()> {
    writer
        .write_bit(metadata.is_some())
        .map_err(DolbyVisionError::Io)?;

    if let Some(meta) = metadata {
        writer
            .write_var(8, meta.target_display_index)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(16, meta.target_max_pq)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(16, meta.target_min_pq)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(8, meta.target_primary_index)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(8, meta.target_eotf)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(16, meta.diagonal_size)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(16, meta.peak_luminance)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(16, meta.diffuse_white_luminance)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(16, meta.ambient_luminance)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(16, meta.surround_reflection)
            .map_err(DolbyVisionError::Io)?;
    }

    Ok(())
}

/// Write Level 9 metadata.
fn write_level9_metadata<W: std::io::Write>(
    writer: &mut BitWriter<W, BigEndian>,
    metadata: Option<&Level9Metadata>,
) -> Result<()> {
    writer
        .write_bit(metadata.is_some())
        .map_err(DolbyVisionError::Io)?;

    if let Some(meta) = metadata {
        writer
            .write_var(8, meta.source_primary_index)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(16, meta.source_max_pq)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(16, meta.source_min_pq)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(16, meta.source_diagonal)
            .map_err(DolbyVisionError::Io)?;
    }

    Ok(())
}

/// Write Level 11 metadata.
fn write_level11_metadata<W: std::io::Write>(
    writer: &mut BitWriter<W, BigEndian>,
    metadata: Option<&Level11Metadata>,
) -> Result<()> {
    writer
        .write_bit(metadata.is_some())
        .map_err(DolbyVisionError::Io)?;

    if let Some(meta) = metadata {
        writer
            .write_var(8, meta.content_type as u8)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(8, meta.whitepoint)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_bit(meta.reference_mode_flag)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(8, meta.sharpness)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(8, meta.noise_reduction)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(8, meta.mpeg_noise_reduction)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(8, meta.frame_rate)
            .map_err(DolbyVisionError::Io)?;

        writer
            .write_var(8, meta.temporal_filter_strength)
            .map_err(DolbyVisionError::Io)?;
    }

    Ok(())
}

/// Add emulation prevention bytes to NAL unit data.
///
/// Prevents start code emulation by inserting 0x03 bytes where needed.
fn add_emulation_prevention(data: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(data.len() + data.len() / 100);
    let mut zero_count = 0;

    for &byte in data {
        if zero_count == 2 && byte <= 0x03 {
            // Insert emulation prevention byte
            output.push(0x03);
            zero_count = 0;
        }

        output.push(byte);

        if byte == 0x00 {
            zero_count += 1;
        } else {
            zero_count = 0;
        }
    }

    output
}

/// Calculate CRC32 for RPU data.
#[allow(dead_code)]
fn calculate_crc32(data: &[u8]) -> u32 {
    const CRC32_POLYNOMIAL: u32 = 0xEDB8_8320;

    let mut crc = 0xFFFF_FFFF;

    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if crc & 1 == 1 {
                crc = (crc >> 1) ^ CRC32_POLYNOMIAL;
            } else {
                crc >>= 1;
            }
        }
    }

    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_empty_rpu() {
        let rpu = DolbyVisionRpu::new(Profile::Profile8);
        let result = write_rpu_bitstream(&rpu);
        assert!(result.is_ok());
        let data = result.expect("data should be valid");
        assert!(!data.is_empty());
    }

    #[test]
    fn test_write_nal_unit() {
        let rpu = DolbyVisionRpu::new(Profile::Profile8);
        let result = write_nal_unit(&rpu);
        assert!(result.is_ok());
        let data = result.expect("data should be valid");
        assert!(data.len() >= 2);
    }

    #[test]
    fn test_emulation_prevention() {
        let data = vec![0x00, 0x00, 0x01, 0x00, 0x00, 0x02];
        let result = add_emulation_prevention(&data);
        assert!(result.len() >= data.len());

        // Should insert 0x03 before 0x01 and 0x02
        assert!(result.contains(&0x03));
    }

    #[test]
    fn test_crc32() {
        let data = b"hello world";
        let crc = calculate_crc32(data);
        assert_ne!(crc, 0);
        assert_ne!(crc, 0xFFFF_FFFF);
    }

    #[test]
    fn test_sei_size_encoding() {
        let mut buffer = Vec::new();
        let mut cursor = Cursor::new(&mut buffer);
        let mut writer = BitWriter::endian(&mut cursor, BigEndian);

        write_sei_size(&mut writer, 100).expect("test expectation failed");
        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer[0], 100);

        buffer.clear();
        cursor = Cursor::new(&mut buffer);
        writer = BitWriter::endian(&mut cursor, BigEndian);

        write_sei_size(&mut writer, 300).expect("test expectation failed");
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer[0], 0xFF);
        assert_eq!(buffer[1], 45);
    }

    /// Verify that a round-trip (write → parse) preserves all metadata fields.
    #[test]
    fn test_rpu_roundtrip() {
        use crate::{
            metadata::{Level1Metadata, Level5Metadata, Level6Metadata, Level8Metadata},
            DolbyVisionRpu, Profile,
        };

        // Build an RPU with known Level 1, 5, 6, and 8 metadata.
        let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
        rpu.level1 = Some(Level1Metadata {
            min_pq: 64,
            avg_pq: 512,
            max_pq: 3500,
        });
        rpu.level5 = Some(Level5Metadata {
            active_area_left_offset: 10,
            active_area_right_offset: 20,
            active_area_top_offset: 5,
            active_area_bottom_offset: 15,
        });
        rpu.level6 = Some(Level6Metadata {
            max_cll: 1000,
            max_fall: 400,
            min_display_mastering_luminance: 50,
            max_display_mastering_luminance: 10000,
            master_display_primaries: [[34000, 16000], [13250, 34500], [7500, 3000]],
            master_display_white_point: [15635, 16450],
        });
        rpu.level8 = Some(Level8Metadata::hdr_1000());

        // Serialize to a raw bitstream.
        let bytes = write_rpu_bitstream(&rpu).expect("write_rpu_bitstream should succeed");
        assert!(!bytes.is_empty(), "Serialized bitstream must not be empty");

        // Parse back.
        let parsed =
            crate::parser::parse_rpu_bitstream(&bytes).expect("parse_rpu_bitstream should succeed");

        // ── Level 1 ─────────────────────────────────────────────────────────
        let l1_orig = rpu.level1.as_ref().expect("l1 must be present");
        let l1_parsed = parsed.level1.as_ref().expect("parsed l1 must be present");
        assert_eq!(
            l1_orig.min_pq, l1_parsed.min_pq,
            "min_pq must survive round-trip"
        );
        assert_eq!(
            l1_orig.avg_pq, l1_parsed.avg_pq,
            "avg_pq must survive round-trip"
        );
        assert_eq!(
            l1_orig.max_pq, l1_parsed.max_pq,
            "max_pq must survive round-trip"
        );

        // ── Level 5 ─────────────────────────────────────────────────────────
        let l5_orig = rpu.level5.as_ref().expect("l5 must be present");
        let l5_parsed = parsed.level5.as_ref().expect("parsed l5 must be present");
        assert_eq!(
            l5_orig.active_area_left_offset, l5_parsed.active_area_left_offset,
            "active_area_left_offset must survive round-trip"
        );
        assert_eq!(
            l5_orig.active_area_right_offset, l5_parsed.active_area_right_offset,
            "active_area_right_offset must survive round-trip"
        );
        assert_eq!(
            l5_orig.active_area_top_offset, l5_parsed.active_area_top_offset,
            "active_area_top_offset must survive round-trip"
        );
        assert_eq!(
            l5_orig.active_area_bottom_offset, l5_parsed.active_area_bottom_offset,
            "active_area_bottom_offset must survive round-trip"
        );

        // ── Level 6 ─────────────────────────────────────────────────────────
        let l6_orig = rpu.level6.as_ref().expect("l6 must be present");
        let l6_parsed = parsed.level6.as_ref().expect("parsed l6 must be present");
        assert_eq!(l6_orig.max_cll, l6_parsed.max_cll, "max_cll round-trip");
        assert_eq!(l6_orig.max_fall, l6_parsed.max_fall, "max_fall round-trip");
        assert_eq!(
            l6_orig.min_display_mastering_luminance, l6_parsed.min_display_mastering_luminance,
            "min_display_mastering_luminance round-trip"
        );
        assert_eq!(
            l6_orig.max_display_mastering_luminance, l6_parsed.max_display_mastering_luminance,
            "max_display_mastering_luminance round-trip"
        );
        assert_eq!(
            l6_orig.master_display_primaries, l6_parsed.master_display_primaries,
            "master_display_primaries round-trip"
        );
        assert_eq!(
            l6_orig.master_display_white_point, l6_parsed.master_display_white_point,
            "master_display_white_point round-trip"
        );

        // ── Level 8 ─────────────────────────────────────────────────────────
        let l8_orig = rpu.level8.as_ref().expect("l8 must be present");
        let l8_parsed = parsed.level8.as_ref().expect("parsed l8 must be present");
        assert_eq!(
            l8_orig.target_max_pq, l8_parsed.target_max_pq,
            "target_max_pq round-trip"
        );
        assert_eq!(
            l8_orig.target_min_pq, l8_parsed.target_min_pq,
            "target_min_pq round-trip"
        );
        assert_eq!(
            l8_orig.target_eotf, l8_parsed.target_eotf,
            "target_eotf round-trip"
        );
        assert_eq!(
            l8_orig.peak_luminance, l8_parsed.peak_luminance,
            "peak_luminance round-trip"
        );
    }
}
