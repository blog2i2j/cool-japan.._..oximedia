//! AV1CodecConfigurationRecord (`av1C` box) builder.
//!
//! Builds the 4-byte AV1CodecConfigurationRecord payload required for the
//! `av1C` config box inside an MP4 `av01` sample entry.
//!
//! Reference: [AV1 Codec ISO Media File Format Binding §2.3.3](https://aomediacodec.github.io/av1-isobmff/#av1codecconfigurationbox)

#![forbid(unsafe_code)]

/// Tries to build an `AV1CodecConfigurationRecord` payload from `extradata`.
///
/// `extradata` may be either:
/// - A 4-byte pre-built `av1C` record starting with `0x81`, used verbatim, or
/// - A complete AV1 access unit or OBU stream; the first Sequence Header OBU is
///   located and parsed to derive the record fields.
///
/// Returns `None` if parsing fails; callers should fall back to a safe default.
pub(super) fn build_av1c_from_extradata(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 2 {
        return None;
    }

    // If the first byte already looks like a valid av1C record (marker=1, version=1)
    // and the caller supplied 4+ bytes, trust it verbatim.
    if data.len() >= 4 && data[0] == 0x81 {
        return Some(data[..4].to_vec());
    }

    // Otherwise attempt to parse an OBU sequence and extract the Sequence Header.
    parse_av1c_from_obus(data)
}

/// Walks OBUs in `data` looking for the first Sequence Header OBU (obu_type=1).
/// Extracts the fields needed for `AV1CodecConfigurationRecord` and returns the
/// 4-byte record on success, or `None` if no valid Sequence Header is found.
fn parse_av1c_from_obus(data: &[u8]) -> Option<Vec<u8>> {
    let mut pos = 0usize;

    while pos < data.len() {
        let obu_header = data[pos];
        pos += 1;

        let obu_type = (obu_header >> 3) & 0x0F;
        let has_size_field = (obu_header & 0x02) != 0;
        let extension_flag = (obu_header & 0x04) != 0;

        if extension_flag {
            if pos >= data.len() {
                break;
            }
            pos += 1; // skip extension byte
        }

        let obu_payload_size = if has_size_field {
            let (sz, consumed) = read_leb128(&data[pos..])?;
            pos += consumed;
            sz
        } else {
            data.len() - pos
        };

        let payload_end = pos + obu_payload_size;
        if payload_end > data.len() {
            break;
        }

        // obu_type == 1 is Sequence Header OBU
        if obu_type == 1 && obu_payload_size >= 1 {
            let sh_data = &data[pos..payload_end];
            return extract_av1c_from_seq_header(sh_data);
        }

        pos = payload_end;
    }

    None
}

/// Reads an unsigned LEB128 value from the start of `buf`.
/// Returns `(value, bytes_consumed)` or `None` on truncation.
fn read_leb128(buf: &[u8]) -> Option<(usize, usize)> {
    let mut value: usize = 0;
    let mut shift = 0usize;

    for (i, &byte) in buf.iter().enumerate() {
        value |= ((byte & 0x7F) as usize) << shift;
        shift += 7;
        if (byte & 0x80) == 0 {
            return Some((value, i + 1));
        }
        if shift >= 35 {
            return None; // overflow guard
        }
    }

    None
}

/// Extracts `AV1CodecConfigurationRecord` fields from a raw Sequence Header OBU payload.
///
/// # AV1CodecConfigurationRecord layout (4 bytes)
///
/// ```text
/// Byte 0: marker[1]=1 | version[4]=1          → always 0x81
/// Byte 1: seq_profile[3] | seq_level_idx_0[5]
/// Byte 2: seq_tier_0[1] | high_bitdepth[1] | twelve_bit[1] | mono_chrome[1]
///         | chroma_subsampling_x[1] | chroma_subsampling_y[1] | chroma_sample_pos[2]
/// Byte 3: initial_presentation_delay_present[1]=0 | reserved[4]=0 | 0[3]=0
/// ```
#[allow(clippy::too_many_lines)]
fn extract_av1c_from_seq_header(data: &[u8]) -> Option<Vec<u8>> {
    if data.is_empty() {
        return None;
    }

    let mut bit_pos = 0usize;

    // Inline bit accessors (avoids an external BitReader dependency for 4 bytes of output)
    let read_bit = |bp: usize| -> Option<u8> {
        let byte_idx = bp / 8;
        let bit_idx = 7 - (bp % 8);
        data.get(byte_idx).map(|b| (b >> bit_idx) & 1)
    };

    let read_bits = |start: usize, count: usize| -> Option<u8> {
        let mut val = 0u8;
        for i in 0..count {
            let b = read_bit(start + i)?;
            val = (val << 1) | b;
        }
        Some(val)
    };

    // seq_profile (3 bits)
    let seq_profile = read_bits(bit_pos, 3)?;
    bit_pos += 3;

    // still_picture (1 bit)
    let still_picture = read_bit(bit_pos)?;
    bit_pos += 1;

    // reduced_still_picture_header (1 bit)
    let reduced = read_bit(bit_pos)?;
    bit_pos += 1;

    let (seq_level_idx_0, seq_tier_0);

    if reduced == 1 {
        seq_level_idx_0 = read_bits(bit_pos, 5)?;
        bit_pos += 5;
        seq_tier_0 = 0u8;
    } else {
        // timing_info_present_flag
        let timing_present = read_bit(bit_pos)?;
        bit_pos += 1;
        if timing_present == 1 {
            bit_pos += 64; // skip timing_info fields
            let decoder_model_present = read_bit(bit_pos)?;
            bit_pos += 1;
            if decoder_model_present == 1 {
                bit_pos += 47; // skip decoder model info
            }
        }
        // initial_display_delay_present_flag (skip)
        bit_pos += 1;
        // operating_points_cnt_minus_1 (5 bits)
        let op_count = read_bits(bit_pos, 5)? as usize + 1;
        bit_pos += 5;

        let mut first_level = 0u8;
        let mut first_tier = 0u8;
        for op_i in 0..op_count {
            bit_pos += 12; // operating_point_idc
            let level = read_bits(bit_pos, 5)?;
            bit_pos += 5;
            let tier = if level > 7 {
                let t = read_bit(bit_pos)?;
                bit_pos += 1;
                t
            } else {
                0
            };
            if op_i == 0 {
                first_level = level;
                first_tier = tier;
            }
        }
        seq_level_idx_0 = first_level;
        seq_tier_0 = first_tier;
    }

    // frame_width_bits_minus_1 + frame_height_bits_minus_1 (4+4 bits)
    let fw_bits = read_bits(bit_pos, 4)? as usize + 1;
    bit_pos += 4;
    let fh_bits = read_bits(bit_pos, 4)? as usize + 1;
    bit_pos += 4;
    bit_pos += fw_bits + fh_bits; // skip actual frame dimensions

    if reduced == 0 {
        let fid = read_bit(bit_pos)?;
        bit_pos += 1;
        if fid == 1 {
            bit_pos += 7; // delta_frame_id_length_minus_2 + additional_frame_id_length_minus_1
        }
        bit_pos += 7; // various tool flags
        let order_hint = read_bit(bit_pos)?;
        bit_pos += 1;
        if order_hint == 1 {
            bit_pos += 2; // enable_jnt_comp + enable_ref_frame_mvs
        }
        // seq_choose_screen_content_tools
        let sc = read_bit(bit_pos)?;
        bit_pos += 1;
        if sc == 0 {
            bit_pos += 1;
        }
        // seq_choose_integer_mv
        let siv = read_bit(bit_pos)?;
        bit_pos += 1;
        if siv == 0 {
            bit_pos += 1;
        }
        if order_hint == 1 {
            bit_pos += 3; // order_hint_bits_minus_1
        }
        bit_pos += 3; // enable_superres + enable_cdef + enable_restoration
    }

    // ── color_config ─────────────────────────────────────────────────────────
    let high_bitdepth = read_bit(bit_pos)?;
    bit_pos += 1;

    let twelve_bit = if seq_profile == 2 && high_bitdepth == 1 {
        let tb = read_bit(bit_pos)?;
        bit_pos += 1;
        tb
    } else {
        0
    };

    let monochrome = if seq_profile == 1 || (still_picture == 1 && reduced == 0) {
        0u8
    } else {
        let m = read_bit(bit_pos)?;
        bit_pos += 1;
        m
    };

    let color_description_present = read_bit(bit_pos)?;
    bit_pos += 1;
    if color_description_present == 1 {
        bit_pos += 24; // color_primaries + transfer_characteristics + matrix_coefficients
    }

    let (subsampling_x, subsampling_y, chroma_sample_pos);

    if monochrome == 1 {
        bit_pos += 1; // color_range
        subsampling_x = 1u8;
        subsampling_y = 1u8;
        chroma_sample_pos = 0u8;
    } else if seq_profile == 0 {
        subsampling_x = 1u8;
        subsampling_y = 1u8;
        bit_pos += 1; // color_range
        chroma_sample_pos = {
            let csp = read_bits(bit_pos, 2).unwrap_or(0);
            bit_pos += 2;
            csp
        };
    } else if seq_profile == 1 {
        subsampling_x = 0u8;
        subsampling_y = 0u8;
        bit_pos += 1; // color_range
        chroma_sample_pos = 0;
    } else {
        bit_pos += 1; // color_range
        let bit_depth_val: u8 = if seq_profile == 2 && twelve_bit == 1 {
            12
        } else if high_bitdepth == 1 {
            10
        } else {
            8
        };
        if bit_depth_val == 12 {
            subsampling_x = read_bit(bit_pos).unwrap_or(1);
            bit_pos += 1;
            subsampling_y = if subsampling_x == 1 {
                let sy = read_bit(bit_pos).unwrap_or(1);
                bit_pos += 1;
                sy
            } else {
                0
            };
        } else {
            subsampling_x = 1u8;
            subsampling_y = 0u8;
        }
        chroma_sample_pos = if subsampling_x == 1 && subsampling_y == 1 {
            let csp = read_bits(bit_pos, 2).unwrap_or(0);
            bit_pos += 2;
            csp
        } else {
            0
        };
    }

    let _ = bit_pos; // silence unused-variable warning after final use

    // ── Assemble the 4-byte record ────────────────────────────────────────────
    let byte0: u8 = 0x81; // marker=1, version=1
    let byte1: u8 = ((seq_profile & 0x07) << 5) | (seq_level_idx_0 & 0x1F);
    let byte2: u8 = (seq_tier_0 << 7)
        | (high_bitdepth << 6)
        | (twelve_bit << 5)
        | (monochrome << 4)
        | (subsampling_x << 3)
        | (subsampling_y << 2)
        | (chroma_sample_pos & 0x03);
    let byte3: u8 = 0x00; // initial_presentation_delay_present=0, reserved

    Some(vec![byte0, byte1, byte2, byte3])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prebuilt_av1c_passthrough() {
        let input = vec![0x81u8, 0x08, 0x0C, 0x00];
        let result = build_av1c_from_extradata(&input).expect("should succeed");
        assert_eq!(result[0], 0x81);
        assert_eq!(result, input);
    }

    #[test]
    fn test_empty_returns_none() {
        assert!(build_av1c_from_extradata(&[]).is_none());
        assert!(build_av1c_from_extradata(&[0x00]).is_none());
    }

    #[test]
    fn test_short_data_returns_none() {
        assert!(build_av1c_from_extradata(&[0x00, 0x00]).is_none());
    }

    #[test]
    fn test_leb128_single_byte() {
        assert_eq!(read_leb128(&[0x05]), Some((5, 1)));
        assert_eq!(read_leb128(&[0x00]), Some((0, 1)));
        assert_eq!(read_leb128(&[0x7F]), Some((127, 1)));
    }

    #[test]
    fn test_leb128_multi_byte() {
        // 128 = 0x80, 0x01
        assert_eq!(read_leb128(&[0x80, 0x01]), Some((128, 2)));
        // 300 = 0xAC, 0x02
        assert_eq!(read_leb128(&[0xAC, 0x02]), Some((300, 2)));
    }

    #[test]
    fn test_leb128_empty_returns_none() {
        assert!(read_leb128(&[]).is_none());
    }
}
