//! ICC profile parsing and representation.
//!
//! This module provides tools for parsing and representing ICC color profiles.

use crate::error::{CalibrationError, CalibrationResult};
use crate::{Illuminant, Matrix3x3};
use serde::{Deserialize, Serialize};

/// ICC profile version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IccProfileVersion {
    /// ICC v2.
    V2,
    /// ICC v4.
    V4,
}

/// ICC color profile.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IccProfile {
    /// Profile description.
    pub description: String,
    /// Profile version.
    pub version: IccProfileVersion,
    /// Color space to XYZ transformation matrix.
    pub to_xyz_matrix: Matrix3x3,
    /// XYZ to color space transformation matrix.
    pub from_xyz_matrix: Matrix3x3,
    /// Profile white point.
    pub white_point: Illuminant,
    /// Profile creation date (Unix timestamp).
    pub creation_date: u64,
}

// ── ICC binary constants ───────────────────────────────────────────────────────

const HEADER_SIZE: usize = 128;
const TAG_ENTRY_SIZE: usize = 12;

// Four-character-code helpers as u32 big-endian
const SIG_ACSP: u32 = 0x6163_7370; // 'acsp'
const SIG_MNTR: u32 = 0x6D6E_7472; // 'mntr'
const SIG_RGB_: u32 = 0x5247_4220; // 'RGB '
const SIG_XYZ_: u32 = 0x5859_5A20; // 'XYZ '
const SIG_XYZ_TYPE: u32 = 0x5859_5A20; // XYZType signature
const SIG_DESC: u32 = 0x6465_7363; // 'desc'
const SIG_MLUC: u32 = 0x6D6C_7563; // 'mluc'
const SIG_WTPT: u32 = 0x7774_7074; // 'wtpt'
const SIG_RXYZ: u32 = 0x7258_595A; // 'rXYZ'
const SIG_GXYZ: u32 = 0x6758_595A; // 'gXYZ'
const SIG_BXYZ: u32 = 0x6258_595A; // 'bXYZ'

const VERSION_V2: u32 = 0x0210_0000;
const VERSION_V4: u32 = 0x0400_0000;

/// D50 PCS illuminant XYZ values written in ICC headers (fixed canonical values).
const D50_PCS_X: f64 = 0.964_2;
const D50_PCS_Y: f64 = 1.0;
const D50_PCS_Z: f64 = 0.824_9;

// ── s15Fixed16 helpers ─────────────────────────────────────────────────────────

#[allow(clippy::cast_possible_wrap)]
fn f64_to_s15f16(v: f64) -> i32 {
    (v * 65536.0).round() as i32
}

fn s15f16_to_f64(raw: i32) -> f64 {
    f64::from(raw) / 65536.0
}

// ── u32 / u16 big-endian read helpers ─────────────────────────────────────────

fn read_u32_be(data: &[u8], offset: usize) -> CalibrationResult<u32> {
    if offset + 4 > data.len() {
        return Err(CalibrationError::IccParseError(format!(
            "offset {offset} out of bounds (len={})",
            data.len()
        )));
    }
    Ok(u32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

fn read_i32_be(data: &[u8], offset: usize) -> CalibrationResult<i32> {
    if offset + 4 > data.len() {
        return Err(CalibrationError::IccParseError(format!(
            "offset {offset} out of bounds (len={})",
            data.len()
        )));
    }
    Ok(i32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

fn read_u16_be(data: &[u8], offset: usize) -> CalibrationResult<u16> {
    if offset + 2 > data.len() {
        return Err(CalibrationError::IccParseError(format!(
            "offset {offset} out of bounds (len={})",
            data.len()
        )));
    }
    Ok(u16::from_be_bytes([data[offset], data[offset + 1]]))
}

// ── Write helpers ──────────────────────────────────────────────────────────────

fn push_u32_be(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_be_bytes());
}

fn push_i32_be(buf: &mut Vec<u8>, v: i32) {
    buf.extend_from_slice(&v.to_be_bytes());
}

fn push_u16_be(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_be_bytes());
}

fn push_u64_zero(buf: &mut Vec<u8>) {
    buf.extend_from_slice(&[0u8; 8]);
}

// ── Creation-date helpers ──────────────────────────────────────────────────────

/// Parse a 6×u16 (year, month, day, hour, min, sec) date-time block (big-endian)
/// and return an approximation of the Unix timestamp.
///
/// Uses the proleptic Gregorian calendar (days since 1970-01-01).
fn parse_creation_date(data: &[u8], offset: usize) -> CalibrationResult<u64> {
    let year = read_u16_be(data, offset)? as u64;
    let month = read_u16_be(data, offset + 2)? as u64;
    let day = read_u16_be(data, offset + 4)? as u64;
    let hour = read_u16_be(data, offset + 6)? as u64;
    let min = read_u16_be(data, offset + 8)? as u64;
    let sec = read_u16_be(data, offset + 10)? as u64;

    // If all fields are zero, return 0 (unknown / not set).
    if year == 0 && month == 0 && day == 0 {
        return Ok(0);
    }

    // Approximate days since Unix epoch (1970-01-01) using the standard formula.
    // Month and day are 1-based in ICC, guard against zero-month.
    let m = month.max(1);
    let d = day.max(1);

    // Days in each year, leap years (simplified):
    // days_since_epoch ≈ (year-1970)*365 + leap_days + month_days + day
    let y = year;
    let leap_days = if y >= 1970 {
        let span = y - 1970;
        span / 4 - span / 100 + span / 400
    } else {
        0
    };

    // Cumulative days at start of each month (non-leap; we correct for leap below).
    const MONTH_DAYS: [u64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let m_idx = (m.saturating_sub(1) as usize).min(11);
    let month_day_offset = MONTH_DAYS[m_idx];

    // Add 1 if month > Feb and current year is a leap year.
    let is_leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
    let leap_adj: u64 = if m > 2 && is_leap { 1 } else { 0 };

    let days_since_epoch =
        (y.saturating_sub(1970)) * 365 + leap_days + month_day_offset + leap_adj + d - 1;

    Ok(days_since_epoch * 86400 + hour * 3600 + min * 60 + sec)
}

/// Write a Unix timestamp as 6×u16 (year, month, day, hour, min, sec) into `buf`.
fn push_creation_date(buf: &mut Vec<u8>, ts: u64) {
    if ts == 0 {
        for _ in 0..6 {
            push_u16_be(buf, 0);
        }
        return;
    }

    let secs_in_day = ts % 86400;
    let mut days = ts / 86400;

    let hour = secs_in_day / 3600;
    let min = (secs_in_day % 3600) / 60;
    let sec = secs_in_day % 60;

    // Compute year from days since 1970.
    let mut year = 1970u64;
    loop {
        let days_in_year: u64 = if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
            366
        } else {
            365
        };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }

    let is_leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let days_per_month: [u64; 12] = [
        31,
        if is_leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 1u64;
    for dm in &days_per_month {
        if days < *dm {
            break;
        }
        days -= dm;
        month += 1;
    }
    let day = days + 1; // 1-based

    push_u16_be(buf, year as u16);
    push_u16_be(buf, month as u16);
    push_u16_be(buf, day as u16);
    push_u16_be(buf, hour as u16);
    push_u16_be(buf, min as u16);
    push_u16_be(buf, sec as u16);
}

// ── XYZNumber tag helpers ──────────────────────────────────────────────────────

/// Parse an XYZType tag blob starting at `tag_offset` within `data`.
/// Layout: 4B signature + 4B reserved + 3 × s15Fixed16.
fn parse_xyz_tag(data: &[u8], tag_offset: usize) -> CalibrationResult<[f64; 3]> {
    // Check we have at least 20 bytes (8-byte header + 12 bytes for 3 × s15F16).
    if tag_offset + 20 > data.len() {
        return Err(CalibrationError::IccParseError(format!(
            "XYZType tag at offset {tag_offset}: insufficient data"
        )));
    }
    let sig = read_u32_be(data, tag_offset)?;
    if sig != SIG_XYZ_TYPE {
        return Err(CalibrationError::IccParseError(format!(
            "XYZType tag: unexpected signature {sig:#010x}"
        )));
    }
    let x = s15f16_to_f64(read_i32_be(data, tag_offset + 8)?);
    let y = s15f16_to_f64(read_i32_be(data, tag_offset + 12)?);
    let z = s15f16_to_f64(read_i32_be(data, tag_offset + 16)?);
    Ok([x, y, z])
}

/// Serialize an XYZType tag blob (20 bytes).
fn xyz_tag_bytes(xyz: &[f64; 3]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(20);
    push_u32_be(&mut buf, SIG_XYZ_TYPE); // type signature 'XYZ '
    push_u32_be(&mut buf, 0); // reserved
    push_i32_be(&mut buf, f64_to_s15f16(xyz[0]));
    push_i32_be(&mut buf, f64_to_s15f16(xyz[1]));
    push_i32_be(&mut buf, f64_to_s15f16(xyz[2]));
    buf
}

// ── Description tag helpers ────────────────────────────────────────────────────

/// Parse a `desc` or `mluc` tag blob and extract an ASCII/Latin-1 string.
fn parse_desc_tag(data: &[u8], tag_offset: usize, tag_size: usize) -> CalibrationResult<String> {
    if tag_offset + 4 > data.len() {
        return Err(CalibrationError::IccParseError(
            "desc tag: too short to read signature".to_string(),
        ));
    }
    let sig = read_u32_be(data, tag_offset)?;

    match sig {
        SIG_DESC => parse_desc_v2(data, tag_offset, tag_size),
        SIG_MLUC => parse_mluc_tag(data, tag_offset, tag_size),
        _ => Err(CalibrationError::IccParseError(format!(
            "desc/mluc tag: unexpected signature {sig:#010x}"
        ))),
    }
}

/// Parse an ICC v2 `desc` tag.
/// Layout: 4B sig + 4B reserved + 4B ascii_length + ascii_bytes (null-terminated).
fn parse_desc_v2(data: &[u8], tag_offset: usize, _tag_size: usize) -> CalibrationResult<String> {
    if tag_offset + 12 > data.len() {
        return Err(CalibrationError::IccParseError(
            "desc tag (v2): too short for header".to_string(),
        ));
    }
    let ascii_len = read_u32_be(data, tag_offset + 8)? as usize;
    let start = tag_offset + 12;
    if ascii_len == 0 {
        return Ok(String::new());
    }
    let end = start
        .checked_add(ascii_len)
        .ok_or_else(|| CalibrationError::IccParseError("desc tag: length overflow".to_string()))?;
    if end > data.len() {
        return Err(CalibrationError::IccParseError(format!(
            "desc tag (v2): ascii_len {ascii_len} exceeds data length"
        )));
    }
    let raw = &data[start..end];
    // Trim null terminator if present.
    let raw = raw.split(|&b| b == 0).next().unwrap_or(raw);
    Ok(String::from_utf8_lossy(raw).into_owned())
}

/// Parse an ICC v4 `mluc` (MultiLocalizedUnicode) tag — extract first record as UTF-16BE.
fn parse_mluc_tag(data: &[u8], tag_offset: usize, tag_size: usize) -> CalibrationResult<String> {
    // mluc layout:
    //   4B sig + 4B reserved + 4B record_count + 4B record_size (usually 12)
    //   Then records: 2B lang + 2B country + 4B length_in_bytes + 4B offset_from_tag_start
    if tag_offset + 16 > data.len() {
        return Err(CalibrationError::IccParseError(
            "mluc tag: too short for header".to_string(),
        ));
    }
    let record_count = read_u32_be(data, tag_offset + 8)? as usize;
    if record_count == 0 {
        return Ok(String::new());
    }
    let record_size = read_u32_be(data, tag_offset + 12)? as usize;
    if record_size < 12 {
        return Err(CalibrationError::IccParseError(
            "mluc tag: record_size too small".to_string(),
        ));
    }
    // First record starts at tag_offset + 16.
    let rec_start = tag_offset + 16;
    if rec_start + record_size > data.len() {
        return Err(CalibrationError::IccParseError(
            "mluc tag: first record out of bounds".to_string(),
        ));
    }
    let str_len_bytes = read_u32_be(data, rec_start + 4)? as usize; // length in bytes (UTF-16BE)
    let str_offset_from_tag = read_u32_be(data, rec_start + 8)? as usize;
    let abs_str_start = tag_offset + str_offset_from_tag;
    let abs_str_end = abs_str_start.checked_add(str_len_bytes).ok_or_else(|| {
        CalibrationError::IccParseError("mluc: string length overflow".to_string())
    })?;
    let _ = tag_size; // used for doc context
    if abs_str_end > data.len() {
        return Err(CalibrationError::IccParseError(
            "mluc tag: string data out of bounds".to_string(),
        ));
    }
    // Decode UTF-16BE.
    let utf16_bytes = &data[abs_str_start..abs_str_end];
    let code_units: Vec<u16> = utf16_bytes
        .chunks_exact(2)
        .map(|ch| u16::from_be_bytes([ch[0], ch[1]]))
        .collect();
    Ok(String::from_utf16_lossy(&code_units))
}

/// Serialize a `desc` tag blob (ICC v2 format).
/// Layout: 4B sig + 4B reserved (0) + 4B ascii_length + ascii_bytes + null + padding.
fn desc_tag_bytes(desc: &str) -> Vec<u8> {
    let ascii: Vec<u8> = desc.bytes().collect();
    let ascii_len = ascii.len() + 1; // include null terminator in length field
    let mut buf = Vec::new();
    push_u32_be(&mut buf, SIG_DESC); // 'desc'
    push_u32_be(&mut buf, 0); // reserved
    push_u32_be(&mut buf, ascii_len as u32); // length including null
    buf.extend_from_slice(&ascii);
    buf.push(0u8); // null terminator
                   // Pad to 4-byte alignment.
    while buf.len() % 4 != 0 {
        buf.push(0u8);
    }
    buf
}

// ── Illuminant detection ───────────────────────────────────────────────────────

/// Map an XYZ triplet to the closest `Illuminant`, or `D50` if nothing is within ±0.01.
fn xyz_to_illuminant(xyz: &[f64; 3]) -> Illuminant {
    let candidates = [
        Illuminant::A,
        Illuminant::D50,
        Illuminant::D55,
        Illuminant::D65,
        Illuminant::D75,
        Illuminant::E,
        Illuminant::F2,
        Illuminant::F7,
        Illuminant::F11,
    ];
    for &ill in &candidates {
        let ref_xyz = ill.xyz();
        // Normalize to Y=1 before comparing (profiles store Y=1 XYZ).
        let y = ref_xyz[1]; // always 1.0 by definition
        if (xyz[0] - ref_xyz[0] / y).abs() < 0.01
            && (xyz[1] - ref_xyz[1] / y).abs() < 0.01
            && (xyz[2] - ref_xyz[2] / y).abs() < 0.01
        {
            return ill;
        }
    }
    Illuminant::D50 // fallback
}

// ── IccProfile implementation ──────────────────────────────────────────────────

impl IccProfile {
    /// Create a new ICC profile.
    #[must_use]
    pub fn new(description: String, to_xyz_matrix: Matrix3x3, white_point: Illuminant) -> Self {
        let from_xyz_matrix = Self::compute_inverse_matrix(&to_xyz_matrix);

        Self {
            description,
            version: IccProfileVersion::V4,
            to_xyz_matrix,
            from_xyz_matrix,
            white_point,
            creation_date: 0, // Placeholder
        }
    }

    /// Parse an ICC profile from bytes.
    ///
    /// # Arguments
    ///
    /// * `data` - ICC profile data
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails.
    pub fn from_bytes(data: &[u8]) -> CalibrationResult<Self> {
        // ── Step 1: validate minimum header size and 'acsp' signature ──────────
        if data.len() < HEADER_SIZE {
            return Err(CalibrationError::IccParseError(format!(
                "data too short: {} bytes (minimum {})",
                data.len(),
                HEADER_SIZE
            )));
        }
        let acsp = read_u32_be(data, 36)?;
        if acsp != SIG_ACSP {
            return Err(CalibrationError::IccParseError(format!(
                "not an ICC profile: 'acsp' signature missing (got {acsp:#010x})"
            )));
        }

        // ── Step 2: parse version ──────────────────────────────────────────────
        let ver_raw = read_u32_be(data, 8)?;
        let version = match ver_raw >> 24 {
            2 => IccProfileVersion::V2,
            4 => IccProfileVersion::V4,
            _ => {
                // Accept any version but default to V4.
                IccProfileVersion::V4
            }
        };

        // ── Step 3: parse creation date (offset 24, 12 bytes = 6×u16) ─────────
        let creation_date = parse_creation_date(data, 24)?;

        // ── Step 4: parse tag table ────────────────────────────────────────────
        // Tag table starts immediately after the 128-byte header.
        let tag_count = read_u32_be(data, HEADER_SIZE)? as usize;

        // Collect (signature, offset, size) entries.
        let mut desc_pos: Option<(usize, usize)> = None;
        let mut wtpt_pos: Option<(usize, usize)> = None;
        let mut rxyz_pos: Option<(usize, usize)> = None;
        let mut gxyz_pos: Option<(usize, usize)> = None;
        let mut bxyz_pos: Option<(usize, usize)> = None;

        let table_start = HEADER_SIZE + 4;
        for i in 0..tag_count {
            let entry_offset = table_start + i * TAG_ENTRY_SIZE;
            if entry_offset + TAG_ENTRY_SIZE > data.len() {
                break;
            }
            let sig = read_u32_be(data, entry_offset)?;
            let tag_off = read_u32_be(data, entry_offset + 4)? as usize;
            let tag_size = read_u32_be(data, entry_offset + 8)? as usize;

            match sig {
                SIG_DESC => desc_pos = Some((tag_off, tag_size)),
                SIG_WTPT => wtpt_pos = Some((tag_off, tag_size)),
                SIG_RXYZ => rxyz_pos = Some((tag_off, tag_size)),
                SIG_GXYZ => gxyz_pos = Some((tag_off, tag_size)),
                SIG_BXYZ => bxyz_pos = Some((tag_off, tag_size)),
                _ => {} // ignore unknown tags
            }
        }

        // ── Step 5: parse required tags ────────────────────────────────────────
        let description = if let Some((off, sz)) = desc_pos {
            parse_desc_tag(data, off, sz)?
        } else {
            String::new()
        };

        let white_point = if let Some((off, _)) = wtpt_pos {
            let xyz = parse_xyz_tag(data, off)?;
            xyz_to_illuminant(&xyz)
        } else {
            Illuminant::D50
        };

        // Parse primary XYZ tags — build to_xyz_matrix.
        let r_xyz = rxyz_pos
            .map(|(off, _)| parse_xyz_tag(data, off))
            .transpose()?
            .unwrap_or([1.0, 0.0, 0.0]);
        let g_xyz = gxyz_pos
            .map(|(off, _)| parse_xyz_tag(data, off))
            .transpose()?
            .unwrap_or([0.0, 1.0, 0.0]);
        let b_xyz = bxyz_pos
            .map(|(off, _)| parse_xyz_tag(data, off))
            .transpose()?
            .unwrap_or([0.0, 0.0, 1.0]);

        // Columns of to_xyz_matrix are rXYZ, gXYZ, bXYZ.
        let to_xyz_matrix: Matrix3x3 = [
            [r_xyz[0], g_xyz[0], b_xyz[0]], // row 0: X from each primary
            [r_xyz[1], g_xyz[1], b_xyz[1]], // row 1: Y from each primary
            [r_xyz[2], g_xyz[2], b_xyz[2]], // row 2: Z from each primary
        ];
        let from_xyz_matrix = Self::compute_inverse_matrix(&to_xyz_matrix);

        Ok(Self {
            description,
            version,
            to_xyz_matrix,
            from_xyz_matrix,
            white_point,
            creation_date,
        })
    }

    /// Serialize the ICC profile to bytes.
    ///
    /// Produces a well-formed ICC binary that can be round-tripped through
    /// [`IccProfile::from_bytes`].
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_bytes(&self) -> CalibrationResult<Vec<u8>> {
        // ── Step 1: build tag data blobs ───────────────────────────────────────

        // desc tag
        let desc_blob = desc_tag_bytes(&self.description);

        // wtpt tag — white point XYZ (Y-normalized)
        let wp_xyz = self.white_point.xyz();
        let wtpt_blob = xyz_tag_bytes(&wp_xyz);

        // Primary color tags — columns of to_xyz_matrix → individual XYZ triplets.
        let r_xyz = [
            self.to_xyz_matrix[0][0],
            self.to_xyz_matrix[1][0],
            self.to_xyz_matrix[2][0],
        ];
        let g_xyz = [
            self.to_xyz_matrix[0][1],
            self.to_xyz_matrix[1][1],
            self.to_xyz_matrix[2][1],
        ];
        let b_xyz = [
            self.to_xyz_matrix[0][2],
            self.to_xyz_matrix[1][2],
            self.to_xyz_matrix[2][2],
        ];
        let rxyz_blob = xyz_tag_bytes(&r_xyz);
        let gxyz_blob = xyz_tag_bytes(&g_xyz);
        let bxyz_blob = xyz_tag_bytes(&b_xyz);

        // ── Step 2: calculate layout ───────────────────────────────────────────
        // 5 tags in the tag table.
        const TAG_COUNT: u32 = 5;
        let tag_table_size = 4 + (TAG_COUNT as usize) * TAG_ENTRY_SIZE; // 4 + 5*12 = 64
        let data_start = HEADER_SIZE + tag_table_size; // 128 + 64 = 192

        let desc_off = data_start;
        let wtpt_off = desc_off + desc_blob.len();
        let rxyz_off = wtpt_off + wtpt_blob.len();
        let gxyz_off = rxyz_off + rxyz_blob.len();
        let bxyz_off = gxyz_off + gxyz_blob.len();
        let total_size = bxyz_off + bxyz_blob.len();

        // ── Step 3: write header (128 bytes) ──────────────────────────────────
        let mut buf: Vec<u8> = Vec::with_capacity(total_size);

        // 0-3: total profile size
        push_u32_be(&mut buf, total_size as u32);
        // 4-7: CMM type (0 = unspecified)
        push_u32_be(&mut buf, 0);
        // 8-11: version
        let ver_u32 = match self.version {
            IccProfileVersion::V2 => VERSION_V2,
            IccProfileVersion::V4 => VERSION_V4,
        };
        push_u32_be(&mut buf, ver_u32);
        // 12-15: device class 'mntr'
        push_u32_be(&mut buf, SIG_MNTR);
        // 16-19: color space 'RGB '
        push_u32_be(&mut buf, SIG_RGB_);
        // 20-23: PCS 'XYZ '
        push_u32_be(&mut buf, SIG_XYZ_);
        // 24-35: creation date/time (6 × u16 BE)
        push_creation_date(&mut buf, self.creation_date);
        // 36-39: 'acsp'
        push_u32_be(&mut buf, SIG_ACSP);
        // 40-43: platform signature (0)
        push_u32_be(&mut buf, 0);
        // 44-47: flags (0)
        push_u32_be(&mut buf, 0);
        // 48-51: device manufacturer (0)
        push_u32_be(&mut buf, 0);
        // 52-55: device model (0)
        push_u32_be(&mut buf, 0);
        // 56-63: device attributes (0, 8 bytes)
        push_u64_zero(&mut buf);
        // 64-67: rendering intent (0 = perceptual)
        push_u32_be(&mut buf, 0);
        // 68-79: PCS illuminant XYZ in s15Fixed16 (D50 canonical)
        push_i32_be(&mut buf, f64_to_s15f16(D50_PCS_X));
        push_i32_be(&mut buf, f64_to_s15f16(D50_PCS_Y));
        push_i32_be(&mut buf, f64_to_s15f16(D50_PCS_Z));
        // 80-83: profile creator (0)
        push_u32_be(&mut buf, 0);
        // 84-99: profile ID / MD5 (16 zero bytes)
        buf.extend_from_slice(&[0u8; 16]);
        // 100-127: reserved (28 zero bytes)
        buf.extend_from_slice(&[0u8; 28]);

        debug_assert_eq!(buf.len(), HEADER_SIZE, "header must be exactly 128 bytes");

        // ── Step 4: write tag table ────────────────────────────────────────────
        push_u32_be(&mut buf, TAG_COUNT);

        // desc
        push_u32_be(&mut buf, SIG_DESC);
        push_u32_be(&mut buf, desc_off as u32);
        push_u32_be(&mut buf, desc_blob.len() as u32);

        // wtpt
        push_u32_be(&mut buf, SIG_WTPT);
        push_u32_be(&mut buf, wtpt_off as u32);
        push_u32_be(&mut buf, wtpt_blob.len() as u32);

        // rXYZ
        push_u32_be(&mut buf, SIG_RXYZ);
        push_u32_be(&mut buf, rxyz_off as u32);
        push_u32_be(&mut buf, rxyz_blob.len() as u32);

        // gXYZ
        push_u32_be(&mut buf, SIG_GXYZ);
        push_u32_be(&mut buf, gxyz_off as u32);
        push_u32_be(&mut buf, gxyz_blob.len() as u32);

        // bXYZ
        push_u32_be(&mut buf, SIG_BXYZ);
        push_u32_be(&mut buf, bxyz_off as u32);
        push_u32_be(&mut buf, bxyz_blob.len() as u32);

        debug_assert_eq!(
            buf.len(),
            data_start,
            "tag table must end at data_start offset"
        );

        // ── Step 5: write tag data blobs ───────────────────────────────────────
        buf.extend_from_slice(&desc_blob);
        buf.extend_from_slice(&wtpt_blob);
        buf.extend_from_slice(&rxyz_blob);
        buf.extend_from_slice(&gxyz_blob);
        buf.extend_from_slice(&bxyz_blob);

        debug_assert_eq!(
            buf.len(),
            total_size,
            "final size must match calculated total"
        );

        Ok(buf)
    }

    /// Convert an RGB color to XYZ using this profile.
    #[must_use]
    pub fn rgb_to_xyz(&self, rgb: &[f64; 3]) -> [f64; 3] {
        self.apply_matrix(&self.to_xyz_matrix, rgb)
    }

    /// Convert an XYZ color to RGB using this profile.
    #[must_use]
    pub fn xyz_to_rgb(&self, xyz: &[f64; 3]) -> [f64; 3] {
        self.apply_matrix(&self.from_xyz_matrix, xyz)
    }

    /// Apply a 3x3 matrix to a color.
    fn apply_matrix(&self, matrix: &Matrix3x3, color: &[f64; 3]) -> [f64; 3] {
        [
            matrix[0][0] * color[0] + matrix[0][1] * color[1] + matrix[0][2] * color[2],
            matrix[1][0] * color[0] + matrix[1][1] * color[1] + matrix[1][2] * color[2],
            matrix[2][0] * color[0] + matrix[2][1] * color[1] + matrix[2][2] * color[2],
        ]
    }

    /// Compute the inverse of a 3x3 matrix.
    fn compute_inverse_matrix(matrix: &Matrix3x3) -> Matrix3x3 {
        // Compute determinant
        let det = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);

        if det.abs() < 1e-10 {
            // Matrix is singular, return identity
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        }

        let inv_det = 1.0 / det;

        // Compute inverse using adjugate method
        [
            [
                (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inv_det,
                (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inv_det,
                (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inv_det,
            ],
            [
                (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inv_det,
                (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inv_det,
                (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inv_det,
            ],
            [
                (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inv_det,
                (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inv_det,
                (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inv_det,
            ],
        ]
    }

    /// Validate the ICC profile.
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails.
    pub fn validate(&self) -> CalibrationResult<()> {
        // Check that matrices are not all zeros
        let to_xyz_sum: f64 = self.to_xyz_matrix.iter().flatten().sum();
        let from_xyz_sum: f64 = self.from_xyz_matrix.iter().flatten().sum();

        if to_xyz_sum.abs() < 1e-10 {
            return Err(CalibrationError::IccInvalidProfile(
                "to_xyz_matrix is zero".to_string(),
            ));
        }

        if from_xyz_sum.abs() < 1e-10 {
            return Err(CalibrationError::IccInvalidProfile(
                "from_xyz_matrix is zero".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Original tests (preserved) ─────────────────────────────────────────────

    #[test]
    fn test_icc_profile_new() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let profile = IccProfile::new("Test Profile".to_string(), identity, Illuminant::D65);

        assert_eq!(profile.description, "Test Profile");
        assert_eq!(profile.version, IccProfileVersion::V4);
        assert_eq!(profile.white_point, Illuminant::D65);
    }

    #[test]
    fn test_icc_profile_rgb_to_xyz() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let profile = IccProfile::new("Test Profile".to_string(), identity, Illuminant::D65);

        let rgb = [0.5, 0.6, 0.7];
        let xyz = profile.rgb_to_xyz(&rgb);

        assert!((xyz[0] - 0.5).abs() < 1e-10);
        assert!((xyz[1] - 0.6).abs() < 1e-10);
        assert!((xyz[2] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_icc_profile_xyz_to_rgb() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let profile = IccProfile::new("Test Profile".to_string(), identity, Illuminant::D65);

        let xyz = [0.5, 0.6, 0.7];
        let rgb = profile.xyz_to_rgb(&xyz);

        assert!((rgb[0] - 0.5).abs() < 1e-10);
        assert!((rgb[1] - 0.6).abs() < 1e-10);
        assert!((rgb[2] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_icc_profile_roundtrip() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let profile = IccProfile::new("Test Profile".to_string(), identity, Illuminant::D65);

        let rgb = [0.3, 0.5, 0.7];
        let xyz = profile.rgb_to_xyz(&rgb);
        let rgb2 = profile.xyz_to_rgb(&xyz);

        assert!((rgb2[0] - rgb[0]).abs() < 1e-10);
        assert!((rgb2[1] - rgb[1]).abs() < 1e-10);
        assert!((rgb2[2] - rgb[2]).abs() < 1e-10);
    }

    #[test]
    fn test_icc_profile_validate() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let profile = IccProfile::new("Test Profile".to_string(), identity, Illuminant::D65);

        assert!(profile.validate().is_ok());
    }

    #[test]
    fn test_icc_profile_validate_invalid() {
        let zero_matrix = [[0.0; 3]; 3];

        let profile = IccProfile::new("Test Profile".to_string(), zero_matrix, Illuminant::D65);

        assert!(profile.validate().is_err());
    }

    #[test]
    fn test_compute_inverse_matrix() {
        let matrix = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];

        let inverse = IccProfile::compute_inverse_matrix(&matrix);

        // Inverse of 2I should be 0.5I
        assert!((inverse[0][0] - 0.5).abs() < 1e-10);
        assert!((inverse[1][1] - 0.5).abs() < 1e-10);
        assert!((inverse[2][2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compute_inverse_matrix_singular() {
        let singular = [[0.0; 3]; 3];

        let inverse = IccProfile::compute_inverse_matrix(&singular);

        // Should return identity for singular matrix
        assert!((inverse[0][0] - 1.0).abs() < 1e-10);
        assert!((inverse[1][1] - 1.0).abs() < 1e-10);
        assert!((inverse[2][2] - 1.0).abs() < 1e-10);
    }

    // ── New tests for from_bytes / to_bytes ────────────────────────────────────

    /// Build a minimal `IccProfile`, serialize it, parse it back, and check that
    /// all fields survive the round-trip within s15Fixed16 quantization tolerance.
    #[test]
    fn test_round_trip() {
        // Use sRGB-like primaries matrix.
        let to_xyz: Matrix3x3 = [
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505],
        ];
        let original = IccProfile {
            description: "sRGB Test".to_string(),
            version: IccProfileVersion::V4,
            to_xyz_matrix: to_xyz,
            from_xyz_matrix: IccProfile::compute_inverse_matrix(&to_xyz),
            white_point: Illuminant::D65,
            creation_date: 0, // zero = write/read as zeros
        };

        let bytes = original
            .to_bytes()
            .expect("to_bytes must succeed for valid profile");
        assert!(bytes.len() >= HEADER_SIZE, "serialized length >= 128");

        let parsed =
            IccProfile::from_bytes(&bytes).expect("from_bytes must succeed on our own output");

        assert_eq!(parsed.description, original.description);
        assert_eq!(parsed.version, original.version);
        assert_eq!(parsed.white_point, original.white_point);
        assert_eq!(parsed.creation_date, original.creation_date);

        // Matrices survive s15Fixed16 quantization: tolerance 1e-4 per entry.
        for row in 0..3 {
            for col in 0..3 {
                let orig = original.to_xyz_matrix[row][col];
                let got = parsed.to_xyz_matrix[row][col];
                assert!(
                    (orig - got).abs() < 1e-4,
                    "to_xyz_matrix[{row}][{col}]: {orig} vs {got}"
                );
            }
        }
    }

    /// from_bytes with a 'wrong acsp' signature must return Err.
    #[test]
    fn test_from_bytes_invalid_signature() {
        // Build a 128-byte block with wrong bytes at offset 36-39.
        let mut data = vec![0u8; HEADER_SIZE];
        // Write wrong magic at bytes 36-39.
        data[36] = b'X';
        data[37] = b'X';
        data[38] = b'X';
        data[39] = b'X';
        let result = IccProfile::from_bytes(&data);
        assert!(result.is_err(), "expected Err for invalid 'acsp' signature");
        let err_str = format!("{}", result.expect_err("checked above"));
        assert!(
            err_str.contains("acsp") || err_str.contains("ICC"),
            "error message should mention acsp: {err_str}"
        );
    }

    /// from_bytes on a 10-byte slice must return Err.
    #[test]
    fn test_from_bytes_too_short() {
        let tiny = vec![0u8; 10];
        let result = IccProfile::from_bytes(&tiny);
        assert!(result.is_err(), "expected Err for too-short input");
        let err_str = format!("{}", result.expect_err("checked above"));
        assert!(
            err_str.contains("too short") || err_str.contains("128") || err_str.contains("10"),
            "error should mention size: {err_str}"
        );
    }

    /// Round-trip with a non-trivial creation_date (a known Unix timestamp).
    #[test]
    fn test_round_trip_creation_date() {
        // 2024-01-15 12:00:00 UTC (approximate, from days arithmetic).
        // days_since_epoch = (2024-1970)*365 + leap_corrections + month + day offsets.
        // We use an explicit value that must round-trip exactly.
        let ts: u64 = 1_705_320_000; // 2024-01-15 12:00:00 UTC (approx)
        let to_xyz: Matrix3x3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let original = IccProfile {
            description: "DateTest".to_string(),
            version: IccProfileVersion::V2,
            to_xyz_matrix: to_xyz,
            from_xyz_matrix: IccProfile::compute_inverse_matrix(&to_xyz),
            white_point: Illuminant::D50,
            creation_date: ts,
        };
        let bytes = original.to_bytes().expect("to_bytes succeeded");
        let parsed = IccProfile::from_bytes(&bytes).expect("from_bytes succeeded");
        // Allow ±2 seconds tolerance for calendar conversion rounding.
        assert!(
            parsed.creation_date.abs_diff(ts) <= 2,
            "creation_date round-trip: expected {ts}, got {}",
            parsed.creation_date
        );
        assert_eq!(parsed.version, IccProfileVersion::V2);
    }

    /// Verify s15Fixed16 conversion round-trips correctly.
    #[test]
    fn test_s15f16_roundtrip() {
        for &v in &[0.0f64, 1.0, -1.0, 0.9642, 0.8249, 0.4124, 0.0722] {
            let encoded = f64_to_s15f16(v);
            let decoded = s15f16_to_f64(encoded);
            assert!(
                (v - decoded).abs() < 1.6e-5,
                "s15Fixed16 round-trip failed for {v}: got {decoded}"
            );
        }
    }
}
