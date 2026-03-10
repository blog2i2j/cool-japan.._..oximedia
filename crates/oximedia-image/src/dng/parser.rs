//! Minimal TIFF parser for DNG file reading.

use crate::error::{ImageError, ImageResult};

// ==========================================
// TIFF Parser (minimal, in-memory for DNG)
// ==========================================

#[derive(Debug, Clone)]
pub(crate) struct TiffIfd {
    pub(crate) entries: Vec<TiffEntry>,
}

#[derive(Debug, Clone)]
pub(crate) struct TiffEntry {
    pub(crate) tag: u16,
    pub(crate) data_type: u16,
    pub(crate) count: u32,
    pub(crate) value_offset: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ByteOrder {
    LittleEndian,
    BigEndian,
}

pub(crate) struct TiffParser {
    pub(crate) byte_order: ByteOrder,
}

impl TiffParser {
    /// Parse TIFF header and all IFDs from raw bytes.
    pub(crate) fn parse(data: &[u8]) -> ImageResult<(ByteOrder, Vec<TiffIfd>)> {
        if data.len() < 8 {
            return Err(ImageError::invalid_format("Data too short for TIFF header"));
        }

        let byte_order = match (data[0], data[1]) {
            (0x49, 0x49) => ByteOrder::LittleEndian,
            (0x4D, 0x4D) => ByteOrder::BigEndian,
            _ => return Err(ImageError::invalid_format("Invalid TIFF byte order marker")),
        };

        let parser = TiffParser { byte_order };
        let version = parser.read_u16(data, 2)?;
        if version != 42 {
            return Err(ImageError::invalid_format(format!(
                "Invalid TIFF version: {version} (expected 42)"
            )));
        }

        let mut ifd_offset = parser.read_u32(data, 4)? as usize;
        let mut ifds = Vec::new();

        // Parse all IFDs in chain
        let max_ifds = 64; // safety limit
        for _ in 0..max_ifds {
            if ifd_offset == 0 || ifd_offset >= data.len() {
                break;
            }
            let (ifd, next_offset) = parser.read_ifd(data, ifd_offset)?;
            ifds.push(ifd);
            ifd_offset = next_offset.unwrap_or(0);
        }

        if ifds.is_empty() {
            return Err(ImageError::invalid_format("No IFDs found in TIFF"));
        }

        Ok((byte_order, ifds))
    }

    pub(crate) fn read_u16(&self, data: &[u8], offset: usize) -> ImageResult<u16> {
        if offset + 2 > data.len() {
            return Err(ImageError::invalid_format(
                "Unexpected end of data reading u16",
            ));
        }
        Ok(match self.byte_order {
            ByteOrder::LittleEndian => u16::from_le_bytes([data[offset], data[offset + 1]]),
            ByteOrder::BigEndian => u16::from_be_bytes([data[offset], data[offset + 1]]),
        })
    }

    pub(crate) fn read_u32(&self, data: &[u8], offset: usize) -> ImageResult<u32> {
        if offset + 4 > data.len() {
            return Err(ImageError::invalid_format(
                "Unexpected end of data reading u32",
            ));
        }
        Ok(match self.byte_order {
            ByteOrder::LittleEndian => u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]),
            ByteOrder::BigEndian => u32::from_be_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]),
        })
    }

    fn read_ifd(&self, data: &[u8], offset: usize) -> ImageResult<(TiffIfd, Option<usize>)> {
        if offset + 2 > data.len() {
            return Err(ImageError::invalid_format("IFD offset out of bounds"));
        }
        let entry_count = self.read_u16(data, offset)? as usize;
        let entries_start = offset + 2;
        let entries_end = entries_start + entry_count * 12;

        if entries_end + 4 > data.len() {
            return Err(ImageError::invalid_format("IFD entries extend beyond data"));
        }

        let mut entries = Vec::with_capacity(entry_count);
        for i in 0..entry_count {
            let entry_offset = entries_start + i * 12;
            let tag = self.read_u16(data, entry_offset)?;
            let data_type = self.read_u16(data, entry_offset + 2)?;
            let count = self.read_u32(data, entry_offset + 4)?;
            let value_offset = self.read_u32(data, entry_offset + 8)?;

            entries.push(TiffEntry {
                tag,
                data_type,
                count,
                value_offset,
            });
        }

        let next_ifd_offset = self.read_u32(data, entries_end)? as usize;
        let next = if next_ifd_offset > 0 {
            Some(next_ifd_offset)
        } else {
            None
        };

        Ok((TiffIfd { entries }, next))
    }

    /// Get a single u32 value from an IFD tag.
    pub(crate) fn get_tag_value_u32(&self, ifd: &TiffIfd, tag: u16, data: &[u8]) -> Option<u32> {
        let entry = ifd.entries.iter().find(|e| e.tag == tag)?;
        let type_size = tiff_type_size(entry.data_type);
        let total_size = type_size * entry.count as usize;

        if total_size <= 4 {
            // Value is inline
            if entry.data_type == 3 {
                // SHORT
                Some(entry.value_offset & 0xFFFF)
            } else {
                Some(entry.value_offset)
            }
        } else {
            let offset = entry.value_offset as usize;
            self.read_u32(data, offset).ok()
        }
    }

    /// Get a single u16 value from an IFD tag.
    pub(crate) fn get_tag_value_u16(&self, ifd: &TiffIfd, tag: u16, _data: &[u8]) -> Option<u16> {
        let entry = ifd.entries.iter().find(|e| e.tag == tag)?;
        let type_size = tiff_type_size(entry.data_type);
        let total_size = type_size * entry.count as usize;

        if total_size <= 4 {
            Some((entry.value_offset & 0xFFFF) as u16)
        } else {
            None
        }
    }

    /// Get a string value from an IFD tag.
    pub(crate) fn get_tag_value_string(
        &self,
        ifd: &TiffIfd,
        tag: u16,
        data: &[u8],
    ) -> Option<String> {
        let entry = ifd.entries.iter().find(|e| e.tag == tag)?;
        let count = entry.count as usize;
        let type_size = tiff_type_size(entry.data_type);
        let total_size = type_size * count;

        if total_size <= 4 {
            // Inline string
            let bytes = match self.byte_order {
                ByteOrder::LittleEndian => entry.value_offset.to_le_bytes(),
                ByteOrder::BigEndian => entry.value_offset.to_be_bytes(),
            };
            let len = count.min(4);
            Some(
                String::from_utf8_lossy(&bytes[..len])
                    .trim_end_matches('\0')
                    .to_string(),
            )
        } else {
            let offset = entry.value_offset as usize;
            if offset + count > data.len() {
                return None;
            }
            Some(
                String::from_utf8_lossy(&data[offset..offset + count])
                    .trim_end_matches('\0')
                    .to_string(),
            )
        }
    }

    /// Get multiple f64 values from an IFD tag (RATIONAL or SRATIONAL).
    pub(crate) fn get_tag_values_f64(
        &self,
        ifd: &TiffIfd,
        tag: u16,
        data: &[u8],
    ) -> Option<Vec<f64>> {
        let entry = ifd.entries.iter().find(|e| e.tag == tag)?;
        let count = entry.count as usize;
        let type_size = tiff_type_size(entry.data_type);
        let total_size = type_size * count;
        let offset = if total_size <= 4 {
            // For rationals this can't happen (8 bytes each), but handle SHORT/LONG
            return self.get_tag_values_as_f64_inline(entry);
        } else {
            entry.value_offset as usize
        };

        let mut values = Vec::with_capacity(count);
        match entry.data_type {
            5 => {
                // RATIONAL (unsigned)
                for i in 0..count {
                    let num_off = offset + i * 8;
                    let num = self.read_u32(data, num_off).ok()? as f64;
                    let den = self.read_u32(data, num_off + 4).ok()? as f64;
                    if den.abs() < f64::EPSILON {
                        values.push(0.0);
                    } else {
                        values.push(num / den);
                    }
                }
            }
            10 => {
                // SRATIONAL (signed)
                for i in 0..count {
                    let num_off = offset + i * 8;
                    let num = self.read_u32(data, num_off).ok()? as i32 as f64;
                    let den = self.read_u32(data, num_off + 4).ok()? as i32 as f64;
                    if den.abs() < f64::EPSILON {
                        values.push(0.0);
                    } else {
                        values.push(num / den);
                    }
                }
            }
            3 => {
                // SHORT
                for i in 0..count {
                    let val = self.read_u16(data, offset + i * 2).ok()?;
                    values.push(f64::from(val));
                }
            }
            4 => {
                // LONG
                for i in 0..count {
                    let val = self.read_u32(data, offset + i * 4).ok()?;
                    values.push(val as f64);
                }
            }
            _ => return None,
        }

        Some(values)
    }

    /// Get multiple u32 values from an IFD tag.
    pub(crate) fn get_tag_values_u32(
        &self,
        ifd: &TiffIfd,
        tag: u16,
        data: &[u8],
    ) -> Option<Vec<u32>> {
        let entry = ifd.entries.iter().find(|e| e.tag == tag)?;
        let count = entry.count as usize;
        let type_size = tiff_type_size(entry.data_type);
        let total_size = type_size * count;

        if total_size <= 4 && count == 1 {
            return Some(vec![entry.value_offset]);
        }

        let offset = entry.value_offset as usize;
        let mut values = Vec::with_capacity(count);

        match entry.data_type {
            3 => {
                // SHORT
                for i in 0..count {
                    let val = self.read_u16(data, offset + i * 2).ok()?;
                    values.push(u32::from(val));
                }
            }
            4 => {
                // LONG
                for i in 0..count {
                    let val = self.read_u32(data, offset + i * 4).ok()?;
                    values.push(val);
                }
            }
            _ => return None,
        }

        Some(values)
    }

    /// Get raw bytes from an IFD tag.
    pub(crate) fn get_tag_raw_bytes<'a>(
        &self,
        ifd: &TiffIfd,
        tag: u16,
        data: &'a [u8],
    ) -> Option<&'a [u8]> {
        let entry = ifd.entries.iter().find(|e| e.tag == tag)?;
        let count = entry.count as usize;
        let type_size = tiff_type_size(entry.data_type);
        let total_size = type_size * count;

        if total_size <= 4 {
            None // inline data, not a slice
        } else {
            let offset = entry.value_offset as usize;
            if offset + total_size > data.len() {
                return None;
            }
            Some(&data[offset..offset + total_size])
        }
    }

    fn get_tag_values_as_f64_inline(&self, entry: &TiffEntry) -> Option<Vec<f64>> {
        match entry.data_type {
            3 => {
                // SHORT inline
                let lo = (entry.value_offset & 0xFFFF) as u16;
                let hi = ((entry.value_offset >> 16) & 0xFFFF) as u16;
                let count = entry.count as usize;
                let mut vals = Vec::with_capacity(count);
                if count >= 1 {
                    vals.push(f64::from(lo));
                }
                if count >= 2 {
                    vals.push(f64::from(hi));
                }
                Some(vals)
            }
            4 => {
                // LONG inline
                if entry.count == 1 {
                    Some(vec![entry.value_offset as f64])
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// Returns the byte size of a TIFF data type.
pub(crate) fn tiff_type_size(data_type: u16) -> usize {
    match data_type {
        1 | 2 | 6 | 7 => 1, // BYTE, ASCII, SBYTE, UNDEFINED
        3 | 8 => 2,         // SHORT, SSHORT
        4 | 9 | 11 => 4,    // LONG, SLONG, FLOAT
        5 | 10 | 12 => 8,   // RATIONAL, SRATIONAL, DOUBLE
        _ => 1,
    }
}
