//! HEIF/AVIF still image container support (pure-Rust parser).
//!
//! Implements ISO Base Media File Format (ISOBMFF) box parsing for HEIF and AVIF containers.
//!
//! # Features
//! - Box traversal (ftyp, meta, hdlr, pitm, iinf, iloc, iprp/ipco/ipma)
//! - ImageSpatialExtentsProperty (ispe) for dimensions
//! - ColourInformationBox (colr) for color space
//! - AVIF and HEIC brand detection
//! - Metadata extraction without decoding the AV1/HEVC bitstream

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::error::{ImageError, ImageResult};

// ── Box reader ────────────────────────────────────────────────────────────────

/// A parsed ISO BMFF box.
#[derive(Debug, Clone)]
pub struct HeifBox {
    /// 4-byte box type (e.g. "ftyp", "mdat").
    pub box_type: [u8; 4],
    /// Box payload (excluding the 8-byte header).
    pub data: Vec<u8>,
}

impl HeifBox {
    /// Returns the box type as a string (ASCII).
    #[must_use]
    pub fn type_str(&self) -> &str {
        std::str::from_utf8(&self.box_type).unwrap_or("????")
    }

    /// Returns the full box size (header + payload).
    #[must_use]
    pub fn full_size(&self) -> usize {
        self.data.len() + 8
    }

    /// Parse nested boxes from `self.data`.
    pub fn children(&self) -> ImageResult<Vec<HeifBox>> {
        parse_boxes(&self.data)
    }

    /// Parse this box as a FullBox: returns (version, flags, payload).
    pub fn full_box_fields(&self) -> ImageResult<(u8, u32, &[u8])> {
        if self.data.len() < 4 {
            return Err(ImageError::invalid_format("FullBox too short"));
        }
        let version = self.data[0];
        let flags = u32::from_be_bytes([0, self.data[1], self.data[2], self.data[3]]);
        Ok((version, flags, &self.data[4..]))
    }
}

/// Parse a sequence of ISO BMFF boxes from `data`.
pub fn parse_boxes(data: &[u8]) -> ImageResult<Vec<HeifBox>> {
    let mut boxes = Vec::new();
    let mut pos = 0usize;

    while pos + 8 <= data.len() {
        let size32 = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        let box_type = [data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]];

        let (box_size, header_size) = if size32 == 0 {
            // Box extends to end of file
            (data.len() - pos, 8)
        } else if size32 == 1 {
            // 64-bit size
            if pos + 16 > data.len() {
                return Err(ImageError::invalid_format("Extended size box truncated"));
            }
            let size64 = u64::from_be_bytes([
                data[pos + 8],
                data[pos + 9],
                data[pos + 10],
                data[pos + 11],
                data[pos + 12],
                data[pos + 13],
                data[pos + 14],
                data[pos + 15],
            ]) as usize;
            (size64, 16)
        } else {
            (size32 as usize, 8)
        };

        if box_size < header_size || pos + box_size > data.len() {
            return Err(ImageError::invalid_format(format!(
                "Box '{}' size {} exceeds data length {}",
                std::str::from_utf8(&box_type).unwrap_or("????"),
                box_size,
                data.len() - pos
            )));
        }

        let payload = data[pos + header_size..pos + box_size].to_vec();
        boxes.push(HeifBox {
            box_type,
            data: payload,
        });
        pos += box_size;
    }

    Ok(boxes)
}

/// Find the first box with `box_type` in a list.
pub fn find_box<'a>(boxes: &'a [HeifBox], type_str: &str) -> Option<&'a HeifBox> {
    let ty = type_str.as_bytes();
    boxes
        .iter()
        .find(|b| b.box_type.len() == 4 && ty.len() == 4 && b.box_type == ty[..4])
}

// ── ftyp box ─────────────────────────────────────────────────────────────────

/// HEIF/AVIF file type box (ftyp).
#[derive(Debug, Clone)]
pub struct FtypBox {
    /// Major brand (4 bytes, e.g. "avif", "heic").
    pub major_brand: [u8; 4],
    /// Minor version.
    pub minor_version: u32,
    /// Compatible brands.
    pub compatible_brands: Vec<[u8; 4]>,
}

impl FtypBox {
    /// Returns the major brand as a string.
    #[must_use]
    pub fn major_brand_str(&self) -> &str {
        std::str::from_utf8(&self.major_brand).unwrap_or("????")
    }

    /// Returns true if this is an AVIF file.
    #[must_use]
    pub fn is_avif(&self) -> bool {
        &self.major_brand == b"avif" || self.compatible_brands.iter().any(|b| b == b"avif")
    }

    /// Returns true if this is a HEIC file.
    #[must_use]
    pub fn is_heic(&self) -> bool {
        &self.major_brand == b"heic" || self.compatible_brands.iter().any(|b| b == b"heic")
    }

    fn parse(b: &HeifBox) -> ImageResult<Self> {
        let data = &b.data;
        if data.len() < 8 {
            return Err(ImageError::invalid_format("ftyp box too short"));
        }
        let major_brand = [data[0], data[1], data[2], data[3]];
        let minor_version = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        let mut compatible = Vec::new();
        let mut pos = 8;
        while pos + 4 <= data.len() {
            compatible.push([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            pos += 4;
        }
        Ok(Self {
            major_brand,
            minor_version,
            compatible_brands: compatible,
        })
    }
}

// ── Item info (iinf / infe) ───────────────────────────────────────────────────

/// An item information entry (infe box).
#[derive(Debug, Clone)]
pub struct HeifItem {
    /// Item ID.
    pub item_id: u32,
    /// Item type (4 bytes, e.g. "av01", "hvc1", "Exif").
    pub item_type: [u8; 4],
    /// Item name (may be empty).
    pub item_name: String,
    /// Protection index (0 = unprotected).
    pub protection_index: u16,
}

impl HeifItem {
    /// Returns the item type as a string.
    #[must_use]
    pub fn item_type_str(&self) -> &str {
        std::str::from_utf8(&self.item_type).unwrap_or("????")
    }

    fn parse(b: &HeifBox) -> ImageResult<Self> {
        let (version, _flags, payload) = b.full_box_fields()?;
        if version == 2 {
            if payload.len() < 6 {
                return Err(ImageError::invalid_format("infe v2 too short"));
            }
            let item_id = u16::from_be_bytes([payload[0], payload[1]]) as u32;
            let protection_index = u16::from_be_bytes([payload[2], payload[3]]);
            let item_type = if payload.len() >= 8 {
                [payload[4], payload[5], payload[6], payload[7]]
            } else {
                [0; 4]
            };
            // item_name follows as null-terminated string
            let name_start = 8.min(payload.len());
            let name_end = payload[name_start..]
                .iter()
                .position(|&b| b == 0)
                .map(|p| name_start + p)
                .unwrap_or(payload.len());
            let item_name = String::from_utf8_lossy(&payload[name_start..name_end]).into_owned();
            Ok(Self {
                item_id,
                item_type,
                item_name,
                protection_index,
            })
        } else if version == 3 {
            if payload.len() < 8 {
                return Err(ImageError::invalid_format("infe v3 too short"));
            }
            let item_id = u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);
            let protection_index = u16::from_be_bytes([payload[4], payload[5]]);
            let item_type = if payload.len() >= 10 {
                [payload[6], payload[7], payload[8], payload[9]]
            } else {
                [0; 4]
            };
            let name_start = 10.min(payload.len());
            let name_end = payload[name_start..]
                .iter()
                .position(|&b| b == 0)
                .map(|p| name_start + p)
                .unwrap_or(payload.len());
            let item_name = String::from_utf8_lossy(&payload[name_start..name_end]).into_owned();
            Ok(Self {
                item_id,
                item_type,
                item_name,
                protection_index,
            })
        } else {
            // v0/v1 fallback
            if payload.len() < 4 {
                return Err(ImageError::invalid_format("infe v0 too short"));
            }
            let item_id = u16::from_be_bytes([payload[0], payload[1]]) as u32;
            let protection_index = u16::from_be_bytes([payload[2], payload[3]]);
            Ok(Self {
                item_id,
                item_type: *b"hvc1",
                item_name: String::new(),
                protection_index,
            })
        }
    }
}

// ── Item location (iloc) ──────────────────────────────────────────────────────

/// An extent within an item location.
#[derive(Debug, Clone)]
pub struct HeifExtent {
    /// Offset of this extent (relative to base_offset).
    pub offset: u64,
    /// Length of this extent.
    pub length: u64,
}

/// An item location entry (iloc).
#[derive(Debug, Clone)]
pub struct HeifItemLocation {
    /// Item ID.
    pub item_id: u32,
    /// Construction method (0=file, 1=idat, 2=item).
    pub construction_method: u8,
    /// Base offset.
    pub base_offset: u64,
    /// Data extents.
    pub extents: Vec<HeifExtent>,
}

// ── Image spatial extents (ispe) ──────────────────────────────────────────────

/// ImageSpatialExtentsProperty: image dimensions.
#[derive(Debug, Clone, Copy)]
pub struct IspeProperty {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl IspeProperty {
    fn parse(b: &HeifBox) -> ImageResult<Self> {
        let (_version, _flags, payload) = b.full_box_fields()?;
        if payload.len() < 8 {
            return Err(ImageError::invalid_format("ispe too short"));
        }
        let width = u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);
        let height = u32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]);
        Ok(Self { width, height })
    }
}

// ── Colour information (colr) ─────────────────────────────────────────────────

/// ColourInformationBox colour type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColrType {
    /// nclx (on-screen colors).
    Nclx,
    /// rICC / prof (embedded ICC profile reference).
    Icc,
    /// Unknown.
    Unknown,
}

/// ColourInformationBox.
#[derive(Debug, Clone)]
pub struct ColrBox {
    /// Colour type.
    pub colour_type: ColrType,
    /// Colour primaries (nclx only).
    pub colour_primaries: u16,
    /// Transfer characteristics (nclx only).
    pub transfer_characteristics: u16,
    /// Matrix coefficients (nclx only).
    pub matrix_coefficients: u16,
    /// Full-range flag (nclx only).
    pub full_range: bool,
}

impl ColrBox {
    fn parse(b: &HeifBox) -> ImageResult<Self> {
        let data = &b.data;
        if data.len() < 4 {
            return Err(ImageError::invalid_format("colr too short"));
        }
        let colour_type_bytes = [data[0], data[1], data[2], data[3]];
        let colour_type = match &colour_type_bytes {
            b"nclx" => ColrType::Nclx,
            b"rICC" | b"prof" => ColrType::Icc,
            _ => ColrType::Unknown,
        };
        let (colour_primaries, transfer_characteristics, matrix_coefficients, full_range) =
            if colour_type == ColrType::Nclx && data.len() >= 11 {
                let cp = u16::from_be_bytes([data[4], data[5]]);
                let tc = u16::from_be_bytes([data[6], data[7]]);
                let mc = u16::from_be_bytes([data[8], data[9]]);
                let fr = data[10] & 0x80 != 0;
                (cp, tc, mc, fr)
            } else {
                (0, 0, 0, false)
            };
        Ok(Self {
            colour_type,
            colour_primaries,
            transfer_characteristics,
            matrix_coefficients,
            full_range,
        })
    }
}

// ── Primary item (pitm) ───────────────────────────────────────────────────────

fn parse_pitm(b: &HeifBox) -> ImageResult<u32> {
    let (version, _flags, payload) = b.full_box_fields()?;
    if version == 0 {
        if payload.len() < 2 {
            return Err(ImageError::invalid_format("pitm v0 too short"));
        }
        Ok(u16::from_be_bytes([payload[0], payload[1]]) as u32)
    } else {
        if payload.len() < 4 {
            return Err(ImageError::invalid_format("pitm v1 too short"));
        }
        Ok(u32::from_be_bytes([
            payload[0], payload[1], payload[2], payload[3],
        ]))
    }
}

// ── HEIF container ────────────────────────────────────────────────────────────

/// Parsed HEIF/AVIF container.
#[derive(Debug, Clone)]
pub struct HeifContainer {
    /// File type box.
    pub ftyp: FtypBox,
    /// Primary item ID.
    pub primary_item_id: u32,
    /// Item info entries.
    pub items: Vec<HeifItem>,
    /// Image spatial extents properties.
    pub ispe_properties: Vec<IspeProperty>,
    /// Colour information box.
    pub colr: Option<ColrBox>,
    /// Item locations.
    pub item_locations: Vec<HeifItemLocation>,
    /// Raw top-level boxes.
    pub raw_boxes: Vec<HeifBox>,
}

impl HeifContainer {
    /// Find item by ID.
    pub fn item_by_id(&self, id: u32) -> Option<&HeifItem> {
        self.items.iter().find(|it| it.item_id == id)
    }

    /// Returns the primary item.
    pub fn primary_item(&self) -> Option<&HeifItem> {
        self.item_by_id(self.primary_item_id)
    }

    /// Returns the first ispe (image dimensions).
    pub fn image_dimensions(&self) -> Option<(u32, u32)> {
        self.ispe_properties.first().map(|p| (p.width, p.height))
    }
}

// ── AVIF metadata ─────────────────────────────────────────────────────────────

/// Extracted AVIF/HEIF metadata.
#[derive(Debug, Clone)]
pub struct AvifMetadata {
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Bit depth (typically 8, 10, or 12).
    pub bit_depth: u8,
    /// Whether this is HDR content.
    pub is_hdr: bool,
    /// Primary color space description.
    pub color_space: String,
    /// File format: "avif" or "heic".
    pub format: String,
}

/// Extract high-level metadata from a parsed HEIF container.
pub fn extract_metadata(container: &HeifContainer) -> ImageResult<AvifMetadata> {
    let (width, height) = container
        .image_dimensions()
        .ok_or_else(|| ImageError::invalid_format("No image dimensions found in HEIF container"))?;

    let format = container.ftyp.major_brand_str().to_string();

    let (is_hdr, color_space, bit_depth) = if let Some(colr) = &container.colr {
        // BT.2020 primaries = 9, PQ transfer = 16, HLG = 18
        let hdr = colr.transfer_characteristics == 16 || colr.transfer_characteristics == 18;
        let cs = match colr.colour_primaries {
            1 => "BT.709".to_string(),
            9 => "BT.2020".to_string(),
            12 => "DCI-P3".to_string(),
            _ => format!("primaries={}", colr.colour_primaries),
        };
        let depth = if hdr { 10u8 } else { 8u8 };
        (hdr, cs, depth)
    } else {
        (false, "sRGB".to_string(), 8u8)
    };

    Ok(AvifMetadata {
        width,
        height,
        bit_depth,
        is_hdr,
        color_space,
        format,
    })
}

// ── iloc parse ────────────────────────────────────────────────────────────────

fn parse_iloc(b: &HeifBox) -> ImageResult<Vec<HeifItemLocation>> {
    let (version, _flags, payload) = b.full_box_fields()?;
    if payload.is_empty() {
        return Ok(Vec::new());
    }
    let offset_size = (payload[0] >> 4) & 0x0F;
    let length_size = payload[0] & 0x0F;
    let base_offset_size = (payload[1] >> 4) & 0x0F;
    let index_size = if version >= 1 { payload[1] & 0x0F } else { 0 };
    let mut pos = 2;

    let item_count = if version < 2 {
        if pos + 2 > payload.len() {
            return Ok(Vec::new());
        }
        let count = u16::from_be_bytes([payload[pos], payload[pos + 1]]) as u32;
        pos += 2;
        count
    } else {
        if pos + 4 > payload.len() {
            return Ok(Vec::new());
        }
        let count = u32::from_be_bytes([
            payload[pos],
            payload[pos + 1],
            payload[pos + 2],
            payload[pos + 3],
        ]);
        pos += 4;
        count
    };

    let mut locations = Vec::with_capacity(item_count as usize);

    for _ in 0..item_count {
        let item_id = if version < 2 {
            if pos + 2 > payload.len() {
                break;
            }
            let id = u16::from_be_bytes([payload[pos], payload[pos + 1]]) as u32;
            pos += 2;
            id
        } else {
            if pos + 4 > payload.len() {
                break;
            }
            let id = u32::from_be_bytes([
                payload[pos],
                payload[pos + 1],
                payload[pos + 2],
                payload[pos + 3],
            ]);
            pos += 4;
            id
        };

        let construction_method = if version >= 1 {
            if pos + 2 > payload.len() {
                break;
            }
            let cm = payload[pos + 1] & 0x0F;
            pos += 2;
            cm
        } else {
            0
        };

        if pos + 2 > payload.len() {
            break;
        }
        let _data_ref = u16::from_be_bytes([payload[pos], payload[pos + 1]]);
        pos += 2;

        let base_offset = read_uint(payload, &mut pos, base_offset_size as usize);

        if pos + 2 > payload.len() {
            break;
        }
        let extent_count = u16::from_be_bytes([payload[pos], payload[pos + 1]]) as usize;
        pos += 2;

        let mut extents = Vec::with_capacity(extent_count);
        for _ in 0..extent_count {
            if version >= 1 && index_size > 0 {
                let _ = read_uint(payload, &mut pos, index_size as usize);
            }
            let offset = read_uint(payload, &mut pos, offset_size as usize);
            let length = read_uint(payload, &mut pos, length_size as usize);
            extents.push(HeifExtent { offset, length });
        }

        locations.push(HeifItemLocation {
            item_id,
            construction_method,
            base_offset,
            extents,
        });
    }

    Ok(locations)
}

fn read_uint(data: &[u8], pos: &mut usize, size: usize) -> u64 {
    if size == 0 {
        return 0;
    }
    let end = (*pos + size).min(data.len());
    let actual = end - *pos;
    let mut v = 0u64;
    for i in 0..actual {
        v = (v << 8) | data[*pos + i] as u64;
    }
    *pos = end;
    v
}

// ── iprp / ipco / ispe parsing ────────────────────────────────────────────────

fn parse_iprp(b: &HeifBox) -> ImageResult<(Vec<IspeProperty>, Option<ColrBox>)> {
    let children = b.children()?;
    let mut ispe_vec = Vec::new();
    let mut colr_box = None;

    if let Some(ipco) = find_box(&children, "ipco") {
        let props = ipco.children()?;
        for prop in &props {
            if prop.type_str() == "ispe" {
                if let Ok(ispe) = IspeProperty::parse(prop) {
                    ispe_vec.push(ispe);
                }
            } else if prop.type_str() == "colr" {
                colr_box = ColrBox::parse(prop).ok();
            }
        }
    }

    Ok((ispe_vec, colr_box))
}

fn parse_meta(
    b: &HeifBox,
) -> ImageResult<(
    u32,
    Vec<HeifItem>,
    Vec<IspeProperty>,
    Option<ColrBox>,
    Vec<HeifItemLocation>,
)> {
    // meta is a FullBox (skip 4-byte version+flags)
    let data = &b.data;
    if data.len() < 4 {
        return Err(ImageError::invalid_format("meta box too short"));
    }
    let children = parse_boxes(&data[4..])?;

    let primary_item_id = find_box(&children, "pitm")
        .and_then(|b| parse_pitm(b).ok())
        .unwrap_or(0);

    let items: Vec<HeifItem> = find_box(&children, "iinf")
        .and_then(|iinf_box| {
            let payload = &iinf_box.data;
            // iinf is a FullBox: 4 bytes version+flags + 2 or 4 bytes count
            if payload.len() < 6 {
                return None;
            }
            let _version = payload[0];
            let nested = parse_boxes(&payload[6..]).ok()?;
            let items: Vec<HeifItem> = nested
                .iter()
                .filter(|b| b.type_str() == "infe")
                .filter_map(|b| HeifItem::parse(b).ok())
                .collect();
            Some(items)
        })
        .unwrap_or_default();

    let (ispe_props, colr) = find_box(&children, "iprp")
        .and_then(|b| parse_iprp(b).ok())
        .unwrap_or_default();

    let locs = find_box(&children, "iloc")
        .and_then(|b| parse_iloc(b).ok())
        .unwrap_or_default();

    Ok((primary_item_id, items, ispe_props, colr, locs))
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse a HEIF/AVIF container from bytes.
pub fn parse_heif(data: &[u8]) -> ImageResult<HeifContainer> {
    if data.len() < 12 {
        return Err(ImageError::invalid_format("HEIF data too short"));
    }

    let raw_boxes = parse_boxes(data)?;

    let ftyp_box = find_box(&raw_boxes, "ftyp")
        .ok_or_else(|| ImageError::invalid_format("No ftyp box found"))?;
    let ftyp = FtypBox::parse(ftyp_box)?;

    if !ftyp.is_avif() && !ftyp.is_heic() {
        // Check for mif1 (generic HEIF)
        let is_heif = ftyp
            .compatible_brands
            .iter()
            .any(|b| b == b"mif1" || b == b"msf1");
        if !is_heif && &ftyp.major_brand != b"mif1" {
            return Err(ImageError::unsupported(format!(
                "Unsupported HEIF brand: '{}'",
                ftyp.major_brand_str()
            )));
        }
    }

    let (primary_item_id, items, ispe_properties, colr, item_locations) =
        find_box(&raw_boxes, "meta")
            .and_then(|b| parse_meta(b).ok())
            .unwrap_or_default();

    Ok(HeifContainer {
        ftyp,
        primary_item_id,
        items,
        ispe_properties,
        colr,
        item_locations,
        raw_boxes,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal ftyp box for testing.
    fn minimal_ftyp(brand: &[u8; 4]) -> Vec<u8> {
        let mut data = Vec::new();
        // size (8 + 4 + 4 = 16 bytes)
        data.extend_from_slice(&16u32.to_be_bytes());
        data.extend_from_slice(b"ftyp");
        data.extend_from_slice(brand); // major brand
        data.extend_from_slice(&0u32.to_be_bytes()); // minor version
        data
    }

    fn minimal_ispe_box(w: u32, h: u32) -> Vec<u8> {
        let mut data = Vec::new();
        // ispe: 8 header + 4 fullbox + 8 payload = 20
        data.extend_from_slice(&20u32.to_be_bytes());
        data.extend_from_slice(b"ispe");
        data.extend_from_slice(&[0u8, 0, 0, 0]); // version + flags
        data.extend_from_slice(&w.to_be_bytes());
        data.extend_from_slice(&h.to_be_bytes());
        data
    }

    #[test]
    fn test_parse_boxes_single_ftyp() {
        let ftyp = minimal_ftyp(b"avif");
        let boxes = parse_boxes(&ftyp).expect("parse boxes");
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].type_str(), "ftyp");
    }

    #[test]
    fn test_ftyp_is_avif() {
        let raw = minimal_ftyp(b"avif");
        let boxes = parse_boxes(&raw).expect("parse");
        let ftyp = FtypBox::parse(&boxes[0]).expect("ftyp");
        assert!(ftyp.is_avif());
        assert!(!ftyp.is_heic());
    }

    #[test]
    fn test_ftyp_is_heic() {
        let raw = minimal_ftyp(b"heic");
        let boxes = parse_boxes(&raw).expect("parse");
        let ftyp = FtypBox::parse(&boxes[0]).expect("ftyp");
        assert!(ftyp.is_heic());
        assert!(!ftyp.is_avif());
    }

    #[test]
    fn test_ftyp_major_brand_str() {
        let raw = minimal_ftyp(b"mif1");
        let boxes = parse_boxes(&raw).expect("parse");
        let ftyp = FtypBox::parse(&boxes[0]).expect("ftyp");
        assert_eq!(ftyp.major_brand_str(), "mif1");
    }

    #[test]
    fn test_parse_ispe_property() {
        let ispe_raw = minimal_ispe_box(1920, 1080);
        let boxes = parse_boxes(&ispe_raw).expect("parse ispe");
        assert_eq!(boxes.len(), 1);
        let ispe = IspeProperty::parse(&boxes[0]).expect("ispe parse");
        assert_eq!(ispe.width, 1920);
        assert_eq!(ispe.height, 1080);
    }

    #[test]
    fn test_box_type_str() {
        let b = HeifBox {
            box_type: *b"mdat",
            data: vec![],
        };
        assert_eq!(b.type_str(), "mdat");
    }

    #[test]
    fn test_box_full_size() {
        let b = HeifBox {
            box_type: *b"ftyp",
            data: vec![0u8; 8],
        };
        assert_eq!(b.full_size(), 16); // 8 header + 8 data
    }

    #[test]
    fn test_parse_boxes_empty_data() {
        let boxes = parse_boxes(&[]).expect("empty");
        assert!(boxes.is_empty());
    }

    #[test]
    fn test_parse_boxes_truncated() {
        // Only 4 bytes, not enough for a box header
        let data = &[0u8, 0, 0, 8];
        let result = parse_boxes(data);
        // Either returns empty or an error — should not panic
        if let Ok(boxes) = result {
            assert!(boxes.is_empty());
        }
    }

    #[test]
    fn test_parse_boxes_multiple() {
        let mut data = minimal_ftyp(b"avif");
        // Append a minimal mdat box
        let mdat_size = 8u32;
        data.extend_from_slice(&mdat_size.to_be_bytes());
        data.extend_from_slice(b"mdat");
        let boxes = parse_boxes(&data).expect("parse multiple");
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[0].type_str(), "ftyp");
        assert_eq!(boxes[1].type_str(), "mdat");
    }

    #[test]
    fn test_find_box_found() {
        let b1 = HeifBox {
            box_type: *b"ftyp",
            data: vec![],
        };
        let b2 = HeifBox {
            box_type: *b"mdat",
            data: vec![],
        };
        let boxes = vec![b1, b2];
        let found = find_box(&boxes, "mdat");
        assert!(found.is_some());
        assert_eq!(found.expect("found").type_str(), "mdat");
    }

    #[test]
    fn test_find_box_not_found() {
        let b = HeifBox {
            box_type: *b"ftyp",
            data: vec![],
        };
        let boxes = vec![b];
        assert!(find_box(&boxes, "meta").is_none());
    }

    #[test]
    fn test_parse_heif_empty_data() {
        let result = parse_heif(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_heif_no_ftyp() {
        // A mdat box without ftyp
        let mut data = vec![0u8; 8];
        data[3] = 8;
        data[4..8].copy_from_slice(b"mdat");
        let result = parse_heif(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_metadata_no_dimensions() {
        let container = HeifContainer {
            ftyp: FtypBox {
                major_brand: *b"avif",
                minor_version: 0,
                compatible_brands: vec![],
            },
            primary_item_id: 1,
            items: vec![],
            ispe_properties: vec![],
            colr: None,
            item_locations: vec![],
            raw_boxes: vec![],
        };
        let result = extract_metadata(&container);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_metadata_with_dimensions() {
        let container = HeifContainer {
            ftyp: FtypBox {
                major_brand: *b"avif",
                minor_version: 0,
                compatible_brands: vec![],
            },
            primary_item_id: 1,
            items: vec![],
            ispe_properties: vec![IspeProperty {
                width: 3840,
                height: 2160,
            }],
            colr: None,
            item_locations: vec![],
            raw_boxes: vec![],
        };
        let meta = extract_metadata(&container).expect("metadata");
        assert_eq!(meta.width, 3840);
        assert_eq!(meta.height, 2160);
        assert_eq!(meta.format, "avif");
        assert!(!meta.is_hdr);
    }

    #[test]
    fn test_extract_metadata_hdr() {
        let container = HeifContainer {
            ftyp: FtypBox {
                major_brand: *b"avif",
                minor_version: 0,
                compatible_brands: vec![],
            },
            primary_item_id: 1,
            items: vec![],
            ispe_properties: vec![IspeProperty {
                width: 1920,
                height: 1080,
            }],
            colr: Some(ColrBox {
                colour_type: ColrType::Nclx,
                colour_primaries: 9,          // BT.2020
                transfer_characteristics: 16, // PQ
                matrix_coefficients: 9,
                full_range: false,
            }),
            item_locations: vec![],
            raw_boxes: vec![],
        };
        let meta = extract_metadata(&container).expect("metadata");
        assert!(meta.is_hdr);
        assert_eq!(meta.bit_depth, 10);
        assert!(meta.color_space.contains("2020"));
    }

    #[test]
    fn test_colr_nclx_parse() {
        let mut data = vec![0u8; 11];
        data[0..4].copy_from_slice(b"nclx");
        // colour_primaries = 9 (BT.2020)
        data[4] = 0;
        data[5] = 9;
        // transfer_characteristics = 16 (PQ)
        data[6] = 0;
        data[7] = 16;
        // matrix = 9
        data[8] = 0;
        data[9] = 9;
        // full range
        data[10] = 0x80;
        let b = HeifBox {
            box_type: *b"colr",
            data,
        };
        let colr = ColrBox::parse(&b).expect("colr");
        assert_eq!(colr.colour_type, ColrType::Nclx);
        assert_eq!(colr.colour_primaries, 9);
        assert_eq!(colr.transfer_characteristics, 16);
        assert!(colr.full_range);
    }

    #[test]
    fn test_heif_item_location_parse_empty() {
        // iloc with empty payload
        let b = HeifBox {
            box_type: *b"iloc",
            data: vec![0u8; 4],
        }; // version+flags only
        let locs = parse_iloc(&b).expect("iloc empty");
        assert!(locs.is_empty());
    }
}
