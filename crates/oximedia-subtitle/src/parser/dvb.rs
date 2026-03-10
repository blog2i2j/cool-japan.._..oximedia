//! DVB (Digital Video Broadcasting) subtitle decoder.
//!
//! DVB subtitles are bitmap-based subtitles used in European digital television.
//! They are defined in ETSI EN 300 743.
//!
//! This module implements a complete DVB subtitle decoder including:
//! - Page, region, CLUT, and object data segment parsing
//! - Run-Length Encoding (RLE) pixel decoding for 2-bit, 4-bit, and 8-bit depths
//! - YCbCr → RGBA colour conversion per BT.601
//! - Full bitmap compositing from region and object references

use crate::{SubtitleError, SubtitleResult};
use std::collections::HashMap;

/// DVB subtitle decoder.
pub struct DvbDecoder {
    /// Page compositions.
    pages: HashMap<u16, PageComposition>,
    /// Region compositions.
    regions: HashMap<u8, RegionComposition>,
    /// CLUT definitions (Color Look-Up Tables).
    cluts: HashMap<u8, Clut>,
    /// Object data.
    objects: HashMap<u16, ObjectData>,
    /// Display definition.
    display_definition: Option<DisplayDefinition>,
    /// Decoded subtitles (as bitmaps).
    subtitles: Vec<DvbSubtitle>,
}

/// A DVB subtitle (bitmap-based).
#[derive(Clone, Debug)]
pub struct DvbSubtitle {
    /// Start time in milliseconds.
    pub start_time: i64,
    /// End time in milliseconds.
    pub end_time: i64,
    /// Page ID.
    pub page_id: u16,
    /// Regions to display.
    pub regions: Vec<RegionDisplay>,
}

/// Region display information.
#[derive(Clone, Debug)]
pub struct RegionDisplay {
    /// Region ID.
    pub region_id: u8,
    /// Horizontal position.
    pub x: u16,
    /// Vertical position.
    pub y: u16,
    /// Region width.
    pub width: u16,
    /// Region height.
    pub height: u16,
    /// Bitmap data (RGBA, row-major).
    pub bitmap: Vec<u8>,
}

/// Page composition.
#[derive(Clone, Debug)]
struct PageComposition {
    page_id: u16,
    page_timeout: u8,
    page_version: u8,
    page_state: u8,
    regions: Vec<RegionReference>,
}

/// Reference to a region in a page.
#[derive(Clone, Debug)]
struct RegionReference {
    region_id: u8,
    horizontal_address: u16,
    vertical_address: u16,
}

/// Region composition.
#[derive(Clone, Debug)]
struct RegionComposition {
    region_id: u8,
    region_version: u8,
    region_width: u16,
    region_height: u16,
    region_depth: u8,
    clut_id: u8,
    objects: Vec<ObjectReference>,
}

/// Reference to an object in a region.
#[derive(Clone, Debug)]
struct ObjectReference {
    object_id: u16,
    object_type: u8,
    horizontal_position: u16,
    vertical_position: u16,
    foreground_pixel_code: u8,
    background_pixel_code: u8,
}

/// Color Look-Up Table.
#[derive(Clone, Debug)]
struct Clut {
    clut_id: u8,
    clut_version: u8,
    entries: HashMap<u8, ClutEntry>,
}

/// CLUT entry (color).
#[derive(Clone, Copy, Debug)]
struct ClutEntry {
    r: u8,
    g: u8,
    b: u8,
    t: u8, // Alpha (255 = fully opaque, 0 = transparent)
}

impl ClutEntry {
    /// Transparent black entry used as a default when an index is missing.
    const TRANSPARENT: Self = Self {
        r: 0,
        g: 0,
        b: 0,
        t: 0,
    };
}

/// Object data (bitmap).
#[derive(Clone, Debug)]
struct ObjectData {
    object_id: u16,
    object_version: u8,
    coding_method: u8,
    non_modifying_color_flag: bool,
    top_field_data: Vec<u8>,
    bottom_field_data: Vec<u8>,
}

/// Display definition.
#[derive(Clone, Debug)]
struct DisplayDefinition {
    width: u16,
    height: u16,
}

// ---------------------------------------------------------------------------
// RLE pixel decoding
// ---------------------------------------------------------------------------

/// Decode a 2-bit per pixel RLE data stream (ETSI EN 300 743 §10.4.4.1).
///
/// Pixel values are indices into a 4-entry CLUT.  The data ends when the
/// scanner exhausts `data` bytes.  Each encoded unit is variable length:
///
/// - `00` switch code followed by:
///   - `00` → end of line
///   - `01` → 2 transparent pixels
///   - `10 nn` (2 bits) → `nn+3` pixels of value 0 (transparent run)
///   - `11 cc nn` (2+2 bits) → `nn+4` pixels of colour `cc`
/// - Non-zero 2-bit pixel value → 1 pixel of that value
fn decode_2bit_rle(data: &[u8], width: u16) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width as usize);
    // We need bit-level access.
    let mut bit_offset: usize = 0;
    let total_bits = data.len() * 8;

    let read_bits = |offset: usize, count: usize| -> Option<u8> {
        if offset + count > total_bits {
            return None;
        }
        let mut val: u8 = 0;
        for i in 0..count {
            let byte_idx = (offset + i) / 8;
            let bit_idx = 7 - ((offset + i) % 8);
            val = (val << 1) | ((data[byte_idx] >> bit_idx) & 1);
        }
        Some(val)
    };

    // Process rows until we hit an end-of-line marker or exhaust bits.
    loop {
        // Read 2-bit pixel.
        let Some(code) = read_bits(bit_offset, 2) else {
            break;
        };
        bit_offset += 2;

        if code != 0 {
            // Non-zero → single pixel of colour `code`.
            pixels.push(code);
        } else {
            // Switch code — read the next 2 bits for the sub-command.
            let Some(sub) = read_bits(bit_offset, 2) else {
                break;
            };
            bit_offset += 2;

            match sub {
                0b00 => {
                    // End of line — break inner loop; caller may call again for next row.
                    break;
                }
                0b01 => {
                    // 2 transparent pixels.
                    pixels.push(0);
                    pixels.push(0);
                }
                0b10 => {
                    // Run of transparent: next 2 bits = run_length - 3.
                    let Some(run_bits) = read_bits(bit_offset, 2) else {
                        break;
                    };
                    bit_offset += 2;
                    let run = run_bits as usize + 3;
                    for _ in 0..run {
                        pixels.push(0);
                    }
                }
                0b11 => {
                    // Run of colour: next 2 bits = colour, next 2 bits = run_length - 4.
                    let Some(colour) = read_bits(bit_offset, 2) else {
                        break;
                    };
                    bit_offset += 2;
                    let Some(run_bits) = read_bits(bit_offset, 2) else {
                        break;
                    };
                    bit_offset += 2;
                    let run = run_bits as usize + 4;
                    for _ in 0..run {
                        pixels.push(colour);
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    pixels
}

/// Decode a 4-bit per pixel RLE data stream (ETSI EN 300 743 §10.4.4.2).
///
/// Pixels are 4-bit CLUT indices.  The stream is byte-aligned; each byte
/// encodes 2 pixels or a run-length escape sequence.
///
/// - Top nibble non-zero → 1 pixel of that value; then process bottom nibble.
/// - Top nibble zero, bottom nibble zero → end of line.
/// - Top nibble zero, bottom nibble 0x1–0xF → `bottom + 2` transparent pixels.
/// - Top nibble 0x0 and top of next byte = 0 in a pair → run-length escapes:
///   - `00 00 00` → end of line
///   - `00 0n` (n=1–7) → `n+2` transparent pixels
///   - `00 1n cc` → `n+4` pixels of colour `cc` (4-bit, uses both nibbles of next byte)
fn decode_4bit_rle(data: &[u8], width: u16) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width as usize);
    let mut i = 0; // byte index
    let mut high: bool = true; // which nibble we're reading

    let next_nibble = |idx: &mut usize, high_flag: &mut bool| -> Option<u8> {
        if *idx >= data.len() {
            return None;
        }
        let nibble = if *high_flag {
            (data[*idx] >> 4) & 0x0F
        } else {
            let n = data[*idx] & 0x0F;
            *idx += 1;
            n
        };
        *high_flag = !*high_flag;
        Some(nibble)
    };

    loop {
        let Some(n1) = next_nibble(&mut i, &mut high) else {
            break;
        };

        if n1 != 0 {
            // Non-zero nibble → single pixel of colour n1.
            pixels.push(n1);
        } else {
            // n1 == 0 — run-length escape or end of line.
            let Some(n2) = next_nibble(&mut i, &mut high) else {
                break;
            };

            if n2 == 0 {
                // End of line — align to byte boundary.
                // Note: the assignments to i/high here are intentionally omitted
                // because we immediately break and the values are not used after.
                break;
            } else if (n2 & 0x08) == 0 {
                // n2 in 0x01–0x07: n2+2 transparent pixels.
                let run = n2 as usize + 2;
                for _ in 0..run {
                    pixels.push(0);
                }
            } else {
                // n2 in 0x08–0x0F: next nibble is the run colour; run = (n2 & 0x07) + 4.
                let run = (n2 & 0x07) as usize + 4;
                let Some(colour) = next_nibble(&mut i, &mut high) else {
                    break;
                };
                for _ in 0..run {
                    pixels.push(colour);
                }
            }
        }
    }

    pixels
}

/// Decode an 8-bit per pixel RLE data stream (ETSI EN 300 743 §10.4.4.3).
///
/// Each byte is either a direct pixel value or a run-length escape:
///
/// - Non-zero byte → 1 pixel of that colour index.
/// - `0x00` followed by:
///   - `0x00` → end of line.
///   - `0x01`–`0x7F` → `n` transparent pixels.
///   - `0x80`–`0xFF` → next byte is colour; `n - 0x80 + 3` pixels of that colour.
fn decode_8bit_rle(data: &[u8], width: u16) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width as usize);
    let mut i = 0;

    while i < data.len() {
        let b = data[i];
        i += 1;

        if b != 0 {
            // Single pixel.
            pixels.push(b);
        } else {
            // Escape sequence.
            if i >= data.len() {
                break;
            }
            let b2 = data[i];
            i += 1;

            if b2 == 0 {
                // End of line.
                break;
            } else if b2 <= 0x7F {
                // b2 transparent pixels.
                for _ in 0..b2 {
                    pixels.push(0);
                }
            } else {
                // Run of colour: length = b2 - 0x80 + 3.
                let run = (b2 as usize).saturating_sub(0x80).saturating_add(3);
                if i >= data.len() {
                    break;
                }
                let colour = data[i];
                i += 1;
                for _ in 0..run {
                    pixels.push(colour);
                }
            }
        }
    }

    pixels
}

// ---------------------------------------------------------------------------
// DvbDecoder implementation
// ---------------------------------------------------------------------------

impl DvbDecoder {
    /// Create a new DVB subtitle decoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pages: HashMap::new(),
            regions: HashMap::new(),
            cluts: HashMap::new(),
            objects: HashMap::new(),
            display_definition: None,
            subtitles: Vec::new(),
        }
    }

    /// Decode DVB subtitle segment.
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    pub fn decode_segment(&mut self, data: &[u8], timestamp_ms: i64) -> SubtitleResult<()> {
        if data.len() < 6 {
            return Err(SubtitleError::ParseError(
                "DVB segment too short".to_string(),
            ));
        }

        // Parse segment header
        let sync_byte = data[0];
        if sync_byte != 0x0F {
            return Err(SubtitleError::ParseError(
                "Invalid DVB sync byte".to_string(),
            ));
        }

        let segment_type = data[1];
        let page_id = u16::from_be_bytes([data[2], data[3]]);
        let segment_length = u16::from_be_bytes([data[4], data[5]]) as usize;

        if data.len() < 6 + segment_length {
            return Err(SubtitleError::ParseError(
                "DVB segment data too short".to_string(),
            ));
        }

        let segment_data = &data[6..6 + segment_length];

        match segment_type {
            0x10 => self.decode_page_composition(page_id, segment_data)?,
            0x11 => self.decode_region_composition(segment_data)?,
            0x12 => self.decode_clut_definition(segment_data)?,
            0x13 => self.decode_object_data(segment_data)?,
            0x14 => self.decode_display_definition(segment_data)?,
            0x80 => {
                // End of display set
                self.render_page(page_id, timestamp_ms)?;
            }
            _ => {
                // Unknown segment type - skip
            }
        }

        Ok(())
    }

    /// Decode page composition segment.
    fn decode_page_composition(&mut self, page_id: u16, data: &[u8]) -> SubtitleResult<()> {
        if data.len() < 2 {
            return Ok(());
        }

        let page_timeout = data[0];
        let page_version_state = data[1];
        let page_version = (page_version_state >> 4) & 0x0F;
        let page_state = page_version_state & 0x03;

        let mut regions = Vec::new();
        let mut pos = 2;

        while pos + 6 <= data.len() {
            let region_id = data[pos];
            let horizontal_address = u16::from_be_bytes([data[pos + 2], data[pos + 3]]);
            let vertical_address = u16::from_be_bytes([data[pos + 4], data[pos + 5]]);

            regions.push(RegionReference {
                region_id,
                horizontal_address,
                vertical_address,
            });

            pos += 6;
        }

        let page = PageComposition {
            page_id,
            page_timeout,
            page_version,
            page_state,
            regions,
        };

        self.pages.insert(page_id, page);
        Ok(())
    }

    /// Decode region composition segment.
    fn decode_region_composition(&mut self, data: &[u8]) -> SubtitleResult<()> {
        if data.len() < 10 {
            return Ok(());
        }

        let region_id = data[0];
        let region_version = (data[1] >> 4) & 0x0F;
        let _fill_flag = (data[1] & 0x08) != 0;
        let region_width = u16::from_be_bytes([data[2], data[3]]);
        let region_height = u16::from_be_bytes([data[4], data[5]]);
        let _region_level_of_compatibility = (data[6] >> 5) & 0x07;
        let region_depth = (data[6] >> 2) & 0x07;
        let clut_id = data[7];

        let mut objects = Vec::new();
        let mut pos = 10;

        while pos + 6 <= data.len() {
            let object_id = u16::from_be_bytes([data[pos], data[pos + 1]]);
            let object_type = (data[pos + 2] >> 6) & 0x03;
            let _object_provider_flag = (data[pos + 2] >> 4) & 0x03;
            let horizontal_position = u16::from_be_bytes([data[pos + 2] & 0x0F, data[pos + 3]]);
            let vertical_position = u16::from_be_bytes([data[pos + 4] & 0x0F, data[pos + 5]]);

            let (foreground_pixel_code, background_pixel_code) =
                if object_type == 0x01 || object_type == 0x02 {
                    if pos + 8 <= data.len() {
                        (data[pos + 6], data[pos + 7])
                    } else {
                        (0, 0)
                    }
                } else {
                    (0, 0)
                };

            objects.push(ObjectReference {
                object_id,
                object_type,
                horizontal_position,
                vertical_position,
                foreground_pixel_code,
                background_pixel_code,
            });

            pos += if object_type == 0x01 || object_type == 0x02 {
                8
            } else {
                6
            };
        }

        let region = RegionComposition {
            region_id,
            region_version,
            region_width,
            region_height,
            region_depth,
            clut_id,
            objects,
        };

        self.regions.insert(region_id, region);
        Ok(())
    }

    /// Decode CLUT definition segment (ETSI EN 300 743 §7.2.4).
    ///
    /// Each entry provides Y/Cb/Cr/T values.  If the `full_range_flag` is set
    /// the values occupy 4 bytes (Y, Cr, Cb, T each 8-bit).  Otherwise a
    /// compact 2-byte encoding is used (Y: 6-bit, Cr/Cb: 4-bit each, T: 2-bit).
    fn decode_clut_definition(&mut self, data: &[u8]) -> SubtitleResult<()> {
        if data.len() < 2 {
            return Ok(());
        }

        let clut_id = data[0];
        let clut_version = (data[1] >> 4) & 0x0F;

        let mut entries = HashMap::new();
        let mut pos = 2;

        while pos + 2 <= data.len() {
            let clut_entry_id = data[pos];
            let clut_entry_flags = data[pos + 1];

            let _entry_2bit = (clut_entry_flags & 0x80) != 0;
            let _entry_4bit = (clut_entry_flags & 0x40) != 0;
            let _entry_8bit = (clut_entry_flags & 0x20) != 0;
            let full_range_flag = (clut_entry_flags & 0x01) != 0;

            let (y, cr, cb, t, advance) = if full_range_flag {
                // Full range: 4 bytes (Y, Cr, Cb, T).
                if pos + 6 > data.len() {
                    break;
                }
                (
                    data[pos + 2],
                    data[pos + 3],
                    data[pos + 4],
                    data[pos + 5],
                    6,
                )
            } else {
                // Compact: 2 bytes.
                //   Byte 0: Y[7:2] (6 bits), Cr[3:2] (2 bits)
                //   Byte 1: Cr[1:0] (2 bits), Cb[3:0] (4 bits), T[1:0] (2 bits)
                if pos + 4 > data.len() {
                    break;
                }
                let b0 = data[pos + 2];
                let b1 = data[pos + 3];
                let y_val = b0 & 0xFC; // 6-bit Y scaled to 8-bit
                let cr_val = ((b0 & 0x03) << 6) | ((b1 & 0xC0) >> 2); // 4-bit Cr scaled
                let cb_val = (b1 & 0x3C) << 2; // 4-bit Cb scaled
                let t_val = (b1 & 0x03) * 85; // 2-bit T → 0/85/170/255
                (y_val, cr_val, cb_val, t_val, 4)
            };

            let (r, g, b) = Self::ycrcb_to_rgb(y, cr, cb);
            let alpha = 255 - t; // t=0 → fully opaque, t=255 → transparent

            entries.insert(clut_entry_id, ClutEntry { r, g, b, t: alpha });

            pos += advance;
        }

        let clut = Clut {
            clut_id,
            clut_version,
            entries,
        };

        self.cluts.insert(clut_id, clut);
        Ok(())
    }

    /// Decode object data segment (ETSI EN 300 743 §7.2.5).
    ///
    /// Stores the raw top-field and bottom-field RLE data for later rendering.
    fn decode_object_data(&mut self, data: &[u8]) -> SubtitleResult<()> {
        if data.len() < 3 {
            return Ok(());
        }

        let object_id = u16::from_be_bytes([data[0], data[1]]);
        let object_version = (data[2] >> 4) & 0x0F;
        let coding_method = (data[2] >> 2) & 0x03;
        let non_modifying_color_flag = (data[2] & 0x02) != 0;

        // coding_method == 0x00: run-length encoded pixels.
        // For coding_method == 0x01: character-coded (not bitmap) — skip.
        let (top_field_data, bottom_field_data) = if coding_method == 0x00 && data.len() >= 7 {
            // Bytes 3-4: top field data block address (offset from byte 0 of segment).
            // Bytes 5-6: bottom field data block address.
            let top_addr = u16::from_be_bytes([data[3], data[4]]) as usize;
            let bot_addr = u16::from_be_bytes([data[5], data[6]]) as usize;

            // Both addresses are relative to byte 0 of the segment payload (data).
            let top_end = if bot_addr > top_addr {
                bot_addr
            } else {
                data.len()
            };
            let bot_end = data.len();

            let top = if top_addr < data.len() {
                data[top_addr..top_end.min(data.len())].to_vec()
            } else {
                Vec::new()
            };
            let bot = if bot_addr < data.len() && bot_addr != top_addr {
                data[bot_addr..bot_end].to_vec()
            } else {
                Vec::new()
            };
            (top, bot)
        } else {
            // Store raw remainder for non-RLE objects (e.g., character coded).
            (data[3..].to_vec(), Vec::new())
        };

        let object = ObjectData {
            object_id,
            object_version,
            coding_method,
            non_modifying_color_flag,
            top_field_data,
            bottom_field_data,
        };

        self.objects.insert(object_id, object);
        Ok(())
    }

    /// Decode display definition segment.
    fn decode_display_definition(&mut self, data: &[u8]) -> SubtitleResult<()> {
        if data.len() < 5 {
            return Ok(());
        }

        let width = u16::from_be_bytes([data[1], data[2]]);
        let height = u16::from_be_bytes([data[3], data[4]]);

        self.display_definition = Some(DisplayDefinition { width, height });
        Ok(())
    }

    /// Render a complete display set (page) to one or more `DvbSubtitle` bitmaps.
    ///
    /// For each region referenced by the page, composites all object bitmaps
    /// into the region canvas using the region's CLUT, then stores the result.
    fn render_page(&mut self, page_id: u16, timestamp_ms: i64) -> SubtitleResult<()> {
        let page = match self.pages.get(&page_id).cloned() {
            Some(p) => p,
            None => return Ok(()),
        };

        let mut regions_display = Vec::new();

        for region_ref in &page.regions {
            let region = match self.regions.get(&region_ref.region_id).cloned() {
                Some(r) => r,
                None => continue,
            };

            let width = region.region_width;
            let height = region.region_height;
            let pixel_count = (width as usize) * (height as usize);

            // Allocate RGBA bitmap initialised to transparent black.
            let mut bitmap = vec![0u8; pixel_count * 4];

            // Fetch the CLUT for this region (may be absent — use defaults).
            let empty_clut = Clut {
                clut_id: 0,
                clut_version: 0,
                entries: HashMap::new(),
            };
            let clut = self.cluts.get(&region.clut_id).unwrap_or(&empty_clut);

            // Composite each object into the region bitmap.
            for obj_ref in &region.objects {
                if let Some(obj) = self.objects.get(&obj_ref.object_id).cloned() {
                    // Decode the pixel indices from the RLE stream.
                    let pixel_indices = Self::decode_object_pixels(
                        &obj,
                        region_ref
                            .horizontal_address
                            .saturating_add(obj_ref.horizontal_position),
                        region_ref
                            .vertical_address
                            .saturating_add(obj_ref.vertical_position),
                        width,
                        height,
                        region.region_depth,
                    );

                    // Map indices through the CLUT and blit into the region bitmap.
                    let obj_x = obj_ref.horizontal_position as usize;
                    let obj_y = obj_ref.vertical_position as usize;

                    for (row_idx, row_pixels) in pixel_indices.iter().enumerate() {
                        let dst_y = obj_y + row_idx;
                        if dst_y >= height as usize {
                            break;
                        }
                        for (col_idx, &idx) in row_pixels.iter().enumerate() {
                            let dst_x = obj_x + col_idx;
                            if dst_x >= width as usize {
                                break;
                            }
                            let pixel_pos = (dst_y * width as usize + dst_x) * 4;
                            let entry = clut
                                .entries
                                .get(&idx)
                                .copied()
                                .unwrap_or(ClutEntry::TRANSPARENT);
                            bitmap[pixel_pos] = entry.r;
                            bitmap[pixel_pos + 1] = entry.g;
                            bitmap[pixel_pos + 2] = entry.b;
                            bitmap[pixel_pos + 3] = entry.t;
                        }
                    }
                }
            }

            regions_display.push(RegionDisplay {
                region_id: region.region_id,
                x: region_ref.horizontal_address,
                y: region_ref.vertical_address,
                width,
                height,
                bitmap,
            });
        }

        let timeout_ms = i64::from(page.page_timeout) * 1000;
        let end_time = if timeout_ms > 0 {
            timestamp_ms + timeout_ms
        } else {
            timestamp_ms + 5000 // Default 5 seconds
        };

        self.subtitles.push(DvbSubtitle {
            start_time: timestamp_ms,
            end_time,
            page_id,
            regions: regions_display,
        });

        Ok(())
    }

    /// Decode pixel indices from an `ObjectData` RLE stream.
    ///
    /// Returns a 2-D vector: `result[row][col]` is a CLUT index.
    /// The outer dimension is rows, the inner is pixels within that row.
    fn decode_object_pixels(
        obj: &ObjectData,
        _x: u16,
        _y: u16,
        region_width: u16,
        region_height: u16,
        region_depth: u8,
    ) -> Vec<Vec<u8>> {
        // coding_method 0x00 = RLE bitmap; anything else returns empty.
        if obj.coding_method != 0x00 {
            return Vec::new();
        }

        // DVB interlaces top and bottom fields.  We interleave them to produce
        // progressive rows: even rows from top field, odd rows from bottom field.
        let mut rows: Vec<Vec<u8>> = Vec::new();

        // Determine bits-per-pixel from region_depth.
        // region_depth: 1 = 2bpp, 2 = 4bpp, 3 = 8bpp (ETSI EN 300 743 §7.2.3).
        let bpp = match region_depth {
            1 => 2,
            2 => 4,
            _ => 8,
        };

        // Parse all top-field lines.
        let top_rows = Self::parse_field_lines(&obj.top_field_data, bpp, region_width);
        // Parse all bottom-field lines.
        let bot_rows = Self::parse_field_lines(&obj.bottom_field_data, bpp, region_width);

        // Interleave: top → even rows (0, 2, 4, …), bottom → odd rows (1, 3, 5, …).
        let max_field_rows = top_rows.len().max(bot_rows.len());
        for i in 0..max_field_rows {
            if let Some(row) = top_rows.get(i) {
                rows.push(row.clone());
            }
            if let Some(row) = bot_rows.get(i) {
                rows.push(row.clone());
            }
        }

        // Clamp to region height.
        rows.truncate(region_height as usize);
        rows
    }

    /// Parse one field of RLE data into a list of pixel-index rows.
    fn parse_field_lines(data: &[u8], bpp: u8, width: u16) -> Vec<Vec<u8>> {
        let mut rows: Vec<Vec<u8>> = Vec::new();
        // Each line ends at an end-of-line marker; the data contains multiple
        // consecutive lines.  We slice off each line by scanning for EOL markers.

        // For simplicity, decode the entire field as a flat pixel stream then
        // split by line.  Line splits are implicit from the RLE escape sequences.
        // We process the data byte-by-byte using the appropriate decoder,
        // calling it repeatedly for each row until the data is exhausted.

        // We wrap each per-line decoder in a loop that advances a cursor.
        // Because the decoder functions operate on slices without cursor state,
        // we re-implement a cursor-based decoder inline for each bpp.

        match bpp {
            2 => {
                // 2-bit: bit-level; every row ends at a 0x00 switch code → 0x00 sub.
                // Feed the entire field to the single-row decoder and collect rows.
                // The simple approach: run the whole decoder which stops at first EOL.
                // To get all rows, call it repeatedly on remaining data.
                let mut offset = 0;
                let total_bits = data.len() * 8;
                loop {
                    if offset >= total_bits {
                        break;
                    }
                    let (row_pixels, consumed_bits) = decode_2bit_rle_row(data, offset, width);
                    if consumed_bits == 0 {
                        break;
                    }
                    rows.push(row_pixels);
                    offset += consumed_bits;
                }
            }
            4 => {
                let mut pos = 0;
                let mut high = true;
                loop {
                    if pos >= data.len() {
                        break;
                    }
                    let (row_pixels, new_pos, new_high) =
                        decode_4bit_rle_row(data, pos, high, width);
                    if new_pos == pos && new_high == high {
                        break;
                    }
                    rows.push(row_pixels);
                    pos = new_pos;
                    high = new_high;
                }
            }
            _ => {
                // 8-bit
                let mut pos = 0;
                loop {
                    if pos >= data.len() {
                        break;
                    }
                    let (row_pixels, new_pos) = decode_8bit_rle_row(data, pos, width);
                    if new_pos == pos {
                        break;
                    }
                    rows.push(row_pixels);
                    pos = new_pos;
                }
            }
        }

        rows
    }

    /// Convert YCbCr (BT.601) to RGB.
    ///
    /// Uses the standard BT.601 matrix:
    /// ```text
    /// R = Y + 1.402 × (Cr − 128)
    /// G = Y − 0.344136 × (Cb − 128) − 0.714136 × (Cr − 128)
    /// B = Y + 1.772 × (Cb − 128)
    /// ```
    fn ycrcb_to_rgb(y: u8, cr: u8, cb: u8) -> (u8, u8, u8) {
        let y_f = f32::from(y);
        let cr_f = f32::from(cr) - 128.0;
        let cb_f = f32::from(cb) - 128.0;

        let r = (y_f + 1.402 * cr_f).clamp(0.0, 255.0) as u8;
        let g = (y_f - 0.344136 * cb_f - 0.714136 * cr_f).clamp(0.0, 255.0) as u8;
        let b = (y_f + 1.772 * cb_f).clamp(0.0, 255.0) as u8;

        (r, g, b)
    }

    /// Get all decoded subtitles.
    #[must_use]
    pub fn take_subtitles(&mut self) -> Vec<DvbSubtitle> {
        std::mem::take(&mut self.subtitles)
    }

    /// Finalize decoding and return all subtitles.
    #[must_use]
    pub fn finalize(self) -> Vec<DvbSubtitle> {
        self.subtitles
    }
}

impl Default for DvbDecoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Row-level RLE decoders (return rows + consumed byte/bit counts)
// ---------------------------------------------------------------------------

/// Decode one 2-bit RLE row starting at `bit_offset`.
///
/// Returns `(pixels, bits_consumed)`.  `bits_consumed == 0` signals no
/// progress (end of data).
fn decode_2bit_rle_row(data: &[u8], start_bit: usize, width: u16) -> (Vec<u8>, usize) {
    let mut pixels: Vec<u8> = Vec::with_capacity(width as usize);
    let total_bits = data.len() * 8;
    let mut bit_offset = start_bit;

    let read_bits = |offset: usize, count: usize| -> Option<u8> {
        if offset + count > total_bits {
            return None;
        }
        let mut val: u8 = 0;
        for i in 0..count {
            let byte_idx = (offset + i) / 8;
            let bit_idx = 7 - ((offset + i) % 8);
            val = (val << 1) | ((data[byte_idx] >> bit_idx) & 1);
        }
        Some(val)
    };

    loop {
        let Some(code) = read_bits(bit_offset, 2) else {
            break;
        };
        bit_offset += 2;

        if code != 0 {
            pixels.push(code);
        } else {
            let Some(sub) = read_bits(bit_offset, 2) else {
                break;
            };
            bit_offset += 2;
            match sub {
                0b00 => break, // end of line
                0b01 => {
                    pixels.push(0);
                    pixels.push(0);
                }
                0b10 => {
                    let Some(run_bits) = read_bits(bit_offset, 2) else {
                        break;
                    };
                    bit_offset += 2;
                    let run = run_bits as usize + 3;
                    for _ in 0..run {
                        pixels.push(0);
                    }
                }
                0b11 => {
                    let Some(colour) = read_bits(bit_offset, 2) else {
                        break;
                    };
                    bit_offset += 2;
                    let Some(run_bits) = read_bits(bit_offset, 2) else {
                        break;
                    };
                    bit_offset += 2;
                    let run = run_bits as usize + 4;
                    for _ in 0..run {
                        pixels.push(colour);
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    let consumed = bit_offset - start_bit;
    (pixels, consumed)
}

/// Decode one 4-bit RLE row starting at byte `pos` / nibble `high`.
///
/// Returns `(pixels, new_pos, new_high)`.  If `new_pos == pos && new_high == high`,
/// no progress was made.
fn decode_4bit_rle_row(
    data: &[u8],
    start_pos: usize,
    start_high: bool,
    width: u16,
) -> (Vec<u8>, usize, bool) {
    let mut pixels: Vec<u8> = Vec::with_capacity(width as usize);
    let mut pos = start_pos;
    let mut high = start_high;

    let next_nibble = |p: &mut usize, h: &mut bool| -> Option<u8> {
        if *p >= data.len() {
            return None;
        }
        let nibble = if *h {
            (data[*p] >> 4) & 0x0F
        } else {
            let n = data[*p] & 0x0F;
            *p += 1;
            n
        };
        *h = !*h;
        Some(nibble)
    };

    loop {
        let Some(n1) = next_nibble(&mut pos, &mut high) else {
            break;
        };

        if n1 != 0 {
            pixels.push(n1);
        } else {
            let Some(n2) = next_nibble(&mut pos, &mut high) else {
                break;
            };

            if n2 == 0 {
                // End of line — byte-align.
                if !high {
                    pos += 1;
                    high = true;
                }
                break;
            } else if (n2 & 0x08) == 0 {
                let run = n2 as usize + 2;
                for _ in 0..run {
                    pixels.push(0);
                }
            } else {
                let run = (n2 & 0x07) as usize + 4;
                let Some(colour) = next_nibble(&mut pos, &mut high) else {
                    break;
                };
                for _ in 0..run {
                    pixels.push(colour);
                }
            }
        }
    }

    (pixels, pos, high)
}

/// Decode one 8-bit RLE row starting at byte `pos`.
///
/// Returns `(pixels, new_pos)`.  If `new_pos == pos`, no progress was made.
fn decode_8bit_rle_row(data: &[u8], start_pos: usize, width: u16) -> (Vec<u8>, usize) {
    let mut pixels: Vec<u8> = Vec::with_capacity(width as usize);
    let mut i = start_pos;

    while i < data.len() {
        let b = data[i];
        i += 1;

        if b != 0 {
            pixels.push(b);
        } else {
            if i >= data.len() {
                break;
            }
            let b2 = data[i];
            i += 1;

            if b2 == 0 {
                break; // end of line
            } else if b2 <= 0x7F {
                for _ in 0..b2 {
                    pixels.push(0);
                }
            } else {
                let run = (b2 as usize).saturating_sub(0x80).saturating_add(3);
                if i >= data.len() {
                    break;
                }
                let colour = data[i];
                i += 1;
                for _ in 0..run {
                    pixels.push(colour);
                }
            }
        }
    }

    (pixels, i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ycrcb_conversion_white() {
        // Y=255, Cr=128, Cb=128 → R=255, G=255, B=255
        let (r, g, b) = DvbDecoder::ycrcb_to_rgb(255, 128, 128);
        assert_eq!(r, 255);
        assert_eq!(g, 255);
        assert_eq!(b, 255);
    }

    #[test]
    fn test_ycrcb_conversion_black() {
        let (r, g, b) = DvbDecoder::ycrcb_to_rgb(0, 128, 128);
        assert_eq!(r, 0);
        assert_eq!(g, 0);
        assert_eq!(b, 0);
    }

    #[test]
    fn test_decoder_creation() {
        let decoder = DvbDecoder::new();
        assert_eq!(decoder.pages.len(), 0);
        assert_eq!(decoder.regions.len(), 0);
    }

    // --- 8-bit RLE ---

    #[test]
    fn test_8bit_rle_single_pixel() {
        let data = [0x05]; // colour 5
        let pixels = decode_8bit_rle(&data, 8);
        assert_eq!(pixels, vec![5]);
    }

    #[test]
    fn test_8bit_rle_transparent_run() {
        // 0x00 0x03 → 3 transparent pixels
        let data = [0x00, 0x03];
        let pixels = decode_8bit_rle(&data, 8);
        assert_eq!(pixels, vec![0, 0, 0]);
    }

    #[test]
    fn test_8bit_rle_colour_run() {
        // 0x00 0x84 0x07 → (0x84 - 0x80 + 3) = 7 pixels of colour 7
        let data = [0x00, 0x84, 0x07];
        let pixels = decode_8bit_rle(&data, 16);
        assert_eq!(pixels.len(), 7);
        assert!(pixels.iter().all(|&p| p == 7));
    }

    #[test]
    fn test_8bit_rle_end_of_line() {
        // 0x01 0x00 0x00 (EOL) 0x02 — should stop at EOL
        let data = [0x01, 0x00, 0x00, 0x02];
        let pixels = decode_8bit_rle(&data, 8);
        assert_eq!(pixels, vec![1]);
    }

    // --- 4-bit RLE ---

    #[test]
    fn test_4bit_rle_single_pixels() {
        // Byte 0xAB → nibbles 0xA and 0xB → two pixels (colours 10 and 11)
        let (pixels, _p, _h) = decode_4bit_rle_row(&[0xAB], 0, true, 8);
        assert_eq!(pixels, vec![10, 11]);
    }

    #[test]
    fn test_4bit_rle_transparent_run() {
        // Nibbles: 0x0 (escape), 0x3 (run=3+2=5 transparent), then EOL nibbles 0x0 0x0
        // Encoded: 0x03 0x00
        let (pixels, _, _) = decode_4bit_rle_row(&[0x03, 0x00], 0, true, 16);
        assert_eq!(pixels, vec![0u8; 5]);
    }

    #[test]
    fn test_4bit_rle_colour_run() {
        // Nibbles: 0x0 (escape), 0x9 (0x08|0x01 → run=1+4=5), 0x7 (colour 7), EOL: 0x0 0x0
        // Encoded in bytes:
        //   nibble 0: 0x0 → top of 0x09 = 0x0
        //   nibble 1: 0x9 → bot of 0x09 = 0x9
        //   nibble 2: 0x7 → top of 0x70 = 0x7
        //   nibble 3: 0x0 → bot of 0x70 = 0x0 (EOL part 1)
        //   nibble 4: 0x0 → top of 0x00 = 0x0 (EOL part 2)
        let data = [0x09, 0x70, 0x00];
        let (pixels, _, _) = decode_4bit_rle_row(&data, 0, true, 16);
        assert_eq!(pixels.len(), 5);
        assert!(pixels.iter().all(|&p| p == 7));
    }

    // --- 2-bit RLE ---

    #[test]
    fn test_2bit_rle_single_pixel() {
        // Bits: 01 → colour 1, then end (00 00 sub → EOL)
        // 0x40 = 0100 0000 → nibble: 01 = colour 1, 00 = switch, 00 = EOL (but we only have 8 bits)
        let data = [0x40]; // bits: 0100 0000
        let (pixels, consumed) = decode_2bit_rle_row(&data, 0, 8);
        // First 2 bits: 01 → push 1
        // Next 2 bits: 00 → switch code; next 2 bits: 00 → EOL
        assert_eq!(pixels, vec![1]);
        assert_eq!(consumed, 6); // consumed 6 bits
    }

    #[test]
    fn test_2bit_rle_transparent_pair() {
        // Bits: 00 01 → switch(00) then sub(01) = 2 transparent pixels; 00 00 = EOL
        // 0x10 = 0001 0000 → bits: 00 01 00 00
        let data = [0x10];
        let (pixels, _) = decode_2bit_rle_row(&data, 0, 8);
        assert_eq!(pixels, vec![0, 0]);
    }

    // --- Full decode_segment smoke test ---

    #[test]
    fn test_decode_segment_invalid_sync() {
        let mut dec = DvbDecoder::new();
        let bad = [0xFF, 0x10, 0x00, 0x01, 0x00, 0x00];
        let result = dec.decode_segment(&bad, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_segment_page_composition() {
        let mut dec = DvbDecoder::new();
        // sync=0x0F, type=0x10 (page), page_id=0x0001, length=0x0002
        // page_timeout=5, page_version_state=0x10
        let data = [0x0F, 0x10, 0x00, 0x01, 0x00, 0x02, 0x05, 0x10];
        dec.decode_segment(&data, 1000).expect("decode");
        assert!(dec.pages.contains_key(&1));
    }
}
