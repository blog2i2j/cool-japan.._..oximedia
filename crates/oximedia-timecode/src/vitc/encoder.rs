//! VITC Encoder - Generate timecode for video scan lines
//!
//! This module implements a complete VITC encoder that:
//! - Encodes timecode and user bits to VITC bit patterns
//! - Generates pixel data for embedding in video lines
//! - Calculates and inserts CRC checksums
//! - Handles field synchronization
//! - Supports all standard video formats

use super::constants::*;
use super::{VideoStandard, VitcWriterConfig};
use crate::{FrameRate, Timecode, TimecodeError};

/// VITC encoder
pub struct VitcEncoder {
    /// Configuration
    #[allow(dead_code)]
    config: VitcWriterConfig,
    /// Current field being encoded
    current_field: u8,
}

impl VitcEncoder {
    /// Create a new VITC encoder
    pub fn new(config: VitcWriterConfig) -> Self {
        VitcEncoder {
            config,
            current_field: 1,
        }
    }

    /// Encode a timecode to pixel data for a video line
    pub fn encode_line(
        &mut self,
        timecode: &Timecode,
        field: u8,
    ) -> Result<Vec<u8>, TimecodeError> {
        // Create bit array
        let bits = self.timecode_to_bits(timecode, field)?;

        // Convert bits to pixels
        let pixels = self.bits_to_pixels(&bits);

        Ok(pixels)
    }

    /// Convert timecode to VITC bit array
    fn timecode_to_bits(
        &self,
        timecode: &Timecode,
        field: u8,
    ) -> Result<[bool; BITS_PER_LINE], TimecodeError> {
        let mut bits = [false; BITS_PER_LINE];

        // Start sync bits (0-1): 11 (white-white)
        bits[0] = true;
        bits[1] = true;

        // Data bits start at position 2
        let mut data_bits = [false; DATA_BITS];

        // Decompose timecode
        let frame_units = timecode.frames % 10;
        let frame_tens = timecode.frames / 10;
        let second_units = timecode.seconds % 10;
        let second_tens = timecode.seconds / 10;
        let minute_units = timecode.minutes % 10;
        let minute_tens = timecode.minutes / 10;
        let hour_units = timecode.hours % 10;
        let hour_tens = timecode.hours / 10;

        // Encode frame units (bits 0-3)
        self.encode_bcd(&mut data_bits, 0, frame_units);

        // User bits 1 (bits 4-7)
        self.encode_nibble(&mut data_bits, 4, (timecode.user_bits & 0xF) as u8);

        // Frame tens (bits 8-9)
        self.encode_bcd(&mut data_bits, 8, frame_tens);

        // Drop frame flag (bit 10)
        data_bits[10] = timecode.frame_rate.drop_frame;

        // Color frame flag (bit 11)
        data_bits[11] = false;

        // User bits 2 (bits 12-15)
        self.encode_nibble(&mut data_bits, 12, ((timecode.user_bits >> 4) & 0xF) as u8);

        // Second units (bits 16-19)
        self.encode_bcd(&mut data_bits, 16, second_units);

        // User bits 3 (bits 20-23)
        self.encode_nibble(&mut data_bits, 20, ((timecode.user_bits >> 8) & 0xF) as u8);

        // Second tens (bits 24-26)
        self.encode_bcd(&mut data_bits, 24, second_tens);

        // Field mark (bit 27) - 0 for field 1, 1 for field 2
        data_bits[27] = field == 2;

        // User bits 4 (bits 28-31)
        self.encode_nibble(&mut data_bits, 28, ((timecode.user_bits >> 12) & 0xF) as u8);

        // Minute units (bits 32-35)
        self.encode_bcd(&mut data_bits, 32, minute_units);

        // User bits 5 (bits 36-39)
        self.encode_nibble(&mut data_bits, 36, ((timecode.user_bits >> 16) & 0xF) as u8);

        // Minute tens (bits 40-42)
        self.encode_bcd(&mut data_bits, 40, minute_tens);

        // Binary group flag (bit 43)
        data_bits[43] = false;

        // User bits 6 (bits 44-47)
        self.encode_nibble(&mut data_bits, 44, ((timecode.user_bits >> 20) & 0xF) as u8);

        // Hour units (bits 48-51)
        self.encode_bcd(&mut data_bits, 48, hour_units);

        // User bits 7 (bits 52-55)
        self.encode_nibble(&mut data_bits, 52, ((timecode.user_bits >> 24) & 0xF) as u8);

        // Hour tens (bits 56-57)
        self.encode_bcd(&mut data_bits, 56, hour_tens);

        // Reserved bit (58)
        data_bits[58] = false;

        // User bits 8 (bits 59-73)
        self.encode_nibble(&mut data_bits, 59, ((timecode.user_bits >> 28) & 0xF) as u8);

        // Calculate and insert CRC (bits 74-81)
        let crc = self.calculate_crc(&data_bits[0..72]);
        for i in 0..8 {
            data_bits[74 + i] = (crc & (1 << i)) != 0;
        }

        // Copy data bits to main bit array
        bits[SYNC_START_BITS..(DATA_BITS + SYNC_START_BITS)]
            .copy_from_slice(&data_bits[..DATA_BITS]);

        // End sync bits (84-89): 001111 (black-black-white-white-white-white)
        bits[84] = false;
        bits[85] = false;
        bits[86] = true;
        bits[87] = true;
        bits[88] = true;
        bits[89] = true;

        Ok(bits)
    }

    /// Encode a BCD digit
    fn encode_bcd(&self, bits: &mut [bool; DATA_BITS], start: usize, value: u8) {
        for i in 0..4 {
            if start + i < DATA_BITS {
                bits[start + i] = (value & (1 << i)) != 0;
            }
        }
    }

    /// Encode a 4-bit nibble
    fn encode_nibble(&self, bits: &mut [bool; DATA_BITS], start: usize, value: u8) {
        for i in 0..4 {
            if start + i < DATA_BITS {
                bits[start + i] = (value & (1 << i)) != 0;
            }
        }
    }

    /// Calculate CRC for VITC
    fn calculate_crc(&self, bits: &[bool]) -> u8 {
        let mut crc = 0u8;

        for &bit in bits {
            let feedback = ((crc & 0x80) != 0) ^ bit;
            crc <<= 1;
            if feedback {
                crc ^= 0x07; // Polynomial: x^8 + x^2 + x^1 + x^0
            }
        }

        crc
    }

    /// Convert bits to pixel data
    fn bits_to_pixels(&self, bits: &[bool; BITS_PER_LINE]) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(BITS_PER_LINE * PIXELS_PER_BIT);

        for &bit in bits {
            let level = if bit { WHITE_LEVEL } else { BLACK_LEVEL };

            // Each bit is PIXELS_PER_BIT pixels wide
            for _ in 0..PIXELS_PER_BIT {
                pixels.push(level);
            }
        }

        pixels
    }

    /// Reset encoder state
    pub fn reset(&mut self) {
        self.current_field = 1;
    }

    /// Set current field
    pub fn set_field(&mut self, field: u8) {
        self.current_field = field;
    }

    /// Get current field
    pub fn field(&self) -> u8 {
        self.current_field
    }
}

/// Multi-line VITC writer for redundancy
pub struct MultiLineVitcWriter {
    /// Encoder
    encoder: VitcEncoder,
    /// Lines to write
    lines: Vec<u16>,
}

impl MultiLineVitcWriter {
    /// Create a multi-line writer
    pub fn new(config: VitcWriterConfig) -> Self {
        let lines = config.scan_lines.clone();
        MultiLineVitcWriter {
            encoder: VitcEncoder::new(config),
            lines,
        }
    }

    /// Encode timecode for all configured lines
    pub fn encode_all_lines(
        &mut self,
        timecode: &Timecode,
        field: u8,
    ) -> Result<Vec<(u16, Vec<u8>)>, TimecodeError> {
        let mut results = Vec::new();

        for &line in &self.lines {
            let pixels = self.encoder.encode_line(timecode, field)?;
            results.push((line, pixels));
        }

        Ok(results)
    }
}

/// VITC line buffer for video frame integration
pub struct VitcLineBuffer {
    /// Video standard
    #[allow(dead_code)]
    video_standard: VideoStandard,
    /// Frame buffer (stores VITC lines only)
    lines: Vec<(u16, u8, Vec<u8>)>, // (line_number, field, pixels)
}

impl VitcLineBuffer {
    /// Create a new line buffer
    pub fn new(video_standard: VideoStandard) -> Self {
        VitcLineBuffer {
            video_standard,
            lines: Vec::new(),
        }
    }

    /// Add a VITC line to the buffer
    pub fn add_line(&mut self, line_number: u16, field: u8, pixels: Vec<u8>) {
        // Remove existing line with same line number and field
        self.lines
            .retain(|(ln, f, _)| !(*ln == line_number && *f == field));

        self.lines.push((line_number, field, pixels));
    }

    /// Get all lines for a specific field
    pub fn get_field_lines(&self, field: u8) -> Vec<(u16, &[u8])> {
        self.lines
            .iter()
            .filter(|(_, f, _)| *f == field)
            .map(|(ln, _, pixels)| (*ln, pixels.as_slice()))
            .collect()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.lines.clear();
    }

    /// Get total lines
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }
}

/// Pixel level adjustment for different video standards
pub struct PixelLevelAdjuster {
    /// Black level
    black_level: u8,
    /// White level
    white_level: u8,
}

impl PixelLevelAdjuster {
    /// Create adjuster for video standard
    pub fn new(video_standard: VideoStandard) -> Self {
        match video_standard {
            VideoStandard::Ntsc => PixelLevelAdjuster {
                black_level: 16,
                white_level: 235,
            },
            VideoStandard::Pal => PixelLevelAdjuster {
                black_level: 16,
                white_level: 235,
            },
        }
    }

    /// Create with custom levels
    pub fn with_levels(black_level: u8, white_level: u8) -> Self {
        PixelLevelAdjuster {
            black_level,
            white_level,
        }
    }

    /// Adjust bit to pixel level
    pub fn bit_to_pixel(&self, bit: bool) -> u8 {
        if bit {
            self.white_level
        } else {
            self.black_level
        }
    }

    /// Get black level
    pub fn black_level(&self) -> u8 {
        self.black_level
    }

    /// Get white level
    pub fn white_level(&self) -> u8 {
        self.white_level
    }
}

/// Rise time shaper for cleaner edges
pub struct RiseTimeShaper {
    /// Rise time in pixels
    rise_time_pixels: usize,
}

impl RiseTimeShaper {
    /// Create with rise time
    pub fn new(rise_time_pixels: usize) -> Self {
        RiseTimeShaper {
            rise_time_pixels: rise_time_pixels.max(1),
        }
    }

    /// Shape pixel transitions
    pub fn shape_pixels(&self, pixels: &[u8]) -> Vec<u8> {
        let mut shaped = Vec::with_capacity(pixels.len());

        if pixels.is_empty() {
            return shaped;
        }

        shaped.push(pixels[0]);

        for i in 1..pixels.len() {
            let current = pixels[i];
            let prev = pixels[i - 1];

            if current != prev {
                // Transition detected - apply shaping
                let diff = current as i16 - prev as i16;
                for j in 0..self.rise_time_pixels.min(pixels.len() - i) {
                    let progress = (j + 1) as f32 / self.rise_time_pixels as f32;
                    let value = prev as f32 + diff as f32 * progress;
                    shaped.push(value as u8);
                }
                // Skip the shaped pixels
                for _ in 0..self.rise_time_pixels.min(pixels.len() - i) {
                    if i < pixels.len() {
                        shaped.push(current);
                    }
                }
            } else {
                shaped.push(current);
            }
        }

        shaped.truncate(pixels.len());
        shaped
    }
}

/// Blanking level inserter
pub struct BlankingInserter {
    /// Blanking level
    blanking_level: u8,
}

impl BlankingInserter {
    /// Create with blanking level
    pub fn new(blanking_level: u8) -> Self {
        BlankingInserter { blanking_level }
    }

    /// Insert blanking before and after VITC
    pub fn insert_blanking(
        &self,
        vitc_pixels: &[u8],
        total_width: usize,
        start_offset: usize,
    ) -> Vec<u8> {
        let mut full_line = vec![self.blanking_level; total_width];

        // Copy VITC pixels at the specified offset
        let end_offset = (start_offset + vitc_pixels.len()).min(total_width);
        for (i, &pixel) in vitc_pixels.iter().enumerate() {
            if start_offset + i < end_offset {
                full_line[start_offset + i] = pixel;
            }
        }

        full_line
    }
}

/// VITC frame generator for continuous encoding
pub struct VitcFrameGenerator {
    /// Writer
    writer: MultiLineVitcWriter,
    /// Current timecode
    current_timecode: Option<Timecode>,
    /// Frame rate
    #[allow(dead_code)]
    frame_rate: FrameRate,
}

impl VitcFrameGenerator {
    /// Create a new frame generator
    pub fn new(config: VitcWriterConfig) -> Self {
        let frame_rate = config.frame_rate;
        VitcFrameGenerator {
            writer: MultiLineVitcWriter::new(config),
            current_timecode: None,
            frame_rate,
        }
    }

    /// Set starting timecode
    pub fn set_timecode(&mut self, timecode: Timecode) {
        self.current_timecode = Some(timecode);
    }

    /// Generate VITC for next frame
    pub fn generate_frame(&mut self) -> Result<Vec<(u16, u8, Vec<u8>)>, TimecodeError> {
        if let Some(ref mut tc) = self.current_timecode {
            let mut results = Vec::new();

            // Generate for field 1
            let field1_lines = self.writer.encode_all_lines(tc, 1)?;
            for (line, pixels) in field1_lines {
                results.push((line, 1, pixels));
            }

            // Generate for field 2
            let field2_lines = self.writer.encode_all_lines(tc, 2)?;
            for (line, pixels) in field2_lines {
                results.push((line, 2, pixels));
            }

            // Increment timecode
            tc.increment()?;

            Ok(results)
        } else {
            Err(TimecodeError::InvalidConfiguration)
        }
    }

    /// Get current timecode
    pub fn current_timecode(&self) -> Option<&Timecode> {
        self.current_timecode.as_ref()
    }
}

/// User bits helpers for VITC
pub struct VitcUserBitsHelper;

impl VitcUserBitsHelper {
    /// Validate user bits for VITC
    pub fn validate_user_bits(user_bits: u32) -> bool {
        // User bits are 32 bits, all values are valid
        let _ = user_bits;
        true
    }

    /// Extract user bits group
    pub fn extract_group(user_bits: u32, group: u8) -> u8 {
        let shift = (group as u32) * 4;
        ((user_bits >> shift) & 0xF) as u8
    }

    /// Set user bits group
    pub fn set_group(user_bits: u32, group: u8, value: u8) -> u32 {
        let shift = (group as u32) * 4;
        let mask = !(0xF << shift);
        (user_bits & mask) | ((value as u32 & 0xF) << shift)
    }
}

/// Pixel pattern validator
pub struct PixelPatternValidator;

impl PixelPatternValidator {
    /// Validate that pixel pattern is suitable for VITC
    pub fn validate_pattern(pixels: &[u8]) -> bool {
        if pixels.len() < BITS_PER_LINE * PIXELS_PER_BIT {
            return false;
        }

        // Check that pixels are within valid range
        for &pixel in pixels {
            if !(16..=235).contains(&pixel) {
                return false;
            }
        }

        true
    }

    /// Check sync pattern
    pub fn check_sync_pattern(pixels: &[u8]) -> bool {
        if pixels.len() < 4 {
            return false;
        }

        // First 4 pixels should be white (start sync)
        pixels[0] > 200 && pixels[1] > 200 && pixels[2] > 200 && pixels[3] > 200
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let config = VitcWriterConfig::default();
        let encoder = VitcEncoder::new(config);
        assert_eq!(encoder.field(), 1);
    }

    #[test]
    fn test_encode_line() {
        let config = VitcWriterConfig::default();
        let mut encoder = VitcEncoder::new(config);

        let timecode = Timecode::new(1, 2, 3, 4, FrameRate::Fps25).expect("valid timecode");
        let pixels = encoder
            .encode_line(&timecode, 1)
            .expect("encode should succeed");

        assert_eq!(pixels.len(), BITS_PER_LINE * PIXELS_PER_BIT);
    }

    #[test]
    fn test_crc_calculation() {
        let config = VitcWriterConfig::default();
        let encoder = VitcEncoder::new(config);

        let bits = [false; 72];
        let crc = encoder.calculate_crc(&bits);
        assert_eq!(crc, 0);
    }

    #[test]
    fn test_pixel_level_adjuster() {
        let adjuster = PixelLevelAdjuster::new(VideoStandard::Pal);
        assert_eq!(adjuster.bit_to_pixel(false), 16);
        assert_eq!(adjuster.bit_to_pixel(true), 235);
    }

    #[test]
    fn test_user_bits_helper() {
        let user_bits = 0x12345678u32;
        assert_eq!(VitcUserBitsHelper::extract_group(user_bits, 0), 0x8);
        assert_eq!(VitcUserBitsHelper::extract_group(user_bits, 1), 0x7);

        let modified = VitcUserBitsHelper::set_group(user_bits, 0, 0xA);
        assert_eq!(VitcUserBitsHelper::extract_group(modified, 0), 0xA);
    }

    #[test]
    fn test_line_buffer() {
        let mut buffer = VitcLineBuffer::new(VideoStandard::Pal);
        buffer.add_line(19, 1, vec![16; 180]);
        buffer.add_line(21, 1, vec![16; 180]);

        assert_eq!(buffer.line_count(), 2);

        let field1_lines = buffer.get_field_lines(1);
        assert_eq!(field1_lines.len(), 2);
    }
}
