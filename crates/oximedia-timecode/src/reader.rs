//! Timecode reader/decoder module.
//!
//! Provides VITC parsing, LTC decode approximation, and timecode validation.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use crate::{FrameRate, FrameRateInfo, Timecode, TimecodeError};

/// Result of a VITC line parse attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VitcParseResult {
    /// Parsed timecode, if successful.
    pub timecode: Option<Timecode>,
    /// Whether the CRC matched.
    pub crc_ok: bool,
    /// Raw nibbles extracted from the line.
    pub raw_nibbles: Vec<u8>,
}

/// VITC (Vertical Interval Timecode) parser.
///
/// Parses timecode data embedded in the vertical blanking interval of a video signal.
#[derive(Debug, Clone)]
pub struct VitcParser {
    frame_rate: FrameRate,
    clock_period_samples: usize,
    sync_threshold: f32,
}

impl VitcParser {
    /// Create a new VITC parser for the given frame rate and pixel clock.
    pub fn new(frame_rate: FrameRate, pixels_per_line: usize) -> Self {
        // VITC uses 90 bits per line: 2 sync bits + 8 groups of 9 bits + 2 sync bits
        let clock_period_samples = pixels_per_line / 90;
        Self {
            frame_rate,
            clock_period_samples: clock_period_samples.max(1),
            sync_threshold: 0.5,
        }
    }

    /// Set the sync detection threshold (0.0–1.0).
    pub fn set_sync_threshold(&mut self, threshold: f32) {
        self.sync_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Parse a VITC scan line represented as normalized pixel values (0.0–1.0).
    pub fn parse_line(&self, pixels: &[f32]) -> VitcParseResult {
        if pixels.len() < 90 {
            return VitcParseResult {
                timecode: None,
                crc_ok: false,
                raw_nibbles: Vec::new(),
            };
        }

        let bits = self.sample_bits(pixels);
        if bits.len() < 90 {
            return VitcParseResult {
                timecode: None,
                crc_ok: false,
                raw_nibbles: Vec::new(),
            };
        }

        // Check sync pattern: bits 0-1 should be 1,0 and bits 88-89 should be 0,1
        let sync_ok = bits[0] == 1 && bits[1] == 0 && bits[88] == 0 && bits[89] == 1;
        if !sync_ok {
            return VitcParseResult {
                timecode: None,
                crc_ok: false,
                raw_nibbles: Vec::new(),
            };
        }

        // Extract 8 groups of 9 bits (bits 2..89)
        let mut nibbles = Vec::with_capacity(16);
        let mut crc_accum: u8 = 0;
        for group in 0..8 {
            let base = 2 + group * 9;
            let lo_nibble =
                bits[base] | (bits[base + 1] << 1) | (bits[base + 2] << 2) | (bits[base + 3] << 3);
            let hi_nibble = bits[base + 4]
                | (bits[base + 5] << 1)
                | (bits[base + 6] << 2)
                | (bits[base + 7] << 3);
            // bit 8 of each group is the CRC bit for that group
            let _group_crc = bits[base + 8];
            nibbles.push(lo_nibble);
            nibbles.push(hi_nibble);
            crc_accum ^= lo_nibble ^ hi_nibble;
        }

        let crc_ok = crc_accum == 0;

        // Decode timecode from nibbles
        // Group 0: frames units/tens
        // Group 1: seconds units/tens
        // Group 2: minutes units/tens
        // Group 3: hours units/tens
        let frames_units = nibbles[0] & 0x0F;
        let frames_tens = nibbles[1] & 0x03;
        let seconds_units = nibbles[2] & 0x0F;
        let seconds_tens = nibbles[3] & 0x07;
        let minutes_units = nibbles[4] & 0x0F;
        let minutes_tens = nibbles[5] & 0x07;
        let hours_units = nibbles[6] & 0x0F;
        let hours_tens = nibbles[7] & 0x03;

        let frames = frames_tens * 10 + frames_units;
        let seconds = seconds_tens * 10 + seconds_units;
        let minutes = minutes_tens * 10 + minutes_units;
        let hours = hours_tens * 10 + hours_units;

        let timecode = Timecode::new(hours, minutes, seconds, frames, self.frame_rate).ok();

        VitcParseResult {
            timecode,
            crc_ok,
            raw_nibbles: nibbles,
        }
    }

    /// Sample bits from pixel array at clock-period intervals.
    fn sample_bits(&self, pixels: &[f32]) -> Vec<u8> {
        let period = self.clock_period_samples.max(1);
        let count = pixels.len() / period;
        pixels
            .chunks(period)
            .take(count)
            .map(|chunk| {
                let avg = chunk.iter().sum::<f32>() / chunk.len() as f32;
                if avg >= self.sync_threshold {
                    1u8
                } else {
                    0u8
                }
            })
            .collect()
    }
}

/// LTC (Linear Timecode) decoder state.
#[derive(Debug, Clone)]
pub struct LtcDecoder {
    frame_rate: FrameRate,
    sample_rate: u32,
    /// Internal ring buffer for edge detection.
    buffer: Vec<f32>,
    buffer_pos: usize,
    half_period_samples: usize,
    decoded_bits: Vec<u8>,
    last_sample: f32,
    bit_count: usize,
}

impl LtcDecoder {
    /// Create a new LTC decoder.
    pub fn new(frame_rate: FrameRate, sample_rate: u32) -> Self {
        let fps = frame_rate.frames_per_second() as f64;
        let bits_per_frame = 80usize;
        let samples_per_frame = sample_rate as f64 / fps;
        let samples_per_bit = samples_per_frame / bits_per_frame as f64;
        let half_period = (samples_per_bit / 2.0).round() as usize;

        Self {
            frame_rate,
            sample_rate,
            buffer: vec![0.0; half_period * 2],
            buffer_pos: 0,
            half_period_samples: half_period.max(1),
            decoded_bits: Vec::with_capacity(80),
            last_sample: 0.0,
            bit_count: 0,
        }
    }

    /// Feed audio samples and return any decoded timecodes.
    pub fn feed(&mut self, samples: &[f32]) -> Vec<Timecode> {
        let mut results = Vec::new();

        for &s in samples {
            // Detect zero crossings for biphase mark decoding
            let crossed = (s >= 0.0) != (self.last_sample >= 0.0);
            self.last_sample = s;

            if crossed {
                self.bit_count += 1;
                // Every two half-periods is one bit
                if self.bit_count.is_multiple_of(2) {
                    self.decoded_bits.push(1);
                } else {
                    self.decoded_bits.push(0);
                }

                if self.decoded_bits.len() >= 80 {
                    if let Some(tc) = self.try_decode_frame() {
                        results.push(tc);
                    }
                    self.decoded_bits.clear();
                    self.bit_count = 0;
                }
            }

            self.buffer[self.buffer_pos] = s;
            self.buffer_pos = (self.buffer_pos + 1) % self.buffer.len();
        }

        results
    }

    /// Attempt to decode a timecode frame from accumulated bits.
    fn try_decode_frame(&self) -> Option<Timecode> {
        if self.decoded_bits.len() < 80 {
            return None;
        }

        // LTC sync word: bits 64-79 = 0011111111111101
        let sync_pattern = [0u8, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1];
        let sync_ok = self.decoded_bits[64..80]
            .iter()
            .zip(sync_pattern.iter())
            .all(|(a, b)| a == b);

        if !sync_ok {
            return None;
        }

        let frames = self.extract_bcd(0, 4, 2);
        let seconds = self.extract_bcd(8, 4, 3);
        let minutes = self.extract_bcd(24, 4, 3);
        let hours = self.extract_bcd(40, 4, 2);

        Timecode::new(hours, minutes, seconds, frames, self.frame_rate).ok()
    }

    /// Extract BCD value from bit array.
    fn extract_bcd(&self, offset: usize, unit_bits: usize, tens_bits: usize) -> u8 {
        let mut units = 0u8;
        for i in 0..unit_bits {
            if offset + i < self.decoded_bits.len() {
                units |= self.decoded_bits[offset + i] << i;
            }
        }
        let mut tens = 0u8;
        for i in 0..tens_bits {
            let idx = offset + unit_bits + 1 + i; // +1 skips user bit
            if idx < self.decoded_bits.len() {
                tens |= self.decoded_bits[idx] << i;
            }
        }
        tens * 10 + units
    }

    /// Get the configured frame rate.
    pub fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }

    /// Get the configured sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// Timecode validator.
#[derive(Debug, Clone)]
pub struct TimecodeValidator {
    frame_rate: FrameRate,
    last_tc: Option<Timecode>,
    discontinuity_count: u32,
    validate_continuity: bool,
}

impl TimecodeValidator {
    /// Create a new timecode validator.
    pub fn new(frame_rate: FrameRate) -> Self {
        Self {
            frame_rate,
            last_tc: None,
            discontinuity_count: 0,
            validate_continuity: true,
        }
    }

    /// Enable or disable continuity validation.
    pub fn set_validate_continuity(&mut self, enabled: bool) {
        self.validate_continuity = enabled;
    }

    /// Validate a timecode value.
    pub fn validate(&self, tc: &Timecode) -> Result<(), TimecodeError> {
        let fps = self.frame_rate.frames_per_second() as u8;
        if tc.hours > 23 {
            return Err(TimecodeError::InvalidHours);
        }
        if tc.minutes > 59 {
            return Err(TimecodeError::InvalidMinutes);
        }
        if tc.seconds > 59 {
            return Err(TimecodeError::InvalidSeconds);
        }
        if tc.frames >= fps {
            return Err(TimecodeError::InvalidFrames);
        }
        if self.frame_rate.is_drop_frame()
            && tc.seconds == 0
            && tc.frames < 2
            && !tc.minutes.is_multiple_of(10)
        {
            return Err(TimecodeError::InvalidDropFrame);
        }
        Ok(())
    }

    /// Validate and check continuity with the previous timecode.
    pub fn validate_sequence(&mut self, tc: Timecode) -> Result<bool, TimecodeError> {
        self.validate(&tc)?;

        let is_continuous = if let Some(ref last) = self.last_tc {
            if self.validate_continuity {
                let expected = last.to_frames() + 1;
                tc.to_frames() == expected
            } else {
                true
            }
        } else {
            true
        };

        if !is_continuous {
            self.discontinuity_count += 1;
        }

        self.last_tc = Some(tc);
        Ok(is_continuous)
    }

    /// Get the number of discontinuities detected.
    pub fn discontinuity_count(&self) -> u32 {
        self.discontinuity_count
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.last_tc = None;
        self.discontinuity_count = 0;
    }

    /// Get last validated timecode.
    pub fn last_timecode(&self) -> Option<&Timecode> {
        self.last_tc.as_ref()
    }
}

/// Parse a timecode string in HH:MM:SS:FF or HH:MM:SS;FF format.
pub fn parse_timecode_string(s: &str, frame_rate: FrameRate) -> Result<Timecode, TimecodeError> {
    let bytes = s.as_bytes();
    if bytes.len() < 11 {
        return Err(TimecodeError::InvalidConfiguration);
    }

    let parse_two = |b: &[u8]| -> Result<u8, TimecodeError> {
        if b.len() < 2 {
            return Err(TimecodeError::InvalidConfiguration);
        }
        let hi = (b[0] as char)
            .to_digit(10)
            .ok_or(TimecodeError::InvalidConfiguration)? as u8;
        let lo = (b[1] as char)
            .to_digit(10)
            .ok_or(TimecodeError::InvalidConfiguration)? as u8;
        Ok(hi * 10 + lo)
    };

    let hours = parse_two(&bytes[0..2])?;
    if bytes[2] != b':' {
        return Err(TimecodeError::InvalidConfiguration);
    }
    let minutes = parse_two(&bytes[3..5])?;
    if bytes[5] != b':' {
        return Err(TimecodeError::InvalidConfiguration);
    }
    let seconds = parse_two(&bytes[6..8])?;
    let sep = bytes[8];
    if sep != b':' && sep != b';' {
        return Err(TimecodeError::InvalidConfiguration);
    }
    let frames = parse_two(&bytes[9..11])?;

    Timecode::new(hours, minutes, seconds, frames, frame_rate)
}

/// VITC line number selector for standard formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VitcLine {
    /// Line 14 (common for 525-line systems).
    Line14,
    /// Line 16 (alternate for 525-line systems).
    Line16,
    /// Line 19 (common for 625-line systems).
    Line19,
    /// Line 21 (alternate for 625-line systems).
    Line21,
    /// Custom line number.
    Custom(u16),
}

impl VitcLine {
    /// Get the line number.
    pub fn line_number(&self) -> u16 {
        match self {
            VitcLine::Line14 => 14,
            VitcLine::Line16 => 16,
            VitcLine::Line19 => 19,
            VitcLine::Line21 => 21,
            VitcLine::Custom(n) => *n,
        }
    }
}

/// Decoded timecode with metadata about source and confidence.
#[derive(Debug, Clone)]
pub struct DecodedTimecode {
    /// The timecode value.
    pub timecode: Timecode,
    /// Frame rate information.
    pub frame_rate_info: FrameRateInfo,
    /// Confidence score 0.0–1.0.
    pub confidence: f32,
    /// Source type.
    pub source: TimecodeSource,
}

/// Source of a decoded timecode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimecodeSource {
    /// Linear Timecode from audio track.
    Ltc,
    /// Vertical Interval Timecode from video.
    Vitc,
    /// MIDI Timecode.
    Mtc,
    /// Embedded metadata.
    Metadata,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_timecode_string_25fps() {
        let tc =
            parse_timecode_string("01:02:03:04", FrameRate::Fps25).expect("valid timecode string");
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 2);
        assert_eq!(tc.seconds, 3);
        assert_eq!(tc.frames, 4);
    }

    #[test]
    fn test_parse_timecode_string_drop_frame() {
        let tc = parse_timecode_string("01:02:03;04", FrameRate::Fps2997DF)
            .expect("valid timecode string");
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.seconds, 3);
        assert_eq!(tc.frames, 4);
    }

    #[test]
    fn test_parse_timecode_string_invalid() {
        assert!(parse_timecode_string("bad", FrameRate::Fps25).is_err());
        assert!(parse_timecode_string("01:02:03X04", FrameRate::Fps25).is_err());
    }

    #[test]
    fn test_vitc_parser_new() {
        let parser = VitcParser::new(FrameRate::Fps25, 720);
        assert_eq!(parser.frame_rate, FrameRate::Fps25);
        assert!(parser.clock_period_samples >= 1);
    }

    #[test]
    fn test_vitc_parser_short_line() {
        let parser = VitcParser::new(FrameRate::Fps25, 720);
        let short = vec![0.5f32; 10];
        let result = parser.parse_line(&short);
        assert!(result.timecode.is_none());
        assert!(!result.crc_ok);
    }

    #[test]
    fn test_vitc_parser_all_zeros() {
        let parser = VitcParser::new(FrameRate::Fps25, 720);
        let pixels = vec![0.0f32; 720];
        let result = parser.parse_line(&pixels);
        // No sync pattern so should fail
        assert!(result.timecode.is_none());
    }

    #[test]
    fn test_ltc_decoder_new() {
        let dec = LtcDecoder::new(FrameRate::Fps25, 48000);
        assert_eq!(dec.frame_rate(), FrameRate::Fps25);
        assert_eq!(dec.sample_rate(), 48000);
    }

    #[test]
    fn test_ltc_decoder_feed_silence() {
        let mut dec = LtcDecoder::new(FrameRate::Fps25, 48000);
        let silence = vec![0.0f32; 48000];
        let results = dec.feed(&silence);
        // Silence produces no crossings, no timecodes
        assert!(results.is_empty());
    }

    #[test]
    fn test_timecode_validator_valid() {
        let validator = TimecodeValidator::new(FrameRate::Fps25);
        let tc = Timecode::new(1, 2, 3, 4, FrameRate::Fps25).expect("valid timecode");
        assert!(validator.validate(&tc).is_ok());
    }

    #[test]
    fn test_timecode_validator_invalid_frames() {
        let validator = TimecodeValidator::new(FrameRate::Fps25);
        let tc = Timecode {
            hours: 0,
            minutes: 0,
            seconds: 0,
            frames: 30, // invalid for 25fps
            frame_rate: FrameRateInfo {
                fps: 25,
                drop_frame: false,
            },
            user_bits: 0,
        };
        assert!(validator.validate(&tc).is_err());
    }

    #[test]
    fn test_timecode_validator_sequence_continuity() {
        let mut validator = TimecodeValidator::new(FrameRate::Fps25);
        let tc1 = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid timecode");
        let tc2 = Timecode::new(0, 0, 0, 1, FrameRate::Fps25).expect("valid timecode");
        assert!(validator
            .validate_sequence(tc1)
            .expect("validation should succeed"));
        assert!(validator
            .validate_sequence(tc2)
            .expect("validation should succeed"));
        assert_eq!(validator.discontinuity_count(), 0);
    }

    #[test]
    fn test_timecode_validator_sequence_discontinuity() {
        let mut validator = TimecodeValidator::new(FrameRate::Fps25);
        let tc1 = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid timecode");
        let tc2 = Timecode::new(0, 0, 1, 5, FrameRate::Fps25).expect("valid timecode");
        validator
            .validate_sequence(tc1)
            .expect("validation should succeed");
        let cont = validator
            .validate_sequence(tc2)
            .expect("validation should succeed");
        assert!(!cont);
        assert_eq!(validator.discontinuity_count(), 1);
    }

    #[test]
    fn test_timecode_validator_reset() {
        let mut validator = TimecodeValidator::new(FrameRate::Fps25);
        let tc = Timecode::new(0, 0, 1, 5, FrameRate::Fps25).expect("valid timecode");
        validator
            .validate_sequence(tc)
            .expect("validation should succeed");
        validator.reset();
        assert_eq!(validator.discontinuity_count(), 0);
        assert!(validator.last_timecode().is_none());
    }

    #[test]
    fn test_vitc_line_numbers() {
        assert_eq!(VitcLine::Line14.line_number(), 14);
        assert_eq!(VitcLine::Line19.line_number(), 19);
        assert_eq!(VitcLine::Custom(22).line_number(), 22);
    }

    #[test]
    fn test_timecode_source_eq() {
        assert_eq!(TimecodeSource::Ltc, TimecodeSource::Ltc);
        assert_ne!(TimecodeSource::Ltc, TimecodeSource::Vitc);
    }
}
