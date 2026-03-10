//! LTC (Linear Timecode) bit-level parser.
//!
//! Parses the raw 80-bit LTC word into a `LtcFrame` containing a decoded
//! `Timecode`.  The parser operates on a slice of `LtcBit` values
//! (i.e. clock-qualified biphase-mark decoded bits) and locates the sync
//! word, then reconstructs the timecode fields.
//!
//! # SMPTE 12M LTC word layout (80 bits, LSB first per group)
//! - Bits 0-3:   frame units
//! - Bit 4:      user bit 1
//! - Bit 5:      user bit 2  (actually these are user bit nibbles interleaved)
//! - Bits 4,6,10,12,18,20,26,28: user-bit nibble pairs
//! - Bits 8-9:   frame tens + drop-frame flag (bit 10) + color-frame (bit 11)
//! - Bits 16-19: seconds units
//! - Bits 24-26: seconds tens
//! - Bits 32-35: minutes units
//! - Bits 40-42: minutes tens
//! - Bits 48-51: hours units
//! - Bits 56-57: hours tens
//! - Bits 64-79: sync word 0xBFFC (0011111111111101 in LS→MS order)

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use crate::{FrameRateInfo, Timecode, TimecodeError};

// ── LtcBit ────────────────────────────────────────────────────────────────────

/// A single logical bit in an LTC data stream after biphase-mark decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LtcBit {
    /// Logical zero.
    Zero,
    /// Logical one.
    One,
}

impl LtcBit {
    /// Convert to `u8` (0 or 1).
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Zero => 0,
            Self::One => 1,
        }
    }
}

impl From<bool> for LtcBit {
    fn from(b: bool) -> Self {
        if b {
            Self::One
        } else {
            Self::Zero
        }
    }
}

impl From<u8> for LtcBit {
    fn from(v: u8) -> Self {
        if v != 0 {
            Self::One
        } else {
            Self::Zero
        }
    }
}

// ── LtcFrame ─────────────────────────────────────────────────────────────────

/// A fully decoded LTC frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LtcFrame {
    /// The decoded timecode.
    pub timecode: Timecode,
    /// 32-bit user bits extracted from the LTC word.
    pub user_bits: u32,
    /// Drop-frame flag as encoded in the bitstream.
    pub drop_frame: bool,
    /// Color-frame flag.
    pub color_frame: bool,
    /// Biphase-mark polarity correction bit.
    pub biphase_polarity: bool,
    /// Byte position (bit 0 offset) within the source buffer where this frame began.
    pub bit_offset: usize,
}

// ── LtcParser ────────────────────────────────────────────────────────────────

/// Parses raw LTC bit streams into `LtcFrame` values.
///
/// # Example
/// ```
/// use oximedia_timecode::ltc_parser::{LtcBit, LtcParser};
///
/// let parser = LtcParser::new(30, false);
/// // Build a minimal 80-bit all-zero LTC word (plus sync word)
/// let mut bits = vec![LtcBit::Zero; 64];
/// // Append sync word: 0011 1111 1111 1101  (LS bit first)
/// let sync: [u8; 16] = [0,0,1,1, 1,1,1,1, 1,1,1,1, 1,1,0,1];
/// bits.extend(sync.iter().map(|&b| LtcBit::from(b)));
/// let frames = parser.decode_bits(&bits);
/// assert_eq!(frames.len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct LtcParser {
    /// Nominal frames-per-second (30 for NTSC, 25 for PAL, etc.)
    pub fps: u8,
    /// Whether drop-frame mode should be assumed when not encoded.
    pub default_drop_frame: bool,
}

impl LtcParser {
    /// The 16-bit LTC sync word value (bit 64..79 of each LTC word).
    /// In bit order from bit-64 to bit-79 (LSB first): 0011 1111 1111 1101
    const SYNC_BITS: [u8; 16] = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1];

    /// Create a new parser.
    ///
    /// `fps` should be 24, 25, or 30.  `default_drop_frame` sets the drop-frame
    /// assumption for streams that do not encode the flag.
    pub fn new(fps: u8, default_drop_frame: bool) -> Self {
        Self {
            fps,
            default_drop_frame,
        }
    }

    /// Scan a slice of bits for valid LTC frames and return all decoded frames.
    ///
    /// The function searches for the 16-bit sync word at the end of each
    /// candidate 80-bit window.
    pub fn decode_bits(&self, bits: &[LtcBit]) -> Vec<LtcFrame> {
        if bits.len() < 80 {
            return Vec::new();
        }
        let mut frames = Vec::new();
        let mut i = 0;
        while i + 80 <= bits.len() {
            // Check sync word at bits [i+64 .. i+80]
            if self.check_sync(bits, i + 64) {
                if let Ok(frame) = self.decode_frame(bits, i) {
                    frames.push(frame);
                    i += 80;
                    continue;
                }
            }
            i += 1;
        }
        frames
    }

    /// Decode a single 80-bit LTC word starting at `offset`.
    pub fn decode_frame(&self, bits: &[LtcBit], offset: usize) -> Result<LtcFrame, TimecodeError> {
        if offset + 80 > bits.len() {
            return Err(TimecodeError::BufferTooSmall);
        }

        let word = &bits[offset..offset + 80];

        // Helper: extract a nibble (4 bits) from positions in `word`
        let nibble = |positions: [usize; 4]| -> u8 {
            positions
                .iter()
                .enumerate()
                .map(|(shift, &pos)| word[pos].as_u8() << shift)
                .sum()
        };

        // Frame units (bits 0-3)
        let frame_units = nibble([0, 1, 2, 3]);
        // Frame tens (bits 8-9)
        let frame_tens = nibble([8, 9, 0, 0]) & 0x03;
        let drop_frame = word[10].as_u8() != 0;
        let color_frame = word[11].as_u8() != 0;

        // Seconds units (bits 16-19)
        let sec_units = nibble([16, 17, 18, 19]);
        // Seconds tens (bits 24-26)
        let sec_tens = nibble([24, 25, 26, 0]) & 0x07;

        // Minutes units (bits 32-35)
        let min_units = nibble([32, 33, 34, 35]);
        // Minutes tens (bits 40-42)
        let min_tens = nibble([40, 41, 42, 0]) & 0x07;

        // Hours units (bits 48-51)
        let hr_units = nibble([48, 49, 50, 51]);
        // Hours tens (bits 56-57)
        let hr_tens = nibble([56, 57, 0, 0]) & 0x03;
        let biphase_polarity = word[58].as_u8() != 0;

        let frames = frame_tens * 10 + frame_units;
        let seconds = sec_tens * 10 + sec_units;
        let minutes = min_tens * 10 + min_units;
        let hours = hr_tens * 10 + hr_units;

        // User bits (8 nibbles spread across even bit positions 4,6,12,14,20,22,28,30,36,38...)
        // Simplified: extract the 8 user-bit nibble positions
        let ub_positions: [[usize; 4]; 8] = [
            [4, 5, 0, 0], // UB1 (2 bits only in some specs; use 4 for simplicity)
            [6, 7, 0, 0],
            [12, 13, 0, 0],
            [14, 15, 0, 0],
            [22, 23, 0, 0],
            [28, 29, 0, 0], // crossing into next byte area
            [36, 37, 0, 0],
            [44, 45, 0, 0],
        ];
        let mut user_bits: u32 = 0;
        for (idx, pos) in ub_positions.iter().enumerate() {
            let nibval = (word[pos[0]].as_u8() | (word[pos[1]].as_u8() << 1)) as u32;
            user_bits |= nibval << (idx * 2);
        }

        let frame_rate_info = FrameRateInfo {
            fps: self.fps,
            drop_frame,
        };

        let timecode = Timecode {
            hours,
            minutes,
            seconds,
            frames,
            frame_rate: frame_rate_info,
            user_bits,
        };

        Ok(LtcFrame {
            timecode,
            user_bits,
            drop_frame,
            color_frame,
            biphase_polarity,
            bit_offset: offset,
        })
    }

    /// Return `true` if the 16 bits at `offset` match the LTC sync word.
    pub fn check_sync(&self, bits: &[LtcBit], offset: usize) -> bool {
        if offset + 16 > bits.len() {
            return false;
        }
        Self::SYNC_BITS
            .iter()
            .enumerate()
            .all(|(i, &expected)| bits[offset + i].as_u8() == expected)
    }

    /// Encode a `Timecode` into an 80-bit LTC word (returned as `Vec<LtcBit>`).
    ///
    /// This is the inverse of `decode_frame` and is useful for round-trip tests.
    pub fn encode_frame(&self, tc: &Timecode) -> Vec<LtcBit> {
        let mut word = vec![LtcBit::Zero; 80];

        let set_bit = |word: &mut Vec<LtcBit>, pos: usize, val: u8| {
            word[pos] = LtcBit::from(val);
        };

        // Frame units / tens
        let fu = tc.frames % 10;
        let ft = tc.frames / 10;
        for i in 0..4 {
            set_bit(&mut word, i, (fu >> i) & 1);
        }
        set_bit(&mut word, 8, ft & 1);
        set_bit(&mut word, 9, (ft >> 1) & 1);
        set_bit(&mut word, 10, tc.frame_rate.drop_frame as u8);

        // Seconds
        let su = tc.seconds % 10;
        let st = tc.seconds / 10;
        for i in 0..4 {
            set_bit(&mut word, 16 + i, (su >> i) & 1);
        }
        for i in 0..3 {
            set_bit(&mut word, 24 + i, (st >> i) & 1);
        }

        // Minutes
        let mu = tc.minutes % 10;
        let mt = tc.minutes / 10;
        for i in 0..4 {
            set_bit(&mut word, 32 + i, (mu >> i) & 1);
        }
        for i in 0..3 {
            set_bit(&mut word, 40 + i, (mt >> i) & 1);
        }

        // Hours
        let hu = tc.hours % 10;
        let ht = tc.hours / 10;
        for i in 0..4 {
            set_bit(&mut word, 48 + i, (hu >> i) & 1);
        }
        for i in 0..2 {
            set_bit(&mut word, 56 + i, (ht >> i) & 1);
        }

        // Sync word
        for (i, &b) in Self::SYNC_BITS.iter().enumerate() {
            set_bit(&mut word, 64 + i, b);
        }

        word
    }
}

/// Helper: build an 80-bit LTC word from raw field values (for test construction).
pub fn build_ltc_word(
    hours: u8,
    minutes: u8,
    seconds: u8,
    frames: u8,
    drop_frame: bool,
    fps: u8,
) -> Vec<LtcBit> {
    use crate::FrameRateInfo;
    let tc = Timecode {
        hours,
        minutes,
        seconds,
        frames,
        frame_rate: FrameRateInfo { fps, drop_frame },
        user_bits: 0,
    };
    let parser = LtcParser::new(fps, drop_frame);
    parser.encode_frame(&tc)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_parser() -> LtcParser {
        LtcParser::new(25, false)
    }

    #[test]
    fn test_ltcbit_from_bool() {
        assert_eq!(LtcBit::from(true), LtcBit::One);
        assert_eq!(LtcBit::from(false), LtcBit::Zero);
    }

    #[test]
    fn test_ltcbit_from_u8() {
        assert_eq!(LtcBit::from(1u8), LtcBit::One);
        assert_eq!(LtcBit::from(0u8), LtcBit::Zero);
        assert_eq!(LtcBit::from(255u8), LtcBit::One);
    }

    #[test]
    fn test_ltcbit_as_u8() {
        assert_eq!(LtcBit::One.as_u8(), 1);
        assert_eq!(LtcBit::Zero.as_u8(), 0);
    }

    #[test]
    fn test_check_sync_valid() {
        let mut bits = vec![LtcBit::Zero; 80];
        for (i, &b) in LtcParser::SYNC_BITS.iter().enumerate() {
            bits[64 + i] = LtcBit::from(b);
        }
        assert!(make_parser().check_sync(&bits, 64));
    }

    #[test]
    fn test_check_sync_invalid() {
        let bits = vec![LtcBit::Zero; 80];
        assert!(!make_parser().check_sync(&bits, 64));
    }

    #[test]
    fn test_check_sync_too_short() {
        let bits = vec![LtcBit::Zero; 10];
        assert!(!make_parser().check_sync(&bits, 0));
    }

    #[test]
    fn test_encode_decode_roundtrip_simple() {
        let parser = make_parser();
        let tc = Timecode {
            hours: 1,
            minutes: 2,
            seconds: 3,
            frames: 4,
            frame_rate: crate::FrameRateInfo {
                fps: 25,
                drop_frame: false,
            },
            user_bits: 0,
        };
        let encoded = parser.encode_frame(&tc);
        assert_eq!(encoded.len(), 80);
        let decoded = parser
            .decode_frame(&encoded, 0)
            .expect("decode should succeed");
        assert_eq!(decoded.timecode.hours, 1);
        assert_eq!(decoded.timecode.minutes, 2);
        assert_eq!(decoded.timecode.seconds, 3);
        assert_eq!(decoded.timecode.frames, 4);
    }

    #[test]
    fn test_encode_decode_midnight() {
        let parser = make_parser();
        let tc = Timecode {
            hours: 0,
            minutes: 0,
            seconds: 0,
            frames: 0,
            frame_rate: crate::FrameRateInfo {
                fps: 25,
                drop_frame: false,
            },
            user_bits: 0,
        };
        let encoded = parser.encode_frame(&tc);
        let decoded = parser
            .decode_frame(&encoded, 0)
            .expect("decode should succeed");
        assert_eq!(decoded.timecode.hours, 0);
        assert_eq!(decoded.timecode.seconds, 0);
    }

    #[test]
    fn test_decode_bits_finds_one_frame() {
        let parser = make_parser();
        let tc = Timecode {
            hours: 0,
            minutes: 1,
            seconds: 2,
            frames: 3,
            frame_rate: crate::FrameRateInfo {
                fps: 25,
                drop_frame: false,
            },
            user_bits: 0,
        };
        let bits = parser.encode_frame(&tc);
        let frames = parser.decode_bits(&bits);
        assert_eq!(frames.len(), 1);
    }

    #[test]
    fn test_decode_bits_empty() {
        assert!(make_parser().decode_bits(&[]).is_empty());
    }

    #[test]
    fn test_decode_bits_too_short() {
        let bits = vec![LtcBit::Zero; 40];
        assert!(make_parser().decode_bits(&bits).is_empty());
    }

    #[test]
    fn test_decode_frame_buffer_too_small() {
        let bits = vec![LtcBit::Zero; 79];
        let err = make_parser().decode_frame(&bits, 0);
        assert_eq!(err, Err(TimecodeError::BufferTooSmall));
    }

    #[test]
    fn test_decode_drop_frame_flag() {
        let parser = LtcParser::new(30, true);
        let tc = Timecode {
            hours: 0,
            minutes: 0,
            seconds: 5,
            frames: 0,
            frame_rate: crate::FrameRateInfo {
                fps: 30,
                drop_frame: true,
            },
            user_bits: 0,
        };
        let encoded = parser.encode_frame(&tc);
        let decoded = parser
            .decode_frame(&encoded, 0)
            .expect("decode should succeed");
        assert!(decoded.drop_frame);
    }

    #[test]
    fn test_build_ltc_word_length() {
        let word = build_ltc_word(1, 2, 3, 4, false, 25);
        assert_eq!(word.len(), 80);
    }

    #[test]
    fn test_decode_frame_bit_offset() {
        let parser = make_parser();
        let tc = Timecode {
            hours: 0,
            minutes: 0,
            seconds: 0,
            frames: 0,
            frame_rate: crate::FrameRateInfo {
                fps: 25,
                drop_frame: false,
            },
            user_bits: 0,
        };
        let bits = parser.encode_frame(&tc);
        let decoded = parser
            .decode_frame(&bits, 0)
            .expect("decode should succeed");
        assert_eq!(decoded.bit_offset, 0);
    }

    #[test]
    fn test_ltcframe_color_frame_default_false() {
        let parser = make_parser();
        let tc = Timecode {
            hours: 0,
            minutes: 0,
            seconds: 0,
            frames: 0,
            frame_rate: crate::FrameRateInfo {
                fps: 25,
                drop_frame: false,
            },
            user_bits: 0,
        };
        let encoded = parser.encode_frame(&tc);
        let decoded = parser
            .decode_frame(&encoded, 0)
            .expect("decode should succeed");
        assert!(!decoded.color_frame);
    }
}
