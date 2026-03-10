//! MIDI Timecode (MTC) implementation.
//!
//! MIDI Timecode is a standard for synchronizing MIDI devices to a timecode
//! reference. It transmits timecode as either Full Frame SysEx messages or
//! as a sequence of 8 quarter-frame messages.

/// MTC frame rate codes and values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum MtcFrameRate {
    /// 24 fps (film)
    Fps24,
    /// 25 fps (PAL)
    Fps25,
    /// 29.97 fps drop frame (NTSC)
    Fps2997,
    /// 30 fps (non-drop frame)
    Fps30,
}

impl MtcFrameRate {
    /// Get the MTC frame rate code (0-3) as encoded in the SysEx message.
    ///
    /// - 0 = 24 fps
    /// - 1 = 25 fps
    /// - 2 = 29.97 fps (drop frame)
    /// - 3 = 30 fps
    #[must_use]
    pub fn code(&self) -> u8 {
        match self {
            MtcFrameRate::Fps24 => 0,
            MtcFrameRate::Fps25 => 1,
            MtcFrameRate::Fps2997 => 2,
            MtcFrameRate::Fps30 => 3,
        }
    }

    /// Get the frame rate as a floating point value.
    #[must_use]
    pub fn frames_per_sec(&self) -> f32 {
        match self {
            MtcFrameRate::Fps24 => 24.0,
            MtcFrameRate::Fps25 => 25.0,
            MtcFrameRate::Fps2997 => 29.97,
            MtcFrameRate::Fps30 => 30.0,
        }
    }

    /// Get the integer frame count per second (ceiling).
    #[must_use]
    pub fn frames_per_sec_int(&self) -> u8 {
        match self {
            MtcFrameRate::Fps24 => 24,
            MtcFrameRate::Fps25 => 25,
            MtcFrameRate::Fps2997 => 30, // counted as 30 in MTC
            MtcFrameRate::Fps30 => 30,
        }
    }

    /// Create from the 2-bit MTC rate code.
    #[must_use]
    pub fn from_code(code: u8) -> Option<Self> {
        match code & 0x03 {
            0 => Some(MtcFrameRate::Fps24),
            1 => Some(MtcFrameRate::Fps25),
            2 => Some(MtcFrameRate::Fps2997),
            3 => Some(MtcFrameRate::Fps30),
            _ => None,
        }
    }
}

/// MIDI Timecode value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub struct MtcTimecode {
    /// Hours (0-23)
    pub hours: u8,
    /// Minutes (0-59)
    pub minutes: u8,
    /// Seconds (0-59)
    pub seconds: u8,
    /// Frames (0 to frames_per_sec - 1)
    pub frames: u8,
    /// Frame rate
    pub frame_rate: MtcFrameRate,
}

impl MtcTimecode {
    /// Create a new MTC timecode value.
    #[must_use]
    pub fn new(hours: u8, minutes: u8, seconds: u8, frames: u8, frame_rate: MtcFrameRate) -> Self {
        Self {
            hours: hours.min(23),
            minutes: minutes.min(59),
            seconds: seconds.min(59),
            frames: frames.min(frame_rate.frames_per_sec_int() - 1),
            frame_rate,
        }
    }

    /// Convert to total frame count since 00:00:00:00.
    #[must_use]
    pub fn to_frame_count(&self) -> u64 {
        let fps = u64::from(self.frame_rate.frames_per_sec_int());
        let hours = u64::from(self.hours);
        let minutes = u64::from(self.minutes);
        let seconds = u64::from(self.seconds);
        let frames = u64::from(self.frames);

        hours * 3600 * fps + minutes * 60 * fps + seconds * fps + frames
    }

    /// Create an MTC timecode from a frame count.
    #[must_use]
    pub fn from_frame_count(frame_count: u64, rate: MtcFrameRate) -> Self {
        let fps = u64::from(rate.frames_per_sec_int());
        let total_seconds = frame_count / fps;
        let frames = (frame_count % fps) as u8;
        let seconds = (total_seconds % 60) as u8;
        let total_minutes = total_seconds / 60;
        let minutes = (total_minutes % 60) as u8;
        let hours = ((total_minutes / 60) % 24) as u8;

        Self::new(hours, minutes, seconds, frames, rate)
    }
}

/// MTC Full Frame message encoder/decoder.
///
/// Full Frame SysEx format: `[0xF0, 0x7F, 0x7F, 0x01, 0x01, hh, mm, ss, ff, 0xF7]`
/// where `hh` encodes both the hours (5 bits) and rate code (2 bits).
#[allow(dead_code)]
pub struct MtcFullFrame;

impl MtcFullFrame {
    /// Encode an MTC timecode as a Full Frame SysEx message.
    ///
    /// The `hh` byte encodes: `0rrhhhhh` where `rr` is the rate code and
    /// `hhhhh` is the hours value.
    #[must_use]
    pub fn encode(tc: &MtcTimecode) -> Vec<u8> {
        let rate_code = tc.frame_rate.code();
        let hh = (rate_code << 5) | (tc.hours & 0x1F);

        vec![
            0xF0, // SysEx start
            0x7F, // Real-time universal SysEx
            0x7F, // All devices
            0x01, // MTC
            0x01, // Full Frame
            hh, tc.minutes, tc.seconds, tc.frames, 0xF7, // SysEx end
        ]
    }

    /// Decode a Full Frame SysEx message into an MTC timecode.
    ///
    /// Returns `None` if the data is not a valid MTC Full Frame message.
    #[must_use]
    pub fn decode(data: &[u8]) -> Option<MtcTimecode> {
        if data.len() < 10 {
            return None;
        }
        if data[0] != 0xF0
            || data[1] != 0x7F
            || data[2] != 0x7F
            || data[3] != 0x01
            || data[4] != 0x01
            || data[9] != 0xF7
        {
            return None;
        }

        let hh = data[5];
        let rate_code = (hh >> 5) & 0x03;
        let hours = hh & 0x1F;
        let minutes = data[6];
        let seconds = data[7];
        let frames = data[8];

        let frame_rate = MtcFrameRate::from_code(rate_code)?;

        Some(MtcTimecode::new(
            hours, minutes, seconds, frames, frame_rate,
        ))
    }
}

/// MTC quarter-frame message utilities.
///
/// MTC is transmitted as 8 quarter-frame messages per timecode frame.
/// Each message carries 4 bits of timecode data.
#[allow(dead_code)]
pub struct MtcQuarterFrame;

impl MtcQuarterFrame {
    /// Encode one of the 8 quarter-frame pieces.
    ///
    /// Each quarter-frame message is a 2-byte sequence: `[0xF1, data]`
    /// where `data` is `ppppdddd` (piece number in high nibble, data in low).
    ///
    /// The 8 pieces (0-7) carry:
    /// - 0: frames low nibble
    /// - 1: frames high nibble (2 bits)
    /// - 2: seconds low nibble
    /// - 3: seconds high nibble (3 bits)
    /// - 4: minutes low nibble
    /// - 5: minutes high nibble (3 bits)
    /// - 6: hours low nibble
    /// - 7: hours high nibble + rate code (3 bits)
    ///
    /// Returns the data byte (the second byte of the 0xF1 message pair).
    #[must_use]
    pub fn encode_quarter(tc: &MtcTimecode, piece: u8) -> u8 {
        let piece = piece & 0x07;
        let data: u8 = match piece {
            0 => tc.frames & 0x0F,
            1 => (tc.frames >> 4) & 0x01,
            2 => tc.seconds & 0x0F,
            3 => (tc.seconds >> 4) & 0x03,
            4 => tc.minutes & 0x0F,
            5 => (tc.minutes >> 4) & 0x03,
            6 => tc.hours & 0x0F,
            7 => ((tc.frame_rate.code() & 0x03) << 1) | ((tc.hours >> 4) & 0x01),
            _ => 0,
        };
        (piece << 4) | (data & 0x0F)
    }
}

/// MTC receiver that assembles quarter-frame messages into timecodes.
#[derive(Debug)]
#[allow(dead_code)]
pub struct MtcReceiver {
    /// Accumulated quarter-frame data (8 nibbles)
    nibbles: [u8; 8],
    /// Number of quarter frames received in current sequence
    count: usize,
    /// Whether we have received a complete set of 8 quarter frames
    complete: bool,
}

impl MtcReceiver {
    /// Create a new MTC receiver.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nibbles: [0u8; 8],
            count: 0,
            complete: false,
        }
    }

    /// Process a single MTC quarter-frame data byte.
    ///
    /// Returns `Some(MtcTimecode)` when all 8 quarter frames have been received
    /// and assembled into a complete timecode.
    pub fn process_message(&mut self, msg: u8) -> Option<MtcTimecode> {
        let piece = (msg >> 4) & 0x07;
        let data = msg & 0x0F;

        self.nibbles[piece as usize] = data;
        self.count += 1;

        // We need all 8 pieces to reconstruct a timecode
        if self.count >= 8 {
            self.complete = true;
            self.count = 0;
            return self.assemble();
        }

        None
    }

    /// Assemble accumulated nibbles into a timecode.
    fn assemble(&self) -> Option<MtcTimecode> {
        let frames = self.nibbles[0] | (self.nibbles[1] << 4);
        let seconds = self.nibbles[2] | (self.nibbles[3] << 4);
        let minutes = self.nibbles[4] | (self.nibbles[5] << 4);
        let hours = self.nibbles[6] | ((self.nibbles[7] & 0x01) << 4);
        let rate_code = (self.nibbles[7] >> 1) & 0x03;

        let frame_rate = MtcFrameRate::from_code(rate_code)?;
        Some(MtcTimecode::new(
            hours, minutes, seconds, frames, frame_rate,
        ))
    }

    /// Reset the receiver state.
    pub fn reset(&mut self) {
        self.nibbles = [0u8; 8];
        self.count = 0;
        self.complete = false;
    }

    /// Check if a complete timecode has been received.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.complete
    }
}

impl Default for MtcReceiver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mtc_frame_rate_code() {
        assert_eq!(MtcFrameRate::Fps24.code(), 0);
        assert_eq!(MtcFrameRate::Fps25.code(), 1);
        assert_eq!(MtcFrameRate::Fps2997.code(), 2);
        assert_eq!(MtcFrameRate::Fps30.code(), 3);
    }

    #[test]
    fn test_mtc_frame_rate_from_code() {
        assert_eq!(MtcFrameRate::from_code(0), Some(MtcFrameRate::Fps24));
        assert_eq!(MtcFrameRate::from_code(1), Some(MtcFrameRate::Fps25));
        assert_eq!(MtcFrameRate::from_code(2), Some(MtcFrameRate::Fps2997));
        assert_eq!(MtcFrameRate::from_code(3), Some(MtcFrameRate::Fps30));
    }

    #[test]
    fn test_mtc_frame_rate_fps() {
        assert!((MtcFrameRate::Fps24.frames_per_sec() - 24.0).abs() < f32::EPSILON);
        assert!((MtcFrameRate::Fps25.frames_per_sec() - 25.0).abs() < f32::EPSILON);
        assert!((MtcFrameRate::Fps2997.frames_per_sec() - 29.97).abs() < 0.001);
        assert!((MtcFrameRate::Fps30.frames_per_sec() - 30.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mtc_timecode_new() {
        let tc = MtcTimecode::new(1, 2, 3, 4, MtcFrameRate::Fps25);
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 2);
        assert_eq!(tc.seconds, 3);
        assert_eq!(tc.frames, 4);
    }

    #[test]
    fn test_mtc_timecode_to_frame_count() {
        let tc = MtcTimecode::new(0, 0, 1, 0, MtcFrameRate::Fps25);
        assert_eq!(tc.to_frame_count(), 25);

        let tc2 = MtcTimecode::new(1, 0, 0, 0, MtcFrameRate::Fps30);
        assert_eq!(tc2.to_frame_count(), 3600 * 30);
    }

    #[test]
    fn test_mtc_timecode_from_frame_count() {
        let tc = MtcTimecode::from_frame_count(3600 * 25, MtcFrameRate::Fps25);
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 0);
        assert_eq!(tc.seconds, 0);
        assert_eq!(tc.frames, 0);
    }

    #[test]
    fn test_mtc_timecode_roundtrip() {
        let original = MtcTimecode::new(1, 30, 45, 12, MtcFrameRate::Fps25);
        let frames = original.to_frame_count();
        let recovered = MtcTimecode::from_frame_count(frames, MtcFrameRate::Fps25);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_mtc_full_frame_encode() {
        let tc = MtcTimecode::new(1, 2, 3, 4, MtcFrameRate::Fps25);
        let data = MtcFullFrame::encode(&tc);
        assert_eq!(data.len(), 10);
        assert_eq!(data[0], 0xF0);
        assert_eq!(data[1], 0x7F);
        assert_eq!(data[2], 0x7F);
        assert_eq!(data[3], 0x01);
        assert_eq!(data[4], 0x01);
        // hh: rate_code=1 (25fps), hours=1 => (1<<5) | 1 = 33
        assert_eq!(data[5], (1 << 5) | 1);
        assert_eq!(data[6], 2);
        assert_eq!(data[7], 3);
        assert_eq!(data[8], 4);
        assert_eq!(data[9], 0xF7);
    }

    #[test]
    fn test_mtc_full_frame_decode() {
        let tc = MtcTimecode::new(2, 10, 30, 15, MtcFrameRate::Fps30);
        let encoded = MtcFullFrame::encode(&tc);
        let decoded = MtcFullFrame::decode(&encoded).expect("should succeed");
        assert_eq!(decoded.hours, 2);
        assert_eq!(decoded.minutes, 10);
        assert_eq!(decoded.seconds, 30);
        assert_eq!(decoded.frames, 15);
        assert_eq!(decoded.frame_rate, MtcFrameRate::Fps30);
    }

    #[test]
    fn test_mtc_full_frame_decode_invalid() {
        assert!(MtcFullFrame::decode(&[]).is_none());
        assert!(MtcFullFrame::decode(&[0xF0, 0x00, 0x7F, 0x01, 0x01, 0, 0, 0, 0, 0xF7]).is_none());
    }

    #[test]
    fn test_mtc_receiver_assemble() {
        let tc = MtcTimecode::new(1, 2, 3, 4, MtcFrameRate::Fps25);
        let mut receiver = MtcReceiver::new();

        // Send all 8 quarter frames
        let mut result = None;
        for piece in 0..8u8 {
            let byte = MtcQuarterFrame::encode_quarter(&tc, piece);
            result = receiver.process_message(byte);
        }

        let decoded = result.expect("result should be ok");
        assert_eq!(decoded.hours, tc.hours);
        assert_eq!(decoded.minutes, tc.minutes);
        assert_eq!(decoded.seconds, tc.seconds);
        assert_eq!(decoded.frames, tc.frames);
        assert_eq!(decoded.frame_rate, tc.frame_rate);
    }

    #[test]
    fn test_mtc_receiver_reset() {
        let mut receiver = MtcReceiver::new();
        receiver.process_message(0x00);
        receiver.reset();
        assert!(!receiver.is_complete());
    }

    #[test]
    fn test_mtc_receiver_default() {
        let receiver = MtcReceiver::default();
        assert!(!receiver.is_complete());
    }
}
