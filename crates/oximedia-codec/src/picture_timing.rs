//! Picture timing supplemental enhancement information (SEI) types.
//!
//! Provides structures for representing H.264/H.265 picture timing SEI messages,
//! including clock timestamps and PicStruct values used to signal interlacing
//! and repeat-field behaviour.

#![allow(dead_code)]

/// Pic-struct values from H.264 Table D-1 / H.265 Table D.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PicStruct {
    /// Frame (progressive).
    Frame,
    /// Top field only.
    TopField,
    /// Bottom field only.
    BottomField,
    /// Top field, then bottom field.
    TopBottomField,
    /// Bottom field, then top field.
    BottomTopField,
    /// Top field, bottom field, then top field repeated.
    TopBottomTopField,
    /// Bottom field, top field, then bottom field repeated.
    BottomTopBottomField,
    /// Frame doubling.
    FrameDoubling,
    /// Frame tripling.
    FrameTripling,
}

impl PicStruct {
    /// Returns the number of progressive-scan frames implied by this PicStruct.
    ///
    /// For field-pair types this is 1; for doubling/tripling it is 2/3.
    pub fn progressive_frame_count(self) -> u32 {
        match self {
            Self::Frame
            | Self::TopField
            | Self::BottomField
            | Self::TopBottomField
            | Self::BottomTopField
            | Self::TopBottomTopField
            | Self::BottomTopBottomField => 1,
            Self::FrameDoubling => 2,
            Self::FrameTripling => 3,
        }
    }

    /// Returns `true` if this PicStruct represents a progressive frame.
    pub fn is_progressive(self) -> bool {
        matches!(
            self,
            Self::Frame | Self::FrameDoubling | Self::FrameTripling
        )
    }

    /// Returns `true` if this PicStruct involves field repetition.
    pub fn has_repeated_field(self) -> bool {
        matches!(self, Self::TopBottomTopField | Self::BottomTopBottomField)
    }
}

/// A clock timestamp embedded in a picture timing SEI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClockTimestamp {
    /// Hours component (0–23).
    pub hours: u8,
    /// Minutes component (0–59).
    pub minutes: u8,
    /// Seconds component (0–59).
    pub seconds: u8,
    /// N_frames field from the bitstream.
    pub n_frames: u32,
    /// Discontinuity flag.
    pub discontinuity: bool,
    /// Clock timestamp count (ct_type): 0=progressive, 1=interlaced, 2=unknown.
    pub ct_type: u8,
    /// Nuit field clock (numerator of offset from nominal).
    pub nuit_field_based: bool,
    /// Counting type (0–6).
    pub counting_type: u8,
}

impl ClockTimestamp {
    /// Converts the timestamp to whole milliseconds from midnight.
    ///
    /// Does not account for `n_frames` (frame-level precision is codec-dependent).
    pub fn to_ms(self) -> u64 {
        let h = self.hours as u64;
        let m = self.minutes as u64;
        let s = self.seconds as u64;
        (h * 3600 + m * 60 + s) * 1000
    }

    /// Returns `true` when the discontinuity flag is set.
    pub fn is_discontinuity(self) -> bool {
        self.discontinuity
    }
}

/// Picture timing SEI payload.
#[derive(Debug, Clone)]
pub struct PictureTiming {
    /// CpbDpbDelaysPresentFlag controls whether cpb/dpb removal delays appear.
    pub cpb_dpb_delays_present: bool,
    /// Cpb removal delay in ticks.
    pub cpb_removal_delay: u32,
    /// Dpb output delay in ticks.
    pub dpb_output_delay: u32,
    /// PicStruct field (present when `pic_struct_present_flag` is set in VUI).
    pub pic_struct: Option<PicStruct>,
    /// Up to 3 optional clock timestamps.
    pub clock_timestamps: Vec<ClockTimestamp>,
    /// `repeat_first_field` flag extracted from the SEI.
    pub repeat_first_field: bool,
}

impl PictureTiming {
    /// Creates a minimal PictureTiming with no clock timestamps.
    pub fn new(cpb_removal_delay: u32, dpb_output_delay: u32) -> Self {
        Self {
            cpb_dpb_delays_present: true,
            cpb_removal_delay,
            dpb_output_delay,
            pic_struct: None,
            clock_timestamps: Vec::new(),
            repeat_first_field: false,
        }
    }

    /// Returns `true` when the `repeat_first_field` flag is set.
    pub fn is_repeat_first_field(&self) -> bool {
        self.repeat_first_field
    }

    /// Returns the PicStruct if present.
    pub fn pic_struct(&self) -> Option<PicStruct> {
        self.pic_struct
    }

    /// Number of clock timestamps attached.
    pub fn timestamp_count(&self) -> usize {
        self.clock_timestamps.len()
    }
}

/// Parser for picture timing SEI payloads.
#[derive(Debug, Default)]
pub struct PictureTimingParser {
    pic_struct_present: bool,
    cpb_dpb_delays_present: bool,
}

impl PictureTimingParser {
    /// Creates a new parser.
    ///
    /// `pic_struct_present` — from SPS VUI `pic_struct_present_flag`.
    /// `cpb_dpb_delays_present` — from HRD parameters presence.
    pub fn new(pic_struct_present: bool, cpb_dpb_delays_present: bool) -> Self {
        Self {
            pic_struct_present,
            cpb_dpb_delays_present,
        }
    }

    /// Parses a raw SEI payload byte slice into a [`PictureTiming`].
    ///
    /// Returns `None` if the slice is too short to contain valid data.
    pub fn parse(&self, data: &[u8]) -> Option<PictureTiming> {
        if data.is_empty() {
            return None;
        }
        // Simplified: treat first byte as encoded PicStruct (0–8).
        let raw_ps = data[0] & 0x0F;
        let pic_struct = if self.pic_struct_present {
            Some(Self::decode_pic_struct(raw_ps))
        } else {
            None
        };

        let (cpb_removal_delay, dpb_output_delay) =
            if self.cpb_dpb_delays_present && data.len() >= 5 {
                let cpb = u16::from_be_bytes([data[1], data[2]]) as u32;
                let dpb = u16::from_be_bytes([data[3], data[4]]) as u32;
                (cpb, dpb)
            } else {
                (0, 0)
            };

        let repeat_first_field = data.len() >= 6 && (data[5] & 0x01) != 0;

        Some(PictureTiming {
            cpb_dpb_delays_present: self.cpb_dpb_delays_present,
            cpb_removal_delay,
            dpb_output_delay,
            pic_struct,
            clock_timestamps: Vec::new(),
            repeat_first_field,
        })
    }

    fn decode_pic_struct(raw: u8) -> PicStruct {
        match raw {
            1 => PicStruct::TopField,
            2 => PicStruct::BottomField,
            3 => PicStruct::TopBottomField,
            4 => PicStruct::BottomTopField,
            5 => PicStruct::TopBottomTopField,
            6 => PicStruct::BottomTopBottomField,
            7 => PicStruct::FrameDoubling,
            8 => PicStruct::FrameTripling,
            _ => PicStruct::Frame,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pic_struct_progressive_frame_count_frame() {
        assert_eq!(PicStruct::Frame.progressive_frame_count(), 1);
    }

    #[test]
    fn test_pic_struct_progressive_frame_count_doubling() {
        assert_eq!(PicStruct::FrameDoubling.progressive_frame_count(), 2);
    }

    #[test]
    fn test_pic_struct_progressive_frame_count_tripling() {
        assert_eq!(PicStruct::FrameTripling.progressive_frame_count(), 3);
    }

    #[test]
    fn test_pic_struct_field_count_is_one() {
        assert_eq!(PicStruct::TopField.progressive_frame_count(), 1);
        assert_eq!(PicStruct::BottomField.progressive_frame_count(), 1);
    }

    #[test]
    fn test_pic_struct_is_progressive() {
        assert!(PicStruct::Frame.is_progressive());
        assert!(PicStruct::FrameDoubling.is_progressive());
        assert!(!PicStruct::TopField.is_progressive());
    }

    #[test]
    fn test_pic_struct_has_repeated_field() {
        assert!(PicStruct::TopBottomTopField.has_repeated_field());
        assert!(PicStruct::BottomTopBottomField.has_repeated_field());
        assert!(!PicStruct::TopBottomField.has_repeated_field());
    }

    #[test]
    fn test_clock_timestamp_to_ms_zero() {
        let ts = ClockTimestamp {
            hours: 0,
            minutes: 0,
            seconds: 0,
            n_frames: 0,
            discontinuity: false,
            ct_type: 0,
            nuit_field_based: false,
            counting_type: 0,
        };
        assert_eq!(ts.to_ms(), 0);
    }

    #[test]
    fn test_clock_timestamp_to_ms_one_hour() {
        let ts = ClockTimestamp {
            hours: 1,
            minutes: 0,
            seconds: 0,
            n_frames: 0,
            discontinuity: false,
            ct_type: 0,
            nuit_field_based: false,
            counting_type: 0,
        };
        assert_eq!(ts.to_ms(), 3_600_000);
    }

    #[test]
    fn test_clock_timestamp_to_ms_mixed() {
        let ts = ClockTimestamp {
            hours: 1,
            minutes: 30,
            seconds: 45,
            n_frames: 0,
            discontinuity: false,
            ct_type: 0,
            nuit_field_based: false,
            counting_type: 0,
        };
        // 1h + 30m + 45s = 5445s = 5_445_000 ms
        assert_eq!(ts.to_ms(), 5_445_000);
    }

    #[test]
    fn test_clock_timestamp_is_discontinuity() {
        let ts = ClockTimestamp {
            hours: 0,
            minutes: 0,
            seconds: 0,
            n_frames: 0,
            discontinuity: true,
            ct_type: 0,
            nuit_field_based: false,
            counting_type: 0,
        };
        assert!(ts.is_discontinuity());
    }

    #[test]
    fn test_picture_timing_new() {
        let pt = PictureTiming::new(100, 200);
        assert_eq!(pt.cpb_removal_delay, 100);
        assert_eq!(pt.dpb_output_delay, 200);
        assert!(!pt.is_repeat_first_field());
        assert_eq!(pt.timestamp_count(), 0);
    }

    #[test]
    fn test_picture_timing_repeat_first_field() {
        let mut pt = PictureTiming::new(0, 0);
        pt.repeat_first_field = true;
        assert!(pt.is_repeat_first_field());
    }

    #[test]
    fn test_parser_empty_data_returns_none() {
        let parser = PictureTimingParser::new(false, false);
        assert!(parser.parse(&[]).is_none());
    }

    #[test]
    fn test_parser_frame_pic_struct() {
        let parser = PictureTimingParser::new(true, false);
        // byte 0 lower nibble = 0 → Frame
        let pt = parser.parse(&[0x00]).expect("parse should succeed");
        assert_eq!(pt.pic_struct, Some(PicStruct::Frame));
    }

    #[test]
    fn test_parser_top_field_pic_struct() {
        let parser = PictureTimingParser::new(true, false);
        let pt = parser.parse(&[0x01]).expect("parse should succeed");
        assert_eq!(pt.pic_struct, Some(PicStruct::TopField));
    }

    #[test]
    fn test_parser_cpb_dpb_delays() {
        let parser = PictureTimingParser::new(false, true);
        // cpb = 0x0064 = 100, dpb = 0x00C8 = 200
        let data = [0x00, 0x00, 0x64, 0x00, 0xC8, 0x00];
        let pt = parser.parse(&data).expect("parse should succeed");
        assert_eq!(pt.cpb_removal_delay, 100);
        assert_eq!(pt.dpb_output_delay, 200);
    }

    #[test]
    fn test_parser_repeat_first_field_flag() {
        let parser = PictureTimingParser::new(false, true);
        let data = [0x00, 0x00, 0x01, 0x00, 0x02, 0x01];
        let pt = parser.parse(&data).expect("parse should succeed");
        assert!(pt.is_repeat_first_field());
    }

    #[test]
    fn test_parser_no_pic_struct_when_flag_false() {
        let parser = PictureTimingParser::new(false, false);
        let pt = parser.parse(&[0x07]).expect("parse should succeed");
        assert!(pt.pic_struct.is_none());
    }
}
