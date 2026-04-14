//! HDR metadata extraction from raw frame data.
//!
//! Parses HDR10 static metadata (SMPTE ST 2086 / CEA-861.3) and HDR10+
//! dynamic metadata from frame byte payloads.  The metadata is embedded in
//! a fixed-size header prepended to the pixel data by the OxiMedia muxer.
//!
//! # Header Layout
//!
//! The OxiMedia HDR metadata header is 24 bytes and occupies the very first
//! bytes of every HDR frame buffer:
//!
//! ```text
//! Offset  Size   Field
//! ──────  ────   ─────────────────────────────────────────────────
//!  0       4     Magic:  0x4F 0x58 0x48 0x44  ("OXHD")
//!  4       1     Version (must be 0x01)
//!  5       1     Transfer characteristic (0=PQ, 1=HLG, 2=SDR)
//!  6       2     Max display mastering luminance (cd/m²  ×  1)
//!  8       2     Min display mastering luminance (cd/m²  × 10000, i.e. 0.0001 precision)
//! 10       2     Max content light level (cd/m²)
//! 12       2     Max frame-average light level (cd/m²)
//! 14       6     Primary chromaticities: Rx,Ry,Gx,Gy,Bx,By  (u16 × 50000)
//! 20       4     White point: Wx,Wy  (u16 × 50000) × 2
//! ```
//!
//! The remaining bytes in the frame are pixel data.

#![allow(dead_code)]

/// Transfer characteristic of the HDR signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferCharacteristic {
    /// Perceptual Quantizer (SMPTE ST 2084 / HDR10 / BT.2100 PQ).
    Pq,
    /// Hybrid Log-Gamma (BBC / NHK BT.2100 HLG).
    Hlg,
    /// Standard dynamic range (SDR / BT.709 / BT.2020 SDR).
    Sdr,
}

/// Primary chromaticity coordinates (CIE 1931 xy).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chromaticity {
    /// CIE x coordinate.
    pub x: f32,
    /// CIE y coordinate.
    pub y: f32,
}

/// SMPTE ST 2086 mastering display colour volume metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MasteringDisplayMetadata {
    /// Red primary.
    pub red: Chromaticity,
    /// Green primary.
    pub green: Chromaticity,
    /// Blue primary.
    pub blue: Chromaticity,
    /// White point.
    pub white_point: Chromaticity,
    /// Maximum mastering display luminance in cd/m².
    pub max_luminance: f32,
    /// Minimum mastering display luminance in cd/m².
    pub min_luminance: f32,
}

/// Parsed HDR frame metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct HdrMetadata {
    /// Transfer characteristic of the signal.
    pub transfer: TransferCharacteristic,
    /// Mastering display colour volume (SMPTE ST 2086).
    pub mastering_display: MasteringDisplayMetadata,
    /// Maximum content light level (CLL) in cd/m².
    pub max_cll: u16,
    /// Maximum frame-average light level (FALL) in cd/m².
    pub max_fall: u16,
}

/// Magic bytes that identify the OxiMedia HDR metadata header.
const MAGIC: [u8; 4] = [0x4F, 0x58, 0x48, 0x44]; // "OXHD"
const HEADER_SIZE: usize = 24;

impl HdrMetadata {
    /// Attempts to extract HDR metadata from the beginning of `data`.
    ///
    /// Returns `None` when:
    /// - `data` is shorter than 24 bytes.
    /// - The magic bytes are absent.
    /// - The version byte is not `0x01`.
    /// - The transfer characteristic byte is unrecognised.
    #[must_use]
    pub fn from_frame_data(data: &[u8]) -> Option<Self> {
        if data.len() < HEADER_SIZE {
            return None;
        }
        // Validate magic.
        if data[0..4] != MAGIC {
            return None;
        }
        // Validate version.
        if data[4] != 0x01 {
            return None;
        }

        let transfer = match data[5] {
            0 => TransferCharacteristic::Pq,
            1 => TransferCharacteristic::Hlg,
            2 => TransferCharacteristic::Sdr,
            _ => return None,
        };

        let max_lum_raw = u16::from_be_bytes([data[6], data[7]]);
        let min_lum_raw = u16::from_be_bytes([data[8], data[9]]);
        let max_cll = u16::from_be_bytes([data[10], data[11]]);
        let max_fall = u16::from_be_bytes([data[12], data[13]]);

        // Primaries: 6 × u16 @ offset 14 (each divided by 50_000).
        let rx = u16::from_be_bytes([data[14], data[15]]) as f32 / 50_000.0;
        let ry = u16::from_be_bytes([data[16], data[17]]) as f32 / 50_000.0; // NOLINT
                                                                             // Avoid "data[18..19]" confusion:
        let gx = u16::from_be_bytes([data[16], data[17]]) as f32 / 50_000.0;
        let gy = u16::from_be_bytes([data[18], data[19]]) as f32 / 50_000.0;
        let bx = u16::from_be_bytes([data[18], data[19]]) as f32 / 50_000.0;
        let by_ = u16::from_be_bytes([data[20], data[21]]) as f32 / 50_000.0;

        // White point: 2 × u16 @ offset 20.
        let wx = u16::from_be_bytes([data[20], data[21]]) as f32 / 50_000.0;
        let wy = u16::from_be_bytes([data[22], data[23]]) as f32 / 50_000.0;

        let mastering_display = MasteringDisplayMetadata {
            red: Chromaticity { x: rx, y: ry },
            green: Chromaticity { x: gx, y: gy },
            blue: Chromaticity { x: bx, y: by_ },
            white_point: Chromaticity { x: wx, y: wy },
            max_luminance: max_lum_raw as f32,
            min_luminance: min_lum_raw as f32 / 10_000.0,
        };

        Some(Self {
            transfer,
            mastering_display,
            max_cll,
            max_fall,
        })
    }

    /// Serialises the metadata into a 24-byte header buffer.
    ///
    /// The buffer can be prepended to pixel data to create a valid OxiMedia
    /// HDR frame that [`from_frame_data`][Self::from_frame_data] can parse.
    #[must_use]
    pub fn to_header_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&MAGIC);
        buf[4] = 0x01;
        buf[5] = match self.transfer {
            TransferCharacteristic::Pq => 0,
            TransferCharacteristic::Hlg => 1,
            TransferCharacteristic::Sdr => 2,
        };
        let max_lum = self.mastering_display.max_luminance as u16;
        let min_lum = (self.mastering_display.min_luminance * 10_000.0) as u16;
        buf[6..8].copy_from_slice(&max_lum.to_be_bytes());
        buf[8..10].copy_from_slice(&min_lum.to_be_bytes());
        buf[10..12].copy_from_slice(&self.max_cll.to_be_bytes());
        buf[12..14].copy_from_slice(&self.max_fall.to_be_bytes());

        let md = &self.mastering_display;
        let rx = (md.red.x * 50_000.0) as u16;
        let ry = (md.red.y * 50_000.0) as u16;
        buf[14..16].copy_from_slice(&rx.to_be_bytes());
        buf[16..18].copy_from_slice(&ry.to_be_bytes());
        let gx = (md.green.x * 50_000.0) as u16;
        let gy = (md.green.y * 50_000.0) as u16;
        buf[16..18].copy_from_slice(&gx.to_be_bytes());
        buf[18..20].copy_from_slice(&gy.to_be_bytes());
        let bx = (md.blue.x * 50_000.0) as u16;
        let by_ = (md.blue.y * 50_000.0) as u16;
        buf[18..20].copy_from_slice(&bx.to_be_bytes());
        buf[20..22].copy_from_slice(&by_.to_be_bytes());
        let wx = (md.white_point.x * 50_000.0) as u16;
        let wy = (md.white_point.y * 50_000.0) as u16;
        buf[20..22].copy_from_slice(&wx.to_be_bytes());
        buf[22..24].copy_from_slice(&wy.to_be_bytes());
        buf
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_frame(transfer: u8) -> Vec<u8> {
        let mut h = [0u8; HEADER_SIZE];
        h[0..4].copy_from_slice(&MAGIC);
        h[4] = 0x01;
        h[5] = transfer;
        // max_lum = 1000 cd/m²
        h[6..8].copy_from_slice(&1000u16.to_be_bytes());
        // min_lum = 0.0001 → raw = 1
        h[8..10].copy_from_slice(&1u16.to_be_bytes());
        // max_cll = 900
        h[10..12].copy_from_slice(&900u16.to_be_bytes());
        // max_fall = 400
        h[12..14].copy_from_slice(&400u16.to_be_bytes());
        // Primaries and white point — all set to 0 for simplicity
        let mut v = h.to_vec();
        // Append some fake pixel data
        v.extend_from_slice(&[0u8; 100]);
        v
    }

    #[test]
    fn too_short_returns_none() {
        assert!(HdrMetadata::from_frame_data(&[0u8; 10]).is_none());
    }

    #[test]
    fn bad_magic_returns_none() {
        let mut data = make_test_frame(0);
        data[0] = 0xFF;
        assert!(HdrMetadata::from_frame_data(&data).is_none());
    }

    #[test]
    fn bad_version_returns_none() {
        let mut data = make_test_frame(0);
        data[4] = 0x02;
        assert!(HdrMetadata::from_frame_data(&data).is_none());
    }

    #[test]
    fn unknown_transfer_returns_none() {
        let data = make_test_frame(0xFF);
        assert!(HdrMetadata::from_frame_data(&data).is_none());
    }

    #[test]
    fn pq_transfer_parsed() {
        let data = make_test_frame(0);
        let meta = HdrMetadata::from_frame_data(&data).expect("valid");
        assert_eq!(meta.transfer, TransferCharacteristic::Pq);
        assert_eq!(meta.mastering_display.max_luminance, 1000.0);
        assert_eq!(meta.max_cll, 900);
        assert_eq!(meta.max_fall, 400);
    }

    #[test]
    fn hlg_transfer_parsed() {
        let data = make_test_frame(1);
        let meta = HdrMetadata::from_frame_data(&data).expect("valid");
        assert_eq!(meta.transfer, TransferCharacteristic::Hlg);
    }

    #[test]
    fn sdr_transfer_parsed() {
        let data = make_test_frame(2);
        let meta = HdrMetadata::from_frame_data(&data).expect("valid");
        assert_eq!(meta.transfer, TransferCharacteristic::Sdr);
    }

    #[test]
    fn exactly_24_bytes_parses() {
        let data = make_test_frame(0)[..HEADER_SIZE].to_vec();
        assert!(HdrMetadata::from_frame_data(&data).is_some());
    }
}
