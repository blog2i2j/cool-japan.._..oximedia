//! Initialization segment generation.
//!
//! Handles generation of MP4 initialization segments (ftyp + moov).

#![forbid(unsafe_code)]

use oximedia_core::{CodecId, OxiError, OxiResult};
use std::io::Write;

use crate::StreamInfo;

/// Builder for MP4 initialization segments.
#[derive(Debug)]
pub struct InitSegmentBuilder {
    streams: Vec<StreamInfo>,
    brand: String,
    compatible_brands: Vec<String>,
    creation_time: u64,
}

impl Default for InitSegmentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl InitSegmentBuilder {
    /// Creates a new initialization segment builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            streams: Vec::new(),
            brand: "iso5".into(),
            compatible_brands: vec!["iso5".into(), "iso6".into(), "mp41".into()],
            creation_time: 0,
        }
    }

    /// Adds a stream to the initialization segment.
    pub fn add_stream(&mut self, info: StreamInfo) -> &mut Self {
        self.streams.push(info);
        self
    }

    /// Sets the major brand.
    pub fn with_brand(&mut self, brand: impl Into<String>) -> &mut Self {
        self.brand = brand.into();
        self
    }

    /// Adds a compatible brand.
    pub fn add_compatible_brand(&mut self, brand: impl Into<String>) -> &mut Self {
        self.compatible_brands.push(brand.into());
        self
    }

    /// Sets the creation time (seconds since 1904-01-01 00:00:00 UTC).
    pub const fn with_creation_time(&mut self, time: u64) -> &mut Self {
        self.creation_time = time;
        self
    }

    /// Builds the initialization segment.
    ///
    /// # Errors
    ///
    /// Returns `Err` if no streams have been added or if writing fails.
    pub fn build(&self) -> OxiResult<Vec<u8>> {
        if self.streams.is_empty() {
            return Err(OxiError::InvalidData("No streams added".into()));
        }

        let mut data = Vec::new();

        // Write ftyp box
        self.write_ftyp(&mut data)?;

        // Write moov box
        self.write_moov(&mut data)?;

        Ok(data)
    }

    /// Writes the ftyp (file type) box.
    fn write_ftyp(&self, writer: &mut Vec<u8>) -> OxiResult<()> {
        let mut box_data = Vec::new();

        // Major brand
        box_data.extend_from_slice(self.brand.as_bytes().get(..4).unwrap_or(b"iso5"));

        // Minor version
        box_data.extend_from_slice(&0u32.to_be_bytes());

        // Compatible brands
        for brand in &self.compatible_brands {
            box_data.extend_from_slice(brand.as_bytes().get(..4).unwrap_or(b"    "));
        }

        self.write_box(writer, b"ftyp", &box_data)?;

        Ok(())
    }

    /// Writes the moov (movie) box.
    fn write_moov(&self, writer: &mut Vec<u8>) -> OxiResult<()> {
        let mut moov_data = Vec::new();

        // Write mvhd (movie header)
        self.write_mvhd(&mut moov_data)?;

        // Write mvex (movie extends) for fragmented MP4
        self.write_mvex(&mut moov_data)?;

        // Write trak (track) boxes for each stream
        for (index, stream) in self.streams.iter().enumerate() {
            self.write_trak(&mut moov_data, stream, index)?;
        }

        self.write_box(writer, b"moov", &moov_data)?;

        Ok(())
    }

    /// Writes the mvhd (movie header) box.
    fn write_mvhd(&self, writer: &mut Vec<u8>) -> OxiResult<()> {
        let mut mvhd_data = Vec::new();

        // Version and flags
        mvhd_data.extend_from_slice(&0u32.to_be_bytes());

        // Creation time
        mvhd_data.extend_from_slice(&self.creation_time.to_be_bytes()[4..]);

        // Modification time
        mvhd_data.extend_from_slice(&self.creation_time.to_be_bytes()[4..]);

        // Timescale
        mvhd_data.extend_from_slice(&1000u32.to_be_bytes());

        // Duration (0 for fragmented MP4)
        mvhd_data.extend_from_slice(&0u32.to_be_bytes());

        // Rate (1.0 in 16.16 fixed point)
        mvhd_data.extend_from_slice(&0x0001_0000u32.to_be_bytes());

        // Volume (1.0 in 8.8 fixed point)
        mvhd_data.extend_from_slice(&0x0100u16.to_be_bytes());

        // Reserved
        mvhd_data.extend_from_slice(&[0u8; 10]);

        // Matrix (identity matrix)
        let matrix: [i32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];
        for value in &matrix {
            mvhd_data.extend_from_slice(&value.to_be_bytes());
        }

        // Pre-defined
        mvhd_data.extend_from_slice(&[0u8; 24]);

        // Next track ID
        #[allow(clippy::cast_possible_truncation)]
        {
            mvhd_data.extend_from_slice(&((self.streams.len() + 1) as u32).to_be_bytes());
        }

        self.write_box(writer, b"mvhd", &mvhd_data)?;

        Ok(())
    }

    /// Writes the mvex (movie extends) box.
    fn write_mvex(&self, writer: &mut Vec<u8>) -> OxiResult<()> {
        let mut mvex_data = Vec::new();

        // Write trex (track extends) for each stream
        for (index, _) in self.streams.iter().enumerate() {
            self.write_trex(&mut mvex_data, index)?;
        }

        self.write_box(writer, b"mvex", &mvex_data)?;

        Ok(())
    }

    /// Writes the trex (track extends) box.
    fn write_trex(&self, writer: &mut Vec<u8>, track_index: usize) -> OxiResult<()> {
        let mut trex_data = Vec::new();

        // Version and flags
        trex_data.extend_from_slice(&0u32.to_be_bytes());

        // Track ID (1-based)
        #[allow(clippy::cast_possible_truncation)]
        {
            trex_data.extend_from_slice(&((track_index + 1) as u32).to_be_bytes());
        }

        // Default sample description index
        trex_data.extend_from_slice(&1u32.to_be_bytes());

        // Default sample duration
        trex_data.extend_from_slice(&0u32.to_be_bytes());

        // Default sample size
        trex_data.extend_from_slice(&0u32.to_be_bytes());

        // Default sample flags
        trex_data.extend_from_slice(&0u32.to_be_bytes());

        self.write_box(writer, b"trex", &trex_data)?;

        Ok(())
    }

    /// Writes a trak (track) box.
    fn write_trak(
        &self,
        writer: &mut Vec<u8>,
        stream: &StreamInfo,
        track_index: usize,
    ) -> OxiResult<()> {
        let mut trak_data = Vec::new();

        // Write tkhd (track header)
        self.write_tkhd(&mut trak_data, stream, track_index)?;

        // Write mdia (media)
        self.write_mdia(&mut trak_data, stream)?;

        self.write_box(writer, b"trak", &trak_data)?;

        Ok(())
    }

    /// Writes the tkhd (track header) box.
    fn write_tkhd(
        &self,
        writer: &mut Vec<u8>,
        _stream: &StreamInfo,
        track_index: usize,
    ) -> OxiResult<()> {
        let mut tkhd_data = Vec::new();

        // Version and flags (track enabled, in movie, in preview)
        tkhd_data.extend_from_slice(&0x0000_0007_u32.to_be_bytes());

        // Creation time
        tkhd_data.extend_from_slice(&self.creation_time.to_be_bytes()[4..]);

        // Modification time
        tkhd_data.extend_from_slice(&self.creation_time.to_be_bytes()[4..]);

        // Track ID (1-based)
        #[allow(clippy::cast_possible_truncation)]
        {
            tkhd_data.extend_from_slice(&((track_index + 1) as u32).to_be_bytes());
        }

        // Reserved
        tkhd_data.extend_from_slice(&[0u8; 4]);

        // Duration (0 for fragmented MP4)
        tkhd_data.extend_from_slice(&0u32.to_be_bytes());

        // Reserved
        tkhd_data.extend_from_slice(&[0u8; 8]);

        // Layer
        tkhd_data.extend_from_slice(&0u16.to_be_bytes());

        // Alternate group
        tkhd_data.extend_from_slice(&0u16.to_be_bytes());

        // Volume (1.0 for audio, 0.0 for video)
        tkhd_data.extend_from_slice(&0x0100u16.to_be_bytes());

        // Reserved
        tkhd_data.extend_from_slice(&[0u8; 2]);

        // Matrix (identity matrix)
        let matrix: [i32; 9] = [0x0001_0000, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000];
        for value in &matrix {
            tkhd_data.extend_from_slice(&value.to_be_bytes());
        }

        // Width and height (0 for audio)
        tkhd_data.extend_from_slice(&0u32.to_be_bytes());
        tkhd_data.extend_from_slice(&0u32.to_be_bytes());

        self.write_box(writer, b"tkhd", &tkhd_data)?;

        Ok(())
    }

    /// Writes the mdia (media) box.
    fn write_mdia(&self, writer: &mut Vec<u8>, stream: &StreamInfo) -> OxiResult<()> {
        let mut mdia_data = Vec::new();

        // Write mdhd (media header)
        self.write_mdhd(&mut mdia_data, stream)?;

        // Write hdlr (handler reference)
        self.write_hdlr(&mut mdia_data, stream)?;

        // Write minf (media information)
        self.write_minf(&mut mdia_data, stream)?;

        self.write_box(writer, b"mdia", &mdia_data)?;

        Ok(())
    }

    /// Writes the mdhd (media header) box.
    fn write_mdhd(&self, writer: &mut Vec<u8>, stream: &StreamInfo) -> OxiResult<()> {
        let mut mdhd_data = Vec::new();

        // Version and flags
        mdhd_data.extend_from_slice(&0u32.to_be_bytes());

        // Creation time
        mdhd_data.extend_from_slice(&self.creation_time.to_be_bytes()[4..]);

        // Modification time
        mdhd_data.extend_from_slice(&self.creation_time.to_be_bytes()[4..]);

        // Timescale
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            mdhd_data.extend_from_slice(&(stream.timebase.den as u32).to_be_bytes());
        }

        // Duration (0 for fragmented MP4)
        mdhd_data.extend_from_slice(&0u32.to_be_bytes());

        // Language (undetermined)
        mdhd_data.extend_from_slice(&0x55c4u16.to_be_bytes());

        // Pre-defined
        mdhd_data.extend_from_slice(&[0u8; 2]);

        self.write_box(writer, b"mdhd", &mdhd_data)?;

        Ok(())
    }

    /// Writes the hdlr (handler reference) box.
    fn write_hdlr(&self, writer: &mut Vec<u8>, stream: &StreamInfo) -> OxiResult<()> {
        let mut hdlr_data = Vec::new();

        // Version and flags
        hdlr_data.extend_from_slice(&0u32.to_be_bytes());

        // Pre-defined
        hdlr_data.extend_from_slice(&[0u8; 4]);

        // Handler type
        let handler = match stream.codec {
            CodecId::Av1 | CodecId::Vp8 | CodecId::Vp9 => b"vide",
            CodecId::Opus | CodecId::Flac | CodecId::Vorbis => b"soun",
            _ => b"data",
        };
        hdlr_data.extend_from_slice(handler);

        // Reserved
        hdlr_data.extend_from_slice(&[0u8; 12]);

        // Name (null-terminated)
        hdlr_data.push(0);

        self.write_box(writer, b"hdlr", &hdlr_data)?;

        Ok(())
    }

    /// Writes the minf (media information) box.
    fn write_minf(&self, writer: &mut Vec<u8>, _stream: &StreamInfo) -> OxiResult<()> {
        let minf_data = Vec::new();

        // In a real implementation, we would write vmhd/smhd and other boxes here
        // For now, just write a placeholder

        self.write_box(writer, b"minf", &minf_data)?;

        Ok(())
    }

    /// Writes a box with the given type and data.
    #[allow(
        clippy::unused_self,
        clippy::trivially_copy_pass_by_ref,
        clippy::cast_possible_truncation
    )]
    fn write_box(&self, writer: &mut Vec<u8>, box_type: &[u8; 4], data: &[u8]) -> OxiResult<()> {
        let size = (data.len() + 8) as u32;
        writer
            .write_all(&size.to_be_bytes())
            .map_err(|e: std::io::Error| OxiError::from(e))?;
        writer
            .write_all(box_type)
            .map_err(|e: std::io::Error| OxiError::from(e))?;
        writer
            .write_all(data)
            .map_err(|e: std::io::Error| OxiError::from(e))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::Rational;

    #[test]
    fn test_init_segment_builder() {
        let mut builder = InitSegmentBuilder::new();

        let mut stream_info = StreamInfo::new(0, CodecId::Opus, Rational::new(1, 48000));
        stream_info.codec_params = crate::stream::CodecParams::audio(48000, 2);

        builder
            .add_stream(stream_info)
            .with_brand("iso5")
            .add_compatible_brand("mp41");

        let result = builder.build();
        assert!(result.is_ok());

        let data = result.expect("operation should succeed");
        assert!(!data.is_empty());

        // Check for ftyp box
        assert_eq!(&data[4..8], b"ftyp");
    }

    #[test]
    fn test_init_segment_no_streams() {
        let builder = InitSegmentBuilder::new();
        let result = builder.build();
        assert!(result.is_err());
    }
}
