//! YUV4MPEG2 (Y4M) container muxer.
//!
//! Writes raw video frames in Y4M format, suitable for piping to other tools
//! or storing uncompressed video sequences.
//!
//! # Example
//!
//! ```ignore
//! use oximedia_container::mux::y4m::{Y4mMuxer, Y4mMuxerBuilder};
//! use oximedia_container::demux::y4m::{Y4mChroma, Y4mInterlace};
//!
//! let mut buf = Vec::new();
//! let mut muxer = Y4mMuxerBuilder::new(1920, 1080)
//!     .fps(30, 1)
//!     .chroma(Y4mChroma::C420jpeg)
//!     .interlace(Y4mInterlace::Progressive)
//!     .build(&mut buf)?;
//!
//! // Write frames
//! let frame = vec![0u8; 3110400]; // 1920x1080 420
//! muxer.write_frame(&frame)?;
//! muxer.finish()?;
//! ```

use std::io::Write;

use oximedia_core::{OxiError, OxiResult};

use crate::demux::y4m::{Y4mChroma, Y4mHeader, Y4mInterlace};

// ---------------------------------------------------------------------------
// Y4M Muxer Builder
// ---------------------------------------------------------------------------

/// Builder for configuring and creating a [`Y4mMuxer`].
///
/// At minimum, width and height must be provided. Other parameters
/// use sensible defaults if not specified.
///
/// # Defaults
///
/// - Frame rate: 25/1
/// - Chroma: 420jpeg
/// - Interlace: progressive
/// - Pixel aspect ratio: 0:0 (unknown)
pub struct Y4mMuxerBuilder {
    width: u32,
    height: u32,
    fps_num: u32,
    fps_den: u32,
    chroma: Y4mChroma,
    interlace: Y4mInterlace,
    par_num: u32,
    par_den: u32,
    comment: Option<String>,
}

impl Y4mMuxerBuilder {
    /// Creates a new builder with the given frame dimensions.
    ///
    /// # Arguments
    ///
    /// * `width` - Frame width in pixels (must be > 0)
    /// * `height` - Frame height in pixels (must be > 0)
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            fps_num: 25,
            fps_den: 1,
            chroma: Y4mChroma::C420jpeg,
            interlace: Y4mInterlace::Progressive,
            par_num: 0,
            par_den: 0,
            comment: None,
        }
    }

    /// Sets the frame rate as numerator/denominator.
    ///
    /// Common values:
    /// - 24/1 for 24fps
    /// - 25/1 for PAL
    /// - 30000/1001 for NTSC 29.97fps
    /// - 60/1 for 60fps
    #[must_use]
    pub const fn fps(mut self, num: u32, den: u32) -> Self {
        self.fps_num = num;
        self.fps_den = den;
        self
    }

    /// Sets the chroma subsampling format.
    #[must_use]
    pub const fn chroma(mut self, chroma: Y4mChroma) -> Self {
        self.chroma = chroma;
        self
    }

    /// Sets the interlacing mode.
    #[must_use]
    pub const fn interlace(mut self, interlace: Y4mInterlace) -> Self {
        self.interlace = interlace;
        self
    }

    /// Sets the pixel aspect ratio as numerator/denominator.
    ///
    /// Common values:
    /// - 1:1 for square pixels
    /// - 10:11 for NTSC DV (720x480)
    /// - 59:54 for PAL DV (720x576)
    #[must_use]
    pub const fn aspect_ratio(mut self, num: u32, den: u32) -> Self {
        self.par_num = num;
        self.par_den = den;
        self
    }

    /// Sets an optional comment to include in the header.
    #[must_use]
    pub fn comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = Some(comment.into());
        self
    }

    /// Builds the Y4M muxer and writes the header to the given writer.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Width or height is zero
    /// - Frame rate denominator is zero
    /// - Writing the header to the writer fails
    pub fn build<W: Write>(self, writer: W) -> OxiResult<Y4mMuxer<W>> {
        if self.width == 0 {
            return Err(OxiError::InvalidData(
                "Width must be greater than zero".into(),
            ));
        }
        if self.height == 0 {
            return Err(OxiError::InvalidData(
                "Height must be greater than zero".into(),
            ));
        }
        if self.fps_den == 0 {
            return Err(OxiError::InvalidData(
                "Frame rate denominator must be non-zero".into(),
            ));
        }

        let header = Y4mHeader {
            width: self.width,
            height: self.height,
            fps_num: self.fps_num,
            fps_den: self.fps_den,
            interlace: self.interlace,
            par_num: self.par_num,
            par_den: self.par_den,
            chroma: self.chroma,
            comment: self.comment,
        };

        let frame_size = header.frame_size();

        let mut muxer = Y4mMuxer {
            writer,
            header,
            header_written: false,
            frame_count: 0,
            frame_size,
        };

        muxer.write_header()?;

        Ok(muxer)
    }
}

// ---------------------------------------------------------------------------
// Y4M Muxer
// ---------------------------------------------------------------------------

/// Y4M (YUV4MPEG2) muxer for writing uncompressed video.
///
/// Writes a Y4M stream header followed by raw frame data. Each frame
/// is prefixed with a `FRAME` tag line.
///
/// Use [`Y4mMuxerBuilder`] to create instances.
pub struct Y4mMuxer<W: Write> {
    writer: W,
    header: Y4mHeader,
    header_written: bool,
    frame_count: u64,
    frame_size: usize,
}

impl<W: Write> Y4mMuxer<W> {
    /// Returns a reference to the Y4M header.
    #[must_use]
    pub fn header(&self) -> &Y4mHeader {
        &self.header
    }

    /// Returns the number of frames written so far.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Returns the expected frame size in bytes.
    #[must_use]
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Writes the Y4M stream header.
    ///
    /// This is called automatically by the builder. You should not need
    /// to call this directly.
    fn write_header(&mut self) -> OxiResult<()> {
        if self.header_written {
            return Err(OxiError::InvalidData("Header already written".into()));
        }

        let mut header_line = format!(
            "YUV4MPEG2 W{} H{} F{}:{} I{} C{}",
            self.header.width,
            self.header.height,
            self.header.fps_num,
            self.header.fps_den,
            self.header.interlace.as_char(),
            self.header.chroma.as_str(),
        );

        // Add pixel aspect ratio if specified
        if self.header.par_num > 0 && self.header.par_den > 0 {
            header_line.push_str(&format!(
                " A{}:{}",
                self.header.par_num, self.header.par_den
            ));
        }

        // Add comment if present
        if let Some(ref comment) = self.header.comment {
            header_line.push_str(&format!(" X{comment}"));
        }

        header_line.push('\n');

        self.writer
            .write_all(header_line.as_bytes())
            .map_err(OxiError::Io)?;

        self.header_written = true;
        Ok(())
    }

    /// Writes a single raw video frame.
    ///
    /// The data must be exactly `frame_size()` bytes of planar YUV data
    /// in the format specified by the header's chroma subsampling.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The header has not been written
    /// - The frame data size does not match the expected size
    /// - An I/O error occurs
    pub fn write_frame(&mut self, data: &[u8]) -> OxiResult<()> {
        if !self.header_written {
            return Err(OxiError::InvalidData("Header not yet written".into()));
        }

        if data.len() != self.frame_size {
            return Err(OxiError::BufferTooSmall {
                needed: self.frame_size,
                have: data.len(),
            });
        }

        self.writer.write_all(b"FRAME\n").map_err(OxiError::Io)?;
        self.writer.write_all(data).map_err(OxiError::Io)?;

        self.frame_count += 1;
        Ok(())
    }

    /// Flushes and finalizes the Y4M stream.
    ///
    /// Call this after writing all frames to ensure all data is flushed
    /// to the underlying writer.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing the writer fails.
    pub fn finish(&mut self) -> OxiResult<()> {
        self.writer.flush().map_err(OxiError::Io)?;
        Ok(())
    }

    /// Consumes the muxer and returns the underlying writer.
    ///
    /// The writer is flushed before being returned.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing the writer fails.
    pub fn into_writer(mut self) -> OxiResult<W> {
        self.writer.flush().map_err(OxiError::Io)?;
        Ok(self.writer)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demux::y4m::Y4mDemuxer;
    use std::io::Cursor;

    #[test]
    #[ignore]
    fn test_builder_defaults() {
        let mut buf = Vec::new();
        let muxer = Y4mMuxerBuilder::new(320, 240)
            .build(&mut buf)
            .expect("should build");
        assert_eq!(muxer.header().width, 320);
        assert_eq!(muxer.header().height, 240);
        assert_eq!(muxer.header().fps_num, 25);
        assert_eq!(muxer.header().fps_den, 1);
        assert_eq!(muxer.header().chroma, Y4mChroma::C420jpeg);
        assert_eq!(muxer.header().interlace, Y4mInterlace::Progressive);
        assert_eq!(muxer.frame_count(), 0);
    }

    #[test]
    #[ignore]
    fn test_builder_custom() {
        let mut buf = Vec::new();
        let muxer = Y4mMuxerBuilder::new(1920, 1080)
            .fps(30000, 1001)
            .chroma(Y4mChroma::C422)
            .interlace(Y4mInterlace::TopFirst)
            .aspect_ratio(1, 1)
            .comment("test")
            .build(&mut buf)
            .expect("should build");

        assert_eq!(muxer.header().fps_num, 30000);
        assert_eq!(muxer.header().fps_den, 1001);
        assert_eq!(muxer.header().chroma, Y4mChroma::C422);
        assert_eq!(muxer.header().interlace, Y4mInterlace::TopFirst);
        assert_eq!(muxer.header().par_num, 1);
        assert_eq!(muxer.header().par_den, 1);
    }

    #[test]
    #[ignore]
    fn test_builder_zero_width() {
        let mut buf = Vec::new();
        let result = Y4mMuxerBuilder::new(0, 240).build(&mut buf);
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_builder_zero_height() {
        let mut buf = Vec::new();
        let result = Y4mMuxerBuilder::new(320, 0).build(&mut buf);
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_builder_zero_fps_den() {
        let mut buf = Vec::new();
        let result = Y4mMuxerBuilder::new(320, 240).fps(30, 0).build(&mut buf);
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_write_header_format() {
        let mut buf = Vec::new();
        let _muxer = Y4mMuxerBuilder::new(1920, 1080)
            .fps(24, 1)
            .chroma(Y4mChroma::C444)
            .interlace(Y4mInterlace::Progressive)
            .build(&mut buf)
            .expect("should build");

        let header_str = String::from_utf8_lossy(&buf);
        assert!(header_str.starts_with("YUV4MPEG2 "));
        assert!(header_str.contains("W1920"));
        assert!(header_str.contains("H1080"));
        assert!(header_str.contains("F24:1"));
        assert!(header_str.contains("Ip"));
        assert!(header_str.contains("C444"));
        assert!(header_str.ends_with('\n'));
    }

    #[test]
    #[ignore]
    fn test_write_header_with_par() {
        let mut buf = Vec::new();
        let _muxer = Y4mMuxerBuilder::new(720, 480)
            .aspect_ratio(10, 11)
            .build(&mut buf)
            .expect("should build");

        let header_str = String::from_utf8_lossy(&buf);
        assert!(header_str.contains("A10:11"));
    }

    #[test]
    #[ignore]
    fn test_write_header_with_comment() {
        let mut buf = Vec::new();
        let _muxer = Y4mMuxerBuilder::new(320, 240)
            .comment("hello world")
            .build(&mut buf)
            .expect("should build");

        let header_str = String::from_utf8_lossy(&buf);
        assert!(header_str.contains("Xhello world"));
    }

    #[test]
    #[ignore]
    fn test_write_single_frame() {
        let mut buf = Vec::new();
        let mut muxer = Y4mMuxerBuilder::new(4, 4)
            .chroma(Y4mChroma::C420jpeg)
            .build(&mut buf)
            .expect("should build");

        // 4x4 420: 24 bytes
        let frame = vec![128u8; 24];
        muxer.write_frame(&frame).expect("should write frame");
        assert_eq!(muxer.frame_count(), 1);
        muxer.finish().expect("should finish");

        // Verify the output contains FRAME tag
        let output = String::from_utf8_lossy(&buf);
        assert!(output.contains("FRAME\n"));
    }

    #[test]
    #[ignore]
    fn test_write_frame_wrong_size() {
        let mut buf = Vec::new();
        let mut muxer = Y4mMuxerBuilder::new(4, 4)
            .chroma(Y4mChroma::C420jpeg)
            .build(&mut buf)
            .expect("should build");

        // Wrong size: 10 bytes instead of 24
        let frame = vec![0u8; 10];
        let result = muxer.write_frame(&frame);
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_write_multiple_frames() {
        let mut buf = Vec::new();
        let mut muxer = Y4mMuxerBuilder::new(4, 4)
            .chroma(Y4mChroma::C420jpeg)
            .fps(30, 1)
            .build(&mut buf)
            .expect("should build");

        for i in 0..5u8 {
            let frame = vec![i; 24];
            muxer.write_frame(&frame).expect("should write frame");
        }
        assert_eq!(muxer.frame_count(), 5);
        muxer.finish().expect("should finish");
    }

    // -- Round-trip tests --

    #[test]
    #[ignore]
    fn test_round_trip_420() {
        round_trip_test(8, 8, Y4mChroma::C420jpeg, 3);
    }

    #[test]
    #[ignore]
    fn test_round_trip_422() {
        round_trip_test(8, 8, Y4mChroma::C422, 2);
    }

    #[test]
    #[ignore]
    fn test_round_trip_444() {
        round_trip_test(4, 4, Y4mChroma::C444, 4);
    }

    #[test]
    #[ignore]
    fn test_round_trip_444alpha() {
        round_trip_test(4, 4, Y4mChroma::C444alpha, 2);
    }

    #[test]
    #[ignore]
    fn test_round_trip_mono() {
        round_trip_test(8, 8, Y4mChroma::Mono, 3);
    }

    #[test]
    #[ignore]
    fn test_round_trip_420mpeg2() {
        round_trip_test(16, 16, Y4mChroma::C420mpeg2, 2);
    }

    #[test]
    #[ignore]
    fn test_round_trip_420paldv() {
        round_trip_test(16, 16, Y4mChroma::C420paldv, 2);
    }

    #[test]
    #[ignore]
    fn test_round_trip_ntsc() {
        let mut buf = Vec::new();
        {
            let mut muxer = Y4mMuxerBuilder::new(1920, 1080)
                .fps(30000, 1001)
                .chroma(Y4mChroma::C420jpeg)
                .interlace(Y4mInterlace::TopFirst)
                .aspect_ratio(1, 1)
                .build(&mut buf)
                .expect("should build");

            let frame_size = muxer.frame_size();
            let frame = vec![42u8; frame_size];
            muxer.write_frame(&frame).expect("should write");
            muxer.finish().expect("should finish");
        }

        let mut demuxer = Y4mDemuxer::new(Cursor::new(buf)).expect("should parse");
        assert_eq!(demuxer.header().width, 1920);
        assert_eq!(demuxer.header().height, 1080);
        assert_eq!(demuxer.header().fps_num, 30000);
        assert_eq!(demuxer.header().fps_den, 1001);
        assert_eq!(demuxer.header().interlace, Y4mInterlace::TopFirst);
        assert_eq!(demuxer.header().par_num, 1);
        assert_eq!(demuxer.header().par_den, 1);

        let frame = demuxer
            .read_frame()
            .expect("should read")
            .expect("frame should exist");
        assert!(frame.iter().all(|&b| b == 42));
    }

    #[test]
    #[ignore]
    fn test_round_trip_odd_dimensions() {
        // Odd dimensions test chroma plane rounding
        round_trip_test(7, 5, Y4mChroma::C420jpeg, 2);
        round_trip_test(7, 5, Y4mChroma::C422, 2);
    }

    #[test]
    #[ignore]
    fn test_into_writer() {
        let buf = Vec::new();
        let muxer = Y4mMuxerBuilder::new(4, 4).build(buf).expect("should build");

        let recovered = muxer.into_writer().expect("should return writer");
        assert!(!recovered.is_empty());
    }

    /// Helper function for round-trip testing.
    fn round_trip_test(width: u32, height: u32, chroma: Y4mChroma, num_frames: usize) {
        let frame_size = chroma.bytes_per_frame(width, height);

        // Write
        let mut buf = Vec::new();
        {
            let mut muxer = Y4mMuxerBuilder::new(width, height)
                .fps(30, 1)
                .chroma(chroma)
                .interlace(Y4mInterlace::Progressive)
                .build(&mut buf)
                .expect("should build muxer");

            for i in 0..num_frames {
                let fill = (i & 0xFF) as u8;
                let frame = vec![fill; frame_size];
                muxer.write_frame(&frame).expect("should write frame");
            }
            muxer.finish().expect("should finish");
        }

        // Read back
        let mut demuxer = Y4mDemuxer::new(Cursor::new(buf)).expect("should parse header");
        assert_eq!(demuxer.width(), width);
        assert_eq!(demuxer.height(), height);
        assert_eq!(demuxer.chroma(), chroma);
        assert_eq!(demuxer.fps(), (30, 1));

        for i in 0..num_frames {
            let frame = demuxer
                .read_frame()
                .expect("should read frame")
                .unwrap_or_else(|| panic!("frame {i} should exist"));
            assert_eq!(frame.len(), frame_size, "frame {i} size mismatch");
            let expected_fill = (i & 0xFF) as u8;
            assert!(
                frame.iter().all(|&b| b == expected_fill),
                "frame {i} data mismatch"
            );
        }

        // Should be EOF now
        assert!(demuxer.read_frame().expect("should not error").is_none());
    }
}
