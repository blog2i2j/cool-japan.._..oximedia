//! YUV4MPEG2 (Y4M) container demuxer.
//!
//! Implements the Y4M uncompressed video sequence format.
//! Y4M is a simple format widely used for testing and piping raw YUV video
//! between tools.
//!
//! # Format Overview
//!
//! A Y4M file consists of a text header line starting with `YUV4MPEG2`,
//! followed by one or more frames, each prefixed with a `FRAME` tag line.
//! Frame data is raw planar YUV (Y plane, Cb plane, Cr plane).
//!
//! # Supported Chroma Formats
//!
//! - 420jpeg, 420mpeg2, 420paldv (4:2:0 variants)
//! - 422 (4:2:2)
//! - 444 (4:4:4)
//! - 444alpha (4:4:4 with alpha channel)
//! - mono (grayscale)
//!
//! # Example
//!
//! ```ignore
//! use oximedia_container::demux::y4m::Y4mDemuxer;
//!
//! let data = b"YUV4MPEG2 W4 H4 F30:1 Ip C420jpeg\nFRAME\n...";
//! let mut demuxer = Y4mDemuxer::new(std::io::Cursor::new(data));
//! let header = demuxer.header();
//! ```

use std::io::{BufRead, BufReader, Read};

use oximedia_core::{OxiError, OxiResult};

/// Y4M file header magic string.
const Y4M_MAGIC: &str = "YUV4MPEG2";

/// Y4M frame tag.
const FRAME_TAG: &str = "FRAME";

/// Maximum header line length to prevent malicious inputs.
const MAX_HEADER_LINE_LEN: usize = 4096;

// ---------------------------------------------------------------------------
// Chroma subsampling
// ---------------------------------------------------------------------------

/// Chroma subsampling mode for Y4M frames.
///
/// Determines the relationship between luma and chroma plane sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Y4mChroma {
    /// 4:2:0, JPEG/JFIF style (cosited horizontally, interstitial vertically).
    C420jpeg,
    /// 4:2:0, MPEG-2 style (cosited on left column).
    C420mpeg2,
    /// 4:2:0, PAL DV style.
    C420paldv,
    /// 4:2:2, chroma cosited horizontally.
    C422,
    /// 4:4:4, no subsampling.
    C444,
    /// 4:4:4 with alpha channel.
    C444alpha,
    /// Monochrome (luma only, no chroma planes).
    Mono,
}

impl Y4mChroma {
    /// Returns the total number of bytes per frame for the given dimensions.
    ///
    /// This accounts for all planes (Y, Cb, Cr, and alpha if applicable).
    #[must_use]
    pub fn bytes_per_frame(self, width: u32, height: u32) -> usize {
        let w = width as usize;
        let h = height as usize;
        let luma = w * h;
        match self {
            Self::C420jpeg | Self::C420mpeg2 | Self::C420paldv => {
                // Chroma planes are half width and half height each
                let chroma_w = (w + 1) / 2;
                let chroma_h = (h + 1) / 2;
                luma + 2 * chroma_w * chroma_h
            }
            Self::C422 => {
                // Chroma planes are half width, full height
                let chroma_w = (w + 1) / 2;
                luma + 2 * chroma_w * h
            }
            Self::C444 => {
                // Chroma planes are same size as luma
                luma * 3
            }
            Self::C444alpha => {
                // Y + Cb + Cr + Alpha, all full resolution
                luma * 4
            }
            Self::Mono => {
                // Luma only
                luma
            }
        }
    }

    /// Returns the Y4M string representation for writing headers.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::C420jpeg => "420jpeg",
            Self::C420mpeg2 => "420mpeg2",
            Self::C420paldv => "420paldv",
            Self::C422 => "422",
            Self::C444 => "444",
            Self::C444alpha => "444alpha",
            Self::Mono => "mono",
        }
    }

    /// Parses a chroma string from the Y4M header.
    ///
    /// Returns `None` for unrecognized strings.
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "420jpeg" | "420" => Some(Self::C420jpeg),
            "420mpeg2" => Some(Self::C420mpeg2),
            "420paldv" => Some(Self::C420paldv),
            "422" => Some(Self::C422),
            "444" => Some(Self::C444),
            "444alpha" => Some(Self::C444alpha),
            "mono" => Some(Self::Mono),
            _ => None,
        }
    }
}

impl std::fmt::Display for Y4mChroma {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Interlacing mode
// ---------------------------------------------------------------------------

/// Interlacing mode for Y4M video.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Y4mInterlace {
    /// Progressive scan (non-interlaced).
    Progressive,
    /// Top-field-first interlaced.
    TopFirst,
    /// Bottom-field-first interlaced.
    BottomFirst,
    /// Mixed interlaced and progressive content.
    Mixed,
}

impl Y4mInterlace {
    /// Returns the Y4M character representation.
    #[must_use]
    pub const fn as_char(self) -> char {
        match self {
            Self::Progressive => 'p',
            Self::TopFirst => 't',
            Self::BottomFirst => 'b',
            Self::Mixed => 'm',
        }
    }

    /// Parses an interlace character from the Y4M header.
    #[must_use]
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            'p' => Some(Self::Progressive),
            't' => Some(Self::TopFirst),
            'b' => Some(Self::BottomFirst),
            'm' => Some(Self::Mixed),
            _ => None,
        }
    }
}

impl std::fmt::Display for Y4mInterlace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_char())
    }
}

// ---------------------------------------------------------------------------
// Y4M Header
// ---------------------------------------------------------------------------

/// Parsed Y4M file header containing all stream parameters.
///
/// All Y4M files must specify at least width and height.
/// Other parameters have defaults if omitted from the header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Y4mHeader {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame rate numerator.
    pub fps_num: u32,
    /// Frame rate denominator.
    pub fps_den: u32,
    /// Interlacing mode.
    pub interlace: Y4mInterlace,
    /// Pixel aspect ratio numerator.
    pub par_num: u32,
    /// Pixel aspect ratio denominator.
    pub par_den: u32,
    /// Chroma subsampling format.
    pub chroma: Y4mChroma,
    /// Optional comment from the header (X parameter).
    pub comment: Option<String>,
}

impl Y4mHeader {
    /// Returns the number of bytes per raw frame.
    #[must_use]
    pub fn frame_size(&self) -> usize {
        self.chroma.bytes_per_frame(self.width, self.height)
    }

    /// Returns the frame rate as a floating-point value.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn fps(&self) -> f64 {
        if self.fps_den == 0 {
            0.0
        } else {
            self.fps_num as f64 / self.fps_den as f64
        }
    }

    /// Returns the pixel aspect ratio as a floating-point value.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn pixel_aspect_ratio(&self) -> f64 {
        if self.par_den == 0 {
            1.0
        } else {
            self.par_num as f64 / self.par_den as f64
        }
    }
}

impl Default for Y4mHeader {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            fps_num: 25,
            fps_den: 1,
            interlace: Y4mInterlace::Progressive,
            par_num: 0,
            par_den: 0,
            chroma: Y4mChroma::C420jpeg,
            comment: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Header parsing
// ---------------------------------------------------------------------------

/// Parses a Y4M header line into a [`Y4mHeader`].
///
/// The line must start with `YUV4MPEG2` followed by space-separated parameters.
/// At minimum, `W<width>` and `H<height>` must be present.
fn parse_header(line: &str) -> OxiResult<Y4mHeader> {
    let line = line.trim_end_matches(['\n', '\r']);

    if !line.starts_with(Y4M_MAGIC) {
        return Err(OxiError::Parse {
            offset: 0,
            message: format!(
                "Expected Y4M magic '{}', got '{}'",
                Y4M_MAGIC,
                &line[..line.len().min(20)]
            ),
        });
    }

    let params_str = &line[Y4M_MAGIC.len()..];
    let mut header = Y4mHeader::default();
    let mut found_width = false;
    let mut found_height = false;

    for token in params_str.split_whitespace() {
        if token.is_empty() {
            continue;
        }

        let tag = token.as_bytes()[0];
        let value = &token[1..];

        match tag {
            b'W' => {
                header.width = value.parse::<u32>().map_err(|_| OxiError::Parse {
                    offset: 0,
                    message: format!("Invalid width: '{value}'"),
                })?;
                if header.width == 0 {
                    return Err(OxiError::Parse {
                        offset: 0,
                        message: "Width must be greater than zero".into(),
                    });
                }
                found_width = true;
            }
            b'H' => {
                header.height = value.parse::<u32>().map_err(|_| OxiError::Parse {
                    offset: 0,
                    message: format!("Invalid height: '{value}'"),
                })?;
                if header.height == 0 {
                    return Err(OxiError::Parse {
                        offset: 0,
                        message: "Height must be greater than zero".into(),
                    });
                }
                found_height = true;
            }
            b'F' => {
                let (num, den) = parse_ratio(value).ok_or_else(|| OxiError::Parse {
                    offset: 0,
                    message: format!("Invalid frame rate: '{value}'"),
                })?;
                if den == 0 {
                    return Err(OxiError::Parse {
                        offset: 0,
                        message: "Frame rate denominator must be non-zero".into(),
                    });
                }
                header.fps_num = num;
                header.fps_den = den;
            }
            b'I' => {
                if let Some(first_char) = value.chars().next() {
                    header.interlace =
                        Y4mInterlace::from_char(first_char).ok_or_else(|| OxiError::Parse {
                            offset: 0,
                            message: format!("Invalid interlace mode: '{first_char}'"),
                        })?;
                }
            }
            b'A' => {
                let (num, den) = parse_ratio(value).ok_or_else(|| OxiError::Parse {
                    offset: 0,
                    message: format!("Invalid pixel aspect ratio: '{value}'"),
                })?;
                header.par_num = num;
                header.par_den = den;
            }
            b'C' => {
                header.chroma = Y4mChroma::from_str(value).ok_or_else(|| OxiError::Parse {
                    offset: 0,
                    message: format!("Unknown chroma subsampling: '{value}'"),
                })?;
            }
            b'X' => {
                // Comment parameter - everything after 'X' is the comment
                header.comment = Some(value.to_string());
            }
            _ => {
                // Unknown parameters are silently ignored per spec
            }
        }
    }

    if !found_width {
        return Err(OxiError::Parse {
            offset: 0,
            message: "Missing required width parameter (W)".into(),
        });
    }
    if !found_height {
        return Err(OxiError::Parse {
            offset: 0,
            message: "Missing required height parameter (H)".into(),
        });
    }

    Ok(header)
}

/// Parses a ratio string like "30:1" or "30000:1001".
fn parse_ratio(s: &str) -> Option<(u32, u32)> {
    let mut parts = s.splitn(2, ':');
    let num_str = parts.next()?;
    let den_str = parts.next()?;
    let num = num_str.parse::<u32>().ok()?;
    let den = den_str.parse::<u32>().ok()?;
    Some((num, den))
}

/// Parses frame parameters from a FRAME line.
///
/// Frame lines start with "FRAME" and may contain optional `X` comment params.
/// Returns `Ok(())` if the line is a valid frame tag, `Err` otherwise.
fn validate_frame_tag(line: &str) -> OxiResult<()> {
    let trimmed = line.trim_end_matches(['\n', '\r']);
    if !trimmed.starts_with(FRAME_TAG) {
        return Err(OxiError::Parse {
            offset: 0,
            message: format!(
                "Expected FRAME tag, got '{}'",
                &trimmed[..trimmed.len().min(20)]
            ),
        });
    }

    // After "FRAME" there must be either nothing, a space, or parameters
    let rest = &trimmed[FRAME_TAG.len()..];
    if !rest.is_empty() && !rest.starts_with(' ') {
        return Err(OxiError::Parse {
            offset: 0,
            message: format!("Invalid FRAME tag: '{}'", &trimmed[..trimmed.len().min(30)]),
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Y4M Demuxer
// ---------------------------------------------------------------------------

/// Y4M (YUV4MPEG2) demuxer.
///
/// Reads a Y4M stream and provides access to the header and raw video frames.
///
/// # Lifecycle
///
/// 1. Create with `Y4mDemuxer::new(reader)`
/// 2. Access header via `header()` (parsed automatically on construction)
/// 3. Call `read_frame()` in a loop to get raw YUV data
///
/// # Example
///
/// ```ignore
/// use oximedia_container::demux::y4m::Y4mDemuxer;
/// use std::io::Cursor;
///
/// let data = build_y4m_data(); // your Y4M bytes
/// let mut demuxer = Y4mDemuxer::new(Cursor::new(data))?;
///
/// println!("{}x{} @ {} fps",
///     demuxer.width(), demuxer.height(), demuxer.header().fps());
///
/// while let Some(frame) = demuxer.read_frame()? {
///     process_frame(&frame);
/// }
/// ```
pub struct Y4mDemuxer<R: Read> {
    reader: BufReader<R>,
    header: Y4mHeader,
    frame_count: u64,
    frame_size: usize,
    eof: bool,
    line_buf: String,
}

impl<R: Read> Y4mDemuxer<R> {
    /// Creates a new Y4M demuxer by parsing the stream header.
    ///
    /// The header line is read and parsed immediately. If the header
    /// is malformed or the stream is empty, an error is returned.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The stream cannot be read
    /// - The header does not start with `YUV4MPEG2`
    /// - Required parameters (W, H) are missing
    /// - Parameter values are invalid
    pub fn new(reader: R) -> OxiResult<Self> {
        let mut buf_reader = BufReader::new(reader);
        let mut line = String::with_capacity(256);

        // Read the header line
        let bytes_read = buf_reader.read_line(&mut line).map_err(OxiError::Io)?;

        if bytes_read == 0 {
            return Err(OxiError::UnexpectedEof);
        }

        if bytes_read > MAX_HEADER_LINE_LEN {
            return Err(OxiError::Parse {
                offset: 0,
                message: format!(
                    "Header line too long: {bytes_read} bytes (max {MAX_HEADER_LINE_LEN})"
                ),
            });
        }

        let header = parse_header(&line)?;
        let frame_size = header.frame_size();

        Ok(Self {
            reader: buf_reader,
            header,
            frame_count: 0,
            frame_size,
            eof: false,
            line_buf: String::with_capacity(128),
        })
    }

    /// Returns a reference to the parsed Y4M header.
    #[must_use]
    pub fn header(&self) -> &Y4mHeader {
        &self.header
    }

    /// Returns the frame width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.header.width
    }

    /// Returns the frame height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.header.height
    }

    /// Returns the frame rate as (numerator, denominator).
    #[must_use]
    pub fn fps(&self) -> (u32, u32) {
        (self.header.fps_num, self.header.fps_den)
    }

    /// Returns the chroma subsampling format.
    #[must_use]
    pub fn chroma(&self) -> Y4mChroma {
        self.header.chroma
    }

    /// Returns the number of frames read so far.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Returns the number of bytes per raw frame.
    #[must_use]
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Returns true if end of stream has been reached.
    #[must_use]
    pub fn is_eof(&self) -> bool {
        self.eof
    }

    /// Reads the next raw video frame.
    ///
    /// Returns `Ok(Some(data))` with the raw planar YUV data for one frame,
    /// or `Ok(None)` if end of stream is reached.
    ///
    /// The frame data layout is:
    /// - Y plane: `width * height` bytes
    /// - Cb plane: size depends on chroma subsampling
    /// - Cr plane: same size as Cb
    /// - Alpha plane (only for C444alpha): `width * height` bytes
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The FRAME tag is missing or invalid
    /// - The frame data is truncated
    /// - An I/O error occurs
    pub fn read_frame(&mut self) -> OxiResult<Option<Vec<u8>>> {
        if self.eof {
            return Ok(None);
        }

        // Read the FRAME tag line
        self.line_buf.clear();
        let bytes_read = self
            .reader
            .read_line(&mut self.line_buf)
            .map_err(OxiError::Io)?;

        if bytes_read == 0 {
            self.eof = true;
            return Ok(None);
        }

        validate_frame_tag(&self.line_buf)?;

        // Read the raw frame data
        let mut frame_data = vec![0u8; self.frame_size];
        self.read_exact_into(&mut frame_data)?;

        self.frame_count += 1;
        Ok(Some(frame_data))
    }

    /// Reads all remaining frames into a vector.
    ///
    /// This is a convenience method that calls `read_frame()` in a loop.
    /// For large files, consider using `read_frame()` directly to avoid
    /// loading everything into memory.
    ///
    /// # Errors
    ///
    /// Returns an error if any frame read fails.
    pub fn read_all_frames(&mut self) -> OxiResult<Vec<Vec<u8>>> {
        let mut frames = Vec::new();
        while let Some(frame) = self.read_frame()? {
            frames.push(frame);
        }
        Ok(frames)
    }

    /// Reads exactly `buf.len()` bytes from the reader.
    fn read_exact_into(&mut self, buf: &mut [u8]) -> OxiResult<()> {
        let mut read = 0;
        while read < buf.len() {
            let n = self.reader.read(&mut buf[read..]).map_err(OxiError::Io)?;
            if n == 0 {
                self.eof = true;
                return Err(OxiError::Parse {
                    offset: 0,
                    message: format!(
                        "Truncated frame data: expected {} bytes, got {}",
                        buf.len(),
                        read
                    ),
                });
            }
            read += n;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // -- Header parsing tests --

    #[test]
    #[ignore]
    fn test_parse_header_minimal() {
        let header = parse_header("YUV4MPEG2 W320 H240").expect("should parse");
        assert_eq!(header.width, 320);
        assert_eq!(header.height, 240);
        // Defaults
        assert_eq!(header.fps_num, 25);
        assert_eq!(header.fps_den, 1);
        assert_eq!(header.interlace, Y4mInterlace::Progressive);
        assert_eq!(header.chroma, Y4mChroma::C420jpeg);
    }

    #[test]
    #[ignore]
    fn test_parse_header_full() {
        let line = "YUV4MPEG2 W1920 H1080 F30000:1001 It A1:1 C422 Xtest_comment";
        let header = parse_header(line).expect("should parse");
        assert_eq!(header.width, 1920);
        assert_eq!(header.height, 1080);
        assert_eq!(header.fps_num, 30000);
        assert_eq!(header.fps_den, 1001);
        assert_eq!(header.interlace, Y4mInterlace::TopFirst);
        assert_eq!(header.par_num, 1);
        assert_eq!(header.par_den, 1);
        assert_eq!(header.chroma, Y4mChroma::C422);
        assert_eq!(header.comment.as_deref(), Some("test_comment"));
    }

    #[test]
    #[ignore]
    fn test_parse_header_all_chroma() {
        for (cs, expected) in &[
            ("420jpeg", Y4mChroma::C420jpeg),
            ("420", Y4mChroma::C420jpeg),
            ("420mpeg2", Y4mChroma::C420mpeg2),
            ("420paldv", Y4mChroma::C420paldv),
            ("422", Y4mChroma::C422),
            ("444", Y4mChroma::C444),
            ("444alpha", Y4mChroma::C444alpha),
            ("mono", Y4mChroma::Mono),
        ] {
            let line = format!("YUV4MPEG2 W8 H8 C{cs}");
            let header = parse_header(&line).unwrap_or_else(|e| {
                panic!("Failed to parse chroma '{cs}': {e}");
            });
            assert_eq!(header.chroma, *expected, "chroma mismatch for '{cs}'");
        }
    }

    #[test]
    #[ignore]
    fn test_parse_header_all_interlace() {
        for (c, expected) in &[
            ('p', Y4mInterlace::Progressive),
            ('t', Y4mInterlace::TopFirst),
            ('b', Y4mInterlace::BottomFirst),
            ('m', Y4mInterlace::Mixed),
        ] {
            let line = format!("YUV4MPEG2 W8 H8 I{c}");
            let header = parse_header(&line).unwrap_or_else(|e| {
                panic!("Failed to parse interlace '{c}': {e}");
            });
            assert_eq!(header.interlace, *expected, "interlace mismatch for '{c}'");
        }
    }

    #[test]
    #[ignore]
    fn test_parse_header_missing_magic() {
        let result = parse_header("NOT_Y4M W320 H240");
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_parse_header_missing_width() {
        let result = parse_header("YUV4MPEG2 H240");
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_parse_header_missing_height() {
        let result = parse_header("YUV4MPEG2 W320");
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_parse_header_zero_width() {
        let result = parse_header("YUV4MPEG2 W0 H240");
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_parse_header_zero_height() {
        let result = parse_header("YUV4MPEG2 W320 H0");
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_parse_header_invalid_width() {
        let result = parse_header("YUV4MPEG2 Wabc H240");
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_parse_header_invalid_fps() {
        let result = parse_header("YUV4MPEG2 W320 H240 Fabc:def");
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_parse_header_zero_fps_den() {
        let result = parse_header("YUV4MPEG2 W320 H240 F30:0");
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_parse_header_unknown_chroma() {
        let result = parse_header("YUV4MPEG2 W320 H240 Cunknown");
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_parse_header_invalid_interlace() {
        let result = parse_header("YUV4MPEG2 W320 H240 Ix");
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_parse_header_unknown_params_ignored() {
        // Unknown parameters should be silently ignored
        let header = parse_header("YUV4MPEG2 W8 H8 Zunknown").expect("should parse");
        assert_eq!(header.width, 8);
        assert_eq!(header.height, 8);
    }

    #[test]
    #[ignore]
    fn test_parse_header_trailing_newline() {
        let header = parse_header("YUV4MPEG2 W320 H240\n").expect("should parse");
        assert_eq!(header.width, 320);
        assert_eq!(header.height, 240);
    }

    // -- Chroma bytes_per_frame tests --

    #[test]
    #[ignore]
    fn test_chroma_bytes_per_frame_420() {
        // 8x8: Y=64, Cb=16, Cr=16 = 96
        assert_eq!(Y4mChroma::C420jpeg.bytes_per_frame(8, 8), 96);
        // 1920x1080: Y=2073600, Cb=518400, Cr=518400 = 3110400
        assert_eq!(Y4mChroma::C420jpeg.bytes_per_frame(1920, 1080), 3110400);
    }

    #[test]
    #[ignore]
    fn test_chroma_bytes_per_frame_420_odd() {
        // 7x7: Y=49, chroma_w=4, chroma_h=4, Cb=16, Cr=16 = 81
        assert_eq!(Y4mChroma::C420jpeg.bytes_per_frame(7, 7), 81);
    }

    #[test]
    #[ignore]
    fn test_chroma_bytes_per_frame_422() {
        // 8x8: Y=64, Cb=32, Cr=32 = 128
        assert_eq!(Y4mChroma::C422.bytes_per_frame(8, 8), 128);
    }

    #[test]
    #[ignore]
    fn test_chroma_bytes_per_frame_444() {
        // 8x8: Y=64, Cb=64, Cr=64 = 192
        assert_eq!(Y4mChroma::C444.bytes_per_frame(8, 8), 192);
    }

    #[test]
    #[ignore]
    fn test_chroma_bytes_per_frame_444alpha() {
        // 8x8: Y=64, Cb=64, Cr=64, A=64 = 256
        assert_eq!(Y4mChroma::C444alpha.bytes_per_frame(8, 8), 256);
    }

    #[test]
    #[ignore]
    fn test_chroma_bytes_per_frame_mono() {
        // 8x8: Y=64
        assert_eq!(Y4mChroma::Mono.bytes_per_frame(8, 8), 64);
    }

    // -- Demuxer tests --

    /// Helper: build a minimal Y4M byte stream with given dimensions, chroma, and frames.
    fn build_y4m_stream(width: u32, height: u32, chroma: Y4mChroma, frame_count: usize) -> Vec<u8> {
        let mut data = Vec::new();
        // Header
        let header_line = format!(
            "YUV4MPEG2 W{width} H{height} F30:1 Ip C{}\n",
            chroma.as_str()
        );
        data.extend_from_slice(header_line.as_bytes());

        let frame_size = chroma.bytes_per_frame(width, height);
        for i in 0..frame_count {
            data.extend_from_slice(b"FRAME\n");
            // Fill frame with a repeating byte pattern for verification
            let fill_byte = (i & 0xFF) as u8;
            data.extend(std::iter::repeat(fill_byte).take(frame_size));
        }
        data
    }

    #[test]
    #[ignore]
    fn test_demuxer_empty_stream() {
        let result = Y4mDemuxer::new(Cursor::new(Vec::<u8>::new()));
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_demuxer_header_only() {
        let data = b"YUV4MPEG2 W8 H8 F25:1 Ip C420jpeg\n";
        let mut demuxer = Y4mDemuxer::new(Cursor::new(data.to_vec())).expect("should parse header");
        assert_eq!(demuxer.width(), 8);
        assert_eq!(demuxer.height(), 8);
        assert_eq!(demuxer.fps(), (25, 1));
        assert_eq!(demuxer.chroma(), Y4mChroma::C420jpeg);
        assert_eq!(demuxer.frame_count(), 0);

        // No frames -> returns None
        let frame = demuxer.read_frame().expect("should not error");
        assert!(frame.is_none());
        assert!(demuxer.is_eof());
    }

    #[test]
    #[ignore]
    fn test_demuxer_single_frame_420() {
        let data = build_y4m_stream(4, 4, Y4mChroma::C420jpeg, 1);
        let mut demuxer = Y4mDemuxer::new(Cursor::new(data)).expect("should parse header");

        let frame = demuxer.read_frame().expect("should read frame");
        assert!(frame.is_some());
        let frame = frame.expect("frame should be Some");

        // 4x4 420: Y=16, Cb=4, Cr=4 = 24
        assert_eq!(frame.len(), 24);
        // All bytes should be 0 (first frame fill pattern)
        assert!(frame.iter().all(|&b| b == 0));

        assert_eq!(demuxer.frame_count(), 1);

        // No more frames
        let frame = demuxer.read_frame().expect("should not error");
        assert!(frame.is_none());
    }

    #[test]
    #[ignore]
    fn test_demuxer_multiple_frames() {
        let num_frames = 5;
        let data = build_y4m_stream(4, 4, Y4mChroma::C420jpeg, num_frames);
        let mut demuxer = Y4mDemuxer::new(Cursor::new(data)).expect("should parse header");

        for i in 0..num_frames {
            let frame = demuxer
                .read_frame()
                .expect("should read frame")
                .unwrap_or_else(|| panic!("frame {i} should exist"));
            // Each frame filled with its index as byte
            assert!(
                frame.iter().all(|&b| b == i as u8),
                "frame {i} data mismatch"
            );
        }

        assert_eq!(demuxer.frame_count(), num_frames as u64);
        assert!(demuxer.read_frame().expect("should not error").is_none());
    }

    #[test]
    #[ignore]
    fn test_demuxer_422() {
        let data = build_y4m_stream(8, 8, Y4mChroma::C422, 2);
        let mut demuxer = Y4mDemuxer::new(Cursor::new(data)).expect("should parse header");
        assert_eq!(demuxer.chroma(), Y4mChroma::C422);

        let frame = demuxer
            .read_frame()
            .expect("should read")
            .expect("frame should exist");
        // 8x8 422: Y=64, Cb=32, Cr=32 = 128
        assert_eq!(frame.len(), 128);
    }

    #[test]
    #[ignore]
    fn test_demuxer_444() {
        let data = build_y4m_stream(4, 4, Y4mChroma::C444, 1);
        let mut demuxer = Y4mDemuxer::new(Cursor::new(data)).expect("should parse header");

        let frame = demuxer
            .read_frame()
            .expect("should read")
            .expect("frame should exist");
        // 4x4 444: 16*3 = 48
        assert_eq!(frame.len(), 48);
    }

    #[test]
    #[ignore]
    fn test_demuxer_mono() {
        let data = build_y4m_stream(4, 4, Y4mChroma::Mono, 1);
        let mut demuxer = Y4mDemuxer::new(Cursor::new(data)).expect("should parse header");

        let frame = demuxer
            .read_frame()
            .expect("should read")
            .expect("frame should exist");
        // 4x4 mono: 16
        assert_eq!(frame.len(), 16);
    }

    #[test]
    #[ignore]
    fn test_demuxer_444alpha() {
        let data = build_y4m_stream(4, 4, Y4mChroma::C444alpha, 1);
        let mut demuxer = Y4mDemuxer::new(Cursor::new(data)).expect("should parse header");

        let frame = demuxer
            .read_frame()
            .expect("should read")
            .expect("frame should exist");
        // 4x4 444alpha: 16*4 = 64
        assert_eq!(frame.len(), 64);
    }

    #[test]
    #[ignore]
    fn test_demuxer_truncated_frame() {
        let mut data = Vec::new();
        data.extend_from_slice(b"YUV4MPEG2 W4 H4 C420jpeg\n");
        data.extend_from_slice(b"FRAME\n");
        // Only provide 10 bytes when 24 are needed
        data.extend_from_slice(&[0u8; 10]);

        let mut demuxer = Y4mDemuxer::new(Cursor::new(data)).expect("should parse header");
        let result = demuxer.read_frame();
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_demuxer_missing_frame_tag() {
        let mut data = Vec::new();
        data.extend_from_slice(b"YUV4MPEG2 W4 H4 C420jpeg\n");
        data.extend_from_slice(b"NOT_FRAME\n");
        data.extend_from_slice(&[0u8; 24]);

        let mut demuxer = Y4mDemuxer::new(Cursor::new(data)).expect("should parse header");
        let result = demuxer.read_frame();
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_demuxer_frame_with_params() {
        let mut data = Vec::new();
        data.extend_from_slice(b"YUV4MPEG2 W4 H4 C420jpeg\n");
        data.extend_from_slice(b"FRAME Xsome_comment\n");
        data.extend_from_slice(&[128u8; 24]);

        let mut demuxer = Y4mDemuxer::new(Cursor::new(data)).expect("should parse header");
        let frame = demuxer
            .read_frame()
            .expect("should read frame")
            .expect("frame should exist");
        assert_eq!(frame.len(), 24);
        assert!(frame.iter().all(|&b| b == 128));
    }

    #[test]
    #[ignore]
    fn test_demuxer_read_all_frames() {
        let data = build_y4m_stream(4, 4, Y4mChroma::C420jpeg, 3);
        let mut demuxer = Y4mDemuxer::new(Cursor::new(data)).expect("should parse header");

        let frames = demuxer.read_all_frames().expect("should read all");
        assert_eq!(frames.len(), 3);
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(frame.len(), 24);
            assert!(
                frame.iter().all(|&b| b == i as u8),
                "frame {i} data mismatch"
            );
        }
    }

    #[test]
    #[ignore]
    fn test_demuxer_ntsc_framerate() {
        let data = b"YUV4MPEG2 W1920 H1080 F30000:1001\n";
        let demuxer = Y4mDemuxer::new(Cursor::new(data.to_vec())).expect("should parse");
        assert_eq!(demuxer.fps(), (30000, 1001));
        let fps = demuxer.header().fps();
        assert!((fps - 29.97).abs() < 0.01);
    }

    #[test]
    #[ignore]
    fn test_demuxer_pixel_aspect_ratio() {
        let data = b"YUV4MPEG2 W720 H480 A10:11\n";
        let demuxer = Y4mDemuxer::new(Cursor::new(data.to_vec())).expect("should parse");
        assert_eq!(demuxer.header().par_num, 10);
        assert_eq!(demuxer.header().par_den, 11);
        let par = demuxer.header().pixel_aspect_ratio();
        assert!((par - 10.0 / 11.0).abs() < 0.001);
    }

    #[test]
    #[ignore]
    fn test_header_frame_size() {
        let header = Y4mHeader {
            width: 1920,
            height: 1080,
            chroma: Y4mChroma::C420jpeg,
            ..Y4mHeader::default()
        };
        assert_eq!(header.frame_size(), 3110400);
    }

    #[test]
    #[ignore]
    fn test_validate_frame_tag_valid() {
        assert!(validate_frame_tag("FRAME\n").is_ok());
        assert!(validate_frame_tag("FRAME").is_ok());
        assert!(validate_frame_tag("FRAME Xcomment\n").is_ok());
    }

    #[test]
    #[ignore]
    fn test_validate_frame_tag_invalid() {
        assert!(validate_frame_tag("FRAMES\n").is_err());
        assert!(validate_frame_tag("FRAMEX\n").is_err());
        assert!(validate_frame_tag("frame\n").is_err());
        assert!(validate_frame_tag("").is_err());
    }

    #[test]
    #[ignore]
    fn test_chroma_display() {
        assert_eq!(format!("{}", Y4mChroma::C420jpeg), "420jpeg");
        assert_eq!(format!("{}", Y4mChroma::C422), "422");
        assert_eq!(format!("{}", Y4mChroma::C444), "444");
        assert_eq!(format!("{}", Y4mChroma::Mono), "mono");
    }

    #[test]
    #[ignore]
    fn test_interlace_display() {
        assert_eq!(format!("{}", Y4mInterlace::Progressive), "p");
        assert_eq!(format!("{}", Y4mInterlace::TopFirst), "t");
        assert_eq!(format!("{}", Y4mInterlace::BottomFirst), "b");
        assert_eq!(format!("{}", Y4mInterlace::Mixed), "m");
    }

    #[test]
    #[ignore]
    fn test_parse_ratio() {
        assert_eq!(parse_ratio("30:1"), Some((30, 1)));
        assert_eq!(parse_ratio("30000:1001"), Some((30000, 1001)));
        assert_eq!(parse_ratio("0:0"), Some((0, 0)));
        assert_eq!(parse_ratio("30"), None);
        assert_eq!(parse_ratio("abc:def"), None);
        assert_eq!(parse_ratio(":1"), None);
    }

    #[test]
    #[ignore]
    fn test_header_default_fps_zero_den() {
        let mut header = Y4mHeader::default();
        header.width = 8;
        header.height = 8;
        header.fps_den = 0;
        assert_eq!(header.fps(), 0.0);
    }

    #[test]
    #[ignore]
    fn test_header_default_par_zero_den() {
        let mut header = Y4mHeader::default();
        header.par_den = 0;
        assert_eq!(header.pixel_aspect_ratio(), 1.0);
    }
}
