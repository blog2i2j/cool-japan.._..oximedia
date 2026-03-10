//! Format probing and detection.

use crate::ContainerFormat;
use oximedia_core::OxiError;

/// Result of format probing.
///
/// Contains the detected format and a confidence score indicating
/// how certain the detection is.
#[derive(Clone, Debug)]
pub struct ProbeResult {
    /// Detected container format.
    pub format: ContainerFormat,
    /// Confidence score from 0.0 to 1.0.
    ///
    /// Higher values indicate greater confidence in the detection.
    /// A score of 1.0 means the format is certain (e.g., unique magic bytes).
    pub confidence: f32,
}

impl ProbeResult {
    /// Creates a new probe result.
    #[must_use]
    pub const fn new(format: ContainerFormat, confidence: f32) -> Self {
        Self { format, confidence }
    }
}

/// Magic byte signatures for container detection.
const MATROSKA_MAGIC: &[u8] = &[0x1A, 0x45, 0xDF, 0xA3]; // EBML header
const OGG_MAGIC: &[u8] = b"OggS";
const FLAC_MAGIC: &[u8] = b"fLaC";
const RIFF_MAGIC: &[u8] = b"RIFF";
const WAVE_MAGIC: &[u8] = b"WAVE";
const ISOBMFF_FTYP: &[u8] = b"ftyp";
const WEBVTT_MAGIC: &[u8] = b"WEBVTT";
const Y4M_MAGIC: &[u8] = b"YUV4MPEG2";
const MPEG_TS_SYNC: u8 = 0x47; // MPEG-TS sync byte
const TS_PACKET_SIZE: usize = 188;

/// Probe the container format from raw bytes.
///
/// Analyzes the first few bytes of media data to detect the container format.
/// Returns the detected format and a confidence score.
///
/// # Arguments
///
/// * `data` - At least the first 12 bytes of the file (more bytes improve detection)
///
/// # Errors
///
/// Returns `OxiError::UnknownFormat` if the format cannot be detected.
///
/// # Example
///
/// ```
/// use oximedia_container::{probe_format, ContainerFormat};
///
/// // WebM/Matroska header
/// let data = [0x1A, 0x45, 0xDF, 0xA3, 0x01, 0x00, 0x00, 0x00];
/// let result = probe_format(&data)?;
/// assert_eq!(result.format, ContainerFormat::Matroska);
/// ```
pub fn probe_format(data: &[u8]) -> Result<ProbeResult, OxiError> {
    if data.len() < 4 {
        return Err(OxiError::UnknownFormat);
    }

    // Check Matroska/WebM (EBML header)
    if data.starts_with(MATROSKA_MAGIC) {
        // WebM is detected by DocType, but for initial probe we return Matroska
        return Ok(ProbeResult {
            format: ContainerFormat::Matroska,
            confidence: 0.95,
        });
    }

    // Check Ogg
    if data.starts_with(OGG_MAGIC) {
        return Ok(ProbeResult {
            format: ContainerFormat::Ogg,
            confidence: 0.99,
        });
    }

    // Check Y4M (YUV4MPEG2)
    if data.len() >= Y4M_MAGIC.len() && data.starts_with(Y4M_MAGIC) {
        return Ok(ProbeResult {
            format: ContainerFormat::Y4m,
            confidence: 0.99,
        });
    }

    // Check FLAC
    if data.starts_with(FLAC_MAGIC) {
        return Ok(ProbeResult {
            format: ContainerFormat::Flac,
            confidence: 0.99,
        });
    }

    // Check WAV (RIFF + WAVE)
    if data.len() >= 12 && data.starts_with(RIFF_MAGIC) && &data[8..12] == WAVE_MAGIC {
        return Ok(ProbeResult {
            format: ContainerFormat::Wav,
            confidence: 0.99,
        });
    }

    // Check ISOBMFF/MP4 (ftyp box)
    if data.len() >= 8 && &data[4..8] == ISOBMFF_FTYP {
        return Ok(ProbeResult {
            format: ContainerFormat::Mp4,
            confidence: 0.90,
        });
    }

    // Check MPEG-TS (sync byte pattern every 188 bytes)
    // We need at least 2 packets (376 bytes) for reliable detection
    if data.len() >= TS_PACKET_SIZE * 2 {
        let mut sync_count = 0;
        let max_checks = (data.len() / TS_PACKET_SIZE).min(3);

        for i in 0..max_checks {
            if data[i * TS_PACKET_SIZE] == MPEG_TS_SYNC {
                sync_count += 1;
            } else {
                break;
            }
        }

        if sync_count >= 2 {
            return Ok(ProbeResult {
                format: ContainerFormat::MpegTs,
                confidence: 0.95,
            });
        }
    } else if data.len() >= TS_PACKET_SIZE && data[0] == MPEG_TS_SYNC {
        // Single packet check (lower confidence)
        return Ok(ProbeResult {
            format: ContainerFormat::MpegTs,
            confidence: 0.60,
        });
    }

    // Check WebVTT
    if data.starts_with(WEBVTT_MAGIC) {
        return Ok(ProbeResult {
            format: ContainerFormat::WebVtt,
            confidence: 0.99,
        });
    }

    // Check SRT (heuristic: starts with a number followed by newline and timestamp)
    // SRT format: "1\n00:00:00,000 --> 00:00:02,000\n"
    if data.len() >= 20 {
        // Convert to string to check pattern
        if let Ok(text) = std::str::from_utf8(&data[..data.len().min(100)]) {
            let lines: Vec<&str> = text.lines().take(3).collect();
            if lines.len() >= 2
                && lines[0].trim().chars().all(|c| c.is_ascii_digit())
                && lines[1].contains("-->")
                && lines[1].contains(',')
            {
                return Ok(ProbeResult {
                    format: ContainerFormat::Srt,
                    confidence: 0.85,
                });
            }
        }
    }

    Err(OxiError::UnknownFormat)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_matroska() {
        let data = [0x1A, 0x45, 0xDF, 0xA3, 0x01, 0x00, 0x00, 0x00];
        let result = probe_format(&data).expect("operation should succeed");
        assert_eq!(result.format, ContainerFormat::Matroska);
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn test_probe_ogg() {
        let data = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00";
        let result = probe_format(data).expect("operation should succeed");
        assert_eq!(result.format, ContainerFormat::Ogg);
    }

    #[test]
    fn test_probe_flac() {
        let data = b"fLaC\x00\x00\x00\x22";
        let result = probe_format(data).expect("operation should succeed");
        assert_eq!(result.format, ContainerFormat::Flac);
    }

    #[test]
    fn test_probe_wav() {
        let data = b"RIFF\x00\x00\x00\x00WAVEfmt ";
        let result = probe_format(data).expect("operation should succeed");
        assert_eq!(result.format, ContainerFormat::Wav);
    }

    #[test]
    fn test_probe_unknown() {
        let data = [0x00, 0x00, 0x00, 0x00];
        assert!(probe_format(&data).is_err());
    }

    #[test]
    fn test_probe_too_short() {
        let data = [0x1A, 0x45];
        assert!(probe_format(&data).is_err());
    }
}
