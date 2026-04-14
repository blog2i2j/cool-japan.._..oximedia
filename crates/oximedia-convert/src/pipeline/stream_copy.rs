// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Stream copy (remux) mode for the conversion pipeline.
//!
//! When the source and target containers use compatible codecs, stream copy
//! bypasses the decode/encode cycle entirely and simply remuxes the compressed
//! bitstream into the new container. This is dramatically faster and
//! bit-exact with respect to the original media.
//!
//! # When to use stream copy
//!
//! - Changing containers without re-encoding (e.g. MKV -> WebM when codecs match)
//! - Extracting a time range from a file (cut without re-encode)
//! - Stripping or adding metadata without touching the media streams
//! - Changing the muxing parameters (segment duration, etc.)
//!
//! # Limitations
//!
//! - Source codec must be supported by the target container
//! - Frame-accurate cuts require re-encoding near cut points (not yet supported)
//! - Filters/effects cannot be applied in stream copy mode

#![allow(dead_code)]

use crate::formats::{AudioCodec, ContainerFormat, VideoCodec};
use crate::{ConversionError, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

// ── Codec/Container Compatibility ──────────────────────────────────────────

/// Describes which codecs a container format can carry.
#[derive(Debug, Clone)]
pub struct ContainerCodecCompat {
    /// Container format.
    pub container: ContainerFormat,
    /// Video codecs supported by this container.
    pub video_codecs: Vec<VideoCodec>,
    /// Audio codecs supported by this container.
    pub audio_codecs: Vec<AudioCodec>,
}

/// Static compatibility table for patent-free containers.
fn container_compat_table() -> Vec<ContainerCodecCompat> {
    vec![
        ContainerCodecCompat {
            container: ContainerFormat::Webm,
            video_codecs: vec![VideoCodec::Vp8, VideoCodec::Vp9, VideoCodec::Av1],
            audio_codecs: vec![AudioCodec::Opus, AudioCodec::Vorbis],
        },
        ContainerCodecCompat {
            container: ContainerFormat::Matroska,
            video_codecs: vec![
                VideoCodec::Vp8,
                VideoCodec::Vp9,
                VideoCodec::Av1,
                VideoCodec::Theora,
            ],
            audio_codecs: vec![
                AudioCodec::Opus,
                AudioCodec::Vorbis,
                AudioCodec::Flac,
                AudioCodec::Pcm,
            ],
        },
        ContainerCodecCompat {
            container: ContainerFormat::Ogg,
            video_codecs: vec![VideoCodec::Theora],
            audio_codecs: vec![AudioCodec::Opus, AudioCodec::Vorbis, AudioCodec::Flac],
        },
        ContainerCodecCompat {
            container: ContainerFormat::Wav,
            video_codecs: vec![],
            audio_codecs: vec![AudioCodec::Pcm],
        },
        ContainerCodecCompat {
            container: ContainerFormat::Flac,
            video_codecs: vec![],
            audio_codecs: vec![AudioCodec::Flac],
        },
    ]
}

/// Check if a given video codec is supported by the target container.
#[must_use]
pub fn is_video_codec_compatible(codec: &VideoCodec, target: &ContainerFormat) -> bool {
    container_compat_table()
        .iter()
        .any(|c| c.container == *target && c.video_codecs.contains(codec))
}

/// Check if a given audio codec is supported by the target container.
#[must_use]
pub fn is_audio_codec_compatible(codec: &AudioCodec, target: &ContainerFormat) -> bool {
    container_compat_table()
        .iter()
        .any(|c| c.container == *target && c.audio_codecs.contains(codec))
}

// ── Stream Copy Decision ───────────────────────────────────────────────────

/// Information about the streams in the source file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceStreams {
    /// Video codec of the source (if any).
    pub video_codec: Option<VideoCodec>,
    /// Audio codec of the source (if any).
    pub audio_codec: Option<AudioCodec>,
    /// Source container format.
    pub source_container: ContainerFormat,
    /// Total duration of the source media.
    pub duration: Option<Duration>,
    /// Whether the source has subtitle streams.
    pub has_subtitles: bool,
}

/// Requirements for the output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputRequirements {
    /// Target container format.
    pub target_container: ContainerFormat,
    /// Whether video re-encoding is requested (resolution/bitrate/filter changes).
    pub needs_video_reencode: bool,
    /// Whether audio re-encoding is requested.
    pub needs_audio_reencode: bool,
    /// Time range to extract (if any). Uses start/end in seconds.
    pub time_range: Option<(f64, f64)>,
    /// Whether to strip subtitles.
    pub strip_subtitles: bool,
}

/// The result of a stream copy eligibility analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamCopyDecision {
    /// Whether the video stream can be copied without re-encoding.
    pub copy_video: bool,
    /// Whether the audio stream can be copied without re-encoding.
    pub copy_audio: bool,
    /// Reason if video cannot be copied.
    pub video_reason: Option<String>,
    /// Reason if audio cannot be copied.
    pub audio_reason: Option<String>,
}

impl StreamCopyDecision {
    /// Whether at least one stream can be copied.
    #[must_use]
    pub fn any_copy(&self) -> bool {
        self.copy_video || self.copy_audio
    }

    /// Whether all streams can be copied (full remux).
    #[must_use]
    pub fn full_copy(&self) -> bool {
        self.copy_video && self.copy_audio
    }

    /// Whether no streams can be copied (full re-encode required).
    #[must_use]
    pub fn no_copy(&self) -> bool {
        !self.copy_video && !self.copy_audio
    }
}

/// Analyze whether stream copy is possible for the given source and output.
#[must_use]
pub fn analyze_stream_copy(
    source: &SourceStreams,
    output: &OutputRequirements,
) -> StreamCopyDecision {
    let (copy_video, video_reason) = analyze_video_copy(source, output);
    let (copy_audio, audio_reason) = analyze_audio_copy(source, output);

    StreamCopyDecision {
        copy_video,
        copy_audio,
        video_reason,
        audio_reason,
    }
}

fn analyze_video_copy(
    source: &SourceStreams,
    output: &OutputRequirements,
) -> (bool, Option<String>) {
    // No video stream → nothing to copy
    let codec = match &source.video_codec {
        Some(c) => c,
        None => return (false, Some("No video stream in source".to_string())),
    };

    // Explicit re-encode requested
    if output.needs_video_reencode {
        return (
            false,
            Some("Video re-encoding explicitly requested".to_string()),
        );
    }

    // Check container compatibility
    if !is_video_codec_compatible(codec, &output.target_container) {
        return (
            false,
            Some(format!(
                "Video codec {:?} is not compatible with target container {:?}",
                codec, output.target_container
            )),
        );
    }

    (true, None)
}

fn analyze_audio_copy(
    source: &SourceStreams,
    output: &OutputRequirements,
) -> (bool, Option<String>) {
    let codec = match &source.audio_codec {
        Some(c) => c,
        None => return (false, Some("No audio stream in source".to_string())),
    };

    if output.needs_audio_reencode {
        return (
            false,
            Some("Audio re-encoding explicitly requested".to_string()),
        );
    }

    if !is_audio_codec_compatible(codec, &output.target_container) {
        return (
            false,
            Some(format!(
                "Audio codec {:?} is not compatible with target container {:?}",
                codec, output.target_container
            )),
        );
    }

    (true, None)
}

// ── Stream Copy Executor ───────────────────────────────────────────────────

/// Result of a stream copy operation.
#[derive(Debug, Clone)]
pub struct StreamCopyResult {
    /// Input file path.
    pub input: PathBuf,
    /// Output file path.
    pub output: PathBuf,
    /// Whether video was stream-copied.
    pub video_copied: bool,
    /// Whether audio was stream-copied.
    pub audio_copied: bool,
    /// Input file size.
    pub input_size: u64,
    /// Output file size.
    pub output_size: u64,
    /// Processing duration.
    pub duration: Duration,
    /// Speed ratio (input duration / processing time).
    pub speed_ratio: f64,
}

/// Execute a stream copy (remux) operation.
///
/// This reads the source file, validates codec compatibility, and writes
/// the output with the new container format. In the current implementation
/// the bitstream is passed through as-is (no container-level rewriting yet;
/// that will be integrated with the oximedia-container crate).
pub fn execute_stream_copy(
    input: &Path,
    output: &Path,
    source: &SourceStreams,
    output_req: &OutputRequirements,
) -> Result<StreamCopyResult> {
    let start = std::time::Instant::now();

    // Validate input exists
    if !input.exists() {
        return Err(ConversionError::InvalidInput(format!(
            "Input file not found: {}",
            input.display()
        )));
    }

    // Check if stream copy is possible
    let decision = analyze_stream_copy(source, output_req);
    if decision.no_copy() {
        let reasons: Vec<String> = [decision.video_reason, decision.audio_reason]
            .iter()
            .filter_map(|r| r.clone())
            .collect();
        return Err(ConversionError::Transcode(format!(
            "Stream copy not possible: {}",
            reasons.join("; ")
        )));
    }

    // Read input
    let data = std::fs::read(input).map_err(ConversionError::Io)?;
    let input_size = data.len() as u64;

    // Apply time range extraction if specified
    let output_data = if let Some((start_sec, end_sec)) = output_req.time_range {
        extract_time_range(&data, start_sec, end_sec, source)
    } else {
        data
    };

    // Create output directory if needed
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            std::fs::create_dir_all(parent).map_err(|e| {
                ConversionError::InvalidOutput(format!(
                    "Cannot create output directory: {e}"
                ))
            })?;
        }
    }

    // Write output
    let output_size = output_data.len() as u64;
    std::fs::write(output, &output_data).map_err(|e| {
        ConversionError::InvalidOutput(format!(
            "Failed to write output '{}': {e}",
            output.display()
        ))
    })?;

    let elapsed = start.elapsed();
    let speed_ratio = source
        .duration
        .map(|d| d.as_secs_f64() / elapsed.as_secs_f64().max(0.001))
        .unwrap_or(0.0);

    Ok(StreamCopyResult {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        video_copied: decision.copy_video,
        audio_copied: decision.copy_audio,
        input_size,
        output_size,
        duration: elapsed,
        speed_ratio,
    })
}

/// Extract a byte-proportional time range from the data.
///
/// This is a simplified extraction that estimates byte offsets from time
/// proportions. Full keyframe-aware seeking requires container demuxing
/// integration (future work).
fn extract_time_range(
    data: &[u8],
    start_sec: f64,
    end_sec: f64,
    source: &SourceStreams,
) -> Vec<u8> {
    let total_duration = source
        .duration
        .map(|d| d.as_secs_f64())
        .unwrap_or(1.0)
        .max(0.001);

    let start_frac = (start_sec / total_duration).clamp(0.0, 1.0);
    let end_frac = (end_sec / total_duration).clamp(0.0, 1.0);

    if start_frac >= end_frac {
        return Vec::new();
    }

    let start_byte = (data.len() as f64 * start_frac) as usize;
    let end_byte = (data.len() as f64 * end_frac) as usize;

    let start_byte = start_byte.min(data.len());
    let end_byte = end_byte.min(data.len());

    data[start_byte..end_byte].to_vec()
}

/// Estimate the speedup of stream copy vs re-encoding.
///
/// Returns an estimated speedup factor (e.g. 50x means stream copy is ~50
/// times faster than re-encoding the same content).
#[must_use]
pub fn estimate_speedup(input_size_bytes: u64) -> f64 {
    // Stream copy is essentially I/O bound.
    // Encoding speed estimate: ~5 MB/s for software encoding
    // Copy speed estimate: ~500 MB/s for SSD I/O
    let encode_speed = 5_000_000.0_f64; // bytes/sec
    let copy_speed = 500_000_000.0_f64; // bytes/sec

    let encode_time = input_size_bytes as f64 / encode_speed;
    let copy_time = input_size_bytes as f64 / copy_speed;

    if copy_time > 0.0 {
        encode_time / copy_time
    } else {
        100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vp9_compatible_with_webm() {
        assert!(is_video_codec_compatible(
            &VideoCodec::Vp9,
            &ContainerFormat::Webm
        ));
    }

    #[test]
    fn test_vp8_compatible_with_webm() {
        assert!(is_video_codec_compatible(
            &VideoCodec::Vp8,
            &ContainerFormat::Webm
        ));
    }

    #[test]
    fn test_av1_compatible_with_mkv() {
        assert!(is_video_codec_compatible(
            &VideoCodec::Av1,
            &ContainerFormat::Matroska
        ));
    }

    #[test]
    fn test_theora_not_compatible_with_webm() {
        assert!(!is_video_codec_compatible(
            &VideoCodec::Theora,
            &ContainerFormat::Webm
        ));
    }

    #[test]
    fn test_opus_compatible_with_ogg() {
        assert!(is_audio_codec_compatible(
            &AudioCodec::Opus,
            &ContainerFormat::Ogg
        ));
    }

    #[test]
    fn test_pcm_not_compatible_with_webm() {
        assert!(!is_audio_codec_compatible(
            &AudioCodec::Pcm,
            &ContainerFormat::Webm
        ));
    }

    #[test]
    fn test_flac_compatible_with_mkv() {
        assert!(is_audio_codec_compatible(
            &AudioCodec::Flac,
            &ContainerFormat::Matroska
        ));
    }

    #[test]
    fn test_full_copy_vp9_opus_webm_to_mkv() {
        let source = SourceStreams {
            video_codec: Some(VideoCodec::Vp9),
            audio_codec: Some(AudioCodec::Opus),
            source_container: ContainerFormat::Webm,
            duration: Some(Duration::from_secs(60)),
            has_subtitles: false,
        };
        let output = OutputRequirements {
            target_container: ContainerFormat::Matroska,
            needs_video_reencode: false,
            needs_audio_reencode: false,
            time_range: None,
            strip_subtitles: false,
        };

        let decision = analyze_stream_copy(&source, &output);
        assert!(decision.full_copy());
        assert!(decision.any_copy());
        assert!(!decision.no_copy());
        assert!(decision.video_reason.is_none());
        assert!(decision.audio_reason.is_none());
    }

    #[test]
    fn test_no_copy_theora_to_webm() {
        let source = SourceStreams {
            video_codec: Some(VideoCodec::Theora),
            audio_codec: Some(AudioCodec::Vorbis),
            source_container: ContainerFormat::Ogg,
            duration: None,
            has_subtitles: false,
        };
        let output = OutputRequirements {
            target_container: ContainerFormat::Webm,
            needs_video_reencode: false,
            needs_audio_reencode: false,
            time_range: None,
            strip_subtitles: false,
        };

        let decision = analyze_stream_copy(&source, &output);
        assert!(!decision.copy_video); // Theora not supported in WebM
        assert!(decision.copy_audio); // Vorbis is supported in WebM
        assert!(decision.video_reason.is_some());
    }

    #[test]
    fn test_no_copy_when_reencode_requested() {
        let source = SourceStreams {
            video_codec: Some(VideoCodec::Vp9),
            audio_codec: Some(AudioCodec::Opus),
            source_container: ContainerFormat::Webm,
            duration: None,
            has_subtitles: false,
        };
        let output = OutputRequirements {
            target_container: ContainerFormat::Webm,
            needs_video_reencode: true,
            needs_audio_reencode: true,
            time_range: None,
            strip_subtitles: false,
        };

        let decision = analyze_stream_copy(&source, &output);
        assert!(decision.no_copy());
    }

    #[test]
    fn test_partial_copy_video_only() {
        let source = SourceStreams {
            video_codec: Some(VideoCodec::Vp9),
            audio_codec: Some(AudioCodec::Opus),
            source_container: ContainerFormat::Webm,
            duration: None,
            has_subtitles: false,
        };
        let output = OutputRequirements {
            target_container: ContainerFormat::Matroska,
            needs_video_reencode: false,
            needs_audio_reencode: true,
            time_range: None,
            strip_subtitles: false,
        };

        let decision = analyze_stream_copy(&source, &output);
        assert!(decision.copy_video);
        assert!(!decision.copy_audio);
        assert!(decision.any_copy());
        assert!(!decision.full_copy());
    }

    #[test]
    fn test_no_video_stream() {
        let source = SourceStreams {
            video_codec: None,
            audio_codec: Some(AudioCodec::Opus),
            source_container: ContainerFormat::Ogg,
            duration: None,
            has_subtitles: false,
        };
        let output = OutputRequirements {
            target_container: ContainerFormat::Webm,
            needs_video_reencode: false,
            needs_audio_reencode: false,
            time_range: None,
            strip_subtitles: false,
        };

        let decision = analyze_stream_copy(&source, &output);
        assert!(!decision.copy_video);
        assert!(decision.copy_audio);
    }

    #[test]
    fn test_extract_time_range_middle() {
        let data = vec![0u8; 1000];
        let source = SourceStreams {
            video_codec: None,
            audio_codec: None,
            source_container: ContainerFormat::Wav,
            duration: Some(Duration::from_secs(10)),
            has_subtitles: false,
        };

        // Extract middle 50% (2.5s to 7.5s of 10s)
        let extracted = extract_time_range(&data, 2.5, 7.5, &source);
        assert_eq!(extracted.len(), 500); // 50% of 1000
    }

    #[test]
    fn test_extract_time_range_invalid() {
        let data = vec![0u8; 1000];
        let source = SourceStreams {
            video_codec: None,
            audio_codec: None,
            source_container: ContainerFormat::Wav,
            duration: Some(Duration::from_secs(10)),
            has_subtitles: false,
        };

        // Start after end → empty
        let extracted = extract_time_range(&data, 5.0, 2.0, &source);
        assert!(extracted.is_empty());
    }

    #[test]
    fn test_estimate_speedup_nonzero() {
        let speedup = estimate_speedup(100_000_000); // 100 MB
        assert!(speedup > 1.0, "stream copy should be faster: {speedup}");
        assert!(speedup < 1000.0, "speedup should be reasonable: {speedup}");
    }

    #[test]
    fn test_execute_stream_copy_real_file() {
        let dir = std::env::temp_dir().join("oximedia_stream_copy_test");
        let _ = std::fs::create_dir_all(&dir);
        let input_path = dir.join("input.bin");
        let output_path = dir.join("output.bin");

        // Write test data
        let data = vec![42u8; 4096];
        std::fs::write(&input_path, &data).expect("write input");

        let source = SourceStreams {
            video_codec: Some(VideoCodec::Vp9),
            audio_codec: Some(AudioCodec::Opus),
            source_container: ContainerFormat::Webm,
            duration: Some(Duration::from_secs(10)),
            has_subtitles: false,
        };
        let output_req = OutputRequirements {
            target_container: ContainerFormat::Matroska,
            needs_video_reencode: false,
            needs_audio_reencode: false,
            time_range: None,
            strip_subtitles: false,
        };

        let result = execute_stream_copy(&input_path, &output_path, &source, &output_req)
            .expect("stream copy should succeed");

        assert!(result.video_copied);
        assert!(result.audio_copied);
        assert_eq!(result.input_size, 4096);
        assert_eq!(result.output_size, 4096);
        assert!(result.speed_ratio > 0.0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_execute_stream_copy_with_time_range() {
        let dir = std::env::temp_dir().join("oximedia_stream_copy_range_test");
        let _ = std::fs::create_dir_all(&dir);
        let input_path = dir.join("input.bin");
        let output_path = dir.join("output.bin");

        let data = vec![99u8; 10000];
        std::fs::write(&input_path, &data).expect("write input");

        let source = SourceStreams {
            video_codec: Some(VideoCodec::Vp9),
            audio_codec: Some(AudioCodec::Opus),
            source_container: ContainerFormat::Webm,
            duration: Some(Duration::from_secs(100)),
            has_subtitles: false,
        };
        let output_req = OutputRequirements {
            target_container: ContainerFormat::Matroska,
            needs_video_reencode: false,
            needs_audio_reencode: false,
            time_range: Some((0.0, 50.0)), // first half
            strip_subtitles: false,
        };

        let result = execute_stream_copy(&input_path, &output_path, &source, &output_req)
            .expect("stream copy should succeed");

        assert_eq!(result.output_size, 5000); // half of 10000

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_execute_stream_copy_missing_input() {
        let source = SourceStreams {
            video_codec: Some(VideoCodec::Vp9),
            audio_codec: Some(AudioCodec::Opus),
            source_container: ContainerFormat::Webm,
            duration: None,
            has_subtitles: false,
        };
        let output_req = OutputRequirements {
            target_container: ContainerFormat::Matroska,
            needs_video_reencode: false,
            needs_audio_reencode: false,
            time_range: None,
            strip_subtitles: false,
        };

        let result = execute_stream_copy(
            Path::new("/nonexistent/file.webm"),
            Path::new("/tmp/out.mkv"),
            &source,
            &output_req,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_stream_copy_decision_display_logic() {
        let d = StreamCopyDecision {
            copy_video: true,
            copy_audio: true,
            video_reason: None,
            audio_reason: None,
        };
        assert!(d.full_copy());
        assert!(d.any_copy());
        assert!(!d.no_copy());

        let d2 = StreamCopyDecision {
            copy_video: false,
            copy_audio: false,
            video_reason: Some("incompatible".to_string()),
            audio_reason: Some("incompatible".to_string()),
        };
        assert!(!d2.full_copy());
        assert!(!d2.any_copy());
        assert!(d2.no_copy());
    }

    #[test]
    fn test_container_compat_wav_audio_only() {
        assert!(is_audio_codec_compatible(
            &AudioCodec::Pcm,
            &ContainerFormat::Wav
        ));
        assert!(!is_video_codec_compatible(
            &VideoCodec::Vp9,
            &ContainerFormat::Wav
        ));
    }
}
