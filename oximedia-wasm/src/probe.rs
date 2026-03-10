//! Format probing for WASM.
//!
//! This module exposes the `probe_format` functionality to JavaScript,
//! allowing detection of media container formats from raw bytes.

use crate::container::ContainerFormat;
use crate::types::{JsStreamInfo, MediaInfo};
use wasm_bindgen::prelude::*;

use crate::utils::to_js_error;

/// Result of format probing.
///
/// Contains the detected format and confidence score.
///
/// # JavaScript Example
///
/// ```javascript
/// const data = new Uint8Array([0x1A, 0x45, 0xDF, 0xA3, ...]);
/// const result = oximedia.probe_format(data);
/// console.log('Format:', result.format());
/// console.log('Confidence:', result.confidence());
/// ```
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct WasmProbeResult {
    format: ContainerFormat,
    confidence: f32,
}

impl WasmProbeResult {
    /// Internal constructor (not exposed to JavaScript).
    #[must_use]
    pub(crate) const fn new_internal(format: ContainerFormat, confidence: f32) -> Self {
        Self { format, confidence }
    }
}

#[wasm_bindgen]
impl WasmProbeResult {
    /// Returns the detected container format as a string.
    ///
    /// Possible values:
    /// - `"Matroska"` - Matroska/`WebM` container
    /// - `"Ogg"` - Ogg container
    /// - `"Flac"` - FLAC audio
    /// - `"Wav"` - WAV audio
    /// - `"Mp4"` - MP4/ISOBMFF container (AV1/VP9 only)
    #[must_use]
    pub fn format(&self) -> String {
        format!("{:?}", self.format)
    }

    /// Returns the confidence score from 0.0 to 1.0.
    ///
    /// Higher values indicate greater confidence in the detection.
    #[must_use]
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Returns a human-readable description of the format.
    #[must_use]
    pub fn description(&self) -> String {
        match self.format {
            ContainerFormat::Matroska => "Matroska/WebM container (.mkv, .webm)".to_string(),
            ContainerFormat::Ogg => "Ogg container (.ogg, .opus, .oga)".to_string(),
            ContainerFormat::Flac => "FLAC audio (.flac)".to_string(),
            ContainerFormat::Wav => "WAV audio (.wav)".to_string(),
            ContainerFormat::Mp4 => "MP4/ISOBMFF container (.mp4)".to_string(),
        }
    }

    /// Returns true if the format is a video container.
    #[must_use]
    pub fn is_video_container(&self) -> bool {
        matches!(
            self.format,
            ContainerFormat::Matroska | ContainerFormat::Mp4
        )
    }

    /// Returns true if the format is an audio-only container.
    #[must_use]
    pub fn is_audio_only(&self) -> bool {
        matches!(self.format, ContainerFormat::Flac | ContainerFormat::Wav)
    }
}

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
/// Throws a JavaScript exception if the format cannot be detected.
///
/// # JavaScript Example
///
/// ```javascript
/// import * as oximedia from 'oximedia-wasm';
///
/// // WebM/Matroska header
/// const data = new Uint8Array([
///     0x1A, 0x45, 0xDF, 0xA3,  // EBML magic
///     0x01, 0x00, 0x00, 0x00,
///     0x00, 0x00, 0x00, 0x1F,
/// ]);
///
/// try {
///     const result = oximedia.probe_format(data);
///     console.log('Detected format:', result.format());
///     console.log('Confidence:', (result.confidence() * 100).toFixed(1) + '%');
/// } catch (e) {
///     console.error('Failed to probe format:', e);
/// }
/// ```
#[wasm_bindgen]
pub fn probe_format(data: &[u8]) -> Result<WasmProbeResult, JsValue> {
    let result = crate::container::probe_format(data).map_err(to_js_error)?;

    Ok(WasmProbeResult {
        format: result.format,
        confidence: result.confidence,
    })
}

/// Probe media data and return comprehensive `MediaInfo` as a JavaScript value.
///
/// Unlike [`probe_format`] which returns a `WasmProbeResult` wrapper object,
/// this function returns a plain JavaScript object that can be inspected
/// directly without calling getter methods.
///
/// # Arguments
///
/// * `data` - At least the first 12 bytes of the file (more bytes improve detection)
///
/// # Returns
///
/// A serialized [`MediaInfo`] object with `format`, `confidence`, `streams`, etc.
///
/// # Errors
///
/// Throws a JavaScript exception if the format cannot be detected or serialization fails.
///
/// # JavaScript Example
///
/// ```javascript
/// import * as oximedia from 'oximedia-wasm';
///
/// const data = new Uint8Array([0x1A, 0x45, 0xDF, 0xA3, 0, 0, 0, 0]);
/// try {
///     const info = oximedia.probe_media(data);
///     console.log('Format:', info.format);
///     console.log('Confidence:', (info.confidence * 100).toFixed(1) + '%');
///     console.log('Is video:', info.is_video_container);
///     for (const stream of info.streams) {
///         console.log(`Stream ${stream.index}: ${stream.codec} (${stream.media_type})`);
///     }
/// } catch (e) {
///     console.error('Probe failed:', e);
/// }
/// ```
#[wasm_bindgen]
pub fn probe_media(data: &[u8]) -> Result<JsValue, JsValue> {
    let result = crate::container::probe_format(data).map_err(to_js_error)?;

    let probe_result = WasmProbeResult {
        format: result.format,
        confidence: result.confidence,
    };

    // Build a demuxer to extract stream information
    let mut demuxer = crate::demuxer::WasmDemuxer::new(data);
    let streams: Vec<JsStreamInfo> = match demuxer.probe() {
        Ok(_) => demuxer.streams().iter().map(JsStreamInfo::from).collect(),
        Err(_) => Vec::new(),
    };

    let stream_count = streams.len();
    let info = MediaInfo {
        format: probe_result.format(),
        format_description: probe_result.description(),
        confidence: probe_result.confidence(),
        is_video_container: probe_result.is_video_container(),
        is_audio_only: probe_result.is_audio_only(),
        streams,
        stream_count,
    };

    serde_json::to_string(&info)
        .map_err(|e| crate::utils::js_err(&format!("Serialization error: {e}")))
        .and_then(|json| {
            js_sys::JSON::parse(&json)
                .map_err(|e| crate::utils::js_err(&format!("JSON parse error: {e:?}")))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_webm() {
        let data = [0x1A, 0x45, 0xDF, 0xA3, 0x01, 0x00, 0x00, 0x00];
        let result = probe_format(&data).expect("probe should succeed");
        assert_eq!(result.format(), "Matroska");
        assert!(result.confidence() > 0.9);
        assert!(result.is_video_container());
    }

    #[test]
    fn test_probe_ogg() {
        let data = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00";
        let result = probe_format(data).expect("probe should succeed");
        assert_eq!(result.format(), "Ogg");
    }

    #[test]
    fn test_probe_flac() {
        let data = b"fLaC\x00\x00\x00\x22";
        let result = probe_format(data).expect("probe should succeed");
        assert_eq!(result.format(), "Flac");
        assert!(result.is_audio_only());
    }

    #[test]
    fn test_probe_unknown() {
        // Test the underlying container probe without WASM JsValue conversion
        // which panics in native test environments.
        let data = [0xFF; 16];
        assert!(crate::container::probe_format(&data).is_err());
    }

    #[test]
    fn test_media_info_serialization() {
        // Validate that MediaInfo can be constructed and serialized to JSON.
        // The full probe_media() uses js_sys::JSON::parse which is not available
        // in native test environments, so we test the data layer directly.
        use crate::types::{JsStreamInfo, MediaInfo};

        let info = MediaInfo {
            format: "Matroska".to_string(),
            format_description: "Matroska/WebM container (.mkv, .webm)".to_string(),
            confidence: 0.95,
            is_video_container: true,
            is_audio_only: false,
            streams: vec![JsStreamInfo {
                index: 0,
                codec: "Vp9".to_string(),
                media_type: "Video".to_string(),
                duration_seconds: None,
                timebase_num: 1,
                timebase_den: 1000,
                width: Some(1920),
                height: Some(1080),
                sample_rate: None,
                channels: None,
            }],
            stream_count: 1,
        };

        let json = serde_json::to_string(&info).expect("serde_json::to_string should succeed");
        assert!(json.contains("\"format\":\"Matroska\""));
        assert!(json.contains("\"confidence\":0.95"));
        assert!(json.contains("\"is_video_container\":true"));
        assert!(json.contains("\"stream_count\":1"));
        assert!(json.contains("\"codec\":\"Vp9\""));
    }

    #[test]
    fn test_probe_media_unknown_returns_err() {
        // Verify that probe_media on unknown data returns an Err.
        // We test via the underlying container::probe_format to avoid JsValue in native.
        let data = [0xFF; 16];
        assert!(crate::container::probe_format(&data).is_err());
    }
}
