// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Codec encoder factory for MJPEG and APV intra-frame codecs.
//!
//! This module provides [`make_video_encoder`], which constructs a boxed
//! [`oximedia_codec::VideoEncoder`] for a given [`CodecId`].  Currently
//! dispatched codecs are:
//!
//! | `CodecId`       | Encoder              | Feature gate |
//! |-----------------|----------------------|--------------|
//! | `CodecId::Mjpeg`| `MjpegEncoder`       | `mjpeg`      |
//! | `CodecId::Apv`  | `ApvEncoder`         | `apv`        |
//!
//! Callers that need other codecs (AV1, VP9, …) use the existing
//! stream-copy path in `frame_pipeline`.

use crate::{Result, TranscodeError};
use oximedia_codec::traits::VideoEncoder;
#[cfg(feature = "mjpeg")]
use oximedia_codec::CodecError;
use oximedia_core::CodecId;

/// Parameters used to instantiate an intra-frame video encoder.
#[derive(Debug, Clone)]
pub struct VideoEncoderParams {
    /// Frame width in pixels (must be > 0).
    pub width: u32,
    /// Frame height in pixels (must be > 0).
    pub height: u32,
    /// Quality/QP value.  Interpretation depends on the codec:
    /// - MJPEG: JPEG quality 1-100 (higher = better).
    /// - APV: quantisation parameter 0-63 (lower = better).
    pub quality: u8,
}

impl VideoEncoderParams {
    /// Create a new parameter set.
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::InvalidInput`] if width or height is zero.
    pub fn new(width: u32, height: u32, quality: u8) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(TranscodeError::InvalidInput(
                "width and height must be non-zero".into(),
            ));
        }
        Ok(Self {
            width,
            height,
            quality,
        })
    }
}

/// Build a boxed [`VideoEncoder`] for the specified codec.
///
/// # Errors
///
/// - [`TranscodeError::Unsupported`] if `codec_id` is not MJPEG or APV.
/// - [`TranscodeError::CodecError`] if the underlying encoder rejects the
///   parameters.
pub fn make_video_encoder(
    codec_id: CodecId,
    params: &VideoEncoderParams,
) -> Result<Box<dyn VideoEncoder>> {
    match codec_id {
        CodecId::Mjpeg => make_mjpeg_encoder(params),
        CodecId::Apv => make_apv_encoder(params),
        other => Err(TranscodeError::Unsupported(format!(
            "codec {other:?} is not handled by codec_dispatch; \
             use the stream-copy pipeline for this codec"
        ))),
    }
}

// ─── MJPEG ───────────────────────────────────────────────────────────────────

#[cfg(feature = "mjpeg")]
fn make_mjpeg_encoder(params: &VideoEncoderParams) -> Result<Box<dyn VideoEncoder>> {
    use oximedia_codec::{MjpegConfig, MjpegEncoder};

    let config = MjpegConfig::new(params.width, params.height)
        .map_err(|e| TranscodeError::CodecError(e.to_string()))?
        .with_quality(params.quality);

    let encoder = MjpegEncoder::new(config)
        .map_err(|e: CodecError| TranscodeError::CodecError(e.to_string()))?;

    Ok(Box::new(encoder))
}

#[cfg(not(feature = "mjpeg"))]
fn make_mjpeg_encoder(_params: &VideoEncoderParams) -> Result<Box<dyn VideoEncoder>> {
    Err(TranscodeError::Unsupported(
        "MJPEG support requires the `mjpeg` feature of oximedia-codec".into(),
    ))
}

// ─── APV ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "apv")]
fn make_apv_encoder(params: &VideoEncoderParams) -> Result<Box<dyn VideoEncoder>> {
    use oximedia_codec::{ApvConfig, ApvEncoder};

    let config = ApvConfig::new(params.width, params.height)
        .map_err(|e| TranscodeError::CodecError(e.to_string()))?
        .with_qp(params.quality);

    let encoder = ApvEncoder::new(config).map_err(|e| TranscodeError::CodecError(e.to_string()))?;

    Ok(Box::new(encoder))
}

#[cfg(not(feature = "apv"))]
fn make_apv_encoder(_params: &VideoEncoderParams) -> Result<Box<dyn VideoEncoder>> {
    Err(TranscodeError::Unsupported(
        "APV support requires the `apv` feature of oximedia-codec".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_new_valid() {
        let p = VideoEncoderParams::new(1920, 1080, 85);
        assert!(p.is_ok());
        let p = p.expect("valid params");
        assert_eq!(p.width, 1920);
        assert_eq!(p.height, 1080);
        assert_eq!(p.quality, 85);
    }

    #[test]
    fn test_params_zero_width() {
        assert!(VideoEncoderParams::new(0, 1080, 85).is_err());
    }

    #[test]
    fn test_params_zero_height() {
        assert!(VideoEncoderParams::new(1920, 0, 85).is_err());
    }

    #[test]
    fn test_unsupported_codec() {
        let p = VideoEncoderParams::new(320, 240, 30).expect("valid");
        let result = make_video_encoder(CodecId::Vp9, &p);
        assert!(result.is_err());
        // Extract the error without requiring Debug on the Ok variant.
        if let Err(e) = result {
            assert!(matches!(e, TranscodeError::Unsupported(_)));
        }
    }

    #[cfg(feature = "mjpeg")]
    #[test]
    fn test_make_mjpeg_encoder() {
        let p = VideoEncoderParams::new(320, 240, 85).expect("valid");
        let enc = make_video_encoder(CodecId::Mjpeg, &p);
        assert!(enc.is_ok(), "MJPEG encoder should build");
        let enc = enc.expect("ok");
        assert_eq!(enc.codec(), CodecId::Mjpeg);
    }

    #[cfg(feature = "apv")]
    #[test]
    fn test_make_apv_encoder() {
        let p = VideoEncoderParams::new(320, 240, 22).expect("valid");
        let enc = make_video_encoder(CodecId::Apv, &p);
        assert!(enc.is_ok(), "APV encoder should build");
        let enc = enc.expect("ok");
        assert_eq!(enc.codec(), CodecId::Apv);
    }

    #[cfg(not(feature = "mjpeg"))]
    #[test]
    fn test_mjpeg_disabled() {
        let p = VideoEncoderParams::new(320, 240, 85).expect("valid");
        let result = make_video_encoder(CodecId::Mjpeg, &p);
        assert!(matches!(result, Err(TranscodeError::Unsupported(_))));
    }

    #[cfg(not(feature = "apv"))]
    #[test]
    fn test_apv_disabled() {
        let p = VideoEncoderParams::new(320, 240, 22).expect("valid");
        let result = make_video_encoder(CodecId::Apv, &p);
        assert!(matches!(result, Err(TranscodeError::Unsupported(_))));
    }
}
