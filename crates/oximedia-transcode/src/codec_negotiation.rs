//! Codec capability negotiation for the transcoding pipeline.
//!
//! This module provides a sophisticated codec negotiation system that matches
//! source codec characteristics to the best available target codec, with
//! multi-level fallback chains, capability matrices, and container compatibility
//! checks.
//!
//! # Overview
//!
//! When building a transcode pipeline it is often necessary to choose a target
//! codec that:
//! 1. Is compatible with the requested output container.
//! 2. Supports the required feature set (HDR, high bit-depth, …).
//! 3. Delivers the best quality/efficiency trade-off for the media type.
//! 4. Falls back gracefully when the first-choice codec is unavailable.
//!
//! [`CodecNegotiator`] encodes all of this logic through a capability matrix
//! and an ordered [`FallbackChain`].

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::TranscodeError;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Media class — separates video and audio negotiation paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum MediaClass {
    /// A video (picture-bearing) codec.
    #[default]
    Video,
    /// An audio codec.
    Audio,
}

/// A capability flag that a codec may or may not support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CodecCapability {
    /// The codec can encode/decode 10-bit samples.
    TenBit,
    /// The codec can encode/decode 12-bit samples.
    TwelveBit,
    /// Supports HDR metadata (PQ / HLG / Dolby Vision).
    HdrMetadata,
    /// The codec supports lossless operation.
    Lossless,
    /// Multi-thread / tile parallel encoding is available.
    ParallelEncoding,
    /// Supports variable bit-rate operation.
    VariableBitrate,
    /// The codec can operate on alpha-channel (transparency) data.
    AlphaChannel,
    /// The codec supports spatial audio (HOA / binaural) metadata.
    SpatialAudio,
    /// Multi-channel audio (>2 channels).
    MultiChannel,
    /// Low-latency encoding mode.
    LowLatency,
}

/// Describes a single codec entry in the capability matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecEntry {
    /// Canonical codec identifier (e.g. `"vp9"`, `"av1"`, `"opus"`).
    pub id: String,
    /// Human-readable display name.
    pub display_name: String,
    /// Whether this is a video or audio codec.
    pub media_class: MediaClass,
    /// Containers this codec can be placed in (e.g. `["webm", "mp4"]`).
    pub containers: Vec<String>,
    /// Capabilities this codec provides.
    pub capabilities: Vec<CodecCapability>,
    /// Relative encoding cost (1.0 = baseline). Higher = slower/costlier.
    pub encoding_cost: f32,
    /// Relative quality score at equal bitrate (higher is better).
    pub quality_score: f32,
    /// Whether this codec is patent-free.
    pub patent_free: bool,
}

impl CodecEntry {
    /// Returns `true` if the codec supports the given capability.
    #[must_use]
    pub fn has_capability(&self, cap: CodecCapability) -> bool {
        self.capabilities.contains(&cap)
    }

    /// Returns `true` if the codec is compatible with the given container.
    #[must_use]
    pub fn supports_container(&self, container: &str) -> bool {
        let c = container.to_lowercase();
        self.containers.iter().any(|ct| ct.as_str() == c.as_str())
    }
}

/// A request to negotiate a codec.
#[derive(Debug, Clone, Default)]
pub struct NegotiationRequest {
    /// Desired target codec id.  When `None` the negotiator picks the best.
    pub preferred_codec: Option<String>,
    /// Required output container (e.g. `"webm"`, `"mp4"`, `"mkv"`).
    pub container: Option<String>,
    /// Capabilities that *must* be present in the selected codec.
    pub required_capabilities: Vec<CodecCapability>,
    /// Capabilities that are preferred but not mandatory.
    pub preferred_capabilities: Vec<CodecCapability>,
    /// Media class to negotiate.
    pub media_class: MediaClass,
    /// Whether only patent-free codecs are acceptable.
    pub patent_free_only: bool,
    /// Source codec id (used for same-codec copy preference).
    pub source_codec: Option<String>,
}

/// The outcome of a codec negotiation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegotiationResult {
    /// The selected codec entry.
    pub codec: CodecEntry,
    /// The fallback depth at which this codec was chosen (0 = first choice).
    pub fallback_depth: usize,
    /// Whether the chosen codec matches the source codec (stream copy eligible).
    pub is_copy_eligible: bool,
    /// Score used to rank this codec (higher is better).
    pub score: f32,
}

/// An ordered list of codec ids to try, from most-preferred to least.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FallbackChain {
    /// Ordered codec ids.
    pub chain: Vec<String>,
}

impl FallbackChain {
    /// Creates a new empty fallback chain.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a codec id to the end of the chain.
    #[must_use]
    pub fn then(mut self, codec_id: impl Into<String>) -> Self {
        self.chain.push(codec_id.into());
        self
    }

    /// Returns the standard patent-free video fallback chain.
    ///
    /// Preference order: AV1 → VP9 → VP8 → FFV1 (lossless fallback).
    #[must_use]
    pub fn patent_free_video() -> Self {
        Self::new()
            .then("av1")
            .then("vp9")
            .then("vp8")
            .then("ffv1")
    }

    /// Returns the standard patent-free audio fallback chain.
    ///
    /// Preference order: Opus → Vorbis → FLAC.
    #[must_use]
    pub fn patent_free_audio() -> Self {
        Self::new().then("opus").then("vorbis").then("flac")
    }

    /// Returns a streaming-optimised video fallback chain.
    ///
    /// Preference order: AV1 → VP9 → VP8.
    #[must_use]
    pub fn streaming_video() -> Self {
        Self::new().then("av1").then("vp9").then("vp8")
    }
}

/// Core negotiation engine.
///
/// Holds a capability matrix of known codecs and resolves [`NegotiationRequest`]s
/// against it, producing a ranked [`NegotiationResult`].
#[derive(Debug, Clone)]
pub struct CodecNegotiator {
    /// All known codecs, keyed by canonical id.
    matrix: HashMap<String, CodecEntry>,
}

impl Default for CodecNegotiator {
    fn default() -> Self {
        Self::with_default_matrix()
    }
}

impl CodecNegotiator {
    /// Creates a new negotiator with an empty capability matrix.
    #[must_use]
    pub fn new() -> Self {
        Self {
            matrix: HashMap::new(),
        }
    }

    /// Creates a negotiator pre-populated with the built-in patent-free codec
    /// matrix.
    #[must_use]
    pub fn with_default_matrix() -> Self {
        let mut n = Self::new();
        n.register_default_codecs();
        n
    }

    /// Registers a codec in the capability matrix.
    ///
    /// If a codec with the same `id` already exists it is replaced.
    pub fn register(&mut self, entry: CodecEntry) {
        self.matrix.insert(entry.id.clone(), entry);
    }

    /// Looks up a codec by id.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&CodecEntry> {
        self.matrix.get(id)
    }

    /// Negotiate the best codec for `req`, walking `chain` until a suitable
    /// match is found.
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::Unsupported`] when no codec in the chain
    /// satisfies the constraints.
    pub fn negotiate(
        &self,
        req: &NegotiationRequest,
        chain: &FallbackChain,
    ) -> Result<NegotiationResult, TranscodeError> {
        // Build the ordered list of candidates from the chain, augmented by
        // the caller's preferred codec (inserted at the front).
        let mut candidates: Vec<String> = Vec::new();
        if let Some(ref pref) = req.preferred_codec {
            candidates.push(pref.clone());
        }
        for id in &chain.chain {
            if !candidates.contains(id) {
                candidates.push(id.clone());
            }
        }

        for (depth, id) in candidates.iter().enumerate() {
            if let Some(entry) = self.matrix.get(id.as_str()) {
                if let Some(result) = self.evaluate(entry, req, depth) {
                    return Ok(result);
                }
            }
        }

        Err(TranscodeError::Unsupported(format!(
            "No suitable {:?} codec found for the given constraints",
            req.media_class
        )))
    }

    /// Negotiate without a custom chain, using the built-in patent-free chain
    /// for the requested media class.
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::Unsupported`] when no codec satisfies the
    /// constraints.
    pub fn negotiate_default(
        &self,
        req: &NegotiationRequest,
    ) -> Result<NegotiationResult, TranscodeError> {
        let chain = match req.media_class {
            MediaClass::Video => FallbackChain::patent_free_video(),
            MediaClass::Audio => FallbackChain::patent_free_audio(),
        };
        self.negotiate(req, &chain)
    }

    /// Returns all registered codecs for the given media class, sorted by
    /// quality score descending.
    #[must_use]
    pub fn all_for_class(&self, class: MediaClass) -> Vec<&CodecEntry> {
        let mut entries: Vec<&CodecEntry> = self
            .matrix
            .values()
            .filter(|e| e.media_class == class)
            .collect();
        entries.sort_by(|a, b| b.quality_score.partial_cmp(&a.quality_score).unwrap_or(std::cmp::Ordering::Equal));
        entries
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Returns `Some(result)` if `entry` satisfies all hard constraints in
    /// `req`, or `None` if any constraint is violated.
    fn evaluate(
        &self,
        entry: &CodecEntry,
        req: &NegotiationRequest,
        depth: usize,
    ) -> Option<NegotiationResult> {
        // Hard filter: media class must match.
        if entry.media_class != req.media_class {
            return None;
        }

        // Hard filter: patent-free requirement.
        if req.patent_free_only && !entry.patent_free {
            return None;
        }

        // Hard filter: container compatibility.
        if let Some(ref container) = req.container {
            if !entry.supports_container(container) {
                return None;
            }
        }

        // Hard filter: required capabilities.
        for &cap in &req.required_capabilities {
            if !entry.has_capability(cap) {
                return None;
            }
        }

        // Score: start from quality_score, penalise encoding cost, reward
        // preferred capabilities and copy eligibility.
        let mut score: f32 = entry.quality_score - (entry.encoding_cost - 1.0) * 0.1;

        for &cap in &req.preferred_capabilities {
            if entry.has_capability(cap) {
                score += 0.1;
            }
        }

        // Bonus for matching source codec (stream copy).
        let is_copy_eligible = req
            .source_codec
            .as_deref()
            .map_or(false, |src| src == entry.id.as_str());
        if is_copy_eligible {
            score += 1.0;
        }

        // Small penalty for deeper fallback positions.
        score -= depth as f32 * 0.05;

        Some(NegotiationResult {
            codec: entry.clone(),
            fallback_depth: depth,
            is_copy_eligible,
            score,
        })
    }

    /// Populates the matrix with the built-in patent-free codec entries.
    fn register_default_codecs(&mut self) {
        let video_entries = vec![
            CodecEntry {
                id: "av1".into(),
                display_name: "AV1".into(),
                media_class: MediaClass::Video,
                containers: vec!["mp4".into(), "mkv".into(), "webm".into(), "ogg".into()],
                capabilities: vec![
                    CodecCapability::TenBit,
                    CodecCapability::TwelveBit,
                    CodecCapability::HdrMetadata,
                    CodecCapability::Lossless,
                    CodecCapability::ParallelEncoding,
                    CodecCapability::VariableBitrate,
                ],
                encoding_cost: 4.0,
                quality_score: 0.95,
                patent_free: true,
            },
            CodecEntry {
                id: "vp9".into(),
                display_name: "VP9".into(),
                media_class: MediaClass::Video,
                containers: vec!["webm".into(), "mkv".into(), "mp4".into()],
                capabilities: vec![
                    CodecCapability::TenBit,
                    CodecCapability::HdrMetadata,
                    CodecCapability::Lossless,
                    CodecCapability::ParallelEncoding,
                    CodecCapability::VariableBitrate,
                ],
                encoding_cost: 3.0,
                quality_score: 0.90,
                patent_free: true,
            },
            CodecEntry {
                id: "vp8".into(),
                display_name: "VP8".into(),
                media_class: MediaClass::Video,
                containers: vec!["webm".into(), "mkv".into()],
                capabilities: vec![CodecCapability::VariableBitrate],
                encoding_cost: 1.5,
                quality_score: 0.75,
                patent_free: true,
            },
            CodecEntry {
                id: "ffv1".into(),
                display_name: "FFV1".into(),
                media_class: MediaClass::Video,
                containers: vec!["mkv".into(), "avi".into()],
                capabilities: vec![
                    CodecCapability::TenBit,
                    CodecCapability::TwelveBit,
                    CodecCapability::Lossless,
                    CodecCapability::AlphaChannel,
                    CodecCapability::ParallelEncoding,
                ],
                encoding_cost: 2.0,
                quality_score: 1.0,
                patent_free: true,
            },
            CodecEntry {
                id: "theora".into(),
                display_name: "Theora".into(),
                media_class: MediaClass::Video,
                containers: vec!["ogv".into(), "ogg".into(), "mkv".into()],
                capabilities: vec![CodecCapability::VariableBitrate],
                encoding_cost: 1.2,
                quality_score: 0.65,
                patent_free: true,
            },
        ];

        let audio_entries = vec![
            CodecEntry {
                id: "opus".into(),
                display_name: "Opus".into(),
                media_class: MediaClass::Audio,
                containers: vec![
                    "webm".into(),
                    "mkv".into(),
                    "mp4".into(),
                    "ogg".into(),
                    "opus".into(),
                ],
                capabilities: vec![
                    CodecCapability::MultiChannel,
                    CodecCapability::SpatialAudio,
                    CodecCapability::LowLatency,
                    CodecCapability::VariableBitrate,
                ],
                encoding_cost: 1.0,
                quality_score: 0.95,
                patent_free: true,
            },
            CodecEntry {
                id: "vorbis".into(),
                display_name: "Vorbis".into(),
                media_class: MediaClass::Audio,
                containers: vec!["webm".into(), "mkv".into(), "ogg".into()],
                capabilities: vec![CodecCapability::MultiChannel, CodecCapability::VariableBitrate],
                encoding_cost: 1.1,
                quality_score: 0.85,
                patent_free: true,
            },
            CodecEntry {
                id: "flac".into(),
                display_name: "FLAC".into(),
                media_class: MediaClass::Audio,
                containers: vec!["mkv".into(), "flac".into(), "mp4".into(), "ogg".into()],
                capabilities: vec![
                    CodecCapability::Lossless,
                    CodecCapability::MultiChannel,
                    CodecCapability::TenBit,
                    CodecCapability::TwelveBit,
                ],
                encoding_cost: 1.5,
                quality_score: 1.0,
                patent_free: true,
            },
            CodecEntry {
                id: "pcm_s16le".into(),
                display_name: "PCM 16-bit LE".into(),
                media_class: MediaClass::Audio,
                containers: vec!["wav".into(), "mkv".into(), "avi".into()],
                capabilities: vec![
                    CodecCapability::Lossless,
                    CodecCapability::MultiChannel,
                    CodecCapability::LowLatency,
                ],
                encoding_cost: 0.5,
                quality_score: 1.0,
                patent_free: true,
            },
            CodecEntry {
                id: "pcm_s24le".into(),
                display_name: "PCM 24-bit LE".into(),
                media_class: MediaClass::Audio,
                containers: vec!["wav".into(), "mkv".into(), "avi".into()],
                capabilities: vec![
                    CodecCapability::Lossless,
                    CodecCapability::MultiChannel,
                    CodecCapability::LowLatency,
                    CodecCapability::TenBit,
                ],
                encoding_cost: 0.5,
                quality_score: 1.0,
                patent_free: true,
            },
        ];

        for entry in video_entries.into_iter().chain(audio_entries) {
            self.register(entry);
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Quickly negotiate the best patent-free video codec for the given container.
///
/// # Errors
///
/// Returns [`TranscodeError::Unsupported`] when no match is found.
pub fn negotiate_video_codec(
    container: &str,
    required: &[CodecCapability],
) -> Result<NegotiationResult, TranscodeError> {
    let negotiator = CodecNegotiator::with_default_matrix();
    let req = NegotiationRequest {
        media_class: MediaClass::Video,
        container: Some(container.to_lowercase()),
        required_capabilities: required.to_vec(),
        patent_free_only: true,
        ..Default::default()
    };
    negotiator.negotiate_default(&req)
}

/// Quickly negotiate the best patent-free audio codec for the given container.
///
/// # Errors
///
/// Returns [`TranscodeError::Unsupported`] when no match is found.
pub fn negotiate_audio_codec(
    container: &str,
    required: &[CodecCapability],
) -> Result<NegotiationResult, TranscodeError> {
    let negotiator = CodecNegotiator::with_default_matrix();
    let req = NegotiationRequest {
        media_class: MediaClass::Audio,
        container: Some(container.to_lowercase()),
        required_capabilities: required.to_vec(),
        patent_free_only: true,
        ..Default::default()
    };
    negotiator.negotiate_default(&req)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn negotiator() -> CodecNegotiator {
        CodecNegotiator::with_default_matrix()
    }

    // --- video negotiation ---------------------------------------------------

    #[test]
    fn test_video_webm_av1_first_choice() {
        let n = negotiator();
        let req = NegotiationRequest {
            media_class: MediaClass::Video,
            container: Some("webm".into()),
            patent_free_only: true,
            ..Default::default()
        };
        let result = n.negotiate_default(&req).expect("should find codec");
        assert_eq!(result.codec.id, "av1");
        assert_eq!(result.fallback_depth, 0);
    }

    #[test]
    fn test_video_fallback_to_vp9_when_av1_not_in_chain() {
        let n = negotiator();
        let chain = FallbackChain::new().then("vp9").then("vp8");
        let req = NegotiationRequest {
            media_class: MediaClass::Video,
            container: Some("webm".into()),
            patent_free_only: true,
            ..Default::default()
        };
        let result = n.negotiate(&req, &chain).expect("should find vp9");
        assert_eq!(result.codec.id, "vp9");
        assert_eq!(result.fallback_depth, 0);
    }

    #[test]
    fn test_video_lossless_requires_capability() {
        let n = negotiator();
        let req = NegotiationRequest {
            media_class: MediaClass::Video,
            container: Some("mkv".into()),
            required_capabilities: vec![CodecCapability::Lossless],
            patent_free_only: true,
            ..Default::default()
        };
        let result = n.negotiate_default(&req).expect("should find lossless codec");
        assert!(result.codec.has_capability(CodecCapability::Lossless));
    }

    #[test]
    fn test_video_twelve_bit_required() {
        let n = negotiator();
        let req = NegotiationRequest {
            media_class: MediaClass::Video,
            container: Some("mkv".into()),
            required_capabilities: vec![CodecCapability::TwelveBit],
            patent_free_only: true,
            ..Default::default()
        };
        let result = n.negotiate_default(&req).expect("12-bit codec found");
        assert!(result.codec.has_capability(CodecCapability::TwelveBit));
    }

    #[test]
    fn test_video_unsupported_container_returns_error() {
        let n = negotiator();
        let req = NegotiationRequest {
            media_class: MediaClass::Video,
            // "ts" is not supported by any default codec
            container: Some("ts".into()),
            patent_free_only: true,
            ..Default::default()
        };
        assert!(n.negotiate_default(&req).is_err());
    }

    // --- audio negotiation ---------------------------------------------------

    #[test]
    fn test_audio_webm_opus_first_choice() {
        let result = negotiate_audio_codec("webm", &[]).expect("opus for webm");
        assert_eq!(result.codec.id, "opus");
    }

    #[test]
    fn test_audio_flac_lossless_in_mkv() {
        let n = negotiator();
        let req = NegotiationRequest {
            media_class: MediaClass::Audio,
            container: Some("mkv".into()),
            required_capabilities: vec![CodecCapability::Lossless],
            patent_free_only: true,
            ..Default::default()
        };
        let result = n.negotiate_default(&req).expect("lossless audio in mkv");
        assert!(result.codec.has_capability(CodecCapability::Lossless));
    }

    #[test]
    fn test_audio_spatial_audio_selects_opus() {
        let n = negotiator();
        let req = NegotiationRequest {
            media_class: MediaClass::Audio,
            container: Some("mkv".into()),
            required_capabilities: vec![CodecCapability::SpatialAudio],
            patent_free_only: true,
            ..Default::default()
        };
        let result = n.negotiate_default(&req).expect("spatial audio codec");
        assert_eq!(result.codec.id, "opus");
    }

    // --- copy-eligibility ----------------------------------------------------

    #[test]
    fn test_copy_eligible_when_source_matches() {
        let n = negotiator();
        let req = NegotiationRequest {
            media_class: MediaClass::Video,
            container: Some("webm".into()),
            source_codec: Some("vp9".into()),
            preferred_codec: Some("vp9".into()),
            patent_free_only: true,
            ..Default::default()
        };
        let chain = FallbackChain::new().then("vp9");
        let result = n.negotiate(&req, &chain).expect("vp9 negotiated");
        assert!(result.is_copy_eligible, "should be copy eligible");
    }

    // --- fallback chain helpers ----------------------------------------------

    #[test]
    fn test_fallback_chain_then_builds_correctly() {
        let chain = FallbackChain::new().then("av1").then("vp9").then("vp8");
        assert_eq!(chain.chain, vec!["av1", "vp9", "vp8"]);
    }

    #[test]
    fn test_all_for_class_sorted_by_quality() {
        let n = negotiator();
        let videos = n.all_for_class(MediaClass::Video);
        for window in videos.windows(2) {
            assert!(
                window[0].quality_score >= window[1].quality_score,
                "Expected descending quality order"
            );
        }
    }

    // --- convenience functions -----------------------------------------------

    #[test]
    fn test_negotiate_video_codec_convenience() {
        let result = negotiate_video_codec("webm", &[]).expect("video codec for webm");
        assert!(result.codec.patent_free);
        assert_eq!(result.codec.media_class, MediaClass::Video);
    }

    #[test]
    fn test_negotiate_video_codec_unsatisfiable_capability() {
        // No patent-free video codec supports a fictional capability.
        // We simulate this by requiring both SpatialAudio AND Lossless on video.
        let result = negotiate_video_codec(
            "webm",
            &[CodecCapability::SpatialAudio, CodecCapability::LowLatency],
        );
        assert!(result.is_err(), "should be unsatisfiable");
    }
}
