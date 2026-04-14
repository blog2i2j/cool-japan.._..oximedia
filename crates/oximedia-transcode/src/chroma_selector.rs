//! Chroma subsampling selector for per-codec optimal chroma mode.
//!
//! Video encoders differ in which chroma subsampling modes (4:4:4, 4:2:2, 4:2:0) they
//! support, and which modes produce the best quality-vs-bitrate tradeoff for a given
//! use-case.  This module provides:
//!
//! - [`ChromaSubsampling`] — an enum covering all three standard subsampling modes.
//! - [`ChromaPolicy`] — a preference ordering used when selecting a subsampling mode
//!   for an encode (e.g. "prefer highest quality", "prefer lowest bitrate").
//! - [`ChromaSelector`] — the main selector that, given a codec and a policy, returns
//!   the optimal [`ChromaSubsampling`].
//! - [`ChromaCompatibility`] — a query interface that reports which modes a given codec
//!   actually supports.

use std::fmt;

/// Standard chroma subsampling modes.
///
/// The notation `J:a:b` describes how many chroma samples appear per two rows of
/// `J` luma samples:
/// - `4:4:4` — no subsampling; full chroma resolution.
/// - `4:2:2` — horizontal halving of chroma; commonly used in broadcast.
/// - `4:2:0` — both horizontal and vertical halving; most common in consumer video.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ChromaSubsampling {
    /// 4:2:0 — lowest chroma resolution; smallest bitrate.
    Yuv420,
    /// 4:2:2 — medium chroma resolution; used in broadcast mastering.
    Yuv422,
    /// 4:4:4 — full chroma resolution; used in screen-content, VFX, and archival.
    Yuv444,
}

impl ChromaSubsampling {
    /// Returns the standard notation string (e.g. `"4:2:0"`).
    #[must_use]
    pub fn notation(self) -> &'static str {
        match self {
            Self::Yuv420 => "4:2:0",
            Self::Yuv422 => "4:2:2",
            Self::Yuv444 => "4:4:4",
        }
    }

    /// Relative chroma data size compared to 4:2:0 (normalised to 1.0).
    ///
    /// This is a rough guide; actual encoded file size depends on codec efficiency.
    #[must_use]
    pub fn relative_chroma_size(self) -> f32 {
        match self {
            Self::Yuv420 => 1.0,
            Self::Yuv422 => 2.0,
            Self::Yuv444 => 4.0,
        }
    }

    /// Horizontal chroma downscale factor relative to luma.
    #[must_use]
    pub fn horizontal_factor(self) -> u32 {
        match self {
            Self::Yuv420 | Self::Yuv422 => 2,
            Self::Yuv444 => 1,
        }
    }

    /// Vertical chroma downscale factor relative to luma.
    #[must_use]
    pub fn vertical_factor(self) -> u32 {
        match self {
            Self::Yuv420 => 2,
            Self::Yuv422 | Self::Yuv444 => 1,
        }
    }

    /// Parse a subsampling notation string (`"4:2:0"`, `"4:2:2"`, `"4:4:4"`).
    #[must_use]
    pub fn from_notation(s: &str) -> Option<Self> {
        match s.trim() {
            "4:2:0" | "yuv420" | "yuv420p" => Some(Self::Yuv420),
            "4:2:2" | "yuv422" | "yuv422p" => Some(Self::Yuv422),
            "4:4:4" | "yuv444" | "yuv444p" => Some(Self::Yuv444),
            _ => None,
        }
    }
}

impl fmt::Display for ChromaSubsampling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.notation())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Codecs
// ──────────────────────────────────────────────────────────────────────────────

/// Patent-free video codecs supported by [`ChromaSelector`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChromaCodec {
    /// VP8 — supports 4:2:0 only.
    Vp8,
    /// VP9 — supports 4:2:0, 4:2:2, and 4:4:4.
    Vp9,
    /// AV1 — supports 4:2:0, 4:2:2, and 4:4:4.
    Av1,
    /// FFV1 (lossless) — supports 4:2:0, 4:2:2, and 4:4:4.
    Ffv1,
    /// Theora — supports 4:2:0, 4:2:2, and 4:4:4.
    Theora,
}

impl ChromaCodec {
    /// Returns the set of [`ChromaSubsampling`] modes this codec supports,
    /// ordered from lowest to highest quality.
    #[must_use]
    pub fn supported_modes(self) -> &'static [ChromaSubsampling] {
        match self {
            Self::Vp8 => &[ChromaSubsampling::Yuv420],
            Self::Vp9 | Self::Av1 | Self::Ffv1 | Self::Theora => &[
                ChromaSubsampling::Yuv420,
                ChromaSubsampling::Yuv422,
                ChromaSubsampling::Yuv444,
            ],
        }
    }

    /// Returns `true` if `mode` is supported by this codec.
    #[must_use]
    pub fn supports(self, mode: ChromaSubsampling) -> bool {
        self.supported_modes().contains(&mode)
    }

    /// Canonical name of this codec (lowercase).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Vp8 => "vp8",
            Self::Vp9 => "vp9",
            Self::Av1 => "av1",
            Self::Ffv1 => "ffv1",
            Self::Theora => "theora",
        }
    }

    /// Parse a codec name string (case-insensitive).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().trim() {
            "vp8" | "libvpx" => Some(Self::Vp8),
            "vp9" | "libvpx-vp9" => Some(Self::Vp9),
            "av1" | "libaom-av1" | "libsvtav1" | "librav1e" => Some(Self::Av1),
            "ffv1" => Some(Self::Ffv1),
            "theora" | "libtheora" => Some(Self::Theora),
            _ => None,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Policy
// ──────────────────────────────────────────────────────────────────────────────

/// Policy that governs how the selector picks among supported subsampling modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChromaPolicy {
    /// Prefer the lowest subsampling supported (smallest encoded size).
    ///
    /// For most consumer-video codecs this resolves to 4:2:0.
    #[default]
    MinBitrate,

    /// Prefer the highest subsampling supported (highest chroma quality).
    ///
    /// Useful for VFX compositing, screen-content, and archival workflows.
    MaxQuality,

    /// Prefer 4:2:2 when supported; fall back to 4:2:0.
    ///
    /// Matches typical broadcast-mastering requirements (e.g. ProRes, XDCAM).
    BroadcastPreferred,

    /// Prefer a specific mode; fall back to the closest supported mode.
    Exact(ChromaSubsampling),
}

// ──────────────────────────────────────────────────────────────────────────────
// Errors
// ──────────────────────────────────────────────────────────────────────────────

/// Errors from the chroma selector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChromaSelectorError {
    /// The codec name was not recognised.
    UnknownCodec(String),
    /// The requested mode is not supported and no fallback exists.
    ModeNotSupported {
        /// The mode that was requested.
        requested: ChromaSubsampling,
        /// The codec that rejected it.
        codec: &'static str,
    },
}

impl fmt::Display for ChromaSelectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownCodec(name) => write!(f, "unknown codec '{name}'"),
            Self::ModeNotSupported { requested, codec } => write!(
                f,
                "chroma mode {} is not supported by codec '{codec}'",
                requested.notation()
            ),
        }
    }
}

impl std::error::Error for ChromaSelectorError {}

// ──────────────────────────────────────────────────────────────────────────────
// Selector
// ──────────────────────────────────────────────────────────────────────────────

/// Selects the optimal chroma subsampling mode for a codec and policy.
///
/// # Example
///
/// ```
/// use oximedia_transcode::chroma_selector::{ChromaSelector, ChromaCodec, ChromaPolicy};
///
/// let mode = ChromaSelector::select(ChromaCodec::Vp9, ChromaPolicy::MaxQuality)
///     .expect("VP9 supports 4:4:4");
/// assert_eq!(mode.notation(), "4:4:4");
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ChromaSelector;

impl ChromaSelector {
    /// Select the optimal [`ChromaSubsampling`] for `codec` under `policy`.
    ///
    /// # Errors
    ///
    /// Returns [`ChromaSelectorError::ModeNotSupported`] if the exact mode
    /// requested by [`ChromaPolicy::Exact`] is not supported and there are
    /// no supported modes (which cannot happen for a well-formed codec table,
    /// but is handled defensively).
    pub fn select(
        codec: ChromaCodec,
        policy: ChromaPolicy,
    ) -> Result<ChromaSubsampling, ChromaSelectorError> {
        let supported = codec.supported_modes();

        match policy {
            ChromaPolicy::MinBitrate => {
                // Lowest subsampling = first element (ordered ascending).
                supported
                    .iter()
                    .copied()
                    .min()
                    .ok_or(ChromaSelectorError::ModeNotSupported {
                        requested: ChromaSubsampling::Yuv420,
                        codec: codec.name(),
                    })
            }
            ChromaPolicy::MaxQuality => {
                // Highest subsampling = last element (ordered ascending).
                supported
                    .iter()
                    .copied()
                    .max()
                    .ok_or(ChromaSelectorError::ModeNotSupported {
                        requested: ChromaSubsampling::Yuv444,
                        codec: codec.name(),
                    })
            }
            ChromaPolicy::BroadcastPreferred => {
                // Prefer 4:2:2; fall back to the best available.
                if supported.contains(&ChromaSubsampling::Yuv422) {
                    Ok(ChromaSubsampling::Yuv422)
                } else {
                    // Fall back to highest supported (e.g. VP8 → 4:2:0).
                    supported
                        .iter()
                        .copied()
                        .max()
                        .ok_or(ChromaSelectorError::ModeNotSupported {
                            requested: ChromaSubsampling::Yuv422,
                            codec: codec.name(),
                        })
                }
            }
            ChromaPolicy::Exact(mode) => {
                if supported.contains(&mode) {
                    Ok(mode)
                } else {
                    Err(ChromaSelectorError::ModeNotSupported {
                        requested: mode,
                        codec: codec.name(),
                    })
                }
            }
        }
    }

    /// Select using codec name strings.
    ///
    /// # Errors
    ///
    /// Returns [`ChromaSelectorError::UnknownCodec`] if the name is unrecognised,
    /// or [`ChromaSelectorError::ModeNotSupported`] as per [`Self::select`].
    pub fn select_by_name(
        codec_name: &str,
        policy: ChromaPolicy,
    ) -> Result<ChromaSubsampling, ChromaSelectorError> {
        let codec = ChromaCodec::from_name(codec_name)
            .ok_or_else(|| ChromaSelectorError::UnknownCodec(codec_name.to_owned()))?;
        Self::select(codec, policy)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Compatibility query helper
// ──────────────────────────────────────────────────────────────────────────────

/// Query interface for chroma-subsampling codec compatibility.
pub struct ChromaCompatibility;

impl ChromaCompatibility {
    /// Returns all [`ChromaSubsampling`] modes supported by `codec`.
    #[must_use]
    pub fn supported_modes(codec: ChromaCodec) -> &'static [ChromaSubsampling] {
        codec.supported_modes()
    }

    /// Returns `true` if `codec` supports `mode`.
    #[must_use]
    pub fn supports(codec: ChromaCodec, mode: ChromaSubsampling) -> bool {
        codec.supports(mode)
    }

    /// Returns every codec that supports `mode`.
    #[must_use]
    pub fn codecs_supporting(mode: ChromaSubsampling) -> Vec<ChromaCodec> {
        [
            ChromaCodec::Vp8,
            ChromaCodec::Vp9,
            ChromaCodec::Av1,
            ChromaCodec::Ffv1,
            ChromaCodec::Theora,
        ]
        .iter()
        .copied()
        .filter(|c| c.supports(mode))
        .collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ChromaSubsampling ──────────────────────────────────────────────────────

    #[test]
    fn test_notation_strings() {
        assert_eq!(ChromaSubsampling::Yuv420.notation(), "4:2:0");
        assert_eq!(ChromaSubsampling::Yuv422.notation(), "4:2:2");
        assert_eq!(ChromaSubsampling::Yuv444.notation(), "4:4:4");
    }

    #[test]
    fn test_from_notation_aliases() {
        assert_eq!(
            ChromaSubsampling::from_notation("yuv420p"),
            Some(ChromaSubsampling::Yuv420)
        );
        assert_eq!(
            ChromaSubsampling::from_notation("4:2:2"),
            Some(ChromaSubsampling::Yuv422)
        );
        assert_eq!(
            ChromaSubsampling::from_notation("yuv444"),
            Some(ChromaSubsampling::Yuv444)
        );
        assert_eq!(ChromaSubsampling::from_notation("unknown"), None);
    }

    #[test]
    fn test_relative_chroma_size_ordering() {
        assert!(
            ChromaSubsampling::Yuv420.relative_chroma_size()
                < ChromaSubsampling::Yuv422.relative_chroma_size()
        );
        assert!(
            ChromaSubsampling::Yuv422.relative_chroma_size()
                < ChromaSubsampling::Yuv444.relative_chroma_size()
        );
    }

    #[test]
    fn test_vertical_factor_420_vs_422() {
        assert_eq!(ChromaSubsampling::Yuv420.vertical_factor(), 2);
        assert_eq!(ChromaSubsampling::Yuv422.vertical_factor(), 1);
        assert_eq!(ChromaSubsampling::Yuv444.vertical_factor(), 1);
    }

    #[test]
    fn test_display_format() {
        assert_eq!(format!("{}", ChromaSubsampling::Yuv422), "4:2:2");
    }

    // ── ChromaCodec ────────────────────────────────────────────────────────────

    #[test]
    fn test_vp8_only_supports_420() {
        let supported = ChromaCodec::Vp8.supported_modes();
        assert_eq!(supported, &[ChromaSubsampling::Yuv420]);
        assert!(!ChromaCodec::Vp8.supports(ChromaSubsampling::Yuv422));
        assert!(!ChromaCodec::Vp8.supports(ChromaSubsampling::Yuv444));
    }

    #[test]
    fn test_vp9_supports_all_three() {
        assert!(ChromaCodec::Vp9.supports(ChromaSubsampling::Yuv420));
        assert!(ChromaCodec::Vp9.supports(ChromaSubsampling::Yuv422));
        assert!(ChromaCodec::Vp9.supports(ChromaSubsampling::Yuv444));
    }

    #[test]
    fn test_codec_from_name_aliases() {
        assert_eq!(ChromaCodec::from_name("libvpx-vp9"), Some(ChromaCodec::Vp9));
        assert_eq!(ChromaCodec::from_name("libaom-av1"), Some(ChromaCodec::Av1));
        assert_eq!(ChromaCodec::from_name("libsvtav1"), Some(ChromaCodec::Av1));
        assert_eq!(ChromaCodec::from_name("xyz"), None);
    }

    // ── ChromaSelector ─────────────────────────────────────────────────────────

    #[test]
    fn test_min_bitrate_vp9_gives_420() {
        let mode = ChromaSelector::select(ChromaCodec::Vp9, ChromaPolicy::MinBitrate)
            .expect("VP9 supports 4:2:0");
        assert_eq!(mode, ChromaSubsampling::Yuv420);
    }

    #[test]
    fn test_max_quality_vp9_gives_444() {
        let mode = ChromaSelector::select(ChromaCodec::Vp9, ChromaPolicy::MaxQuality)
            .expect("VP9 supports 4:4:4");
        assert_eq!(mode, ChromaSubsampling::Yuv444);
    }

    #[test]
    fn test_max_quality_vp8_gives_420_fallback() {
        // VP8 only supports 4:2:0, so MaxQuality still resolves to 4:2:0.
        let mode = ChromaSelector::select(ChromaCodec::Vp8, ChromaPolicy::MaxQuality)
            .expect("VP8 supports 4:2:0");
        assert_eq!(mode, ChromaSubsampling::Yuv420);
    }

    #[test]
    fn test_broadcast_preferred_vp9_gives_422() {
        let mode =
            ChromaSelector::select(ChromaCodec::Vp9, ChromaPolicy::BroadcastPreferred)
                .expect("VP9 supports 4:2:2");
        assert_eq!(mode, ChromaSubsampling::Yuv422);
    }

    #[test]
    fn test_broadcast_preferred_vp8_fallback_to_420() {
        // VP8 doesn't have 4:2:2, so fallback to best available = 4:2:0.
        let mode =
            ChromaSelector::select(ChromaCodec::Vp8, ChromaPolicy::BroadcastPreferred)
                .expect("VP8 falls back to 4:2:0");
        assert_eq!(mode, ChromaSubsampling::Yuv420);
    }

    #[test]
    fn test_exact_mode_supported() {
        let mode = ChromaSelector::select(
            ChromaCodec::Av1,
            ChromaPolicy::Exact(ChromaSubsampling::Yuv422),
        )
        .expect("AV1 supports 4:2:2");
        assert_eq!(mode, ChromaSubsampling::Yuv422);
    }

    #[test]
    fn test_exact_mode_unsupported_returns_error() {
        let err = ChromaSelector::select(
            ChromaCodec::Vp8,
            ChromaPolicy::Exact(ChromaSubsampling::Yuv444),
        )
        .expect_err("VP8 does not support 4:4:4");
        assert!(matches!(err, ChromaSelectorError::ModeNotSupported { .. }));
    }

    #[test]
    fn test_select_by_name_valid() {
        let mode = ChromaSelector::select_by_name("av1", ChromaPolicy::MaxQuality)
            .expect("AV1 is known");
        assert_eq!(mode, ChromaSubsampling::Yuv444);
    }

    #[test]
    fn test_select_by_name_unknown_codec() {
        let err = ChromaSelector::select_by_name("h264", ChromaPolicy::MinBitrate)
            .expect_err("h264 is not in the patent-free list");
        assert!(matches!(err, ChromaSelectorError::UnknownCodec(_)));
    }

    // ── ChromaCompatibility ────────────────────────────────────────────────────

    #[test]
    fn test_codecs_supporting_422_excludes_vp8() {
        let codecs = ChromaCompatibility::codecs_supporting(ChromaSubsampling::Yuv422);
        assert!(!codecs.contains(&ChromaCodec::Vp8));
        assert!(codecs.contains(&ChromaCodec::Vp9));
        assert!(codecs.contains(&ChromaCodec::Av1));
    }

    #[test]
    fn test_error_display_unknown_codec() {
        let err = ChromaSelectorError::UnknownCodec("h265".to_owned());
        assert!(err.to_string().contains("h265"));
    }
}
