// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Smart codec selection for target containers.
//!
//! Selects the best codec for a given container, optionally favouring quality
//! over encoding speed / compatibility.

/// Select the best codec for the given `container`.
///
/// When `prefer_quality` is `true`, the selection favours codecs with superior
/// compression efficiency (e.g. AV1 > VP9 > VP8 for WebM).  When `false`, it
/// favours faster-to-encode / more widely-compatible codecs.
///
/// Returns a `&'static str` codec identifier compatible with
/// [`ContainerCodecMatrix::is_compatible`][crate::compat_matrix::ContainerCodecMatrix::is_compatible].
///
/// # Examples
///
/// ```
/// use oximedia_convert::smart_select::select_codec_for_container;
///
/// assert_eq!(select_codec_for_container("webm", true), "av1");
/// assert_eq!(select_codec_for_container("webm", false), "vp9");
/// assert_eq!(select_codec_for_container("mp4", true), "hevc");
/// assert_eq!(select_codec_for_container("mp4", false), "avc");
/// ```
pub fn select_codec_for_container(container: &str, prefer_quality: bool) -> &'static str {
    match container.to_ascii_lowercase().as_str() {
        "webm" => {
            if prefer_quality {
                "av1"
            } else {
                "vp9"
            }
        }
        "mkv" | "matroska" => {
            if prefer_quality {
                "av1"
            } else {
                "avc"
            }
        }
        "mp4" | "m4v" => {
            if prefer_quality {
                "hevc"
            } else {
                "avc"
            }
        }
        "mov" => {
            if prefer_quality {
                "hevc"
            } else {
                "avc"
            }
        }
        "ts" | "m2ts" | "mts" => "avc",
        "ogg" | "ogv" => "theora",
        "oga" | "opus" => "opus",
        "flac" => "flac",
        "wav" => "pcm",
        "avi" => "avc",
        // For unknown containers default to VP9 (broadly compatible open codec)
        _ => "vp9",
    }
}

/// Select the best *audio* codec for `container`.
///
/// When `prefer_quality` is `true`, lossless or high-efficiency codecs are
/// preferred.  Otherwise a fast, lossy codec is returned.
pub fn select_audio_codec_for_container(container: &str, prefer_quality: bool) -> &'static str {
    match container.to_ascii_lowercase().as_str() {
        "webm" => "opus",
        "mkv" | "matroska" => {
            if prefer_quality {
                "flac"
            } else {
                "opus"
            }
        }
        "mp4" | "m4v" | "m4a" => "aac",
        "mov" => "aac",
        "ts" | "m2ts" | "mts" => "aac",
        "ogg" | "ogv" | "oga" => {
            if prefer_quality {
                "flac"
            } else {
                "vorbis"
            }
        }
        "flac" => "flac",
        "wav" => "pcm",
        "avi" => "mp3",
        _ => "opus",
    }
}

/// Codec preference score: higher is better quality/compression.
///
/// Useful for sorting candidates returned from the compatibility matrix.
pub fn codec_quality_score(codec: &str) -> u8 {
    match codec.to_ascii_lowercase().as_str() {
        "av1" => 100,
        "hevc" | "h265" => 90,
        "vp9" => 85,
        "avc" | "h264" => 75,
        "mpeg4" => 60,
        "vp8" => 55,
        "theora" => 50,
        "mpeg2" => 40,
        "flac" => 95,
        "opus" => 90,
        "vorbis" => 80,
        "aac" => 75,
        "mp3" => 65,
        "pcm" => 100, // Lossless, but large
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn webm_quality_gives_av1() {
        assert_eq!(select_codec_for_container("webm", true), "av1");
    }

    #[test]
    fn webm_speed_gives_vp9() {
        assert_eq!(select_codec_for_container("webm", false), "vp9");
    }

    #[test]
    fn mp4_quality_gives_hevc() {
        assert_eq!(select_codec_for_container("mp4", true), "hevc");
    }

    #[test]
    fn mp4_speed_gives_avc() {
        assert_eq!(select_codec_for_container("mp4", false), "avc");
    }

    #[test]
    fn mkv_quality_gives_av1() {
        assert_eq!(select_codec_for_container("mkv", true), "av1");
    }

    #[test]
    fn wav_gives_pcm() {
        assert_eq!(select_codec_for_container("wav", true), "pcm");
        assert_eq!(select_codec_for_container("wav", false), "pcm");
    }

    #[test]
    fn audio_webm_gives_opus() {
        assert_eq!(select_audio_codec_for_container("webm", true), "opus");
        assert_eq!(select_audio_codec_for_container("webm", false), "opus");
    }

    #[test]
    fn audio_mp4_gives_aac() {
        assert_eq!(select_audio_codec_for_container("mp4", false), "aac");
    }

    #[test]
    fn quality_scores_are_ordered() {
        assert!(codec_quality_score("av1") > codec_quality_score("vp9"));
        assert!(codec_quality_score("vp9") > codec_quality_score("vp8"));
        assert!(codec_quality_score("hevc") > codec_quality_score("avc"));
    }

    #[test]
    fn case_insensitive_container() {
        assert_eq!(select_codec_for_container("WEBM", true), "av1");
        assert_eq!(select_codec_for_container("MP4", false), "avc");
    }
}
