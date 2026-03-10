//! Codec name mapping between FFmpeg codec identifiers and OxiMedia codec IDs.
//!
//! FFmpeg uses names like `"libaom-av1"`, `"libvpx-vp9"`, `"libopus"`, etc.
//! OxiMedia uses its own internal codec identifiers. This module provides
//! lookup tables with semantic categories indicating whether a mapping is
//! a direct match or a patent-motivated substitution.

use std::collections::HashMap;

/// Classification of how a codec mapping was resolved.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CodecCategory {
    /// The FFmpeg codec maps directly to a supported OxiMedia codec.
    DirectMatch,
    /// The FFmpeg codec is patent-encumbered and was substituted with a free alternative.
    PatentSubstituted,
    /// Stream-copy passthrough — no transcoding performed.
    Copy,
}

/// A resolved codec entry, pairing an OxiMedia codec ID with its mapping category.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecEntry {
    /// The OxiMedia canonical codec name.
    pub oxi_name: &'static str,
    /// How this codec was resolved.
    pub category: CodecCategory,
}

impl CodecEntry {
    const fn direct(oxi_name: &'static str) -> Self {
        Self {
            oxi_name,
            category: CodecCategory::DirectMatch,
        }
    }

    const fn substituted(oxi_name: &'static str) -> Self {
        Self {
            oxi_name,
            category: CodecCategory::PatentSubstituted,
        }
    }

    const fn copy() -> Self {
        Self {
            oxi_name: "copy",
            category: CodecCategory::Copy,
        }
    }
}

/// Static table of (ffmpeg_name, CodecEntry) pairs.
/// All hyphens in keys are normalised to underscores before lookup.
static CODEC_TABLE: &[(&str, CodecEntry)] = &[
    // ── AV1 — direct matches ──────────────────────────────────────────────────
    ("libaom_av1", CodecEntry::direct("av1")),
    ("libsvtav1", CodecEntry::direct("av1")),
    ("librav1e", CodecEntry::direct("av1")),
    ("av1", CodecEntry::direct("av1")),
    ("av1_vaapi", CodecEntry::direct("av1")),
    ("av1_nvenc", CodecEntry::direct("av1")),
    ("av1_amf", CodecEntry::direct("av1")),
    // ── AV1 — patent substitutions (H.264 / H.265 / legacy) ──────────────────
    ("libx264", CodecEntry::substituted("av1")),
    ("h264", CodecEntry::substituted("av1")),
    ("libx265", CodecEntry::substituted("av1")),
    ("hevc", CodecEntry::substituted("av1")),
    ("h265", CodecEntry::substituted("av1")),
    ("prores", CodecEntry::substituted("av1")),
    ("prores_ks", CodecEntry::substituted("av1")),
    ("dnxhd", CodecEntry::substituted("av1")),
    ("dnxhr", CodecEntry::substituted("av1")),
    ("cineform", CodecEntry::substituted("av1")),
    ("cfhd", CodecEntry::substituted("av1")),
    ("wmv1", CodecEntry::substituted("av1")),
    ("wmv2", CodecEntry::substituted("av1")),
    ("wmv3", CodecEntry::substituted("av1")),
    ("vc1", CodecEntry::substituted("av1")),
    ("mpeg1video", CodecEntry::substituted("av1")),
    ("mpeg2video", CodecEntry::substituted("av1")),
    ("mpeg4", CodecEntry::substituted("av1")),
    ("msmpeg4v3", CodecEntry::substituted("av1")),
    ("msmpeg4v2", CodecEntry::substituted("av1")),
    ("msmpeg4v1", CodecEntry::substituted("av1")),
    ("rv10", CodecEntry::substituted("av1")),
    ("rv20", CodecEntry::substituted("av1")),
    ("rv30", CodecEntry::substituted("av1")),
    ("rv40", CodecEntry::substituted("av1")),
    ("h263", CodecEntry::substituted("av1")),
    ("h261", CodecEntry::substituted("av1")),
    ("mjpeg", CodecEntry::substituted("av1")),
    ("mjpegb", CodecEntry::substituted("av1")),
    // ── VP9 ───────────────────────────────────────────────────────────────────
    ("libvpx_vp9", CodecEntry::direct("vp9")),
    ("vp9", CodecEntry::direct("vp9")),
    ("vp9_vaapi", CodecEntry::direct("vp9")),
    ("vp9_cuvid", CodecEntry::direct("vp9")),
    // ── VP8 ───────────────────────────────────────────────────────────────────
    ("libvpx", CodecEntry::direct("vp8")),
    ("vp8", CodecEntry::direct("vp8")),
    // Theora maps to VP8 (closest open equivalent)
    ("libtheora", CodecEntry::direct("vp8")),
    ("theora", CodecEntry::direct("vp8")),
    // ── Opus — direct matches ─────────────────────────────────────────────────
    ("libopus", CodecEntry::direct("opus")),
    ("opus", CodecEntry::direct("opus")),
    // ── Opus — patent substitutions (AAC, MP3, AMR) ───────────────────────────
    ("aac", CodecEntry::substituted("opus")),
    ("libfdk_aac", CodecEntry::substituted("opus")),
    ("aac_at", CodecEntry::substituted("opus")),
    ("mp3", CodecEntry::substituted("opus")),
    ("libmp3lame", CodecEntry::substituted("opus")),
    ("mp3float", CodecEntry::substituted("opus")),
    ("mp3adu", CodecEntry::substituted("opus")),
    ("mp3on4", CodecEntry::substituted("opus")),
    ("mp2", CodecEntry::substituted("opus")),
    ("mp2float", CodecEntry::substituted("opus")),
    ("amr_nb", CodecEntry::substituted("opus")),
    ("amr_wb", CodecEntry::substituted("opus")),
    ("libopencore_amrnb", CodecEntry::substituted("opus")),
    ("libopencore_amrwb", CodecEntry::substituted("opus")),
    ("gsm", CodecEntry::substituted("opus")),
    ("gsm_ms", CodecEntry::substituted("opus")),
    ("wmavoice", CodecEntry::substituted("opus")),
    ("wma", CodecEntry::substituted("opus")),
    ("wmav2", CodecEntry::substituted("opus")),
    ("wmapro", CodecEntry::substituted("opus")),
    ("atrac3", CodecEntry::substituted("opus")),
    ("atrac3p", CodecEntry::substituted("opus")),
    ("eac3", CodecEntry::substituted("flac")),
    ("ac3", CodecEntry::substituted("flac")),
    ("ac3_fixed", CodecEntry::substituted("flac")),
    ("truehd", CodecEntry::substituted("flac")),
    ("mlp", CodecEntry::substituted("flac")),
    ("dts", CodecEntry::substituted("flac")),
    ("dca", CodecEntry::substituted("flac")),
    // ── Vorbis ────────────────────────────────────────────────────────────────
    ("libvorbis", CodecEntry::direct("vorbis")),
    ("vorbis", CodecEntry::direct("vorbis")),
    // ── FLAC — direct matches ─────────────────────────────────────────────────
    ("flac", CodecEntry::direct("flac")),
    // ALAC maps to FLAC (closest lossless equivalent)
    ("alac", CodecEntry::direct("flac")),
    // PCM variants map to FLAC (lossless path)
    ("pcm_s16le", CodecEntry::direct("flac")),
    ("pcm_s16be", CodecEntry::direct("flac")),
    ("pcm_s24le", CodecEntry::direct("flac")),
    ("pcm_s24be", CodecEntry::direct("flac")),
    ("pcm_s32le", CodecEntry::direct("flac")),
    ("pcm_s32be", CodecEntry::direct("flac")),
    ("pcm_f32le", CodecEntry::direct("flac")),
    ("pcm_f32be", CodecEntry::direct("flac")),
    ("pcm_f64le", CodecEntry::direct("flac")),
    ("pcm_f64be", CodecEntry::direct("flac")),
    ("pcm_u8", CodecEntry::direct("flac")),
    ("pcm_alaw", CodecEntry::direct("flac")),
    ("pcm_mulaw", CodecEntry::direct("flac")),
    // ── Stream copy ───────────────────────────────────────────────────────────
    ("copy", CodecEntry::copy()),
];

/// Mapping from FFmpeg codec names to OxiMedia codec entries.
pub struct CodecMap {
    table: HashMap<String, CodecEntry>,
}

impl Default for CodecMap {
    fn default() -> Self {
        Self::new()
    }
}

impl CodecMap {
    /// Build the codec map with all known FFmpeg → OxiMedia mappings.
    pub fn new() -> Self {
        let mut table: HashMap<String, CodecEntry> = HashMap::with_capacity(CODEC_TABLE.len());
        for (key, entry) in CODEC_TABLE {
            // Store with normalised hyphens-as-underscores key.
            table.insert(key.replace('-', "_"), entry.clone());
        }
        Self { table }
    }

    /// Look up an FFmpeg codec name and return the corresponding [`CodecEntry`].
    ///
    /// Returns `None` for completely unrecognised names.
    pub fn lookup(&self, ffmpeg_name: &str) -> Option<&CodecEntry> {
        let normalised = ffmpeg_name.to_lowercase().replace('-', "_");
        self.table.get(normalised.as_str())
    }

    /// Look up the OxiMedia codec name for an FFmpeg codec, or return the
    /// input unchanged for unrecognised codecs.
    pub fn oxi_name<'a>(&self, ffmpeg_name: &'a str) -> &'a str {
        self.lookup(ffmpeg_name)
            .map(|e| e.oxi_name)
            .unwrap_or(ffmpeg_name)
    }

    /// Return `true` if the codec name maps to a supported OxiMedia codec.
    pub fn is_supported(&self, ffmpeg_name: &str) -> bool {
        self.lookup(ffmpeg_name).is_some()
    }

    /// Return `true` if the codec maps to a `PatentSubstituted` entry.
    pub fn is_patent_substituted(&self, ffmpeg_name: &str) -> bool {
        matches!(
            self.lookup(ffmpeg_name),
            Some(CodecEntry {
                category: CodecCategory::PatentSubstituted,
                ..
            })
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_av1() {
        let map = CodecMap::new();
        let e = map.lookup("libaom-av1").expect("should exist");
        assert_eq!(e.oxi_name, "av1");
        assert_eq!(e.category, CodecCategory::DirectMatch);
    }

    #[test]
    fn test_patent_h264() {
        let map = CodecMap::new();
        let e = map.lookup("libx264").expect("should exist");
        assert_eq!(e.oxi_name, "av1");
        assert_eq!(e.category, CodecCategory::PatentSubstituted);
    }

    #[test]
    fn test_patent_aac() {
        let map = CodecMap::new();
        let e = map.lookup("aac").expect("should exist");
        assert_eq!(e.oxi_name, "opus");
        assert_eq!(e.category, CodecCategory::PatentSubstituted);
    }

    #[test]
    fn test_copy() {
        let map = CodecMap::new();
        let e = map.lookup("copy").expect("should exist");
        assert_eq!(e.category, CodecCategory::Copy);
    }

    #[test]
    fn test_hyphen_normalisation() {
        let map = CodecMap::new();
        // Both hyphenated and underscored forms should resolve.
        assert!(map.lookup("libvpx-vp9").is_some());
        assert!(map.lookup("libvpx_vp9").is_some());
    }

    #[test]
    fn test_unknown_codec() {
        let map = CodecMap::new();
        assert!(map.lookup("my_custom_codec").is_none());
    }

    #[test]
    fn test_flac_pcm_variants() {
        let map = CodecMap::new();
        for name in &["pcm_s16le", "pcm_s24le", "pcm_f32le", "pcm_u8"] {
            let e = map
                .lookup(name)
                .unwrap_or_else(|| panic!("{} missing", name));
            assert_eq!(e.oxi_name, "flac");
        }
    }

    #[test]
    fn test_ac3_flac_substitution() {
        let map = CodecMap::new();
        let e = map.lookup("ac3").expect("ac3 should map");
        assert_eq!(e.oxi_name, "flac");
        assert_eq!(e.category, CodecCategory::PatentSubstituted);
    }

    #[test]
    fn test_patent_video_codecs_become_av1() {
        let map = CodecMap::new();
        for name in &[
            "libx264", "libx265", "h264", "hevc", "prores", "dnxhd", "dnxhr",
        ] {
            let entry = map.lookup(name);
            assert!(entry.is_some(), "should know codec {}", name);
            let e = entry.expect("already checked");
            assert_eq!(e.oxi_name, "av1", "patent codec {} should map to av1", name);
            assert!(
                matches!(e.category, CodecCategory::PatentSubstituted),
                "{} should be PatentSubstituted",
                name
            );
        }
    }

    #[test]
    fn test_patent_audio_codecs_become_opus() {
        let map = CodecMap::new();
        for name in &["aac", "libfdk_aac", "mp3", "libmp3lame", "amr_nb", "amr_wb"] {
            let entry = map.lookup(name);
            assert!(entry.is_some(), "should know audio codec {}", name);
            let e = entry.expect("already checked");
            assert_eq!(
                e.oxi_name, "opus",
                "patent audio {} should map to opus",
                name
            );
        }
    }

    #[test]
    fn test_direct_video_codecs() {
        let map = CodecMap::new();
        assert_eq!(map.lookup("av1").expect("av1").oxi_name, "av1");
        assert_eq!(
            map.lookup("libaom-av1").expect("libaom-av1").oxi_name,
            "av1"
        );
        assert_eq!(map.lookup("libsvtav1").expect("libsvtav1").oxi_name, "av1");
        assert_eq!(map.lookup("vp9").expect("vp9").oxi_name, "vp9");
        assert_eq!(
            map.lookup("libvpx-vp9").expect("libvpx-vp9").oxi_name,
            "vp9"
        );
        assert_eq!(map.lookup("vp8").expect("vp8").oxi_name, "vp8");
        assert_eq!(map.lookup("libvpx").expect("libvpx").oxi_name, "vp8");
    }

    #[test]
    fn test_direct_audio_codecs() {
        let map = CodecMap::new();
        assert_eq!(map.lookup("opus").expect("opus").oxi_name, "opus");
        assert_eq!(map.lookup("libopus").expect("libopus").oxi_name, "opus");
        assert_eq!(map.lookup("vorbis").expect("vorbis").oxi_name, "vorbis");
        assert_eq!(
            map.lookup("libvorbis").expect("libvorbis").oxi_name,
            "vorbis"
        );
        assert_eq!(map.lookup("flac").expect("flac").oxi_name, "flac");
    }

    #[test]
    fn test_copy_passthrough_category() {
        let map = CodecMap::new();
        let entry = map.lookup("copy").expect("copy");
        assert_eq!(entry.oxi_name, "copy");
        assert!(matches!(entry.category, CodecCategory::Copy));
    }

    #[test]
    fn test_is_patent_substituted_helper() {
        let map = CodecMap::new();
        assert!(map.is_patent_substituted("libx264"), "libx264 is patent");
        assert!(map.is_patent_substituted("aac"), "aac is patent");
        assert!(!map.is_patent_substituted("av1"), "av1 is not patent");
        assert!(!map.is_patent_substituted("opus"), "opus is not patent");
        assert!(!map.is_patent_substituted("copy"), "copy is not patent");
    }

    #[test]
    fn test_is_supported_helper() {
        let map = CodecMap::new();
        assert!(map.is_supported("av1"));
        assert!(map.is_supported("libx264")); // known but substituted
        assert!(!map.is_supported("nonexistent_codec_xyz"));
    }

    #[test]
    fn test_oxi_name_passthrough_for_unknown() {
        let map = CodecMap::new();
        // Unknown codec names are returned as-is by oxi_name()
        let result = map.oxi_name("completely_unknown_codec");
        assert_eq!(result, "completely_unknown_codec");
    }

    #[test]
    fn test_case_insensitive_lookup() {
        let map = CodecMap::new();
        // Uppercase input should be normalised and found
        assert!(
            map.lookup("LIBX264").is_some(),
            "LIBX264 should be found case-insensitively"
        );
        assert!(
            map.lookup("AV1").is_some(),
            "AV1 should be found case-insensitively"
        );
        assert!(
            map.lookup("OPUS").is_some(),
            "OPUS should be found case-insensitively"
        );
    }

    #[test]
    fn test_theora_maps_to_vp8() {
        let map = CodecMap::new();
        let e = map.lookup("theora").expect("theora should be known");
        assert_eq!(e.oxi_name, "vp8");
        assert!(matches!(e.category, CodecCategory::DirectMatch));
    }

    #[test]
    fn test_alac_maps_to_flac() {
        let map = CodecMap::new();
        let e = map.lookup("alac").expect("alac should be known");
        assert_eq!(e.oxi_name, "flac");
    }

    #[test]
    fn test_dts_maps_to_flac() {
        let map = CodecMap::new();
        let e = map.lookup("dts").expect("dts should be known");
        assert_eq!(e.oxi_name, "flac");
        assert!(matches!(e.category, CodecCategory::PatentSubstituted));
    }

    #[test]
    fn test_all_pcm_variants_map_to_flac() {
        let map = CodecMap::new();
        for name in &[
            "pcm_s16le",
            "pcm_s16be",
            "pcm_s24le",
            "pcm_s24be",
            "pcm_s32le",
            "pcm_f32le",
            "pcm_f32be",
            "pcm_u8",
            "pcm_alaw",
            "pcm_mulaw",
        ] {
            let e = map
                .lookup(name)
                .unwrap_or_else(|| panic!("{} should be known", name));
            assert_eq!(e.oxi_name, "flac", "{} should map to flac", name);
        }
    }

    #[test]
    fn test_unknown_codec_returns_none() {
        let map = CodecMap::new();
        assert!(map.lookup("nonexistent_codec_xyz").is_none());
        assert!(map.lookup("h264_made_up").is_none());
    }

    #[test]
    fn test_wmv_variants_map_to_av1() {
        let map = CodecMap::new();
        for name in &["wmv1", "wmv2", "wmv3", "vc1"] {
            let e = map
                .lookup(name)
                .unwrap_or_else(|| panic!("{} should be known", name));
            assert_eq!(e.oxi_name, "av1", "{} should map to av1", name);
        }
    }

    #[test]
    fn test_wma_variants_map_to_opus() {
        let map = CodecMap::new();
        for name in &["wma", "wmav2", "wmapro"] {
            let e = map
                .lookup(name)
                .unwrap_or_else(|| panic!("{} should be known", name));
            assert_eq!(e.oxi_name, "opus", "{} should map to opus", name);
        }
    }
}
