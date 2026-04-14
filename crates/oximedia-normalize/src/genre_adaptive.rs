//! Genre-adaptive loudness normalisation targets.
//!
//! Different content genres have different loudness expectations:
//! - **Music** streams are normalised to −14 LUFS (Spotify/YouTube delivery).
//! - **Speech** content is normalised to −16 LUFS (podcast/audiobook delivery).
//! - **Mixed** content (music + voice) uses −15 LUFS as a reasonable midpoint.
//!
//! [`GenreNormalizer`] encapsulates the genre → target-LUFS mapping and can be
//! used as a thin adaptor layer on top of the main [`crate::Normalizer`].
//!
//! # Example
//!
//! ```
//! use oximedia_normalize::genre_adaptive::GenreNormalizer;
//!
//! let n = GenreNormalizer::new("music");
//! assert!((n.target_lufs() - (-14.0)).abs() < 0.01);
//!
//! let n2 = GenreNormalizer::new("speech");
//! assert!((n2.target_lufs() - (-16.0)).abs() < 0.01);
//!
//! let n3 = GenreNormalizer::new("mixed");
//! assert!((n3.target_lufs() - (-15.0)).abs() < 0.01);
//!
//! // Unknown genres fall back to mixed
//! let n4 = GenreNormalizer::new("unknown_genre");
//! assert!((n4.target_lufs() - (-15.0)).abs() < 0.01);
//! ```

/// Genre label used for target-LUFS lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Genre {
    /// Music content (−14 LUFS).
    Music,
    /// Speech / podcast / audiobook content (−16 LUFS).
    Speech,
    /// Mixed music and speech content (−15 LUFS).
    Mixed,
}

impl Genre {
    /// Parse a human-readable genre string (case-insensitive) into a [`Genre`].
    ///
    /// The following strings map to [`Genre::Music`]: `"music"`.
    /// The following strings map to [`Genre::Speech`]: `"speech"`, `"podcast"`,
    /// `"audiobook"`, `"voice"`.
    /// Everything else maps to [`Genre::Mixed`].
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "music" => Self::Music,
            "speech" | "podcast" | "audiobook" | "voice" => Self::Speech,
            _ => Self::Mixed,
        }
    }

    /// Return the recommended normalisation target for this genre in LUFS.
    ///
    /// | Genre  | Target   | Rationale                                    |
    /// |--------|----------|----------------------------------------------|
    /// | Music  | −14 LUFS | Spotify / YouTube streaming standard         |
    /// | Speech | −16 LUFS | Apple Podcasts / Spotify Podcasts standard   |
    /// | Mixed  | −15 LUFS | Midpoint between music and speech targets    |
    #[must_use]
    pub fn target_lufs(&self) -> f32 {
        match self {
            Self::Music => -14.0,
            Self::Speech => -16.0,
            Self::Mixed => -15.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GenreNormalizer
// ─────────────────────────────────────────────────────────────────────────────

/// Genre-aware loudness normalisation helper.
///
/// Stores a [`Genre`] classification and exposes the corresponding LUFS target.
/// This is intentionally a lightweight value type — the actual audio processing
/// is delegated to the main [`crate::Normalizer`] pipeline; this struct only
/// holds the genre metadata and derived target.
#[derive(Debug, Clone)]
pub struct GenreNormalizer {
    /// Classified genre.
    genre: Genre,
}

impl GenreNormalizer {
    /// Create a new normaliser for the given genre string.
    ///
    /// The `genre` parameter is parsed via [`Genre::from_str`] and is therefore
    /// case-insensitive.  Unknown genres default to [`Genre::Mixed`].
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_normalize::genre_adaptive::GenreNormalizer;
    ///
    /// let n = GenreNormalizer::new("Music");
    /// assert_eq!(n.genre_label(), "music");
    /// ```
    #[must_use]
    pub fn new(genre: &str) -> Self {
        Self {
            genre: Genre::from_str(genre),
        }
    }

    /// Return the recommended LUFS target for this genre.
    ///
    /// | Genre   | Return value |
    /// |---------|-------------|
    /// | Music   | −14.0       |
    /// | Speech  | −16.0       |
    /// | Mixed   | −15.0       |
    #[must_use]
    pub fn target_lufs(&self) -> f32 {
        self.genre.target_lufs()
    }

    /// Return a lowercase string label for the detected genre.
    #[must_use]
    pub fn genre_label(&self) -> &str {
        match self.genre {
            Genre::Music => "music",
            Genre::Speech => "speech",
            Genre::Mixed => "mixed",
        }
    }

    /// Return the classified [`Genre`] enum value.
    #[must_use]
    pub fn genre(&self) -> &Genre {
        &self.genre
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_music_target() {
        let n = GenreNormalizer::new("music");
        assert!((n.target_lufs() - (-14.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_speech_target() {
        let n = GenreNormalizer::new("speech");
        assert!((n.target_lufs() - (-16.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mixed_target() {
        let n = GenreNormalizer::new("mixed");
        assert!((n.target_lufs() - (-15.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_podcast_maps_to_speech() {
        let n = GenreNormalizer::new("podcast");
        assert_eq!(n.genre(), &Genre::Speech);
        assert!((n.target_lufs() - (-16.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_case_insensitive() {
        let n = GenreNormalizer::new("MUSIC");
        assert_eq!(n.genre(), &Genre::Music);
    }

    #[test]
    fn test_unknown_genre_falls_back_to_mixed() {
        let n = GenreNormalizer::new("jingle");
        assert_eq!(n.genre(), &Genre::Mixed);
        assert!((n.target_lufs() - (-15.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_genre_label() {
        assert_eq!(GenreNormalizer::new("music").genre_label(), "music");
        assert_eq!(GenreNormalizer::new("speech").genre_label(), "speech");
        assert_eq!(GenreNormalizer::new("unknown").genre_label(), "mixed");
    }
}
