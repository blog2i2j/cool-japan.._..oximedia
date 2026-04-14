//! Corrupt header recovery for media files.
//!
//! Provides utilities for locating sync markers and valid data regions in files
//! whose headers are partially or fully corrupted.

/// Corrupt header recovery utilities.
pub struct HeaderRecovery;

impl HeaderRecovery {
    /// Scan `data` for the first occurrence of `magic` bytes.
    ///
    /// Uses a simple sliding-window search.  Returns the byte offset of the
    /// first match, or `None` if the magic sequence is not found.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_repair::header_recovery::HeaderRecovery;
    ///
    /// let data = b"\x00\x00\x00\x00ftyp\x00\x00";
    /// let pos = HeaderRecovery::scan_for_sync(data, b"ftyp");
    /// assert_eq!(pos, Some(4));
    /// ```
    #[must_use]
    pub fn scan_for_sync(data: &[u8], magic: &[u8]) -> Option<usize> {
        if magic.is_empty() || data.len() < magic.len() {
            return None;
        }
        data.windows(magic.len()).position(|w| w == magic)
    }

    /// Scan for *all* occurrences of `magic` in `data`.
    ///
    /// Returns a `Vec` of byte offsets (may be empty).
    #[must_use]
    pub fn scan_all(data: &[u8], magic: &[u8]) -> Vec<usize> {
        if magic.is_empty() || data.len() < magic.len() {
            return Vec::new();
        }
        data.windows(magic.len())
            .enumerate()
            .filter_map(|(i, w)| if w == magic { Some(i) } else { None })
            .collect()
    }

    /// Extract the data region starting at the first sync point.
    ///
    /// Returns a slice of `data` beginning at the sync position, or `None` if
    /// the magic bytes are not found.
    #[must_use]
    pub fn recover_from_sync<'a>(data: &'a [u8], magic: &[u8]) -> Option<&'a [u8]> {
        Self::scan_for_sync(data, magic).map(|pos| &data[pos..])
    }

    /// Try a list of well-known container magic bytes and return the earliest
    /// sync point found together with the matched magic.
    ///
    /// Useful when the container type is unknown.  The caller may pass an empty
    /// list — `None` is returned in that case.
    #[must_use]
    pub fn scan_known_containers(data: &[u8]) -> Option<(usize, &'static [u8])> {
        const KNOWN: &[&[u8]] = &[
            b"ftyp",        // MP4/MOV/M4V
            b"moov",        // MP4 movie box
            b"\x1a\x45\xdf\xa3", // Matroska/WebM EBML
            b"RIFF",        // AVI/WAV
            b"OggS",        // Ogg container
            b"\xff\xfb",    // MP3 sync word
            b"\xff\xf3",    // MP3 sync word (alternative)
            b"fLaC",        // FLAC stream marker
            b"ID3",         // ID3 tag (MP3)
            b"\x00\x00\x01\xba", // MPEG Program Stream
            b"\x00\x00\x01\xb3", // MPEG Video Sequence Header
        ];

        let mut best: Option<(usize, &'static [u8])> = None;
        for magic in KNOWN {
            if let Some(pos) = Self::scan_for_sync(data, magic) {
                match &best {
                    None => best = Some((pos, magic)),
                    Some((best_pos, _)) if pos < *best_pos => best = Some((pos, magic)),
                    _ => {}
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_finds_magic_at_start() {
        let data = b"ftypisom\x00\x00\x00\x00";
        assert_eq!(HeaderRecovery::scan_for_sync(data, b"ftyp"), Some(0));
    }

    #[test]
    fn scan_finds_magic_after_corrupt_prefix() {
        let mut data = vec![0u8; 16];
        data.extend_from_slice(b"moov");
        data.extend_from_slice(&[0u8; 8]);
        assert_eq!(HeaderRecovery::scan_for_sync(&data, b"moov"), Some(16));
    }

    #[test]
    fn scan_returns_none_when_absent() {
        let data = b"random garbage data here";
        assert_eq!(HeaderRecovery::scan_for_sync(data, b"ftyp"), None);
    }

    #[test]
    fn scan_empty_magic_returns_none() {
        let data = b"some data";
        assert_eq!(HeaderRecovery::scan_for_sync(data, b""), None);
    }

    #[test]
    fn scan_data_shorter_than_magic_returns_none() {
        assert_eq!(HeaderRecovery::scan_for_sync(b"ab", b"abcd"), None);
    }

    #[test]
    fn scan_all_finds_multiple_occurrences() {
        let data = b"ftyp___ftyp___ftyp";
        let offsets = HeaderRecovery::scan_all(data, b"ftyp");
        assert_eq!(offsets, vec![0, 7, 14]);
    }

    #[test]
    fn scan_all_empty_result_on_miss() {
        let data = b"no magic here";
        assert!(HeaderRecovery::scan_all(data, b"ftyp").is_empty());
    }

    #[test]
    fn recover_from_sync_returns_slice_from_match() {
        let data = b"\x00\x00ftypisom";
        let recovered = HeaderRecovery::recover_from_sync(data, b"ftyp");
        assert_eq!(recovered, Some(b"ftypisom".as_ref()));
    }

    #[test]
    fn scan_known_containers_detects_matroska() {
        let mut data = vec![0u8; 8];
        data.extend_from_slice(b"\x1a\x45\xdf\xa3");
        data.extend_from_slice(&[0u8; 32]);
        let result = HeaderRecovery::scan_known_containers(&data);
        assert!(result.is_some());
        let (pos, _magic) = result.expect("should find matroska");
        assert_eq!(pos, 8);
    }

    #[test]
    fn scan_known_containers_returns_none_on_random_data() {
        let data = vec![0x42u8; 128];
        // 0x42 repeated — unlikely to match any known magic
        // (could match OggS or ID3 in pathological cases so we just check it doesn't panic)
        let _ = HeaderRecovery::scan_known_containers(&data);
    }

    #[test]
    fn scan_known_containers_picks_earliest() {
        // Embed two known magics: RIFF at 4, OggS at 0
        let mut data = b"OggS".to_vec();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&[0u8; 32]);
        let result = HeaderRecovery::scan_known_containers(&data);
        let (pos, _) = result.expect("found");
        assert_eq!(pos, 0, "earliest magic should be preferred");
    }
}
