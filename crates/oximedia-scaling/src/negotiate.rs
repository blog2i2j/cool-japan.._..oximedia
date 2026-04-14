//! Resolution negotiation between client capabilities and server offerings.
//!
//! When a client connects to a media server it typically advertises a set of
//! supported resolutions.  The server likewise has a set of resolutions it can
//! deliver.  `negotiate_resolution` selects the best common resolution
//! using the following heuristic:
//!
//! 1. Find the intersection of client and server capability sets.
//! 2. From that intersection, return the resolution with the **greatest
//!    pixel count** (`width × height`) — i.e., the highest quality both
//!    parties support.
//!
//! Returns `None` when no common resolution exists.
//!
//! # Example
//!
//! ```
//! use oximedia_scaling::negotiate::negotiate_resolution;
//!
//! let client = [(1920, 1080), (1280, 720), (640, 480)];
//! let server = [(3840, 2160), (1920, 1080), (1280, 720)];
//!
//! let best = negotiate_resolution(&client, &server);
//! assert_eq!(best, Some((1920, 1080)));
//! ```

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Select the best mutually-supported resolution.
///
/// # Parameters
///
/// * `client_caps` — slice of `(width, height)` pairs the client can handle.
/// * `server_caps` — slice of `(width, height)` pairs the server can offer.
///
/// # Returns
///
/// `Some((width, height))` — the common resolution with the largest pixel
/// count, or `None` if there is no common resolution.
pub fn negotiate_resolution(
    client_caps: &[(u32, u32)],
    server_caps: &[(u32, u32)],
) -> Option<(u32, u32)> {
    // Build an intersection: a resolution is common iff both sides list it.
    // We iterate client caps and check server membership for O(n*m); for
    // realistic capability list lengths (< 20 entries) this is negligible.
    let mut common: Vec<(u32, u32)> = client_caps
        .iter()
        .copied()
        .filter(|cap| server_caps.contains(cap))
        .collect();

    // Remove duplicates that may appear when either list has repeated entries.
    common.sort_unstable();
    common.dedup();

    // Select the resolution with the maximum pixel count.
    common.into_iter().max_by_key(|&(w, h)| w as u64 * h as u64)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_negotiation() {
        let client = [(1920, 1080), (1280, 720), (640, 480)];
        let server = [(3840, 2160), (1920, 1080), (1280, 720)];
        assert_eq!(negotiate_resolution(&client, &server), Some((1920, 1080)));
    }

    #[test]
    fn test_no_common_resolution() {
        let client = [(640, 480), (320, 240)];
        let server = [(3840, 2160), (1920, 1080)];
        assert_eq!(negotiate_resolution(&client, &server), None);
    }

    #[test]
    fn test_single_common() {
        let client = [(1280, 720)];
        let server = [(1280, 720), (1920, 1080)];
        assert_eq!(negotiate_resolution(&client, &server), Some((1280, 720)));
    }

    #[test]
    fn test_empty_client() {
        let server = [(1920, 1080)];
        assert_eq!(negotiate_resolution(&[], &server), None);
    }

    #[test]
    fn test_empty_server() {
        let client = [(1920, 1080)];
        assert_eq!(negotiate_resolution(&client, &[]), None);
    }

    #[test]
    fn test_both_empty() {
        assert_eq!(negotiate_resolution(&[], &[]), None);
    }

    #[test]
    fn test_selects_highest_resolution() {
        // All three are common; expect 4K (largest pixel count).
        let client = [(3840, 2160), (1920, 1080), (1280, 720)];
        let server = [(3840, 2160), (1920, 1080), (1280, 720)];
        assert_eq!(negotiate_resolution(&client, &server), Some((3840, 2160)));
    }

    #[test]
    fn test_duplicate_entries_handled() {
        let client = [(1280, 720), (1280, 720), (1920, 1080)];
        let server = [(1280, 720), (1920, 1080), (1920, 1080)];
        assert_eq!(negotiate_resolution(&client, &server), Some((1920, 1080)));
    }

    #[test]
    fn test_portrait_resolutions() {
        let client = [(720, 1280), (480, 854)];
        let server = [(720, 1280), (1080, 1920)];
        assert_eq!(negotiate_resolution(&client, &server), Some((720, 1280)));
    }

    #[test]
    fn test_equal_pixel_count_returns_one() {
        // Both 1024×768 and 768×1024 have the same pixel count but are distinct.
        let client = [(1024, 768)];
        let server = [(1024, 768), (768, 1024)];
        // Only (1024, 768) is common.
        assert_eq!(negotiate_resolution(&client, &server), Some((1024, 768)));
    }
}
