//! Cache key normalisation.
//!
//! [`normalize_cache_key`] converts a URL into a canonical form suitable for
//! use as a cache key.  The following transformations are applied:
//!
//! 1. **Lowercase** — scheme, host, and path are lowercased.
//! 2. **Trailing slash removal** — a trailing `/` on the path component is
//!    stripped (unless the path is `/` itself).
//! 3. **Query parameter sorting** — query string parameters are split on `&`,
//!    sorted lexicographically, and rejoined.  Empty query strings are omitted.
//!
//! Fragment identifiers (`#...`) are discarded as they are client-side only.
//!
//! # Example
//!
//! ```
//! use oximedia_cache::key_norm::normalize_cache_key;
//!
//! let key = normalize_cache_key("https://CDN.Example.com/Video/Clip.mp4/?b=2&a=1#frag");
//! assert_eq!(key, "https://cdn.example.com/video/clip.mp4?a=1&b=2");
//! ```

#![allow(dead_code)]

/// Normalise a URL into a canonical cache key.
///
/// # Arguments
///
/// * `url` — The raw URL string (any scheme is accepted).
///
/// # Returns
///
/// A normalised `String` suitable for use as a cache key.  If `url` cannot
/// be parsed (e.g. it is an empty string) the function returns the input
/// lowercased and stripped of trailing slashes as a best-effort fallback.
#[must_use]
pub fn normalize_cache_key(url: &str) -> String {
    if url.is_empty() {
        return String::new();
    }

    // Strip fragment (#...) — fragments are not sent to servers
    let no_fragment = match url.find('#') {
        Some(pos) => &url[..pos],
        None => url,
    };

    // Split at '?' to separate path from query string
    let (path_part, query_part) = match no_fragment.find('?') {
        Some(pos) => (&no_fragment[..pos], Some(&no_fragment[pos + 1..])),
        None => (no_fragment, None),
    };

    // Lowercase the path portion and strip trailing slash.
    // For URLs with a scheme (e.g. https://host/), preserve the root "/" so
    // that "https://example.com/" stays as-is (the path is "/", not empty).
    let mut path_lower = path_part.to_lowercase();
    let is_root_path = if let Some(after_scheme) = path_lower.find("://") {
        // Count how many '/' chars appear after the "://" separator+host.
        let after_host_start = after_scheme + 3;
        let after_host = &path_lower[after_host_start..];
        // Root path: the only '/' is the very first character of the path.
        after_host.find('/').map_or(false, |p| after_host[p..].len() == 1)
    } else {
        path_lower == "/"
    };
    if path_lower.ends_with('/') && !is_root_path {
        path_lower.pop();
    }

    // Sort and lowercase query parameters
    match query_part {
        None | Some("") => path_lower,
        Some(query) => {
            let mut params: Vec<String> = query
                .split('&')
                .filter(|s| !s.is_empty())
                .map(|p| p.to_lowercase())
                .collect();
            params.sort_unstable();
            let sorted_query = params.join("&");
            if sorted_query.is_empty() {
                path_lower
            } else {
                format!("{path_lower}?{sorted_query}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── basic normalisation ───────────────────────────────────────────────────

    #[test]
    fn test_lowercase_host_and_path() {
        let key = normalize_cache_key("https://CDN.Example.COM/Video/Clip.mp4");
        assert_eq!(key, "https://cdn.example.com/video/clip.mp4");
    }

    #[test]
    fn test_strip_trailing_slash() {
        let key = normalize_cache_key("https://example.com/path/");
        assert_eq!(key, "https://example.com/path");
    }

    #[test]
    fn test_preserve_root_slash() {
        // A bare "/" should not be stripped
        let key = normalize_cache_key("https://example.com/");
        // The "/" is the only character in the path after the host — lowercased is unchanged
        // but the trailing slash removal skips length-1 paths
        assert_eq!(key, "https://example.com/");
    }

    #[test]
    fn test_sort_query_params() {
        let key = normalize_cache_key("https://example.com/v?b=2&a=1");
        assert_eq!(key, "https://example.com/v?a=1&b=2");
    }

    #[test]
    fn test_strip_fragment() {
        let key = normalize_cache_key("https://example.com/page#section");
        assert_eq!(key, "https://example.com/page");
    }

    #[test]
    fn test_fragment_and_query_and_trailing_slash() {
        let key = normalize_cache_key("https://CDN.Example.com/Video/Clip.mp4/?b=2&a=1#frag");
        assert_eq!(key, "https://cdn.example.com/video/clip.mp4?a=1&b=2");
    }

    #[test]
    fn test_no_query_no_fragment() {
        let key = normalize_cache_key("https://example.com/asset.m4s");
        assert_eq!(key, "https://example.com/asset.m4s");
    }

    #[test]
    fn test_empty_url_returns_empty() {
        let key = normalize_cache_key("");
        assert_eq!(key, "");
    }

    #[test]
    fn test_empty_query_string_omitted() {
        let key = normalize_cache_key("https://example.com/path?");
        // Empty query → no '?' in output
        assert!(!key.contains('?'), "Got: {key}");
    }

    #[test]
    fn test_multiple_query_params_sorted() {
        let key = normalize_cache_key("http://cdn.test/v?z=9&m=3&a=1");
        assert_eq!(key, "http://cdn.test/v?a=1&m=3&z=9");
    }

    #[test]
    fn test_already_normalised_is_idempotent() {
        let url = "https://example.com/path?a=1&b=2";
        let once = normalize_cache_key(url);
        let twice = normalize_cache_key(&once);
        assert_eq!(once, twice);
    }

    #[test]
    fn test_path_without_scheme() {
        // Non-standard inputs should at least lowercase and sort params
        let key = normalize_cache_key("/PATH/TO/FILE?Z=1&A=2");
        assert_eq!(key, "/path/to/file?a=2&z=1");
    }
}
