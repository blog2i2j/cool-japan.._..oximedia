#![allow(dead_code)]
//! Path resolution and normalization for storage object keys.
//!
//! Handles joining, splitting, normalizing, and validating hierarchical
//! object key paths used across all storage backends.

use std::fmt;

/// Maximum allowed depth for resolved paths.
const MAX_DEPTH: usize = 64;

/// Maximum allowed length for a single path segment.
const MAX_SEGMENT_LEN: usize = 255;

/// Maximum total path length.
const MAX_PATH_LEN: usize = 1024;

/// A validated, normalized object key path.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResolvedPath {
    /// Normalized segments (no empty segments, no `.` or `..`).
    segments: Vec<String>,
}

impl ResolvedPath {
    /// Resolve a raw path string into a normalized `ResolvedPath`.
    ///
    /// Returns `None` if the path is invalid (empty after normalization,
    /// too deep, or contains forbidden characters).
    pub fn resolve(raw: &str) -> Option<Self> {
        if raw.is_empty() {
            return None;
        }
        let mut segments = Vec::new();
        for part in raw.split('/') {
            if part.is_empty() || part == "." {
                continue;
            }
            if part == ".." {
                segments.pop();
                continue;
            }
            if part.len() > MAX_SEGMENT_LEN {
                return None;
            }
            if part.contains('\0') {
                return None;
            }
            segments.push(part.to_string());
        }
        if segments.is_empty() {
            return None;
        }
        if segments.len() > MAX_DEPTH {
            return None;
        }
        let path = segments.join("/");
        if path.len() > MAX_PATH_LEN {
            return None;
        }
        Some(Self { segments })
    }

    /// Join another relative path onto this one.
    pub fn join(&self, relative: &str) -> Option<Self> {
        let combined = format!("{}/{}", self.as_str(), relative);
        Self::resolve(&combined)
    }

    /// Parent path (all segments except the last).
    pub fn parent(&self) -> Option<Self> {
        if self.segments.len() <= 1 {
            return None;
        }
        Some(Self {
            segments: self.segments[..self.segments.len() - 1].to_vec(),
        })
    }

    /// The final segment (file name or leaf directory).
    pub fn file_name(&self) -> &str {
        self.segments.last().map_or("", std::string::String::as_str)
    }

    /// The file extension, if any (without leading dot).
    pub fn extension(&self) -> Option<&str> {
        let name = self.file_name();
        let dot = name.rfind('.')?;
        if dot == 0 || dot == name.len() - 1 {
            return None;
        }
        Some(&name[dot + 1..])
    }

    /// Number of path segments.
    pub fn depth(&self) -> usize {
        self.segments.len()
    }

    /// All path segments.
    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    /// Render as a `/`-separated string.
    pub fn as_str(&self) -> String {
        self.segments.join("/")
    }

    /// Whether this path starts with the given prefix path.
    pub fn starts_with(&self, prefix: &ResolvedPath) -> bool {
        if prefix.segments.len() > self.segments.len() {
            return false;
        }
        self.segments[..prefix.segments.len()] == prefix.segments[..]
    }

    /// Strip a prefix, returning the remaining relative portion.
    pub fn strip_prefix(&self, prefix: &ResolvedPath) -> Option<Self> {
        if !self.starts_with(prefix) {
            return None;
        }
        let remaining = self.segments[prefix.segments.len()..].to_vec();
        if remaining.is_empty() {
            return None;
        }
        Some(Self {
            segments: remaining,
        })
    }
}

impl fmt::Display for ResolvedPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Utility to match paths against a glob-like pattern (supports `*` and `**`).
#[derive(Debug, Clone)]
pub struct PathMatcher {
    /// The pattern segments.
    pattern_segments: Vec<String>,
}

impl PathMatcher {
    /// Create a new matcher from a glob pattern (e.g. `"media/**/raw/*.mp4"`).
    pub fn new(pattern: &str) -> Self {
        let pattern_segments: Vec<String> = pattern
            .split('/')
            .filter(|s| !s.is_empty())
            .map(std::string::ToString::to_string)
            .collect();
        Self { pattern_segments }
    }

    /// Check whether a resolved path matches this pattern.
    pub fn matches(&self, path: &ResolvedPath) -> bool {
        Self::match_segments(&self.pattern_segments, &path.segments)
    }

    /// Recursive segment matching.
    fn match_segments(pattern: &[String], segments: &[String]) -> bool {
        if pattern.is_empty() {
            return segments.is_empty();
        }
        let pat = &pattern[0];
        if pat == "**" {
            // `**` can match zero or more segments
            for i in 0..=segments.len() {
                if Self::match_segments(&pattern[1..], &segments[i..]) {
                    return true;
                }
            }
            return false;
        }
        if segments.is_empty() {
            return false;
        }
        if Self::segment_matches(pat, &segments[0]) {
            Self::match_segments(&pattern[1..], &segments[1..])
        } else {
            false
        }
    }

    /// Match a single segment against a pattern segment (`*` = any).
    fn segment_matches(pattern: &str, segment: &str) -> bool {
        if pattern == "*" {
            return true;
        }
        // Simple prefix-star: "*.ext" or "prefix*"
        if let Some(suffix) = pattern.strip_prefix('*') {
            return segment.ends_with(suffix);
        }
        if let Some(prefix) = pattern.strip_suffix('*') {
            return segment.starts_with(prefix);
        }
        pattern == segment
    }

    /// Number of pattern segments.
    pub fn segment_count(&self) -> usize {
        self.pattern_segments.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ResolvedPath ───────────────────────────────────────────────────────

    #[test]
    fn test_resolve_simple() {
        let p = ResolvedPath::resolve("media/video/clip.mp4").expect("valid resolved path");
        assert_eq!(p.depth(), 3);
        assert_eq!(p.as_str(), "media/video/clip.mp4");
    }

    #[test]
    fn test_resolve_strips_dots() {
        let p =
            ResolvedPath::resolve("media/./video/../audio/track.ogg").expect("valid resolved path");
        assert_eq!(p.as_str(), "media/audio/track.ogg");
    }

    #[test]
    fn test_resolve_empty_returns_none() {
        assert!(ResolvedPath::resolve("").is_none());
    }

    #[test]
    fn test_resolve_only_dots_returns_none() {
        assert!(ResolvedPath::resolve("../../..").is_none());
    }

    #[test]
    fn test_resolve_null_byte_rejected() {
        assert!(ResolvedPath::resolve("bad\0name").is_none());
    }

    #[test]
    fn test_join() {
        let base = ResolvedPath::resolve("media/video").expect("valid resolved path");
        let joined = base.join("raw/clip.mp4").expect("join should succeed");
        assert_eq!(joined.as_str(), "media/video/raw/clip.mp4");
    }

    #[test]
    fn test_parent() {
        let p = ResolvedPath::resolve("a/b/c").expect("valid resolved path");
        let par = p.parent().expect("parent should exist");
        assert_eq!(par.as_str(), "a/b");
    }

    #[test]
    fn test_parent_root_none() {
        let p = ResolvedPath::resolve("single").expect("valid resolved path");
        assert!(p.parent().is_none());
    }

    #[test]
    fn test_file_name() {
        let p = ResolvedPath::resolve("dir/file.txt").expect("valid resolved path");
        assert_eq!(p.file_name(), "file.txt");
    }

    #[test]
    fn test_extension() {
        let p = ResolvedPath::resolve("dir/file.mp4").expect("valid resolved path");
        assert_eq!(p.extension(), Some("mp4"));
    }

    #[test]
    fn test_extension_none() {
        let p = ResolvedPath::resolve("dir/noext").expect("valid resolved path");
        assert!(p.extension().is_none());
    }

    #[test]
    fn test_starts_with() {
        let p = ResolvedPath::resolve("a/b/c/d").expect("valid resolved path");
        let prefix = ResolvedPath::resolve("a/b").expect("valid resolved path");
        assert!(p.starts_with(&prefix));
    }

    #[test]
    fn test_strip_prefix() {
        let p = ResolvedPath::resolve("media/video/clip.mp4").expect("valid resolved path");
        let prefix = ResolvedPath::resolve("media").expect("valid resolved path");
        let rest = p
            .strip_prefix(&prefix)
            .expect("strip prefix should succeed");
        assert_eq!(rest.as_str(), "video/clip.mp4");
    }

    #[test]
    fn test_display() {
        let p = ResolvedPath::resolve("a/b").expect("valid resolved path");
        assert_eq!(format!("{p}"), "a/b");
    }

    // ── PathMatcher ────────────────────────────────────────────────────────

    #[test]
    fn test_matcher_exact() {
        let m = PathMatcher::new("media/video");
        let p = ResolvedPath::resolve("media/video").expect("valid resolved path");
        assert!(m.matches(&p));
    }

    #[test]
    fn test_matcher_star_segment() {
        let m = PathMatcher::new("media/*/clip.mp4");
        let p = ResolvedPath::resolve("media/video/clip.mp4").expect("valid resolved path");
        assert!(m.matches(&p));
    }

    #[test]
    fn test_matcher_double_star() {
        let m = PathMatcher::new("media/**/clip.mp4");
        let p = ResolvedPath::resolve("media/a/b/c/clip.mp4").expect("valid resolved path");
        assert!(m.matches(&p));
    }

    #[test]
    fn test_matcher_double_star_zero_depth() {
        let m = PathMatcher::new("media/**/clip.mp4");
        let p = ResolvedPath::resolve("media/clip.mp4").expect("valid resolved path");
        assert!(m.matches(&p));
    }

    #[test]
    fn test_matcher_ext_star() {
        let m = PathMatcher::new("media/*.mp4");
        let p = ResolvedPath::resolve("media/clip.mp4").expect("valid resolved path");
        assert!(m.matches(&p));
        let p2 = ResolvedPath::resolve("media/clip.mkv").expect("valid resolved path");
        assert!(!m.matches(&p2));
    }

    #[test]
    fn test_matcher_no_match() {
        let m = PathMatcher::new("archive/old");
        let p = ResolvedPath::resolve("media/video").expect("valid resolved path");
        assert!(!m.matches(&p));
    }

    #[test]
    fn test_matcher_segment_count() {
        let m = PathMatcher::new("a/b/c");
        assert_eq!(m.segment_count(), 3);
    }
}
