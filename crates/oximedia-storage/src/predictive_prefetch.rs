//! Predictive prefetching based on access pattern analysis.
//!
//! Monitors object access patterns and predicts which objects are likely to be
//! requested next.  Two access pattern types are detected:
//!
//! - **Sequential**: objects whose keys share a common prefix and differ only in
//!   a monotonically increasing numeric suffix (e.g. `chunk-001.ts`,
//!   `chunk-002.ts`, ...).
//!
//! - **Random**: no discernible pattern — prefetching is suppressed.
//!
//! A `PrefetchAdvisor` maintains a sliding window of recent accesses per key
//! prefix, detects the dominant pattern, and returns a list of predicted keys
//! that should be prefetched into the cache.
//!
//! # Usage
//!
//! ```rust
//! use oximedia_storage::predictive_prefetch::{PrefetchAdvisor, PrefetchConfig};
//!
//! let config = PrefetchConfig::default();
//! let mut advisor = PrefetchAdvisor::new(config);
//!
//! advisor.record_access("video/segment-001.ts");
//! advisor.record_access("video/segment-002.ts");
//! advisor.record_access("video/segment-003.ts");
//!
//! let predictions = advisor.predict("video/segment-003.ts");
//! // predictions may contain "video/segment-004.ts", "video/segment-005.ts"
//! ```

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for the prefetch advisor.
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Number of recent accesses to keep per prefix group.
    pub window_size: usize,
    /// Minimum number of sequential accesses before predicting.
    pub min_sequential_run: usize,
    /// How many objects ahead to predict for sequential patterns.
    pub lookahead: usize,
    /// Minimum ratio of sequential accesses in the window to classify as sequential.
    pub sequential_threshold: f64,
    /// Maximum number of distinct prefix groups to track.
    pub max_tracked_prefixes: usize,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            window_size: 16,
            min_sequential_run: 3,
            lookahead: 3,
            sequential_threshold: 0.6,
            max_tracked_prefixes: 256,
        }
    }
}

// ─── AccessPattern ──────────────────────────────────────────────────────────

/// Detected access pattern for a prefix group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Keys are accessed in monotonically increasing order.
    Sequential,
    /// No discernible ordering.
    Random,
    /// Not enough data to determine.
    Unknown,
}

// ─── ParsedKey ──────────────────────────────────────────────────────────────

/// A parsed key split into prefix, numeric suffix, and extension.
///
/// For example, `media/segment-042.ts` becomes:
/// - prefix: `media/segment-`
/// - number: 42
/// - extension: `.ts`
#[derive(Debug, Clone)]
struct ParsedKey {
    prefix: String,
    number: u64,
    extension: String,
}

/// Attempt to parse a key into prefix + trailing number + extension.
///
/// Returns `None` if no trailing number is found before the extension.
fn parse_key(key: &str) -> Option<ParsedKey> {
    // Split off the extension (last `.` segment).
    let (stem, extension) = match key.rfind('.') {
        Some(dot_pos) => (&key[..dot_pos], &key[dot_pos..]),
        None => (key, ""),
    };

    // Find the trailing numeric segment in the stem.
    let num_start = stem
        .char_indices()
        .rev()
        .take_while(|(_, c)| c.is_ascii_digit())
        .last()
        .map(|(i, _)| i);

    let num_start = match num_start {
        Some(i) if i < stem.len() => i,
        _ => return None,
    };

    let prefix = &stem[..num_start];
    let number_str = &stem[num_start..];

    if number_str.is_empty() {
        return None;
    }

    let number = number_str.parse::<u64>().ok()?;

    Some(ParsedKey {
        prefix: prefix.to_string(),
        number,
        extension: extension.to_string(),
    })
}

/// Reconstruct a key from prefix + number + extension, zero-padded to `width`.
fn reconstruct_key(prefix: &str, number: u64, extension: &str, width: usize) -> String {
    format!("{prefix}{number:0>width$}{extension}", width = width)
}

// ─── PrefixTracker ──────────────────────────────────────────────────────────

/// Tracks recent numeric values for a given prefix group.
#[derive(Debug)]
struct PrefixTracker {
    /// Recent numeric suffixes in access order.
    recent: VecDeque<u64>,
    /// The extension observed for this prefix group.
    extension: String,
    /// Width of the zero-padded number field (e.g. 3 for "001").
    number_width: usize,
    /// Maximum window size.
    max_window: usize,
}

impl PrefixTracker {
    fn new(extension: String, number_width: usize, max_window: usize) -> Self {
        Self {
            recent: VecDeque::with_capacity(max_window),
            extension,
            number_width,
            max_window,
        }
    }

    fn record(&mut self, number: u64) {
        if self.recent.len() >= self.max_window {
            self.recent.pop_front();
        }
        self.recent.push_back(number);
    }

    /// Determine the fraction of consecutive pairs that are strictly increasing.
    fn sequential_fraction(&self) -> f64 {
        if self.recent.len() < 2 {
            return 0.0;
        }
        let pairs = self.recent.len() - 1;
        let sequential_count = self
            .recent
            .iter()
            .zip(self.recent.iter().skip(1))
            .filter(|(a, b)| **b > **a)
            .count();
        sequential_count as f64 / pairs as f64
    }

    /// Detect the current access pattern.
    fn detect_pattern(&self, config: &PrefetchConfig) -> AccessPattern {
        if self.recent.len() < config.min_sequential_run {
            return AccessPattern::Unknown;
        }
        if self.sequential_fraction() >= config.sequential_threshold {
            AccessPattern::Sequential
        } else {
            AccessPattern::Random
        }
    }

    /// Return the highest numeric value seen in the window.
    fn max_number(&self) -> Option<u64> {
        self.recent.iter().copied().max()
    }

    /// Return the most recent number.
    fn last_number(&self) -> Option<u64> {
        self.recent.back().copied()
    }

    /// Compute the most common step size between consecutive sequential values.
    fn dominant_step(&self) -> u64 {
        if self.recent.len() < 2 {
            return 1;
        }
        let mut step_counts: HashMap<u64, usize> = HashMap::new();
        for pair in self.recent.iter().zip(self.recent.iter().skip(1)) {
            if *pair.1 > *pair.0 {
                let step = *pair.1 - *pair.0;
                *step_counts.entry(step).or_insert(0) += 1;
            }
        }
        step_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map_or(1, |(step, _)| step)
    }
}

// ─── PrefetchAdvisor ────────────────────────────────────────────────────────

/// Predicts which objects should be prefetched based on observed access patterns.
pub struct PrefetchAdvisor {
    config: PrefetchConfig,
    /// Trackers keyed by (prefix, extension).
    trackers: HashMap<String, PrefixTracker>,
    /// Order of tracker insertion for LRU eviction.
    tracker_order: VecDeque<String>,
}

impl PrefetchAdvisor {
    /// Create a new advisor with the given configuration.
    pub fn new(config: PrefetchConfig) -> Self {
        Self {
            config,
            trackers: HashMap::new(),
            tracker_order: VecDeque::new(),
        }
    }

    /// Record an object access.
    pub fn record_access(&mut self, key: &str) {
        let parsed = match parse_key(key) {
            Some(p) => p,
            None => return, // key has no numeric suffix — cannot track
        };

        let tracker_key = format!("{}{}", parsed.prefix, parsed.extension);

        // Evict oldest tracker if at capacity and this is a new key
        if !self.trackers.contains_key(&tracker_key)
            && self.trackers.len() >= self.config.max_tracked_prefixes
        {
            if let Some(oldest) = self.tracker_order.pop_front() {
                self.trackers.remove(&oldest);
            }
        }

        let number_width = {
            // Compute the width from the original key's numeric segment
            let stem = key.rfind('.').map_or(key, |i| &key[..i]);
            let num_chars = stem
                .chars()
                .rev()
                .take_while(|c| c.is_ascii_digit())
                .count();
            num_chars.max(1)
        };

        let tracker = self.trackers.entry(tracker_key.clone()).or_insert_with(|| {
            self.tracker_order.push_back(tracker_key);
            PrefixTracker::new(
                parsed.extension.clone(),
                number_width,
                self.config.window_size,
            )
        });

        tracker.record(parsed.number);
    }

    /// Predict which keys should be prefetched given the most recent access.
    ///
    /// Returns an empty vec if the access pattern is random or unknown, or if
    /// the key cannot be parsed.
    pub fn predict(&self, key: &str) -> Vec<String> {
        let parsed = match parse_key(key) {
            Some(p) => p,
            None => return Vec::new(),
        };

        let tracker_key = format!("{}{}", parsed.prefix, parsed.extension);
        let tracker = match self.trackers.get(&tracker_key) {
            Some(t) => t,
            None => return Vec::new(),
        };

        if tracker.detect_pattern(&self.config) != AccessPattern::Sequential {
            return Vec::new();
        }

        let step = tracker.dominant_step();
        let last = tracker.last_number().unwrap_or(parsed.number);

        (1..=self.config.lookahead)
            .map(|i| {
                let next_num = last + step * i as u64;
                reconstruct_key(
                    &parsed.prefix,
                    next_num,
                    &parsed.extension,
                    tracker.number_width,
                )
            })
            .collect()
    }

    /// Get the detected access pattern for a given key's prefix group.
    pub fn pattern_for(&self, key: &str) -> AccessPattern {
        let parsed = match parse_key(key) {
            Some(p) => p,
            None => return AccessPattern::Unknown,
        };
        let tracker_key = format!("{}{}", parsed.prefix, parsed.extension);
        self.trackers
            .get(&tracker_key)
            .map_or(AccessPattern::Unknown, |t| t.detect_pattern(&self.config))
    }

    /// Number of currently tracked prefix groups.
    pub fn tracked_prefixes(&self) -> usize {
        self.trackers.len()
    }

    /// Clear all tracking state.
    pub fn clear(&mut self) {
        self.trackers.clear();
        self.tracker_order.clear();
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_key_basic() {
        let p = parse_key("video/segment-003.ts").expect("should parse");
        assert_eq!(p.prefix, "video/segment-");
        assert_eq!(p.number, 3);
        assert_eq!(p.extension, ".ts");
    }

    #[test]
    fn test_parse_key_no_extension() {
        let p = parse_key("chunk42").expect("should parse");
        assert_eq!(p.prefix, "chunk");
        assert_eq!(p.number, 42);
        assert_eq!(p.extension, "");
    }

    #[test]
    fn test_parse_key_no_number() {
        assert!(parse_key("readme.txt").is_none());
        assert!(parse_key("nodigits").is_none());
    }

    #[test]
    fn test_reconstruct_key() {
        let key = reconstruct_key("seg-", 7, ".ts", 3);
        assert_eq!(key, "seg-007.ts");
    }

    #[test]
    fn test_sequential_detection() {
        let config = PrefetchConfig {
            min_sequential_run: 3,
            sequential_threshold: 0.6,
            ..PrefetchConfig::default()
        };
        let mut advisor = PrefetchAdvisor::new(config);

        advisor.record_access("v/s-001.ts");
        advisor.record_access("v/s-002.ts");
        advisor.record_access("v/s-003.ts");

        assert_eq!(advisor.pattern_for("v/s-003.ts"), AccessPattern::Sequential);
    }

    #[test]
    fn test_random_detection() {
        let config = PrefetchConfig {
            min_sequential_run: 3,
            sequential_threshold: 0.6,
            window_size: 8,
            ..PrefetchConfig::default()
        };
        let mut advisor = PrefetchAdvisor::new(config);

        // Random access pattern
        advisor.record_access("v/s-010.ts");
        advisor.record_access("v/s-003.ts");
        advisor.record_access("v/s-007.ts");
        advisor.record_access("v/s-001.ts");

        assert_eq!(advisor.pattern_for("v/s-001.ts"), AccessPattern::Random);
    }

    #[test]
    fn test_predict_sequential() {
        let config = PrefetchConfig {
            min_sequential_run: 3,
            lookahead: 2,
            ..PrefetchConfig::default()
        };
        let mut advisor = PrefetchAdvisor::new(config);

        advisor.record_access("data/chunk-001.bin");
        advisor.record_access("data/chunk-002.bin");
        advisor.record_access("data/chunk-003.bin");

        let predictions = advisor.predict("data/chunk-003.bin");
        assert_eq!(predictions.len(), 2);
        assert_eq!(predictions[0], "data/chunk-004.bin");
        assert_eq!(predictions[1], "data/chunk-005.bin");
    }

    #[test]
    fn test_predict_empty_for_random() {
        let config = PrefetchConfig {
            min_sequential_run: 3,
            ..PrefetchConfig::default()
        };
        let mut advisor = PrefetchAdvisor::new(config);

        advisor.record_access("d/c-100.bin");
        advisor.record_access("d/c-050.bin");
        advisor.record_access("d/c-075.bin");
        advisor.record_access("d/c-025.bin");

        let predictions = advisor.predict("d/c-025.bin");
        assert!(predictions.is_empty());
    }

    #[test]
    fn test_predict_empty_for_unknown_key() {
        let advisor = PrefetchAdvisor::new(PrefetchConfig::default());
        let predictions = advisor.predict("no-number.txt");
        assert!(predictions.is_empty());
    }

    #[test]
    fn test_step_detection() {
        let config = PrefetchConfig {
            min_sequential_run: 3,
            lookahead: 2,
            ..PrefetchConfig::default()
        };
        let mut advisor = PrefetchAdvisor::new(config);

        // Step of 2
        advisor.record_access("frame-002.jpg");
        advisor.record_access("frame-004.jpg");
        advisor.record_access("frame-006.jpg");

        let predictions = advisor.predict("frame-006.jpg");
        assert_eq!(predictions.len(), 2);
        assert_eq!(predictions[0], "frame-008.jpg");
        assert_eq!(predictions[1], "frame-010.jpg");
    }

    #[test]
    fn test_tracker_eviction() {
        let config = PrefetchConfig {
            max_tracked_prefixes: 2,
            min_sequential_run: 2,
            ..PrefetchConfig::default()
        };
        let mut advisor = PrefetchAdvisor::new(config);

        advisor.record_access("a/x-001.ts");
        advisor.record_access("b/x-001.ts");
        // This should evict "a/x-" tracker
        advisor.record_access("c/x-001.ts");

        assert_eq!(advisor.tracked_prefixes(), 2);
        // "a" tracker should have been evicted
        assert_eq!(advisor.pattern_for("a/x-001.ts"), AccessPattern::Unknown);
    }

    #[test]
    fn test_clear() {
        let mut advisor = PrefetchAdvisor::new(PrefetchConfig::default());
        advisor.record_access("seg-001.ts");
        advisor.record_access("seg-002.ts");
        assert_eq!(advisor.tracked_prefixes(), 1);

        advisor.clear();
        assert_eq!(advisor.tracked_prefixes(), 0);
    }

    #[test]
    fn test_sequential_fraction_edge() {
        let mut tracker = PrefixTracker::new(".ts".to_string(), 3, 8);
        // Single entry — fraction should be 0
        tracker.record(1);
        assert!((tracker.sequential_fraction() - 0.0).abs() < f64::EPSILON);

        // Two sequential — fraction 1.0
        tracker.record(2);
        assert!((tracker.sequential_fraction() - 1.0).abs() < f64::EPSILON);
    }
}
