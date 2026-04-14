//! Rights conflict detection: overlapping territory+time windows.
//!
//! [`RightsConflictDetector`] scans a slice of [`RightsWindow`] entries and
//! identifies pairs whose territory codes and time ranges overlap.

/// A rights window representing a grant over a territory for a time period.
#[derive(Debug, Clone)]
pub struct RightsWindow {
    /// Unique identifier for this rights window.
    pub id: u64,
    /// ISO 3166-1 alpha-2 territory code (e.g. `"US"`, `"GB"`) or `"*"` for
    /// worldwide.
    pub territory: String,
    /// Unix timestamp (seconds) when the rights window begins.
    pub start_ts: u64,
    /// Unix timestamp (seconds) when the rights window ends (exclusive).
    pub end_ts: u64,
}

impl RightsWindow {
    /// Create a new rights window.
    pub fn new(id: u64, territory: impl Into<String>, start_ts: u64, end_ts: u64) -> Self {
        Self {
            id,
            territory: territory.into(),
            start_ts,
            end_ts,
        }
    }

    /// Returns `true` if this window's territory overlaps with `other`.
    ///
    /// A worldwide territory (`"*"`) overlaps with every other territory.
    fn territory_overlaps(&self, other: &RightsWindow) -> bool {
        self.territory == "*"
            || other.territory == "*"
            || self.territory == other.territory
    }

    /// Returns `true` if this window's time range overlaps with `other`.
    ///
    /// Two ranges `[a_start, a_end)` and `[b_start, b_end)` overlap when
    /// `a_start < b_end && b_start < a_end`.
    fn time_overlaps(&self, other: &RightsWindow) -> bool {
        self.start_ts < other.end_ts && other.start_ts < self.end_ts
    }
}

/// Detects pairs of [`RightsWindow`] entries that conflict (overlap in both
/// territory and time).
pub struct RightsConflictDetector;

impl RightsConflictDetector {
    /// Find all pairs `(i, j)` where `i < j` and `rights[i]` and `rights[j]`
    /// overlap in both territory and time range.
    ///
    /// Returns a `Vec` of index pairs into the supplied slice.
    pub fn find_overlaps(rights: &[RightsWindow]) -> Vec<(usize, usize)> {
        let mut overlaps = Vec::new();
        for i in 0..rights.len() {
            for j in (i + 1)..rights.len() {
                let a = &rights[i];
                let b = &rights[j];
                if a.territory_overlaps(b) && a.time_overlaps(b) {
                    overlaps.push((i, j));
                }
            }
        }
        overlaps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn win(id: u64, territory: &str, start: u64, end: u64) -> RightsWindow {
        RightsWindow::new(id, territory, start, end)
    }

    #[test]
    fn test_no_overlap_different_territory() {
        let windows = vec![
            win(1, "US", 0, 100),
            win(2, "GB", 0, 100),
        ];
        assert!(RightsConflictDetector::find_overlaps(&windows).is_empty());
    }

    #[test]
    fn test_no_overlap_adjacent_time() {
        let windows = vec![
            win(1, "US", 0, 100),
            win(2, "US", 100, 200),
        ];
        assert!(RightsConflictDetector::find_overlaps(&windows).is_empty());
    }

    #[test]
    fn test_overlap_same_territory() {
        let windows = vec![
            win(1, "US", 0, 200),
            win(2, "US", 100, 300),
        ];
        let result = RightsConflictDetector::find_overlaps(&windows);
        assert_eq!(result, vec![(0, 1)]);
    }

    #[test]
    fn test_overlap_worldwide_wildcard() {
        let windows = vec![
            win(1, "*", 0, 200),
            win(2, "US", 100, 300),
        ];
        let result = RightsConflictDetector::find_overlaps(&windows);
        assert_eq!(result, vec![(0, 1)]);
    }

    #[test]
    fn test_multiple_overlaps() {
        let windows = vec![
            win(1, "US", 0, 300),
            win(2, "US", 100, 400),
            win(3, "US", 200, 500),
        ];
        let result = RightsConflictDetector::find_overlaps(&windows);
        // (0,1), (0,2), (1,2) all overlap
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_empty_slice() {
        let windows: Vec<RightsWindow> = vec![];
        assert!(RightsConflictDetector::find_overlaps(&windows).is_empty());
    }

    #[test]
    fn test_single_window_no_overlap() {
        let windows = vec![win(1, "US", 0, 100)];
        assert!(RightsConflictDetector::find_overlaps(&windows).is_empty());
    }

    #[test]
    fn test_worldwide_vs_worldwide_overlaps() {
        let windows = vec![
            win(1, "*", 0, 200),
            win(2, "*", 100, 300),
        ];
        let result = RightsConflictDetector::find_overlaps(&windows);
        assert_eq!(result, vec![(0, 1)]);
    }

    #[test]
    fn test_time_overlap_one_contains_other() {
        // Window 2 is entirely inside window 1.
        let windows = vec![
            win(1, "DE", 0, 1000),
            win(2, "DE", 100, 200),
        ];
        let result = RightsConflictDetector::find_overlaps(&windows);
        assert_eq!(result, vec![(0, 1)]);
    }

    #[test]
    fn test_non_overlapping_time_with_same_territory() {
        // Gap between windows: end of first equals start of second (exclusive range).
        let windows = vec![
            win(1, "FR", 0, 50),
            win(2, "FR", 50, 100),
        ];
        assert!(RightsConflictDetector::find_overlaps(&windows).is_empty());
    }

    #[test]
    fn test_overlap_ids_are_correct_indices() {
        let windows = vec![
            win(10, "JP", 0, 100),
            win(20, "JP", 50, 150),
            win(30, "US", 0, 100),
        ];
        // Only windows 0 and 1 overlap (same territory JP, overlapping time).
        let result = RightsConflictDetector::find_overlaps(&windows);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], (0, 1));
    }
}
