//! Query suggestion via prefix matching and edit-distance ranking.
//!
//! [`QuerySuggester`] accepts an index of known query strings at construction
//! time and answers prefix-based suggestions at query time.  Candidates are
//! first filtered to those sharing the given prefix, then ranked in ascending
//! order of Levenshtein edit distance from the prefix query.  Ties are broken
//! lexicographically to ensure deterministic output.
//!
//! # Algorithm
//!
//! 1. **Prefix filter** — retain only index terms that start with `prefix`
//!    (case-insensitive).
//! 2. **Edit-distance sort** — compute the Levenshtein distance between
//!    `prefix` and each candidate, then sort ascending.  The edit distance
//!    between a prefix and a longer string that starts with that prefix is
//!    always `len(string) - len(prefix)`, so shorter matches rank first.
//! 3. **Deduplication** — duplicate strings (after lowercasing) are removed.
//!
//! The implementation uses the standard DP recurrence for Levenshtein distance
//! with O(m·n) time and O(n) space (two-row optimisation).
//!
//! # Example
//!
//! ```
//! use oximedia_search::suggest::QuerySuggester;
//!
//! let index = vec![
//!     "audio".to_string(),
//!     "audio mixing".to_string(),
//!     "audition".to_string(),
//!     "video".to_string(),
//! ];
//! let suggester = QuerySuggester::new(&index);
//! let suggestions = suggester.suggest("aud");
//! assert!(suggestions.iter().any(|s| s == "audio"));
//! assert!(suggestions.iter().all(|s| s.to_lowercase().starts_with("aud")));
//! ```

#![allow(dead_code)]

// ─────────────────────────────────────────────────────────────────────────────
// QuerySuggester
// ─────────────────────────────────────────────────────────────────────────────

/// Query suggester backed by an in-memory index of known query strings.
///
/// Suggestions are prefix-filtered then ranked by Levenshtein distance from
/// the query prefix, with lexicographic tie-breaking.
#[derive(Debug)]
pub struct QuerySuggester {
    /// Lower-cased index terms.
    index: Vec<String>,
    /// Original (non-lowercased) versions, aligned with `index`.
    originals: Vec<String>,
}

impl QuerySuggester {
    /// Build a `QuerySuggester` from a slice of known query strings.
    ///
    /// Duplicate strings (case-insensitive) are retained; callers that want
    /// deduplication should deduplicate the input before construction.
    pub fn new(index: &[String]) -> Self {
        let originals: Vec<String> = index.to_vec();
        let lower: Vec<String> = originals.iter().map(|s| s.to_lowercase()).collect();
        Self {
            index: lower,
            originals,
        }
    }

    /// Return suggestions for `prefix`.
    ///
    /// The prefix comparison is **case-insensitive**.  The returned strings
    /// use the original casing from the index.
    ///
    /// Results are:
    /// 1. Filtered to terms that start with `prefix` (case-insensitive).
    /// 2. Sorted ascending by Levenshtein distance from `prefix`.
    /// 3. Tie-broken lexicographically on the original string.
    /// 4. Deduplicated (first occurrence wins when case-folded duplicates exist).
    pub fn suggest(&self, prefix: &str) -> Vec<String> {
        let prefix_lower = prefix.to_lowercase();

        // Collect (distance, original) pairs for candidates.
        let mut candidates: Vec<(usize, &str)> = self
            .index
            .iter()
            .zip(self.originals.iter())
            .filter(|(lower, _)| lower.starts_with(&prefix_lower))
            .map(|(lower, orig)| {
                let dist = levenshtein_distance(&prefix_lower, lower);
                (dist, orig.as_str())
            })
            .collect();

        // Sort: primary = distance asc, secondary = lexicographic asc.
        candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));

        // Deduplicate on lowercase form; first occurrence (lowest dist) wins.
        let mut seen = std::collections::HashSet::new();
        candidates
            .into_iter()
            .filter(|(_, orig)| seen.insert(orig.to_lowercase()))
            .map(|(_, orig)| orig.to_string())
            .collect()
    }

    /// Return up to `limit` suggestions for `prefix`.
    pub fn suggest_top(&self, prefix: &str, limit: usize) -> Vec<String> {
        self.suggest(prefix).into_iter().take(limit).collect()
    }

    /// Number of terms in the index.
    pub fn index_size(&self) -> usize {
        self.index.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Levenshtein distance (O(m·n) DP, O(n) space)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Levenshtein (edit) distance between `a` and `b`.
///
/// Uses a two-row DP approach to keep memory usage O(min(|a|, |b|)).
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let m = a_chars.len();
    let n = b_chars.len();

    // Optimise: work on the shorter string in the inner loop.
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // `prev[j]` = cost of aligning a[..i-1] with b[..j].
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];

    for (i, &ca) in a_chars.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in b_chars.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1) // deletion
                .min(curr[j] + 1) // insertion
                .min(prev[j] + cost); // substitution
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn idx(terms: &[&str]) -> Vec<String> {
        terms.iter().map(|s| s.to_string()).collect()
    }

    // ── levenshtein_distance ─────────────────────────────────────────────────

    #[test]
    fn test_lev_equal_strings() {
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
    }

    #[test]
    fn test_lev_empty_strings() {
        assert_eq!(levenshtein_distance("", ""), 0);
    }

    #[test]
    fn test_lev_one_empty() {
        assert_eq!(levenshtein_distance("abc", ""), 3);
        assert_eq!(levenshtein_distance("", "abc"), 3);
    }

    #[test]
    fn test_lev_substitution() {
        // "kitten" → "sitten" (1 sub)
        assert_eq!(levenshtein_distance("kitten", "sitten"), 1);
    }

    #[test]
    fn test_lev_insertion() {
        assert_eq!(levenshtein_distance("ab", "abc"), 1);
    }

    #[test]
    fn test_lev_deletion() {
        assert_eq!(levenshtein_distance("abc", "ab"), 1);
    }

    #[test]
    fn test_lev_classic_kitten_sitting() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
    }

    // ── QuerySuggester ───────────────────────────────────────────────────────

    #[test]
    fn test_suggest_prefix_match() {
        let s = QuerySuggester::new(&idx(&["audio", "audio mixing", "audition", "video"]));
        let suggestions = s.suggest("aud");
        assert!(
            suggestions.iter().any(|x| x == "audio"),
            "should suggest 'audio'"
        );
        assert!(
            suggestions.iter().any(|x| x == "audio mixing"),
            "should suggest 'audio mixing'"
        );
        assert!(
            suggestions.iter().any(|x| x == "audition"),
            "should suggest 'audition'"
        );
        assert!(
            !suggestions.iter().any(|x| x == "video"),
            "should not suggest 'video'"
        );
    }

    #[test]
    fn test_suggest_case_insensitive() {
        let s = QuerySuggester::new(&idx(&["Audio", "AUDIO MIXING", "video"]));
        let suggestions = s.suggest("aud");
        assert!(
            suggestions.iter().any(|x| x == "Audio"),
            "case-insensitive prefix match should work"
        );
    }

    #[test]
    fn test_suggest_empty_prefix_returns_all() {
        let s = QuerySuggester::new(&idx(&["a", "b", "c"]));
        let suggestions = s.suggest("");
        assert_eq!(suggestions.len(), 3);
    }

    #[test]
    fn test_suggest_no_match_returns_empty() {
        let s = QuerySuggester::new(&idx(&["audio", "video"]));
        let suggestions = s.suggest("xyz");
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_suggest_shorter_first() {
        // "audio" is shorter than "audio mixing" — should rank before it.
        let s = QuerySuggester::new(&idx(&["audio mixing", "audio"]));
        let suggestions = s.suggest("audio");
        assert_eq!(suggestions[0], "audio");
    }

    #[test]
    fn test_suggest_top_limits_results() {
        let s =
            QuerySuggester::new(&idx(&["alpha", "also", "altitude", "always", "algorithm"]));
        let top = s.suggest_top("al", 3);
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn test_suggest_deduplication() {
        // Two entries that are the same when lowercased.
        let s = QuerySuggester::new(&idx(&["Audio", "audio"]));
        let suggestions = s.suggest("aud");
        assert_eq!(suggestions.len(), 1, "duplicate entries should be deduplicated");
    }

    #[test]
    fn test_index_size() {
        let s = QuerySuggester::new(&idx(&["a", "b", "c"]));
        assert_eq!(s.index_size(), 3);
    }

    #[test]
    fn test_suggest_full_word_exact() {
        let s = QuerySuggester::new(&idx(&["audio", "audiophile"]));
        let suggestions = s.suggest("audio");
        assert!(suggestions.iter().any(|x| x == "audio"));
        assert!(suggestions.iter().any(|x| x == "audiophile"));
    }

    #[test]
    fn test_suggest_empty_index() {
        let s = QuerySuggester::new(&[]);
        assert!(s.suggest("any").is_empty());
    }
}
