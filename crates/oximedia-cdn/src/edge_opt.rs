//! Edge bandwidth optimisation — select the most appropriate format for a
//! given client bitrate.
//!
//! [`EdgeBandwidthOptimizer`] accepts a client's measured downlink bandwidth
//! and a list of `(format_name, required_bitrate_kbps)` pairs, then returns
//! the name of the best format the client can receive without buffering.

/// Edge-side bandwidth optimizer.
///
/// Given a client's available bitrate and a set of available media formats
/// with their minimum bitrate requirements, it selects the highest-quality
/// format that fits within the client's budget.
///
/// # Example
/// ```
/// use oximedia_cdn::edge_opt::EdgeBandwidthOptimizer;
///
/// let formats = [
///     ("4K", 20_000_u32),
///     ("1080p", 8_000_u32),
///     ("720p",  4_000_u32),
///     ("480p",  2_000_u32),
/// ];
/// let opt = EdgeBandwidthOptimizer::new();
/// // Client with 10 Mbps → 1080p
/// assert_eq!(opt.select_format(10_000, &formats), "1080p");
/// // Client with 1.5 Mbps → 480p (the only one that fits)
/// assert_eq!(opt.select_format(1_500, &formats), "480p");
/// ```
pub struct EdgeBandwidthOptimizer;

impl EdgeBandwidthOptimizer {
    /// Create a new optimizer.
    pub fn new() -> Self {
        Self
    }

    /// Select the best format for the given client bitrate.
    ///
    /// Iterates through `available_formats` and picks the format with the
    /// highest required bitrate that is still ≤ `client_bitrate_kbps`.
    ///
    /// If no format fits the client's bitrate the lowest-bitrate format is
    /// returned as a fallback (graceful degradation).
    ///
    /// # Panics
    /// Panics if `available_formats` is empty.
    pub fn select_format<'a>(
        &self,
        client_bitrate_kbps: u32,
        available_formats: &[(&'a str, u32)],
    ) -> &'a str {
        assert!(
            !available_formats.is_empty(),
            "available_formats must not be empty"
        );

        // Find the best-fitting format (highest bitrate ≤ client bitrate).
        let best = available_formats
            .iter()
            .filter(|(_, required)| *required <= client_bitrate_kbps)
            .max_by_key(|(_, required)| *required);

        match best {
            Some((name, _)) => name,
            None => {
                // Client bitrate is below all formats — return the cheapest.
                available_formats
                    .iter()
                    .min_by_key(|(_, r)| *r)
                    .map(|(name, _)| *name)
                    .unwrap_or(available_formats[0].0)
            }
        }
    }
}

impl Default for EdgeBandwidthOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn formats() -> Vec<(&'static str, u32)> {
        vec![
            ("4K", 20_000),
            ("1080p", 8_000),
            ("720p", 4_000),
            ("480p", 2_000),
            ("360p", 1_000),
        ]
    }

    #[test]
    fn test_exact_match() {
        let opt = EdgeBandwidthOptimizer::new();
        assert_eq!(opt.select_format(8_000, &formats()), "1080p");
    }

    #[test]
    fn test_best_fitting() {
        let opt = EdgeBandwidthOptimizer::new();
        assert_eq!(opt.select_format(10_000, &formats()), "1080p");
    }

    #[test]
    fn test_full_bandwidth() {
        let opt = EdgeBandwidthOptimizer::new();
        assert_eq!(opt.select_format(25_000, &formats()), "4K");
    }

    #[test]
    fn test_fallback_to_lowest() {
        let opt = EdgeBandwidthOptimizer::new();
        // Client bitrate below all formats → fallback to lowest
        assert_eq!(opt.select_format(500, &formats()), "360p");
    }

    #[test]
    fn test_single_format() {
        let opt = EdgeBandwidthOptimizer::new();
        let fmts = [("only", 5_000_u32)];
        assert_eq!(opt.select_format(1_000, &fmts), "only");
        assert_eq!(opt.select_format(9_000, &fmts), "only");
    }
}
