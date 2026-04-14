//! Quality tiers and adaptive tier selection for proxy workflows.
//!
//! A [`ProxyTier`] defines a named proxy quality level with a maximum width
//! and target bitrate.  [`ProxyTierSelector`] chooses the best tier for a
//! given source resolution and available network bandwidth.

/// A single proxy quality tier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyTier {
    /// Human-readable name (e.g. `"1080p"`, `"720p"`, `"360p"`).
    pub name: String,
    /// Maximum horizontal resolution in pixels.
    pub max_width: u32,
    /// Target bitrate in kilobits per second.
    pub bitrate_kbps: u32,
}

impl ProxyTier {
    /// Create a new proxy tier.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_proxy::tiers::ProxyTier;
    ///
    /// let tier = ProxyTier::new("720p", 1280, 3_000);
    /// assert_eq!(tier.name, "720p");
    /// assert_eq!(tier.max_width, 1280);
    /// assert_eq!(tier.bitrate_kbps, 3_000);
    /// ```
    #[must_use]
    pub fn new(name: &str, max_width: u32, bitrate_kbps: u32) -> Self {
        Self {
            name: name.to_string(),
            max_width,
            bitrate_kbps,
        }
    }

    /// Return `true` if this tier is suitable for the given source width and
    /// network bandwidth.
    ///
    /// A tier is suitable when:
    /// - `original_w <= self.max_width` **or** the tier's `max_width >=
    ///   original_w` (the tier can represent the full source), **and**
    /// - the tier's `bitrate_kbps <= network_kbps` (enough bandwidth).
    #[must_use]
    pub fn is_suitable(&self, original_w: u32, network_kbps: u32) -> bool {
        self.bitrate_kbps <= network_kbps && self.max_width <= original_w.max(self.max_width)
    }
}

/// Selects the best [`ProxyTier`] for a given source resolution and available
/// network bandwidth.
#[derive(Debug, Default)]
pub struct ProxyTierSelector {
    tiers: Vec<ProxyTier>,
}

impl ProxyTierSelector {
    /// Create a new selector with no tiers.
    #[must_use]
    pub fn new() -> Self {
        Self { tiers: Vec::new() }
    }

    /// Create a selector pre-populated with standard broadcast/streaming tiers.
    ///
    /// Tiers (highest to lowest): 4K (3840px, 20 Mbps), 1080p (1920px, 8 Mbps),
    /// 720p (1280px, 3 Mbps), 480p (854px, 1.5 Mbps), 360p (640px, 800 Kbps).
    #[must_use]
    pub fn standard() -> Self {
        let mut s = Self::new();
        s.add_tier(ProxyTier::new("4K",    3840, 20_000));
        s.add_tier(ProxyTier::new("1080p", 1920,  8_000));
        s.add_tier(ProxyTier::new("720p",  1280,  3_000));
        s.add_tier(ProxyTier::new("480p",   854,  1_500));
        s.add_tier(ProxyTier::new("360p",   640,    800));
        s
    }

    /// Add a tier to the selector.
    pub fn add_tier(&mut self, tier: ProxyTier) {
        self.tiers.push(tier);
    }

    /// Select the best tier for the given source width and network bandwidth.
    ///
    /// Returns the highest-quality tier whose `bitrate_kbps <= network_kbps`
    /// and whose `max_width <= original_w` (or the closest match below).  If
    /// no tier fits within the network budget, the lowest-bitrate tier is
    /// returned as a fallback.
    ///
    /// Returns `None` only when no tiers have been added.
    #[must_use]
    pub fn select(&self, original_w: u32, network_kbps: u32) -> Option<&ProxyTier> {
        if self.tiers.is_empty() {
            return None;
        }

        // Filter tiers that fit the bandwidth and don't exceed the source width
        let mut candidates: Vec<&ProxyTier> = self
            .tiers
            .iter()
            .filter(|t| t.bitrate_kbps <= network_kbps && t.max_width <= original_w)
            .collect();

        if candidates.is_empty() {
            // Fallback: pick the tier with the lowest bitrate regardless of width
            return self.tiers.iter().min_by_key(|t| t.bitrate_kbps);
        }

        // Among candidates, prefer the highest bitrate (best quality)
        candidates.sort_by(|a, b| b.bitrate_kbps.cmp(&a.bitrate_kbps));
        candidates.into_iter().next()
    }

    /// Return a reference to all registered tiers.
    #[must_use]
    pub fn tiers(&self) -> &[ProxyTier] {
        &self.tiers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxy_tier_new_stores_fields() {
        let t = ProxyTier::new("test", 1920, 5_000);
        assert_eq!(t.name, "test");
        assert_eq!(t.max_width, 1920);
        assert_eq!(t.bitrate_kbps, 5_000);
    }

    #[test]
    fn selector_empty_returns_none() {
        let sel = ProxyTierSelector::new();
        assert!(sel.select(1920, 10_000).is_none());
    }

    #[test]
    fn selector_picks_best_fitting_tier() {
        let mut sel = ProxyTierSelector::new();
        sel.add_tier(ProxyTier::new("HD",  1920, 8_000));
        sel.add_tier(ProxyTier::new("SD",   854, 1_500));
        sel.add_tier(ProxyTier::new("Low",  640,   800));

        // 10 Mbps, 1080p source → should pick HD
        let t = sel.select(1920, 10_000).expect("tier");
        assert_eq!(t.name, "HD");
    }

    #[test]
    fn selector_constrained_bandwidth_picks_lower_tier() {
        let sel = ProxyTierSelector::standard();
        // Only 2 Mbps → 720p (3 Mbps) is too heavy, should fall back to 480p
        let t = sel.select(1920, 2_000).expect("tier");
        assert!(t.bitrate_kbps <= 2_000, "bitrate={}", t.bitrate_kbps);
    }

    #[test]
    fn selector_fallback_when_all_exceed_budget() {
        let mut sel = ProxyTierSelector::new();
        sel.add_tier(ProxyTier::new("A", 1920, 5_000));
        sel.add_tier(ProxyTier::new("B", 1280, 3_000));
        // Only 500 Kbps — both exceed budget
        let t = sel.select(1920, 500).expect("fallback");
        // Fallback selects the tier with the lowest bitrate
        assert_eq!(t.name, "B");
    }

    #[test]
    fn standard_selector_has_five_tiers() {
        let sel = ProxyTierSelector::standard();
        assert_eq!(sel.tiers().len(), 5);
    }

    #[test]
    fn tier_is_suitable_checks_bandwidth() {
        let t = ProxyTier::new("720p", 1280, 3_000);
        assert!(t.is_suitable(1920, 5_000));
        assert!(!t.is_suitable(1920, 1_000));
    }
}
