//! Preset chaining and composition.
//!
//! Allows multiple presets to be composed into an ordered pipeline where
//! each step overrides or merges specific settings from the previous step.
//! This is useful for building multi-pass encoding workflows or applying
//! successive refinement layers on top of a base preset.

#![allow(dead_code)]

use std::collections::HashMap;

// ── ChainPriority ──────────────────────────────────────────────────────────

/// Determines how conflicting values are resolved when two chain links
/// specify the same parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainPriority {
    /// Later links overwrite earlier links (last-write-wins).
    LastWins,
    /// Earlier links take precedence; later links only fill gaps.
    FirstWins,
    /// Pick the numerically higher value (for bitrates, quality, etc.).
    HigherWins,
    /// Pick the numerically lower value (for latency, file-size, etc.).
    LowerWins,
}

impl Default for ChainPriority {
    fn default() -> Self {
        Self::LastWins
    }
}

// ── ChainParam ─────────────────────────────────────────────────────────────

/// A single parameter override stored in a chain link.
#[derive(Debug, Clone, PartialEq)]
pub enum ChainParam {
    /// An integer parameter (bitrate, width, height, etc.).
    Int(i64),
    /// A floating-point parameter (quality factor, CRF, etc.).
    Float(f64),
    /// A string parameter (codec name, profile, etc.).
    Text(String),
    /// A boolean flag.
    Bool(bool),
}

impl ChainParam {
    /// Return the integer value if this is an `Int` variant.
    #[must_use]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Return the float value if this is a `Float` variant.
    #[must_use]
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Return a string reference if this is a `Text` variant.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(v) => Some(v),
            _ => None,
        }
    }

    /// Return the bool value if this is a `Bool` variant.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

// ── ChainLink ──────────────────────────────────────────────────────────────

/// A single link in a preset chain.
///
/// Each link carries a human-readable label and a set of named parameter
/// overrides that will be merged according to the chain's [`ChainPriority`].
#[derive(Debug, Clone)]
pub struct ChainLink {
    /// Human-readable label for this link (e.g. "base-1080p", "hdr-overlay").
    pub label: String,
    /// Parameter overrides keyed by canonical parameter name.
    pub params: HashMap<String, ChainParam>,
    /// Whether this link is enabled (disabled links are skipped during merge).
    pub enabled: bool,
}

impl ChainLink {
    /// Create a new, enabled chain link with the given label and no overrides.
    #[must_use]
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            params: HashMap::new(),
            enabled: true,
        }
    }

    /// Set a parameter override.
    pub fn set(&mut self, key: &str, value: ChainParam) {
        self.params.insert(key.to_string(), value);
    }

    /// Builder-style parameter setter.
    #[must_use]
    pub fn with_param(mut self, key: &str, value: ChainParam) -> Self {
        self.set(key, value);
        self
    }

    /// Disable this link so it is skipped during merge.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enable this link.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Get a parameter by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&ChainParam> {
        self.params.get(key)
    }

    /// Number of parameter overrides in this link.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.params.len()
    }
}

// ── PresetChain ────────────────────────────────────────────────────────────

/// An ordered sequence of [`ChainLink`]s merged using a [`ChainPriority`].
///
/// Call [`PresetChain::resolve`] to flatten the chain into a single
/// parameter map.
#[derive(Debug, Clone)]
pub struct PresetChain {
    /// Human-readable name for this chain.
    pub name: String,
    /// Ordered links (index 0 is applied first).
    links: Vec<ChainLink>,
    /// Conflict-resolution strategy.
    priority: ChainPriority,
}

impl PresetChain {
    /// Create a new, empty preset chain.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            links: Vec::new(),
            priority: ChainPriority::default(),
        }
    }

    /// Create a chain with a specific priority strategy.
    #[must_use]
    pub fn with_priority(mut self, priority: ChainPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Append a link to the end of the chain.
    pub fn push(&mut self, link: ChainLink) {
        self.links.push(link);
    }

    /// Insert a link at the given position.
    ///
    /// # Panics
    /// Panics if `index > self.len()`.
    pub fn insert(&mut self, index: usize, link: ChainLink) {
        self.links.insert(index, link);
    }

    /// Remove and return the link at the given position.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    pub fn remove(&mut self, index: usize) -> ChainLink {
        self.links.remove(index)
    }

    /// Number of links (including disabled ones).
    #[must_use]
    pub fn len(&self) -> usize {
        self.links.len()
    }

    /// Whether the chain contains no links.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// Number of *enabled* links.
    #[must_use]
    pub fn enabled_count(&self) -> usize {
        self.links.iter().filter(|l| l.enabled).count()
    }

    /// Immutable access to a link by index.
    #[must_use]
    pub fn get_link(&self, index: usize) -> Option<&ChainLink> {
        self.links.get(index)
    }

    /// Mutable access to a link by index.
    pub fn get_link_mut(&mut self, index: usize) -> Option<&mut ChainLink> {
        self.links.get_mut(index)
    }

    /// Resolve (flatten) the chain into a single parameter map.
    ///
    /// Disabled links are skipped. Conflicts are resolved according to the
    /// chain's [`ChainPriority`].
    #[must_use]
    pub fn resolve(&self) -> HashMap<String, ChainParam> {
        let mut result: HashMap<String, ChainParam> = HashMap::new();

        for link in self.links.iter().filter(|l| l.enabled) {
            for (key, value) in &link.params {
                match self.priority {
                    ChainPriority::LastWins => {
                        result.insert(key.clone(), value.clone());
                    }
                    ChainPriority::FirstWins => {
                        result.entry(key.clone()).or_insert_with(|| value.clone());
                    }
                    ChainPriority::HigherWins => {
                        let insert = match result.get(key) {
                            Some(ChainParam::Int(existing)) => {
                                value.as_int().map_or(false, |v| v > *existing)
                            }
                            Some(ChainParam::Float(existing)) => {
                                value.as_float().map_or(false, |v| v > *existing)
                            }
                            None => true,
                            _ => false,
                        };
                        if insert {
                            result.insert(key.clone(), value.clone());
                        }
                    }
                    ChainPriority::LowerWins => {
                        let insert = match result.get(key) {
                            Some(ChainParam::Int(existing)) => {
                                value.as_int().map_or(false, |v| v < *existing)
                            }
                            Some(ChainParam::Float(existing)) => {
                                value.as_float().map_or(false, |v| v < *existing)
                            }
                            None => true,
                            _ => false,
                        };
                        if insert {
                            result.insert(key.clone(), value.clone());
                        }
                    }
                }
            }
        }

        result
    }

    /// Return the labels of all enabled links, in order.
    #[must_use]
    pub fn enabled_labels(&self) -> Vec<&str> {
        self.links
            .iter()
            .filter(|l| l.enabled)
            .map(|l| l.label.as_str())
            .collect()
    }

    /// Return all unique parameter keys mentioned across all enabled links.
    #[must_use]
    pub fn all_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self
            .links
            .iter()
            .filter(|l| l.enabled)
            .flat_map(|l| l.params.keys().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        keys.sort();
        keys
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn base_link() -> ChainLink {
        ChainLink::new("base")
            .with_param("bitrate", ChainParam::Int(5_000_000))
            .with_param("codec", ChainParam::Text("h264".into()))
            .with_param("crf", ChainParam::Float(23.0))
    }

    fn overlay_link() -> ChainLink {
        ChainLink::new("overlay")
            .with_param("bitrate", ChainParam::Int(8_000_000))
            .with_param("hdr", ChainParam::Bool(true))
    }

    // ── ChainLink ──

    #[test]
    fn test_chain_link_creation() {
        let link = ChainLink::new("test");
        assert_eq!(link.label, "test");
        assert!(link.enabled);
        assert_eq!(link.param_count(), 0);
    }

    #[test]
    fn test_chain_link_with_param() {
        let link = base_link();
        assert_eq!(link.param_count(), 3);
        assert_eq!(
            link.get("bitrate").expect("get should succeed").as_int(),
            Some(5_000_000)
        );
    }

    #[test]
    fn test_chain_link_disable_enable() {
        let mut link = ChainLink::new("x");
        assert!(link.enabled);
        link.disable();
        assert!(!link.enabled);
        link.enable();
        assert!(link.enabled);
    }

    #[test]
    fn test_chain_param_as_text() {
        let p = ChainParam::Text("hevc".into());
        assert_eq!(p.as_text(), Some("hevc"));
        assert_eq!(p.as_int(), None);
    }

    #[test]
    fn test_chain_param_as_bool() {
        let p = ChainParam::Bool(true);
        assert_eq!(p.as_bool(), Some(true));
        assert_eq!(p.as_float(), None);
    }

    // ── PresetChain ──

    #[test]
    fn test_chain_empty() {
        let chain = PresetChain::new("empty");
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        assert!(chain.resolve().is_empty());
    }

    #[test]
    fn test_chain_last_wins_default() {
        let mut chain = PresetChain::new("test");
        chain.push(base_link());
        chain.push(overlay_link());
        let resolved = chain.resolve();
        // overlay's bitrate (8M) should overwrite base's (5M)
        assert_eq!(
            resolved
                .get("bitrate")
                .expect("get should succeed")
                .as_int(),
            Some(8_000_000)
        );
        // codec comes from base only
        assert_eq!(
            resolved.get("codec").expect("get should succeed").as_text(),
            Some("h264")
        );
        // hdr comes from overlay
        assert_eq!(
            resolved.get("hdr").expect("get should succeed").as_bool(),
            Some(true)
        );
    }

    #[test]
    fn test_chain_first_wins() {
        let mut chain = PresetChain::new("fw").with_priority(ChainPriority::FirstWins);
        chain.push(base_link());
        chain.push(overlay_link());
        let resolved = chain.resolve();
        // base's bitrate (5M) wins
        assert_eq!(
            resolved
                .get("bitrate")
                .expect("get should succeed")
                .as_int(),
            Some(5_000_000)
        );
    }

    #[test]
    fn test_chain_higher_wins() {
        let mut chain = PresetChain::new("hw").with_priority(ChainPriority::HigherWins);
        chain.push(base_link());
        chain.push(overlay_link());
        let resolved = chain.resolve();
        assert_eq!(
            resolved
                .get("bitrate")
                .expect("get should succeed")
                .as_int(),
            Some(8_000_000)
        );
    }

    #[test]
    fn test_chain_lower_wins() {
        let mut chain = PresetChain::new("lw").with_priority(ChainPriority::LowerWins);
        chain.push(base_link());
        chain.push(overlay_link());
        let resolved = chain.resolve();
        assert_eq!(
            resolved
                .get("bitrate")
                .expect("get should succeed")
                .as_int(),
            Some(5_000_000)
        );
    }

    #[test]
    fn test_chain_disabled_link_skipped() {
        let mut chain = PresetChain::new("skip");
        chain.push(base_link());
        let mut disabled = overlay_link();
        disabled.disable();
        chain.push(disabled);
        let resolved = chain.resolve();
        assert_eq!(
            resolved
                .get("bitrate")
                .expect("get should succeed")
                .as_int(),
            Some(5_000_000)
        );
        assert!(!resolved.contains_key("hdr"));
    }

    #[test]
    fn test_chain_enabled_count() {
        let mut chain = PresetChain::new("ec");
        chain.push(base_link());
        let mut d = overlay_link();
        d.disable();
        chain.push(d);
        assert_eq!(chain.len(), 2);
        assert_eq!(chain.enabled_count(), 1);
    }

    #[test]
    fn test_chain_enabled_labels() {
        let mut chain = PresetChain::new("labels");
        chain.push(base_link());
        chain.push(overlay_link());
        assert_eq!(chain.enabled_labels(), vec!["base", "overlay"]);
    }

    #[test]
    fn test_chain_all_keys() {
        let mut chain = PresetChain::new("keys");
        chain.push(base_link());
        chain.push(overlay_link());
        let keys = chain.all_keys();
        assert!(keys.contains(&"bitrate".to_string()));
        assert!(keys.contains(&"codec".to_string()));
        assert!(keys.contains(&"hdr".to_string()));
        assert!(keys.contains(&"crf".to_string()));
    }

    #[test]
    fn test_chain_insert_and_remove() {
        let mut chain = PresetChain::new("ir");
        chain.push(base_link());
        chain.push(overlay_link());
        let mid = ChainLink::new("mid");
        chain.insert(1, mid);
        assert_eq!(chain.len(), 3);
        assert_eq!(
            chain.get_link(1).expect("get_link should succeed").label,
            "mid"
        );
        let removed = chain.remove(1);
        assert_eq!(removed.label, "mid");
        assert_eq!(chain.len(), 2);
    }
}
