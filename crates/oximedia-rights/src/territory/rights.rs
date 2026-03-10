//! Territory rights management for content distribution

#![allow(dead_code)]

/// Represents a geographic territory with ISO code
#[derive(Debug, Clone, PartialEq)]
pub struct Territory {
    /// ISO 3166-1 alpha-2 territory code (e.g. "US", "GB")
    pub code: String,
    /// Human-readable name (e.g. "United States")
    pub name: String,
    /// Geographic / political region (e.g. "North America")
    pub region: String,
}

impl Territory {
    /// Create a new territory
    pub fn new(code: &str, name: &str, region: &str) -> Self {
        Self {
            code: code.to_string(),
            name: name.to_string(),
            region: region.to_string(),
        }
    }

    /// Return a full list of commonly used worldwide territories
    pub fn worldwide() -> Vec<Self> {
        let mut all = Self::north_america();
        all.extend(Self::europe());
        all.extend([
            Territory::new("JP", "Japan", "Asia"),
            Territory::new("CN", "China", "Asia"),
            Territory::new("IN", "India", "Asia"),
            Territory::new("KR", "South Korea", "Asia"),
            Territory::new("AU", "Australia", "Oceania"),
            Territory::new("NZ", "New Zealand", "Oceania"),
            Territory::new("BR", "Brazil", "South America"),
            Territory::new("AR", "Argentina", "South America"),
        ]);
        all
    }

    /// Return commonly used European territories
    pub fn europe() -> Vec<Self> {
        vec![
            Territory::new("GB", "United Kingdom", "Europe"),
            Territory::new("DE", "Germany", "Europe"),
            Territory::new("FR", "France", "Europe"),
            Territory::new("IT", "Italy", "Europe"),
            Territory::new("ES", "Spain", "Europe"),
            Territory::new("NL", "Netherlands", "Europe"),
            Territory::new("SE", "Sweden", "Europe"),
            Territory::new("NO", "Norway", "Europe"),
            Territory::new("DK", "Denmark", "Europe"),
            Territory::new("FI", "Finland", "Europe"),
            Territory::new("PL", "Poland", "Europe"),
            Territory::new("PT", "Portugal", "Europe"),
            Territory::new("CH", "Switzerland", "Europe"),
            Territory::new("AT", "Austria", "Europe"),
            Territory::new("BE", "Belgium", "Europe"),
        ]
    }

    /// Return commonly used North American territories
    pub fn north_america() -> Vec<Self> {
        vec![
            Territory::new("US", "United States", "North America"),
            Territory::new("CA", "Canada", "North America"),
            Territory::new("MX", "Mexico", "North America"),
        ]
    }
}

/// Rights configuration for a piece of content in specific territories
#[derive(Debug, Clone)]
pub struct TerritoryRights {
    /// Content identifier this rights record belongs to
    pub content_id: String,
    /// Territory codes explicitly allowed
    pub allowed: Vec<String>,
    /// Territory codes explicitly blocked (overrides allowed)
    pub blocked: Vec<String>,
    /// Optional Unix timestamp after which rights expire
    pub expires_at: Option<u64>,
}

impl TerritoryRights {
    /// Create a new territory rights record for the given content (deny-all by default)
    pub fn new(content_id: &str) -> Self {
        Self {
            content_id: content_id.to_string(),
            allowed: Vec::new(),
            blocked: Vec::new(),
            expires_at: None,
        }
    }

    /// Explicitly allow a territory by code
    pub fn allow(&mut self, territory_code: &str) {
        let code = territory_code.to_uppercase();
        if !self.allowed.contains(&code) {
            self.allowed.push(code);
        }
    }

    /// Explicitly block a territory by code (takes precedence over allowed)
    pub fn block(&mut self, territory_code: &str) {
        let code = territory_code.to_uppercase();
        if !self.blocked.contains(&code) {
            self.blocked.push(code);
        }
    }

    /// Allow all territories in the worldwide list
    pub fn allow_worldwide(&mut self) {
        for territory in Territory::worldwide() {
            self.allow(&territory.code);
        }
    }

    /// Check whether a territory is allowed (blocked list takes precedence)
    pub fn is_allowed(&self, territory_code: &str) -> bool {
        let code = territory_code.to_uppercase();
        if self.blocked.contains(&code) {
            return false;
        }
        self.allowed.contains(&code)
    }

    /// Check whether the rights have expired relative to the given Unix timestamp
    pub fn is_expired(&self, now: u64) -> bool {
        self.expires_at.is_some_and(|exp| now >= exp)
    }

    /// Set an expiry timestamp
    pub fn set_expires_at(&mut self, timestamp: u64) {
        self.expires_at = Some(timestamp);
    }

    /// Number of explicitly allowed territories
    pub fn allowed_count(&self) -> usize {
        self.allowed.len()
    }

    /// Number of explicitly blocked territories
    pub fn blocked_count(&self) -> usize {
        self.blocked.len()
    }
}

// ── Windowed territory rights ────────────────────────────────────────────────

/// A time-bound geographic rights window.
///
/// Represents the right to distribute content in specific territories during
/// a defined time period, supporting typical media distribution patterns
/// like theatrical → home video → streaming cascades per territory.
#[derive(Debug, Clone)]
pub struct TerritoryWindow {
    /// Unique window identifier.
    pub id: String,
    /// Content identifier.
    pub content_id: String,
    /// Territory codes this window covers.
    pub territories: Vec<String>,
    /// Start timestamp (Unix seconds).
    pub start_ts: i64,
    /// End timestamp (Unix seconds).
    pub end_ts: i64,
    /// Distribution channel (e.g. "theatrical", "streaming", "broadcast").
    pub channel: String,
    /// Whether exclusivity is granted (blocks other licensees in these
    /// territories during this window).
    pub exclusive: bool,
}

impl TerritoryWindow {
    /// Create a new territory window.
    pub fn new(
        id: impl Into<String>,
        content_id: impl Into<String>,
        channel: impl Into<String>,
        start_ts: i64,
        end_ts: i64,
    ) -> Self {
        Self {
            id: id.into(),
            content_id: content_id.into(),
            territories: Vec::new(),
            start_ts,
            end_ts,
            channel: channel.into(),
            exclusive: false,
        }
    }

    /// Add a territory code to this window.
    pub fn add_territory(&mut self, code: &str) {
        let code = code.to_uppercase();
        if !self.territories.contains(&code) {
            self.territories.push(code);
        }
    }

    /// Set exclusivity.
    pub fn set_exclusive(&mut self, exclusive: bool) {
        self.exclusive = exclusive;
    }

    /// Check whether a territory and timestamp fall within this window.
    pub fn covers(&self, territory_code: &str, ts: i64) -> bool {
        let code = territory_code.to_uppercase();
        ts >= self.start_ts && ts < self.end_ts && self.territories.contains(&code)
    }

    /// Duration in days.
    pub fn duration_days(&self) -> u64 {
        ((self.end_ts - self.start_ts).max(0) as u64) / 86_400
    }

    /// Whether the window is currently active at the given timestamp.
    pub fn is_active(&self, ts: i64) -> bool {
        ts >= self.start_ts && ts < self.end_ts
    }
}

/// Manages a set of territory windows for windowed distribution rights.
#[derive(Debug, Default)]
pub struct TerritoryWindowManager {
    windows: Vec<TerritoryWindow>,
}

impl TerritoryWindowManager {
    /// Create an empty manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a window.
    pub fn add_window(&mut self, window: TerritoryWindow) {
        self.windows.push(window);
    }

    /// Check whether content is distributable in a territory at a given time,
    /// considering all registered windows.
    pub fn is_allowed(&self, content_id: &str, territory_code: &str, ts: i64) -> bool {
        self.windows
            .iter()
            .any(|w| w.content_id == content_id && w.covers(territory_code, ts))
    }

    /// Find all active windows for a content ID at a given time.
    pub fn active_windows(&self, content_id: &str, ts: i64) -> Vec<&TerritoryWindow> {
        self.windows
            .iter()
            .filter(|w| w.content_id == content_id && w.is_active(ts))
            .collect()
    }

    /// Detect conflicts: overlapping exclusive windows for the same content
    /// and territory.
    pub fn find_conflicts(&self) -> Vec<(String, String)> {
        let mut conflicts = Vec::new();
        for (i, a) in self.windows.iter().enumerate() {
            if !a.exclusive {
                continue;
            }
            for b in self.windows.iter().skip(i + 1) {
                if !b.exclusive || a.content_id != b.content_id {
                    continue;
                }
                // Check time overlap
                if a.start_ts >= b.end_ts || b.start_ts >= a.end_ts {
                    continue;
                }
                // Check territory overlap
                for t in &a.territories {
                    if b.territories.contains(t) {
                        conflicts.push((a.id.clone(), b.id.clone()));
                        break;
                    }
                }
            }
        }
        conflicts
    }

    /// Total number of windows.
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    /// Whether the manager has no windows.
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_territory_new_stores_fields() {
        let t = Territory::new("US", "United States", "North America");
        assert_eq!(t.code, "US");
        assert_eq!(t.name, "United States");
        assert_eq!(t.region, "North America");
    }

    #[test]
    fn test_north_america_returns_three_territories() {
        let na = Territory::north_america();
        assert_eq!(na.len(), 3);
        let codes: Vec<&str> = na.iter().map(|t| t.code.as_str()).collect();
        assert!(codes.contains(&"US"));
        assert!(codes.contains(&"CA"));
        assert!(codes.contains(&"MX"));
    }

    #[test]
    fn test_europe_returns_territories() {
        let eu = Territory::europe();
        assert!(!eu.is_empty());
        let codes: Vec<&str> = eu.iter().map(|t| t.code.as_str()).collect();
        assert!(codes.contains(&"GB"));
        assert!(codes.contains(&"DE"));
        assert!(codes.contains(&"FR"));
    }

    #[test]
    fn test_worldwide_includes_both_regions() {
        let ww = Territory::worldwide();
        let codes: Vec<&str> = ww.iter().map(|t| t.code.as_str()).collect();
        assert!(codes.contains(&"US"));
        assert!(codes.contains(&"GB"));
        assert!(codes.contains(&"JP"));
        assert!(codes.contains(&"AU"));
    }

    #[test]
    fn test_territory_rights_new_deny_all() {
        let rights = TerritoryRights::new("content-1");
        assert!(!rights.is_allowed("US"));
        assert!(!rights.is_allowed("GB"));
        assert_eq!(rights.allowed_count(), 0);
        assert_eq!(rights.blocked_count(), 0);
    }

    #[test]
    fn test_allow_single_territory() {
        let mut rights = TerritoryRights::new("content-2");
        rights.allow("US");
        assert!(rights.is_allowed("US"));
        assert!(!rights.is_allowed("GB"));
    }

    #[test]
    fn test_allow_is_case_insensitive() {
        let mut rights = TerritoryRights::new("content-3");
        rights.allow("us");
        assert!(rights.is_allowed("US"));
        assert!(rights.is_allowed("us"));
    }

    #[test]
    fn test_block_overrides_allow() {
        let mut rights = TerritoryRights::new("content-4");
        rights.allow("US");
        rights.block("US");
        assert!(!rights.is_allowed("US"));
    }

    #[test]
    fn test_allow_worldwide_grants_access() {
        let mut rights = TerritoryRights::new("content-5");
        rights.allow_worldwide();
        assert!(rights.is_allowed("US"));
        assert!(rights.is_allowed("GB"));
        assert!(rights.is_allowed("JP"));
        assert!(!rights.is_allowed("XX")); // Not in worldwide list
    }

    #[test]
    fn test_not_expired_when_no_expiry_set() {
        let rights = TerritoryRights::new("content-6");
        assert!(!rights.is_expired(9_999_999_999));
    }

    #[test]
    fn test_is_expired_after_timestamp() {
        let mut rights = TerritoryRights::new("content-7");
        rights.set_expires_at(1_000_000);
        assert!(rights.is_expired(1_000_001));
        assert!(rights.is_expired(1_000_000));
        assert!(!rights.is_expired(999_999));
    }

    #[test]
    fn test_allow_does_not_duplicate() {
        let mut rights = TerritoryRights::new("content-8");
        rights.allow("US");
        rights.allow("US");
        assert_eq!(rights.allowed_count(), 1);
    }

    #[test]
    fn test_block_does_not_duplicate() {
        let mut rights = TerritoryRights::new("content-9");
        rights.block("CN");
        rights.block("CN");
        assert_eq!(rights.blocked_count(), 1);
    }

    // ── TerritoryWindow tests ────────────────────────────────────────────

    #[test]
    fn test_territory_window_covers() {
        let mut win = TerritoryWindow::new("w1", "movie-1", "theatrical", 1000, 2000);
        win.add_territory("US");
        win.add_territory("CA");

        assert!(win.covers("US", 1500));
        assert!(win.covers("ca", 1000)); // case-insensitive
        assert!(!win.covers("US", 2000)); // end is exclusive
        assert!(!win.covers("GB", 1500)); // wrong territory
    }

    #[test]
    fn test_territory_window_duration() {
        let win = TerritoryWindow::new("w2", "movie-1", "streaming", 0, 86_400 * 30);
        assert_eq!(win.duration_days(), 30);
    }

    #[test]
    fn test_territory_window_exclusive() {
        let mut win = TerritoryWindow::new("w3", "movie-1", "broadcast", 0, 1000);
        assert!(!win.exclusive);
        win.set_exclusive(true);
        assert!(win.exclusive);
    }

    #[test]
    fn test_window_manager_is_allowed() {
        let mut mgr = TerritoryWindowManager::new();
        let mut w = TerritoryWindow::new("w1", "film-A", "theatrical", 1000, 5000);
        w.add_territory("US");
        mgr.add_window(w);

        assert!(mgr.is_allowed("film-A", "US", 3000));
        assert!(!mgr.is_allowed("film-A", "US", 6000)); // outside window
        assert!(!mgr.is_allowed("film-A", "GB", 3000)); // wrong territory
        assert!(!mgr.is_allowed("film-B", "US", 3000)); // wrong content
    }

    #[test]
    fn test_window_manager_find_conflicts() {
        let mut mgr = TerritoryWindowManager::new();
        let mut w1 = TerritoryWindow::new("w1", "film-A", "theatrical", 1000, 5000);
        w1.add_territory("US");
        w1.set_exclusive(true);

        let mut w2 = TerritoryWindow::new("w2", "film-A", "streaming", 3000, 8000);
        w2.add_territory("US");
        w2.set_exclusive(true);

        mgr.add_window(w1);
        mgr.add_window(w2);

        let conflicts = mgr.find_conflicts();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0], ("w1".to_string(), "w2".to_string()));
    }

    #[test]
    fn test_window_manager_no_conflict_disjoint_time() {
        let mut mgr = TerritoryWindowManager::new();
        let mut w1 = TerritoryWindow::new("w1", "film-A", "theatrical", 1000, 3000);
        w1.add_territory("US");
        w1.set_exclusive(true);

        let mut w2 = TerritoryWindow::new("w2", "film-A", "streaming", 3000, 6000);
        w2.add_territory("US");
        w2.set_exclusive(true);

        mgr.add_window(w1);
        mgr.add_window(w2);

        assert!(mgr.find_conflicts().is_empty());
    }

    #[test]
    fn test_window_manager_no_conflict_different_territory() {
        let mut mgr = TerritoryWindowManager::new();
        let mut w1 = TerritoryWindow::new("w1", "film-A", "theatrical", 1000, 5000);
        w1.add_territory("US");
        w1.set_exclusive(true);

        let mut w2 = TerritoryWindow::new("w2", "film-A", "theatrical", 2000, 6000);
        w2.add_territory("GB");
        w2.set_exclusive(true);

        mgr.add_window(w1);
        mgr.add_window(w2);

        assert!(mgr.find_conflicts().is_empty());
    }
}
