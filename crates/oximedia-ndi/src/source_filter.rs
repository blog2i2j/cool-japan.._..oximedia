//! NDI source filtering and network scan utilities.
//!
//! Provides mDNS-based discovery simulation, source registry with filtering,
//! and network scan helpers for locating NDI sources by group, name, or IP range.

#![allow(dead_code)]

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::{Duration, Instant};

/// Criteria used to filter discovered NDI sources.
#[derive(Debug, Clone)]
pub struct FilterCriteria {
    /// Optional name substring to match (case-insensitive).
    pub name_contains: Option<String>,
    /// Optional group name to match exactly.
    pub group: Option<String>,
    /// Optional IP address prefix (e.g. "192.168.1.").
    pub ip_prefix: Option<String>,
    /// Maximum age of a source entry before it is considered stale.
    pub max_age: Option<Duration>,
}

impl FilterCriteria {
    /// Create an empty (accept-all) filter.
    pub fn new() -> Self {
        Self {
            name_contains: None,
            group: None,
            ip_prefix: None,
            max_age: None,
        }
    }

    /// Filter by name substring.
    pub fn with_name(mut self, name: &str) -> Self {
        self.name_contains = Some(name.to_lowercase());
        self
    }

    /// Filter by group name.
    pub fn with_group(mut self, group: &str) -> Self {
        self.group = Some(group.to_string());
        self
    }

    /// Filter by IP prefix.
    pub fn with_ip_prefix(mut self, prefix: &str) -> Self {
        self.ip_prefix = Some(prefix.to_string());
        self
    }

    /// Discard entries older than `max_age`.
    pub fn with_max_age(mut self, max_age: Duration) -> Self {
        self.max_age = Some(max_age);
        self
    }
}

impl Default for FilterCriteria {
    fn default() -> Self {
        Self::new()
    }
}

/// A discovered NDI source entry held in the registry.
#[derive(Debug, Clone)]
pub struct SourceEntry {
    /// Unique source name.
    pub name: String,
    /// Network address.
    pub address: SocketAddr,
    /// Groups the source belongs to.
    pub groups: Vec<String>,
    /// When this entry was last seen.
    pub last_seen: Instant,
}

impl SourceEntry {
    /// Create a new source entry with the current timestamp.
    pub fn new(name: &str, address: SocketAddr, groups: Vec<String>) -> Self {
        Self {
            name: name.to_string(),
            address,
            groups,
            last_seen: Instant::now(),
        }
    }

    /// Refresh the last-seen timestamp.
    pub fn refresh(&mut self) {
        self.last_seen = Instant::now();
    }

    /// Check whether this entry matches the given filter criteria.
    pub fn matches(&self, criteria: &FilterCriteria) -> bool {
        if let Some(ref substr) = criteria.name_contains {
            if !self.name.to_lowercase().contains(substr.as_str()) {
                return false;
            }
        }
        if let Some(ref group) = criteria.group {
            if !self.groups.iter().any(|g| g == group) {
                return false;
            }
        }
        if let Some(ref prefix) = criteria.ip_prefix {
            if !self.address.ip().to_string().starts_with(prefix.as_str()) {
                return false;
            }
        }
        if let Some(max_age) = criteria.max_age {
            if self.last_seen.elapsed() > max_age {
                return false;
            }
        }
        true
    }
}

/// In-memory registry of discovered NDI sources with filtering support.
#[derive(Debug)]
pub struct SourceFilterRegistry {
    entries: HashMap<String, SourceEntry>,
}

impl SourceFilterRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Insert or update a source entry.
    pub fn upsert(&mut self, entry: SourceEntry) {
        let key = entry.name.clone();
        self.entries
            .entry(key)
            .and_modify(|e| {
                e.address = entry.address;
                e.groups = entry.groups.clone();
                e.refresh();
            })
            .or_insert(entry);
    }

    /// Remove a source by name. Returns true if it existed.
    pub fn remove(&mut self, name: &str) -> bool {
        self.entries.remove(name).is_some()
    }

    /// Return all entries matching `criteria`.
    pub fn filter(&self, criteria: &FilterCriteria) -> Vec<&SourceEntry> {
        self.entries
            .values()
            .filter(|e| e.matches(criteria))
            .collect()
    }

    /// Return all entries.
    pub fn all(&self) -> Vec<&SourceEntry> {
        self.entries.values().collect()
    }

    /// Remove all entries that are stale (older than `max_age`).
    pub fn evict_stale(&mut self, max_age: Duration) {
        self.entries.retain(|_, e| e.last_seen.elapsed() <= max_age);
    }

    /// Return the number of registered sources.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return true if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for SourceFilterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Simulate a network scan over a /24 IPv4 subnet and return candidate `SocketAddr`s.
///
/// In production this would probe each address; here we return the address list
/// for the caller to use (e.g. in async connection attempts).
pub fn scan_subnet_candidates(base: Ipv4Addr, port: u16) -> Vec<SocketAddr> {
    let octets = base.octets();
    (1u8..=254)
        .map(|last| {
            let addr = Ipv4Addr::new(octets[0], octets[1], octets[2], last);
            SocketAddr::new(IpAddr::V4(addr), port)
        })
        .collect()
}

/// Parse a "hostname_or_ip:port" string into a `SocketAddr`, defaulting to
/// port 5960 (NDI default) if no port is given.
pub fn parse_ndi_address(s: &str) -> Option<SocketAddr> {
    if let Ok(addr) = s.parse::<SocketAddr>() {
        return Some(addr);
    }
    // Try appending the default NDI port
    let with_port = format!("{s}:5960");
    with_port.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_addr(ip: &str) -> SocketAddr {
        format!("{ip}:5960").parse().expect("expected valid parse")
    }

    #[test]
    fn test_filter_criteria_default_accepts_all() {
        let criteria = FilterCriteria::new();
        let entry = SourceEntry::new("Cam1", make_addr("192.168.1.10"), vec!["public".into()]);
        assert!(entry.matches(&criteria));
    }

    #[test]
    fn test_filter_by_name_case_insensitive() {
        let criteria = FilterCriteria::new().with_name("cam");
        let entry = SourceEntry::new("Camera1", make_addr("10.0.0.1"), vec![]);
        assert!(entry.matches(&criteria));
        let entry2 = SourceEntry::new("Switcher", make_addr("10.0.0.2"), vec![]);
        assert!(!entry2.matches(&criteria));
    }

    #[test]
    fn test_filter_by_group() {
        let criteria = FilterCriteria::new().with_group("studio");
        let entry_yes = SourceEntry::new("Cam1", make_addr("10.0.0.1"), vec!["studio".into()]);
        let entry_no = SourceEntry::new("Cam2", make_addr("10.0.0.2"), vec!["public".into()]);
        assert!(entry_yes.matches(&criteria));
        assert!(!entry_no.matches(&criteria));
    }

    #[test]
    fn test_filter_by_ip_prefix() {
        let criteria = FilterCriteria::new().with_ip_prefix("192.168.1.");
        let entry_yes = SourceEntry::new("A", make_addr("192.168.1.50"), vec![]);
        let entry_no = SourceEntry::new("B", make_addr("10.0.0.1"), vec![]);
        assert!(entry_yes.matches(&criteria));
        assert!(!entry_no.matches(&criteria));
    }

    #[test]
    fn test_filter_by_max_age_fresh() {
        let criteria = FilterCriteria::new().with_max_age(Duration::from_secs(60));
        let entry = SourceEntry::new("Fresh", make_addr("10.0.0.1"), vec![]);
        assert!(entry.matches(&criteria));
    }

    #[test]
    fn test_registry_upsert_and_len() {
        let mut reg = SourceFilterRegistry::new();
        assert!(reg.is_empty());
        reg.upsert(SourceEntry::new("Cam1", make_addr("10.0.0.1"), vec![]));
        reg.upsert(SourceEntry::new("Cam2", make_addr("10.0.0.2"), vec![]));
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn test_registry_upsert_updates_existing() {
        let mut reg = SourceFilterRegistry::new();
        reg.upsert(SourceEntry::new(
            "Cam1",
            make_addr("10.0.0.1"),
            vec!["public".into()],
        ));
        reg.upsert(SourceEntry::new(
            "Cam1",
            make_addr("10.0.0.99"),
            vec!["studio".into()],
        ));
        assert_eq!(reg.len(), 1);
        let e = &reg.all()[0];
        assert_eq!(e.address, make_addr("10.0.0.99"));
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = SourceFilterRegistry::new();
        reg.upsert(SourceEntry::new("Cam1", make_addr("10.0.0.1"), vec![]));
        assert!(reg.remove("Cam1"));
        assert!(!reg.remove("Cam1"));
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_filter() {
        let mut reg = SourceFilterRegistry::new();
        reg.upsert(SourceEntry::new(
            "Studio1",
            make_addr("10.0.0.1"),
            vec!["studio".into()],
        ));
        reg.upsert(SourceEntry::new(
            "Remote1",
            make_addr("10.0.1.1"),
            vec!["remote".into()],
        ));
        let criteria = FilterCriteria::new().with_group("studio");
        let results = reg.filter(&criteria);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Studio1");
    }

    #[test]
    fn test_registry_evict_stale_keeps_fresh() {
        let mut reg = SourceFilterRegistry::new();
        reg.upsert(SourceEntry::new("Fresh", make_addr("10.0.0.1"), vec![]));
        reg.evict_stale(Duration::from_secs(60));
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_scan_subnet_candidates_count() {
        let base = "192.168.1.0".parse().expect("expected valid parse");
        let candidates = scan_subnet_candidates(base, 5960);
        assert_eq!(candidates.len(), 254);
        assert_eq!(candidates[0].port(), 5960);
    }

    #[test]
    fn test_scan_subnet_candidates_range() {
        let base = "10.0.0.0".parse().expect("expected valid parse");
        let candidates = scan_subnet_candidates(base, 5960);
        let first_ip = match candidates[0].ip() {
            std::net::IpAddr::V4(v4) => v4.octets()[3],
            _ => panic!("expected IPv4"),
        };
        let last_ip = match candidates[253].ip() {
            std::net::IpAddr::V4(v4) => v4.octets()[3],
            _ => panic!("expected IPv4"),
        };
        assert_eq!(first_ip, 1);
        assert_eq!(last_ip, 254);
    }

    #[test]
    fn test_parse_ndi_address_with_port() {
        let addr = parse_ndi_address("192.168.1.100:5961");
        assert!(addr.is_some());
        assert_eq!(addr.expect("expected addr to be Some/Ok").port(), 5961);
    }

    #[test]
    fn test_parse_ndi_address_without_port() {
        let addr = parse_ndi_address("192.168.1.100");
        assert!(addr.is_some());
        assert_eq!(addr.expect("expected addr to be Some/Ok").port(), 5960);
    }

    #[test]
    fn test_parse_ndi_address_invalid() {
        let addr = parse_ndi_address("not_a_valid_address!!!");
        assert!(addr.is_none());
    }
}
