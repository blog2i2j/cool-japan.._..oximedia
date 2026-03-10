//! NDI group management.
//!
//! Groups provide a logical namespace for NDI sources, allowing senders and receivers
//! to filter the set of available sources by group membership.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]

/// A single NDI source with its network location and URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdiSource {
    /// Human-readable source name (e.g. `"Camera 1"`).
    pub name: String,
    /// IP address of the source host.
    pub ip: String,
    /// UDP/TCP port on which the source is listening.
    pub port: u16,
    /// Full URL string (e.g. `"192.168.1.10:5960"`).
    pub url_address: String,
}

impl NdiSource {
    /// Create a new `NdiSource`.  The `url_address` field is derived automatically
    /// from `ip` and `port`.
    pub fn new(name: &str, ip: &str, port: u16) -> Self {
        let url_address = format!("{ip}:{port}");
        Self {
            name: name.to_string(),
            ip: ip.to_string(),
            port,
            url_address,
        }
    }

    /// Return a human-readable address string such as `"192.168.1.10:5960"`.
    pub fn address_string(&self) -> String {
        self.url_address.clone()
    }
}

/// A named collection of [`NdiSource`] entries.
#[derive(Debug, Clone)]
pub struct NdiGroup {
    /// Group name (e.g. `"Studio A"` or `"public"`).
    pub name: String,
    /// Sources belonging to this group.
    pub sources: Vec<NdiSource>,
}

impl NdiGroup {
    /// Create a new, empty group with the given name.
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            sources: Vec::new(),
        }
    }
}

/// Registry that maintains multiple [`NdiGroup`] instances and offers
/// CRUD operations on groups and sources.
#[derive(Debug, Default)]
pub struct NdiGroupManager {
    groups: Vec<NdiGroup>,
}

impl NdiGroupManager {
    /// Create a new empty group manager.
    pub fn new() -> Self {
        Self { groups: Vec::new() }
    }

    /// Create a new group with `name` and return a mutable reference to it.
    /// If a group with that name already exists the existing group is returned.
    pub fn create_group(&mut self, name: &str) -> &mut NdiGroup {
        if let Some(pos) = self.groups.iter().position(|g| g.name == name) {
            return &mut self.groups[pos];
        }
        self.groups.push(NdiGroup::new(name));
        self.groups
            .last_mut()
            .expect("group was just pushed so last_mut is always Some")
    }

    /// Add `source` to the group named `group`.
    ///
    /// Returns `Err` if the group does not exist.
    pub fn add_source(&mut self, group: &str, source: NdiSource) -> Result<(), String> {
        match self.groups.iter_mut().find(|g| g.name == group) {
            Some(g) => {
                g.sources.push(source);
                Ok(())
            }
            None => Err(format!("group '{group}' not found")),
        }
    }

    /// Return references to all sources in the named group.
    ///
    /// Returns an empty slice if the group does not exist.
    pub fn sources_in_group(&self, group: &str) -> Vec<&NdiSource> {
        self.groups
            .iter()
            .find(|g| g.name == group)
            .map(|g| g.sources.iter().collect())
            .unwrap_or_default()
    }

    /// Return references to every source across all groups.
    pub fn all_sources(&self) -> Vec<&NdiSource> {
        self.groups.iter().flat_map(|g| g.sources.iter()).collect()
    }

    /// Return the names of all known groups.
    pub fn groups(&self) -> Vec<&str> {
        self.groups.iter().map(|g| g.name.as_str()).collect()
    }

    /// Remove the source named `source_name` from the group named `group`.
    ///
    /// Returns `true` if the source was found and removed, `false` otherwise.
    pub fn remove_source(&mut self, group: &str, source_name: &str) -> bool {
        match self.groups.iter_mut().find(|g| g.name == group) {
            Some(g) => {
                let before = g.sources.len();
                g.sources.retain(|s| s.name != source_name);
                g.sources.len() < before
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_new_address() {
        let s = NdiSource::new("Cam1", "10.0.0.1", 5960);
        assert_eq!(s.address_string(), "10.0.0.1:5960");
        assert_eq!(s.name, "Cam1");
        assert_eq!(s.ip, "10.0.0.1");
        assert_eq!(s.port, 5960);
    }

    #[test]
    fn test_create_group() {
        let mut mgr = NdiGroupManager::new();
        mgr.create_group("Studio");
        assert_eq!(mgr.groups(), vec!["Studio"]);
    }

    #[test]
    fn test_create_group_idempotent() {
        let mut mgr = NdiGroupManager::new();
        mgr.create_group("A");
        mgr.create_group("A");
        assert_eq!(mgr.groups().len(), 1);
    }

    #[test]
    fn test_add_source_ok() {
        let mut mgr = NdiGroupManager::new();
        mgr.create_group("public");
        let src = NdiSource::new("Cam1", "192.168.1.10", 5960);
        assert!(mgr.add_source("public", src).is_ok());
        assert_eq!(mgr.sources_in_group("public").len(), 1);
    }

    #[test]
    fn test_add_source_missing_group() {
        let mut mgr = NdiGroupManager::new();
        let src = NdiSource::new("Cam1", "192.168.1.10", 5960);
        assert!(mgr.add_source("nonexistent", src).is_err());
    }

    #[test]
    fn test_sources_in_group_empty() {
        let mgr = NdiGroupManager::new();
        assert!(mgr.sources_in_group("x").is_empty());
    }

    #[test]
    fn test_all_sources_across_groups() {
        let mut mgr = NdiGroupManager::new();
        mgr.create_group("A");
        mgr.create_group("B");
        mgr.add_source("A", NdiSource::new("Cam1", "1.1.1.1", 5960))
            .expect("failed to add source Cam1 to group A");
        mgr.add_source("B", NdiSource::new("Cam2", "1.1.1.2", 5960))
            .expect("failed to add source Cam2 to group B");
        assert_eq!(mgr.all_sources().len(), 2);
    }

    #[test]
    fn test_remove_source_ok() {
        let mut mgr = NdiGroupManager::new();
        mgr.create_group("public");
        mgr.add_source("public", NdiSource::new("Cam1", "1.1.1.1", 5960))
            .expect("failed to add source to public group");
        assert!(mgr.remove_source("public", "Cam1"));
        assert!(mgr.sources_in_group("public").is_empty());
    }

    #[test]
    fn test_remove_source_not_found() {
        let mut mgr = NdiGroupManager::new();
        mgr.create_group("public");
        assert!(!mgr.remove_source("public", "NoSuchCam"));
    }

    #[test]
    fn test_remove_source_missing_group() {
        let mut mgr = NdiGroupManager::new();
        assert!(!mgr.remove_source("ghost", "Cam1"));
    }

    #[test]
    fn test_groups_list() {
        let mut mgr = NdiGroupManager::new();
        mgr.create_group("alpha");
        mgr.create_group("beta");
        let names = mgr.groups();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
    }

    #[test]
    fn test_multiple_sources_in_group() {
        let mut mgr = NdiGroupManager::new();
        mgr.create_group("G");
        mgr.add_source("G", NdiSource::new("A", "1.1.1.1", 100))
            .expect("failed to add source A to group G");
        mgr.add_source("G", NdiSource::new("B", "1.1.1.2", 101))
            .expect("failed to add source B to group G");
        mgr.add_source("G", NdiSource::new("C", "1.1.1.3", 102))
            .expect("failed to add source C to group G");
        assert_eq!(mgr.sources_in_group("G").len(), 3);
    }
}
