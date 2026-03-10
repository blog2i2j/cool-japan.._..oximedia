//! NDI source routing and switching.
//!
//! This module provides routing tables, switching state management, and a simple
//! video mixer for multi-source NDI production workflows.

#![allow(dead_code)]

/// An NDI source route mapping a source name/IP to an optional alias.
#[derive(Debug, Clone)]
pub struct NdiRoute {
    /// The NDI source name as advertised on the network.
    pub source_name: String,
    /// IPv4 address of the source, as four octets.
    pub source_ip: [u8; 4],
    /// Port number of the source.
    pub source_port: u16,
    /// Optional human-readable alias for easier reference.
    pub alias: Option<String>,
}

impl NdiRoute {
    /// Creates a new `NdiRoute`.
    pub fn new(
        source_name: impl Into<String>,
        source_ip: [u8; 4],
        source_port: u16,
        alias: Option<String>,
    ) -> Self {
        Self {
            source_name: source_name.into(),
            source_ip,
            source_port,
            alias,
        }
    }

    /// Returns a dotted-decimal string representation of the source IP.
    pub fn ip_string(&self) -> String {
        let [a, b, c, d] = self.source_ip;
        format!("{}.{}.{}.{}", a, b, c, d)
    }
}

/// Manages a table of NDI source routes.
///
/// Routes can be added, removed, and looked up by name or alias.
#[derive(Debug, Default)]
pub struct NdiRouter {
    routes: Vec<NdiRoute>,
}

impl NdiRouter {
    /// Creates a new empty `NdiRouter`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a route to the routing table.
    ///
    /// If a route with the same `source_name` already exists, it is replaced.
    pub fn add_route(&mut self, route: NdiRoute) {
        // Remove existing route with same name if present
        self.routes.retain(|r| r.source_name != route.source_name);
        self.routes.push(route);
    }

    /// Removes the route with the given source name.
    pub fn remove_route(&mut self, source_name: &str) {
        self.routes.retain(|r| r.source_name != source_name);
    }

    /// Finds a route by its alias.
    ///
    /// Returns `None` if no route has the specified alias.
    pub fn find_by_alias(&self, alias: &str) -> Option<&NdiRoute> {
        self.routes
            .iter()
            .find(|r| r.alias.as_deref() == Some(alias))
    }

    /// Finds a route by its source name.
    pub fn find_by_name(&self, source_name: &str) -> Option<&NdiRoute> {
        self.routes.iter().find(|r| r.source_name == source_name)
    }

    /// Returns all routes in the routing table.
    pub fn all_routes(&self) -> Vec<&NdiRoute> {
        self.routes.iter().collect()
    }

    /// Returns the number of routes currently in the table.
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }
}

/// Tracks switching state between NDI sources.
///
/// Records current and previous source names and supports undo of the last switch.
#[derive(Debug, Clone)]
pub struct NdiSwitch {
    /// Name of the currently active source.
    pub current: Option<String>,
    /// Name of the previously active source (before the last switch).
    pub previous: Option<String>,
    /// Number of switches performed.
    pub switch_count: u64,
    /// Transition time in milliseconds for smooth switching.
    pub transition_ms: u32,
}

impl NdiSwitch {
    /// Creates a new `NdiSwitch` with no active source.
    pub fn new() -> Self {
        Self {
            current: None,
            previous: None,
            switch_count: 0,
            transition_ms: 0,
        }
    }

    /// Creates a new `NdiSwitch` with a specified transition time.
    pub fn with_transition(transition_ms: u32) -> Self {
        Self {
            current: None,
            previous: None,
            switch_count: 0,
            transition_ms,
        }
    }

    /// Switches to the named source.
    ///
    /// The previous source is saved so `undo()` can revert the switch.
    pub fn switch_to(&mut self, source_name: &str) {
        self.previous = self.current.take();
        self.current = Some(source_name.to_string());
        self.switch_count += 1;
    }

    /// Reverts to the previous source.
    ///
    /// Returns `true` if there was a previous source to revert to, `false` otherwise.
    pub fn undo(&mut self) -> bool {
        if self.previous.is_some() {
            let prev = self.previous.take();
            self.previous = self.current.take();
            self.current = prev;
            self.switch_count += 1;
            true
        } else {
            false
        }
    }

    /// Returns the current source name, if any.
    pub fn current_source(&self) -> Option<&str> {
        self.current.as_deref()
    }

    /// Returns the previous source name, if any.
    pub fn previous_source(&self) -> Option<&str> {
        self.previous.as_deref()
    }
}

impl Default for NdiSwitch {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple NDI video mixer supporting level-based crossfades between sources.
///
/// Each source has an associated mix level (0.0 = silent/invisible, 1.0 = full).
#[derive(Debug, Default)]
pub struct NdiVideoMixer {
    /// List of (source_name, mix_level) pairs.
    pub sources: Vec<(String, f32)>,
}

impl NdiVideoMixer {
    /// Creates a new empty `NdiVideoMixer`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a source with the given mix level.
    ///
    /// If a source with the same name already exists, its level is updated.
    /// Levels are clamped to [0.0, 1.0].
    pub fn add_source(&mut self, name: &str, level: f32) {
        let clamped = level.clamp(0.0, 1.0);
        if let Some(entry) = self.sources.iter_mut().find(|(n, _)| n == name) {
            entry.1 = clamped;
        } else {
            self.sources.push((name.to_string(), clamped));
        }
    }

    /// Sets the mix level for an existing source.
    ///
    /// Does nothing if the source is not found. Level is clamped to [0.0, 1.0].
    pub fn set_level(&mut self, name: &str, level: f32) {
        let clamped = level.clamp(0.0, 1.0);
        if let Some(entry) = self.sources.iter_mut().find(|(n, _)| n == name) {
            entry.1 = clamped;
        }
    }

    /// Normalizes all source levels so they sum to 1.0.
    ///
    /// If the total is zero (all sources are at 0.0), levels are left unchanged
    /// to avoid division by zero.
    pub fn normalize_levels(&mut self) {
        let total: f32 = self.sources.iter().map(|(_, l)| l).sum();
        if total > 0.0 {
            for (_, level) in &mut self.sources {
                *level /= total;
            }
        }
    }

    /// Returns the mix level for the named source, or `None` if not found.
    pub fn get_level(&self, name: &str) -> Option<f32> {
        self.sources
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, l)| *l)
    }

    /// Returns the number of sources in the mixer.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Removes a source from the mixer.
    pub fn remove_source(&mut self, name: &str) {
        self.sources.retain(|(n, _)| n != name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndi_route_ip_string() {
        let route = NdiRoute::new("CAM1", [192, 168, 1, 10], 5960, None);
        assert_eq!(route.ip_string(), "192.168.1.10");
    }

    #[test]
    fn test_router_add_and_find() {
        let mut router = NdiRouter::new();
        let route = NdiRoute::new("CAM1", [10, 0, 0, 1], 5960, Some("main".to_string()));
        router.add_route(route);

        assert!(router.find_by_name("CAM1").is_some());
        assert!(router.find_by_alias("main").is_some());
    }

    #[test]
    fn test_router_remove() {
        let mut router = NdiRouter::new();
        router.add_route(NdiRoute::new("CAM1", [10, 0, 0, 1], 5960, None));
        router.add_route(NdiRoute::new("CAM2", [10, 0, 0, 2], 5960, None));

        router.remove_route("CAM1");
        assert!(router.find_by_name("CAM1").is_none());
        assert!(router.find_by_name("CAM2").is_some());
        assert_eq!(router.route_count(), 1);
    }

    #[test]
    fn test_router_all_routes() {
        let mut router = NdiRouter::new();
        router.add_route(NdiRoute::new("A", [1, 0, 0, 0], 5960, None));
        router.add_route(NdiRoute::new("B", [2, 0, 0, 0], 5960, None));

        assert_eq!(router.all_routes().len(), 2);
    }

    #[test]
    fn test_router_replace_existing() {
        let mut router = NdiRouter::new();
        router.add_route(NdiRoute::new("CAM1", [10, 0, 0, 1], 5960, None));
        router.add_route(NdiRoute::new(
            "CAM1",
            [10, 0, 0, 2],
            5961,
            Some("updated".to_string()),
        ));

        assert_eq!(router.route_count(), 1);
        let route = router
            .find_by_name("CAM1")
            .expect("expected to find by name");
        assert_eq!(route.source_port, 5961);
    }

    #[test]
    fn test_ndi_switch_basic() {
        let mut sw = NdiSwitch::new();
        assert!(sw.current_source().is_none());

        sw.switch_to("CAM1");
        assert_eq!(sw.current_source(), Some("CAM1"));
        assert_eq!(sw.switch_count, 1);
    }

    #[test]
    fn test_ndi_switch_undo() {
        let mut sw = NdiSwitch::new();
        sw.switch_to("CAM1");
        sw.switch_to("CAM2");

        assert!(sw.undo());
        assert_eq!(sw.current_source(), Some("CAM1"));
    }

    #[test]
    fn test_ndi_switch_undo_no_previous() {
        let mut sw = NdiSwitch::new();
        sw.switch_to("CAM1");
        assert!(!sw.undo()); // no previous source
    }

    #[test]
    fn test_ndi_switch_previous_source() {
        let mut sw = NdiSwitch::new();
        sw.switch_to("CAM1");
        sw.switch_to("CAM2");
        assert_eq!(sw.previous_source(), Some("CAM1"));
    }

    #[test]
    fn test_video_mixer_add_source() {
        let mut mixer = NdiVideoMixer::new();
        mixer.add_source("CAM1", 0.7);
        assert_eq!(mixer.get_level("CAM1"), Some(0.7));
    }

    #[test]
    fn test_video_mixer_set_level() {
        let mut mixer = NdiVideoMixer::new();
        mixer.add_source("CAM1", 0.5);
        mixer.set_level("CAM1", 0.9);
        assert!((mixer.get_level("CAM1").expect("expected level to exist") - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_video_mixer_normalize() {
        let mut mixer = NdiVideoMixer::new();
        mixer.add_source("CAM1", 0.5);
        mixer.add_source("CAM2", 0.5);
        mixer.normalize_levels();

        let total: f32 = mixer.sources.iter().map(|(_, l)| l).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_video_mixer_normalize_zero() {
        let mut mixer = NdiVideoMixer::new();
        mixer.add_source("CAM1", 0.0);
        mixer.normalize_levels(); // should not panic
        assert_eq!(mixer.get_level("CAM1"), Some(0.0));
    }
}
