#![allow(dead_code)]
//! # Playout Media Router
//!
//! Signal routing for playout outputs. Maps named programme sources to one or
//! more delivery targets (SDI, IP multicast, RTMP, file record). Supports
//! hot-swap and priority-based failover.

use std::collections::HashMap;

/// Delivery target for a routed signal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RouteTarget {
    /// Serial Digital Interface output (e.g. `"SDI-1"`).
    Sdi(String),
    /// IP multicast group (e.g. `"239.0.0.1:5000"`).
    IpMulticast(String),
    /// RTMP push destination URL.
    Rtmp(String),
    /// SRT caller/listener URL.
    Srt(String),
    /// Record to a local file path.
    FileRecord(String),
    /// NDI source name.
    Ndi(String),
}

impl RouteTarget {
    /// Return a short human-readable label for the target type.
    pub fn kind(&self) -> &'static str {
        match self {
            RouteTarget::Sdi(_) => "SDI",
            RouteTarget::IpMulticast(_) => "IP-Multicast",
            RouteTarget::Rtmp(_) => "RTMP",
            RouteTarget::Srt(_) => "SRT",
            RouteTarget::FileRecord(_) => "File",
            RouteTarget::Ndi(_) => "NDI",
        }
    }

    /// Return the target address / path string.
    pub fn address(&self) -> &str {
        match self {
            RouteTarget::Sdi(s)
            | RouteTarget::IpMulticast(s)
            | RouteTarget::Rtmp(s)
            | RouteTarget::Srt(s)
            | RouteTarget::FileRecord(s)
            | RouteTarget::Ndi(s) => s.as_str(),
        }
    }
}

/// State of a single playout route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RouteState {
    /// Route is active and passing signal.
    Active,
    /// Route is configured but not passing signal.
    #[default]
    Inactive,
    /// Route is in error state.
    Faulted,
}

/// A named playout route connecting a source to one or more targets.
#[derive(Debug, Clone)]
pub struct PlayoutRoute {
    /// Unique route name.
    pub name: String,
    /// Name of the source programme feed.
    pub source: String,
    /// Delivery targets.
    pub targets: Vec<RouteTarget>,
    /// Current state.
    pub state: RouteState,
    /// Optional backup source used when primary is faulted.
    pub backup_source: Option<String>,
    /// Route priority (higher = preferred during failover).
    pub priority: u8,
}

impl PlayoutRoute {
    /// Create a new inactive route.
    pub fn new(name: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            source: source.into(),
            targets: Vec::new(),
            state: RouteState::Inactive,
            backup_source: None,
            priority: 100,
        }
    }

    /// Add a delivery target to this route.
    pub fn add_target(&mut self, target: RouteTarget) {
        self.targets.push(target);
    }

    /// Activate the route.
    pub fn activate(&mut self) {
        self.state = RouteState::Active;
    }

    /// Deactivate the route.
    pub fn deactivate(&mut self) {
        self.state = RouteState::Inactive;
    }

    /// Mark the route as faulted and switch to backup if available.
    pub fn fault(&mut self) {
        self.state = RouteState::Faulted;
        if let Some(backup) = &self.backup_source.clone() {
            self.source = backup.clone();
            self.state = RouteState::Active;
        }
    }

    /// Return `true` when signal is flowing.
    pub fn is_active(&self) -> bool {
        self.state == RouteState::Active
    }

    /// Number of configured targets.
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }
}

/// Error type for router operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RouterError {
    /// A route with this name already exists.
    DuplicateRoute(String),
    /// No route with this name was found.
    NotFound(String),
    /// Cannot remove an active route without deactivating first.
    RouteIsActive(String),
}

impl std::fmt::Display for RouterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RouterError::DuplicateRoute(n) => write!(f, "Route '{n}' already exists"),
            RouterError::NotFound(n) => write!(f, "Route '{n}' not found"),
            RouterError::RouteIsActive(n) => write!(f, "Route '{n}' is active; deactivate first"),
        }
    }
}

/// Result type for router operations.
pub type RouterResult<T> = Result<T, RouterError>;

/// Central playout signal router.
///
/// Manages a collection of [`PlayoutRoute`] entries. Provides hot-add,
/// hot-remove, activation, and statistics helpers.
pub struct PlayoutRouter {
    routes: HashMap<String, PlayoutRoute>,
}

impl PlayoutRouter {
    /// Create a new router with no routes.
    pub fn new() -> Self {
        Self {
            routes: HashMap::new(),
        }
    }

    /// Register a new route. Fails if a route with the same name already exists.
    pub fn add_route(&mut self, route: PlayoutRoute) -> RouterResult<()> {
        if self.routes.contains_key(&route.name) {
            return Err(RouterError::DuplicateRoute(route.name.clone()));
        }
        self.routes.insert(route.name.clone(), route);
        Ok(())
    }

    /// Remove a route. Fails if the route is currently active.
    pub fn remove_route(&mut self, name: &str) -> RouterResult<PlayoutRoute> {
        let route = self
            .routes
            .get(name)
            .ok_or_else(|| RouterError::NotFound(name.to_string()))?;
        if route.is_active() {
            return Err(RouterError::RouteIsActive(name.to_string()));
        }
        self.routes
            .remove(name)
            .ok_or_else(|| RouterError::NotFound(name.to_string()))
    }

    /// Activate a route by name.
    pub fn activate(&mut self, name: &str) -> RouterResult<()> {
        self.routes
            .get_mut(name)
            .ok_or_else(|| RouterError::NotFound(name.to_string()))
            .map(PlayoutRoute::activate)
    }

    /// Deactivate a route by name.
    pub fn deactivate(&mut self, name: &str) -> RouterResult<()> {
        self.routes
            .get_mut(name)
            .ok_or_else(|| RouterError::NotFound(name.to_string()))
            .map(PlayoutRoute::deactivate)
    }

    /// Trigger a fault on a route, engaging backup if configured.
    pub fn fault(&mut self, name: &str) -> RouterResult<()> {
        self.routes
            .get_mut(name)
            .ok_or_else(|| RouterError::NotFound(name.to_string()))
            .map(PlayoutRoute::fault)
    }

    /// Get a reference to a route.
    pub fn get(&self, name: &str) -> Option<&PlayoutRoute> {
        self.routes.get(name)
    }

    /// Get a mutable reference to a route.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut PlayoutRoute> {
        self.routes.get_mut(name)
    }

    /// Return all route names sorted alphabetically.
    pub fn route_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.routes.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// Return the number of active routes.
    pub fn active_count(&self) -> usize {
        self.routes.values().filter(|r| r.is_active()).count()
    }

    /// Return total number of routes.
    pub fn len(&self) -> usize {
        self.routes.len()
    }

    /// Return `true` if no routes are registered.
    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }

    /// Deactivate all routes.
    pub fn deactivate_all(&mut self) {
        for route in self.routes.values_mut() {
            route.deactivate();
        }
    }
}

impl Default for PlayoutRouter {
    fn default() -> Self {
        Self::new()
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_route(name: &str, source: &str) -> PlayoutRoute {
        PlayoutRoute::new(name, source)
    }

    #[test]
    fn test_route_target_kind() {
        assert_eq!(RouteTarget::Sdi("SDI-1".into()).kind(), "SDI");
        assert_eq!(RouteTarget::Rtmp("rtmp://x".into()).kind(), "RTMP");
        assert_eq!(RouteTarget::Ndi("Studio".into()).kind(), "NDI");
    }

    #[test]
    fn test_route_target_address() {
        let t = RouteTarget::IpMulticast("239.0.0.1:5000".into());
        assert_eq!(t.address(), "239.0.0.1:5000");
    }

    #[test]
    fn test_new_route_is_inactive() {
        let r = make_route("PGM-1", "Live-Feed");
        assert_eq!(r.state, RouteState::Inactive);
        assert!(!r.is_active());
    }

    #[test]
    fn test_activate_route() {
        let mut r = make_route("PGM-1", "Live-Feed");
        r.activate();
        assert!(r.is_active());
    }

    #[test]
    fn test_deactivate_route() {
        let mut r = make_route("PGM-1", "Live-Feed");
        r.activate();
        r.deactivate();
        assert!(!r.is_active());
    }

    #[test]
    fn test_add_target_count() {
        let mut r = make_route("R", "S");
        r.add_target(RouteTarget::Sdi("SDI-1".into()));
        r.add_target(RouteTarget::Rtmp("rtmp://x".into()));
        assert_eq!(r.target_count(), 2);
    }

    #[test]
    fn test_fault_no_backup() {
        let mut r = make_route("R", "S");
        r.activate();
        r.fault();
        assert_eq!(r.state, RouteState::Faulted);
    }

    #[test]
    fn test_fault_with_backup_switches_source() {
        let mut r = make_route("R", "Primary");
        r.backup_source = Some("Backup".into());
        r.activate();
        r.fault();
        assert_eq!(r.source, "Backup");
        assert!(r.is_active());
    }

    #[test]
    fn test_router_add_and_get() {
        let mut router = PlayoutRouter::new();
        router
            .add_route(make_route("PGM", "Feed"))
            .expect("should succeed in test");
        assert!(router.get("PGM").is_some());
    }

    #[test]
    fn test_router_duplicate_rejected() {
        let mut router = PlayoutRouter::new();
        router
            .add_route(make_route("PGM", "F1"))
            .expect("should succeed in test");
        let err = router.add_route(make_route("PGM", "F2")).unwrap_err();
        assert!(matches!(err, RouterError::DuplicateRoute(_)));
    }

    #[test]
    fn test_router_remove_inactive() {
        let mut router = PlayoutRouter::new();
        router
            .add_route(make_route("PGM", "Feed"))
            .expect("should succeed in test");
        let removed = router.remove_route("PGM").expect("should succeed in test");
        assert_eq!(removed.name, "PGM");
        assert!(router.is_empty());
    }

    #[test]
    fn test_router_remove_active_fails() {
        let mut router = PlayoutRouter::new();
        router
            .add_route(make_route("PGM", "Feed"))
            .expect("should succeed in test");
        router.activate("PGM").expect("should succeed in test");
        assert!(matches!(
            router.remove_route("PGM").unwrap_err(),
            RouterError::RouteIsActive(_)
        ));
    }

    #[test]
    fn test_active_count() {
        let mut router = PlayoutRouter::new();
        router
            .add_route(make_route("R1", "S1"))
            .expect("should succeed in test");
        router
            .add_route(make_route("R2", "S2"))
            .expect("should succeed in test");
        router.activate("R1").expect("should succeed in test");
        assert_eq!(router.active_count(), 1);
    }

    #[test]
    fn test_deactivate_all() {
        let mut router = PlayoutRouter::new();
        router
            .add_route(make_route("R1", "S1"))
            .expect("should succeed in test");
        router
            .add_route(make_route("R2", "S2"))
            .expect("should succeed in test");
        router.activate("R1").expect("should succeed in test");
        router.activate("R2").expect("should succeed in test");
        router.deactivate_all();
        assert_eq!(router.active_count(), 0);
    }

    #[test]
    fn test_route_names_sorted() {
        let mut router = PlayoutRouter::new();
        router
            .add_route(make_route("Zulu", "Z"))
            .expect("should succeed in test");
        router
            .add_route(make_route("Alpha", "A"))
            .expect("should succeed in test");
        let names = router.route_names();
        assert_eq!(names[0], "Alpha");
        assert_eq!(names[1], "Zulu");
    }

    #[test]
    fn test_not_found_error() {
        let mut router = PlayoutRouter::new();
        let err = router.activate("ghost").unwrap_err();
        assert!(matches!(err, RouterError::NotFound(_)));
    }
}
