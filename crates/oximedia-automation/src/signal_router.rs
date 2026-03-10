#![allow(dead_code)]
//! Signal routing matrix management for broadcast automation.
//!
//! Manages a virtual crosspoint routing matrix that maps input signals to
//! output destinations. Supports named sources and destinations, route
//! locking, salvos (preset route configurations), and route history
//! tracking. Integrates with broadcast routers via abstracted commands.

use std::collections::{HashMap, HashSet};
use std::fmt;

/// Unique identifier for a signal source.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceId(pub String);

impl fmt::Display for SourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a signal destination.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DestinationId(pub String);

impl fmt::Display for DestinationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Signal type flowing through the router.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    /// Video signal.
    Video,
    /// Audio signal.
    Audio,
    /// Combined audio-video signal.
    AudioVideo,
    /// Data/metadata signal.
    Data,
}

impl fmt::Display for SignalType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Video => write!(f, "Video"),
            Self::Audio => write!(f, "Audio"),
            Self::AudioVideo => write!(f, "A/V"),
            Self::Data => write!(f, "Data"),
        }
    }
}

/// A single route mapping a source to a destination.
#[derive(Debug, Clone)]
pub struct Route {
    /// Source signal.
    pub source: SourceId,
    /// Destination output.
    pub destination: DestinationId,
    /// Signal type.
    pub signal_type: SignalType,
    /// Whether this route is locked (cannot be changed without unlock).
    pub locked: bool,
    /// Timestamp when route was established (ms since epoch).
    pub established_ms: u64,
    /// Who set this route.
    pub set_by: String,
}

impl fmt::Display for Route {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lock = if self.locked { " [LOCKED]" } else { "" };
        write!(
            f,
            "{} -> {} ({signal_type}){lock}",
            self.source,
            self.destination,
            signal_type = self.signal_type,
        )
    }
}

/// Error type for routing operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteError {
    /// The destination route is locked.
    RouteLocked {
        /// Destination that is locked.
        destination: String,
    },
    /// Source not found.
    SourceNotFound {
        /// The source that was not found.
        source: String,
    },
    /// Destination not found.
    DestinationNotFound {
        /// The destination that was not found.
        destination: String,
    },
    /// Salvo not found.
    SalvoNotFound {
        /// The salvo name that was not found.
        name: String,
    },
}

impl fmt::Display for RouteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RouteLocked { destination } => {
                write!(f, "Route to {destination} is locked")
            }
            Self::SourceNotFound { source } => write!(f, "Source {source} not found"),
            Self::DestinationNotFound { destination } => {
                write!(f, "Destination {destination} not found")
            }
            Self::SalvoNotFound { name } => write!(f, "Salvo '{name}' not found"),
        }
    }
}

/// A saved route preset (salvo).
#[derive(Debug, Clone)]
pub struct Salvo {
    /// Name of this salvo.
    pub name: String,
    /// Description.
    pub description: String,
    /// Routes in this salvo (destination -> source).
    pub routes: HashMap<String, String>,
}

impl Salvo {
    /// Create a new empty salvo.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            routes: HashMap::new(),
        }
    }

    /// Add a route mapping to this salvo.
    pub fn add_route(&mut self, destination: impl Into<String>, source: impl Into<String>) {
        self.routes.insert(destination.into(), source.into());
    }

    /// Number of routes in this salvo.
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }
}

/// Record of a past route change.
#[derive(Debug, Clone)]
pub struct RouteHistoryEntry {
    /// Timestamp of the change (ms since epoch).
    pub timestamp_ms: u64,
    /// Destination that was changed.
    pub destination: String,
    /// Previous source (None if no prior route).
    pub previous_source: Option<String>,
    /// New source.
    pub new_source: String,
    /// Who made the change.
    pub changed_by: String,
}

/// The signal routing matrix.
#[derive(Debug)]
pub struct SignalRouter {
    /// Registered sources.
    sources: HashSet<String>,
    /// Registered destinations.
    destinations: HashSet<String>,
    /// Active routes: destination -> Route.
    active_routes: HashMap<String, Route>,
    /// Saved salvos.
    salvos: HashMap<String, Salvo>,
    /// Route change history.
    history: Vec<RouteHistoryEntry>,
    /// Maximum history entries to retain.
    max_history: usize,
}

impl SignalRouter {
    /// Create a new empty signal router.
    pub fn new() -> Self {
        Self {
            sources: HashSet::new(),
            destinations: HashSet::new(),
            active_routes: HashMap::new(),
            salvos: HashMap::new(),
            history: Vec::new(),
            max_history: 10000,
        }
    }

    /// Register a source.
    pub fn add_source(&mut self, name: impl Into<String>) {
        self.sources.insert(name.into());
    }

    /// Register a destination.
    pub fn add_destination(&mut self, name: impl Into<String>) {
        self.destinations.insert(name.into());
    }

    /// Get all registered source names.
    pub fn sources(&self) -> Vec<&str> {
        self.sources
            .iter()
            .map(std::string::String::as_str)
            .collect()
    }

    /// Get all registered destination names.
    pub fn destinations(&self) -> Vec<&str> {
        self.destinations
            .iter()
            .map(std::string::String::as_str)
            .collect()
    }

    /// Set a route from source to destination.
    pub fn set_route(
        &mut self,
        source: &str,
        destination: &str,
        signal_type: SignalType,
        set_by: &str,
        now_ms: u64,
    ) -> Result<(), RouteError> {
        if !self.sources.contains(source) {
            return Err(RouteError::SourceNotFound {
                source: source.to_string(),
            });
        }
        if !self.destinations.contains(destination) {
            return Err(RouteError::DestinationNotFound {
                destination: destination.to_string(),
            });
        }

        // Check lock
        if let Some(existing) = self.active_routes.get(destination) {
            if existing.locked {
                return Err(RouteError::RouteLocked {
                    destination: destination.to_string(),
                });
            }
        }

        // Record history
        let previous_source = self
            .active_routes
            .get(destination)
            .map(|r| r.source.0.clone());
        self.history.push(RouteHistoryEntry {
            timestamp_ms: now_ms,
            destination: destination.to_string(),
            previous_source,
            new_source: source.to_string(),
            changed_by: set_by.to_string(),
        });
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        let route = Route {
            source: SourceId(source.to_string()),
            destination: DestinationId(destination.to_string()),
            signal_type,
            locked: false,
            established_ms: now_ms,
            set_by: set_by.to_string(),
        };
        self.active_routes.insert(destination.to_string(), route);
        Ok(())
    }

    /// Lock a destination route.
    pub fn lock_route(&mut self, destination: &str) -> bool {
        if let Some(route) = self.active_routes.get_mut(destination) {
            route.locked = true;
            return true;
        }
        false
    }

    /// Unlock a destination route.
    pub fn unlock_route(&mut self, destination: &str) -> bool {
        if let Some(route) = self.active_routes.get_mut(destination) {
            route.locked = false;
            return true;
        }
        false
    }

    /// Get the active route for a destination.
    pub fn get_route(&self, destination: &str) -> Option<&Route> {
        self.active_routes.get(destination)
    }

    /// Get all active routes.
    pub fn active_routes(&self) -> &HashMap<String, Route> {
        &self.active_routes
    }

    /// Clear the route for a destination (if not locked).
    pub fn clear_route(&mut self, destination: &str) -> Result<(), RouteError> {
        if let Some(route) = self.active_routes.get(destination) {
            if route.locked {
                return Err(RouteError::RouteLocked {
                    destination: destination.to_string(),
                });
            }
        }
        self.active_routes.remove(destination);
        Ok(())
    }

    /// Save a salvo.
    pub fn save_salvo(&mut self, salvo: Salvo) {
        self.salvos.insert(salvo.name.clone(), salvo);
    }

    /// Recall (execute) a salvo, applying all its routes.
    pub fn recall_salvo(
        &mut self,
        name: &str,
        set_by: &str,
        now_ms: u64,
    ) -> Result<usize, RouteError> {
        let salvo = self
            .salvos
            .get(name)
            .cloned()
            .ok_or_else(|| RouteError::SalvoNotFound {
                name: name.to_string(),
            })?;

        let mut applied = 0;
        for (dest, src) in &salvo.routes {
            if self
                .set_route(src, dest, SignalType::AudioVideo, set_by, now_ms)
                .is_ok()
            {
                applied += 1;
            }
        }
        Ok(applied)
    }

    /// Get salvo by name.
    pub fn get_salvo(&self, name: &str) -> Option<&Salvo> {
        self.salvos.get(name)
    }

    /// Get route change history.
    pub fn history(&self) -> &[RouteHistoryEntry] {
        &self.history
    }

    /// Number of active routes.
    pub fn active_route_count(&self) -> usize {
        self.active_routes.len()
    }

    /// Number of saved salvos.
    pub fn salvo_count(&self) -> usize {
        self.salvos.len()
    }
}

impl Default for SignalRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_router() -> SignalRouter {
        let mut router = SignalRouter::new();
        router.add_source("CAM1");
        router.add_source("CAM2");
        router.add_source("VTR1");
        router.add_destination("PGM");
        router.add_destination("PVW");
        router.add_destination("REC");
        router
    }

    #[test]
    fn test_source_destination_registration() {
        let router = setup_router();
        assert_eq!(router.sources().len(), 3);
        assert_eq!(router.destinations().len(), 3);
    }

    #[test]
    fn test_set_route_success() {
        let mut router = setup_router();
        let result = router.set_route("CAM1", "PGM", SignalType::Video, "op1", 1000);
        assert!(result.is_ok());
        let route = router.get_route("PGM").expect("get_route should succeed");
        assert_eq!(route.source.0, "CAM1");
        assert_eq!(route.signal_type, SignalType::Video);
    }

    #[test]
    fn test_set_route_source_not_found() {
        let mut router = setup_router();
        let result = router.set_route("NOEXIST", "PGM", SignalType::Video, "op1", 1000);
        assert!(matches!(result, Err(RouteError::SourceNotFound { .. })));
    }

    #[test]
    fn test_set_route_destination_not_found() {
        let mut router = setup_router();
        let result = router.set_route("CAM1", "NOEXIST", SignalType::Video, "op1", 1000);
        assert!(matches!(
            result,
            Err(RouteError::DestinationNotFound { .. })
        ));
    }

    #[test]
    fn test_route_locking() {
        let mut router = setup_router();
        router
            .set_route("CAM1", "PGM", SignalType::Video, "op1", 1000)
            .expect("operation should succeed");
        assert!(router.lock_route("PGM"));
        let result = router.set_route("CAM2", "PGM", SignalType::Video, "op1", 2000);
        assert!(matches!(result, Err(RouteError::RouteLocked { .. })));
    }

    #[test]
    fn test_route_unlocking() {
        let mut router = setup_router();
        router
            .set_route("CAM1", "PGM", SignalType::Video, "op1", 1000)
            .expect("operation should succeed");
        router.lock_route("PGM");
        router.unlock_route("PGM");
        let result = router.set_route("CAM2", "PGM", SignalType::Video, "op1", 2000);
        assert!(result.is_ok());
    }

    #[test]
    fn test_clear_route() {
        let mut router = setup_router();
        router
            .set_route("CAM1", "PGM", SignalType::Video, "op1", 1000)
            .expect("operation should succeed");
        assert!(router.clear_route("PGM").is_ok());
        assert!(router.get_route("PGM").is_none());
    }

    #[test]
    fn test_clear_locked_route_fails() {
        let mut router = setup_router();
        router
            .set_route("CAM1", "PGM", SignalType::Video, "op1", 1000)
            .expect("operation should succeed");
        router.lock_route("PGM");
        assert!(matches!(
            router.clear_route("PGM"),
            Err(RouteError::RouteLocked { .. })
        ));
    }

    #[test]
    fn test_salvo_save_and_recall() {
        let mut router = setup_router();
        let mut salvo = Salvo::new("default_config");
        salvo.add_route("PGM", "CAM1");
        salvo.add_route("PVW", "CAM2");
        assert_eq!(salvo.route_count(), 2);
        router.save_salvo(salvo);
        let applied = router
            .recall_salvo("default_config", "op1", 1000)
            .expect("recall_salvo should succeed");
        assert_eq!(applied, 2);
        assert_eq!(router.active_route_count(), 2);
    }

    #[test]
    fn test_salvo_not_found() {
        let mut router = setup_router();
        let result = router.recall_salvo("nonexistent", "op1", 1000);
        assert!(matches!(result, Err(RouteError::SalvoNotFound { .. })));
    }

    #[test]
    fn test_route_history() {
        let mut router = setup_router();
        router
            .set_route("CAM1", "PGM", SignalType::Video, "op1", 1000)
            .expect("operation should succeed");
        router
            .set_route("CAM2", "PGM", SignalType::Video, "op1", 2000)
            .expect("operation should succeed");
        assert_eq!(router.history().len(), 2);
        assert_eq!(router.history()[1].previous_source.as_deref(), Some("CAM1"));
        assert_eq!(router.history()[1].new_source, "CAM2");
    }

    #[test]
    fn test_signal_type_display() {
        assert_eq!(SignalType::Video.to_string(), "Video");
        assert_eq!(SignalType::Audio.to_string(), "Audio");
        assert_eq!(SignalType::AudioVideo.to_string(), "A/V");
        assert_eq!(SignalType::Data.to_string(), "Data");
    }

    #[test]
    fn test_route_error_display() {
        let err = RouteError::RouteLocked {
            destination: "PGM".to_string(),
        };
        assert!(err.to_string().contains("PGM"));
        let err2 = RouteError::SourceNotFound {
            source: "CAM9".to_string(),
        };
        assert!(err2.to_string().contains("CAM9"));
    }

    #[test]
    fn test_route_display() {
        let route = Route {
            source: SourceId("CAM1".to_string()),
            destination: DestinationId("PGM".to_string()),
            signal_type: SignalType::Video,
            locked: true,
            established_ms: 1000,
            set_by: "op1".to_string(),
        };
        let s = route.to_string();
        assert!(s.contains("CAM1"));
        assert!(s.contains("PGM"));
        assert!(s.contains("LOCKED"));
    }
}
