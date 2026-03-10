#![allow(dead_code)]
//! Media source routing for broadcast automation.
//!
//! Routes media sources (cameras, servers, live feeds) to output destinations
//! (program bus, preview bus, recording outputs) with crosspoint matrix control
//! and salvos (preset routing configurations).

use std::collections::HashMap;

/// Identifier for a routing source.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceId(String);

impl SourceId {
    /// Create a new source ID.
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }

    /// Return the string value.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Identifier for a routing destination.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DestinationId(String);

impl DestinationId {
    /// Create a new destination ID.
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }

    /// Return the string value.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Type of media being routed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaType {
    /// Video signal.
    Video,
    /// Audio signal.
    Audio,
    /// Combined audio/video.
    AudioVideo,
    /// Data/metadata stream.
    Data,
}

/// A routing source.
#[derive(Debug, Clone)]
pub struct RouteSource {
    /// Source identifier.
    pub id: SourceId,
    /// Human-readable name.
    pub name: String,
    /// Media type.
    pub media_type: MediaType,
    /// Whether this source is currently available.
    pub available: bool,
}

impl RouteSource {
    /// Create a new route source.
    pub fn new(id: &str, name: &str, media_type: MediaType) -> Self {
        Self {
            id: SourceId::new(id),
            name: name.to_string(),
            media_type,
            available: true,
        }
    }
}

/// A routing destination.
#[derive(Debug, Clone)]
pub struct RouteDestination {
    /// Destination identifier.
    pub id: DestinationId,
    /// Human-readable name.
    pub name: String,
    /// Accepted media type.
    pub accepted_type: MediaType,
    /// Currently connected source, if any.
    pub connected_source: Option<SourceId>,
}

impl RouteDestination {
    /// Create a new route destination.
    pub fn new(id: &str, name: &str, accepted_type: MediaType) -> Self {
        Self {
            id: DestinationId::new(id),
            name: name.to_string(),
            accepted_type,
            connected_source: None,
        }
    }

    /// Check if a source is currently connected.
    pub fn is_connected(&self) -> bool {
        self.connected_source.is_some()
    }
}

/// A single route connecting a source to a destination.
#[derive(Debug, Clone, PartialEq)]
pub struct Route {
    /// Source ID.
    pub source: SourceId,
    /// Destination ID.
    pub destination: DestinationId,
}

impl Route {
    /// Create a new route.
    pub fn new(source: &str, destination: &str) -> Self {
        Self {
            source: SourceId::new(source),
            destination: DestinationId::new(destination),
        }
    }
}

/// A salvo is a named preset of multiple routes applied atomically.
#[derive(Debug, Clone)]
pub struct Salvo {
    /// Salvo name.
    pub name: String,
    /// Routes in this salvo.
    pub routes: Vec<Route>,
    /// Description.
    pub description: String,
}

impl Salvo {
    /// Create a new empty salvo.
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            routes: Vec::new(),
            description: description.to_string(),
        }
    }

    /// Add a route to this salvo.
    pub fn add_route(&mut self, route: Route) {
        self.routes.push(route);
    }

    /// Return the number of routes in this salvo.
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }
}

/// Error type for media routing operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RoutingError {
    /// Source not found.
    SourceNotFound(String),
    /// Destination not found.
    DestinationNotFound(String),
    /// Media type mismatch between source and destination.
    MediaTypeMismatch {
        /// Source media type.
        source_type: MediaType,
        /// Destination accepted type.
        dest_type: MediaType,
    },
    /// Source is unavailable.
    SourceUnavailable(String),
    /// Salvo not found.
    SalvoNotFound(String),
}

impl std::fmt::Display for RoutingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SourceNotFound(id) => write!(f, "Source not found: {id}"),
            Self::DestinationNotFound(id) => write!(f, "Destination not found: {id}"),
            Self::MediaTypeMismatch { .. } => write!(f, "Media type mismatch"),
            Self::SourceUnavailable(id) => write!(f, "Source unavailable: {id}"),
            Self::SalvoNotFound(name) => write!(f, "Salvo not found: {name}"),
        }
    }
}

/// The media router manages the crosspoint matrix of sources and destinations.
pub struct MediaRouter {
    /// Registered sources.
    sources: HashMap<String, RouteSource>,
    /// Registered destinations.
    destinations: HashMap<String, RouteDestination>,
    /// Named salvos.
    salvos: HashMap<String, Salvo>,
    /// History of route changes (source_id, dest_id, timestamp_ms).
    route_history: Vec<(String, String, u64)>,
}

impl MediaRouter {
    /// Create a new empty media router.
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            destinations: HashMap::new(),
            salvos: HashMap::new(),
            route_history: Vec::new(),
        }
    }

    /// Register a source.
    pub fn add_source(&mut self, source: RouteSource) {
        self.sources.insert(source.id.as_str().to_string(), source);
    }

    /// Register a destination.
    pub fn add_destination(&mut self, dest: RouteDestination) {
        self.destinations.insert(dest.id.as_str().to_string(), dest);
    }

    /// Get the number of registered sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Get the number of registered destinations.
    pub fn destination_count(&self) -> usize {
        self.destinations.len()
    }

    /// Connect a source to a destination.
    pub fn connect(&mut self, source_id: &str, dest_id: &str) -> Result<(), RoutingError> {
        // Validate source exists and is available
        let source = self
            .sources
            .get(source_id)
            .ok_or_else(|| RoutingError::SourceNotFound(source_id.to_string()))?;
        if !source.available {
            return Err(RoutingError::SourceUnavailable(source_id.to_string()));
        }
        let source_type = source.media_type;

        // Validate destination exists
        let dest = self
            .destinations
            .get(dest_id)
            .ok_or_else(|| RoutingError::DestinationNotFound(dest_id.to_string()))?;

        // Check media type compatibility
        if !media_types_compatible(source_type, dest.accepted_type) {
            return Err(RoutingError::MediaTypeMismatch {
                source_type,
                dest_type: dest.accepted_type,
            });
        }

        // Apply route
        if let Some(d) = self.destinations.get_mut(dest_id) {
            d.connected_source = Some(SourceId::new(source_id));
        }
        self.route_history
            .push((source_id.to_string(), dest_id.to_string(), 0));
        Ok(())
    }

    /// Disconnect a destination.
    pub fn disconnect(&mut self, dest_id: &str) -> Result<(), RoutingError> {
        let dest = self
            .destinations
            .get_mut(dest_id)
            .ok_or_else(|| RoutingError::DestinationNotFound(dest_id.to_string()))?;
        dest.connected_source = None;
        Ok(())
    }

    /// Get the source currently connected to a destination.
    pub fn get_connection(&self, dest_id: &str) -> Option<&SourceId> {
        self.destinations
            .get(dest_id)
            .and_then(|d| d.connected_source.as_ref())
    }

    /// Register a salvo.
    pub fn add_salvo(&mut self, salvo: Salvo) {
        self.salvos.insert(salvo.name.clone(), salvo);
    }

    /// Execute a salvo by name.
    pub fn execute_salvo(&mut self, name: &str) -> Result<usize, RoutingError> {
        let salvo = self
            .salvos
            .get(name)
            .ok_or_else(|| RoutingError::SalvoNotFound(name.to_string()))?
            .clone();
        let mut applied = 0;
        for route in &salvo.routes {
            self.connect(route.source.as_str(), route.destination.as_str())?;
            applied += 1;
        }
        Ok(applied)
    }

    /// Return the route change history length.
    pub fn history_len(&self) -> usize {
        self.route_history.len()
    }

    /// List all currently active connections as (source_id, dest_id) pairs.
    pub fn active_connections(&self) -> Vec<(String, String)> {
        self.destinations
            .iter()
            .filter_map(|(dest_id, dest)| {
                dest.connected_source
                    .as_ref()
                    .map(|src| (src.as_str().to_string(), dest_id.clone()))
            })
            .collect()
    }
}

impl Default for MediaRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if two media types are compatible for routing.
fn media_types_compatible(source: MediaType, dest: MediaType) -> bool {
    match (source, dest) {
        (a, b) if a == b => true,
        (MediaType::AudioVideo, MediaType::Video) => true,
        (MediaType::AudioVideo, MediaType::Audio) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_id() {
        let id = SourceId::new("cam1");
        assert_eq!(id.as_str(), "cam1");
    }

    #[test]
    fn test_destination_id() {
        let id = DestinationId::new("pgm");
        assert_eq!(id.as_str(), "pgm");
    }

    #[test]
    fn test_route_source_creation() {
        let src = RouteSource::new("cam1", "Camera 1", MediaType::Video);
        assert_eq!(src.name, "Camera 1");
        assert!(src.available);
    }

    #[test]
    fn test_route_destination_not_connected() {
        let dest = RouteDestination::new("pgm", "Program", MediaType::Video);
        assert!(!dest.is_connected());
    }

    #[test]
    fn test_salvo_creation_and_add() {
        let mut salvo = Salvo::new("preset1", "Default routing");
        salvo.add_route(Route::new("cam1", "pgm"));
        salvo.add_route(Route::new("cam2", "pvw"));
        assert_eq!(salvo.route_count(), 2);
    }

    #[test]
    fn test_media_router_add_source_dest() {
        let mut router = MediaRouter::new();
        router.add_source(RouteSource::new("cam1", "Camera 1", MediaType::Video));
        router.add_destination(RouteDestination::new("pgm", "Program", MediaType::Video));
        assert_eq!(router.source_count(), 1);
        assert_eq!(router.destination_count(), 1);
    }

    #[test]
    fn test_media_router_connect() {
        let mut router = MediaRouter::new();
        router.add_source(RouteSource::new("cam1", "Camera 1", MediaType::Video));
        router.add_destination(RouteDestination::new("pgm", "Program", MediaType::Video));
        let result = router.connect("cam1", "pgm");
        assert!(result.is_ok());
        assert_eq!(
            router
                .get_connection("pgm")
                .expect("get_connection should succeed")
                .as_str(),
            "cam1"
        );
    }

    #[test]
    fn test_media_router_connect_source_not_found() {
        let mut router = MediaRouter::new();
        router.add_destination(RouteDestination::new("pgm", "Program", MediaType::Video));
        let result = router.connect("cam_missing", "pgm");
        assert!(matches!(result, Err(RoutingError::SourceNotFound(_))));
    }

    #[test]
    fn test_media_router_connect_dest_not_found() {
        let mut router = MediaRouter::new();
        router.add_source(RouteSource::new("cam1", "Camera 1", MediaType::Video));
        let result = router.connect("cam1", "missing_dest");
        assert!(matches!(result, Err(RoutingError::DestinationNotFound(_))));
    }

    #[test]
    fn test_media_router_type_mismatch() {
        let mut router = MediaRouter::new();
        router.add_source(RouteSource::new("mic1", "Mic 1", MediaType::Audio));
        router.add_destination(RouteDestination::new("pgm", "Program", MediaType::Video));
        let result = router.connect("mic1", "pgm");
        assert!(matches!(
            result,
            Err(RoutingError::MediaTypeMismatch { .. })
        ));
    }

    #[test]
    fn test_media_router_disconnect() {
        let mut router = MediaRouter::new();
        router.add_source(RouteSource::new("cam1", "Camera 1", MediaType::Video));
        router.add_destination(RouteDestination::new("pgm", "Program", MediaType::Video));
        router
            .connect("cam1", "pgm")
            .expect("connect should succeed");
        router.disconnect("pgm").expect("disconnect should succeed");
        assert!(router.get_connection("pgm").is_none());
    }

    #[test]
    fn test_execute_salvo() {
        let mut router = MediaRouter::new();
        router.add_source(RouteSource::new("cam1", "Camera 1", MediaType::Video));
        router.add_source(RouteSource::new("cam2", "Camera 2", MediaType::Video));
        router.add_destination(RouteDestination::new("pgm", "Program", MediaType::Video));
        router.add_destination(RouteDestination::new("pvw", "Preview", MediaType::Video));

        let mut salvo = Salvo::new("preset1", "Two camera");
        salvo.add_route(Route::new("cam1", "pgm"));
        salvo.add_route(Route::new("cam2", "pvw"));
        router.add_salvo(salvo);

        let applied = router
            .execute_salvo("preset1")
            .expect("execute_salvo should succeed");
        assert_eq!(applied, 2);
        assert_eq!(
            router
                .get_connection("pgm")
                .expect("get_connection should succeed")
                .as_str(),
            "cam1"
        );
        assert_eq!(
            router
                .get_connection("pvw")
                .expect("get_connection should succeed")
                .as_str(),
            "cam2"
        );
    }

    #[test]
    fn test_active_connections() {
        let mut router = MediaRouter::new();
        router.add_source(RouteSource::new("cam1", "Camera 1", MediaType::Video));
        router.add_destination(RouteDestination::new("pgm", "Program", MediaType::Video));
        router.add_destination(RouteDestination::new("pvw", "Preview", MediaType::Video));
        router
            .connect("cam1", "pgm")
            .expect("connect should succeed");
        let conns = router.active_connections();
        assert_eq!(conns.len(), 1);
    }

    #[test]
    fn test_audiovideo_compatible_with_video() {
        assert!(media_types_compatible(
            MediaType::AudioVideo,
            MediaType::Video
        ));
    }

    #[test]
    fn test_audiovideo_compatible_with_audio() {
        assert!(media_types_compatible(
            MediaType::AudioVideo,
            MediaType::Audio
        ));
    }

    #[test]
    fn test_video_incompatible_with_audio() {
        assert!(!media_types_compatible(MediaType::Video, MediaType::Audio));
    }
}
