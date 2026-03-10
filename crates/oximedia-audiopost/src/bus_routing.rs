//! Audio bus routing for mixing consoles and post-production sessions.
//!
//! Models bus types, point-to-point routes, and a router that manages
//! all signal paths in a mix session.

#![allow(dead_code)]

/// Classification of a bus in the signal path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BusType {
    /// Main mix output.
    Master,
    /// Auxiliary send/return (effects, monitors).
    Aux,
    /// Subgroup bus for grouped channels.
    Group,
    /// Pre-fader send bus.
    Send,
    /// Return from an external processor.
    Return,
}

impl BusType {
    /// Returns `true` if this bus type represents a return path.
    #[must_use]
    pub fn is_return(self) -> bool {
        matches!(self, BusType::Return)
    }

    /// Returns `true` if this bus can carry the master mix.
    #[must_use]
    pub fn is_master(self) -> bool {
        matches!(self, BusType::Master)
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            BusType::Master => "Master",
            BusType::Aux => "Aux",
            BusType::Group => "Group",
            BusType::Send => "Send",
            BusType::Return => "Return",
        }
    }
}

/// Unique bus identifier (wraps a numeric index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BusId(pub u32);

/// A single routed connection between two buses with gain applied.
#[derive(Debug, Clone)]
pub struct BusRoute {
    /// Source bus identifier.
    pub from: BusId,
    /// Destination bus identifier.
    pub to: BusId,
    /// Gain applied on this route in decibels.
    pub gain_db: f32,
    /// Whether this route is currently active.
    pub enabled: bool,
}

impl BusRoute {
    /// Create a new route with unity gain.
    #[must_use]
    pub fn new(from: BusId, to: BusId) -> Self {
        Self {
            from,
            to,
            gain_db: 0.0,
            enabled: true,
        }
    }

    /// Create a route with a specified gain.
    #[must_use]
    pub fn with_gain(mut self, gain_db: f32) -> Self {
        self.gain_db = gain_db;
        self
    }

    /// Linear gain factor derived from `gain_db`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn gain_linear(&self) -> f32 {
        10.0_f32.powf(self.gain_db / 20.0)
    }

    /// Returns `true` if this route carries signal (enabled and gain is finite).
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.enabled && self.gain_db.is_finite()
    }
}

/// Configuration for a single named bus.
#[derive(Debug, Clone)]
pub struct BusConfig {
    /// Unique ID for this bus.
    pub id: BusId,
    /// Display name.
    pub name: String,
    /// Bus type classification.
    pub bus_type: BusType,
    /// Fader level in dB.
    pub fader_db: f32,
    /// Whether the bus is muted.
    pub muted: bool,
}

impl BusConfig {
    /// Create a new bus configuration.
    #[must_use]
    pub fn new(id: u32, name: impl Into<String>, bus_type: BusType) -> Self {
        Self {
            id: BusId(id),
            name: name.into(),
            bus_type,
            fader_db: 0.0,
            muted: false,
        }
    }

    /// Number of routes originating from this bus (computed externally; parameter provided).
    #[must_use]
    pub fn route_count(routes: &[BusRoute], bus_id: BusId) -> usize {
        routes.iter().filter(|r| r.from == bus_id).count()
    }
}

/// Central manager for all bus routing in a session.
#[derive(Debug, Default)]
pub struct BusRouter {
    buses: Vec<BusConfig>,
    routes: Vec<BusRoute>,
}

impl BusRouter {
    /// Create an empty router.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a bus configuration.
    pub fn add_bus(&mut self, config: BusConfig) {
        self.buses.push(config);
    }

    /// Add a route between two buses.
    pub fn add_route(&mut self, route: BusRoute) {
        self.routes.push(route);
    }

    /// Find the first route from `from` to `to`, if any.
    #[must_use]
    pub fn find_route(&self, from: BusId, to: BusId) -> Option<&BusRoute> {
        self.routes.iter().find(|r| r.from == from && r.to == to)
    }

    /// Total number of registered routes.
    #[must_use]
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }

    /// Total number of registered buses.
    #[must_use]
    pub fn bus_count(&self) -> usize {
        self.buses.len()
    }

    /// Look up a bus config by ID.
    #[must_use]
    pub fn bus(&self, id: BusId) -> Option<&BusConfig> {
        self.buses.iter().find(|b| b.id == id)
    }

    /// All active routes (enabled and carrying signal).
    pub fn active_routes(&self) -> impl Iterator<Item = &BusRoute> {
        self.routes.iter().filter(|r| r.is_active())
    }

    /// Remove a route between two buses, returning it if it existed.
    pub fn remove_route(&mut self, from: BusId, to: BusId) -> Option<BusRoute> {
        if let Some(pos) = self
            .routes
            .iter()
            .position(|r| r.from == from && r.to == to)
        {
            Some(self.routes.remove(pos))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_router() -> BusRouter {
        let mut r = BusRouter::new();
        r.add_bus(BusConfig::new(0, "Master", BusType::Master));
        r.add_bus(BusConfig::new(1, "Aux 1", BusType::Aux));
        r.add_bus(BusConfig::new(2, "Group A", BusType::Group));
        r
    }

    #[test]
    fn test_bus_type_is_return() {
        assert!(BusType::Return.is_return());
        assert!(!BusType::Aux.is_return());
        assert!(!BusType::Master.is_return());
    }

    #[test]
    fn test_bus_type_is_master() {
        assert!(BusType::Master.is_master());
        assert!(!BusType::Group.is_master());
    }

    #[test]
    fn test_bus_type_names() {
        assert_eq!(BusType::Master.name(), "Master");
        assert_eq!(BusType::Aux.name(), "Aux");
        assert_eq!(BusType::Group.name(), "Group");
        assert_eq!(BusType::Send.name(), "Send");
        assert_eq!(BusType::Return.name(), "Return");
    }

    #[test]
    fn test_bus_route_gain_linear_unity() {
        let route = BusRoute::new(BusId(0), BusId(1));
        assert!((route.gain_linear() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_bus_route_gain_linear_minus6db() {
        let route = BusRoute::new(BusId(0), BusId(1)).with_gain(-6.0);
        let lin = route.gain_linear();
        assert!((lin - 0.501).abs() < 0.001, "got {lin}");
    }

    #[test]
    fn test_bus_route_is_active() {
        let mut route = BusRoute::new(BusId(0), BusId(1));
        assert!(route.is_active());
        route.enabled = false;
        assert!(!route.is_active());
    }

    #[test]
    fn test_router_add_route_and_find() {
        let mut router = make_router();
        router.add_route(BusRoute::new(BusId(2), BusId(0)));
        let found = router.find_route(BusId(2), BusId(0));
        assert!(found.is_some());
    }

    #[test]
    fn test_router_find_route_missing() {
        let router = make_router();
        assert!(router.find_route(BusId(99), BusId(0)).is_none());
    }

    #[test]
    fn test_router_route_count() {
        let mut router = make_router();
        assert_eq!(router.route_count(), 0);
        router.add_route(BusRoute::new(BusId(1), BusId(0)));
        router.add_route(BusRoute::new(BusId(2), BusId(0)));
        assert_eq!(router.route_count(), 2);
    }

    #[test]
    fn test_router_bus_count() {
        let router = make_router();
        assert_eq!(router.bus_count(), 3);
    }

    #[test]
    fn test_router_bus_lookup() {
        let router = make_router();
        let bus = router.bus(BusId(1)).expect("operation should succeed");
        assert_eq!(bus.name, "Aux 1");
        assert_eq!(bus.bus_type, BusType::Aux);
    }

    #[test]
    fn test_router_remove_route() {
        let mut router = make_router();
        router.add_route(BusRoute::new(BusId(1), BusId(0)));
        let removed = router.remove_route(BusId(1), BusId(0));
        assert!(removed.is_some());
        assert_eq!(router.route_count(), 0);
    }

    #[test]
    fn test_router_active_routes() {
        let mut router = make_router();
        let mut r1 = BusRoute::new(BusId(1), BusId(0));
        r1.enabled = false;
        router.add_route(r1);
        router.add_route(BusRoute::new(BusId(2), BusId(0)));
        let active: Vec<_> = router.active_routes().collect();
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn test_bus_config_route_count_helper() {
        let routes = vec![
            BusRoute::new(BusId(1), BusId(0)),
            BusRoute::new(BusId(1), BusId(2)),
            BusRoute::new(BusId(2), BusId(0)),
        ];
        assert_eq!(BusConfig::route_count(&routes, BusId(1)), 2);
        assert_eq!(BusConfig::route_count(&routes, BusId(2)), 1);
        assert_eq!(BusConfig::route_count(&routes, BusId(0)), 0);
    }
}
