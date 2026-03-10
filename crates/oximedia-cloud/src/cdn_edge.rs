#![allow(dead_code)]
//! CDN edge node management and selection.

/// Geographic coordinates of an edge location.
#[derive(Debug, Clone, Copy)]
pub struct GeoCoord {
    /// Latitude in degrees.
    pub lat: f64,
    /// Longitude in degrees.
    pub lon: f64,
}

impl GeoCoord {
    /// Creates a new `GeoCoord`.
    pub fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }

    /// Computes an approximate great-circle distance in kilometres to another coordinate
    /// using the Haversine formula.
    #[allow(clippy::cast_precision_loss)]
    pub fn distance_km(&self, other: &GeoCoord) -> f64 {
        const R: f64 = 6371.0;
        let dlat = (other.lat - self.lat).to_radians();
        let dlon = (other.lon - self.lon).to_radians();
        let a = (dlat / 2.0).sin().powi(2)
            + self.lat.to_radians().cos()
                * other.lat.to_radians().cos()
                * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
        R * c
    }
}

/// Represents the physical location of a CDN edge node.
#[derive(Debug, Clone)]
pub struct EdgeLocation {
    /// Human-readable name of the location.
    pub name: String,
    /// ISO 3166-1 alpha-2 country code.
    pub country_code: String,
    /// Geographic position.
    pub coord: GeoCoord,
    /// Reported round-trip latency in milliseconds.
    pub latency_ms: u32,
}

impl EdgeLocation {
    /// Creates a new edge location descriptor.
    pub fn new(
        name: impl Into<String>,
        country_code: impl Into<String>,
        coord: GeoCoord,
        latency_ms: u32,
    ) -> Self {
        Self {
            name: name.into(),
            country_code: country_code.into(),
            coord,
            latency_ms,
        }
    }

    /// Returns a score inversely proportional to latency (higher is better).
    ///
    /// Score formula: `10_000 / (latency_ms + 1)` clamped to `[0, 10_000]`.
    pub fn latency_score(&self) -> u32 {
        10_000 / (self.latency_ms + 1)
    }

    /// Returns true if the location has a latency below the given threshold.
    pub fn is_low_latency(&self, threshold_ms: u32) -> bool {
        self.latency_ms < threshold_ms
    }
}

/// Health status of a CDN edge node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeHealth {
    /// Node is fully operational.
    Healthy,
    /// Node is degraded but still serving.
    Degraded,
    /// Node is offline and not serving traffic.
    Offline,
}

impl NodeHealth {
    /// Returns true if the node is serving traffic (Healthy or Degraded).
    pub fn is_serving(&self) -> bool {
        matches!(self, NodeHealth::Healthy | NodeHealth::Degraded)
    }
}

/// A CDN edge node combining location data with operational status.
#[derive(Debug, Clone)]
pub struct CdnEdgeNode {
    /// Unique node identifier.
    pub id: String,
    /// Physical location of the node.
    pub location: EdgeLocation,
    /// Current health status.
    pub health: NodeHealth,
    /// Current load as a percentage in `[0, 100]`.
    pub load_pct: u8,
    /// Maximum requests per second this node can handle.
    pub capacity_rps: u32,
}

impl CdnEdgeNode {
    /// Creates a new edge node.
    pub fn new(id: impl Into<String>, location: EdgeLocation, capacity_rps: u32) -> Self {
        Self {
            id: id.into(),
            location,
            health: NodeHealth::Healthy,
            load_pct: 0,
            capacity_rps,
        }
    }

    /// Returns true if the node is healthy and not overloaded (load < 90%).
    pub fn is_healthy(&self) -> bool {
        self.health == NodeHealth::Healthy && self.load_pct < 90
    }

    /// Returns the remaining capacity as a fraction of total capacity.
    #[allow(clippy::cast_precision_loss)]
    pub fn available_capacity_fraction(&self) -> f64 {
        1.0 - (self.load_pct as f64 / 100.0)
    }

    /// Computes a combined score (higher is better) balancing latency and load.
    pub fn selection_score(&self) -> u32 {
        if !self.health.is_serving() {
            return 0;
        }
        let latency_score = self.location.latency_score();
        let load_penalty = u32::from(self.load_pct);
        latency_score.saturating_sub(load_penalty)
    }
}

/// Selects the best edge node for a given client.
#[derive(Debug, Default)]
pub struct EdgeSelector {
    nodes: Vec<CdnEdgeNode>,
}

impl EdgeSelector {
    /// Creates an empty `EdgeSelector`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a node to the selector pool.
    pub fn add_node(&mut self, node: CdnEdgeNode) {
        self.nodes.push(node);
    }

    /// Returns the number of nodes in the pool.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of healthy (is_healthy == true) nodes.
    pub fn healthy_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_healthy()).count()
    }

    /// Selects the best node for the given client location.
    ///
    /// Returns `None` if there are no healthy nodes.
    pub fn select_best(&self, _client: &GeoCoord) -> Option<&CdnEdgeNode> {
        self.nodes
            .iter()
            .filter(|n| n.is_healthy())
            .max_by_key(|n| n.selection_score())
    }

    /// Removes a node by ID. Returns true if the node was found and removed.
    pub fn remove_node(&mut self, id: &str) -> bool {
        let before = self.nodes.len();
        self.nodes.retain(|n| n.id != id);
        self.nodes.len() < before
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_location(latency_ms: u32) -> EdgeLocation {
        EdgeLocation::new("Test PoP", "US", GeoCoord::new(40.0, -74.0), latency_ms)
    }

    fn make_node(id: &str, latency_ms: u32, load: u8) -> CdnEdgeNode {
        let mut node = CdnEdgeNode::new(id, make_location(latency_ms), 10_000);
        node.load_pct = load;
        node
    }

    #[test]
    fn test_latency_score_low_latency() {
        let loc = make_location(1);
        // 10_000 / 2 = 5000
        assert_eq!(loc.latency_score(), 5000);
    }

    #[test]
    fn test_latency_score_high_latency() {
        let loc = make_location(9999);
        // 10_000 / 10_000 = 1
        assert_eq!(loc.latency_score(), 1);
    }

    #[test]
    fn test_is_low_latency() {
        let loc = make_location(50);
        assert!(loc.is_low_latency(100));
        assert!(!loc.is_low_latency(50));
    }

    #[test]
    fn test_node_is_healthy() {
        let node = make_node("n1", 10, 50);
        assert!(node.is_healthy());
    }

    #[test]
    fn test_node_not_healthy_overloaded() {
        let node = make_node("n1", 10, 95);
        assert!(!node.is_healthy());
    }

    #[test]
    fn test_node_not_healthy_offline() {
        let mut node = make_node("n1", 10, 20);
        node.health = NodeHealth::Offline;
        assert!(!node.is_healthy());
    }

    #[test]
    fn test_node_health_is_serving() {
        assert!(NodeHealth::Healthy.is_serving());
        assert!(NodeHealth::Degraded.is_serving());
        assert!(!NodeHealth::Offline.is_serving());
    }

    #[test]
    fn test_selection_score_offline_zero() {
        let mut node = make_node("n1", 1, 0);
        node.health = NodeHealth::Offline;
        assert_eq!(node.selection_score(), 0);
    }

    #[test]
    fn test_selector_add_node() {
        let mut sel = EdgeSelector::new();
        sel.add_node(make_node("n1", 20, 10));
        assert_eq!(sel.node_count(), 1);
    }

    #[test]
    fn test_selector_healthy_count() {
        let mut sel = EdgeSelector::new();
        sel.add_node(make_node("n1", 20, 10));
        let mut n2 = make_node("n2", 30, 95);
        n2.health = NodeHealth::Offline;
        sel.add_node(n2);
        assert_eq!(sel.healthy_count(), 1);
    }

    #[test]
    fn test_select_best_returns_lowest_latency() {
        let mut sel = EdgeSelector::new();
        sel.add_node(make_node("slow", 200, 10));
        sel.add_node(make_node("fast", 5, 10));
        let client = GeoCoord::new(40.0, -74.0);
        let best = sel.select_best(&client).expect("best should be valid");
        assert_eq!(best.id, "fast");
    }

    #[test]
    fn test_select_best_no_healthy_nodes() {
        let mut sel = EdgeSelector::new();
        let mut n = make_node("n1", 10, 95);
        n.health = NodeHealth::Offline;
        sel.add_node(n);
        let client = GeoCoord::new(0.0, 0.0);
        assert!(sel.select_best(&client).is_none());
    }

    #[test]
    fn test_remove_node() {
        let mut sel = EdgeSelector::new();
        sel.add_node(make_node("n1", 10, 10));
        sel.add_node(make_node("n2", 20, 10));
        let removed = sel.remove_node("n1");
        assert!(removed);
        assert_eq!(sel.node_count(), 1);
    }

    #[test]
    fn test_remove_nonexistent_node() {
        let mut sel = EdgeSelector::new();
        assert!(!sel.remove_node("ghost"));
    }

    #[test]
    fn test_geo_distance_same_point() {
        let a = GeoCoord::new(51.5, -0.1);
        assert!(a.distance_km(&a) < 0.001);
    }

    #[test]
    fn test_available_capacity_fraction() {
        let node = make_node("n1", 10, 40);
        let frac = node.available_capacity_fraction();
        assert!((frac - 0.6).abs() < 1e-9);
    }
}
