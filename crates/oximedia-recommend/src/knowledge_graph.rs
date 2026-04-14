//! Knowledge graph-based recommendations.
//!
//! Provides a `MediaKnowledgeGraph` with entity nodes (director, genre, actor,
//! studio) and typed edges (directed_by, genre_is, acted_in). Graph traversal
//! finds related content via shared entities, enabling recommendations that
//! go beyond collaborative or content-based signals.

use std::collections::{HashMap, HashSet, VecDeque};

/// Type of entity in the knowledge graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityKind {
    /// A media item (movie, show, video, album, etc.)
    Media,
    /// A director / creator.
    Director,
    /// A genre or category.
    Genre,
    /// An actor / performer.
    Actor,
    /// A production studio or label.
    Studio,
    /// A writer / screenwriter.
    Writer,
    /// A tag or keyword.
    Tag,
}

impl std::fmt::Display for EntityKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Media => write!(f, "Media"),
            Self::Director => write!(f, "Director"),
            Self::Genre => write!(f, "Genre"),
            Self::Actor => write!(f, "Actor"),
            Self::Studio => write!(f, "Studio"),
            Self::Writer => write!(f, "Writer"),
            Self::Tag => write!(f, "Tag"),
        }
    }
}

/// Type of relationship between entities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationKind {
    /// Media → Director.
    DirectedBy,
    /// Media → Genre.
    GenreIs,
    /// Media → Actor.
    ActedIn,
    /// Media → Studio.
    ProducedBy,
    /// Media → Writer.
    WrittenBy,
    /// Media → Tag.
    TaggedWith,
    /// Media → Media (e.g. sequel, remake).
    RelatedTo,
}

impl std::fmt::Display for RelationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DirectedBy => write!(f, "directed_by"),
            Self::GenreIs => write!(f, "genre_is"),
            Self::ActedIn => write!(f, "acted_in"),
            Self::ProducedBy => write!(f, "produced_by"),
            Self::WrittenBy => write!(f, "written_by"),
            Self::TaggedWith => write!(f, "tagged_with"),
            Self::RelatedTo => write!(f, "related_to"),
        }
    }
}

/// A node in the knowledge graph.
#[derive(Debug, Clone)]
pub struct Entity {
    /// Unique identifier for this entity.
    pub id: String,
    /// Kind of entity.
    pub kind: EntityKind,
    /// Human-readable name.
    pub name: String,
    /// Optional properties (arbitrary key-value).
    pub properties: HashMap<String, String>,
}

impl Entity {
    /// Create a new entity.
    pub fn new(id: impl Into<String>, kind: EntityKind, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind,
            name: name.into(),
            properties: HashMap::new(),
        }
    }

    /// Add a property.
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }
}

/// A directed edge in the knowledge graph.
#[derive(Debug, Clone)]
pub struct Edge {
    /// Source entity ID.
    pub from: String,
    /// Target entity ID.
    pub to: String,
    /// Relationship kind.
    pub relation: RelationKind,
    /// Edge weight (default 1.0).
    pub weight: f32,
}

/// A scored recommendation result from graph traversal.
#[derive(Debug, Clone)]
pub struct GraphRecommendation {
    /// Entity ID of the recommended media item.
    pub entity_id: String,
    /// Score based on graph proximity / shared entities.
    pub score: f32,
    /// Explanation path: sequence of (entity_id, relation) hops.
    pub path: Vec<(String, RelationKind)>,
}

/// Configuration for graph-based recommendation queries.
#[derive(Debug, Clone)]
pub struct GraphQueryConfig {
    /// Maximum traversal depth (hops from source).
    pub max_depth: usize,
    /// Maximum number of results to return.
    pub max_results: usize,
    /// Decay factor per hop (score multiplied by this per edge).
    pub decay: f32,
    /// Filter results to only these entity kinds (empty = all).
    pub result_kinds: HashSet<EntityKind>,
}

impl Default for GraphQueryConfig {
    fn default() -> Self {
        let mut result_kinds = HashSet::new();
        result_kinds.insert(EntityKind::Media);
        Self {
            max_depth: 3,
            max_results: 20,
            decay: 0.7,
            result_kinds,
        }
    }
}

/// A media knowledge graph for recommendation.
///
/// Stores entities (media items, directors, genres, actors, studios) and
/// edges (directed_by, genre_is, acted_in, etc.). Supports BFS-based
/// traversal to find related content.
#[derive(Debug)]
pub struct MediaKnowledgeGraph {
    /// All entities by ID.
    entities: HashMap<String, Entity>,
    /// Adjacency list: entity_id → `Vec<Edge>`.
    adjacency: HashMap<String, Vec<Edge>>,
    /// Reverse adjacency: entity_id → `Vec<Edge>` (for traversing backwards).
    reverse_adjacency: HashMap<String, Vec<Edge>>,
    /// Total number of edges.
    edge_count: usize,
}

impl MediaKnowledgeGraph {
    /// Create a new empty knowledge graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            adjacency: HashMap::new(),
            reverse_adjacency: HashMap::new(),
            edge_count: 0,
        }
    }

    /// Add an entity to the graph.
    pub fn add_entity(&mut self, entity: Entity) {
        let id = entity.id.clone();
        self.entities.insert(id.clone(), entity);
        self.adjacency.entry(id.clone()).or_default();
        self.reverse_adjacency.entry(id).or_default();
    }

    /// Add a directed edge.
    pub fn add_edge(&mut self, from: &str, to: &str, relation: RelationKind, weight: f32) {
        let edge = Edge {
            from: from.to_string(),
            to: to.to_string(),
            relation,
            weight,
        };
        self.adjacency
            .entry(from.to_string())
            .or_default()
            .push(edge.clone());
        self.reverse_adjacency
            .entry(to.to_string())
            .or_default()
            .push(edge);
        self.edge_count += 1;
    }

    /// Add a bidirectional edge (adds both forward and reverse).
    pub fn add_bidirectional_edge(
        &mut self,
        a: &str,
        b: &str,
        relation: RelationKind,
        weight: f32,
    ) {
        self.add_edge(a, b, relation, weight);
        self.add_edge(b, a, relation, weight);
    }

    /// Get an entity by ID.
    #[must_use]
    pub fn get_entity(&self, id: &str) -> Option<&Entity> {
        self.entities.get(id)
    }

    /// Get outgoing edges for an entity.
    #[must_use]
    pub fn get_edges(&self, entity_id: &str) -> &[Edge] {
        self.adjacency.get(entity_id).map_or(&[], |v| v.as_slice())
    }

    /// Get incoming edges for an entity.
    #[must_use]
    pub fn get_reverse_edges(&self, entity_id: &str) -> &[Edge] {
        self.reverse_adjacency
            .get(entity_id)
            .map_or(&[], |v| v.as_slice())
    }

    /// Number of entities.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Number of edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    /// Find related content via BFS traversal from a source entity.
    ///
    /// Traverses the graph outward from `source_id`, accumulating scores
    /// with exponential decay per hop. Returns scored recommendations
    /// filtered to the configured entity kinds.
    #[must_use]
    pub fn find_related(
        &self,
        source_id: &str,
        config: &GraphQueryConfig,
    ) -> Vec<GraphRecommendation> {
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(source_id.to_string());

        // BFS queue: (entity_id, current_score, path, depth)
        let mut queue: VecDeque<(String, f32, Vec<(String, RelationKind)>, usize)> =
            VecDeque::new();
        queue.push_back((source_id.to_string(), 1.0, Vec::new(), 0));

        let mut results: HashMap<String, GraphRecommendation> = HashMap::new();

        while let Some((current_id, current_score, path, depth)) = queue.pop_front() {
            if depth >= config.max_depth {
                continue;
            }

            let edges = self.get_edges(&current_id);
            for edge in edges {
                let next_score = current_score * config.decay * edge.weight;
                if next_score < 1e-6 {
                    continue;
                }

                let mut next_path = path.clone();
                next_path.push((edge.to.clone(), edge.relation));

                // Check if this is a result entity
                if let Some(entity) = self.entities.get(&edge.to) {
                    if config.result_kinds.is_empty() || config.result_kinds.contains(&entity.kind)
                    {
                        // Don't include the source itself
                        if edge.to != source_id {
                            let entry = results.entry(edge.to.clone()).or_insert_with(|| {
                                GraphRecommendation {
                                    entity_id: edge.to.clone(),
                                    score: 0.0,
                                    path: Vec::new(),
                                }
                            });
                            // Accumulate score from multiple paths
                            entry.score += next_score;
                            if entry.path.is_empty() {
                                entry.path = next_path.clone();
                            }
                        }
                    }
                }

                // Continue traversal if not visited
                if !visited.contains(&edge.to) {
                    visited.insert(edge.to.clone());
                    queue.push_back((edge.to.clone(), next_score, next_path, depth + 1));
                }
            }
        }

        let mut recs: Vec<GraphRecommendation> = results.into_values().collect();
        recs.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        recs.truncate(config.max_results);
        recs
    }

    /// Find media items that share entities with the given media item.
    ///
    /// This is a convenience method that traverses through shared directors,
    /// actors, genres, studios, etc. and returns related media items.
    #[must_use]
    pub fn find_related_media(
        &self,
        media_id: &str,
        max_results: usize,
    ) -> Vec<GraphRecommendation> {
        let config = GraphQueryConfig {
            max_depth: 2, // media → entity → media
            max_results,
            decay: 0.8,
            ..Default::default()
        };
        self.find_related(media_id, &config)
    }

    /// Get all entities of a specific kind.
    #[must_use]
    pub fn entities_of_kind(&self, kind: EntityKind) -> Vec<&Entity> {
        self.entities.values().filter(|e| e.kind == kind).collect()
    }

    /// Count shared entities between two media items (e.g. shared actors, genres).
    #[must_use]
    pub fn shared_entity_count(&self, media_a: &str, media_b: &str) -> usize {
        let targets_a: HashSet<&str> = self
            .get_edges(media_a)
            .iter()
            .map(|e| e.to.as_str())
            .collect();
        let targets_b: HashSet<&str> = self
            .get_edges(media_b)
            .iter()
            .map(|e| e.to.as_str())
            .collect();
        targets_a.intersection(&targets_b).count()
    }

    /// Remove an entity and all associated edges.
    pub fn remove_entity(&mut self, entity_id: &str) {
        self.entities.remove(entity_id);

        // Remove outgoing edges
        if let Some(edges) = self.adjacency.remove(entity_id) {
            self.edge_count = self.edge_count.saturating_sub(edges.len());
            // Clean reverse adjacency
            for edge in &edges {
                if let Some(rev) = self.reverse_adjacency.get_mut(&edge.to) {
                    rev.retain(|e| e.from != entity_id);
                }
            }
        }

        // Remove incoming edges
        if let Some(edges) = self.reverse_adjacency.remove(entity_id) {
            self.edge_count = self.edge_count.saturating_sub(edges.len());
            for edge in &edges {
                if let Some(fwd) = self.adjacency.get_mut(&edge.from) {
                    fwd.retain(|e| e.to != entity_id);
                }
            }
        }
    }
}

impl Default for MediaKnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_sample_graph() -> MediaKnowledgeGraph {
        let mut g = MediaKnowledgeGraph::new();

        // Media items
        g.add_entity(Entity::new("movie1", EntityKind::Media, "The Matrix"));
        g.add_entity(Entity::new("movie2", EntityKind::Media, "John Wick"));
        g.add_entity(Entity::new("movie3", EntityKind::Media, "Speed"));
        g.add_entity(Entity::new("movie4", EntityKind::Media, "Inception"));
        g.add_entity(Entity::new("movie5", EntityKind::Media, "Constantine"));

        // Entities
        g.add_entity(Entity::new("keanu", EntityKind::Actor, "Keanu Reeves"));
        g.add_entity(Entity::new(
            "nolan",
            EntityKind::Director,
            "Christopher Nolan",
        ));
        g.add_entity(Entity::new(
            "wachowski",
            EntityKind::Director,
            "The Wachowskis",
        ));
        g.add_entity(Entity::new("action", EntityKind::Genre, "Action"));
        g.add_entity(Entity::new("scifi", EntityKind::Genre, "Sci-Fi"));
        g.add_entity(Entity::new("wb", EntityKind::Studio, "Warner Bros"));

        // Edges: movie1 (The Matrix)
        g.add_edge("movie1", "keanu", RelationKind::ActedIn, 1.0);
        g.add_edge("movie1", "wachowski", RelationKind::DirectedBy, 1.0);
        g.add_edge("movie1", "action", RelationKind::GenreIs, 1.0);
        g.add_edge("movie1", "scifi", RelationKind::GenreIs, 1.0);
        g.add_edge("movie1", "wb", RelationKind::ProducedBy, 1.0);

        // movie2 (John Wick) – shares Keanu and Action
        g.add_edge("movie2", "keanu", RelationKind::ActedIn, 1.0);
        g.add_edge("movie2", "action", RelationKind::GenreIs, 1.0);

        // movie3 (Speed) – shares Keanu
        g.add_edge("movie3", "keanu", RelationKind::ActedIn, 1.0);
        g.add_edge("movie3", "action", RelationKind::GenreIs, 1.0);

        // movie4 (Inception) – shares Action and WB
        g.add_edge("movie4", "nolan", RelationKind::DirectedBy, 1.0);
        g.add_edge("movie4", "action", RelationKind::GenreIs, 1.0);
        g.add_edge("movie4", "scifi", RelationKind::GenreIs, 1.0);
        g.add_edge("movie4", "wb", RelationKind::ProducedBy, 1.0);

        // movie5 (Constantine) – shares Keanu
        g.add_edge("movie5", "keanu", RelationKind::ActedIn, 1.0);

        // Reverse edges so we can traverse entity → media
        g.add_edge("keanu", "movie1", RelationKind::ActedIn, 1.0);
        g.add_edge("keanu", "movie2", RelationKind::ActedIn, 1.0);
        g.add_edge("keanu", "movie3", RelationKind::ActedIn, 1.0);
        g.add_edge("keanu", "movie5", RelationKind::ActedIn, 1.0);
        g.add_edge("wachowski", "movie1", RelationKind::DirectedBy, 1.0);
        g.add_edge("action", "movie1", RelationKind::GenreIs, 1.0);
        g.add_edge("action", "movie2", RelationKind::GenreIs, 1.0);
        g.add_edge("action", "movie3", RelationKind::GenreIs, 1.0);
        g.add_edge("action", "movie4", RelationKind::GenreIs, 1.0);
        g.add_edge("scifi", "movie1", RelationKind::GenreIs, 1.0);
        g.add_edge("scifi", "movie4", RelationKind::GenreIs, 1.0);
        g.add_edge("wb", "movie1", RelationKind::ProducedBy, 1.0);
        g.add_edge("wb", "movie4", RelationKind::ProducedBy, 1.0);
        g.add_edge("nolan", "movie4", RelationKind::DirectedBy, 1.0);

        g
    }

    #[test]
    fn test_graph_creation() {
        let g = MediaKnowledgeGraph::new();
        assert_eq!(g.entity_count(), 0);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_add_entities_and_edges() {
        let g = build_sample_graph();
        assert_eq!(g.entity_count(), 11);
        assert!(g.edge_count() > 0);
    }

    #[test]
    fn test_get_entity() {
        let g = build_sample_graph();
        let entity = g.get_entity("keanu");
        assert!(entity.is_some());
        let keanu = entity.expect("entity should exist");
        assert_eq!(keanu.kind, EntityKind::Actor);
        assert_eq!(keanu.name, "Keanu Reeves");
    }

    #[test]
    fn test_get_entity_nonexistent() {
        let g = build_sample_graph();
        assert!(g.get_entity("nonexistent").is_none());
    }

    #[test]
    fn test_get_edges() {
        let g = build_sample_graph();
        let edges = g.get_edges("movie1");
        assert!(!edges.is_empty());
        // movie1 has 5 outgoing edges
        assert_eq!(edges.len(), 5);
    }

    #[test]
    fn test_find_related_from_matrix() {
        let g = build_sample_graph();
        let results = g.find_related_media("movie1", 10);
        assert!(!results.is_empty());

        // John Wick shares Keanu + Action → should be high
        let john_wick = results.iter().find(|r| r.entity_id == "movie2");
        assert!(john_wick.is_some(), "John Wick should be related to Matrix");
        let jw_score = john_wick.expect("should exist").score;
        assert!(jw_score > 0.0);
    }

    #[test]
    fn test_find_related_excludes_source() {
        let g = build_sample_graph();
        let results = g.find_related_media("movie1", 10);
        assert!(
            results.iter().all(|r| r.entity_id != "movie1"),
            "Source should not appear in results"
        );
    }

    #[test]
    fn test_find_related_max_results() {
        let g = build_sample_graph();
        let results = g.find_related_media("movie1", 2);
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_find_related_sorted_by_score() {
        let g = build_sample_graph();
        let results = g.find_related_media("movie1", 10);
        for window in results.windows(2) {
            assert!(window[0].score >= window[1].score);
        }
    }

    #[test]
    fn test_shared_entity_count() {
        let g = build_sample_graph();
        // Matrix and John Wick share keanu + action
        let shared = g.shared_entity_count("movie1", "movie2");
        assert_eq!(shared, 2);
    }

    #[test]
    fn test_shared_entity_count_no_overlap() {
        let g = build_sample_graph();
        // movie4 (Inception) and movie5 (Constantine) share nothing in direct edges
        let shared = g.shared_entity_count("movie4", "movie5");
        assert_eq!(shared, 0);
    }

    #[test]
    fn test_entities_of_kind() {
        let g = build_sample_graph();
        let media = g.entities_of_kind(EntityKind::Media);
        assert_eq!(media.len(), 5);
        let actors = g.entities_of_kind(EntityKind::Actor);
        assert_eq!(actors.len(), 1);
    }

    #[test]
    fn test_entity_with_property() {
        let entity = Entity::new("test", EntityKind::Media, "Test")
            .with_property("year", "2023")
            .with_property("rating", "8.5");
        assert_eq!(entity.properties.len(), 2);
        assert_eq!(
            entity.properties.get("year").map(String::as_str),
            Some("2023")
        );
    }

    #[test]
    fn test_entity_kind_display() {
        assert_eq!(EntityKind::Media.to_string(), "Media");
        assert_eq!(EntityKind::Director.to_string(), "Director");
        assert_eq!(EntityKind::Actor.to_string(), "Actor");
    }

    #[test]
    fn test_relation_kind_display() {
        assert_eq!(RelationKind::DirectedBy.to_string(), "directed_by");
        assert_eq!(RelationKind::ActedIn.to_string(), "acted_in");
        assert_eq!(RelationKind::GenreIs.to_string(), "genre_is");
    }

    #[test]
    fn test_graph_query_config_default() {
        let config = GraphQueryConfig::default();
        assert_eq!(config.max_depth, 3);
        assert_eq!(config.max_results, 20);
        assert!(config.result_kinds.contains(&EntityKind::Media));
    }

    #[test]
    fn test_find_related_custom_config() {
        let g = build_sample_graph();
        let mut result_kinds = HashSet::new();
        result_kinds.insert(EntityKind::Actor);
        let config = GraphQueryConfig {
            max_depth: 1,
            max_results: 10,
            decay: 1.0,
            result_kinds,
        };
        let results = g.find_related("movie1", &config);
        // Should find Keanu Reeves (actor connected to movie1)
        assert!(!results.is_empty());
        let keanu = results.iter().find(|r| r.entity_id == "keanu");
        assert!(keanu.is_some());
    }

    #[test]
    fn test_remove_entity() {
        let mut g = build_sample_graph();
        let initial_edges = g.edge_count();
        g.remove_entity("keanu");
        assert!(g.get_entity("keanu").is_none());
        assert!(g.edge_count() < initial_edges);
    }

    #[test]
    fn test_bidirectional_edge() {
        let mut g = MediaKnowledgeGraph::new();
        g.add_entity(Entity::new("a", EntityKind::Media, "A"));
        g.add_entity(Entity::new("b", EntityKind::Media, "B"));
        g.add_bidirectional_edge("a", "b", RelationKind::RelatedTo, 1.0);
        assert_eq!(g.edge_count(), 2);
        assert_eq!(g.get_edges("a").len(), 1);
        assert_eq!(g.get_edges("b").len(), 1);
    }

    #[test]
    fn test_graph_recommendation_path() {
        let g = build_sample_graph();
        let results = g.find_related_media("movie1", 10);
        for rec in &results {
            assert!(
                !rec.path.is_empty(),
                "Path should not be empty for related items"
            );
        }
    }

    #[test]
    fn test_find_related_empty_graph() {
        let g = MediaKnowledgeGraph::new();
        let results = g.find_related_media("nonexistent", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_decay_reduces_distant_scores() {
        let g = build_sample_graph();
        // depth 2 with decay 0.5 should yield lower scores than depth 2 with decay 0.9
        let config_low = GraphQueryConfig {
            max_depth: 2,
            max_results: 10,
            decay: 0.5,
            ..Default::default()
        };
        let config_high = GraphQueryConfig {
            max_depth: 2,
            max_results: 10,
            decay: 0.9,
            ..Default::default()
        };
        let low = g.find_related("movie1", &config_low);
        let high = g.find_related("movie1", &config_high);
        if !low.is_empty() && !high.is_empty() {
            // The top score with high decay should be >= top score with low decay
            assert!(high[0].score >= low[0].score);
        }
    }
}
