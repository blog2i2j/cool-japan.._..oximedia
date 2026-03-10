//! Filter graph builder and execution.
//!
//! The filter graph connects nodes together to form a processing pipeline.
//! Use [`GraphBuilder`] to construct graphs with compile-time safety.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeRuntime, NodeState, NodeType};
use crate::port::{Connection, PortId};

/// A filter graph that processes media through connected nodes.
#[allow(dead_code)]
pub struct FilterGraph {
    /// Nodes in the graph indexed by ID.
    nodes: HashMap<NodeId, NodeRuntime>,
    /// Connections between nodes.
    connections: Vec<Connection>,
    /// Topologically sorted node order for execution.
    execution_order: Vec<NodeId>,
    /// Source nodes (entry points).
    source_nodes: Vec<NodeId>,
    /// Sink nodes (exit points).
    sink_nodes: Vec<NodeId>,
    /// Next available node ID.
    next_id: u64,
}

impl FilterGraph {
    /// Create a new empty filter graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            connections: Vec::new(),
            execution_order: Vec::new(),
            source_nodes: Vec::new(),
            sink_nodes: Vec::new(),
            next_id: 0,
        }
    }

    /// Create a new graph builder.
    #[must_use]
    pub fn builder() -> GraphBuilder<Empty> {
        GraphBuilder::new()
    }

    /// Get a node by ID.
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<&dyn Node> {
        self.nodes.get(&id).map(|r| r.node())
    }

    /// Get a mutable node by ID.
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut dyn Node> {
        self.nodes.get_mut(&id).map(|r| r.node_mut())
    }

    /// Get all node IDs.
    #[must_use]
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.nodes.keys().copied().collect()
    }

    /// Get the execution order.
    #[must_use]
    pub fn execution_order(&self) -> &[NodeId] {
        &self.execution_order
    }

    /// Get source nodes.
    #[must_use]
    pub fn source_nodes(&self) -> &[NodeId] {
        &self.source_nodes
    }

    /// Get sink nodes.
    #[must_use]
    pub fn sink_nodes(&self) -> &[NodeId] {
        &self.sink_nodes
    }

    /// Get connections.
    #[must_use]
    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    /// Check if the graph is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get the number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Initialize all nodes for processing.
    pub fn initialize(&mut self) -> GraphResult<()> {
        for id in &self.execution_order.clone() {
            if let Some(runtime) = self.nodes.get_mut(id) {
                runtime.node_mut().initialize()?;
            }
        }
        Ok(())
    }

    /// Process one step of the graph.
    ///
    /// Processes nodes in topological order, passing frames through connections.
    pub fn process_step(&mut self) -> GraphResult<bool> {
        let mut processed_any = false;

        for id in self.execution_order.clone() {
            let runtime = self
                .nodes
                .get_mut(&id)
                .ok_or(GraphError::NodeNotFound(id))?;

            // Skip nodes that are done
            if runtime.node().state() == NodeState::Done {
                continue;
            }

            // Process the node
            runtime.node_mut().set_state(NodeState::Processing)?;
            runtime.process()?;
            runtime.node_mut().set_state(NodeState::Idle)?;
            processed_any = true;

            // Transfer outputs to connected inputs
            for conn in &self.connections.clone() {
                if conn.from_node == id {
                    // Get output from source
                    let frame = {
                        let source = self
                            .nodes
                            .get_mut(&conn.from_node)
                            .ok_or(GraphError::NodeNotFound(conn.from_node))?;
                        source.pop_output(conn.from_port)?
                    };

                    // Push to destination if we have a frame
                    if let Some(frame) = frame {
                        let dest = self
                            .nodes
                            .get_mut(&conn.to_node)
                            .ok_or(GraphError::NodeNotFound(conn.to_node))?;
                        dest.push_input(conn.to_port, frame)?;
                    }
                }
            }
        }

        Ok(processed_any)
    }

    /// Push a frame to a source node.
    pub fn push_frame(
        &mut self,
        node_id: NodeId,
        port: PortId,
        frame: FilterFrame,
    ) -> GraphResult<()> {
        let runtime = self
            .nodes
            .get_mut(&node_id)
            .ok_or(GraphError::NodeNotFound(node_id))?;
        runtime.push_input(port, frame)
    }

    /// Pull a frame from a sink node.
    pub fn pull_frame(
        &mut self,
        node_id: NodeId,
        port: PortId,
    ) -> GraphResult<Option<FilterFrame>> {
        let runtime = self
            .nodes
            .get_mut(&node_id)
            .ok_or(GraphError::NodeNotFound(node_id))?;
        runtime.pop_output(port)
    }

    /// Reset all nodes to initial state.
    pub fn reset(&mut self) -> GraphResult<()> {
        for runtime in self.nodes.values_mut() {
            runtime.node_mut().reset()?;
        }
        Ok(())
    }

    /// Flush all nodes.
    pub fn flush(&mut self) -> GraphResult<Vec<FilterFrame>> {
        let mut frames = Vec::new();

        for id in &self.execution_order.clone() {
            if let Some(runtime) = self.nodes.get_mut(id) {
                let flushed = runtime.node_mut().flush()?;
                frames.extend(flushed);
            }
        }

        Ok(frames)
    }

    /// Add a node to the graph (internal).
    fn add_node_internal(&mut self, node: Box<dyn Node>) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;

        // Classify node type
        match node.node_type() {
            NodeType::Source => self.source_nodes.push(id),
            NodeType::Sink => self.sink_nodes.push(id),
            NodeType::Filter => {}
        }

        self.nodes.insert(id, NodeRuntime::new(node));
        id
    }

    /// Add a connection between nodes (internal).
    fn add_connection_internal(&mut self, connection: Connection) -> GraphResult<()> {
        // Verify nodes exist
        if !self.nodes.contains_key(&connection.from_node) {
            return Err(GraphError::NodeNotFound(connection.from_node));
        }
        if !self.nodes.contains_key(&connection.to_node) {
            return Err(GraphError::NodeNotFound(connection.to_node));
        }

        // Check for duplicate connections
        if self.connections.contains(&connection) {
            return Err(GraphError::ConnectionExists {
                from_node: connection.from_node,
                from_port: connection.from_port,
                to_node: connection.to_node,
                to_port: connection.to_port,
            });
        }

        // Verify ports exist and formats are compatible
        {
            let from_node = self
                .nodes
                .get(&connection.from_node)
                .ok_or(GraphError::NodeNotFound(connection.from_node))?;
            let to_node = self
                .nodes
                .get(&connection.to_node)
                .ok_or(GraphError::NodeNotFound(connection.to_node))?;

            let from_port = from_node.node().output_port(connection.from_port).ok_or(
                GraphError::PortNotFound {
                    node: connection.from_node,
                    port: connection.from_port,
                },
            )?;

            let to_port =
                to_node
                    .node()
                    .input_port(connection.to_port)
                    .ok_or(GraphError::PortNotFound {
                        node: connection.to_node,
                        port: connection.to_port,
                    })?;

            // Check port type compatibility
            if from_port.port_type != to_port.port_type {
                return Err(GraphError::PortTypeMismatch {
                    expected: format!("{:?}", to_port.port_type),
                    actual: format!("{:?}", from_port.port_type),
                });
            }

            // Check format compatibility
            if !from_port.format.is_compatible(&to_port.format) {
                return Err(GraphError::IncompatibleFormats {
                    source_format: format!("{}", from_port.format),
                    dest_format: format!("{}", to_port.format),
                });
            }
        }

        self.connections.push(connection);
        Ok(())
    }

    /// Compute topological sort for execution order.
    fn compute_execution_order(&mut self) -> GraphResult<()> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        // Initialize
        for &id in self.nodes.keys() {
            in_degree.insert(id, 0);
            adjacency.insert(id, Vec::new());
        }

        // Build adjacency list and in-degrees
        for conn in &self.connections {
            adjacency
                .get_mut(&conn.from_node)
                .ok_or(GraphError::NodeNotFound(conn.from_node))?
                .push(conn.to_node);
            *in_degree
                .get_mut(&conn.to_node)
                .ok_or(GraphError::NodeNotFound(conn.to_node))? += 1;
        }

        // Kahn's algorithm
        let mut queue: VecDeque<NodeId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut order = Vec::new();

        while let Some(id) = queue.pop_front() {
            order.push(id);

            let neighbors: Vec<NodeId> = adjacency
                .get(&id)
                .ok_or(GraphError::NodeNotFound(id))?
                .clone();
            for neighbor in neighbors {
                let deg = in_degree
                    .get_mut(&neighbor)
                    .ok_or(GraphError::NodeNotFound(neighbor))?;
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(neighbor);
                }
            }
        }

        // Check for cycle
        if order.len() != self.nodes.len() {
            // Find a node that's part of the cycle
            let cycle_node = in_degree
                .iter()
                .find(|(_, &deg)| deg > 0)
                .map_or(NodeId(0), |(&id, _)| id);
            return Err(GraphError::CycleDetected(cycle_node));
        }

        self.execution_order = order;
        Ok(())
    }

    /// Validate the graph configuration.
    fn validate(&self) -> GraphResult<()> {
        if self.nodes.is_empty() {
            return Err(GraphError::EmptyGraph);
        }

        if self.source_nodes.is_empty() {
            return Err(GraphError::NoSourceNodes);
        }

        if self.sink_nodes.is_empty() {
            return Err(GraphError::NoSinkNodes);
        }

        // Check all required inputs are connected
        for (id, runtime) in &self.nodes {
            for input in runtime.node().inputs() {
                if input.required {
                    let connected = self
                        .connections
                        .iter()
                        .any(|c| c.to_node == *id && c.to_port == input.id);
                    if !connected && runtime.node().node_type() != NodeType::Source {
                        return Err(GraphError::ConfigurationError(format!(
                            "Required input '{}' on node {:?} is not connected",
                            input.name, id
                        )));
                    }
                }
            }
        }

        Ok(())
    }
}

impl Default for FilterGraph {
    fn default() -> Self {
        Self::new()
    }
}

// Type-state markers for the builder
/// Empty graph state.
pub struct Empty;
/// Graph has at least one node.
pub struct HasNodes;
/// Graph has connections.
pub struct HasConnections;
/// Graph is ready to build.
pub struct Ready;

/// Builder for constructing filter graphs with type-state pattern.
///
/// The builder ensures that graphs are constructed correctly:
/// 1. Add nodes
/// 2. Connect nodes
/// 3. Build the graph
pub struct GraphBuilder<State> {
    graph: FilterGraph,
    _state: std::marker::PhantomData<State>,
}

impl GraphBuilder<Empty> {
    /// Create a new graph builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: FilterGraph::new(),
            _state: std::marker::PhantomData,
        }
    }

    /// Add the first node to the graph.
    pub fn add_node(mut self, node: Box<dyn Node>) -> (GraphBuilder<HasNodes>, NodeId) {
        let id = self.graph.add_node_internal(node);
        (
            GraphBuilder {
                graph: self.graph,
                _state: std::marker::PhantomData,
            },
            id,
        )
    }
}

impl Default for GraphBuilder<Empty> {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphBuilder<HasNodes> {
    /// Add another node to the graph.
    pub fn add_node(mut self, node: Box<dyn Node>) -> (Self, NodeId) {
        let id = self.graph.add_node_internal(node);
        (self, id)
    }

    /// Connect two nodes.
    pub fn connect(
        mut self,
        from_node: NodeId,
        from_port: PortId,
        to_node: NodeId,
        to_port: PortId,
    ) -> GraphResult<GraphBuilder<HasConnections>> {
        let connection = Connection::new(from_node, from_port, to_node, to_port);
        self.graph.add_connection_internal(connection)?;
        Ok(GraphBuilder {
            graph: self.graph,
            _state: std::marker::PhantomData,
        })
    }

    /// Build the graph without any connections (single node graph).
    pub fn build(mut self) -> GraphResult<FilterGraph> {
        self.graph.validate()?;
        self.graph.compute_execution_order()?;
        Ok(self.graph)
    }
}

impl GraphBuilder<HasConnections> {
    /// Add another node to the graph.
    pub fn add_node(mut self, node: Box<dyn Node>) -> (Self, NodeId) {
        let id = self.graph.add_node_internal(node);
        (self, id)
    }

    /// Add another connection.
    pub fn connect(
        mut self,
        from_node: NodeId,
        from_port: PortId,
        to_node: NodeId,
        to_port: PortId,
    ) -> GraphResult<Self> {
        let connection = Connection::new(from_node, from_port, to_node, to_port);
        self.graph.add_connection_internal(connection)?;
        Ok(self)
    }

    /// Build the filter graph.
    pub fn build(mut self) -> GraphResult<FilterGraph> {
        self.graph.validate()?;
        self.graph.compute_execution_order()?;
        Ok(self.graph)
    }
}

/// Find all paths between two nodes in the graph.
#[allow(dead_code)]
fn find_paths(graph: &FilterGraph, from: NodeId, to: NodeId) -> Vec<Vec<NodeId>> {
    let mut paths = Vec::new();
    let mut current_path = vec![from];
    let mut visited = HashSet::new();

    find_paths_recursive(graph, from, to, &mut current_path, &mut visited, &mut paths);
    paths
}

fn find_paths_recursive(
    graph: &FilterGraph,
    current: NodeId,
    target: NodeId,
    path: &mut Vec<NodeId>,
    visited: &mut HashSet<NodeId>,
    paths: &mut Vec<Vec<NodeId>>,
) {
    if current == target {
        paths.push(path.clone());
        return;
    }

    visited.insert(current);

    for conn in graph.connections() {
        if conn.from_node == current && !visited.contains(&conn.to_node) {
            path.push(conn.to_node);
            find_paths_recursive(graph, conn.to_node, target, path, visited, paths);
            path.pop();
        }
    }

    visited.remove(&current);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::video::{NullSink, PassthroughFilter};

    #[test]
    fn test_graph_builder() {
        let source = PassthroughFilter::new_source(NodeId(0), "source");
        let sink = NullSink::new(NodeId(0), "sink");

        let (builder, source_id) = GraphBuilder::new().add_node(Box::new(source));
        let (builder, sink_id) = builder.add_node(Box::new(sink));

        let graph = builder
            .connect(source_id, PortId(0), sink_id, PortId(0))
            .expect("operation should succeed")
            .build()
            .expect("operation should succeed");

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.source_nodes().len(), 1);
        assert_eq!(graph.sink_nodes().len(), 1);
    }

    #[test]
    fn test_execution_order() {
        let source = PassthroughFilter::new_source(NodeId(0), "source");
        let filter = PassthroughFilter::new(NodeId(0), "filter");
        let sink = NullSink::new(NodeId(0), "sink");

        let (builder, source_id) = GraphBuilder::new().add_node(Box::new(source));
        let (builder, filter_id) = builder.add_node(Box::new(filter));
        let (builder, sink_id) = builder.add_node(Box::new(sink));

        let graph = builder
            .connect(source_id, PortId(0), filter_id, PortId(0))
            .expect("operation should succeed")
            .connect(filter_id, PortId(0), sink_id, PortId(0))
            .expect("operation should succeed")
            .build()
            .expect("operation should succeed");

        let order = graph.execution_order();
        assert_eq!(order.len(), 3);

        // Source should come before filter, filter before sink
        let source_pos = order
            .iter()
            .position(|&id| id == source_id)
            .expect("iter should succeed");
        let filter_pos = order
            .iter()
            .position(|&id| id == filter_id)
            .expect("iter should succeed");
        let sink_pos = order
            .iter()
            .position(|&id| id == sink_id)
            .expect("iter should succeed");

        assert!(source_pos < filter_pos);
        assert!(filter_pos < sink_pos);
    }

    #[test]
    fn test_empty_graph_error() {
        let builder = GraphBuilder::<Empty>::new();
        // Cannot call build on empty builder due to type state
        // This test verifies the type state prevents invalid usage
        let _ = builder; // Just verify it compiles
    }

    #[test]
    fn test_graph_reset() {
        let source = PassthroughFilter::new_source(NodeId(0), "source");
        let sink = NullSink::new(NodeId(0), "sink");

        let (builder, source_id) = GraphBuilder::new().add_node(Box::new(source));
        let (builder, sink_id) = builder.add_node(Box::new(sink));

        let mut graph = builder
            .connect(source_id, PortId(0), sink_id, PortId(0))
            .expect("operation should succeed")
            .build()
            .expect("operation should succeed");

        // Initialize and reset
        graph.initialize().expect("initialize should succeed");
        graph.reset().expect("reset should succeed");

        // Nodes should be back to idle
        for id in graph.node_ids() {
            let node = graph.node(id).expect("node should succeed");
            assert_eq!(node.state(), NodeState::Idle);
        }
    }
}
