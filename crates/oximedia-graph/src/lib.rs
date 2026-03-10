//! Filter graph pipeline for `OxiMedia`.
//!
//! This crate provides a filter graph implementation for processing media data
//! through a pipeline of operations. The graph connects nodes (filters) together
//! to form a processing pipeline.
//!
//! # Architecture
//!
//! The filter graph consists of:
//!
//! - **Nodes**: Processing units that implement the [`node::Node`] trait
//! - **Ports**: Connection points for data flow between nodes
//! - **Connections**: Links between output and input ports
//! - **Frames**: Data units passed through the graph
//!
//! # Example
//!
//! ```
//! use oximedia_graph::graph::GraphBuilder;
//! use oximedia_graph::filters::video::{PassthroughFilter, NullSink};
//! use oximedia_graph::node::NodeId;
//! use oximedia_graph::port::PortId;
//!
//! // Create a simple graph: source -> sink
//! let source = PassthroughFilter::new_source(NodeId(0), "source");
//! let sink = NullSink::new(NodeId(0), "sink");
//!
//! let (builder, source_id) = GraphBuilder::new().add_node(Box::new(source));
//! let (builder, sink_id) = builder.add_node(Box::new(sink));
//!
//! let graph = builder
//!     .connect(source_id, PortId(0), sink_id, PortId(0))
//!     ?
//!     .build()
//!     ?;
//!
//! assert_eq!(graph.node_count(), 2);
//! ```
//!
//! # Node Types
//!
//! Nodes are classified into three types:
//!
//! - **Source**: Entry points that produce frames (e.g., decoders)
//! - **Filter**: Transform frames (e.g., scalers, color converters)
//! - **Sink**: Consume frames (e.g., encoders, displays)
//!
//! # Frame Flow
//!
//! Frames flow through the graph following the connections. The graph
//! automatically computes a topological order for execution to ensure
//! correct data flow.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
// Allow common pedantic lints for this crate
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::redundant_closure_for_method_calls,
    clippy::similar_names,
    dead_code,
    clippy::pedantic
)]

pub mod context;
pub mod data_flow;
pub mod edge_weight;
pub mod error;
pub mod filters;
pub mod frame;
pub mod graph;
pub mod graph_stats;
pub mod layout;
pub mod metrics_graph;
pub mod node;
pub mod node_registry;
pub mod optimization;
pub mod port;
pub mod processing_graph;
pub mod profiling;
pub mod scheduler;
pub mod serialize;
pub mod visualization;

// Wave-10 new modules
pub mod graph_validation;
pub mod pipeline_graph;
pub mod subgraph;

// Wave-13 new modules
pub mod cycle_detect;
pub mod graph_merge;
pub mod topological;

// Wave-14 new modules
pub mod dependency_graph;
pub mod graph_partition;
pub mod node_cache;

// Re-export commonly used items
pub use context::{GraphContext, ProcessingStats};
pub use error::{GraphError, GraphResult};
pub use frame::{FilterFrame, FramePool, FrameRef};
pub use graph::{FilterGraph, GraphBuilder};
pub use node::{Node, NodeConfig, NodeId, NodeState, NodeType};
pub use port::{Connection, InputPort, OutputPort, PortFormat, PortId, PortType};
