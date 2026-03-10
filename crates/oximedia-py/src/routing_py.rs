//! Python bindings for audio/video routing operations.
//!
//! Provides `PySignalRouter`, `PyRouteNode`, `PyRouteConnection`,
//! and standalone functions for routing management from Python.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_node_type(ntype: &str) -> PyResult<()> {
    match ntype.to_lowercase().as_str() {
        "input" | "output" | "mixer" | "splitter" | "processor" | "monitor" | "bus" => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "Unknown node type '{}'. Supported: input, output, mixer, splitter, processor, monitor, bus",
            other
        ))),
    }
}

fn validate_matrix_type(mtype: &str) -> PyResult<()> {
    match mtype.to_lowercase().as_str() {
        "audio" | "video" | "madi" | "dante" | "nmos" | "sdi" | "ip" => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "Unknown matrix type '{}'. Supported: audio, video, madi, dante, nmos, sdi, ip",
            other
        ))),
    }
}

// ---------------------------------------------------------------------------
// PyRouteNode
// ---------------------------------------------------------------------------

/// Represents a node in a routing graph.
#[pyclass]
#[derive(Clone)]
pub struct PyRouteNode {
    /// Node identifier.
    #[pyo3(get)]
    pub id: u32,
    /// Node name.
    #[pyo3(get)]
    pub name: String,
    /// Node type: input, output, mixer, splitter, processor, monitor.
    #[pyo3(get)]
    pub node_type: String,
    /// Number of channels.
    #[pyo3(get)]
    pub channels: u32,
    /// Display label.
    #[pyo3(get)]
    pub label: String,
}

#[pymethods]
impl PyRouteNode {
    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("id".to_string(), self.id.to_string());
        m.insert("name".to_string(), self.name.clone());
        m.insert("node_type".to_string(), self.node_type.clone());
        m.insert("channels".to_string(), self.channels.to_string());
        m.insert("label".to_string(), self.label.clone());
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "PyRouteNode(id={}, name='{}', type='{}', ch={})",
            self.id, self.name, self.node_type, self.channels
        )
    }
}

// ---------------------------------------------------------------------------
// PyRouteConnection
// ---------------------------------------------------------------------------

/// Represents a connection between two routing nodes.
#[pyclass]
#[derive(Clone)]
pub struct PyRouteConnection {
    /// Source node ID.
    #[pyo3(get)]
    pub source_id: u32,
    /// Destination node ID.
    #[pyo3(get)]
    pub destination_id: u32,
    /// Gain in dB.
    #[pyo3(get)]
    pub gain_db: f32,
    /// Whether the connection is active.
    #[pyo3(get)]
    pub active: bool,
}

#[pymethods]
impl PyRouteConnection {
    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("source_id".to_string(), self.source_id.to_string());
        m.insert(
            "destination_id".to_string(),
            self.destination_id.to_string(),
        );
        m.insert("gain_db".to_string(), format!("{:.1}", self.gain_db));
        m.insert("active".to_string(), self.active.to_string());
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "PyRouteConnection(src={}, dst={}, gain={:.1}dB, active={})",
            self.source_id, self.destination_id, self.gain_db, self.active
        )
    }
}

// ---------------------------------------------------------------------------
// PySignalRouter
// ---------------------------------------------------------------------------

/// Signal routing engine for audio/video.
#[pyclass]
pub struct PySignalRouter {
    name: String,
    matrix_type: String,
    inputs: u32,
    outputs: u32,
    nodes: Vec<PyRouteNode>,
    connections: Vec<PyRouteConnection>,
    next_id: u32,
}

#[pymethods]
impl PySignalRouter {
    /// Create a new signal router.
    #[new]
    #[pyo3(signature = (name, inputs, outputs, matrix_type="audio"))]
    fn new(name: &str, inputs: u32, outputs: u32, matrix_type: &str) -> PyResult<Self> {
        validate_matrix_type(matrix_type)?;
        Ok(Self {
            name: name.to_string(),
            matrix_type: matrix_type.to_string(),
            inputs,
            outputs,
            nodes: Vec::new(),
            connections: Vec::new(),
            next_id: 0,
        })
    }

    /// Add a node to the routing graph.
    #[pyo3(signature = (name, node_type, channels=2, label=None))]
    fn add_node(
        &mut self,
        name: &str,
        node_type: &str,
        channels: u32,
        label: Option<&str>,
    ) -> PyResult<PyRouteNode> {
        validate_node_type(node_type)?;
        let node = PyRouteNode {
            id: self.next_id,
            name: name.to_string(),
            node_type: node_type.to_string(),
            channels,
            label: label.unwrap_or(name).to_string(),
        };
        self.next_id += 1;
        self.nodes.push(node.clone());
        Ok(node)
    }

    /// Connect two nodes.
    #[pyo3(signature = (source_id, destination_id, gain_db=0.0))]
    fn connect(
        &mut self,
        source_id: u32,
        destination_id: u32,
        gain_db: f32,
    ) -> PyResult<PyRouteConnection> {
        let src_exists = self.nodes.iter().any(|n| n.id == source_id);
        let dst_exists = self.nodes.iter().any(|n| n.id == destination_id);
        if !src_exists {
            return Err(PyValueError::new_err(format!(
                "Source node {} not found",
                source_id
            )));
        }
        if !dst_exists {
            return Err(PyValueError::new_err(format!(
                "Destination node {} not found",
                destination_id
            )));
        }
        let conn = PyRouteConnection {
            source_id,
            destination_id,
            gain_db,
            active: true,
        };
        self.connections.push(conn.clone());
        Ok(conn)
    }

    /// Disconnect two nodes.
    fn disconnect(&mut self, source_id: u32, destination_id: u32) -> PyResult<bool> {
        let before = self.connections.len();
        self.connections
            .retain(|c| !(c.source_id == source_id && c.destination_id == destination_id));
        Ok(self.connections.len() < before)
    }

    /// Get all nodes.
    fn nodes(&self) -> Vec<PyRouteNode> {
        self.nodes.clone()
    }

    /// Get all connections.
    fn connections(&self) -> Vec<PyRouteConnection> {
        self.connections.clone()
    }

    /// Validate the routing configuration.
    fn validate(&self) -> HashMap<String, String> {
        let orphan_count = self
            .nodes
            .iter()
            .filter(|n| {
                !self
                    .connections
                    .iter()
                    .any(|c| c.source_id == n.id || c.destination_id == n.id)
            })
            .count();
        let mut m = HashMap::new();
        m.insert("is_valid".to_string(), (orphan_count == 0).to_string());
        m.insert("node_count".to_string(), self.nodes.len().to_string());
        m.insert(
            "connection_count".to_string(),
            self.connections.len().to_string(),
        );
        m.insert("orphan_nodes".to_string(), orphan_count.to_string());
        m
    }

    /// Get router info.
    fn info(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("name".to_string(), self.name.clone());
        m.insert("matrix_type".to_string(), self.matrix_type.clone());
        m.insert("inputs".to_string(), self.inputs.to_string());
        m.insert("outputs".to_string(), self.outputs.to_string());
        m.insert("node_count".to_string(), self.nodes.len().to_string());
        m.insert(
            "connection_count".to_string(),
            self.connections.len().to_string(),
        );
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "PySignalRouter(name='{}', type='{}', {}x{}, nodes={}, conns={})",
            self.name,
            self.matrix_type,
            self.inputs,
            self.outputs,
            self.nodes.len(),
            self.connections.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Create a new signal router.
#[pyfunction]
#[pyo3(signature = (name, inputs, outputs, matrix_type="audio"))]
pub fn create_router(
    name: &str,
    inputs: u32,
    outputs: u32,
    matrix_type: &str,
) -> PyResult<PySignalRouter> {
    PySignalRouter::new(name, inputs, outputs, matrix_type)
}

/// Validate a route configuration (as a dict).
#[pyfunction]
pub fn validate_route(config: HashMap<String, String>) -> PyResult<HashMap<String, String>> {
    let mut result = HashMap::new();
    let has_source = config.contains_key("source");
    let has_dest = config.contains_key("destination");
    result.insert("is_valid".to_string(), (has_source && has_dest).to_string());
    result.insert("has_source".to_string(), has_source.to_string());
    result.insert("has_destination".to_string(), has_dest.to_string());
    Ok(result)
}

/// List supported matrix types.
#[pyfunction]
pub fn list_matrix_types() -> Vec<String> {
    vec![
        "audio".to_string(),
        "video".to_string(),
        "madi".to_string(),
        "dante".to_string(),
        "nmos".to_string(),
        "sdi".to_string(),
        "ip".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all routing bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRouteNode>()?;
    m.add_class::<PyRouteConnection>()?;
    m.add_class::<PySignalRouter>()?;
    m.add_function(wrap_pyfunction!(create_router, m)?)?;
    m.add_function(wrap_pyfunction!(validate_route, m)?)?;
    m.add_function(wrap_pyfunction!(list_matrix_types, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_node_type() {
        assert!(validate_node_type("input").is_ok());
        assert!(validate_node_type("mixer").is_ok());
        assert!(validate_node_type("bad").is_err());
    }

    #[test]
    fn test_validate_matrix_type() {
        assert!(validate_matrix_type("audio").is_ok());
        assert!(validate_matrix_type("dante").is_ok());
        assert!(validate_matrix_type("bad").is_err());
    }

    #[test]
    fn test_validate_route_function() {
        let mut config = HashMap::new();
        config.insert("source".to_string(), "mic1".to_string());
        config.insert("destination".to_string(), "monitor".to_string());
        let result = validate_route(config).expect("should succeed");
        assert_eq!(result.get("is_valid").map(|s| s.as_str()), Some("true"));
    }

    #[test]
    fn test_validate_route_missing_dest() {
        let mut config = HashMap::new();
        config.insert("source".to_string(), "mic1".to_string());
        let result = validate_route(config).expect("should succeed");
        assert_eq!(result.get("is_valid").map(|s| s.as_str()), Some("false"));
    }

    #[test]
    fn test_list_matrix_types() {
        let types = list_matrix_types();
        assert!(types.contains(&"audio".to_string()));
        assert!(types.contains(&"dante".to_string()));
    }
}
