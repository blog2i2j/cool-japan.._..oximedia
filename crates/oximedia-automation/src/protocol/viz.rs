//! Vizrt graphics control protocol simulation.
//!
//! Vizrt engines accept XML-based commands over TCP. This module provides
//! a simple string-based XML encoder/decoder without external XML libraries.

#![allow(dead_code)]

use std::collections::HashMap;

/// Actions that can be performed on a Viz scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VizAction {
    /// Load a scene from disk
    Load,
    /// Continue to the next keyframe
    Continue,
    /// Continue all active scenes
    ContinueAll,
    /// Take the scene out (off-air)
    Out,
    /// Take a scene in (on-air)
    TakeIn,
    /// Take a scene out (off-air, alias)
    TakeOut,
    /// Set a text plug value
    SetText,
    /// Set an image plug value
    SetImage,
}

impl VizAction {
    /// Return the Viz command string for this action.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            VizAction::Load => "LOAD",
            VizAction::Continue => "CONTINUE",
            VizAction::ContinueAll => "CONTINUE_ALL",
            VizAction::Out => "OUT",
            VizAction::TakeIn => "TAKE_IN",
            VizAction::TakeOut => "TAKE_OUT",
            VizAction::SetText => "SET_TEXT",
            VizAction::SetImage => "SET_IMAGE",
        }
    }
}

/// A command directed at a Viz engine.
#[derive(Debug, Clone)]
pub struct VizCommand {
    /// Scene tree path (e.g. `/SCENE*LOWER_THIRD`)
    pub scene_path: String,
    /// Action to perform
    pub action: VizAction,
    /// Optional parameters (plug name -> value)
    pub params: HashMap<String, String>,
}

impl VizCommand {
    /// Create a new `VizCommand`.
    pub fn new(scene_path: impl Into<String>, action: VizAction) -> Self {
        Self {
            scene_path: scene_path.into(),
            action,
            params: HashMap::new(),
        }
    }

    /// Add a parameter.
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }
}

/// Snapshot of a Viz engine's runtime state.
#[derive(Debug, Clone)]
pub struct VizEngineState {
    /// Currently active scene path, if any
    pub active_scene: Option<String>,
    /// Whether the engine is currently rendering
    pub is_rendering: bool,
    /// Current render frame rate
    pub fps: f32,
    /// Total frames rendered since startup
    pub frame_count: u64,
}

impl Default for VizEngineState {
    fn default() -> Self {
        Self {
            active_scene: None,
            is_rendering: false,
            fps: 25.0,
            frame_count: 0,
        }
    }
}

/// Viz engine protocol encoder/decoder.
pub struct VizProtocol;

impl VizProtocol {
    /// Encode a [`VizCommand`] into an XML command string.
    ///
    /// Output format:
    /// ```xml
    /// <viz_command action="LOAD" scene="/SCENE*LOWER_THIRD">
    ///   <param name="text" value="Hello"/>
    /// </viz_command>
    /// ```
    #[must_use]
    pub fn format_command(cmd: &VizCommand) -> String {
        let mut xml = format!(
            r#"<viz_command action="{}" scene="{}">"#,
            cmd.action.as_str(),
            Self::escape_xml(&cmd.scene_path)
        );

        // Sort for deterministic output.
        let mut sorted: Vec<(&String, &String)> = cmd.params.iter().collect();
        sorted.sort_by_key(|(k, _)| k.as_str());

        for (k, v) in sorted {
            xml.push_str(&format!(
                r#"<param name="{}" value="{}"/>"#,
                Self::escape_xml(k),
                Self::escape_xml(v)
            ));
        }

        xml.push_str("</viz_command>");
        xml
    }

    /// Parse a simple Viz engine state response from an XML-like string.
    ///
    /// The parser uses basic string scanning rather than a full XML library.
    ///
    /// # Errors
    /// Returns `Err` if no `<viz_state …/>` element can be found.
    pub fn parse_response(xml: &str) -> Result<VizEngineState, String> {
        // Expect something like:
        // <viz_state rendering="true" fps="25.0" frames="12345" scene="/LOWER_THIRD"/>
        if !xml.contains("viz_state") {
            return Err("No viz_state element found in response".to_string());
        }

        let is_rendering = Self::extract_attr(xml, "rendering").is_some_and(|v| v == "true");

        let fps = Self::extract_attr(xml, "fps")
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(25.0);

        let frame_count = Self::extract_attr(xml, "frames")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);

        let active_scene = Self::extract_attr(xml, "scene");

        Ok(VizEngineState {
            active_scene,
            is_rendering,
            fps,
            frame_count,
        })
    }

    /// Extract the value of an XML attribute `name="value"` from `xml`.
    fn extract_attr(xml: &str, name: &str) -> Option<String> {
        let needle = format!(r#"{name}=""#);
        let start = xml.find(&needle)? + needle.len();
        let rest = &xml[start..];
        let end = rest.find('"')?;
        Some(rest[..end].to_string())
    }

    /// Minimal XML escaping for attribute values.
    fn escape_xml(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    }
}

/// Node type within a Viz scene graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VizNodeType {
    /// Root container node
    Root,
    /// Container (group) node
    Container,
    /// Geometry node
    Geom,
    /// Text plug node
    Text,
    /// Image plug node
    Image,
}

/// A node in the Viz scene graph.
#[derive(Debug, Clone)]
pub struct VizNode {
    /// Node name
    pub name: String,
    /// Node type classification
    pub node_type: VizNodeType,
    /// Names of child nodes
    pub children: Vec<String>,
}

impl VizNode {
    /// Create a new leaf node with no children.
    pub fn new(name: impl Into<String>, node_type: VizNodeType) -> Self {
        Self {
            name: name.into(),
            node_type,
            children: Vec::new(),
        }
    }

    /// Add a child node name.
    pub fn add_child(&mut self, child_name: impl Into<String>) {
        self.children.push(child_name.into());
    }
}

/// A hierarchical Viz scene graph.
#[derive(Debug, Default)]
pub struct VizSceneGraph {
    /// All nodes keyed by name
    pub nodes: HashMap<String, VizNode>,
}

impl VizSceneGraph {
    /// Create an empty scene graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a node in the graph.
    pub fn add_node(&mut self, node: VizNode) {
        self.nodes.insert(node.name.clone(), node);
    }

    /// Find a node by name.
    pub fn get_node(&self, name: &str) -> Option<&VizNode> {
        self.nodes.get(name)
    }

    /// Return the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viz_action_as_str() {
        assert_eq!(VizAction::Load.as_str(), "LOAD");
        assert_eq!(VizAction::Continue.as_str(), "CONTINUE");
        assert_eq!(VizAction::ContinueAll.as_str(), "CONTINUE_ALL");
        assert_eq!(VizAction::Out.as_str(), "OUT");
        assert_eq!(VizAction::TakeIn.as_str(), "TAKE_IN");
        assert_eq!(VizAction::TakeOut.as_str(), "TAKE_OUT");
        assert_eq!(VizAction::SetText.as_str(), "SET_TEXT");
        assert_eq!(VizAction::SetImage.as_str(), "SET_IMAGE");
    }

    #[test]
    fn test_format_command_basic() {
        let cmd = VizCommand::new("/SCENE*LOWER_THIRD", VizAction::Load);
        let xml = VizProtocol::format_command(&cmd);
        assert!(xml.contains("action=\"LOAD\""));
        assert!(xml.contains("scene=\"/SCENE*LOWER_THIRD\""));
    }

    #[test]
    fn test_format_command_with_params() {
        let cmd = VizCommand::new("/SCENE*TICKER", VizAction::SetText)
            .with_param("text", "Breaking News");
        let xml = VizProtocol::format_command(&cmd);
        assert!(xml.contains(r#"name="text""#));
        assert!(xml.contains(r#"value="Breaking News""#));
    }

    #[test]
    fn test_format_command_xml_escaping() {
        let cmd = VizCommand::new("/SCENE*TEST", VizAction::SetText).with_param("text", "A&B");
        let xml = VizProtocol::format_command(&cmd);
        assert!(xml.contains("A&amp;B"));
    }

    #[test]
    fn test_parse_response_ok() {
        let xml = r#"<viz_state rendering="true" fps="25.0" frames="100" scene="/LOWER_THIRD"/>"#;
        let state = VizProtocol::parse_response(xml).expect("parse_response should succeed");
        assert!(state.is_rendering);
        assert!((state.fps - 25.0).abs() < f32::EPSILON);
        assert_eq!(state.frame_count, 100);
        assert_eq!(state.active_scene.as_deref(), Some("/LOWER_THIRD"));
    }

    #[test]
    fn test_parse_response_not_rendering() {
        let xml = r#"<viz_state rendering="false" fps="25.0" frames="0"/>"#;
        let state = VizProtocol::parse_response(xml).expect("parse_response should succeed");
        assert!(!state.is_rendering);
        assert!(state.active_scene.is_none());
    }

    #[test]
    fn test_parse_response_missing_element() {
        let result = VizProtocol::parse_response("<other_element/>");
        assert!(result.is_err());
    }

    #[test]
    fn test_viz_scene_graph_add_get() {
        let mut graph = VizSceneGraph::new();
        graph.add_node(VizNode::new("Root", VizNodeType::Root));
        graph.add_node(VizNode::new("Text1", VizNodeType::Text));
        assert_eq!(graph.node_count(), 2);
        assert!(graph.get_node("Root").is_some());
    }

    #[test]
    fn test_viz_node_children() {
        let mut node = VizNode::new("Container", VizNodeType::Container);
        node.add_child("TextA");
        node.add_child("ImageB");
        assert_eq!(node.children.len(), 2);
    }

    #[test]
    fn test_viz_node_type_variants() {
        let types = [
            VizNodeType::Root,
            VizNodeType::Container,
            VizNodeType::Geom,
            VizNodeType::Text,
            VizNodeType::Image,
        ];
        assert_eq!(types.len(), 5);
    }
}
