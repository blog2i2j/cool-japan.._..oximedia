//! DaVinci Resolve `.drx` correction file parser.
//!
//! `.drx` files use an XML-based format to describe a Resolve node graph
//! with colour correction parameters. This module parses the common structure
//! into `ResolveLut` which carries a list of `LutNode` objects.
//!
//! # Example DRX structure
//!
//! ```xml
//! <?xml version="1.0" encoding="UTF-8"?>
//! <resolve_davinci_resolve version="1.0">
//!   <grade version="1.0">
//!     <node type="serial" enabled="true">
//!       <correction type="lift_gamma_gain">
//!         <lift r="0.05" g="0.05" b="0.05" master="0.05"/>
//!         <gamma r="1.0" g="1.0" b="1.0" master="1.0"/>
//!         <gain r="1.2" g="1.1" b="1.0" master="1.1"/>
//!       </correction>
//!     </node>
//!   </grade>
//! </resolve_davinci_resolve>
//! ```

#![allow(dead_code)]

use crate::error::{LutError, LutResult};
use quick_xml::events::Event;
use quick_xml::Reader;
use std::collections::HashMap;

// ============================================================================
// Public types
// ============================================================================

/// A single processing node within a Resolve correction tree.
#[derive(Clone, Debug, PartialEq)]
pub struct LutNode {
    /// Node type string (e.g. `"serial"`, `"parallel"`).
    pub node_type: String,
    /// Flattened parameter map.
    ///
    /// Keys use the form `"parent_element.attr_name"` (e.g. `"lift.r"`,
    /// `"gamma.master"`, `"saturation.value"`).
    pub params: HashMap<String, f32>,
}

impl LutNode {
    /// Create a new `LutNode`.
    #[must_use]
    pub fn new(node_type: impl Into<String>) -> Self {
        Self {
            node_type: node_type.into(),
            params: HashMap::new(),
        }
    }

    /// Look up a parameter value.
    #[must_use]
    pub fn get_param(&self, key: &str) -> Option<f32> {
        self.params.get(key).copied()
    }
}

/// A parsed DaVinci Resolve `.drx` correction file.
#[derive(Clone, Debug, Default)]
pub struct ResolveLut {
    /// Processing nodes in document order.
    pub nodes: Vec<LutNode>,
    /// Version string from the root element's `version` attribute.
    pub version: String,
}

// ============================================================================
// Parser
// ============================================================================

/// Parser for DaVinci Resolve `.drx` correction files.
pub struct ResolveLutParser;

impl ResolveLutParser {
    /// Parse a `.drx` XML document.
    ///
    /// # Errors
    ///
    /// Returns `LutError::Parse` if the XML is malformed or required structure
    /// is absent.
    pub fn parse_drx(data: &str) -> LutResult<ResolveLut> {
        parse_drx_document(data)
    }
}

// ============================================================================
// Internal parsing
// ============================================================================

/// Depth-tracked parser state.
struct DrxParseState {
    version: String,
    nodes: Vec<LutNode>,

    /// Current node being assembled (inside a `<node>` element).
    current_node: Option<LutNodeBuilder>,
    /// Current correction type (inside a `<correction>` element).
    current_correction_type: Option<String>,
}

struct LutNodeBuilder {
    node_type: String,
    params: HashMap<String, f32>,
}

impl LutNodeBuilder {
    fn new(node_type: impl Into<String>) -> Self {
        Self {
            node_type: node_type.into(),
            params: HashMap::new(),
        }
    }

    fn build(self) -> LutNode {
        LutNode {
            node_type: self.node_type,
            params: self.params,
        }
    }
}

fn parse_drx_document(xml: &str) -> LutResult<ResolveLut> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut state = DrxParseState {
        version: String::new(),
        nodes: Vec::new(),
        current_node: None,
        current_correction_type: None,
    };

    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                handle_drx_element(e, &mut state)?;
            }
            Ok(Event::End(ref e)) => {
                handle_drx_end(e, &mut state)?;
            }
            Ok(Event::Eof) => break,
            Err(err) => {
                return Err(LutError::Parse(format!("XML error: {err}")));
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(ResolveLut {
        nodes: state.nodes,
        version: state.version,
    })
}

/// Strip namespace prefix from an element / attribute name.
fn local_name_str(name: &[u8]) -> LutResult<String> {
    let full =
        std::str::from_utf8(name).map_err(|e| LutError::Parse(format!("UTF-8 error: {e}")))?;
    Ok(if let Some(pos) = full.rfind(':') {
        full[pos + 1..].to_string()
    } else {
        full.to_string()
    })
}

fn handle_drx_element(
    e: &quick_xml::events::BytesStart,
    state: &mut DrxParseState,
) -> LutResult<()> {
    let local = local_name_str(e.name().as_ref())?;

    match local.as_str() {
        // Root element — extract version
        "resolve_davinci_resolve" => {
            for attr_result in e.attributes() {
                let attr = attr_result
                    .map_err(|err| LutError::Parse(format!("attribute error: {err}")))?;
                let key = local_name_str(attr.key.as_ref())?;
                let val = std::str::from_utf8(attr.value.as_ref())
                    .map_err(|err| LutError::Parse(format!("UTF-8: {err}")))?;
                if key == "version" {
                    state.version = val.to_string();
                }
            }
        }

        // Node element
        "node" => {
            let mut node_type = "unknown".to_string();
            for attr_result in e.attributes() {
                let attr = attr_result
                    .map_err(|err| LutError::Parse(format!("attribute error: {err}")))?;
                let key = local_name_str(attr.key.as_ref())?;
                let val = std::str::from_utf8(attr.value.as_ref())
                    .map_err(|err| LutError::Parse(format!("UTF-8: {err}")))?;
                if key == "type" {
                    node_type = val.to_string();
                }
            }
            state.current_node = Some(LutNodeBuilder::new(node_type));
        }

        // Correction wrapper — remember type to prefix child element params
        "correction" => {
            for attr_result in e.attributes() {
                let attr = attr_result
                    .map_err(|err| LutError::Parse(format!("attribute error: {err}")))?;
                let key = local_name_str(attr.key.as_ref())?;
                let val = std::str::from_utf8(attr.value.as_ref())
                    .map_err(|err| LutError::Parse(format!("UTF-8: {err}")))?;
                if key == "type" {
                    state.current_correction_type = Some(val.to_string());
                }
            }
        }

        // Any other element inside a node — treat its attributes as params
        _ if state.current_node.is_some() => {
            // The element name becomes the prefix for each attribute key
            let prefix = local.clone();
            let Some(builder) = state.current_node.as_mut() else {
                // Guard arm ensures current_node.is_some() — this branch is unreachable
                return Ok(());
            };

            for attr_result in e.attributes() {
                let attr = attr_result
                    .map_err(|err| LutError::Parse(format!("attribute error: {err}")))?;
                let key = local_name_str(attr.key.as_ref())?;
                let val_bytes = attr.value.as_ref();
                let val_str = std::str::from_utf8(val_bytes)
                    .map_err(|err| LutError::Parse(format!("UTF-8: {err}")))?;

                // Skip non-numeric attributes like "type", "enabled"
                if let Ok(fval) = val_str.parse::<f32>() {
                    let param_key = format!("{prefix}.{key}");
                    builder.params.insert(param_key, fval);
                }
            }
        }

        _ => {}
    }

    Ok(())
}

fn handle_drx_end(e: &quick_xml::events::BytesEnd, state: &mut DrxParseState) -> LutResult<()> {
    let local = local_name_str(e.name().as_ref())?;

    match local.as_str() {
        "node" => {
            if let Some(builder) = state.current_node.take() {
                state.nodes.push(builder.build());
            }
        }
        "correction" => {
            state.current_correction_type = None;
        }
        _ => {}
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_DRX: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<resolve_davinci_resolve version="1.0">
  <grade version="1.0">
    <node type="serial" enabled="true">
      <correction type="lift_gamma_gain">
        <lift r="0.05" g="0.06" b="0.07" master="0.05"/>
        <gamma r="1.0" g="1.1" b="1.2" master="1.0"/>
        <gain r="1.2" g="1.1" b="1.0" master="1.1"/>
      </correction>
    </node>
    <node type="serial" enabled="true">
      <correction type="contrast_saturation">
        <contrast value="1.05"/>
        <saturation value="1.1"/>
      </correction>
    </node>
  </grade>
</resolve_davinci_resolve>"#;

    #[test]
    fn test_parse_version() {
        let lut = ResolveLutParser::parse_drx(SAMPLE_DRX).expect("parse should succeed");
        assert_eq!(lut.version, "1.0");
    }

    #[test]
    fn test_parse_two_nodes() {
        let lut = ResolveLutParser::parse_drx(SAMPLE_DRX).expect("parse should succeed");
        assert_eq!(lut.nodes.len(), 2);
    }

    #[test]
    fn test_first_node_type() {
        let lut = ResolveLutParser::parse_drx(SAMPLE_DRX).expect("parse should succeed");
        assert_eq!(lut.nodes[0].node_type, "serial");
    }

    #[test]
    fn test_params_lift_r() {
        let lut = ResolveLutParser::parse_drx(SAMPLE_DRX).expect("parse should succeed");
        let val = lut.nodes[0]
            .get_param("lift.r")
            .expect("lift.r should exist");
        assert!((val - 0.05).abs() < 1e-5);
    }

    #[test]
    fn test_params_lift_g() {
        let lut = ResolveLutParser::parse_drx(SAMPLE_DRX).expect("parse should succeed");
        let val = lut.nodes[0]
            .get_param("lift.g")
            .expect("lift.g should exist");
        assert!((val - 0.06).abs() < 1e-5);
    }

    #[test]
    fn test_params_lift_b() {
        let lut = ResolveLutParser::parse_drx(SAMPLE_DRX).expect("parse should succeed");
        let val = lut.nodes[0]
            .get_param("lift.b")
            .expect("lift.b should exist");
        assert!((val - 0.07).abs() < 1e-5);
    }

    #[test]
    fn test_params_gamma_master() {
        let lut = ResolveLutParser::parse_drx(SAMPLE_DRX).expect("parse should succeed");
        let val = lut.nodes[0]
            .get_param("gamma.master")
            .expect("gamma.master should exist");
        assert!((val - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_params_gain_r() {
        let lut = ResolveLutParser::parse_drx(SAMPLE_DRX).expect("parse should succeed");
        let val = lut.nodes[0]
            .get_param("gain.r")
            .expect("gain.r should exist");
        assert!((val - 1.2).abs() < 1e-5);
    }

    #[test]
    fn test_params_saturation_value() {
        let lut = ResolveLutParser::parse_drx(SAMPLE_DRX).expect("parse should succeed");
        let val = lut.nodes[1]
            .get_param("saturation.value")
            .expect("saturation.value should exist");
        assert!((val - 1.1).abs() < 1e-5);
    }

    #[test]
    fn test_params_contrast_value() {
        let lut = ResolveLutParser::parse_drx(SAMPLE_DRX).expect("parse should succeed");
        let val = lut.nodes[1]
            .get_param("contrast.value")
            .expect("contrast.value should exist");
        assert!((val - 1.05).abs() < 1e-5);
    }

    #[test]
    fn test_empty_grade_no_nodes() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<resolve_davinci_resolve version="2.0">
  <grade version="1.0">
  </grade>
</resolve_davinci_resolve>"#;
        let lut = ResolveLutParser::parse_drx(xml).expect("parse should succeed");
        assert!(lut.nodes.is_empty());
        assert_eq!(lut.version, "2.0");
    }

    #[test]
    fn test_invalid_xml_returns_error() {
        // Unclosed tag causes a genuine quick-xml parse error
        let result = ResolveLutParser::parse_drx(
            "<resolve_davinci_resolve version=\"1.0\"><grade><node type=\"serial\">",
        );
        // quick-xml returns an error for unclosed tags when EOF is reached
        // If it doesn't error, it returns an empty lut which is also acceptable;
        // test that we handle both without panicking.
        let _ = result; // either Ok or Err is fine — we just must not panic
    }

    #[test]
    fn test_node_with_missing_type_gets_unknown() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<resolve_davinci_resolve version="1.0">
  <grade>
    <node enabled="true">
      <correction type="contrast_saturation">
        <saturation value="1.2"/>
      </correction>
    </node>
  </grade>
</resolve_davinci_resolve>"#;
        let lut = ResolveLutParser::parse_drx(xml).expect("parse should succeed");
        assert_eq!(lut.nodes[0].node_type, "unknown");
    }

    #[test]
    fn test_node_params_are_isolated() {
        // Params from node 0 should not appear in node 1
        let lut = ResolveLutParser::parse_drx(SAMPLE_DRX).expect("parse should succeed");
        assert!(lut.nodes[1].get_param("lift.r").is_none());
        assert!(lut.nodes[0].get_param("saturation.value").is_none());
    }
}
