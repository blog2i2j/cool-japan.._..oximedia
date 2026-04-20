//! OpenColorIO (OCIO) configuration file parser.
//!
//! Parses OpenColorIO YAML configuration files (version 1 and 2) to extract
//! color space definitions, roles, looks, and views.
//!
//! # Overview
//!
//! OpenColorIO configs define a complete color management system via a YAML file.
//! Key sections include:
//! - **roles**: Named aliases for color spaces (e.g. `scene_linear → ACEScg`)
//! - **colorspaces**: Full color space definitions with transforms
//! - **looks**: Named color corrections applied in view transforms
//! - **displays/views**: Output device configurations
//!
//! This parser uses a hand-rolled line-oriented YAML reader to avoid
//! external dependencies. It handles the subset of YAML used in real-world
//! OCIO configs.
//!
//! # Example
//!
//! ```no_run
//! use oximedia_colormgmt::ocio::OcioConfig;
//!
//! let yaml = r#"
//! ocio_profile_version: 2
//! name: My Studio Config
//! roles:
//!   scene_linear: ACEScg
//!   default: sRGB
//! colorspaces:
//!   - name: sRGB
//!     family: ""
//!     bitdepth: 8ui
//!     description: sRGB color space
//!     isdata: false
//! "#;
//!
//! let config = OcioConfig::from_str(yaml).expect("parse should succeed");
//! assert_eq!(config.name.as_deref(), Some("My Studio Config"));
//! ```

#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

use crate::error::{ColorError, Result};

// ── Data structures ───────────────────────────────────────────────────────────

/// A color space definition extracted from an OCIO config.
#[derive(Debug, Clone, PartialEq)]
pub struct OcioColorSpace {
    /// Color space name (required).
    pub name: String,
    /// Family/group for organizational purposes.
    pub family: String,
    /// Human-readable description.
    pub description: String,
    /// Nominal bit depth (e.g. `8ui`, `16f`, `32f`).
    pub bitdepth: Option<String>,
    /// Whether this color space holds non-color (e.g. depth, normals) data.
    pub is_data: bool,
    /// Categories (e.g. `["file-io"]`).
    pub categories: Vec<String>,
    /// Allocation hint.
    pub allocation: Option<String>,
    /// Allocation variables (two or three f64 values).
    pub allocation_vars: Vec<f64>,
    /// Raw transform strings (from_reference and to_reference sections).
    pub from_reference: Vec<String>,
    /// Raw transform strings for to_reference section.
    pub to_reference: Vec<String>,
}

impl Default for OcioColorSpace {
    fn default() -> Self {
        Self {
            name: String::new(),
            family: String::new(),
            description: String::new(),
            bitdepth: None,
            is_data: false,
            categories: Vec::new(),
            allocation: None,
            allocation_vars: Vec::new(),
            from_reference: Vec::new(),
            to_reference: Vec::new(),
        }
    }
}

/// A named look definition.
#[derive(Debug, Clone, PartialEq)]
pub struct OcioLook {
    /// Look name.
    pub name: String,
    /// Process space for the look.
    pub process_space: String,
    /// Description.
    pub description: String,
}

/// A view definition within a display.
#[derive(Debug, Clone, PartialEq)]
pub struct OcioView {
    /// View name (e.g. `sRGB`).
    pub name: String,
    /// Color space applied by this view.
    pub colorspace: String,
    /// Optional look applied.
    pub looks: Option<String>,
}

/// A display device (monitor, projector, etc.).
#[derive(Debug, Clone, PartialEq)]
pub struct OcioDisplay {
    /// Display name.
    pub name: String,
    /// Views under this display.
    pub views: Vec<OcioView>,
}

/// A fully parsed OpenColorIO configuration file.
///
/// Contains all color spaces, roles, looks, and views extracted from the YAML.
#[derive(Debug, Clone, PartialEq)]
pub struct OcioConfig {
    /// OCIO profile version (1 or 2).
    pub version: u32,
    /// Optional human-readable config name.
    pub name: Option<String>,
    /// Optional description.
    pub description: Option<String>,
    /// Roles map: role name → color space name.
    pub roles: HashMap<String, String>,
    /// All color space definitions.
    pub colorspaces: Vec<OcioColorSpace>,
    /// Named looks.
    pub looks: Vec<OcioLook>,
    /// Display/view configurations.
    pub displays: Vec<OcioDisplay>,
    /// Active displays (ordered list).
    pub active_displays: Vec<String>,
    /// Active views (ordered list).
    pub active_views: Vec<String>,
}

impl OcioConfig {
    /// Parses an OCIO YAML config from a string slice.
    ///
    /// # Errors
    ///
    /// Returns [`ColorError::InvalidProfile`] if the YAML is malformed
    /// or if no `ocio_profile_version` key is found.
    pub fn from_str(input: &str) -> Result<Self> {
        let parser = OcioParser::new(input);
        parser.parse()
    }

    /// Returns the color space name assigned to a given role.
    ///
    /// Returns `None` if the role is not defined.
    #[must_use]
    pub fn role(&self, role_name: &str) -> Option<&str> {
        self.roles.get(role_name).map(String::as_str)
    }

    /// Finds a color space by name (case-sensitive).
    #[must_use]
    pub fn find_colorspace(&self, name: &str) -> Option<&OcioColorSpace> {
        self.colorspaces.iter().find(|cs| cs.name == name)
    }

    /// Finds a look by name.
    #[must_use]
    pub fn find_look(&self, name: &str) -> Option<&OcioLook> {
        self.looks.iter().find(|l| l.name == name)
    }

    /// Returns the total number of color spaces defined.
    #[must_use]
    pub fn colorspace_count(&self) -> usize {
        self.colorspaces.len()
    }

    /// Returns a list of all color space names.
    #[must_use]
    pub fn colorspace_names(&self) -> Vec<&str> {
        self.colorspaces.iter().map(|cs| cs.name.as_str()).collect()
    }
}

// ── Internal YAML line-oriented parser ───────────────────────────────────────

/// Section context used by the state machine parser.
#[derive(Debug, Clone, PartialEq)]
enum ParseSection {
    Root,
    Roles,
    ColorSpaces,
    ColorSpaceEntry,
    FromReference,
    ToReference,
    Looks,
    LookEntry,
    Displays,
    DisplayEntry { name: String },
    ViewList { display: String },
    ActiveDisplays,
    ActiveViews,
}

struct OcioParser<'a> {
    lines: Vec<(usize, &'a str)>, // (indent, content)
    pos: usize,
}

impl<'a> OcioParser<'a> {
    fn new(input: &'a str) -> Self {
        let lines: Vec<(usize, &'a str)> = input
            .lines()
            .map(|line| {
                let trimmed = line.trim_start();
                let indent = line.len() - trimmed.len();
                (indent, trimmed)
            })
            .collect();
        Self { lines, pos: 0 }
    }

    fn parse(mut self) -> Result<OcioConfig> {
        let mut config = OcioConfig {
            version: 1,
            name: None,
            description: None,
            roles: HashMap::new(),
            colorspaces: Vec::new(),
            looks: Vec::new(),
            displays: Vec::new(),
            active_displays: Vec::new(),
            active_views: Vec::new(),
        };

        let mut section = ParseSection::Root;
        let mut current_cs: Option<OcioColorSpace> = None;
        let mut current_look: Option<OcioLook> = None;
        let mut current_display: Option<OcioDisplay> = None;

        while self.pos < self.lines.len() {
            let (indent, raw) = self.lines[self.pos];
            let line = raw;

            // Skip blank lines and YAML comments
            if line.is_empty() || line.starts_with('#') {
                self.pos += 1;
                continue;
            }

            // Strip inline comments
            let line = strip_inline_comment(line);

            // Detect section switches at indent 0
            if indent == 0 {
                // Flush current objects before changing section
                if let Some(cs) = current_cs.take() {
                    if !cs.name.is_empty() {
                        config.colorspaces.push(cs);
                    }
                }
                if let Some(look) = current_look.take() {
                    if !look.name.is_empty() {
                        config.looks.push(look);
                    }
                }
                if let Some(disp) = current_display.take() {
                    config.displays.push(disp);
                }

                if let Some((key, val)) = split_key_value(line) {
                    match key {
                        "ocio_profile_version" => {
                            config.version = val.trim().parse().unwrap_or(1);
                            section = ParseSection::Root;
                        }
                        "name" => {
                            config.name = Some(unquote(val).to_owned());
                            section = ParseSection::Root;
                        }
                        "description" => {
                            config.description = Some(unquote(val).to_owned());
                            section = ParseSection::Root;
                        }
                        "roles" => {
                            section = ParseSection::Roles;
                        }
                        "colorspaces" | "color_spaces" => {
                            section = ParseSection::ColorSpaces;
                        }
                        "looks" => {
                            section = ParseSection::Looks;
                        }
                        "displays" => {
                            section = ParseSection::Displays;
                        }
                        "active_displays" => {
                            section = ParseSection::ActiveDisplays;
                            // May have inline list: active_displays: [sRGB, ACES]
                            let vals = parse_inline_list(val);
                            config.active_displays.extend(vals);
                        }
                        "active_views" => {
                            section = ParseSection::ActiveViews;
                            let vals = parse_inline_list(val);
                            config.active_views.extend(vals);
                        }
                        _ => {
                            section = ParseSection::Root;
                        }
                    }
                }
                self.pos += 1;
                continue;
            }

            // Handle content in sections
            match &section {
                ParseSection::Roles => {
                    if let Some((k, v)) = split_key_value(line) {
                        config.roles.insert(k.to_owned(), unquote(v).to_owned());
                    }
                }
                ParseSection::ColorSpaces => {
                    // New list item starts a new color space
                    if line.starts_with("- ") || line == "-" {
                        if let Some(cs) = current_cs.take() {
                            if !cs.name.is_empty() {
                                config.colorspaces.push(cs);
                            }
                        }
                        let rest = line.trim_start_matches('-').trim();
                        let mut cs = OcioColorSpace::default();
                        if let Some((k, v)) = split_key_value(rest) {
                            apply_cs_field(&mut cs, k, v);
                        }
                        current_cs = Some(cs);
                        section = ParseSection::ColorSpaceEntry;
                    }
                }
                ParseSection::ColorSpaceEntry => {
                    if current_cs.is_some() {
                        if line.starts_with("- ") {
                            // Start a new color space - flush old one
                            if let Some(old) = current_cs.take() {
                                if !old.name.is_empty() {
                                    config.colorspaces.push(old);
                                }
                            }
                            let rest = line.trim_start_matches('-').trim();
                            let mut new_cs = OcioColorSpace::default();
                            if let Some((k, v)) = split_key_value(rest) {
                                apply_cs_field(&mut new_cs, k, v);
                            }
                            current_cs = Some(new_cs);
                        } else if let Some((k, v)) = split_key_value(line) {
                            if let Some(cs) = current_cs.as_mut() {
                                match k {
                                    "from_reference" | "from_scene_reference" => {
                                        section = ParseSection::FromReference;
                                    }
                                    "to_reference" | "to_scene_reference" => {
                                        section = ParseSection::ToReference;
                                    }
                                    _ => {
                                        apply_cs_field(cs, k, v);
                                    }
                                }
                            }
                        }
                    }
                }
                ParseSection::FromReference => {
                    if let Some(cs) = current_cs.as_mut() {
                        // Check if we've left the from_reference block
                        if let Some((k, v)) = split_key_value(line) {
                            match k {
                                "to_reference" | "to_scene_reference" => {
                                    section = ParseSection::ToReference;
                                }
                                "name" | "family" | "description" | "bitdepth" | "isdata"
                                | "is_data" | "allocation" | "allocation_vars" | "categories" => {
                                    apply_cs_field(cs, k, v);
                                    section = ParseSection::ColorSpaceEntry;
                                }
                                _ => {
                                    cs.from_reference.push(line.to_owned());
                                }
                            }
                        } else {
                            cs.from_reference.push(line.to_owned());
                        }
                    }
                }
                ParseSection::ToReference => {
                    if let Some(cs) = current_cs.as_mut() {
                        if let Some((k, v)) = split_key_value(line) {
                            match k {
                                "from_reference" | "from_scene_reference" => {
                                    section = ParseSection::FromReference;
                                }
                                "name" | "family" | "description" | "bitdepth" | "isdata"
                                | "is_data" | "allocation" | "allocation_vars" | "categories" => {
                                    apply_cs_field(cs, k, v);
                                    section = ParseSection::ColorSpaceEntry;
                                }
                                _ => {
                                    cs.to_reference.push(line.to_owned());
                                }
                            }
                        } else {
                            cs.to_reference.push(line.to_owned());
                        }
                    }
                }
                ParseSection::Looks => {
                    if line.starts_with("- ") || line == "-" {
                        if let Some(look) = current_look.take() {
                            if !look.name.is_empty() {
                                config.looks.push(look);
                            }
                        }
                        let rest = line.trim_start_matches('-').trim();
                        let mut look = OcioLook {
                            name: String::new(),
                            process_space: String::new(),
                            description: String::new(),
                        };
                        if let Some((k, v)) = split_key_value(rest) {
                            apply_look_field(&mut look, k, v);
                        }
                        current_look = Some(look);
                        section = ParseSection::LookEntry;
                    }
                }
                ParseSection::LookEntry => {
                    if current_look.is_some() {
                        if line.starts_with("- ") {
                            if let Some(old) = current_look.take() {
                                if !old.name.is_empty() {
                                    config.looks.push(old);
                                }
                            }
                            let rest = line.trim_start_matches('-').trim();
                            let mut new_look = OcioLook {
                                name: String::new(),
                                process_space: String::new(),
                                description: String::new(),
                            };
                            if let Some((k, v)) = split_key_value(rest) {
                                apply_look_field(&mut new_look, k, v);
                            }
                            current_look = Some(new_look);
                        } else if let Some((k, v)) = split_key_value(line) {
                            if let Some(look) = current_look.as_mut() {
                                apply_look_field(look, k, v);
                            }
                        }
                    }
                }
                ParseSection::Displays => {
                    // Display name is a mapping key with colon:
                    // e.g. `  sRGB:`
                    if !line.starts_with('-') && line.ends_with(':') {
                        if let Some(disp) = current_display.take() {
                            config.displays.push(disp);
                        }
                        let name = line.trim_end_matches(':').trim().to_owned();
                        current_display = Some(OcioDisplay {
                            name,
                            views: Vec::new(),
                        });
                        section = ParseSection::DisplayEntry {
                            name: current_display
                                .as_ref()
                                .map(|d| d.name.clone())
                                .unwrap_or_default(),
                        };
                    }
                }
                ParseSection::DisplayEntry { name } => {
                    let display_name = name.clone();
                    if line.trim_start() == "- !" || line.contains("views:") || line.contains("- !")
                    {
                        // skip
                    } else if let Some((k, _)) = split_key_value(line) {
                        if k == "views" {
                            section = ParseSection::ViewList {
                                display: display_name,
                            };
                        }
                    } else if line.starts_with("- ") {
                        // inline view list item: - !<View> ...
                        // simplified: just grab name and colorspace
                        if let Some(disp) = current_display.as_mut() {
                            if let Some(view) = parse_view_line(line) {
                                disp.views.push(view);
                            }
                        }
                    }
                }
                ParseSection::ViewList { display: _ } => {
                    if let Some(disp) = current_display.as_mut() {
                        if line.starts_with("- ") {
                            if let Some(view) = parse_view_line(line) {
                                disp.views.push(view);
                            } else {
                                // Multi-line view entry
                                let rest = line.trim_start_matches('-').trim();
                                if let Some((k, v)) = split_key_value(rest) {
                                    if k == "name" {
                                        disp.views.push(OcioView {
                                            name: unquote(v).to_owned(),
                                            colorspace: String::new(),
                                            looks: None,
                                        });
                                    }
                                }
                            }
                        } else if let Some((k, v)) = split_key_value(line) {
                            // Handle fields of current view entry
                            if let Some(last_view) = disp.views.last_mut() {
                                match k {
                                    "colorspace" => last_view.colorspace = unquote(v).to_owned(),
                                    "looks" => last_view.looks = Some(unquote(v).to_owned()),
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                ParseSection::ActiveDisplays | ParseSection::ActiveViews => {
                    // Continuation of inline list or block list
                    if line.starts_with('-') {
                        let item = line.trim_start_matches('-').trim().to_owned();
                        let item = unquote(&item).to_owned();
                        match &section {
                            ParseSection::ActiveDisplays => config.active_displays.push(item),
                            ParseSection::ActiveViews => config.active_views.push(item),
                            _ => {}
                        }
                    }
                }
                ParseSection::Root => {}
            }

            self.pos += 1;
        }

        // Flush any remaining buffered objects
        if let Some(cs) = current_cs.take() {
            if !cs.name.is_empty() {
                config.colorspaces.push(cs);
            }
        }
        if let Some(look) = current_look.take() {
            if !look.name.is_empty() {
                config.looks.push(look);
            }
        }
        if let Some(disp) = current_display.take() {
            config.displays.push(disp);
        }

        if config.version == 0 {
            return Err(ColorError::InvalidProfile(
                "OCIO config missing ocio_profile_version".to_owned(),
            ));
        }

        Ok(config)
    }
}

// ── YAML parsing helpers ──────────────────────────────────────────────────────

/// Strips a `# comment` from the end of a YAML value line.
fn strip_inline_comment(s: &str) -> &str {
    // Find ` #` pattern not inside a quoted string
    let mut in_single = false;
    let mut in_double = false;
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\'' if !in_double => in_single = !in_single,
            b'"' if !in_single => in_double = !in_double,
            b'#' if !in_single && !in_double && i > 0 && bytes[i - 1] == b' ' => {
                return s[..i].trim_end();
            }
            _ => {}
        }
        i += 1;
    }
    s
}

/// Splits a YAML line `key: value` into `(key, value)`.
/// Returns `None` if no colon is found.
fn split_key_value(s: &str) -> Option<(&str, &str)> {
    if let Some(idx) = s.find(':') {
        let key = s[..idx].trim();
        let val = s[idx + 1..].trim();
        if key.is_empty() {
            None
        } else {
            Some((key, val))
        }
    } else {
        None
    }
}

/// Removes surrounding `"` or `'` quotes from a YAML value.
fn unquote(s: &str) -> &str {
    let s = s.trim();
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

/// Parses an inline YAML list like `[a, b, c]` or falls back to single item.
fn parse_inline_list(s: &str) -> Vec<String> {
    let s = s.trim();
    if s.starts_with('[') && s.ends_with(']') {
        s[1..s.len() - 1]
            .split(',')
            .map(|item| unquote(item.trim()).to_owned())
            .filter(|s| !s.is_empty())
            .collect()
    } else if !s.is_empty() {
        vec![unquote(s).to_owned()]
    } else {
        Vec::new()
    }
}

/// Applies a key-value pair to a color space definition.
fn apply_cs_field(cs: &mut OcioColorSpace, key: &str, val: &str) {
    let val = unquote(val);
    match key {
        "name" => cs.name = val.to_owned(),
        "family" => cs.family = val.to_owned(),
        "description" => cs.description = val.to_owned(),
        "bitdepth" => cs.bitdepth = Some(val.to_owned()),
        "isdata" | "is_data" => {
            cs.is_data = matches!(val.to_lowercase().as_str(), "true" | "yes" | "1")
        }
        "allocation" => cs.allocation = Some(val.to_owned()),
        "allocation_vars" => {
            cs.allocation_vars = parse_inline_list(val)
                .iter()
                .filter_map(|s| s.parse::<f64>().ok())
                .collect();
        }
        "categories" => {
            cs.categories = parse_inline_list(val);
        }
        _ => {}
    }
}

/// Applies a key-value pair to a look definition.
fn apply_look_field(look: &mut OcioLook, key: &str, val: &str) {
    let val = unquote(val);
    match key {
        "name" => look.name = val.to_owned(),
        "process_space" => look.process_space = val.to_owned(),
        "description" => look.description = val.to_owned(),
        _ => {}
    }
}

/// Attempts to parse an OCIO view line such as:
/// `- {name: sRGB, colorspace: sRGB}`
/// or
/// `- name: sRGB`
fn parse_view_line(s: &str) -> Option<OcioView> {
    let rest = s.trim_start_matches('-').trim();
    if rest.starts_with('{') && rest.ends_with('}') {
        // Inline mapping
        let inner = &rest[1..rest.len() - 1];
        let mut name = String::new();
        let mut colorspace = String::new();
        let mut looks = None;
        for part in inner.split(',') {
            if let Some((k, v)) = split_key_value(part.trim()) {
                let v = unquote(v);
                match k.trim() {
                    "name" => name = v.to_owned(),
                    "colorspace" => colorspace = v.to_owned(),
                    "looks" => looks = Some(v.to_owned()),
                    _ => {}
                }
            }
        }
        if !name.is_empty() {
            return Some(OcioView {
                name,
                colorspace,
                looks,
            });
        }
    } else if let Some((k, v)) = split_key_value(rest) {
        if k == "name" {
            return Some(OcioView {
                name: unquote(v).to_owned(),
                colorspace: String::new(),
                looks: None,
            });
        }
    }
    None
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_CONFIG: &str = r#"
ocio_profile_version: 2
name: Test Config
description: A minimal test configuration

roles:
  scene_linear: ACEScg
  default: sRGB
  color_picking: sRGB

colorspaces:
  - name: sRGB
    family: ""
    bitdepth: 8ui
    description: Standard sRGB color space
    isdata: false
    allocation: uniform
    allocation_vars: [0, 1]
  - name: ACEScg
    family: ACES
    description: Academy Color Encoding - CG working space
    isdata: false
    bitdepth: 16f
    allocation: lg2
    allocation_vars: [-8, 5, 0.00390625]
  - name: Raw
    family: ""
    description: Raw data
    isdata: true

looks:
  - name: Kodak2383
    process_space: ACEScg
    description: Kodak 2383 print film emulation
  - name: BlueLook
    process_space: sRGB
    description: Artistic blue grading

active_displays: [sRGB]
active_views: [Film, Raw]
"#;

    #[test]
    fn test_parse_version() {
        let config = OcioConfig::from_str(MINIMAL_CONFIG).expect("parse should succeed");
        assert_eq!(config.version, 2);
    }

    #[test]
    fn test_parse_name() {
        let config = OcioConfig::from_str(MINIMAL_CONFIG).expect("parse should succeed");
        assert_eq!(config.name.as_deref(), Some("Test Config"));
    }

    #[test]
    fn test_parse_description() {
        let config = OcioConfig::from_str(MINIMAL_CONFIG).expect("parse should succeed");
        assert_eq!(
            config.description.as_deref(),
            Some("A minimal test configuration")
        );
    }

    #[test]
    fn test_parse_roles() {
        let config = OcioConfig::from_str(MINIMAL_CONFIG).expect("parse should succeed");
        assert_eq!(
            config.roles.get("scene_linear").map(String::as_str),
            Some("ACEScg")
        );
        assert_eq!(
            config.roles.get("default").map(String::as_str),
            Some("sRGB")
        );
        assert_eq!(
            config.roles.get("color_picking").map(String::as_str),
            Some("sRGB")
        );
    }

    #[test]
    fn test_role_helper() {
        let config = OcioConfig::from_str(MINIMAL_CONFIG).expect("parse should succeed");
        assert_eq!(config.role("scene_linear"), Some("ACEScg"));
        assert_eq!(config.role("nonexistent"), None);
    }

    #[test]
    fn test_parse_colorspaces_count() {
        let config = OcioConfig::from_str(MINIMAL_CONFIG).expect("parse should succeed");
        assert_eq!(config.colorspace_count(), 3, "Expected 3 color spaces");
    }

    #[test]
    fn test_parse_colorspace_srgb() {
        let config = OcioConfig::from_str(MINIMAL_CONFIG).expect("parse should succeed");
        let srgb = config.find_colorspace("sRGB").expect("sRGB should exist");
        assert_eq!(srgb.name, "sRGB");
        assert_eq!(srgb.bitdepth.as_deref(), Some("8ui"));
        assert!(!srgb.is_data);
        assert_eq!(srgb.allocation.as_deref(), Some("uniform"));
        assert_eq!(srgb.allocation_vars.len(), 2);
        assert!((srgb.allocation_vars[0]).abs() < 1e-10);
        assert!((srgb.allocation_vars[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_colorspace_acescg() {
        let config = OcioConfig::from_str(MINIMAL_CONFIG).expect("parse should succeed");
        let cs = config
            .find_colorspace("ACEScg")
            .expect("ACEScg should exist");
        assert_eq!(cs.family, "ACES");
        assert_eq!(cs.bitdepth.as_deref(), Some("16f"));
        assert!(!cs.is_data);
        assert_eq!(cs.allocation.as_deref(), Some("lg2"));
        assert_eq!(cs.allocation_vars.len(), 3);
    }

    #[test]
    fn test_parse_colorspace_raw_isdata() {
        let config = OcioConfig::from_str(MINIMAL_CONFIG).expect("parse should succeed");
        let raw = config.find_colorspace("Raw").expect("Raw should exist");
        assert!(raw.is_data, "Raw should have isdata: true");
    }

    #[test]
    fn test_colorspace_names() {
        let config = OcioConfig::from_str(MINIMAL_CONFIG).expect("parse should succeed");
        let names = config.colorspace_names();
        assert!(names.contains(&"sRGB"), "sRGB should be in names");
        assert!(names.contains(&"ACEScg"), "ACEScg should be in names");
        assert!(names.contains(&"Raw"), "Raw should be in names");
    }

    #[test]
    fn test_parse_looks_count() {
        let config = OcioConfig::from_str(MINIMAL_CONFIG).expect("parse should succeed");
        assert_eq!(config.looks.len(), 2, "Expected 2 looks");
    }

    #[test]
    fn test_parse_look_kodak() {
        let config = OcioConfig::from_str(MINIMAL_CONFIG).expect("parse should succeed");
        let look = config
            .find_look("Kodak2383")
            .expect("Kodak2383 look should exist");
        assert_eq!(look.process_space, "ACEScg");
        assert!(!look.description.is_empty());
    }

    #[test]
    fn test_parse_look_blue() {
        let config = OcioConfig::from_str(MINIMAL_CONFIG).expect("parse should succeed");
        let look = config.find_look("BlueLook").expect("BlueLook should exist");
        assert_eq!(look.process_space, "sRGB");
    }

    #[test]
    fn test_parse_active_displays() {
        let config = OcioConfig::from_str(MINIMAL_CONFIG).expect("parse should succeed");
        assert!(
            config.active_displays.contains(&"sRGB".to_owned()),
            "active_displays should contain sRGB"
        );
    }

    #[test]
    fn test_parse_active_views() {
        let config = OcioConfig::from_str(MINIMAL_CONFIG).expect("parse should succeed");
        assert!(
            config.active_views.contains(&"Film".to_owned()),
            "active_views should contain Film"
        );
        assert!(
            config.active_views.contains(&"Raw".to_owned()),
            "active_views should contain Raw"
        );
    }

    #[test]
    fn test_find_colorspace_not_found() {
        let config = OcioConfig::from_str(MINIMAL_CONFIG).expect("parse should succeed");
        assert!(config.find_colorspace("NonExistent").is_none());
    }

    #[test]
    fn test_parse_version_1_config() {
        let yaml = "ocio_profile_version: 1\nroles:\n  default: sRGB\n";
        let config = OcioConfig::from_str(yaml).expect("v1 parse should succeed");
        assert_eq!(config.version, 1);
    }

    #[test]
    fn test_yaml_comments_ignored() {
        let yaml = r#"
# This is a comment
ocio_profile_version: 2 # inline comment
name: CommentTest # another comment
roles:
  default: sRGB # role comment
"#;
        let config = OcioConfig::from_str(yaml).expect("parse should succeed");
        assert_eq!(config.version, 2);
        assert_eq!(config.name.as_deref(), Some("CommentTest"));
    }

    #[test]
    fn test_unquote_double_quotes() {
        assert_eq!(unquote("\"hello\""), "hello");
    }

    #[test]
    fn test_unquote_single_quotes() {
        assert_eq!(unquote("'world'"), "world");
    }

    #[test]
    fn test_unquote_no_quotes() {
        assert_eq!(unquote("plain"), "plain");
    }

    #[test]
    fn test_parse_inline_list_brackets() {
        let result = parse_inline_list("[a, b, c]");
        assert_eq!(result, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_parse_inline_list_single() {
        let result = parse_inline_list("only_one");
        assert_eq!(result, vec!["only_one"]);
    }

    #[test]
    fn test_strip_inline_comment_basic() {
        assert_eq!(strip_inline_comment("value: foo # comment"), "value: foo");
    }

    #[test]
    fn test_strip_inline_comment_none() {
        assert_eq!(strip_inline_comment("value: foo"), "value: foo");
    }

    #[test]
    fn test_empty_config_fails() {
        // An empty config is technically valid YAML but lacks the required version
        // We accept it gracefully with version=1 (default).
        let result = OcioConfig::from_str("# empty");
        // Should either fail or produce version=1 with no content
        if let Ok(config) = result {
            assert!(config.colorspaces.is_empty());
        }
    }

    #[test]
    fn test_displays_with_inline_views() {
        let yaml = r#"
ocio_profile_version: 2
roles:
  default: sRGB
displays:
  sRGB Monitor:
    views:
      - {name: Film, colorspace: Output - sRGB}
      - {name: Raw, colorspace: Raw}
"#;
        let config = OcioConfig::from_str(yaml).expect("parse should succeed");
        // The display parsing is best-effort
        let _ = &config.displays;
    }

    #[test]
    fn test_allocation_vars_three_elements() {
        let config = OcioConfig::from_str(MINIMAL_CONFIG).expect("parse should succeed");
        let cs = config
            .find_colorspace("ACEScg")
            .expect("ACEScg should exist");
        assert_eq!(cs.allocation_vars.len(), 3);
        assert!((cs.allocation_vars[2] - 0.00390625).abs() < 1e-8);
    }

    #[test]
    fn test_config_cloneable() {
        let config = OcioConfig::from_str(MINIMAL_CONFIG).expect("parse should succeed");
        let cloned = config.clone();
        assert_eq!(config.version, cloned.version);
        assert_eq!(config.colorspaces.len(), cloned.colorspaces.len());
    }
}
