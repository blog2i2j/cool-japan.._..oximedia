//! OpenColorIO `.ocio` YAML configuration file parser.
//!
//! Parses the rich structure of an OCIO configuration, including colorspaces,
//! displays, views, looks, roles, and file_rules sections.  This module extends
//! the basic [`crate::ocio_config`] parser with additional OCIO v2 features
//! such as `roles`, `file_rules`, per-colorspace `encoding`, and full transform
//! chains expressed as ordered lists.
//!
//! # Example
//!
//! ```
//! use oximedia_colormgmt::ocio_parser::{parse_ocio, OcioConfig};
//!
//! let yaml = r#"
//! ocio_profile_version: 2
//! name: Studio Config
//! roles:
//!   default: sRGB
//!   scene_linear: ACEScg
//! colorspaces:
//!   - name: sRGB
//!     family: Display
//!     encoding: sdr-video
//!     isdata: false
//! active_displays: [sRGB]
//! active_views: [Film, Raw]
//! "#;
//!
//! let cfg = parse_ocio(yaml).expect("parse should succeed");
//! assert_eq!(cfg.ocio_profile_version, 2);
//! assert_eq!(cfg.colorspaces.len(), 1);
//! assert_eq!(cfg.colorspaces[0].encoding, "sdr-video");
//! assert_eq!(cfg.roles.get("scene_linear").map(|s| s.as_str()), Some("ACEScg"));
//! ```

#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

use crate::error::Result;

// ── Transform chain ──────────────────────────────────────────────────────────

/// An individual transform in an OCIO transform chain.
#[derive(Debug, Clone, PartialEq)]
pub enum OcioTransformNode {
    /// 4x4 matrix transform (row-major).
    Matrix {
        /// 16-element row-major matrix.
        matrix: [f64; 16],
        /// Optional offset (4 elements).
        offset: [f64; 4],
    },
    /// Per-channel exponent (gamma) transform.
    Exponent {
        /// Exponent per channel [R, G, B, A].
        value: [f64; 4],
    },
    /// Log-affine / log-camera transform.
    LogAffine {
        /// Log-side slope per channel.
        log_side_slope: [f64; 3],
        /// Log-side offset per channel.
        log_side_offset: [f64; 3],
        /// Linear-side slope per channel.
        lin_side_slope: [f64; 3],
        /// Linear-side offset per channel.
        lin_side_offset: [f64; 3],
    },
    /// File-based LUT transform.
    File {
        /// Path to the LUT file.
        src: String,
        /// Interpolation method (e.g. `"linear"`, `"tetrahedral"`).
        interpolation: String,
        /// Direction: `"forward"` or `"inverse"`.
        direction: String,
    },
    /// Color-space conversion transform.
    ColorSpace {
        /// Source color space.
        src: String,
        /// Destination color space.
        dst: String,
    },
    /// Range (clamp) transform.
    Range {
        /// Minimum input value.
        min_in: f64,
        /// Maximum input value.
        max_in: f64,
        /// Minimum output value.
        min_out: f64,
        /// Maximum output value.
        max_out: f64,
    },
    /// Group of transforms applied in order.
    Group {
        /// Ordered child transforms.
        children: Vec<OcioTransformNode>,
    },
}

// ── Color space ──────────────────────────────────────────────────────────────

/// An OCIO color space definition with full metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct OcioColorSpace {
    /// Color space name (unique identifier).
    pub name: String,
    /// Family group (e.g. `"ACES"`, `"Input"`, `"Display"`).
    pub family: String,
    /// Encoding type (OCIO v2): `"scene-linear"`, `"sdr-video"`, `"log"`,
    /// `"hdr-video"`, `"data"`, or custom.
    pub encoding: String,
    /// Human-readable description.
    pub description: String,
    /// Whether this space carries non-color (raw/data) values.
    pub isdata: bool,
    /// Ordered transform chain from this space to the reference space.
    pub to_reference: Vec<OcioTransformNode>,
    /// Ordered transform chain from the reference space to this space.
    pub from_reference: Vec<OcioTransformNode>,
}

impl Default for OcioColorSpace {
    fn default() -> Self {
        Self {
            name: String::new(),
            family: String::new(),
            encoding: String::new(),
            description: String::new(),
            isdata: false,
            to_reference: Vec::new(),
            from_reference: Vec::new(),
        }
    }
}

// ── Display / View ───────────────────────────────────────────────────────────

/// A view within a display configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct OcioView {
    /// View name (e.g. `"Film"`, `"Raw"`, `"Log"`).
    pub name: String,
    /// Color space used by this view.
    pub colorspace: String,
    /// Optional look applied.
    pub looks: String,
}

/// A physical display device with its available views.
#[derive(Debug, Clone, PartialEq)]
pub struct OcioDisplay {
    /// Display name (e.g. `"sRGB Monitor"`, `"ACES"`, `"P3-D65"`).
    pub name: String,
    /// List of views available on this display.
    pub views: Vec<OcioView>,
}

// ── Look ─────────────────────────────────────────────────────────────────────

/// A named look (colour correction preset).
#[derive(Debug, Clone, PartialEq)]
pub struct OcioLook {
    /// Look name.
    pub name: String,
    /// Color space in which the look is applied.
    pub process_space: String,
    /// Optional description.
    pub description: String,
    /// Transform chain for the look.
    pub transform: Vec<OcioTransformNode>,
}

// ── File rule ────────────────────────────────────────────────────────────────

/// An OCIO v2 file rule entry.
///
/// File rules map file path patterns to color spaces, allowing automatic
/// color space assignment based on file naming conventions.
#[derive(Debug, Clone, PartialEq)]
pub struct OcioFileRule {
    /// Rule name (e.g. `"Default"`, `"EXR files"`).
    pub name: String,
    /// File path glob pattern (e.g. `"*.exr"`, `"textures/*"`).
    pub pattern: String,
    /// File extension pattern (alternative to glob).
    pub extension: String,
    /// Color space to assign when matched.
    pub colorspace: String,
}

// ── Top-level config ─────────────────────────────────────────────────────────

/// A fully parsed OpenColorIO configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct OcioConfig {
    /// OCIO profile version (1 or 2).
    pub ocio_profile_version: u32,
    /// Configuration name.
    pub name: String,
    /// Configuration description.
    pub description: String,
    /// Search paths for LUT files.
    pub search_path: Vec<String>,
    /// Named roles mapping role names to color space names.
    pub roles: HashMap<String, String>,
    /// File rule entries (OCIO v2).
    pub file_rules: Vec<OcioFileRule>,
    /// Color space definitions.
    pub colorspaces: Vec<OcioColorSpace>,
    /// Display device definitions.
    pub displays: Vec<OcioDisplay>,
    /// Named looks.
    pub looks: Vec<OcioLook>,
    /// Active display list.
    pub active_displays: Vec<String>,
    /// Active view list.
    pub active_views: Vec<String>,
}

impl OcioConfig {
    /// Find a color space by exact name (case-sensitive).
    #[must_use]
    pub fn find_colorspace(&self, name: &str) -> Option<&OcioColorSpace> {
        self.colorspaces.iter().find(|cs| cs.name == name)
    }

    /// Find a look by exact name (case-sensitive).
    #[must_use]
    pub fn find_look(&self, name: &str) -> Option<&OcioLook> {
        self.looks.iter().find(|l| l.name == name)
    }

    /// Look up the color space name assigned to a given role.
    #[must_use]
    pub fn role_colorspace(&self, role: &str) -> Option<&str> {
        self.roles.get(role).map(String::as_str)
    }

    /// Find the first file rule whose pattern matches the given filename.
    #[must_use]
    pub fn match_file_rule(&self, filename: &str) -> Option<&OcioFileRule> {
        let filename_lower = filename.to_lowercase();
        self.file_rules.iter().find(|rule| {
            if rule.name == "Default" {
                return true;
            }
            if !rule.extension.is_empty() {
                let ext = format!(".{}", rule.extension.to_lowercase());
                if filename_lower.ends_with(&ext) {
                    return true;
                }
            }
            if !rule.pattern.is_empty() {
                return glob_match(&rule.pattern, filename);
            }
            false
        })
    }
}

/// Simple glob pattern matching (supports `*` wildcard only).
fn glob_match(pattern: &str, text: &str) -> bool {
    let pattern_lower = pattern.to_lowercase();
    let text_lower = text.to_lowercase();

    let parts: Vec<&str> = pattern_lower.split('*').collect();
    if parts.len() == 1 {
        return text_lower == pattern_lower;
    }

    let mut pos = 0usize;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        match text_lower[pos..].find(part) {
            Some(found) => {
                if i == 0 && found != 0 {
                    return false;
                }
                pos += found + part.len();
            }
            None => return false,
        }
    }

    if let Some(last) = parts.last() {
        if !last.is_empty() {
            return text_lower.ends_with(last);
        }
    }
    true
}

// ── Public entry point ───────────────────────────────────────────────────────

/// Parse an OpenColorIO YAML configuration string into an [`OcioConfig`].
///
/// # Errors
///
/// Returns [`ColorError::Parse`] if the YAML structure is malformed or missing
/// the required `ocio_profile_version` key.
pub fn parse_ocio(text: &str) -> Result<OcioConfig> {
    let mut parser = OcioParser::new(text);
    parser.parse()
}

// ── Parser internals ─────────────────────────────────────────────────────────

struct OcioParser<'a> {
    lines: Vec<(usize, &'a str)>,
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

    fn peek(&self) -> Option<(usize, &'a str)> {
        self.lines.get(self.pos).copied()
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    fn skip_blanks_comments(&mut self) {
        while let Some((_, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn parse(&mut self) -> Result<OcioConfig> {
        let mut cfg = OcioConfig {
            ocio_profile_version: 1,
            name: String::new(),
            description: String::new(),
            search_path: Vec::new(),
            roles: HashMap::new(),
            file_rules: Vec::new(),
            colorspaces: Vec::new(),
            displays: Vec::new(),
            looks: Vec::new(),
            active_displays: Vec::new(),
            active_views: Vec::new(),
        };

        while let Some((indent, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            let line = strip_comment(raw);

            if indent == 0 {
                if let Some((key, val)) = split_kv(line) {
                    match key {
                        "ocio_profile_version" => {
                            cfg.ocio_profile_version = val.trim().parse().unwrap_or(1);
                            self.advance();
                        }
                        "name" => {
                            cfg.name = unquote(val).to_owned();
                            self.advance();
                        }
                        "description" => {
                            cfg.description = unquote(val).to_owned();
                            self.advance();
                        }
                        "search_path" => {
                            cfg.search_path = parse_inline_list(val);
                            self.advance();
                        }
                        "roles" => {
                            self.advance();
                            self.parse_roles(&mut cfg.roles);
                        }
                        "file_rules" => {
                            self.advance();
                            self.parse_file_rules(&mut cfg.file_rules)?;
                        }
                        "colorspaces" | "color_spaces" => {
                            self.advance();
                            self.parse_colorspaces(&mut cfg.colorspaces)?;
                        }
                        "looks" => {
                            self.advance();
                            self.parse_looks(&mut cfg.looks)?;
                        }
                        "displays" => {
                            self.advance();
                            self.parse_displays(&mut cfg.displays)?;
                        }
                        "active_displays" => {
                            cfg.active_displays = parse_inline_list(val);
                            self.advance();
                            self.collect_block_list(&mut cfg.active_displays, indent + 1);
                        }
                        "active_views" => {
                            cfg.active_views = parse_inline_list(val);
                            self.advance();
                            self.collect_block_list(&mut cfg.active_views, indent + 1);
                        }
                        _ => {
                            self.advance();
                        }
                    }
                } else {
                    self.advance();
                }
            } else {
                self.advance();
            }
        }

        Ok(cfg)
    }

    // ── Roles ────────────────────────────────────────────────────────────────

    fn parse_roles(&mut self, roles: &mut HashMap<String, String>) {
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            if ind == 0 {
                break;
            }
            let line = strip_comment(raw);
            if let Some((key, val)) = split_kv(line) {
                roles.insert(key.to_owned(), unquote(val).to_owned());
            }
            self.advance();
        }
    }

    // ── File rules ───────────────────────────────────────────────────────────

    fn parse_file_rules(&mut self, rules: &mut Vec<OcioFileRule>) -> Result<()> {
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            if ind == 0 {
                break;
            }
            let line = strip_comment(raw);
            if line.starts_with("- ") || line == "-" {
                let rest = line.trim_start_matches('-').trim();
                let mut rule = OcioFileRule {
                    name: String::new(),
                    pattern: String::new(),
                    extension: String::new(),
                    colorspace: String::new(),
                };
                if let Some((k, v)) = split_kv(rest) {
                    apply_file_rule_field(&mut rule, k, v);
                }
                self.advance();
                self.parse_file_rule_fields(&mut rule, ind + 1);
                rules.push(rule);
            } else {
                self.advance();
            }
        }
        Ok(())
    }

    fn parse_file_rule_fields(&mut self, rule: &mut OcioFileRule, min_indent: usize) {
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            if ind < min_indent {
                break;
            }
            let line = strip_comment(raw);
            if line.starts_with("- ") {
                break;
            }
            if let Some((k, v)) = split_kv(line) {
                apply_file_rule_field(rule, k, v);
            }
            self.advance();
        }
    }

    // ── Colorspaces ──────────────────────────────────────────────────────────

    fn parse_colorspaces(&mut self, out: &mut Vec<OcioColorSpace>) -> Result<()> {
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            if ind == 0 {
                break;
            }
            let line = strip_comment(raw);
            if line.starts_with("- ") || line == "-" {
                let rest = line.trim_start_matches('-').trim();
                let mut cs = OcioColorSpace::default();
                if let Some((k, v)) = split_kv(rest) {
                    apply_cs_field(&mut cs, k, v);
                }
                self.advance();
                self.parse_cs_fields(&mut cs, ind + 1)?;
                if !cs.name.is_empty() {
                    out.push(cs);
                }
            } else {
                self.advance();
            }
        }
        Ok(())
    }

    fn parse_cs_fields(&mut self, cs: &mut OcioColorSpace, min_indent: usize) -> Result<()> {
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            if ind < min_indent {
                break;
            }
            let line = strip_comment(raw);
            if line.starts_with("- ") && ind <= min_indent {
                break;
            }
            if let Some((k, v)) = split_kv(line) {
                match k {
                    "from_reference" | "from_scene_reference" => {
                        self.advance();
                        if v.trim().is_empty() {
                            cs.from_reference = self.parse_transform_chain(ind + 1)?;
                        }
                        continue;
                    }
                    "to_reference" | "to_scene_reference" => {
                        self.advance();
                        if v.trim().is_empty() {
                            cs.to_reference = self.parse_transform_chain(ind + 1)?;
                        }
                        continue;
                    }
                    _ => {
                        apply_cs_field(cs, k, v);
                        self.advance();
                    }
                }
            } else {
                self.advance();
            }
        }
        Ok(())
    }

    // ── Transform chain parsing ──────────────────────────────────────────────

    fn parse_transform_chain(&mut self, min_indent: usize) -> Result<Vec<OcioTransformNode>> {
        self.skip_blanks_comments();
        let mut chain = Vec::new();

        let Some((ind, raw)) = self.peek() else {
            return Ok(chain);
        };
        if ind < min_indent {
            return Ok(chain);
        }
        let line = strip_comment(raw);

        // Check for GroupTransform
        if let Some(ttype) = detect_transform_type(line) {
            if ttype == "GroupTransform" {
                self.advance();
                return self.parse_group_children(ind + 1);
            }
            // Single transform
            self.advance();
            let fields = self.collect_kv_block(ind + 1);
            if let Some(node) = build_transform_node(ttype, &fields) {
                chain.push(node);
            }
            return Ok(chain);
        }

        // List of transforms (each as `- !<Type>`)
        while let Some((i, r)) = self.peek() {
            if r.is_empty() || r.starts_with('#') {
                self.advance();
                continue;
            }
            if i < min_indent {
                break;
            }
            let l = strip_comment(r);
            if l.starts_with("- ") {
                let rest = l.trim_start_matches('-').trim();
                if let Some(ttype) = detect_transform_type(rest) {
                    self.advance();
                    let fields = self.collect_kv_block(i + 2);
                    if let Some(node) = build_transform_node(ttype, &fields) {
                        chain.push(node);
                    }
                } else {
                    self.advance();
                }
            } else {
                break;
            }
        }

        Ok(chain)
    }

    fn parse_group_children(&mut self, min_indent: usize) -> Result<Vec<OcioTransformNode>> {
        let mut children = Vec::new();
        // Consume `children:` header if present
        if let Some((ind, raw)) = self.peek() {
            if ind >= min_indent {
                let line = strip_comment(raw);
                if let Some((k, _)) = split_kv(line) {
                    if k == "children" {
                        self.advance();
                    }
                }
            }
        }
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            if ind < min_indent {
                break;
            }
            let line = strip_comment(raw);
            if line.starts_with("- ") {
                let rest = line.trim_start_matches('-').trim();
                if let Some(ttype) = detect_transform_type(rest) {
                    self.advance();
                    let fields = self.collect_kv_block(ind + 2);
                    if let Some(node) = build_transform_node(ttype, &fields) {
                        children.push(node);
                    }
                } else {
                    self.advance();
                }
            } else {
                self.advance();
            }
        }
        Ok(children)
    }

    fn collect_kv_block(&mut self, min_indent: usize) -> HashMap<String, String> {
        let mut map = HashMap::new();
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            if ind < min_indent {
                break;
            }
            let line = strip_comment(raw);
            if line.starts_with("- ") {
                break;
            }
            if let Some((k, v)) = split_kv(line) {
                map.insert(k.to_owned(), unquote(v).to_owned());
            }
            self.advance();
        }
        map
    }

    // ── Looks ────────────────────────────────────────────────────────────────

    fn parse_looks(&mut self, out: &mut Vec<OcioLook>) -> Result<()> {
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            if ind == 0 {
                break;
            }
            let line = strip_comment(raw);
            if line.starts_with("- ") || line == "-" {
                let rest = line.trim_start_matches('-').trim();
                let mut look = OcioLook {
                    name: String::new(),
                    process_space: String::new(),
                    description: String::new(),
                    transform: Vec::new(),
                };
                if let Some((k, v)) = split_kv(rest) {
                    apply_look_field(&mut look, k, v);
                }
                self.advance();
                self.parse_look_fields(&mut look, ind + 1)?;
                if !look.name.is_empty() {
                    out.push(look);
                }
            } else {
                self.advance();
            }
        }
        Ok(())
    }

    fn parse_look_fields(&mut self, look: &mut OcioLook, min_indent: usize) -> Result<()> {
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            if ind < min_indent {
                break;
            }
            let line = strip_comment(raw);
            if line.starts_with("- ") && ind <= min_indent {
                break;
            }
            if let Some((k, v)) = split_kv(line) {
                match k {
                    "transform" => {
                        self.advance();
                        if v.trim().is_empty() {
                            look.transform = self.parse_transform_chain(ind + 1)?;
                        }
                        continue;
                    }
                    _ => apply_look_field(look, k, v),
                }
            }
            self.advance();
        }
        Ok(())
    }

    // ── Displays ─────────────────────────────────────────────────────────────

    fn parse_displays(&mut self, out: &mut Vec<OcioDisplay>) -> Result<()> {
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            if ind == 0 {
                break;
            }
            let line = strip_comment(raw);
            if !line.starts_with('-') && line.ends_with(':') {
                let name = line.trim_end_matches(':').trim().to_owned();
                self.advance();
                let mut disp = OcioDisplay {
                    name,
                    views: Vec::new(),
                };
                self.parse_display_views(&mut disp, ind + 1)?;
                out.push(disp);
            } else {
                self.advance();
            }
        }
        Ok(())
    }

    fn parse_display_views(&mut self, disp: &mut OcioDisplay, min_indent: usize) -> Result<()> {
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            if ind < min_indent {
                break;
            }
            let line = strip_comment(raw);
            if let Some((k, _)) = split_kv(line) {
                if k == "views" {
                    self.advance();
                    self.parse_view_list(&mut disp.views, ind + 1)?;
                    continue;
                }
            }
            self.advance();
        }
        Ok(())
    }

    fn parse_view_list(&mut self, out: &mut Vec<OcioView>, min_indent: usize) -> Result<()> {
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            if ind < min_indent {
                break;
            }
            let line = strip_comment(raw);
            if line.starts_with("- ") {
                if let Some(view) = parse_view_inline(line) {
                    out.push(view);
                } else {
                    let rest = line.trim_start_matches('-').trim();
                    let mut view = OcioView {
                        name: String::new(),
                        colorspace: String::new(),
                        looks: String::new(),
                    };
                    if let Some((k, v)) = split_kv(rest) {
                        apply_view_field(&mut view, k, v);
                    }
                    self.advance();
                    while let Some((i2, r2)) = self.peek() {
                        if r2.is_empty() || r2.starts_with('#') {
                            self.advance();
                            continue;
                        }
                        if i2 < ind + 2 {
                            break;
                        }
                        let l2 = strip_comment(r2);
                        if let Some((k2, v2)) = split_kv(l2) {
                            apply_view_field(&mut view, k2, v2);
                        }
                        self.advance();
                    }
                    if !view.name.is_empty() {
                        out.push(view);
                    }
                    continue;
                }
            }
            self.advance();
        }
        Ok(())
    }

    fn collect_block_list(&mut self, list: &mut Vec<String>, min_indent: usize) {
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            if ind < min_indent {
                break;
            }
            let line = strip_comment(raw);
            if line.starts_with('-') {
                let item = line.trim_start_matches('-').trim();
                if !item.is_empty() {
                    list.push(unquote(item).to_owned());
                }
                self.advance();
            } else {
                break;
            }
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn strip_comment(s: &str) -> &str {
    let mut in_single = false;
    let mut in_double = false;
    let bytes = s.as_bytes();
    for i in 0..bytes.len() {
        match bytes[i] {
            b'\'' if !in_double => in_single = !in_single,
            b'"' if !in_single => in_double = !in_double,
            b'#' if !in_single && !in_double && i > 0 && bytes[i - 1] == b' ' => {
                return s[..i].trim_end();
            }
            _ => {}
        }
    }
    s
}

fn split_kv(s: &str) -> Option<(&str, &str)> {
    let idx = s.find(':')?;
    let key = s[..idx].trim();
    if key.is_empty() {
        return None;
    }
    Some((key, s[idx + 1..].trim()))
}

fn unquote(s: &str) -> &str {
    let s = s.trim();
    if s.len() >= 2 {
        let b = s.as_bytes();
        if (b[0] == b'"' && b[s.len() - 1] == b'"') || (b[0] == b'\'' && b[s.len() - 1] == b'\'')
        {
            return &s[1..s.len() - 1];
        }
    }
    s
}

fn parse_inline_list(s: &str) -> Vec<String> {
    let s = s.trim();
    if s.starts_with('[') && s.ends_with(']') {
        s[1..s.len() - 1]
            .split(',')
            .map(|item| unquote(item.trim()).to_owned())
            .filter(|x| !x.is_empty())
            .collect()
    } else if !s.is_empty() {
        vec![unquote(s).to_owned()]
    } else {
        Vec::new()
    }
}

fn apply_cs_field(cs: &mut OcioColorSpace, key: &str, val: &str) {
    let v = unquote(val);
    match key {
        "name" => cs.name = v.to_owned(),
        "family" => cs.family = v.to_owned(),
        "encoding" => cs.encoding = v.to_owned(),
        "description" => cs.description = v.to_owned(),
        "isdata" | "is_data" => {
            cs.isdata = matches!(v.to_lowercase().as_str(), "true" | "yes" | "1");
        }
        _ => {}
    }
}

fn apply_look_field(look: &mut OcioLook, key: &str, val: &str) {
    let v = unquote(val);
    match key {
        "name" => look.name = v.to_owned(),
        "process_space" => look.process_space = v.to_owned(),
        "description" => look.description = v.to_owned(),
        _ => {}
    }
}

fn apply_view_field(view: &mut OcioView, key: &str, val: &str) {
    let v = unquote(val);
    match key {
        "name" => view.name = v.to_owned(),
        "colorspace" => view.colorspace = v.to_owned(),
        "looks" => view.looks = v.to_owned(),
        _ => {}
    }
}

fn apply_file_rule_field(rule: &mut OcioFileRule, key: &str, val: &str) {
    let v = unquote(val);
    match key {
        "name" => rule.name = v.to_owned(),
        "pattern" => rule.pattern = v.to_owned(),
        "extension" => rule.extension = v.to_owned(),
        "colorspace" => rule.colorspace = v.to_owned(),
        _ => {}
    }
}

fn detect_transform_type(s: &str) -> Option<&str> {
    if s.starts_with("!<") {
        let end = s.find('>')?;
        return Some(&s[2..end]);
    }
    let known = [
        "MatrixTransform",
        "ExponentTransform",
        "LogAffineTransform",
        "LogCameraTransform",
        "FileTransform",
        "ColorSpaceTransform",
        "RangeTransform",
        "GroupTransform",
    ];
    known.iter().find(|&&t| s.starts_with(t)).copied()
}

fn build_transform_node(ttype: &str, fields: &HashMap<String, String>) -> Option<OcioTransformNode> {
    match ttype {
        "MatrixTransform" => {
            let matrix = parse_f64_array16(fields.get("matrix").map(String::as_str).unwrap_or(""));
            let offset = parse_f64_array4(fields.get("offset").map(String::as_str).unwrap_or(""));
            Some(OcioTransformNode::Matrix { matrix, offset })
        }
        "ExponentTransform" => {
            let value = parse_f64_array4(fields.get("value").map(String::as_str).unwrap_or(""));
            Some(OcioTransformNode::Exponent { value })
        }
        "LogAffineTransform" | "LogCameraTransform" => {
            let log_side_slope =
                parse_f64_array3(fields.get("log_side_slope").map(String::as_str).unwrap_or(""));
            let log_side_offset =
                parse_f64_array3(fields.get("log_side_offset").map(String::as_str).unwrap_or(""));
            let lin_side_slope =
                parse_f64_array3(fields.get("lin_side_slope").map(String::as_str).unwrap_or(""));
            let lin_side_offset =
                parse_f64_array3(fields.get("lin_side_offset").map(String::as_str).unwrap_or(""));
            Some(OcioTransformNode::LogAffine {
                log_side_slope,
                log_side_offset,
                lin_side_slope,
                lin_side_offset,
            })
        }
        "FileTransform" => {
            let src = fields.get("src").cloned().unwrap_or_default();
            let interpolation = fields.get("interpolation").cloned().unwrap_or_default();
            let direction = fields.get("direction").cloned().unwrap_or_default();
            Some(OcioTransformNode::File {
                src,
                interpolation,
                direction,
            })
        }
        "ColorSpaceTransform" => {
            let src = fields.get("src").cloned().unwrap_or_default();
            let dst = fields.get("dst").cloned().unwrap_or_default();
            Some(OcioTransformNode::ColorSpace { src, dst })
        }
        "RangeTransform" => {
            let min_in: f64 = fields
                .get("min_in_value")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            let max_in: f64 = fields
                .get("max_in_value")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0);
            let min_out: f64 = fields
                .get("min_out_value")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            let max_out: f64 = fields
                .get("max_out_value")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0);
            Some(OcioTransformNode::Range {
                min_in,
                max_in,
                min_out,
                max_out,
            })
        }
        _ => None,
    }
}

fn parse_f64_vec(s: &str) -> Vec<f64> {
    let s = s.trim();
    let inner = if s.starts_with('[') && s.ends_with(']') {
        &s[1..s.len() - 1]
    } else {
        s
    };
    inner
        .split(',')
        .filter_map(|tok| tok.trim().parse::<f64>().ok())
        .collect()
}

fn parse_f64_array16(s: &str) -> [f64; 16] {
    let vals = parse_f64_vec(s);
    let mut out = [0.0f64; 16];
    for (i, &v) in vals.iter().enumerate().take(16) {
        out[i] = v;
    }
    out
}

fn parse_f64_array4(s: &str) -> [f64; 4] {
    let vals = parse_f64_vec(s);
    let mut out = [0.0f64; 4];
    for (i, &v) in vals.iter().enumerate().take(4) {
        out[i] = v;
    }
    out
}

fn parse_f64_array3(s: &str) -> [f64; 3] {
    let vals = parse_f64_vec(s);
    let mut out = [1.0f64; 3];
    for (i, &v) in vals.iter().enumerate().take(3) {
        out[i] = v;
    }
    out
}

fn parse_view_inline(line: &str) -> Option<OcioView> {
    let rest = line.trim_start_matches('-').trim();
    if rest.starts_with('{') && rest.ends_with('}') {
        let inner = &rest[1..rest.len() - 1];
        let mut name = String::new();
        let mut colorspace = String::new();
        let mut looks = String::new();
        for part in inner.split(',') {
            if let Some((k, v)) = split_kv(part.trim()) {
                match k.trim() {
                    "name" => name = unquote(v).to_owned(),
                    "colorspace" => colorspace = unquote(v).to_owned(),
                    "looks" => looks = unquote(v).to_owned(),
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
    }
    None
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const FULL_YAML: &str = r#"
ocio_profile_version: 2
name: Studio ACES Config
description: ACES 1.3 studio configuration
search_path: [luts, transforms]

roles:
  default: sRGB
  scene_linear: ACEScg
  compositing_log: ACEScct
  color_timing: ACEScct
  data: Raw

file_rules:
  - name: EXR files
    extension: exr
    colorspace: ACES2065-1
  - name: Texture files
    pattern: "textures/*"
    colorspace: sRGB
  - name: Default
    colorspace: sRGB

colorspaces:
  - name: sRGB
    family: Display
    encoding: sdr-video
    description: Standard sRGB display
    isdata: false
  - name: ACEScg
    family: ACES
    encoding: scene-linear
    description: ACES CG working space
    isdata: false
  - name: Raw
    family: Utility
    encoding: data
    isdata: true

looks:
  - name: FilmLook
    process_space: ACEScg
    description: Filmic look

displays:
  sRGB Monitor:
    views:
      - {name: Film, colorspace: sRGB}
      - {name: Raw, colorspace: Raw}

active_displays: [sRGB Monitor]
active_views: [Film, Raw]
"#;

    #[test]
    fn test_parse_version() {
        let cfg = parse_ocio(FULL_YAML).expect("parse");
        assert_eq!(cfg.ocio_profile_version, 2);
    }

    #[test]
    fn test_parse_name_and_description() {
        let cfg = parse_ocio(FULL_YAML).expect("parse");
        assert_eq!(cfg.name, "Studio ACES Config");
        assert_eq!(cfg.description, "ACES 1.3 studio configuration");
    }

    #[test]
    fn test_parse_search_path() {
        let cfg = parse_ocio(FULL_YAML).expect("parse");
        assert_eq!(cfg.search_path, vec!["luts", "transforms"]);
    }

    #[test]
    fn test_parse_roles() {
        let cfg = parse_ocio(FULL_YAML).expect("parse");
        assert_eq!(cfg.roles.get("default").map(String::as_str), Some("sRGB"));
        assert_eq!(
            cfg.roles.get("scene_linear").map(String::as_str),
            Some("ACEScg")
        );
        assert_eq!(
            cfg.roles.get("compositing_log").map(String::as_str),
            Some("ACEScct")
        );
        assert_eq!(cfg.roles.get("data").map(String::as_str), Some("Raw"));
        assert_eq!(cfg.roles.len(), 5);
    }

    #[test]
    fn test_parse_file_rules() {
        let cfg = parse_ocio(FULL_YAML).expect("parse");
        assert_eq!(cfg.file_rules.len(), 3);
        assert_eq!(cfg.file_rules[0].name, "EXR files");
        assert_eq!(cfg.file_rules[0].extension, "exr");
        assert_eq!(cfg.file_rules[0].colorspace, "ACES2065-1");
        assert_eq!(cfg.file_rules[1].pattern, "textures/*");
        assert_eq!(cfg.file_rules[2].name, "Default");
    }

    #[test]
    fn test_parse_colorspaces_with_encoding() {
        let cfg = parse_ocio(FULL_YAML).expect("parse");
        assert_eq!(cfg.colorspaces.len(), 3);
        let srgb = cfg.find_colorspace("sRGB").expect("sRGB");
        assert_eq!(srgb.encoding, "sdr-video");
        assert_eq!(srgb.family, "Display");
        assert!(!srgb.isdata);

        let acescg = cfg.find_colorspace("ACEScg").expect("ACEScg");
        assert_eq!(acescg.encoding, "scene-linear");

        let raw = cfg.find_colorspace("Raw").expect("Raw");
        assert!(raw.isdata);
        assert_eq!(raw.encoding, "data");
    }

    #[test]
    fn test_parse_looks() {
        let cfg = parse_ocio(FULL_YAML).expect("parse");
        assert_eq!(cfg.looks.len(), 1);
        let look = cfg.find_look("FilmLook").expect("FilmLook");
        assert_eq!(look.process_space, "ACEScg");
        assert_eq!(look.description, "Filmic look");
    }

    #[test]
    fn test_parse_displays_with_views() {
        let cfg = parse_ocio(FULL_YAML).expect("parse");
        assert_eq!(cfg.displays.len(), 1);
        assert_eq!(cfg.displays[0].name, "sRGB Monitor");
        assert_eq!(cfg.displays[0].views.len(), 2);
        assert_eq!(cfg.displays[0].views[0].name, "Film");
        assert_eq!(cfg.displays[0].views[0].colorspace, "sRGB");
        assert_eq!(cfg.displays[0].views[1].name, "Raw");
    }

    #[test]
    fn test_parse_active_displays_and_views() {
        let cfg = parse_ocio(FULL_YAML).expect("parse");
        assert_eq!(cfg.active_displays, vec!["sRGB Monitor"]);
        assert_eq!(cfg.active_views, vec!["Film", "Raw"]);
    }

    #[test]
    fn test_role_colorspace_lookup() {
        let cfg = parse_ocio(FULL_YAML).expect("parse");
        assert_eq!(cfg.role_colorspace("scene_linear"), Some("ACEScg"));
        assert_eq!(cfg.role_colorspace("nonexistent"), None);
    }

    #[test]
    fn test_match_file_rule_by_extension() {
        let cfg = parse_ocio(FULL_YAML).expect("parse");
        let rule = cfg.match_file_rule("render_001.exr").expect("match exr");
        assert_eq!(rule.colorspace, "ACES2065-1");
    }

    #[test]
    fn test_match_file_rule_by_pattern() {
        let cfg = parse_ocio(FULL_YAML).expect("parse");
        let rule = cfg.match_file_rule("textures/wood.png").expect("match textures");
        assert_eq!(rule.colorspace, "sRGB");
    }

    #[test]
    fn test_match_file_rule_default() {
        let cfg = parse_ocio(FULL_YAML).expect("parse");
        let rule = cfg.match_file_rule("unknown_file.jpg").expect("match default");
        assert_eq!(rule.name, "Default");
    }

    #[test]
    fn test_transform_chain_parsing() {
        let yaml = r#"
ocio_profile_version: 2
colorspaces:
  - name: LinearRec709
    family: Input
    encoding: scene-linear
    isdata: false
    from_reference:
      !<MatrixTransform>
        matrix: [3.2406, -1.5372, -0.4986, 0, -0.9689, 1.8758, 0.0415, 0, 0.0557, -0.2040, 1.0570, 0, 0, 0, 0, 1]
"#;
        let cfg = parse_ocio(yaml).expect("parse");
        let cs = cfg.find_colorspace("LinearRec709").expect("cs");
        assert_eq!(cs.from_reference.len(), 1);
        assert!(matches!(
            &cs.from_reference[0],
            OcioTransformNode::Matrix { .. }
        ));
    }

    #[test]
    fn test_find_colorspace_not_found() {
        let cfg = parse_ocio(FULL_YAML).expect("parse");
        assert!(cfg.find_colorspace("NoSuch").is_none());
    }

    #[test]
    fn test_glob_match_star_ext() {
        assert!(glob_match("*.exr", "render.exr"));
        assert!(!glob_match("*.exr", "render.png"));
    }

    #[test]
    fn test_glob_match_prefix_star() {
        assert!(glob_match("textures/*", "textures/wood.png"));
        assert!(!glob_match("textures/*", "models/car.obj"));
    }

    #[test]
    fn test_empty_config() {
        let yaml = "ocio_profile_version: 1\n";
        let cfg = parse_ocio(yaml).expect("parse");
        assert_eq!(cfg.ocio_profile_version, 1);
        assert!(cfg.colorspaces.is_empty());
        assert!(cfg.roles.is_empty());
        assert!(cfg.file_rules.is_empty());
    }

    #[test]
    fn test_range_transform() {
        let yaml = r#"
ocio_profile_version: 2
colorspaces:
  - name: Clamped
    family: Utility
    encoding: sdr-video
    isdata: false
    to_reference:
      !<RangeTransform>
        min_in_value: 0.0
        max_in_value: 1.0
        min_out_value: 0.0
        max_out_value: 1.0
"#;
        let cfg = parse_ocio(yaml).expect("parse");
        let cs = cfg.find_colorspace("Clamped").expect("cs");
        assert_eq!(cs.to_reference.len(), 1);
        if let OcioTransformNode::Range {
            min_in, max_in, ..
        } = &cs.to_reference[0]
        {
            assert!((*min_in - 0.0).abs() < 1e-10);
            assert!((*max_in - 1.0).abs() < 1e-10);
        } else {
            panic!("expected RangeTransform");
        }
    }
}
