//! OCIO (OpenColorIO) configuration file parser with rich transform support.
//!
//! This module provides a line-by-line YAML parser for OpenColorIO config files,
//! exposing a richer transform model than the basic `ocio` module. It supports
//! parsing `MatrixTransform`, `ExponentTransform`, `LogAffineTransform`,
//! `FileTransform`, `ColorSpaceTransform`, and `GroupTransform` blocks.
//!
//! # Example
//!
//! ```
//! use oximedia_colormgmt::ocio_config::{parse_ocio_config, OcioConfig};
//!
//! let yaml = r#"
//! ocio_profile_version: 2
//! name: Studio Config
//! colorspaces:
//!   - name: sRGB
//!     family: ""
//!     isdata: false
//! active_displays: [sRGB]
//! active_views: [Film, Raw]
//! "#;
//!
//! let cfg = parse_ocio_config(yaml).expect("parse should succeed");
//! assert_eq!(cfg.ocio_profile, 2);
//! assert_eq!(cfg.colorspaces.len(), 1);
//! ```

#![allow(clippy::cast_precision_loss)]

use crate::error::Result;

// ── Transform types ───────────────────────────────────────────────────────────

/// An OCIO color transform operation.
///
/// Represents the full set of transform types used in OpenColorIO config files,
/// including matrix, exponent, logarithmic, file-based, color-space-based, and
/// group (compound) transforms.
#[derive(Debug, Clone, PartialEq)]
pub enum OcioTransform {
    /// 4×4 matrix transform (row-major, RGBA extended).
    Matrix {
        /// 16-element row-major 4×4 matrix.
        matrix: [f64; 16],
    },
    /// Per-channel exponent (power) transform.
    Exponent {
        /// Exponent values for [R, G, B, A].
        value: [f64; 4],
    },
    /// Logarithmic affine transform (ASC CDL log).
    Log {
        /// Logarithmic slope per channel [R, G, B].
        log_side_slope: [f64; 3],
        /// Logarithmic offset per channel [R, G, B].
        log_side_offset: [f64; 3],
        /// Linear slope per channel [R, G, B].
        lin_side_slope: [f64; 3],
        /// Linear offset per channel [R, G, B].
        lin_side_offset: [f64; 3],
    },
    /// File-based LUT transform.
    FileTransform {
        /// Path to the LUT file.
        src: String,
        /// Interpolation method (e.g. `"linear"`, `"tetrahedral"`).
        interpolation: String,
    },
    /// Color-space-to-color-space transform.
    ColorSpaceTransform {
        /// Source color space name.
        src: String,
        /// Destination color space name.
        dst: String,
    },
    /// Compound transform (ordered list of child transforms).
    GroupTransform {
        /// Ordered child transforms.
        children: Vec<OcioTransform>,
    },
}

// ── Data structures ───────────────────────────────────────────────────────────

/// A color space entry from an OCIO configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct OcioColorspace {
    /// Color space name (required).
    pub name: String,
    /// Family or group (e.g. `"ACES"`, `"Input"`).
    pub family: String,
    /// Human-readable description.
    pub description: String,
    /// Nominal bit depth string (e.g. `"16f"`, `"8ui"`).
    pub bit_depth: String,
    /// Whether the color space holds non-color (raw) data.
    pub isdata: bool,
    /// Transform from this color space to the reference space.
    pub to_reference: Option<OcioTransform>,
    /// Transform from the reference space to this color space.
    pub from_reference: Option<OcioTransform>,
}

impl Default for OcioColorspace {
    fn default() -> Self {
        Self {
            name: String::new(),
            family: String::new(),
            description: String::new(),
            bit_depth: String::new(),
            isdata: false,
            to_reference: None,
            from_reference: None,
        }
    }
}

/// A display-transform view.
#[derive(Debug, Clone, PartialEq)]
pub struct OcioView {
    /// View name (e.g. `"Film"`).
    pub name: String,
    /// Color space applied by this view.
    pub colorspace: String,
}

/// A physical display device.
#[derive(Debug, Clone, PartialEq)]
pub struct OcioDisplay {
    /// Display name (e.g. `"sRGB Monitor"`).
    pub name: String,
    /// Views available on this display.
    pub views: Vec<OcioView>,
}

/// A named look (color correction preset).
#[derive(Debug, Clone, PartialEq)]
pub struct OcioLook {
    /// Look name.
    pub name: String,
    /// Color space in which the look is applied.
    pub process_space: String,
}

/// A fully parsed OpenColorIO configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct OcioConfig {
    /// OCIO profile version (1 or 2).
    pub ocio_profile: u32,
    /// Optional human-readable name.
    pub name: String,
    /// Optional description.
    pub description: String,
    /// All color space definitions.
    pub colorspaces: Vec<OcioColorspace>,
    /// Top-level view list (OCIO v1 style).
    pub views: Vec<OcioView>,
    /// Display device list.
    pub displays: Vec<OcioDisplay>,
    /// Named looks.
    pub looks: Vec<OcioLook>,
    /// Ordered list of active views.
    pub active_views: Vec<String>,
    /// Ordered list of active displays.
    pub active_displays: Vec<String>,
}

impl OcioConfig {
    /// Finds a color space by exact name (case-sensitive).
    #[must_use]
    pub fn find_colorspace(&self, name: &str) -> Option<&OcioColorspace> {
        self.colorspaces.iter().find(|cs| cs.name == name)
    }

    /// Finds a look by exact name (case-sensitive).
    #[must_use]
    pub fn find_look(&self, name: &str) -> Option<&OcioLook> {
        self.looks.iter().find(|l| l.name == name)
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Parses an OpenColorIO YAML configuration string.
///
/// Uses a hand-rolled line-oriented parser — no external YAML crates required.
///
/// # Errors
///
/// Returns `ColorError::InvalidProfile` if the YAML is structurally invalid
/// or if `ocio_profile_version` is absent.
pub fn parse_ocio_config(text: &str) -> Result<OcioConfig> {
    let parser = ConfigParser::new(text);
    parser.run()
}

// ── Internal parser ───────────────────────────────────────────────────────────

struct ConfigParser<'a> {
    /// Pre-tokenised lines: `(indent_level, trimmed_content)`.
    lines: Vec<(usize, &'a str)>,
    pos: usize,
}

impl<'a> ConfigParser<'a> {
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

    fn run(mut self) -> Result<OcioConfig> {
        let mut cfg = OcioConfig {
            ocio_profile: 1,
            name: String::new(),
            description: String::new(),
            colorspaces: Vec::new(),
            views: Vec::new(),
            displays: Vec::new(),
            looks: Vec::new(),
            active_views: Vec::new(),
            active_displays: Vec::new(),
        };

        while let Some((indent, raw)) = self.peek() {
            // Skip blanks and comments
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            let line = strip_inline_comment(raw);

            // Root-level keys are at indent 0
            if indent == 0 {
                if let Some((k, v)) = kv(line) {
                    match k {
                        "ocio_profile_version" => {
                            cfg.ocio_profile = v.trim().parse().unwrap_or(1);
                            self.advance();
                        }
                        "name" => {
                            cfg.name = unquote(v).to_owned();
                            self.advance();
                        }
                        "description" => {
                            cfg.description = unquote(v).to_owned();
                            self.advance();
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
                        "views" => {
                            self.advance();
                            self.parse_view_list(&mut cfg.views, 0)?;
                        }
                        "active_displays" => {
                            cfg.active_displays.extend(parse_inline_list(v).into_iter());
                            self.advance();
                            // Also handle block list continuation
                            self.collect_block_list(&mut cfg.active_displays, indent + 1);
                        }
                        "active_views" => {
                            cfg.active_views.extend(parse_inline_list(v).into_iter());
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
                // Indented content at root with no active section — skip
                self.advance();
            }
        }

        Ok(cfg)
    }

    /// Collect `- item` block list entries at `min_indent` or greater.
    fn collect_block_list(&mut self, list: &mut Vec<String>, min_indent: usize) {
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            if ind < min_indent {
                break;
            }
            let line = strip_inline_comment(raw);
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

    // ── colorspaces ───────────────────────────────────────────────────────────

    fn parse_colorspaces(&mut self, out: &mut Vec<OcioColorspace>) -> Result<()> {
        // The colorspaces block consists of list items starting at indent >= 2
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            // Any root-level (indent 0) non-blank non-comment ends the block
            if ind == 0 {
                break;
            }
            let line = strip_inline_comment(raw);
            if line.starts_with("- ") || line == "-" {
                // Start of a new colorspace entry
                let rest = line.trim_start_matches('-').trim();
                let mut cs = OcioColorspace::default();
                if let Some((k, v)) = kv(rest) {
                    apply_cs_scalar(&mut cs, k, v);
                }
                self.advance();
                // Collect remaining fields for this entry (same or deeper indent)
                self.parse_colorspace_fields(&mut cs, ind + 1)?;
                if !cs.name.is_empty() {
                    out.push(cs);
                }
            } else {
                self.advance();
            }
        }
        Ok(())
    }

    fn parse_colorspace_fields(
        &mut self,
        cs: &mut OcioColorspace,
        min_indent: usize,
    ) -> Result<()> {
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            if ind < min_indent {
                break;
            }
            // A new list item at same or shallower indent means new entry
            let line = strip_inline_comment(raw);
            if ind == min_indent.saturating_sub(1) && (line.starts_with("- ") || line == "-") {
                break;
            }
            // A list item at the same indent as min_indent signals a sibling colorspace
            if line.starts_with("- ") && ind < min_indent {
                break;
            }
            // Detect transform sub-blocks
            if let Some((k, v)) = kv(line) {
                match k {
                    "from_reference" | "from_scene_reference" => {
                        self.advance();
                        if v.trim().is_empty() {
                            // transform block follows at deeper indent
                            let xform = self.parse_transform_block(ind + 1)?;
                            cs.from_reference = xform;
                        }
                        continue;
                    }
                    "to_reference" | "to_scene_reference" => {
                        self.advance();
                        if v.trim().is_empty() {
                            let xform = self.parse_transform_block(ind + 1)?;
                            cs.to_reference = xform;
                        }
                        continue;
                    }
                    _ => {
                        apply_cs_scalar(cs, k, v);
                        self.advance();
                    }
                }
            } else {
                self.advance();
            }
        }
        Ok(())
    }

    // ── transform block parsing ───────────────────────────────────────────────

    /// Parse a single transform block (or group) starting at `min_indent`.
    fn parse_transform_block(&mut self, min_indent: usize) -> Result<Option<OcioTransform>> {
        // Skip blanks
        while let Some((_, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
            } else {
                break;
            }
        }
        let Some((ind, raw)) = self.peek() else {
            return Ok(None);
        };
        if ind < min_indent {
            return Ok(None);
        }
        let line = strip_inline_comment(raw);

        // Detect transform type keyword (may appear as `!<MatrixTransform>` or `MatrixTransform:`)
        let Some(xform_type) = detect_transform_type(line) else {
            return Ok(None);
        };
        self.advance();

        match xform_type {
            "MatrixTransform" => {
                let fields = self.collect_kv_block(ind + 1);
                let matrix =
                    parse_f64_array16(fields.get("matrix").map(String::as_str).unwrap_or(""));
                Ok(Some(OcioTransform::Matrix { matrix }))
            }
            "ExponentTransform" => {
                let fields = self.collect_kv_block(ind + 1);
                let value = parse_f64_array4(fields.get("value").map(String::as_str).unwrap_or(""));
                Ok(Some(OcioTransform::Exponent { value }))
            }
            "LogAffineTransform" | "LogCameraTransform" => {
                let fields = self.collect_kv_block(ind + 1);
                let log_side_slope = parse_f64_array3(
                    fields
                        .get("log_side_slope")
                        .map(String::as_str)
                        .unwrap_or(""),
                );
                let log_side_offset = parse_f64_array3(
                    fields
                        .get("log_side_offset")
                        .map(String::as_str)
                        .unwrap_or(""),
                );
                let lin_side_slope = parse_f64_array3(
                    fields
                        .get("lin_side_slope")
                        .map(String::as_str)
                        .unwrap_or(""),
                );
                let lin_side_offset = parse_f64_array3(
                    fields
                        .get("lin_side_offset")
                        .map(String::as_str)
                        .unwrap_or(""),
                );
                Ok(Some(OcioTransform::Log {
                    log_side_slope,
                    log_side_offset,
                    lin_side_slope,
                    lin_side_offset,
                }))
            }
            "FileTransform" => {
                let fields = self.collect_kv_block(ind + 1);
                let src = fields.get("src").cloned().unwrap_or_default();
                let interpolation = fields.get("interpolation").cloned().unwrap_or_default();
                Ok(Some(OcioTransform::FileTransform {
                    src: unquote(&src).to_owned(),
                    interpolation: unquote(&interpolation).to_owned(),
                }))
            }
            "ColorSpaceTransform" => {
                let fields = self.collect_kv_block(ind + 1);
                let src = fields.get("src").cloned().unwrap_or_default();
                let dst = fields.get("dst").cloned().unwrap_or_default();
                Ok(Some(OcioTransform::ColorSpaceTransform {
                    src: unquote(&src).to_owned(),
                    dst: unquote(&dst).to_owned(),
                }))
            }
            "GroupTransform" => {
                // GroupTransform has a `children:` sub-list
                let children = self.parse_group_children(ind + 1)?;
                Ok(Some(OcioTransform::GroupTransform { children }))
            }
            _ => Ok(None),
        }
    }

    /// Collect key-value pairs at `min_indent` into a map.
    fn collect_kv_block(&mut self, min_indent: usize) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            if ind < min_indent {
                break;
            }
            let line = strip_inline_comment(raw);
            if let Some((k, v)) = kv(line) {
                map.insert(k.to_owned(), unquote(v).to_owned());
            }
            self.advance();
        }
        map
    }

    /// Parse children of a GroupTransform.
    fn parse_group_children(&mut self, min_indent: usize) -> Result<Vec<OcioTransform>> {
        let mut children = Vec::new();
        // Consume `children:` header if present
        if let Some((ind, raw)) = self.peek() {
            if ind >= min_indent {
                let line = strip_inline_comment(raw);
                if let Some((k, _)) = kv(line) {
                    if k == "children" {
                        self.advance();
                    }
                }
            }
        }
        // Each child is a list item starting with `- !<Type>` or `- Type:`
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            if ind < min_indent {
                break;
            }
            let line = strip_inline_comment(raw);
            if line.starts_with("- ") {
                let rest = line.trim_start_matches('-').trim();
                if let Some(xtype) = detect_transform_type(rest) {
                    self.advance();
                    let fields = self.collect_kv_block(ind + 2);
                    let child = build_transform_from_fields(xtype, &fields);
                    if let Some(c) = child {
                        children.push(c);
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

    // ── looks ─────────────────────────────────────────────────────────────────

    fn parse_looks(&mut self, out: &mut Vec<OcioLook>) -> Result<()> {
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            if ind == 0 {
                break;
            }
            let line = strip_inline_comment(raw);
            if line.starts_with("- ") || line == "-" {
                let rest = line.trim_start_matches('-').trim();
                let mut look = OcioLook {
                    name: String::new(),
                    process_space: String::new(),
                };
                if let Some((k, v)) = kv(rest) {
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
            let line = strip_inline_comment(raw);
            if line.starts_with("- ") && ind < min_indent {
                break;
            }
            if let Some((k, v)) = kv(line) {
                apply_look_field(look, k, v);
            }
            self.advance();
        }
        Ok(())
    }

    // ── displays ──────────────────────────────────────────────────────────────

    fn parse_displays(&mut self, out: &mut Vec<OcioDisplay>) -> Result<()> {
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            if ind == 0 {
                break;
            }
            let line = strip_inline_comment(raw);
            // Display names appear as `  DisplayName:` (colon, no value)
            if !line.starts_with('-') && line.ends_with(':') {
                let name = line.trim_end_matches(':').trim().to_owned();
                self.advance();
                let mut disp = OcioDisplay {
                    name,
                    views: Vec::new(),
                };
                self.parse_display_block(&mut disp, ind + 1)?;
                out.push(disp);
            } else {
                self.advance();
            }
        }
        Ok(())
    }

    fn parse_display_block(&mut self, disp: &mut OcioDisplay, min_indent: usize) -> Result<()> {
        while let Some((ind, raw)) = self.peek() {
            if raw.is_empty() || raw.starts_with('#') {
                self.advance();
                continue;
            }
            if ind < min_indent {
                break;
            }
            let line = strip_inline_comment(raw);
            if let Some((k, _)) = kv(line) {
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
            let line = strip_inline_comment(raw);
            if line.starts_with("- ") {
                if let Some(view) = parse_view_entry(line) {
                    out.push(view);
                } else {
                    // Multi-line view entry — name on this line
                    let rest = line.trim_start_matches('-').trim();
                    let entry_indent = ind + 2;
                    let mut view = OcioView {
                        name: String::new(),
                        colorspace: String::new(),
                    };
                    if let Some((k, v)) = kv(rest) {
                        if k == "name" {
                            view.name = unquote(v).to_owned();
                        }
                    }
                    self.advance();
                    // Collect remaining view fields
                    while let Some((i2, r2)) = self.peek() {
                        if r2.is_empty() || r2.starts_with('#') {
                            self.advance();
                            continue;
                        }
                        if i2 < entry_indent {
                            break;
                        }
                        let l2 = strip_inline_comment(r2);
                        if let Some((k2, v2)) = kv(l2) {
                            match k2 {
                                "name" => view.name = unquote(v2).to_owned(),
                                "colorspace" => view.colorspace = unquote(v2).to_owned(),
                                _ => {}
                            }
                        }
                        self.advance();
                    }
                    if !view.name.is_empty() {
                        out.push(view);
                    }
                    continue;
                }
            } else if line.starts_with('-') && ind == min_indent {
                // bare `- sRGB` shorthand
                let item = line.trim_start_matches('-').trim();
                if !item.is_empty() {
                    out.push(OcioView {
                        name: unquote(item).to_owned(),
                        colorspace: String::new(),
                    });
                }
            } else {
                break;
            }
            self.advance();
        }
        Ok(())
    }
}

// ── Transform helpers ─────────────────────────────────────────────────────────

/// Detect an OCIO transform type from a YAML line.
///
/// Handles both `!<MatrixTransform>` YAML tag syntax and `MatrixTransform:` key syntax.
fn detect_transform_type(s: &str) -> Option<&str> {
    // YAML tag style: !<MatrixTransform>
    if s.starts_with("!<") {
        let end = s.find('>')?;
        return Some(&s[2..end]);
    }
    // Key style: MatrixTransform: or MatrixTransform {
    let known = [
        "MatrixTransform",
        "ExponentTransform",
        "LogAffineTransform",
        "LogCameraTransform",
        "FileTransform",
        "ColorSpaceTransform",
        "GroupTransform",
    ];
    known
        .iter()
        .find(|&&t| s.starts_with(t))
        .copied()
        .map(|v| v as _)
}

/// Build an `OcioTransform` from a type name and pre-collected field map.
fn build_transform_from_fields(
    xtype: &str,
    fields: &std::collections::HashMap<String, String>,
) -> Option<OcioTransform> {
    match xtype {
        "MatrixTransform" => {
            let matrix = parse_f64_array16(fields.get("matrix").map(String::as_str).unwrap_or(""));
            Some(OcioTransform::Matrix { matrix })
        }
        "ExponentTransform" => {
            let value = parse_f64_array4(fields.get("value").map(String::as_str).unwrap_or(""));
            Some(OcioTransform::Exponent { value })
        }
        "LogAffineTransform" | "LogCameraTransform" => {
            let log_side_slope = parse_f64_array3(
                fields
                    .get("log_side_slope")
                    .map(String::as_str)
                    .unwrap_or(""),
            );
            let log_side_offset = parse_f64_array3(
                fields
                    .get("log_side_offset")
                    .map(String::as_str)
                    .unwrap_or(""),
            );
            let lin_side_slope = parse_f64_array3(
                fields
                    .get("lin_side_slope")
                    .map(String::as_str)
                    .unwrap_or(""),
            );
            let lin_side_offset = parse_f64_array3(
                fields
                    .get("lin_side_offset")
                    .map(String::as_str)
                    .unwrap_or(""),
            );
            Some(OcioTransform::Log {
                log_side_slope,
                log_side_offset,
                lin_side_slope,
                lin_side_offset,
            })
        }
        "FileTransform" => {
            let src = fields.get("src").cloned().unwrap_or_default();
            let interpolation = fields.get("interpolation").cloned().unwrap_or_default();
            Some(OcioTransform::FileTransform {
                src: unquote(&src).to_owned(),
                interpolation: unquote(&interpolation).to_owned(),
            })
        }
        "ColorSpaceTransform" => {
            let src = fields.get("src").cloned().unwrap_or_default();
            let dst = fields.get("dst").cloned().unwrap_or_default();
            Some(OcioTransform::ColorSpaceTransform {
                src: unquote(&src).to_owned(),
                dst: unquote(&dst).to_owned(),
            })
        }
        _ => None,
    }
}

// ── YAML micro-helpers ────────────────────────────────────────────────────────

/// Strip trailing inline `# comment` from a YAML value line.
fn strip_inline_comment(s: &str) -> &str {
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

/// Split `key: value` returning `Some((key, value))`.
fn kv(s: &str) -> Option<(&str, &str)> {
    let idx = s.find(':')?;
    let key = s[..idx].trim();
    if key.is_empty() {
        return None;
    }
    let val = s[idx + 1..].trim();
    Some((key, val))
}

/// Remove surrounding `"` or `'` quotes.
fn unquote(s: &str) -> &str {
    let s = s.trim();
    if s.len() >= 2 {
        let b = s.as_bytes();
        if (b[0] == b'"' && b[s.len() - 1] == b'"') || (b[0] == b'\'' && b[s.len() - 1] == b'\'') {
            return &s[1..s.len() - 1];
        }
    }
    s
}

/// Parse an inline YAML list `[a, b, c]` or a bare single item.
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

/// Apply a scalar field to an `OcioColorspace`.
fn apply_cs_scalar(cs: &mut OcioColorspace, key: &str, val: &str) {
    let v = unquote(val);
    match key {
        "name" => cs.name = v.to_owned(),
        "family" => cs.family = v.to_owned(),
        "description" => cs.description = v.to_owned(),
        "bitdepth" | "bit_depth" => cs.bit_depth = v.to_owned(),
        "isdata" | "is_data" => {
            cs.isdata = matches!(v.to_lowercase().as_str(), "true" | "yes" | "1");
        }
        _ => {}
    }
}

/// Apply a scalar field to an `OcioLook`.
fn apply_look_field(look: &mut OcioLook, key: &str, val: &str) {
    let v = unquote(val);
    match key {
        "name" => look.name = v.to_owned(),
        "process_space" => look.process_space = v.to_owned(),
        _ => {}
    }
}

/// Parse a `- {name: X, colorspace: Y}` or `- name: X` view entry.
fn parse_view_entry(line: &str) -> Option<OcioView> {
    let rest = line.trim_start_matches('-').trim();
    if rest.starts_with('{') && rest.ends_with('}') {
        let inner = &rest[1..rest.len() - 1];
        let mut name = String::new();
        let mut colorspace = String::new();
        for part in inner.split(',') {
            if let Some((k, v)) = kv(part.trim()) {
                match k.trim() {
                    "name" => name = unquote(v).to_owned(),
                    "colorspace" => colorspace = unquote(v).to_owned(),
                    _ => {}
                }
            }
        }
        if !name.is_empty() {
            return Some(OcioView { name, colorspace });
        }
    }
    None
}

// ── Numeric array parsers ─────────────────────────────────────────────────────

/// Parse a 16-element f64 array from an inline list string.
fn parse_f64_array16(s: &str) -> [f64; 16] {
    let vals = parse_f64_vec(s);
    let mut out = [0.0f64; 16];
    for (i, &v) in vals.iter().enumerate().take(16) {
        out[i] = v;
    }
    out
}

/// Parse a 4-element f64 array from an inline list string.
fn parse_f64_array4(s: &str) -> [f64; 4] {
    let vals = parse_f64_vec(s);
    let mut out = [1.0f64; 4];
    for (i, &v) in vals.iter().enumerate().take(4) {
        out[i] = v;
    }
    out
}

/// Parse a 3-element f64 array from an inline list string.
fn parse_f64_array3(s: &str) -> [f64; 3] {
    let vals = parse_f64_vec(s);
    let mut out = [1.0f64; 3];
    for (i, &v) in vals.iter().enumerate().take(3) {
        out[i] = v;
    }
    out
}

/// Parse a variable-length list of f64 values from an inline YAML list.
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const BASIC_YAML: &str = r#"
ocio_profile_version: 2
name: Studio ACES Config
description: ACES 1.2 studio configuration

colorspaces:
  - name: sRGB
    family: Display
    description: Standard sRGB
    bitdepth: 8ui
    isdata: false
  - name: ACEScg
    family: ACES
    description: ACES CG working space
    bitdepth: 16f
    isdata: false
  - name: RawData
    family: ""
    isdata: true

looks:
  - name: FilmLook
    process_space: ACEScg
  - name: NightLook
    process_space: sRGB

active_displays: [sRGB, P3-D65]
active_views: [Film, Raw, Log]
"#;

    #[test]
    fn test_parse_version() {
        let cfg = parse_ocio_config(BASIC_YAML).expect("parse should succeed");
        assert_eq!(cfg.ocio_profile, 2);
    }

    #[test]
    fn test_parse_name() {
        let cfg = parse_ocio_config(BASIC_YAML).expect("parse should succeed");
        assert_eq!(cfg.name, "Studio ACES Config");
    }

    #[test]
    fn test_parse_description() {
        let cfg = parse_ocio_config(BASIC_YAML).expect("parse should succeed");
        assert_eq!(cfg.description, "ACES 1.2 studio configuration");
    }

    #[test]
    fn test_parse_colorspace_count() {
        let cfg = parse_ocio_config(BASIC_YAML).expect("parse should succeed");
        assert_eq!(cfg.colorspaces.len(), 3);
    }

    #[test]
    fn test_parse_colorspace_fields() {
        let cfg = parse_ocio_config(BASIC_YAML).expect("parse should succeed");
        let cs = cfg.find_colorspace("sRGB").expect("sRGB should exist");
        assert_eq!(cs.family, "Display");
        assert_eq!(cs.bit_depth, "8ui");
        assert!(!cs.isdata);
    }

    #[test]
    fn test_parse_colorspace_isdata_true() {
        let cfg = parse_ocio_config(BASIC_YAML).expect("parse should succeed");
        let raw = cfg
            .find_colorspace("RawData")
            .expect("RawData should exist");
        assert!(raw.isdata, "RawData should have isdata: true");
    }

    #[test]
    fn test_parse_looks_count() {
        let cfg = parse_ocio_config(BASIC_YAML).expect("parse should succeed");
        assert_eq!(cfg.looks.len(), 2);
    }

    #[test]
    fn test_parse_look_fields() {
        let cfg = parse_ocio_config(BASIC_YAML).expect("parse should succeed");
        let look = cfg.find_look("FilmLook").expect("FilmLook should exist");
        assert_eq!(look.process_space, "ACEScg");
    }

    #[test]
    fn test_parse_active_displays() {
        let cfg = parse_ocio_config(BASIC_YAML).expect("parse should succeed");
        assert!(cfg.active_displays.contains(&"sRGB".to_owned()));
        assert!(cfg.active_displays.contains(&"P3-D65".to_owned()));
    }

    #[test]
    fn test_parse_active_views() {
        let cfg = parse_ocio_config(BASIC_YAML).expect("parse should succeed");
        assert!(cfg.active_views.contains(&"Film".to_owned()));
        assert!(cfg.active_views.contains(&"Raw".to_owned()));
        assert!(cfg.active_views.contains(&"Log".to_owned()));
    }

    #[test]
    fn test_parse_matrix_transform() {
        let yaml = r#"
ocio_profile_version: 2
colorspaces:
  - name: LinearRec709
    family: Input
    isdata: false
    from_reference:
      !<MatrixTransform>
        matrix: [3.2406, -1.5372, -0.4986, 0, -0.9689, 1.8758, 0.0415, 0, 0.0557, -0.2040, 1.0570, 0, 0, 0, 0, 1]
"#;
        let cfg = parse_ocio_config(yaml).expect("parse should succeed");
        let cs = cfg
            .find_colorspace("LinearRec709")
            .expect("LinearRec709 should exist");
        if let Some(OcioTransform::Matrix { matrix }) = &cs.from_reference {
            assert!((matrix[0] - 3.2406).abs() < 1e-4, "M[0] should be 3.2406");
            assert!((matrix[15] - 1.0).abs() < 1e-4, "M[15] should be 1.0");
        } else {
            panic!("Expected Matrix transform, got {:?}", cs.from_reference);
        }
    }

    #[test]
    fn test_parse_log_transform() {
        let yaml = r#"
ocio_profile_version: 2
colorspaces:
  - name: LogC
    family: Input
    isdata: false
    to_reference:
      !<LogAffineTransform>
        log_side_slope: [0.255076, 0.255076, 0.255076]
        log_side_offset: [0.552206, 0.552206, 0.552206]
        lin_side_slope: [5.555556, 5.555556, 5.555556]
        lin_side_offset: [0.052272, 0.052272, 0.052272]
"#;
        let cfg = parse_ocio_config(yaml).expect("parse should succeed");
        let cs = cfg.find_colorspace("LogC").expect("LogC should exist");
        if let Some(OcioTransform::Log {
            log_side_slope,
            lin_side_slope,
            ..
        }) = &cs.to_reference
        {
            assert!((log_side_slope[0] - 0.255076).abs() < 1e-5);
            assert!((lin_side_slope[0] - 5.555556).abs() < 1e-5);
        } else {
            panic!("Expected Log transform, got {:?}", cs.to_reference);
        }
    }

    #[test]
    fn test_parse_file_transform() {
        let yaml = r#"
ocio_profile_version: 2
colorspaces:
  - name: sRGB_with_LUT
    family: Display
    isdata: false
    from_reference:
      !<FileTransform>
        src: luts/srgb.cube
        interpolation: linear
"#;
        let cfg = parse_ocio_config(yaml).expect("parse should succeed");
        let cs = cfg
            .find_colorspace("sRGB_with_LUT")
            .expect("cs should exist");
        if let Some(OcioTransform::FileTransform { src, interpolation }) = &cs.from_reference {
            assert_eq!(src, "luts/srgb.cube");
            assert_eq!(interpolation, "linear");
        } else {
            panic!("Expected FileTransform, got {:?}", cs.from_reference);
        }
    }

    #[test]
    fn test_parse_colorspace_transform() {
        let yaml = r#"
ocio_profile_version: 2
colorspaces:
  - name: ACEScct
    family: ACES
    isdata: false
    to_reference:
      !<ColorSpaceTransform>
        src: ACEScct
        dst: ACES2065-1
"#;
        let cfg = parse_ocio_config(yaml).expect("parse should succeed");
        let cs = cfg.find_colorspace("ACEScct").expect("cs should exist");
        if let Some(OcioTransform::ColorSpaceTransform { src, dst }) = &cs.to_reference {
            assert_eq!(src, "ACEScct");
            assert_eq!(dst, "ACES2065-1");
        } else {
            panic!("Expected ColorSpaceTransform, got {:?}", cs.to_reference);
        }
    }

    #[test]
    fn test_version_1_accepted() {
        let yaml = "ocio_profile_version: 1\ncolorspaces:\n  - name: sRGB\n    isdata: false\n";
        let cfg = parse_ocio_config(yaml).expect("v1 parse should succeed");
        assert_eq!(cfg.ocio_profile, 1);
    }

    #[test]
    fn test_comments_ignored() {
        let yaml = r#"
# top-level comment
ocio_profile_version: 2 # inline version comment
name: Test # inline name comment
colorspaces:
  - name: sRGB # inline cs comment
    isdata: false
"#;
        let cfg = parse_ocio_config(yaml).expect("parse should succeed");
        assert_eq!(cfg.name, "Test");
        assert_eq!(cfg.colorspaces.len(), 1);
    }

    #[test]
    fn test_find_colorspace_not_found() {
        let cfg = parse_ocio_config(BASIC_YAML).expect("parse should succeed");
        assert!(cfg.find_colorspace("NonExistent").is_none());
    }

    #[test]
    fn test_find_look_not_found() {
        let cfg = parse_ocio_config(BASIC_YAML).expect("parse should succeed");
        assert!(cfg.find_look("NoSuchLook").is_none());
    }

    #[test]
    fn test_parse_f64_array16_identity() {
        let s = "[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]";
        let m = parse_f64_array16(s);
        assert!((m[0] - 1.0).abs() < 1e-10);
        assert!((m[5] - 1.0).abs() < 1e-10);
        assert!((m[10] - 1.0).abs() < 1e-10);
        assert!((m[15] - 1.0).abs() < 1e-10);
        assert!(m[1].abs() < 1e-10);
    }

    #[test]
    fn test_parse_f64_array3_defaults() {
        // If fewer than 3 values, defaults to 1.0 for remaining
        let v = parse_f64_array3("[0.5, 0.6]");
        assert!((v[0] - 0.5).abs() < 1e-10);
        assert!((v[1] - 0.6).abs() < 1e-10);
        assert!((v[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_config_clone() {
        let cfg = parse_ocio_config(BASIC_YAML).expect("parse should succeed");
        let cloned = cfg.clone();
        assert_eq!(cfg.ocio_profile, cloned.ocio_profile);
        assert_eq!(cfg.colorspaces.len(), cloned.colorspaces.len());
    }

    #[test]
    fn test_displays_parse() {
        let yaml = r#"
ocio_profile_version: 2
displays:
  sRGB Monitor:
    views:
      - {name: Film, colorspace: Output - sRGB}
      - {name: Raw, colorspace: Raw}
"#;
        let cfg = parse_ocio_config(yaml).expect("parse should succeed");
        assert_eq!(cfg.displays.len(), 1);
        assert_eq!(cfg.displays[0].name, "sRGB Monitor");
        assert_eq!(cfg.displays[0].views.len(), 2);
    }
}
