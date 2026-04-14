//! Subtitle style inheritance resolution.
//!
//! Subtitle formats such as ASS/SSA and TTML support a hierarchy where a
//! child cue can override only some properties of a parent style, inheriting
//! the remainder from the parent.  This module provides
//! [`StyleInheritance::resolve`] which merges a parent and child
//! [`SubtitleStyle`] following this rule:
//!
//! > The child value is used when it differs from the **default** style;
//! > otherwise the parent value is used.
//!
//! This means that an "unset" child field (one that was never explicitly
//! changed from the default) falls back to the parent's value.

use crate::style::{SubtitleStyle, FontWeight, FontStyle, Color, Alignment, VerticalAlignment};

/// Subtitle style inheritance resolver.
///
/// # Example
///
/// ```rust
/// use oximedia_subtitle::style::SubtitleStyle;
/// use oximedia_subtitle::style_inherit::StyleInheritance;
///
/// let parent = SubtitleStyle::default();
/// let mut child = SubtitleStyle::default();
/// child.font_size = 36.0;          // explicitly set on child
/// // primary_color not set → inherits from parent
///
/// let resolved = StyleInheritance::resolve(&parent, &child);
/// assert_eq!(resolved.font_size, 36.0);
/// ```
pub struct StyleInheritance;

impl StyleInheritance {
    /// Resolve a style by merging `parent` and `child`.
    ///
    /// For each field the child value is used only if it **differs** from
    /// [`SubtitleStyle::default()`]; otherwise the parent's value is kept.
    /// This preserves explicit child overrides while inheriting everything
    /// else from the parent.
    #[must_use]
    pub fn resolve(parent: &SubtitleStyle, child: &SubtitleStyle) -> SubtitleStyle {
        let def = SubtitleStyle::default();

        SubtitleStyle {
            font_size: if (child.font_size - def.font_size).abs() > f32::EPSILON {
                child.font_size
            } else {
                parent.font_size
            },
            font_weight: if child.font_weight != def.font_weight {
                child.font_weight
            } else {
                parent.font_weight
            },
            font_style: if child.font_style != def.font_style {
                child.font_style
            } else {
                parent.font_style
            },
            primary_color: if child.primary_color != def.primary_color {
                child.primary_color
            } else {
                parent.primary_color
            },
            secondary_color: if child.secondary_color != def.secondary_color {
                child.secondary_color
            } else {
                parent.secondary_color
            },
            outline: if child.outline != def.outline {
                child.outline
            } else {
                parent.outline
            },
            shadow: if child.shadow != def.shadow {
                child.shadow
            } else {
                parent.shadow
            },
            alignment: if child.alignment != def.alignment {
                child.alignment
            } else {
                parent.alignment
            },
            vertical_alignment: if child.vertical_alignment != def.vertical_alignment {
                child.vertical_alignment
            } else {
                parent.vertical_alignment
            },
            position: if child.position != def.position {
                child.position
            } else {
                parent.position
            },
            margin_left: if child.margin_left != def.margin_left {
                child.margin_left
            } else {
                parent.margin_left
            },
            margin_right: if child.margin_right != def.margin_right {
                child.margin_right
            } else {
                parent.margin_right
            },
            margin_top: if child.margin_top != def.margin_top {
                child.margin_top
            } else {
                parent.margin_top
            },
            margin_bottom: if child.margin_bottom != def.margin_bottom {
                child.margin_bottom
            } else {
                parent.margin_bottom
            },
            line_spacing: if (child.line_spacing - def.line_spacing).abs() > f32::EPSILON {
                child.line_spacing
            } else {
                parent.line_spacing
            },
            word_wrap: if child.word_wrap != def.word_wrap {
                child.word_wrap
            } else {
                parent.word_wrap
            },
            max_width: if child.max_width != def.max_width {
                child.max_width
            } else {
                parent.max_width
            },
            background_color: if child.background_color != def.background_color {
                child.background_color
            } else {
                parent.background_color
            },
            background_padding: if (child.background_padding - def.background_padding).abs()
                > f32::EPSILON
            {
                child.background_padding
            } else {
                parent.background_padding
            },
        }
    }

    /// Merge a slice of styles in order, each inheriting from the previous.
    ///
    /// Returns the cumulative resolved style after applying all layers.
    /// If `styles` is empty, returns [`SubtitleStyle::default()`].
    #[must_use]
    pub fn resolve_chain(styles: &[SubtitleStyle]) -> SubtitleStyle {
        match styles {
            [] => SubtitleStyle::default(),
            [single] => single.clone(),
            [first, rest @ ..] => rest
                .iter()
                .fold(first.clone(), |acc, child| Self::resolve(&acc, child)),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_child_font_size_overrides_parent() {
        let mut parent = SubtitleStyle::default();
        parent.font_size = 24.0;
        let mut child = SubtitleStyle::default();
        child.font_size = 36.0;

        let resolved = StyleInheritance::resolve(&parent, &child);
        assert!((resolved.font_size - 36.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_child_inherits_parent_font_size_when_default() {
        let mut parent = SubtitleStyle::default();
        parent.font_size = 48.0;
        let child = SubtitleStyle::default(); // font_size == default

        let resolved = StyleInheritance::resolve(&parent, &child);
        assert!((resolved.font_size - 48.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_child_bold_overrides_parent_normal() {
        let parent = SubtitleStyle::default(); // Normal weight
        let mut child = SubtitleStyle::default();
        child.font_weight = FontWeight::Bold;

        let resolved = StyleInheritance::resolve(&parent, &child);
        assert_eq!(resolved.font_weight, FontWeight::Bold);
    }

    #[test]
    fn test_child_inherits_parent_bold() {
        let mut parent = SubtitleStyle::default();
        parent.font_weight = FontWeight::Bold;
        let child = SubtitleStyle::default(); // Normal (default)

        let resolved = StyleInheritance::resolve(&parent, &child);
        assert_eq!(resolved.font_weight, FontWeight::Bold);
    }

    #[test]
    fn test_parent_and_child_same_returns_parent() {
        let parent = SubtitleStyle::default();
        let child = SubtitleStyle::default();
        let resolved = StyleInheritance::resolve(&parent, &child);
        assert_eq!(resolved, parent);
    }

    #[test]
    fn test_resolve_chain_single() {
        let mut s = SubtitleStyle::default();
        s.font_size = 20.0;
        let result = StyleInheritance::resolve_chain(&[s.clone()]);
        assert!((result.font_size - 20.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_resolve_chain_empty() {
        let result = StyleInheritance::resolve_chain(&[]);
        assert_eq!(result, SubtitleStyle::default());
    }

    #[test]
    fn test_resolve_chain_three_layers() {
        let mut layer0 = SubtitleStyle::default();
        layer0.font_size = 20.0;

        let mut layer1 = SubtitleStyle::default();
        layer1.font_weight = FontWeight::Bold; // override weight

        let mut layer2 = SubtitleStyle::default();
        layer2.font_size = 32.0; // override size from layer0

        let resolved = StyleInheritance::resolve_chain(&[layer0, layer1, layer2]);
        assert!((resolved.font_size - 32.0).abs() < f32::EPSILON);
        assert_eq!(resolved.font_weight, FontWeight::Bold);
    }
}
