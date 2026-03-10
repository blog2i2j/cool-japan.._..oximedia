#![allow(dead_code)]
//! EDL comment parsing and management.
//!
//! CMX 3600 EDL files use lines beginning with `*` or `>>>` to carry
//! supplementary metadata about events.  This module provides
//! [`CommentType`], [`EdlComment`], and [`CommentBlock`] for parsing
//! and managing these comment lines.

use std::fmt;

/// The type of an EDL comment line.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CommentType {
    /// `* FROM CLIP NAME:` -- source clip name.
    FromClipName,
    /// `* TO CLIP NAME:` -- destination clip name.
    ToClipName,
    /// `* SOURCE FILE:` -- path or URL of the source media.
    SourceFile,
    /// `* EFFECT NAME:` -- name of an applied effect.
    EffectName,
    /// `* LOC:` -- locator / marker with a timecode and note.
    Locator,
    /// `* ASC_SOP` -- ASC CDL SOP values.
    AscSop,
    /// `* ASC_SAT` -- ASC CDL saturation value.
    AscSat,
    /// `>>> SPEED:` -- motion effect speed comment.
    Speed,
    /// Generic comment not matching a known pattern.
    Generic,
}

impl CommentType {
    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::FromClipName => "FROM CLIP NAME",
            Self::ToClipName => "TO CLIP NAME",
            Self::SourceFile => "SOURCE FILE",
            Self::EffectName => "EFFECT NAME",
            Self::Locator => "LOC",
            Self::AscSop => "ASC_SOP",
            Self::AscSat => "ASC_SAT",
            Self::Speed => "SPEED",
            Self::Generic => "COMMENT",
        }
    }

    /// The prefix used in CMX 3600 lines for this comment type.
    #[must_use]
    pub const fn prefix(&self) -> &'static str {
        match self {
            Self::FromClipName => "* FROM CLIP NAME:",
            Self::ToClipName => "* TO CLIP NAME:",
            Self::SourceFile => "* SOURCE FILE:",
            Self::EffectName => "* EFFECT NAME:",
            Self::Locator => "* LOC:",
            Self::AscSop => "* ASC_SOP",
            Self::AscSat => "* ASC_SAT",
            Self::Speed => ">>> SPEED:",
            Self::Generic => "*",
        }
    }

    /// Detect the comment type from a raw EDL line.
    #[must_use]
    pub fn detect(line: &str) -> Option<Self> {
        let trimmed = line.trim();
        if trimmed.starts_with("* FROM CLIP NAME:") {
            Some(Self::FromClipName)
        } else if trimmed.starts_with("* TO CLIP NAME:") {
            Some(Self::ToClipName)
        } else if trimmed.starts_with("* SOURCE FILE:") {
            Some(Self::SourceFile)
        } else if trimmed.starts_with("* EFFECT NAME:") {
            Some(Self::EffectName)
        } else if trimmed.starts_with("* LOC:") {
            Some(Self::Locator)
        } else if trimmed.starts_with("* ASC_SOP") {
            Some(Self::AscSop)
        } else if trimmed.starts_with("* ASC_SAT") {
            Some(Self::AscSat)
        } else if trimmed.starts_with(">>> SPEED:") {
            Some(Self::Speed)
        } else if trimmed.starts_with('*') {
            Some(Self::Generic)
        } else {
            None
        }
    }
}

impl fmt::Display for CommentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// A single parsed EDL comment.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EdlComment {
    /// Comment type.
    comment_type: CommentType,
    /// The value portion of the comment (after the prefix).
    value: String,
    /// The original raw line from the EDL file.
    raw: String,
}

impl EdlComment {
    /// Create a comment from its type and value.
    #[must_use]
    pub fn new(comment_type: CommentType, value: impl Into<String>) -> Self {
        let value = value.into();
        let raw = if comment_type == CommentType::Generic {
            format!("* {value}")
        } else {
            format!("{} {value}", comment_type.prefix())
        };
        Self {
            comment_type,
            value,
            raw,
        }
    }

    /// Parse a comment from a raw EDL line.
    #[must_use]
    pub fn parse(line: &str) -> Option<Self> {
        let ct = CommentType::detect(line)?;
        let trimmed = line.trim();
        let value = match ct {
            CommentType::Generic => trimmed.strip_prefix('*').unwrap_or("").trim().to_string(),
            _ => trimmed
                .strip_prefix(ct.prefix())
                .unwrap_or("")
                .trim()
                .to_string(),
        };
        Some(Self {
            comment_type: ct,
            value,
            raw: line.to_string(),
        })
    }

    /// Comment type.
    #[must_use]
    pub fn comment_type(&self) -> &CommentType {
        &self.comment_type
    }

    /// The value portion.
    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }

    /// The original raw line.
    #[must_use]
    pub fn raw(&self) -> &str {
        &self.raw
    }

    /// Reconstruct the EDL line from type + value.
    #[must_use]
    pub fn to_edl_line(&self) -> String {
        if self.comment_type == CommentType::Generic {
            format!("* {}", self.value)
        } else {
            format!("{} {}", self.comment_type.prefix(), self.value)
        }
    }
}

impl fmt::Display for EdlComment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_edl_line())
    }
}

/// A block of comments associated with a single EDL event.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CommentBlock {
    comments: Vec<EdlComment>,
}

impl CommentBlock {
    /// Create an empty comment block.
    #[must_use]
    pub fn new() -> Self {
        Self {
            comments: Vec::new(),
        }
    }

    /// Parse multiple lines into a comment block.
    ///
    /// Lines that are not recognised as comments are silently skipped.
    #[must_use]
    pub fn parse(lines: &[&str]) -> Self {
        let comments: Vec<EdlComment> = lines
            .iter()
            .filter_map(|line| EdlComment::parse(line))
            .collect();
        Self { comments }
    }

    /// Add a comment to the block.
    pub fn push(&mut self, comment: EdlComment) {
        self.comments.push(comment);
    }

    /// Number of comments.
    #[must_use]
    pub fn len(&self) -> usize {
        self.comments.len()
    }

    /// Whether the block is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.comments.is_empty()
    }

    /// Get all comments of a specific type.
    #[must_use]
    pub fn of_type(&self, ct: &CommentType) -> Vec<&EdlComment> {
        self.comments
            .iter()
            .filter(|c| &c.comment_type == ct)
            .collect()
    }

    /// Get the first `FROM CLIP NAME` value, if any.
    #[must_use]
    pub fn from_clip_name(&self) -> Option<&str> {
        self.of_type(&CommentType::FromClipName)
            .first()
            .map(|c| c.value.as_str())
    }

    /// Get the first `SOURCE FILE` value, if any.
    #[must_use]
    pub fn source_file(&self) -> Option<&str> {
        self.of_type(&CommentType::SourceFile)
            .first()
            .map(|c| c.value.as_str())
    }

    /// Iterator over all comments.
    pub fn iter(&self) -> impl Iterator<Item = &EdlComment> {
        self.comments.iter()
    }

    /// Render all comments as EDL lines (one per line).
    #[must_use]
    pub fn to_edl_string(&self) -> String {
        self.comments
            .iter()
            .map(|c| c.to_edl_line())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comment_type_detect_from_clip() {
        let ct = CommentType::detect("* FROM CLIP NAME: shot001.mov");
        assert_eq!(ct, Some(CommentType::FromClipName));
    }

    #[test]
    fn test_comment_type_detect_to_clip() {
        let ct = CommentType::detect("* TO CLIP NAME: shot002.mov");
        assert_eq!(ct, Some(CommentType::ToClipName));
    }

    #[test]
    fn test_comment_type_detect_source_file() {
        let ct = CommentType::detect("* SOURCE FILE: /media/clip.mxf");
        assert_eq!(ct, Some(CommentType::SourceFile));
    }

    #[test]
    fn test_comment_type_detect_effect() {
        let ct = CommentType::detect("* EFFECT NAME: CROSS DISSOLVE");
        assert_eq!(ct, Some(CommentType::EffectName));
    }

    #[test]
    fn test_comment_type_detect_speed() {
        let ct = CommentType::detect(">>> SPEED: 050.0");
        assert_eq!(ct, Some(CommentType::Speed));
    }

    #[test]
    fn test_comment_type_detect_generic() {
        let ct = CommentType::detect("* some random note");
        assert_eq!(ct, Some(CommentType::Generic));
    }

    #[test]
    fn test_comment_type_detect_none() {
        let ct = CommentType::detect("001  AX  V  C");
        assert!(ct.is_none());
    }

    #[test]
    fn test_edl_comment_parse_from_clip() {
        let c = EdlComment::parse("* FROM CLIP NAME: shot001.mov").expect("failed to parse");
        assert_eq!(*c.comment_type(), CommentType::FromClipName);
        assert_eq!(c.value(), "shot001.mov");
    }

    #[test]
    fn test_edl_comment_parse_generic() {
        let c = EdlComment::parse("* This is a note").expect("failed to parse");
        assert_eq!(*c.comment_type(), CommentType::Generic);
        assert_eq!(c.value(), "This is a note");
    }

    #[test]
    fn test_edl_comment_to_edl_line() {
        let c = EdlComment::new(CommentType::FromClipName, "shot001.mov");
        assert_eq!(c.to_edl_line(), "* FROM CLIP NAME: shot001.mov");
    }

    #[test]
    fn test_edl_comment_display() {
        let c = EdlComment::new(CommentType::Generic, "hello");
        assert_eq!(format!("{c}"), "* hello");
    }

    #[test]
    fn test_comment_block_parse() {
        let lines = vec![
            "* FROM CLIP NAME: shot001.mov",
            "* SOURCE FILE: /media/shot001.mov",
            "001  AX  V  C  ...",
            "* some note",
        ];
        let block = CommentBlock::parse(&lines);
        assert_eq!(block.len(), 3);
    }

    #[test]
    fn test_comment_block_from_clip_name() {
        let mut block = CommentBlock::new();
        block.push(EdlComment::new(CommentType::FromClipName, "interview.mov"));
        assert_eq!(block.from_clip_name(), Some("interview.mov"));
    }

    #[test]
    fn test_comment_block_source_file() {
        let mut block = CommentBlock::new();
        block.push(EdlComment::new(CommentType::SourceFile, "/media/clip.mxf"));
        assert_eq!(block.source_file(), Some("/media/clip.mxf"));
    }

    #[test]
    fn test_comment_block_of_type() {
        let mut block = CommentBlock::new();
        block.push(EdlComment::new(CommentType::Generic, "note 1"));
        block.push(EdlComment::new(CommentType::FromClipName, "clip.mov"));
        block.push(EdlComment::new(CommentType::Generic, "note 2"));
        assert_eq!(block.of_type(&CommentType::Generic).len(), 2);
        assert_eq!(block.of_type(&CommentType::FromClipName).len(), 1);
    }

    #[test]
    fn test_comment_block_to_edl_string() {
        let mut block = CommentBlock::new();
        block.push(EdlComment::new(CommentType::FromClipName, "clip.mov"));
        block.push(EdlComment::new(CommentType::Generic, "note"));
        let s = block.to_edl_string();
        assert!(s.contains("* FROM CLIP NAME: clip.mov"));
        assert!(s.contains("* note"));
    }

    #[test]
    fn test_comment_type_label() {
        assert_eq!(CommentType::FromClipName.label(), "FROM CLIP NAME");
        assert_eq!(CommentType::Speed.label(), "SPEED");
        assert_eq!(CommentType::Generic.label(), "COMMENT");
    }
}
