#![allow(dead_code)]
//! Subtitle multiplexer for embedding subtitle tracks into Matroska containers.
//!
//! Provides subtitle track insertion into MKV and WebM containers,
//! supporting SRT, WebVTT, ASS/SSA, and TTML subtitle formats.
//!
//! # Architecture
//!
//! The subtitle muxer converts subtitle cues into Matroska-compatible data blocks.
//! Each subtitle format has its own serialization strategy:
//!
//! - **WebVTT**: Stored as `S_TEXT/WEBVTT` codec, with cue payloads as `BlockAdditional`.
//! - **ASS/SSA**: Stored as `S_TEXT/ASS` codec, with `[Script Info]` and `[V4+ Styles]`
//!   sections in the codec private data and individual dialogue lines in each block.
//! - **SRT**: Stored as `S_TEXT/UTF8` codec, plain text cue payloads.
//! - **TTML**: Stored as `S_TEXT/TTML` codec, XML fragments per cue.
//!
//! # Example
//!
//! ```
//! use oximedia_container::subtitle_mux::{
//!     SubtitleFormat, SubtitleCue, SubtitleTrackConfig, SubtitleMuxer,
//! };
//!
//! let config = SubtitleTrackConfig {
//!     track_id: 3,
//!     format: SubtitleFormat::WebVtt,
//!     language: "en".to_owned(),
//!     name: Some("English".to_owned()),
//!     is_default: true,
//! };
//! let mut muxer = SubtitleMuxer::new(config);
//! muxer.add_cue(SubtitleCue {
//!     start_ms: 1000,
//!     end_ms: 4000,
//!     text: "Hello, world!".to_owned(),
//!     style: None,
//! });
//! muxer.sort_cues();
//!
//! let blocks = muxer.serialize_blocks();
//! assert_eq!(blocks.len(), 1);
//! assert!(blocks[0].duration_ms > 0);
//! ```

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::fmt;

// ─── Subtitle format ───────────────────────────────────────────────────────

/// Supported subtitle format for muxing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubtitleFormat {
    /// SubRip (.srt) — stored as `S_TEXT/UTF8` in Matroska.
    Srt,
    /// WebVTT (.vtt) — stored as `S_TEXT/WEBVTT` in Matroska.
    WebVtt,
    /// Advanced SubStation Alpha (.ass) — stored as `S_TEXT/ASS` in Matroska.
    Ass,
    /// Timed Text Markup Language (.ttml) — stored as `S_TEXT/TTML` in Matroska.
    Ttml,
}

impl SubtitleFormat {
    /// Returns the Matroska codec ID string for this format.
    #[must_use]
    pub fn matroska_codec_id(&self) -> &'static str {
        match self {
            Self::Srt => "S_TEXT/UTF8",
            Self::WebVtt => "S_TEXT/WEBVTT",
            Self::Ass => "S_TEXT/ASS",
            Self::Ttml => "S_TEXT/TTML",
        }
    }

    /// Returns the MIME type for this subtitle format.
    #[must_use]
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Srt => "application/x-subrip",
            Self::WebVtt => "text/vtt",
            Self::Ass => "text/x-ssa",
            Self::Ttml => "application/ttml+xml",
        }
    }

    /// Returns the common file extension (without leading dot).
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Srt => "srt",
            Self::WebVtt => "vtt",
            Self::Ass => "ass",
            Self::Ttml => "ttml",
        }
    }
}

impl fmt::Display for SubtitleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.matroska_codec_id())
    }
}

// ─── Subtitle cue ──────────────────────────────────────────────────────────

/// A subtitle cue to be muxed into the container.
#[derive(Debug, Clone)]
pub struct SubtitleCue {
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// The subtitle text content.
    pub text: String,
    /// Optional style/positioning metadata.
    pub style: Option<String>,
}

impl SubtitleCue {
    /// Returns the duration of this cue in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Returns true if the cue is empty (no text content).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }

    /// Returns true if the cue has zero or negative duration.
    #[must_use]
    pub fn is_zero_duration(&self) -> bool {
        self.end_ms <= self.start_ms
    }

    /// Checks if this cue overlaps temporally with another cue.
    #[must_use]
    pub fn overlaps(&self, other: &SubtitleCue) -> bool {
        self.start_ms < other.end_ms && other.start_ms < self.end_ms
    }
}

// ─── ASS style ─────────────────────────────────────────────────────────────

/// An ASS/SSA style definition for subtitle rendering.
#[derive(Debug, Clone)]
pub struct AssStyle {
    /// Style name.
    pub name: String,
    /// Font name (e.g. "Arial").
    pub fontname: String,
    /// Font size in points.
    pub fontsize: u32,
    /// Primary colour as `&HAABBGGRR`.
    pub primary_colour: String,
    /// Secondary colour as `&HAABBGGRR`.
    pub secondary_colour: String,
    /// Outline colour as `&HAABBGGRR`.
    pub outline_colour: String,
    /// Back (shadow) colour as `&HAABBGGRR`.
    pub back_colour: String,
    /// Bold flag (-1 for bold, 0 for normal).
    pub bold: i32,
    /// Italic flag (-1 for italic, 0 for normal).
    pub italic: i32,
    /// Border style: 1 = outline + shadow, 3 = opaque box.
    pub border_style: u32,
    /// Outline thickness in pixels.
    pub outline: f32,
    /// Shadow distance in pixels.
    pub shadow: f32,
    /// Alignment (numpad style: 1-9).
    pub alignment: u32,
    /// Left margin in pixels.
    pub margin_l: u32,
    /// Right margin in pixels.
    pub margin_r: u32,
    /// Vertical margin in pixels.
    pub margin_v: u32,
}

impl Default for AssStyle {
    fn default() -> Self {
        Self {
            name: "Default".to_owned(),
            fontname: "Arial".to_owned(),
            fontsize: 20,
            primary_colour: "&H00FFFFFF".to_owned(),
            secondary_colour: "&H000000FF".to_owned(),
            outline_colour: "&H00000000".to_owned(),
            back_colour: "&H00000000".to_owned(),
            bold: 0,
            italic: 0,
            border_style: 1,
            outline: 2.0,
            shadow: 2.0,
            alignment: 2,
            margin_l: 10,
            margin_r: 10,
            margin_v: 10,
        }
    }
}

impl AssStyle {
    /// Serialise this style to an ASS `Style:` line.
    #[must_use]
    pub fn to_ass_line(&self) -> String {
        format!(
            "Style: {},{},{},{},{},{},{},{},{},0,{},{:.1},{:.1},{},{},{},{}",
            self.name,
            self.fontname,
            self.fontsize,
            self.primary_colour,
            self.secondary_colour,
            self.outline_colour,
            self.back_colour,
            self.bold,
            self.italic,
            self.border_style,
            self.outline,
            self.shadow,
            self.alignment,
            self.margin_l,
            self.margin_r,
            self.margin_v,
        )
    }
}

// ─── ASS script info ───────────────────────────────────────────────────────

/// ASS script metadata header.
#[derive(Debug, Clone)]
pub struct AssScriptInfo {
    /// Title of the script.
    pub title: String,
    /// Script type version.
    pub script_type: String,
    /// Playback resolution width.
    pub play_res_x: u32,
    /// Playback resolution height.
    pub play_res_y: u32,
    /// Timer speed (100.0000 = normal).
    pub timer: String,
}

impl Default for AssScriptInfo {
    fn default() -> Self {
        Self {
            title: "OxiMedia Subtitles".to_owned(),
            script_type: "v4.00+".to_owned(),
            play_res_x: 1920,
            play_res_y: 1080,
            timer: "100.0000".to_owned(),
        }
    }
}

impl AssScriptInfo {
    /// Serialise to the `[Script Info]` section text.
    #[must_use]
    pub fn to_section(&self) -> String {
        let mut out = String::with_capacity(256);
        out.push_str("[Script Info]\n");
        out.push_str(&format!("Title: {}\n", self.title));
        out.push_str(&format!("ScriptType: {}\n", self.script_type));
        out.push_str(&format!("PlayResX: {}\n", self.play_res_x));
        out.push_str(&format!("PlayResY: {}\n", self.play_res_y));
        out.push_str(&format!("Timer: {}\n", self.timer));
        out
    }
}

// ─── Serialised block ──────────────────────────────────────────────────────

/// A serialised subtitle block ready for embedding into a Matroska container.
#[derive(Debug, Clone)]
pub struct SubtitleBlock {
    /// Block start time in milliseconds (Matroska cluster-relative).
    pub start_ms: u64,
    /// Block duration in milliseconds.
    pub duration_ms: u64,
    /// Serialised payload (UTF-8 text data for the block).
    pub payload: Vec<u8>,
}

impl SubtitleBlock {
    /// Returns the payload as a UTF-8 string, if valid.
    #[must_use]
    pub fn payload_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.payload).ok()
    }

    /// Returns true if the payload is non-empty.
    #[must_use]
    pub fn is_non_empty(&self) -> bool {
        !self.payload.is_empty()
    }
}

// ─── Track config ──────────────────────────────────────────────────────────

/// Configuration for a subtitle track in the mux output.
#[derive(Debug, Clone)]
pub struct SubtitleTrackConfig {
    /// Track identifier.
    pub track_id: u32,
    /// Subtitle format.
    pub format: SubtitleFormat,
    /// Language code (BCP-47, e.g. "en", "ja").
    pub language: String,
    /// Track name / label.
    pub name: Option<String>,
    /// Whether this is the default subtitle track.
    pub is_default: bool,
}

impl SubtitleTrackConfig {
    /// Returns the Matroska codec ID for this track's format.
    #[must_use]
    pub fn codec_id(&self) -> &'static str {
        self.format.matroska_codec_id()
    }
}

// ─── WebVTT codec private data ─────────────────────────────────────────────

/// Generates WebVTT codec private data for Matroska.
///
/// The codec private data for `S_TEXT/WEBVTT` is the WebVTT file header
/// (the `WEBVTT` magic line plus optional metadata).
#[derive(Debug, Clone, Default)]
pub struct WebVttCodecPrivate {
    /// Optional header comment lines.
    pub header_comments: Vec<String>,
    /// Optional CSS style blocks.
    pub style_blocks: Vec<String>,
}

impl WebVttCodecPrivate {
    /// Creates empty codec private data.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a header comment.
    #[must_use]
    pub fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.header_comments.push(comment.into());
        self
    }

    /// Adds a CSS STYLE block.
    #[must_use]
    pub fn with_style(mut self, css: impl Into<String>) -> Self {
        self.style_blocks.push(css.into());
        self
    }

    /// Serialise to the binary codec private data.
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = String::with_capacity(128);
        out.push_str("WEBVTT\n");
        for comment in &self.header_comments {
            out.push_str(&format!("NOTE {comment}\n"));
        }
        out.push('\n');
        for style in &self.style_blocks {
            out.push_str("STYLE\n");
            out.push_str(style);
            out.push_str("\n\n");
        }
        out.into_bytes()
    }
}

// ─── ASS codec private data ───────────────────────────────────────────────

/// Generates ASS/SSA codec private data for Matroska.
///
/// The codec private data for `S_TEXT/ASS` contains the `[Script Info]` and
/// `[V4+ Styles]` sections.  Individual dialogue lines are stored per-block.
#[derive(Debug, Clone)]
pub struct AssCodecPrivate {
    /// Script info header.
    pub script_info: AssScriptInfo,
    /// Style definitions.
    pub styles: Vec<AssStyle>,
}

impl Default for AssCodecPrivate {
    fn default() -> Self {
        Self {
            script_info: AssScriptInfo::default(),
            styles: vec![AssStyle::default()],
        }
    }
}

impl AssCodecPrivate {
    /// Creates codec private data with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a style definition.
    #[must_use]
    pub fn with_style(mut self, style: AssStyle) -> Self {
        self.styles.push(style);
        self
    }

    /// Serialise to the binary codec private data.
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = String::with_capacity(512);
        out.push_str(&self.script_info.to_section());
        out.push('\n');
        out.push_str("[V4+ Styles]\n");
        out.push_str("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, StrikeOut, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV\n");
        for style in &self.styles {
            out.push_str(&style.to_ass_line());
            out.push('\n');
        }
        out.push('\n');
        out.push_str("[Events]\n");
        out.push_str(
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
        );
        out.into_bytes()
    }
}

// ─── Timestamp formatting ──────────────────────────────────────────────────

/// Formats milliseconds to SRT timestamp format `HH:MM:SS,mmm`.
#[must_use]
fn format_srt_timestamp(ms: u64) -> String {
    let hours = ms / 3_600_000;
    let minutes = (ms % 3_600_000) / 60_000;
    let seconds = (ms % 60_000) / 1000;
    let millis = ms % 1000;
    format!("{hours:02}:{minutes:02}:{seconds:02},{millis:03}")
}

/// Formats milliseconds to WebVTT timestamp format `HH:MM:SS.mmm`.
#[must_use]
fn format_webvtt_timestamp(ms: u64) -> String {
    let hours = ms / 3_600_000;
    let minutes = (ms % 3_600_000) / 60_000;
    let seconds = (ms % 60_000) / 1000;
    let millis = ms % 1000;
    format!("{hours:02}:{minutes:02}:{seconds:02}.{millis:03}")
}

/// Formats milliseconds to ASS timestamp format `H:MM:SS.cc` (centiseconds).
#[must_use]
fn format_ass_timestamp(ms: u64) -> String {
    let hours = ms / 3_600_000;
    let minutes = (ms % 3_600_000) / 60_000;
    let seconds = (ms % 60_000) / 1000;
    let centis = (ms % 1000) / 10;
    format!("{hours}:{minutes:02}:{seconds:02}.{centis:02}")
}

// ─── Block serializers ─────────────────────────────────────────────────────

/// Serialise a cue for SRT format in Matroska (`S_TEXT/UTF8`).
///
/// Each Matroska block contains just the plain text of the subtitle line.
fn serialize_srt_block(cue: &SubtitleCue) -> Vec<u8> {
    cue.text.as_bytes().to_vec()
}

/// Serialise a cue for WebVTT format in Matroska (`S_TEXT/WEBVTT`).
///
/// Matroska WebVTT blocks contain three sections separated by `\n\n`:
/// 1. Cue identifier (optional, empty here)
/// 2. Cue settings (optional)
/// 3. Cue payload text
fn serialize_webvtt_block(cue: &SubtitleCue) -> Vec<u8> {
    let mut out = String::with_capacity(cue.text.len() + 32);
    // Section 1: cue identifier (empty)
    out.push_str("\n\n");
    // Section 2: cue settings (position/alignment if style present)
    if let Some(ref style) = cue.style {
        out.push_str(style);
    }
    out.push_str("\n\n");
    // Section 3: cue payload
    out.push_str(&cue.text);
    out.into_bytes()
}

/// Serialise a cue for ASS format in Matroska (`S_TEXT/ASS`).
///
/// Each Matroska block for ASS contains a single `Dialogue:` line in a
/// simplified format: `ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text`.
fn serialize_ass_block(cue: &SubtitleCue, read_order: u32) -> Vec<u8> {
    let style_name = cue.style.as_deref().unwrap_or("Default");
    // ASS block in Matroska: stripped dialogue fields without timestamps
    // (timestamps are carried by the Matroska block header)
    let line = format!("{read_order},0,{style_name},,0,0,0,,{}", cue.text);
    line.into_bytes()
}

/// Serialise a cue for TTML format in Matroska (`S_TEXT/TTML`).
///
/// Each block contains a minimal TTML `<p>` element with timing attributes.
fn serialize_ttml_block(cue: &SubtitleCue) -> Vec<u8> {
    let begin = format_ttml_timestamp(cue.start_ms);
    let end = format_ttml_timestamp(cue.end_ms);
    let text = xml_escape(&cue.text);
    let xml = format!("<p begin=\"{begin}\" end=\"{end}\">{text}</p>");
    xml.into_bytes()
}

/// Formats milliseconds to TTML timestamp format `HH:MM:SS.mmm`.
#[must_use]
fn format_ttml_timestamp(ms: u64) -> String {
    let hours = ms / 3_600_000;
    let minutes = (ms % 3_600_000) / 60_000;
    let seconds = (ms % 60_000) / 1000;
    let millis = ms % 1000;
    format!("{hours:02}:{minutes:02}:{seconds:02}.{millis:03}")
}

/// Minimal XML entity escaping for TTML text content.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

// ─── Subtitle muxer ───────────────────────────────────────────────────────

/// Subtitle muxer that collects cues and serialises them for container embedding.
///
/// Produces [`SubtitleBlock`] records suitable for writing as Matroska
/// `BlockGroup` elements on a subtitle track.
#[derive(Debug)]
pub struct SubtitleMuxer {
    /// Track configuration.
    config: SubtitleTrackConfig,
    /// Collected cues in presentation order.
    cues: Vec<SubtitleCue>,
    /// ASS codec private data (only used for ASS format).
    ass_private: Option<AssCodecPrivate>,
    /// WebVTT codec private data (only used for WebVTT format).
    webvtt_private: Option<WebVttCodecPrivate>,
    /// Custom metadata tags for the subtitle track.
    metadata: HashMap<String, String>,
}

impl SubtitleMuxer {
    /// Creates a new subtitle muxer with the given track configuration.
    #[must_use]
    pub fn new(config: SubtitleTrackConfig) -> Self {
        Self {
            config,
            cues: Vec::new(),
            ass_private: None,
            webvtt_private: None,
            metadata: HashMap::new(),
        }
    }

    /// Sets the ASS codec private data (only meaningful for ASS format).
    pub fn set_ass_private(&mut self, private: AssCodecPrivate) {
        self.ass_private = Some(private);
    }

    /// Sets the WebVTT codec private data (only meaningful for WebVTT format).
    pub fn set_webvtt_private(&mut self, private: WebVttCodecPrivate) {
        self.webvtt_private = Some(private);
    }

    /// Adds a metadata tag to the subtitle track.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Adds a cue to the track.
    pub fn add_cue(&mut self, cue: SubtitleCue) {
        self.cues.push(cue);
    }

    /// Returns the number of cues.
    #[must_use]
    pub fn cue_count(&self) -> usize {
        self.cues.len()
    }

    /// Returns the track configuration.
    #[must_use]
    pub fn config(&self) -> &SubtitleTrackConfig {
        &self.config
    }

    /// Sorts cues by start time.
    pub fn sort_cues(&mut self) {
        self.cues.sort_by_key(|c| c.start_ms);
    }

    /// Returns a reference to all cues.
    #[must_use]
    pub fn cues(&self) -> &[SubtitleCue] {
        &self.cues
    }

    /// Removes cues with zero duration or empty text.
    pub fn remove_invalid_cues(&mut self) {
        self.cues.retain(|c| !c.is_empty() && !c.is_zero_duration());
    }

    /// Returns the total duration span of all cues in milliseconds.
    #[must_use]
    pub fn total_duration_ms(&self) -> u64 {
        let min_start = self.cues.iter().map(|c| c.start_ms).min().unwrap_or(0);
        let max_end = self.cues.iter().map(|c| c.end_ms).max().unwrap_or(0);
        max_end.saturating_sub(min_start)
    }

    /// Detects overlapping cues and returns pairs of indices that overlap.
    #[must_use]
    pub fn find_overlaps(&self) -> Vec<(usize, usize)> {
        let mut overlaps = Vec::new();
        for i in 0..self.cues.len() {
            for j in (i + 1)..self.cues.len() {
                if self.cues[i].overlaps(&self.cues[j]) {
                    overlaps.push((i, j));
                }
            }
        }
        overlaps
    }

    /// Generates the codec private data bytes for this track's format.
    ///
    /// Returns `None` for formats that do not require codec private data (SRT).
    #[must_use]
    pub fn codec_private_data(&self) -> Option<Vec<u8>> {
        match self.config.format {
            SubtitleFormat::Srt => None,
            SubtitleFormat::WebVtt => {
                let private = self
                    .webvtt_private
                    .as_ref()
                    .map_or_else(|| WebVttCodecPrivate::new().serialize(), |p| p.serialize());
                Some(private)
            }
            SubtitleFormat::Ass => {
                let private = self
                    .ass_private
                    .as_ref()
                    .map_or_else(|| AssCodecPrivate::new().serialize(), |p| p.serialize());
                Some(private)
            }
            SubtitleFormat::Ttml => {
                // TTML codec private is the document header
                let header = concat!(
                    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n",
                    "<tt xmlns=\"http://www.w3.org/ns/ttml\">\n",
                    "<body><div>\n"
                );
                Some(header.as_bytes().to_vec())
            }
        }
    }

    /// Serialises all cues into Matroska-compatible subtitle blocks.
    ///
    /// Cues are serialised according to the configured [`SubtitleFormat`].
    /// The cues should be sorted before calling this method (use `sort_cues`).
    #[must_use]
    pub fn serialize_blocks(&self) -> Vec<SubtitleBlock> {
        let mut blocks = Vec::with_capacity(self.cues.len());
        for (idx, cue) in self.cues.iter().enumerate() {
            if cue.is_empty() || cue.is_zero_duration() {
                continue;
            }
            let payload = match self.config.format {
                SubtitleFormat::Srt => serialize_srt_block(cue),
                SubtitleFormat::WebVtt => serialize_webvtt_block(cue),
                SubtitleFormat::Ass => {
                    let read_order = idx as u32;
                    serialize_ass_block(cue, read_order)
                }
                SubtitleFormat::Ttml => serialize_ttml_block(cue),
            };
            blocks.push(SubtitleBlock {
                start_ms: cue.start_ms,
                duration_ms: cue.duration_ms(),
                payload,
            });
        }
        blocks
    }

    /// Serialises all cues to a standalone subtitle file string.
    ///
    /// This is useful for extracting embedded subtitles back to a file.
    #[must_use]
    pub fn serialize_to_file_string(&self) -> String {
        match self.config.format {
            SubtitleFormat::Srt => self.serialize_srt_file(),
            SubtitleFormat::WebVtt => self.serialize_webvtt_file(),
            SubtitleFormat::Ass => self.serialize_ass_file(),
            SubtitleFormat::Ttml => self.serialize_ttml_file(),
        }
    }

    fn serialize_srt_file(&self) -> String {
        let mut out = String::with_capacity(self.cues.len() * 80);
        for (idx, cue) in self.cues.iter().enumerate() {
            if cue.is_empty() {
                continue;
            }
            out.push_str(&format!("{}\n", idx + 1));
            out.push_str(&format!(
                "{} --> {}\n",
                format_srt_timestamp(cue.start_ms),
                format_srt_timestamp(cue.end_ms)
            ));
            out.push_str(&cue.text);
            out.push_str("\n\n");
        }
        out
    }

    fn serialize_webvtt_file(&self) -> String {
        let mut out = String::with_capacity(self.cues.len() * 80 + 64);
        out.push_str("WEBVTT\n\n");
        for cue in &self.cues {
            if cue.is_empty() {
                continue;
            }
            out.push_str(&format!(
                "{} --> {}",
                format_webvtt_timestamp(cue.start_ms),
                format_webvtt_timestamp(cue.end_ms)
            ));
            if let Some(ref style) = cue.style {
                out.push(' ');
                out.push_str(style);
            }
            out.push('\n');
            out.push_str(&cue.text);
            out.push_str("\n\n");
        }
        out
    }

    fn serialize_ass_file(&self) -> String {
        let private = self.ass_private.clone().unwrap_or_default();
        let mut out = String::with_capacity(self.cues.len() * 100 + 512);
        out.push_str(&String::from_utf8_lossy(&private.serialize()));
        for (idx, cue) in self.cues.iter().enumerate() {
            if cue.is_empty() {
                continue;
            }
            let style_name = cue.style.as_deref().unwrap_or("Default");
            out.push_str(&format!(
                "Dialogue: 0,{},{},{style_name},,0,0,0,,{}\n",
                format_ass_timestamp(cue.start_ms),
                format_ass_timestamp(cue.end_ms),
                cue.text
            ));
            let _ = idx; // used for clarity only
        }
        out
    }

    fn serialize_ttml_file(&self) -> String {
        let mut out = String::with_capacity(self.cues.len() * 100 + 256);
        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        out.push_str("<tt xmlns=\"http://www.w3.org/ns/ttml\">\n");
        out.push_str("<body><div>\n");
        for cue in &self.cues {
            if cue.is_empty() {
                continue;
            }
            let begin = format_ttml_timestamp(cue.start_ms);
            let end = format_ttml_timestamp(cue.end_ms);
            let text = xml_escape(&cue.text);
            out.push_str(&format!(
                "  <p begin=\"{begin}\" end=\"{end}\">{text}</p>\n"
            ));
        }
        out.push_str("</div></body>\n");
        out.push_str("</tt>\n");
        out
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(format: SubtitleFormat) -> SubtitleTrackConfig {
        SubtitleTrackConfig {
            track_id: 1,
            format,
            language: "en".to_owned(),
            name: Some("English".to_owned()),
            is_default: true,
        }
    }

    fn sample_cues() -> Vec<SubtitleCue> {
        vec![
            SubtitleCue {
                start_ms: 1000,
                end_ms: 4000,
                text: "Hello, world!".to_owned(),
                style: None,
            },
            SubtitleCue {
                start_ms: 5000,
                end_ms: 8000,
                text: "Second line".to_owned(),
                style: None,
            },
            SubtitleCue {
                start_ms: 9000,
                end_ms: 12000,
                text: "Third line".to_owned(),
                style: Some("position:50%".to_owned()),
            },
        ]
    }

    #[test]
    fn test_subtitle_format_codec_ids() {
        assert_eq!(SubtitleFormat::Srt.matroska_codec_id(), "S_TEXT/UTF8");
        assert_eq!(SubtitleFormat::WebVtt.matroska_codec_id(), "S_TEXT/WEBVTT");
        assert_eq!(SubtitleFormat::Ass.matroska_codec_id(), "S_TEXT/ASS");
        assert_eq!(SubtitleFormat::Ttml.matroska_codec_id(), "S_TEXT/TTML");
    }

    #[test]
    fn test_subtitle_format_mime_types() {
        assert_eq!(SubtitleFormat::Srt.mime_type(), "application/x-subrip");
        assert_eq!(SubtitleFormat::WebVtt.mime_type(), "text/vtt");
        assert_eq!(SubtitleFormat::Ass.mime_type(), "text/x-ssa");
        assert_eq!(SubtitleFormat::Ttml.mime_type(), "application/ttml+xml");
    }

    #[test]
    fn test_subtitle_format_extensions() {
        assert_eq!(SubtitleFormat::Srt.extension(), "srt");
        assert_eq!(SubtitleFormat::WebVtt.extension(), "vtt");
        assert_eq!(SubtitleFormat::Ass.extension(), "ass");
        assert_eq!(SubtitleFormat::Ttml.extension(), "ttml");
    }

    #[test]
    fn test_cue_duration_and_overlap() {
        let cue1 = SubtitleCue {
            start_ms: 1000,
            end_ms: 3000,
            text: "A".to_owned(),
            style: None,
        };
        let cue2 = SubtitleCue {
            start_ms: 2000,
            end_ms: 5000,
            text: "B".to_owned(),
            style: None,
        };
        let cue3 = SubtitleCue {
            start_ms: 4000,
            end_ms: 6000,
            text: "C".to_owned(),
            style: None,
        };

        assert_eq!(cue1.duration_ms(), 2000);
        assert!(cue1.overlaps(&cue2));
        assert!(!cue1.overlaps(&cue3));
        assert!(cue2.overlaps(&cue3));
    }

    #[test]
    fn test_cue_empty_and_zero_duration() {
        let empty = SubtitleCue {
            start_ms: 0,
            end_ms: 1000,
            text: "   ".to_owned(),
            style: None,
        };
        assert!(empty.is_empty());

        let zero_dur = SubtitleCue {
            start_ms: 1000,
            end_ms: 1000,
            text: "text".to_owned(),
            style: None,
        };
        assert!(zero_dur.is_zero_duration());
    }

    #[test]
    fn test_muxer_srt_serialization() {
        let mut muxer = SubtitleMuxer::new(make_config(SubtitleFormat::Srt));
        for cue in sample_cues() {
            muxer.add_cue(cue);
        }
        muxer.sort_cues();

        let blocks = muxer.serialize_blocks();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].start_ms, 1000);
        assert_eq!(blocks[0].duration_ms, 3000);
        assert_eq!(blocks[0].payload_str(), Some("Hello, world!"));

        // SRT has no codec private data
        assert!(muxer.codec_private_data().is_none());
    }

    #[test]
    fn test_muxer_webvtt_serialization() {
        let mut muxer = SubtitleMuxer::new(make_config(SubtitleFormat::WebVtt));
        muxer.add_cue(SubtitleCue {
            start_ms: 1000,
            end_ms: 3000,
            text: "WebVTT cue".to_owned(),
            style: Some("align:center".to_owned()),
        });

        let blocks = muxer.serialize_blocks();
        assert_eq!(blocks.len(), 1);

        let payload = blocks[0].payload_str().expect("valid utf8");
        // WebVTT blocks have three sections separated by \n\n
        assert!(payload.contains("align:center"));
        assert!(payload.contains("WebVTT cue"));

        // WebVTT has codec private data
        let private = muxer.codec_private_data().expect("has private");
        let private_str = std::str::from_utf8(&private).expect("valid utf8");
        assert!(private_str.starts_with("WEBVTT"));
    }

    #[test]
    fn test_muxer_ass_serialization() {
        let mut muxer = SubtitleMuxer::new(make_config(SubtitleFormat::Ass));
        let ass_private = AssCodecPrivate::new();
        muxer.set_ass_private(ass_private);

        muxer.add_cue(SubtitleCue {
            start_ms: 0,
            end_ms: 2000,
            text: "ASS dialogue".to_owned(),
            style: Some("CustomStyle".to_owned()),
        });

        let blocks = muxer.serialize_blocks();
        assert_eq!(blocks.len(), 1);

        let payload = blocks[0].payload_str().expect("valid utf8");
        // ASS block format: ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text
        assert!(payload.contains("CustomStyle"));
        assert!(payload.contains("ASS dialogue"));
        assert!(payload.starts_with("0,0,"));

        // ASS codec private data should contain [Script Info] and [V4+ Styles]
        let private = muxer.codec_private_data().expect("has private");
        let private_str = std::str::from_utf8(&private).expect("valid utf8");
        assert!(private_str.contains("[Script Info]"));
        assert!(private_str.contains("[V4+ Styles]"));
        assert!(private_str.contains("[Events]"));
    }

    #[test]
    fn test_muxer_ttml_serialization() {
        let mut muxer = SubtitleMuxer::new(make_config(SubtitleFormat::Ttml));
        muxer.add_cue(SubtitleCue {
            start_ms: 500,
            end_ms: 2500,
            text: "TTML <bold> & text".to_owned(),
            style: None,
        });

        let blocks = muxer.serialize_blocks();
        assert_eq!(blocks.len(), 1);

        let payload = blocks[0].payload_str().expect("valid utf8");
        assert!(payload.contains("<p begin=\"00:00:00.500\""));
        assert!(payload.contains("end=\"00:00:02.500\""));
        // XML entities should be escaped
        assert!(payload.contains("&lt;bold&gt;"));
        assert!(payload.contains("&amp;"));
    }

    #[test]
    fn test_muxer_remove_invalid_cues() {
        let mut muxer = SubtitleMuxer::new(make_config(SubtitleFormat::Srt));
        muxer.add_cue(SubtitleCue {
            start_ms: 0,
            end_ms: 1000,
            text: "valid".to_owned(),
            style: None,
        });
        muxer.add_cue(SubtitleCue {
            start_ms: 1000,
            end_ms: 1000,
            text: "zero duration".to_owned(),
            style: None,
        });
        muxer.add_cue(SubtitleCue {
            start_ms: 2000,
            end_ms: 3000,
            text: "  ".to_owned(),
            style: None,
        });
        muxer.add_cue(SubtitleCue {
            start_ms: 4000,
            end_ms: 5000,
            text: "also valid".to_owned(),
            style: None,
        });

        assert_eq!(muxer.cue_count(), 4);
        muxer.remove_invalid_cues();
        assert_eq!(muxer.cue_count(), 2);
        assert_eq!(muxer.cues()[0].text, "valid");
        assert_eq!(muxer.cues()[1].text, "also valid");
    }

    #[test]
    fn test_muxer_find_overlaps() {
        let mut muxer = SubtitleMuxer::new(make_config(SubtitleFormat::Srt));
        muxer.add_cue(SubtitleCue {
            start_ms: 0,
            end_ms: 3000,
            text: "A".to_owned(),
            style: None,
        });
        muxer.add_cue(SubtitleCue {
            start_ms: 2000,
            end_ms: 5000,
            text: "B".to_owned(),
            style: None,
        });
        muxer.add_cue(SubtitleCue {
            start_ms: 6000,
            end_ms: 8000,
            text: "C".to_owned(),
            style: None,
        });

        let overlaps = muxer.find_overlaps();
        assert_eq!(overlaps.len(), 1);
        assert_eq!(overlaps[0], (0, 1));
    }

    #[test]
    fn test_muxer_total_duration() {
        let mut muxer = SubtitleMuxer::new(make_config(SubtitleFormat::Srt));
        for cue in sample_cues() {
            muxer.add_cue(cue);
        }
        assert_eq!(muxer.total_duration_ms(), 11000); // 1000 -> 12000
    }

    #[test]
    fn test_serialize_srt_file() {
        let mut muxer = SubtitleMuxer::new(make_config(SubtitleFormat::Srt));
        muxer.add_cue(SubtitleCue {
            start_ms: 1000,
            end_ms: 3000,
            text: "Line one".to_owned(),
            style: None,
        });
        muxer.add_cue(SubtitleCue {
            start_ms: 5000,
            end_ms: 7000,
            text: "Line two".to_owned(),
            style: None,
        });

        let file = muxer.serialize_to_file_string();
        assert!(file.contains("1\n00:00:01,000 --> 00:00:03,000\nLine one"));
        assert!(file.contains("2\n00:00:05,000 --> 00:00:07,000\nLine two"));
    }

    #[test]
    fn test_serialize_webvtt_file() {
        let mut muxer = SubtitleMuxer::new(make_config(SubtitleFormat::WebVtt));
        muxer.add_cue(SubtitleCue {
            start_ms: 62_500,
            end_ms: 65_000,
            text: "Hello".to_owned(),
            style: None,
        });

        let file = muxer.serialize_to_file_string();
        assert!(file.starts_with("WEBVTT\n"));
        assert!(file.contains("00:01:02.500 --> 00:01:05.000"));
    }

    #[test]
    fn test_serialize_ass_file() {
        let mut muxer = SubtitleMuxer::new(make_config(SubtitleFormat::Ass));
        muxer.add_cue(SubtitleCue {
            start_ms: 0,
            end_ms: 2000,
            text: "Hello".to_owned(),
            style: None,
        });

        let file = muxer.serialize_to_file_string();
        assert!(file.contains("[Script Info]"));
        assert!(file.contains("[V4+ Styles]"));
        assert!(file.contains("[Events]"));
        assert!(file.contains("Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,Hello"));
    }

    #[test]
    fn test_serialize_ttml_file() {
        let mut muxer = SubtitleMuxer::new(make_config(SubtitleFormat::Ttml));
        muxer.add_cue(SubtitleCue {
            start_ms: 0,
            end_ms: 1000,
            text: "Test & <value>".to_owned(),
            style: None,
        });

        let file = muxer.serialize_to_file_string();
        assert!(file.contains("xmlns=\"http://www.w3.org/ns/ttml\""));
        assert!(file.contains("&amp;"));
        assert!(file.contains("&lt;value&gt;"));
        assert!(file.contains("</tt>"));
    }

    #[test]
    fn test_ass_style_serialization() {
        let style = AssStyle::default();
        let line = style.to_ass_line();
        assert!(line.starts_with("Style: Default,Arial,20,"));
        assert!(line.contains("&H00FFFFFF"));
    }

    #[test]
    fn test_webvtt_codec_private_with_style() {
        let private = WebVttCodecPrivate::new()
            .with_comment("Generated by OxiMedia")
            .with_style("::cue { color: white; }");

        let data = private.serialize();
        let text = std::str::from_utf8(&data).expect("valid utf8");
        assert!(text.starts_with("WEBVTT\n"));
        assert!(text.contains("NOTE Generated by OxiMedia"));
        assert!(text.contains("STYLE\n::cue { color: white; }"));
    }

    #[test]
    fn test_timestamp_formatting() {
        assert_eq!(format_srt_timestamp(0), "00:00:00,000");
        assert_eq!(format_srt_timestamp(3_661_500), "01:01:01,500");
        assert_eq!(format_webvtt_timestamp(3_661_500), "01:01:01.500");
        assert_eq!(format_ass_timestamp(3_661_500), "1:01:01.50");
        assert_eq!(format_ttml_timestamp(3_661_500), "01:01:01.500");
    }

    #[test]
    fn test_subtitle_block_payload_str() {
        let block = SubtitleBlock {
            start_ms: 0,
            duration_ms: 1000,
            payload: b"Hello".to_vec(),
        };
        assert_eq!(block.payload_str(), Some("Hello"));
        assert!(block.is_non_empty());

        let empty_block = SubtitleBlock {
            start_ms: 0,
            duration_ms: 0,
            payload: vec![],
        };
        assert!(!empty_block.is_non_empty());
    }

    #[test]
    fn test_muxer_metadata() {
        let mut muxer = SubtitleMuxer::new(make_config(SubtitleFormat::Srt));
        muxer.set_metadata("source", "broadcast");
        assert_eq!(muxer.config().codec_id(), "S_TEXT/UTF8");
    }
}
