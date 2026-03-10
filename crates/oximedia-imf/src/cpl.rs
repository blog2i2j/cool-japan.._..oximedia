//! Composition Playlist (CPL) - SMPTE ST 2067-3
//!
//! The CPL defines the editorial composition of an IMF package, specifying:
//! - Edit rate (timeline frame rate)
//! - Sequences (video, audio, markers, etc.)
//! - Resources within each sequence
//! - Timing information (entry point, duration, source duration)
//! - Composition timecode

use crate::{ImfError, ImfResult};
use chrono::{DateTime, Utc};
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::{Reader, Writer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::str::FromStr;
use uuid::Uuid;

/// Edit rate (rational number representing frames per second)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EditRate {
    numerator: u32,
    denominator: u32,
}

impl EditRate {
    /// Create a new edit rate
    pub fn new(numerator: u32, denominator: u32) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// Get the numerator
    pub fn numerator(&self) -> u32 {
        self.numerator
    }

    /// Get the denominator
    pub fn denominator(&self) -> u32 {
        self.denominator
    }

    /// Convert to floating point value
    pub fn as_f64(&self) -> f64 {
        f64::from(self.numerator) / f64::from(self.denominator)
    }

    /// 23.976 fps (film rate)
    pub fn fps_23_976() -> Self {
        Self::new(24000, 1001)
    }

    /// 24 fps (film rate)
    pub fn fps_24() -> Self {
        Self::new(24, 1)
    }

    /// 25 fps (PAL)
    pub fn fps_25() -> Self {
        Self::new(25, 1)
    }

    /// 29.97 fps (NTSC)
    pub fn fps_29_97() -> Self {
        Self::new(30000, 1001)
    }

    /// 30 fps
    pub fn fps_30() -> Self {
        Self::new(30, 1)
    }

    /// 50 fps (PAL progressive)
    pub fn fps_50() -> Self {
        Self::new(50, 1)
    }

    /// 59.94 fps (NTSC progressive)
    pub fn fps_59_94() -> Self {
        Self::new(60000, 1001)
    }

    /// 60 fps
    pub fn fps_60() -> Self {
        Self::new(60, 1)
    }
}

impl std::fmt::Display for EditRate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.numerator, self.denominator)
    }
}

impl FromStr for EditRate {
    type Err = ImfError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(' ').collect();
        if parts.len() != 2 {
            return Err(ImfError::InvalidEditRate(format!(
                "Expected 'numerator denominator', got '{s}'"
            )));
        }

        let numerator = parts[0]
            .parse()
            .map_err(|_| ImfError::InvalidEditRate(format!("Invalid numerator: {}", parts[0])))?;
        let denominator = parts[1]
            .parse()
            .map_err(|_| ImfError::InvalidEditRate(format!("Invalid denominator: {}", parts[1])))?;

        Ok(Self::new(numerator, denominator))
    }
}

/// Composition timecode
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositionTimecode {
    start_timecode: String,
    edit_rate: EditRate,
    drop_frame: bool,
}

impl CompositionTimecode {
    /// Create a new composition timecode
    pub fn new(start_timecode: String, edit_rate: EditRate, drop_frame: bool) -> Self {
        Self {
            start_timecode,
            edit_rate,
            drop_frame,
        }
    }

    /// Get the start timecode
    pub fn start_timecode(&self) -> &str {
        &self.start_timecode
    }

    /// Get the edit rate
    pub fn edit_rate(&self) -> EditRate {
        self.edit_rate
    }

    /// Is drop frame
    pub fn is_drop_frame(&self) -> bool {
        self.drop_frame
    }
}

/// Sequence type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SequenceType {
    /// Main image sequence (video)
    MainImage,
    /// Main audio sequence
    MainAudio,
    /// Subtitle sequence
    Subtitle,
    /// Marker sequence
    Marker,
    /// Auxiliary data
    AuxData,
    /// Unknown/custom sequence type
    Unknown,
}

impl FromStr for SequenceType {
    type Err = ImfError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "MainImageSequence" => Ok(Self::MainImage),
            "MainAudioSequence" => Ok(Self::MainAudio),
            "SubtitlesSequence" => Ok(Self::Subtitle),
            "MarkerSequence" => Ok(Self::Marker),
            "AuxDataSequence" => Ok(Self::AuxData),
            _ => Ok(Self::Unknown),
        }
    }
}

impl std::fmt::Display for SequenceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::MainImage => "MainImageSequence",
            Self::MainAudio => "MainAudioSequence",
            Self::Subtitle => "SubtitlesSequence",
            Self::Marker => "MarkerSequence",
            Self::AuxData => "AuxDataSequence",
            Self::Unknown => "UnknownSequence",
        };
        write!(f, "{s}")
    }
}

/// Resource element in a sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    id: Uuid,
    edit_rate: EditRate,
    intrinsic_duration: u64,
    entry_point: Option<u64>,
    source_duration: Option<u64>,
    repeat_count: u32,
    track_file_id: Uuid,
    source_encoding: Option<String>,
    hash: Option<String>,
}

impl Resource {
    /// Create a new resource
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: Uuid,
        edit_rate: EditRate,
        intrinsic_duration: u64,
        entry_point: Option<u64>,
        source_duration: Option<u64>,
        repeat_count: u32,
        track_file_id: Uuid,
    ) -> Self {
        Self {
            id,
            edit_rate,
            intrinsic_duration,
            entry_point,
            source_duration,
            repeat_count,
            track_file_id,
            source_encoding: None,
            hash: None,
        }
    }

    /// Get the resource ID
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Get the edit rate
    pub fn edit_rate(&self) -> EditRate {
        self.edit_rate
    }

    /// Get the intrinsic duration
    pub fn intrinsic_duration(&self) -> u64 {
        self.intrinsic_duration
    }

    /// Get the entry point
    pub fn entry_point(&self) -> Option<u64> {
        self.entry_point
    }

    /// Get the source duration
    pub fn source_duration(&self) -> Option<u64> {
        self.source_duration
    }

    /// Get the repeat count
    pub fn repeat_count(&self) -> u32 {
        self.repeat_count
    }

    /// Get the track file ID
    pub fn track_file_id(&self) -> Uuid {
        self.track_file_id
    }

    /// Get the effective duration (considering entry point and source duration)
    pub fn effective_duration(&self) -> u64 {
        self.source_duration
            .unwrap_or(self.intrinsic_duration - self.entry_point.unwrap_or(0))
            * u64::from(self.repeat_count)
    }

    /// Set source encoding
    pub fn with_source_encoding(mut self, encoding: String) -> Self {
        self.source_encoding = Some(encoding);
        self
    }

    /// Set hash
    pub fn with_hash(mut self, hash: String) -> Self {
        self.hash = Some(hash);
        self
    }
}

/// Marker resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Marker {
    id: Uuid,
    label: String,
    offset: u64,
    annotation: Option<String>,
}

impl Marker {
    /// Create a new marker
    pub fn new(id: Uuid, label: String, offset: u64) -> Self {
        Self {
            id,
            label,
            offset,
            annotation: None,
        }
    }

    /// Get the marker ID
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Get the label
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Get the offset
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Get the annotation
    pub fn annotation(&self) -> Option<&str> {
        self.annotation.as_deref()
    }

    /// Set annotation
    pub fn with_annotation(mut self, annotation: String) -> Self {
        self.annotation = Some(annotation);
        self
    }
}

/// Sequence in a composition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sequence {
    id: Uuid,
    sequence_type: SequenceType,
    track_id: Uuid,
    resources: Vec<Resource>,
    markers: Vec<Marker>,
}

impl Sequence {
    /// Create a new sequence
    pub fn new(id: Uuid, sequence_type: SequenceType, track_id: Uuid) -> Self {
        Self {
            id,
            sequence_type,
            track_id,
            resources: Vec::new(),
            markers: Vec::new(),
        }
    }

    /// Get the sequence ID
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Get the sequence type
    pub fn sequence_type(&self) -> SequenceType {
        self.sequence_type
    }

    /// Get the track ID
    pub fn track_id(&self) -> Uuid {
        self.track_id
    }

    /// Get the resources
    pub fn resources(&self) -> &[Resource] {
        &self.resources
    }

    /// Get the markers
    pub fn markers(&self) -> &[Marker] {
        &self.markers
    }

    /// Add a resource
    pub fn add_resource(&mut self, resource: Resource) {
        self.resources.push(resource);
    }

    /// Add a marker
    pub fn add_marker(&mut self, marker: Marker) {
        self.markers.push(marker);
    }

    /// Calculate total duration of the sequence
    pub fn total_duration(&self) -> u64 {
        self.resources
            .iter()
            .map(Resource::effective_duration)
            .sum()
    }
}

/// Main image sequence (video)
pub type MainImageSequence = Sequence;

/// Main audio sequence
pub type MainAudioSequence = Sequence;

/// Marker sequence
pub type MarkerSequence = Sequence;

/// Segment in a composition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    id: Uuid,
    annotation: Option<String>,
    sequences: Vec<Sequence>,
}

impl Segment {
    /// Create a new segment
    pub fn new(id: Uuid) -> Self {
        Self {
            id,
            annotation: None,
            sequences: Vec::new(),
        }
    }

    /// Get the segment ID
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Get the annotation
    pub fn annotation(&self) -> Option<&str> {
        self.annotation.as_deref()
    }

    /// Set annotation
    pub fn set_annotation(&mut self, annotation: String) {
        self.annotation = Some(annotation);
    }

    /// Get the sequences
    pub fn sequences(&self) -> &[Sequence] {
        &self.sequences
    }

    /// Add a sequence
    pub fn add_sequence(&mut self, sequence: Sequence) {
        self.sequences.push(sequence);
    }

    /// Get sequences by type
    pub fn sequences_by_type(&self, sequence_type: SequenceType) -> Vec<&Sequence> {
        self.sequences
            .iter()
            .filter(|s| s.sequence_type == sequence_type)
            .collect()
    }
}

/// Composition Playlist (CPL) - SMPTE ST 2067-3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionPlaylist {
    id: Uuid,
    annotation: Option<String>,
    issue_date: DateTime<Utc>,
    issuer: Option<String>,
    creator: Option<String>,
    content_originator: Option<String>,
    content_title: String,
    content_kind: String,
    content_version_id: Option<String>,
    edit_rate: EditRate,
    total_runtime: Option<String>,
    segments: Vec<Segment>,
    composition_timecode: Option<CompositionTimecode>,
    locale_list: Vec<String>,
    extension_properties: HashMap<String, String>,
}

impl CompositionPlaylist {
    /// Create a new CPL
    pub fn new(id: Uuid, content_title: String, edit_rate: EditRate) -> Self {
        Self {
            id,
            annotation: None,
            issue_date: Utc::now(),
            issuer: None,
            creator: None,
            content_originator: None,
            content_title,
            content_kind: "feature".to_string(),
            content_version_id: None,
            edit_rate,
            total_runtime: None,
            segments: Vec::new(),
            composition_timecode: None,
            locale_list: Vec::new(),
            extension_properties: HashMap::new(),
        }
    }

    /// Get the CPL ID
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Get the annotation
    pub fn annotation(&self) -> Option<&str> {
        self.annotation.as_deref()
    }

    /// Set annotation
    pub fn set_annotation(&mut self, annotation: String) {
        self.annotation = Some(annotation);
    }

    /// Get the issue date
    pub fn issue_date(&self) -> DateTime<Utc> {
        self.issue_date
    }

    /// Set issue date
    pub fn set_issue_date(&mut self, date: DateTime<Utc>) {
        self.issue_date = date;
    }

    /// Get the issuer
    pub fn issuer(&self) -> Option<&str> {
        self.issuer.as_deref()
    }

    /// Set issuer
    pub fn set_issuer(&mut self, issuer: String) {
        self.issuer = Some(issuer);
    }

    /// Get the creator
    pub fn creator(&self) -> Option<&str> {
        self.creator.as_deref()
    }

    /// Set creator
    pub fn set_creator(&mut self, creator: String) {
        self.creator = Some(creator);
    }

    /// Get the content originator
    pub fn content_originator(&self) -> Option<&str> {
        self.content_originator.as_deref()
    }

    /// Set content originator
    pub fn set_content_originator(&mut self, originator: String) {
        self.content_originator = Some(originator);
    }

    /// Get the content title
    pub fn content_title(&self) -> &str {
        &self.content_title
    }

    /// Set content title
    pub fn set_content_title(&mut self, title: String) {
        self.content_title = title;
    }

    /// Get the content kind
    pub fn content_kind(&self) -> &str {
        &self.content_kind
    }

    /// Set content kind
    pub fn set_content_kind(&mut self, kind: String) {
        self.content_kind = kind;
    }

    /// Get the edit rate
    pub fn edit_rate(&self) -> EditRate {
        self.edit_rate
    }

    /// Get the segments
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// Add a segment
    pub fn add_segment(&mut self, segment: Segment) {
        self.segments.push(segment);
    }

    /// Get all sequences across all segments
    pub fn sequences(&self) -> Vec<&Sequence> {
        self.segments
            .iter()
            .flat_map(|seg| seg.sequences.iter())
            .collect()
    }

    /// Calculate total duration in frames
    pub fn total_duration(&self) -> u64 {
        self.segments
            .iter()
            .flat_map(|seg| seg.sequences.iter())
            .filter(|seq| seq.sequence_type == SequenceType::MainImage)
            .map(Sequence::total_duration)
            .max()
            .unwrap_or(0)
    }

    /// Get the composition timecode
    pub fn composition_timecode(&self) -> Option<&CompositionTimecode> {
        self.composition_timecode.as_ref()
    }

    /// Set composition timecode
    pub fn set_composition_timecode(&mut self, timecode: CompositionTimecode) {
        self.composition_timecode = Some(timecode);
    }

    /// Parse CPL from XML
    pub fn from_xml<R: BufRead>(reader: R) -> ImfResult<Self> {
        CplParser::parse(reader)
    }

    /// Write CPL to XML
    pub fn to_xml<W: Write>(&self, writer: W) -> ImfResult<()> {
        CplWriter::write(self, writer)
    }
}

/// CPL XML parser
struct CplParser;

impl CplParser {
    #[allow(clippy::too_many_lines)]
    fn parse<R: BufRead>(reader: R) -> ImfResult<CompositionPlaylist> {
        let mut xml_reader = Reader::from_reader(reader);
        xml_reader.config_mut().trim_text(true);

        let mut buf = Vec::new();
        let mut text_buffer = String::new();

        // CPL fields
        let mut id: Option<Uuid> = None;
        let mut annotation: Option<String> = None;
        let mut issue_date: Option<DateTime<Utc>> = None;
        let mut issuer: Option<String> = None;
        let mut creator: Option<String> = None;
        let mut content_originator: Option<String> = None;
        let mut content_title: Option<String> = None;
        let mut content_kind: Option<String> = None;
        let mut edit_rate: Option<EditRate> = None;
        let mut segments: Vec<Segment> = Vec::new();

        // State for parsing nested structures
        let mut in_segment = false;
        let mut current_segment: Option<Segment> = None;
        let mut in_sequence = false;
        let mut current_sequence: Option<Sequence> = None;
        let mut in_resource = false;
        let mut current_resource_data: HashMap<String, String> = HashMap::new();
        let mut current_element;

        loop {
            match xml_reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) => {
                    current_element = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    text_buffer.clear();

                    match current_element.as_str() {
                        "SegmentList" | "Segment" => {
                            if current_element == "Segment" {
                                in_segment = true;
                                current_segment = Some(Segment::new(Uuid::new_v4()));
                            }
                        }
                        "SequenceList" | "MainImageSequence" | "MainAudioSequence"
                        | "MarkerSequence" => {
                            if current_element != "SequenceList" {
                                in_sequence = true;
                                let seq_type = SequenceType::from_str(&current_element)?;
                                current_sequence =
                                    Some(Sequence::new(Uuid::new_v4(), seq_type, Uuid::new_v4()));
                            }
                        }
                        "ResourceList" | "Resource" => {
                            if current_element == "Resource" {
                                in_resource = true;
                                current_resource_data.clear();
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Event::Text(e)) => {
                    text_buffer = String::from_utf8_lossy(e.as_ref()).to_string();
                }
                Ok(Event::End(e)) => {
                    let element_name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                    if in_resource {
                        match element_name.as_str() {
                            "Id" | "EditRate" | "IntrinsicDuration" | "EntryPoint"
                            | "SourceDuration" | "RepeatCount" | "TrackFileId" => {
                                current_resource_data
                                    .insert(element_name.clone(), text_buffer.clone());
                            }
                            "Resource" => {
                                if let Some(ref mut seq) = current_sequence {
                                    let resource = Self::build_resource(&current_resource_data)?;
                                    seq.add_resource(resource);
                                }
                                in_resource = false;
                            }
                            _ => {}
                        }
                    } else if in_sequence {
                        match element_name.as_str() {
                            "Id" => {
                                if let Some(ref mut seq) = current_sequence {
                                    seq.id = Uuid::parse_str(&text_buffer)?;
                                }
                            }
                            "TrackId" => {
                                if let Some(ref mut seq) = current_sequence {
                                    seq.track_id = Uuid::parse_str(&text_buffer)?;
                                }
                            }
                            "MainImageSequence" | "MainAudioSequence" | "MarkerSequence" => {
                                if let (Some(seq), Some(ref mut seg)) =
                                    (current_sequence.take(), current_segment.as_mut())
                                {
                                    seg.add_sequence(seq);
                                }
                                in_sequence = false;
                            }
                            _ => {}
                        }
                    } else if in_segment {
                        match element_name.as_str() {
                            "Id" => {
                                if let Some(ref mut seg) = current_segment {
                                    seg.id = Uuid::parse_str(&text_buffer)?;
                                }
                            }
                            "Annotation" => {
                                if let Some(ref mut seg) = current_segment {
                                    seg.set_annotation(text_buffer.clone());
                                }
                            }
                            "Segment" => {
                                if let Some(seg) = current_segment.take() {
                                    segments.push(seg);
                                }
                                in_segment = false;
                            }
                            _ => {}
                        }
                    } else {
                        // Top-level elements
                        match element_name.as_str() {
                            "Id" => id = Some(Uuid::parse_str(&text_buffer)?),
                            "Annotation" => annotation = Some(text_buffer.clone()),
                            "IssueDate" => {
                                issue_date = Some(
                                    DateTime::parse_from_rfc3339(&text_buffer)
                                        .map_err(|e| {
                                            ImfError::InvalidStructure(format!(
                                                "Invalid IssueDate: {e}"
                                            ))
                                        })?
                                        .with_timezone(&Utc),
                                );
                            }
                            "Issuer" => issuer = Some(text_buffer.clone()),
                            "Creator" => creator = Some(text_buffer.clone()),
                            "ContentOriginator" => content_originator = Some(text_buffer.clone()),
                            "ContentTitle" => content_title = Some(text_buffer.clone()),
                            "ContentKind" => content_kind = Some(text_buffer.clone()),
                            "EditRate" => edit_rate = Some(EditRate::from_str(&text_buffer)?),
                            _ => {}
                        }
                    }

                    text_buffer.clear();
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(ImfError::XmlError(format!("XML parse error: {e}"))),
                _ => {}
            }
            buf.clear();
        }

        // Build CPL
        let id = id.ok_or_else(|| ImfError::MissingElement("Id".to_string()))?;
        let content_title =
            content_title.ok_or_else(|| ImfError::MissingElement("ContentTitle".to_string()))?;
        let edit_rate =
            edit_rate.ok_or_else(|| ImfError::MissingElement("EditRate".to_string()))?;

        let mut cpl = CompositionPlaylist::new(id, content_title, edit_rate);
        cpl.annotation = annotation;
        cpl.issue_date = issue_date.unwrap_or_else(Utc::now);
        cpl.issuer = issuer;
        cpl.creator = creator;
        cpl.content_originator = content_originator;
        cpl.content_kind = content_kind.unwrap_or_else(|| "feature".to_string());
        cpl.segments = segments;

        Ok(cpl)
    }

    fn build_resource(data: &HashMap<String, String>) -> ImfResult<Resource> {
        let id = data
            .get("Id")
            .ok_or_else(|| ImfError::MissingElement("Resource Id".to_string()))?;
        let id = Uuid::parse_str(id)?;

        let edit_rate = data
            .get("EditRate")
            .ok_or_else(|| ImfError::MissingElement("Resource EditRate".to_string()))?;
        let edit_rate = EditRate::from_str(edit_rate)?;

        let intrinsic_duration = data
            .get("IntrinsicDuration")
            .ok_or_else(|| ImfError::MissingElement("IntrinsicDuration".to_string()))?;
        let intrinsic_duration: u64 = intrinsic_duration
            .parse()
            .map_err(|_| ImfError::InvalidStructure("Invalid IntrinsicDuration".to_string()))?;

        let entry_point = data.get("EntryPoint").and_then(|s| s.parse().ok());
        let source_duration = data.get("SourceDuration").and_then(|s| s.parse().ok());

        let repeat_count = data
            .get("RepeatCount")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        let track_file_id = data
            .get("TrackFileId")
            .ok_or_else(|| ImfError::MissingElement("TrackFileId".to_string()))?;
        let track_file_id = Uuid::parse_str(track_file_id)?;

        Ok(Resource::new(
            id,
            edit_rate,
            intrinsic_duration,
            entry_point,
            source_duration,
            repeat_count,
            track_file_id,
        ))
    }
}

/// CPL XML writer
struct CplWriter;

impl CplWriter {
    fn write<W: Write>(cpl: &CompositionPlaylist, writer: W) -> ImfResult<()> {
        let mut xml_writer = Writer::new_with_indent(writer, b' ', 2);

        // XML declaration
        xml_writer
            .write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        // Root element
        let mut root = BytesStart::new("CompositionPlaylist");
        root.push_attribute(("xmlns", "http://www.smpte-ra.org/schemas/2067-3/2016"));
        xml_writer
            .write_event(Event::Start(root))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        // Write fields
        Self::write_element(&mut xml_writer, "Id", &format!("urn:uuid:{}", cpl.id))?;

        if let Some(ref annotation) = cpl.annotation {
            Self::write_element(&mut xml_writer, "Annotation", annotation)?;
        }

        Self::write_element(&mut xml_writer, "IssueDate", &cpl.issue_date.to_rfc3339())?;

        if let Some(ref issuer) = cpl.issuer {
            Self::write_element(&mut xml_writer, "Issuer", issuer)?;
        }

        if let Some(ref creator) = cpl.creator {
            Self::write_element(&mut xml_writer, "Creator", creator)?;
        }

        if let Some(ref originator) = cpl.content_originator {
            Self::write_element(&mut xml_writer, "ContentOriginator", originator)?;
        }

        Self::write_element(&mut xml_writer, "ContentTitle", &cpl.content_title)?;
        Self::write_element(&mut xml_writer, "ContentKind", &cpl.content_kind)?;

        // Edit rate
        Self::write_element(
            &mut xml_writer,
            "EditRate",
            &format!("{} {}", cpl.edit_rate.numerator, cpl.edit_rate.denominator),
        )?;

        // Segments
        Self::write_segments(&mut xml_writer, &cpl.segments)?;

        // Close root
        xml_writer
            .write_event(Event::End(BytesEnd::new("CompositionPlaylist")))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        Ok(())
    }

    fn write_element<W: Write>(writer: &mut Writer<W>, name: &str, content: &str) -> ImfResult<()> {
        writer
            .write_event(Event::Start(BytesStart::new(name)))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;
        writer
            .write_event(Event::Text(BytesText::new(content)))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;
        writer
            .write_event(Event::End(BytesEnd::new(name)))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;
        Ok(())
    }

    fn write_segments<W: Write>(writer: &mut Writer<W>, segments: &[Segment]) -> ImfResult<()> {
        writer
            .write_event(Event::Start(BytesStart::new("SegmentList")))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        for segment in segments {
            Self::write_segment(writer, segment)?;
        }

        writer
            .write_event(Event::End(BytesEnd::new("SegmentList")))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        Ok(())
    }

    fn write_segment<W: Write>(writer: &mut Writer<W>, segment: &Segment) -> ImfResult<()> {
        writer
            .write_event(Event::Start(BytesStart::new("Segment")))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        Self::write_element(writer, "Id", &format!("urn:uuid:{}", segment.id))?;

        if let Some(ref annotation) = segment.annotation {
            Self::write_element(writer, "Annotation", annotation)?;
        }

        // Sequences
        writer
            .write_event(Event::Start(BytesStart::new("SequenceList")))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        for sequence in &segment.sequences {
            Self::write_sequence(writer, sequence)?;
        }

        writer
            .write_event(Event::End(BytesEnd::new("SequenceList")))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        writer
            .write_event(Event::End(BytesEnd::new("Segment")))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        Ok(())
    }

    fn write_sequence<W: Write>(writer: &mut Writer<W>, sequence: &Sequence) -> ImfResult<()> {
        let seq_type_str = sequence.sequence_type.to_string();

        writer
            .write_event(Event::Start(BytesStart::new(seq_type_str.as_str())))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        Self::write_element(writer, "Id", &format!("urn:uuid:{}", sequence.id))?;
        Self::write_element(
            writer,
            "TrackId",
            &format!("urn:uuid:{}", sequence.track_id),
        )?;

        // Resources
        writer
            .write_event(Event::Start(BytesStart::new("ResourceList")))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        for resource in &sequence.resources {
            Self::write_resource(writer, resource)?;
        }

        writer
            .write_event(Event::End(BytesEnd::new("ResourceList")))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        writer
            .write_event(Event::End(BytesEnd::new(seq_type_str.as_str())))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        Ok(())
    }

    fn write_resource<W: Write>(writer: &mut Writer<W>, resource: &Resource) -> ImfResult<()> {
        writer
            .write_event(Event::Start(BytesStart::new("Resource")))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        Self::write_element(writer, "Id", &format!("urn:uuid:{}", resource.id))?;

        Self::write_element(
            writer,
            "EditRate",
            &format!(
                "{} {}",
                resource.edit_rate.numerator, resource.edit_rate.denominator
            ),
        )?;

        Self::write_element(
            writer,
            "IntrinsicDuration",
            &resource.intrinsic_duration.to_string(),
        )?;

        if let Some(entry_point) = resource.entry_point {
            Self::write_element(writer, "EntryPoint", &entry_point.to_string())?;
        }

        if let Some(source_duration) = resource.source_duration {
            Self::write_element(writer, "SourceDuration", &source_duration.to_string())?;
        }

        if resource.repeat_count != 1 {
            Self::write_element(writer, "RepeatCount", &resource.repeat_count.to_string())?;
        }

        Self::write_element(
            writer,
            "TrackFileId",
            &format!("urn:uuid:{}", resource.track_file_id),
        )?;

        writer
            .write_event(Event::End(BytesEnd::new("Resource")))
            .map_err(|e| ImfError::XmlError(format!("Write error: {e}")))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edit_rate() {
        let rate = EditRate::fps_24();
        assert_eq!(rate.numerator(), 24);
        assert_eq!(rate.denominator(), 1);
        assert!((rate.as_f64() - 24.0).abs() < f64::EPSILON);

        let rate = EditRate::fps_23_976();
        assert_eq!(rate.numerator(), 24000);
        assert_eq!(rate.denominator(), 1001);
    }

    #[test]
    fn test_edit_rate_parsing() {
        let rate = EditRate::from_str("24 1").expect("rate should be valid");
        assert_eq!(rate, EditRate::fps_24());

        let rate = EditRate::from_str("24000 1001").expect("rate should be valid");
        assert_eq!(rate, EditRate::fps_23_976());
    }

    #[test]
    fn test_sequence_type() {
        assert_eq!(
            SequenceType::from_str("MainImageSequence").expect("test expectation failed"),
            SequenceType::MainImage
        );
        assert_eq!(
            SequenceType::from_str("MainAudioSequence").expect("test expectation failed"),
            SequenceType::MainAudio
        );
    }

    #[test]
    fn test_resource_duration() {
        let resource = Resource::new(
            Uuid::new_v4(),
            EditRate::fps_24(),
            1000,
            Some(100),
            Some(500),
            2,
            Uuid::new_v4(),
        );

        assert_eq!(resource.effective_duration(), 1000); // 500 * 2
    }

    #[test]
    fn test_cpl_creation() {
        let mut cpl = CompositionPlaylist::new(
            Uuid::new_v4(),
            "Test Composition".to_string(),
            EditRate::fps_24(),
        );

        cpl.set_creator("OxiMedia".to_string());
        cpl.set_issuer("Test Studio".to_string());

        assert_eq!(cpl.content_title(), "Test Composition");
        assert_eq!(cpl.creator(), Some("OxiMedia"));
        assert_eq!(cpl.edit_rate(), EditRate::fps_24());
    }
}
