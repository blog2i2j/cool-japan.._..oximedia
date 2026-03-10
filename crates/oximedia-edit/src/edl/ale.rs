//! ALE (Avid Log Exchange) format parser and writer.
//!
//! ALE is a tab-delimited format used by Avid editing systems to exchange
//! metadata about clips. It includes information like:
//! - Clip names and tape IDs
//! - Timecode in/out points
//! - Scene/take/camera metadata
//! - Sound roll information
//! - Custom metadata fields
//!
//! # Format Structure
//!
//! An ALE file consists of:
//! 1. Header section with format metadata
//! 2. Column definition line
//! 3. Data section with tab-separated values
//!
//! # Example
//!
//! ```text
//! Heading
//! FIELD_DELIM\tTABS
//! VIDEO_FORMAT\t1080p
//! AUDIO_FORMAT\t48kHz
//! FPS\t24
//!
//! Column
//! Name\tTape\tStart\tEnd\tDuration\tScene\tTake
//!
//! Data
//! CLIP001\tA001\t01:00:00:00\t01:00:05:00\t00:00:05:00\t1\t1
//! CLIP002\tA001\t01:00:10:00\t01:00:15:00\t00:00:05:00\t1\t2
//! ```

use super::{Edl, EdlError, EdlEvent, EdlResult, EditType, Timecode};
use oximedia_core::Rational;
use std::collections::HashMap;

/// ALE file structure.
#[derive(Debug, Clone)]
pub struct AleFile {
    /// Header metadata.
    pub header: HashMap<String, String>,
    /// Column names.
    pub columns: Vec<String>,
    /// Data rows (each row is a map of column name to value).
    pub data: Vec<HashMap<String, String>>,
}

/// ALE parser.
pub struct AleParser {
    frame_rate: Rational,
    audio_format: String,
    video_format: String,
}

impl AleParser {
    /// Create a new ALE parser with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            frame_rate: Rational::new(24, 1),
            audio_format: "48kHz".to_string(),
            video_format: "1080p".to_string(),
        }
    }

    /// Parse an ALE file.
    pub fn parse(&mut self, content: &str) -> EdlResult<AleFile> {
        let lines: Vec<&str> = content.lines().collect();
        let mut header = HashMap::new();
        let mut columns = Vec::new();
        let mut data = Vec::new();

        let mut section = Section::None;
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i].trim();

            // Skip empty lines
            if line.is_empty() {
                i += 1;
                continue;
            }

            // Detect sections
            if line == "Heading" {
                section = Section::Heading;
                i += 1;
                continue;
            } else if line == "Column" {
                section = Section::Column;
                i += 1;
                continue;
            } else if line == "Data" {
                section = Section::Data;
                i += 1;
                continue;
            }

            match section {
                Section::None => {
                    // Before any section marker
                    i += 1;
                }
                Section::Heading => {
                    self.parse_header_line(line, &mut header)?;
                    i += 1;
                }
                Section::Column => {
                    columns = self.parse_column_line(line);
                    section = Section::Data;
                    i += 1;
                }
                Section::Data => {
                    if !columns.is_empty() {
                        let row = self.parse_data_line(line, &columns)?;
                        data.push(row);
                    }
                    i += 1;
                }
            }
        }

        // Update parser settings from header
        if let Some(fps) = header.get("FPS") {
            if let Ok(fps_val) = fps.parse::<i32>() {
                self.frame_rate = Rational::new(fps_val, 1);
            }
        }

        if let Some(audio) = header.get("AUDIO_FORMAT") {
            self.audio_format = audio.clone();
        }

        if let Some(video) = header.get("VIDEO_FORMAT") {
            self.video_format = video.clone();
        }

        Ok(AleFile {
            header,
            columns,
            data,
        })
    }

    /// Parse a header line (key-value pair).
    fn parse_header_line(
        &self,
        line: &str,
        header: &mut HashMap<String, String>,
    ) -> EdlResult<()> {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 2 {
            header.insert(parts[0].to_string(), parts[1].to_string());
        }
        Ok(())
    }

    /// Parse column definition line.
    fn parse_column_line(&self, line: &str) -> Vec<String> {
        line.split('\t').map(|s| s.trim().to_string()).collect()
    }

    /// Parse data line.
    fn parse_data_line(
        &self,
        line: &str,
        columns: &[String],
    ) -> EdlResult<HashMap<String, String>> {
        let values: Vec<&str> = line.split('\t').collect();
        let mut row = HashMap::new();

        for (i, column) in columns.iter().enumerate() {
            if i < values.len() {
                row.insert(column.clone(), values[i].trim().to_string());
            }
        }

        Ok(row)
    }

    /// Convert ALE file to EDL.
    pub fn to_edl(&self, ale: &AleFile) -> EdlResult<Edl> {
        let title = ale
            .header
            .get("TITLE")
            .or_else(|| ale.header.get("PROJECT"))
            .unwrap_or(&String::from("Untitled"))
            .clone();

        let mut edl = Edl::new(title, self.frame_rate, false);

        // Add metadata from header
        for (key, value) in &ale.header {
            edl.metadata.insert(key.clone(), value.clone());
        }

        // Convert each row to an event
        for (idx, row) in ale.data.iter().enumerate() {
            let event = self.row_to_event(row, idx + 1)?;
            edl.add_event(event);
        }

        Ok(edl)
    }

    /// Convert a data row to an EDL event.
    fn row_to_event(&self, row: &HashMap<String, String>, number: usize) -> EdlResult<EdlEvent> {
        // Extract standard fields
        let name = row.get("Name").unwrap_or(&String::new()).clone();
        let tape = row
            .get("Tape")
            .or_else(|| row.get("Source File"))
            .unwrap_or(&String::new())
            .clone();

        let start = row.get("Start").or_else(|| row.get("Mark In"));
        let end = row.get("End").or_else(|| row.get("Mark Out"));

        let source_in = if let Some(tc) = start {
            Timecode::parse(tc, self.frame_rate)?
        } else {
            Timecode::new(0, 0, 0, 0, false, self.frame_rate)
        };

        let source_out = if let Some(tc) = end {
            Timecode::parse(tc, self.frame_rate)?
        } else {
            Timecode::new(0, 0, 0, 0, false, self.frame_rate)
        };

        // For ALE, record timecode is typically sequential
        let record_in = Timecode::from_frames(
            (number as i64 - 1) * 150,
            self.frame_rate,
            false,
        );
        let duration = source_out.to_frames() - source_in.to_frames();
        let record_out = Timecode::from_frames(
            record_in.to_frames() + duration,
            self.frame_rate,
            false,
        );

        // Determine track type
        let track = if let Some(tracks) = row.get("Tracks") {
            tracks.clone()
        } else {
            "V".to_string()
        };

        // Create metadata from all fields
        let mut metadata = HashMap::new();
        for (key, value) in row {
            metadata.insert(key.to_lowercase().replace(' ', "_"), value.clone());
        }

        // Add clip name if present
        if !name.is_empty() {
            metadata.insert("clip_name".to_string(), name);
        }

        Ok(EdlEvent {
            number: number as u32,
            reel: tape,
            track,
            edit_type: EditType::Cut,
            source_in,
            source_out,
            record_in,
            record_out,
            transition_duration: None,
            motion_effect: None,
            comments: Vec::new(),
            metadata,
        })
    }
}

impl Default for AleParser {
    fn default() -> Self {
        Self::new()
    }
}

/// ALE writer.
pub struct AleWriter {
    columns: Vec<String>,
    include_header: bool,
}

impl AleWriter {
    /// Create a new ALE writer with default columns.
    #[must_use]
    pub fn new() -> Self {
        Self {
            columns: vec![
                "Name".to_string(),
                "Tape".to_string(),
                "Start".to_string(),
                "End".to_string(),
                "Duration".to_string(),
                "Scene".to_string(),
                "Take".to_string(),
            ],
            include_header: true,
        }
    }

    /// Set custom columns.
    #[must_use]
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.columns = columns;
        self
    }

    /// Set whether to include header section.
    #[must_use]
    pub fn with_header(mut self, include: bool) -> Self {
        self.include_header = include;
        self
    }

    /// Write an EDL to ALE format.
    pub fn write(&self, edl: &Edl) -> EdlResult<String> {
        let mut output = String::new();

        // Write header section
        if self.include_header {
            output.push_str("Heading\n");
            output.push_str("FIELD_DELIM\tTABS\n");

            // Extract frame rate
            let fps = edl.frame_rate.to_f64() as i32;
            output.push_str(&format!("FPS\t{}\n", fps));

            // Add metadata
            for (key, value) in &edl.metadata {
                output.push_str(&format!("{}\t{}\n", key.to_uppercase(), value));
            }

            output.push('\n');
        }

        // Write column section
        output.push_str("Column\n");
        output.push_str(&self.columns.join("\t"));
        output.push_str("\n\n");

        // Write data section
        output.push_str("Data\n");
        for event in &edl.events {
            self.write_event(&mut output, event);
        }

        Ok(output)
    }

    /// Write a single event as a data row.
    fn write_event(&self, output: &mut String, event: &EdlEvent) {
        let mut values = Vec::new();

        for column in &self.columns {
            let value = match column.as_str() {
                "Name" => event
                    .metadata
                    .get("clip_name")
                    .unwrap_or(&String::new())
                    .clone(),
                "Tape" | "Source File" => event.reel.clone(),
                "Start" | "Mark In" => event.source_in.format(),
                "End" | "Mark Out" => event.source_out.format(),
                "Duration" => {
                    let duration_frames = event.source_out.to_frames() - event.source_in.to_frames();
                    Timecode::from_frames(
                        duration_frames,
                        event.source_in.frame_rate,
                        event.source_in.drop_frame,
                    )
                    .format()
                }
                "Scene" => event
                    .metadata
                    .get("scene")
                    .unwrap_or(&String::new())
                    .clone(),
                "Take" => event
                    .metadata
                    .get("take")
                    .unwrap_or(&String::new())
                    .clone(),
                "Tracks" => event.track.clone(),
                _ => event
                    .metadata
                    .get(&column.to_lowercase().replace(' ', "_"))
                    .unwrap_or(&String::new())
                    .clone(),
            };

            values.push(value);
        }

        output.push_str(&values.join("\t"));
        output.push('\n');
    }
}

impl Default for AleWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Section types in ALE file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    None,
    Heading,
    Column,
    Data,
}

/// Parse an ALE file from string.
pub fn parse(content: &str) -> EdlResult<Edl> {
    let mut parser = AleParser::new();
    let ale = parser.parse(content)?;
    parser.to_edl(&ale)
}

/// Write an EDL to ALE format.
pub fn write(edl: &Edl) -> EdlResult<String> {
    AleWriter::new().write(edl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ale() {
        let content = r"Heading
FIELD_DELIM	TABS
FPS	24

Column
Name	Tape	Start	End	Duration

Data
CLIP001	A001	01:00:00:00	01:00:05:00	00:00:05:00
CLIP002	A001	01:00:10:00	01:00:15:00	00:00:05:00
";

        let mut parser = AleParser::new();
        let ale = parser.parse(content).expect("ale should be valid");

        assert_eq!(ale.header.get("FPS"), Some(&"24".to_string()));
        assert_eq!(ale.columns.len(), 5);
        assert_eq!(ale.data.len(), 2);
        assert_eq!(ale.data[0].get("Name"), Some(&"CLIP001".to_string()));
    }

    #[test]
    fn test_ale_to_edl() {
        let content = r"Heading
FIELD_DELIM	TABS
FPS	24

Column
Name	Tape	Start	End

Data
CLIP001	A001	01:00:00:00	01:00:05:00
CLIP002	A001	01:00:10:00	01:00:15:00
";

        let edl = parse(content).expect("edl should be valid");
        assert_eq!(edl.events.len(), 2);
        assert_eq!(edl.events[0].reel, "A001");
        assert_eq!(
            edl.events[0].metadata.get("clip_name"),
            Some(&"CLIP001".to_string())
        );
    }

    #[test]
    fn test_write_ale() {
        let mut edl = Edl::new("Test Project".to_string(), Rational::new(24, 1), false);

        let mut metadata = HashMap::new();
        metadata.insert("clip_name".to_string(), "CLIP001".to_string());
        metadata.insert("scene".to_string(), "1".to_string());
        metadata.insert("take".to_string(), "1".to_string());

        let event = EdlEvent {
            number: 1,
            reel: "A001".to_string(),
            track: "V".to_string(),
            edit_type: EditType::Cut,
            source_in: Timecode::new(1, 0, 0, 0, false, Rational::new(24, 1)),
            source_out: Timecode::new(1, 0, 5, 0, false, Rational::new(24, 1)),
            record_in: Timecode::new(1, 0, 0, 0, false, Rational::new(24, 1)),
            record_out: Timecode::new(1, 0, 5, 0, false, Rational::new(24, 1)),
            transition_duration: None,
            motion_effect: None,
            comments: Vec::new(),
            metadata,
        };

        edl.add_event(event);

        let output = write(&edl).expect("output should be valid");
        assert!(output.contains("Heading"));
        assert!(output.contains("Column"));
        assert!(output.contains("Data"));
        assert!(output.contains("CLIP001"));
        assert!(output.contains("A001"));
    }

    #[test]
    fn test_ale_custom_columns() {
        let mut edl = Edl::new("Test".to_string(), Rational::new(24, 1), false);

        let mut metadata = HashMap::new();
        metadata.insert("clip_name".to_string(), "CLIP001".to_string());
        metadata.insert("camera".to_string(), "A".to_string());

        let event = EdlEvent {
            number: 1,
            reel: "A001".to_string(),
            track: "V".to_string(),
            edit_type: EditType::Cut,
            source_in: Timecode::new(1, 0, 0, 0, false, Rational::new(24, 1)),
            source_out: Timecode::new(1, 0, 5, 0, false, Rational::new(24, 1)),
            record_in: Timecode::new(1, 0, 0, 0, false, Rational::new(24, 1)),
            record_out: Timecode::new(1, 0, 5, 0, false, Rational::new(24, 1)),
            transition_duration: None,
            motion_effect: None,
            comments: Vec::new(),
            metadata,
        };

        edl.add_event(event);

        let writer = AleWriter::new().with_columns(vec![
            "Name".to_string(),
            "Tape".to_string(),
            "Camera".to_string(),
        ]);

        let output = writer.write(&edl).expect("output should be valid");
        assert!(output.contains("Camera"));
    }
}
