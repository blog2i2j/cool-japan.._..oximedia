//! Teletext subtitle format support (EBU Teletext / ETS 300 706).
//!
//! Provides a structured representation of teletext pages with per-cell
//! colour and double-height metadata, a parser that extracts subtitle text
//! from pages in the standard subtitle page range (800–899), and a converter
//! that produces SRT output.
//!
//! # Overview
//!
//! Teletext broadcasts subtitle data on dedicated magazine/page combinations.
//! In the UK, page 888 is traditionally used; continental European broadcasters
//! commonly use 777 or pages in the 800–899 range.  This module uses the
//! 800–899 decimal range as the subtitle page criterion, as specified by the
//! EBU Teletext standard.

#![allow(dead_code)]

// ============================================================================
// Colour types
// ============================================================================

/// Teletext foreground/background colour (ETS 300 706 §12.2).
///
/// The eight colours correspond directly to the three-bit colour code used
/// in teletext set-colour control characters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtColor {
    /// Black (0).
    Black,
    /// Red (1).
    Red,
    /// Green (2).
    Green,
    /// Yellow (3).
    Yellow,
    /// Blue (4).
    Blue,
    /// Magenta (5).
    Magenta,
    /// Cyan (6).
    Cyan,
    /// White (7) — the default foreground colour.
    White,
}

impl TtColor {
    /// Decode a 3-bit colour code (0–7) to `TtColor`.
    ///
    /// Returns `None` for values outside `0..=7`.
    #[must_use]
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Black),
            1 => Some(Self::Red),
            2 => Some(Self::Green),
            3 => Some(Self::Yellow),
            4 => Some(Self::Blue),
            5 => Some(Self::Magenta),
            6 => Some(Self::Cyan),
            7 => Some(Self::White),
            _ => None,
        }
    }

    /// Return the 3-bit colour code for this colour.
    #[must_use]
    pub fn to_code(self) -> u8 {
        match self {
            Self::Black => 0,
            Self::Red => 1,
            Self::Green => 2,
            Self::Yellow => 3,
            Self::Blue => 4,
            Self::Magenta => 5,
            Self::Cyan => 6,
            Self::White => 7,
        }
    }
}

impl Default for TtColor {
    /// Default foreground colour is `White`.
    fn default() -> Self {
        Self::White
    }
}

// ============================================================================
// Page / row / cell types
// ============================================================================

/// A single character cell on a teletext page.
#[derive(Debug, Clone, PartialEq)]
pub struct TeletextCell {
    /// The Unicode character to display.
    pub character: char,
    /// Foreground colour.
    pub foreground_color: TtColor,
    /// Background colour.
    pub background_color: TtColor,
    /// Whether the cell should be rendered at double height.
    /// When `true`, the cell occupies two row heights visually, but text
    /// extraction ignores this flag and treats it as a normal character.
    pub double_height: bool,
}

impl TeletextCell {
    /// Create a plain white-on-black cell for the given character.
    #[must_use]
    pub fn plain(ch: char) -> Self {
        Self {
            character: ch,
            foreground_color: TtColor::White,
            background_color: TtColor::Black,
            double_height: false,
        }
    }

    /// Create a cell with explicit colour attributes.
    #[must_use]
    pub fn with_colors(
        ch: char,
        fg: TtColor,
        bg: TtColor,
        double_height: bool,
    ) -> Self {
        Self {
            character: ch,
            foreground_color: fg,
            background_color: bg,
            double_height,
        }
    }
}

/// A single row (line) in a teletext page, containing up to 40 cells.
#[derive(Debug, Clone)]
pub struct TeletextRow {
    /// Row number within the page (0 = header row, 1–24 = data rows).
    pub row: u8,
    /// Cell data for this row.
    pub cells: Vec<TeletextCell>,
}

impl TeletextRow {
    /// Create an empty row.
    #[must_use]
    pub fn new(row: u8) -> Self {
        Self {
            row,
            cells: Vec::new(),
        }
    }

    /// Extract the text content of this row, ignoring colour and double-height
    /// metadata.  Trailing whitespace is trimmed.
    #[must_use]
    pub fn text(&self) -> String {
        let raw: String = self.cells.iter().map(|c| c.character).collect();
        raw.trim_end().to_string()
    }
}

/// A full teletext page with magazine and page number metadata.
#[derive(Debug, Clone)]
pub struct TeletextPage {
    /// Magazine number (1–8 per ETS 300 706).
    pub magazine: u8,
    /// Page number within the magazine (0x00–0xFF, displayed as two hex digits).
    pub page: u8,
    /// Subpage code (0x0000–0x3F7F, used when a page has multiple variants).
    pub subpage: u16,
    /// Row data (up to 25 rows; row 0 is the header row).
    pub rows: Vec<TeletextRow>,
}

impl TeletextPage {
    /// Create a new empty teletext page.
    #[must_use]
    pub fn new(magazine: u8, page: u8) -> Self {
        Self {
            magazine,
            page,
            subpage: 0,
            rows: Vec::new(),
        }
    }

    /// Return the three-digit decimal page number used in broadcast listings
    /// (e.g., magazine 8, page 0x88 → decimal page number 888).
    ///
    /// The formula is: `magazine * 100 + (page_bcd_tens * 10 + page_bcd_units)`.
    /// For simplicity, `page` is treated as BCD here.
    #[must_use]
    pub fn decimal_page_number(&self) -> u16 {
        let tens = (self.page >> 4) as u16;
        let units = (self.page & 0x0F) as u16;
        self.magazine as u16 * 100 + tens * 10 + units
    }
}

// ============================================================================
// Subtitle record
// ============================================================================

/// A subtitle entry extracted from a teletext page.
#[derive(Debug, Clone)]
pub struct TeletextSubtitle {
    /// Decimal page number (e.g., 888).
    pub page_number: u16,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// Non-empty text lines extracted from the page.
    pub lines: Vec<String>,
}

impl TeletextSubtitle {
    /// Return the subtitle text as a single string with lines joined by `\n`.
    #[must_use]
    pub fn text(&self) -> String {
        self.lines.join("\n")
    }

    /// Duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

// ============================================================================
// Parser
// ============================================================================

/// Parses `TeletextPage` structures and extracts subtitle text.
///
/// Only pages whose decimal page number falls in the subtitle range 800–899
/// are processed; all other pages return `None`.
pub struct TeletextParser;

impl TeletextParser {
    /// Attempt to extract a `TeletextSubtitle` from `page`.
    ///
    /// Returns `None` when:
    /// - The decimal page number is outside 800–899.
    /// - The page contains no displayable text after stripping empty rows.
    #[must_use]
    pub fn parse_page(page: &TeletextPage) -> Option<TeletextSubtitle> {
        let page_number = page.decimal_page_number();

        // Only subtitle pages 800–899.
        if !(800..=899).contains(&page_number) {
            return None;
        }

        // Collect non-empty text lines.  Row 0 (the header) is skipped as it
        // typically contains the page number and date/time, not subtitle text.
        let lines: Vec<String> = page
            .rows
            .iter()
            .filter(|r| r.row > 0) // skip header row
            .map(|r| r.text())
            .filter(|t| !t.is_empty())
            .collect();

        if lines.is_empty() {
            return None;
        }

        Some(TeletextSubtitle {
            page_number,
            start_ms: 0,
            end_ms: 0,
            lines,
        })
    }
}

// ============================================================================
// SRT converter
// ============================================================================

/// Converts a slice of `TeletextSubtitle` records to SRT format.
pub struct TeletextConverter;

impl TeletextConverter {
    /// Render `subtitles` as an SRT string.
    ///
    /// Each subtitle becomes one SRT block.  Timing is formatted as
    /// `HH:MM:SS,mmm --> HH:MM:SS,mmm`.
    #[must_use]
    pub fn to_srt(subtitles: &[TeletextSubtitle]) -> String {
        let mut out = String::new();

        for (idx, sub) in subtitles.iter().enumerate() {
            let seq = idx + 1;
            let start = format_srt_time(sub.start_ms);
            let end = format_srt_time(sub.end_ms);
            let text = sub.text();

            out.push_str(&format!("{seq}\n{start} --> {end}\n{text}\n\n"));
        }

        out
    }
}

/// Format a millisecond timestamp as `HH:MM:SS,mmm`.
fn format_srt_time(ms: u64) -> String {
    let total_s = ms / 1_000;
    let millis = ms % 1_000;
    let h = total_s / 3_600;
    let m = (total_s % 3_600) / 60;
    let s = total_s % 60;
    format!("{h:02}:{m:02}:{s:02},{millis:03}")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper builders ───────────────────────────────────────────────────────

    fn make_page_with_rows(magazine: u8, page: u8, row_texts: &[&str]) -> TeletextPage {
        let mut p = TeletextPage::new(magazine, page);
        for (i, &text) in row_texts.iter().enumerate() {
            let mut row = TeletextRow::new(i as u8);
            for ch in text.chars() {
                row.cells.push(TeletextCell::plain(ch));
            }
            p.rows.push(row);
        }
        p
    }

    // ── TtColor tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_tt_color_from_code_valid() {
        assert_eq!(TtColor::from_code(0), Some(TtColor::Black));
        assert_eq!(TtColor::from_code(1), Some(TtColor::Red));
        assert_eq!(TtColor::from_code(7), Some(TtColor::White));
    }

    #[test]
    fn test_tt_color_from_code_invalid() {
        assert_eq!(TtColor::from_code(8), None);
        assert_eq!(TtColor::from_code(255), None);
    }

    #[test]
    fn test_tt_color_round_trip() {
        for code in 0u8..=7 {
            let color = TtColor::from_code(code).expect("valid code");
            assert_eq!(color.to_code(), code);
        }
    }

    #[test]
    fn test_tt_color_default_is_white() {
        assert_eq!(TtColor::default(), TtColor::White);
    }

    // ── TeletextPage decimal_page_number tests ────────────────────────────────

    #[test]
    fn test_decimal_page_number_888() {
        // Magazine 8, page = 0x88 (BCD: tens=8, units=8)
        let page = TeletextPage::new(8, 0x88);
        assert_eq!(page.decimal_page_number(), 888);
    }

    #[test]
    fn test_decimal_page_number_777() {
        // Magazine 7, page = 0x77
        let page = TeletextPage::new(7, 0x77);
        assert_eq!(page.decimal_page_number(), 777);
    }

    #[test]
    fn test_decimal_page_number_100() {
        // Magazine 1, page = 0x00
        let page = TeletextPage::new(1, 0x00);
        assert_eq!(page.decimal_page_number(), 100);
    }

    // ── TeletextParser subtitle page range tests ──────────────────────────────

    #[test]
    fn test_page_in_subtitle_range_accepted() {
        // Magazine 8, page 0x88 → decimal 888
        let page = make_page_with_rows(
            8, 0x88,
            &["", "Foreign dialogue line one", "Foreign dialogue line two"],
        );
        let result = TeletextParser::parse_page(&page);
        assert!(result.is_some(), "page 888 should be accepted");
        let sub = result.unwrap();
        assert_eq!(sub.page_number, 888);
        assert_eq!(sub.lines.len(), 2);
    }

    #[test]
    fn test_page_out_of_range_ignored_low() {
        // Magazine 1, page 0x00 → decimal 100 (below 800)
        let page = make_page_with_rows(1, 0x00, &["", "Some text on a non-subtitle page"]);
        let result = TeletextParser::parse_page(&page);
        assert!(result.is_none(), "page 100 is not a subtitle page");
    }

    #[test]
    fn test_page_out_of_range_ignored_high() {
        // Magazine 9 is out of standard range, but let's use magazine 8, page 0x99 → 899+?
        // Magazine 9 × 100 = 900 → outside range
        let page = make_page_with_rows(9, 0x00, &["", "Text"]);
        let result = TeletextParser::parse_page(&page);
        assert!(result.is_none(), "page >= 900 is not a subtitle page");
    }

    #[test]
    fn test_page_boundary_800_accepted() {
        // Magazine 8, page 0x00 → decimal 800
        let page = make_page_with_rows(8, 0x00, &["", "Boundary subtitle"]);
        let result = TeletextParser::parse_page(&page);
        assert!(result.is_some(), "page 800 is the lower boundary");
    }

    #[test]
    fn test_empty_subtitle_page_returns_none() {
        // Page in range but no text in rows > 0
        let mut page = TeletextPage::new(8, 0x88);
        let empty_row = TeletextRow::new(1); // no cells
        page.rows.push(empty_row);
        let result = TeletextParser::parse_page(&page);
        assert!(result.is_none(), "empty page should return None");
    }

    #[test]
    fn test_double_height_ignored_in_text_extraction() {
        let mut page = TeletextPage::new(8, 0x88);
        let mut row = TeletextRow::new(1);
        // Mix of double-height and normal cells
        row.cells.push(TeletextCell::with_colors(
            'H', TtColor::White, TtColor::Black, true,
        ));
        row.cells.push(TeletextCell::with_colors(
            'i', TtColor::White, TtColor::Black, false,
        ));
        page.rows.push(row);

        let result = TeletextParser::parse_page(&page);
        assert!(result.is_some());
        let sub = result.unwrap();
        assert_eq!(sub.lines[0], "Hi", "double_height should not alter character extraction");
    }

    // ── TeletextConverter SRT output tests ────────────────────────────────────

    #[test]
    fn test_srt_output_format() {
        let subs = vec![TeletextSubtitle {
            page_number: 888,
            start_ms: 1_000,
            end_ms: 4_000,
            lines: vec!["Hello Teletext".to_string()],
        }];
        let srt = TeletextConverter::to_srt(&subs);
        assert!(srt.contains("1\n"), "SRT must start with sequence number");
        assert!(srt.contains("00:00:01,000 --> 00:00:04,000"), "SRT timing must be correct");
        assert!(srt.contains("Hello Teletext"), "SRT must contain subtitle text");
    }

    #[test]
    fn test_srt_multi_line_output() {
        let subs = vec![TeletextSubtitle {
            page_number: 888,
            start_ms: 5_500,
            end_ms: 9_000,
            lines: vec!["Line one".to_string(), "Line two".to_string()],
        }];
        let srt = TeletextConverter::to_srt(&subs);
        assert!(srt.contains("00:00:05,500 --> 00:00:09,000"));
        assert!(srt.contains("Line one\nLine two"));
    }

    #[test]
    fn test_srt_empty_input() {
        let srt = TeletextConverter::to_srt(&[]);
        assert!(srt.is_empty());
    }

    #[test]
    fn test_srt_sequence_numbers() {
        let subs: Vec<TeletextSubtitle> = (0..3)
            .map(|i| TeletextSubtitle {
                page_number: 888,
                start_ms: i * 5_000,
                end_ms: i * 5_000 + 4_000,
                lines: vec![format!("Cue {i}")],
            })
            .collect();
        let srt = TeletextConverter::to_srt(&subs);
        assert!(srt.contains("1\n"));
        assert!(srt.contains("2\n"));
        assert!(srt.contains("3\n"));
    }

    // ── format_srt_time helper ────────────────────────────────────────────────

    #[test]
    fn test_format_srt_time_zero() {
        assert_eq!(format_srt_time(0), "00:00:00,000");
    }

    #[test]
    fn test_format_srt_time_hours() {
        // 3 661 500 ms = 1h 1m 1s 500ms
        assert_eq!(format_srt_time(3_661_500), "01:01:01,500");
    }

    // ── TeletextRow text extraction ───────────────────────────────────────────

    #[test]
    fn test_row_text_trims_trailing_spaces() {
        let mut row = TeletextRow::new(1);
        for ch in "Hello   ".chars() {
            row.cells.push(TeletextCell::plain(ch));
        }
        assert_eq!(row.text(), "Hello");
    }

    // ── TeletextSubtitle helpers ──────────────────────────────────────────────

    #[test]
    fn test_subtitle_text_joins_lines() {
        let sub = TeletextSubtitle {
            page_number: 888,
            start_ms: 0,
            end_ms: 3_000,
            lines: vec!["First".to_string(), "Second".to_string()],
        };
        assert_eq!(sub.text(), "First\nSecond");
    }

    #[test]
    fn test_subtitle_duration() {
        let sub = TeletextSubtitle {
            page_number: 888,
            start_ms: 1_000,
            end_ms: 4_000,
            lines: vec!["Test".to_string()],
        };
        assert_eq!(sub.duration_ms(), 3_000);
    }
}
