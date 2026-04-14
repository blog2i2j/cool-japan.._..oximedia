//! EIA-608 (CEA-608) real-time caption decoding from a byte stream.
//!
//! This module implements a stateful streaming decoder for EIA/CEA-608 closed
//! captions. Unlike the batch decoder in `cea608_decoder`, this module is
//! designed for live caption sources where byte pairs arrive continuously and
//! display events must be emitted incrementally.
//!
//! # Caption Modes
//!
//! CEA-608 supports three display modes:
//! - **Pop-on**: captions are loaded into an off-screen buffer and displayed
//!   all at once on `End-of-Caption` (EOC) command.
//! - **Roll-up**: rows scroll up when a carriage return is received; text
//!   appears immediately.
//! - **Paint-on**: characters appear immediately as they are received.
//!
//! # Usage
//!
//! ```rust
//! use oximedia_subtitle::parser::eia608_realtime::{RealtimeCea608Decoder, CaptionEvent};
//!
//! let mut decoder = RealtimeCea608Decoder::new();
//!
//! // Feed byte pairs with timestamps (milliseconds)
//! decoder.feed(0x94, 0x20, 0);     // Resume caption loading
//! decoder.feed(0x48, 0x49, 100);   // 'H', 'I'
//! decoder.feed(0x94, 0x2F, 500);   // End of caption
//!
//! let events = decoder.drain_events();
//! assert!(!events.is_empty());
//! ```

use crate::Subtitle;
use std::collections::VecDeque;

// ── Character buffer ─────────────────────────────────────────────────────────

const ROWS: usize = 15;
const COLS: usize = 32;

/// A single styled character cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CharCell {
    /// The Unicode character.
    pub ch: char,
    /// Whether the character is in italics.
    pub italic: bool,
    /// Whether the character is underlined.
    pub underline: bool,
    /// Color index (0-6; see `CaptionColor`).
    pub color: u8,
}

impl CharCell {
    /// Create a blank (space) cell.
    #[must_use]
    pub const fn blank() -> Self {
        Self {
            ch: ' ',
            italic: false,
            underline: false,
            color: 0,
        }
    }

    /// Returns `true` if this cell contains a visible character.
    #[must_use]
    pub fn is_visible(&self) -> bool {
        !self.ch.is_whitespace()
    }
}

/// A 15×32 caption screen buffer.
#[derive(Clone, Debug)]
struct ScreenBuffer {
    cells: [[CharCell; COLS]; ROWS],
}

impl ScreenBuffer {
    fn new() -> Self {
        Self {
            cells: [[CharCell::blank(); COLS]; ROWS],
        }
    }

    fn clear(&mut self) {
        *self = Self::new();
    }

    fn set(&mut self, row: usize, col: usize, cell: CharCell) {
        if row < ROWS && col < COLS {
            self.cells[row][col] = cell;
        }
    }

    fn clear_cell(&mut self, row: usize, col: usize) {
        if row < ROWS && col < COLS {
            self.cells[row][col] = CharCell::blank();
        }
    }

    /// Scroll all rows up by one, clearing the bottom row.
    fn scroll_up(&mut self) {
        for r in 0..ROWS - 1 {
            self.cells[r] = self.cells[r + 1];
        }
        self.cells[ROWS - 1] = [CharCell::blank(); COLS];
    }

    /// Extract all non-empty rows as a newline-joined string.
    fn to_text(&self) -> String {
        self.cells
            .iter()
            .map(|row| {
                row.iter()
                    .map(|c| c.ch)
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .filter(|s| !s.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Check whether any cell contains a visible character.
    fn has_content(&self) -> bool {
        self.cells
            .iter()
            .any(|row| row.iter().any(CharCell::is_visible))
    }
}

// ── Display mode ─────────────────────────────────────────────────────────────

/// CEA-608 caption display mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayMode {
    /// Pop-on: caption is staged in a non-display buffer.
    PopOn,
    /// Roll-up: rows scroll.
    RollUp,
    /// Paint-on: characters appear immediately.
    PaintOn,
}

// ── Events ───────────────────────────────────────────────────────────────────

/// An event emitted by the real-time decoder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CaptionEvent {
    /// A complete caption has been assembled and should be displayed.
    DisplayCaption {
        /// Start time in milliseconds.
        start_ms: i64,
        /// Text content.
        text: String,
    },
    /// The currently displayed caption should be cleared.
    ClearCaption {
        /// Timestamp when the clear occurred.
        timestamp_ms: i64,
    },
    /// Incremental text update for roll-up / paint-on mode.
    RollUpText {
        /// Timestamp.
        timestamp_ms: i64,
        /// Current visible text.
        text: String,
    },
}

impl CaptionEvent {
    /// Convert this event into a `Subtitle` if it carries display text.
    ///
    /// For `DisplayCaption` events an approximate end time of
    /// `start_ms + 5000` ms is used; callers should set a proper end time
    /// based on the subsequent `ClearCaption` event.
    #[must_use]
    pub fn to_subtitle(&self) -> Option<Subtitle> {
        match self {
            Self::DisplayCaption { start_ms, text } => {
                Some(Subtitle::new(*start_ms, start_ms + 5_000, text.clone()))
            }
            _ => None,
        }
    }
}

// ── Decoder ──────────────────────────────────────────────────────────────────

/// Stateful streaming decoder for EIA/CEA-608 closed captions.
///
/// Feed byte pairs with [`Self::feed`] and collect events with
/// [`Self::drain_events`].
pub struct RealtimeCea608Decoder {
    /// Current display mode.
    mode: DisplayMode,
    /// Display buffer (visible to viewer).
    display_buf: ScreenBuffer,
    /// Non-display buffer (pop-on staging).
    stage_buf: ScreenBuffer,
    /// Current cursor row (0-based).
    cursor_row: usize,
    /// Current cursor column (0-based).
    cursor_col: usize,
    /// Number of roll-up rows (2, 3, or 4).
    rollup_rows: usize,
    /// Current italic state.
    italic: bool,
    /// Current underline state.
    underline: bool,
    /// Current color index.
    color: u8,
    /// Timestamp when the current caption started.
    caption_start: Option<i64>,
    /// Last byte pair to detect duplicates (CEA-608 doubles each byte pair).
    last_pair: Option<(u8, u8)>,
    /// Queued events awaiting collection.
    events: VecDeque<CaptionEvent>,
}

impl RealtimeCea608Decoder {
    /// Create a new real-time CEA-608 decoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mode: DisplayMode::PopOn,
            display_buf: ScreenBuffer::new(),
            stage_buf: ScreenBuffer::new(),
            cursor_row: ROWS - 1,
            cursor_col: 0,
            rollup_rows: 3,
            italic: false,
            underline: false,
            color: 0,
            caption_start: None,
            last_pair: None,
            events: VecDeque::new(),
        }
    }

    /// Feed a single CEA-608 byte pair at the given timestamp (milliseconds).
    ///
    /// This method is idempotent for the mandatory double-transmission of
    /// control codes: duplicate control pairs are silently ignored.
    pub fn feed(&mut self, byte1: u8, byte2: u8, timestamp_ms: i64) {
        // Remove parity bits
        let b1 = byte1 & 0x7F;
        let b2 = byte2 & 0x7F;

        // Null padding — ignore
        if b1 == 0x00 && b2 == 0x00 {
            return;
        }

        // Control codes are transmitted twice; skip the duplicate
        let is_control = (0x10..=0x1F).contains(&b1);
        if is_control {
            if self.last_pair == Some((b1, b2)) {
                self.last_pair = None;
                return;
            }
            self.last_pair = Some((b1, b2));
        } else {
            self.last_pair = None;
        }

        if is_control {
            self.handle_control(b1, b2, timestamp_ms);
        } else if b1 >= 0x20 {
            // Printable characters
            self.add_char(map_char(b1), timestamp_ms);
            if b2 >= 0x20 {
                self.add_char(map_char(b2), timestamp_ms);
            }
        }
    }

    /// Drain all pending `CaptionEvent`s accumulated since the last drain.
    pub fn drain_events(&mut self) -> Vec<CaptionEvent> {
        self.events.drain(..).collect()
    }

    /// Collect all completed subtitles emitted so far.
    ///
    /// Only `DisplayCaption` events produce subtitles; roll-up and clear events
    /// are discarded. The subtitle end time is approximated at `start + 5 s`.
    #[must_use]
    pub fn collect_subtitles(&self) -> Vec<Subtitle> {
        self.events
            .iter()
            .filter_map(CaptionEvent::to_subtitle)
            .collect()
    }

    // ── Private helpers ───────────────────────────────────────────────────

    fn handle_control(&mut self, b1: u8, b2: u8, ts: i64) {
        match (b1, b2) {
            // ── Mode switches ─────────────────────────────────────────────
            // Resume caption loading → pop-on
            (0x14 | 0x1C, 0x20) => {
                self.mode = DisplayMode::PopOn;
                if self.caption_start.is_none() {
                    self.caption_start = Some(ts);
                }
            }
            // End of caption → display pop-on buffer
            (0x14 | 0x1C, 0x2F) => {
                if self.mode == DisplayMode::PopOn {
                    std::mem::swap(&mut self.display_buf, &mut self.stage_buf);
                    self.stage_buf.clear();
                    let text = self.display_buf.to_text();
                    if !text.trim().is_empty() {
                        let start = self.caption_start.unwrap_or(ts);
                        self.events.push_back(CaptionEvent::DisplayCaption {
                            start_ms: start,
                            text,
                        });
                    }
                    self.caption_start = None;
                }
            }
            // Roll-up 2
            (0x14 | 0x1C, 0x25) => {
                self.mode = DisplayMode::RollUp;
                self.rollup_rows = 2;
                if self.caption_start.is_none() {
                    self.caption_start = Some(ts);
                }
            }
            // Roll-up 3
            (0x14 | 0x1C, 0x26) => {
                self.mode = DisplayMode::RollUp;
                self.rollup_rows = 3;
                if self.caption_start.is_none() {
                    self.caption_start = Some(ts);
                }
            }
            // Roll-up 4
            (0x14 | 0x1C, 0x27) => {
                self.mode = DisplayMode::RollUp;
                self.rollup_rows = 4;
                if self.caption_start.is_none() {
                    self.caption_start = Some(ts);
                }
            }
            // Resume direct captioning → paint-on
            (0x14 | 0x1C, 0x29) => {
                self.mode = DisplayMode::PaintOn;
                if self.caption_start.is_none() {
                    self.caption_start = Some(ts);
                }
            }

            // ── Carriage return (roll-up scroll) ──────────────────────────
            (0x14 | 0x1C, 0x2D) => {
                if self.mode == DisplayMode::RollUp {
                    self.display_buf.scroll_up();
                    let text = self.display_buf.to_text();
                    if !text.trim().is_empty() {
                        self.events.push_back(CaptionEvent::RollUpText {
                            timestamp_ms: ts,
                            text,
                        });
                    }
                    self.cursor_col = 0;
                }
            }

            // ── Erase displayed memory ────────────────────────────────────
            (0x14 | 0x1C, 0x2C) => {
                self.display_buf.clear();
                self.events
                    .push_back(CaptionEvent::ClearCaption { timestamp_ms: ts });
            }
            // ── Erase non-displayed memory ────────────────────────────────
            (0x14 | 0x1C, 0x2E) => {
                self.stage_buf.clear();
            }

            // ── Backspace ─────────────────────────────────────────────────
            (0x14 | 0x1C, 0x21) => {
                if self.cursor_col > 0 {
                    self.cursor_col -= 1;
                    let row = self.cursor_row;
                    let col = self.cursor_col;
                    self.active_buf_mut().clear_cell(row, col);
                }
            }

            // ── Mid-row style codes ───────────────────────────────────────
            (0x11, b2) if (0x20..=0x2F).contains(&b2) => {
                self.underline = (b2 & 0x01) != 0;
                let style = (b2 & 0x0E) >> 1;
                if style == 7 {
                    self.italic = true;
                } else {
                    self.italic = false;
                    self.color = style;
                }
            }

            // ── Special characters (channel 1 only) ───────────────────────
            (0x11, b2) if (0x30..=0x3F).contains(&b2) => {
                if let Some(ch) = special_char(b2) {
                    self.add_char(ch, ts);
                }
            }

            // ── PAC (Preamble Address Code) ───────────────────────────────
            (b1, b2) if (0x10..=0x1F).contains(&b1) => {
                let row = pac_row(b1);
                self.cursor_row = row;
                self.cursor_col = 0;
                self.underline = (b2 & 0x01) != 0;
                let style = (b2 & 0x0E) >> 1;
                if style == 7 {
                    self.italic = true;
                } else {
                    self.italic = false;
                    self.color = style;
                }
            }

            _ => {}
        }
    }

    /// Add a character to the active buffer at the current cursor position.
    fn add_char(&mut self, ch: char, ts: i64) {
        let cell = CharCell {
            ch,
            italic: self.italic,
            underline: self.underline,
            color: self.color,
        };
        let row = self.cursor_row;
        let col = self.cursor_col;
        self.active_buf_mut().set(row, col, cell);
        self.cursor_col = (self.cursor_col + 1).min(COLS - 1);

        // For paint-on mode emit incremental updates
        if self.mode == DisplayMode::PaintOn {
            let text = self.display_buf.to_text();
            if !text.trim().is_empty() {
                self.events.push_back(CaptionEvent::RollUpText {
                    timestamp_ms: ts,
                    text,
                });
            }
        }
    }

    /// Return a mutable reference to the buffer that receives characters in
    /// the current mode.
    fn active_buf_mut(&mut self) -> &mut ScreenBuffer {
        match self.mode {
            DisplayMode::PopOn => &mut self.stage_buf,
            DisplayMode::RollUp | DisplayMode::PaintOn => &mut self.display_buf,
        }
    }
}

impl Default for RealtimeCea608Decoder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Map a CEA-608 basic character byte to Unicode.
fn map_char(b: u8) -> char {
    match b {
        0x2A => 'á',
        0x5C => 'é',
        0x5E => 'í',
        0x5F => 'ó',
        0x60 => 'ú',
        0x7B => 'ç',
        0x7C => '÷',
        0x7D => 'Ñ',
        0x7E => 'ñ',
        0x7F => '█',
        b if b >= 0x20 => b as char,
        _ => ' ',
    }
}

/// Map a CEA-608 special character byte (0x30–0x3F) to Unicode.
fn special_char(b2: u8) -> Option<char> {
    Some(match b2 {
        0x30 => '®',
        0x31 => '°',
        0x32 => '½',
        0x33 => '¿',
        0x34 => '™',
        0x35 => '¢',
        0x36 => '£',
        0x37 => '♪',
        0x38 => 'à',
        0x39 => '\u{00A0}', // NBSP
        0x3A => 'è',
        0x3B => 'â',
        0x3C => 'ê',
        0x3D => 'î',
        0x3E => 'ô',
        0x3F => 'û',
        _ => return None,
    })
}

/// Compute the caption row (0-based) from a PAC first byte.
fn pac_row(b1: u8) -> usize {
    let high = if b1 & 0x08 != 0 { 8 } else { 0 };
    let low = (b1 & 0x07) as usize;
    (high + low).min(ROWS - 1)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pop_on_basic() {
        let mut dec = RealtimeCea608Decoder::new();
        // Resume caption loading
        dec.feed(0x94, 0x20, 0);
        // 'H' 'I'
        dec.feed(0x48, 0x49, 100);
        // End of caption
        dec.feed(0x94, 0x2F, 500);

        let events = dec.drain_events();
        let displays: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, CaptionEvent::DisplayCaption { .. }))
            .collect();
        assert_eq!(displays.len(), 1, "expected exactly one DisplayCaption");
        if let CaptionEvent::DisplayCaption { text, start_ms } = &displays[0] {
            assert!(text.contains("HI"), "text should contain HI, got: {text}");
            assert_eq!(*start_ms, 0);
        }
    }

    #[test]
    fn test_duplicate_control_ignored() {
        let mut dec = RealtimeCea608Decoder::new();
        // CEA-608 requires sending each control pair twice; second should be ignored
        dec.feed(0x94, 0x20, 0);
        dec.feed(0x94, 0x20, 0); // duplicate — should be ignored
        dec.feed(0x48, 0x49, 100);
        dec.feed(0x94, 0x2F, 500);
        dec.feed(0x94, 0x2F, 500); // duplicate

        let events = dec.drain_events();
        let count = events
            .iter()
            .filter(|e| matches!(e, CaptionEvent::DisplayCaption { .. }))
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_erase_displayed_emits_clear() {
        let mut dec = RealtimeCea608Decoder::new();
        dec.feed(0x94, 0x2C, 1000); // Erase displayed memory
        let events = dec.drain_events();
        assert!(events
            .iter()
            .any(|e| matches!(e, CaptionEvent::ClearCaption { .. })));
    }

    #[test]
    fn test_rollup_carriage_return_emits_event() {
        let mut dec = RealtimeCea608Decoder::new();
        dec.feed(0x94, 0x25, 0); // Roll-up 2 rows
        dec.feed(0x48, 0x49, 100); // 'H' 'I'
        dec.feed(0x94, 0x2D, 200); // Carriage return

        let events = dec.drain_events();
        // Should have at least one RollUpText event
        assert!(events
            .iter()
            .any(|e| matches!(e, CaptionEvent::RollUpText { .. })));
    }

    #[test]
    fn test_special_char_registered() {
        assert_eq!(special_char(0x30), Some('®'));
        assert_eq!(special_char(0x37), Some('♪'));
        assert_eq!(special_char(0x3F), Some('û'));
        assert_eq!(special_char(0x00), None);
    }

    #[test]
    fn test_pac_row_mapping() {
        assert_eq!(pac_row(0x10), 0);
        assert_eq!(pac_row(0x17), 7);
        assert_eq!(pac_row(0x18), 8);
        assert_eq!(pac_row(0x1F), 14);
    }

    #[test]
    fn test_map_char_basic_ascii() {
        assert_eq!(map_char(0x41), 'A');
        assert_eq!(map_char(0x61), 'a');
        assert_eq!(map_char(0x20), ' ');
    }

    #[test]
    fn test_map_char_special() {
        assert_eq!(map_char(0x2A), 'á');
        assert_eq!(map_char(0x7E), 'ñ');
    }

    #[test]
    fn test_backspace() {
        let mut dec = RealtimeCea608Decoder::new();
        dec.feed(0x94, 0x20, 0); // pop-on
        dec.feed(0x48, 0x49, 100); // 'H' 'I'
        dec.feed(0x94, 0x21, 150); // backspace — erases 'I'
        dec.feed(0x94, 0x2F, 500); // EOC

        let events = dec.drain_events();
        let text = events.iter().find_map(|e| {
            if let CaptionEvent::DisplayCaption { text, .. } = e {
                Some(text.clone())
            } else {
                None
            }
        });
        // After backspace the 'I' should be erased
        let text = text.expect("should have display caption");
        assert!(!text.contains("HI"), "I should have been erased, got: {text}");
    }

    #[test]
    fn test_collect_subtitles() {
        let mut dec = RealtimeCea608Decoder::new();
        dec.feed(0x94, 0x20, 0);
        dec.feed(0x48, 0x69, 100); // 'H' 'i'
        dec.feed(0x94, 0x2F, 500);
        let subs = dec.collect_subtitles();
        // collect_subtitles reads from pending events queue
        assert_eq!(subs.len(), 1);
        assert!(subs[0].text.contains("Hi"));
    }

    #[test]
    fn test_null_padding_ignored() {
        let mut dec = RealtimeCea608Decoder::new();
        dec.feed(0x00, 0x00, 0);
        dec.feed(0x80, 0x80, 10);
        let events = dec.drain_events();
        assert!(events.is_empty(), "null pairs should produce no events");
    }

    #[test]
    fn test_erase_non_displayed_memory() {
        let mut dec = RealtimeCea608Decoder::new();
        dec.feed(0x94, 0x20, 0); // pop-on
        dec.feed(0x48, 0x49, 100); // 'H' 'I' into stage buf
        dec.feed(0x94, 0x2E, 200); // Erase non-displayed
        dec.feed(0x94, 0x2F, 500); // EOC — stage buf is empty, swap and display

        let events = dec.drain_events();
        // After erasing the stage buf there should be no DisplayCaption with text
        let has_nonempty_display = events.iter().any(|e| {
            if let CaptionEvent::DisplayCaption { text, .. } = e {
                !text.trim().is_empty()
            } else {
                false
            }
        });
        assert!(
            !has_nonempty_display,
            "stage was erased so display should be empty"
        );
    }

    #[test]
    fn test_screen_buffer_scroll_up() {
        let mut buf = ScreenBuffer::new();
        buf.set(
            ROWS - 2,
            0,
            CharCell {
                ch: 'A',
                ..CharCell::blank()
            },
        );
        buf.scroll_up();
        assert_eq!(buf.cells[ROWS - 3][0].ch, 'A');
        assert!(!buf.cells[ROWS - 1].iter().any(|c| c.is_visible()));
    }

    #[test]
    fn test_screen_buffer_has_content() {
        let mut buf = ScreenBuffer::new();
        assert!(!buf.has_content());
        buf.set(
            0,
            0,
            CharCell {
                ch: 'X',
                ..CharCell::blank()
            },
        );
        assert!(buf.has_content());
    }
}
