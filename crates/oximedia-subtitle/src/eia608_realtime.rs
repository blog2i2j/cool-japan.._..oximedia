//! EIA-608 (CEA-608) realtime caption decoder for live streams.
//!
//! Processes 2-byte pairs as they arrive from the bitstream and decodes them
//! into caption commands or printable characters.  Implements roll-up caption
//! display logic with duplicate-pair detection as required by the CEA-608
//! specification (each pair is transmitted twice; only the first occurrence
//! should be acted upon).
//!
//! ## Example
//! ```
//! use oximedia_subtitle::eia608_realtime::{Eia608Decoder, Eia608Command};
//!
//! let mut dec = Eia608Decoder::new();
//! // Push printable ASCII 'H' (0x48) on channel 1 (b1 = 0x48 with parity)
//! dec.push_pair(0x48, 0x69); // H, i
//! assert_eq!(dec.current_line(), "Hi");
//! ```

// ============================================================================
// Eia608Command
// ============================================================================

/// A decoded EIA-608 caption command or character pair.
#[derive(Debug, Clone, PartialEq)]
pub enum Eia608Command {
    /// One or two printable characters decoded from a byte pair.
    Characters(char, Option<char>),
    /// End of Caption (flip double-buffer) — byte pair 0x14/0x1C + 0x2F.
    EndOfCaption,
    /// Erase Displayed Memory — 0x14/0x1C + 0x2C.
    EraseDisplayed,
    /// Erase Non-Displayed Memory — 0x14/0x1C + 0x2E.
    EraseNonDisplayed,
    /// Carriage Return — 0x14/0x1C + 0x2D.
    CarriageReturn,
    /// Pop-on: Resume Caption Loading — 0x14/0x1C + 0x20.
    ResumeCaptionLoading,
    /// Paint-on: Resume Direct Captioning — 0x14/0x1C + 0x29.
    ResumeDirectCaptioning,
    /// Roll-up caption mode: RU2 (0x25), RU3 (0x26), RU4 (0x27).
    RollUp(u8),
    /// Tab offset (1–3 columns) from miscellaneous control 0x17 + 0x21–0x23.
    TabOffset(u8),
    /// Unknown or unhandled byte pair.
    Unknown(u8, u8),
}

impl Eia608Command {
    /// Decode a 2-byte EIA-608 pair.
    ///
    /// Both bytes have their parity bit (bit 7) stripped before decoding.
    #[must_use]
    pub fn decode(b1: u8, b2: u8) -> Self {
        // Strip parity bit from both bytes
        let b1 = b1 & 0x7F;
        let b2 = b2 & 0x7F;

        match b1 {
            // Channel 1 control codes (0x14) and channel 2 control codes (0x1C)
            0x14 | 0x1C => match b2 {
                0x20 => Self::ResumeCaptionLoading,
                0x25 => Self::RollUp(2),
                0x26 => Self::RollUp(3),
                0x27 => Self::RollUp(4),
                0x29 => Self::ResumeDirectCaptioning,
                0x2C => Self::EraseDisplayed,
                0x2D => Self::CarriageReturn,
                0x2E => Self::EraseNonDisplayed,
                0x2F => Self::EndOfCaption,
                _ => Self::Unknown(b1, b2),
            },
            // Miscellaneous control (tab offsets)
            0x17 => match b2 {
                0x21 => Self::TabOffset(1),
                0x22 => Self::TabOffset(2),
                0x23 => Self::TabOffset(3),
                _ => Self::Unknown(b1, b2),
            },
            // Printable characters — both bytes may carry a character
            _ => {
                let c1 = decode_caption_char(b1);
                let c2 = decode_caption_char(b2);
                match (c1, c2) {
                    (Some(ch1), c2_opt) => Self::Characters(ch1, c2_opt),
                    (None, Some(ch2)) => Self::Characters(ch2, None),
                    (None, None) => Self::Unknown(b1, b2),
                }
            }
        }
    }

    /// Returns `true` if this command is a control command (non-printable).
    #[must_use]
    pub fn is_control(&self) -> bool {
        !matches!(self, Self::Characters(_, _))
    }

    /// Returns `true` if this command carries printable character(s).
    #[must_use]
    pub fn is_printable(&self) -> bool {
        matches!(self, Self::Characters(_, _))
    }
}

/// Decode a single EIA-608 byte (parity already stripped) to a Rust `char`.
///
/// Returns `None` for null bytes (0x00) or pure-parity bytes.
fn decode_caption_char(b: u8) -> Option<char> {
    if b == 0x00 || b == 0x80 {
        return None;
    }
    // EIA-608 printable range is 0x20–0x7E (basic ASCII subset)
    if b >= 0x20 && b <= 0x7E {
        Some(b as char)
    } else {
        None
    }
}

// ============================================================================
// Eia608Decoder
// ============================================================================

/// EIA-608 realtime caption display state with roll-up support.
///
/// Maintains up to 4 lines of rolling captions and handles duplicate-pair
/// detection as required by CEA-608.
pub struct Eia608Decoder {
    /// Completed display lines (oldest first).
    lines: Vec<String>,
    /// Number of rows for roll-up mode (2, 3, or 4).
    roll_up_rows: u8,
    /// The currently-being-built caption line.
    current_line: String,
    /// Last received byte pair for duplicate detection.
    last_pair: (u8, u8),
}

impl Eia608Decoder {
    /// Create a new decoder in roll-up 2 mode.
    #[must_use]
    pub fn new() -> Self {
        Self {
            lines: Vec::new(),
            roll_up_rows: 2,
            current_line: String::new(),
            last_pair: (0xFF, 0xFF), // sentinel — no valid pair is (0xFF,0xFF)
        }
    }

    /// Process a 2-byte pair from the bitstream.
    ///
    /// Returns the current display text (all visible lines joined with '\n')
    /// whenever the display changes, or `None` if the pair was a duplicate or
    /// had no visible effect.
    pub fn push_pair(&mut self, b1: u8, b2: u8) -> Option<String> {
        // CEA-608 §7.7: each non-null pair is sent twice; skip the duplicate.
        if (b1, b2) == self.last_pair && (b1 & 0x7F) != 0x00 {
            self.last_pair = (0xFF, 0xFF); // reset so next distinct pair is accepted
            return None;
        }
        self.last_pair = (b1, b2);

        let cmd = Eia608Command::decode(b1, b2);
        let mut changed = false;

        match &cmd {
            Eia608Command::Characters(c1, c2) => {
                self.current_line.push(*c1);
                if let Some(c) = c2 {
                    self.current_line.push(*c);
                }
                changed = true;
            }
            Eia608Command::CarriageReturn => {
                let finished = std::mem::take(&mut self.current_line);
                self.lines.push(finished);
                // Keep only the last `roll_up_rows` lines
                let max = self.roll_up_rows as usize;
                if self.lines.len() > max {
                    let drain_count = self.lines.len() - max;
                    self.lines.drain(..drain_count);
                }
                changed = true;
            }
            Eia608Command::EraseDisplayed => {
                self.lines.clear();
                self.current_line.clear();
                changed = true;
            }
            Eia608Command::RollUp(rows) => {
                self.roll_up_rows = *rows;
                changed = true;
            }
            Eia608Command::EndOfCaption
            | Eia608Command::EraseNonDisplayed
            | Eia608Command::ResumeCaptionLoading
            | Eia608Command::ResumeDirectCaptioning => {
                changed = true;
            }
            Eia608Command::TabOffset(_) | Eia608Command::Unknown(_, _) => {}
        }

        if changed {
            Some(self.display_text())
        } else {
            None
        }
    }

    /// The lines currently visible on screen (at most `roll_up_rows` entries).
    #[must_use]
    pub fn current_display(&self) -> Vec<&str> {
        let max = self.roll_up_rows as usize;
        let visible_lines: Vec<&str> = self.lines.iter().map(String::as_str).collect();
        let start = visible_lines.len().saturating_sub(max);
        visible_lines[start..].to_vec()
    }

    /// The caption text currently being assembled (not yet committed).
    #[must_use]
    pub fn current_line(&self) -> &str {
        &self.current_line
    }

    /// The current roll-up row count (2, 3, or 4).
    #[must_use]
    pub fn roll_up_rows(&self) -> u8 {
        self.roll_up_rows
    }

    /// Reset all state to initial conditions.
    pub fn reset(&mut self) {
        self.lines.clear();
        self.current_line.clear();
        self.roll_up_rows = 2;
        self.last_pair = (0xFF, 0xFF);
    }

    /// Collect all visible text into a single string, lines joined by '\n'.
    fn display_text(&self) -> String {
        let mut parts: Vec<&str> = self.current_display();
        if !self.current_line.is_empty() {
            parts.push(&self.current_line);
        }
        parts.join("\n")
    }
}

impl Default for Eia608Decoder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Eia608Command::decode
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_printable_chars() {
        let cmd = Eia608Command::decode(b'H', b'i');
        assert_eq!(cmd, Eia608Command::Characters('H', Some('i')));
    }

    #[test]
    fn test_decode_printable_single_char_null_second() {
        // Second byte is null → only one character
        let cmd = Eia608Command::decode(b'A', 0x00);
        assert_eq!(cmd, Eia608Command::Characters('A', None));
    }

    #[test]
    fn test_decode_end_of_caption() {
        // 0x14 + 0x2F  (parity bit already clear)
        let cmd = Eia608Command::decode(0x14, 0x2F);
        assert_eq!(cmd, Eia608Command::EndOfCaption);
    }

    #[test]
    fn test_decode_end_of_caption_ch2() {
        // Channel 2 variant 0x1C + 0x2F
        let cmd = Eia608Command::decode(0x1C, 0x2F);
        assert_eq!(cmd, Eia608Command::EndOfCaption);
    }

    #[test]
    fn test_decode_erase_displayed() {
        let cmd = Eia608Command::decode(0x14, 0x2C);
        assert_eq!(cmd, Eia608Command::EraseDisplayed);
    }

    #[test]
    fn test_decode_erase_non_displayed() {
        let cmd = Eia608Command::decode(0x14, 0x2E);
        assert_eq!(cmd, Eia608Command::EraseNonDisplayed);
    }

    #[test]
    fn test_decode_roll_up_2() {
        let cmd = Eia608Command::decode(0x14, 0x25);
        assert_eq!(cmd, Eia608Command::RollUp(2));
    }

    #[test]
    fn test_decode_roll_up_3() {
        let cmd = Eia608Command::decode(0x14, 0x26);
        assert_eq!(cmd, Eia608Command::RollUp(3));
    }

    #[test]
    fn test_decode_roll_up_4() {
        let cmd = Eia608Command::decode(0x14, 0x27);
        assert_eq!(cmd, Eia608Command::RollUp(4));
    }

    #[test]
    fn test_decode_tab_offset() {
        assert_eq!(Eia608Command::decode(0x17, 0x21), Eia608Command::TabOffset(1));
        assert_eq!(Eia608Command::decode(0x17, 0x22), Eia608Command::TabOffset(2));
        assert_eq!(Eia608Command::decode(0x17, 0x23), Eia608Command::TabOffset(3));
    }

    #[test]
    fn test_is_control_and_is_printable() {
        let ctrl = Eia608Command::decode(0x14, 0x2C);
        assert!(ctrl.is_control());
        assert!(!ctrl.is_printable());

        let print = Eia608Command::decode(b'X', b'Y');
        assert!(!print.is_control());
        assert!(print.is_printable());
    }

    #[test]
    fn test_parity_bit_stripped() {
        // 'H' with parity bit set = 0xC8; should still decode to 'H'
        let cmd = Eia608Command::decode(0xC8, 0x00);
        assert_eq!(cmd, Eia608Command::Characters('H', None));
    }

    // -----------------------------------------------------------------------
    // Eia608Decoder
    // -----------------------------------------------------------------------

    #[test]
    fn test_push_pair_appends_chars_to_current_line() {
        let mut dec = Eia608Decoder::new();
        dec.push_pair(b'H', b'i');
        assert_eq!(dec.current_line(), "Hi");
    }

    #[test]
    fn test_duplicate_pair_is_skipped() {
        let mut dec = Eia608Decoder::new();
        dec.push_pair(b'A', b'B'); // first occurrence — accepted
        dec.push_pair(b'A', b'B'); // duplicate — skipped
        assert_eq!(dec.current_line(), "AB");
    }

    #[test]
    fn test_carriage_return_scrolls_lines() {
        let mut dec = Eia608Decoder::new();
        dec.push_pair(b'H', b'i'); // "Hi"
        // Force last_pair reset so CR is accepted (different byte pair)
        dec.push_pair(0x14, 0x2D); // CarriageReturn
        // "Hi" should now be in lines, current_line empty
        assert!(dec.current_line().is_empty());
        let display = dec.current_display();
        assert_eq!(display, vec!["Hi"]);
    }

    #[test]
    fn test_roll_up_limits_visible_lines() {
        let mut dec = Eia608Decoder::new();
        dec.push_pair(0x14, 0x25); // RU2

        // Push 3 lines
        for word in &["Line1", "Line2", "Line3"] {
            for ch in word.bytes() {
                dec.push_pair(ch, 0x00);
            }
            dec.push_pair(0x14, 0x2D); // CR
        }

        // Only 2 lines should be visible
        let display = dec.current_display();
        assert!(display.len() <= 2);
    }

    #[test]
    fn test_erase_displayed_clears_all() {
        let mut dec = Eia608Decoder::new();
        dec.push_pair(b'X', b'Y');
        dec.push_pair(0x14, 0x2C); // EraseDisplayed
        assert!(dec.current_display().is_empty());
        assert!(dec.current_line().is_empty());
    }

    #[test]
    fn test_reset_clears_state() {
        let mut dec = Eia608Decoder::new();
        dec.push_pair(b'A', b'B');
        dec.push_pair(0x14, 0x26); // RU3
        dec.reset();
        assert!(dec.current_line().is_empty());
        assert!(dec.current_display().is_empty());
        assert_eq!(dec.roll_up_rows(), 2);
    }

    #[test]
    fn test_roll_up_rows_updated() {
        let mut dec = Eia608Decoder::new();
        assert_eq!(dec.roll_up_rows(), 2);
        dec.push_pair(0x14, 0x27); // RU4
        assert_eq!(dec.roll_up_rows(), 4);
    }

    #[test]
    fn test_current_display_respects_roll_up_rows() {
        let mut dec = Eia608Decoder::new();
        dec.push_pair(0x14, 0x27); // RU4 — 4 rows visible

        // Push 5 completed lines
        for i in 0u8..5 {
            dec.push_pair(b'0' + i, 0x00);
            dec.push_pair(0x14, 0x2D); // CR
        }

        let display = dec.current_display();
        assert!(display.len() <= 4, "display has {} lines", display.len());
    }
}
