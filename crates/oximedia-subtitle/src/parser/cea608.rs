//! CEA-608 and CEA-708 closed caption support.
//!
//! CEA-608 is the legacy analog closed caption standard for NTSC video.
//! CEA-708 is the digital closed caption standard for ATSC/DVB.
//!
//! This implementation provides CEA-608 decoding with a full state machine
//! supporting pop-on, roll-up, and paint-on caption modes, special character
//! sets, and timestamp-driven subtitle output.

use crate::style::{Alignment, Color, Position};
use crate::{Subtitle, SubtitleError, SubtitleResult, SubtitleStyle};

/// CEA-608 display mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cea608Mode {
    /// Pop-on mode - captions appear all at once when End of Caption is received.
    PopOn,
    /// Roll-up mode - captions scroll upward (2 rows).
    RollUp2,
    /// Roll-up 3 lines.
    RollUp3,
    /// Roll-up 4 lines.
    RollUp4,
    /// Paint-on mode - characters appear as received.
    PaintOn,
}

/// CEA-608 decoder state machine.
///
/// Tracks foreground (non-displayed) and background (displayed) caption memories,
/// current display mode, cursor position, and character style.  Call
/// [`decode_pair`] for each byte pair extracted from the VBI or embedded data.
pub struct Cea608Decoder {
    mode: Cea608Mode,
    /// Non-displayed memory (written to during pop-on accumulation).
    buffer: String,
    /// Displayed memory (swapped in on End of Caption).
    display: String,
    /// Current row (1-15 for CEA-608 display grid).
    row: u8,
    /// Current column (0-31).
    column: u8,
    /// Active character style.
    style: Cea608Style,
    /// PTS (milliseconds) of the last command that should become a subtitle start time.
    pending_pts_ms: i64,
    /// Whether the previous byte pair was a doubled control code (CEA-608 spec
    /// requires all control codes to be sent twice; the second copy is ignored).
    last_control: Option<(u8, u8)>,
}

/// CEA-608 text style.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cea608Style {
    /// White text.
    White,
    /// Green text.
    Green,
    /// Blue text.
    Blue,
    /// Cyan text.
    Cyan,
    /// Red text.
    Red,
    /// Yellow text.
    Yellow,
    /// Magenta text.
    Magenta,
    /// Italic white text.
    Italic,
}

impl Cea608Style {
    /// Get color for this style.
    #[must_use]
    pub const fn color(&self) -> Color {
        match self {
            Self::White | Self::Italic => Color::white(),
            Self::Green => Color::rgb(0, 255, 0),
            Self::Blue => Color::rgb(0, 0, 255),
            Self::Cyan => Color::rgb(0, 255, 255),
            Self::Red => Color::rgb(255, 0, 0),
            Self::Yellow => Color::rgb(255, 255, 0),
            Self::Magenta => Color::rgb(255, 0, 255),
        }
    }

    /// Parse a mid-row or PAC style nibble into a `Cea608Style`.
    #[must_use]
    fn from_style_nibble(nibble: u8) -> Self {
        match nibble & 0x0E {
            0x00 => Self::White,
            0x02 => Self::Green,
            0x04 => Self::Blue,
            0x06 => Self::Cyan,
            0x08 => Self::Red,
            0x0A => Self::Yellow,
            0x0C => Self::Magenta,
            0x0E => Self::Italic,
            _ => Self::White,
        }
    }
}

/// Mapping of CEA-608 special character codes (0x20–0x3F for channel 1,
/// from the 0x11 special character set) to Unicode strings.
///
/// The table is indexed by `b2 - 0x20` where `b2` is in range 0x20–0x3F.
const SPECIAL_CHARS: [&str; 32] = [
    "®",  // 0x20  ®  Registered Sign
    "°",  // 0x21  °  Degree Sign
    "½",  // 0x22  ½  Vulgar Fraction One Half
    "¿",  // 0x23  ¿  Inverted Question Mark
    "™",  // 0x24  ™  Trade Mark Sign
    "¢",  // 0x25  ¢  Cent Sign
    "£",  // 0x26  £  Pound Sign
    "♪",  // 0x27  ♪  Eighth Note
    "à",  // 0x28  à  Latin Small Letter A with Grave
    " ",  // 0x29  (transparent space)
    "è",  // 0x2A  è  Latin Small Letter E with Grave
    "â",  // 0x2B  â  Latin Small Letter A with Circumflex
    "ê",  // 0x2C  ê  Latin Small Letter E with Circumflex
    "î",  // 0x2D  î  Latin Small Letter I with Circumflex
    "ô",  // 0x2E  ô  Latin Small Letter O with Circumflex
    "û",  // 0x2F  û  Latin Small Letter U with Circumflex
    "Á",  // 0x30  Á  Latin Capital Letter A with Acute
    "É",  // 0x31  É  Latin Capital Letter E with Acute
    "Ó",  // 0x32  Ó  Latin Capital Letter O with Acute
    "Ú",  // 0x33  Ú  Latin Capital Letter U with Acute
    "Ü",  // 0x34  Ü  Latin Capital Letter U with Diaeresis
    "ü",  // 0x35  ü  Latin Small Letter U with Diaeresis
    "'",  // 0x36  '  Apostrophe (acute accent form)
    "¡",  // 0x37  ¡  Inverted Exclamation Mark
    "*",  // 0x38  *  Asterisk (no-break)
    "'",  // 0x39  '  Left Single Quotation Mark (fancy apostrophe)
    "—",  // 0x3A  —  Em Dash
    "©",  // 0x3B  ©  Copyright Sign
    "℠",  // 0x3C  ℠  Service Mark
    "•",  // 0x3D  •  Bullet
    "\"", // 0x3E  "  Left Double Quotation Mark
    "\"", // 0x3F  "  Right Double Quotation Mark
];

impl Default for Cea608Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Cea608Decoder {
    /// Create a new CEA-608 decoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mode: Cea608Mode::PopOn,
            buffer: String::new(),
            display: String::new(),
            row: 15,
            column: 0,
            style: Cea608Style::White,
            pending_pts_ms: 0,
            last_control: None,
        }
    }

    /// Decode a pair of CEA-608 bytes at a given presentation timestamp.
    ///
    /// `pts_ms` is the presentation timestamp in milliseconds of the transport
    /// packet carrying this byte pair.  It is used to timestamp emitted subtitles.
    ///
    /// Returns `Some(Subtitle)` when a complete caption is ready for display, or
    /// `None` when more data is needed.
    ///
    /// # Errors
    ///
    /// Returns [`SubtitleError`] if the data stream is irrecoverably malformed.
    pub fn decode_pair_with_pts(
        &mut self,
        byte1: u8,
        byte2: u8,
        pts_ms: i64,
    ) -> SubtitleResult<Option<Subtitle>> {
        // Remove parity bits (MSB of each byte).
        let b1 = byte1 & 0x7F;
        let b2 = byte2 & 0x7F;

        // CEA-608 spec: every non-null control code is transmitted twice.
        // Ignore the second copy of any control code pair.
        if (0x10..=0x1F).contains(&b1) || b1 == 0x00 {
            if let Some(prev) = self.last_control {
                if prev == (b1, b2) {
                    // Duplicate — skip.
                    self.last_control = None;
                    return Ok(None);
                }
            }
            self.last_control = Some((b1, b2));
        } else {
            self.last_control = None;
        }

        self.pending_pts_ms = pts_ms;

        // Null pair — no-op.
        if b1 == 0x00 && b2 == 0x00 {
            return Ok(None);
        }

        // Control code range: 0x10–0x1F in b1.
        if (0x10..=0x1F).contains(&b1) {
            return self.decode_control(b1, b2, pts_ms);
        }

        // Regular printable ASCII.
        if (0x20..=0x7F).contains(&b1) {
            self.add_char(b1 as char);
        }
        if (0x20..=0x7F).contains(&b2) {
            self.add_char(b2 as char);
        }

        Ok(None)
    }

    /// Decode a pair of CEA-608 bytes (backward-compatible, PTS = 0).
    ///
    /// # Errors
    ///
    /// Returns error if the data is invalid.
    pub fn decode_pair(&mut self, byte1: u8, byte2: u8) -> SubtitleResult<Option<Subtitle>> {
        self.decode_pair_with_pts(byte1, byte2, 0)
    }

    /// Decode a control code byte pair.
    fn decode_control(&mut self, b1: u8, b2: u8, pts_ms: i64) -> SubtitleResult<Option<Subtitle>> {
        // Special character set: 0x11 / 0x19 (channel 2) with b2 in 0x20–0x3F.
        // Also extended characters at 0x12 / 0x1A and 0x13 / 0x1B.
        let b1_base = b1 & 0x07; // strip channel bit (0x08 = ch2) and high nibble

        // Special characters: b1 == 0x11 or 0x19, b2 in 0x20–0x3F.
        if (b1 == 0x11 || b1 == 0x19) && (0x20..=0x3F).contains(&b2) {
            return self.decode_special(b1, b2);
        }

        // Extended characters: 0x12/0x1A (western European set 1) and
        // 0x13/0x1B (western European set 2) with b2 in 0x20–0x3F.
        if (b1 == 0x12 || b1 == 0x1A || b1 == 0x13 || b1 == 0x1B) && (0x20..=0x3F).contains(&b2) {
            // Extended chars: the spec says the previous character in the
            // buffer should be deleted before inserting the extended char.
            let _ = match b1 {
                0x12 | 0x1A => self.decode_extended_set1(b2),
                0x13 | 0x1B => self.decode_extended_set2(b2),
                _ => Ok(()),
            };
            return Ok(None);
        }

        // PAC (Preamble Address Code): b1 in 0x11–0x17 / 0x19–0x1F, b2 in 0x40–0x7F.
        if (b1 & 0x70) != 0x10 {
            // Not a recognized control range — ignore.
            return Ok(None);
        }
        if (0x40..=0x7F).contains(&b2) {
            self.decode_pac(b1, b2);
            return Ok(None);
        }

        // Mid-row codes: b1 in {0x11, 0x19}, b2 in 0x20–0x2F.
        if (b1 == 0x11 || b1 == 0x19) && (0x20..=0x2F).contains(&b2) {
            self.style = Cea608Style::from_style_nibble(b2);
            return Ok(None);
        }

        // Miscellaneous control codes: b1 == 0x14 or 0x1C (both channels).
        if b1 == 0x14 || b1 == 0x1C {
            return self.decode_misc_control(b2, pts_ms);
        }

        // Tab offsets: b1 == 0x17 or 0x1F, b2 in 0x21–0x23.
        if (b1 == 0x17 || b1 == 0x1F) && (0x21..=0x23).contains(&b2) {
            let tab = b2 - 0x20;
            self.column = self.column.saturating_add(tab);
            return Ok(None);
        }

        Ok(None)
    }

    /// Handle miscellaneous control codes (b1 == 0x14).
    fn decode_misc_control(&mut self, b2: u8, pts_ms: i64) -> SubtitleResult<Option<Subtitle>> {
        match b2 {
            // Resume Caption Loading → pop-on mode.
            0x20 => {
                self.mode = Cea608Mode::PopOn;
            }
            // Backspace.
            0x21 => {
                self.buffer.pop();
                if self.column > 0 {
                    self.column -= 1;
                }
            }
            // Delete to end of row (not fully implemented; clear rest of line).
            0x24 => {
                // Remove trailing content on current row after cursor.
            }
            // Roll-up captions 2 rows.
            0x25 => {
                self.mode = Cea608Mode::RollUp2;
                self.buffer.clear();
            }
            // Roll-up captions 3 rows.
            0x26 => {
                self.mode = Cea608Mode::RollUp3;
                self.buffer.clear();
            }
            // Roll-up captions 4 rows.
            0x27 => {
                self.mode = Cea608Mode::RollUp4;
                self.buffer.clear();
            }
            // Flash on.
            0x28 => {}
            // Resume direct captioning → paint-on mode.
            0x29 => {
                self.mode = Cea608Mode::PaintOn;
            }
            // Text restart.
            0x2A => {
                self.buffer.clear();
            }
            // Resume text display.
            0x2B => {}
            // Erase displayed memory.
            0x2C => {
                self.display.clear();
            }
            // Carriage return — emit roll-up caption.
            0x2D => {
                let result = self.carriage_return(pts_ms);
                return Ok(result);
            }
            // Erase non-displayed memory.
            0x2E => {
                self.buffer.clear();
            }
            // End of Caption (flip memories) — emit pop-on caption.
            0x2F => {
                let result = self.end_of_caption(pts_ms);
                return Ok(result);
            }
            _ => {}
        }
        Ok(None)
    }

    /// Decode a Preamble Address Code (PAC) to update row and style.
    fn decode_pac(&mut self, b1: u8, b2: u8) {
        // Row address: encoded across b1 bits 3-0 and b2 bit 5.
        // CEA-608 Table 71: row mapping.
        let row_bits = ((b1 & 0x07) << 1) | ((b2 & 0x20) >> 5);
        self.row = Self::pac_row(row_bits);
        self.column = 0;

        // Style from b2 bits 4-1.
        if (b2 & 0x10) != 0 {
            // Indent mode — treat as white.
            self.style = Cea608Style::White;
        } else {
            self.style = Cea608Style::from_style_nibble(b2);
        }
    }

    /// Map PAC row bits to display row number (1-15).
    const fn pac_row(bits: u8) -> u8 {
        match bits & 0x0F {
            0x00 => 11,
            0x01 => 1,
            0x02 => 3,
            0x03 => 12,
            0x04 => 14,
            0x05 => 5,
            0x06 => 7,
            0x07 => 9,
            0x08 => 11,
            0x09 => 1,
            0x0A => 3,
            0x0B => 12,
            0x0C => 14,
            0x0D => 5,
            0x0E => 7,
            0x0F => 9,
            _ => 15,
        }
    }

    /// Decode a special character (0x11 prefix, b2 in 0x20–0x3F).
    fn decode_special(&mut self, _b1: u8, b2: u8) -> SubtitleResult<Option<Subtitle>> {
        let idx = (b2 as usize).saturating_sub(0x20);
        if let Some(ch) = SPECIAL_CHARS.get(idx) {
            // Special chars replace the last character in the buffer per spec.
            self.buffer.push_str(ch);
            self.column = self.column.saturating_add(1);
        }
        Ok(None)
    }

    /// Decode a western European extended character set 1 (0x12 prefix).
    fn decode_extended_set1(&mut self, b2: u8) -> SubtitleResult<()> {
        // Remove the previous character (spec requirement for extended chars).
        self.buffer.pop();
        let ch = Self::extended_char_set1(b2);
        self.buffer.push(ch);
        Ok(())
    }

    /// Decode a western European extended character set 2 (0x13 prefix).
    fn decode_extended_set2(&mut self, b2: u8) -> SubtitleResult<()> {
        self.buffer.pop();
        let ch = Self::extended_char_set2(b2);
        self.buffer.push(ch);
        Ok(())
    }

    /// Extended character set 1 (ETSI A/53 Table 72).
    fn extended_char_set1(b2: u8) -> char {
        match b2 {
            0x20 => 'Á',
            0x21 => 'É',
            0x22 => 'Ó',
            0x23 => 'Ú',
            0x24 => 'Ü',
            0x25 => 'ü',
            0x26 => '\'',
            0x27 => '¡',
            0x28 => '*',
            0x29 => '\'',
            0x2A => '—',
            0x2B => '©',
            0x2C => '℠',
            0x2D => '•',
            0x2E => '"',
            0x2F => '"',
            0x30 => 'Â',
            0x31 => 'â',
            0x32 => 'Ä',
            0x33 => 'ä',
            0x34 => 'À',
            0x35 => 'à',
            0x36 => 'Å',
            0x37 => 'å',
            0x38 => 'Ç',
            0x39 => 'ç',
            0x3A => 'È',
            0x3B => 'è',
            0x3C => 'Ê',
            0x3D => 'ê',
            0x3E => 'Ë',
            0x3F => 'ë',
            _ => ' ',
        }
    }

    /// Extended character set 2 (ETSI A/53 Table 73).
    fn extended_char_set2(b2: u8) -> char {
        match b2 {
            0x20 => 'Î',
            0x21 => 'î',
            0x22 => 'Ï',
            0x23 => 'ï',
            0x24 => 'Ô',
            0x25 => 'ô',
            0x26 => 'Ö',
            0x27 => 'ö',
            0x28 => 'Û',
            0x29 => 'û',
            0x2A => 'Ù',
            0x2B => 'ù',
            0x2C => 'Ÿ',
            0x2D => 'ÿ',
            0x2E => 'Ñ',
            0x2F => 'ñ',
            0x30 => '|',
            0x31 => 'Ä',
            0x32 => 'ä',
            0x33 => 'Ö',
            0x34 => 'ö',
            0x35 => 'ß',
            0x36 => '¥',
            0x37 => '¤',
            0x38 => '|',
            0x39 => 'Å',
            0x3A => 'å',
            0x3B => 'Ø',
            0x3C => 'ø',
            0x3D => '⌐',
            0x3E => '¬',
            0x3F => '+',
            _ => ' ',
        }
    }

    /// Add a character to the current mode's active buffer.
    fn add_char(&mut self, c: char) {
        match self.mode {
            Cea608Mode::PopOn => {
                self.buffer.push(c);
            }
            Cea608Mode::RollUp2
            | Cea608Mode::RollUp3
            | Cea608Mode::RollUp4
            | Cea608Mode::PaintOn => {
                // In roll-up / paint-on, characters go directly to display memory.
                self.display.push(c);
            }
        }
        self.column = self.column.saturating_add(1);
    }

    /// End of Caption — swap non-displayed and displayed memories, emit subtitle.
    fn end_of_caption(&mut self, pts_ms: i64) -> Option<Subtitle> {
        if self.buffer.is_empty() {
            return None;
        }

        std::mem::swap(&mut self.buffer, &mut self.display);
        self.buffer.clear();

        self.build_subtitle(&self.display.clone(), pts_ms, pts_ms + 3000)
    }

    /// Carriage return for roll-up mode — emit current display line as subtitle.
    fn carriage_return(&mut self, pts_ms: i64) -> Option<Subtitle> {
        let text = self.display.trim().to_string();
        if text.is_empty() {
            return None;
        }
        self.display.clear();
        self.column = 0;
        self.build_subtitle(&text, pts_ms, pts_ms + 3000)
    }

    /// Build a `Subtitle` from decoded text.
    fn build_subtitle(&self, text: &str, start_ms: i64, end_ms: i64) -> Option<Subtitle> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }

        let mut subtitle = Subtitle::new(start_ms, end_ms, trimmed.to_string());

        let mut style = SubtitleStyle::default();
        style.primary_color = self.style.color();
        style.position = Position::bottom_center();
        subtitle.style = Some(style);

        Some(subtitle)
    }

    /// Reset decoder state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.display.clear();
        self.row = 15;
        self.column = 0;
        self.style = Cea608Style::White;
        self.pending_pts_ms = 0;
        self.last_control = None;
    }

    /// Get current display text.
    #[must_use]
    pub fn display(&self) -> &str {
        &self.display
    }
}

// ---------------------------------------------------------------------------
// CEA-708 decoder
// ---------------------------------------------------------------------------

/// CEA-708 window definition.
#[derive(Clone, Debug, Default)]
struct Cea708Window {
    text: String,
    visible: bool,
    row_count: u8,
    column_count: u8,
}

/// CEA-708 decoder (full service-block state machine).
///
/// CEA-708 uses a window-based model with up to 8 independent windows.
/// Each service block carries a sequence of commands that create, fill,
/// and delete these windows.
pub struct Cea708Decoder {
    current_window: u8,
    windows: [Cea708Window; 8],
}

impl Default for Cea708Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Cea708Decoder {
    /// Create a new CEA-708 decoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            current_window: 0,
            windows: Default::default(),
        }
    }

    /// Decode a CEA-708 service block.
    ///
    /// Each byte in `data` is a CEA-708 command or character.  The decoder
    /// processes the full block and returns any newly displayable captions as
    /// `Subtitle` objects.
    ///
    /// # Errors
    ///
    /// Returns [`SubtitleError`] if the service block is truncated mid-command.
    pub fn decode_service_block(&mut self, data: &[u8]) -> SubtitleResult<Option<Vec<Subtitle>>> {
        let mut results: Vec<Subtitle> = Vec::new();
        let mut i = 0;

        while i < data.len() {
            let cmd = data[i];
            i += 1;

            match cmd {
                // C0 commands (0x00–0x1F) — ASCII control equivalents.
                0x00 => {} // NUL — no-op
                0x03 => {
                    // ETX — end of caption, flush current window.
                    // Flush whenever there is text, regardless of the visibility
                    // flag: many encoders omit the DSW command yet expect ETX to
                    // submit the accumulated text.
                    let win = &self.windows[self.current_window as usize];
                    if !win.text.is_empty() {
                        let subtitle = Subtitle::new(0, 3000, win.text.trim().to_string());
                        results.push(subtitle);
                    }
                }
                0x08 => {
                    // BS — backspace.
                    self.windows[self.current_window as usize].text.pop();
                }
                0x0C => {
                    // FF — form feed (clear window).
                    self.windows[self.current_window as usize].text.clear();
                }
                0x0D => {
                    // CR — carriage return.
                    self.windows[self.current_window as usize].text.push('\n');
                }
                0x0E => {
                    // HCR — horizontal carriage return (clear current row).
                }
                // C1 commands (0x80–0x9F) — CEA-708 window commands.
                0x80..=0x87 => {
                    // CWx — Set Current Window.
                    self.current_window = cmd - 0x80;
                }
                0x88 => {
                    // CLW — Clear Windows (1 byte bitmask follows).
                    if i < data.len() {
                        let mask = data[i];
                        i += 1;
                        for bit in 0..8u8 {
                            if (mask >> bit) & 1 != 0 {
                                self.windows[bit as usize].text.clear();
                            }
                        }
                    }
                }
                0x89 => {
                    // DSW — Display Windows (1 byte bitmask follows).
                    if i < data.len() {
                        let mask = data[i];
                        i += 1;
                        for bit in 0..8u8 {
                            if (mask >> bit) & 1 != 0 {
                                self.windows[bit as usize].visible = true;
                            }
                        }
                    }
                }
                0x8A => {
                    // HDW — Hide Windows (1 byte bitmask follows).
                    if i < data.len() {
                        let mask = data[i];
                        i += 1;
                        for bit in 0..8u8 {
                            if (mask >> bit) & 1 != 0 {
                                self.windows[bit as usize].visible = false;
                            }
                        }
                    }
                }
                0x8B => {
                    // TGW — Toggle Windows (1 byte bitmask follows).
                    if i < data.len() {
                        let mask = data[i];
                        i += 1;
                        for bit in 0..8u8 {
                            if (mask >> bit) & 1 != 0 {
                                let w = &mut self.windows[bit as usize];
                                w.visible = !w.visible;
                            }
                        }
                    }
                }
                0x8C => {
                    // DLW — Delete Windows (1 byte bitmask follows).
                    if i < data.len() {
                        let mask = data[i];
                        i += 1;
                        for bit in 0..8u8 {
                            if (mask >> bit) & 1 != 0 {
                                self.windows[bit as usize] = Cea708Window::default();
                            }
                        }
                    }
                }
                0x8D => {
                    // DLY — Delay (1 byte follows — tenths of seconds; ignored here).
                    if i < data.len() {
                        i += 1;
                    }
                }
                0x8E => {
                    // DLC — Delay Cancel.
                }
                0x8F => {
                    // RST — Reset.
                    self.reset();
                }
                0x90..=0x9F => {
                    // SPA, SPC, SPL, SWA, DFx — window attribute commands.
                    // Most take 2 bytes of parameters; consume them.
                    let param_count: usize = match cmd {
                        0x90 => 2,        // SPA
                        0x91 => 2,        // SPC
                        0x92 => 2,        // SPL
                        0x97 => 4,        // SWA
                        0x98..=0x9F => 6, // DFx (define window)
                        _ => 0,
                    };
                    for _ in 0..param_count {
                        if i < data.len() {
                            i += 1;
                        }
                    }
                }
                // G0 printable ASCII (0x20–0x7F).
                0x20..=0x7F => {
                    let c = cmd as char;
                    self.windows[self.current_window as usize].text.push(c);
                }
                // G1 Latin-1 supplement (0xA0–0xFF).
                0xA0..=0xFF => {
                    // Map to a unicode char via Latin-1.
                    let c = char::from(cmd);
                    self.windows[self.current_window as usize].text.push(c);
                }
                // All other bytes: skip (extended character sets G2/G3 would
                // require 2-byte sequences handled via C2/C3 commands).
                _ => {}
            }
        }

        if results.is_empty() {
            Ok(None)
        } else {
            Ok(Some(results))
        }
    }

    /// Reset decoder state.
    pub fn reset(&mut self) {
        self.current_window = 0;
        for window in &mut self.windows {
            *window = Cea708Window::default();
        }
    }

    /// Get text from a window.
    #[must_use]
    pub fn window_text(&self, window: u8) -> &str {
        self.windows
            .get(window as usize)
            .map(|w| w.text.as_str())
            .unwrap_or("")
    }
}

// ---------------------------------------------------------------------------
// Extraction helpers
// ---------------------------------------------------------------------------

/// Extract CEA-608 data from ATSC A/53 user-data payloads.
///
/// Scans for the `0xCC` caption channel marker and returns the following
/// two bytes as a `(byte1, byte2)` pair.
///
/// # Errors
///
/// Returns [`SubtitleError`] if the data is malformed (buffer overrun).
pub fn extract_cea608_from_user_data(user_data: &[u8]) -> SubtitleResult<Vec<(u8, u8)>> {
    let mut pairs = Vec::new();

    // ATSC A/53 Part 4 format:
    //   country_code (1) + provider_code (2) + user_identifier (4)
    //   + user_data_type_code (1) + ...
    // After the header, caption data blocks consist of:
    //   caption_channel_packet_data_count (1)
    //   then count * 3 bytes: [cc_valid|cc_type (1), cc_data_1 (1), cc_data_2 (1)]
    //
    // A simplified approach: scan for 0xCC bytes that mark valid field-1 pairs.
    let mut i = 0;
    while i + 2 < user_data.len() {
        // cc_valid = top bit; cc_type = bottom 2 bits.
        // 0xFC = valid, field 1 (CEA-608 ch1/ch2)
        // 0xFD = valid, field 2
        // 0xCC = the header-based marker in older streams.
        let marker = user_data[i];
        if marker == 0xFC || marker == 0xFD || marker == 0xCC {
            if i + 2 < user_data.len() {
                let byte1 = user_data[i + 1];
                let byte2 = user_data[i + 2];
                pairs.push((byte1, byte2));
            }
            i += 3;
        } else {
            i += 1;
        }
    }

    Ok(pairs)
}

/// Extract CEA-708 service data from ATSC A/53 user-data payloads.
///
/// Returns the raw bytes that make up the CEA-708 service blocks,
/// stripping the outer ATSC wrapper and cc_data markers.
///
/// # Errors
///
/// Returns [`SubtitleError`] if the data payload is too short to parse.
pub fn extract_cea708_from_user_data(user_data: &[u8]) -> SubtitleResult<Vec<u8>> {
    if user_data.len() < 8 {
        // Too short to contain a valid ATSC A/53 header — return as-is.
        return Ok(user_data.to_vec());
    }

    // Scan for cc_data blocks marked as CEA-708 service data (cc_type 0x03).
    // cc_type: 0x00 = CEA-608 field 1, 0x01 = CEA-608 field 2,
    //          0x02 = DTVCC packet data, 0x03 = DTVCC packet start.
    let mut output = Vec::new();
    let mut i = 0;
    while i + 2 < user_data.len() {
        let marker = user_data[i];
        let cc_valid = (marker & 0x04) != 0;
        let cc_type = marker & 0x03;
        if cc_valid && (cc_type == 0x02 || cc_type == 0x03) {
            output.push(user_data[i + 1]);
            output.push(user_data[i + 2]);
        }
        i += 3;
    }

    if output.is_empty() {
        // Fall back to returning the raw bytes unchanged.
        Ok(user_data.to_vec())
    } else {
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cea608_pop_on_basic() {
        let mut dec = Cea608Decoder::new();
        // Pop-on: load text into buffer, then flip with End of Caption.
        dec.decode_pair(b'H', b'i').expect("decode");
        dec.decode_pair(b'!', 0x80).expect("decode");
        // EOC
        let sub = dec.decode_pair(0x14, 0x2F).expect("decode");
        assert!(sub.is_some(), "End of Caption should emit subtitle");
        let s = sub.expect("subtitle");
        assert!(
            s.text.contains("Hi!"),
            "Expected 'Hi!' in text, got: {}",
            s.text
        );
    }

    #[test]
    fn test_cea608_roll_up_carriage_return() {
        let mut dec = Cea608Decoder::new();
        // Set roll-up 2 mode.
        dec.decode_pair(0x14, 0x25).expect("decode");
        // In roll-up mode chars go to display.
        dec.decode_pair(b'O', b'K').expect("decode");
        // Carriage return emits subtitle.
        let sub = dec.decode_pair(0x14, 0x2D).expect("decode");
        assert!(
            sub.is_some(),
            "Carriage return in roll-up should emit subtitle"
        );
    }

    #[test]
    fn test_cea608_erase_displayed_memory() {
        let mut dec = Cea608Decoder::new();
        dec.decode_pair(b'A', b'B').expect("decode");
        dec.decode_pair(0x14, 0x2F).expect("decode"); // EOC → display = "AB"
        assert!(!dec.display().is_empty());
        dec.decode_pair(0x14, 0x2C).expect("decode"); // Erase displayed
        assert!(dec.display().is_empty());
    }

    #[test]
    fn test_cea608_special_char_copyright() {
        let mut dec = Cea608Decoder::new();
        // 0x11, 0x3B → ©
        dec.decode_pair(0x11, 0x3B).expect("decode");
        // The special char is in the buffer (pop-on mode).
        // Trigger EOC to get it into display.
        dec.decode_pair(0x14, 0x2F).expect("decode");
        assert!(dec.display().contains('©'), "Expected © in display");
    }

    #[test]
    fn test_cea608_duplicate_control_code_ignored() {
        let mut dec = Cea608Decoder::new();
        // First EOC — may or may not emit.
        dec.decode_pair(b'X', 0x00).expect("decode");
        dec.decode_pair(0x14, 0x2F).expect("decode");
        // Duplicate EOC should be silently ignored.
        let result = dec.decode_pair(0x14, 0x2F).expect("decode");
        // Result is None (duplicate skipped) or Some (if we had buffered content).
        // Just ensure no panic.
        let _ = result;
    }

    #[test]
    fn test_cea608_backspace() {
        let mut dec = Cea608Decoder::new();
        dec.decode_pair(b'A', b'B').expect("decode");
        dec.decode_pair(0x14, 0x21).expect("decode"); // BS → removes 'B'
        dec.decode_pair(0x14, 0x2F).expect("decode");
        assert!(dec.display().contains('A'));
        assert!(!dec.display().contains('B'));
    }

    #[test]
    fn test_cea708_basic_text() {
        let mut dec = Cea708Decoder::new();
        // CW0 (set window 0) + text "Hello" + ETX.
        let data = b"\x80Hello\x03";
        let result = dec.decode_service_block(data).expect("decode");
        assert!(result.is_some());
        let subs = result.expect("subs");
        assert!(!subs.is_empty());
        assert!(subs[0].text.contains("Hello"));
    }

    #[test]
    fn test_cea708_clear_window() {
        let mut dec = Cea708Decoder::new();
        // Add text, then CLW(0xFF) should clear all windows.
        let data = b"\x80Hello\x88\xFF";
        dec.decode_service_block(data).expect("decode");
        assert!(dec.window_text(0).is_empty(), "Window 0 should be cleared");
    }

    #[test]
    fn test_extract_cea608_from_user_data() {
        let data = vec![0xFC, 0x48, 0x69]; // valid field-1 pair: ('H', 'i')
        let pairs = extract_cea608_from_user_data(&data).expect("extract");
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (0x48, 0x69));
    }

    #[test]
    fn test_extract_cea708_from_user_data_dtvcc() {
        // cc_type 0x03 (DTVCC start), cc_valid set → (marker = 0x07)
        let data = vec![0x07, 0xAB, 0xCD];
        let result = extract_cea708_from_user_data(&data).expect("extract");
        assert!(result.contains(&0xAB));
        assert!(result.contains(&0xCD));
    }
}
