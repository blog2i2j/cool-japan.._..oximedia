#![allow(dead_code)]
//! TTS pronunciation hints for accessible media.
//!
//! This module provides a rich, composable ruleset for transforming raw media
//! text into forms that text-to-speech engines can pronounce correctly and
//! naturally.  It handles three distinct problem areas:
//!
//! 1. **Phonetic alternatives** — map words whose default pronunciation is
//!    wrong or ambiguous to an explicit phonetic string or to a preferred
//!    spelling variant.
//! 2. **Abbreviation expansion** — expand initialisms, acronyms and domain
//!    shorthand before they reach the synthesiser so they are not spelled out
//!    letter by letter when that would be inappropriate.
//! 3. **Number and symbol verbalization** — convert numerals, currency signs,
//!    percentages and common punctuation into the words a reader would
//!    naturally say aloud.
//!
//! The main entry point is [`TtsHintEngine`].  Build one with a
//! [`TtsHintConfig`] and call [`TtsHintEngine::process`] to transform a
//! string.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Phonetic hint
// ---------------------------------------------------------------------------

/// How a phonetic hint should be applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhoneticMode {
    /// Replace the word with a different spelling that most TTS engines
    /// will pronounce correctly (e.g. "GIF" → "jif").
    SpellingVariant,
    /// Emit an IPA string wrapped in SSML `<phoneme>` markup.
    /// (The TTS engine must support SSML phoneme elements.)
    Ipa,
    /// Emit an X-SAMPA string wrapped in SSML `<phoneme>` markup.
    XSampa,
}

/// A single phonetic pronunciation hint.
#[derive(Debug, Clone)]
pub struct PhoneticHint {
    /// The source word or phrase (case-insensitive matching).
    pub word: String,
    /// Replacement pronunciation string.
    pub pronunciation: String,
    /// How to apply the replacement.
    pub mode: PhoneticMode,
}

impl PhoneticHint {
    /// Create a spelling-variant hint.
    #[must_use]
    pub fn spelling(word: impl Into<String>, pronunciation: impl Into<String>) -> Self {
        Self {
            word: word.into(),
            pronunciation: pronunciation.into(),
            mode: PhoneticMode::SpellingVariant,
        }
    }

    /// Create an IPA hint.
    #[must_use]
    pub fn ipa(word: impl Into<String>, ipa: impl Into<String>) -> Self {
        Self {
            word: word.into(),
            pronunciation: ipa.into(),
            mode: PhoneticMode::Ipa,
        }
    }
}

// ---------------------------------------------------------------------------
// Abbreviation entry
// ---------------------------------------------------------------------------

/// Determines how an abbreviation is expanded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbbrevExpansionStyle {
    /// Always expand (e.g. "Dr." → "Doctor").
    AlwaysExpand,
    /// Speak as letters (e.g. "FBI" → "F B I").
    SpellOut,
    /// Speak as a word (e.g. "NATO" → "Nayto").
    Acronym,
}

/// One abbreviation→expansion mapping.
#[derive(Debug, Clone)]
pub struct AbbrevEntry {
    /// The abbreviated form, e.g. `"km"`, `"BBC"`, `"Dr."`.
    pub abbreviation: String,
    /// The expanded form, e.g. `"kilometre"`, `"British Broadcasting Corporation"`.
    pub expansion: String,
    /// How the expansion should be presented to the TTS engine.
    pub style: AbbrevExpansionStyle,
}

impl AbbrevEntry {
    /// Create an always-expand entry.
    #[must_use]
    pub fn expand(abbrev: impl Into<String>, expansion: impl Into<String>) -> Self {
        Self {
            abbreviation: abbrev.into(),
            expansion: expansion.into(),
            style: AbbrevExpansionStyle::AlwaysExpand,
        }
    }

    /// Create a spell-out entry.
    #[must_use]
    pub fn spell_out(abbrev: impl Into<String>, expansion: impl Into<String>) -> Self {
        Self {
            abbreviation: abbrev.into(),
            expansion: expansion.into(),
            style: AbbrevExpansionStyle::SpellOut,
        }
    }

    /// Create an acronym entry (spoken as a word).
    #[must_use]
    pub fn acronym(abbrev: impl Into<String>, expansion: impl Into<String>) -> Self {
        Self {
            abbreviation: abbrev.into(),
            expansion: expansion.into(),
            style: AbbrevExpansionStyle::Acronym,
        }
    }
}

// ---------------------------------------------------------------------------
// Number / symbol verbalization rules
// ---------------------------------------------------------------------------

/// Locale-specific options for number verbalization.
#[derive(Debug, Clone)]
pub struct NumberVerbalizationOptions {
    /// Word to use for the decimal separator (default: `"point"`).
    pub decimal_word: String,
    /// Word to use for negative numbers (default: `"minus"`).
    pub negative_word: String,
    /// Whether to verbalize bare integers with thousand separators
    /// (e.g. 1,000 → "one thousand").  Default: `true`.
    pub expand_thousands: bool,
    /// Currency symbol → spoken name mapping (e.g. `"$"` → `"dollars"`).
    pub currency_map: HashMap<String, String>,
    /// Whether to expand percentages (e.g. `"50%"` → `"50 percent"`).
    /// Default: `true`.
    pub expand_percent: bool,
}

impl Default for NumberVerbalizationOptions {
    fn default() -> Self {
        let mut currency_map = HashMap::new();
        currency_map.insert("$".into(), "dollar".into());
        currency_map.insert("€".into(), "euro".into());
        currency_map.insert("£".into(), "pound".into());
        currency_map.insert("¥".into(), "yen".into());
        currency_map.insert("₩".into(), "won".into());
        Self {
            decimal_word: "point".into(),
            negative_word: "minus".into(),
            expand_thousands: true,
            currency_map,
            expand_percent: true,
        }
    }
}

// ---------------------------------------------------------------------------
// TtsHintConfig
// ---------------------------------------------------------------------------

/// Configuration bundle for [`TtsHintEngine`].
#[derive(Debug, Clone)]
pub struct TtsHintConfig {
    /// Phonetic replacement hints.
    pub phonetic_hints: Vec<PhoneticHint>,
    /// Abbreviation expansion rules.
    pub abbreviations: Vec<AbbrevEntry>,
    /// Number/symbol verbalization options.
    pub number_options: NumberVerbalizationOptions,
    /// When `true`, apply built-in common-English abbreviation defaults.
    pub use_default_abbreviations: bool,
    /// When `true`, apply built-in phonetic hint defaults.
    pub use_default_phonetics: bool,
}

impl Default for TtsHintConfig {
    fn default() -> Self {
        Self {
            phonetic_hints: Vec::new(),
            abbreviations: Vec::new(),
            number_options: NumberVerbalizationOptions::default(),
            use_default_abbreviations: true,
            use_default_phonetics: true,
        }
    }
}

impl TtsHintConfig {
    /// Create a minimal, empty configuration.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            use_default_abbreviations: false,
            use_default_phonetics: false,
            ..Default::default()
        }
    }

    /// Add a phonetic hint.
    #[must_use]
    pub fn with_phonetic(mut self, hint: PhoneticHint) -> Self {
        self.phonetic_hints.push(hint);
        self
    }

    /// Add an abbreviation entry.
    #[must_use]
    pub fn with_abbreviation(mut self, entry: AbbrevEntry) -> Self {
        self.abbreviations.push(entry);
        self
    }
}

// ---------------------------------------------------------------------------
// TtsHintEngine
// ---------------------------------------------------------------------------

/// The main engine that transforms text using pronunciation hints.
///
/// Build via [`TtsHintEngine::new`] and call [`TtsHintEngine::process`].
pub struct TtsHintEngine {
    /// Lookup map: lowercase word → phonetic hint.
    phonetic_map: HashMap<String, PhoneticHint>,
    /// Lookup map: lowercase abbreviation → abbreviation entry.
    abbrev_map: HashMap<String, AbbrevEntry>,
    /// Number/symbol options.
    number_opts: NumberVerbalizationOptions,
}

impl TtsHintEngine {
    /// Construct an engine from the given configuration.
    #[must_use]
    pub fn new(config: TtsHintConfig) -> Self {
        let mut phonetic_map: HashMap<String, PhoneticHint> = HashMap::new();
        let mut abbrev_map: HashMap<String, AbbrevEntry> = HashMap::new();

        if config.use_default_phonetics {
            for hint in default_phonetic_hints() {
                phonetic_map.insert(hint.word.to_lowercase(), hint);
            }
        }
        for hint in config.phonetic_hints {
            phonetic_map.insert(hint.word.to_lowercase(), hint);
        }

        if config.use_default_abbreviations {
            for entry in default_abbreviations() {
                abbrev_map.insert(entry.abbreviation.to_lowercase(), entry);
            }
        }
        for entry in config.abbreviations {
            abbrev_map.insert(entry.abbreviation.to_lowercase(), entry);
        }

        Self {
            phonetic_map,
            abbrev_map,
            number_opts: config.number_options,
        }
    }

    /// Process `text`, applying all hint rules, and return the transformed
    /// string.
    ///
    /// Processing order:
    /// 1. Number/currency/percent verbalization
    /// 2. Abbreviation expansion
    /// 3. Phonetic hint substitution
    #[must_use]
    pub fn process(&self, text: &str) -> String {
        let after_numbers = self.verbalize_numbers(text);
        let after_abbrev = self.expand_abbreviations(&after_numbers);
        self.apply_phonetics(&after_abbrev)
    }

    // -----------------------------------------------------------------------
    // Number verbalization
    // -----------------------------------------------------------------------

    /// Verbalize numbers, currency values and percentages in `text`.
    fn verbalize_numbers(&self, text: &str) -> String {
        let opts = &self.number_opts;
        let mut result = String::with_capacity(text.len() + 32);
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            // ---- currency prefix (e.g. $12.50) ----
            if let Some((sym_len, currency_name)) = self.match_currency(&chars, i) {
                let num_start = i + sym_len;
                if let Some((num_str, num_end)) = extract_number(&chars, num_start) {
                    // "12.50 dollars"
                    result.push_str(&verbalize_number_string(&num_str, opts));
                    result.push(' ');
                    result.push_str(&pluralise(&currency_name, &num_str));
                    i = num_end;
                    continue;
                }
            }

            // ---- signed or bare number ----
            let is_minus = chars[i] == '-'
                && i + 1 < len
                && chars[i + 1].is_ascii_digit();
            let digit_start = if is_minus { i + 1 } else { i };

            if chars[i].is_ascii_digit() || is_minus {
                if let Some((num_str, num_end)) = extract_number(&chars, digit_start) {
                    // check for trailing '%'
                    if opts.expand_percent && num_end < len && chars[num_end] == '%' {
                        if is_minus {
                            result.push_str(&opts.negative_word);
                            result.push(' ');
                        }
                        result.push_str(&verbalize_number_string(&num_str, opts));
                        result.push_str(" percent");
                        i = num_end + 1;
                        continue;
                    }
                    if is_minus {
                        result.push_str(&opts.negative_word);
                        result.push(' ');
                    }
                    result.push_str(&verbalize_number_string(&num_str, opts));
                    i = num_end;
                    continue;
                }
            }

            result.push(chars[i]);
            i += 1;
        }
        result
    }

    /// Returns `(symbol_char_count, currency_name)` when chars[i..] starts
    /// with a known currency symbol.
    fn match_currency(&self, chars: &[char], i: usize) -> Option<(usize, String)> {
        // Try multi-char symbols first (up to 3 chars), then single char
        for sym_len in [3usize, 2, 1] {
            if i + sym_len > chars.len() {
                continue;
            }
            let sym: String = chars[i..i + sym_len].iter().collect();
            if let Some(name) = self.number_opts.currency_map.get(&sym) {
                return Some((sym_len, name.clone()));
            }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Abbreviation expansion
    // -----------------------------------------------------------------------

    /// Expand abbreviations found in `text`.
    ///
    /// A naive word-boundary tokenizer is used: sequences of word characters
    /// (including `'`, `.`) are treated as tokens and looked up in the abbrev
    /// map.  Non-word characters are passed through unchanged.
    fn expand_abbreviations(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len() + 64);
        let mut chars = text.chars().peekable();

        while let Some(&ch) = chars.peek() {
            if ch.is_alphanumeric() || ch == '\'' {
                // Collect word token (may include trailing period for "Dr.")
                let mut token = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_alphanumeric() || c == '\'' {
                        token.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                // Check for trailing period (abbreviation like "Dr.")
                let mut has_dot = false;
                if chars.peek() == Some(&'.') {
                    // Peek ahead — if next after '.' is whitespace/end it's an
                    // abbreviation dot, not sentence end.
                    token.push('.');
                    has_dot = true;
                    chars.next();
                }

                let key = token.to_lowercase();
                if let Some(entry) = self.abbrev_map.get(&key) {
                    match entry.style {
                        AbbrevExpansionStyle::AlwaysExpand => {
                            result.push_str(&entry.expansion);
                        }
                        AbbrevExpansionStyle::SpellOut => {
                            let spelled = spell_out_letters(&entry.abbreviation.replace('.', ""));
                            result.push_str(&spelled);
                        }
                        AbbrevExpansionStyle::Acronym => {
                            result.push_str(&entry.expansion);
                        }
                    }
                } else if has_dot {
                    // Remove trailing dot we added for lookup, restore it
                    result.push_str(token.trim_end_matches('.'));
                    result.push('.');
                } else {
                    result.push_str(&token);
                }
            } else {
                result.push(ch);
                chars.next();
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Phonetic substitution
    // -----------------------------------------------------------------------

    /// Apply phonetic hints to `text`.
    ///
    /// For `SpellingVariant` hints the word is replaced directly.
    /// For `Ipa`/`XSampa` hints an SSML `<phoneme>` element is emitted.
    fn apply_phonetics(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len() + 128);
        let mut chars = text.chars().peekable();

        while let Some(&ch) = chars.peek() {
            if ch.is_alphabetic() {
                let mut token = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_alphabetic() || c == '\'' {
                        token.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                let key = token.to_lowercase();
                if let Some(hint) = self.phonetic_map.get(&key) {
                    match hint.mode {
                        PhoneticMode::SpellingVariant => {
                            result.push_str(&hint.pronunciation);
                        }
                        PhoneticMode::Ipa => {
                            result.push_str(&format!(
                                r#"<phoneme alphabet="ipa" ph="{}">{}</phoneme>"#,
                                hint.pronunciation, token
                            ));
                        }
                        PhoneticMode::XSampa => {
                            result.push_str(&format!(
                                r#"<phoneme alphabet="x-sampa" ph="{}">{}</phoneme>"#,
                                hint.pronunciation, token
                            ));
                        }
                    }
                } else {
                    result.push_str(&token);
                }
            } else {
                result.push(ch);
                chars.next();
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Number of phonetic hints loaded.
    #[must_use]
    pub fn phonetic_hint_count(&self) -> usize {
        self.phonetic_map.len()
    }

    /// Number of abbreviation entries loaded.
    #[must_use]
    pub fn abbreviation_count(&self) -> usize {
        self.abbrev_map.len()
    }
}

// ---------------------------------------------------------------------------
// Built-in defaults
// ---------------------------------------------------------------------------

/// A curated set of common English phonetic hints.
fn default_phonetic_hints() -> Vec<PhoneticHint> {
    vec![
        PhoneticHint::spelling("GIF", "jif"),
        PhoneticHint::spelling("JPEG", "jay-peg"),
        PhoneticHint::spelling("SQL", "sequel"),
        PhoneticHint::spelling("SCSI", "scuzzy"),
        PhoneticHint::spelling("cache", "cash"),
        PhoneticHint::spelling("segue", "seg-way"),
        PhoneticHint::spelling("quinoa", "keen-wah"),
        PhoneticHint::spelling("meme", "meem"),
        PhoneticHint::ipa("schedule", "ˈskɛdʒuːl"),
        PhoneticHint::ipa("hierarchy", "ˈhaɪərɑːki"),
    ]
}

/// A curated set of common English abbreviation expansions.
fn default_abbreviations() -> Vec<AbbrevEntry> {
    vec![
        AbbrevEntry::expand("dr.", "Doctor"),
        AbbrevEntry::expand("mr.", "Mister"),
        AbbrevEntry::expand("mrs.", "Missus"),
        AbbrevEntry::expand("ms.", "Miss"),
        AbbrevEntry::expand("prof.", "Professor"),
        AbbrevEntry::expand("st.", "Saint"),
        AbbrevEntry::expand("jr.", "Junior"),
        AbbrevEntry::expand("sr.", "Senior"),
        AbbrevEntry::expand("vs.", "versus"),
        AbbrevEntry::expand("etc.", "et cetera"),
        AbbrevEntry::expand("i.e.", "that is"),
        AbbrevEntry::expand("e.g.", "for example"),
        AbbrevEntry::expand("approx.", "approximately"),
        AbbrevEntry::expand("min.", "minute"),
        AbbrevEntry::expand("max.", "maximum"),
        AbbrevEntry::expand("km", "kilometre"),
        AbbrevEntry::expand("cm", "centimetre"),
        AbbrevEntry::expand("mm", "millimetre"),
        AbbrevEntry::expand("kg", "kilogram"),
        AbbrevEntry::expand("lb.", "pound"),
        AbbrevEntry::spell_out("BBC", "British Broadcasting Corporation"),
        AbbrevEntry::spell_out("FBI", "Federal Bureau of Investigation"),
        AbbrevEntry::spell_out("CIA", "Central Intelligence Agency"),
        AbbrevEntry::spell_out("CEO", "Chief Executive Officer"),
        AbbrevEntry::spell_out("HTML", "HyperText Markup Language"),
        AbbrevEntry::acronym("NASA", "National Aeronautics and Space Administration"),
        AbbrevEntry::acronym("NATO", "North Atlantic Treaty Organisation"),
        AbbrevEntry::acronym("UNICEF", "United Nations Children's Fund"),
        AbbrevEntry::acronym("OPEC", "Organisation of the Petroleum Exporting Countries"),
    ]
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Extract a number token (integer or decimal) starting at `start` in `chars`.
/// Returns `(raw_number_string, end_index)`.
fn extract_number(chars: &[char], start: usize) -> Option<(String, usize)> {
    if start >= chars.len() || !chars[start].is_ascii_digit() {
        return None;
    }
    let mut s = String::new();
    let mut i = start;
    let mut has_dot = false;

    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_digit() {
            s.push(c);
            i += 1;
        } else if c == ',' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
            // thousand separator — skip for verbalization, keep digits
            i += 1;
        } else if c == '.' && !has_dot && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
            s.push(c);
            has_dot = true;
            i += 1;
        } else {
            break;
        }
    }
    if s.is_empty() {
        None
    } else {
        Some((s, i))
    }
}

/// Produce a spoken representation of a number string.
/// For simplicity this converts the raw string: integers are spoken digit-by-
/// digit groups (thousands) when small, decimals split at the period.
fn verbalize_number_string(num: &str, opts: &NumberVerbalizationOptions) -> String {
    if let Some(dot_pos) = num.find('.') {
        let int_part = &num[..dot_pos];
        let frac_part = &num[dot_pos + 1..];
        let int_spoken = verbalize_integer_str(int_part, opts);
        let frac_spoken = frac_part
            .chars()
            .map(|c| digit_word(c).unwrap_or(c.to_string()))
            .collect::<Vec<_>>()
            .join(" ");
        format!("{} {} {}", int_spoken, opts.decimal_word, frac_spoken)
    } else {
        verbalize_integer_str(num, opts)
    }
}

/// Speak an integer string.  For values ≤ 9_999_999 we use named thousands;
/// for larger values we fall back to digit-by-digit.
fn verbalize_integer_str(s: &str, opts: &NumberVerbalizationOptions) -> String {
    let trimmed = s.trim_start_matches('0');
    if trimmed.is_empty() {
        return "zero".into();
    }
    let value: Option<u64> = trimmed.parse().ok();
    match value {
        Some(n) if opts.expand_thousands && n < 10_000_000 => number_to_words(n),
        _ => s
            .chars()
            .map(|c| digit_word(c).unwrap_or(c.to_string()))
            .collect::<Vec<_>>()
            .join(" "),
    }
}

/// Convert a small non-negative integer to English words.
fn number_to_words(n: u64) -> String {
    const ONES: &[&str] = &[
        "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen",
    ];
    const TENS: &[&str] = &[
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];

    if n == 0 {
        return "zero".into();
    }
    if n < 20 {
        return ONES[n as usize].into();
    }
    if n < 100 {
        let tens = TENS[(n / 10) as usize];
        let one = ONES[(n % 10) as usize];
        return if one.is_empty() {
            tens.into()
        } else {
            format!("{tens}-{one}")
        };
    }
    if n < 1_000 {
        let hundreds = ONES[(n / 100) as usize];
        let rest = n % 100;
        return if rest == 0 {
            format!("{hundreds} hundred")
        } else {
            format!("{hundreds} hundred {}", number_to_words(rest))
        };
    }
    if n < 1_000_000 {
        let thousands = number_to_words(n / 1_000);
        let rest = n % 1_000;
        return if rest == 0 {
            format!("{thousands} thousand")
        } else {
            format!("{thousands} thousand {}", number_to_words(rest))
        };
    }
    // Millions
    let millions = number_to_words(n / 1_000_000);
    let rest = n % 1_000_000;
    if rest == 0 {
        format!("{millions} million")
    } else {
        format!("{millions} million {}", number_to_words(rest))
    }
}

/// Produce a word for a single ASCII digit character.
fn digit_word(c: char) -> Option<String> {
    match c {
        '0' => Some("zero".into()),
        '1' => Some("one".into()),
        '2' => Some("two".into()),
        '3' => Some("three".into()),
        '4' => Some("four".into()),
        '5' => Some("five".into()),
        '6' => Some("six".into()),
        '7' => Some("seven".into()),
        '8' => Some("eight".into()),
        '9' => Some("nine".into()),
        _ => None,
    }
}

/// Produce a naive plural form: append "s" when the number string is not "1".
fn pluralise(word: &str, num_str: &str) -> String {
    if num_str == "1" || num_str.starts_with("1.") {
        word.to_string()
    } else {
        format!("{word}s")
    }
}

/// Spell out the letters of a word separated by spaces (e.g. "FBI" → "F B I").
fn spell_out_letters(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .map(|c| c.to_uppercase().to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> TtsHintEngine {
        TtsHintEngine::new(TtsHintConfig::default())
    }

    fn empty_engine() -> TtsHintEngine {
        TtsHintEngine::new(TtsHintConfig::empty())
    }

    // --- phonetic hints ---

    #[test]
    fn test_spelling_variant_substitution() {
        let engine = engine();
        let out = engine.process("Save the file as GIF format");
        assert!(out.contains("jif"), "expected 'jif', got: {out}");
    }

    #[test]
    fn test_ipa_phoneme_emitted() {
        let engine = engine();
        let out = engine.process("Please schedule a meeting");
        assert!(out.contains("<phoneme"), "expected SSML phoneme tag: {out}");
        assert!(out.contains("ipa"), "expected ipa alphabet: {out}");
    }

    #[test]
    fn test_custom_phonetic_hint() {
        let config = TtsHintConfig::empty()
            .with_phonetic(PhoneticHint::spelling("colonel", "kernel"));
        let engine = TtsHintEngine::new(config);
        let out = engine.process("The colonel ordered a retreat");
        assert!(out.contains("kernel"), "got: {out}");
    }

    // --- abbreviation expansion ---

    #[test]
    fn test_title_abbreviation_expanded() {
        let engine = engine();
        let out = engine.process("Dr. Smith will see you now");
        assert!(out.contains("Doctor"), "expected Doctor, got: {out}");
        assert!(!out.contains("Dr."), "Dr. should be gone: {out}");
    }

    #[test]
    fn test_etc_expanded() {
        let engine = engine();
        let out = engine.process("cats, dogs, etc. are common pets");
        assert!(out.contains("et cetera"), "got: {out}");
    }

    #[test]
    fn test_spell_out_acronym() {
        let engine = engine();
        let out = engine.process("The BBC reported today");
        // SpellOut style → "B B C"
        assert!(out.contains("B B C") || out.contains("B B"), "got: {out}");
    }

    #[test]
    fn test_custom_abbreviation() {
        let config = TtsHintConfig::empty()
            .with_abbreviation(AbbrevEntry::expand("OxiMedia", "Oxi Media framework"));
        let engine = TtsHintEngine::new(config);
        let out = engine.process("OxiMedia is great");
        assert!(out.contains("Oxi Media framework"), "got: {out}");
    }

    // --- number verbalization ---

    #[test]
    fn test_integer_verbalization() {
        let engine = empty_engine();
        let out = engine.process("There are 42 chapters");
        assert!(out.contains("forty-two"), "got: {out}");
    }

    #[test]
    fn test_decimal_verbalization() {
        let engine = empty_engine();
        let out = engine.process("The ratio is 3.14");
        assert!(out.contains("three"), "got: {out}");
        assert!(out.contains("point"), "got: {out}");
        assert!(out.contains("one") || out.contains("four"), "got: {out}");
    }

    #[test]
    fn test_percent_verbalization() {
        let engine = empty_engine();
        let out = engine.process("75% complete");
        assert!(out.contains("seventy-five percent"), "got: {out}");
    }

    #[test]
    fn test_currency_verbalization() {
        let engine = empty_engine();
        let out = engine.process("It costs $5");
        assert!(out.contains("five"), "got: {out}");
        assert!(out.contains("dollar"), "got: {out}");
    }

    #[test]
    fn test_zero_verbalized() {
        let engine = empty_engine();
        let out = engine.process("0 items");
        assert!(out.contains("zero"), "got: {out}");
    }

    #[test]
    fn test_number_to_words_small() {
        assert_eq!(number_to_words(0), "zero");
        assert_eq!(number_to_words(1), "one");
        assert_eq!(number_to_words(13), "thirteen");
        assert_eq!(number_to_words(42), "forty-two");
        assert_eq!(number_to_words(100), "one hundred");
        assert_eq!(number_to_words(1_000), "one thousand");
        assert_eq!(number_to_words(1_001), "one thousand one");
    }

    #[test]
    fn test_spell_out_letters_helper() {
        assert_eq!(spell_out_letters("FBI"), "F B I");
        assert_eq!(spell_out_letters("BBC"), "B B C");
    }

    #[test]
    fn test_engine_hint_counts() {
        let engine = engine();
        assert!(engine.phonetic_hint_count() > 0);
        assert!(engine.abbreviation_count() > 0);
    }

    #[test]
    fn test_plain_text_passthrough() {
        let engine = empty_engine();
        let text = "Hello world, this is plain text.";
        let out = engine.process(text);
        // No substitutions — should pass through unchanged (modulo whitespace)
        assert_eq!(out.trim(), text.trim());
    }

    #[test]
    fn test_negative_number() {
        let engine = empty_engine();
        let out = engine.process("-5 degrees");
        assert!(out.contains("minus"), "got: {out}");
        assert!(out.contains("five"), "got: {out}");
    }
}
