//! Fallback font chain for missing glyphs.
//!
//! Provides a `FontChain` that holds an ordered list of fonts and a `GlyphResolver`
//! that queries them sequentially until a glyph is found. Includes `ScriptRange`
//! detection for coverage reporting across Latin, CJK, Arabic, Devanagari, and Emoji.
//!
//! # Example
//!
//! ```
//! use oximedia_subtitle::fallback_fonts::{FontChain, FontEntry, ScriptRange};
//!
//! let mut chain = FontChain::new();
//! let font = FontEntry::new("Arial", vec![ScriptRange::Latin]);
//! chain.add(font);
//! assert_eq!(chain.len(), 1);
//! ```

use std::collections::HashSet;

/// Unicode script ranges for coverage detection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScriptRange {
    /// Basic Latin + Latin Extended (U+0000..U+024F).
    Latin,
    /// CJK Unified Ideographs and related blocks.
    Cjk,
    /// Arabic script block (U+0600..U+06FF).
    Arabic,
    /// Devanagari script block (U+0900..U+097F).
    Devanagari,
    /// Emoji and symbols (various blocks).
    Emoji,
    /// Cyrillic (U+0400..U+04FF).
    Cyrillic,
    /// Greek (U+0370..U+03FF).
    Greek,
    /// Hangul (Korean) (U+AC00..U+D7AF).
    Hangul,
    /// Thai (U+0E00..U+0E7F).
    Thai,
    /// Hebrew (U+0590..U+05FF).
    Hebrew,
}

impl ScriptRange {
    /// Detect the primary script range for a character.
    #[must_use]
    pub fn detect(ch: char) -> Option<Self> {
        let cp = ch as u32;
        match cp {
            0x0000..=0x024F | 0x1E00..=0x1EFF => Some(Self::Latin),
            0x0370..=0x03FF => Some(Self::Greek),
            0x0400..=0x04FF => Some(Self::Cyrillic),
            0x0590..=0x05FF => Some(Self::Hebrew),
            0x0600..=0x06FF | 0x0750..=0x077F | 0x08A0..=0x08FF => Some(Self::Arabic),
            0x0900..=0x097F => Some(Self::Devanagari),
            0x0E00..=0x0E7F => Some(Self::Thai),
            0x3000..=0x303F
            | 0x3040..=0x309F
            | 0x30A0..=0x30FF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2A6DF => Some(Self::Cjk),
            0xAC00..=0xD7AF => Some(Self::Hangul),
            0x1F300..=0x1F9FF | 0x2600..=0x26FF | 0x2700..=0x27BF | 0xFE00..=0xFE0F => {
                Some(Self::Emoji)
            }
            _ => None,
        }
    }

    /// Return a representative character for testing this script range.
    #[must_use]
    pub const fn sample_char(self) -> char {
        match self {
            Self::Latin => 'A',
            Self::Cjk => '\u{4E00}',
            Self::Arabic => '\u{0627}',
            Self::Devanagari => '\u{0905}',
            Self::Emoji => '\u{1F600}',
            Self::Cyrillic => '\u{0410}',
            Self::Greek => '\u{0391}',
            Self::Hangul => '\u{AC00}',
            Self::Thai => '\u{0E01}',
            Self::Hebrew => '\u{05D0}',
        }
    }
}

/// A font entry in the fallback chain.
#[derive(Clone, Debug)]
pub struct FontEntry {
    /// Font family name.
    pub name: String,
    /// Scripts this font covers.
    pub coverage: Vec<ScriptRange>,
    /// Optional glyph data (codepoints this font can render).
    /// If empty, coverage is used as an approximation.
    glyph_set: HashSet<u32>,
    /// Priority weight (lower = higher priority within same script).
    pub priority: i32,
}

impl FontEntry {
    /// Create a new font entry with name and script coverage.
    #[must_use]
    pub fn new(name: impl Into<String>, coverage: Vec<ScriptRange>) -> Self {
        Self {
            name: name.into(),
            coverage,
            glyph_set: HashSet::new(),
            priority: 0,
        }
    }

    /// Set priority (lower = higher priority).
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Add explicit glyph coverage by codepoints.
    pub fn add_glyphs(&mut self, codepoints: &[u32]) {
        for &cp in codepoints {
            self.glyph_set.insert(cp);
        }
    }

    /// Check if this font claims to support a given character.
    ///
    /// If explicit glyph data is loaded, checks the glyph set.
    /// Otherwise falls back to script-range coverage.
    #[must_use]
    pub fn has_glyph(&self, ch: char) -> bool {
        if !self.glyph_set.is_empty() {
            return self.glyph_set.contains(&(ch as u32));
        }
        // Fall back to script coverage
        if let Some(script) = ScriptRange::detect(ch) {
            self.coverage.contains(&script)
        } else {
            // Unknown script — only claim coverage if this is a "universal" font
            // (covers 3+ script ranges)
            self.coverage.len() >= 3
        }
    }
}

/// Coverage report for a font.
#[derive(Clone, Debug)]
pub struct FontCoverage {
    /// Font name.
    pub font_name: String,
    /// Scripts covered.
    pub scripts: Vec<ScriptRange>,
    /// Number of explicit glyphs registered.
    pub explicit_glyph_count: usize,
}

/// An ordered chain of fallback fonts.
#[derive(Clone, Debug, Default)]
pub struct FontChain {
    fonts: Vec<FontEntry>,
}

impl FontChain {
    /// Create an empty font chain.
    #[must_use]
    pub fn new() -> Self {
        Self { fonts: Vec::new() }
    }

    /// Add a font to the end of the chain.
    pub fn add(&mut self, font: FontEntry) {
        self.fonts.push(font);
    }

    /// Insert a font at a specific position.
    pub fn insert(&mut self, index: usize, font: FontEntry) {
        let idx = index.min(self.fonts.len());
        self.fonts.insert(idx, font);
    }

    /// Number of fonts in the chain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fonts.len()
    }

    /// Whether the chain is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fonts.is_empty()
    }

    /// Get an iterator over the fonts.
    pub fn iter(&self) -> impl Iterator<Item = &FontEntry> {
        self.fonts.iter()
    }

    /// Get coverage reports for all fonts.
    #[must_use]
    pub fn coverage_reports(&self) -> Vec<FontCoverage> {
        self.fonts
            .iter()
            .map(|f| FontCoverage {
                font_name: f.name.clone(),
                scripts: f.coverage.clone(),
                explicit_glyph_count: f.glyph_set.len(),
            })
            .collect()
    }

    /// Remove a font by name.
    pub fn remove(&mut self, name: &str) {
        self.fonts.retain(|f| f.name != name);
    }

    /// Get all scripts covered by any font in the chain.
    #[must_use]
    pub fn all_covered_scripts(&self) -> HashSet<ScriptRange> {
        let mut scripts = HashSet::new();
        for font in &self.fonts {
            for &s in &font.coverage {
                scripts.insert(s);
            }
        }
        scripts
    }

    /// Find scripts not covered by any font.
    #[must_use]
    pub fn uncovered_scripts(&self, required: &[ScriptRange]) -> Vec<ScriptRange> {
        let covered = self.all_covered_scripts();
        required
            .iter()
            .filter(|s| !covered.contains(s))
            .copied()
            .collect()
    }
}

/// Resolves glyphs by querying fonts in a chain sequentially.
pub struct GlyphResolver<'a> {
    chain: &'a FontChain,
}

/// Result of resolving a single glyph.
#[derive(Clone, Debug)]
pub struct GlyphResolveResult {
    /// Name of the font that provides this glyph.
    pub font_name: String,
    /// Index in the font chain.
    pub font_index: usize,
    /// The character resolved.
    pub character: char,
}

/// Result of resolving all glyphs in a string.
#[derive(Clone, Debug)]
pub struct TextResolveResult {
    /// Successfully resolved glyphs.
    pub resolved: Vec<GlyphResolveResult>,
    /// Characters that could not be resolved by any font.
    pub missing: Vec<char>,
    /// Number of font switches needed to render the text.
    pub font_switches: usize,
}

impl<'a> GlyphResolver<'a> {
    /// Create a new resolver for the given chain.
    #[must_use]
    pub fn new(chain: &'a FontChain) -> Self {
        Self { chain }
    }

    /// Resolve a single character, returning the first font that can render it.
    #[must_use]
    pub fn resolve(&self, ch: char) -> Option<GlyphResolveResult> {
        for (idx, font) in self.chain.fonts.iter().enumerate() {
            if font.has_glyph(ch) {
                return Some(GlyphResolveResult {
                    font_name: font.name.clone(),
                    font_index: idx,
                    character: ch,
                });
            }
        }
        None
    }

    /// Resolve all characters in a string.
    #[must_use]
    pub fn resolve_text(&self, text: &str) -> TextResolveResult {
        let mut resolved = Vec::new();
        let mut missing = Vec::new();
        let mut font_switches = 0;
        let mut last_font_index: Option<usize> = None;

        for ch in text.chars() {
            // Skip whitespace and control characters
            if ch.is_whitespace() || ch.is_control() {
                continue;
            }

            if let Some(result) = self.resolve(ch) {
                if let Some(last) = last_font_index {
                    if last != result.font_index {
                        font_switches += 1;
                    }
                }
                last_font_index = Some(result.font_index);
                resolved.push(result);
            } else {
                missing.push(ch);
            }
        }

        TextResolveResult {
            resolved,
            missing,
            font_switches,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_script_detection_latin() {
        assert_eq!(ScriptRange::detect('A'), Some(ScriptRange::Latin));
        assert_eq!(ScriptRange::detect('z'), Some(ScriptRange::Latin));
    }

    #[test]
    fn test_script_detection_cjk() {
        assert_eq!(ScriptRange::detect('\u{4E00}'), Some(ScriptRange::Cjk));
        assert_eq!(ScriptRange::detect('\u{3042}'), Some(ScriptRange::Cjk)); // Hiragana
    }

    #[test]
    fn test_script_detection_arabic() {
        assert_eq!(ScriptRange::detect('\u{0627}'), Some(ScriptRange::Arabic));
    }

    #[test]
    fn test_script_detection_devanagari() {
        assert_eq!(
            ScriptRange::detect('\u{0905}'),
            Some(ScriptRange::Devanagari)
        );
    }

    #[test]
    fn test_script_detection_emoji() {
        assert_eq!(ScriptRange::detect('\u{1F600}'), Some(ScriptRange::Emoji));
    }

    #[test]
    fn test_font_entry_has_glyph_by_coverage() {
        let font = FontEntry::new("TestFont", vec![ScriptRange::Latin, ScriptRange::Greek]);
        assert!(font.has_glyph('A'));
        assert!(font.has_glyph('\u{0391}')); // Greek Alpha
        assert!(!font.has_glyph('\u{4E00}')); // CJK
    }

    #[test]
    fn test_font_entry_has_glyph_explicit() {
        let mut font = FontEntry::new("ExplicitFont", vec![]);
        font.add_glyphs(&[0x41, 0x42, 0x43]); // A, B, C
        assert!(font.has_glyph('A'));
        assert!(font.has_glyph('B'));
        assert!(!font.has_glyph('D'));
    }

    #[test]
    fn test_font_chain_basic() {
        let mut chain = FontChain::new();
        assert!(chain.is_empty());

        chain.add(FontEntry::new("Arial", vec![ScriptRange::Latin]));
        chain.add(FontEntry::new("NotoSansCJK", vec![ScriptRange::Cjk]));
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn test_glyph_resolver_fallback() {
        let mut chain = FontChain::new();
        chain.add(FontEntry::new("Arial", vec![ScriptRange::Latin]));
        chain.add(FontEntry::new("NotoSansCJK", vec![ScriptRange::Cjk]));

        let resolver = GlyphResolver::new(&chain);

        let latin_result = resolver.resolve('A');
        assert!(latin_result.is_some());
        let latin = latin_result.expect("test");
        assert_eq!(latin.font_name, "Arial");
        assert_eq!(latin.font_index, 0);

        let cjk_result = resolver.resolve('\u{4E00}');
        assert!(cjk_result.is_some());
        let cjk = cjk_result.expect("test");
        assert_eq!(cjk.font_name, "NotoSansCJK");
        assert_eq!(cjk.font_index, 1);
    }

    #[test]
    fn test_glyph_resolver_missing() {
        let mut chain = FontChain::new();
        chain.add(FontEntry::new("LatinOnly", vec![ScriptRange::Latin]));

        let resolver = GlyphResolver::new(&chain);
        let result = resolver.resolve('\u{0905}'); // Devanagari
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_text_mixed() {
        let mut chain = FontChain::new();
        chain.add(FontEntry::new("Arial", vec![ScriptRange::Latin]));
        chain.add(FontEntry::new("NotoSansCJK", vec![ScriptRange::Cjk]));

        let resolver = GlyphResolver::new(&chain);
        // Mix of Latin and CJK
        let result = resolver.resolve_text("Hello\u{4E16}\u{754C}");
        assert!(result.missing.is_empty());
        assert!(result.font_switches >= 1);
    }

    #[test]
    fn test_coverage_reports() {
        let mut chain = FontChain::new();
        chain.add(FontEntry::new("Font1", vec![ScriptRange::Latin]));
        let mut font2 = FontEntry::new("Font2", vec![ScriptRange::Cjk]);
        font2.add_glyphs(&[0x4E00, 0x4E01]);
        chain.add(font2);

        let reports = chain.coverage_reports();
        assert_eq!(reports.len(), 2);
        assert_eq!(reports[0].font_name, "Font1");
        assert_eq!(reports[0].explicit_glyph_count, 0);
        assert_eq!(reports[1].font_name, "Font2");
        assert_eq!(reports[1].explicit_glyph_count, 2);
    }

    #[test]
    fn test_uncovered_scripts() {
        let mut chain = FontChain::new();
        chain.add(FontEntry::new("Latin", vec![ScriptRange::Latin]));

        let required = vec![
            ScriptRange::Latin,
            ScriptRange::Cjk,
            ScriptRange::Arabic,
        ];
        let uncovered = chain.uncovered_scripts(&required);
        assert_eq!(uncovered.len(), 2);
        assert!(uncovered.contains(&ScriptRange::Cjk));
        assert!(uncovered.contains(&ScriptRange::Arabic));
    }

    #[test]
    fn test_font_chain_remove() {
        let mut chain = FontChain::new();
        chain.add(FontEntry::new("A", vec![ScriptRange::Latin]));
        chain.add(FontEntry::new("B", vec![ScriptRange::Cjk]));
        assert_eq!(chain.len(), 2);

        chain.remove("A");
        assert_eq!(chain.len(), 1);
        assert_eq!(
            chain.iter().next().map(|f| f.name.as_str()),
            Some("B")
        );
    }

    #[test]
    fn test_font_chain_insert() {
        let mut chain = FontChain::new();
        chain.add(FontEntry::new("First", vec![ScriptRange::Latin]));
        chain.add(FontEntry::new("Last", vec![ScriptRange::Cjk]));
        chain.insert(1, FontEntry::new("Middle", vec![ScriptRange::Arabic]));

        let names: Vec<&str> = chain.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["First", "Middle", "Last"]);
    }
}
