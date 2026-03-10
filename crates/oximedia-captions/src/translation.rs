//! Translation support for captions

use crate::error::{CaptionError, Result};
use crate::types::{Caption, CaptionTrack, Language};
use std::collections::HashMap;

/// Translation project managing multiple language tracks
pub struct TranslationProject {
    /// Source language track
    pub source: CaptionTrack,
    /// Translated tracks (language code -> track)
    pub translations: HashMap<String, CaptionTrack>,
    /// Translation memory (source text -> translations by language)
    pub memory: HashMap<String, HashMap<String, String>>,
}

impl TranslationProject {
    /// Create a new translation project
    #[must_use]
    pub fn new(source: CaptionTrack) -> Self {
        Self {
            source,
            translations: HashMap::new(),
            memory: HashMap::new(),
        }
    }

    /// Add a translation track
    pub fn add_translation(&mut self, track: CaptionTrack) {
        self.translations.insert(track.language.code.clone(), track);
    }

    /// Get a translation track
    #[must_use]
    pub fn get_translation(&self, language_code: &str) -> Option<&CaptionTrack> {
        self.translations.get(language_code)
    }

    /// Get mutable translation track
    pub fn get_translation_mut(&mut self, language_code: &str) -> Option<&mut CaptionTrack> {
        self.translations.get_mut(language_code)
    }

    /// Update translation memory from existing translations
    pub fn update_memory(&mut self) {
        for (lang_code, track) in &self.translations {
            for (source_cap, trans_cap) in self.source.captions.iter().zip(&track.captions) {
                self.memory
                    .entry(source_cap.text.clone())
                    .or_default()
                    .insert(lang_code.clone(), trans_cap.text.clone());
            }
        }
    }

    /// Get translation suggestion from memory
    #[must_use]
    pub fn get_suggestion(&self, source_text: &str, target_language: &str) -> Option<&String> {
        self.memory
            .get(source_text)
            .and_then(|trans| trans.get(target_language))
    }

    /// Synchronize timing across all translations
    pub fn sync_timing(&mut self) -> Result<()> {
        for track in self.translations.values_mut() {
            if track.captions.len() != self.source.captions.len() {
                return Err(CaptionError::Translation(
                    "Caption count mismatch between source and translation".to_string(),
                ));
            }

            for (source_cap, trans_cap) in self.source.captions.iter().zip(&mut track.captions) {
                trans_cap.start = source_cap.start;
                trans_cap.end = source_cap.end;
            }
        }

        Ok(())
    }

    /// Detect language of a text
    #[must_use]
    pub fn detect_language(text: &str) -> Option<Language> {
        if let Some(info) = whatlang::detect(text) {
            let code = match info.lang() {
                whatlang::Lang::Eng => "en",
                whatlang::Lang::Spa => "es",
                whatlang::Lang::Fra => "fr",
                whatlang::Lang::Deu => "de",
                whatlang::Lang::Jpn => "ja",
                whatlang::Lang::Ara => "ar",
                whatlang::Lang::Rus => "ru",
                whatlang::Lang::Por => "pt",
                whatlang::Lang::Ita => "it",
                whatlang::Lang::Nld => "nl",
                _ => return None,
            };

            let name = format!("{:?}", info.lang());
            let rtl = matches!(info.lang(), whatlang::Lang::Ara | whatlang::Lang::Heb);

            Some(Language::new(code.to_string(), name, rtl))
        } else {
            None
        }
    }
}

/// Export track for translation
pub fn export_for_translation(track: &CaptionTrack, format: TranslationFormat) -> Result<String> {
    match format {
        TranslationFormat::Csv => export_csv(track),
        TranslationFormat::Xliff => export_xliff(track),
        TranslationFormat::Po => export_po(track),
    }
}

/// Translation export format
#[derive(Debug, Clone, Copy)]
pub enum TranslationFormat {
    /// CSV format (simple)
    Csv,
    /// XLIFF format (industry standard)
    Xliff,
    /// Gettext PO format
    Po,
}

fn export_csv(track: &CaptionTrack) -> Result<String> {
    let mut output = String::new();
    output.push_str("ID,Start,End,Text\n");

    for caption in &track.captions {
        let start = caption.start.to_string();
        let end = caption.end.to_string();
        let text = caption.text.replace('"', "\"\"");
        output.push_str(&format!(
            "{},\"{}\",\"{}\",\"{}\"\n",
            caption.id, start, end, text
        ));
    }

    Ok(output)
}

fn export_xliff(track: &CaptionTrack) -> Result<String> {
    let mut output = String::new();

    output.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    output.push_str("<xliff version=\"1.2\" xmlns=\"urn:oasis:names:tc:xliff:document:1.2\">\n");
    output.push_str(&format!(
        "  <file source-language=\"{}\" datatype=\"plaintext\">\n",
        track.language.code
    ));
    output.push_str("    <body>\n");

    for caption in &track.captions {
        output.push_str(&format!("      <trans-unit id=\"{}\">\n", caption.id));
        output.push_str(&format!(
            "        <source>{}</source>\n",
            escape_xml(&caption.text)
        ));
        output.push_str("        <target></target>\n");
        output.push_str("      </trans-unit>\n");
    }

    output.push_str("    </body>\n");
    output.push_str("  </file>\n");
    output.push_str("</xliff>\n");

    Ok(output)
}

fn export_po(track: &CaptionTrack) -> Result<String> {
    let mut output = String::new();

    output.push_str("# Caption translation file\n");
    output.push_str(&format!(
        "msgid \"\"\nmsgstr \"\"\n\"Language: {}\\n\"\n\n",
        track.language.code
    ));

    for caption in &track.captions {
        output.push_str(&format!("#: {}\n", caption.id));
        output.push_str(&format!("msgid \"{}\"\n", escape_po(&caption.text)));
        output.push_str("msgstr \"\"\n\n");
    }

    Ok(output)
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn escape_po(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// Import translated captions
pub fn import_translation(
    source: &CaptionTrack,
    translated_data: &str,
    format: TranslationFormat,
    target_language: Language,
) -> Result<CaptionTrack> {
    match format {
        TranslationFormat::Csv => import_csv(source, translated_data, target_language),
        TranslationFormat::Xliff => import_xliff(source, translated_data, target_language),
        TranslationFormat::Po => import_po(source, translated_data, target_language),
    }
}

fn import_csv(
    source: &CaptionTrack,
    data: &str,
    target_language: Language,
) -> Result<CaptionTrack> {
    let mut track = CaptionTrack::new(target_language);
    let lines: Vec<&str> = data.lines().skip(1).collect(); // Skip header

    for line in lines {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 4 {
            // Parse ID, find matching source caption, create translated caption
            let text = parts[3].trim_matches('"').replace("\"\"", "\"");
            // Simplified - should match by ID
            if let Some(source_cap) = source.captions.first() {
                let caption = Caption::new(source_cap.start, source_cap.end, text);
                track.add_caption(caption)?;
            }
        }
    }

    Ok(track)
}

fn import_xliff(
    _source: &CaptionTrack,
    _data: &str,
    target_language: Language,
) -> Result<CaptionTrack> {
    // Simplified implementation
    Ok(CaptionTrack::new(target_language))
}

fn import_po(
    _source: &CaptionTrack,
    _data: &str,
    target_language: Language,
) -> Result<CaptionTrack> {
    // Simplified implementation
    Ok(CaptionTrack::new(target_language))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Timestamp;

    #[test]
    fn test_translation_project() {
        let mut source = CaptionTrack::new(Language::english());
        source
            .add_caption(Caption::new(
                Timestamp::from_secs(1),
                Timestamp::from_secs(3),
                "Hello".to_string(),
            ))
            .expect("operation should succeed in test");

        let mut project = TranslationProject::new(source);

        let mut spanish = CaptionTrack::new(Language::spanish());
        spanish
            .add_caption(Caption::new(
                Timestamp::from_secs(1),
                Timestamp::from_secs(3),
                "Hola".to_string(),
            ))
            .expect("operation should succeed in test");

        project.add_translation(spanish);
        assert!(project.get_translation("es").is_some());
    }

    #[test]
    fn test_export_csv() {
        let mut track = CaptionTrack::new(Language::english());
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(1),
                Timestamp::from_secs(3),
                "Test".to_string(),
            ))
            .expect("operation should succeed in test");

        let csv =
            export_for_translation(&track, TranslationFormat::Csv).expect("export should succeed");
        assert!(csv.contains("ID,Start,End,Text"));
        assert!(csv.contains("Test"));
    }

    #[test]
    fn test_language_detection() {
        let english = "This is an English sentence.";
        let lang = TranslationProject::detect_language(english);
        assert!(lang.is_some());
        assert_eq!(lang.expect("language detection should succeed").code, "en");
    }
}
