//! Natural-language template engine for audio description scene text.
//!
//! Provides a lightweight, zero-dependency template system that generates
//! human-readable scene descriptions ("A person walks briskly through a
//! sunlit corridor.") from structured scene data.  Multiple style
//! variations (concise, descriptive, cinematic) are supported.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Style variations
// ---------------------------------------------------------------------------

/// Writing style for generated scene descriptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptionStyle {
    /// Short, factual sentences optimised for tight dialogue gaps.
    Concise,
    /// Full prose with rich adjectives and spatial context.
    Descriptive,
    /// Dramatic, cinematically-flavoured language.
    Cinematic,
    /// Simple, accessible language for broad audiences.
    Plain,
}

impl DescriptionStyle {
    /// Return the adverb qualifier used by this style.
    #[must_use]
    pub const fn qualifier(&self) -> &'static str {
        match self {
            Self::Concise => "",
            Self::Descriptive => "slowly and deliberately",
            Self::Cinematic => "with purpose",
            Self::Plain => "carefully",
        }
    }
}

// ---------------------------------------------------------------------------
// Subject / Action / Setting descriptors
// ---------------------------------------------------------------------------

/// Describes the subject of the scene (who/what).
#[derive(Debug, Clone)]
pub struct SubjectDescriptor {
    /// Base noun phrase, e.g. "a person", "two children".
    pub noun_phrase: String,
    /// Optional appearance detail, e.g. "in a red coat".
    pub appearance: Option<String>,
}

impl SubjectDescriptor {
    /// Create a subject with just a noun phrase.
    #[must_use]
    pub fn new(noun_phrase: impl Into<String>) -> Self {
        Self {
            noun_phrase: noun_phrase.into(),
            appearance: None,
        }
    }

    /// Set an appearance detail.
    #[must_use]
    pub fn with_appearance(mut self, appearance: impl Into<String>) -> Self {
        self.appearance = Some(appearance.into());
        self
    }

    /// Render the subject as a string.
    #[must_use]
    pub fn render(&self) -> String {
        match &self.appearance {
            Some(a) => format!("{} {}", self.noun_phrase, a),
            None => self.noun_phrase.clone(),
        }
    }
}

/// Describes the primary action in the scene.
#[derive(Debug, Clone)]
pub struct ActionDescriptor {
    /// Verb phrase, e.g. "walks", "runs", "stands still".
    pub verb_phrase: String,
    /// Optional manner adverb, e.g. "briskly", "slowly".
    pub manner: Option<String>,
}

impl ActionDescriptor {
    /// Create an action descriptor.
    #[must_use]
    pub fn new(verb_phrase: impl Into<String>) -> Self {
        Self {
            verb_phrase: verb_phrase.into(),
            manner: None,
        }
    }

    /// Set a manner adverb.
    #[must_use]
    pub fn with_manner(mut self, manner: impl Into<String>) -> Self {
        self.manner = Some(manner.into());
        self
    }

    /// Render as a string.
    #[must_use]
    pub fn render(&self) -> String {
        match &self.manner {
            Some(m) if !m.is_empty() => format!("{} {}", self.verb_phrase, m),
            _ => self.verb_phrase.clone(),
        }
    }
}

/// Describes the setting/environment.
#[derive(Debug, Clone)]
pub struct SettingDescriptor {
    /// Preposition phrase, e.g. "through a corridor", "in a park".
    pub location_phrase: String,
    /// Optional lighting description, e.g. "bathed in golden light".
    pub lighting: Option<String>,
}

impl SettingDescriptor {
    /// Create a setting descriptor.
    #[must_use]
    pub fn new(location_phrase: impl Into<String>) -> Self {
        Self {
            location_phrase: location_phrase.into(),
            lighting: None,
        }
    }

    /// Set a lighting description.
    #[must_use]
    pub fn with_lighting(mut self, lighting: impl Into<String>) -> Self {
        self.lighting = Some(lighting.into());
        self
    }

    /// Render as a string.
    #[must_use]
    pub fn render(&self) -> String {
        match &self.lighting {
            Some(l) => format!("{}, {}", self.location_phrase, l),
            None => self.location_phrase.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Scene context
// ---------------------------------------------------------------------------

/// Structured scene context fed into the template engine.
#[derive(Debug, Clone)]
pub struct SceneContext {
    /// Primary subject of the scene.
    pub subject: SubjectDescriptor,
    /// Primary action.
    pub action: ActionDescriptor,
    /// Setting/environment.
    pub setting: SettingDescriptor,
    /// Optional emotional tone, e.g. "tense", "joyful".
    pub tone: Option<String>,
    /// Extra key–value metadata for advanced templates.
    pub extra: HashMap<String, String>,
}

impl SceneContext {
    /// Create a minimal scene context.
    #[must_use]
    pub fn new(
        subject: SubjectDescriptor,
        action: ActionDescriptor,
        setting: SettingDescriptor,
    ) -> Self {
        Self {
            subject,
            action,
            setting,
            tone: None,
            extra: HashMap::new(),
        }
    }

    /// Set an emotional tone.
    #[must_use]
    pub fn with_tone(mut self, tone: impl Into<String>) -> Self {
        self.tone = Some(tone.into());
        self
    }

    /// Insert extra metadata.
    #[must_use]
    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Template engine
// ---------------------------------------------------------------------------

/// Natural-language template engine for scene descriptions.
///
/// # Example
///
/// ```
/// use oximedia_access::audio_desc::template::{
///     SceneTemplateEngine, DescriptionStyle,
///     SubjectDescriptor, ActionDescriptor, SettingDescriptor, SceneContext,
/// };
///
/// let engine = SceneTemplateEngine::new();
/// let ctx = SceneContext::new(
///     SubjectDescriptor::new("a person"),
///     ActionDescriptor::new("walks"),
///     SettingDescriptor::new("through a sunlit corridor"),
/// );
/// let text = engine.generate(&ctx, DescriptionStyle::Descriptive);
/// assert!(text.contains("person"));
/// ```
pub struct SceneTemplateEngine {
    /// Custom template overrides (style → template string).
    custom_templates: HashMap<String, String>,
}

impl SceneTemplateEngine {
    /// Create a new template engine.
    #[must_use]
    pub fn new() -> Self {
        Self {
            custom_templates: HashMap::new(),
        }
    }

    /// Register a custom template for a style.
    ///
    /// Template placeholders:
    /// - `{subject}` – rendered subject descriptor
    /// - `{action}` – rendered action descriptor
    /// - `{setting}` – rendered setting descriptor
    /// - `{tone}` – emotional tone (empty string if unset)
    /// - `{qualifier}` – style qualifier adverb
    pub fn register_template(&mut self, style_key: impl Into<String>, template: impl Into<String>) {
        self.custom_templates
            .insert(style_key.into(), template.into());
    }

    /// Generate a description string for the given context and style.
    #[must_use]
    pub fn generate(&self, ctx: &SceneContext, style: DescriptionStyle) -> String {
        let subject = ctx.subject.render();
        let action = ctx.action.render();
        let setting = ctx.setting.render();
        let qualifier = style.qualifier();
        let tone = ctx.tone.as_deref().unwrap_or("");

        // Check for custom template override
        let style_key = format!("{style:?}");
        if let Some(tpl) = self.custom_templates.get(&style_key) {
            return self.render_template(tpl, &subject, &action, &setting, tone, qualifier);
        }

        match style {
            DescriptionStyle::Concise => Self::generate_concise(&subject, &action, &setting),
            DescriptionStyle::Descriptive => {
                Self::generate_descriptive(&subject, &action, &setting, tone)
            }
            DescriptionStyle::Cinematic => {
                Self::generate_cinematic(&subject, &action, &setting, tone)
            }
            DescriptionStyle::Plain => Self::generate_plain(&subject, &action, &setting),
        }
    }

    /// Generate multiple style variations for comparison.
    #[must_use]
    pub fn generate_all_styles(&self, ctx: &SceneContext) -> Vec<(DescriptionStyle, String)> {
        let styles = [
            DescriptionStyle::Concise,
            DescriptionStyle::Descriptive,
            DescriptionStyle::Cinematic,
            DescriptionStyle::Plain,
        ];
        styles.iter().map(|&s| (s, self.generate(ctx, s))).collect()
    }

    /// Select the most suitable style for a given available duration in ms.
    ///
    /// Shorter gaps prefer concise style; longer gaps allow descriptive style.
    #[must_use]
    pub fn select_style_for_duration(available_ms: i64) -> DescriptionStyle {
        if available_ms < 1500 {
            DescriptionStyle::Concise
        } else if available_ms < 3000 {
            DescriptionStyle::Plain
        } else if available_ms < 5000 {
            DescriptionStyle::Descriptive
        } else {
            DescriptionStyle::Cinematic
        }
    }

    // ---- private helpers -----------------------------------------------

    fn generate_concise(subject: &str, action: &str, setting: &str) -> String {
        let s = format!("{subject} {action} {setting}.");
        Self::capitalise_first(&s)
    }

    fn generate_descriptive(subject: &str, action: &str, setting: &str, tone: &str) -> String {
        let tone_clause = if tone.is_empty() {
            String::new()
        } else {
            format!(" The atmosphere feels {tone}.")
        };
        let s = format!(
            "{subject} moves {setting}, as {subject} {action} with unhurried grace.{tone_clause}"
        );
        Self::capitalise_first(&s)
    }

    fn generate_cinematic(subject: &str, action: &str, setting: &str, tone: &str) -> String {
        let tone_clause = if tone.is_empty() {
            String::from("The moment hangs in the air.")
        } else {
            format!("A {tone} tension fills the scene.")
        };
        let s = format!("With quiet determination, {subject} {action} {setting}. {tone_clause}");
        Self::capitalise_first(&s)
    }

    fn generate_plain(subject: &str, action: &str, setting: &str) -> String {
        let s = format!("{subject} is {action} {setting}.");
        Self::capitalise_first(&s)
    }

    fn render_template(
        &self,
        template: &str,
        subject: &str,
        action: &str,
        setting: &str,
        tone: &str,
        qualifier: &str,
    ) -> String {
        template
            .replace("{subject}", subject)
            .replace("{action}", action)
            .replace("{setting}", setting)
            .replace("{tone}", tone)
            .replace("{qualifier}", qualifier)
    }

    fn capitalise_first(s: &str) -> String {
        let mut chars = s.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => {
                let upper: String = first.to_uppercase().collect();
                upper + chars.as_str()
            }
        }
    }
}

impl Default for SceneTemplateEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx() -> SceneContext {
        SceneContext::new(
            SubjectDescriptor::new("a person"),
            ActionDescriptor::new("walks"),
            SettingDescriptor::new("through a sunlit corridor"),
        )
    }

    #[test]
    fn test_concise_contains_key_parts() {
        let engine = SceneTemplateEngine::new();
        let text = engine.generate(&make_ctx(), DescriptionStyle::Concise);
        assert!(text.contains("person"), "Expected 'person' in: {text}");
        assert!(text.contains("walks"), "Expected 'walks' in: {text}");
        assert!(text.contains("corridor"), "Expected 'corridor' in: {text}");
        assert!(text.ends_with('.'), "Expected trailing period in: {text}");
    }

    #[test]
    fn test_descriptive_style() {
        let engine = SceneTemplateEngine::new();
        let text = engine.generate(&make_ctx(), DescriptionStyle::Descriptive);
        assert!(text.contains("person"), "missing 'person': {text}");
        assert!(text.contains("corridor"), "missing 'corridor': {text}");
    }

    #[test]
    fn test_cinematic_style() {
        let engine = SceneTemplateEngine::new();
        let text = engine.generate(&make_ctx(), DescriptionStyle::Cinematic);
        assert!(text.contains("person"), "missing 'person': {text}");
        assert!(!text.is_empty());
    }

    #[test]
    fn test_plain_style() {
        let engine = SceneTemplateEngine::new();
        let text = engine.generate(&make_ctx(), DescriptionStyle::Plain);
        assert!(text.contains("person"), "missing 'person': {text}");
    }

    #[test]
    fn test_tone_included_in_descriptive() {
        let engine = SceneTemplateEngine::new();
        let ctx = make_ctx().with_tone("tense");
        let text = engine.generate(&ctx, DescriptionStyle::Descriptive);
        assert!(text.contains("tense"), "Expected tone in: {text}");
    }

    #[test]
    fn test_tone_included_in_cinematic() {
        let engine = SceneTemplateEngine::new();
        let ctx = make_ctx().with_tone("melancholic");
        let text = engine.generate(&ctx, DescriptionStyle::Cinematic);
        assert!(text.contains("melancholic"), "Expected tone in: {text}");
    }

    #[test]
    fn test_subject_with_appearance() {
        let engine = SceneTemplateEngine::new();
        let ctx = SceneContext::new(
            SubjectDescriptor::new("a woman").with_appearance("in a red coat"),
            ActionDescriptor::new("runs"),
            SettingDescriptor::new("across an empty square"),
        );
        let text = engine.generate(&ctx, DescriptionStyle::Concise);
        assert!(text.contains("red coat"), "Expected 'red coat' in: {text}");
    }

    #[test]
    fn test_action_with_manner() {
        let engine = SceneTemplateEngine::new();
        let ctx = SceneContext::new(
            SubjectDescriptor::new("a child"),
            ActionDescriptor::new("skips").with_manner("joyfully"),
            SettingDescriptor::new("in the garden"),
        );
        let text = engine.generate(&ctx, DescriptionStyle::Concise);
        assert!(text.contains("joyfully"), "Expected 'joyfully' in: {text}");
    }

    #[test]
    fn test_setting_with_lighting() {
        let engine = SceneTemplateEngine::new();
        let ctx = SceneContext::new(
            SubjectDescriptor::new("two figures"),
            ActionDescriptor::new("stand"),
            SettingDescriptor::new("on a rooftop").with_lighting("bathed in neon light"),
        );
        let text = engine.generate(&ctx, DescriptionStyle::Concise);
        assert!(
            text.contains("neon light"),
            "Expected 'neon light' in: {text}"
        );
    }

    #[test]
    fn test_all_styles_non_empty() {
        let engine = SceneTemplateEngine::new();
        let variations = engine.generate_all_styles(&make_ctx());
        assert_eq!(variations.len(), 4);
        for (style, text) in &variations {
            assert!(!text.is_empty(), "Style {style:?} produced empty text");
        }
    }

    #[test]
    fn test_first_letter_capitalised() {
        // Concise style should always start with an uppercase letter
        let engine = SceneTemplateEngine::new();
        let ctx = SceneContext::new(
            SubjectDescriptor::new("a dog"),
            ActionDescriptor::new("sits"),
            SettingDescriptor::new("by the fire"),
        );
        let text = engine.generate(&ctx, DescriptionStyle::Concise);
        let first_char = text.chars().next().expect("first_char should be valid");
        assert!(
            first_char.is_uppercase(),
            "Expected uppercase start: {text}"
        );
    }

    #[test]
    fn test_select_style_for_duration() {
        assert_eq!(
            SceneTemplateEngine::select_style_for_duration(800),
            DescriptionStyle::Concise
        );
        assert_eq!(
            SceneTemplateEngine::select_style_for_duration(2000),
            DescriptionStyle::Plain
        );
        assert_eq!(
            SceneTemplateEngine::select_style_for_duration(4000),
            DescriptionStyle::Descriptive
        );
        assert_eq!(
            SceneTemplateEngine::select_style_for_duration(6000),
            DescriptionStyle::Cinematic
        );
    }

    #[test]
    fn test_custom_template() {
        let mut engine = SceneTemplateEngine::new();
        engine.register_template("Concise", "Quick view: {subject} {action} {setting}.");
        let text = engine.generate(&make_ctx(), DescriptionStyle::Concise);
        assert!(
            text.starts_with("Quick view:"),
            "Expected custom template: {text}"
        );
    }

    #[test]
    fn test_extra_metadata_accessible() {
        let ctx = make_ctx().with_extra("camera_angle", "low");
        assert_eq!(
            ctx.extra.get("camera_angle").map(String::as_str),
            Some("low")
        );
    }
}
