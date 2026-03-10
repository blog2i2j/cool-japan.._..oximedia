#![allow(dead_code)]
//! AAF selector and alternation segment support.
//!
//! Implements the AAF Selector component, which represents a set of alternative
//! segments where exactly one is the "selected" (active) segment at any time.
//! This is used in professional editing for maintaining alternate takes,
//! multi-language audio, and conditional content selection.
//!
//! # AAF Specification
//!
//! A Selector contains one or more Components as alternates. The first
//! component is considered the "selected" (active) segment. The remaining
//! components are stored as alternates that can be swapped in during editing.

use std::collections::HashMap;
use uuid::Uuid;

/// Selection strategy for choosing among alternates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SelectionStrategy {
    /// Manual selection by editor/user.
    Manual,
    /// Select based on language/locale preference.
    LanguageBased,
    /// Select based on content rating.
    RatingBased,
    /// Select based on target platform/delivery.
    PlatformBased,
    /// Select based on quality/resolution tier.
    QualityBased,
    /// Automatic selection via script/automation.
    Scripted,
}

impl SelectionStrategy {
    /// Returns a human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Manual => "Manual",
            Self::LanguageBased => "Language-Based",
            Self::RatingBased => "Rating-Based",
            Self::PlatformBased => "Platform-Based",
            Self::QualityBased => "Quality-Based",
            Self::Scripted => "Scripted",
        }
    }

    /// Whether the strategy can be evaluated automatically.
    #[must_use]
    pub const fn is_automatic(self) -> bool {
        !matches!(self, Self::Manual)
    }
}

/// An individual alternate segment within a selector.
#[derive(Debug, Clone)]
pub struct Alternate {
    /// Unique identifier for this alternate.
    pub alternate_id: Uuid,
    /// Human-readable label (e.g., "Take 3", "English", "4K Version").
    pub label: String,
    /// Duration in edit units.
    pub duration: i64,
    /// Start offset in edit units.
    pub start_offset: i64,
    /// Associated mob ID (source reference).
    pub mob_id: Option<Uuid>,
    /// Track ID within the mob.
    pub track_id: Option<u32>,
    /// Language code (ISO 639-1) if applicable.
    pub language: Option<String>,
    /// Priority for automatic selection (lower = higher priority).
    pub priority: u32,
    /// User-assigned metadata tags.
    pub tags: Vec<String>,
    /// Whether this alternate is marked as approved.
    pub approved: bool,
}

impl Alternate {
    /// Create a new alternate.
    #[must_use]
    pub fn new(label: impl Into<String>, duration: i64) -> Self {
        Self {
            alternate_id: Uuid::new_v4(),
            label: label.into(),
            duration,
            start_offset: 0,
            mob_id: None,
            track_id: None,
            language: None,
            priority: 100,
            tags: Vec::new(),
            approved: false,
        }
    }

    /// Set language code.
    #[must_use]
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Set priority.
    #[must_use]
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set source reference.
    #[must_use]
    pub fn with_source(mut self, mob_id: Uuid, track_id: u32) -> Self {
        self.mob_id = Some(mob_id);
        self.track_id = Some(track_id);
        self
    }

    /// Add a tag.
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        self.tags.push(tag.into());
    }

    /// Check if this alternate has a given tag.
    #[must_use]
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

/// A Selector component holding multiple alternates with one selected.
#[derive(Debug, Clone)]
pub struct Selector {
    /// Unique identifier for this selector.
    pub selector_id: Uuid,
    /// Name of this selector.
    pub name: String,
    /// Index of the currently selected alternate.
    selected_index: usize,
    /// All alternates (the first is the default selected).
    alternates: Vec<Alternate>,
    /// Selection strategy.
    pub strategy: SelectionStrategy,
    /// Whether alternates should all have the same duration.
    pub enforce_equal_duration: bool,
    /// User-defined comment/notes.
    pub comment: Option<String>,
}

impl Selector {
    /// Create a new empty selector.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            selector_id: Uuid::new_v4(),
            name: name.into(),
            selected_index: 0,
            alternates: Vec::new(),
            strategy: SelectionStrategy::Manual,
            enforce_equal_duration: false,
            comment: None,
        }
    }

    /// Create a selector with a strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: SelectionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Add an alternate. The first alternate added becomes the selected one.
    pub fn add_alternate(&mut self, alternate: Alternate) {
        self.alternates.push(alternate);
    }

    /// Get the number of alternates.
    #[must_use]
    pub fn alternate_count(&self) -> usize {
        self.alternates.len()
    }

    /// Get all alternates.
    #[must_use]
    pub fn alternates(&self) -> &[Alternate] {
        &self.alternates
    }

    /// Get the currently selected alternate.
    #[must_use]
    pub fn selected(&self) -> Option<&Alternate> {
        self.alternates.get(self.selected_index)
    }

    /// Get the current selection index.
    #[must_use]
    pub fn selected_index(&self) -> usize {
        self.selected_index
    }

    /// Set the selected alternate by index. Returns false if index is out of range.
    pub fn select(&mut self, index: usize) -> bool {
        if index < self.alternates.len() {
            self.selected_index = index;
            true
        } else {
            false
        }
    }

    /// Select an alternate by its UUID. Returns false if not found.
    pub fn select_by_id(&mut self, id: &Uuid) -> bool {
        if let Some(idx) = self.alternates.iter().position(|a| a.alternate_id == *id) {
            self.selected_index = idx;
            true
        } else {
            false
        }
    }

    /// Select the alternate with the best (lowest) priority.
    pub fn select_best_priority(&mut self) -> bool {
        if self.alternates.is_empty() {
            return false;
        }
        if let Some((idx, _)) = self
            .alternates
            .iter()
            .enumerate()
            .min_by_key(|(_, a)| a.priority)
        {
            self.selected_index = idx;
            true
        } else {
            false
        }
    }

    /// Select by language. Returns false if no match found.
    pub fn select_by_language(&mut self, language: &str) -> bool {
        if let Some(idx) = self
            .alternates
            .iter()
            .position(|a| a.language.as_deref() == Some(language))
        {
            self.selected_index = idx;
            true
        } else {
            false
        }
    }

    /// Find alternates matching a tag.
    #[must_use]
    pub fn find_by_tag(&self, tag: &str) -> Vec<&Alternate> {
        self.alternates.iter().filter(|a| a.has_tag(tag)).collect()
    }

    /// Get the effective duration (duration of the selected alternate).
    #[must_use]
    pub fn duration(&self) -> i64 {
        self.selected().map_or(0, |a| a.duration)
    }

    /// Remove an alternate by index. Adjusts selection if necessary.
    pub fn remove_alternate(&mut self, index: usize) -> Option<Alternate> {
        if index >= self.alternates.len() {
            return None;
        }
        let removed = self.alternates.remove(index);
        if self.alternates.is_empty() {
            self.selected_index = 0;
        } else if self.selected_index >= self.alternates.len() {
            self.selected_index = self.alternates.len() - 1;
        }
        Some(removed)
    }

    /// Swap two alternates by index.
    pub fn swap_alternates(&mut self, a: usize, b: usize) -> bool {
        if a < self.alternates.len() && b < self.alternates.len() {
            self.alternates.swap(a, b);
            // Update selected index if one of the swapped items was selected
            if self.selected_index == a {
                self.selected_index = b;
            } else if self.selected_index == b {
                self.selected_index = a;
            }
            true
        } else {
            false
        }
    }

    /// Validate the selector structure.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.alternates.is_empty() {
            issues.push("Selector has no alternates".to_string());
        }

        if self.selected_index >= self.alternates.len() && !self.alternates.is_empty() {
            issues.push(format!(
                "Selected index {} is out of range (max {})",
                self.selected_index,
                self.alternates.len() - 1
            ));
        }

        if self.enforce_equal_duration && self.alternates.len() > 1 {
            let first_dur = self.alternates[0].duration;
            for (i, alt) in self.alternates.iter().enumerate().skip(1) {
                if alt.duration != first_dur {
                    issues.push(format!(
                        "Alternate {} has duration {} but expected {} (equal duration enforced)",
                        i, alt.duration, first_dur
                    ));
                }
            }
        }

        issues
    }

    /// Get all approved alternates.
    #[must_use]
    pub fn approved_alternates(&self) -> Vec<&Alternate> {
        self.alternates.iter().filter(|a| a.approved).collect()
    }

    /// Get all available languages across alternates.
    #[must_use]
    pub fn available_languages(&self) -> Vec<&str> {
        self.alternates
            .iter()
            .filter_map(|a| a.language.as_deref())
            .collect()
    }
}

/// Manages a collection of selectors within a composition.
#[derive(Debug, Clone)]
pub struct SelectorRegistry {
    /// All selectors indexed by ID.
    selectors: HashMap<Uuid, Selector>,
    /// Name-to-ID index.
    name_index: HashMap<String, Uuid>,
}

impl SelectorRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            selectors: HashMap::new(),
            name_index: HashMap::new(),
        }
    }

    /// Register a selector.
    pub fn register(&mut self, selector: Selector) {
        let id = selector.selector_id;
        self.name_index.insert(selector.name.clone(), id);
        self.selectors.insert(id, selector);
    }

    /// Get a selector by ID.
    #[must_use]
    pub fn get(&self, id: &Uuid) -> Option<&Selector> {
        self.selectors.get(id)
    }

    /// Get a mutable selector by ID.
    pub fn get_mut(&mut self, id: &Uuid) -> Option<&mut Selector> {
        self.selectors.get_mut(id)
    }

    /// Find a selector by name.
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<&Selector> {
        self.name_index
            .get(name)
            .and_then(|id| self.selectors.get(id))
    }

    /// Get total number of selectors.
    #[must_use]
    pub fn count(&self) -> usize {
        self.selectors.len()
    }

    /// Get all selectors.
    #[must_use]
    pub fn all_selectors(&self) -> Vec<&Selector> {
        self.selectors.values().collect()
    }

    /// Remove a selector by ID.
    pub fn remove(&mut self, id: &Uuid) -> Option<Selector> {
        if let Some(sel) = self.selectors.remove(id) {
            self.name_index.retain(|_, v| v != id);
            Some(sel)
        } else {
            None
        }
    }
}

impl Default for SelectorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selection_strategy_label() {
        assert_eq!(SelectionStrategy::Manual.label(), "Manual");
        assert_eq!(SelectionStrategy::LanguageBased.label(), "Language-Based");
        assert_eq!(SelectionStrategy::RatingBased.label(), "Rating-Based");
        assert_eq!(SelectionStrategy::PlatformBased.label(), "Platform-Based");
        assert_eq!(SelectionStrategy::QualityBased.label(), "Quality-Based");
        assert_eq!(SelectionStrategy::Scripted.label(), "Scripted");
    }

    #[test]
    fn test_selection_strategy_automatic() {
        assert!(!SelectionStrategy::Manual.is_automatic());
        assert!(SelectionStrategy::LanguageBased.is_automatic());
        assert!(SelectionStrategy::Scripted.is_automatic());
    }

    #[test]
    fn test_alternate_creation() {
        let alt = Alternate::new("Take 1", 1000);
        assert_eq!(alt.label, "Take 1");
        assert_eq!(alt.duration, 1000);
        assert_eq!(alt.start_offset, 0);
        assert!(!alt.approved);
        assert!(alt.tags.is_empty());
    }

    #[test]
    fn test_alternate_builder() {
        let mob_id = Uuid::new_v4();
        let alt = Alternate::new("English VO", 500)
            .with_language("en")
            .with_priority(10)
            .with_source(mob_id, 3);
        assert_eq!(alt.language.as_deref(), Some("en"));
        assert_eq!(alt.priority, 10);
        assert_eq!(alt.mob_id, Some(mob_id));
        assert_eq!(alt.track_id, Some(3));
    }

    #[test]
    fn test_alternate_tags() {
        let mut alt = Alternate::new("Take 2", 1000);
        alt.add_tag("preferred");
        alt.add_tag("color-graded");
        assert!(alt.has_tag("preferred"));
        assert!(!alt.has_tag("rejected"));
        assert_eq!(alt.tags.len(), 2);
    }

    #[test]
    fn test_selector_creation() {
        let sel =
            Selector::new("Language Selector").with_strategy(SelectionStrategy::LanguageBased);
        assert_eq!(sel.name, "Language Selector");
        assert_eq!(sel.strategy, SelectionStrategy::LanguageBased);
        assert_eq!(sel.alternate_count(), 0);
    }

    #[test]
    fn test_selector_add_and_select() {
        let mut sel = Selector::new("Takes");
        sel.add_alternate(Alternate::new("Take 1", 1000));
        sel.add_alternate(Alternate::new("Take 2", 1200));
        sel.add_alternate(Alternate::new("Take 3", 900));
        assert_eq!(sel.alternate_count(), 3);
        assert_eq!(sel.selected_index(), 0);
        assert_eq!(
            sel.selected().expect("selected should succeed").label,
            "Take 1"
        );
        assert_eq!(sel.duration(), 1000);

        assert!(sel.select(2));
        assert_eq!(
            sel.selected().expect("selected should succeed").label,
            "Take 3"
        );
        assert_eq!(sel.duration(), 900);

        assert!(!sel.select(10));
    }

    #[test]
    fn test_selector_select_by_id() {
        let mut sel = Selector::new("Test");
        let alt = Alternate::new("Target", 500);
        let target_id = alt.alternate_id;
        sel.add_alternate(Alternate::new("First", 300));
        sel.add_alternate(alt);
        assert!(sel.select_by_id(&target_id));
        assert_eq!(
            sel.selected().expect("selected should succeed").label,
            "Target"
        );
        assert!(!sel.select_by_id(&Uuid::new_v4()));
    }

    #[test]
    fn test_selector_select_by_language() {
        let mut sel = Selector::new("Languages").with_strategy(SelectionStrategy::LanguageBased);
        sel.add_alternate(Alternate::new("English", 1000).with_language("en"));
        sel.add_alternate(Alternate::new("Japanese", 1000).with_language("ja"));
        sel.add_alternate(Alternate::new("French", 1000).with_language("fr"));

        assert!(sel.select_by_language("ja"));
        assert_eq!(
            sel.selected().expect("selected should succeed").label,
            "Japanese"
        );
        assert!(!sel.select_by_language("de"));
    }

    #[test]
    fn test_selector_select_best_priority() {
        let mut sel = Selector::new("Priority");
        sel.add_alternate(Alternate::new("Low", 100).with_priority(50));
        sel.add_alternate(Alternate::new("High", 100).with_priority(1));
        sel.add_alternate(Alternate::new("Medium", 100).with_priority(25));

        assert!(sel.select_best_priority());
        assert_eq!(
            sel.selected().expect("selected should succeed").label,
            "High"
        );
    }

    #[test]
    fn test_selector_find_by_tag() {
        let mut sel = Selector::new("Tagged");
        let mut alt1 = Alternate::new("A", 100);
        alt1.add_tag("approved");
        let mut alt2 = Alternate::new("B", 100);
        alt2.add_tag("approved");
        alt2.add_tag("color-graded");
        sel.add_alternate(alt1);
        sel.add_alternate(alt2);
        sel.add_alternate(Alternate::new("C", 100));

        assert_eq!(sel.find_by_tag("approved").len(), 2);
        assert_eq!(sel.find_by_tag("color-graded").len(), 1);
        assert_eq!(sel.find_by_tag("none").len(), 0);
    }

    #[test]
    fn test_selector_remove_alternate() {
        let mut sel = Selector::new("Test");
        sel.add_alternate(Alternate::new("A", 100));
        sel.add_alternate(Alternate::new("B", 200));
        sel.add_alternate(Alternate::new("C", 300));
        sel.select(2);

        // Remove the selected one
        let removed = sel.remove_alternate(2);
        assert!(removed.is_some());
        assert_eq!(removed.expect("test expectation failed").label, "C");
        // Selection index should adjust
        assert!(sel.selected_index() < sel.alternate_count());
    }

    #[test]
    fn test_selector_swap_alternates() {
        let mut sel = Selector::new("Swap");
        sel.add_alternate(Alternate::new("A", 100));
        sel.add_alternate(Alternate::new("B", 200));
        sel.select(0);

        assert!(sel.swap_alternates(0, 1));
        // After swap, selected_index should follow the originally selected item
        assert_eq!(sel.selected().expect("selected should succeed").label, "A");
        assert_eq!(sel.alternates()[0].label, "B");
        assert!(!sel.swap_alternates(0, 99));
    }

    #[test]
    fn test_selector_validate() {
        let sel = Selector::new("Empty");
        let issues = sel.validate();
        assert!(!issues.is_empty());

        let mut sel2 = Selector::new("Good");
        sel2.add_alternate(Alternate::new("A", 100));
        assert!(sel2.validate().is_empty());
    }

    #[test]
    fn test_selector_validate_equal_duration() {
        let mut sel = Selector::new("Equal Duration");
        sel.enforce_equal_duration = true;
        sel.add_alternate(Alternate::new("A", 100));
        sel.add_alternate(Alternate::new("B", 200));
        let issues = sel.validate();
        assert!(issues.iter().any(|i| i.contains("equal duration")));
    }

    #[test]
    fn test_selector_available_languages() {
        let mut sel = Selector::new("Langs");
        sel.add_alternate(Alternate::new("EN", 100).with_language("en"));
        sel.add_alternate(Alternate::new("JA", 100).with_language("ja"));
        sel.add_alternate(Alternate::new("NoLang", 100));
        let langs = sel.available_languages();
        assert_eq!(langs.len(), 2);
        assert!(langs.contains(&"en"));
        assert!(langs.contains(&"ja"));
    }

    #[test]
    fn test_registry_operations() {
        let mut reg = SelectorRegistry::new();
        let mut sel = Selector::new("Language Selector");
        sel.add_alternate(Alternate::new("EN", 100));
        let sel_id = sel.selector_id;
        reg.register(sel);

        assert_eq!(reg.count(), 1);
        assert!(reg.get(&sel_id).is_some());
        assert!(reg.find_by_name("Language Selector").is_some());
        assert!(reg.find_by_name("Nonexistent").is_none());

        let removed = reg.remove(&sel_id);
        assert!(removed.is_some());
        assert_eq!(reg.count(), 0);
    }
}
