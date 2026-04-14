#![allow(dead_code)]

//! Stream deck and hotkey integration for scene switching and actions.
//!
//! Provides a configurable hotkey registry that maps keyboard shortcuts and
//! stream-deck button presses to streaming actions such as scene switching,
//! mute toggling, replay saving, and custom user-defined callbacks.
//!
//! # Architecture
//!
//! - **`HotkeyRegistry`**: Central registry that owns action bindings and
//!   dispatches incoming key events to the correct handler.
//! - **`KeyCombo`**: A combination of modifier keys and a primary key code.
//! - **`StreamAction`**: An enum of built-in actions that can be triggered.
//! - **`ActionBinding`**: Associates a `KeyCombo` (or deck button ID) with
//!   a `StreamAction`.
//! - **`StreamDeckLayout`**: Grid-based button layout for hardware decks.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::{GamingError, GamingResult};

// ---------------------------------------------------------------------------
// Key definitions
// ---------------------------------------------------------------------------

/// Modifier keys that can be combined with a primary key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modifier {
    /// Control / Cmd.
    Ctrl,
    /// Shift.
    Shift,
    /// Alt / Option.
    Alt,
    /// Super / Win / Meta.
    Super,
}

/// A virtual key code (platform-independent).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyCode {
    /// Function keys F1-F24.
    F(u8),
    /// Numeric keys 0-9.
    Num(u8),
    /// Letter keys A-Z.
    Letter(char),
    /// Numpad keys 0-9.
    Numpad(u8),
    /// A stream-deck hardware button by zero-based index.
    DeckButton(u16),
}

impl std::fmt::Display for KeyCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F(n) => write!(f, "F{n}"),
            Self::Num(n) => write!(f, "{n}"),
            Self::Letter(c) => write!(f, "{}", c.to_ascii_uppercase()),
            Self::Numpad(n) => write!(f, "Numpad{n}"),
            Self::DeckButton(id) => write!(f, "Deck#{id}"),
        }
    }
}

/// A key combination: zero or more modifiers + a primary key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KeyCombo {
    /// Modifier keys held.
    pub modifiers: Vec<Modifier>,
    /// Primary key.
    pub key: KeyCode,
}

impl KeyCombo {
    /// Create a new key combo with no modifiers.
    #[must_use]
    pub fn new(key: KeyCode) -> Self {
        Self {
            modifiers: Vec::new(),
            key,
        }
    }

    /// Create a key combo with a single modifier.
    #[must_use]
    pub fn with_modifier(key: KeyCode, modifier: Modifier) -> Self {
        Self {
            modifiers: vec![modifier],
            key,
        }
    }

    /// Create a key combo with multiple modifiers.
    #[must_use]
    pub fn with_modifiers(key: KeyCode, modifiers: &[Modifier]) -> Self {
        let mut mods: Vec<Modifier> = modifiers.to_vec();
        mods.sort_by_key(|m| *m as u8);
        mods.dedup();
        Self {
            modifiers: mods,
            key,
        }
    }

    /// Human-readable label for this combo (e.g. "Ctrl+Shift+F1").
    #[must_use]
    pub fn label(&self) -> String {
        let mut parts: Vec<String> = self
            .modifiers
            .iter()
            .map(|m| match m {
                Modifier::Ctrl => "Ctrl".to_string(),
                Modifier::Shift => "Shift".to_string(),
                Modifier::Alt => "Alt".to_string(),
                Modifier::Super => "Super".to_string(),
            })
            .collect();
        parts.push(format!("{}", self.key));
        parts.join("+")
    }

    /// Canonical key for hashing — modifiers sorted.
    fn canonical(&self) -> Self {
        let mut mods = self.modifiers.clone();
        mods.sort_by_key(|m| *m as u8);
        mods.dedup();
        Self {
            modifiers: mods,
            key: self.key,
        }
    }
}

impl std::fmt::Display for KeyCombo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// Stream actions
// ---------------------------------------------------------------------------

/// Built-in streaming actions triggered by hotkeys or deck buttons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamAction {
    /// Switch to a named scene.
    SwitchScene(String),
    /// Toggle mute on a named audio source.
    ToggleMute(String),
    /// Start streaming.
    StartStream,
    /// Stop streaming.
    StopStream,
    /// Pause streaming.
    PauseStream,
    /// Resume streaming.
    ResumeStream,
    /// Save instant replay.
    SaveReplay,
    /// Toggle recording.
    ToggleRecording,
    /// Take a screenshot.
    Screenshot,
    /// Toggle a source's visibility in the current scene.
    ToggleSource(String),
    /// Increase volume of a named audio source by a percentage (0-100).
    VolumeUp(String, u8),
    /// Decrease volume of a named audio source by a percentage (0-100).
    VolumeDown(String, u8),
    /// Trigger a stinger transition to the named scene.
    StingerTransition(String),
    /// Custom user-defined action with a string identifier.
    Custom(String),
}

impl StreamAction {
    /// Human-readable description of the action.
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Self::SwitchScene(s) => format!("Switch to scene '{s}'"),
            Self::ToggleMute(s) => format!("Toggle mute on '{s}'"),
            Self::StartStream => "Start streaming".into(),
            Self::StopStream => "Stop streaming".into(),
            Self::PauseStream => "Pause streaming".into(),
            Self::ResumeStream => "Resume streaming".into(),
            Self::SaveReplay => "Save instant replay".into(),
            Self::ToggleRecording => "Toggle recording".into(),
            Self::Screenshot => "Take screenshot".into(),
            Self::ToggleSource(s) => format!("Toggle source '{s}'"),
            Self::VolumeUp(s, pct) => format!("Volume +{pct}% on '{s}'"),
            Self::VolumeDown(s, pct) => format!("Volume -{pct}% on '{s}'"),
            Self::StingerTransition(s) => format!("Stinger transition to '{s}'"),
            Self::Custom(id) => format!("Custom action '{id}'"),
        }
    }
}

// ---------------------------------------------------------------------------
// Action binding
// ---------------------------------------------------------------------------

/// An association between a key combination and a stream action.
#[derive(Debug, Clone)]
pub struct ActionBinding {
    /// The key combination that triggers this action.
    pub combo: KeyCombo,
    /// The action to perform.
    pub action: StreamAction,
    /// Whether this binding is currently enabled.
    pub enabled: bool,
    /// Optional human-readable label for display on stream deck buttons.
    pub label: Option<String>,
    /// Optional icon identifier for stream deck.
    pub icon: Option<String>,
}

impl ActionBinding {
    /// Create a new enabled binding.
    #[must_use]
    pub fn new(combo: KeyCombo, action: StreamAction) -> Self {
        Self {
            combo,
            action,
            enabled: true,
            label: None,
            icon: None,
        }
    }

    /// Set a display label.
    #[must_use]
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    /// Set an icon identifier.
    #[must_use]
    pub fn with_icon(mut self, icon: &str) -> Self {
        self.icon = Some(icon.to_string());
        self
    }

    /// Disable this binding.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enable this binding.
    pub fn enable(&mut self) {
        self.enabled = true;
    }
}

// ---------------------------------------------------------------------------
// Stream Deck Layout
// ---------------------------------------------------------------------------

/// A hardware stream deck button layout (grid of rows x columns).
#[derive(Debug, Clone)]
pub struct StreamDeckLayout {
    /// Number of rows on the deck.
    pub rows: u8,
    /// Number of columns on the deck.
    pub cols: u8,
    /// Page index (for multi-page decks).
    pub current_page: u16,
    /// Total number of pages.
    pub total_pages: u16,
    /// Mapping from button index to action binding.
    pub buttons: HashMap<u16, ActionBinding>,
}

impl StreamDeckLayout {
    /// Create a new deck layout.
    ///
    /// # Errors
    ///
    /// Returns error if rows or cols is zero.
    pub fn new(rows: u8, cols: u8) -> GamingResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(GamingError::InvalidConfig(
                "Stream deck must have at least 1 row and 1 column".into(),
            ));
        }
        Ok(Self {
            rows,
            cols,
            current_page: 0,
            total_pages: 1,
            buttons: HashMap::new(),
        })
    }

    /// Total number of buttons on this page.
    #[must_use]
    pub fn button_count(&self) -> u16 {
        u16::from(self.rows) * u16::from(self.cols)
    }

    /// Assign an action to a button index.
    ///
    /// # Errors
    ///
    /// Returns error if the button index exceeds the grid size.
    pub fn assign_button(&mut self, index: u16, action: StreamAction) -> GamingResult<()> {
        if index >= self.button_count() {
            return Err(GamingError::InvalidConfig(format!(
                "Button index {index} exceeds grid size {}",
                self.button_count()
            )));
        }
        let combo = KeyCombo::new(KeyCode::DeckButton(index));
        let binding = ActionBinding::new(combo, action);
        self.buttons.insert(index, binding);
        Ok(())
    }

    /// Remove a button assignment.
    pub fn clear_button(&mut self, index: u16) {
        self.buttons.remove(&index);
    }

    /// Get the action assigned to a button, if any.
    #[must_use]
    pub fn get_button_action(&self, index: u16) -> Option<&StreamAction> {
        self.buttons.get(&index).map(|b| &b.action)
    }

    /// Switch to the next page.
    pub fn next_page(&mut self) {
        if self.current_page + 1 < self.total_pages {
            self.current_page += 1;
        }
    }

    /// Switch to the previous page.
    pub fn prev_page(&mut self) {
        self.current_page = self.current_page.saturating_sub(1);
    }

    /// Set total pages.
    pub fn set_total_pages(&mut self, pages: u16) {
        self.total_pages = pages.max(1);
    }
}

// ---------------------------------------------------------------------------
// Dispatch result
// ---------------------------------------------------------------------------

/// Result of dispatching a key event through the registry.
#[derive(Debug, Clone)]
pub struct DispatchResult {
    /// The action that was triggered.
    pub action: StreamAction,
    /// When the action was triggered.
    pub triggered_at: Instant,
    /// The key combo that triggered it.
    pub combo: KeyCombo,
}

// ---------------------------------------------------------------------------
// Hotkey Registry
// ---------------------------------------------------------------------------

/// Central hotkey registry managing all action bindings and dispatching
/// incoming key events to the appropriate handlers.
#[derive(Debug)]
pub struct HotkeyRegistry {
    /// Bindings indexed by canonical key combo.
    bindings: HashMap<KeyCombo, ActionBinding>,
    /// History of dispatched actions (for undo/audit).
    dispatch_history: Vec<DispatchResult>,
    /// Maximum history length.
    max_history: usize,
    /// Whether the registry is globally enabled.
    enabled: bool,
    /// Cooldown between repeated triggers of the same key combo.
    cooldown: Duration,
    /// Last trigger time per combo.
    last_trigger: HashMap<KeyCombo, Instant>,
}

impl HotkeyRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            dispatch_history: Vec::new(),
            max_history: 1000,
            enabled: true,
            cooldown: Duration::from_millis(200),
            last_trigger: HashMap::new(),
        }
    }

    /// Register a new hotkey binding.
    ///
    /// # Errors
    ///
    /// Returns error if the key combo is already bound.
    pub fn register(&mut self, binding: ActionBinding) -> GamingResult<()> {
        let key = binding.combo.canonical();
        if self.bindings.contains_key(&key) {
            return Err(GamingError::InvalidConfig(format!(
                "Key combo '{}' is already bound",
                key.label()
            )));
        }
        self.bindings.insert(key, binding);
        Ok(())
    }

    /// Register or overwrite a binding (upsert).
    pub fn register_or_replace(&mut self, binding: ActionBinding) {
        let key = binding.combo.canonical();
        self.bindings.insert(key, binding);
    }

    /// Unregister a binding by key combo.
    pub fn unregister(&mut self, combo: &KeyCombo) -> bool {
        let key = combo.canonical();
        self.bindings.remove(&key).is_some()
    }

    /// Dispatch a key event. Returns the triggered action if a binding matches
    /// and is enabled, and the cooldown has elapsed.
    pub fn dispatch(&mut self, combo: &KeyCombo) -> Option<DispatchResult> {
        if !self.enabled {
            return None;
        }

        let key = combo.canonical();
        let binding = self.bindings.get(&key)?;
        if !binding.enabled {
            return None;
        }

        // Check cooldown
        let now = Instant::now();
        if let Some(last) = self.last_trigger.get(&key) {
            if now.duration_since(*last) < self.cooldown {
                return None;
            }
        }

        let result = DispatchResult {
            action: binding.action.clone(),
            triggered_at: now,
            combo: combo.clone(),
        };

        self.last_trigger.insert(key, now);

        // Add to history
        self.dispatch_history.push(result.clone());
        if self.dispatch_history.len() > self.max_history {
            self.dispatch_history.remove(0);
        }

        Some(result)
    }

    /// Number of registered bindings.
    #[must_use]
    pub fn binding_count(&self) -> usize {
        self.bindings.len()
    }

    /// Get all registered bindings.
    #[must_use]
    pub fn bindings(&self) -> Vec<&ActionBinding> {
        self.bindings.values().collect()
    }

    /// Get dispatch history.
    #[must_use]
    pub fn history(&self) -> &[DispatchResult] {
        &self.dispatch_history
    }

    /// Clear dispatch history.
    pub fn clear_history(&mut self) {
        self.dispatch_history.clear();
    }

    /// Set the cooldown duration between repeated key presses.
    pub fn set_cooldown(&mut self, cooldown: Duration) {
        self.cooldown = cooldown;
    }

    /// Enable the registry globally.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable the registry globally (all dispatches will be ignored).
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if globally enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Set up a standard set of default bindings for common streaming actions.
    ///
    /// # Errors
    ///
    /// Returns error if any binding conflicts with an existing one.
    pub fn load_defaults(&mut self) -> GamingResult<()> {
        let defaults = vec![
            ActionBinding::new(
                KeyCombo::with_modifier(KeyCode::F(1), Modifier::Ctrl),
                StreamAction::SwitchScene("Scene 1".into()),
            ),
            ActionBinding::new(
                KeyCombo::with_modifier(KeyCode::F(2), Modifier::Ctrl),
                StreamAction::SwitchScene("Scene 2".into()),
            ),
            ActionBinding::new(
                KeyCombo::with_modifier(KeyCode::F(3), Modifier::Ctrl),
                StreamAction::SwitchScene("Scene 3".into()),
            ),
            ActionBinding::new(
                KeyCombo::with_modifier(KeyCode::F(5), Modifier::Ctrl),
                StreamAction::StartStream,
            ),
            ActionBinding::new(
                KeyCombo::with_modifier(KeyCode::F(6), Modifier::Ctrl),
                StreamAction::StopStream,
            ),
            ActionBinding::new(
                KeyCombo::with_modifier(KeyCode::F(7), Modifier::Ctrl),
                StreamAction::SaveReplay,
            ),
            ActionBinding::new(
                KeyCombo::with_modifier(KeyCode::F(8), Modifier::Ctrl),
                StreamAction::ToggleRecording,
            ),
            ActionBinding::new(
                KeyCombo::with_modifier(KeyCode::F(9), Modifier::Ctrl),
                StreamAction::Screenshot,
            ),
            ActionBinding::new(
                KeyCombo::with_modifiers(KeyCode::Letter('m'), &[Modifier::Ctrl, Modifier::Shift]),
                StreamAction::ToggleMute("Microphone".into()),
            ),
        ];

        for binding in defaults {
            self.register(binding)?;
        }

        Ok(())
    }
}

impl Default for HotkeyRegistry {
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

    #[test]
    fn test_key_combo_label() {
        let combo = KeyCombo::with_modifiers(KeyCode::F(1), &[Modifier::Ctrl, Modifier::Shift]);
        let label = combo.label();
        assert!(label.contains("Ctrl"));
        assert!(label.contains("Shift"));
        assert!(label.contains("F1"));
    }

    #[test]
    fn test_key_combo_no_modifiers() {
        let combo = KeyCombo::new(KeyCode::Letter('a'));
        assert_eq!(combo.label(), "A");
    }

    #[test]
    fn test_key_combo_display() {
        let combo = KeyCombo::with_modifier(KeyCode::F(5), Modifier::Ctrl);
        let display = format!("{combo}");
        assert!(display.contains("Ctrl"));
        assert!(display.contains("F5"));
    }

    #[test]
    fn test_keycode_display() {
        assert_eq!(format!("{}", KeyCode::F(12)), "F12");
        assert_eq!(format!("{}", KeyCode::Num(5)), "5");
        assert_eq!(format!("{}", KeyCode::Letter('x')), "X");
        assert_eq!(format!("{}", KeyCode::Numpad(3)), "Numpad3");
        assert_eq!(format!("{}", KeyCode::DeckButton(7)), "Deck#7");
    }

    #[test]
    fn test_stream_action_description() {
        let action = StreamAction::SwitchScene("Main".into());
        assert!(action.description().contains("Main"));

        assert!(StreamAction::SaveReplay.description().contains("replay"));
        assert!(StreamAction::Screenshot
            .description()
            .contains("screenshot"));
    }

    #[test]
    fn test_action_binding_enable_disable() {
        let combo = KeyCombo::new(KeyCode::F(1));
        let mut binding = ActionBinding::new(combo, StreamAction::StartStream);
        assert!(binding.enabled);

        binding.disable();
        assert!(!binding.enabled);

        binding.enable();
        assert!(binding.enabled);
    }

    #[test]
    fn test_action_binding_with_label_and_icon() {
        let combo = KeyCombo::new(KeyCode::DeckButton(0));
        let binding = ActionBinding::new(combo, StreamAction::StartStream)
            .with_label("Go Live")
            .with_icon("play_icon");
        assert_eq!(binding.label.as_deref(), Some("Go Live"));
        assert_eq!(binding.icon.as_deref(), Some("play_icon"));
    }

    #[test]
    fn test_registry_register_and_dispatch() {
        let mut registry = HotkeyRegistry::new();
        registry.set_cooldown(Duration::ZERO); // disable cooldown for test

        let combo = KeyCombo::with_modifier(KeyCode::F(1), Modifier::Ctrl);
        let binding = ActionBinding::new(combo.clone(), StreamAction::StartStream);
        registry.register(binding).expect("register should succeed");

        assert_eq!(registry.binding_count(), 1);

        let result = registry.dispatch(&combo);
        assert!(result.is_some());
        let result = result.expect("dispatch result");
        assert_eq!(result.action, StreamAction::StartStream);
    }

    #[test]
    fn test_registry_duplicate_binding_rejected() {
        let mut registry = HotkeyRegistry::new();
        let combo = KeyCombo::new(KeyCode::F(1));
        let binding1 = ActionBinding::new(combo.clone(), StreamAction::StartStream);
        let binding2 = ActionBinding::new(combo, StreamAction::StopStream);

        registry.register(binding1).expect("first register ok");
        assert!(registry.register(binding2).is_err());
    }

    #[test]
    fn test_registry_register_or_replace() {
        let mut registry = HotkeyRegistry::new();
        let combo = KeyCombo::new(KeyCode::F(1));
        let binding1 = ActionBinding::new(combo.clone(), StreamAction::StartStream);
        let binding2 = ActionBinding::new(combo.clone(), StreamAction::StopStream);

        registry.register_or_replace(binding1);
        registry.register_or_replace(binding2);
        assert_eq!(registry.binding_count(), 1);

        registry.set_cooldown(Duration::ZERO);
        let result = registry.dispatch(&combo);
        assert!(result.is_some());
        assert_eq!(
            result.expect("dispatch result").action,
            StreamAction::StopStream
        );
    }

    #[test]
    fn test_registry_unregister() {
        let mut registry = HotkeyRegistry::new();
        let combo = KeyCombo::new(KeyCode::F(1));
        let binding = ActionBinding::new(combo.clone(), StreamAction::StartStream);
        registry.register(binding).expect("register");

        assert!(registry.unregister(&combo));
        assert_eq!(registry.binding_count(), 0);

        // Unregistering again returns false
        assert!(!registry.unregister(&combo));
    }

    #[test]
    fn test_registry_disabled_dispatch_returns_none() {
        let mut registry = HotkeyRegistry::new();
        registry.set_cooldown(Duration::ZERO);
        let combo = KeyCombo::new(KeyCode::F(1));
        let binding = ActionBinding::new(combo.clone(), StreamAction::StartStream);
        registry.register(binding).expect("register");

        registry.disable();
        assert!(!registry.is_enabled());
        assert!(registry.dispatch(&combo).is_none());

        registry.enable();
        assert!(registry.dispatch(&combo).is_some());
    }

    #[test]
    fn test_registry_disabled_binding_not_dispatched() {
        let mut registry = HotkeyRegistry::new();
        registry.set_cooldown(Duration::ZERO);
        let combo = KeyCombo::new(KeyCode::F(1));
        let mut binding = ActionBinding::new(combo.clone(), StreamAction::StartStream);
        binding.disable();
        registry.register(binding).expect("register");

        assert!(registry.dispatch(&combo).is_none());
    }

    #[test]
    fn test_registry_dispatch_history() {
        let mut registry = HotkeyRegistry::new();
        registry.set_cooldown(Duration::ZERO);

        let combo = KeyCombo::new(KeyCode::F(1));
        let binding = ActionBinding::new(combo.clone(), StreamAction::SaveReplay);
        registry.register(binding).expect("register");

        registry.dispatch(&combo);
        registry.dispatch(&combo);

        assert_eq!(registry.history().len(), 2);

        registry.clear_history();
        assert!(registry.history().is_empty());
    }

    #[test]
    fn test_registry_unbound_key_returns_none() {
        let mut registry = HotkeyRegistry::new();
        let combo = KeyCombo::new(KeyCode::F(12));
        assert!(registry.dispatch(&combo).is_none());
    }

    #[test]
    fn test_registry_load_defaults() {
        let mut registry = HotkeyRegistry::new();
        registry.load_defaults().expect("load defaults");
        assert!(registry.binding_count() >= 9);
    }

    // -- StreamDeckLayout tests --

    #[test]
    fn test_deck_layout_creation() {
        let layout = StreamDeckLayout::new(3, 5).expect("valid layout");
        assert_eq!(layout.button_count(), 15);
    }

    #[test]
    fn test_deck_layout_zero_rows() {
        assert!(StreamDeckLayout::new(0, 5).is_err());
    }

    #[test]
    fn test_deck_layout_zero_cols() {
        assert!(StreamDeckLayout::new(3, 0).is_err());
    }

    #[test]
    fn test_deck_assign_and_get_button() {
        let mut layout = StreamDeckLayout::new(3, 5).expect("valid");
        layout
            .assign_button(0, StreamAction::StartStream)
            .expect("assign");
        layout
            .assign_button(14, StreamAction::StopStream)
            .expect("assign");

        assert_eq!(
            layout.get_button_action(0),
            Some(&StreamAction::StartStream)
        );
        assert_eq!(
            layout.get_button_action(14),
            Some(&StreamAction::StopStream)
        );
        assert!(layout.get_button_action(7).is_none());
    }

    #[test]
    fn test_deck_assign_out_of_range() {
        let mut layout = StreamDeckLayout::new(3, 5).expect("valid");
        assert!(layout.assign_button(15, StreamAction::StartStream).is_err());
    }

    #[test]
    fn test_deck_clear_button() {
        let mut layout = StreamDeckLayout::new(3, 5).expect("valid");
        layout
            .assign_button(0, StreamAction::StartStream)
            .expect("assign");
        layout.clear_button(0);
        assert!(layout.get_button_action(0).is_none());
    }

    #[test]
    fn test_deck_page_navigation() {
        let mut layout = StreamDeckLayout::new(3, 5).expect("valid");
        layout.set_total_pages(3);
        assert_eq!(layout.current_page, 0);

        layout.next_page();
        assert_eq!(layout.current_page, 1);

        layout.next_page();
        assert_eq!(layout.current_page, 2);

        // Should not go past last page
        layout.next_page();
        assert_eq!(layout.current_page, 2);

        layout.prev_page();
        assert_eq!(layout.current_page, 1);

        layout.prev_page();
        assert_eq!(layout.current_page, 0);

        // Should not go below 0
        layout.prev_page();
        assert_eq!(layout.current_page, 0);
    }

    #[test]
    fn test_volume_actions() {
        let up = StreamAction::VolumeUp("Game".into(), 10);
        assert!(up.description().contains("+10%"));

        let down = StreamAction::VolumeDown("Mic".into(), 5);
        assert!(down.description().contains("-5%"));
    }

    #[test]
    fn test_custom_action() {
        let action = StreamAction::Custom("run_ad_break".into());
        assert!(action.description().contains("run_ad_break"));
    }

    #[test]
    fn test_canonical_key_combo_dedup_modifiers() {
        let combo = KeyCombo::with_modifiers(
            KeyCode::F(1),
            &[Modifier::Ctrl, Modifier::Ctrl, Modifier::Shift],
        );
        // Duplicated Ctrl should be deduped
        assert_eq!(combo.modifiers.len(), 2);
    }
}
