#![allow(dead_code)]
//! Cue-based trigger system for broadcast automation.
//!
//! This module provides a cue trigger framework that fires automation events
//! based on timecode cues, DTMF tones, GPI signals, and manual operator cues.
//! Cue triggers are the backbone of frame-accurate broadcast automation,
//! enabling precise insertion of graphics, ad breaks, and source switches.

use std::collections::HashMap;
use std::fmt;

/// Type of cue source that initiates a trigger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CueSourceType {
    /// Timecode-based cue embedded in the media stream.
    Timecode,
    /// DTMF tone detected in the audio channel.
    Dtmf,
    /// GPI (General Purpose Interface) hardware input.
    Gpi,
    /// Manual operator cue from the control surface.
    Manual,
    /// SCTE-35 splice signal in transport stream.
    Scte35,
    /// Network trigger received via automation protocol.
    Network,
}

impl fmt::Display for CueSourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Timecode => write!(f, "Timecode"),
            Self::Dtmf => write!(f, "DTMF"),
            Self::Gpi => write!(f, "GPI"),
            Self::Manual => write!(f, "Manual"),
            Self::Scte35 => write!(f, "SCTE-35"),
            Self::Network => write!(f, "Network"),
        }
    }
}

/// Priority level for cue triggers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CuePriority {
    /// Low priority - can be deferred.
    Low,
    /// Normal priority - standard processing.
    Normal,
    /// High priority - processed before normal cues.
    High,
    /// Critical priority - immediate execution, cannot be deferred.
    Critical,
}

impl Default for CuePriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Current state of a cue trigger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CueTriggerState {
    /// Trigger is armed and waiting.
    Armed,
    /// Trigger has fired and action is pending.
    Fired,
    /// Trigger action has been executed.
    Executed,
    /// Trigger was disarmed before firing.
    Disarmed,
    /// Trigger encountered an error.
    Error,
}

/// Action to perform when a cue trigger fires.
#[derive(Debug, Clone, PartialEq)]
pub enum CueAction {
    /// Switch to a different video source.
    SwitchSource {
        /// Target source identifier.
        source_id: String,
    },
    /// Start an ad break.
    StartAdBreak {
        /// Duration of the ad break in frames.
        duration_frames: u64,
    },
    /// Insert a graphic overlay.
    InsertGraphic {
        /// Graphic template identifier.
        template_id: String,
        /// Duration in frames, or None for indefinite.
        duration_frames: Option<u64>,
    },
    /// Trigger an audio ducking event.
    AudioDuck {
        /// Target audio level in dB.
        target_db: f64,
        /// Ramp time in milliseconds.
        ramp_ms: u32,
    },
    /// Execute a custom macro.
    ExecuteMacro {
        /// Macro name to execute.
        macro_name: String,
    },
    /// No operation (used for testing).
    Noop,
}

/// A single cue trigger definition.
#[derive(Debug, Clone)]
pub struct CueTrigger {
    /// Unique identifier for this trigger.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Source type that activates this trigger.
    pub source_type: CueSourceType,
    /// Priority level.
    pub priority: CuePriority,
    /// Action to perform when triggered.
    pub action: CueAction,
    /// Current state of the trigger.
    pub state: CueTriggerState,
    /// Pre-roll offset in frames (fire this many frames early).
    pub pre_roll_frames: i64,
    /// Whether this trigger can fire multiple times.
    pub repeatable: bool,
    /// Number of times this trigger has fired.
    pub fire_count: u64,
    /// Optional timecode cue point (as frame number).
    pub cue_frame: Option<u64>,
    /// Whether the trigger is enabled.
    pub enabled: bool,
}

impl CueTrigger {
    /// Create a new cue trigger.
    pub fn new(id: impl Into<String>, name: impl Into<String>, source_type: CueSourceType, action: CueAction) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            source_type,
            priority: CuePriority::default(),
            action,
            state: CueTriggerState::Armed,
            pre_roll_frames: 0,
            repeatable: false,
            fire_count: 0,
            cue_frame: None,
            enabled: true,
        }
    }

    /// Set the priority level.
    pub fn with_priority(mut self, priority: CuePriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set the pre-roll offset in frames.
    pub fn with_pre_roll(mut self, frames: i64) -> Self {
        self.pre_roll_frames = frames;
        self
    }

    /// Set whether the trigger is repeatable.
    pub fn with_repeatable(mut self, repeatable: bool) -> Self {
        self.repeatable = repeatable;
        self
    }

    /// Set the cue frame for timecode triggers.
    pub fn with_cue_frame(mut self, frame: u64) -> Self {
        self.cue_frame = Some(frame);
        self
    }

    /// Arm the trigger for firing.
    pub fn arm(&mut self) {
        if self.enabled {
            self.state = CueTriggerState::Armed;
        }
    }

    /// Disarm the trigger.
    pub fn disarm(&mut self) {
        self.state = CueTriggerState::Disarmed;
    }

    /// Fire the trigger.
    pub fn fire(&mut self) -> bool {
        if self.state != CueTriggerState::Armed || !self.enabled {
            return false;
        }
        self.state = CueTriggerState::Fired;
        self.fire_count += 1;
        true
    }

    /// Mark the trigger action as executed.
    pub fn mark_executed(&mut self) {
        if self.state == CueTriggerState::Fired {
            self.state = CueTriggerState::Executed;
            if self.repeatable {
                self.state = CueTriggerState::Armed;
            }
        }
    }

    /// Check if the trigger should fire at the given frame number.
    #[allow(clippy::cast_precision_loss)]
    pub fn should_fire_at_frame(&self, current_frame: u64) -> bool {
        if self.state != CueTriggerState::Armed || !self.enabled {
            return false;
        }
        if let Some(cue_frame) = self.cue_frame {
            let adjusted = if self.pre_roll_frames >= 0 {
                cue_frame.saturating_sub(self.pre_roll_frames as u64)
            } else {
                cue_frame + self.pre_roll_frames.unsigned_abs()
            };
            current_frame >= adjusted
        } else {
            false
        }
    }

    /// Check if the trigger is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self.state, CueTriggerState::Executed | CueTriggerState::Disarmed | CueTriggerState::Error)
            && !self.repeatable
    }
}

/// Manages a collection of cue triggers for a channel.
#[derive(Debug)]
pub struct CueTriggerManager {
    /// Map of trigger ID to trigger.
    triggers: HashMap<String, CueTrigger>,
    /// Maximum number of triggers allowed.
    max_triggers: usize,
    /// Whether to auto-disarm fired non-repeatable triggers.
    auto_cleanup: bool,
}

impl CueTriggerManager {
    /// Create a new cue trigger manager.
    pub fn new(max_triggers: usize) -> Self {
        Self {
            triggers: HashMap::new(),
            max_triggers,
            auto_cleanup: true,
        }
    }

    /// Set auto-cleanup mode.
    pub fn with_auto_cleanup(mut self, enabled: bool) -> Self {
        self.auto_cleanup = enabled;
        self
    }

    /// Add a trigger to the manager.
    pub fn add_trigger(&mut self, trigger: CueTrigger) -> Result<(), String> {
        if self.triggers.len() >= self.max_triggers {
            return Err(format!("Maximum trigger count ({}) reached", self.max_triggers));
        }
        if self.triggers.contains_key(&trigger.id) {
            return Err(format!("Trigger with id '{}' already exists", trigger.id));
        }
        self.triggers.insert(trigger.id.clone(), trigger);
        Ok(())
    }

    /// Remove a trigger by ID.
    pub fn remove_trigger(&mut self, id: &str) -> Option<CueTrigger> {
        self.triggers.remove(id)
    }

    /// Get a reference to a trigger by ID.
    pub fn get_trigger(&self, id: &str) -> Option<&CueTrigger> {
        self.triggers.get(id)
    }

    /// Get a mutable reference to a trigger by ID.
    pub fn get_trigger_mut(&mut self, id: &str) -> Option<&mut CueTrigger> {
        self.triggers.get_mut(id)
    }

    /// Get all triggers that should fire at the given frame.
    pub fn triggers_at_frame(&self, frame: u64) -> Vec<&CueTrigger> {
        let mut ready: Vec<&CueTrigger> = self
            .triggers
            .values()
            .filter(|t| t.should_fire_at_frame(frame))
            .collect();
        ready.sort_by(|a, b| b.priority.cmp(&a.priority));
        ready
    }

    /// Fire all triggers that are ready at the given frame.
    pub fn fire_at_frame(&mut self, frame: u64) -> Vec<String> {
        let ids: Vec<String> = self
            .triggers
            .values()
            .filter(|t| t.should_fire_at_frame(frame))
            .map(|t| t.id.clone())
            .collect();

        let mut fired = Vec::new();
        for id in &ids {
            if let Some(trigger) = self.triggers.get_mut(id) {
                if trigger.fire() {
                    fired.push(id.clone());
                }
            }
        }
        fired
    }

    /// Get the number of triggers.
    pub fn count(&self) -> usize {
        self.triggers.len()
    }

    /// Get the number of armed triggers.
    pub fn armed_count(&self) -> usize {
        self.triggers.values().filter(|t| t.state == CueTriggerState::Armed).count()
    }

    /// Clean up terminal (non-repeatable, executed) triggers.
    pub fn cleanup_terminal(&mut self) -> usize {
        let terminal_ids: Vec<String> = self
            .triggers
            .iter()
            .filter(|(_, t)| t.is_terminal())
            .map(|(id, _)| id.clone())
            .collect();
        let count = terminal_ids.len();
        for id in terminal_ids {
            self.triggers.remove(&id);
        }
        count
    }

    /// Arm all triggers.
    pub fn arm_all(&mut self) {
        for trigger in self.triggers.values_mut() {
            trigger.arm();
        }
    }

    /// Disarm all triggers.
    pub fn disarm_all(&mut self) {
        for trigger in self.triggers.values_mut() {
            trigger.disarm();
        }
    }

    /// Get triggers filtered by source type.
    pub fn triggers_by_source(&self, source_type: CueSourceType) -> Vec<&CueTrigger> {
        self.triggers
            .values()
            .filter(|t| t.source_type == source_type)
            .collect()
    }
}

impl Default for CueTriggerManager {
    fn default() -> Self {
        Self::new(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cue_source_type_display() {
        assert_eq!(CueSourceType::Timecode.to_string(), "Timecode");
        assert_eq!(CueSourceType::Dtmf.to_string(), "DTMF");
        assert_eq!(CueSourceType::Gpi.to_string(), "GPI");
        assert_eq!(CueSourceType::Manual.to_string(), "Manual");
        assert_eq!(CueSourceType::Scte35.to_string(), "SCTE-35");
        assert_eq!(CueSourceType::Network.to_string(), "Network");
    }

    #[test]
    fn test_cue_priority_ordering() {
        assert!(CuePriority::Critical > CuePriority::High);
        assert!(CuePriority::High > CuePriority::Normal);
        assert!(CuePriority::Normal > CuePriority::Low);
    }

    #[test]
    fn test_cue_trigger_creation() {
        let trigger = CueTrigger::new("t1", "Test Trigger", CueSourceType::Timecode, CueAction::Noop);
        assert_eq!(trigger.id, "t1");
        assert_eq!(trigger.name, "Test Trigger");
        assert_eq!(trigger.state, CueTriggerState::Armed);
        assert_eq!(trigger.fire_count, 0);
        assert!(trigger.enabled);
    }

    #[test]
    fn test_cue_trigger_builder() {
        let trigger = CueTrigger::new("t2", "Priority Trigger", CueSourceType::Gpi, CueAction::Noop)
            .with_priority(CuePriority::High)
            .with_pre_roll(5)
            .with_repeatable(true)
            .with_cue_frame(1000);
        assert_eq!(trigger.priority, CuePriority::High);
        assert_eq!(trigger.pre_roll_frames, 5);
        assert!(trigger.repeatable);
        assert_eq!(trigger.cue_frame, Some(1000));
    }

    #[test]
    fn test_cue_trigger_fire() {
        let mut trigger = CueTrigger::new("t3", "Fire Test", CueSourceType::Manual, CueAction::Noop);
        assert!(trigger.fire());
        assert_eq!(trigger.state, CueTriggerState::Fired);
        assert_eq!(trigger.fire_count, 1);
        // Cannot fire again when already fired
        assert!(!trigger.fire());
    }

    #[test]
    fn test_cue_trigger_disarm() {
        let mut trigger = CueTrigger::new("t4", "Disarm Test", CueSourceType::Manual, CueAction::Noop);
        trigger.disarm();
        assert_eq!(trigger.state, CueTriggerState::Disarmed);
        assert!(!trigger.fire());
    }

    #[test]
    fn test_cue_trigger_repeatable() {
        let mut trigger = CueTrigger::new("t5", "Repeat Test", CueSourceType::Manual, CueAction::Noop)
            .with_repeatable(true);
        assert!(trigger.fire());
        trigger.mark_executed();
        // Repeatable triggers re-arm after execution
        assert_eq!(trigger.state, CueTriggerState::Armed);
        assert!(trigger.fire());
        assert_eq!(trigger.fire_count, 2);
    }

    #[test]
    fn test_cue_trigger_should_fire_at_frame() {
        let trigger = CueTrigger::new("t6", "Frame Test", CueSourceType::Timecode, CueAction::Noop)
            .with_cue_frame(100)
            .with_pre_roll(5);
        assert!(!trigger.should_fire_at_frame(90));
        assert!(trigger.should_fire_at_frame(95));
        assert!(trigger.should_fire_at_frame(100));
        assert!(trigger.should_fire_at_frame(110));
    }

    #[test]
    fn test_cue_trigger_is_terminal() {
        let mut trigger = CueTrigger::new("t7", "Terminal Test", CueSourceType::Manual, CueAction::Noop);
        assert!(!trigger.is_terminal());
        trigger.fire();
        trigger.mark_executed();
        assert!(trigger.is_terminal());
    }

    #[test]
    fn test_cue_trigger_manager_add_remove() {
        let mut manager = CueTriggerManager::new(10);
        let trigger = CueTrigger::new("t1", "Trigger 1", CueSourceType::Manual, CueAction::Noop);
        assert!(manager.add_trigger(trigger).is_ok());
        assert_eq!(manager.count(), 1);
        assert!(manager.remove_trigger("t1").is_some());
        assert_eq!(manager.count(), 0);
    }

    #[test]
    fn test_cue_trigger_manager_duplicate_id() {
        let mut manager = CueTriggerManager::new(10);
        let t1 = CueTrigger::new("dup", "First", CueSourceType::Manual, CueAction::Noop);
        let t2 = CueTrigger::new("dup", "Second", CueSourceType::Manual, CueAction::Noop);
        assert!(manager.add_trigger(t1).is_ok());
        assert!(manager.add_trigger(t2).is_err());
    }

    #[test]
    fn test_cue_trigger_manager_max_triggers() {
        let mut manager = CueTriggerManager::new(2);
        let t1 = CueTrigger::new("a", "A", CueSourceType::Manual, CueAction::Noop);
        let t2 = CueTrigger::new("b", "B", CueSourceType::Manual, CueAction::Noop);
        let t3 = CueTrigger::new("c", "C", CueSourceType::Manual, CueAction::Noop);
        assert!(manager.add_trigger(t1).is_ok());
        assert!(manager.add_trigger(t2).is_ok());
        assert!(manager.add_trigger(t3).is_err());
    }

    #[test]
    fn test_cue_trigger_manager_fire_at_frame() {
        let mut manager = CueTriggerManager::new(10);
        let t1 = CueTrigger::new("early", "Early", CueSourceType::Timecode, CueAction::Noop)
            .with_cue_frame(50);
        let t2 = CueTrigger::new("late", "Late", CueSourceType::Timecode, CueAction::Noop)
            .with_cue_frame(200);
        manager.add_trigger(t1).expect("add_trigger should succeed");
        manager.add_trigger(t2).expect("add_trigger should succeed");
        let fired = manager.fire_at_frame(100);
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0], "early");
    }

    #[test]
    fn test_cue_trigger_manager_arm_disarm_all() {
        let mut manager = CueTriggerManager::new(10);
        let t1 = CueTrigger::new("x", "X", CueSourceType::Manual, CueAction::Noop);
        let t2 = CueTrigger::new("y", "Y", CueSourceType::Manual, CueAction::Noop);
        manager.add_trigger(t1).expect("add_trigger should succeed");
        manager.add_trigger(t2).expect("add_trigger should succeed");
        manager.disarm_all();
        assert_eq!(manager.armed_count(), 0);
        manager.arm_all();
        assert_eq!(manager.armed_count(), 2);
    }

    #[test]
    fn test_cue_trigger_manager_cleanup_terminal() {
        let mut manager = CueTriggerManager::new(10);
        let mut t1 = CueTrigger::new("done", "Done", CueSourceType::Manual, CueAction::Noop);
        t1.fire();
        t1.mark_executed();
        let t2 = CueTrigger::new("alive", "Alive", CueSourceType::Manual, CueAction::Noop);
        manager.add_trigger(t1).expect("add_trigger should succeed");
        manager.add_trigger(t2).expect("add_trigger should succeed");
        let cleaned = manager.cleanup_terminal();
        assert_eq!(cleaned, 1);
        assert_eq!(manager.count(), 1);
    }

    #[test]
    fn test_cue_trigger_manager_by_source() {
        let mut manager = CueTriggerManager::new(10);
        let t1 = CueTrigger::new("gpi1", "GPI 1", CueSourceType::Gpi, CueAction::Noop);
        let t2 = CueTrigger::new("tc1", "TC 1", CueSourceType::Timecode, CueAction::Noop);
        let t3 = CueTrigger::new("gpi2", "GPI 2", CueSourceType::Gpi, CueAction::Noop);
        manager.add_trigger(t1).expect("add_trigger should succeed");
        manager.add_trigger(t2).expect("add_trigger should succeed");
        manager.add_trigger(t3).expect("add_trigger should succeed");
        let gpi_triggers = manager.triggers_by_source(CueSourceType::Gpi);
        assert_eq!(gpi_triggers.len(), 2);
    }
}
