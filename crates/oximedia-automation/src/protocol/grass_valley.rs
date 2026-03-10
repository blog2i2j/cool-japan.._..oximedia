//! Grass Valley (ENPS / Ignite) automation protocol simulation.
//!
//! Command format:
//! ```text
//! VERB NOUN key1=val1 key2=val2 …
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

/// A parsed Grass Valley automation command.
#[derive(Debug, Clone)]
pub struct GvgCommand {
    /// Action verb (e.g. `PLAY`, `STOP`, `LOAD`)
    pub verb: String,
    /// Target noun / resource (e.g. `CLIP`, `CHANNEL`)
    pub noun: String,
    /// Key-value parameter pairs
    pub params: HashMap<String, String>,
}

impl GvgCommand {
    /// Create a new `GvgCommand` with no parameters.
    pub fn new(verb: impl Into<String>, noun: impl Into<String>) -> Self {
        Self {
            verb: verb.into(),
            noun: noun.into(),
            params: HashMap::new(),
        }
    }

    /// Add a parameter to the command.
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }
}

/// Parser for the Grass Valley text protocol.
pub struct GvgParser;

impl GvgParser {
    /// Parse a single protocol line into a [`GvgCommand`].
    ///
    /// Expected format: `VERB NOUN [key=value …]`
    ///
    /// # Errors
    /// Returns `Err` if the line has fewer than two tokens.
    pub fn parse(line: &str) -> Result<GvgCommand, String> {
        let mut tokens = line.split_whitespace();

        let verb = tokens
            .next()
            .ok_or_else(|| "Missing verb".to_string())?
            .to_string();

        let noun = tokens
            .next()
            .ok_or_else(|| "Missing noun".to_string())?
            .to_string();

        let mut params = HashMap::new();
        for token in tokens {
            if let Some((k, v)) = token.split_once('=') {
                params.insert(k.to_string(), v.to_string());
            }
            // Tokens without '=' are silently ignored (forward compatibility).
        }

        Ok(GvgCommand { verb, noun, params })
    }
}

/// Formatter for the Grass Valley text protocol.
pub struct GvgProtocol;

impl GvgProtocol {
    /// Format a [`GvgCommand`] into its wire-format string.
    #[must_use]
    pub fn format_command(cmd: &GvgCommand) -> String {
        let mut parts = vec![cmd.verb.clone(), cmd.noun.clone()];

        // Sort params for deterministic output.
        let mut sorted: Vec<(&String, &String)> = cmd.params.iter().collect();
        sorted.sort_by_key(|(k, _)| k.as_str());

        for (k, v) in sorted {
            parts.push(format!("{k}={v}"));
        }

        parts.join(" ")
    }
}

/// A single GVG device event.
#[derive(Debug, Clone)]
pub struct GvgEvent {
    /// Unix timestamp (seconds)
    pub timestamp: u64,
    /// Device identifier
    pub device_id: String,
    /// Event type string (e.g. `CLIP_LOADED`, `ERROR`)
    pub event_type: String,
    /// Human-readable description
    pub description: String,
}

impl GvgEvent {
    /// Create a new `GvgEvent`.
    pub fn new(
        timestamp: u64,
        device_id: impl Into<String>,
        event_type: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            timestamp,
            device_id: device_id.into(),
            event_type: event_type.into(),
            description: description.into(),
        }
    }
}

/// Append-only event log for Grass Valley device events.
#[derive(Debug, Default)]
pub struct GvgEventLog {
    /// Ordered list of events (oldest first)
    pub events: Vec<GvgEvent>,
}

impl GvgEventLog {
    /// Create an empty event log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an event to the log.
    pub fn push(&mut self, event: GvgEvent) {
        self.events.push(event);
    }

    /// Return all events matching a given device ID.
    pub fn events_for_device(&self, device_id: &str) -> Vec<&GvgEvent> {
        self.events
            .iter()
            .filter(|e| e.device_id == device_id)
            .collect()
    }

    /// Return the total number of recorded events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Return `true` if no events have been recorded.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_command() {
        let cmd = GvgParser::parse("PLAY CLIP").expect("parse should succeed");
        assert_eq!(cmd.verb, "PLAY");
        assert_eq!(cmd.noun, "CLIP");
        assert!(cmd.params.is_empty());
    }

    #[test]
    fn test_parse_command_with_params() {
        let cmd = GvgParser::parse("LOAD CLIP id=clip001 channel=1").expect("parse should succeed");
        assert_eq!(cmd.verb, "LOAD");
        assert_eq!(cmd.noun, "CLIP");
        assert_eq!(cmd.params.get("id").map(String::as_str), Some("clip001"));
        assert_eq!(cmd.params.get("channel").map(String::as_str), Some("1"));
    }

    #[test]
    fn test_parse_missing_verb_error() {
        let result = GvgParser::parse("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_missing_noun_error() {
        let result = GvgParser::parse("PLAY");
        assert!(result.is_err());
    }

    #[test]
    fn test_format_command_no_params() {
        let cmd = GvgCommand::new("STOP", "CHANNEL");
        let line = GvgProtocol::format_command(&cmd);
        assert_eq!(line, "STOP CHANNEL");
    }

    #[test]
    fn test_format_command_with_params() {
        let cmd = GvgCommand::new("CUE", "CLIP").with_param("tc", "01:00:00:00");
        let line = GvgProtocol::format_command(&cmd);
        assert_eq!(line, "CUE CLIP tc=01:00:00:00");
    }

    #[test]
    fn test_parse_format_roundtrip() {
        let original = "EJECT DECK slot=3";
        let cmd = GvgParser::parse(original).expect("parse should succeed");
        let formatted = GvgProtocol::format_command(&cmd);
        // Re-parse and verify semantic equivalence
        let reparsed = GvgParser::parse(&formatted).expect("parse should succeed");
        assert_eq!(reparsed.verb, cmd.verb);
        assert_eq!(reparsed.noun, cmd.noun);
        assert_eq!(reparsed.params, cmd.params);
    }

    #[test]
    fn test_event_log_push_and_count() {
        let mut log = GvgEventLog::new();
        log.push(GvgEvent::new(1000, "dev-1", "CLIP_LOADED", "Clip loaded"));
        log.push(GvgEvent::new(2000, "dev-2", "ERROR", "Disk error"));
        assert_eq!(log.len(), 2);
        assert!(!log.is_empty());
    }

    #[test]
    fn test_event_log_filter_by_device() {
        let mut log = GvgEventLog::new();
        log.push(GvgEvent::new(1, "dev-A", "START", ""));
        log.push(GvgEvent::new(2, "dev-B", "STOP", ""));
        log.push(GvgEvent::new(3, "dev-A", "STOP", ""));

        let events = log.events_for_device("dev-A");
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_event_log_empty() {
        let log = GvgEventLog::new();
        assert!(log.is_empty());
    }
}
