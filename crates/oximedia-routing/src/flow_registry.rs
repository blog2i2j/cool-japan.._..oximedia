#![allow(dead_code)]
//! NMOS IS-04 flow registry.
//!
//! Tracks the lifecycle of media flows as defined by NMOS IS-04.  Each flow
//! has a unique id, belongs to a parent source, carries format metadata, and
//! can be subscribed to by one or more receivers.
//!
//! # Design
//!
//! - [`Flow`] — immutable description of a media flow.
//! - [`FlowRegistry`] — mutable store that creates, updates, removes, and
//!   looks up flows; also manages subscriptions.
//! - [`FlowFormat`] — the essence type / format of the flow (video, audio, …).
//! - [`FlowSubscription`] — binding of a receiver to a flow.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// FlowFormat
// ---------------------------------------------------------------------------

/// Essence type and codec information for a flow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlowFormat {
    /// Raw or compressed video.
    Video {
        /// Codec name, e.g. `"raw"`, `"H.264"`, `"HEVC"`.
        codec: String,
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Interlace / progressive indicator.
        interlaced: bool,
    },
    /// Raw or compressed audio.
    Audio {
        /// Codec name, e.g. `"L24"`, `"AAC"`.
        codec: String,
        /// Sample rate in Hz.
        sample_rate: u32,
        /// Channel count.
        channels: u8,
        /// Bit depth.
        bit_depth: u8,
    },
    /// Ancillary / metadata essence.
    Data {
        /// Ancillary data type identifier, e.g. `"urn:x-nmos:format:data.sdp"`.
        data_type: String,
    },
    /// Multiplexed (multi-essence) flow.
    Mux,
}

impl fmt::Display for FlowFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Video {
                codec,
                width,
                height,
                interlaced,
            } => {
                let scan = if *interlaced { "i" } else { "p" };
                write!(f, "video/{codec} {width}x{height}{scan}")
            }
            Self::Audio {
                codec,
                sample_rate,
                channels,
                bit_depth,
            } => {
                write!(
                    f,
                    "audio/{codec} {sample_rate}Hz {channels}ch {bit_depth}bit"
                )
            }
            Self::Data { data_type } => write!(f, "data/{data_type}"),
            Self::Mux => write!(f, "mux"),
        }
    }
}

// ---------------------------------------------------------------------------
// FlowState
// ---------------------------------------------------------------------------

/// Operational state of a flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowState {
    /// Flow is active and producing essence.
    Active,
    /// Flow is defined but not currently producing essence.
    Inactive,
    /// Flow has been deprecated and may be removed.
    Deprecated,
}

impl fmt::Display for FlowState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Inactive => write!(f, "inactive"),
            Self::Deprecated => write!(f, "deprecated"),
        }
    }
}

// ---------------------------------------------------------------------------
// Flow
// ---------------------------------------------------------------------------

/// A media flow as described by NMOS IS-04.
#[derive(Debug, Clone)]
pub struct Flow {
    /// Unique flow identifier (UUID-style string).
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// Optional longer description.
    pub description: String,
    /// The source to which this flow belongs.
    pub source_id: String,
    /// Optional parent flow (e.g., for sub-flows).
    pub parent_id: Option<String>,
    /// Essence format of this flow.
    pub format: FlowFormat,
    /// Current operational state.
    pub state: FlowState,
    /// Arbitrary tags for filtering / grouping.
    pub tags: HashMap<String, String>,
    /// Grain rate numerator (frames/samples per second numerator).
    pub grain_rate_numerator: u32,
    /// Grain rate denominator.
    pub grain_rate_denominator: u32,
}

impl Flow {
    /// Creates a new flow with `Active` state and no tags.
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        source_id: impl Into<String>,
        format: FlowFormat,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            description: String::new(),
            source_id: source_id.into(),
            parent_id: None,
            format,
            state: FlowState::Active,
            tags: HashMap::new(),
            grain_rate_numerator: 1,
            grain_rate_denominator: 1,
        }
    }

    /// Sets the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Sets the parent flow id.
    pub fn with_parent(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_id = Some(parent_id.into());
        self
    }

    /// Sets the grain rate.
    pub fn with_grain_rate(mut self, numerator: u32, denominator: u32) -> Self {
        self.grain_rate_numerator = numerator;
        self.grain_rate_denominator = denominator;
        self
    }

    /// Adds a tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Returns the grain rate as a floating-point value.
    pub fn grain_rate_hz(&self) -> f64 {
        if self.grain_rate_denominator == 0 {
            0.0
        } else {
            self.grain_rate_numerator as f64 / self.grain_rate_denominator as f64
        }
    }

    /// Returns `true` if this flow is currently producing essence.
    pub fn is_active(&self) -> bool {
        self.state == FlowState::Active
    }
}

impl fmt::Display for Flow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Flow({}, '{}', src={}, fmt={}, state={})",
            self.id, self.label, self.source_id, self.format, self.state
        )
    }
}

// ---------------------------------------------------------------------------
// FlowSubscription
// ---------------------------------------------------------------------------

/// A binding between a receiver and a flow.
#[derive(Debug, Clone)]
pub struct FlowSubscription {
    /// Unique subscription id.
    pub id: String,
    /// The flow being subscribed to.
    pub flow_id: String,
    /// The receiver that subscribed.
    pub receiver_id: String,
    /// Whether this subscription is currently active.
    pub active: bool,
}

impl FlowSubscription {
    /// Creates a new active subscription.
    pub fn new(
        id: impl Into<String>,
        flow_id: impl Into<String>,
        receiver_id: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            flow_id: flow_id.into(),
            receiver_id: receiver_id.into(),
            active: true,
        }
    }
}

// ---------------------------------------------------------------------------
// FlowRegistryError
// ---------------------------------------------------------------------------

/// Errors returned by [`FlowRegistry`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlowRegistryError {
    /// A flow with the given id already exists.
    DuplicateFlowId(String),
    /// No flow exists with the given id.
    FlowNotFound(String),
    /// The specified source id is empty or invalid.
    InvalidSourceId,
    /// A subscription with the given id already exists.
    DuplicateSubscriptionId(String),
    /// No subscription exists with the given id.
    SubscriptionNotFound(String),
}

impl fmt::Display for FlowRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateFlowId(id) => write!(f, "flow id already registered: {id}"),
            Self::FlowNotFound(id) => write!(f, "flow not found: {id}"),
            Self::InvalidSourceId => write!(f, "source id must not be empty"),
            Self::DuplicateSubscriptionId(id) => {
                write!(f, "subscription id already registered: {id}")
            }
            Self::SubscriptionNotFound(id) => write!(f, "subscription not found: {id}"),
        }
    }
}

impl std::error::Error for FlowRegistryError {}

// ---------------------------------------------------------------------------
// FlowRegistry
// ---------------------------------------------------------------------------

/// Registry of NMOS IS-04 media flows.
///
/// Flows are keyed by their unique string id.  The registry also maintains a
/// set of subscriptions so callers can track which receivers are consuming
/// which flows.
#[derive(Debug, Default)]
pub struct FlowRegistry {
    flows: HashMap<String, Flow>,
    subscriptions: HashMap<String, FlowSubscription>,
    /// Index: source_id → set of flow ids belonging to that source.
    source_index: HashMap<String, HashSet<String>>,
    /// Index: flow_id → set of subscription ids attached to it.
    flow_subscriptions: HashMap<String, HashSet<String>>,
}

impl FlowRegistry {
    /// Creates an empty flow registry.
    pub fn new() -> Self {
        Self::default()
    }

    // -----------------------------------------------------------------------
    // Flow lifecycle
    // -----------------------------------------------------------------------

    /// Registers a new flow.  Returns an error if the id is already in use or
    /// the source id is empty.
    pub fn register_flow(&mut self, flow: Flow) -> Result<(), FlowRegistryError> {
        if flow.source_id.is_empty() {
            return Err(FlowRegistryError::InvalidSourceId);
        }
        if self.flows.contains_key(&flow.id) {
            return Err(FlowRegistryError::DuplicateFlowId(flow.id.clone()));
        }
        let flow_id = flow.id.clone();
        let source_id = flow.source_id.clone();
        self.flows.insert(flow_id.clone(), flow);
        self.source_index
            .entry(source_id)
            .or_default()
            .insert(flow_id.clone());
        self.flow_subscriptions.entry(flow_id).or_default();
        Ok(())
    }

    /// Updates an existing flow.  Returns an error if the flow is not found.
    pub fn update_flow(&mut self, flow: Flow) -> Result<(), FlowRegistryError> {
        if !self.flows.contains_key(&flow.id) {
            return Err(FlowRegistryError::FlowNotFound(flow.id.clone()));
        }
        let old_source = self
            .flows
            .get(&flow.id)
            .map(|f| f.source_id.clone())
            .unwrap_or_default();
        // Update source index if source changed
        if old_source != flow.source_id {
            if let Some(ids) = self.source_index.get_mut(&old_source) {
                ids.remove(&flow.id);
            }
            self.source_index
                .entry(flow.source_id.clone())
                .or_default()
                .insert(flow.id.clone());
        }
        self.flows.insert(flow.id.clone(), flow);
        Ok(())
    }

    /// Removes a flow and all its subscriptions.  Returns an error if not found.
    pub fn remove_flow(&mut self, flow_id: &str) -> Result<Flow, FlowRegistryError> {
        let flow = self
            .flows
            .remove(flow_id)
            .ok_or_else(|| FlowRegistryError::FlowNotFound(flow_id.to_string()))?;

        // Remove from source index
        if let Some(ids) = self.source_index.get_mut(&flow.source_id) {
            ids.remove(flow_id);
        }

        // Remove all subscriptions attached to this flow
        if let Some(sub_ids) = self.flow_subscriptions.remove(flow_id) {
            for sub_id in &sub_ids {
                self.subscriptions.remove(sub_id);
            }
        }

        Ok(flow)
    }

    /// Sets the state of a flow.
    pub fn set_flow_state(
        &mut self,
        flow_id: &str,
        state: FlowState,
    ) -> Result<(), FlowRegistryError> {
        let flow = self
            .flows
            .get_mut(flow_id)
            .ok_or_else(|| FlowRegistryError::FlowNotFound(flow_id.to_string()))?;
        flow.state = state;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Flow lookup
    // -----------------------------------------------------------------------

    /// Returns a reference to the flow with the given id.
    pub fn get_flow(&self, flow_id: &str) -> Option<&Flow> {
        self.flows.get(flow_id)
    }

    /// Returns all flows belonging to the given source.
    pub fn flows_for_source(&self, source_id: &str) -> Vec<&Flow> {
        let ids = match self.source_index.get(source_id) {
            Some(s) => s,
            None => return Vec::new(),
        };
        ids.iter().filter_map(|id| self.flows.get(id)).collect()
    }

    /// Returns all active flows.
    pub fn active_flows(&self) -> Vec<&Flow> {
        self.flows
            .values()
            .filter(|f| f.state == FlowState::Active)
            .collect()
    }

    /// Returns all flows matching the given format discriminant.
    pub fn flows_by_format_kind(&self, kind: FlowFormatKind) -> Vec<&Flow> {
        self.flows
            .values()
            .filter(|f| FlowFormatKind::from(&f.format) == kind)
            .collect()
    }

    /// Total number of registered flows.
    pub fn flow_count(&self) -> usize {
        self.flows.len()
    }

    // -----------------------------------------------------------------------
    // Subscriptions
    // -----------------------------------------------------------------------

    /// Adds a subscription to a flow.
    pub fn subscribe(&mut self, subscription: FlowSubscription) -> Result<(), FlowRegistryError> {
        if self.subscriptions.contains_key(&subscription.id) {
            return Err(FlowRegistryError::DuplicateSubscriptionId(
                subscription.id.clone(),
            ));
        }
        if !self.flows.contains_key(&subscription.flow_id) {
            return Err(FlowRegistryError::FlowNotFound(
                subscription.flow_id.clone(),
            ));
        }
        let sub_id = subscription.id.clone();
        let flow_id = subscription.flow_id.clone();
        self.subscriptions.insert(sub_id.clone(), subscription);
        self.flow_subscriptions
            .entry(flow_id)
            .or_default()
            .insert(sub_id);
        Ok(())
    }

    /// Removes a subscription.
    pub fn unsubscribe(&mut self, subscription_id: &str) -> Result<(), FlowRegistryError> {
        let sub = self
            .subscriptions
            .remove(subscription_id)
            .ok_or_else(|| FlowRegistryError::SubscriptionNotFound(subscription_id.to_string()))?;

        if let Some(ids) = self.flow_subscriptions.get_mut(&sub.flow_id) {
            ids.remove(subscription_id);
        }

        Ok(())
    }

    /// Returns all subscriptions for the given flow.
    pub fn subscriptions_for_flow(&self, flow_id: &str) -> Vec<&FlowSubscription> {
        let ids = match self.flow_subscriptions.get(flow_id) {
            Some(s) => s,
            None => return Vec::new(),
        };
        ids.iter()
            .filter_map(|id| self.subscriptions.get(id))
            .collect()
    }

    /// Returns the number of active subscriptions across all flows.
    pub fn subscription_count(&self) -> usize {
        self.subscriptions.len()
    }

    /// Returns a reference to a subscription by id.
    pub fn get_subscription(&self, subscription_id: &str) -> Option<&FlowSubscription> {
        self.subscriptions.get(subscription_id)
    }
}

// ---------------------------------------------------------------------------
// FlowFormatKind
// ---------------------------------------------------------------------------

/// Discriminant for matching flows by essence category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowFormatKind {
    /// Video essence.
    Video,
    /// Audio essence.
    Audio,
    /// Data / ancillary essence.
    Data,
    /// Multiplexed multi-essence.
    Mux,
}

impl From<&FlowFormat> for FlowFormatKind {
    fn from(fmt: &FlowFormat) -> Self {
        match fmt {
            FlowFormat::Video { .. } => Self::Video,
            FlowFormat::Audio { .. } => Self::Audio,
            FlowFormat::Data { .. } => Self::Data,
            FlowFormat::Mux => Self::Mux,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn make_video_flow(id: &str, source_id: &str) -> Flow {
        Flow::new(
            id,
            "Test Video Flow",
            source_id,
            FlowFormat::Video {
                codec: "raw".to_string(),
                width: 1920,
                height: 1080,
                interlaced: false,
            },
        )
    }

    fn make_audio_flow(id: &str, source_id: &str) -> Flow {
        Flow::new(
            id,
            "Test Audio Flow",
            source_id,
            FlowFormat::Audio {
                codec: "L24".to_string(),
                sample_rate: 48000,
                channels: 2,
                bit_depth: 24,
            },
        )
    }

    #[test]
    fn test_register_and_retrieve_flow() -> TestResult {
        let mut registry = FlowRegistry::new();
        let flow = make_video_flow("flow-001", "src-001");
        registry.register_flow(flow)?;
        let retrieved = registry
            .get_flow("flow-001")
            .ok_or("flow-001 not found in registry")?;
        assert_eq!(retrieved.id, "flow-001");
        Ok(())
    }

    #[test]
    fn test_register_duplicate_id_returns_error() -> TestResult {
        let mut registry = FlowRegistry::new();
        registry.register_flow(make_video_flow("flow-dup", "src-1"))?;
        let result = registry.register_flow(make_video_flow("flow-dup", "src-2"));
        assert_eq!(
            result,
            Err(FlowRegistryError::DuplicateFlowId("flow-dup".to_string()))
        );
        Ok(())
    }

    #[test]
    fn test_register_empty_source_id_returns_error() {
        let mut registry = FlowRegistry::new();
        let flow = make_video_flow("flow-bad-src", "");
        let result = registry.register_flow(flow);
        assert_eq!(result, Err(FlowRegistryError::InvalidSourceId));
    }

    #[test]
    fn test_remove_flow() -> TestResult {
        let mut registry = FlowRegistry::new();
        registry.register_flow(make_video_flow("flow-rm", "src-1"))?;
        let removed = registry.remove_flow("flow-rm");
        assert!(removed.is_ok());
        assert!(registry.get_flow("flow-rm").is_none());
        Ok(())
    }

    #[test]
    fn test_remove_nonexistent_flow_returns_error() {
        let mut registry = FlowRegistry::new();
        let result = registry.remove_flow("ghost");
        assert!(matches!(result, Err(FlowRegistryError::FlowNotFound(_))));
    }

    #[test]
    fn test_flows_for_source() -> TestResult {
        let mut registry = FlowRegistry::new();
        registry.register_flow(make_video_flow("flow-v", "src-A"))?;
        registry.register_flow(make_audio_flow("flow-a", "src-A"))?;
        registry.register_flow(make_video_flow("flow-other", "src-B"))?;
        let flows = registry.flows_for_source("src-A");
        assert_eq!(flows.len(), 2);
        Ok(())
    }

    #[test]
    fn test_active_flows_filter() -> TestResult {
        let mut registry = FlowRegistry::new();
        registry.register_flow(make_video_flow("flow-act", "src-1"))?;
        registry.register_flow(make_audio_flow("flow-inact", "src-1"))?;
        registry.set_flow_state("flow-inact", FlowState::Inactive)?;
        let active = registry.active_flows();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "flow-act");
        Ok(())
    }

    #[test]
    fn test_set_flow_state() -> TestResult {
        let mut registry = FlowRegistry::new();
        registry.register_flow(make_video_flow("flow-s", "src-1"))?;
        registry.set_flow_state("flow-s", FlowState::Deprecated)?;
        let state = registry
            .get_flow("flow-s")
            .ok_or("flow-s not found after state change")?
            .state;
        assert_eq!(state, FlowState::Deprecated);
        Ok(())
    }

    #[test]
    fn test_subscribe_and_unsubscribe() -> TestResult {
        let mut registry = FlowRegistry::new();
        registry.register_flow(make_video_flow("flow-sub", "src-1"))?;
        let sub = FlowSubscription::new("sub-001", "flow-sub", "recv-001");
        registry.subscribe(sub)?;
        assert_eq!(registry.subscription_count(), 1);

        registry.unsubscribe("sub-001")?;
        assert_eq!(registry.subscription_count(), 0);
        Ok(())
    }

    #[test]
    fn test_subscribe_to_nonexistent_flow_returns_error() {
        let mut registry = FlowRegistry::new();
        let sub = FlowSubscription::new("sub-x", "no-flow", "recv-x");
        let result = registry.subscribe(sub);
        assert!(matches!(result, Err(FlowRegistryError::FlowNotFound(_))));
    }

    #[test]
    fn test_subscriptions_removed_on_flow_removal() -> TestResult {
        let mut registry = FlowRegistry::new();
        registry.register_flow(make_video_flow("flow-cascade", "src-1"))?;
        let sub = FlowSubscription::new("sub-cascade", "flow-cascade", "recv-1");
        registry.subscribe(sub)?;
        assert_eq!(registry.subscription_count(), 1);
        registry.remove_flow("flow-cascade")?;
        assert_eq!(registry.subscription_count(), 0);
        Ok(())
    }

    #[test]
    fn test_flow_format_kind_from_format() {
        let video_fmt = FlowFormat::Video {
            codec: "raw".to_string(),
            width: 1920,
            height: 1080,
            interlaced: false,
        };
        let audio_fmt = FlowFormat::Audio {
            codec: "L24".to_string(),
            sample_rate: 48000,
            channels: 2,
            bit_depth: 24,
        };
        assert_eq!(FlowFormatKind::from(&video_fmt), FlowFormatKind::Video);
        assert_eq!(FlowFormatKind::from(&audio_fmt), FlowFormatKind::Audio);
        assert_eq!(FlowFormatKind::from(&FlowFormat::Mux), FlowFormatKind::Mux);
    }

    #[test]
    fn test_flow_grain_rate() {
        let flow = make_video_flow("flow-gr", "src-1").with_grain_rate(30000, 1001);
        let hz = flow.grain_rate_hz();
        assert!((hz - 29.97).abs() < 0.01, "expected ~29.97 Hz, got {hz}");
    }

    #[test]
    fn test_flow_display() {
        let flow = make_audio_flow("flow-disp", "src-disp");
        let s = format!("{flow}");
        assert!(s.contains("flow-disp"));
        assert!(s.contains("src-disp"));
        assert!(s.contains("audio"));
    }

    #[test]
    fn test_flows_by_format_kind() -> TestResult {
        let mut registry = FlowRegistry::new();
        registry.register_flow(make_video_flow("fv1", "src-1"))?;
        registry.register_flow(make_video_flow("fv2", "src-1"))?;
        registry.register_flow(make_audio_flow("fa1", "src-1"))?;
        let videos = registry.flows_by_format_kind(FlowFormatKind::Video);
        assert_eq!(videos.len(), 2);
        let audios = registry.flows_by_format_kind(FlowFormatKind::Audio);
        assert_eq!(audios.len(), 1);
        Ok(())
    }
}
