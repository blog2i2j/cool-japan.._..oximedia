//! Automation system for mixer parameters.
//!
//! Provides comprehensive automation with multiple modes, curves, and keyframe editing.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use uuid::Uuid;

use crate::{BusId, ChannelId, EffectId};

/// Unique automation identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AutomationId(pub Uuid);

/// Automation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AutomationMode {
    /// Play back recorded automation.
    #[default]
    Read,
    /// Record all parameter changes.
    Write,
    /// Record only when touching controls.
    Touch,
    /// Continue last value after release (like Touch but latches).
    Latch,
    /// Apply relative changes to existing automation.
    Trim,
    /// Automation disabled.
    Off,
}

/// Automation parameter type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AutomationParameter {
    /// Channel gain/volume.
    ChannelGain(ChannelId),
    /// Channel pan.
    ChannelPan(ChannelId),
    /// Channel mute.
    ChannelMute(ChannelId),
    /// Channel solo.
    ChannelSolo(ChannelId),
    /// Channel send level.
    ChannelSend {
        /// Channel ID.
        channel: ChannelId,
        /// Send slot number.
        send: usize,
    },
    /// Effect parameter.
    EffectParameter {
        /// Channel ID.
        channel: ChannelId,
        /// Effect ID.
        effect: EffectId,
        /// Parameter name.
        parameter: String,
    },
    /// Bus gain.
    BusGain(BusId),
    /// Bus mute.
    BusMute(BusId),
    /// Master gain.
    MasterGain,
    /// Master mute.
    MasterMute,
}

/// Automation curve type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AutomationCurve {
    /// Linear interpolation.
    #[default]
    Linear,
    /// Exponential curve (for gain/volume).
    Exponential,
    /// S-curve (smooth acceleration/deceleration).
    SCurve,
    /// Step (no interpolation).
    Step,
    /// Logarithmic curve.
    Logarithmic,
    /// Cubic Bezier curve.
    Bezier,
}

/// Automation point (keyframe).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationPoint {
    /// Time in samples.
    pub time_samples: u64,
    /// Parameter value.
    pub value: f32,
    /// Curve type to next point.
    pub curve: AutomationCurve,
    /// Curve tension (for Bezier, S-curve).
    pub tension: f32,
}

impl AutomationPoint {
    /// Create a new automation point.
    #[must_use]
    pub fn new(time_samples: u64, value: f32) -> Self {
        Self {
            time_samples,
            value,
            curve: AutomationCurve::Linear,
            tension: 0.5,
        }
    }

    /// Create with custom curve.
    #[must_use]
    pub fn with_curve(time_samples: u64, value: f32, curve: AutomationCurve) -> Self {
        Self {
            time_samples,
            value,
            curve,
            tension: 0.5,
        }
    }
}

/// Automation lane for a single parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationLane {
    /// Lane ID.
    pub id: AutomationId,
    /// Parameter being automated.
    pub parameter: AutomationParameter,
    /// Automation mode.
    pub mode: AutomationMode,
    /// Automation points (sorted by time).
    points: BTreeMap<u64, AutomationPoint>,
    /// Current value (when not reading automation).
    pub current_value: f32,
    /// Default value.
    pub default_value: f32,
    /// Lane is enabled.
    pub enabled: bool,
    /// Currently touching (for Touch mode).
    pub touching: bool,
    /// Last touch time.
    last_touch_time: Option<u64>,
}

impl AutomationLane {
    /// Create a new automation lane.
    #[must_use]
    pub fn new(parameter: AutomationParameter, default_value: f32) -> Self {
        Self {
            id: AutomationId(Uuid::new_v4()),
            parameter,
            mode: AutomationMode::Read,
            points: BTreeMap::new(),
            current_value: default_value,
            default_value,
            enabled: true,
            touching: false,
            last_touch_time: None,
        }
    }

    /// Add automation point.
    pub fn add_point(&mut self, point: AutomationPoint) {
        self.points.insert(point.time_samples, point);
    }

    /// Remove automation point at time.
    pub fn remove_point(&mut self, time_samples: u64) -> Option<AutomationPoint> {
        self.points.remove(&time_samples)
    }

    /// Remove all points in time range.
    pub fn remove_range(&mut self, start_samples: u64, end_samples: u64) {
        self.points
            .retain(|&time, _| time < start_samples || time > end_samples);
    }

    /// Get all points.
    #[must_use]
    pub fn points(&self) -> &BTreeMap<u64, AutomationPoint> {
        &self.points
    }

    /// Get mutable points.
    #[must_use]
    pub fn points_mut(&mut self) -> &mut BTreeMap<u64, AutomationPoint> {
        &mut self.points
    }

    /// Clear all points.
    pub fn clear(&mut self) {
        self.points.clear();
    }

    /// Get value at specific time with interpolation.
    #[must_use]
    pub fn get_value_at(&self, time_samples: u64) -> f32 {
        if !self.enabled {
            return self.current_value;
        }

        match self.mode {
            AutomationMode::Read => self.interpolate_value(time_samples),
            _ => self.current_value, // Off/Write/Touch/Latch/Trim use current_value
        }
    }

    /// Interpolate value between points.
    fn interpolate_value(&self, time_samples: u64) -> f32 {
        if self.points.is_empty() {
            return self.default_value;
        }

        // Find surrounding points
        let mut before: Option<&AutomationPoint> = None;
        let mut after: Option<&AutomationPoint> = None;

        for point in self.points.values() {
            if point.time_samples <= time_samples {
                before = Some(point);
            } else if after.is_none() {
                after = Some(point);
                break;
            }
        }

        match (before, after) {
            (Some(p1), Some(p2)) => {
                // Interpolate between points
                #[allow(clippy::cast_precision_loss)]
                let t = (time_samples - p1.time_samples) as f32
                    / (p2.time_samples - p1.time_samples) as f32;
                self.interpolate_curve(p1.value, p2.value, t, p1.curve, p1.tension)
            }
            (Some(p), None) | (None, Some(p)) => p.value, // After/before point
            (None, None) => self.default_value,
        }
    }

    /// Interpolate between two values using specified curve.
    #[allow(clippy::too_many_arguments)]
    fn interpolate_curve(
        &self,
        v1: f32,
        v2: f32,
        t: f32,
        curve: AutomationCurve,
        tension: f32,
    ) -> f32 {
        match curve {
            AutomationCurve::Linear => v1 + (v2 - v1) * t,
            AutomationCurve::Exponential => {
                if v1 <= 0.0 || v2 <= 0.0 {
                    v1 + (v2 - v1) * t
                } else {
                    v1 * (v2 / v1).powf(t)
                }
            }
            AutomationCurve::SCurve => {
                let smooth_t = self.smooth_step(t);
                v1 + (v2 - v1) * smooth_t
            }
            AutomationCurve::Step => {
                if t < 1.0 {
                    v1
                } else {
                    v2
                }
            }
            AutomationCurve::Logarithmic => {
                let log_t = if t <= 0.0 {
                    0.0
                } else if t >= 1.0 {
                    1.0
                } else {
                    t.ln() / (1.0_f32).ln()
                };
                v1 + (v2 - v1) * log_t
            }
            AutomationCurve::Bezier => {
                let bezier_t = self.cubic_bezier(t, tension);
                v1 + (v2 - v1) * bezier_t
            }
        }
    }

    /// Smooth step function (S-curve).
    #[allow(clippy::unused_self)]
    fn smooth_step(&self, t: f32) -> f32 {
        t * t * (3.0 - 2.0 * t)
    }

    /// Cubic Bezier interpolation.
    #[allow(clippy::unused_self)]
    fn cubic_bezier(&self, t: f32, tension: f32) -> f32 {
        let p1 = tension;
        let p2 = 1.0 - tension;
        3.0 * (1.0 - t).powi(2) * t * p1 + 3.0 * (1.0 - t) * t.powi(2) * p2 + t.powi(3)
    }

    /// Set value at current time (for Write/Touch/Latch modes).
    pub fn set_value(&mut self, value: f32, time_samples: u64) {
        self.current_value = value;

        match self.mode {
            AutomationMode::Write => {
                // Always write points
                self.add_point(AutomationPoint::new(time_samples, value));
            }
            AutomationMode::Touch => {
                if self.touching {
                    self.add_point(AutomationPoint::new(time_samples, value));
                    self.last_touch_time = Some(time_samples);
                }
            }
            AutomationMode::Latch => {
                if self.touching {
                    self.add_point(AutomationPoint::new(time_samples, value));
                    self.last_touch_time = Some(time_samples);
                }
                // In Latch mode, continue writing even after release
                else if self.last_touch_time.is_some() {
                    self.add_point(AutomationPoint::new(time_samples, value));
                }
            }
            AutomationMode::Trim => {
                // Apply relative offset to existing automation
                if self.touching {
                    let existing_value = self.interpolate_value(time_samples);
                    let offset = value - existing_value;
                    self.apply_trim_offset(offset);
                }
            }
            _ => {}
        }
    }

    /// Apply trim offset to all points.
    fn apply_trim_offset(&mut self, offset: f32) {
        for point in self.points.values_mut() {
            point.value += offset;
        }
    }

    /// Start touching parameter.
    pub fn start_touch(&mut self, time_samples: u64) {
        self.touching = true;
        self.last_touch_time = Some(time_samples);
    }

    /// Stop touching parameter.
    pub fn stop_touch(&mut self, _time_samples: u64) {
        self.touching = false;
        // In Latch mode, keep last_touch_time to continue latching
        if self.mode != AutomationMode::Latch {
            self.last_touch_time = None;
        }
    }

    /// Thin automation points (remove redundant points).
    pub fn thin_automation(&mut self, threshold: f32) {
        if self.points.len() < 3 {
            return;
        }

        let mut to_remove = Vec::new();
        let points_vec: Vec<_> = self.points.iter().collect();

        for i in 1..points_vec.len() - 1 {
            let (_, p_prev) = points_vec[i - 1];
            let (time, p_curr) = points_vec[i];
            let (_, p_next) = points_vec[i + 1];

            // Calculate interpolated value
            #[allow(clippy::cast_precision_loss)]
            let t = (p_curr.time_samples - p_prev.time_samples) as f32
                / (p_next.time_samples - p_prev.time_samples) as f32;
            let interpolated = p_prev.value + (p_next.value - p_prev.value) * t;

            // Remove if difference is below threshold
            if (p_curr.value - interpolated).abs() < threshold {
                to_remove.push(*time);
            }
        }

        for time in to_remove {
            self.points.remove(&time);
        }
    }

    /// Get point count.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.points.len()
    }
}

/// Automation data container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationData {
    /// All automation lanes.
    lanes: Vec<AutomationLane>,
    /// Global automation mode.
    pub global_mode: AutomationMode,
    /// Automation enabled globally.
    pub enabled: bool,
}

impl AutomationData {
    /// Create new automation data.
    #[must_use]
    pub fn new() -> Self {
        Self {
            lanes: Vec::new(),
            global_mode: AutomationMode::Read,
            enabled: true,
        }
    }

    /// Add automation lane.
    pub fn add_lane(&mut self, lane: AutomationLane) {
        self.lanes.push(lane);
    }

    /// Get lane by parameter.
    #[must_use]
    pub fn get_lane(&self, parameter: &AutomationParameter) -> Option<&AutomationLane> {
        self.lanes.iter().find(|lane| &lane.parameter == parameter)
    }

    /// Get mutable lane by parameter.
    #[must_use]
    pub fn get_lane_mut(&mut self, parameter: &AutomationParameter) -> Option<&mut AutomationLane> {
        self.lanes
            .iter_mut()
            .find(|lane| &lane.parameter == parameter)
    }

    /// Get or create lane.
    pub fn get_or_create_lane(
        &mut self,
        parameter: &AutomationParameter,
        default_value: f32,
    ) -> &mut AutomationLane {
        if !self.lanes.iter().any(|lane| lane.parameter == *parameter) {
            let lane = AutomationLane::new(parameter.clone(), default_value);
            self.lanes.push(lane);
        }
        // Safety: we just ensured the lane exists above.
        self.lanes
            .iter_mut()
            .find(|lane| &lane.parameter == parameter)
            .expect("lane was just inserted")
    }

    /// Remove lane by parameter.
    pub fn remove_lane(&mut self, parameter: &AutomationParameter) {
        self.lanes.retain(|lane| &lane.parameter != parameter);
    }

    /// Get all lanes.
    #[must_use]
    pub fn lanes(&self) -> &[AutomationLane] {
        &self.lanes
    }

    /// Get mutable lanes.
    #[must_use]
    pub fn lanes_mut(&mut self) -> &mut Vec<AutomationLane> {
        &mut self.lanes
    }

    /// Clear all automation.
    pub fn clear(&mut self) {
        self.lanes.clear();
    }

    /// Process automation at current time.
    pub fn process(&mut self, time_samples: u64) {
        for lane in &mut self.lanes {
            if lane.enabled && self.enabled && lane.mode == AutomationMode::Read {
                lane.current_value = lane.get_value_at(time_samples);
            }
        }
    }

    /// Get automation value for parameter.
    #[must_use]
    pub fn get_value(&self, parameter: &AutomationParameter, time_samples: u64) -> Option<f32> {
        self.get_lane(parameter)
            .map(|lane| lane.get_value_at(time_samples))
    }

    /// Set automation value.
    pub fn set_value(
        &mut self,
        parameter: &AutomationParameter,
        value: f32,
        time_samples: u64,
        default_value: f32,
    ) {
        let lane = self.get_or_create_lane(parameter, default_value);
        lane.set_value(value, time_samples);
    }
}

impl Default for AutomationData {
    fn default() -> Self {
        Self::new()
    }
}

/// Automation snapshot (scene).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationSnapshot {
    /// Snapshot ID.
    pub id: Uuid,
    /// Snapshot name.
    pub name: String,
    /// Parameter values.
    pub values: BTreeMap<AutomationParameter, f32>,
    /// Snapshot time (optional).
    pub time_samples: Option<u64>,
}

impl AutomationSnapshot {
    /// Create new snapshot.
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            values: BTreeMap::new(),
            time_samples: None,
        }
    }

    /// Set parameter value.
    pub fn set_value(&mut self, parameter: AutomationParameter, value: f32) {
        self.values.insert(parameter, value);
    }

    /// Get parameter value.
    #[must_use]
    pub fn get_value(&self, parameter: &AutomationParameter) -> Option<f32> {
        self.values.get(parameter).copied()
    }

    /// Apply snapshot to automation data.
    pub fn apply_to(&self, automation: &mut AutomationData, time_samples: u64) {
        for (parameter, &value) in &self.values {
            automation.set_value(parameter, value, time_samples, value);
        }
    }
}

/// Automation snapshot manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotManager {
    /// All snapshots.
    snapshots: Vec<AutomationSnapshot>,
    /// Currently active snapshot.
    active_snapshot: Option<Uuid>,
}

impl SnapshotManager {
    /// Create new snapshot manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            active_snapshot: None,
        }
    }

    /// Add snapshot.
    pub fn add_snapshot(&mut self, snapshot: AutomationSnapshot) -> Uuid {
        let id = snapshot.id;
        self.snapshots.push(snapshot);
        id
    }

    /// Get snapshot by ID.
    #[must_use]
    pub fn get_snapshot(&self, id: Uuid) -> Option<&AutomationSnapshot> {
        self.snapshots.iter().find(|s| s.id == id)
    }

    /// Get mutable snapshot by ID.
    #[must_use]
    pub fn get_snapshot_mut(&mut self, id: Uuid) -> Option<&mut AutomationSnapshot> {
        self.snapshots.iter_mut().find(|s| s.id == id)
    }

    /// Remove snapshot.
    pub fn remove_snapshot(&mut self, id: Uuid) {
        self.snapshots.retain(|s| s.id != id);
        if self.active_snapshot == Some(id) {
            self.active_snapshot = None;
        }
    }

    /// Get all snapshots.
    #[must_use]
    pub fn snapshots(&self) -> &[AutomationSnapshot] {
        &self.snapshots
    }

    /// Recall snapshot.
    pub fn recall_snapshot(
        &mut self,
        id: Uuid,
        automation: &mut AutomationData,
        time_samples: u64,
    ) {
        if let Some(snapshot) = self.get_snapshot(id) {
            snapshot.apply_to(automation, time_samples);
            self.active_snapshot = Some(id);
        }
    }

    /// Get active snapshot ID.
    #[must_use]
    pub fn active_snapshot(&self) -> Option<Uuid> {
        self.active_snapshot
    }

    /// Create snapshot from current automation state.
    pub fn capture_snapshot(
        &mut self,
        name: String,
        automation: &AutomationData,
        time_samples: u64,
    ) -> Uuid {
        let mut snapshot = AutomationSnapshot::new(name);
        snapshot.time_samples = Some(time_samples);

        for lane in automation.lanes() {
            snapshot.set_value(lane.parameter.clone(), lane.current_value);
        }

        self.add_snapshot(snapshot)
    }
}

impl Default for SnapshotManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Tempo-based automation time conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TempoMap {
    /// Tempo changes (`time_samples` -> BPM).
    tempo_changes: BTreeMap<u64, f64>,
    /// Default tempo (BPM).
    default_tempo: f64,
}

impl TempoMap {
    /// Create new tempo map.
    #[must_use]
    pub fn new(default_tempo: f64) -> Self {
        Self {
            tempo_changes: BTreeMap::new(),
            default_tempo,
        }
    }

    /// Add tempo change.
    pub fn add_tempo_change(&mut self, time_samples: u64, bpm: f64) {
        self.tempo_changes.insert(time_samples, bpm);
    }

    /// Get tempo at time.
    #[must_use]
    pub fn get_tempo_at(&self, time_samples: u64) -> f64 {
        self.tempo_changes
            .range(..=time_samples)
            .next_back()
            .map_or(self.default_tempo, |(_, &bpm)| bpm)
    }

    /// Convert musical time to samples.
    #[must_use]
    pub fn musical_to_samples(&self, beats: f64, sample_rate: u32) -> u64 {
        let tempo = self.default_tempo;
        let seconds = beats * 60.0 / tempo;
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let result = (seconds * f64::from(sample_rate)) as u64;
        result
    }

    /// Convert samples to musical time.
    #[must_use]
    pub fn samples_to_musical(&self, time_samples: u64, sample_rate: u32) -> f64 {
        let tempo = self.get_tempo_at(time_samples);
        #[allow(clippy::cast_precision_loss)]
        let seconds = time_samples as f64 / f64::from(sample_rate);
        seconds * tempo / 60.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_automation_point() {
        let point = AutomationPoint::new(1000, 0.5);
        assert_eq!(point.time_samples, 1000);
        assert_eq!(point.value, 0.5);
    }

    #[test]
    fn test_automation_lane() {
        let param = AutomationParameter::MasterGain;
        let mut lane = AutomationLane::new(param, 1.0);

        lane.add_point(AutomationPoint::new(0, 0.5));
        lane.add_point(AutomationPoint::new(1000, 0.8));

        assert_eq!(lane.point_count(), 2);

        // Test interpolation
        let value = lane.get_value_at(500);
        assert!((value - 0.65).abs() < 0.01); // Linear interpolation
    }

    #[test]
    fn test_automation_curves() {
        let param = AutomationParameter::MasterGain;
        let mut lane = AutomationLane::new(param, 1.0);

        let mut p1 = AutomationPoint::new(0, 0.0);
        p1.curve = AutomationCurve::Step;
        lane.add_point(p1);
        lane.add_point(AutomationPoint::new(1000, 1.0));

        // Step curve should keep first value
        let value = lane.get_value_at(500);
        assert!((value - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_automation_data() {
        let mut data = AutomationData::new();
        let param = AutomationParameter::MasterGain;

        // Set lane to Write mode so it records automation
        let lane = data.get_or_create_lane(&param, 1.0);
        lane.mode = AutomationMode::Write;

        data.set_value(&param, 0.8, 1000, 1.0);

        // Switch to Read mode to read back automation
        let lane = data.get_lane_mut(&param).expect("lane should be valid");
        lane.mode = AutomationMode::Read;

        let value = data.get_value(&param, 1000);
        assert_eq!(value, Some(0.8));
    }

    #[test]
    fn test_snapshot() {
        let mut snapshot = AutomationSnapshot::new("Scene 1".to_string());
        let param = AutomationParameter::MasterGain;

        snapshot.set_value(param.clone(), 0.7);
        assert_eq!(snapshot.get_value(&param), Some(0.7));
    }

    #[test]
    fn test_snapshot_manager() {
        let mut manager = SnapshotManager::new();
        let snapshot = AutomationSnapshot::new("Test".to_string());
        let id = manager.add_snapshot(snapshot);

        assert!(manager.get_snapshot(id).is_some());

        manager.remove_snapshot(id);
        assert!(manager.get_snapshot(id).is_none());
    }

    #[test]
    fn test_tempo_map() {
        let tempo_map = TempoMap::new(120.0);

        assert_eq!(tempo_map.get_tempo_at(0), 120.0);

        let samples = tempo_map.musical_to_samples(4.0, 48000); // 4 beats at 120 BPM
        assert_eq!(samples, 96000); // 2 seconds at 48kHz

        let beats = tempo_map.samples_to_musical(96000, 48000);
        assert!((beats - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_automation_mode_write() {
        let param = AutomationParameter::MasterGain;
        let mut lane = AutomationLane::new(param, 1.0);
        lane.mode = AutomationMode::Write;

        lane.set_value(0.5, 1000);
        lane.set_value(0.6, 2000);

        assert_eq!(lane.point_count(), 2);
    }

    #[test]
    fn test_automation_mode_touch() {
        let param = AutomationParameter::MasterGain;
        let mut lane = AutomationLane::new(param, 1.0);
        lane.mode = AutomationMode::Touch;

        // Should not write when not touching
        lane.set_value(0.5, 1000);
        assert_eq!(lane.point_count(), 0);

        // Should write when touching
        lane.start_touch(2000);
        lane.set_value(0.6, 2000);
        assert_eq!(lane.point_count(), 1);

        lane.stop_touch(3000);
        lane.set_value(0.7, 3000);
        assert_eq!(lane.point_count(), 1); // No new point after stop
    }

    #[test]
    fn test_thin_automation() {
        let param = AutomationParameter::MasterGain;
        let mut lane = AutomationLane::new(param, 1.0);

        // Add points with linear progression
        lane.add_point(AutomationPoint::new(0, 0.0));
        lane.add_point(AutomationPoint::new(500, 0.5));
        lane.add_point(AutomationPoint::new(1000, 1.0));

        assert_eq!(lane.point_count(), 3);

        // Middle point is redundant (linear interpolation)
        lane.thin_automation(0.01);
        assert_eq!(lane.point_count(), 2); // Should remove middle point
    }
}
