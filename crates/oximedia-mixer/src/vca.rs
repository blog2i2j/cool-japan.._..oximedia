//! VCA (Voltage Controlled Amplifier) group management.
//!
//! Provides VCA groups for linking and controlling multiple channels simultaneously,
//! with snapshot support for scene recall.

/// A VCA group that controls the gain and mute state of a set of channels.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct VcaGroup {
    /// Unique group identifier.
    pub id: u32,
    /// Display name.
    pub name: String,
    /// Channel indices belonging to this group.
    pub channels: Vec<u32>,
    /// Group gain in dB.
    pub gain_db: f32,
    /// Muted state.
    pub muted: bool,
}

impl VcaGroup {
    /// Create a new VCA group.
    #[must_use]
    #[allow(dead_code)]
    pub fn new(id: u32, name: impl Into<String>, channels: Vec<u32>) -> Self {
        Self {
            id,
            name: name.into(),
            channels,
            gain_db: 0.0,
            muted: false,
        }
    }

    /// Compute the effective linear gain.
    ///
    /// Returns `0.0` if muted, otherwise converts `gain_db` to linear gain.
    #[must_use]
    #[allow(dead_code)]
    pub fn effective_gain(&self) -> f32 {
        if self.muted {
            0.0
        } else {
            db_to_linear(self.gain_db)
        }
    }
}

/// Convert dB to linear gain.
#[inline]
#[allow(dead_code)]
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Convert linear gain to dB.
#[inline]
#[allow(dead_code)]
fn linear_to_db(linear: f32) -> f32 {
    if linear <= 0.0 {
        -f32::INFINITY
    } else {
        20.0 * linear.log10()
    }
}

/// Manages a collection of VCA groups.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct VcaManager {
    groups: Vec<VcaGroup>,
    next_id: u32,
}

impl VcaManager {
    /// Create a new VCA manager.
    #[must_use]
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            groups: Vec::new(),
            next_id: 0,
        }
    }

    /// Create a new VCA group and return its ID.
    #[allow(dead_code)]
    pub fn create_group(&mut self, name: &str, channels: Vec<u32>) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.groups.push(VcaGroup::new(id, name, channels));
        id
    }

    /// Set the gain (in dB) for a group.
    ///
    /// Does nothing if the group ID is not found.
    #[allow(dead_code)]
    pub fn set_gain(&mut self, group_id: u32, gain_db: f32) {
        if let Some(g) = self.groups.iter_mut().find(|g| g.id == group_id) {
            g.gain_db = gain_db;
        }
    }

    /// Set the mute state for a group.
    ///
    /// Does nothing if the group ID is not found.
    #[allow(dead_code)]
    pub fn mute(&mut self, group_id: u32, mute: bool) {
        if let Some(g) = self.groups.iter_mut().find(|g| g.id == group_id) {
            g.muted = mute;
        }
    }

    /// Apply the effective gain of a group to the per-channel gain array.
    ///
    /// Multiplies `channel_gains[ch]` by the group's effective gain for every channel
    /// index in the group. Does nothing if `group_id` is not found.
    #[allow(dead_code)]
    pub fn apply_to_channels(&self, group_id: u32, channel_gains: &mut Vec<f32>) {
        if let Some(g) = self.groups.iter().find(|g| g.id == group_id) {
            let gain = g.effective_gain();
            for &ch in &g.channels {
                let idx = ch as usize;
                if idx < channel_gains.len() {
                    channel_gains[idx] *= gain;
                }
            }
        }
    }

    /// Get a reference to a group by ID.
    #[must_use]
    #[allow(dead_code)]
    pub fn get_group(&self, group_id: u32) -> Option<&VcaGroup> {
        self.groups.iter().find(|g| g.id == group_id)
    }

    /// Capture a snapshot of all VCA group states.
    #[must_use]
    #[allow(dead_code)]
    pub fn capture_snapshot(&self) -> VcaSnapshot {
        VcaSnapshot {
            groups: self
                .groups
                .iter()
                .map(|g| (g.id, g.gain_db, g.muted))
                .collect(),
        }
    }

    /// Restore VCA group states from a snapshot.
    ///
    /// Only groups whose IDs exist in both the snapshot and the manager are updated.
    #[allow(dead_code)]
    pub fn restore_snapshot(&mut self, snap: &VcaSnapshot) {
        for &(id, gain_db, muted) in &snap.groups {
            if let Some(g) = self.groups.iter_mut().find(|g| g.id == id) {
                g.gain_db = gain_db;
                g.muted = muted;
            }
        }
    }
}

/// A lightweight snapshot of all VCA group states for scene recall.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct VcaSnapshot {
    /// `(group_id, gain_db, muted)` for each group.
    pub groups: Vec<(u32, f32, bool)>,
}

impl VcaSnapshot {
    /// Create an empty snapshot.
    #[must_use]
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self { groups: Vec::new() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vca_group_effective_gain_unity() {
        let g = VcaGroup::new(0, "Test", vec![0, 1]);
        assert!((g.effective_gain() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_vca_group_effective_gain_muted() {
        let mut g = VcaGroup::new(0, "Test", vec![0]);
        g.muted = true;
        assert_eq!(g.effective_gain(), 0.0);
    }

    #[test]
    fn test_vca_group_effective_gain_minus6db() {
        let mut g = VcaGroup::new(0, "Test", vec![0]);
        g.gain_db = -6.0;
        // -6dB ≈ 0.501
        assert!((g.effective_gain() - 0.501_187_2).abs() < 1e-4);
    }

    #[test]
    fn test_vca_manager_create_group() {
        let mut mgr = VcaManager::new();
        let id = mgr.create_group("Drums", vec![0, 1, 2]);
        assert_eq!(id, 0);
        assert!(mgr.get_group(id).is_some());
    }

    #[test]
    fn test_vca_manager_set_gain() {
        let mut mgr = VcaManager::new();
        let id = mgr.create_group("Synth", vec![3]);
        mgr.set_gain(id, -3.0);
        let g = mgr.get_group(id).expect("g should be valid");
        assert!((g.gain_db - (-3.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_vca_manager_mute() {
        let mut mgr = VcaManager::new();
        let id = mgr.create_group("Vocals", vec![5]);
        assert!(!mgr.get_group(id).expect("get_group should succeed").muted);
        mgr.mute(id, true);
        assert!(mgr.get_group(id).expect("get_group should succeed").muted);
    }

    #[test]
    fn test_vca_manager_apply_to_channels() {
        let mut mgr = VcaManager::new();
        let id = mgr.create_group("Group", vec![0, 2]);
        mgr.set_gain(id, 0.0); // unity

        let mut gains = vec![1.0f32, 1.0, 1.0, 1.0];
        mgr.apply_to_channels(id, &mut gains);

        // Channels 0 and 2 should remain at 1.0
        assert!((gains[0] - 1.0).abs() < 1e-5);
        assert!((gains[2] - 1.0).abs() < 1e-5);
        // Channels 1 and 3 should be untouched
        assert!((gains[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_vca_manager_apply_muted() {
        let mut mgr = VcaManager::new();
        let id = mgr.create_group("Group", vec![0, 1]);
        mgr.mute(id, true);

        let mut gains = vec![1.0f32, 1.0];
        mgr.apply_to_channels(id, &mut gains);
        assert_eq!(gains[0], 0.0);
        assert_eq!(gains[1], 0.0);
    }

    #[test]
    fn test_vca_snapshot_capture_restore() {
        let mut mgr = VcaManager::new();
        let id = mgr.create_group("Test", vec![0]);
        mgr.set_gain(id, -12.0);

        let snap = mgr.capture_snapshot();
        assert_eq!(snap.groups.len(), 1);
        assert!((snap.groups[0].1 - (-12.0)).abs() < f32::EPSILON);

        // Change then restore
        mgr.set_gain(id, 0.0);
        mgr.restore_snapshot(&snap);
        assert!(
            (mgr.get_group(id).expect("get_group should succeed").gain_db - (-12.0)).abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn test_vca_snapshot_restore_unknown_group_ignored() {
        let mut mgr = VcaManager::new();
        let snap = VcaSnapshot {
            groups: vec![(99, -6.0, false)],
        };
        // Should not panic
        mgr.restore_snapshot(&snap);
    }

    #[test]
    fn test_db_to_linear_zero_db() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_to_db_unity() {
        assert!((linear_to_db(1.0)).abs() < 1e-6);
    }
}
