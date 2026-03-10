#![allow(dead_code)]
//! Audio bus routing and management for post-production mixing.
//!
//! Provides a flexible bus architecture with support for aux sends,
//! group buses, master bus, and monitor buses. Each bus has its own
//! gain staging, mute/solo, insert points, and metering.

use std::collections::HashMap;

/// Unique identifier for an audio bus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BusId(u32);

impl BusId {
    /// Create a new bus identifier.
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Return the raw identifier value.
    pub fn value(self) -> u32 {
        self.0
    }
}

/// Type of audio bus.
#[derive(Debug, Clone, PartialEq)]
pub enum BusType {
    /// Main output bus.
    Master,
    /// Group / sub-mix bus.
    Group,
    /// Auxiliary send bus.
    Aux,
    /// Monitor bus for control room.
    Monitor,
    /// Direct output bus.
    Direct,
    /// Stem bus (dialogue, music, effects, etc.).
    Stem,
}

/// Metering mode for a bus.
#[derive(Debug, Clone, PartialEq)]
pub enum MeteringMode {
    /// Peak metering.
    Peak,
    /// RMS metering.
    Rms,
    /// VU-style metering.
    Vu,
    /// K-system metering (K-14, K-20, etc.).
    KSystem(u8),
    /// Loudness metering (LUFS).
    Lufs,
}

/// Channel configuration for a bus.
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelConfig {
    /// Mono.
    Mono,
    /// Stereo.
    Stereo,
    /// LCR (left, center, right).
    Lcr,
    /// 5.1 surround.
    Surround51,
    /// 7.1 surround.
    Surround71,
    /// Dolby Atmos (object-based, logical channels).
    Atmos,
    /// Custom channel count.
    Custom(u32),
}

impl ChannelConfig {
    /// Return the number of channels for this configuration.
    pub fn channel_count(&self) -> u32 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Lcr => 3,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
            Self::Atmos => 16, // logical bed channels
            Self::Custom(n) => *n,
        }
    }
}

/// An insert point on a bus (pre-fader or post-fader).
#[derive(Debug, Clone)]
pub struct InsertPoint {
    /// Name of the inserted effect or processor.
    pub name: String,
    /// Whether the insert is enabled.
    pub enabled: bool,
    /// Whether the insert is pre-fader.
    pub pre_fader: bool,
    /// Wet/dry mix for the insert (0.0..1.0).
    pub mix: f64,
}

impl InsertPoint {
    /// Create a new insert point.
    pub fn new(name: impl Into<String>, pre_fader: bool) -> Self {
        Self {
            name: name.into(),
            enabled: true,
            pre_fader,
            mix: 1.0,
        }
    }

    /// Set the mix amount.
    pub fn with_mix(mut self, mix: f64) -> Self {
        self.mix = mix.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable the insert.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// A send from a bus to another bus.
#[derive(Debug, Clone)]
pub struct BusSend {
    /// Target bus ID.
    pub target: BusId,
    /// Send level (linear gain).
    pub level: f64,
    /// Whether this is a pre-fader send.
    pub pre_fader: bool,
    /// Whether the send is enabled.
    pub enabled: bool,
}

impl BusSend {
    /// Create a new bus send.
    pub fn new(target: BusId, level: f64, pre_fader: bool) -> Self {
        Self {
            target,
            level: level.max(0.0),
            pre_fader,
            enabled: true,
        }
    }
}

/// Metering data captured from a bus.
#[derive(Debug, Clone)]
pub struct MeterReading {
    /// Peak level per channel in dBFS.
    pub peak_dbfs: Vec<f64>,
    /// RMS level per channel in dBFS.
    pub rms_dbfs: Vec<f64>,
    /// Whether any channel is clipping.
    pub clipping: bool,
}

impl MeterReading {
    /// Create a new meter reading.
    pub fn new(num_channels: usize) -> Self {
        Self {
            peak_dbfs: vec![-f64::INFINITY; num_channels],
            rms_dbfs: vec![-f64::INFINITY; num_channels],
            clipping: false,
        }
    }

    /// Return the maximum peak across all channels.
    pub fn max_peak(&self) -> f64 {
        self.peak_dbfs
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Return the maximum RMS across all channels.
    pub fn max_rms(&self) -> f64 {
        self.rms_dbfs
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

/// An audio bus with full gain staging and routing.
#[derive(Debug)]
pub struct AudioBus {
    /// Unique identifier.
    pub id: BusId,
    /// Human-readable name.
    pub name: String,
    /// Bus type.
    pub bus_type: BusType,
    /// Channel configuration.
    pub channel_config: ChannelConfig,
    /// Fader gain in dB.
    pub gain_db: f64,
    /// Pan position (-1.0 left, 0.0 center, 1.0 right).
    pub pan: f64,
    /// Whether the bus is muted.
    pub muted: bool,
    /// Whether the bus is soloed.
    pub soloed: bool,
    /// Insert points.
    pub inserts: Vec<InsertPoint>,
    /// Sends to other buses.
    pub sends: Vec<BusSend>,
    /// Metering mode.
    pub metering_mode: MeteringMode,
    /// Latest meter reading.
    pub meter_reading: MeterReading,
    /// Phase invert flag.
    pub phase_invert: bool,
    /// Output bus destination (if not a master bus).
    pub output_bus: Option<BusId>,
    /// Metadata.
    pub tags: HashMap<String, String>,
}

impl AudioBus {
    /// Create a new audio bus.
    pub fn new(
        id: BusId,
        name: impl Into<String>,
        bus_type: BusType,
        channel_config: ChannelConfig,
    ) -> Self {
        let num_ch = channel_config.channel_count() as usize;
        Self {
            id,
            name: name.into(),
            bus_type,
            channel_config,
            gain_db: 0.0,
            pan: 0.0,
            muted: false,
            soloed: false,
            inserts: Vec::new(),
            sends: Vec::new(),
            metering_mode: MeteringMode::Peak,
            meter_reading: MeterReading::new(num_ch),
            phase_invert: false,
            output_bus: None,
            tags: HashMap::new(),
        }
    }

    /// Set the gain in dB.
    pub fn set_gain(&mut self, db: f64) {
        self.gain_db = db.clamp(-120.0, 24.0);
    }

    /// Set the pan position.
    pub fn set_pan(&mut self, pan: f64) {
        self.pan = pan.clamp(-1.0, 1.0);
    }

    /// Toggle mute.
    pub fn toggle_mute(&mut self) {
        self.muted = !self.muted;
    }

    /// Toggle solo.
    pub fn toggle_solo(&mut self) {
        self.soloed = !self.soloed;
    }

    /// Set phase inversion.
    pub fn set_phase_invert(&mut self, invert: bool) {
        self.phase_invert = invert;
    }

    /// Add an insert point.
    pub fn add_insert(&mut self, insert: InsertPoint) {
        self.inserts.push(insert);
    }

    /// Add a send.
    pub fn add_send(&mut self, send: BusSend) {
        self.sends.push(send);
    }

    /// Set the output bus.
    pub fn set_output(&mut self, bus_id: BusId) {
        self.output_bus = Some(bus_id);
    }

    /// Compute the linear gain from the dB fader value.
    #[allow(clippy::cast_precision_loss)]
    pub fn linear_gain(&self) -> f64 {
        if self.muted {
            return 0.0;
        }
        10.0_f64.powf(self.gain_db / 20.0)
    }

    /// Return the number of channels.
    pub fn num_channels(&self) -> u32 {
        self.channel_config.channel_count()
    }

    /// Add a tag.
    pub fn set_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.tags.insert(key.into(), value.into());
    }
}

/// Manager that owns and routes a set of audio buses.
#[derive(Debug)]
pub struct BusManager {
    /// All buses keyed by ID.
    buses: HashMap<BusId, AudioBus>,
    /// Counter for generating bus IDs.
    next_id: u32,
    /// The master bus ID (if one has been created).
    master_bus: Option<BusId>,
}

impl Default for BusManager {
    fn default() -> Self {
        Self::new()
    }
}

impl BusManager {
    /// Create a new bus manager.
    pub fn new() -> Self {
        Self {
            buses: HashMap::new(),
            next_id: 1,
            master_bus: None,
        }
    }

    /// Create and add a bus, returning its ID.
    pub fn create_bus(
        &mut self,
        name: impl Into<String>,
        bus_type: BusType,
        channel_config: ChannelConfig,
    ) -> BusId {
        let id = BusId::new(self.next_id);
        self.next_id += 1;
        let bus = AudioBus::new(id, name, bus_type.clone(), channel_config);
        if bus_type == BusType::Master && self.master_bus.is_none() {
            self.master_bus = Some(id);
        }
        self.buses.insert(id, bus);
        id
    }

    /// Get a reference to a bus by ID.
    pub fn get_bus(&self, id: BusId) -> Option<&AudioBus> {
        self.buses.get(&id)
    }

    /// Get a mutable reference to a bus by ID.
    pub fn get_bus_mut(&mut self, id: BusId) -> Option<&mut AudioBus> {
        self.buses.get_mut(&id)
    }

    /// Get the master bus ID.
    pub fn master_bus_id(&self) -> Option<BusId> {
        self.master_bus
    }

    /// Remove a bus by ID.
    pub fn remove_bus(&mut self, id: BusId) -> Option<AudioBus> {
        if self.master_bus == Some(id) {
            self.master_bus = None;
        }
        self.buses.remove(&id)
    }

    /// Return the total number of buses.
    pub fn bus_count(&self) -> usize {
        self.buses.len()
    }

    /// List all bus IDs.
    pub fn bus_ids(&self) -> Vec<BusId> {
        self.buses.keys().copied().collect()
    }

    /// Route a bus to the master bus.
    pub fn route_to_master(&mut self, bus_id: BusId) -> bool {
        if let Some(master_id) = self.master_bus {
            if let Some(bus) = self.buses.get_mut(&bus_id) {
                bus.set_output(master_id);
                return true;
            }
        }
        false
    }

    /// Mute all buses except soloed ones.
    pub fn apply_solo_logic(&mut self) {
        let any_soloed = self.buses.values().any(|b| b.soloed);
        if any_soloed {
            for bus in self.buses.values_mut() {
                if !bus.soloed && bus.bus_type != BusType::Master {
                    bus.muted = true;
                }
            }
        }
    }

    /// Clear solo on all buses and unmute.
    pub fn clear_solos(&mut self) {
        for bus in self.buses.values_mut() {
            bus.soloed = false;
            bus.muted = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bus_id() {
        let id = BusId::new(7);
        assert_eq!(id.value(), 7);
    }

    #[test]
    fn test_channel_config_counts() {
        assert_eq!(ChannelConfig::Mono.channel_count(), 1);
        assert_eq!(ChannelConfig::Stereo.channel_count(), 2);
        assert_eq!(ChannelConfig::Surround51.channel_count(), 6);
        assert_eq!(ChannelConfig::Surround71.channel_count(), 8);
        assert_eq!(ChannelConfig::Custom(4).channel_count(), 4);
    }

    #[test]
    fn test_new_bus_defaults() {
        let bus = AudioBus::new(
            BusId::new(1),
            "Main",
            BusType::Master,
            ChannelConfig::Stereo,
        );
        assert_eq!(bus.name, "Main");
        assert_eq!(bus.bus_type, BusType::Master);
        assert!(!bus.muted);
        assert!(!bus.soloed);
        assert!((bus.gain_db - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_set_gain_clamped() {
        let mut bus = AudioBus::new(BusId::new(1), "B", BusType::Group, ChannelConfig::Stereo);
        bus.set_gain(-200.0);
        assert!((bus.gain_db - (-120.0)).abs() < f64::EPSILON);
        bus.set_gain(100.0);
        assert!((bus.gain_db - 24.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_linear_gain() {
        let mut bus = AudioBus::new(BusId::new(1), "B", BusType::Group, ChannelConfig::Stereo);
        bus.set_gain(0.0);
        assert!((bus.linear_gain() - 1.0).abs() < 1e-10);

        bus.muted = true;
        assert!((bus.linear_gain() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_toggle_mute() {
        let mut bus = AudioBus::new(BusId::new(1), "B", BusType::Group, ChannelConfig::Stereo);
        assert!(!bus.muted);
        bus.toggle_mute();
        assert!(bus.muted);
        bus.toggle_mute();
        assert!(!bus.muted);
    }

    #[test]
    fn test_toggle_solo() {
        let mut bus = AudioBus::new(BusId::new(1), "B", BusType::Group, ChannelConfig::Stereo);
        assert!(!bus.soloed);
        bus.toggle_solo();
        assert!(bus.soloed);
    }

    #[test]
    fn test_insert_point() {
        let insert = InsertPoint::new("EQ", true).with_mix(0.5);
        assert_eq!(insert.name, "EQ");
        assert!(insert.pre_fader);
        assert!((insert.mix - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bus_send() {
        let send = BusSend::new(BusId::new(2), 0.7, true);
        assert_eq!(send.target, BusId::new(2));
        assert!((send.level - 0.7).abs() < f64::EPSILON);
        assert!(send.pre_fader);
    }

    #[test]
    fn test_meter_reading() {
        let mut reading = MeterReading::new(2);
        assert_eq!(reading.peak_dbfs.len(), 2);
        assert!(!reading.clipping);

        reading.peak_dbfs[0] = -6.0;
        reading.peak_dbfs[1] = -12.0;
        assert!((reading.max_peak() - (-6.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_manager_create_bus() {
        let mut mgr = BusManager::new();
        let id = mgr.create_bus("Master", BusType::Master, ChannelConfig::Stereo);
        assert_eq!(mgr.bus_count(), 1);
        assert_eq!(mgr.master_bus_id(), Some(id));
    }

    #[test]
    fn test_manager_route_to_master() {
        let mut mgr = BusManager::new();
        let master = mgr.create_bus("Master", BusType::Master, ChannelConfig::Stereo);
        let group = mgr.create_bus("Dialogue", BusType::Group, ChannelConfig::Stereo);
        assert!(mgr.route_to_master(group));
        let bus = mgr.get_bus(group).expect("get_bus should succeed");
        assert_eq!(bus.output_bus, Some(master));
    }

    #[test]
    fn test_manager_remove_bus() {
        let mut mgr = BusManager::new();
        let id = mgr.create_bus("Test", BusType::Aux, ChannelConfig::Mono);
        assert_eq!(mgr.bus_count(), 1);
        let removed = mgr.remove_bus(id);
        assert!(removed.is_some());
        assert_eq!(mgr.bus_count(), 0);
    }

    #[test]
    fn test_apply_solo_logic() {
        let mut mgr = BusManager::new();
        let _master = mgr.create_bus("Master", BusType::Master, ChannelConfig::Stereo);
        let dlg = mgr.create_bus("DLG", BusType::Group, ChannelConfig::Stereo);
        let sfx = mgr.create_bus("SFX", BusType::Group, ChannelConfig::Stereo);

        mgr.get_bus_mut(dlg)
            .expect("get_bus_mut should succeed")
            .soloed = true;
        mgr.apply_solo_logic();

        assert!(!mgr.get_bus(dlg).expect("get_bus should succeed").muted);
        assert!(mgr.get_bus(sfx).expect("get_bus should succeed").muted);
    }

    #[test]
    fn test_clear_solos() {
        let mut mgr = BusManager::new();
        let id = mgr.create_bus("Bus", BusType::Group, ChannelConfig::Stereo);
        mgr.get_bus_mut(id)
            .expect("get_bus_mut should succeed")
            .soloed = true;
        mgr.get_bus_mut(id)
            .expect("get_bus_mut should succeed")
            .muted = true;
        mgr.clear_solos();
        let bus = mgr.get_bus(id).expect("get_bus should succeed");
        assert!(!bus.soloed);
        assert!(!bus.muted);
    }

    #[test]
    fn test_default_manager() {
        let mgr = BusManager::default();
        assert_eq!(mgr.bus_count(), 0);
        assert!(mgr.master_bus_id().is_none());
    }

    #[test]
    fn test_bus_phase_invert() {
        let mut bus = AudioBus::new(BusId::new(1), "B", BusType::Direct, ChannelConfig::Mono);
        assert!(!bus.phase_invert);
        bus.set_phase_invert(true);
        assert!(bus.phase_invert);
    }

    #[test]
    fn test_bus_tags() {
        let mut bus = AudioBus::new(BusId::new(1), "B", BusType::Stem, ChannelConfig::Surround51);
        bus.set_tag("category", "music");
        assert_eq!(bus.tags.get("category").map(|s| s.as_str()), Some("music"));
    }
}
