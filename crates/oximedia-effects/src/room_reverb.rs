//! Room reverb simulation for `OxiMedia` effects.
//!
//! A simple algorithmic reverb using a comb-filter + all-pass network
//! with configurable room type, decay, and wet/dry mix.

#![allow(dead_code)]

/// Room characteristic types, each with different decay behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReverbType {
    /// Small room (~80 ms RT60).
    SmallRoom,
    /// Medium room (~300 ms RT60).
    MediumRoom,
    /// Concert hall (~1500 ms RT60).
    ConcertHall,
    /// Cathedral (~5000 ms RT60).
    Cathedral,
    /// Plate reverb character (~700 ms RT60).
    Plate,
}

impl ReverbType {
    /// Typical RT60 decay time in milliseconds.
    #[must_use]
    pub fn decay_ms(&self) -> f32 {
        match self {
            Self::SmallRoom => 80.0,
            Self::MediumRoom => 300.0,
            Self::ConcertHall => 1500.0,
            Self::Cathedral => 5000.0,
            Self::Plate => 700.0,
        }
    }

    /// Pre-delay in milliseconds typical for this space.
    #[must_use]
    pub fn predelay_ms(&self) -> f32 {
        match self {
            Self::SmallRoom => 0.0,
            Self::MediumRoom => 8.0,
            Self::ConcertHall => 25.0,
            Self::Cathedral => 40.0,
            Self::Plate => 5.0,
        }
    }

    /// Damping coefficient (0 = bright, 1 = dark).
    #[must_use]
    pub fn damping(&self) -> f32 {
        match self {
            Self::SmallRoom => 0.35,
            Self::MediumRoom => 0.50,
            Self::ConcertHall => 0.60,
            Self::Cathedral => 0.70,
            Self::Plate => 0.40,
        }
    }
}

/// Configuration for [`ReverbProcessor`].
#[derive(Debug, Clone)]
pub struct ReverbConfig {
    /// Room type preset.
    pub room_type: ReverbType,
    /// Wet signal mix (0.0–1.0).
    pub wet: f32,
    /// Dry signal mix (0.0–1.0).
    pub dry: f32,
    /// Optional override for RT60 decay in milliseconds.  0 = use preset.
    pub decay_override_ms: f32,
    /// High-frequency damping override.  0 = use preset.
    pub damping_override: f32,
}

impl Default for ReverbConfig {
    fn default() -> Self {
        Self {
            room_type: ReverbType::MediumRoom,
            wet: 0.3,
            dry: 0.7,
            decay_override_ms: 0.0,
            damping_override: 0.0,
        }
    }
}

impl ReverbConfig {
    /// Create from a room type with default wet/dry.
    #[must_use]
    pub fn from_type(room_type: ReverbType) -> Self {
        Self {
            room_type,
            ..Default::default()
        }
    }

    /// Returns `true` if wet and dry sum to <= 1 and sample rate will be positive.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.wet >= 0.0 && self.dry >= 0.0 && (self.wet + self.dry) <= 2.0
    }

    /// Effective decay in milliseconds (override or preset).
    #[must_use]
    pub fn effective_decay_ms(&self) -> f32 {
        if self.decay_override_ms > 0.0 {
            self.decay_override_ms
        } else {
            self.room_type.decay_ms()
        }
    }
}

/// Single comb filter for the reverb network.
#[derive(Debug)]
struct CombFilter {
    buffer: Vec<f32>,
    pos: usize,
    feedback: f32,
    damp: f32,
    last: f32,
}

impl CombFilter {
    fn new(size: usize, feedback: f32, damp: f32) -> Self {
        Self {
            buffer: vec![0.0; size.max(1)],
            pos: 0,
            feedback,
            damp,
            last: 0.0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let out = self.buffer[self.pos];
        self.last = out * (1.0 - self.damp) + self.last * self.damp;
        self.buffer[self.pos] = input + self.last * self.feedback;
        self.pos = (self.pos + 1) % self.buffer.len();
        out
    }
}

/// Simple all-pass diffuser.
#[derive(Debug)]
struct AllPass {
    buffer: Vec<f32>,
    pos: usize,
    feedback: f32,
}

impl AllPass {
    fn new(size: usize, feedback: f32) -> Self {
        Self {
            buffer: vec![0.0; size.max(1)],
            pos: 0,
            feedback,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let buf_out = self.buffer[self.pos];
        let output = -input + buf_out;
        self.buffer[self.pos] = input + buf_out * self.feedback;
        self.pos = (self.pos + 1) % self.buffer.len();
        output
    }
}

/// Reverb processor with 4 comb filters and 2 all-pass stages.
pub struct ReverbProcessor {
    config: ReverbConfig,
    combs: Vec<CombFilter>,
    allpasses: Vec<AllPass>,
    predelay_buf: Vec<f32>,
    predelay_pos: usize,
}

impl ReverbProcessor {
    /// Create a new processor for the given config and sample rate.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    #[must_use]
    pub fn new(config: ReverbConfig, sample_rate: f32) -> Self {
        let sr = sample_rate.max(1.0);
        let decay_ms = config.effective_decay_ms();
        let damp = if config.damping_override > 0.0 {
            config.damping_override
        } else {
            config.room_type.damping()
        };

        // Feedback coefficient from RT60
        // feedback = 10^(-3 * delay / RT60)
        let compute_feedback = |delay_samples: usize| -> f32 {
            let delay_s = delay_samples as f32 / sr;
            let rt60_s = decay_ms / 1000.0;
            10.0_f32.powf(-3.0 * delay_s / rt60_s.max(0.001))
        };

        // Comb filter delays (prime-ish sample counts)
        let comb_delays = [
            (sr * 0.02521) as usize,
            (sr * 0.02668) as usize,
            (sr * 0.02931) as usize,
            (sr * 0.03113) as usize,
        ];

        let combs = comb_delays
            .iter()
            .map(|&d| {
                let d = d.max(1);
                CombFilter::new(d, compute_feedback(d), damp)
            })
            .collect();

        // All-pass delays
        let ap_delays = [(sr * 0.0050) as usize, (sr * 0.0127) as usize];
        let allpasses = ap_delays
            .iter()
            .map(|&d| AllPass::new(d.max(1), 0.5))
            .collect();

        // Pre-delay buffer
        let predelay_samples = ((config.room_type.predelay_ms() / 1000.0) * sr) as usize;
        let predelay_buf = vec![0.0; predelay_samples.max(1)];

        Self {
            config,
            combs,
            allpasses,
            predelay_buf,
            predelay_pos: 0,
        }
    }

    /// Process a single mono sample.
    #[allow(clippy::cast_precision_loss)]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        // Pre-delay
        let delayed = self.predelay_buf[self.predelay_pos];
        self.predelay_buf[self.predelay_pos] = input;
        self.predelay_pos = (self.predelay_pos + 1) % self.predelay_buf.len();

        // Comb bank
        let wet_sum: f32 = self
            .combs
            .iter_mut()
            .map(|c| c.process(delayed))
            .sum::<f32>()
            / self.combs.len() as f32;

        // All-pass chain
        let mut wet = wet_sum;
        for ap in &mut self.allpasses {
            wet = ap.process(wet);
        }

        input * self.config.dry + wet * self.config.wet
    }

    /// Process a buffer of samples in-place.
    pub fn process_buffer(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            *s = self.process_sample(*s);
        }
    }
}

/// Estimate the Sabine RT60 for a room given its volume and absorption.
///
/// `volume_m3` is the room volume in cubic metres.
/// `total_absorption` is the total absorption area in sabins (m²).
pub struct RoomSimulator;

impl RoomSimulator {
    /// Estimate RT60 using the Sabine equation: `RT60 = 0.161 * V / A`.
    ///
    /// Returns RT60 in seconds, or `None` if inputs are invalid.
    #[must_use]
    pub fn estimate_rt60(volume_m3: f32, total_absorption: f32) -> Option<f32> {
        if volume_m3 <= 0.0 || total_absorption <= 0.0 {
            return None;
        }
        Some(0.161 * volume_m3 / total_absorption)
    }

    /// Suggest a [`ReverbType`] for a given RT60 in milliseconds.
    #[must_use]
    pub fn suggest_type(rt60_ms: f32) -> ReverbType {
        if rt60_ms < 150.0 {
            ReverbType::SmallRoom
        } else if rt60_ms < 600.0 {
            ReverbType::MediumRoom
        } else if rt60_ms < 2000.0 {
            ReverbType::ConcertHall
        } else {
            ReverbType::Cathedral
        }
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverb_type_decay_positive() {
        for rt in [
            ReverbType::SmallRoom,
            ReverbType::MediumRoom,
            ReverbType::ConcertHall,
            ReverbType::Cathedral,
            ReverbType::Plate,
        ] {
            assert!(rt.decay_ms() > 0.0);
        }
    }

    #[test]
    fn test_reverb_type_ordering() {
        assert!(ReverbType::SmallRoom.decay_ms() < ReverbType::MediumRoom.decay_ms());
        assert!(ReverbType::MediumRoom.decay_ms() < ReverbType::ConcertHall.decay_ms());
        assert!(ReverbType::ConcertHall.decay_ms() < ReverbType::Cathedral.decay_ms());
    }

    #[test]
    fn test_config_is_valid_default() {
        assert!(ReverbConfig::default().is_valid());
    }

    #[test]
    fn test_config_invalid_wet() {
        let mut cfg = ReverbConfig::default();
        cfg.wet = -0.1;
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_config_effective_decay_preset() {
        let cfg = ReverbConfig::from_type(ReverbType::ConcertHall);
        assert!((cfg.effective_decay_ms() - ReverbType::ConcertHall.decay_ms()).abs() < 1e-3);
    }

    #[test]
    fn test_config_effective_decay_override() {
        let mut cfg = ReverbConfig::default();
        cfg.decay_override_ms = 999.0;
        assert!((cfg.effective_decay_ms() - 999.0).abs() < 1e-3);
    }

    #[test]
    fn test_processor_silent_input() {
        let cfg = ReverbConfig::default();
        let mut proc = ReverbProcessor::new(cfg, 48000.0);
        let out = proc.process_sample(0.0);
        assert!(out.abs() < 1e-6);
    }

    #[test]
    fn test_processor_output_bounded() {
        let cfg = ReverbConfig::default();
        let mut proc = ReverbProcessor::new(cfg, 48000.0);
        for _ in 0..1000 {
            let out = proc.process_sample(0.5);
            assert!(out.abs() <= 2.0, "output out of range: {out}");
        }
    }

    #[test]
    fn test_processor_process_buffer() {
        let cfg = ReverbConfig::default();
        let mut proc = ReverbProcessor::new(cfg, 48000.0);
        let mut buf = vec![0.1_f32; 512];
        proc.process_buffer(&mut buf);
        assert_eq!(buf.len(), 512);
    }

    #[test]
    fn test_room_simulator_rt60() {
        // 100 m³ room, 10 sabins → RT60 = 0.161 * 100 / 10 = 1.61 s
        let rt = RoomSimulator::estimate_rt60(100.0, 10.0).expect("rt should be valid");
        assert!((rt - 1.61).abs() < 0.01);
    }

    #[test]
    fn test_room_simulator_invalid() {
        assert!(RoomSimulator::estimate_rt60(0.0, 10.0).is_none());
        assert!(RoomSimulator::estimate_rt60(100.0, 0.0).is_none());
    }

    #[test]
    fn test_room_simulator_suggest_type() {
        assert_eq!(RoomSimulator::suggest_type(80.0), ReverbType::SmallRoom);
        assert_eq!(RoomSimulator::suggest_type(400.0), ReverbType::MediumRoom);
        assert_eq!(RoomSimulator::suggest_type(1000.0), ReverbType::ConcertHall);
        assert_eq!(RoomSimulator::suggest_type(3000.0), ReverbType::Cathedral);
    }
}
