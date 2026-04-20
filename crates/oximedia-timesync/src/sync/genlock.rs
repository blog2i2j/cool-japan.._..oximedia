//! Genlock reference generation for video synchronization.

use std::time::{Duration, Instant};

/// Video frame rate for genlock.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenlockFrameRate {
    /// Numerator
    pub num: u32,
    /// Denominator
    pub den: u32,
}

impl GenlockFrameRate {
    /// 23.976 fps (24000/1001)
    pub const FPS_23_976: Self = Self {
        num: 24000,
        den: 1001,
    };
    /// 24 fps
    pub const FPS_24: Self = Self { num: 24, den: 1 };
    /// 25 fps (PAL)
    pub const FPS_25: Self = Self { num: 25, den: 1 };
    /// 29.97 fps (30000/1001)
    pub const FPS_29_97: Self = Self {
        num: 30000,
        den: 1001,
    };
    /// 30 fps
    pub const FPS_30: Self = Self { num: 30, den: 1 };
    /// 50 fps
    pub const FPS_50: Self = Self { num: 50, den: 1 };
    /// 59.94 fps (60000/1001)
    pub const FPS_59_94: Self = Self {
        num: 60000,
        den: 1001,
    };
    /// 60 fps
    pub const FPS_60: Self = Self { num: 60, den: 1 };

    /// Get frame duration.
    #[must_use]
    pub fn frame_duration(&self) -> Duration {
        let nanos = (u128::from(self.den) * 1_000_000_000) / u128::from(self.num);
        Duration::from_nanos(nanos as u64)
    }

    /// Get frames per second as float.
    #[must_use]
    pub fn as_f64(&self) -> f64 {
        f64::from(self.num) / f64::from(self.den)
    }
}

/// Genlock generator.
pub struct GenlockGenerator {
    /// Frame rate
    frame_rate: GenlockFrameRate,
    /// Start time
    start_time: Instant,
    /// Current frame number
    frame_number: u64,
    /// Phase adjustment (nanoseconds)
    phase_adjust_ns: i64,
}

impl GenlockGenerator {
    /// Create a new genlock generator.
    #[must_use]
    pub fn new(frame_rate: GenlockFrameRate) -> Self {
        Self {
            frame_rate,
            start_time: Instant::now(),
            frame_number: 0,
            phase_adjust_ns: 0,
        }
    }

    /// Get next frame time.
    pub fn next_frame_time(&mut self) -> Instant {
        let frame_duration = self.frame_rate.frame_duration();
        let ideal_time = self.start_time + frame_duration * self.frame_number as u32;

        // Apply phase adjustment
        let adjusted_time = if self.phase_adjust_ns >= 0 {
            ideal_time + Duration::from_nanos(self.phase_adjust_ns as u64)
        } else {
            ideal_time
                .checked_sub(Duration::from_nanos((-self.phase_adjust_ns) as u64))
                .unwrap_or(ideal_time)
        };

        self.frame_number += 1;
        adjusted_time
    }

    /// Adjust phase (positive = delay, negative = advance).
    pub fn adjust_phase(&mut self, adjust_ns: i64) {
        self.phase_adjust_ns += adjust_ns;

        // Limit phase adjustment to ±1 frame
        let frame_nanos = self.frame_rate.frame_duration().as_nanos() as i64;
        if self.phase_adjust_ns > frame_nanos {
            self.phase_adjust_ns = frame_nanos;
        } else if self.phase_adjust_ns < -frame_nanos {
            self.phase_adjust_ns = -frame_nanos;
        }
    }

    /// Reset genlock to current time.
    pub fn reset(&mut self) {
        self.start_time = Instant::now();
        self.frame_number = 0;
        self.phase_adjust_ns = 0;
    }

    /// Get current frame number.
    #[must_use]
    pub fn frame_number(&self) -> u64 {
        self.frame_number
    }

    /// Get phase adjustment.
    #[must_use]
    pub fn phase_adjust(&self) -> i64 {
        self.phase_adjust_ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genlock_frame_rate() {
        let fps25 = GenlockFrameRate::FPS_25;
        assert_eq!(fps25.as_f64(), 25.0);

        let duration = fps25.frame_duration();
        assert_eq!(duration.as_millis(), 40); // 1000ms / 25fps = 40ms
    }

    #[test]
    fn test_genlock_generator() {
        let mut gen = GenlockGenerator::new(GenlockFrameRate::FPS_25);

        let t1 = gen.next_frame_time();
        let t2 = gen.next_frame_time();

        let diff = t2.duration_since(t1);
        assert!((diff.as_millis() as i64 - 40).abs() < 2); // ~40ms between frames
    }

    #[test]
    fn test_phase_adjustment() {
        let mut gen = GenlockGenerator::new(GenlockFrameRate::FPS_25);

        gen.adjust_phase(1_000_000); // 1ms advance
        assert_eq!(gen.phase_adjust(), 1_000_000);

        gen.reset();
        assert_eq!(gen.phase_adjust(), 0);
        assert_eq!(gen.frame_number(), 0);
    }
}
