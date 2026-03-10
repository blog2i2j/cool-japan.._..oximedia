//! Frame pacing for consistent frame delivery.
//!
//! Ensures frames are delivered at precise intervals for smooth playback.

use crate::{GamingError, GamingResult};
use std::time::{Duration, Instant};

/// Frame pacer for consistent frame delivery.
#[allow(dead_code)]
pub struct FramePacer {
    mode: PacingMode,
    target_frame_time: Duration,
    last_frame_time: Option<Instant>,
    /// Number of frames paced so far.
    pub frame_count: u64,
}

/// Frame pacing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacingMode {
    /// Fixed frame rate
    Fixed,
    /// Variable frame rate (within bounds)
    Variable,
    /// Adaptive pacing based on system performance
    Adaptive,
}

/// Frame timing information.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FrameTimingInfo {
    /// Frame number
    pub frame_number: u64,
    /// Actual frame time
    pub actual_time: Duration,
    /// Target frame time
    pub target_time: Duration,
    /// Timing error (actual - target)
    pub timing_error: Duration,
    /// Should drop frame
    pub should_drop: bool,
}

impl FramePacer {
    /// Create a new frame pacer.
    ///
    /// # Errors
    ///
    /// Returns error if target framerate is invalid.
    pub fn new(target_fps: u32, mode: PacingMode) -> GamingResult<Self> {
        if target_fps == 0 || target_fps > 240 {
            return Err(GamingError::InvalidConfig(
                "Target FPS must be between 1 and 240".to_string(),
            ));
        }

        let target_frame_time = Duration::from_secs_f64(1.0 / f64::from(target_fps));

        Ok(Self {
            mode,
            target_frame_time,
            last_frame_time: None,
            frame_count: 0,
        })
    }

    /// Wait for next frame time.
    ///
    /// # Errors
    ///
    /// Returns error if timing calculation fails.
    pub async fn wait_for_next_frame(&mut self) -> GamingResult<FrameTimingInfo> {
        let now = Instant::now();
        self.frame_count += 1;

        let (should_wait, timing_error) = if let Some(last_time) = self.last_frame_time {
            let elapsed = now.duration_since(last_time);

            if elapsed < self.target_frame_time {
                // elapsed < target_frame_time, so the subtraction cannot underflow.
                let wait_time = self.target_frame_time.saturating_sub(elapsed);
                tokio::time::sleep(wait_time).await;
                (true, Duration::ZERO)
            } else {
                // elapsed >= target_frame_time, so the subtraction cannot underflow.
                (false, elapsed.saturating_sub(self.target_frame_time))
            }
        } else {
            (false, Duration::ZERO)
        };

        let last_frame_instant = if should_wait { Instant::now() } else { now };
        self.last_frame_time = Some(last_frame_instant);

        Ok(FrameTimingInfo {
            frame_number: self.frame_count,
            actual_time: last_frame_instant.duration_since(now),
            target_time: self.target_frame_time,
            timing_error,
            should_drop: timing_error > self.target_frame_time,
        })
    }

    /// Reset frame pacer.
    pub fn reset(&mut self) {
        self.last_frame_time = None;
        self.frame_count = 0;
    }

    /// Get current frame count.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get target frame time.
    #[must_use]
    pub fn target_frame_time(&self) -> Duration {
        self.target_frame_time
    }

    /// Change target framerate.
    ///
    /// # Errors
    ///
    /// Returns error if framerate is invalid.
    pub fn set_target_fps(&mut self, fps: u32) -> GamingResult<()> {
        if fps == 0 || fps > 240 {
            return Err(GamingError::InvalidConfig(
                "Target FPS must be between 1 and 240".to_string(),
            ));
        }

        self.target_frame_time = Duration::from_secs_f64(1.0 / f64::from(fps));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_pacer_creation() {
        let pacer = FramePacer::new(60, PacingMode::Fixed).expect("valid frame pacer");
        assert_eq!(pacer.frame_count(), 0);
    }

    #[test]
    fn test_invalid_fps() {
        let result = FramePacer::new(0, PacingMode::Fixed);
        assert!(result.is_err());

        let result = FramePacer::new(300, PacingMode::Fixed);
        assert!(result.is_err());
    }

    #[test]
    fn test_target_frame_time() {
        let pacer = FramePacer::new(60, PacingMode::Fixed).expect("valid frame pacer");
        let expected = Duration::from_secs_f64(1.0 / 60.0);
        assert!((pacer.target_frame_time().as_secs_f64() - expected.as_secs_f64()).abs() < 0.0001);
    }

    #[tokio::test]
    async fn test_wait_for_next_frame() {
        let mut pacer = FramePacer::new(60, PacingMode::Fixed).expect("valid frame pacer");

        let timing = pacer
            .wait_for_next_frame()
            .await
            .expect("frame timing should succeed");
        assert_eq!(timing.frame_number, 1);
        assert!(!timing.should_drop);
    }

    #[test]
    fn test_reset() {
        let mut pacer = FramePacer::new(60, PacingMode::Fixed).expect("valid frame pacer");
        pacer.frame_count = 100;
        pacer.reset();
        assert_eq!(pacer.frame_count(), 0);
    }

    #[test]
    fn test_set_target_fps() {
        let mut pacer = FramePacer::new(60, PacingMode::Fixed).expect("valid frame pacer");
        pacer.set_target_fps(120).expect("set fps should succeed");

        let expected = Duration::from_secs_f64(1.0 / 120.0);
        assert!((pacer.target_frame_time().as_secs_f64() - expected.as_secs_f64()).abs() < 0.0001);
    }
}
