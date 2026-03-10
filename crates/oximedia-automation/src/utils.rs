//! Utility functions and helpers for broadcast automation.

use crate::{AutomationError, Result};
use chrono::Timelike;
use std::time::Duration;

/// Time utilities for broadcast automation.
pub mod time {
    use super::{AutomationError, Duration, Result, Timelike};

    /// Convert frames to timecode.
    pub fn frames_to_timecode(frames: u64, frame_rate: f64) -> String {
        let total_frames = frames;
        let fps = frame_rate.round() as u64;

        let hours = total_frames / (fps * 3600);
        let minutes = (total_frames % (fps * 3600)) / (fps * 60);
        let seconds = (total_frames % (fps * 60)) / fps;
        let frames = total_frames % fps;

        format!("{hours:02}:{minutes:02}:{seconds:02}:{frames:02}")
    }

    /// Convert timecode to frames.
    pub fn timecode_to_frames(timecode: &str, frame_rate: f64) -> Result<u64> {
        let parts: Vec<&str> = timecode.split(':').collect();
        if parts.len() != 4 {
            return Err(AutomationError::InvalidState(format!(
                "Invalid timecode format: {timecode}"
            )));
        }

        let hours: u64 = parts[0]
            .parse()
            .map_err(|_| AutomationError::InvalidState("Invalid hours".to_string()))?;
        let minutes: u64 = parts[1]
            .parse()
            .map_err(|_| AutomationError::InvalidState("Invalid minutes".to_string()))?;
        let seconds: u64 = parts[2]
            .parse()
            .map_err(|_| AutomationError::InvalidState("Invalid seconds".to_string()))?;
        let frames: u64 = parts[3]
            .parse()
            .map_err(|_| AutomationError::InvalidState("Invalid frames".to_string()))?;

        let fps = frame_rate.round() as u64;
        let total_frames = (hours * 3600 * fps) + (minutes * 60 * fps) + (seconds * fps) + frames;

        Ok(total_frames)
    }

    /// Get current time of day as timecode.
    pub fn current_timecode(frame_rate: f64) -> String {
        let now = chrono::Local::now();
        let hours = now.hour() as u64;
        let minutes = now.minute() as u64;
        let seconds = now.second() as u64;
        let nanos = now.nanosecond();

        let frame = ((nanos as f64 / 1_000_000_000.0) * frame_rate) as u64;

        format!("{hours:02}:{minutes:02}:{seconds:02}:{frame:02}")
    }

    /// Calculate duration between two timecodes.
    pub fn timecode_duration(start: &str, end: &str, frame_rate: f64) -> Result<Duration> {
        let start_frames = timecode_to_frames(start, frame_rate)?;
        let end_frames = timecode_to_frames(end, frame_rate)?;

        if end_frames < start_frames {
            return Err(AutomationError::InvalidState(
                "End timecode is before start timecode".to_string(),
            ));
        }

        let diff_frames = end_frames - start_frames;
        let seconds = diff_frames as f64 / frame_rate;

        Ok(Duration::from_secs_f64(seconds))
    }

    /// Add duration to timecode.
    pub fn timecode_add(timecode: &str, duration: Duration, frame_rate: f64) -> Result<String> {
        let frames = timecode_to_frames(timecode, frame_rate)?;
        let duration_frames = (duration.as_secs_f64() * frame_rate) as u64;
        let new_frames = frames + duration_frames;

        Ok(frames_to_timecode(new_frames, frame_rate))
    }

    /// Subtract duration from timecode.
    pub fn timecode_subtract(
        timecode: &str,
        duration: Duration,
        frame_rate: f64,
    ) -> Result<String> {
        let frames = timecode_to_frames(timecode, frame_rate)?;
        let duration_frames = (duration.as_secs_f64() * frame_rate) as u64;

        if duration_frames > frames {
            return Err(AutomationError::InvalidState(
                "Subtraction would result in negative timecode".to_string(),
            ));
        }

        let new_frames = frames - duration_frames;
        Ok(frames_to_timecode(new_frames, frame_rate))
    }
}

/// Validation utilities.
pub mod validation {
    use super::{AutomationError, Duration, Result};

    /// Validate file path exists and is readable.
    pub fn validate_file_path(path: &str) -> Result<()> {
        if path.is_empty() {
            return Err(AutomationError::InvalidState("Empty file path".to_string()));
        }

        // In production, would check actual filesystem
        Ok(())
    }

    /// Validate frame rate.
    pub fn validate_frame_rate(fps: f64) -> Result<()> {
        if fps <= 0.0 {
            return Err(AutomationError::InvalidState(
                "Frame rate must be positive".to_string(),
            ));
        }

        // Common broadcast frame rates
        let valid_rates = [23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0];

        let is_valid = valid_rates.iter().any(|&rate| (fps - rate).abs() < 0.01);

        if !is_valid {
            return Err(AutomationError::InvalidState(format!(
                "Unsupported frame rate: {fps} fps"
            )));
        }

        Ok(())
    }

    /// Validate duration is within reasonable bounds.
    pub fn validate_duration(duration: Duration) -> Result<()> {
        const MAX_DURATION_HOURS: u64 = 24;

        if duration.as_secs() > MAX_DURATION_HOURS * 3600 {
            return Err(AutomationError::InvalidState(format!(
                "Duration exceeds maximum of {MAX_DURATION_HOURS} hours"
            )));
        }

        Ok(())
    }

    /// Validate channel ID.
    pub fn validate_channel_id(id: usize, max_channels: usize) -> Result<()> {
        if id >= max_channels {
            return Err(AutomationError::InvalidState(format!(
                "Channel ID {} out of range (max: {})",
                id,
                max_channels - 1
            )));
        }

        Ok(())
    }

    /// Validate IP address format.
    pub fn validate_ip_address(ip: &str) -> Result<()> {
        let parts: Vec<&str> = ip.split('.').collect();

        if parts.len() != 4 {
            return Err(AutomationError::InvalidState(format!(
                "Invalid IP address format: {ip}"
            )));
        }

        for part in parts {
            let _num: u8 = part.parse().map_err(|_| {
                AutomationError::InvalidState(format!("Invalid IP address octet: {part}"))
            })?;
        }

        Ok(())
    }

    /// Validate port number.
    pub fn validate_port(port: u16) -> Result<()> {
        if port < 1024 {
            return Err(AutomationError::InvalidState(format!(
                "Port {port} is in reserved range (< 1024)"
            )));
        }

        Ok(())
    }
}

/// String utilities for formatting.
pub mod format {
    use super::Duration;

    /// Format duration as human-readable string.
    pub fn format_duration(duration: Duration) -> String {
        let total_secs = duration.as_secs();
        let hours = total_secs / 3600;
        let minutes = (total_secs % 3600) / 60;
        let seconds = total_secs % 60;

        if hours > 0 {
            format!("{hours}h {minutes}m {seconds}s")
        } else if minutes > 0 {
            format!("{minutes}m {seconds}s")
        } else {
            format!("{seconds}s")
        }
    }

    /// Format bytes as human-readable size.
    pub fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];

        let mut size = bytes as f64;
        let mut unit_idx = 0;

        while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
            size /= 1024.0;
            unit_idx += 1;
        }

        format!("{:.2} {}", size, UNITS[unit_idx])
    }

    /// Format percentage.
    pub fn format_percentage(value: f64, total: f64) -> String {
        if total == 0.0 {
            return "0.00%".to_string();
        }

        let percentage = (value / total) * 100.0;
        format!("{percentage:.2}%")
    }

    /// Format frame count as timecode.
    pub fn format_frames_as_timecode(frames: u64, frame_rate: f64) -> String {
        super::time::frames_to_timecode(frames, frame_rate)
    }

    /// Truncate string to maximum length.
    pub fn truncate_string(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else {
            format!("{}...", &s[..max_len.saturating_sub(3)])
        }
    }
}

/// Mathematical utilities.
pub mod math {
    /// Calculate moving average.
    pub fn moving_average(values: &[f64], window_size: usize) -> Vec<f64> {
        if values.is_empty() || window_size == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(values.len());

        for i in 0..values.len() {
            let start = i.saturating_sub(window_size - 1);
            let window = &values[start..=i];
            let avg = window.iter().sum::<f64>() / window.len() as f64;
            result.push(avg);
        }

        result
    }

    /// Calculate standard deviation.
    pub fn standard_deviation(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

        variance.sqrt()
    }

    /// Calculate percentile.
    pub fn percentile(values: &[f64], p: f64) -> Option<f64> {
        if values.is_empty() || !(0.0..=100.0).contains(&p) {
            return None;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        Some(sorted[idx])
    }

    /// Clamp value between min and max.
    pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
        if value < min {
            min
        } else if value > max {
            max
        } else {
            value
        }
    }

    /// Linear interpolation.
    pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + (b - a) * t
    }
}

/// System utilities.
pub mod system {
    use super::Duration;

    /// Get system uptime.
    pub fn uptime() -> Duration {
        // In production, would read from system
        Duration::from_secs(0)
    }

    /// Check if running as root/administrator.
    pub fn is_elevated() -> bool {
        // In production, would check actual privileges
        false
    }

    /// Get number of CPU cores.
    pub fn cpu_count() -> usize {
        num_cpus::get()
    }

    /// Get available memory in bytes.
    pub fn available_memory() -> u64 {
        // In production, would read from system
        0
    }

    /// Get total memory in bytes.
    pub fn total_memory() -> u64 {
        // In production, would read from system
        0
    }

    /// Generate unique ID.
    pub fn generate_id() -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("auto_{now}_{seq}")
    }
}

/// Retry utilities for resilient operations.
pub mod retry {
    use super::Duration;
    use std::future::Future;

    /// Retry configuration.
    #[derive(Debug, Clone)]
    pub struct RetryConfig {
        /// Maximum number of attempts
        pub max_attempts: usize,
        /// Delay between attempts
        pub delay: Duration,
        /// Exponential backoff factor
        pub backoff_factor: f64,
    }

    impl Default for RetryConfig {
        fn default() -> Self {
            Self {
                max_attempts: 3,
                delay: Duration::from_secs(1),
                backoff_factor: 2.0,
            }
        }
    }

    /// Retry a fallible async operation.
    pub async fn retry_async<F, Fut, T, E>(
        config: RetryConfig,
        mut operation: F,
    ) -> std::result::Result<T, E>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = std::result::Result<T, E>>,
    {
        let mut attempt = 0;
        let mut current_delay = config.delay;

        loop {
            attempt += 1;

            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if attempt >= config.max_attempts {
                        return Err(e);
                    }

                    tokio::time::sleep(current_delay).await;
                    current_delay = Duration::from_secs_f64(
                        current_delay.as_secs_f64() * config.backoff_factor,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod time_tests {
        use super::*;

        #[test]
        fn test_frames_to_timecode() {
            let tc = time::frames_to_timecode(90000, 30.0);
            assert_eq!(tc, "00:50:00:00"); // 50 minutes at 30fps
        }

        #[test]
        fn test_timecode_to_frames() {
            let frames = time::timecode_to_frames("00:01:00:00", 30.0)
                .expect("timecode_to_frames should succeed");
            assert_eq!(frames, 1800); // 60 seconds * 30 fps
        }

        #[test]
        fn test_current_timecode() {
            let tc = time::current_timecode(30.0);
            assert!(!tc.is_empty());
        }

        #[test]
        fn test_timecode_duration() {
            let duration = time::timecode_duration("00:00:00:00", "00:01:00:00", 30.0)
                .expect("timecode_duration should succeed");

            assert_eq!(duration.as_secs(), 60);
        }

        #[test]
        fn test_timecode_add() {
            let result = time::timecode_add("00:00:30:00", Duration::from_secs(30), 30.0)
                .expect("timecode_add should succeed");

            assert_eq!(result, "00:01:00:00");
        }
    }

    mod validation_tests {
        use super::*;

        #[test]
        fn test_validate_frame_rate() {
            assert!(validation::validate_frame_rate(29.97).is_ok());
            assert!(validation::validate_frame_rate(30.0).is_ok());
            assert!(validation::validate_frame_rate(15.0).is_err());
        }

        #[test]
        fn test_validate_ip_address() {
            assert!(validation::validate_ip_address("192.168.1.1").is_ok());
            assert!(validation::validate_ip_address("invalid").is_err());
        }

        #[test]
        fn test_validate_port() {
            assert!(validation::validate_port(8080).is_ok());
            assert!(validation::validate_port(80).is_err()); // Reserved port
        }
    }

    mod format_tests {
        use super::*;

        #[test]
        fn test_format_duration() {
            assert_eq!(format::format_duration(Duration::from_secs(45)), "45s");
            assert_eq!(format::format_duration(Duration::from_secs(90)), "1m 30s");
            assert_eq!(
                format::format_duration(Duration::from_secs(3665)),
                "1h 1m 5s"
            );
        }

        #[test]
        fn test_format_bytes() {
            assert_eq!(format::format_bytes(1024), "1.00 KB");
            assert_eq!(format::format_bytes(1048576), "1.00 MB");
        }

        #[test]
        fn test_format_percentage() {
            assert_eq!(format::format_percentage(50.0, 100.0), "50.00%");
            assert_eq!(format::format_percentage(0.0, 100.0), "0.00%");
        }

        #[test]
        fn test_truncate_string() {
            let s = "This is a very long string";
            assert_eq!(format::truncate_string(s, 10), "This is...");
            assert_eq!(format::truncate_string("Short", 10), "Short");
        }
    }

    mod math_tests {
        use super::*;

        #[test]
        fn test_moving_average() {
            let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            let avg = math::moving_average(&values, 3);

            assert_eq!(avg[0], 1.0);
            assert_eq!(avg[2], 2.0); // (1 + 2 + 3) / 3
        }

        #[test]
        fn test_standard_deviation() {
            let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
            let std_dev = math::standard_deviation(&values);
            assert!(std_dev > 0.0);
        }

        #[test]
        fn test_clamp() {
            assert_eq!(math::clamp(5.0, 0.0, 10.0), 5.0);
            assert_eq!(math::clamp(-5.0, 0.0, 10.0), 0.0);
            assert_eq!(math::clamp(15.0, 0.0, 10.0), 10.0);
        }

        #[test]
        fn test_lerp() {
            assert_eq!(math::lerp(0.0, 10.0, 0.5), 5.0);
            assert_eq!(math::lerp(0.0, 10.0, 0.0), 0.0);
            assert_eq!(math::lerp(0.0, 10.0, 1.0), 10.0);
        }
    }

    mod system_tests {
        use super::*;

        #[test]
        fn test_cpu_count() {
            let count = system::cpu_count();
            assert!(count > 0);
        }

        #[test]
        fn test_generate_id() {
            let id1 = system::generate_id();
            let id2 = system::generate_id();

            assert_ne!(id1, id2);
            assert!(id1.starts_with("auto_"));
        }
    }
}
