//! Timestamp correction functionality.
//!
//! This module provides functions to fix invalid timestamps
//! in media files.

/// Timestamp issue type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestampIssue {
    /// Negative timestamp.
    Negative,
    /// Out of order timestamp.
    OutOfOrder,
    /// Timestamp gap/jump.
    Gap,
    /// Duplicate timestamp.
    Duplicate,
}

/// Fix timestamp issues in a sequence.
pub fn fix_timestamps(timestamps: &mut [i64]) -> Vec<TimestampIssue> {
    let mut issues = Vec::new();

    if timestamps.is_empty() {
        return issues;
    }

    // Fix negative timestamps
    for ts in timestamps.iter_mut() {
        if *ts < 0 {
            issues.push(TimestampIssue::Negative);
            *ts = 0;
        }
    }

    // Fix out-of-order timestamps
    for i in 1..timestamps.len() {
        if timestamps[i] < timestamps[i - 1] {
            issues.push(TimestampIssue::OutOfOrder);
            // Interpolate
            timestamps[i] = timestamps[i - 1] + 1;
        }
    }

    // Detect and fix large gaps
    let avg_delta = calculate_average_delta(timestamps);
    for i in 1..timestamps.len() {
        let delta = timestamps[i] - timestamps[i - 1];
        if delta > avg_delta * 10 {
            issues.push(TimestampIssue::Gap);
            // Smooth the gap
            timestamps[i] = timestamps[i - 1] + avg_delta;
        }
    }

    // Fix duplicates
    for i in 1..timestamps.len() {
        if timestamps[i] == timestamps[i - 1] {
            issues.push(TimestampIssue::Duplicate);
            timestamps[i] = timestamps[i - 1] + 1;
        }
    }

    issues
}

/// Calculate average timestamp delta.
pub fn calculate_average_delta(timestamps: &[i64]) -> i64 {
    if timestamps.len() < 2 {
        return 40; // Default to ~25fps
    }

    let mut sum = 0i64;
    let mut count = 0;

    for i in 1..timestamps.len() {
        let delta = timestamps[i] - timestamps[i - 1];
        if delta > 0 && delta < 10000 {
            // Ignore extreme values
            sum += delta;
            count += 1;
        }
    }

    if count > 0 {
        sum / count
    } else {
        40
    }
}

/// Normalize timestamps to start from zero.
pub fn normalize_timestamps(timestamps: &mut [i64]) {
    if timestamps.is_empty() {
        return;
    }

    let min_ts = match timestamps.iter().min() {
        Some(&v) => v,
        None => return,
    };
    if min_ts != 0 {
        for ts in timestamps.iter_mut() {
            *ts -= min_ts;
        }
    }
}

/// Smooth timestamp jitter.
pub fn smooth_timestamps(timestamps: &mut [i64], window_size: usize) {
    if timestamps.len() < window_size {
        return;
    }

    let mut smoothed = timestamps.to_vec();

    for i in window_size..timestamps.len() - window_size {
        let mut sum = 0i64;
        for j in i.saturating_sub(window_size)..=(i + window_size).min(timestamps.len() - 1) {
            sum += timestamps[j];
        }
        smoothed[i] = sum / ((window_size * 2 + 1) as i64);
    }

    timestamps.copy_from_slice(&smoothed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fix_negative_timestamps() {
        let mut timestamps = vec![-100, 0, 100, 200];
        let issues = fix_timestamps(&mut timestamps);

        assert!(issues.contains(&TimestampIssue::Negative));
        assert!(timestamps.iter().all(|&ts| ts >= 0));
    }

    #[test]
    fn test_fix_out_of_order() {
        let mut timestamps = vec![0, 100, 50, 300];
        let issues = fix_timestamps(&mut timestamps);

        assert!(issues.contains(&TimestampIssue::OutOfOrder));
        assert!(timestamps.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn test_fix_duplicates() {
        let mut timestamps = vec![0, 100, 100, 200];
        let issues = fix_timestamps(&mut timestamps);

        assert!(issues.contains(&TimestampIssue::Duplicate));
        assert!(timestamps.windows(2).all(|w| w[0] != w[1]));
    }

    #[test]
    fn test_normalize_timestamps() {
        let mut timestamps = vec![1000, 1100, 1200, 1300];
        normalize_timestamps(&mut timestamps);

        assert_eq!(timestamps[0], 0);
        assert_eq!(timestamps[3], 300);
    }

    #[test]
    fn test_calculate_average_delta() {
        let timestamps = vec![0, 40, 80, 120, 160];
        let avg = calculate_average_delta(&timestamps);
        assert_eq!(avg, 40);
    }
}
