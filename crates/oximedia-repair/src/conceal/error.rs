//! Error concealment strategies.
//!
//! This module provides general error concealment functions.

use crate::Result;

/// Concealment strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcealmentStrategy {
    /// Copy previous frame/sample.
    CopyPrevious,
    /// Interpolate between frames/samples.
    Interpolate,
    /// Insert silence/black.
    InsertBlank,
}

/// Apply error concealment to data.
///
/// Scans `data` for runs of consecutive zero bytes longer than
/// `MIN_CORRUPT_RUN` and applies the chosen `strategy` to each detected
/// corrupt region.
///
/// - `CopyPrevious`: fills each corrupt region with the byte value that
///   immediately preceded the run.
/// - `Interpolate`: linearly interpolates between the byte before and the
///   byte after the corrupt region.
/// - `InsertBlank`: leaves zeros in place (explicit silence / black).
pub fn apply_concealment(data: &mut [u8], strategy: ConcealmentStrategy) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }

    // Detect corrupt regions (runs of zeros longer than threshold)
    let areas = detect_concealment_areas(data);

    match strategy {
        ConcealmentStrategy::InsertBlank => {
            // Already zeros – nothing to do.
        }
        ConcealmentStrategy::CopyPrevious => {
            for &(start, end) in &areas {
                // Use the byte just before the corrupt region, or 128 (neutral)
                let fill = if start > 0 { data[start - 1] } else { 128 };
                for byte in &mut data[start..end] {
                    *byte = fill;
                }
            }
        }
        ConcealmentStrategy::Interpolate => {
            for &(start, end) in &areas {
                let before = if start > 0 {
                    f64::from(data[start - 1])
                } else {
                    128.0
                };
                let after = if end < data.len() {
                    f64::from(data[end])
                } else {
                    128.0
                };
                let region_len = end - start;
                if region_len == 0 {
                    continue;
                }
                for (i, byte) in data[start..end].iter_mut().enumerate() {
                    let t = (i + 1) as f64 / (region_len + 1) as f64;
                    *byte = (before * (1.0 - t) + after * t).clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    Ok(())
}

/// Detect areas needing concealment.
pub fn detect_concealment_areas(data: &[u8]) -> Vec<(usize, usize)> {
    let mut areas = Vec::new();
    let mut start = None;

    for (i, &byte) in data.iter().enumerate() {
        // Simple heuristic: consecutive zeros
        if byte == 0 {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start {
            if i - s > 100 {
                // Only conceal large runs
                areas.push((s, i));
            }
            start = None;
        }
    }

    areas
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_concealment_areas() {
        let mut data = vec![1; 300];
        for i in 50..160 {
            data[i] = 0;
        }

        let areas = detect_concealment_areas(&data);
        assert!(!areas.is_empty());
        assert_eq!(areas[0], (50, 160));
    }

    #[test]
    fn test_apply_concealment_copy_previous() {
        let mut data = vec![200u8; 300];
        for i in 50..160 {
            data[i] = 0;
        }

        apply_concealment(&mut data, ConcealmentStrategy::CopyPrevious)
            .expect("concealment should succeed");
        // Corrupt region should be filled with 200 (byte before run)
        assert_eq!(data[50], 200);
        assert_eq!(data[100], 200);
        assert_eq!(data[159], 200);
    }

    #[test]
    fn test_apply_concealment_interpolate() {
        let mut data = vec![0u8; 300];
        data[49] = 100;
        // zeros from 50..160
        data[160] = 200;
        for i in 161..300 {
            data[i] = 200;
        }

        apply_concealment(&mut data, ConcealmentStrategy::Interpolate)
            .expect("concealment should succeed");
        // First element in region should be close to 100
        assert!(data[50] > 90 && data[50] < 120, "data[50] = {}", data[50]);
        // Last element should be close to 200
        assert!(
            data[159] > 180 && data[159] < 210,
            "data[159] = {}",
            data[159]
        );
    }

    #[test]
    fn test_apply_concealment_insert_blank() {
        let mut data = vec![1u8; 300];
        for i in 50..160 {
            data[i] = 0;
        }

        apply_concealment(&mut data, ConcealmentStrategy::InsertBlank)
            .expect("concealment should succeed");
        // Should remain zeros
        assert_eq!(data[50], 0);
        assert_eq!(data[100], 0);
    }

    #[test]
    fn test_apply_concealment_empty_data() {
        let mut data: Vec<u8> = Vec::new();
        apply_concealment(&mut data, ConcealmentStrategy::CopyPrevious)
            .expect("empty data should succeed");
    }
}
