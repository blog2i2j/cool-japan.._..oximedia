//! System clock adjustment via FFI.
//!
//! This module requires unsafe code to interface with system calls.
#![allow(unsafe_code)]

use crate::error::{TimeSyncError, TimeSyncResult};

/// Get current system time (nanoseconds since Unix epoch).
pub fn get_system_time() -> TimeSyncResult<u64> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| TimeSyncError::ClockAdjust(format!("System time error: {e}")))?;

    Ok(duration.as_secs() * 1_000_000_000 + u64::from(duration.subsec_nanos()))
}

/// Adjust system clock (requires elevated privileges).
///
/// `offset_ns` is the signed offset in nanoseconds to apply to the clock.
/// `freq_adjust_ppb` is the signed frequency adjustment in parts-per-billion.
///
/// On Linux this uses `clock_adjtime(CLOCK_REALTIME, …)` with `ADJ_OFFSET | ADJ_FREQUENCY`.
/// On macOS this uses `adjtime(2)`.
/// On other targets (WASM, Windows) the function returns an error indicating the
/// operation is not supported.
pub fn adjust_system_clock(offset_ns: i64, freq_adjust_ppb: f64) -> TimeSyncResult<()> {
    // Guard: refuse obviously dangerous step-adjustments that should go through
    // settimeofday/SetSystemTime instead.
    if offset_ns.abs() > 1_000_000_000 {
        return Err(TimeSyncError::ClockAdjust(
            "Offset too large for safe slew adjustment (>1 s); use a step adjustment instead"
                .to_string(),
        ));
    }

    #[cfg(target_os = "linux")]
    {
        linux_adjtime(offset_ns, freq_adjust_ppb)
    }

    #[cfg(target_os = "macos")]
    {
        macos_adjtime(offset_ns, freq_adjust_ppb)
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = freq_adjust_ppb;
        Err(TimeSyncError::ClockAdjust(
            "Clock adjustment is not supported on this platform".to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Linux implementation
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
fn linux_adjtime(offset_ns: i64, freq_adjust_ppb: f64) -> TimeSyncResult<()> {
    // `adjtimex` / `clock_adjtime` expect:
    //   modes  = ADJ_OFFSET | ADJ_FREQUENCY (or ADJ_NANO if nano precision is used)
    //   offset = nanoseconds when ADJ_NANO is set, otherwise microseconds
    //   freq   = scaled-ppm: 1 ppm = 65536 kernel units
    //            ppb → ppm = ppb / 1000
    //            kernel_freq = (ppb / 1000) * 65536 = ppb * 65.536
    //
    // We use ADJ_NANO so we can pass nanoseconds directly.

    // ADJ_OFFSET | ADJ_FREQUENCY | ADJ_NANO
    const ADJ_OFFSET: libc::c_uint = 0x0001;
    const ADJ_FREQUENCY: libc::c_uint = 0x0002;
    const ADJ_NANO: libc::c_uint = 0x2000;

    let mut tx: libc::timex = unsafe { std::mem::zeroed() };
    tx.modes = (ADJ_OFFSET | ADJ_FREQUENCY | ADJ_NANO) as _;
    tx.offset = offset_ns as libc::c_long;
    // kernel freq units: scaled-ppm where 1 ppm = 2^16 = 65536
    tx.freq = (freq_adjust_ppb * 65.536) as libc::c_long;

    let ret = unsafe { libc::clock_adjtime(libc::CLOCK_REALTIME, &mut tx) };
    if ret < 0 {
        let errno = unsafe { *libc::__errno_location() };
        return Err(TimeSyncError::ClockAdjust(format!(
            "clock_adjtime(2) failed: errno {errno}"
        )));
    }

    tracing::debug!(
        offset_ns,
        freq_adjust_ppb,
        clock_state = ret,
        "Applied clock adjustment via clock_adjtime"
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// macOS implementation
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
fn macos_adjtime(offset_ns: i64, freq_adjust_ppb: f64) -> TimeSyncResult<()> {
    // macOS does not expose clock_adjtime; use adjtime(2) instead.
    // adjtime(2) accepts a `struct timeval` (seconds + microseconds) and
    // slews the clock at a rate of up to ~500 µs/s until the delta is consumed.
    //
    // Frequency adjustment is not directly supported by adjtime(2); we log a
    // warning and apply only the offset.
    if freq_adjust_ppb.abs() > f64::EPSILON {
        tracing::warn!(
            freq_adjust_ppb,
            "macOS adjtime(2) does not support frequency adjustment; only offset will be applied"
        );
    }

    // Convert nanoseconds to seconds + microseconds.
    let total_us = offset_ns / 1_000; // truncate to microseconds
    let tv = libc::timeval {
        tv_sec: (total_us / 1_000_000) as libc::time_t,
        tv_usec: (total_us % 1_000_000) as libc::suseconds_t,
    };

    let ret = unsafe { libc::adjtime(&tv, std::ptr::null_mut()) };
    if ret != 0 {
        let errno = unsafe { *libc::__error() };
        return Err(TimeSyncError::ClockAdjust(format!(
            "adjtime(2) failed: errno {errno}"
        )));
    }

    tracing::debug!(offset_ns, "Applied clock adjustment via adjtime");

    Ok(())
}

/// Get clock adjustment limits.
pub struct ClockLimits {
    /// Maximum offset adjustment (nanoseconds)
    pub max_offset_ns: i64,
    /// Maximum frequency adjustment (ppb)
    pub max_freq_ppb: f64,
    /// Whether clock adjustment is supported
    pub supported: bool,
}

impl Default for ClockLimits {
    fn default() -> Self {
        Self {
            max_offset_ns: 500_000_000, // 500 ms
            max_freq_ppb: 500.0,        // 500 ppb = 0.5 ppm
            supported: cfg!(target_os = "linux") || cfg!(target_os = "macos"),
        }
    }
}

/// Get platform clock adjustment limits.
#[must_use]
pub fn get_clock_limits() -> ClockLimits {
    ClockLimits::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_system_time() {
        let time = get_system_time().expect("should succeed in test");
        assert!(time > 0);
    }

    #[test]
    fn test_adjust_clock_validation() {
        // Large adjustment must be rejected before any syscall.
        let result = adjust_system_clock(2_000_000_000, 10.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_clock_limits() {
        let limits = get_clock_limits();
        assert!(limits.max_offset_ns > 0);
        assert!(limits.max_freq_ppb > 0.0);
    }
}
