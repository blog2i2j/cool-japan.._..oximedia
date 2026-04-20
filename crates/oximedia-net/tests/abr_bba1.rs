//! Integration tests for the BBA-1 adaptive bitrate strategy.
//!
//! These tests verify the pure-function [`oximedia_net::abr::bba1::select_variant`]
//! against the invariants specified by Huang et al. (SIGCOMM 2014).

use std::time::Duration;

use oximedia_net::abr::{
    bba1::{select_variant, BbaParams},
    streaming::AbrVariant,
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Build a minimal [`AbrVariant`] from a bandwidth value.
fn variant(bandwidth: u64) -> AbrVariant {
    AbrVariant {
        bandwidth,
        width: 0,
        height: 0,
        codecs: String::new(),
        uri: String::new(),
        name: String::new(),
        frame_rate: None,
        hdcp_level: None,
    }
}

/// Default BBA-1 parameters: B = 30 s, r = 10 s, c = 20 s.
fn params() -> BbaParams {
    BbaParams::default()
}

/// Three-rung ladder: 500 kbps / 1.5 Mbps / 4 Mbps, sorted ascending.
fn variants_3() -> Vec<AbrVariant> {
    vec![variant(500_000), variant(1_500_000), variant(4_000_000)]
}

// ─── Tests ───────────────────────────────────────────────────────────────────

/// Buffer 5 s falls below reservoir (10 s) → must always pick index 0.
#[test]
fn bba1_reservoir_picks_lowest() {
    assert_eq!(
        select_variant(Duration::from_secs(5), &params(), &variants_3()),
        0,
        "buffer below reservoir must select lowest variant"
    );
}

/// Buffer at the midpoint of the cushion region [10 s, 30 s] = 20 s.
/// Cushion position = (20 - 10) / (30 - 10) = 0.5.
/// Selected index = round(0.5 * 2) = round(1.0) = 1 → middle variant.
#[test]
fn bba1_cushion_midpoint_picks_middle() {
    assert_eq!(
        select_variant(Duration::from_secs(20), &params(), &variants_3()),
        1,
        "cushion midpoint (20 s) must select middle variant"
    );
}

/// Buffer at or above reservoir + cushion = 30 s → highest quality.
#[test]
fn bba1_upper_picks_highest() {
    assert_eq!(
        select_variant(Duration::from_secs(30), &params(), &variants_3()),
        2,
        "buffer at cushion upper bound must select highest variant"
    );
}

/// Cold start: buffer level zero → still in reservoir → lowest quality.
#[test]
fn bba1_cold_start_zero_buffer_picks_lowest() {
    assert_eq!(
        select_variant(Duration::ZERO, &params(), &variants_3()),
        0,
        "zero buffer must always select lowest variant"
    );
}

/// A single-variant ladder should always return index 0 regardless of buffer.
#[test]
fn bba1_single_variant_always_returns_zero() {
    let v = vec![variant(1_000_000)];
    for secs in [0u64, 5, 15, 30, 60] {
        assert_eq!(
            select_variant(Duration::from_secs(secs), &params(), &v),
            0,
            "single variant at {secs} s should always return 0"
        );
    }
}

/// Simulate 100 segments with an oscillating buffer between 12 s and 25 s.
///
/// BBA-1's continuous, monotone mapping means a smooth buffer oscillation
/// should never cause the selected index to jump by more than 1 rung upward
/// in a single step — enforcing the "no wild upswing" property.
#[test]
fn bba1_oscillation_free_100_segments() {
    let mut last_idx = 0usize;
    let mut violations = 0usize;

    for i in 0..100usize {
        let buf_secs = 12.0 + 13.0 * ((i as f64 * 0.3_f64).sin().abs());
        let buf = Duration::from_secs_f64(buf_secs);
        let new_idx = select_variant(buf, &params(), &variants_3());

        if new_idx > last_idx + 1 {
            violations += 1;
        }
        last_idx = new_idx;
    }

    assert_eq!(
        violations, 0,
        "BBA-1 must not jump more than 1 rung upward per step over 100 segments"
    );
}
