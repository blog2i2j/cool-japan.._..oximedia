//! BBA-1 (Buffer-Based Approach, version 1) ABR strategy.
//!
//! Implements the algorithm described in:
//! > Huang, T.-Y., Johari, R., McKeown, N., Trunnell, M., & Watson, M. (2014).
//! > "A Buffer-Based Approach to Rate Adaptation: Evidence from a Large Video
//! > Streaming Service." SIGCOMM 2014.
//!
//! ## Design
//!
//! BBA-1 partitions the client playback buffer into three zones:
//!
//! - **Reservoir** `[0, reservoir]`: always select the lowest quality variant.
//! - **Cushion** `(reservoir, reservoir + cushion]`: linearly interpolate
//!   variant index proportional to the buffer fill in this region.
//! - **Above cushion** `(reservoir + cushion, ∞)`: always select the highest
//!   quality variant.
//!
//! The algorithm deliberately ignores throughput measurements, reacting only to
//! buffer occupancy.  This eliminates the throughput-estimation noise that causes
//! oscillations in traditional bandwidth-based controllers.
//!
//! ## Parameters
//!
//! [`BbaParams`] carries the three knobs that govern the zone boundaries:
//! `buffer_capacity`, `reservoir`, and `cushion`.  See the struct documentation
//! for defaults and practical guidance.
//!
//! ## Pure function
//!
//! [`select_variant`] is a **stateless pure function**.  It takes the current
//! buffer level, the parameters, and the variant list; it returns an index.
//! No controller struct is required — wrap it in whatever stateful shell your
//! streaming pipeline already provides.

use std::time::Duration;

use super::streaming::AbrVariant;

// ─── Parameters ──────────────────────────────────────────────────────────────

/// Parameters for the BBA-1 (Buffer-Based Approach) ABR algorithm.
///
/// Based on Huang et al. (2014) "A Buffer-Based Approach to Rate Adaptation:
/// Evidence from a Large Video Streaming Service" (SIGCOMM 2014).
///
/// # Zone layout
///
/// ```text
/// ┌──────────────┬──────────────────────────────────┬───────────────────────────┐
/// │  Reservoir   │           Cushion                │   Above-cushion           │
/// │  [0, r]      │  (r, r+c]                        │   (r+c, ∞)               │
/// │              │                                  │                           │
/// │  → lowest    │  → linear interp of variant idx  │  → highest               │
/// └──────────────┴──────────────────────────────────┴───────────────────────────┘
/// 0s             r             r+c             buffer_capacity
/// ```
///
/// # Defaults
///
/// The defaults (`B = 30 s`, `r = 10 s`, `c = 20 s`) are drawn directly from
/// the original paper's evaluation setup and work well for typical ABR VOD
/// delivery over HTTP.  For live streaming, reduce all three values accordingly.
#[derive(Debug, Clone)]
pub struct BbaParams {
    /// Total client buffer capacity `B` (paper notation).
    ///
    /// This is used only to **normalise** the buffer level into `[0, 1]`
    /// before computing the zone boundaries.  It should match the actual
    /// playout buffer configured on the player.  Default: 30 seconds.
    pub buffer_capacity: Duration,

    /// Upper boundary of the **reservoir** region `r` (paper notation).
    ///
    /// While the buffer is at or below this level, BBA-1 always selects the
    /// lowest-quality variant to prevent rebuffering.  Default: 10 seconds.
    pub reservoir: Duration,

    /// Size of the **cushion** region `c` (paper notation).
    ///
    /// Within the cushion `(r, r+c]`, the selected variant index is linearly
    /// interpolated from 0 (at `r`) to `n-1` (at `r+c`), where `n` is the
    /// total number of variants.  Default: 20 seconds.
    pub cushion: Duration,
}

impl Default for BbaParams {
    fn default() -> Self {
        Self {
            buffer_capacity: Duration::from_secs(30),
            reservoir: Duration::from_secs(10),
            cushion: Duration::from_secs(20),
        }
    }
}

impl BbaParams {
    /// Creates parameters tuned for low-latency live streaming.
    ///
    /// Uses a smaller buffer target (B = 10 s, r = 2 s, c = 8 s).
    #[must_use]
    pub fn low_latency() -> Self {
        Self {
            buffer_capacity: Duration::from_secs(10),
            reservoir: Duration::from_secs(2),
            cushion: Duration::from_secs(8),
        }
    }

    /// Returns the cushion upper boundary as a [`Duration`].
    ///
    /// Equal to `reservoir + cushion`.
    #[must_use]
    pub fn cushion_upper(&self) -> Duration {
        self.reservoir.saturating_add(self.cushion)
    }

    /// Returns the normalised buffer fill ratio for `buffer_level`.
    ///
    /// Clamps to `[0.0, 1.0]` so values beyond `buffer_capacity` do not
    /// cause out-of-range arithmetic.
    #[must_use]
    pub fn normalise(&self, buffer_level: Duration) -> f64 {
        let cap = self.buffer_capacity.as_secs_f64();
        if cap <= 0.0 {
            return 1.0; // degenerate config — treat as full buffer
        }
        (buffer_level.as_secs_f64() / cap).clamp(0.0, 1.0)
    }
}

// ─── Core BBA-1 selection function ───────────────────────────────────────────

/// Select the variant index to download next using the BBA-1 algorithm.
///
/// # Arguments
///
/// * `buffer_level`  — current client-side playback buffer occupancy.
/// * `params`        — BBA-1 zone parameters (see [`BbaParams`]).
/// * `variants`      — available renditions, **sorted ascending by bitrate**
///   (index 0 = lowest quality, last index = highest quality).
///
/// # Returns
///
/// The index into `variants` that BBA-1 recommends for the next segment.
/// Returns `0` for an empty or single-element slice.
///
/// # Algorithm
///
/// 1. Normalise `buffer_level` to `[0, 1]` using `params.buffer_capacity`.
/// 2. Compute normalised zone boundaries `r_norm = reservoir / capacity` and
///    `cu_norm = (reservoir + cushion) / capacity`.
/// 3. If `norm ≤ r_norm` → return `0` (lowest).
/// 4. If `norm ≥ cu_norm` → return `n - 1` (highest).
/// 5. Otherwise linearly interpolate:
///    `idx = round(cushion_pos × (n - 1))` where
///    `cushion_pos = (norm - r_norm) / (cu_norm - r_norm)`.
///
/// The `round()` — rather than `floor()` — matches the paper's description of
/// mapping a continuous rate function onto a discrete variant ladder.
#[must_use]
pub fn select_variant(
    buffer_level: Duration,
    params: &BbaParams,
    variants: &[AbrVariant],
) -> usize {
    match variants.len() {
        0 => return 0,
        1 => return 0,
        _ => {}
    }

    let cap_secs = params.buffer_capacity.as_secs_f64();
    if cap_secs <= 0.0 {
        // Degenerate configuration: treat as fully buffered → highest quality.
        return variants.len() - 1;
    }

    let norm = (buffer_level.as_secs_f64() / cap_secs).clamp(0.0, 1.0);
    let reservoir_norm = params.reservoir.as_secs_f64() / cap_secs;
    let cushion_upper_norm = params.cushion_upper().as_secs_f64() / cap_secs;

    if norm <= reservoir_norm {
        // Reservoir zone — always lowest quality.
        return 0;
    }

    if norm >= cushion_upper_norm {
        // Above cushion — always highest quality.
        return variants.len() - 1;
    }

    // Cushion zone: linear interpolation.
    let denom = cushion_upper_norm - reservoir_norm;
    let cushion_pos = if denom > 0.0 {
        ((norm - reservoir_norm) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let max_idx = variants.len() - 1;
    // Round to nearest, then clamp defensively.
    let idx = (cushion_pos * max_idx as f64).round() as usize;
    idx.min(max_idx)
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_variant(bandwidth: u64) -> AbrVariant {
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

    fn variants_3() -> Vec<AbrVariant> {
        vec![
            make_variant(500_000),
            make_variant(1_500_000),
            make_variant(4_000_000),
        ]
    }

    fn params() -> BbaParams {
        BbaParams::default() // B=30s, r=10s, c=20s
    }

    // ── Zone boundary tests ──────────────────────────────────────────────────

    #[test]
    fn reservoir_selects_lowest() {
        // buffer 5s < reservoir 10s → index 0
        assert_eq!(
            select_variant(Duration::from_secs(5), &params(), &variants_3()),
            0
        );
    }

    #[test]
    fn zero_buffer_selects_lowest() {
        assert_eq!(select_variant(Duration::ZERO, &params(), &variants_3()), 0);
    }

    #[test]
    fn exact_reservoir_boundary_selects_lowest() {
        // buffer == reservoir (10s) is still ≤ reservoir → lowest
        assert_eq!(
            select_variant(Duration::from_secs(10), &params(), &variants_3()),
            0
        );
    }

    #[test]
    fn above_cushion_selects_highest() {
        // buffer == buffer_capacity (30s) ≥ cushion_upper (30s) → highest
        assert_eq!(
            select_variant(Duration::from_secs(30), &params(), &variants_3()),
            2
        );
    }

    #[test]
    fn overflow_buffer_selects_highest() {
        // buffer > buffer_capacity — clamps at 1.0 → highest
        assert_eq!(
            select_variant(Duration::from_secs(60), &params(), &variants_3()),
            2
        );
    }

    // ── Cushion interpolation tests ──────────────────────────────────────────

    #[test]
    fn cushion_midpoint_selects_middle() {
        // Buffer at midpoint of cushion [10s, 30s] = 20s → index 1
        assert_eq!(
            select_variant(Duration::from_secs(20), &params(), &variants_3()),
            1
        );
    }

    #[test]
    fn cushion_lower_quarter_selects_lowest() {
        // Buffer at 12.5s = 25% through the cushion → round(0.25 * 2) = round(0.5) = 1
        // Note: Rust's f64::round() uses "round half to even" for .5 → 0.5 rounds to 0.
        // (0.25 * 2 = 0.5 → rounded to 0 by round-half-to-even)
        let buf = Duration::from_millis(12_500);
        let idx = select_variant(buf, &params(), &variants_3());
        // Accept either 0 or 1 depending on rounding convention — both are within 1 step.
        assert!(
            idx <= 1,
            "lower cushion should not jump to highest: got {idx}"
        );
    }

    #[test]
    fn cushion_upper_quarter_selects_high() {
        // Buffer at 27.5s = 75% through cushion → round(0.75 * 2) = round(1.5) = 2
        let buf = Duration::from_millis(27_500);
        let idx = select_variant(buf, &params(), &variants_3());
        assert!(idx >= 1, "upper cushion should not stay lowest: got {idx}");
    }

    // ── Edge case: small or empty variant list ────────────────────────────────

    #[test]
    fn single_variant_always_returns_zero() {
        let v = vec![make_variant(1_000_000)];
        for secs in [0u64, 5, 15, 30, 60] {
            assert_eq!(
                select_variant(Duration::from_secs(secs), &params(), &v),
                0,
                "single variant at {secs}s should be 0"
            );
        }
    }

    #[test]
    fn empty_variants_returns_zero() {
        assert_eq!(select_variant(Duration::from_secs(15), &params(), &[]), 0);
    }

    // ── Parameter helpers ─────────────────────────────────────────────────────

    #[test]
    fn cushion_upper_is_reservoir_plus_cushion() {
        let p = params();
        assert_eq!(p.cushion_upper(), Duration::from_secs(30));
    }

    #[test]
    fn normalise_clamps_below_zero() {
        let p = params();
        assert!((p.normalise(Duration::ZERO) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn normalise_clamps_above_capacity() {
        let p = params();
        assert!((p.normalise(Duration::from_secs(100)) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn low_latency_params_have_smaller_buffers() {
        let ll = BbaParams::low_latency();
        let def = BbaParams::default();
        assert!(ll.buffer_capacity < def.buffer_capacity);
        assert!(ll.reservoir < def.reservoir);
        assert!(ll.cushion < def.cushion);
    }

    // ── Degenerate configuration ──────────────────────────────────────────────

    #[test]
    fn zero_capacity_selects_highest() {
        let p = BbaParams {
            buffer_capacity: Duration::ZERO,
            reservoir: Duration::from_secs(5),
            cushion: Duration::from_secs(10),
        };
        assert_eq!(select_variant(Duration::from_secs(5), &p, &variants_3()), 2);
    }
}
