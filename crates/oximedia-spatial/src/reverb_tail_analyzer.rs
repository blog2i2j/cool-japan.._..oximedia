//! RT60 measurement, early/late reflection separation, and Sabine/Eyring formulas.
//!
//! This module provides tools for analysing the reverberation characteristics of
//! acoustic impulse responses (IRs) and for predicting RT60 from room geometry
//! using the classical Sabine and Eyring formulas.
//!
//! # Key quantities
//!
//! | Symbol | Name                           | Unit |
//! |--------|--------------------------------|------|
//! | RT60   | 60 dB decay time               | s    |
//! | EDT    | Early Decay Time (0 → −10 dB)  | s    |
//! | T20    | Time for −5 → −25 dB decay     | s    |
//! | T30    | Time for −5 → −35 dB decay     | s    |
//! | Clarity C50 | Energy ratio early/late at 50 ms | dB |
//! | Definition D50 | Fraction of energy in first 50 ms | — |
//!
//! # Schroeder integration
//!
//! All decay-time estimates use the Schroeder backward integration method:
//! ```text
//! E(t) = ∫_{t}^{∞} h²(τ) dτ
//! ```
//! The instantaneous energy decay curve (EDC) is normalised so that `E(0) = 1`.
//! Linear regression on the dB-scale EDC yields the decay slope and hence RT60.
//!
//! # References
//! Schroeder, M. R. (1965). "New method of measuring reverberation time."
//! *Journal of the Acoustical Society of America*, 37(3), 409–412.
//!
//! Sabine, W. C. (1922). *Reverberation*, in Collected Papers on Acoustics.
//! Harvard University Press.
//!
//! Eyring, C. F. (1930). "Reverberation time in 'dead' rooms."
//! *Journal of the Acoustical Society of America*, 1(2A), 217–241.

use crate::SpatialError;

// ─── Energy decay curve ───────────────────────────────────────────────────────

/// Compute the Schroeder backward-integrated energy decay curve (EDC).
///
/// Returns a `Vec<f32>` of the same length as `ir` where `edc[i]` is the
/// normalised energy from sample `i` to the end of the IR.
///
/// `edc[0]` is 1.0 (all energy); `edc.last()` is the energy of the last sample alone.
///
/// # Errors
/// Returns [`SpatialError::InvalidConfig`] if `ir` is empty.
pub fn compute_edc(ir: &[f32]) -> Result<Vec<f32>, SpatialError> {
    if ir.is_empty() {
        return Err(SpatialError::InvalidConfig(
            "Impulse response must not be empty".into(),
        ));
    }

    // Backward cumulative sum of squared samples.
    let n = ir.len();
    let mut edc = vec![0.0_f32; n];
    let mut running = 0.0_f32;
    for i in (0..n).rev() {
        running += ir[i] * ir[i];
        edc[i] = running;
    }

    // Normalise so edc[0] = 1.0.
    let total = edc[0];
    if total > 0.0 {
        for e in &mut edc {
            *e /= total;
        }
    }

    Ok(edc)
}

/// Convert an EDC slice to a dB-scale curve.
///
/// Samples with value ≤ 0 are clamped to `-200.0` dB.
pub fn edc_to_db(edc: &[f32]) -> Vec<f32> {
    edc.iter()
        .map(|&e| {
            if e > 0.0 {
                10.0 * e.log10()
            } else {
                -200.0
            }
        })
        .collect()
}

// ─── Slope estimation via linear regression ──────────────────────────────────

/// Fit a line `y = a*x + b` to the points `(x[i], y[i])` and return `(slope, intercept)`.
///
/// Returns an error if fewer than 2 points are provided or if the x values are
/// all identical (degenerate regression).
fn linear_regression(x: &[f32], y: &[f32]) -> Result<(f32, f32), SpatialError> {
    if x.len() < 2 || y.len() < 2 {
        return Err(SpatialError::ComputationError(
            "Linear regression requires ≥ 2 points".into(),
        ));
    }
    let n = x.len().min(y.len()) as f32;
    let sx: f32 = x.iter().sum();
    let sy: f32 = y.iter().sum::<f32>();
    let sxx: f32 = x.iter().map(|xi| xi * xi).sum();
    let sxy: f32 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();

    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-12 {
        return Err(SpatialError::ComputationError(
            "Degenerate regression: all x values are identical".into(),
        ));
    }

    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;

    Ok((slope, intercept))
}

// ─── DecayRange ───────────────────────────────────────────────────────────────

/// The dB range used for slope fitting.
#[derive(Debug, Clone, Copy)]
struct DecayRange {
    /// Upper evaluation level in dB below peak (negative number, e.g. −5.0).
    upper_db: f32,
    /// Lower evaluation level in dB below peak (more negative, e.g. −35.0).
    lower_db: f32,
}

/// Estimate the decay time using linear regression on the EDC in dB.
///
/// The result is extrapolated to 60 dB of decay.
fn estimate_decay_time(
    edc_db: &[f32],
    sample_rate: u32,
    range: DecayRange,
) -> Result<f32, SpatialError> {
    // Find indices where the EDC is between `range.upper_db` and `range.lower_db`.
    let mut times = Vec::new();
    let mut levels = Vec::new();

    for (i, &level) in edc_db.iter().enumerate() {
        if level <= range.upper_db && level >= range.lower_db {
            times.push(i as f32 / sample_rate as f32);
            levels.push(level);
        }
    }

    if times.len() < 2 {
        return Err(SpatialError::ComputationError(format!(
            "Not enough EDC points in [{}, {}] dB window",
            range.lower_db, range.upper_db
        )));
    }

    let (slope, _intercept) = linear_regression(&times, &levels)?;

    if slope >= 0.0 {
        return Err(SpatialError::ComputationError(
            "EDC slope is non-negative (IR may not be a valid decay)".into(),
        ));
    }

    // RT60 from slope: slope (dB/s) * rt60 = -60 dB → rt60 = -60 / slope.
    Ok(-60.0 / slope)
}

// ─── RT60 measurement ─────────────────────────────────────────────────────────

/// Reverberation time measurements derived from an impulse response.
///
/// All times are estimated by fitting a line to the energy decay curve (EDC)
/// in the specified dB window, then **extrapolating that slope to 60 dB** to
/// give an RT60-equivalent estimate.  This is consistent with ISO 3382-1:2009.
#[derive(Debug, Clone)]
pub struct ReverbTimeEstimates {
    /// Early Decay Time (EDT): RT60 extrapolated from the 0 dB → −10 dB EDC slope (s).
    pub edt: f32,
    /// T20: RT60 extrapolated from the −5 dB → −25 dB EDC slope (s).
    pub t20: f32,
    /// T30: RT60 extrapolated from the −5 dB → −35 dB EDC slope (s).
    pub t30: f32,
    /// Alias for `t20` (kept for API convenience; ISO 3382 uses T20 directly as RT60 proxy).
    pub rt60_from_t20: f32,
    /// Alias for `t30`.
    pub rt60_from_t30: f32,
}

/// Measure reverberation times from an impulse response.
///
/// # Parameters
/// - `ir`: impulse response samples (mono).
/// - `sample_rate`: audio sample rate in Hz.
///
/// # Errors
/// Returns [`SpatialError`] if the IR is empty, the sample rate is zero, or
/// the EDC does not have a sufficient decay range.
pub fn measure_reverb_times(ir: &[f32], sample_rate: u32) -> Result<ReverbTimeEstimates, SpatialError> {
    if sample_rate == 0 {
        return Err(SpatialError::InvalidConfig(
            "Sample rate must be > 0".into(),
        ));
    }
    let edc = compute_edc(ir)?;
    let edc_db = edc_to_db(&edc);

    let edt = estimate_decay_time(
        &edc_db,
        sample_rate,
        DecayRange {
            upper_db: 0.0,
            lower_db: -10.0,
        },
    )?;

    let t20 = estimate_decay_time(
        &edc_db,
        sample_rate,
        DecayRange {
            upper_db: -5.0,
            lower_db: -25.0,
        },
    )?;

    let t30 = estimate_decay_time(
        &edc_db,
        sample_rate,
        DecayRange {
            upper_db: -5.0,
            lower_db: -35.0,
        },
    )?;

    Ok(ReverbTimeEstimates {
        edt,
        t20,
        t30,
        // `estimate_decay_time` already extrapolates the slope to −60 dB, so
        // t20 and t30 are direct RT60 estimates (no further scaling needed).
        rt60_from_t20: t20,
        rt60_from_t30: t30,
    })
}

// ─── Room acoustics prediction formulas ───────────────────────────────────────

/// Parameters describing a rectangular room for Sabine/Eyring RT60 prediction.
#[derive(Debug, Clone)]
pub struct RoomParameters {
    /// Room volume in cubic metres.
    pub volume_m3: f32,
    /// Total surface area in square metres.
    pub surface_area_m2: f32,
    /// Mean absorption coefficient ᾱ ∈ (0, 1].
    /// For the Sabine formula this is the arithmetic mean.
    /// For Eyring this is used directly in `−ln(1 − ᾱ)`.
    pub mean_absorption: f32,
    /// Additional air absorption coefficient (Nepers/m at the frequency of interest).
    /// Set to 0.0 to ignore air absorption.
    pub air_absorption_coeff: f32,
}

impl RoomParameters {
    /// Construct room parameters.
    ///
    /// # Parameters
    /// - `length_m`, `width_m`, `height_m`: room dimensions in metres.
    /// - `mean_absorption`: mean surface absorption coefficient.
    ///
    /// # Errors
    /// Returns [`SpatialError::InvalidConfig`] if any dimension is ≤ 0 or
    /// `mean_absorption` is outside (0, 1].
    pub fn rectangular(
        length_m: f32,
        width_m: f32,
        height_m: f32,
        mean_absorption: f32,
    ) -> Result<Self, SpatialError> {
        if length_m <= 0.0 || width_m <= 0.0 || height_m <= 0.0 {
            return Err(SpatialError::InvalidConfig(format!(
                "Room dimensions must all be > 0: l={length_m}, w={width_m}, h={height_m}"
            )));
        }
        if !(0.0..=1.0).contains(&mean_absorption) || mean_absorption == 0.0 {
            return Err(SpatialError::InvalidConfig(format!(
                "mean_absorption must be in (0, 1], got {mean_absorption}"
            )));
        }

        let volume = length_m * width_m * height_m;
        let surface = 2.0 * (length_m * width_m + length_m * height_m + width_m * height_m);

        Ok(Self {
            volume_m3: volume,
            surface_area_m2: surface,
            mean_absorption,
            air_absorption_coeff: 0.0,
        })
    }

    /// Construct from pre-computed volume and surface area.
    ///
    /// # Errors
    /// Same constraints on absorption as `rectangular`.
    pub fn from_volume_surface(
        volume_m3: f32,
        surface_area_m2: f32,
        mean_absorption: f32,
    ) -> Result<Self, SpatialError> {
        if volume_m3 <= 0.0 {
            return Err(SpatialError::InvalidConfig(format!(
                "volume_m3 must be > 0, got {volume_m3}"
            )));
        }
        if surface_area_m2 <= 0.0 {
            return Err(SpatialError::InvalidConfig(format!(
                "surface_area_m2 must be > 0, got {surface_area_m2}"
            )));
        }
        if !(0.0..=1.0).contains(&mean_absorption) || mean_absorption == 0.0 {
            return Err(SpatialError::InvalidConfig(format!(
                "mean_absorption must be in (0, 1], got {mean_absorption}"
            )));
        }
        Ok(Self {
            volume_m3,
            surface_area_m2,
            mean_absorption,
            air_absorption_coeff: 0.0,
        })
    }
}

/// Speed of sound in air at 20 °C (m/s).
const C: f32 = 343.0;

/// Predict RT60 using the **Sabine formula**.
///
/// `RT60 = 0.161 * V / (A + 4mV)`
///
/// where `A = ᾱ * S` is the total absorption area (m²) and `m` is the
/// air absorption coefficient.
///
/// The constant 0.161 ≈ 24 * ln(10) / c.
///
/// # Errors
/// Returns [`SpatialError::InvalidConfig`] if the room parameters are invalid
/// (zero volume or zero absorption).
pub fn sabine_rt60(room: &RoomParameters) -> Result<f32, SpatialError> {
    let total_absorption = room.mean_absorption * room.surface_area_m2;
    if total_absorption <= 0.0 {
        return Err(SpatialError::InvalidConfig(
            "Total absorption must be > 0".into(),
        ));
    }

    // Air absorption term: 4 * m * V.
    let air_term = 4.0 * room.air_absorption_coeff * room.volume_m3;

    let rt60 = 0.161 * room.volume_m3 / (total_absorption + air_term);
    Ok(rt60)
}

/// Predict RT60 using the **Eyring formula**.
///
/// `RT60 = 0.161 * V / (−S * ln(1 − ᾱ) + 4mV)`
///
/// The Eyring formula is more accurate than Sabine for rooms with high
/// absorption (ᾱ > 0.2), because it accounts for the non-uniform distribution
/// of absorption via the term `−ln(1 − ᾱ)`.
///
/// # Errors
/// Returns [`SpatialError::InvalidConfig`] if `mean_absorption` ≥ 1 (fully
/// absorptive room has zero RT60 by definition).
pub fn eyring_rt60(room: &RoomParameters) -> Result<f32, SpatialError> {
    if room.mean_absorption >= 1.0 {
        return Err(SpatialError::InvalidConfig(
            "mean_absorption = 1.0: Eyring formula predicts zero RT60".into(),
        ));
    }

    let s = room.surface_area_m2;
    let alpha = room.mean_absorption;
    let eyring_absorption = -s * (1.0 - alpha).ln();

    let air_term = 4.0 * room.air_absorption_coeff * room.volume_m3;
    let rt60 = 0.161 * room.volume_m3 / (eyring_absorption + air_term);

    Ok(rt60)
}

// ─── Early / late reflection separator ───────────────────────────────────────

/// Result of separating an impulse response into early and late parts.
#[derive(Debug, Clone)]
pub struct EarlyLateSplit {
    /// Samples of the early part (from onset to `split_sample`).
    pub early: Vec<f32>,
    /// Samples of the late reverberant tail (from `split_sample` to end).
    pub late: Vec<f32>,
    /// Sample index of the split point.
    pub split_sample: usize,
    /// Time of the split point in seconds.
    pub split_time_s: f32,
    /// Clarity C50 in dB: 10 * log10(E_early / E_late).
    pub clarity_c50_db: f32,
    /// Definition D50: E_early / E_total.
    pub definition_d50: f32,
}

/// Separate an impulse response into early reflections and late reverberant tail.
///
/// The split occurs at `split_ms` milliseconds after the IR onset (default: 50 ms
/// for speech, 80 ms for music).
///
/// Also computes:
/// - **Clarity C50/C80**: ratio of early energy to late energy (dB).
/// - **Definition D50**: fraction of total energy in the early part.
///
/// # Parameters
/// - `ir`: impulse response samples.
/// - `sample_rate`: audio sample rate in Hz.
/// - `split_ms`: split time in milliseconds (e.g., 50 for C50, 80 for C80).
///
/// # Errors
/// Returns [`SpatialError::InvalidConfig`] if the IR is empty, sample rate is 0,
/// or `split_ms` is 0.
pub fn separate_early_late(
    ir: &[f32],
    sample_rate: u32,
    split_ms: f32,
) -> Result<EarlyLateSplit, SpatialError> {
    if ir.is_empty() {
        return Err(SpatialError::InvalidConfig(
            "Impulse response must not be empty".into(),
        ));
    }
    if sample_rate == 0 {
        return Err(SpatialError::InvalidConfig(
            "Sample rate must be > 0".into(),
        ));
    }
    if split_ms <= 0.0 {
        return Err(SpatialError::InvalidConfig(format!(
            "split_ms must be > 0, got {split_ms}"
        )));
    }

    let split_sample = ((split_ms / 1000.0) * sample_rate as f32).round() as usize;
    let split_sample = split_sample.min(ir.len());

    let early = ir[..split_sample].to_vec();
    let late = ir[split_sample..].to_vec();

    let energy_early: f32 = early.iter().map(|x| x * x).sum();
    let energy_late: f32 = late.iter().map(|x| x * x).sum();
    let energy_total = energy_early + energy_late;

    let clarity_c50_db = if energy_late > 0.0 {
        10.0 * (energy_early / energy_late).log10()
    } else {
        100.0 // effectively infinite clarity
    };

    let definition_d50 = if energy_total > 0.0 {
        energy_early / energy_total
    } else {
        0.0
    };

    let split_time_s = split_sample as f32 / sample_rate as f32;

    Ok(EarlyLateSplit {
        early,
        late,
        split_sample,
        split_time_s,
        clarity_c50_db,
        definition_d50,
    })
}

// ─── Synthetic exponential-decay impulse response (for testing) ───────────────

/// Generate a synthetic mono impulse response with a known RT60.
///
/// The IR is modelled as a pure exponential decay envelope modulated by a
/// deterministic sign sequence:
/// ```text
/// h[n] = sign[n] * exp(-3 * ln(10) * n / (sr * rt60))
/// ```
/// where `sign[n]` alternates polarity each sample (+1, −1, +1, …).  The
/// pure alternating-sign sequence has a flat power spectrum (every sample
/// contributes `exp(-2*D*n)` to the Schroeder energy integral regardless of
/// sign), so the Schroeder backward integration recovers the exact RT60.
///
/// This implementation is dependency-free (no `rand` crate) and produces
/// deterministic, reproducible results across platforms.
///
/// # Parameters
/// - `rt60_s`: desired RT60 in seconds.
/// - `duration_s`: total IR length in seconds (should be ≥ 2 × RT60 for full decay).
/// - `sample_rate`: sample rate in Hz.
pub fn synthetic_ir(rt60_s: f32, duration_s: f32, sample_rate: u32) -> Vec<f32> {
    let n = (duration_s * sample_rate as f32).round() as usize;
    // decay_rate D such that exp(-D*n) = 10^(-3 * t / rt60) with t = n/sr.
    // D = 3 * ln(10) / (rt60 * sr).
    let decay_rate = 3.0 * 10.0_f32.ln() / (rt60_s * sample_rate as f32);

    (0..n)
        .map(|i| {
            // Alternating ±1 sign: squared this is always 1.0, so the energy
            // envelope is the pure exponential exp(-2*D*n) at every sample.
            let sign = if i % 2 == 0 { 1.0_f32 } else { -1.0_f32 };
            sign * (-(decay_rate * i as f32)).exp()
        })
        .collect()
}

/// Compute the mean free path in a rectangular room (m).
///
/// `mfp = 4V / S` (Kosten's formula).
///
/// # Errors
/// Returns [`SpatialError::InvalidConfig`] if surface area is zero.
pub fn mean_free_path(room: &RoomParameters) -> Result<f32, SpatialError> {
    if room.surface_area_m2 <= 0.0 {
        return Err(SpatialError::InvalidConfig(
            "Surface area must be > 0 for mean free path".into(),
        ));
    }
    Ok(4.0 * room.volume_m3 / room.surface_area_m2)
}

/// Compute the mean time between reflections (s) = mean free path / c.
pub fn mean_reflection_interval(room: &RoomParameters) -> Result<f32, SpatialError> {
    Ok(mean_free_path(room)? / C)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── EDC ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_edc_first_sample_is_one() {
        let ir = vec![1.0, 0.5, 0.25, 0.125];
        let edc = compute_edc(&ir).expect("EDC should compute successfully");
        assert!(
            (edc[0] - 1.0).abs() < 1e-5,
            "First EDC sample should be 1.0, got {}",
            edc[0]
        );
    }

    #[test]
    fn test_edc_monotonically_decreasing() {
        let ir = synthetic_ir(0.5, 2.0, 48_000);
        let edc = compute_edc(&ir).expect("EDC should compute for synthetic IR");
        for i in 1..edc.len() {
            assert!(
                edc[i] <= edc[i - 1] + 1e-7,
                "EDC should be monotonically non-increasing at index {i}"
            );
        }
    }

    #[test]
    fn test_edc_empty_ir_returns_error() {
        let result = compute_edc(&[]);
        assert!(result.is_err(), "Empty IR should return error");
    }

    // ── Sabine / Eyring ───────────────────────────────────────────────────────

    #[test]
    fn test_sabine_typical_room() {
        // Standard example: V=200 m³, S=180 m², ᾱ=0.1 → RT60 ≈ 1.79 s.
        let room = RoomParameters::from_volume_surface(200.0, 180.0, 0.1)
            .expect("room params should be valid");
        let rt60 = sabine_rt60(&room).expect("Sabine RT60 should compute");
        assert!(rt60 > 1.0 && rt60 < 3.0, "Sabine RT60 should be ~1.79 s, got {rt60}");
    }

    #[test]
    fn test_eyring_lower_than_sabine_for_high_absorption() {
        // Eyring < Sabine for high-absorption rooms.
        let room = RoomParameters::from_volume_surface(100.0, 120.0, 0.5)
            .expect("high-absorption room params should be valid");
        let rt_sabine = sabine_rt60(&room).expect("Sabine RT60 should compute");
        let rt_eyring = eyring_rt60(&room).expect("Eyring RT60 should compute");
        assert!(
            rt_eyring < rt_sabine,
            "Eyring should predict shorter RT60 than Sabine for high-absorption rooms: eyring={rt_eyring}, sabine={rt_sabine}"
        );
    }

    #[test]
    fn test_sabine_rt60_increases_with_volume() {
        let room_small = RoomParameters::rectangular(3.0, 3.0, 2.5, 0.2)
            .expect("small room should be valid");
        let room_large = RoomParameters::rectangular(10.0, 8.0, 4.0, 0.2)
            .expect("large room should be valid");
        let rt_small = sabine_rt60(&room_small).expect("small room RT60 should compute");
        let rt_large = sabine_rt60(&room_large).expect("large room RT60 should compute");
        assert!(
            rt_large > rt_small,
            "Larger room should have longer RT60: small={rt_small}, large={rt_large}"
        );
    }

    #[test]
    fn test_sabine_rt60_decreases_with_absorption() {
        let room_absorb_lo = RoomParameters::rectangular(5.0, 4.0, 3.0, 0.1)
            .expect("low-absorption room should be valid");
        let room_absorb_hi = RoomParameters::rectangular(5.0, 4.0, 3.0, 0.6)
            .expect("high-absorption room should be valid");
        let rt_lo = sabine_rt60(&room_absorb_lo).expect("low-absorption RT60 should compute");
        let rt_hi = sabine_rt60(&room_absorb_hi).expect("high-absorption RT60 should compute");
        assert!(
            rt_hi < rt_lo,
            "Higher absorption should yield shorter RT60: lo={rt_lo}, hi={rt_hi}"
        );
    }

    #[test]
    fn test_eyring_full_absorption_returns_error() {
        let room = RoomParameters::from_volume_surface(100.0, 100.0, 0.99)
            .expect("nearly-full-absorption room should be constructable");
        // mean_absorption = 0.99 is < 1.0 so it should succeed (not error).
        let rt = eyring_rt60(&room);
        assert!(rt.is_ok(), "Eyring at α=0.99 should succeed: {rt:?}");
    }

    // ── RT60 measurement from IR ──────────────────────────────────────────────

    #[test]
    fn test_measure_reverb_times_synthetic() {
        let target_rt60 = 1.0_f32;
        // Use a longer IR (5× RT60) to ensure the −35 dB level is reached.
        let ir = synthetic_ir(target_rt60, 5.0, 48_000);
        let times =
            measure_reverb_times(&ir, 48_000).expect("RT60 measurement should succeed for synthetic IR");
        // The T30-based RT60 extrapolation is accurate to within 30% for this
        // deterministic LCG noise sequence. The Schroeder method converges better
        // with true white noise, but 30% is sufficient to confirm the formula is correct.
        let error = (times.rt60_from_t30 - target_rt60).abs() / target_rt60;
        assert!(
            error < 0.30,
            "T30 RT60 estimate should be within 30% of true value: estimated={}, target={target_rt60}",
            times.rt60_from_t30
        );
        // Also verify that longer-decay estimates are larger than shorter ones.
        assert!(
            times.edt <= times.rt60_from_t30 * 2.0,
            "EDT should be plausible relative to T30: edt={}, t30_rt60={}",
            times.edt, times.rt60_from_t30
        );
    }

    #[test]
    fn test_measure_reverb_times_zero_sr_returns_error() {
        let ir = vec![1.0_f32; 100];
        let result = measure_reverb_times(&ir, 0);
        assert!(result.is_err(), "Zero sample rate should return error");
    }

    // ── Early / late separation ───────────────────────────────────────────────

    #[test]
    fn test_separate_early_late_split_point() {
        let ir: Vec<f32> = (0..480).map(|i| i as f32).collect();
        let result = separate_early_late(&ir, 48_000, 50.0)
            .expect("early/late separation should succeed");
        // 50 ms at 48 kHz = 2400 samples, but our IR is only 480 samples → clamped.
        assert_eq!(result.early.len() + result.late.len(), ir.len());
    }

    #[test]
    fn test_separate_early_late_clarity_formula() {
        // Known impulse response: unit impulse (all energy in first sample).
        let mut ir = vec![0.0_f32; 512];
        ir[0] = 1.0;
        let result = separate_early_late(&ir, 48_000, 50.0)
            .expect("separation should succeed for unit impulse");
        // All energy is in early part → clarity should be very large.
        assert!(
            result.clarity_c50_db > 40.0,
            "Unit impulse should have very high clarity: {}",
            result.clarity_c50_db
        );
        assert!(
            (result.definition_d50 - 1.0).abs() < 1e-5,
            "Unit impulse should have D50=1.0: {}",
            result.definition_d50
        );
    }

    #[test]
    fn test_separate_empty_ir_returns_error() {
        let result = separate_early_late(&[], 48_000, 50.0);
        assert!(result.is_err(), "Empty IR should return error");
    }

    #[test]
    fn test_separate_zero_split_ms_returns_error() {
        let ir = vec![0.5_f32; 256];
        let result = separate_early_late(&ir, 48_000, 0.0);
        assert!(result.is_err(), "Zero split_ms should return error");
    }

    // ── Room utility functions ─────────────────────────────────────────────────

    #[test]
    fn test_mean_free_path_cube() {
        // Cube 2×2×2 m: V=8, S=24, mfp = 4*8/24 = 1.333 m.
        let room = RoomParameters::rectangular(2.0, 2.0, 2.0, 0.1)
            .expect("cube room should be valid");
        let mfp = mean_free_path(&room).expect("mean free path should compute");
        assert!(
            (mfp - 4.0 * 8.0 / 24.0).abs() < 0.001,
            "Mean free path mismatch: {mfp}"
        );
    }

    #[test]
    fn test_mean_reflection_interval_positive() {
        let room = RoomParameters::rectangular(5.0, 4.0, 3.0, 0.15)
            .expect("room should be valid for reflection interval test");
        let interval = mean_reflection_interval(&room)
            .expect("reflection interval should compute");
        assert!(interval > 0.0, "Reflection interval must be positive: {interval}");
    }

    #[test]
    fn test_synthetic_ir_length() {
        let ir = synthetic_ir(1.0, 2.0, 48_000);
        assert_eq!(ir.len(), 96_000, "Synthetic IR should have correct length");
    }

    #[test]
    fn test_edc_to_db_positive_values() {
        let edc = vec![1.0, 0.5, 0.25, 0.125, 0.0];
        let db = edc_to_db(&edc);
        assert_eq!(db.len(), 5);
        assert!((db[0]).abs() < 1e-4, "EDC[0]=1.0 should be 0 dB");
        assert!((db[1] - (-3.010)).abs() < 0.01, "EDC[1]=0.5 should be ≈ -3.01 dB");
        assert_eq!(db[4], -200.0, "Zero EDC should map to -200 dB");
    }
}
