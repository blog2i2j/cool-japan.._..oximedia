//! Channel folding and unfolding: convert between mono, stereo, 5.1, and 7.1.
//!
//! This module provides free functions for common fold-down and unfold-up
//! operations as well as the [`ChannelLayout`] enum and the higher-level
//! [`fold_channels`] / [`unfold_channels`] dispatch functions.
//!
//! ## Fold matrices
//!
//! **5.1 → Stereo** (ITU-R BS.775 standard fold-down):
//! ```text
//! L'  = L  + 0.707·C + 0.707·Ls
//! R'  = R  + 0.707·C + 0.707·Rs
//! (LFE discarded)
//! ```
//!
//! **7.1 → 5.1**:
//! ```text
//! Ls' = Ls + 0.707·Lrs
//! Rs' = Rs + 0.707·Rrs
//! ```

// ---------------------------------------------------------------------------
// ChannelLayout
// ---------------------------------------------------------------------------

/// Identifies a channel configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelLayout {
    /// Single channel.
    Mono,
    /// Left + Right.
    Stereo,
    /// ITU-R BS.775: L, R, C, LFE, Ls, Rs (6 channels).
    Surround51,
    /// Extended: L, R, C, LFE, Lss, Rss, Lrs, Rrs (8 channels).
    Surround71,
    /// Arbitrary channel count.
    Custom(usize),
}

impl ChannelLayout {
    /// Number of channels for this layout.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
            Self::Custom(n) => *n,
        }
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Mono => "Mono",
            Self::Stereo => "Stereo",
            Self::Surround51 => "5.1 Surround",
            Self::Surround71 => "7.1 Surround",
            Self::Custom(_) => "Custom",
        }
    }
}

// ---------------------------------------------------------------------------
// stereo_to_mono
// ---------------------------------------------------------------------------

/// Fold stereo interleaved audio to mono by averaging each L/R pair.
///
/// Input: `[L0, R0, L1, R1, …]`
/// Output: `[M0, M1, …]` where `Mn = (Ln + Rn) / 2`.
///
/// Returns an empty `Vec` if `stereo` is empty.
/// Odd-length input: the final unpaired sample is averaged with 0.
#[must_use]
pub fn stereo_to_mono(stereo: &[f32]) -> Vec<f32> {
    let frames = (stereo.len() + 1) / 2;
    let mut mono = Vec::with_capacity(frames);
    let mut i = 0;
    while i + 1 < stereo.len() {
        mono.push((stereo[i] + stereo[i + 1]) * 0.5);
        i += 2;
    }
    if i < stereo.len() {
        mono.push(stereo[i] * 0.5);
    }
    mono
}

// ---------------------------------------------------------------------------
// mono_to_stereo
// ---------------------------------------------------------------------------

/// Unfold mono audio to stereo by duplicating each sample.
///
/// Output: `[L0, R0, L1, R1, …]` where `Ln = Rn = monoN`.
#[must_use]
pub fn mono_to_stereo(mono: &[f32]) -> Vec<f32> {
    let mut stereo = Vec::with_capacity(mono.len() * 2);
    for &s in mono {
        stereo.push(s);
        stereo.push(s);
    }
    stereo
}

// ---------------------------------------------------------------------------
// surround51_to_stereo
// ---------------------------------------------------------------------------

/// Fold 5.1 interleaved audio to stereo using the ITU-R BS.775 matrix.
///
/// Input frame layout: `[L, R, C, LFE, Ls, Rs]`
///
/// ```text
/// L' = L  + 0.707·C + 0.707·Ls
/// R' = R  + 0.707·C + 0.707·Rs
/// ```
///
/// Returns `Err` if `audio_51.len()` is not a multiple of 6.
pub fn surround51_to_stereo(audio_51: &[f32]) -> Result<Vec<f32>, String> {
    if audio_51.len() % 6 != 0 {
        return Err(format!(
            "Input length {} is not a multiple of 6 (5.1 channels)",
            audio_51.len()
        ));
    }

    const SQRT2_RECIP: f32 = std::f32::consts::FRAC_1_SQRT_2; // 0.7071…

    let frames = audio_51.len() / 6;
    let mut stereo = Vec::with_capacity(frames * 2);

    for frame in audio_51.chunks_exact(6) {
        let (l, r, c, _lfe, ls, rs) = (frame[0], frame[1], frame[2], frame[3], frame[4], frame[5]);
        stereo.push(l + SQRT2_RECIP * c + SQRT2_RECIP * ls);
        stereo.push(r + SQRT2_RECIP * c + SQRT2_RECIP * rs);
    }

    Ok(stereo)
}

// ---------------------------------------------------------------------------
// surround71_to_51
// ---------------------------------------------------------------------------

/// Fold 7.1 interleaved audio to 5.1.
///
/// Input frame layout: `[L, R, C, LFE, Lss, Rss, Lrs, Rrs]`
///
/// ```text
/// Ls' = Lss + 0.707·Lrs
/// Rs' = Rss + 0.707·Rrs
/// ```
///
/// Returns `Err` if `audio_71.len()` is not a multiple of 8.
pub fn surround71_to_51(audio_71: &[f32]) -> Result<Vec<f32>, String> {
    if audio_71.len() % 8 != 0 {
        return Err(format!(
            "Input length {} is not a multiple of 8 (7.1 channels)",
            audio_71.len()
        ));
    }

    const SQRT2_RECIP: f32 = std::f32::consts::FRAC_1_SQRT_2;

    let frames = audio_71.len() / 8;
    let mut out51 = Vec::with_capacity(frames * 6);

    for frame in audio_71.chunks_exact(8) {
        let (l, r, c, lfe, lss, rss, lrs, rrs) = (
            frame[0], frame[1], frame[2], frame[3],
            frame[4], frame[5], frame[6], frame[7],
        );
        // L, R, C, LFE unchanged; fold rear surround into side surround
        out51.push(l);
        out51.push(r);
        out51.push(c);
        out51.push(lfe);
        out51.push(lss + SQRT2_RECIP * lrs);
        out51.push(rss + SQRT2_RECIP * rrs);
    }

    Ok(out51)
}

// ---------------------------------------------------------------------------
// fold_channels
// ---------------------------------------------------------------------------

/// General downmix dispatcher.
///
/// Supported conversions:
/// - `Mono → Mono` (identity)
/// - `Stereo → Mono`
/// - `Stereo → Stereo` (identity)
/// - `Surround51 → Stereo`
/// - `Surround51 → Mono` (fold to stereo then to mono)
/// - `Surround71 → Surround51`
/// - `Surround71 → Stereo`
/// - `Surround71 → Mono`
///
/// Returns `Err` for unsupported pairs or length mismatches.
pub fn fold_channels(
    input: &[f32],
    from: &ChannelLayout,
    to: &ChannelLayout,
) -> Result<Vec<f32>, String> {
    // Validate input length
    let ch = from.channel_count();
    if ch > 0 && input.len() % ch != 0 {
        return Err(format!(
            "Input length {} is not a multiple of {} channels ({:?})",
            input.len(),
            ch,
            from
        ));
    }

    match (from, to) {
        // Identity
        (a, b) if a == b => Ok(input.to_vec()),

        (ChannelLayout::Stereo, ChannelLayout::Mono) => Ok(stereo_to_mono(input)),

        (ChannelLayout::Surround51, ChannelLayout::Stereo) => surround51_to_stereo(input),

        (ChannelLayout::Surround51, ChannelLayout::Mono) => {
            surround51_to_stereo(input).map(|s| stereo_to_mono(&s))
        }

        (ChannelLayout::Surround71, ChannelLayout::Surround51) => surround71_to_51(input),

        (ChannelLayout::Surround71, ChannelLayout::Stereo) => {
            surround71_to_51(input).and_then(|s51| surround51_to_stereo(&s51))
        }

        (ChannelLayout::Surround71, ChannelLayout::Mono) => {
            surround71_to_51(input)
                .and_then(|s51| surround51_to_stereo(&s51))
                .map(|s| stereo_to_mono(&s))
        }

        _ => Err(format!(
            "Unsupported fold: {:?} → {:?}",
            from, to
        )),
    }
}

// ---------------------------------------------------------------------------
// unfold_channels
// ---------------------------------------------------------------------------

/// General upmix dispatcher.
///
/// Supported conversions:
/// - Any layout → itself (identity)
/// - `Mono → Stereo`
/// - `Stereo → Surround51`: L→L, R→R, C=0, LFE=0, Ls=0, Rs=0
/// - `Mono → Surround51`: mono→stereo then stereo→5.1
///
/// Returns `Err` for unsupported pairs or length mismatches.
pub fn unfold_channels(
    input: &[f32],
    from: &ChannelLayout,
    to: &ChannelLayout,
) -> Result<Vec<f32>, String> {
    let ch = from.channel_count();
    if ch > 0 && input.len() % ch != 0 {
        return Err(format!(
            "Input length {} is not a multiple of {} channels ({:?})",
            input.len(),
            ch,
            from
        ));
    }

    match (from, to) {
        // Identity
        (a, b) if a == b => Ok(input.to_vec()),

        (ChannelLayout::Mono, ChannelLayout::Stereo) => Ok(mono_to_stereo(input)),

        (ChannelLayout::Stereo, ChannelLayout::Surround51) => {
            if input.len() % 2 != 0 {
                return Err(format!(
                    "Input length {} is not a multiple of 2 (stereo)",
                    input.len()
                ));
            }
            let frames = input.len() / 2;
            let mut out = Vec::with_capacity(frames * 6);
            for frame in input.chunks_exact(2) {
                out.push(frame[0]); // L
                out.push(frame[1]); // R
                out.push(0.0);      // C
                out.push(0.0);      // LFE
                out.push(0.0);      // Ls
                out.push(0.0);      // Rs
            }
            Ok(out)
        }

        (ChannelLayout::Mono, ChannelLayout::Surround51) => {
            let stereo = mono_to_stereo(input);
            unfold_channels(&stereo, &ChannelLayout::Stereo, &ChannelLayout::Surround51)
        }

        _ => Err(format!(
            "Unsupported unfold: {:?} → {:?}",
            from, to
        )),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stereo_to_mono_average() {
        let stereo = vec![1.0_f32, 0.0, 0.5, 0.5, -0.5, 0.5];
        let mono = stereo_to_mono(&stereo);
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.5).abs() < 1e-6, "frame0: (1+0)/2=0.5");
        assert!((mono[1] - 0.5).abs() < 1e-6, "frame1: (0.5+0.5)/2=0.5");
        assert!((mono[2] - 0.0).abs() < 1e-6, "frame2: (-0.5+0.5)/2=0");
    }

    #[test]
    fn test_mono_to_stereo_duplicate() {
        let mono = vec![0.4_f32, -0.6, 0.0];
        let stereo = mono_to_stereo(&mono);
        assert_eq!(stereo.len(), 6);
        for i in 0..3 {
            let l = stereo[i * 2];
            let r = stereo[i * 2 + 1];
            assert!(
                (l - mono[i]).abs() < 1e-6 && (r - mono[i]).abs() < 1e-6,
                "frame {i}: L={l}, R={r}, mono={}", mono[i]
            );
        }
    }

    #[test]
    fn test_surround51_to_stereo_formula() {
        const K: f32 = std::f32::consts::FRAC_1_SQRT_2;
        // One frame: L=1, R=0, C=1, LFE=0.5, Ls=0, Rs=0
        let input = vec![1.0_f32, 0.0, 1.0, 0.5, 0.0, 0.0];
        let stereo = surround51_to_stereo(&input).expect("fold should succeed");
        assert_eq!(stereo.len(), 2);
        let expected_l = 1.0 + K * 1.0 + K * 0.0;
        let expected_r = 0.0 + K * 1.0 + K * 0.0;
        assert!(
            (stereo[0] - expected_l).abs() < 1e-5,
            "L' expected {expected_l}, got {}", stereo[0]
        );
        assert!(
            (stereo[1] - expected_r).abs() < 1e-5,
            "R' expected {expected_r}, got {}", stereo[1]
        );
    }

    #[test]
    fn test_surround51_to_stereo_bad_length() {
        let result = surround51_to_stereo(&[0.0_f32; 5]);
        assert!(result.is_err(), "Length 5 should fail (not multiple of 6)");
    }

    #[test]
    fn test_surround71_to_51_fold() {
        const K: f32 = std::f32::consts::FRAC_1_SQRT_2;
        // One 7.1 frame: all zeros except Lss=0.5, Rss=0.3, Lrs=0.4, Rrs=0.2
        let frame = vec![0.0_f32, 0.0, 0.0, 0.0, 0.5, 0.3, 0.4, 0.2];
        let out = surround71_to_51(&frame).expect("should succeed");
        assert_eq!(out.len(), 6);
        // Ls' = 0.5 + K*0.4
        let expected_ls = 0.5 + K * 0.4;
        let expected_rs = 0.3 + K * 0.2;
        assert!(
            (out[4] - expected_ls).abs() < 1e-5,
            "Ls' expected {expected_ls}, got {}", out[4]
        );
        assert!(
            (out[5] - expected_rs).abs() < 1e-5,
            "Rs' expected {expected_rs}, got {}", out[5]
        );
    }

    #[test]
    fn test_fold_channels_mono_identity() {
        let input = vec![0.1_f32, 0.2, 0.3];
        let out = fold_channels(&input, &ChannelLayout::Mono, &ChannelLayout::Mono)
            .expect("identity should succeed");
        assert_eq!(out, input);
    }

    #[test]
    fn test_fold_channels_stereo_to_mono() {
        let input = vec![0.6_f32, 0.4];
        let out = fold_channels(&input, &ChannelLayout::Stereo, &ChannelLayout::Mono)
            .expect("stereo→mono");
        assert_eq!(out.len(), 1);
        assert!((out[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_fold_channels_51_to_stereo() {
        let frame = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0]; // Pure L
        let out = fold_channels(&frame, &ChannelLayout::Surround51, &ChannelLayout::Stereo)
            .expect("5.1→stereo");
        assert_eq!(out.len(), 2);
        assert!((out[0] - 1.0).abs() < 1e-5, "L' should be 1.0, got {}", out[0]);
        assert!(out[1].abs() < 1e-5, "R' should be 0.0, got {}", out[1]);
    }

    #[test]
    fn test_unfold_channels_mono_to_stereo() {
        let mono = vec![0.7_f32];
        let out =
            unfold_channels(&mono, &ChannelLayout::Mono, &ChannelLayout::Stereo).expect("ok");
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.7).abs() < 1e-6);
        assert!((out[1] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_unfold_channels_stereo_to_51() {
        let stereo = vec![0.5_f32, 0.3];
        let out =
            unfold_channels(&stereo, &ChannelLayout::Stereo, &ChannelLayout::Surround51)
                .expect("ok");
        assert_eq!(out.len(), 6);
        assert!((out[0] - 0.5).abs() < 1e-6, "L channel");
        assert!((out[1] - 0.3).abs() < 1e-6, "R channel");
        assert!(out[2].abs() < 1e-6, "C should be 0");
        assert!(out[3].abs() < 1e-6, "LFE should be 0");
        assert!(out[4].abs() < 1e-6, "Ls should be 0");
        assert!(out[5].abs() < 1e-6, "Rs should be 0");
    }

    #[test]
    fn test_fold_channels_bad_length() {
        // 5 samples for 6-channel 5.1 → should error
        let result =
            fold_channels(&[0.0_f32; 5], &ChannelLayout::Surround51, &ChannelLayout::Stereo);
        assert!(result.is_err(), "Odd length should produce an error");
    }

    #[test]
    fn test_channel_layout_name() {
        assert_eq!(ChannelLayout::Mono.name(), "Mono");
        assert_eq!(ChannelLayout::Stereo.name(), "Stereo");
        assert_eq!(ChannelLayout::Surround51.name(), "5.1 Surround");
        assert_eq!(ChannelLayout::Surround71.name(), "7.1 Surround");
    }
}
