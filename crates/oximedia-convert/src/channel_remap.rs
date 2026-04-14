// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Audio channel remapping: downmix and upmix utilities.
//!
//! Implements standard ITU-R BS.775 / Dolby downmix coefficients for
//! 5.1 → stereo, and a widely-accepted upmix matrix for stereo → 5.1.
//!
//! All inputs and outputs are interleaved f32 PCM samples.

/// Channel indices for standard 5.1 layout (L, R, C, LFE, Ls, Rs).
const CH_L: usize = 0;
const CH_R: usize = 1;
const CH_C: usize = 2;
const CH_LFE: usize = 3;
const CH_LS: usize = 4;
const CH_RS: usize = 5;

/// Number of channels in a 5.1 signal.
pub const CHANNELS_5_1: usize = 6;

/// Number of channels in a stereo signal.
pub const CHANNELS_STEREO: usize = 2;

/// ITU-R BS.775 centre downmix coefficient (−3 dB).
const CENTER_MIX: f32 = std::f32::consts::FRAC_1_SQRT_2; // ≈ 0.707

/// ITU-R BS.775 surround downmix coefficient (−3 dB).
const SURROUND_MIX: f32 = std::f32::consts::FRAC_1_SQRT_2; // ≈ 0.707

/// LFE mix coefficient: LFE is typically excluded from downmix by default.
/// Set to 0.0 to omit LFE (common in broadcast practice).
const LFE_MIX: f32 = 0.0;

/// Downmix a 5.1 audio buffer to stereo.
///
/// `samples_6ch` must be interleaved in channel order:
/// L, R, C, LFE, Ls, Rs (i.e. 6 samples per frame).
///
/// The output is interleaved stereo (L, R) at the same sample count.
///
/// # Downmix matrix (ITU-R BS.775-3)
/// ```text
/// Lt = L + CENTER_MIX×C + LFE_MIX×LFE + SURROUND_MIX×Ls
/// Rt = R + CENTER_MIX×C + LFE_MIX×LFE + SURROUND_MIX×Rs
/// ```
/// Samples are clamped to `[-1.0, 1.0]`.
///
/// Returns an empty `Vec` if `samples_6ch.len() % 6 != 0`.
pub fn downmix_5_1_to_stereo(samples_6ch: &[f32]) -> Vec<f32> {
    if samples_6ch.len() % CHANNELS_5_1 != 0 {
        return Vec::new();
    }

    let num_frames = samples_6ch.len() / CHANNELS_5_1;
    let mut out = Vec::with_capacity(num_frames * CHANNELS_STEREO);

    for frame in 0..num_frames {
        let base = frame * CHANNELS_5_1;
        let l = samples_6ch[base + CH_L];
        let r = samples_6ch[base + CH_R];
        let c = samples_6ch[base + CH_C];
        let lfe = samples_6ch[base + CH_LFE];
        let ls = samples_6ch[base + CH_LS];
        let rs = samples_6ch[base + CH_RS];

        let lt = (l + CENTER_MIX * c + LFE_MIX * lfe + SURROUND_MIX * ls).clamp(-1.0, 1.0);
        let rt = (r + CENTER_MIX * c + LFE_MIX * lfe + SURROUND_MIX * rs).clamp(-1.0, 1.0);

        out.push(lt);
        out.push(rt);
    }

    out
}

/// Upmix a stereo buffer to 5.1.
///
/// `samples_2ch` must be interleaved stereo (L, R) — 2 samples per frame.
///
/// The output is interleaved 5.1 in channel order: L, R, C, LFE, Ls, Rs.
///
/// # Upmix matrix
/// ```text
/// L   = Lt
/// R   = Rt
/// C   = 0.5 × (Lt + Rt)       (phantom centre blend)
/// LFE = 0.0                   (no LFE generated)
/// Ls  = 0.5 × (Lt − Rt)       (difference signal to surrounds)
/// Rs  = 0.5 × (Rt − Lt)
/// ```
/// All output samples are clamped to `[-1.0, 1.0]`.
///
/// Returns an empty `Vec` if `samples_2ch.len() % 2 != 0`.
pub fn upmix_stereo_to_5_1(samples_2ch: &[f32]) -> Vec<f32> {
    if samples_2ch.len() % CHANNELS_STEREO != 0 {
        return Vec::new();
    }

    let num_frames = samples_2ch.len() / CHANNELS_STEREO;
    let mut out = Vec::with_capacity(num_frames * CHANNELS_5_1);

    for frame in 0..num_frames {
        let base = frame * CHANNELS_STEREO;
        let lt = samples_2ch[base];
        let rt = samples_2ch[base + 1];

        let l = lt;
        let r = rt;
        let c = (0.5 * (lt + rt)).clamp(-1.0, 1.0);
        let lfe = 0.0f32;
        let ls = (0.5 * (lt - rt)).clamp(-1.0, 1.0);
        let rs = (0.5 * (rt - lt)).clamp(-1.0, 1.0);

        out.push(l);
        out.push(r);
        out.push(c);
        out.push(lfe);
        out.push(ls);
        out.push(rs);
    }

    out
}

/// Remap channels using a custom permutation table.
///
/// `mapping` is a slice of length `out_channels` where `mapping[i]` is the
/// input channel index for output channel `i`.  `None` produces silence.
///
/// Returns an empty `Vec` if `samples.len() % in_channels != 0`.
pub fn remap_channels(samples: &[f32], in_channels: usize, mapping: &[Option<usize>]) -> Vec<f32> {
    if in_channels == 0 || samples.len() % in_channels != 0 {
        return Vec::new();
    }

    let out_channels = mapping.len();
    let num_frames = samples.len() / in_channels;
    let mut out = Vec::with_capacity(num_frames * out_channels);

    for frame in 0..num_frames {
        let base = frame * in_channels;
        for &src_ch in mapping {
            match src_ch {
                Some(ch) if ch < in_channels => out.push(samples[base + ch]),
                _ => out.push(0.0),
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a 5.1 interleaved buffer with `frames` frames, each channel set
    /// to a distinct value: L=1, R=2, C=3, LFE=4, Ls=5, Rs=6 for frame 0
    /// (repeating).
    fn make_5_1_buf(frames: usize) -> Vec<f32> {
        let src = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        src.iter().copied().cycle().take(frames * 6).collect()
    }

    #[test]
    fn downmix_output_length() {
        let input = make_5_1_buf(100);
        let out = downmix_5_1_to_stereo(&input);
        assert_eq!(out.len(), 200);
    }

    #[test]
    fn downmix_invalid_input_returns_empty() {
        let bad = vec![0.0f32; 7]; // not divisible by 6
        assert!(downmix_5_1_to_stereo(&bad).is_empty());
    }

    #[test]
    fn downmix_clamp_no_overflow() {
        // Feed full-scale values to every channel → should clamp to [-1.0, 1.0]
        let loud = vec![1.0f32; 6 * 10];
        let out = downmix_5_1_to_stereo(&loud);
        for s in &out {
            assert!(*s >= -1.0 && *s <= 1.0, "sample out of range: {s}");
        }
    }

    #[test]
    fn upmix_output_length() {
        let stereo: Vec<f32> = std::iter::repeat([0.5f32, -0.5])
            .take(100)
            .flatten()
            .collect();
        let out = upmix_stereo_to_5_1(&stereo);
        assert_eq!(out.len(), 600);
    }

    #[test]
    fn upmix_invalid_input_returns_empty() {
        let bad = vec![0.0f32; 3]; // odd length
        assert!(upmix_stereo_to_5_1(&bad).is_empty());
    }

    #[test]
    fn upmix_lfe_is_zero() {
        let stereo: Vec<f32> = std::iter::repeat([0.8f32, 0.6])
            .take(10)
            .flatten()
            .collect();
        let out = upmix_stereo_to_5_1(&stereo);
        for frame in 0..10 {
            let lfe = out[frame * 6 + CH_LFE];
            assert_eq!(lfe, 0.0, "LFE should be zero at frame {frame}");
        }
    }

    #[test]
    fn upmix_then_downmix_roundtrip_length() {
        let stereo: Vec<f32> = (0..200).map(|i| (i as f32 / 200.0) * 0.5).collect();
        let upcast = upmix_stereo_to_5_1(&stereo);
        let downcast = downmix_5_1_to_stereo(&upcast);
        assert_eq!(downcast.len(), 200);
    }

    #[test]
    fn remap_channels_basic() {
        // mono → stereo (duplicate channel 0 to both outputs)
        let mono = vec![0.1f32, 0.2, 0.3];
        let mapping = vec![Some(0), Some(0)];
        let out = remap_channels(&mono, 1, &mapping);
        assert_eq!(out, vec![0.1, 0.1, 0.2, 0.2, 0.3, 0.3]);
    }

    #[test]
    fn remap_channels_silence_on_none() {
        let stereo = vec![0.5f32, -0.5];
        let mapping = vec![Some(0), None, Some(1)];
        let out = remap_channels(&stereo, 2, &mapping);
        assert_eq!(out, vec![0.5, 0.0, -0.5]);
    }
}
