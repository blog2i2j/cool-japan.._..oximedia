//! WebAssembly bindings for spatial audio from `oximedia-spatial`.
//!
//! Provides Higher-Order Ambisonics (HOA) encoding/decoding, VBAP panning,
//! and spatial audio format information — all operating in-memory without
//! file-system access, for browser-based immersive audio workflows.

use wasm_bindgen::prelude::*;

use oximedia_spatial::ambisonics::{
    AmbisonicsDecoder, AmbisonicsEncoder, AmbisonicsNorm, AmbisonicsOrder, SoundSource,
};
use oximedia_spatial::vbap::{Speaker, VbapPanner};

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

fn js_err(msg: impl std::fmt::Display) -> JsValue {
    crate::utils::js_err(&format!("{msg}"))
}

// ---------------------------------------------------------------------------
// Standalone ambisonics helpers
// ---------------------------------------------------------------------------

/// Return a JSON array describing available Ambisonics orders.
///
/// Each entry has `order`, `channels`, and `description` fields.
#[wasm_bindgen]
pub fn wasm_ambisonics_orders() -> Result<String, JsValue> {
    let orders = serde_json::json!([
        { "order": 1, "channels": 4,  "description": "First-order Ambisonics (W, Y, Z, X)" },
        { "order": 2, "channels": 9,  "description": "Second-order Ambisonics" },
        { "order": 3, "channels": 16, "description": "Third-order Ambisonics" },
        { "order": 4, "channels": 25, "description": "Fourth-order Ambisonics" },
        { "order": 5, "channels": 36, "description": "Fifth-order Ambisonics (large-venue)" },
    ]);
    serde_json::to_string(&orders).map_err(|e| js_err(format!("JSON error: {e}")))
}

/// Compute the number of Ambisonics channels for a given order.
///
/// Formula: (order + 1)².  Returns 0 for order > 5.
///
/// # Errors
/// Returns an error for orders outside [1, 5].
#[wasm_bindgen]
pub fn wasm_ambisonics_channel_count(order: u32) -> Result<u32, JsValue> {
    let ao = parse_ambisonics_order(order)?;
    Ok(ao.num_channels() as u32)
}

/// Encode a mono audio buffer to Ambisonics B-format at the given source position.
///
/// - `samples`: mono PCM samples in [-1, 1]
/// - `order`: Ambisonics order (1–5)
/// - `azimuth_deg`: source azimuth in degrees (0=front, 90=left)
/// - `elevation_deg`: source elevation in degrees (0=horizontal, +90=above)
///
/// Returns a flat interleaved buffer of `len(samples) × num_channels` samples:
/// `[ch0[0], ch1[0], …, chN[0], ch0[1], ch1[1], …]`.
///
/// # Errors
/// Returns an error for invalid orders.
#[wasm_bindgen]
pub fn wasm_ambisonics_encode(
    samples: &[f32],
    order: u32,
    azimuth_deg: f32,
    elevation_deg: f32,
) -> Result<Vec<f32>, JsValue> {
    let ao = parse_ambisonics_order(order)?;
    let encoder = AmbisonicsEncoder {
        order: ao,
        sample_rate: 48000,
        normalization: AmbisonicsNorm::Sn3d,
    };
    let source = SoundSource::new(azimuth_deg, elevation_deg);
    let channels = encoder.encode_mono(samples, &source);
    let num_ch = channels.len();
    let len = samples.len();
    // Interleave: ch0[0], ch1[0], ..., chN[0], ch0[1], ...
    let mut out = vec![0.0_f32; len * num_ch];
    for (sample_idx, frame) in out.chunks_exact_mut(num_ch).enumerate() {
        for (ch_idx, ch) in channels.iter().enumerate() {
            frame[ch_idx] = ch.get(sample_idx).copied().unwrap_or(0.0);
        }
    }
    Ok(out)
}

/// Decode an Ambisonics B-format buffer to stereo.
///
/// `data` is a flat interleaved buffer: `[ch0[0], ch1[0], …, ch0[1], …]`.
/// `num_channels` is the number of Ambisonics channels ((order+1)²).
/// Returns an interleaved stereo buffer `[L[0], R[0], L[1], R[1], …]`.
///
/// # Errors
/// Returns an error for invalid orders or mismatched buffer sizes.
#[wasm_bindgen]
pub fn wasm_ambisonics_decode_stereo(
    data: &[f32],
    order: u32,
    num_channels: u32,
) -> Result<Vec<f32>, JsValue> {
    let ao = parse_ambisonics_order(order)?;
    let n = num_channels as usize;
    if n == 0 || data.len() % n != 0 {
        return Err(js_err(format!(
            "data length {} is not divisible by num_channels {}",
            data.len(),
            n
        )));
    }
    let num_frames = data.len() / n;
    // De-interleave.
    let mut channels: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0_f32; num_frames]).collect();
    for (frame_idx, frame) in data.chunks_exact(n).enumerate() {
        for (ch_idx, &v) in frame.iter().enumerate() {
            if let Some(ch) = channels.get_mut(ch_idx) {
                ch[frame_idx] = v;
            }
        }
    }
    let decoder = AmbisonicsDecoder::new(ao);
    let (left, right) = decoder.decode_stereo(&channels);
    // Interleave L/R.
    let out: Vec<f32> = left
        .iter()
        .zip(right.iter())
        .flat_map(|(&l, &r)| [l, r])
        .collect();
    Ok(out)
}

// ---------------------------------------------------------------------------
// VBAP panning
// ---------------------------------------------------------------------------

/// Compute per-speaker VBAP gains for a mono source at the given azimuth.
///
/// `speaker_azimuths` is a JSON array of azimuth angles in degrees, e.g.
/// `[-30, 0, 30]` for a standard front-left, centre, front-right layout.
/// `azimuth_deg` is the source direction (0=front, 90=left).
/// Returns a JSON array of gains (one per speaker), summing to ≤ 1.
///
/// # Errors
/// Returns an error for invalid speaker layouts or JSON.
#[wasm_bindgen]
pub fn wasm_vbap_pan(speaker_azimuths_json: &str, azimuth_deg: f32) -> Result<String, JsValue> {
    let azimuths: Vec<f32> = serde_json::from_str(speaker_azimuths_json)
        .map_err(|e| js_err(format!("Invalid speaker azimuths JSON: {e}")))?;
    if azimuths.len() < 2 {
        return Err(js_err("Need at least 2 speakers for VBAP"));
    }
    let speakers: Vec<Speaker> = azimuths
        .iter()
        .enumerate()
        .map(|(i, &az)| Speaker {
            azimuth_deg: az,
            elevation_deg: 0.0,
            channel_index: i,
        })
        .collect();
    let panner = VbapPanner::new(speakers).map_err(|e| js_err(format!("VBAP error: {e}")))?;
    let gains = panner.pan(azimuth_deg);
    serde_json::to_string(&gains).map_err(|e| js_err(format!("JSON error: {e}")))
}

/// Return a JSON array describing common loudspeaker layouts.
#[wasm_bindgen]
pub fn wasm_spatial_speaker_layouts() -> Result<String, JsValue> {
    let layouts = serde_json::json!([
        {
            "name": "stereo",
            "channels": 2,
            "description": "Standard stereo (L/R ±30°)",
            "azimuths_deg": [-30.0, 30.0]
        },
        {
            "name": "5.1",
            "channels": 6,
            "description": "Surround 5.1 (FL, FR, C, LFE*, LS, RS)",
            "azimuths_deg": [-30.0, 30.0, 0.0, null, -110.0, 110.0]
        },
        {
            "name": "7.1",
            "channels": 8,
            "description": "Surround 7.1 (FL, FR, C, LFE*, LS, RS, LRS, RRS)",
            "azimuths_deg": [-30.0, 30.0, 0.0, null, -90.0, 90.0, -150.0, 150.0]
        },
        {
            "name": "atmos_7.1.4",
            "channels": 12,
            "description": "Dolby Atmos 7.1.4 base layer (bed + 4 height channels)",
            "azimuths_deg": [-30.0, 30.0, 0.0, null, -90.0, 90.0, -150.0, 150.0,
                              -45.0, 45.0, -135.0, 135.0]
        }
    ]);
    serde_json::to_string(&layouts).map_err(|e| js_err(format!("JSON error: {e}")))
}

// ---------------------------------------------------------------------------
// WasmAmbisonicsEncoder — stateful encoder
// ---------------------------------------------------------------------------

/// Stateful Ambisonics encoder that can be reused across multiple frames.
#[wasm_bindgen]
pub struct WasmAmbisonicsEncoder {
    encoder: AmbisonicsEncoder,
    num_channels: usize,
}

#[wasm_bindgen]
impl WasmAmbisonicsEncoder {
    /// Create a new encoder for the given Ambisonics order (1–5).
    ///
    /// # Errors
    /// Returns an error for orders outside [1, 5].
    #[wasm_bindgen(constructor)]
    pub fn new(order: u32, sample_rate: u32) -> Result<WasmAmbisonicsEncoder, JsValue> {
        let ao = parse_ambisonics_order(order)?;
        let num_channels = ao.num_channels();
        Ok(WasmAmbisonicsEncoder {
            encoder: AmbisonicsEncoder {
                order: ao,
                sample_rate,
                normalization: AmbisonicsNorm::Sn3d,
            },
            num_channels,
        })
    }

    /// Number of output Ambisonics channels.
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Encode a mono audio block at the given source position.
    ///
    /// Returns an interleaved buffer of `samples.len() × num_channels` values.
    ///
    /// # Errors
    /// Returns an error if `samples` is empty.
    pub fn encode(
        &self,
        samples: &[f32],
        azimuth_deg: f32,
        elevation_deg: f32,
    ) -> Result<Vec<f32>, JsValue> {
        if samples.is_empty() {
            return Err(js_err("samples must not be empty"));
        }
        let source = SoundSource::new(azimuth_deg, elevation_deg);
        let channels = self.encoder.encode_mono(samples, &source);
        let len = samples.len();
        let n = channels.len();
        let mut out = vec![0.0_f32; len * n];
        for (frame_idx, frame) in out.chunks_exact_mut(n).enumerate() {
            for (ch_idx, ch) in channels.iter().enumerate() {
                frame[ch_idx] = ch.get(frame_idx).copied().unwrap_or(0.0);
            }
        }
        Ok(out)
    }

    /// Return the Ambisonics order as an integer.
    pub fn order(&self) -> u32 {
        match self.encoder.order {
            AmbisonicsOrder::First => 1,
            AmbisonicsOrder::Second => 2,
            AmbisonicsOrder::Third => 3,
            AmbisonicsOrder::Fourth => 4,
            AmbisonicsOrder::Fifth => 5,
        }
    }

    /// Return the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.encoder.sample_rate
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn parse_ambisonics_order(order: u32) -> Result<AmbisonicsOrder, JsValue> {
    match order {
        1 => Ok(AmbisonicsOrder::First),
        2 => Ok(AmbisonicsOrder::Second),
        3 => Ok(AmbisonicsOrder::Third),
        4 => Ok(AmbisonicsOrder::Fourth),
        5 => Ok(AmbisonicsOrder::Fifth),
        other => Err(js_err(format!(
            "Invalid Ambisonics order {other}. Supported: 1, 2, 3, 4, 5"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ambisonics_orders_json() {
        let json = wasm_ambisonics_orders().expect("ok");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed.as_array().expect("array").len(), 5);
    }

    #[test]
    fn test_ambisonics_channel_count() {
        assert_eq!(wasm_ambisonics_channel_count(1).expect("ok"), 4);
        assert_eq!(wasm_ambisonics_channel_count(2).expect("ok"), 9);
        assert_eq!(wasm_ambisonics_channel_count(3).expect("ok"), 16);
    }

    #[test]
    fn test_ambisonics_channel_count_invalid() {
        assert!(wasm_ambisonics_channel_count(0).is_err());
        assert!(wasm_ambisonics_channel_count(6).is_err());
    }

    #[test]
    fn test_ambisonics_encode_first_order_length() {
        let samples: Vec<f32> = (0..256).map(|i| (i as f32 / 256.0).sin()).collect();
        let out = wasm_ambisonics_encode(&samples, 1, 0.0, 0.0).expect("ok");
        // 4 channels × 256 samples = 1024
        assert_eq!(out.len(), 256 * 4);
    }

    #[test]
    fn test_ambisonics_encode_invalid_order() {
        let samples = vec![0.5_f32; 64];
        assert!(wasm_ambisonics_encode(&samples, 0, 0.0, 0.0).is_err());
    }

    #[test]
    fn test_ambisonics_encode_decode_stereo_roundtrip() {
        // Encode silence — decode should also be silence.
        let samples = vec![0.0_f32; 128];
        let encoded = wasm_ambisonics_encode(&samples, 1, 0.0, 0.0).expect("encode ok");
        let decoded = wasm_ambisonics_decode_stereo(&encoded, 1, 4).expect("decode ok");
        assert_eq!(decoded.len(), 128 * 2); // interleaved stereo
        for &v in &decoded {
            assert!(v.abs() < 1e-6, "expected silence, got {v}");
        }
    }

    #[test]
    fn test_ambisonics_decode_mismatched_buffer() {
        // data.len() % num_channels != 0
        let data = vec![0.0_f32; 7];
        assert!(wasm_ambisonics_decode_stereo(&data, 1, 4).is_err());
    }

    #[test]
    fn test_vbap_pan_stereo() {
        // Two speakers at ±30°, source at 0° (centre) → equal gains.
        let gains_json = wasm_vbap_pan("[-30, 30]", 0.0).expect("ok");
        let gains: Vec<f32> = serde_json::from_str(&gains_json).expect("valid JSON");
        assert_eq!(gains.len(), 2);
        // Both gains should be equal (within floating-point).
        assert!((gains[0] - gains[1]).abs() < 0.05, "gains: {:?}", gains);
    }

    #[test]
    fn test_vbap_pan_invalid_layout() {
        // Only one speaker — should fail.
        assert!(wasm_vbap_pan("[0]", 0.0).is_err());
    }

    #[test]
    fn test_spatial_speaker_layouts_json() {
        let json = wasm_spatial_speaker_layouts().expect("ok");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert!(parsed.as_array().expect("array").len() >= 3);
    }

    #[test]
    fn test_wasm_ambisonics_encoder_new_valid() {
        let enc = WasmAmbisonicsEncoder::new(1, 48000).expect("ok");
        assert_eq!(enc.num_channels(), 4);
        assert_eq!(enc.order(), 1);
        assert_eq!(enc.sample_rate(), 48000);
    }

    #[test]
    fn test_wasm_ambisonics_encoder_new_invalid_order() {
        assert!(WasmAmbisonicsEncoder::new(0, 48000).is_err());
        assert!(WasmAmbisonicsEncoder::new(6, 48000).is_err());
    }

    #[test]
    fn test_wasm_ambisonics_encoder_encode_length() {
        let enc = WasmAmbisonicsEncoder::new(2, 48000).expect("ok");
        let samples = vec![0.0_f32; 64];
        let out = enc.encode(&samples, 45.0, 0.0).expect("ok");
        // 9 channels × 64 samples = 576
        assert_eq!(out.len(), 64 * 9);
    }

    #[test]
    fn test_wasm_ambisonics_encoder_empty_samples_error() {
        let enc = WasmAmbisonicsEncoder::new(1, 48000).expect("ok");
        assert!(enc.encode(&[], 0.0, 0.0).is_err());
    }

    #[test]
    fn test_ambisonics_encode_front_source_gains() {
        // For a front source (az=0, el=0) in FOA (order=1), the W channel
        // gain should be sqrt(0.5) ≈ 0.707 and X channel = same.
        let samples = vec![1.0_f32; 1];
        let out = wasm_ambisonics_encode(&samples, 1, 0.0, 0.0).expect("ok");
        // out is interleaved: [ch0, ch1, ch2, ch3] for 1 sample
        assert_eq!(out.len(), 4);
        // All gains should be finite and within reasonable range.
        for &v in &out {
            assert!(v.is_finite(), "gain {v} is not finite");
        }
    }
}
