//! WebAssembly bindings for audio mixing.
//!
//! Provides `WasmAudioMixer` for browser-side audio mixing, plus standalone
//! utility functions for stereo mixing, panning, and gain application.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert dB to linear gain.
fn db_to_linear(db: f64) -> f64 {
    if db <= -120.0 {
        0.0
    } else {
        10.0_f64.powf(db / 20.0)
    }
}

// ---------------------------------------------------------------------------
// Channel data for the mixer
// ---------------------------------------------------------------------------

struct WasmChannel {
    name: String,
    volume: f64, // linear
    pan: f64,    // -1.0 to 1.0
    mute: bool,
}

// ---------------------------------------------------------------------------
// WasmAudioMixer
// ---------------------------------------------------------------------------

/// Browser-side audio mixer.
///
/// Usage:
/// 1. Create a mixer with a sample rate.
/// 2. Add channels with names.
/// 3. Configure volume/pan/mute per channel.
/// 4. Call `mix()` with interleaved multi-channel input to get mixed stereo output.
#[wasm_bindgen]
pub struct WasmAudioMixer {
    sample_rate: u32,
    channels: Vec<WasmChannel>,
    master_volume: f64, // linear
}

#[wasm_bindgen]
impl WasmAudioMixer {
    /// Create a new WASM audio mixer.
    ///
    /// # Arguments
    /// * `sample_rate` - Audio sample rate in Hz.
    ///
    /// # Errors
    /// Returns an error if sample_rate is 0.
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: u32) -> Result<WasmAudioMixer, JsValue> {
        if sample_rate == 0 {
            return Err(crate::utils::js_err("Sample rate must be > 0"));
        }
        Ok(Self {
            sample_rate,
            channels: Vec::new(),
            master_volume: 1.0,
        })
    }

    /// Add a channel and return its index.
    pub fn add_channel(&mut self, name: &str) -> u32 {
        let idx = self.channels.len() as u32;
        self.channels.push(WasmChannel {
            name: name.to_string(),
            volume: 1.0,
            pan: 0.0,
            mute: false,
        });
        idx
    }

    /// Set channel volume (linear: 0.0 = silence, 1.0 = unity, >1.0 = boost).
    pub fn set_channel_volume(&mut self, idx: u32, volume: f64) -> Result<(), JsValue> {
        let ch = self
            .channels
            .get_mut(idx as usize)
            .ok_or_else(|| crate::utils::js_err(&format!("Channel {} not found", idx)))?;
        if volume < 0.0 {
            return Err(crate::utils::js_err("Volume must be >= 0.0"));
        }
        ch.volume = volume;
        Ok(())
    }

    /// Set channel volume in dB.
    pub fn set_channel_volume_db(&mut self, idx: u32, db: f64) -> Result<(), JsValue> {
        let linear = db_to_linear(db);
        self.set_channel_volume(idx, linear)
    }

    /// Set channel pan (-1.0 = left, 0.0 = center, 1.0 = right).
    pub fn set_channel_pan(&mut self, idx: u32, pan: f64) -> Result<(), JsValue> {
        let ch = self
            .channels
            .get_mut(idx as usize)
            .ok_or_else(|| crate::utils::js_err(&format!("Channel {} not found", idx)))?;
        if !(-1.0..=1.0).contains(&pan) {
            return Err(crate::utils::js_err("Pan must be between -1.0 and 1.0"));
        }
        ch.pan = pan;
        Ok(())
    }

    /// Set channel mute state.
    pub fn set_channel_mute(&mut self, idx: u32, mute: bool) -> Result<(), JsValue> {
        let ch = self
            .channels
            .get_mut(idx as usize)
            .ok_or_else(|| crate::utils::js_err(&format!("Channel {} not found", idx)))?;
        ch.mute = mute;
        Ok(())
    }

    /// Set master volume (linear).
    pub fn set_master_volume(&mut self, volume: f64) -> Result<(), JsValue> {
        if volume < 0.0 {
            return Err(crate::utils::js_err("Master volume must be >= 0.0"));
        }
        self.master_volume = volume;
        Ok(())
    }

    /// Mix interleaved multi-channel input into stereo output.
    ///
    /// `inputs_interleaved` contains all channels interleaved:
    ///   [ch0_s0, ch1_s0, ..., chN_s0, ch0_s1, ch1_s1, ..., chN_s1, ...]
    ///
    /// `num_channels` is the number of input channels.
    /// `block_size` is the number of samples per channel.
    ///
    /// Returns interleaved stereo output (L0, R0, L1, R1, ...).
    pub fn mix(
        &self,
        inputs_interleaved: &[f32],
        num_channels: u32,
        block_size: u32,
    ) -> Result<Vec<f32>, JsValue> {
        let nc = num_channels as usize;
        let bs = block_size as usize;

        if nc == 0 {
            return Err(crate::utils::js_err("num_channels must be > 0"));
        }
        if bs == 0 {
            return Err(crate::utils::js_err("block_size must be > 0"));
        }

        let expected = nc * bs;
        if inputs_interleaved.len() < expected {
            return Err(crate::utils::js_err(&format!(
                "Input too small: need {} samples ({} channels x {} block_size), got {}",
                expected,
                nc,
                bs,
                inputs_interleaved.len()
            )));
        }

        let master = self.master_volume as f32;
        let mut output = vec![0.0_f32; bs * 2]; // stereo

        for ch_idx in 0..nc {
            let ch_config = self.channels.get(ch_idx);
            let (volume, pan, mute) = if let Some(ch) = ch_config {
                (ch.volume as f32, ch.pan, ch.mute)
            } else {
                (1.0_f32, 0.0, false)
            };

            if mute {
                continue;
            }

            // Constant-power pan law
            let angle = (pan + 1.0) * std::f64::consts::FRAC_PI_4;
            let left_gain = angle.cos() as f32 * volume * master;
            let right_gain = angle.sin() as f32 * volume * master;

            for s in 0..bs {
                let sample = inputs_interleaved[s * nc + ch_idx];
                output[s * 2] += sample * left_gain;
                output[s * 2 + 1] += sample * right_gain;
            }
        }

        Ok(output)
    }

    /// Get the number of channels.
    pub fn channel_count(&self) -> u32 {
        self.channels.len() as u32
    }

    /// Get mixer info as a JSON string.
    pub fn info(&self) -> String {
        let channels_json: Vec<String> = self
            .channels
            .iter()
            .enumerate()
            .map(|(i, ch)| {
                format!(
                    "{{\"index\":{},\"name\":\"{}\",\"volume\":{:.4},\"pan\":{:.4},\"mute\":{}}}",
                    i, ch.name, ch.volume, ch.pan, ch.mute
                )
            })
            .collect();

        format!(
            "{{\"sample_rate\":{},\"channel_count\":{},\"master_volume\":{:.4},\"channels\":[{}]}}",
            self.sample_rate,
            self.channels.len(),
            self.master_volume,
            channels_json.join(",")
        )
    }

    /// Reset the mixer, removing all channels.
    pub fn reset(&mut self) {
        self.channels.clear();
        self.master_volume = 1.0;
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Mix left and right channels into interleaved stereo with balance control.
///
/// `balance`: -1.0 = full left, 0.0 = center, 1.0 = full right.
#[wasm_bindgen]
pub fn wasm_mix_stereo(left: &[f32], right: &[f32], balance: f64) -> Result<Vec<f32>, JsValue> {
    if !(-1.0..=1.0).contains(&balance) {
        return Err(crate::utils::js_err("Balance must be between -1.0 and 1.0"));
    }

    let len = left.len().max(right.len());
    let left_gain = ((1.0 - balance) / 2.0 + 0.5).min(1.0).max(0.0) as f32;
    let right_gain = ((1.0 + balance) / 2.0 + 0.5).min(1.0).max(0.0) as f32;

    let mut output = Vec::with_capacity(len * 2);
    for i in 0..len {
        let l = left.get(i).copied().unwrap_or(0.0) * left_gain;
        let r = right.get(i).copied().unwrap_or(0.0) * right_gain;
        output.push(l);
        output.push(r);
    }

    Ok(output)
}

/// Apply constant-power panning to mono samples.
///
/// Returns interleaved stereo output (L, R, L, R, ...).
/// `pan`: -1.0 = full left, 0.0 = center, 1.0 = full right.
#[wasm_bindgen]
pub fn wasm_apply_pan(samples: &[f32], pan: f64) -> Result<Vec<f32>, JsValue> {
    if !(-1.0..=1.0).contains(&pan) {
        return Err(crate::utils::js_err("Pan must be between -1.0 and 1.0"));
    }

    let angle = (pan + 1.0) * std::f64::consts::FRAC_PI_4;
    let left_gain = angle.cos() as f32;
    let right_gain = angle.sin() as f32;

    let mut output = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        output.push(s * left_gain);
        output.push(s * right_gain);
    }

    Ok(output)
}

/// Apply gain in dB to audio samples.
#[wasm_bindgen]
pub fn wasm_apply_gain(samples: &[f32], gain_db: f64) -> Result<Vec<f32>, JsValue> {
    let gain = db_to_linear(gain_db) as f32;
    Ok(samples.iter().map(|&s| s * gain).collect())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixer_creation() {
        let m = WasmAudioMixer::new(48000);
        assert!(m.is_ok());
        let m = m.expect("should create mixer");
        assert_eq!(m.channel_count(), 0);
    }

    #[test]
    fn test_mixer_invalid_sample_rate() {
        let m = WasmAudioMixer::new(0);
        assert!(m.is_err());
    }

    #[test]
    fn test_add_channels() {
        let mut m = WasmAudioMixer::new(48000).expect("should create");
        let idx0 = m.add_channel("Vocals");
        let idx1 = m.add_channel("Guitar");
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(m.channel_count(), 2);
    }

    #[test]
    fn test_mix_stereo() {
        let left = vec![1.0_f32; 4];
        let right = vec![0.5_f32; 4];
        let result = wasm_mix_stereo(&left, &right, 0.0);
        assert!(result.is_ok());
        let out = result.expect("should mix");
        assert_eq!(out.len(), 8); // 4 samples * 2 channels
    }

    #[test]
    fn test_apply_gain_zero_db() {
        let samples = vec![0.5_f32; 3];
        let result = wasm_apply_gain(&samples, 0.0);
        assert!(result.is_ok());
        let out = result.expect("should apply gain");
        for &s in &out {
            assert!((s - 0.5).abs() < 0.001);
        }
    }

    #[test]
    fn test_apply_pan_hard_left() {
        let samples = vec![1.0_f32; 2];
        let result = wasm_apply_pan(&samples, -1.0);
        assert!(result.is_ok());
        let out = result.expect("should pan");
        // Hard left: right channel should be near zero
        assert!(out[1].abs() < 0.01);
        assert!(out[0] > 0.9);
    }
}
