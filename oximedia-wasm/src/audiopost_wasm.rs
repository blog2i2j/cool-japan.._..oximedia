//! WebAssembly bindings for audio post-production utilities.
//!
//! Provides delivery spec checking, audio restoration, mixing, and stem export
//! info functions for browser-side audio post-production workflows.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn db_to_linear(db: f64) -> f64 {
    if db <= -120.0 {
        0.0
    } else {
        10.0_f64.powf(db / 20.0)
    }
}

/// Delivery spec parameters.
struct DeliverySpec {
    name: &'static str,
    max_loudness_lkfs: f32,
    max_true_peak_dbtp: f32,
    min_sample_rate: u32,
}

fn get_delivery_spec(spec: &str) -> Result<DeliverySpec, JsValue> {
    match spec {
        "broadcast" => Ok(DeliverySpec {
            name: "broadcast",
            max_loudness_lkfs: -23.0,
            max_true_peak_dbtp: -1.0,
            min_sample_rate: 48000,
        }),
        "cinema" => Ok(DeliverySpec {
            name: "cinema",
            max_loudness_lkfs: -27.0,
            max_true_peak_dbtp: -2.0,
            min_sample_rate: 48000,
        }),
        "streaming" => Ok(DeliverySpec {
            name: "streaming",
            max_loudness_lkfs: -14.0,
            max_true_peak_dbtp: -1.0,
            min_sample_rate: 44100,
        }),
        "podcast" => Ok(DeliverySpec {
            name: "podcast",
            max_loudness_lkfs: -16.0,
            max_true_peak_dbtp: -1.0,
            min_sample_rate: 44100,
        }),
        _ => Err(crate::utils::js_err(&format!(
            "Unknown delivery spec '{}'. Expected: broadcast, cinema, streaming, podcast",
            spec
        ))),
    }
}

// ---------------------------------------------------------------------------
// Delivery spec checking
// ---------------------------------------------------------------------------

/// Check audio samples against a delivery specification.
///
/// Returns a JSON report with check results:
/// ```json
/// {
///   "spec": "broadcast",
///   "passed": true,
///   "checks": [
///     {"name": "sample_rate", "passed": true, "message": "..."},
///     {"name": "true_peak", "passed": true, "message": "..."},
///     {"name": "loudness", "passed": true, "message": "..."}
///   ]
/// }
/// ```
#[wasm_bindgen]
pub fn wasm_check_delivery_spec(
    samples: &[f32],
    sample_rate: u32,
    spec: &str,
) -> Result<String, JsValue> {
    if samples.is_empty() {
        return Err(crate::utils::js_err("No samples provided"));
    }
    if sample_rate == 0 {
        return Err(crate::utils::js_err("Sample rate must be > 0"));
    }

    let delivery = get_delivery_spec(spec)?;

    let mut checks = Vec::new();
    let mut all_passed = true;

    // Sample rate check
    let sr_ok = sample_rate >= delivery.min_sample_rate;
    if !sr_ok {
        all_passed = false;
    }
    checks.push(format!(
        "{{\"name\":\"sample_rate\",\"passed\":{},\"message\":\"Required >= {} Hz, got {} Hz\"}}",
        sr_ok, delivery.min_sample_rate, sample_rate
    ));

    // Peak level check
    let peak = samples.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
    let peak_dbfs = if peak > 0.0 {
        20.0 * (peak as f64).log10()
    } else {
        -120.0
    };
    let peak_ok = peak_dbfs <= delivery.max_true_peak_dbtp as f64;
    if !peak_ok {
        all_passed = false;
    }
    checks.push(format!(
        "{{\"name\":\"true_peak\",\"passed\":{},\"message\":\"Max: {} dBTP, measured: {:.1} dBFS\"}}",
        peak_ok, delivery.max_true_peak_dbtp, peak_dbfs
    ));

    // Loudness check (simplified RMS-based approximation)
    let rms = {
        let sum: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
        (sum / samples.len() as f64).sqrt()
    };
    let loudness_approx = if rms > 0.0 {
        20.0 * rms.log10() - 0.691
    } else {
        -120.0
    };
    let loudness_ok = loudness_approx <= delivery.max_loudness_lkfs as f64 + 1.0;
    if !loudness_ok {
        all_passed = false;
    }
    checks.push(format!(
        "{{\"name\":\"loudness\",\"passed\":{},\"message\":\"Max: {} LKFS, measured: {:.1} LKFS (approx)\"}}",
        loudness_ok, delivery.max_loudness_lkfs, loudness_approx
    ));

    Ok(format!(
        "{{\"spec\":\"{}\",\"passed\":{},\"checks\":[{}]}}",
        delivery.name,
        all_passed,
        checks.join(",")
    ))
}

/// List available delivery specifications as a JSON array.
#[wasm_bindgen]
pub fn wasm_list_delivery_specs() -> String {
    "[\"broadcast\",\"cinema\",\"streaming\",\"podcast\"]".to_string()
}

// ---------------------------------------------------------------------------
// Audio restoration
// ---------------------------------------------------------------------------

/// Apply audio restoration to samples.
///
/// Returns restored audio samples.
#[wasm_bindgen]
pub fn wasm_restore_audio(
    samples: &[f32],
    sample_rate: u32,
    declip: bool,
    dehum: bool,
    decrackle: bool,
    denoise: bool,
) -> Result<Vec<f32>, JsValue> {
    if samples.is_empty() {
        return Err(crate::utils::js_err("No samples provided"));
    }
    if sample_rate == 0 {
        return Err(crate::utils::js_err("Sample rate must be > 0"));
    }

    let mut output = samples.to_vec();

    // Declipping: soft-clip any samples that appear clipped
    if declip {
        let threshold = 0.99_f32;
        for s in &mut output {
            if s.abs() > threshold {
                let sign = s.signum();
                let abs_val = s.abs();
                *s = sign
                    * (threshold
                        + (1.0 - threshold) * ((abs_val - threshold) / (1.0 - threshold)).tanh());
            }
        }
    }

    // Dehum: notch at 50/60 Hz
    if dehum {
        let hum_period_50 = (sample_rate as f64 / 50.0).round() as usize;
        let hum_period_60 = (sample_rate as f64 / 60.0).round() as usize;

        for period in &[hum_period_50, hum_period_60] {
            if *period > 0 && output.len() > *period {
                let mut filtered = vec![0.0_f32; output.len()];
                for i in 0..output.len() {
                    let prev = if i >= *period {
                        output[i - period]
                    } else {
                        0.0
                    };
                    filtered[i] = output[i] - prev * 0.3;
                }
                output = filtered;
            }
        }
    }

    // Decrackle: median filter
    if decrackle {
        let window = 3_usize;
        let half = window / 2;
        let original = output.clone();
        for i in half..output.len().saturating_sub(half) {
            let mut window_samples: Vec<f32> = (0..window)
                .filter_map(|j| original.get(i + j - half).copied())
                .collect();
            window_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if let Some(&median) = window_samples.get(window_samples.len() / 2) {
                if (output[i] - median).abs() > 0.1 {
                    output[i] = median;
                }
            }
        }
    }

    // Denoise: simple exponential smoothing
    if denoise {
        let alpha = 0.15_f32;
        let mut prev = output.first().copied().unwrap_or(0.0);
        for s in &mut output {
            *s = alpha * *s + (1.0 - alpha) * prev;
            prev = *s;
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Audio mixing
// ---------------------------------------------------------------------------

/// Mix multiple audio channels with individual levels.
///
/// `channels`: interleaved channel data [ch0_s0, ch1_s0, ..., ch0_s1, ch1_s1, ...]
/// `num_channels`: number of input channels
/// `samples_per_channel`: number of samples per channel
/// `levels_db`: gain levels in dB for each channel
///
/// Returns mixed mono output.
#[wasm_bindgen]
pub fn wasm_mix_audio(
    channels: &[f32],
    num_channels: u32,
    samples_per_channel: u32,
    levels_db: &[f64],
) -> Result<Vec<f32>, JsValue> {
    let nc = num_channels as usize;
    let spc = samples_per_channel as usize;

    if nc == 0 {
        return Err(crate::utils::js_err("num_channels must be > 0"));
    }
    if spc == 0 {
        return Err(crate::utils::js_err("samples_per_channel must be > 0"));
    }

    let expected = nc * spc;
    if channels.len() < expected {
        return Err(crate::utils::js_err(&format!(
            "Input too small: need {} samples, got {}",
            expected,
            channels.len()
        )));
    }

    let mut output = vec![0.0_f32; spc];

    for ch in 0..nc {
        let gain = if ch < levels_db.len() {
            db_to_linear(levels_db[ch]) as f32
        } else {
            1.0_f32
        };

        for s in 0..spc {
            output[s] += channels[s * nc + ch] * gain;
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Stem export info
// ---------------------------------------------------------------------------

/// Generate stem export configuration as JSON.
///
/// Returns a JSON object describing the stem export setup.
#[wasm_bindgen]
pub fn wasm_export_stems_info(num_stems: u32, sample_rate: u32) -> String {
    let standard_names = ["Dialogue", "Music", "Effects", "Foley", "Ambience"];

    let stems_json: Vec<String> = (0..num_stems)
        .map(|i| {
            let name = if (i as usize) < standard_names.len() {
                standard_names[i as usize]
            } else {
                "Custom"
            };
            format!(
                "{{\"index\":{},\"name\":\"{}\",\"format\":\"wav\",\"bit_depth\":24}}",
                i, name
            )
        })
        .collect();

    format!(
        "{{\"num_stems\":{},\"sample_rate\":{},\"stems\":[{}]}}",
        num_stems,
        sample_rate,
        stems_json.join(",")
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_delivery_broadcast() {
        let samples = vec![0.01_f32; 48000];
        let result = wasm_check_delivery_spec(&samples, 48000, "broadcast");
        assert!(result.is_ok());
        let json = result.expect("should check");
        assert!(json.contains("\"passed\":true"));
    }

    #[test]
    fn test_check_delivery_invalid_spec() {
        let samples = vec![0.01_f32; 100];
        let result = wasm_check_delivery_spec(&samples, 48000, "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_list_delivery_specs() {
        let json = wasm_list_delivery_specs();
        assert!(json.contains("broadcast"));
        assert!(json.contains("podcast"));
    }

    #[test]
    fn test_restore_audio_declip() {
        let samples = vec![1.0_f32; 100];
        let result = wasm_restore_audio(&samples, 48000, true, false, false, false);
        assert!(result.is_ok());
        let restored = result.expect("should restore");
        for &s in &restored {
            assert!(s <= 1.0);
        }
    }

    #[test]
    fn test_mix_audio() {
        // 2 channels, 4 samples per channel, interleaved
        let channels = vec![1.0_f32, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5];
        let levels = vec![0.0, -6.0]; // 0 dB and -6 dB
        let result = wasm_mix_audio(&channels, 2, 4, &levels);
        assert!(result.is_ok());
        let mixed = result.expect("should mix");
        assert_eq!(mixed.len(), 4);
        // Channel 0 at 0dB (1.0) + Channel 1 at -6dB (~0.5 * ~0.5)
        assert!(mixed[0] > 1.0);
    }

    #[test]
    fn test_export_stems_info() {
        let json = wasm_export_stems_info(3, 48000);
        assert!(json.contains("\"num_stems\":3"));
        assert!(json.contains("Dialogue"));
        assert!(json.contains("Music"));
        assert!(json.contains("Effects"));
    }
}
