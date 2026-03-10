//! WASM watermarking operations: embed, detect, quality metrics, capacity.
//!
//! All functions operate on in-memory sample arrays -- no file system access.
//! Results are returned as raw byte/float arrays or JSON strings.

use wasm_bindgen::prelude::*;

use oximedia_watermark::{Algorithm, WatermarkConfig, WatermarkDetector, WatermarkEmbedder};

// ---------------------------------------------------------------------------
// Algorithm parsing helper
// ---------------------------------------------------------------------------

fn parse_algorithm(name: &str) -> Result<Algorithm, JsValue> {
    match name.to_ascii_lowercase().replace('-', "_").as_str() {
        "spread_spectrum" | "spreadspectrum" | "ss" | "dsss" => Ok(Algorithm::SpreadSpectrum),
        "echo" | "echo_hiding" => Ok(Algorithm::Echo),
        "phase" | "phase_coding" => Ok(Algorithm::Phase),
        "lsb" | "steganography" => Ok(Algorithm::Lsb),
        "patchwork" => Ok(Algorithm::Patchwork),
        "qim" | "quantization" => Ok(Algorithm::Qim),
        other => Err(crate::utils::js_err(&format!(
            "Unknown algorithm '{}'. Available: spread_spectrum, echo, phase, lsb, patchwork, qim",
            other
        ))),
    }
}

// ---------------------------------------------------------------------------
// Audio watermark embedding
// ---------------------------------------------------------------------------

/// Embed a watermark into audio samples.
///
/// # Arguments
///
/// * `samples` - f32 PCM audio samples (mono, normalised to -1.0..1.0)
/// * `payload` - Watermark message bytes
/// * `sample_rate` - Audio sample rate in Hz
/// * `algorithm` - Algorithm name (spread_spectrum, echo, phase, lsb, patchwork, qim)
/// * `strength` - Embedding strength (0.0 to 1.0)
///
/// # Returns
///
/// Watermarked audio samples as `Vec<f32>`.
///
/// # Errors
///
/// Returns an error if embedding fails or parameters are invalid.
#[wasm_bindgen]
pub fn wasm_embed_audio_watermark(
    samples: &[f32],
    payload: &[u8],
    sample_rate: u32,
    algorithm: &str,
    strength: f64,
) -> Result<Vec<f32>, JsValue> {
    if samples.is_empty() {
        return Err(crate::utils::js_err("Audio samples cannot be empty"));
    }
    if payload.is_empty() {
        return Err(crate::utils::js_err("Payload cannot be empty"));
    }
    if sample_rate == 0 {
        return Err(crate::utils::js_err("Sample rate must be > 0"));
    }
    if !(0.0..=1.0).contains(&strength) {
        return Err(crate::utils::js_err("Strength must be between 0.0 and 1.0"));
    }

    let algo = parse_algorithm(algorithm)?;
    let config = WatermarkConfig::default()
        .with_algorithm(algo)
        .with_strength(strength as f32)
        .with_psychoacoustic(true);

    let embedder = WatermarkEmbedder::new(config, sample_rate);
    embedder
        .embed(samples, payload)
        .map_err(|e| crate::utils::js_err(&format!("Embedding failed: {}", e)))
}

// ---------------------------------------------------------------------------
// Audio watermark detection
// ---------------------------------------------------------------------------

/// Detect and extract a watermark from audio samples.
///
/// # Arguments
///
/// * `samples` - f32 PCM audio samples (potentially watermarked)
/// * `expected_bits` - Number of bits to extract
/// * `algorithm` - Algorithm used for embedding
///
/// # Returns
///
/// Extracted watermark payload bytes.
///
/// # Errors
///
/// Returns an error if detection fails.
#[wasm_bindgen]
pub fn wasm_detect_audio_watermark(
    samples: &[f32],
    expected_bits: u32,
    algorithm: &str,
) -> Result<Vec<u8>, JsValue> {
    if samples.is_empty() {
        return Err(crate::utils::js_err("Audio samples cannot be empty"));
    }
    if expected_bits == 0 {
        return Err(crate::utils::js_err("Expected bits must be > 0"));
    }

    let algo = parse_algorithm(algorithm)?;
    let config = WatermarkConfig::default().with_algorithm(algo);

    let detector = WatermarkDetector::new(config);
    detector
        .detect(samples, expected_bits as usize)
        .map_err(|e| crate::utils::js_err(&format!("Detection failed: {}", e)))
}

// ---------------------------------------------------------------------------
// Image watermark embedding (spatial domain LSB)
// ---------------------------------------------------------------------------

/// Embed a watermark into raw 8-bit image pixel data using spatial-domain LSB.
///
/// Modifies the least significant bit(s) of pixel values to encode the payload.
///
/// # Arguments
///
/// * `data` - Raw 8-bit pixel data (interleaved RGB/RGBA)
/// * `width` - Image width
/// * `height` - Image height
/// * `channels` - Number of channels (3 for RGB, 4 for RGBA)
/// * `payload` - Watermark payload bytes
/// * `strength` - Bits per pixel to modify (0.0-1.0, maps to 1-4 LSBs)
///
/// # Returns
///
/// Watermarked pixel data.
///
/// # Errors
///
/// Returns an error if the image is too small for the payload.
#[wasm_bindgen]
pub fn wasm_embed_image_watermark(
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    payload: &[u8],
    strength: f64,
) -> Result<Vec<u8>, JsValue> {
    let expected = (width as usize) * (height as usize) * (channels as usize);
    if data.len() < expected {
        return Err(crate::utils::js_err(&format!(
            "Pixel data too small: need {} bytes, got {}",
            expected,
            data.len()
        )));
    }
    if payload.is_empty() {
        return Err(crate::utils::js_err("Payload cannot be empty"));
    }

    // Number of LSBs to use (1-4 based on strength)
    let lsb_count = ((strength * 4.0).clamp(1.0, 4.0)) as u32;

    let payload_bits = payload.len() * 8;
    let available_pixels = (width as usize) * (height as usize) * (channels as usize);

    // We need payload_bits / lsb_count pixels minimum, plus a 32-bit length header
    let needed = (payload_bits as f64 / lsb_count as f64).ceil() as usize + 32;
    if available_pixels < needed {
        return Err(crate::utils::js_err(&format!(
            "Image too small for payload: need {} embeddable values, have {}",
            needed, available_pixels
        )));
    }

    let mut output = data[..expected].to_vec();

    // Embed length header (32 bits) then payload
    let payload_len = payload.len() as u32;
    let mut bit_stream = Vec::with_capacity(32 + payload_bits);

    // Length header
    for bit_idx in 0..32u32 {
        bit_stream.push(((payload_len >> (31 - bit_idx)) & 1) as u8);
    }

    // Payload bits
    for &byte in payload {
        for bit_idx in 0..8u32 {
            bit_stream.push(((byte >> (7 - bit_idx)) & 1) as u8);
        }
    }

    let mask = !(((1u32 << lsb_count) - 1) as u8);
    let mut bit_idx = 0;

    for pixel_byte in output.iter_mut() {
        if bit_idx >= bit_stream.len() {
            break;
        }

        // Clear the LSBs
        *pixel_byte &= mask;

        // Pack up to lsb_count bits
        let mut bits_val: u8 = 0;
        for lsb_pos in 0..lsb_count {
            if bit_idx < bit_stream.len() {
                bits_val |= bit_stream[bit_idx] << (lsb_count - 1 - lsb_pos);
                bit_idx += 1;
            }
        }

        *pixel_byte |= bits_val;
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Image watermark detection (spatial domain LSB)
// ---------------------------------------------------------------------------

/// Detect and extract a watermark from raw 8-bit image pixel data.
///
/// Reads the LSB-encoded payload including length header.
///
/// # Arguments
///
/// * `data` - Potentially watermarked pixel data
/// * `width` - Image width
/// * `height` - Image height
/// * `channels` - Number of channels
/// * `expected_bits` - Hint: number of LSBs used (maps from strength, 1-4)
///
/// # Returns
///
/// Extracted watermark payload bytes.
///
/// # Errors
///
/// Returns an error if extraction fails.
#[wasm_bindgen]
pub fn wasm_detect_image_watermark(
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    expected_bits: u32,
) -> Result<Vec<u8>, JsValue> {
    let expected_size = (width as usize) * (height as usize) * (channels as usize);
    if data.len() < expected_size {
        return Err(crate::utils::js_err("Pixel data too small"));
    }

    let lsb_count = expected_bits.clamp(1, 4);

    // Extract bit stream from LSBs
    let mut bit_stream = Vec::new();
    for &pixel_byte in &data[..expected_size] {
        for lsb_pos in 0..lsb_count {
            let bit = (pixel_byte >> (lsb_count - 1 - lsb_pos)) & 1;
            bit_stream.push(bit);
        }
    }

    // Read 32-bit length header
    if bit_stream.len() < 32 {
        return Err(crate::utils::js_err("Not enough data for length header"));
    }

    let mut payload_len: u32 = 0;
    for bit_idx in 0..32 {
        payload_len = (payload_len << 1) | u32::from(bit_stream[bit_idx]);
    }

    let payload_bits = (payload_len as usize) * 8;
    if bit_stream.len() < 32 + payload_bits {
        return Err(crate::utils::js_err(&format!(
            "Not enough data for payload: header says {} bytes but only {} bits available",
            payload_len,
            bit_stream.len() - 32
        )));
    }

    // Extract payload bytes
    let mut payload = Vec::with_capacity(payload_len as usize);
    for byte_idx in 0..payload_len as usize {
        let mut byte_val: u8 = 0;
        for bit_pos in 0..8 {
            let stream_idx = 32 + byte_idx * 8 + bit_pos;
            byte_val = (byte_val << 1) | bit_stream[stream_idx];
        }
        payload.push(byte_val);
    }

    Ok(payload)
}

// ---------------------------------------------------------------------------
// Quality metrics
// ---------------------------------------------------------------------------

/// Compute quality metrics comparing original and watermarked audio.
///
/// Returns JSON with SNR, ODG, PSNR, correlation, and transparency assessment.
///
/// # Errors
///
/// Returns an error if sample arrays are empty or mismatched in length.
#[wasm_bindgen]
pub fn wasm_watermark_quality(original: &[f32], watermarked: &[f32]) -> Result<String, JsValue> {
    if original.is_empty() || watermarked.is_empty() {
        return Err(crate::utils::js_err("Audio samples cannot be empty"));
    }
    if original.len() != watermarked.len() {
        return Err(crate::utils::js_err(&format!(
            "Sample count mismatch: original={}, watermarked={}",
            original.len(),
            watermarked.len()
        )));
    }

    let metrics = oximedia_watermark::calculate_metrics(original, watermarked);
    let psnr = oximedia_watermark::metrics::calculate_psnr(original, watermarked);
    let correlation = oximedia_watermark::metrics::calculate_correlation(original, watermarked);
    let transparent = metrics.odg > -1.0;

    let result = serde_json::json!({
        "snr_db": metrics.snr_db,
        "odg": metrics.odg,
        "psnr_db": psnr,
        "correlation": correlation,
        "transparent": transparent,
        "sample_count": original.len(),
    });

    serde_json::to_string(&result)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization error: {}", e)))
}

// ---------------------------------------------------------------------------
// Capacity info
// ---------------------------------------------------------------------------

/// Calculate watermark embedding capacity for given audio parameters.
///
/// Returns JSON with capacity in bits and bytes.
///
/// # Errors
///
/// Returns an error for invalid parameters.
#[wasm_bindgen]
pub fn wasm_watermark_capacity(
    sample_count: u32,
    sample_rate: u32,
    algorithm: &str,
) -> Result<String, JsValue> {
    if sample_count == 0 {
        return Err(crate::utils::js_err("Sample count must be > 0"));
    }
    if sample_rate == 0 {
        return Err(crate::utils::js_err("Sample rate must be > 0"));
    }

    let algo = parse_algorithm(algorithm)?;
    let config = WatermarkConfig::default().with_algorithm(algo);
    let embedder = WatermarkEmbedder::new(config, sample_rate);

    let capacity_bits = embedder.capacity(sample_count as usize);
    let capacity_bytes = capacity_bits / 8;
    let duration_secs = sample_count as f64 / sample_rate as f64;

    let result = serde_json::json!({
        "algorithm": algorithm,
        "sample_count": sample_count,
        "sample_rate": sample_rate,
        "duration_seconds": duration_secs,
        "capacity_bits": capacity_bits,
        "capacity_bytes": capacity_bytes,
        "bits_per_second": if duration_secs > 0.0 { capacity_bits as f64 / duration_secs } else { 0.0 },
    });

    serde_json::to_string(&result)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization error: {}", e)))
}

// ---------------------------------------------------------------------------
// Algorithm list
// ---------------------------------------------------------------------------

/// List all available watermarking algorithms with descriptions.
///
/// Returns a JSON array of objects with `name` and `description` fields.
#[wasm_bindgen]
pub fn wasm_list_watermark_algorithms() -> Result<String, JsValue> {
    let algorithms = serde_json::json!([
        {
            "name": "spread_spectrum",
            "description": "Spread Spectrum (DSSS) - robust watermarking using pseudorandom sequences",
            "robustness": "high",
            "capacity": "low",
            "transparency": "high"
        },
        {
            "name": "echo",
            "description": "Echo Hiding - embeds data using echo patterns in audio",
            "robustness": "medium",
            "capacity": "medium",
            "transparency": "high"
        },
        {
            "name": "phase",
            "description": "Phase Coding - modulates DFT phase to encode data",
            "robustness": "medium",
            "capacity": "medium",
            "transparency": "high"
        },
        {
            "name": "lsb",
            "description": "Least Significant Bit - high capacity but fragile steganography",
            "robustness": "low",
            "capacity": "high",
            "transparency": "high"
        },
        {
            "name": "patchwork",
            "description": "Patchwork - statistical watermarking using sample pair modifications",
            "robustness": "medium",
            "capacity": "low",
            "transparency": "high"
        },
        {
            "name": "qim",
            "description": "Quantization Index Modulation - robust quantization-based embedding",
            "robustness": "high",
            "capacity": "medium",
            "transparency": "medium"
        }
    ]);

    serde_json::to_string(&algorithms)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_algorithm_valid() {
        assert!(parse_algorithm("spread_spectrum").is_ok());
        assert!(parse_algorithm("echo").is_ok());
        assert!(parse_algorithm("phase").is_ok());
        assert!(parse_algorithm("lsb").is_ok());
        assert!(parse_algorithm("patchwork").is_ok());
        assert!(parse_algorithm("qim").is_ok());
    }

    #[test]
    fn test_parse_algorithm_invalid() {
        assert!(parse_algorithm("unknown").is_err());
    }

    #[test]
    fn test_quality_metrics() {
        let original: Vec<f32> = vec![0.5; 1000];
        let watermarked: Vec<f32> = original.iter().map(|&s| s + 0.001).collect();

        let result = wasm_watermark_quality(&original, &watermarked).expect("should succeed");
        assert!(result.contains("snr_db"));
        assert!(result.contains("transparent"));
    }

    #[test]
    fn test_quality_empty_error() {
        let empty: Vec<f32> = vec![];
        assert!(wasm_watermark_quality(&empty, &empty).is_err());
    }

    #[test]
    fn test_quality_mismatch_error() {
        let a: Vec<f32> = vec![0.0; 100];
        let b: Vec<f32> = vec![0.0; 50];
        assert!(wasm_watermark_quality(&a, &b).is_err());
    }

    #[test]
    fn test_capacity_info() {
        let result =
            wasm_watermark_capacity(44100, 44100, "spread_spectrum").expect("should succeed");
        assert!(result.contains("capacity_bits"));
        assert!(result.contains("duration_seconds"));
    }

    #[test]
    fn test_list_algorithms() {
        let result = wasm_list_watermark_algorithms().expect("should succeed");
        assert!(result.contains("spread_spectrum"));
        assert!(result.contains("echo"));
        assert!(result.contains("qim"));
    }

    #[test]
    fn test_image_watermark_roundtrip() {
        // 8x8 RGB image
        let width = 8u32;
        let height = 8u32;
        let channels = 3u32;
        let data = vec![128u8; (width * height * channels) as usize];
        let payload = b"Hi";

        let watermarked = wasm_embed_image_watermark(&data, width, height, channels, payload, 0.25)
            .expect("embed should succeed");

        let extracted = wasm_detect_image_watermark(&watermarked, width, height, channels, 1)
            .expect("detect should succeed");

        assert_eq!(&extracted, payload);
    }

    #[test]
    fn test_image_watermark_too_small() {
        let data = vec![128u8; 3]; // 1x1 RGB
        let payload = vec![0u8; 100]; // way too big
        assert!(wasm_embed_image_watermark(&data, 1, 1, 3, &payload, 0.25).is_err());
    }
}
