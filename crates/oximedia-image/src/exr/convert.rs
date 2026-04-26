//! Float format conversion utilities for EXR data.

use half::f16;

/// Converts f16 data to f32.
#[allow(dead_code)]
#[must_use]
pub fn convert_f16_to_f32(f16_data: &[u8]) -> Vec<f32> {
    f16_data
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f16::from_bits(bits).to_f32()
        })
        .collect()
}

/// Converts f32 data to f16.
#[allow(dead_code)]
#[must_use]
pub fn convert_f32_to_f16(f32_data: &[f32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(f32_data.len() * 2);

    for &value in f32_data {
        let f16_value = f16::from_f32(value);
        let bytes = f16_value.to_bits().to_le_bytes();
        output.extend_from_slice(&bytes);
    }

    output
}
