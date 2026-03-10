//! Grain-preserving denoising.
//!
//! Applies denoising while preserving film grain to maintain the
//! cinematic aesthetic of film-sourced content.

use crate::grain::analysis::{GrainMap, GrainPattern};
use crate::spatial::bilateral;
use crate::{DenoiseError, DenoiseResult};
use oximedia_codec::VideoFrame;
use rayon::prelude::*;

/// Apply denoising while preserving film grain.
///
/// Uses the grain map to selectively denoise, preserving grain in
/// detected grain regions while removing noise elsewhere.
///
/// # Arguments
/// * `frame` - Input video frame
/// * `grain_map` - Grain characteristics map
/// * `strength` - Base denoising strength (0.0 - 1.0)
///
/// # Returns
/// Denoised frame with preserved grain
pub fn preserve_grain_denoise(
    frame: &VideoFrame,
    grain_map: &GrainMap,
    strength: f32,
) -> DenoiseResult<VideoFrame> {
    if frame.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    // First, apply strong denoising
    let denoised = bilateral::bilateral_filter(frame, strength)?;

    // Then blend with original based on grain map
    blend_with_grain(frame, &denoised, grain_map)
}

/// Blend denoised frame with original based on grain map.
fn blend_with_grain(
    original: &VideoFrame,
    denoised: &VideoFrame,
    grain_map: &GrainMap,
) -> DenoiseResult<VideoFrame> {
    let mut output = denoised.clone();

    output
        .planes
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(plane_idx, plane)| {
            let original_plane = &original.planes[plane_idx];
            let denoised_plane = &denoised.planes[plane_idx];
            let (width, height) = original.plane_dimensions(plane_idx);

            let mut blended = plane.data.clone();

            for y in 0..(height as usize) {
                for x in 0..(width as usize) {
                    let idx = y * plane.stride + x;

                    // Get grain strength (with bounds checking)
                    let grain_idx =
                        y.min(grain_map.height - 1) * grain_map.width + x.min(grain_map.width - 1);
                    let grain_strength = if grain_idx < grain_map.strength.len() {
                        grain_map.strength[grain_idx]
                    } else {
                        0.0
                    };

                    // Blend: more original in grain regions, more denoised elsewhere
                    let original_val = f32::from(original_plane.data[idx]);
                    let denoised_val = f32::from(denoised_plane.data[idx]);

                    let blend_factor = grain_strength.clamp(0.0, 1.0);
                    let result = blend_factor * original_val + (1.0 - blend_factor) * denoised_val;

                    blended[idx] = result.round().clamp(0.0, 255.0) as u8;
                }
            }

            plane.data = blended;
            Ok::<(), DenoiseError>(())
        })?;

    Ok(output)
}

/// Adaptive grain-preserving denoising.
///
/// Automatically adjusts denoising based on detected grain pattern.
pub fn adaptive_grain_preserve(
    frame: &VideoFrame,
    grain_map: &GrainMap,
    strength: f32,
) -> DenoiseResult<VideoFrame> {
    // Adjust strength based on grain pattern
    let adjusted_strength = match grain_map.pattern_type {
        GrainPattern::None => strength,
        GrainPattern::Fine => strength * 0.8,
        GrainPattern::Medium => strength * 0.6,
        GrainPattern::Coarse => strength * 0.4,
        GrainPattern::DigitalNoise => strength * 1.2, // Stronger for digital noise
    };

    preserve_grain_denoise(frame, grain_map, adjusted_strength)
}

/// Extract and add back grain pattern.
pub fn extract_and_restore_grain(frame: &VideoFrame, strength: f32) -> DenoiseResult<VideoFrame> {
    if frame.planes.is_empty() {
        return Err(DenoiseError::ProcessingError(
            "Frame has no planes".to_string(),
        ));
    }

    // Extract grain by subtracting smoothed version
    let smoothed = bilateral::bilateral_filter(frame, 0.8)?;
    let grain = extract_grain_pattern(frame, &smoothed);

    // Apply denoising
    let denoised = bilateral::bilateral_filter(frame, strength)?;

    // Add grain back
    Ok(add_grain_pattern(&denoised, &grain, strength))
}

/// Extract grain pattern by subtracting smoothed version.
fn extract_grain_pattern(original: &VideoFrame, smoothed: &VideoFrame) -> Vec<Vec<i16>> {
    let mut grain_planes = Vec::new();

    for (plane_idx, original_plane) in original.planes.iter().enumerate() {
        let smoothed_plane = &smoothed.planes[plane_idx];
        let (width, height) = original.plane_dimensions(plane_idx);

        let mut grain = vec![0i16; width as usize * height as usize];

        for y in 0..(height as usize) {
            for x in 0..(width as usize) {
                let idx = y * original_plane.stride + x;
                let orig = i16::from(original_plane.data[idx]);
                let smooth = i16::from(smoothed_plane.data[idx]);
                grain[y * width as usize + x] = orig - smooth;
            }
        }

        grain_planes.push(grain);
    }

    grain_planes
}

/// Add grain pattern back to denoised frame.
fn add_grain_pattern(
    denoised: &VideoFrame,
    grain_planes: &[Vec<i16>],
    grain_strength: f32,
) -> VideoFrame {
    let mut output = denoised.clone();

    output
        .planes
        .par_iter_mut()
        .enumerate()
        .for_each(|(plane_idx, plane)| {
            if plane_idx >= grain_planes.len() {
                return;
            }

            let denoised_plane = &denoised.planes[plane_idx];
            let grain = &grain_planes[plane_idx];
            let (width, height) = denoised.plane_dimensions(plane_idx);

            let mut result = plane.data.clone();

            for y in 0..(height as usize) {
                for x in 0..(width as usize) {
                    let idx = y * plane.stride + x;
                    let grain_idx = y * width as usize + x;

                    if grain_idx < grain.len() {
                        let denoised_val = i16::from(denoised_plane.data[idx]);
                        let grain_val = (f32::from(grain[grain_idx]) * grain_strength) as i16;
                        let final_val = (denoised_val + grain_val).clamp(0, 255);
                        result[idx] = final_val as u8;
                    }
                }
            }

            plane.data = result;
        });

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grain::analysis::analyze_grain;
    use oximedia_core::PixelFormat;

    #[test]
    fn test_preserve_grain_denoise() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let grain_map = analyze_grain(&frame).expect("grain_map should be valid");
        let result = preserve_grain_denoise(&frame, &grain_map, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_adaptive_grain_preserve() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let grain_map = analyze_grain(&frame).expect("grain_map should be valid");
        let result = adaptive_grain_preserve(&frame, &grain_map, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_extract_and_restore_grain() {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        frame.allocate();

        let result = extract_and_restore_grain(&frame, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_grain_extraction() {
        let mut original = VideoFrame::new(PixelFormat::Yuv420p, 32, 32);
        original.allocate();

        let mut smoothed = VideoFrame::new(PixelFormat::Yuv420p, 32, 32);
        smoothed.allocate();

        let grain = extract_grain_pattern(&original, &smoothed);
        assert_eq!(grain.len(), 3); // 3 planes for YUV420p
    }
}
