/// Noise statistics for adaptive processing.
use oximedia_codec::VideoFrame;

/// Noise statistics for adaptive processing.
#[derive(Debug, Clone)]
pub(super) struct NoiseStatistics {
    /// Estimated noise standard deviation for luma.
    pub(super) luma_noise_sigma: f32,
    /// Estimated noise standard deviation for chroma.
    pub(super) chroma_noise_sigma: f32,
    /// Noise level (0.0 = clean, 1.0 = very noisy).
    pub(super) noise_level: f32,
}

impl NoiseStatistics {
    /// Estimate noise statistics from a frame.
    pub(super) fn estimate(frame: &VideoFrame) -> Self {
        let luma_noise_sigma = if let Some(plane) = frame.planes.first() {
            estimate_plane_noise(&plane.data, frame.width, frame.height)
        } else {
            0.0
        };

        let chroma_noise_sigma = if frame.planes.len() > 1 {
            let (h_sub, v_sub) = frame.format.chroma_subsampling();
            let chroma_width = frame.width / h_sub;
            let chroma_height = frame.height / v_sub;

            let u_sigma = if let Some(plane) = frame.planes.get(1) {
                estimate_plane_noise(&plane.data, chroma_width, chroma_height)
            } else {
                0.0
            };

            let v_sigma = if let Some(plane) = frame.planes.get(2) {
                estimate_plane_noise(&plane.data, chroma_width, chroma_height)
            } else {
                0.0
            };

            (u_sigma + v_sigma) / 2.0
        } else {
            0.0
        };

        let noise_level = (luma_noise_sigma / 50.0).clamp(0.0, 1.0);

        Self {
            luma_noise_sigma,
            chroma_noise_sigma,
            noise_level,
        }
    }
}

/// Estimate noise in a single plane using median absolute deviation.
pub fn estimate_plane_noise(data: &[u8], width: u32, height: u32) -> f32 {
    let mut diffs = Vec::new();

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = (y * width + x) as usize;
            let center = data.get(idx).copied().unwrap_or(128) as i32;

            let right = data.get(idx + 1).copied().unwrap_or(128) as i32;
            let bottom = data
                .get((idx as u32 + width) as usize)
                .copied()
                .unwrap_or(128) as i32;

            diffs.push((center - right).abs());
            diffs.push((center - bottom).abs());
        }
    }

    if diffs.is_empty() {
        return 0.0;
    }

    diffs.sort_unstable();
    let median = diffs[diffs.len() / 2] as f32;

    median / 0.6745
}
