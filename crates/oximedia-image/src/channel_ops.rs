#![allow(dead_code)]
//! Per-channel operations for multi-channel image data.
//!
//! Provides utilities for splitting, merging, swapping, and extracting
//! individual color channels from planar or interleaved image buffers.
//! Supports arbitrary channel counts and common color models (RGB, RGBA, YCbCr).

/// Channel layout describing how pixel components are arranged in memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChannelLayout {
    /// Channels are interleaved: RGBRGBRGB...
    Interleaved,
    /// Channels are planar: RRR...GGG...BBB...
    Planar,
}

impl std::fmt::Display for ChannelLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Interleaved => write!(f, "Interleaved"),
            Self::Planar => write!(f, "Planar"),
        }
    }
}

/// Named channel identifiers for common color models.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChannelId {
    /// Red channel.
    Red,
    /// Green channel.
    Green,
    /// Blue channel.
    Blue,
    /// Alpha (transparency) channel.
    Alpha,
    /// Luminance channel.
    Luma,
    /// Cb (blue-difference chroma) channel.
    Cb,
    /// Cr (red-difference chroma) channel.
    Cr,
    /// Custom channel with a numeric index.
    Custom(u8),
}

impl std::fmt::Display for ChannelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Red => write!(f, "R"),
            Self::Green => write!(f, "G"),
            Self::Blue => write!(f, "B"),
            Self::Alpha => write!(f, "A"),
            Self::Luma => write!(f, "Y"),
            Self::Cb => write!(f, "Cb"),
            Self::Cr => write!(f, "Cr"),
            Self::Custom(n) => write!(f, "Ch{n}"),
        }
    }
}

/// Descriptor for a multi-channel image buffer.
#[derive(Clone, Debug)]
pub struct ChannelDescriptor {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Number of channels.
    pub num_channels: usize,
    /// Memory layout.
    pub layout: ChannelLayout,
    /// Channel identifiers (optional, length must match `num_channels` if set).
    pub channel_ids: Vec<ChannelId>,
}

impl ChannelDescriptor {
    /// Creates a new descriptor for an RGB interleaved image.
    pub fn rgb(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            num_channels: 3,
            layout: ChannelLayout::Interleaved,
            channel_ids: vec![ChannelId::Red, ChannelId::Green, ChannelId::Blue],
        }
    }

    /// Creates a new descriptor for an RGBA interleaved image.
    pub fn rgba(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            num_channels: 4,
            layout: ChannelLayout::Interleaved,
            channel_ids: vec![
                ChannelId::Red,
                ChannelId::Green,
                ChannelId::Blue,
                ChannelId::Alpha,
            ],
        }
    }

    /// Creates a new descriptor for a single-channel (grayscale) image.
    pub fn grayscale(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            num_channels: 1,
            layout: ChannelLayout::Interleaved,
            channel_ids: vec![ChannelId::Luma],
        }
    }

    /// Returns the total number of samples in the buffer.
    pub fn total_samples(&self) -> usize {
        self.width * self.height * self.num_channels
    }

    /// Returns the number of samples per channel plane.
    pub fn samples_per_plane(&self) -> usize {
        self.width * self.height
    }
}

/// Extracts a single channel from an interleaved f32 buffer.
///
/// Returns a new vector containing only the samples for channel `ch`.
pub fn extract_channel_f32(
    buf: &[f32],
    desc: &ChannelDescriptor,
    ch: usize,
) -> Result<Vec<f32>, String> {
    if desc.layout != ChannelLayout::Interleaved {
        return Err("extract_channel_f32 expects interleaved layout".to_string());
    }
    if ch >= desc.num_channels {
        return Err(format!(
            "channel index {} out of range (num_channels={})",
            ch, desc.num_channels
        ));
    }
    let expected = desc.total_samples();
    if buf.len() != expected {
        return Err(format!(
            "buffer size mismatch: expected {expected}, got {}",
            buf.len()
        ));
    }

    let pixel_count = desc.samples_per_plane();
    let nc = desc.num_channels;
    let mut out = Vec::with_capacity(pixel_count);
    for i in 0..pixel_count {
        out.push(buf[i * nc + ch]);
    }
    Ok(out)
}

/// Inserts a single channel into an interleaved f32 buffer.
///
/// Overwrites channel `ch` with the data from `channel_data`.
pub fn insert_channel_f32(
    buf: &mut [f32],
    desc: &ChannelDescriptor,
    ch: usize,
    channel_data: &[f32],
) -> Result<(), String> {
    if desc.layout != ChannelLayout::Interleaved {
        return Err("insert_channel_f32 expects interleaved layout".to_string());
    }
    if ch >= desc.num_channels {
        return Err(format!(
            "channel index {} out of range (num_channels={})",
            ch, desc.num_channels
        ));
    }
    let pixel_count = desc.samples_per_plane();
    if channel_data.len() != pixel_count {
        return Err(format!(
            "channel_data size mismatch: expected {pixel_count}, got {}",
            channel_data.len()
        ));
    }
    let nc = desc.num_channels;
    for i in 0..pixel_count {
        buf[i * nc + ch] = channel_data[i];
    }
    Ok(())
}

/// Splits an interleaved buffer into separate planar channels.
pub fn deinterleave_f32(buf: &[f32], desc: &ChannelDescriptor) -> Result<Vec<Vec<f32>>, String> {
    if desc.layout != ChannelLayout::Interleaved {
        return Err("deinterleave_f32 expects interleaved layout".to_string());
    }
    let expected = desc.total_samples();
    if buf.len() != expected {
        return Err(format!(
            "buffer size mismatch: expected {expected}, got {}",
            buf.len()
        ));
    }

    let nc = desc.num_channels;
    let pixel_count = desc.samples_per_plane();
    let mut planes: Vec<Vec<f32>> = (0..nc).map(|_| Vec::with_capacity(pixel_count)).collect();

    for i in 0..pixel_count {
        for c in 0..nc {
            planes[c].push(buf[i * nc + c]);
        }
    }

    Ok(planes)
}

/// Merges separate planar channels into an interleaved buffer.
pub fn interleave_f32(planes: &[Vec<f32>], desc: &ChannelDescriptor) -> Result<Vec<f32>, String> {
    if planes.len() != desc.num_channels {
        return Err(format!(
            "plane count mismatch: expected {}, got {}",
            desc.num_channels,
            planes.len()
        ));
    }
    let pixel_count = desc.samples_per_plane();
    for (i, plane) in planes.iter().enumerate() {
        if plane.len() != pixel_count {
            return Err(format!(
                "plane {i} size mismatch: expected {pixel_count}, got {}",
                plane.len()
            ));
        }
    }

    let nc = desc.num_channels;
    let mut out = vec![0.0_f32; desc.total_samples()];
    for i in 0..pixel_count {
        for c in 0..nc {
            out[i * nc + c] = planes[c][i];
        }
    }
    Ok(out)
}

/// Swaps two channels in an interleaved buffer (in-place).
pub fn swap_channels_f32(
    buf: &mut [f32],
    desc: &ChannelDescriptor,
    ch_a: usize,
    ch_b: usize,
) -> Result<(), String> {
    if desc.layout != ChannelLayout::Interleaved {
        return Err("swap_channels_f32 expects interleaved layout".to_string());
    }
    if ch_a >= desc.num_channels || ch_b >= desc.num_channels {
        return Err(format!(
            "channel index out of range: ch_a={ch_a}, ch_b={ch_b}, num_channels={}",
            desc.num_channels
        ));
    }
    if ch_a == ch_b {
        return Ok(());
    }
    let nc = desc.num_channels;
    let pixel_count = desc.samples_per_plane();
    for i in 0..pixel_count {
        buf.swap(i * nc + ch_a, i * nc + ch_b);
    }
    Ok(())
}

/// Fills a channel in an interleaved buffer with a constant value.
pub fn fill_channel_f32(
    buf: &mut [f32],
    desc: &ChannelDescriptor,
    ch: usize,
    value: f32,
) -> Result<(), String> {
    if desc.layout != ChannelLayout::Interleaved {
        return Err("fill_channel_f32 expects interleaved layout".to_string());
    }
    if ch >= desc.num_channels {
        return Err(format!(
            "channel index {} out of range (num_channels={})",
            ch, desc.num_channels
        ));
    }
    let nc = desc.num_channels;
    let pixel_count = desc.samples_per_plane();
    for i in 0..pixel_count {
        buf[i * nc + ch] = value;
    }
    Ok(())
}

/// Computes per-channel statistics.
#[derive(Clone, Debug)]
pub struct ChannelStats {
    /// Minimum value.
    pub min: f32,
    /// Maximum value.
    pub max: f32,
    /// Mean value.
    pub mean: f32,
    /// Channel identifier.
    pub channel: ChannelId,
}

/// Computes statistics for each channel in an interleaved buffer.
#[allow(clippy::cast_precision_loss)]
pub fn compute_channel_stats(
    buf: &[f32],
    desc: &ChannelDescriptor,
) -> Result<Vec<ChannelStats>, String> {
    if desc.layout != ChannelLayout::Interleaved {
        return Err("compute_channel_stats expects interleaved layout".to_string());
    }
    let expected = desc.total_samples();
    if buf.len() != expected {
        return Err(format!(
            "buffer size mismatch: expected {expected}, got {}",
            buf.len()
        ));
    }

    let nc = desc.num_channels;
    let pixel_count = desc.samples_per_plane();
    let mut results = Vec::with_capacity(nc);

    for c in 0..nc {
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        let mut sum = 0.0_f64;

        for i in 0..pixel_count {
            let v = buf[i * nc + c];
            min_val = min_val.min(v);
            max_val = max_val.max(v);
            sum += f64::from(v);
        }

        let mean = if pixel_count > 0 {
            (sum / pixel_count as f64) as f32
        } else {
            0.0
        };

        let ch_id = if c < desc.channel_ids.len() {
            desc.channel_ids[c]
        } else {
            ChannelId::Custom(c as u8)
        };

        results.push(ChannelStats {
            min: min_val,
            max: max_val,
            mean,
            channel: ch_id,
        });
    }

    Ok(results)
}

/// Premultiplies alpha: multiplies RGB channels by the alpha channel.
///
/// Assumes RGBA interleaved layout with alpha as the last channel.
pub fn premultiply_alpha_f32(buf: &mut [f32], desc: &ChannelDescriptor) -> Result<(), String> {
    if desc.num_channels < 2 {
        return Err("premultiply_alpha requires at least 2 channels".to_string());
    }
    if desc.layout != ChannelLayout::Interleaved {
        return Err("premultiply_alpha expects interleaved layout".to_string());
    }
    let nc = desc.num_channels;
    let alpha_ch = nc - 1;
    let pixel_count = desc.samples_per_plane();
    for i in 0..pixel_count {
        let alpha = buf[i * nc + alpha_ch];
        for c in 0..alpha_ch {
            buf[i * nc + c] *= alpha;
        }
    }
    Ok(())
}

/// Un-premultiplies alpha: divides RGB channels by the alpha channel.
///
/// Pixels with zero alpha are left unchanged.
pub fn unpremultiply_alpha_f32(buf: &mut [f32], desc: &ChannelDescriptor) -> Result<(), String> {
    if desc.num_channels < 2 {
        return Err("unpremultiply_alpha requires at least 2 channels".to_string());
    }
    if desc.layout != ChannelLayout::Interleaved {
        return Err("unpremultiply_alpha expects interleaved layout".to_string());
    }
    let nc = desc.num_channels;
    let alpha_ch = nc - 1;
    let pixel_count = desc.samples_per_plane();
    for i in 0..pixel_count {
        let alpha = buf[i * nc + alpha_ch];
        if alpha > 1e-10 {
            let inv = 1.0 / alpha;
            for c in 0..alpha_ch {
                buf[i * nc + c] *= inv;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_descriptor_rgb() {
        let desc = ChannelDescriptor::rgb(4, 3);
        assert_eq!(desc.total_samples(), 36);
        assert_eq!(desc.samples_per_plane(), 12);
        assert_eq!(desc.num_channels, 3);
    }

    #[test]
    fn test_channel_descriptor_rgba() {
        let desc = ChannelDescriptor::rgba(2, 2);
        assert_eq!(desc.total_samples(), 16);
        assert_eq!(desc.samples_per_plane(), 4);
    }

    #[test]
    fn test_extract_channel() {
        let desc = ChannelDescriptor::rgb(2, 2);
        // R=1.0 G=2.0 B=3.0 for all pixels
        let buf = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let red = extract_channel_f32(&buf, &desc, 0).expect("should succeed in test");
        assert_eq!(red, vec![1.0, 1.0, 1.0, 1.0]);
        let green = extract_channel_f32(&buf, &desc, 1).expect("should succeed in test");
        assert_eq!(green, vec![2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_insert_channel() {
        let desc = ChannelDescriptor::rgb(2, 1);
        let mut buf = vec![0.0; 6];
        let red_data = vec![1.0, 0.5];
        insert_channel_f32(&mut buf, &desc, 0, &red_data).expect("should succeed in test");
        assert_eq!(buf[0], 1.0);
        assert_eq!(buf[3], 0.5);
    }

    #[test]
    fn test_deinterleave_and_interleave() {
        let desc = ChannelDescriptor::rgb(2, 2);
        let buf = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let planes = deinterleave_f32(&buf, &desc).expect("should succeed in test");
        assert_eq!(planes.len(), 3);
        assert_eq!(planes[0], vec![1.0, 4.0, 7.0, 10.0]);
        assert_eq!(planes[1], vec![2.0, 5.0, 8.0, 11.0]);

        let reconstructed = interleave_f32(&planes, &desc).expect("should succeed in test");
        assert_eq!(reconstructed, buf);
    }

    #[test]
    fn test_swap_channels() {
        let desc = ChannelDescriptor::rgb(2, 1);
        let mut buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        swap_channels_f32(&mut buf, &desc, 0, 2).expect("should succeed in test");
        // R and B swapped
        assert_eq!(buf, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn test_fill_channel() {
        let desc = ChannelDescriptor::rgba(2, 1);
        let mut buf = vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0];
        fill_channel_f32(&mut buf, &desc, 3, 1.0).expect("should succeed in test");
        assert_eq!(buf[3], 1.0);
        assert_eq!(buf[7], 1.0);
    }

    #[test]
    fn test_channel_stats() {
        let desc = ChannelDescriptor::rgb(2, 1);
        let buf = vec![0.0, 0.5, 1.0, 1.0, 0.5, 0.0];
        let stats = compute_channel_stats(&buf, &desc).expect("should succeed in test");
        assert_eq!(stats.len(), 3);
        assert_eq!(stats[0].min, 0.0);
        assert_eq!(stats[0].max, 1.0);
        assert!((stats[0].mean - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_premultiply_alpha() {
        let desc = ChannelDescriptor::rgba(1, 1);
        let mut buf = vec![1.0, 0.5, 0.25, 0.5];
        premultiply_alpha_f32(&mut buf, &desc).expect("should succeed in test");
        assert!((buf[0] - 0.5).abs() < 1e-5);
        assert!((buf[1] - 0.25).abs() < 1e-5);
        assert!((buf[2] - 0.125).abs() < 1e-5);
        assert!((buf[3] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_unpremultiply_alpha() {
        let desc = ChannelDescriptor::rgba(1, 1);
        let mut buf = vec![0.5, 0.25, 0.125, 0.5];
        unpremultiply_alpha_f32(&mut buf, &desc).expect("should succeed in test");
        assert!((buf[0] - 1.0).abs() < 1e-5);
        assert!((buf[1] - 0.5).abs() < 1e-5);
        assert!((buf[2] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_unpremultiply_zero_alpha() {
        let desc = ChannelDescriptor::rgba(1, 1);
        let mut buf = vec![0.5, 0.25, 0.125, 0.0];
        unpremultiply_alpha_f32(&mut buf, &desc).expect("should succeed in test");
        // Should be unchanged since alpha is zero
        assert!((buf[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_extract_channel_error() {
        let desc = ChannelDescriptor::rgb(2, 2);
        let buf = vec![0.0; 12];
        let err = extract_channel_f32(&buf, &desc, 5);
        assert!(err.is_err());
    }

    #[test]
    fn test_channel_layout_display() {
        assert_eq!(ChannelLayout::Interleaved.to_string(), "Interleaved");
        assert_eq!(ChannelLayout::Planar.to_string(), "Planar");
    }

    #[test]
    fn test_channel_id_display() {
        assert_eq!(ChannelId::Red.to_string(), "R");
        assert_eq!(ChannelId::Alpha.to_string(), "A");
        assert_eq!(ChannelId::Custom(7).to_string(), "Ch7");
    }
}
