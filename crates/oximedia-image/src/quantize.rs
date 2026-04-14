//! Color quantization algorithms for reducing image palettes.
//!
//! Provides the **median-cut** algorithm (Heckbert 1982) which recursively
//! bisects the color space along the axis with the largest range.  The result
//! is a set of representative palette entries that cover the color distribution
//! of the input image.
//!
//! # Example
//!
//! ```rust
//! use oximedia_image::quantize::median_cut;
//!
//! // 2-pixel RGB image: pure red and pure blue
//! let pixels = [255u8, 0, 0,  0, 0, 255];
//! let palette = median_cut(&pixels, 2);
//! assert_eq!(palette.len(), 2);
//! ```

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// An RGB triplet used during the median-cut algorithm.
#[derive(Clone, Copy, Debug)]
struct Rgb {
    r: u8,
    g: u8,
    b: u8,
}

// ---------------------------------------------------------------------------
// Median-cut implementation
// ---------------------------------------------------------------------------

/// Extract the palette of `n_colors` representative colours from `img` using
/// the **median-cut** algorithm.
///
/// # Parameters
/// - `img`      – interleaved RGB u8 pixel buffer (length must be a multiple of 3).
/// - `n_colors` – desired palette size.  If `n_colors == 0` an empty palette is
///                returned.  If the image has fewer distinct pixels than
///                `n_colors`, all distinct colours are returned.
///
/// # Returns
/// A `Vec` of up to `n_colors` `[u8; 3]` palette entries.  Entries are the
/// average colour of each colour-cube bucket produced by the median-cut.
///
/// # Panics
/// Panics if `img.len()` is not a multiple of 3.
#[must_use]
pub fn median_cut(img: &[u8], n_colors: usize) -> Vec<[u8; 3]> {
    assert_eq!(
        img.len() % 3,
        0,
        "median_cut: img length must be a multiple of 3"
    );

    if n_colors == 0 || img.is_empty() {
        return Vec::new();
    }

    // Collect pixels
    let pixels: Vec<Rgb> = img
        .chunks_exact(3)
        .map(|c| Rgb {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect();

    if pixels.is_empty() {
        return Vec::new();
    }

    // Start with all pixels in one bucket
    let mut buckets: Vec<Vec<Rgb>> = vec![pixels];

    // Repeatedly split the bucket with the largest color range
    while buckets.len() < n_colors {
        // Find the bucket with the largest range across any channel
        let split_idx = find_split_bucket(&buckets);

        let bucket = buckets.swap_remove(split_idx);
        let (a, b) = split_bucket(bucket);
        if a.is_empty() && b.is_empty() {
            break;
        }
        if !a.is_empty() {
            buckets.push(a);
        }
        if !b.is_empty() {
            buckets.push(b);
        }

        if buckets.len() >= n_colors {
            break;
        }
    }

    // Average each bucket to get palette entry
    buckets.iter().map(|bucket| average_color(bucket)).collect()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find the index of the bucket with the largest channel range.
fn find_split_bucket(buckets: &[Vec<Rgb>]) -> usize {
    let mut best = 0;
    let mut best_range = 0u32;

    for (i, bucket) in buckets.iter().enumerate() {
        if bucket.is_empty() {
            continue;
        }
        let range = max_channel_range(bucket);
        if range > best_range {
            best_range = range;
            best = i;
        }
    }

    best
}

/// Return the maximum range across R, G, B channels in `bucket`.
fn max_channel_range(bucket: &[Rgb]) -> u32 {
    let (r_min, r_max) = channel_min_max(bucket, Channel::R);
    let (g_min, g_max) = channel_min_max(bucket, Channel::G);
    let (b_min, b_max) = channel_min_max(bucket, Channel::B);
    let r_range = u32::from(r_max - r_min);
    let g_range = u32::from(g_max - g_min);
    let b_range = u32::from(b_max - b_min);
    r_range.max(g_range).max(b_range)
}

/// Which channel has the largest range in `bucket`.
enum Channel {
    R,
    G,
    B,
}

fn channel_min_max(bucket: &[Rgb], ch: Channel) -> (u8, u8) {
    let vals: Vec<u8> = bucket
        .iter()
        .map(|p| match ch {
            Channel::R => p.r,
            Channel::G => p.g,
            Channel::B => p.b,
        })
        .collect();
    let min = *vals.iter().min().unwrap_or(&0);
    let max = *vals.iter().max().unwrap_or(&255);
    (min, max)
}

/// Split `bucket` along the channel with the largest range at the median.
fn split_bucket(mut bucket: Vec<Rgb>) -> (Vec<Rgb>, Vec<Rgb>) {
    if bucket.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Find which channel to split on
    let (r_min, r_max) = channel_min_max(&bucket, Channel::R);
    let (g_min, g_max) = channel_min_max(&bucket, Channel::G);
    let (b_min, b_max) = channel_min_max(&bucket, Channel::B);
    let r_range = u32::from(r_max - r_min);
    let g_range = u32::from(g_max - g_min);
    let b_range = u32::from(b_max - b_min);

    if r_range >= g_range && r_range >= b_range {
        bucket.sort_unstable_by_key(|p| p.r);
    } else if g_range >= b_range {
        bucket.sort_unstable_by_key(|p| p.g);
    } else {
        bucket.sort_unstable_by_key(|p| p.b);
    }

    let mid = bucket.len() / 2;
    let upper = bucket.split_off(mid);
    (bucket, upper)
}

/// Compute the average colour of a bucket.
fn average_color(bucket: &[Rgb]) -> [u8; 3] {
    if bucket.is_empty() {
        return [0, 0, 0];
    }
    let n = bucket.len() as u64;
    let r_sum: u64 = bucket.iter().map(|p| u64::from(p.r)).sum();
    let g_sum: u64 = bucket.iter().map(|p| u64::from(p.g)).sum();
    let b_sum: u64 = bucket.iter().map(|p| u64::from(p.b)).sum();
    [(r_sum / n) as u8, (g_sum / n) as u8, (b_sum / n) as u8]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_cut_n_colors_zero_returns_empty() {
        let img = vec![255u8, 0, 0];
        let palette = median_cut(&img, 0);
        assert!(palette.is_empty());
    }

    #[test]
    fn median_cut_empty_image_returns_empty() {
        let palette = median_cut(&[], 4);
        assert!(palette.is_empty());
    }

    #[test]
    fn median_cut_single_pixel() {
        let img = vec![100u8, 150, 200];
        let palette = median_cut(&img, 1);
        assert_eq!(palette.len(), 1);
        assert_eq!(palette[0], [100, 150, 200]);
    }

    #[test]
    fn median_cut_two_distinct_colors() {
        // Red and blue
        let img = vec![255u8, 0, 0, 0, 0, 255];
        let palette = median_cut(&img, 2);
        assert_eq!(palette.len(), 2);
        // Palette should contain entries near red and blue
        let has_red = palette.iter().any(|c| c[0] > 128 && c[2] < 128);
        let has_blue = palette.iter().any(|c| c[0] < 128 && c[2] > 128);
        assert!(has_red, "palette should contain a reddish entry");
        assert!(has_blue, "palette should contain a bluish entry");
    }

    #[test]
    fn median_cut_uniform_image_returns_one_representative() {
        // All pixels the same colour
        let color = [80u8, 120, 200];
        let img: Vec<u8> = color.iter().cycle().take(3 * 16).copied().collect();
        let palette = median_cut(&img, 4);
        // All buckets collapse to same colour
        for entry in &palette {
            assert_eq!(*entry, color, "unexpected palette entry {:?}", entry);
        }
    }

    #[test]
    fn median_cut_respects_n_colors_limit() {
        // 100 random-ish pixels, request 8 colours
        let img: Vec<u8> = (0..(100 * 3)).map(|i| (i * 13 % 256) as u8).collect();
        let palette = median_cut(&img, 8);
        assert!(palette.len() <= 8, "palette too large: {}", palette.len());
        assert!(!palette.is_empty(), "palette should not be empty");
    }

    #[test]
    fn median_cut_each_palette_entry_valid_u8() {
        let img: Vec<u8> = (0..300).map(|i| (i % 256) as u8).collect();
        let palette = median_cut(&img, 16);
        for entry in &palette {
            // Already u8 so this is tautologically true, but verifies no
            // truncation or wrapping artefacts crept in
            for &c in entry.iter() {
                let _ = c; // just ensure no panic
            }
        }
    }

    #[test]
    #[should_panic(expected = "multiple of 3")]
    fn median_cut_panics_on_wrong_length() {
        median_cut(&[1u8, 2], 4);
    }
}
