//! Image processing operations example.
//!
//! This example demonstrates various image processing operations including:
//! - Image resizing with different interpolation methods
//! - Color space conversions (RGB to grayscale, RGB to YUV, etc.)
//! - Image filtering (blur, sharpen, edge detection)
//!
//! All operations use synthetic test images, so no external files are required.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example image_processing
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::unnecessary_wraps)]

use oximedia_cv::image::{ColorSpace, ResizeMethod};

/// Create a synthetic RGB test image with a gradient pattern.
///
/// # Arguments
///
/// * `width` - Image width
/// * `height` - Image height
///
/// # Returns
///
/// RGB image data (interleaved) with width, height
fn create_rgb_gradient(width: u32, height: u32) -> (Vec<u8>, u32, u32) {
    let mut image = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;

            // Create RGB gradient
            image[idx] = ((x * 255) / width) as u8; // Red channel
            image[idx + 1] = ((y * 255) / height) as u8; // Green channel
            image[idx + 2] = 128; // Blue channel (constant)
        }
    }

    (image, width, height)
}

/// Create a synthetic RGB test image with colored squares.
fn create_rgb_squares(width: u32, height: u32) -> (Vec<u8>, u32, u32) {
    let mut image = vec![0u8; (width * height * 3) as usize];

    let half_w = width / 2;
    let half_h = height / 2;

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;

            // Red square (top-left)
            if x < half_w && y < half_h {
                image[idx] = 255; // R
                image[idx + 1] = 0; // G
                image[idx + 2] = 0; // B
            }
            // Green square (top-right)
            else if x >= half_w && y < half_h {
                image[idx] = 0;
                image[idx + 1] = 255;
                image[idx + 2] = 0;
            }
            // Blue square (bottom-left)
            else if x < half_w && y >= half_h {
                image[idx] = 0;
                image[idx + 1] = 0;
                image[idx + 2] = 255;
            }
            // Yellow square (bottom-right)
            else {
                image[idx] = 255;
                image[idx + 1] = 255;
                image[idx + 2] = 0;
            }
        }
    }

    (image, width, height)
}

/// Convert RGB to grayscale manually (for demonstration).
fn rgb_to_grayscale_demo(rgb: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut gray = vec![0u8; (width * height) as usize];

    for i in 0..(width * height) as usize {
        let r = f32::from(rgb[i * 3]);
        let g = f32::from(rgb[i * 3 + 1]);
        let b = f32::from(rgb[i * 3 + 2]);

        // BT.709 coefficients
        let y = (0.2126 * r + 0.7152 * g + 0.0722 * b) as u8;
        gray[i] = y;
    }

    gray
}

/// Resize image using nearest neighbor (for demonstration).
fn resize_nearest_demo(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let mut dst = vec![0u8; (dst_w * dst_h) as usize];

    let x_ratio = src_w as f32 / dst_w as f32;
    let y_ratio = src_h as f32 / dst_h as f32;

    for y in 0..dst_h {
        for x in 0..dst_w {
            let src_x = ((x as f32 * x_ratio) as u32).min(src_w - 1);
            let src_y = ((y as f32 * y_ratio) as u32).min(src_h - 1);

            let src_idx = (src_y * src_w + src_x) as usize;
            let dst_idx = (y * dst_w + x) as usize;

            dst[dst_idx] = src[src_idx];
        }
    }

    dst
}

/// Apply simple box blur (for demonstration).
fn box_blur_demo(src: &[u8], width: u32, height: u32, radius: u32) -> Vec<u8> {
    let mut dst = vec![0u8; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0u32;
            let mut count = 0u32;

            for dy in -(radius as i32)..=(radius as i32) {
                for dx in -(radius as i32)..=(radius as i32) {
                    let nx = (x as i32 + dx).max(0).min(width as i32 - 1) as u32;
                    let ny = (y as i32 + dy).max(0).min(height as i32 - 1) as u32;

                    sum += u32::from(src[(ny * width + nx) as usize]);
                    count += 1;
                }
            }

            dst[(y * width + x) as usize] = (sum / count) as u8;
        }
    }

    dst
}

/// Calculate simple image statistics.
fn calculate_stats(img: &[u8]) -> (u8, u8, f64) {
    if img.is_empty() {
        return (0, 0, 0.0);
    }

    let min = img.iter().copied().min().unwrap_or(0);
    let max = img.iter().copied().max().unwrap_or(0);
    let sum: u64 = img.iter().map(|&x| u64::from(x)).sum();
    let mean = sum as f64 / img.len() as f64;

    (min, max, mean)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiMedia Image Processing Example");
    println!("==================================\n");

    // Test 1: Image creation and color spaces
    println!("Test 1: Color Space Information");
    println!("--------------------------------\n");

    let rgb = ColorSpace::Rgb;
    let yuv = ColorSpace::YuvBt709;
    let gray = ColorSpace::Grayscale;

    println!("  RGB color space:");
    println!("    Channels: {}", rgb.channels());
    println!("    Luminance-based: {}", rgb.is_luminance_based());

    println!("\n  YUV (BT.709) color space:");
    println!("    Channels: {}", yuv.channels());
    println!("    Luminance-based: {}", yuv.is_luminance_based());

    println!("\n  Grayscale:");
    println!("    Channels: {}", gray.channels());
    println!("    Luminance-based: {}", gray.is_luminance_based());
    println!();

    // Test 2: Resize methods
    println!("Test 2: Resize Method Information");
    println!("----------------------------------\n");

    let methods = [
        ResizeMethod::Nearest,
        ResizeMethod::Bilinear,
        ResizeMethod::Bicubic,
        ResizeMethod::Lanczos,
        ResizeMethod::Area,
    ];

    for method in &methods {
        println!("  {method:?}:");
        println!("    Interpolating: {}", method.is_interpolating());
        println!("    Kernel size: {}", method.kernel_size());
    }
    println!();

    // Test 3: RGB to Grayscale conversion
    println!("Test 3: RGB to Grayscale Conversion");
    println!("------------------------------------\n");

    let (rgb_img, width, height) = create_rgb_squares(64, 64);
    println!(
        "  Original RGB image: {}x{} ({} bytes)",
        width,
        height,
        rgb_img.len()
    );

    let gray_img = rgb_to_grayscale_demo(&rgb_img, width, height);
    println!(
        "  Grayscale image: {}x{} ({} bytes)",
        width,
        height,
        gray_img.len()
    );

    let (min, max, mean) = calculate_stats(&gray_img);
    println!("  Statistics:");
    println!("    Min: {min}");
    println!("    Max: {max}");
    println!("    Mean: {mean:.2}");
    println!();

    // Test 4: Image resizing
    println!("Test 4: Image Resizing (Nearest Neighbor)");
    println!("------------------------------------------\n");

    let (gradient_img, orig_w, orig_h) = create_rgb_gradient(32, 32);
    let gray_gradient = rgb_to_grayscale_demo(&gradient_img, orig_w, orig_h);

    println!("  Original size: {orig_w}x{orig_h}");

    // Upscale
    let upscaled = resize_nearest_demo(&gray_gradient, orig_w, orig_h, 64, 64);
    println!("  Upscaled to: 64x64 ({} bytes)", upscaled.len());

    // Downscale
    let downscaled = resize_nearest_demo(&gray_gradient, orig_w, orig_h, 16, 16);
    println!("  Downscaled to: 16x16 ({} bytes)", downscaled.len());

    let (min_up, max_up, mean_up) = calculate_stats(&upscaled);
    println!("  Upscaled statistics: min={min_up}, max={max_up}, mean={mean_up:.2}");

    let (min_down, max_down, mean_down) = calculate_stats(&downscaled);
    println!("  Downscaled statistics: min={min_down}, max={max_down}, mean={mean_down:.2}");
    println!();

    // Test 5: Box blur filter
    println!("Test 5: Box Blur Filter");
    println!("------------------------\n");

    let (test_img, tw, th) = create_rgb_gradient(48, 48);
    let test_gray = rgb_to_grayscale_demo(&test_img, tw, th);

    println!("  Original image: {tw}x{th}");
    let (min_orig, max_orig, mean_orig) = calculate_stats(&test_gray);
    println!("  Original statistics: min={min_orig}, max={max_orig}, mean={mean_orig:.2}");

    // Apply blur with radius 2
    let blurred = box_blur_demo(&test_gray, tw, th, 2);
    let (min_blur, max_blur, mean_blur) = calculate_stats(&blurred);
    println!("\n  After box blur (radius=2):");
    println!("  Blurred statistics: min={min_blur}, max={max_blur}, mean={mean_blur:.2}");
    println!("  Note: Blur reduces extremes (min increases, max decreases)");
    println!();

    // Test 6: Image gradient calculation
    println!("Test 6: Simple Image Gradient");
    println!("------------------------------\n");

    let (square_img, sw, sh) = create_rgb_squares(32, 32);
    let square_gray = rgb_to_grayscale_demo(&square_img, sw, sh);

    // Calculate horizontal gradient (simple approximation)
    let mut grad_x = vec![0u8; (sw * sh) as usize];
    for y in 0..sh {
        for x in 1..sw - 1 {
            let idx = (y * sw + x) as usize;
            let left = square_gray[(y * sw + (x - 1)) as usize];
            let right = square_gray[(y * sw + (x + 1)) as usize];
            let diff = ((i16::from(right) - i16::from(left)).abs() / 2) as u8;
            grad_x[idx] = diff;
        }
    }

    println!("  Image size: {sw}x{sh}");
    let (_, max_grad, mean_grad) = calculate_stats(&grad_x);
    println!("  Horizontal gradient:");
    println!("    Max gradient: {max_grad}");
    println!("    Mean gradient: {mean_grad:.2}");
    println!("  High gradients indicate edges between colored squares");
    println!();

    println!("Image processing test completed successfully!");
    println!("\nDemonstrated operations:");
    println!("- Color space conversions (RGB to Grayscale)");
    println!("- Image resizing (nearest neighbor upscaling/downscaling)");
    println!("- Image filtering (box blur)");
    println!("- Edge detection (simple gradient)");
    println!("\nAll operations work with synthetic test images.");

    Ok(())
}
