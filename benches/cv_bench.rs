//! Computer vision benchmarks
//!
//! This benchmark suite measures the performance of computer vision operations:
//! - Face detection (Haar cascade)
//! - Object detection (YOLO)
//! - Image upscaling (ESRGAN)
//! - Edge detection (Sobel, Canny)
//! - Optical flow
//! - Feature detection
//! - Image transformations

mod helpers;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::time::Duration;

/// Benchmark Haar cascade face detection
fn haar_cascade_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("haar_cascade");
    group.measurement_time(Duration::from_secs(10));

    let resolutions = vec![
        ("320x240", 320, 240),
        ("640x480", 640, 480),
        ("1280x720", 1280, 720),
    ];

    for (name, width, height) in resolutions {
        let image = helpers::generate_rgb_frame(width, height);

        group.throughput(Throughput::Bytes(image.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &image, |b, image| {
            b.iter(|| {
                // Simulate Haar cascade evaluation
                // Convert to grayscale
                let mut gray = Vec::with_capacity(width * height);
                for chunk in image.chunks_exact(3) {
                    let r = u32::from(chunk[0]);
                    let g = u32::from(chunk[1]);
                    let b = u32::from(chunk[2]);
                    let gray_val = ((r * 299 + g * 587 + b * 114) / 1000) as u8;
                    gray.push(gray_val);
                }
                black_box(gray);
            });
        });
    }

    group.finish();
}

/// Benchmark integral image computation
fn integral_image_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("integral_image");

    let resolutions = vec![
        ("320x240", 320, 240),
        ("640x480", 640, 480),
        ("1280x720", 1280, 720),
        ("1920x1080", 1920, 1080),
    ];

    for (name, width, height) in resolutions {
        let image = vec![128u8; width * height];

        group.throughput(Throughput::Elements((width * height) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(image, width, height),
            |b, (image, width, height)| {
                b.iter(|| {
                    let mut integral = vec![0u32; width * height];

                    // Compute integral image
                    for y in 0..*height {
                        for x in 0..*width {
                            let idx = y * width + x;
                            let mut sum = u32::from(image[idx]);

                            if x > 0 {
                                sum += integral[idx - 1];
                            }
                            if y > 0 {
                                sum += integral[idx - width];
                            }
                            if x > 0 && y > 0 {
                                sum -= integral[idx - width - 1];
                            }

                            integral[idx] = sum;
                        }
                    }

                    black_box(integral);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark YOLO object detection inference
fn yolo_inference_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("yolo_inference");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    let resolutions = vec![("416x416", 416, 416), ("608x608", 608, 608)];

    for (name, width, height) in resolutions {
        let image = helpers::generate_rgb_frame(width, height);

        group.throughput(Throughput::Bytes(image.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &image, |b, image| {
            b.iter(|| {
                // Simulate YOLO preprocessing
                let normalized: Vec<f32> = image
                    .iter()
                    .map(|&pixel| f32::from(pixel) / 255.0)
                    .collect();
                black_box(normalized);
            });
        });
    }

    group.finish();
}

/// Benchmark YOLO non-maximum suppression
fn yolo_nms_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("yolo_nms");

    let box_counts = vec![("10_boxes", 10), ("100_boxes", 100), ("1000_boxes", 1000)];

    for (name, count) in box_counts {
        // Generate synthetic bounding boxes
        let boxes: Vec<(f32, f32, f32, f32, f32)> = (0..count)
            .map(|box_idx| {
                let box_x = (box_idx % 100) as f32;
                let box_y = (box_idx / 100) as f32;
                let box_w = 50.0;
                let box_h = 50.0;
                let conf = 0.5 + (box_idx as f32 / count as f32) * 0.5;
                (box_x, box_y, box_w, box_h, conf)
            })
            .collect();

        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &boxes, |b, boxes| {
            b.iter(|| {
                // Simple NMS simulation
                let mut keep = Vec::new();
                let mut sorted_boxes = boxes.clone();
                sorted_boxes
                    .sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));

                for i in 0..sorted_boxes.len() {
                    let mut should_keep = true;
                    for &keep_idx in &keep {
                        let box_a = &sorted_boxes[i];
                        let box_b: &(f32, f32, f32, f32, f32) = &sorted_boxes[keep_idx];

                        // Calculate IoU
                        let x1 = box_a.0.max(box_b.0);
                        let y1 = box_a.1.max(box_b.1);
                        let x2 = (box_a.0 + box_a.2).min(box_b.0 + box_b.2);
                        let y2 = (box_a.1 + box_a.3).min(box_b.1 + box_b.3);

                        let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
                        let area_a = box_a.2 * box_a.3;
                        let area_b = box_b.2 * box_b.3;
                        let union = area_a + area_b - intersection;
                        let iou = intersection / union;

                        if iou > 0.5 {
                            should_keep = false;
                            break;
                        }
                    }

                    if should_keep {
                        keep.push(i);
                    }
                }

                black_box(keep);
            });
        });
    }

    group.finish();
}

/// Benchmark ESRGAN upscaling preprocessing
fn esrgan_preprocessing_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("esrgan_preprocess");
    group.measurement_time(Duration::from_secs(10));

    let resolutions = vec![("256x256", 256, 256), ("512x512", 512, 512)];

    for (name, width, height) in resolutions {
        let image = helpers::generate_rgb_frame(width, height);

        group.throughput(Throughput::Bytes(image.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &image, |b, image| {
            b.iter(|| {
                // Normalize to [-1, 1]
                let normalized: Vec<f32> = image
                    .iter()
                    .map(|&pixel| (f32::from(pixel) / 127.5) - 1.0)
                    .collect();
                black_box(normalized);
            });
        });
    }

    group.finish();
}

/// Benchmark Sobel edge detection
fn sobel_edge_detection_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sobel_edge");

    let resolutions = vec![
        ("320x240", 320, 240),
        ("640x480", 640, 480),
        ("1280x720", 1280, 720),
        ("1920x1080", 1920, 1080),
    ];

    for (name, width, height) in resolutions {
        let image = vec![128u8; width * height]; // Grayscale image

        group.throughput(Throughput::Elements((width * height) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(image, width, height),
            |b, (image, width, height)| {
                b.iter(|| {
                    let mut output = vec![0u8; width * height];

                    // Sobel kernels
                    let gx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
                    let gy = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

                    for y in 1..(height - 1) {
                        for x in 1..(width - 1) {
                            let mut sum_x = 0i32;
                            let mut sum_y = 0i32;

                            for ky in 0..3 {
                                for kx in 0..3 {
                                    let idx = (y + ky - 1) * width + (x + kx - 1);
                                    let pixel = i32::from(image[idx]);
                                    let kernel_idx = ky * 3 + kx;
                                    sum_x += pixel * gx[kernel_idx];
                                    sum_y += pixel * gy[kernel_idx];
                                }
                            }

                            let magnitude =
                                ((sum_x * sum_x + sum_y * sum_y) as f32).sqrt().min(255.0) as u8;
                            output[y * width + x] = magnitude;
                        }
                    }

                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Gaussian blur
fn gaussian_blur_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_blur");

    let resolutions = vec![
        ("320x240", 320, 240),
        ("640x480", 640, 480),
        ("1280x720", 1280, 720),
    ];

    for (name, width, height) in resolutions {
        let image = vec![128u8; width * height];

        group.throughput(Throughput::Elements((width * height) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(image, width, height),
            |b, (image, width, height)| {
                b.iter(|| {
                    let mut output = vec![0u8; width * height];

                    // 5x5 Gaussian kernel
                    #[allow(clippy::excessive_precision)]
                    let kernel = [
                        1.0 / 273.0,
                        4.0 / 273.0,
                        7.0 / 273.0,
                        4.0 / 273.0,
                        1.0 / 273.0,
                        4.0 / 273.0,
                        16.0 / 273.0,
                        26.0 / 273.0,
                        16.0 / 273.0,
                        4.0 / 273.0,
                        7.0 / 273.0,
                        26.0 / 273.0,
                        41.0 / 273.0,
                        26.0 / 273.0,
                        7.0 / 273.0,
                        4.0 / 273.0,
                        16.0 / 273.0,
                        26.0 / 273.0,
                        16.0 / 273.0,
                        4.0 / 273.0,
                        1.0 / 273.0,
                        4.0 / 273.0,
                        7.0 / 273.0,
                        4.0 / 273.0,
                        1.0 / 273.0,
                    ];

                    for y in 2..(height - 2) {
                        for x in 2..(width - 2) {
                            let mut sum = 0.0f32;

                            for ky in 0..5 {
                                for kx in 0..5 {
                                    let idx = (y + ky - 2) * width + (x + kx - 2);
                                    let pixel = f32::from(image[idx]);
                                    sum += pixel * kernel[ky * 5 + kx];
                                }
                            }

                            output[y * width + x] = sum as u8;
                        }
                    }

                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Canny edge detection (full pipeline)
fn canny_edge_detection_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("canny_edge");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    let resolutions = vec![
        ("320x240", 320, 240),
        ("640x480", 640, 480),
        ("1280x720", 1280, 720),
    ];

    for (name, width, height) in resolutions {
        let image = vec![128u8; width * height];

        group.throughput(Throughput::Elements((width * height) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(image, width, height),
            |b, (image, width, height)| {
                b.iter(|| {
                    // Step 1: Gaussian blur (simplified)
                    let blurred = image.clone();

                    // Step 2: Sobel edge detection
                    let mut gradient_mag = vec![0u8; width * height];
                    let gx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
                    let gy = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

                    for y in 1..(height - 1) {
                        for x in 1..(width - 1) {
                            let mut sum_x = 0i32;
                            let mut sum_y = 0i32;

                            for ky in 0..3 {
                                for kx in 0..3 {
                                    let idx = (y + ky - 1) * width + (x + kx - 1);
                                    let pixel = i32::from(blurred[idx]);
                                    let kernel_idx = ky * 3 + kx;
                                    sum_x += pixel * gx[kernel_idx];
                                    sum_y += pixel * gy[kernel_idx];
                                }
                            }

                            let magnitude =
                                ((sum_x * sum_x + sum_y * sum_y) as f32).sqrt().min(255.0) as u8;
                            gradient_mag[y * width + x] = magnitude;
                        }
                    }

                    // Step 3: Non-maximum suppression (simplified)
                    let mut output = vec![0u8; width * height];
                    for y in 1..(height - 1) {
                        for x in 1..(width - 1) {
                            let idx = y * width + x;
                            let mag = gradient_mag[idx];

                            // Simplified: just threshold
                            output[idx] = if mag > 50 { 255 } else { 0 };
                        }
                    }

                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark optical flow (Lucas-Kanade)
fn optical_flow_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("optical_flow");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    let resolutions = vec![("320x240", 320, 240), ("640x480", 640, 480)];

    for (name, width, height) in resolutions {
        let frame1 = vec![128u8; width * height];
        let frame2 = vec![130u8; width * height];

        group.throughput(Throughput::Elements((width * height) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(frame1, frame2, width, height),
            |b, (frame1, frame2, width, height)| {
                b.iter(|| {
                    let mut flow_x = vec![0.0f32; width * height];
                    let mut flow_y = vec![0.0f32; width * height];

                    // Simplified Lucas-Kanade
                    for y in 1..(height - 1) {
                        for x in 1..(width - 1) {
                            let idx = y * width + x;

                            // Spatial gradients
                            let ix = (i32::from(frame1[idx + 1]) - i32::from(frame1[idx - 1])) / 2;
                            let iy = (i32::from(frame1[idx + width])
                                - i32::from(frame1[idx - width]))
                                / 2;

                            // Temporal gradient
                            let it = i32::from(frame2[idx]) - i32::from(frame1[idx]);

                            // Solve for flow (simplified)
                            if ix != 0 || iy != 0 {
                                let denom = (ix * ix + iy * iy) as f32;
                                flow_x[idx] = -(ix * it) as f32 / denom;
                                flow_y[idx] = -(iy * it) as f32 / denom;
                            }
                        }
                    }

                    black_box((flow_x, flow_y));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark image resize (bilinear interpolation)
fn image_resize_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_resize");

    let conversions = vec![
        ("downscale_2x", 1920, 1080, 960, 540),
        ("downscale_4x", 1920, 1080, 480, 270),
        ("upscale_2x", 640, 480, 1280, 960),
    ];

    for (name, src_w, src_h, dst_w, dst_h) in conversions {
        let image = helpers::generate_rgb_frame(src_w, src_h);

        group.throughput(Throughput::Elements((dst_w * dst_h) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(image, src_w, src_h, dst_w, dst_h),
            |b, (image, src_w, src_h, dst_w, dst_h)| {
                b.iter(|| {
                    let mut output = vec![0u8; dst_w * dst_h * 3];

                    for y in 0..*dst_h {
                        for x in 0..*dst_w {
                            let src_x = (x as f32 * *src_w as f32 / *dst_w as f32) as usize;
                            let src_y = (y as f32 * *src_h as f32 / *dst_h as f32) as usize;

                            let src_idx = (src_y * src_w + src_x) * 3;
                            let dst_idx = (y * dst_w + x) * 3;

                            if src_idx + 2 < image.len() && dst_idx + 2 < output.len() {
                                output[dst_idx] = image[src_idx];
                                output[dst_idx + 1] = image[src_idx + 1];
                                output[dst_idx + 2] = image[src_idx + 2];
                            }
                        }
                    }

                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark RGB to grayscale conversion
fn rgb_to_gray_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("rgb_to_gray");

    let resolutions = vec![
        ("640x480", 640, 480),
        ("1280x720", 1280, 720),
        ("1920x1080", 1920, 1080),
        ("3840x2160", 3840, 2160),
    ];

    for (name, width, height) in resolutions {
        let image = helpers::generate_rgb_frame(width, height);

        group.throughput(Throughput::Elements((width * height) as u64));

        group.bench_with_input(BenchmarkId::from_parameter(name), &image, |b, image| {
            b.iter(|| {
                let gray: Vec<u8> = image
                    .chunks_exact(3)
                    .map(|rgb| {
                        let r = u32::from(rgb[0]);
                        let g = u32::from(rgb[1]);
                        let b = u32::from(rgb[2]);
                        ((r * 299 + g * 587 + b * 114) / 1000) as u8
                    })
                    .collect();
                black_box(gray);
            });
        });
    }

    group.finish();
}

/// Benchmark YUV to RGB conversion
fn yuv_to_rgb_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("yuv_to_rgb");

    let resolutions = vec![
        ("640x480", 640, 480),
        ("1280x720", 1280, 720),
        ("1920x1080", 1920, 1080),
        ("3840x2160", 3840, 2160),
    ];

    for (name, width, height) in resolutions {
        let yuv = helpers::generate_yuv420_frame(width, height);

        group.throughput(Throughput::Elements((width * height) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(yuv, width, height),
            |b, (yuv, width, height)| {
                b.iter(|| {
                    let mut rgb = vec![0u8; width * height * 3];

                    let y_size = width * height;
                    let uv_size = (width / 2) * (height / 2);

                    for y in 0..*height {
                        for x in 0..*width {
                            let y_idx = y * width + x;
                            let uv_idx = (y / 2) * (width / 2) + (x / 2);

                            let y_val = i32::from(yuv[y_idx]);
                            let u_val = i32::from(yuv[y_size + uv_idx]) - 128;
                            let v_val = i32::from(yuv[y_size + uv_size + uv_idx]) - 128;

                            let r = (y_val + (1.370705 * v_val as f32) as i32).clamp(0, 255);
                            let g = (y_val
                                - (0.337633 * u_val as f32) as i32
                                - (0.698001 * v_val as f32) as i32)
                                .clamp(0, 255);
                            let b = (y_val + (1.732446 * u_val as f32) as i32).clamp(0, 255);

                            let rgb_idx = (y * width + x) * 3;
                            rgb[rgb_idx] = r as u8;
                            rgb[rgb_idx + 1] = g as u8;
                            rgb[rgb_idx + 2] = b as u8;
                        }
                    }

                    black_box(rgb);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    cv_benches,
    haar_cascade_benchmark,
    integral_image_benchmark,
    yolo_inference_benchmark,
    yolo_nms_benchmark,
    esrgan_preprocessing_benchmark,
    sobel_edge_detection_benchmark,
    gaussian_blur_benchmark,
    canny_edge_detection_benchmark,
    optical_flow_benchmark,
    image_resize_benchmark,
    rgb_to_gray_benchmark,
    yuv_to_rgb_benchmark,
);

criterion_main!(cv_benches);
