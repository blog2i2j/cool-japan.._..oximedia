//! Debug utilities for IVTC analysis.

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_truncation)]

use oximedia_codec::{Plane, VideoFrame};

use super::analysis::FieldMetrics;

/// Generate a visual representation of field combing.
#[must_use]
pub fn visualize_combing(frame: &VideoFrame) -> Option<VideoFrame> {
    if frame.planes.is_empty() {
        return None;
    }

    let mut output = VideoFrame::new(frame.format, frame.width, frame.height);
    output.timestamp = frame.timestamp;
    output.frame_type = frame.frame_type;
    output.color_info = frame.color_info;

    let plane = &frame.planes[0];
    let height = frame.height as usize;
    let width = frame.width as usize;

    let mut dst_data = vec![0u8; (width * height) as usize];

    for y in 1..height - 1 {
        let row_prev = plane.row(y - 1);
        let row_curr = plane.row(y);
        let row_next = plane.row(y + 1);

        for x in 0..width {
            let prev = row_prev.get(x).copied().unwrap_or(0) as i32;
            let curr = row_curr.get(x).copied().unwrap_or(0) as i32;
            let next = row_next.get(x).copied().unwrap_or(0) as i32;

            let interp = (prev + next) / 2;
            let diff = (curr - interp).abs();

            // Highlight combing artifacts
            let vis = if diff > 15 { 255 } else { (diff * 10).min(255) };

            dst_data[y * width + x] = vis as u8;
        }
    }

    output.planes.push(Plane::new(dst_data, width));
    Some(output)
}

/// Generate field separation visualization.
#[must_use]
pub fn visualize_fields(frame: &VideoFrame, field: u8) -> Option<VideoFrame> {
    if frame.planes.is_empty() {
        return None;
    }

    let mut output = VideoFrame::new(frame.format, frame.width, frame.height);
    output.timestamp = frame.timestamp;
    output.frame_type = frame.frame_type;
    output.color_info = frame.color_info;

    let plane = &frame.planes[0];
    let height = frame.height as usize;
    let width = frame.width as usize;

    let mut dst_data = vec![0u8; (width * height) as usize];

    for y in 0..height {
        let row = plane.row(y);

        for x in 0..width {
            let pixel = if (y % 2) == field as usize {
                row.get(x).copied().unwrap_or(0)
            } else {
                128 // Gray for opposite field
            };

            dst_data[y * width + x] = pixel;
        }
    }

    output.planes.push(Plane::new(dst_data, width));
    Some(output)
}

/// Calculate detailed statistics for debugging.
#[must_use]
pub fn calculate_debug_stats(frame: &VideoFrame) -> DebugStats {
    let metrics = FieldMetrics::calculate(frame);

    DebugStats {
        width: frame.width,
        height: frame.height,
        field_metrics: metrics,
        plane_count: frame.planes.len(),
    }
}

/// Debug statistics.
#[derive(Clone, Debug)]
pub struct DebugStats {
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Field metrics.
    pub field_metrics: FieldMetrics,
    /// Number of planes.
    pub plane_count: usize,
}
