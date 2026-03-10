#![allow(dead_code)]
//! Deinterlacing configuration and processing helpers.

/// The field order of an interlaced video signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldOrder {
    /// Top (odd) field is displayed first.
    TopFieldFirst,
    /// Bottom (even) field is displayed first.
    BottomFieldFirst,
    /// The material is already progressive — no deinterlacing needed.
    Progressive,
}

/// Algorithm used to convert interlaced fields to progressive frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeinterlaceMethod {
    /// Drop one field and scale the remaining field up — fastest, lowest quality.
    FieldDrop,
    /// Blend the two fields together — simple, introduces motion blur.
    Blend,
    /// Bob deinterlacing: each field becomes a full progressive frame.
    Bob,
    /// Weave: combine two consecutive fields into one frame (motion artifacts).
    Weave,
    /// Motion-adaptive deinterlacing — high quality, more compute.
    MotionAdaptive,
    /// Yadif algorithm: temporal + spatial interpolation.
    Yadif,
}

impl DeinterlaceMethod {
    /// Frame-rate multiplier relative to the input field rate.
    ///
    /// `Bob` produces one output frame per field (×2), `Yadif` can be
    /// configured to do the same.  All other methods produce one output frame
    /// per two fields (×1).
    pub fn output_frame_rate_multiplier(&self) -> u32 {
        match self {
            DeinterlaceMethod::Bob | DeinterlaceMethod::Yadif => 2,
            _ => 1,
        }
    }
}

/// Configuration for a deinterlacing operation.
#[derive(Debug, Clone)]
pub struct DeinterlaceConfig {
    /// The field order of the input material.
    pub field_order: FieldOrder,
    /// The deinterlacing algorithm to apply.
    pub method: DeinterlaceMethod,
    /// Number of threads to use for processing (0 = auto).
    pub threads: u32,
}

impl DeinterlaceConfig {
    /// Create a new [`DeinterlaceConfig`] with default thread count (0 = auto).
    pub fn new(field_order: FieldOrder, method: DeinterlaceMethod) -> Self {
        Self {
            field_order,
            method,
            threads: 0,
        }
    }

    /// Whether the chosen method uses temporal information (i.e. references
    /// more than one field).
    pub fn is_temporal(&self) -> bool {
        matches!(
            self.method,
            DeinterlaceMethod::MotionAdaptive | DeinterlaceMethod::Yadif
        )
    }

    /// Whether the input is already progressive (no processing needed).
    pub fn is_progressive_passthrough(&self) -> bool {
        self.field_order == FieldOrder::Progressive
    }
}

/// A single video field extracted from an interlaced frame.
#[derive(Debug, Clone)]
pub struct VideoField {
    /// Which field this is (0 = top / odd, 1 = bottom / even).
    pub field_index: u8,
    /// Width of the field in pixels.
    pub width: u32,
    /// Height of the field in pixels (half the frame height).
    pub height: u32,
    /// Raw luma byte data (Y plane only for simplicity).
    pub luma: Vec<u8>,
}

impl VideoField {
    /// Create a new [`VideoField`] with blank luma.
    pub fn blank(field_index: u8, width: u32, height: u32) -> Self {
        Self {
            field_index,
            width,
            height,
            luma: vec![0u8; (width * height) as usize],
        }
    }
}

/// Processes video fields into progressive frames.
#[derive(Debug)]
pub struct DeinterlaceProcessor {
    config: DeinterlaceConfig,
}

impl DeinterlaceProcessor {
    /// Create a new [`DeinterlaceProcessor`].
    pub fn new(config: DeinterlaceConfig) -> Self {
        Self { config }
    }

    /// Access the current configuration.
    pub fn config(&self) -> &DeinterlaceConfig {
        &self.config
    }

    /// Process a single [`VideoField`] and return a progressive frame as raw
    /// luma bytes (full-height output, one byte per pixel row × width).
    ///
    /// For `FieldDrop` the field luma is returned as-is (half-height).
    /// For `Bob` and `Blend` the missing interlaced lines are reconstructed by
    /// averaging the two spatially adjacent field lines (bob deinterlacing).
    /// The edge lines are handled by duplicating the nearest available line.
    /// For temporal methods (`MotionAdaptive`, `Yadif`) a single-field fallback
    /// using the same bob approach is applied (temporal neighbours are needed for
    /// full quality but are not available in this single-field interface).
    pub fn process_field(&self, field: &VideoField) -> Vec<u8> {
        if self.config.is_progressive_passthrough() {
            return field.luma.clone();
        }

        match self.config.method {
            DeinterlaceMethod::FieldDrop => {
                // Return field data unchanged (caller would up-scale in practice).
                field.luma.clone()
            }
            DeinterlaceMethod::Bob
            | DeinterlaceMethod::Blend
            | DeinterlaceMethod::Weave
            | DeinterlaceMethod::MotionAdaptive
            | DeinterlaceMethod::Yadif => {
                // Bob deinterlacing: reconstruct a full-height progressive frame
                // from a single field.
                //
                // field_index == 0  → top/even field  → rows 0, 2, 4, …  are known
                //                                      → rows 1, 3, 5, …  are missing
                // field_index == 1  → bottom/odd field → rows 0, 2, 4, …  are missing
                //                                      → rows 1, 3, 5, …  are known
                //
                // The field luma stores only the field's own lines packed
                // sequentially (height rows of width bytes each).  The output is
                // a full progressive frame with 2×height rows.
                bob_deinterlace(field)
            }
        }
    }
}

/// Reconstruct a full-height progressive frame from a single interlaced field
/// using bob deinterlacing.
///
/// The `field.luma` buffer contains `field.height` rows of `field.width` bytes,
/// each row representing one spatial line belonging to this field.
///
/// Output layout (2 × `field.height` rows):
///
/// * Even field (`field_index == 0`): known lines sit at even output row indices
///   (0, 2, 4, …).  Missing odd lines are interpolated.
/// * Odd field (`field_index == 1`): known lines sit at odd output row indices
///   (1, 3, 5, …).  Missing even lines are interpolated.
///
/// Interpolation rule:
/// * Interior missing line: average of the two adjacent known lines.
/// * Edge missing line (top or bottom): duplicate the nearest known line.
fn bob_deinterlace(field: &VideoField) -> Vec<u8> {
    let w = field.width as usize;
    let fh = field.height as usize; // number of lines in this field
    let full_h = fh * 2; // output progressive frame height

    let mut out = vec![0u8; w * full_h];

    // Determine which output rows belong to this field.
    // field_index 0 → even rows (0, 2, 4, …)
    // field_index 1 → odd  rows (1, 3, 5, …)
    let field_row_offset = (field.field_index & 1) as usize; // 0 or 1

    // Copy known field lines into their output positions.
    for fi in 0..fh {
        let out_row = field_row_offset + fi * 2;
        let src_start = fi * w;
        let dst_start = out_row * w;
        out[dst_start..dst_start + w].copy_from_slice(&field.luma[src_start..src_start + w]);
    }

    // Fill in the missing (interpolated) rows.
    // Missing rows are those at the opposite parity: start at `1 - field_row_offset`.
    let missing_offset = 1 - field_row_offset;
    for mi in 0..fh {
        let missing_row = missing_offset + mi * 2;

        // The two adjacent known rows in the output.
        // Since known rows are spaced 2 apart, the neighbours are:
        //   above known row: missing_row - 1  (if ≥ 0)
        //   below known row: missing_row + 1  (if < full_h)
        let above_known = missing_row.checked_sub(1);
        let below_known = {
            let r = missing_row + 1;
            if r < full_h {
                Some(r)
            } else {
                None
            }
        };

        let dst_start = missing_row * w;

        match (above_known, below_known) {
            (Some(above), Some(below)) => {
                // Interior: average the two adjacent known lines.
                // Collect the averaged values first to avoid simultaneous borrows.
                let above_start = above * w;
                let below_start = below * w;
                let averaged: Vec<u8> = (0..w)
                    .map(|x| {
                        let a = out[above_start + x] as u16;
                        let b = out[below_start + x] as u16;
                        ((a + b + 1) / 2) as u8
                    })
                    .collect();
                out[dst_start..dst_start + w].copy_from_slice(&averaged);
            }
            (None, Some(below)) => {
                // Top edge: duplicate the line below.
                // dst_start < below_start always holds here because the missing
                // row is above the first known row.
                let below_start = below * w;
                // Split so that the source and destination slices are disjoint.
                // dst_start is always 0 for the top-edge case, and below_start > dst_start.
                let (left, right) = out.split_at_mut(below_start);
                left[dst_start..dst_start + w].copy_from_slice(&right[..w]);
            }
            (Some(above), None) => {
                // Bottom edge: duplicate the line above.
                // The above row is already written, so we can read and write
                // from separate slices by splitting.
                let above_start = above * w;
                // Both above_start and dst_start are in `out`, but dst_start > above_start.
                let (left, right) = out.split_at_mut(dst_start);
                right[..w].copy_from_slice(&left[above_start..above_start + w]);
            }
            (None, None) => {
                // Single-line degenerate case: leave as zero.
            }
        }
    }

    out
}

impl DeinterlaceProcessor {
    /// Output frame rate given the input frame rate (in fps numerator/denominator).
    #[allow(clippy::cast_precision_loss)]
    pub fn output_fps(&self, input_fps_num: u32, input_fps_den: u32) -> f64 {
        let multiplier = self.config.method.output_frame_rate_multiplier();
        (input_fps_num as f64 / input_fps_den as f64) * multiplier as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_order_variants() {
        let tff = FieldOrder::TopFieldFirst;
        let bff = FieldOrder::BottomFieldFirst;
        let prog = FieldOrder::Progressive;
        assert_ne!(tff, bff);
        assert_ne!(tff, prog);
    }

    #[test]
    fn test_frame_rate_multiplier_bob() {
        assert_eq!(DeinterlaceMethod::Bob.output_frame_rate_multiplier(), 2);
    }

    #[test]
    fn test_frame_rate_multiplier_yadif() {
        assert_eq!(DeinterlaceMethod::Yadif.output_frame_rate_multiplier(), 2);
    }

    #[test]
    fn test_frame_rate_multiplier_blend() {
        assert_eq!(DeinterlaceMethod::Blend.output_frame_rate_multiplier(), 1);
    }

    #[test]
    fn test_frame_rate_multiplier_field_drop() {
        assert_eq!(
            DeinterlaceMethod::FieldDrop.output_frame_rate_multiplier(),
            1
        );
    }

    #[test]
    fn test_config_is_temporal_true() {
        let cfg =
            DeinterlaceConfig::new(FieldOrder::TopFieldFirst, DeinterlaceMethod::MotionAdaptive);
        assert!(cfg.is_temporal());
    }

    #[test]
    fn test_config_is_temporal_false() {
        let cfg = DeinterlaceConfig::new(FieldOrder::TopFieldFirst, DeinterlaceMethod::Blend);
        assert!(!cfg.is_temporal());
    }

    #[test]
    fn test_progressive_passthrough() {
        let cfg = DeinterlaceConfig::new(FieldOrder::Progressive, DeinterlaceMethod::FieldDrop);
        assert!(cfg.is_progressive_passthrough());
    }

    #[test]
    fn test_not_progressive_passthrough() {
        let cfg = DeinterlaceConfig::new(FieldOrder::TopFieldFirst, DeinterlaceMethod::FieldDrop);
        assert!(!cfg.is_progressive_passthrough());
    }

    #[test]
    fn test_process_field_progressive_passthrough() {
        let cfg = DeinterlaceConfig::new(FieldOrder::Progressive, DeinterlaceMethod::Bob);
        let proc = DeinterlaceProcessor::new(cfg);
        let field = VideoField::blank(0, 4, 4);
        let out = proc.process_field(&field);
        assert_eq!(out, vec![0u8; 16]);
    }

    #[test]
    fn test_process_field_field_drop() {
        let cfg = DeinterlaceConfig::new(FieldOrder::TopFieldFirst, DeinterlaceMethod::FieldDrop);
        let proc = DeinterlaceProcessor::new(cfg);
        let mut field = VideoField::blank(0, 2, 2);
        field.luma = vec![10, 20, 30, 40];
        let out = proc.process_field(&field);
        assert_eq!(out, vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_process_field_blend() {
        // Blend now uses bob deinterlacing instead of the old halve-each-sample stub.
        // field_index=1 (bottom/odd field), width=2, height=2 field-lines.
        // luma = [row0=[100,200], row1=[50,150]]
        // Bob produces a full-height (4-row) progressive frame:
        //   output row 0 (missing, top edge): duplicate of output row 1 → [100, 200]
        //   output row 1 (known):             field line 0              → [100, 200]
        //   output row 2 (missing, interior): avg of rows 1 & 3        → [75,  175]
        //   output row 3 (known):             field line 1              → [50,  150]
        let cfg = DeinterlaceConfig::new(FieldOrder::BottomFieldFirst, DeinterlaceMethod::Blend);
        let proc = DeinterlaceProcessor::new(cfg);
        let mut field = VideoField::blank(1, 2, 2);
        field.luma = vec![100, 200, 50, 150];
        let out = proc.process_field(&field);
        assert_eq!(
            out.len(),
            8,
            "bob output must be full-height (4 rows × 2 px)"
        );
        assert_eq!(
            out[0..2],
            [100, 200],
            "row 0: top-edge duplication of row 1"
        );
        assert_eq!(out[2..4], [100, 200], "row 1: known field line 0");
        assert_eq!(out[4..6], [75, 175], "row 2: average of rows 1 and 3");
        assert_eq!(out[6..8], [50, 150], "row 3: known field line 1");
    }

    #[test]
    fn test_output_fps_bob() {
        let cfg = DeinterlaceConfig::new(FieldOrder::TopFieldFirst, DeinterlaceMethod::Bob);
        let proc = DeinterlaceProcessor::new(cfg);
        // 25 fps interlaced → 50 fps progressive with Bob
        let fps = proc.output_fps(25, 1);
        assert!((fps - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_output_fps_blend() {
        let cfg = DeinterlaceConfig::new(FieldOrder::TopFieldFirst, DeinterlaceMethod::Blend);
        let proc = DeinterlaceProcessor::new(cfg);
        let fps = proc.output_fps(25, 1);
        assert!((fps - 25.0).abs() < 1e-9);
    }
}
