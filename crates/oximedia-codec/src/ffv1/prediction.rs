//! FFV1 prediction filters.
//!
//! FFV1 uses a median predictor for spatial prediction within each plane.
//! The predictor uses three neighboring samples: left, top, and top-left.
//! The residual (prediction error) is then entropy coded.

/// Median predictor as specified in RFC 9043 Section 3.4.
///
/// Given three neighboring samples:
/// - `left`:     sample at (x-1, y)
/// - `top`:      sample at (x, y-1)
/// - `top_left`: sample at (x-1, y-1)
///
/// The prediction is `median(left, top, left + top - top_left)`.
///
/// This is equivalent to:
/// - If top_left >= max(left, top): min(left, top)
/// - If top_left <= min(left, top): max(left, top)
/// - Otherwise: left + top - top_left
#[inline]
#[must_use]
pub fn predict_median(left: i32, top: i32, top_left: i32) -> i32 {
    let prediction = left.wrapping_add(top).wrapping_sub(top_left);
    if left >= top {
        if top_left >= left {
            top
        } else if top_left <= top {
            left
        } else {
            prediction
        }
    } else {
        if top_left >= top {
            left
        } else if top_left <= left {
            top
        } else {
            prediction
        }
    }
}

/// Encode a line of samples into residuals using median prediction.
///
/// For each sample in `line`, compute the median prediction from the
/// `above` line and previously decoded samples in this line, then
/// store the signed residual.
///
/// At position x=0, `left` is 0 (or the last sample of the previous line
/// depending on context, but FFV1 uses 0 for left at line start).
///
/// `above` must have at least `line.len()` elements. If this is the first
/// line, `above` should be all zeros.
pub fn encode_line(line: &[i32], above: &[i32], residuals: &mut Vec<i32>) {
    residuals.clear();
    residuals.reserve(line.len());

    let width = line.len();
    if width == 0 {
        return;
    }

    for x in 0..width {
        let left = if x > 0 { line[x - 1] } else { 0 };
        let top = above[x];
        let top_left = if x > 0 { above[x - 1] } else { 0 };

        let pred = predict_median(left, top, top_left);
        let residual = line[x] - pred;
        residuals.push(residual);
    }
}

/// Decode a line of samples from residuals using median prediction.
///
/// Reverses the encoding: for each residual, compute the median prediction
/// from the `above` line and previously reconstructed samples in the
/// output line, then add the residual.
///
/// `above` must have at least `residuals.len()` elements. If this is the
/// first line, `above` should be all zeros.
pub fn decode_line(residuals: &[i32], above: &[i32], output: &mut Vec<i32>) {
    output.clear();
    output.reserve(residuals.len());

    let width = residuals.len();
    if width == 0 {
        return;
    }

    for x in 0..width {
        let left = if x > 0 { output[x - 1] } else { 0 };
        let top = above[x];
        let top_left = if x > 0 { above[x - 1] } else { 0 };

        let pred = predict_median(left, top, top_left);
        let sample = pred + residuals[x];
        output.push(sample);
    }
}

/// Compute the context (quantized gradient) for a sample position.
///
/// FFV1 uses five context values derived from local gradients to select
/// the entropy coding context. This function computes a single combined
/// context index from the neighboring sample differences.
///
/// The context is used to select adaptive probability states for the
/// range coder (or run/k parameters for Golomb-Rice).
///
/// Returns a context index in [0, CONTEXT_COUNT).
#[inline]
#[must_use]
pub fn compute_context(top: i32, left: i32, top_left: i32, top_right: i32) -> usize {
    // Gradient components
    let d_top = top - top_left;
    let d_left = left - top_left;
    let d_top_right = top_right - top;

    // Quantize gradients into small ranges
    let q0 = quantize_gradient(d_top);
    let q1 = quantize_gradient(d_left);
    let q2 = quantize_gradient(d_top_right);

    // Combine into a single context index.
    // We fold negative contexts by negating all gradients (sign flipping),
    // so the context index is always non-negative.
    let mut ctx = q0 * 25 + q1 * 5 + q2;

    // If context is negative, flip sign. The decoder will flip the sign
    // of the decoded residual as well.
    if ctx < 0 {
        ctx = -ctx;
    }

    // Clamp to valid context range (0..31)
    (ctx as usize) % super::types::CONTEXT_COUNT
}

/// Quantize a gradient value into a small integer for context derivation.
///
/// Maps arbitrary gradient values into the range [-2, 2] using the
/// quantization thresholds from the FFV1 spec.
#[inline]
fn quantize_gradient(d: i32) -> i32 {
    if d == 0 {
        0
    } else if d < -2 {
        -2
    } else if d < 0 {
        -1
    } else if d > 2 {
        2
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_predict_median_basic() {
        // When all neighbors are the same, prediction equals that value
        assert_eq!(predict_median(100, 100, 100), 100);

        // left=100, top=200, top_left=150 => pred = 100+200-150 = 150
        // Since 100 < 200 and 150 is between them, result is 150
        assert_eq!(predict_median(100, 200, 150), 150);
    }

    #[test]
    #[ignore]
    fn test_predict_median_edge_cases() {
        // left < top: top_left >= top => min(left, top) = left
        assert_eq!(predict_median(10, 20, 30), 10);

        // left < top: top_left <= left => max(left, top) = top
        assert_eq!(predict_median(10, 20, 5), 20);
    }

    #[test]
    #[ignore]
    fn test_predict_median_symmetric() {
        // left >= top case
        assert_eq!(predict_median(200, 100, 50), 200);
        // top_left >= left => top
        assert_eq!(predict_median(200, 100, 250), 100);
    }

    #[test]
    #[ignore]
    fn test_encode_decode_roundtrip() {
        let line = vec![10, 20, 30, 40, 50];
        let above = vec![5, 15, 25, 35, 45];
        let mut residuals = Vec::new();
        let mut decoded = Vec::new();

        encode_line(&line, &above, &mut residuals);
        decode_line(&residuals, &above, &mut decoded);

        assert_eq!(line, decoded);
    }

    #[test]
    #[ignore]
    fn test_encode_decode_first_line() {
        let line = vec![128, 130, 132, 134, 136];
        let above = vec![0, 0, 0, 0, 0]; // first line
        let mut residuals = Vec::new();
        let mut decoded = Vec::new();

        encode_line(&line, &above, &mut residuals);
        decode_line(&residuals, &above, &mut decoded);

        assert_eq!(line, decoded);
    }

    #[test]
    #[ignore]
    fn test_encode_decode_constant_line() {
        // All same value: residuals should be small
        let line = vec![100; 8];
        let above = vec![100; 8];
        let mut residuals = Vec::new();
        let mut decoded = Vec::new();

        encode_line(&line, &above, &mut residuals);

        // All residuals should be zero for constant image
        for &r in &residuals {
            assert_eq!(r, 0);
        }

        decode_line(&residuals, &above, &mut decoded);
        assert_eq!(line, decoded);
    }

    #[test]
    #[ignore]
    fn test_encode_decode_empty() {
        let line: Vec<i32> = vec![];
        let above: Vec<i32> = vec![];
        let mut residuals = Vec::new();
        let mut decoded = Vec::new();

        encode_line(&line, &above, &mut residuals);
        assert!(residuals.is_empty());

        decode_line(&residuals, &above, &mut decoded);
        assert!(decoded.is_empty());
    }

    #[test]
    #[ignore]
    fn test_encode_decode_gradient() {
        // Gradient pattern
        let line: Vec<i32> = (0..16).collect();
        let above: Vec<i32> = (0..16).map(|x| x + 1).collect();
        let mut residuals = Vec::new();
        let mut decoded = Vec::new();

        encode_line(&line, &above, &mut residuals);
        decode_line(&residuals, &above, &mut decoded);

        assert_eq!(line, decoded);
    }

    #[test]
    #[ignore]
    fn test_quantize_gradient() {
        assert_eq!(quantize_gradient(0), 0);
        assert_eq!(quantize_gradient(-1), -1);
        assert_eq!(quantize_gradient(-5), -2);
        assert_eq!(quantize_gradient(1), 1);
        assert_eq!(quantize_gradient(5), 2);
    }

    #[test]
    #[ignore]
    fn test_compute_context_range() {
        // Context should always be in valid range
        for top in [-100, -1, 0, 1, 100] {
            for left in [-100, -1, 0, 1, 100] {
                let ctx = compute_context(top, left, 0, 0);
                assert!(ctx < super::super::types::CONTEXT_COUNT);
            }
        }
    }
}
