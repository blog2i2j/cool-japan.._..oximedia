//! Color space conversion: cvtColor.

use super::image_io::{extract_img, make_image_output};
use pyo3::prelude::*;

/// Convert an image from one color space to another.
///
/// Mirrors `cv2.cvtColor(src, code)`.
#[pyfunction]
#[pyo3(name = "cvtColor")]
pub fn cvt_color(py: Python<'_>, src: Py<PyAny>, code: i32) -> PyResult<Py<PyAny>> {
    let (data, h, w, ch) = extract_img(py, &src)?;

    let (out_data, out_ch) = match code {
        6 | 7 => {
            // COLOR_BGR2GRAY or COLOR_RGB2GRAY
            let gray = convert_to_gray(&data, w, h, ch, code == 7)?;
            (gray, 1usize)
        }
        8 | 9 => {
            // COLOR_GRAY2BGR or COLOR_GRAY2BGRA (we produce 3-ch BGR)
            let bgr = gray_to_bgr(&data, w, h);
            (bgr, 3)
        }
        4 => {
            // COLOR_BGR2RGB or COLOR_RGB2BGR (same operation)
            let swapped = swap_rb(&data, w, h);
            (swapped, 3)
        }
        10 => {
            // COLOR_BGRA2GRAY — treat first 3 ch as BGR
            let gray = bgra_to_gray(&data, w, h)?;
            (gray, 1)
        }
        40 | 41 => {
            // COLOR_BGR2HSV or COLOR_RGB2HSV
            let is_rgb = code == 41;
            let hsv = bgr_to_hsv(&data, w, h, is_rgb);
            (hsv, 3)
        }
        54 | 55 => {
            // COLOR_HSV2BGR or COLOR_HSV2RGB
            let is_rgb = code == 55;
            let bgr = hsv_to_bgr(&data, w, h, is_rgb);
            (bgr, 3)
        }
        44 | 45 => {
            // COLOR_BGR2Lab or COLOR_RGB2Lab
            let is_rgb = code == 45;
            let lab = bgr_to_lab(&data, w, h, is_rgb);
            (lab, 3)
        }
        56 | 57 => {
            // COLOR_Lab2BGR or COLOR_Lab2RGB
            let is_rgb = code == 57;
            let bgr = lab_to_bgr(&data, w, h, is_rgb);
            (bgr, 3)
        }
        82 | 83 => {
            // COLOR_BGR2YUV or COLOR_RGB2YUV
            let is_rgb = code == 83;
            let yuv = bgr_to_yuv(&data, w, h, is_rgb);
            (yuv, 3)
        }
        84 | 85 => {
            // COLOR_YUV2BGR or COLOR_YUV2RGB
            let is_rgb = code == 85;
            let bgr = yuv_to_bgr(&data, w, h, is_rgb);
            (bgr, 3)
        }
        52 => {
            // COLOR_BGR2HLS
            let hls = bgr_to_hls(&data, w, h);
            (hls, 3)
        }
        60 | 61 => {
            // COLOR_HLS2BGR or COLOR_HLS2RGB
            let is_rgb = code == 61;
            let bgr = hls_to_bgr(&data, w, h, is_rgb);
            (bgr, 3)
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "cvtColor: unsupported conversion code {}",
                code
            )));
        }
    };

    make_image_output(py, out_data, h, w, out_ch)
}

// ── Color conversion implementations ────────────────────────────────────────

pub(crate) fn convert_to_gray(
    data: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    is_rgb: bool,
) -> PyResult<Vec<u8>> {
    let mut out = vec![0u8; h * w];
    if ch == 1 {
        out.copy_from_slice(data);
        return Ok(out);
    }
    if ch < 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cvtColor: need 3-channel image for gray conversion",
        ));
    }
    for y in 0..h {
        for x in 0..w {
            let off = (y * w + x) * ch;
            // BT.601 weights
            let (r, g, b) = if is_rgb {
                (data[off], data[off + 1], data[off + 2])
            } else {
                (data[off + 2], data[off + 1], data[off])
            };
            let gray = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8;
            out[y * w + x] = gray;
        }
    }
    Ok(out)
}

fn bgra_to_gray(data: &[u8], w: usize, h: usize) -> PyResult<Vec<u8>> {
    let ch = if data.len() == h * w * 4 { 4 } else { 3 };
    convert_to_gray(data, w, h, ch, false)
}

pub(crate) fn gray_to_bgr(data: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for i in 0..h * w {
        let g = data[i];
        out[i * 3] = g;
        out[i * 3 + 1] = g;
        out[i * 3 + 2] = g;
    }
    out
}

pub(crate) fn swap_rb(data: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = data.to_vec();
    for i in 0..h * w {
        out.swap(i * 3, i * 3 + 2);
    }
    out
}

pub(crate) fn bgr_to_hsv(data: &[u8], w: usize, h: usize, is_rgb: bool) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for i in 0..h * w {
        let off = i * 3;
        let (r, g, b) = if is_rgb {
            (
                data[off] as f32 / 255.0,
                data[off + 1] as f32 / 255.0,
                data[off + 2] as f32 / 255.0,
            )
        } else {
            (
                data[off + 2] as f32 / 255.0,
                data[off + 1] as f32 / 255.0,
                data[off] as f32 / 255.0,
            )
        };
        let (h_val, s, v) = rgb_to_hsv_f(r, g, b);
        out[off] = (h_val * 180.0) as u8; // OpenCV uses 0-180 for H
        out[off + 1] = (s * 255.0) as u8;
        out[off + 2] = (v * 255.0) as u8;
    }
    out
}

pub(crate) fn hsv_to_bgr(data: &[u8], w: usize, h: usize, is_rgb: bool) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for i in 0..h * w {
        let off = i * 3;
        let h_val = data[off] as f32 / 180.0; // OpenCV uses 0-180
        let s = data[off + 1] as f32 / 255.0;
        let v = data[off + 2] as f32 / 255.0;
        let (r, g, b) = hsv_to_rgb_f(h_val, s, v);
        if is_rgb {
            out[off] = (r * 255.0) as u8;
            out[off + 1] = (g * 255.0) as u8;
            out[off + 2] = (b * 255.0) as u8;
        } else {
            out[off] = (b * 255.0) as u8;
            out[off + 1] = (g * 255.0) as u8;
            out[off + 2] = (r * 255.0) as u8;
        }
    }
    out
}

fn rgb_to_hsv_f(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;
    let s = if max > 1e-6 { delta / max } else { 0.0 };
    let h = if delta < 1e-6 {
        0.0
    } else if (max - r).abs() < 1e-6 {
        ((g - b) / delta).rem_euclid(6.0) / 6.0
    } else if (max - g).abs() < 1e-6 {
        ((b - r) / delta + 2.0) / 6.0
    } else {
        ((r - g) / delta + 4.0) / 6.0
    };
    (h, s, v)
}

fn hsv_to_rgb_f(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    if s < 1e-6 {
        return (v, v, v);
    }
    let h6 = h * 6.0;
    let i = h6 as i32;
    let f = h6 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

pub(crate) fn bgr_to_lab(data: &[u8], w: usize, h: usize, is_rgb: bool) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for i in 0..h * w {
        let off = i * 3;
        let (r, g, b) = if is_rgb {
            (
                data[off] as f32 / 255.0,
                data[off + 1] as f32 / 255.0,
                data[off + 2] as f32 / 255.0,
            )
        } else {
            (
                data[off + 2] as f32 / 255.0,
                data[off + 1] as f32 / 255.0,
                data[off] as f32 / 255.0,
            )
        };
        let (l, a, b_val) = rgb_to_lab_f(r, g, b);
        // OpenCV scales: L*[0,100]->uint8, a*[-128,127]->+128, b*[-128,127]->+128
        out[off] = (l * 255.0 / 100.0).clamp(0.0, 255.0) as u8;
        out[off + 1] = ((a + 128.0).clamp(0.0, 255.0)) as u8;
        out[off + 2] = ((b_val + 128.0).clamp(0.0, 255.0)) as u8;
    }
    out
}

pub(crate) fn lab_to_bgr(data: &[u8], w: usize, h: usize, is_rgb: bool) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for i in 0..h * w {
        let off = i * 3;
        let l = data[off] as f32 * 100.0 / 255.0;
        let a = data[off + 1] as f32 - 128.0;
        let b_val = data[off + 2] as f32 - 128.0;
        let (r, g, b) = lab_to_rgb_f(l, a, b_val);
        if is_rgb {
            out[off] = (r * 255.0).clamp(0.0, 255.0) as u8;
            out[off + 1] = (g * 255.0).clamp(0.0, 255.0) as u8;
            out[off + 2] = (b * 255.0).clamp(0.0, 255.0) as u8;
        } else {
            out[off] = (b * 255.0).clamp(0.0, 255.0) as u8;
            out[off + 1] = (g * 255.0).clamp(0.0, 255.0) as u8;
            out[off + 2] = (r * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    out
}

fn rgb_to_lab_f(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // Linearize (gamma removal)
    let lin = |c: f32| {
        if c > 0.04045 {
            ((c + 0.055) / 1.055).powf(2.4)
        } else {
            c / 12.92
        }
    };
    let rl = lin(r);
    let gl = lin(g);
    let bl = lin(b);

    // RGB -> XYZ (D65)
    let x = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375;
    let y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750;
    let z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041;

    // XYZ -> Lab
    let f = |t: f32| {
        if t > 0.008856 {
            t.powf(1.0 / 3.0)
        } else {
            7.787 * t + 16.0 / 116.0
        }
    };
    let fx = f(x / 0.95047);
    let fy = f(y); // yn = 1.0
    let fz = f(z / 1.08883);
    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b_val = 200.0 * (fy - fz);
    (l, a, b_val)
}

fn lab_to_rgb_f(l: f32, a: f32, b_val: f32) -> (f32, f32, f32) {
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b_val / 200.0;
    let f_inv = |t: f32| {
        if t > 0.2069 {
            t.powi(3)
        } else {
            (t - 16.0 / 116.0) / 7.787
        }
    };
    let x = f_inv(fx) * 0.95047;
    let y = f_inv(fy);
    let z = f_inv(fz) * 1.08883;
    // XYZ -> RGB (D65)
    let r = x * 3.2404542 - y * 1.5371385 - z * 0.4985314;
    let g = -x * 0.9692660 + y * 1.8760108 + z * 0.0415560;
    let b = x * 0.0556434 - y * 0.2040259 + z * 1.0572252;
    let gamma = |c: f32| {
        if c > 0.0031308 {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        } else {
            12.92 * c
        }
    };
    (gamma(r), gamma(g), gamma(b))
}

pub(crate) fn bgr_to_yuv(data: &[u8], w: usize, h: usize, is_rgb: bool) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for i in 0..h * w {
        let off = i * 3;
        let (r, g, b) = if is_rgb {
            (data[off] as f32, data[off + 1] as f32, data[off + 2] as f32)
        } else {
            (data[off + 2] as f32, data[off + 1] as f32, data[off] as f32)
        };
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let u = -0.147 * r - 0.289 * g + 0.436 * b + 128.0;
        let v = 0.615 * r - 0.515 * g - 0.100 * b + 128.0;
        out[off] = y.clamp(0.0, 255.0) as u8;
        out[off + 1] = u.clamp(0.0, 255.0) as u8;
        out[off + 2] = v.clamp(0.0, 255.0) as u8;
    }
    out
}

pub(crate) fn yuv_to_bgr(data: &[u8], w: usize, h: usize, is_rgb: bool) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for i in 0..h * w {
        let off = i * 3;
        let y = data[off] as f32;
        let u = data[off + 1] as f32 - 128.0;
        let v = data[off + 2] as f32 - 128.0;
        let r = y + 1.140 * v;
        let g = y - 0.395 * u - 0.581 * v;
        let b = y + 2.032 * u;
        if is_rgb {
            out[off] = r.clamp(0.0, 255.0) as u8;
            out[off + 1] = g.clamp(0.0, 255.0) as u8;
            out[off + 2] = b.clamp(0.0, 255.0) as u8;
        } else {
            out[off] = b.clamp(0.0, 255.0) as u8;
            out[off + 1] = g.clamp(0.0, 255.0) as u8;
            out[off + 2] = r.clamp(0.0, 255.0) as u8;
        }
    }
    out
}

fn bgr_to_hls(data: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for i in 0..h * w {
        let off = i * 3;
        let b = data[off] as f32 / 255.0;
        let g = data[off + 1] as f32 / 255.0;
        let r = data[off + 2] as f32 / 255.0;
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let l = (max + min) / 2.0;
        let delta = max - min;
        let s = if delta < 1e-6 {
            0.0
        } else if l < 0.5 {
            delta / (max + min)
        } else {
            delta / (2.0 - max - min)
        };
        let h_val = if delta < 1e-6 {
            0.0
        } else if (max - r).abs() < 1e-6 {
            ((g - b) / delta).rem_euclid(6.0) / 6.0
        } else if (max - g).abs() < 1e-6 {
            ((b - r) / delta + 2.0) / 6.0
        } else {
            ((r - g) / delta + 4.0) / 6.0
        };
        out[off] = (h_val * 180.0) as u8;
        out[off + 1] = (l * 255.0) as u8;
        out[off + 2] = (s * 255.0) as u8;
    }
    out
}

fn hls_to_bgr(data: &[u8], w: usize, h: usize, is_rgb: bool) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for i in 0..h * w {
        let off = i * 3;
        let h_val = data[off] as f32 / 180.0;
        let l = data[off + 1] as f32 / 255.0;
        let s = data[off + 2] as f32 / 255.0;
        let (r, g, b) = hls_to_rgb_f(h_val, l, s);
        if is_rgb {
            out[off] = (r * 255.0).clamp(0.0, 255.0) as u8;
            out[off + 1] = (g * 255.0).clamp(0.0, 255.0) as u8;
            out[off + 2] = (b * 255.0).clamp(0.0, 255.0) as u8;
        } else {
            out[off] = (b * 255.0).clamp(0.0, 255.0) as u8;
            out[off + 1] = (g * 255.0).clamp(0.0, 255.0) as u8;
            out[off + 2] = (r * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    out
}

fn hls_to_rgb_f(h: f32, l: f32, s: f32) -> (f32, f32, f32) {
    if s < 1e-6 {
        return (l, l, l);
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let hue_to_rgb = |p: f32, q: f32, mut t: f32| -> f32 {
        if t < 0.0 {
            t += 1.0;
        }
        if t > 1.0 {
            t -= 1.0;
        }
        if t < 1.0 / 6.0 {
            return p + (q - p) * 6.0 * t;
        }
        if t < 1.0 / 2.0 {
            return q;
        }
        if t < 2.0 / 3.0 {
            return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
        }
        p
    };
    (
        hue_to_rgb(p, q, h + 1.0 / 3.0),
        hue_to_rgb(p, q, h),
        hue_to_rgb(p, q, h - 1.0 / 3.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── gray_to_bgr ───────────────────────────────────────────────────────────

    #[test]
    fn test_gray_to_bgr_expands_channels() {
        // 2x2 single-channel input: each pixel replicated into BGR triple
        let gray_data = vec![100u8, 150u8, 200u8, 50u8];
        let bgr = gray_to_bgr(&gray_data, 2, 2);
        assert_eq!(bgr.len(), 4 * 3, "output should have 3 channels per pixel");
        assert_eq!(bgr[0], 100, "B channel of pixel 0");
        assert_eq!(bgr[1], 100, "G channel of pixel 0");
        assert_eq!(bgr[2], 100, "R channel of pixel 0");
        assert_eq!(bgr[3], 150, "B channel of pixel 1");
        assert_eq!(bgr[9], 50, "B channel of pixel 3");
    }

    #[test]
    fn test_gray_to_bgr_black_white() {
        let data = vec![0u8, 255u8];
        let bgr = gray_to_bgr(&data, 1, 2);
        assert_eq!(&bgr[0..3], &[0, 0, 0]);
        assert_eq!(&bgr[3..6], &[255, 255, 255]);
    }

    // ── swap_rb ───────────────────────────────────────────────────────────────

    #[test]
    fn test_swap_rb_swaps_first_and_third_channel() {
        // BGR pixel [255, 0, 0] → swap B and R → [0, 0, 255]
        let bgr = vec![255u8, 0, 0];
        let rgb = swap_rb(&bgr, 1, 1);
        assert_eq!(rgb[0], 0, "B→R: first channel should become 0");
        assert_eq!(rgb[1], 0, "G unchanged");
        assert_eq!(rgb[2], 255, "R→B: third channel should become 255");
    }

    #[test]
    fn test_swap_rb_pure_green_unchanged() {
        // [0, 255, 0] only green → swap has no visible effect on G
        let bgr = vec![0u8, 255, 0];
        let rgb = swap_rb(&bgr, 1, 1);
        assert_eq!(rgb[0], 0, "channel 0 (was B=0, now R position)");
        assert_eq!(rgb[1], 255, "G unchanged");
        assert_eq!(rgb[2], 0, "channel 2 (was R=0, now B position)");
    }

    #[test]
    fn test_swap_rb_multi_pixel() {
        // 2 pixels: [10,20,30], [40,50,60]
        let data = vec![10u8, 20, 30, 40, 50, 60];
        let result = swap_rb(&data, 2, 1);
        assert_eq!(&result[0..3], &[30, 20, 10]);
        assert_eq!(&result[3..6], &[60, 50, 40]);
    }

    // ── convert_to_gray (BT.601) ──────────────────────────────────────────────

    #[test]
    fn test_convert_to_gray_white_pixel() {
        let white_bgr = vec![255u8, 255, 255];
        let gray = convert_to_gray(&white_bgr, 1, 1, 3, false)
            .expect("convert_to_gray should not fail on valid input");
        assert_eq!(gray[0], 255, "white BGR → gray should be 255");
    }

    #[test]
    fn test_convert_to_gray_black_pixel() {
        let black_bgr = vec![0u8, 0, 0];
        let gray = convert_to_gray(&black_bgr, 1, 1, 3, false)
            .expect("convert_to_gray should not fail on valid input");
        assert_eq!(gray[0], 0, "black BGR → gray should be 0");
    }

    #[test]
    fn test_convert_to_gray_passthrough_for_single_channel() {
        // If ch == 1, the data should be copied directly
        let data = vec![42u8, 99u8, 7u8];
        let gray = convert_to_gray(&data, 3, 1, 1, false).expect("passthrough should succeed");
        assert_eq!(gray, data);
    }

    #[test]
    fn test_convert_to_gray_pure_blue_bgr() {
        // Pure blue in BGR: [255, 0, 0] → BT.601: 0.114*255 ≈ 29
        let blue_bgr = vec![255u8, 0, 0];
        let gray = convert_to_gray(&blue_bgr, 1, 1, 3, false).expect("convert should succeed");
        let expected = (0.114f32 * 255.0) as u8;
        assert!(
            (gray[0] as i32 - expected as i32).abs() <= 2,
            "pure blue: expected ~{} got {}",
            expected,
            gray[0]
        );
    }

    // ── HSV roundtrip ─────────────────────────────────────────────────────────

    #[test]
    fn test_hsv_roundtrip_pure_blue() {
        // BGR pure blue = [255, 0, 0] → HSV → BGR should give back [255, 0, 0]
        let bgr = vec![255u8, 0, 0];
        let hsv = bgr_to_hsv(&bgr, 1, 1, false);
        let back = hsv_to_bgr(&hsv, 1, 1, false);
        assert!(
            (back[0] as i32 - 255).abs() <= 2,
            "B channel: expected ~255 got {}",
            back[0]
        );
        assert!(back[1] <= 2, "G channel: expected ~0 got {}", back[1]);
        assert!(back[2] <= 2, "R channel: expected ~0 got {}", back[2]);
    }

    #[test]
    fn test_hsv_roundtrip_pure_green() {
        // BGR pure green = [0, 255, 0]
        let bgr = vec![0u8, 255, 0];
        let hsv = bgr_to_hsv(&bgr, 1, 1, false);
        let back = hsv_to_bgr(&hsv, 1, 1, false);
        assert!(back[0] <= 2, "B channel: expected ~0 got {}", back[0]);
        assert!(
            (back[1] as i32 - 255).abs() <= 2,
            "G channel: expected ~255 got {}",
            back[1]
        );
        assert!(back[2] <= 2, "R channel: expected ~0 got {}", back[2]);
    }

    #[test]
    fn test_hsv_white_has_zero_saturation() {
        // White pixel in BGR: [255, 255, 255] → HSV: S should be 0
        let bgr = vec![255u8, 255, 255];
        let hsv = bgr_to_hsv(&bgr, 1, 1, false);
        assert_eq!(hsv[1], 0, "white pixel should have saturation = 0 in HSV");
        assert_eq!(hsv[2], 255, "white pixel should have value = 255 in HSV");
    }

    // ── YUV roundtrip ─────────────────────────────────────────────────────────

    #[test]
    fn test_yuv_roundtrip_mid_grey() {
        // Mid-grey: all channels equal, YUV should be approximately neutral
        let bgr = vec![128u8, 128, 128];
        let yuv = bgr_to_yuv(&bgr, 1, 1, false);
        let back = yuv_to_bgr(&yuv, 1, 1, false);
        for i in 0..3 {
            assert!(
                (back[i] as i32 - bgr[i] as i32).abs() <= 3,
                "channel {}: expected ~{} got {}",
                i,
                bgr[i],
                back[i]
            );
        }
    }

    #[test]
    fn test_yuv_roundtrip_arbitrary_colour() {
        let bgr = vec![100u8, 150u8, 200u8];
        let yuv = bgr_to_yuv(&bgr, 1, 1, false);
        let back = yuv_to_bgr(&yuv, 1, 1, false);
        for i in 0..3 {
            assert!(
                (back[i] as i32 - bgr[i] as i32).abs() <= 3,
                "channel {}: expected {} got {}",
                i,
                bgr[i],
                back[i]
            );
        }
    }

    // ── Lab roundtrip ─────────────────────────────────────────────────────────

    #[test]
    fn test_lab_roundtrip_arbitrary_colour() {
        let bgr = vec![120u8, 80u8, 200u8];
        let lab = bgr_to_lab(&bgr, 1, 1, false);
        let back = lab_to_bgr(&lab, 1, 1, false);
        for i in 0..3 {
            assert!(
                (back[i] as i32 - bgr[i] as i32).abs() <= 5,
                "channel {}: expected {} got {}",
                i,
                bgr[i],
                back[i]
            );
        }
    }

    #[test]
    fn test_lab_roundtrip_white() {
        let bgr = vec![255u8, 255, 255];
        let lab = bgr_to_lab(&bgr, 1, 1, false);
        let back = lab_to_bgr(&lab, 1, 1, false);
        for i in 0..3 {
            assert!(
                (back[i] as i32 - 255i32).abs() <= 5,
                "channel {}: expected ~255 got {}",
                i,
                back[i]
            );
        }
    }

    #[test]
    fn test_lab_roundtrip_black() {
        let bgr = vec![0u8, 0, 0];
        let lab = bgr_to_lab(&bgr, 1, 1, false);
        let back = lab_to_bgr(&lab, 1, 1, false);
        for i in 0..3 {
            assert!(back[i] <= 5, "channel {}: expected ~0 got {}", i, back[i]);
        }
    }
}
