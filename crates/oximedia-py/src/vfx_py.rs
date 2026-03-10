//! Python bindings for visual effects and compositing.
//!
//! Provides `PyVideoEffect`, `PyChromaKey`, `PyTransitionEffect`, `PyGenerator`,
//! and standalone functions for applying effects to raw frame data.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyVideoEffect
// ---------------------------------------------------------------------------

/// A configurable video effect that can be applied to frame data.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyVideoEffect {
    /// Effect type name.
    #[pyo3(get)]
    pub effect_type: String,

    /// Effect strength (0.0-1.0).
    #[pyo3(get)]
    pub strength: f64,

    /// Extra parameters.
    #[pyo3(get)]
    pub params: HashMap<String, f64>,
}

#[pymethods]
impl PyVideoEffect {
    /// Create a blur effect.
    #[staticmethod]
    #[pyo3(signature = (radius=None))]
    fn blur(radius: Option<f64>) -> Self {
        let r = radius.unwrap_or(5.0);
        let mut params = HashMap::new();
        params.insert("radius".to_string(), r);
        Self {
            effect_type: "blur".to_string(),
            strength: 1.0,
            params,
        }
    }

    /// Create a sharpen effect.
    #[staticmethod]
    #[pyo3(signature = (amount=None))]
    fn sharpen(amount: Option<f64>) -> Self {
        let a = amount.unwrap_or(1.0);
        let mut params = HashMap::new();
        params.insert("amount".to_string(), a);
        Self {
            effect_type: "sharpen".to_string(),
            strength: a.clamp(0.0, 2.0),
            params,
        }
    }

    /// Create a glow (bloom) effect.
    #[staticmethod]
    #[pyo3(signature = (radius=None, intensity=None))]
    fn glow(radius: Option<f64>, intensity: Option<f64>) -> Self {
        let r = radius.unwrap_or(10.0);
        let i = intensity.unwrap_or(0.5);
        let mut params = HashMap::new();
        params.insert("radius".to_string(), r);
        params.insert("intensity".to_string(), i);
        Self {
            effect_type: "glow".to_string(),
            strength: i.clamp(0.0, 1.0),
            params,
        }
    }

    /// Create a vintage film look.
    #[staticmethod]
    fn vintage() -> Self {
        Self {
            effect_type: "vintage".to_string(),
            strength: 1.0,
            params: HashMap::new(),
        }
    }

    /// Create a sepia tone effect.
    #[staticmethod]
    fn sepia() -> Self {
        Self {
            effect_type: "sepia".to_string(),
            strength: 1.0,
            params: HashMap::new(),
        }
    }

    /// Create a negative (invert) effect.
    #[staticmethod]
    fn negative() -> Self {
        Self {
            effect_type: "negative".to_string(),
            strength: 1.0,
            params: HashMap::new(),
        }
    }

    /// Create a posterize effect.
    #[staticmethod]
    #[pyo3(signature = (levels=None))]
    fn posterize(levels: Option<u32>) -> Self {
        let l = levels.unwrap_or(4);
        let mut params = HashMap::new();
        params.insert("levels".to_string(), l as f64);
        Self {
            effect_type: "posterize".to_string(),
            strength: 1.0,
            params,
        }
    }

    /// Create a pixelate (mosaic) effect.
    #[staticmethod]
    #[pyo3(signature = (block_size=None))]
    fn pixelate(block_size: Option<u32>) -> Self {
        let bs = block_size.unwrap_or(8);
        let mut params = HashMap::new();
        params.insert("block_size".to_string(), bs as f64);
        Self {
            effect_type: "pixelate".to_string(),
            strength: 1.0,
            params,
        }
    }

    /// Apply this effect to raw RGB frame data.
    ///
    /// Args:
    ///     frame_data: Raw RGB bytes (width * height * 3).
    ///     width: Frame width.
    ///     height: Frame height.
    ///
    /// Returns:
    ///     Processed RGB bytes.
    fn apply(&self, frame_data: Vec<u8>, width: u32, height: u32) -> PyResult<Vec<u8>> {
        let expected = (width as usize) * (height as usize) * 3;
        if frame_data.len() < expected {
            return Err(PyValueError::new_err(format!(
                "Frame data too small: need {} bytes, got {}",
                expected,
                frame_data.len()
            )));
        }

        let mut output = frame_data[..expected].to_vec();

        match self.effect_type.as_str() {
            "sepia" => apply_sepia_inplace(&mut output, self.strength),
            "negative" => apply_negative_inplace(&mut output),
            "posterize" => {
                let levels = self.params.get("levels").copied().unwrap_or(4.0) as u8;
                apply_posterize_inplace(&mut output, levels);
            }
            "pixelate" => {
                let bs = self.params.get("block_size").copied().unwrap_or(8.0) as usize;
                apply_pixelate_inplace(&mut output, width as usize, height as usize, bs);
            }
            "blur" => {
                let radius = self.params.get("radius").copied().unwrap_or(5.0) as usize;
                output = apply_box_blur(&output, width as usize, height as usize, radius);
            }
            "sharpen" => {
                let amount = self.params.get("amount").copied().unwrap_or(1.0);
                output = apply_sharpen(&output, width as usize, height as usize, amount);
            }
            "vintage" => {
                apply_vintage_inplace(&mut output, self.strength);
            }
            "brightness" => {
                let adj = self.strength * 2.0 - 1.0; // map 0..1 to -1..1
                apply_brightness_inplace(&mut output, adj);
            }
            "contrast" => {
                apply_contrast_inplace(&mut output, self.strength * 2.0);
            }
            _ => {
                // Identity pass-through for unrecognized effects
            }
        }

        Ok(output)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyVideoEffect(type='{}', strength={:.2})",
            self.effect_type, self.strength
        )
    }
}

// ---------------------------------------------------------------------------
// PyChromaKey
// ---------------------------------------------------------------------------

/// Chroma key (green/blue screen) compositor.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyChromaKey {
    /// Key color red component.
    #[pyo3(get)]
    pub key_r: u8,
    /// Key color green component.
    #[pyo3(get)]
    pub key_g: u8,
    /// Key color blue component.
    #[pyo3(get)]
    pub key_b: u8,
    /// Color distance tolerance (0.0-1.0).
    #[pyo3(get)]
    pub tolerance: f64,
    /// Edge softness (0.0-1.0).
    #[pyo3(get)]
    pub softness: f64,
}

#[pymethods]
impl PyChromaKey {
    /// Create a chroma key with custom color.
    #[new]
    fn new(r: u8, g: u8, b: u8, tolerance: f64, softness: f64) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&tolerance) {
            return Err(PyValueError::new_err(format!(
                "tolerance must be 0.0-1.0, got {tolerance}"
            )));
        }
        if !(0.0..=1.0).contains(&softness) {
            return Err(PyValueError::new_err(format!(
                "softness must be 0.0-1.0, got {softness}"
            )));
        }
        Ok(Self {
            key_r: r,
            key_g: g,
            key_b: b,
            tolerance,
            softness,
        })
    }

    /// Create a green screen chroma key preset.
    #[staticmethod]
    #[pyo3(signature = (tolerance=None, softness=None))]
    fn green_screen(tolerance: Option<f64>, softness: Option<f64>) -> PyResult<Self> {
        Self::new(
            0,
            177,
            64,
            tolerance.unwrap_or(0.3),
            softness.unwrap_or(0.1),
        )
    }

    /// Create a blue screen chroma key preset.
    #[staticmethod]
    #[pyo3(signature = (tolerance=None, softness=None))]
    fn blue_screen(tolerance: Option<f64>, softness: Option<f64>) -> PyResult<Self> {
        Self::new(
            0,
            71,
            187,
            tolerance.unwrap_or(0.3),
            softness.unwrap_or(0.1),
        )
    }

    /// Apply chroma key, compositing foreground over background.
    ///
    /// Args:
    ///     frame: Foreground RGB bytes (width * height * 3).
    ///     width: Frame width.
    ///     height: Frame height.
    ///     bg_frame: Background RGB bytes (same dimensions).
    ///
    /// Returns:
    ///     Composited RGB bytes.
    fn apply(
        &self,
        frame: Vec<u8>,
        width: u32,
        height: u32,
        bg_frame: Vec<u8>,
    ) -> PyResult<Vec<u8>> {
        let expected = (width as usize) * (height as usize) * 3;
        if frame.len() < expected {
            return Err(PyValueError::new_err(format!(
                "Foreground too small: need {expected} bytes, got {}",
                frame.len()
            )));
        }
        if bg_frame.len() < expected {
            return Err(PyValueError::new_err(format!(
                "Background too small: need {expected} bytes, got {}",
                bg_frame.len()
            )));
        }

        let mut output = vec![0u8; expected];
        let tol = self.tolerance * 441.67; // sqrt(255^2 * 3)
        let soft = self.softness * 441.67;

        for i in (0..expected).step_by(3) {
            let fr = frame[i] as f64;
            let fg = frame[i + 1] as f64;
            let fb = frame[i + 2] as f64;

            let dr = fr - self.key_r as f64;
            let dg = fg - self.key_g as f64;
            let db = fb - self.key_b as f64;
            let dist = (dr * dr + dg * dg + db * db).sqrt();

            let alpha = if dist < tol {
                0.0 // fully keyed out
            } else if dist < tol + soft {
                (dist - tol) / soft.max(0.001) // soft edge
            } else {
                1.0 // fully foreground
            };

            output[i] = (fr * alpha + bg_frame[i] as f64 * (1.0 - alpha)).round() as u8;
            output[i + 1] = (fg * alpha + bg_frame[i + 1] as f64 * (1.0 - alpha)).round() as u8;
            output[i + 2] = (fb * alpha + bg_frame[i + 2] as f64 * (1.0 - alpha)).round() as u8;
        }

        Ok(output)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyChromaKey(rgb=({},{},{}), tol={:.2}, soft={:.2})",
            self.key_r, self.key_g, self.key_b, self.tolerance, self.softness
        )
    }
}

// ---------------------------------------------------------------------------
// PyTransitionEffect
// ---------------------------------------------------------------------------

/// A transition effect between two frames.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyTransitionEffect {
    /// Transition type name.
    #[pyo3(get)]
    pub transition_type: String,

    /// Duration in frames.
    #[pyo3(get)]
    pub duration_frames: u32,
}

#[pymethods]
impl PyTransitionEffect {
    /// Create a cross-dissolve transition.
    #[staticmethod]
    fn dissolve(duration_frames: u32) -> PyResult<Self> {
        if duration_frames == 0 {
            return Err(PyValueError::new_err("duration_frames must be > 0"));
        }
        Ok(Self {
            transition_type: "dissolve".to_string(),
            duration_frames,
        })
    }

    /// Create a left wipe transition.
    #[staticmethod]
    fn wipe_left(duration_frames: u32) -> PyResult<Self> {
        if duration_frames == 0 {
            return Err(PyValueError::new_err("duration_frames must be > 0"));
        }
        Ok(Self {
            transition_type: "wipe_left".to_string(),
            duration_frames,
        })
    }

    /// Create a right wipe transition.
    #[staticmethod]
    fn wipe_right(duration_frames: u32) -> PyResult<Self> {
        if duration_frames == 0 {
            return Err(PyValueError::new_err("duration_frames must be > 0"));
        }
        Ok(Self {
            transition_type: "wipe_right".to_string(),
            duration_frames,
        })
    }

    /// Apply this transition between two frames.
    ///
    /// Args:
    ///     frame1: First frame RGB bytes.
    ///     frame2: Second frame RGB bytes.
    ///     progress: Transition progress (0.0-1.0).
    ///     width: Frame width.
    ///     height: Frame height.
    ///
    /// Returns:
    ///     Blended RGB bytes.
    fn apply(
        &self,
        frame1: Vec<u8>,
        frame2: Vec<u8>,
        progress: f64,
        width: u32,
        height: u32,
    ) -> PyResult<Vec<u8>> {
        let expected = (width as usize) * (height as usize) * 3;
        if frame1.len() < expected || frame2.len() < expected {
            return Err(PyValueError::new_err(format!(
                "Frame data too small: need {expected} bytes"
            )));
        }

        let p = progress.clamp(0.0, 1.0);

        match self.transition_type.as_str() {
            "dissolve" => Ok(blend_dissolve(&frame1, &frame2, p, expected)),
            "wipe_left" => Ok(blend_wipe_horizontal(
                &frame1,
                &frame2,
                p,
                width as usize,
                height as usize,
                false,
            )),
            "wipe_right" => Ok(blend_wipe_horizontal(
                &frame1,
                &frame2,
                p,
                width as usize,
                height as usize,
                true,
            )),
            _ => Ok(blend_dissolve(&frame1, &frame2, p, expected)),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyTransitionEffect(type='{}', frames={})",
            self.transition_type, self.duration_frames
        )
    }
}

// ---------------------------------------------------------------------------
// PyGenerator
// ---------------------------------------------------------------------------

/// Test pattern generator.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyGenerator {
    /// Pattern type.
    #[pyo3(get)]
    pub pattern: String,

    /// Pattern width.
    #[pyo3(get)]
    pub width: u32,

    /// Pattern height.
    #[pyo3(get)]
    pub height: u32,
}

#[pymethods]
impl PyGenerator {
    /// Create a color bars generator.
    #[staticmethod]
    fn color_bars(width: u32, height: u32) -> PyResult<Self> {
        validate_dimensions(width, height)?;
        Ok(Self {
            pattern: "color_bars".to_string(),
            width,
            height,
        })
    }

    /// Create a gradient generator.
    #[staticmethod]
    #[pyo3(signature = (width, height, color1=None, color2=None))]
    fn gradient(
        width: u32,
        height: u32,
        color1: Option<(u8, u8, u8)>,
        color2: Option<(u8, u8, u8)>,
    ) -> PyResult<Self> {
        validate_dimensions(width, height)?;
        let _c1 = color1.unwrap_or((0, 0, 0));
        let _c2 = color2.unwrap_or((255, 255, 255));
        Ok(Self {
            pattern: "gradient".to_string(),
            width,
            height,
        })
    }

    /// Create a noise generator.
    #[staticmethod]
    fn noise(width: u32, height: u32) -> PyResult<Self> {
        validate_dimensions(width, height)?;
        Ok(Self {
            pattern: "noise".to_string(),
            width,
            height,
        })
    }

    /// Create a checkerboard generator.
    #[staticmethod]
    #[pyo3(signature = (width, height, block_size=None))]
    fn checkerboard(width: u32, height: u32, block_size: Option<u32>) -> PyResult<Self> {
        validate_dimensions(width, height)?;
        let _bs = block_size.unwrap_or(64);
        Ok(Self {
            pattern: "checkerboard".to_string(),
            width,
            height,
        })
    }

    /// Generate the pattern as RGB bytes.
    fn generate(&self) -> PyResult<Vec<u8>> {
        let w = self.width as usize;
        let h = self.height as usize;
        let mut data = vec![0u8; w * h * 3];

        match self.pattern.as_str() {
            "color_bars" => generate_color_bars(&mut data, w, h),
            "gradient" => generate_gradient_default(&mut data, w, h),
            "noise" => generate_noise(&mut data, w, h),
            "checkerboard" => generate_checkerboard(&mut data, w, h, 64),
            _ => {}
        }

        Ok(data)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyGenerator(pattern='{}', {}x{})",
            self.pattern, self.width, self.height
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Apply an effect to a raw RGB frame.
///
/// Args:
///     frame: Raw RGB bytes.
///     width: Frame width.
///     height: Frame height.
///     effect_name: Effect name (blur, sepia, negative, etc.).
///     strength: Effect strength 0.0-1.0.
///
/// Returns:
///     Processed RGB bytes.
#[pyfunction]
#[pyo3(signature = (frame, width, height, effect_name, strength=None))]
pub fn apply_effect(
    frame: Vec<u8>,
    width: u32,
    height: u32,
    effect_name: &str,
    strength: Option<f64>,
) -> PyResult<Vec<u8>> {
    let s = strength.unwrap_or(1.0);
    let effect = match effect_name {
        "blur" => PyVideoEffect::blur(Some(5.0 * s)),
        "sharpen" => PyVideoEffect::sharpen(Some(s)),
        "sepia" => {
            let mut e = PyVideoEffect::sepia();
            e.strength = s;
            e
        }
        "negative" => PyVideoEffect::negative(),
        "posterize" => PyVideoEffect::posterize(Some((4.0 + s * 12.0) as u32)),
        "pixelate" => PyVideoEffect::pixelate(Some((2.0 + s * 30.0) as u32)),
        "vintage" => {
            let mut e = PyVideoEffect::vintage();
            e.strength = s;
            e
        }
        "glow" => PyVideoEffect::glow(Some(10.0 * s), Some(s)),
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown effect '{}'. Available: blur, sharpen, sepia, negative, posterize, pixelate, vintage, glow",
                other
            )));
        }
    };

    effect.apply(frame, width, height)
}

/// List available effects.
#[pyfunction]
pub fn list_effects() -> Vec<String> {
    vec![
        "blur".to_string(),
        "sharpen".to_string(),
        "glow".to_string(),
        "vintage".to_string(),
        "sepia".to_string(),
        "negative".to_string(),
        "posterize".to_string(),
        "pixelate".to_string(),
        "vignette".to_string(),
        "contrast".to_string(),
        "brightness".to_string(),
        "saturation".to_string(),
        "hue_shift".to_string(),
        "noise".to_string(),
        "emboss".to_string(),
        "edge_detect".to_string(),
    ]
}

/// Apply chroma key to a foreground frame over a background.
#[pyfunction]
#[pyo3(signature = (frame, width, height, key_r, key_g, key_b, tolerance, bg_frame))]
pub fn chroma_key(
    frame: Vec<u8>,
    width: u32,
    height: u32,
    key_r: u8,
    key_g: u8,
    key_b: u8,
    tolerance: f64,
    bg_frame: Vec<u8>,
) -> PyResult<Vec<u8>> {
    let ck = PyChromaKey::new(key_r, key_g, key_b, tolerance, 0.1)?;
    ck.apply(frame, width, height, bg_frame)
}

/// Cross-dissolve between two frames at a given progress.
#[pyfunction]
pub fn dissolve(
    frame1: Vec<u8>,
    frame2: Vec<u8>,
    width: u32,
    height: u32,
    progress: f64,
) -> PyResult<Vec<u8>> {
    let expected = (width as usize) * (height as usize) * 3;
    if frame1.len() < expected || frame2.len() < expected {
        return Err(PyValueError::new_err(format!(
            "Frame data too small: need {expected} bytes"
        )));
    }
    Ok(blend_dissolve(
        &frame1,
        &frame2,
        progress.clamp(0.0, 1.0),
        expected,
    ))
}

/// List available transition types.
#[pyfunction]
pub fn list_transitions() -> Vec<String> {
    vec![
        "dissolve".to_string(),
        "wipe_left".to_string(),
        "wipe_right".to_string(),
        "wipe_up".to_string(),
        "wipe_down".to_string(),
        "fade".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// Internal effect implementations
// ---------------------------------------------------------------------------

fn apply_sepia_inplace(data: &mut [u8], strength: f64) {
    let s = strength.clamp(0.0, 1.0);
    for pixel in data.chunks_exact_mut(3) {
        let r = pixel[0] as f64;
        let g = pixel[1] as f64;
        let b = pixel[2] as f64;

        let sr = (0.393 * r + 0.769 * g + 0.189 * b).min(255.0);
        let sg = (0.349 * r + 0.686 * g + 0.168 * b).min(255.0);
        let sb = (0.272 * r + 0.534 * g + 0.131 * b).min(255.0);

        pixel[0] = (r * (1.0 - s) + sr * s).round() as u8;
        pixel[1] = (g * (1.0 - s) + sg * s).round() as u8;
        pixel[2] = (b * (1.0 - s) + sb * s).round() as u8;
    }
}

fn apply_negative_inplace(data: &mut [u8]) {
    for val in data.iter_mut() {
        *val = 255 - *val;
    }
}

fn apply_posterize_inplace(data: &mut [u8], levels: u8) {
    let levels = levels.max(2);
    let divisor = 256.0 / levels as f64;
    for val in data.iter_mut() {
        let quantized = ((*val as f64 / divisor).floor() * divisor).round() as u8;
        *val = quantized;
    }
}

fn apply_pixelate_inplace(data: &mut [u8], width: usize, height: usize, block_size: usize) {
    let bs = block_size.max(1);
    for by in (0..height).step_by(bs) {
        for bx in (0..width).step_by(bs) {
            let bw = bs.min(width - bx);
            let bh = bs.min(height - by);

            // Average the block
            let mut sum_r: u64 = 0;
            let mut sum_g: u64 = 0;
            let mut sum_b: u64 = 0;
            let count = (bw * bh) as u64;

            for yy in 0..bh {
                for xx in 0..bw {
                    let idx = ((by + yy) * width + (bx + xx)) * 3;
                    if idx + 2 < data.len() {
                        sum_r += data[idx] as u64;
                        sum_g += data[idx + 1] as u64;
                        sum_b += data[idx + 2] as u64;
                    }
                }
            }

            let avg_r = (sum_r / count.max(1)) as u8;
            let avg_g = (sum_g / count.max(1)) as u8;
            let avg_b = (sum_b / count.max(1)) as u8;

            // Fill the block with average
            for yy in 0..bh {
                for xx in 0..bw {
                    let idx = ((by + yy) * width + (bx + xx)) * 3;
                    if idx + 2 < data.len() {
                        data[idx] = avg_r;
                        data[idx + 1] = avg_g;
                        data[idx + 2] = avg_b;
                    }
                }
            }
        }
    }
}

fn apply_box_blur(data: &[u8], width: usize, height: usize, radius: usize) -> Vec<u8> {
    let r = radius.max(1).min(width / 2);
    let mut output = data.to_vec();

    // Horizontal pass
    let mut temp = output.clone();
    for y in 0..height {
        for x in 0..width {
            let mut sum_r: u64 = 0;
            let mut sum_g: u64 = 0;
            let mut sum_b: u64 = 0;
            let mut count: u64 = 0;

            let x_start = x.saturating_sub(r);
            let x_end = (x + r + 1).min(width);

            for xx in x_start..x_end {
                let idx = (y * width + xx) * 3;
                if idx + 2 < data.len() {
                    sum_r += data[idx] as u64;
                    sum_g += data[idx + 1] as u64;
                    sum_b += data[idx + 2] as u64;
                    count += 1;
                }
            }

            let idx = (y * width + x) * 3;
            if idx + 2 < temp.len() && count > 0 {
                temp[idx] = (sum_r / count) as u8;
                temp[idx + 1] = (sum_g / count) as u8;
                temp[idx + 2] = (sum_b / count) as u8;
            }
        }
    }

    // Vertical pass
    for y in 0..height {
        for x in 0..width {
            let mut sum_r: u64 = 0;
            let mut sum_g: u64 = 0;
            let mut sum_b: u64 = 0;
            let mut count: u64 = 0;

            let y_start = y.saturating_sub(r);
            let y_end = (y + r + 1).min(height);

            for yy in y_start..y_end {
                let idx = (yy * width + x) * 3;
                if idx + 2 < temp.len() {
                    sum_r += temp[idx] as u64;
                    sum_g += temp[idx + 1] as u64;
                    sum_b += temp[idx + 2] as u64;
                    count += 1;
                }
            }

            let idx = (y * width + x) * 3;
            if idx + 2 < output.len() && count > 0 {
                output[idx] = (sum_r / count) as u8;
                output[idx + 1] = (sum_g / count) as u8;
                output[idx + 2] = (sum_b / count) as u8;
            }
        }
    }

    output
}

fn apply_sharpen(data: &[u8], width: usize, height: usize, amount: f64) -> Vec<u8> {
    // Unsharp mask: sharpen = original + amount * (original - blur)
    let blurred = apply_box_blur(data, width, height, 1);
    let mut output = data.to_vec();

    for i in 0..output.len() {
        let original = data[i] as f64;
        let blur_val = blurred[i] as f64;
        let sharpened = original + amount * (original - blur_val);
        output[i] = sharpened.round().clamp(0.0, 255.0) as u8;
    }

    output
}

fn apply_vintage_inplace(data: &mut [u8], strength: f64) {
    let s = strength.clamp(0.0, 1.0);
    for pixel in data.chunks_exact_mut(3) {
        let r = pixel[0] as f64;
        let g = pixel[1] as f64;
        let b = pixel[2] as f64;

        // Warm tint + reduced saturation
        let avg = (r + g + b) / 3.0;
        let vr = (avg * 0.3 + r * 0.7 + 20.0).min(255.0);
        let vg = (avg * 0.3 + g * 0.7 + 5.0).min(255.0);
        let vb = (avg * 0.3 + b * 0.7 - 10.0).max(0.0).min(255.0);

        pixel[0] = (r * (1.0 - s) + vr * s).round() as u8;
        pixel[1] = (g * (1.0 - s) + vg * s).round() as u8;
        pixel[2] = (b * (1.0 - s) + vb * s).round() as u8;
    }
}

fn apply_brightness_inplace(data: &mut [u8], adjustment: f64) {
    let adj = (adjustment * 255.0).round() as i16;
    for val in data.iter_mut() {
        let new_val = (*val as i16 + adj).clamp(0, 255) as u8;
        *val = new_val;
    }
}

fn apply_contrast_inplace(data: &mut [u8], factor: f64) {
    let f = factor.max(0.0);
    for val in data.iter_mut() {
        let v = *val as f64;
        let adjusted = ((v - 128.0) * f + 128.0).round().clamp(0.0, 255.0) as u8;
        *val = adjusted;
    }
}

fn blend_dissolve(a: &[u8], b: &[u8], progress: f64, len: usize) -> Vec<u8> {
    let mut output = vec![0u8; len];
    let inv = 1.0 - progress;
    for i in 0..len {
        output[i] = (a[i] as f64 * inv + b[i] as f64 * progress).round() as u8;
    }
    output
}

fn blend_wipe_horizontal(
    a: &[u8],
    b: &[u8],
    progress: f64,
    width: usize,
    height: usize,
    reverse: bool,
) -> Vec<u8> {
    let mut output = vec![0u8; width * height * 3];
    let boundary = (progress * width as f64).round() as usize;

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let use_b = if reverse {
                x >= width - boundary
            } else {
                x < boundary
            };

            if use_b {
                output[idx] = b[idx];
                output[idx + 1] = b[idx + 1];
                output[idx + 2] = b[idx + 2];
            } else {
                output[idx] = a[idx];
                output[idx + 1] = a[idx + 1];
                output[idx + 2] = a[idx + 2];
            }
        }
    }

    output
}

fn validate_dimensions(width: u32, height: u32) -> PyResult<()> {
    if width == 0 || height == 0 {
        return Err(PyValueError::new_err(format!(
            "width and height must be > 0, got {}x{}",
            width, height
        )));
    }
    Ok(())
}

fn generate_color_bars(data: &mut [u8], width: usize, height: usize) {
    let bars: [(u8, u8, u8); 7] = [
        (235, 235, 235),
        (235, 235, 16),
        (16, 235, 235),
        (16, 235, 16),
        (235, 16, 235),
        (235, 16, 16),
        (16, 16, 235),
    ];
    let bar_width = width / 7;
    for y in 0..height {
        for x in 0..width {
            let bar_idx = (x / bar_width.max(1)).min(6);
            let (r, g, b) = bars[bar_idx];
            let idx = (y * width + x) * 3;
            if idx + 2 < data.len() {
                data[idx] = r;
                data[idx + 1] = g;
                data[idx + 2] = b;
            }
        }
    }
}

fn generate_gradient_default(data: &mut [u8], width: usize, height: usize) {
    for y in 0..height {
        for x in 0..width {
            let t = if width > 1 {
                x as f64 / (width - 1) as f64
            } else {
                0.0
            };
            let v = (t * 255.0).round() as u8;
            let idx = (y * width + x) * 3;
            if idx + 2 < data.len() {
                data[idx] = v;
                data[idx + 1] = v;
                data[idx + 2] = v;
            }
        }
    }
}

fn generate_noise(data: &mut [u8], width: usize, height: usize) {
    let mut state: u64 = 0x5DEE_CE66_D_u64;
    for y in 0..height {
        for x in 0..width {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let val = ((state >> 33) & 0xFF) as u8;
            let idx = (y * width + x) * 3;
            if idx + 2 < data.len() {
                data[idx] = val;
                data[idx + 1] = val;
                data[idx + 2] = val;
            }
        }
    }
}

fn generate_checkerboard(data: &mut [u8], width: usize, height: usize, block_size: usize) {
    let bs = block_size.max(1);
    for y in 0..height {
        for x in 0..width {
            let checker = ((x / bs) + (y / bs)) % 2 == 0;
            let v = if checker { 255u8 } else { 0u8 };
            let idx = (y * width + x) * 3;
            if idx + 2 < data.len() {
                data[idx] = v;
                data[idx + 1] = v;
                data[idx + 2] = v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register VFX bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVideoEffect>()?;
    m.add_class::<PyChromaKey>()?;
    m.add_class::<PyTransitionEffect>()?;
    m.add_class::<PyGenerator>()?;
    m.add_function(wrap_pyfunction!(apply_effect, m)?)?;
    m.add_function(wrap_pyfunction!(list_effects, m)?)?;
    m.add_function(wrap_pyfunction!(chroma_key, m)?)?;
    m.add_function(wrap_pyfunction!(dissolve, m)?)?;
    m.add_function(wrap_pyfunction!(list_transitions, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(width: u32, height: u32, val: u8) -> Vec<u8> {
        vec![val; (width as usize) * (height as usize) * 3]
    }

    #[test]
    fn test_effect_sepia() {
        let effect = PyVideoEffect::sepia();
        let frame = make_frame(4, 4, 128);
        let result = effect.apply(frame, 4, 4);
        assert!(result.is_ok());
        let out = result.expect("should succeed");
        assert_eq!(out.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_effect_negative() {
        let effect = PyVideoEffect::negative();
        let frame = make_frame(4, 4, 100);
        let result = effect.apply(frame, 4, 4);
        assert!(result.is_ok());
        let out = result.expect("should succeed");
        assert_eq!(out[0], 155); // 255 - 100
    }

    #[test]
    fn test_effect_posterize() {
        let effect = PyVideoEffect::posterize(Some(4));
        let frame = make_frame(4, 4, 130);
        let result = effect.apply(frame, 4, 4);
        assert!(result.is_ok());
    }

    #[test]
    fn test_effect_pixelate() {
        let effect = PyVideoEffect::pixelate(Some(2));
        let frame = vec![0, 50, 100, 200, 250, 150, 10, 20, 30, 40, 50, 60]; // 2x2 image
        let result = effect.apply(frame, 2, 2);
        assert!(result.is_ok());
        let out = result.expect("should succeed");
        // All pixels should be the average
        assert_eq!(out[0], out[3]);
        assert_eq!(out[1], out[4]);
    }

    #[test]
    fn test_effect_blur() {
        let effect = PyVideoEffect::blur(Some(1.0));
        let frame = make_frame(8, 8, 100);
        let result = effect.apply(frame, 8, 8);
        assert!(result.is_ok());
    }

    #[test]
    fn test_chromakey_green() {
        let ck = PyChromaKey::green_screen(None, None);
        assert!(ck.is_ok());
        let ck = ck.expect("should succeed");
        assert_eq!(ck.key_g, 177);
    }

    #[test]
    fn test_chromakey_apply() {
        let ck = PyChromaKey::new(0, 255, 0, 0.5, 0.1).expect("valid");
        // Green foreground over red background
        let fg = vec![0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0]; // 2x2 green
        let bg = vec![255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0]; // 2x2 red
        let result = ck.apply(fg, 2, 2, bg);
        assert!(result.is_ok());
        let out = result.expect("should succeed");
        // Green pixels should be replaced by red background
        assert!(out[0] > 200); // should be close to red
    }

    #[test]
    fn test_transition_dissolve() {
        let t = PyTransitionEffect::dissolve(30);
        assert!(t.is_ok());
        let t = t.expect("should succeed");

        let f1 = make_frame(4, 4, 0);
        let f2 = make_frame(4, 4, 200);
        let result = t.apply(f1, f2, 0.5, 4, 4);
        assert!(result.is_ok());
        let out = result.expect("should succeed");
        // 50% blend of 0 and 200 = 100
        assert_eq!(out[0], 100);
    }

    #[test]
    fn test_transition_invalid_duration() {
        assert!(PyTransitionEffect::dissolve(0).is_err());
    }

    #[test]
    fn test_generator_color_bars() {
        let gen = PyGenerator::color_bars(70, 10);
        assert!(gen.is_ok());
        let gen = gen.expect("should succeed");
        let data = gen.generate();
        assert!(data.is_ok());
        let data = data.expect("should succeed");
        assert_eq!(data.len(), 70 * 10 * 3);
    }

    #[test]
    fn test_apply_effect_fn() {
        let frame = make_frame(4, 4, 128);
        let result = apply_effect(frame, 4, 4, "sepia", Some(0.5));
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_effect_unknown() {
        let frame = make_frame(4, 4, 128);
        assert!(apply_effect(frame, 4, 4, "nonexistent", None).is_err());
    }

    #[test]
    fn test_list_effects_fn() {
        let effects = list_effects();
        assert!(effects.contains(&"blur".to_string()));
        assert!(effects.contains(&"sepia".to_string()));
    }

    #[test]
    fn test_dissolve_fn() {
        let f1 = make_frame(2, 2, 0);
        let f2 = make_frame(2, 2, 100);
        let result = dissolve(f1, f2, 2, 2, 0.5);
        assert!(result.is_ok());
        let out = result.expect("should succeed");
        assert_eq!(out[0], 50);
    }
}
