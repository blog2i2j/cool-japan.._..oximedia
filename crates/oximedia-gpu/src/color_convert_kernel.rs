//! GPU color space conversion kernels (CPU simulation).
//!
//! Provides batch color space conversion operations simulated with Rayon
//! parallelism, matching GPU compute-shader semantics.
//!
//! Supported conversions:
//! - RGB ↔ YUV with BT.601, BT.709, BT.2020 coefficients
//! - Limited range ↔ full range (studio swing ↔ JPEG)
//! - Packed RGBA ↔ planar YUV (4:4:4)
//!
//! # Example
//!
//! ```rust
//! use oximedia_gpu::color_convert_kernel::{ColorConvertKernel, ColorStandard, RangeMode};
//!
//! let input = vec![235u8, 128, 44, 255]; // 1×1 RGBA pixel
//! let mut output = vec![0u8; 4];         // 1×1 YUVA output
//! ColorConvertKernel::rgb_to_yuv(&input, &mut output, 1, 1, ColorStandard::Bt709, RangeMode::Full)
//!     .expect("conversion failed");
//! ```

use rayon::prelude::*;
use thiserror::Error;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors returned by color conversion kernel operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ColorKernelError {
    /// Source or destination buffer has incorrect length.
    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },
    /// Image dimensions are invalid (zero width or height).
    #[error("Invalid dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },
    /// Pixel count overflows usize.
    #[error("Pixel count overflow for {width}x{height}")]
    PixelCountOverflow { width: u32, height: u32 },
}

// ─── ColorStandard ────────────────────────────────────────────────────────────

/// Color standard / primaries used for the YCbCr matrix coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorStandard {
    /// ITU-R BT.601 (SD television, SDTV)
    Bt601,
    /// ITU-R BT.709 (HD television, sRGB)
    Bt709,
    /// ITU-R BT.2020 (Ultra-HD, HDR10)
    Bt2020,
}

impl ColorStandard {
    /// Returns the `(Kr, Kb)` luminance coefficients for the standard.
    ///
    /// Kg is derived as `1 - Kr - Kb`.
    #[must_use]
    pub fn kr_kb(self) -> (f32, f32) {
        match self {
            Self::Bt601 => (0.299, 0.114),
            Self::Bt709 => (0.2126, 0.0722),
            Self::Bt2020 => (0.2627, 0.0593),
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Bt601 => "BT.601",
            Self::Bt709 => "BT.709",
            Self::Bt2020 => "BT.2020",
        }
    }
}

// ─── RangeMode ───────────────────────────────────────────────────────────────

/// Quantization range for Y/Cb/Cr values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RangeMode {
    /// Full range: Y ∈ [0, 255], Cb/Cr ∈ [0, 255], center at 128.
    Full,
    /// Limited (studio) range: Y ∈ [16, 235], Cb/Cr ∈ [16, 240], center at 128.
    Limited,
}

// ─── ConversionMatrix ─────────────────────────────────────────────────────────

/// 3×3 forward (RGB→YCbCr) and inverse (YCbCr→RGB) matrices derived from
/// [`ColorStandard`] and [`RangeMode`].
#[derive(Debug, Clone)]
pub struct ConversionMatrix {
    /// Forward matrix coefficients [row_major; 9].
    pub fwd: [f32; 9],
    /// Inverse matrix coefficients [row_major; 9].
    pub inv: [f32; 9],
    /// Y output offset after matrix multiply (0 for full range, 16 for limited).
    pub y_bias: f32,
    /// Cb/Cr output offset (128 for both ranges).
    pub c_bias: f32,
    /// Y input offset when inverting (subtract before multiply).
    pub y_input_bias: f32,
    /// Scale factor for Y channel (full → 1.0, limited → 219/255).
    pub y_scale: f32,
    /// Scale factor for Cb/Cr channels (full → 1.0, limited → 224/255).
    pub c_scale: f32,
}

impl ConversionMatrix {
    /// Build a conversion matrix for the given color standard and range mode.
    #[must_use]
    pub fn new(standard: ColorStandard, range: RangeMode) -> Self {
        let (kr, kb) = standard.kr_kb();
        let kg = 1.0 - kr - kb;

        // Forward matrix: RGB → YCbCr (normalised, 0..1 inputs → 0..1 outputs)
        // Y  =  Kr·R + Kg·G + Kb·B
        // Cb = (B - Y) / (2·(1 - Kb))
        // Cr = (R - Y) / (2·(1 - Kr))
        let cb_scale = 0.5 / (1.0 - kb);
        let cr_scale = 0.5 / (1.0 - kr);

        // Row 0: Y
        let f0 = kr;
        let f1 = kg;
        let f2 = kb;
        // Row 1: Cb
        let f3 = -kr * cb_scale;
        let f4 = -kg * cb_scale;
        let f5 = (1.0 - kb) * cb_scale;
        // Row 2: Cr
        let f6 = (1.0 - kr) * cr_scale;
        let f7 = -kg * cr_scale;
        let f8 = -kb * cr_scale;

        let fwd = [f0, f1, f2, f3, f4, f5, f6, f7, f8];

        // Inverse matrix: YCbCr → RGB (normalised)
        // R = Y                + Cr / cr_scale
        // G = Y - Cb * (Kb / Kg) * cb_scale_inv - Cr * (Kr / Kg) * cr_scale_inv
        // B = Y + Cb / cb_scale
        let cb_scale_inv = 2.0 * (1.0 - kb);
        let cr_scale_inv = 2.0 * (1.0 - kr);
        // Row 0: R = Y + 0·Cb + Cr·cr_scale_inv
        let i0 = 1.0_f32;
        let i1 = 0.0_f32;
        let i2 = cr_scale_inv;
        // Row 1: G = Y - Cb·(kb*(2*(1-kb))/kg) - Cr·(kr*(2*(1-kr))/kg)
        let i3 = 1.0_f32;
        let i4 = -(kb * cb_scale_inv) / kg;
        let i5 = -(kr * cr_scale_inv) / kg;
        // Row 2: B = Y + Cb·cb_scale_inv + 0·Cr
        let i6 = 1.0_f32;
        let i7 = cb_scale_inv;
        let i8 = 0.0_f32;

        let inv = [i0, i1, i2, i3, i4, i5, i6, i7, i8];

        let (y_bias, c_bias, y_input_bias, y_scale, c_scale) = match range {
            RangeMode::Full => (0.0, 128.0, 0.0, 1.0, 1.0),
            RangeMode::Limited => (16.0, 128.0, 16.0, 219.0 / 255.0, 224.0 / 255.0),
        };

        Self {
            fwd,
            inv,
            y_bias,
            c_bias,
            y_input_bias,
            y_scale,
            c_scale,
        }
    }
}

// ─── BatchConvertStats ────────────────────────────────────────────────────────

/// Statistics produced after a batch color conversion.
#[derive(Debug, Clone, Default)]
pub struct BatchConvertStats {
    /// Total number of pixels processed.
    pub pixels_processed: u64,
    /// Number of pixel values clamped during conversion (out-of-range inputs).
    pub clamped_count: u64,
}

// ─── ColorConvertKernel ───────────────────────────────────────────────────────

/// GPU-style color space conversion kernel (CPU simulation via Rayon).
///
/// All operations work on packed RGBA (4 bytes per pixel) buffers.
/// The alpha channel is always passed through unchanged.
#[derive(Debug, Clone)]
pub struct ColorConvertKernel {
    standard: ColorStandard,
    range: RangeMode,
    matrix: ConversionMatrix,
}

impl ColorConvertKernel {
    /// Create a new kernel with the given color standard and range mode.
    #[must_use]
    pub fn new(standard: ColorStandard, range: RangeMode) -> Self {
        let matrix = ConversionMatrix::new(standard, range);
        Self {
            standard,
            range,
            matrix,
        }
    }

    /// The color standard used by this kernel.
    #[must_use]
    pub fn standard(&self) -> ColorStandard {
        self.standard
    }

    /// The range mode used by this kernel.
    #[must_use]
    pub fn range(&self) -> RangeMode {
        self.range
    }

    // ── Static helpers ────────────────────────────────────────────────────────

    /// Validate packed-RGBA buffer dimensions.
    fn validate_rgba(buf: &[u8], width: u32, height: u32) -> Result<usize, ColorKernelError> {
        if width == 0 || height == 0 {
            return Err(ColorKernelError::InvalidDimensions { width, height });
        }
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .ok_or(ColorKernelError::PixelCountOverflow { width, height })?;
        let expected = pixels * 4;
        if buf.len() != expected {
            return Err(ColorKernelError::BufferSizeMismatch {
                expected,
                actual: buf.len(),
            });
        }
        Ok(pixels)
    }

    // ── RGB → YUV (packed) ────────────────────────────────────────────────────

    /// Convert packed RGBA → packed YUVA (in-place style: src/dst separate).
    ///
    /// The A channel is passed through unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`ColorKernelError`] if buffer lengths or dimensions are invalid.
    pub fn rgb_to_yuv(
        src: &[u8],
        dst: &mut [u8],
        width: u32,
        height: u32,
        standard: ColorStandard,
        range: RangeMode,
    ) -> Result<BatchConvertStats, ColorKernelError> {
        Self::validate_rgba(src, width, height)?;
        let pixels = Self::validate_rgba(dst, width, height)?;
        let matrix = ConversionMatrix::new(standard, range);

        // Parallel chunk processing: 4 bytes per pixel.
        let clamped = std::sync::atomic::AtomicU64::new(0);
        src.par_chunks(4)
            .zip(dst.par_chunks_mut(4))
            .for_each(|(s, d)| {
                let r = s[0] as f32 / 255.0;
                let g = s[1] as f32 / 255.0;
                let b = s[2] as f32 / 255.0;
                let m = &matrix.fwd;

                // Y (luma)
                let y_norm = m[0] * r + m[1] * g + m[2] * b;
                // Cb (blue-difference chroma)
                let cb_norm = m[3] * r + m[4] * g + m[5] * b;
                // Cr (red-difference chroma)
                let cr_norm = m[6] * r + m[7] * g + m[8] * b;

                let y_raw = y_norm * matrix.y_scale * 255.0 + matrix.y_bias;
                let cb_raw = cb_norm * matrix.c_scale * 255.0 + matrix.c_bias;
                let cr_raw = cr_norm * matrix.c_scale * 255.0 + matrix.c_bias;

                let (y_clamped, cb_clamped, cr_clamped) = clamp3(y_raw, cb_raw, cr_raw);

                d[0] = y_clamped;
                d[1] = cb_clamped;
                d[2] = cr_clamped;
                d[3] = s[3]; // alpha pass-through

                let needs_clamp = y_clamped != y_raw.round() as u8
                    || cb_clamped != cb_raw.round() as u8
                    || cr_clamped != cr_raw.round() as u8;
                if needs_clamp {
                    clamped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            });

        Ok(BatchConvertStats {
            pixels_processed: pixels as u64,
            clamped_count: clamped.load(std::sync::atomic::Ordering::Relaxed),
        })
    }

    /// The cached [`ConversionMatrix`] for this kernel's standard and range.
    #[must_use]
    pub fn matrix(&self) -> &ConversionMatrix {
        &self.matrix
    }

    /// Instance method variant of [`rgb_to_yuv`].
    ///
    /// Uses the pre-built cached matrix rather than constructing a new one.
    ///
    /// [`rgb_to_yuv`]: ColorConvertKernel::rgb_to_yuv
    pub fn convert_rgb_to_yuv(
        &self,
        src: &[u8],
        dst: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<BatchConvertStats, ColorKernelError> {
        Self::validate_rgba(src, width, height)?;
        let pixels = Self::validate_rgba(dst, width, height)?;
        let matrix = &self.matrix;
        let clamped = std::sync::atomic::AtomicU64::new(0);
        src.par_chunks(4)
            .zip(dst.par_chunks_mut(4))
            .for_each(|(s, d)| {
                let r = s[0] as f32 / 255.0;
                let g = s[1] as f32 / 255.0;
                let b = s[2] as f32 / 255.0;
                let m = &matrix.fwd;
                let y_raw =
                    (m[0] * r + m[1] * g + m[2] * b) * matrix.y_scale * 255.0 + matrix.y_bias;
                let cb_raw =
                    (m[3] * r + m[4] * g + m[5] * b) * matrix.c_scale * 255.0 + matrix.c_bias;
                let cr_raw =
                    (m[6] * r + m[7] * g + m[8] * b) * matrix.c_scale * 255.0 + matrix.c_bias;
                let (y, cb, cr) = clamp3(y_raw, cb_raw, cr_raw);
                d[0] = y;
                d[1] = cb;
                d[2] = cr;
                d[3] = s[3];
                let needs_clamp = y != y_raw.round() as u8
                    || cb != cb_raw.round() as u8
                    || cr != cr_raw.round() as u8;
                if needs_clamp {
                    clamped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            });
        Ok(BatchConvertStats {
            pixels_processed: pixels as u64,
            clamped_count: clamped.load(std::sync::atomic::Ordering::Relaxed),
        })
    }

    // ── YUV → RGB (packed) ────────────────────────────────────────────────────

    /// Convert packed YUVA → packed RGBA.
    ///
    /// The A channel is passed through unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`ColorKernelError`] if buffer lengths or dimensions are invalid.
    pub fn yuv_to_rgb(
        src: &[u8],
        dst: &mut [u8],
        width: u32,
        height: u32,
        standard: ColorStandard,
        range: RangeMode,
    ) -> Result<BatchConvertStats, ColorKernelError> {
        Self::validate_rgba(src, width, height)?;
        let pixels = Self::validate_rgba(dst, width, height)?;
        let matrix = ConversionMatrix::new(standard, range);

        let clamped = std::sync::atomic::AtomicU64::new(0);
        src.par_chunks(4)
            .zip(dst.par_chunks_mut(4))
            .for_each(|(s, d)| {
                let y = (s[0] as f32 - matrix.y_input_bias) / (matrix.y_scale * 255.0);
                let cb = (s[1] as f32 - matrix.c_bias) / (matrix.c_scale * 255.0);
                let cr = (s[2] as f32 - matrix.c_bias) / (matrix.c_scale * 255.0);
                let m = &matrix.inv;

                let r_raw = (m[0] * y + m[1] * cb + m[2] * cr) * 255.0;
                let g_raw = (m[3] * y + m[4] * cb + m[5] * cr) * 255.0;
                let b_raw = (m[6] * y + m[7] * cb + m[8] * cr) * 255.0;

                let (r_c, g_c, b_c) = clamp3(r_raw, g_raw, b_raw);
                d[0] = r_c;
                d[1] = g_c;
                d[2] = b_c;
                d[3] = s[3]; // alpha

                let needs_clamp = r_c != r_raw.round().clamp(0.0, 255.0) as u8
                    || g_c != g_raw.round().clamp(0.0, 255.0) as u8
                    || b_c != b_raw.round().clamp(0.0, 255.0) as u8;
                if needs_clamp {
                    clamped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            });

        Ok(BatchConvertStats {
            pixels_processed: pixels as u64,
            clamped_count: clamped.load(std::sync::atomic::Ordering::Relaxed),
        })
    }

    /// Instance method variant of [`yuv_to_rgb`].
    ///
    /// Uses the pre-built cached matrix rather than constructing a new one.
    ///
    /// [`yuv_to_rgb`]: ColorConvertKernel::yuv_to_rgb
    pub fn convert_yuv_to_rgb(
        &self,
        src: &[u8],
        dst: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<BatchConvertStats, ColorKernelError> {
        Self::validate_rgba(src, width, height)?;
        let pixels = Self::validate_rgba(dst, width, height)?;
        let matrix = &self.matrix;
        let clamped = std::sync::atomic::AtomicU64::new(0);
        src.par_chunks(4)
            .zip(dst.par_chunks_mut(4))
            .for_each(|(s, d)| {
                let y = (s[0] as f32 - matrix.y_input_bias) / (matrix.y_scale * 255.0);
                let cb = (s[1] as f32 - matrix.c_bias) / (matrix.c_scale * 255.0);
                let cr = (s[2] as f32 - matrix.c_bias) / (matrix.c_scale * 255.0);
                let m = &matrix.inv;
                let r_raw = (m[0] * y + m[1] * cb + m[2] * cr) * 255.0;
                let g_raw = (m[3] * y + m[4] * cb + m[5] * cr) * 255.0;
                let b_raw = (m[6] * y + m[7] * cb + m[8] * cr) * 255.0;
                let (r, g, b) = clamp3(r_raw, g_raw, b_raw);
                d[0] = r;
                d[1] = g;
                d[2] = b;
                d[3] = s[3];
                let needs_clamp = r != r_raw.round().clamp(0.0, 255.0) as u8
                    || g != g_raw.round().clamp(0.0, 255.0) as u8
                    || b != b_raw.round().clamp(0.0, 255.0) as u8;
                if needs_clamp {
                    clamped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            });
        Ok(BatchConvertStats {
            pixels_processed: pixels as u64,
            clamped_count: clamped.load(std::sync::atomic::Ordering::Relaxed),
        })
    }

    // ── Limited ↔ Full range expansion / compression ──────────────────────────

    /// Expand a limited-range (studio swing) packed RGBA buffer to full range.
    ///
    /// Y channel: `[16, 235]` → `[0, 255]`.
    /// Cb/Cr channels: `[16, 240]` → `[0, 255]`.
    ///
    /// # Errors
    ///
    /// Returns [`ColorKernelError`] on dimension or size mismatch.
    pub fn expand_limited_to_full(
        src: &[u8],
        dst: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<BatchConvertStats, ColorKernelError> {
        Self::validate_rgba(src, width, height)?;
        let pixels = Self::validate_rgba(dst, width, height)?;

        src.par_chunks(4)
            .zip(dst.par_chunks_mut(4))
            .for_each(|(s, d)| {
                // Y channel: limited [16..235] → full [0..255]
                let y = ((s[0] as f32 - 16.0) * 255.0 / 219.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                // Cb channel: limited [16..240] → full [0..255]
                let cb = ((s[1] as f32 - 128.0) * 255.0 / 224.0 + 128.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                // Cr channel
                let cr = ((s[2] as f32 - 128.0) * 255.0 / 224.0 + 128.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                d[0] = y;
                d[1] = cb;
                d[2] = cr;
                d[3] = s[3];
            });

        Ok(BatchConvertStats {
            pixels_processed: pixels as u64,
            clamped_count: 0,
        })
    }

    /// Compress a full-range packed YUVA buffer to limited (studio swing) range.
    ///
    /// Y channel: `[0, 255]` → `[16, 235]`.
    /// Cb/Cr channels: `[0, 255]` → `[16, 240]`.
    ///
    /// # Errors
    ///
    /// Returns [`ColorKernelError`] on dimension or size mismatch.
    pub fn compress_full_to_limited(
        src: &[u8],
        dst: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<BatchConvertStats, ColorKernelError> {
        Self::validate_rgba(src, width, height)?;
        let pixels = Self::validate_rgba(dst, width, height)?;

        src.par_chunks(4)
            .zip(dst.par_chunks_mut(4))
            .for_each(|(s, d)| {
                let y = (s[0] as f32 * 219.0 / 255.0 + 16.0)
                    .round()
                    .clamp(16.0, 235.0) as u8;
                let cb = ((s[1] as f32 - 128.0) * 224.0 / 255.0 + 128.0)
                    .round()
                    .clamp(16.0, 240.0) as u8;
                let cr = ((s[2] as f32 - 128.0) * 224.0 / 255.0 + 128.0)
                    .round()
                    .clamp(16.0, 240.0) as u8;
                d[0] = y;
                d[1] = cb;
                d[2] = cr;
                d[3] = s[3];
            });

        Ok(BatchConvertStats {
            pixels_processed: pixels as u64,
            clamped_count: 0,
        })
    }

    // ── Packed → Planar (4:4:4) ───────────────────────────────────────────────

    /// Convert packed RGBA to planar YUV 4:4:4 (separate Y, Cb, Cr planes).
    ///
    /// Returns `(Y_plane, Cb_plane, Cr_plane)`, each `width * height` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`ColorKernelError`] on dimension or size mismatch.
    pub fn rgba_to_planar_yuv444(
        src: &[u8],
        width: u32,
        height: u32,
        standard: ColorStandard,
        range: RangeMode,
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), ColorKernelError> {
        let pixels = Self::validate_rgba(src, width, height)?;
        let matrix = ConversionMatrix::new(standard, range);

        let mut y_plane = vec![0u8; pixels];
        let mut cb_plane = vec![0u8; pixels];
        let mut cr_plane = vec![0u8; pixels];

        // Parallel computation of each pixel's YCbCr.
        let results: Vec<(u8, u8, u8)> = src
            .par_chunks(4)
            .map(|s| {
                let r = s[0] as f32 / 255.0;
                let g = s[1] as f32 / 255.0;
                let b = s[2] as f32 / 255.0;
                let m = &matrix.fwd;
                let y_raw =
                    (m[0] * r + m[1] * g + m[2] * b) * matrix.y_scale * 255.0 + matrix.y_bias;
                let cb_raw =
                    (m[3] * r + m[4] * g + m[5] * b) * matrix.c_scale * 255.0 + matrix.c_bias;
                let cr_raw =
                    (m[6] * r + m[7] * g + m[8] * b) * matrix.c_scale * 255.0 + matrix.c_bias;
                let (y, cb, cr) = clamp3(y_raw, cb_raw, cr_raw);
                (y, cb, cr)
            })
            .collect();

        for (i, (y, cb, cr)) in results.into_iter().enumerate() {
            y_plane[i] = y;
            cb_plane[i] = cb;
            cr_plane[i] = cr;
        }

        Ok((y_plane, cb_plane, cr_plane))
    }

    /// Convert planar YUV 4:4:4 to packed RGBA.
    ///
    /// Alpha channel is set to 255.
    ///
    /// # Errors
    ///
    /// Returns [`ColorKernelError`] if any plane size or dimensions are invalid.
    pub fn planar_yuv444_to_rgba(
        y_plane: &[u8],
        cb_plane: &[u8],
        cr_plane: &[u8],
        width: u32,
        height: u32,
        standard: ColorStandard,
        range: RangeMode,
    ) -> Result<Vec<u8>, ColorKernelError> {
        if width == 0 || height == 0 {
            return Err(ColorKernelError::InvalidDimensions { width, height });
        }
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .ok_or(ColorKernelError::PixelCountOverflow { width, height })?;
        for (plane, name) in [y_plane, cb_plane, cr_plane].iter().zip(["Y", "Cb", "Cr"]) {
            if plane.len() != pixels {
                return Err(ColorKernelError::BufferSizeMismatch {
                    expected: pixels,
                    actual: plane.len(),
                });
            }
            let _ = name;
        }

        let matrix = ConversionMatrix::new(standard, range);
        let rgba: Vec<u8> = y_plane
            .par_iter()
            .zip(cb_plane.par_iter())
            .zip(cr_plane.par_iter())
            .flat_map(|((&y, &cb), &cr)| {
                let yn = (y as f32 - matrix.y_input_bias) / (matrix.y_scale * 255.0);
                let cbn = (cb as f32 - matrix.c_bias) / (matrix.c_scale * 255.0);
                let crn = (cr as f32 - matrix.c_bias) / (matrix.c_scale * 255.0);
                let m = &matrix.inv;
                let r_raw = (m[0] * yn + m[1] * cbn + m[2] * crn) * 255.0;
                let g_raw = (m[3] * yn + m[4] * cbn + m[5] * crn) * 255.0;
                let b_raw = (m[6] * yn + m[7] * cbn + m[8] * crn) * 255.0;
                let (r, g, b) = clamp3(r_raw, g_raw, b_raw);
                [r, g, b, 255u8]
            })
            .collect();

        Ok(rgba)
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Clamp three f32 channel values to `[0.0, 255.0]` and convert to u8.
#[inline]
fn clamp3(a: f32, b: f32, c: f32) -> (u8, u8, u8) {
    (
        a.round().clamp(0.0, 255.0) as u8,
        b.round().clamp(0.0, 255.0) as u8,
        c.round().clamp(0.0, 255.0) as u8,
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ColorStandard ─────────────────────────────────────────────────────────

    #[test]
    fn test_color_standard_kr_kb_bt601() {
        let (kr, kb) = ColorStandard::Bt601.kr_kb();
        assert!((kr - 0.299).abs() < 1e-6);
        assert!((kb - 0.114).abs() < 1e-6);
    }

    #[test]
    fn test_color_standard_kr_kb_bt709() {
        let (kr, kb) = ColorStandard::Bt709.kr_kb();
        assert!((kr - 0.2126).abs() < 1e-6);
        assert!((kb - 0.0722).abs() < 1e-6);
    }

    #[test]
    fn test_color_standard_kr_kb_bt2020() {
        let (kr, kb) = ColorStandard::Bt2020.kr_kb();
        assert!((kr - 0.2627).abs() < 1e-6);
        assert!((kb - 0.0593).abs() < 1e-6);
    }

    #[test]
    fn test_color_standard_kg_sums_to_one() {
        for std in [
            ColorStandard::Bt601,
            ColorStandard::Bt709,
            ColorStandard::Bt2020,
        ] {
            let (kr, kb) = std.kr_kb();
            let kg = 1.0 - kr - kb;
            assert!(
                (kr + kg + kb - 1.0).abs() < 1e-5,
                "{}: kr+kg+kb != 1",
                std.label()
            );
        }
    }

    // ── ConversionMatrix ──────────────────────────────────────────────────────

    #[test]
    fn test_conversion_matrix_full_range_bias() {
        let m = ConversionMatrix::new(ColorStandard::Bt709, RangeMode::Full);
        assert_eq!(m.y_bias, 0.0);
        assert_eq!(m.c_bias, 128.0);
        assert!((m.y_scale - 1.0).abs() < 1e-6);
        assert!((m.c_scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_conversion_matrix_limited_range_bias() {
        let m = ConversionMatrix::new(ColorStandard::Bt709, RangeMode::Limited);
        assert_eq!(m.y_bias, 16.0);
        assert_eq!(m.c_bias, 128.0);
        assert!((m.y_scale - 219.0 / 255.0).abs() < 1e-6);
        assert!((m.c_scale - 224.0 / 255.0).abs() < 1e-6);
    }

    // ── rgb_to_yuv / yuv_to_rgb round-trip ───────────────────────────────────

    fn make_rgba_pixel(r: u8, g: u8, b: u8) -> Vec<u8> {
        vec![r, g, b, 255]
    }

    fn roundtrip_pixel(
        r: u8,
        g: u8,
        b: u8,
        standard: ColorStandard,
        range: RangeMode,
        tolerance: i16,
    ) {
        let src = make_rgba_pixel(r, g, b);
        let mut yuv = vec![0u8; 4];
        ColorConvertKernel::rgb_to_yuv(&src, &mut yuv, 1, 1, standard, range).unwrap();
        let mut rgb_back = vec![0u8; 4];
        ColorConvertKernel::yuv_to_rgb(&yuv, &mut rgb_back, 1, 1, standard, range).unwrap();
        for (i, (&orig, &back)) in src.iter().zip(rgb_back.iter()).enumerate().take(3) {
            let diff = (orig as i16 - back as i16).abs();
            assert!(
                diff <= tolerance,
                "channel {i}: orig={orig} back={back} diff={diff} > tol={tolerance} (std={}, range={:?})",
                standard.label(), range
            );
        }
        // Alpha must be preserved exactly
        assert_eq!(rgb_back[3], 255);
    }

    #[test]
    fn test_roundtrip_white_bt709_full() {
        roundtrip_pixel(255, 255, 255, ColorStandard::Bt709, RangeMode::Full, 2);
    }

    #[test]
    fn test_roundtrip_black_bt709_full() {
        roundtrip_pixel(0, 0, 0, ColorStandard::Bt709, RangeMode::Full, 2);
    }

    #[test]
    fn test_roundtrip_red_bt709_full() {
        roundtrip_pixel(255, 0, 0, ColorStandard::Bt709, RangeMode::Full, 2);
    }

    #[test]
    fn test_roundtrip_green_bt709_full() {
        roundtrip_pixel(0, 255, 0, ColorStandard::Bt709, RangeMode::Full, 2);
    }

    #[test]
    fn test_roundtrip_blue_bt709_full() {
        roundtrip_pixel(0, 0, 255, ColorStandard::Bt709, RangeMode::Full, 2);
    }

    #[test]
    fn test_roundtrip_gray_bt709_full() {
        roundtrip_pixel(128, 128, 128, ColorStandard::Bt709, RangeMode::Full, 2);
    }

    #[test]
    fn test_roundtrip_bt601_full() {
        roundtrip_pixel(180, 90, 60, ColorStandard::Bt601, RangeMode::Full, 2);
    }

    #[test]
    fn test_roundtrip_bt2020_full() {
        roundtrip_pixel(100, 200, 150, ColorStandard::Bt2020, RangeMode::Full, 2);
    }

    #[test]
    fn test_roundtrip_bt709_limited() {
        // Limited-range round-trip has ~2 LSB precision loss.
        roundtrip_pixel(200, 100, 50, ColorStandard::Bt709, RangeMode::Limited, 3);
    }

    // ── Error handling ────────────────────────────────────────────────────────

    #[test]
    fn test_rgb_to_yuv_zero_dimensions() {
        let src = vec![0u8; 4];
        let mut dst = vec![0u8; 4];
        let err = ColorConvertKernel::rgb_to_yuv(
            &src,
            &mut dst,
            0,
            1,
            ColorStandard::Bt709,
            RangeMode::Full,
        );
        assert!(matches!(
            err,
            Err(ColorKernelError::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn test_rgb_to_yuv_buffer_mismatch() {
        let src = vec![0u8; 4];
        let mut dst = vec![0u8; 8]; // wrong size
        let err = ColorConvertKernel::rgb_to_yuv(
            &src,
            &mut dst,
            1,
            1,
            ColorStandard::Bt709,
            RangeMode::Full,
        );
        assert!(matches!(
            err,
            Err(ColorKernelError::BufferSizeMismatch { .. })
        ));
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_pixels_processed() {
        let src = vec![128u8; 4 * 16]; // 16 pixels
        let mut dst = vec![0u8; 4 * 16];
        let stats = ColorConvertKernel::rgb_to_yuv(
            &src,
            &mut dst,
            4,
            4,
            ColorStandard::Bt709,
            RangeMode::Full,
        )
        .unwrap();
        assert_eq!(stats.pixels_processed, 16);
    }

    // ── Limited ↔ Full range ──────────────────────────────────────────────────

    #[test]
    fn test_limited_to_full_y_white() {
        // Y=235 (limited white) → Y=255 (full white)
        let src = vec![235u8, 128, 128, 255]; // Y=235, Cb=128, Cr=128
        let mut dst = vec![0u8; 4];
        ColorConvertKernel::expand_limited_to_full(&src, &mut dst, 1, 1).unwrap();
        assert_eq!(dst[0], 255, "limited Y=235 should map to full Y=255");
    }

    #[test]
    fn test_limited_to_full_y_black() {
        // Y=16 (limited black) → Y=0 (full black)
        let src = vec![16u8, 128, 128, 255];
        let mut dst = vec![0u8; 4];
        ColorConvertKernel::expand_limited_to_full(&src, &mut dst, 1, 1).unwrap();
        assert_eq!(dst[0], 0, "limited Y=16 should map to full Y=0");
    }

    #[test]
    fn test_compress_full_to_limited_white() {
        // Y=255 (full white) → Y=235 (limited white)
        let src = vec![255u8, 128, 128, 255];
        let mut dst = vec![0u8; 4];
        ColorConvertKernel::compress_full_to_limited(&src, &mut dst, 1, 1).unwrap();
        assert_eq!(dst[0], 235, "full Y=255 should compress to limited Y=235");
    }

    #[test]
    fn test_compress_and_expand_roundtrip() {
        let src = vec![128u8, 128, 128, 255];
        let mut limited = vec![0u8; 4];
        ColorConvertKernel::compress_full_to_limited(&src, &mut limited, 1, 1).unwrap();
        let mut back = vec![0u8; 4];
        ColorConvertKernel::expand_limited_to_full(&limited, &mut back, 1, 1).unwrap();
        for i in 0..3 {
            let diff = (src[i] as i16 - back[i] as i16).abs();
            assert!(diff <= 2, "channel {i}: diff={diff}");
        }
    }

    // ── Planar YUV 4:4:4 ─────────────────────────────────────────────────────

    #[test]
    fn test_rgba_to_planar_yuv444_size() {
        let src = vec![128u8; 4 * 4 * 4]; // 4×4 RGBA
        let (y, cb, cr) = ColorConvertKernel::rgba_to_planar_yuv444(
            &src,
            4,
            4,
            ColorStandard::Bt709,
            RangeMode::Full,
        )
        .unwrap();
        assert_eq!(y.len(), 16);
        assert_eq!(cb.len(), 16);
        assert_eq!(cr.len(), 16);
    }

    #[test]
    fn test_planar_yuv444_roundtrip() {
        let src: Vec<u8> = (0..4 * 4 * 4).map(|i| (i * 17 % 256) as u8).collect();
        let (y, cb, cr) = ColorConvertKernel::rgba_to_planar_yuv444(
            &src,
            4,
            4,
            ColorStandard::Bt709,
            RangeMode::Full,
        )
        .unwrap();
        let rgba = ColorConvertKernel::planar_yuv444_to_rgba(
            &y,
            &cb,
            &cr,
            4,
            4,
            ColorStandard::Bt709,
            RangeMode::Full,
        )
        .unwrap();
        assert_eq!(rgba.len(), 4 * 4 * 4);
        for i in (0..rgba.len()).step_by(4).take(3) {
            let dr = (src[i] as i16 - rgba[i] as i16).abs();
            let dg = (src[i + 1] as i16 - rgba[i + 1] as i16).abs();
            let db = (src[i + 2] as i16 - rgba[i + 2] as i16).abs();
            assert!(dr <= 3, "R channel diff={dr} at pixel {}", i / 4);
            assert!(dg <= 3, "G channel diff={dg} at pixel {}", i / 4);
            assert!(db <= 3, "B channel diff={db} at pixel {}", i / 4);
        }
    }

    // ── Instance methods ──────────────────────────────────────────────────────

    #[test]
    fn test_kernel_instance_standard_and_range() {
        let k = ColorConvertKernel::new(ColorStandard::Bt2020, RangeMode::Limited);
        assert_eq!(k.standard(), ColorStandard::Bt2020);
        assert_eq!(k.range(), RangeMode::Limited);
    }

    #[test]
    fn test_kernel_instance_convert_rgb_to_yuv() {
        let k = ColorConvertKernel::new(ColorStandard::Bt709, RangeMode::Full);
        let src = vec![100u8, 150, 200, 255];
        let mut dst = vec![0u8; 4];
        let stats = k.convert_rgb_to_yuv(&src, &mut dst, 1, 1).unwrap();
        assert_eq!(stats.pixels_processed, 1);
        assert_eq!(dst[3], 255); // alpha preserved
    }

    #[test]
    fn test_multi_pixel_batch() {
        let w = 8u32;
        let h = 8u32;
        let src: Vec<u8> = (0..w * h * 4).map(|i| (i % 256) as u8).collect();
        let mut yuv = vec![0u8; (w * h * 4) as usize];
        let mut rgb_back = vec![0u8; (w * h * 4) as usize];
        ColorConvertKernel::rgb_to_yuv(&src, &mut yuv, w, h, ColorStandard::Bt709, RangeMode::Full)
            .unwrap();
        ColorConvertKernel::yuv_to_rgb(
            &yuv,
            &mut rgb_back,
            w,
            h,
            ColorStandard::Bt709,
            RangeMode::Full,
        )
        .unwrap();
        // Alpha channel must be preserved
        for i in (3..src.len()).step_by(4) {
            assert_eq!(rgb_back[i], src[i]);
        }
    }
}
