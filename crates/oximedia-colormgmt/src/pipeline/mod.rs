//! Color transformation pipeline for chaining multiple operations.

use crate::colorspaces::ColorSpace;
use crate::gamut::{GamutMapper, GamutMappingAlgorithm};
use crate::hdr::{ToneMapper, ToneMappingOperator};
use crate::math::matrix::Matrix3x3;
use crate::transforms::lut::{Lut1D, Lut3D};

/// A color transformation pipeline that chains multiple operations.
///
/// # Examples
///
/// ```
/// use oximedia_colormgmt::pipeline::{ColorPipeline, ColorTransform};
/// use oximedia_colormgmt::colorspaces::ColorSpace;
///
/// let srgb = ColorSpace::srgb()?;
/// let rec2020 = ColorSpace::rec2020()?;
///
/// let mut pipeline = ColorPipeline::new();
/// pipeline.add_transform(ColorTransform::Linearize(srgb.clone()));
/// pipeline.add_transform(ColorTransform::ColorSpaceConversion {
///     from: srgb,
///     to: rec2020.clone(),
/// });
/// pipeline.add_transform(ColorTransform::Delinearize(rec2020));
///
/// let input = [0.5, 0.3, 0.7];
/// let output = pipeline.transform_pixel(input);
/// ```
pub struct ColorPipeline {
    transforms: Vec<ColorTransform>,
    name: String,
}

/// A single color transformation step.
#[derive(Clone)]
pub enum ColorTransform {
    /// Apply matrix transformation to RGB
    Matrix(Matrix3x3),

    /// Apply 1D LUT to each channel
    Lut1D(Lut1D),

    /// Apply 3D LUT
    Lut3D(Lut3D),

    /// Linearize using a color space's transfer function
    Linearize(ColorSpace),

    /// Delinearize using a color space's transfer function
    Delinearize(ColorSpace),

    /// Convert between color spaces
    ColorSpaceConversion {
        /// Source color space
        from: ColorSpace,
        /// Destination color space
        to: ColorSpace,
    },

    /// Apply gamut mapping
    GamutMap(GamutMapper),

    /// Apply tone mapping (HDR to SDR)
    ToneMap(ToneMapper),

    /// Apply gamma correction
    Gamma(f64),

    /// Apply exposure adjustment (in stops)
    Exposure(f64),

    /// Apply brightness adjustment
    Brightness(f64),

    /// Apply contrast adjustment
    Contrast(f64),

    /// Apply saturation adjustment
    Saturation(f64),

    /// Custom function
    Custom(fn([f64; 3]) -> [f64; 3]),
}

impl ColorPipeline {
    /// Creates a new empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
            name: "Color Pipeline".to_string(),
        }
    }

    /// Creates a new pipeline with a name.
    #[must_use]
    pub fn with_name(name: String) -> Self {
        Self {
            transforms: Vec::new(),
            name,
        }
    }

    /// Adds a transform to the pipeline.
    pub fn add_transform(&mut self, transform: ColorTransform) {
        self.transforms.push(transform);
    }

    /// Inserts a transform at the specified position.
    pub fn insert_transform(&mut self, index: usize, transform: ColorTransform) {
        self.transforms.insert(index, transform);
    }

    /// Removes a transform at the specified position.
    pub fn remove_transform(&mut self, index: usize) -> Option<ColorTransform> {
        if index < self.transforms.len() {
            Some(self.transforms.remove(index))
        } else {
            None
        }
    }

    /// Returns the number of transforms in the pipeline.
    #[must_use]
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Returns true if the pipeline is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }

    /// Clears all transforms from the pipeline.
    pub fn clear(&mut self) {
        self.transforms.clear();
    }

    /// Applies the pipeline to a single RGB pixel.
    #[must_use]
    pub fn transform_pixel(&self, mut rgb: [f64; 3]) -> [f64; 3] {
        for transform in &self.transforms {
            rgb = Self::apply_transform(transform, rgb);
        }
        rgb
    }

    /// Applies the pipeline to an image (slice of RGB triplets).
    ///
    /// # Panics
    ///
    /// Panics if the data length is not a multiple of 3.
    pub fn transform_image(&self, data: &mut [f64]) {
        assert_eq!(
            data.len() % 3,
            0,
            "Image data length must be a multiple of 3"
        );

        for chunk in data.chunks_exact_mut(3) {
            let rgb = [chunk[0], chunk[1], chunk[2]];
            let result = self.transform_pixel(rgb);
            chunk[0] = result[0];
            chunk[1] = result[1];
            chunk[2] = result[2];
        }
    }

    /// Applies the pipeline to an image in parallel using rayon.
    ///
    /// # Panics
    ///
    /// Panics if the data length is not a multiple of 3.
    #[cfg(feature = "rayon")]
    pub fn transform_image_parallel(&self, data: &mut [f64]) {
        use rayon::prelude::*;

        assert_eq!(
            data.len() % 3,
            0,
            "Image data length must be a multiple of 3"
        );

        data.par_chunks_exact_mut(3).for_each(|chunk| {
            let rgb = [chunk[0], chunk[1], chunk[2]];
            let result = self.transform_pixel(rgb);
            chunk[0] = result[0];
            chunk[1] = result[1];
            chunk[2] = result[2];
        });
    }

    fn apply_transform(transform: &ColorTransform, rgb: [f64; 3]) -> [f64; 3] {
        match transform {
            ColorTransform::Matrix(matrix) => {
                crate::math::matrix::multiply_matrix_vector(matrix, rgb)
            }
            ColorTransform::Lut1D(lut) => lut.apply(rgb),
            ColorTransform::Lut3D(lut) => lut.apply(rgb).unwrap_or(rgb),
            ColorTransform::Linearize(cs) => cs.linearize(rgb),
            ColorTransform::Delinearize(cs) => cs.delinearize(rgb),
            ColorTransform::ColorSpaceConversion { from, to } => {
                crate::transforms::rgb_to_rgb(&rgb, from, to)
            }
            ColorTransform::GamutMap(mapper) => mapper.map(rgb, None),
            ColorTransform::ToneMap(mapper) => mapper.apply(rgb),
            ColorTransform::Gamma(gamma) => crate::transforms::parametric::apply_gamma(rgb, *gamma),
            ColorTransform::Exposure(exposure) => {
                crate::transforms::parametric::apply_exposure(rgb, *exposure)
            }
            ColorTransform::Brightness(brightness) => {
                crate::transforms::parametric::apply_brightness(rgb, *brightness)
            }
            ColorTransform::Contrast(contrast) => {
                crate::transforms::parametric::apply_contrast(rgb, *contrast)
            }
            ColorTransform::Saturation(saturation) => {
                crate::transforms::parametric::apply_saturation(rgb, *saturation)
            }
            ColorTransform::Custom(func) => func(rgb),
        }
    }

    /// Returns the name of this pipeline.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Default for ColorPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating common color pipelines.
pub struct PipelineBuilder {
    pipeline: ColorPipeline,
}

impl PipelineBuilder {
    /// Creates a new pipeline builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pipeline: ColorPipeline::new(),
        }
    }

    /// Sets the pipeline name.
    #[must_use]
    pub fn with_name(mut self, name: String) -> Self {
        self.pipeline.name = name;
        self
    }

    /// Adds linearization.
    #[must_use]
    pub fn linearize(mut self, color_space: ColorSpace) -> Self {
        self.pipeline
            .add_transform(ColorTransform::Linearize(color_space));
        self
    }

    /// Adds delinearization.
    #[must_use]
    pub fn delinearize(mut self, color_space: ColorSpace) -> Self {
        self.pipeline
            .add_transform(ColorTransform::Delinearize(color_space));
        self
    }

    /// Adds color space conversion.
    #[must_use]
    pub fn convert(mut self, from: ColorSpace, to: ColorSpace) -> Self {
        self.pipeline
            .add_transform(ColorTransform::ColorSpaceConversion { from, to });
        self
    }

    /// Adds gamut mapping.
    #[must_use]
    pub fn gamut_map(mut self, algorithm: GamutMappingAlgorithm) -> Self {
        self.pipeline
            .add_transform(ColorTransform::GamutMap(GamutMapper::new(algorithm)));
        self
    }

    /// Adds tone mapping.
    #[must_use]
    pub fn tone_map(mut self, operator: ToneMappingOperator, peak_in: f64, peak_out: f64) -> Self {
        self.pipeline
            .add_transform(ColorTransform::ToneMap(ToneMapper::new(
                operator, peak_in, peak_out,
            )));
        self
    }

    /// Adds a matrix transform.
    #[must_use]
    pub fn matrix(mut self, matrix: Matrix3x3) -> Self {
        self.pipeline.add_transform(ColorTransform::Matrix(matrix));
        self
    }

    /// Adds a 3D LUT.
    #[must_use]
    pub fn lut3d(mut self, lut: Lut3D) -> Self {
        self.pipeline.add_transform(ColorTransform::Lut3D(lut));
        self
    }

    /// Builds the pipeline.
    #[must_use]
    pub fn build(self) -> ColorPipeline {
        self.pipeline
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_pipeline() {
        let pipeline = ColorPipeline::new();
        let rgb = [0.5, 0.3, 0.7];
        let result = pipeline.transform_pixel(rgb);
        assert_eq!(result, rgb);
    }

    #[test]
    fn test_gamma_pipeline() {
        let mut pipeline = ColorPipeline::new();
        pipeline.add_transform(ColorTransform::Gamma(2.2));
        pipeline.add_transform(ColorTransform::Gamma(1.0 / 2.2));

        let rgb = [0.5, 0.3, 0.7];
        let result = pipeline.transform_pixel(rgb);

        assert!((result[0] - rgb[0]).abs() < 1e-10);
        assert!((result[1] - rgb[1]).abs() < 1e-10);
        assert!((result[2] - rgb[2]).abs() < 1e-10);
    }

    #[test]
    fn test_pipeline_builder() {
        let srgb = ColorSpace::srgb().expect("sRGB color space creation should succeed");
        let rec2020 = ColorSpace::rec2020().expect("Rec.2020 color space creation should succeed");

        let pipeline = PipelineBuilder::new()
            .with_name("sRGB to Rec.2020".to_string())
            .linearize(srgb.clone())
            .convert(srgb, rec2020.clone())
            .delinearize(rec2020)
            .build();

        assert_eq!(pipeline.len(), 3);
        assert_eq!(pipeline.name(), "sRGB to Rec.2020");
    }

    #[test]
    fn test_pipeline_operations() {
        let mut pipeline = ColorPipeline::new();
        pipeline.add_transform(ColorTransform::Gamma(2.2));

        assert_eq!(pipeline.len(), 1);
        assert!(!pipeline.is_empty());

        pipeline.remove_transform(0);
        assert_eq!(pipeline.len(), 0);
        assert!(pipeline.is_empty());
    }

    #[test]
    fn test_transform_image() {
        let pipeline = ColorPipeline::new();
        let mut data = vec![0.5, 0.3, 0.7, 0.2, 0.4, 0.6];

        pipeline.transform_image(&mut data);

        // Should be unchanged (empty pipeline)
        assert_eq!(data, vec![0.5, 0.3, 0.7, 0.2, 0.4, 0.6]);
    }
}
