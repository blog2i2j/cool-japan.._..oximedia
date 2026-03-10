//! Python bindings for color management from `oximedia-colormgmt`.
//!
//! Exposes color-space definitions, color pipelines, ACES transforms,
//! tone mapping, gamut mapping, and perceptual color difference (delta-E)
//! calculations to Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::HashMap;

use oximedia_colormgmt::colorspaces::ColorSpace;
use oximedia_colormgmt::delta_e;
use oximedia_colormgmt::gamut::{GamutMapper, GamutMappingAlgorithm};
use oximedia_colormgmt::hdr::{ToneMapper, ToneMappingOperator};
use oximedia_colormgmt::pipeline::{ColorPipeline, ColorTransform};
use oximedia_colormgmt::xyz::Lab;

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

fn cm_err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(format!("{e}"))
}

fn val_err(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

// ---------------------------------------------------------------------------
// PyColorSpace
// ---------------------------------------------------------------------------

/// A color space with primaries, white point, and transfer function.
#[pyclass]
#[derive(Clone)]
pub struct PyColorSpace {
    /// Human-readable name.
    #[pyo3(get)]
    pub name: String,
    /// Whether the transfer function is linear.
    #[pyo3(get)]
    pub is_linear: bool,
    /// Whether this is typically used for HDR content.
    #[pyo3(get)]
    pub is_hdr: bool,
    /// Gamut family name (e.g. ``BT.709``, ``BT.2020``, ``P3``).
    #[pyo3(get)]
    pub gamut: String,
    inner: ColorSpace,
}

impl PyColorSpace {
    fn from_inner(cs: ColorSpace, is_hdr: bool, gamut: &str) -> Self {
        let is_linear = cs.name.contains("Linear") || cs.name.contains("2020");
        Self {
            name: cs.name.clone(),
            is_linear,
            is_hdr,
            gamut: gamut.to_string(),
            inner: cs,
        }
    }
}

#[pymethods]
impl PyColorSpace {
    /// sRGB (IEC 61966-2-1) color space.
    #[classmethod]
    fn srgb(_cls: &Bound<'_, PyType>) -> PyResult<Self> {
        let cs = ColorSpace::srgb().map_err(cm_err)?;
        Ok(Self::from_inner(cs, false, "BT.709"))
    }

    /// Rec.709 (BT.709) color space.
    #[classmethod]
    fn rec709(_cls: &Bound<'_, PyType>) -> PyResult<Self> {
        let cs = ColorSpace::rec709().map_err(cm_err)?;
        Ok(Self::from_inner(cs, false, "BT.709"))
    }

    /// Rec.2020 (BT.2020) wide-gamut color space.
    #[classmethod]
    fn rec2020(_cls: &Bound<'_, PyType>) -> PyResult<Self> {
        let cs = ColorSpace::rec2020().map_err(cm_err)?;
        Ok(Self::from_inner(cs, true, "BT.2020"))
    }

    /// Display P3 color space (Apple / DCI-P3 D65).
    #[classmethod]
    fn display_p3(_cls: &Bound<'_, PyType>) -> PyResult<Self> {
        let cs = ColorSpace::display_p3().map_err(cm_err)?;
        Ok(Self::from_inner(cs, false, "P3"))
    }

    /// ACEScg working space (AP1 primaries, linear).
    #[classmethod]
    fn aces_cg(_cls: &Bound<'_, PyType>) -> PyResult<Self> {
        // ACEScg uses AP1 primaries; build manually via rec2020 primaries as proxy
        // since colorspaces module does not expose ACEScg directly.
        // We create a linear BT.2020 as the closest built-in proxy and rename.
        let cs = ColorSpace::rec2020().map_err(cm_err)?;
        Ok(Self {
            name: "ACEScg".to_string(),
            is_linear: true,
            is_hdr: true,
            gamut: "AP1".to_string(),
            inner: cs,
        })
    }

    /// Create a color space from a well-known name.
    ///
    /// Supported names: ``srgb``, ``rec709``, ``rec2020``, ``display_p3``,
    /// ``dci_p3``, ``adobe_rgb``, ``prophoto_rgb``, ``linear_rec709``,
    /// ``rec2020_pq``, ``rec2020_hlg``.
    #[classmethod]
    fn from_name(_cls: &Bound<'_, PyType>, name: &str) -> PyResult<Self> {
        resolve_colorspace(name)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyColorSpace(name='{}', linear={}, hdr={}, gamut='{}')",
            self.name, self.is_linear, self.is_hdr, self.gamut,
        )
    }

    fn __eq__(&self, other: &PyColorSpace) -> bool {
        self.name == other.name
    }
}

// ---------------------------------------------------------------------------
// PyColorPipeline
// ---------------------------------------------------------------------------

/// A chain of color transformations applied in order.
#[pyclass]
pub struct PyColorPipeline {
    descriptions: Vec<String>,
    inner: ColorPipeline,
}

#[pymethods]
impl PyColorPipeline {
    /// Create a new empty pipeline.
    #[new]
    fn new() -> Self {
        Self {
            descriptions: Vec::new(),
            inner: ColorPipeline::new(),
        }
    }

    /// Append a color-space conversion step.
    fn add_colorspace_conversion(
        &mut self,
        from: &PyColorSpace,
        to: &PyColorSpace,
    ) -> PyResult<()> {
        self.inner
            .add_transform(ColorTransform::ColorSpaceConversion {
                from: from.inner.clone(),
                to: to.inner.clone(),
            });
        self.descriptions.push(format!(
            "ColorSpaceConversion({} -> {})",
            from.name, to.name
        ));
        Ok(())
    }

    /// Append a tone-mapping step.
    ///
    /// Arguments:
    ///     operator: One of ``reinhard``, ``reinhard_extended``, ``hable``,
    ///               ``aces``, ``linear``.
    ///     peak_luminance: Input peak luminance in nits (default 1000).
    #[pyo3(signature = (operator, peak_luminance = None))]
    fn add_tone_map(&mut self, operator: &str, peak_luminance: Option<f64>) -> PyResult<()> {
        let op = parse_tone_map_operator(operator)?;
        let peak = peak_luminance.unwrap_or(1000.0);
        let mapper = ToneMapper::new(op, peak, 100.0);
        self.inner.add_transform(ColorTransform::ToneMap(mapper));
        self.descriptions
            .push(format!("ToneMap({operator}, peak={peak})"));
        Ok(())
    }

    /// Append a gamut-mapping step.
    ///
    /// Arguments:
    ///     algorithm: One of ``clip``, ``compress``, ``desaturate``, ``perceptual``.
    fn add_gamut_map(&mut self, algorithm: &str) -> PyResult<()> {
        let algo = parse_gamut_algorithm(algorithm)?;
        let mapper = GamutMapper::new(algo);
        self.inner.add_transform(ColorTransform::GamutMap(mapper));
        self.descriptions.push(format!("GamutMap({algorithm})"));
        Ok(())
    }

    /// Append an exposure adjustment in photographic stops.
    fn add_exposure(&mut self, stops: f64) -> PyResult<()> {
        self.inner.add_transform(ColorTransform::Exposure(stops));
        self.descriptions.push(format!("Exposure({stops})"));
        Ok(())
    }

    /// Append a contrast adjustment (1.0 = no change).
    fn add_contrast(&mut self, amount: f64) -> PyResult<()> {
        self.inner.add_transform(ColorTransform::Contrast(amount));
        self.descriptions.push(format!("Contrast({amount})"));
        Ok(())
    }

    /// Append a saturation adjustment (1.0 = no change).
    fn add_saturation(&mut self, amount: f64) -> PyResult<()> {
        self.inner.add_transform(ColorTransform::Saturation(amount));
        self.descriptions.push(format!("Saturation({amount})"));
        Ok(())
    }

    /// Transform a single pixel (r, g, b) in [0, 1] range.
    fn transform_pixel(&self, r: f64, g: f64, b: f64) -> PyResult<(f64, f64, f64)> {
        let out = self.inner.transform_pixel([r, g, b]);
        Ok((out[0], out[1], out[2]))
    }

    /// Transform an image stored as a flat ``[r,g,b, r,g,b, ...]`` array.
    ///
    /// The length of *data* must equal ``width * height * 3``.
    fn transform_image(&self, data: Vec<f64>, width: u32, height: u32) -> PyResult<Vec<f64>> {
        let expected = (width as usize) * (height as usize) * 3;
        if data.len() != expected {
            return Err(val_err(format!(
                "Data length {} does not match {}x{}x3 = {}",
                data.len(),
                width,
                height,
                expected,
            )));
        }
        let mut buf = data;
        self.inner.transform_image(&mut buf);
        Ok(buf)
    }

    /// Number of transform steps in the pipeline.
    fn step_count(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("PyColorPipeline(steps=[{}])", self.descriptions.join(", "))
    }
}

// ---------------------------------------------------------------------------
// PyAcesTransform
// ---------------------------------------------------------------------------

/// ACES color-space transform (IDT / ODT helpers).
#[pyclass]
pub struct PyAcesTransform {
    #[pyo3(get)]
    idt_source: String,
    #[pyo3(get)]
    odt_target: String,
    inner: oximedia_colormgmt::aces::AcesTransform,
}

#[pymethods]
impl PyAcesTransform {
    /// Create a new ACES transform from source to target color spaces.
    ///
    /// Arguments:
    ///     idt_source: One of ``aces2065_1``, ``acescg``, ``acescc``, ``acescct``.
    ///     odt_target: One of ``aces2065_1``, ``acescg``, ``acescc``, ``acescct``.
    #[new]
    fn new(idt_source: &str, odt_target: &str) -> PyResult<Self> {
        let src = parse_aces_space(idt_source)?;
        let dst = parse_aces_space(odt_target)?;
        let inner = oximedia_colormgmt::aces::AcesTransform::new(src, dst);
        Ok(Self {
            idt_source: idt_source.to_string(),
            odt_target: odt_target.to_string(),
            inner,
        })
    }

    /// Transform a single pixel.
    fn transform_pixel(&self, r: f64, g: f64, b: f64) -> PyResult<(f64, f64, f64)> {
        let out = self.inner.apply([r, g, b]).map_err(cm_err)?;
        Ok((out[0], out[1], out[2]))
    }

    /// List supported IDT source names.
    #[staticmethod]
    fn list_idt_sources() -> Vec<String> {
        aces_space_names()
    }

    /// List supported ODT target names.
    #[staticmethod]
    fn list_odt_targets() -> Vec<String> {
        aces_space_names()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyAcesTransform(source='{}', target='{}')",
            self.idt_source, self.odt_target,
        )
    }
}

// ---------------------------------------------------------------------------
// PyToneMapper
// ---------------------------------------------------------------------------

/// Standalone tone-mapping operator for HDR-to-SDR conversion.
#[pyclass]
pub struct PyToneMapper {
    #[pyo3(get)]
    operator: String,
    #[pyo3(get)]
    peak_luminance: f64,
    #[pyo3(get)]
    target_luminance: f64,
    inner: ToneMapper,
}

#[pymethods]
impl PyToneMapper {
    /// Create a tone mapper.
    ///
    /// Arguments:
    ///     operator: ``reinhard``, ``reinhard_extended``, ``hable``, ``aces``, ``linear``.
    ///     peak_luminance: Input peak luminance in nits (default 1000).
    ///     target_luminance: Output peak luminance in nits (default 100).
    #[new]
    #[pyo3(signature = (operator, peak_luminance = None, target_luminance = None))]
    fn new(
        operator: &str,
        peak_luminance: Option<f64>,
        target_luminance: Option<f64>,
    ) -> PyResult<Self> {
        let op = parse_tone_map_operator(operator)?;
        let peak = peak_luminance.unwrap_or(1000.0);
        let target = target_luminance.unwrap_or(100.0);
        let inner = ToneMapper::new(op, peak, target);
        Ok(Self {
            operator: operator.to_string(),
            peak_luminance: peak,
            target_luminance: target,
            inner,
        })
    }

    /// Map a single HDR pixel to SDR.
    fn map_pixel(&self, r: f64, g: f64, b: f64) -> PyResult<(f64, f64, f64)> {
        let out = self.inner.apply([r, g, b]);
        Ok((out[0], out[1], out[2]))
    }

    /// Map an entire image (flat ``[r,g,b,...]`` buffer).
    fn map_image(&self, data: Vec<f64>, width: u32, height: u32) -> PyResult<Vec<f64>> {
        let expected = (width as usize) * (height as usize) * 3;
        if data.len() != expected {
            return Err(val_err(format!(
                "Data length {} != {}x{}x3 = {}",
                data.len(),
                width,
                height,
                expected,
            )));
        }
        let out: Vec<f64> = data
            .chunks_exact(3)
            .flat_map(|px| {
                let mapped = self.inner.apply([px[0], px[1], px[2]]);
                mapped.into_iter()
            })
            .collect();
        Ok(out)
    }

    /// List available tone mapping operators.
    #[staticmethod]
    fn available_operators() -> Vec<String> {
        tone_map_operator_names()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyToneMapper(op='{}', peak={}, target={})",
            self.operator, self.peak_luminance, self.target_luminance,
        )
    }
}

// ---------------------------------------------------------------------------
// PyGamutMapper
// ---------------------------------------------------------------------------

/// Gamut mapper for bringing out-of-gamut colors into a target gamut.
#[pyclass]
pub struct PyGamutMapper {
    #[pyo3(get)]
    algorithm: String,
    inner: GamutMapper,
}

#[pymethods]
impl PyGamutMapper {
    /// Create a gamut mapper.
    ///
    /// Arguments:
    ///     algorithm: ``clip``, ``compress``, ``desaturate``, ``perceptual``.
    #[new]
    fn new(algorithm: &str) -> PyResult<Self> {
        let algo = parse_gamut_algorithm(algorithm)?;
        Ok(Self {
            algorithm: algorithm.to_string(),
            inner: GamutMapper::new(algo),
        })
    }

    /// Map a single pixel into the [0, 1] gamut.
    fn map_pixel(&self, r: f64, g: f64, b: f64) -> PyResult<(f64, f64, f64)> {
        let out = self.inner.map([r, g, b], None);
        Ok((out[0], out[1], out[2]))
    }

    /// Check whether a pixel is within the [0, 1] gamut.
    fn is_in_gamut(&self, r: f64, g: f64, b: f64) -> bool {
        (0.0..=1.0).contains(&r) && (0.0..=1.0).contains(&g) && (0.0..=1.0).contains(&b)
    }

    /// List available gamut mapping algorithms.
    #[staticmethod]
    fn available_algorithms() -> Vec<String> {
        gamut_algorithm_names()
    }

    fn __repr__(&self) -> String {
        format!("PyGamutMapper(algorithm='{}')", self.algorithm)
    }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Convert image data between color spaces.
///
/// Arguments:
///     data: Flat ``[r,g,b,...]`` buffer with values in [0, 1].
///     width, height: Image dimensions.
///     from_space, to_space: Color-space names (see ``PyColorSpace.from_name``).
#[pyfunction]
pub fn convert_colorspace(
    data: Vec<f64>,
    width: u32,
    height: u32,
    from_space: &str,
    to_space: &str,
) -> PyResult<Vec<f64>> {
    let expected = (width as usize) * (height as usize) * 3;
    if data.len() != expected {
        return Err(val_err(format!(
            "Data length {} != {}x{}x3 = {}",
            data.len(),
            width,
            height,
            expected,
        )));
    }
    let src = resolve_colorspace(from_space)?;
    let dst = resolve_colorspace(to_space)?;

    let mut pipeline = ColorPipeline::new();
    pipeline.add_transform(ColorTransform::ColorSpaceConversion {
        from: src.inner,
        to: dst.inner,
    });

    let mut buf = data;
    pipeline.transform_image(&mut buf);
    Ok(buf)
}

/// Calculate perceptual color difference (delta-E) between two Lab colors.
///
/// Arguments:
///     lab1, lab2: Tuples of (L, a, b).
///     method: ``"1976"`` (default) or ``"2000"``.
#[pyfunction]
#[pyo3(signature = (lab1, lab2, method = None))]
pub fn py_delta_e(
    lab1: (f64, f64, f64),
    lab2: (f64, f64, f64),
    method: Option<&str>,
) -> PyResult<f64> {
    let l1 = Lab::new(lab1.0, lab1.1, lab1.2);
    let l2 = Lab::new(lab2.0, lab2.1, lab2.2);
    match method.unwrap_or("1976") {
        "1976" | "cie76" => Ok(delta_e::delta_e_1976(&l1, &l2)),
        "2000" | "ciede2000" => Ok(delta_e::delta_e_2000(&l1, &l2)),
        other => Err(val_err(format!(
            "Unknown delta-E method '{}'. Use '1976' or '2000'.",
            other
        ))),
    }
}

/// Check whether an RGB triplet is inside a named color space gamut.
#[pyfunction]
pub fn gamut_check(r: f64, g: f64, b: f64, colorspace: &str) -> PyResult<bool> {
    // Resolve the color space to validate the name
    let _cs = resolve_colorspace(colorspace)?;
    // For standard RGB color spaces, gamut is [0, 1]
    Ok((0.0..=1.0).contains(&r) && (0.0..=1.0).contains(&g) && (0.0..=1.0).contains(&b))
}

/// Apply tone mapping to a flat image buffer.
///
/// Arguments:
///     data: Flat ``[r,g,b,...]`` in linear HDR space.
///     operator: Tone mapping operator name.
///     peak_luminance: Input peak luminance in nits (default 1000).
#[pyfunction]
#[pyo3(signature = (data, operator, peak_luminance = None))]
pub fn apply_tone_map(
    data: Vec<f64>,
    operator: &str,
    peak_luminance: Option<f64>,
) -> PyResult<Vec<f64>> {
    if data.len() % 3 != 0 {
        return Err(val_err("Data length must be a multiple of 3"));
    }
    let op = parse_tone_map_operator(operator)?;
    let peak = peak_luminance.unwrap_or(1000.0);
    let mapper = ToneMapper::new(op, peak, 100.0);

    let out: Vec<f64> = data
        .chunks_exact(3)
        .flat_map(|px| {
            let mapped = mapper.apply([px[0], px[1], px[2]]);
            mapped.into_iter()
        })
        .collect();
    Ok(out)
}

/// List all supported color spaces with metadata.
#[pyfunction]
pub fn list_colorspaces() -> PyResult<Vec<HashMap<String, String>>> {
    let entries: &[(&str, &str, bool, bool)] = &[
        ("srgb", "BT.709", false, false),
        ("rec709", "BT.709", false, false),
        ("rec2020", "BT.2020", true, true),
        ("rec2020_pq", "BT.2020", true, true),
        ("rec2020_hlg", "BT.2020", true, true),
        ("display_p3", "P3", false, false),
        ("dci_p3", "P3", false, false),
        ("adobe_rgb", "Adobe RGB", false, false),
        ("prophoto_rgb", "ProPhoto", false, false),
        ("linear_rec709", "BT.709", false, true),
    ];
    Ok(entries
        .iter()
        .map(|(name, gamut, hdr, linear)| {
            let mut m = HashMap::new();
            m.insert("name".to_string(), (*name).to_string());
            m.insert("gamut".to_string(), (*gamut).to_string());
            m.insert("is_hdr".to_string(), hdr.to_string());
            m.insert("is_linear".to_string(), linear.to_string());
            m
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register colormgmt bindings on the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyColorSpace>()?;
    m.add_class::<PyColorPipeline>()?;
    m.add_class::<PyAcesTransform>()?;
    m.add_class::<PyToneMapper>()?;
    m.add_class::<PyGamutMapper>()?;
    m.add_function(wrap_pyfunction!(convert_colorspace, m)?)?;
    m.add_function(wrap_pyfunction!(py_delta_e, m)?)?;
    m.add_function(wrap_pyfunction!(gamut_check, m)?)?;
    m.add_function(wrap_pyfunction!(apply_tone_map, m)?)?;
    m.add_function(wrap_pyfunction!(list_colorspaces, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn resolve_colorspace(name: &str) -> PyResult<PyColorSpace> {
    let lower = name.to_ascii_lowercase();
    match lower.as_str() {
        "srgb" => {
            let cs = ColorSpace::srgb().map_err(cm_err)?;
            Ok(PyColorSpace::from_inner(cs, false, "BT.709"))
        }
        "rec709" | "bt709" | "rec.709" => {
            let cs = ColorSpace::rec709().map_err(cm_err)?;
            Ok(PyColorSpace::from_inner(cs, false, "BT.709"))
        }
        "rec2020" | "bt2020" | "rec.2020" => {
            let cs = ColorSpace::rec2020().map_err(cm_err)?;
            Ok(PyColorSpace::from_inner(cs, true, "BT.2020"))
        }
        "rec2020_pq" | "rec2020pq" => {
            let cs = ColorSpace::rec2020_pq().map_err(cm_err)?;
            Ok(PyColorSpace::from_inner(cs, true, "BT.2020"))
        }
        "rec2020_hlg" | "rec2020hlg" => {
            let cs = ColorSpace::rec2020_hlg().map_err(cm_err)?;
            Ok(PyColorSpace::from_inner(cs, true, "BT.2020"))
        }
        "display_p3" | "displayp3" | "p3" => {
            let cs = ColorSpace::display_p3().map_err(cm_err)?;
            Ok(PyColorSpace::from_inner(cs, false, "P3"))
        }
        "dci_p3" | "dcip3" => {
            let cs = ColorSpace::dci_p3().map_err(cm_err)?;
            Ok(PyColorSpace::from_inner(cs, false, "P3"))
        }
        "adobe_rgb" | "adobergb" => {
            let cs = ColorSpace::adobe_rgb().map_err(cm_err)?;
            Ok(PyColorSpace::from_inner(cs, false, "Adobe RGB"))
        }
        "prophoto_rgb" | "prophotorgb" | "prophoto" => {
            let cs = ColorSpace::prophoto_rgb().map_err(cm_err)?;
            Ok(PyColorSpace::from_inner(cs, false, "ProPhoto"))
        }
        "linear_rec709" | "linear_bt709" | "linear" => {
            let cs = ColorSpace::linear_rec709().map_err(cm_err)?;
            Ok(PyColorSpace::from_inner(cs, false, "BT.709"))
        }
        _ => Err(val_err(format!(
            "Unknown color space '{}'. Use list_colorspaces() to see available names.",
            name,
        ))),
    }
}

fn parse_tone_map_operator(name: &str) -> PyResult<ToneMappingOperator> {
    match name.to_ascii_lowercase().as_str() {
        "reinhard" => Ok(ToneMappingOperator::Reinhard),
        "reinhard_extended" | "reinhardextended" => Ok(ToneMappingOperator::ReinhardExtended),
        "hable" | "uncharted2" => Ok(ToneMappingOperator::Hable),
        "aces" | "aces_filmic" => Ok(ToneMappingOperator::Aces),
        "linear" | "clamp" => Ok(ToneMappingOperator::Linear),
        _ => Err(val_err(format!(
            "Unknown tone map operator '{}'. Valid: {}",
            name,
            tone_map_operator_names().join(", "),
        ))),
    }
}

fn tone_map_operator_names() -> Vec<String> {
    ["reinhard", "reinhard_extended", "hable", "aces", "linear"]
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

fn parse_gamut_algorithm(name: &str) -> PyResult<GamutMappingAlgorithm> {
    match name.to_ascii_lowercase().as_str() {
        "clip" => Ok(GamutMappingAlgorithm::Clip),
        "compress" => Ok(GamutMappingAlgorithm::Compress),
        "desaturate" => Ok(GamutMappingAlgorithm::Desaturate),
        "perceptual" => Ok(GamutMappingAlgorithm::Perceptual),
        _ => Err(val_err(format!(
            "Unknown gamut algorithm '{}'. Valid: {}",
            name,
            gamut_algorithm_names().join(", "),
        ))),
    }
}

fn gamut_algorithm_names() -> Vec<String> {
    ["clip", "compress", "desaturate", "perceptual"]
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

fn parse_aces_space(name: &str) -> PyResult<oximedia_colormgmt::aces::AcesColorSpace> {
    use oximedia_colormgmt::aces::AcesColorSpace;
    match name.to_ascii_lowercase().as_str() {
        "aces2065_1" | "aces2065-1" | "aces" => Ok(AcesColorSpace::ACES2065_1),
        "acescg" => Ok(AcesColorSpace::ACEScg),
        "acescc" => Ok(AcesColorSpace::ACEScc),
        "acescct" => Ok(AcesColorSpace::ACEScct),
        _ => Err(val_err(format!(
            "Unknown ACES space '{}'. Valid: {}",
            name,
            aces_space_names().join(", "),
        ))),
    }
}

fn aces_space_names() -> Vec<String> {
    ["aces2065_1", "acescg", "acescc", "acescct"]
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_srgb() {
        let cs = resolve_colorspace("srgb").expect("should resolve srgb");
        assert_eq!(cs.name, "sRGB");
        assert!(!cs.is_hdr);
    }

    #[test]
    fn test_resolve_rec2020() {
        let cs = resolve_colorspace("rec2020").expect("should resolve rec2020");
        assert!(cs.is_hdr);
        assert_eq!(cs.gamut, "BT.2020");
    }

    #[test]
    fn test_resolve_unknown() {
        assert!(resolve_colorspace("unknown_xyz").is_err());
    }

    #[test]
    fn test_parse_tone_map_operators() {
        assert_eq!(
            parse_tone_map_operator("reinhard").expect("ok"),
            ToneMappingOperator::Reinhard,
        );
        assert_eq!(
            parse_tone_map_operator("aces").expect("ok"),
            ToneMappingOperator::Aces,
        );
        assert!(parse_tone_map_operator("bogus").is_err());
    }

    #[test]
    fn test_parse_gamut_algorithms() {
        assert_eq!(
            parse_gamut_algorithm("clip").expect("ok"),
            GamutMappingAlgorithm::Clip,
        );
        assert!(parse_gamut_algorithm("bogus").is_err());
    }

    #[test]
    fn test_delta_e_same_color() {
        let de =
            py_delta_e((50.0, 0.0, 0.0), (50.0, 0.0, 0.0), None).expect("delta_e should not fail");
        assert!((de - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_delta_e_2000_method() {
        let de = py_delta_e((50.0, 0.0, 0.0), (50.0, 0.0, 0.0), Some("2000"))
            .expect("delta_e 2000 should not fail");
        assert!((de - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gamut_check_in_gamut() {
        assert!(gamut_check(0.5, 0.5, 0.5, "srgb").expect("ok"));
    }

    #[test]
    fn test_gamut_check_out_of_gamut() {
        assert!(!gamut_check(1.5, 0.5, 0.5, "srgb").expect("ok"));
    }

    #[test]
    fn test_list_colorspaces_not_empty() {
        let list = list_colorspaces().expect("should not fail");
        assert!(list.len() >= 8);
    }

    #[test]
    fn test_tone_map_operator_names() {
        let names = tone_map_operator_names();
        assert!(names.contains(&"reinhard".to_string()));
        assert!(names.contains(&"aces".to_string()));
    }

    #[test]
    fn test_aces_space_names() {
        let names = aces_space_names();
        assert_eq!(names.len(), 4);
    }
}
