//! Python bindings for Dolby Vision RPU metadata.
//!
//! Provides `PyDolbyVisionAnalyzer`, `PyDvMetadata`, `PyDvProfile`,
//! and standalone functions for analyzing and converting Dolby Vision metadata.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn profile_number(p: oximedia_dolbyvision::Profile) -> u8 {
    match p {
        oximedia_dolbyvision::Profile::Profile5 => 5,
        oximedia_dolbyvision::Profile::Profile7 => 7,
        oximedia_dolbyvision::Profile::Profile8 => 8,
        oximedia_dolbyvision::Profile::Profile8_1 => 81,
        oximedia_dolbyvision::Profile::Profile8_4 => 84,
    }
}

fn profile_description(p: oximedia_dolbyvision::Profile) -> &'static str {
    match p {
        oximedia_dolbyvision::Profile::Profile5 => "IPT-PQ, backward compatible with HDR10",
        oximedia_dolbyvision::Profile::Profile7 => "MEL + BL, single track, full enhancement",
        oximedia_dolbyvision::Profile::Profile8 => "BL only, backward compatible with HDR10",
        oximedia_dolbyvision::Profile::Profile8_1 => "Low-latency variant of Profile 8",
        oximedia_dolbyvision::Profile::Profile8_4 => "HLG-based, backward compatible with HLG",
    }
}

fn parse_profile(value: u8) -> PyResult<oximedia_dolbyvision::Profile> {
    oximedia_dolbyvision::Profile::from_u8(value).ok_or_else(|| {
        PyValueError::new_err(format!(
            "Unknown Dolby Vision profile: {value}. Supported: 5, 7, 8, 81, 84"
        ))
    })
}

// ---------------------------------------------------------------------------
// PyDvProfile
// ---------------------------------------------------------------------------

/// Dolby Vision profile information.
#[pyclass]
#[derive(Clone)]
pub struct PyDvProfile {
    /// Profile number.
    #[pyo3(get)]
    pub number: u8,
    /// Human-readable description.
    #[pyo3(get)]
    pub description: String,
    /// Whether this profile is backward compatible.
    #[pyo3(get)]
    pub backward_compatible: bool,
    /// Whether this profile uses MEL.
    #[pyo3(get)]
    pub has_mel: bool,
    /// Whether this profile is HLG-based.
    #[pyo3(get)]
    pub is_hlg: bool,
    /// Whether this profile is low-latency.
    #[pyo3(get)]
    pub is_low_latency: bool,
}

#[pymethods]
impl PyDvProfile {
    /// Create profile info from a profile number.
    #[new]
    fn new(number: u8) -> PyResult<Self> {
        let p = parse_profile(number)?;
        Ok(Self {
            number,
            description: profile_description(p).to_string(),
            backward_compatible: p.is_backward_compatible(),
            has_mel: p.has_mel(),
            is_hlg: p.is_hlg(),
            is_low_latency: p.is_low_latency(),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PyDvProfile(number={}, description='{}')",
            self.number, self.description
        )
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("number".to_string(), self.number.to_string());
        m.insert("description".to_string(), self.description.clone());
        m.insert(
            "backward_compatible".to_string(),
            self.backward_compatible.to_string(),
        );
        m.insert("has_mel".to_string(), self.has_mel.to_string());
        m.insert("is_hlg".to_string(), self.is_hlg.to_string());
        m.insert(
            "is_low_latency".to_string(),
            self.is_low_latency.to_string(),
        );
        m
    }
}

// ---------------------------------------------------------------------------
// PyDvMetadata
// ---------------------------------------------------------------------------

/// Dolby Vision RPU metadata summary.
#[pyclass]
#[derive(Clone)]
pub struct PyDvMetadata {
    /// Profile number.
    #[pyo3(get)]
    pub profile: u8,
    /// RPU format version.
    #[pyo3(get)]
    pub rpu_format: u16,
    /// Whether Level 1 metadata is present.
    #[pyo3(get)]
    pub has_level1: bool,
    /// Whether Level 2 metadata is present.
    #[pyo3(get)]
    pub has_level2: bool,
    /// Whether Level 5 metadata is present.
    #[pyo3(get)]
    pub has_level5: bool,
    /// Whether Level 6 metadata is present.
    #[pyo3(get)]
    pub has_level6: bool,
    /// Whether VDR DM data is present.
    #[pyo3(get)]
    pub has_vdr_dm: bool,
}

#[pymethods]
impl PyDvMetadata {
    fn __repr__(&self) -> String {
        format!(
            "PyDvMetadata(profile={}, rpu_format={}, levels=[{}{}{}{}])",
            self.profile,
            self.rpu_format,
            if self.has_level1 { "L1 " } else { "" },
            if self.has_level2 { "L2 " } else { "" },
            if self.has_level5 { "L5 " } else { "" },
            if self.has_level6 { "L6" } else { "" },
        )
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("profile".to_string(), self.profile.to_string());
        m.insert("rpu_format".to_string(), self.rpu_format.to_string());
        m.insert("has_level1".to_string(), self.has_level1.to_string());
        m.insert("has_level2".to_string(), self.has_level2.to_string());
        m.insert("has_level5".to_string(), self.has_level5.to_string());
        m.insert("has_level6".to_string(), self.has_level6.to_string());
        m.insert("has_vdr_dm".to_string(), self.has_vdr_dm.to_string());
        m
    }
}

// ---------------------------------------------------------------------------
// PyDolbyVisionAnalyzer
// ---------------------------------------------------------------------------

/// Dolby Vision metadata analyzer.
#[pyclass]
pub struct PyDolbyVisionAnalyzer {
    profile: oximedia_dolbyvision::Profile,
    rpu: oximedia_dolbyvision::DolbyVisionRpu,
}

#[pymethods]
impl PyDolbyVisionAnalyzer {
    /// Create a new analyzer for the given profile.
    #[new]
    #[pyo3(signature = (profile=8))]
    fn new(profile: u8) -> PyResult<Self> {
        let p = parse_profile(profile)?;
        let rpu = oximedia_dolbyvision::DolbyVisionRpu::new(p);
        Ok(Self { profile: p, rpu })
    }

    /// Get the RPU metadata summary.
    fn get_metadata(&self) -> PyDvMetadata {
        PyDvMetadata {
            profile: profile_number(self.profile),
            rpu_format: self.rpu.header.rpu_format,
            has_level1: self.rpu.level1.is_some(),
            has_level2: self.rpu.level2.is_some(),
            has_level5: self.rpu.level5.is_some(),
            has_level6: self.rpu.level6.is_some(),
            has_vdr_dm: self.rpu.vdr_dm_data.is_some(),
        }
    }

    /// Get profile information.
    fn get_profile(&self) -> PyDvProfile {
        PyDvProfile {
            number: profile_number(self.profile),
            description: profile_description(self.profile).to_string(),
            backward_compatible: self.profile.is_backward_compatible(),
            has_mel: self.profile.has_mel(),
            is_hlg: self.profile.is_hlg(),
            is_low_latency: self.profile.is_low_latency(),
        }
    }

    /// Validate the RPU structure.
    fn validate(&self) -> PyResult<bool> {
        match self.rpu.validate() {
            Ok(()) => Ok(true),
            Err(e) => Err(PyValueError::new_err(format!("Validation failed: {e}"))),
        }
    }

    /// List all supported profiles.
    #[staticmethod]
    fn supported_profiles() -> Vec<PyDvProfile> {
        let profiles = [5u8, 7, 8, 81, 84];
        profiles
            .iter()
            .filter_map(|&n| {
                let p = oximedia_dolbyvision::Profile::from_u8(n)?;
                Some(PyDvProfile {
                    number: n,
                    description: profile_description(p).to_string(),
                    backward_compatible: p.is_backward_compatible(),
                    has_mel: p.has_mel(),
                    is_hlg: p.is_hlg(),
                    is_low_latency: p.is_low_latency(),
                })
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyDolbyVisionAnalyzer(profile={})",
            profile_number(self.profile)
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Analyze Dolby Vision profile properties.
#[pyfunction]
pub fn analyze_dv(profile: u8) -> PyResult<HashMap<String, String>> {
    let p = parse_profile(profile)?;
    let rpu = oximedia_dolbyvision::DolbyVisionRpu::new(p);
    let mut m = HashMap::new();
    m.insert("profile".to_string(), profile.to_string());
    m.insert(
        "description".to_string(),
        profile_description(p).to_string(),
    );
    m.insert(
        "backward_compatible".to_string(),
        p.is_backward_compatible().to_string(),
    );
    m.insert("has_mel".to_string(), p.has_mel().to_string());
    m.insert("is_hlg".to_string(), p.is_hlg().to_string());
    m.insert("is_low_latency".to_string(), p.is_low_latency().to_string());
    m.insert("rpu_format".to_string(), rpu.header.rpu_format.to_string());
    m.insert("valid".to_string(), rpu.validate().is_ok().to_string());
    Ok(m)
}

/// Convert Dolby Vision metadata between profiles.
#[pyfunction]
pub fn convert_dv_metadata(from_profile: u8, to_profile: u8) -> PyResult<PyDvMetadata> {
    let _source = parse_profile(from_profile)?;
    let target = parse_profile(to_profile)?;
    let rpu = oximedia_dolbyvision::DolbyVisionRpu::new(target);
    rpu.validate()
        .map_err(|e| PyValueError::new_err(format!("Target profile validation failed: {e}")))?;

    Ok(PyDvMetadata {
        profile: to_profile,
        rpu_format: rpu.header.rpu_format,
        has_level1: rpu.level1.is_some(),
        has_level2: rpu.level2.is_some(),
        has_level5: rpu.level5.is_some(),
        has_level6: rpu.level6.is_some(),
        has_vdr_dm: rpu.vdr_dm_data.is_some(),
    })
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register Dolby Vision bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDvProfile>()?;
    m.add_class::<PyDvMetadata>()?;
    m.add_class::<PyDolbyVisionAnalyzer>()?;
    m.add_function(wrap_pyfunction!(analyze_dv, m)?)?;
    m.add_function(wrap_pyfunction!(convert_dv_metadata, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_profile_valid() {
        assert!(parse_profile(5).is_ok());
        assert!(parse_profile(8).is_ok());
        assert!(parse_profile(84).is_ok());
    }

    #[test]
    fn test_parse_profile_invalid() {
        assert!(parse_profile(99).is_err());
        assert!(parse_profile(0).is_err());
    }

    #[test]
    fn test_profile_description_nonempty() {
        let desc = profile_description(oximedia_dolbyvision::Profile::Profile8);
        assert!(!desc.is_empty());
    }

    #[test]
    fn test_dv_metadata_repr() {
        let meta = PyDvMetadata {
            profile: 8,
            rpu_format: 0,
            has_level1: true,
            has_level2: false,
            has_level5: true,
            has_level6: false,
            has_vdr_dm: false,
        };
        let repr = meta.__repr__();
        assert!(repr.contains("profile=8"));
    }

    #[test]
    fn test_supported_profiles_count() {
        let profiles = PyDolbyVisionAnalyzer::supported_profiles();
        assert_eq!(profiles.len(), 5);
    }
}
