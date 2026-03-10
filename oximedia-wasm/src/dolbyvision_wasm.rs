//! WebAssembly bindings for Dolby Vision metadata utilities.
//!
//! Provides browser-side Dolby Vision metadata parsing, profile enumeration,
//! and validation.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Parse Dolby Vision metadata for the given profile and return JSON summary.
///
/// # Arguments
/// * `profile` - Profile number: 5, 7, 8, 81, or 84.
///
/// # Returns
/// JSON object with metadata summary.
///
/// # Errors
/// Returns an error if the profile is not recognized.
#[wasm_bindgen]
pub fn wasm_parse_dv_metadata(profile: u8) -> Result<String, JsValue> {
    let p = oximedia_dolbyvision::Profile::from_u8(profile).ok_or_else(|| {
        crate::utils::js_err(&format!(
            "Unknown profile: {profile}. Supported: 5, 7, 8, 81, 84"
        ))
    })?;

    let rpu = oximedia_dolbyvision::DolbyVisionRpu::new(p);

    let description = match p {
        oximedia_dolbyvision::Profile::Profile5 => "IPT-PQ, backward compatible with HDR10",
        oximedia_dolbyvision::Profile::Profile7 => "MEL + BL, single track, full enhancement",
        oximedia_dolbyvision::Profile::Profile8 => "BL only, backward compatible with HDR10",
        oximedia_dolbyvision::Profile::Profile8_1 => "Low-latency variant of Profile 8",
        oximedia_dolbyvision::Profile::Profile8_4 => "HLG-based, backward compatible with HLG",
    };

    Ok(format!(
        "{{\"profile\":{profile},\"description\":\"{description}\",\
         \"backward_compatible\":{},\"has_mel\":{},\"is_hlg\":{},\"is_low_latency\":{},\
         \"rpu_format\":{},\"has_level1\":{},\"has_level2\":{},\"has_level5\":{},\
         \"has_level6\":{},\"has_vdr_dm\":{}}}",
        p.is_backward_compatible(),
        p.has_mel(),
        p.is_hlg(),
        p.is_low_latency(),
        rpu.header.rpu_format,
        rpu.level1.is_some(),
        rpu.level2.is_some(),
        rpu.level5.is_some(),
        rpu.level6.is_some(),
        rpu.vdr_dm_data.is_some(),
    ))
}

/// Return a JSON array of all supported Dolby Vision profiles.
#[wasm_bindgen]
pub fn wasm_dv_profiles() -> String {
    let profiles = [5u8, 7, 8, 81, 84];
    let entries: Vec<String> = profiles
        .iter()
        .filter_map(|&n| {
            let p = oximedia_dolbyvision::Profile::from_u8(n)?;
            let desc = match p {
                oximedia_dolbyvision::Profile::Profile5 => "IPT-PQ, backward compatible with HDR10",
                oximedia_dolbyvision::Profile::Profile7 => "MEL + BL, single track",
                oximedia_dolbyvision::Profile::Profile8 => {
                    "BL only, backward compatible with HDR10"
                }
                oximedia_dolbyvision::Profile::Profile8_1 => "Low-latency variant",
                oximedia_dolbyvision::Profile::Profile8_4 => "HLG-based, backward compatible",
            };
            Some(format!(
                "{{\"profile\":{n},\"description\":\"{desc}\",\
                 \"backward_compatible\":{},\"has_mel\":{},\"is_hlg\":{},\"is_low_latency\":{}}}",
                p.is_backward_compatible(),
                p.has_mel(),
                p.is_hlg(),
                p.is_low_latency()
            ))
        })
        .collect();
    format!("[{}]", entries.join(","))
}

/// Validate a Dolby Vision profile configuration.
///
/// # Arguments
/// * `profile` - Profile number to validate.
///
/// # Returns
/// JSON object with validation results.
///
/// # Errors
/// Returns an error if the profile is unknown.
#[wasm_bindgen]
pub fn wasm_validate_dv(profile: u8) -> Result<String, JsValue> {
    let p = oximedia_dolbyvision::Profile::from_u8(profile).ok_or_else(|| {
        crate::utils::js_err(&format!(
            "Unknown profile: {profile}. Supported: 5, 7, 8, 81, 84"
        ))
    })?;

    let rpu = oximedia_dolbyvision::DolbyVisionRpu::new(p);
    let valid = rpu.validate().is_ok();
    let error_msg = match rpu.validate() {
        Ok(()) => String::new(),
        Err(e) => e.to_string(),
    };

    Ok(format!(
        "{{\"profile\":{profile},\"valid\":{valid},\"error\":\"{error_msg}\"}}",
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dv_metadata_valid() {
        let result = wasm_parse_dv_metadata(8);
        assert!(result.is_ok());
        let json = result.expect("parse");
        assert!(json.contains("\"profile\":8"));
        assert!(json.contains("backward_compatible"));
    }

    #[test]
    fn test_parse_dv_metadata_invalid() {
        let result = wasm_parse_dv_metadata(99);
        assert!(result.is_err());
    }

    #[test]
    fn test_dv_profiles() {
        let json = wasm_dv_profiles();
        assert!(json.contains("\"profile\":5"));
        assert!(json.contains("\"profile\":84"));
    }

    #[test]
    fn test_validate_dv_valid() {
        let result = wasm_validate_dv(8);
        assert!(result.is_ok());
        let json = result.expect("validate");
        assert!(json.contains("\"valid\":true"));
    }

    #[test]
    fn test_validate_dv_invalid_profile() {
        let result = wasm_validate_dv(99);
        assert!(result.is_err());
    }
}
