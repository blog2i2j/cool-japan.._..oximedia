//! WebAssembly bindings for `oximedia-timecode`.
//!
//! All functions are synchronous and return JSON strings for easy consumption
//! from JavaScript.

use wasm_bindgen::prelude::*;

use oximedia_timecode::{FrameRate, Timecode, TimecodeError};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Map a `TimecodeError` to a `JsValue` error string.
fn tc_js_err(e: TimecodeError) -> JsValue {
    crate::utils::js_err(&format!("{e}"))
}

/// Resolve (fps, drop_frame) to a `FrameRate` enum variant.
fn resolve_frame_rate(fps: f64, drop_frame: bool) -> Result<FrameRate, JsValue> {
    let rounded = (fps * 1000.0).round() / 1000.0;

    match rounded {
        v if (v - 23.976).abs() < 0.01 => Ok(FrameRate::Fps23976),
        v if (v - 24.0).abs() < 0.01 => Ok(FrameRate::Fps24),
        v if (v - 25.0).abs() < 0.01 => Ok(FrameRate::Fps25),
        v if (v - 29.97).abs() < 0.01 => {
            if drop_frame {
                Ok(FrameRate::Fps2997DF)
            } else {
                Ok(FrameRate::Fps2997NDF)
            }
        }
        v if (v - 30.0).abs() < 0.01 => Ok(FrameRate::Fps30),
        v if (v - 50.0).abs() < 0.01 => Ok(FrameRate::Fps50),
        v if (v - 59.94).abs() < 0.01 => Ok(FrameRate::Fps5994),
        v if (v - 60.0).abs() < 0.01 => Ok(FrameRate::Fps60),
        _ => Err(crate::utils::js_err(&format!(
            "Unsupported frame rate: {fps}. Supported: 23.976, 24, 25, 29.97, 30, 50, 59.94, 60"
        ))),
    }
}

/// Parse a timecode string in "HH:MM:SS:FF" or "HH:MM:SS;FF" format.
fn parse_tc_str(tc_str: &str, fr: FrameRate) -> Result<Timecode, JsValue> {
    let cleaned = tc_str.trim();
    let parts: Vec<&str> = cleaned.split([':', ';']).collect();
    if parts.len() != 4 {
        return Err(crate::utils::js_err(&format!(
            "Invalid timecode: '{tc_str}'. Expected HH:MM:SS:FF or HH:MM:SS;FF"
        )));
    }

    let hours: u8 = parts[0]
        .parse()
        .map_err(|_| crate::utils::js_err(&format!("Invalid hours: '{}'", parts[0])))?;
    let minutes: u8 = parts[1]
        .parse()
        .map_err(|_| crate::utils::js_err(&format!("Invalid minutes: '{}'", parts[1])))?;
    let seconds: u8 = parts[2]
        .parse()
        .map_err(|_| crate::utils::js_err(&format!("Invalid seconds: '{}'", parts[2])))?;
    let frames: u8 = parts[3]
        .parse()
        .map_err(|_| crate::utils::js_err(&format!("Invalid frames: '{}'", parts[3])))?;

    // If ';' separator is present and frame rate is NDF 29.97, promote to DF.
    let effective_rate = if cleaned.contains(';') && !fr.is_drop_frame() {
        match fr {
            FrameRate::Fps2997NDF => FrameRate::Fps2997DF,
            other => other,
        }
    } else {
        fr
    };

    Timecode::new(hours, minutes, seconds, frames, effective_rate).map_err(tc_js_err)
}

/// Serialize a `Timecode` and its `FrameRate` to a JSON string.
fn tc_to_json(tc: &Timecode, fr: FrameRate) -> String {
    let total_frames = tc.to_frames();
    let (num, den) = fr.as_rational();
    let seconds = if num > 0 {
        total_frames as f64 * den as f64 / num as f64
    } else {
        0.0
    };

    format!(
        r#"{{"hours":{},"minutes":{},"seconds":{},"frames":{},"total_frames":{},"is_drop_frame":{},"timecode":"{}","seconds_value":{:.6}}}"#,
        tc.hours,
        tc.minutes,
        tc.seconds,
        tc.frames,
        total_frames,
        tc.frame_rate.drop_frame,
        tc,
        seconds,
    )
}

// ---------------------------------------------------------------------------
// Public WASM API
// ---------------------------------------------------------------------------

/// Parse a timecode string and return a JSON object with all fields.
///
/// # Returns
/// JSON: `{"hours":1,"minutes":2,"seconds":3,"frames":4,"total_frames":93829,
///          "is_drop_frame":false,"timecode":"01:02:03:04","seconds_value":3723.16}`
#[wasm_bindgen]
pub fn wasm_parse_timecode(tc_str: &str, fps: f64, drop_frame: bool) -> Result<String, JsValue> {
    let fr = resolve_frame_rate(fps, drop_frame)?;
    let tc = parse_tc_str(tc_str, fr)?;
    Ok(tc_to_json(&tc, fr))
}

/// Convert a frame count to a timecode JSON object.
#[wasm_bindgen]
pub fn wasm_frames_to_timecode(frames: u64, fps: f64, drop_frame: bool) -> Result<String, JsValue> {
    let fr = resolve_frame_rate(fps, drop_frame)?;
    let tc = Timecode::from_frames(frames, fr).map_err(tc_js_err)?;
    Ok(tc_to_json(&tc, fr))
}

/// Add two timecodes (by summing their frame counts).
#[wasm_bindgen]
pub fn wasm_timecode_add(
    tc1: &str,
    tc2: &str,
    fps: f64,
    drop_frame: bool,
) -> Result<String, JsValue> {
    let fr = resolve_frame_rate(fps, drop_frame)?;
    let a = parse_tc_str(tc1, fr)?;
    let b = parse_tc_str(tc2, fr)?;
    let total = a.to_frames() + b.to_frames();
    let result = Timecode::from_frames(total, fr).map_err(tc_js_err)?;
    Ok(tc_to_json(&result, fr))
}

/// Subtract tc2 from tc1 (by their frame counts).
#[wasm_bindgen]
pub fn wasm_timecode_subtract(
    tc1: &str,
    tc2: &str,
    fps: f64,
    drop_frame: bool,
) -> Result<String, JsValue> {
    let fr = resolve_frame_rate(fps, drop_frame)?;
    let a = parse_tc_str(tc1, fr)?;
    let b = parse_tc_str(tc2, fr)?;
    let af = a.to_frames();
    let bf = b.to_frames();
    if bf > af {
        return Err(crate::utils::js_err(
            "Cannot subtract: result would be negative",
        ));
    }
    let result = Timecode::from_frames(af - bf, fr).map_err(tc_js_err)?;
    Ok(tc_to_json(&result, fr))
}

/// Convert a timecode string to seconds.
#[wasm_bindgen]
pub fn wasm_timecode_to_seconds(tc_str: &str, fps: f64, drop_frame: bool) -> Result<f64, JsValue> {
    let fr = resolve_frame_rate(fps, drop_frame)?;
    let tc = parse_tc_str(tc_str, fr)?;
    let total_frames = tc.to_frames();
    let (num, den) = fr.as_rational();
    if num == 0 {
        return Ok(0.0);
    }
    Ok(total_frames as f64 * den as f64 / num as f64)
}

/// Convert a seconds value to a timecode JSON object.
#[wasm_bindgen]
pub fn wasm_seconds_to_timecode(
    seconds: f64,
    fps: f64,
    drop_frame: bool,
) -> Result<String, JsValue> {
    let fr = resolve_frame_rate(fps, drop_frame)?;
    let fps_float = fr.as_float();
    let total_frames = (seconds * fps_float).round() as u64;
    let tc = Timecode::from_frames(total_frames, fr).map_err(tc_js_err)?;
    Ok(tc_to_json(&tc, fr))
}

/// Convert a timecode string to its absolute frame count.
#[wasm_bindgen]
pub fn wasm_timecode_to_frames(tc_str: &str, fps: f64, drop_frame: bool) -> Result<u64, JsValue> {
    let fr = resolve_frame_rate(fps, drop_frame)?;
    let tc = parse_tc_str(tc_str, fr)?;
    Ok(tc.to_frames())
}

/// List all supported frame rates as a JSON array.
///
/// Each element: `{"fps":25.0,"is_drop_frame":false,"name":"25 fps (PAL)","rational":[25,1]}`
#[wasm_bindgen]
pub fn wasm_list_frame_rates() -> Result<String, JsValue> {
    let rates = [
        FrameRate::Fps23976,
        FrameRate::Fps24,
        FrameRate::Fps25,
        FrameRate::Fps2997DF,
        FrameRate::Fps2997NDF,
        FrameRate::Fps30,
        FrameRate::Fps50,
        FrameRate::Fps5994,
        FrameRate::Fps60,
    ];

    let entries: Vec<String> = rates
        .iter()
        .map(|fr| {
            let (num, den) = fr.as_rational();
            let name = match fr {
                FrameRate::Fps23976 => "23.976 fps",
                FrameRate::Fps24 => "24 fps",
                FrameRate::Fps25 => "25 fps (PAL)",
                FrameRate::Fps2997DF => "29.97 fps DF (NTSC)",
                FrameRate::Fps2997NDF => "29.97 fps NDF (NTSC)",
                FrameRate::Fps30 => "30 fps",
                FrameRate::Fps50 => "50 fps",
                FrameRate::Fps5994 => "59.94 fps",
                FrameRate::Fps60 => "60 fps",
            };
            format!(
                r#"{{"fps":{},"is_drop_frame":{},"name":"{}","rational":[{},{}]}}"#,
                fr.as_float(),
                fr.is_drop_frame(),
                name,
                num,
                den,
            )
        })
        .collect();

    Ok(format!("[{}]", entries.join(",")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_frame_rate_25() {
        let fr = resolve_frame_rate(25.0, false);
        assert!(fr.is_ok());
        assert_eq!(fr.ok(), Some(FrameRate::Fps25));
    }

    #[test]
    fn test_resolve_frame_rate_29_97_df() {
        let fr = resolve_frame_rate(29.97, true);
        assert!(fr.is_ok());
        assert_eq!(fr.ok(), Some(FrameRate::Fps2997DF));
    }

    #[test]
    fn test_resolve_frame_rate_unsupported() {
        let fr = resolve_frame_rate(12.5, false);
        assert!(fr.is_err());
    }

    #[test]
    fn test_parse_tc_str_valid() {
        let tc = parse_tc_str("01:02:03:04", FrameRate::Fps25);
        assert!(tc.is_ok());
        let tc = tc.expect("should parse");
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 2);
        assert_eq!(tc.seconds, 3);
        assert_eq!(tc.frames, 4);
    }

    #[test]
    fn test_parse_tc_str_invalid_format() {
        let tc = parse_tc_str("01:02:03", FrameRate::Fps25);
        assert!(tc.is_err());
    }

    #[test]
    fn test_tc_to_json_format() {
        let tc = Timecode::new(1, 0, 0, 0, FrameRate::Fps25).expect("valid tc");
        let json = tc_to_json(&tc, FrameRate::Fps25);
        assert!(json.contains("\"hours\":1"));
        assert!(json.contains("\"total_frames\":90000"));
        assert!(json.contains("\"is_drop_frame\":false"));
    }

    #[test]
    fn test_wasm_list_frame_rates_json() {
        let result = wasm_list_frame_rates();
        assert!(result.is_ok());
        let json = result.expect("should return json");
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        // Should contain all 9 frame rates
        assert!(json.contains("23.976"));
        assert!(json.contains("59.94"));
    }

    #[test]
    fn test_wasm_timecode_to_frames() {
        let frames = wasm_timecode_to_frames("01:00:00:00", 25.0, false);
        assert!(frames.is_ok());
        assert_eq!(frames.ok(), Some(90000));
    }

    #[test]
    fn test_wasm_timecode_to_seconds() {
        let secs = wasm_timecode_to_seconds("00:01:00:00", 25.0, false);
        assert!(secs.is_ok());
        let s = secs.expect("should convert");
        assert!((s - 60.0).abs() < 0.01);
    }
}
