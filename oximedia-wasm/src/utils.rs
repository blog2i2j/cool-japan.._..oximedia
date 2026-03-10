//! Utilities for WASM bindings.

use oximedia_core::OxiError;
use wasm_bindgen::JsValue;

/// Create a `JsValue` from a string message, compatible with native tests.
///
/// In `wasm32` context this creates a real JS string value. On non-wasm
/// targets (e.g. unit-test compilation) calling `JsValue::from_str` crosses
/// an `extern "C"` boundary that panics, which aborts because panicking
/// across FFI is undefined behaviour. Returning `JsValue::null()` instead
/// keeps error-path tests from SIGABRTing while preserving the `Err` variant.
#[inline]
pub fn js_err(msg: &str) -> JsValue {
    #[cfg(target_arch = "wasm32")]
    {
        JsValue::from_str(msg)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = msg;
        JsValue::null()
    }
}

/// Convert `OxiError` to `JsValue` for JavaScript exception handling.
///
/// This allows Rust errors to be propagated to JavaScript as exceptions.
pub fn to_js_error(err: OxiError) -> JsValue {
    js_err(&format!("OxiMedia Error: {err}"))
}

/// Convert `OxiError` to JavaScript `Error` object.
///
/// Provides more detailed error information for JavaScript consumers.
#[allow(dead_code)]
pub fn to_js_error_object(err: OxiError) -> js_sys::Error {
    js_sys::Error::new(&format!("OxiMedia Error: {err}"))
}
