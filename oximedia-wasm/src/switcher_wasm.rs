//! WebAssembly bindings for live production video switching.
//!
//! Provides `WasmSwitcher` for browser-based source switching and
//! transition management, plus standalone utility functions.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// WasmSwitcher
// ---------------------------------------------------------------------------

/// Browser-side live production switcher.
#[wasm_bindgen]
pub struct WasmSwitcher {
    me_rows: usize,
    sources: Vec<SourceData>,
    program_input: usize,
    preview_input: usize,
    active_me: usize,
}

struct SourceData {
    index: usize,
    name: String,
    source_type: String,
}

#[wasm_bindgen]
impl WasmSwitcher {
    /// Create a new switcher with the given number of M/E rows.
    #[wasm_bindgen(constructor)]
    pub fn new(me_rows: u32) -> Result<WasmSwitcher, JsValue> {
        if me_rows == 0 || me_rows > 4 {
            return Err(crate::utils::js_err("me_rows must be 1-4"));
        }
        Ok(Self {
            me_rows: me_rows as usize,
            sources: Vec::new(),
            program_input: 0,
            preview_input: 0,
            active_me: 0,
        })
    }

    /// Add a source. Returns the assigned input index.
    pub fn add_source(&mut self, name: &str, source_type: &str) -> u32 {
        let index = self.sources.len();
        self.sources.push(SourceData {
            index,
            name: name.to_string(),
            source_type: source_type.to_string(),
        });
        index as u32
    }

    /// Get the number of sources.
    pub fn source_count(&self) -> u32 {
        self.sources.len() as u32
    }

    /// Set the program source.
    pub fn set_program(&mut self, input: u32) -> Result<(), JsValue> {
        let idx = input as usize;
        if idx >= self.sources.len() && idx > 0 {
            return Err(crate::utils::js_err(&format!(
                "Input {} out of range",
                input
            )));
        }
        self.program_input = idx;
        Ok(())
    }

    /// Set the preview source.
    pub fn set_preview(&mut self, input: u32) -> Result<(), JsValue> {
        let idx = input as usize;
        if idx >= self.sources.len() && idx > 0 {
            return Err(crate::utils::js_err(&format!(
                "Input {} out of range",
                input
            )));
        }
        self.preview_input = idx;
        Ok(())
    }

    /// Perform a cut (swap program and preview).
    pub fn cut(&mut self) {
        std::mem::swap(&mut self.program_input, &mut self.preview_input);
    }

    /// Get the current program input index.
    pub fn program(&self) -> u32 {
        self.program_input as u32
    }

    /// Get the current preview input index.
    pub fn preview(&self) -> u32 {
        self.preview_input as u32
    }

    /// Get the active M/E row.
    pub fn active_me_row(&self) -> u32 {
        self.active_me as u32
    }

    /// Set the active M/E row.
    pub fn set_active_me_row(&mut self, row: u32) -> Result<(), JsValue> {
        let r = row as usize;
        if r >= self.me_rows {
            return Err(crate::utils::js_err(&format!(
                "M/E row {} out of range (0..{})",
                row, self.me_rows
            )));
        }
        self.active_me = r;
        Ok(())
    }

    /// Get switcher info as JSON.
    pub fn info(&self) -> String {
        format!(
            "{{\"me_rows\":{},\"sources\":{},\"program\":{},\"preview\":{},\"active_me\":{}}}",
            self.me_rows,
            self.sources.len(),
            self.program_input,
            self.preview_input,
            self.active_me,
        )
    }

    /// List sources as JSON.
    pub fn sources_json(&self) -> String {
        let items: Vec<String> = self
            .sources
            .iter()
            .map(|s| {
                format!(
                    "{{\"index\":{},\"name\":\"{}\",\"type\":\"{}\"}}",
                    s.index, s.name, s.source_type,
                )
            })
            .collect();
        format!("[{}]", items.join(","))
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Create a transition descriptor as JSON.
///
/// # Arguments
/// * `transition_type` - "cut", "mix", "wipe", or "dve"
/// * `duration_frames` - Duration in frames (0 for cut)
///
/// Returns a JSON string with transition parameters.
#[wasm_bindgen]
pub fn wasm_create_transition(
    transition_type: &str,
    duration_frames: u32,
) -> Result<String, JsValue> {
    match transition_type {
        "cut" | "mix" | "wipe" | "dve" => {}
        other => {
            return Err(crate::utils::js_err(&format!(
                "Unknown transition type '{}'. Use: cut, mix, wipe, dve",
                other
            )));
        }
    }

    let actual_duration = if transition_type == "cut" {
        0
    } else {
        duration_frames
    };

    Ok(format!(
        "{{\"type\":\"{}\",\"duration_frames\":{}}}",
        transition_type, actual_duration,
    ))
}

/// List available transition types as a JSON array.
#[wasm_bindgen]
pub fn wasm_list_transition_types() -> String {
    "[\"cut\",\"mix\",\"wipe\",\"dve\"]".to_string()
}

/// List available switcher presets as a JSON array.
#[wasm_bindgen]
pub fn wasm_list_switcher_presets() -> String {
    "[{\"name\":\"basic\",\"me_rows\":1,\"inputs\":8,\"aux\":2},\
     {\"name\":\"professional\",\"me_rows\":2,\"inputs\":20,\"aux\":6},\
     {\"name\":\"broadcast\",\"me_rows\":4,\"inputs\":40,\"aux\":10}]"
        .to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_switcher_creation() {
        let sw = WasmSwitcher::new(1);
        assert!(sw.is_ok());
        let sw = sw.expect("valid");
        assert_eq!(sw.source_count(), 0);
    }

    #[test]
    fn test_switcher_add_source() {
        let mut sw = WasmSwitcher::new(1).expect("valid");
        let idx = sw.add_source("Cam1", "sdi");
        assert_eq!(idx, 0);
        assert_eq!(sw.source_count(), 1);
    }

    #[test]
    fn test_switcher_cut() {
        let mut sw = WasmSwitcher::new(1).expect("valid");
        sw.add_source("Cam1", "sdi");
        sw.add_source("Cam2", "ndi");

        sw.set_program(0).expect("valid");
        sw.set_preview(1).expect("valid");
        assert_eq!(sw.program(), 0);
        assert_eq!(sw.preview(), 1);

        sw.cut();
        assert_eq!(sw.program(), 1);
        assert_eq!(sw.preview(), 0);
    }

    #[test]
    fn test_create_transition() {
        let result = wasm_create_transition("mix", 30);
        assert!(result.is_ok());
        let r = result.expect("valid");
        assert!(r.contains("\"type\":\"mix\""));
        assert!(r.contains("\"duration_frames\":30"));
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_create_transition_invalid() {
        let result = wasm_create_transition("unknown", 30);
        assert!(result.is_err());
    }
}
