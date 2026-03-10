//! WebAssembly bindings for batch processing utilities.
//!
//! Provides batch configuration validation, time estimation, and a
//! browser-side job queue for planning batch operations.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Validate a batch configuration JSON and return warnings/errors.
///
/// Accepts a JSON string with fields like:
/// ```json
/// {"max_parallel": 4, "retry_count": 2, "stop_on_error": false, "overwrite": true}
/// ```
///
/// Returns a JSON result with validation status and any warnings.
#[wasm_bindgen]
pub fn wasm_validate_batch_config(config_json: &str) -> Result<String, JsValue> {
    let parsed: serde_json::Value = serde_json::from_str(config_json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid JSON: {e}")))?;

    let mut warnings: Vec<String> = Vec::new();
    let mut errors: Vec<String> = Vec::new();

    // Validate max_parallel
    if let Some(mp) = parsed.get("max_parallel") {
        if let Some(n) = mp.as_u64() {
            if n == 0 {
                errors.push("max_parallel must be greater than 0".to_string());
            } else if n > 64 {
                warnings.push(format!(
                    "max_parallel={n} is very high; consider reducing for stability"
                ));
            }
        } else {
            errors.push("max_parallel must be a positive integer".to_string());
        }
    }

    // Validate retry_count
    if let Some(rc) = parsed.get("retry_count") {
        if let Some(n) = rc.as_u64() {
            if n > 10 {
                warnings.push(format!(
                    "retry_count={n} is unusually high; typical values are 0-3"
                ));
            }
        } else {
            errors.push("retry_count must be a non-negative integer".to_string());
        }
    }

    // Validate output_dir
    if let Some(od) = parsed.get("output_dir") {
        if let Some(s) = od.as_str() {
            if s.is_empty() {
                errors.push("output_dir must not be empty if specified".to_string());
            }
        }
    }

    let is_valid = errors.is_empty();

    Ok(format!(
        "{{\"valid\":{is_valid},\"errors\":{errors},\"warnings\":{warnings}}}",
        is_valid = is_valid,
        errors = serde_json::to_string(&errors).unwrap_or_else(|_| "[]".to_string()),
        warnings = serde_json::to_string(&warnings).unwrap_or_else(|_| "[]".to_string()),
    ))
}

/// Estimate the total processing time for a batch of jobs.
///
/// # Arguments
/// * `job_count` - Total number of jobs.
/// * `avg_duration_secs` - Average processing time per job in seconds.
/// * `parallel` - Number of parallel workers.
///
/// Returns a JSON string with estimated times.
#[wasm_bindgen]
pub fn wasm_estimate_batch_time(
    job_count: u32,
    avg_duration_secs: f64,
    parallel: u32,
) -> Result<String, JsValue> {
    if parallel == 0 {
        return Err(crate::utils::js_err("parallel must be greater than 0"));
    }
    if avg_duration_secs < 0.0 {
        return Err(crate::utils::js_err(
            "avg_duration_secs must be non-negative",
        ));
    }

    let effective_parallel = parallel.min(job_count).max(1);
    let batches = (job_count as f64 / effective_parallel as f64).ceil();
    let total_secs = batches * avg_duration_secs;
    let total_mins = total_secs / 60.0;
    let total_hours = total_mins / 60.0;

    // Estimate with overhead (5% per batch for scheduling)
    let overhead_factor = 1.05;
    let total_with_overhead = total_secs * overhead_factor;

    Ok(format!(
        "{{\"job_count\":{job_count},\"parallel\":{effective_parallel},\
         \"estimated_seconds\":{total_secs:.1},\"estimated_minutes\":{total_mins:.2},\
         \"estimated_hours\":{total_hours:.3},\"with_overhead_seconds\":{total_with_overhead:.1},\
         \"batches\":{batches:.0}}}"
    ))
}

// ---------------------------------------------------------------------------
// WasmBatchQueue
// ---------------------------------------------------------------------------

/// A job entry in the batch queue.
struct WasmBatchJob {
    input_name: String,
    output_name: String,
    preset: String,
}

/// Browser-side batch job queue for planning and validation.
#[wasm_bindgen]
pub struct WasmBatchQueue {
    jobs: Vec<WasmBatchJob>,
}

#[wasm_bindgen]
impl WasmBatchQueue {
    /// Create a new empty batch queue.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { jobs: Vec::new() }
    }

    /// Add a job to the queue.
    ///
    /// # Arguments
    /// * `input_name` - Input file name or identifier.
    /// * `output_name` - Output file name or identifier.
    /// * `preset` - Processing preset name.
    pub fn add_job(
        &mut self,
        input_name: &str,
        output_name: &str,
        preset: &str,
    ) -> Result<(), JsValue> {
        if input_name.is_empty() {
            return Err(crate::utils::js_err("input_name must not be empty"));
        }
        if output_name.is_empty() {
            return Err(crate::utils::js_err("output_name must not be empty"));
        }
        self.jobs.push(WasmBatchJob {
            input_name: input_name.to_string(),
            output_name: output_name.to_string(),
            preset: preset.to_string(),
        });
        Ok(())
    }

    /// Remove a job by index.
    pub fn remove_job(&mut self, index: u32) -> Result<(), JsValue> {
        let idx = index as usize;
        if idx >= self.jobs.len() {
            return Err(crate::utils::js_err(&format!(
                "Index {} out of range (0..{})",
                index,
                self.jobs.len()
            )));
        }
        self.jobs.remove(idx);
        Ok(())
    }

    /// Get the number of jobs in the queue.
    pub fn job_count(&self) -> u32 {
        self.jobs.len() as u32
    }

    /// Serialize the queue to a JSON string.
    pub fn to_json(&self) -> Result<String, JsValue> {
        let jobs_json: Vec<String> = self
            .jobs
            .iter()
            .map(|j| {
                format!(
                    "{{\"input\":\"{}\",\"output\":\"{}\",\"preset\":\"{}\"}}",
                    j.input_name, j.output_name, j.preset
                )
            })
            .collect();
        Ok(format!("{{\"jobs\":[{}]}}", jobs_json.join(",")))
    }

    /// Validate the queue and return any issues.
    ///
    /// Returns a JSON string with validation results.
    pub fn validate(&self) -> Result<String, JsValue> {
        let mut warnings: Vec<String> = Vec::new();

        if self.jobs.is_empty() {
            warnings.push("Queue is empty".to_string());
        }

        // Check for duplicate outputs
        let mut seen_outputs = std::collections::HashSet::new();
        for job in &self.jobs {
            if !seen_outputs.insert(&job.output_name) {
                warnings.push(format!("Duplicate output name: '{}'", job.output_name));
            }
        }

        // Check for same input/output
        for job in &self.jobs {
            if job.input_name == job.output_name {
                warnings.push(format!(
                    "Input and output are the same: '{}'",
                    job.input_name
                ));
            }
        }

        let is_valid = warnings.is_empty();
        Ok(format!(
            "{{\"valid\":{is_valid},\"warnings\":{warnings},\"job_count\":{count}}}",
            is_valid = is_valid,
            warnings = serde_json::to_string(&warnings).unwrap_or_else(|_| "[]".to_string()),
            count = self.jobs.len(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_valid_config() {
        let json = r#"{"max_parallel": 4, "retry_count": 2, "stop_on_error": false}"#;
        let result = wasm_validate_batch_config(json);
        assert!(result.is_ok());
        let output = result.expect("should succeed");
        assert!(output.contains("\"valid\":true"));
    }

    #[test]
    fn test_validate_invalid_parallel() {
        let json = r#"{"max_parallel": 0}"#;
        let result = wasm_validate_batch_config(json);
        assert!(result.is_ok());
        let output = result.expect("should succeed");
        assert!(output.contains("\"valid\":false"));
    }

    #[test]
    fn test_estimate_batch_time() {
        let result = wasm_estimate_batch_time(10, 60.0, 2);
        assert!(result.is_ok());
        let json = result.expect("should succeed");
        assert!(json.contains("\"estimated_seconds\":300.0"));
    }

    #[test]
    fn test_batch_queue_operations() {
        let mut queue = WasmBatchQueue::new();
        assert_eq!(queue.job_count(), 0);

        let r = queue.add_job("input.mkv", "output.webm", "default");
        assert!(r.is_ok());
        assert_eq!(queue.job_count(), 1);

        let r = queue.add_job("input2.mkv", "output2.webm", "hd");
        assert!(r.is_ok());
        assert_eq!(queue.job_count(), 2);

        let r = queue.remove_job(0);
        assert!(r.is_ok());
        assert_eq!(queue.job_count(), 1);
    }

    #[test]
    fn test_batch_queue_validate_duplicate_outputs() {
        let mut queue = WasmBatchQueue::new();
        let _ = queue.add_job("a.mkv", "output.webm", "default");
        let _ = queue.add_job("b.mkv", "output.webm", "default");
        let result = queue.validate();
        assert!(result.is_ok());
        let json = result.expect("should succeed");
        assert!(json.contains("Duplicate output"));
    }
}
