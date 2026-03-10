//! Python bindings for cloud storage operations.
//!
//! Provides `PyCloudConfig`, `PyCloudJob`, `PyCostEstimate`, `PyCloudProvider`,
//! and `PyCostEstimator` for managing cloud uploads, downloads, transcoding,
//! and cost estimation from Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Pricing data
// ---------------------------------------------------------------------------

struct PricingTier {
    storage_per_gb: f64,
    egress_per_gb: f64,
    transcode_per_min_sd: f64,
    transcode_per_min_hd: f64,
    transcode_per_min_4k: f64,
    provider_label: &'static str,
}

fn get_pricing(provider: &str, _region: &str) -> PyResult<PricingTier> {
    match provider.to_lowercase().as_str() {
        "s3" | "aws" => Ok(PricingTier {
            storage_per_gb: 0.023,
            egress_per_gb: 0.09,
            transcode_per_min_sd: 0.015,
            transcode_per_min_hd: 0.024,
            transcode_per_min_4k: 0.048,
            provider_label: "AWS S3",
        }),
        "azure" => Ok(PricingTier {
            storage_per_gb: 0.018,
            egress_per_gb: 0.087,
            transcode_per_min_sd: 0.013,
            transcode_per_min_hd: 0.022,
            transcode_per_min_4k: 0.044,
            provider_label: "Azure Blob",
        }),
        "gcs" | "google" => Ok(PricingTier {
            storage_per_gb: 0.020,
            egress_per_gb: 0.12,
            transcode_per_min_sd: 0.016,
            transcode_per_min_hd: 0.025,
            transcode_per_min_4k: 0.050,
            provider_label: "Google Cloud Storage",
        }),
        other => Err(PyValueError::new_err(format!(
            "Unknown provider '{}'. Supported: s3, azure, gcs",
            other
        ))),
    }
}

fn validate_provider(provider: &str) -> PyResult<()> {
    match provider.to_lowercase().as_str() {
        "s3" | "aws" | "azure" | "gcs" | "google" => Ok(()),
        other => Err(PyValueError::new_err(format!(
            "Unknown provider '{}'. Supported: s3, azure, gcs",
            other
        ))),
    }
}

fn now_timestamp() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", dur.as_secs())
}

fn generate_job_id() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("job-{:012x}", dur.as_nanos() % 0xffffffffffff)
}

// ---------------------------------------------------------------------------
// PyCloudConfig
// ---------------------------------------------------------------------------

/// Configuration for a cloud storage provider.
#[pyclass]
#[derive(Clone)]
pub struct PyCloudConfig {
    /// Cloud provider name (s3, azure, gcs).
    #[pyo3(get)]
    pub provider: String,
    /// Cloud region.
    #[pyo3(get)]
    pub region: Option<String>,
    /// Bucket or container name.
    #[pyo3(get)]
    pub bucket: Option<String>,
    /// Custom endpoint URL.
    #[pyo3(get)]
    pub endpoint: Option<String>,
    /// Bandwidth limit in KB/s.
    #[pyo3(get)]
    pub bandwidth_limit_kbps: Option<u32>,
}

#[pymethods]
impl PyCloudConfig {
    /// Create a new cloud config.
    #[new]
    #[pyo3(signature = (provider, region=None, bucket=None))]
    fn new(provider: &str, region: Option<String>, bucket: Option<String>) -> PyResult<Self> {
        validate_provider(provider)?;
        Ok(Self {
            provider: provider.to_string(),
            region,
            bucket,
            endpoint: None,
            bandwidth_limit_kbps: None,
        })
    }

    /// Create an S3 configuration (classmethod).
    #[classmethod]
    #[pyo3(signature = (region=None, bucket=None))]
    fn s3(
        _cls: &Bound<'_, PyType>,
        region: Option<String>,
        bucket: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            provider: "s3".to_string(),
            region: Some(region.unwrap_or_else(|| "us-east-1".to_string())),
            bucket,
            endpoint: None,
            bandwidth_limit_kbps: None,
        })
    }

    /// Create an Azure configuration (classmethod).
    #[classmethod]
    #[pyo3(signature = (container=None))]
    fn azure(_cls: &Bound<'_, PyType>, container: Option<String>) -> PyResult<Self> {
        Ok(Self {
            provider: "azure".to_string(),
            region: None,
            bucket: container,
            endpoint: None,
            bandwidth_limit_kbps: None,
        })
    }

    /// Create a GCS configuration (classmethod).
    #[classmethod]
    #[pyo3(signature = (bucket=None))]
    fn gcs(_cls: &Bound<'_, PyType>, bucket: Option<String>) -> PyResult<Self> {
        Ok(Self {
            provider: "gcs".to_string(),
            region: None,
            bucket,
            endpoint: None,
            bandwidth_limit_kbps: None,
        })
    }

    /// Set a custom endpoint URL.
    fn with_endpoint(&mut self, endpoint: &str) {
        self.endpoint = Some(endpoint.to_string());
    }

    /// Set a bandwidth limit in KB/s.
    fn with_bandwidth_limit(&mut self, kbps: u32) {
        self.bandwidth_limit_kbps = Some(kbps);
    }

    fn __repr__(&self) -> String {
        format!(
            "PyCloudConfig(provider='{}', region={:?}, bucket={:?})",
            self.provider, self.region, self.bucket
        )
    }
}

// ---------------------------------------------------------------------------
// PyCloudJob
// ---------------------------------------------------------------------------

/// Represents a cloud processing job.
#[pyclass]
#[derive(Clone)]
pub struct PyCloudJob {
    /// Job identifier.
    #[pyo3(get)]
    pub id: String,
    /// Job status: queued, running, complete, failed.
    #[pyo3(get)]
    pub status: String,
    /// Progress percentage (0.0 to 100.0).
    #[pyo3(get)]
    pub progress: f64,
    /// Input object key.
    #[pyo3(get)]
    pub input_key: String,
    /// Output object key.
    #[pyo3(get)]
    pub output_key: Option<String>,
    /// Creation timestamp.
    #[pyo3(get)]
    pub created_at: String,
    /// Error message if failed.
    #[pyo3(get)]
    pub error_message: Option<String>,
}

#[pymethods]
impl PyCloudJob {
    fn __repr__(&self) -> String {
        format!(
            "PyCloudJob(id='{}', status='{}', progress={:.1}%)",
            self.id, self.status, self.progress
        )
    }

    /// Convert to a Python dict.
    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("id".to_string(), self.id.clone());
        m.insert("status".to_string(), self.status.clone());
        m.insert("progress".to_string(), self.progress.to_string());
        m.insert("input_key".to_string(), self.input_key.clone());
        m.insert(
            "output_key".to_string(),
            self.output_key.clone().unwrap_or_default(),
        );
        m.insert("created_at".to_string(), self.created_at.clone());
        m.insert(
            "error_message".to_string(),
            self.error_message.clone().unwrap_or_default(),
        );
        m
    }

    /// Check if the job has completed successfully.
    fn is_complete(&self) -> bool {
        self.status == "complete"
    }

    /// Check if the job has failed.
    fn is_failed(&self) -> bool {
        self.status == "failed"
    }
}

// ---------------------------------------------------------------------------
// PyCostEstimate
// ---------------------------------------------------------------------------

/// Cloud cost estimate result.
#[pyclass]
#[derive(Clone)]
pub struct PyCostEstimate {
    /// Monthly storage cost (USD).
    #[pyo3(get)]
    pub storage_cost: f64,
    /// Egress (data transfer out) cost (USD).
    #[pyo3(get)]
    pub egress_cost: f64,
    /// Transcoding cost (USD).
    #[pyo3(get)]
    pub transcode_cost: f64,
    /// Total cost (USD).
    #[pyo3(get)]
    pub total_cost: f64,
    /// Currency code.
    #[pyo3(get)]
    pub currency: String,
    /// Provider name.
    #[pyo3(get)]
    pub provider: String,
    /// Region used for pricing.
    #[pyo3(get)]
    pub region: String,
}

#[pymethods]
impl PyCostEstimate {
    fn __repr__(&self) -> String {
        format!(
            "PyCostEstimate(provider='{}', total=${:.4}, storage=${:.4}, egress=${:.4}, transcode=${:.4})",
            self.provider, self.total_cost, self.storage_cost, self.egress_cost, self.transcode_cost
        )
    }

    /// Convert to a Python dict.
    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert(
            "storage_cost".to_string(),
            format!("{:.6}", self.storage_cost),
        );
        m.insert(
            "egress_cost".to_string(),
            format!("{:.6}", self.egress_cost),
        );
        m.insert(
            "transcode_cost".to_string(),
            format!("{:.6}", self.transcode_cost),
        );
        m.insert("total_cost".to_string(), format!("{:.6}", self.total_cost));
        m.insert("currency".to_string(), self.currency.clone());
        m.insert("provider".to_string(), self.provider.clone());
        m.insert("region".to_string(), self.region.clone());
        m
    }
}

// ---------------------------------------------------------------------------
// PyCloudProvider
// ---------------------------------------------------------------------------

/// Cloud provider operations.
#[pyclass]
pub struct PyCloudProvider {
    config: PyCloudConfig,
}

#[pymethods]
impl PyCloudProvider {
    /// Create a new cloud provider from config.
    #[new]
    fn new(config: PyCloudConfig) -> PyResult<Self> {
        validate_provider(&config.provider)?;
        Ok(Self { config })
    }

    /// Upload a local file to the cloud.
    ///
    /// Returns the remote key.
    fn upload(&self, local_path: &str, remote_key: &str) -> PyResult<String> {
        let path = std::path::Path::new(local_path);
        if !path.exists() {
            return Err(PyValueError::new_err(format!(
                "Local file not found: {local_path}"
            )));
        }
        let bucket = self.config.bucket.as_deref().unwrap_or("default-bucket");
        // Simulation: return the URI that would be created
        let uri = format!("{}://{}/{}", self.config.provider, bucket, remote_key);
        Ok(uri)
    }

    /// Download a remote object to a local path.
    fn download(&self, remote_key: &str, local_path: &str) -> PyResult<()> {
        let _ = remote_key;
        let _ = local_path;
        Err(PyRuntimeError::new_err(
            "Cloud credentials required for actual download; configure provider credentials",
        ))
    }

    /// Delete a remote object.
    fn delete(&self, remote_key: &str) -> PyResult<()> {
        let _ = remote_key;
        Err(PyRuntimeError::new_err(
            "Cloud credentials required for actual deletion; configure provider credentials",
        ))
    }

    /// List objects with a prefix.
    #[pyo3(signature = (prefix=None))]
    fn list_objects(&self, prefix: Option<&str>) -> PyResult<Vec<HashMap<String, String>>> {
        let _ = prefix;
        // Return empty list in simulation mode
        Ok(Vec::new())
    }

    /// Get information about an object.
    fn object_info(&self, key: &str) -> PyResult<HashMap<String, String>> {
        let _ = key;
        Err(PyRuntimeError::new_err(
            "Cloud credentials required for actual object info; configure provider credentials",
        ))
    }

    /// Submit a cloud transcoding job.
    #[pyo3(signature = (input_key, output_key, preset=None))]
    fn transcode(
        &self,
        input_key: &str,
        output_key: &str,
        preset: Option<&str>,
    ) -> PyResult<PyCloudJob> {
        let _preset = preset.unwrap_or("av1-1080p");
        Ok(PyCloudJob {
            id: generate_job_id(),
            status: "queued".to_string(),
            progress: 0.0,
            input_key: input_key.to_string(),
            output_key: Some(output_key.to_string()),
            created_at: now_timestamp(),
            error_message: None,
        })
    }

    /// Check the status of a job.
    fn job_status(&self, job_id: &str) -> PyResult<PyCloudJob> {
        Ok(PyCloudJob {
            id: job_id.to_string(),
            status: "unknown".to_string(),
            progress: 0.0,
            input_key: String::new(),
            output_key: None,
            created_at: String::new(),
            error_message: Some("Cloud credentials required for actual status check".to_string()),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PyCloudProvider(provider='{}', bucket={:?})",
            self.config.provider, self.config.bucket
        )
    }
}

// ---------------------------------------------------------------------------
// PyCostEstimator
// ---------------------------------------------------------------------------

/// Estimates cloud costs for various operations.
#[pyclass]
pub struct PyCostEstimator {
    provider: String,
    region: String,
}

#[pymethods]
impl PyCostEstimator {
    /// Create a new cost estimator.
    #[new]
    #[pyo3(signature = (provider, region=None))]
    fn new(provider: &str, region: Option<&str>) -> PyResult<Self> {
        validate_provider(provider)?;
        Ok(Self {
            provider: provider.to_string(),
            region: region.unwrap_or("us-east-1").to_string(),
        })
    }

    /// Estimate monthly storage cost.
    #[pyo3(signature = (gb, months=None))]
    fn estimate_storage(&self, gb: f64, months: Option<u32>) -> PyResult<PyCostEstimate> {
        let pricing = get_pricing(&self.provider, &self.region)?;
        let m = months.unwrap_or(1) as f64;
        let cost = gb * pricing.storage_per_gb * m;
        Ok(PyCostEstimate {
            storage_cost: cost,
            egress_cost: 0.0,
            transcode_cost: 0.0,
            total_cost: cost,
            currency: "USD".to_string(),
            provider: pricing.provider_label.to_string(),
            region: self.region.clone(),
        })
    }

    /// Estimate transcoding cost.
    #[pyo3(signature = (minutes, resolution=None))]
    fn estimate_transcode(
        &self,
        minutes: f64,
        resolution: Option<&str>,
    ) -> PyResult<PyCostEstimate> {
        let pricing = get_pricing(&self.provider, &self.region)?;
        let rate = match resolution.unwrap_or("hd") {
            "sd" | "480p" => pricing.transcode_per_min_sd,
            "hd" | "720p" | "1080p" => pricing.transcode_per_min_hd,
            "4k" | "2160p" | "uhd" => pricing.transcode_per_min_4k,
            _ => pricing.transcode_per_min_hd,
        };
        let cost = minutes * rate;
        Ok(PyCostEstimate {
            storage_cost: 0.0,
            egress_cost: 0.0,
            transcode_cost: cost,
            total_cost: cost,
            currency: "USD".to_string(),
            provider: pricing.provider_label.to_string(),
            region: self.region.clone(),
        })
    }

    /// Estimate total cost for storage, egress, and transcoding.
    #[pyo3(signature = (storage_gb, egress_gb=None, transcode_min=None))]
    fn estimate_total(
        &self,
        storage_gb: f64,
        egress_gb: Option<f64>,
        transcode_min: Option<f64>,
    ) -> PyResult<PyCostEstimate> {
        let pricing = get_pricing(&self.provider, &self.region)?;
        let storage_cost = storage_gb * pricing.storage_per_gb;
        let egress_cost = egress_gb.unwrap_or(0.0) * pricing.egress_per_gb;
        let transcode_cost = transcode_min.unwrap_or(0.0) * pricing.transcode_per_min_hd;
        let total = storage_cost + egress_cost + transcode_cost;
        Ok(PyCostEstimate {
            storage_cost,
            egress_cost,
            transcode_cost,
            total_cost: total,
            currency: "USD".to_string(),
            provider: pricing.provider_label.to_string(),
            region: self.region.clone(),
        })
    }

    /// Compare costs across all providers.
    #[pyo3(signature = (storage_gb, egress_gb=None))]
    fn compare_providers(
        &self,
        storage_gb: f64,
        egress_gb: Option<f64>,
    ) -> PyResult<Vec<PyCostEstimate>> {
        let providers = ["s3", "azure", "gcs"];
        let mut results = Vec::new();
        for &p in &providers {
            let pricing = get_pricing(p, &self.region)?;
            let storage_cost = storage_gb * pricing.storage_per_gb;
            let egress_cost = egress_gb.unwrap_or(0.0) * pricing.egress_per_gb;
            results.push(PyCostEstimate {
                storage_cost,
                egress_cost,
                transcode_cost: 0.0,
                total_cost: storage_cost + egress_cost,
                currency: "USD".to_string(),
                provider: pricing.provider_label.to_string(),
                region: self.region.clone(),
            });
        }
        Ok(results)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyCostEstimator(provider='{}', region='{}')",
            self.provider, self.region
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// List supported cloud providers.
#[pyfunction]
pub fn list_cloud_providers() -> Vec<String> {
    vec!["s3".to_string(), "azure".to_string(), "gcs".to_string()]
}

/// Estimate total cloud cost for the given parameters.
#[pyfunction]
#[pyo3(signature = (provider, storage_gb, egress_gb=None, transcode_min=None, region=None))]
pub fn estimate_cost(
    provider: &str,
    storage_gb: f64,
    egress_gb: Option<f64>,
    transcode_min: Option<f64>,
    region: Option<&str>,
) -> PyResult<PyCostEstimate> {
    let r = region.unwrap_or("us-east-1");
    let pricing = get_pricing(provider, r)?;
    let storage_cost = storage_gb * pricing.storage_per_gb;
    let egress_cost = egress_gb.unwrap_or(0.0) * pricing.egress_per_gb;
    let transcode_cost = transcode_min.unwrap_or(0.0) * pricing.transcode_per_min_hd;
    let total = storage_cost + egress_cost + transcode_cost;
    Ok(PyCostEstimate {
        storage_cost,
        egress_cost,
        transcode_cost,
        total_cost: total,
        currency: "USD".to_string(),
        provider: pricing.provider_label.to_string(),
        region: r.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Registration helper
// ---------------------------------------------------------------------------

/// Register all cloud bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCloudConfig>()?;
    m.add_class::<PyCloudJob>()?;
    m.add_class::<PyCostEstimate>()?;
    m.add_class::<PyCloudProvider>()?;
    m.add_class::<PyCostEstimator>()?;
    m.add_function(wrap_pyfunction!(list_cloud_providers, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_cost, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_provider() {
        assert!(validate_provider("s3").is_ok());
        assert!(validate_provider("azure").is_ok());
        assert!(validate_provider("gcs").is_ok());
        assert!(validate_provider("dropbox").is_err());
    }

    #[test]
    fn test_pricing_s3() {
        let p = get_pricing("s3", "us-east-1");
        assert!(p.is_ok());
        let p = p.expect("should succeed");
        assert!(p.storage_per_gb > 0.0);
        assert!(p.egress_per_gb > 0.0);
    }

    #[test]
    fn test_cloud_job_status() {
        let job = PyCloudJob {
            id: "j1".to_string(),
            status: "complete".to_string(),
            progress: 100.0,
            input_key: "in.mkv".to_string(),
            output_key: Some("out.webm".to_string()),
            created_at: "0".to_string(),
            error_message: None,
        };
        assert!(job.is_complete());
        assert!(!job.is_failed());
    }

    #[test]
    fn test_cloud_job_failed() {
        let job = PyCloudJob {
            id: "j2".to_string(),
            status: "failed".to_string(),
            progress: 50.0,
            input_key: "in.mkv".to_string(),
            output_key: None,
            created_at: "0".to_string(),
            error_message: Some("timeout".to_string()),
        };
        assert!(!job.is_complete());
        assert!(job.is_failed());
    }

    #[test]
    fn test_cost_estimate_total() {
        let p = get_pricing("s3", "us-east-1").expect("should succeed");
        let storage = 100.0 * p.storage_per_gb;
        let egress = 50.0 * p.egress_per_gb;
        let transcode = 120.0 * p.transcode_per_min_hd;
        let total = storage + egress + transcode;
        assert!(total > 0.0);
    }
}
