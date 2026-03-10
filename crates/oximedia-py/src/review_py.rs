//! Python bindings for review and approval workflows.
//!
//! Provides `PyReviewSession`, `PyAnnotation`, `PyReviewStatus` and standalone
//! functions for creating reviews, adding annotations, and managing approvals.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_timestamp() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", dur.as_secs())
}

fn gen_id(prefix: &str) -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{prefix}-{:016x}", dur.as_nanos())
}

// ---------------------------------------------------------------------------
// PyAnnotation
// ---------------------------------------------------------------------------

/// A frame-accurate annotation/comment on reviewed content.
#[pyclass]
#[derive(Clone)]
pub struct PyAnnotation {
    /// Annotation identifier.
    #[pyo3(get)]
    pub id: String,
    /// Author name.
    #[pyo3(get)]
    pub author: String,
    /// Annotation message.
    #[pyo3(get)]
    pub message: String,
    /// Type: general, issue, suggestion, question.
    #[pyo3(get)]
    pub annotation_type: String,
    /// Frame number (optional).
    #[pyo3(get)]
    pub frame: Option<u64>,
    /// Timecode (optional).
    #[pyo3(get)]
    pub timecode: Option<String>,
    /// Whether annotation is resolved.
    #[pyo3(get)]
    pub resolved: bool,
    /// Creation timestamp.
    #[pyo3(get)]
    pub created_at: String,
}

#[pymethods]
impl PyAnnotation {
    #[new]
    #[pyo3(signature = (author, message, annotation_type="general", frame=None, timecode=None))]
    fn new(
        author: &str,
        message: &str,
        annotation_type: &str,
        frame: Option<u64>,
        timecode: Option<String>,
    ) -> PyResult<Self> {
        let valid_type = match annotation_type {
            "general" | "issue" | "suggestion" | "question" => annotation_type.to_string(),
            _ => {
                return Err(PyValueError::new_err(
                    "Type must be general, issue, suggestion, or question",
                ))
            }
        };
        Ok(Self {
            id: gen_id("ann"),
            author: author.to_string(),
            message: message.to_string(),
            annotation_type: valid_type,
            frame,
            timecode,
            resolved: false,
            created_at: now_timestamp(),
        })
    }

    /// Mark the annotation as resolved.
    fn resolve(&mut self) {
        self.resolved = true;
    }

    /// Mark the annotation as unresolved.
    fn unresolve(&mut self) {
        self.resolved = false;
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("id".to_string(), self.id.clone());
        m.insert("author".to_string(), self.author.clone());
        m.insert("message".to_string(), self.message.clone());
        m.insert("type".to_string(), self.annotation_type.clone());
        m.insert(
            "frame".to_string(),
            self.frame.map_or(String::new(), |f| f.to_string()),
        );
        m.insert(
            "timecode".to_string(),
            self.timecode.clone().unwrap_or_default(),
        );
        m.insert("resolved".to_string(), self.resolved.to_string());
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "PyAnnotation(author='{}', type='{}', resolved={})",
            self.author, self.annotation_type, self.resolved
        )
    }
}

// ---------------------------------------------------------------------------
// PyReviewStatus
// ---------------------------------------------------------------------------

/// Summary status of a review session.
#[pyclass]
#[derive(Clone)]
pub struct PyReviewStatus {
    /// Current status: pending, approved, rejected, conditionally_approved.
    #[pyo3(get)]
    pub status: String,
    /// Total annotations.
    #[pyo3(get)]
    pub total_annotations: u32,
    /// Open issues count.
    #[pyo3(get)]
    pub open_issues: u32,
    /// Resolved issues count.
    #[pyo3(get)]
    pub resolved_issues: u32,
    /// Total approvals.
    #[pyo3(get)]
    pub approvals: u32,
    /// Total rejections.
    #[pyo3(get)]
    pub rejections: u32,
}

#[pymethods]
impl PyReviewStatus {
    fn is_approved(&self) -> bool {
        self.status == "approved"
    }

    fn is_rejected(&self) -> bool {
        self.status == "rejected"
    }

    fn is_pending(&self) -> bool {
        self.status == "pending"
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("status".to_string(), self.status.clone());
        m.insert(
            "total_annotations".to_string(),
            self.total_annotations.to_string(),
        );
        m.insert("open_issues".to_string(), self.open_issues.to_string());
        m.insert("approvals".to_string(), self.approvals.to_string());
        m.insert("rejections".to_string(), self.rejections.to_string());
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "PyReviewStatus(status='{}', open_issues={}, approvals={})",
            self.status, self.open_issues, self.approvals
        )
    }
}

// ---------------------------------------------------------------------------
// PyReviewSession
// ---------------------------------------------------------------------------

/// A review and approval session.
#[pyclass]
pub struct PyReviewSession {
    /// Session identifier.
    #[pyo3(get)]
    pub id: String,
    /// Session title.
    #[pyo3(get)]
    pub title: String,
    /// Content identifier.
    #[pyo3(get)]
    pub content_id: String,
    /// Workflow type.
    #[pyo3(get)]
    pub workflow: String,
    /// Description.
    #[pyo3(get, set)]
    pub description: String,
    /// Creation timestamp.
    #[pyo3(get)]
    pub created_at: String,
    annotations: Vec<PyAnnotation>,
    approval_count: u32,
    rejection_count: u32,
    status: String,
}

#[pymethods]
impl PyReviewSession {
    #[new]
    #[pyo3(signature = (title, content_id, workflow="simple", description=None))]
    fn new(
        title: &str,
        content_id: &str,
        workflow: &str,
        description: Option<&str>,
    ) -> PyResult<Self> {
        let valid_wf = match workflow {
            "simple" | "multi-stage" | "parallel" | "sequential" => workflow.to_string(),
            _ => {
                return Err(PyValueError::new_err(
                    "Workflow must be simple, multi-stage, parallel, or sequential",
                ))
            }
        };
        Ok(Self {
            id: gen_id("review"),
            title: title.to_string(),
            content_id: content_id.to_string(),
            workflow: valid_wf,
            description: description.unwrap_or("").to_string(),
            created_at: now_timestamp(),
            annotations: Vec::new(),
            approval_count: 0,
            rejection_count: 0,
            status: "pending".to_string(),
        })
    }

    /// Add an annotation to the session.
    fn add_annotation(&mut self, annotation: PyAnnotation) {
        self.annotations.push(annotation);
    }

    /// Get all annotations.
    fn annotations(&self) -> Vec<PyAnnotation> {
        self.annotations.clone()
    }

    /// Get open issues (unresolved issue-type annotations).
    fn open_issues(&self) -> Vec<PyAnnotation> {
        self.annotations
            .iter()
            .filter(|a| a.annotation_type == "issue" && !a.resolved)
            .cloned()
            .collect()
    }

    /// Approve the content.
    #[pyo3(signature = (approver, note=None))]
    fn approve(&mut self, approver: &str, note: Option<&str>) -> PyResult<PyReviewStatus> {
        let _ = (approver, note);
        self.approval_count += 1;

        let open = self
            .annotations
            .iter()
            .filter(|a| a.annotation_type == "issue" && !a.resolved)
            .count() as u32;

        if open == 0 {
            self.status = "approved".to_string();
        } else {
            self.status = "conditionally_approved".to_string();
        }

        Ok(self.get_status())
    }

    /// Reject the content.
    fn reject(&mut self, reviewer: &str, reason: &str) -> PyResult<PyReviewStatus> {
        let _ = (reviewer, reason);
        self.rejection_count += 1;
        self.status = "rejected".to_string();
        Ok(self.get_status())
    }

    /// Get the current status.
    fn get_status(&self) -> PyReviewStatus {
        let open_issues = self
            .annotations
            .iter()
            .filter(|a| a.annotation_type == "issue" && !a.resolved)
            .count() as u32;
        let resolved = self
            .annotations
            .iter()
            .filter(|a| a.annotation_type == "issue" && a.resolved)
            .count() as u32;

        PyReviewStatus {
            status: self.status.clone(),
            total_annotations: self.annotations.len() as u32,
            open_issues,
            resolved_issues: resolved,
            approvals: self.approval_count,
            rejections: self.rejection_count,
        }
    }

    /// Export annotations as JSON string.
    fn export_json(&self) -> PyResult<String> {
        let data: Vec<HashMap<String, String>> =
            self.annotations.iter().map(|a| a.to_dict()).collect();
        serde_json::to_string_pretty(&data)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialize error: {e}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "PyReviewSession(title='{}', status='{}', annotations={})",
            self.title,
            self.status,
            self.annotations.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Create a new review session.
#[pyfunction]
#[pyo3(signature = (title, content_id, workflow="simple", description=None))]
pub fn create_review(
    title: &str,
    content_id: &str,
    workflow: &str,
    description: Option<&str>,
) -> PyResult<PyReviewSession> {
    PyReviewSession::new(title, content_id, workflow, description)
}

/// Create an annotation.
#[pyfunction]
#[pyo3(signature = (author, message, annotation_type="general", frame=None, timecode=None))]
pub fn annotate(
    author: &str,
    message: &str,
    annotation_type: &str,
    frame: Option<u64>,
    timecode: Option<String>,
) -> PyResult<PyAnnotation> {
    PyAnnotation::new(author, message, annotation_type, frame, timecode)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all review bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAnnotation>()?;
    m.add_class::<PyReviewStatus>()?;
    m.add_class::<PyReviewSession>()?;
    m.add_function(wrap_pyfunction!(create_review, m)?)?;
    m.add_function(wrap_pyfunction!(annotate, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_annotation_creation() {
        let ann = PyAnnotation::new("bob", "Fix color", "issue", Some(1500), None);
        assert!(ann.is_ok());
        let a = ann.expect("should create");
        assert_eq!(a.annotation_type, "issue");
        assert!(!a.resolved);
    }

    #[test]
    fn test_annotation_invalid_type() {
        let result = PyAnnotation::new("bob", "Test", "invalid", None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_annotation_resolve() {
        let mut ann = PyAnnotation::new("bob", "Fix", "issue", None, None).expect("should create");
        assert!(!ann.resolved);
        ann.resolve();
        assert!(ann.resolved);
        ann.unresolve();
        assert!(!ann.resolved);
    }

    #[test]
    fn test_review_session_workflow() {
        let session = PyReviewSession::new("Test Review", "video-123", "simple", None);
        assert!(session.is_ok());
        let s = session.expect("should create");
        let status = s.get_status();
        assert!(status.is_pending());
    }

    #[test]
    fn test_review_session_invalid_workflow() {
        let result = PyReviewSession::new("Test", "v1", "invalid", None);
        assert!(result.is_err());
    }
}
