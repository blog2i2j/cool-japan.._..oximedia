//! REST API for batch processing

use crate::error::BatchError;
use crate::job::BatchJob;
use crate::{BatchEngine, JobId};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// API server state
#[derive(Clone)]
pub struct ApiState {
    engine: Arc<BatchEngine>,
}

/// Job submission request
#[derive(Debug, Serialize, Deserialize)]
pub struct SubmitJobRequest {
    job: BatchJob,
}

/// Job submission response
#[derive(Debug, Serialize, Deserialize)]
pub struct SubmitJobResponse {
    job_id: String,
}

/// Job status response
#[derive(Debug, Serialize, Deserialize)]
pub struct JobStatusResponse {
    job_id: String,
    status: String,
}

/// API router
pub fn create_router(engine: Arc<BatchEngine>) -> Router {
    let state = ApiState { engine };

    Router::new()
        .route("/api/v1/jobs", post(submit_job))
        .route("/api/v1/jobs", get(list_jobs))
        .route("/api/v1/jobs/{id}", get(get_job_status))
        .route("/api/v1/jobs/{id}", delete(cancel_job))
        .route("/api/v1/health", get(health_check))
        .with_state(state)
}

async fn submit_job(
    State(state): State<ApiState>,
    Json(request): Json<SubmitJobRequest>,
) -> std::result::Result<impl IntoResponse, (StatusCode, String)> {
    let job_id = state
        .engine
        .submit_job(request.job)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(SubmitJobResponse {
        job_id: job_id.to_string(),
    }))
}

async fn list_jobs(
    State(state): State<ApiState>,
) -> std::result::Result<impl IntoResponse, (StatusCode, String)> {
    let jobs = state
        .engine
        .list_jobs()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(jobs))
}

async fn get_job_status(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> std::result::Result<impl IntoResponse, (StatusCode, String)> {
    let job_id = JobId::from_string(id.clone());
    let status = state
        .engine
        .get_job_status(&job_id)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    Ok(Json(JobStatusResponse {
        job_id: id,
        status: status.to_string(),
    }))
}

async fn cancel_job(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> std::result::Result<impl IntoResponse, (StatusCode, String)> {
    let job_id = JobId::from_string(id);
    state
        .engine
        .cancel_job(&job_id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(StatusCode::NO_CONTENT)
}

async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "service": "oximedia-batch"
    }))
}

/// Start the API server
///
/// # Arguments
///
/// * `engine` - Batch engine instance
/// * `addr` - Server address (e.g., "0.0.0.0:3000")
///
/// # Errors
///
/// Returns an error if server startup fails
pub async fn start_server(engine: Arc<BatchEngine>, addr: &str) -> crate::error::Result<()> {
    let app = create_router(engine);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| BatchError::ApiError(e.to_string()))?;

    tracing::info!("API server listening on {}", addr);

    axum::serve(listener, app)
        .await
        .map_err(|e| BatchError::ApiError(e.to_string()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_create_router() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let engine = Arc::new(BatchEngine::new(db_path, 2).expect("failed to create"));

        let router = create_router(engine);
        assert!(std::mem::size_of_val(&router) > 0);
    }

    #[tokio::test]
    async fn test_health_check() {
        let response = health_check().await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
