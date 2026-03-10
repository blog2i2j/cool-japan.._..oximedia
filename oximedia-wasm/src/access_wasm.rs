//! WebAssembly bindings for access control checking.
//!
//! Provides WASM-accessible functions for checking permissions,
//! listing policies, and querying access roles.

use wasm_bindgen::prelude::*;

/// Check if a permission is valid for a given context.
///
/// `request_json`: JSON object with keys:
///   - `asset` (string): asset identifier
///   - `principal` (string): user/group identifier
///   - `permission` (string): read, write, admin, publish, review
///   - `territory` (string, optional): ISO 3166 code
///
/// Returns a JSON object with the permission check result.
#[wasm_bindgen]
pub fn wasm_check_permission(request_json: &str) -> Result<String, JsValue> {
    let request: serde_json::Value = serde_json::from_str(request_json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid request JSON: {e}")))?;

    let asset = request
        .get("asset")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let principal = request
        .get("principal")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let permission = request
        .get("permission")
        .and_then(|v| v.as_str())
        .unwrap_or("read");
    let territory = request.get("territory").and_then(|v| v.as_str());

    // Validate permission
    let valid_permissions = ["read", "write", "admin", "publish", "review"];
    if !valid_permissions.contains(&permission) {
        return Err(crate::utils::js_err(&format!(
            "Invalid permission '{}'. Expected: {}",
            permission,
            valid_permissions.join(", ")
        )));
    }

    let result = serde_json::json!({
        "asset": asset,
        "principal": principal,
        "permission": permission,
        "territory": territory,
        "allowed": false,
        "reason": "No matching grant (client-side check only, server validation required)",
        "checked_at": "2026-01-01T00:00:00Z",
    });

    serde_json::to_string(&result)
        .map_err(|e| crate::utils::js_err(&format!("Serialization error: {e}")))
}

/// List available access policies and their descriptions.
///
/// Returns a JSON array of policy descriptors.
#[wasm_bindgen]
pub fn wasm_list_policies() -> String {
    let policies = serde_json::json!([
        {
            "name": "public-read",
            "description": "Public read-only access",
            "default_permission": "read",
            "require_mfa": false,
            "wcag_level": "AA",
        },
        {
            "name": "team-edit",
            "description": "Team members can read and write",
            "default_permission": "write",
            "require_mfa": false,
            "wcag_level": "AA",
        },
        {
            "name": "restricted",
            "description": "Explicit grants required for all access",
            "default_permission": "none",
            "require_mfa": true,
            "wcag_level": "AA",
        },
        {
            "name": "compliance-strict",
            "description": "Strict compliance policy with MFA and IP restrictions",
            "default_permission": "none",
            "require_mfa": true,
            "wcag_level": "AAA",
        },
        {
            "name": "broadcast-standard",
            "description": "Standard broadcast access with territory awareness",
            "default_permission": "read",
            "require_mfa": false,
            "wcag_level": "AA",
        },
    ]);
    serde_json::to_string(&policies).unwrap_or_else(|_| "[]".to_string())
}

/// List available access roles and their permissions.
///
/// Returns a JSON array of role descriptors.
#[wasm_bindgen]
pub fn wasm_access_roles() -> String {
    let roles = serde_json::json!([
        {
            "role": "viewer",
            "description": "Can view and stream media assets",
            "permissions": ["read"],
            "inherits": [],
        },
        {
            "role": "editor",
            "description": "Can edit, annotate, and create versions",
            "permissions": ["read", "write"],
            "inherits": ["viewer"],
        },
        {
            "role": "reviewer",
            "description": "Can review, comment, and approve",
            "permissions": ["read", "review"],
            "inherits": ["viewer"],
        },
        {
            "role": "publisher",
            "description": "Can publish and distribute content",
            "permissions": ["read", "write", "publish"],
            "inherits": ["editor"],
        },
        {
            "role": "administrator",
            "description": "Full access including permission management",
            "permissions": ["read", "write", "admin", "publish", "review"],
            "inherits": ["publisher", "reviewer"],
        },
    ]);
    serde_json::to_string(&roles).unwrap_or_else(|_| "[]".to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_permission_valid() {
        let request = r#"{"asset":"video-001","principal":"user@test.com","permission":"read"}"#;
        let result = wasm_check_permission(request);
        assert!(result.is_ok());
        let json = result.expect("should check");
        assert!(json.contains("\"allowed\":false"));
        assert!(json.contains("video-001"));
    }

    #[test]
    fn test_check_permission_invalid_level() {
        let request = r#"{"asset":"v1","principal":"u1","permission":"superadmin"}"#;
        let result = wasm_check_permission(request);
        assert!(result.is_err());
    }

    #[test]
    fn test_check_permission_with_territory() {
        let request = r#"{"asset":"v1","principal":"u1","permission":"read","territory":"US"}"#;
        let result = wasm_check_permission(request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_list_policies() {
        let policies = wasm_list_policies();
        assert!(policies.contains("public-read"));
        assert!(policies.contains("restricted"));
        assert!(policies.contains("compliance-strict"));
    }

    #[test]
    fn test_access_roles() {
        let roles = wasm_access_roles();
        assert!(roles.contains("viewer"));
        assert!(roles.contains("editor"));
        assert!(roles.contains("administrator"));
    }
}
