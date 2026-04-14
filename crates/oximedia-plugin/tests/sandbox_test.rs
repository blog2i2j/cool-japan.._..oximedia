//! Integration tests for sandbox permission enforcement.
//!
//! Verifies that the sandbox correctly blocks operations the plugin does not
//! have permission to perform, and allows operations that are permitted.

use oximedia_plugin::{
    PermissionSet, PluginSandbox, SandboxConfig, SandboxContext, SandboxError, PERM_AUDIO,
    PERM_FILESYSTEM, PERM_GPU, PERM_MEMORY_LARGE, PERM_NETWORK, PERM_VIDEO,
};
use std::time::Duration;

// ── Helper ────────────────────────────────────────────────────────────────────

fn ctx_with_perms(flags: u32) -> SandboxContext {
    SandboxContext::new(SandboxConfig {
        permissions: PermissionSet::new().grant(flags),
        ..SandboxConfig::default()
    })
}

fn ctx_no_perms() -> SandboxContext {
    SandboxContext::new(SandboxConfig::default())
}

fn ctx_all_perms() -> SandboxContext {
    SandboxContext::new(SandboxConfig::permissive())
}

// ── Permission denial tests ────────────────────────────────────────────────

#[test]
fn test_network_denied_without_permission() {
    let ctx = ctx_no_perms();
    let err = ctx.check_permission(PERM_NETWORK).expect_err("should deny");
    assert!(
        matches!(err, SandboxError::PermissionDenied { requested, .. } if requested == PERM_NETWORK)
    );
}

#[test]
fn test_filesystem_denied_without_permission() {
    let ctx = ctx_no_perms();
    let err = ctx
        .check_permission(PERM_FILESYSTEM)
        .expect_err("should deny");
    assert!(matches!(
        err,
        SandboxError::PermissionDenied {
            requested,
            ..
        } if requested == PERM_FILESYSTEM
    ));
}

#[test]
fn test_gpu_denied_without_permission() {
    let ctx = ctx_no_perms();
    assert!(matches!(
        ctx.check_permission(PERM_GPU),
        Err(SandboxError::PermissionDenied { .. })
    ));
}

#[test]
fn test_audio_denied_without_permission() {
    let ctx = ctx_no_perms();
    assert!(matches!(
        ctx.check_permission(PERM_AUDIO),
        Err(SandboxError::PermissionDenied { .. })
    ));
}

#[test]
fn test_video_denied_without_permission() {
    let ctx = ctx_no_perms();
    assert!(matches!(
        ctx.check_permission(PERM_VIDEO),
        Err(SandboxError::PermissionDenied { .. })
    ));
}

#[test]
fn test_memory_large_denied_without_permission() {
    let ctx = ctx_no_perms();
    assert!(matches!(
        ctx.check_permission(PERM_MEMORY_LARGE),
        Err(SandboxError::PermissionDenied { .. })
    ));
}

// ── Permission grant tests ─────────────────────────────────────────────────

#[test]
fn test_network_allowed_with_permission() {
    let ctx = ctx_with_perms(PERM_NETWORK);
    assert!(ctx.check_permission(PERM_NETWORK).is_ok());
    // Other permissions are still denied.
    assert!(ctx.check_permission(PERM_FILESYSTEM).is_err());
}

#[test]
fn test_filesystem_allowed_with_permission() {
    let ctx = ctx_with_perms(PERM_FILESYSTEM);
    assert!(ctx.check_permission(PERM_FILESYSTEM).is_ok());
    assert!(ctx.check_permission(PERM_GPU).is_err());
}

#[test]
fn test_all_permissions_allowed_with_permissive_config() {
    let ctx = ctx_all_perms();
    assert!(ctx.check_permission(PERM_NETWORK).is_ok());
    assert!(ctx.check_permission(PERM_FILESYSTEM).is_ok());
    assert!(ctx.check_permission(PERM_GPU).is_ok());
    assert!(ctx.check_permission(PERM_AUDIO).is_ok());
    assert!(ctx.check_permission(PERM_VIDEO).is_ok());
    assert!(ctx.check_permission(PERM_MEMORY_LARGE).is_ok());
}

// ── Compound permission checks ─────────────────────────────────────────────

#[test]
fn test_compound_permission_all_bits_must_be_granted() {
    let ctx = ctx_with_perms(PERM_NETWORK); // only network
                                            // Requesting NETWORK | FILESYSTEM fails because FILESYSTEM is absent.
    assert!(ctx
        .check_permission(PERM_NETWORK | PERM_FILESYSTEM)
        .is_err());
}

#[test]
fn test_compound_permission_succeeds_when_all_granted() {
    let ctx = ctx_with_perms(PERM_NETWORK | PERM_FILESYSTEM);
    assert!(ctx.check_permission(PERM_NETWORK | PERM_FILESYSTEM).is_ok());
}

// ── Memory limit enforcement ──────────────────────────────────────────────

#[test]
fn test_memory_within_limit_allowed() {
    let ctx = SandboxContext::new(SandboxConfig {
        max_memory_mb: 1,
        ..SandboxConfig::default()
    });
    assert!(ctx.check_memory(512 * 1024).is_ok()); // 512 KiB < 1 MiB
}

#[test]
fn test_memory_exceeds_limit_denied() {
    let ctx = SandboxContext::new(SandboxConfig {
        max_memory_mb: 1,
        ..SandboxConfig::default()
    });
    let err = ctx.check_memory(2 * 1024 * 1024).expect_err("should deny");
    assert!(matches!(err, SandboxError::MemoryExceeded { .. }));
}

#[test]
fn test_memory_cumulative_exceeds_limit_denied() {
    let ctx = SandboxContext::new(SandboxConfig {
        max_memory_mb: 1,
        ..SandboxConfig::default()
    });
    ctx.check_memory(600 * 1024).expect("first");
    let err = ctx
        .check_memory(600 * 1024)
        .expect_err("second should exceed");
    assert!(matches!(err, SandboxError::MemoryExceeded { .. }));
    // First allocation should have been preserved.
    assert_eq!(ctx.used_memory_bytes(), 600 * 1024);
}

// ── Timeout enforcement ───────────────────────────────────────────────────

#[test]
fn test_timeout_not_yet_exceeded_allowed() {
    let ctx = SandboxContext::new(SandboxConfig {
        timeout_ms: 60_000,
        ..SandboxConfig::default()
    });
    assert!(ctx.check_timeout().is_ok());
}

#[test]
fn test_timeout_exceeded_denied() {
    let ctx = SandboxContext::new(SandboxConfig {
        timeout_ms: 0, // Immediately exceeded
        ..SandboxConfig::default()
    });
    std::thread::sleep(Duration::from_millis(1));
    let err = ctx.check_timeout().expect_err("should time out");
    assert!(matches!(err, SandboxError::Timeout { .. }));
}

// ── PluginSandbox::run enforcement ────────────────────────────────────────

#[test]
fn test_sandbox_run_enforces_permission_in_closure() {
    let sb = PluginSandbox::new(SandboxConfig::default()); // no perms
    let result = sb.run(|ctx| {
        ctx.check_permission(PERM_NETWORK)?;
        Ok(())
    });
    assert!(matches!(result, Err(SandboxError::PermissionDenied { .. })));
}

#[test]
fn test_sandbox_run_allows_permitted_operations() {
    let sb = PluginSandbox::new(SandboxConfig {
        permissions: PermissionSet::new().grant(PERM_FILESYSTEM),
        ..SandboxConfig::default()
    });
    let result = sb.run(|ctx| {
        ctx.check_permission(PERM_FILESYSTEM)?;
        ctx.check_memory(1024)?;
        Ok(42u32)
    });
    assert_eq!(result.expect("allowed"), 42);
}

#[test]
fn test_sandbox_run_enforces_memory_limit() {
    let sb = PluginSandbox::new(SandboxConfig {
        permissions: PermissionSet::with_all(),
        max_memory_mb: 1,
        ..SandboxConfig::default()
    });
    let result = sb.run(|ctx| {
        ctx.check_memory(2 * 1024 * 1024)?; // 2 MiB > limit
        Ok(())
    });
    assert!(matches!(result, Err(SandboxError::MemoryExceeded { .. })));
}

// ── Error display messages ────────────────────────────────────────────────

#[test]
fn test_permission_denied_error_displays_hex_flags() {
    let err = SandboxError::PermissionDenied {
        requested: PERM_NETWORK,
        available: 0x00,
    };
    let msg = err.to_string();
    assert!(msg.contains("permission denied"));
    assert!(msg.contains("01")); // hex of PERM_NETWORK
}

#[test]
fn test_memory_exceeded_error_displays_bytes() {
    let err = SandboxError::MemoryExceeded {
        used: 2 * 1024 * 1024,
        limit: 1 * 1024 * 1024,
    };
    let msg = err.to_string();
    assert!(msg.contains("memory limit exceeded"));
}

#[test]
fn test_timeout_error_displays_elapsed_ms() {
    let err = SandboxError::Timeout { elapsed_ms: 7_500 };
    let msg = err.to_string();
    assert!(msg.contains("timeout"));
    assert!(msg.contains("7500"));
}

// ── Fine-grained path enforcement tests ──────────────────────────────────────

use std::path::Path;

/// A filesystem path within the allow-list is permitted.
#[test]
fn test_path_enforcement_allowed_path_succeeds() {
    let ctx = SandboxContext::new(SandboxConfig {
        permissions: PermissionSet::new()
            .grant(PERM_FILESYSTEM)
            .allow_path("/tmp/plugin-work"),
        ..SandboxConfig::default()
    });
    assert!(ctx
        .check_path(Path::new("/tmp/plugin-work/output.bin"))
        .is_ok());
}

/// An exact match on the allowed path is also permitted.
#[test]
fn test_path_enforcement_exact_match_succeeds() {
    let ctx = SandboxContext::new(SandboxConfig {
        permissions: PermissionSet::new()
            .grant(PERM_FILESYSTEM)
            .allow_path("/tmp/plugin-work"),
        ..SandboxConfig::default()
    });
    assert!(ctx.check_path(Path::new("/tmp/plugin-work")).is_ok());
}

/// A path outside the allow-list is denied even with PERM_FILESYSTEM.
#[test]
fn test_path_enforcement_path_outside_allowlist_denied() {
    let ctx = SandboxContext::new(SandboxConfig {
        permissions: PermissionSet::new()
            .grant(PERM_FILESYSTEM)
            .allow_path("/tmp/plugin-work"),
        ..SandboxConfig::default()
    });
    let err = ctx
        .check_path(Path::new("/var/log/system.log"))
        .expect_err("must be denied");
    assert!(
        matches!(err, SandboxError::PathDenied { .. }),
        "expected PathDenied, got {err:?}"
    );
}

/// Without PERM_FILESYSTEM the path check returns PermissionDenied.
#[test]
fn test_path_enforcement_no_fs_perm_gives_permission_denied() {
    let ctx = SandboxContext::new(SandboxConfig {
        permissions: PermissionSet::new().allow_path("/tmp").grant(PERM_NETWORK),
        ..SandboxConfig::default()
    });
    let err = ctx
        .check_path(Path::new("/tmp/allowed-looking-file"))
        .expect_err("must be denied");
    assert!(
        matches!(err, SandboxError::PermissionDenied { .. }),
        "expected PermissionDenied without PERM_FILESYSTEM, got {err:?}"
    );
}

/// Multiple allowed paths — each is accessible, others are not.
#[test]
fn test_path_enforcement_multiple_allowlist_entries() {
    let ctx = SandboxContext::new(SandboxConfig {
        permissions: PermissionSet::new()
            .grant(PERM_FILESYSTEM)
            .allow_path("/tmp/media")
            .allow_path("/var/cache/oximedia"),
        ..SandboxConfig::default()
    });
    assert!(ctx.check_path(Path::new("/tmp/media/frame.png")).is_ok());
    assert!(ctx
        .check_path(Path::new("/var/cache/oximedia/segment.ts"))
        .is_ok());
    assert!(ctx.check_path(Path::new("/etc/passwd")).is_err());
}

/// deny_path removes an entry from the allow-list.
#[test]
fn test_path_enforcement_deny_path_revokes_access() {
    let perms = PermissionSet::new()
        .grant(PERM_FILESYSTEM)
        .allow_path("/tmp/a")
        .allow_path("/tmp/b")
        .deny_path(Path::new("/tmp/a"));
    let ctx = SandboxContext::new(SandboxConfig {
        permissions: perms,
        ..SandboxConfig::default()
    });
    assert!(
        ctx.check_path(Path::new("/tmp/a/file.bin")).is_err(),
        "/tmp/a must be revoked"
    );
    assert!(ctx.check_path(Path::new("/tmp/b/file.bin")).is_ok());
}

/// Empty allow-list (with PERM_FILESYSTEM) permits any path.
#[test]
fn test_path_enforcement_empty_allowlist_permits_all_paths() {
    let ctx = SandboxContext::new(SandboxConfig {
        permissions: PermissionSet::new().grant(PERM_FILESYSTEM),
        ..SandboxConfig::default()
    });
    assert!(ctx.check_path(Path::new("/etc/passwd")).is_ok());
    assert!(ctx.check_path(Path::new("/home/user/.config")).is_ok());
}

// ── CPU quota enforcement tests ───────────────────────────────────────────────

/// CPU quota: a single charge within the limit succeeds.
#[test]
fn test_cpu_enforcement_single_charge_within_limit_ok() {
    let ctx = SandboxContext::new(SandboxConfig {
        max_cpu_ns: 1_000_000,
        ..SandboxConfig::default()
    });
    assert!(ctx.charge_cpu_ns(500_000).is_ok());
    assert_eq!(ctx.used_cpu_ns(), 500_000);
}

/// CPU quota: accumulated charges within limit all succeed.
#[test]
fn test_cpu_enforcement_accumulated_charges_within_limit() {
    let ctx = SandboxContext::new(SandboxConfig {
        max_cpu_ns: 1_000_000,
        ..SandboxConfig::default()
    });
    ctx.charge_cpu_ns(300_000).expect("first");
    ctx.charge_cpu_ns(300_000).expect("second");
    ctx.charge_cpu_ns(300_000).expect("third");
    assert_eq!(ctx.used_cpu_ns(), 900_000);
}

/// CPU quota: exceeding budget returns CpuExceeded and rolls back.
#[test]
fn test_cpu_enforcement_exceeding_budget_denied_and_rolled_back() {
    let ctx = SandboxContext::new(SandboxConfig {
        max_cpu_ns: 100_000,
        ..SandboxConfig::default()
    });
    ctx.charge_cpu_ns(80_000).expect("first");
    let err = ctx.charge_cpu_ns(30_000).expect_err("must exceed");
    assert!(
        matches!(err, SandboxError::CpuExceeded),
        "expected CpuExceeded, got {err:?}"
    );
    assert_eq!(
        ctx.used_cpu_ns(),
        80_000,
        "counter must be rolled back to pre-exceeded value"
    );
}

/// CPU quota: unlimited (max_cpu_ns = 0) never triggers CpuExceeded.
#[test]
fn test_cpu_enforcement_unlimited_never_exceeded() {
    let ctx = SandboxContext::new(SandboxConfig {
        max_cpu_ns: 0,
        ..SandboxConfig::default()
    });
    ctx.charge_cpu_ns(u64::MAX / 2).expect("should not exceed");
    assert_eq!(ctx.used_cpu_ns(), u64::MAX / 2);
}

// ── Combined permission + memory + path enforcement ───────────────────────────

/// Permission OK but memory exceeded still blocks execution.
#[test]
fn test_combined_perm_ok_but_memory_exceeded_blocks() {
    let sb = PluginSandbox::new(SandboxConfig {
        permissions: PermissionSet::new().grant(PERM_FILESYSTEM),
        max_memory_mb: 1,
        ..SandboxConfig::default()
    });
    let result = sb.run(|ctx| {
        ctx.check_permission(PERM_FILESYSTEM)?;
        ctx.check_memory(2 * 1024 * 1024)?;
        Ok(())
    });
    assert!(
        matches!(result, Err(SandboxError::MemoryExceeded { .. })),
        "expected MemoryExceeded but got {result:?}"
    );
}

/// All checks pass in a carefully configured sandbox.
#[test]
fn test_combined_all_checks_pass_in_configured_sandbox() {
    let sb = PluginSandbox::new(SandboxConfig {
        permissions: PermissionSet::new()
            .grant(PERM_FILESYSTEM)
            .grant(PERM_NETWORK)
            .allow_path("/tmp"),
        max_memory_mb: 256,
        max_cpu_ns: 10_000_000,
        timeout_ms: 60_000,
        max_cpu_percent: 100,
    });
    let result = sb.run(|ctx| {
        ctx.check_permission(PERM_FILESYSTEM)?;
        ctx.check_permission(PERM_NETWORK)?;
        ctx.check_path(Path::new("/tmp/output"))?;
        ctx.check_memory(1024)?;
        ctx.charge_cpu_ns(5_000)?;
        ctx.check_timeout()?;
        Ok("all good")
    });
    assert_eq!(result.expect("all checks passed"), "all good");
}

/// PathDenied error display contains the denied path string.
#[test]
fn test_path_denied_error_display_contains_path() {
    let err = SandboxError::PathDenied {
        path: "/etc/shadow".to_string(),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("/etc/shadow"),
        "error message must contain the denied path"
    );
}

/// CpuExceeded error display mentions CPU.
#[test]
fn test_cpu_exceeded_error_display() {
    let err = SandboxError::CpuExceeded;
    assert!(err.to_string().to_lowercase().contains("cpu"));
}
