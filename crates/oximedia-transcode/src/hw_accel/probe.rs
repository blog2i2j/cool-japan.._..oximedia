//! Hardware probe trait and implementation stubs.
//!
//! [`HwProbe`] is a `Send + Sync` trait that abstracts platform detection.
//! [`SystemProbe`] dispatches to the OS-specific implementation at compile
//! time.  [`MockProbe`] is for use in unit tests.

use super::capabilities::HwAccelCapabilities;

// ─── HwProbe trait ───────────────────────────────────────────────────────────

/// Abstraction over hardware acceleration probing.
///
/// Implement this trait (or use [`MockProbe`]) to supply capability data
/// without touching OS APIs — useful for unit tests and deterministic CI.
pub trait HwProbe: Send + Sync {
    /// Probe the system and return discovered capabilities.
    fn probe(&self) -> HwAccelCapabilities;
}

// ─── SystemProbe ─────────────────────────────────────────────────────────────

/// Real system probe — dispatches to OS-specific code at compile time.
///
/// On macOS: parses `sysctl` output to identify the Apple Silicon/Intel
/// chip family, then returns a static codec table.
///
/// On Linux: walks `/sys/class/drm/` to find DRM devices, resolves vendor
/// IDs to driver families, and maps those to supported codecs.
///
/// On Windows / WASM / other: returns [`HwAccelCapabilities::none()`].
pub struct SystemProbe;

// ─── MockProbe ───────────────────────────────────────────────────────────────

/// A test double that returns a pre-built capability set.
///
/// ```
/// use oximedia_transcode::{HwAccelCapabilities, MockProbe, HwProbe};
///
/// let probe = MockProbe(HwAccelCapabilities::none());
/// assert!(probe.probe().is_empty());
/// ```
pub struct MockProbe(pub HwAccelCapabilities);

impl HwProbe for MockProbe {
    fn probe(&self) -> HwAccelCapabilities {
        self.0.clone()
    }
}

// ─── Platform dispatch ────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
impl HwProbe for SystemProbe {
    fn probe(&self) -> HwAccelCapabilities {
        crate::hw_accel::macos::probe_macos()
    }
}

#[cfg(target_os = "linux")]
impl HwProbe for SystemProbe {
    fn probe(&self) -> HwAccelCapabilities {
        crate::hw_accel::linux::probe_linux()
    }
}

#[cfg(target_os = "windows")]
impl HwProbe for SystemProbe {
    fn probe(&self) -> HwAccelCapabilities {
        tracing::warn!(
            "HW accel detection not yet implemented on Windows; \
             returning empty capabilities"
        );
        HwAccelCapabilities::none()
    }
}

#[cfg(target_arch = "wasm32")]
impl HwProbe for SystemProbe {
    fn probe(&self) -> HwAccelCapabilities {
        HwAccelCapabilities::none()
    }
}

#[cfg(not(any(
    target_os = "macos",
    target_os = "linux",
    target_os = "windows",
    target_arch = "wasm32"
)))]
impl HwProbe for SystemProbe {
    fn probe(&self) -> HwAccelCapabilities {
        HwAccelCapabilities::none()
    }
}
