//! macOS hardware acceleration probe.
//!
//! Uses `sysctl` to identify the chip family, then applies a static
//! VideoToolbox codec table.  No C FFI or Objective-C runtimes are required.
//!
//! ## Chip detection algorithm
//!
//! 1. Query `sysctl -n hw.optional.arm64` — `"1"` means Apple Silicon.
//! 2. Query `sysctl -n machdep.cpu.brand_string` — parse `M1`/`M2`/`M3`/`M4`
//!    from the brand string.
//! 3. Apply the codec table below.
//!
//! | Chip family | Decode | Encode |
//! |---|---|---|
//! | M1 / M2 | h264, hevc | h264, hevc |
//! | M3 / M4 | h264, hevc, av1 | h264, hevc |
//! | Intel Mac | h264, hevc | h264 |
//! | Unknown | (empty) | |
//!
//! VideoToolbox max resolution is fixed at 8192 × 4320 (Apple platform limit).

use super::capabilities::{HwAccelCapabilities, HwAccelDevice, HwKind};
use std::process::Command;

/// Probe macOS VideoToolbox capabilities via `sysctl`.
///
/// Returns [`HwAccelCapabilities::none()`] silently on any error.
pub(crate) fn probe_macos() -> HwAccelCapabilities {
    match try_probe_macos() {
        Ok(caps) => caps,
        Err(e) => {
            tracing::debug!("macOS HW accel probe failed (benign): {e}");
            HwAccelCapabilities::none()
        }
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Possible Apple Silicon / Intel chip families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChipFamily {
    /// Apple M1 or M2 (no AV1 decode acceleration).
    AppleM1M2,
    /// Apple M3 or M4 (hardware AV1 decode engine).
    AppleM3M4,
    /// Intel Mac (limited VideoToolbox, no AV1).
    IntelMac,
    /// Unknown / undetectable.
    Unknown,
}

fn sysctl_n(key: &str) -> Result<String, String> {
    let out = Command::new("sysctl")
        .arg("-n")
        .arg(key)
        .output()
        .map_err(|e| format!("sysctl spawn failed: {e}"))?;

    if !out.status.success() {
        return Err(format!("sysctl -n {key} exited with status {}", out.status));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn detect_chip_family() -> ChipFamily {
    // ------------------------------------------------------------------
    // Step 1: Is this Apple Silicon?
    // ------------------------------------------------------------------
    let is_arm64 = sysctl_n("hw.optional.arm64")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(false);

    if is_arm64 {
        // ------------------------------------------------------------------
        // Step 2: Identify M-generation from brand string.
        //
        // Typical strings:
        //   "Apple M1"
        //   "Apple M2 Pro"
        //   "Apple M3 Max"
        //   "Apple M4"
        // ------------------------------------------------------------------
        let brand = sysctl_n("machdep.cpu.brand_string").unwrap_or_default();
        let upper = brand.to_uppercase();
        // M3 / M4 have AV1 hardware decode.
        if upper.contains("M3") || upper.contains("M4") {
            return ChipFamily::AppleM3M4;
        }
        // M1 / M2 do not have AV1 hardware.
        if upper.contains("M1") || upper.contains("M2") {
            return ChipFamily::AppleM1M2;
        }
        // ARM but unrecognised generation — be conservative.
        return ChipFamily::AppleM1M2;
    }

    // ------------------------------------------------------------------
    // Step 3: Intel Mac — VideoToolbox is still present.
    // ------------------------------------------------------------------
    let brand = sysctl_n("machdep.cpu.brand_string").unwrap_or_default();
    if brand.to_uppercase().contains("INTEL") {
        return ChipFamily::IntelMac;
    }

    ChipFamily::Unknown
}

fn try_probe_macos() -> Result<HwAccelCapabilities, String> {
    let chip = detect_chip_family();

    if chip == ChipFamily::Unknown {
        return Ok(HwAccelCapabilities::none());
    }

    let (codecs, supports_hdr) = match chip {
        ChipFamily::AppleM1M2 => (
            // VideoToolbox on M1/M2: h264 + hevc encode & decode.
            // HDR (10-bit HEVC) is supported.
            vec!["h264".to_string(), "hevc".to_string()],
            true,
        ),
        ChipFamily::AppleM3M4 => (
            // M3/M4 add hardware AV1 decode (encode remains software only).
            vec!["h264".to_string(), "hevc".to_string(), "av1".to_string()],
            true,
        ),
        ChipFamily::IntelMac => (
            // Intel VideoToolbox: h264 decode/encode; hevc decode only on
            // 6th-gen+. AV1 is software-only. We advertise hevc conservatively.
            vec!["h264".to_string(), "hevc".to_string()],
            false,
        ),
        ChipFamily::Unknown => unreachable!("handled above"),
    };

    let device = HwAccelDevice {
        kind: HwKind::VideoToolbox,
        driver: None,
        render_node: None,
        supported_codecs: codecs,
        // VideoToolbox maximum is 8 K (8192 × 4320) per Apple documentation.
        max_width: 8192,
        max_height: 4320,
        supports_hdr,
    };

    Ok(HwAccelCapabilities {
        devices: vec![device],
    })
}
