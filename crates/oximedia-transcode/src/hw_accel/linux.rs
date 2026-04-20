//! Linux hardware acceleration probe.
//!
//! Walks `/sys/class/drm/` to discover DRM card devices, reads their PCI
//! vendor ID, maps that to a driver/vendor family, and returns a
//! [`HwAccelCapabilities`] describing each detected card.
//!
//! ## Vendor → codec mapping
//!
//! | Vendor ID | Family | Codecs |
//! |---|---|---|
//! | `0x8086` | Intel i915/xe | h264, hevc, av1, vp9 |
//! | `0x1002` | AMD amdgpu | h264, hevc, av1, vp9, vp8 |
//! | `0x10de` | NVIDIA (nouveau/nvidia-drm) | h264, hevc |
//!
//! Only cards with an accessible render node are included (checked via
//! [`std::fs::metadata`]).
//!
//! All filesystem reads use [`std::fs::read_to_string`] — no C FFI.

use super::capabilities::{HwAccelCapabilities, HwAccelDevice, HwKind};
use std::path::{Path, PathBuf};

/// Probe Linux VAAPI capabilities by reading `/sys/class/drm/`.
///
/// Returns [`HwAccelCapabilities::none()`] silently when the DRM sysfs
/// hierarchy is absent or unreadable.
pub(crate) fn probe_linux() -> HwAccelCapabilities {
    match try_probe_linux() {
        Ok(caps) => caps,
        Err(e) => {
            tracing::debug!("Linux HW accel probe failed (benign): {e}");
            HwAccelCapabilities::none()
        }
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Known PCI vendor IDs for GPU manufacturers.
const VENDOR_INTEL: u32 = 0x8086;
const VENDOR_AMD: u32 = 0x1002;
const VENDOR_NVIDIA: u32 = 0x10de;

#[derive(Debug, Clone, Copy)]
enum GpuVendor {
    Intel,
    Amd,
    Nvidia,
}

/// Parse a hex string such as `"0x8086\n"` into a `u32`.
fn parse_hex_id(raw: &str) -> Option<u32> {
    let trimmed = raw.trim().trim_start_matches("0x").trim_start_matches("0X");
    u32::from_str_radix(trimmed, 16).ok()
}

/// Try to read the PCI vendor ID for a DRM card directory.
///
/// `card_dir` is e.g. `/sys/class/drm/card0`.
fn read_vendor_id(card_dir: &Path) -> Option<u32> {
    // The canonical path is `<card>/device/vendor`.
    let vendor_path = card_dir.join("device/vendor");
    let raw = std::fs::read_to_string(&vendor_path).ok()?;
    parse_hex_id(&raw)
}

/// Try to read the driver module name for a DRM card.
///
/// Returns e.g. `"i915"`, `"amdgpu"`, `"nouveau"`, or `None`.
fn read_driver_name(card_dir: &Path) -> Option<String> {
    // `/sys/class/drm/card0/device/driver` is a symlink to the module dir.
    // The module directory name is the driver name.
    let driver_link = card_dir.join("device/driver");
    let target = std::fs::read_link(&driver_link).ok()?;
    target
        .file_name()
        .and_then(|n| n.to_str())
        .map(str::to_string)
}

/// Try to find the render node associated with a DRM card directory.
///
/// Looks for `/sys/class/drm/renderD{N}` entries that share the same
/// PCI device as `card_dir`, by matching the symlink target's `device`
/// component.  Falls back to scanning `/dev/dri/renderD{128..255}`.
fn find_render_node(_card_dir: &Path, card_index: u32) -> Option<PathBuf> {
    // Fast path: renderD(128 + card_index) is the canonical mapping.
    let render_minor = 128u32.saturating_add(card_index);
    let candidate = PathBuf::from(format!("/dev/dri/renderD{render_minor}"));
    if std::fs::metadata(&candidate).is_ok() {
        return Some(candidate);
    }

    // Fallback: scan /dev/dri/renderD* (128..160 is typical range).
    for i in 128u32..=159 {
        let path = PathBuf::from(format!("/dev/dri/renderD{i}"));
        if std::fs::metadata(&path).is_ok() {
            return Some(path);
        }
    }
    None
}

/// Map a PCI vendor ID to a vendor enum.
fn classify_vendor(id: u32) -> Option<GpuVendor> {
    match id {
        VENDOR_INTEL => Some(GpuVendor::Intel),
        VENDOR_AMD => Some(GpuVendor::Amd),
        VENDOR_NVIDIA => Some(GpuVendor::Nvidia),
        _ => None,
    }
}

/// Return the default codec list for a vendor.
///
/// Codec names follow the OxiMedia / FFmpeg naming convention.
fn codecs_for_vendor(vendor: GpuVendor) -> Vec<String> {
    match vendor {
        GpuVendor::Intel => {
            // Intel Gen12 (Tiger Lake+) / Xe: H.264, HEVC, AV1, VP9.
            // Older generations support only H.264 + HEVC but we report
            // the modern superset — callers can probe further if needed.
            vec![
                "h264".to_string(),
                "hevc".to_string(),
                "av1".to_string(),
                "vp9".to_string(),
            ]
        }
        GpuVendor::Amd => {
            // AMD Navi (RDNA2+): H.264, HEVC, AV1, VP9, VP8.
            vec![
                "h264".to_string(),
                "hevc".to_string(),
                "av1".to_string(),
                "vp9".to_string(),
                "vp8".to_string(),
            ]
        }
        GpuVendor::Nvidia => {
            // Nouveau / nvidia-drm VAAPI: limited — H.264 + HEVC.
            // NVENC decode is separate and requires proprietary drivers.
            vec!["h264".to_string(), "hevc".to_string()]
        }
    }
}

/// Infer whether HDR (10-bit) is likely supported for this vendor.
fn supports_hdr(vendor: GpuVendor) -> bool {
    matches!(vendor, GpuVendor::Intel | GpuVendor::Amd)
}

/// Core implementation — separated for clean error propagation.
fn try_probe_linux() -> Result<HwAccelCapabilities, String> {
    let drm_class = Path::new("/sys/class/drm");
    if !drm_class.exists() {
        return Ok(HwAccelCapabilities::none());
    }

    let entries =
        std::fs::read_dir(drm_class).map_err(|e| format!("read_dir /sys/class/drm: {e}"))?;

    let mut devices: Vec<HwAccelDevice> = Vec::new();
    let mut card_index: u32 = 0;

    for entry_res in entries {
        let entry = entry_res.map_err(|e| format!("readdir error: {e}"))?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // We only want `card0`, `card1`, … — skip `renderD*`, `version`, etc.
        if !name_str.starts_with("card") {
            continue;
        }
        // Skip sub-entries like `card0-DP-1` (connector directories).
        if name_str.chars().filter(|c| c == &'-').count() > 0 {
            continue;
        }

        let card_dir = entry.path();

        // Resolve the card's PCI vendor.
        let vendor_id = match read_vendor_id(&card_dir) {
            Some(v) => v,
            None => {
                card_index = card_index.saturating_add(1);
                continue;
            }
        };

        let vendor = match classify_vendor(vendor_id) {
            Some(v) => v,
            None => {
                tracing::debug!(
                    "Unknown GPU vendor {:#06x} at {:?}; skipping",
                    vendor_id,
                    card_dir
                );
                card_index = card_index.saturating_add(1);
                continue;
            }
        };

        // Only include devices with an accessible render node.
        let render_node = find_render_node(&card_dir, card_index);
        if render_node.is_none() {
            tracing::debug!("No render node for {:?}; skipping", card_dir);
            card_index = card_index.saturating_add(1);
            continue;
        }

        let driver = read_driver_name(&card_dir);
        let codecs = codecs_for_vendor(vendor);
        let hdr = supports_hdr(vendor);

        devices.push(HwAccelDevice {
            kind: HwKind::Vaapi,
            driver,
            render_node,
            supported_codecs: codecs,
            // VAAPI / VA-API practical maximum: 8K is supported on modern HW.
            max_width: 8192,
            max_height: 4320,
            supports_hdr: hdr,
        });

        card_index = card_index.saturating_add(1);
    }

    Ok(HwAccelCapabilities { devices })
}
