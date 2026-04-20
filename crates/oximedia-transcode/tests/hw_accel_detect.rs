//! Integration tests for the HW acceleration detection subsystem.
//!
//! All tests use [`MockProbe`] so they run deterministically on every CI
//! platform.  The optional live-system test requires
//! `OXIMEDIA_TEST_LIVE_HW=1`.

use oximedia_transcode::{
    detect_hw_accel_with_probe, HwAccelCapabilities, HwAccelDevice, HwKind, MockProbe,
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn make_vt_device(codecs: Vec<&str>, supports_hdr: bool) -> HwAccelDevice {
    HwAccelDevice {
        kind: HwKind::VideoToolbox,
        driver: None,
        render_node: None,
        supported_codecs: codecs.into_iter().map(str::to_string).collect(),
        max_width: 8192,
        max_height: 4320,
        supports_hdr,
    }
}

fn make_vaapi_device(codecs: Vec<&str>, driver: &str) -> HwAccelDevice {
    HwAccelDevice {
        kind: HwKind::Vaapi,
        driver: Some(driver.to_string()),
        render_node: Some(std::path::PathBuf::from("/dev/dri/renderD128")),
        supported_codecs: codecs.into_iter().map(str::to_string).collect(),
        max_width: 8192,
        max_height: 4320,
        supports_hdr: true,
    }
}

fn make_m1_caps() -> HwAccelCapabilities {
    HwAccelCapabilities {
        devices: vec![make_vt_device(vec!["h264", "hevc"], true)],
    }
}

fn make_m3_caps() -> HwAccelCapabilities {
    HwAccelCapabilities {
        devices: vec![make_vt_device(vec!["h264", "hevc", "av1"], true)],
    }
}

fn make_intel_gen12_caps() -> HwAccelCapabilities {
    HwAccelCapabilities {
        devices: vec![make_vaapi_device(
            vec!["h264", "hevc", "av1", "vp9"],
            "i915",
        )],
    }
}

fn make_amd_navi_caps() -> HwAccelCapabilities {
    HwAccelCapabilities {
        devices: vec![make_vaapi_device(
            vec!["h264", "hevc", "av1", "vp9", "vp8"],
            "amdgpu",
        )],
    }
}

// ─── Mock tests ───────────────────────────────────────────────────────────────

#[test]
fn hw_accel_mock_m1_has_hevc() {
    let caps = detect_hw_accel_with_probe(&MockProbe(make_m1_caps()));
    assert!(!caps.is_empty(), "M1 caps should not be empty");
    let device = caps
        .device_for_codec("hevc")
        .expect("M1 should support hevc");
    assert_eq!(device.kind, HwKind::VideoToolbox);
    assert!(device.supports_hdr, "M1 HEVC should support HDR");
}

#[test]
fn hw_accel_mock_m1_does_not_have_av1() {
    let caps = detect_hw_accel_with_probe(&MockProbe(make_m1_caps()));
    assert!(
        caps.device_for_codec("av1").is_none(),
        "M1 should NOT support AV1 HW decode"
    );
}

#[test]
fn hw_accel_mock_m3_has_av1() {
    let caps = detect_hw_accel_with_probe(&MockProbe(make_m3_caps()));
    let device = caps
        .device_for_codec("av1")
        .expect("M3 should support AV1 HW decode");
    assert_eq!(device.kind, HwKind::VideoToolbox);
    assert_eq!(device.max_width, 8192);
    assert_eq!(device.max_height, 4320);
}

#[test]
fn hw_accel_mock_m3_has_hevc() {
    let caps = detect_hw_accel_with_probe(&MockProbe(make_m3_caps()));
    assert!(
        caps.device_for_codec("hevc").is_some(),
        "M3 should also support HEVC"
    );
}

#[test]
fn hw_accel_mock_intel_gen12_has_av1() {
    let caps = detect_hw_accel_with_probe(&MockProbe(make_intel_gen12_caps()));
    let device = caps
        .device_for_codec("av1")
        .expect("Intel Gen12 should support AV1 via VAAPI");
    assert_eq!(device.kind, HwKind::Vaapi);
    assert_eq!(device.driver.as_deref(), Some("i915"));
    assert!(
        device.render_node.is_some(),
        "VAAPI device should have a render node"
    );
}

#[test]
fn hw_accel_mock_intel_gen12_has_vp9() {
    let caps = detect_hw_accel_with_probe(&MockProbe(make_intel_gen12_caps()));
    assert!(caps.device_for_codec("vp9").is_some());
}

#[test]
fn hw_accel_mock_empty_no_panic() {
    let probe = MockProbe(HwAccelCapabilities::none());
    let caps = detect_hw_accel_with_probe(&probe);
    assert!(caps.is_empty());
    assert!(caps.device_for_codec("h264").is_none());
    assert!(caps.device_for_codec("av1").is_none());
}

#[test]
fn hw_accel_mock_amd_navi_has_hevc() {
    let caps = detect_hw_accel_with_probe(&MockProbe(make_amd_navi_caps()));
    let device = caps
        .device_for_codec("hevc")
        .expect("AMD Navi should support HEVC via VAAPI");
    assert_eq!(device.kind, HwKind::Vaapi);
    assert_eq!(device.driver.as_deref(), Some("amdgpu"));
}

#[test]
fn hw_accel_mock_amd_navi_has_vp8() {
    let caps = detect_hw_accel_with_probe(&MockProbe(make_amd_navi_caps()));
    assert!(
        caps.device_for_codec("vp8").is_some(),
        "AMD Navi should support VP8 via VAAPI"
    );
}

#[test]
fn hw_accel_mock_codec_comparison_is_case_insensitive() {
    let caps = detect_hw_accel_with_probe(&MockProbe(make_m3_caps()));
    // "HEVC" and "hevc" should both match.
    assert!(caps.device_for_codec("HEVC").is_some());
    assert!(caps.device_for_codec("hevc").is_some());
    assert!(caps.device_for_codec("AV1").is_some());
}

#[test]
fn hw_accel_mock_device_max_resolution() {
    let caps = detect_hw_accel_with_probe(&MockProbe(make_m1_caps()));
    let device = &caps.devices[0];
    assert_eq!(
        device.max_width, 8192,
        "VideoToolbox max width should be 8192"
    );
    assert_eq!(
        device.max_height, 4320,
        "VideoToolbox max height should be 4320"
    );
}

// ─── Live system test (opt-in) ────────────────────────────────────────────────

#[cfg(target_os = "macos")]
#[test]
fn hw_accel_live_system_probe() {
    if std::env::var("OXIMEDIA_TEST_LIVE_HW").is_err() {
        // Skip unless explicitly enabled.
        return;
    }
    let caps = oximedia_transcode::detect_hw_accel_caps();
    assert!(
        !caps.is_empty(),
        "Live macOS system probe should find at least one VideoToolbox device"
    );
    let vt = caps
        .devices
        .iter()
        .find(|d| d.kind == HwKind::VideoToolbox)
        .expect("macOS should always have a VideoToolbox device");
    assert!(
        vt.supported_codecs.contains(&"h264".to_string()),
        "VideoToolbox should always support H.264"
    );
}
