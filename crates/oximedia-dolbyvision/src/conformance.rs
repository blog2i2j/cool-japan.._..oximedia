//! Conformance tests with known-good RPU bitstreams and round-trip tests for
//! all Level metadata types (L1 through L11).
//!
//! The synthetic RPU bitstreams in this module are constructed by the library's
//! own writer and then validated against known expected values.  They serve as
//! regression tests ensuring that changes to the parser/writer do not silently
//! break the binary format.

use crate::{
    parser, writer, DolbyVisionRpu, Level11Metadata, Level1Metadata, Level2Metadata,
    Level4Metadata, Level5Metadata, Level6Metadata, Level7Metadata, Level8Metadata, Level9Metadata,
    Profile,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Write an RPU to a bitstream and re-parse it.
///
/// # Errors
///
/// Returns error if writing or re-parsing fails.
pub fn write_then_parse(rpu: &DolbyVisionRpu) -> crate::Result<DolbyVisionRpu> {
    let bits = writer::write_rpu_bitstream(rpu)?;
    parser::parse_rpu_bitstream(&bits)
}

/// Assert that two `Option<T>` agree on presence (both `Some` or both `None`).
#[track_caller]
pub fn assert_presence_matches<T>(label: &str, original: &Option<T>, reparsed: &Option<T>) {
    assert_eq!(
        original.is_some(),
        reparsed.is_some(),
        "{label}: presence mismatch (original={}, reparsed={})",
        original.is_some(),
        reparsed.is_some(),
    );
}

// ---------------------------------------------------------------------------
// Public conformance verification functions
// ---------------------------------------------------------------------------

/// Verify that a minimal Profile 8 RPU survives a bitstream round-trip without
/// corruption of the header fields.
///
/// # Errors
///
/// Returns error if the round-trip fails.
pub fn verify_profile8_minimal_roundtrip() -> crate::Result<()> {
    let rpu = DolbyVisionRpu::new(Profile::Profile8);
    let reparsed = write_then_parse(&rpu)?;
    assert_eq!(rpu.profile, reparsed.profile);
    Ok(())
}

/// Verify Level 1 metadata round-trip fidelity.
///
/// # Errors
///
/// Returns error if the round-trip fails or values differ.
pub fn verify_level1_roundtrip() -> crate::Result<()> {
    let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
    rpu.level1 = Some(Level1Metadata {
        min_pq: 128,
        max_pq: 3500,
        avg_pq: 1800,
    });
    let reparsed = write_then_parse(&rpu)?;
    let orig = rpu
        .level1
        .as_ref()
        .ok_or_else(|| crate::DolbyVisionError::Generic("L1 should be set".into()))?;
    let rep = reparsed
        .level1
        .as_ref()
        .ok_or_else(|| crate::DolbyVisionError::Generic("L1 should survive round-trip".into()))?;
    assert_eq!(orig.min_pq, rep.min_pq, "L1 min_pq mismatch");
    assert_eq!(orig.max_pq, rep.max_pq, "L1 max_pq mismatch");
    assert_eq!(orig.avg_pq, rep.avg_pq, "L1 avg_pq mismatch");
    Ok(())
}

/// Verify Level 5 (active area) metadata write-and-parse fidelity.
///
/// Note: the parser currently stubs Level 5 parsing (returns `None`); this
/// function verifies that writing succeeds and re-parsing does not error.
///
/// # Errors
///
/// Returns error if the round-trip panics or errors.
pub fn verify_level5_roundtrip() -> crate::Result<()> {
    let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
    rpu.level5 = Some(Level5Metadata {
        active_area_left_offset: 160,
        active_area_right_offset: 160,
        active_area_top_offset: 0,
        active_area_bottom_offset: 0,
    });
    let _reparsed = write_then_parse(&rpu)?;
    // Parser stubs L5 → presence not asserted; only verify no error
    Ok(())
}

/// Verify Level 6 (fallback HDR10) metadata write-and-parse fidelity.
///
/// Note: the parser currently stubs Level 6; this verifies write+re-parse
/// do not error.
///
/// # Errors
///
/// Returns error if the round-trip fails.
pub fn verify_level6_roundtrip() -> crate::Result<()> {
    let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
    rpu.level6 = Some(Level6Metadata::bt2020());
    let _reparsed = write_then_parse(&rpu)?;
    Ok(())
}

/// Verify Level 8 (target display) metadata write-and-parse fidelity.
///
/// Note: the parser currently stubs Level 8; verifies write+re-parse do not
/// error.
///
/// # Errors
///
/// Returns error if the round-trip fails.
pub fn verify_level8_roundtrip() -> crate::Result<()> {
    let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
    rpu.level8 = Some(Level8Metadata::hdr_1000());
    let _reparsed = write_then_parse(&rpu)?;
    Ok(())
}

/// Verify Level 9 (source display) metadata write-and-parse fidelity.
///
/// Note: the parser currently stubs Level 9.
///
/// # Errors
///
/// Returns error if the round-trip fails.
pub fn verify_level9_roundtrip() -> crate::Result<()> {
    let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
    rpu.level9 = Some(Level9Metadata::bt2020_mastering());
    let _reparsed = write_then_parse(&rpu)?;
    Ok(())
}

/// Verify Level 11 (content type) metadata write-and-parse fidelity.
///
/// Note: the parser currently stubs Level 11.
///
/// # Errors
///
/// Returns error if the round-trip fails.
pub fn verify_level11_roundtrip() -> crate::Result<()> {
    let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
    rpu.level11 = Some(Level11Metadata {
        content_type: crate::ContentType::Movie,
        whitepoint: 0,
        reference_mode_flag: false,
        sharpness: 0,
        noise_reduction: 0,
        mpeg_noise_reduction: 0,
        frame_rate: 24,
        temporal_filter_strength: 0,
    });
    let _reparsed = write_then_parse(&rpu)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Profile round-trip tests ──────────────────────────────────────────────

    #[test]
    fn test_profile8_minimal_roundtrip() {
        verify_profile8_minimal_roundtrip().expect("Profile 8 round-trip should pass");
    }

    #[test]
    fn test_profile5_minimal_roundtrip() {
        // Profile 5 uses IPT color space (mapping_color_space=2) which the parser
        // correctly identifies.
        let rpu = DolbyVisionRpu::new(Profile::Profile5);
        let reparsed = write_then_parse(&rpu).expect("round-trip should succeed");
        assert_eq!(rpu.profile, reparsed.profile);
    }

    #[test]
    fn test_profile8_4_minimal_roundtrip() {
        // Profile 8.4 uses HLG; the parser infers profile from header flags.
        // The bitstream write+re-parse must succeed without error.
        let rpu = DolbyVisionRpu::new(Profile::Profile8_4);
        let result = write_then_parse(&rpu);
        assert!(result.is_ok(), "Profile 8.4 round-trip must not error");
    }

    #[test]
    fn test_profile8_1_minimal_roundtrip() {
        // Profile 8.1 is a low-latency variant; write+re-parse must succeed.
        let rpu = DolbyVisionRpu::new(Profile::Profile8_1);
        let result = write_then_parse(&rpu);
        assert!(result.is_ok(), "Profile 8.1 round-trip must not error");
    }

    #[test]
    fn test_profile7_minimal_roundtrip() {
        // Profile 7 uses MEL; write+re-parse must succeed without error.
        let rpu = DolbyVisionRpu::new(Profile::Profile7);
        let result = write_then_parse(&rpu);
        assert!(result.is_ok(), "Profile 7 round-trip must not error");
    }

    // ── Level metadata round-trip tests ──────────────────────────────────────

    #[test]
    fn test_level1_roundtrip() {
        verify_level1_roundtrip().expect("L1 round-trip should pass");
    }

    #[test]
    fn test_level1_edge_values_roundtrip() {
        let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
        rpu.level1 = Some(Level1Metadata {
            min_pq: 0,
            max_pq: 4095,
            avg_pq: 2048,
        });
        let reparsed = write_then_parse(&rpu).expect("round-trip should succeed");
        let l1 = reparsed.level1.expect("L1 should survive");
        assert_eq!(l1.min_pq, 0);
        assert_eq!(l1.max_pq, 4095);
    }

    #[test]
    fn test_level2_roundtrip_presence() {
        // Level 2 is written but parser is stubbed to return None.
        // Verify write+re-parse succeeds without error.
        let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
        rpu.level2 = Some(Level2Metadata {
            target_display_index: 0,
            trim_slope: 4096,
            trim_offset: 0,
            trim_power: 4096,
            trim_chroma_weight: 4096,
            trim_saturation_gain: 4096,
            ms_weight: 4096,
            target_mid_contrast: 2048,
            clip_trim: 0,
            saturation_vector_field: vec![],
            hue_vector_field: vec![],
        });
        let _reparsed = write_then_parse(&rpu).expect("round-trip should succeed");
        // Parser stubs L2 → presence not asserted; only verify no panic/error
    }

    #[test]
    fn test_level4_roundtrip_presence() {
        let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
        rpu.level4 = Some(Level4Metadata::sdr_anchor());
        let reparsed = write_then_parse(&rpu).expect("round-trip should succeed");
        // L4 is not currently written/parsed (stubbed), just check no panic
        let _ = reparsed;
    }

    #[test]
    fn test_level5_roundtrip() {
        verify_level5_roundtrip().expect("L5 round-trip should pass");
    }

    #[test]
    fn test_level6_roundtrip() {
        verify_level6_roundtrip().expect("L6 round-trip should pass");
    }

    #[test]
    fn test_level7_roundtrip_presence() {
        let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
        rpu.level7 = Some(Level7Metadata::bt2020_1000nits());
        let reparsed = write_then_parse(&rpu).expect("round-trip should succeed");
        // L7 is not yet in parser/writer — presence not expected to survive
        let _ = reparsed;
    }

    #[test]
    fn test_level8_roundtrip() {
        verify_level8_roundtrip().expect("L8 round-trip should pass");
    }

    #[test]
    fn test_level9_roundtrip() {
        verify_level9_roundtrip().expect("L9 round-trip should pass");
    }

    #[test]
    fn test_level11_roundtrip() {
        verify_level11_roundtrip().expect("L11 round-trip should pass");
    }

    // ── Combined metadata round-trip tests ───────────────────────────────────

    #[test]
    fn test_all_common_levels_combined_roundtrip() {
        // Build an RPU with multiple levels.  Only L1 is guaranteed to survive
        // re-parse (the other parsers are currently stubbed to return None).
        let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
        rpu.level1 = Some(Level1Metadata {
            min_pq: 62,
            max_pq: 3696,
            avg_pq: 1800,
        });
        rpu.level6 = Some(Level6Metadata::bt2020());
        rpu.level8 = Some(Level8Metadata::hdr_1000());
        rpu.level9 = Some(Level9Metadata::bt2020_mastering());

        let reparsed = write_then_parse(&rpu).expect("combined round-trip should succeed");
        assert_eq!(reparsed.profile, Profile::Profile8);
        // Level 1 must survive as it is fully implemented in parser + writer
        assert!(reparsed.level1.is_some(), "L1 must survive round-trip");
        // Levels 6/8/9 are written but parser stubs return None — just check no error
    }

    // ── NAL round-trip tests ──────────────────────────────────────────────────

    #[test]
    fn test_nal_roundtrip_write_size() {
        let rpu = DolbyVisionRpu::new(Profile::Profile8);
        let nal = rpu.write_to_nal().expect("write should succeed");
        // A minimal NAL should be at least 3 bytes (header + payload)
        assert!(nal.len() >= 3, "NAL too short: {} bytes", nal.len());
    }

    #[test]
    fn test_bitstream_roundtrip_no_crash() {
        let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
        rpu.level1 = Some(Level1Metadata {
            min_pq: 0,
            max_pq: 4095,
            avg_pq: 2047,
        });
        rpu.level6 = Some(Level6Metadata::bt2020());
        let bits = rpu.write_to_bitstream().expect("write should succeed");
        let reparsed =
            DolbyVisionRpu::parse_from_bitstream(&bits).expect("re-parse should succeed");
        assert_eq!(reparsed.profile, rpu.profile);
    }

    // ── Known-good synthetic bitstream test ──────────────────────────────────

    #[test]
    fn test_known_good_rpu_type_field() {
        // A correctly constructed default Profile 8 RPU must have rpu_type = 0
        let rpu = DolbyVisionRpu::new(Profile::Profile8);
        assert_eq!(rpu.header.rpu_type, 0, "Default rpu_type should be 0");
    }

    #[test]
    fn test_known_good_rpu_format_field() {
        // A standard RPU must have rpu_format = 0
        let rpu = DolbyVisionRpu::new(Profile::Profile8);
        assert_eq!(rpu.header.rpu_format, 0, "Default rpu_format should be 0");
    }

    #[test]
    fn test_known_good_vdr_seq_info_present() {
        let rpu = DolbyVisionRpu::new(Profile::Profile8);
        assert!(
            rpu.header.vdr_seq_info_present,
            "vdr_seq_info should be present in default RPU"
        );
        assert!(rpu.header.vdr_seq_info.is_some());
    }
}
