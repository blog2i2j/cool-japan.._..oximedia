//! SMPTE ST 2052-1 TTML profile compliance.
//!
//! Implements validation and serialisation helpers for the SMPTE Timed Text
//! (SMPTE-TT) profile defined in SMPTE ST 2052-1 and its companion
//! ST 2052-1:2013 amendment.  SMPTE-TT is a constrained TTML profile used in
//! broadcast delivery (DECE, ATSC 3.0, IMSC referencing the SMPTE profile).

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ── SMPTE profile identifier ──────────────────────────────────────────────────

/// SMPTE ST 2052-1 sub-profile identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SmpteProfile {
    /// SMPTE-TT base profile (ST 2052-1:2010).
    Base,
    /// SMPTE-TT with CEA-708 style constraints.
    Cea708Compat,
    /// SMPTE-TT with High Definition (HD) constraints.
    Hd,
}

impl SmpteProfile {
    /// Return the registered SMPTE profile URI.
    #[must_use]
    pub fn uri(self) -> &'static str {
        match self {
            Self::Base => "http://www.smpte-ra.org/schemas/2052-1/2010/smpte-tt#STT-Base",
            Self::Cea708Compat => {
                "http://www.smpte-ra.org/schemas/2052-1/2010/smpte-tt#STT-CEA708"
            }
            Self::Hd => "http://www.smpte-ra.org/schemas/2052-1/2010/smpte-tt#STT-HD",
        }
    }

    /// Return the human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Base => "SMPTE-TT Base",
            Self::Cea708Compat => "SMPTE-TT CEA-708 Compatible",
            Self::Hd => "SMPTE-TT HD",
        }
    }
}

// ── Validation ────────────────────────────────────────────────────────────────

/// A single compliance issue found during SMPTE-TT validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmpteIssue {
    /// Short rule identifier (e.g. `"SMPTE-001"`).
    pub rule_id: String,
    /// Human-readable description of the violation.
    pub message: String,
    /// Whether this issue is fatal (`true`) or advisory (`false`).
    pub is_error: bool,
}

impl SmpteIssue {
    /// Create an error-level issue.
    #[must_use]
    pub fn error(rule_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            rule_id: rule_id.into(),
            message: message.into(),
            is_error: true,
        }
    }

    /// Create an advisory-level issue.
    #[must_use]
    pub fn advisory(rule_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            rule_id: rule_id.into(),
            message: message.into(),
            is_error: false,
        }
    }
}

/// Validates a raw TTML XML string for SMPTE ST 2052-1 compliance.
///
/// Performs lightweight heuristic checks suitable for build-time or pipeline
/// QC.  For production, replace the string-based probes with a full XML DOM
/// walk using `quick-xml`.
#[derive(Debug, Default)]
pub struct SmpteTtValidator;

impl SmpteTtValidator {
    /// Create a new validator.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Validate `xml` against the given SMPTE-TT `profile`.
    ///
    /// Returns a (possibly empty) list of [`SmpteIssue`]s.
    #[must_use]
    pub fn validate(&self, xml: &str, profile: SmpteProfile) -> Vec<SmpteIssue> {
        let mut issues = Vec::new();

        // SMPTE-001: The root <tt> element MUST carry a `ttp:profile` attribute
        // referencing the correct SMPTE profile URI.
        let uri = profile.uri();
        if !xml.contains(uri) && !xml.contains("ttp:profile") {
            issues.push(SmpteIssue::error(
                "SMPTE-001",
                format!(
                    "ttp:profile is absent or does not reference the expected URI: {uri}",
                ),
            ));
        }

        // SMPTE-002: `xml:lang` MUST be present on the root element.
        if !xml.contains("xml:lang") && !xml.contains("xmlLang") {
            issues.push(SmpteIssue::error(
                "SMPTE-002",
                "xml:lang attribute is required on the root <tt> element.",
            ));
        }

        // SMPTE-003: `tts:extent` MUST be present to establish the display area.
        if !xml.contains("tts:extent") && !xml.contains("ttsExtent") {
            issues.push(SmpteIssue::error(
                "SMPTE-003",
                "tts:extent is required to define the root container extent.",
            ));
        }

        // SMPTE-004 (advisory): SMPTE-TT recommends frame-accurate timing via
        // `frameRate` and `subFrameRate`; warn if neither is present.
        if !xml.contains("frameRate") && !xml.contains("ttp:frameRate") {
            issues.push(SmpteIssue::advisory(
                "SMPTE-004",
                "ttp:frameRate is absent; frame-accurate timing may not be achievable.",
            ));
        }

        match profile {
            SmpteProfile::Hd => {
                // HD profile: images and fonts MUST use absolute px values.
                // Heuristic: warn if em units are used in a font-size declaration.
                if xml.contains("fontSize") && xml.contains("em") {
                    issues.push(SmpteIssue::advisory(
                        "SMPTE-HD-001",
                        "SMPTE-TT HD profile recommends absolute px font sizes; \
                         em units detected.",
                    ));
                }
            }
            SmpteProfile::Cea708Compat => {
                // CEA-708: colour values MUST include full RGBA.
                if xml.contains("tts:color") && !xml.contains('#') {
                    issues.push(SmpteIssue::advisory(
                        "SMPTE-CEA-001",
                        "CEA-708 compatible profile recommends #RRGGBBAA colour notation.",
                    ));
                }
            }
            SmpteProfile::Base => {}
        }

        issues
    }
}

// ── SMPTE-TT timing helpers ───────────────────────────────────────────────────

/// Convert a SMPTE drop-frame timecode string (`HH:MM:SS;FF`) to milliseconds.
///
/// Uses the standard 29.97 fps (30000/1001) drop-frame formula.
/// Returns `None` if the string cannot be parsed.
#[must_use]
pub fn drop_frame_to_ms(tc: &str) -> Option<u64> {
    // Format: HH:MM:SS;FF  (semicolon indicates drop-frame)
    let parts: Vec<&str> = tc.splitn(4, [':', ';']).collect();
    if parts.len() != 4 {
        return None;
    }
    let h: u64 = parts[0].parse().ok()?;
    let m: u64 = parts[1].parse().ok()?;
    let s: u64 = parts[2].parse().ok()?;
    let f: u64 = parts[3].parse().ok()?;

    // 30000/1001 drop-frame: total_frames = 30*60*60*h + 30*60*m + 30*s + f
    //   minus drop frames = 2*(total_minutes - total_minutes/10)
    let total_minutes = 60 * h + m;
    let drop_frames = 2 * (total_minutes - total_minutes / 10);
    let total_frames = (108_000 * h) + (1_800 * m) + (30 * s) + f - drop_frames;

    // Each frame ≈ 1001/30000 seconds = 1001000/30000 ms
    let ms = total_frames * 1_001_000 / 30_000;
    Some(ms)
}

/// Convert milliseconds to a SMPTE drop-frame timecode string (`HH:MM:SS;FF`).
#[must_use]
pub fn ms_to_drop_frame(ms: u64) -> String {
    // Reverse of drop-frame: compute total frames from ms, then apply drop-frame offsets.
    let total_frames = ms * 30_000 / 1_001_000;
    // 10-minute blocks contain 17982 frames (18000 - 18)
    let d = total_frames / 17_982;
    let m = total_frames % 17_982;
    // Within a 10-minute block, 2-minute sub-blocks contain 1798 frames (1800 - 2)
    let (m2, f_off) = if m < 2 {
        (0, m)
    } else {
        let m2 = (m - 2) / 1_798 + 1;
        let f_off = (m - 2) % 1_798;
        (m2, f_off)
    };
    let frames_in_min = f_off + if m2 > 0 { 2 } else { 0 };
    let seconds = frames_in_min / 30;
    let frames = frames_in_min % 30;
    let minutes = d * 10 + m2;
    let hours = minutes / 60;
    let mins = minutes % 60;
    let secs = seconds % 60;
    format!("{hours:02}:{mins:02}:{secs:02};{frames:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_uri_base() {
        let uri = SmpteProfile::Base.uri();
        assert!(uri.contains("smpte-ra.org"));
        assert!(uri.contains("STT-Base"));
    }

    #[test]
    fn test_profile_name() {
        assert_eq!(SmpteProfile::Hd.name(), "SMPTE-TT HD");
    }

    #[test]
    fn test_validator_missing_profile_uri() {
        let v = SmpteTtValidator::new();
        let xml = r#"<tt xml:lang="en" tts:extent="1920px 1080px"/>"#;
        let issues = v.validate(xml, SmpteProfile::Base);
        assert!(
            issues.iter().any(|i| i.rule_id == "SMPTE-001"),
            "missing ttp:profile should trigger SMPTE-001"
        );
    }

    #[test]
    fn test_validator_missing_lang() {
        let v = SmpteTtValidator::new();
        let uri = SmpteProfile::Base.uri();
        let xml = format!(r#"<tt ttp:profile="{uri}" tts:extent="1920px 1080px"/>"#);
        let issues = v.validate(&xml, SmpteProfile::Base);
        assert!(
            issues.iter().any(|i| i.rule_id == "SMPTE-002"),
            "missing xml:lang should trigger SMPTE-002"
        );
    }

    #[test]
    fn test_validator_compliant_base() {
        let v = SmpteTtValidator::new();
        let uri = SmpteProfile::Base.uri();
        // Provide all required attributes except frameRate (advisory only).
        let xml = format!(
            r#"<tt xml:lang="en" ttp:profile="{uri}" tts:extent="1920px 1080px"/>"#
        );
        let issues = v.validate(&xml, SmpteProfile::Base);
        // Only the advisory SMPTE-004 (frameRate absent) should remain.
        let errors: Vec<_> = issues.iter().filter(|i| i.is_error).collect();
        assert!(
            errors.is_empty(),
            "compliant document should have no errors; got: {errors:?}"
        );
    }

    #[test]
    fn test_drop_frame_to_ms_zero() {
        let ms = drop_frame_to_ms("00:00:00;00");
        assert_eq!(ms, Some(0));
    }

    #[test]
    fn test_drop_frame_to_ms_invalid() {
        assert!(drop_frame_to_ms("not-a-timecode").is_none());
    }
}
