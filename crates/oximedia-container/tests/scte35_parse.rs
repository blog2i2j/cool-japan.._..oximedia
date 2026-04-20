//! Integration tests for SCTE-35 parsing and emission via public API.
//!
//! These tests exercise the public re-exports from `oximedia_container` root
//! and confirm that the parser and emitter are wire-compatible.

use oximedia_container::mux::mpegts::scte35::SpliceInsertConfig;
use oximedia_container::{
    emit_splice_insert, emit_splice_null, emit_time_signal, parse_splice_info_section,
    Scte35Config, Scte35Parser, SpliceCommand, SpliceInfoSection,
};
use std::time::Duration;

// ─── Helper ──────────────────────────────────────────────────────────────────

/// Parse a section using both the free function and the parser method, asserting they agree.
fn parse_both(data: &[u8]) -> SpliceInfoSection {
    let section_a =
        parse_splice_info_section(data).expect("parse_splice_info_section should succeed");

    let mut parser = Scte35Parser::new(Scte35Config::default());
    let section_b = parser
        .parse(data)
        .expect("Scte35Parser::parse should succeed");

    assert_eq!(
        section_a.splice_command, section_b.splice_command,
        "free function and method must agree on splice_command"
    );
    assert_eq!(
        section_a.crc32, section_b.crc32,
        "free function and method must agree on crc32"
    );

    section_a
}

// ─── Roundtrip: splice_null ───────────────────────────────────────────────────

#[test]
fn splice_null_roundtrip() {
    let bytes = emit_splice_null();
    assert!(!bytes.is_empty(), "emit_splice_null must produce bytes");

    let section = parse_both(&bytes);
    assert_eq!(
        section.splice_command,
        SpliceCommand::Null,
        "splice_null must roundtrip as SpliceCommand::Null"
    );
    assert!(!section.encrypted, "section must not be marked encrypted");
    assert_eq!(section.protocol_version, 0);
}

// ─── Roundtrip: time_signal (immediate) ──────────────────────────────────────

#[test]
fn time_signal_immediate_roundtrip() {
    let bytes = emit_time_signal(None);
    assert!(!bytes.is_empty());

    let section = parse_both(&bytes);
    if let SpliceCommand::TimeSignal(t) = &section.splice_command {
        assert!(
            !t.time_specified,
            "immediate time signal must have time_specified=false"
        );
        assert!(t.pts_time.is_none());
    } else {
        panic!("Expected TimeSignal, got {:?}", section.splice_command);
    }
}

// ─── Roundtrip: time_signal (with PTS) ───────────────────────────────────────

#[test]
fn time_signal_with_pts_roundtrip() {
    // Emit a time signal for PTS = 90,000 ticks (1 second at 90 kHz)
    let one_second = Duration::from_secs(1);
    let bytes = emit_time_signal(Some(one_second));
    assert!(!bytes.is_empty());

    let section = parse_both(&bytes);
    if let SpliceCommand::TimeSignal(t) = &section.splice_command {
        assert!(t.time_specified, "should have time_specified=true");
        let pts = t.pts_time.expect("should have a pts_time");
        assert_eq!(pts, 90_000, "1 second should encode as 90,000 at 90 kHz");
    } else {
        panic!("Expected TimeSignal, got {:?}", section.splice_command);
    }
}

// ─── Roundtrip: splice_insert (immediate, out-of-network) ────────────────────

#[test]
fn splice_insert_immediate_roundtrip() {
    let cfg = SpliceInsertConfig {
        event_id: 42,
        out_of_network: true,
        duration: None,
        auto_return: false,
        splice_pts: None, // None = immediate
        unique_program_id: 1,
    };
    let bytes = emit_splice_insert(&cfg);
    assert!(!bytes.is_empty());

    let section = parse_both(&bytes);
    if let SpliceCommand::Insert(ins) = &section.splice_command {
        assert_eq!(ins.event_id, 42);
        assert!(!ins.event_cancel);
        assert!(ins.out_of_network);
        assert!(ins.program_splice, "program_splice must be set");
        assert!(ins.immediate);
        assert!(
            ins.splice_time.is_none(),
            "immediate splice has no splice_time"
        );
        assert_eq!(ins.unique_program_id, 1);
    } else {
        panic!("Expected Insert, got {:?}", section.splice_command);
    }
}

// ─── Roundtrip: splice_insert (with duration) ────────────────────────────────

#[test]
fn splice_insert_with_duration_roundtrip() {
    // 30-second break
    let cfg = SpliceInsertConfig {
        event_id: 99,
        out_of_network: true,
        duration: Some(Duration::from_secs(30)),
        auto_return: true,
        splice_pts: None, // immediate
        unique_program_id: 7,
    };
    let bytes = emit_splice_insert(&cfg);
    let section = parse_both(&bytes);
    if let SpliceCommand::Insert(ins) = &section.splice_command {
        assert_eq!(ins.event_id, 99);
        let bd = ins.duration.as_ref().expect("should have break_duration");
        // 30 seconds × 90,000 = 2,700,000 ticks
        assert_eq!(
            bd.duration, 2_700_000,
            "break duration must survive roundtrip"
        );
        assert!(bd.auto_return);
    } else {
        panic!("Expected Insert, got {:?}", section.splice_command);
    }
}

// ─── Free function error cases ────────────────────────────────────────────────

#[test]
fn free_function_rejects_empty_input() {
    let result = parse_splice_info_section(&[]);
    assert!(result.is_err(), "empty slice must be rejected");
}

#[test]
fn free_function_rejects_wrong_table_id() {
    // Build a minimal valid section, then corrupt the table ID.
    let mut bytes = emit_splice_null();
    bytes[0] = 0x00; // not 0xFC
    let result = parse_splice_info_section(&bytes);
    assert!(result.is_err(), "wrong table_id must be rejected");
}

#[test]
fn free_function_rejects_crc_mismatch() {
    let mut bytes = emit_splice_null();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF; // corrupt CRC
    let result = parse_splice_info_section(&bytes);
    assert!(result.is_err(), "CRC mismatch must be rejected");
}

// ─── Multiple sections in sequence ───────────────────────────────────────────

#[test]
fn parse_three_sections_in_sequence() {
    let null_bytes = emit_splice_null();
    let ts_bytes = emit_time_signal(Some(Duration::from_secs(5)));
    let ins_bytes = emit_splice_insert(&SpliceInsertConfig {
        event_id: 1,
        out_of_network: false,
        duration: None,
        auto_return: false,
        splice_pts: None,
        unique_program_id: 0,
    });

    let sections: Vec<SpliceInfoSection> = [null_bytes, ts_bytes, ins_bytes]
        .iter()
        .map(|b| parse_splice_info_section(b).expect("each section should parse"))
        .collect();

    assert!(matches!(sections[0].splice_command, SpliceCommand::Null));
    assert!(matches!(
        sections[1].splice_command,
        SpliceCommand::TimeSignal(_)
    ));
    assert!(matches!(
        sections[2].splice_command,
        SpliceCommand::Insert(_)
    ));
}
