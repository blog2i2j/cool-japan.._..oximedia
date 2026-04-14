//! Wave 3 integration and unit tests for oximedia-switcher.
//!
//! Covers:
//! - Macro recording and playback for all `MacroCommand` variants
//! - Concurrent `set_program` / `set_preview` / `cut` access via `Arc<Mutex<Switcher>>`
//! - Chroma key alpha mask generation on synthetic green-screen frames
//! - Auto-transition completion and program/preview swap
//! - Fade-to-black controller drives output to full black

use crate::chroma::ChromaKey;
use crate::ftb_control::{FtbConfig, FtbController, FtbCurve, FtbState};
use crate::macro_engine::{Macro, MacroCommand, MacroPlayer, MacroRecorder};
use crate::transition::TransitionConfig;
use crate::{Switcher, SwitcherConfig};

// ── Macro recording/playback: all MacroCommand variants ──────────────────────

#[test]
fn test_macro_record_and_playback_all_variants() {
    let mut recorder = MacroRecorder::new();
    recorder
        .start_recording(1, "Wave3 Full Test".to_string())
        .expect("start recording");

    let commands = vec![
        MacroCommand::SelectProgram {
            me_row: 0,
            input: 1,
        },
        MacroCommand::SelectPreview {
            me_row: 0,
            input: 2,
        },
        MacroCommand::Cut { me_row: 0 },
        MacroCommand::Auto { me_row: 0 },
        MacroCommand::SetTransition {
            me_row: 0,
            transition_type: "Mix".to_string(),
        },
        MacroCommand::SetKeyerOnAir {
            keyer_id: 0,
            on_air: true,
        },
        MacroCommand::SetDskOnAir {
            dsk_id: 0,
            on_air: false,
        },
        MacroCommand::SelectAux {
            aux_id: 0,
            input: 3,
        },
        MacroCommand::LoadMediaPool { slot_id: 2 },
        MacroCommand::Wait { duration_ms: 500 },
        MacroCommand::RunMacro { macro_id: 99 },
    ];

    for cmd in &commands {
        recorder
            .record_command(cmd.clone())
            .expect("record command");
    }

    let recorded = recorder.stop_recording().expect("stop recording");
    assert_eq!(recorded.command_count(), commands.len());

    // Playback: consume all commands in order.
    let mut player = MacroPlayer::new();
    player.play(recorded).expect("play macro");
    assert!(player.is_playing());

    let mut played_count = 0usize;
    while let Some(cmd) = player.next_command() {
        let desc = cmd.description();
        assert!(!desc.is_empty(), "command description must not be empty");
        played_count += 1;
    }
    assert_eq!(played_count, commands.len());
    assert!(!player.is_playing());
}

#[test]
fn test_macro_record_all_variant_descriptions() {
    // Verify that every MacroCommand variant produces a non-empty description.
    let variants = vec![
        MacroCommand::SelectProgram {
            me_row: 1,
            input: 4,
        },
        MacroCommand::SelectPreview {
            me_row: 1,
            input: 5,
        },
        MacroCommand::Cut { me_row: 2 },
        MacroCommand::Auto { me_row: 2 },
        MacroCommand::SetTransition {
            me_row: 0,
            transition_type: "Wipe".to_string(),
        },
        MacroCommand::SetKeyerOnAir {
            keyer_id: 1,
            on_air: true,
        },
        MacroCommand::SetKeyerOnAir {
            keyer_id: 2,
            on_air: false,
        },
        MacroCommand::SetDskOnAir {
            dsk_id: 0,
            on_air: true,
        },
        MacroCommand::SelectAux {
            aux_id: 1,
            input: 6,
        },
        MacroCommand::LoadMediaPool { slot_id: 0 },
        MacroCommand::Wait { duration_ms: 1000 },
        MacroCommand::RunMacro { macro_id: 5 },
    ];

    for v in &variants {
        let desc = v.description();
        assert!(!desc.is_empty(), "empty description for variant: {desc:?}");
    }
}

#[test]
fn test_macro_looping_playback() {
    let mut m = Macro::new(0, "Loop".to_string());
    m.add_command(MacroCommand::Cut { me_row: 0 });
    m.set_loop_count(3);

    let mut player = MacroPlayer::new();
    player.play(m).expect("play");

    let mut count = 0usize;
    while player.next_command().is_some() {
        count += 1;
        assert!(count <= 100, "macro loop did not terminate");
    }
    // 1 command × 3 loops = 3 total command emissions.
    assert_eq!(count, 3);
}

// ── Concurrent access: multiple threads calling set_program/set_preview/cut ──

#[test]
fn test_concurrent_switcher_access() {
    use std::sync::{Arc, Mutex};

    // SwitcherConfig: 1 ME, 8 inputs, 2 aux.
    let config = SwitcherConfig::new(1, 8, 2);
    let switcher = Arc::new(Mutex::new(Switcher::new(config).expect("create switcher")));

    let num_threads = 8usize;
    let iterations = 20usize;

    let mut handles = Vec::with_capacity(num_threads);

    for t in 0..num_threads {
        let sw = Arc::clone(&switcher);
        let handle = std::thread::spawn(move || {
            for i in 0..iterations {
                let input = (t + i) % 8; // inputs 0-7
                let mut s = sw.lock().expect("lock switcher");
                // set_program and set_preview accept 0-based inputs here.
                // BusManager allows inputs from 0..num_inputs.
                s.set_program(0, input).expect("set_program");
                s.set_preview(0, (input + 1) % 8).expect("set_preview");
                // cut swaps program and preview
                s.cut(0).expect("cut");
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().expect("thread should not panic");
    }

    // Post-condition: switcher is still usable.
    let mut s = switcher.lock().expect("lock");
    s.set_program(0, 1).expect("post-concurrent set_program");
}

// ── Chroma key: alpha mask on synthetic green-screen data ─────────────────────

#[test]
fn test_chroma_key_green_screen_mask() {
    let key = ChromaKey::new_green();

    // Build a synthetic 4×4 frame: top half pure green, bottom half pure red.
    // Row-major RGBA: (0,255,0) for green pixels, (255,0,0) for red.
    const W: usize = 4;
    const H: usize = 4;

    let mut alpha_mask = Vec::with_capacity(W * H);

    for row in 0..H {
        for _ in 0..W {
            let (r, g, b) = if row < H / 2 {
                // Top half: pure green
                (0u8, 255u8, 0u8)
            } else {
                // Bottom half: pure red
                (255u8, 0u8, 0u8)
            };
            alpha_mask.push(key.calculate_alpha(r, g, b));
        }
    }

    // Green pixels (rows 0-1): alpha should be very low (transparent).
    for row in 0..(H / 2) {
        for col in 0..W {
            let alpha = alpha_mask[row * W + col];
            assert!(
                alpha < 0.3,
                "green pixel at ({row},{col}) should be transparent, got alpha={alpha}"
            );
        }
    }

    // Red pixels (rows 2-3): alpha should be high (opaque).
    for row in (H / 2)..H {
        for col in 0..W {
            let alpha = alpha_mask[row * W + col];
            assert!(
                alpha > 0.7,
                "red pixel at ({row},{col}) should be opaque, got alpha={alpha}"
            );
        }
    }
}

#[test]
fn test_chroma_key_blue_screen_mask() {
    let key = ChromaKey::new_blue();

    // Pure blue → transparent; pure red → opaque.
    let alpha_blue = key.calculate_alpha(0, 0, 255);
    let alpha_red = key.calculate_alpha(255, 0, 0);

    assert!(
        alpha_blue < 0.3,
        "pure blue should be keyed out, got {alpha_blue}"
    );
    assert!(
        alpha_red > 0.7,
        "pure red should pass through, got {alpha_red}"
    );
}

#[test]
fn test_chroma_key_spill_suppression_preserves_non_green() {
    let key = ChromaKey::new_green();

    // Red pixel — suppress_spill must not corrupt non-key-colour pixels.
    let (r_out, g_out, b_out) = key.suppress_spill(220, 30, 30);
    // Red channel should be close to original (within reasonable tolerance).
    assert!(
        r_out >= 200,
        "red channel should not be significantly reduced: {r_out}"
    );
    // Green should remain low.
    assert!(
        g_out < 60,
        "green channel of red pixel should stay low: {g_out}"
    );
    // Blue should remain close to original.
    assert!(
        b_out <= 50,
        "blue channel of red pixel should not spike: {b_out}"
    );
}

// ── Auto-transition: N frames until completion, then program/preview swap ─────

#[test]
fn test_auto_transition_completes_and_swaps() {
    let config = SwitcherConfig {
        me_rows: 1,
        num_inputs: 8,
        num_aux: 2,
        upstream_keyers_per_me: 2,
        downstream_keyers: 1,
        frame_rate: crate::sync::FrameRate::Fps25,
        media_pool_capacity: 5,
        max_macros: 10,
    };
    let mut switcher = Switcher::new(config).expect("create switcher");

    // Wire up a short mix transition so the test completes quickly.
    switcher
        .set_transition_config(0, TransitionConfig::mix(10))
        .expect("set transition config");

    switcher.set_program(0, 1).expect("set program");
    switcher.set_preview(0, 2).expect("set preview");

    switcher.auto_transition(0).expect("start auto transition");

    // The transition_engine for M/E row 0 should be in progress now.
    assert!(
        switcher
            .transition_engine(0)
            .is_some_and(|e| e.is_in_progress()),
        "transition should be in progress after auto_transition()"
    );

    // Advance frames until the transition completes (cap at 200 to avoid infinite loops).
    let max_frames = 200usize;
    let mut frame = 0usize;
    loop {
        switcher.process_frame().expect("process_frame");
        frame += 1;

        let still_in_progress = switcher
            .transition_engine(0)
            .is_some_and(|e| e.is_in_progress());

        if !still_in_progress {
            break;
        }
        assert!(
            frame < max_frames,
            "auto-transition did not complete within {max_frames} frames"
        );
    }

    // After completion, program should be the old preview (input 2).
    let program_after = switcher
        .bus_manager()
        .get_program(0)
        .expect("get program after transition");
    assert_eq!(
        program_after, 2,
        "program should have swapped to old preview (2), got {program_after}"
    );
}

// ── FTB control: output reaches full black after N frames ─────────────────────

#[test]
fn test_ftb_full_fade_to_black() {
    const DURATION: u32 = 5;
    let config = FtbConfig::new(DURATION, FtbCurve::Linear);
    let mut ctrl = FtbController::new(config).expect("create FtbController");

    assert_eq!(ctrl.state(), FtbState::Normal);
    assert!((ctrl.fade_level()).abs() < f64::EPSILON);

    // Initiate fade.
    ctrl.toggle();
    assert_eq!(ctrl.state(), FtbState::FadingToBlack);

    // Advance through all frames.
    for _ in 0..DURATION {
        ctrl.advance_frame();
    }
    assert_eq!(ctrl.state(), FtbState::Black);
    assert!(ctrl.is_black());
    assert!((ctrl.fade_level() - 1.0).abs() < f64::EPSILON);

    // Apply to a non-zero frame buffer — all bytes must become 0.
    let mut frame_data = vec![200u8; 64];
    ctrl.apply_to_frame(&mut frame_data);
    assert!(
        frame_data.iter().all(|&b| b == 0),
        "all frame bytes must be 0 after full FTB"
    );
}

#[test]
fn test_ftb_roundtrip() {
    const DURATION: u32 = 8;
    let config = FtbConfig::new(DURATION, FtbCurve::SCurve);
    let mut ctrl = FtbController::new(config).expect("create FtbController");

    // Fade to black.
    ctrl.toggle();
    for _ in 0..DURATION {
        ctrl.advance_frame();
    }
    assert_eq!(ctrl.state(), FtbState::Black);

    // Fade back to normal.
    ctrl.toggle();
    for _ in 0..DURATION {
        ctrl.advance_frame();
    }
    assert_eq!(ctrl.state(), FtbState::Normal);
    assert!((ctrl.fade_level()).abs() < f64::EPSILON);
    assert!((ctrl.video_level() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_ftb_monotonic_fade_level() {
    const DURATION: u32 = 20;
    let config = FtbConfig::new(DURATION, FtbCurve::Linear);
    let mut ctrl = FtbController::new(config).expect("create FtbController");

    ctrl.toggle();
    let mut prev_level = 0.0f64;

    for _ in 0..DURATION {
        ctrl.advance_frame();
        let level = ctrl.fade_level();
        assert!(
            level >= prev_level,
            "fade_level must be monotonically non-decreasing: {prev_level} -> {level}"
        );
        prev_level = level;
    }
    assert!((prev_level - 1.0).abs() < 1e-9);
}
