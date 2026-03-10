//! Scene management integration tests.

use oximedia_gaming::scene::{
    hotkey::{Hotkey, HotkeyAction, HotkeyManager},
    manager::{Scene, SceneManager},
    transition::{SceneTransition, TransitionType},
};
use std::time::Duration;

#[test]
fn test_scene_manager_creation() {
    let manager = SceneManager::new();
    assert_eq!(manager.scene_count(), 0);
    assert_eq!(manager.active_scene(), None);
}

#[test]
fn test_add_scene() {
    let mut manager = SceneManager::new();

    manager.add_scene(Scene {
        name: "Gameplay".to_string(),
        description: "Main gameplay scene".to_string(),
    });

    assert_eq!(manager.scene_count(), 1);
}

#[test]
fn test_remove_scene() {
    let mut manager = SceneManager::new();

    manager.add_scene(Scene {
        name: "Gameplay".to_string(),
        description: "Main gameplay scene".to_string(),
    });

    manager
        .remove_scene("Gameplay")
        .expect("remove scene should succeed");
    assert_eq!(manager.scene_count(), 0);
}

#[test]
fn test_switch_scene() {
    let mut manager = SceneManager::new();

    manager.add_scene(Scene {
        name: "Gameplay".to_string(),
        description: "Main gameplay scene".to_string(),
    });

    manager
        .switch_to("Gameplay")
        .expect("switch should succeed");
    assert_eq!(manager.active_scene(), Some("Gameplay"));
}

#[test]
fn test_switch_nonexistent_scene() {
    let mut manager = SceneManager::new();

    assert!(manager.switch_to("Nonexistent").is_err());
}

#[test]
fn test_remove_active_scene() {
    let mut manager = SceneManager::new();

    manager.add_scene(Scene {
        name: "Gameplay".to_string(),
        description: "Main gameplay scene".to_string(),
    });

    manager
        .switch_to("Gameplay")
        .expect("switch should succeed");
    assert!(manager.remove_scene("Gameplay").is_err());
}

#[test]
fn test_multiple_scenes() {
    let mut manager = SceneManager::new();

    let scenes = ["Gameplay", "Starting Soon", "BRB", "End Screen"];

    for name in scenes {
        manager.add_scene(Scene {
            name: name.to_string(),
            description: format!("{} scene", name),
        });
    }

    assert_eq!(manager.scene_count(), 4);

    for name in scenes {
        manager.switch_to(name).expect("switch should succeed");
        assert_eq!(manager.active_scene(), Some(name));
    }
}

#[test]
fn test_scene_transitions() {
    let transitions = [
        TransitionType::Cut,
        TransitionType::Fade,
        TransitionType::SlideLeft,
        TransitionType::SlideRight,
        TransitionType::Swipe,
    ];

    for transition_type in transitions {
        let transition = SceneTransition::new(transition_type, Duration::from_millis(300));
        assert_eq!(transition.transition_type, transition_type);
    }
}

#[test]
fn test_transition_durations() {
    let durations = [
        Duration::from_millis(100),
        Duration::from_millis(300),
        Duration::from_millis(500),
        Duration::from_secs(1),
    ];

    for duration in durations {
        let transition = SceneTransition::new(TransitionType::Fade, duration);
        assert_eq!(transition.duration, duration);
    }
}

#[test]
fn test_hotkey_manager() {
    let manager = HotkeyManager::new();
    assert_eq!(manager.hotkey_count(), 0);
}

#[test]
fn test_register_hotkey() {
    let mut manager = HotkeyManager::new();

    manager.register(Hotkey {
        key: "F1".to_string(),
        action: HotkeyAction::SwitchScene("Gameplay".to_string()),
    });

    assert_eq!(manager.hotkey_count(), 1);
}

#[test]
fn test_unregister_hotkey() {
    let mut manager = HotkeyManager::new();

    manager.register(Hotkey {
        key: "F1".to_string(),
        action: HotkeyAction::SwitchScene("Gameplay".to_string()),
    });

    manager.unregister("F1");
    assert_eq!(manager.hotkey_count(), 0);
}

#[test]
fn test_multiple_hotkeys() {
    let mut manager = HotkeyManager::new();

    manager.register(Hotkey {
        key: "F1".to_string(),
        action: HotkeyAction::SwitchScene("Gameplay".to_string()),
    });

    manager.register(Hotkey {
        key: "F2".to_string(),
        action: HotkeyAction::StartStream,
    });

    manager.register(Hotkey {
        key: "F3".to_string(),
        action: HotkeyAction::StopStream,
    });

    assert_eq!(manager.hotkey_count(), 3);
}

#[test]
fn test_all_hotkey_actions() {
    let actions = [
        HotkeyAction::SwitchScene("Test".to_string()),
        HotkeyAction::StartStream,
        HotkeyAction::StopStream,
        HotkeyAction::StartRecording,
        HotkeyAction::StopRecording,
        HotkeyAction::SaveReplay,
    ];

    for (i, action) in actions.iter().enumerate() {
        let mut manager = HotkeyManager::new();
        manager.register(Hotkey {
            key: format!("F{}", i + 1),
            action: action.clone(),
        });
        assert_eq!(manager.hotkey_count(), 1);
    }
}

#[test]
fn test_transition_defaults() {
    let transition = SceneTransition::default();
    assert_eq!(transition.transition_type, TransitionType::Fade);
    assert_eq!(transition.duration, Duration::from_millis(300));
}
