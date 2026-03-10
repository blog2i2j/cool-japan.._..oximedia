//! Scene switching example.

use oximedia_gaming::scene::{Scene, SceneManager, SceneTransition, TransitionType};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiMedia Gaming - Scene Switching Example ===\n");

    // Create scene manager
    let mut manager = SceneManager::new();

    // Add scenes
    println!("Creating scenes...");
    manager.add_scene(Scene {
        name: "Gameplay".to_string(),
        description: "Main gameplay with webcam PiP".to_string(),
    });

    manager.add_scene(Scene {
        name: "Starting Soon".to_string(),
        description: "Pre-stream starting soon screen".to_string(),
    });

    manager.add_scene(Scene {
        name: "BRB".to_string(),
        description: "Be Right Back screen".to_string(),
    });

    manager.add_scene(Scene {
        name: "End Screen".to_string(),
        description: "Thanks for watching screen".to_string(),
    });

    println!("Scenes created: {}\n", manager.scene_count());

    // Create transition
    let transition = SceneTransition::new(TransitionType::Fade, Duration::from_millis(300));

    println!(
        "Transition: {:?} ({:?})\n",
        transition.transition_type, transition.duration
    );

    // Simulate scene switching
    println!("Starting scene sequence...\n");

    // Starting Soon
    manager.switch_to("Starting Soon")?;
    println!(
        "→ Switched to: {}",
        manager.active_scene().expect("active scene should exist")
    );
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Gameplay
    manager.switch_to("Gameplay")?;
    println!(
        "→ Switched to: {}",
        manager.active_scene().expect("active scene should exist")
    );
    tokio::time::sleep(Duration::from_secs(5)).await;

    // BRB
    manager.switch_to("BRB")?;
    println!(
        "→ Switched to: {}",
        manager.active_scene().expect("active scene should exist")
    );
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Back to Gameplay
    manager.switch_to("Gameplay")?;
    println!(
        "→ Switched to: {}",
        manager.active_scene().expect("active scene should exist")
    );
    tokio::time::sleep(Duration::from_secs(3)).await;

    // End Screen
    manager.switch_to("End Screen")?;
    println!(
        "→ Switched to: {}",
        manager.active_scene().expect("active scene should exist")
    );

    println!("\nScene switching complete!");

    Ok(())
}
