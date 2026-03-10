//! Audio module integration tests.

use oximedia_gaming::audio::{
    AudioMixer, AudioSource, GameAudioCapture, MicConfig, MicrophoneCapture, MixerConfig,
    MusicPlayer, MusicTrack,
};

#[test]
fn test_audio_mixer_creation() {
    let config = MixerConfig::default();
    let mixer = AudioMixer::new(config).expect("valid audio mixer");

    assert_eq!(mixer.source_count(), 0);
}

#[test]
fn test_mixer_invalid_channels() {
    let mut config = MixerConfig::default();
    config.channels = 0;

    assert!(AudioMixer::new(config).is_err());
}

#[test]
fn test_add_remove_source() {
    let config = MixerConfig::default();
    let mut mixer = AudioMixer::new(config).expect("valid audio mixer");

    let source = AudioSource {
        name: "Game".to_string(),
        volume: 1.0,
        muted: false,
    };

    mixer.add_source(source);
    assert_eq!(mixer.source_count(), 1);

    mixer.remove_source("Game");
    assert_eq!(mixer.source_count(), 0);
}

#[test]
fn test_set_source_volume() {
    let config = MixerConfig::default();
    let mut mixer = AudioMixer::new(config).expect("valid audio mixer");

    mixer.add_source(AudioSource {
        name: "Game".to_string(),
        volume: 1.0,
        muted: false,
    });

    mixer
        .set_source_volume("Game", 0.5)
        .expect("set volume should succeed");
    mixer
        .set_source_volume("Game", 0.0)
        .expect("set volume should succeed");
    mixer
        .set_source_volume("Game", 1.0)
        .expect("set volume should succeed");
}

#[test]
fn test_set_source_mute() {
    let config = MixerConfig::default();
    let mut mixer = AudioMixer::new(config).expect("valid audio mixer");

    mixer.add_source(AudioSource {
        name: "Game".to_string(),
        volume: 1.0,
        muted: false,
    });

    mixer
        .set_source_mute("Game", true)
        .expect("set mute should succeed");
    mixer
        .set_source_mute("Game", false)
        .expect("set mute should succeed");
}

#[test]
fn test_nonexistent_source() {
    let config = MixerConfig::default();
    let mut mixer = AudioMixer::new(config).expect("valid audio mixer");

    assert!(mixer.set_source_volume("Nonexistent", 0.5).is_err());
    assert!(mixer.set_source_mute("Nonexistent", true).is_err());
}

#[test]
fn test_multiple_sources() {
    let config = MixerConfig::default();
    let mut mixer = AudioMixer::new(config).expect("valid audio mixer");

    let sources = ["Game", "Microphone", "Music", "Discord"];

    for name in sources {
        mixer.add_source(AudioSource {
            name: name.to_string(),
            volume: 1.0,
            muted: false,
        });
    }

    assert_eq!(mixer.source_count(), 4);

    for name in sources {
        mixer.remove_source(name);
    }

    assert_eq!(mixer.source_count(), 0);
}

#[test]
fn test_game_audio_capture() {
    let _capture = GameAudioCapture::new();
}

#[test]
fn test_list_audio_devices() {
    let devices = GameAudioCapture::list_devices().expect("list devices should succeed");
    assert!(!devices.is_empty());

    for device in &devices {
        assert!(device.sample_rate > 0);
        assert!(device.channels > 0);
    }
}

#[test]
fn test_microphone_capture() {
    let config = MicConfig::default();
    let _mic = MicrophoneCapture::new(config);
}

#[test]
fn test_mic_start_stop() {
    let config = MicConfig::default();
    let mut mic = MicrophoneCapture::new(config);

    mic.start().expect("start should succeed");
    mic.stop();
}

#[test]
fn test_mic_config_defaults() {
    let config = MicConfig::default();

    assert_eq!(config.device_id, "default");
    assert_eq!(config.sample_rate, 48000);
    assert!(config.noise_suppression);
    assert!(config.echo_cancellation);
}

#[test]
fn test_music_player() {
    let player = MusicPlayer::new();
    assert_eq!(player.volume(), 0.5);
}

#[test]
fn test_music_player_volume() {
    let mut player = MusicPlayer::new();

    player.set_volume(0.8);
    assert_eq!(player.volume(), 0.8);

    player.set_volume(0.0);
    assert_eq!(player.volume(), 0.0);

    player.set_volume(1.0);
    assert_eq!(player.volume(), 1.0);
}

#[test]
fn test_music_player_volume_clamping() {
    let mut player = MusicPlayer::new();

    // Should clamp to 0.0
    player.set_volume(-0.5);
    assert_eq!(player.volume(), 0.0);

    // Should clamp to 1.0
    player.set_volume(1.5);
    assert_eq!(player.volume(), 1.0);
}

#[test]
fn test_music_player_play_stop() {
    let mut player = MusicPlayer::new();

    let track = MusicTrack {
        path: "/path/to/music.opus".to_string(),
        title: "Background Music".to_string(),
        volume: 0.3,
    };

    player.play(track).expect("play should succeed");
    player.stop();
}

#[test]
fn test_mixer_config_defaults() {
    let config = MixerConfig::default();

    assert_eq!(config.sample_rate, 48000);
    assert_eq!(config.channels, 2);
    assert_eq!(config.master_volume, 1.0);
}

#[test]
fn test_multiple_sample_rates() {
    let sample_rates = [44100, 48000, 96000];

    for sample_rate in sample_rates {
        let mut config = MixerConfig::default();
        config.sample_rate = sample_rate;

        let mixer = AudioMixer::new(config).expect("valid audio mixer");
        assert_eq!(mixer.config.sample_rate, sample_rate);
    }
}

#[test]
fn test_stereo_and_mono() {
    let channels = [1, 2];

    for channel_count in channels {
        let mut config = MixerConfig::default();
        config.channels = channel_count;

        let mixer = AudioMixer::new(config).expect("valid audio mixer");
        assert_eq!(mixer.config.channels, channel_count);
    }
}
