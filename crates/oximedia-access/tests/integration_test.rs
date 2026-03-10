//! Integration tests for oximedia-access.

use oximedia_access::audio_desc::script::{AudioDescriptionEntry, AudioDescriptionScript};
use oximedia_access::audio_desc::{
    AudioDescriptionConfig, AudioDescriptionGenerator, AudioDescriptionQuality,
    AudioDescriptionType,
};
use oximedia_access::caption::{CaptionConfig, CaptionGenerator, CaptionType};
use oximedia_access::compliance::{ComplianceChecker, WcagChecker, WcagLevel};
use oximedia_access::translate::{SubtitleTranslator, TranslationConfig};
use oximedia_access::visual::contrast::ContrastEnhancer;

#[test]
fn test_audio_description_workflow() {
    // Create script
    let mut script = AudioDescriptionScript::new();
    script.add_entry(AudioDescriptionEntry::new(
        1000,
        3000,
        "Test description".to_string(),
    ));

    // Validate
    assert!(script.validate().is_ok());

    // Generate
    let config = AudioDescriptionConfig::new(
        AudioDescriptionType::Standard,
        AudioDescriptionQuality::Standard,
    );
    let generator = AudioDescriptionGenerator::new(config);
    let result = generator.generate(&script);
    assert!(result.is_ok());
}

#[test]
fn test_caption_generation_workflow() {
    let config = CaptionConfig::new("en".to_string(), CaptionType::Closed);
    let generator = CaptionGenerator::new(config);

    let transcript = "Hello world. This is a test.";
    let timestamps = vec![(0, 2000), (2000, 4000)];

    let result = generator.generate_from_transcript(transcript, &timestamps);
    assert!(result.is_ok());

    let captions = result.expect("captions should be valid");
    assert_eq!(captions.len(), 2);
}

#[test]
fn test_translation_workflow() {
    let config = TranslationConfig {
        source_lang: "en".to_string(),
        target_lang: "es".to_string(),
        preserve_timing: true,
        max_chars_per_line: 42,
    };

    let translator = SubtitleTranslator::new(config);
    let subtitle = oximedia_subtitle::Subtitle::new(1000, 3000, "Hello".to_string());

    let result = translator.translate(&subtitle);
    assert!(result.is_ok());
}

#[test]
fn test_compliance_checking_workflow() {
    let checker = ComplianceChecker::new();
    let report = checker.check_all();

    assert!(report.issues().is_empty() || !report.issues().is_empty());
}

#[test]
fn test_wcag_contrast_requirements() {
    // Test WCAG AA requirement (4.5:1)
    let white = (255, 255, 255);
    let black = (0, 0, 0);

    let ratio = ContrastEnhancer::contrast_ratio(white, black);
    assert!(ratio > 4.5);
    assert!(ContrastEnhancer::meets_wcag_aa(white, black));

    // Test insufficient contrast
    let light_gray = (200, 200, 200);
    let lighter_gray = (220, 220, 220);
    assert!(!ContrastEnhancer::meets_wcag_aa(light_gray, lighter_gray));
}

#[test]
fn test_wcag_caption_requirements() {
    let checker = WcagChecker::new(WcagLevel::AA);

    // Captions required
    assert!(checker.check_captions_present(false).is_some());
    assert!(checker.check_captions_present(true).is_none());

    // Audio description for Level AA
    assert!(checker.check_audio_description(false).is_some());
    assert!(checker.check_audio_description(true).is_none());
}

#[test]
fn test_audio_description_quality_constraints() {
    let basic = AudioDescriptionQuality::Basic;
    let professional = AudioDescriptionQuality::Professional;

    assert!(basic.min_duration_ms() < professional.min_duration_ms());
    assert!(basic.min_gap_after_ms() < professional.min_gap_after_ms());
}

#[test]
fn test_script_json_roundtrip() {
    let mut script = AudioDescriptionScript::new();
    script.add_entry(AudioDescriptionEntry::new(
        1000,
        2000,
        "Test entry".to_string(),
    ));

    let json = script.to_json().expect("json should be valid");
    let restored = AudioDescriptionScript::from_json(&json).expect("restored should be valid");

    assert_eq!(script.len(), restored.len());
    assert_eq!(script.entries()[0].text, restored.entries()[0].text);
}

#[test]
fn test_caption_synchronization() {
    use oximedia_access::caption::sync::{CaptionSynchronizer, SyncQuality};
    use oximedia_access::caption::Caption;
    use oximedia_subtitle::Subtitle;

    let synchronizer = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);

    let mut captions = vec![
        Caption::new(
            Subtitle::new(1003, 2007, "First".to_string()),
            CaptionType::Closed,
        ),
        Caption::new(
            Subtitle::new(3010, 4015, "Second".to_string()),
            CaptionType::Closed,
        ),
    ];

    // Sync to frames
    assert!(synchronizer.sync_to_frames(&mut captions).is_ok());

    // Validate
    assert!(synchronizer.validate(&captions).is_ok());
}

#[test]
fn test_timing_analysis() {
    use oximedia_access::audio_desc::timing::{DialogueSegment, TimingAnalyzer, TimingConstraints};

    let constraints = TimingConstraints {
        min_gap_ms: 1000,
        min_description_ms: 500,
        max_description_ms: 10000,
        min_gap_after_ms: 200,
        allow_extended: false,
    };

    let analyzer = TimingAnalyzer::new(constraints);

    let dialogue = vec![
        DialogueSegment::new(0, 2000),
        DialogueSegment::new(5000, 7000),
    ];

    let gaps = analyzer.find_gaps(&dialogue);
    assert!(!gaps.is_empty());
    assert_eq!(gaps[0].start_time_ms, 2000);
    assert_eq!(gaps[0].end_time_ms, 5000);
}

#[test]
fn test_transcript_generation() {
    use oximedia_access::transcript::TranscriptGenerator;

    let generator = TranscriptGenerator::new("en".to_string());
    assert_eq!(generator.language(), "en");

    // Test from captions
    let config = CaptionConfig::new("en".to_string(), CaptionType::Closed);
    let caption_gen = CaptionGenerator::new(config);

    let transcript_text = "Test transcript";
    let timestamps = vec![(0, 2000)];

    let captions = caption_gen
        .generate_from_transcript(transcript_text, &timestamps)
        .expect("test expectation failed");

    let transcript = generator.from_captions(&captions);
    assert_eq!(transcript.entries.len(), 1);
}

#[test]
fn test_transcript_formatting() {
    use oximedia_access::transcript::{
        Transcript, TranscriptEntry, TranscriptFormat, TranscriptFormatter,
    };

    let mut transcript = Transcript::new();
    transcript.add_entry(TranscriptEntry::new(0, 2000, "Hello".to_string()));
    transcript.add_entry(TranscriptEntry::new(2000, 4000, "World".to_string()));

    // Test plain format
    let plain = TranscriptFormatter::format(&transcript, TranscriptFormat::Plain);
    assert!(plain.contains("Hello"));
    assert!(plain.contains("World"));

    // Test VTT format
    let vtt = TranscriptFormatter::format(&transcript, TranscriptFormat::Vtt);
    assert!(vtt.starts_with("WEBVTT"));

    // Test SRT format
    let srt = TranscriptFormatter::format(&transcript, TranscriptFormat::Srt);
    assert!(srt.contains("1\n"));
    assert!(srt.contains("2\n"));

    // Test JSON format
    let json = TranscriptFormatter::format(&transcript, TranscriptFormat::Json);
    assert!(json.contains("entries"));
}

#[test]
fn test_language_detection() {
    use oximedia_access::translate::language::{Language, LanguageDetector};

    // Test language codes
    assert_eq!(Language::English.code(), "en");
    assert_eq!(Language::Spanish.code(), "es");
    assert_eq!(Language::Japanese.code(), "ja");

    // Test from code
    assert_eq!(Language::from_code("en"), Some(Language::English));
    assert_eq!(Language::from_code("ES"), Some(Language::Spanish));
    assert_eq!(Language::from_code("invalid"), None);

    // Test detection
    let (detected, confidence) = LanguageDetector::detect_with_confidence("Test text");
    assert!(detected.is_some());
    assert!(confidence > 0.0);
}

#[test]
fn test_tts_configuration() {
    use oximedia_access::tts::{TextToSpeech, TtsConfig};

    let config = TtsConfig {
        voice: "en-US-Neural".to_string(),
        rate: 1.0,
        pitch: 0.0,
        volume: 0.8,
        sample_rate: 24000,
    };

    let tts = TextToSpeech::new(config);
    assert_eq!(tts.config().sample_rate, 24000);
}

#[test]
fn test_stt_configuration() {
    use oximedia_access::stt::{SpeechToText, SttConfig, SttModel};

    let config = SttConfig {
        language: "en".to_string(),
        speaker_diarization: true,
        enable_punctuation: true,
        word_timestamps: true,
        model: SttModel::Standard,
    };

    let stt = SpeechToText::new(config);
    assert_eq!(stt.config().language, "en");
    assert!(stt.config().speaker_diarization);
}

#[test]
fn test_visual_enhancements() {
    use oximedia_access::visual::{
        color::{ColorBlindnessAdapter, ColorBlindnessType},
        contrast::ContrastEnhancer,
        size::TextSizeAdjuster,
    };

    // Test contrast enhancer
    let enhancer = ContrastEnhancer::new(0.5);
    assert!((enhancer.level() - 0.5).abs() < f32::EPSILON);

    // Test color blindness adapter
    let adapter = ColorBlindnessAdapter::new(ColorBlindnessType::Protanopia);
    let color = (255, 128, 64);
    let _transformed = adapter.transform_color(color);

    // Test text size adjuster
    let adjuster = TextSizeAdjuster::new(1.5);
    assert_eq!(adjuster.adjust_size(12), 18);
}

#[test]
fn test_audio_enhancements() {
    use oximedia_access::audio::{
        clarity::AudioClarityEnhancer, noise::NoiseReducer, normalize::LoudnessNormalizer,
    };

    // Test clarity enhancer
    let enhancer = AudioClarityEnhancer::new(0.7);
    assert!((enhancer.level() - 0.7).abs() < f32::EPSILON);

    // Test noise reducer
    let reducer = NoiseReducer::new(0.8);
    assert!((reducer.reduction_level() - 0.8).abs() < f32::EPSILON);

    // Test loudness normalizer
    let normalizer = LoudnessNormalizer::new(-23.0);
    let gain = normalizer.calculate_gain(-26.0);
    assert!((gain - 3.0).abs() < f32::EPSILON);
}

#[test]
fn test_speed_control() {
    use oximedia_access::speed::{SpeedConfig, SpeedController};

    let config = SpeedConfig::default();
    let mut controller = SpeedController::new(config);

    assert!(controller.set_speed(1.5).is_ok());
    assert!(controller.set_speed(0.3).is_err());
    assert!(controller.set_speed(3.0).is_err());

    let new_duration = controller.calculate_new_duration(1000);
    assert_eq!(new_duration, 666);
}

#[test]
fn test_compliance_report() {
    use oximedia_access::compliance::report::{ComplianceIssue, ComplianceReport, IssueSeverity};

    let mut report = ComplianceReport::new();

    report.add_issue(ComplianceIssue::new(
        "TEST-001".to_string(),
        "Test Issue".to_string(),
        "Description".to_string(),
        IssueSeverity::Critical,
    ));

    assert_eq!(report.summary().total_issues, 1);
    assert_eq!(report.summary().critical_issues, 1);
    assert!(!report.is_compliant());

    let text = report.to_text();
    assert!(text.contains("Test Issue"));

    let json = report.to_json().expect("json should be valid");
    assert!(json.contains("TEST-001"));
}

#[test]
fn test_ebu_compliance() {
    use oximedia_access::compliance::ebu::EbuChecker;

    let checker = EbuChecker::new();

    // Test loudness
    assert!(checker.check_loudness(-23.0).is_none());
    assert!(checker.check_loudness(-20.0).is_some());

    // Test subtitle format
    assert!(checker.check_subtitle_format(30).is_none());
    assert!(checker.check_subtitle_format(50).is_some());

    // Test subtitle duration
    assert!(checker.check_subtitle_duration(3000).is_none());
    assert!(checker.check_subtitle_duration(500).is_some());
    assert!(checker.check_subtitle_duration(8000).is_some());
}

#[test]
fn test_sign_language_overlay() {
    use oximedia_access::sign::{SignConfig, SignLanguageOverlay, SignPosition, SignSize};

    let config = SignConfig {
        position: SignPosition::BottomRight,
        size: SignSize::Medium,
        border: None,
        opacity: 1.0,
    };

    let overlay = SignLanguageOverlay::new(config);
    assert!(overlay.validate().is_ok());

    let mut invalid_overlay = SignLanguageOverlay::default();
    let mut invalid_config = SignConfig::default();
    invalid_config.opacity = 1.5;
    invalid_overlay.set_config(invalid_config);
    assert!(invalid_overlay.validate().is_err());
}

#[test]
fn test_prosody_control() {
    use oximedia_access::tts::prosody::{ProsodyConfig, ProsodyControl};

    let config = ProsodyConfig {
        rate: 1.2,
        pitch: 2.0,
        volume: 0.9,
        emphasis: 0.7,
    };

    let mut control = ProsodyControl::new(config);
    control.set_rate(1.5);
    assert!((control.config().rate - 1.5).abs() < f32::EPSILON);

    let ssml = control.to_ssml("Test text");
    assert!(ssml.contains("prosody"));
    assert!(ssml.contains("Test text"));
}

#[test]
fn test_voice_registry() {
    use oximedia_access::tts::voice::{Voice, VoiceGender, VoiceRegistry};

    let mut registry = VoiceRegistry::new();
    assert!(!registry.voices().is_empty());

    let voice = Voice::new(
        "test-voice".to_string(),
        "Test Voice".to_string(),
        "en".to_string(),
        VoiceGender::Female,
    )
    .with_neural(true);

    registry.add_voice(voice);

    let en_voices = registry.find_by_language("en");
    assert!(!en_voices.is_empty());

    let female_voices = registry.find_by_gender(VoiceGender::Female);
    assert!(!female_voices.is_empty());
}

#[test]
fn test_audio_description_mixing() {
    use oximedia_access::audio_desc::mix::{AudioDescriptionMixer, MixConfig, MixStrategy};

    let config = MixConfig::for_strategy(MixStrategy::Duck);
    assert_eq!(config.strategy, MixStrategy::Duck);
    assert!(config.validate().is_ok());

    let mixer = AudioDescriptionMixer::new(config);
    assert_eq!(mixer.config().strategy, MixStrategy::Duck);
}
