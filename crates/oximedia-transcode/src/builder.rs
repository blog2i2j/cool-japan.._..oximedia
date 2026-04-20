//! Fluent builder API for creating transcode configurations.

use crate::{
    AbrLadder, MultiPassMode, NormalizationConfig, PresetConfig, QualityConfig, QualityMode,
    RateControlMode, Result, TranscodeConfig, TranscodeError,
};

/// Fluent builder for creating transcode configurations.
pub struct TranscodeBuilder {
    config: TranscodeConfig,
}

impl TranscodeBuilder {
    /// Creates a new transcode builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: TranscodeConfig::default(),
        }
    }

    /// Sets the input file path.
    #[must_use]
    pub fn input(mut self, path: impl Into<String>) -> Self {
        self.config.input = Some(path.into());
        self
    }

    /// Sets the output file path.
    #[must_use]
    pub fn output(mut self, path: impl Into<String>) -> Self {
        self.config.output = Some(path.into());
        self
    }

    /// Sets the video codec.
    #[must_use]
    pub fn video_codec(mut self, codec: impl Into<String>) -> Self {
        self.config.video_codec = Some(codec.into());
        self
    }

    /// Sets the audio codec.
    #[must_use]
    pub fn audio_codec(mut self, codec: impl Into<String>) -> Self {
        self.config.audio_codec = Some(codec.into());
        self
    }

    /// Sets the video bitrate in bits per second.
    #[must_use]
    pub fn video_bitrate(mut self, bitrate: u64) -> Self {
        self.config.video_bitrate = Some(bitrate);
        self
    }

    /// Sets the audio bitrate in bits per second.
    #[must_use]
    pub fn audio_bitrate(mut self, bitrate: u64) -> Self {
        self.config.audio_bitrate = Some(bitrate);
        self
    }

    /// Sets the output resolution.
    #[must_use]
    pub fn resolution(mut self, width: u32, height: u32) -> Self {
        self.config.width = Some(width);
        self.config.height = Some(height);
        self
    }

    /// Sets the output frame rate.
    #[must_use]
    pub fn frame_rate(mut self, num: u32, den: u32) -> Self {
        self.config.frame_rate = Some((num, den));
        self
    }

    /// Sets the multi-pass encoding mode.
    #[must_use]
    pub fn multi_pass(mut self, mode: MultiPassMode) -> Self {
        self.config.multi_pass = Some(mode);
        self
    }

    /// Sets the quality mode.
    #[must_use]
    pub fn quality(mut self, mode: QualityMode) -> Self {
        self.config.quality_mode = Some(mode);
        self
    }

    /// Enables audio normalization with the default standard (EBU R128).
    #[must_use]
    pub fn normalize_audio(mut self) -> Self {
        self.config.normalize_audio = true;
        self
    }

    /// Sets a specific loudness standard for audio normalization.
    #[must_use]
    pub fn loudness_standard(mut self, standard: crate::LoudnessStandard) -> Self {
        self.config.loudness_standard = Some(standard);
        self.config.normalize_audio = true;
        self
    }

    /// Enables or disables hardware acceleration.
    #[must_use]
    pub fn hw_accel(mut self, enable: bool) -> Self {
        self.config.hw_accel = enable;
        self
    }

    /// Enables or disables metadata preservation.
    #[must_use]
    pub fn preserve_metadata(mut self, enable: bool) -> Self {
        self.config.preserve_metadata = enable;
        self
    }

    /// Sets the subtitle mode.
    #[must_use]
    pub fn subtitles(mut self, mode: crate::SubtitleMode) -> Self {
        self.config.subtitle_mode = Some(mode);
        self
    }

    /// Sets the chapter mode.
    #[must_use]
    pub fn chapters(mut self, mode: crate::ChapterMode) -> Self {
        self.config.chapter_mode = Some(mode);
        self
    }

    /// Applies a preset configuration.
    #[must_use]
    pub fn preset(mut self, preset: PresetConfig) -> Self {
        if let Some(codec) = preset.video_codec {
            self.config.video_codec = Some(codec);
        }
        if let Some(codec) = preset.audio_codec {
            self.config.audio_codec = Some(codec);
        }
        if let Some(bitrate) = preset.video_bitrate {
            self.config.video_bitrate = Some(bitrate);
        }
        if let Some(bitrate) = preset.audio_bitrate {
            self.config.audio_bitrate = Some(bitrate);
        }
        if let Some(width) = preset.width {
            self.config.width = Some(width);
        }
        if let Some(height) = preset.height {
            self.config.height = Some(height);
        }
        if let Some(fps) = preset.frame_rate {
            self.config.frame_rate = Some(fps);
        }
        if let Some(mode) = preset.quality_mode {
            self.config.quality_mode = Some(mode);
        }
        self
    }

    /// Builds the transcode configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn build(self) -> Result<TranscodeConfig> {
        // Validate required fields
        if self.config.input.is_none() {
            return Err(TranscodeError::InvalidInput(
                "Input path is required".to_string(),
            ));
        }

        if self.config.output.is_none() {
            return Err(TranscodeError::InvalidOutput(
                "Output path is required".to_string(),
            ));
        }

        Ok(self.config)
    }

    /// Builds and validates the configuration, consuming the builder.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or validation fails.
    pub fn validate(self) -> Result<TranscodeConfig> {
        let config = self.build()?;

        // Perform additional validation
        use crate::validation::{InputValidator, OutputValidator};

        if let Some(ref input) = config.input {
            InputValidator::validate_path(input)?;
        }

        if let Some(ref output) = config.output {
            OutputValidator::validate_path(output, true)?;
        }

        if let Some(ref codec) = config.video_codec {
            OutputValidator::validate_codec(codec)?;
        }

        if let Some(ref codec) = config.audio_codec {
            OutputValidator::validate_codec(codec)?;
        }

        if let (Some(width), Some(height)) = (config.width, config.height) {
            OutputValidator::validate_resolution(width, height)?;
        }

        if let Some((num, den)) = config.frame_rate {
            OutputValidator::validate_frame_rate(num, den)?;
        }

        Ok(config)
    }
}

impl Default for TranscodeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced builder with fluent API for complex scenarios.
pub struct AdvancedTranscodeBuilder {
    #[allow(dead_code)]
    builder: TranscodeBuilder,
    quality_config: Option<QualityConfig>,
    #[allow(dead_code)]
    normalization_config: Option<NormalizationConfig>,
    #[allow(dead_code)]
    abr_ladder: Option<AbrLadder>,
}

#[allow(dead_code)]
impl AdvancedTranscodeBuilder {
    /// Creates a new advanced builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            builder: TranscodeBuilder::new(),
            quality_config: None,
            normalization_config: None,
            abr_ladder: None,
        }
    }

    /// Sets the input file.
    #[must_use]
    pub fn input(mut self, path: impl Into<String>) -> Self {
        self.builder = self.builder.input(path);
        self
    }

    /// Sets the output file.
    #[must_use]
    pub fn output(mut self, path: impl Into<String>) -> Self {
        self.builder = self.builder.output(path);
        self
    }

    /// Sets the quality configuration.
    #[must_use]
    #[allow(dead_code)]
    pub fn quality_config(mut self, config: QualityConfig) -> Self {
        self.quality_config = Some(config);
        self
    }

    /// Sets the normalization configuration.
    #[allow(dead_code)]
    #[must_use]
    pub fn normalization_config(mut self, config: NormalizationConfig) -> Self {
        self.normalization_config = Some(config);
        self
    }

    /// Sets the ABR ladder for adaptive streaming.
    #[allow(dead_code)]
    #[must_use]
    pub fn abr_ladder(mut self, ladder: AbrLadder) -> Self {
        self.abr_ladder = Some(ladder);
        self
    }

    /// Sets constant quality mode with CRF.
    #[must_use]
    pub fn crf(mut self, value: u8) -> Self {
        if let Some(ref mut config) = self.quality_config {
            config.rate_control = RateControlMode::Crf(value);
        } else {
            let mut config = QualityConfig::default();
            config.rate_control = RateControlMode::Crf(value);
            self.quality_config = Some(config);
        }
        self
    }

    /// Sets constant bitrate mode.
    #[must_use]
    pub fn cbr(mut self, bitrate: u64) -> Self {
        if let Some(ref mut config) = self.quality_config {
            config.rate_control = RateControlMode::Cbr(bitrate);
        } else {
            let mut config = QualityConfig::default();
            config.rate_control = RateControlMode::Cbr(bitrate);
            self.quality_config = Some(config);
        }
        self.builder = self.builder.video_bitrate(bitrate);
        self
    }

    /// Sets variable bitrate mode.
    #[must_use]
    pub fn vbr(mut self, target: u64, max: u64) -> Self {
        if let Some(ref mut config) = self.quality_config {
            config.rate_control = RateControlMode::Vbr { target, max };
        } else {
            let mut config = QualityConfig::default();
            config.rate_control = RateControlMode::Vbr { target, max };
            self.quality_config = Some(config);
        }
        self.builder = self.builder.video_bitrate(target);
        self
    }

    /// Builds the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn build(self) -> Result<TranscodeConfig> {
        self.builder.build()
    }
}

impl Default for AdvancedTranscodeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_in() -> String {
        std::env::temp_dir()
            .join("oximedia-transcode-builder-input.mp4")
            .to_string_lossy()
            .into_owned()
    }

    fn tmp_out() -> String {
        std::env::temp_dir()
            .join("oximedia-transcode-builder-output.mp4")
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn test_builder_basic() {
        let (ti, to) = (tmp_in(), tmp_out());
        let config = TranscodeBuilder::new()
            .input(&ti)
            .output(&to)
            .video_codec("vp9")
            .audio_codec("opus")
            .build()
            .expect("should succeed in test");

        assert_eq!(config.input, Some(ti));
        assert_eq!(config.output, Some(to));
        assert_eq!(config.video_codec, Some("vp9".to_string()));
        assert_eq!(config.audio_codec, Some("opus".to_string()));
    }

    #[test]
    fn test_builder_missing_input() {
        let result = TranscodeBuilder::new().output(tmp_out()).build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_missing_output() {
        let result = TranscodeBuilder::new().input(tmp_in()).build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_with_resolution() {
        let config = TranscodeBuilder::new()
            .input(tmp_in())
            .output(tmp_out())
            .resolution(1920, 1080)
            .build()
            .expect("should succeed in test");

        assert_eq!(config.width, Some(1920));
        assert_eq!(config.height, Some(1080));
    }

    #[test]
    fn test_builder_with_quality() {
        let config = TranscodeBuilder::new()
            .input(tmp_in())
            .output(tmp_out())
            .quality(QualityMode::High)
            .build()
            .expect("should succeed in test");

        assert_eq!(config.quality_mode, Some(QualityMode::High));
    }

    #[test]
    fn test_builder_with_multipass() {
        let config = TranscodeBuilder::new()
            .input(tmp_in())
            .output(tmp_out())
            .multi_pass(MultiPassMode::TwoPass)
            .build()
            .expect("should succeed in test");

        assert_eq!(config.multi_pass, Some(MultiPassMode::TwoPass));
    }

    #[test]
    fn test_builder_with_normalization() {
        let config = TranscodeBuilder::new()
            .input(tmp_in())
            .output(tmp_out())
            .normalize_audio()
            .build()
            .expect("should succeed in test");

        assert!(config.normalize_audio);
    }

    #[test]
    fn test_advanced_builder_crf() {
        let (ti, to) = (tmp_in(), tmp_out());
        let config = AdvancedTranscodeBuilder::new()
            .input(&ti)
            .output(&to)
            .crf(23)
            .build()
            .expect("should succeed in test");

        assert_eq!(config.input, Some(ti));
        assert_eq!(config.output, Some(to));
    }

    #[test]
    fn test_advanced_builder_cbr() {
        let config = AdvancedTranscodeBuilder::new()
            .input(tmp_in())
            .output(tmp_out())
            .cbr(5_000_000)
            .build()
            .expect("should succeed in test");

        assert_eq!(config.video_bitrate, Some(5_000_000));
    }

    #[test]
    fn test_advanced_builder_vbr() {
        let config = AdvancedTranscodeBuilder::new()
            .input(tmp_in())
            .output(tmp_out())
            .vbr(5_000_000, 8_000_000)
            .build()
            .expect("should succeed in test");

        assert_eq!(config.video_bitrate, Some(5_000_000));
    }
}
