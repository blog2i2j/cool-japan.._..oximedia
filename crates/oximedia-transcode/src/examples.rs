//! Comprehensive examples for common transcoding scenarios.
//!
//! This module provides pre-built example configurations for various
//! transcoding use cases.

use crate::{
    AbrLadder, AudioFilter, LoudnessStandard, MultiPassMode, NormalizationConfig, QualityMode,
    TranscodeBuilder, TranscodePipeline, VideoFilter,
};

/// `YouTube` upload optimization example.
///
/// # Example
///
/// ```rust,no_run
/// use oximedia_transcode::examples;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let config = examples::youtube_1080p_upload(
///     "input.mp4",
///     "output.mp4"
/// )?;
/// # Ok(())
/// # }
/// ```
#[must_use]
pub fn youtube_1080p_upload(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .quality(QualityMode::High)
        .build()
}

/// High-quality archival transcode example.
#[must_use]
pub fn archival_transcode(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("vp9")
        .audio_codec("opus")
        .quality(QualityMode::VeryHigh)
        .multipass(MultiPassMode::TwoPass)
        .build()
}

/// Social media optimized transcode (Instagram).
#[must_use]
pub fn instagram_square(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    let _video_filters = VideoFilter::new().scale(1080, 1080).sharpen(0.5);

    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .quality(QualityMode::High)
        .build()
}

/// Broadcast-ready transcode with loudness normalization.
#[must_use]
pub fn broadcast_hd_ebu(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    let normalization = NormalizationConfig::new(LoudnessStandard::EbuR128);

    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .quality(QualityMode::VeryHigh)
        .normalization(normalization)
        .build()
}

/// Low-latency streaming transcode.
#[must_use]
pub fn low_latency_stream(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .quality(QualityMode::Medium)
        .build()
}

/// 4K HDR transcode example.
#[must_use]
pub fn hdr_4k_transcode(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("vp9")
        .audio_codec("opus")
        .quality(QualityMode::VeryHigh)
        .multipass(MultiPassMode::TwoPass)
        .build()
}

/// Adaptive bitrate ladder for HLS/DASH.
pub fn create_abr_ladder(input: &str, output_dir: &str) {
    let ladder = AbrLadder::hls_standard();

    // In a real implementation, would create multiple output files
    // for each rung in the ladder
    for rung in &ladder.rungs {
        let output = format!("{}/output_{}p.mp4", output_dir, rung.height);
        let _ = TranscodePipelineBuilder::new()
            .input(input)
            .output(&output)
            .video_codec(&rung.codec)
            .build();
    }
}

/// Parallel batch transcode example.
pub fn batch_transcode_parallel(inputs: Vec<&str>, outputs: Vec<&str>) {
    use crate::ParallelEncodeBuilder;

    let mut builder = ParallelEncodeBuilder::new().max_parallel(4);

    for (input, output) in inputs.iter().zip(outputs.iter()) {
        if let Ok(config) = TranscodeBuilder::new()
            .input(*input)
            .output(*output)
            .video_codec("h264")
            .audio_codec("aac")
            .quality(QualityMode::Medium)
            .build()
        {
            builder = builder.add_job(config);
        }
    }

    let _encoder = builder.build();
}

/// Deinterlace and upscale SD to HD.
#[must_use]
pub fn sd_to_hd_upscale(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    let _video_filters = VideoFilter::new()
        .deinterlace()
        .scale(1920, 1080)
        .sharpen(1.0);

    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .quality(QualityMode::High)
        .build()
}

/// Crop and resize for different aspect ratios.
#[must_use]
pub fn crop_to_widescreen(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    let _video_filters = VideoFilter::new()
        .crop(1920, 800, 0, 140) // Crop to 2.40:1
        .scale(1920, 1080); // Letterbox to 16:9

    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .quality(QualityMode::High)
        .build()
}

/// Audio ducking and mixing example.
#[must_use]
pub fn audio_ducking(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    let _audio_filters = AudioFilter::new()
        .compress(-20.0, 4.0)
        .normalize(-23.0)
        .fade_in(1.0)
        .fade_out(58.0, 2.0);

    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .audio_codec("opus")
        .build()
}

/// Professional color grading workflow.
#[must_use]
pub fn color_grade(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    let _video_filters = VideoFilter::new().color_correct(0.05, 1.1, 1.15);

    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .quality(QualityMode::VeryHigh)
        .multipass(MultiPassMode::TwoPass)
        .build()
}

/// Film restoration workflow.
#[must_use]
pub fn film_restoration(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    let _video_filters = VideoFilter::new()
        .deinterlace()
        .denoise(2.0)
        .sharpen(0.8)
        .color_correct(0.0, 1.05, 1.0);

    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("vp9")
        .audio_codec("opus")
        .quality(QualityMode::VeryHigh)
        .multipass(MultiPassMode::ThreePass)
        .build()
}

/// Podcast audio optimization.
#[must_use]
pub fn podcast_audio(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    let _audio_filters = AudioFilter::new()
        .highpass(80.0) // Remove low-frequency rumble
        .compress(-18.0, 3.0) // Compress dynamic range
        .normalize(-16.0); // Normalize to podcast standard

    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .audio_codec("opus")
        .build()
}

/// Screen recording optimization.
#[must_use]
pub fn screen_recording_optimize(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("opus")
        .quality(QualityMode::High)
        .build()
}

/// Anime/animation optimized encoding.
#[must_use]
pub fn anime_encode(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("opus")
        .quality(QualityMode::VeryHigh)
        .build()
}

/// Gaming video optimization.
#[must_use]
pub fn gaming_video(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("opus")
        .quality(QualityMode::High)
        .build()
}

/// Music video encoding.
#[must_use]
pub fn music_video(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    let normalization = NormalizationConfig::new(LoudnessStandard::Spotify);

    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("vp9")
        .audio_codec("opus")
        .quality(QualityMode::VeryHigh)
        .normalization(normalization)
        .multipass(MultiPassMode::TwoPass)
        .build()
}

/// News broadcast optimized.
#[must_use]
pub fn news_broadcast(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    let normalization = NormalizationConfig::new(LoudnessStandard::AtscA85);

    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .quality(QualityMode::VeryHigh)
        .normalization(normalization)
        .build()
}

/// Sports broadcast encoding.
#[must_use]
pub fn sports_broadcast(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .quality(QualityMode::VeryHigh)
        .build()
}

/// E-learning content optimization.
#[must_use]
pub fn elearning_content(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .quality(QualityMode::Medium)
        .build()
}

/// Security camera footage optimization.
#[must_use]
pub fn security_footage(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("opus")
        .quality(QualityMode::Low)
        .build()
}

/// Time-lapse video creation.
#[must_use]
pub fn timelapse_create(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    let _video_filters = VideoFilter::new().framerate(30.0);

    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .quality(QualityMode::High)
        .build()
}

/// Slow motion video creation.
#[must_use]
pub fn slow_motion_create(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    let _video_filters = VideoFilter::new().framerate(120.0);

    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .quality(QualityMode::VeryHigh)
        .build()
}

/// Documentary film encoding.
#[must_use]
pub fn documentary_encode(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    let normalization = NormalizationConfig::new(LoudnessStandard::EbuR128);

    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("vp9")
        .audio_codec("opus")
        .quality(QualityMode::VeryHigh)
        .normalization(normalization)
        .multipass(MultiPassMode::TwoPass)
        .build()
}

/// Corporate video production.
#[must_use]
pub fn corporate_video(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .quality(QualityMode::High)
        .build()
}

/// Wedding video encoding.
#[must_use]
pub fn wedding_video(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    let _video_filters = VideoFilter::new().color_correct(0.1, 1.05, 1.1);

    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .quality(QualityMode::VeryHigh)
        .multipass(MultiPassMode::TwoPass)
        .build()
}

/// Real estate video tour.
#[must_use]
pub fn real_estate_tour(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .quality(QualityMode::High)
        .build()
}

/// Medical/scientific video encoding.
#[must_use]
pub fn medical_video(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .quality(QualityMode::VeryHigh)
        .build()
}

/// Drone footage optimization.
#[must_use]
pub fn drone_footage(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    let _video_filters = VideoFilter::new().denoise(1.0).sharpen(0.5);

    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("vp9")
        .audio_codec("opus")
        .quality(QualityMode::VeryHigh)
        .multipass(MultiPassMode::TwoPass)
        .build()
}

/// GoPro/action camera footage.
#[must_use]
pub fn action_camera_footage(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    let _video_filters = VideoFilter::new().denoise(1.5);

    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .quality(QualityMode::High)
        .build()
}

/// Product demo video.
#[must_use]
pub fn product_demo(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .quality(QualityMode::High)
        .build()
}

/// Recipe/cooking video.
#[must_use]
pub fn cooking_video(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    let _video_filters = VideoFilter::new().color_correct(0.05, 1.1, 1.2); // Enhance food colors

    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .quality(QualityMode::High)
        .build()
}

/// Travel vlog encoding.
#[must_use]
pub fn travel_vlog(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    let _video_filters = VideoFilter::new().color_correct(0.05, 1.05, 1.1);

    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .quality(QualityMode::High)
        .build()
}

/// Fashion/beauty video.
#[must_use]
pub fn fashion_video(input: &str, output: &str) -> crate::Result<TranscodePipeline> {
    let _video_filters = VideoFilter::new()
        .sharpen(0.5)
        .color_correct(0.1, 1.05, 1.05);

    TranscodePipelineBuilder::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .quality(QualityMode::VeryHigh)
        .build()
}

/// Placeholder builder struct (temporary for examples).
struct TranscodePipelineBuilder {
    input: String,
    output: String,
    video_codec: Option<String>,
    audio_codec: Option<String>,
    quality: Option<QualityMode>,
    multipass: Option<MultiPassMode>,
    normalization: Option<NormalizationConfig>,
}

impl TranscodePipelineBuilder {
    fn new() -> Self {
        Self {
            input: String::new(),
            output: String::new(),
            video_codec: None,
            audio_codec: None,
            quality: None,
            multipass: None,
            normalization: None,
        }
    }

    fn input(mut self, input: &str) -> Self {
        self.input = input.to_string();
        self
    }

    fn output(mut self, output: &str) -> Self {
        self.output = output.to_string();
        self
    }

    fn video_codec(mut self, codec: &str) -> Self {
        self.video_codec = Some(codec.to_string());
        self
    }

    fn audio_codec(mut self, codec: &str) -> Self {
        self.audio_codec = Some(codec.to_string());
        self
    }

    fn quality(mut self, quality: QualityMode) -> Self {
        self.quality = Some(quality);
        self
    }

    fn multipass(mut self, mode: MultiPassMode) -> Self {
        self.multipass = Some(mode);
        self
    }

    fn normalization(mut self, config: NormalizationConfig) -> Self {
        self.normalization = Some(config);
        self
    }

    fn build(self) -> crate::Result<TranscodePipeline> {
        TranscodePipeline::builder()
            .input(&self.input)
            .output(&self.output)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-transcode-examples-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn test_youtube_example() {
        let i = tmp_str("input.mp4");
        let o = tmp_str("output.mp4");
        let _pipeline = youtube_1080p_upload(&i, &o);
    }

    #[test]
    fn test_archival_example() {
        let i = tmp_str("input.mp4");
        let o = tmp_str("output.mkv");
        let _pipeline = archival_transcode(&i, &o);
    }

    #[test]
    fn test_instagram_example() {
        let i = tmp_str("input.mp4");
        let o = tmp_str("output.mp4");
        let _pipeline = instagram_square(&i, &o);
    }
}
