use std::str::FromStr;

use crate::diagnostics::TranslationError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncoderQualityPreset {
    Ultrafast,
    Superfast,
    Veryfast,
    Faster,
    Fast,
    Medium,
    Slow,
    Slower,
    Veryslow,
    Placebo,
}

impl FromStr for EncoderQualityPreset {
    type Err = TranslationError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ultrafast" => Ok(Self::Ultrafast),
            "superfast" => Ok(Self::Superfast),
            "veryfast" => Ok(Self::Veryfast),
            "faster" => Ok(Self::Faster),
            "fast" => Ok(Self::Fast),
            "medium" => Ok(Self::Medium),
            "slow" => Ok(Self::Slow),
            "slower" => Ok(Self::Slower),
            "veryslow" => Ok(Self::Veryslow),
            "placebo" => Ok(Self::Placebo),
            _ => Err(TranslationError::ParseError(format!(
                "unknown preset '{}'",
                s
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncoderTune {
    Film,
    Animation,
    Grain,
    Stillimage,
    Fastdecode,
    Zerolatency,
    Psnr,
    Ssim,
}

impl FromStr for EncoderTune {
    type Err = TranslationError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "film" => Ok(Self::Film),
            "animation" => Ok(Self::Animation),
            "grain" => Ok(Self::Grain),
            "stillimage" => Ok(Self::Stillimage),
            "fastdecode" => Ok(Self::Fastdecode),
            "zerolatency" => Ok(Self::Zerolatency),
            "psnr" => Ok(Self::Psnr),
            "ssim" => Ok(Self::Ssim),
            _ => Err(TranslationError::ParseError(format!(
                "unknown tune '{}'",
                s
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncoderProfile {
    Baseline,
    Main,
    High,
    High10,
    High422,
    High444,
}

impl FromStr for EncoderProfile {
    type Err = TranslationError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "baseline" => Ok(Self::Baseline),
            "main" => Ok(Self::Main),
            "high" => Ok(Self::High),
            "high10" => Ok(Self::High10),
            "high422" => Ok(Self::High422),
            "high444" => Ok(Self::High444),
            _ => Err(TranslationError::ParseError(format!(
                "unknown profile '{}'",
                s
            ))),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EncoderQualityOptions {
    pub preset: Option<EncoderQualityPreset>,
    pub tune: Option<EncoderTune>,
    pub profile: Option<EncoderProfile>,
}
