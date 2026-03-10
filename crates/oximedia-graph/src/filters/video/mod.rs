//! Video filters for the filter graph.
//!
//! This module provides video processing filters including:
//!
//! - [`PassthroughFilter`] - Passes video frames through unchanged
//! - [`NullSink`] - Discards frames (useful for benchmarking)
//! - [`ScaleFilter`] - Rescales video to different resolutions
//! - [`CropFilter`] - Crops video to a specific region
//! - [`PadFilter`] - Adds padding/borders around video
//! - [`ColorConvertFilter`] - Converts between color spaces
//! - [`FpsFilter`] - Adjusts frame rate
//! - [`DeinterlaceFilter`] - Converts interlaced to progressive
//! - [`IvtcFilter`] - Inverse telecine (removes pulldown)
//! - [`OverlayFilter`] - Composites videos together
//! - [`TonemapFilter`] - HDR to SDR tone mapping
//! - [`DelogoFilter`] - Removes logos and watermarks
//! - [`DenoiseFilter`] - Spatial and temporal noise reduction
//! - [`Lut3dFilter`] - 3D LUT color grading
//! - [`TimecodeFilter`] - Burn-in timecode and metadata overlay

mod colorspace;
mod crop;
mod deinterlace;
mod delogo;
mod denoise;
mod fps;
mod grading;
mod ivtc;
mod log;
mod lut;
mod null;
mod overlay;
mod pad;
mod passthrough;
mod scale;
mod timecode;
mod tonemap;

// Re-export passthrough and null
pub use null::NullSink;
pub use passthrough::PassthroughFilter;

// Re-export scale filter
pub use scale::{
    calculate_aspect_fill, calculate_aspect_fit, BilinearScaler, NearestNeighborScaler,
    ScaleAlgorithm, ScaleConfig, ScaleFilter,
};

// Re-export crop filter
pub use crop::{BorderDetector, CropConfig, CropFilter, CropRegion};

// Re-export pad filter
pub use pad::{
    letterbox_16_9, letterbox_4_3, letterbox_cinemascope, PadColor, PadConfig, PadFilter, PadValues,
};

// Re-export colorspace filter
pub use colorspace::{
    downsample_444_to_420, upsample_420_to_444, ChromaFormat, ColorConvertConfig,
    ColorConvertFilter, ColorMatrix,
};

// Re-export fps filter
pub use fps::{EofAction, FpsConfig, FpsFilter, FpsMode, FrameRateDetector};

// Re-export deinterlace filter
pub use deinterlace::{
    DeinterlaceConfig, DeinterlaceFilter, DeinterlaceMode, FieldOrder, InterlaceDetector,
};

// Re-export ivtc filter
pub use ivtc::{
    debug, framerate, CadenceDetector, CustomPattern, DetectionSensitivity, FieldMetrics,
    IvtcConfig, IvtcFilter, MatchMode, MotionCompensation, PatternAnalysisResults, PatternAnalyzer,
    PostProcessMode, TelecinePattern,
};

// Re-export overlay filter
pub use overlay::{
    create_color_overlay, create_gradient_overlay, Alignment, BlendMode, OverlayConfig,
    OverlayFilter,
};

// Re-export tonemap filter
pub use tonemap::{
    ColorMatrix3x3, ColorPrimaries, HdrMetadata, TonemapAlgorithm, TonemapConfig, TonemapFilter,
    TonemapParams, TransferFunction,
};

// Re-export delogo filter
pub use delogo::{
    advanced_inpainting, color, detection, mask, metrics, DelogoConfig, DelogoFilter, DelogoMethod,
    LogoDetection, Rectangle,
};

// Re-export denoise filter
pub use denoise::{DenoiseConfig, DenoiseFilter, DenoiseMethod, MotionQuality, TemporalMode};

// Re-export lut filter
pub use lut::{
    export_3dl_file, export_csv_file, export_cube_file, load_lut_file, parse_3dl_file,
    parse_csv_file, parse_cube_file, procedural, utils, CacheStats, ColorChannel, GpuLutHints,
    Lut1d, Lut3d, Lut3dConfig, Lut3dFilter, LutAnalysis, LutBlendMode, LutCache, LutColorSpace,
    LutFormat, LutInterpolation, LutSize, RgbColor,
};

// Re-export timecode filter
pub use timecode::{
    presets, templates, Color, FrameContext, MetadataField, MetadataTemplate, MultiLineText,
    OverlayElement, Position, ProgressBar, SafeAreaOverlay, TextAlignment, TextStyle,
    TimecodeConfig, TimecodeFilter, TimecodeFormat,
};

// Re-export grading filter
pub use grading::{
    AscCdl, ColorGradingConfig, ColorGradingFilter, ColorWheel, ColorWheels, Curve,
    CurveInterpolation, CurvePoint, HslColor, HslQualifier, HueVsLumCurve, HueVsSatCurve,
    LiftGammaGain, LogOffsetPower, RgbCurves, TemperatureTint,
};

// Re-export log filter
pub use log::{
    AcesCct, AcesProxy10, ArriLogC3, ArriLogC4, BlackmagicFilm5, CanonCLog, CineonLog, DjiDLog,
    LogConverter, LogDirection, LogFormat, LogLinearFilter, PanasonicVLog, RedLog3G10, SonySLog3,
};
