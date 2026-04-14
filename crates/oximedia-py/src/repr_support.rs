//! `__repr__` and `__str__` implementations for Python-exposed types.
//!
//! Provides a [`ReprBuilder`] utility that constructs informative `repr()` and
//! `str()` output for any OxiMedia pyclass, following the Python convention:
//!
//! - `__repr__`: `ClassName(field=value, ...)` — round-trippable where possible
//! - `__str__`:  human-readable one-liner summary
//!
//! Also provides ready-made wrappers for common media types (resolution,
//! duration, sample rate, codec info, etc.) that can be used across modules.

use std::collections::BTreeMap;
use std::fmt;

// ---------------------------------------------------------------------------
// ReprBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing Python-style `__repr__` strings.
///
/// # Example
/// ```
/// use oximedia::repr_support::ReprBuilder;
///
/// let repr = ReprBuilder::new("EncoderConfig")
///     .field("width", &1920)
///     .field("height", &1080)
///     .field_opt("crf", &Some(28.0f32))
///     .build();
/// assert_eq!(repr, "EncoderConfig(width=1920, height=1080, crf=28.0)");
/// ```
#[derive(Debug, Clone)]
pub struct ReprBuilder {
    class_name: String,
    fields: Vec<(String, String)>,
}

impl ReprBuilder {
    /// Create a new builder for the given class name.
    pub fn new(class_name: impl Into<String>) -> Self {
        Self {
            class_name: class_name.into(),
            fields: Vec::new(),
        }
    }

    /// Append a field with a `Debug`-formatted value.
    pub fn field(mut self, name: &str, value: &dyn fmt::Debug) -> Self {
        self.fields.push((name.to_string(), format!("{value:?}")));
        self
    }

    /// Append a field that renders using `Display` instead of `Debug`.
    pub fn field_display(mut self, name: &str, value: &dyn fmt::Display) -> Self {
        self.fields.push((name.to_string(), format!("{value}")));
        self
    }

    /// Append an optional field — omitted if `None`.
    pub fn field_opt<T: fmt::Debug>(mut self, name: &str, value: &Option<T>) -> Self {
        if let Some(v) = value {
            self.fields.push((name.to_string(), format!("{v:?}")));
        }
        self
    }

    /// Append a boolean field — only shown when `true`.
    pub fn field_bool(mut self, name: &str, value: bool) -> Self {
        if value {
            self.fields.push((name.to_string(), "True".to_string()));
        }
        self
    }

    /// Append a field with a raw string value (no quoting).
    pub fn field_raw(mut self, name: &str, value: impl Into<String>) -> Self {
        self.fields.push((name.to_string(), value.into()));
        self
    }

    /// Build the final `__repr__` string: `ClassName(field=value, ...)`.
    pub fn build(&self) -> String {
        let pairs: Vec<String> = self
            .fields
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        format!("{}({})", self.class_name, pairs.join(", "))
    }
}

impl fmt::Display for ReprBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.build())
    }
}

// ---------------------------------------------------------------------------
// StrBuilder
// ---------------------------------------------------------------------------

/// Builder for human-readable `__str__` output.
///
/// Unlike `ReprBuilder`, this produces a natural-language summary rather
/// than a Python-constructor-style string.
#[derive(Debug, Clone)]
pub struct StrBuilder {
    prefix: String,
    parts: Vec<String>,
}

impl StrBuilder {
    /// Create a new builder with a leading type label.
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            parts: Vec::new(),
        }
    }

    /// Append a descriptive part.
    pub fn part(mut self, text: impl Into<String>) -> Self {
        self.parts.push(text.into());
        self
    }

    /// Append an optional part — omitted if `None`.
    pub fn part_opt(mut self, text: Option<impl Into<String>>) -> Self {
        if let Some(t) = text {
            self.parts.push(t.into());
        }
        self
    }

    /// Build the final `__str__` string: `<prefix>: <part1>, <part2>, ...`.
    pub fn build(&self) -> String {
        if self.parts.is_empty() {
            self.prefix.clone()
        } else {
            format!("{}: {}", self.prefix, self.parts.join(", "))
        }
    }
}

impl fmt::Display for StrBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.build())
    }
}

// ---------------------------------------------------------------------------
// MediaRepr — common media-type formatters
// ---------------------------------------------------------------------------

/// Utility functions for formatting common media fields in repr/str output.
pub struct MediaRepr;

impl MediaRepr {
    /// Format a resolution pair: `"1920x1080"`.
    pub fn resolution(width: u32, height: u32) -> String {
        format!("{width}x{height}")
    }

    /// Format a duration in seconds to `"HH:MM:SS.mmm"`.
    pub fn duration_hms(seconds: f64) -> String {
        if seconds < 0.0 {
            return "negative".to_string();
        }
        let total_ms = (seconds * 1000.0) as u64;
        let h = total_ms / 3_600_000;
        let m = (total_ms % 3_600_000) / 60_000;
        let s = (total_ms % 60_000) / 1000;
        let ms = total_ms % 1000;
        if h > 0 {
            format!("{h:02}:{m:02}:{s:02}.{ms:03}")
        } else {
            format!("{m:02}:{s:02}.{ms:03}")
        }
    }

    /// Format a sample rate: `"48000 Hz"` or `"48 kHz"`.
    pub fn sample_rate(hz: u32) -> String {
        if hz >= 1000 && hz % 1000 == 0 {
            format!("{} kHz", hz / 1000)
        } else {
            format!("{hz} Hz")
        }
    }

    /// Format a frame rate as a fraction or decimal: `"29.97 fps"`.
    pub fn frame_rate(num: u32, den: u32) -> String {
        if den == 0 {
            return "unknown fps".to_string();
        }
        #[allow(clippy::cast_precision_loss)]
        let fps = num as f64 / den as f64;
        if (fps - fps.round()).abs() < 0.001 {
            format!("{} fps", fps.round() as u32)
        } else {
            format!("{fps:.2} fps")
        }
    }

    /// Format a bitrate: `"3500 kbps"` or `"5.2 Mbps"`.
    pub fn bitrate_kbps(kbps: u32) -> String {
        if kbps >= 1000 {
            #[allow(clippy::cast_precision_loss)]
            let mbps = kbps as f64 / 1000.0;
            if (mbps - mbps.round()).abs() < 0.05 {
                format!("{} Mbps", mbps.round() as u32)
            } else {
                format!("{mbps:.1} Mbps")
            }
        } else {
            format!("{kbps} kbps")
        }
    }

    /// Format a byte count: `"1.5 MB"`, `"320 KB"`, `"512 B"`.
    pub fn byte_count(bytes: u64) -> String {
        const KB: u64 = 1024;
        const MB: u64 = 1024 * 1024;
        const GB: u64 = 1024 * 1024 * 1024;

        if bytes >= GB {
            #[allow(clippy::cast_precision_loss)]
            let gb = bytes as f64 / GB as f64;
            format!("{gb:.1} GB")
        } else if bytes >= MB {
            #[allow(clippy::cast_precision_loss)]
            let mb = bytes as f64 / MB as f64;
            format!("{mb:.1} MB")
        } else if bytes >= KB {
            #[allow(clippy::cast_precision_loss)]
            let kb = bytes as f64 / KB as f64;
            format!("{kb:.1} KB")
        } else {
            format!("{bytes} B")
        }
    }

    /// Format a channel count as a layout label: `"mono"`, `"stereo"`, `"5.1"`.
    pub fn channel_layout(channels: u32) -> String {
        match channels {
            1 => "mono".to_string(),
            2 => "stereo".to_string(),
            6 => "5.1".to_string(),
            8 => "7.1".to_string(),
            n => format!("{n}ch"),
        }
    }

    /// Format a codec name into a display string.
    pub fn codec_display(codec: &str) -> String {
        match codec.to_lowercase().as_str() {
            "av1" => "AV1".to_string(),
            "vp9" => "VP9".to_string(),
            "vp8" => "VP8".to_string(),
            "opus" => "Opus".to_string(),
            "vorbis" => "Vorbis".to_string(),
            "flac" => "FLAC".to_string(),
            "pcm" => "PCM".to_string(),
            "theora" => "Theora".to_string(),
            other => other.to_uppercase(),
        }
    }
}

// ---------------------------------------------------------------------------
// TypeReprMap — registry of custom repr formatters
// ---------------------------------------------------------------------------

/// Registry mapping type names to their repr/str format templates.
///
/// This allows Python wrapper code to look up the display template
/// for a given OxiMedia type at runtime.
#[derive(Debug, Clone)]
pub struct TypeReprMap {
    entries: BTreeMap<String, ReprTemplate>,
}

/// Template describing which fields to include in repr/str output.
#[derive(Debug, Clone)]
pub struct ReprTemplate {
    /// Type name displayed in repr.
    pub type_name: String,
    /// Field names (in display order) for __repr__.
    pub repr_fields: Vec<String>,
    /// Parts format string fragments for __str__.
    pub str_parts: Vec<String>,
}

impl ReprTemplate {
    /// Create a new template.
    pub fn new(type_name: impl Into<String>) -> Self {
        Self {
            type_name: type_name.into(),
            repr_fields: Vec::new(),
            str_parts: Vec::new(),
        }
    }

    /// Add a repr field name.
    pub fn with_repr_field(mut self, field: impl Into<String>) -> Self {
        self.repr_fields.push(field.into());
        self
    }

    /// Add a str part fragment.
    pub fn with_str_part(mut self, part: impl Into<String>) -> Self {
        self.str_parts.push(part.into());
        self
    }
}

impl TypeReprMap {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Register a template for a given type.
    pub fn register(&mut self, template: ReprTemplate) {
        self.entries
            .insert(template.type_name.clone(), template);
    }

    /// Look up the template for a type.
    pub fn get(&self, type_name: &str) -> Option<&ReprTemplate> {
        self.entries.get(type_name)
    }

    /// Number of registered templates.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Create a default registry pre-populated with common OxiMedia types.
    pub fn default_registry() -> Self {
        let mut map = Self::new();

        map.register(
            ReprTemplate::new("VideoFrame")
                .with_repr_field("width")
                .with_repr_field("height")
                .with_repr_field("format")
                .with_repr_field("pts")
                .with_str_part("resolution")
                .with_str_part("format")
                .with_str_part("timestamp"),
        );

        map.register(
            ReprTemplate::new("AudioFrame")
                .with_repr_field("sample_rate")
                .with_repr_field("channels")
                .with_repr_field("samples")
                .with_repr_field("format")
                .with_str_part("sample_rate")
                .with_str_part("channels")
                .with_str_part("samples"),
        );

        map.register(
            ReprTemplate::new("EncoderConfig")
                .with_repr_field("width")
                .with_repr_field("height")
                .with_repr_field("framerate")
                .with_repr_field("crf")
                .with_str_part("resolution")
                .with_str_part("framerate")
                .with_str_part("quality"),
        );

        map.register(
            ReprTemplate::new("Packet")
                .with_repr_field("stream_index")
                .with_repr_field("pts")
                .with_repr_field("size")
                .with_str_part("stream")
                .with_str_part("timestamp")
                .with_str_part("size"),
        );

        map.register(
            ReprTemplate::new("StreamInfo")
                .with_repr_field("index")
                .with_repr_field("codec")
                .with_repr_field("kind")
                .with_str_part("index")
                .with_str_part("codec")
                .with_str_part("type"),
        );

        map.register(
            ReprTemplate::new("QualityScore")
                .with_repr_field("psnr")
                .with_repr_field("ssim")
                .with_str_part("psnr")
                .with_str_part("ssim"),
        );

        map
    }

    /// List all registered type names (sorted).
    pub fn type_names(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }
}

impl Default for TypeReprMap {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── ReprBuilder ───────────────────────────────────────────────────────

    #[test]
    fn test_repr_builder_basic() {
        let r = ReprBuilder::new("Foo")
            .field("x", &42)
            .field("y", &"hello")
            .build();
        assert_eq!(r, r#"Foo(x=42, y="hello")"#);
    }

    #[test]
    fn test_repr_builder_empty_fields() {
        let r = ReprBuilder::new("Empty").build();
        assert_eq!(r, "Empty()");
    }

    #[test]
    fn test_repr_builder_field_display() {
        let r = ReprBuilder::new("Res")
            .field_display("resolution", &"1920x1080")
            .build();
        assert_eq!(r, "Res(resolution=1920x1080)");
    }

    #[test]
    fn test_repr_builder_field_opt_some() {
        let v: Option<f32> = Some(28.0);
        let r = ReprBuilder::new("Cfg").field_opt("crf", &v).build();
        assert!(r.contains("crf=28.0"));
    }

    #[test]
    fn test_repr_builder_field_opt_none() {
        let v: Option<f32> = None;
        let r = ReprBuilder::new("Cfg").field_opt("crf", &v).build();
        assert_eq!(r, "Cfg()");
    }

    #[test]
    fn test_repr_builder_field_bool_true() {
        let r = ReprBuilder::new("Opts")
            .field_bool("verbose", true)
            .build();
        assert_eq!(r, "Opts(verbose=True)");
    }

    #[test]
    fn test_repr_builder_field_bool_false() {
        let r = ReprBuilder::new("Opts")
            .field_bool("verbose", false)
            .build();
        assert_eq!(r, "Opts()");
    }

    #[test]
    fn test_repr_builder_field_raw() {
        let r = ReprBuilder::new("T")
            .field_raw("codec", "AV1")
            .build();
        assert_eq!(r, "T(codec=AV1)");
    }

    #[test]
    fn test_repr_builder_display_trait() {
        let r = ReprBuilder::new("X").field("v", &1);
        assert_eq!(format!("{r}"), "X(v=1)");
    }

    // ── StrBuilder ────────────────────────────────────────────────────────

    #[test]
    fn test_str_builder_basic() {
        let s = StrBuilder::new("Video")
            .part("1920x1080")
            .part("AV1")
            .build();
        assert_eq!(s, "Video: 1920x1080, AV1");
    }

    #[test]
    fn test_str_builder_empty_parts() {
        let s = StrBuilder::new("Empty").build();
        assert_eq!(s, "Empty");
    }

    #[test]
    fn test_str_builder_with_opt_none() {
        let s = StrBuilder::new("Audio")
            .part("48 kHz")
            .part_opt(None::<String>)
            .build();
        assert_eq!(s, "Audio: 48 kHz");
    }

    #[test]
    fn test_str_builder_display_trait() {
        let s = StrBuilder::new("T").part("ok");
        assert_eq!(format!("{s}"), "T: ok");
    }

    // ── MediaRepr ─────────────────────────────────────────────────────────

    #[test]
    fn test_resolution() {
        assert_eq!(MediaRepr::resolution(1920, 1080), "1920x1080");
        assert_eq!(MediaRepr::resolution(3840, 2160), "3840x2160");
    }

    #[test]
    fn test_duration_hms() {
        assert_eq!(MediaRepr::duration_hms(0.0), "00:00.000");
        assert_eq!(MediaRepr::duration_hms(61.5), "01:01.500");
        assert_eq!(MediaRepr::duration_hms(3661.123), "01:01:01.123");
    }

    #[test]
    fn test_duration_hms_negative() {
        assert_eq!(MediaRepr::duration_hms(-1.0), "negative");
    }

    #[test]
    fn test_sample_rate() {
        assert_eq!(MediaRepr::sample_rate(48000), "48 kHz");
        assert_eq!(MediaRepr::sample_rate(44100), "44100 Hz");
        assert_eq!(MediaRepr::sample_rate(96000), "96 kHz");
    }

    #[test]
    fn test_frame_rate() {
        assert_eq!(MediaRepr::frame_rate(30, 1), "30 fps");
        assert_eq!(MediaRepr::frame_rate(30000, 1001), "29.97 fps");
        assert_eq!(MediaRepr::frame_rate(24, 1), "24 fps");
    }

    #[test]
    fn test_frame_rate_zero_den() {
        assert_eq!(MediaRepr::frame_rate(30, 0), "unknown fps");
    }

    #[test]
    fn test_bitrate_kbps() {
        assert_eq!(MediaRepr::bitrate_kbps(500), "500 kbps");
        assert_eq!(MediaRepr::bitrate_kbps(3500), "3.5 Mbps");
        assert_eq!(MediaRepr::bitrate_kbps(5000), "5 Mbps");
    }

    #[test]
    fn test_byte_count() {
        assert_eq!(MediaRepr::byte_count(512), "512 B");
        assert_eq!(MediaRepr::byte_count(1024), "1.0 KB");
        assert_eq!(MediaRepr::byte_count(1_048_576), "1.0 MB");
        assert_eq!(MediaRepr::byte_count(1_073_741_824), "1.0 GB");
    }

    #[test]
    fn test_channel_layout() {
        assert_eq!(MediaRepr::channel_layout(1), "mono");
        assert_eq!(MediaRepr::channel_layout(2), "stereo");
        assert_eq!(MediaRepr::channel_layout(6), "5.1");
        assert_eq!(MediaRepr::channel_layout(8), "7.1");
        assert_eq!(MediaRepr::channel_layout(4), "4ch");
    }

    #[test]
    fn test_codec_display() {
        assert_eq!(MediaRepr::codec_display("av1"), "AV1");
        assert_eq!(MediaRepr::codec_display("opus"), "Opus");
        assert_eq!(MediaRepr::codec_display("flac"), "FLAC");
        assert_eq!(MediaRepr::codec_display("vorbis"), "Vorbis");
        assert_eq!(MediaRepr::codec_display("xyz"), "XYZ");
    }

    // ── TypeReprMap ───────────────────────────────────────────────────────

    #[test]
    fn test_type_repr_map_empty() {
        let m = TypeReprMap::new();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn test_type_repr_map_register_and_get() {
        let mut m = TypeReprMap::new();
        m.register(
            ReprTemplate::new("Foo")
                .with_repr_field("x")
                .with_str_part("desc"),
        );
        assert_eq!(m.len(), 1);
        let t = m.get("Foo").expect("should find Foo");
        assert_eq!(t.repr_fields, vec!["x"]);
    }

    #[test]
    fn test_type_repr_map_get_missing() {
        let m = TypeReprMap::new();
        assert!(m.get("Bar").is_none());
    }

    #[test]
    fn test_default_registry_has_common_types() {
        let m = TypeReprMap::default_registry();
        assert!(m.get("VideoFrame").is_some());
        assert!(m.get("AudioFrame").is_some());
        assert!(m.get("EncoderConfig").is_some());
        assert!(m.get("Packet").is_some());
        assert!(m.get("StreamInfo").is_some());
        assert!(m.get("QualityScore").is_some());
        assert!(m.len() >= 6);
    }

    #[test]
    fn test_type_names_sorted() {
        let m = TypeReprMap::default_registry();
        let names = m.type_names();
        let mut sorted_names = names.clone();
        sorted_names.sort();
        assert_eq!(names, sorted_names);
    }

    // ── ReprTemplate ──────────────────────────────────────────────────────

    #[test]
    fn test_repr_template_chain() {
        let t = ReprTemplate::new("Cfg")
            .with_repr_field("a")
            .with_repr_field("b")
            .with_str_part("s1")
            .with_str_part("s2");
        assert_eq!(t.type_name, "Cfg");
        assert_eq!(t.repr_fields.len(), 2);
        assert_eq!(t.str_parts.len(), 2);
    }

    // ── Integration: ReprBuilder + MediaRepr ──────────────────────────────

    #[test]
    fn test_repr_with_media_helpers() {
        let r = ReprBuilder::new("VideoStream")
            .field_raw("resolution", MediaRepr::resolution(1920, 1080))
            .field_raw("fps", MediaRepr::frame_rate(24, 1))
            .field_raw("codec", MediaRepr::codec_display("av1"))
            .build();
        assert!(r.contains("1920x1080"));
        assert!(r.contains("24 fps"));
        assert!(r.contains("AV1"));
    }

    #[test]
    fn test_str_with_media_helpers() {
        let s = StrBuilder::new("Audio Stream")
            .part(MediaRepr::sample_rate(48000))
            .part(MediaRepr::channel_layout(2))
            .part(MediaRepr::codec_display("opus"))
            .build();
        assert_eq!(s, "Audio Stream: 48 kHz, stereo, Opus");
    }
}
