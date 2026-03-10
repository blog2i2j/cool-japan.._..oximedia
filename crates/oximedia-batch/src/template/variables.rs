//! Template variable definitions and presets

use crate::template::TemplateContext;
use chrono::Utc;
use std::collections::HashMap;

/// Preset template variables
pub struct PresetVariables;

impl PresetVariables {
    /// Load date/time variables
    pub fn load_datetime(context: &mut TemplateContext) {
        let now = Utc::now();

        context.set("date".to_string(), now.format("%Y-%m-%d").to_string());
        context.set("time".to_string(), now.format("%H:%M:%S").to_string());
        context.set("timestamp".to_string(), now.timestamp().to_string());
        context.set("year".to_string(), now.format("%Y").to_string());
        context.set("month".to_string(), now.format("%m").to_string());
        context.set("day".to_string(), now.format("%d").to_string());
        context.set("hour".to_string(), now.format("%H").to_string());
        context.set("minute".to_string(), now.format("%M").to_string());
        context.set("second".to_string(), now.format("%S").to_string());
    }

    /// Load system variables
    pub fn load_system(context: &mut TemplateContext) {
        if let Ok(hostname) = hostname::get() {
            context.set(
                "hostname".to_string(),
                hostname.to_string_lossy().to_string(),
            );
        }

        if let Ok(user) = std::env::var("USER") {
            context.set("user".to_string(), user);
        }

        context.set("os".to_string(), std::env::consts::OS.to_string());
        context.set("arch".to_string(), std::env::consts::ARCH.to_string());
    }

    /// Load environment variables
    pub fn load_env(context: &mut TemplateContext, prefix: Option<&str>) {
        for (key, value) in std::env::vars() {
            if let Some(p) = prefix {
                if key.starts_with(p) {
                    context.set(format!("env_{key}"), value);
                }
            } else {
                context.set(format!("env_{key}"), value);
            }
        }
    }
}

/// Common template presets
pub struct TemplatePresets;

impl TemplatePresets {
    /// Get web transcoding template
    #[must_use]
    pub fn web_transcode() -> HashMap<String, String> {
        let mut preset = HashMap::new();
        preset.insert("output".to_string(), "{stem}_web.mp4".to_string());
        preset.insert("codec".to_string(), "h264".to_string());
        preset.insert("bitrate".to_string(), "2000k".to_string());
        preset.insert("resolution".to_string(), "1280x720".to_string());
        preset
    }

    /// Get mobile transcoding template
    #[must_use]
    pub fn mobile_transcode() -> HashMap<String, String> {
        let mut preset = HashMap::new();
        preset.insert("output".to_string(), "{stem}_mobile.mp4".to_string());
        preset.insert("codec".to_string(), "h264".to_string());
        preset.insert("bitrate".to_string(), "1000k".to_string());
        preset.insert("resolution".to_string(), "854x480".to_string());
        preset
    }

    /// Get broadcast transcoding template
    #[must_use]
    pub fn broadcast_transcode() -> HashMap<String, String> {
        let mut preset = HashMap::new();
        preset.insert("output".to_string(), "{stem}_broadcast.mxf".to_string());
        preset.insert("codec".to_string(), "mpeg2video".to_string());
        preset.insert("bitrate".to_string(), "50000k".to_string());
        preset.insert("resolution".to_string(), "1920x1080".to_string());
        preset
    }

    /// Get archive template
    #[must_use]
    pub fn archive() -> HashMap<String, String> {
        let mut preset = HashMap::new();
        preset.insert(
            "output".to_string(),
            "archive/{year}/{month}/{stem}.mov".to_string(),
        );
        preset.insert("codec".to_string(), "prores".to_string());
        preset
    }

    /// Get proxy template
    #[must_use]
    pub fn proxy() -> HashMap<String, String> {
        let mut preset = HashMap::new();
        preset.insert("output".to_string(), "proxies/{stem}_proxy.mp4".to_string());
        preset.insert("codec".to_string(), "h264".to_string());
        preset.insert("bitrate".to_string(), "500k".to_string());
        preset.insert("resolution".to_string(), "640x360".to_string());
        preset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_datetime() {
        let mut context = TemplateContext::new();
        PresetVariables::load_datetime(&mut context);

        assert!(context.get("date").is_some());
        assert!(context.get("time").is_some());
        assert!(context.get("timestamp").is_some());
        assert!(context.get("year").is_some());
        assert!(context.get("month").is_some());
        assert!(context.get("day").is_some());
    }

    #[test]
    fn test_load_system() {
        let mut context = TemplateContext::new();
        PresetVariables::load_system(&mut context);

        assert!(context.get("os").is_some());
        assert!(context.get("arch").is_some());
    }

    #[test]
    fn test_web_transcode_preset() {
        let preset = TemplatePresets::web_transcode();

        assert_eq!(preset.get("codec"), Some(&"h264".to_string()));
        assert_eq!(preset.get("bitrate"), Some(&"2000k".to_string()));
        assert!(preset.contains_key("output"));
    }

    #[test]
    fn test_mobile_transcode_preset() {
        let preset = TemplatePresets::mobile_transcode();

        assert_eq!(preset.get("codec"), Some(&"h264".to_string()));
        assert_eq!(preset.get("bitrate"), Some(&"1000k".to_string()));
    }

    #[test]
    fn test_broadcast_transcode_preset() {
        let preset = TemplatePresets::broadcast_transcode();

        assert_eq!(preset.get("codec"), Some(&"mpeg2video".to_string()));
        assert!(preset
            .get("output")
            .expect("failed to get value")
            .ends_with(".mxf"));
    }

    #[test]
    fn test_archive_preset() {
        let preset = TemplatePresets::archive();

        assert_eq!(preset.get("codec"), Some(&"prores".to_string()));
        assert!(preset
            .get("output")
            .expect("failed to get value")
            .contains("{year}"));
    }

    #[test]
    fn test_proxy_preset() {
        let preset = TemplatePresets::proxy();

        assert_eq!(preset.get("codec"), Some(&"h264".to_string()));
        assert!(preset
            .get("output")
            .expect("failed to get value")
            .contains("proxy"));
    }
}
