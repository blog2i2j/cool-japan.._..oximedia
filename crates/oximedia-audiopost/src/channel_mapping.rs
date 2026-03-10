#![allow(dead_code)]
//! Channel mapping and routing configuration for audio post-production.
//!
//! Defines how audio channels are mapped between source and destination
//! formats (e.g., stereo to 5.1, 7.1 to Atmos beds). Supports arbitrary
//! gain matrices, channel labelling, and validation of routing configurations.

use std::collections::HashMap;

/// Standard channel labels following common broadcast conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelLabel {
    /// Left front.
    Left,
    /// Right front.
    Right,
    /// Centre front.
    Center,
    /// Low-frequency effects.
    Lfe,
    /// Left surround.
    LeftSurround,
    /// Right surround.
    RightSurround,
    /// Left rear surround (7.1).
    LeftRear,
    /// Right rear surround (7.1).
    RightRear,
    /// Left height.
    LeftHeight,
    /// Right height.
    RightHeight,
    /// Mono / single channel.
    Mono,
    /// Custom labelled channel.
    Custom(u16),
}

impl std::fmt::Display for ChannelLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Left => write!(f, "L"),
            Self::Right => write!(f, "R"),
            Self::Center => write!(f, "C"),
            Self::Lfe => write!(f, "LFE"),
            Self::LeftSurround => write!(f, "Ls"),
            Self::RightSurround => write!(f, "Rs"),
            Self::LeftRear => write!(f, "Lrs"),
            Self::RightRear => write!(f, "Rrs"),
            Self::LeftHeight => write!(f, "Lh"),
            Self::RightHeight => write!(f, "Rh"),
            Self::Mono => write!(f, "M"),
            Self::Custom(id) => write!(f, "Ch{id}"),
        }
    }
}

/// A standard channel layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelLayout {
    /// Single channel.
    Mono,
    /// Two-channel stereo.
    Stereo,
    /// 5.1 surround (L, R, C, LFE, Ls, Rs).
    Surround51,
    /// 7.1 surround (L, R, C, LFE, Ls, Rs, Lrs, Rrs).
    Surround71,
    /// Custom layout.
    Custom(Vec<ChannelLabel>),
}

impl ChannelLayout {
    /// Return the channels for this layout.
    pub fn channels(&self) -> Vec<ChannelLabel> {
        match self {
            Self::Mono => vec![ChannelLabel::Mono],
            Self::Stereo => vec![ChannelLabel::Left, ChannelLabel::Right],
            Self::Surround51 => vec![
                ChannelLabel::Left,
                ChannelLabel::Right,
                ChannelLabel::Center,
                ChannelLabel::Lfe,
                ChannelLabel::LeftSurround,
                ChannelLabel::RightSurround,
            ],
            Self::Surround71 => vec![
                ChannelLabel::Left,
                ChannelLabel::Right,
                ChannelLabel::Center,
                ChannelLabel::Lfe,
                ChannelLabel::LeftSurround,
                ChannelLabel::RightSurround,
                ChannelLabel::LeftRear,
                ChannelLabel::RightRear,
            ],
            Self::Custom(channels) => channels.clone(),
        }
    }

    /// Return the number of channels.
    pub fn channel_count(&self) -> usize {
        self.channels().len()
    }
}

impl std::fmt::Display for ChannelLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mono => write!(f, "Mono"),
            Self::Stereo => write!(f, "Stereo"),
            Self::Surround51 => write!(f, "5.1"),
            Self::Surround71 => write!(f, "7.1"),
            Self::Custom(ch) => write!(f, "Custom({}ch)", ch.len()),
        }
    }
}

/// A single routing connection from one source channel to one destination channel.
#[derive(Debug, Clone)]
pub struct ChannelRoute {
    /// Source channel index.
    pub src_index: usize,
    /// Source channel label.
    pub src_label: ChannelLabel,
    /// Destination channel index.
    pub dst_index: usize,
    /// Destination channel label.
    pub dst_label: ChannelLabel,
    /// Gain multiplier applied to this route (linear, 1.0 = unity).
    pub gain: f64,
    /// Whether this route is currently active.
    pub active: bool,
}

impl ChannelRoute {
    /// Create a new route with unity gain.
    pub fn new(
        src_index: usize,
        src_label: ChannelLabel,
        dst_index: usize,
        dst_label: ChannelLabel,
    ) -> Self {
        Self {
            src_index,
            src_label,
            dst_index,
            dst_label,
            gain: 1.0,
            active: true,
        }
    }

    /// Set the gain for this route.
    pub fn with_gain(mut self, gain: f64) -> Self {
        self.gain = gain;
        self
    }

    /// Disable this route.
    pub fn disabled(mut self) -> Self {
        self.active = false;
        self
    }

    /// Convert gain to decibels.
    pub fn gain_db(&self) -> f64 {
        if self.gain <= 0.0 {
            return f64::NEG_INFINITY;
        }
        20.0 * self.gain.log10()
    }
}

/// A complete channel mapping configuration.
#[derive(Debug, Clone)]
pub struct ChannelMapping {
    /// Identifier for this mapping.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Source layout.
    pub source_layout: ChannelLayout,
    /// Destination layout.
    pub destination_layout: ChannelLayout,
    /// All routes in this mapping.
    pub routes: Vec<ChannelRoute>,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, String>,
}

impl ChannelMapping {
    /// Create a new channel mapping.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        source_layout: ChannelLayout,
        destination_layout: ChannelLayout,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            source_layout,
            destination_layout,
            routes: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a route.
    pub fn add_route(&mut self, route: ChannelRoute) {
        self.routes.push(route);
    }

    /// Return the number of routes.
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }

    /// Return only active routes.
    pub fn active_routes(&self) -> Vec<&ChannelRoute> {
        self.routes.iter().filter(|r| r.active).collect()
    }

    /// Build the gain matrix as a 2D array \[dst\]\[src\].
    pub fn gain_matrix(&self) -> Vec<Vec<f64>> {
        let src_count = self.source_layout.channel_count();
        let dst_count = self.destination_layout.channel_count();
        let mut matrix = vec![vec![0.0_f64; src_count]; dst_count];

        for route in &self.routes {
            if route.active && route.dst_index < dst_count && route.src_index < src_count {
                matrix[route.dst_index][route.src_index] += route.gain;
            }
        }
        matrix
    }

    /// Validate the mapping for common issues.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();
        let src_count = self.source_layout.channel_count();
        let dst_count = self.destination_layout.channel_count();

        for (i, route) in self.routes.iter().enumerate() {
            if route.src_index >= src_count {
                issues.push(format!(
                    "route {i}: src_index {} out of range (max {})",
                    route.src_index,
                    src_count - 1
                ));
            }
            if route.dst_index >= dst_count {
                issues.push(format!(
                    "route {i}: dst_index {} out of range (max {})",
                    route.dst_index,
                    dst_count - 1
                ));
            }
            if route.gain < 0.0 {
                issues.push(format!("route {i}: negative gain {}", route.gain));
            }
        }

        // Check for unmapped destination channels
        let mapped_dst: std::collections::HashSet<usize> = self
            .routes
            .iter()
            .filter(|r| r.active)
            .map(|r| r.dst_index)
            .collect();
        for i in 0..dst_count {
            if !mapped_dst.contains(&i) {
                issues.push(format!("destination channel {i} has no active route"));
            }
        }

        issues
    }

    /// Attach metadata.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

/// Create a standard stereo-to-5.1 upmix mapping with typical coefficients.
#[allow(clippy::cast_precision_loss)]
pub fn stereo_to_51_upmix(id: impl Into<String>) -> ChannelMapping {
    let src = ChannelLayout::Stereo;
    let dst = ChannelLayout::Surround51;
    let mut mapping = ChannelMapping::new(id, "Stereo to 5.1 Upmix", src, dst);

    // L -> L (unity)
    mapping.add_route(ChannelRoute::new(
        0,
        ChannelLabel::Left,
        0,
        ChannelLabel::Left,
    ));
    // R -> R (unity)
    mapping.add_route(ChannelRoute::new(
        1,
        ChannelLabel::Right,
        1,
        ChannelLabel::Right,
    ));
    // L+R -> Center (-3dB each)
    let center_gain = 0.707; // -3dB
    mapping.add_route(
        ChannelRoute::new(0, ChannelLabel::Left, 2, ChannelLabel::Center).with_gain(center_gain),
    );
    mapping.add_route(
        ChannelRoute::new(1, ChannelLabel::Right, 2, ChannelLabel::Center).with_gain(center_gain),
    );
    // LFE silence (index 3)
    mapping
        .add_route(ChannelRoute::new(0, ChannelLabel::Left, 3, ChannelLabel::Lfe).with_gain(0.0));
    // L -> Ls (attenuated)
    mapping.add_route(
        ChannelRoute::new(0, ChannelLabel::Left, 4, ChannelLabel::LeftSurround).with_gain(0.5),
    );
    // R -> Rs (attenuated)
    mapping.add_route(
        ChannelRoute::new(1, ChannelLabel::Right, 5, ChannelLabel::RightSurround).with_gain(0.5),
    );

    mapping
}

/// Create a 5.1-to-stereo downmix mapping with standard coefficients.
pub fn surround51_to_stereo_downmix(id: impl Into<String>) -> ChannelMapping {
    let src = ChannelLayout::Surround51;
    let dst = ChannelLayout::Stereo;
    let mut mapping = ChannelMapping::new(id, "5.1 to Stereo Downmix", src, dst);

    // L -> L
    mapping.add_route(ChannelRoute::new(
        0,
        ChannelLabel::Left,
        0,
        ChannelLabel::Left,
    ));
    // R -> R
    mapping.add_route(ChannelRoute::new(
        1,
        ChannelLabel::Right,
        1,
        ChannelLabel::Right,
    ));
    // C -> L,R at -3dB
    mapping.add_route(
        ChannelRoute::new(2, ChannelLabel::Center, 0, ChannelLabel::Left).with_gain(0.707),
    );
    mapping.add_route(
        ChannelRoute::new(2, ChannelLabel::Center, 1, ChannelLabel::Right).with_gain(0.707),
    );
    // Ls -> L at -6dB
    mapping.add_route(
        ChannelRoute::new(4, ChannelLabel::LeftSurround, 0, ChannelLabel::Left).with_gain(0.5),
    );
    // Rs -> R at -6dB
    mapping.add_route(
        ChannelRoute::new(5, ChannelLabel::RightSurround, 1, ChannelLabel::Right).with_gain(0.5),
    );

    mapping
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_label_display() {
        assert_eq!(format!("{}", ChannelLabel::Left), "L");
        assert_eq!(format!("{}", ChannelLabel::Right), "R");
        assert_eq!(format!("{}", ChannelLabel::Center), "C");
        assert_eq!(format!("{}", ChannelLabel::Lfe), "LFE");
        assert_eq!(format!("{}", ChannelLabel::Custom(42)), "Ch42");
    }

    #[test]
    fn test_channel_layout_counts() {
        assert_eq!(ChannelLayout::Mono.channel_count(), 1);
        assert_eq!(ChannelLayout::Stereo.channel_count(), 2);
        assert_eq!(ChannelLayout::Surround51.channel_count(), 6);
        assert_eq!(ChannelLayout::Surround71.channel_count(), 8);
    }

    #[test]
    fn test_channel_layout_display() {
        assert_eq!(format!("{}", ChannelLayout::Mono), "Mono");
        assert_eq!(format!("{}", ChannelLayout::Stereo), "Stereo");
        assert_eq!(format!("{}", ChannelLayout::Surround51), "5.1");
        assert_eq!(format!("{}", ChannelLayout::Surround71), "7.1");
    }

    #[test]
    fn test_channel_route_gain_db() {
        let route = ChannelRoute::new(0, ChannelLabel::Left, 0, ChannelLabel::Left);
        // Unity gain = 0 dB
        assert!((route.gain_db() - 0.0).abs() < 0.01);

        let half = ChannelRoute::new(0, ChannelLabel::Left, 0, ChannelLabel::Left).with_gain(0.5);
        assert!((half.gain_db() - (-6.02)).abs() < 0.1);
    }

    #[test]
    fn test_channel_route_zero_gain_db() {
        let route = ChannelRoute::new(0, ChannelLabel::Left, 0, ChannelLabel::Left).with_gain(0.0);
        assert!(route.gain_db().is_infinite());
    }

    #[test]
    fn test_channel_route_disabled() {
        let route = ChannelRoute::new(0, ChannelLabel::Left, 0, ChannelLabel::Left).disabled();
        assert!(!route.active);
    }

    #[test]
    fn test_channel_mapping_basic() {
        let mut mapping =
            ChannelMapping::new("m1", "Test", ChannelLayout::Stereo, ChannelLayout::Stereo);
        mapping.add_route(ChannelRoute::new(
            0,
            ChannelLabel::Left,
            0,
            ChannelLabel::Left,
        ));
        mapping.add_route(ChannelRoute::new(
            1,
            ChannelLabel::Right,
            1,
            ChannelLabel::Right,
        ));
        assert_eq!(mapping.route_count(), 2);
        assert_eq!(mapping.active_routes().len(), 2);
    }

    #[test]
    fn test_gain_matrix() {
        let mut mapping =
            ChannelMapping::new("m1", "Test", ChannelLayout::Stereo, ChannelLayout::Stereo);
        mapping.add_route(
            ChannelRoute::new(0, ChannelLabel::Left, 0, ChannelLabel::Left).with_gain(0.8),
        );
        mapping.add_route(
            ChannelRoute::new(1, ChannelLabel::Right, 1, ChannelLabel::Right).with_gain(0.9),
        );

        let matrix = mapping.gain_matrix();
        assert_eq!(matrix.len(), 2);
        assert!((matrix[0][0] - 0.8).abs() < f64::EPSILON);
        assert!((matrix[1][1] - 0.9).abs() < f64::EPSILON);
        assert!((matrix[0][1] - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_validate_good_mapping() {
        let mut mapping =
            ChannelMapping::new("m1", "Test", ChannelLayout::Stereo, ChannelLayout::Stereo);
        mapping.add_route(ChannelRoute::new(
            0,
            ChannelLabel::Left,
            0,
            ChannelLabel::Left,
        ));
        mapping.add_route(ChannelRoute::new(
            1,
            ChannelLabel::Right,
            1,
            ChannelLabel::Right,
        ));
        let issues = mapping.validate();
        assert!(issues.is_empty(), "Expected no issues but got: {issues:?}");
    }

    #[test]
    fn test_validate_out_of_range() {
        let mut mapping =
            ChannelMapping::new("m1", "Test", ChannelLayout::Mono, ChannelLayout::Mono);
        mapping.add_route(ChannelRoute::new(
            5,
            ChannelLabel::Left,
            0,
            ChannelLabel::Mono,
        ));
        let issues = mapping.validate();
        assert!(issues.iter().any(|i| i.contains("src_index")));
    }

    #[test]
    fn test_validate_unmapped_destination() {
        let mapping =
            ChannelMapping::new("m1", "Test", ChannelLayout::Stereo, ChannelLayout::Stereo);
        // No routes — both destinations unmapped
        let issues = mapping.validate();
        assert_eq!(issues.len(), 2);
    }

    #[test]
    fn test_stereo_to_51_upmix() {
        let mapping = stereo_to_51_upmix("upmix1");
        assert_eq!(mapping.source_layout.channel_count(), 2);
        assert_eq!(mapping.destination_layout.channel_count(), 6);
        assert!(mapping.route_count() >= 6);
        let issues = mapping.validate();
        assert!(issues.is_empty(), "Upmix validation failed: {issues:?}");
    }

    #[test]
    fn test_51_to_stereo_downmix() {
        let mapping = surround51_to_stereo_downmix("downmix1");
        assert_eq!(mapping.source_layout.channel_count(), 6);
        assert_eq!(mapping.destination_layout.channel_count(), 2);
        assert!(mapping.route_count() >= 4);
        let issues = mapping.validate();
        assert!(issues.is_empty(), "Downmix validation failed: {issues:?}");
    }

    #[test]
    fn test_metadata() {
        let mut mapping =
            ChannelMapping::new("m1", "Test", ChannelLayout::Stereo, ChannelLayout::Stereo);
        mapping.set_metadata("standard", "ITU-R BS.775");
        assert_eq!(
            mapping
                .metadata
                .get("standard")
                .expect("failed to get value"),
            "ITU-R BS.775"
        );
    }
}
