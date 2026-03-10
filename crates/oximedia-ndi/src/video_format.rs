//! NDI video format negotiation and color space handling.
#![allow(dead_code)]

/// Color space used in an NDI video stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NdiColorSpace {
    /// Standard dynamic range BT.709.
    Bt709,
    /// Standard dynamic range BT.601.
    Bt601,
    /// High dynamic range PQ (SMPTE ST 2084).
    Bt2100Pq,
    /// High dynamic range HLG (ITU-R BT.2100).
    Bt2100Hlg,
    /// sRGB color space for display-referred content.
    Srgb,
}

impl NdiColorSpace {
    /// Returns `true` if this color space is an HDR variant.
    pub fn is_hdr(self) -> bool {
        matches!(self, NdiColorSpace::Bt2100Pq | NdiColorSpace::Bt2100Hlg)
    }

    /// Returns a human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            NdiColorSpace::Bt709 => "BT.709",
            NdiColorSpace::Bt601 => "BT.601",
            NdiColorSpace::Bt2100Pq => "BT.2100 PQ",
            NdiColorSpace::Bt2100Hlg => "BT.2100 HLG",
            NdiColorSpace::Srgb => "sRGB",
        }
    }
}

/// Pixel format / chroma subsampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 4:2:0 planar (YUV420P).
    Yuv420,
    /// 4:2:2 interleaved (UYVY).
    Yuv422,
    /// 4:4:4 planar.
    Yuv444,
    /// BGRA packed (32-bit).
    Bgra,
    /// RGBA packed (32-bit).
    Rgba,
}

/// A complete NDI video format descriptor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NdiVideoFormat {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame-rate numerator.
    pub fps_num: u32,
    /// Frame-rate denominator.
    pub fps_den: u32,
    /// Pixel / chroma format.
    pub pixel_format: PixelFormat,
    /// Color space and transfer function.
    pub color_space: NdiColorSpace,
    /// Progressive (`true`) or interlaced (`false`).
    pub progressive: bool,
}

impl NdiVideoFormat {
    /// Create a new `NdiVideoFormat`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        width: u32,
        height: u32,
        fps_num: u32,
        fps_den: u32,
        pixel_format: PixelFormat,
        color_space: NdiColorSpace,
        progressive: bool,
    ) -> Self {
        Self {
            width,
            height,
            fps_num,
            fps_den,
            pixel_format,
            color_space,
            progressive,
        }
    }

    /// Create a standard 1080p30 SDR format.
    pub fn hd_1080p30() -> Self {
        Self::new(
            1920,
            1080,
            30,
            1,
            PixelFormat::Yuv422,
            NdiColorSpace::Bt709,
            true,
        )
    }

    /// Create a standard 1080p60 SDR format.
    pub fn hd_1080p60() -> Self {
        Self::new(
            1920,
            1080,
            60,
            1,
            PixelFormat::Yuv422,
            NdiColorSpace::Bt709,
            true,
        )
    }

    /// Create a 4K UHD 30p SDR format.
    pub fn uhd_4k_30p() -> Self {
        Self::new(
            3840,
            2160,
            30,
            1,
            PixelFormat::Yuv422,
            NdiColorSpace::Bt709,
            true,
        )
    }

    /// Create a 4K UHD 30p HDR (PQ) format.
    pub fn uhd_4k_30p_hdr() -> Self {
        Self::new(
            3840,
            2160,
            30,
            1,
            PixelFormat::Yuv422,
            NdiColorSpace::Bt2100Pq,
            true,
        )
    }

    /// Pixel rate in pixels per second.
    #[allow(clippy::cast_precision_loss)]
    pub fn pixel_rate(&self) -> f64 {
        let total_pixels = self.width as f64 * self.height as f64;
        let fps = self.fps_num as f64 / self.fps_den as f64;
        total_pixels * fps
    }

    /// Returns `true` when width >= 3840 (UHD 4K or larger).
    pub fn is_4k(&self) -> bool {
        self.width >= 3840
    }

    /// Frame rate as a floating-point number.
    #[allow(clippy::cast_precision_loss)]
    pub fn frame_rate(&self) -> f64 {
        self.fps_num as f64 / self.fps_den as f64
    }

    /// Returns `true` if this format uses an HDR color space.
    pub fn is_hdr(&self) -> bool {
        self.color_space.is_hdr()
    }

    /// Bytes per pixel for the chosen pixel format (exact or average for subsampled).
    #[allow(clippy::cast_precision_loss)]
    pub fn bytes_per_pixel(&self) -> f64 {
        match self.pixel_format {
            PixelFormat::Yuv420 => 1.5,
            PixelFormat::Yuv422 => 2.0,
            PixelFormat::Yuv444 => 3.0,
            PixelFormat::Bgra | PixelFormat::Rgba => 4.0,
        }
    }

    /// Approximate uncompressed frame size in bytes.
    #[allow(clippy::cast_precision_loss)]
    pub fn frame_size_bytes(&self) -> u64 {
        let pixels = self.width as f64 * self.height as f64;
        (pixels * self.bytes_per_pixel()) as u64
    }
}

/// Negotiates a mutually compatible NDI video format between sender and receiver.
#[derive(Debug, Default)]
pub struct NdiFormatNegotiator {
    preferred_color_spaces: Vec<NdiColorSpace>,
    preferred_pixel_formats: Vec<PixelFormat>,
}

impl NdiFormatNegotiator {
    /// Create a new negotiator with empty preference lists.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a preferred color space (highest priority first).
    pub fn prefer_color_space(mut self, cs: NdiColorSpace) -> Self {
        self.preferred_color_spaces.push(cs);
        self
    }

    /// Add a preferred pixel format (highest priority first).
    pub fn prefer_pixel_format(mut self, pf: PixelFormat) -> Self {
        self.preferred_pixel_formats.push(pf);
        self
    }

    /// Returns `true` if `a` and `b` are compatible (same dimensions and frame rate).
    pub fn compatible(a: &NdiVideoFormat, b: &NdiVideoFormat) -> bool {
        a.width == b.width
            && a.height == b.height
            && a.fps_num == b.fps_num
            && a.fps_den == b.fps_den
    }

    /// Select the best format from `candidates` based on this negotiator's preferences.
    ///
    /// Returns `None` if `candidates` is empty.
    pub fn select_best<'a>(&self, candidates: &'a [NdiVideoFormat]) -> Option<&'a NdiVideoFormat> {
        // Score each candidate; higher = better.
        candidates.iter().max_by_key(|fmt| {
            let cs_score = self
                .preferred_color_spaces
                .iter()
                .position(|&cs| cs == fmt.color_space)
                .map_or(0usize, |pos| self.preferred_color_spaces.len() - pos);
            let pf_score = self
                .preferred_pixel_formats
                .iter()
                .position(|&pf| pf == fmt.pixel_format)
                .map_or(0usize, |pos| self.preferred_pixel_formats.len() - pos);
            cs_score * 1000 + pf_score
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_space_is_hdr_false() {
        assert!(!NdiColorSpace::Bt709.is_hdr());
        assert!(!NdiColorSpace::Bt601.is_hdr());
        assert!(!NdiColorSpace::Srgb.is_hdr());
    }

    #[test]
    fn test_color_space_is_hdr_true() {
        assert!(NdiColorSpace::Bt2100Pq.is_hdr());
        assert!(NdiColorSpace::Bt2100Hlg.is_hdr());
    }

    #[test]
    fn test_color_space_label() {
        assert_eq!(NdiColorSpace::Bt709.label(), "BT.709");
        assert_eq!(NdiColorSpace::Bt2100Pq.label(), "BT.2100 PQ");
    }

    #[test]
    fn test_hd_1080p30_dimensions() {
        let fmt = NdiVideoFormat::hd_1080p30();
        assert_eq!(fmt.width, 1920);
        assert_eq!(fmt.height, 1080);
    }

    #[test]
    fn test_hd_1080p30_frame_rate() {
        let fmt = NdiVideoFormat::hd_1080p30();
        assert!((fmt.frame_rate() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pixel_rate_hd() {
        let fmt = NdiVideoFormat::hd_1080p60();
        let expected = 1920.0 * 1080.0 * 60.0;
        assert!((fmt.pixel_rate() - expected).abs() < 1.0);
    }

    #[test]
    fn test_is_4k_false() {
        assert!(!NdiVideoFormat::hd_1080p30().is_4k());
    }

    #[test]
    fn test_is_4k_true() {
        assert!(NdiVideoFormat::uhd_4k_30p().is_4k());
    }

    #[test]
    fn test_is_hdr_true() {
        assert!(NdiVideoFormat::uhd_4k_30p_hdr().is_hdr());
    }

    #[test]
    fn test_is_hdr_false() {
        assert!(!NdiVideoFormat::hd_1080p30().is_hdr());
    }

    #[test]
    fn test_frame_size_yuv422() {
        let fmt = NdiVideoFormat::hd_1080p30();
        let expected = (1920 * 1080 * 2) as u64;
        assert_eq!(fmt.frame_size_bytes(), expected);
    }

    #[test]
    fn test_compatible_same_format() {
        let a = NdiVideoFormat::hd_1080p30();
        let b = NdiVideoFormat::hd_1080p30();
        assert!(NdiFormatNegotiator::compatible(&a, &b));
    }

    #[test]
    fn test_compatible_different_res() {
        let a = NdiVideoFormat::hd_1080p30();
        let b = NdiVideoFormat::uhd_4k_30p();
        assert!(!NdiFormatNegotiator::compatible(&a, &b));
    }

    #[test]
    fn test_select_best_empty() {
        let neg = NdiFormatNegotiator::new();
        assert!(neg.select_best(&[]).is_none());
    }

    #[test]
    fn test_select_best_prefers_hdr() {
        let sdr = NdiVideoFormat::uhd_4k_30p();
        let hdr = NdiVideoFormat::uhd_4k_30p_hdr();
        let candidates = [sdr, hdr];
        let neg = NdiFormatNegotiator::new()
            .prefer_color_space(NdiColorSpace::Bt2100Pq)
            .prefer_color_space(NdiColorSpace::Bt709);
        let best = neg
            .select_best(&candidates)
            .expect("expected best selection");
        assert_eq!(best.color_space, NdiColorSpace::Bt2100Pq);
    }

    #[test]
    fn test_select_best_single() {
        let fmt = NdiVideoFormat::hd_1080p60();
        let neg = NdiFormatNegotiator::new();
        assert_eq!(
            neg.select_best(&[fmt])
                .expect("expected best selection")
                .width,
            1920
        );
    }
}
