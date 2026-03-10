#![allow(dead_code)]
//! Alpha mask and layer compositing for broadcast graphics.
//!
//! Provides mask-based compositing operations including rectangular masks,
//! elliptical masks, gradient masks, and feathered edges for smooth transitions.

/// Mask blending operation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaskOp {
    /// Intersect two masks (min).
    Intersect,
    /// Union two masks (max).
    Union,
    /// Subtract second from first.
    Subtract,
    /// Exclusive-or of two masks.
    Xor,
}

/// A single-channel alpha mask stored as row-major `f64` values in `[0.0, 1.0]`.
#[derive(Clone, Debug)]
pub struct AlphaMask {
    /// Width of the mask in pixels.
    width: u32,
    /// Height of the mask in pixels.
    height: u32,
    /// Row-major alpha data.
    data: Vec<f64>,
}

impl AlphaMask {
    /// Create a fully opaque mask.
    pub fn opaque(width: u32, height: u32) -> Self {
        let len = (width as usize) * (height as usize);
        Self {
            width,
            height,
            data: vec![1.0; len],
        }
    }

    /// Create a fully transparent mask.
    pub fn transparent(width: u32, height: u32) -> Self {
        let len = (width as usize) * (height as usize);
        Self {
            width,
            height,
            data: vec![0.0; len],
        }
    }

    /// Create a rectangular mask with optional feather radius.
    #[allow(clippy::cast_precision_loss)]
    pub fn rectangle(
        width: u32,
        height: u32,
        x: u32,
        y: u32,
        rect_w: u32,
        rect_h: u32,
        feather: f64,
    ) -> Self {
        let len = (width as usize) * (height as usize);
        let mut data = vec![0.0_f64; len];
        let x1 = x as f64;
        let y1 = y as f64;
        let x2 = (x + rect_w) as f64;
        let y2 = (y + rect_h) as f64;

        for py in 0..height {
            for px in 0..width {
                let fx = px as f64;
                let fy = py as f64;
                let dx = if fx < x1 {
                    x1 - fx
                } else if fx > x2 {
                    fx - x2
                } else {
                    0.0
                };
                let dy = if fy < y1 {
                    y1 - fy
                } else if fy > y2 {
                    fy - y2
                } else {
                    0.0
                };
                let dist = (dx * dx + dy * dy).sqrt();
                let alpha = if feather > 0.0 {
                    (1.0 - dist / feather).clamp(0.0, 1.0)
                } else if dist <= 0.0 {
                    1.0
                } else {
                    0.0
                };
                data[py as usize * width as usize + px as usize] = alpha;
            }
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Create an elliptical mask with optional feather.
    #[allow(clippy::cast_precision_loss)]
    pub fn ellipse(
        width: u32,
        height: u32,
        cx: f64,
        cy: f64,
        rx: f64,
        ry: f64,
        feather: f64,
    ) -> Self {
        let len = (width as usize) * (height as usize);
        let mut data = vec![0.0_f64; len];

        for py in 0..height {
            for px in 0..width {
                let fx = px as f64 - cx;
                let fy = py as f64 - cy;
                let norm = if rx > 0.0 && ry > 0.0 {
                    ((fx / rx).powi(2) + (fy / ry).powi(2)).sqrt()
                } else {
                    f64::MAX
                };
                let alpha = if feather > 0.0 {
                    let edge = 1.0;
                    let outer = edge + feather / rx.min(ry).max(1.0);
                    if norm <= edge {
                        1.0
                    } else if norm >= outer {
                        0.0
                    } else {
                        1.0 - (norm - edge) / (outer - edge)
                    }
                } else if norm <= 1.0 {
                    1.0
                } else {
                    0.0
                };
                data[py as usize * width as usize + px as usize] = alpha;
            }
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Create a vertical gradient mask (top to bottom, 0.0 to 1.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn vertical_gradient(width: u32, height: u32) -> Self {
        let len = (width as usize) * (height as usize);
        let mut data = vec![0.0_f64; len];
        let h = if height > 1 { (height - 1) as f64 } else { 1.0 };
        for py in 0..height {
            let alpha = py as f64 / h;
            for px in 0..width {
                data[py as usize * width as usize + px as usize] = alpha;
            }
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Create a horizontal gradient mask (left to right, 0.0 to 1.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn horizontal_gradient(width: u32, height: u32) -> Self {
        let len = (width as usize) * (height as usize);
        let mut data = vec![0.0_f64; len];
        let w = if width > 1 { (width - 1) as f64 } else { 1.0 };
        for py in 0..height {
            for px in 0..width {
                let alpha = px as f64 / w;
                data[py as usize * width as usize + px as usize] = alpha;
            }
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Create a radial gradient mask from center.
    #[allow(clippy::cast_precision_loss)]
    pub fn radial_gradient(width: u32, height: u32, cx: f64, cy: f64, radius: f64) -> Self {
        let len = (width as usize) * (height as usize);
        let mut data = vec![0.0_f64; len];
        let r = radius.max(1.0);
        for py in 0..height {
            for px in 0..width {
                let dx = px as f64 - cx;
                let dy = py as f64 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let alpha = (1.0 - dist / r).clamp(0.0, 1.0);
                data[py as usize * width as usize + px as usize] = alpha;
            }
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Get the mask width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the mask height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get the alpha value at a specific pixel.
    pub fn get(&self, x: u32, y: u32) -> f64 {
        if x < self.width && y < self.height {
            self.data[y as usize * self.width as usize + x as usize]
        } else {
            0.0
        }
    }

    /// Set the alpha value at a specific pixel.
    pub fn set(&mut self, x: u32, y: u32, alpha: f64) {
        if x < self.width && y < self.height {
            self.data[y as usize * self.width as usize + x as usize] = alpha.clamp(0.0, 1.0);
        }
    }

    /// Invert the mask.
    pub fn invert(&mut self) {
        for v in &mut self.data {
            *v = 1.0 - *v;
        }
    }

    /// Return an inverted copy of the mask.
    pub fn inverted(&self) -> Self {
        let data = self.data.iter().map(|v| 1.0 - v).collect();
        Self {
            width: self.width,
            height: self.height,
            data,
        }
    }

    /// Combine with another mask using the specified operation.
    ///
    /// Both masks must have the same dimensions.
    pub fn combine(&self, other: &AlphaMask, op: MaskOp) -> Option<Self> {
        if self.width != other.width || self.height != other.height {
            return None;
        }
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| match op {
                MaskOp::Intersect => a.min(b),
                MaskOp::Union => a.max(b),
                MaskOp::Subtract => (a - b).max(0.0),
                MaskOp::Xor => (a + b - 2.0 * a * b).clamp(0.0, 1.0),
            })
            .collect();
        Some(Self {
            width: self.width,
            height: self.height,
            data,
        })
    }

    /// Apply a threshold to the mask, converting it to a hard edge.
    pub fn threshold(&mut self, value: f64) {
        for v in &mut self.data {
            *v = if *v >= value { 1.0 } else { 0.0 };
        }
    }

    /// Calculate the average alpha of the mask.
    pub fn average_alpha(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.data.iter().sum();
        sum / self.data.len() as f64
    }

    /// Get the raw data as a slice.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Apply a gaussian-like box blur pass to soften the mask.
    #[allow(clippy::cast_precision_loss)]
    pub fn blur(&mut self, radius: u32) {
        if radius == 0 {
            return;
        }
        let w = self.width as usize;
        let h = self.height as usize;
        let r = radius as usize;

        // Horizontal pass
        let mut tmp = vec![0.0_f64; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0;
                let mut count = 0u32;
                let start = x.saturating_sub(r);
                let end = (x + r + 1).min(w);
                for sx in start..end {
                    sum += self.data[y * w + sx];
                    count += 1;
                }
                tmp[y * w + x] = sum / f64::from(count);
            }
        }
        // Vertical pass
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0;
                let mut count = 0u32;
                let start = y.saturating_sub(r);
                let end = (y + r + 1).min(h);
                for sy in start..end {
                    sum += tmp[sy * w + x];
                    count += 1;
                }
                self.data[y * w + x] = sum / f64::from(count);
            }
        }
    }
}

/// A compositing layer that pairs an alpha mask with an opacity.
#[derive(Clone, Debug)]
pub struct MaskLayer {
    /// Layer name.
    name: String,
    /// The alpha mask.
    mask: AlphaMask,
    /// Global opacity multiplier.
    opacity: f64,
    /// Whether this layer is visible.
    visible: bool,
}

impl MaskLayer {
    /// Create a new mask layer.
    pub fn new(name: &str, mask: AlphaMask, opacity: f64) -> Self {
        Self {
            name: name.to_string(),
            mask,
            opacity: opacity.clamp(0.0, 1.0),
            visible: true,
        }
    }

    /// Get the layer name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the effective alpha at a pixel (mask * opacity).
    pub fn effective_alpha(&self, x: u32, y: u32) -> f64 {
        if !self.visible {
            return 0.0;
        }
        self.mask.get(x, y) * self.opacity
    }

    /// Set the layer opacity.
    pub fn set_opacity(&mut self, opacity: f64) {
        self.opacity = opacity.clamp(0.0, 1.0);
    }

    /// Get the layer opacity.
    pub fn opacity(&self) -> f64 {
        self.opacity
    }

    /// Toggle visibility.
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Check if visible.
    pub fn is_visible(&self) -> bool {
        self.visible
    }

    /// Get a reference to the mask.
    pub fn mask(&self) -> &AlphaMask {
        &self.mask
    }

    /// Get a mutable reference to the mask.
    pub fn mask_mut(&mut self) -> &mut AlphaMask {
        &mut self.mask
    }
}

/// Layer compositing stack.
#[derive(Clone, Debug, Default)]
pub struct LayerStack {
    /// Layers from bottom to top.
    layers: Vec<MaskLayer>,
}

impl LayerStack {
    /// Create an empty layer stack.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Push a layer on top.
    pub fn push(&mut self, layer: MaskLayer) {
        self.layers.push(layer);
    }

    /// Remove a layer by index.
    pub fn remove(&mut self, index: usize) -> Option<MaskLayer> {
        if index < self.layers.len() {
            Some(self.layers.remove(index))
        } else {
            None
        }
    }

    /// Get the number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if the stack is empty.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Flatten all visible layers into a single alpha mask using "over" compositing.
    pub fn flatten(&self, width: u32, height: u32) -> AlphaMask {
        let mut result = AlphaMask::transparent(width, height);
        for layer in &self.layers {
            if !layer.is_visible() {
                continue;
            }
            for y in 0..height {
                for x in 0..width {
                    let src = layer.effective_alpha(x, y);
                    let dst = result.get(x, y);
                    // Standard "over" compositing for alpha
                    let out = src + dst * (1.0 - src);
                    result.set(x, y, out);
                }
            }
        }
        result
    }

    /// Get a layer by index.
    pub fn get(&self, index: usize) -> Option<&MaskLayer> {
        self.layers.get(index)
    }

    /// Find a layer by name.
    pub fn find_by_name(&self, name: &str) -> Option<&MaskLayer> {
        self.layers.iter().find(|l| l.name() == name)
    }
}

/// Compute a smooth-step interpolation for feathering.
#[allow(clippy::cast_precision_loss)]
fn _smooth_step(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opaque_mask() {
        let m = AlphaMask::opaque(10, 10);
        assert_eq!(m.width(), 10);
        assert_eq!(m.height(), 10);
        assert!((m.get(0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((m.get(9, 9) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_transparent_mask() {
        let m = AlphaMask::transparent(8, 8);
        assert!((m.get(4, 4)).abs() < f64::EPSILON);
        assert!((m.average_alpha()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rectangle_mask_hard() {
        let m = AlphaMask::rectangle(20, 20, 5, 5, 10, 10, 0.0);
        // Inside
        assert!((m.get(10, 10) - 1.0).abs() < f64::EPSILON);
        // Outside
        assert!((m.get(0, 0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ellipse_mask() {
        let m = AlphaMask::ellipse(100, 100, 50.0, 50.0, 30.0, 20.0, 0.0);
        // Center should be opaque
        assert!((m.get(50, 50) - 1.0).abs() < f64::EPSILON);
        // Far corner should be transparent
        assert!((m.get(0, 0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vertical_gradient() {
        let m = AlphaMask::vertical_gradient(10, 11);
        assert!((m.get(0, 0)).abs() < f64::EPSILON);
        assert!((m.get(0, 10) - 1.0).abs() < f64::EPSILON);
        assert!((m.get(5, 5) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_horizontal_gradient() {
        let m = AlphaMask::horizontal_gradient(11, 10);
        assert!((m.get(0, 5)).abs() < f64::EPSILON);
        assert!((m.get(10, 5) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_radial_gradient() {
        let m = AlphaMask::radial_gradient(20, 20, 10.0, 10.0, 10.0);
        // Center should be 1.0
        assert!((m.get(10, 10) - 1.0).abs() < f64::EPSILON);
        // Edge should be near 0
        assert!(m.get(0, 0) < 0.1);
    }

    #[test]
    fn test_invert() {
        let mut m = AlphaMask::opaque(4, 4);
        m.invert();
        assert!((m.get(0, 0)).abs() < f64::EPSILON);
        let inv = m.inverted();
        assert!((inv.get(0, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_combine_intersect() {
        let a = AlphaMask::opaque(4, 4);
        let b = AlphaMask::transparent(4, 4);
        let c = a.combine(&b, MaskOp::Intersect).expect("c should be valid");
        assert!((c.get(0, 0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_combine_union() {
        let a = AlphaMask::opaque(4, 4);
        let b = AlphaMask::transparent(4, 4);
        let c = a.combine(&b, MaskOp::Union).expect("c should be valid");
        assert!((c.get(0, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_combine_dimension_mismatch() {
        let a = AlphaMask::opaque(4, 4);
        let b = AlphaMask::opaque(5, 4);
        assert!(a.combine(&b, MaskOp::Union).is_none());
    }

    #[test]
    fn test_threshold() {
        let mut m = AlphaMask::vertical_gradient(1, 11);
        m.threshold(0.5);
        assert!((m.get(0, 0)).abs() < f64::EPSILON);
        assert!((m.get(0, 10) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mask_layer_effective_alpha() {
        let mask = AlphaMask::opaque(4, 4);
        let layer = MaskLayer::new("fg", mask, 0.5);
        assert!((layer.effective_alpha(0, 0) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mask_layer_visibility() {
        let mask = AlphaMask::opaque(4, 4);
        let mut layer = MaskLayer::new("fg", mask, 1.0);
        layer.set_visible(false);
        assert!((layer.effective_alpha(0, 0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_layer_stack_flatten() {
        let mut stack = LayerStack::new();
        let m1 = AlphaMask::opaque(4, 4);
        stack.push(MaskLayer::new("base", m1, 0.5));
        let flat = stack.flatten(4, 4);
        assert!((flat.get(0, 0) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_layer_stack_find_by_name() {
        let mut stack = LayerStack::new();
        stack.push(MaskLayer::new("a", AlphaMask::opaque(4, 4), 1.0));
        stack.push(MaskLayer::new("b", AlphaMask::opaque(4, 4), 0.5));
        assert!(stack.find_by_name("b").is_some());
        assert!(stack.find_by_name("c").is_none());
        assert_eq!(stack.len(), 2);
        assert!(!stack.is_empty());
    }

    #[test]
    fn test_blur_mask() {
        let mut m = AlphaMask::opaque(10, 10);
        m.set(0, 0, 0.0);
        m.blur(1);
        // After blur the corner value should be non-zero (pulled from neighbors)
        assert!(m.get(0, 0) > 0.0);
    }

    #[test]
    fn test_set_and_get() {
        let mut m = AlphaMask::transparent(5, 5);
        m.set(2, 3, 0.75);
        assert!((m.get(2, 3) - 0.75).abs() < f64::EPSILON);
        // Out of bounds get returns 0
        assert!((m.get(10, 10)).abs() < f64::EPSILON);
    }
}
