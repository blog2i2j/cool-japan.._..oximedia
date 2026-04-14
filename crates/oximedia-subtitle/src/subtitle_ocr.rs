//! OCR-like text extraction from bitmap-based subtitle formats.
//!
//! This module provides pattern-matching based text recognition for
//! bitmap-based subtitle formats such as PGS (Blu-ray Presentation Graphic
//! Stream) and VobSub (DVD).  It does **not** use neural networks; instead it
//! uses a template-matching approach where each glyph is compared to a built-in
//! or user-supplied glyph atlas via normalised cross-correlation (NCC).
//!
//! # Design
//!
//! 1. A [`GlyphTemplate`] holds a small binary bitmap for a single character.
//! 2. A [`GlyphAtlas`] is a collection of templates, one per character.
//! 3. [`BitmapOcr`] scans a source bitmap and extracts connected components
//!    (blobs); each blob is resized to a canonical height and matched against
//!    the atlas using NCC.
//! 4. Blobs are sorted left-to-right and assembled into a string.
//!
//! # Limitations
//!
//! - Works best with monospace or near-monospace bitmap subtitle fonts.
//! - Very small (<4 px) or very large (>200 px) blobs are ignored.
//! - Ligatures and diacritics on separate bitmap layers are not handled.
//!
//! # Example
//!
//! ```rust
//! use oximedia_subtitle::subtitle_ocr::{BitmapOcr, GlyphAtlas, BitmapImage};
//!
//! // Build a minimal atlas with an 'A' glyph (5×7 binary bitmap)
//! let mut atlas = GlyphAtlas::new();
//! atlas.add_ascii_template('A', &[
//!     0,1,1,1,0,
//!     1,0,0,0,1,
//!     1,0,0,0,1,
//!     1,1,1,1,1,
//!     1,0,0,0,1,
//!     1,0,0,0,1,
//!     1,0,0,0,1,
//! ], 5, 7);
//!
//! let ocr = BitmapOcr::new(atlas);
//!
//! // Create a synthetic test bitmap (15×7, white 'A' on black)
//! let width = 15usize;
//! let height = 7usize;
//! let mut pixels = vec![0u8; width * height];
//! // Place the 'A' glyph at column 5
//! let glyph: &[u8] = &[
//!     0,1,1,1,0,
//!     1,0,0,0,1,
//!     1,0,0,0,1,
//!     1,1,1,1,1,
//!     1,0,0,0,1,
//!     1,0,0,0,1,
//!     1,0,0,0,1,
//! ];
//! for row in 0..7usize {
//!     for col in 0..5usize {
//!         pixels[row * width + col + 5] = glyph[row * 5 + col] * 255;
//!     }
//! }
//! let img = BitmapImage { pixels, width, height };
//! let text = ocr.extract_text(&img, 0.6);
//! assert!(text.contains('A'), "expected 'A' in: {text:?}");
//! ```

// ── Types ─────────────────────────────────────────────────────────────────────

/// A grayscale or binary bitmap.
#[derive(Clone, Debug)]
pub struct BitmapImage {
    /// Pixel data, row-major, one byte per pixel (0 = black, 255 = white).
    pub pixels: Vec<u8>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

impl BitmapImage {
    /// Create a new blank (all-black) image.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            pixels: vec![0u8; width * height],
            width,
            height,
        }
    }

    /// Return the pixel value at `(col, row)`, or 0 if out of bounds.
    #[must_use]
    pub fn pixel(&self, col: usize, row: usize) -> u8 {
        if col < self.width && row < self.height {
            self.pixels[row * self.width + col]
        } else {
            0
        }
    }

    /// Set a pixel value.
    pub fn set_pixel(&mut self, col: usize, row: usize, value: u8) {
        if col < self.width && row < self.height {
            self.pixels[row * self.width + col] = value;
        }
    }

    /// Binarise: pixels >= threshold become 255, others 0.
    #[must_use]
    pub fn binarise(&self, threshold: u8) -> Self {
        let pixels = self
            .pixels
            .iter()
            .map(|&p| if p >= threshold { 255 } else { 0 })
            .collect();
        Self {
            pixels,
            width: self.width,
            height: self.height,
        }
    }

    /// Nearest-neighbour resize to `new_width` × `new_height`.
    #[must_use]
    pub fn resize(&self, new_width: usize, new_height: usize) -> Self {
        if new_width == 0 || new_height == 0 {
            return Self::new(new_width, new_height);
        }
        let mut out = Self::new(new_width, new_height);
        for row in 0..new_height {
            let src_row = (row * self.height) / new_height;
            for col in 0..new_width {
                let src_col = (col * self.width) / new_width;
                out.set_pixel(col, row, self.pixel(src_col, src_row));
            }
        }
        out
    }

    /// Extract a rectangular sub-image.
    #[must_use]
    pub fn crop(&self, x: usize, y: usize, w: usize, h: usize) -> Self {
        let mut out = Self::new(w, h);
        for row in 0..h {
            for col in 0..w {
                out.set_pixel(col, row, self.pixel(x + col, y + row));
            }
        }
        out
    }
}

// ── Glyph template ────────────────────────────────────────────────────────────

/// A binary glyph template used for pattern matching.
#[derive(Clone, Debug)]
pub struct GlyphTemplate {
    /// The character this template represents.
    pub character: char,
    /// Binary pixels (0 = background, 1 = foreground), row-major.
    pub pixels: Vec<f32>,
    /// Template width.
    pub width: usize,
    /// Template height.
    pub height: usize,
}

impl GlyphTemplate {
    /// Create a glyph template from a binary slice (0/1 values).
    #[must_use]
    pub fn new(character: char, binary: &[u8], width: usize, height: usize) -> Self {
        let pixels = binary
            .iter()
            .map(|&b| if b != 0 { 1.0_f32 } else { 0.0_f32 })
            .collect();
        Self {
            character,
            pixels,
            width,
            height,
        }
    }

    /// Compute the normalised cross-correlation between this template and a
    /// same-sized patch (as f32 pixels, 0.0–1.0).
    ///
    /// Returns a value in [-1, 1]; higher is a better match.
    #[must_use]
    pub fn ncc_score(&self, patch: &[f32]) -> f32 {
        if patch.len() != self.pixels.len() || self.pixels.is_empty() {
            return 0.0;
        }
        let n = self.pixels.len() as f32;
        let mean_t = self.pixels.iter().sum::<f32>() / n;
        let mean_p = patch.iter().sum::<f32>() / n;

        let mut num = 0.0_f32;
        let mut den_t = 0.0_f32;
        let mut den_p = 0.0_f32;

        for (&t, &p) in self.pixels.iter().zip(patch.iter()) {
            let dt = t - mean_t;
            let dp = p - mean_p;
            num += dt * dp;
            den_t += dt * dt;
            den_p += dp * dp;
        }

        let denom = (den_t * den_p).sqrt();
        if denom < 1e-6 {
            0.0
        } else {
            num / denom
        }
    }
}

// ── Glyph atlas ───────────────────────────────────────────────────────────────

/// A collection of glyph templates indexed by character.
#[derive(Clone, Debug, Default)]
pub struct GlyphAtlas {
    templates: Vec<GlyphTemplate>,
}

impl GlyphAtlas {
    /// Create an empty atlas.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a glyph template.
    pub fn add(&mut self, template: GlyphTemplate) {
        self.templates.push(template);
    }

    /// Convenience helper: add an ASCII character template from a binary slice.
    pub fn add_ascii_template(
        &mut self,
        ch: char,
        binary: &[u8],
        width: usize,
        height: usize,
    ) {
        self.add(GlyphTemplate::new(ch, binary, width, height));
    }

    /// Number of templates in the atlas.
    #[must_use]
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Whether the atlas is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }

    /// Find the best-matching character for a normalised patch image.
    ///
    /// The patch is resized to match each template's dimensions before scoring.
    /// Returns `None` if the atlas is empty or no template scores above
    /// `min_score`.
    #[must_use]
    pub fn best_match(&self, patch: &BitmapImage, min_score: f32) -> Option<char> {
        if self.templates.is_empty() {
            return None;
        }

        let mut best_char = None;
        let mut best_score = min_score;

        for tmpl in &self.templates {
            // Resize patch to template dimensions
            let resized = patch.resize(tmpl.width, tmpl.height);
            // Normalise to [0, 1]
            let patch_f32: Vec<f32> = resized.pixels.iter().map(|&p| p as f32 / 255.0).collect();
            let score = tmpl.ncc_score(&patch_f32);
            if score > best_score {
                best_score = score;
                best_char = Some(tmpl.character);
            }
        }

        best_char
    }
}

// ── Connected component labelling ────────────────────────────────────────────

/// An axis-aligned bounding box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bbox {
    /// Leftmost column.
    pub x: usize,
    /// Topmost row.
    pub y: usize,
    /// Width.
    pub w: usize,
    /// Height.
    pub h: usize,
}

impl Bbox {
    /// Horizontal centre.
    #[must_use]
    pub fn cx(&self) -> usize {
        self.x + self.w / 2
    }
}

/// Find connected components (blobs) in a binary image using a simple
/// two-pass union-find algorithm.
///
/// Returns a list of bounding boxes, one per component.
#[must_use]
pub fn find_blobs(binary: &BitmapImage, min_area: usize, max_area: usize) -> Vec<Bbox> {
    if binary.width == 0 || binary.height == 0 {
        return Vec::new();
    }

    let w = binary.width;
    let h = binary.height;
    let n = w * h;
    let mut labels = vec![0usize; n];
    let mut parent = vec![0usize; n + 1];
    // Initialise union-find
    for i in 0..=n {
        parent[i] = i;
    }
    let mut next_label = 1usize;

    // First pass: assign provisional labels
    for row in 0..h {
        for col in 0..w {
            if binary.pixel(col, row) == 0 {
                continue;
            }
            let idx = row * w + col;
            let mut neighbours = Vec::with_capacity(4);
            if col > 0 && binary.pixel(col - 1, row) != 0 {
                neighbours.push(labels[idx - 1]);
            }
            if row > 0 && binary.pixel(col, row - 1) != 0 {
                neighbours.push(labels[idx - w]);
            }

            if neighbours.is_empty() {
                labels[idx] = next_label;
                next_label += 1;
            } else {
                let min_label = *neighbours.iter().min().unwrap_or(&next_label);
                labels[idx] = min_label;
                for &nb in &neighbours {
                    union(&mut parent, min_label, nb);
                }
            }
        }
    }

    // Flatten labels
    for i in 0..n {
        if labels[i] != 0 {
            labels[i] = find(&mut parent, labels[i]);
        }
    }

    // Compute bounding boxes
    use std::collections::HashMap;
    let mut boxes: HashMap<usize, (usize, usize, usize, usize)> = HashMap::new(); // label -> (min_x, min_y, max_x, max_y)
    for row in 0..h {
        for col in 0..w {
            let lbl = labels[row * w + col];
            if lbl == 0 {
                continue;
            }
            let e = boxes.entry(lbl).or_insert((col, row, col, row));
            if col < e.0 {
                e.0 = col;
            }
            if row < e.1 {
                e.1 = row;
            }
            if col > e.2 {
                e.2 = col;
            }
            if row > e.3 {
                e.3 = row;
            }
        }
    }

    let mut result: Vec<Bbox> = boxes
        .values()
        .filter_map(|&(x0, y0, x1, y1)| {
            let bw = x1 - x0 + 1;
            let bh = y1 - y0 + 1;
            let area = bw * bh;
            if area >= min_area && area <= max_area {
                Some(Bbox {
                    x: x0,
                    y: y0,
                    w: bw,
                    h: bh,
                })
            } else {
                None
            }
        })
        .collect();

    // Sort blobs left-to-right
    result.sort_by_key(|b| b.x);
    result
}

fn find(parent: &mut [usize], x: usize) -> usize {
    if parent[x] != x {
        parent[x] = find(parent, parent[x]);
    }
    parent[x]
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[rb] = ra;
    }
}

// ── BitmapOcr ────────────────────────────────────────────────────────────────

/// Heuristic row detector: finds the dominant row band occupied by subtitle
/// text by looking at the row with the most foreground pixels in the bottom
/// quarter of the image.
fn detect_subtitle_row_band(binary: &BitmapImage) -> (usize, usize) {
    let h = binary.height;
    let w = binary.width;
    let start_row = h * 3 / 4;

    let mut best_row = start_row;
    let mut best_count = 0usize;

    for row in start_row..h {
        let count = (0..w)
            .filter(|&c| binary.pixel(c, row) != 0)
            .count();
        if count > best_count {
            best_count = count;
            best_row = row;
        }
    }

    // Expand to neighbouring rows that also have content
    let threshold = best_count / 4;
    let mut top = best_row;
    let mut bottom = best_row;

    while top > 0 {
        let count = (0..w)
            .filter(|&c| binary.pixel(c, top - 1) != 0)
            .count();
        if count >= threshold {
            top -= 1;
        } else {
            break;
        }
    }
    while bottom + 1 < h {
        let count = (0..w)
            .filter(|&c| binary.pixel(c, bottom + 1) != 0)
            .count();
        if count >= threshold {
            bottom += 1;
        } else {
            break;
        }
    }

    (top, bottom)
}

/// Bitmap OCR engine.
///
/// Uses template matching via normalised cross-correlation to identify
/// characters in bitmap subtitle images.
pub struct BitmapOcr {
    atlas: GlyphAtlas,
    /// Binarisation threshold (0–255).
    pub binarise_threshold: u8,
    /// Minimum glyph area in pixels.
    pub min_glyph_area: usize,
    /// Maximum glyph area in pixels.
    pub max_glyph_area: usize,
    /// Gap width (in pixels) between two blobs that is considered a word space.
    pub word_gap_px: usize,
}

impl BitmapOcr {
    /// Create a new `BitmapOcr` with the given glyph atlas.
    #[must_use]
    pub fn new(atlas: GlyphAtlas) -> Self {
        Self {
            atlas,
            binarise_threshold: 128,
            min_glyph_area: 4,
            max_glyph_area: 10_000,
            word_gap_px: 8,
        }
    }

    /// Extract text from a subtitle bitmap.
    ///
    /// The image is binarised, blobs are detected, matched against the atlas,
    /// and assembled left-to-right.  A space character is inserted when the
    /// horizontal gap between two consecutive blobs exceeds `word_gap_px`.
    ///
    /// `min_score` controls how strict the matching is (0.0–1.0; try 0.5–0.7).
    #[must_use]
    pub fn extract_text(&self, image: &BitmapImage, min_score: f32) -> String {
        let binary = image.binarise(self.binarise_threshold);
        let blobs = find_blobs(&binary, self.min_glyph_area, self.max_glyph_area);

        if blobs.is_empty() {
            return String::new();
        }

        let mut result = String::new();
        let mut prev_right: Option<usize> = None;

        for blob in &blobs {
            // Insert space if gap is large
            if let Some(right) = prev_right {
                let gap = blob.x.saturating_sub(right);
                if gap >= self.word_gap_px {
                    result.push(' ');
                }
            }

            let patch = binary.crop(blob.x, blob.y, blob.w, blob.h);
            if let Some(ch) = self.atlas.best_match(&patch, min_score) {
                result.push(ch);
            } else {
                result.push('?');
            }

            prev_right = Some(blob.x + blob.w);
        }

        result
    }

    /// Extract text only from the subtitle region (bottom quarter) of a frame.
    ///
    /// This is useful for full video frames where the subtitle appears at the
    /// bottom.
    #[must_use]
    pub fn extract_subtitle_region(&self, image: &BitmapImage, min_score: f32) -> String {
        let binary = image.binarise(self.binarise_threshold);
        let (top, bottom) = detect_subtitle_row_band(&binary);
        let h = bottom.saturating_sub(top) + 1;
        if h == 0 {
            return String::new();
        }
        let cropped = image.crop(0, top, image.width, h);
        self.extract_text(&cropped, min_score)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal 5×7 binary glyph for 'I' (vertical bar)
    const GLYPH_I: &[u8] = &[
        1, 1, 1, 1, 1,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        1, 1, 1, 1, 1,
    ];

    // A minimal 5×7 binary glyph for 'H'
    const GLYPH_H: &[u8] = &[
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
    ];

    fn make_atlas() -> GlyphAtlas {
        let mut atlas = GlyphAtlas::new();
        atlas.add_ascii_template('I', GLYPH_I, 5, 7);
        atlas.add_ascii_template('H', GLYPH_H, 5, 7);
        atlas
    }

    /// Render a single glyph into a wider bitmap at a given x offset.
    fn render_glyph_at(pixels: &mut [u8], img_w: usize, glyph: &[u8], gx: usize, gy: usize, gw: usize, gh: usize) {
        for row in 0..gh {
            for col in 0..gw {
                pixels[(gy + row) * img_w + (gx + col)] = glyph[row * gw + col] * 255;
            }
        }
    }

    #[test]
    fn test_bitmap_image_pixel() {
        let mut img = BitmapImage::new(4, 4);
        img.set_pixel(2, 1, 200);
        assert_eq!(img.pixel(2, 1), 200);
        assert_eq!(img.pixel(3, 3), 0);
        assert_eq!(img.pixel(10, 10), 0); // out of bounds
    }

    #[test]
    fn test_binarise() {
        let img = BitmapImage {
            pixels: vec![50, 100, 150, 200],
            width: 4,
            height: 1,
        };
        let bin = img.binarise(128);
        assert_eq!(bin.pixels, vec![0, 0, 255, 255]);
    }

    #[test]
    fn test_crop() {
        let mut img = BitmapImage::new(6, 6);
        img.set_pixel(2, 2, 255);
        let crop = img.crop(1, 1, 3, 3);
        // Pixel at (2,2) in original = (1,1) in crop
        assert_eq!(crop.pixel(1, 1), 255);
        assert_eq!(crop.pixel(0, 0), 0);
    }

    #[test]
    fn test_resize_preserves_nonzero() {
        let mut img = BitmapImage::new(10, 10);
        for i in 0..10 {
            img.set_pixel(i, i, 255);
        }
        let resized = img.resize(5, 5);
        // At least some pixels should remain non-zero
        assert!(resized.pixels.iter().any(|&p| p > 0));
    }

    #[test]
    fn test_glyph_template_ncc_self() {
        let tmpl = GlyphTemplate::new('A', &[0, 1, 0, 1, 1, 1, 0, 1, 0], 3, 3);
        let patch: Vec<f32> = vec![0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let score = tmpl.ncc_score(&patch);
        assert!(score > 0.99, "self-NCC should be ~1.0, got {score}");
    }

    #[test]
    fn test_glyph_template_ncc_opposite() {
        let tmpl = GlyphTemplate::new('A', &[1, 0, 1, 0, 0, 0, 1, 0, 1], 3, 3);
        // Inverted patch → NCC should be very negative
        let patch: Vec<f32> = vec![0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let score = tmpl.ncc_score(&patch);
        assert!(score < 0.0, "NCC of opposite should be negative, got {score}");
    }

    #[test]
    fn test_glyph_atlas_best_match_i() {
        let atlas = make_atlas();
        // Build an image of the 'I' glyph
        let mut img = BitmapImage::new(5, 7);
        for row in 0..7 {
            for col in 0..5 {
                img.set_pixel(col, row, GLYPH_I[row * 5 + col] * 255);
            }
        }
        let ch = atlas.best_match(&img, 0.5);
        assert_eq!(ch, Some('I'), "should recognise 'I'");
    }

    #[test]
    fn test_glyph_atlas_best_match_h() {
        let atlas = make_atlas();
        let mut img = BitmapImage::new(5, 7);
        for row in 0..7 {
            for col in 0..5 {
                img.set_pixel(col, row, GLYPH_H[row * 5 + col] * 255);
            }
        }
        let ch = atlas.best_match(&img, 0.5);
        assert_eq!(ch, Some('H'), "should recognise 'H'");
    }

    #[test]
    fn test_find_blobs_single() {
        let mut img = BitmapImage::new(10, 10);
        // Place a 3×3 blob at (3, 3)
        for row in 3..6 {
            for col in 3..6 {
                img.set_pixel(col, row, 255);
            }
        }
        let blobs = find_blobs(&img, 1, 1000);
        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].x, 3);
        assert_eq!(blobs[0].y, 3);
        assert_eq!(blobs[0].w, 3);
        assert_eq!(blobs[0].h, 3);
    }

    #[test]
    fn test_find_blobs_two_separate() {
        let mut img = BitmapImage::new(20, 10);
        // Blob 1 at x=1
        img.set_pixel(1, 1, 255);
        img.set_pixel(2, 1, 255);
        // Blob 2 at x=15 (well separated)
        img.set_pixel(15, 1, 255);
        img.set_pixel(16, 1, 255);
        let blobs = find_blobs(&img, 1, 1000);
        assert_eq!(blobs.len(), 2, "should find two blobs");
        // Sorted left-to-right
        assert!(blobs[0].x < blobs[1].x);
    }

    #[test]
    fn test_extract_text_single_char() {
        let atlas = make_atlas();
        let ocr = BitmapOcr::new(atlas);

        let img_w = 5;
        let img_h = 7;
        let mut pixels = vec![0u8; img_w * img_h];
        render_glyph_at(&mut pixels, img_w, GLYPH_I, 0, 0, 5, 7);
        let img = BitmapImage { pixels, width: img_w, height: img_h };

        let text = ocr.extract_text(&img, 0.5);
        assert!(text.contains('I'), "expected 'I', got: {text:?}");
    }

    #[test]
    fn test_extract_text_two_chars() {
        let atlas = make_atlas();
        let ocr = BitmapOcr::new(atlas);

        let img_w = 12; // 5 + 2 gap + 5
        let img_h = 7;
        let mut pixels = vec![0u8; img_w * img_h];
        render_glyph_at(&mut pixels, img_w, GLYPH_H, 0, 0, 5, 7);
        render_glyph_at(&mut pixels, img_w, GLYPH_I, 7, 0, 5, 7);
        let img = BitmapImage { pixels, width: img_w, height: img_h };

        let text = ocr.extract_text(&img, 0.5);
        assert!(text.contains('H'), "expected 'H', got: {text:?}");
        assert!(text.contains('I'), "expected 'I', got: {text:?}");
    }

    #[test]
    fn test_empty_image_returns_empty() {
        let atlas = make_atlas();
        let ocr = BitmapOcr::new(atlas);
        let img = BitmapImage::new(100, 50);
        let text = ocr.extract_text(&img, 0.5);
        assert!(text.is_empty(), "empty image should give empty text");
    }

    #[test]
    fn test_atlas_empty_returns_none() {
        let atlas = GlyphAtlas::new();
        let img = BitmapImage::new(5, 7);
        assert_eq!(atlas.best_match(&img, 0.0), None);
    }

    #[test]
    fn test_find_blobs_area_filter() {
        let mut img = BitmapImage::new(20, 10);
        // Tiny blob (area=1) – should be filtered out with min_area=2
        img.set_pixel(5, 5, 255);
        // Larger blob
        for col in 10..15 {
            img.set_pixel(col, 5, 255);
        }
        let blobs = find_blobs(&img, 2, 1000);
        assert_eq!(blobs.len(), 1, "tiny blob should be filtered");
        assert_eq!(blobs[0].x, 10);
    }
}
