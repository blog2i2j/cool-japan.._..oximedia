//! WebAssembly bindings for `oximedia-multicam` multi-camera production.
//!
//! Provides `WasmMultiCamCompositor` for creating multi-view layouts
//! and standalone compositing functions in the browser.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// WasmMultiCamCompositor
// ---------------------------------------------------------------------------

/// Browser-side multi-camera frame compositor.
#[wasm_bindgen]
pub struct WasmMultiCamCompositor {
    width: u32,
    height: u32,
    spacing: u32,
}

#[wasm_bindgen]
impl WasmMultiCamCompositor {
    /// Create a new compositor with the given output resolution.
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> Result<WasmMultiCamCompositor, JsValue> {
        if width == 0 || height == 0 {
            return Err(crate::utils::js_err("Width and height must be > 0"));
        }
        Ok(Self {
            width,
            height,
            spacing: 4,
        })
    }

    /// Set grid spacing in pixels.
    pub fn set_spacing(&mut self, spacing: u32) {
        self.spacing = spacing;
    }

    /// Composite multiple frames into a grid layout.
    ///
    /// `frames_data` contains all frames concatenated: each frame is
    /// `frame_w * frame_h * 4` bytes of RGBA data.
    ///
    /// Returns composited RGBA pixel data at the output resolution.
    pub fn composite_grid(
        &self,
        frames_data: &[u8],
        frame_w: u32,
        frame_h: u32,
        num_frames: u32,
        cols: u32,
    ) -> Result<Vec<u8>, JsValue> {
        if num_frames == 0 {
            return Err(crate::utils::js_err("num_frames must be > 0"));
        }
        if frame_w == 0 || frame_h == 0 {
            return Err(crate::utils::js_err("Frame dimensions must be > 0"));
        }

        let frame_size = (frame_w as usize) * (frame_h as usize) * 4;
        let expected_total = frame_size * (num_frames as usize);
        if frames_data.len() < expected_total {
            return Err(crate::utils::js_err(&format!(
                "frames_data too small: need {} bytes for {} frames at {}x{}, got {}",
                expected_total,
                num_frames,
                frame_w,
                frame_h,
                frames_data.len()
            )));
        }

        let grid_cols = if cols == 0 {
            let (_, c) =
                oximedia_multicam::composite::grid::GridCompositor::optimal_grid_for_angles(
                    num_frames as usize,
                );
            c
        } else {
            cols as usize
        };

        let grid_rows = ((num_frames as usize) + grid_cols - 1) / grid_cols;

        let mut grid =
            oximedia_multicam::composite::grid::GridCompositor::new(self.width, self.height);
        grid.set_spacing(self.spacing);
        let cells = grid.calculate_grid(grid_rows, grid_cols);

        let out_size = (self.width as usize) * (self.height as usize) * 4;
        let mut output = vec![0u8; out_size];

        for i in 0..(num_frames as usize) {
            if i >= cells.len() {
                break;
            }
            let (cx, cy, cw, ch) = cells[i];
            let frame_start = i * frame_size;
            let frame_end = frame_start + frame_size;
            let frame = &frames_data[frame_start..frame_end];

            blit_scaled(
                frame,
                frame_w,
                frame_h,
                &mut output,
                self.width,
                self.height,
                cx,
                cy,
                cw,
                ch,
            );
        }

        Ok(output)
    }

    /// Composite two frames in picture-in-picture layout.
    ///
    /// `main_frame` and `pip_frame` are RGBA data for the main and PIP views.
    pub fn composite_pip(
        &self,
        main_frame: &[u8],
        pip_frame: &[u8],
        main_w: u32,
        main_h: u32,
        pip_x: u32,
        pip_y: u32,
        pip_w: u32,
        pip_h: u32,
    ) -> Result<Vec<u8>, JsValue> {
        let expected_main = (main_w as usize) * (main_h as usize) * 4;
        if main_frame.len() < expected_main {
            return Err(crate::utils::js_err(&format!(
                "Main frame too small: need {} bytes, got {}",
                expected_main,
                main_frame.len()
            )));
        }

        let out_size = (self.width as usize) * (self.height as usize) * 4;
        let mut output = vec![0u8; out_size];

        // Blit main frame scaled to output
        blit_scaled(
            main_frame,
            main_w,
            main_h,
            &mut output,
            self.width,
            self.height,
            0,
            0,
            self.width,
            self.height,
        );

        // Determine PIP source dimensions from data
        let pip_pixel_count = pip_frame.len() / 4;
        if pip_pixel_count == 0 {
            return Err(crate::utils::js_err("PIP frame is empty"));
        }
        let pip_src_w = (pip_pixel_count as f64).sqrt() as u32;
        let pip_src_h = if pip_src_w > 0 {
            (pip_pixel_count as u32) / pip_src_w
        } else {
            1
        };

        // Blit PIP overlay
        blit_scaled(
            pip_frame,
            pip_src_w,
            pip_src_h,
            &mut output,
            self.width,
            self.height,
            pip_x,
            pip_y,
            pip_w,
            pip_h,
        );

        Ok(output)
    }

    /// Get output width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get output height.
    pub fn height(&self) -> u32 {
        self.height
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Composite multiple frames into a PIP layout.
///
/// Both frames must be `main_w * main_h * 4` RGBA bytes.
#[wasm_bindgen]
pub fn wasm_composite_pip(
    main_frame: &[u8],
    overlay: &[u8],
    main_w: u32,
    main_h: u32,
    pip_x: u32,
    pip_y: u32,
    pip_w: u32,
    pip_h: u32,
) -> Result<Vec<u8>, JsValue> {
    let expected = (main_w as usize) * (main_h as usize) * 4;
    if main_frame.len() < expected {
        return Err(crate::utils::js_err(&format!(
            "Main frame too small: need {} bytes, got {}",
            expected,
            main_frame.len()
        )));
    }

    let mut output = main_frame.to_vec();

    let ov_pixel_count = overlay.len() / 4;
    if ov_pixel_count == 0 {
        return Err(crate::utils::js_err("Overlay frame is empty"));
    }
    let ov_w = (ov_pixel_count as f64).sqrt() as u32;
    let ov_h = if ov_w > 0 {
        (ov_pixel_count as u32) / ov_w
    } else {
        1
    };

    blit_scaled(
        overlay,
        ov_w,
        ov_h,
        &mut output,
        main_w,
        main_h,
        pip_x,
        pip_y,
        pip_w,
        pip_h,
    );

    Ok(output)
}

/// Composite frames into a grid layout.
///
/// `frames` contains all frames concatenated:
/// each frame is `frame_w * frame_h * 4` bytes.
#[wasm_bindgen]
pub fn wasm_composite_grid(
    frames: &[u8],
    frame_w: u32,
    frame_h: u32,
    num_frames: u32,
    cols: u32,
) -> Result<Vec<u8>, JsValue> {
    if num_frames == 0 || frame_w == 0 || frame_h == 0 {
        return Err(crate::utils::js_err(
            "num_frames, frame_w, frame_h must all be > 0",
        ));
    }

    let frame_size = (frame_w as usize) * (frame_h as usize) * 4;
    let expected = frame_size * (num_frames as usize);
    if frames.len() < expected {
        return Err(crate::utils::js_err(&format!(
            "frames data too small: need {} bytes, got {}",
            expected,
            frames.len()
        )));
    }

    let grid_cols = if cols == 0 {
        let (_, c) = oximedia_multicam::composite::grid::GridCompositor::optimal_grid_for_angles(
            num_frames as usize,
        );
        c as u32
    } else {
        cols
    };
    let grid_rows = ((num_frames as u32) + grid_cols - 1) / grid_cols;

    let out_w = frame_w * grid_cols;
    let out_h = frame_h * grid_rows;
    let out_size = (out_w as usize) * (out_h as usize) * 4;
    let mut output = vec![0u8; out_size];

    for i in 0..(num_frames as usize) {
        let col = (i as u32) % grid_cols;
        let row = (i as u32) / grid_cols;
        let cx = col * frame_w;
        let cy = row * frame_h;

        let frame_start = i * frame_size;
        let frame_end = frame_start + frame_size;
        let frame = &frames[frame_start..frame_end];

        blit_scaled(
            frame,
            frame_w,
            frame_h,
            &mut output,
            out_w,
            out_h,
            cx,
            cy,
            frame_w,
            frame_h,
        );
    }

    Ok(output)
}

/// List available multi-camera layouts as a JSON array.
#[wasm_bindgen]
pub fn wasm_list_layouts() -> String {
    "[{\"name\":\"grid\",\"description\":\"Auto-sized grid layout\"},\
     {\"name\":\"pip\",\"description\":\"Picture-in-picture\"},\
     {\"name\":\"side_by_side\",\"description\":\"Horizontal split\"},\
     {\"name\":\"stack\",\"description\":\"Vertical stack\"}]"
        .to_string()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Nearest-neighbor scaled blit from source into destination region.
fn blit_scaled(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst: &mut [u8],
    dst_total_w: u32,
    dst_total_h: u32,
    dst_x: u32,
    dst_y: u32,
    region_w: u32,
    region_h: u32,
) {
    if src_w == 0 || src_h == 0 || region_w == 0 || region_h == 0 {
        return;
    }

    for ry in 0..region_h {
        for rx in 0..region_w {
            let px = dst_x + rx;
            let py = dst_y + ry;
            if px >= dst_total_w || py >= dst_total_h {
                continue;
            }

            let sx = (rx as u64 * src_w as u64 / region_w as u64) as u32;
            let sy = (ry as u64 * src_h as u64 / region_h as u64) as u32;
            let sx = sx.min(src_w - 1);
            let sy = sy.min(src_h - 1);

            let src_idx = ((sy * src_w + sx) * 4) as usize;
            let dst_idx = ((py * dst_total_w + px) * 4) as usize;

            if src_idx + 3 < src.len() && dst_idx + 3 < dst.len() {
                dst[dst_idx] = src[src_idx];
                dst[dst_idx + 1] = src[src_idx + 1];
                dst[dst_idx + 2] = src[src_idx + 2];
                dst[dst_idx + 3] = src[src_idx + 3];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgba(w: u32, h: u32, val: u8) -> Vec<u8> {
        vec![val; (w as usize) * (h as usize) * 4]
    }

    #[test]
    fn test_compositor_creation() {
        let c = WasmMultiCamCompositor::new(1920, 1080);
        assert!(c.is_ok());
    }

    #[test]
    fn test_compositor_zero_size() {
        assert!(WasmMultiCamCompositor::new(0, 100).is_err());
    }

    #[test]
    fn test_composite_grid_standalone() {
        let frame1 = make_rgba(10, 10, 100);
        let frame2 = make_rgba(10, 10, 200);
        let mut frames = Vec::new();
        frames.extend_from_slice(&frame1);
        frames.extend_from_slice(&frame2);

        let result = wasm_composite_grid(&frames, 10, 10, 2, 2);
        assert!(result.is_ok());
        // 2 cols, 1 row -> 20x10
        assert_eq!(result.expect("ok").len(), 20 * 10 * 4);
    }

    #[test]
    fn test_composite_pip_standalone() {
        let main_frame = make_rgba(20, 20, 100);
        let pip_frame = make_rgba(10, 10, 200);

        let result = wasm_composite_pip(&main_frame, &pip_frame, 20, 20, 10, 10, 8, 8);
        assert!(result.is_ok());
        assert_eq!(result.expect("ok").len(), 20 * 20 * 4);
    }

    #[test]
    fn test_list_layouts() {
        let json = wasm_list_layouts();
        assert!(json.contains("grid"));
        assert!(json.contains("pip"));
    }

    #[test]
    fn test_compositor_grid_method() {
        let c = WasmMultiCamCompositor::new(100, 100).expect("ok");
        let frame1 = make_rgba(50, 50, 80);
        let frame2 = make_rgba(50, 50, 160);
        let mut frames = Vec::new();
        frames.extend_from_slice(&frame1);
        frames.extend_from_slice(&frame2);

        let result = c.composite_grid(&frames, 50, 50, 2, 2);
        assert!(result.is_ok());
        assert_eq!(result.expect("ok").len(), 100 * 100 * 4);
    }

    #[test]
    fn test_blit_scaled() {
        let src = vec![255u8; 4 * 4 * 4];
        let mut dst = vec![0u8; 8 * 8 * 4];
        blit_scaled(&src, 4, 4, &mut dst, 8, 8, 0, 0, 8, 8);
        assert_eq!(dst[0], 255);
    }
}
