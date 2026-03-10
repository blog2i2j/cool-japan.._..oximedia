// Color space utilities for denoise operations.

/// Convert RGB to YCbCr.
#[must_use]
pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let rf = r as f32;
    let gf = g as f32;
    let bf = b as f32;

    let y = (0.299 * rf + 0.587 * gf + 0.114 * bf)
        .round()
        .clamp(0.0, 255.0) as u8;
    let cb = (128.0 - 0.168736 * rf - 0.331264 * gf + 0.5 * bf)
        .round()
        .clamp(0.0, 255.0) as u8;
    let cr = (128.0 + 0.5 * rf - 0.418688 * gf - 0.081312 * bf)
        .round()
        .clamp(0.0, 255.0) as u8;

    (y, cb, cr)
}

/// Convert YCbCr to RGB.
#[must_use]
pub fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let yf = y as f32;
    let cbf = cb as f32 - 128.0;
    let crf = cr as f32 - 128.0;

    let r = (yf + 1.402 * crf).round().clamp(0.0, 255.0) as u8;
    let g = (yf - 0.344136 * cbf - 0.714136 * crf)
        .round()
        .clamp(0.0, 255.0) as u8;
    let b = (yf + 1.772 * cbf).round().clamp(0.0, 255.0) as u8;

    (r, g, b)
}

/// Separate luma and chroma processing.
pub fn process_luma_chroma<F>(data: &mut [u8], width: u32, height: u32, mut f: F)
where
    F: FnMut(&mut [u8], &mut [u8], &mut [u8]),
{
    if data.len() < (width * height * 3) as usize {
        return;
    }

    let pixel_count = (width * height) as usize;
    let mut y_plane = vec![0u8; pixel_count];
    let mut cb_plane = vec![0u8; pixel_count];
    let mut cr_plane = vec![0u8; pixel_count];

    for i in 0..pixel_count {
        let r = data[i * 3];
        let g = data[i * 3 + 1];
        let b = data[i * 3 + 2];

        let (y, cb, cr) = rgb_to_ycbcr(r, g, b);
        y_plane[i] = y;
        cb_plane[i] = cb;
        cr_plane[i] = cr;
    }

    f(&mut y_plane, &mut cb_plane, &mut cr_plane);

    for i in 0..pixel_count {
        let y = y_plane[i];
        let cb = cb_plane[i];
        let cr = cr_plane[i];

        let (r, g, b) = ycbcr_to_rgb(y, cb, cr);
        data[i * 3] = r;
        data[i * 3 + 1] = g;
        data[i * 3 + 2] = b;
    }
}
