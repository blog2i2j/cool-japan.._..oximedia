// GPU Gaussian pyramid downsample pass for motion estimation.
//
// Reads luma samples from `input_buf` (u32 values, range 0–255 packed as u32),
// box-averages each 2×2 neighbourhood, and writes the result to `output_buf`.
// All coordinates are clamped so fractional-size frames are handled correctly.

struct PyramidUniforms {
    in_width:  u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
}

@group(0) @binding(0) var<uniform>           u:          PyramidUniforms;
@group(0) @binding(1) var<storage, read>     input_buf:  array<u32>;
@group(0) @binding(2) var<storage, read_write> output_buf: array<u32>;

fn load_pixel(x: u32, y: u32) -> u32 {
    let cx = min(x, u.in_width  - 1u);
    let cy = min(y, u.in_height - 1u);
    return input_buf[cy * u.in_width + cx];
}

@compute @workgroup_size(8, 8)
fn downsample_r8(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.out_width || gid.y >= u.out_height { return; }

    let sx = gid.x * 2u;
    let sy = gid.y * 2u;

    // Box-average of the 2×2 source neighbourhood.
    let v00 = load_pixel(sx,      sy);
    let v10 = load_pixel(sx + 1u, sy);
    let v01 = load_pixel(sx,      sy + 1u);
    let v11 = load_pixel(sx + 1u, sy + 1u);

    let avg = (v00 + v10 + v01 + v11 + 2u) / 4u;   // +2 for rounding
    output_buf[gid.y * u.out_width + gid.x] = avg;
}
