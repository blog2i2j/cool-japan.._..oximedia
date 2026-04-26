// GPU sub-pixel refinement pass for motion estimation.
//
// Reads integer motion vectors from `mv_in` and performs a 3-point parabola
// fit at ±0.5 px to produce a refined floating-point MV in `mv_out`.
// One thread per block.

struct SubpixUniforms {
    frame_width:  u32,
    frame_height: u32,
    block_size:   u32,
    num_blocks:   u32,
}

@group(0) @binding(0) var<uniform>              u:       SubpixUniforms;
@group(0) @binding(1) var<storage, read>        ref_buf: array<u32>;
@group(0) @binding(2) var<storage, read>        cur_buf: array<u32>;
@group(0) @binding(3) var<storage, read>        mv_in:   array<vec4<i32>>;
@group(0) @binding(4) var<storage, read_write>  mv_out:  array<vec2<f32>>;

fn load_ref_f(xi: i32, yi: i32) -> f32 {
    let x = u32(clamp(xi, 0, i32(u.frame_width)  - 1));
    let y = u32(clamp(yi, 0, i32(u.frame_height) - 1));
    return f32(ref_buf[y * u.frame_width + x]);
}

fn load_cur_f(xi: i32, yi: i32) -> f32 {
    let x = u32(clamp(xi, 0, i32(u.frame_width)  - 1));
    let y = u32(clamp(yi, 0, i32(u.frame_height) - 1));
    return f32(cur_buf[y * u.frame_width + x]);
}

// Bilinear sample from a storage buffer at floating-point coords.
fn sample_bilinear_ref(xf: f32, yf: f32) -> f32 {
    let x0 = clamp(i32(floor(xf)), 0, i32(u.frame_width)  - 1);
    let y0 = clamp(i32(floor(yf)), 0, i32(u.frame_height) - 1);
    let x1 = clamp(x0 + 1, 0, i32(u.frame_width)  - 1);
    let y1 = clamp(y0 + 1, 0, i32(u.frame_height) - 1);
    let fx = xf - floor(xf);
    let fy = yf - floor(yf);
    let v00 = f32(ref_buf[u32(y0) * u.frame_width + u32(x0)]);
    let v10 = f32(ref_buf[u32(y0) * u.frame_width + u32(x1)]);
    let v01 = f32(ref_buf[u32(y1) * u.frame_width + u32(x0)]);
    let v11 = f32(ref_buf[u32(y1) * u.frame_width + u32(x1)]);
    return mix(mix(v00, v10, fx), mix(v01, v11, fx), fy);
}

// SAD between the current integer block and the reference at a floating-point
// displacement (dx, dy) using bilinear interpolation.
fn compute_sad_subpix(bx: i32, by: i32, dx: f32, dy: f32) -> f32 {
    let bs  = i32(u.block_size);
    var s: f32 = 0.0;
    for (var ry: i32 = 0; ry < bs; ry = ry + 1) {
        for (var rx: i32 = 0; rx < bs; rx = rx + 1) {
            let cur_val = load_cur_f(bx + rx, by + ry);
            let ref_val = sample_bilinear_ref(
                f32(bx + rx) + dx,
                f32(by + ry) + dy,
            );
            s = s + abs(cur_val - ref_val);
        }
    }
    return s;
}

@compute @workgroup_size(64)
fn subpixel_refine(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.num_blocks { return; }

    let block_idx = i32(gid.x);
    let mv        = mv_in[block_idx];

    let blocks_x = max(i32(u.frame_width)  / i32(u.block_size), 1);
    let bx        = (block_idx % blocks_x) * i32(u.block_size);
    let by        = (block_idx / blocks_x) * i32(u.block_size);

    let cx = f32(mv.x);
    let cy = f32(mv.y);

    // ── 3-point parabola fit along X ────────────────────────────────────────
    let s0x = compute_sad_subpix(bx, by, cx - 0.5, cy);
    let s1x = compute_sad_subpix(bx, by, cx,        cy);
    let s2x = compute_sad_subpix(bx, by, cx + 0.5, cy);

    var best_dx = cx;
    let denom_x = s0x - 2.0 * s1x + s2x;
    if s0x < s1x && s0x < s2x {
        best_dx = cx - 0.5;
    } else if s2x < s1x {
        best_dx = cx + 0.5;
    } else if abs(denom_x) > 1e-6 {
        best_dx = cx + 0.5 * (s0x - s2x) / denom_x;
    }

    // ── 3-point parabola fit along Y (using refined X) ─────────────────────
    let s0y = compute_sad_subpix(bx, by, best_dx, cy - 0.5);
    let s1y = compute_sad_subpix(bx, by, best_dx, cy);
    let s2y = compute_sad_subpix(bx, by, best_dx, cy + 0.5);

    var best_dy = cy;
    let denom_y = s0y - 2.0 * s1y + s2y;
    if s0y < s1y && s0y < s2y {
        best_dy = cy - 0.5;
    } else if s2y < s1y {
        best_dy = cy + 0.5;
    } else if abs(denom_y) > 1e-6 {
        best_dy = cy + 0.5 * (s0y - s2y) / denom_y;
    }

    mv_out[gid.x] = vec2<f32>(best_dx, best_dy);
}
