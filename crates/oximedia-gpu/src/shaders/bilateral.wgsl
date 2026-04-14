// Bilateral filter compute shader for GPU-accelerated edge-preserving denoising.
//
// Algorithm: for each output pixel, compute a weighted average of all neighboring
// pixels within the kernel radius, where the weight is the product of:
//   - a spatial Gaussian that depends on pixel distance
//   - a range Gaussian that depends on intensity (colour) difference
//
// Binding layout (matches BilateralParams in denoise.rs):
//   @binding(0): input RGBA image  (read-only storage, packed as u32)
//   @binding(1): output RGBA image (read-write storage, packed as u32)
//   @binding(2): uniform params    (BilateralParams, 32 bytes)

struct BilateralParams {
    width:              u32,
    height:             u32,
    kernel_radius:      u32,
    _pad:               u32,
    sigma_spatial:      f32,
    sigma_range:        f32,
    inv_two_sigma_s_sq: f32,
    inv_two_sigma_r_sq: f32,
}

@group(0) @binding(0) var<storage, read>       input_data:  array<u32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<u32>;
@group(0) @binding(2) var<uniform>             params:      BilateralParams;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn unpack_rgba(packed: u32) -> vec4<f32> {
    let r = f32((packed >> 24u) & 0xFFu);
    let g = f32((packed >> 16u) & 0xFFu);
    let b = f32((packed >>  8u) & 0xFFu);
    let a = f32( packed         & 0xFFu);
    return vec4<f32>(r, g, b, a);
}

fn pack_rgba(v: vec4<f32>) -> u32 {
    let ri = u32(clamp(v.r, 0.0, 255.0));
    let gi = u32(clamp(v.g, 0.0, 255.0));
    let bi = u32(clamp(v.b, 0.0, 255.0));
    let ai = u32(clamp(v.a, 0.0, 255.0));
    return (ri << 24u) | (gi << 16u) | (bi << 8u) | ai;
}

// ── Main entry point ─────────────────────────────────────────────────────────

@compute
@workgroup_size(16, 16)
fn bilateral_filter_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let px = gid.x;
    let py = gid.y;

    if px >= params.width || py >= params.height {
        return;
    }

    let w  = params.width;
    let h  = params.height;
    let r  = i32(params.kernel_radius);

    // Centre pixel colour (floating-point, 0–255 range).
    let center_packed = input_data[py * w + px];
    let center        = unpack_rgba(center_packed);

    var acc        = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var weight_sum = 0.0;

    for (var dy = -r; dy <= r; dy++) {
        for (var dx = -r; dx <= r; dx++) {
            let sx = clamp(i32(px) + dx, 0, i32(w) - 1);
            let sy = clamp(i32(py) + dy, 0, i32(h) - 1);

            let neighbor = unpack_rgba(input_data[u32(sy) * w + u32(sx)]);

            // Spatial Gaussian weight: exp(-(dx²+dy²) / (2·σ_s²))
            let spatial_dist_sq = f32(dx * dx + dy * dy);
            let w_spatial       = exp(-spatial_dist_sq * params.inv_two_sigma_s_sq);

            // Range Gaussian weight: exp(-‖c_p − c_q‖² / (2·σ_r²))
            // Use only RGB channels (not alpha) for the range distance.
            let delta = center.rgb - neighbor.rgb;
            let range_dist_sq = dot(delta, delta);
            let w_range       = exp(-range_dist_sq * params.inv_two_sigma_r_sq);

            let w_total  = w_spatial * w_range;
            weight_sum  += w_total;
            acc         += w_total * neighbor;
        }
    }

    var result: vec4<f32>;
    if weight_sum > 0.0 {
        result = acc / weight_sum;
    } else {
        result = center;
    }

    output_data[py * w + px] = pack_rgba(result);
}
