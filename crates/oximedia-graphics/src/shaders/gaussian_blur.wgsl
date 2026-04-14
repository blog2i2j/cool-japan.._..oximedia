// Separable Gaussian blur compute shader
//
// Two-pass approach: run once with horizontal=1, then again with horizontal=0.
// Each invocation reads from input_tex and writes to output_tex.

struct BlurParams {
    kernel_radius: u32,
    sigma: f32,
    horizontal: u32, // 1 = horizontal pass, 0 = vertical pass
    _pad: u32,
}

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: BlurParams;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);

    // Guard: skip threads outside image bounds
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    // Early-out when sigma is effectively zero (no blur)
    if params.sigma <= 0.0 {
        let px = textureLoad(input_tex, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
        textureStore(output_tex, vec2<i32>(i32(gid.x), i32(gid.y)), px);
        return;
    }

    let inv_two_sigma_sq = -0.5 / (params.sigma * params.sigma);

    var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var weight_sum: f32 = 0.0;

    let radius = i32(params.kernel_radius);
    let max_coord = vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1);

    for (var i: i32 = -radius; i <= radius; i = i + 1) {
        let dist_sq = f32(i * i);
        let w = exp(dist_sq * inv_two_sigma_sq);

        var coord: vec2<i32>;
        if params.horizontal == 1u {
            coord = vec2<i32>(clamp(i32(gid.x) + i, 0, max_coord.x), i32(gid.y));
        } else {
            coord = vec2<i32>(i32(gid.x), clamp(i32(gid.y) + i, 0, max_coord.y));
        }

        sum = sum + w * textureLoad(input_tex, coord, 0);
        weight_sum = weight_sum + w;
    }

    let result = sum / weight_sum;
    textureStore(output_tex, vec2<i32>(i32(gid.x), i32(gid.y)), result);
}
