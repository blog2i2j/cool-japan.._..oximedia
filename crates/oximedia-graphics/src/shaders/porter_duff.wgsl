// Porter-Duff compositing compute shader
//
// Implements all 12 canonical Porter-Duff operators plus Plus.
// Operates in premultiplied alpha space.
//
// Op codes:
//   0 = Clear      4 = DstOver    8 = SrcOut
//   1 = Src        5 = SrcIn      9 = DstOut
//   2 = Dst        6 = DstIn     10 = SrcAtop
//   3 = SrcOver    7 = SrcOut    11 = DstAtop
//  12 = Xor       13 = Plus

struct PdParams {
    op: u32,
    width: u32,
    height: u32,
    _pad: u32,
}

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var dst_tex: texture_2d<f32>;
@group(0) @binding(2) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> params: PdParams;

// Compute Porter-Duff factor pair (Fs, Fd) for the given operator
fn pd_factors(op: u32, a_src: f32, a_dst: f32) -> vec2<f32> {
    switch op {
        // Clear
        case 0u: { return vec2<f32>(0.0, 0.0); }
        // Src
        case 1u: { return vec2<f32>(1.0, 0.0); }
        // Dst
        case 2u: { return vec2<f32>(0.0, 1.0); }
        // SrcOver
        case 3u: { return vec2<f32>(1.0, 1.0 - a_src); }
        // DstOver
        case 4u: { return vec2<f32>(1.0 - a_dst, 1.0); }
        // SrcIn
        case 5u: { return vec2<f32>(a_dst, 0.0); }
        // DstIn
        case 6u: { return vec2<f32>(0.0, a_src); }
        // SrcOut
        case 7u: { return vec2<f32>(1.0 - a_dst, 0.0); }
        // DstOut
        case 8u: { return vec2<f32>(0.0, 1.0 - a_src); }
        // SrcAtop
        case 9u: { return vec2<f32>(a_dst, 1.0 - a_src); }
        // DstAtop
        case 10u: { return vec2<f32>(1.0 - a_dst, a_src); }
        // Xor
        case 11u: { return vec2<f32>(1.0 - a_dst, 1.0 - a_src); }
        // Plus (additive)
        case 12u: { return vec2<f32>(1.0, 1.0); }
        // Default: SrcOver
        default: { return vec2<f32>(1.0, 1.0 - a_src); }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));

    let src = textureLoad(src_tex, coord, 0);
    let dst = textureLoad(dst_tex, coord, 0);

    // Premultiply alpha
    let src_a = src.a;
    let dst_a = dst.a;
    let src_pm = vec4<f32>(src.rgb * src_a, src_a);
    let dst_pm = vec4<f32>(dst.rgb * dst_a, dst_a);

    let f = pd_factors(params.op, src_a, dst_a);

    var out = src_pm * f.x + dst_pm * f.y;

    // Un-premultiply for storage (avoid division by zero)
    let out_a = clamp(out.a, 0.0, 1.0);
    var result: vec4<f32>;
    if out_a > 0.001 {
        result = vec4<f32>(clamp(out.rgb / out_a, vec3<f32>(0.0), vec3<f32>(1.0)), out_a);
    } else {
        result = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    textureStore(output_tex, coord, result);
}
