// Gradient fill compute shader
//
// Evaluates linear or radial gradients and writes to an output texture.
// Color stops are provided via a storage buffer (up to 32 stops).
//
// Gradient types:
//   0 = Linear (from start to end point)
//   1 = Radial (from center outward)

struct GradientParams {
    gradient_type: u32,  // 0 = linear, 1 = radial
    num_stops: u32,      // number of active colour stops (max 32)
    width: u32,
    height: u32,
    // Linear: start and end in normalised [0,1] coords
    // Radial: center.xy and radius (start_x,start_y used as center, end_x as radius)
    start_x: f32,
    start_y: f32,
    end_x: f32,
    end_y: f32,
}

struct ColorStop {
    color: vec4<f32>,  // rgba
    position: f32,     // [0, 1]
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: GradientParams;
@group(0) @binding(2) var<storage, read> stops: array<ColorStop, 32>;

// Linearly interpolate between two colors
fn lerp_color(a: vec4<f32>, b: vec4<f32>, t: f32) -> vec4<f32> {
    return mix(a, b, vec4<f32>(t));
}

// Evaluate the gradient at a given parameter t in [0, 1]
fn evaluate_gradient(t: f32) -> vec4<f32> {
    let t_clamped = clamp(t, 0.0, 1.0);

    // If only one stop or below first stop, return first colour
    if params.num_stops <= 1u {
        return stops[0].color;
    }

    // If at or past the last stop, return last colour
    let last_idx = params.num_stops - 1u;
    if t_clamped <= stops[0].position {
        return stops[0].color;
    }
    if t_clamped >= stops[last_idx].position {
        return stops[last_idx].color;
    }

    // Find the pair of stops that bracket t_clamped
    var lower_idx: u32 = 0u;
    for (var i: u32 = 1u; i < params.num_stops; i = i + 1u) {
        if stops[i].position <= t_clamped {
            lower_idx = i;
        }
    }
    let upper_idx = lower_idx + 1u;

    let lower_pos = stops[lower_idx].position;
    let upper_pos = stops[upper_idx].position;
    let span = upper_pos - lower_pos;

    var local_t: f32;
    if span > 0.0001 {
        local_t = (t_clamped - lower_pos) / span;
    } else {
        local_t = 0.0;
    }

    return lerp_color(stops[lower_idx].color, stops[upper_idx].color, local_t);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }

    let uv = vec2<f32>(
        f32(gid.x) / f32(params.width - 1u),
        f32(gid.y) / f32(params.height - 1u)
    );

    var t: f32;

    if params.gradient_type == 0u {
        // Linear gradient: project pixel onto the start->end line
        let start = vec2<f32>(params.start_x, params.start_y);
        let end = vec2<f32>(params.end_x, params.end_y);
        let dir = end - start;
        let len_sq = dot(dir, dir);
        if len_sq > 0.0001 {
            t = dot(uv - start, dir) / len_sq;
        } else {
            t = 0.0;
        }
    } else {
        // Radial gradient: distance from center / radius
        let center = vec2<f32>(params.start_x, params.start_y);
        let radius = params.end_x;
        if radius > 0.0001 {
            t = length(uv - center) / radius;
        } else {
            t = 0.0;
        }
    }

    let color = evaluate_gradient(t);
    textureStore(output_tex, vec2<i32>(i32(gid.x), i32(gid.y)), color);
}
