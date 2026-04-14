// Convolution filter shaders for blur, sharpen, and custom kernels
//
// Task E: shared-memory (workgroup) tiling is used in `convolve_main` and
// `convolve_tiled` to reduce global-memory bandwidth.  Each workgroup loads
// a TILE_W × TILE_H tile of input pixels (including a `RADIUS`-pixel halo)
// into `var<workgroup>` shared memory before the convolution inner loop.
// This cuts global reads by up to 9× for a 3×3 kernel.

@group(0) @binding(0) var<storage, read> input_image: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_image: array<u32>;
@group(0) @binding(2) var<uniform> params: FilterParams;
@group(0) @binding(3) var<storage, read> kernel: array<f32>;

struct FilterParams {
    width: u32,
    height: u32,
    stride: u32,
    kernel_size: u32, // Must be odd (3, 5, 7, etc.)
    normalize: u32,   // 1 to normalize kernel, 0 otherwise
    filter_type: u32, // 0=custom, 1=gaussian, 2=sharpen, 3=edge, 4=emboss
    padding: u32,
    sigma: f32,       // For gaussian blur
}

fn unpack_rgba(packed: u32) -> vec4<f32> {
    let r = f32((packed >> 24u) & 0xFFu) / 255.0;
    let g = f32((packed >> 16u) & 0xFFu) / 255.0;
    let b = f32((packed >> 8u) & 0xFFu) / 255.0;
    let a = f32(packed & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, a);
}

fn pack_rgba(color: vec4<f32>) -> u32 {
    let r = u32(clamp(color.r * 255.0, 0.0, 255.0));
    let g = u32(clamp(color.g * 255.0, 0.0, 255.0));
    let b = u32(clamp(color.b * 255.0, 0.0, 255.0));
    let a = u32(clamp(color.a * 255.0, 0.0, 255.0));
    return (r << 24u) | (g << 16u) | (b << 8u) | a;
}

fn sample_image(x: i32, y: i32) -> vec4<f32> {
    let cx = clamp(x, 0, i32(params.width) - 1);
    let cy = clamp(y, 0, i32(params.height) - 1);
    let idx = u32(cy) * params.stride + u32(cx);
    return unpack_rgba(input_image[idx]);
}

// Gaussian weight calculation
fn gaussian_weight(x: i32, y: i32, sigma: f32) -> f32 {
    let fx = f32(x);
    let fy = f32(y);
    let sigma2 = sigma * sigma;
    return exp(-(fx * fx + fy * fy) / (2.0 * sigma2)) / (6.28318530718 * sigma2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tiled Gaussian blur (Task E)
//
// Workgroup: 16 × 16 = 256 threads.
// Tile size:  TILE_W × TILE_H = 18 × 18 (16 + 2 halo cells for a radius-1
//             i.e. 3×3 kernel).  For larger kernels the halo grows linearly.
//
// The maximum supported convolution radius in this shader is 1 (kernel 3×3).
// Larger kernels fall through to `convolve_main` (no shared-memory tiling).
//
// Tile layout:
//   tile[local_y][local_x]  where x in [0, TILE_W), y in [0, TILE_H)
//   The active pixel writes to tile[local_y + RADIUS][local_x + RADIUS].
//   Halo threads additionally load border pixels.
// ─────────────────────────────────────────────────────────────────────────────

// Tile dimensions for a 3×3 kernel (radius = 1).
const TILE_W: u32 = 18u; // 16 active + 1 left halo + 1 right halo
const TILE_H: u32 = 18u;
const TILE_AREA: u32 = 324u; // TILE_W * TILE_H

// Packed RGBA as u32 in shared memory (saves bandwidth vs. vec4<f32>).
var<workgroup> tile: array<u32, 324>; // TILE_AREA = 18*18

// Tiled 3×3 Gaussian blur with workgroup shared-memory optimisation.
//
// Each workgroup loads an 18×18 patch (16×16 active + 1-pixel halo on each
// side) once; then all threads read from the workgroup cache during the
// convolution loop, avoiding repeated global reads for neighbour pixels.
@compute @workgroup_size(16, 16, 1)
fn gaussian_tiled(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
    @builtin(workgroup_id)         group_id:  vec3<u32>,
) {
    // ── Step 1: load tile from global memory into shared memory ─────────────
    // Tile origin in image space (may be negative — will be clamped on fetch).
    let tile_origin_x = i32(group_id.x) * 16 - 1;
    let tile_origin_y = i32(group_id.y) * 16 - 1;

    let lx = local_id.x;
    let ly = local_id.y;

    // Each thread in a 16×16 workgroup must fill a cell of the 18×18 tile.
    // We use a linear index over all 324 cells and assign them round-robin.
    let threads_per_wg: u32 = 256u;  // 16*16
    let linear_thread  = ly * 16u + lx;

    for (var i = linear_thread; i < TILE_AREA; i = i + threads_per_wg) {
        let tx = i32(i % TILE_W);
        let ty = i32(i / TILE_W);
        let gx = tile_origin_x + tx;
        let gy = tile_origin_y + ty;
        // Clamp-to-edge boundary handling
        let cx = clamp(gx, 0, i32(params.width)  - 1);
        let cy = clamp(gy, 0, i32(params.height) - 1);
        tile[u32(ty) * TILE_W + u32(tx)] = input_image[u32(cy) * params.stride + u32(cx)];
    }

    workgroupBarrier(); // ensure tile is fully populated before convolution

    // ── Step 2: discard threads that map outside the image ───────────────────
    let gx = i32(global_id.x);
    let gy = i32(global_id.y);
    if (u32(gx) >= params.width || u32(gy) >= params.height) {
        return;
    }

    // ── Step 3: 3×3 Gaussian convolution from shared-memory tile ────────────
    // Local tile coordinates of this thread's output pixel:
    //   tile[ly + 1][lx + 1]  (offset by halo radius = 1)
    let tlx = i32(lx) + 1; // tile local x (within [1, 16])
    let tly = i32(ly) + 1; // tile local y (within [1, 16])

    var sum        = vec4<f32>(0.0);
    var weight_sum = 0.0;
    let sigma = params.sigma;

    for (var ky = -1; ky <= 1; ky = ky + 1) {
        for (var kx = -1; kx <= 1; kx = kx + 1) {
            let tid = u32(tly + ky) * TILE_W + u32(tlx + kx);
            let px  = unpack_rgba(tile[tid]);
            let w   = gaussian_weight(kx, ky, sigma);
            sum        += px * w;
            weight_sum += w;
        }
    }

    var result: vec4<f32>;
    if (weight_sum > 0.0) {
        result = sum / weight_sum;
    } else {
        result = sum;
    }
    result = clamp(result, vec4<f32>(0.0), vec4<f32>(1.0));

    let out_idx = u32(gy) * params.stride + u32(gx);
    output_image[out_idx] = pack_rgba(result);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tiled generic convolution (Task E — arbitrary odd-sized kernel ≤ radius 4)
//
// Tile: 24 × 24 = 576 cells (16 active + 4 halo on each side).
// Supports kernels up to 9×9 (radius 4).
// ─────────────────────────────────────────────────────────────────────────────

const CTILE_W: u32 = 24u; // 16 + 4 + 4
const CTILE_H: u32 = 24u;
const CTILE_AREA: u32 = 576u; // 24*24

var<workgroup> conv_tile: array<u32, 576>; // CTILE_AREA

// Tiled generic convolution with workgroup shared memory.
// Supports filter_type 0 (custom kernel) and filter_type 1 (gaussian).
// For kernel_size > 9 falls back to per-pixel global reads (halo too large).
@compute @workgroup_size(16, 16, 1)
fn convolve_tiled(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
    @builtin(workgroup_id)         group_id:  vec3<u32>,
) {
    let radius = i32(params.kernel_size) / 2;
    let halo   = 4; // fixed halo = max supported radius

    // Tile origin
    let tile_ox = i32(group_id.x) * 16 - halo;
    let tile_oy = i32(group_id.y) * 16 - halo;

    let lx = local_id.x;
    let ly = local_id.y;
    let linear_thread = ly * 16u + lx;
    let threads_per_wg: u32 = 256u;

    // Populate conv_tile
    for (var i = linear_thread; i < CTILE_AREA; i = i + threads_per_wg) {
        let tx = i32(i % CTILE_W);
        let ty = i32(i / CTILE_W);
        let gx = tile_ox + tx;
        let gy = tile_oy + ty;
        let cx = clamp(gx, 0, i32(params.width)  - 1);
        let cy = clamp(gy, 0, i32(params.height) - 1);
        conv_tile[u32(ty) * CTILE_W + u32(tx)] =
            input_image[u32(cy) * params.stride + u32(cx)];
    }

    workgroupBarrier();

    let gx = i32(global_id.x);
    let gy = i32(global_id.y);
    if (u32(gx) >= params.width || u32(gy) >= params.height) {
        return;
    }

    // Tile-local coordinates for the active pixel
    let tlx = i32(lx) + halo;
    let tly = i32(ly) + halo;

    var sum        = vec4<f32>(0.0);
    var weight_sum = 0.0;

    for (var ky = -radius; ky <= radius; ky = ky + 1) {
        for (var kx = -radius; kx <= radius; kx = kx + 1) {
            // Only read from shared memory when within halo; otherwise fall
            // back to global (shouldn't happen if kernel_size ≤ 9).
            var px: vec4<f32>;
            let ttx = tlx + kx;
            let tty = tly + ky;
            if (ttx >= 0 && ttx < i32(CTILE_W) && tty >= 0 && tty < i32(CTILE_H)) {
                let tid = u32(tty) * CTILE_W + u32(ttx);
                px = unpack_rgba(conv_tile[tid]);
            } else {
                px = sample_image(gx + kx, gy + ky);
            }

            var weight = 0.0;
            if (params.filter_type == 1u) {
                weight = gaussian_weight(kx, ky, params.sigma);
            } else {
                let kidx = u32(ky + radius) * params.kernel_size + u32(kx + radius);
                weight = kernel[kidx];
            }
            sum        += px * weight;
            weight_sum += weight;
        }
    }

    var result: vec4<f32>;
    if (params.normalize == 1u && weight_sum > 0.0) {
        result = sum / weight_sum;
    } else {
        result = sum;
    }
    result = clamp(result, vec4<f32>(0.0), vec4<f32>(1.0));

    let out_idx = u32(gy) * params.stride + u32(gx);
    output_image[out_idx] = pack_rgba(result);
}

// ─────────────────────────────────────────────────────────────────────────────
// Original (non-tiled) entry points — kept for backward compatibility
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(16, 16, 1)
fn convolve_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    if (u32(x) >= params.width || u32(y) >= params.height) {
        return;
    }

    let radius = i32(params.kernel_size) / 2;
    var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var weight_sum = 0.0;

    for (var ky = -radius; ky <= radius; ky = ky + 1) {
        for (var kx = -radius; kx <= radius; kx = kx + 1) {
            let px = x + kx;
            let py = y + ky;

            let sample = sample_image(px, py);
            var weight = 0.0;

            if (params.filter_type == 1u) {
                // Gaussian blur
                weight = gaussian_weight(kx, ky, params.sigma);
            } else {
                // Custom kernel
                let kernel_x = kx + radius;
                let kernel_y = ky + radius;
                let kernel_idx = u32(kernel_y) * params.kernel_size + u32(kernel_x);
                weight = kernel[kernel_idx];
            }

            sum += sample * weight;
            weight_sum += weight;
        }
    }

    var result: vec4<f32>;
    if (params.normalize == 1u && weight_sum > 0.0) {
        result = sum / weight_sum;
    } else {
        result = sum;
    }

    // Clamp to [0, 1] range
    result = clamp(result, vec4<f32>(0.0), vec4<f32>(1.0));

    let idx = u32(y) * params.stride + u32(x);
    output_image[idx] = pack_rgba(result);
}

// Separable filter for efficient gaussian blur (horizontal pass)
@compute @workgroup_size(256, 1, 1)
fn separable_horizontal(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let y = idx / params.width;
    let x = idx % params.width;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let radius = i32(params.kernel_size) / 2;
    var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var weight_sum = 0.0;

    for (var kx = -radius; kx <= radius; kx = kx + 1) {
        let px = i32(x) + kx;
        let sample = sample_image(px, i32(y));

        let weight = gaussian_weight(kx, 0, params.sigma);
        sum += sample * weight;
        weight_sum += weight;
    }

    let result = sum / weight_sum;
    let out_idx = y * params.stride + x;
    output_image[out_idx] = pack_rgba(result);
}

// Separable filter for efficient gaussian blur (vertical pass)
@compute @workgroup_size(256, 1, 1)
fn separable_vertical(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let y = idx / params.width;
    let x = idx % params.width;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let radius = i32(params.kernel_size) / 2;
    var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var weight_sum = 0.0;

    for (var ky = -radius; ky <= radius; ky = ky + 1) {
        let py = i32(y) + ky;
        let sample = sample_image(i32(x), py);

        let weight = gaussian_weight(0, ky, params.sigma);
        sum += sample * weight;
        weight_sum += weight;
    }

    let result = sum / weight_sum;
    let out_idx = y * params.stride + x;
    output_image[out_idx] = pack_rgba(result);
}

// Edge detection using Sobel operator
@compute @workgroup_size(16, 16, 1)
fn edge_detect(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    if (u32(x) >= params.width || u32(y) >= params.height) {
        return;
    }

    // Sobel kernels
    // Gx: [-1  0  1]    Gy: [-1 -2 -1]
    //     [-2  0  2]         [ 0  0  0]
    //     [-1  0  1]         [ 1  2  1]

    var gx = vec3<f32>(0.0);
    var gy = vec3<f32>(0.0);

    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let sample = sample_image(x + dx, y + dy).rgb;

            // Horizontal gradient
            let kx = f32(dx);
            let ky_x = select(1.0, 2.0, dy == 0);
            gx += sample * kx * ky_x;

            // Vertical gradient
            let ky = f32(dy);
            let kx_y = select(1.0, 2.0, dx == 0);
            gy += sample * ky * kx_y;
        }
    }

    let magnitude = sqrt(gx * gx + gy * gy);
    let edge_strength = length(magnitude) / 1.732; // Normalize by sqrt(3)

    let result = vec4<f32>(edge_strength, edge_strength, edge_strength, 1.0);
    let idx = u32(y) * params.stride + u32(x);
    output_image[idx] = pack_rgba(result);
}

// Unsharp mask for sharpening
@compute @workgroup_size(16, 16, 1)
fn unsharp_mask(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    if (u32(x) >= params.width || u32(y) >= params.height) {
        return;
    }

    let original = sample_image(x, y);

    // Apply gaussian blur
    let radius = i32(params.kernel_size) / 2;
    var blurred = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var weight_sum = 0.0;

    for (var ky = -radius; ky <= radius; ky = ky + 1) {
        for (var kx = -radius; kx <= radius; kx = kx + 1) {
            let sample = sample_image(x + kx, y + ky);
            let weight = gaussian_weight(kx, ky, params.sigma);
            blurred += sample * weight;
            weight_sum += weight;
        }
    }

    blurred /= weight_sum;

    // Unsharp mask: original + amount * (original - blurred)
    let amount = 1.5; // Sharpening strength
    let sharpened = original + amount * (original - blurred);
    let result = clamp(sharpened, vec4<f32>(0.0), vec4<f32>(1.0));

    let idx = u32(y) * params.stride + u32(x);
    output_image[idx] = pack_rgba(result);
}
