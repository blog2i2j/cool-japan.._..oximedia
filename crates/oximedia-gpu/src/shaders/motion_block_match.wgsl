// GPU block-matching pass for motion estimation using SAD cost metric.
//
// Each workgroup handles ONE block. Within the workgroup the 16×16 = 256
// threads each evaluate one candidate motion vector (centred around the seed
// from the previous pyramid level). A workgroup-shared parallel tree reduction
// finds the minimum SAD candidate without any subgroup operations, so the
// shader compiles on all wgpu backends.
//
// Uniform layout (must match `BlockMatchUniforms` on the Rust side):
//   block_size   (u32) — block edge length in pixels
//   search_half  (u32) — half the search window (the window is 16×16, so 8)
//   frame_width  (u32)
//   frame_height (u32)
//   mv_seed_x    (i32) — integer MV seed from the coarser pyramid level
//   mv_seed_y    (i32)
//   blocks_x     (u32) — number of blocks along X
//   blocks_y     (u32)
//
// Output buffer: `vec4<i32>` per block — (dx, dy, sad, _pad).

struct BlockMatchUniforms {
    block_size:   u32,
    search_half:  u32,
    frame_width:  u32,
    frame_height: u32,
    mv_seed_x:    i32,
    mv_seed_y:    i32,
    blocks_x:     u32,
    blocks_y:     u32,
}

@group(0) @binding(0) var<uniform>              u:       BlockMatchUniforms;
@group(0) @binding(1) var<storage, read>        ref_buf: array<u32>;
@group(0) @binding(2) var<storage, read>        cur_buf: array<u32>;
@group(0) @binding(3) var<storage, read_write>  mv_out:  array<vec4<i32>>;

// Workgroup-shared SAD cache: one u32 per thread (16×16 = 256 slots).
var<workgroup> sad_cache: array<u32, 256>;
var<workgroup> idx_cache: array<u32, 256>;

fn load_ref(x: i32, y: i32) -> u32 {
    let cx = u32(clamp(x, 0, i32(u.frame_width)  - 1));
    let cy = u32(clamp(y, 0, i32(u.frame_height) - 1));
    return ref_buf[cy * u.frame_width + cx];
}

fn load_cur(x: i32, y: i32) -> u32 {
    let cx = u32(clamp(x, 0, i32(u.frame_width)  - 1));
    let cy = u32(clamp(y, 0, i32(u.frame_height) - 1));
    return cur_buf[cy * u.frame_width + cx];
}

@compute @workgroup_size(16, 16)
fn block_match(
    @builtin(workgroup_id)          wgid:  vec3<u32>,
    @builtin(local_invocation_id)   lid:   vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    // Skip out-of-range workgroups (can happen when blocks_x * blocks_y is not
    // a multiple of the dispatch grid).
    if wgid.x >= u.blocks_x || wgid.y >= u.blocks_y {
        sad_cache[lidx] = 0xFFFFFFFFu;
        idx_cache[lidx]  = lidx;
        workgroupBarrier();
        return;
    }

    // Block origin in the current frame (pixel coords).
    let bx = i32(wgid.x * u.block_size);
    let by = i32(wgid.y * u.block_size);

    // This thread's candidate displacement, centred on the seed.
    let half = i32(u.search_half);
    let dx = i32(lid.x) - half + u.mv_seed_x;
    let dy = i32(lid.y) - half + u.mv_seed_y;

    // Compute SAD between current block at (bx, by) and the reference block
    // shifted by (dx, dy).
    var sad: u32 = 0u;
    let bs = i32(u.block_size);
    for (var ry: i32 = 0; ry < bs; ry = ry + 1) {
        for (var rx: i32 = 0; rx < bs; rx = rx + 1) {
            let cur_val = load_cur(bx + rx,      by + ry);
            let ref_val = load_ref(bx + rx + dx, by + ry + dy);
            let diff    = i32(cur_val) - i32(ref_val);
            sad = sad + u32(abs(diff));
        }
    }

    sad_cache[lidx] = sad;
    idx_cache[lidx]  = lidx;
    workgroupBarrier();

    // Parallel tree reduction (128 → 64 → 32 → 16 → 8 → 4 → 2 → 1).
    var stride: u32 = 128u;
    loop {
        if stride == 0u { break; }
        if lidx < stride {
            if sad_cache[lidx + stride] < sad_cache[lidx] {
                sad_cache[lidx] = sad_cache[lidx + stride];
                idx_cache[lidx]  = idx_cache[lidx + stride];
            }
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if lidx == 0u {
        let best    = idx_cache[0];
        let best_dx = i32(best % 16u) - half + u.mv_seed_x;
        let best_dy = i32(best / 16u) - half + u.mv_seed_y;
        let block_idx = i32(wgid.y * u.blocks_x + wgid.x);
        mv_out[block_idx] = vec4<i32>(best_dx, best_dy, i32(sad_cache[0]), 0);
    }
}
