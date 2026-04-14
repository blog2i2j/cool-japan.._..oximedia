use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion};
use oximedia_simd::deblock_filter::DeblockParams;
use oximedia_simd::hadamard::WhtSize;
use oximedia_simd::motion_search::MotionVector;
use oximedia_simd::satd::SatdBlockSize;
use oximedia_simd::{
    deblock_filter, detect_cpu_features, entropy_coding, forward_dct, hadamard, interpolate,
    inverse_dct, motion_search, psnr, resize, sad, satd, ssim, BlockSize, DctSize,
    InterpolationFilter,
};
use std::hint::black_box;

fn bench_dct_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_dct");

    // Detect CPU features to show which implementation is being tested
    let features = detect_cpu_features();
    eprintln!("CPU Features: {features:?}");

    // 4x4 DCT
    let input_4x4 = vec![100i16; 16];
    let mut output_4x4 = vec![0i16; 16];
    group.bench_function("4x4", |b| {
        b.iter(|| {
            forward_dct(
                black_box(&input_4x4),
                black_box(&mut output_4x4),
                black_box(DctSize::Dct4x4),
            )
        });
    });

    // 8x8 DCT
    let input_8x8 = vec![100i16; 64];
    let mut output_8x8 = vec![0i16; 64];
    group.bench_function("8x8", |b| {
        b.iter(|| {
            forward_dct(
                black_box(&input_8x8),
                black_box(&mut output_8x8),
                black_box(DctSize::Dct8x8),
            )
        });
    });

    // 16x16 DCT
    let input_16x16 = vec![100i16; 256];
    let mut output_16x16 = vec![0i16; 256];
    group.bench_function("16x16", |b| {
        b.iter(|| {
            forward_dct(
                black_box(&input_16x16),
                black_box(&mut output_16x16),
                black_box(DctSize::Dct16x16),
            )
        });
    });

    // 32x32 DCT
    let input_32x32 = vec![100i16; 1024];
    let mut output_32x32 = vec![0i16; 1024];
    group.bench_function("32x32", |b| {
        b.iter(|| {
            forward_dct(
                black_box(&input_32x32),
                black_box(&mut output_32x32),
                black_box(DctSize::Dct32x32),
            )
        });
    });

    group.finish();
}

fn bench_dct_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("inverse_dct");

    // 4x4 IDCT
    let input_4x4 = vec![100i16; 16];
    let mut output_4x4 = vec![0i16; 16];
    group.bench_function("4x4", |b| {
        b.iter(|| {
            inverse_dct(
                black_box(&input_4x4),
                black_box(&mut output_4x4),
                black_box(DctSize::Dct4x4),
            )
        });
    });

    // 8x8 IDCT
    let input_8x8 = vec![100i16; 64];
    let mut output_8x8 = vec![0i16; 64];
    group.bench_function("8x8", |b| {
        b.iter(|| {
            inverse_dct(
                black_box(&input_8x8),
                black_box(&mut output_8x8),
                black_box(DctSize::Dct8x8),
            )
        });
    });

    group.finish();
}

fn bench_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolation");

    let width = 64;
    let height = 64;
    let stride = 64;

    // Allocate extra space for filter taps
    let src = vec![128u8; (height + 16) * stride];
    let mut dst = vec![0u8; height * stride];

    // Bilinear
    group.bench_function("bilinear_64x64", |b| {
        b.iter(|| {
            interpolate(
                black_box(&src),
                black_box(&mut dst),
                black_box(stride),
                black_box(stride),
                black_box(width),
                black_box(height),
                black_box(8),
                black_box(8),
                black_box(InterpolationFilter::Bilinear),
            )
        });
    });

    // Bicubic
    group.bench_function("bicubic_64x64", |b| {
        b.iter(|| {
            interpolate(
                black_box(&src),
                black_box(&mut dst),
                black_box(stride),
                black_box(stride),
                black_box(width),
                black_box(height),
                black_box(8),
                black_box(8),
                black_box(InterpolationFilter::Bicubic),
            )
        });
    });

    // 8-tap
    group.bench_function("8tap_64x64", |b| {
        b.iter(|| {
            interpolate(
                black_box(&src),
                black_box(&mut dst),
                black_box(stride),
                black_box(stride),
                black_box(width),
                black_box(height),
                black_box(8),
                black_box(8),
                black_box(InterpolationFilter::EightTap),
            )
        });
    });

    group.finish();
}

fn bench_sad(c: &mut Criterion) {
    let mut group = c.benchmark_group("sad");

    // 16x16 SAD
    let src1_16 = vec![100u8; 16 * 16];
    let src2_16 = vec![110u8; 16 * 16];
    group.bench_function("16x16", |b| {
        b.iter(|| {
            sad(
                black_box(&src1_16),
                black_box(&src2_16),
                black_box(16),
                black_box(16),
                black_box(BlockSize::Block16x16),
            )
        });
    });

    // 32x32 SAD
    let src1_32 = vec![100u8; 32 * 32];
    let src2_32 = vec![110u8; 32 * 32];
    group.bench_function("32x32", |b| {
        b.iter(|| {
            sad(
                black_box(&src1_32),
                black_box(&src2_32),
                black_box(32),
                black_box(32),
                black_box(BlockSize::Block32x32),
            )
        });
    });

    // 64x64 SAD
    let src1_64 = vec![100u8; 64 * 64];
    let src2_64 = vec![110u8; 64 * 64];
    group.bench_function("64x64", |b| {
        b.iter(|| {
            sad(
                black_box(&src1_64),
                black_box(&src2_64),
                black_box(64),
                black_box(64),
                black_box(BlockSize::Block64x64),
            )
        });
    });

    group.finish();
}

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    group.throughput(criterion::Throughput::Bytes((64 * 64) as u64));

    let src1 = vec![100u8; 64 * 64];
    let src2 = vec![110u8; 64 * 64];

    group.bench_function("sad_64x64_throughput", |b| {
        b.iter(|| {
            sad(
                black_box(&src1),
                black_box(&src2),
                black_box(64),
                black_box(64),
                black_box(BlockSize::Block64x64),
            )
        });
    });

    group.finish();
}

// ── SATD 8×8 ─────────────────────────────────────────────────────────────────

fn bench_satd(c: &mut Criterion) {
    let mut group: BenchmarkGroup<criterion::measurement::WallTime> = c.benchmark_group("satd");

    let src_8x8: Vec<u8> = (0..64).map(|i| (i * 17 % 255) as u8).collect();
    let ref_8x8: Vec<u8> = (0..64).map(|i| (i * 11 % 255) as u8).collect();
    group.bench_function("satd_8x8", |b| {
        b.iter(|| {
            satd::satd(
                black_box(&src_8x8),
                black_box(&ref_8x8),
                SatdBlockSize::Block8x8,
            )
        });
    });

    let src_4x4: Vec<u8> = (0..16).map(|i| (i * 13 % 200) as u8).collect();
    let ref_4x4: Vec<u8> = (0..16).map(|i| (i * 7 % 200) as u8).collect();
    group.bench_function("satd_4x4", |b| {
        b.iter(|| {
            satd::satd(
                black_box(&src_4x4),
                black_box(&ref_4x4),
                SatdBlockSize::Block4x4,
            )
        });
    });

    group.finish();
}

// ── SSIM 128×128 ─────────────────────────────────────────────────────────────

fn bench_ssim(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssim");

    let width = 128usize;
    let height = 128usize;
    let src: Vec<u8> = (0..width * height).map(|i| (i * 3 % 255) as u8).collect();
    let ref_: Vec<u8> = (0..width * height).map(|i| (i * 5 % 255) as u8).collect();
    group.throughput(criterion::Throughput::Bytes((width * height) as u64));
    group.bench_function("ssim_128x128", |b| {
        b.iter(|| {
            ssim::ssim_luma(
                black_box(&src),
                black_box(&ref_),
                black_box(width),  // width
                black_box(height), // height
                black_box(width),  // stride
            )
        });
    });

    group.finish();
}

// ── PSNR 128×128 ─────────────────────────────────────────────────────────────

fn bench_psnr(c: &mut Criterion) {
    let mut group = c.benchmark_group("psnr");

    let n = 128 * 128;
    let src: Vec<u8> = (0..n).map(|i| (i * 3 % 255) as u8).collect();
    let ref_: Vec<u8> = (0..n).map(|i| (i * 5 % 255) as u8).collect();
    group.throughput(criterion::Throughput::Bytes(n as u64));
    group.bench_function("psnr_128x128", |b| {
        b.iter(|| psnr::psnr_u8(black_box(&src), black_box(&ref_)));
    });

    group.finish();
}

// ── Deblock filter — 64-byte block ───────────────────────────────────────────

fn bench_deblock(c: &mut Criterion) {
    let mut group = c.benchmark_group("deblock_filter");

    // A 16-pixel-wide, 8-row block with a sharp boundary in the middle.
    let mut block: Vec<u8> = (0..128)
        .map(|i| if i < 64 { 64u8 } else { 192u8 })
        .collect();
    let params = DeblockParams::default_medium();
    group.bench_function("deblock_horizontal_64byte", |b| {
        b.iter(|| {
            deblock_filter::deblock_horizontal(
                black_box(&mut block),
                black_box(16), // width
                black_box(8),  // height
                black_box(16), // stride
                black_box(&params),
            )
        });
    });

    group.finish();
}

// ── Motion search — diamond 8×8 ──────────────────────────────────────────────

fn bench_motion_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("motion_search");

    // 32×32 reference frame, 8×8 source block.
    let ref_w = 32usize;
    let ref_h = 32usize;
    let ref_frame: Vec<u8> = (0..ref_w * ref_h).map(|i| (i * 7 % 255) as u8).collect();
    let src_block: Vec<u8> = (0..ref_w * ref_h).map(|i| (i * 3 % 255) as u8).collect();
    let initial_mv = MotionVector::new(0, 0);

    group.bench_function("diamond_search_8x8", |b| {
        b.iter(|| {
            motion_search::diamond_search(
                black_box(&src_block),
                black_box(ref_w),
                black_box(&ref_frame),
                black_box(ref_w),
                black_box(ref_w),
                black_box(ref_h),
                black_box(8),
                black_box(8),
                black_box(4i32),
                black_box(4i32),
                black_box(initial_mv),
                black_box(8i32),
            )
        });
    });

    group.finish();
}

// ── Resize bilinear 64×64 → 32×32 ────────────────────────────────────────────

fn bench_resize(c: &mut Criterion) {
    let mut group = c.benchmark_group("resize");

    let src: Vec<u8> = (0..64 * 64).map(|i| (i * 3 % 255) as u8).collect();
    let mut dst = vec![0u8; 32 * 32];
    group.throughput(criterion::Throughput::Bytes((64 * 64) as u64));
    group.bench_function("bilinear_64x64_to_32x32", |b| {
        b.iter(|| {
            resize::resize_bilinear(
                black_box(&src),
                black_box(64),
                black_box(64),
                black_box(&mut dst),
                black_box(32),
                black_box(32),
            )
        });
    });

    group.finish();
}

// ── Entropy coding — encode/decode ───────────────────────────────────────────

fn bench_entropy_coding(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_coding");

    // Encode/decode a 32-element sequence using 4-bit values.
    let values: Vec<u8> = (0..32).map(|i| (i % 16) as u8).collect();

    group.bench_function("pack_bits_4bpp", |b| {
        b.iter(|| entropy_coding::pack_bits(black_box(&values), black_box(4)));
    });

    let packed = entropy_coding::pack_bits(&values, 4).expect("pack_bits should succeed");
    group.bench_function("unpack_bits_4bpp", |b| {
        b.iter(|| entropy_coding::unpack_bits(black_box(&packed), black_box(4), black_box(32)));
    });

    group.finish();
}

// ── Hadamard 4×4 ─────────────────────────────────────────────────────────────

fn bench_hadamard(c: &mut Criterion) {
    let mut group = c.benchmark_group("hadamard");

    let src_4x4: Vec<u8> = (0..16).map(|i| (i * 13 % 200) as u8).collect();
    let ref_4x4: Vec<u8> = (0..16).map(|i| (i * 7 % 200) as u8).collect();
    group.bench_function("wht_satd_4x4", |b| {
        b.iter(|| {
            hadamard::wht_satd(
                black_box(&src_4x4),
                black_box(&ref_4x4),
                black_box(WhtSize::N4),
            )
        });
    });

    let mut block_4x4 = vec![0i32; 16];
    group.bench_function("wht_2d_inplace_4x4", |b| {
        b.iter(|| {
            // Reset to avoid accumulation across iterations.
            for (j, v) in block_4x4.iter_mut().enumerate() {
                *v = (j as i32 * 3) % 100;
            }
            hadamard::wht_2d_inplace(black_box(&mut block_4x4), black_box(WhtSize::N4))
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_dct_forward,
    bench_dct_inverse,
    bench_interpolation,
    bench_sad,
    bench_throughput,
    bench_satd,
    bench_ssim,
    bench_psnr,
    bench_deblock,
    bench_motion_search,
    bench_resize,
    bench_entropy_coding,
    bench_hadamard,
);

criterion_main!(benches);
