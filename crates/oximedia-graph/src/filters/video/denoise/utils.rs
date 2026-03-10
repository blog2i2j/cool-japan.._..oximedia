/// Utility functions for denoise operations.

/// Clamp a value to a range.
#[must_use]
pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}

/// Linear interpolation.
#[must_use]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Smoothstep interpolation.
#[must_use]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Compute Gaussian weight.
#[must_use]
pub fn gaussian_weight(distance: f32, sigma: f32) -> f32 {
    let sigma_sq = sigma * sigma;
    (-(distance * distance) / (2.0 * sigma_sq)).exp()
}

/// Compute bilateral weight.
#[must_use]
pub fn bilateral_weight(
    spatial_dist: f32,
    intensity_diff: f32,
    sigma_space: f32,
    sigma_color: f32,
) -> f32 {
    let space_weight = gaussian_weight(spatial_dist, sigma_space);
    let color_weight = gaussian_weight(intensity_diff, sigma_color);
    space_weight * color_weight
}

/// Fast approximation of exp().
#[must_use]
pub fn fast_exp(x: f32) -> f32 {
    if x < -10.0 {
        return 0.0;
    }
    if x > 10.0 {
        return 1.0;
    }
    x.exp()
}

/// Convert decibels to linear scale.
#[must_use]
pub fn db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

/// Convert linear scale to decibels.
#[must_use]
pub fn linear_to_db(linear: f32) -> f32 {
    20.0 * linear.log10()
}
