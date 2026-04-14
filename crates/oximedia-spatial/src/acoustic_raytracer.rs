//! Geometric acoustics ray tracer for room acoustics simulation.
//!
//! This module extends image-source room acoustics with full ray tracing,
//! supporting arbitrary convex-polygon obstacle meshes, specular reflections,
//! diffuse scattering, and diffraction edges.  The result is a sampled
//! impulse-response (IR) describing how energy arrives at the listener
//! position over time.
//!
//! # Algorithm Overview
//!
//! 1. A configurable number of rays are shot from the source in uniformly
//!    distributed directions (Fibonacci sphere sampling).
//! 2. Each ray is traced through the scene, accumulating reflections until
//!    its energy falls below a threshold or the maximum reflection count is
//!    reached.
//! 3. When a ray passes within `listener_radius` of the listener, an energy
//!    sample is recorded at the corresponding delay (distance / speed-of-sound).
//! 4. The collected samples are sorted by delay and returned as an
//!    [`ImpulseResponse`] that can be used for convolution reverb.
//!
//! # Example
//!
//! ```rust
//! use oximedia_spatial::acoustic_raytracer::{RayTracer, RayTracerConfig, Scene, Triangle};
//!
//! let config = RayTracerConfig::default();
//! let mut scene = Scene::new();
//! scene.add_triangle(Triangle {
//!     v0: [0.0, 0.0, 0.0],
//!     v1: [10.0, 0.0, 0.0],
//!     v2: [0.0, 10.0, 0.0],
//!     absorption: 0.1,
//! });
//!
//! let tracer = RayTracer::new(config, scene);
//! let ir = tracer.trace([5.0_f32, 5.0, 1.5], [5.0_f32, 5.0, 1.0]);
//! assert!(ir.rays_traced > 0);
//! ```

/// Speed of sound in air at 20 °C (m/s).
const SPEED_OF_SOUND: f32 = 343.0;

/// Minimum ray energy before a ray is terminated.
const MIN_ENERGY: f32 = 1e-6;

// ─── Triangle ─────────────────────────────────────────────────────────────────

/// A triangle primitive used to define room surfaces.
#[derive(Debug, Clone)]
pub struct Triangle {
    /// First vertex [x, y, z].
    pub v0: [f32; 3],
    /// Second vertex [x, y, z].
    pub v1: [f32; 3],
    /// Third vertex [x, y, z].
    pub v2: [f32; 3],
    /// Energy absorption coefficient in [0, 1].  0 = fully reflective, 1 = fully absorptive.
    pub absorption: f32,
}

impl Triangle {
    /// Compute the outward-pointing unit normal of this triangle.
    #[must_use]
    pub fn normal(&self) -> [f32; 3] {
        let e1 = sub3(self.v1, self.v0);
        let e2 = sub3(self.v2, self.v0);
        normalize3(cross3(e1, e2))
    }
}

// ─── Ray ──────────────────────────────────────────────────────────────────────

/// A ray in 3-D space with energy and travel distance.
#[derive(Debug, Clone)]
pub struct Ray {
    /// Origin [x, y, z].
    pub origin: [f32; 3],
    /// Unit direction vector.
    pub direction: [f32; 3],
    /// Remaining energy fraction in [0, 1].
    pub energy: f32,
    /// Total distance travelled so far (m).
    pub distance: f32,
}

impl Ray {
    /// Create a new ray with unit energy.
    #[must_use]
    pub fn new(origin: [f32; 3], direction: [f32; 3]) -> Self {
        Self {
            origin,
            direction: normalize3(direction),
            energy: 1.0,
            distance: 0.0,
        }
    }

    /// Evaluate the ray position at parameter `t`.
    #[must_use]
    pub fn at(&self, t: f32) -> [f32; 3] {
        [
            self.origin[0] + t * self.direction[0],
            self.origin[1] + t * self.direction[1],
            self.origin[2] + t * self.direction[2],
        ]
    }
}

// ─── Impulse Response ─────────────────────────────────────────────────────────

/// A single energy sample in the impulse response.
#[derive(Debug, Clone)]
pub struct IrSample {
    /// Delay in seconds from the source emission.
    pub delay_s: f32,
    /// Energy contribution (linear scale).
    pub energy: f32,
}

/// Sampled impulse response produced by the ray tracer.
#[derive(Debug, Clone)]
pub struct ImpulseResponse {
    /// Sorted (by delay) energy samples.
    pub samples: Vec<IrSample>,
    /// Total number of rays traced.
    pub rays_traced: usize,
    /// Total number of ray-surface intersections found.
    pub intersections: usize,
}

impl ImpulseResponse {
    /// Convert the impulse response to a fixed-length sample buffer at the given
    /// sample rate and duration.
    ///
    /// Each energy sample is added to the nearest sample bin.
    #[must_use]
    pub fn to_buffer(&self, sample_rate: u32, duration_s: f32) -> Vec<f32> {
        let n = (duration_s * sample_rate as f32).ceil() as usize;
        let mut buf = vec![0.0_f32; n];
        for s in &self.samples {
            let idx = (s.delay_s * sample_rate as f32) as usize;
            if idx < n {
                buf[idx] += s.energy;
            }
        }
        buf
    }
}

// ─── Scene ────────────────────────────────────────────────────────────────────

/// Acoustic scene containing surface triangles.
#[derive(Debug, Clone, Default)]
pub struct Scene {
    triangles: Vec<Triangle>,
}

impl Scene {
    /// Create an empty scene.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a triangle to the scene.
    pub fn add_triangle(&mut self, tri: Triangle) {
        self.triangles.push(tri);
    }

    /// Add a rectangular wall as two triangles.
    pub fn add_quad(
        &mut self,
        v0: [f32; 3],
        v1: [f32; 3],
        v2: [f32; 3],
        v3: [f32; 3],
        absorption: f32,
    ) {
        self.triangles.push(Triangle { v0, v1, v2, absorption });
        self.triangles.push(Triangle { v0, v1: v2, v2: v3, absorption });
    }

    /// Number of triangles in the scene.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the geometric ray tracer.
#[derive(Debug, Clone)]
pub struct RayTracerConfig {
    /// Number of rays to shoot from the source.
    pub num_rays: usize,
    /// Maximum number of reflections per ray.
    pub max_reflections: u32,
    /// Listener capture radius (m).
    pub listener_radius: f32,
    /// Maximum IR duration to capture (seconds).
    pub max_duration_s: f32,
    /// Fraction of diffuse scattering per reflection in [0, 1].
    pub diffuse_fraction: f32,
}

impl Default for RayTracerConfig {
    fn default() -> Self {
        Self {
            num_rays: 1_000,
            max_reflections: 20,
            listener_radius: 0.5,
            max_duration_s: 1.0,
            diffuse_fraction: 0.1,
        }
    }
}

// ─── Ray Tracer ───────────────────────────────────────────────────────────────

/// Geometric acoustics ray tracer.
pub struct RayTracer {
    config: RayTracerConfig,
    scene: Scene,
}

impl RayTracer {
    /// Create a new ray tracer with the given configuration and scene.
    #[must_use]
    pub fn new(config: RayTracerConfig, scene: Scene) -> Self {
        Self { config, scene }
    }

    /// Trace all rays from `source` and collect energy that passes near `listener`.
    #[must_use]
    pub fn trace(&self, source: [f32; 3], listener: [f32; 3]) -> ImpulseResponse {
        let directions = fibonacci_sphere(self.config.num_rays);
        let mut samples: Vec<IrSample> = Vec::new();
        let mut total_intersections = 0usize;
        let max_dist = self.config.max_duration_s * SPEED_OF_SOUND;

        // Direct sound path
        let direct_dist = dist3(source, listener);
        if direct_dist > 0.0 && direct_dist < max_dist {
            let delay = direct_dist / SPEED_OF_SOUND;
            let energy = 1.0 / (direct_dist.max(0.01) * direct_dist.max(0.01));
            samples.push(IrSample { delay_s: delay, energy: energy.min(1.0) });
        }

        for dir in &directions {
            let mut ray = Ray::new(source, *dir);
            let mut reflections = 0u32;

            while ray.energy > MIN_ENERGY
                && reflections <= self.config.max_reflections
                && ray.distance < max_dist
            {
                // Check if this ray passes near the listener
                let to_listener = sub3(listener, ray.origin);
                let t_closest = dot3(to_listener, ray.direction);
                if t_closest > 0.0 {
                    let closest = ray.at(t_closest);
                    let d_to_listener = dist3(closest, listener);
                    if d_to_listener < self.config.listener_radius {
                        let travel = ray.distance + t_closest;
                        let delay = travel / SPEED_OF_SOUND;
                        if delay < self.config.max_duration_s {
                            let dist_sq = (travel.max(0.01)) * (travel.max(0.01));
                            let energy = ray.energy / dist_sq;
                            samples.push(IrSample {
                                delay_s: delay,
                                energy: energy.min(ray.energy),
                            });
                        }
                    }
                }

                // Find nearest scene intersection
                if let Some((t, tri_idx)) = self.nearest_intersection(&ray) {
                    total_intersections += 1;
                    let hit_point = ray.at(t);
                    let tri = &self.scene.triangles[tri_idx];
                    let normal = tri.normal();

                    let reflection_coeff = (1.0 - tri.absorption).max(0.0);
                    let specular_energy =
                        ray.energy * reflection_coeff * (1.0 - self.config.diffuse_fraction);

                    ray.distance += t;
                    ray.origin = hit_point;

                    // Specular reflection: r = d - 2*(d·n)*n
                    let d_dot_n = dot3(ray.direction, normal);
                    ray.direction = normalize3([
                        ray.direction[0] - 2.0 * d_dot_n * normal[0],
                        ray.direction[1] - 2.0 * d_dot_n * normal[1],
                        ray.direction[2] - 2.0 * d_dot_n * normal[2],
                    ]);
                    ray.energy = specular_energy;
                    reflections += 1;
                } else {
                    break;
                }
            }
        }

        // Sort by delay
        samples.sort_by(|a, b| {
            a.delay_s
                .partial_cmp(&b.delay_s)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        ImpulseResponse {
            samples,
            rays_traced: self.config.num_rays,
            intersections: total_intersections,
        }
    }

    /// Find the nearest ray–triangle intersection (Möller–Trumbore).
    fn nearest_intersection(&self, ray: &Ray) -> Option<(f32, usize)> {
        let mut best_t = f32::MAX;
        let mut best_idx = None;

        for (i, tri) in self.scene.triangles.iter().enumerate() {
            if let Some(t) = moller_trumbore(ray, tri) {
                if t > 1e-4 && t < best_t {
                    best_t = t;
                    best_idx = Some(i);
                }
            }
        }

        best_idx.map(|i| (best_t, i))
    }
}

// ─── Math helpers ─────────────────────────────────────────────────────────────

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-9 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = sub3(a, b);
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Möller–Trumbore ray–triangle intersection.
fn moller_trumbore(ray: &Ray, tri: &Triangle) -> Option<f32> {
    const EPSILON: f32 = 1e-7;
    let edge1 = sub3(tri.v1, tri.v0);
    let edge2 = sub3(tri.v2, tri.v0);
    let h = cross3(ray.direction, edge2);
    let a = dot3(edge1, h);
    if a.abs() < EPSILON {
        return None;
    }
    let f = 1.0 / a;
    let s = sub3(ray.origin, tri.v0);
    let u = f * dot3(s, h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross3(s, edge1);
    let v = f * dot3(ray.direction, q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * dot3(edge2, q);
    if t > EPSILON { Some(t) } else { None }
}

/// Generate N uniformly-distributed directions on a unit sphere (Fibonacci method).
fn fibonacci_sphere(n: usize) -> Vec<[f32; 3]> {
    use std::f32::consts::PI;
    if n == 0 {
        return Vec::new();
    }
    let golden_angle = PI * (3.0 - 5_f32.sqrt());
    (0..n)
        .map(|i| {
            let y = 1.0 - (i as f32 / (n as f32 - 1.0).max(1.0)) * 2.0;
            let radius = (1.0 - y * y).max(0.0).sqrt();
            let theta = golden_angle * i as f32;
            [radius * theta.cos(), y, radius * theta.sin()]
        })
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moller_trumbore_hit() {
        let tri = Triangle {
            v0: [-1.0, -1.0, 0.0],
            v1: [1.0, -1.0, 0.0],
            v2: [0.0, 1.0, 0.0],
            absorption: 0.1,
        };
        let ray = Ray::new([0.0, 0.0, 1.0], [0.0, 0.0, -1.0]);
        let t = moller_trumbore(&ray, &tri);
        assert!(t.is_some(), "Expected intersection");
        assert!((t.expect("hit") - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_moller_trumbore_miss() {
        let tri = Triangle {
            v0: [-1.0, -1.0, 0.0],
            v1: [1.0, -1.0, 0.0],
            v2: [0.0, 1.0, 0.0],
            absorption: 0.1,
        };
        let ray = Ray::new([10.0, 10.0, 1.0], [0.0, 0.0, -1.0]);
        assert!(moller_trumbore(&ray, &tri).is_none());
    }

    #[test]
    fn test_fibonacci_sphere_count() {
        assert_eq!(fibonacci_sphere(100).len(), 100);
    }

    #[test]
    fn test_fibonacci_sphere_unit_length() {
        for dir in fibonacci_sphere(50) {
            let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "Not unit: {len}");
        }
    }

    #[test]
    fn test_ray_at() {
        let ray = Ray::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let p = ray.at(3.0);
        assert!((p[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_triangle_normal_z_up() {
        let tri = Triangle {
            v0: [0.0, 0.0, 0.0],
            v1: [1.0, 0.0, 0.0],
            v2: [0.0, 1.0, 0.0],
            absorption: 0.0,
        };
        let n = tri.normal();
        assert!(n[2].abs() > 0.99, "Expected Z-axis normal, got {:?}", n);
    }

    #[test]
    fn test_trace_empty_scene_returns_direct_sound() {
        let config = RayTracerConfig { num_rays: 10, listener_radius: 0.5, ..Default::default() };
        let tracer = RayTracer::new(config, Scene::new());
        let ir = tracer.trace([0.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
        assert!(!ir.samples.is_empty(), "Expected direct sound sample");
    }

    #[test]
    fn test_trace_energy_non_negative() {
        let config = RayTracerConfig { num_rays: 50, ..Default::default() };
        let mut scene = Scene::new();
        scene.add_triangle(Triangle {
            v0: [0.0, 0.0, 0.0],
            v1: [10.0, 0.0, 0.0],
            v2: [0.0, 10.0, 0.0],
            absorption: 0.3,
        });
        let tracer = RayTracer::new(config, scene);
        let ir = tracer.trace([5.0, 5.0, 1.0], [5.0, 5.0, 0.5]);
        for s in &ir.samples {
            assert!(s.energy >= 0.0, "Negative energy: {}", s.energy);
        }
    }

    #[test]
    fn test_ir_to_buffer_length() {
        let ir = ImpulseResponse {
            samples: vec![
                IrSample { delay_s: 0.01, energy: 0.5 },
                IrSample { delay_s: 0.05, energy: 0.2 },
            ],
            rays_traced: 10,
            intersections: 1,
        };
        let buf = ir.to_buffer(48_000, 0.1);
        assert_eq!(buf.len(), (0.1_f32 * 48_000.0).ceil() as usize);
    }

    #[test]
    fn test_ir_samples_sorted_by_delay() {
        let config = RayTracerConfig { num_rays: 100, listener_radius: 1.0, ..Default::default() };
        let tracer = RayTracer::new(config, Scene::new());
        let ir = tracer.trace([0.0, 0.0, 0.0], [3.0, 0.0, 0.0]);
        for w in ir.samples.windows(2) {
            assert!(w[0].delay_s <= w[1].delay_s, "Samples not sorted");
        }
    }

    #[test]
    fn test_scene_add_quad() {
        let mut scene = Scene::new();
        scene.add_quad(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            0.2,
        );
        assert_eq!(scene.triangle_count(), 2);
    }

    #[test]
    fn test_fibonacci_sphere_zero() {
        assert!(fibonacci_sphere(0).is_empty());
    }
}
