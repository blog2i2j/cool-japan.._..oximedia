//! Pure-Rust linear algebra primitives for virtual production.
//!
//! Provides Point2, Point3, Vector3, Vector6, Matrix3, Matrix4, Matrix6,
//! Quaternion, and UnitQuaternion types as drop-in replacements for nalgebra.

pub mod matrix;
pub mod quaternion;
pub mod vector;

pub use matrix::{Matrix3, Matrix3x6, Matrix4, Matrix6};
pub use quaternion::{Quaternion, Unit, UnitQuaternion};
pub use vector::{Point2, Point3, Vector3, Vector6};
