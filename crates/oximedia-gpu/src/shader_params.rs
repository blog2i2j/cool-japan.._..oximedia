//! Shader parameter management — param types, individual params, and uniform blocks.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// The data type of a shader parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamType {
    /// 32-bit float scalar.
    Float,
    /// 32-bit signed integer scalar.
    Int,
    /// Two-component float vector.
    Vec2,
    /// Three-component float vector.
    Vec3,
    /// Four-component float vector.
    Vec4,
    /// 4×4 float matrix.
    Matrix4x4,
    /// Texture / sampler handle.
    Texture,
}

impl ParamType {
    /// Returns the byte size of this parameter type in GPU memory.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Float => 4,
            Self::Int => 4,
            Self::Vec2 => 8,
            Self::Vec3 => 12,
            Self::Vec4 => 16,
            Self::Matrix4x4 => 64,
            // Texture handles are represented as 8-byte opaque handles.
            Self::Texture => 8,
        }
    }
}

/// A single named parameter within a uniform block.
#[derive(Debug, Clone)]
pub struct ShaderParam {
    /// The GLSL/WGSL name of the parameter.
    pub name: String,
    /// The data type.
    pub param_type: ParamType,
    /// Byte offset within the owning uniform block.
    pub offset: u32,
}

impl ShaderParam {
    /// Creates a new `ShaderParam`.
    #[must_use]
    pub fn new(name: impl Into<String>, param_type: ParamType, offset: u32) -> Self {
        Self {
            name: name.into(),
            param_type,
            offset,
        }
    }

    /// The byte offset *past* the end of this parameter (exclusive end offset).
    #[must_use]
    pub fn end_offset(&self) -> u32 {
        self.offset + self.param_type.byte_size() as u32
    }
}

/// A named uniform block containing multiple [`ShaderParam`] entries.
#[derive(Debug)]
pub struct UniformBlock {
    /// Ordered list of parameters in this block.
    pub params: Vec<ShaderParam>,
    /// Name of the uniform block (e.g. `"Globals"`).
    pub name: String,
}

impl UniformBlock {
    /// Creates a new empty `UniformBlock` with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            params: Vec::new(),
            name: name.into(),
        }
    }

    /// Appends a parameter to the block.
    pub fn add_param(&mut self, param: ShaderParam) {
        self.params.push(param);
    }

    /// Finds the first parameter with the given name.
    #[must_use]
    pub fn find(&self, name: &str) -> Option<&ShaderParam> {
        self.params.iter().find(|p| p.name == name)
    }

    /// Returns the total size in bytes of all parameters in this block.
    ///
    /// This is the sum of each parameter's byte size (not the end offset of
    /// the last parameter, since params may not be tightly packed).
    #[must_use]
    pub fn total_size_bytes(&self) -> u32 {
        self.params
            .iter()
            .map(|p| p.param_type.byte_size() as u32)
            .sum()
    }

    /// Returns the number of parameters in this block.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.params.len()
    }

    /// Returns `true` if the total size is a multiple of 16 (std140 alignment).
    #[must_use]
    pub fn is_aligned(&self) -> bool {
        self.total_size_bytes() % 16 == 0
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_byte_size() {
        assert_eq!(ParamType::Float.byte_size(), 4);
    }

    #[test]
    fn test_int_byte_size() {
        assert_eq!(ParamType::Int.byte_size(), 4);
    }

    #[test]
    fn test_vec2_byte_size() {
        assert_eq!(ParamType::Vec2.byte_size(), 8);
    }

    #[test]
    fn test_vec3_byte_size() {
        assert_eq!(ParamType::Vec3.byte_size(), 12);
    }

    #[test]
    fn test_vec4_byte_size() {
        assert_eq!(ParamType::Vec4.byte_size(), 16);
    }

    #[test]
    fn test_matrix4x4_byte_size() {
        assert_eq!(ParamType::Matrix4x4.byte_size(), 64);
    }

    #[test]
    fn test_texture_byte_size() {
        assert_eq!(ParamType::Texture.byte_size(), 8);
    }

    #[test]
    fn test_shader_param_end_offset() {
        let p = ShaderParam::new("brightness", ParamType::Float, 0);
        assert_eq!(p.end_offset(), 4);
    }

    #[test]
    fn test_shader_param_end_offset_vec4() {
        let p = ShaderParam::new("color", ParamType::Vec4, 16);
        assert_eq!(p.end_offset(), 32);
    }

    #[test]
    fn test_shader_param_end_offset_matrix() {
        let p = ShaderParam::new("mvp", ParamType::Matrix4x4, 64);
        assert_eq!(p.end_offset(), 128);
    }

    #[test]
    fn test_uniform_block_add_and_count() {
        let mut block = UniformBlock::new("Globals");
        block.add_param(ShaderParam::new("time", ParamType::Float, 0));
        block.add_param(ShaderParam::new("resolution", ParamType::Vec2, 4));
        assert_eq!(block.param_count(), 2);
    }

    #[test]
    fn test_uniform_block_find_existing() {
        let mut block = UniformBlock::new("Params");
        block.add_param(ShaderParam::new("gamma", ParamType::Float, 0));
        let found = block.find("gamma");
        assert!(found.is_some());
        assert_eq!(found.expect("parameter should be found").offset, 0);
    }

    #[test]
    fn test_uniform_block_find_missing() {
        let block = UniformBlock::new("Params");
        assert!(block.find("nonexistent").is_none());
    }

    #[test]
    fn test_uniform_block_total_size() {
        // Float(4) + Vec4(16) = 20
        let mut block = UniformBlock::new("Mix");
        block.add_param(ShaderParam::new("alpha", ParamType::Float, 0));
        block.add_param(ShaderParam::new("tint", ParamType::Vec4, 4));
        assert_eq!(block.total_size_bytes(), 20);
    }

    #[test]
    fn test_uniform_block_is_aligned_true() {
        // Vec4(16) is divisible by 16.
        let mut block = UniformBlock::new("Aligned");
        block.add_param(ShaderParam::new("v", ParamType::Vec4, 0));
        assert!(block.is_aligned());
    }

    #[test]
    fn test_uniform_block_is_aligned_false() {
        // Float(4) is not divisible by 16.
        let mut block = UniformBlock::new("Unaligned");
        block.add_param(ShaderParam::new("x", ParamType::Float, 0));
        assert!(!block.is_aligned());
    }

    #[test]
    fn test_uniform_block_matrix_is_aligned() {
        // Matrix4x4(64) is divisible by 16.
        let mut block = UniformBlock::new("MVP");
        block.add_param(ShaderParam::new("mvp", ParamType::Matrix4x4, 0));
        assert!(block.is_aligned());
    }
}
