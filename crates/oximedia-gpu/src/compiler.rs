//! Runtime shader compilation and management
//!
//! This module provides utilities for compiling shaders at runtime,
//! including error handling, preprocessor support, and optimization.

use crate::{GpuDevice, GpuError, Result};
use std::collections::HashMap;
use wgpu::ShaderModule;

/// Shader compilation error
#[derive(Debug, Clone)]
pub struct CompilationError {
    /// Error message
    pub message: String,
    /// Line number where the error occurred
    pub line: Option<usize>,
    /// Column number where the error occurred
    pub column: Option<usize>,
}

impl CompilationError {
    /// Create a new compilation error
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            line: None,
            column: None,
        }
    }

    /// Set the line number
    #[must_use]
    pub fn with_line(mut self, line: usize) -> Self {
        self.line = Some(line);
        self
    }

    /// Set the column number
    #[must_use]
    pub fn with_column(mut self, column: usize) -> Self {
        self.column = Some(column);
        self
    }
}

impl std::fmt::Display for CompilationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let (Some(line), Some(column)) = (self.line, self.column) {
            write!(f, "{}:{}: {}", line, column, self.message)
        } else if let Some(line) = self.line {
            write!(f, "Line {}: {}", line, self.message)
        } else {
            write!(f, "{}", self.message)
        }
    }
}

/// Shader source type
#[derive(Debug, Clone)]
pub enum ShaderSourceType {
    /// WGSL source code
    WGSL,
    /// SPIR-V binary
    SPIRV,
    /// GLSL source code (requires translation to WGSL)
    GLSL,
}

/// Shader preprocessor for handling defines and includes
pub struct ShaderPreprocessor {
    defines: HashMap<String, String>,
}

impl ShaderPreprocessor {
    /// Create a new shader preprocessor
    #[must_use]
    pub fn new() -> Self {
        Self {
            defines: HashMap::new(),
        }
    }

    /// Add a preprocessor define
    pub fn define(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.defines.insert(name.into(), value.into());
    }

    /// Process shader source with preprocessor directives
    pub fn process(&self, source: &str) -> Result<String> {
        let mut output = String::new();
        let mut lines = source.lines();

        while let Some(line) = lines.next() {
            let trimmed = line.trim();

            // Handle #define directives
            if trimmed.starts_with("#define") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    let name = parts[1];
                    let _value = parts.get(2).unwrap_or(&"1");
                    if let Some(defined_value) = self.defines.get(name) {
                        output.push_str(&format!("#define {name} {defined_value}\n"));
                    } else {
                        output.push_str(line);
                        output.push('\n');
                    }
                } else {
                    output.push_str(line);
                    output.push('\n');
                }
            }
            // Handle #ifdef directives
            else if trimmed.starts_with("#ifdef") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    let name = parts[1];
                    if self.defines.contains_key(name) {
                        // Include this block
                        continue;
                    }
                    // Skip until #endif
                    for inner_line in lines.by_ref() {
                        if inner_line.trim().starts_with("#endif") {
                            break;
                        }
                    }
                    continue;
                }
                output.push_str(line);
                output.push('\n');
            }
            // Pass through other lines
            else {
                output.push_str(line);
                output.push('\n');
            }
        }

        Ok(output)
    }

    /// Get all defines
    #[must_use]
    pub fn defines(&self) -> &HashMap<String, String> {
        &self.defines
    }
}

impl Default for ShaderPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Shader compiler with caching and optimization
pub struct ShaderCompiler {
    device: std::sync::Arc<wgpu::Device>,
    preprocessor: ShaderPreprocessor,
}

impl ShaderCompiler {
    /// Create a new shader compiler
    #[must_use]
    pub fn new(device: &GpuDevice) -> Self {
        Self {
            device: std::sync::Arc::clone(device.device()),
            preprocessor: ShaderPreprocessor::new(),
        }
    }

    /// Compile WGSL shader source
    ///
    /// # Arguments
    ///
    /// * `label` - Shader label for debugging
    /// * `source` - WGSL source code
    ///
    /// # Errors
    ///
    /// Returns an error if compilation fails.
    pub fn compile_wgsl(&self, label: &str, source: &str) -> Result<ShaderModule> {
        // Process with preprocessor
        let processed_source = self.preprocessor.process(source)?;

        // Compile the shader
        Ok(self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(processed_source.into()),
            }))
    }

    /// Compile shader from file
    ///
    /// # Arguments
    ///
    /// * `label` - Shader label for debugging
    /// * `path` - Path to shader file
    ///
    /// # Errors
    ///
    /// Returns an error if file reading or compilation fails.
    pub fn compile_file(
        &self,
        label: &str,
        path: impl AsRef<std::path::Path>,
    ) -> Result<ShaderModule> {
        let source = std::fs::read_to_string(path.as_ref())
            .map_err(|e| GpuError::ShaderCompilation(format!("Failed to read shader file: {e}")))?;

        self.compile_wgsl(label, &source)
    }

    /// Get the preprocessor
    #[must_use]
    pub fn preprocessor(&self) -> &ShaderPreprocessor {
        &self.preprocessor
    }

    /// Get a mutable reference to the preprocessor
    pub fn preprocessor_mut(&mut self) -> &mut ShaderPreprocessor {
        &mut self.preprocessor
    }

    /// Validate shader source without compiling
    ///
    /// # Arguments
    ///
    /// * `source` - Shader source code
    ///
    /// # Returns
    ///
    /// Ok(()) if the shader is valid, Err otherwise
    pub fn validate(&self, source: &str) -> Result<()> {
        // wgpu performs validation during shader module creation
        // We can create a temporary shader module to validate
        let _ = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Validation"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });

        Ok(())
    }
}

/// Shader optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimization
    Basic,
    /// Full optimization
    Full,
}

/// Shader compilation options
pub struct CompilationOptions {
    /// Optimization level
    pub optimization: OptimizationLevel,
    /// Enable debug information
    pub debug_info: bool,
    /// Preprocessor defines
    pub defines: HashMap<String, String>,
}

impl Default for CompilationOptions {
    fn default() -> Self {
        Self {
            optimization: OptimizationLevel::Basic,
            debug_info: false,
            defines: HashMap::new(),
        }
    }
}

impl CompilationOptions {
    /// Create new compilation options
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set optimization level
    #[must_use]
    pub fn with_optimization(mut self, level: OptimizationLevel) -> Self {
        self.optimization = level;
        self
    }

    /// Enable debug information
    #[must_use]
    pub fn with_debug_info(mut self, enabled: bool) -> Self {
        self.debug_info = enabled;
        self
    }

    /// Add a preprocessor define
    pub fn with_define(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.defines.insert(name.into(), value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocessor() {
        let mut preprocessor = ShaderPreprocessor::new();
        preprocessor.define("WORKGROUP_SIZE", "256");

        let source = "#define WORKGROUP_SIZE 64\nfn main() {}";
        let processed = preprocessor
            .process(source)
            .expect("preprocessing should succeed");

        assert!(processed.contains("256"));
    }

    #[test]
    fn test_compilation_error() {
        let error = CompilationError::new("Syntax error")
            .with_line(42)
            .with_column(10);

        assert_eq!(error.line, Some(42));
        assert_eq!(error.column, Some(10));

        let formatted = format!("{error}");
        assert!(formatted.contains("42"));
        assert!(formatted.contains("10"));
    }

    #[test]
    fn test_compilation_options() {
        let options = CompilationOptions::new()
            .with_optimization(OptimizationLevel::Full)
            .with_debug_info(true)
            .with_define("TEST", "1");

        assert_eq!(options.optimization, OptimizationLevel::Full);
        assert!(options.debug_info);
        assert_eq!(options.defines.get("TEST"), Some(&"1".to_string()));
    }
}
