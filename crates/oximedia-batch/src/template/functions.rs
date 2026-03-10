//! Template functions for data transformation

use crate::error::{BatchError, Result};
use crate::template::TemplateContext;

/// Built-in template functions
pub struct TemplateFunctions;

impl TemplateFunctions {
    /// Create a new template functions registry
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Call a template function
    ///
    /// # Arguments
    ///
    /// * `name` - Function name
    /// * `args` - Function arguments
    /// * `context` - Template context
    ///
    /// # Errors
    ///
    /// Returns an error if the function fails
    pub fn call(&self, name: &str, args: &str, context: &TemplateContext) -> Result<String> {
        match name {
            "uppercase" => Self::uppercase(args, context),
            "lowercase" => Self::lowercase(args, context),
            "replace" => Self::replace(args, context),
            "trim" => Self::trim(args, context),
            "substr" => Self::substr(args, context),
            "pad" => Self::pad(args, context),
            "format_number" => Self::format_number(args, context),
            "date" => Self::date(args, context),
            _ => Err(BatchError::TemplateError(format!(
                "Unknown function: {name}"
            ))),
        }
    }

    fn uppercase(args: &str, context: &TemplateContext) -> Result<String> {
        let value = Self::resolve_arg(args, context)?;
        Ok(value.to_uppercase())
    }

    fn lowercase(args: &str, context: &TemplateContext) -> Result<String> {
        let value = Self::resolve_arg(args, context)?;
        Ok(value.to_lowercase())
    }

    fn replace(args: &str, context: &TemplateContext) -> Result<String> {
        let parts: Vec<&str> = args.split(',').collect();
        if parts.len() != 3 {
            return Err(BatchError::TemplateError(
                "replace() requires 3 arguments".to_string(),
            ));
        }

        let value = Self::resolve_arg(parts[0].trim(), context)?;
        let from = parts[1].trim().trim_matches('"');
        let to = parts[2].trim().trim_matches('"');

        Ok(value.replace(from, to))
    }

    fn trim(args: &str, context: &TemplateContext) -> Result<String> {
        let value = Self::resolve_arg(args, context)?;
        Ok(value.trim().to_string())
    }

    fn substr(args: &str, context: &TemplateContext) -> Result<String> {
        let parts: Vec<&str> = args.split(',').collect();
        if parts.len() < 2 {
            return Err(BatchError::TemplateError(
                "substr() requires at least 2 arguments".to_string(),
            ));
        }

        let value = Self::resolve_arg(parts[0].trim(), context)?;
        let start: usize = parts[1]
            .trim()
            .parse()
            .map_err(|e| BatchError::TemplateError(format!("Invalid start index: {e}")))?;

        if parts.len() == 3 {
            let length: usize = parts[2]
                .trim()
                .parse()
                .map_err(|e| BatchError::TemplateError(format!("Invalid length: {e}")))?;
            Ok(value.chars().skip(start).take(length).collect())
        } else {
            Ok(value.chars().skip(start).collect())
        }
    }

    fn pad(args: &str, context: &TemplateContext) -> Result<String> {
        let parts: Vec<&str> = args.split(',').collect();
        if parts.len() != 2 {
            return Err(BatchError::TemplateError(
                "pad() requires 2 arguments".to_string(),
            ));
        }

        let value = Self::resolve_arg(parts[0].trim(), context)?;
        let width: usize = parts[1]
            .trim()
            .parse()
            .map_err(|e| BatchError::TemplateError(format!("Invalid width: {e}")))?;

        Ok(format!("{value:0>width$}"))
    }

    fn format_number(args: &str, context: &TemplateContext) -> Result<String> {
        let value = Self::resolve_arg(args, context)?;
        let num: u64 = value
            .parse()
            .map_err(|e| BatchError::TemplateError(format!("Invalid number: {e}")))?;

        // Format with thousands separator
        let formatted = num
            .to_string()
            .as_bytes()
            .rchunks(3)
            .rev()
            .map(std::str::from_utf8)
            .collect::<std::result::Result<Vec<&str>, _>>()
            .map_err(|e| BatchError::TemplateError(format!("Format error: {e}")))?
            .join(",");

        Ok(formatted)
    }

    #[allow(clippy::unnecessary_wraps)]
    fn date(args: &str, _context: &TemplateContext) -> Result<String> {
        let format = args.trim_matches('"');

        let now = chrono::Utc::now();
        let formatted = if format.is_empty() {
            now.format("%Y-%m-%d").to_string()
        } else {
            now.format(format).to_string()
        };

        Ok(formatted)
    }

    #[allow(clippy::unnecessary_wraps)]
    fn resolve_arg(arg: &str, context: &TemplateContext) -> Result<String> {
        let trimmed = arg.trim();

        // Check if it's a string literal
        if trimmed.starts_with('"') && trimmed.ends_with('"') {
            return Ok(trimmed.trim_matches('"').to_string());
        }

        // Check if it's a variable reference
        if let Some(value) = context.get(trimmed) {
            return Ok(value.clone());
        }

        // Return as-is if not found
        Ok(trimmed.to_string())
    }
}

impl Default for TemplateFunctions {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uppercase() {
        let funcs = TemplateFunctions::new();
        let mut context = TemplateContext::new();
        context.set("name".to_string(), "test".to_string());

        let result = funcs.call("uppercase", "name", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "TEST");
    }

    #[test]
    fn test_lowercase() {
        let funcs = TemplateFunctions::new();
        let mut context = TemplateContext::new();
        context.set("name".to_string(), "TEST".to_string());

        let result = funcs.call("lowercase", "name", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "test");
    }

    #[test]
    fn test_replace() {
        let funcs = TemplateFunctions::new();
        let mut context = TemplateContext::new();
        context.set("codec".to_string(), "h264".to_string());

        let result = funcs.call("replace", "codec, \"h264\", \"hevc\"", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "hevc");
    }

    #[test]
    fn test_trim() {
        let funcs = TemplateFunctions::new();
        let mut context = TemplateContext::new();
        context.set("name".to_string(), "  test  ".to_string());

        let result = funcs.call("trim", "name", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "test");
    }

    #[test]
    fn test_substr() {
        let funcs = TemplateFunctions::new();
        let mut context = TemplateContext::new();
        context.set("text".to_string(), "hello world".to_string());

        let result = funcs.call("substr", "text, 0, 5", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "hello");
    }

    #[test]
    fn test_pad() {
        let funcs = TemplateFunctions::new();
        let mut context = TemplateContext::new();
        context.set("num".to_string(), "42".to_string());

        let result = funcs.call("pad", "num, 5", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "00042");
    }

    #[test]
    fn test_date() {
        let funcs = TemplateFunctions::new();
        let context = TemplateContext::new();

        let result = funcs.call("date", "\"\"", &context);
        assert!(result.is_ok());
        // Should return current date in YYYY-MM-DD format
        let date_str = result.expect("result should be valid");
        assert!(date_str.len() >= 10);
    }

    #[test]
    fn test_unknown_function() {
        let funcs = TemplateFunctions::new();
        let context = TemplateContext::new();

        let result = funcs.call("unknown", "", &context);
        assert!(result.is_err());
    }
}
