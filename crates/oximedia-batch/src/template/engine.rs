//! Template rendering engine

use crate::error::{BatchError, Result};
use crate::template::functions::TemplateFunctions;
use crate::template::TemplateContext;
use regex::Regex;

/// Template engine for processing templates
pub struct TemplateEngine {
    functions: TemplateFunctions,
}

impl TemplateEngine {
    /// Create a new template engine
    #[must_use]
    pub fn new() -> Self {
        Self {
            functions: TemplateFunctions::new(),
        }
    }

    /// Render a template with context
    ///
    /// # Arguments
    ///
    /// * `template` - Template string
    /// * `context` - Template context
    ///
    /// # Errors
    ///
    /// Returns an error if rendering fails
    pub fn render(&self, template: &str, context: &TemplateContext) -> Result<String> {
        let mut result = template.to_string();

        // Process simple variable substitution: {variable}
        result = Self::substitute_variables(&result, context)?;

        // Process function calls: {function(args)}
        result = self.process_functions(&result, context)?;

        // Process conditionals: {if condition}...{else}...{endif}
        result = Self::process_conditionals(&result, context)?;

        Ok(result)
    }

    fn substitute_variables(template: &str, context: &TemplateContext) -> Result<String> {
        let re = Regex::new(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
            .map_err(|e| BatchError::TemplateError(e.to_string()))?;

        let mut result = template.to_string();

        for cap in re.captures_iter(template) {
            if let Some(var_name) = cap.get(1) {
                let var_name_str = var_name.as_str();
                if let Some(value) = context.get(var_name_str) {
                    let pattern = format!("{{{var_name_str}}}");
                    result = result.replace(&pattern, value);
                } else {
                    // Variable not found, leave as-is or error
                    tracing::warn!("Variable not found: {}", var_name_str);
                }
            }
        }

        Ok(result)
    }

    fn process_functions(&self, template: &str, context: &TemplateContext) -> Result<String> {
        let re = Regex::new(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]*)\)\}")
            .map_err(|e| BatchError::TemplateError(e.to_string()))?;

        let mut result = template.to_string();

        for cap in re.captures_iter(template) {
            if let (Some(func_name), Some(args)) = (cap.get(1), cap.get(2)) {
                let func_name_str = func_name.as_str();
                let args_str = args.as_str();

                let output = self.functions.call(func_name_str, args_str, context)?;

                let pattern = format!("{{{func_name_str}({args_str})}}");
                result = result.replace(&pattern, &output);
            }
        }

        Ok(result)
    }

    #[allow(clippy::unnecessary_wraps)]
    fn process_conditionals(template: &str, context: &TemplateContext) -> Result<String> {
        // Process {if condition}...{else}...{endif} blocks
        // Using iterative approach to handle nested conditionals isn't needed - just handle one level

        let mut result = template.to_string();

        // Pattern: {if condition}...{else}...{endif} or {if condition}...{endif}
        // We process from innermost (no nested ifs) outward
        while let Some(if_start) = result.find("{if ") {
            // Find matching {endif}
            let Some(endif_offset) = result[if_start..].find("{endif}") else {
                break; // malformed template
            };
            let endif_pos = if_start + endif_offset;

            // Extract the condition
            let cond_start = if_start + 4; // skip "{if "
            let Some(cond_offset) = result[cond_start..].find('}') else {
                break;
            };
            let cond_end = cond_start + cond_offset;
            let condition = result[cond_start..cond_end].trim().to_string();

            // Find the content between {if ...} and {endif}
            let content_start = cond_end + 1;
            let block_content = &result[content_start..endif_pos];

            // Split on {else} if present
            let (true_content, false_content) = if let Some(else_pos) = block_content.find("{else}")
            {
                (&block_content[..else_pos], &block_content[else_pos + 6..])
            } else {
                (block_content, "")
            };

            // Evaluate condition
            let condition_true = Self::evaluate_condition(&condition, context);

            // Replace the entire {if ...}...{endif} block
            let replacement = if condition_true {
                true_content
            } else {
                false_content
            };
            let full_block = &result[if_start..endif_pos + 7]; // +7 for "{endif}"
            let result_new = result.replacen(full_block, replacement, 1);
            result = result_new;
        }

        Ok(result)
    }

    fn evaluate_condition(condition: &str, context: &TemplateContext) -> bool {
        let condition = condition.trim();

        // Negation: !varname
        if let Some(var) = condition.strip_prefix('!') {
            let val = context
                .get(var.trim())
                .map_or("", std::string::String::as_str);
            return val.is_empty() || val == "false" || val == "0";
        }

        // Equality: varname==value
        if let Some(eq_pos) = condition.find("==") {
            let var_name = condition[..eq_pos].trim();
            let expected = condition[eq_pos + 2..].trim().trim_matches('"');
            let actual = context
                .get(var_name)
                .map_or("", std::string::String::as_str);
            return actual == expected;
        }

        // Inequality: varname!=value
        if let Some(ne_pos) = condition.find("!=") {
            let var_name = condition[..ne_pos].trim();
            let expected = condition[ne_pos + 2..].trim().trim_matches('"');
            let actual = context
                .get(var_name)
                .map_or("", std::string::String::as_str);
            return actual != expected;
        }

        // Simple truth check: varname (truthy if non-empty, non-zero, non-"false")
        let val = context
            .get(condition)
            .map_or("", std::string::String::as_str);
        !val.is_empty() && val != "false" && val != "0"
    }
}

impl Default for TemplateEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = TemplateEngine::new();
        let _ = engine; // engine created successfully
    }

    #[test]
    fn test_simple_variable_substitution() {
        let engine = TemplateEngine::new();
        let mut context = TemplateContext::new();
        context.set("filename".to_string(), "test.mp4".to_string());

        let result = engine.render("Output: {filename}", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "Output: test.mp4");
    }

    #[test]
    fn test_multiple_variables() {
        let engine = TemplateEngine::new();
        let mut context = TemplateContext::new();
        context.set("name".to_string(), "video".to_string());
        context.set("ext".to_string(), "mp4".to_string());

        let result = engine.render("{name}.{ext}", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "video.mp4");
    }

    #[test]
    fn test_missing_variable() {
        let engine = TemplateEngine::new();
        let context = TemplateContext::new();

        let result = engine.render("{missing}", &context);
        assert!(result.is_ok());
        // Missing variables are left as-is
        assert_eq!(result.expect("result should be valid"), "{missing}");
    }

    #[test]
    fn test_conditional_true() {
        let engine = TemplateEngine::new();
        let mut context = TemplateContext::new();
        context.set("enabled".to_string(), "true".to_string());

        let result = engine.render("{if enabled}yes{endif}", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "yes");
    }

    #[test]
    fn test_conditional_false_missing() {
        let engine = TemplateEngine::new();
        let context = TemplateContext::new();

        let result = engine.render("{if enabled}yes{endif}", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "");
    }

    #[test]
    fn test_conditional_else() {
        let engine = TemplateEngine::new();
        let mut context = TemplateContext::new();
        context.set("enabled".to_string(), "false".to_string());

        let result = engine.render("{if enabled}yes{else}no{endif}", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "no");
    }

    #[test]
    fn test_conditional_negation() {
        let engine = TemplateEngine::new();
        let context = TemplateContext::new();

        let result = engine.render("{if !missing}not set{endif}", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "not set");
    }

    #[test]
    fn test_conditional_equality() {
        let engine = TemplateEngine::new();
        let mut context = TemplateContext::new();
        context.set("codec".to_string(), "h264".to_string());

        let result = engine.render("{if codec==h264}mp4{else}other{endif}", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "mp4");
    }

    #[test]
    fn test_conditional_inequality() {
        let engine = TemplateEngine::new();
        let mut context = TemplateContext::new();
        context.set("codec".to_string(), "vp9".to_string());

        let result = engine.render("{if codec!=h264}not-h264{endif}", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "not-h264");
    }

    #[test]
    fn test_conditional_zero_is_falsy() {
        let engine = TemplateEngine::new();
        let mut context = TemplateContext::new();
        context.set("count".to_string(), "0".to_string());

        let result = engine.render("{if count}has items{else}empty{endif}", &context);
        assert!(result.is_ok());
        assert_eq!(result.expect("result should be valid"), "empty");
    }
}
