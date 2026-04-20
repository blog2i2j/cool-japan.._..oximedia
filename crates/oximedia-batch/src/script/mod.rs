//! Lua scripting support for custom job logic

use crate::error::{BatchError, Result};
use mlua::{Lua, Table, Value};
use std::path::Path;

/// Lua script executor
pub struct ScriptExecutor {
    lua: Lua,
}

impl ScriptExecutor {
    /// Create a new script executor
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails
    pub fn new() -> Result<Self> {
        let lua = Lua::new();

        // Load standard libraries
        lua.load_std_libs(mlua::StdLib::ALL_SAFE)
            .map_err(|e| BatchError::ScriptError(e.to_string()))?;

        Ok(Self { lua })
    }

    /// Execute a Lua script file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the script file
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails
    pub fn execute_file(&self, path: &Path) -> Result<Value> {
        let script = std::fs::read_to_string(path)?;
        self.execute(&script)
    }

    /// Execute a Lua script string
    ///
    /// # Arguments
    ///
    /// * `script` - Script source code
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails
    pub fn execute(&self, script: &str) -> Result<Value> {
        self.lua
            .load(script)
            .eval()
            .map_err(|e| BatchError::ScriptError(e.to_string()))
    }

    /// Evaluate a boolean expression
    ///
    /// # Arguments
    ///
    /// * `expression` - Boolean expression
    ///
    /// # Errors
    ///
    /// Returns an error if evaluation fails
    pub fn evaluate_bool(&self, expression: &str) -> Result<bool> {
        let result = self.execute(expression)?;

        match result {
            Value::Boolean(b) => Ok(b),
            _ => Err(BatchError::ScriptError(
                "Expression did not evaluate to boolean".to_string(),
            )),
        }
    }

    /// Set a global variable
    ///
    /// # Arguments
    ///
    /// * `name` - Variable name
    /// * `value` - Variable value
    ///
    /// # Errors
    ///
    /// Returns an error if setting fails
    pub fn set_global(&self, name: &str, value: Value) -> Result<()> {
        self.lua
            .globals()
            .set(name, value)
            .map_err(|e| BatchError::ScriptError(e.to_string()))
    }

    /// Get a global variable
    ///
    /// # Arguments
    ///
    /// * `name` - Variable name
    ///
    /// # Errors
    ///
    /// Returns an error if getting fails
    pub fn get_global(&self, name: &str) -> Result<Value> {
        self.lua
            .globals()
            .get(name)
            .map_err(|e| BatchError::ScriptError(e.to_string()))
    }

    /// Call a Lua function
    ///
    /// # Arguments
    ///
    /// * `func_name` - Function name
    /// * `args` - Function arguments
    ///
    /// # Errors
    ///
    /// Returns an error if calling fails
    pub fn call_function(&self, func_name: &str, args: &[Value]) -> Result<Value> {
        let globals = self.lua.globals();
        let func: mlua::Function = globals
            .get(func_name)
            .map_err(|e| BatchError::ScriptError(e.to_string()))?;

        match args.len() {
            0 => func
                .call(())
                .map_err(|e| BatchError::ScriptError(e.to_string())),
            1 => func
                .call(args[0].clone())
                .map_err(|e| BatchError::ScriptError(e.to_string())),
            _ => {
                let table = self.lua.create_table()?;
                for (i, arg) in args.iter().enumerate() {
                    table.set(i + 1, arg.clone())?;
                }
                func.call::<Value>(mlua::Value::Table(table))
                    .map_err(|e| BatchError::ScriptError(e.to_string()))
            }
        }
    }

    /// Create a context table for scripting
    ///
    /// # Errors
    ///
    /// Returns an error if creation fails
    pub fn create_context(&self) -> Result<Table> {
        self.lua
            .create_table()
            .map_err(|e| BatchError::ScriptError(e.to_string()))
    }
}

impl Default for ScriptExecutor {
    fn default() -> Self {
        // Lua::new() is infallible; load_std_libs uses ALL_SAFE which never fails
        // in practice. If it does fail, we fall back to a bare Lua instance.
        let lua = Lua::new();
        let _ = lua.load_std_libs(mlua::StdLib::ALL_SAFE);
        Self { lua }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_script_executor_creation() {
        let result = ScriptExecutor::new();
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_simple_script() {
        let executor = ScriptExecutor::new().expect("failed to create");
        let result = executor.execute("return 42");

        assert!(result.is_ok());
        let is_integer_42 = matches!(result.expect("result should be valid"), Value::Integer(42));
        assert!(is_integer_42);
    }

    #[test]
    fn test_execute_boolean_expression() {
        let executor = ScriptExecutor::new().expect("failed to create");
        let result = executor.execute("return 10 > 5");

        assert!(result.is_ok());
        let is_true = matches!(
            result.expect("result should be valid"),
            Value::Boolean(true)
        );
        assert!(is_true);
    }

    #[test]
    fn test_evaluate_bool() {
        let executor = ScriptExecutor::new().expect("failed to create");
        let result = executor.evaluate_bool("return true");

        assert!(result.is_ok());
        assert!(result.expect("result should be valid"));
    }

    #[test]
    fn test_set_and_get_global() {
        let executor = ScriptExecutor::new().expect("failed to create");

        executor
            .set_global(
                "test_var",
                Value::String(
                    executor
                        .lua
                        .create_string("hello")
                        .expect("create_string should succeed"),
                ),
            )
            .expect("operation should succeed");

        let is_hello = {
            let value = executor
                .get_global("test_var")
                .expect("get_global should succeed");
            matches!(value, Value::String(ref s) if s.to_str().expect("path should be valid UTF-8") == "hello")
        };
        assert!(is_hello);
    }

    #[test]
    fn test_call_function() {
        let executor = ScriptExecutor::new().expect("failed to create");

        // Define a function
        executor
            .execute("function double(x) return x * 2 end")
            .expect("operation should succeed");

        // Call the function
        let result = executor.call_function("double", &[Value::Integer(21)]);

        assert!(result.is_ok());
        let is_42 = matches!(result.expect("result should be valid"), Value::Integer(42));
        assert!(is_42);
    }

    #[test]
    fn test_create_context() {
        let executor = ScriptExecutor::new().expect("failed to create");
        let result = executor.create_context();

        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_script() {
        let executor = ScriptExecutor::new().expect("failed to create");
        let result = executor.execute("invalid lua code @#$");

        assert!(result.is_err());
    }
}
