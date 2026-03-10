//! Lua scripting engine for custom automation workflows.

use crate::{AutomationError, Result};
use mlua::{Lua, Table, Value};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, error, info};

/// Script execution context.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScriptContext {
    /// Context variables
    pub variables: HashMap<String, String>,
    /// Channel ID
    pub channel_id: Option<usize>,
}

impl ScriptContext {
    /// Create a new script context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a variable.
    pub fn set_variable(&mut self, key: String, value: String) {
        self.variables.insert(key, value);
    }

    /// Get a variable.
    pub fn get_variable(&self, key: &str) -> Option<&String> {
        self.variables.get(key)
    }

    /// Set channel ID.
    pub fn with_channel(mut self, channel_id: usize) -> Self {
        self.channel_id = Some(channel_id);
        self
    }
}

/// Lua scripting engine.
pub struct ScriptEngine {
    lua: Lua,
    context: ScriptContext,
}

impl ScriptEngine {
    /// Create a new script engine.
    pub fn new() -> Result<Self> {
        info!("Creating Lua scripting engine");

        let lua = Lua::new();

        Ok(Self {
            lua,
            context: ScriptContext::default(),
        })
    }

    /// Create with context.
    pub fn with_context(context: ScriptContext) -> Result<Self> {
        let mut engine = Self::new()?;
        engine.context = context;
        Ok(engine)
    }

    /// Load automation API into Lua.
    pub fn load_api(&self) -> Result<()> {
        debug!("Loading automation API into Lua");

        self.lua
            .globals()
            .set("automation", self.create_api_table()?)
            .map_err(|e| AutomationError::Scripting(format!("Failed to load API: {e}")))?;

        Ok(())
    }

    /// Create automation API table.
    fn create_api_table(&self) -> Result<Table<'_>> {
        let table = self
            .lua
            .create_table()
            .map_err(|e| AutomationError::Scripting(format!("Failed to create table: {e}")))?;

        // Add API functions
        let log_fn = self
            .lua
            .create_function(|_, message: String| {
                info!("Script log: {}", message);
                Ok(())
            })
            .map_err(|e| AutomationError::Scripting(format!("Failed to create function: {e}")))?;

        table
            .set("log", log_fn)
            .map_err(|e| AutomationError::Scripting(format!("Failed to set function: {e}")))?;

        Ok(table)
    }

    /// Execute Lua script.
    pub fn execute<'lua>(&'lua self, script: &str) -> Result<Value<'lua>> {
        debug!("Executing Lua script");

        self.lua.load(script).eval().map_err(|e| {
            error!("Script execution error: {}", e);
            AutomationError::Scripting(format!("Execution error: {e}"))
        })
    }

    /// Execute script file.
    pub fn execute_file<'lua>(&'lua self, path: &str) -> Result<Value<'lua>> {
        info!("Executing Lua script file: {}", path);

        let script = std::fs::read_to_string(path)
            .map_err(|e| AutomationError::Scripting(format!("Failed to read script: {e}")))?;

        self.execute(&script)
    }

    /// Get context.
    pub fn context(&self) -> &ScriptContext {
        &self.context
    }

    /// Set context variable.
    pub fn set_variable(&mut self, key: String, value: String) -> Result<()> {
        self.context.set_variable(key.clone(), value.clone());

        // Also set in Lua globals
        self.lua
            .globals()
            .set(key, value)
            .map_err(|e| AutomationError::Scripting(format!("Failed to set variable: {e}")))?;

        Ok(())
    }

    /// Get Lua global variable.
    pub fn get_global<'lua>(&'lua self, key: &str) -> Result<Value<'lua>> {
        self.lua
            .globals()
            .get(key)
            .map_err(|e| AutomationError::Scripting(format!("Failed to get global: {e}")))
    }

    /// Call Lua function.
    pub fn call_function<'lua>(&'lua self, name: &str, args: Vec<Value<'lua>>) -> Result<String> {
        debug!("Calling Lua function: {}", name);

        let func: mlua::Function = self
            .lua
            .globals()
            .get(name)
            .map_err(|e| AutomationError::Scripting(format!("Function not found: {e}")))?;

        let result: mlua::Value = func
            .call(mlua::MultiValue::from_vec(args))
            .map_err(|e| AutomationError::Scripting(format!("Function call error: {e}")))?;

        // Convert result to string
        Ok(format!("{result:?}"))
    }
}

impl Default for ScriptEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create script engine")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_script_engine_creation() {
        let engine = ScriptEngine::new();
        assert!(engine.is_ok());
    }

    #[test]
    fn test_execute_simple_script() {
        let engine = ScriptEngine::new().expect("new should succeed");
        let result = engine.execute("return 1 + 1");
        assert!(result.is_ok());
    }

    #[test]
    fn test_set_variable() {
        let mut engine = ScriptEngine::new().expect("new should succeed");
        engine
            .set_variable("test".to_string(), "value".to_string())
            .expect("operation should succeed");

        let value = engine
            .get_global("test")
            .expect("get_global should succeed");
        assert!(matches!(value, Value::String(_)));
    }

    #[test]
    fn test_script_context() {
        let mut context = ScriptContext::new();
        context.set_variable("key".to_string(), "value".to_string());

        assert_eq!(context.get_variable("key"), Some(&"value".to_string()));
    }
}
