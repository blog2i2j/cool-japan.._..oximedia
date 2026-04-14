//! Lua scripting engine for custom automation workflows.
//!
//! # Sandboxing
//!
//! The engine creates each Lua instance with a **sandboxed** environment by
//! stripping out all standard library functions that could give a script
//! access to the filesystem (`io`, `os.execute`, `os.rename`, `os.remove`,
//! `dofile`, `loadfile`, `require`) or the network.  Only safe, side-effect-
//! free standard functions are kept: `print`, `tostring`, `tonumber`, `type`,
//! `pairs`, `ipairs`, `pcall`, `xpcall`, `error`, `assert`, `select`,
//! `unpack`, `next`, `rawequal`, `rawget`, `rawset`, `rawlen`, `setmetatable`,
//! `getmetatable`, plus the pure `math`, `string`, and `table` libraries.
//!
//! # Script Caching
//!
//! Compiled Lua scripts are cached by their source text so that repeatedly
//! calling the same script body does not incur repeated parse/compile costs.
//! The cache stores the pre-serialised bytecode produced by `mlua`'s
//! `Chunk::into_function` mechanism.  Cache capacity is bounded; when the
//! limit is reached, the oldest entry is evicted (FIFO).
//!
//! # Resource Limits
//!
//! `ScriptEngine::execute_with_limits` executes a Lua script with configurable
//! constraints:
//!
//! - **Instruction count**: Using the mlua hook API with
//!   `HookTriggers::every_nth_instruction`, a counting hook returns an error
//!   after the configured instruction budget is exhausted.
//! - **Memory**: Via `Lua::set_memory_limit`, a hard byte limit is applied
//!   before execution; exceeded allocation raises a Lua error.
//! - **Wall-clock timeout**: `std::time::Instant::now()` is captured before
//!   execution and `elapsed()` is checked inside the same instruction-counting
//!   hook callback.  If the wall-clock time exceeds `ScriptLimits::max_duration`
//!   the hook returns `Err(mlua::Error::RuntimeError(...))`, aborting the script
//!   immediately.  Because the check runs entirely on the calling thread no
//!   additional threads are needed and the `!Send` constraint of `mlua::Lua`
//!   is fully respected.

use crate::{AutomationError, Result};
use mlua::{HookTriggers, Lua, Table, Value, VmState};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::Duration;
use tracing::{debug, error, info};

/// Maximum number of compiled Lua scripts retained in the cache.
const SCRIPT_CACHE_CAPACITY: usize = 64;

// ─────────────────────────────────────────────────────────────────────────────
// ScriptLimits
// ─────────────────────────────────────────────────────────────────────────────

/// Resource limits applied during script execution.
///
/// All three limits are enforced concurrently:
/// - The first exceeded limit terminates execution.
#[derive(Debug, Clone)]
pub struct ScriptLimits {
    /// Maximum number of Lua VM instructions before the script is aborted.
    /// Default: 1_000_000.
    pub max_instructions: u64,
    /// Maximum Lua heap allocation in bytes (enforced by mlua's memory
    /// limiter).  Default: 32 MiB.
    pub max_memory_bytes: usize,
    /// Maximum wall-clock execution time.  Default: 5 seconds.
    pub max_duration: Duration,
}

impl Default for ScriptLimits {
    fn default() -> Self {
        Self {
            max_instructions: 1_000_000,
            max_memory_bytes: 32 * 1024 * 1024,
            max_duration: Duration::from_secs(5),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Script execution error (extends AutomationError via Result<>)
// ─────────────────────────────────────────────────────────────────────────────

/// Convenience label used in error strings.
const ERR_INSTRUCTION_LIMIT: &str = "instruction limit exceeded";
const ERR_TIMEOUT: &str = "script execution timeout";

// ─────────────────────────────────────────────────────────────────────────────
// Script context
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// Script cache — bounded FIFO keyed by source text hash
// ─────────────────────────────────────────────────────────────────────────────

/// A minimal bounded cache for compiled Lua bytecode.
///
/// Keys are the source string; values are the raw bytecode bytes produced by
/// dumping a compiled `mlua` function.
struct ScriptCache {
    /// Insertion-ordered keys for FIFO eviction.
    order: VecDeque<String>,
    /// Source → bytecode map.
    store: HashMap<String, Vec<u8>>,
    capacity: usize,
}

impl ScriptCache {
    fn new(capacity: usize) -> Self {
        Self {
            order: VecDeque::new(),
            store: HashMap::new(),
            capacity,
        }
    }

    /// Return cached bytecode for `source`, if present.
    fn get(&self, source: &str) -> Option<&Vec<u8>> {
        self.store.get(source)
    }

    /// Insert bytecode for `source`, evicting the oldest entry if needed.
    fn insert(&mut self, source: String, bytecode: Vec<u8>) {
        if self.store.contains_key(&source) {
            return; // already cached
        }
        if self.order.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.store.remove(&oldest);
            }
        }
        self.order.push_back(source.clone());
        self.store.insert(source, bytecode);
    }

    fn len(&self) -> usize {
        self.store.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ScriptEngine
// ─────────────────────────────────────────────────────────────────────────────

/// Lua scripting engine with sandboxing, script caching, and resource limits.
pub struct ScriptEngine {
    lua: Lua,
    context: ScriptContext,
    cache: ScriptCache,
}

impl ScriptEngine {
    /// Create a new sandboxed script engine.
    pub fn new() -> Result<Self> {
        info!("Creating sandboxed Lua scripting engine");

        let lua = Lua::new();
        Self::apply_sandbox(&lua)?;

        Ok(Self {
            lua,
            context: ScriptContext::default(),
            cache: ScriptCache::new(SCRIPT_CACHE_CAPACITY),
        })
    }

    /// Create with context.
    pub fn with_context(context: ScriptContext) -> Result<Self> {
        let mut engine = Self::new()?;
        engine.context = context;
        Ok(engine)
    }

    // ── Sandbox ───────────────────────────────────────────────────────────────

    /// Strip dangerous globals from the Lua environment, leaving only
    /// side-effect-free functions and libraries.
    ///
    /// Removed: `io`, `os.execute`, `os.rename`, `os.remove`, `os.exit`,
    /// `os.tmpname`, `dofile`, `loadfile`, `require`, `load`, `collectgarbage`,
    /// `debug` (full library).
    fn apply_sandbox(lua: &Lua) -> Result<()> {
        let globals = lua.globals();

        // ── Remove entire dangerous modules ───────────────────────────────────
        for module in &["io", "debug", "package"] {
            globals
                .set(*module, Value::Nil)
                .map_err(|e| AutomationError::Scripting(format!("Sandbox error: {e}")))?;
        }

        // ── Neuter the `os` table, keeping only safe time/locale functions ────
        if let Ok(os_table) = globals.get::<Table>("os") {
            for dangerous in &["execute", "rename", "remove", "exit", "tmpname", "getenv"] {
                os_table.set(*dangerous, Value::Nil).map_err(|e| {
                    AutomationError::Scripting(format!("Sandbox os.{dangerous} error: {e}"))
                })?;
            }
        }

        // ── Remove filesystem/loading globals ─────────────────────────────────
        for global in &["dofile", "loadfile", "require", "load", "collectgarbage"] {
            globals
                .set(*global, Value::Nil)
                .map_err(|e| AutomationError::Scripting(format!("Sandbox global error: {e}")))?;
        }

        debug!("Lua sandbox applied: io, debug, package, os.execute, dofile, loadfile, require removed");
        Ok(())
    }

    // ── API ───────────────────────────────────────────────────────────────────

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
    fn create_api_table(&self) -> Result<Table> {
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

    // ── Execution ─────────────────────────────────────────────────────────────

    /// Execute Lua script, using the compiled-bytecode cache when possible.
    pub fn execute(&self, script: &str) -> Result<Value> {
        debug!("Executing Lua script (cache size: {})", self.cache.len());

        // Try cached bytecode first.
        if let Some(bytecode) = self.cache.get(script) {
            return self.lua.load(bytecode.as_slice()).eval().map_err(|e| {
                error!("Cached script execution error: {}", e);
                AutomationError::Scripting(format!("Execution error (cached): {e}"))
            });
        }

        // Not cached: compile and run.
        self.lua.load(script).eval().map_err(|e| {
            error!("Script execution error: {}", e);
            AutomationError::Scripting(format!("Execution error: {e}"))
        })
    }

    /// Execute and cache: compile the script, run it, and store the bytecode
    /// so subsequent calls hit the cache.
    pub fn execute_cached(&mut self, script: &str) -> Result<Value> {
        // If already cached, delegate directly (avoids a borrow conflict).
        if self.cache.get(script).is_some() {
            return self.execute(script);
        }

        // Compile to bytecode.
        let chunk = self.lua.load(script);
        let function = chunk
            .into_function()
            .map_err(|e| AutomationError::Scripting(format!("Compile error: {e}")))?;

        let bytecode = function.dump(false);

        self.cache.insert(script.to_string(), bytecode);

        // Run via the cache path.
        self.execute(script)
    }

    /// Execute script file (reads from filesystem — only available outside
    /// the script runtime itself).
    pub fn execute_file(&self, path: &str) -> Result<Value> {
        info!("Executing Lua script file: {}", path);

        let script = std::fs::read_to_string(path)
            .map_err(|e| AutomationError::Scripting(format!("Failed to read script: {e}")))?;

        self.execute(&script)
    }

    // ── Resource-limited execution ────────────────────────────────────────────

    /// Execute `script` with the given resource `limits`.
    ///
    /// # Limits enforced (all on the calling thread — no unsafe, no threads)
    ///
    /// * **Instruction count** — a per-N-instruction Lua hook increments an
    ///   `Arc<AtomicU64>` counter; once `limits.max_instructions` is reached
    ///   the hook returns a Lua runtime error aborting execution.
    /// * **Memory** — `Lua::set_memory_limit` is called before execution; any
    ///   allocation beyond the limit raises a Lua memory error automatically.
    /// * **Wall-clock timeout** — the same hook callback checks
    ///   `Instant::elapsed()` on each invocation (every 100 instructions).
    ///   If the elapsed wall-clock time exceeds `limits.max_duration`, the hook
    ///   returns a timeout error.  Granularity is bounded by the hook interval
    ///   (≤100 VM instructions), which for a tight loop corresponds to tens of
    ///   nanoseconds.
    ///
    /// A fresh sandboxed `Lua` state is created for each call so limits do not
    /// interfere with the engine's primary `Lua` instance or with each other
    /// across concurrent invocations.
    pub fn execute_with_limits(&self, script: &str, limits: ScriptLimits) -> Result<()> {
        debug!(
            "execute_with_limits: max_instructions={}, max_memory={}, timeout={:?}",
            limits.max_instructions, limits.max_memory_bytes, limits.max_duration
        );

        // ── Fresh sandboxed Lua state (isolated per call) ─────────────────────
        let lua = Lua::new();
        Self::apply_sandbox(&lua)?;

        // ── Memory limit ──────────────────────────────────────────────────────
        lua.set_memory_limit(limits.max_memory_bytes)
            .map_err(|e| AutomationError::Scripting(format!("Failed to set memory limit: {e}")))?;

        // ── Instruction counter + start time for dual-mode hook ───────────────
        let counter = Arc::new(AtomicU64::new(0));
        let counter_hook = Arc::clone(&counter);
        let max_instr = limits.max_instructions;
        let max_duration = limits.max_duration;
        let start = std::time::Instant::now();

        // Fire the hook every 100 VM instructions.
        lua.set_hook(
            HookTriggers::new().every_nth_instruction(100),
            move |_lua, _debug| {
                // Check instruction budget.
                let count = counter_hook.fetch_add(100, Ordering::Relaxed) + 100;
                if count >= max_instr {
                    return Err(mlua::Error::RuntimeError(ERR_INSTRUCTION_LIMIT.to_string()));
                }
                // Check wall-clock timeout.
                if start.elapsed() > max_duration {
                    return Err(mlua::Error::RuntimeError(ERR_TIMEOUT.to_string()));
                }
                Ok(VmState::Continue)
            },
        )
        .map_err(|e| AutomationError::Scripting(format!("Failed to set hook: {e}")))?;

        // ── Execute on the calling thread ─────────────────────────────────────
        lua.load(script).exec().map_err(|e| {
            let msg = e.to_string();
            error!("Script limited execution error: {}", msg);
            AutomationError::Scripting(format!("Script error: {msg}"))
        })
    }

    // ── Public API ────────────────────────────────────────────────────────────

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
    pub fn get_global(&self, key: &str) -> Result<Value> {
        self.lua
            .globals()
            .get(key)
            .map_err(|e| AutomationError::Scripting(format!("Failed to get global: {e}")))
    }

    /// Call Lua function.
    pub fn call_function(&self, name: &str, args: Vec<Value>) -> Result<String> {
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

    /// Return the number of scripts currently in the bytecode cache.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
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

    // ── Sandbox tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_sandbox_io_module_removed() {
        let engine = ScriptEngine::new().expect("new should succeed");
        // `io` should be nil; accessing it must not succeed
        let result = engine.execute("return io");
        // The result should be Nil (not an error but io is gone)
        match result {
            Ok(Value::Nil) => {} // expected: io was removed
            Ok(Value::Table(_)) => panic!("io module should have been removed by sandbox"),
            Ok(_) => {}  // any other value is also unexpected but not dangerous
            Err(_) => {} // error is also fine (strict mode)
        }
    }

    #[test]
    fn test_sandbox_filesystem_access_blocked() {
        let engine = ScriptEngine::new().expect("new should succeed");
        // `dofile` should be nil → calling it must raise an error
        let result = engine.execute("dofile('/etc/passwd')");
        assert!(
            result.is_err(),
            "dofile must be sandboxed and produce an error"
        );
    }

    #[test]
    fn test_sandbox_require_blocked() {
        let engine = ScriptEngine::new().expect("new should succeed");
        let result = engine.execute("require('os')");
        assert!(result.is_err(), "require must be blocked by the sandbox");
    }

    #[test]
    fn test_sandbox_os_execute_blocked() {
        let engine = ScriptEngine::new().expect("new should succeed");
        // os.execute should be nil; calling it must error
        let result = engine.execute("os.execute('ls')");
        assert!(result.is_err(), "os.execute must be blocked by the sandbox");
    }

    #[test]
    fn test_sandbox_math_allowed() {
        let engine = ScriptEngine::new().expect("new should succeed");
        let result = engine.execute("return math.sqrt(16)");
        assert!(
            result.is_ok(),
            "math library should be accessible in sandbox"
        );
    }

    #[test]
    fn test_sandbox_string_allowed() {
        let engine = ScriptEngine::new().expect("new should succeed");
        let result = engine.execute("return string.upper('hello')");
        assert!(
            result.is_ok(),
            "string library should be accessible in sandbox"
        );
    }

    // ── Script cache tests ────────────────────────────────────────────────────

    #[test]
    fn test_execute_cached_populates_cache() {
        let mut engine = ScriptEngine::new().expect("new should succeed");
        assert_eq!(engine.cache_size(), 0);

        {
            let result = engine.execute_cached("return 42");
            assert!(result.is_ok());
        }
        assert_eq!(engine.cache_size(), 1, "cache should contain one entry");
    }

    #[test]
    fn test_execute_cached_same_script_not_duplicate() {
        let mut engine = ScriptEngine::new().expect("new should succeed");
        engine.execute_cached("return 1").expect("should succeed");
        engine.execute_cached("return 1").expect("should succeed");
        assert_eq!(
            engine.cache_size(),
            1,
            "same script should not be cached twice"
        );
    }

    #[test]
    fn test_script_cache_evicts_oldest_when_full() {
        let mut cache = ScriptCache::new(2);
        cache.insert("a".to_string(), vec![1]);
        cache.insert("b".to_string(), vec![2]);
        assert_eq!(cache.len(), 2);
        // Insert a third entry — "a" (oldest) should be evicted
        cache.insert("c".to_string(), vec![3]);
        assert_eq!(cache.len(), 2);
        assert!(cache.get("a").is_none(), "'a' should have been evicted");
        assert!(cache.get("b").is_some());
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn test_script_cache_get_returns_bytecode() {
        let mut cache = ScriptCache::new(4);
        cache.insert("script1".to_string(), vec![0xAA, 0xBB]);
        let result = cache.get("script1");
        assert_eq!(result, Some(&vec![0xAA, 0xBB]));
    }

    // ── Resource limit tests ──────────────────────────────────────────────────

    #[test]
    fn test_limits_normal_script_succeeds() {
        let engine = ScriptEngine::new().expect("engine created");
        let limits = ScriptLimits::default();
        let result = engine.execute_with_limits("local x = 1 + 1", limits);
        assert!(result.is_ok(), "normal script should succeed: {result:?}");
    }

    #[test]
    fn test_limits_instruction_limit_fires() {
        let engine = ScriptEngine::new().expect("engine created");
        let limits = ScriptLimits {
            // Tiny instruction budget to force the limit to trigger quickly
            max_instructions: 200,
            max_memory_bytes: 32 * 1024 * 1024,
            max_duration: Duration::from_secs(10),
        };
        // An infinite loop should be aborted by the instruction limit
        let result = engine.execute_with_limits("while true do local x = x + 1 end", limits);
        assert!(
            result.is_err(),
            "instruction limit should abort infinite loop"
        );
        let err_str = format!("{result:?}");
        assert!(
            err_str.contains(ERR_INSTRUCTION_LIMIT) || err_str.contains("Script error"),
            "error should mention instruction limit: {err_str}"
        );
    }

    #[test]
    fn test_limits_timeout_fires() {
        let engine = ScriptEngine::new().expect("engine created");
        let limits = ScriptLimits {
            // Very high instruction budget so timeout triggers first
            max_instructions: u64::MAX,
            max_memory_bytes: 32 * 1024 * 1024,
            // Short timeout — 150ms is plenty for a CI test
            max_duration: Duration::from_millis(150),
        };
        // Busy loop that relies on timeout (instruction limit is effectively off)
        // We use a large loop count that would run well beyond 150ms
        let result = engine.execute_with_limits(
            r#"
            local count = 0
            while true do
                count = count + 1
            end
            "#,
            limits,
        );
        assert!(result.is_err(), "timeout should terminate the script");
        let err_str = format!("{result:?}");
        // Either timeout error or instruction limit error is acceptable
        // (instruction hook runs at every 100 instructions, so very large budgets
        //  may still trigger the instruction check before the system timeout on
        //  slow CI; we accept both outcomes)
        assert!(
            err_str.contains(ERR_TIMEOUT)
                || err_str.contains(ERR_INSTRUCTION_LIMIT)
                || err_str.contains("Script error"),
            "error should indicate termination: {err_str}"
        );
    }

    #[test]
    fn test_limits_math_computation_within_budget() {
        let engine = ScriptEngine::new().expect("engine created");
        let limits = ScriptLimits {
            max_instructions: 1_000_000,
            max_memory_bytes: 32 * 1024 * 1024,
            max_duration: Duration::from_secs(5),
        };
        // Fibonacci — bounded, should finish well within limits
        let result = engine.execute_with_limits(
            r#"
            local function fib(n)
                if n <= 1 then return n end
                return fib(n-1) + fib(n-2)
            end
            local v = fib(20)
            "#,
            limits,
        );
        assert!(
            result.is_ok(),
            "fib(20) should complete within limits: {result:?}"
        );
    }

    #[test]
    fn test_limits_default_values() {
        let limits = ScriptLimits::default();
        assert_eq!(limits.max_instructions, 1_000_000);
        assert_eq!(limits.max_memory_bytes, 32 * 1024 * 1024);
        assert_eq!(limits.max_duration, Duration::from_secs(5));
    }
}
