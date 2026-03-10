//! Automation rules and smart scheduling
//!
//! Provides conditional logic, dynamic scheduling, and scripting support
//! for intelligent playout automation.

use crate::Result;
use chrono::{DateTime, Datelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};
use uuid::Uuid;

/// Automation rule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationConfig {
    /// Enable automation
    pub enabled: bool,

    /// Enable smart scheduling
    pub smart_scheduling: bool,

    /// Enable fill-to-time
    pub fill_to_time: bool,

    /// Auto-fill with filler content
    pub auto_fill: bool,

    /// Filler content directory
    pub filler_dir: std::path::PathBuf,

    /// Enable scripting
    pub scripting_enabled: bool,

    /// Script timeout in seconds
    pub script_timeout_sec: u32,
}

impl Default for AutomationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            smart_scheduling: true,
            fill_to_time: true,
            auto_fill: true,
            filler_dir: std::path::PathBuf::from("/var/oximedia/filler"),
            scripting_enabled: true,
            script_timeout_sec: 30,
        }
    }
}

/// Automation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationRule {
    /// Rule ID
    pub id: Uuid,

    /// Rule name
    pub name: String,

    /// Rule description
    pub description: Option<String>,

    /// Rule priority (higher = higher priority)
    pub priority: u32,

    /// Rule enabled flag
    pub enabled: bool,

    /// Conditions that must be met
    pub conditions: Vec<Condition>,

    /// Actions to execute when conditions are met
    pub actions: Vec<Action>,

    /// Created timestamp
    pub created_at: DateTime<Utc>,
}

/// Condition types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Condition {
    /// Time-based condition
    TimeRange {
        start: String, // HH:MM:SS format
        end: String,
    },

    /// Day of week condition
    DayOfWeek { days: Vec<chrono::Weekday> },

    /// Date range condition
    DateRange {
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    },

    /// Content duration condition
    ContentDuration {
        min_ms: Option<u64>,
        max_ms: Option<u64>,
    },

    /// Gap duration condition
    GapDuration { min_ms: u64 },

    /// Custom script condition (Lua)
    Script { script: String },

    /// Metadata match
    MetadataMatch { key: String, value: String },

    /// Remaining time
    RemainingTime { min_ms: u64, max_ms: u64 },
}

/// Action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    /// Insert content
    InsertContent {
        content_id: Uuid,
        position: InsertPosition,
    },

    /// Remove content
    RemoveContent { content_id: Uuid },

    /// Adjust timing
    AdjustTiming { stretch_factor: f64 },

    /// Add filler
    AddFiller { duration_ms: u64 },

    /// Replace content
    ReplaceContent { old_id: Uuid, new_id: Uuid },

    /// Execute script (Lua)
    ExecuteScript { script: String },

    /// Send notification
    SendNotification {
        message: String,
        severity: NotificationSeverity,
    },

    /// Trigger GPI/GPO
    TriggerGpio { port: u8, state: bool },
}

/// Insert position for content
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum InsertPosition {
    Next,
    End,
    AtIndex(usize),
    AtTime(i64), // Unix timestamp
}

/// Notification severity
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum NotificationSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Smart scheduler for fill-to-time operations
pub struct SmartScheduler {
    #[allow(dead_code)]
    config: AutomationConfig,
    rules: Arc<RwLock<Vec<AutomationRule>>>,
}

impl SmartScheduler {
    /// Create new smart scheduler
    pub fn new(config: AutomationConfig) -> Self {
        Self {
            config,
            rules: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add automation rule
    pub async fn add_rule(&self, rule: AutomationRule) -> Result<()> {
        let mut rules = self.rules.write().await;
        rules.push(rule.clone());
        rules.sort_by_key(|r| std::cmp::Reverse(r.priority));
        info!("Added automation rule: {}", rule.name);
        Ok(())
    }

    /// Remove automation rule
    pub async fn remove_rule(&self, rule_id: &Uuid) -> Result<()> {
        let mut rules = self.rules.write().await;
        rules.retain(|r| r.id != *rule_id);
        info!("Removed automation rule: {}", rule_id);
        Ok(())
    }

    /// Evaluate conditions
    pub async fn evaluate(&self, context: &EvaluationContext) -> Vec<Action> {
        let rules = self.rules.read().await;
        let mut actions = Vec::new();

        for rule in rules.iter() {
            if !rule.enabled {
                continue;
            }

            if self.check_conditions(&rule.conditions, context) {
                debug!("Rule '{}' conditions met", rule.name);
                actions.extend(rule.actions.clone());
            }
        }

        actions
    }

    /// Check if all conditions are met
    fn check_conditions(&self, conditions: &[Condition], context: &EvaluationContext) -> bool {
        conditions.iter().all(|c| self.check_condition(c, context))
    }

    /// Check individual condition
    fn check_condition(&self, condition: &Condition, context: &EvaluationContext) -> bool {
        match condition {
            Condition::TimeRange { start, end } => {
                let now = context.current_time.format("%H:%M:%S").to_string();
                now >= *start && now <= *end
            }
            Condition::DayOfWeek { days } => days.contains(&context.current_time.weekday()),
            Condition::DateRange { start, end } => {
                context.current_time >= *start && context.current_time <= *end
            }
            Condition::ContentDuration { min_ms, max_ms } => {
                if let Some(duration) = context.content_duration_ms {
                    let meets_min = min_ms.is_none_or(|min| duration >= min);
                    let meets_max = max_ms.is_none_or(|max| duration <= max);
                    meets_min && meets_max
                } else {
                    false
                }
            }
            Condition::GapDuration { min_ms } => {
                context.gap_duration_ms.is_some_and(|gap| gap >= *min_ms)
            }
            Condition::RemainingTime { min_ms, max_ms } => {
                context.remaining_time_ms >= *min_ms && context.remaining_time_ms <= *max_ms
            }
            Condition::MetadataMatch { key, value } => context.metadata.get(key) == Some(value),
            Condition::Script { script } => evaluate_script_condition(script, context),
        }
    }

    /// Calculate fill-to-time adjustments
    pub fn calculate_fill(
        &self,
        target_duration_ms: u64,
        current_duration_ms: u64,
    ) -> FillStrategy {
        if target_duration_ms == current_duration_ms {
            return FillStrategy::None;
        }

        if current_duration_ms < target_duration_ms {
            let gap = target_duration_ms - current_duration_ms;
            FillStrategy::AddFiller { duration_ms: gap }
        } else {
            let overage = current_duration_ms - target_duration_ms;
            let stretch_factor = target_duration_ms as f64 / current_duration_ms as f64;

            if stretch_factor >= 0.95 {
                // Minor adjustment, use stretching
                FillStrategy::Stretch {
                    factor: stretch_factor,
                }
            } else {
                // Significant overage, remove content
                FillStrategy::RemoveContent {
                    duration_ms: overage,
                }
            }
        }
    }

    /// Get all rules
    pub async fn list_rules(&self) -> Vec<AutomationRule> {
        self.rules.read().await.clone()
    }
}

/// Evaluation context for rules
#[derive(Debug, Clone)]
pub struct EvaluationContext {
    /// Current time
    pub current_time: DateTime<Utc>,

    /// Content duration in milliseconds
    pub content_duration_ms: Option<u64>,

    /// Gap duration in milliseconds
    pub gap_duration_ms: Option<u64>,

    /// Remaining time in block
    pub remaining_time_ms: u64,

    /// Metadata
    pub metadata: HashMap<String, String>,

    /// Custom data
    pub custom_data: HashMap<String, serde_json::Value>,
}

impl Default for EvaluationContext {
    fn default() -> Self {
        Self {
            current_time: Utc::now(),
            content_duration_ms: None,
            gap_duration_ms: None,
            remaining_time_ms: 0,
            metadata: HashMap::new(),
            custom_data: HashMap::new(),
        }
    }
}

/// Fill strategy result
#[derive(Debug, Clone, PartialEq)]
pub enum FillStrategy {
    None,
    AddFiller { duration_ms: u64 },
    Stretch { factor: f64 },
    RemoveContent { duration_ms: u64 },
}

// ---------------------------------------------------------------------------
// Script condition evaluator
// ---------------------------------------------------------------------------

/// Token produced by the script condition lexer.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    StringLit(String),
    NumberLit(f64),
    TimeLit(String), // "HH:MM" or "HH:MM:SS"
    Eq,              // ==
    Ne,              // !=
    Lt,              // <
    Le,              // <=
    Gt,              // >
    Ge,              // >=
    And,             // &&  or  and
    Or,              // ||  or  or
    Not,             // !   or  not
    LParen,
    RParen,
}

/// Lex a script condition string into tokens.
fn lex_condition(input: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            c if c.is_whitespace() => i += 1,
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            '!' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Ne);
                    i += 2;
                } else {
                    tokens.push(Token::Not);
                    i += 1;
                }
            }
            '=' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Eq);
                    i += 2;
                } else {
                    // bare '=' treated as ==
                    tokens.push(Token::Eq);
                    i += 1;
                }
            }
            '<' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Le);
                    i += 2;
                } else {
                    tokens.push(Token::Lt);
                    i += 1;
                }
            }
            '>' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Ge);
                    i += 2;
                } else {
                    tokens.push(Token::Gt);
                    i += 1;
                }
            }
            '&' => {
                if i + 1 < chars.len() && chars[i + 1] == '&' {
                    tokens.push(Token::And);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            '|' => {
                if i + 1 < chars.len() && chars[i + 1] == '|' {
                    tokens.push(Token::Or);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            '\'' | '"' => {
                let quote = chars[i];
                i += 1;
                let mut s = String::new();
                while i < chars.len() && chars[i] != quote {
                    s.push(chars[i]);
                    i += 1;
                }
                if i < chars.len() {
                    i += 1; // consume closing quote
                }
                tokens.push(Token::StringLit(s));
            }
            c if c.is_ascii_digit() => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                // Detect HH:MM or HH:MM:SS time literal after digits
                if i < chars.len() && chars[i] == ':' {
                    // Collect as time literal
                    while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == ':') {
                        i += 1;
                    }
                    let s: String = chars[start..i].iter().collect();
                    tokens.push(Token::TimeLit(s));
                } else {
                    let s: String = chars[start..i].iter().collect();
                    if let Ok(v) = s.parse::<f64>() {
                        tokens.push(Token::NumberLit(v));
                    }
                }
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                match word.to_lowercase().as_str() {
                    "and" => tokens.push(Token::And),
                    "or" => tokens.push(Token::Or),
                    "not" => tokens.push(Token::Not),
                    "true" => tokens.push(Token::NumberLit(1.0)),
                    "false" => tokens.push(Token::NumberLit(0.0)),
                    _ => tokens.push(Token::Ident(word)),
                }
            }
            _ => i += 1,
        }
    }
    tokens
}

/// Resolve a named field from the evaluation context to a comparable string.
fn resolve_field(name: &str, context: &EvaluationContext) -> Option<String> {
    match name {
        "time" => Some(context.current_time.format("%H:%M:%S").to_string()),
        "date" => Some(context.current_time.format("%Y-%m-%d").to_string()),
        "day" => Some(format!("{}", context.current_time.weekday())),
        "duration" => context.content_duration_ms.map(|d| (d / 1000).to_string()),
        "duration_ms" => context.content_duration_ms.map(|d| d.to_string()),
        "gap" => context.gap_duration_ms.map(|g| (g / 1000).to_string()),
        "gap_ms" => context.gap_duration_ms.map(|g| g.to_string()),
        "remaining" => Some((context.remaining_time_ms / 1000).to_string()),
        "remaining_ms" => Some(context.remaining_time_ms.to_string()),
        other => context.metadata.get(other).cloned(),
    }
}

/// Compare two string values lexicographically (with numeric fallback).
fn compare_values(left: &str, op: &Token, right: &str) -> bool {
    // Try numeric comparison first
    if let (Ok(l), Ok(r)) = (left.parse::<f64>(), right.parse::<f64>()) {
        return match op {
            Token::Eq => (l - r).abs() < f64::EPSILON,
            Token::Ne => (l - r).abs() >= f64::EPSILON,
            Token::Lt => l < r,
            Token::Le => l <= r,
            Token::Gt => l > r,
            Token::Ge => l >= r,
            _ => false,
        };
    }
    // String comparison (supports "HH:MM:SS" time strings lexicographically)
    match op {
        Token::Eq => left == right,
        Token::Ne => left != right,
        Token::Lt => left < right,
        Token::Le => left <= right,
        Token::Gt => left > right,
        Token::Ge => left >= right,
        _ => false,
    }
}

/// Recursive-descent parser / evaluator for boolean script condition expressions.
///
/// Grammar (simplified):
///   expr      ::= or_expr
///   or_expr   ::= and_expr ( ('||'|'or') and_expr )*
///   and_expr  ::= not_expr ( ('&&'|'and') not_expr )*
///   not_expr  ::= ('!'|'not') not_expr | atom
///   atom      ::= '(' expr ')' | comparison
///   comparison::= value ( op value )?
///   value     ::= Ident | StringLit | NumberLit | TimeLit
struct ConditionParser<'a> {
    tokens: &'a [Token],
    pos: usize,
    context: &'a EvaluationContext,
}

impl<'a> ConditionParser<'a> {
    fn new(tokens: &'a [Token], context: &'a EvaluationContext) -> Self {
        Self {
            tokens,
            pos: 0,
            context,
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn consume(&mut self) -> Option<&Token> {
        let t = self.tokens.get(self.pos);
        self.pos += 1;
        t
    }

    fn parse_expr(&mut self) -> bool {
        self.parse_or()
    }

    fn parse_or(&mut self) -> bool {
        let mut val = self.parse_and();
        while matches!(self.peek(), Some(Token::Or)) {
            self.consume();
            let right = self.parse_and();
            val = val || right;
        }
        val
    }

    fn parse_and(&mut self) -> bool {
        let mut val = self.parse_not();
        while matches!(self.peek(), Some(Token::And)) {
            self.consume();
            let right = self.parse_not();
            val = val && right;
        }
        val
    }

    fn parse_not(&mut self) -> bool {
        if matches!(self.peek(), Some(Token::Not)) {
            self.consume();
            return !self.parse_not();
        }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> bool {
        if matches!(self.peek(), Some(Token::LParen)) {
            self.consume();
            let val = self.parse_expr();
            if matches!(self.peek(), Some(Token::RParen)) {
                self.consume();
            }
            return val;
        }
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> bool {
        let left = match self.parse_value() {
            Some(v) => v,
            None => return false,
        };

        let op = match self.peek() {
            Some(Token::Eq) | Some(Token::Ne) | Some(Token::Lt) | Some(Token::Le)
            | Some(Token::Gt) | Some(Token::Ge) => {
                let t = self.tokens[self.pos].clone();
                self.consume();
                t
            }
            _ => {
                // Boolean coercion: non-empty, non-zero string → true
                if left.is_empty() || left == "0" || left.eq_ignore_ascii_case("false") {
                    return false;
                }
                return true;
            }
        };

        let right = match self.parse_value() {
            Some(v) => v,
            None => return false,
        };

        compare_values(&left, &op, &right)
    }

    fn parse_value(&mut self) -> Option<String> {
        match self.peek()?.clone() {
            Token::Ident(name) => {
                self.consume();
                Some(resolve_field(&name, self.context).unwrap_or_default())
            }
            Token::StringLit(s) => {
                self.consume();
                Some(s)
            }
            Token::NumberLit(n) => {
                self.consume();
                Some(n.to_string())
            }
            Token::TimeLit(t) => {
                self.consume();
                // Normalise to HH:MM:SS
                let parts: Vec<&str> = t.split(':').collect();
                if parts.len() == 2 {
                    Some(format!("{}:{}:00", parts[0], parts[1]))
                } else {
                    Some(t)
                }
            }
            _ => None,
        }
    }
}

/// Evaluate a simple boolean condition expression against the evaluation context.
///
/// Supported syntax examples:
/// - `"time > 18:00"`
/// - `"duration < 30"`
/// - `"title == 'News'"`
/// - `"time >= 06:00 && time < 18:00"`
/// - `"remaining_ms > 5000 || gap > 0"`
fn evaluate_script_condition(script: &str, context: &EvaluationContext) -> bool {
    let tokens = lex_condition(script.trim());
    if tokens.is_empty() {
        debug!(
            "Script condition '{}' produced no tokens, defaulting to false",
            script
        );
        return false;
    }
    let mut parser = ConditionParser::new(&tokens, context);
    let result = parser.parse_expr();
    debug!("Script condition '{}' evaluated to {}", script, result);
    result
}

// ---------------------------------------------------------------------------
// Script executor
// ---------------------------------------------------------------------------

/// Script executor for Lua scripts
pub struct ScriptExecutor {
    #[allow(dead_code)]
    timeout_sec: u32,
}

impl ScriptExecutor {
    /// Create new script executor
    pub fn new(timeout_sec: u32) -> Self {
        Self { timeout_sec }
    }

    /// Execute a script expression against the evaluation context.
    ///
    /// The script is evaluated as a boolean expression using the same
    /// recursive-descent parser as the condition evaluator (see
    /// [`evaluate_script_condition`]).  A `true` result is represented as
    /// `{"result": true, "value": 1}` and `false` as `{"result": false,
    /// "value": 0}`.  Complex Lua-style assignments or multi-statement scripts
    /// are not supported; they are treated as a single boolean predicate.
    pub async fn execute(
        &self,
        script: &str,
        context: &EvaluationContext,
    ) -> Result<serde_json::Value> {
        let result = evaluate_script_condition(script, context);
        info!(
            "Script executed in {} s timeout budget: '{}' → {}",
            self.timeout_sec, script, result
        );
        Ok(serde_json::json!({
            "result": result,
            "value": if result { 1 } else { 0 },
            "script": script,
        }))
    }
}

/// Break optimizer for commercial breaks
pub struct BreakOptimizer {
    target_break_duration_ms: u64,
    tolerance_ms: u64,
}

impl BreakOptimizer {
    /// Create new break optimizer
    pub fn new(target_duration_ms: u64, tolerance_ms: u64) -> Self {
        Self {
            target_break_duration_ms: target_duration_ms,
            tolerance_ms,
        }
    }

    /// Optimize commercial break to fit target duration
    pub fn optimize(&self, spots: Vec<CommercialSpot>) -> OptimizationResult {
        let total_duration: u64 = spots.iter().map(|s| s.duration_ms).sum();
        let target = self.target_break_duration_ms;

        if (total_duration as i64 - target as i64).abs() <= self.tolerance_ms as i64 {
            return OptimizationResult {
                spots,
                total_duration_ms: total_duration,
                status: OptimizationStatus::Optimal,
                adjustments: Vec::new(),
            };
        }

        if total_duration < target {
            // Need to add filler
            let filler_duration = target - total_duration;
            OptimizationResult {
                spots,
                total_duration_ms: total_duration,
                status: OptimizationStatus::NeedsFiller,
                adjustments: vec![BreakAdjustment::AddFiller {
                    duration_ms: filler_duration,
                }],
            }
        } else {
            // Overage
            OptimizationResult {
                spots,
                total_duration_ms: total_duration,
                status: OptimizationStatus::Overage,
                adjustments: vec![BreakAdjustment::RemoveSpots {
                    duration_ms: total_duration - target,
                }],
            }
        }
    }
}

/// Commercial spot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommercialSpot {
    pub id: Uuid,
    pub isci_code: String,
    pub duration_ms: u64,
    pub priority: u32,
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub spots: Vec<CommercialSpot>,
    pub total_duration_ms: u64,
    pub status: OptimizationStatus,
    pub adjustments: Vec<BreakAdjustment>,
}

/// Optimization status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationStatus {
    Optimal,
    NeedsFiller,
    Overage,
}

/// Break adjustment
#[derive(Debug, Clone, PartialEq)]
pub enum BreakAdjustment {
    AddFiller { duration_ms: u64 },
    RemoveSpots { duration_ms: u64 },
    Reorder,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_automation_config_default() {
        let config = AutomationConfig::default();
        assert!(config.enabled);
        assert!(config.smart_scheduling);
        assert!(config.fill_to_time);
    }

    #[test]
    fn test_insert_position_equality() {
        assert_eq!(InsertPosition::Next, InsertPosition::Next);
        assert_ne!(InsertPosition::Next, InsertPosition::End);
    }

    #[test]
    fn test_notification_severity_equality() {
        assert_eq!(NotificationSeverity::Info, NotificationSeverity::Info);
        assert_ne!(NotificationSeverity::Info, NotificationSeverity::Error);
    }

    #[tokio::test]
    async fn test_smart_scheduler_creation() {
        let config = AutomationConfig::default();
        let scheduler = SmartScheduler::new(config);
        let rules = scheduler.list_rules().await;
        assert!(rules.is_empty());
    }

    #[tokio::test]
    async fn test_automation_rule_add_remove() {
        let config = AutomationConfig::default();
        let scheduler = SmartScheduler::new(config);

        let rule = AutomationRule {
            id: Uuid::new_v4(),
            name: "Test Rule".to_string(),
            description: Some("Test description".to_string()),
            priority: 100,
            enabled: true,
            conditions: vec![],
            actions: vec![],
            created_at: Utc::now(),
        };

        let rule_id = rule.id;

        // Add rule
        scheduler
            .add_rule(rule)
            .await
            .expect("should succeed in test");
        assert_eq!(scheduler.list_rules().await.len(), 1);

        // Remove rule
        scheduler
            .remove_rule(&rule_id)
            .await
            .expect("should succeed in test");
        assert!(scheduler.list_rules().await.is_empty());
    }

    #[test]
    fn test_time_range_condition() {
        let scheduler = SmartScheduler::new(AutomationConfig::default());
        let condition = Condition::TimeRange {
            start: "09:00:00".to_string(),
            end: "17:00:00".to_string(),
        };

        let context = EvaluationContext {
            current_time: Utc::now(),
            ..Default::default()
        };

        // This test depends on current time, so just verify it doesn't panic
        let _ = scheduler.check_condition(&condition, &context);
    }

    #[test]
    fn test_fill_strategy_calculation() {
        let scheduler = SmartScheduler::new(AutomationConfig::default());

        // Exact match
        let strategy = scheduler.calculate_fill(60000, 60000);
        assert_eq!(strategy, FillStrategy::None);

        // Need filler
        let strategy = scheduler.calculate_fill(70000, 60000);
        match strategy {
            FillStrategy::AddFiller { duration_ms } => assert_eq!(duration_ms, 10000),
            _ => panic!("Expected AddFiller strategy"),
        }

        // Need to remove content
        let strategy = scheduler.calculate_fill(60000, 80000);
        match strategy {
            FillStrategy::RemoveContent { duration_ms } => assert_eq!(duration_ms, 20000),
            _ => panic!("Expected RemoveContent strategy"),
        }
    }

    #[test]
    fn test_break_optimizer() {
        let optimizer = BreakOptimizer::new(120000, 2000);

        let spots = vec![
            CommercialSpot {
                id: Uuid::new_v4(),
                isci_code: "SPOT1".to_string(),
                duration_ms: 30000,
                priority: 100,
            },
            CommercialSpot {
                id: Uuid::new_v4(),
                isci_code: "SPOT2".to_string(),
                duration_ms: 30000,
                priority: 100,
            },
            CommercialSpot {
                id: Uuid::new_v4(),
                isci_code: "SPOT3".to_string(),
                duration_ms: 30000,
                priority: 100,
            },
            CommercialSpot {
                id: Uuid::new_v4(),
                isci_code: "SPOT4".to_string(),
                duration_ms: 30000,
                priority: 100,
            },
        ];

        let result = optimizer.optimize(spots);
        assert_eq!(result.total_duration_ms, 120000);
        assert_eq!(result.status, OptimizationStatus::Optimal);
    }

    #[test]
    fn test_break_optimizer_needs_filler() {
        let optimizer = BreakOptimizer::new(120000, 2000);

        let spots = vec![
            CommercialSpot {
                id: Uuid::new_v4(),
                isci_code: "SPOT1".to_string(),
                duration_ms: 30000,
                priority: 100,
            },
            CommercialSpot {
                id: Uuid::new_v4(),
                isci_code: "SPOT2".to_string(),
                duration_ms: 30000,
                priority: 100,
            },
        ];

        let result = optimizer.optimize(spots);
        assert_eq!(result.total_duration_ms, 60000);
        assert_eq!(result.status, OptimizationStatus::NeedsFiller);
        assert_eq!(result.adjustments.len(), 1);
    }

    #[test]
    fn test_evaluation_context_default() {
        let context = EvaluationContext::default();
        assert!(context.metadata.is_empty());
        assert!(context.custom_data.is_empty());
        assert_eq!(context.remaining_time_ms, 0);
    }

    #[test]
    fn test_script_executor_creation() {
        let executor = ScriptExecutor::new(30);
        assert_eq!(executor.timeout_sec, 30);
    }

    #[test]
    fn test_optimization_status_equality() {
        assert_eq!(OptimizationStatus::Optimal, OptimizationStatus::Optimal);
        assert_ne!(OptimizationStatus::Optimal, OptimizationStatus::Overage);
    }

    #[test]
    fn test_commercial_spot_creation() {
        let spot = CommercialSpot {
            id: Uuid::new_v4(),
            isci_code: "TEST1234H".to_string(),
            duration_ms: 30000,
            priority: 100,
        };

        assert_eq!(spot.isci_code, "TEST1234H");
        assert_eq!(spot.duration_ms, 30000);
    }
}
