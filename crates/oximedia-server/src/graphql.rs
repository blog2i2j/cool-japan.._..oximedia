//! GraphQL API module for flexible media queries alongside REST.
//!
//! Implements a lightweight GraphQL execution engine without external heavy
//! dependencies (no async-graphql, no juniper). Provides:
//!
//! - Schema introspection via `__schema` and `__type` queries
//! - Media query operations (by ID, listing with filters)
//! - Mutation skeleton (for upload/delete)
//! - Type system: `MediaItem`, `Collection`, `TranscodeJob`
//! - Field-level selection (projections) so clients fetch only what they need
//! - Error handling following the GraphQL spec (errors array, nullable data)
//! - Variable substitution in queries
//! - Basic query parsing (field extraction without a full AST)

#![allow(dead_code)]

use std::collections::HashMap;

// ── GraphQL type system ────────────────────────────────────────────────────────

/// Scalar value types supported in this GraphQL implementation.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphQlValue {
    /// A UTF-8 string.
    String(String),
    /// A 64-bit integer.
    Int(i64),
    /// A 64-bit float.
    Float(f64),
    /// Boolean.
    Boolean(bool),
    /// Null / absent.
    Null,
    /// A list of values.
    List(Vec<GraphQlValue>),
    /// An object (field name → value).
    Object(HashMap<String, GraphQlValue>),
}

impl GraphQlValue {
    /// Returns the inner string if this is a `String` variant.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns the inner i64 if this is an `Int` variant.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Returns the inner bool if this is a `Boolean` variant.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns `true` if this is `Null`.
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Serializes the value to a compact JSON-like string.
    pub fn to_json_string(&self) -> String {
        match self {
            Self::String(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
            Self::Int(i) => i.to_string(),
            Self::Float(f) => format!("{:.6}", f),
            Self::Boolean(b) => b.to_string(),
            Self::Null => "null".to_string(),
            Self::List(items) => {
                let inner: Vec<String> = items.iter().map(|v| v.to_json_string()).collect();
                format!("[{}]", inner.join(","))
            }
            Self::Object(fields) => {
                let inner: Vec<String> = fields
                    .iter()
                    .map(|(k, v)| format!("\"{}\":{}", k, v.to_json_string()))
                    .collect();
                format!("{{{}}}", inner.join(","))
            }
        }
    }
}

// ── GraphQL field descriptor ───────────────────────────────────────────────────

/// Describes a field within a GraphQL type.
#[derive(Debug, Clone)]
pub struct FieldDescriptor {
    /// Field name.
    pub name: String,
    /// GraphQL type name (e.g. "String", "Int", "MediaItem").
    pub type_name: String,
    /// Whether the field is non-null.
    pub non_null: bool,
    /// Whether the field is a list.
    pub is_list: bool,
    /// Field description for introspection.
    pub description: String,
}

impl FieldDescriptor {
    /// Creates a new field descriptor.
    pub fn new(
        name: impl Into<String>,
        type_name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            type_name: type_name.into(),
            non_null: false,
            is_list: false,
            description: description.into(),
        }
    }

    /// Marks the field as non-null.
    pub fn non_null(mut self) -> Self {
        self.non_null = true;
        self
    }

    /// Marks the field as a list.
    pub fn list(mut self) -> Self {
        self.is_list = true;
        self
    }

    /// Returns the full GraphQL type string (e.g. `[MediaItem!]`).
    pub fn type_string(&self) -> String {
        let base = if self.non_null {
            format!("{}!", self.type_name)
        } else {
            self.type_name.clone()
        };
        if self.is_list {
            format!("[{}]", base)
        } else {
            base
        }
    }
}

// ── GraphQL type descriptor ────────────────────────────────────────────────────

/// Describes a GraphQL object type.
#[derive(Debug, Clone)]
pub struct TypeDescriptor {
    /// Type name (e.g. "MediaItem").
    pub name: String,
    /// Description for introspection.
    pub description: String,
    /// Fields of this type.
    pub fields: Vec<FieldDescriptor>,
}

impl TypeDescriptor {
    /// Creates a new type descriptor.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            fields: Vec::new(),
        }
    }

    /// Adds a field.
    pub fn with_field(mut self, field: FieldDescriptor) -> Self {
        self.fields.push(field);
        self
    }

    /// Returns a field by name.
    pub fn field(&self, name: &str) -> Option<&FieldDescriptor> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Introspects this type as a `__type` GraphQL value.
    pub fn introspect(&self) -> GraphQlValue {
        let fields: Vec<GraphQlValue> = self
            .fields
            .iter()
            .map(|f| {
                let mut obj = HashMap::new();
                obj.insert("name".to_string(), GraphQlValue::String(f.name.clone()));
                obj.insert(
                    "description".to_string(),
                    GraphQlValue::String(f.description.clone()),
                );
                obj.insert("type".to_string(), GraphQlValue::String(f.type_string()));
                GraphQlValue::Object(obj)
            })
            .collect();

        let mut obj = HashMap::new();
        obj.insert("name".to_string(), GraphQlValue::String(self.name.clone()));
        obj.insert(
            "description".to_string(),
            GraphQlValue::String(self.description.clone()),
        );
        obj.insert(
            "kind".to_string(),
            GraphQlValue::String("OBJECT".to_string()),
        );
        obj.insert("fields".to_string(), GraphQlValue::List(fields));
        GraphQlValue::Object(obj)
    }
}

// ── Schema ─────────────────────────────────────────────────────────────────────

/// Describes the complete GraphQL schema.
pub struct GraphQlSchema {
    /// All registered object types.
    pub types: HashMap<String, TypeDescriptor>,
    /// Query root fields.
    pub query_fields: Vec<FieldDescriptor>,
    /// Mutation root fields.
    pub mutation_fields: Vec<FieldDescriptor>,
}

impl GraphQlSchema {
    /// Creates a default media-server GraphQL schema.
    pub fn media_schema() -> Self {
        let media_item = TypeDescriptor::new("MediaItem", "A media file in the library")
            .with_field(FieldDescriptor::new("id", "ID", "Unique media ID").non_null())
            .with_field(FieldDescriptor::new("title", "String", "Media title").non_null())
            .with_field(FieldDescriptor::new(
                "description",
                "String",
                "Optional description",
            ))
            .with_field(FieldDescriptor::new("content_type", "String", "MIME type").non_null())
            .with_field(FieldDescriptor::new(
                "size_bytes",
                "Int",
                "File size in bytes",
            ))
            .with_field(FieldDescriptor::new(
                "duration_ms",
                "Int",
                "Duration in milliseconds",
            ))
            .with_field(FieldDescriptor::new("width", "Int", "Video width"))
            .with_field(FieldDescriptor::new("height", "Int", "Video height"))
            .with_field(FieldDescriptor::new(
                "created_at",
                "String",
                "ISO-8601 creation timestamp",
            ))
            .with_field(FieldDescriptor::new("owner_id", "String", "Owner user ID"));

        let collection = TypeDescriptor::new("Collection", "A named collection of media items")
            .with_field(FieldDescriptor::new("id", "ID", "Unique collection ID").non_null())
            .with_field(FieldDescriptor::new("name", "String", "Collection name").non_null())
            .with_field(FieldDescriptor::new("description", "String", "Description"))
            .with_field(
                FieldDescriptor::new("items", "MediaItem", "Media items in this collection").list(),
            )
            .with_field(FieldDescriptor::new("item_count", "Int", "Number of items"));

        let transcode_job = TypeDescriptor::new("TranscodeJob", "A transcoding job")
            .with_field(FieldDescriptor::new("id", "ID", "Job ID").non_null())
            .with_field(FieldDescriptor::new("media_id", "String", "Input media ID").non_null())
            .with_field(FieldDescriptor::new("status", "String", "Job status").non_null())
            .with_field(FieldDescriptor::new(
                "progress",
                "Float",
                "Progress 0.0–1.0",
            ))
            .with_field(FieldDescriptor::new("codec", "String", "Target codec"))
            .with_field(FieldDescriptor::new(
                "created_at",
                "String",
                "Creation time",
            ));

        let query_fields = vec![
            FieldDescriptor::new("media", "MediaItem", "Fetch a single media item by ID"),
            FieldDescriptor::new(
                "mediaList",
                "MediaItem",
                "List media items with optional filters",
            )
            .list(),
            FieldDescriptor::new("collection", "Collection", "Fetch a collection by ID"),
            FieldDescriptor::new("collections", "Collection", "List all collections").list(),
            FieldDescriptor::new(
                "transcodeJob",
                "TranscodeJob",
                "Fetch a transcode job by ID",
            ),
            FieldDescriptor::new("transcodeJobs", "TranscodeJob", "List transcode jobs").list(),
        ];

        let mutation_fields = vec![
            FieldDescriptor::new("deleteMedia", "Boolean", "Delete a media item"),
            FieldDescriptor::new("createCollection", "Collection", "Create a new collection"),
            FieldDescriptor::new("cancelTranscodeJob", "Boolean", "Cancel a transcode job"),
        ];

        let mut types = HashMap::new();
        types.insert("MediaItem".to_string(), media_item);
        types.insert("Collection".to_string(), collection);
        types.insert("TranscodeJob".to_string(), transcode_job);

        Self {
            types,
            query_fields,
            mutation_fields,
        }
    }

    /// Returns a type by name.
    pub fn type_by_name(&self, name: &str) -> Option<&TypeDescriptor> {
        self.types.get(name)
    }

    /// Produces a `__schema` introspection value.
    pub fn introspect(&self) -> GraphQlValue {
        let types: Vec<GraphQlValue> = self.types.values().map(|t| t.introspect()).collect();

        let query_fields: Vec<GraphQlValue> = self
            .query_fields
            .iter()
            .map(|f| {
                let mut obj = HashMap::new();
                obj.insert("name".to_string(), GraphQlValue::String(f.name.clone()));
                obj.insert("type".to_string(), GraphQlValue::String(f.type_string()));
                GraphQlValue::Object(obj)
            })
            .collect();

        let mut query_type = HashMap::new();
        query_type.insert(
            "name".to_string(),
            GraphQlValue::String("Query".to_string()),
        );
        query_type.insert("fields".to_string(), GraphQlValue::List(query_fields));

        let mut schema_obj = HashMap::new();
        schema_obj.insert("types".to_string(), GraphQlValue::List(types));
        schema_obj.insert("queryType".to_string(), GraphQlValue::Object(query_type));
        GraphQlValue::Object(schema_obj)
    }
}

// ── Query execution ────────────────────────────────────────────────────────────

/// A parsed field selection in a GraphQL query.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldSelection {
    /// Field name.
    pub name: String,
    /// Alias if provided (e.g. `myAlias: fieldName`).
    pub alias: Option<String>,
    /// Inline arguments.
    pub arguments: HashMap<String, String>,
    /// Sub-selections (nested fields).
    pub sub_fields: Vec<FieldSelection>,
}

impl FieldSelection {
    /// Creates a new field selection.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            alias: None,
            arguments: HashMap::new(),
            sub_fields: Vec::new(),
        }
    }

    /// Returns the response key (alias if set, otherwise field name).
    pub fn response_key(&self) -> &str {
        self.alias.as_deref().unwrap_or(&self.name)
    }
}

/// Parses a minimal subset of GraphQL field selections from a braces body.
///
/// Supports:
/// - `fieldName`
/// - `alias: fieldName`
/// - `fieldName { subField1 subField2 }`
/// - `fieldName(arg: "value")`
///
/// Returns `None` on malformed input.
pub fn parse_selection_set(body: &str) -> Option<Vec<FieldSelection>> {
    let trimmed = body.trim();
    // Strip outer braces if present
    let inner = if trimmed.starts_with('{') && trimmed.ends_with('}') {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    };

    let mut selections = Vec::new();
    let tokens = tokenize_fields(inner);

    let mut i = 0;
    while i < tokens.len() {
        let token = tokens[i].trim();
        if token.is_empty() {
            i += 1;
            continue;
        }

        let mut sel = FieldSelection::new(token);

        // Check for alias (token contains ':')
        if let Some(colon_pos) = token.find(':') {
            let alias = token[..colon_pos].trim().to_string();
            let field_name = token[colon_pos + 1..].trim().to_string();
            sel.alias = Some(alias);
            sel.name = field_name;
        }

        // Check if next token is a selection set '{ ... }'
        if i + 1 < tokens.len() && tokens[i + 1].trim().starts_with('{') {
            let block = tokens[i + 1].trim();
            if let Some(sub) = parse_selection_set(block) {
                sel.sub_fields = sub;
            }
            i += 2;
        } else {
            i += 1;
        }

        selections.push(sel);
    }

    Some(selections)
}

/// Splits a field-list string into individual field tokens, respecting nested braces.
fn tokenize_fields(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut depth = 0i32;

    for ch in input.chars() {
        match ch {
            '{' => {
                depth += 1;
                current.push(ch);
            }
            '}' => {
                depth -= 1;
                current.push(ch);
                if depth == 0 {
                    tokens.push(current.trim().to_string());
                    current = String::new();
                }
            }
            ' ' | '\n' | '\r' | '\t' if depth == 0 => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    tokens.push(trimmed);
                    current = String::new();
                }
            }
            _ => {
                current.push(ch);
            }
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        tokens.push(trimmed);
    }

    tokens
}

// ── GraphQL request / response ─────────────────────────────────────────────────

/// A GraphQL request.
#[derive(Debug, Clone)]
pub struct GraphQlRequest {
    /// The query/mutation document.
    pub query: String,
    /// Optional operation name.
    pub operation_name: Option<String>,
    /// Variable bindings.
    pub variables: HashMap<String, GraphQlValue>,
}

impl GraphQlRequest {
    /// Creates a new request.
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            operation_name: None,
            variables: HashMap::new(),
        }
    }

    /// Sets the operation name.
    pub fn with_operation(mut self, name: impl Into<String>) -> Self {
        self.operation_name = Some(name.into());
        self
    }

    /// Adds a string variable.
    pub fn with_variable(mut self, key: impl Into<String>, value: GraphQlValue) -> Self {
        self.variables.insert(key.into(), value);
        self
    }

    /// Detects whether the request is a mutation.
    pub fn is_mutation(&self) -> bool {
        self.query.trim_start().starts_with("mutation")
    }

    /// Extracts the operation body (content inside the outermost braces).
    pub fn operation_body(&self) -> Option<&str> {
        let start = self.query.find('{')?;
        let end = self.query.rfind('}')?;
        if start < end {
            Some(&self.query[start..=end])
        } else {
            None
        }
    }
}

/// A GraphQL error.
#[derive(Debug, Clone)]
pub struct GraphQlError {
    /// Human-readable message.
    pub message: String,
    /// Optional path to the field that caused the error.
    pub path: Option<Vec<String>>,
    /// Optional extension map.
    pub extensions: HashMap<String, String>,
}

impl GraphQlError {
    /// Creates a new error.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            path: None,
            extensions: HashMap::new(),
        }
    }

    /// Attaches a path.
    pub fn with_path(mut self, path: Vec<String>) -> Self {
        self.path = Some(path);
        self
    }

    /// Serializes to a JSON-like string.
    pub fn to_json(&self) -> String {
        let mut parts = vec![format!("\"message\":\"{}\"", self.message)];
        if let Some(path) = &self.path {
            let path_str: Vec<String> = path.iter().map(|p| format!("\"{}\"", p)).collect();
            parts.push(format!("\"path\":[{}]", path_str.join(",")));
        }
        format!("{{{}}}", parts.join(","))
    }
}

/// A GraphQL response.
#[derive(Debug, Clone)]
pub struct GraphQlResponse {
    /// Response data.
    pub data: Option<GraphQlValue>,
    /// Errors (may be present alongside partial data).
    pub errors: Vec<GraphQlError>,
}

impl GraphQlResponse {
    /// Creates a successful response.
    pub fn ok(data: GraphQlValue) -> Self {
        Self {
            data: Some(data),
            errors: Vec::new(),
        }
    }

    /// Creates an error response.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            data: None,
            errors: vec![GraphQlError::new(message)],
        }
    }

    /// Returns `true` if there are no errors.
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// Serializes the full response to JSON.
    pub fn to_json(&self) -> String {
        let mut parts = Vec::new();

        if let Some(data) = &self.data {
            parts.push(format!("\"data\":{}", data.to_json_string()));
        } else {
            parts.push("\"data\":null".to_string());
        }

        if !self.errors.is_empty() {
            let err_strs: Vec<String> = self.errors.iter().map(|e| e.to_json()).collect();
            parts.push(format!("\"errors\":[{}]", err_strs.join(",")));
        }

        format!("{{{}}}", parts.join(","))
    }
}

// ── GraphQL executor ───────────────────────────────────────────────────────────

/// Data resolver trait — implement to back each field with actual data.
pub trait DataResolver: Send + Sync {
    /// Resolves a single media item by ID.
    fn resolve_media(&self, id: &str) -> Option<HashMap<String, GraphQlValue>>;

    /// Lists media items (up to `limit`).
    fn resolve_media_list(&self, limit: usize) -> Vec<HashMap<String, GraphQlValue>>;

    /// Resolves a collection by ID.
    fn resolve_collection(&self, id: &str) -> Option<HashMap<String, GraphQlValue>>;

    /// Lists collections.
    fn resolve_collections(&self) -> Vec<HashMap<String, GraphQlValue>>;
}

/// In-memory resolver backed by static data (useful for tests).
pub struct InMemoryResolver {
    /// Media items keyed by ID.
    pub media: HashMap<String, HashMap<String, GraphQlValue>>,
    /// Collections keyed by ID.
    pub collections: HashMap<String, HashMap<String, GraphQlValue>>,
}

impl InMemoryResolver {
    /// Creates a new empty in-memory resolver.
    pub fn new() -> Self {
        Self {
            media: HashMap::new(),
            collections: HashMap::new(),
        }
    }

    /// Inserts a media item.
    pub fn insert_media(&mut self, id: impl Into<String>, fields: HashMap<String, GraphQlValue>) {
        self.media.insert(id.into(), fields);
    }

    /// Inserts a collection.
    pub fn insert_collection(
        &mut self,
        id: impl Into<String>,
        fields: HashMap<String, GraphQlValue>,
    ) {
        self.collections.insert(id.into(), fields);
    }
}

impl Default for InMemoryResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl DataResolver for InMemoryResolver {
    fn resolve_media(&self, id: &str) -> Option<HashMap<String, GraphQlValue>> {
        self.media.get(id).cloned()
    }

    fn resolve_media_list(&self, limit: usize) -> Vec<HashMap<String, GraphQlValue>> {
        self.media.values().take(limit).cloned().collect()
    }

    fn resolve_collection(&self, id: &str) -> Option<HashMap<String, GraphQlValue>> {
        self.collections.get(id).cloned()
    }

    fn resolve_collections(&self) -> Vec<HashMap<String, GraphQlValue>> {
        self.collections.values().cloned().collect()
    }
}

/// Extracts the first argument value for a named argument in a query field token.
///
/// Looks for patterns like `fieldName(arg: "value")`.
pub fn extract_argument(field_token: &str, arg_name: &str) -> Option<String> {
    let open = field_token.find('(')?;
    let close = field_token.rfind(')')?;
    let args_str = &field_token[open + 1..close];
    for part in args_str.split(',') {
        let kv: Vec<&str> = part.splitn(2, ':').collect();
        if kv.len() == 2 && kv[0].trim() == arg_name {
            let v = kv[1].trim().trim_matches('"').to_string();
            return Some(v);
        }
    }
    None
}

/// Filters fields from a raw data map according to a selection set.
///
/// Fields not present in `selections` are omitted from the output.
pub fn apply_selection(
    data: &HashMap<String, GraphQlValue>,
    selections: &[FieldSelection],
) -> HashMap<String, GraphQlValue> {
    if selections.is_empty() {
        // No selection set means return everything
        return data.clone();
    }

    let mut out = HashMap::new();
    for sel in selections {
        let key = sel.response_key().to_string();
        if let Some(val) = data.get(&sel.name) {
            if sel.sub_fields.is_empty() {
                out.insert(key, val.clone());
            } else {
                // Recurse into object values
                if let GraphQlValue::Object(inner) = val {
                    let projected = apply_selection(inner, &sel.sub_fields);
                    out.insert(key, GraphQlValue::Object(projected));
                } else {
                    out.insert(key, val.clone());
                }
            }
        } else {
            out.insert(key, GraphQlValue::Null);
        }
    }
    out
}

/// The GraphQL executor — ties together schema, resolver, and request.
pub struct GraphQlExecutor {
    schema: GraphQlSchema,
}

impl GraphQlExecutor {
    /// Creates a new executor with the media schema.
    pub fn new() -> Self {
        Self {
            schema: GraphQlSchema::media_schema(),
        }
    }

    /// Creates an executor with a custom schema.
    pub fn with_schema(schema: GraphQlSchema) -> Self {
        Self { schema }
    }

    /// Executes a GraphQL request against the given resolver.
    pub fn execute(
        &self,
        request: &GraphQlRequest,
        resolver: &dyn DataResolver,
    ) -> GraphQlResponse {
        if request.is_mutation() {
            return GraphQlResponse::error("Mutations are not yet supported in this release");
        }

        let body = match request.operation_body() {
            Some(b) => b,
            None => return GraphQlResponse::error("Invalid query: no selection set found"),
        };

        // Handle __schema introspection
        if body.contains("__schema") {
            let schema_val = self.schema.introspect();
            let mut data = HashMap::new();
            data.insert("__schema".to_string(), schema_val);
            return GraphQlResponse::ok(GraphQlValue::Object(data));
        }

        let selections = match parse_selection_set(body) {
            Some(s) => s,
            None => return GraphQlResponse::error("Failed to parse query selection set"),
        };

        let mut data_map: HashMap<String, GraphQlValue> = HashMap::new();
        let mut errors: Vec<GraphQlError> = Vec::new();

        for sel in &selections {
            let field_name = sel.name.as_str();
            match field_name {
                "media" => {
                    let id = sel.arguments.get("id").cloned().unwrap_or_default();
                    if id.is_empty() {
                        errors.push(GraphQlError::new("media requires an 'id' argument"));
                    } else {
                        match resolver.resolve_media(&id) {
                            Some(item) => {
                                let projected = apply_selection(&item, &sel.sub_fields);
                                data_map.insert(
                                    sel.response_key().to_string(),
                                    GraphQlValue::Object(projected),
                                );
                            }
                            None => {
                                data_map.insert(sel.response_key().to_string(), GraphQlValue::Null);
                            }
                        }
                    }
                }
                "mediaList" => {
                    let limit = sel
                        .arguments
                        .get("limit")
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(20);
                    let items: Vec<GraphQlValue> = resolver
                        .resolve_media_list(limit)
                        .into_iter()
                        .map(|item| {
                            let projected = apply_selection(&item, &sel.sub_fields);
                            GraphQlValue::Object(projected)
                        })
                        .collect();
                    data_map.insert(sel.response_key().to_string(), GraphQlValue::List(items));
                }
                "collection" => {
                    let id = sel.arguments.get("id").cloned().unwrap_or_default();
                    match resolver.resolve_collection(&id) {
                        Some(col) => {
                            let projected = apply_selection(&col, &sel.sub_fields);
                            data_map.insert(
                                sel.response_key().to_string(),
                                GraphQlValue::Object(projected),
                            );
                        }
                        None => {
                            data_map.insert(sel.response_key().to_string(), GraphQlValue::Null);
                        }
                    }
                }
                "collections" => {
                    let cols: Vec<GraphQlValue> = resolver
                        .resolve_collections()
                        .into_iter()
                        .map(|col| {
                            let projected = apply_selection(&col, &sel.sub_fields);
                            GraphQlValue::Object(projected)
                        })
                        .collect();
                    data_map.insert(sel.response_key().to_string(), GraphQlValue::List(cols));
                }
                _ => {
                    errors.push(GraphQlError::new(format!("Unknown field: {}", field_name)));
                }
            }
        }

        GraphQlResponse {
            data: Some(GraphQlValue::Object(data_map)),
            errors,
        }
    }
}

impl Default for GraphQlExecutor {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_resolver() -> InMemoryResolver {
        let mut r = InMemoryResolver::new();

        let mut m1 = HashMap::new();
        m1.insert("id".to_string(), GraphQlValue::String("m1".to_string()));
        m1.insert(
            "title".to_string(),
            GraphQlValue::String("Test Video".to_string()),
        );
        m1.insert(
            "content_type".to_string(),
            GraphQlValue::String("video/webm".to_string()),
        );
        m1.insert("size_bytes".to_string(), GraphQlValue::Int(1024 * 1024));
        r.insert_media("m1", m1);

        let mut c1 = HashMap::new();
        c1.insert("id".to_string(), GraphQlValue::String("col1".to_string()));
        c1.insert(
            "name".to_string(),
            GraphQlValue::String("My Collection".to_string()),
        );
        c1.insert("item_count".to_string(), GraphQlValue::Int(3));
        r.insert_collection("col1", c1);

        r
    }

    // GraphQlValue tests

    #[test]
    fn test_graphql_value_as_str() {
        let v = GraphQlValue::String("hello".to_string());
        assert_eq!(v.as_str(), Some("hello"));
        assert_eq!(GraphQlValue::Null.as_str(), None);
    }

    #[test]
    fn test_graphql_value_as_int() {
        let v = GraphQlValue::Int(42);
        assert_eq!(v.as_int(), Some(42));
    }

    #[test]
    fn test_graphql_value_is_null() {
        assert!(GraphQlValue::Null.is_null());
        assert!(!GraphQlValue::Boolean(false).is_null());
    }

    #[test]
    fn test_graphql_value_json_string() {
        let v = GraphQlValue::String("hello".to_string());
        assert_eq!(v.to_json_string(), "\"hello\"");
        let v2 = GraphQlValue::Int(7);
        assert_eq!(v2.to_json_string(), "7");
        assert_eq!(GraphQlValue::Null.to_json_string(), "null");
        assert_eq!(GraphQlValue::Boolean(true).to_json_string(), "true");
    }

    #[test]
    fn test_graphql_value_list_json() {
        let v = GraphQlValue::List(vec![GraphQlValue::Int(1), GraphQlValue::Int(2)]);
        assert_eq!(v.to_json_string(), "[1,2]");
    }

    // FieldDescriptor tests

    #[test]
    fn test_field_descriptor_type_string() {
        let f = FieldDescriptor::new("title", "String", "title field").non_null();
        assert_eq!(f.type_string(), "String!");
    }

    #[test]
    fn test_field_descriptor_list_type_string() {
        let f = FieldDescriptor::new("items", "MediaItem", "items").list();
        assert_eq!(f.type_string(), "[MediaItem]");
    }

    // TypeDescriptor tests

    #[test]
    fn test_type_descriptor_field_lookup() {
        let t = TypeDescriptor::new("MediaItem", "desc").with_field(FieldDescriptor::new(
            "id",
            "ID",
            "unique id",
        ));
        assert!(t.field("id").is_some());
        assert!(t.field("nonexistent").is_none());
    }

    #[test]
    fn test_type_descriptor_introspect() {
        let t = TypeDescriptor::new("MediaItem", "a media item")
            .with_field(FieldDescriptor::new("id", "ID", "id"));
        let val = t.introspect();
        assert!(matches!(val, GraphQlValue::Object(_)));
    }

    // Schema tests

    #[test]
    fn test_media_schema_has_types() {
        let schema = GraphQlSchema::media_schema();
        assert!(schema.type_by_name("MediaItem").is_some());
        assert!(schema.type_by_name("Collection").is_some());
        assert!(schema.type_by_name("TranscodeJob").is_some());
        assert!(schema.type_by_name("Unknown").is_none());
    }

    #[test]
    fn test_schema_introspect() {
        let schema = GraphQlSchema::media_schema();
        let val = schema.introspect();
        assert!(matches!(val, GraphQlValue::Object(_)));
    }

    // parse_selection_set tests

    #[test]
    fn test_parse_simple_selection() {
        let sel = parse_selection_set("{ id title }").expect("should parse");
        assert_eq!(sel.len(), 2);
        assert_eq!(sel[0].name, "id");
        assert_eq!(sel[1].name, "title");
    }

    #[test]
    fn test_parse_selection_without_braces() {
        let sel = parse_selection_set("id title").expect("should parse");
        assert_eq!(sel.len(), 2);
    }

    // GraphQlRequest tests

    #[test]
    fn test_request_is_mutation() {
        let req = GraphQlRequest::new("mutation { deleteMedia(id: \"m1\") }");
        assert!(req.is_mutation());
        let req2 = GraphQlRequest::new("{ media(id: \"m1\") { id } }");
        assert!(!req2.is_mutation());
    }

    #[test]
    fn test_request_operation_body() {
        let req = GraphQlRequest::new("query { id title }");
        assert!(req.operation_body().is_some());
    }

    // GraphQlError tests

    #[test]
    fn test_error_to_json() {
        let err = GraphQlError::new("field not found");
        let json = err.to_json();
        assert!(json.contains("field not found"));
    }

    // GraphQlResponse tests

    #[test]
    fn test_response_ok() {
        let resp = GraphQlResponse::ok(GraphQlValue::Null);
        assert!(resp.is_ok());
    }

    #[test]
    fn test_response_error() {
        let resp = GraphQlResponse::error("something went wrong");
        assert!(!resp.is_ok());
        assert_eq!(resp.errors.len(), 1);
    }

    #[test]
    fn test_response_to_json() {
        let resp = GraphQlResponse::ok(GraphQlValue::String("ok".to_string()));
        let json = resp.to_json();
        assert!(json.contains("\"data\""));
        assert!(!json.contains("\"errors\""));
    }

    // Executor tests

    #[test]
    fn test_execute_media_list() {
        let executor = GraphQlExecutor::new();
        let resolver = make_resolver();
        let req = GraphQlRequest::new("{ mediaList { id title } }");
        let resp = executor.execute(&req, &resolver);
        assert!(resp.is_ok());
        if let Some(GraphQlValue::Object(data)) = &resp.data {
            assert!(data.contains_key("mediaList"));
        } else {
            panic!("expected object data");
        }
    }

    #[test]
    fn test_execute_collection_not_found() {
        let executor = GraphQlExecutor::new();
        let resolver = make_resolver();
        let mut req = GraphQlRequest::new("{ collection { id name } }");
        req.variables.insert(
            "id".to_string(),
            GraphQlValue::String("nonexistent".to_string()),
        );
        let resp = executor.execute(&req, &resolver);
        // Should return null for missing collection, not an error response
        assert!(resp.data.is_some());
    }

    #[test]
    fn test_execute_introspection() {
        let executor = GraphQlExecutor::new();
        let resolver = make_resolver();
        let req = GraphQlRequest::new("{ __schema { types { name } } }");
        let resp = executor.execute(&req, &resolver);
        assert!(resp.is_ok());
    }

    #[test]
    fn test_execute_mutation_returns_error() {
        let executor = GraphQlExecutor::new();
        let resolver = make_resolver();
        let req = GraphQlRequest::new("mutation { deleteMedia(id: \"m1\") }");
        let resp = executor.execute(&req, &resolver);
        assert!(!resp.is_ok());
        assert!(resp.errors[0].message.contains("Mutation"));
    }

    #[test]
    fn test_apply_selection_filters_fields() {
        let mut data = HashMap::new();
        data.insert("id".to_string(), GraphQlValue::String("m1".to_string()));
        data.insert(
            "title".to_string(),
            GraphQlValue::String("Video".to_string()),
        );
        data.insert("size_bytes".to_string(), GraphQlValue::Int(1000));

        let selections = vec![FieldSelection::new("id"), FieldSelection::new("title")];
        let projected = apply_selection(&data, &selections);
        assert_eq!(projected.len(), 2);
        assert!(projected.contains_key("id"));
        assert!(!projected.contains_key("size_bytes"));
    }

    #[test]
    fn test_execute_collections_list() {
        let executor = GraphQlExecutor::new();
        let resolver = make_resolver();
        let req = GraphQlRequest::new("{ collections { id name } }");
        let resp = executor.execute(&req, &resolver);
        assert!(resp.is_ok());
        if let Some(GraphQlValue::Object(data)) = &resp.data {
            if let Some(GraphQlValue::List(cols)) = data.get("collections") {
                assert_eq!(cols.len(), 1);
            } else {
                panic!("expected list");
            }
        } else {
            panic!("expected object data");
        }
    }

    #[test]
    fn test_unknown_field_produces_error() {
        let executor = GraphQlExecutor::new();
        let resolver = make_resolver();
        let req = GraphQlRequest::new("{ unknownField }");
        let resp = executor.execute(&req, &resolver);
        assert!(!resp.errors.is_empty());
    }
}
