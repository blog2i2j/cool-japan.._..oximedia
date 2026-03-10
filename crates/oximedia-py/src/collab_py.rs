//! Python bindings for collaborative editing sessions.
//!
//! Provides `PyCollabSession`, `PyCollabUser`, `PyComment` and standalone functions
//! for creating/joining sessions and managing collaborative editing workflows.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_timestamp() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", dur.as_secs())
}

fn gen_id(prefix: &str) -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{prefix}-{:016x}", dur.as_nanos())
}

// ---------------------------------------------------------------------------
// PyCollabUser
// ---------------------------------------------------------------------------

/// A user in a collaborative session.
#[pyclass]
#[derive(Clone)]
pub struct PyCollabUser {
    /// Unique user identifier.
    #[pyo3(get)]
    pub id: String,
    /// Display name.
    #[pyo3(get)]
    pub name: String,
    /// Role: owner, editor, viewer.
    #[pyo3(get)]
    pub role: String,
    /// Assigned color for presence.
    #[pyo3(get)]
    pub color: String,
    /// Whether user is currently active.
    #[pyo3(get)]
    pub active: bool,
    /// Timestamp when user joined.
    #[pyo3(get)]
    pub joined_at: String,
}

#[pymethods]
impl PyCollabUser {
    #[new]
    fn new(name: &str, role: &str) -> PyResult<Self> {
        let valid_role = match role {
            "owner" | "editor" | "viewer" => role.to_string(),
            _ => {
                return Err(PyValueError::new_err(
                    "Role must be owner, editor, or viewer",
                ))
            }
        };
        let colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"];
        let dur = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let color_idx = dur.as_nanos() as usize % colors.len();

        Ok(Self {
            id: gen_id("user"),
            name: name.to_string(),
            role: valid_role,
            color: colors[color_idx].to_string(),
            active: true,
            joined_at: now_timestamp(),
        })
    }

    fn can_write(&self) -> bool {
        self.role == "owner" || self.role == "editor"
    }

    fn can_manage(&self) -> bool {
        self.role == "owner"
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("id".to_string(), self.id.clone());
        m.insert("name".to_string(), self.name.clone());
        m.insert("role".to_string(), self.role.clone());
        m.insert("color".to_string(), self.color.clone());
        m.insert("active".to_string(), self.active.to_string());
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "PyCollabUser(name='{}', role='{}', active={})",
            self.name, self.role, self.active
        )
    }
}

// ---------------------------------------------------------------------------
// PyComment
// ---------------------------------------------------------------------------

/// A comment in a collaborative session.
#[pyclass]
#[derive(Clone)]
pub struct PyComment {
    /// Comment identifier.
    #[pyo3(get)]
    pub id: String,
    /// Author name.
    #[pyo3(get)]
    pub author: String,
    /// Comment text.
    #[pyo3(get)]
    pub message: String,
    /// Optional timecode reference.
    #[pyo3(get)]
    pub timecode: Option<String>,
    /// Creation timestamp.
    #[pyo3(get)]
    pub created_at: String,
    /// Whether the comment is resolved.
    #[pyo3(get)]
    pub resolved: bool,
}

#[pymethods]
impl PyComment {
    #[new]
    #[pyo3(signature = (author, message, timecode=None))]
    fn new(author: &str, message: &str, timecode: Option<String>) -> Self {
        Self {
            id: gen_id("comment"),
            author: author.to_string(),
            message: message.to_string(),
            timecode,
            created_at: now_timestamp(),
            resolved: false,
        }
    }

    fn resolve(&mut self) {
        self.resolved = true;
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("id".to_string(), self.id.clone());
        m.insert("author".to_string(), self.author.clone());
        m.insert("message".to_string(), self.message.clone());
        m.insert(
            "timecode".to_string(),
            self.timecode.clone().unwrap_or_default(),
        );
        m.insert("resolved".to_string(), self.resolved.to_string());
        m
    }

    fn __repr__(&self) -> String {
        format!(
            "PyComment(author='{}', resolved={}, message='{}')",
            self.author,
            self.resolved,
            if self.message.len() > 40 {
                format!("{}...", &self.message[..40])
            } else {
                self.message.clone()
            }
        )
    }
}

// ---------------------------------------------------------------------------
// PyCollabSession
// ---------------------------------------------------------------------------

/// A collaborative editing session.
#[pyclass]
pub struct PyCollabSession {
    /// Session identifier.
    #[pyo3(get)]
    pub id: String,
    /// Project name.
    #[pyo3(get)]
    pub project: String,
    /// Session name.
    #[pyo3(get)]
    pub name: String,
    /// Maximum users.
    #[pyo3(get)]
    pub max_users: usize,
    /// Offline editing enabled.
    #[pyo3(get)]
    pub offline_enabled: bool,
    /// Session status.
    #[pyo3(get)]
    pub status: String,
    /// Creation timestamp.
    #[pyo3(get)]
    pub created_at: String,
    users: Vec<PyCollabUser>,
    comments: Vec<PyComment>,
}

#[pymethods]
impl PyCollabSession {
    #[new]
    #[pyo3(signature = (project, name, owner_name, max_users=10, offline=false))]
    fn new(
        project: &str,
        name: &str,
        owner_name: &str,
        max_users: usize,
        offline: bool,
    ) -> PyResult<Self> {
        let owner = PyCollabUser::new(owner_name, "owner")?;
        Ok(Self {
            id: gen_id("session"),
            project: project.to_string(),
            name: name.to_string(),
            max_users,
            offline_enabled: offline,
            status: "active".to_string(),
            created_at: now_timestamp(),
            users: vec![owner],
            comments: Vec::new(),
        })
    }

    /// Add a user to the session.
    fn add_user(&mut self, user: PyCollabUser) -> PyResult<()> {
        if self.users.len() >= self.max_users {
            return Err(PyRuntimeError::new_err(format!(
                "Session full: {}/{} users",
                self.users.len(),
                self.max_users
            )));
        }
        if self.users.iter().any(|u| u.name == user.name) {
            return Err(PyValueError::new_err(format!(
                "User already in session: {}",
                user.name
            )));
        }
        self.users.push(user);
        Ok(())
    }

    /// Remove a user by name.
    fn remove_user(&mut self, name: &str) -> PyResult<()> {
        let before = self.users.len();
        self.users.retain(|u| u.name != name);
        if self.users.len() == before {
            return Err(PyValueError::new_err(format!("User not found: {name}")));
        }
        Ok(())
    }

    /// Add a comment to the session.
    fn add_comment(&mut self, comment: PyComment) {
        self.comments.push(comment);
    }

    /// Get all users.
    fn users(&self) -> Vec<PyCollabUser> {
        self.users.clone()
    }

    /// Get all comments.
    fn comments(&self) -> Vec<PyComment> {
        self.comments.clone()
    }

    /// Get the user count.
    fn user_count(&self) -> usize {
        self.users.len()
    }

    /// Get the comment count.
    fn comment_count(&self) -> usize {
        self.comments.len()
    }

    /// Close the session.
    fn close(&mut self) {
        self.status = "closed".to_string();
    }

    fn __repr__(&self) -> String {
        format!(
            "PyCollabSession(name='{}', project='{}', users={}, status='{}')",
            self.name,
            self.project,
            self.users.len(),
            self.status
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Create a new collaborative session.
#[pyfunction]
#[pyo3(signature = (project, name, owner_name, max_users=10, offline=false))]
pub fn create_session(
    project: &str,
    name: &str,
    owner_name: &str,
    max_users: usize,
    offline: bool,
) -> PyResult<PyCollabSession> {
    PyCollabSession::new(project, name, owner_name, max_users, offline)
}

/// Create a user and join an existing session.
#[pyfunction]
pub fn join_session(session: &mut PyCollabSession, name: &str, role: &str) -> PyResult<()> {
    let user = PyCollabUser::new(name, role)?;
    session.add_user(user)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all collab bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCollabUser>()?;
    m.add_class::<PyComment>()?;
    m.add_class::<PyCollabSession>()?;
    m.add_function(wrap_pyfunction!(create_session, m)?)?;
    m.add_function(wrap_pyfunction!(join_session, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_id() {
        let id = gen_id("session");
        assert!(id.starts_with("session-"));
    }

    #[test]
    fn test_collab_user_permissions() {
        let owner = PyCollabUser::new("alice", "owner").expect("should create");
        assert!(owner.can_write());
        assert!(owner.can_manage());

        let editor = PyCollabUser::new("bob", "editor").expect("should create");
        assert!(editor.can_write());
        assert!(!editor.can_manage());

        let viewer = PyCollabUser::new("eve", "viewer").expect("should create");
        assert!(!viewer.can_write());
        assert!(!viewer.can_manage());
    }

    #[test]
    fn test_invalid_role() {
        let result = PyCollabUser::new("test", "admin");
        assert!(result.is_err());
    }

    #[test]
    fn test_comment_resolve() {
        let mut comment = PyComment::new("bob", "Fix this", None);
        assert!(!comment.resolved);
        comment.resolve();
        assert!(comment.resolved);
    }

    #[test]
    fn test_comment_repr() {
        let comment = PyComment::new("bob", "Test message", None);
        let repr = comment.__repr__();
        assert!(repr.contains("bob"));
        assert!(repr.contains("Test message"));
    }
}
