//! Music sync licensing for video content.
//!
//! "Sync licensing" (synchronisation licensing) governs the rights to pair a
//! musical composition or recording with moving images.  This module models the
//! full lifecycle of a sync licence request from initial submission through
//! approval, rejection, or counter-offer.
//!
//! # Workflow overview
//! ```text
//! submit_request() → Pending
//!   ├─ approve()       → Approved(SyncLicenseTerm)
//!   ├─ reject()        → Rejected { reason }
//!   └─ counter_offer() → CounterOffer(SyncLicenseTerm)
//!                           └─ approve() / reject() …
//! ```
//!
//! A request that has already been decided (Approved, Rejected, or expired)
//! cannot be mutated again without creating a new request.

#![allow(missing_docs)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use thiserror::Error;

// ── LicenseError ─────────────────────────────────────────────────────────────

/// Errors returned by [`SyncLicenseManager`] operations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum LicenseError {
    /// The specified request ID does not exist.
    #[error("license request not found: {0}")]
    NotFound(String),

    /// The request has already been decided (approved, rejected, or expired)
    /// and cannot be updated.
    #[error("license request already decided: {0}")]
    AlreadyDecided(String),

    /// The supplied license terms are invalid (e.g. zero duration, negative
    /// fee, or empty territory).
    #[error("invalid license terms")]
    InvalidTerms,
}

// ── SyncLicenseType ───────────────────────────────────────────────────────────

/// The distribution context for which a sync license is sought.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyncLicenseType {
    /// Theatrical / cinema release.
    Theatrical,
    /// Free-to-air or pay TV broadcast.
    Broadcast,
    /// Online / streaming digital distribution.
    Digital,
    /// Theatrical or online trailer.
    Trailer,
    /// Commercial advertisement.
    Advertisement,
    /// Background / ambient music (e.g. in a scene, restaurant, etc.).
    BackgroundMusic,
}

// ── SyncLicenseTerm ───────────────────────────────────────────────────────────

/// The concrete terms under which a sync license is (or would be) granted.
#[derive(Debug, Clone, PartialEq)]
pub struct SyncLicenseTerm {
    /// Distribution context.
    pub license_type: SyncLicenseType,
    /// ISO 3166-1 alpha-2 territory code, or `"WORLD"` for worldwide rights.
    pub territory: String,
    /// How many years the license is valid.
    pub duration_years: u8,
    /// Whether the rights are granted exclusively to the licensee.
    pub exclusive: bool,
    /// One-time licensing fee in US dollars.
    pub fee_usd: f64,
    /// Optional additional restrictions (e.g. "no political advertising",
    /// "web only", "max 30-second clip").
    pub restrictions: Vec<String>,
}

impl SyncLicenseTerm {
    /// Return `true` if the terms are internally consistent and non-degenerate.
    pub fn is_valid(&self) -> bool {
        self.duration_years > 0 && self.fee_usd >= 0.0 && !self.territory.is_empty()
    }
}

// ── SyncLicenseRequest ────────────────────────────────────────────────────────

/// A request to obtain a sync license pairing music with a video production.
#[derive(Debug, Clone)]
pub struct SyncLicenseRequest {
    /// Identifier of the music track being licensed.
    pub music_id: String,
    /// Identifier of the video production.
    pub video_id: String,
    /// Free-text description of how the music will be used.
    pub usage_context: String,
    /// The terms the requestor is proposing.
    pub requested_terms: SyncLicenseTerm,
}

// ── SyncLicenseStatus ─────────────────────────────────────────────────────────

/// Current lifecycle state of a sync license request.
#[derive(Debug, Clone, PartialEq)]
pub enum SyncLicenseStatus {
    /// The request has been submitted and awaits review.
    Pending,
    /// The request has been approved with the enclosed terms.
    Approved(SyncLicenseTerm),
    /// The request has been declined.
    Rejected {
        /// Human-readable explanation.
        reason: String,
    },
    /// The rights-holder has proposed alternative terms.
    CounterOffer(SyncLicenseTerm),
    /// The request window has closed without a decision.
    Expired,
}

impl SyncLicenseStatus {
    /// Return `true` when no further state transitions are permitted.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            SyncLicenseStatus::Approved(_)
                | SyncLicenseStatus::Rejected { .. }
                | SyncLicenseStatus::Expired
        )
    }
}

// ── Internal record ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct LicenseRecord {
    request: SyncLicenseRequest,
    status: SyncLicenseStatus,
    /// Auto-incrementing sequence number used to generate stable request IDs.
    _seq: u64,
}

// ── SyncLicenseManager ────────────────────────────────────────────────────────

/// Central manager for music sync licence requests.
#[derive(Debug, Default)]
pub struct SyncLicenseManager {
    records: HashMap<String, LicenseRecord>,
    next_seq: u64,
}

impl SyncLicenseManager {
    /// Create a new, empty `SyncLicenseManager`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // ── Submission ────────────────────────────────────────────────────────────

    /// Submit a new sync license request.
    ///
    /// Returns the generated `request_id` (a stable string unique within this
    /// manager instance).  The initial status is [`SyncLicenseStatus::Pending`].
    pub fn submit_request(&mut self, request: SyncLicenseRequest) -> String {
        let seq = self.next_seq;
        self.next_seq += 1;

        // Build a deterministic request ID: "slr-<seq>-<music_id>-<video_id>".
        // Using a simple format avoids a UUID dependency while still producing
        // unique, human-readable identifiers within a single manager instance.
        let request_id = format!("slr-{}-{}-{}", seq, request.music_id, request.video_id);

        self.records.insert(
            request_id.clone(),
            LicenseRecord {
                request,
                status: SyncLicenseStatus::Pending,
                _seq: seq,
            },
        );
        request_id
    }

    // ── Decisions ─────────────────────────────────────────────────────────────

    /// Approve a pending (or counter-offered) request with the supplied terms.
    ///
    /// # Errors
    /// * [`LicenseError::NotFound`] – request does not exist.
    /// * [`LicenseError::AlreadyDecided`] – request is in a terminal state.
    /// * [`LicenseError::InvalidTerms`] – the supplied terms are degenerate.
    pub fn approve(
        &mut self,
        request_id: &str,
        terms: SyncLicenseTerm,
    ) -> Result<(), LicenseError> {
        if !terms.is_valid() {
            return Err(LicenseError::InvalidTerms);
        }
        let record = self
            .records
            .get_mut(request_id)
            .ok_or_else(|| LicenseError::NotFound(request_id.to_string()))?;
        if record.status.is_terminal() {
            return Err(LicenseError::AlreadyDecided(request_id.to_string()));
        }
        record.status = SyncLicenseStatus::Approved(terms);
        Ok(())
    }

    /// Reject a pending (or counter-offered) request with a reason.
    ///
    /// # Errors
    /// * [`LicenseError::NotFound`] – request does not exist.
    /// * [`LicenseError::AlreadyDecided`] – request is in a terminal state.
    pub fn reject(&mut self, request_id: &str, reason: String) -> Result<(), LicenseError> {
        let record = self
            .records
            .get_mut(request_id)
            .ok_or_else(|| LicenseError::NotFound(request_id.to_string()))?;
        if record.status.is_terminal() {
            return Err(LicenseError::AlreadyDecided(request_id.to_string()));
        }
        record.status = SyncLicenseStatus::Rejected { reason };
        Ok(())
    }

    /// Respond to a pending request with a counter-offer.
    ///
    /// A counter-offer replaces the current status with
    /// [`SyncLicenseStatus::CounterOffer`] and is itself not terminal, so it
    /// can subsequently be approved or rejected.
    ///
    /// # Errors
    /// * [`LicenseError::NotFound`] – request does not exist.
    /// * [`LicenseError::AlreadyDecided`] – request is in a terminal state.
    /// * [`LicenseError::InvalidTerms`] – the supplied counter-offer terms are
    ///   degenerate.
    pub fn counter_offer(
        &mut self,
        request_id: &str,
        terms: SyncLicenseTerm,
    ) -> Result<(), LicenseError> {
        if !terms.is_valid() {
            return Err(LicenseError::InvalidTerms);
        }
        let record = self
            .records
            .get_mut(request_id)
            .ok_or_else(|| LicenseError::NotFound(request_id.to_string()))?;
        if record.status.is_terminal() {
            return Err(LicenseError::AlreadyDecided(request_id.to_string()));
        }
        record.status = SyncLicenseStatus::CounterOffer(terms);
        Ok(())
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Return the current status of a request.
    ///
    /// Returns `None` if no request with that ID exists.
    #[must_use]
    pub fn status(&self, request_id: &str) -> Option<&SyncLicenseStatus> {
        self.records.get(request_id).map(|r| &r.status)
    }

    /// Return references to the approved [`SyncLicenseTerm`]s for all approved
    /// requests where `music_id` matches the original request's music ID.
    #[must_use]
    pub fn active_licenses_for_music(&self, music_id: &str) -> Vec<&SyncLicenseTerm> {
        self.records
            .values()
            .filter(|r| r.request.music_id == music_id)
            .filter_map(|r| {
                if let SyncLicenseStatus::Approved(ref terms) = r.status {
                    Some(terms)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return references to all requests for a given music ID, regardless of
    /// status.  Useful for auditing the full history.
    #[must_use]
    pub fn requests_for_music(&self, music_id: &str) -> Vec<(&str, &SyncLicenseStatus)> {
        self.records
            .iter()
            .filter(|(_, r)| r.request.music_id == music_id)
            .map(|(id, r)| (id.as_str(), &r.status))
            .collect()
    }

    /// Number of requests currently tracked.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Return `true` when no requests have been submitted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_terms() -> SyncLicenseTerm {
        SyncLicenseTerm {
            license_type: SyncLicenseType::Digital,
            territory: "US".to_string(),
            duration_years: 3,
            exclusive: false,
            fee_usd: 2500.0,
            restrictions: vec![],
        }
    }

    fn trailer_terms() -> SyncLicenseTerm {
        SyncLicenseTerm {
            license_type: SyncLicenseType::Trailer,
            territory: "WORLD".to_string(),
            duration_years: 1,
            exclusive: true,
            fee_usd: 10_000.0,
            restrictions: vec!["no political advertising".to_string()],
        }
    }

    fn make_request(music_id: &str, video_id: &str) -> SyncLicenseRequest {
        SyncLicenseRequest {
            music_id: music_id.to_string(),
            video_id: video_id.to_string(),
            usage_context: "background score".to_string(),
            requested_terms: default_terms(),
        }
    }

    // ── submit → Pending ──────────────────────────────────────────────────────

    #[test]
    fn test_submit_returns_pending_status() {
        let mut mgr = SyncLicenseManager::new();
        let id = mgr.submit_request(make_request("music-1", "vid-1"));
        assert_eq!(mgr.status(&id), Some(&SyncLicenseStatus::Pending));
    }

    #[test]
    fn test_submit_generates_unique_ids_for_same_music_and_video() {
        let mut mgr = SyncLicenseManager::new();
        let id1 = mgr.submit_request(make_request("music-1", "vid-1"));
        let id2 = mgr.submit_request(make_request("music-1", "vid-1"));
        assert_ne!(id1, id2);
    }

    // ── approve ───────────────────────────────────────────────────────────────

    #[test]
    fn test_approve_changes_status_to_approved() {
        let mut mgr = SyncLicenseManager::new();
        let id = mgr.submit_request(make_request("music-1", "vid-1"));
        mgr.approve(&id, default_terms()).unwrap();
        assert!(matches!(
            mgr.status(&id),
            Some(SyncLicenseStatus::Approved(_))
        ));
    }

    #[test]
    fn test_approve_not_found_returns_error() {
        let mut mgr = SyncLicenseManager::new();
        let err = mgr.approve("nonexistent", default_terms()).unwrap_err();
        assert!(matches!(err, LicenseError::NotFound(_)));
    }

    #[test]
    fn test_approve_after_rejection_returns_already_decided() {
        let mut mgr = SyncLicenseManager::new();
        let id = mgr.submit_request(make_request("music-2", "vid-2"));
        mgr.reject(&id, "fee too low".to_string()).unwrap();
        let err = mgr.approve(&id, default_terms()).unwrap_err();
        assert!(matches!(err, LicenseError::AlreadyDecided(_)));
    }

    // ── reject ────────────────────────────────────────────────────────────────

    #[test]
    fn test_reject_changes_status_to_rejected_with_reason() {
        let mut mgr = SyncLicenseManager::new();
        let id = mgr.submit_request(make_request("music-3", "vid-3"));
        mgr.reject(&id, "rights not available in territory".to_string())
            .unwrap();
        assert!(matches!(
            mgr.status(&id),
            Some(SyncLicenseStatus::Rejected { reason }) if reason.contains("territory")
        ));
    }

    #[test]
    fn test_reject_after_approval_returns_already_decided() {
        let mut mgr = SyncLicenseManager::new();
        let id = mgr.submit_request(make_request("music-4", "vid-4"));
        mgr.approve(&id, default_terms()).unwrap();
        let err = mgr.reject(&id, "changed mind".to_string()).unwrap_err();
        assert!(matches!(err, LicenseError::AlreadyDecided(_)));
    }

    // ── counter_offer ─────────────────────────────────────────────────────────

    #[test]
    fn test_counter_offer_changes_status() {
        let mut mgr = SyncLicenseManager::new();
        let id = mgr.submit_request(make_request("music-5", "vid-5"));
        mgr.counter_offer(&id, trailer_terms()).unwrap();
        assert!(matches!(
            mgr.status(&id),
            Some(SyncLicenseStatus::CounterOffer(_))
        ));
    }

    #[test]
    fn test_approve_counter_offer_succeeds() {
        let mut mgr = SyncLicenseManager::new();
        let id = mgr.submit_request(make_request("music-6", "vid-6"));
        mgr.counter_offer(&id, trailer_terms()).unwrap();
        // The counter-offer can be approved.
        mgr.approve(&id, trailer_terms()).unwrap();
        assert!(matches!(
            mgr.status(&id),
            Some(SyncLicenseStatus::Approved(_))
        ));
    }

    #[test]
    fn test_counter_offer_after_approval_returns_already_decided() {
        let mut mgr = SyncLicenseManager::new();
        let id = mgr.submit_request(make_request("music-7", "vid-7"));
        mgr.approve(&id, default_terms()).unwrap();
        let err = mgr.counter_offer(&id, trailer_terms()).unwrap_err();
        assert!(matches!(err, LicenseError::AlreadyDecided(_)));
    }

    // ── active_licenses_for_music ─────────────────────────────────────────────

    #[test]
    fn test_active_licenses_for_music_returns_only_approved() {
        let mut mgr = SyncLicenseManager::new();
        let id1 = mgr.submit_request(make_request("music-8", "vid-1"));
        let id2 = mgr.submit_request(make_request("music-8", "vid-2"));
        let _id3 = mgr.submit_request(make_request("music-8", "vid-3"));

        mgr.approve(&id1, default_terms()).unwrap();
        mgr.reject(&id2, "budget".to_string()).unwrap();
        // id3 stays Pending

        let active = mgr.active_licenses_for_music("music-8");
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].territory, "US");
    }

    #[test]
    fn test_active_licenses_for_music_empty_when_none_approved() {
        let mut mgr = SyncLicenseManager::new();
        let _id = mgr.submit_request(make_request("music-9", "vid-9"));
        let active = mgr.active_licenses_for_music("music-9");
        assert!(active.is_empty());
    }

    // ── invalid terms ─────────────────────────────────────────────────────────

    #[test]
    fn test_approve_with_invalid_terms_zero_duration_returns_error() {
        let mut mgr = SyncLicenseManager::new();
        let id = mgr.submit_request(make_request("music-10", "vid-10"));
        let bad_terms = SyncLicenseTerm {
            license_type: SyncLicenseType::Broadcast,
            territory: "US".to_string(),
            duration_years: 0, // invalid
            exclusive: false,
            fee_usd: 1000.0,
            restrictions: vec![],
        };
        let err = mgr.approve(&id, bad_terms).unwrap_err();
        assert_eq!(err, LicenseError::InvalidTerms);
    }

    #[test]
    fn test_counter_offer_with_empty_territory_returns_error() {
        let mut mgr = SyncLicenseManager::new();
        let id = mgr.submit_request(make_request("music-11", "vid-11"));
        let bad_terms = SyncLicenseTerm {
            license_type: SyncLicenseType::Theatrical,
            territory: String::new(), // invalid
            duration_years: 2,
            exclusive: false,
            fee_usd: 5000.0,
            restrictions: vec![],
        };
        let err = mgr.counter_offer(&id, bad_terms).unwrap_err();
        assert_eq!(err, LicenseError::InvalidTerms);
    }
}
