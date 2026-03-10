#![allow(dead_code)]
//! Rights negotiation workflows and counter-offer tracking.
//!
//! Models the negotiation process between licensors and licensees,
//! tracking proposals, counter-offers, and final agreements with
//! a complete negotiation history.

use std::collections::HashMap;

/// Status of a negotiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NegotiationStatus {
    /// Negotiation has been initiated.
    Initiated,
    /// A proposal has been sent.
    ProposalSent,
    /// A counter-offer has been received.
    CounterOfferReceived,
    /// Both parties have agreed.
    Agreed,
    /// Negotiation was rejected by one party.
    Rejected,
    /// Negotiation has expired without agreement.
    Expired,
    /// Negotiation was cancelled.
    Cancelled,
}

impl NegotiationStatus {
    /// Return a human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            NegotiationStatus::Initiated => "initiated",
            NegotiationStatus::ProposalSent => "proposal_sent",
            NegotiationStatus::CounterOfferReceived => "counter_offer_received",
            NegotiationStatus::Agreed => "agreed",
            NegotiationStatus::Rejected => "rejected",
            NegotiationStatus::Expired => "expired",
            NegotiationStatus::Cancelled => "cancelled",
        }
    }

    /// Check if the negotiation is still active (can receive offers).
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            NegotiationStatus::Initiated
                | NegotiationStatus::ProposalSent
                | NegotiationStatus::CounterOfferReceived
        )
    }

    /// Check if the negotiation reached a final state.
    #[must_use]
    pub fn is_final(&self) -> bool {
        matches!(
            self,
            NegotiationStatus::Agreed
                | NegotiationStatus::Rejected
                | NegotiationStatus::Expired
                | NegotiationStatus::Cancelled
        )
    }
}

/// A single term in a negotiation (e.g., price, territory, duration).
#[derive(Debug, Clone)]
pub struct NegotiationTerm {
    /// Name of the term (e.g., "price", "territory", "duration_months").
    pub name: String,
    /// Proposed value.
    pub value: String,
    /// Whether this term is negotiable.
    pub negotiable: bool,
}

impl NegotiationTerm {
    /// Create a new negotiation term.
    #[must_use]
    pub fn new(name: &str, value: &str, negotiable: bool) -> Self {
        Self {
            name: name.to_string(),
            value: value.to_string(),
            negotiable,
        }
    }

    /// Create a fixed (non-negotiable) term.
    #[must_use]
    pub fn fixed(name: &str, value: &str) -> Self {
        Self::new(name, value, false)
    }

    /// Create a negotiable term.
    #[must_use]
    pub fn negotiable(name: &str, value: &str) -> Self {
        Self::new(name, value, true)
    }
}

/// A proposal or counter-offer in a negotiation.
#[derive(Debug, Clone)]
pub struct Offer {
    /// Unique offer identifier.
    pub offer_id: String,
    /// Who made this offer.
    pub from: String,
    /// Who this offer is directed to.
    pub to: String,
    /// Terms included in the offer.
    pub terms: Vec<NegotiationTerm>,
    /// Timestamp as ISO 8601 string.
    pub timestamp: String,
    /// Optional notes from the offeror.
    pub notes: String,
    /// Whether this is the initial proposal (true) or a counter-offer (false).
    pub is_initial: bool,
}

impl Offer {
    /// Create a new offer.
    #[must_use]
    pub fn new(offer_id: &str, from: &str, to: &str, is_initial: bool) -> Self {
        Self {
            offer_id: offer_id.to_string(),
            from: from.to_string(),
            to: to.to_string(),
            terms: Vec::new(),
            timestamp: String::new(),
            notes: String::new(),
            is_initial,
        }
    }

    /// Add a term to the offer.
    #[must_use]
    pub fn with_term(mut self, term: NegotiationTerm) -> Self {
        self.terms.push(term);
        self
    }

    /// Set notes.
    #[must_use]
    pub fn with_notes(mut self, notes: &str) -> Self {
        self.notes = notes.to_string();
        self
    }

    /// Set timestamp.
    #[must_use]
    pub fn with_timestamp(mut self, ts: &str) -> Self {
        self.timestamp = ts.to_string();
        self
    }

    /// Get a term by name.
    #[must_use]
    pub fn get_term(&self, name: &str) -> Option<&NegotiationTerm> {
        self.terms.iter().find(|t| t.name == name)
    }

    /// Get the number of terms.
    #[must_use]
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// List all negotiable term names.
    #[must_use]
    pub fn negotiable_terms(&self) -> Vec<&str> {
        self.terms
            .iter()
            .filter(|t| t.negotiable)
            .map(|t| t.name.as_str())
            .collect()
    }

    /// List all fixed term names.
    #[must_use]
    pub fn fixed_terms(&self) -> Vec<&str> {
        self.terms
            .iter()
            .filter(|t| !t.negotiable)
            .map(|t| t.name.as_str())
            .collect()
    }
}

/// A complete negotiation session between two parties.
#[derive(Debug, Clone)]
pub struct Negotiation {
    /// Unique negotiation identifier.
    pub negotiation_id: String,
    /// The licensor (rights holder).
    pub licensor: String,
    /// The licensee (rights seeker).
    pub licensee: String,
    /// Asset or content being negotiated.
    pub asset_id: String,
    /// Current status.
    pub status: NegotiationStatus,
    /// All offers exchanged (chronological order).
    offers: Vec<Offer>,
    /// Deadline as ISO 8601 string (empty = no deadline).
    pub deadline: String,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl Negotiation {
    /// Create a new negotiation.
    #[must_use]
    pub fn new(negotiation_id: &str, licensor: &str, licensee: &str, asset_id: &str) -> Self {
        Self {
            negotiation_id: negotiation_id.to_string(),
            licensor: licensor.to_string(),
            licensee: licensee.to_string(),
            asset_id: asset_id.to_string(),
            status: NegotiationStatus::Initiated,
            offers: Vec::new(),
            deadline: String::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set a deadline.
    #[must_use]
    pub fn with_deadline(mut self, deadline: &str) -> Self {
        self.deadline = deadline.to_string();
        self
    }

    /// Submit a proposal (initial offer from licensor to licensee).
    pub fn submit_proposal(&mut self, offer: Offer) {
        self.offers.push(offer);
        self.status = NegotiationStatus::ProposalSent;
    }

    /// Submit a counter-offer.
    pub fn submit_counter_offer(&mut self, offer: Offer) {
        self.offers.push(offer);
        self.status = NegotiationStatus::CounterOfferReceived;
    }

    /// Accept the current terms — move to Agreed.
    pub fn accept(&mut self) {
        if self.status.is_active() {
            self.status = NegotiationStatus::Agreed;
        }
    }

    /// Reject the negotiation.
    pub fn reject(&mut self) {
        if self.status.is_active() {
            self.status = NegotiationStatus::Rejected;
        }
    }

    /// Cancel the negotiation.
    pub fn cancel(&mut self) {
        if self.status.is_active() {
            self.status = NegotiationStatus::Cancelled;
        }
    }

    /// Mark the negotiation as expired.
    pub fn expire(&mut self) {
        if self.status.is_active() {
            self.status = NegotiationStatus::Expired;
        }
    }

    /// Get all offers.
    #[must_use]
    pub fn offers(&self) -> &[Offer] {
        &self.offers
    }

    /// Get the number of offers exchanged.
    #[must_use]
    pub fn offer_count(&self) -> usize {
        self.offers.len()
    }

    /// Get the latest offer, if any.
    #[must_use]
    pub fn latest_offer(&self) -> Option<&Offer> {
        self.offers.last()
    }

    /// Get the initial offer, if any.
    #[must_use]
    pub fn initial_offer(&self) -> Option<&Offer> {
        self.offers.first()
    }

    /// Check if the negotiation has a deadline.
    #[must_use]
    pub fn has_deadline(&self) -> bool {
        !self.deadline.is_empty()
    }

    /// Add metadata.
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Get metadata value.
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }
}

/// Registry for tracking multiple negotiations.
pub struct NegotiationTracker {
    /// All negotiations indexed by ID.
    negotiations: HashMap<String, Negotiation>,
}

impl NegotiationTracker {
    /// Create a new tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            negotiations: HashMap::new(),
        }
    }

    /// Register a negotiation.
    pub fn register(&mut self, negotiation: Negotiation) {
        self.negotiations
            .insert(negotiation.negotiation_id.clone(), negotiation);
    }

    /// Get a negotiation by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&Negotiation> {
        self.negotiations.get(id)
    }

    /// Get a mutable negotiation by ID.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut Negotiation> {
        self.negotiations.get_mut(id)
    }

    /// Find all active negotiations.
    #[must_use]
    pub fn find_active(&self) -> Vec<&Negotiation> {
        self.negotiations
            .values()
            .filter(|n| n.status.is_active())
            .collect()
    }

    /// Find negotiations by asset ID.
    #[must_use]
    pub fn find_by_asset(&self, asset_id: &str) -> Vec<&Negotiation> {
        self.negotiations
            .values()
            .filter(|n| n.asset_id == asset_id)
            .collect()
    }

    /// Find negotiations involving a specific party (as licensor or licensee).
    #[must_use]
    pub fn find_by_party(&self, party: &str) -> Vec<&Negotiation> {
        self.negotiations
            .values()
            .filter(|n| n.licensor == party || n.licensee == party)
            .collect()
    }

    /// Count all negotiations.
    #[must_use]
    pub fn count(&self) -> usize {
        self.negotiations.len()
    }

    /// Count active negotiations.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.negotiations
            .values()
            .filter(|n| n.status.is_active())
            .count()
    }
}

impl Default for NegotiationTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_negotiation_status_labels() {
        assert_eq!(NegotiationStatus::Initiated.label(), "initiated");
        assert_eq!(NegotiationStatus::Agreed.label(), "agreed");
        assert_eq!(NegotiationStatus::Cancelled.label(), "cancelled");
    }

    #[test]
    fn test_negotiation_status_active() {
        assert!(NegotiationStatus::Initiated.is_active());
        assert!(NegotiationStatus::ProposalSent.is_active());
        assert!(NegotiationStatus::CounterOfferReceived.is_active());
        assert!(!NegotiationStatus::Agreed.is_active());
        assert!(!NegotiationStatus::Rejected.is_active());
    }

    #[test]
    fn test_negotiation_status_final() {
        assert!(!NegotiationStatus::Initiated.is_final());
        assert!(NegotiationStatus::Agreed.is_final());
        assert!(NegotiationStatus::Rejected.is_final());
        assert!(NegotiationStatus::Expired.is_final());
        assert!(NegotiationStatus::Cancelled.is_final());
    }

    #[test]
    fn test_negotiation_term() {
        let fixed = NegotiationTerm::fixed("territory", "US");
        assert!(!fixed.negotiable);

        let neg = NegotiationTerm::negotiable("price", "10000");
        assert!(neg.negotiable);
    }

    #[test]
    fn test_offer_creation() {
        let offer = Offer::new("o1", "licensor", "licensee", true)
            .with_term(NegotiationTerm::negotiable("price", "50000"))
            .with_term(NegotiationTerm::fixed("territory", "US"))
            .with_notes("Initial proposal")
            .with_timestamp("2024-06-01T12:00:00Z");

        assert_eq!(offer.term_count(), 2);
        assert!(offer.is_initial);
        assert_eq!(offer.negotiable_terms(), vec!["price"]);
        assert_eq!(offer.fixed_terms(), vec!["territory"]);
    }

    #[test]
    fn test_offer_get_term() {
        let offer = Offer::new("o2", "a", "b", true)
            .with_term(NegotiationTerm::negotiable("price", "1000"));
        let term = offer.get_term("price");
        assert!(term.is_some());
        assert_eq!(
            term.expect("rights test operation should succeed").value,
            "1000"
        );
        assert!(offer.get_term("nonexistent").is_none());
    }

    #[test]
    fn test_negotiation_lifecycle() {
        let mut neg = Negotiation::new("n1", "Studio A", "Platform B", "movie-001");
        assert_eq!(neg.status, NegotiationStatus::Initiated);

        neg.submit_proposal(Offer::new("o1", "Studio A", "Platform B", true));
        assert_eq!(neg.status, NegotiationStatus::ProposalSent);

        neg.submit_counter_offer(Offer::new("o2", "Platform B", "Studio A", false));
        assert_eq!(neg.status, NegotiationStatus::CounterOfferReceived);

        neg.accept();
        assert_eq!(neg.status, NegotiationStatus::Agreed);
        assert_eq!(neg.offer_count(), 2);
    }

    #[test]
    fn test_negotiation_reject() {
        let mut neg = Negotiation::new("n2", "A", "B", "asset-1");
        neg.submit_proposal(Offer::new("o1", "A", "B", true));
        neg.reject();
        assert_eq!(neg.status, NegotiationStatus::Rejected);
        // Reject again should be no-op (already final)
        neg.reject();
        assert_eq!(neg.status, NegotiationStatus::Rejected);
    }

    #[test]
    fn test_negotiation_cancel() {
        let mut neg = Negotiation::new("n3", "A", "B", "asset-2");
        neg.cancel();
        assert_eq!(neg.status, NegotiationStatus::Cancelled);
    }

    #[test]
    fn test_negotiation_expire() {
        let mut neg =
            Negotiation::new("n4", "A", "B", "asset-3").with_deadline("2024-12-31T23:59:59Z");
        assert!(neg.has_deadline());
        neg.expire();
        assert_eq!(neg.status, NegotiationStatus::Expired);
    }

    #[test]
    fn test_negotiation_metadata() {
        let mut neg = Negotiation::new("n5", "A", "B", "asset-4");
        neg.set_metadata("category", "premium");
        assert_eq!(neg.get_metadata("category"), Some("premium"));
        assert_eq!(neg.get_metadata("missing"), None);
    }

    #[test]
    fn test_negotiation_offers() {
        let mut neg = Negotiation::new("n6", "A", "B", "asset-5");
        assert!(neg.latest_offer().is_none());
        assert!(neg.initial_offer().is_none());

        neg.submit_proposal(Offer::new("o1", "A", "B", true));
        neg.submit_counter_offer(Offer::new("o2", "B", "A", false));

        assert_eq!(
            neg.initial_offer()
                .expect("rights test operation should succeed")
                .offer_id,
            "o1"
        );
        assert_eq!(
            neg.latest_offer()
                .expect("rights test operation should succeed")
                .offer_id,
            "o2"
        );
    }

    #[test]
    fn test_tracker_basic() {
        let mut tracker = NegotiationTracker::new();
        tracker.register(Negotiation::new("n1", "A", "B", "asset-1"));
        assert_eq!(tracker.count(), 1);
        assert!(tracker.get("n1").is_some());
        assert!(tracker.get("missing").is_none());
    }

    #[test]
    fn test_tracker_find_active() {
        let mut tracker = NegotiationTracker::new();
        tracker.register(Negotiation::new("n1", "A", "B", "a1"));
        let mut agreed = Negotiation::new("n2", "C", "D", "a2");
        agreed.accept();
        tracker.register(agreed);

        assert_eq!(tracker.find_active().len(), 1);
        assert_eq!(tracker.active_count(), 1);
    }

    #[test]
    fn test_tracker_find_by_asset() {
        let mut tracker = NegotiationTracker::new();
        tracker.register(Negotiation::new("n1", "A", "B", "movie-1"));
        tracker.register(Negotiation::new("n2", "C", "D", "movie-1"));
        tracker.register(Negotiation::new("n3", "E", "F", "movie-2"));

        let results = tracker.find_by_asset("movie-1");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_tracker_find_by_party() {
        let mut tracker = NegotiationTracker::new();
        tracker.register(Negotiation::new("n1", "Alice", "Bob", "a1"));
        tracker.register(Negotiation::new("n2", "Carol", "Alice", "a2"));
        tracker.register(Negotiation::new("n3", "Dave", "Eve", "a3"));

        let results = tracker.find_by_party("Alice");
        assert_eq!(results.len(), 2);
    }
}
