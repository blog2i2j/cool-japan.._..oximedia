//! Session-based recommendation engine.
//!
//! Recommends content to watch next based on the user's current session activity,
//! applying recency-weighted completion signals, genre overlap scoring, and
//! dislike penalties to produce an ordered list of candidate items.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// Type of interaction the user performed on an item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionEventType {
    /// The user started playing the item.
    Play,
    /// The user paused the item mid-playback.
    Pause,
    /// The user skipped forward without completing.
    Skip,
    /// The user watched the item to completion (≥ 90 % progress).
    Complete,
    /// The user explicitly liked the item.
    Like,
    /// The user explicitly disliked the item.
    Dislike,
}

/// A single interaction event recorded during a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEvent {
    /// Unique identifier of the content item.
    pub item_id: String,
    /// Unix timestamp (seconds) when the event occurred.
    pub timestamp_secs: u64,
    /// The type of interaction.
    pub event_type: SessionEventType,
    /// Playback position at the time of the event, expressed as a fraction 0.0–1.0.
    pub position_pct: f32,
}

/// Aggregated session context passed to the recommender.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionContext {
    /// Ordered list of interaction events in the current session.
    pub events: Vec<SessionEvent>,
    /// Optional user identifier (used for logging / personalisation hooks).
    pub user_id: Option<String>,
}

/// A content item available for recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogItem {
    /// Unique item identifier.
    pub id: String,
    /// Genre tags associated with the item.
    pub genres: Vec<String>,
    /// Total duration in seconds.
    pub duration_secs: u32,
    /// Global popularity score in the range 0.0–1.0.
    pub popularity_score: f32,
}

/// The scored recommendation for a single catalog item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionScore {
    /// Identifier of the recommended item.
    pub item_id: String,
    /// Composite score (higher = more recommended).
    pub score: f32,
    /// Human-readable explanation fragments.
    pub reasons: Vec<String>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Decay factor applied per second of age (λ in e^{-λt}).
const RECENCY_LAMBDA: f64 = 1.0 / 1800.0; // half-life ≈ 20 min

/// Fraction of playback considered a "completion" signal even without a
/// [`SessionEventType::Complete`] event.
const COMPLETION_THRESHOLD: f32 = 0.85;

/// Multiplier applied to genre scores for items the user completed.
const COMPLETE_BOOST: f32 = 1.5;

/// Penalty multiplier applied to genre scores for items the user disliked.
const DISLIKE_PENALTY: f32 = 0.25;

/// Penalty applied when the candidate itself was disliked (subtracted from score).
const DIRECT_DISLIKE_PENALTY: f32 = 0.6;

/// Score added per overlapping genre with a positively-engaged item.
const GENRE_OVERLAP_BASE: f32 = 0.15;

/// Computes the Jaccard overlap coefficient between two genre slices (returns 0 if both empty).
fn genre_jaccard(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.iter().filter(|g| b.contains(g)).count();
    let union = {
        let mut combined: Vec<&String> = a.iter().collect();
        for g in b {
            if !combined.contains(&g) {
                combined.push(g);
            }
        }
        combined.len()
    };
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Computes a recency weight for an event relative to the most recent event
/// timestamp in the session.  Returns a value in (0, 1].
fn recency_weight(event_ts: u64, latest_ts: u64) -> f32 {
    let age_secs = latest_ts.saturating_sub(event_ts) as f64;
    f64::exp(-RECENCY_LAMBDA * age_secs) as f32
}

// ──────────────────────────────────────────────────────────────────────────────
// Per-item session signals
// ──────────────────────────────────────────────────────────────────────────────

/// Aggregated signals extracted from session events for one item.
#[derive(Debug, Default)]
struct ItemSignal {
    /// Recency-weighted positive engagement (play / complete / like).
    positive_weight: f32,
    /// Raw dislike signal (latest position_pct of a dislike event).
    dislike_weight: f32,
    /// Whether the item was ever completed or reached the completion threshold.
    completed: bool,
    /// Whether the item was explicitly liked.
    liked: bool,
    /// Whether the item was explicitly disliked.
    disliked: bool,
    /// Genres of this item as seen during the session (may be empty if not in catalog).
    genres: Vec<String>,
}

/// Derives per-item signals from a session context, optionally enriching genres
/// from the catalog.
fn extract_signals(
    context: &SessionContext,
    catalog_map: &HashMap<&str, &CatalogItem>,
) -> HashMap<String, ItemSignal> {
    let latest_ts = context
        .events
        .iter()
        .map(|e| e.timestamp_secs)
        .max()
        .unwrap_or(0);

    let mut signals: HashMap<String, ItemSignal> = HashMap::new();

    for event in &context.events {
        let sig = signals.entry(event.item_id.clone()).or_default();

        // Enrich genres from catalog if available.
        if sig.genres.is_empty() {
            if let Some(item) = catalog_map.get(event.item_id.as_str()) {
                sig.genres = item.genres.clone();
            }
        }

        let w = recency_weight(event.timestamp_secs, latest_ts);

        match event.event_type {
            SessionEventType::Play | SessionEventType::Pause => {
                sig.positive_weight += w * event.position_pct;
            }
            SessionEventType::Skip => {
                // Skip at early position is neutral; late skip is mild positive.
                sig.positive_weight += w * event.position_pct * 0.5;
            }
            SessionEventType::Complete => {
                sig.completed = true;
                sig.positive_weight += w * 1.0;
            }
            SessionEventType::Like => {
                sig.liked = true;
                sig.positive_weight += w * 1.2;
            }
            SessionEventType::Dislike => {
                sig.disliked = true;
                sig.dislike_weight += w;
            }
        }

        // Implicit completion check.
        if event.position_pct >= COMPLETION_THRESHOLD {
            sig.completed = true;
        }
    }

    signals
}

// ──────────────────────────────────────────────────────────────────────────────
// SessionRecommender
// ──────────────────────────────────────────────────────────────────────────────

/// Recommends content based on the user's in-session behaviour.
///
/// The algorithm proceeds in three stages:
///
/// 1. **Signal extraction** — aggregate positive/negative engagement per viewed item.
/// 2. **Genre profile construction** — build a weighted map from genre → affinity, applying
///    completion boosts and dislike penalties.
/// 3. **Candidate scoring** — for each unseen catalog item compute:
///    - Genre affinity score (Jaccard × weight).
///    - Popularity fallback (dominant when session is empty).
///    - Direct-dislike penalty (subtracted if the candidate itself was disliked).
#[derive(Debug, Default)]
pub struct SessionRecommender;

impl SessionRecommender {
    /// Create a new `SessionRecommender`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Produce up to `n` recommendations given the current `context` and `catalog`.
    ///
    /// When the session contains no positive events the fallback is pure popularity ranking.
    /// Returns a `Vec<String>` of item IDs ordered by descending score.
    #[must_use]
    pub fn recommend(
        &self,
        context: &SessionContext,
        catalog: &[CatalogItem],
        n: usize,
    ) -> Vec<String> {
        if n == 0 {
            return Vec::new();
        }

        // Build fast lookup for catalog items.
        let catalog_map: HashMap<&str, &CatalogItem> =
            catalog.iter().map(|c| (c.id.as_str(), c)).collect();

        let signals = extract_signals(context, &catalog_map);

        // Build a genre-level affinity profile.
        let genre_affinity = self.build_genre_affinity(&signals);

        // Score every catalog item that was not already interacted with.
        let mut scores: Vec<SessionScore> = catalog
            .iter()
            .filter_map(|item| {
                let sig = signals.get(&item.id);
                // Skip items the user has already actively engaged with (prevent re-recommending
                // items that have a positive signal and were completed/liked), but keep
                // disliked items visible so the penalty can be measured in tests.
                if let Some(s) = sig {
                    if s.completed || s.liked {
                        return None;
                    }
                }
                Some(self.score_item(item, sig, &genre_affinity, genre_affinity.is_empty()))
            })
            .collect();

        // Sort descending by score.
        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(n);
        scores.into_iter().map(|s| s.item_id).collect()
    }

    /// Produce scored recommendations with reason strings.
    #[must_use]
    pub fn recommend_scored(
        &self,
        context: &SessionContext,
        catalog: &[CatalogItem],
        n: usize,
    ) -> Vec<SessionScore> {
        if n == 0 {
            return Vec::new();
        }

        let catalog_map: HashMap<&str, &CatalogItem> =
            catalog.iter().map(|c| (c.id.as_str(), c)).collect();
        let signals = extract_signals(context, &catalog_map);
        let genre_affinity = self.build_genre_affinity(&signals);
        let popularity_fallback = genre_affinity.is_empty();

        let mut scores: Vec<SessionScore> = catalog
            .iter()
            .filter_map(|item| {
                let sig = signals.get(&item.id);
                if let Some(s) = sig {
                    if s.completed || s.liked {
                        return None;
                    }
                }
                Some(self.score_item(item, sig, &genre_affinity, popularity_fallback))
            })
            .collect();

        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(n);
        scores
    }

    /// Build a weighted genre affinity map from session signals.
    fn build_genre_affinity(&self, signals: &HashMap<String, ItemSignal>) -> HashMap<String, f32> {
        let mut affinity: HashMap<String, f32> = HashMap::new();

        for sig in signals.values() {
            if sig.disliked {
                // Subtract genre weight for disliked items.
                let weight = sig.dislike_weight * DISLIKE_PENALTY;
                for genre in &sig.genres {
                    *affinity.entry(genre.clone()).or_insert(0.0) -= weight;
                }
            } else {
                let mut weight = sig.positive_weight;
                if sig.completed {
                    weight *= COMPLETE_BOOST;
                }
                if sig.liked {
                    weight *= 1.3;
                }
                for genre in &sig.genres {
                    *affinity.entry(genre.clone()).or_insert(0.0) += weight * GENRE_OVERLAP_BASE;
                }
            }
        }

        affinity
    }

    /// Score a single candidate catalog item.
    fn score_item(
        &self,
        item: &CatalogItem,
        sig: Option<&ItemSignal>,
        genre_affinity: &HashMap<String, f32>,
        popularity_fallback: bool,
    ) -> SessionScore {
        let mut reasons: Vec<String> = Vec::new();

        let base_score = if popularity_fallback {
            // No session signal — use popularity directly.
            reasons.push(format!("popularity_fallback:{:.2}", item.popularity_score));
            item.popularity_score
        } else {
            // Genre affinity contribution.
            let mut genre_score = 0.0_f32;
            for genre in &item.genres {
                if let Some(&aff) = genre_affinity.get(genre) {
                    genre_score += aff;
                }
            }
            // Normalise by genre count to avoid favouring items with many genres.
            if !item.genres.is_empty() {
                genre_score /= item.genres.len() as f32;
            }
            if genre_score > 0.0 {
                reasons.push(format!("genre_affinity:{genre_score:.3}"));
            }

            // Popularity as a tie-breaker (weighted at 20 %).
            let pop_contribution = item.popularity_score * 0.2;
            if pop_contribution > 0.0 {
                reasons.push(format!("popularity:{:.2}", item.popularity_score));
            }
            genre_score + pop_contribution
        };

        // Apply direct dislike penalty if this specific item was disliked.
        let mut score = base_score;
        if let Some(s) = sig {
            if s.disliked {
                score -= DIRECT_DISLIKE_PENALTY;
                reasons.push("direct_dislike_penalty".to_string());
            }
        }

        SessionScore {
            item_id: item.id.clone(),
            score,
            reasons,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_catalog() -> Vec<CatalogItem> {
        vec![
            CatalogItem {
                id: "action_1".to_string(),
                genres: vec!["action".to_string(), "thriller".to_string()],
                duration_secs: 5400,
                popularity_score: 0.8,
            },
            CatalogItem {
                id: "action_2".to_string(),
                genres: vec!["action".to_string()],
                duration_secs: 4800,
                popularity_score: 0.6,
            },
            CatalogItem {
                id: "comedy_1".to_string(),
                genres: vec!["comedy".to_string()],
                duration_secs: 3600,
                popularity_score: 0.9,
            },
            CatalogItem {
                id: "drama_1".to_string(),
                genres: vec!["drama".to_string()],
                duration_secs: 6000,
                popularity_score: 0.5,
            },
            CatalogItem {
                id: "action_3".to_string(),
                genres: vec!["action".to_string(), "sci-fi".to_string()],
                duration_secs: 7200,
                popularity_score: 0.7,
            },
            CatalogItem {
                id: "already_liked".to_string(),
                genres: vec!["action".to_string()],
                duration_secs: 5000,
                popularity_score: 0.75,
            },
        ]
    }

    fn event(item_id: &str, ts: u64, evt: SessionEventType, pos: f32) -> SessionEvent {
        SessionEvent {
            item_id: item_id.to_string(),
            timestamp_secs: ts,
            event_type: evt,
            position_pct: pos,
        }
    }

    // 1. Empty session falls back to popularity ranking.
    #[test]
    fn test_empty_session_popularity_fallback() {
        let ctx = SessionContext::default();
        let catalog = make_catalog();
        let rec = SessionRecommender::new();
        let results = rec.recommend(&ctx, &catalog, 3);
        assert_eq!(results.len(), 3);
        // comedy_1 has popularity 0.9 — should be first.
        assert_eq!(results[0], "comedy_1");
    }

    // 2. N-limit respected even when catalog is larger.
    #[test]
    fn test_n_limit_respected() {
        let ctx = SessionContext::default();
        let catalog = make_catalog();
        let rec = SessionRecommender::new();
        let results = rec.recommend(&ctx, &catalog, 2);
        assert_eq!(results.len(), 2);
    }

    // 3. n = 0 returns empty list.
    #[test]
    fn test_zero_n_returns_empty() {
        let ctx = SessionContext::default();
        let catalog = make_catalog();
        let rec = SessionRecommender::new();
        assert!(rec.recommend(&ctx, &catalog, 0).is_empty());
    }

    // 4. Completing an action item boosts other action items.
    #[test]
    fn test_complete_boosts_similar_genre() {
        let ctx = SessionContext {
            events: vec![event("action_1", 1000, SessionEventType::Complete, 1.0)],
            user_id: None,
        };
        let catalog = make_catalog();
        let rec = SessionRecommender::new();
        let results = rec.recommend(&ctx, &catalog, 5);
        // action_1 itself is excluded (completed); action_2 and action_3 should rank ahead of comedy_1.
        let action_pos: Vec<usize> = results
            .iter()
            .enumerate()
            .filter(|(_, id)| id.starts_with("action"))
            .map(|(i, _)| i)
            .collect();
        let comedy_pos = results.iter().position(|id| id == "comedy_1");
        assert!(
            action_pos.iter().any(|&p| comedy_pos.map_or(true, |c| p < c)),
            "action items should rank before comedy after completing an action item, got: {results:?}"
        );
    }

    // 5. Liked item is excluded from results (not re-recommended).
    #[test]
    fn test_liked_item_excluded() {
        let ctx = SessionContext {
            events: vec![event("already_liked", 1000, SessionEventType::Like, 0.5)],
            user_id: None,
        };
        let catalog = make_catalog();
        let rec = SessionRecommender::new();
        let results = rec.recommend(&ctx, &catalog, 10);
        assert!(!results.contains(&"already_liked".to_string()));
    }

    // 6. Disliked item is penalised (scores lower than neutral items).
    #[test]
    fn test_dislike_penalises_item() {
        let ctx = SessionContext {
            events: vec![event("action_2", 1000, SessionEventType::Dislike, 0.1)],
            user_id: None,
        };
        let catalog = make_catalog();
        let rec = SessionRecommender::new();
        let scores = rec.recommend_scored(&ctx, &catalog, 10);
        // Find action_2 score vs comedy_1 score.
        let action2 = scores.iter().find(|s| s.item_id == "action_2");
        let comedy1 = scores.iter().find(|s| s.item_id == "comedy_1");
        if let (Some(a), Some(c)) = (action2, comedy1) {
            assert!(
                a.score < c.score,
                "disliked action_2 ({:.3}) should score below comedy_1 ({:.3})",
                a.score,
                c.score
            );
        }
    }

    // 7. Dislike of a genre reduces affinity for similar items.
    #[test]
    fn test_dislike_genre_penalises_similar() {
        // Dislike action_1; action_2 should score lower than comedy_1.
        let ctx = SessionContext {
            events: vec![event("action_1", 1000, SessionEventType::Dislike, 0.2)],
            user_id: None,
        };
        let catalog = make_catalog();
        let rec = SessionRecommender::new();
        let scores = rec.recommend_scored(&ctx, &catalog, 10);
        let action2 = scores.iter().find(|s| s.item_id == "action_2");
        let comedy1 = scores.iter().find(|s| s.item_id == "comedy_1");
        if let (Some(a), Some(c)) = (action2, comedy1) {
            // action_2 shares the "action" genre with disliked action_1.
            assert!(
                a.score < c.score,
                "action_2 score ({:.3}) should be below comedy_1 ({:.3}) after disliking action genre",
                a.score,
                c.score
            );
        }
    }

    // 8. Session with multiple positive events accumulates genre affinity correctly.
    #[test]
    fn test_multiple_positive_events_accumulate() {
        let ctx = SessionContext {
            events: vec![
                event("action_1", 1000, SessionEventType::Complete, 1.0),
                event("action_2", 2000, SessionEventType::Like, 0.8),
            ],
            user_id: Some("user_abc".to_string()),
        };
        let catalog = make_catalog();
        let rec = SessionRecommender::new();
        let results = rec.recommend(&ctx, &catalog, 5);
        // action_3 shares "action" genre with both completed/liked items.
        assert!(
            results.contains(&"action_3".to_string()),
            "action_3 should appear in results: {results:?}"
        );
        // action_3 should rank before drama_1 (no shared genre).
        let action3_pos = results.iter().position(|r| r == "action_3");
        let drama_pos = results.iter().position(|r| r == "drama_1");
        if let (Some(a), Some(d)) = (action3_pos, drama_pos) {
            assert!(a < d, "action_3 should rank before drama_1");
        }
    }

    // 9. Scored recommendations include reason strings.
    #[test]
    fn test_scored_reasons_present() {
        let ctx = SessionContext::default();
        let catalog = make_catalog();
        let rec = SessionRecommender::new();
        let scores = rec.recommend_scored(&ctx, &catalog, 3);
        for s in &scores {
            assert!(
                !s.reasons.is_empty(),
                "item {} should have at least one reason",
                s.item_id
            );
        }
    }

    // 10. Empty catalog returns empty results.
    #[test]
    fn test_empty_catalog_returns_empty() {
        let ctx = SessionContext::default();
        let rec = SessionRecommender::new();
        assert!(rec.recommend(&ctx, &[], 5).is_empty());
    }
}
