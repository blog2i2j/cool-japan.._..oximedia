//! Relevance scoring using BM25 and TF-IDF.
//!
//! Implements the Okapi BM25 ranking algorithm (Robertson & Walker, 1994)
//! and classic TF-IDF scoring for search result relevance ranking.

use crate::SearchResultItem;

/// Per-term scoring context for BM25 computation.
#[derive(Debug, Clone)]
pub struct TermScoreContext {
    /// Raw term frequency in the document.
    pub term_freq: f32,
    /// Number of documents containing this term.
    pub doc_freq: usize,
    /// Total number of documents in the corpus.
    pub total_docs: usize,
    /// Length of the document in tokens.
    pub doc_length: usize,
}

/// Relevance scorer implementing Okapi BM25 and TF-IDF algorithms.
///
/// BM25 parameters:
/// - `k1` controls term frequency saturation (typical: 1.2-2.0)
/// - `b` controls document length normalisation (0.0 = none, 1.0 = full)
pub struct RelevanceScorer {
    /// BM25 k1 parameter (term frequency saturation).
    k1: f32,
    /// BM25 b parameter (document length normalisation).
    b: f32,
}

impl RelevanceScorer {
    /// Create a new relevance scorer with standard BM25 defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }

    /// Create a scorer with custom BM25 parameters.
    #[must_use]
    pub const fn with_params(k1: f32, b: f32) -> Self {
        Self { k1, b }
    }

    /// Compute BM25 IDF component for a term.
    ///
    /// Uses the Robertson-Sparck Jones IDF formula:
    /// `IDF = ln((N - df + 0.5) / (df + 0.5) + 1)`
    ///
    /// This formulation avoids negative IDF values for very common terms.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn bm25_idf(total_docs: usize, doc_freq: usize) -> f32 {
        if total_docs == 0 || doc_freq == 0 {
            return 0.0;
        }
        let n = total_docs as f32;
        let df = doc_freq as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Compute the BM25 score for a single term in a document.
    ///
    /// Full Okapi BM25 formula:
    /// `score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))`
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn bm25_term_score(&self, ctx: &TermScoreContext, avg_doc_length: f32) -> f32 {
        let idf = Self::bm25_idf(ctx.total_docs, ctx.doc_freq);
        let tf = ctx.term_freq;
        let dl = ctx.doc_length as f32;

        let normalised_tf =
            (tf * (self.k1 + 1.0)) / (tf + self.k1 * (1.0 - self.b + self.b * dl / avg_doc_length));

        idf * normalised_tf
    }

    /// Score search results using BM25.
    ///
    /// Each result's existing score is treated as a raw term frequency proxy.
    /// The score is replaced with a proper BM25 score using estimated
    /// corpus statistics derived from the result set itself.
    #[allow(clippy::cast_precision_loss)]
    pub fn score_bm25(&self, results: &mut [SearchResultItem], avg_doc_length: f32) {
        if results.is_empty() {
            return;
        }

        let total_docs = results.len().max(1);

        // Count how many results have a non-zero score (proxy for doc_freq).
        let doc_freq = results.iter().filter(|r| r.score > 0.0).count().max(1);

        let idf = Self::bm25_idf(total_docs, doc_freq);

        for result in results.iter_mut() {
            let tf = result.score;
            if tf <= 0.0 {
                result.score = 0.0;
                continue;
            }
            // Estimate doc_length from matched_fields count (proxy).
            let dl = (result.matched_fields.len() as f32 * 50.0).max(1.0);

            let normalised_tf = (tf * (self.k1 + 1.0))
                / (tf + self.k1 * (1.0 - self.b + self.b * dl / avg_doc_length));

            result.score = idf * normalised_tf;
        }
    }

    /// Apply TF-IDF scoring to search results.
    ///
    /// Uses logarithmic TF: `(1 + ln(tf))` and standard IDF: `ln(N / df)`.
    /// Each result's existing score is treated as raw term frequency.
    #[allow(clippy::cast_precision_loss)]
    pub fn score_tfidf(&self, results: &mut [SearchResultItem]) {
        if results.is_empty() {
            return;
        }

        let total_docs = results.len().max(1);
        let doc_freq = results.iter().filter(|r| r.score > 0.0).count().max(1);
        let idf = ((total_docs as f32) / (doc_freq as f32)).ln().max(0.0);

        for result in results.iter_mut() {
            let tf = result.score;
            if tf <= 0.0 {
                result.score = 0.0;
                continue;
            }
            // Logarithmic TF normalisation: 1 + ln(tf)
            let log_tf = 1.0 + tf.ln();
            result.score = log_tf * idf;
        }
    }

    /// Compute a combined BM25 + field-boost score for a result.
    ///
    /// This applies field-specific weights before BM25 scoring.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn combined_score(
        &self,
        term_freq: f32,
        doc_freq: usize,
        total_docs: usize,
        doc_length: usize,
        avg_doc_length: f32,
        field_boost: f32,
    ) -> f32 {
        let ctx = TermScoreContext {
            term_freq,
            doc_freq,
            total_docs,
            doc_length,
        };
        self.bm25_term_score(&ctx, avg_doc_length) * field_boost
    }
}

impl Default for RelevanceScorer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_result(score: f32, fields: Vec<String>) -> SearchResultItem {
        SearchResultItem {
            asset_id: Uuid::new_v4(),
            score,
            title: None,
            description: None,
            file_path: String::new(),
            mime_type: None,
            duration_ms: None,
            created_at: 0,
            matched_fields: fields,
            thumbnail_url: None,
        }
    }

    #[test]
    fn test_scorer_bm25_increases_relevant_scores() {
        let scorer = RelevanceScorer::new();
        let mut results = vec![make_result(1.0, vec!["title".to_string()])];

        scorer.score_bm25(&mut results, 100.0);
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn test_scorer_bm25_zero_tf_yields_zero() {
        let scorer = RelevanceScorer::new();
        let mut results = vec![make_result(0.0, vec![])];
        scorer.score_bm25(&mut results, 100.0);
        assert!((results[0].score - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_scorer_tfidf_produces_positive_score() {
        let scorer = RelevanceScorer::new();
        // Include a zero-TF result so that doc_freq < total_docs, giving IDF > 0.
        let mut results = vec![
            make_result(3.0, vec!["title".to_string()]),
            make_result(1.0, vec!["body".to_string()]),
            make_result(0.0, vec![]),
        ];
        scorer.score_tfidf(&mut results);
        assert!(results[0].score > 0.0);
        assert!(results[1].score > 0.0);
        // The zero-TF document should remain zero.
        assert!((results[2].score - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bm25_idf_common_term_low() {
        // When all docs contain the term, IDF is low but non-negative.
        let idf = RelevanceScorer::bm25_idf(100, 100);
        assert!(idf >= 0.0);
        assert!(idf < 1.0);
    }

    #[test]
    fn test_bm25_idf_rare_term_high() {
        let idf = RelevanceScorer::bm25_idf(1000, 1);
        assert!(idf > 5.0);
    }

    #[test]
    fn test_bm25_term_score_basic() {
        let scorer = RelevanceScorer::new();
        let ctx = TermScoreContext {
            term_freq: 3.0,
            doc_freq: 10,
            total_docs: 1000,
            doc_length: 200,
        };
        let score = scorer.bm25_term_score(&ctx, 150.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_custom_params() {
        let scorer = RelevanceScorer::with_params(2.0, 0.5);
        let ctx = TermScoreContext {
            term_freq: 1.0,
            doc_freq: 5,
            total_docs: 100,
            doc_length: 50,
        };
        let score = scorer.bm25_term_score(&ctx, 100.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_combined_score_with_boost() {
        let scorer = RelevanceScorer::new();
        let base = scorer.combined_score(2.0, 10, 1000, 100, 150.0, 1.0);
        let boosted = scorer.combined_score(2.0, 10, 1000, 100, 150.0, 2.0);
        assert!((boosted - base * 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_bm25_empty_results() {
        let scorer = RelevanceScorer::new();
        let mut results: Vec<SearchResultItem> = vec![];
        scorer.score_bm25(&mut results, 100.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_tfidf_empty_results() {
        let scorer = RelevanceScorer::new();
        let mut results: Vec<SearchResultItem> = vec![];
        scorer.score_tfidf(&mut results);
        assert!(results.is_empty());
    }
}
