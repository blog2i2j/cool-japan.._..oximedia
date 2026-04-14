//! Search engine implementation using Tantivy
//!
//! Provides full-text search capabilities:
//! - Full-text search across asset metadata
//! - Faceted search
//! - Date range queries
//! - Advanced query language
//! - Search result ranking

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, RwLock};
use tantivy::{
    collector::{Count, TopDocs},
    directory::MmapDirectory,
    query::{BooleanQuery, FuzzyTermQuery, Occur, Query, QueryParser, RangeQuery, TermQuery},
    schema::*,
    Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument,
};
use uuid::Uuid;

use crate::{MamError, Result};

/// Search engine powered by Tantivy
pub struct SearchEngine {
    #[allow(dead_code)]
    index: Index,
    reader: IndexReader,
    writer: Arc<RwLock<IndexWriter>>,
    schema: Schema,
    query_parser: QueryParser,
}

/// Search query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Full-text search query
    pub query: String,
    /// Filters to apply
    pub filters: SearchFilters,
    /// Number of results to return
    pub limit: usize,
    /// Offset for pagination
    pub offset: usize,
    /// Sort by field
    pub sort_by: Option<SortField>,
    /// Sort order
    pub sort_order: SortOrder,
}

/// Search filters
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFilters {
    pub mime_types: Vec<String>,
    pub keywords: Vec<String>,
    pub categories: Vec<String>,
    pub min_duration: Option<i64>,
    pub max_duration: Option<i64>,
    pub min_width: Option<i32>,
    pub max_width: Option<i32>,
    pub min_height: Option<i32>,
    pub max_height: Option<i32>,
    pub date_from: Option<i64>,
    pub date_to: Option<i64>,
}

/// Sort field
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SortField {
    Relevance,
    CreatedAt,
    UpdatedAt,
    Duration,
    FileSize,
    Title,
}

/// Sort order
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SortOrder {
    Ascending,
    Descending,
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub asset_id: Uuid,
    pub score: f32,
    pub title: Option<String>,
    pub description: Option<String>,
    pub filename: String,
    pub mime_type: Option<String>,
    pub created_at: i64,
}

/// Search results with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    pub results: Vec<SearchResult>,
    pub total: usize,
    pub limit: usize,
    pub offset: usize,
    pub facets: SearchFacets,
}

/// Faceted search results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFacets {
    pub mime_types: Vec<FacetCount>,
    pub keywords: Vec<FacetCount>,
    pub categories: Vec<FacetCount>,
}

/// Facet count
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetCount {
    pub value: String,
    pub count: usize,
}

/// Document to index
#[derive(Debug, Clone)]
pub struct IndexDocument {
    pub asset_id: Uuid,
    pub filename: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub keywords: Vec<String>,
    pub categories: Vec<String>,
    pub mime_type: Option<String>,
    pub duration_ms: Option<i64>,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub file_size: Option<i64>,
    pub created_at: i64,
    pub updated_at: i64,
}

impl SearchEngine {
    /// Create a new search engine
    ///
    /// # Errors
    ///
    /// Returns an error if index creation fails
    pub fn new(index_path: &str) -> Result<Self> {
        // Build schema
        let mut schema_builder = Schema::builder();

        // ID field (stored)
        schema_builder.add_text_field("asset_id", STRING | STORED);

        // Text fields for full-text search
        schema_builder.add_text_field("filename", TEXT | STORED);
        schema_builder.add_text_field("title", TEXT | STORED);
        schema_builder.add_text_field("description", TEXT | STORED);
        schema_builder.add_text_field("keywords", TEXT);
        schema_builder.add_text_field("categories", TEXT);

        // Facet fields
        schema_builder.add_facet_field("mime_type_facet", INDEXED);
        schema_builder.add_facet_field("keyword_facet", INDEXED);
        schema_builder.add_facet_field("category_facet", INDEXED);

        // Stored fields
        schema_builder.add_text_field("mime_type", STRING | STORED);

        // Numeric fields for filtering
        schema_builder.add_i64_field("duration_ms", INDEXED | STORED);
        schema_builder.add_i64_field("width", INDEXED);
        schema_builder.add_i64_field("height", INDEXED);
        schema_builder.add_i64_field("file_size", INDEXED | STORED);
        schema_builder.add_i64_field("created_at", INDEXED | STORED);
        schema_builder.add_i64_field("updated_at", INDEXED);

        let schema = schema_builder.build();

        // Create or open index
        let index_path = Path::new(index_path);
        let index = if index_path.exists() {
            Index::open(MmapDirectory::open(index_path)?)?
        } else {
            std::fs::create_dir_all(index_path)?;
            Index::create_in_dir(index_path, schema.clone())?
        };

        // Create index writer
        let writer = index.writer(50_000_000)?; // 50MB heap

        // Create index reader
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        // Create query parser for full-text search
        let query_parser = QueryParser::for_index(
            &index,
            vec![
                schema
                    .get_field("filename")
                    .expect("invariant: schema field registered at construction"),
                schema
                    .get_field("title")
                    .expect("invariant: schema field registered at construction"),
                schema
                    .get_field("description")
                    .expect("invariant: schema field registered at construction"),
                schema
                    .get_field("keywords")
                    .expect("invariant: schema field registered at construction"),
                schema
                    .get_field("categories")
                    .expect("invariant: schema field registered at construction"),
            ],
        );

        Ok(Self {
            index,
            reader,
            writer: Arc::new(RwLock::new(writer)),
            schema,
            query_parser,
        })
    }

    /// Index a document
    ///
    /// # Errors
    ///
    /// Returns an error if indexing fails
    pub fn index_document(&self, doc: IndexDocument) -> Result<()> {
        let mut tantivy_doc = TantivyDocument::new();

        // Add fields
        tantivy_doc.add_text(
            self.schema
                .get_field("asset_id")
                .expect("invariant: schema field registered at construction"),
            doc.asset_id.to_string(),
        );

        tantivy_doc.add_text(
            self.schema
                .get_field("filename")
                .expect("invariant: schema field registered at construction"),
            &doc.filename,
        );

        if let Some(title) = &doc.title {
            tantivy_doc.add_text(
                self.schema
                    .get_field("title")
                    .expect("invariant: schema field registered at construction"),
                title,
            );
        }

        if let Some(description) = &doc.description {
            tantivy_doc.add_text(
                self.schema
                    .get_field("description")
                    .expect("invariant: schema field registered at construction"),
                description,
            );
        }

        // Add keywords
        for keyword in &doc.keywords {
            tantivy_doc.add_text(
                self.schema
                    .get_field("keywords")
                    .expect("invariant: schema field registered at construction"),
                keyword,
            );
            tantivy_doc.add_facet(
                self.schema
                    .get_field("keyword_facet")
                    .expect("invariant: schema field registered at construction"),
                Facet::from(&format!("/keyword/{keyword}")),
            );
        }

        // Add categories
        for category in &doc.categories {
            tantivy_doc.add_text(
                self.schema
                    .get_field("categories")
                    .expect("invariant: schema field registered at construction"),
                category,
            );
            tantivy_doc.add_facet(
                self.schema
                    .get_field("category_facet")
                    .expect("invariant: schema field registered at construction"),
                Facet::from(&format!("/category/{category}")),
            );
        }

        if let Some(mime_type) = &doc.mime_type {
            tantivy_doc.add_text(
                self.schema
                    .get_field("mime_type")
                    .expect("invariant: schema field registered at construction"),
                mime_type,
            );
            tantivy_doc.add_facet(
                self.schema
                    .get_field("mime_type_facet")
                    .expect("invariant: schema field registered at construction"),
                Facet::from(&format!("/mime/{mime_type}")),
            );
        }

        // Add numeric fields
        if let Some(duration) = doc.duration_ms {
            tantivy_doc.add_i64(
                self.schema
                    .get_field("duration_ms")
                    .expect("invariant: schema field registered at construction"),
                duration,
            );
        }

        if let Some(width) = doc.width {
            tantivy_doc.add_i64(
                self.schema
                    .get_field("width")
                    .expect("invariant: schema field registered at construction"),
                i64::from(width),
            );
        }

        if let Some(height) = doc.height {
            tantivy_doc.add_i64(
                self.schema
                    .get_field("height")
                    .expect("invariant: schema field registered at construction"),
                i64::from(height),
            );
        }

        if let Some(file_size) = doc.file_size {
            tantivy_doc.add_i64(
                self.schema
                    .get_field("file_size")
                    .expect("invariant: schema field registered at construction"),
                file_size,
            );
        }

        tantivy_doc.add_i64(
            self.schema
                .get_field("created_at")
                .expect("invariant: schema field registered at construction"),
            doc.created_at,
        );
        tantivy_doc.add_i64(
            self.schema
                .get_field("updated_at")
                .expect("invariant: schema field registered at construction"),
            doc.updated_at,
        );

        // Add to index
        let writer = self
            .writer
            .write()
            .map_err(|_| MamError::Internal("Failed to acquire write lock".to_string()))?;

        writer.add_document(tantivy_doc)?;

        Ok(())
    }

    /// Commit changes to index
    ///
    /// # Errors
    ///
    /// Returns an error if commit fails
    pub fn commit(&self) -> Result<()> {
        let mut writer = self
            .writer
            .write()
            .map_err(|_| MamError::Internal("Failed to acquire write lock".to_string()))?;

        writer.commit()?;

        Ok(())
    }

    /// Delete document by asset ID
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails
    pub fn delete_document(&self, asset_id: Uuid) -> Result<()> {
        let term = Term::from_field_text(
            self.schema
                .get_field("asset_id")
                .expect("invariant: schema field registered at construction"),
            &asset_id.to_string(),
        );

        let writer = self
            .writer
            .write()
            .map_err(|_| MamError::Internal("Failed to acquire write lock".to_string()))?;

        writer.delete_term(term);

        Ok(())
    }

    /// Search documents
    ///
    /// # Errors
    ///
    /// Returns an error if search fails
    pub fn search(&self, query: SearchQuery) -> Result<SearchResults> {
        let searcher = self.reader.searcher();

        // Build main query
        let text_query = self.query_parser.parse_query(&query.query)?;

        // Build filter queries
        let mut filter_queries: Vec<(Occur, Box<dyn Query>)> = Vec::new();

        // MIME type filters
        if !query.filters.mime_types.is_empty() {
            let mut mime_queries: Vec<(Occur, Box<dyn Query>)> = Vec::new();
            for mime_type in &query.filters.mime_types {
                let term = Term::from_field_text(
                    self.schema
                        .get_field("mime_type")
                        .expect("invariant: schema field registered at construction"),
                    mime_type,
                );
                mime_queries.push((
                    Occur::Should,
                    Box::new(TermQuery::new(term, Default::default())),
                ));
            }
            filter_queries.push((Occur::Must, Box::new(BooleanQuery::new(mime_queries))));
        }

        // Duration range
        if query.filters.min_duration.is_some() || query.filters.max_duration.is_some() {
            let field = self.schema.get_field("duration_ms")?;
            let min = query.filters.min_duration.unwrap_or(i64::MIN);
            let max = query.filters.max_duration.unwrap_or(i64::MAX);
            let range = RangeQuery::new(
                std::ops::Bound::Included(Term::from_field_i64(field, min)),
                std::ops::Bound::Included(Term::from_field_i64(field, max)),
            );
            filter_queries.push((Occur::Must, Box::new(range)));
        }

        // Width range
        if query.filters.min_width.is_some() || query.filters.max_width.is_some() {
            let field = self.schema.get_field("width")?;
            let min = i64::from(query.filters.min_width.unwrap_or(0));
            let max = i64::from(query.filters.max_width.unwrap_or(i32::MAX));
            let range = RangeQuery::new(
                std::ops::Bound::Included(Term::from_field_i64(field, min)),
                std::ops::Bound::Included(Term::from_field_i64(field, max)),
            );
            filter_queries.push((Occur::Must, Box::new(range)));
        }

        // Date range
        if query.filters.date_from.is_some() || query.filters.date_to.is_some() {
            let field = self.schema.get_field("created_at")?;
            let min = query.filters.date_from.unwrap_or(0);
            let max = query.filters.date_to.unwrap_or(i64::MAX);
            let range = RangeQuery::new(
                std::ops::Bound::Included(Term::from_field_i64(field, min)),
                std::ops::Bound::Included(Term::from_field_i64(field, max)),
            );
            filter_queries.push((Occur::Must, Box::new(range)));
        }

        // Combine all queries
        let mut all_queries = vec![(Occur::Must, text_query)];
        all_queries.extend(filter_queries);

        let final_query = BooleanQuery::new(all_queries);

        // Get total count
        let count_collector = Count;
        let total = searcher.search(&final_query, &count_collector)?;

        // Get top docs
        let top_docs_collector = TopDocs::with_limit(query.limit)
            .and_offset(query.offset)
            .order_by_score();
        let top_docs = searcher.search(&final_query, &top_docs_collector)?;

        // Convert to search results
        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;

            let asset_id_str = retrieved_doc
                .get_first(
                    self.schema
                        .get_field("asset_id")
                        .expect("invariant: schema field registered at construction"),
                )
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let asset_id = Uuid::parse_str(asset_id_str)
                .map_err(|_| MamError::Internal("Invalid asset ID in index".to_string()))?;

            let title = retrieved_doc
                .get_first(
                    self.schema
                        .get_field("title")
                        .expect("invariant: schema field registered at construction"),
                )
                .and_then(|v| v.as_str())
                .map(String::from);

            let description = retrieved_doc
                .get_first(
                    self.schema
                        .get_field("description")
                        .expect("invariant: schema field registered at construction"),
                )
                .and_then(|v| v.as_str())
                .map(String::from);

            let filename = retrieved_doc
                .get_first(
                    self.schema
                        .get_field("filename")
                        .expect("invariant: schema field registered at construction"),
                )
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let mime_type = retrieved_doc
                .get_first(
                    self.schema
                        .get_field("mime_type")
                        .expect("invariant: schema field registered at construction"),
                )
                .and_then(|v| v.as_str())
                .map(String::from);

            let created_at = retrieved_doc
                .get_first(
                    self.schema
                        .get_field("created_at")
                        .expect("invariant: schema field registered at construction"),
                )
                .and_then(|v| v.as_i64())
                .unwrap_or(0);

            results.push(SearchResult {
                asset_id,
                score,
                title,
                description,
                filename,
                mime_type,
                created_at,
            });
        }

        // Collect facets by querying the facet fields
        let facets = self.collect_facets(&final_query, &searcher)?;

        Ok(SearchResults {
            results,
            total,
            limit: query.limit,
            offset: query.offset,
            facets,
        })
    }

    /// Collect facets from a search query using Tantivy's facet collector.
    fn collect_facets(
        &self,
        query: &dyn tantivy::query::Query,
        searcher: &tantivy::Searcher,
    ) -> Result<SearchFacets> {
        use tantivy::collector::FacetCollector;

        // Collect mime_type facets
        let mut mime_facet_collector = FacetCollector::for_field("mime_type_facet");
        mime_facet_collector.add_facet("/mime");
        let mime_counts = searcher.search(query, &mime_facet_collector)?;
        let mime_types: Vec<FacetCount> = mime_counts
            .get("/mime")
            .map(|(facet, count)| FacetCount {
                value: facet
                    .to_path_string()
                    .trim_start_matches("/mime/")
                    .to_string(),
                count: count as usize,
            })
            .collect();

        // Collect keyword facets
        let mut kw_facet_collector = FacetCollector::for_field("keyword_facet");
        kw_facet_collector.add_facet("/keyword");
        let kw_counts = searcher.search(query, &kw_facet_collector)?;
        let keywords: Vec<FacetCount> = kw_counts
            .get("/keyword")
            .map(|(facet, count)| FacetCount {
                value: facet
                    .to_path_string()
                    .trim_start_matches("/keyword/")
                    .to_string(),
                count: count as usize,
            })
            .collect();

        // Collect category facets
        let mut cat_facet_collector = FacetCollector::for_field("category_facet");
        cat_facet_collector.add_facet("/category");
        let cat_counts = searcher.search(query, &cat_facet_collector)?;
        let categories: Vec<FacetCount> = cat_counts
            .get("/category")
            .map(|(facet, count)| FacetCount {
                value: facet
                    .to_path_string()
                    .trim_start_matches("/category/")
                    .to_string(),
                count: count as usize,
            })
            .collect();

        Ok(SearchFacets {
            mime_types,
            keywords,
            categories,
        })
    }

    /// Fuzzy search for suggestions
    ///
    /// # Errors
    ///
    /// Returns an error if search fails
    pub fn fuzzy_search(&self, term: &str, field: &str, limit: usize) -> Result<Vec<String>> {
        let searcher = self.reader.searcher();

        let field = self.schema.get_field(field)?;
        let term_obj = Term::from_field_text(field, term);

        let query = FuzzyTermQuery::new(term_obj, 2, true);

        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit).order_by_score())?;

        let mut suggestions = Vec::new();
        for (_score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(value) = doc.get_first(field).and_then(|v| v.as_str()) {
                suggestions.push(value.to_string());
            }
        }

        Ok(suggestions)
    }

    /// Check search engine health
    ///
    /// # Errors
    ///
    /// Returns an error if health check fails
    pub fn check_health(&self) -> Result<()> {
        // Try to create a searcher
        let _searcher = self.reader.searcher();
        Ok(())
    }

    /// Get index statistics
    ///
    /// # Errors
    ///
    /// Returns an error if getting stats fails
    pub fn get_statistics(&self) -> Result<IndexStatistics> {
        let searcher = self.reader.searcher();
        let segment_ids = searcher
            .segment_readers()
            .iter()
            .map(|r| format!("{:?}", r.segment_id()))
            .collect();

        Ok(IndexStatistics {
            num_docs: searcher.num_docs() as usize,
            num_segments: searcher.segment_readers().len(),
            segment_ids,
        })
    }
}

/// Index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStatistics {
    pub num_docs: usize,
    pub num_segments: usize,
    pub segment_ids: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_query_serialization() {
        let query = SearchQuery {
            query: "test".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            sort_by: Some(SortField::Relevance),
            sort_order: SortOrder::Descending,
        };

        let json = serde_json::to_string(&query).expect("should succeed in test");
        let deserialized: SearchQuery =
            serde_json::from_str(&json).expect("should succeed in test");

        assert_eq!(deserialized.query, "test");
        assert_eq!(deserialized.limit, 10);
    }

    #[test]
    fn test_facet_count() {
        let facet = FacetCount {
            value: "video/mp4".to_string(),
            count: 42,
        };

        assert_eq!(facet.count, 42);
    }
}
