//! Full-text search implementation using Tantivy.

use crate::error::{SearchError, SearchResult};
use crate::index::builder::IndexDocument;
use crate::SearchResultItem;
use std::path::Path;
use std::sync::{Arc, RwLock};
use tantivy::{
    collector::TopDocs,
    directory::MmapDirectory,
    query::QueryParser,
    schema::{Facet, Schema, Term, Value, INDEXED, STORED, STRING, TEXT},
    Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument,
};
use uuid::Uuid;

/// Text search index using Tantivy
pub struct TextSearchIndex {
    index: Index,
    reader: IndexReader,
    writer: Arc<RwLock<IndexWriter>>,
    schema: Schema,
    query_parser: QueryParser,
}

impl TextSearchIndex {
    /// Create a new text search index
    ///
    /// # Errors
    ///
    /// Returns an error if index creation fails
    pub fn new(index_path: &Path) -> SearchResult<Self> {
        // Build schema
        let mut schema_builder = Schema::builder();

        // ID fields
        schema_builder.add_text_field("asset_id", STRING | STORED);

        // Text fields for full-text search
        schema_builder.add_text_field("file_path", STRING | STORED);
        schema_builder.add_text_field("title", TEXT | STORED);
        schema_builder.add_text_field("description", TEXT | STORED);
        schema_builder.add_text_field("keywords", TEXT);
        schema_builder.add_text_field("categories", TEXT);
        schema_builder.add_text_field("transcript", TEXT);
        schema_builder.add_text_field("scene_tags", TEXT);
        schema_builder.add_text_field("detected_objects", TEXT);

        // Stored fields
        schema_builder.add_text_field("mime_type", STRING | STORED);
        schema_builder.add_text_field("format", STRING | STORED);
        schema_builder.add_text_field("codec", STRING | STORED);
        schema_builder.add_text_field("resolution", STRING | STORED);

        // Numeric fields
        schema_builder.add_i64_field("duration_ms", INDEXED | STORED);
        schema_builder.add_i64_field("file_size", INDEXED | STORED);
        schema_builder.add_i64_field("bitrate", INDEXED | STORED);
        schema_builder.add_f64_field("framerate", INDEXED | STORED);
        schema_builder.add_i64_field("created_at", INDEXED | STORED);
        schema_builder.add_i64_field("modified_at", INDEXED | STORED);

        // Facet fields
        schema_builder.add_facet_field("mime_type_facet", INDEXED);
        schema_builder.add_facet_field("format_facet", INDEXED);
        schema_builder.add_facet_field("codec_facet", INDEXED);
        schema_builder.add_facet_field("resolution_facet", INDEXED);

        let schema = schema_builder.build();

        // Create or open index
        let index = if index_path.exists() {
            Index::open(MmapDirectory::open(index_path).map_err(tantivy::TantivyError::from)?)?
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

        // Create query parser
        let query_parser = QueryParser::for_index(
            &index,
            vec![
                schema.get_field("title")?,
                schema.get_field("description")?,
                schema.get_field("keywords")?,
                schema.get_field("categories")?,
                schema.get_field("transcript")?,
                schema.get_field("scene_tags")?,
                schema.get_field("detected_objects")?,
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

    /// Add a document to the index
    ///
    /// # Errors
    ///
    /// Returns an error if indexing fails
    pub fn add_document(&self, doc: &IndexDocument) -> SearchResult<()> {
        let mut tantivy_doc = TantivyDocument::new();

        // Add ID
        tantivy_doc.add_text(self.schema.get_field("asset_id")?, doc.asset_id.to_string());

        // Add text fields
        tantivy_doc.add_text(self.schema.get_field("file_path")?, &doc.file_path);

        if let Some(ref title) = doc.title {
            tantivy_doc.add_text(self.schema.get_field("title")?, title);
        }

        if let Some(ref description) = doc.description {
            tantivy_doc.add_text(self.schema.get_field("description")?, description);
        }

        // Add keywords
        let keywords_field = self.schema.get_field("keywords")?;
        for keyword in &doc.keywords {
            tantivy_doc.add_text(keywords_field, keyword);
        }

        // Add categories
        let categories_field = self.schema.get_field("categories")?;
        for category in &doc.categories {
            tantivy_doc.add_text(categories_field, category);
        }

        // Add transcript
        if let Some(ref transcript) = doc.transcript {
            tantivy_doc.add_text(self.schema.get_field("transcript")?, transcript);
        }

        // Add scene tags
        let scene_tags_field = self.schema.get_field("scene_tags")?;
        for tag in &doc.scene_tags {
            tantivy_doc.add_text(scene_tags_field, tag);
        }

        // Add detected objects
        let detected_objects_field = self.schema.get_field("detected_objects")?;
        for obj in &doc.detected_objects {
            tantivy_doc.add_text(detected_objects_field, obj);
        }

        // Add metadata fields
        if let Some(ref mime_type) = doc.mime_type {
            tantivy_doc.add_text(self.schema.get_field("mime_type")?, mime_type);
            tantivy_doc.add_facet(
                self.schema.get_field("mime_type_facet")?,
                Facet::from(&format!("/mime/{mime_type}")),
            );
        }

        if let Some(ref format) = doc.format {
            tantivy_doc.add_text(self.schema.get_field("format")?, format);
            tantivy_doc.add_facet(
                self.schema.get_field("format_facet")?,
                Facet::from(&format!("/format/{format}")),
            );
        }

        if let Some(ref codec) = doc.codec {
            tantivy_doc.add_text(self.schema.get_field("codec")?, codec);
            tantivy_doc.add_facet(
                self.schema.get_field("codec_facet")?,
                Facet::from(&format!("/codec/{codec}")),
            );
        }

        if let Some(ref resolution) = doc.resolution {
            tantivy_doc.add_text(self.schema.get_field("resolution")?, resolution);
            tantivy_doc.add_facet(
                self.schema.get_field("resolution_facet")?,
                Facet::from(&format!("/resolution/{resolution}")),
            );
        }

        // Add numeric fields
        if let Some(duration_ms) = doc.duration_ms {
            tantivy_doc.add_i64(self.schema.get_field("duration_ms")?, duration_ms);
        }

        if let Some(file_size) = doc.file_size {
            tantivy_doc.add_i64(self.schema.get_field("file_size")?, file_size);
        }

        if let Some(bitrate) = doc.bitrate {
            tantivy_doc.add_i64(self.schema.get_field("bitrate")?, bitrate);
        }

        if let Some(framerate) = doc.framerate {
            tantivy_doc.add_f64(self.schema.get_field("framerate")?, framerate);
        }

        tantivy_doc.add_i64(self.schema.get_field("created_at")?, doc.created_at);
        tantivy_doc.add_i64(self.schema.get_field("modified_at")?, doc.modified_at);

        // Add to index
        let writer = self
            .writer
            .write()
            .map_err(|_| SearchError::Other("Failed to acquire write lock".to_string()))?;

        writer.add_document(tantivy_doc)?;

        Ok(())
    }

    /// Search the index
    ///
    /// # Errors
    ///
    /// Returns an error if search fails
    pub fn search(&self, query_str: &str, limit: usize) -> SearchResult<Vec<SearchResultItem>> {
        let searcher = self.reader.searcher();

        // Parse query
        let query = self.query_parser.parse_query(query_str)?;

        // Execute search
        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit).order_by_score())?;

        // Pre-resolve fields before the loop to avoid repeated lookups
        let field_asset_id = self.schema.get_field("asset_id")?;
        let field_title = self.schema.get_field("title")?;
        let field_description = self.schema.get_field("description")?;
        let field_file_path = self.schema.get_field("file_path")?;
        let field_mime_type = self.schema.get_field("mime_type")?;
        let field_duration_ms = self.schema.get_field("duration_ms")?;
        let field_created_at = self.schema.get_field("created_at")?;

        // Convert results
        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;

            let asset_id_str = retrieved_doc
                .get_first(field_asset_id)
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let asset_id = Uuid::parse_str(asset_id_str)
                .map_err(|_| SearchError::Other("Invalid asset ID in index".to_string()))?;

            let title = retrieved_doc
                .get_first(field_title)
                .and_then(|v| v.as_str())
                .map(String::from);

            let description = retrieved_doc
                .get_first(field_description)
                .and_then(|v| v.as_str())
                .map(String::from);

            let file_path = retrieved_doc
                .get_first(field_file_path)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let mime_type = retrieved_doc
                .get_first(field_mime_type)
                .and_then(|v| v.as_str())
                .map(String::from);

            let duration_ms = retrieved_doc
                .get_first(field_duration_ms)
                .and_then(|v| v.as_i64());

            let created_at = retrieved_doc
                .get_first(field_created_at)
                .and_then(|v| v.as_i64())
                .unwrap_or(0);

            results.push(SearchResultItem {
                asset_id,
                score,
                title,
                description,
                file_path,
                mime_type,
                duration_ms,
                created_at,
                modified_at: None,
                file_size: None,
                matched_fields: vec![],
                thumbnail_url: None,
            });
        }

        Ok(results)
    }

    /// Commit pending changes
    ///
    /// # Errors
    ///
    /// Returns an error if commit fails
    pub fn commit(&self) -> SearchResult<()> {
        let mut writer = self
            .writer
            .write()
            .map_err(|_| SearchError::Other("Failed to acquire write lock".to_string()))?;

        writer.commit()?;

        Ok(())
    }

    /// Delete a document
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails
    pub fn delete(&self, asset_id: Uuid) -> SearchResult<()> {
        let term = Term::from_field_text(self.schema.get_field("asset_id")?, &asset_id.to_string());

        let writer = self
            .writer
            .write()
            .map_err(|_| SearchError::Other("Failed to acquire write lock".to_string()))?;

        writer.delete_term(term);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_result_item() {
        let item = SearchResultItem {
            asset_id: Uuid::new_v4(),
            score: 1.0,
            title: Some("Test".to_string()),
            description: None,
            file_path: "/test.mp4".to_string(),
            mime_type: Some("video/mp4".to_string()),
            duration_ms: Some(1000),
            created_at: 0,
            modified_at: None,
            file_size: None,
            matched_fields: vec![],
            thumbnail_url: None,
        };

        assert_eq!(item.score, 1.0);
    }
}
