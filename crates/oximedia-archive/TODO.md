# oximedia-archive TODO

## Current Status
- 30 source files across modules: `archive_verify`, `asset_manifest`, `audit_trail`, `batch_archive`, `catalog`, `catalog_export`, `checksum` (sqlite-gated), `dedup_archive`, `fixity` (sqlite-gated), `format_registry`, `indexing`, `ingest_log`, `integrity_scan`, `migration`, `preservation`, `preservation_policy`, `quarantine` (sqlite-gated), `report` (sqlite-gated), `restore_plan`, `retention_schedule`, `search_index`, `split_archive`, `streaming_compress`, `tape`, `validate`, `version_history`
- Multi-algorithm checksumming (BLAKE3, SHA-256, MD5, CRC32)
- Async `ArchiveVerifier` with SQLite-backed fixity checking and PREMIS event logging
- Parallel verification, auto-quarantine, BagIt support
- Dependencies: blake3, sha2, md-5, crc32fast, sqlx (optional), chrono, tokio, rayon, regex, csv

## Enhancements
- [ ] Implement true parallel file verification in `verify_files` using tokio/rayon (currently sequential await loop)
- [ ] Add incremental checksumming in `checksum` module (resume interrupted verification)
- [ ] Implement configurable quarantine policies in `quarantine` (max size, auto-cleanup after N days)
- [ ] Add `catalog_export` CSV and JSON export for archive manifests
- [x] Implement hierarchical catalog organization in `catalog` (collections, sub-collections)
- [x] Add file format identification (magic bytes) in `validate` beyond extension checking
- [ ] Implement `search_index` full-text search over metadata fields
- [x] Add `retention_schedule` enforcement: automatic deletion/archival when retention period expires
- [x] Improve `migration` module with dry-run support and rollback capability
- [x] Add sidecar file generation with human-readable checksum manifests

## New Features
- [ ] Add cloud storage backend support (S3-compatible) for `archive_verify` remote verification
- [ ] Implement LTFS (Linear Tape File System) metadata support in `tape` module
- [ ] Add BagIt profile validation (verify bag meets specific profile requirements)
- [ ] Implement media-specific validation in `validate` (check MKV/FLAC/PNG structure integrity)
- [ ] Add deduplication reporting in `dedup_archive` (space savings summary, duplicate file list)
- [ ] Implement notification system for fixity check failures (webhook/email integration points)
- [x] Add archive health dashboard data generation (summary stats, trend over time)
- [x] Implement `split_archive` with configurable split strategies (by size, date, collection)

## Performance
- [ ] Use memory-mapped I/O for checksum computation on large files in `checksum`
- [ ] Implement streaming hash computation (process file in chunks) to reduce memory footprint
- [x] Add parallel checksum computation (compute BLAKE3 + SHA-256 + CRC32 simultaneously per file)
- [ ] Optimize `integrity_scan` with file modification time checks to skip unchanged files
- [ ] Use connection pooling and batch SQL inserts in `fixity` for high-volume verification
- [ ] Add `streaming_compress` with configurable buffer sizes for throughput tuning

## Testing
- [ ] Add round-trip test: ingest file -> compute checksums -> verify -> confirm match
- [ ] Test `quarantine` workflow: corrupt file -> auto-detect -> quarantine -> restore
- [ ] Test `fixity` scheduled checking with mock clock (verify interval enforcement)
- [ ] Add test for `batch_archive` with 100+ synthetic files
- [ ] Test `catalog` operations: add/remove/search/export
- [ ] Test `version_history` with multiple versions of the same file (verify diff tracking)

## Documentation
- [ ] Document sqlite vs non-sqlite feature gating and which modules require database
- [ ] Add OAIS reference model diagram mapping modules to OAIS functional entities
- [ ] Document preservation workflow: ingest -> checksum -> validate -> catalog -> fixity schedule
