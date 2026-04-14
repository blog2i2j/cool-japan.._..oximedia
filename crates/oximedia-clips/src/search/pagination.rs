//! Pagination support for search and clip listing results.
//!
//! This module provides `PageRequest` and `PageResult` types for cursor-based and
//! offset-based pagination of clip search results. Both in-memory and database-backed
//! queries can use these primitives to return bounded result pages alongside metadata
//! needed for navigation (total count, whether a next page exists, etc.).

use crate::clip::Clip;

/// A request for a single page of results using offset-based pagination.
///
/// The first page uses `offset = 0`. Subsequent pages advance by `page_size`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageRequest {
    /// Zero-based index of the first result to return.
    pub offset: usize,
    /// Maximum number of results per page.  Must be at least 1.
    pub page_size: usize,
}

impl PageRequest {
    /// Creates a new page request.
    ///
    /// `page_size` is clamped to a minimum of 1 so callers never receive an
    /// infinite or zero-length page.
    #[must_use]
    pub fn new(offset: usize, page_size: usize) -> Self {
        Self {
            offset,
            page_size: page_size.max(1),
        }
    }

    /// Creates a request for the first page.
    #[must_use]
    pub fn first(page_size: usize) -> Self {
        Self::new(0, page_size)
    }

    /// Returns a `PageRequest` for the immediately following page, or `None`
    /// if `current` already reaches beyond `total_items`.
    #[must_use]
    pub fn next_page(&self, total_items: usize) -> Option<Self> {
        let next_offset = self.offset + self.page_size;
        if next_offset < total_items {
            Some(Self::new(next_offset, self.page_size))
        } else {
            None
        }
    }

    /// Returns the 1-based page number corresponding to this request.
    #[must_use]
    pub fn page_number(&self) -> usize {
        self.offset / self.page_size + 1
    }

    /// Returns the total number of pages given a total item count.
    #[must_use]
    pub fn total_pages(&self, total_items: usize) -> usize {
        if self.page_size == 0 {
            return 0;
        }
        (total_items + self.page_size - 1) / self.page_size
    }
}

impl Default for PageRequest {
    fn default() -> Self {
        Self::new(0, 50)
    }
}

/// The result of a paginated query.
#[derive(Debug, Clone)]
pub struct PageResult<T> {
    /// The items on this page.
    pub items: Vec<T>,
    /// Total number of items across all pages.
    pub total_items: usize,
    /// The request that produced this result.
    pub request: PageRequest,
}

impl<T> PageResult<T> {
    /// Creates a new page result.
    #[must_use]
    pub fn new(items: Vec<T>, total_items: usize, request: PageRequest) -> Self {
        Self {
            items,
            total_items,
            request,
        }
    }

    /// Returns the number of items on this page.
    #[must_use]
    pub fn page_len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` if there is at least one more page after this one.
    #[must_use]
    pub fn has_next_page(&self) -> bool {
        self.request.offset + self.request.page_size < self.total_items
    }

    /// Returns `true` if this is not the first page.
    #[must_use]
    pub fn has_prev_page(&self) -> bool {
        self.request.offset > 0
    }

    /// Returns a `PageRequest` for the next page, or `None` if this is the last page.
    #[must_use]
    pub fn next_page_request(&self) -> Option<PageRequest> {
        self.request.next_page(self.total_items)
    }

    /// Returns the 1-based page number.
    #[must_use]
    pub fn page_number(&self) -> usize {
        self.request.page_number()
    }

    /// Returns the total number of pages.
    #[must_use]
    pub fn total_pages(&self) -> usize {
        self.request.total_pages(self.total_items)
    }

    /// Applies a mapping function to each item, producing a new `PageResult`.
    pub fn map<U, F: Fn(T) -> U>(self, f: F) -> PageResult<U> {
        PageResult {
            items: self.items.into_iter().map(f).collect(),
            total_items: self.total_items,
            request: self.request,
        }
    }
}

/// Applies offset/limit pagination to an in-memory slice of clips, returning
/// owned clones of the selected window together with pagination metadata.
///
/// This helper is suitable for in-memory search results where the full result
/// set is already available.  For large libraries, prefer issuing paginated SQL
/// queries directly.
#[must_use]
pub fn paginate_clips(clips: &[Clip], request: PageRequest) -> PageResult<Clip> {
    let total_items = clips.len();
    let start = request.offset.min(total_items);
    let end = (request.offset + request.page_size).min(total_items);
    let items: Vec<Clip> = clips[start..end].to_vec();
    PageResult::new(items, total_items, request)
}

/// Applies pagination to any slice by returning references into it.
///
/// Compared to `paginate_clips` this avoids cloning: the caller receives
/// shared references valid for the lifetime of the source slice.
#[must_use]
pub fn paginate_refs<'a, T>(items: &'a [T], request: PageRequest) -> PageResult<&'a T> {
    let total_items = items.len();
    let start = request.offset.min(total_items);
    let end = (request.offset + request.page_size).min(total_items);
    let page: Vec<&T> = items[start..end].iter().collect();
    PageResult::new(page, total_items, request)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_clips(n: usize) -> Vec<Clip> {
        (0..n)
            .map(|i| {
                let mut c = Clip::new(PathBuf::from(format!("/clip_{i}.mov")));
                c.set_name(&format!("Clip {i}"));
                c
            })
            .collect()
    }

    #[test]
    fn test_page_request_new_clamps_page_size() {
        let req = PageRequest::new(0, 0);
        assert_eq!(req.page_size, 1);
    }

    #[test]
    fn test_page_request_first() {
        let req = PageRequest::first(10);
        assert_eq!(req.offset, 0);
        assert_eq!(req.page_size, 10);
    }

    #[test]
    fn test_page_number() {
        let req = PageRequest::new(20, 10);
        assert_eq!(req.page_number(), 3);
    }

    #[test]
    fn test_total_pages() {
        let req = PageRequest::new(0, 10);
        assert_eq!(req.total_pages(25), 3);
        assert_eq!(req.total_pages(20), 2);
        assert_eq!(req.total_pages(0), 0);
    }

    #[test]
    fn test_next_page_returns_none_at_end() {
        let req = PageRequest::new(40, 10);
        assert!(req.next_page(50).is_none());
    }

    #[test]
    fn test_next_page_returns_some_when_more_items() {
        let req = PageRequest::new(0, 10);
        let next = req.next_page(25);
        assert!(next.is_some());
        assert_eq!(next.unwrap().offset, 10);
    }

    #[test]
    fn test_paginate_clips_first_page() {
        let clips = make_clips(25);
        let req = PageRequest::first(10);
        let page = paginate_clips(&clips, req);
        assert_eq!(page.page_len(), 10);
        assert_eq!(page.total_items, 25);
        assert!(page.has_next_page());
        assert!(!page.has_prev_page());
    }

    #[test]
    fn test_paginate_clips_last_page() {
        let clips = make_clips(25);
        let req = PageRequest::new(20, 10);
        let page = paginate_clips(&clips, req);
        assert_eq!(page.page_len(), 5);
        assert!(!page.has_next_page());
        assert!(page.has_prev_page());
    }

    #[test]
    fn test_paginate_empty_slice() {
        let clips: Vec<Clip> = Vec::new();
        let req = PageRequest::first(10);
        let page = paginate_clips(&clips, req);
        assert_eq!(page.page_len(), 0);
        assert_eq!(page.total_items, 0);
        assert!(!page.has_next_page());
    }

    #[test]
    fn test_paginate_refs() {
        let data: Vec<i32> = (0..15).collect();
        let req = PageRequest::new(5, 5);
        let page = paginate_refs(&data, req);
        assert_eq!(page.page_len(), 5);
        assert_eq!(*page.items[0], 5);
        assert_eq!(*page.items[4], 9);
    }

    #[test]
    fn test_page_result_map() {
        let data: Vec<i32> = (1..=10).collect();
        let req = PageRequest::first(10);
        let page = paginate_refs(&data, req);
        let doubled = page.map(|&x| x * 2);
        assert_eq!(doubled.items[0], 2);
        assert_eq!(doubled.total_items, 10);
    }

    #[test]
    fn test_page_result_total_pages() {
        let clips = make_clips(30);
        let req = PageRequest::new(0, 7);
        let page = paginate_clips(&clips, req);
        assert_eq!(page.total_pages(), 5); // ceil(30/7) = 5
    }

    #[test]
    fn test_next_page_request_chaining() {
        let clips = make_clips(100);
        let mut req = PageRequest::first(20);
        let mut pages_visited = 0usize;
        loop {
            let page = paginate_clips(&clips, req);
            pages_visited += 1;
            match page.next_page_request() {
                Some(next) => req = next,
                None => break,
            }
        }
        assert_eq!(pages_visited, 5);
    }
}
