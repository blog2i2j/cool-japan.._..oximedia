#![allow(dead_code)]
//! Job tagging — attach, remove, and query string tags on jobs.

use std::collections::{HashMap, HashSet};

/// A validated job tag.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JobTag(String);

impl JobTag {
    /// Create a new tag from a raw string.
    ///
    /// Returns `None` if the string is empty or contains whitespace.
    #[must_use]
    pub fn new(raw: impl Into<String>) -> Option<Self> {
        let s = raw.into();
        if s.is_empty() || s.contains(char::is_whitespace) {
            None
        } else {
            Some(Self(s))
        }
    }

    /// Returns `true` if the tag string is valid (non-empty, no whitespace).
    #[must_use]
    pub fn is_valid(raw: &str) -> bool {
        !raw.is_empty() && !raw.contains(char::is_whitespace)
    }

    /// The underlying tag string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for JobTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A job id paired with a set of tags.
#[derive(Debug, Clone)]
pub struct TaggedJob {
    /// Identifier for the underlying job.
    pub job_id: String,
    tags: HashSet<JobTag>,
}

impl TaggedJob {
    /// Create a new tagged job with no tags.
    #[must_use]
    pub fn new(job_id: impl Into<String>) -> Self {
        Self {
            job_id: job_id.into(),
            tags: HashSet::new(),
        }
    }

    /// Attach a tag. Returns `true` if the tag was newly added.
    pub fn attach(&mut self, tag: JobTag) -> bool {
        self.tags.insert(tag)
    }

    /// Remove a tag. Returns `true` if the tag was present.
    pub fn detach(&mut self, tag: &JobTag) -> bool {
        self.tags.remove(tag)
    }

    /// Returns `true` if this job has the given tag.
    #[must_use]
    pub fn has_tag(&self, tag: &JobTag) -> bool {
        self.tags.contains(tag)
    }

    /// Number of tags attached to this job.
    #[must_use]
    pub fn tag_count(&self) -> usize {
        self.tags.len()
    }

    /// Return all tags as a sorted vec of strings for deterministic output.
    #[must_use]
    pub fn tag_strings(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.tags.iter().map(|t| t.as_str()).collect();
        v.sort_unstable();
        v
    }
}

/// An index mapping tags to the set of job ids that carry them.
#[derive(Debug, Default)]
pub struct JobTagIndex {
    /// tag → set of job_ids
    index: HashMap<String, HashSet<String>>,
    /// job_id → set of tags
    job_tags: HashMap<String, HashSet<String>>,
}

impl JobTagIndex {
    /// Create an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tag to a job.
    pub fn tag(&mut self, job_id: impl Into<String>, tag: JobTag) {
        let jid = job_id.into();
        let tag_str = tag.0.clone();
        self.index
            .entry(tag_str.clone())
            .or_default()
            .insert(jid.clone());
        self.job_tags.entry(jid).or_default().insert(tag_str);
    }

    /// Remove a tag from a job.
    pub fn untag(&mut self, job_id: &str, tag: &JobTag) {
        let tag_str = tag.as_str();
        if let Some(jobs) = self.index.get_mut(tag_str) {
            jobs.remove(job_id);
            if jobs.is_empty() {
                self.index.remove(tag_str);
            }
        }
        if let Some(tags) = self.job_tags.get_mut(job_id) {
            tags.remove(tag_str);
            if tags.is_empty() {
                self.job_tags.remove(job_id);
            }
        }
    }

    /// Return the set of job ids that have the given tag.
    #[must_use]
    pub fn jobs_with_tag(&self, tag: &JobTag) -> HashSet<&str> {
        self.index
            .get(tag.as_str())
            .map(|s| s.iter().map(|id| id.as_str()).collect())
            .unwrap_or_default()
    }

    /// Return the number of distinct tags in the index.
    #[must_use]
    pub fn tag_count(&self) -> usize {
        self.index.len()
    }

    /// Return the tags attached to a specific job.
    #[must_use]
    pub fn tags_for_job(&self, job_id: &str) -> HashSet<&str> {
        self.job_tags
            .get(job_id)
            .map(|s| s.iter().map(|t| t.as_str()).collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_tag_valid() {
        assert!(JobTag::is_valid("video"));
        assert!(JobTag::is_valid("high-priority"));
    }

    #[test]
    fn test_job_tag_invalid_empty() {
        assert!(!JobTag::is_valid(""));
    }

    #[test]
    fn test_job_tag_invalid_whitespace() {
        assert!(!JobTag::is_valid("with space"));
        assert!(!JobTag::is_valid("tab\there"));
    }

    #[test]
    fn test_job_tag_new_some() {
        let t = JobTag::new("encode");
        assert!(t.is_some());
        assert_eq!(t.expect("test expectation failed").as_str(), "encode");
    }

    #[test]
    fn test_job_tag_new_none_empty() {
        assert!(JobTag::new("").is_none());
    }

    #[test]
    fn test_job_tag_new_none_whitespace() {
        assert!(JobTag::new("bad tag").is_none());
    }

    #[test]
    fn test_job_tag_display() {
        let t = JobTag::new("mytag").expect("t should be valid");
        assert_eq!(format!("{t}"), "mytag");
    }

    #[test]
    fn test_tagged_job_has_tag() {
        let mut j = TaggedJob::new("job-1");
        let t = JobTag::new("audio").expect("t should be valid");
        j.attach(t.clone());
        assert!(j.has_tag(&t));
    }

    #[test]
    fn test_tagged_job_has_tag_false() {
        let j = TaggedJob::new("job-2");
        let t = JobTag::new("video").expect("t should be valid");
        assert!(!j.has_tag(&t));
    }

    #[test]
    fn test_tagged_job_detach() {
        let mut j = TaggedJob::new("job-3");
        let t = JobTag::new("batch").expect("t should be valid");
        j.attach(t.clone());
        assert_eq!(j.tag_count(), 1);
        assert!(j.detach(&t));
        assert_eq!(j.tag_count(), 0);
    }

    #[test]
    fn test_tagged_job_tag_count() {
        let mut j = TaggedJob::new("job-4");
        j.attach(JobTag::new("a").expect("attach should succeed"));
        j.attach(JobTag::new("b").expect("attach should succeed"));
        j.attach(JobTag::new("c").expect("attach should succeed"));
        assert_eq!(j.tag_count(), 3);
    }

    #[test]
    fn test_tagged_job_tag_strings_sorted() {
        let mut j = TaggedJob::new("job-5");
        j.attach(JobTag::new("z").expect("attach should succeed"));
        j.attach(JobTag::new("a").expect("attach should succeed"));
        let strings = j.tag_strings();
        assert_eq!(strings, vec!["a", "z"]);
    }

    #[test]
    fn test_job_tag_index_tag_and_jobs_with_tag() {
        let mut idx = JobTagIndex::new();
        let t = JobTag::new("urgent").expect("t should be valid");
        idx.tag("job-a", t.clone());
        idx.tag("job-b", t.clone());
        let jobs = idx.jobs_with_tag(&t);
        assert_eq!(jobs.len(), 2);
        assert!(jobs.contains("job-a"));
        assert!(jobs.contains("job-b"));
    }

    #[test]
    fn test_job_tag_index_untag() {
        let mut idx = JobTagIndex::new();
        let t = JobTag::new("slow").expect("t should be valid");
        idx.tag("job-x", t.clone());
        assert_eq!(idx.jobs_with_tag(&t).len(), 1);
        idx.untag("job-x", &t);
        assert_eq!(idx.jobs_with_tag(&t).len(), 0);
    }

    #[test]
    fn test_job_tag_index_tag_count() {
        let mut idx = JobTagIndex::new();
        idx.tag("j1", JobTag::new("t1").expect("tag should succeed"));
        idx.tag("j2", JobTag::new("t2").expect("tag should succeed"));
        idx.tag("j3", JobTag::new("t1").expect("tag should succeed")); // same tag as j1
        assert_eq!(idx.tag_count(), 2);
    }

    #[test]
    fn test_job_tag_index_tags_for_job() {
        let mut idx = JobTagIndex::new();
        idx.tag("jq", JobTag::new("video").expect("tag should succeed"));
        idx.tag("jq", JobTag::new("hd").expect("tag should succeed"));
        let tags = idx.tags_for_job("jq");
        assert!(tags.contains("video"));
        assert!(tags.contains("hd"));
    }
}
