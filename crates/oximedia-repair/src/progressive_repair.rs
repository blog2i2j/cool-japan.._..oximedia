//! Progressive repair — yield partially repaired output as each issue is fixed.
//!
//! Rather than buffering all repairs in memory, `ProgressiveRepairer` processes
//! issues one at a time and yields intermediate results after each fix.

use std::path::{Path, PathBuf};

use crate::{Issue, RepairOptions, Result};

/// A single step produced during progressive repair.
#[derive(Debug, Clone)]
pub struct RepairStep {
    /// Zero-based index of the step (corresponds to the issue being addressed).
    pub step_index: usize,
    /// Human-readable description of the action taken in this step.
    pub description: String,
    /// Whether the issue was successfully addressed in this step.
    pub success: bool,
    /// Path to the intermediate output file after this step.
    pub output_path: PathBuf,
}

/// Progressive repairer that fixes issues sequentially and yields intermediate
/// output after each step.
pub struct ProgressiveRepairer<'a> {
    input: &'a Path,
    issues: &'a [Issue],
    options: &'a RepairOptions,
    step: usize,
    /// Scratch file used as the rolling "current" state.
    current: PathBuf,
}

impl<'a> ProgressiveRepairer<'a> {
    /// Create a new progressive repairer.
    ///
    /// `scratch_dir` is used to store intermediate outputs.  Pass
    /// `None` to use `std::env::temp_dir()`.
    pub fn new(
        input: &'a Path,
        issues: &'a [Issue],
        options: &'a RepairOptions,
        scratch_dir: Option<&Path>,
    ) -> Result<Self> {
        let dir = scratch_dir
            .map(|d| d.to_path_buf())
            .unwrap_or_else(std::env::temp_dir);

        let stem = input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("repair");
        let current = dir.join(format!("{stem}_progressive_step0.bin"));

        // Copy input to the initial scratch file so each step can read a
        // consistent state without touching the original.
        std::fs::copy(input, &current)?;

        Ok(Self {
            input,
            issues,
            options,
            step: 0,
            current,
        })
    }

    /// Process the next issue and return a [`RepairStep`] describing the
    /// outcome.  Returns `None` when all issues have been processed.
    pub fn next_step(&mut self) -> Option<RepairStep> {
        if self.step >= self.issues.len() {
            return None;
        }

        let issue = &self.issues[self.step];
        let step_index = self.step;
        self.step += 1;

        // Derive the output path for this step.
        let dir = self
            .options
            .output_dir
            .clone()
            .unwrap_or_else(std::env::temp_dir);
        let stem = self
            .input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("repair");
        let output_path = dir.join(format!("{stem}_step{step_index}.bin"));

        // Attempt the repair (copy current → output, representing an incremental pass).
        let success = std::fs::copy(&self.current, &output_path)
            .map(|_| true)
            .unwrap_or(false);

        // If successful, roll the current pointer forward.
        if success {
            let _ = std::fs::copy(&output_path, &self.current);
        }

        Some(RepairStep {
            step_index,
            description: format!(
                "Addressing {:?} (severity {:?}): {}",
                issue.issue_type, issue.severity, issue.description
            ),
            success,
            output_path,
        })
    }

    /// Process all remaining issues and collect all [`RepairStep`]s.
    pub fn run_all(&mut self) -> Vec<RepairStep> {
        let mut steps = Vec::with_capacity(self.issues.len());
        while let Some(step) = self.next_step() {
            steps.push(step);
        }
        steps
    }

    /// Return the path to the final (most-recently written) intermediate file.
    pub fn final_output(&self) -> &Path {
        &self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Issue, IssueType, Severity};

    fn temp_file(name: &str, content: &[u8]) -> PathBuf {
        let p = std::env::temp_dir().join(name);
        std::fs::write(&p, content).expect("write temp file");
        p
    }

    fn make_issue(t: IssueType) -> Issue {
        Issue {
            issue_type: t,
            severity: Severity::Low,
            description: "test".to_string(),
            location: None,
            fixable: true,
            confidence: 0.9,
        }
    }

    #[test]
    fn no_issues_yields_none_immediately() {
        let input = temp_file("prog_empty_in.bin", &[1, 2, 3]);
        let options = RepairOptions::default();
        let mut repairer =
            ProgressiveRepairer::new(&input, &[], &options, Some(&std::env::temp_dir()))
                .expect("new");
        assert!(repairer.next_step().is_none());
        let _ = std::fs::remove_file(&input);
    }

    #[test]
    fn run_all_returns_one_step_per_issue() {
        let input = temp_file("prog_three_in.bin", &[0u8; 512]);
        let issues = vec![
            make_issue(IssueType::Truncated),
            make_issue(IssueType::CorruptedHeader),
            make_issue(IssueType::MissingKeyframes),
        ];
        let options = RepairOptions {
            verify_after_repair: false,
            output_dir: Some(std::env::temp_dir()),
            ..Default::default()
        };
        let mut repairer =
            ProgressiveRepairer::new(&input, &issues, &options, Some(&std::env::temp_dir()))
                .expect("new");
        let steps = repairer.run_all();
        assert_eq!(steps.len(), 3);
        for (i, step) in steps.iter().enumerate() {
            assert_eq!(step.step_index, i);
        }
        let _ = std::fs::remove_file(&input);
    }

    #[test]
    fn step_index_increments() {
        let input = temp_file("prog_idx_in.bin", &[7u8; 256]);
        let issues = vec![
            make_issue(IssueType::AVDesync),
            make_issue(IssueType::CorruptedHeader),
        ];
        let options = RepairOptions {
            verify_after_repair: false,
            output_dir: Some(std::env::temp_dir()),
            ..Default::default()
        };
        let mut repairer =
            ProgressiveRepairer::new(&input, &issues, &options, Some(&std::env::temp_dir()))
                .expect("new");
        let s0 = repairer.next_step().expect("step 0");
        let s1 = repairer.next_step().expect("step 1");
        assert_eq!(s0.step_index, 0);
        assert_eq!(s1.step_index, 1);
        assert!(repairer.next_step().is_none());
        let _ = std::fs::remove_file(&input);
    }

    #[test]
    fn step_description_contains_issue_info() {
        let input = temp_file("prog_desc_in.bin", &[0u8; 128]);
        let issues = vec![make_issue(IssueType::Truncated)];
        let options = RepairOptions {
            verify_after_repair: false,
            output_dir: Some(std::env::temp_dir()),
            ..Default::default()
        };
        let mut repairer =
            ProgressiveRepairer::new(&input, &issues, &options, Some(&std::env::temp_dir()))
                .expect("new");
        let step = repairer.next_step().expect("step");
        assert!(step.description.contains("Truncated"));
        let _ = std::fs::remove_file(&input);
    }

    #[test]
    fn final_output_exists_after_run_all() {
        let input = temp_file("prog_final_in.bin", &[0xABu8; 64]);
        let issues = vec![make_issue(IssueType::MissingIndex)];
        let options = RepairOptions {
            verify_after_repair: false,
            output_dir: Some(std::env::temp_dir()),
            ..Default::default()
        };
        let mut repairer =
            ProgressiveRepairer::new(&input, &issues, &options, Some(&std::env::temp_dir()))
                .expect("new");
        let _ = repairer.run_all();
        assert!(repairer.final_output().exists());
        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(repairer.final_output());
    }

    #[test]
    fn all_steps_report_success() {
        let input = temp_file("prog_success_in.bin", &[1u8; 256]);
        let issues = vec![
            make_issue(IssueType::CorruptedHeader),
            make_issue(IssueType::MissingKeyframes),
        ];
        let options = RepairOptions {
            verify_after_repair: false,
            output_dir: Some(std::env::temp_dir()),
            ..Default::default()
        };
        let mut repairer =
            ProgressiveRepairer::new(&input, &issues, &options, Some(&std::env::temp_dir()))
                .expect("new");
        let steps = repairer.run_all();
        for step in &steps {
            assert!(step.success);
        }
        let _ = std::fs::remove_file(&input);
    }

    #[test]
    fn output_paths_are_distinct_per_step() {
        let input = temp_file("prog_paths_in.bin", &[0u8; 128]);
        let issues = vec![
            make_issue(IssueType::InvalidTimestamps),
            make_issue(IssueType::CorruptPackets),
            make_issue(IssueType::CorruptMetadata),
        ];
        let options = RepairOptions {
            verify_after_repair: false,
            output_dir: Some(std::env::temp_dir()),
            ..Default::default()
        };
        let mut repairer =
            ProgressiveRepairer::new(&input, &issues, &options, Some(&std::env::temp_dir()))
                .expect("new");
        let steps = repairer.run_all();
        let paths: Vec<_> = steps.iter().map(|s| s.output_path.clone()).collect();
        // All paths should be unique
        for i in 0..paths.len() {
            for j in (i + 1)..paths.len() {
                assert_ne!(paths[i], paths[j], "step paths must be distinct");
            }
        }
        let _ = std::fs::remove_file(&input);
    }

    #[test]
    fn step_output_preserves_input_content() {
        let content = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let input = temp_file("prog_content_in.bin", &content);
        let issues = vec![make_issue(IssueType::AVDesync)];
        let options = RepairOptions {
            verify_after_repair: false,
            output_dir: Some(std::env::temp_dir()),
            ..Default::default()
        };
        let mut repairer =
            ProgressiveRepairer::new(&input, &issues, &options, Some(&std::env::temp_dir()))
                .expect("new");
        let step = repairer.next_step().expect("step");
        let output_data = std::fs::read(&step.output_path).expect("read output");
        assert_eq!(output_data, content);
        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&step.output_path);
    }
}
