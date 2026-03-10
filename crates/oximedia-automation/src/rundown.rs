//! Broadcast rundown automation.
//!
//! Provides types for managing on-air rundowns including item status,
//! presenter assignments and automation cues.

/// Editorial and operational status of a rundown item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RundownStatus {
    /// Item is being edited and is not yet approved.
    Draft,
    /// Item has been approved for broadcast.
    Approved,
    /// Item is currently on air.
    Live,
    /// Item has finished airing.
    Completed,
    /// Item was deliberately skipped.
    Skipped,
}

impl RundownStatus {
    /// Return `true` if the item can still be edited (i.e. not on air or finished).
    pub fn can_edit(&self) -> bool {
        matches!(self, Self::Draft | Self::Approved)
    }

    /// Return `true` if the item has reached a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Skipped)
    }
}

/// A single item inside a broadcast rundown.
#[derive(Debug, Clone)]
pub struct RundownItem {
    /// Unique identifier for this item.
    pub id: u64,
    /// Display title of the item.
    pub title: String,
    /// Planned duration in milliseconds.
    pub duration_ms: u32,
    /// Current editorial/operational status.
    pub status: RundownStatus,
    /// Whether automation should handle this item's cue automatically.
    pub cue_automation: bool,
    /// Optional presenter name assigned to this item.
    pub presenter: Option<String>,
}

impl RundownItem {
    /// Create a new rundown item in `Draft` status.
    pub fn new(id: u64, title: &str, duration_ms: u32) -> Self {
        Self {
            id,
            title: title.to_string(),
            duration_ms,
            status: RundownStatus::Draft,
            cue_automation: false,
            presenter: None,
        }
    }

    /// Mark this item for automation cue (builder pattern).
    pub fn with_automation(mut self) -> Self {
        self.cue_automation = true;
        self
    }

    /// Assign a presenter to this item (builder pattern).
    pub fn with_presenter(mut self, p: &str) -> Self {
        self.presenter = Some(p.to_string());
        self
    }
}

/// A broadcast rundown containing an ordered list of items.
#[derive(Debug)]
pub struct Rundown {
    /// Name / episode identifier of the rundown.
    pub name: String,
    /// Scheduled air time in milliseconds (epoch-relative).
    pub air_time_ms: u64,
    items: Vec<RundownItem>,
}

impl Rundown {
    /// Create a new, empty rundown.
    pub fn new(name: &str, air_time_ms: u64) -> Self {
        Self {
            name: name.to_string(),
            air_time_ms,
            items: vec![],
        }
    }

    /// Append an item to the end of the rundown.
    pub fn add_item(&mut self, item: RundownItem) {
        self.items.push(item);
    }

    /// Return the sum of all item durations in milliseconds.
    pub fn total_duration_ms(&self) -> u32 {
        self.items
            .iter()
            .map(|i| i.duration_ms)
            .fold(0u32, u32::saturating_add)
    }

    /// Return a reference to the first item that is neither completed nor skipped.
    pub fn next_item(&self) -> Option<&RundownItem> {
        self.items.iter().find(|i| !i.status.is_terminal())
    }

    /// Approve the item with the given `id`.
    ///
    /// Returns `true` if the item was found and its status updated.
    pub fn approve_item(&mut self, id: u64) -> bool {
        for item in &mut self.items {
            if item.id == id && item.status.can_edit() {
                item.status = RundownStatus::Approved;
                return true;
            }
        }
        false
    }

    /// Skip the item with the given `id`.
    ///
    /// Returns `true` if the item was found and skipped.
    pub fn skip_item(&mut self, id: u64) -> bool {
        for item in &mut self.items {
            if item.id == id && !item.status.is_terminal() {
                item.status = RundownStatus::Skipped;
                return true;
            }
        }
        false
    }

    /// Return references to all items that have automation cue enabled.
    pub fn automation_items(&self) -> Vec<&RundownItem> {
        self.items.iter().filter(|i| i.cue_automation).collect()
    }

    /// Return the total number of items in the rundown.
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Return a reference to all items.
    pub fn items(&self) -> &[RundownItem] {
        &self.items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_can_edit_draft() {
        assert!(RundownStatus::Draft.can_edit());
    }

    #[test]
    fn test_status_can_edit_approved() {
        assert!(RundownStatus::Approved.can_edit());
    }

    #[test]
    fn test_status_cannot_edit_live() {
        assert!(!RundownStatus::Live.can_edit());
    }

    #[test]
    fn test_status_is_terminal_completed() {
        assert!(RundownStatus::Completed.is_terminal());
    }

    #[test]
    fn test_status_is_terminal_skipped() {
        assert!(RundownStatus::Skipped.is_terminal());
    }

    #[test]
    fn test_status_not_terminal_draft() {
        assert!(!RundownStatus::Draft.is_terminal());
    }

    #[test]
    fn test_item_builder_automation() {
        let item = RundownItem::new(1, "News Intro", 30_000).with_automation();
        assert!(item.cue_automation);
    }

    #[test]
    fn test_item_builder_presenter() {
        let item = RundownItem::new(2, "Weather", 60_000).with_presenter("Jane Doe");
        assert_eq!(item.presenter.as_deref(), Some("Jane Doe"));
    }

    #[test]
    fn test_rundown_total_duration_empty() {
        let rd = Rundown::new("Morning News", 0);
        assert_eq!(rd.total_duration_ms(), 0);
    }

    #[test]
    fn test_rundown_total_duration_sum() {
        let mut rd = Rundown::new("Evening News", 0);
        rd.add_item(RundownItem::new(1, "Intro", 10_000));
        rd.add_item(RundownItem::new(2, "Sport", 20_000));
        assert_eq!(rd.total_duration_ms(), 30_000);
    }

    #[test]
    fn test_rundown_next_item_skips_terminal() {
        let mut rd = Rundown::new("Late News", 0);
        let mut first = RundownItem::new(1, "Lead", 5_000);
        first.status = RundownStatus::Completed;
        rd.add_item(first);
        rd.add_item(RundownItem::new(2, "Second", 5_000));
        let next = rd.next_item().expect("next_item should succeed");
        assert_eq!(next.id, 2);
    }

    #[test]
    fn test_rundown_approve_item() {
        let mut rd = Rundown::new("Test", 0);
        rd.add_item(RundownItem::new(1, "Item", 1_000));
        let result = rd.approve_item(1);
        assert!(result);
        assert_eq!(rd.items()[0].status, RundownStatus::Approved);
    }

    #[test]
    fn test_rundown_approve_item_not_found() {
        let mut rd = Rundown::new("Test", 0);
        assert!(!rd.approve_item(99));
    }

    #[test]
    fn test_rundown_skip_item() {
        let mut rd = Rundown::new("Test", 0);
        rd.add_item(RundownItem::new(10, "Skip Me", 2_000));
        assert!(rd.skip_item(10));
        assert_eq!(rd.items()[0].status, RundownStatus::Skipped);
    }

    #[test]
    fn test_rundown_skip_already_terminal() {
        let mut rd = Rundown::new("Test", 0);
        let mut item = RundownItem::new(1, "Done", 1_000);
        item.status = RundownStatus::Completed;
        rd.add_item(item);
        assert!(!rd.skip_item(1));
    }

    #[test]
    fn test_rundown_automation_items() {
        let mut rd = Rundown::new("Auto Test", 0);
        rd.add_item(RundownItem::new(1, "Manual", 5_000));
        rd.add_item(RundownItem::new(2, "Auto", 5_000).with_automation());
        rd.add_item(RundownItem::new(3, "Also Auto", 5_000).with_automation());
        let auto_items = rd.automation_items();
        assert_eq!(auto_items.len(), 2);
    }

    #[test]
    fn test_rundown_next_item_none_when_all_terminal() {
        let mut rd = Rundown::new("Done", 0);
        let mut item = RundownItem::new(1, "Past", 1_000);
        item.status = RundownStatus::Skipped;
        rd.add_item(item);
        assert!(rd.next_item().is_none());
    }
}
