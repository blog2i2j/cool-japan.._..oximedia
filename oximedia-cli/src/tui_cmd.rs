//! TUI command — interactive terminal UI for OxiMedia CLI.
//!
//! Launches a ratatui-based interactive interface with file browser,
//! command reference, and system information panels.

use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Tabs},
    Terminal,
};
use std::io::Stdout;

// ── Tab identifiers ───────────────────────────────────────────────────────────

const TABS: &[&str] = &["  Files  ", "  Commands  ", "  About  "];
const TAB_FILES: usize = 0;
const TAB_COMMANDS: usize = 1;
const TAB_ABOUT: usize = 2;

// ── Application state ─────────────────────────────────────────────────────────

struct App {
    tab_index: usize,
    file_list: Vec<FileEntry>,
    file_state: ListState,
    command_list: Vec<(&'static str, &'static str)>,
    command_state: ListState,
    selected_file_info: Option<String>,
    status_message: String,
}

#[derive(Clone)]
struct FileEntry {
    name: String,
    size_bytes: u64,
    is_dir: bool,
}

impl App {
    fn new() -> Result<Self> {
        let file_list = load_current_dir_files()?;
        let command_list = all_commands();

        let mut file_state = ListState::default();
        if !file_list.is_empty() {
            file_state.select(Some(0));
        }

        let mut command_state = ListState::default();
        if !command_list.is_empty() {
            command_state.select(Some(0));
        }

        Ok(Self {
            tab_index: 0,
            file_list,
            file_state,
            command_list,
            command_state,
            selected_file_info: None,
            status_message:
                "Press 'q' to quit | Tab to switch tabs | ↑↓ to navigate | Enter for details"
                    .to_string(),
        })
    }

    fn next_tab(&mut self) {
        self.tab_index = (self.tab_index + 1) % TABS.len();
        self.selected_file_info = None;
    }

    fn prev_tab(&mut self) {
        self.tab_index = (self.tab_index + TABS.len() - 1) % TABS.len();
        self.selected_file_info = None;
    }

    fn cursor_up(&mut self) {
        match self.tab_index {
            TAB_FILES => {
                let len = self.file_list.len();
                if len == 0 {
                    return;
                }
                let i = match self.file_state.selected() {
                    Some(i) if i > 0 => i - 1,
                    Some(_) => len - 1,
                    None => 0,
                };
                self.file_state.select(Some(i));
                self.selected_file_info = None;
            }
            TAB_COMMANDS => {
                let len = self.command_list.len();
                if len == 0 {
                    return;
                }
                let i = match self.command_state.selected() {
                    Some(i) if i > 0 => i - 1,
                    Some(_) => len - 1,
                    None => 0,
                };
                self.command_state.select(Some(i));
            }
            _ => {}
        }
    }

    fn cursor_down(&mut self) {
        match self.tab_index {
            TAB_FILES => {
                let len = self.file_list.len();
                if len == 0 {
                    return;
                }
                let i = match self.file_state.selected() {
                    Some(i) => (i + 1) % len,
                    None => 0,
                };
                self.file_state.select(Some(i));
                self.selected_file_info = None;
            }
            TAB_COMMANDS => {
                let len = self.command_list.len();
                if len == 0 {
                    return;
                }
                let i = match self.command_state.selected() {
                    Some(i) => (i + 1) % len,
                    None => 0,
                };
                self.command_state.select(Some(i));
            }
            _ => {}
        }
    }

    fn on_enter(&mut self) {
        if self.tab_index == TAB_FILES {
            if let Some(idx) = self.file_state.selected() {
                if let Some(entry) = self.file_list.get(idx) {
                    let ext = std::path::Path::new(&entry.name)
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("—");
                    let kind = if entry.is_dir { "Directory" } else { "File" };
                    let size_str = format_size(entry.size_bytes);
                    self.selected_file_info = Some(format!(
                        "{kind}: {}\n  Size: {}\n  Extension: .{ext}",
                        entry.name, size_str,
                    ));
                    self.status_message = format!("Selected: {}", entry.name);
                }
            }
        }
    }
}

// ── File helpers ─────────────────────────────────────────────────────────────

fn load_current_dir_files() -> Result<Vec<FileEntry>> {
    let cwd = std::env::current_dir().context("Failed to get current directory")?;
    let mut entries: Vec<FileEntry> = Vec::new();

    let read_dir = std::fs::read_dir(&cwd).context("Failed to read current directory")?;

    for entry in read_dir {
        let entry = entry.context("Failed to read directory entry")?;
        let meta = entry.metadata().context("Failed to read file metadata")?;
        let name = entry.file_name().to_string_lossy().to_string();
        let size_bytes = if meta.is_file() { meta.len() } else { 0 };
        entries.push(FileEntry {
            name,
            size_bytes,
            is_dir: meta.is_dir(),
        });
    }

    entries.sort_by(|a, b| {
        // Directories first, then alphabetical
        b.is_dir.cmp(&a.is_dir).then(a.name.cmp(&b.name))
    });

    Ok(entries)
}

fn format_size(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.2} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.2} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

// ── Command list ─────────────────────────────────────────────────────────────

fn all_commands() -> Vec<(&'static str, &'static str)> {
    vec![
        ("probe", "Inspect media file format and streams"),
        ("transcode", "Re-encode video/audio to another codec"),
        ("extract", "Pull individual frames from video"),
        ("batch", "Process a whole directory of files"),
        ("scene", "Detect scene cuts and classify shots"),
        ("audio", "Loudness metering, normalisation, beat detection"),
        ("subtitle", "Convert, extract, burn-in subtitles"),
        ("filter", "Apply standalone filter graph"),
        ("lut", "Apply, inspect, or convert LUT files"),
        ("denoise", "Reduce video noise / grain"),
        ("stabilize", "Remove camera shake from video"),
        ("edl", "Parse, validate, and export EDL files"),
        ("package", "HLS / DASH adaptive-bitrate packaging"),
        ("forensics", "Tamper detection and provenance analysis"),
        ("stream", "HLS/DASH serve, ingest, record"),
        (
            "search",
            "Content search: text, visual similarity, fingerprint",
        ),
        ("timecode", "Timecode conversion and calculation"),
        ("repair", "Media file repair and recovery"),
        ("color", "Color management: convert, matrix, Delta E"),
        ("playlist", "Generate, validate, and simulate playlists"),
        ("conform", "QC/conformance checking and fixing"),
        ("archive", "IMF/archive packaging and extraction"),
        ("watermark", "Digital audio watermarking"),
        ("tui", "Launch this interactive terminal UI"),
    ]
}

// ── Rendering ─────────────────────────────────────────────────────────────────

fn draw(terminal: &mut Terminal<CrosstermBackend<Stdout>>, app: &mut App) -> Result<()> {
    terminal.draw(|frame| {
        let area = frame.area();

        // Top-level layout: title bar / tabs, body, status bar
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // tabs
                Constraint::Min(0),    // content
                Constraint::Length(2), // status bar
            ])
            .split(area);

        render_tabs(frame, chunks[0], app);
        render_body(frame, chunks[1], app);
        render_status(frame, chunks[2], app);
    })?;
    Ok(())
}

fn render_tabs(frame: &mut ratatui::Frame, area: Rect, app: &App) {
    let tab_titles: Vec<Line> = TABS
        .iter()
        .map(|t| Line::from(Span::styled(*t, Style::default().fg(Color::White))))
        .collect();

    let tabs = Tabs::new(tab_titles)
        .block(
            Block::default().borders(Borders::ALL).title(Span::styled(
                " OxiMedia TUI ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
        )
        .select(app.tab_index)
        .style(Style::default().fg(Color::DarkGray))
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );

    frame.render_widget(tabs, area);
}

fn render_body(frame: &mut ratatui::Frame, area: Rect, app: &mut App) {
    match app.tab_index {
        TAB_FILES => render_files_tab(frame, area, app),
        TAB_COMMANDS => render_commands_tab(frame, area, app),
        TAB_ABOUT => render_about_tab(frame, area),
        _ => {}
    }
}

fn render_files_tab(frame: &mut ratatui::Frame, area: Rect, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    // File list
    let items: Vec<ListItem> = app
        .file_list
        .iter()
        .map(|f| {
            let icon = if f.is_dir { "d " } else { "  " };
            let size_str = if f.is_dir {
                String::new()
            } else {
                format!(" ({})", format_size(f.size_bytes))
            };
            let label = format!("{}{}{}", icon, f.name, size_str);
            let style = if f.is_dir {
                Style::default().fg(Color::Cyan)
            } else {
                Style::default().fg(Color::White)
            };
            ListItem::new(label).style(style)
        })
        .collect();

    let file_list_widget = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Files (current directory) "),
        )
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("> ");

    frame.render_stateful_widget(file_list_widget, chunks[0], &mut app.file_state);

    // Detail panel
    let detail_text = if let Some(ref info) = app.selected_file_info {
        info.clone()
    } else {
        "Press Enter on a file to see details".to_string()
    };

    let detail = Paragraph::new(detail_text)
        .block(Block::default().borders(Borders::ALL).title(" Details "))
        .style(Style::default().fg(Color::Gray))
        .wrap(ratatui::widgets::Wrap { trim: true });

    frame.render_widget(detail, chunks[1]);
}

fn render_commands_tab(frame: &mut ratatui::Frame, area: Rect, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(area);

    // Command name list
    let cmd_items: Vec<ListItem> = app
        .command_list
        .iter()
        .map(|(name, _)| {
            ListItem::new(format!("  oximedia {name}")).style(Style::default().fg(Color::Green))
        })
        .collect();

    let cmd_list_widget = List::new(cmd_items)
        .block(Block::default().borders(Borders::ALL).title(" Commands "))
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("> ");

    frame.render_stateful_widget(cmd_list_widget, chunks[0], &mut app.command_state);

    // Description panel
    let description = if let Some(idx) = app.command_state.selected() {
        app.command_list
            .get(idx)
            .map(|(name, desc)| format!("Command: oximedia {name}\n\n{desc}"))
            .unwrap_or_default()
    } else {
        "Select a command to see its description.".to_string()
    };

    let desc_widget = Paragraph::new(description)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Description "),
        )
        .style(Style::default().fg(Color::Gray))
        .wrap(ratatui::widgets::Wrap { trim: true });

    frame.render_widget(desc_widget, chunks[1]);
}

fn render_about_tab(frame: &mut ratatui::Frame, area: Rect) {
    let version = env!("CARGO_PKG_VERSION");
    let text = format!(
        r#"OxiMedia — Sovereign Media Framework
Version: {version}

A patent-free, pure-Rust reconstruction of FFmpeg + OpenCV.

Supported codecs (video): AV1, VP9, VP8, Theora
Supported codecs (audio): Opus, Vorbis, FLAC, PCM
Supported containers:      Matroska, WebM, Ogg, FLAC, WAV

Homepage: https://github.com/cool-japan/oximedia
License:  Apache-2.0
Author:   COOLJAPAN OU (Team Kitasan)

Keyboard shortcuts:
  q / Ctrl+C    Quit
  Tab / →       Next tab
  ← / Shift+Tab Previous tab
  ↑ / ↓         Navigate list
  Enter         Show file/command details
"#
    );

    let about = Paragraph::new(text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" About OxiMedia "),
        )
        .style(Style::default().fg(Color::White))
        .wrap(ratatui::widgets::Wrap { trim: false });

    frame.render_widget(about, area);
}

fn render_status(frame: &mut ratatui::Frame, area: Rect, app: &App) {
    let status = Paragraph::new(Line::from(vec![
        Span::styled(" ", Style::default()),
        Span::styled(&app.status_message, Style::default().fg(Color::DarkGray)),
    ]))
    .block(Block::default().borders(Borders::TOP));

    frame.render_widget(status, area);
}

// ── Terminal setup / teardown ─────────────────────────────────────────────────

fn setup_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode().context("Failed to enable raw mode")?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen).context("Failed to enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    Terminal::new(backend).context("Failed to create terminal")
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode().context("Failed to disable raw mode")?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)
        .context("Failed to leave alternate screen")?;
    terminal.show_cursor().context("Failed to show cursor")
}

// ── Main event loop ───────────────────────────────────────────────────────────

/// Launch the interactive TUI.
pub fn run_tui() -> Result<()> {
    let mut terminal = setup_terminal()?;

    // Install a panic hook that restores the terminal before printing the panic.
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Best-effort: ignore errors during panic cleanup.
        let _ = disable_raw_mode();
        let _ = execute!(std::io::stdout(), LeaveAlternateScreen);
        original_hook(info);
    }));

    let mut app = App::new().context("Failed to initialise TUI app state")?;
    let tick_duration = std::time::Duration::from_millis(250);

    loop {
        draw(&mut terminal, &mut app)?;

        if event::poll(tick_duration).context("Event poll failed")? {
            match event::read().context("Event read failed")? {
                Event::Key(key) => {
                    // Ctrl+C → quit
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.code == KeyCode::Char('c')
                    {
                        break;
                    }
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Tab | KeyCode::Right => app.next_tab(),
                        KeyCode::BackTab | KeyCode::Left => app.prev_tab(),
                        KeyCode::Up => app.cursor_up(),
                        KeyCode::Down => app.cursor_down(),
                        KeyCode::Enter => app.on_enter(),
                        _ => {}
                    }
                }
                Event::Resize(_, _) => {
                    // ratatui handles resize automatically on next draw.
                }
                _ => {}
            }
        }
    }

    restore_terminal(&mut terminal)?;
    Ok(())
}
