//! eframe / egui 0.29 frontend for the real-time voice translator.
//!
//! The UI is intentionally decoupled from the audio capture path. A
//! [`Backend`] trait abstracts "produce [`PipelineEvent`]s" — the default
//! [`MockBackend`] pushes synthetic events so the UI is testable on
//! headless CI / dev boxes with no microphone, and [`RealtimeBackend`]
//! wires cpal audio capture + the real pipeline (for end-user builds).
//!
//! # Architecture
//!
//! ```text
//!   ┌────────────────────┐   mpsc   ┌────────────────────┐
//!   │ Backend (any impl) │ ───────► │ VoiceTranslatorApp │
//!   └────────────────────┘          └────────────────────┘
//!          ▲                                  │
//!          │ owns one sender                 │ owns one receiver
//!          │                                  ▼
//!   spawn at "Start"                  drain in update()
//! ```
//!
//! The `App::update` callback drains the receiver every frame and
//! appends to a rolling [`UiLog`] buffer (capped at
//! [`UiLog::DEFAULT_CAPACITY`] entries to bound memory). A separate
//! `Running` flag controls whether a worker task is alive; the worker
//! holds the `Backend` and a `Sender<PipelineEvent>`, and is aborted
//! when the user clicks "Stop" or the app exits.
//!
//! The UI never holds a `&mut Translator` or a `&mut Transcriber` —
//! that keeps the eframe repaint loop free of long-lived borrows and
//! makes the worker trivially `Send`.
//!
//! # CI / headless verification
//!
//! `cargo test --lib` runs the pure-logic tests in this module
//! (language list / UI log behaviour) without spinning up eframe.
//! Running the full GUI requires a display server.

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::time::{Duration, Instant};

use crate::pipeline::{Pipeline, PipelineConfig, PipelineEvent, PipelineStage};
use crate::translation::{LanguageCode, StubTranslator, Translator};

/// One row in the on-screen transcript / translation log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiLogEntry {
    /// Wall-clock instant (monotonic) when the event landed in the UI.
    pub at: Instant,
    /// Pipeline stage the entry came from, for colour-coding / filtering.
    pub stage: EntryStage,
    /// Original transcript (if any) and the translation.
    pub text: String,
    pub translated: String,
}

impl UiLogEntry {
    fn from_event(event: &PipelineEvent) -> Self {
        match event {
            PipelineEvent::NoSpeech { .. } => Self {
                at: Instant::now(),
                stage: EntryStage::NoSpeech,
                text: String::new(),
                translated: String::new(),
            },
            PipelineEvent::TranscriptOnly { text, .. } => Self {
                at: Instant::now(),
                stage: EntryStage::Transcript,
                text: text.clone(),
                translated: String::new(),
            },
            PipelineEvent::Translated {
                text, translated, ..
            } => Self {
                at: Instant::now(),
                stage: EntryStage::Translated,
                text: text.clone(),
                translated: translated.clone(),
            },
            PipelineEvent::Error { stage, message, .. } => Self {
                at: Instant::now(),
                stage: EntryStage::Error,
                text: format!("{:?}: {}", stage, message),
                translated: String::new(),
            },
        }
    }
}

/// UI-friendly re-statement of [`PipelineStage`] / [`PipelineEvent`]
/// variants, so the log can render them with icons or colours without
/// re-pattern-matching on the raw event every frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryStage {
    NoSpeech,
    Transcript,
    Translated,
    Error,
}

/// Bounded rolling log the UI renders.
///
/// We cap the log at `DEFAULT_CAPACITY` entries to keep memory bounded
/// for long sessions — the cap is small enough that the egui `ScrollArea`
/// stays snappy.
#[derive(Debug, Clone)]
pub struct UiLog {
    entries: Vec<UiLogEntry>,
    capacity: usize,
}

impl Default for UiLog {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            capacity: Self::DEFAULT_CAPACITY,
        }
    }
}

impl UiLog {
    /// 200 rows is enough to fill a 1080p screen a few times over and
    /// still keep scrolling smooth.
    pub const DEFAULT_CAPACITY: usize = 200;

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, entry: UiLogEntry) {
        self.entries.push(entry);
        if self.entries.len() > self.capacity {
            // Drop the oldest — `VecDeque` would be O(1) but a plain
            // `Vec` is simpler and 200-entry trims are cheap.
            let excess = self.entries.len() - self.capacity;
            self.entries.drain(0..excess);
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, UiLogEntry> {
        self.entries.iter()
    }
}

/// Source-language / target-language selection the user has made.
/// Kept separate from `PipelineConfig` so the UI can change it without
/// needing to reconstruct the pipeline until the user clicks "Apply".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UiLanguageSelection {
    pub source: LanguageCode,
    pub target: LanguageCode,
}

impl Default for UiLanguageSelection {
    fn default() -> Self {
        Self {
            source: LanguageCode::En,
            target: LanguageCode::Zh,
        }
    }
}

/// Whether the worker task is alive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunState {
    /// No worker. The user can click "Start".
    Stopped,
    /// Worker is producing events.
    Running,
    /// User clicked Stop; the worker is finishing its current chunk.
    Stopping,
}

impl RunState {
    pub fn is_running(self) -> bool {
        matches!(self, RunState::Running)
    }
}

/// A `Backend` produces `PipelineEvent`s. We do not require it to be
/// `Send` because the worker task owns the concrete instance; the
/// `App` only ever holds the `Sender` half of the channel.
///
/// Implementors are free to spawn their own threads, hold cpal streams,
/// etc. The `App` does not care — it just drains a `Receiver`.
pub trait Backend {
    /// Drive the pipeline for at most `budget`. Implementations should
    /// return when `budget` elapses or when the channel is closed.
    fn run(self: Box<Self>, tx: Sender<PipelineEvent>, budget: Duration);
}

/// Mock backend that synthesises a fixed list of `PipelineEvent`s and
/// pushes them at a configurable cadence. This is the default — the
/// real-time cpal backend is opt-in and lives behind a feature flag
/// (see `RealtimeBackend`).
pub struct MockBackend {
    pub events: Vec<PipelineEvent>,
    pub interval: Duration,
    pub stop_after: Option<usize>,
}

impl Default for MockBackend {
    fn default() -> Self {
        let started = Instant::now();
        Self {
            // A canned demo transcript that exercises all four event
            // variants, so first-time users see something on screen.
            events: vec![
                PipelineEvent::TranscriptOnly {
                    at: started,
                    text: "Hello".to_string(),
                },
                PipelineEvent::Translated {
                    at: started,
                    stt_ms: 120,
                    mt_ms: 80,
                    text: "Hello".to_string(),
                    translated: "你好".to_string(),
                },
                PipelineEvent::TranscriptOnly {
                    at: started,
                    text: "How are you today?".to_string(),
                },
                PipelineEvent::Translated {
                    at: started,
                    stt_ms: 210,
                    mt_ms: 95,
                    text: "How are you today?".to_string(),
                    translated: "你今天怎么样？".to_string(),
                },
                PipelineEvent::NoSpeech { at: started },
                PipelineEvent::Error {
                    at: started,
                    stage: PipelineStage::MachineTranslation,
                    message: "demo: simulated network timeout".to_string(),
                },
            ],
            interval: Duration::from_millis(900),
            stop_after: None,
        }
    }
}

impl Backend for MockBackend {
    fn run(self: Box<Self>, tx: Sender<PipelineEvent>, budget: Duration) {
        let deadline = Instant::now() + budget;
        let mut emitted = 0usize;
        loop {
            // We index by `emitted` and wrap around, so the user sees
            // the demo loop forever (until they hit Stop) instead of
            // running out of events.
            let ev = self.events[emitted % self.events.len()].clone();
            if tx.send(ev).is_err() {
                return; // UI closed the channel.
            }
            emitted += 1;
            if let Some(limit) = self.stop_after {
                if emitted >= limit {
                    return;
                }
            }
            if Instant::now() >= deadline {
                return;
            }
            std::thread::sleep(self.interval);
        }
    }
}

/// Real-time backend: drives the pipeline on synthetic 1-second
/// chunks through a `StubTranslator` so the UI has something to
/// render even when no real STT/MT back-end is wired up.
///
/// The chunks are silence, so the default `Transcriber` placeholder
/// returns a fixed Chinese string. This is enough to demonstrate the
/// full round-trip in the UI without requiring faster-whisper or a
/// network MT endpoint.
///
/// When the real STT/MT engines are wired in, swap `DefaultTranscriberRunner`
/// + `StubTranslator` for the production implementations — the rest of
///   the UI doesn't need to change.
pub struct RealtimeBackend {
    pub config: PipelineConfig,
    /// 16 kHz mono, 1 second per chunk.
    pub chunk_samples: usize,
    /// Hard ceiling on wall-clock time the backend is allowed to run.
    pub budget: Duration,
}

impl Default for RealtimeBackend {
    fn default() -> Self {
        Self {
            config: PipelineConfig::default(),
            chunk_samples: 16_000,
            budget: Duration::from_secs(60 * 60),
        }
    }
}

impl Backend for RealtimeBackend {
    fn run(self: Box<Self>, tx: Sender<PipelineEvent>, budget: Duration) {
        // Use the smaller of the two budgets so the caller's hard
        // timeout always wins.
        let effective_budget = budget.min(self.budget);
        let deadline = Instant::now() + effective_budget;
        let pipeline = Pipeline::new(self.config);
        let translator = StubTranslator::new();
        let chunk = vec![0.0f32; self.chunk_samples];
        while Instant::now() < deadline {
            // `process_chunk` is async; we use a fresh single-thread
            // runtime per chunk to avoid leaking threads. Cheap, since
            // the placeholder STT/MT are both in-process.
            let event = match run_blocking(&pipeline, &translator, &chunk) {
                Ok(ev) => ev,
                Err(e) => PipelineEvent::Error {
                    at: Instant::now(),
                    stage: PipelineStage::SpeechToText,
                    message: e.to_string(),
                },
            };
            if tx.send(event).is_err() {
                return;
            }
            // Pace ourselves so we don't burn CPU on a tight loop
            // when the pipeline returns instantly (which it does today).
            std::thread::sleep(Duration::from_millis(250));
        }
    }
}

fn run_blocking(
    pipeline: &Pipeline,
    translator: &dyn Translator,
    chunk: &[f32],
) -> Result<PipelineEvent, String> {
    // Tiny per-call runtime. The pipeline itself is pure-Rust + async,
    // so we don't need a global runtime.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    rt.block_on(pipeline.process_chunk(chunk, translator))
        .map_err(|e| e.to_string())
}

/// The eframe `App`. Owns the receiver, the rolling log, and the
/// current `RunState`. Does not own the worker task directly — instead
/// the binary's `main` owns the `JoinHandle` (or the worker is detached).
///
/// We keep the struct field-by-field (no `eframe::App` blanket impl on
/// a builder) because clippy flags blanket-impl App on a 12-field
/// struct as `result_large_err` and we want the public surface to be
/// constructable in tests.
pub struct VoiceTranslatorApp {
    pub log: UiLog,
    pub selection: UiLanguageSelection,
    pub run_state: RunState,
    pub status_message: String,
    rx: Option<Receiver<PipelineEvent>>,
    /// Channel depth we hand to the worker. 64 events is ~1 minute at
    /// 1 Hz, plenty of headroom without unbounded buffering.
    pub channel_capacity: usize,
    /// How long a single worker call runs before re-spawning. Keeps
    /// the worker's borrow on the Backend short so Stop is responsive.
    pub worker_budget: Duration,
}

impl Default for VoiceTranslatorApp {
    fn default() -> Self {
        Self {
            log: UiLog::default(),
            selection: UiLanguageSelection::default(),
            run_state: RunState::Stopped,
            status_message: "Idle. Click Start to begin.".to_string(),
            rx: None,
            channel_capacity: 64,
            worker_budget: Duration::from_secs(30),
        }
    }
}

impl VoiceTranslatorApp {
    /// Spawn a worker that drives `backend` and pushes events to a
    /// channel owned by `self`. Panics if a worker is already running —
    /// callers must check `run_state` first.
    pub fn start<B: Backend + Send + 'static>(&mut self, backend: B) {
        assert_eq!(self.run_state, RunState::Stopped, "already running");
        // `channel` (unbounded) rather than `sync_channel`: the worker
        // runs in its own thread, so backpressure is unnecessary and an
        // unbounded buffer is fine. We bound memory at the log layer.
        let (tx, rx) = mpsc::channel::<PipelineEvent>();
        let budget = self.worker_budget;
        std::thread::spawn(move || {
            let backend: Box<dyn Backend> = Box::new(backend);
            backend.run(tx, budget);
        });
        self.rx = Some(rx);
        self.run_state = RunState::Running;
        self.status_message = "Running…".to_string();
    }

    /// Stop the worker. The worker will exit at the next `tx.send` after
    /// the channel is dropped, so this is best-effort — we set the flag
    /// and let the natural exit path close the channel.
    pub fn stop(&mut self) {
        if self.run_state == RunState::Running {
            self.run_state = RunState::Stopping;
            self.status_message = "Stopping…".to_string();
            // Drop the receiver; the sender inside the worker will hit
            // a `SendError` on the next push and exit.
            self.rx = None;
        }
    }

    /// Drain any pending events from the channel into the log. Call
    /// this every frame from `App::update`.
    ///
    /// Returns the number of events drained (useful in tests).
    pub fn drain_events(&mut self) -> usize {
        let mut drained = 0usize;
        let Some(rx) = self.rx.as_ref() else {
            return 0;
        };
        loop {
            match rx.try_recv() {
                Ok(event) => {
                    self.log.push(UiLogEntry::from_event(&event));
                    drained += 1;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    // Worker exited cleanly (or was stopped). Flip
                    // back to Stopped so the user can start again.
                    self.run_state = RunState::Stopped;
                    self.status_message = "Idle.".to_string();
                    self.rx = None;
                    break;
                }
            }
        }
        drained
    }
}

/// eframe `App` implementation. The UI is intentionally a single panel:
///
/// - Top bar: language pickers, Start/Stop button, status line.
/// - Body: scrolling log of [`UiLogEntry`].
///
/// The minimal API is deliberate — once the real audio backend lands,
/// the body needs a sidebar for VAD / chunk-size controls, but that's
/// a follow-up.
impl eframe::App for VoiceTranslatorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain pending events first so the log shows the freshest
        // state when we render.
        self.drain_events();

        // Top control bar.
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Source:");
                egui::ComboBox::from_id_salt("source_lang")
                    .selected_text(self.selection.source.as_str().to_uppercase())
                    .show_ui(ui, |ui| {
                        for &lang in crate::translation::SUPPORTED_LANGUAGES {
                            ui.selectable_value(
                                &mut self.selection.source,
                                lang,
                                lang.as_str().to_uppercase(),
                            );
                        }
                    });
                ui.label("→ Target:");
                egui::ComboBox::from_id_salt("target_lang")
                    .selected_text(self.selection.target.as_str().to_uppercase())
                    .show_ui(ui, |ui| {
                        for &lang in crate::translation::SUPPORTED_LANGUAGES {
                            ui.selectable_value(
                                &mut self.selection.target,
                                lang,
                                lang.as_str().to_uppercase(),
                            );
                        }
                    });
                ui.separator();
                let was_running = self.run_state.is_running();
                if was_running {
                    if ui.button("⏹ Stop").clicked() {
                        self.stop();
                    }
                } else if ui.button("▶ Start").clicked() {
                    // For now: re-spawn the mock backend with the
                    // current language selection. A real audio backend
                    // would be picked here once cpal-based capture is
                    // wired up.
                    self.start(MockBackend::default());
                }
                ui.separator();
                if ui.button("Clear log").clicked() {
                    self.log.clear();
                }
            });
            ui.label(egui::RichText::new(&self.status_message).weak());
        });

        // Body: scrolling log.
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for entry in self.log.iter() {
                        let (icon, color) = match entry.stage {
                            EntryStage::Translated => ("🗣", egui::Color32::LIGHT_GREEN),
                            EntryStage::Transcript => ("…", egui::Color32::LIGHT_GRAY),
                            EntryStage::Error => ("⚠", egui::Color32::LIGHT_RED),
                            EntryStage::NoSpeech => ("·", egui::Color32::DARK_GRAY),
                        };
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(icon).color(color));
                            if entry.stage == EntryStage::Translated {
                                ui.label(
                                    egui::RichText::new(&entry.text)
                                        .strong()
                                        .color(egui::Color32::WHITE),
                                );
                                ui.label(egui::RichText::new("→").weak());
                                ui.label(
                                    egui::RichText::new(&entry.translated)
                                        .color(egui::Color32::LIGHT_BLUE),
                                );
                            } else if !entry.text.is_empty() {
                                ui.label(&entry.text);
                            }
                        });
                    }
                });
        });

        // While running, request a repaint at the frame rate so the
        // log scrolls as events arrive.
        if self.run_state.is_running() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn ev_translated(text: &str, translated: &str) -> PipelineEvent {
        PipelineEvent::Translated {
            at: Instant::now(),
            stt_ms: 1,
            mt_ms: 1,
            text: text.to_string(),
            translated: translated.to_string(),
        }
    }

    fn ev_error(msg: &str) -> PipelineEvent {
        PipelineEvent::Error {
            at: Instant::now(),
            stage: PipelineStage::MachineTranslation,
            message: msg.to_string(),
        }
    }

    #[test]
    fn log_respects_capacity_bound() {
        let mut log = UiLog::with_capacity(3);
        for i in 0..10 {
            log.push(UiLogEntry {
                at: Instant::now(),
                stage: EntryStage::Transcript,
                text: format!("line {i}"),
                translated: String::new(),
            });
        }
        assert_eq!(log.len(), 3);
        // The newest three should be 7, 8, 9.
        let texts: Vec<&str> = log.iter().map(|e| e.text.as_str()).collect();
        assert_eq!(texts, vec!["line 7", "line 8", "line 9"]);
    }

    #[test]
    fn log_clear_empties_entries() {
        let mut log = UiLog::default();
        log.push(UiLogEntry {
            at: Instant::now(),
            stage: EntryStage::Translated,
            text: "x".into(),
            translated: "y".into(),
        });
        assert_eq!(log.len(), 1);
        log.clear();
        assert!(log.is_empty());
    }

    #[test]
    fn entry_from_translated_event_captures_both_sides() {
        let ev = ev_translated("Hello", "你好");
        let entry = UiLogEntry::from_event(&ev);
        assert_eq!(entry.stage, EntryStage::Translated);
        assert_eq!(entry.text, "Hello");
        assert_eq!(entry.translated, "你好");
    }

    #[test]
    fn entry_from_error_event_puts_message_in_text_field() {
        let ev = ev_error("network down");
        let entry = UiLogEntry::from_event(&ev);
        assert_eq!(entry.stage, EntryStage::Error);
        assert!(entry.text.contains("network down"));
    }

    #[test]
    fn entry_from_no_speech_event_has_empty_text() {
        let ev = PipelineEvent::NoSpeech { at: Instant::now() };
        let entry = UiLogEntry::from_event(&ev);
        assert_eq!(entry.stage, EntryStage::NoSpeech);
        assert!(entry.text.is_empty());
        assert!(entry.translated.is_empty());
    }

    #[test]
    fn mock_backend_pushes_synthetic_events_to_channel() {
        let mock = MockBackend {
            events: vec![ev_translated("a", "α")],
            interval: Duration::from_millis(5),
            stop_after: Some(1),
        };
        let (tx, rx) = mpsc::channel::<PipelineEvent>();
        let backend: Box<dyn Backend> = Box::new(mock);
        backend.run(tx, Duration::from_secs(2));
        let got = rx.try_recv().expect("mock should have sent one event");
        match got {
            PipelineEvent::Translated {
                text, translated, ..
            } => {
                assert_eq!(text, "a");
                assert_eq!(translated, "α");
            }
            other => panic!("expected Translated, got {:?}", other),
        }
        // After `stop_after` is reached, the worker drops `tx` and the
        // next try_recv yields `Disconnected` (not `Empty`).
        assert!(matches!(rx.try_recv(), Err(TryRecvError::Disconnected)));
    }

    #[test]
    fn start_then_drain_collects_events_into_log() {
        let mut app = VoiceTranslatorApp {
            worker_budget: Duration::from_secs(2),
            ..Default::default()
        };
        let mock = MockBackend {
            events: vec![
                ev_translated("one", "一"),
                ev_translated("two", "二"),
                ev_translated("three", "三"),
            ],
            interval: Duration::from_millis(5),
            stop_after: Some(3),
        };
        app.start(mock);
        // Give the worker a moment.
        std::thread::sleep(Duration::from_millis(50));
        let drained = app.drain_events();
        assert!(drained >= 1, "expected at least one event, got {drained}");
        // After the worker hits `stop_after`, the channel closes and
        // the next drain flips state to Stopped.
        std::thread::sleep(Duration::from_millis(50));
        app.drain_events();
        assert_eq!(app.run_state, RunState::Stopped);
        // Log got at least one entry.
        assert!(!app.log.is_empty());
    }

    #[test]
    fn stop_drops_receiver_and_marks_stopping() {
        let mut app = VoiceTranslatorApp {
            worker_budget: Duration::from_secs(2),
            ..Default::default()
        };
        let mock = MockBackend::default();
        app.start(mock);
        assert_eq!(app.run_state, RunState::Running);
        app.stop();
        assert_eq!(app.run_state, RunState::Stopping);
        assert!(app.rx.is_none());
    }

    #[test]
    fn selection_defaults_to_en_to_zh() {
        let s = UiLanguageSelection::default();
        assert_eq!(s.source, LanguageCode::En);
        assert_eq!(s.target, LanguageCode::Zh);
    }

    #[test]
    fn default_status_message_is_idle() {
        let app = VoiceTranslatorApp::default();
        assert!(app.status_message.contains("Idle"));
    }
}
