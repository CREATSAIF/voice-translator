//! Voice Translator - Real-time speech-to-speech translation
//!
//! Architecture:
//! 1. Audio capture via cpal (real-time microphone input)
//! 2. Voice Activity Detection (VAD) to detect speech segments
//! 3. Speech-to-Text via faster-whisper (local)
//! 4. Translation via API or local model
//! 5. Display results in real-time
//!
//! The end-to-end orchestrator lives in [`pipeline`]; the eframe
//! frontend lives in [`ui`]. See README.md for the development roadmap.

pub mod audio;
pub mod pipeline;
pub mod transcription;
pub mod translation;
pub mod ui;
pub mod vad;

pub use audio::AudioCapture;
pub use pipeline::{Pipeline, PipelineConfig, PipelineEvent, PipelineStage};
pub use transcription::Transcriber;
pub use translation::{CloudTranslator, LanguageCode, StubTranslator, Translator};
pub use ui::{
    Backend, MockBackend, RealtimeBackend, RunState, UiLanguageSelection, UiLog, UiLogEntry,
    VoiceTranslatorApp,
};
pub use vad::{rms_energy, Vad, VadConfig, VadEvent};
