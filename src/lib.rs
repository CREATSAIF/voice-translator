//! Voice Translator - Real-time speech-to-speech translation
//!
//! Architecture:
//! 1. Audio capture via cpal (real-time microphone input)
//! 2. Voice Activity Detection (VAD) to detect speech segments
//! 3. Speech-to-Text via faster-whisper (local)
//! 4. Translation via API or local model
//! 5. Display results in real-time
//!
//! Note: `translation` and `ui` modules are planned but not yet implemented.
//! See README.md for the development roadmap.

pub mod audio;
pub mod transcription;

pub use audio::AudioCapture;
pub use transcription::Transcriber;
