//! Transcription module using faster-whisper
//! Runs Whisper locally for speech-to-text

use std::process::Command;

/// Transcriber configuration
#[derive(Debug, Clone)]
pub struct TranscriptionConfig {
    pub model_size: String, // e.g., "tiny", "base", "small", "medium"
    pub language: Option<String>,
    pub device: String, // "cuda" or "cpu"
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            model_size: "base".to_string(),
            language: Some("zh".to_string()),
            device: "cuda".to_string(),
        }
    }
}

/// Transcription result
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub language: String,
    pub confidence: f32,
    pub start_time_ms: u64,
    pub end_time_ms: u64,
}

/// Transcriber using faster-whisper
pub struct Transcriber {
    config: TranscriptionConfig,
}

impl Transcriber {
    pub fn new(config: TranscriptionConfig) -> Self {
        Self { config }
    }

    /// Transcribe audio data
    pub async fn transcribe(&self, audio_data: &[f32]) -> Result<TranscriptionResult, String> {
        // Placeholder - actual implementation would use faster-whisper
        // For now, return a mock result
        Ok(TranscriptionResult {
            text: "测试文本".to_string(),
            language: self
                .config
                .language
                .clone()
                .unwrap_or_else(|| "zh".to_string()),
            confidence: 0.95,
            start_time_ms: 0,
            end_time_ms: 1000,
        })
    }

    /// Check if faster-whisper is available
    pub fn is_available() -> bool {
        // Check if python with faster-whisper is available
        let output = Command::new("python3")
            .args(["-c", "import faster_whisper; print('ok')"])
            .output();

        output.map(|o| o.status.success()).unwrap_or(false)
    }
}
