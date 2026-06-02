//! Voice Activity Detection (energy-based)
//!
//! Mirrors the Python `VAD` class in `voice_translator.py` so the Rust and
//! Python implementations behave identically for the same input.
//!
//! Algorithm:
//!   energy = sqrt(mean(audio^2))            (RMS)
//!   if energy > threshold: speech frame, reset silence counter
//!   else: silence frame, increment silence counter
//!
//! The richer [`VadEvent`] return type also signals when an utterance has
//! ended (silence has lasted longer than `silence_chunks_to_end`).

/// VAD configuration
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// RMS energy above which a frame is considered speech.
    /// Defaults to 0.02 to match the Python `VAD_THRESHOLD` constant.
    pub threshold: f32,
    /// Number of consecutive silence chunks that ends an utterance.
    /// `0` disables utterance-end detection. Defaults to 5 (500ms at 100ms chunks).
    pub silence_chunks_to_end: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.02,
            silence_chunks_to_end: 5,
        }
    }
}

/// Events emitted by [`Vad::detect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    /// This frame contained speech (energy above threshold).
    Speech,
    /// This frame was silence, but the utterance has not yet ended.
    Silence,
    /// The utterance has ended (silence exceeded `silence_chunks_to_end`).
    /// Also returned for every subsequent silence frame until [`Vad::reset`].
    UtteranceEnded,
}

/// Energy-based Voice Activity Detection.
#[derive(Debug, Clone)]
pub struct Vad {
    config: VadConfig,
    is_speaking: bool,
    silence_chunks: u32,
}

impl Vad {
    /// Create a new VAD with the given config.
    pub fn new(config: VadConfig) -> Self {
        Self {
            config,
            is_speaking: false,
            silence_chunks: 0,
        }
    }

    /// Create a VAD with a custom threshold (and default silence-end behavior).
    pub fn with_threshold(threshold: f32) -> Self {
        Self::new(VadConfig {
            threshold,
            ..VadConfig::default()
        })
    }

    /// Process one audio frame and return the event.
    ///
    /// `audio` is a slice of mono `f32` samples in the range `[-1.0, 1.0]`.
    /// An empty slice is treated as silence.
    pub fn detect(&mut self, audio: &[f32]) -> VadEvent {
        if audio.is_empty() {
            return self.handle_silence();
        }

        let energy = rms_energy(audio);
        if energy > self.config.threshold {
            self.is_speaking = true;
            self.silence_chunks = 0;
            VadEvent::Speech
        } else {
            self.handle_silence()
        }
    }

    fn handle_silence(&mut self) -> VadEvent {
        self.silence_chunks += 1;
        self.is_speaking = false;

        if self.config.silence_chunks_to_end > 0
            && self.silence_chunks >= self.config.silence_chunks_to_end
        {
            VadEvent::UtteranceEnded
        } else {
            VadEvent::Silence
        }
    }

    /// Reset internal state (silence counter and speaking flag).
    pub fn reset(&mut self) {
        self.silence_chunks = 0;
        self.is_speaking = false;
    }

    /// Whether the most recent frame was classified as speech.
    pub fn is_speaking(&self) -> bool {
        self.is_speaking
    }

    /// Number of consecutive silence frames since the last speech frame (or reset).
    pub fn silence_chunks(&self) -> u32 {
        self.silence_chunks
    }

    /// Current threshold (for inspection / debugging).
    pub fn threshold(&self) -> f32 {
        self.config.threshold
    }
}

/// Root mean square energy of an audio frame.
///
/// Equivalent to `numpy.sqrt(numpy.mean(audio**2))`. Returns 0.0 for an empty slice.
pub fn rms_energy(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }
    let mut sum_sq = 0.0_f64;
    for &s in audio {
        let s = s as f64;
        sum_sq += s * s;
    }
    (sum_sq / audio.len() as f64).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_is_below_default_threshold() {
        let mut vad = Vad::with_threshold(0.02);
        let silence: Vec<f32> = vec![0.0; 1600];
        assert_eq!(vad.detect(&silence), VadEvent::Silence);
        assert!(!vad.is_speaking());
    }

    #[test]
    fn loud_audio_triggers_speech() {
        let mut vad = Vad::with_threshold(0.02);
        let speech: Vec<f32> = vec![0.1; 1600];
        assert_eq!(vad.detect(&speech), VadEvent::Speech);
        assert!(vad.is_speaking());
        assert_eq!(vad.silence_chunks(), 0);
    }

    #[test]
    fn silence_counter_increments() {
        let mut vad = Vad::with_threshold(0.02);
        let silence: Vec<f32> = vec![0.0; 1600];
        for _ in 0..5 {
            vad.detect(&silence);
        }
        assert_eq!(vad.silence_chunks(), 5);
    }

    #[test]
    fn reset_clears_state() {
        let mut vad = Vad::with_threshold(0.02);
        let silence: Vec<f32> = vec![0.0; 1600];
        for _ in 0..5 {
            vad.detect(&silence);
        }
        vad.reset();
        assert_eq!(vad.silence_chunks(), 0);
        assert!(!vad.is_speaking());
    }

    #[test]
    fn speech_resets_silence_counter() {
        let mut vad = Vad::with_threshold(0.02);
        let silence: Vec<f32> = vec![0.0; 1600];
        let speech: Vec<f32> = vec![0.1; 1600];
        vad.detect(&silence);
        vad.detect(&silence);
        assert_eq!(vad.silence_chunks(), 2);
        vad.detect(&speech);
        assert_eq!(vad.silence_chunks(), 0);
    }

    #[test]
    fn utterance_ends_after_configured_silence() {
        let mut vad = Vad::new(VadConfig {
            threshold: 0.02,
            silence_chunks_to_end: 3,
        });
        let silence: Vec<f32> = vec![0.0; 1600];
        assert_eq!(vad.detect(&silence), VadEvent::Silence);
        assert_eq!(vad.detect(&silence), VadEvent::Silence);
        assert_eq!(vad.detect(&silence), VadEvent::UtteranceEnded);
        // subsequent silence frames keep emitting UtteranceEnded
        assert_eq!(vad.detect(&silence), VadEvent::UtteranceEnded);
    }

    #[test]
    fn empty_audio_is_silence() {
        let mut vad = Vad::with_threshold(0.02);
        assert_eq!(vad.detect(&[]), VadEvent::Silence);
    }

    #[test]
    fn rms_energy_of_zeros_is_zero() {
        assert_eq!(rms_energy(&[0.0; 100]), 0.0);
    }

    #[test]
    fn rms_energy_of_constant() {
        // RMS of a constant |c| is exactly |c|.
        let audio = vec![0.1_f32; 1000];
        let e = rms_energy(&audio);
        assert!((e - 0.1).abs() < 1e-6, "expected 0.1, got {e}");
    }

    #[test]
    fn rms_energy_of_empty_is_zero() {
        assert_eq!(rms_energy(&[]), 0.0);
    }

    #[test]
    fn rms_energy_silent_below_threshold() {
        // Tiny noise: 0.001 amplitude -> RMS = 0.001 < threshold 0.02
        let silence = vec![0.001_f32; 1600];
        let mut vad = Vad::with_threshold(0.02);
        assert_eq!(vad.detect(&silence), VadEvent::Silence);
    }
}
