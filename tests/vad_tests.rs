//! Integration tests for the VAD module.
//!
//! These tests live in `tests/` (not `src/`) so they exercise the public
//! API the same way downstream code would.

use voice_translator::vad::{rms_energy, Vad, VadConfig, VadEvent};

#[test]
fn detect_silence_returns_silence() {
    let mut vad = Vad::with_threshold(0.02);
    let silence: Vec<f32> = vec![0.0; 1600];
    assert_eq!(vad.detect(&silence), VadEvent::Silence);
    assert!(!vad.is_speaking());
}

#[test]
fn detect_speech_returns_speech() {
    let mut vad = Vad::with_threshold(0.02);
    let speech: Vec<f32> = vec![0.1; 1600];
    assert_eq!(vad.detect(&speech), VadEvent::Speech);
    assert!(vad.is_speaking());
}

#[test]
fn silence_counter_accumulates() {
    let mut vad = Vad::with_threshold(0.02);
    let silence: Vec<f32> = vec![0.0; 1600];
    for _ in 0..7 {
        vad.detect(&silence);
    }
    assert_eq!(vad.silence_chunks(), 7);
}

#[test]
fn reset_clears_silence_counter() {
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
    assert_eq!(vad.detect(&silence), VadEvent::UtteranceEnded);
}

#[test]
fn silence_chunks_to_end_zero_disables_ending() {
    let mut vad = Vad::new(VadConfig {
        threshold: 0.02,
        silence_chunks_to_end: 0,
    });
    let silence: Vec<f32> = vec![0.0; 1600];
    for _ in 0..20 {
        assert_eq!(vad.detect(&silence), VadEvent::Silence);
    }
}

#[test]
fn empty_audio_frame_is_silence() {
    let mut vad = Vad::with_threshold(0.02);
    assert_eq!(vad.detect(&[]), VadEvent::Silence);
}

#[test]
fn rms_zero_for_zeros() {
    assert_eq!(rms_energy(&[0.0_f32; 1024]), 0.0);
}

#[test]
fn rms_zero_for_empty() {
    assert_eq!(rms_energy(&[]), 0.0);
}

#[test]
fn rms_matches_numpy_for_constant() {
    // numpy: np.sqrt(np.mean(np.full(1000, 0.1)**2)) == 0.1
    let audio = vec![0.1_f32; 1000];
    let e = rms_energy(&audio);
    assert!((e - 0.1).abs() < 1e-6, "expected 0.1, got {e}");
}

#[test]
fn rms_silent_below_default_threshold() {
    // 0.001 amplitude -> RMS = 0.001 < 0.02 (default threshold)
    let silence = vec![0.001_f32; 1600];
    let mut vad = Vad::with_threshold(0.02);
    assert_eq!(vad.detect(&silence), VadEvent::Silence);
}

#[test]
fn threshold_accessor_returns_configured_value() {
    let vad = Vad::with_threshold(0.05);
    assert!((vad.threshold() - 0.05).abs() < 1e-9);
}
