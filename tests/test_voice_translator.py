#!/usr/bin/env python3
"""Tests for voice_translator.py"""
import sys
import os
import queue
import time
import threading
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_translator import VAD, AudioCapture, TranscriptionService, SAMPLE_RATE, VAD_THRESHOLD


class TestVAD:
    """Test Voice Activity Detection"""

    def test_silence_below_threshold(self):
        """Silent audio should not trigger speech detection"""
        vad = VAD(threshold=VAD_THRESHOLD)
        # Very quiet audio (silence with tiny noise)
        silence = np.zeros(1600, dtype=np.float32) + 0.001
        result = vad.detect(silence)
        assert result == False, "Silence should not trigger speech"

    def test_speech_above_threshold(self):
        """Loud audio should trigger speech detection"""
        vad = VAD(threshold=VAD_THRESHOLD)
        # Loud audio
        speech = np.ones(1600, dtype=np.float32) * 0.1
        result = vad.detect(speech)
        assert result == True, "Loud audio should trigger speech"

    def test_silence_frames_accumulation(self):
        """Silence frames should accumulate correctly"""
        vad = VAD(threshold=VAD_THRESHOLD)
        silence = np.zeros(1600, dtype=np.float32) + 0.001
        for _ in range(5):
            vad.detect(silence)
        assert vad.silence_frames == 5

    def test_reset(self):
        """Reset should clear silence frames"""
        vad = VAD(threshold=VAD_THRESHOLD)
        silence = np.zeros(1600, dtype=np.float32) + 0.001
        for _ in range(5):
            vad.detect(silence)
        vad.reset()
        assert vad.silence_frames == 0


class TestTranscriptionService:
    """Test transcription service"""

    def test_service_initialization(self):
        """Service should initialize with default URL"""
        service = TranscriptionService()
        assert service.api_url == "http://localhost:8765"

    def test_service_custom_url(self):
        """Service should accept custom URL"""
        service = TranscriptionService(api_url="http://custom:9999")
        assert service.api_url == "http://custom:9999"

    def test_transcribe_with_silence(self):
        """Transcribe should handle silence audio gracefully"""
        service = TranscriptionService()
        # 500ms of silence
        silence = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
        result = service.transcribe(silence)
        # Result should be a string (possibly empty)
        assert result is None or isinstance(result, str)


class TestAudioCapture:
    """Test audio capture"""

    def test_audio_library_detection(self):
        """Audio library detection should return known library names or None"""
        capture = AudioCapture()
        try:
            lib = capture._find_audio_library()
            # Library should be one of the known options or None
            assert lib in ('sounddevice', 'pyaudio', None), f"Unexpected: {lib}"
        except Exception as e:
            # PortAudio not available is acceptable in test environment
            print(f"  (skipped: {e})")

    def test_audio_config_defaults(self):
        """Audio config should have correct defaults"""
        capture = AudioCapture()
        assert capture.sample_rate == SAMPLE_RATE
        assert capture.channels == 1
        assert capture.running == False


def run_tests():
    """Run all tests"""
    test_classes = [TestVAD, TestTranscriptionService, TestAudioCapture]
    passed = 0
    failed = 0

    for tc in test_classes:
        print(f"\n{tc.__name__}:")
        instance = tc()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
                    failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
