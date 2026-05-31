#!/usr/bin/env python3
"""
Voice Translator - Real-time speech to translation
Uses FunASR API for transcription and translation

Architecture:
1. Audio capture via pyaudio/sounddevice (microphone input)
2. Voice Activity Detection (energy-based VAD)
3. Speech-to-Text via FunASR API (paraformer-zh model)
4. Translation via HTTP API or local logic
5. Display results in real-time
"""

import os
import sys
import time
import queue
import threading
import argparse
import logging
import json
import base64
import struct
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
FUNASR_API_URL = os.environ.get("FUNASR_API_URL", "http://localhost:8765")
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 100  # 100ms chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 1600 samples
VAD_THRESHOLD = 0.02  # Energy threshold for speech detection
SILENCE_DURATION_MS = 500  # 500ms of silence to end utterance
MAX_UTTERANCE_DURATION_MS = 30000  # 30 seconds max

class AudioCapture:
    """Capture audio from microphone"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.running = False
        self.stream = None
        self.audio_queue = queue.Queue()
        
    def _find_audio_library(self):
        """Find available audio library"""
        try:
            import sounddevice as sd
            return 'sounddevice'
        except ImportError:
            pass
        
        try:
            import pyaudio
            return 'pyaudio'
        except ImportError:
            return None
    
    def start(self):
        """Start audio capture"""
        library = self._find_audio_library()
        if library == 'sounddevice':
            self._start_sounddevice()
        elif library == 'pyaudio':
            self._start_pyaudio()
        else:
            raise RuntimeError("No audio library available. Install sounddevice or pyaudio.")
        
    def _start_sounddevice(self):
        import sounddevice as sd
        
        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio status: {status}")
            self.audio_queue.put(indata.copy())
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            callback=callback,
            blocksize=CHUNK_SIZE
        )
        self.running = True
        self.stream.start()
        logger.info("Audio capture started (sounddevice)")
        
    def _start_pyaudio(self):
        import pyaudio
        
        p = pyaudio.PyAudio()
        self.stream = p.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=lambda in_data, frame_count, time_info, status: (
                (self.audio_queue.put(np.frombuffer(in_data, dtype=np.float32)), 
                 (in_data, pyaudio.paContinue))[0]
            )
        )
        self.running = True
        self.stream.start_stream()
        logger.info("Audio capture started (pyaudio)")
        
    def stop(self):
        """Stop audio capture"""
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        logger.info("Audio capture stopped")
        
    def read(self):
        """Read audio chunk"""
        try:
            return self.audio_queue.get(timeout=1.0)
        except queue.Empty:
            return None


class VAD:
    """Energy-based Voice Activity Detection"""
    
    def __init__(self, threshold=VAD_THRESHOLD):
        self.threshold = threshold
        self.is_speaking = False
        self.silence_frames = 0
        
    def detect(self, audio_chunk):
        """Detect speech in audio chunk"""
        if audio_chunk is None:
            return False
            
        # Calculate energy
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        
        if energy > self.threshold:
            self.is_speaking = True
            self.silence_frames = 0
            return True
        else:
            self.silence_frames += 1
            self.is_speaking = False
            return False
            
    def reset(self):
        self.silence_frames = 0


class TranscriptionService:
    """Transcription using FunASR API"""
    
    def __init__(self, api_url=FUNASR_API_URL):
        self.api_url = api_url
        self.session = None
        
    def _get_session(self):
        if self.session is None:
            import requests
            self.session = requests.Session()
        return self.session
        
    def transcribe(self, audio_data):
        """Send audio to FunASR API for transcription"""
        try:
            import requests
            
            # Convert to bytes (16-bit PCM)
            audio_bytes = struct.pack(f'{len(audio_data)}h', 
                                      *(int(x * 32767) for x in audio_data))
            
            response = self._get_session().post(
                f"{self.api_url}/recognize",
                data=audio_bytes,
                headers={"Content-Type": "audio/pcm"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("result", "No result")
            else:
                logger.error(f"Transcription failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None


class VoiceTranslator:
    """Main voice translator"""
    
    def __init__(self, args):
        self.args = args
        self.audio_capture = AudioCapture()
        self.vad = VAD()
        self.transcriber = TranscriptionService()
        
        self.utterance_buffer = []
        self.last_speech_time = time.time()
        self.running = True
        
    def process_audio(self):
        """Process audio chunks"""
        while self.running:
            audio_chunk = self.audio_capture.read()
            
            if audio_chunk is None:
                continue
                
            is_speech = self.vad.detect(audio_chunk)
            
            if is_speech:
                self.utterance_buffer.append(audio_chunk)
                self.last_speech_time = time.time()
            elif self.utterance_buffer:
                # Check if silence duration exceeded
                silence_ms = (time.time() - self.last_speech_time) * 1000
                if silence_ms > SILENCE_DURATION_MS:
                    # Process utterance
                    self._process_utterance()
                    
    def _process_utterance(self):
        """Process completed utterance"""
        if not self.utterance_buffer:
            return
            
        # Concatenate audio
        audio_data = np.concatenate(self.utterance_buffer)
        self.utterance_buffer = []
        self.vad.reset()
        
        # Skip if too short
        if len(audio_data) < SAMPLE_RATE * 0.3:  # Less than 300ms
            return
            
        # Transcribe
        logger.info(f"Transcribing {len(audio_data)/SAMPLE_RATE:.1f}s of audio...")
        result = self.transcriber.transcribe(audio_data)
        
        if result:
            duration = len(audio_data) / SAMPLE_RATE
            timestamp = time.strftime("%H:%M:%S")
            print(f"\n[{timestamp}] ⏱ {duration:.1f}s")
            print(f"{'─'*60}")
            print(f"  识别: {result}")
            
    def start(self):
        """Start translation"""
        logger.info("Starting voice translator...")
        logger.info(f"FunASR API: {FUNASR_API_URL}")
        
        try:
            self.audio_capture.start()
            self.process_audio()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.audio_capture.stop()
            
    def stop(self):
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="Voice Translator")
    parser.add_argument("--api-url", default=FUNASR_API_URL, 
                       help="FunASR API URL")
    parser.add_argument("--threshold", type=float, default=VAD_THRESHOLD,
                       help="VAD energy threshold")
    parser.add_argument("--silence-ms", type=int, default=SILENCE_DURATION_MS,
                       help="Silence duration to end utterance (ms)")
    
    args = parser.parse_args()
    
    translator = VoiceTranslator(args)
    translator.start()


if __name__ == "__main__":
    main()