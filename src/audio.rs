//! Audio capture module using cpal
//! Captures microphone input in real-time chunks

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Audio capture configuration
#[derive(Debug, Clone)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub buffer_size: usize,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            buffer_size: 4096,
        }
    }
}

/// Audio chunk with timestamp
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub data: Vec<f32>,
    pub sample_rate: u32,
    pub timestamp_us: u64,
}

impl AudioChunk {
    pub fn new(data: Vec<f32>, sample_rate: u32, timestamp_us: u64) -> Self {
        Self {
            data,
            sample_rate,
            timestamp_us,
        }
    }
}

/// Audio capture state
pub struct AudioCapture {
    config: AudioConfig,
    running: Arc<AtomicBool>,
    stream: Option<cpal::Stream>,
}

impl AudioCapture {
    pub fn new(config: AudioConfig) -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or("No input device available")?;

        let supported_config = device
            .default_input_config()
            .map_err(|e| format!("Failed to get default input config: {}", e))?;

        Ok(Self {
            config,
            running: Arc::new(AtomicBool::new(false)),
            stream: None,
        })
    }

    /// Start capturing audio with callback
    pub fn start<F>(&mut self, on_audio: F) -> Result<(), String>
    where
        F: Fn(AudioChunk) + Send + 'static,
    {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or("No input device available")?;

        let config = device
            .default_input_config()
            .map_err(|e| format!("Failed to get input config: {}", e))?;

        let sample_rate = config.sample_rate().0;
        let channels = config.channels();

        let running = self.running.clone();
        running.store(true, Ordering::SeqCst);

        let err_fn = |err| eprintln!("Audio stream error: {}", err);

        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => {
                let on_audio = Arc::new(on_audio);
                device.build_input_stream(
                    &config.into(),
                    move |data: &[f32], _| {
                        if running.load(Ordering::SeqCst) {
                            let chunk = AudioChunk::new(
                                data.to_vec(),
                                sample_rate,
                                std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_micros() as u64,
                            );
                            on_audio(chunk);
                        }
                    },
                    err_fn,
                    None,
                )
            }
            _ => return Err("Unsupported sample format".to_string()),
        }
        .map_err(|e| format!("Failed to build input stream: {}", e))?;

        stream
            .play()
            .map_err(|e| format!("Failed to start stream: {}", e))?;

        self.stream = Some(stream);
        Ok(())
    }

    /// Stop capturing
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        self.stream = None;
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        self.stop();
    }
}
