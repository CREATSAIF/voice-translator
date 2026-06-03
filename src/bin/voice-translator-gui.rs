//! Real-time voice translator GUI — eframe entry point.
//!
//! The binary is intentionally thin: it parses a handful of CLI flags,
//! constructs a [`VoiceTranslatorApp`], and hands it to eframe. All the
//! actual UI logic lives in [`voice_translator::ui`], which is unit-tested
//! without a display server so the GUI can be exercised in CI.
//!
//! # Usage
//!
//! ```text
//! # Default: mock backend (synthetic events, no microphone needed).
//! cargo run --bin voice-translator-gui
//!
//! # Real pipeline (uses the in-process stub STT/MT — useful for
//! # end-to-end smoke tests; swap to faster-whisper / a cloud MT
//! # endpoint once those are wired up).
//! cargo run --bin voice-translator-gui -- --realtime
//!
//! # Different language pair.
//! cargo run --bin voice-translator-gui -- --source zh --target en
//! ```
//!
//! # Exit codes
//!
//! - `0` — clean shutdown via the window close button.
//! - non-zero — eframe returned an error before the run started
//!   (very rare; usually a display-server problem).

use std::time::Duration;

use voice_translator::translation::{LanguageCode, SUPPORTED_LANGUAGES};
use voice_translator::ui::{MockBackend, RealtimeBackend, VoiceTranslatorApp};

/// Command-line configuration. CLI parsing is hand-rolled (no `clap`
/// dep) because the surface area is small and a hand-written parser
/// keeps the binary free of proc-macro dependencies.
#[derive(Debug, Clone)]
struct Args {
    /// Use the real-time cpal-based backend instead of the mock.
    realtime: bool,
    source: LanguageCode,
    target: LanguageCode,
    /// Worker budget (seconds) — how long one backend run is allowed
    /// before the worker exits and the UI flips to Stopped. Useful for
    /// CI / smoke tests; 0 means "use the default".
    budget_secs: u64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            realtime: false,
            source: LanguageCode::En,
            target: LanguageCode::Zh,
            budget_secs: 0,
        }
    }
}

fn parse_lang(s: &str) -> Result<LanguageCode, String> {
    SUPPORTED_LANGUAGES
        .iter()
        .copied()
        .find(|l| l.as_str() == s.to_ascii_lowercase())
        .ok_or_else(|| {
            format!(
                "unsupported language '{}' (supported: {})",
                s,
                SUPPORTED_LANGUAGES
                    .iter()
                    .map(|l| l.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut iter = std::env::args().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--realtime" => args.realtime = true,
            "--mock" => args.realtime = false,
            "--source" => {
                let v = iter.next().ok_or("--source needs a value")?;
                args.source = parse_lang(&v)?;
            }
            "--target" => {
                let v = iter.next().ok_or("--target needs a value")?;
                args.target = parse_lang(&v)?;
            }
            "--budget" => {
                let v = iter.next().ok_or("--budget needs a value in seconds")?;
                args.budget_secs = v
                    .parse::<u64>()
                    .map_err(|e| format!("invalid --budget: {e}"))?;
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    if args.source == args.target {
        return Err(format!(
            "source and target must differ (both are '{}')",
            args.source.as_str()
        ));
    }
    Ok(args)
}

fn print_help() {
    eprintln!("voice-translator-gui — eframe frontend for real-time voice translation");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("    voice-translator-gui [FLAGS]");
    eprintln!();
    eprintln!("FLAGS:");
    eprintln!("    --mock              Use the mock backend (default, no microphone needed).");
    eprintln!("    --realtime          Use the real-time pipeline backend.");
    eprintln!("    --source LANG       Source language (default: en).");
    eprintln!("    --target LANG       Target language (default: zh).");
    eprintln!("    --budget SECONDS    Worker budget in seconds (0 = default 30s).");
    eprintln!("    -h, --help          Print this help and exit.");
    eprintln!();
    eprintln!(
        "SUPPORTED LANGUAGES: {}",
        SUPPORTED_LANGUAGES
            .iter()
            .map(|l| l.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = parse_args().map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
        Box::<dyn std::error::Error + Send + Sync>::from(format!("{e}\n\ntry --help"))
    })?;

    let mut app = VoiceTranslatorApp::default();
    app.selection.source = args.source;
    app.selection.target = args.target;
    if args.budget_secs > 0 {
        app.worker_budget = Duration::from_secs(args.budget_secs);
    }

    // Pick the backend per the CLI flag. The mock backend is the
    // default because it produces a canned demo transcript without
    // requiring a microphone or a network MT endpoint, so the binary
    // is useful for a first-time user just trying the UI.
    if args.realtime {
        let backend = RealtimeBackend {
            config: voice_translator::pipeline::PipelineConfig {
                source: args.source,
                target: args.target,
                ..Default::default()
            },
            ..Default::default()
        };
        app.start(backend);
    } else {
        app.start(MockBackend::default());
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Voice Translator")
            .with_inner_size([720.0, 540.0]),
        ..Default::default()
    };

    // `eframe::Error` doesn't satisfy `Send + Sync` (it carries a
    // `winit` PlatformError with a raw pointer), so we can't box it
    // into `Box<dyn Error + Send + Sync>` directly. Format it as a
    // string and box *that* — same approach as the egui-029-fixes
    // skill recommends.
    eframe::run_native(
        "Voice Translator",
        native_options,
        Box::new(|_cc| Ok(Box::new(app))),
    )
    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
        Box::<dyn std::error::Error + Send + Sync>::from(e.to_string())
    })
}
