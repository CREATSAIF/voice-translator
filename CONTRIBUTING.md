# Contributing to Voice Translator

Thank you for your interest in contributing!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/CREATSAIF/voice-translator.git
cd voice-translator

# Install Python dependencies
pip install -r requirements.txt

# For Rust compilation (optional)
cargo build --release
```

## Project Structure

```
voice-translator/
├── voice_translator.py    # Main Python application
├── Cargo.toml             # Rust dependencies (optional acceleration)
├── Cargo.lock             # Dependency lock file
├── requirements.txt       # Python dependencies
├── README.md              # Main documentation
├── .github/
│   └── workflows/
│       └── ci.yml         # GitHub Actions CI
├── tests/
│   ├── __init__.py
│   └── test_voice_translator.py  # Test suite
└── src/                   # Rust source (optional)
```

## Coding Standards (Python)

- Follow PEP 8 style guidelines
- Add type hints to all function signatures
- Include docstrings for classes and public functions
- Add unit tests for new features

## Coding Standards (Rust)

- Follow Rust standard formatting (`cargo fmt`)
- Pass clippy checks (`cargo clippy -- -D warnings`)
- Write unit tests for new functionality (`cargo test`)

## Testing

### Python Tests

```bash
# Run Python tests
python3 -m pytest tests/ -v

# Run specific test
python3 -m pytest tests/test_voice_translator.py::TestVAD -v
```

### Rust Tests

```bash
# Run Rust tests
cargo test --verbose
```

## Running the Application

```bash
# Set up FunASR API server (required for transcription)
# The FunASR server should be running at http://localhost:8765

# Run the voice translator
python3 voice_translator.py

# Or use the Rust binary (if compiled)
./target/release/voice-translator
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. For Python: ensure code follows PEP 8
5. For Rust: run `cargo fmt && cargo clippy`
6. Run tests: `python3 -m pytest tests/` or `cargo test`
7. Commit with a clear message
8. Push to your fork and submit a PR

## Code Review Process

- PRs require at least 1 approval
- All CI checks must pass
- Address review feedback promptly
- Squash commits before merging

## FunASR API Configuration

The application requires a FunASR server running locally:

```bash
# Install FunASR
pip install funasr

# Start the server (example)
python -m funasr_server &
# Server runs on http://localhost:8765 by default
```

Environment variables:
- `FUNASR_API_URL` - Override the FunASR server URL (default: `http://localhost:8765`)

## Reporting Issues

- Use GitHub Issues for bugs and feature requests
- Include your OS, Python/Rust version, and error details
- For audio issues, describe your microphone setup
- For API issues, include FunASR server logs
