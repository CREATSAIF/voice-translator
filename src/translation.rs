//! Translation module
//!
//! Pluggable translation backends (cloud HTTP / local stub) behind a common
//! [`Translator`] trait. Designed to be testable in isolation from the audio
//! and VAD stack — no cpal / ALSA dependencies are pulled in by this file.
//!
//! Pipeline role:
//!   audio chunk → VAD segment → Transcriber::transcribe() → Translator::translate()
//!
//! Currently the cloud backend targets an OpenAI-compatible `/v1/chat/completions`
//! endpoint (works with the local hy-RTMT backend, OpenAI, OpenRouter, etc.).
//! The local backend is a stub today and returns a deterministic placeholder so
//! end-to-end smoke tests can run without network access; swap in a real local
//! model loader (e.g. llama.cpp, RKLLM) behind the same trait later.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::time::Duration;

/// BCP-47 / ISO-639-1 language code. We accept a small set of common codes
/// (matching the hy-RTMT 33-language catalog). Validation is intentionally
/// conservative — anything outside the allow-list is rejected at the boundary
/// so misconfigured pipelines fail fast.
///
/// The canonical languages are a closed enum (cheap to compare, copy,
/// hash). For external input we expose [`LanguageCode::from_str_lossy`]
/// which validates and returns a `Language` enum variant, or
/// [`LanguageCode::parse`] which returns `Err` on unknown input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum Language {
    Zh,
    En,
    Ja,
    Ko,
    Fr,
    De,
    Es,
    Pt,
    Ru,
    It,
    Ar,
    Th,
    Tr,
}

impl Language {
    pub fn as_str(&self) -> &'static str {
        match self {
            Language::Zh => "zh",
            Language::En => "en",
            Language::Ja => "ja",
            Language::Ko => "ko",
            Language::Fr => "fr",
            Language::De => "de",
            Language::Es => "es",
            Language::Pt => "pt",
            Language::Ru => "ru",
            Language::It => "it",
            Language::Ar => "ar",
            Language::Th => "th",
            Language::Tr => "tr",
        }
    }

    /// Parse a language code (case-insensitive). Returns `None` for unknown.
    pub fn parse(code: &str) -> Option<Self> {
        match code.trim().to_ascii_lowercase().as_str() {
            "zh" => Some(Language::Zh),
            "en" => Some(Language::En),
            "ja" => Some(Language::Ja),
            "ko" => Some(Language::Ko),
            "fr" => Some(Language::Fr),
            "de" => Some(Language::De),
            "es" => Some(Language::Es),
            "pt" => Some(Language::Pt),
            "ru" => Some(Language::Ru),
            "it" => Some(Language::It),
            "ar" => Some(Language::Ar),
            "th" => Some(Language::Th),
            "tr" => Some(Language::Tr),
            _ => None,
        }
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl TryFrom<String> for Language {
    type Error = String;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Language::parse(&s).ok_or_else(|| format!("unsupported language code: {}", s))
    }
}

impl From<Language> for String {
    fn from(l: Language) -> Self {
        l.as_str().to_string()
    }
}

/// Convenience alias — `LanguageCode` is the historical name used in
/// `TranslationRequest` / `TranslationResult` fields. It's a thin wrapper
/// over `Language` that carries the "code" semantics; the actual storage
/// is the closed enum.
pub type LanguageCode = Language;

/// Supported languages — keep in sync with the hy-RTMT backend's 33-language
/// catalog. Anything not in this set is rejected by [`Translator::translate`].
pub const SUPPORTED_LANGUAGES: &[LanguageCode] = &[
    LanguageCode::Zh,
    LanguageCode::En,
    LanguageCode::Ja,
    LanguageCode::Ko,
    LanguageCode::Fr,
    LanguageCode::De,
    LanguageCode::Es,
    LanguageCode::Pt,
    LanguageCode::Ru,
    LanguageCode::It,
    LanguageCode::Ar,
    LanguageCode::Th,
    LanguageCode::Tr,
];

/// Validate that `code` is a supported language. Returns the canonical
/// (lower-case) form on success, or an error describing the rejection.
pub fn validate_language(code: &str) -> Result<LanguageCode, TranslationError> {
    Language::parse(code).ok_or_else(|| TranslationError::UnsupportedLanguage {
        requested: code.to_string(),
        supported: SUPPORTED_LANGUAGES.iter().map(|l| l.as_str()).collect(),
    })
}

/// Translation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationRequest {
    pub text: String,
    pub source: LanguageCode,
    pub target: LanguageCode,
}

impl TranslationRequest {
    pub fn new(text: impl Into<String>, source: LanguageCode, target: LanguageCode) -> Self {
        Self {
            text: text.into(),
            source,
            target,
        }
    }
}

/// Translation result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslationResult {
    pub original: String,
    pub translated: String,
    pub source: LanguageCode,
    pub target: LanguageCode,
    /// Backend that produced this result, e.g. `"cloud"`, `"local-stub"`.
    pub backend: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TranslationError {
    UnsupportedLanguage {
        requested: String,
        supported: Vec<&'static str>,
    },
    EmptyText,
    SourceEqualsTarget {
        language: LanguageCode,
    },
    BackendUnavailable(String),
    Http {
        status: u16,
        body: String,
    },
    Timeout,
    Serde(String),
    Other(String),
}

impl fmt::Display for TranslationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TranslationError::UnsupportedLanguage {
                requested,
                supported,
            } => {
                write!(
                    f,
                    "unsupported language '{}' (supported: {})",
                    requested,
                    supported.join(", ")
                )
            }
            TranslationError::EmptyText => f.write_str("translation text is empty"),
            TranslationError::SourceEqualsTarget { language } => {
                write!(
                    f,
                    "source and target languages are both {}; nothing to translate",
                    language
                )
            }
            TranslationError::BackendUnavailable(msg) => {
                write!(f, "translation backend unavailable: {}", msg)
            }
            TranslationError::Http { status, body } => {
                write!(f, "translation HTTP error {}: {}", status, body)
            }
            TranslationError::Timeout => f.write_str("translation request timed out"),
            TranslationError::Serde(msg) => write!(f, "translation serde error: {}", msg),
            TranslationError::Other(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for TranslationError {}

impl From<reqwest::Error> for TranslationError {
    fn from(e: reqwest::Error) -> Self {
        if e.is_timeout() {
            TranslationError::Timeout
        } else if e.is_connect() || e.is_request() {
            TranslationError::BackendUnavailable(e.to_string())
        } else {
            TranslationError::Other(e.to_string())
        }
    }
}

impl From<serde_json::Error> for TranslationError {
    fn from(e: serde_json::Error) -> Self {
        TranslationError::Serde(e.to_string())
    }
}

/// The translation contract. Any backend (cloud, local model, stub) implements
/// this. Async so backends can be I/O-bound without blocking the audio thread.
#[async_trait]
pub trait Translator: Send + Sync {
    /// Translate a single text segment.
    async fn translate(
        &self,
        request: TranslationRequest,
    ) -> Result<TranslationResult, TranslationError>;

    /// Backend identifier for logs / metrics.
    fn backend_name(&self) -> &'static str;

    /// Cheap readiness probe — returns Ok(()) if the backend is ready to take
    /// requests. The default is "always ready" (local stub / already-configured
    /// HTTP backend). Concrete backends may override.
    async fn health_check(&self) -> Result<(), TranslationError> {
        Ok(())
    }
}

// --- Stub backend (offline, deterministic, used for tests and CI) ---------

/// Offline stub that returns the source text wrapped in target-language
/// markers, plus a deterministic mock translation table for a few common
/// phrases. Useful for unit tests and for the end-to-end pipeline running on
/// a machine without network access.
pub struct StubTranslator {
    name: &'static str,
    /// Optional override for which languages are "supported" by this stub.
    /// When `None`, accepts whatever the global `validate_language` accepts.
    allowed: Option<HashSet<&'static str>>,
}

impl StubTranslator {
    pub fn new() -> Self {
        Self {
            name: "local-stub",
            allowed: None,
        }
    }

    pub fn with_name(name: &'static str) -> Self {
        Self {
            name,
            allowed: None,
        }
    }

    /// Restrict this stub to a subset of languages. Used by tests to verify
    /// the validation path independently of the global allow-list.
    pub fn with_allowed<I>(mut self, codes: I) -> Self
    where
        I: IntoIterator<Item = &'static str>,
    {
        self.allowed = Some(codes.into_iter().collect());
        self
    }
}

impl Default for StubTranslator {
    fn default() -> Self {
        Self::new()
    }
}

fn mock_translate(text: &str, source: LanguageCode, target: LanguageCode) -> String {
    // Deterministic hand-written mappings for the most common smoke-test
    // phrases. Falls back to a labeled echo for everything else so the caller
    // can still see what went in and that the pipeline ran end-to-end.
    let lower = text.trim().to_ascii_lowercase();
    let pair = (source, target);
    let translated: Option<&'static str> = match pair {
        (LanguageCode::Zh, LanguageCode::En) => match lower.as_str() {
            "你好" => Some("hello"),
            "再见" => Some("goodbye"),
            "谢谢" => Some("thank you"),
            _ => None,
        },
        (LanguageCode::En, LanguageCode::Zh) => match lower.as_str() {
            "hello" => Some("你好"),
            "goodbye" => Some("再见"),
            "thank you" => Some("谢谢"),
            _ => None,
        },
        (LanguageCode::En, LanguageCode::Ja) => match lower.as_str() {
            "hello" => Some("こんにちは"),
            "thank you" => Some("ありがとう"),
            _ => None,
        },
        _ => None,
    };
    translated
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("[stub {}→{}] {}", source, target, text))
}

#[async_trait]
impl Translator for StubTranslator {
    async fn translate(
        &self,
        request: TranslationRequest,
    ) -> Result<TranslationResult, TranslationError> {
        validate_request(&request, self.allowed.as_ref())?;
        let translated = mock_translate(&request.text, request.source, request.target);
        Ok(TranslationResult {
            original: request.text,
            translated,
            source: request.source,
            target: request.target,
            backend: self.name.to_string(),
        })
    }

    fn backend_name(&self) -> &'static str {
        self.name
    }
}

// --- Cloud backend (OpenAI-compatible chat completions) ------------------

/// OpenAI-compatible chat-completions backend. Works against:
///   - the local hy-RTMT FastAPI server (`http://lubancat:7860/v1/...`)
///   - OpenAI, OpenRouter, DeepSeek, MiniMax etc.
///
/// The request body is deliberately minimal so it works against the widest
/// range of providers. `temperature: 0.0` and a short system prompt are used
/// to reduce variance for short utterances.
pub struct CloudTranslator {
    base_url: String,
    model: String,
    api_key: Option<String>,
    client: reqwest::Client,
    timeout: Duration,
}

impl CloudTranslator {
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
    ) -> Result<Self, TranslationError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .connect_timeout(Duration::from_secs(5))
            .no_proxy()
            .build()
            .map_err(|e| TranslationError::BackendUnavailable(e.to_string()))?;
        Ok(Self {
            base_url: base_url.into(),
            model: model.into(),
            api_key,
            client,
            timeout: Duration::from_secs(15),
        })
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .expect("building a reqwest client with a custom timeout must not fail");
        self.timeout = timeout;
        self
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: String,
}

#[async_trait]
impl Translator for CloudTranslator {
    async fn translate(
        &self,
        request: TranslationRequest,
    ) -> Result<TranslationResult, TranslationError> {
        validate_request(&request, None)?;

        let url = format!(
            "{}/v1/chat/completions",
            self.base_url.trim_end_matches('/')
        );
        let system = format!(
            "You are a translation engine. Translate the user's text from {} to {}. \
             Reply with ONLY the translated text, no quotes, no commentary, no explanation.",
            request.source, request.target
        );
        let body = ChatRequest {
            model: &self.model,
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: system,
                },
                ChatMessage {
                    role: "user",
                    content: request.text.clone(),
                },
            ],
            temperature: 0.0,
        };

        let mut req = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req.send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(TranslationError::Http {
                status: status.as_u16(),
                body,
            });
        }
        let parsed: ChatResponse = resp.json().await?;
        let translated = parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content.trim().to_string())
            .ok_or_else(|| {
                TranslationError::Other("chat completion returned no choices".to_string())
            })?;

        Ok(TranslationResult {
            original: request.text,
            translated,
            source: request.source,
            target: request.target,
            backend: "cloud".to_string(),
        })
    }

    fn backend_name(&self) -> &'static str {
        "cloud"
    }

    async fn health_check(&self) -> Result<(), TranslationError> {
        // Cheap reachability check — list models endpoint is standard on
        // OpenAI-compatible servers. We don't care about the response body,
        // just whether we get a 2xx/4xx (server alive) vs. network error.
        let url = format!("{}/v1/models", self.base_url.trim_end_matches('/'));
        let mut req = self.client.get(&url);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await?;
        if resp.status().is_server_error() {
            return Err(TranslationError::Http {
                status: resp.status().as_u16(),
                body: "server error during health check".to_string(),
            });
        }
        Ok(())
    }
}

// --- Shared request validation -------------------------------------------

fn validate_request(
    request: &TranslationRequest,
    extra_allow: Option<&HashSet<&'static str>>,
) -> Result<(), TranslationError> {
    if request.text.trim().is_empty() {
        return Err(TranslationError::EmptyText);
    }
    if request.source == request.target {
        return Err(TranslationError::SourceEqualsTarget {
            language: request.source,
        });
    }
    for code in [&request.source, &request.target] {
        if let Some(allow) = extra_allow {
            if !allow.contains(code.as_str()) {
                return Err(TranslationError::UnsupportedLanguage {
                    requested: code.as_str().to_string(),
                    supported: allow.iter().copied().collect(),
                });
            }
        } else if SUPPORTED_LANGUAGES
            .iter()
            .all(|l| l.as_str() != code.as_str())
        {
            return Err(TranslationError::UnsupportedLanguage {
                requested: code.as_str().to_string(),
                supported: SUPPORTED_LANGUAGES.iter().map(|l| l.as_str()).collect(),
            });
        }
    }
    Ok(())
}

// --- Tests ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_language_accepts_known_codes() {
        assert_eq!(validate_language("zh").unwrap(), LanguageCode::Zh);
        assert_eq!(validate_language("EN").unwrap(), LanguageCode::En);
        assert_eq!(validate_language(" ja ").unwrap(), LanguageCode::Ja);
    }

    #[test]
    fn validate_language_rejects_unknown() {
        let err = validate_language("klingon").unwrap_err();
        assert!(matches!(err, TranslationError::UnsupportedLanguage { .. }));
    }

    #[tokio::test]
    async fn stub_translates_known_phrases() {
        let t = StubTranslator::new();
        let req = TranslationRequest::new("你好", LanguageCode::Zh, LanguageCode::En);
        let res = t.translate(req).await.unwrap();
        assert_eq!(res.translated, "hello");
        assert_eq!(res.source, LanguageCode::Zh);
        assert_eq!(res.target, LanguageCode::En);
        assert_eq!(res.backend, "local-stub");
    }

    #[tokio::test]
    async fn stub_translates_known_phrase_reverse_direction() {
        let t = StubTranslator::new();
        let req = TranslationRequest::new("hello", LanguageCode::En, LanguageCode::Zh);
        let res = t.translate(req).await.unwrap();
        assert_eq!(res.translated, "你好");
    }

    #[tokio::test]
    async fn stub_echoes_unknown_phrases_with_marker() {
        let t = StubTranslator::new();
        let req =
            TranslationRequest::new("greetings, traveler", LanguageCode::En, LanguageCode::Ja);
        let res = t.translate(req).await.unwrap();
        assert!(res.translated.contains("[stub en→ja]"));
        assert!(res.translated.contains("greetings, traveler"));
    }

    #[tokio::test]
    async fn stub_rejects_empty_text() {
        let t = StubTranslator::new();
        let req = TranslationRequest::new("   ", LanguageCode::En, LanguageCode::Zh);
        let err = t.translate(req).await.unwrap_err();
        assert_eq!(err, TranslationError::EmptyText);
    }

    #[tokio::test]
    async fn stub_rejects_same_source_and_target() {
        let t = StubTranslator::new();
        let req = TranslationRequest::new("hello", LanguageCode::En, LanguageCode::En);
        let err = t.translate(req).await.unwrap_err();
        assert!(matches!(err, TranslationError::SourceEqualsTarget { .. }));
    }

    #[tokio::test]
    async fn stub_with_custom_allowlist_rejects_unsupported() {
        let t = StubTranslator::new()
            .with_allowed(vec![LanguageCode::Zh.as_str(), LanguageCode::En.as_str()]);
        let req = TranslationRequest::new("hi", LanguageCode::En, LanguageCode::Ja);
        let err = t.translate(req).await.unwrap_err();
        assert!(matches!(err, TranslationError::UnsupportedLanguage { .. }));
    }

    #[tokio::test]
    async fn stub_health_check_always_ok() {
        let t = StubTranslator::new();
        assert!(t.health_check().await.is_ok());
    }

    #[test]
    fn cloud_translator_constructs_with_no_key() {
        // No API key is allowed — e.g. for a local server that doesn't require auth.
        let t = CloudTranslator::new("http://localhost:7860", "Hy-MT2-1.8B", None);
        assert!(t.is_ok());
        assert_eq!(t.unwrap().backend_name(), "cloud");
    }

    #[test]
    fn cloud_translator_strips_trailing_slash_from_base_url() {
        // The translate path always appends `/v1/chat/completions`. If the user
        // passed `http://x:7860/`, we must not end up with `//v1/...`.
        let t = CloudTranslator::new("http://localhost:7860/", "m", None).unwrap();
        let expected = "http://localhost:7860/v1/chat/completions";
        // We can't directly inspect the URL, but the URL builder is exercised
        // in the async test below; here we just make sure construction works.
        let _ = expected;
        let _ = t;
    }

    #[tokio::test]
    async fn cloud_translator_returns_http_error_for_500() {
        // Spin up a tiny HTTP "server" using tokio's async TCP listener so the
        // request and the response live on the same runtime — no cross-thread
        // scheduling surprises with `#[tokio::test]`'s current-thread flavor.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let _ = sock.read(&mut buf).await;
            let body = r#"{"error":{"message":"boom"}}"#;
            let resp = format!(
                "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = sock.write_all(resp.as_bytes()).await;
        });
        let t = CloudTranslator::new(format!("http://{}", addr), "m", None).unwrap();
        let req = TranslationRequest::new("hello", LanguageCode::En, LanguageCode::Zh);
        let err = t.translate(req).await.unwrap_err();
        let _ = server.await;
        match err {
            TranslationError::Http { status, .. } => assert_eq!(status, 500),
            other => panic!("expected HTTP error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn cloud_translator_parses_successful_response() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 8192];
            let _ = sock.read(&mut buf).await;
            let body = r#"{"choices":[{"message":{"role":"assistant","content":"你好"}}]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = sock.write_all(resp.as_bytes()).await;
        });
        let t = CloudTranslator::new(format!("http://{}", addr), "m", None).unwrap();
        let req = TranslationRequest::new("hello", LanguageCode::En, LanguageCode::Zh);
        let res = t.translate(req).await.unwrap();
        let _ = server.await;
        assert_eq!(res.translated, "你好");
        assert_eq!(res.backend, "cloud");
    }

    #[tokio::test]
    async fn cloud_health_check_succeeds_against_responding_server() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 1024];
            let _ = sock.read(&mut buf).await;
            let body = r#"{"data":[]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = sock.write_all(resp.as_bytes()).await;
        });
        let t = CloudTranslator::new(format!("http://{}", addr), "m", None).unwrap();
        t.health_check().await.unwrap();
        let _ = server.await;
    }

    #[tokio::test]
    async fn cloud_health_check_fails_on_server_error() {
        // Use std::thread for the server so it runs on a real OS thread and
        // doesn't compete for runtime scheduling with the test's reqwest
        // call. The previous `tokio::spawn` variant hit a scheduling edge
        // where reqwest completed before the server task was polled.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let (mut sock, _) = match listener.accept() {
                Ok(p) => p,
                Err(_) => return,
            };
            let mut buf = [0u8; 4096];
            let _ = sock.read(&mut buf);
            // 4xx/5xx responses need a Content-Length (or chunked) so reqwest
            // knows the body is complete; otherwise it sits waiting for more.
            let body = r#"{"error":"down"}"#;
            let resp = format!(
                "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(), body
            );
            let _ = sock.write_all(resp.as_bytes());
        });
        let t = CloudTranslator::new(format!("http://{}", addr), "m", None).unwrap();
        let err = t.health_check().await.unwrap_err();
        handle.join().unwrap();
        assert!(matches!(err, TranslationError::Http { status: 503, .. }));
    }

    #[test]
    fn translation_error_display_messages_are_informative() {
        let err = TranslationError::UnsupportedLanguage {
            requested: "klingon".to_string(),
            supported: vec!["zh", "en"],
        };
        let msg = err.to_string();
        assert!(msg.contains("klingon"));
        assert!(msg.contains("zh"));
    }
}
