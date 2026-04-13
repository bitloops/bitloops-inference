mod ollama_chat;
mod openai_chat_completions;

use std::collections::BTreeMap;
use std::time::Duration;

use bitloops_inference_protocol::{
    ErrorResponse, ProviderCapabilities, ProviderKind, ProviderMetadata, ResponseMode, TokenUsage,
};
use serde_json::{Map, Value, json};

use crate::config::ProfileConfig;
use crate::json::extract_json_object;

pub use ollama_chat::OllamaChatProvider;
pub use openai_chat_completions::OpenAiChatCompletionsProvider;

#[derive(Clone, Debug, PartialEq)]
pub struct InferenceRequest {
    pub system_prompt: String,
    pub user_prompt: String,
    pub response_mode: ResponseMode,
    pub temperature: f32,
    pub max_output_tokens: u32,
    pub metadata: Option<Map<String, Value>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InferenceResponse {
    pub text: String,
    pub parsed_json: Option<Value>,
    pub usage: Option<TokenUsage>,
    pub finish_reason: Option<String>,
    pub provider_name: String,
    pub model_name: String,
}

pub trait InferenceProvider {
    fn metadata(&self) -> ProviderMetadata;
    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse, ProviderError>;
}

type ProviderFactory = fn(&ProfileConfig) -> Result<Box<dyn InferenceProvider>, ProviderError>;

pub struct ProviderRegistry {
    factories: BTreeMap<ProviderKind, ProviderFactory>,
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        let mut factories: BTreeMap<ProviderKind, ProviderFactory> = BTreeMap::new();
        factories.insert(ProviderKind::OpenAiChatCompletions, |profile| {
            Ok(Box::new(OpenAiChatCompletionsProvider::new(
                profile.clone(),
            )))
        });
        factories.insert(ProviderKind::OllamaChat, |profile| {
            Ok(Box::new(OllamaChatProvider::new(profile.clone())))
        });
        Self { factories }
    }
}

impl ProviderRegistry {
    pub fn create(
        &self,
        profile: &ProfileConfig,
    ) -> Result<Box<dyn InferenceProvider>, ProviderError> {
        let factory = self.factories.get(&profile.kind).ok_or_else(|| {
            ProviderError::invalid_config(format!(
                "unsupported provider kind '{}'",
                profile.kind.as_str()
            ))
        })?;

        factory(profile)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProviderError {
    pub code: String,
    pub message: String,
    pub details: Option<Value>,
}

impl ProviderError {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            code: "invalid_request".to_owned(),
            message: message.into(),
            details: None,
        }
    }

    pub fn invalid_config(message: impl Into<String>) -> Self {
        Self {
            code: "invalid_config".to_owned(),
            message: message.into(),
            details: None,
        }
    }

    pub fn invalid_provider_response(message: impl Into<String>, details: Option<Value>) -> Self {
        Self {
            code: "invalid_provider_response".to_owned(),
            message: message.into(),
            details,
        }
    }

    pub fn provider_http_error(status: u16, details: Option<Value>) -> Self {
        Self {
            code: "provider_http_error".to_owned(),
            message: format!("provider returned HTTP {status}"),
            details,
        }
    }

    pub fn provider_transport_error(message: impl Into<String>) -> Self {
        let message = message.into();
        let lower = message.to_ascii_lowercase();
        let code = if lower.contains("timeout") || lower.contains("timed out") {
            "provider_timeout"
        } else {
            "provider_transport_error"
        };

        Self {
            code: code.to_owned(),
            message,
            details: None,
        }
    }

    pub fn into_error_response(self) -> ErrorResponse {
        ErrorResponse {
            code: self.code,
            message: self.message,
            details: self.details,
        }
    }
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for ProviderError {}

pub(crate) fn default_capabilities() -> ProviderCapabilities {
    ProviderCapabilities {
        response_modes: vec![ResponseMode::Text, ResponseMode::JsonObject],
        usage_reporting: true,
    }
}

pub(crate) fn parse_json_response_mode(
    text: &str,
    response_mode: ResponseMode,
) -> Result<Option<Value>, ProviderError> {
    match response_mode {
        ResponseMode::Text => Ok(None),
        ResponseMode::JsonObject => extract_json_object(text).map(Some).ok_or_else(|| {
            ProviderError::invalid_provider_response(
                "provider response did not contain a valid JSON object",
                Some(json!({ "text": text })),
            )
        }),
    }
}

pub(crate) fn post_json(
    url: &str,
    timeout_secs: u64,
    headers: &[(&str, String)],
    payload: &Value,
) -> Result<Value, ProviderError> {
    let mut request = ureq::post(url)
        .timeout(Duration::from_secs(timeout_secs))
        .set("Content-Type", "application/json");

    for (name, value) in headers {
        request = request.set(name, value.as_str());
    }

    match request.send_json(payload.clone()) {
        Ok(response) => response.into_json::<Value>().map_err(|error| {
            ProviderError::invalid_provider_response(
                format!("failed to parse provider response body as JSON: {error}"),
                None,
            )
        }),
        Err(ureq::Error::Status(status, response)) => {
            let body = response.into_string().unwrap_or_default();
            let details = serde_json::from_str::<Value>(&body)
                .ok()
                .map(|json_body| json!({ "status": status, "body": json_body }))
                .or_else(|| {
                    if body.is_empty() {
                        None
                    } else {
                        Some(json!({ "status": status, "body": body }))
                    }
                });

            Err(ProviderError::provider_http_error(status, details))
        }
        Err(ureq::Error::Transport(error)) => {
            Err(ProviderError::provider_transport_error(error.to_string()))
        }
    }
}
