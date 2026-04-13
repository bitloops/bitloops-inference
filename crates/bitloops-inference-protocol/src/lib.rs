use std::io::{self, Write};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

pub const PROTOCOL_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    #[serde(rename = "openai_chat_completions")]
    OpenAiChatCompletions,
    OllamaChat,
}

impl ProviderKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenAiChatCompletions => "openai_chat_completions",
            Self::OllamaChat => "ollama_chat",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseMode {
    Text,
    JsonObject,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RequestEnvelope {
    pub request_id: String,
    #[serde(flatten)]
    pub payload: RequestPayload,
}

impl RequestEnvelope {
    pub fn from_json_line(line: &str) -> Result<Self, ProtocolCodecError> {
        serde_json::from_str(line).map_err(ProtocolCodecError::Deserialize)
    }

    pub fn to_json_line(&self) -> Result<String, ProtocolCodecError> {
        serde_json::to_string(self).map_err(ProtocolCodecError::Serialize)
    }

    pub fn write_json_line(&self, writer: &mut impl Write) -> Result<(), ProtocolCodecError> {
        write_json_line(writer, self)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RequestPayload {
    Describe(DescribeRequest),
    Infer(InferRequest),
    Shutdown(ShutdownRequest),
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct DescribeRequest {}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct InferRequest {
    pub system_prompt: String,
    pub user_prompt: String,
    pub response_mode: ResponseMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Map<String, Value>>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ShutdownRequest {}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResponseEnvelope {
    pub request_id: String,
    #[serde(flatten)]
    pub payload: ResponsePayload,
}

impl ResponseEnvelope {
    pub fn from_json_line(line: &str) -> Result<Self, ProtocolCodecError> {
        serde_json::from_str(line).map_err(ProtocolCodecError::Deserialize)
    }

    pub fn to_json_line(&self) -> Result<String, ProtocolCodecError> {
        serde_json::to_string(self).map_err(ProtocolCodecError::Serialize)
    }

    pub fn write_json_line(&self, writer: &mut impl Write) -> Result<(), ProtocolCodecError> {
        write_json_line(writer, self)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsePayload {
    Describe(DescribeResponse),
    Infer(InferResponse),
    Shutdown(ShutdownResponse),
    Error(ErrorResponse),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DescribeResponse {
    pub protocol_version: u32,
    pub runtime_name: String,
    pub runtime_version: String,
    pub profile_name: String,
    pub provider: ProviderMetadata,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ProviderMetadata {
    pub kind: ProviderKind,
    pub provider_name: String,
    pub model_name: String,
    pub endpoint: String,
    pub capabilities: ProviderCapabilities,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ProviderCapabilities {
    pub response_modes: Vec<ResponseMode>,
    pub usage_reporting: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct InferResponse {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parsed_json: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    pub provider_name: String,
    pub model_name: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ShutdownResponse {}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ErrorResponse {
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

pub fn write_json_line(
    writer: &mut impl Write,
    value: &impl Serialize,
) -> Result<(), ProtocolCodecError> {
    serde_json::to_writer(&mut *writer, value).map_err(ProtocolCodecError::Serialize)?;
    writer.write_all(b"\n").map_err(ProtocolCodecError::Io)?;
    writer.flush().map_err(ProtocolCodecError::Io)
}

#[derive(Debug, Error)]
pub enum ProtocolCodecError {
    #[error("failed to serialise JSON: {0}")]
    Serialize(serde_json::Error),
    #[error("failed to deserialise JSON: {0}")]
    Deserialize(serde_json::Error),
    #[error("failed to write JSON line: {0}")]
    Io(io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_round_trips_as_json_line() {
        let request = RequestEnvelope {
            request_id: "request-1".to_owned(),
            payload: RequestPayload::Infer(InferRequest {
                system_prompt: "system".to_owned(),
                user_prompt: "user".to_owned(),
                response_mode: ResponseMode::JsonObject,
                temperature: Some(0.1),
                max_output_tokens: Some(200),
                metadata: None,
            }),
        };

        let line = request.to_json_line().expect("request should serialise");
        let decoded = RequestEnvelope::from_json_line(&line).expect("request should round-trip");

        assert_eq!(decoded, request);
    }

    #[test]
    fn response_writes_a_trailing_newline() {
        let response = ResponseEnvelope {
            request_id: "request-2".to_owned(),
            payload: ResponsePayload::Shutdown(ShutdownResponse::default()),
        };
        let mut output = Vec::new();

        response
            .write_json_line(&mut output)
            .expect("response should write");

        assert!(
            String::from_utf8(output)
                .expect("output should be utf-8")
                .ends_with('\n')
        );
    }
}
