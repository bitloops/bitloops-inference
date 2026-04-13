use std::io::{self, BufRead};
use std::path::Path;

use bitloops_inference_protocol::{
    DescribeRequest, DescribeResponse, ErrorResponse, InferRequest, PROTOCOL_VERSION,
    RequestEnvelope, RequestPayload, ResponseEnvelope, ResponsePayload, ShutdownResponse,
};
use serde_json::Value;

use crate::AppError;
use crate::config::{InferenceConfig, ProfileConfig};
use crate::provider::{InferenceProvider, InferenceRequest, ProviderError, ProviderRegistry};

const RUNTIME_NAME: &str = "bitloops-inference";

pub struct Runtime {
    profile_name: String,
    profile: ProfileConfig,
    provider: Box<dyn InferenceProvider>,
}

impl Runtime {
    pub fn new(
        profile_name: String,
        profile: &ProfileConfig,
        registry: ProviderRegistry,
    ) -> Result<Self, ProviderError> {
        let provider = registry.create(profile)?;
        Ok(Self {
            profile_name,
            profile: profile.clone(),
            provider,
        })
    }

    pub fn describe(&self) -> DescribeResponse {
        DescribeResponse {
            protocol_version: PROTOCOL_VERSION,
            runtime_name: RUNTIME_NAME.to_owned(),
            runtime_version: env!("CARGO_PKG_VERSION").to_owned(),
            profile_name: self.profile_name.clone(),
            provider: self.provider.metadata(),
        }
    }

    fn handle_request(&self, request: RequestEnvelope) -> (ResponseEnvelope, bool) {
        let request_id = request.request_id;

        match request.payload {
            RequestPayload::Describe(DescribeRequest {}) => (
                ResponseEnvelope {
                    request_id,
                    payload: ResponsePayload::Describe(self.describe()),
                },
                false,
            ),
            RequestPayload::Infer(infer_request) => {
                let response = match self.infer(infer_request) {
                    Ok(infer_response) => {
                        ResponsePayload::Infer(bitloops_inference_protocol::InferResponse {
                            text: infer_response.text,
                            parsed_json: infer_response.parsed_json,
                            usage: infer_response.usage,
                            finish_reason: infer_response.finish_reason,
                            provider_name: infer_response.provider_name,
                            model_name: infer_response.model_name,
                        })
                    }
                    Err(error) => ResponsePayload::Error(error.into_error_response()),
                };

                (
                    ResponseEnvelope {
                        request_id,
                        payload: response,
                    },
                    false,
                )
            }
            RequestPayload::Shutdown(_) => (
                ResponseEnvelope {
                    request_id,
                    payload: ResponsePayload::Shutdown(ShutdownResponse::default()),
                },
                true,
            ),
        }
    }

    fn infer(
        &self,
        request: InferRequest,
    ) -> Result<crate::provider::InferenceResponse, ProviderError> {
        let canonical_request = self.resolve_request(request)?;
        self.provider.infer(&canonical_request)
    }

    fn resolve_request(&self, request: InferRequest) -> Result<InferenceRequest, ProviderError> {
        let temperature = request
            .temperature
            .or(self.profile.temperature)
            .ok_or_else(|| {
                ProviderError::invalid_request(
                    "temperature was missing from the request and no profile default was configured",
                )
            })?;
        let max_output_tokens = request
            .max_output_tokens
            .or(self.profile.max_output_tokens)
            .ok_or_else(|| {
                ProviderError::invalid_request(
                    "max_output_tokens was missing from the request and no profile default was configured",
                )
            })?;

        Ok(InferenceRequest {
            system_prompt: request.system_prompt,
            user_prompt: request.user_prompt,
            response_mode: request.response_mode,
            temperature,
            max_output_tokens,
            metadata: request.metadata,
        })
    }
}

pub fn run_stdio(config_path: &Path, profile_name: &str) -> Result<(), AppError> {
    let config = InferenceConfig::load(config_path)?;
    let profile = config.profile(profile_name)?;
    let runtime = Runtime::new(
        profile_name.to_owned(),
        profile,
        ProviderRegistry::default(),
    )?;
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = stdin.lock();
    let mut writer = stdout.lock();
    let mut line = String::new();

    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }

        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }

        let (response, should_shutdown) = match decode_request(line) {
            Ok(request) => runtime.handle_request(request),
            Err(error_response) => (
                ResponseEnvelope {
                    request_id: error_response.0,
                    payload: ResponsePayload::Error(error_response.1),
                },
                false,
            ),
        };

        response.write_json_line(&mut writer)?;
        if should_shutdown {
            break;
        }
    }

    Ok(())
}

fn decode_request(line: &str) -> Result<RequestEnvelope, (String, ErrorResponse)> {
    RequestEnvelope::from_json_line(line).map_err(|error| {
        let request_id = extract_request_id(line).unwrap_or_else(|| "unknown".to_owned());
        (
            request_id,
            ErrorResponse {
                code: "invalid_request".to_owned(),
                message: error.to_string(),
                details: None,
            },
        )
    })
}

fn extract_request_id(line: &str) -> Option<String> {
    let value: Value = serde_json::from_str(line).ok()?;
    value
        .get("request_id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

#[cfg(test)]
mod tests {
    use bitloops_inference_protocol::{ProviderKind, RequestPayload, ResponseMode};

    use super::*;

    fn profile() -> ProfileConfig {
        ProfileConfig {
            kind: ProviderKind::OllamaChat,
            provider_name: "ollama".to_owned(),
            model: "qwen2.5-coder:14b".to_owned(),
            base_url: "http://127.0.0.1:11434/api/chat".to_owned(),
            api_key: None,
            temperature: Some(0.1),
            timeout_secs: 120,
            max_output_tokens: Some(200),
        }
    }

    #[test]
    fn extracts_request_id_for_invalid_requests() {
        let request =
            decode_request(r#"{"request_id":"123","type":"infer","response_mode":"text"}"#)
                .expect_err("request should be invalid");

        assert_eq!(request.0, "123");
        assert_eq!(request.1.code, "invalid_request");
    }

    #[test]
    fn uses_profile_defaults_for_missing_inference_fields() {
        let runtime = Runtime::new(
            "ollama_local".to_owned(),
            &profile(),
            ProviderRegistry::default(),
        )
        .expect("runtime should build");
        let request = InferRequest {
            system_prompt: "system".to_owned(),
            user_prompt: "user".to_owned(),
            response_mode: ResponseMode::Text,
            temperature: None,
            max_output_tokens: None,
            metadata: None,
        };

        let resolved = runtime
            .resolve_request(request)
            .expect("request should resolve");

        assert_eq!(resolved.temperature, 0.1);
        assert_eq!(resolved.max_output_tokens, 200);
    }

    #[test]
    fn describe_request_builds_describe_response() {
        let runtime = Runtime::new(
            "ollama_local".to_owned(),
            &profile(),
            ProviderRegistry::default(),
        )
        .expect("runtime should build");
        let (response, should_shutdown) = runtime.handle_request(RequestEnvelope {
            request_id: "describe-1".to_owned(),
            payload: RequestPayload::Describe(DescribeRequest {}),
        });

        assert!(!should_shutdown);
        match response.payload {
            ResponsePayload::Describe(describe) => {
                assert_eq!(describe.protocol_version, PROTOCOL_VERSION);
                assert_eq!(describe.profile_name, "ollama_local");
            }
            other => panic!("expected describe response, got {other:?}"),
        }
    }
}
