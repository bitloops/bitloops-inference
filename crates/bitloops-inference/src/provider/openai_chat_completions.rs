use bitloops_inference_protocol::{ProviderMetadata, ResponseMode, TokenUsage};
use serde_json::{Map, Value, json};

use crate::config::ProfileConfig;
use crate::provider::{
    InferenceProvider, InferenceRequest, InferenceResponse, ProviderError, default_capabilities,
    parse_json_response_mode, post_json,
};

pub struct OpenAiChatCompletionsProvider {
    profile: ProfileConfig,
}

const BITLOOPS_PLATFORM_PROVIDER_NAME: &str = "bitloops";
const REFRESH_CACHE_HEADER: &str = "X-Bitloops-Refresh-Cache";
const REFRESH_CACHE_METADATA_FIELD: &str = "bitloops_refresh_cache";

impl OpenAiChatCompletionsProvider {
    pub fn new(profile: ProfileConfig) -> Self {
        Self { profile }
    }

    fn build_payload(&self, request: &InferenceRequest) -> Value {
        let mut payload = json!({
            "model": self.profile.model,
            "messages": [
                {
                    "role": "system",
                    "content": request.system_prompt,
                },
                {
                    "role": "user",
                    "content": request.user_prompt,
                }
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
            "stream": false,
        });

        if request.response_mode == ResponseMode::JsonObject {
            payload["response_format"] = json!({ "type": "json_object" });
        }

        payload
    }

    fn build_headers(&self, refresh_cache: bool) -> Vec<(&'static str, String)> {
        let mut headers = Vec::new();
        if let Some(api_key) = &self.profile.api_key {
            headers.push(("Authorization", format!("Bearer {api_key}")));
        }
        if refresh_cache {
            headers.push((REFRESH_CACHE_HEADER, "true".to_owned()));
        }
        headers
    }

    fn infer_once(
        &self,
        request: &InferenceRequest,
        refresh_cache: bool,
    ) -> Result<InferenceResponse, ProviderError> {
        let payload = self.build_payload(request);
        let headers = self.build_headers(refresh_cache);

        let body = post_json(
            &self.profile.base_url,
            self.profile.timeout_secs,
            &headers,
            &payload,
        )?;

        self.parse_response(body, request.response_mode)
    }

    fn should_retry_with_cache_refresh(
        &self,
        request: &InferenceRequest,
        error: &ProviderError,
        already_refreshed: bool,
    ) -> bool {
        !already_refreshed
            && self.profile.provider_name == BITLOOPS_PLATFORM_PROVIDER_NAME
            && request.response_mode == ResponseMode::JsonObject
            && error.code == "invalid_provider_response"
    }

    fn parse_response(
        &self,
        body: Value,
        response_mode: ResponseMode,
    ) -> Result<InferenceResponse, ProviderError> {
        let choice = body
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .ok_or_else(|| {
                ProviderError::invalid_provider_response(
                    "provider response did not contain choices[0]",
                    Some(body.clone()),
                )
            })?;

        let text = extract_message_content(choice).ok_or_else(|| {
            ProviderError::invalid_provider_response(
                "provider response did not contain assistant message content",
                Some(choice.clone()),
            )
        })?;

        let parsed_json = parse_json_response_mode(&text, response_mode)?;
        let model_name = body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(self.profile.model.as_str())
            .to_owned();
        let usage = body.get("usage").and_then(parse_usage);
        let finish_reason = choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);

        Ok(InferenceResponse {
            text,
            parsed_json,
            usage,
            finish_reason,
            provider_name: self.profile.provider_name.clone(),
            model_name,
        })
    }
}

impl InferenceProvider for OpenAiChatCompletionsProvider {
    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata {
            kind: self.profile.kind,
            provider_name: self.profile.provider_name.clone(),
            model_name: self.profile.model.clone(),
            endpoint: self.profile.base_url.clone(),
            capabilities: default_capabilities(),
        }
    }

    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse, ProviderError> {
        let refresh_cache = metadata_requests_refresh_cache(request.metadata.as_ref());
        match self.infer_once(request, refresh_cache) {
            Ok(response) => Ok(response),
            Err(error) if self.should_retry_with_cache_refresh(request, &error, refresh_cache) => {
                self.infer_once(request, true)
            }
            Err(error) => Err(error),
        }
    }
}

fn metadata_requests_refresh_cache(metadata: Option<&Map<String, Value>>) -> bool {
    metadata
        .and_then(|metadata| metadata.get(REFRESH_CACHE_METADATA_FIELD))
        .is_some_and(|value| match value {
            Value::Bool(value) => *value,
            Value::String(value) => value.eq_ignore_ascii_case("true"),
            _ => false,
        })
}

fn extract_message_content(choice: &Value) -> Option<String> {
    let content = choice.get("message")?.get("content")?;
    if let Some(text) = content.as_str() {
        return Some(text.to_owned());
    }

    let parts = content.as_array()?;
    let collected = parts
        .iter()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>();
    if collected.is_empty() {
        None
    } else {
        Some(collected.join(""))
    }
}

fn parse_usage(usage: &Value) -> Option<TokenUsage> {
    let prompt_tokens = usage.get("prompt_tokens")?.as_u64()? as u32;
    let completion_tokens = usage.get("completion_tokens")?.as_u64()? as u32;
    let total_tokens = usage.get("total_tokens")?.as_u64()? as u32;

    Some(TokenUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
    })
}

#[cfg(test)]
mod tests {
    use bitloops_inference_protocol::ProviderKind;

    use super::*;
    use crate::provider::InferenceRequest;

    fn profile() -> ProfileConfig {
        ProfileConfig {
            task: crate::config::ProfileTask::TextGeneration,
            kind: ProviderKind::OpenAiChatCompletions,
            provider_name: "openai".to_owned(),
            model: "gpt-4.1-mini".to_owned(),
            base_url: "https://example.com/v1/chat/completions".to_owned(),
            api_key: Some("secret".to_owned()),
            temperature: Some(0.1),
            timeout_secs: 60,
            max_output_tokens: Some(200),
            runtime_command: None,
            runtime_args: Vec::new(),
            startup_timeout_secs: 60,
        }
    }

    fn bitloops_profile() -> ProfileConfig {
        ProfileConfig {
            provider_name: "bitloops".to_owned(),
            base_url: "https://platform.example.com/v1/chat/completions".to_owned(),
            ..profile()
        }
    }

    fn request(response_mode: ResponseMode) -> InferenceRequest {
        InferenceRequest {
            system_prompt: "You summarise diffs.".to_owned(),
            user_prompt: "Summarise this patch.".to_owned(),
            response_mode,
            temperature: 0.1,
            max_output_tokens: 200,
            metadata: None,
        }
    }

    fn request_with_metadata(
        response_mode: ResponseMode,
        metadata: Map<String, Value>,
    ) -> InferenceRequest {
        InferenceRequest {
            metadata: Some(metadata),
            ..request(response_mode)
        }
    }

    #[test]
    fn builds_json_mode_payload() {
        let provider = OpenAiChatCompletionsProvider::new(profile());
        let payload = provider.build_payload(&request(ResponseMode::JsonObject));

        assert_eq!(payload["model"], "gpt-4.1-mini");
        assert_eq!(payload["messages"][0]["role"], "system");
        assert_eq!(payload["messages"][1]["role"], "user");
        assert_eq!(payload["stream"], false);
        assert_eq!(payload["response_format"]["type"], "json_object");
    }

    #[test]
    fn builds_text_mode_payload_without_response_format() {
        let provider = OpenAiChatCompletionsProvider::new(profile());
        let payload = provider.build_payload(&request(ResponseMode::Text));

        assert_eq!(payload["model"], "gpt-4.1-mini");
        assert!(payload.get("response_format").is_none());
    }

    #[test]
    fn metadata_refresh_flag_adds_refresh_cache_header() {
        let provider = OpenAiChatCompletionsProvider::new(profile());
        let request = request_with_metadata(
            ResponseMode::Text,
            Map::from_iter([(REFRESH_CACHE_METADATA_FIELD.to_owned(), json!(true))]),
        );

        let headers =
            provider.build_headers(metadata_requests_refresh_cache(request.metadata.as_ref()));

        assert!(
            headers
                .iter()
                .any(|(name, value)| *name == REFRESH_CACHE_HEADER && value == "true")
        );
    }

    #[test]
    fn openai_profile_does_not_refresh_cache_by_default() {
        let provider = OpenAiChatCompletionsProvider::new(profile());

        let headers = provider.build_headers(metadata_requests_refresh_cache(
            request(ResponseMode::JsonObject).metadata.as_ref(),
        ));

        assert!(
            headers
                .iter()
                .all(|(name, _)| *name != REFRESH_CACHE_HEADER)
        );
    }

    #[test]
    fn bitloops_json_parse_failure_retries_once_with_refresh_cache() {
        let provider = OpenAiChatCompletionsProvider::new(bitloops_profile());
        let request = request(ResponseMode::JsonObject);
        let error = ProviderError::invalid_provider_response(
            "provider response did not contain JSON",
            None,
        );

        assert!(provider.should_retry_with_cache_refresh(&request, &error, false));
        assert!(!provider.should_retry_with_cache_refresh(&request, &error, true));
    }

    #[test]
    fn openai_json_parse_failure_does_not_retry_with_refresh_cache_by_default() {
        let provider = OpenAiChatCompletionsProvider::new(profile());
        let request = request(ResponseMode::JsonObject);
        let error = ProviderError::invalid_provider_response(
            "provider response did not contain JSON",
            None,
        );

        assert!(!provider.should_retry_with_cache_refresh(&request, &error, false));
    }

    #[test]
    fn parses_successful_response() {
        let provider = OpenAiChatCompletionsProvider::new(profile());
        let response = provider
            .parse_response(
                json!({
                    "model": "gpt-4.1-mini",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "{\"summary\":\"Adds isolation\",\"confidence\":0.91}"
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 7,
                        "total_tokens": 19
                    }
                }),
                ResponseMode::JsonObject,
            )
            .expect("response should parse");

        assert_eq!(response.provider_name, "openai");
        assert_eq!(response.model_name, "gpt-4.1-mini");
        assert_eq!(response.finish_reason.as_deref(), Some("stop"));
        assert_eq!(
            response.parsed_json.as_ref().expect("json")["confidence"],
            0.91
        );
        assert_eq!(response.usage.expect("usage").total_tokens, 19);
    }

    #[test]
    fn extracts_json_from_surrounding_prose() {
        let provider = OpenAiChatCompletionsProvider::new(profile());
        let fixture = include_str!("../../tests/fixtures/openai_surrounding_prose_response.json");
        let body: Value = serde_json::from_str(fixture).expect("fixture should parse");

        let response = provider
            .parse_response(body, ResponseMode::JsonObject)
            .expect("response should parse");

        assert_eq!(
            response.parsed_json.expect("json")["summary"],
            "Keeps the provider boundary isolated"
        );
    }

    #[test]
    fn rejects_malformed_json_when_json_mode_is_requested() {
        let provider = OpenAiChatCompletionsProvider::new(profile());
        let fixture = include_str!("../../tests/fixtures/openai_malformed_json_response.json");
        let body: Value = serde_json::from_str(fixture).expect("fixture should parse");

        let error = provider
            .parse_response(body, ResponseMode::JsonObject)
            .expect_err("response should fail");

        assert_eq!(error.code, "invalid_provider_response");
    }
}
