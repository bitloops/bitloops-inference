use bitloops_inference_protocol::{ProviderMetadata, ResponseMode, TokenUsage};
use serde_json::{Value, json};

use crate::config::ProfileConfig;
use crate::provider::{
    InferenceProvider, InferenceRequest, InferenceResponse, ProviderError, default_capabilities,
    parse_json_response_mode, post_json,
};

pub struct OllamaChatProvider {
    profile: ProfileConfig,
}

impl OllamaChatProvider {
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
            "stream": false,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
            }
        });

        if request.response_mode == ResponseMode::JsonObject {
            payload["format"] = json!("json");
        }

        payload
    }

    fn parse_response(
        &self,
        body: Value,
        response_mode: ResponseMode,
    ) -> Result<InferenceResponse, ProviderError> {
        let text = body
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ProviderError::invalid_provider_response(
                    "provider response did not contain message.content",
                    Some(body.clone()),
                )
            })?
            .to_owned();

        let parsed_json = parse_json_response_mode(&text, response_mode)?;
        let model_name = body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(self.profile.model.as_str())
            .to_owned();
        let usage = parse_usage(&body);
        let finish_reason = body
            .get("done_reason")
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

impl InferenceProvider for OllamaChatProvider {
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
        let payload = self.build_payload(request);
        let body = post_json(
            &self.profile.base_url,
            self.profile.timeout_secs,
            &[],
            &payload,
        )?;

        self.parse_response(body, request.response_mode)
    }
}

fn parse_usage(body: &Value) -> Option<TokenUsage> {
    let prompt_tokens = body.get("prompt_eval_count")?.as_u64()? as u32;
    let completion_tokens = body.get("eval_count")?.as_u64()? as u32;

    Some(TokenUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    })
}

#[cfg(test)]
mod tests {
    use bitloops_inference_protocol::ProviderKind;

    use super::*;
    use crate::provider::InferenceRequest;

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

    #[test]
    fn builds_json_mode_payload() {
        let provider = OllamaChatProvider::new(profile());
        let payload = provider.build_payload(&request(ResponseMode::JsonObject));

        assert_eq!(payload["model"], "qwen2.5-coder:14b");
        assert_eq!(payload["format"], "json");
        assert_eq!(payload["options"]["num_predict"], 200);
    }

    #[test]
    fn parses_successful_response() {
        let provider = OllamaChatProvider::new(profile());
        let response = provider
            .parse_response(
                json!({
                    "model": "qwen2.5-coder:14b",
                    "message": {
                        "role": "assistant",
                        "content": "{\"summary\":\"Uses Ollama\",\"confidence\":0.88}"
                    },
                    "done_reason": "stop",
                    "prompt_eval_count": 10,
                    "eval_count": 5
                }),
                ResponseMode::JsonObject,
            )
            .expect("response should parse");

        assert_eq!(response.model_name, "qwen2.5-coder:14b");
        assert_eq!(
            response.parsed_json.expect("json")["summary"],
            "Uses Ollama"
        );
        assert_eq!(response.usage.expect("usage").total_tokens, 15);
    }
}
