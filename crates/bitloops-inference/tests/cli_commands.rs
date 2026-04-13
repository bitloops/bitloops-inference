mod common;

use assert_cmd::Command;
use bitloops_inference_protocol::{ProviderKind, ResponseEnvelope, ResponsePayload};
use serde_json::Value;

use common::write_config;

#[test]
fn validate_config_reports_profile_names() {
    let config = write_config(
        r#"
            [inference.profiles.openai_fast]
            kind = "openai_chat_completions"
            provider_name = "openai"
            model = "gpt-4.1-mini"
            base_url = "https://example.com/v1/chat/completions"
            api_key = "secret"
            temperature = 0.1
            timeout_secs = 60
            max_output_tokens = 200

            [inference.profiles.ollama_local]
            kind = "ollama_chat"
            provider_name = "ollama"
            model = "qwen2.5-coder:14b"
            base_url = "http://127.0.0.1:11434/api/chat"
            temperature = 0.1
            timeout_secs = 120
            max_output_tokens = 200
        "#,
    );

    let output = Command::cargo_bin("bitloops-inference")
        .expect("binary should exist")
        .arg("validate-config")
        .arg("--config")
        .arg(config.path())
        .output()
        .expect("command should run");

    assert!(output.status.success());
    let stdout: Value = serde_json::from_slice(&output.stdout).expect("stdout should be JSON");
    assert_eq!(stdout["status"], "ok");
    assert_eq!(stdout["profiles"][0], "ollama_local");
    assert_eq!(stdout["profiles"][1], "openai_fast");
}

#[test]
fn describe_profile_returns_protocol_shaped_json() {
    let config = write_config(
        r#"
            [inference.profiles.openai_fast]
            kind = "openai_chat_completions"
            provider_name = "openai"
            model = "gpt-4.1-mini"
            base_url = "https://example.com/v1/chat/completions"
            api_key = "secret"
            temperature = 0.1
            timeout_secs = 60
            max_output_tokens = 200
        "#,
    );

    let output = Command::cargo_bin("bitloops-inference")
        .expect("binary should exist")
        .arg("describe-profile")
        .arg("--config")
        .arg(config.path())
        .arg("--profile")
        .arg("openai_fast")
        .output()
        .expect("command should run");

    assert!(output.status.success());
    let response = ResponseEnvelope::from_json_line(
        String::from_utf8(output.stdout)
            .expect("stdout should be utf-8")
            .trim_end(),
    )
    .expect("response should parse");

    match response.payload {
        ResponsePayload::Describe(describe) => {
            assert_eq!(describe.profile_name, "openai_fast");
            assert_eq!(describe.provider.kind, ProviderKind::OpenAiChatCompletions);
            assert_eq!(describe.provider.provider_name, "openai");
        }
        other => panic!("expected describe response, got {other:?}"),
    }
}
