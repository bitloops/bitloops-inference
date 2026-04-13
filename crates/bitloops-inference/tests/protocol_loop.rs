mod common;

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::thread;
use std::time::Duration;

use assert_cmd::cargo::CommandCargoExt;
use bitloops_inference_protocol::{
    ErrorResponse, InferRequest, RequestEnvelope, RequestPayload, ResponseEnvelope, ResponseMode,
    ResponsePayload, ShutdownRequest,
};
use mockito::Server;

use common::write_config;

#[test]
fn openai_runtime_handles_describe_infer_and_shutdown() {
    let mut server = Server::new();
    let fixture = include_str!("fixtures/openai_surrounding_prose_response.json");
    let mock = server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(fixture)
        .create();

    let config = write_config(&format!(
        r#"
            [inference.profiles.openai_fast]
            kind = "openai_chat_completions"
            provider_name = "openai"
            model = "gpt-4.1-mini"
            base_url = "{}/v1/chat/completions"
            api_key = "secret"
            temperature = 0.1
            timeout_secs = 60
            max_output_tokens = 200
        "#,
        server.url()
    ));

    let mut runtime = RuntimeHarness::spawn(config.path(), "openai_fast");
    runtime.send(&RequestEnvelope {
        request_id: "describe-1".to_owned(),
        payload: RequestPayload::Describe(bitloops_inference_protocol::DescribeRequest {}),
    });
    let describe = runtime.read();
    match describe.payload {
        ResponsePayload::Describe(describe) => {
            assert_eq!(describe.profile_name, "openai_fast");
            assert_eq!(describe.provider.provider_name, "openai");
        }
        other => panic!("expected describe response, got {other:?}"),
    }

    runtime.send(&RequestEnvelope {
        request_id: "infer-1".to_owned(),
        payload: RequestPayload::Infer(InferRequest {
            system_prompt: "You summarise semantic diffs.".to_owned(),
            user_prompt: "Summarise this change.".to_owned(),
            response_mode: ResponseMode::JsonObject,
            temperature: Some(0.1),
            max_output_tokens: Some(200),
            metadata: None,
        }),
    });
    let infer = runtime.read();
    match infer.payload {
        ResponsePayload::Infer(infer) => {
            assert_eq!(infer.provider_name, "openai");
            assert_eq!(infer.model_name, "gpt-4.1-mini");
            assert_eq!(
                infer.parsed_json.expect("json")["summary"],
                "Keeps the provider boundary isolated"
            );
        }
        other => panic!("expected infer response, got {other:?}"),
    }

    runtime.send(&RequestEnvelope {
        request_id: "shutdown-1".to_owned(),
        payload: RequestPayload::Shutdown(ShutdownRequest {}),
    });
    let shutdown = runtime.read();
    assert!(matches!(shutdown.payload, ResponsePayload::Shutdown(_)));

    runtime.finish();
    mock.assert();
}

#[test]
fn ollama_runtime_handles_infer_and_shutdown() {
    let mut server = Server::new();
    let fixture = include_str!("fixtures/ollama_success_response.json");
    let mock = server
        .mock("POST", "/api/chat")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(fixture)
        .create();

    let config = write_config(&format!(
        r#"
            [inference.profiles.ollama_local]
            kind = "ollama_chat"
            provider_name = "ollama"
            model = "qwen2.5-coder:14b"
            base_url = "{}/api/chat"
            temperature = 0.1
            timeout_secs = 60
            max_output_tokens = 200
        "#,
        server.url()
    ));

    let mut runtime = RuntimeHarness::spawn(config.path(), "ollama_local");
    runtime.send(&RequestEnvelope {
        request_id: "infer-1".to_owned(),
        payload: RequestPayload::Infer(InferRequest {
            system_prompt: "You summarise semantic diffs.".to_owned(),
            user_prompt: "Summarise this change.".to_owned(),
            response_mode: ResponseMode::JsonObject,
            temperature: Some(0.1),
            max_output_tokens: Some(200),
            metadata: None,
        }),
    });
    let infer = runtime.read();
    match infer.payload {
        ResponsePayload::Infer(infer) => {
            assert_eq!(infer.provider_name, "ollama");
            assert_eq!(
                infer.parsed_json.expect("json")["summary"],
                "Uses the local Ollama service"
            );
        }
        other => panic!("expected infer response, got {other:?}"),
    }

    runtime.send(&RequestEnvelope {
        request_id: "shutdown-1".to_owned(),
        payload: RequestPayload::Shutdown(ShutdownRequest {}),
    });
    let shutdown = runtime.read();
    assert!(matches!(shutdown.payload, ResponsePayload::Shutdown(_)));

    runtime.finish();
    mock.assert();
}

#[test]
fn http_errors_are_normalised() {
    let mut server = Server::new();
    let fixture = include_str!("fixtures/openai_http_error.json");
    let mock = server
        .mock("POST", "/v1/chat/completions")
        .with_status(429)
        .with_header("content-type", "application/json")
        .with_body(fixture)
        .create();

    let config = write_config(&format!(
        r#"
            [inference.profiles.openai_fast]
            kind = "openai_chat_completions"
            provider_name = "openai"
            model = "gpt-4.1-mini"
            base_url = "{}/v1/chat/completions"
            api_key = "secret"
            temperature = 0.1
            timeout_secs = 60
            max_output_tokens = 200
        "#,
        server.url()
    ));

    let mut runtime = RuntimeHarness::spawn(config.path(), "openai_fast");
    runtime.send(&RequestEnvelope {
        request_id: "infer-http-error".to_owned(),
        payload: RequestPayload::Infer(InferRequest {
            system_prompt: "You summarise semantic diffs.".to_owned(),
            user_prompt: "Summarise this change.".to_owned(),
            response_mode: ResponseMode::Text,
            temperature: Some(0.1),
            max_output_tokens: Some(200),
            metadata: None,
        }),
    });
    let response = runtime.read();
    assert_error(response, "provider_http_error");

    runtime.send(&RequestEnvelope {
        request_id: "shutdown-1".to_owned(),
        payload: RequestPayload::Shutdown(ShutdownRequest {}),
    });
    let shutdown = runtime.read();
    assert!(matches!(shutdown.payload, ResponsePayload::Shutdown(_)));

    runtime.finish();
    mock.assert();
}

#[test]
fn malformed_json_object_is_reported() {
    let mut server = Server::new();
    let fixture = include_str!("fixtures/openai_malformed_json_response.json");
    let mock = server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(fixture)
        .create();

    let config = write_config(&format!(
        r#"
            [inference.profiles.openai_fast]
            kind = "openai_chat_completions"
            provider_name = "openai"
            model = "gpt-4.1-mini"
            base_url = "{}/v1/chat/completions"
            api_key = "secret"
            temperature = 0.1
            timeout_secs = 60
            max_output_tokens = 200
        "#,
        server.url()
    ));

    let mut runtime = RuntimeHarness::spawn(config.path(), "openai_fast");
    runtime.send(&RequestEnvelope {
        request_id: "infer-bad-json".to_owned(),
        payload: RequestPayload::Infer(InferRequest {
            system_prompt: "You summarise semantic diffs.".to_owned(),
            user_prompt: "Summarise this change.".to_owned(),
            response_mode: ResponseMode::JsonObject,
            temperature: Some(0.1),
            max_output_tokens: Some(200),
            metadata: None,
        }),
    });
    let response = runtime.read();
    assert_error(response, "invalid_provider_response");

    runtime.send(&RequestEnvelope {
        request_id: "shutdown-1".to_owned(),
        payload: RequestPayload::Shutdown(ShutdownRequest {}),
    });
    let shutdown = runtime.read();
    assert!(matches!(shutdown.payload, ResponsePayload::Shutdown(_)));

    runtime.finish();
    mock.assert();
}

#[test]
fn timeouts_are_normalised() {
    let (url, handle) = start_slow_http_server(Duration::from_secs(3));
    let config = write_config(&format!(
        r#"
            [inference.profiles.openai_fast]
            kind = "openai_chat_completions"
            provider_name = "openai"
            model = "gpt-4.1-mini"
            base_url = "{url}"
            api_key = "secret"
            temperature = 0.1
            timeout_secs = 1
            max_output_tokens = 200
        "#
    ));

    let mut runtime = RuntimeHarness::spawn(config.path(), "openai_fast");
    runtime.send(&RequestEnvelope {
        request_id: "infer-timeout".to_owned(),
        payload: RequestPayload::Infer(InferRequest {
            system_prompt: "You summarise semantic diffs.".to_owned(),
            user_prompt: "Summarise this change.".to_owned(),
            response_mode: ResponseMode::Text,
            temperature: Some(0.1),
            max_output_tokens: Some(200),
            metadata: None,
        }),
    });
    let response = runtime.read();
    assert_error(response, "provider_timeout");

    runtime.send(&RequestEnvelope {
        request_id: "shutdown-1".to_owned(),
        payload: RequestPayload::Shutdown(ShutdownRequest {}),
    });
    let shutdown = runtime.read();
    assert!(matches!(shutdown.payload, ResponsePayload::Shutdown(_)));

    runtime.finish();
    handle.join().expect("slow server thread should finish");
}

fn assert_error(response: ResponseEnvelope, expected_code: &str) {
    match response.payload {
        ResponsePayload::Error(ErrorResponse { code, .. }) => assert_eq!(code, expected_code),
        other => panic!("expected error response, got {other:?}"),
    }
}

fn start_slow_http_server(delay: Duration) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener should bind");
    let address = listener.local_addr().expect("address should exist");
    let handle = thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buffer = [0u8; 4096];
            let _ = stream.read(&mut buffer);
            thread::sleep(delay);

            let body = r#"{"model":"gpt-4.1-mini","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(response.as_bytes());
        }
    });

    (format!("http://{address}/v1/chat/completions"), handle)
}

struct RuntimeHarness {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl RuntimeHarness {
    fn spawn(config_path: &Path, profile_name: &str) -> Self {
        let mut command =
            Command::cargo_bin("bitloops-inference").expect("binary should be discoverable");
        let mut child = command
            .arg("run")
            .arg("--config")
            .arg(config_path)
            .arg("--profile")
            .arg(profile_name)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("runtime should spawn");

        let stdin = child.stdin.take().expect("stdin should be piped");
        let stdout = BufReader::new(child.stdout.take().expect("stdout should be piped"));

        Self {
            child,
            stdin,
            stdout,
        }
    }

    fn send(&mut self, request: &RequestEnvelope) {
        request
            .write_json_line(&mut self.stdin)
            .expect("request should be written");
    }

    fn read(&mut self) -> ResponseEnvelope {
        let mut line = String::new();
        self.stdout
            .read_line(&mut line)
            .expect("response should be readable");
        ResponseEnvelope::from_json_line(line.trim_end()).expect("response should parse")
    }

    fn finish(mut self) {
        drop(self.stdin);
        let status = self.child.wait().expect("runtime should exit");
        assert!(status.success(), "runtime exited with status {status}");
    }
}
