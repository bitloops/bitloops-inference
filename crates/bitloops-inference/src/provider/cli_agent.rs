use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use bitloops_inference_protocol::{ProviderCapabilities, ProviderMetadata, ResponseMode};
use serde_json::{Value, json};

use crate::config::ProfileConfig;
use crate::json::extract_json_object;
use crate::provider::{InferenceProvider, InferenceRequest, InferenceResponse, ProviderError};

pub struct CodexExecProvider {
    profile: ProfileConfig,
}

impl CodexExecProvider {
    pub fn new(profile: ProfileConfig) -> Self {
        Self { profile }
    }

    fn build_command(
        &self,
        request: &InferenceRequest,
        schema_path: &Path,
        result_path: &Path,
    ) -> Result<CliCommand, ProviderError> {
        let command = self.runtime_command()?;
        let mut args = self.profile.runtime_args.clone();
        args.push("exec".to_string());
        // Codex CLI currently exposes reasoning effort as a config override. Other
        // Bitloops profile fields such as temperature and max_output_tokens are not
        // emitted here because Codex does not document stable config keys for them.
        if let Some(thinking_level) = self.profile.thinking_level {
            let Some(effort) = thinking_level.codex_reasoning_effort() else {
                return Err(ProviderError::invalid_config(
                    "codex_exec does not support thinking_level 'max'",
                ));
            };
            args.extend([
                "-c".to_string(),
                format!("model_reasoning_effort=\"{effort}\""),
            ]);
        }
        args.extend([
            "--model".to_string(),
            self.profile.model.clone(),
            "--sandbox".to_string(),
            "read-only".to_string(),
            "--ephemeral".to_string(),
            "--output-schema".to_string(),
            schema_path.display().to_string(),
            "--output-last-message".to_string(),
            result_path.display().to_string(),
            prompt_for_cli_agent(request),
        ]);
        Ok(CliCommand {
            command,
            args,
            cwd: workspace_path(request),
            stdin: None,
        })
    }
}

impl InferenceProvider for CodexExecProvider {
    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata {
            kind: self.profile.kind,
            provider_name: self.profile.provider_name.clone(),
            model_name: self.profile.model.clone(),
            endpoint: self.profile.runtime_command.clone().unwrap_or_default(),
            capabilities: ProviderCapabilities {
                response_modes: vec![ResponseMode::JsonObject],
                usage_reporting: false,
                structured_output: vec!["json_object".to_string(), "json_schema".to_string()],
            },
        }
    }

    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse, ProviderError> {
        let schema = request
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("json_schema"))
            .ok_or_else(|| {
                ProviderError::invalid_request("codex_exec requires metadata.json_schema")
            })?;
        let temp_dir = tempfile::tempdir().map_err(|err| {
            ProviderError::provider_transport_error(format!(
                "failed to create temporary directory for codex_exec: {err}"
            ))
        })?;
        let schema_path = temp_dir.path().join("schema.json");
        let result_path = temp_dir.path().join("result.json");
        write_json_file(&schema_path, schema)?;

        let command = self.build_command(request, &schema_path, &result_path)?;
        let output = run_cli_command(command, self.profile.timeout_secs)?;
        let text = if result_path.is_file() {
            fs::read_to_string(&result_path).map_err(|err| {
                ProviderError::invalid_provider_response(
                    format!("failed to read codex_exec result file: {err}"),
                    Some(json!({ "path": result_path.display().to_string() })),
                )
            })?
        } else if output.stdout_truncated {
            return Err(truncated_stdout_error("codex_exec", output.stdout));
        } else {
            output.stdout.clone()
        };
        let parsed_json = parse_cli_json(&text)?;

        Ok(InferenceResponse {
            text,
            parsed_json: Some(parsed_json),
            usage: None,
            finish_reason: Some("stop".to_string()),
            provider_name: self.profile.provider_name.clone(),
            model_name: self.profile.model.clone(),
        })
    }
}

pub struct ClaudeCodePrintProvider {
    profile: ProfileConfig,
}

impl ClaudeCodePrintProvider {
    pub fn new(profile: ProfileConfig) -> Self {
        Self { profile }
    }

    fn build_command(&self, request: &InferenceRequest) -> Result<CliCommand, ProviderError> {
        let command = self.runtime_command()?;
        let mut args = self.profile.runtime_args.clone();
        args.push("-p".to_string());
        if let Some(thinking_level) = self.profile.thinking_level {
            args.extend([
                "--effort".to_string(),
                thinking_level.claude_effort().to_string(),
            ]);
        }
        args.extend([
            "--model".to_string(),
            self.profile.model.clone(),
            "--output-format".to_string(),
            "json".to_string(),
            "--input-format".to_string(),
            "text".to_string(),
        ]);
        if let Some(schema) = json_schema_argument(request)? {
            args.extend(["--json-schema".to_string(), schema]);
        }
        args.extend(["--allowedTools".to_string(), "Read,Grep,Glob".to_string()]);
        Ok(CliCommand {
            command,
            args,
            cwd: workspace_path(request),
            stdin: Some(prompt_for_cli_agent(request)),
        })
    }
}

impl InferenceProvider for ClaudeCodePrintProvider {
    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata {
            kind: self.profile.kind,
            provider_name: self.profile.provider_name.clone(),
            model_name: self.profile.model.clone(),
            endpoint: self.profile.runtime_command.clone().unwrap_or_default(),
            capabilities: ProviderCapabilities {
                response_modes: vec![ResponseMode::JsonObject],
                usage_reporting: false,
                structured_output: vec!["json_object".to_string(), "json_schema".to_string()],
            },
        }
    }

    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse, ProviderError> {
        let command = self.build_command(request)?;
        let output = run_cli_command(command, self.profile.timeout_secs)?;
        if output.stdout_truncated {
            return Err(truncated_stdout_error("claude_code_print", output.stdout));
        }
        let parsed_json = parse_claude_json(&output.stdout)?;

        Ok(InferenceResponse {
            text: output.stdout,
            parsed_json: Some(parsed_json),
            usage: None,
            finish_reason: Some("stop".to_string()),
            provider_name: self.profile.provider_name.clone(),
            model_name: self.profile.model.clone(),
        })
    }
}

trait RuntimeCommand {
    fn runtime_command(&self) -> Result<String, ProviderError>;
}

impl RuntimeCommand for CodexExecProvider {
    fn runtime_command(&self) -> Result<String, ProviderError> {
        self.profile.runtime_command.clone().ok_or_else(|| {
            ProviderError::invalid_config("codex_exec runtime command is not configured")
        })
    }
}

impl RuntimeCommand for ClaudeCodePrintProvider {
    fn runtime_command(&self) -> Result<String, ProviderError> {
        self.profile.runtime_command.clone().ok_or_else(|| {
            ProviderError::invalid_config("claude_code_print runtime command is not configured")
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliCommand {
    command: String,
    args: Vec<String>,
    cwd: Option<PathBuf>,
    stdin: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliOutput {
    stdout: String,
    stdout_truncated: bool,
}

const MAX_CLI_CAPTURE_BYTES: usize = 4 * 1024 * 1024;
const OUTPUT_COLLECT_TIMEOUT: Duration = Duration::from_secs(1);

#[derive(Debug, Clone, PartialEq, Eq)]
struct CapturedStream {
    text: String,
    truncated: bool,
}

fn prompt_for_cli_agent(request: &InferenceRequest) -> String {
    format!(
        "{}\n\n{}",
        request.system_prompt.trim(),
        request.user_prompt.trim()
    )
}

fn workspace_path(request: &InferenceRequest) -> Option<PathBuf> {
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("workspace_path"))
        .and_then(Value::as_str)
        .map(PathBuf::from)
}

fn json_schema_argument(request: &InferenceRequest) -> Result<Option<String>, ProviderError> {
    let Some(schema) = request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("json_schema"))
    else {
        return Ok(None);
    };

    serde_json::to_string(schema).map(Some).map_err(|err| {
        ProviderError::invalid_request(format!("failed to serialise JSON schema: {err}"))
    })
}

fn write_json_file(path: &Path, value: &Value) -> Result<(), ProviderError> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|err| {
        ProviderError::invalid_request(format!("failed to serialise JSON schema: {err}"))
    })?;
    fs::write(path, bytes).map_err(|err| {
        ProviderError::provider_transport_error(format!(
            "failed to write temporary JSON file {}: {err}",
            path.display()
        ))
    })
}

fn spawn_stream_reader<R>(
    command_name: &str,
    stream_name: &'static str,
    mut stream: R,
) -> mpsc::Receiver<Result<CapturedStream, String>>
where
    R: Read + Send + 'static,
{
    let (sender, receiver) = mpsc::channel();
    let command_name = command_name.to_string();

    thread::spawn(move || {
        let mut bytes = Vec::new();
        let mut truncated = false;
        let mut buffer = [0_u8; 8192];

        loop {
            match stream.read(&mut buffer) {
                Ok(0) => break,
                Ok(count) => {
                    let remaining = MAX_CLI_CAPTURE_BYTES.saturating_sub(bytes.len());
                    if remaining > 0 {
                        let keep = remaining.min(count);
                        bytes.extend_from_slice(&buffer[..keep]);
                    }
                    if count > remaining {
                        truncated = true;
                    }
                }
                Err(err) => {
                    let _ = sender.send(Err(format!(
                        "failed to read {stream_name} from CLI agent `{command_name}`: {err}"
                    )));
                    return;
                }
            }
        }

        let _ = sender.send(Ok(CapturedStream {
            text: String::from_utf8_lossy(&bytes).to_string(),
            truncated,
        }));
    });

    receiver
}

fn collect_stream_output(
    receiver: mpsc::Receiver<Result<CapturedStream, String>>,
    command_name: &str,
    stream_name: &str,
) -> Result<CapturedStream, ProviderError> {
    receiver
        .recv_timeout(OUTPUT_COLLECT_TIMEOUT)
        .map_err(|err| {
            ProviderError::provider_transport_error(format!(
                "failed to collect {stream_name} from CLI agent `{command_name}`: {err}"
            ))
        })?
        .map_err(ProviderError::provider_transport_error)
}

fn run_cli_command(command: CliCommand, timeout_secs: u64) -> Result<CliOutput, ProviderError> {
    let mut process = Command::new(&command.command);
    process.args(&command.args);
    if let Some(cwd) = &command.cwd {
        process.current_dir(cwd);
    }
    if command.stdin.is_some() {
        process.stdin(Stdio::piped());
    } else {
        process.stdin(Stdio::null());
    }
    process.stdout(Stdio::piped());
    process.stderr(Stdio::piped());

    let mut child = process.spawn().map_err(|err| {
        ProviderError::provider_transport_error(format!(
            "failed to spawn CLI agent `{}`: {err}",
            command.command
        ))
    })?;

    let child_stdout = child.stdout.take().ok_or_else(|| {
        ProviderError::provider_transport_error(format!(
            "failed to open stdout for CLI agent `{}`",
            command.command
        ))
    })?;
    let child_stderr = child.stderr.take().ok_or_else(|| {
        ProviderError::provider_transport_error(format!(
            "failed to open stderr for CLI agent `{}`",
            command.command
        ))
    })?;

    let stdout_receiver = spawn_stream_reader(&command.command, "stdout", child_stdout);
    let stderr_receiver = spawn_stream_reader(&command.command, "stderr", child_stderr);

    let stdin_result_receiver = if let Some(stdin) = command.stdin {
        let mut child_stdin = child.stdin.take().ok_or_else(|| {
            ProviderError::provider_transport_error(format!(
                "failed to open stdin for CLI agent `{}`",
                command.command
            ))
        })?;
        let command_name = command.command.clone();
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let result = child_stdin.write_all(stdin.as_bytes()).map_err(|err| {
                format!("failed to write stdin to CLI agent `{command_name}`: {err}")
            });
            let _ = sender.send(result);
        });
        Some(receiver)
    } else {
        None
    };

    let mut stdin_result: Option<Result<(), String>> = None;
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    let status = loop {
        if stdin_result.is_none()
            && let Some(receiver) = &stdin_result_receiver
        {
            match receiver.try_recv() {
                Ok(result) => stdin_result = Some(result),
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    stdin_result = Some(Err(format!(
                        "failed to write stdin to CLI agent `{}`: stdin writer stopped",
                        command.command
                    )));
                }
            }
        }

        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(ProviderError::provider_transport_error(format!(
                    "CLI agent `{}` timed out after {timeout_secs}s",
                    command.command
                )));
            }
            Ok(None) => thread::sleep(Duration::from_millis(25)),
            Err(err) => {
                let _ = child.kill();
                return Err(ProviderError::provider_transport_error(format!(
                    "failed to poll CLI agent `{}`: {err}",
                    command.command
                )));
            }
        }
    };

    if stdin_result.is_none()
        && let Some(receiver) = &stdin_result_receiver
    {
        stdin_result = receiver.try_recv().ok();
    }

    let stdout = collect_stream_output(stdout_receiver, &command.command, "stdout")?;
    let stderr = collect_stream_output(stderr_receiver, &command.command, "stderr")?;

    if !status.success() {
        return Err(ProviderError::invalid_provider_response(
            format!(
                "CLI agent `{}` exited with status {}",
                command.command, status
            ),
            Some(json!({
                "status": status.to_string(),
                "stdout": stdout.text,
                "stderr": stderr.text,
                "stdout_truncated": stdout.truncated,
                "stderr_truncated": stderr.truncated,
            })),
        ));
    }
    if let Some(Err(message)) = stdin_result {
        return Err(ProviderError::provider_transport_error(message));
    }

    Ok(CliOutput {
        stdout: stdout.text,
        stdout_truncated: stdout.truncated,
    })
}

fn parse_cli_json(text: &str) -> Result<Value, ProviderError> {
    extract_json_object(text).ok_or_else(|| {
        ProviderError::invalid_provider_response(
            "CLI agent response did not contain a valid JSON object",
            Some(json!({ "text": text })),
        )
    })
}

fn truncated_stdout_error(provider_name: &str, stdout: String) -> ProviderError {
    ProviderError::invalid_provider_response(
        format!("{provider_name} stdout exceeded the CLI capture limit"),
        Some(json!({
            "stdout": stdout,
            "stdout_truncated": true,
            "max_capture_bytes": MAX_CLI_CAPTURE_BYTES,
        })),
    )
}

fn parse_claude_json(text: &str) -> Result<Value, ProviderError> {
    let outer = parse_cli_json(text)?;
    if let Some(structured_output) = outer.get("structured_output") {
        if structured_output.is_object() {
            return Ok(structured_output.clone());
        }

        return Err(ProviderError::invalid_provider_response(
            "Claude CLI structured_output field was not a JSON object",
            Some(json!({
                "field": "structured_output",
                "value": structured_output.clone(),
                "outer": outer.clone(),
            })),
        ));
    }

    if let Some(result) = outer.get("result") {
        if let Some(result_text) = result.as_str() {
            return parse_cli_json(result_text);
        }
        if result.is_object() {
            return Ok(result.clone());
        }
    }
    Ok(outer)
}

#[cfg(test)]
mod tests {
    use bitloops_inference_protocol::ProviderKind;

    use super::*;

    fn profile(kind: ProviderKind, command: &str, args: &[&str]) -> ProfileConfig {
        ProfileConfig {
            task: crate::config::ProfileTask::StructuredGeneration,
            kind,
            provider_name: kind.as_str().to_string(),
            model: "model".to_string(),
            base_url: String::new(),
            api_key: None,
            temperature: Some(0.1),
            timeout_secs: 30,
            max_output_tokens: Some(4096),
            thinking_level: None,
            runtime_command: Some(command.to_string()),
            runtime_args: args.iter().map(|arg| arg.to_string()).collect(),
            startup_timeout_secs: 5,
        }
    }

    fn request() -> InferenceRequest {
        InferenceRequest {
            system_prompt: "system".to_string(),
            user_prompt: "user".to_string(),
            response_mode: ResponseMode::JsonObject,
            temperature: 0.1,
            max_output_tokens: 4096,
            metadata: Some(serde_json::Map::from_iter([
                ("json_schema".to_string(), json!({ "type": "object" })),
                ("workspace_path".to_string(), json!("/tmp/repo")),
            ])),
        }
    }

    #[test]
    fn codex_exec_builds_expected_command() {
        let provider = CodexExecProvider::new(profile(ProviderKind::CodexExec, "codex", &[]));
        let command = provider
            .build_command(
                &request(),
                Path::new("/tmp/schema.json"),
                Path::new("/tmp/result.json"),
            )
            .expect("command");

        assert_eq!(command.command, "codex");
        assert_eq!(command.args[0], "exec");
        assert!(command.args.contains(&"--model".to_string()));
        assert!(command.args.contains(&"model".to_string()));
        assert!(command.args.contains(&"--sandbox".to_string()));
        assert!(command.args.contains(&"read-only".to_string()));
        assert!(command.args.contains(&"--ephemeral".to_string()));
        assert!(command.args.contains(&"--output-schema".to_string()));
        assert!(command.args.contains(&"/tmp/schema.json".to_string()));
        assert!(command.args.contains(&"--output-last-message".to_string()));
        assert!(command.args.contains(&"/tmp/result.json".to_string()));
        assert_eq!(command.cwd.as_deref(), Some(Path::new("/tmp/repo")));
        assert_eq!(command.stdin, None);
    }

    #[test]
    fn codex_exec_passes_configured_thinking_level() {
        let mut profile = profile(ProviderKind::CodexExec, "codex", &[]);
        profile.thinking_level = Some(crate::config::ThinkingLevel::ExtraHigh);
        let provider = CodexExecProvider::new(profile);
        let command = provider
            .build_command(
                &request(),
                Path::new("/tmp/schema.json"),
                Path::new("/tmp/result.json"),
            )
            .expect("command");

        assert_eq!(command.command, "codex");
        assert_eq!(
            command.args,
            vec![
                "exec".to_string(),
                "-c".to_string(),
                "model_reasoning_effort=\"xhigh\"".to_string(),
                "--model".to_string(),
                "model".to_string(),
                "--sandbox".to_string(),
                "read-only".to_string(),
                "--ephemeral".to_string(),
                "--output-schema".to_string(),
                "/tmp/schema.json".to_string(),
                "--output-last-message".to_string(),
                "/tmp/result.json".to_string(),
                "system\n\nuser".to_string(),
            ]
        );
    }

    #[test]
    fn codex_exec_preserves_runtime_args_before_exec_subcommand() {
        let provider = CodexExecProvider::new(profile(
            ProviderKind::CodexExec,
            "codex",
            &["-c", "model_verbosity=\"low\""],
        ));
        let command = provider
            .build_command(
                &request(),
                Path::new("/tmp/schema.json"),
                Path::new("/tmp/result.json"),
            )
            .expect("command");

        assert_eq!(
            &command.args[..4],
            &[
                "-c".to_string(),
                "model_verbosity=\"low\"".to_string(),
                "exec".to_string(),
                "--model".to_string(),
            ]
        );
    }

    #[test]
    fn codex_exec_does_not_emit_unsupported_temperature_or_token_args() {
        let provider = CodexExecProvider::new(profile(ProviderKind::CodexExec, "codex", &[]));
        let command = provider
            .build_command(
                &request(),
                Path::new("/tmp/schema.json"),
                Path::new("/tmp/result.json"),
            )
            .expect("command");

        let joined = command.args.join(" ");
        assert!(!joined.contains("temperature"));
        assert!(!joined.contains("max_output_tokens"));
        assert!(!joined.contains("model_max_output_tokens"));
    }

    #[test]
    fn claude_code_print_builds_stdin_command_with_model_schema_and_tools() {
        let provider =
            ClaudeCodePrintProvider::new(profile(ProviderKind::ClaudeCodePrint, "claude", &[]));
        let command = provider.build_command(&request()).expect("command");
        let schema = serde_json::to_string(&json!({ "type": "object" })).expect("schema");

        assert_eq!(command.command, "claude");
        assert_eq!(
            command.args,
            vec![
                "-p".to_string(),
                "--model".to_string(),
                "model".to_string(),
                "--output-format".to_string(),
                "json".to_string(),
                "--input-format".to_string(),
                "text".to_string(),
                "--json-schema".to_string(),
                schema,
                "--allowedTools".to_string(),
                "Read,Grep,Glob".to_string(),
            ]
        );
        assert_eq!(command.stdin.as_deref(), Some("system\n\nuser"));
        assert_eq!(command.cwd.as_deref(), Some(Path::new("/tmp/repo")));
    }

    #[test]
    fn claude_code_print_passes_configured_effort() {
        let mut profile = profile(ProviderKind::ClaudeCodePrint, "claude", &[]);
        profile.thinking_level = Some(crate::config::ThinkingLevel::Max);
        let provider = ClaudeCodePrintProvider::new(profile);
        let command = provider.build_command(&request()).expect("command");
        let schema = serde_json::to_string(&json!({ "type": "object" })).expect("schema");

        assert_eq!(command.command, "claude");
        assert_eq!(
            command.args,
            vec![
                "-p".to_string(),
                "--effort".to_string(),
                "max".to_string(),
                "--model".to_string(),
                "model".to_string(),
                "--output-format".to_string(),
                "json".to_string(),
                "--input-format".to_string(),
                "text".to_string(),
                "--json-schema".to_string(),
                schema,
                "--allowedTools".to_string(),
                "Read,Grep,Glob".to_string(),
            ]
        );
        assert_eq!(command.stdin.as_deref(), Some("system\n\nuser"));
    }

    #[test]
    fn json_extraction_rejects_malformed_output() {
        let error = parse_cli_json("not json").expect_err("malformed output should fail");

        assert_eq!(error.code, "invalid_provider_response");
    }

    #[test]
    fn truncated_stdout_error_includes_capture_limit() {
        let error = truncated_stdout_error("claude_code_print", "partial".to_string());

        assert_eq!(error.code, "invalid_provider_response");
        assert!(error.message.contains("claude_code_print"));
        let details = error.details.expect("details");
        assert_eq!(details["stdout"], "partial");
        assert_eq!(details["stdout_truncated"], true);
        assert_eq!(details["max_capture_bytes"], MAX_CLI_CAPTURE_BYTES);
    }

    #[test]
    fn driver_failure_includes_status_and_stderr() {
        let error = run_cli_command(
            CliCommand {
                command: "/bin/sh".to_string(),
                args: vec![
                    "-c".to_string(),
                    "echo stdout text; echo stderr text >&2; exit 7".to_string(),
                ],
                cwd: None,
                stdin: None,
            },
            5,
        )
        .expect_err("failing command should return provider error");

        assert_eq!(error.code, "invalid_provider_response");
        let details = error.details.expect("details");
        assert!(details["status"].as_str().unwrap().contains('7'));
        assert_eq!(details["stdout"], "stdout text\n");
        assert_eq!(details["stderr"], "stderr text\n");
    }

    #[test]
    fn driver_failure_with_large_stderr_returns_bounded_diagnostics() {
        let error = run_cli_command(
            CliCommand {
                command: "/bin/sh".to_string(),
                args: vec![
                    "-c".to_string(),
                    "echo stdout text; dd if=/dev/zero bs=1024 count=5120 2>/dev/null | tr '\\000' e >&2; exit 7".to_string(),
                ],
                cwd: None,
                stdin: None,
            },
            5,
        )
        .expect_err("failing command should return provider error");

        assert_eq!(error.code, "invalid_provider_response");
        let details = error.details.expect("details");
        assert!(details["status"].as_str().unwrap().contains('7'));
        assert_eq!(details["stdout"], "stdout text\n");
        assert_eq!(details["stdout_truncated"], false);
        assert_eq!(details["stderr_truncated"], true);
        assert_eq!(
            details["stderr"].as_str().expect("stderr").len(),
            MAX_CLI_CAPTURE_BYTES
        );
    }

    #[test]
    fn run_cli_command_writes_configured_stdin() {
        let output = run_cli_command(
            CliCommand {
                command: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), "cat".to_string()],
                cwd: None,
                stdin: Some("system\n\nuser".to_string()),
            },
            5,
        )
        .expect("command should receive stdin");

        assert_eq!(output.stdout, "system\n\nuser");
    }

    #[test]
    fn run_cli_command_drains_large_stderr_before_waiting_for_exit() {
        let output = run_cli_command(
            CliCommand {
                command: "/bin/sh".to_string(),
                args: vec![
                    "-c".to_string(),
                    "printf '{\"summary\":\"ok\"}'; dd if=/dev/zero bs=1024 count=1024 2>/dev/null | tr '\\000' x >&2".to_string(),
                ],
                cwd: None,
                stdin: None,
            },
            2,
        )
        .expect("large stderr should not block child exit");

        assert_eq!(output.stdout, "{\"summary\":\"ok\"}");
    }

    #[test]
    fn run_cli_command_drains_large_stdout_before_waiting_for_exit() {
        let output = run_cli_command(
            CliCommand {
                command: "/bin/sh".to_string(),
                args: vec![
                    "-c".to_string(),
                    "printf '{\"summary\":\"ok\"}\\n'; dd if=/dev/zero bs=1024 count=1024 2>/dev/null | tr '\\000' x".to_string(),
                ],
                cwd: None,
                stdin: None,
            },
            2,
        )
        .expect("large stdout should not block child exit");

        assert!(output.stdout.starts_with("{\"summary\":\"ok\"}\n"));
        assert!(output.stdout.len() >= 1024 * 1024);
    }

    #[test]
    fn run_cli_command_timeout_includes_blocked_stdin_write() {
        let (sender, receiver) = std::sync::mpsc::channel();
        let stdin = "x".repeat(1024 * 1024);
        thread::spawn(move || {
            let result = run_cli_command(
                CliCommand {
                    command: "/bin/sh".to_string(),
                    args: vec!["-c".to_string(), "sleep 5".to_string()],
                    cwd: None,
                    stdin: Some(stdin),
                },
                1,
            )
            .map(|_| String::new())
            .map_err(|error| format!("{}:{}", error.code, error.message));
            let _ = sender.send(result);
        });

        let result = receiver
            .recv_timeout(Duration::from_secs(3))
            .expect("stdin write should not bypass timeout");
        let error = result.expect_err("command should time out");
        assert!(error.starts_with("provider_timeout:"));
        assert!(error.contains("timed out after 1s"));
    }

    #[test]
    fn run_cli_command_timeout_does_not_wait_for_stdin_descendant() {
        let (sender, receiver) = std::sync::mpsc::channel();
        let stdin = "x".repeat(1024 * 1024);
        thread::spawn(move || {
            let result = run_cli_command(
                CliCommand {
                    command: "/bin/sh".to_string(),
                    args: vec![
                        "-c".to_string(),
                        "exec 3<&0; sh -c 'sleep 5' 3<&3 & wait".to_string(),
                    ],
                    cwd: None,
                    stdin: Some(stdin),
                },
                1,
            )
            .map(|_| String::new())
            .map_err(|error| format!("{}:{}", error.code, error.message));
            let _ = sender.send(result);
        });

        let result = receiver
            .recv_timeout(Duration::from_secs(3))
            .expect("timeout should not wait for inherited stdin handles");
        let error = result.expect_err("command should time out");
        assert!(error.starts_with("provider_timeout:"));
        assert!(error.contains("timed out after 1s"));
    }

    #[test]
    fn driver_failure_with_stdin_includes_status_and_stderr() {
        let error = run_cli_command(
            CliCommand {
                command: "/bin/sh".to_string(),
                args: vec![
                    "-c".to_string(),
                    "echo stdout text; echo stderr text >&2; exit 7".to_string(),
                ],
                cwd: None,
                stdin: Some("x".repeat(1024 * 1024)),
            },
            5,
        )
        .expect_err("failing command should return provider error");

        assert_eq!(error.code, "invalid_provider_response");
        let details = error.details.expect("details");
        assert!(details["status"].as_str().unwrap().contains('7'));
        assert_eq!(details["stdout"], "stdout text\n");
        assert_eq!(details["stderr"], "stderr text\n");
    }

    #[test]
    fn delayed_driver_failure_with_stdin_includes_status_and_stderr() {
        let error = run_cli_command(
            CliCommand {
                command: "/bin/sh".to_string(),
                args: vec![
                    "-c".to_string(),
                    "exec 0<&-; echo stdout text; echo stderr text >&2; sleep 1; exit 7"
                        .to_string(),
                ],
                cwd: None,
                stdin: Some("x".repeat(1024 * 1024)),
            },
            5,
        )
        .expect_err("failing command should return provider error");

        assert_eq!(error.code, "invalid_provider_response");
        let details = error.details.expect("details");
        assert!(details["status"].as_str().unwrap().contains('7'));
        assert_eq!(details["stdout"], "stdout text\n");
        assert_eq!(details["stderr"], "stderr text\n");
    }

    #[test]
    fn claude_result_field_is_parsed_as_json() {
        let parsed =
            parse_claude_json(r#"{"result":"{\"summary\":\"ok\"}"}"#).expect("claude json");

        assert_eq!(parsed["summary"], "ok");
    }

    #[test]
    fn claude_result_field_can_be_json_object() {
        let parsed = parse_claude_json(r#"{"result":{"summary":"ok"}}"#).expect("claude json");

        assert_eq!(parsed["summary"], "ok");
    }

    #[test]
    fn claude_structured_output_field_is_preferred() {
        let parsed = parse_claude_json(
            r#"{"result":"","structured_output":{"roles":[{"canonical_key":"http_route_handler"}]}}"#,
        )
        .expect("claude structured output");

        assert_eq!(parsed["roles"][0]["canonical_key"], "http_route_handler");
    }

    #[test]
    fn claude_structured_output_wins_over_prose_result() {
        let parsed = parse_claude_json(
            r#"{"result":"I've inferred roles...","structured_output":{"roles":[]}}"#,
        )
        .expect("claude structured output");

        assert_eq!(parsed["roles"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn claude_structured_output_must_be_object() {
        let error = parse_claude_json(r#"{"result":"{}","structured_output":"not-object"}"#)
            .expect_err("non-object structured output should fail");

        assert_eq!(error.code, "invalid_provider_response");
        assert!(
            error.message.contains("structured_output"),
            "error should identify the malformed structured_output field: {}",
            error.message
        );
        let details = error.details.expect("details");
        assert_eq!(details["field"], "structured_output");
        assert_eq!(details["value"], "not-object");
    }
}
