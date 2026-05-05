use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
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
        args.extend([
            "exec".to_string(),
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
        args.extend([
            "-p".to_string(),
            "--output-format".to_string(),
            "json".to_string(),
            "--allowedTools".to_string(),
            "Read,Grep,Glob".to_string(),
            prompt_for_cli_agent(request),
        ]);
        Ok(CliCommand {
            command,
            args,
            cwd: workspace_path(request),
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
                structured_output: vec![
                    "json_object".to_string(),
                    "prompt_schema_guided".to_string(),
                ],
            },
        }
    }

    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse, ProviderError> {
        let command = self.build_command(request)?;
        let output = run_cli_command(command, self.profile.timeout_secs)?;
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliOutput {
    stdout: String,
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

fn run_cli_command(command: CliCommand, timeout_secs: u64) -> Result<CliOutput, ProviderError> {
    let mut process = Command::new(&command.command);
    process.args(&command.args);
    if let Some(cwd) = &command.cwd {
        process.current_dir(cwd);
    }
    process.stdin(Stdio::null());
    process.stdout(Stdio::piped());
    process.stderr(Stdio::piped());

    let mut child = process.spawn().map_err(|err| {
        ProviderError::provider_transport_error(format!(
            "failed to spawn CLI agent `{}`: {err}",
            command.command
        ))
    })?;

    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
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
    }

    let output = child.wait_with_output().map_err(|err| {
        ProviderError::provider_transport_error(format!(
            "failed to collect CLI agent `{}` output: {err}",
            command.command
        ))
    })?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() {
        return Err(ProviderError::invalid_provider_response(
            format!(
                "CLI agent `{}` exited with status {}",
                command.command, output.status
            ),
            Some(json!({
                "status": output.status.to_string(),
                "stdout": stdout,
                "stderr": stderr,
            })),
        ));
    }

    Ok(CliOutput { stdout })
}

fn parse_cli_json(text: &str) -> Result<Value, ProviderError> {
    extract_json_object(text).ok_or_else(|| {
        ProviderError::invalid_provider_response(
            "CLI agent response did not contain a valid JSON object",
            Some(json!({ "text": text })),
        )
    })
}

fn parse_claude_json(text: &str) -> Result<Value, ProviderError> {
    let outer = parse_cli_json(text)?;
    if let Some(result) = outer.get("result").and_then(Value::as_str) {
        return parse_cli_json(result);
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
    }

    #[test]
    fn claude_code_print_builds_read_only_command() {
        let provider =
            ClaudeCodePrintProvider::new(profile(ProviderKind::ClaudeCodePrint, "claude", &[]));
        let command = provider.build_command(&request()).expect("command");

        assert_eq!(command.command, "claude");
        assert_eq!(command.args[0], "-p");
        assert!(command.args.contains(&"--output-format".to_string()));
        assert!(command.args.contains(&"json".to_string()));
        assert!(command.args.contains(&"--allowedTools".to_string()));
        assert!(command.args.contains(&"Read,Grep,Glob".to_string()));
    }

    #[test]
    fn json_extraction_rejects_malformed_output() {
        let error = parse_cli_json("not json").expect_err("malformed output should fail");

        assert_eq!(error.code, "invalid_provider_response");
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
}
