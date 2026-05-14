# Claude Code Print Invocation Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `claude_code_print` structured-generation provider pass prompts, model selection, and JSON Schema to Claude Code correctly without the prompt being consumed by variadic `--allowedTools`.

**Architecture:** Keep the provider boundary inside `crates/bitloops-inference/src/provider/cli_agent.rs`. Extend the internal `CliCommand` helper to support optional stdin input, use stdin for Claude prompts, keep Codex unchanged, and pass Claude-specific structured-output arguments in argv. Add focused unit tests plus one child-process protocol-loop test using a fake Claude script.

**Tech Stack:** Rust 2024, `std::process::Command`, serde/serde_json, existing `assert_cmd` child-process integration tests, existing `mockito` HTTP tests.

---

## File Structure

- Modify `crates/bitloops-inference/src/provider/cli_agent.rs`: add stdin support to `CliCommand`, pass Claude prompt through stdin, pass `--model`, pass `--json-schema` when metadata contains `json_schema`, and update Claude parsing for object-valued `result`.
- Modify `crates/bitloops-inference/tests/protocol_loop.rs`: add a fake-Claude protocol-loop regression test that fails when Claude receives no stdin prompt.
- Modify `README.md`: update the `claude_code_print` documentation so it describes stdin prompt passing, model selection, and schema passing.

No protocol crate changes are needed. The existing `InferRequest.metadata` shape already carries `json_schema` and `workspace_path`.

## Expected Runtime Behavior

The fixed Claude invocation should be semantically equivalent to:

```bash
claude -p \
  --model <profile.model> \
  --output-format json \
  --input-format text \
  --json-schema '<metadata.json_schema as compact JSON>' \
  --allowedTools Read,Grep,Glob
```

The combined prompt from `prompt_for_cli_agent(request)` must be written to child stdin. The prompt must not be appended as a positional argv argument after `--allowedTools`.

`--json-schema` is included only when `metadata.json_schema` exists. This preserves compatibility for callers that use `claude_code_print` without schema metadata, while using the schema supplied by Bitloops structured-generation calls.

---

### Task 1: Add Failing Claude Provider Unit Tests

**Files:**
- Modify: `crates/bitloops-inference/src/provider/cli_agent.rs:376-388`
- Test: `crates/bitloops-inference/src/provider/cli_agent.rs`

- [x] **Step 1: Replace the current Claude command unit test**

Replace `claude_code_print_builds_read_only_command` with this stricter test:

```rust
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
```

- [x] **Step 2: Add a parser regression test for object-valued Claude `result`**

Add this test after `claude_result_field_is_parsed_as_json`:

```rust
    #[test]
    fn claude_result_field_can_be_json_object() {
        let parsed = parse_claude_json(r#"{"result":{"summary":"ok"}}"#).expect("claude json");

        assert_eq!(parsed["summary"], "ok");
    }
```

- [x] **Step 3: Run the targeted test and confirm it fails**

Run:

```bash
cargo test -p bitloops-inference provider::cli_agent::tests::claude_code_print_builds_stdin_command_with_model_schema_and_tools -- --nocapture
```

Expected: fail before implementation. The first failure may be a compile error like:

```text
error[E0609]: no field `stdin` on type `CliCommand`
```

If `stdin` support already exists from a parallel change, expected failure is an assertion mismatch because the current command omits `--model`, omits `--json-schema`, and appends the prompt as argv.

---

### Task 2: Add Optional Stdin Support To `CliCommand`

**Files:**
- Modify: `crates/bitloops-inference/src/provider/cli_agent.rs:1-4`
- Modify: `crates/bitloops-inference/src/provider/cli_agent.rs:23-49`
- Modify: `crates/bitloops-inference/src/provider/cli_agent.rs:191-196`
- Modify: `crates/bitloops-inference/src/provider/cli_agent.rs:232-294`
- Test: `crates/bitloops-inference/src/provider/cli_agent.rs`

- [x] **Step 1: Import `Write` for child stdin**

Change the top imports from:

```rust
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
```

to:

```rust
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
```

- [x] **Step 2: Add `stdin` to `CliCommand`**

Replace the current struct with:

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
struct CliCommand {
    command: String,
    args: Vec<String>,
    cwd: Option<PathBuf>,
    stdin: Option<String>,
}
```

- [x] **Step 3: Keep Codex stdin disabled**

In `CodexExecProvider::build_command`, change the returned `CliCommand` to:

```rust
        Ok(CliCommand {
            command,
            args,
            cwd: workspace_path(request),
            stdin: None,
        })
```

- [x] **Step 4: Teach `run_cli_command` to write optional stdin**

Replace `run_cli_command` with:

```rust
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

    if let Some(stdin) = command.stdin {
        let mut child_stdin = child.stdin.take().ok_or_else(|| {
            ProviderError::provider_transport_error(format!(
                "failed to open stdin for CLI agent `{}`",
                command.command
            ))
        })?;
        child_stdin.write_all(stdin.as_bytes()).map_err(|err| {
            ProviderError::provider_transport_error(format!(
                "failed to write stdin to CLI agent `{}`: {err}",
                command.command
            ))
        })?;
    }

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
```

- [x] **Step 5: Update existing direct `CliCommand` test constructors**

In `driver_failure_includes_status_and_stderr`, add `stdin: None`:

```rust
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
```

- [x] **Step 6: Add a unit test proving stdin is written**

Add this test after `driver_failure_includes_status_and_stderr`:

```rust
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
```

- [x] **Step 7: Run the stdin-focused tests**

Run:

```bash
cargo test -p bitloops-inference provider::cli_agent::tests::run_cli_command_writes_configured_stdin -- --nocapture
cargo test -p bitloops-inference provider::cli_agent::tests::driver_failure_includes_status_and_stderr -- --nocapture
```

Expected: both tests pass after this task. The Claude command-shape test from Task 1 should still fail until Task 3 changes the Claude command.

---

### Task 3: Fix Claude Command Construction And Parsing

**Files:**
- Modify: `crates/bitloops-inference/src/provider/cli_agent.rs:118-134`
- Modify: `crates/bitloops-inference/src/provider/cli_agent.rs:144-150`
- Modify: `crates/bitloops-inference/src/provider/cli_agent.rs:306-311`
- Test: `crates/bitloops-inference/src/provider/cli_agent.rs`

- [x] **Step 1: Replace `ClaudeCodePrintProvider::build_command`**

Replace the current method with:

```rust
    fn build_command(&self, request: &InferenceRequest) -> Result<CliCommand, ProviderError> {
        let command = self.runtime_command()?;
        let mut args = self.profile.runtime_args.clone();
        args.extend([
            "-p".to_string(),
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
```

- [x] **Step 2: Add `json_schema_argument` helper**

Add this helper after `workspace_path`:

```rust
fn json_schema_argument(request: &InferenceRequest) -> Result<Option<String>, ProviderError> {
    let Some(schema) = request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("json_schema"))
    else {
        return Ok(None);
    };

    serde_json::to_string(schema)
        .map(Some)
        .map_err(|err| ProviderError::invalid_request(format!("failed to serialise JSON schema: {err}")))
}
```

After running `cargo fmt`, rustfmt may wrap the `map_err` expression. Keep the error message text unchanged.

- [x] **Step 3: Update Claude provider metadata capabilities**

Change the Claude `structured_output` list from:

```rust
                structured_output: vec![
                    "json_object".to_string(),
                    "prompt_schema_guided".to_string(),
                ],
```

to:

```rust
                structured_output: vec!["json_object".to_string(), "json_schema".to_string()],
```

- [x] **Step 4: Accept object-valued Claude `result` fields**

Replace `parse_claude_json` with:

```rust
fn parse_claude_json(text: &str) -> Result<Value, ProviderError> {
    let outer = parse_cli_json(text)?;
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
```

- [x] **Step 5: Run the Claude provider unit tests**

Run:

```bash
cargo test -p bitloops-inference provider::cli_agent::tests::claude -- --nocapture
```

Expected: all matching Claude provider tests pass, including:

```text
claude_code_print_builds_stdin_command_with_model_schema_and_tools ... ok
claude_result_field_is_parsed_as_json ... ok
claude_result_field_can_be_json_object ... ok
```

- [x] **Step 6: Run all CLI-agent unit tests**

Run:

```bash
cargo test -p bitloops-inference provider::cli_agent::tests -- --nocapture
```

Expected: all CLI-agent unit tests pass.

---

### Task 4: Add Protocol-Loop Regression Coverage For Claude

**Files:**
- Modify: `crates/bitloops-inference/tests/protocol_loop.rs:374-380`
- Test: `crates/bitloops-inference/tests/protocol_loop.rs`

- [x] **Step 1: Add a fake-Claude integration test**

Add this test after `codex_exec_runtime_returns_normalised_parsed_json`:

```rust
#[test]
fn claude_code_print_runtime_passes_prompt_model_and_schema() {
    let temp = tempfile::tempdir().expect("tempdir");
    let script = temp.path().join("fake-claude.sh");
    std::fs::write(
        &script,
        r#"#!/bin/sh
model=""
schema=""
allowed_tools=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model)
      shift
      model="$1"
      ;;
    --json-schema)
      shift
      schema="$1"
      ;;
    --allowedTools|--allowed-tools)
      shift
      allowed_tools="$1"
      ;;
  esac
  shift || true
done

prompt="$(cat)"
case "$prompt" in
  *system*user*) ;;
  *)
    echo "missing prompt on stdin" >&2
    exit 8
    ;;
esac

if [ "$model" != "claude-haiku-4-5" ]; then
  echo "missing model" >&2
  exit 9
fi

case "$schema" in
  *'"type":"object"'*) ;;
  *)
    echo "missing schema" >&2
    exit 10
    ;;
esac

if [ "$allowed_tools" != "Read,Grep,Glob" ]; then
  echo "missing allowed tools" >&2
  exit 11
fi

printf '{"result":"{\"summary\":\"ok\",\"risk_level\":\"low\"}"}'
"#,
    )
    .expect("write fake claude");

    let config = write_config(&format!(
        r#"
            [inference.runtimes.claude]
            command = "/bin/sh"
            args = ["{}"]
            startup_timeout_secs = 5
            request_timeout_secs = 30

            [inference.profiles.local_agent]
            task = "structured_generation"
            driver = "claude_code_print"
            runtime = "claude"
            model = "claude-haiku-4-5"
            temperature = "0.1"
            max_output_tokens = 4096
        "#,
        script.display()
    ));

    let mut runtime = RuntimeHarness::spawn(config.path(), "local_agent");
    runtime.send(&RequestEnvelope {
        request_id: "infer-claude-structured".to_owned(),
        payload: RequestPayload::Infer(InferRequest {
            system_prompt: "system".to_owned(),
            user_prompt: "user".to_owned(),
            response_mode: ResponseMode::JsonObject,
            temperature: None,
            max_output_tokens: None,
            metadata: Some(serde_json::Map::from_iter([(
                "json_schema".to_string(),
                json!({
                    "type": "object",
                    "properties": {
                        "summary": { "type": "string" },
                        "risk_level": { "type": "string" }
                    },
                    "required": ["summary", "risk_level"]
                }),
            )])),
        }),
    });

    let response = runtime.read();
    match response.payload {
        ResponsePayload::Infer(infer) => {
            assert_eq!(infer.provider_name, "claude");
            assert_eq!(infer.model_name, "claude-haiku-4-5");
            assert_eq!(infer.parsed_json.expect("json")["summary"], "ok");
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
}
```

- [x] **Step 2: Run the new integration test**

Run:

```bash
cargo test -p bitloops-inference --test protocol_loop claude_code_print_runtime_passes_prompt_model_and_schema -- --nocapture
```

Expected: pass. This test would fail against the original implementation because the fake script receives an empty stdin prompt and exits with `missing prompt on stdin`.

---

### Task 5: Update Documentation

**Files:**
- Modify: `README.md:100`

- [x] **Step 1: Replace the structured-generation provider paragraph**

Replace the current paragraph:

```markdown
`codex_exec` writes a temporary JSON Schema file, runs `codex exec --output-schema <schema-file> --output-last-message <result-file>`, and returns the parsed result file as `parsed_json`. `claude_code_print` runs `claude -p --output-format json --allowedTools Read,Grep,Glob` and treats schema adherence as prompt-guided JSON rather than strict schema enforcement.
```

with:

```markdown
`codex_exec` writes a temporary JSON Schema file, runs `codex exec --output-schema <schema-file> --output-last-message <result-file>`, and returns the parsed result file as `parsed_json`. `claude_code_print` runs `claude -p --model <model> --output-format json --input-format text --json-schema <schema> --allowedTools Read,Grep,Glob`, writes the combined prompt to stdin, and returns Claude Code's JSON output as `parsed_json`. The `--json-schema` argument is included when the inference request metadata contains `json_schema`.
```

- [x] **Step 2: Run a documentation grep check**

Run:

```bash
rg -n "claude_code_print|allowedTools|json-schema|input-format" README.md crates/bitloops-inference/src/provider/cli_agent.rs
```

Expected: README and source both describe stdin input, model passing, schema passing, and allowed tools.

---

### Task 6: Full Verification

**Files:**
- Verify only; no new source edits expected.

- [x] **Step 1: Format**

Run:

```bash
cargo fmt --all -- --check
```

Expected: exit 0. If this fails, run `cargo fmt --all`, inspect the formatting diff, then rerun the check.

- [x] **Step 2: Run the workspace test suite**

Run:

```bash
cargo test --workspace
```

Expected: all existing and new tests pass.

- [x] **Step 3: Confirm the affected unit test no longer allows prompt-after-variadic-tools**

Run:

```bash
cargo test -p bitloops-inference provider::cli_agent::tests::claude_code_print_builds_stdin_command_with_model_schema_and_tools -- --nocapture
```

Expected: pass, with `command.args` ending in:

```text
--allowedTools Read,Grep,Glob
```

and with `command.stdin` containing:

```text
system

user
```

- [x] **Step 4: Review git diff**

Run:

```bash
git diff -- crates/bitloops-inference/src/provider/cli_agent.rs crates/bitloops-inference/tests/protocol_loop.rs README.md
```

Expected:

- `CliCommand` has `stdin: Option<String>`.
- Codex commands set `stdin: None`.
- Claude commands set `stdin: Some(prompt_for_cli_agent(request))`.
- Claude args include `--model`, `--output-format json`, `--input-format text`, optional `--json-schema`, and `--allowedTools Read,Grep,Glob`.
- The combined prompt is not present in `command.args`.
- Claude parser accepts both string and object `result` values.
- README documents the new Claude invocation behavior.

- [x] **Step 5: Commit**

Run:

```bash
git add crates/bitloops-inference/src/provider/cli_agent.rs crates/bitloops-inference/tests/protocol_loop.rs README.md
git commit -m "fix: pass claude prompts through stdin"
```

Expected: commit succeeds after all verification steps pass.
