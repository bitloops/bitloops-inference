# bitloops-inference

`bitloops-inference` is a small Rust workspace that runs semantic-summary inference out of process for Bitloops. Bitloops launches the runtime as a child process, speaks a versioned line-delimited JSON protocol over `stdin` and `stdout`, and leaves all provider-specific HTTP, auth, parsing, and error handling inside this repository.

## Workspace layout

- `bitloops-inference-protocol`: shared protocol types, versioning, and JSON-line serialisation helpers.
- `bitloops-inference`: config loading, CLI, provider registry, provider implementations, and the stdio runtime loop.

## Why this exists

Bitloops core stays provider-agnostic. Adding or changing a summary provider only requires a new `bitloops-inference` release rather than a Bitloops release.

## CLI

```bash
bitloops-inference run --config config.toml --profile openai_fast
bitloops-inference validate-config --config config.toml
bitloops-inference describe-profile --config config.toml --profile openai_fast
```

`run` reserves `stdout` strictly for line-delimited JSON protocol responses. Diagnostics and failures go to `stderr`.

## Config

`bitloops-inference` reads the Bitloops daemon inference config. Text-generation and structured-generation profiles live under `[inference.profiles.<name>]` and reference a runtime from `[inference.runtimes.<name>]`.

```toml
[inference.runtimes.bitloops_inference]
request_timeout_secs = 60

[inference.profiles.openai_fast]
task = "text_generation"
driver = "openai_chat_completions"
runtime = "bitloops_inference"
model = "gpt-4.1-mini"
base_url = "https://api.openai.com/v1/chat/completions"
api_key = "${OPENAI_API_KEY}"
temperature = "0.1"
max_output_tokens = 200

[inference.profiles.ollama_local]
task = "text_generation"
driver = "ollama_chat"
runtime = "bitloops_inference"
model = "qwen2.5-coder:14b"
base_url = "http://127.0.0.1:11434/api/chat"
temperature = "0.1"
max_output_tokens = 200
```

String fields support `${ENV_VAR}` interpolation. Missing environment variables fail validation immediately. Profiles unrelated to text or structured generation in the same daemon config are ignored by `bitloops-inference`.

The public Bitloops platform gateway has a dedicated `bitloops_platform_chat` driver. It defaults to the production Bitloops platform endpoint, and the Bitloops host can optionally provide `base_url` when a test or non-production override is needed:

```toml
[inference.runtimes.bitloops_inference]
request_timeout_secs = 300

[inference.profiles.platform_summary]
task = "text_generation"
driver = "bitloops_platform_chat"
runtime = "bitloops_inference"
model = "ministral-3-3b-instruct"
api_key = "${BITLOOPS_PLATFORM_GATEWAY_TOKEN}"
temperature = "0.1"
max_output_tokens = 200
```

If `base_url` is omitted, `bitloops-inference` uses `https://platform.bitloops.net/v1/chat/completions`. When `base_url` is present, it overrides that default for the selected profile.

## Supported drivers

- `openai_chat_completions`
- `bitloops_platform_chat`
- `ollama_chat`
- `codex_exec`
- `claude_code_print`

All providers normalise their outputs into one canonical inference response with `text`, optional `parsed_json`, optional token usage, finish reason, provider name, and model name.

Structured-generation CLI profiles use the runtime command and args directly:

```toml
[inference.runtimes.codex]
command = "codex"
args = []
startup_timeout_secs = 5
request_timeout_secs = 300

[inference.profiles.local_agent]
task = "structured_generation"
driver = "codex_exec"
runtime = "codex"
model = "gpt-5.4-mini"
temperature = "0.1"
max_output_tokens = 4096
thinking_level = "extra_high"
```

`thinking_level` is optional and only supported by local CLI-agent drivers. For `codex_exec`, supported values are `low`, `medium`, `high`, `extra_high`, and `xhigh`; both `extra_high` and `xhigh` run Codex with `model_reasoning_effort="xhigh"`. For `claude_code_print`, supported values are Claude Code's native effort names: `low`, `medium`, `high`, `xhigh`, and `max`.

`codex_exec` writes a temporary JSON Schema file, runs `codex exec --output-schema <schema-file> --output-last-message <result-file>`, and returns the parsed result file as `parsed_json`. `claude_code_print` runs `claude -p --model <model> --output-format json --input-format text --json-schema <schema> --allowedTools Read,Grep,Glob`, writes the combined prompt to stdin, and returns Claude Code's JSON output as `parsed_json`. When `thinking_level` is present, it is passed as provider-specific CLI configuration. The `--json-schema` argument is included when the inference request metadata contains `json_schema`.

## How Bitloops calls it

1. Start the runtime once for a selected profile.
2. Send JSON requests over `stdin`, one line per request.
3. Read one JSON response line per request from `stdout`.
4. Send `shutdown` when the session is finished.

Example request stream:

```json
{"request_id":"1","type":"describe"}
{"request_id":"2","type":"infer","system_prompt":"You write terse semantic summaries.","user_prompt":"Summarise this diff.","response_mode":"json_object","temperature":0.1,"max_output_tokens":200}
{"request_id":"3","type":"shutdown"}
```

Example responses:

```json
{"request_id":"1","type":"describe","protocol_version":1,"runtime_name":"bitloops-inference","runtime_version":"0.1.2","profile_name":"openai_fast","provider":{"kind":"openai_chat_completions","provider_name":"openai","model_name":"gpt-4.1-mini","endpoint":"https://api.openai.com/v1/chat/completions","capabilities":{"response_modes":["text","json_object"],"usage_reporting":true}}}
{"request_id":"2","type":"infer","text":"{\"summary\":\"Adds provider isolation\",\"confidence\":0.92}","parsed_json":{"summary":"Adds provider isolation","confidence":0.92},"usage":{"prompt_tokens":120,"completion_tokens":24,"total_tokens":144},"finish_reason":"stop","provider_name":"openai","model_name":"gpt-4.1-mini"}
{"request_id":"3","type":"shutdown"}
```

## Running manually

Run config validation first:

```bash
cargo run -p bitloops-inference -- validate-config --config ./bitloops-daemon-config.toml
```

Describe a profile:

```bash
cargo run -p bitloops-inference -- describe-profile --config ./bitloops-daemon-config.toml --profile ollama_local
```

Start the stdio runtime:

```bash
cargo run -p bitloops-inference -- run --config ./bitloops-daemon-config.toml --profile ollama_local
```

You can then write protocol lines to `stdin` manually or from another process.

## Testing

The test suite avoids live network calls. Provider integrations use mocked HTTP servers and the stdio loop is exercised through spawned child-process tests.

```bash
cargo nextest run
cargo dev-clippy
```

There is also an ad hoc manual performance runner that hits a live provider and prints JSON latency analytics, including per-request timings, min/max/mean/median, p95, p99, throughput, and token totals when the provider reports usage. It does not make assertions or act as part of the automated test suite. Each request appends a random cache-buster suffix to the prompt to reduce the chance of provider-side caching affecting the timings. It expects the worker count, prompt, run count, and token through environment variables:

```bash
BITLOOPS_INFERENCE_PERF_WORKERS=4 \
BITLOOPS_INFERENCE_PERF_RUNS=20 \
BITLOOPS_INFERENCE_PERF_PROMPT="Summarise the benefits of isolating provider logic." \
BITLOOPS_PLATFORM_GATEWAY_TOKEN=... \
cargo run -p bitloops-inference --bin bitloops-inference-perf
```

Optional overrides:

- `BITLOOPS_INFERENCE_PERF_DRIVER`: `bitloops_platform_chat` (default) or `openai_chat_completions`
- `BITLOOPS_INFERENCE_PERF_BASE_URL`: override the default provider endpoint
- `BITLOOPS_INFERENCE_PERF_MODEL`: override the default model for the selected driver
- `BITLOOPS_INFERENCE_PERF_SYSTEM_PROMPT`: override the default system prompt
- `BITLOOPS_INFERENCE_PERF_TIMEOUT_SECS`
- `BITLOOPS_INFERENCE_PERF_TEMPERATURE`
- `BITLOOPS_INFERENCE_PERF_MAX_OUTPUT_TOKENS`

## CI and releases

GitHub Actions runs a lean hosted-runner CI pipeline for formatting, clippy, `nextest`, and native release-build smoke checks on Linux, macOS, and Windows.

Tagged releases are published from `v*` tags. The release workflow builds packaged artefacts for:

- `aarch64-apple-darwin`
- `x86_64-apple-darwin`
- `x86_64-unknown-linux-musl`
- `aarch64-unknown-linux-musl`
- `x86_64-pc-windows-msvc`
- `aarch64-pc-windows-msvc`

macOS signing and notarisation use the same secret and variable names as the main Bitloops repository:

- Secrets: `APPLE_CERT_P12_BASE64`, `APPLE_CERT_PASSWORD`, `APPSTORE_CONNECT_API_KEY_P8_BASE64`
- Variables: `APPLE_SIGNING_IDENTITY`, `APPSTORE_CONNECT_KEY_ID`, `APPSTORE_CONNECT_ISSUER_ID`

Optional release notification:

- Secret: `SLACK_WEBHOOK_URL`

## Future work

Possible later provider families include `anthropic_messages` and other explicit provider integrations. v1 deliberately avoids a generic mapping DSL, streaming, batching, local in-process model serving, and runtime orchestration.
