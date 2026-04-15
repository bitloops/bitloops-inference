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

`bitloops-inference` reads the Bitloops daemon inference config. Text-generation profiles live under `[inference.profiles.<name>]` and reference a runtime from `[inference.runtimes.<name>]`.

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

String fields support `${ENV_VAR}` interpolation. Missing environment variables fail validation immediately. Non-text-generation profiles in the same daemon config are ignored by `bitloops-inference`.

The public Bitloops platform gateway works through the same `openai_chat_completions` driver. Point `base_url` at the gateway’s chat-completions endpoint and set `api_key` to the shared bearer token:

```toml
[inference.runtimes.bitloops_inference]
request_timeout_secs = 300

[inference.profiles.platform_summary]
task = "text_generation"
driver = "openai_chat_completions"
runtime = "bitloops_inference"
model = "ministral-3-3b-instruct"
base_url = "https://platform.example.com/v1/chat/completions"
api_key = "${BITLOOPS_PLATFORM_GATEWAY_TOKEN}"
temperature = "0.1"
max_output_tokens = 200
```

`bitloops-inference` treats the gateway as another OpenAI-compatible backend. No extra CLI flags, provider kind, or driver name are required.

## Supported drivers

- `openai_chat_completions`
- `ollama_chat`

Both providers normalise their outputs into one canonical inference response with `text`, optional `parsed_json`, optional token usage, finish reason, provider name, and model name.

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
{"request_id":"1","type":"describe","protocol_version":1,"runtime_name":"bitloops-inference","runtime_version":"0.1.0","profile_name":"openai_fast","provider":{"kind":"openai_chat_completions","provider_name":"openai","model_name":"gpt-4.1-mini","endpoint":"https://api.openai.com/v1/chat/completions","capabilities":{"response_modes":["text","json_object"],"usage_reporting":true}}}
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
