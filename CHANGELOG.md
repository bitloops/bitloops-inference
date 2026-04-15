## [0.1.2] - 2026-04-15

### Added

- A dedicated `bitloops_platform_chat` driver for the public Bitloops platform gateway, with a default production chat-completions endpoint and support for per-profile host overrides.
- CLI and protocol-loop coverage for Bitloops gateway profile descriptions, gateway-specific HTTP error envelopes, and host override handling.

### Changed

- The Bitloops gateway is now identified as provider `bitloops` instead of being configured as a plain `openai_chat_completions` profile, while still using the OpenAI chat-completions provider kind under the hood.
- The README now documents the dedicated Bitloops gateway driver and lists it alongside the other supported drivers.

## [0.1.1] - 2026-04-14

### Added

- Support for reading text-generation profiles from the Bitloops daemon config schema, including shared runtimes under `[inference.runtimes.<name>]`.
- Validation for legacy profile fields, unsupported profile keys, missing runtime references, invalid numeric values, and Ollama chat URLs that do not target `/api/chat`.

### Changed

- `bitloops-inference` now reads `task`, `driver`, and `runtime` from daemon-style profile definitions instead of the older per-profile `kind`, `provider_name`, and `timeout_secs` fields.
- Non-`text_generation` inference profiles are ignored during config loading and profile discovery, so CLI commands only expose runnable text-generation profiles.
- Documentation and test fixtures now use the daemon config layout and string-based temperature examples with environment interpolation.

### Fixed

- Request timeouts are now resolved from the referenced runtime’s `request_timeout_secs` value instead of per-profile timeout settings.
- Config validation now reports when a config file does not define any text-generation profiles.

## [0.1.0] - 2026-04-13

### Added

- Initial `bitloops-inference` Rust workspace with a shared protocol crate and a stdio runtime for out-of-process Bitloops inference.
- Protocol v1 request and response types for `describe`, `infer`, and `shutdown`, using line-delimited JSON over `stdin` and `stdout`.
- `run`, `validate-config`, and `describe-profile` CLI commands for running the runtime and inspecting configured inference profiles.
- OpenAI Chat Completions and Ollama Chat providers with normalised text and `json_object` responses, usage reporting, finish reasons, and provider-specific HTTP error handling.
- TOML-based profile configuration with environment-variable interpolation and default inference settings for temperature and output token limits.
- Mocked provider integration tests, child-process protocol-loop tests, hosted-runner CI, and release automation for macOS, Linux, and Windows artefacts.

### Fixed

- Intel macOS release builds now use the correct hosted runner label in the release workflow.
- Release packaging and GitHub Release artefact publication now generate the expected target-specific archives and clean up stale assets.
