use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use bitloops_inference_protocol::ProviderKind;
use serde::Deserialize;
use thiserror::Error;
use toml::{Table, Value};

const BITLOOPS_PLATFORM_CHAT_DRIVER: &str = "bitloops_platform_chat";
const DEFAULT_BITLOOPS_PLATFORM_CHAT_COMPLETIONS_URL: &str =
    "https://platform.bitloops.net/v1/chat/completions";
const DEFAULT_RUNTIME_STARTUP_TIMEOUT_SECS: u64 = 60;
const DEFAULT_RUNTIME_REQUEST_TIMEOUT_SECS: u64 = 300;

#[derive(Clone, Debug, PartialEq)]
pub struct InferenceConfig {
    profiles: BTreeMap<String, ProfileConfig>,
}

impl InferenceConfig {
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let content = fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;

        Self::parse_from_str_with_lookup(path, &content, &|name| std::env::var(name).ok())
    }

    pub fn profile(&self, name: &str) -> Result<&ProfileConfig, ConfigError> {
        self.profiles
            .get(name)
            .ok_or_else(|| ConfigError::MissingProfile(name.to_owned()))
    }

    pub fn profile_names(&self) -> Vec<String> {
        self.profiles.keys().cloned().collect()
    }

    fn parse_from_str_with_lookup(
        path: &Path,
        content: &str,
        lookup: &impl Fn(&str) -> Option<String>,
    ) -> Result<Self, ConfigError> {
        let value = content
            .parse::<Value>()
            .map_err(|source| ConfigError::Parse {
                path: path.to_path_buf(),
                source,
            })?;

        let raw: RawConfig = value.try_into().map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            source,
        })?;

        let mut profiles = BTreeMap::new();
        for (name, raw_profile) in raw.inference.profiles {
            reject_legacy_profile_fields(&name, &raw_profile)?;
            let task = required_profile_string(&name, &raw_profile, "task", lookup)?;
            if !matches!(
                task.as_str(),
                "text_generation"
                    | "text-generation"
                    | "structured_generation"
                    | "structured-generation"
            ) {
                continue;
            }

            let profile =
                ProfileConfig::from_table(&name, &raw_profile, &raw.inference.runtimes, lookup)?;
            profiles.insert(name, profile);
        }

        if profiles.is_empty() {
            return Err(ConfigError::Validation(
                "config must define at least one text_generation or structured_generation profile under [inference.profiles.<name>]"
                    .to_owned(),
            ));
        }

        Ok(Self { profiles })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProfileConfig {
    pub task: ProfileTask,
    pub kind: ProviderKind,
    pub provider_name: String,
    pub model: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub temperature: Option<f32>,
    pub timeout_secs: u64,
    pub max_output_tokens: Option<u32>,
    pub thinking_level: Option<ThinkingLevel>,
    pub runtime_command: Option<String>,
    pub runtime_args: Vec<String>,
    pub startup_timeout_secs: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThinkingLevel {
    Low,
    Medium,
    High,
    ExtraHigh,
    Max,
}

impl ThinkingLevel {
    pub const fn codex_reasoning_effort(self) -> Option<&'static str> {
        match self {
            Self::Low => Some("low"),
            Self::Medium => Some("medium"),
            Self::High => Some("high"),
            Self::ExtraHigh => Some("xhigh"),
            Self::Max => None,
        }
    }

    pub const fn claude_effort(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::ExtraHigh => "xhigh",
            Self::Max => "max",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfileTask {
    TextGeneration,
    StructuredGeneration,
}

impl ProfileTask {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TextGeneration => "text_generation",
            Self::StructuredGeneration => "structured_generation",
        }
    }
}

fn parse_profile_task(profile_name: &str, raw: &str) -> Result<ProfileTask, ConfigError> {
    match raw {
        "text_generation" | "text-generation" => Ok(ProfileTask::TextGeneration),
        "structured_generation" | "structured-generation" => Ok(ProfileTask::StructuredGeneration),
        other => Err(ConfigError::Validation(format!(
            "profile '{profile_name}' field 'task' has unsupported value '{other}'"
        ))),
    }
}

impl ProfileConfig {
    fn from_table<F>(
        profile_name: &str,
        raw: &Table,
        runtimes: &BTreeMap<String, RawRuntimeConfig>,
        lookup: &F,
    ) -> Result<Self, ConfigError>
    where
        F: Fn(&str) -> Option<String>,
    {
        ensure_only_allowed_fields(
            profile_name,
            raw,
            &[
                "task",
                "driver",
                "runtime",
                "model",
                "base_url",
                "api_key",
                "temperature",
                "max_output_tokens",
                "thinking_level",
            ],
        )?;

        let task = parse_profile_task(
            profile_name,
            &required_profile_string(profile_name, raw, "task", lookup)?,
        )?;

        let driver = required_profile_string(profile_name, raw, "driver", lookup)?;
        let driver_spec = provider_for_driver(profile_name, &driver)?;

        let model = required_profile_string(profile_name, raw, "model", lookup)?;
        let base_url = if driver_spec.requires_base_url {
            let base_url =
                resolve_base_url(profile_name, raw, driver_spec.default_base_url, lookup)?;
            validate_http_url(profile_name, "base_url", &base_url)?;

            if driver_spec.kind == ProviderKind::OllamaChat && !is_ollama_chat_endpoint(&base_url) {
                return Err(ConfigError::Validation(format!(
                    "profile '{profile_name}' field 'base_url' must target the Ollama chat endpoint '/api/chat'"
                )));
            }
            base_url
        } else {
            optional_profile_string(profile_name, raw, "base_url", lookup)?.unwrap_or_default()
        };

        let api_key = optional_profile_string(profile_name, raw, "api_key", lookup)?;
        let temperature = required_profile_f32(profile_name, raw, "temperature", lookup)?;
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(ConfigError::Validation(format!(
                "profile '{profile_name}' field 'temperature' must be a finite value greater than or equal to 0"
            )));
        }

        let max_output_tokens =
            required_profile_u32(profile_name, raw, "max_output_tokens", lookup)?;
        if max_output_tokens == 0 {
            return Err(ConfigError::Validation(format!(
                "profile '{profile_name}' field 'max_output_tokens' must be greater than 0"
            )));
        }
        let thinking_level =
            optional_thinking_level(profile_name, raw, "thinking_level", driver_spec.kind, lookup)?;

        let runtime_name = required_profile_string(profile_name, raw, "runtime", lookup)?;
        let runtime = runtimes.get(&runtime_name).ok_or_else(|| {
            ConfigError::Validation(format!(
                "profile '{profile_name}' references unknown runtime '{runtime_name}'"
            ))
        })?;
        let timeout_secs = runtime
            .request_timeout_secs
            .unwrap_or(DEFAULT_RUNTIME_REQUEST_TIMEOUT_SECS);
        if timeout_secs == 0 {
            return Err(ConfigError::Validation(format!(
                "runtime '{runtime_name}' field 'request_timeout_secs' must be greater than 0"
            )));
        }
        let startup_timeout_secs = runtime
            .startup_timeout_secs
            .unwrap_or(DEFAULT_RUNTIME_STARTUP_TIMEOUT_SECS);
        if startup_timeout_secs == 0 {
            return Err(ConfigError::Validation(format!(
                "runtime '{runtime_name}' field 'startup_timeout_secs' must be greater than 0"
            )));
        }
        let runtime_command = optional_runtime_string(&runtime.command, lookup)?;
        if !driver_spec.requires_base_url && runtime_command.is_none() {
            return Err(ConfigError::Validation(format!(
                "runtime '{runtime_name}' field 'command' is required by profile '{profile_name}'"
            )));
        }
        let runtime_args = runtime_args(&runtime.args, lookup)?;

        if task != driver_spec.task {
            return Err(ConfigError::Validation(format!(
                "profile '{profile_name}' uses driver '{driver}' for task '{}' but declared task '{}'",
                driver_spec.task.as_str(),
                task.as_str()
            )));
        }

        Ok(Self {
            task,
            kind: driver_spec.kind,
            provider_name: driver_spec.provider_name.to_owned(),
            model,
            base_url,
            api_key,
            temperature: Some(temperature),
            timeout_secs,
            max_output_tokens: Some(max_output_tokens),
            thinking_level,
            runtime_command,
            runtime_args,
            startup_timeout_secs,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DriverSpec {
    kind: ProviderKind,
    provider_name: &'static str,
    default_base_url: Option<&'static str>,
    task: ProfileTask,
    requires_base_url: bool,
}

fn provider_for_driver(profile_name: &str, driver: &str) -> Result<DriverSpec, ConfigError> {
    match driver {
        "ollama_chat" => Ok(DriverSpec {
            kind: ProviderKind::OllamaChat,
            provider_name: "ollama",
            default_base_url: None,
            task: ProfileTask::TextGeneration,
            requires_base_url: true,
        }),
        "openai_chat_completions" => Ok(DriverSpec {
            kind: ProviderKind::OpenAiChatCompletions,
            provider_name: "openai",
            default_base_url: None,
            task: ProfileTask::TextGeneration,
            requires_base_url: true,
        }),
        BITLOOPS_PLATFORM_CHAT_DRIVER => Ok(DriverSpec {
            kind: ProviderKind::OpenAiChatCompletions,
            provider_name: "bitloops",
            default_base_url: Some(DEFAULT_BITLOOPS_PLATFORM_CHAT_COMPLETIONS_URL),
            task: ProfileTask::TextGeneration,
            requires_base_url: true,
        }),
        "codex_exec" => Ok(DriverSpec {
            kind: ProviderKind::CodexExec,
            provider_name: "codex",
            default_base_url: None,
            task: ProfileTask::StructuredGeneration,
            requires_base_url: false,
        }),
        "claude_code_print" => Ok(DriverSpec {
            kind: ProviderKind::ClaudeCodePrint,
            provider_name: "claude",
            default_base_url: None,
            task: ProfileTask::StructuredGeneration,
            requires_base_url: false,
        }),
        other => Err(ConfigError::Validation(format!(
            "profile '{profile_name}' field 'driver' has unsupported value '{other}'"
        ))),
    }
}

fn resolve_base_url(
    profile_name: &str,
    table: &Table,
    default_base_url: Option<&'static str>,
    lookup: &impl Fn(&str) -> Option<String>,
) -> Result<String, ConfigError> {
    match table.get("base_url") {
        Some(_) => required_profile_string(profile_name, table, "base_url", lookup),
        None => default_base_url.map(str::to_owned).ok_or_else(|| {
            ConfigError::Validation(format!(
                "profile '{profile_name}' field 'base_url' is required"
            ))
        }),
    }
}

fn reject_legacy_profile_fields(profile_name: &str, table: &Table) -> Result<(), ConfigError> {
    let legacy_fields: Vec<&str> = ["kind", "provider_name", "timeout_secs"]
        .into_iter()
        .filter(|field| table.contains_key(*field))
        .collect();

    if legacy_fields.is_empty() {
        return Ok(());
    }

    Err(ConfigError::Validation(format!(
        "profile '{profile_name}' uses legacy inference field(s): {}; bitloops-inference now expects the Bitloops daemon schema with [inference.runtimes.<name>] and [inference.profiles.<name>]",
        legacy_fields.join(", ")
    )))
}

fn ensure_only_allowed_fields(
    profile_name: &str,
    table: &Table,
    allowed_fields: &[&str],
) -> Result<(), ConfigError> {
    let allowed: BTreeSet<&str> = allowed_fields.iter().copied().collect();
    let unsupported: Vec<&str> = table
        .keys()
        .map(String::as_str)
        .filter(|field| !allowed.contains(*field))
        .collect();

    if unsupported.is_empty() {
        return Ok(());
    }

    Err(ConfigError::Validation(format!(
        "profile '{profile_name}' contains unsupported field(s): {}; supported fields are {}",
        unsupported.join(", "),
        allowed_fields.join(", ")
    )))
}

fn required_profile_string(
    profile_name: &str,
    table: &Table,
    field_name: &str,
    lookup: &impl Fn(&str) -> Option<String>,
) -> Result<String, ConfigError> {
    let value = table.get(field_name).ok_or_else(|| {
        ConfigError::Validation(format!(
            "profile '{profile_name}' field '{field_name}' is required"
        ))
    })?;
    let Some(value) = value.as_str() else {
        return Err(ConfigError::Validation(format!(
            "profile '{profile_name}' field '{field_name}' must be a string"
        )));
    };
    let value = interpolate_string(value, lookup)?;
    validate_non_empty(profile_name, field_name, value)
}

fn optional_profile_string(
    profile_name: &str,
    table: &Table,
    field_name: &str,
    lookup: &impl Fn(&str) -> Option<String>,
) -> Result<Option<String>, ConfigError> {
    let Some(value) = table.get(field_name) else {
        return Ok(None);
    };
    let Some(value) = value.as_str() else {
        return Err(ConfigError::Validation(format!(
            "profile '{profile_name}' field '{field_name}' must be a string"
        )));
    };
    let Some(value) = interpolate_optional_string(value, lookup)? else {
        return Ok(None);
    };
    Ok(Some(validate_non_empty(profile_name, field_name, value)?))
}

fn required_profile_f32(
    profile_name: &str,
    table: &Table,
    field_name: &str,
    lookup: &impl Fn(&str) -> Option<String>,
) -> Result<f32, ConfigError> {
    let value = table.get(field_name).ok_or_else(|| {
        ConfigError::Validation(format!(
            "profile '{profile_name}' field '{field_name}' is required"
        ))
    })?;

    match value {
        Value::Float(number) => Ok(*number as f32),
        Value::Integer(number) => Ok(*number as f32),
        Value::String(raw) => {
            let raw = interpolate_string(raw, lookup)?;
            raw.trim().parse::<f32>().map_err(|_| {
                ConfigError::Validation(format!(
                    "profile '{profile_name}' field '{field_name}' must be a valid number"
                ))
            })
        }
        _ => Err(ConfigError::Validation(format!(
            "profile '{profile_name}' field '{field_name}' must be a string or number"
        ))),
    }
}

fn required_profile_u32(
    profile_name: &str,
    table: &Table,
    field_name: &str,
    lookup: &impl Fn(&str) -> Option<String>,
) -> Result<u32, ConfigError> {
    let value = table.get(field_name).ok_or_else(|| {
        ConfigError::Validation(format!(
            "profile '{profile_name}' field '{field_name}' is required"
        ))
    })?;

    match value {
        Value::Integer(number) => u32::try_from(*number).map_err(|_| {
            ConfigError::Validation(format!(
                "profile '{profile_name}' field '{field_name}' must fit within a u32"
            ))
        }),
        Value::String(raw) => {
            let raw = interpolate_string(raw, lookup)?;
            raw.trim().parse::<u32>().map_err(|_| {
                ConfigError::Validation(format!(
                    "profile '{profile_name}' field '{field_name}' must be an integer"
                ))
            })
        }
        _ => Err(ConfigError::Validation(format!(
            "profile '{profile_name}' field '{field_name}' must be an integer"
        ))),
    }
}

fn optional_thinking_level(
    profile_name: &str,
    table: &Table,
    field_name: &str,
    kind: ProviderKind,
    lookup: &impl Fn(&str) -> Option<String>,
) -> Result<Option<ThinkingLevel>, ConfigError> {
    let Some(raw) = optional_profile_string(profile_name, table, field_name, lookup)? else {
        return Ok(None);
    };

    match kind {
        ProviderKind::CodexExec => parse_codex_thinking_level(profile_name, field_name, &raw),
        ProviderKind::ClaudeCodePrint => parse_claude_thinking_level(profile_name, field_name, &raw),
        ProviderKind::OpenAiChatCompletions | ProviderKind::OllamaChat => {
            Err(ConfigError::Validation(format!(
                "profile '{profile_name}' field '{field_name}' is only supported for local CLI-agent drivers"
            )))
        }
    }
    .map(Some)
}

fn parse_codex_thinking_level(
    profile_name: &str,
    field_name: &str,
    raw: &str,
) -> Result<ThinkingLevel, ConfigError> {
    match raw {
        "low" => Ok(ThinkingLevel::Low),
        "medium" => Ok(ThinkingLevel::Medium),
        "high" => Ok(ThinkingLevel::High),
        "extra_high" | "xhigh" => Ok(ThinkingLevel::ExtraHigh),
        other => Err(ConfigError::Validation(format!(
            "profile '{profile_name}' field '{field_name}' has unsupported value '{other}' for driver 'codex_exec'; supported values are low, medium, high, extra_high, xhigh"
        ))),
    }
}

fn parse_claude_thinking_level(
    profile_name: &str,
    field_name: &str,
    raw: &str,
) -> Result<ThinkingLevel, ConfigError> {
    match raw {
        "low" => Ok(ThinkingLevel::Low),
        "medium" => Ok(ThinkingLevel::Medium),
        "high" => Ok(ThinkingLevel::High),
        "xhigh" => Ok(ThinkingLevel::ExtraHigh),
        "max" => Ok(ThinkingLevel::Max),
        other => Err(ConfigError::Validation(format!(
            "profile '{profile_name}' field '{field_name}' has unsupported value '{other}' for driver 'claude_code_print'; supported values are low, medium, high, xhigh, max"
        ))),
    }
}

fn validate_non_empty(
    profile_name: &str,
    field_name: &str,
    value: String,
) -> Result<String, ConfigError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(ConfigError::Validation(format!(
            "profile '{profile_name}' field '{field_name}' must not be blank"
        )));
    }

    Ok(trimmed.to_owned())
}

fn validate_http_url(profile_name: &str, field_name: &str, value: &str) -> Result<(), ConfigError> {
    if value.starts_with("http://") || value.starts_with("https://") {
        return Ok(());
    }

    Err(ConfigError::Validation(format!(
        "profile '{profile_name}' field '{field_name}' must start with http:// or https://"
    )))
}

fn is_ollama_chat_endpoint(base_url: &str) -> bool {
    base_url.trim_end_matches('/').ends_with("/api/chat")
}

fn interpolate_string(
    input: &str,
    lookup: &impl Fn(&str) -> Option<String>,
) -> Result<String, ConfigError> {
    let mut cursor = 0usize;
    let mut output = String::with_capacity(input.len());

    while let Some(start) = input[cursor..].find("${") {
        let absolute_start = cursor + start;
        output.push_str(&input[cursor..absolute_start]);

        let variable_start = absolute_start + 2;
        let remainder = &input[variable_start..];
        let Some(end_offset) = remainder.find('}') else {
            return Err(ConfigError::Interpolation(
                "unterminated environment placeholder".to_owned(),
            ));
        };
        let variable_name = &remainder[..end_offset];
        if variable_name.is_empty() {
            return Err(ConfigError::Interpolation(
                "environment placeholder must not be empty".to_owned(),
            ));
        }

        let Some(value) = lookup(variable_name) else {
            return Err(ConfigError::Interpolation(format!(
                "environment variable '{variable_name}' is not set"
            )));
        };

        output.push_str(&value);
        cursor = variable_start + end_offset + 1;
    }

    output.push_str(&input[cursor..]);
    Ok(output)
}

fn interpolate_optional_string(
    input: &str,
    lookup: &impl Fn(&str) -> Option<String>,
) -> Result<Option<String>, ConfigError> {
    let trimmed = input.trim();
    if let Some(variable_name) = trimmed
        .strip_prefix("${")
        .and_then(|value| value.strip_suffix('}'))
    {
        if variable_name.is_empty() {
            return Err(ConfigError::Interpolation(
                "environment placeholder must not be empty".to_owned(),
            ));
        }

        return Ok(lookup(variable_name)
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty()));
    }

    interpolate_string(input, lookup).map(Some)
}

fn optional_runtime_string(
    input: &Option<String>,
    lookup: &impl Fn(&str) -> Option<String>,
) -> Result<Option<String>, ConfigError> {
    let Some(input) = input.as_deref() else {
        return Ok(None);
    };
    interpolate_optional_string(input, lookup)
}

fn runtime_args(
    input: &[String],
    lookup: &impl Fn(&str) -> Option<String>,
) -> Result<Vec<String>, ConfigError> {
    input
        .iter()
        .filter_map(|arg| match interpolate_optional_string(arg, lookup) {
            Ok(Some(value)) => Some(Ok(value)),
            Ok(None) => None,
            Err(err) => Some(Err(err)),
        })
        .collect()
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    inference: RawInferenceConfig,
}

#[derive(Debug, Default, Deserialize)]
struct RawInferenceConfig {
    #[serde(default)]
    runtimes: BTreeMap<String, RawRuntimeConfig>,
    #[serde(default)]
    profiles: BTreeMap<String, Table>,
}

#[derive(Debug, Default, Deserialize)]
struct RawRuntimeConfig {
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    startup_timeout_secs: Option<u64>,
    #[serde(default)]
    request_timeout_secs: Option<u64>,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read config file {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse config file {path}: {source}")]
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },
    #[error("config validation failed: {0}")]
    Validation(String),
    #[error("config interpolation failed: {0}")]
    Interpolation(String),
    #[error("profile '{0}' was not found in the config")]
    MissingProfile(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_config(
        content: &str,
        lookup: &impl Fn(&str) -> Option<String>,
    ) -> Result<InferenceConfig, ConfigError> {
        InferenceConfig::parse_from_str_with_lookup(Path::new("test-config.toml"), content, lookup)
    }

    #[test]
    fn interpolates_environment_variables_in_daemon_style_config() {
        let config = parse_config(
            r#"
                [inference.runtimes.bitloops_inference]
                request_timeout_secs = 120

                [inference.profiles.local_code]
                task = "embeddings"
                driver = "ollama_embeddings"
                model = "nomic-embed-text"
                base_url = "http://127.0.0.1:11434/api/embed"

                [inference.profiles.summary_local]
                task = "text_generation"
                driver = "openai_chat_completions"
                runtime = "bitloops_inference"
                model = "gpt-4.1-mini"
                base_url = "${BASE_URL}/v1/chat/completions"
                api_key = "${API_KEY}"
                temperature = "${TEMPERATURE}"
                max_output_tokens = 200
            "#,
            &|name| match name {
                "BASE_URL" => Some("https://example.com".to_owned()),
                "API_KEY" => Some("secret".to_owned()),
                "TEMPERATURE" => Some("0.1".to_owned()),
                _ => None,
            },
        )
        .expect("config should parse");

        let profile = config
            .profile("summary_local")
            .expect("profile should exist");
        assert_eq!(config.profile_names(), vec!["summary_local".to_owned()]);
        assert_eq!(profile.kind, ProviderKind::OpenAiChatCompletions);
        assert_eq!(profile.provider_name, "openai");
        assert_eq!(profile.base_url, "https://example.com/v1/chat/completions");
        assert_eq!(profile.api_key.as_deref(), Some("secret"));
        assert_eq!(profile.temperature, Some(0.1));
        assert_eq!(profile.timeout_secs, 120);
        assert_eq!(profile.max_output_tokens, Some(200));
    }

    #[test]
    fn bitloops_platform_driver_defaults_to_production_gateway() {
        let config = parse_config(
            r#"
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
            "#,
            &|name| match name {
                "BITLOOPS_PLATFORM_GATEWAY_TOKEN" => Some("secret".to_owned()),
                _ => None,
            },
        )
        .expect("config should parse");

        let profile = config
            .profile("platform_summary")
            .expect("profile should exist");
        assert_eq!(profile.kind, ProviderKind::OpenAiChatCompletions);
        assert_eq!(profile.provider_name, "bitloops");
        assert_eq!(
            profile.base_url,
            DEFAULT_BITLOOPS_PLATFORM_CHAT_COMPLETIONS_URL
        );
    }

    #[test]
    fn bitloops_platform_driver_allows_base_url_override() {
        let config = parse_config(
            r#"
                [inference.runtimes.bitloops_inference]
                request_timeout_secs = 300

                [inference.profiles.platform_summary]
                task = "text_generation"
                driver = "bitloops_platform_chat"
                runtime = "bitloops_inference"
                model = "ministral-3-3b-instruct"
                base_url = "https://platform.example.com/v1/chat/completions"
                api_key = "${BITLOOPS_PLATFORM_GATEWAY_TOKEN}"
                temperature = "0.1"
                max_output_tokens = 200
            "#,
            &|name| match name {
                "BITLOOPS_PLATFORM_GATEWAY_TOKEN" => Some("secret".to_owned()),
                _ => None,
            },
        )
        .expect("config should parse");

        let profile = config
            .profile("platform_summary")
            .expect("profile should exist");
        assert_eq!(
            profile.base_url,
            "https://platform.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn bitloops_platform_driver_allows_missing_optional_api_key_env() {
        let config = parse_config(
            r#"
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
            "#,
            &|_| None,
        )
        .expect("config should parse");

        let profile = config
            .profile("platform_summary")
            .expect("profile should exist");
        assert_eq!(profile.api_key, None);
        assert_eq!(
            profile.base_url,
            DEFAULT_BITLOOPS_PLATFORM_CHAT_COMPLETIONS_URL
        );
    }

    #[test]
    fn accepts_structured_generation_cli_profile() {
        let config = parse_config(
            r#"
                [inference.runtimes.codex]
                command = "codex"
                args = ["--ask-for-approval", "never"]
                startup_timeout_secs = 5
                request_timeout_secs = 300

                [inference.profiles.local_agent]
                task = "structured_generation"
                driver = "codex_exec"
                runtime = "codex"
                model = "gpt-5.4-mini"
                temperature = "0.1"
                max_output_tokens = 4096
            "#,
            &|_| None,
        )
        .expect("config should parse");

        let profile = config.profile("local_agent").expect("profile should exist");
        assert_eq!(config.profile_names(), vec!["local_agent".to_owned()]);
        assert_eq!(profile.task, ProfileTask::StructuredGeneration);
        assert_eq!(profile.kind, ProviderKind::CodexExec);
        assert_eq!(profile.runtime_command.as_deref(), Some("codex"));
        assert_eq!(
            profile.runtime_args,
            vec!["--ask-for-approval".to_string(), "never".to_string()]
        );
        assert_eq!(profile.startup_timeout_secs, 5);
        assert_eq!(profile.timeout_secs, 300);
    }

    #[test]
    fn accepts_codex_thinking_level_extra_high() {
        let config = parse_config(
            r#"
                [inference.runtimes.codex]
                command = "codex"
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
            "#,
            &|_| None,
        )
        .expect("config should parse");

        let profile = config.profile("local_agent").expect("profile should exist");
        assert_eq!(profile.thinking_level, Some(ThinkingLevel::ExtraHigh));
    }

    #[test]
    fn accepts_codex_native_xhigh_alias() {
        let config = parse_config(
            r#"
                [inference.runtimes.codex]
                command = "codex"
                request_timeout_secs = 300

                [inference.profiles.local_agent]
                task = "structured_generation"
                driver = "codex_exec"
                runtime = "codex"
                model = "gpt-5.4-mini"
                temperature = "0.1"
                max_output_tokens = 4096
                thinking_level = "xhigh"
            "#,
            &|_| None,
        )
        .expect("config should parse");

        let profile = config.profile("local_agent").expect("profile should exist");
        assert_eq!(profile.thinking_level, Some(ThinkingLevel::ExtraHigh));
    }

    #[test]
    fn accepts_claude_thinking_level_max() {
        let config = parse_config(
            r#"
                [inference.runtimes.claude]
                command = "claude"
                startup_timeout_secs = 5
                request_timeout_secs = 300

                [inference.profiles.local_agent]
                task = "structured_generation"
                driver = "claude_code_print"
                runtime = "claude"
                model = "claude-opus-4-7"
                temperature = "0.1"
                max_output_tokens = 4096
                thinking_level = "max"
            "#,
            &|_| None,
        )
        .expect("config should parse");

        let profile = config.profile("local_agent").expect("profile should exist");
        assert_eq!(profile.thinking_level, Some(ThinkingLevel::Max));
    }

    #[test]
    fn rejects_unsupported_codex_thinking_level() {
        let error = parse_config(
            r#"
                [inference.runtimes.codex]
                command = "codex"
                request_timeout_secs = 300

                [inference.profiles.local_agent]
                task = "structured_generation"
                driver = "codex_exec"
                runtime = "codex"
                model = "gpt-5.4-mini"
                temperature = "0.1"
                max_output_tokens = 4096
                thinking_level = "max"
            "#,
            &|_| None,
        )
        .expect_err("config should fail");

        assert!(error.to_string().contains("thinking_level"));
        assert!(error.to_string().contains("codex_exec"));
        assert!(
            error
                .to_string()
                .contains("low, medium, high, extra_high, xhigh")
        );
    }

    #[test]
    fn rejects_thinking_level_for_http_profiles() {
        let error = parse_config(
            r#"
                [inference.runtimes.bitloops_inference]
                request_timeout_secs = 120

                [inference.profiles.summary_local]
                task = "text_generation"
                driver = "openai_chat_completions"
                runtime = "bitloops_inference"
                model = "gpt-4.1-mini"
                base_url = "https://example.com/v1/chat/completions"
                temperature = "0.1"
                max_output_tokens = 200
                thinking_level = "high"
            "#,
            &|_| None,
        )
        .expect_err("config should fail");

        assert!(error.to_string().contains("thinking_level"));
        assert!(
            error
                .to_string()
                .contains("only supported for local CLI-agent drivers")
        );
    }

    #[test]
    fn fails_when_environment_variable_is_missing() {
        let error = parse_config(
            r#"
                [inference.runtimes.bitloops_inference]
                request_timeout_secs = 120

                [inference.profiles.summary_local]
                task = "text_generation"
                driver = "ollama_chat"
                runtime = "bitloops_inference"
                model = "qwen2.5-coder:14b"
                base_url = "${OLLAMA_URL}/api/chat"
                temperature = "0.1"
                max_output_tokens = 200
            "#,
            &|_| None,
        )
        .expect_err("config should fail");

        assert!(
            error
                .to_string()
                .contains("environment variable 'OLLAMA_URL' is not set")
        );
    }

    #[test]
    fn rejects_legacy_profile_schema() {
        let error = parse_config(
            r#"
                [inference.profiles.summary_local]
                kind = "ollama_chat"
                provider_name = "ollama"
                model = "qwen2.5-coder:14b"
                base_url = "http://127.0.0.1:11434/api/chat"
                timeout_secs = 120
                temperature = 0.1
                max_output_tokens = 200
            "#,
            &|_| None,
        )
        .expect_err("config should fail");

        assert!(error.to_string().contains("legacy inference field"));
    }

    #[test]
    fn rejects_missing_runtime_reference() {
        let error = parse_config(
            r#"
                [inference.runtimes.bitloops_inference]
                request_timeout_secs = 120

                [inference.profiles.summary_local]
                task = "text_generation"
                driver = "ollama_chat"
                runtime = "missing_runtime"
                model = "qwen2.5-coder:14b"
                base_url = "http://127.0.0.1:11434/api/chat"
                temperature = "0.1"
                max_output_tokens = 200
            "#,
            &|_| None,
        )
        .expect_err("config should fail");

        assert!(
            error
                .to_string()
                .contains("unknown runtime 'missing_runtime'")
        );
    }

    #[test]
    fn rejects_zero_runtime_timeout() {
        let error = parse_config(
            r#"
                [inference.runtimes.bitloops_inference]
                request_timeout_secs = 0

                [inference.profiles.summary_local]
                task = "text_generation"
                driver = "openai_chat_completions"
                runtime = "bitloops_inference"
                model = "gpt-4.1-mini"
                base_url = "https://example.com/v1/chat/completions"
                temperature = "0.1"
                max_output_tokens = 200
            "#,
            &|_| None,
        )
        .expect_err("config should fail");

        assert!(error.to_string().contains("request_timeout_secs"));
    }

    #[test]
    fn rejects_ollama_host_root_url() {
        let error = parse_config(
            r#"
                [inference.runtimes.bitloops_inference]
                request_timeout_secs = 120

                [inference.profiles.summary_local]
                task = "text_generation"
                driver = "ollama_chat"
                runtime = "bitloops_inference"
                model = "qwen2.5-coder:14b"
                base_url = "http://127.0.0.1:11434"
                temperature = "0.1"
                max_output_tokens = 200
            "#,
            &|_| None,
        )
        .expect_err("config should fail");

        assert!(
            error
                .to_string()
                .contains("Ollama chat endpoint '/api/chat'")
        );
    }
}
