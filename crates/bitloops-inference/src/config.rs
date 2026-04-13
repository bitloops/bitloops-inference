use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use bitloops_inference_protocol::ProviderKind;
use serde::Deserialize;
use thiserror::Error;

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
        let mut value = content
            .parse::<toml::Value>()
            .map_err(|source| ConfigError::Parse {
                path: path.to_path_buf(),
                source,
            })?;
        interpolate_toml_value(&mut value, lookup)?;

        let raw: RawConfig = value.try_into().map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            source,
        })?;

        let mut profiles = BTreeMap::new();
        for (name, raw_profile) in raw.inference.profiles {
            let profile = ProfileConfig::from_raw(&name, raw_profile)?;
            profiles.insert(name, profile);
        }

        if profiles.is_empty() {
            return Err(ConfigError::Validation(
                "config must define at least one profile under [inference.profiles.<name>]"
                    .to_owned(),
            ));
        }

        Ok(Self { profiles })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProfileConfig {
    pub kind: ProviderKind,
    pub provider_name: String,
    pub model: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub temperature: Option<f32>,
    pub timeout_secs: u64,
    pub max_output_tokens: Option<u32>,
}

impl ProfileConfig {
    fn from_raw(profile_name: &str, raw: RawProfileConfig) -> Result<Self, ConfigError> {
        let provider_name = validate_non_empty(profile_name, "provider_name", raw.provider_name)?;
        let model = validate_non_empty(profile_name, "model", raw.model)?;
        let base_url = validate_non_empty(profile_name, "base_url", raw.base_url)?;

        if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
            return Err(ConfigError::Validation(format!(
                "profile '{profile_name}' field 'base_url' must start with http:// or https://"
            )));
        }

        let api_key = match raw.api_key {
            Some(api_key) => Some(validate_non_empty(profile_name, "api_key", api_key)?),
            None => None,
        };

        if let Some(temperature) = raw.temperature
            && (!temperature.is_finite() || temperature < 0.0)
        {
            return Err(ConfigError::Validation(format!(
                "profile '{profile_name}' field 'temperature' must be a finite value greater than or equal to 0"
            )));
        }

        if let Some(max_output_tokens) = raw.max_output_tokens
            && max_output_tokens == 0
        {
            return Err(ConfigError::Validation(format!(
                "profile '{profile_name}' field 'max_output_tokens' must be greater than 0"
            )));
        }

        let timeout_secs = raw.timeout_secs.unwrap_or(60);
        if timeout_secs == 0 {
            return Err(ConfigError::Validation(format!(
                "profile '{profile_name}' field 'timeout_secs' must be greater than 0"
            )));
        }

        Ok(Self {
            kind: raw.kind,
            provider_name,
            model,
            base_url,
            api_key,
            temperature: raw.temperature,
            timeout_secs,
            max_output_tokens: raw.max_output_tokens,
        })
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

fn interpolate_toml_value(
    value: &mut toml::Value,
    lookup: &impl Fn(&str) -> Option<String>,
) -> Result<(), ConfigError> {
    match value {
        toml::Value::String(content) => {
            *content = interpolate_string(content, lookup)?;
            Ok(())
        }
        toml::Value::Array(items) => {
            for item in items {
                interpolate_toml_value(item, lookup)?;
            }
            Ok(())
        }
        toml::Value::Table(table) => {
            for (_, item) in table.iter_mut() {
                interpolate_toml_value(item, lookup)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
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

#[derive(Debug, Deserialize)]
struct RawConfig {
    inference: RawInferenceConfig,
}

#[derive(Debug, Deserialize)]
struct RawInferenceConfig {
    profiles: BTreeMap<String, RawProfileConfig>,
}

#[derive(Debug, Deserialize)]
struct RawProfileConfig {
    kind: ProviderKind,
    provider_name: String,
    model: String,
    base_url: String,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    timeout_secs: Option<u64>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
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
    fn interpolates_environment_variables_in_string_fields() {
        let config = parse_config(
            r#"
                [inference.profiles.openai_fast]
                kind = "openai_chat_completions"
                provider_name = "openai"
                model = "gpt-4.1-mini"
                base_url = "${BASE_URL}/v1/chat/completions"
                api_key = "${API_KEY}"
                temperature = 0.1
                max_output_tokens = 200
            "#,
            &|name| match name {
                "BASE_URL" => Some("https://example.com".to_owned()),
                "API_KEY" => Some("secret".to_owned()),
                _ => None,
            },
        )
        .expect("config should parse");

        let profile = config.profile("openai_fast").expect("profile should exist");
        assert_eq!(profile.base_url, "https://example.com/v1/chat/completions");
        assert_eq!(profile.api_key.as_deref(), Some("secret"));
    }

    #[test]
    fn fails_when_environment_variable_is_missing() {
        let error = parse_config(
            r#"
                [inference.profiles.ollama_local]
                kind = "ollama_chat"
                provider_name = "ollama"
                model = "qwen2.5-coder:14b"
                base_url = "${OLLAMA_URL}/api/chat"
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
    fn rejects_zero_timeout() {
        let error = parse_config(
            r#"
                [inference.profiles.openai_fast]
                kind = "openai_chat_completions"
                provider_name = "openai"
                model = "gpt-4.1-mini"
                base_url = "https://example.com/v1/chat/completions"
                timeout_secs = 0
            "#,
            &|_| None,
        )
        .expect_err("config should fail");

        assert!(error.to_string().contains("timeout_secs"));
    }
}
