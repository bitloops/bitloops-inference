mod cli;
mod config;
mod json;
mod provider;
mod runtime;

use std::ffi::OsString;
use std::io;

use bitloops_inference_protocol::ResponseEnvelope;
use serde_json::json;
use thiserror::Error;

use crate::cli::Command;
use crate::config::InferenceConfig;
use crate::provider::ProviderRegistry;
use crate::runtime::Runtime;

pub fn run_from_args<I, T>(args: I) -> Result<(), AppError>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString>,
{
    match cli::parse_from(args)? {
        Command::Help => {
            println!("{}", cli::usage());
            Ok(())
        }
        Command::Run { config, profile } => runtime::run_stdio(&config, &profile),
        Command::ValidateConfig { config } => {
            let config = InferenceConfig::load(&config)?;
            let profiles = config.profile_names();
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "status": "ok",
                    "profiles": profiles,
                }))?
            );
            Ok(())
        }
        Command::DescribeProfile { config, profile } => {
            let config = InferenceConfig::load(&config)?;
            let profile_config = config.profile(&profile)?;
            let runtime =
                Runtime::new(profile.clone(), profile_config, ProviderRegistry::default())?;
            let response = ResponseEnvelope {
                request_id: "describe-profile".to_owned(),
                payload: bitloops_inference_protocol::ResponsePayload::Describe(runtime.describe()),
            };
            println!("{}", response.to_json_line()?);
            Ok(())
        }
    }
}

#[derive(Debug, Error)]
pub enum AppError {
    #[error(transparent)]
    Cli(#[from] cli::CliError),
    #[error(transparent)]
    Config(#[from] config::ConfigError),
    #[error(transparent)]
    Provider(#[from] provider::ProviderError),
    #[error(transparent)]
    Protocol(#[from] bitloops_inference_protocol::ProtocolCodecError),
    #[error("failed to serialise JSON output: {0}")]
    Json(#[from] serde_json::Error),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
}
