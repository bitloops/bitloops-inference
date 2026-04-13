use std::ffi::OsString;
use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Help,
    Run { config: PathBuf, profile: String },
    ValidateConfig { config: PathBuf },
    DescribeProfile { config: PathBuf, profile: String },
}

pub fn parse_from<I, T>(args: I) -> Result<Command, CliError>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString>,
{
    let mut values = args.into_iter().map(Into::into).collect::<Vec<OsString>>();
    if !values.is_empty() {
        values.remove(0);
    }
    let mut args = pico_args::Arguments::from_vec(values);

    if args.contains("-h") || args.contains("--help") {
        return Ok(Command::Help);
    }

    let Some(subcommand) = args.subcommand()? else {
        return Ok(Command::Help);
    };

    let command = match subcommand.as_str() {
        "run" => Command::Run {
            config: args.value_from_str("--config")?,
            profile: args.value_from_str("--profile")?,
        },
        "validate-config" => Command::ValidateConfig {
            config: args.value_from_str("--config")?,
        },
        "describe-profile" => Command::DescribeProfile {
            config: args.value_from_str("--config")?,
            profile: args.value_from_str("--profile")?,
        },
        other => return Err(CliError::UnknownSubcommand(other.to_owned())),
    };

    let remaining = args.finish();
    if !remaining.is_empty() {
        return Err(CliError::UnusedArguments(
            remaining
                .iter()
                .map(|value| value.to_string_lossy().into_owned())
                .collect(),
        ));
    }

    Ok(command)
}

pub const fn usage() -> &'static str {
    concat!(
        "bitloops-inference\n\n",
        "USAGE:\n",
        "  bitloops-inference run --config <path> --profile <name>\n",
        "  bitloops-inference validate-config --config <path>\n",
        "  bitloops-inference describe-profile --config <path> --profile <name>\n",
    )
}

#[derive(Debug, Error)]
pub enum CliError {
    #[error("CLI argument error: {0}")]
    Argument(#[from] pico_args::Error),
    #[error("unknown subcommand '{0}'")]
    UnknownSubcommand(String),
    #[error("unexpected trailing arguments: {0:?}")]
    UnusedArguments(Vec<String>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_run_command() {
        let command = parse_from([
            "bitloops-inference",
            "run",
            "--config",
            "config.toml",
            "--profile",
            "openai_fast",
        ])
        .expect("run command should parse");

        assert_eq!(
            command,
            Command::Run {
                config: PathBuf::from("config.toml"),
                profile: "openai_fast".to_owned(),
            }
        );
    }
}
