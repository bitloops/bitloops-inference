use std::cmp;
use std::thread;
use std::time::Instant;

use bitloops_inference_protocol::{ProviderKind, ResponseMode, TokenUsage};
use getrandom::getrandom;
use serde::Serialize;
use thiserror::Error;

use crate::config::ProfileConfig;
use crate::provider::{InferenceRequest, InferenceResponse, ProviderError, ProviderRegistry};

const PERF_TOKEN_ENV: &str = "BITLOOPS_PLATFORM_GATEWAY_TOKEN";
const PERF_PROMPT_ENV: &str = "BITLOOPS_INFERENCE_PERF_PROMPT";
const PERF_RUNS_ENV: &str = "BITLOOPS_INFERENCE_PERF_RUNS";
const PERF_WORKERS_ENV: &str = "BITLOOPS_INFERENCE_PERF_WORKERS";
const PERF_DRIVER_ENV: &str = "BITLOOPS_INFERENCE_PERF_DRIVER";
const PERF_BASE_URL_ENV: &str = "BITLOOPS_INFERENCE_PERF_BASE_URL";
const PERF_MODEL_ENV: &str = "BITLOOPS_INFERENCE_PERF_MODEL";
const PERF_SYSTEM_PROMPT_ENV: &str = "BITLOOPS_INFERENCE_PERF_SYSTEM_PROMPT";
const PERF_TIMEOUT_SECS_ENV: &str = "BITLOOPS_INFERENCE_PERF_TIMEOUT_SECS";
const PERF_TEMPERATURE_ENV: &str = "BITLOOPS_INFERENCE_PERF_TEMPERATURE";
const PERF_MAX_OUTPUT_TOKENS_ENV: &str = "BITLOOPS_INFERENCE_PERF_MAX_OUTPUT_TOKENS";

const DEFAULT_SYSTEM_PROMPT: &str = "You are a concise assistant. Reply in plain text.";
const DEFAULT_PLATFORM_DRIVER: &str = "bitloops_platform_chat";
const DEFAULT_PLATFORM_MODEL: &str = "ministral-3-3b-instruct";
const DEFAULT_PLATFORM_BASE_URL: &str = "https://platform.bitloops.net/v1/chat/completions";
const DEFAULT_OPENAI_DRIVER: &str = "openai_chat_completions";
const DEFAULT_OPENAI_MODEL: &str = "gpt-4.1-mini";
const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1/chat/completions";
const DEFAULT_TIMEOUT_SECS: u64 = 300;
const DEFAULT_TEMPERATURE: f32 = 0.1;
const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 200;

pub fn run_perf_report_from_env() -> Result<(), PerfError> {
    let settings = PerfSettings::from_env()?;
    let request = settings.request();
    let suite_started_at = Instant::now();
    let effective_workers = cmp::min(settings.workers, settings.runs);
    let mut handles = Vec::with_capacity(effective_workers);

    for worker_id in 0..effective_workers {
        let settings = settings.clone();
        let request = request.clone();
        let assigned_runs = assigned_runs_for_worker(worker_id, settings.runs, effective_workers);
        let request_offset = request_offset_for_worker(worker_id, settings.runs, effective_workers);

        handles.push(thread::spawn(move || {
            run_worker(
                worker_id,
                assigned_runs,
                request_offset,
                suite_started_at,
                &settings,
                &request,
            )
        }));
    }

    let mut worker_reports = Vec::with_capacity(effective_workers);
    let mut request_reports = Vec::with_capacity(settings.runs);
    for handle in handles {
        let worker_report = handle
            .join()
            .map_err(|_| PerfError::WorkerPanic)?
            .map_err(|error| PerfError::Worker(error.to_string()))?;

        request_reports.extend(worker_report.requests.iter().cloned());
        worker_reports.push(worker_report);
    }

    request_reports.sort_by_key(|report| report.request_index);
    worker_reports.sort_by_key(|report| report.worker_id);

    let analytics = AnalyticsReport::from_reports(
        settings,
        suite_started_at.elapsed().as_secs_f64() * 1_000.0,
        worker_reports,
        request_reports,
    );

    println!("{}", serde_json::to_string_pretty(&analytics)?);
    Ok(())
}

fn run_worker(
    worker_id: usize,
    assigned_runs: usize,
    request_offset: usize,
    suite_started_at: Instant,
    settings: &PerfSettings,
    request: &InferenceRequest,
) -> Result<WorkerReport, WorkerError> {
    let worker_started_at = Instant::now();
    let provider = ProviderRegistry::default().create(&settings.profile())?;
    let mut requests = Vec::with_capacity(assigned_runs);

    for run_index in 0..assigned_runs {
        let request_index = request_offset + run_index;
        let request_id = format!("worker-{worker_id}-run-{request_index}");
        let request = request_with_cache_buster(request, worker_id, request_index)?;
        let started_at_ms = suite_started_at.elapsed().as_secs_f64() * 1_000.0;
        let started_at = Instant::now();
        let result = provider.infer(&request);
        let latency_ms = started_at.elapsed().as_secs_f64() * 1_000.0;

        requests.push(RequestReport::from_result(
            worker_id,
            request_index,
            request_id,
            started_at_ms,
            latency_ms,
            result,
        ));
    }

    Ok(WorkerReport {
        worker_id,
        assigned_runs,
        wall_time_ms: worker_started_at.elapsed().as_secs_f64() * 1_000.0,
        requests,
    })
}

fn request_with_cache_buster(
    request: &InferenceRequest,
    worker_id: usize,
    request_index: usize,
) -> Result<InferenceRequest, WorkerError> {
    let mut request = request.clone();
    let cache_buster = random_cache_buster(worker_id, request_index)?;

    // Each request gets a unique prompt suffix to avoid provider-side cache hits.
    request.user_prompt = format!("{}\n\n[cache-buster:{}]", request.user_prompt, cache_buster);

    Ok(request)
}

fn random_cache_buster(worker_id: usize, request_index: usize) -> Result<String, WorkerError> {
    let mut bytes = [0u8; 16];
    getrandom(&mut bytes).map_err(|error| WorkerError::Random(error.to_string()))?;

    Ok(format!(
        "w{worker_id}-r{request_index}-{}",
        hex_encode(&bytes)
    ))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);

    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }

    encoded
}

fn assigned_runs_for_worker(worker_id: usize, total_runs: usize, workers: usize) -> usize {
    let base = total_runs / workers;
    let remainder = total_runs % workers;
    base + usize::from(worker_id < remainder)
}

fn request_offset_for_worker(worker_id: usize, total_runs: usize, workers: usize) -> usize {
    let base = total_runs / workers;
    let remainder = total_runs % workers;
    (worker_id * base) + cmp::min(worker_id, remainder)
}

#[derive(Clone, Debug, Serialize)]
struct PerfSettings {
    driver: String,
    base_url: String,
    model: String,
    prompt: String,
    system_prompt: String,
    runs: usize,
    workers: usize,
    timeout_secs: u64,
    temperature: f32,
    max_output_tokens: u32,
    token: String,
}

impl PerfSettings {
    fn from_env() -> Result<Self, PerfError> {
        let driver =
            std::env::var(PERF_DRIVER_ENV).unwrap_or_else(|_| DEFAULT_PLATFORM_DRIVER.to_owned());

        let settings = match driver.as_str() {
            DEFAULT_PLATFORM_DRIVER => Self {
                driver,
                base_url: optional_env(PERF_BASE_URL_ENV)
                    .unwrap_or_else(|| DEFAULT_PLATFORM_BASE_URL.to_owned()),
                model: optional_env(PERF_MODEL_ENV)
                    .unwrap_or_else(|| DEFAULT_PLATFORM_MODEL.to_owned()),
                prompt: required_env(PERF_PROMPT_ENV)?,
                system_prompt: optional_env(PERF_SYSTEM_PROMPT_ENV)
                    .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_owned()),
                runs: parse_required_env(PERF_RUNS_ENV)?,
                workers: parse_required_env(PERF_WORKERS_ENV)?,
                timeout_secs: parse_env_with_default(PERF_TIMEOUT_SECS_ENV, DEFAULT_TIMEOUT_SECS)?,
                temperature: parse_env_with_default(PERF_TEMPERATURE_ENV, DEFAULT_TEMPERATURE)?,
                max_output_tokens: parse_env_with_default(
                    PERF_MAX_OUTPUT_TOKENS_ENV,
                    DEFAULT_MAX_OUTPUT_TOKENS,
                )?,
                token: required_env(PERF_TOKEN_ENV)?,
            },
            DEFAULT_OPENAI_DRIVER => Self {
                driver,
                base_url: optional_env(PERF_BASE_URL_ENV)
                    .unwrap_or_else(|| DEFAULT_OPENAI_BASE_URL.to_owned()),
                model: optional_env(PERF_MODEL_ENV)
                    .unwrap_or_else(|| DEFAULT_OPENAI_MODEL.to_owned()),
                prompt: required_env(PERF_PROMPT_ENV)?,
                system_prompt: optional_env(PERF_SYSTEM_PROMPT_ENV)
                    .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_owned()),
                runs: parse_required_env(PERF_RUNS_ENV)?,
                workers: parse_required_env(PERF_WORKERS_ENV)?,
                timeout_secs: parse_env_with_default(PERF_TIMEOUT_SECS_ENV, DEFAULT_TIMEOUT_SECS)?,
                temperature: parse_env_with_default(PERF_TEMPERATURE_ENV, DEFAULT_TEMPERATURE)?,
                max_output_tokens: parse_env_with_default(
                    PERF_MAX_OUTPUT_TOKENS_ENV,
                    DEFAULT_MAX_OUTPUT_TOKENS,
                )?,
                token: required_env(PERF_TOKEN_ENV)?,
            },
            other => {
                return Err(PerfError::Env(format!(
                    "{PERF_DRIVER_ENV} must be '{DEFAULT_PLATFORM_DRIVER}' or '{DEFAULT_OPENAI_DRIVER}', got '{other}'"
                )));
            }
        };

        if settings.workers == 0 {
            return Err(PerfError::Env(format!(
                "{PERF_WORKERS_ENV} must be greater than 0"
            )));
        }

        if settings.runs == 0 {
            return Err(PerfError::Env(format!(
                "{PERF_RUNS_ENV} must be greater than 0"
            )));
        }

        if settings.max_output_tokens == 0 {
            return Err(PerfError::Env(format!(
                "{PERF_MAX_OUTPUT_TOKENS_ENV} must be greater than 0"
            )));
        }

        if settings.timeout_secs == 0 {
            return Err(PerfError::Env(format!(
                "{PERF_TIMEOUT_SECS_ENV} must be greater than 0"
            )));
        }

        if !settings.temperature.is_finite() || settings.temperature < 0.0 {
            return Err(PerfError::Env(format!(
                "{PERF_TEMPERATURE_ENV} must be a finite value greater than or equal to 0"
            )));
        }

        Ok(settings)
    }

    fn profile(&self) -> ProfileConfig {
        let (kind, provider_name) = match self.driver.as_str() {
            DEFAULT_PLATFORM_DRIVER => (ProviderKind::OpenAiChatCompletions, "bitloops"),
            DEFAULT_OPENAI_DRIVER => (ProviderKind::OpenAiChatCompletions, "openai"),
            _ => unreachable!("driver validation should already have run"),
        };

        ProfileConfig {
            task: crate::config::ProfileTask::TextGeneration,
            kind,
            provider_name: provider_name.to_owned(),
            model: self.model.clone(),
            base_url: self.base_url.clone(),
            api_key: Some(self.token.clone()),
            temperature: Some(self.temperature),
            timeout_secs: self.timeout_secs,
            max_output_tokens: Some(self.max_output_tokens),
            thinking_level: None,
            runtime_command: None,
            runtime_args: Vec::new(),
            startup_timeout_secs: 60,
        }
    }

    fn request(&self) -> InferenceRequest {
        InferenceRequest {
            system_prompt: self.system_prompt.clone(),
            user_prompt: self.prompt.clone(),
            response_mode: ResponseMode::Text,
            temperature: self.temperature,
            max_output_tokens: self.max_output_tokens,
            metadata: None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct RequestReport {
    worker_id: usize,
    request_index: usize,
    request_id: String,
    started_at_ms: f64,
    latency_ms: f64,
    success: bool,
    provider_name: Option<String>,
    model_name: Option<String>,
    finish_reason: Option<String>,
    text_length_chars: Option<usize>,
    usage: Option<TokenUsageReport>,
    error_code: Option<String>,
    error_message: Option<String>,
}

impl RequestReport {
    fn from_result(
        worker_id: usize,
        request_index: usize,
        request_id: String,
        started_at_ms: f64,
        latency_ms: f64,
        result: Result<InferenceResponse, ProviderError>,
    ) -> Self {
        match result {
            Ok(response) => Self {
                worker_id,
                request_index,
                request_id,
                started_at_ms,
                latency_ms,
                success: true,
                provider_name: Some(response.provider_name),
                model_name: Some(response.model_name),
                finish_reason: response.finish_reason,
                text_length_chars: Some(response.text.chars().count()),
                usage: response.usage.map(TokenUsageReport::from),
                error_code: None,
                error_message: None,
            },
            Err(error) => Self {
                worker_id,
                request_index,
                request_id,
                started_at_ms,
                latency_ms,
                success: false,
                provider_name: None,
                model_name: None,
                finish_reason: None,
                text_length_chars: None,
                usage: None,
                error_code: Some(error.code),
                error_message: Some(error.message),
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct AnalyticsReport {
    settings: PerfSettingsSnapshot,
    summary: SummaryReport,
    latency_ms: LatencySummary,
    worker_summaries: Vec<WorkerSummary>,
    requests: Vec<RequestReport>,
}

impl AnalyticsReport {
    fn from_reports(
        settings: PerfSettings,
        wall_time_ms: f64,
        worker_reports: Vec<WorkerReport>,
        request_reports: Vec<RequestReport>,
    ) -> Self {
        let successful_requests = request_reports
            .iter()
            .filter(|report| report.success)
            .count();
        let failed_requests = request_reports.len() - successful_requests;
        let latencies = request_reports
            .iter()
            .map(|report| report.latency_ms)
            .collect::<Vec<_>>();
        let requests_with_usage = request_reports
            .iter()
            .filter_map(|report| report.usage.as_ref())
            .collect::<Vec<_>>();
        let total_reported_tokens = requests_with_usage
            .iter()
            .map(|usage| usage.total_tokens)
            .sum::<u32>();
        let total_prompt_tokens = requests_with_usage
            .iter()
            .map(|usage| usage.prompt_tokens)
            .sum::<u32>();
        let total_completion_tokens = requests_with_usage
            .iter()
            .map(|usage| usage.completion_tokens)
            .sum::<u32>();
        let worker_summaries = worker_reports
            .into_iter()
            .map(WorkerSummary::from_worker_report)
            .collect();

        Self {
            settings: settings.snapshot(),
            summary: SummaryReport {
                requested_workers: settings.workers,
                effective_workers: cmp::min(settings.workers, settings.runs),
                total_requests: request_reports.len(),
                successful_requests,
                failed_requests,
                wall_time_ms,
                throughput_requests_per_second: rate_per_second(
                    successful_requests as f64,
                    wall_time_ms,
                ),
                requests_with_usage: requests_with_usage.len(),
                total_reported_tokens,
                total_prompt_tokens,
                total_completion_tokens,
                token_throughput_per_second: if total_reported_tokens == 0 {
                    None
                } else {
                    Some(rate_per_second(total_reported_tokens as f64, wall_time_ms))
                },
            },
            latency_ms: LatencySummary::from_samples(&latencies),
            worker_summaries,
            requests: request_reports,
        }
    }
}

#[derive(Debug, Serialize)]
struct PerfSettingsSnapshot {
    driver: String,
    base_url: String,
    model: String,
    prompt: String,
    system_prompt: String,
    runs: usize,
    workers: usize,
    timeout_secs: u64,
    temperature: f32,
    max_output_tokens: u32,
}

impl PerfSettings {
    fn snapshot(&self) -> PerfSettingsSnapshot {
        PerfSettingsSnapshot {
            driver: self.driver.clone(),
            base_url: self.base_url.clone(),
            model: self.model.clone(),
            prompt: self.prompt.clone(),
            system_prompt: self.system_prompt.clone(),
            runs: self.runs,
            workers: self.workers,
            timeout_secs: self.timeout_secs,
            temperature: self.temperature,
            max_output_tokens: self.max_output_tokens,
        }
    }
}

#[derive(Debug, Serialize)]
struct SummaryReport {
    requested_workers: usize,
    effective_workers: usize,
    total_requests: usize,
    successful_requests: usize,
    failed_requests: usize,
    wall_time_ms: f64,
    throughput_requests_per_second: f64,
    requests_with_usage: usize,
    total_reported_tokens: u32,
    total_prompt_tokens: u32,
    total_completion_tokens: u32,
    token_throughput_per_second: Option<f64>,
}

#[derive(Debug, Serialize)]
struct WorkerSummary {
    worker_id: usize,
    assigned_runs: usize,
    successful_requests: usize,
    failed_requests: usize,
    wall_time_ms: f64,
    latency_ms: LatencySummary,
}

impl WorkerSummary {
    fn from_worker_report(report: WorkerReport) -> Self {
        let successful_requests = report
            .requests
            .iter()
            .filter(|request| request.success)
            .count();
        let failed_requests = report.requests.len() - successful_requests;
        let latencies = report
            .requests
            .iter()
            .map(|request| request.latency_ms)
            .collect::<Vec<_>>();

        Self {
            worker_id: report.worker_id,
            assigned_runs: report.assigned_runs,
            successful_requests,
            failed_requests,
            wall_time_ms: report.wall_time_ms,
            latency_ms: LatencySummary::from_samples(&latencies),
        }
    }
}

#[derive(Clone, Debug)]
struct WorkerReport {
    worker_id: usize,
    assigned_runs: usize,
    wall_time_ms: f64,
    requests: Vec<RequestReport>,
}

#[derive(Clone, Debug, Serialize)]
struct TokenUsageReport {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl From<TokenUsage> for TokenUsageReport {
    fn from(value: TokenUsage) -> Self {
        Self {
            prompt_tokens: value.prompt_tokens,
            completion_tokens: value.completion_tokens,
            total_tokens: value.total_tokens,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct LatencySummary {
    count: usize,
    min: f64,
    max: f64,
    mean: f64,
    median: f64,
    p95: f64,
    p99: f64,
}

impl LatencySummary {
    fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                count: 0,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                median: 0.0,
                p95: 0.0,
                p99: 0.0,
            };
        }

        let mut sorted = samples.to_vec();
        sorted.sort_by(|left, right| left.partial_cmp(right).expect("latency should be finite"));
        let sum = sorted.iter().sum::<f64>();

        Self {
            count: sorted.len(),
            min: sorted[0],
            max: *sorted
                .last()
                .expect("latency summary should have a last item"),
            mean: sum / sorted.len() as f64,
            median: percentile(&sorted, 50.0),
            p95: percentile(&sorted, 95.0),
            p99: percentile(&sorted, 99.0),
        }
    }
}

fn percentile(sorted_samples: &[f64], percentile: f64) -> f64 {
    if sorted_samples.is_empty() {
        return 0.0;
    }

    if sorted_samples.len() == 1 {
        return sorted_samples[0];
    }

    let rank = (percentile / 100.0) * (sorted_samples.len() - 1) as f64;
    let lower_index = rank.floor() as usize;
    let upper_index = rank.ceil() as usize;
    let fraction = rank - lower_index as f64;
    let lower = sorted_samples[lower_index];
    let upper = sorted_samples[upper_index];

    lower + ((upper - lower) * fraction)
}

fn rate_per_second(count: f64, wall_time_ms: f64) -> f64 {
    if wall_time_ms <= 0.0 {
        return 0.0;
    }

    count / (wall_time_ms / 1_000.0)
}

fn required_env(name: &str) -> Result<String, PerfError> {
    let value = std::env::var(name).map_err(|_| {
        PerfError::Env(format!(
            "{name} must be set for the ad hoc performance runner"
        ))
    })?;

    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(PerfError::Env(format!("{name} must not be empty")));
    }

    Ok(trimmed.to_owned())
}

fn optional_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn parse_required_env<T>(name: &str) -> Result<T, PerfError>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    required_env(name)?
        .parse::<T>()
        .map_err(|error| PerfError::Env(format!("failed to parse {name}: {error}")))
}

fn parse_env_with_default<T>(name: &str, default: T) -> Result<T, PerfError>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    match optional_env(name) {
        Some(value) => value
            .parse::<T>()
            .map_err(|error| PerfError::Env(format!("failed to parse {name}: {error}"))),
        None => Ok(default),
    }
}

#[derive(Debug, Error)]
pub enum PerfError {
    #[error("{0}")]
    Env(String),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("worker thread panicked")]
    WorkerPanic,
    #[error("{0}")]
    Worker(String),
}

#[derive(Debug, Error)]
enum WorkerError {
    #[error("failed to generate a random cache-buster: {0}")]
    Random(String),
    #[error(transparent)]
    Provider(#[from] ProviderError),
}
