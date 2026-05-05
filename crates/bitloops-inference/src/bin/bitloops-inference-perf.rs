use std::process::ExitCode;

fn main() -> ExitCode {
    match bitloops_inference::run_perf_report_from_env() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
