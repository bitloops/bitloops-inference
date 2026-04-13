use std::process::ExitCode;

fn main() -> ExitCode {
    match bitloops_inference::run_from_args(std::env::args_os()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
