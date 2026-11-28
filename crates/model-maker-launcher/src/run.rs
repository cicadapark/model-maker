//! Every word that is not the launcher's own goes to the Python side, as the bash `model-maker` sends it:
//! `<python> -m model_maker <words...>` with the bash command's environment.
//!
//! On Linux and macOS the launcher becomes Python (exec: the same process, its signals and its exit code). On
//! Windows it cannot, so it waits for Python and returns its exit code; it ignores Ctrl+C itself (Python, in the same
//! console, gets it and ends its own way), and it put itself into a job object at its start (`own_children`), so
//! when it ends, however it ends, Windows ends Python and everything Python started.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::environment;
use crate::platform::{System, Target};
use crate::tools;

/// The Python that runs the pipeline: MODEL_MAKER_PYTHON if set (as the bash command), else the environment's.
pub fn python(repository: &Path, target: &Target, inherited: &[(OsString, OsString)]) -> PathBuf {
    inherited
        .iter()
        .find(|(name, _)| name == "MODEL_MAKER_PYTHON")
        .map(|(_, value)| PathBuf::from(value))
        .unwrap_or_else(|| repository.join(".venv").join(target.venv_python))
}

/// The command for these words, with the Python side's environment.
pub fn command(
    repository: &Path,
    system: System,
    target: &Target,
    words: &[OsString],
) -> Result<Command, String> {
    let inherited: Vec<(OsString, OsString)> = std::env::vars_os().collect();
    let python = python(repository, target, &inherited);
    if !python.is_file() {
        return Err(format!(
            "no Python environment at {} (run `model-maker setup`, or set MODEL_MAKER_PYTHON)",
            python.display()
        ));
    }
    let mut command = Command::new(&python);
    command.args(["-m", "model_maker"]).args(words);
    let models = repository.join("models");
    let sources = repository.join("sources");
    environment::for_python(system, repository, &models, &sources, inherited.clone())
        .apply(&mut command);
    let tool = tools::simplify_tool(repository, system, tools::program_folder().as_deref());
    if let Some((name, value)) = environment::simplify_setting(tool.as_deref(), &inherited) {
        command.env(name, value);
    }
    Ok(command)
}

/// Runs the words; returns only if Python could not be started (Unix) or has ended (Windows).
pub fn passed_on(repository: &Path, system: System, target: &Target, words: &[OsString]) -> i32 {
    let mut command = match command(repository, system, target, words) {
        Ok(command) => command,
        Err(why) => {
            eprintln!("model-maker: {why}");
            return 1;
        }
    };
    run(&mut command)
}

#[cfg(unix)]
fn run(command: &mut Command) -> i32 {
    use std::os::unix::process::CommandExt;
    let error = command.exec();
    eprintln!("model-maker: could not start Python: {error}");
    127
}

#[cfg(windows)]
fn run(command: &mut Command) -> i32 {
    // Ctrl+C reaches every program in the console: Python ends its own way; the launcher waits for it.
    let _ = ctrlc::set_handler(|| {});
    match command.status() {
        Ok(status) => status.code().unwrap_or(1),
        Err(error) => {
            eprintln!("model-maker: could not start Python: {error}");
            127
        }
    }
}

/// Windows: the launcher puts itself into a new job object that ends all its members when its last handle closes,
/// before it starts anything. Every program it starts, and every program those start, is born inside the job, so
/// there is nothing to race; the handle is not inherited, and it is kept (never closed by us) until the launcher
/// ends, when Windows closes it and ends them all. Elsewhere nothing is needed: on Linux and macOS the launcher
/// becomes Python, and setup's children share its terminal's process group.
#[cfg(windows)]
pub fn own_children() -> Result<(), String> {
    let mut limits = win32job::ExtendedLimitInfo::new();
    limits.limit_kill_on_job_close();
    let job =
        win32job::Job::create_with_limit_info(&limits).map_err(|e| format!("job object: {e}"))?;
    job.assign_current_process()
        .map_err(|e| format!("job object: {e}"))?;
    // Not closed here: closing the last handle would end the launcher itself (it is in the job).
    let _kept_until_exit = job.into_handle();
    Ok(())
}

#[cfg(not(windows))]
pub fn own_children() -> Result<(), String> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::{target, Processor};

    #[test]
    fn model_maker_python_wins_as_in_the_bash_command() {
        let linux = target(System::Linux, Processor::X86_64).unwrap();
        let repository = Path::new("/r");
        assert_eq!(
            python(repository, linux, &[]),
            PathBuf::from("/r/.venv/bin/python")
        );
        let named = vec![(
            OsString::from("MODEL_MAKER_PYTHON"),
            OsString::from("/other/python"),
        )];
        assert_eq!(
            python(repository, linux, &named),
            PathBuf::from("/other/python")
        );
        let windows = target(System::Windows, Processor::X86_64).unwrap();
        assert!(python(repository, windows, &[]).ends_with("Scripts/python.exe"));
    }

    #[test]
    fn a_missing_environment_is_said_plainly() {
        let linux = target(System::Linux, Processor::X86_64).unwrap();
        let nowhere =
            std::env::temp_dir().join(format!("mm-launcher-no-repo-{}", std::process::id()));
        if std::env::var_os("MODEL_MAKER_PYTHON").is_none() {
            let refused = command(&nowhere, System::Linux, linux, &[]).unwrap_err();
            assert!(refused.contains("model-maker setup"), "{refused}");
        }
    }

    #[test]
    fn the_words_follow_dash_m_model_maker_unchanged() {
        let folder = std::env::temp_dir().join(format!("mm-launcher-words-{}", std::process::id()));
        let python = folder.join(".venv/bin/python");
        std::fs::create_dir_all(python.parent().unwrap()).unwrap();
        std::fs::write(&python, b"").unwrap();
        let linux = target(System::Linux, Processor::X86_64).unwrap();
        let words: Vec<OsString> = [
            "make",
            "an old looking chair",
            "--size",
            "0.9",
            "--out",
            "a dir/with \"quotes\"",
        ]
        .iter()
        .map(OsString::from)
        .collect();
        if std::env::var_os("MODEL_MAKER_PYTHON").is_none() {
            let built = command(&folder, System::Linux, linux, &words).unwrap();
            let args: Vec<&std::ffi::OsStr> = built.get_args().collect();
            assert_eq!(args[..2], ["-m", "model_maker"]);
            assert_eq!(
                args[2..]
                    .iter()
                    .map(|a| a.to_os_string())
                    .collect::<Vec<_>>(),
                words
            );
        }
        std::fs::remove_dir_all(&folder).unwrap();
    }
}
