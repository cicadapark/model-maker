//! `model-maker`: the launcher's own commands (check, setup); every other word goes to the Python side (src/run.rs).

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use model_maker_launcher::{check, platform, repository, run, setup};

#[derive(Parser)]
#[command(
    name = "model-maker",
    about = "Make a checked 3-D object from a prompt: set this PC up, check it, run it.",
    after_help = "Every other command (make, batch, finish, repair-texture, inspect, words, ...) is model-maker's \
                  own: `model-maker make --help` shows its options.",
    disable_version_flag = true,
    allow_external_subcommands = true
)]
struct Arguments {
    /// The launcher's version, the repository's version and the lock it installs here.
    #[arg(long)]
    version: bool,
    #[command(subcommand)]
    command: Option<Step>,
}

#[derive(Args)]
struct Where {
    /// Another of this system's locks (e.g. linux-x86_64-cpu for a PC without a card).
    #[arg(long)]
    lock: Option<String>,
    /// The models' folder (default MODEL_MAKER_MODELS, else models/ in the repository).
    #[arg(long)]
    models: Option<PathBuf>,
    /// The view painter's code (default MODEL_MAKER_SOURCES, else sources/ in the repository).
    #[arg(long)]
    sources: Option<PathBuf>,
    /// For tests on a PC without a card: do not require one.
    #[arg(long)]
    processor_only: bool,
}

#[derive(Subcommand)]
enum Step {
    /// Check this PC against what model-maker needs (nothing is changed).
    Check {
        #[command(flatten)]
        place: Where,
    },
    /// Set this PC up: uv, Python, the locked environment, the certificate, the models (each licence first).
    Setup {
        #[command(flatten)]
        place: Where,
        /// Only check this PC (as `model-maker check`).
        #[arg(long)]
        check_only: bool,
        /// Accept every model's licence without asking (each is still shown).
        #[arg(long, alias = "accept-licences")]
        yes: bool,
        /// A Hugging Face access token's file, for FLUX.1 [schnell] (docs/setup.md); sent to the hub only.
        #[arg(long)]
        hf_token_file: Option<PathBuf>,
        /// Everything but the models.
        #[arg(long)]
        skip_models: bool,
        /// Not the self-test at the end.
        #[arg(long)]
        skip_tests: bool,
        /// Not the optional question judge (llama.cpp's server and its 7 GB model).
        #[arg(long)]
        without_question_judge: bool,
        /// Windows: fetch and start Microsoft's Visual C++ runtime if it is missing, without asking.
        #[arg(long)]
        install_cxx_runtime: bool,
        /// The question judge too (the default).
        #[arg(long, conflicts_with = "without_question_judge")]
        with_question_judge: bool,
        /// For a PC without internet: the lock's wheels from this folder or http(s) address only, never the package
        /// index (`tools/ci/locks.py wheels` makes one). Without it, MODEL_MAKER_WHEELS is used when set.
        #[arg(long, value_name = "FOLDER_OR_URL")]
        wheels: Option<String>,
    },
    /// Every other command goes to the Python side unchanged.
    #[command(external_subcommand)]
    Python(Vec<OsString>),
}

/// `--wheels`' stand-in for a caller that passes no options, such as a window's Set up on a PC without internet (the
/// Windows card test's phase 2, 28 Nov 2026: the companion's setup stopped at `uv pip sync` on a PC without internet).
const WHEELS_VARIABLE: &str = "MODEL_MAKER_WHEELS";

/// `--wheels` as given, else the variable's value when it is set and not empty.
fn wheels_chosen(given: Option<String>, variable: Option<OsString>) -> Option<String> {
    given.or_else(|| {
        variable
            .and_then(|value| value.into_string().ok())
            .filter(|value| !value.trim().is_empty())
    })
}

fn version_words(repository: &std::path::Path) -> String {
    let (system, processor) = platform::here();
    let lock = platform::target(system, processor)
        .and_then(|t| platform::chosen_lock(t, None).ok())
        .map_or("none for this PC", |l| l.name);
    format!(
        "model-maker {} (repository {}; lock {lock}; unsigned build)",
        env!("CARGO_PKG_VERSION"),
        repository::version(repository)
    )
}

fn checked(repository: &std::path::Path, place: &Where) -> ExitCode {
    let models = place
        .models
        .clone()
        .or_else(|| std::env::var_os("MODEL_MAKER_MODELS").map(PathBuf::from))
        .unwrap_or_else(|| repository.join("models"));
    let (system, processor) = platform::here();
    let target = platform::target(system, processor);
    let chosen = match target.map(|t| platform::chosen_lock(t, place.lock.as_deref())) {
        Some(Err(refusal)) if place.lock.is_some() => {
            eprintln!("model-maker check: {refusal}");
            return ExitCode::from(2);
        }
        Some(Ok(chosen)) => Some(chosen),
        _ => None,
    };
    let needs_card = !(place.processor_only || chosen.is_some_and(|l| l.name.ends_with("-cpu")));
    println!("model-maker: checking this PC (nothing is changed)");
    let rows = check::rows(
        &check::gather(repository, &models),
        target,
        chosen,
        needs_card,
    );
    print!("{}", check::table(&rows));
    match check::missing_count(&rows) {
        0 => {
            println!("Everything needed is here, or `model-maker setup` installs it.");
            ExitCode::SUCCESS
        }
        n => {
            println!("{n} thing(s) missing; see MISSING above.");
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    if let Err(why) = run::own_children() {
        eprintln!("model-maker: programs it starts may outlive it if it is stopped ({why})");
    }
    let arguments = Arguments::parse();
    let repository = match repository::of_this_program() {
        Ok(found) => found,
        Err(not_beside) => {
            eprintln!("model-maker: {not_beside}");
            return ExitCode::from(2);
        }
    };
    if arguments.version {
        println!("{}", version_words(&repository));
        return ExitCode::SUCCESS;
    }
    match arguments.command {
        Some(Step::Check { place }) => checked(&repository, &place),
        Some(Step::Setup {
            place,
            check_only: true,
            ..
        }) => checked(&repository, &place),
        Some(Step::Setup {
            place,
            yes,
            hf_token_file,
            skip_models,
            skip_tests,
            without_question_judge,
            wheels,
            install_cxx_runtime,
            ..
        }) => {
            let options = setup::Options {
                yes,
                hf_token_file,
                skip_models,
                skip_tests,
                without_question_judge,
                lock: place.lock,
                models: place.models,
                sources: place.sources,
                processor_only: place.processor_only,
                wheels: wheels_chosen(wheels, std::env::var_os(WHEELS_VARIABLE)),
                install_runtime: install_cxx_runtime,
            };
            match setup::run(&repository, &options) {
                Ok(()) => ExitCode::SUCCESS,
                Err(stopped) => {
                    eprintln!("setup: {stopped}");
                    ExitCode::FAILURE
                }
            }
        }
        Some(Step::Python(words)) => {
            let (system, processor) = platform::here();
            let Some(target) = platform::target(system, processor) else {
                eprintln!("model-maker: no pins for this system");
                return ExitCode::from(2);
            };
            std::process::exit(run::passed_on(&repository, system, target, &words));
        }
        None => {
            eprintln!("model-maker: name a command (model-maker --help)");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_takes_a_wheel_folder_or_address() {
        let parsed = Arguments::try_parse_from([
            "model-maker",
            "setup",
            "--skip-tests",
            "--wheels",
            "http://127.0.0.1:8765/wheels/",
        ])
        .unwrap();
        match parsed.command {
            Some(Step::Setup { wheels, .. }) => {
                assert_eq!(wheels.as_deref(), Some("http://127.0.0.1:8765/wheels/"))
            }
            _ => panic!("not setup"),
        }
        let plain = Arguments::try_parse_from(["model-maker", "setup"]).unwrap();
        assert!(matches!(
            plain.command,
            Some(Step::Setup { wheels: None, .. })
        ));
    }

    #[test]
    fn without_the_option_the_wheels_variable_is_used() {
        let given = Some("C:\\wheels".to_string());
        let variable = Some(OsString::from("http://127.0.0.1:8765/wheels/"));
        assert_eq!(wheels_chosen(given.clone(), variable.clone()), given);
        assert_eq!(
            wheels_chosen(None, variable).as_deref(),
            Some("http://127.0.0.1:8765/wheels/")
        );
        assert_eq!(wheels_chosen(None, Some(OsString::from(" "))), None);
        assert_eq!(wheels_chosen(None, None), None);
    }
}
