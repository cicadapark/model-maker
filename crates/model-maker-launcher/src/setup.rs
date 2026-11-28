//! `model-maker setup`: what setup.sh does, the same way on every system, everything inside the repository's folder.
//! A second run checks what is there and carries on where the first stopped.

use std::ffi::OsString;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::check::{self, Verdict};
use crate::platform::{self, Lock, System, Target};
use crate::{download, environment, interpreter, runtime, tools};

/// What the person asked for (setup.sh's options).
#[derive(Debug, Default, Clone)]
pub struct Options {
    /// Accept every model's licence without asking (each is still shown).
    pub yes: bool,
    pub hf_token_file: Option<PathBuf>,
    pub skip_models: bool,
    pub skip_tests: bool,
    pub without_question_judge: bool,
    pub lock: Option<String>,
    pub models: Option<PathBuf>,
    pub sources: Option<PathBuf>,
    /// For tests on a PC without a card (CI, a test PC): the card is not required.
    pub processor_only: bool,
    /// The lock's wheels from this folder or address only, never the package index (a PC without internet).
    pub wheels: Option<String>,
    /// Windows: fetch and start Microsoft's C++ runtime without asking (the companion asked first, in its window).
    pub install_runtime: bool,
}

/// Where setup stopped, in words for the person.
#[derive(Debug)]
pub struct Stopped(pub String);

impl std::fmt::Display for Stopped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn stopped(what: impl std::fmt::Display) -> Stopped {
    Stopped(what.to_string())
}

/// The folders setup works in.
pub struct Place {
    pub repository: PathBuf,
    pub tools: PathBuf,
    pub venv: PathBuf,
    pub models: PathBuf,
    pub sources: PathBuf,
}

impl Place {
    pub fn new(repository: &Path, models: Option<PathBuf>, sources: Option<PathBuf>) -> Place {
        let from_env = |name: &str| std::env::var_os(name).map(PathBuf::from);
        Place {
            repository: repository.to_path_buf(),
            tools: repository.join(".tools"),
            venv: repository.join(".venv"),
            models: models
                .or_else(|| from_env("MODEL_MAKER_MODELS"))
                .unwrap_or_else(|| repository.join("models")),
            sources: sources
                .or_else(|| from_env("MODEL_MAKER_SOURCES"))
                .unwrap_or_else(|| repository.join("sources")),
        }
    }

    pub fn venv_python(&self, target: &Target) -> PathBuf {
        self.venv.join(target.venv_python)
    }
}

/// The setting that has setup also write its steps to a file, one JSON line each (the companion's bars read them;
/// nothing reads the printed lines): `{"schema":1,"event":"began","step":"python"}`, then `{"schema":1,"event":"done"}`
/// or `{"schema":1,"event":"stopped","why":".."}`. Unset, nothing is written.
pub const PROGRESS_SETTING: &str = "MODEL_MAKER_PROGRESS";
pub const PROGRESS_SCHEMA: u32 = 1;

/// Setup's steps in order: the id written to the progress file, and the title printed.
pub const STEPS: &[(&str, &str)] = &[
    ("runtime", "The Visual C++ runtime (Microsoft's)"),
    ("python", "uv and Python"),
    (
        "tools",
        "model-maker-simplify and the question judge's server (optional)",
    ),
    ("models", "The models and their code (each licence first)"),
    ("self-test", "Self-test"),
];

/// `text` as a JSON string, quotes included.
fn json_string(text: &str) -> String {
    let mut quoted = String::with_capacity(text.len() + 2);
    quoted.push('"');
    for c in text.chars() {
        match c {
            '"' => quoted.push_str("\\\""),
            '\\' => quoted.push_str("\\\\"),
            '\n' => quoted.push_str("\\n"),
            c if (c as u32) < 0x20 => quoted.push_str(&format!("\\u{:04x}", c as u32)),
            c => quoted.push(c),
        }
    }
    quoted.push('"');
    quoted
}

/// One line of the progress file: `began` (with its step), `done`, or `stopped` (with why).
pub fn progress_line(event: &str, field: Option<(&str, &str)>) -> String {
    let extra = field
        .map(|(name, value)| format!(",{}:{}", json_string(name), json_string(value)))
        .unwrap_or_default();
    format!(
        "{{\"schema\":{PROGRESS_SCHEMA},\"event\":{}{extra}}}\n",
        json_string(event)
    )
}

fn progress(line: &str) {
    let Some(path) = std::env::var_os(PROGRESS_SETTING) else {
        return;
    };
    use std::io::Write;
    let written = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut file| file.write_all(line.as_bytes()));
    if let Err(error) = written {
        eprintln!("setup: the progress file could not be written ({error})");
    }
}

/// Setup's own record that it ended well, in the tools folder: removed when a setup begins, written at its end, so
/// a stopped setup never leaves an old one. The companion's "ready" is this record and the check, nothing of its own.
pub const DONE_FILE: &str = "setup-done.json";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Done {
    No,
    /// Ended well with --skip-models: a make needs the models still.
    WithoutModels,
    WithModels,
}

pub fn done_line(with_models: bool) -> String {
    format!("{{\"schema\":1,\"models\":{with_models}}}\n")
}

/// What the record in `tools` says.
pub fn done_record(tools: &Path) -> Done {
    let Ok(text) = std::fs::read_to_string(tools.join(DONE_FILE)) else {
        return Done::No;
    };
    if text == done_line(true) {
        Done::WithModels
    } else if text == done_line(false) {
        Done::WithoutModels
    } else {
        Done::No
    }
}

fn step(id: &str) {
    let title = STEPS
        .iter()
        .find(|(known, _)| *known == id)
        .map_or(id, |(_, title)| title);
    println!("\n== {title}");
    progress(&progress_line("began", Some(("step", id))));
}

/// A child program run to its end; its own output goes to the terminal.
fn ran(command: &mut Command, what: &str) -> Result<(), Stopped> {
    let status = command
        .status()
        .map_err(|e| stopped(format!("{what}: could not start it ({e})")))?;
    if status.success() {
        Ok(())
    } else {
        Err(stopped(format!("{what}: it failed ({status})")))
    }
}

fn uv_command(place: &Place, uv: &Path) -> Command {
    let mut command = Command::new(uv);
    environment::for_uv(&place.tools, std::env::vars_os()).apply(&mut command);
    command
}

fn python_command(place: &Place, target: &Target, system: System) -> Command {
    let mut command = Command::new(place.venv_python(target));
    let inherited: Vec<_> = std::env::vars_os().collect();
    environment::for_python(
        system,
        &place.repository,
        &place.models,
        &place.sources,
        inherited.clone(),
    )
    .apply(&mut command);
    let tool = tools::simplify_tool(
        &place.repository,
        system,
        tools::program_folder().as_deref(),
    );
    if let Some((name, value)) = environment::simplify_setting(tool.as_deref(), &inherited) {
        command.env(name, value);
    }
    command
}

/// model-maker-simplify: the prebuilt one, else built here (needs Rust and a C++ compiler).
fn ready_simplify(place: &Place, system: System) -> Result<(), Stopped> {
    if let Some(tool) = tools::simplify_tool(
        &place.repository,
        system,
        tools::program_folder().as_deref(),
    ) {
        println!("model-maker-simplify: {}", tool.display());
        return Ok(());
    }
    let started = Instant::now();
    tools::built_simplify(&place.repository).map_err(stopped)?;
    println!(
        "model-maker-simplify: built in {} s (target/release)",
        started.elapsed().as_secs()
    );
    Ok(())
}

/// The optional question judge's server: installed, kept, or removed (`--without-question-judge`).
fn ready_judge(place: &Place, target: &Target, options: &Options) {
    if options.without_question_judge {
        if let Some(said) = tools::removed_judge(&place.tools) {
            println!("{said}");
        }
    } else if !options.skip_models {
        match tools::installed_judge(&place.tools, target) {
            Ok(said) | Err(tools::JudgeOff(said)) => println!("{said}"),
        }
    }
}

fn version_of(program: &Path, args: &[&str]) -> Option<String> {
    let done = Command::new(program).args(args).output().ok()?;
    done.status
        .success()
        .then(|| String::from_utf8_lossy(&done.stdout).trim().to_string())
}

/// uv, its pinned release, its sum checked, in `.tools/uv`.
fn installed_uv(place: &Place, target: &Target, system: System) -> Result<PathBuf, Stopped> {
    let folder = place.tools.join("uv");
    let uv = folder.join(platform::program(system, "uv"));
    if version_of(&uv, &["--version"])
        .is_some_and(|v| v.split_whitespace().nth(1) == Some(platform::UV_VERSION))
    {
        println!("uv {}: already in .tools/uv", platform::UV_VERSION);
        return Ok(uv);
    }
    let archive = place.tools.join(if system == System::Windows {
        "uv.zip"
    } else {
        "uv.tar.gz"
    });
    download::fetched(&target.uv, &archive).map_err(stopped)?;
    let _ = std::fs::remove_dir_all(&folder);
    let strip = usize::from(target.uv.archive == platform::Archive::TarGz);
    download::unpacked(&archive, target.uv.archive, &folder, strip).map_err(stopped)?;
    let _ = std::fs::remove_file(&archive);
    println!(
        "uv {} (MIT or Apache-2.0): installed in .tools/uv",
        platform::UV_VERSION
    );
    Ok(uv)
}

/// The interpreter (uv's, checked against the publisher's sums, its copyleft module removed) and `.venv`.
fn installed_python(
    place: &Place,
    target: &Target,
    system: System,
    uv: &Path,
) -> Result<(), Stopped> {
    let wanted = platform::PYTHON_VERSION;
    let venv_python = place.venv_python(target);
    let version = "import platform; print(platform.python_version())";
    if version_of(&venv_python, &["-I", "-c", version]).as_deref() == Some(wanted) {
        println!("Python {wanted}: .venv already there");
    } else {
        let _ = std::fs::remove_dir_all(&place.venv);
        ran(
            uv_command(place, uv).args(["python", "install", "--quiet", wanted]),
            "uv python install",
        )?;
        ran(
            uv_command(place, uv)
                .args(["venv", "--quiet", "--python", wanted])
                .arg(&place.venv),
            "uv venv",
        )?;
        println!("Python {wanted}: .venv made");
    }
    if target.interpreter_sums.is_empty() {
        return Err(stopped(format!(
            "no sums of the publisher's interpreter (build {}) are known for this system; refusing to go on",
            platform::PYTHON_BUILD
        )));
    }
    let home = interpreter::home(&place.tools, target);
    let unequal = interpreter::unequal_files(&home, target);
    if !unequal.is_empty() {
        return Err(stopped(format!(
            "the interpreter uv installed is not the publisher's build {} ({} differ); refusing it. Delete \
             .tools/python and run setup again",
            platform::PYTHON_BUILD,
            unequal.join(", ")
        )));
    }
    let removed = interpreter::refused_modules_removed(&home, system)
        .map_err(|e| stopped(format!("{}: {e}", home.display())))?;
    if !removed.is_empty() {
        println!(
            "Python: removed {} (Berkeley DB, a copyleft library; nothing here uses it)",
            removed.join(", ")
        );
    }
    Ok(())
}

/// `uv pip sync`'s words: exactly the lock, every wheel by one of its sums; with `wheels` (a folder or an address,
/// made by `tools/ci/locks.py wheels`), from there alone. A package pinned by its address (PyTorch's builds) still
/// comes from that address, and needs nothing when installed from it already.
fn sync_arguments(venv_python: &Path, lock_file: &Path, wheels: Option<&str>) -> Vec<OsString> {
    let mut words: Vec<OsString> = ["pip", "sync", "--quiet", "--python"]
        .map(OsString::from)
        .into();
    words.push(venv_python.into());
    words.extend(["--require-hashes", "--only-binary", ":all:"].map(OsString::from));
    if let Some(source) = wheels {
        words.extend(["--no-index", "--find-links", source].map(OsString::from));
    }
    words.push(lock_file.into());
    words
}

/// The lock as `--wheels` gives it to uv (in .tools): see `pinned_by_version`.
const WHEELS_LOCK: &str = "lock-for-wheels.txt";

/// The lock with each package pinned by its address (`torch @ https://.../torch-2.10.0%2Bcu128-...whl#sha256=...`)
/// pinned by that wheel's version instead (`torch==2.10.0+cu128`), its sums kept, so uv takes it from the wheel
/// folder too: uv takes an address pin only from that address or its own cache (offline, with the cache gone, it
/// refused one already installed; tried 28 Nov 2026). Every other line as it is.
fn pinned_by_version(lock: &str) -> String {
    let mut out = String::with_capacity(lock.len());
    for line in lock.split_inclusive('\n') {
        out.push_str(&version_pin(line).unwrap_or_else(|| line.to_string()));
    }
    out
}

fn version_pin(line: &str) -> Option<String> {
    if line.starts_with(char::is_whitespace) || line.starts_with('#') {
        return None;
    }
    let (name, rest) = line.split_once(" @ ")?;
    let url_end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    let file = percent_decoded(rest[..url_end].split('#').next()?.rsplit('/').next()?)?;
    let version = file.strip_suffix(".whl")?.split('-').nth(1)?;
    Some(format!("{name}=={version}{}", &rest[url_end..]))
}

fn percent_decoded(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut at = 0;
    while at < bytes.len() {
        if bytes[at] == b'%' {
            let hex = std::str::from_utf8(bytes.get(at + 1..at + 3)?).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            at += 3;
        } else {
            out.push(bytes[at]);
            at += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// `--wheels` as given: an http(s) address as it is, else a folder that must be there (as an absolute path, so uv
/// reads the same folder whatever its own working folder).
fn wheels_source(given: Option<&str>) -> Result<Option<String>, Stopped> {
    let Some(given) = given else {
        return Ok(None);
    };
    if given.starts_with("http://") || given.starts_with("https://") {
        return Ok(Some(given.to_string()));
    }
    let folder = Path::new(given);
    if !folder.is_dir() {
        return Err(stopped(format!(
            "--wheels {given}: no such folder (tools/ci/locks.py wheels makes one)"
        )));
    }
    Ok(Some(
        crate::paths::real(folder).to_string_lossy().into_owned(),
    ))
}

/// Exactly the lock: what is not in it is removed, and every wheel must match one of its sums.
fn installed_packages(
    place: &Place,
    target: &Target,
    lock: &Lock,
    uv: &Path,
    wheels: Option<&str>,
) -> Result<(), Stopped> {
    let started = Instant::now();
    let venv_python = place.venv_python(target);
    let lock_file = place.repository.join(lock.file);
    let wheels = wheels_source(wheels)?;
    let synced = match &wheels {
        None => lock_file.clone(),
        Some(source) => {
            println!("Python packages: the lock's wheels from {source} only (no package index)");
            let text = std::fs::read_to_string(&lock_file)
                .map_err(|e| stopped(format!("{}: {e}", lock_file.display())))?;
            let copy = place.tools.join(WHEELS_LOCK);
            std::fs::write(&copy, pinned_by_version(&text))
                .map_err(|e| stopped(format!("{}: {e}", copy.display())))?;
            copy
        }
    };
    ran(
        uv_command(place, uv).args(sync_arguments(&venv_python, &synced, wheels.as_deref())),
        "uv pip sync",
    )?;
    ran(
        uv_command(place, uv)
            .args(["pip", "check", "--quiet", "--python"])
            .arg(&venv_python),
        "uv pip check",
    )?;
    println!(
        "Python packages: {} installed in {} s",
        lock.file,
        started.elapsed().as_secs()
    );
    ran(
        Command::new(&venv_python)
            .arg("-I")
            .arg(
                place
                    .repository
                    .join("python")
                    .join("model_maker")
                    .join("check_licences.py"),
            )
            .current_dir(&place.repository),
        "the licence check",
    )
}

/// The test certificate that signs the pictures' content credentials (unless the person gives their own).
fn made_certificate(place: &Place, target: &Target, system: System) -> Result<(), Stopped> {
    ran(
        python_command(place, target, system).args([
            "-m",
            "model_maker",
            "credentials",
            "--make-test",
        ]),
        "the test certificate",
    )
}

/// The models, each licence first (setup/fetch_models.py); the access token goes to it alone, by its file name.
fn fetched_models(place: &Place, target: &Target, options: &Options) -> Result<(), Stopped> {
    let mut command = Command::new(place.venv_python(target));
    command
        .arg("-I")
        .arg(place.repository.join("setup").join("fetch_models.py"));
    command
        .arg("--models")
        .arg(&place.models)
        .arg("--sources")
        .arg(&place.sources);
    if options.yes {
        command.arg("--yes");
    }
    if options.without_question_judge {
        command.args(["--without", "question-judge"]);
    }
    if let Some(token_file) = &options.hf_token_file {
        command.arg("--hf-token-file").arg(token_file);
    }
    ran(&mut command, "the models")
}

/// Test modules run at once in the self-test (setup.sh's SELF_TEST_WORKERS). In one process, with nothing shown,
/// the unit tests had not ended after 19 minutes on 8 processors (the Windows test of 26 November 2026).
const SELF_TEST_WORKERS: usize = 4;

/// The unit tests as setup.sh runs them: each module in a process of its own (tools/ci/run_tests.py), each one's
/// verdict printed as it ends, the whole report in `log`.
fn unit_tests(mut python: Command, repository: &Path, log: &Path) -> Command {
    python
        .arg("-I")
        .arg(repository.join("tools").join("ci").join("run_tests.py"))
        .args(["--workers", &SELF_TEST_WORKERS.to_string(), "--log"])
        .arg(log)
        .current_dir(repository);
    python
}

/// The device line of the self-test: on a card's setup, the card PyTorch sees (setup.sh's line, which asserts it);
/// on a processor-only one, the device the pipeline will actually use (model_maker.device's own choice).
const DEVICE_SEEN: &str = r#"import sys
sys.path.insert(0, sys.argv[1])
import torch
from model_maker import device
found = device.pick(torch)
assert sys.argv[2] != "card" or found.backend != "cpu", "PyTorch sees no card"
if found.backend == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"PyTorch {torch.__version__} sees {p.name}, {p.total_memory / 2**30:.0f} GB")
else:
    print(f"PyTorch {torch.__version__} runs on {device.WHAT[found.backend]}")"#;

/// setup.sh's self-test with the models, each with `-I` as there: every model file against the registry's sums
/// (its last two lines shown), every stage's code importing, the device seen. Each stops setup when it fails.
fn model_checks(
    place: &Place,
    target: &Target,
    card_needed: bool,
) -> Vec<(&'static str, Command, Option<usize>)> {
    let python = || {
        let mut command = Command::new(place.venv_python(target));
        command.arg("-I").current_dir(&place.repository);
        command
    };
    let setup_script = |name: &str| place.repository.join("setup").join(name);
    let mut verify = python();
    verify
        .arg(setup_script("fetch_models.py"))
        .arg("--models")
        .arg(&place.models)
        .arg("--sources")
        .arg(&place.sources)
        .arg("--verify");
    let mut smoke = python();
    smoke
        .arg(setup_script("smoke.py"))
        .arg(&place.models)
        .arg(&place.sources);
    let mut device = python();
    device
        .args(["-c", DEVICE_SEEN])
        .arg(place.repository.join("python"))
        .arg(if card_needed { "card" } else { "any" });
    vec![
        ("the models' sums", verify, Some(2)),
        ("the stages' code (setup/smoke.py)", smoke, None),
        ("the device", device, None),
    ]
}

fn self_test(
    place: &Place,
    target: &Target,
    system: System,
    options: &Options,
    card_needed: bool,
) -> Result<(), Stopped> {
    let started = Instant::now();
    let log = place.tools.join("self-test.log");
    println!("self-test: the Python tests (made-up objects, no card; their output goes to .tools/self-test.log)");
    let status = unit_tests(
        python_command(place, target, system),
        &place.repository,
        &log,
    )
    .status()
    .map_err(|e| stopped(format!("self-test: {e}")))?;
    if !status.success() {
        let said = std::fs::read_to_string(&log).unwrap_or_default();
        let tail: Vec<&str> = said.lines().rev().take(40).collect();
        for line in tail.into_iter().rev() {
            println!("{line}");
        }
        println!("self-test: the Python tests FAILED (the whole log: .tools/self-test.log)");
    }
    if !options.skip_models {
        for (what, mut command, last_lines) in model_checks(place, target, card_needed) {
            match last_lines {
                None => ran(&mut command, what)?,
                Some(count) => {
                    let done = command
                        .output()
                        .map_err(|e| stopped(format!("{what}: could not start it ({e})")))?;
                    let said = String::from_utf8_lossy(&done.stdout);
                    let lines: Vec<&str> = said.lines().collect();
                    let shown = if done.status.success() { count } else { 40 };
                    for line in &lines[lines.len().saturating_sub(shown)..] {
                        println!("{line}");
                    }
                    eprint!("{}", String::from_utf8_lossy(&done.stderr));
                    if !done.status.success() {
                        return Err(stopped(format!("{what}: it failed ({})", done.status)));
                    }
                }
            }
        }
    }
    println!("self-test: {} s", started.elapsed().as_secs());
    Ok(())
}

/// Needed things missing, apart from the ones setup itself can help with (the C++ runtime on Windows).
fn blocking(rows: &[check::Row]) -> Vec<&check::Row> {
    rows.iter()
        .filter(|r| matches!(r.verdict, Verdict::Missing(_)) && r.check != "C++ runtime")
        .collect()
}

/// The whole setup; its end is written to the progress file too (`done`, or `stopped` with why).
pub fn run(repository: &Path, options: &Options) -> Result<(), Stopped> {
    let result = every_step(repository, options);
    match &result {
        Ok(()) => progress(&progress_line("done", None)),
        Err(stopped) => progress(&progress_line("stopped", Some(("why", &stopped.0)))),
    }
    result
}

fn every_step(repository: &Path, options: &Options) -> Result<(), Stopped> {
    let started = Instant::now();
    let (system, processor) = platform::here();
    let target =
        platform::target(system, processor).ok_or_else(|| stopped("no pins for this system"))?;
    let lock = platform::chosen_lock(target, options.lock.as_deref()).map_err(stopped)?;
    let place = Place::new(repository, options.models.clone(), options.sources.clone());
    println!(
        "model-maker: setting up this PC (everything goes into {})",
        repository.display()
    );
    let mut facts = check::gather(repository, &place.models);
    if options.skip_models {
        // No room is needed for the models' files: count them as there.
        facts.models_gb = facts.models_gb.max(check::MODELS_GB);
    }
    let needs_card = !(options.processor_only || lock.name.ends_with("-cpu"));
    let rows = check::rows(&facts, Some(target), Some(lock), needs_card);
    print!("{}", check::table(&rows));
    let blocked = blocking(&rows);
    if !blocked.is_empty() {
        return Err(stopped(format!(
            "{} thing(s) missing; see MISSING above. Nothing was changed.",
            blocked.len()
        )));
    }
    std::fs::create_dir_all(&place.tools)
        .map_err(|e| stopped(format!("{}: {e}", place.tools.display())))?;
    let record = place.tools.join(DONE_FILE);
    match std::fs::remove_file(&record) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            return Err(stopped(format!("{}: {e}", record.display())))
        }
        _ => {}
    }
    if !facts.cxx_runtime_missing.is_empty() {
        step("runtime");
        match runtime::offered(
            &place.tools,
            std::io::stdin().is_terminal(),
            options.install_runtime,
        ) {
            runtime::Outcome::Installed => println!("Visual C++ runtime: installed"),
            runtime::Outcome::InstalledRestartWanted => {
                println!(
                    "Visual C++ runtime: installed (Windows asks for a restart; setup carries on)"
                )
            }
            runtime::Outcome::Declined => {
                return Err(stopped(format!(
                    "Nothing else was changed. To go on: {}",
                    runtime::INSTRUCTION
                )))
            }
            runtime::Outcome::Failed(why) => {
                return Err(stopped(format!(
                    "{why}. To go on: {}",
                    runtime::INSTRUCTION
                )))
            }
        }
    }
    step("python");
    let uv = installed_uv(&place, target, system)?;
    installed_python(&place, target, system, &uv)?;
    installed_packages(&place, target, lock, &uv, options.wheels.as_deref())?;
    made_certificate(&place, target, system)?;
    step("tools");
    ready_simplify(&place, system)?;
    ready_judge(&place, target, options);
    if !options.skip_models {
        step("models");
        fetched_models(&place, target, options)?;
    }
    if !options.skip_tests {
        step("self-test");
        self_test(&place, target, system, options, needs_card)?;
    }
    std::fs::write(&record, done_line(!options.skip_models))
        .map_err(|e| stopped(format!("{}: {e}", record.display())))?;
    println!(
        "\nSet up in {} s. Make an object:",
        started.elapsed().as_secs()
    );
    println!("  model-maker make \"an old looking chair\"");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check::Row;

    #[test]
    fn a_missing_cxx_runtime_does_not_block_setup_but_other_missing_things_do() {
        let row = |check: &'static str, missing: bool| Row {
            check,
            found: String::new(),
            verdict: if missing {
                Verdict::Missing("x".into())
            } else {
                Verdict::Ok("ok".into())
            },
        };
        assert!(blocking(&[row("C++ runtime", true), row("git", false)]).is_empty());
        assert_eq!(
            blocking(&[row("C++ runtime", true), row("Memory", true)]).len(),
            1
        );
    }

    #[test]
    fn packages_come_from_the_index_unless_a_wheel_folder_or_address_is_given() {
        let words = |wheels| {
            sync_arguments(
                Path::new("/r/.venv/bin/python"),
                Path::new("/r/lock"),
                wheels,
            )
            .into_iter()
            .map(|w| w.into_string().unwrap())
            .collect::<Vec<_>>()
            .join(" ")
        };
        let exact =
            "pip sync --quiet --python /r/.venv/bin/python --require-hashes --only-binary :all:";
        assert_eq!(words(None), format!("{exact} /r/lock"));
        assert_eq!(
            words(Some("http://127.0.0.1:8765/wheels/")),
            format!("{exact} --no-index --find-links http://127.0.0.1:8765/wheels/ /r/lock")
        );
    }

    #[test]
    fn an_address_pin_becomes_its_wheels_version_with_its_sums() {
        let lock = "# a comment\n\
            numpy==2.4.6 \\\n    --hash=sha256:aa\n\
            torch @ https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp311-cp311-win_amd64.whl#sha256=bb \\\n\
            \x20   --hash=sha256:bb\n\
            odd @ https://example.org/odd.tar.gz\n";
        let pinned = pinned_by_version(lock);
        assert_eq!(
            pinned,
            "# a comment\nnumpy==2.4.6 \\\n    --hash=sha256:aa\ntorch==2.10.0+cu128 \\\n    --hash=sha256:bb\n\
             odd @ https://example.org/odd.tar.gz\n"
        );
    }

    #[test]
    fn a_wheel_folder_must_be_there_and_an_address_is_taken_as_it_is() {
        assert!(wheels_source(None).unwrap().is_none());
        assert_eq!(
            wheels_source(Some("https://example.org/wheels/")).unwrap(),
            Some("https://example.org/wheels/".to_string())
        );
        let refused = wheels_source(Some("/no/such/wheel/folder")).unwrap_err();
        assert!(refused.0.contains("no such folder"), "{refused}");
        let here = std::env::temp_dir();
        let taken = wheels_source(Some(here.to_str().unwrap()))
            .unwrap()
            .unwrap();
        assert!(Path::new(&taken).is_absolute() && Path::new(&taken).is_dir());
    }

    #[test]
    fn progress_lines_are_one_json_object_each() {
        assert_eq!(
            progress_line("began", Some(("step", "python"))),
            "{\"schema\":1,\"event\":\"began\",\"step\":\"python\"}\n"
        );
        assert_eq!(
            progress_line("done", None),
            "{\"schema\":1,\"event\":\"done\"}\n"
        );
        let stopped = progress_line("stopped", Some(("why", "a \"quoted\" path\\x\nand\ttab")));
        assert_eq!(
            stopped,
            "{\"schema\":1,\"event\":\"stopped\",\"why\":\"a \\\"quoted\\\" path\\\\x\\nand\\u0009tab\"}\n"
        );
        assert_eq!(stopped.matches('\n').count(), 1);
    }

    #[test]
    fn the_done_record_says_only_what_setup_wrote() {
        let tools = std::env::temp_dir().join(format!("mm-launcher-done-{}", std::process::id()));
        std::fs::create_dir_all(&tools).unwrap();
        assert_eq!(done_record(&tools), Done::No);
        std::fs::write(tools.join(DONE_FILE), done_line(true)).unwrap();
        assert_eq!(done_record(&tools), Done::WithModels);
        std::fs::write(tools.join(DONE_FILE), done_line(false)).unwrap();
        assert_eq!(done_record(&tools), Done::WithoutModels);
        std::fs::write(tools.join(DONE_FILE), "{\"schema\":2,\"models\":true}\n").unwrap();
        assert_eq!(done_record(&tools), Done::No, "another schema is not read");
        std::fs::remove_dir_all(&tools).unwrap();
    }

    #[test]
    fn every_step_has_an_id_and_a_title() {
        let ids: Vec<&str> = STEPS.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, ["runtime", "python", "tools", "models", "self-test"]);
    }

    #[test]
    fn folders_default_inside_the_repository_and_options_win() {
        let repository = Path::new("/r");
        let place = Place::new(repository, Some(PathBuf::from("/m")), None);
        assert_eq!(place.models, PathBuf::from("/m"));
        assert_eq!(place.tools, PathBuf::from("/r/.tools"));
        let windows = platform::target(System::Windows, platform::Processor::X86_64).unwrap();
        assert!(place.venv_python(windows).ends_with("Scripts/python.exe"));
    }

    /// Every Python step of setup.sh's `self_test` is one of the launcher's too, with the same flags and script.
    #[test]
    fn the_self_test_has_every_step_of_setup_sh() {
        let setup =
            std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../setup.sh"))
                .unwrap();
        let body = setup
            .split("\nself_test() {")
            .nth(1)
            .unwrap()
            .split("\n}")
            .next()
            .unwrap();
        let place = Place::new(Path::new("/r"), None, None);
        let linux = platform::target(System::Linux, platform::Processor::X86_64).unwrap();
        let mut ours: Vec<Vec<String>> = model_checks(&place, linux, true)
            .iter()
            .map(|(_, c, _)| {
                c.get_args()
                    .map(|a| a.to_string_lossy().replace('\\', "/"))
                    .collect()
            })
            .collect();
        let tests = unit_tests(
            Command::new("python"),
            &place.repository,
            &place.tools.join("self-test.log"),
        );
        ours.push(
            tests
                .get_args()
                .map(|a| a.to_string_lossy().replace('\\', "/"))
                .collect(),
        );
        assert!(
            setup.contains(&format!(
                "\nreadonly SELF_TEST_WORKERS={SELF_TEST_WORKERS} "
            )),
            "setup.sh's SELF_TEST_WORKERS is not the launcher's"
        );
        let steps: Vec<&str> = body
            .lines()
            .filter_map(|l| l.split("\"$venv/bin/python\" ").nth(1))
            .collect();
        assert_eq!(steps.len(), 4, "setup.sh's self-test steps were not found");
        for step in steps {
            let words: Vec<&str> = step
                .split(['\'', '>', '|'])
                .next()
                .unwrap()
                .split_whitespace()
                .collect();
            let wanted: Vec<&str> = words.into_iter().filter(|w| !w.contains('$')).collect();
            let found = ours.iter().any(|args| {
                wanted
                    .iter()
                    .all(|w| args.iter().any(|a| a == w || a.ends_with(&format!("/{w}"))))
            });
            assert!(
                found,
                "setup.sh's self-test step `{step}` is not in the launcher's"
            );
        }
    }
}
